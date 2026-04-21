from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from lxml import etree

from pwml_validate import (
    SectionSignature,
    StructureSignature,
    discover_structure_signature,
    repair_tree,
    validate_generated_tree,
    write_json_report,
)


def _singularize(tag: str) -> str:
    overrides = {
        "species": "species",
        "cell-types": "cell-type",
        "subcellular-locations": "subcellular-location",
        "biological-states": "biological-state",
        "element-collections": "element-collection",
        "nucleic-acids": "nucleic-acid",
        "protein-complexes": "protein-complex",
        "reaction-coupled-transports": "reaction-coupled-transport",
        "bound-visualizations": "bound-visualization",
        "compound-locations": "compound-location",
        "element-collection-locations": "element-collection-location",
        "nucleic-acid-locations": "nucleic-acid-location",
        "protein-locations": "protein-location",
        "protein-complex-visualizations": "protein-complex-visualization",
        "reaction-visualizations": "reaction-visualization",
        "reaction-coupled-transport_visualizations": "reaction-coupled-transport-visualization",
        "transport-visualizations": "transport-visualization",
        "interaction-visualizations": "interaction-visualization",
        "sub-pathway-visualizations": "sub-pathway-visualization",
        "vacuous-compound-visualizations": "vacuous-compound-visualization",
        "vacuous-edge-visualizations": "vacuous-edge-visualization",
        "vacuous-nucleic-acid-visualizations": "vacuous-nucleic-acid-visualization",
        "vacuous-element-collection-visualizations": "vacuous-element-collection-visualization",
        "vacuous-protein-visualizations": "vacuous-protein-visualization",
        "drawable-element-locations": "drawable-element-location",
        "membrane-visualizations": "membrane-visualization",
        "label-locations": "label-location",
        "zoom-visualizations": "zoom-visualization",
        "reaction-left-elements": "reaction-left-element",
        "reaction-right-elements": "reaction-right-element",
        "reaction-enzymes": "reaction-enzyme",
        "transport-elements": "transport-element",
        "transport-transporters": "transport-transporter",
        "reaction_compound_visualizations": "reaction-compound-visualization",
        "reaction_element_collection_visualizations": "reaction-element-collection-visualization",
        "reaction_enzyme_visualizations": "reaction-enzyme-visualization",
        "transport_compound_visualizations": "transport-compound-visualization",
        "transport_transporter_visualizations": "transport-transporter-visualization",
        "protein_complex_protein_visualizations": "protein-complex-protein-visualization",
        "protein_complex_compound_visualizations": "protein-complex-compound-visualization",
        "sub_pathway_element_collection_visualizations": "sub-pathway-element-collection-visualization",
        "element-states": "element-state",
        "ec-numbers": "ec-number",
        "synonyms": "synonym",
        "sub-pathways": "sub-pathway",
        "references": "reference",
    }
    if tag in overrides:
        return overrides[tag]
    if tag.endswith("ies") and len(tag) > 3:
        return f"{tag[:-3]}y"
    if tag.endswith("s") and len(tag) > 1:
        return tag[:-1]
    return tag


def _normalize_key(value: str) -> str:
    return value.strip().casefold()


def _is_integer_field(tag: str) -> bool:
    return (
        tag == "id"
        or tag.endswith("-id")
        or tag in {"x", "y", "zindex", "stoichiometry", "p1x", "p1y", "p2x", "p2y", "p3x", "p3y", "degree"}
    )


def _is_boolean_field(tag: str) -> bool:
    return tag in {
        "hidden",
        "spontaneous",
        "currency",
        "complete-membrane",
        "option:end_arrow",
        "option:end_flat_arrow",
        "option:start_arrow",
        "option:start_flat_arrow",
    }


def _as_named_records(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        norm = _normalize_key(name)
        if norm in seen:
            continue
        seen.add(norm)
        record = dict(item)
        record["name"] = name
        out.append(record)
    out.sort(key=lambda rec: (rec["name"].casefold(), rec["name"]))
    return out


def _as_process_list(processes: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    raw = processes.get(key, [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _as_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        s = value.strip()
        if s:
            out.append(s)
    return out


def _grid_positions(
    n: int, start_x: int, start_y: int, dx: int, dy: int, max_cols: int
) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for idx in range(n):
        row = idx // max_cols
        col = idx % max_cols
        coords.append((start_x + col * dx, start_y + row * dy))
    return coords


_CANONICAL_MATCH_RULES: List[Tuple[str, str]] = [
    ("extracellular", "extracellular"),
    ("plasma membrane", "plasma membrane"),
    ("cell membrane", "plasma membrane"),
    ("sarcoplasmic reticulum", "endoplasmic reticulum"),
    ("endoplasmic reticulum", "endoplasmic reticulum"),
    ("cytoplasm", "cytosol"),
    ("cytosol", "cytosol"),
    ("nucleus", "nucleus"),
    ("mitochondrial matrix", "mitochondria"),
    ("mitochondria", "mitochondria"),
    ("lysosome", "lysosome"),
    ("peroxisome", "peroxisome"),
    ("golgi", "golgi"),
]

_CANONICAL_TYPE_ORDER: Dict[str, int] = {
    "extracellular": 0,
    "plasma membrane": 1,
    "cytosol": 2,
    "endoplasmic reticulum": 3,
    "nucleus": 4,
    "mitochondria": 5,
    "lysosome": 6,
    "peroxisome": 7,
    "golgi": 8,
    "unrecognized": 99,
}


def _match_canonical_type(compartment_canonical: str) -> str:
    c = compartment_canonical.strip().casefold()
    for pattern, ctype in _CANONICAL_MATCH_RULES:
        if pattern in c:
            return ctype
    return "unrecognized"


@dataclass
class IdFactory:
    value: int = 1

    def next(self) -> int:
        current = self.value
        self.value += 1
        return current


@dataclass
class BuildResult:
    root: etree._Element
    counts: Dict[str, int]
    geometry_generated: bool
    signature: StructureSignature


class DeterministicPwmlBuilder:
    def __init__(self, extraction: Dict[str, Any], signature: StructureSignature, args: argparse.Namespace) -> None:
        self.extraction = extraction
        self.signature = signature
        self.args = args
        self.aux_ids = IdFactory(1)
        self.ids = self.aux_ids  # alias for visualization/edge sub-objects
        self.state_ids = IdFactory(10000)
        self.compound_ids = IdFactory(20000)
        self.protein_ids = IdFactory(30000)
        self.complex_ids = IdFactory(40000)
        self.reaction_ids = IdFactory(50000)
        self.location_ids = IdFactory(100000)

        entities = extraction.get("entities", {}) if isinstance(extraction, dict) else {}
        self.entities = entities if isinstance(entities, dict) else {}
        self.processes = extraction.get("processes", {}) if isinstance(extraction, dict) else {}
        if not isinstance(self.processes, dict):
            self.processes = {}

        self.entity_records: Dict[str, List[Dict[str, Any]]] = {}
        self.entity_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.element_lookup: Dict[str, Tuple[str, int]] = {}

        self.section_items: Dict[str, List[Dict[str, Any]]] = {}

        self.pathway_id_int = 1
        self.pathway_visualization_id_int = self.pathway_id_int
        self.pathway_visualization_id = f"PathwayVisualization{self.pathway_visualization_id_int}"
        self.pathway_visualization_context_id = f"PathwayVisualizationContext{self.pathway_visualization_id_int}"
        self._state_id_map: Dict[str, int] = {}

    def _prepare_entities(self) -> None:
        key_to_section = {
            "cell_types": "cell-types",
            "species": "species",
            "subcellular_locations": "subcellular-locations",
            "tissues": "tissues",
            "compounds": "compounds",
            "element_collections": "element-collections",
            "nucleic_acids": "nucleic-acids",
            "proteins": "proteins",
            "protein_complexes": "protein-complexes",
        }

        for key, section in key_to_section.items():
            records = _as_named_records(self.entities.get(key, []))
            for record in records:
                if key == "compounds":
                    pw_id = None
                    for k in ["pathbank_compound_id", "pw_compound_id"]:
                        v = record.get(k) or (record.get("mapping_meta") or {}).get(k)
                        if v:
                            try: pw_id = int(v); break
                            except (ValueError, TypeError): pass
                    record["id"] = pw_id if pw_id is not None else self.compound_ids.next()
                elif key == "proteins":
                    pw_id = None
                    for k in ["pathbank_protein_id", "pw_protein_id"]:
                        v = record.get(k) or (record.get("mapping_meta") or {}).get(k)
                        if v:
                            try: pw_id = int(v); break
                            except (ValueError, TypeError): pass
                    record["id"] = pw_id if pw_id is not None else self.protein_ids.next()
                elif key == "protein_complexes":
                    pw_id = None
                    for k in ["pathbank_complex_id", "pw_complex_id"]:
                        v = record.get(k) or (record.get("mapping_meta") or {}).get(k)
                        if v:
                            try: pw_id = int(v); break
                            except (ValueError, TypeError): pass
                    record["id"] = pw_id if pw_id is not None else self.complex_ids.next()
                else:
                    record["id"] = self.ids.next()
            self.entity_records[section] = records
            self.entity_lookup[key] = {_normalize_key(rec["name"]): rec for rec in records}

        for section, element_type in [
            ("compounds", "Compound"),
            ("element-collections", "ElementCollection"),
            ("nucleic-acids", "NucleicAcid"),
            ("proteins", "Protein"),
            ("protein-complexes", "ProteinComplex"),
        ]:
            for rec in self.entity_records.get(section, []):
                self.element_lookup[_normalize_key(rec["name"])] = (element_type, int(rec["id"]))

    def _resolve_ref_id(self, value: Any, lookup_key: str, fallback: bool = False) -> Optional[int]:
        lookup = self.entity_lookup.get(lookup_key, {})
        if isinstance(value, str):
            found = lookup.get(_normalize_key(value))
            if found:
                return int(found["id"])
        if fallback and lookup:
            return int(next(iter(lookup.values()))["id"])
        return None

    def _build_biological_states(self) -> Tuple[List[Dict[str, Any]], int]:
        raw_states = self.extraction.get("biological_states", [])
        records = _as_named_records(raw_states)
        if not records:
            records = [{"name": "Default state"}]

        states: List[Dict[str, Any]] = []
        for record in records:
            sid = self.ids.next()
            state = {
                "id": sid,
                "name": record["name"],
                "tissue-id": self._resolve_ref_id(record.get("tissue"), "tissues"),
                "subcellular-location-id": self._resolve_ref_id(
                    record.get("subcellular_location"), "subcellular_locations", fallback=True
                ),
                "species-id": self._resolve_ref_id(record.get("species"), "species", fallback=True),
                "cell-type-id": self._resolve_ref_id(record.get("cell_type"), "cell_types", fallback=True),
                "pwbs-id": f"PW_BS{sid:06d}",
            }
            states.append(state)

        self._state_id_map: Dict[str, int] = {
            _normalize_key(s["name"]): int(s["id"]) for s in states
        }
        return states, int(states[0]["id"])

    def _build_reactions(self) -> List[Dict[str, Any]]:
        reactions_raw = _as_process_list(self.processes, "reactions")
        out: List[Dict[str, Any]] = []
        for raw in reactions_raw:
            rid = self.reaction_ids.next()
            left: List[Dict[str, Any]] = []
            right: List[Dict[str, Any]] = []

            for name in _as_string_list(raw.get("inputs")):
                resolved = self.element_lookup.get(_normalize_key(name))
                if not resolved:
                    continue
                etype, eid = resolved
                left.append(
                    {
                        "id": self.ids.next(),
                        "element-id": eid,
                        "stoichiometry": 1,
                        "element-type": etype,
                        "currency": False,
                    }
                )
            for name in _as_string_list(raw.get("outputs")):
                resolved = self.element_lookup.get(_normalize_key(name))
                if not resolved:
                    continue
                etype, eid = resolved
                right.append(
                    {
                        "id": self.ids.next(),
                        "element-id": eid,
                        "stoichiometry": 1,
                        "element-type": etype,
                        "currency": False,
                    }
                )

            enzymes: List[Dict[str, Any]] = []
            modifiers = raw.get("modifiers") if isinstance(raw.get("modifiers"), list) else None
            if modifiers is not None:
                for mod in modifiers:
                    if not isinstance(mod, dict):
                        continue
                    role = str(mod.get("role") or "").strip().lower()
                    entity_name = str(mod.get("entity") or "").strip()
                    entity_type = str(mod.get("entity_type") or "").strip().lower()
                    if not entity_name or role not in {"catalyst", "activator", "inhibitor"}:
                        continue
                    entry: Dict[str, Any] = {"id": self.ids.next()}
                    if entity_type == "protein_complex":
                        pc = self.entity_lookup.get("protein_complexes", {}).get(_normalize_key(entity_name))
                        if not pc:
                            continue
                        entry["protein-complex-id"] = int(pc["id"])
                    else:
                        prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(entity_name))
                        if not prot:
                            continue
                        entry["protein-id"] = int(prot["id"])
                    if role == "inhibitor":
                        entry["inhibitor"] = True
                    enzymes.append(entry)
            else:
                for enzyme in raw.get("enzymes", []) if isinstance(raw.get("enzymes"), list) else []:
                    if not isinstance(enzyme, dict):
                        continue
                    pc_name = (
                        str(enzyme.get("protein_complex") or enzyme.get("protein-complex") or "").strip()
                    )
                    prot_name = str(enzyme.get("protein") or "").strip()
                    if pc_name:
                        pc = self.entity_lookup.get("protein_complexes", {}).get(_normalize_key(pc_name))
                        if pc:
                            enzymes.append(
                                {
                                    "id": self.ids.next(),
                                    "protein-complex-id": int(pc["id"]),
                                    "enzyme-class": str(enzyme.get("enzyme_class") or "").strip(),
                                }
                            )
                            continue
                        # protein_complex key may hold a plain protein name (from _clean_enzymes)
                        prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(pc_name))
                        if prot:
                            enzymes.append({"id": self.ids.next(), "protein-id": int(prot["id"])})
                    elif prot_name:
                        prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(prot_name))
                        if prot:
                            enzymes.append({"id": self.ids.next(), "protein-id": int(prot["id"])})

            out.append(
                {
                    "id": rid,
                    "spontaneous": None,
                    "pwr-id": f"PW_R{rid:06d}",
                    "direction": "Right",
                    "reaction-left-elements": left,
                    "reaction-right-elements": right,
                    "reaction-enzymes": enzymes,
                }
            )
        return out

    def _build_transports(self, default_state_id: int) -> List[Dict[str, Any]]:
        transports_raw = _as_process_list(self.processes, "transports")
        out: List[Dict[str, Any]] = []
        for raw in transports_raw:
            tid = self.ids.next()
            cargo = str(raw.get("cargo", "")).strip()

            from_bs = str(raw.get("from_biological_state", "")).strip()
            to_bs = str(raw.get("to_biological_state", "")).strip()
            left_bs_id = self._state_id_map.get(from_bs.casefold(), default_state_id)
            right_bs_id = self._state_id_map.get(to_bs.casefold(), default_state_id)

            elements: List[Dict[str, Any]] = []
            if cargo:
                resolved = self.element_lookup.get(_normalize_key(cargo))
                if resolved:
                    etype, eid = resolved
                    elements.append(
                        {
                            "id": self.ids.next(),
                            "element-id": eid,
                            "stoichiometry": 1,
                            "element-type": etype,
                            "left-biological-state-id": left_bs_id,
                            "right-biological-state-id": right_bs_id,
                            "direction": "Right",
                        }
                    )

            transporters: List[Dict[str, Any]] = []
            for t in raw.get("transporters", []) if isinstance(raw.get("transporters"), list) else []:
                if not isinstance(t, dict):
                    continue
                entity_name = str(t.get("entity", "")).strip()
                if not entity_name:
                    continue
                prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(entity_name))
                if prot:
                    transporters.append({"id": self.ids.next(), "protein-id": int(prot["id"])})
                    continue
                pc = self.entity_lookup.get("protein_complexes", {}).get(_normalize_key(entity_name))
                if pc:
                    transporters.append({"id": self.ids.next(), "protein-complex-id": int(pc["id"])})

            out.append(
                {
                    "id": tid,
                    "pwt-id": f"PW_T{tid:06d}",
                    "transport-type": None,
                    "transport-elements": elements,
                    "transport-transporters": transporters,
                }
            )
        return out

    def _build_interactions(self) -> List[Dict[str, Any]]:
        interactions_raw = _as_process_list(self.processes, "interactions")
        out: List[Dict[str, Any]] = []
        for _ in interactions_raw:
            iid = self.ids.next()
            out.append({"id": iid})
        return out

    def _build_reaction_coupled_transports(self) -> List[Dict[str, Any]]:
        rcts_raw = _as_process_list(self.processes, "reaction_coupled_transports")
        out: List[Dict[str, Any]] = []
        for _ in rcts_raw:
            rid = self.ids.next()
            out.append({"id": rid})
        return out

    def _assign_compartment_regions(
        self,
        raw_states: List[Dict[str, Any]],
        canvas_w: int,
        canvas_h: int,
    ) -> Dict[str, Dict[str, Any]]:
        extra_h = int(canvas_h * 0.18)
        pm_h = int(canvas_h * 0.05)
        cyto_y = extra_h + pm_h
        cyto_h = int(canvas_h * 0.40)
        nuc_h = int(cyto_h * 0.35)
        mito_h = int(cyto_h * 0.30)
        lyso_y = cyto_y + nuc_h + mito_h
        lyso_h = max(cyto_h - nuc_h - mito_h, 50)
        bottom_y = cyto_y + cyto_h
        bottom_h = max(canvas_h - bottom_y, 100)

        type_to_region: Dict[str, Dict[str, Any]] = {
            "extracellular":         {"x": 0,                 "y": 0,              "w": canvas_w,       "h": extra_h,  "label": "Extracellular"},
            "plasma membrane":       {"x": 0,                 "y": extra_h,        "w": canvas_w,       "h": pm_h,     "label": "Plasma Membrane"},
            "cytosol":               {"x": 0,                 "y": cyto_y,         "w": canvas_w,       "h": cyto_h,   "label": "Cytosol"},
            "endoplasmic reticulum": {"x": canvas_w // 2,     "y": cyto_y,         "w": canvas_w // 2,  "h": nuc_h,    "label": "Endoplasmic Reticulum"},
            "nucleus":               {"x": 0,                 "y": cyto_y,         "w": canvas_w // 2,  "h": nuc_h,    "label": "Nucleus"},
            "mitochondria":          {"x": 0,                 "y": cyto_y + nuc_h, "w": canvas_w,       "h": mito_h,   "label": "Mitochondria"},
            "lysosome":              {"x": 0,                 "y": lyso_y,         "w": canvas_w // 3,  "h": lyso_h,   "label": "Lysosome"},
            "peroxisome":            {"x": canvas_w // 3,     "y": lyso_y,         "w": canvas_w // 3,  "h": lyso_h,   "label": "Peroxisome"},
            "golgi":                 {"x": 2 * canvas_w // 3, "y": lyso_y,         "w": canvas_w // 3,  "h": lyso_h,   "label": "Golgi"},
            "unrecognized":          {"x": 0,                 "y": bottom_y,       "w": canvas_w,       "h": bottom_h, "label": "Other"},
        }

        state_ctype: Dict[str, str] = {}
        for s in raw_states:
            name_norm = _normalize_key(str(s.get("name", "")))
            if not name_norm:
                continue
            comp = str(s.get("compartment_canonical", ""))
            state_ctype[name_norm] = _match_canonical_type(comp)

        unique_ctypes = set(state_ctype.values())
        n = len(unique_ctypes)

        if n <= 2:
            sorted_ctypes = sorted(unique_ctypes, key=lambda t: _CANONICAL_TYPE_ORDER.get(t, 99))
            band_h = canvas_h // max(n, 1)
            for i, ctype in enumerate(sorted_ctypes):
                type_to_region[ctype] = {
                    "x": 0,
                    "y": i * band_h,
                    "w": canvas_w,
                    "h": band_h,
                    "label": ctype.title(),
                }

        result: Dict[str, Dict[str, Any]] = {}
        for name_norm, ctype in state_ctype.items():
            result[name_norm] = dict(type_to_region[ctype])
        return result

    def _build_locations_and_visualizations(
        self,
        default_state_id: int,
        reactions: List[Dict[str, Any]],
        transports: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        canvas_w: int = self.args.width
        canvas_h: int = self.args.height
        pad = 30
        dx_left, dy_left = 200, 100
        dx_right, dy_right = 220, 110

        raw_bio_states = _as_named_records(self.extraction.get("biological_states", []))
        compartment_regions = self._assign_compartment_regions(raw_bio_states, canvas_w, canvas_h)

        bs_id_to_region_key: Dict[int, str] = {v: k for k, v in self._state_id_map.items()}
        fallback_region: Dict[str, Any] = {"x": 0, "y": 0, "w": canvas_w, "h": canvas_h, "label": "Default"}

        def region_for(bs_id: int) -> Dict[str, Any]:
            key = bs_id_to_region_key.get(bs_id, "")
            return compartment_regions.get(key, fallback_region)

        def sub_grid_left(region: Dict[str, Any], n: int) -> List[Tuple[int, int]]:
            x0 = region["x"] + pad
            y0 = region["y"] + pad
            w = max(region["w"] // 2 - 2 * pad, 100)
            cols = max(1, w // dx_left)
            return _grid_positions(n, x0, y0, dx_left, dy_left, cols)

        def sub_grid_right(region: Dict[str, Any], n: int) -> List[Tuple[int, int]]:
            x0 = region["x"] + region["w"] // 2 + pad
            y0 = region["y"] + pad
            w = max(region["w"] // 2 - 2 * pad, 100)
            cols = max(1, w // dx_right)
            return _grid_positions(n, x0, y0, dx_right, dy_right, cols)

        compound_locations: List[Dict[str, Any]] = []
        element_collection_locations: List[Dict[str, Any]] = []
        nucleic_acid_locations: List[Dict[str, Any]] = []
        protein_locations: List[Dict[str, Any]] = []
        protein_complex_visualizations: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        reaction_visualizations: List[Dict[str, Any]] = []
        transport_visualizations: List[Dict[str, Any]] = []
        bound_visualizations: List[Dict[str, Any]] = []
        membrane_visualizations: List[Dict[str, Any]] = []

        compound_loc_by_id: Dict[int, Dict[str, Any]] = {}
        element_collection_loc_by_id: Dict[int, Dict[str, Any]] = {}
        nucleic_acid_loc_by_id: Dict[int, Dict[str, Any]] = {}
        protein_loc_by_id: Dict[int, Dict[str, Any]] = {}
        pc_vis_by_pc_id: Dict[int, Dict[str, Any]] = {}

        # Bound-visualizations — one per biological state
        for name_norm, bs_id in self._state_id_map.items():
            region = compartment_regions.get(name_norm, fallback_region)
            bound_visualizations.append({
                "id": self.ids.next(),
                "biological-state-id": bs_id,
                "x": region["x"],
                "y": region["y"],
                "width": str(region["w"]),
                "height": str(region["h"]),
                "zindex": 1,
                "hidden": False,
            })

        # Helper: group entity records by their biological state id
        def group_by_bs(section_key: str) -> Dict[int, List[Dict[str, Any]]]:
            groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for rec in self.entity_records.get(section_key, []):
                entity_state_name = rec.get("biological_state", "")
                bsid = self._state_id_map.get(entity_state_name.strip().casefold(), default_state_id)
                groups[bsid].append(rec)
            return groups

        # Compound locations — left half of each compartment region
        for bs_id, group_recs in sorted(group_by_bs("compounds").items()):
            region = region_for(bs_id)
            for rec, (x, y) in zip(group_recs, sub_grid_left(region, len(group_recs))):
                loc = {
                    "id": self.ids.next(),
                    "compound-id": int(rec["id"]),
                    "biological-state-id": bs_id,
                    "visualization-template-id": 0,
                    "hidden": False,
                    "x": x,
                    "y": y,
                    "zindex": 10,
                    "font-size": "regular",
                    "width": "160",
                    "height": "60",
                }
                compound_locations.append(loc)
                compound_loc_by_id[int(rec["id"])] = loc

        # Element-collection locations — left half of each compartment region
        for bs_id, group_recs in sorted(group_by_bs("element-collections").items()):
            region = region_for(bs_id)
            for rec, (x, y) in zip(group_recs, sub_grid_left(region, len(group_recs))):
                loc = {
                    "id": self.ids.next(),
                    "element-collection-id": int(rec["id"]),
                    "visualization-template-id": 0,
                    "biological-state-id": bs_id,
                    "hidden": False,
                    "x": x,
                    "y": y,
                    "zindex": 10,
                    "font-size": "regular",
                    "width": "180",
                    "height": "70",
                }
                element_collection_locations.append(loc)
                element_collection_loc_by_id[int(rec["id"])] = loc

        # Nucleic-acid locations — left half of each compartment region
        for bs_id, group_recs in sorted(group_by_bs("nucleic-acids").items()):
            region = region_for(bs_id)
            for rec, (x, y) in zip(group_recs, sub_grid_left(region, len(group_recs))):
                loc = {
                    "id": self.ids.next(),
                    "nucleic-acid-id": int(rec["id"]),
                    "biological-state-id": bs_id,
                    "visualization-template-id": 0,
                    "hidden": False,
                    "x": x,
                    "y": y,
                    "zindex": 10,
                    "font-size": "regular",
                    "width": "190",
                    "height": "60",
                }
                nucleic_acid_locations.append(loc)
                nucleic_acid_loc_by_id[int(rec["id"])] = loc

        # Protein locations — right half of each compartment region
        for bs_id, group_recs in sorted(group_by_bs("proteins").items()):
            region = region_for(bs_id)
            for rec, (x, y) in zip(group_recs, sub_grid_right(region, len(group_recs))):
                loc = {
                    "id": self.ids.next(),
                    "protein-id": int(rec["id"]),
                    "biological-state-id": bs_id,
                    "visualization-template-id": 0,
                    "hidden": False,
                    "x": x,
                    "y": y,
                    "zindex": 10,
                    "label-type": "text",
                    "font-size": "regular",
                    "width": "200",
                    "height": "60",
                }
                protein_locations.append(loc)
                protein_loc_by_id[int(rec["id"])] = loc

        for rec in self.entity_records.get("protein-complexes", []):
            pc_protein_vis: List[Dict[str, Any]] = []
            for comp_name in _as_string_list(rec.get("components", [])):
                prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(comp_name))
                if prot:
                    prot_loc = protein_loc_by_id.get(int(prot["id"]))
                    if prot_loc:
                        pc_protein_vis.append({
                            "id": self.ids.next(),
                            "protein-location-id": int(prot_loc["id"]),
                        })
            visualization = {
                "id": self.ids.next(),
                "protein-complex-id": int(rec["id"]),
                "pathway-visualization-id": self.pathway_visualization_id_int,
                "biological-state-id": default_state_id,
                "protein_complex_protein_visualizations": pc_protein_vis,
            }
            protein_complex_visualizations.append(visualization)
            pc_vis_by_pc_id[int(rec["id"])] = visualization

        def location_info(element_type: str, element_id: int) -> Optional[Tuple[int, int, int]]:
            if element_type == "Compound":
                loc = compound_loc_by_id.get(element_id)
                if loc:
                    return int(loc["id"]), int(loc["x"]) + 80, int(loc["y"]) + 30
            elif element_type == "ElementCollection":
                loc = element_collection_loc_by_id.get(element_id)
                if loc:
                    return int(loc["id"]), int(loc["x"]) + 90, int(loc["y"]) + 35
            elif element_type == "NucleicAcid":
                loc = nucleic_acid_loc_by_id.get(element_id)
                if loc:
                    return int(loc["id"]), int(loc["x"]) + 95, int(loc["y"]) + 30
            elif element_type == "Protein":
                loc = protein_loc_by_id.get(element_id)
                if loc:
                    return int(loc["id"]), int(loc["x"]) + 100, int(loc["y"]) + 30
            return None

        # Reaction visualizations — positioned at compartment region centroid
        raw_reactions = _as_process_list(self.processes, "reactions")
        for reaction, raw_rx in zip(reactions, raw_reactions):
            bs_name = str(raw_rx.get("biological_state", "")).strip()
            rx_bs_id = self._state_id_map.get(bs_name.casefold(), default_state_id)
            rx_region = region_for(rx_bs_id)
            rx = rx_region["x"] + rx_region["w"] // 2
            ry = rx_region["y"] + rx_region["h"] // 2

            reaction_compound_visualizations: List[Dict[str, Any]] = []
            reaction_element_collection_visualizations: List[Dict[str, Any]] = []
            reaction_enzyme_visualizations: List[Dict[str, Any]] = []

            for side_key, side in [("reaction-left-elements", "Left"), ("reaction-right-elements", "Right")]:
                for rel in reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []:
                    etype = str(rel.get("element-type") or "")
                    eid = int(rel.get("element-id") or 0)
                    loc = location_info(etype, eid)
                    if not loc:
                        continue
                    location_id, lx, ly = loc
                    edge_id = self.ids.next()
                    if side == "Left":
                        path = f"M{lx} {ly} L{rx} {ry}"
                    else:
                        path = f"M{rx} {ry} L{lx} {ly}"
                    edges.append({
                        "id": edge_id,
                        "path": path,
                        "visualization-template-id": 0,
                        "hidden": False,
                        "zindex": 18,
                    })
                    if etype == "Compound":
                        reaction_compound_visualizations.append({
                            "id": self.ids.next(),
                            "compound-location-id": location_id,
                            "edge-id": edge_id,
                            "side": side,
                        })
                    elif etype == "ElementCollection":
                        reaction_element_collection_visualizations.append({
                            "id": self.ids.next(),
                            "element-collection-location-id": location_id,
                            "edge-id": edge_id,
                            "side": side,
                        })

            for enzyme in reaction.get("reaction-enzymes", []) if isinstance(reaction.get("reaction-enzymes"), list) else []:
                pc_id = enzyme.get("protein-complex-id")
                prot_id = enzyme.get("protein-id")
                if pc_id is not None:
                    pc_vis = pc_vis_by_pc_id.get(int(pc_id))
                    if not pc_vis:
                        continue
                    reaction_enzyme_visualizations.append({
                        "id": self.ids.next(),
                        "reaction-enzyme-id": int(enzyme["id"]),
                        "protein-complex-visualization-id": int(pc_vis["id"]),
                    })
                elif prot_id is not None:
                    prot_loc = protein_loc_by_id.get(int(prot_id))
                    if not prot_loc:
                        continue
                    reaction_enzyme_visualizations.append({
                        "id": self.ids.next(),
                        "reaction-enzyme-id": int(enzyme["id"]),
                        "protein-location-id": int(prot_loc["id"]),
                    })

            reaction_visualizations.append({
                "id": self.ids.next(),
                "pathway-visualization-id": self.pathway_visualization_id_int,
                "reaction-id": int(reaction["id"]),
                "biological-state-id": rx_bs_id,
                "reaction_compound_visualizations": reaction_compound_visualizations,
                "reaction_element_collection_visualizations": reaction_element_collection_visualizations,
                "reaction_enzyme_visualizations": reaction_enzyme_visualizations,
            })

        for transport in transports:
            transport_visualizations.append({
                "id": self.ids.next(),
                "transport-id": int(transport["id"]),
                "pathway-visualization-id": self.pathway_visualization_id_int,
                "transport_compound_visualizations": [],
                "transport_transporter_visualizations": [],
            })

        # Membrane-visualizations at compartment boundaries
        present_ctypes: Set[str] = {
            _match_canonical_type(str(s.get("compartment_canonical", "")))
            for s in raw_bio_states
        }

        extra_h = int(canvas_h * 0.18)
        pm_h = int(canvas_h * 0.05)
        cyto_y = extra_h + pm_h
        cyto_h = int(canvas_h * 0.40)
        nuc_h = int(cyto_h * 0.35)

        # For 1-2 compartments the bands were redistributed; recompute boundary y
        if len(present_ctypes) <= 2:
            sorted_ctypes = sorted(present_ctypes, key=lambda t: _CANONICAL_TYPE_ORDER.get(t, 99))
            band_h = canvas_h // max(len(present_ctypes), 1)
            cyto_y = band_h if len(sorted_ctypes) >= 2 else 0
            nuc_h = band_h // 3

        cytosol_group = {"cytosol", "nucleus", "endoplasmic reticulum", "mitochondria", "lysosome", "peroxisome", "golgi"}
        has_extracellular = "extracellular" in present_ctypes
        has_cytosol = bool(present_ctypes & cytosol_group)
        has_nucleus = "nucleus" in present_ctypes
        has_mitochondria = "mitochondria" in present_ctypes

        if has_extracellular and has_cytosol:
            membrane_visualizations.append({
                "id": self.ids.next(),
                "complete-membrane": True,
                "x": 0,
                "y": cyto_y,
                "width": str(canvas_w),
                "height": "8",
                "zindex": 5,
            })
        if has_nucleus and has_cytosol:
            membrane_visualizations.append({
                "id": self.ids.next(),
                "complete-membrane": True,
                "x": 0,
                "y": cyto_y + nuc_h,
                "width": str(canvas_w // 2),
                "height": "8",
                "zindex": 5,
            })
        if has_mitochondria and has_cytosol:
            membrane_visualizations.append({
                "id": self.ids.next(),
                "complete-membrane": True,
                "x": 0,
                "y": cyto_y + nuc_h,
                "width": str(canvas_w),
                "height": "8",
                "zindex": 5,
            })

        return {
            "compound-locations": compound_locations,
            "element-collection-locations": element_collection_locations,
            "nucleic-acid-locations": nucleic_acid_locations,
            "protein-locations": protein_locations,
            "protein-complex-visualizations": protein_complex_visualizations,
            "edges": edges,
            "reaction-visualizations": reaction_visualizations,
            "transport-visualizations": transport_visualizations,
            "bound-visualizations": bound_visualizations,
            "membrane-visualizations": membrane_visualizations,
        }

    def _populate_sections(self) -> Dict[str, int]:
        self._prepare_entities()
        biological_states, default_state_id = self._build_biological_states()

        self.section_items["cell-types"] = [
            {"id": int(rec["id"]), "name": rec["name"], "ontology-id": rec.get("ontology_id") or rec.get("ontology-id")}
            for rec in self.entity_records.get("cell-types", [])
        ]
        self.section_items["species"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "taxonomy-id": rec.get("taxonomy_id") or rec.get("taxonomy-id"),
                "classification": rec.get("classification"),
                "common-name": rec.get("common_name") or rec.get("common-name"),
            }
            for rec in self.entity_records.get("species", [])
        ]
        self.section_items["subcellular-locations"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "ontology-id": rec.get("ontology_id") or rec.get("ontology-id"),
            }
            for rec in self.entity_records.get("subcellular-locations", [])
        ]
        self.section_items["tissues"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "ontology-id": rec.get("ontology_id") or rec.get("ontology-id"),
                "visualization-template-id": None,
                "drawable-image-id": None,
            }
            for rec in self.entity_records.get("tissues", [])
        ]
        self.section_items["biological-states"] = biological_states

        self.section_items["bounds"] = []
        self.section_items["compounds"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "pwc-id": f"PW_C{int(rec['id']):06d}",
                "short-name": rec["name"],
                "element-states": [],
                "hmdb-id": rec.get("mapped_ids", {}).get("hmdb") or None,
                "kegg-id": rec.get("mapped_ids", {}).get("kegg") or None,
                "chebi-id": rec.get("mapped_ids", {}).get("chebi") or None,
                "pubchem-cid": rec.get("mapped_ids", {}).get("pubchem") or None,
            }
            for rec in self.entity_records.get("compounds", [])
        ]
        self.section_items["element-collections"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "element-type": "Compound",
                "element-id": None,
                "collection-type": "Set",
                "pwec-id": f"PW_EC{int(rec['id']):06d}",
                "external-id": "",
                "external-id-type": "",
                "short-name": rec["name"],
            }
            for rec in self.entity_records.get("element-collections", [])
        ]
        self.section_items["nucleic-acids"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "element-states": [],
            }
            for rec in self.entity_records.get("nucleic-acids", [])
        ]
        default_species_id = self._resolve_ref_id(None, "species", fallback=True)
        self.section_items["proteins"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "species-id": default_species_id,
                "element-states": [],
                "uniprot-id": rec.get("mapped_ids", {}).get("uniprot") or None,
                "ec-numbers": rec.get("ec_numbers", []),
            }
            for rec in self.entity_records.get("proteins", [])
        ]
        protein_complex_items: List[Dict[str, Any]] = []
        for rec in self.entity_records.get("protein-complexes", []):
            members: List[Dict[str, Any]] = []
            for comp_name in _as_string_list(rec.get("components", [])):
                prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(comp_name))
                if prot:
                    members.append({"id": self.ids.next(), "protein-id": int(prot["id"])})
            protein_complex_items.append(
                {
                    "id": int(rec["id"]),
                    "name": rec["name"],
                    "species-id": default_species_id,
                    "pwp-id": f"PW_P{int(rec['id']):06d}",
                    "protein_complex-proteins": members,
                    "element-states": [],
                }
            )
        self.section_items["protein-complexes"] = protein_complex_items

        reactions = self._build_reactions()
        reaction_coupled_transports = self._build_reaction_coupled_transports()
        transports = self._build_transports(default_state_id)
        interactions = self._build_interactions()

        self.section_items["reactions"] = reactions
        self.section_items["reaction-coupled-transports"] = reaction_coupled_transports
        self.section_items["transports"] = transports
        self.section_items["interactions"] = interactions

        self.section_items["bound-visualizations"] = []

        viz = self._build_locations_and_visualizations(default_state_id, reactions, transports)
        self.section_items.update(viz)

        self.section_items["reaction-coupled-transport_visualizations"] = []
        self.section_items["interaction-visualizations"] = []
        self.section_items["sub-pathway-visualizations"] = []

        self.section_items["vacuous-compound-visualizations"] = []
        self.section_items["vacuous-edge-visualizations"] = []
        self.section_items["vacuous-nucleic-acid-visualizations"] = []
        self.section_items["vacuous-element-collection-visualizations"] = []
        self.section_items["vacuous-protein-visualizations"] = []

        self.section_items["drawable-element-locations"] = []
        self.section_items.setdefault("membrane-visualizations", [])
        self.section_items["label-locations"] = []
        self.section_items["zoom-visualizations"] = []

        return {
            "compounds": len(self.section_items.get("compounds", [])),
            "proteins": len(self.section_items.get("proteins", [])),
            "reactions": len(self.section_items.get("reactions", [])),
            "edges": len(self.section_items.get("edges", [])),
        }

    def _append_scalar(
        self,
        parent: etree._Element,
        tag: str,
        value: Any,
        section_sig: Optional[SectionSignature],
    ) -> etree._Element:
        node = etree.SubElement(parent, tag)
        if section_sig is not None:
            is_integer = tag in section_sig.integer_fields
            is_boolean = tag in section_sig.boolean_fields
        else:
            is_integer = _is_integer_field(tag)
            is_boolean = _is_boolean_field(tag)

        if value is None:
            node.set("nil", "true")
            if is_integer:
                node.set("type", "integer")
            elif is_boolean:
                node.set("type", "boolean")
            return node

        if is_boolean or isinstance(value, bool):
            node.set("type", "boolean")
            node.text = "true" if bool(value) else "false"
            return node

        if is_integer and isinstance(value, int):
            node.set("type", "integer")
            node.text = str(value)
            return node

        node.text = str(value)
        return node

    def _emit_item(
        self,
        parent: etree._Element,
        item: Dict[str, Any],
        section_sig: Optional[SectionSignature],
    ) -> None:
        ordered_fields: List[str] = []
        if section_sig:
            ordered_fields.extend(section_sig.required_fields)
        ordered_fields.extend([key for key in item.keys() if key not in ordered_fields])

        for field in ordered_fields:
            has_value = field in item
            if not has_value and not section_sig:
                continue
            if not has_value and section_sig and field not in section_sig.required_fields:
                continue

            value = item.get(field)
            if isinstance(value, list):
                container = etree.SubElement(parent, field)
                item_tag = _singularize(field)
                for entry in value:
                    if isinstance(entry, dict):
                        child = etree.SubElement(container, item_tag)
                        self._emit_item(child, entry, None)
                    else:
                        child = etree.SubElement(container, item_tag)
                        child.text = str(entry)
                continue

            if isinstance(value, dict):
                container = etree.SubElement(parent, field)
                self._emit_item(container, value, None)
                continue

            if not has_value and section_sig and field in section_sig.nil_fields:
                self._append_scalar(parent, field, None, section_sig)
                continue
            if not has_value:
                node = etree.SubElement(parent, field)
                if section_sig and field in section_sig.integer_fields:
                    node.set("type", "integer")
                continue

            self._append_scalar(parent, field, value, section_sig)

    def _emit_section(self, pv: etree._Element, section_tag: str) -> None:
        section_node = etree.SubElement(pv, section_tag)
        items = self.section_items.get(section_tag, [])
        section_sig = self.signature.sections.get(section_tag)
        item_tag = section_sig.item_tag if section_sig else _singularize(section_tag)
        for item in items:
            item_node = etree.SubElement(section_node, item_tag)
            self._emit_item(item_node, item, section_sig)

    def _emit_pathway(self, pv: etree._Element) -> None:
        pathway = etree.SubElement(pv, "pathway")
        first_species_id = self._resolve_ref_id(None, "species", fallback=True)
        pathway_values: Dict[str, Any] = {
            "id": self.pathway_id_int,
            "name": self.args.name,
            "description": self.args.description,
            "subject": self.args.subject,
            "species-id": first_species_id,
            "sub-pathways": [],
            "references": [],
        }

        ordered_fields = self.signature.pathway_children or list(pathway_values.keys())
        for field in ordered_fields:
            value = pathway_values.get(field)
            if isinstance(value, list):
                container = etree.SubElement(pathway, field)
                item_tag = _singularize(field)
                for entry in value:
                    item_node = etree.SubElement(container, item_tag)
                    if isinstance(entry, dict):
                        self._emit_item(item_node, entry, None)
                    else:
                        item_node.text = str(entry)
                continue
            if field not in pathway_values:
                etree.SubElement(pathway, field)
                continue

            node = etree.SubElement(pathway, field)
            if field in {"id", "species-id"}:
                node.set("type", "integer")
            if value is None:
                node.set("nil", "true")
            else:
                node.text = str(value)

    def build(self) -> BuildResult:
        counts = self._populate_sections()

        root = etree.Element(self.signature.root_tag)
        for tag in self.signature.root_children:
            if tag == "named-for-id":
                node = etree.SubElement(root, tag)
                node.set("type", "integer")
                node.text = str(self.pathway_id_int)
            elif tag == "named-for-type":
                node = etree.SubElement(root, tag)
                node.text = "Pathway"
            elif tag == "cached-name":
                node = etree.SubElement(root, tag)
                node.text = self.args.name
            elif tag == "cached-description":
                node = etree.SubElement(root, tag)
                node.text = self.args.description
            elif tag == "cached-subject":
                node = etree.SubElement(root, tag)
                node.text = self.args.subject
            elif tag == "pw-id":
                node = etree.SubElement(root, tag)
                node.text = self.args.pw_id
            elif tag == "pathway-visualization-contexts":
                contexts = etree.SubElement(root, tag)
                context = etree.SubElement(contexts, "pathway-visualization-context")
                pos = etree.SubElement(context, "position")
                pos.text = "Center"
                ctx_id = etree.SubElement(context, "id")
                ctx_id.text = self.pathway_visualization_context_id
                pv = etree.SubElement(context, "pathway-visualization")
                for child_tag in self.signature.pv_children:
                    if child_tag == "height":
                        n = etree.SubElement(pv, "height")
                        n.set("type", "integer")
                        n.text = str(self.args.height)
                    elif child_tag == "width":
                        n = etree.SubElement(pv, "width")
                        n.set("type", "integer")
                        n.text = str(self.args.width)
                    elif child_tag == "background-color":
                        n = etree.SubElement(pv, "background-color")
                        n.text = self.args.background_color
                    elif child_tag == "id":
                        n = etree.SubElement(pv, "id")
                        n.text = self.pathway_visualization_id
                    elif child_tag == "pathway":
                        self._emit_pathway(pv)
                    else:
                        self._emit_section(pv, child_tag)
            else:
                etree.SubElement(root, tag)

        geometry_generated = bool(self.section_items.get("compound-locations") or self.section_items.get("edges"))
        return BuildResult(
            root=root,
            counts=counts,
            geometry_generated=geometry_generated,
            signature=self.signature,
        )


def load_extraction(path: Path | str) -> Dict[str, Any]:
    content = Path(path).read_text(encoding="utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Input extraction JSON must be an object")
    return payload


def run_writer(args: argparse.Namespace) -> Dict[str, Any]:
    signature = discover_structure_signature(args.ref)
    extraction = load_extraction(args.in_path)

    print("Discovered root child order:")
    for tag in signature.root_children:
        print(f" - {tag}")
    print("Discovered pathway-visualization child order:")
    for tag in signature.pv_children:
        print(f" - {tag}")

    builder = DeterministicPwmlBuilder(extraction=extraction, signature=signature, args=args)
    build = builder.build()
    tree = etree.ElementTree(build.root)

    repaired = repair_tree(tree, signature)
    report = validate_generated_tree(repaired, signature)
    if not report["ok"]:
        repaired = repair_tree(repaired, signature)
        report = validate_generated_tree(repaired, signature)

    out_path = Path(args.out)
    repaired.write(str(out_path), encoding="utf-8", xml_declaration=True, pretty_print=True)

    report_path = Path(args.report) if args.report else out_path.with_suffix(".validation.json")
    write_json_report(report, report_path)

    if args.snapshot:
        snapshot_path = Path(args.snapshot)
        snapshot_path.write_text(json.dumps(signature.to_dict(), indent=2), encoding="utf-8")

    print(
        "Emitted counts:"
        f" compounds={build.counts['compounds']},"
        f" proteins={build.counts['proteins']},"
        f" reactions={build.counts['reactions']},"
        f" edges={build.counts['edges']}"
    )
    print(f"Dummy geometry generated: {'yes' if build.geometry_generated else 'no'}")
    print(f"Validation: {'PASS' if report['ok'] else 'FAIL'} ({report['issue_count']} issues)")
    print(f"Validation report: {report_path}")
    print(f"Output PWML: {out_path}")

    return {
        "output": str(out_path),
        "report": str(report_path),
        "ok": report["ok"],
        "issues": report["issue_count"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic PathBank-style PWML writer")
    parser.add_argument("--in", dest="in_path", required=True, help="Input extraction JSON path")
    parser.add_argument("--ref", required=True, help="Reference PWML path")
    parser.add_argument("--out", required=True, help="Output PWML path")
    parser.add_argument("--pw-id", default="PW000000", help="PW identifier for root <pw-id>")
    parser.add_argument("--named-for-id", type=int, default=1, help="Root/pathway integer id")
    parser.add_argument("--name", default="Generated Pathway", help="Pathway name")
    parser.add_argument("--description", default="", help="Pathway description")
    parser.add_argument("--subject", default="Metabolic", help="Pathway subject")
    parser.add_argument("--height", type=int, default=1400, help="Pathway visualization height")
    parser.add_argument("--width", type=int, default=3200, help="Pathway visualization width")
    parser.add_argument("--background-color", default="#FFFFFF", help="Pathway visualization background color")
    parser.add_argument("--report", default="", help="Validation mismatch report path")
    parser.add_argument("--snapshot", default="", help="Optional path to write discovered signature JSON")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_writer(args)


if __name__ == "__main__":
    main()
