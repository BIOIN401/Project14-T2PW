from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from lxml import etree

from t2pw.paths import PROJECT_ROOT
from t2pw.pwml.ir import build_pwml_ir, is_pwml_ir, validate_pwml_ir
from t2pw.pwml.qa import run_pwml_qa
from t2pw.pwml.validate import (
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


def _to_positive_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _component_name(component: Any) -> str:
    if isinstance(component, str):
        return component.strip()
    if not isinstance(component, dict):
        return ""
    return str(
        component.get("protein")
        or component.get("protein_name")
        or component.get("name")
        or component.get("component")
        or component.get("entity")
        or ""
    ).strip()


def _component_stoichiometry(component: Any) -> Optional[int]:
    if isinstance(component, str):
        return 1
    if not isinstance(component, dict):
        return None
    return _to_positive_int(component.get("stoichiometry") or component.get("coefficient"))


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
        self._ir_key_ids: Dict[str, Dict[str, int]] = {}
        self._ir_entity_info: Dict[str, Dict[str, Any]] = {}
        self._ir_pathway_species_id: Optional[int] = None

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
                            try:
                                pw_id = int(v)
                                break
                            except (ValueError, TypeError):
                                pass
                    record["id"] = pw_id if pw_id is not None else self.compound_ids.next()
                elif key == "proteins":
                    pw_id = None
                    for k in ["pathbank_protein_id", "pw_protein_id"]:
                        v = record.get(k) or (record.get("mapping_meta") or {}).get(k)
                        if v:
                            try:
                                pw_id = int(v)
                                break
                            except (ValueError, TypeError):
                                pass
                    record["id"] = pw_id if pw_id is not None else self.protein_ids.next()
                elif key == "protein_complexes":
                    pw_id = None
                    for k in ["pathbank_protein_complex_id", "pathbank_complex_id", "pw_complex_id"]:
                        v = record.get(k) or (record.get("mapping_meta") or {}).get(k)
                        if v:
                            try:
                                pw_id = int(v)
                                break
                            except (ValueError, TypeError):
                                pass
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

    def _protein_complex_members(
        self,
        components: Any,
        *,
        complex_name: str,
        protein_id_by_key: Optional[Dict[str, int]] = None,
        protein_id_by_name: Optional[Dict[str, int]] = None,
        protein_id_by_db_id: Optional[Dict[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        members: List[Dict[str, Any]] = []
        for idx, component in enumerate(components if isinstance(components, list) else []):
            stoich = _component_stoichiometry(component)
            if stoich is None:
                raise ValueError(f"Protein complex '{complex_name}' component[{idx}] is missing stoichiometry.")

            protein_id: Optional[int] = None
            if isinstance(component, dict):
                protein_key = str(component.get("protein_key") or component.get("entity_key") or "").strip()
                if protein_key and protein_id_by_key:
                    protein_id = protein_id_by_key.get(protein_key)
                if protein_id is None and protein_key:
                    for rec in self.entity_records.get("proteins", []):
                        if str(rec.get("key") or "") == protein_key:
                            protein_id = int(rec["id"])
                            break
                if protein_id is None:
                    for db_key in ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"]:
                        db_id = _to_positive_int(component.get(db_key))
                        if db_id is not None and protein_id_by_db_id:
                            protein_id = protein_id_by_db_id.get(db_id)
                        if protein_id is not None:
                            break

            name = _component_name(component)
            if protein_id is None and name:
                if protein_id_by_name:
                    protein_id = protein_id_by_name.get(_normalize_key(name))
                if protein_id is None:
                    prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(name))
                    if prot:
                        protein_id = int(prot["id"])

            if protein_id is None:
                label = name or (
                    str(component.get("protein_key") or component.get("entity_key"))
                    if isinstance(component, dict)
                    else f"component[{idx}]"
                )
                raise ValueError(
                    f"Protein complex '{complex_name}' component '{label}' does not reference an existing protein."
                )

            members.append({"id": self.ids.next(), "protein-id": int(protein_id), "stoichiometry": stoich})

        if not members:
            raise ValueError(f"Protein complex '{complex_name}' has no protein_complex-proteins to export.")
        return members

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
            for component in rec.get("components", []) if isinstance(rec.get("components"), list) else []:
                prot: Optional[Dict[str, Any]] = None
                if isinstance(component, dict):
                    protein_key = str(component.get("protein_key") or component.get("entity_key") or "").strip()
                    if protein_key:
                        prot = next(
                            (
                                p
                                for p in self.entity_records.get("proteins", [])
                                if str(p.get("key") or "") == protein_key
                            ),
                            None,
                        )
                comp_name = _component_name(component)
                if prot is None and comp_name:
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

    def _populate_sections_from_ir(self) -> Dict[str, int]:
        validation = validate_pwml_ir(self.extraction)
        if not validation.get("ok"):
            messages = [
                str(issue.get("message") or issue.get("code") or "PWML IR validation error")
                for issue in validation.get("errors", [])[:5]
                if isinstance(issue, dict)
            ]
            raise ValueError("PWML IR validation failed before serialization: " + "; ".join(messages))

        ir = self.extraction
        self._ir_key_ids = {}
        self._ir_entity_info = {}
        self._ir_pathway_species_id = None
        member_ids: Dict[str, Dict[str, Any]] = {}

        def remember(namespace: str, key: Any, value: int) -> None:
            if key is None:
                return
            self._ir_key_ids.setdefault(namespace, {})[str(key)] = int(value)

        def lookup(namespace: str, key: Any) -> Optional[int]:
            if key is None:
                return None
            return self._ir_key_ids.get(namespace, {}).get(str(key))

        def first_int(record: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
            for key in keys:
                value = record.get(key)
                if value not in (None, ""):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            meta = record.get("mapping_meta") if isinstance(record.get("mapping_meta"), dict) else {}
            for key in keys:
                value = meta.get(key)
                if value not in (None, ""):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            mapped = record.get("mapped_ids") if isinstance(record.get("mapped_ids"), dict) else {}
            for key in keys:
                value = mapped.get(key)
                if value not in (None, ""):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            return None

        def id_for(record: Dict[str, Any], keys: Sequence[str], fallback: IdFactory) -> int:
            return first_int(record, keys) or fallback.next()

        def direction_for(value: Any) -> str:
            text = str(value or "").strip().casefold()
            if text in {"<", "left", "reverse"}:
                return "Left"
            if text in {"<>", "<=>", "both", "reversible"}:
                return "Both"
            return "Right"

        def entity_info(entity_key: Any) -> Optional[Dict[str, Any]]:
            if entity_key is None:
                return None
            return self._ir_entity_info.get(str(entity_key))

        def element_type(entity_type: str) -> str:
            return {
                "compound": "Compound",
                "protein": "Protein",
                "protein_complex": "ProteinComplex",
                "nucleic_acid": "NucleicAcid",
                "element_collection": "ElementCollection",
            }.get(entity_type, "")

        self.section_items = {}

        self.section_items["cell-types"] = []
        for record in ir.get("cell_types", []) if isinstance(ir.get("cell_types"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(record, ["pathwhiz_id", "pathbank_cell_type_id", "pw_cell_type_id"], self.ids)
            remember("cell_types", record.get("key"), rid)
            self.section_items["cell-types"].append(
                {"id": rid, "name": record.get("name", ""), "ontology-id": record.get("ontology_id")}
            )

        self.section_items["species"] = []
        for record in ir.get("species", []) if isinstance(ir.get("species"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(record, ["pathwhiz_id", "pathbank_species_id", "pw_species_id"], self.ids)
            remember("species", record.get("key"), rid)
            if self._ir_pathway_species_id is None:
                self._ir_pathway_species_id = rid
            self.section_items["species"].append(
                {
                    "id": rid,
                    "name": record.get("name", ""),
                    "taxonomy-id": record.get("taxonomy_id"),
                    "classification": record.get("classification"),
                    "common-name": record.get("common_name"),
                }
            )

        self.section_items["subcellular-locations"] = []
        for record in ir.get("subcellular_locations", []) if isinstance(ir.get("subcellular_locations"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(
                record,
                ["pathwhiz_id", "pathbank_subcellular_location_id", "pw_subcellular_location_id"],
                self.ids,
            )
            remember("subcellular_locations", record.get("key"), rid)
            self.section_items["subcellular-locations"].append(
                {"id": rid, "name": record.get("name", ""), "ontology-id": record.get("ontology_id")}
            )

        self.section_items["tissues"] = []
        for record in ir.get("tissues", []) if isinstance(ir.get("tissues"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(record, ["pathwhiz_id", "pathbank_tissue_id", "pw_tissue_id"], self.ids)
            remember("tissues", record.get("key"), rid)
            self.section_items["tissues"].append(
                {
                    "id": rid,
                    "name": record.get("name", ""),
                    "ontology-id": record.get("ontology_id"),
                    "visualization-template-id": None,
                    "drawable-image-id": None,
                }
            )

        biological_states = []
        for record in ir.get("biological_states", []) if isinstance(ir.get("biological_states"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = self.ids.next()
            remember("biological_states", record.get("key"), rid)
            biological_states.append(
                {
                    "id": rid,
                    "name": record.get("name") or record.get("key") or f"State {rid}",
                    "tissue-id": lookup("tissues", record.get("tissue_key")),
                    "subcellular-location-id": lookup(
                        "subcellular_locations", record.get("subcellular_location_key")
                    ),
                    "species-id": lookup("species", record.get("species_key")),
                    "cell-type-id": lookup("cell_types", record.get("cell_type_key")),
                    "pwbs-id": f"PW_BS{rid:06d}",
                }
            )
        self.section_items["biological-states"] = biological_states

        entities = ir.get("entities") if isinstance(ir.get("entities"), dict) else {}
        self.section_items["bounds"] = []
        self.section_items["compounds"] = []
        for record in entities.get("compounds", []) if isinstance(entities.get("compounds"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(record, ["pathwhiz_id", "pathbank_compound_id", "pw_compound_id"], self.compound_ids)
            remember("entities", record.get("key"), rid)
            self._ir_entity_info[str(record.get("key"))] = {"id": rid, "type": "Compound", "entity_type": "compound"}
            mapped_ids = record.get("mapped_ids") if isinstance(record.get("mapped_ids"), dict) else {}
            self.section_items["compounds"].append(
                {
                    "id": rid,
                    "name": record.get("name", ""),
                    "pwc-id": f"PW_C{rid:06d}",
                    "short-name": record.get("name", ""),
                    "element-states": [],
                    "hmdb-id": mapped_ids.get("hmdb") or None,
                    "kegg-id": mapped_ids.get("kegg") or None,
                    "chebi-id": mapped_ids.get("chebi") or None,
                    "pubchem-cid": mapped_ids.get("pubchem") or None,
                }
            )

        self.section_items["element-collections"] = []
        for record in entities.get("element_collections", []) if isinstance(entities.get("element_collections"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(
                record,
                ["pathwhiz_id", "pathbank_element_collection_id", "pw_element_collection_id"],
                self.ids,
            )
            remember("entities", record.get("key"), rid)
            self._ir_entity_info[str(record.get("key"))] = {
                "id": rid,
                "type": "ElementCollection",
                "entity_type": "element_collection",
            }
            self.section_items["element-collections"].append(
                {
                    "id": rid,
                    "name": record.get("name", ""),
                    "element-type": "Compound",
                    "element-id": None,
                    "collection-type": "Set",
                    "pwec-id": f"PW_EC{rid:06d}",
                    "external-id": "",
                    "external-id-type": "",
                    "short-name": record.get("name", ""),
                }
            )

        self.section_items["nucleic-acids"] = []
        for record in entities.get("nucleic_acids", []) if isinstance(entities.get("nucleic_acids"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(record, ["pathwhiz_id", "pathbank_nucleic_acid_id", "pw_nucleic_acid_id"], self.ids)
            remember("entities", record.get("key"), rid)
            self._ir_entity_info[str(record.get("key"))] = {
                "id": rid,
                "type": "NucleicAcid",
                "entity_type": "nucleic_acid",
            }
            self.section_items["nucleic-acids"].append(
                {"id": rid, "name": record.get("name", ""), "element-states": []}
            )

        default_species_id = self._ir_pathway_species_id
        self.section_items["proteins"] = []
        protein_id_by_key: Dict[str, int] = {}
        protein_id_by_name: Dict[str, int] = {}
        protein_id_by_db_id: Dict[int, int] = {}
        for record in entities.get("proteins", []) if isinstance(entities.get("proteins"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(record, ["pathwhiz_id", "pathbank_protein_id", "pw_protein_id"], self.protein_ids)
            remember("entities", record.get("key"), rid)
            self._ir_entity_info[str(record.get("key"))] = {"id": rid, "type": "Protein", "entity_type": "protein"}
            if record.get("key"):
                protein_id_by_key[str(record.get("key"))] = rid
            if record.get("name"):
                protein_id_by_name[_normalize_key(str(record.get("name")))] = rid
            for db_key in ["pathwhiz_id", "pathbank_protein_id", "pw_protein_id"]:
                db_id = _to_positive_int(record.get(db_key))
                if db_id is not None:
                    protein_id_by_db_id[db_id] = rid
            mapped_ids = record.get("mapped_ids") if isinstance(record.get("mapped_ids"), dict) else {}
            self.section_items["proteins"].append(
                {
                    "id": rid,
                    "name": record.get("name", ""),
                    "species-id": default_species_id,
                    "element-states": [],
                    "uniprot-id": mapped_ids.get("uniprot") or None,
                    "ec-numbers": record.get("ec_numbers", []),
                }
            )

        self.section_items["protein-complexes"] = []
        for record in entities.get("protein_complexes", []) if isinstance(entities.get("protein_complexes"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(
                record,
                ["pathwhiz_id", "pathbank_protein_complex_id", "pathbank_complex_id", "pw_complex_id"],
                self.complex_ids,
            )
            remember("entities", record.get("key"), rid)
            self._ir_entity_info[str(record.get("key"))] = {
                "id": rid,
                "type": "ProteinComplex",
                "entity_type": "protein_complex",
            }
            members = self._protein_complex_members(
                record.get("components"),
                complex_name=str(record.get("name") or rid),
                protein_id_by_key=protein_id_by_key,
                protein_id_by_name=protein_id_by_name,
                protein_id_by_db_id=protein_id_by_db_id,
            )
            self.section_items["protein-complexes"].append(
                {
                    "id": rid,
                    "name": record.get("name", ""),
                    "species-id": default_species_id,
                    "pwp-id": f"PW_P{rid:06d}",
                    "protein_complex-proteins": members,
                    "element-states": [],
                }
            )

        process_items = ir.get("processes") if isinstance(ir.get("processes"), dict) else {}

        reactions: List[Dict[str, Any]] = []
        for reaction in process_items.get("reactions", []) if isinstance(process_items.get("reactions"), list) else []:
            if not isinstance(reaction, dict):
                continue
            rid = self.reaction_ids.next()
            remember("processes", reaction.get("key"), rid)
            left: List[Dict[str, Any]] = []
            right: List[Dict[str, Any]] = []
            for side_key, out in [("left", left), ("right", right)]:
                for member in reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []:
                    if not isinstance(member, dict):
                        continue
                    info = entity_info(member.get("entity_key"))
                    if not info:
                        continue
                    mid = self.ids.next()
                    member_ids[str(member.get("key"))] = {
                        "id": mid,
                        "entity_type": member.get("entity_type"),
                        "process_key": reaction.get("key"),
                    }
                    out.append(
                        {
                            "id": mid,
                            "element-id": int(info["id"]),
                            "stoichiometry": int(member.get("stoichiometry") or 1),
                            "element-type": info["type"],
                            "currency": False,
                        }
                    )
            enzymes: List[Dict[str, Any]] = []
            for member in reaction.get("enzymes", []) if isinstance(reaction.get("enzymes"), list) else []:
                if not isinstance(member, dict):
                    continue
                info = entity_info(member.get("entity_key"))
                if not info:
                    continue
                mid = self.ids.next()
                member_ids[str(member.get("key"))] = {
                    "id": mid,
                    "entity_type": member.get("entity_type"),
                    "process_key": reaction.get("key"),
                    "entity_key": member.get("entity_key"),
                }
                item: Dict[str, Any] = {"id": mid}
                if info["entity_type"] == "protein_complex":
                    item["protein-complex-id"] = int(info["id"])
                else:
                    item["protein-id"] = int(info["id"])
                if str(member.get("role") or "").casefold() == "inhibitor":
                    item["inhibitor"] = True
                enzymes.append(item)
            reactions.append(
                {
                    "id": rid,
                    "spontaneous": reaction.get("spontaneous"),
                    "pwr-id": f"PW_R{rid:06d}",
                    "direction": direction_for(reaction.get("direction")),
                    "reaction-left-elements": left,
                    "reaction-right-elements": right,
                    "reaction-enzymes": enzymes,
                }
            )
        self.section_items["reactions"] = reactions

        transports: List[Dict[str, Any]] = []
        for transport in process_items.get("transports", []) if isinstance(process_items.get("transports"), list) else []:
            if not isinstance(transport, dict):
                continue
            tid = self.ids.next()
            remember("processes", transport.get("key"), tid)
            elements_out: List[Dict[str, Any]] = []
            for member in transport.get("transport_elements", []) if isinstance(transport.get("transport_elements"), list) else []:
                if not isinstance(member, dict):
                    continue
                info = entity_info(member.get("entity_key"))
                if not info:
                    continue
                mid = self.ids.next()
                member_ids[str(member.get("key"))] = {
                    "id": mid,
                    "entity_type": member.get("entity_type"),
                    "process_key": transport.get("key"),
                }
                elements_out.append(
                    {
                        "id": mid,
                        "element-id": int(info["id"]),
                        "stoichiometry": int(member.get("stoichiometry") or 1),
                        "element-type": info["type"],
                        "left-biological-state-id": lookup("biological_states", member.get("left_biological_state_key")),
                        "right-biological-state-id": lookup("biological_states", member.get("right_biological_state_key")),
                        "direction": "Right",
                    }
                )
            transporters_out: List[Dict[str, Any]] = []
            for member in transport.get("transporters", []) if isinstance(transport.get("transporters"), list) else []:
                if not isinstance(member, dict):
                    continue
                info = entity_info(member.get("entity_key"))
                if not info:
                    continue
                mid = self.ids.next()
                member_ids[str(member.get("key"))] = {
                    "id": mid,
                    "entity_type": member.get("entity_type"),
                    "process_key": transport.get("key"),
                    "entity_key": member.get("entity_key"),
                }
                item: Dict[str, Any] = {"id": mid, "biological-state-id": lookup("biological_states", member.get("biological_state_key"))}
                if info["entity_type"] == "protein_complex":
                    item["protein-complex-id"] = int(info["id"])
                else:
                    item["protein-id"] = int(info["id"])
                transporters_out.append(item)
            transports.append(
                {
                    "id": tid,
                    "pwt-id": f"PW_T{tid:06d}",
                    "transport-type": transport.get("transport_type"),
                    "transport-elements": elements_out,
                    "transport-transporters": transporters_out,
                }
            )
        self.section_items["transports"] = transports

        self.section_items["reaction-coupled-transports"] = [
            {"id": self.ids.next(), "name": item.get("name", "")}
            for item in process_items.get("reaction_coupled_transports", [])
            if isinstance(item, dict)
        ]
        for item, raw in zip(
            self.section_items["reaction-coupled-transports"],
            process_items.get("reaction_coupled_transports", []) if isinstance(process_items.get("reaction_coupled_transports"), list) else [],
        ):
            if isinstance(raw, dict):
                remember("processes", raw.get("key"), int(item["id"]))

        self.section_items["interactions"] = []
        for interaction in process_items.get("interactions", []) if isinstance(process_items.get("interactions"), list) else []:
            if not isinstance(interaction, dict):
                continue
            iid = self.ids.next()
            remember("processes", interaction.get("key"), iid)
            self.section_items["interactions"].append(
                {
                    "id": iid,
                    "name": interaction.get("name", ""),
                    "interaction-type": interaction.get("interaction_type"),
                }
            )

        self.section_items["bound-visualizations"] = []
        for item in ir.get("bound_visualizations", []) if isinstance(ir.get("bound_visualizations"), list) else []:
            if not isinstance(item, dict):
                continue
            vid = self.ids.next()
            remember("bound_visualizations", item.get("key"), vid)
            self.section_items["bound-visualizations"].append(
                {
                    "id": vid,
                    "biological-state-id": lookup("biological_states", item.get("biological_state_key")),
                    "x": int(item.get("x") or 0),
                    "y": int(item.get("y") or 0),
                    "width": str(item.get("width") or self.args.width),
                    "height": str(item.get("height") or self.args.height),
                    "zindex": int(item.get("zindex") or 1),
                    "hidden": bool(item.get("hidden", False)),
                }
            )

        section_by_location_type = {
            "compound_location": ("compound-locations", "compound-id"),
            "protein_location": ("protein-locations", "protein-id"),
            "nucleic_acid_location": ("nucleic-acid-locations", "nucleic-acid-id"),
            "element_collection_location": ("element-collection-locations", "element-collection-id"),
        }
        for section in [
            "compound-locations",
            "protein-locations",
            "nucleic-acid-locations",
            "element-collection-locations",
        ]:
            self.section_items[section] = []
        for loc in ir.get("locations", []) if isinstance(ir.get("locations"), list) else []:
            if not isinstance(loc, dict):
                continue
            section_info = section_by_location_type.get(str(loc.get("type") or ""))
            info = entity_info(loc.get("entity_key"))
            if not section_info or not info:
                continue
            section, entity_field = section_info
            lid = self.ids.next()
            remember("locations", loc.get("key"), lid)
            self.section_items[section].append(
                {
                    "id": lid,
                    entity_field: int(info["id"]),
                    "biological-state-id": lookup("biological_states", loc.get("biological_state_key")),
                    "visualization-template-id": int(loc.get("visualization_template_id") or 0),
                    "hidden": bool(loc.get("hidden", False)),
                    "x": int(loc.get("x") or 0),
                    "y": int(loc.get("y") or 0),
                    "zindex": int(loc.get("zindex") or 10),
                    "font-size": str(loc.get("font_size") or loc.get("font-size") or "regular"),
                    "width": str(loc.get("width") or 160),
                    "height": str(loc.get("height") or 60),
                }
            )

        self.section_items["protein-complex-visualizations"] = []
        protein_complex_viz_by_entity: Dict[str, int] = {}
        protein_complex_viz_by_entity_state: Dict[Tuple[str, str], int] = {}
        for item in ir.get("protein_complex_visualizations", []) if isinstance(ir.get("protein_complex_visualizations"), list) else []:
            if not isinstance(item, dict):
                continue
            info = entity_info(item.get("entity_key"))
            if not info:
                continue
            vid = self.ids.next()
            remember("protein_complex_visualizations", item.get("key"), vid)
            entity_key = str(item.get("entity_key"))
            biological_state_key = str(item.get("biological_state_key"))
            protein_complex_viz_by_entity[entity_key] = vid
            protein_complex_viz_by_entity_state[(entity_key, biological_state_key)] = vid
            self.section_items["protein-complex-visualizations"].append(
                {
                    "id": vid,
                    "protein-complex-id": int(info["id"]),
                    "pathway-visualization-id": self.pathway_visualization_id_int,
                    "biological-state-id": lookup("biological_states", item.get("biological_state_key")),
                    "protein_complex_protein_visualizations": [],
                }
            )

        self.section_items["edges"] = []
        for edge in ir.get("edges", []) if isinstance(ir.get("edges"), list) else []:
            if not isinstance(edge, dict):
                continue
            eid = self.ids.next()
            remember("edges", edge.get("key"), eid)
            self.section_items["edges"].append(
                {
                    "id": eid,
                    "path": edge.get("path", ""),
                    "visualization-template-id": int(edge.get("visualization_template_id") or 0),
                    "hidden": bool(edge.get("hidden", False)),
                    "zindex": int(edge.get("zindex") or 18),
                }
            )

        self.section_items["reaction-visualizations"] = []
        self.section_items["transport-visualizations"] = []
        self.section_items["reaction-coupled-transport_visualizations"] = []
        self.section_items["interaction-visualizations"] = []
        self.section_items["sub-pathway-visualizations"] = []

        for viz in ir.get("process_visualizations", []) if isinstance(ir.get("process_visualizations"), list) else []:
            if not isinstance(viz, dict):
                continue
            vtype = str(viz.get("type") or "")
            process_id = lookup("processes", viz.get("process_key"))
            if process_id is None:
                continue
            if vtype == "reaction_visualization":
                item = {
                    "id": self.ids.next(),
                    "pathway-visualization-id": self.pathway_visualization_id_int,
                    "reaction-id": process_id,
                    "biological-state-id": lookup("biological_states", viz.get("biological_state_key")),
                    "reaction_compound_visualizations": [],
                    "reaction_element_collection_visualizations": [],
                    "reaction_nucleic_acid_visualizations": [],
                    "reaction_enzyme_visualizations": [],
                }
                for member in viz.get("members", []) if isinstance(viz.get("members"), list) else []:
                    if not isinstance(member, dict):
                        continue
                    minfo = member_ids.get(str(member.get("process_member_key")))
                    loc_id = lookup("locations", member.get("location_key"))
                    edge_id = lookup("edges", member.get("edge_key"))
                    if not minfo:
                        continue
                    role = str(member.get("role") or "")
                    side = "Left" if role == "left" else "Right" if role == "right" else ""
                    mtype = str(member.get("member_type") or minfo.get("entity_type") or "")
                    if role == "enzyme":
                        ev = {"id": self.ids.next(), "reaction-enzyme-id": int(minfo["id"])}
                        entity_key = minfo.get("entity_key")
                        pcv_id = None
                        if entity_key is not None:
                            pcv_id = protein_complex_viz_by_entity_state.get(
                                (str(entity_key), str(viz.get("biological_state_key")))
                            )
                            if pcv_id is None:
                                pcv_id = protein_complex_viz_by_entity.get(str(entity_key))
                        if mtype == "protein_complex" and pcv_id:
                            ev["protein-complex-visualization-id"] = pcv_id
                        elif loc_id is not None:
                            ev["protein-location-id"] = loc_id
                        item["reaction_enzyme_visualizations"].append(ev)
                    elif mtype == "compound" and loc_id is not None and edge_id is not None:
                        item["reaction_compound_visualizations"].append(
                            {"id": self.ids.next(), "compound-location-id": loc_id, "edge-id": edge_id, "side": side}
                        )
                    elif mtype == "element_collection" and loc_id is not None and edge_id is not None:
                        item["reaction_element_collection_visualizations"].append(
                            {
                                "id": self.ids.next(),
                                "element-collection-location-id": loc_id,
                                "edge-id": edge_id,
                                "side": side,
                            }
                        )
                    elif mtype == "nucleic_acid" and loc_id is not None and edge_id is not None:
                        item["reaction_nucleic_acid_visualizations"].append(
                            {"id": self.ids.next(), "nucleic-acid-location-id": loc_id, "edge-id": edge_id, "side": side}
                        )
                self.section_items["reaction-visualizations"].append(item)
            elif vtype == "transport_visualization":
                item = {
                    "id": self.ids.next(),
                    "transport-id": process_id,
                    "pathway-visualization-id": self.pathway_visualization_id_int,
                    "transport_compound_visualizations": [],
                    "transport_transporter_visualizations": [],
                }
                for member in viz.get("members", []) if isinstance(viz.get("members"), list) else []:
                    if not isinstance(member, dict):
                        continue
                    minfo = member_ids.get(str(member.get("process_member_key")))
                    loc_id = lookup("locations", member.get("location_key"))
                    edge_id = lookup("edges", member.get("edge_key"))
                    if not minfo:
                        continue
                    role = str(member.get("role") or "")
                    side = "Left" if role == "left" else "Right" if role == "right" else ""
                    mtype = str(member.get("member_type") or minfo.get("entity_type") or "")
                    if mtype == "compound" and loc_id is not None and edge_id is not None:
                        item["transport_compound_visualizations"].append(
                            {"id": self.ids.next(), "compound-location-id": loc_id, "edge-id": edge_id, "side": side}
                        )
                    elif role == "transporter":
                        tv = {"id": self.ids.next(), "transport-transporter-id": int(minfo["id"])}
                        entity_key = minfo.get("entity_key")
                        pcv_id = None
                        if entity_key is not None:
                            pcv_id = protein_complex_viz_by_entity_state.get(
                                (str(entity_key), str(viz.get("biological_state_key")))
                            )
                            if pcv_id is None:
                                pcv_id = protein_complex_viz_by_entity.get(str(entity_key))
                        if mtype == "protein_complex" and pcv_id:
                            tv["protein-complex-visualization-id"] = pcv_id
                        elif loc_id is not None:
                            tv["protein-location-id"] = loc_id
                        item["transport_transporter_visualizations"].append(tv)
                self.section_items["transport-visualizations"].append(item)

        self.section_items["vacuous-compound-visualizations"] = []
        self.section_items["vacuous-edge-visualizations"] = []
        self.section_items["vacuous-nucleic-acid-visualizations"] = []
        self.section_items["vacuous-element-collection-visualizations"] = []
        self.section_items["vacuous-protein-visualizations"] = []
        self.section_items["drawable-element-locations"] = []
        self.section_items["membrane-visualizations"] = []
        self.section_items["label-locations"] = []
        self.section_items["zoom-visualizations"] = []

        return {
            "compounds": len(self.section_items.get("compounds", [])),
            "proteins": len(self.section_items.get("proteins", [])),
            "reactions": len(self.section_items.get("reactions", [])),
            "edges": len(self.section_items.get("edges", [])),
        }

    def _populate_sections(self) -> Dict[str, int]:
        if is_pwml_ir(self.extraction):
            return self._populate_sections_from_ir()

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
                "hmdb-id": (rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}).get("hmdb") or None,
                "kegg-id": (rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}).get("kegg") or None,
                "chebi-id": (rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}).get("chebi") or None,
                "pubchem-cid": (rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}).get("pubchem") or None,
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
                "uniprot-id": (rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}).get("uniprot") or None,
                "ec-numbers": rec.get("ec_numbers", []),
            }
            for rec in self.entity_records.get("proteins", [])
        ]
        protein_id_by_key = {
            str(rec.get("key")): int(rec["id"])
            for rec in self.entity_records.get("proteins", [])
            if rec.get("key")
        }
        protein_id_by_name = {
            _normalize_key(str(rec.get("name"))): int(rec["id"])
            for rec in self.entity_records.get("proteins", [])
            if rec.get("name")
        }
        protein_id_by_db_id: Dict[int, int] = {}
        for rec in self.entity_records.get("proteins", []):
            for db_key in ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id"]:
                db_id = _to_positive_int(rec.get(db_key))
                if db_id is not None:
                    protein_id_by_db_id[db_id] = int(rec["id"])
        protein_complex_items: List[Dict[str, Any]] = []
        for rec in self.entity_records.get("protein-complexes", []):
            members = self._protein_complex_members(
                rec.get("components"),
                complex_name=str(rec.get("name") or rec.get("id")),
                protein_id_by_key=protein_id_by_key,
                protein_id_by_name=protein_id_by_name,
                protein_id_by_db_id=protein_id_by_db_id,
            )
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
        first_species_id = (
            self._ir_pathway_species_id
            if is_pwml_ir(self.extraction)
            else self._resolve_ref_id(None, "species", fallback=True)
        )
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


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pwml_pipeline_export(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    ir, ir_report = build_pwml_ir(
        payload,
        pathway_name=args.name,
        pathway_subject=args.subject,
        strict_db=not args.non_strict_db,
        width=args.width,
        height=args.height,
    )
    ir.setdefault("pathway", {})["description"] = args.description
    ir_validation = validate_pwml_ir(ir)

    ir_path = out_dir / "final.pwml_ir.json"
    ir_report_path = out_dir / "pwml_ir_report.json"
    ir_validation_path = out_dir / "pwml_ir_validation_report.json"
    pwml_path = out_dir / "pathway.pwml"
    validation_report_path = out_dir / "pwml_validation_report.json"
    qa_report_path = out_dir / "pwml_qa_report.json"

    _write_json(ir_path, ir)
    _write_json(ir_report_path, ir_report)
    _write_json(ir_validation_path, ir_validation)

    if ir_report.get("errors") or ir_validation.get("errors"):
        return {
            "ok": False,
            "pwml_ir": str(ir_path),
            "pwml_ir_report": str(ir_report_path),
            "pwml_ir_validation_report": str(ir_validation_path),
            "error": "PWML IR validation failed.",
        }

    ref_path = Path(args.ref)
    signature = discover_structure_signature(ref_path)
    writer_args = SimpleNamespace(
        name=args.name,
        description=args.description,
        subject=args.subject,
        pw_id="PW000000",
        height=args.height,
        width=args.width,
        background_color=args.background_color,
        ref=str(ref_path),
    )
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=writer_args)
    build_result = builder.build()
    repaired = repair_tree(etree.ElementTree(build_result.root), signature)
    validation_report = validate_generated_tree(repaired, signature)
    xml_bytes = etree.tostring(
        repaired.getroot(),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
    )
    qa_report = run_pwml_qa(xml_bytes)

    pwml_path.write_bytes(xml_bytes)
    _write_json(validation_report_path, validation_report)
    _write_json(qa_report_path, qa_report)

    return {
        "ok": bool(validation_report.get("ok")) and bool(qa_report.get("ok")),
        "pwml_ir": str(ir_path),
        "pwml_ir_report": str(ir_report_path),
        "pwml_ir_validation_report": str(ir_validation_path),
        "pwml_file": str(pwml_path),
        "pwml_validation_report": str(validation_report_path),
        "pwml_qa_report": str(qa_report_path),
        "counts": build_result.counts,
        "pwml_validation_issue_count": validation_report.get("issue_count", 0),
        "pwml_qa_ok": qa_report.get("ok"),
    }


def build_pwml_pipeline_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PWML-first converter: mapped final JSON -> PWML IR -> PWML.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input final mapped JSON path")
    parser.add_argument("--out-dir", default="outputs", help="Output directory for PWML artifacts")
    parser.add_argument("--ref", default=str(PROJECT_ROOT / "reference" / "PW000001.pwml"), help="Reference PWML file")
    parser.add_argument("--name", default="Generated Pathway", help="Pathway name")
    parser.add_argument("--subject", default="Metabolic", help="Pathway subject")
    parser.add_argument("--description", default="", help="Pathway description")
    parser.add_argument("--width", type=int, default=3200, help="PWML canvas width")
    parser.add_argument("--height", type=int, default=1400, help="PWML canvas height")
    parser.add_argument("--background-color", default="#FFFFFF", help="PWML background color")
    parser.add_argument("--non-strict-db", action="store_true", help="Warn instead of erroring on missing DB identities")
    return parser


def pwml_pipeline_cli_main(argv: Optional[List[str]] = None) -> None:
    parser = build_pwml_pipeline_arg_parser()
    args = parser.parse_args(argv)
    summary = run_pwml_pipeline_export(args)
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


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
