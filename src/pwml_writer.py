from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        self.ids = IdFactory(1)

        entities = extraction.get("entities", {}) if isinstance(extraction, dict) else {}
        self.entities = entities if isinstance(entities, dict) else {}
        self.processes = extraction.get("processes", {}) if isinstance(extraction, dict) else {}
        if not isinstance(self.processes, dict):
            self.processes = {}

        self.entity_records: Dict[str, List[Dict[str, Any]]] = {}
        self.entity_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.element_lookup: Dict[str, Tuple[str, int]] = {}

        self.section_items: Dict[str, List[Dict[str, Any]]] = {}

        self.pathway_id_int = args.named_for_id if args.named_for_id > 0 else 1
        self.pathway_visualization_id_int = self.pathway_id_int
        self.pathway_visualization_id = f"PathwayVisualization{self.pathway_visualization_id_int}"
        self.pathway_visualization_context_id = f"PathwayVisualizationContext{self.pathway_visualization_id_int}"

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

        return states, int(states[0]["id"])

    def _build_reactions(self) -> List[Dict[str, Any]]:
        reactions_raw = _as_process_list(self.processes, "reactions")
        out: List[Dict[str, Any]] = []
        for raw in reactions_raw:
            rid = self.ids.next()
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
            for enzyme in raw.get("enzymes", []) if isinstance(raw.get("enzymes"), list) else []:
                if not isinstance(enzyme, dict):
                    continue
                pc_name = (
                    str(enzyme.get("protein_complex") or enzyme.get("protein-complex") or "").strip()
                )
                if not pc_name:
                    continue
                pc = self.entity_lookup.get("protein_complexes", {}).get(_normalize_key(pc_name))
                if not pc:
                    continue
                enzymes.append(
                    {
                        "id": self.ids.next(),
                        "protein-complex-id": int(pc["id"]),
                        "enzyme-class": str(enzyme.get("enzyme_class") or "").strip(),
                    }
                )

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
                            "left-biological-state-id": default_state_id,
                            "right-biological-state-id": default_state_id,
                            "direction": "Right",
                        }
                    )
            out.append(
                {
                    "id": tid,
                    "pwt-id": f"PW_T{tid:06d}",
                    "transport-type": None,
                    "transport-elements": elements,
                    "transport-transporters": [],
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

    def _build_locations_and_visualizations(
        self,
        default_state_id: int,
        reactions: List[Dict[str, Any]],
        transports: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        compound_locations: List[Dict[str, Any]] = []
        element_collection_locations: List[Dict[str, Any]] = []
        nucleic_acid_locations: List[Dict[str, Any]] = []
        protein_locations: List[Dict[str, Any]] = []
        protein_complex_visualizations: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        reaction_visualizations: List[Dict[str, Any]] = []
        transport_visualizations: List[Dict[str, Any]] = []

        compound_pos = _grid_positions(
            len(self.entity_records.get("compounds", [])),
            start_x=120,
            start_y=210,
            dx=250,
            dy=170,
            max_cols=6,
        )
        element_collection_pos = _grid_positions(
            len(self.entity_records.get("element-collections", [])),
            start_x=120,
            start_y=390,
            dx=260,
            dy=180,
            max_cols=5,
        )
        nucleic_acid_pos = _grid_positions(
            len(self.entity_records.get("nucleic-acids", [])),
            start_x=120,
            start_y=620,
            dx=260,
            dy=180,
            max_cols=5,
        )
        protein_pos = _grid_positions(
            len(self.entity_records.get("proteins", [])),
            start_x=120,
            start_y=840,
            dx=250,
            dy=170,
            max_cols=6,
        )

        compound_loc_by_id: Dict[int, Dict[str, Any]] = {}
        element_collection_loc_by_id: Dict[int, Dict[str, Any]] = {}
        nucleic_acid_loc_by_id: Dict[int, Dict[str, Any]] = {}
        protein_loc_by_id: Dict[int, Dict[str, Any]] = {}
        pc_vis_by_pc_id: Dict[int, Dict[str, Any]] = {}

        for rec, (x, y) in zip(self.entity_records.get("compounds", []), compound_pos):
            loc = {
                "id": self.ids.next(),
                "compound-id": int(rec["id"]),
                "biological-state-id": default_state_id,
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

        for rec, (x, y) in zip(self.entity_records.get("element-collections", []), element_collection_pos):
            loc = {
                "id": self.ids.next(),
                "element-collection-id": int(rec["id"]),
                "visualization-template-id": 0,
                "biological-state-id": default_state_id,
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

        for rec, (x, y) in zip(self.entity_records.get("nucleic-acids", []), nucleic_acid_pos):
            loc = {
                "id": self.ids.next(),
                "nucleic-acid-id": int(rec["id"]),
                "biological-state-id": default_state_id,
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

        for rec, (x, y) in zip(self.entity_records.get("proteins", []), protein_pos):
            loc = {
                "id": self.ids.next(),
                "protein-id": int(rec["id"]),
                "biological-state-id": default_state_id,
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
            visualization = {
                "id": self.ids.next(),
                "protein-complex-id": int(rec["id"]),
                "pathway-visualization-id": self.pathway_visualization_id_int,
                "biological-state-id": default_state_id,
                "protein_complex_protein_visualizations": [],
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

        reaction_centers = _grid_positions(
            len(reactions), start_x=260, start_y=540, dx=260, dy=220, max_cols=6
        )
        for reaction, (rx, ry) in zip(reactions, reaction_centers):
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
                    edges.append(
                        {
                            "id": edge_id,
                            "path": path,
                            "visualization-template-id": 0,
                            "hidden": False,
                            "zindex": 18,
                        }
                    )
                    if etype == "Compound":
                        reaction_compound_visualizations.append(
                            {
                                "id": self.ids.next(),
                                "compound-location-id": location_id,
                                "edge-id": edge_id,
                                "side": side,
                            }
                        )
                    elif etype == "ElementCollection":
                        reaction_element_collection_visualizations.append(
                            {
                                "id": self.ids.next(),
                                "element-collection-location-id": location_id,
                                "edge-id": edge_id,
                                "side": side,
                            }
                        )

            for enzyme in reaction.get("reaction-enzymes", []) if isinstance(reaction.get("reaction-enzymes"), list) else []:
                pc_id = int(enzyme.get("protein-complex-id") or 0)
                pc_vis = pc_vis_by_pc_id.get(pc_id)
                if not pc_vis:
                    continue
                reaction_enzyme_visualizations.append(
                    {
                        "id": self.ids.next(),
                        "reaction-enzyme-id": int(enzyme["id"]),
                        "protein-complex-visualization-id": int(pc_vis["id"]),
                    }
                )

            reaction_visualizations.append(
                {
                    "id": self.ids.next(),
                    "pathway-visualization-id": self.pathway_visualization_id_int,
                    "reaction-id": int(reaction["id"]),
                    "biological-state-id": default_state_id,
                    "reaction_compound_visualizations": reaction_compound_visualizations,
                    "reaction_element_collection_visualizations": reaction_element_collection_visualizations,
                    "reaction_enzyme_visualizations": reaction_enzyme_visualizations,
                }
            )

        transport_centers = _grid_positions(
            len(transports), start_x=260, start_y=760, dx=280, dy=220, max_cols=6
        )
        for transport, _ in zip(transports, transport_centers):
            transport_visualizations.append(
                {
                    "id": self.ids.next(),
                    "transport-id": int(transport["id"]),
                    "pathway-visualization-id": self.pathway_visualization_id_int,
                    "transport_compound_visualizations": [],
                    "transport_transporter_visualizations": [],
                }
            )

        return {
            "compound-locations": compound_locations,
            "element-collection-locations": element_collection_locations,
            "nucleic-acid-locations": nucleic_acid_locations,
            "protein-locations": protein_locations,
            "protein-complex-visualizations": protein_complex_visualizations,
            "edges": edges,
            "reaction-visualizations": reaction_visualizations,
            "transport-visualizations": transport_visualizations,
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
            }
            for rec in self.entity_records.get("proteins", [])
        ]
        self.section_items["protein-complexes"] = [
            {
                "id": int(rec["id"]),
                "name": rec["name"],
                "species-id": default_species_id,
                "pwp-id": f"PW_P{int(rec['id']):06d}",
                "protein_complex-proteins": [],
                "element-states": [],
            }
            for rec in self.entity_records.get("protein-complexes", [])
        ]

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
        self.section_items["membrane-visualizations"] = []
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
                node.text = str(self.args.named_for_id)
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
