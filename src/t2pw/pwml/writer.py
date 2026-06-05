from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from lxml import etree

from t2pw.paths import PROJECT_ROOT
from t2pw.pwml.compound_templates import TEMPLATE_DIMS, select_compound_template_id
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

OPTION_TAG_NS = "urn:pathwhiz-option"
ARROW_ANGLE = math.pi / 6
ARROW_LENGTH = 30


def _format_path_number(value: float) -> str:
    return f"{value:.12g}"


def _curved_edge_path(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[str, int, int, int, int]:
    cp1x = x1 + (x2 - x1) // 3
    cp1y = y1 + (y2 - y1) // 3
    cp2x = x1 + 2 * (x2 - x1) // 3
    cp2y = y1 + 2 * (y2 - y1) // 3
    path = f"M{x1} {y1} C{cp1x} {cp1y} {cp2x} {cp2y} {x2} {y2}"
    return path, cp1x, cp1y, cp2x, cp2y


def _compute_arrow_path(tip_x: int, tip_y: int, from_x: int, from_y: int) -> Optional[str]:
    dx = from_x - tip_x
    dy = from_y - tip_y
    distance = math.sqrt(dx**2 + dy**2)
    if distance == 0:
        return None
    r = ARROW_LENGTH / distance
    x4 = (math.cos(ARROW_ANGLE) * dx - math.sin(ARROW_ANGLE) * dy) * r + tip_x
    y4 = (math.sin(ARROW_ANGLE) * dx + math.cos(ARROW_ANGLE) * dy) * r + tip_y
    x5 = (math.cos(-ARROW_ANGLE) * dx - math.sin(-ARROW_ANGLE) * dy) * r + tip_x
    y5 = (math.sin(-ARROW_ANGLE) * dx + math.cos(-ARROW_ANGLE) * dy) * r + tip_y
    return (
        f"M {_format_path_number(x5)} {_format_path_number(y5)} "
        f"L {tip_x} {tip_y} "
        f"L {_format_path_number(x4)} {_format_path_number(y4)}"
    )


def _add_start_arrow(edge: Dict[str, Any], x1: int, y1: int, cp1x: int, cp1y: int) -> None:
    edge["option:start_arrow"] = True
    arrow_path = _compute_arrow_path(x1, y1, cp1x, cp1y)
    if arrow_path is not None:
        edge["option:start_arrow_path"] = arrow_path


def _add_end_arrow(edge: Dict[str, Any], x2: int, y2: int, cp2x: int, cp2y: int) -> None:
    edge["option:end_arrow"] = True
    arrow_path = _compute_arrow_path(x2, y2, cp2x, cp2y)
    if arrow_path is not None:
        edge["option:end_arrow_path"] = arrow_path


def is_non_blocking_pwml_ir_error(issue: Any) -> bool:
    if not isinstance(issue, dict):
        return False
    return (
        issue.get("code") == "compound_db_resolution_failed"
        and issue.get("entity_type") == "compound"
    )


def blocking_pwml_ir_errors(ir_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        issue
        for issue in ir_report.get("errors", [])
        if not is_non_blocking_pwml_ir_error(issue)
    ]


def _assert_reaction_member_anchor(
    loc: Dict[str, Any],
    side: str,
    expected_anchor_x: int,
    expected_anchor_y: int,
) -> None:
    actual_anchor_x = int(loc["x"]) + int(loc["width"]) if side == "Left" else int(loc["x"])
    actual_anchor_y = int(loc["y"]) + int(loc["height"]) // 2
    assert actual_anchor_x == expected_anchor_x
    assert abs(actual_anchor_y - expected_anchor_y) <= 1


def _packed_reaction_stack_tops(
    heights: Sequence[int],
    enzyme_cy: int,
    gap_y: int,
) -> Tuple[int, List[int]]:
    if not heights:
        return 0, []
    total_stack_height = sum(heights) + gap_y * (len(heights) - 1)
    current_top = int(round(enzyme_cy - total_stack_height / 2))
    tops: List[int] = []
    for height in heights:
        tops.append(current_top)
        current_top += height + gap_y
    return total_stack_height, tops


_CURRENCY_COMPOUND_NAMES = {
    "4a carbinolamine tetrahydrobiopterin",
    "4a hydroxy tetrahydrobiopterin",
    "4a hydroxytetrahydrobiopterin",
    "4a tetrahydrobiopterin",
    "acetyl coenzyme a",
    "acetyl coa",
    "adp",
    "adenosine diphosphate",
    "adenosine monophosphate",
    "adenosine triphosphate",
    "amp",
    "ammonia",
    "ammonium",
    "atp",
    "bh4",
    "carbon dioxide",
    "coa",
    "co2",
    "coenzyme a",
    "dihydrobiopterin",
    "fad",
    "fadh2",
    "fmn",
    "fmnh2",
    "h",
    "h+",
    "h2o",
    "h2o2",
    "hydrogen ion",
    "hydrogen peroxide",
    "nad",
    "nad+",
    "nadh",
    "nadp",
    "nadp+",
    "nadph",
    "nh3",
    "nh4+",
    "o2",
    "oxygen",
    "phosphate",
    "pi",
    "ppi",
    "pyrophosphate",
    "tetrahydrobiopterin",
    "water",
}

_CURRENCY_COMPOUND_KEGG_IDS = {
    "C00001",  # H2O
    "C00002",  # ATP
    "C00003",  # NAD+
    "C00004",  # NADH
    "C00005",  # NADPH
    "C00006",  # NADP+
    "C00007",  # O2
    "C00008",  # ADP
    "C00009",  # phosphate
    "C00010",  # CoA
    "C00013",  # pyrophosphate
    "C00014",  # NH3
    "C00016",  # FAD
    "C00020",  # AMP
    "C00027",  # H2O2
    "C00080",  # H+
    "C00255",  # FMN
    "C01352",  # FADH2
    "C00272",  # tetrahydrobiopterin
}

_CURRENCY_COMPOUND_CHEBI_IDS = {
    "15377",  # water
    "15378",  # hydrogen ion
    "15379",  # oxygen
    "15380",  # carbon dioxide
    "15347",  # acetyl-CoA
    "15422",  # ATP
    "15635",  # phosphate
    "15713",  # NADH
    "15846",  # NAD+
    "16027",  # AMP
    "16240",  # hydrogen peroxide
    "16474",  # NADPH
    "16761",  # ADP
    "17621",  # FAD
    "17877",  # pyrophosphate
    "18420",  # magnesium(2+) common currency ion
    "24636",  # proton
    "26523",  # reactive oxygen species-ish, conservative
    "29950",  # ammonia
    "30616",  # tetrahydrobiopterin
    "57287",  # CoA
    "58349",  # NADP+
}


def _currency_name_token(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("coenzyme-a", "coenzyme a").replace("co-a", "coa")
    text = text.replace("β", "beta")
    return re.sub(r"[^a-z0-9+]+", " ", text).strip()


def _is_currency_compound_record(record: Dict[str, Any]) -> bool:
    names = {
        _currency_name_token(record.get("name")),
        _currency_name_token(record.get("short-name")),
        _currency_name_token(record.get("short_name")),
    }
    names.discard("")
    if names & _CURRENCY_COMPOUND_NAMES:
        return True
    for token in names:
        if token.endswith(" coa") or token.endswith(" coenzyme a") or token.startswith("coa "):
            return True
        if "tetrahydrobiopterin" in token or "biopterin" in token:
            return True
    kegg_id = str(record.get("kegg-id") or record.get("kegg_id") or "").strip().upper()
    if kegg_id in _CURRENCY_COMPOUND_KEGG_IDS:
        return True
    chebi_id = str(record.get("chebi-id") or record.get("chebi_id") or "").replace("CHEBI:", "").strip()
    return chebi_id in _CURRENCY_COMPOUND_CHEBI_IDS


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
        "reaction-coupled-transport-visualizations": "reaction-coupled-transport-visualization",
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


def _nonempty_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_trusted_compound_record(record: Dict[str, Any], db_row: Dict[str, Any]) -> bool:
    if db_row:
        return True
    if str(record.get("db_status") or "").strip().casefold() == "matched":
        return True
    db_match = record.get("db_match") if isinstance(record.get("db_match"), dict) else {}
    return str(db_match.get("status") or "").strip().casefold() == "matched"


def _trusted_compound_pwc_id(
    record: Dict[str, Any],
    db_row: Dict[str, Any],
    mapped_ids: Dict[str, Any],
) -> Optional[str]:
    if not _is_trusted_compound_record(record, db_row):
        return None
    return _nonempty_text(db_row.get("pwc_id") or record.get("pwc_id") or mapped_ids.get("pwc_id"))


def _trusted_compound_short_name(record: Dict[str, Any], db_row: Dict[str, Any]) -> Optional[str]:
    if not _is_trusted_compound_record(record, db_row):
        return None
    return _nonempty_text(db_row.get("short_name") or record.get("short_name"))


_TRUSTED_BIOLOGICAL_STATE_STATUSES = {
    "matched",
    "mapped",
    "resolved",
    "verified",
    "db_matched",
}


def _trusted_status(value: Any) -> bool:
    return str(value or "").strip().casefold() in _TRUSTED_BIOLOGICAL_STATE_STATUSES


def _format_pwbs_id(value: Any) -> Optional[str]:
    text = _nonempty_text(value)
    if text is None:
        return None
    if text.casefold().startswith("pw_bs"):
        return text
    parsed = _to_positive_int(text)
    if parsed is not None:
        return f"PW_BS{parsed:06d}"
    return text


def _first_pwbs_id(container: Dict[str, Any]) -> Optional[str]:
    for key in [
        "pwbs_id",
        "pwbs-id",
        "pathwhiz_biological_state_pwbs_id",
        "pathbank_biological_state_pwbs_id",
    ]:
        pwbs_id = _format_pwbs_id(container.get(key))
        if pwbs_id is not None:
            return pwbs_id
    return None


def _first_biological_state_db_id(container: Dict[str, Any], *, allow_plain_id: bool = False) -> Optional[int]:
    keys = [
        "pathbank_biological_state_id",
        "pw_biological_state_id",
        "pathwhiz_biological_state_id",
        "db_id",
    ]
    if allow_plain_id:
        keys.append("id")
    for key in keys:
        db_id = _to_positive_int(container.get(key))
        if db_id is not None:
            return db_id
    return None


def _trusted_biological_state_pwbs_id(record: Dict[str, Any]) -> Optional[str]:
    db_row = record.get("db_row") if isinstance(record.get("db_row"), dict) else {}
    mapping_meta = record.get("mapping_meta") if isinstance(record.get("mapping_meta"), dict) else {}
    mapped_ids = record.get("mapped_ids") if isinstance(record.get("mapped_ids"), dict) else {}
    db_match = record.get("db_match") if isinstance(record.get("db_match"), dict) else {}
    chosen = db_match.get("chosen") if isinstance(db_match.get("chosen"), dict) else {}
    meta_chosen = mapping_meta.get("chosen") if isinstance(mapping_meta.get("chosen"), dict) else {}

    trusted = bool(db_row) or any(
        _trusted_status(container.get("status") or container.get("db_status"))
        for container in [record, mapping_meta, mapped_ids, db_match]
    )
    if not trusted:
        return None

    for container in [db_row, chosen, meta_chosen, record, mapping_meta, mapped_ids]:
        pwbs_id = _first_pwbs_id(container)
        if pwbs_id is not None:
            return pwbs_id

    for container, allow_plain_id in [
        (db_row, True),
        (chosen, True),
        (meta_chosen, True),
        (record, False),
        (mapping_meta, False),
        (mapped_ids, False),
    ]:
        db_id = _first_biological_state_db_id(container, allow_plain_id=allow_plain_id)
        if db_id is not None:
            return f"PW_BS{db_id:06d}"

    return None


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
        self.layout_debug_counts: Dict[str, int] = {
            "shared_intermediates_detected": 0,
            "shared_intermediates_skipped_cofactor": 0,
            "shared_intermediate_locations_reused": 0,
        }
        self.layout_debug_stacks: List[Dict[str, Any]] = []
        self.layout_debug_shared_intermediates: List[Dict[str, Any]] = []

        self.pathway_id_int = 1
        self.pathway_visualization_id_int = self.pathway_id_int
        self.pathway_visualization_id = f"PathwayVisualization{self.pathway_visualization_id_int}"
        self.pathway_visualization_context_id = f"PathwayVisualizationContext{self.pathway_visualization_id_int}"
        self._state_id_map: Dict[str, int] = {}
        self._ir_key_ids: Dict[str, Dict[str, int]] = {}
        self._ir_entity_info: Dict[str, Dict[str, Any]] = {}
        self._ir_pathway_species_id: Optional[int] = None

    def _make_compound_identity_fields_optional(self) -> None:
        compound_sig = self.signature.sections.get("compounds")
        if compound_sig is None:
            return
        # Reference PWML includes these fields, but Rails must receive them
        # absent for novel compounds so it can allocate safe DB-backed values.
        compound_sig.required_fields = [
            field
            for field in compound_sig.required_fields
            if field not in {"pwc-id", "short-name"}
        ]

    def _make_biological_state_identity_fields_optional(self) -> None:
        biological_state_sig = self.signature.sections.get("biological-states")
        if biological_state_sig is None:
            return
        # pwbs-id is a global database identity, not a local XML reference.
        # Generated states must omit it so Rails can resolve/create by context.
        biological_state_sig.required_fields = [
            field
            for field in biological_state_sig.required_fields
            if field != "pwbs-id"
        ]

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
            }
            pwbs_id = _trusted_biological_state_pwbs_id(record)
            if pwbs_id is not None:
                state["pwbs-id"] = pwbs_id
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
                entity_name = str(
                    t.get("entity")
                    or t.get("protein")
                    or t.get("protein_complex")
                    or t.get("protein-complex")
                    or ""
                ).strip()
                if not entity_name:
                    continue
                entity_type = str(t.get("entity_type") or t.get("type") or "").strip().lower()
                pc = self.entity_lookup.get("protein_complexes", {}).get(_normalize_key(entity_name))
                if entity_type == "protein_complex" and pc:
                    transporters.append({"id": self.ids.next(), "protein-complex-id": int(pc["id"])})
                    continue
                prot = self.entity_lookup.get("proteins", {}).get(_normalize_key(entity_name))
                if prot:
                    transporters.append({"id": self.ids.next(), "protein-id": int(prot["id"])})
                    continue
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
        substrate_gap = 46
        product_gap = 51
        node_spacing_y = 220
        compound_gap_y = 30
        rxn_step_x = 800
        protein_w, protein_h = 150, 70
        protein_gap_x = 10

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
        compound_loc_by_rxn: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        compound_loc_by_state: Dict[Tuple[int, int], Dict[str, Any]] = {}
        element_collection_loc_by_id: Dict[int, Dict[str, Any]] = {}
        element_collection_loc_by_rxn: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        nucleic_acid_loc_by_id: Dict[int, Dict[str, Any]] = {}
        nucleic_acid_loc_by_rxn: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        protein_loc_by_id: Dict[int, Dict[str, Any]] = {}
        pc_vis_by_pc_id: Dict[int, Dict[str, Any]] = {}
        transport_layouts: Dict[int, Dict[str, int]] = {}

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

        raw_reactions = _as_process_list(self.processes, "reactions")
        reaction_bs_ids: Dict[int, int] = {}
        reaction_indices_by_bs: Dict[int, List[int]] = defaultdict(list)
        for idx, _reaction in enumerate(reactions):
            raw_rx = raw_reactions[idx] if idx < len(raw_reactions) else {}
            bs_name = str(raw_rx.get("biological_state", "")).strip()
            bs_id = self._state_id_map.get(bs_name.casefold(), default_state_id)
            reaction_bs_ids[idx] = bs_id
            reaction_indices_by_bs[bs_id].append(idx)

        reaction_layouts: Dict[int, Dict[str, int]] = {}
        for bs_id, reaction_indices in sorted(reaction_indices_by_bs.items()):
            region = region_for(bs_id)
            n_rxns = len(reaction_indices)
            enzyme_cx_base = region["x"] + region["w"] // 2 - (n_rxns - 1) * rxn_step_x // 2
            enzyme_cy = max(region["y"] + region["h"] // 2, 360)
            for k, reaction_idx in enumerate(reaction_indices):
                enzyme_cx = enzyme_cx_base + k * rxn_step_x
                enzyme_x = enzyme_cx - protein_w // 2
                enzyme_y = enzyme_cy - protein_h // 2
                reaction_layouts[reaction_idx] = {
                    "biological-state-id": bs_id,
                    "enzyme-cx": enzyme_cx,
                    "enzyme-cy": enzyme_cy,
                    "enzyme-x": enzyme_x,
                    "enzyme-y": enzyme_y,
                    "compound-stack-cy": enzyme_cy,
                }

        # Compound locations — left half of each compartment region
        compound_rec_by_id = {
            int(rec["id"]): rec
            for rec in self.entity_records.get("compounds", [])
            if rec.get("id") is not None
        }

        def add_compound_location(compound_id: int, bs_id: int, x: int, y: int) -> Dict[str, Any]:
            rec = compound_rec_by_id.get(compound_id, {"id": compound_id})
            template_id = select_compound_template_id(rec)
            width, height = TEMPLATE_DIMS.get(template_id, TEMPLATE_DIMS[3])
            loc = {
                "id": self.ids.next(),
                "compound-id": compound_id,
                "biological-state-id": bs_id,
                "visualization-template-id": template_id,
                "hidden": False,
                "x": x,
                "y": y,
                "zindex": 10,
                "font-size": "regular",
                "width": str(width),
                "height": str(height),
            }
            compound_locations.append(loc)
            compound_loc_by_state[(compound_id, bs_id)] = loc
            compound_loc_by_id.setdefault(compound_id, loc)
            return loc

        def ensure_compound_location(compound_id: int, bs_id: int) -> Dict[str, Any]:
            loc = compound_loc_by_state.get((compound_id, bs_id))
            if loc:
                return loc
            region = region_for(bs_id)
            return add_compound_location(compound_id, bs_id, region["x"] + pad, region["y"] + pad)

        def add_element_collection_location(
            element_collection_id: int,
            bs_id: int,
            x: int,
            y: int,
        ) -> Dict[str, Any]:
            width, height = TEMPLATE_DIMS[81]
            loc = {
                "id": self.ids.next(),
                "element-collection-id": element_collection_id,
                "visualization-template-id": 0,
                "biological-state-id": bs_id,
                "hidden": False,
                "x": x,
                "y": y,
                "zindex": 10,
                "font-size": "regular",
                "width": str(width),
                "height": str(height),
            }
            element_collection_locations.append(loc)
            element_collection_loc_by_id.setdefault(element_collection_id, loc)
            return loc

        def add_nucleic_acid_location(
            nucleic_acid_id: int,
            bs_id: int,
            x: int,
            y: int,
        ) -> Dict[str, Any]:
            width, height = TEMPLATE_DIMS[81]
            loc = {
                "id": self.ids.next(),
                "nucleic-acid-id": nucleic_acid_id,
                "biological-state-id": bs_id,
                "visualization-template-id": 0,
                "hidden": False,
                "x": x,
                "y": y,
                "zindex": 10,
                "font-size": "regular",
                "width": str(width),
                "height": str(height),
            }
            nucleic_acid_locations.append(loc)
            nucleic_acid_loc_by_id.setdefault(nucleic_acid_id, loc)
            return loc

        reaction_compound_ids: set[int] = set()
        for reaction in reactions:
            for side_key in ["reaction-left-elements", "reaction-right-elements"]:
                elements = reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []
                for rel in elements:
                    if not isinstance(rel, dict):
                        continue
                    if str(rel.get("element-type") or "").casefold() != "compound":
                        continue
                    compound_id = _to_positive_int(rel.get("element-id"))
                    if compound_id is not None:
                        reaction_compound_ids.add(compound_id)

        total_compound_count = len(self.entity_records.get("compounds", []))
        grid_placed_compound_count = 0
        for bs_id, group_recs in sorted(group_by_bs("compounds").items()):
            region = region_for(bs_id)
            grid_recs = [
                rec
                for rec in group_recs
                if _to_positive_int(rec.get("id")) not in reaction_compound_ids
            ]
            grid_placed_compound_count += len(grid_recs)
            for rec, (x, y) in zip(grid_recs, sub_grid_left(region, len(grid_recs))):
                loc = add_compound_location(int(rec["id"]), bs_id, x, y)
                compound_loc_by_id[int(rec["id"])] = loc
        self.layout_debug_counts.update(
            {
                "compound_total": total_compound_count,
                "compound_grid_skipped_reaction_used": len(reaction_compound_ids),
                "compound_grid_placed_non_reaction": grid_placed_compound_count,
            }
        )

        # Element-collection locations — left half of each compartment region
        for bs_id, group_recs in sorted(group_by_bs("element-collections").items()):
            region = region_for(bs_id)
            for rec, (x, y) in zip(group_recs, sub_grid_left(region, len(group_recs))):
                loc = add_element_collection_location(int(rec["id"]), bs_id, x, y)
                element_collection_loc_by_id[int(rec["id"])] = loc

        # Nucleic-acid locations — left half of each compartment region
        for bs_id, group_recs in sorted(group_by_bs("nucleic-acids").items()):
            region = region_for(bs_id)
            for rec, (x, y) in zip(group_recs, sub_grid_left(region, len(group_recs))):
                loc = add_nucleic_acid_location(int(rec["id"]), bs_id, x, y)
                nucleic_acid_loc_by_id[int(rec["id"])] = loc

        # Protein locations: reaction enzymes sit at reaction-center positions.
        def protein_complex_component_ids(protein_complex_id: int) -> List[int]:
            rec = next(
                (
                    pc
                    for pc in self.entity_records.get("protein-complexes", [])
                    if int(pc.get("id") or 0) == protein_complex_id
                ),
                None,
            )
            if not rec:
                return []

            protein_ids: List[int] = []
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
                    protein_id = int(prot["id"])
                    if protein_id not in protein_ids:
                        protein_ids.append(protein_id)
            return protein_ids

        def reaction_enzyme_protein_ids(reaction: Dict[str, Any]) -> List[int]:
            protein_ids: List[int] = []
            for enzyme in reaction.get("reaction-enzymes", []) if isinstance(reaction.get("reaction-enzymes"), list) else []:
                pc_id = enzyme.get("protein-complex-id")
                prot_id = enzyme.get("protein-id")
                if prot_id is not None:
                    protein_id = int(prot_id)
                    if protein_id not in protein_ids:
                        protein_ids.append(protein_id)
                elif pc_id is not None:
                    for protein_id in protein_complex_component_ids(int(pc_id)):
                        if protein_id not in protein_ids:
                            protein_ids.append(protein_id)
            return protein_ids

        def add_protein_location(protein_id: int, bs_id: int, x: int, y: int, label_type: str = "text") -> None:
            if protein_id in protein_loc_by_id:
                return
            loc = {
                "id": self.ids.next(),
                "protein-id": protein_id,
                "biological-state-id": bs_id,
                "visualization-template-id": 0,
                "hidden": False,
                "x": x,
                "y": y,
                "zindex": 10,
                "label-type": label_type,
                "font-size": "regular",
                "width": str(protein_w),
                "height": str(protein_h),
            }
            protein_locations.append(loc)
            protein_loc_by_id[protein_id] = loc

        def place_protein_location(
            protein_id: int,
            bs_id: int,
            x: int,
            y: int,
            label_type: str = "text",
        ) -> Optional[Dict[str, Any]]:
            loc = protein_loc_by_id.get(protein_id)
            if loc:
                loc["biological-state-id"] = bs_id
                loc["x"] = x
                loc["y"] = y
                loc["label-type"] = label_type
                return loc
            add_protein_location(protein_id, bs_id, x, y, label_type)
            return protein_loc_by_id.get(protein_id)

        # Protein locations: enzyme proteins sit at reaction-center positions.
        for reaction_idx, reaction in enumerate(reactions):
            layout = reaction_layouts.get(reaction_idx)
            if not layout:
                continue
            enzyme_protein_ids = reaction_enzyme_protein_ids(reaction)
            if not enzyme_protein_ids:
                continue
            total_w = len(enzyme_protein_ids) * protein_w + (len(enzyme_protein_ids) - 1) * protein_gap_x
            x0 = layout["enzyme-cx"] - total_w // 2
            for j, protein_id in enumerate(enzyme_protein_ids):
                add_protein_location(
                    protein_id,
                    int(layout["biological-state-id"]),
                    x0 + j * (protein_w + protein_gap_x),
                    layout["enzyme-y"],
                    "subunit" if len(enzyme_protein_ids) > 1 else "text",
                )

        # Preserve locations for non-enzyme proteins that may still be referenced elsewhere.
        for bs_id, group_recs in sorted(group_by_bs("proteins").items()):
            region = region_for(bs_id)
            x = region["x"] + region["w"] - pad - protein_w
            y = region["y"] + pad
            for rec in group_recs:
                protein_id = int(rec["id"])
                if protein_id in protein_loc_by_id:
                    continue
                add_protein_location(protein_id, bs_id, x, y)
                y += protein_h + protein_gap_x

        def element_dims(element_type: str, element_id: int) -> Tuple[int, int]:
            if element_type == "Compound":
                rec = compound_rec_by_id.get(element_id, {"id": element_id})
                template_id = select_compound_template_id(rec)
                return TEMPLATE_DIMS.get(template_id, TEMPLATE_DIMS[3])
            if element_type in ("ElementCollection", "NucleicAcid"):
                return TEMPLATE_DIMS[81]
            return TEMPLATE_DIMS[3]

        for reaction_idx, reaction in enumerate(reactions):
            layout = reaction_layouts.get(reaction_idx)
            if not layout:
                continue
            bs_id = int(layout["biological-state-id"])
            enzyme_left_x = layout["enzyme-x"]
            enzyme_right_x = enzyme_left_x + protein_w
            compound_stack_cy = layout["compound-stack-cy"]
            for side_key in ["reaction-left-elements", "reaction-right-elements"]:
                side = "Left" if side_key == "reaction-left-elements" else "Right"
                elements = reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []
                stack_members: List[Dict[str, Any]] = []
                for rel in elements:
                    if not isinstance(rel, dict):
                        continue
                    etype = str(rel.get("element-type") or "")
                    eid = int(rel.get("element-id") or 0)
                    if etype not in ("Compound", "ElementCollection", "NucleicAcid"):
                        continue
                    width, height = element_dims(etype, eid)
                    stack_members.append(
                        {
                            "rel": rel,
                            "etype": etype,
                            "eid": eid,
                            "width": width,
                            "height": height,
                        }
                    )
                heights = [int(member["height"]) for member in stack_members]
                total_stack_height, stack_tops = _packed_reaction_stack_tops(
                    heights,
                    compound_stack_cy,
                    compound_gap_y,
                )
                self.layout_debug_stacks.append(
                    {
                        "reaction_idx": reaction_idx,
                        "side": side,
                        "element_ids": [member["eid"] for member in stack_members],
                        "heights": heights,
                        "total_stack_height": total_stack_height,
                        "enzyme_cy": compound_stack_cy,
                        "y_positions": stack_tops,
                    }
                )
                for member, y in zip(stack_members, stack_tops):
                    etype = member["etype"]
                    eid = int(member["eid"])
                    width = int(member["width"])
                    height = int(member["height"])
                    expected_anchor_x = (
                        enzyme_left_x - substrate_gap
                        if side_key == "reaction-left-elements"
                        else enzyme_right_x + product_gap
                    )
                    expected_anchor_y = y + height // 2
                    x = expected_anchor_x - width if side == "Left" else expected_anchor_x
                    if y < 0:
                        import warnings
                        warnings.warn(
                            f"DEBUG layout: reaction_idx={reaction_idx} side={side_key} "
                            f"y={y} (anchor_y={expected_anchor_y} height={height} compound_stack_cy={compound_stack_cy})"
                        )
                    if etype == "Compound":
                        state_key = (eid, bs_id)
                        had_state_loc = state_key in compound_loc_by_state
                        previous_state_loc = compound_loc_by_state.get(state_key)
                        loc = add_compound_location(eid, bs_id, x, y)
                        _assert_reaction_member_anchor(loc, side, expected_anchor_x, expected_anchor_y)
                        if had_state_loc and previous_state_loc is not None:
                            compound_loc_by_state[state_key] = previous_state_loc
                        else:
                            compound_loc_by_state.pop(state_key, None)
                        compound_loc_by_rxn[(eid, reaction_idx, side)] = loc
                    elif etype == "ElementCollection":
                        loc = add_element_collection_location(eid, bs_id, x, y)
                        _assert_reaction_member_anchor(loc, side, expected_anchor_x, expected_anchor_y)
                        element_collection_loc_by_rxn[(eid, reaction_idx, side)] = loc
                    elif etype == "NucleicAcid":
                        loc = add_nucleic_acid_location(eid, bs_id, x, y)
                        _assert_reaction_member_anchor(loc, side, expected_anchor_x, expected_anchor_y)
                        nucleic_acid_loc_by_rxn[(eid, reaction_idx, side)] = loc

        # Transport layout is vertical across compartment boundaries. Cargo gets
        # one location per biological state; transporter proteins sit on the
        # membrane boundary between the source and destination compartments.
        for transport_idx, transport in enumerate(transports):
            elements = transport.get("transport-elements", [])
            if not isinstance(elements, list) or not elements:
                continue

            first_compound: Optional[Dict[str, Any]] = None
            for element in elements:
                if str(element.get("element-type") or "") == "Compound":
                    first_compound = element
                    break
            if not first_compound:
                continue

            source_bs_id = int(first_compound.get("left-biological-state-id") or default_state_id)
            source_region = region_for(source_bs_id)
            membrane_y = int(source_region["y"] + source_region["h"])
            center_x = int(source_region["x"] + source_region["w"] // 2)
            center_x += (transport_idx - (len(transports) - 1) // 2) * rxn_step_x
            center_x = max(source_region["x"] + pad + protein_w // 2, center_x)
            center_x = min(source_region["x"] + source_region["w"] - pad - protein_w // 2, center_x)

            transporter_x = center_x - protein_w // 2
            transporter_y = membrane_y
            transport_layouts[int(transport["id"])] = {
                "x": transporter_x,
                "y": transporter_y,
                "width": protein_w,
                "height": protein_h,
                "center-x": center_x,
            }

            compound_count = sum(1 for element in elements if str(element.get("element-type") or "") == "Compound")
            compound_idx = 0
            for element in elements:
                if str(element.get("element-type") or "") != "Compound":
                    continue
                compound_id = int(element.get("element-id") or 0)
                if not compound_id:
                    continue
                left_bs_id = int(element.get("left-biological-state-id") or default_state_id)
                right_bs_id = int(element.get("right-biological-state-id") or default_state_id)
                cargo_center_x = center_x + (compound_idx - (compound_count - 1) // 2) * node_spacing_y
                source_loc = ensure_compound_location(compound_id, left_bs_id)
                dest_loc = ensure_compound_location(compound_id, right_bs_id)
                source_loc["x"] = cargo_center_x - int(source_loc["width"]) // 2
                source_loc["y"] = membrane_y - int(source_loc["height"]) - pad
                dest_loc["x"] = cargo_center_x - int(dest_loc["width"]) // 2
                dest_loc["y"] = membrane_y + protein_h + pad
                compound_idx += 1

            transporters = transport.get("transport-transporters", [])
            transporter_protein_ids: List[int] = []
            for transporter in transporters if isinstance(transporters, list) else []:
                prot_id = transporter.get("protein-id")
                pc_id = transporter.get("protein-complex-id")
                if prot_id is not None:
                    protein_id = int(prot_id)
                    if protein_id not in transporter_protein_ids:
                        transporter_protein_ids.append(protein_id)
                elif pc_id is not None:
                    for protein_id in protein_complex_component_ids(int(pc_id)):
                        if protein_id not in transporter_protein_ids:
                            transporter_protein_ids.append(protein_id)

            total_w = len(transporter_protein_ids) * protein_w + max(0, len(transporter_protein_ids) - 1) * protein_gap_x
            x0 = center_x - total_w // 2 if transporter_protein_ids else transporter_x
            for j, protein_id in enumerate(transporter_protein_ids):
                place_protein_location(
                    protein_id,
                    source_bs_id,
                    x0 + j * (protein_w + protein_gap_x),
                    transporter_y,
                    "subunit" if len(transporter_protein_ids) > 1 else "text",
                )

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

        def loc_center(loc: Dict[str, Any]) -> Tuple[int, int, int]:
            return (
                int(loc["id"]),
                int(loc["x"]) + int(loc["width"]) // 2,
                int(loc["y"]) + int(loc["height"]) // 2,
            )

        def loc_side_center(loc: Dict[str, Any], side: str) -> Tuple[int, int, int]:
            x = int(loc["x"])
            if side == "Left":
                x += int(loc["width"])
            return (
                int(loc["id"]),
                x,
                int(loc["y"]) + int(loc["height"]) // 2,
            )

        def _rxn_loc(
            by_rxn: Dict,
            by_id: Dict,
            entity_label: str,
            element_id: int,
            side: Optional[str],
            reaction_idx: Optional[int],
        ) -> Optional[Dict[str, Any]]:
            """Return the location dict for a reaction participant.

            When reaction_idx and side are both provided the lookup is strictly
            per-reaction: no fallback to the global by-id map.  A missing entry
            is a hard error so that mis-anchored edges are caught immediately
            rather than silently pointing at the wrong visual instance.

            When called without reaction context (reaction_idx or side absent)
            the by-id map is used, which is the correct path for non-reaction
            lookups (e.g. transport or orphan compound queries).
            """
            if reaction_idx is not None and side:
                loc = by_rxn.get((element_id, reaction_idx, side))
                if loc is None:
                    raise RuntimeError(
                        f"Missing per-reaction {entity_label} location for "
                        f"{entity_label}={element_id}, reaction_idx={reaction_idx}, side={side}"
                    )
                return loc
            return by_id.get(element_id)

        def location_info(
            element_type: str,
            element_id: int,
            side: Optional[str] = None,
            reaction_idx: Optional[int] = None,
        ) -> Optional[Tuple[int, int, int]]:
            if element_type == "Compound":
                loc = _rxn_loc(
                    compound_loc_by_rxn, compound_loc_by_id,
                    "compound", element_id, side, reaction_idx,
                )
                if loc:
                    return loc_side_center(loc, side) if side else loc_center(loc)
            elif element_type == "ElementCollection":
                loc = _rxn_loc(
                    element_collection_loc_by_rxn, element_collection_loc_by_id,
                    "element-collection", element_id, side, reaction_idx,
                )
                if loc:
                    return loc_side_center(loc, side) if side else loc_center(loc)
            elif element_type == "NucleicAcid":
                loc = _rxn_loc(
                    nucleic_acid_loc_by_rxn, nucleic_acid_loc_by_id,
                    "nucleic-acid", element_id, side, reaction_idx,
                )
                if loc:
                    return loc_side_center(loc, side) if side else loc_center(loc)
            elif element_type == "Protein":
                loc = protein_loc_by_id.get(element_id)
                if loc:
                    return loc_side_center(loc, side) if side else loc_center(loc)
            return None

        # Reaction visualizations: edges connect directly to enzyme box faces.
        def reaction_has_enzyme(reaction: Dict[str, Any]) -> bool:
            enzymes = reaction.get("reaction-enzymes", [])
            return isinstance(enzymes, list) and bool(enzymes)

        def reaction_enzyme_box(reaction_idx: int, reaction: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
            for protein_id in reaction_enzyme_protein_ids(reaction):
                prot_loc = protein_loc_by_id.get(protein_id)
                if prot_loc:
                    return (
                        int(prot_loc["x"]),
                        int(prot_loc["y"]),
                        int(prot_loc["width"]),
                        int(prot_loc["height"]),
                    )
            if not reaction_has_enzyme(reaction):
                return None
            layout = reaction_layouts.get(reaction_idx)
            if layout:
                return layout["enzyme-x"], layout["enzyme-y"], protein_w, protein_h
            return None

        for reaction_idx, reaction in enumerate(reactions):
            rx_bs_id = reaction_bs_ids.get(reaction_idx, default_state_id)
            enzyme_box = reaction_enzyme_box(reaction_idx, reaction)
            if enzyme_box:
                enzyme_x, enzyme_y, enzyme_width, enzyme_height = enzyme_box
                enzyme_left_x = enzyme_x
                enzyme_right_x = enzyme_x + enzyme_width
                enzyme_cy = enzyme_y + enzyme_height // 2
            else:
                rx_region = region_for(rx_bs_id)
                layout = reaction_layouts.get(reaction_idx)
                enzyme_left_x = layout["enzyme-x"] if layout else rx_region["x"] + rx_region["w"] // 2
                enzyme_right_x = enzyme_left_x + 70
                enzyme_cy = layout["enzyme-cy"] if layout else rx_region["y"] + rx_region["h"] // 2

            no_enzyme = enzyme_box is None
            no_enzyme_virtual_left = enzyme_left_x
            if no_enzyme:
                left_elements = (
                    reaction.get("reaction-left-elements", [])
                    if isinstance(reaction.get("reaction-left-elements"), list)
                    else []
                )
                for rel in left_elements:
                    etype = str(rel.get("element-type") or "")
                    eid = int(rel.get("element-id") or 0)
                    loc = location_info(etype, eid, "Left", reaction_idx)
                    if loc:
                        no_enzyme_virtual_left = loc[1] + substrate_gap
                        break
                enzyme_left_x = no_enzyme_virtual_left
                enzyme_right_x = no_enzyme_virtual_left + 70

            reaction_compound_visualizations: List[Dict[str, Any]] = []
            reaction_element_collection_visualizations: List[Dict[str, Any]] = []
            reaction_enzyme_visualizations: List[Dict[str, Any]] = []

            for side_key, side in [("reaction-left-elements", "Left"), ("reaction-right-elements", "Right")]:
                for rel in reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []:
                    etype = str(rel.get("element-type") or "")
                    eid = int(rel.get("element-id") or 0)
                    loc = location_info(etype, eid, side, reaction_idx)
                    if not loc:
                        continue
                    location_id, lx, ly = loc
                    edge_id = self.ids.next()
                    if no_enzyme and side == "Left":
                        x1, y1, x2, y2 = enzyme_right_x, enzyme_cy, enzyme_right_x, enzyme_cy
                        hidden = True
                    elif no_enzyme:
                        x1, y1, x2, y2 = lx, enzyme_cy, no_enzyme_virtual_left, enzyme_cy
                        hidden = False
                    elif side == "Left":
                        x1, y1, x2, y2 = lx, ly, enzyme_left_x, enzyme_cy
                        hidden = False
                    else:
                        x1, y1, x2, y2 = lx, ly, enzyme_right_x, enzyme_cy
                        hidden = False
                    path, cp1x, cp1y, cp2x, cp2y = _curved_edge_path(x1, y1, x2, y2)
                    edge = {
                        "id": edge_id,
                        "path": path,
                        "visualization-template-id": 5,
                        "hidden": hidden,
                        "zindex": 18,
                    }
                    if side == "Left":
                        _add_end_arrow(edge, x2, y2, cp2x, cp2y)
                    else:
                        _add_start_arrow(edge, x1, y1, cp1x, cp1y)
                    edges.append(edge)
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
            layout = transport_layouts.get(int(transport["id"]))
            if layout:
                transporter_top_x = layout["x"] + layout["width"] // 2
                transporter_top_y = layout["y"]
                transporter_bottom_x = transporter_top_x
                transporter_bottom_y = layout["y"] + layout["height"]
            else:
                rx_region = region_for(default_state_id)
                transporter_top_x = rx_region["x"] + rx_region["w"] // 2
                transporter_top_y = rx_region["y"] + rx_region["h"] // 2
                transporter_bottom_x = transporter_top_x
                transporter_bottom_y = transporter_top_y + protein_h

            transport_compound_visualizations: List[Dict[str, Any]] = []
            transport_transporter_visualizations: List[Dict[str, Any]] = []

            for element in transport.get("transport-elements", []) if isinstance(transport.get("transport-elements"), list) else []:
                if str(element.get("element-type") or "") != "Compound":
                    continue
                compound_id = int(element.get("element-id") or 0)
                if not compound_id:
                    continue
                left_bs_id = int(element.get("left-biological-state-id") or default_state_id)
                right_bs_id = int(element.get("right-biological-state-id") or default_state_id)

                for bs_id, side in [(left_bs_id, "Left"), (right_bs_id, "Right")]:
                    loc = ensure_compound_location(compound_id, bs_id)
                    location_id = int(loc["id"])
                    loc_x = int(loc["x"])
                    loc_y = int(loc["y"])
                    loc_w = int(loc["width"])
                    loc_h = int(loc["height"])
                    edge_id = self.ids.next()
                    if side == "Left":
                        anchor_x = loc_x + loc_w // 2
                        anchor_y = loc_y + loc_h
                        x1, y1, x2, y2 = anchor_x, anchor_y, transporter_top_x, transporter_top_y
                    else:
                        anchor_x = loc_x + loc_w // 2
                        anchor_y = loc_y
                        x1, y1, x2, y2 = anchor_x, anchor_y, transporter_bottom_x, transporter_bottom_y
                    path, cp1x, cp1y, cp2x, cp2y = _curved_edge_path(x1, y1, x2, y2)
                    edge = {
                        "id": edge_id,
                        "path": path,
                        "visualization-template-id": 83,
                        "hidden": False,
                        "zindex": 18,
                    }
                    if side == "Left":
                        _add_end_arrow(edge, x2, y2, cp2x, cp2y)
                    else:
                        _add_start_arrow(edge, x1, y1, cp1x, cp1y)
                    edges.append(edge)
                    transport_compound_visualizations.append(
                        {
                            "id": self.ids.next(),
                            "compound-location-id": location_id,
                            "edge-id": edge_id,
                            "side": side,
                        }
                    )

            for transporter in transport.get("transport-transporters", []) if isinstance(transport.get("transport-transporters"), list) else []:
                pc_id = transporter.get("protein-complex-id")
                prot_id = transporter.get("protein-id")
                if pc_id is not None:
                    pc_vis = pc_vis_by_pc_id.get(int(pc_id))
                    if not pc_vis:
                        continue
                    transport_transporter_visualizations.append(
                        {
                            "id": self.ids.next(),
                            "protein-complex-visualization-id": int(pc_vis["id"]),
                            "transport-transporter-id": int(transporter["id"]),
                        }
                    )
                elif prot_id is not None:
                    prot_loc = protein_loc_by_id.get(int(prot_id))
                    if not prot_loc:
                        continue
                    transport_transporter_visualizations.append(
                        {
                            "id": self.ids.next(),
                            "protein-location-id": int(prot_loc["id"]),
                            "transport-transporter-id": int(transporter["id"]),
                        }
                    )

            transport_visualizations.append({
                "id": self.ids.next(),
                "transport-id": int(transport["id"]),
                "pathway-visualization-id": self.pathway_visualization_id_int,
                "transport_compound_visualizations": transport_compound_visualizations,
                "transport_transporter_visualizations": transport_transporter_visualizations,
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

        def document_id_for(
            record: Dict[str, Any],
            *,
            pw_keys: Sequence[str],
            pathbank_keys: Sequence[str],
            fallback: IdFactory,
        ) -> int:
            explicit = first_int(record, pw_keys)
            if explicit is not None:
                return explicit
            pathwhiz = first_int(record, ["pathwhiz_id"])
            pathbank_values = {first_int(record, [key]) for key in pathbank_keys}
            pathbank_values.discard(None)
            if pathwhiz is not None and pathwhiz not in pathbank_values:
                return pathwhiz
            return fallback.next()

        def protein_template_id(value: Any) -> int:
            template_id = _to_positive_int(value)
            if template_id is None or template_id == 4:
                return 2
            return template_id

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
            state = {
                "id": rid,
                "name": record.get("name") or record.get("key") or f"State {rid}",
                "tissue-id": lookup("tissues", record.get("tissue_key")),
                "subcellular-location-id": lookup(
                    "subcellular_locations", record.get("subcellular_location_key")
                ),
                "species-id": lookup("species", record.get("species_key")),
                "cell-type-id": lookup("cell_types", record.get("cell_type_key")),
            }
            pwbs_id = _trusted_biological_state_pwbs_id(record)
            if pwbs_id is not None:
                state["pwbs-id"] = pwbs_id
            biological_states.append(state)
        self.section_items["biological-states"] = biological_states

        entities = ir.get("entities") if isinstance(ir.get("entities"), dict) else {}
        self.section_items["bounds"] = []
        raw_entity_by_key: Dict[str, Dict[str, Any]] = {}
        protein_key_by_name: Dict[str, str] = {}
        protein_key_by_db_id: Dict[int, str] = {}
        for bucket_name, bucket in entities.items():
            for record in bucket if isinstance(bucket, list) else []:
                if not isinstance(record, dict):
                    continue
                key = str(record.get("key") or "")
                if key:
                    raw_entity_by_key[key] = record
                if bucket_name == "proteins" and key:
                    name = str(record.get("name") or "").strip()
                    if name:
                        protein_key_by_name[_normalize_key(name)] = key
                    for db_key in ["pathwhiz_id", "pathbank_protein_id", "pw_protein_id", "protein_id"]:
                        db_id = _to_positive_int(record.get(db_key))
                        if db_id is not None:
                            protein_key_by_db_id[db_id] = key

        self.section_items["compounds"] = []
        for record in entities.get("compounds", []) if isinstance(entities.get("compounds"), list) else []:
            if not isinstance(record, dict):
                continue
            db_row = record.get("db_row") if isinstance(record.get("db_row"), dict) else {}
            rid = document_id_for(
                record,
                pw_keys=["pw_compound_id", "pathbank_compound_id"],
                pathbank_keys=[],
                fallback=self.compound_ids,
            )
            remember("entities", record.get("key"), rid)
            self._ir_entity_info[str(record.get("key"))] = {
                "id": rid,
                "type": "Compound",
                "entity_type": "compound",
                "template_id": select_compound_template_id(record),
            }
            mapped_ids = record.get("mapped_ids") if isinstance(record.get("mapped_ids"), dict) else {}
            chebi_id = (
                db_row.get("chebi_id")
                or record.get("chebi_id")
                or mapped_ids.get("chebi")
                or None
            )
            if chebi_id is not None:
                chebi_id = str(chebi_id).replace("CHEBI:", "").strip()
            pwc_id = _trusted_compound_pwc_id(record, db_row, mapped_ids)
            short_name = _trusted_compound_short_name(record, db_row)
            compound_item = {
                "id": rid,
                "name": db_row.get("name") or record.get("name", ""),
                "description": db_row.get("description") or record.get("description"),
                "cas": db_row.get("cas") or record.get("cas"),
            }
            if pwc_id is not None:
                compound_item["pwc-id"] = pwc_id
            if short_name is not None:
                compound_item["short-name"] = short_name
            compound_item.update(
                {
                    "element-states": [],
                    "hmdb-id": db_row.get("hmdb_id") or record.get("hmdb_id") or mapped_ids.get("hmdb") or None,
                    "kegg-id": db_row.get("kegg_id") or record.get("kegg_id") or mapped_ids.get("kegg") or None,
                    "chebi-id": chebi_id or None,
                    "pubchem-cid": db_row.get("pubchem_cid") or record.get("pubchem_cid") or mapped_ids.get("pubchem") or None,
                    "biocyc-id": db_row.get("biocyc_id") or record.get("biocyc_id") or mapped_ids.get("biocyc") or None,
                    "chemspider-id": (
                        db_row.get("chemspider_id") or record.get("chemspider_id") or mapped_ids.get("chemspider") or None
                    ),
                    "drugbank-id": db_row.get("drugbank_id") or record.get("drugbank_id") or mapped_ids.get("drugbank") or None,
                }
            )
            self.section_items["compounds"].append(compound_item)

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
            rid = document_id_for(
                record,
                pw_keys=["pw_protein_id"],
                pathbank_keys=["pathbank_protein_id"],
                fallback=self.protein_ids,
            )
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
            uniprot_id = (
                mapped_ids.get("uniprot")
                or record.get("uniprot")
                or record.get("uniprot_id")
                or record.get("uniprot-id")
                or None
            )
            drugbank_id = (
                mapped_ids.get("drugbank")
                or record.get("drugbank")
                or record.get("drugbank_id")
                or record.get("drugbank-id")
                or None
            )
            self.section_items["proteins"].append(
                {
                    "id": rid,
                    "name": record.get("name", ""),
                    "species-id": default_species_id,
                    "element-states": [],
                    "uniprot-id": uniprot_id,
                    "drugbank-id": drugbank_id,
                    "ec-numbers": record.get("ec_numbers", []),
                }
            )

        self.section_items["protein-complexes"] = []
        for record in entities.get("protein_complexes", []) if isinstance(entities.get("protein_complexes"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = document_id_for(
                record,
                pw_keys=["pw_complex_id"],
                pathbank_keys=["pathbank_protein_complex_id", "pathbank_complex_id"],
                fallback=self.complex_ids,
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
            seen_enzyme_targets: Set[Tuple[str, int]] = set()
            for member in reaction.get("enzymes", []) if isinstance(reaction.get("enzymes"), list) else []:
                if not isinstance(member, dict):
                    continue
                info = entity_info(member.get("entity_key"))
                if not info:
                    continue
                enzyme_target = (str(info["entity_type"]), int(info["id"]))
                if enzyme_target in seen_enzyme_targets:
                    continue
                seen_enzyme_targets.add(enzyme_target)
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
                item["enzyme-class"] = member.get("enzyme_class") or member.get("enzyme-class")
                enzymes.append(item)
            reactions.append(
                {
                    "id": rid,
                    "spontaneous": bool(reaction.get("spontaneous", False)),
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
        reaction_compound_entity_keys: set[str] = set()
        for reaction in process_items.get("reactions", []) if isinstance(process_items.get("reactions"), list) else []:
            if not isinstance(reaction, dict):
                continue
            for side_key in ["left", "right"]:
                members = reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []
                for member in members:
                    if not isinstance(member, dict):
                        continue
                    entity_key = str(member.get("entity_key") or "")
                    info = entity_info(entity_key)
                    if info and info.get("entity_type") == "compound":
                        reaction_compound_entity_keys.add(entity_key)
        total_compound_count = len(self.section_items.get("compounds", []))
        grid_placed_compound_count = 0
        skipped_base_location_by_id: Dict[int, Dict[str, Any]] = {}
        location_by_entity_state: Dict[Tuple[str, str], int] = {}
        protein_location_by_entity_state: Dict[Tuple[str, str], int] = {}
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
            if str(loc.get("type") or "") == "protein_location":
                protein_location_by_entity_state[
                    (str(loc.get("entity_key") or ""), str(loc.get("biological_state_key") or ""))
                ] = lid
            location_by_entity_state[
                (str(loc.get("entity_key") or ""), str(loc.get("biological_state_key") or ""))
            ] = lid
            item = {
                    "id": lid,
                    entity_field: int(info["id"]),
                    "biological-state-id": lookup("biological_states", loc.get("biological_state_key")),
                    "visualization-template-id": int(loc.get("visualization_template_id") or (3 if section == "compound-locations" else 0)),
                    "hidden": bool(loc.get("hidden", False)),
                    "x": int(loc.get("x") or 0),
                    "y": int(loc.get("y") or 0),
                    "zindex": int(loc.get("zindex") or 10),
                    "font-size": str(loc.get("font_size") or loc.get("font-size") or "regular"),
                    "width": str(loc.get("width") or 160),
                    "height": str(loc.get("height") or 60),
                }
            if str(loc.get("type") or "") == "protein_location":
                item["visualization-template-id"] = protein_template_id(loc.get("visualization_template_id"))
                item["zindex"] = int(loc.get("zindex") or 8)
                item["label-type"] = str(loc.get("label_type") or loc.get("label-type") or "subunit")
                item["width"] = str(loc.get("width") or 150)
                item["height"] = str(loc.get("height") or 70)
            skip_base_compound_location = (
                section == "compound-locations"
                and info.get("entity_type") == "compound"
                and str(loc.get("entity_key") or "") in reaction_compound_entity_keys
            )
            if skip_base_compound_location:
                skipped_base_location_by_id[lid] = item
            else:
                if section == "compound-locations":
                    grid_placed_compound_count += 1
                self.section_items[section].append(item)
        self.layout_debug_counts.update(
            {
                "compound_total": total_compound_count,
                "compound_grid_skipped_reaction_used": len(reaction_compound_entity_keys),
                "compound_grid_placed_non_reaction": grid_placed_compound_count,
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
            pc_protein_vis: List[Dict[str, Any]] = []
            raw_pc = raw_entity_by_key.get(entity_key, {})
            for component in raw_pc.get("components", []) if isinstance(raw_pc.get("components"), list) else []:
                protein_key = ""
                if isinstance(component, dict):
                    protein_key = str(component.get("protein_key") or component.get("entity_key") or "").strip()
                    if not protein_key:
                        for db_key in ["pathwhiz_id", "pathbank_protein_id", "pw_protein_id", "protein_id"]:
                            db_id = _to_positive_int(component.get(db_key))
                            if db_id is not None and db_id in protein_key_by_db_id:
                                protein_key = protein_key_by_db_id[db_id]
                                break
                comp_name = _component_name(component)
                if not protein_key and comp_name:
                    protein_key = protein_key_by_name.get(_normalize_key(comp_name), "")
                protein_location_id = protein_location_by_entity_state.get((protein_key, biological_state_key))
                if protein_location_id is None and protein_key:
                    protein_info = entity_info(protein_key)
                    biological_state_id = lookup("biological_states", item.get("biological_state_key"))
                    if protein_info is not None and biological_state_id is not None:
                        protein_location_id = self.ids.next()
                        protein_location_by_entity_state[(protein_key, biological_state_key)] = protein_location_id
                        location_by_entity_state[(protein_key, biological_state_key)] = protein_location_id
                        self.section_items["protein-locations"].append(
                            {
                                "id": protein_location_id,
                                "protein-id": int(protein_info["id"]),
                                "biological-state-id": biological_state_id,
                                "visualization-template-id": protein_template_id(item.get("visualization_template_id")),
                                "hidden": bool(item.get("hidden", False)),
                                "x": int(item.get("x") or 0),
                                "y": int(item.get("y") or 0),
                                "zindex": int(item.get("zindex") or 8),
                                "label-type": "subunit",
                                "font-size": "regular",
                                "width": "150",
                                "height": "70",
                            }
                        )
                if protein_location_id is not None:
                    pc_protein_vis.append({"id": self.ids.next(), "protein-location-id": protein_location_id})
            self.section_items["protein-complex-visualizations"].append(
                {
                    "id": vid,
                    "protein-complex-id": int(info["id"]),
                    "pathway-visualization-id": self.pathway_visualization_id_int,
                    "biological-state-id": lookup("biological_states", item.get("biological_state_key")),
                    "protein_complex_protein_visualizations": pc_protein_vis,
                    "protein_complex_compound_visualizations": [],
                }
            )

        loc_by_id: Dict[int, Dict[str, Any]] = {}
        for section in [
            "compound-locations",
            "protein-locations",
            "nucleic-acid-locations",
            "element-collection-locations",
        ]:
            for loc in self.section_items.get(section, []):
                loc_by_id[int(loc["id"])] = loc
        loc_by_id.update(skipped_base_location_by_id)

        protein_complex_viz_by_id = {
            int(viz["id"]): viz
            for viz in self.section_items.get("protein-complex-visualizations", [])
        }

        layout_substrate_gap = 46
        layout_product_gap = 51
        layout_compound_gap_y = 30
        layout_rxn_step_x = 800
        layout_protein_w, layout_protein_h = 150, 70
        layout_protein_gap_x = 10

        state_region_by_key: Dict[str, Dict[str, int]] = {}
        for bound in ir.get("bound_visualizations", []) if isinstance(ir.get("bound_visualizations"), list) else []:
            if not isinstance(bound, dict):
                continue
            biological_state_key = str(bound.get("biological_state_key") or "")
            if biological_state_key:
                state_region_by_key[biological_state_key] = {
                    "x": int(bound.get("x") or 0),
                    "y": int(bound.get("y") or 0),
                    "w": int(bound.get("width") or self.args.width),
                    "h": int(bound.get("height") or self.args.height),
                }

        def ir_region_for_state(state_key: str) -> Dict[str, int]:
            return state_region_by_key.get(
                state_key,
                {"x": 0, "y": 0, "w": int(self.args.width), "h": int(self.args.height)},
            )

        raw_reactions_by_key = {
            str(reaction.get("key")): reaction
            for reaction in process_items.get("reactions", [])
            if isinstance(reaction, dict) and reaction.get("key") is not None
        }
        reaction_keys_by_state: Dict[str, List[str]] = defaultdict(list)
        for reaction in process_items.get("reactions", []) if isinstance(process_items.get("reactions"), list) else []:
            if not isinstance(reaction, dict):
                continue
            reaction_key = str(reaction.get("key") or "")
            biological_state_key = str(reaction.get("biological_state_key") or "")
            if reaction_key:
                reaction_keys_by_state[biological_state_key].append(reaction_key)

        reaction_layout_by_key: Dict[str, Dict[str, int]] = {}
        for biological_state_key, reaction_keys in reaction_keys_by_state.items():
            region = ir_region_for_state(biological_state_key)
            n_rxns = len(reaction_keys)
            enzyme_cx_base = region["x"] + region["w"] // 2 - (n_rxns - 1) * layout_rxn_step_x // 2
            enzyme_cy = max(region["y"] + region["h"] // 2, 360)
            for k, reaction_key in enumerate(reaction_keys):
                enzyme_cx = enzyme_cx_base + k * layout_rxn_step_x
                enzyme_x = enzyme_cx - layout_protein_w // 2
                enzyme_y = enzyme_cy - layout_protein_h // 2
                reaction_layout_by_key[reaction_key] = {
                    "enzyme-cx": enzyme_cx,
                    "enzyme-cy": enzyme_cy,
                    "enzyme-x": enzyme_x,
                    "enzyme-y": enzyme_y,
                    "compound-stack-cy": enzyme_cy,
                }

        def ensure_protein_location(protein_key: str, biological_state_key: str) -> Optional[int]:
            loc_id = protein_location_by_entity_state.get((protein_key, biological_state_key))
            if loc_id is not None:
                return loc_id
            protein_info = entity_info(protein_key)
            biological_state_id = lookup("biological_states", biological_state_key)
            if protein_info is None or biological_state_id is None:
                return None
            loc_id = self.ids.next()
            protein_location_by_entity_state[(protein_key, biological_state_key)] = loc_id
            location_by_entity_state[(protein_key, biological_state_key)] = loc_id
            loc = {
                "id": loc_id,
                "protein-id": int(protein_info["id"]),
                "biological-state-id": biological_state_id,
                "visualization-template-id": 2,
                "hidden": False,
                "x": 0,
                "y": 0,
                "zindex": 8,
                "label-type": "subunit",
                "font-size": "regular",
                "width": str(layout_protein_w),
                "height": str(layout_protein_h),
            }
            self.section_items["protein-locations"].append(loc)
            loc_by_id[loc_id] = loc
            return loc_id

        def enzyme_protein_location_ids(reaction: Dict[str, Any], biological_state_key: str) -> List[int]:
            loc_ids: List[int] = []
            for member in reaction.get("enzymes", []) if isinstance(reaction.get("enzymes"), list) else []:
                if not isinstance(member, dict):
                    continue
                entity_key = str(member.get("entity_key") or "")
                info = entity_info(entity_key)
                if not info:
                    continue
                if info["entity_type"] == "protein":
                    loc_id = ensure_protein_location(entity_key, biological_state_key)
                    if loc_id is not None and loc_id not in loc_ids:
                        loc_ids.append(loc_id)
                elif info["entity_type"] == "protein_complex":
                    pcv_id = protein_complex_viz_by_entity_state.get((entity_key, biological_state_key))
                    if pcv_id is None:
                        pcv_id = protein_complex_viz_by_entity.get(entity_key)
                    pcv = protein_complex_viz_by_id.get(int(pcv_id)) if pcv_id is not None else None
                    if pcv:
                        for protein_viz in pcv.get("protein_complex_protein_visualizations", []):
                            loc_id = int(protein_viz.get("protein-location-id") or 0)
                            if loc_id and loc_id not in loc_ids:
                                loc_ids.append(loc_id)
            return loc_ids

        enzyme_loc_ids_by_reaction_key: Dict[str, List[int]] = {}
        for reaction_key, reaction in raw_reactions_by_key.items():
            biological_state_key = str(reaction.get("biological_state_key") or "")
            layout = reaction_layout_by_key.get(reaction_key)
            if not layout:
                continue
            loc_ids = enzyme_protein_location_ids(reaction, biological_state_key)
            enzyme_loc_ids_by_reaction_key[reaction_key] = loc_ids
            if not loc_ids:
                continue
            total_w = len(loc_ids) * layout_protein_w + (len(loc_ids) - 1) * layout_protein_gap_x
            x0 = layout["enzyme-cx"] - total_w // 2
            for j, loc_id in enumerate(loc_ids):
                loc = loc_by_id.get(loc_id)
                if not loc:
                    continue
                loc["x"] = x0 + j * (layout_protein_w + layout_protein_gap_x)
                loc["y"] = layout["enzyme-y"]
                loc["width"] = str(layout_protein_w)
                loc["height"] = str(layout_protein_h)

        reaction_member_location_by_key: Dict[str, int] = {}

        def location_section_for_record(loc: Dict[str, Any]) -> Optional[str]:
            if "compound-id" in loc:
                return "compound-locations"
            if "element-collection-id" in loc:
                return "element-collection-locations"
            if "nucleic-acid-id" in loc:
                return "nucleic-acid-locations"
            return None

        def add_reaction_member_location(
            member: Dict[str, Any],
            biological_state_key: str,
            x: int,
            y: int,
            width: Optional[int] = None,
            height: Optional[int] = None,
        ) -> Optional[int]:
            entity_key = str(member.get("entity_key") or "")
            member_state_key = str(member.get("biological_state_key") or biological_state_key)
            base_loc_id = location_by_entity_state.get((entity_key, member_state_key))
            if base_loc_id is None:
                base_loc_id = lookup("locations", member.get("location_key"))
            if base_loc_id is None:
                return None
            base_loc = loc_by_id.get(base_loc_id)
            if not base_loc:
                return None
            section = location_section_for_record(base_loc)
            if not section:
                return None
            loc_id = self.ids.next()
            loc = dict(base_loc)
            loc["id"] = loc_id
            loc["x"] = x
            loc["y"] = y
            if width is not None:
                loc["width"] = str(width)
            if height is not None:
                loc["height"] = str(height)
            biological_state_id = lookup("biological_states", member_state_key)
            if biological_state_id is not None:
                loc["biological-state-id"] = biological_state_id
            self.section_items[section].append(loc)
            loc_by_id[loc_id] = loc
            member_key = str(member.get("key") or "")
            if member_key:
                reaction_member_location_by_key[member_key] = loc_id
            return loc_id

        def reaction_member_dims(member: Dict[str, Any], loc: Dict[str, Any]) -> Tuple[int, int]:
            info = entity_info(member.get("entity_key"))
            if info and info.get("entity_type") == "compound":
                template_id = _to_positive_int(info.get("template_id"))
                if template_id is None:
                    template_id = _to_positive_int(loc.get("visualization-template-id"))
                return TEMPLATE_DIMS.get(template_id or 3, TEMPLATE_DIMS[3])
            return int(loc["width"]), int(loc["height"])

        compound_record_by_id = {
            int(compound["id"]): compound
            for compound in self.section_items.get("compounds", [])
            if isinstance(compound, dict) and compound.get("id") is not None
        }

        reaction_idx_by_key = {reaction_key: idx for idx, reaction_key in enumerate(raw_reactions_by_key)}
        reaction_sequence = list(raw_reactions_by_key.items())

        def compound_member_identity(
            member: Dict[str, Any],
            reaction_biological_state_key: str,
        ) -> Optional[Tuple[int, str]]:
            info = entity_info(member.get("entity_key"))
            if not info or info.get("entity_type") != "compound":
                return None
            state_key = str(member.get("biological_state_key") or reaction_biological_state_key)
            return int(info["id"]), state_key

        participant_occurrences: Dict[Tuple[int, str], List[Tuple[int, str, str, str]]] = defaultdict(list)
        left_occurrences: Dict[Tuple[int, str], List[Tuple[int, str, str, str]]] = defaultdict(list)
        right_occurrences: Dict[Tuple[int, str], List[Tuple[int, str, str, str]]] = defaultdict(list)
        for idx, (reaction_key, reaction) in enumerate(reaction_sequence):
            biological_state_key = str(reaction.get("biological_state_key") or "")
            for side_key, by_side in [("left", left_occurrences), ("right", right_occurrences)]:
                members = reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []
                for member in members:
                    if not isinstance(member, dict):
                        continue
                    identity = compound_member_identity(member, biological_state_key)
                    if identity is None:
                        continue
                    occurrence = (idx, reaction_key, side_key, str(member.get("key") or ""))
                    participant_occurrences[identity].append(occurrence)
                    by_side[identity].append(occurrence)

        shared_from_previous: Dict[Tuple[int, str, int], int] = {}
        for idx in range(len(reaction_sequence) - 1):
            producer_key, producer = reaction_sequence[idx]
            consumer_key, consumer = reaction_sequence[idx + 1]
            producer_state_key = str(producer.get("biological_state_key") or "")
            consumer_state_key = str(consumer.get("biological_state_key") or "")
            producer_right = producer.get("right", []) if isinstance(producer.get("right"), list) else []
            consumer_left = consumer.get("left", []) if isinstance(consumer.get("left"), list) else []
            producer_products = {
                identity
                for member in producer_right
                if isinstance(member, dict)
                for identity in [compound_member_identity(member, producer_state_key)]
                if identity is not None
            }
            consumer_substrates = {
                identity
                for member in consumer_left
                if isinstance(member, dict)
                for identity in [compound_member_identity(member, consumer_state_key)]
                if identity is not None
            }
            candidates: List[Tuple[int, str]] = []
            for identity in sorted(producer_products & consumer_substrates):
                compound_id, state_key = identity
                compound_record = compound_record_by_id.get(compound_id, {})
                debug_base = {
                    "compound_id": compound_id,
                    "compound_name": compound_record.get("name", ""),
                    "biological_state_key": state_key,
                    "producer_rxn_idx": idx,
                    "producer_reaction_key": producer_key,
                    "consumer_rxn_idx": idx + 1,
                    "consumer_reaction_key": consumer_key,
                }
                if _is_currency_compound_record(compound_record):
                    self.layout_debug_shared_intermediates.append({**debug_base, "action": "skipped_cofactor"})
                    continue
                if (
                    len(right_occurrences.get(identity, [])) != 1
                    or len(left_occurrences.get(identity, [])) != 1
                    or len(participant_occurrences.get(identity, [])) != 2
                ):
                    self.layout_debug_shared_intermediates.append(
                        {**debug_base, "action": "skipped_ambiguous_participation"}
                    )
                    continue
                candidates.append(identity)
            if len(candidates) > 1:
                for compound_id, state_key in candidates:
                    compound_record = compound_record_by_id.get(compound_id, {})
                    self.layout_debug_shared_intermediates.append(
                        {
                            "action": "skipped_branching_ambiguity",
                            "compound_id": compound_id,
                            "compound_name": compound_record.get("name", ""),
                            "biological_state_key": state_key,
                            "producer_rxn_idx": idx,
                            "producer_reaction_key": producer_key,
                            "consumer_rxn_idx": idx + 1,
                            "consumer_reaction_key": consumer_key,
                        }
                    )
                continue
            for compound_id, state_key in candidates:
                shared_from_previous[(compound_id, state_key, idx + 1)] = idx
                compound_record = compound_record_by_id.get(compound_id, {})
                self.layout_debug_shared_intermediates.append(
                    {
                        "action": "detected_shared_intermediate",
                        "compound_id": compound_id,
                        "compound_name": compound_record.get("name", ""),
                        "biological_state_key": state_key,
                        "producer_rxn_idx": idx,
                        "producer_reaction_key": producer_key,
                        "consumer_rxn_idx": idx + 1,
                        "consumer_reaction_key": consumer_key,
                    }
                )
        self.layout_debug_counts.update(
            {
                "shared_intermediates_detected": sum(
                    1
                    for item in self.layout_debug_shared_intermediates
                    if item.get("action") == "detected_shared_intermediate"
                ),
                "shared_intermediates_skipped_cofactor": sum(
                    1
                    for item in self.layout_debug_shared_intermediates
                    if item.get("action") == "skipped_cofactor"
                ),
            }
        )
        compound_loc_by_rxn_side: Dict[Tuple[int, str, int, str], int] = {}

        for reaction_key, reaction in raw_reactions_by_key.items():
            layout = reaction_layout_by_key.get(reaction_key)
            if not layout:
                continue
            reaction_idx = reaction_idx_by_key.get(reaction_key, 0)
            biological_state_key = str(reaction.get("biological_state_key") or "")
            enzyme_left_x = layout["enzyme-x"]
            enzyme_right_x = enzyme_left_x + layout_protein_w
            compound_stack_cy = layout["compound-stack-cy"]
            for side_key in ["left", "right"]:
                members = reaction.get(side_key, []) if isinstance(reaction.get(side_key), list) else []
                side = "Left" if side_key == "left" else "Right"
                stack_members: List[Dict[str, Any]] = []
                for member in members:
                    if not isinstance(member, dict):
                        continue
                    entity_key = str(member.get("entity_key") or "")
                    member_state_key = str(member.get("biological_state_key") or biological_state_key)
                    identity = compound_member_identity(member, biological_state_key)
                    if side_key == "left" and identity is not None:
                        compound_id, state_key = identity
                        producer_idx = shared_from_previous.get((compound_id, state_key, reaction_idx))
                        if producer_idx is not None:
                            shared_loc_id = compound_loc_by_rxn_side.get(
                                (compound_id, state_key, producer_idx, "Right")
                            )
                            if shared_loc_id is not None:
                                member_key = str(member.get("key") or "")
                                if member_key:
                                    reaction_member_location_by_key[member_key] = shared_loc_id
                                compound_loc_by_rxn_side[(compound_id, state_key, reaction_idx, "Left")] = shared_loc_id
                                compound_record = compound_record_by_id.get(compound_id, {})
                                self.layout_debug_shared_intermediates.append(
                                    {
                                        "action": "reused_location",
                                        "compound_id": compound_id,
                                        "compound_name": compound_record.get("name", ""),
                                        "biological_state_key": state_key,
                                        "producer_rxn_idx": producer_idx,
                                        "consumer_rxn_idx": reaction_idx,
                                        "location_id": shared_loc_id,
                                    }
                                )
                                continue
                    base_loc_id = location_by_entity_state.get((entity_key, member_state_key))
                    if base_loc_id is None:
                        base_loc_id = lookup("locations", member.get("location_key"))
                    if base_loc_id is None:
                        continue
                    loc = loc_by_id.get(base_loc_id)
                    if not loc:
                        continue
                    width, height = reaction_member_dims(member, loc)
                    stack_members.append(
                        {
                            "member": member,
                            "entity_key": entity_key,
                            "width": width,
                            "height": height,
                        }
                    )
                heights = [int(member["height"]) for member in stack_members]
                total_stack_height, stack_tops = _packed_reaction_stack_tops(
                    heights,
                    compound_stack_cy,
                    layout_compound_gap_y,
                )
                self.layout_debug_stacks.append(
                    {
                        "reaction_idx": reaction_idx,
                        "reaction_key": reaction_key,
                        "side": side,
                        "element_ids": [member["entity_key"] for member in stack_members],
                        "heights": heights,
                        "total_stack_height": total_stack_height,
                        "enzyme_cy": compound_stack_cy,
                        "y_positions": stack_tops,
                    }
                )
                for stack_member, y in zip(stack_members, stack_tops):
                    member = stack_member["member"]
                    width = int(stack_member["width"])
                    height = int(stack_member["height"])
                    expected_anchor_x = (
                        enzyme_left_x - layout_substrate_gap
                        if side_key == "left"
                        else enzyme_right_x + layout_product_gap
                    )
                    expected_anchor_y = y + height // 2
                    x = expected_anchor_x - width if side == "Left" else expected_anchor_x
                    if y < 0:
                        import warnings
                        warnings.warn(
                            f"DEBUG layout (IR): reaction_key={reaction_key} side={side_key} "
                            f"y={y} (anchor_y={expected_anchor_y} height={height} compound_stack_cy={compound_stack_cy})"
                        )
                    loc_id = add_reaction_member_location(
                        member,
                        biological_state_key,
                        x,
                        y,
                        width,
                        height,
                    )
                    if loc_id is not None:
                        written_loc = loc_by_id[loc_id]
                        _assert_reaction_member_anchor(written_loc, side, expected_anchor_x, expected_anchor_y)
                        identity = compound_member_identity(member, biological_state_key)
                        if identity is not None:
                            compound_id, state_key = identity
                            compound_loc_by_rxn_side[(compound_id, state_key, reaction_idx, side)] = loc_id

        self.layout_debug_counts["shared_intermediate_locations_reused"] = sum(
            1
            for item in self.layout_debug_shared_intermediates
            if item.get("action") == "reused_location"
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

        edge_by_id = {int(edge["id"]): edge for edge in self.section_items["edges"]}

        def loc_connection_point(loc_id: int, side: str) -> Optional[Tuple[int, int]]:
            loc = loc_by_id.get(loc_id)
            if not loc:
                return None
            x = int(loc["x"])
            if side == "Left":
                x += int(loc["width"])
            return x, int(loc["y"]) + int(loc["height"]) // 2

        def reaction_key_has_enzyme(reaction_key: str) -> bool:
            reaction = raw_reactions_by_key.get(reaction_key)
            if not reaction:
                return False
            enzymes = reaction.get("enzymes", [])
            return isinstance(enzymes, list) and bool(enzymes)

        def enzyme_box_for_reaction_key(reaction_key: str) -> Optional[Tuple[int, int, int, int]]:
            for loc_id in enzyme_loc_ids_by_reaction_key.get(reaction_key, []):
                loc = loc_by_id.get(loc_id)
                if loc:
                    return int(loc["x"]), int(loc["y"]), int(loc["width"]), int(loc["height"])
            if not reaction_key_has_enzyme(reaction_key):
                return None
            layout = reaction_layout_by_key.get(reaction_key)
            if layout:
                return layout["enzyme-x"], layout["enzyme-y"], layout_protein_w, layout_protein_h
            return None

        for viz in ir.get("process_visualizations", []) if isinstance(ir.get("process_visualizations"), list) else []:
            if not isinstance(viz, dict) or str(viz.get("type") or "") != "reaction_visualization":
                continue
            reaction_key = str(viz.get("process_key") or "")
            enzyme_box = enzyme_box_for_reaction_key(reaction_key)
            if enzyme_box:
                enzyme_x, enzyme_y, enzyme_width, enzyme_height = enzyme_box
                enzyme_left = enzyme_x
                enzyme_right = enzyme_x + enzyme_width
                enzyme_cy = enzyme_y + enzyme_height // 2
            else:
                reaction = raw_reactions_by_key.get(reaction_key, {})
                biological_state_key = str(reaction.get("biological_state_key") or viz.get("biological_state_key") or "")
                region = ir_region_for_state(biological_state_key)
                layout = reaction_layout_by_key.get(reaction_key)
                enzyme_left = layout["enzyme-x"] if layout else region["x"] + region["w"] // 2
                enzyme_right = enzyme_left + 70
                enzyme_cy = layout["enzyme-cy"] if layout else region["y"] + region["h"] // 2
                for member in viz.get("members", []) if isinstance(viz.get("members"), list) else []:
                    if not isinstance(member, dict) or str(member.get("role") or "") != "left":
                        continue
                    loc_id = reaction_member_location_by_key.get(str(member.get("process_member_key") or ""))
                    if loc_id is None:
                        loc_id = lookup("locations", member.get("location_key"))
                    if loc_id is None:
                        continue
                    point = loc_connection_point(loc_id, "Left")
                    if point is not None:
                        enzyme_left = point[0] + layout_substrate_gap
                        enzyme_right = enzyme_left + 70
                        break
            no_enzyme = enzyme_box is None
            for member in viz.get("members", []) if isinstance(viz.get("members"), list) else []:
                if not isinstance(member, dict):
                    continue
                role = str(member.get("role") or "")
                side = "Left" if role == "left" else "Right" if role == "right" else ""
                if not side:
                    continue
                loc_id = reaction_member_location_by_key.get(str(member.get("process_member_key") or ""))
                if loc_id is None:
                    loc_id = lookup("locations", member.get("location_key"))
                edge_id = lookup("edges", member.get("edge_key"))
                if loc_id is None or edge_id is None:
                    continue
                point = loc_connection_point(loc_id, side)
                edge = edge_by_id.get(edge_id)
                if point is None or edge is None:
                    continue
                px, py = point
                edge["visualization-template-id"] = 5
                edge.pop("option:start_arrow", None)
                edge.pop("option:start_arrow_path", None)
                edge.pop("option:end_arrow", None)
                edge.pop("option:end_arrow_path", None)
                if no_enzyme and side == "Left":
                    x1, y1, x2, y2 = enzyme_right, enzyme_cy, enzyme_right, enzyme_cy
                    edge["hidden"] = True
                elif no_enzyme:
                    x1, y1, x2, y2 = px, enzyme_cy, enzyme_left, enzyme_cy
                    edge["hidden"] = False
                elif side == "Left":
                    x1, y1, x2, y2 = px, py, enzyme_left, enzyme_cy
                    edge["hidden"] = False
                else:
                    x1, y1, x2, y2 = px, py, enzyme_right, enzyme_cy
                    edge["hidden"] = False
                edge["path"], cp1x, cp1y, cp2x, cp2y = _curved_edge_path(x1, y1, x2, y2)
                if side == "Left":
                    _add_end_arrow(edge, x2, y2, cp2x, cp2y)
                else:
                    _add_start_arrow(edge, x1, y1, cp1x, cp1y)

        self.section_items["reaction-visualizations"] = []
        self.section_items["transport-visualizations"] = []
        self.section_items["reaction-coupled-transport-visualizations"] = []
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
                    loc_id = reaction_member_location_by_key.get(str(member.get("process_member_key") or ""))
                    if loc_id is None:
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
        self.section_items["compounds"] = []
        for rec in self.entity_records.get("compounds", []):
            mapped_ids = rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}
            db_row = rec.get("db_row") if isinstance(rec.get("db_row"), dict) else {}
            compound_item = {
                "id": int(rec["id"]),
                "name": rec["name"],
            }
            pwc_id = _trusted_compound_pwc_id(rec, db_row, mapped_ids)
            short_name = _trusted_compound_short_name(rec, db_row)
            if pwc_id is not None:
                compound_item["pwc-id"] = pwc_id
            if short_name is not None:
                compound_item["short-name"] = short_name
            compound_item.update(
                {
                    "element-states": [],
                    "hmdb-id": mapped_ids.get("hmdb") or None,
                    "kegg-id": mapped_ids.get("kegg") or None,
                    "chebi-id": mapped_ids.get("chebi") or None,
                    "pubchem-cid": mapped_ids.get("pubchem") or None,
                }
            )
            self.section_items["compounds"].append(compound_item)
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
                "uniprot-id": (
                    (rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}).get("uniprot")
                    or rec.get("uniprot")
                    or rec.get("uniprot_id")
                    or rec.get("uniprot-id")
                    or None
                ),
                "drugbank-id": (
                    (rec.get("mapped_ids") if isinstance(rec.get("mapped_ids"), dict) else {}).get("drugbank")
                    or rec.get("drugbank")
                    or rec.get("drugbank_id")
                    or rec.get("drugbank-id")
                    or None
                ),
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

        self.section_items["reaction-coupled-transport-visualizations"] = []
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
        node_tag = tag
        if tag.startswith("option:"):
            node_tag = f"{{{OPTION_TAG_NS}}}{tag.split(':', 1)[1]}"
        node = etree.SubElement(parent, node_tag)
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
        self._make_compound_identity_fields_optional()
        self._make_biological_state_identity_fields_optional()
        counts = self._populate_sections()

        root = etree.Element(self.signature.root_tag, nsmap={"option": OPTION_TAG_NS})
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

    blocking_ir_errors = blocking_pwml_ir_errors(ir_report)
    if blocking_ir_errors or ir_validation.get("errors"):
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
