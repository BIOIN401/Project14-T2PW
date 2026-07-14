from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lxml import etree

from t2pw.paths import PROJECT_ROOT
from t2pw.pipeline.process_normalizer import normalize_process_payload
from t2pw.pwml.compound_templates import TEMPLATE_DIMS, select_compound_template_id
from t2pw.pwml.ir import build_pwml_ir, is_pwml_ir, validate_pwml_ir
from t2pw.pwml.qa import run_pwml_qa
from t2pw.pwml.validate import (
    SectionSignature,
    StructureSignature,
    discover_structure_signature,
    repair_tree,
    validate_generated_tree,
)

OPTION_TAG_NS = "urn:pathwhiz-option"
ARROW_ANGLE = math.pi / 6
ARROW_LENGTH = 30
DEFAULT_CANVAS_WIDTH = 6400
DEFAULT_CANVAS_HEIGHT = 1400
LAYOUT_PAD = 30
REACTION_STEP_X = 800
REACTION_ROW_HEIGHT = 700
REACTION_PROTEIN_W = 150
REACTION_PROTEIN_H = 70


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


def _elbow_edge_path(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[str, int, int, int, int]:
    mid_y = int(round((y1 + y2) / 2))
    path = f"M{x1} {y1} L{x1} {mid_y} L{x2} {mid_y} L{x2} {y2}"
    return path, x1, mid_y, x2, mid_y


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
    opts: Dict[str, Any] = {"start_arrow": True, "start_flat_arrow": False, "end_arrow": False, "end_flat_arrow": False}
    arrow_path = _compute_arrow_path(x1, y1, cp1x, cp1y)
    if arrow_path is not None:
        opts["start_arrow_path"] = arrow_path
    edge["options"] = json.dumps(opts)


def _add_end_arrow(edge: Dict[str, Any], x2: int, y2: int, cp2x: int, cp2y: int) -> None:
    opts: Dict[str, Any] = {"end_arrow": True, "end_flat_arrow": False, "start_arrow": False, "start_flat_arrow": False}
    arrow_path = _compute_arrow_path(x2, y2, cp2x, cp2y)
    if arrow_path is not None:
        opts["end_arrow_path"] = arrow_path
    edge["options"] = json.dumps(opts)


def _add_no_arrow(edge: Dict[str, Any]) -> None:
    opts: Dict[str, Any] = {"end_arrow": False, "end_flat_arrow": False, "start_arrow": False, "start_flat_arrow": False}
    edge["options"] = json.dumps(opts)


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


def get_rect_anchor_toward_target(
    loc: Dict[str, Any],
    target_x: int,
    target_y: int,
    mode: str = "auto",
) -> Tuple[int, int]:
    x = int(loc["x"])
    y = int(loc["y"])
    width = int(loc["width"])
    height = int(loc["height"])
    center_x = x + width / 2
    center_y = y + height / 2

    if mode == "horizontal":
        return x + (width if target_x >= center_x else 0), int(round(center_y))
    if mode == "vertical":
        return int(round(center_x)), y + (height if target_y >= center_y else 0)
    if mode != "auto":
        raise ValueError(f"Unsupported rectangle anchor mode: {mode}")

    dx = target_x - center_x
    dy = target_y - center_y
    half_w = width / 2
    half_h = height / 2

    if dx == 0 and dy == 0:
        return int(round(center_x)), y
    if dx == 0:
        return int(round(center_x)), y + (height if dy > 0 else 0)
    if dy == 0:
        return x + (width if dx > 0 else 0), int(round(center_y))

    scale = min(half_w / abs(dx), half_h / abs(dy))
    return int(round(center_x + dx * scale)), int(round(center_y + dy * scale))


def _reaction_visual_side(layout: Dict[str, int], substrate_side: bool) -> str:
    ltr = int(layout.get("row-dir", 1)) >= 0
    if substrate_side:
        return "Left" if ltr else "Right"
    return "Right" if ltr else "Left"


def _reaction_side_anchor_x(
    visual_side: str,
    gap: int,
    enzyme_left_x: int,
    enzyme_right_x: int,
) -> int:
    return enzyme_left_x - gap if visual_side == "Left" else enzyme_right_x + gap


def _reaction_member_x_for_anchor(anchor_x: int, width: int, visual_side: str) -> int:
    return anchor_x - width if visual_side == "Left" else anchor_x


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
        "protein_complex-proteins": "protein-complex-protein",
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
    }








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


def _reaction_cols_for_region_width(
    region_w: int,
    *,
    pad: int = LAYOUT_PAD,
    rxn_step_x: int = REACTION_STEP_X,
) -> int:
    return max(1, (max(0, int(region_w)) - 2 * pad) // rxn_step_x)


def _reaction_row_count(
    n_rxns: int,
    region_w: int,
    *,
    pad: int = LAYOUT_PAD,
    rxn_step_x: int = REACTION_STEP_X,
) -> int:
    cols = _reaction_cols_for_region_width(region_w, pad=pad, rxn_step_x=rxn_step_x)
    return max(1, math.ceil(max(0, n_rxns) / cols))


def _reaction_region_height(
    n_rxns: int,
    region_w: int,
    *,
    pad: int = LAYOUT_PAD,
    rxn_step_x: int = REACTION_STEP_X,
    row_height: int = REACTION_ROW_HEIGHT,
) -> int:
    if n_rxns <= 0:
        return 0
    return _reaction_row_count(n_rxns, region_w, pad=pad, rxn_step_x=rxn_step_x) * row_height + 2 * pad


def _serpentine_reaction_position(
    region: Dict[str, Any],
    index_in_state: int,
    *,
    pad: int = LAYOUT_PAD,
    rxn_step_x: int = REACTION_STEP_X,
    row_height: int = REACTION_ROW_HEIGHT,
    protein_w: int = REACTION_PROTEIN_W,
    protein_h: int = REACTION_PROTEIN_H,
) -> Dict[str, int]:
    cols = _reaction_cols_for_region_width(int(region["w"]), pad=pad, rxn_step_x=rxn_step_x)
    row = index_in_state // cols
    col_in_row = index_in_state % cols
    if row % 2 == 0:
        enzyme_cx = int(region["x"]) + pad + rxn_step_x // 2 + col_in_row * rxn_step_x
    else:
        enzyme_cx = int(region["x"]) + int(region["w"]) - pad - rxn_step_x // 2 - col_in_row * rxn_step_x
    enzyme_cy = int(region["y"]) + pad + row_height // 2 + row * row_height
    return {
        "row": row,
        "row-dir": 1 if row % 2 == 0 else -1,
        "enzyme-cx": enzyme_cx,
        "enzyme-cy": enzyme_cy,
        "enzyme-x": enzyme_cx - protein_w // 2,
        "enzyme-y": enzyme_cy - protein_h // 2,
        "compound-stack-cy": enzyme_cy,
    }


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
        if not is_pwml_ir(extraction):
            raise ValueError("DeterministicPwmlBuilder requires validated PWML IR input.")
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



    def _protein_complex_members(
        self,
        components: Any,
        *,
        complex_name: str,
        protein_id_by_key: Dict[str, int],
        protein_id_by_name: Dict[str, int],
        protein_id_by_db_id: Dict[int, int],
        protein_id_by_uniprot: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        members: List[Dict[str, Any]] = []
        for idx, component in enumerate(components if isinstance(components, list) else []):
            stoich = _component_stoichiometry(component)
            if stoich is None:
                raise ValueError(f"Protein complex '{complex_name}' component[{idx}] is missing stoichiometry.")

            protein_id: Optional[int] = None
            if isinstance(component, dict):
                protein_key = str(component.get("protein_key") or component.get("entity_key") or "").strip()
                if protein_key:
                    protein_id = protein_id_by_key.get(protein_key)
                if protein_id is None:
                    for db_key in ["pathbank_protein_id", "pw_protein_id", "pathwhiz_id", "protein_id"]:
                        db_id = _to_positive_int(component.get(db_key))
                        if db_id is not None:
                            protein_id = protein_id_by_db_id.get(db_id)
                        if protein_id is not None:
                            break
                if protein_id is None:
                    mapped_ids = component.get("mapped_ids") if isinstance(component.get("mapped_ids"), dict) else {}
                    uniprot = str(
                        component.get("uniprot")
                        or component.get("uniprot_id")
                        or component.get("uniprot-id")
                        or mapped_ids.get("uniprot")
                        or mapped_ids.get("uniprot_id")
                        or ""
                    ).strip().casefold()
                    if uniprot:
                        protein_id = protein_id_by_uniprot.get(uniprot)

            name = _component_name(component)
            if protein_id is None and name:
                protein_id = protein_id_by_name.get(_normalize_key(name))

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
        species_id_by_name: Dict[str, int] = {}
        species_id_by_external_id: Dict[str, int] = {}
        for record in ir.get("species", []) if isinstance(ir.get("species"), list) else []:
            if not isinstance(record, dict):
                continue
            rid = id_for(record, ["pathwhiz_id", "pathbank_species_id", "pw_species_id"], self.ids)
            remember("species", record.get("key"), rid)
            if record.get("name"):
                species_id_by_name[_normalize_key(str(record.get("name")))] = rid
            for species_id_key in ("pathwhiz_id", "pathbank_species_id", "pw_species_id", "species_id", "taxonomy_id"):
                species_id_value = record.get(species_id_key)
                if species_id_value not in (None, ""):
                    species_id_by_external_id[str(species_id_value).strip().casefold()] = rid
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

        def protein_species_id(record: Dict[str, Any]) -> Optional[int]:
            resolved = lookup("species", record.get("species_key"))
            if resolved is not None:
                return resolved
            species_ref = record.get("species_ref") if isinstance(record.get("species_ref"), dict) else {}
            resolved = lookup("species", species_ref.get("key"))
            if resolved is not None:
                return resolved
            candidates: List[Any] = [
                record.get("species"), record.get("organism"), record.get("species_id"),
                record.get("pathbank_species_id"), record.get("taxonomy_id"), species_ref.get("name"),
                species_ref.get("species_id"), species_ref.get("pathbank_species_id"), species_ref.get("taxonomy_id"),
            ]
            mapping_meta = record.get("mapping_meta") if isinstance(record.get("mapping_meta"), dict) else {}
            candidates.extend([mapping_meta.get("species"), mapping_meta.get("species_id")])
            for candidate in candidates:
                if isinstance(candidate, dict):
                    candidate = candidate.get("name") or candidate.get("id")
                if candidate in (None, ""):
                    continue
                text = str(candidate).strip()
                resolved = species_id_by_name.get(_normalize_key(text))
                if resolved is None:
                    resolved = species_id_by_external_id.get(text.casefold())
                if resolved is not None:
                    return resolved
            return default_species_id

        self.section_items["proteins"] = []
        protein_id_by_key: Dict[str, int] = {}
        protein_id_by_name: Dict[str, int] = {}
        protein_id_by_db_id: Dict[int, int] = {}
        protein_id_by_uniprot: Dict[str, int] = {}
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
            mapped_pathbank_id = _to_positive_int(mapped_ids.get("pathbank_protein_id"))
            if mapped_pathbank_id is not None:
                protein_id_by_db_id[mapped_pathbank_id] = rid
            uniprot_id = (
                mapped_ids.get("uniprot")
                or record.get("uniprot")
                or record.get("uniprot_id")
                or record.get("uniprot-id")
                or None
            )
            if uniprot_id:
                protein_id_by_uniprot[str(uniprot_id).strip().casefold()] = rid
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
                    "species-id": protein_species_id(record),
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
                protein_id_by_uniprot=protein_id_by_uniprot,
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
        layout_rxn_step_x = REACTION_STEP_X
        layout_protein_w, layout_protein_h = REACTION_PROTEIN_W, REACTION_PROTEIN_H
        layout_protein_gap_x = 10

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

        existing_region_by_key: Dict[str, Dict[str, int]] = {}
        for bound in ir.get("bound_visualizations", []) if isinstance(ir.get("bound_visualizations"), list) else []:
            if not isinstance(bound, dict):
                continue
            biological_state_key = str(bound.get("biological_state_key") or "")
            if biological_state_key:
                existing_region_by_key[biological_state_key] = {
                    "x": int(bound.get("x") or 0),
                    "y": int(bound.get("y") or 0),
                    "w": int(bound.get("width") or self.args.width),
                    "h": int(bound.get("height") or self.args.height),
                }

        state_region_by_key: Dict[str, Dict[str, int]] = {}
        state_keys = [
            str(state.get("key") or "")
            for state in ir.get("biological_states", [])
            if isinstance(state, dict) and state.get("key") is not None
        ]
        y = 0
        for state_key in state_keys:
            existing_region = existing_region_by_key.get(
                state_key,
                {"x": 0, "y": y, "w": int(self.args.width), "h": int(self.args.height) // max(1, len(state_keys))},
            )
            n_rxns = len(reaction_keys_by_state.get(state_key, []))
            region_h = max(
                int(existing_region.get("h", 0)),
                _reaction_region_height(n_rxns, int(self.args.width)),
            )
            state_region_by_key[state_key] = {
                "x": 0,
                "y": y,
                "w": int(self.args.width),
                "h": region_h,
            }
            y += region_h
        if y < int(self.args.height) and state_keys:
            state_region_by_key[state_keys[-1]]["h"] += int(self.args.height) - y
            y = int(self.args.height)
        if y > int(self.args.height):
            self.args.height = y

        biological_state_id_by_key = {
            state_key: lookup("biological_states", state_key)
            for state_key in state_keys
        }
        for bound in self.section_items.get("bound-visualizations", []):
            bound_bs_id = bound.get("biological-state-id")
            for state_key, bs_id in biological_state_id_by_key.items():
                if bs_id != bound_bs_id:
                    continue
                region = state_region_by_key.get(state_key)
                if region:
                    bound["x"] = region["x"]
                    bound["y"] = region["y"]
                    bound["width"] = str(region["w"])
                    bound["height"] = str(region["h"])
                break

        def ir_region_for_state(state_key: str) -> Dict[str, int]:
            return state_region_by_key.get(
                state_key,
                {"x": 0, "y": 0, "w": int(self.args.width), "h": int(self.args.height)},
            )

        reaction_layout_by_key: Dict[str, Dict[str, int]] = {}
        for biological_state_key, reaction_keys in reaction_keys_by_state.items():
            region = ir_region_for_state(biological_state_key)
            for k, reaction_key in enumerate(reaction_keys):
                reaction_layout_by_key[reaction_key] = _serpentine_reaction_position(
                    region,
                    k,
                    pad=LAYOUT_PAD,
                    rxn_step_x=layout_rxn_step_x,
                    protein_w=layout_protein_w,
                    protein_h=layout_protein_h,
                )

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
        row_transition_source_by_member_key: Dict[str, str] = {}

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
                is_substrate_side = side_key == "left"
                side = _reaction_visual_side(layout, is_substrate_side)
                gap = layout_substrate_gap if is_substrate_side else layout_product_gap
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
                            producer_layout = reaction_layout_by_key.get(reaction_sequence[producer_idx][0], layout)
                            producer_side = _reaction_visual_side(producer_layout, False)
                            shared_loc_id = compound_loc_by_rxn_side.get(
                                (compound_id, state_key, producer_idx, producer_side)
                            )
                            if shared_loc_id is not None:
                                member_key = str(member.get("key") or "")
                                if member_key:
                                    reaction_member_location_by_key[member_key] = shared_loc_id
                                    row_transition_source_by_member_key[member_key] = reaction_sequence[producer_idx][0]
                                compound_loc_by_rxn_side[(compound_id, state_key, reaction_idx, side)] = shared_loc_id
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
                    expected_anchor_x = _reaction_side_anchor_x(
                        side,
                        gap,
                        enzyme_left_x,
                        enzyme_right_x,
                    )
                    expected_anchor_y = y + height // 2
                    x = _reaction_member_x_for_anchor(expected_anchor_x, width, side)
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

        def loc_side_connection_point(loc_id: int, side: str) -> Optional[Tuple[int, int]]:
            loc = loc_by_id.get(loc_id)
            if not loc:
                return None
            x = int(loc["x"])
            if side == "Left":
                x += int(loc["width"])
            return x, int(loc["y"]) + int(loc["height"]) // 2

        def loc_connection_point(
            loc_id: int,
            target_x: int,
            target_y: int,
            mode: str = "auto",
        ) -> Optional[Tuple[int, int]]:
            loc = loc_by_id.get(loc_id)
            if not loc:
                return None
            return get_rect_anchor_toward_target(loc, target_x, target_y, mode=mode)

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
                substrate_side = _reaction_visual_side(layout or {"row-dir": 1}, True)
                for member in viz.get("members", []) if isinstance(viz.get("members"), list) else []:
                    if not isinstance(member, dict) or str(member.get("role") or "") != "left":
                        continue
                    loc_id = reaction_member_location_by_key.get(str(member.get("process_member_key") or ""))
                    if loc_id is None:
                        loc_id = lookup("locations", member.get("location_key"))
                    if loc_id is None:
                        continue
                    point = loc_side_connection_point(loc_id, substrate_side)
                    if point is not None:
                        if substrate_side == "Left":
                            enzyme_left = point[0] + layout_substrate_gap
                        else:
                            enzyme_left = point[0] - layout_substrate_gap - 70
                        enzyme_right = enzyme_left + 70
                        break
            no_enzyme = enzyme_box is None
            target_box = {
                "x": enzyme_left,
                "y": enzyme_cy - (
                    enzyme_box[3] // 2 if enzyme_box is not None else layout_protein_h // 2
                ),
                "width": str(
                    enzyme_box[2] if enzyme_box is not None else enzyme_right - enzyme_left
                ),
                "height": str(enzyme_box[3] if enzyme_box is not None else layout_protein_h),
            }
            for member in viz.get("members", []) if isinstance(viz.get("members"), list) else []:
                if not isinstance(member, dict):
                    continue
                role = str(member.get("role") or "")
                is_substrate_side = role == "left"
                is_product_side = role == "right"
                if not is_substrate_side and not is_product_side:
                    continue
                layout = reaction_layout_by_key.get(reaction_key, {"row-dir": 1})
                side = _reaction_visual_side(layout, is_substrate_side)
                target = (
                    (enzyme_left, enzyme_cy)
                    if side == "Left"
                    else (enzyme_right, enzyme_cy)
                )
                loc_id = reaction_member_location_by_key.get(str(member.get("process_member_key") or ""))
                if loc_id is None:
                    loc_id = lookup("locations", member.get("location_key"))
                edge_id = lookup("edges", member.get("edge_key"))
                if loc_id is None or edge_id is None:
                    continue
                member_key = str(member.get("process_member_key") or "")
                source_reaction_key = row_transition_source_by_member_key.get(member_key)
                source_layout = (
                    reaction_layout_by_key.get(source_reaction_key)
                    if source_reaction_key is not None
                    else None
                )
                target_layout = reaction_layout_by_key.get(reaction_key)
                is_row_transition = (
                    source_layout is not None
                    and target_layout is not None
                    and int(source_layout.get("row", -1)) != int(target_layout.get("row", -1))
                    and not no_enzyme
                )
                point = loc_connection_point(
                    loc_id,
                    target[0],
                    target[1],
                    mode="vertical" if is_row_transition else "auto",
                )
                edge = edge_by_id.get(edge_id)
                if point is None or edge is None:
                    continue
                px, py = point
                edge["visualization-template-id"] = 5
                edge.pop("options", None)
                if no_enzyme and is_substrate_side:
                    x1, y1, x2, y2 = target[0], target[1], target[0], target[1]
                    edge["hidden"] = True
                elif no_enzyme:
                    x1, y1, x2, y2 = px, py, target[0], target[1]
                    edge["hidden"] = False
                elif is_row_transition:
                    target_anchor = get_rect_anchor_toward_target(target_box, px, py, mode="vertical")
                    x1, y1, x2, y2 = px, py, target_anchor[0], target_anchor[1]
                    edge["hidden"] = False
                else:
                    x1, y1, x2, y2 = px, py, target[0], target[1]
                    edge["hidden"] = False
                if is_row_transition:
                    edge["path"], cp1x, cp1y, cp2x, cp2y = _elbow_edge_path(x1, y1, x2, y2)
                else:
                    edge["path"], cp1x, cp1y, cp2x, cp2y = _curved_edge_path(x1, y1, x2, y2)
                if is_substrate_side:
                    _add_no_arrow(edge)
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
                    layout = reaction_layout_by_key.get(str(viz.get("process_key") or ""), {"row-dir": 1})
                    side = (
                        _reaction_visual_side(layout, True)
                        if role == "left"
                        else _reaction_visual_side(layout, False)
                        if role == "right"
                        else ""
                    )
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
        first_species_id = self._ir_pathway_species_id
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
        counts = self._populate_sections_from_ir()

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


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pwml_pipeline_export(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")
    payload, export_normalization_report = normalize_process_payload(payload)

    export_normalization_report_path = out_dir / "pwml_export_normalization_report.json"
    _write_json(export_normalization_report_path, export_normalization_report)
    normalization_gate = export_normalization_report.get("gate")
    if not isinstance(normalization_gate, dict) or normalization_gate.get("ok") is not True:
        return {
            "ok": False,
            "export_normalization_report": str(export_normalization_report_path),
            "error": "Post-normalization gate failed before PWML export.",
        }

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
            "export_normalization_report": str(export_normalization_report_path),
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
        "export_normalization_report": str(export_normalization_report_path),
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
    parser.add_argument("--width", type=int, default=DEFAULT_CANVAS_WIDTH, help="PWML canvas width")
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


def main() -> None:
    pwml_pipeline_cli_main()


if __name__ == "__main__":
    main()
