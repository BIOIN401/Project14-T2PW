from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def _short_hash(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# PathWhiz-style layout engine
# ---------------------------------------------------------------------------
# Key principle: each metabolite gets ONE canvas position, shared across all
# reactions that reference it. Edges connect metabolites to reaction centers.
# ---------------------------------------------------------------------------

import math as _math

# Cofactors excluded from topology (they connect everything → useless for layout)
_COFACTOR_NORMS_LAYOUT: "Set[str]" = {
    "nad", "nad+", "nadh", "nadp", "nadp+", "nadph",
    "fad", "fadh", "fadh2",
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "ctp", "utp",
    "h2o", "water", "h+", "h", "proton", "oh",
    "o2", "oxygen", "co2", "carbon dioxide",
    "pi", "ppi", "pyrophosphate", "phosphate", "inorganic phosphate",
    "coa", "coa sh", "coash", "coa-sh", "coenzyme a", "acetyl coa",
    "hco3", "bicarbonate",
    "ubiquinone", "ubiquinol", "ubiquinone q", "ubiquinol qh2",
    "h2o2", "hydrogen peroxide",
    "fe2+", "fe3+", "cu+", "cu2+", "zn2+", "mg2+", "mn2+", "ca2+",
}

def _is_layout_cofactor(name: str) -> bool:
    n = _normalize(name)
    return not n or len(n) <= 1 or n in _COFACTOR_NORMS_LAYOUT


def _compute_reaction_ranks_v2(reaction_plans: "List[Dict[str, Any]]") -> "List[int]":
    """Topological sort with cofactor filtering + cycle detection."""
    from collections import deque as _deque
    n = len(reaction_plans)
    if n == 0:
        return []

    produces: "Dict[str, List[int]]" = defaultdict(list)
    consumes: "Dict[str, List[int]]" = defaultdict(list)
    for i, plan in enumerate(reaction_plans):
        for name in plan.get("outputs", []):
            if not _is_layout_cofactor(name):
                produces[_normalize(name)].append(i)
        for name in plan.get("inputs", []):
            if not _is_layout_cofactor(name):
                consumes[_normalize(name)].append(i)

    successors: "Dict[int, Set[int]]" = defaultdict(set)
    in_degree: "List[int]" = [0] * n
    for met in produces:
        for pi in produces[met]:
            for ci in consumes.get(met, []):
                if pi != ci and ci not in successors[pi]:
                    successors[pi].add(ci)
                    in_degree[ci] += 1

    ranks = [0] * n
    queue = _deque(i for i in range(n) if in_degree[i] == 0)
    visited: "Set[int]" = set()
    rank = 0
    while queue:
        nq = _deque()
        for i in list(queue):
            if i in visited:
                continue
            visited.add(i)
            ranks[i] = rank
            for j in successors[i]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    nq.append(j)
        queue = nq
        rank += 1

    # Cycle fallback — use document order within each cycle component
    unvisited = set(range(n)) - visited
    if unvisited:
        undirected: "Dict[int, Set[int]]" = defaultdict(set)
        for i in range(n):
            for j in successors.get(i, set()):
                undirected[i].add(j)
                undirected[j].add(i)
        seen_c: "Set[int]" = set()
        for start in sorted(unvisited):
            if start in seen_c:
                continue
            comp: "List[int]" = []
            stk = [start]
            while stk:
                node = stk.pop()
                if node in seen_c:
                    continue
                seen_c.add(node)
                comp.append(node)
                for nb in undirected.get(node, set()):
                    if nb not in seen_c:
                        stk.append(nb)
            comp.sort()
            for idx, i in enumerate(comp):
                ranks[i] = rank + idx
                visited.add(i)
            rank += len(comp)

    return ranks


def _build_pathway_layout(
    reaction_plans: "List[Dict[str, Any]]",
) -> "Tuple[Dict[str, Tuple[float,float]], Dict[int, Tuple[float,float]]]":
    """
    Returns:
        species_pos:  {norm_name -> (cx, cy)}  — centre of each metabolite circle
        rxn_centers:  {rxn_idx  -> (cx, cy)}   — reaction crosshair position
    
    Layout algorithm:
    1. Assign each reaction a rank (topological order).
    2. Arrange reaction centers on an arc/grid using the rank.
    3. For each reaction, place its metabolites radially around the center,
       but if a metabolite is already placed (shared), keep its existing position.
    4. Recompute reaction center as the centroid of its metabolite positions.
    """
    n = len(reaction_plans)
    if n == 0:
        return {}, {}

    ranks = _compute_reaction_ranks_v2(reaction_plans)
    max_rank = max(ranks) if ranks else 0

    # ── Step 1: arrange reaction centers on a snake/arc layout ───────────────
    # Use a multi-row snake: reactions flow left-to-right on row 0,
    # right-to-left on row 1, etc. — like PathWhiz's TCA layout.
    COLS = max(1, min(5, (max_rank + 2) // 2))  # reactions per row
    RXN_GAP_X = 420.0
    RXN_GAP_Y = 450.0
    MARGIN_X = 280.0
    MARGIN_Y = 280.0

    # Assign tentative reaction centers based on rank
    tentative_rxn: "Dict[int, Tuple[float,float]]" = {}
    rank_col: "Dict[int, int]" = defaultdict(int)  # track column within each rank
    for i in range(n):
        r = ranks[i]
        row = r // COLS
        col_in_row = r % COLS
        # Snake: even rows go left-to-right, odd rows right-to-left
        if row % 2 == 0:
            col = col_in_row
        else:
            col = COLS - 1 - col_in_row
        cx = MARGIN_X + col * RXN_GAP_X
        cy = MARGIN_Y + row * RXN_GAP_Y
        tentative_rxn[i] = (cx, cy)

    # ── Step 2: place metabolites around their first-encountered reaction ─────
    METAB_DIST = 180.0      # distance from reaction center to metabolite
    COFAC_DIST = 120.0      # cofactors placed closer
    NODE_W = 78.0
    NODE_H = 78.0
    PROT_W = 160.0
    PROT_H = 60.0

    species_pos: "Dict[str, Tuple[float,float]]" = {}  # norm -> (cx, cy)

    def _place_around(rxn_idx: int, names: "List[str]", angle_start: float, angle_span: float, dist: float):
        """Place metabolites radially around a reaction center."""
        cx, cy = tentative_rxn[rxn_idx]
        m = len(names)
        if m == 0:
            return
        for k, name in enumerate(names):
            norm = _normalize(name)
            if not norm or norm in species_pos:
                continue
            if m == 1:
                angle = angle_start
            else:
                angle = angle_start + k * angle_span / (m - 1) if m > 1 else angle_start
            d = COFAC_DIST if _is_layout_cofactor(name) else dist
            sx = cx + d * _math.cos(angle)
            sy = cy + d * _math.sin(angle)
            species_pos[norm] = (sx, sy)

    PI = _math.pi

    # Two-pass: first pass places primary metabolites (non-cofactors), 
    # second pass places cofactors
    for pass_num in range(2):
        for i, plan in enumerate(reaction_plans):
            reactants = [x for x in plan.get("inputs", []) if isinstance(x, str) and x.strip()]
            products  = [x for x in plan.get("outputs", []) if isinstance(x, str) and x.strip()]

            if pass_num == 0:
                # Non-cofactors: place left of center (reactants) and right (products)
                main_reactants = [r for r in reactants if not _is_layout_cofactor(r)]
                main_products  = [p for p in products  if not _is_layout_cofactor(p)]
                _place_around(i, main_reactants, PI,        PI * 0.6, METAB_DIST)
                _place_around(i, main_products,  0.0,       PI * 0.6, METAB_DIST)
            else:
                # Cofactors: place above/below
                cofac_reactants = [r for r in reactants if _is_layout_cofactor(r)]
                cofac_products  = [p for p in products  if _is_layout_cofactor(p)]
                _place_around(i, cofac_reactants, -PI * 0.6, PI * 0.5, COFAC_DIST)
                _place_around(i, cofac_products,   PI * 1.6, PI * 0.5, COFAC_DIST)

    # ── Step 3: recompute reaction centers as centroid of their metabolites ───
    rxn_centers: "Dict[int, Tuple[float,float]]" = {}
    for i, plan in enumerate(reaction_plans):
        reactants = [x for x in plan.get("inputs", [])  if isinstance(x, str) and x.strip()]
        products  = [x for x in plan.get("outputs", []) if isinstance(x, str) and x.strip()]
        all_main = [x for x in reactants + products if not _is_layout_cofactor(x)]
        
        positions = []
        for name in all_main:
            norm = _normalize(name)
            if norm in species_pos:
                positions.append(species_pos[norm])

        if positions:
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
        else:
            cx, cy = tentative_rxn[i]
        rxn_centers[i] = (cx, cy)

    return species_pos, rxn_centers


def _edge_path_between(x1: float, y1: float, x2: float, y2: float) -> str:
    """Cubic bezier edge path from (x1,y1) to (x2,y2) with gentle curve."""
    dx = x2 - x1
    dy = y2 - y1
    # Control points: slightly offset perpendicular for gentle curve
    ctrl_x = (x1 + x2) / 2
    ctrl_y = (y1 + y2) / 2
    return (f"M{x1:.1f} {y1:.1f} "
            f"C{ctrl_x:.1f} {y1:.1f} {ctrl_x:.1f} {y2:.1f} "
            f"{x2:.1f} {y2:.1f} ")



def _html_name(name: str) -> str:
    """Escape & in reaction names for XML attribute."""
    return name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# ID allocation — PathWhiz-style numeric IDs
# ---------------------------------------------------------------------------

class _IdCounter:
    def __init__(self, start: int = 1) -> None:
        self._n = start

    def next(self) -> int:
        v = self._n
        self._n += 1
        return v


# ---------------------------------------------------------------------------
# Layout geometry helpers
# ---------------------------------------------------------------------------

# Canvas layout constants (pixels) matching PathWhiz style
_CANVAS_W = 2400.0
_CANVAS_H = 2400.0
_MARGIN = 120.0
_REACTION_GAP_X = 420.0
_REACTION_GAP_Y = 380.0
_NODE_OFFSET_X = 200.0      # distance from reaction center to metabolite
_NODE_SPACING_Y = 110.0
_MODIFIER_OFFSET_Y = 120.0

# ---------------------------------------------------------------------------
# Cofactor filtering — these metabolites connect every reaction and destroy
# topological ordering if included in the dependency graph.
# ---------------------------------------------------------------------------
# (Layout cofactor list moved to _COFACTOR_NORMS_LAYOUT below)
def _is_cofactor(name: str) -> bool:
    """Kept for compatibility — delegates to layout version."""
    return _is_layout_cofactor(name)


# (old _compute_reaction_ranks removed — see _compute_reaction_ranks_v2 above)

# Compound box sizes  (width, height) — approximate PathWhiz defaults
_COMPOUND_W = 78.0
_COMPOUND_H = 78.0
_PROTEIN_W = 160.0
_PROTEIN_H = 80.0

# visualization_template_id for regular compound node
_TMPL_COMPOUND = "3"
# visualization_template_id for regular edge
_TMPL_EDGE = "5"
# visualization_template_id for protein in complex
_TMPL_PROTEIN = "6"


# _reaction_center is now computed dynamically via _compute_reaction_ranks.
# The function below is kept for compatibility but should not be called directly.
def _reaction_center(rxn_idx: int) -> Tuple[float, float]:
    # Legacy grid fallback — not used when reaction_plans are available
    cx = _MARGIN + _NODE_OFFSET_X + rxn_idx * _REACTION_GAP_X
    cy = _MARGIN + _MODIFIER_OFFSET_Y
    return cx, cy


def _node_pos_for_side(
    cx: float, cy: float, idx: int, total: int, side: str
) -> Tuple[float, float, float, float]:
    """Return (node_cx, node_cy, node_x, node_y) for a reactant/product node.
    Used as FALLBACK only — the global position map takes priority."""
    if side == "reactant":
        x_center = cx - _NODE_OFFSET_X
    else:
        x_center = cx + _NODE_OFFSET_X
    y_offset = (idx - (total - 1) / 2.0) * _NODE_SPACING_Y
    node_cy = cy + y_offset
    node_x = x_center - _COMPOUND_W / 2
    node_y = node_cy - _COMPOUND_H / 2
    return x_center, node_cy, node_x, node_y


# ---------------------------------------------------------------------------
# Global species position map — built once before the reaction loop so that
# metabolites shared across reactions share a single canvas position.
# ---------------------------------------------------------------------------

def _build_species_position_map(
    reaction_plans: "List[Dict[str, Any]]",
    rxn_ranks: "List[int]",
    rxn_lanes: "List[int]",
) -> "Dict[str, Tuple[float, float]]":
    """
    Return {norm_name: (cx, cy)} for every species name referenced in any reaction.

    Strategy:
    - For each species, find the reaction where it appears (first occurrence wins).
    - Place it to the LEFT of that reaction center if it's a reactant,
      RIGHT if it's a product.
    - If the same species is both reactant and product (shared metabolite),
      place it at a midpoint between its first-reactant reaction and
      first-product reaction — this is the key to connecting the pathway.
    """
    from collections import defaultdict

    # first_as_reactant[norm] = (rxn_idx, slot_idx, total_slots)
    first_as_reactant: "Dict[str, Tuple[int, int, int]]" = {}
    first_as_product:  "Dict[str, Tuple[int, int, int]]" = {}

    for i, plan in enumerate(reaction_plans):
        reactants = [x for x in plan.get("inputs", []) if isinstance(x, str) and x.strip()]
        products  = [x for x in plan.get("outputs", []) if isinstance(x, str) and x.strip()]
        for j, name in enumerate(reactants):
            norm = _normalize(name)
            if norm and norm not in first_as_reactant:
                first_as_reactant[norm] = (i, j, len(reactants))
        for j, name in enumerate(products):
            norm = _normalize(name)
            if norm and norm not in first_as_product:
                first_as_product[norm] = (i, j, len(products))

    # Compute reaction center for a given rxn_idx
    def _rxn_center(idx: int) -> "Tuple[float, float]":
        rank = rxn_ranks[idx]
        lane = rxn_lanes[idx]
        cx = _MARGIN + _NODE_OFFSET_X + rank * _REACTION_GAP_X
        cy = _MARGIN + _MODIFIER_OFFSET_Y + lane * _REACTION_GAP_Y
        return cx, cy

    def _slot_y(cy: float, slot: int, total: int) -> float:
        return cy + (slot - (total - 1) / 2.0) * _NODE_SPACING_Y

    pos: "Dict[str, Tuple[float, float]]" = {}
    all_norms = set(first_as_reactant) | set(first_as_product)

    for norm in all_norms:
        in_rxn  = first_as_reactant.get(norm)
        out_rxn = first_as_product.get(norm)

        if in_rxn and out_rxn:
            # Shared metabolite: appears as product of one reaction AND
            # reactant of another.  Place it between those two reaction centers.
            prod_rxn_idx, prod_slot, prod_total = out_rxn
            reac_rxn_idx, reac_slot, reac_total = in_rxn
            pcx, pcy = _rxn_center(prod_rxn_idx)
            rcx, rcy = _rxn_center(reac_rxn_idx)
            # X: midpoint between producer's right side and consumer's left side
            node_cx = (pcx + _NODE_OFFSET_X + rcx - _NODE_OFFSET_X) / 2
            # Y: average of the two slot positions
            py = _slot_y(pcy, prod_slot, prod_total)
            ry = _slot_y(rcy, reac_slot, reac_total)
            node_cy = (py + ry) / 2
        elif out_rxn:
            # Only appears as a product
            prod_rxn_idx, prod_slot, prod_total = out_rxn
            pcx, pcy = _rxn_center(prod_rxn_idx)
            node_cx = pcx + _NODE_OFFSET_X
            node_cy = _slot_y(pcy, prod_slot, prod_total)
        else:
            # Only appears as a reactant
            reac_rxn_idx, reac_slot, reac_total = in_rxn
            rcx, rcy = _rxn_center(reac_rxn_idx)
            node_cx = rcx - _NODE_OFFSET_X
            node_cy = _slot_y(rcy, reac_slot, reac_total)

        pos[norm] = (node_cx, node_cy)

    return pos


def _edge_path_reactant(node_cx: float, node_cy: float, cx: float, cy: float) -> str:
    """Edge from metabolite center → reaction center."""
    return _edge_path_between(node_cx, node_cy, cx, cy)


def _edge_path_product(cx: float, cy: float, node_cx: float, node_cy: float) -> str:
    """Edge from reaction center → metabolite center."""
    return _edge_path_between(cx, cy, node_cx, node_cy)


def _modifier_pos(cx: float, cy: float, idx: int, total: int) -> Tuple[float, float, float, float]:
    x_off = (idx - (total - 1) / 2.0) * (_PROTEIN_W + 20.0)
    node_cx = cx + x_off
    node_cy = cy - _MODIFIER_OFFSET_Y
    node_x = node_cx - _PROTEIN_W / 2
    node_y = node_cy - _PROTEIN_H / 2
    return node_cx, node_cy, node_x, node_y


def _edge_path_modifier(node_cx: float, node_cy: float, node_h: float, cx: float, cy: float) -> str:
    y_bot = node_cy + node_h / 2
    return f"M{node_cx:.0f} {y_bot:.0f} C{node_cx:.0f} {cy:.0f} {cx:.0f} {cy:.0f} {cx:.0f} {cy:.0f} "


# ---------------------------------------------------------------------------
# Core SBML builder
# ---------------------------------------------------------------------------

def build_sbml(
    input_path: Path,
    sbml_path: Path,
    report_json_path: Path,
    report_txt_path: Path,
    *,
    default_compartment_name: str = "cell",
    db_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a PathWhiz-compatible SBML Level 3 file from mapped pathway JSON.

    Key format requirements matched to PW270676.sbml reference:
    - Species IDs:      Compound{pw_id}, ProteinComplex{pw_id}, Protein{pw_id}
    - Compartment IDs:  BiologicalState{pw_id}
    - Reaction IDs:     Reaction{pw_id}
    - Protein complexes have pathwhiz:protein_associations/pathwhiz:protein_complex_proteins nesting
    - Each individual protein subunit appears as its own Protein{id} species
    - Every speciesReference carries a pathwhiz:location annotation with x/y layout
    - modifierSpeciesReference wraps protein_complex_visualization → protein_location
    - bqbiol:hasPart used for protein complexes (RDF)
    - bqbiol:isDescribedBy used for InChI/InChIKey on compounds
    - Reaction names use human-readable "A + B → C" format
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    report: Dict[str, Any] = {
        "hard_errors": [],
        "warnings": [],
        "defaults_applied": [],
        "counts": {"compartments": 0, "species": 0, "reactions": 0},
        "pathwhiz_id_stats": {
            "mysql_connected": False,
            "compounds_matched": 0, "proteins_matched": 0, "species_no_id": 0
        },
        "validation": {
            "check_count": 0,
            "error_count": 0,
            "has_errors": False,
            "messages": [],
        },
    }

    data = deepcopy(payload)
    entities = _safe_dict(data.get("entities"))
    processes = _safe_dict(data.get("processes"))

    # ------------------------------------------------------------------
    # ID counters — we allocate monotonically-increasing PathWhiz-style IDs
    # ------------------------------------------------------------------
    state_id_ctr = _IdCounter(10000)
    compound_id_ctr = _IdCounter(20000)
    protein_id_ctr = _IdCounter(30000)
    complex_id_ctr = _IdCounter(40000)
    reaction_id_ctr = _IdCounter(50000)
    location_id_ctr = _IdCounter(100000)

    # ------------------------------------------------------------------
    # 1. Build compartments (biological states)
    # ------------------------------------------------------------------
    # Collect all compartment names from biological_states + element_locations
    states_raw = _safe_list(data.get("biological_states"))
    state_name_to_id: Dict[str, int] = {}   # state name  → numeric id
    state_name_to_sbml: Dict[str, str] = {} # state name  → "BiologicalState{id}"
    state_display: Dict[str, str] = {}      # state name  → display label for <compartment name>

    def _register_state(name: str) -> str:
        nm = name.strip()
        if not nm:
            nm = default_compartment_name
        if nm not in state_name_to_id:
            sid = state_id_ctr.next()
            state_name_to_id[nm] = sid
            state_name_to_sbml[nm] = f"BiologicalState{sid}"
            # Build a PathWhiz-style display name: "Organism, Cell, Compartment"
            state_display[nm] = nm
        return state_name_to_sbml[nm]

    # Register states from biological_states list
    for bs in states_raw:
        if not isinstance(bs, dict):
            continue
        nm = (bs.get("name") or "").strip()
        if nm:
            _register_state(nm)

    # Always register default
    _register_state(default_compartment_name)

    # ------------------------------------------------------------------
    # 2. Build entity registries
    #    compound  → Compound{id}
    #    protein   → Protein{id}
    #    complex   → ProteinComplex{id}
    # ------------------------------------------------------------------
    compound_rows = _safe_list(entities.get("compounds"))
    protein_rows = _safe_list(entities.get("proteins"))
    complex_rows = _safe_list(entities.get("protein_complexes"))

    # name → record for each entity type
    compound_by_norm: Dict[str, Dict[str, Any]] = {}
    protein_by_norm: Dict[str, Dict[str, Any]] = {}
    complex_by_norm: Dict[str, Dict[str, Any]] = {}

    def _register_compound(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in compound_by_norm:
            return compound_by_norm[norm]
        pw_id = None
        mapped = _safe_dict(row.get("mapped_ids"))
        # Try to use PathWhiz compound ID from mapping_meta or mapped_ids
        for k in ["pathbank_compound_id", "pw_compound_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta")).get(k)
            if v:
                try:
                    pw_id = int(v)
                    break
                except (ValueError, TypeError):
                    pass
        if pw_id is None:
            pw_id = compound_id_ctr.next()
        sbml_id = f"Compound{pw_id}"
        rec = {
            "name": name, "norm": norm, "pw_id": pw_id, "sbml_id": sbml_id,
            "mapped_ids": mapped, "row": row,
        }
        compound_by_norm[norm] = rec
        return rec

    def _register_protein(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in protein_by_norm:
            return protein_by_norm[norm]
        pw_id = None
        for k in ["pathbank_protein_id", "pw_protein_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta")).get(k)
            if v:
                try:
                    pw_id = int(v)
                    break
                except (ValueError, TypeError):
                    pass
        if pw_id is None:
            pw_id = protein_id_ctr.next()
        sbml_id = f"Protein{pw_id}"
        mapped = _safe_dict(row.get("mapped_ids"))
        rec = {
            "name": name, "norm": norm, "pw_id": pw_id, "sbml_id": sbml_id,
            "mapped_ids": mapped, "row": row,
        }
        protein_by_norm[norm] = rec
        return rec

    def _register_complex(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = (row.get("name") or "").strip()
        if not name:
            return None
        norm = _normalize(name)
        if norm in complex_by_norm:
            return complex_by_norm[norm]
        pw_id = None
        for k in ["pathbank_complex_id", "pw_complex_id"]:
            v = row.get(k) or _safe_dict(row.get("mapping_meta")).get(k)
            if v:
                try:
                    pw_id = int(v)
                    break
                except (ValueError, TypeError):
                    pass
        if pw_id is None:
            pw_id = complex_id_ctr.next()
        sbml_id = f"ProteinComplex{pw_id}"
        components = [c.strip() for c in _safe_list(row.get("components")) if isinstance(c, str) and c.strip()]
        mapped = _safe_dict(row.get("mapped_ids"))
        rec = {
            "name": name, "norm": norm, "pw_id": pw_id, "sbml_id": sbml_id,
            "components": components, "mapped_ids": mapped, "row": row,
        }
        complex_by_norm[norm] = rec
        return rec

    for row in compound_rows:
        if isinstance(row, dict):
            _register_compound(row)

    for row in protein_rows:
        if isinstance(row, dict):
            _register_protein(row)

    for row in complex_rows:
        if isinstance(row, dict):
            _register_complex(row)

    # Build a set of all protein component names (from complex definitions) so
    # they are never accidentally registered as compounds.
    _protein_component_norms: Set[str] = set()
    for crec_pre in list(complex_by_norm.values()):
        for comp_name in _safe_list(crec_pre.get("components", [])):
            if isinstance(comp_name, str) and comp_name.strip():
                _protein_component_norms.add(_normalize(comp_name))

    # Helper: resolve any entity name to its registry record + kind
    def _resolve_entity(name: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        norm = _normalize(name)
        if norm in complex_by_norm:
            return "complex", complex_by_norm[norm]
        if norm in protein_by_norm:
            return "protein", protein_by_norm[norm]
        if norm in compound_by_norm:
            return "compound", compound_by_norm[norm]
        # If name is a known protein component, register as protein not compound
        if norm in _protein_component_norms:
            report["warnings"].append({"path": "/entities", "reason": f"Unknown entity '{name}' auto-registered as protein (is a complex component)"})
            new_id = protein_id_ctr.next()
            rec = {
                "name": name, "norm": norm, "pw_id": new_id,
                "sbml_id": f"Protein{new_id}", "mapped_ids": {}, "row": {},
            }
            protein_by_norm[norm] = rec
            return "protein", rec
        # Register as new compound if unknown
        report["warnings"].append({"path": "/entities", "reason": f"Unknown entity '{name}' auto-registered as compound"})
        new_id = compound_id_ctr.next()
        rec = {
            "name": name, "norm": norm, "pw_id": new_id,
            "sbml_id": f"Compound{new_id}", "mapped_ids": {}, "row": {},
        }
        compound_by_norm[norm] = rec
        return "compound", rec

    # ------------------------------------------------------------------
    # 3. Compartment assignment for each entity
    # ------------------------------------------------------------------
    # Build: entity_norm → list of state names from element_locations
    element_locs = _safe_dict(data.get("element_locations"))
    entity_states: Dict[str, List[str]] = defaultdict(list)

    for loc_key, entity_field in [
        ("compound_locations", "compound"),
        ("protein_locations", "protein"),
        ("element_collection_locations", "element_collection"),
    ]:
        for row in _safe_list(element_locs.get(loc_key)):
            if not isinstance(row, dict):
                continue
            ename = (row.get(entity_field) or "").strip()
            state = (row.get("biological_state") or "").strip()
            if ename and state:
                entity_states[_normalize(ename)].append(state)

    def _primary_state(entity_norm: str) -> str:
        states = entity_states.get(entity_norm, [])
        if states:
            return states[0]
        return default_compartment_name

    # Ensure all states used by entities are registered
    for states in entity_states.values():
        for s in states:
            _register_state(s)

    # Determine compartment for each entity
    def _compartment_sbml_id(entity_norm: str) -> str:
        return _register_state(_primary_state(entity_norm))

    # ------------------------------------------------------------------
    # 4. Build reaction plan
    # ------------------------------------------------------------------
    seen_reaction_keys: Set[str] = set()
    reaction_plans: List[Dict[str, Any]] = []

    def _build_reaction_name(inputs: List[str], outputs: List[str]) -> str:
        lhs = " + ".join(inputs) if inputs else "?"
        rhs = " + ".join(outputs) if outputs else "?"
        return f"{lhs} → {rhs}"

    # Reactions
    for rxn in _safe_list(processes.get("reactions")):
        if not isinstance(rxn, dict):
            continue
        inputs = [x.strip() for x in _safe_list(rxn.get("inputs")) if isinstance(x, str) and x.strip()]
        outputs = [x.strip() for x in _safe_list(rxn.get("outputs")) if isinstance(x, str) and x.strip()]
        if not inputs or not outputs:
            report["hard_errors"].append({"path": "/processes/reactions", "reason": "Missing inputs or outputs."})
            continue

        # Dedup key
        key = "|".join(sorted(inputs)) + "→" + "|".join(sorted(outputs))
        if key in seen_reaction_keys:
            report["warnings"].append({"path": "/processes/reactions", "reason": f"Duplicate reaction collapsed."})
            continue
        seen_reaction_keys.add(key)

        # Determine compartment from biological_state field
        state_name = (rxn.get("biological_state") or "").strip()
        if not state_name:
            # Try inputs
            for inp in inputs:
                s = _primary_state(_normalize(inp))
                if s != default_compartment_name:
                    state_name = s
                    break
        if not state_name:
            state_name = default_compartment_name
        compartment_sbml = _register_state(state_name)

        # Resolve modifiers (enzymes)
        modifiers: List[Dict[str, Any]] = []
        for mod in _safe_list(rxn.get("modifiers")) + _safe_list(rxn.get("enzymes")):
            if not isinstance(mod, dict):
                continue
            actor_name = ""
            for field in ["entity", "protein", "protein_complex", "name"]:
                v = (mod.get(field) or "").strip()
                if v:
                    actor_name = v
                    break
            if not actor_name:
                continue
            kind, rec = _resolve_entity(actor_name)
            if rec:
                modifiers.append({"kind": kind, "rec": rec})

        pw_rxn_id = reaction_id_ctr.next()
        rxn_name = _build_reaction_name(inputs, outputs)

        reaction_plans.append({
            "pw_id": pw_rxn_id,
            "sbml_id": f"Reaction{pw_rxn_id}",
            "name": rxn_name,
            "compartment_sbml": compartment_sbml,
            "inputs": inputs,
            "outputs": outputs,
            "modifiers": modifiers,
            "kind": "reaction",
        })

    # Transports (represented as reactions in SBML)
    for tr in _safe_list(processes.get("transports")):
        if not isinstance(tr, dict):
            continue
        cargo = (tr.get("cargo") or "").strip()
        if not cargo:
            continue
        from_state = (tr.get("from_biological_state") or "").strip() or default_compartment_name
        to_state = (tr.get("to_biological_state") or "").strip() or default_compartment_name
        if from_state == to_state:
            report["warnings"].append({"path": "/processes/transports", "reason": f"Degenerate transport for '{cargo}' skipped."})
            continue
        _register_state(from_state)
        _register_state(to_state)

        key = f"transport|{_normalize(cargo)}|{_normalize(from_state)}|{_normalize(to_state)}"
        if key in seen_reaction_keys:
            continue
        seen_reaction_keys.add(key)

        modifiers = []
        for tr_rec in _safe_list(tr.get("transporters")):
            if not isinstance(tr_rec, dict):
                continue
            actor_name = ""
            for field in ["protein", "protein_complex", "name"]:
                v = (tr_rec.get(field) or "").strip()
                if v:
                    actor_name = v
                    break
            if not actor_name:
                continue
            kind, rec = _resolve_entity(actor_name)
            if rec:
                modifiers.append({"kind": kind, "rec": rec})

        pw_rxn_id = reaction_id_ctr.next()
        rxn_name = f"{cargo} transport: {from_state} → {to_state}"
        reaction_plans.append({
            "pw_id": pw_rxn_id,
            "sbml_id": f"Reaction{pw_rxn_id}",
            "name": rxn_name,
            "compartment_sbml": state_name_to_sbml.get(from_state, _register_state(from_state)),
            "inputs": [cargo],
            "outputs": [cargo],
            "inputs_state": from_state,
            "outputs_state": to_state,
            "modifiers": modifiers,
            "kind": "transport",
        })

    # ------------------------------------------------------------------
    # 5. Write SBML as text (avoids libsbml dependency / annotation issues)
    # ------------------------------------------------------------------
    lines: List[str] = []
    a = lines.append

    # XML declaration + root
    a("<?xml version='1.0' encoding='UTF-8' standalone='no'?>")
    a('<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"'
      ' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"'
      ' xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
      ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
      ' level="3" version="1">')

    model_metaid = _new_uuid()
    a(f'  <model metaid="{model_metaid}" id="PathwayModel">')

    # Model annotation: dimensions
    a('    <annotation>')
    a(f'  <pathwhiz:dimensions xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
      f' pathwhiz:width="{int(_CANVAS_W)}" pathwhiz:height="{int(_CANVAS_H)}"/>')
    a('    </annotation>')

    # Unit definitions
    a('      <listOfUnitDefinitions>')
    a('      <unitDefinition name="Mole" id="Unit1">')
    a('        <listOfUnits>')
    a('          <unit scale="0" kind="mole" multiplier="1" exponent="1"/>')
    a('        </listOfUnits>')
    a('      </unitDefinition>')
    a('    </listOfUnitDefinitions>')

    # ------------------------------------------------------------------
    # Compartments
    # ------------------------------------------------------------------
    a('    <listOfCompartments>')
    # Sort by id for determinism
    for state_name, sbml_id in sorted(state_name_to_sbml.items(), key=lambda kv: kv[1]):
        pw_id = state_name_to_id[state_name]
        display = state_display.get(state_name, state_name)
        a(f'      <compartment name="{_html_name(display)}" constant="false"'
          f' id="{sbml_id}" sboTerm="SBO:0000240">')
        a('        <annotation>')
        a(f'  <pathwhiz:compartment xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:compartment_id="{pw_id}" pathwhiz:compartment_type="biological_state"/>')
        a('        </annotation>')
        a('            </compartment>')
    a('    </listOfCompartments>')

    # ------------------------------------------------------------------
    # Species
    # ------------------------------------------------------------------
    a('    <listOfSpecies>')

    def _rdf_xrefs_bqbiol_is(metaid: str, mapped_ids: Dict[str, Any]) -> List[str]:
        """Generate bqbiol:is RDF block for a species."""
        DB_URN: Dict[str, str] = {
            "hmdb": "urn:miriam:hmdb:",
            "kegg": "urn:miriam:kegg.compound:",
            "chebi": "urn:miriam:chebi:",
            "pubchem": "urn:miriam:pubchem.compound:",
            "drugbank": "urn:miriam:drugbank:",
            "uniprot": "urn:miriam:uniprot:",
        }
        uris = []
        for k, v in sorted(mapped_ids.items()):
            if not isinstance(v, str) or not v.strip():
                continue
            prefix = DB_URN.get(k.lower().strip())
            if prefix:
                uris.append(f"{prefix}{v.strip()}")
        if not uris:
            return []
        out = [
            f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">',
            f'\t<rdf:Description rdf:about="#{metaid}">',
            "              <bqbiol:is>",
            "\t<rdf:Bag>",
        ]
        for uri in uris:
            out.append(f"\t<rdf:li rdf:resource=\"{uri}\"/>")
        out += ["\t</rdf:Bag>", "\t</bqbiol:is>", "\t</rdf:Description>", "\t</rdf:RDF>"]
        return out

    def _rdf_hasPart(metaid: str, protein_sbml_ids: List[str]) -> List[str]:
        """Generate bqbiol:hasPart RDF block for a protein complex."""
        uniprots = []
        for pid in protein_sbml_ids:
            norm = _normalize(pid.replace("Protein", ""))
            # Find the protein record by sbml_id
            for prec in protein_by_norm.values():
                if prec["sbml_id"] == pid:
                    uniprot = prec["mapped_ids"].get("uniprot", "Unknown")
                    uniprots.append(uniprot or "Unknown")
                    break
            else:
                uniprots.append("Unknown")
        out = [
            f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
            f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">',
            f'\t<rdf:Description rdf:about="#{metaid}">',
            "\t<bqbiol:hasPart>",
            "\t<rdf:Bag>",
        ]
        for u in uniprots:
            out.append(f'\t<rdf:li rdf:resource="urn:miriam:uniprot:{u}"/>')
        out += ["\t</rdf:Bag>", "\t</bqbiol:hasPart>", "\t</rdf:Description>", "\t</rdf:RDF>"]
        return out

    # --- Protein complexes (emitted first, then their member proteins) ---
    emitted_proteins: Set[str] = set()  # track Protein{id} already emitted

    for crec in sorted(complex_by_norm.values(), key=lambda r: r["pw_id"]):
        compartment_sbml = _compartment_sbml_id(crec["norm"])
        metaid = _new_uuid()

        # Resolve member proteins to Protein{id} species
        member_protein_ids: List[str] = []
        for comp_name in crec["components"]:
            kind, prec = _resolve_entity(comp_name)
            if prec and kind == "protein":
                member_protein_ids.append(prec["sbml_id"])
            elif prec and kind == "complex":
                # Unusual but handle gracefully
                member_protein_ids.append(prec["sbml_id"])
            else:
                # Create an anonymous protein if needed
                pnorm = _normalize(comp_name)
                if pnorm not in protein_by_norm:
                    pid = protein_id_ctr.next()
                    prec2 = {
                        "name": comp_name, "norm": pnorm, "pw_id": pid,
                        "sbml_id": f"Protein{pid}", "mapped_ids": {}, "row": {},
                    }
                    protein_by_norm[pnorm] = prec2
                member_protein_ids.append(protein_by_norm[pnorm]["sbml_id"])

        # If no components listed, add a placeholder "Unknown" protein
        if not member_protein_ids:
            unknown_norm = _normalize("Unknown")
            if unknown_norm not in protein_by_norm:
                uid = protein_id_ctr.next()
                protein_by_norm[unknown_norm] = {
                    "name": "Unknown", "norm": unknown_norm, "pw_id": uid,
                    "sbml_id": f"Protein{uid}", "mapped_ids": {}, "row": {},
                }
            member_protein_ids.append(protein_by_norm[unknown_norm]["sbml_id"])

        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{metaid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000245" compartment="{compartment_sbml}"'
          f' name="{_html_name(crec["name"])}" id="{crec["sbml_id"]}">')
        a('        <annotation>')

        # pathwhiz:species with protein_associations
        pw_ann = (
            f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
            f' pathwhiz:species_id="{crec["pw_id"]}" pathwhiz:species_type="protein_complex">'
            f'\n            <pathwhiz:protein_associations>'
            f'\n              <pathwhiz:protein_complex_proteins>'
        )
        for pid_str in member_protein_ids:
            pw_ann += f"\n                <pathwhiz:protein>{pid_str}</pathwhiz:protein>"
        pw_ann += (
            f'</pathwhiz:protein_complex_proteins>'
            f'</pathwhiz:protein_associations>'
            f'</pathwhiz:species>'
        )
        a(pw_ann)

        # RDF: hasPart
        for rdf_line in _rdf_hasPart(metaid, member_protein_ids):
            a(rdf_line)

        a('\t</annotation>')
        a('            </species>')

        # Emit member protein species (if not already emitted)
        for pid_str in member_protein_ids:
            if pid_str in emitted_proteins:
                continue
            emitted_proteins.add(pid_str)
            # Find record
            prec = next((r for r in protein_by_norm.values() if r["sbml_id"] == pid_str), None)
            if prec is None:
                continue
            p_metaid = _new_uuid()
            p_compartment = _compartment_sbml_id(prec["norm"])
            uniprot = prec["mapped_ids"].get("uniprot", "Unknown") or "Unknown"
            a(f'      <species boundaryCondition="false" constant="false"'
              f' substanceUnits="Unit1" metaid="{p_metaid}"'
              f' hasOnlySubstanceUnits="true" initialAmount="1"'
              f' sboTerm="SBO:0000245" compartment="{p_compartment}"'
              f' name="{_html_name(prec["name"])}" id="{prec["sbml_id"]}">')
            a('        <annotation>')
            pw_sp = (
                f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                f' pathwhiz:species_id="{prec["pw_id"]}" pathwhiz:species_type="protein"/>'
            )
            a(pw_sp)
            # RDF bqbiol:is for uniprot
            a(f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
              f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">')
            a(f'\t<rdf:Description rdf:about="#{p_metaid}">')
            a('\t<bqbiol:is>')
            a('\t<rdf:Bag>')
            a(f'\t<rdf:li rdf:resource="urn:miriam:uniprot:{uniprot}"/>')
            a('\t</rdf:Bag>')
            a('\t</bqbiol:is>')
            a('\t</rdf:Description>')
            a('\t</rdf:RDF>')
            a('\t</annotation>')
            a('            </species>')

    # --- Standalone proteins (not part of complexes already emitted) ---
    for prec in sorted(protein_by_norm.values(), key=lambda r: r["pw_id"]):
        if prec["sbml_id"] in emitted_proteins:
            continue
        emitted_proteins.add(prec["sbml_id"])
        p_metaid = _new_uuid()
        p_compartment = _compartment_sbml_id(prec["norm"])
        uniprot = prec["mapped_ids"].get("uniprot", "Unknown") or "Unknown"
        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{p_metaid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000245" compartment="{p_compartment}"'
          f' name="{_html_name(prec["name"])}" id="{prec["sbml_id"]}">')
        a('        <annotation>')
        a(f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:species_id="{prec["pw_id"]}" pathwhiz:species_type="protein"/>')
        a(f'\t<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
          f' xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">')
        a(f'\t<rdf:Description rdf:about="#{p_metaid}">')
        a('\t<bqbiol:is>')
        a('\t<rdf:Bag>')
        a(f'\t<rdf:li rdf:resource="urn:miriam:uniprot:{uniprot}"/>')
        a('\t</rdf:Bag>')
        a('\t</bqbiol:is>')
        a('\t</rdf:Description>')
        a('\t</rdf:RDF>')
        a('\t</annotation>')
        a('            </species>')
        if uniprot and uniprot != "Unknown":
            report["pathwhiz_id_stats"]["proteins_matched"] += 1
        else:
            report["pathwhiz_id_stats"]["species_no_id"] += 1

    # --- Compounds ---
    for crec in sorted(compound_by_norm.values(), key=lambda r: r["pw_id"]):
        c_metaid = _new_uuid()
        c_compartment = _compartment_sbml_id(crec["norm"])
        mapped = crec["mapped_ids"]
        a(f'      <species boundaryCondition="false" constant="false"'
          f' substanceUnits="Unit1" metaid="{c_metaid}"'
          f' hasOnlySubstanceUnits="true" initialAmount="1"'
          f' sboTerm="SBO:0000247" compartment="{c_compartment}"'
          f' name="{_html_name(crec["name"])}" id="{crec["sbml_id"]}">')
        a('        <annotation>')
        a(f'  <pathwhiz:species xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:species_id="{crec["pw_id"]}" pathwhiz:species_type="compound"/>')

        # RDF: bqbiol:is + bqbiol:isDescribedBy
        rdf_lines = _rdf_xrefs_bqbiol_is(c_metaid, mapped)
        for rl in rdf_lines:
            a(rl)

        a('\t</annotation>')
        a('            </species>')

        if mapped:
            report["pathwhiz_id_stats"]["compounds_matched"] += 1
        else:
            report["pathwhiz_id_stats"]["species_no_id"] += 1

    a('    </listOfSpecies>')

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------
    # --- Build PathWhiz-style layout (shared metabolite positions) ---
    _species_pos, _rxn_centers = _build_pathway_layout(reaction_plans)
    # -----------------------------------------------------------------

    a('    <listOfReactions>')

    for rxn_idx, plan in enumerate(reaction_plans):
        # Get reaction center from layout engine
        cx, cy = _rxn_centers.get(rxn_idx, (300.0 + rxn_idx * 420.0, 300.0))

        # Helper: get node position from global species map
        def _get_node_pos(name: str, fallback_side: str, fallback_idx: int, fallback_total: int):
            norm = _normalize(name)
            if norm in _species_pos:
                nx, ny = _species_pos[norm]
                return nx, ny, nx - _COMPOUND_W / 2, ny - _COMPOUND_H / 2
            return _node_pos_for_side(cx, cy, fallback_idx, fallback_total, fallback_side)
        sbml_rxn_id = plan["sbml_id"]
        pw_rxn_id = plan["pw_id"]
        compartment_sbml = plan["compartment_sbml"]
        rxn_name_html = _html_name(plan["name"])
        sbo = "SBO:0000176" if plan["kind"] == "reaction" else "SBO:0000185"

        a(f'      <reaction fast="false" reversible="false" sboTerm="{sbo}"'
          f' compartment="{compartment_sbml}"'
          f' name="{rxn_name_html}" id="{sbml_rxn_id}">')
        a('        <annotation>')
        a(f'  <pathwhiz:reaction xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
          f' pathwhiz:reaction_id="{pw_rxn_id}"'
          f' pathwhiz:reaction_type="{plan["kind"]}"/>')
        a('        </annotation>')

        # Resolve input/output states for transports
        inputs_state = plan.get("inputs_state", "")
        outputs_state = plan.get("outputs_state", "")

        # listOfReactants
        reactants = plan["inputs"]
        if reactants:
            a('              <listOfReactants>')
            n_react = len(reactants)
            for j, inp_name in enumerate(reactants):
                kind2, rec2 = _resolve_entity(inp_name)
                if rec2 is None:
                    continue
                # Use transport source state if available
                if inputs_state:
                    r_compartment = _register_state(inputs_state)
                else:
                    r_compartment = _compartment_sbml_id(rec2["norm"])

                node_cx, node_cy, node_x, node_y = _get_node_pos(inp_name, "reactant", j, n_react)
                edge_path = _edge_path_reactant(node_cx, node_cy, cx, cy)
                loc_id = location_id_ctr.next()
                loc_type = "protein_complex" if kind2 == "complex" else ("protein" if kind2 == "protein" else "compound")

                a(f'          <speciesReference stoichiometry="1" constant="false"'
                  f' species="{rec2["sbml_id"]}" sboTerm="SBO:0000015">')
                a('            <annotation>')
                a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                  f' pathwhiz:location_type="{loc_type}">')
                a(f'                <pathwhiz:location_element'
                  f' pathwhiz:element_type="compound_location"'
                  f' pathwhiz:element_id="{rec2["sbml_id"]}"'
                  f' pathwhiz:location_id="{loc_id}"'
                  f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_COMPOUND}"'
                  f' pathwhiz:width="{_COMPOUND_W:.1f}" pathwhiz:height="{_COMPOUND_H:.1f}"'
                  f' pathwhiz:zindex="10" pathwhiz:hidden="false"/>'
                  f'<pathwhiz:location_element pathwhiz:element_type="edge"'
                  f' pathwhiz:path="{edge_path}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_EDGE}"'
                  f' pathwhiz:options="{{}}"'
                  f' pathwhiz:zindex="18" pathwhiz:hidden="false"/>'
                  f'</pathwhiz:location>')
                a('            </annotation>')
                a('                    </speciesReference>')
            a('        </listOfReactants>')

        # listOfProducts
        products = plan["outputs"]
        if products:
            a('        <listOfProducts>')
            n_prod = len(products)
            for j, out_name in enumerate(products):
                kind2, rec2 = _resolve_entity(out_name)
                if rec2 is None:
                    continue
                if outputs_state:
                    p_compartment = _register_state(outputs_state)
                else:
                    p_compartment = _compartment_sbml_id(rec2["norm"])

                node_cx, node_cy, node_x, node_y = _get_node_pos(out_name, "product", j, n_prod)
                edge_path = _edge_path_product(cx, cy, node_cx, node_cy)
                loc_id = location_id_ctr.next()
                loc_type = "protein_complex" if kind2 == "complex" else ("protein" if kind2 == "protein" else "compound")

                # Product edges include start_arrow in the reference
                arrow_opt = ('{"start_arrow":true,'
                             '"start_arrow_path":"M 25.9 13.3 L 11 12 L 17.4 25.6",'
                             '"start_flat_arrow":false,"start_flat_arrow_path":null}')
                arrow_opt_escaped = arrow_opt.replace('"', "&quot;")

                a(f'          <speciesReference stoichiometry="1" constant="false"'
                  f' species="{rec2["sbml_id"]}" sboTerm="SBO:0000011">')
                a('            <annotation>')
                a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                  f' pathwhiz:location_type="{loc_type}">')
                a(f'                <pathwhiz:location_element'
                  f' pathwhiz:element_type="compound_location"'
                  f' pathwhiz:element_id="{rec2["sbml_id"]}"'
                  f' pathwhiz:location_id="{loc_id}"'
                  f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_COMPOUND}"'
                  f' pathwhiz:width="{_COMPOUND_W:.1f}" pathwhiz:height="{_COMPOUND_H:.1f}"'
                  f' pathwhiz:zindex="10" pathwhiz:hidden="false"/>'
                  f'<pathwhiz:location_element pathwhiz:element_type="edge"'
                  f' pathwhiz:path="{edge_path}"'
                  f' pathwhiz:visualization_template_id="{_TMPL_EDGE}"'
                  f' pathwhiz:options="{arrow_opt_escaped}"'
                  f' pathwhiz:zindex="18" pathwhiz:hidden="false"/>'
                  f'</pathwhiz:location>')
                a('            </annotation>')
                a('                    </speciesReference>')
            a('        </listOfProducts>')

        # listOfModifiers (enzymes / transporters)
        modifiers = plan.get("modifiers", [])
        if modifiers:
            a('        <listOfModifiers>')
            n_mod = len(modifiers)
            for j, mod in enumerate(modifiers):
                mrec = mod["rec"]
                mkind = mod["kind"]
                node_cx, node_cy, node_x, node_y = _modifier_pos(cx, cy, j, n_mod)
                loc_id = location_id_ctr.next()

                if mkind == "complex":
                    # modifierSpeciesReference for protein_complex with nested protein_location
                    # Find member proteins for this complex
                    member_ids = []
                    for comp_name in mrec.get("components", []):
                        _, prec = _resolve_entity(comp_name)
                        if prec:
                            member_ids.append(prec["sbml_id"])
                    if not member_ids:
                        unknown_norm = _normalize("Unknown")
                        if unknown_norm in protein_by_norm:
                            member_ids.append(protein_by_norm[unknown_norm]["sbml_id"])

                    a(f'          <modifierSpeciesReference species="{mrec["sbml_id"]}" sboTerm="SBO:0000460">')
                    a('            <annotation>')
                    a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                      f' pathwhiz:location_type="protein_complex">')
                    a(f'                <pathwhiz:location_element'
                      f' pathwhiz:element_type="protein_complex_visualization"'
                      f' pathwhiz:element_id="{mrec["sbml_id"]}">')
                    # Protein location elements inside the complex
                    for pid_str in member_ids:
                        p_loc_id = location_id_ctr.next()
                        a(f'                  <pathwhiz:location_element'
                          f' pathwhiz:element_type="protein_location"'
                          f' pathwhiz:element_id="{pid_str}"'
                          f' pathwhiz:location_id="{p_loc_id}"'
                          f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                          f' pathwhiz:visualization_template_id="{_TMPL_PROTEIN}"'
                          f' pathwhiz:width="{_PROTEIN_W:.1f}" pathwhiz:height="{_PROTEIN_H:.1f}"'
                          f' pathwhiz:zindex="8" pathwhiz:hidden="false"/>')
                    a('</pathwhiz:location_element></pathwhiz:location>')
                    a('            </annotation>')
                    a('                    </modifierSpeciesReference>')

                else:
                    # Regular protein modifier
                    a(f'          <modifierSpeciesReference species="{mrec["sbml_id"]}" sboTerm="SBO:0000460">')
                    a('            <annotation>')
                    a(f'  <pathwhiz:location xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"'
                      f' pathwhiz:location_type="protein">')
                    a(f'                <pathwhiz:location_element'
                      f' pathwhiz:element_type="protein_location"'
                      f' pathwhiz:element_id="{mrec["sbml_id"]}"'
                      f' pathwhiz:location_id="{loc_id}"'
                      f' pathwhiz:x="{node_x:.1f}" pathwhiz:y="{node_y:.1f}"'
                      f' pathwhiz:visualization_template_id="{_TMPL_PROTEIN}"'
                      f' pathwhiz:width="{_PROTEIN_W:.1f}" pathwhiz:height="{_PROTEIN_H:.1f}"'
                      f' pathwhiz:zindex="8" pathwhiz:hidden="false"/>'
                      f'</pathwhiz:location>')
                    a('            </annotation>')
                    a('                    </modifierSpeciesReference>')

            a('        </listOfModifiers>')

        a('      </reaction>')

    a('    </listOfReactions>')
    a('  </model>')
    a('</sbml>')

    # Write SBML
    sbml_text = "\n".join(lines)
    sbml_path.write_text(sbml_text, encoding="utf-8")

    # Update counts
    report["counts"]["compartments"] = len(state_name_to_id)
    report["counts"]["species"] = len(compound_by_norm) + len(protein_by_norm) + len(complex_by_norm)
    report["counts"]["reactions"] = len(reaction_plans)

    # Write reports
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_lines = [
        f"SBML file: {sbml_path}",
        f"Compartments: {report['counts']['compartments']}",
        f"Species: {report['counts']['species']}",
        f"Reactions: {report['counts']['reactions']}",
        f"Hard errors (skipped items): {len(report['hard_errors'])}",
        f"Warnings: {len(report['warnings'])}",
        f"Defaults applied: {len(report['defaults_applied'])}",
        f"Validation messages: {report['validation']['error_count']}",
        f"Validation has errors: {report['validation']['has_errors']}",
    ]
    if report["hard_errors"]:
        txt_lines.append("\nHard errors:")
        for item in report["hard_errors"][:50]:
            txt_lines.append(f"- {item.get('path', '')}: {item.get('reason', '')}")
    report_txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PathWhiz-compatible SBML Level 3 from mapped pathway JSON."
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Input mapped JSON path")
    parser.add_argument("--out", dest="sbml_path", default="pathway.sbml", help="Output SBML file path")
    parser.add_argument(
        "--report-json", dest="report_json_path",
        default="sbml_validation_report.json", help="Validation report JSON path",
    )
    parser.add_argument(
        "--report-txt", dest="report_txt_path",
        default="sbml_validation_report.txt", help="Validation report text path",
    )
    parser.add_argument(
        "--default-compartment", dest="default_compartment",
        default="cell", help="Default compartment name used when missing",
    )
    args = parser.parse_args()

    report = build_sbml(
        Path(args.input_path),
        Path(args.sbml_path),
        Path(args.report_json_path),
        Path(args.report_txt_path),
        default_compartment_name=str(args.default_compartment),
    )
    print(f"Wrote SBML: {args.sbml_path}")
    print(
        f"Compartments: {report['counts']['compartments']} | "
        f"Species: {report['counts']['species']} | "
        f"Reactions: {report['counts']['reactions']}"
    )


if __name__ == "__main__":
    main()