"""
layout_engine.py — PathWhiz-style layout engine for any biological pathway.

Provides build_pathway_layout(reaction_plans) -> (species_pos, rxn_centers)
where:
  species_pos  = {norm_name: (cx, cy)}   centre of each node
  rxn_centers  = {rxn_idx:  (cx, cy)}    centre of each reaction square

Strategy:
  1. Score every reaction by its primary non-cofactor output's position in
     the canonical TCA metabolite order. Reactions with a TCA score < 20
     are placed on a circle (the main metabolic ring). Others go in a strip.
  2. For non-TCA (linear/signal) pathways the code falls back to placing
     reactions in a horizontal chain connected by their shared metabolites.
  3. Cofactors/side-chains radiate inward toward the canvas centre.
  4. Enzyme modifier boxes are placed outward from the reaction square.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

# ── Canvas / ring constants ───────────────────────────────────────────────────
CANVAS_W  = 2400.0
CANVAS_H  = 2400.0
CYCLE_CX  = 1200.0
CYCLE_CY  = 1200.0
RXN_RING  =  640.0   # radius for cycle reaction squares
MET_RING  =  820.0   # radius for main-cycle metabolite nodes

COFAC_INWARD  = 170.0   # cofactors sit this far INWARD from reaction square
SIDE_DIST     = 200.0   # non-cycle side-chain metabolite offset
ENZYME_DIST   = 220.0   # enzyme box outward offset (informational; used in json_to_sbml)

# Linear layout (non-cycle fallback)
LINEAR_MARGIN  = 250.0
LINEAR_STEP_X  = 420.0
LINEAR_STEP_Y  = 0.0

# Node sizes (for reference by callers)
COMPOUND_W = 100.0
COMPOUND_H = 100.0
SMALL_W    =  60.0
SMALL_H    =  40.0
PROTEIN_W  = 160.0
PROTEIN_H  =  70.0

# ── Cofactor vocabulary ───────────────────────────────────────────────────────
_COFAC_TOKENS: frozenset = frozenset([
    "nad","nadh","nadp","nadph","fad","fadh","fadh2",
    "atp","adp","amp","gtp","gdp","gmp","ctp","utp",
    "h2o","water","h","h+","proton","oh","hydroxide",
    "o2","oxygen","co2","carbondioxide","carbonate",
    "pi","ppi","pyrophosphate","phosphate","inorganicphosphate",
    "coa","coash","hscoa","coenzymea","acetylcoa",
    "hco3","bicarbonate",
    "ubiquinone","ubiquinol","quinone","quinol",
    "ubiquinoneq","ubiquinolqh2",
    "h2o2","hydrogenperoxide",
    "e","electron",
    "fe2","fe3","cu","cu2","zn2","mg2","mn2","ca2",
    "nh3","ammonia","no","nitricoxide",
])

def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())

def _norm(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", s)

def _is_cofactor(name: str) -> bool:
    c = _compact(name)
    if not c or len(c) <= 1:
        return True
    if c in _COFAC_TOKENS:
        return True
    for tok in ("nad","atp","adp","gtp","gdp","coash","coa","h2o","ubiquin",
                "proton","phosphat","co2","bicarbonate","fadh","hscoa","quinon"):
        if tok in c:
            return True
    return False

# ── TCA cycle metabolite order ────────────────────────────────────────────────
_TCA_METS: List[str] = [
    "oxaloacetate","citrate","cisaconitate","aconitate","isocitrate",
    "oxalosuccinate","alphaketoglutarate","ketoglutarate","2oxoglutarate",
    "succinylcoa","succinate","fumarate","lmalate","malate",
]

def _tca_rank(name: str) -> float:
    c = _compact(name)
    for i, met in enumerate(_TCA_METS):
        if met == c or (len(c) > 3 and met in c) or (len(met) > 3 and c in met):
            return float(i)
    return 999.0

def _rxn_tca_rank(plan: Dict) -> float:
    best = 999.0
    # Rank by primary non-cofactor outputs
    for o in plan.get("outputs", []):
        if not _is_cofactor(o):
            r = _tca_rank(o)
            if r < best:
                best = r
    # Fallback: check inputs (ranked slightly worse to prefer output-based)
    if best >= 999.0:
        for inp in plan.get("inputs", []):
            if not _is_cofactor(inp):
                r = _tca_rank(inp)
                if r < best:
                    best = r + 0.5
    return best


# ── Layout helpers ────────────────────────────────────────────────────────────

def _angle_between(ridx_a: int, ridx_b: int,
                   rxn_centers: Dict[int, Tuple[float, float]]) -> float:
    """Midpoint angle between two reactions on the ring."""
    ax_, ay_ = rxn_centers[ridx_a]
    bx_, by_ = rxn_centers[ridx_b]
    angle_a = math.atan2(ay_ - CYCLE_CY, ax_ - CYCLE_CX)
    angle_b = math.atan2(by_ - CYCLE_CY, bx_ - CYCLE_CX)
    da = angle_b - angle_a
    if da >  math.pi: da -= 2 * math.pi
    if da < -math.pi: da += 2 * math.pi
    return angle_a + da / 2


# ── Main entry point ──────────────────────────────────────────────────────────

def build_pathway_layout(
    reaction_plans: List[Dict[str, Any]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    """
    Returns
    -------
    species_pos : {norm_name -> (cx, cy)}   metabolite node centre
    rxn_centers : {rxn_idx  -> (cx, cy)}    reaction square centre
    """
    n = len(reaction_plans)
    if n == 0:
        return {}, {}

    species_pos:  Dict[str, Tuple[float, float]] = {}
    rxn_centers:  Dict[int, Tuple[float, float]] = {}

    # ── 1. Sort reactions by TCA rank ────────────────────────────────────────
    TCA_THRESHOLD = 20.0
    ranked = sorted(range(n), key=lambda i: _rxn_tca_rank(reaction_plans[i]))
    cycle_rxns = [i for i in ranked if _rxn_tca_rank(reaction_plans[i]) < TCA_THRESHOLD]
    extra_rxns = [i for i in ranked if _rxn_tca_rank(reaction_plans[i]) >= TCA_THRESHOLD]

    is_linear = len(cycle_rxns) < 3

    if is_linear:
        # ── Linear / signalling pathway layout ───────────────────────────────
        # All reactions in a horizontal chain; metabolites above/below
        all_rxns = extra_rxns if extra_rxns else list(range(n))
        cols = min(n, 6)
        row_h = 450.0
        col_w = LINEAR_STEP_X
        start_x = CANVAS_CX - (cols - 1) * col_w / 2 if n > 1 else CANVAS_CX
        start_y = CANVAS_CY - (math.ceil(n / cols) - 1) * row_h / 2

        for k, ridx in enumerate(all_rxns):
            row, col = divmod(k, cols)
            cx = start_x + col * col_w
            cy = start_y + row * row_h
            rxn_centers[ridx] = (cx, cy)

    else:
        # ── Circular layout for cyclic pathways ───────────────────────────────
        m = len(cycle_rxns)
        for k, ridx in enumerate(cycle_rxns):
            # Clockwise from top-right (-80° so top of circle starts at about 1 o'clock)
            angle = math.radians(-80 + 360 * k / m)
            cx = CYCLE_CX + RXN_RING * math.cos(angle)
            cy = CYCLE_CY + RXN_RING * math.sin(angle)
            rxn_centers[ridx] = (cx, cy)

        # Extra reactions: strip below the circle
        if extra_rxns:
            strip_y    = CYCLE_CY + RXN_RING + 370.0
            strip_step = min(360.0, 2000.0 / max(len(extra_rxns), 1))
            strip_x0   = CYCLE_CX - (len(extra_rxns) - 1) * strip_step / 2
            for j, ridx in enumerate(extra_rxns):
                rxn_centers[ridx] = (strip_x0 + j * strip_step, strip_y)

        # ── 2. Place main-cycle metabolites between adjacent reaction squares ─
        for k in range(m):
            ridx_a = cycle_rxns[k]
            ridx_b = cycle_rxns[(k + 1) % m]
            plan_a = reaction_plans[ridx_a]
            plan_b = reaction_plans[ridx_b]

            outs_a = {_norm(o) for o in plan_a.get("outputs", []) if not _is_cofactor(o)}
            ins_b  = {_norm(i) for i in plan_b.get("inputs",  []) if not _is_cofactor(i)}
            shared = outs_a & ins_b

            mid_angle = _angle_between(ridx_a, ridx_b, rxn_centers)
            for name_norm in shared:
                if name_norm not in species_pos:
                    species_pos[name_norm] = (
                        CYCLE_CX + MET_RING * math.cos(mid_angle),
                        CYCLE_CY + MET_RING * math.sin(mid_angle),
                    )

    # ── 3. Place unplaced non-cofactor metabolites radially ──────────────────
    for ridx, plan in enumerate(reaction_plans):
        rcx, rcy = rxn_centers[ridx]

        # Outward direction from canvas centre
        dx, dy = rcx - CYCLE_CX, rcy - CYCLE_CY
        d = math.hypot(dx, dy)
        if d < 1:
            outx, outy = 1.0, 0.0
        else:
            outx, outy = dx / d, dy / d
        perpx, perpy = -outy, outx   # tangential

        unplaced = [
            (name, "out") for name in plan.get("outputs", []) if not _is_cofactor(name)
            and _norm(name) not in species_pos
        ] + [
            (name, "in") for name in plan.get("inputs", []) if not _is_cofactor(name)
            and _norm(name) not in species_pos
        ]

        for slot, (name, side) in enumerate(unplaced):
            nn = _norm(name)
            if nn in species_pos:
                continue
            sign = 1 if side == "in" else -1
            fan  = (slot - (len(unplaced) - 1) / 2) * 150.0
            dist = SIDE_DIST * (1.3 if not is_linear else 1.0)
            species_pos[nn] = (
                rcx + outx * dist + perpx * fan * sign,
                rcy + outy * dist + perpy * fan * sign,
            )

    # ── 4. Place cofactors inward (toward canvas centre) ─────────────────────
    for ridx, plan in enumerate(reaction_plans):
        rcx, rcy = rxn_centers[ridx]
        dx, dy   = rcx - CYCLE_CX, rcy - CYCLE_CY
        d        = math.hypot(dx, dy)
        if d < 1: inx, iny = -1.0, 0.0
        else:     inx, iny = -dx / d, -dy / d
        perpx, perpy = -iny, inx

        all_cof = [o for o in plan.get("inputs",  []) if _is_cofactor(o)] + \
                  [o for o in plan.get("outputs", []) if _is_cofactor(o)]

        for slot, name in enumerate(all_cof):
            nn = _norm(name)
            if nn in species_pos:
                continue
            fan = (slot - (len(all_cof) - 1) / 2) * 95.0
            species_pos[nn] = (
                rcx + inx * COFAC_INWARD + perpx * fan,
                rcy + iny * COFAC_INWARD + perpy * fan,
            )

    return species_pos, rxn_centers


# Alias so canvas centre is accessible without importing CYCLE_CX/CY
CANVAS_CX = CYCLE_CX
CANVAS_CY = CYCLE_CY
