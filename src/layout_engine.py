"""
layout_engine.py — PathWhiz-style circular layout engine

Drop-in replacement for _build_pathway_layout in json_to_sbml.py.
Returns (species_pos, rxn_centers) with proper circular TCA coordinates.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

# ── Canvas constants ──────────────────────────────────────────────────────────
CANVAS_W  = 2400.0
CANVAS_H  = 2400.0
CYCLE_CX  = 1200.0
CYCLE_CY  = 1200.0
RXN_RING  =  640.0   # reaction squares sit on this radius
MET_RING  =  820.0   # main-cycle metabolites sit on outer ring

COFACTOR_DIST = 160.0  # side-chain distance from reaction square
ENZYME_DIST   = 220.0  # enzyme-box outward distance (not used here; for ref)

COMPOUND_W =  100.0
COMPOUND_H =  100.0
SMALL_W    =   60.0
SMALL_H    =   40.0
PROTEIN_W  =  160.0
PROTEIN_H  =   70.0

# ── Cofactor set ──────────────────────────────────────────────────────────────
_COFACTOR_TOKENS = frozenset([
    "nad","nadh","nadp","nadph","fad","fadh","fadh2",
    "atp","adp","amp","gtp","gdp","gmp","ctp","utp",
    "h2o","water","h","h+","proton","oh",
    "o2","oxygen","co2","carbondioxide",
    "pi","ppi","pyrophosphate","phosphate","inorganicphosphate",
    "coa","coash","hscoa","coenzymea","acetylcoa",
    "hco3","bicarbonate",
    "ubiquinone","ubiquinol","ubiquinoneq","ubiquinolqh2",
    "h2o2","hydrogenperoxide",
    "fe2","fe3","cu","cu2","zn2","mg2","mn2","ca2",
])

def _norm(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", s)

def _compact(name: str) -> str:
    """Remove all whitespace for cofactor matching."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())

def _is_cofactor(name: str) -> bool:
    c = _compact(name)
    if not c or len(c) <= 1:
        return True
    if c in _COFACTOR_TOKENS:
        return True
    # Substring checks for common patterns
    for tok in ("nad","atp","adp","gtp","gdp","coa","h2o","ubiquin",
                "proton","phosphat","co2","bicarbonate","fadh","hscoa"):
        if tok in c:
            return True
    return False

# ── TCA canonical order ───────────────────────────────────────────────────────
# Rank metabolites by their position in the canonical TCA cycle
_TCA_METS = [
    "oxaloacetate","citrate","cisaconitate","aconitate",
    "isocitrate","oxalosuccinate","alphaketoglutarate","ketoglutarate",
    "succinylcoa","succinate","fumarate","lmalate","malate",
]

def _tca_rank_metabolite(name: str) -> float:
    c = _compact(name)
    for i, met in enumerate(_TCA_METS):
        if met == c or met in c or c in met:
            return float(i)
    return 999.0

def _rxn_tca_rank(plan: Dict) -> float:
    """Rank a reaction by its primary non-cofactor outputs."""
    best = 999.0
    for o in plan.get("outputs", []):
        if not _is_cofactor(o):
            r = _tca_rank_metabolite(o)
            if r < best:
                best = r
    # Fallback: check inputs too (rank slightly worse than output-based)
    if best == 999.0:
        for inp in plan.get("inputs", []):
            if not _is_cofactor(inp):
                r = _tca_rank_metabolite(inp)
                if r < best:
                    best = r + 0.5
    return best

def _bezier(x1: float, y1: float, x2: float, y2: float,
            bulge: float = 0.18) -> str:
    """Cubic bezier from (x1,y1) to (x2,y2) with a gentle curve."""
    mx, my = (x1+x2)/2, (y1+y2)/2
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return f"M{x1:.0f} {y1:.0f} L{x2:.0f} {y2:.0f} "
    px = -dy/L * bulge * L
    py =  dx/L * bulge * L
    c1x, c1y = mx+px, my+py
    return (f"M{x1:.0f} {y1:.0f} "
            f"C{c1x:.0f} {c1y:.0f} {c1x:.0f} {c1y:.0f} "
            f"{x2:.0f} {y2:.0f} ")

# ── Main layout function ──────────────────────────────────────────────────────

def build_pathway_layout(
    reaction_plans: List[Dict[str, Any]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    """
    Returns:
        species_pos : {norm_name -> (cx, cy)}   metabolite centre
        rxn_centers : {rxn_idx  -> (cx, cy)}    reaction square centre
    """
    n = len(reaction_plans)
    if n == 0:
        return {}, {}

    # ── Step 1: rank reactions by TCA position ────────────────────────────────
    ranked = sorted(range(n), key=lambda i: _rxn_tca_rank(reaction_plans[i]))

    # Reactions with a valid TCA rank go on the circle; others go externally
    TCA_THRESHOLD = 20.0
    cycle_rxns   = [i for i in ranked if _rxn_tca_rank(reaction_plans[i]) < TCA_THRESHOLD]
    extra_rxns   = [i for i in ranked if _rxn_tca_rank(reaction_plans[i]) >= TCA_THRESHOLD]

    # ── Step 2: place reaction squares on the circle ──────────────────────────
    rxn_centers: Dict[int, Tuple[float, float]] = {}
    m = len(cycle_rxns)

    for k, ridx in enumerate(cycle_rxns):
        # Clockwise from top-right (TCA flows clockwise in PathWhiz style)
        angle = math.radians(-80 + 360 * k / max(m, 1))
        cx = CYCLE_CX + RXN_RING * math.cos(angle)
        cy = CYCLE_CY + RXN_RING * math.sin(angle)
        rxn_centers[ridx] = (cx, cy)

    # Extra reactions in a strip below the circle
    if extra_rxns:
        strip_y   = CYCLE_CY + RXN_RING + 350.0
        strip_step = min(350.0, 2200.0 / max(len(extra_rxns), 1))
        strip_x0  = CYCLE_CX - (len(extra_rxns)-1) * strip_step / 2
        for j, ridx in enumerate(extra_rxns):
            rxn_centers[ridx] = (strip_x0 + j*strip_step, strip_y)

    # ── Step 3: place main-cycle metabolites BETWEEN reactions ────────────────
    species_pos: Dict[str, Tuple[float, float]] = {}

    # Build a connections table: which reaction produces which compound
    # (non-cofactors only)
    prod_to_rxn: Dict[str, int] = {}   # norm_name -> rxn_idx that produces it
    for ridx, plan in enumerate(reaction_plans):
        for o in plan.get("outputs", []):
            if not _is_cofactor(o):
                prod_to_rxn[_norm(o)] = ridx

    # For each consecutive pair of cycle reactions, find the metabolite
    # that links them (output of rxn_k, input of rxn_{k+1})
    for k in range(m):
        ridx_a = cycle_rxns[k]
        ridx_b = cycle_rxns[(k+1) % m]
        plan_a = reaction_plans[ridx_a]
        plan_b = reaction_plans[ridx_b]

        outs_a = {_norm(o) for o in plan_a.get("outputs", []) if not _is_cofactor(o)}
        ins_b  = {_norm(i) for i in plan_b.get("inputs",  []) if not _is_cofactor(i)}
        shared = outs_a & ins_b

        # Place shared metabolites on the OUTER ring between the two squares
        ax_, ay_ = rxn_centers[ridx_a]
        bx_, by_ = rxn_centers[ridx_b]
        angle_a = math.atan2(ay_-CYCLE_CY, ax_-CYCLE_CX)
        angle_b = math.atan2(by_-CYCLE_CY, bx_-CYCLE_CX)
        # Average angle (handle wraparound)
        da = angle_b - angle_a
        if da > math.pi:  da -= 2*math.pi
        if da < -math.pi: da += 2*math.pi
        mid_angle = angle_a + da/2

        for name_norm in shared:
            if name_norm not in species_pos:
                mx = CYCLE_CX + MET_RING * math.cos(mid_angle)
                my = CYCLE_CY + MET_RING * math.sin(mid_angle)
                species_pos[name_norm] = (mx, my)

    # ── Step 4: place unplaced non-cofactor metabolites radially ─────────────
    # These are side-chains: products/reactants unique to one reaction
    for ridx, plan in enumerate(reaction_plans):
        rcx, rcy = rxn_centers[ridx]

        # Direction: outward from canvas centre, perpendicular to the ring
        dx, dy = rcx - CYCLE_CX, rcy - CYCLE_CY
        d = math.hypot(dx, dy)
        if d < 1:
            outx, outy = 1.0, 0.0
        else:
            outx, outy = dx/d, dy/d
        perpx, perpy = -outy, outx  # perpendicular (tangential)

        all_mets = (
            [(o, "out") for o in plan.get("outputs", []) if not _is_cofactor(o)] +
            [(i, "in")  for i in plan.get("inputs",  []) if not _is_cofactor(i)]
        )
        unplaced = [(name, side) for name, side in all_mets
                    if _norm(name) not in species_pos]

        for slot, (name, side) in enumerate(unplaced):
            norm_name = _norm(name)
            if norm_name in species_pos:
                continue
            # Fan around the reaction square, offset outward
            fan = (slot - (len(unplaced)-1)/2) * 150.0
            dist = COFACTOR_DIST * (1.4 if ridx in set(cycle_rxns) else 1.0)
            # For inputs: place slightly clockwise; for outputs: counter-clockwise
            sign = 1 if side == "in" else -1
            nx = rcx + outx*dist + perpx*fan*sign
            ny = rcy + outy*dist + perpy*fan*sign
            species_pos[norm_name] = (nx, ny)

    # ── Step 5: place cofactors inward (toward canvas centre) ────────────────
    COFAC_INWARD = 170.0   # distance inward from reaction square

    for ridx, plan in enumerate(reaction_plans):
        rcx, rcy = rxn_centers[ridx]
        dx, dy = rcx - CYCLE_CX, rcy - CYCLE_CY
        d = math.hypot(dx, dy)
        if d < 1: inx, iny = -1.0, 0.0
        else:     inx, iny = -dx/d, -dy/d   # inward
        perpx, perpy = -iny, inx

        cofacs_in  = [o for o in plan.get("inputs",  []) if _is_cofactor(o)]
        cofacs_out = [o for o in plan.get("outputs", []) if _is_cofactor(o)]
        all_cof = cofacs_in + cofacs_out

        for slot, name in enumerate(all_cof):
            norm_name = _norm(name)
            if norm_name in species_pos:
                continue
            fan = (slot - (len(all_cof)-1)/2) * 90.0
            nx = rcx + inx*COFAC_INWARD + perpx*fan
            ny = rcy + iny*COFAC_INWARD + perpy*fan
            species_pos[norm_name] = (nx, ny)

    return species_pos, rxn_centers
