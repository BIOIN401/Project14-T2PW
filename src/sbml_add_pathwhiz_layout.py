#!/usr/bin/env python3
"""sbml_add_pathwhiz_layout.py  (rewritten – connected pathway layout)

Produces a unified, connected pathway diagram instead of isolated per-reaction grids.

Key improvements over the original:
- Shared species (same name across reactions) are placed ONCE and reused.
- Reactions are arranged in a left-to-right topological flow based on
  metabolite connectivity (product of reaction N → reactant of reaction N+1).
- Proteins/enzymes are rendered above their reaction center.
- Compartment boxes are drawn around their member species.
- All PathWhiz-style annotations (per-speciesReference location_element blocks
  plus the backward-compat model-level blocks) are emitted so that
  sbml_render_pathwhiz_like.py still works without modification.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ── Namespace setup ───────────────────────────────────────────────────────────
SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
PW_NS   = "http://www.spmdb.ca/pathwhiz"

ET.register_namespace("",        SBML_NS)
ET.register_namespace("pathwhiz", PW_NS)
ET.register_namespace("bqbiol",   "http://biomodels.net/biology-qualifiers/")
ET.register_namespace("rdf",      "http://www.w3.org/1999/02/22-rdf-syntax-ns#")

NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}

# ── Layout constants (pixels) ─────────────────────────────────────────────────
MARGIN          = 100.0   # canvas edge padding
RXN_STEP_X      = 380.0   # horizontal distance between consecutive reaction centers
LANE_HEIGHT     = 280.0   # vertical distance between parallel reaction lanes
RXN_CY_BASE     = 260.0   # baseline Y for reaction centers (first lane)

COMPOUND_W      = 78.0
COMPOUND_H      = 78.0
PROTEIN_W       = 150.0
PROTEIN_H       = 60.0
REACTANT_OFFSET = 170.0   # horizontal distance from rxn center to reactant column
PRODUCT_OFFSET  = 170.0   # horizontal distance from rxn center to product column
NODE_SPACING_Y  = 90.0    # vertical spacing between stacked species
MODIFIER_ABOVE  = 110.0   # how far above the rxn center enzymes sit

TMPL_COMPOUND   = "3"
TMPL_PROTEIN    = "6"
TMPL_EDGE       = "5"
Z_NODE          = 10
Z_EDGE          = 18
Z_ENZYME        = 8
COMPARTMENT_PAD = 40.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _q(tag: str, ns: str = SBML_NS) -> str:
    return f"{{{ns}}}{tag}"


def _norm(name: str) -> str:
    """Normalise species name for identity matching."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _is_protein(sid: str, name: str) -> bool:
    sid_l = sid.lower()
    name_l = name.lower()
    if sid_l.startswith("p_"):
        return True
    keywords = ["enzyme","protein","kinase","peroxidase","transporter","atpase",
                "phosphatase","deiodinase","symporter","receptor","ligase","reductase"]
    return any(k in name_l for k in keywords)


def _loc_counter(start: int = 2000) -> "list[int]":
    return [start]


def _next_lid(c: list) -> str:
    c[0] += 1
    return str(c[0])


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class SpecNode:
    sid: str          # original SBML species id
    name: str
    compartment: str
    is_protein: bool
    # layout position (set during placement)
    x: float = 0.0
    y: float = 0.0
    placed: bool = False


@dataclass
class RxnNode:
    rid: str
    name: str
    compartment: str
    reactant_sids: List[str] = field(default_factory=list)
    product_sids:  List[str] = field(default_factory=list)
    modifier_sids: List[str] = field(default_factory=list)
    # layout
    cx: float = 0.0
    cy: float = 0.0
    topo_rank: int = 0   # left-to-right order
    lane: int = 0        # vertical lane


# ── Topology: assign left-to-right ranks via BFS on the reaction graph ────────
def _assign_ranks(rxns: List[RxnNode], species_map: Dict[str, SpecNode]) -> None:
    """Topological BFS: rank reactions by metabolite dependency."""
    # Map: species sid → list of reaction indices that produce it
    produces: Dict[str, List[int]] = defaultdict(list)
    # Map: species sid → list of reaction indices that consume it
    consumes: Dict[str, List[int]] = defaultdict(list)

    for i, r in enumerate(rxns):
        for sid in r.product_sids:
            produces[sid].append(i)
        for sid in r.reactant_sids:
            consumes[sid].append(i)

    # Build reaction dependency graph: r_i → r_j if r_i produces something r_j consumes
    deps: Dict[int, Set[int]] = defaultdict(set)
    for sid in produces:
        for pi in produces[sid]:
            for ci in consumes.get(sid, []):
                if pi != ci:
                    deps[pi].add(ci)

    # BFS from reactions with no predecessors
    in_degree = [0] * len(rxns)
    for pi, successors in deps.items():
        for ci in successors:
            in_degree[ci] += 1

    queue = deque(i for i in range(len(rxns)) if in_degree[i] == 0)
    rank = 0
    while queue:
        next_queue: deque = deque()
        for i in list(queue):
            rxns[i].topo_rank = rank
            for j in deps[i]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    next_queue.append(j)
        queue = next_queue
        rank += 1

    # Any unreached reactions (cycles) get the next rank
    for r in rxns:
        if r.topo_rank == 0 and in_degree[rxns.index(r)] > 0:
            r.topo_rank = rank


def _assign_lanes(rxns: List[RxnNode]) -> None:
    """Stack reactions with the same rank into vertical lanes."""
    rank_counts: Dict[int, int] = defaultdict(int)
    for r in rxns:
        r.lane = rank_counts[r.topo_rank]
        rank_counts[r.topo_rank] += 1


# ── Place reaction centers ────────────────────────────────────────────────────
def _place_reactions(rxns: List[RxnNode]) -> None:
    for r in rxns:
        r.cx = MARGIN + r.topo_rank * RXN_STEP_X + REACTANT_OFFSET
        r.cy = RXN_CY_BASE + r.lane * LANE_HEIGHT


# ── Place species nodes ───────────────────────────────────────────────────────
def _stack_y(center_y: float, index: int, total: int) -> float:
    return center_y + (index - (total - 1) / 2.0) * NODE_SPACING_Y


def _place_species(
    rxns: List[RxnNode],
    species_map: Dict[str, SpecNode],
) -> None:
    """Place each species node.  If a species appears in multiple reactions,
    place it at the FIRST use and reuse that position for subsequent reactions."""

    for r in rxns:
        # Reactants (left of reaction center)
        n_r = len(r.reactant_sids)
        for j, sid in enumerate(r.reactant_sids):
            sp = species_map.get(sid)
            if sp is None or sp.placed:
                continue
            sp.x = r.cx - REACTANT_OFFSET - COMPOUND_W / 2
            sp.y = _stack_y(r.cy, j, n_r) - COMPOUND_H / 2
            sp.placed = True

        # Products (right of reaction center)
        n_p = len(r.product_sids)
        for j, sid in enumerate(r.product_sids):
            sp = species_map.get(sid)
            if sp is None or sp.placed:
                continue
            sp.x = r.cx + PRODUCT_OFFSET - COMPOUND_W / 2
            sp.y = _stack_y(r.cy, j, n_p) - COMPOUND_H / 2
            sp.placed = True

        # Modifiers (above reaction center)
        n_m = len(r.modifier_sids)
        for j, sid in enumerate(r.modifier_sids):
            sp = species_map.get(sid)
            if sp is None or sp.placed:
                continue
            x_off = (j - (n_m - 1) / 2.0) * (PROTEIN_W + 20.0)
            sp.x = r.cx + x_off - PROTEIN_W / 2
            sp.y = r.cy - MODIFIER_ABOVE - PROTEIN_H / 2
            sp.placed = True

    # Any remaining unplaced species (not connected to any reaction)
    fallback_x = MARGIN
    fallback_y = RXN_CY_BASE + len(rxns) * LANE_HEIGHT + 80.0
    for sp in species_map.values():
        if not sp.placed:
            sp.x = fallback_x
            sp.y = fallback_y
            sp.placed = True
            fallback_x += COMPOUND_W + 20.0


# ── Edge path builders ────────────────────────────────────────────────────────
def _edge_reactant(sp: SpecNode, rx: float, ry: float) -> str:
    """Horizontal line from right edge of compound node to reaction center."""
    x1 = sp.x + COMPOUND_W
    y1 = sp.y + COMPOUND_H / 2
    return f"M{x1:.1f} {y1:.1f} L{rx:.1f} {ry:.1f} "


def _edge_product(rx: float, ry: float, sp: SpecNode) -> str:
    x2 = sp.x
    y2 = sp.y + COMPOUND_H / 2
    return f"M{rx:.1f} {ry:.1f} L{x2:.1f} {y2:.1f} "


def _edge_modifier(sp: SpecNode, rx: float, ry: float) -> str:
    """From bottom center of protein box down to reaction center."""
    mx = sp.x + PROTEIN_W / 2
    my = sp.y + PROTEIN_H
    return f"M{mx:.1f} {my:.1f} L{rx:.1f} {ry:.1f} "


# ── Build per-speciesReference annotation (PathWhiz format) ──────────────────
def _make_specref_annotation(
    sid: str,
    sp: SpecNode,
    edge_path: str,
    role: str,   # "reactant" | "product" | "modifier"
    lid_counter: list,
) -> ET.Element:
    w = PROTEIN_W if sp.is_protein else COMPOUND_W
    h = PROTEIN_H if sp.is_protein else COMPOUND_H
    loc_type = "protein" if sp.is_protein else "compound"
    el_type  = "protein_location" if sp.is_protein else "compound_location"

    ann = ET.Element(_q("annotation"))
    wrapper = ET.SubElement(ann, _q("location", PW_NS))
    wrapper.set(f"{{{PW_NS}}}location_type", loc_type)

    node_le = ET.SubElement(wrapper, _q("location_element", PW_NS))
    node_le.set(f"{{{PW_NS}}}element_type", el_type)
    node_le.set(f"{{{PW_NS}}}element_id", sid)
    node_le.set(f"{{{PW_NS}}}location_id", _next_lid(lid_counter))
    node_le.set(f"{{{PW_NS}}}x", f"{sp.x:.1f}")
    node_le.set(f"{{{PW_NS}}}y", f"{sp.y:.1f}")
    node_le.set(f"{{{PW_NS}}}visualization_template_id",
                TMPL_PROTEIN if sp.is_protein else TMPL_COMPOUND)
    node_le.set(f"{{{PW_NS}}}width",  f"{w:.1f}")
    node_le.set(f"{{{PW_NS}}}height", f"{h:.1f}")
    node_le.set(f"{{{PW_NS}}}zindex", str(Z_ENZYME if role == "modifier" else Z_NODE))
    node_le.set(f"{{{PW_NS}}}hidden", "false")

    edge_le = ET.SubElement(wrapper, _q("location_element", PW_NS))
    edge_le.set(f"{{{PW_NS}}}element_type", "edge")
    edge_le.set(f"{{{PW_NS}}}path", edge_path)
    edge_le.set(f"{{{PW_NS}}}visualization_template_id", TMPL_EDGE)
    edge_le.set(f"{{{PW_NS}}}zindex", str(Z_EDGE))
    edge_le.set(f"{{{PW_NS}}}hidden", "false")

    return ann


# ── Compartment bounding boxes ────────────────────────────────────────────────
def _compartment_bounds(
    species_map: Dict[str, SpecNode],
    comp_id: str,
) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for sp in species_map.values():
        if sp.compartment != comp_id:
            continue
        w = PROTEIN_W if sp.is_protein else COMPOUND_W
        h = PROTEIN_H if sp.is_protein else COMPOUND_H
        xs.extend([sp.x, sp.x + w])
        ys.extend([sp.y, sp.y + h])
    if not xs:
        return None
    return (
        min(xs) - COMPARTMENT_PAD,
        min(ys) - COMPARTMENT_PAD,
        max(xs) + COMPARTMENT_PAD,
        max(ys) + COMPARTMENT_PAD,
    )


# ── Main entry ────────────────────────────────────────────────────────────────
def add_pathwhiz_layout(in_path: str, out_path: str) -> None:
    tree = ET.parse(in_path)
    root = tree.getroot()

    model = root.find("sbml:model", NS)
    if model is None:
        raise ValueError("No <model> found in SBML file.")

    # ── 1. Collect species ────────────────────────────────────────────────────
    species_map: Dict[str, SpecNode] = {}
    comp_names: Dict[str, str] = {}

    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        cid = c.get("id") or ""
        if cid:
            comp_names[cid] = c.get("name") or cid

    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid  = sp.get("id") or ""
        name = sp.get("name") or sid
        comp = sp.get("compartment") or ""
        if sid:
            species_map[sid] = SpecNode(
                sid=sid,
                name=name,
                compartment=comp,
                is_protein=_is_protein(sid, name),
            )

    # ── 2. Collect reactions ──────────────────────────────────────────────────
    rxns: List[RxnNode] = []
    for rxn_el in root.findall(".//sbml:listOfReactions/sbml:reaction", NS):
        rid  = rxn_el.get("id") or ""
        rname= rxn_el.get("name") or rid
        rcomp= rxn_el.get("compartment") or ""
        reactants = [
            r.get("species") for r in
            rxn_el.findall("./sbml:listOfReactants/sbml:speciesReference", NS)
            if r.get("species")
        ]
        products = [
            p.get("species") for p in
            rxn_el.findall("./sbml:listOfProducts/sbml:speciesReference", NS)
            if p.get("species")
        ]
        modifiers = [
            m.get("species") for m in
            rxn_el.findall("./sbml:listOfModifiers/sbml:modifierSpeciesReference", NS)
            if m.get("species")
        ]
        rxns.append(RxnNode(
            rid=rid, name=rname, compartment=rcomp,
            reactant_sids=reactants,
            product_sids=products,
            modifier_sids=modifiers,
        ))

    # ── 3. Topology → placement ───────────────────────────────────────────────
    _assign_ranks(rxns, species_map)
    _assign_lanes(rxns)
    _place_reactions(rxns)
    _place_species(rxns, species_map)

    lid_counter = _loc_counter()

    # ── 4. Annotate compartments ──────────────────────────────────────────────
    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        cid = c.get("id") or ""
        ann = c.find("sbml:annotation", NS)
        if ann is None:
            ann = ET.SubElement(c, _q("annotation"))
        for old in list(ann.findall(f"{{{PW_NS}}}compartment")):
            ann.remove(old)
        existing_cid = None
        for old in list(ann.findall(f"{{{PW_NS}}}pathwhiz")):
            ann.remove(old)
        pw_comp = ET.SubElement(ann, _q("compartment", PW_NS))
        pw_comp.set(f"{{{PW_NS}}}compartment_type", "biological_state")

    # ── 5. Annotate species ───────────────────────────────────────────────────
    for sp_el in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp_el.get("id") or ""
        sp = species_map.get(sid)
        if sp is None:
            continue
        ann = sp_el.find("sbml:annotation", NS)
        if ann is None:
            ann = ET.SubElement(sp_el, _q("annotation"))
        for old in list(ann.findall(f"{{{PW_NS}}}species")):
            ann.remove(old)
        pw_sp = ET.SubElement(ann, _q("species", PW_NS))
        pw_sp.set(f"{{{PW_NS}}}species_type", "protein" if sp.is_protein else "compound")

    # ── 6. Annotate reactions + speciesReferences ─────────────────────────────
    rxn_by_id = {r.rid: r for r in rxns}
    for rxn_el in root.findall(".//sbml:listOfReactions/sbml:reaction", NS):
        rid = rxn_el.get("id") or ""
        r = rxn_by_id.get(rid)
        if r is None:
            continue

        # Reaction annotation
        ann = rxn_el.find("sbml:annotation", NS)
        if ann is None:
            ann = ET.SubElement(rxn_el, _q("annotation"))
        for old in list(ann.findall(f"{{{PW_NS}}}reaction")):
            ann.remove(old)
        rtype = "transport" if "transport" in rid.lower() else "reaction"
        pw_rxn = ET.SubElement(ann, _q("reaction", PW_NS))
        pw_rxn.set(f"{{{PW_NS}}}reaction_type", rtype)

        # Reactant speciesReferences
        for ref in rxn_el.findall("./sbml:listOfReactants/sbml:speciesReference", NS):
            sid = ref.get("species") or ""
            sp = species_map.get(sid)
            if sp is None:
                continue
            edge_path = _edge_reactant(sp, r.cx, r.cy)
            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _make_specref_annotation(sid, sp, edge_path, "reactant", lid_counter))

        # Product speciesReferences
        for ref in rxn_el.findall("./sbml:listOfProducts/sbml:speciesReference", NS):
            sid = ref.get("species") or ""
            sp = species_map.get(sid)
            if sp is None:
                continue
            edge_path = _edge_product(r.cx, r.cy, sp)
            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _make_specref_annotation(sid, sp, edge_path, "product", lid_counter))

        # Modifier speciesReferences
        for ref in rxn_el.findall("./sbml:listOfModifiers/sbml:modifierSpeciesReference", NS):
            sid = ref.get("species") or ""
            sp = species_map.get(sid)
            if sp is None:
                continue
            edge_path = _edge_modifier(sp, r.cx, r.cy)
            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _make_specref_annotation(sid, sp, edge_path, "modifier", lid_counter))

    # ── 7. Model annotation: canvas dimensions + backward-compat elements ─────
    # Calculate canvas size from all placed species
    all_x = [sp.x + (PROTEIN_W if sp.is_protein else COMPOUND_W) for sp in species_map.values()]
    all_y = [sp.y + (PROTEIN_H if sp.is_protein else COMPOUND_H) for sp in species_map.values()]
    # Also include reaction centers
    for r in rxns:
        all_x.append(r.cx + PRODUCT_OFFSET + COMPOUND_W)
        all_y.append(r.cy + MODIFIER_ABOVE + PROTEIN_H)

    canvas_w = (max(all_x) + MARGIN) if all_x else 1200.0
    canvas_h = (max(all_y) + MARGIN) if all_y else 800.0

    model_ann = model.find("sbml:annotation", NS)
    if model_ann is None:
        model_ann = ET.SubElement(model, _q("annotation"))

    # Remove old layout elements
    for tag in (f"{{{PW_NS}}}dimensions", f"{{{PW_NS}}}location_element", f"{{{PW_NS}}}pathwhiz"):
        for old in list(model_ann.findall(tag)):
            model_ann.remove(old)

    # Canvas dimensions
    dims = ET.SubElement(model_ann, _q("dimensions", PW_NS))
    dims.set(f"{{{PW_NS}}}width",  f"{canvas_w:.0f}")
    dims.set(f"{{{PW_NS}}}height", f"{canvas_h:.0f}")

    # Backward-compat: species location_elements at model level
    for sid, sp in species_map.items():
        if not sp.placed:
            continue
        w = PROTEIN_W if sp.is_protein else COMPOUND_W
        h = PROTEIN_H if sp.is_protein else COMPOUND_H
        etype = "protein_location" if sp.is_protein else "compound_location"
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type",  etype)
        le.set(f"{{{PW_NS}}}element_id",    sid)
        le.set(f"{{{PW_NS}}}x",     f"{sp.x:.2f}")
        le.set(f"{{{PW_NS}}}y",     f"{sp.y:.2f}")
        le.set(f"{{{PW_NS}}}width", f"{w:.2f}")
        le.set(f"{{{PW_NS}}}height",f"{h:.2f}")
        le.set(f"{{{PW_NS}}}zindex","50")
        le.set(f"{{{PW_NS}}}hidden","false")

    # Backward-compat: compartment bounding boxes
    for cid, cname in comp_names.items():
        bounds = _compartment_bounds(species_map, cid)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type",  "element_collection_location")
        le.set(f"{{{PW_NS}}}element_id",    cid)
        le.set(f"{{{PW_NS}}}x",      f"{x1:.2f}")
        le.set(f"{{{PW_NS}}}y",      f"{y1:.2f}")
        le.set(f"{{{PW_NS}}}width",  f"{x2 - x1:.2f}")
        le.set(f"{{{PW_NS}}}height", f"{y2 - y1:.2f}")
        le.set(f"{{{PW_NS}}}zindex", "5")
        le.set(f"{{{PW_NS}}}hidden", "false")

    # Backward-compat: reaction center squares
    for r in rxns:
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type",  "reaction_center")
        le.set(f"{{{PW_NS}}}element_id",    r.rid)
        le.set(f"{{{PW_NS}}}x",     f"{r.cx - 5:.2f}")
        le.set(f"{{{PW_NS}}}y",     f"{r.cy - 5:.2f}")
        le.set(f"{{{PW_NS}}}width", "10.00")
        le.set(f"{{{PW_NS}}}height","10.00")
        le.set(f"{{{PW_NS}}}zindex","35")
        le.set(f"{{{PW_NS}}}hidden","false")

    # Backward-compat: edges at model level
    for r in rxns:
        for sid in r.reactant_sids:
            sp = species_map.get(sid)
            if sp is None:
                continue
            path = _edge_reactant(sp, r.cx, r.cy)
            le = ET.SubElement(model_ann, _q("location_element", PW_NS))
            le.set(f"{{{PW_NS}}}element_type", "edge")
            le.set(f"{{{PW_NS}}}element_id",   f"{sid}__to__{r.rid}")
            le.set(f"{{{PW_NS}}}x","0"); le.set(f"{{{PW_NS}}}y","0")
            le.set(f"{{{PW_NS}}}width","0"); le.set(f"{{{PW_NS}}}height","0")
            le.set(f"{{{PW_NS}}}zindex","30")
            le.set(f"{{{PW_NS}}}hidden","false")
            le.set(f"{{{PW_NS}}}path", path)

        for sid in r.product_sids:
            sp = species_map.get(sid)
            if sp is None:
                continue
            path = _edge_product(r.cx, r.cy, sp)
            le = ET.SubElement(model_ann, _q("location_element", PW_NS))
            le.set(f"{{{PW_NS}}}element_type", "edge")
            le.set(f"{{{PW_NS}}}element_id",   f"{r.rid}__to__{sid}")
            le.set(f"{{{PW_NS}}}x","0"); le.set(f"{{{PW_NS}}}y","0")
            le.set(f"{{{PW_NS}}}width","0"); le.set(f"{{{PW_NS}}}height","0")
            le.set(f"{{{PW_NS}}}zindex","30")
            le.set(f"{{{PW_NS}}}hidden","false")
            le.set(f"{{{PW_NS}}}path", path)

        for sid in r.modifier_sids:
            sp = species_map.get(sid)
            if sp is None:
                continue
            path = _edge_modifier(sp, r.cx, r.cy)
            le = ET.SubElement(model_ann, _q("location_element", PW_NS))
            le.set(f"{{{PW_NS}}}element_type", "edge")
            le.set(f"{{{PW_NS}}}element_id",   f"{sid}__mod__{r.rid}")
            le.set(f"{{{PW_NS}}}x","0"); le.set(f"{{{PW_NS}}}y","0")
            le.set(f"{{{PW_NS}}}width","0"); le.set(f"{{{PW_NS}}}height","0")
            le.set(f"{{{PW_NS}}}zindex","30")
            le.set(f"{{{PW_NS}}}hidden","false")
            le.set(f"{{{PW_NS}}}path", path)
            le.set(f"{{{PW_NS}}}visualization_template_id","83")  # dashed modifier

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add connected pathway layout annotations to an SBML file."
    )
    ap.add_argument("--in",  dest="inp", required=True, help="Input SBML file")
    ap.add_argument("--out", dest="out", required=True, help="Output annotated SBML file")
    args = ap.parse_args()
    add_pathwhiz_layout(args.inp, args.out)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
