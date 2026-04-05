#!/usr/bin/env python3
"""sbml_add_pathwhiz_layout.py  (v4 — cofactor-filtered + cycle-aware layout)

Two root causes fixed vs previous versions:

1. NAME DEDUPLICATION (v3):
   Same metabolite in different compartments has different SBML IDs but the
   same name. We group by normalised name so shared metabolites get one node.

2. COFACTOR FILTERING + CYCLE DETECTION (v4 — this version):
   Cofactors like NAD+, H2O, CoA-SH appear in almost every reaction.
   Using them for topology makes every reaction rank-0 (all parallel lanes).
   Fix: exclude cofactors when building the dependency graph.

   Cyclic pathways (TCA cycle, Calvin cycle) have no topological start node.
   Fix: detect cycles and fall back to SBML document order within each cycle.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
PW_NS   = "http://www.spmdb.ca/pathwhiz"

ET.register_namespace("",         SBML_NS)
ET.register_namespace("pathwhiz", PW_NS)
ET.register_namespace("bqbiol",   "http://biomodels.net/biology-qualifiers/")
ET.register_namespace("rdf",      "http://www.w3.org/1999/02/22-rdf-syntax-ns#")

NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}

# ── Layout constants ──────────────────────────────────────────────────────────
MARGIN          = 120.0
RXN_STEP_X      = 400.0   # horizontal distance between consecutive ranks
LANE_H          = 300.0   # vertical distance between parallel lanes
BASE_CY         = 280.0   # Y of first lane reaction center

COMPOUND_W      = 78.0
COMPOUND_H      = 78.0
PROTEIN_W       = 160.0
PROTEIN_H       = 60.0

REACTANT_OFFSET = 180.0
PRODUCT_OFFSET  = 180.0
NODE_SPACING_Y  = 95.0
MODIFIER_ABOVE  = 120.0
COMPARTMENT_PAD = 50.0

TMPL_COMPOUND   = "3"
TMPL_PROTEIN    = "6"
TMPL_EDGE       = "5"
TMPL_EDGE_MOD   = "83"   # dashed for modifiers
Z_NODE = 10; Z_ENZYME = 8; Z_EDGE = 18; Z_COMP = 5

# ── Cofactor list (excluded from topology) ────────────────────────────────────
# These metabolites are so ubiquitous that linking reactions through them
# destroys topological ordering — every reaction ends up at rank 0.
_COFACTOR_NORMS: Set[str] = {
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
    "succinate coa sh", "coash",
}


def _q(tag: str, ns: str = SBML_NS) -> str:
    return f"{{{ns}}}{tag}"


def _norm(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", s)


def _is_cofactor(name: str) -> bool:
    n = _norm(name)
    if not n or len(n) <= 1:
        return True
    # exact match against known list
    if n in _COFACTOR_NORMS:
        return True
    # normalised forms of the list
    if n in {_norm(c) for c in _COFACTOR_NORMS}:
        return True
    return False


def _is_protein(sid: str, name: str) -> bool:
    if sid.lower().startswith("p_"):
        return True
    kw = [
        "enzyme", "protein", "kinase", "peroxidase", "transporter", "atpase",
        "phosphatase", "deiodinase", "symporter", "receptor", "ligase",
        "reductase", "synthase", "dehydrogenase", "complex", "transferase",
        "isomerase", "mutase", "carboxylase", "oxidase", "hydrolase",
        "lyase", "epimerase",
    ]
    return any(k in name.lower() for k in kw)


_lid = [0]


def _next_lid() -> str:
    _lid[0] += 1
    return str(_lid[0])


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class CanonNode:
    canon_name: str
    display_name: str
    is_protein: bool
    sids: List[str] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    placed: bool = False

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def w(self) -> float:
        return PROTEIN_W if self.is_protein else COMPOUND_W

    @property
    def h(self) -> float:
        return PROTEIN_H if self.is_protein else COMPOUND_H


@dataclass
class RxnNode:
    rid: str
    doc_order: int        # position in SBML document (fallback for cycles)
    reactant_canons: List[str] = field(default_factory=list)
    product_canons:  List[str] = field(default_factory=list)
    modifier_canons: List[str] = field(default_factory=list)
    reactant_sids:   List[str] = field(default_factory=list)
    product_sids:    List[str] = field(default_factory=list)
    modifier_sids:   List[str] = field(default_factory=list)
    cx: float = 0.0
    cy: float = 0.0
    rank: int = 0
    lane: int = 0


# ── Topology with cofactor filtering + cycle detection ────────────────────────
def _assign_ranks(rxns: List[RxnNode]) -> None:
    n = len(rxns)
    if n == 0:
        return

    # Build dependency graph using only non-cofactor metabolites
    produces: Dict[str, List[int]] = defaultdict(list)
    consumes: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rxns):
        for cn in r.product_canons:
            if not _is_cofactor(cn):
                produces[cn].append(i)
        for cn in r.reactant_canons:
            if not _is_cofactor(cn):
                consumes[cn].append(i)

    # Directed edges: i -> j  (i produces something j consumes)
    successors: Dict[int, Set[int]] = defaultdict(set)
    in_degree = [0] * n
    for met in produces:
        for pi in produces[met]:
            for ci in consumes.get(met, []):
                if pi != ci and ci not in successors[pi]:
                    successors[pi].add(ci)
                    in_degree[ci] += 1

    # Kahn's algorithm for BFS topological sort
    ranks = [0] * n
    queue: deque = deque(i for i in range(n) if in_degree[i] == 0)
    visited: Set[int] = set()
    rank = 0
    while queue:
        nq: deque = deque()
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

    # Handle unvisited nodes — they're in cycles.
    # Find connected components among unvisited nodes and assign
    # sequential ranks by SBML document order within each component.
    unvisited = set(range(n)) - visited
    if unvisited:
        # Build undirected adjacency for component finding
        undirected: Dict[int, Set[int]] = defaultdict(set)
        for i in range(n):
            for j in successors.get(i, set()):
                undirected[i].add(j)
                undirected[j].add(i)

        # Find connected components within unvisited
        seen: Set[int] = set()
        components: List[List[int]] = []
        for start in unvisited:
            if start in seen:
                continue
            comp: List[int] = []
            stk = [start]
            while stk:
                node = stk.pop()
                if node in seen:
                    continue
                seen.add(node)
                comp.append(node)
                for nb in undirected.get(node, set()):
                    if nb not in seen:
                        stk.append(nb)
            components.append(comp)

        for comp in components:
            # Sort by document order so the pathway reads sensibly
            comp_sorted = sorted(comp, key=lambda i: rxns[i].doc_order)

            # Try a BFS within the component first (partial order may exist)
            local_in = defaultdict(int)
            local_suc: Dict[int, Set[int]] = defaultdict(set)
            comp_set = set(comp)
            for i in comp:
                for j in successors.get(i, set()):
                    if j in comp_set and j not in local_suc[i]:
                        local_suc[i].add(j)
                        local_in[j] += 1

            local_starts = [i for i in comp if local_in[i] == 0]

            if local_starts:
                # Has a local start — BFS within component
                lq: deque = deque(local_starts)
                lvis: Set[int] = set()
                r = rank
                while lq:
                    nlq: deque = deque()
                    for i in list(lq):
                        if i in lvis:
                            continue
                        lvis.add(i)
                        ranks[i] = r
                        visited.add(i)
                        for j in local_suc[i]:
                            local_in[j] -= 1
                            if local_in[j] == 0:
                                nlq.append(j)
                    lq = nlq
                    r += 1
                # Any still unvisited in this component (inner cycles)
                for idx, i in enumerate(comp_sorted):
                    if i not in lvis:
                        ranks[i] = rank + idx
                        visited.add(i)
                rank = r
            else:
                # Pure cycle — use document order
                for idx, i in enumerate(comp_sorted):
                    ranks[i] = rank + idx
                    visited.add(i)
                rank += len(comp)

    # Apply ranks and assign lanes
    for i, r in enumerate(rxns):
        r.rank = ranks[i]

    rank_counts: Dict[int, int] = defaultdict(int)
    for r in rxns:
        r.lane = rank_counts[r.rank]
        rank_counts[r.rank] += 1


def _place_reactions(rxns: List[RxnNode]) -> None:
    for r in rxns:
        r.cx = MARGIN + REACTANT_OFFSET + r.rank * RXN_STEP_X
        r.cy = BASE_CY + r.lane * LANE_H


def _stack_y(center_y: float, idx: int, total: int) -> float:
    return center_y + (idx - (total - 1) / 2.0) * NODE_SPACING_Y


def _place_nodes(rxns: List[RxnNode], nodes: Dict[str, CanonNode]) -> None:
    for r in rxns:
        nr = len(r.reactant_canons)
        for j, cn in enumerate(r.reactant_canons):
            nd = nodes.get(cn)
            if nd is None or nd.placed:
                continue
            nd.x = r.cx - REACTANT_OFFSET - nd.w / 2
            nd.y = _stack_y(r.cy, j, nr) - nd.h / 2
            nd.placed = True

        np_ = len(r.product_canons)
        for j, cn in enumerate(r.product_canons):
            nd = nodes.get(cn)
            if nd is None or nd.placed:
                continue
            nd.x = r.cx + PRODUCT_OFFSET - nd.w / 2
            nd.y = _stack_y(r.cy, j, np_) - nd.h / 2
            nd.placed = True

        nm = len(r.modifier_canons)
        for j, cn in enumerate(r.modifier_canons):
            nd = nodes.get(cn)
            if nd is None or nd.placed:
                continue
            x_off = (j - (nm - 1) / 2.0) * (PROTEIN_W + 20.0)
            nd.x = r.cx + x_off - nd.w / 2
            nd.y = r.cy - MODIFIER_ABOVE - nd.h / 2
            nd.placed = True

    # Fallback for unplaced nodes
    fx, fy = MARGIN, BASE_CY + (max((r.lane for r in rxns), default=0) + 1) * LANE_H + 60.0
    for nd in nodes.values():
        if not nd.placed:
            nd.x, nd.y = fx, fy
            nd.placed = True
            fx += nd.w + 20.0


# ── Edge paths ────────────────────────────────────────────────────────────────
def _path_reactant(nd: CanonNode, rx: float, ry: float) -> str:
    x1 = nd.x + nd.w
    y1 = nd.cy
    mx = (x1 + rx) / 2
    return f"M{x1:.1f} {y1:.1f} C{mx:.1f} {y1:.1f} {mx:.1f} {ry:.1f} {rx:.1f} {ry:.1f} "


def _path_product(rx: float, ry: float, nd: CanonNode) -> str:
    x2 = nd.x
    y2 = nd.cy
    mx = (rx + x2) / 2
    return f"M{rx:.1f} {ry:.1f} C{mx:.1f} {ry:.1f} {mx:.1f} {y2:.1f} {x2:.1f} {y2:.1f} "


def _path_modifier(nd: CanonNode, rx: float, ry: float) -> str:
    mx = nd.cx
    my = nd.y + nd.h
    return f"M{mx:.1f} {my:.1f} C{mx:.1f} {ry:.1f} {rx:.1f} {ry:.1f} {rx:.1f} {ry:.1f} "


# ── PathWhiz annotation builders ──────────────────────────────────────────────
def _make_specref_ann(nd: CanonNode, sid: str,
                      edge_path: str, role: str) -> ET.Element:
    loc_type  = "protein" if nd.is_protein else "compound"
    el_type   = "protein_location" if nd.is_protein else "compound_location"
    tmpl      = TMPL_PROTEIN if nd.is_protein else TMPL_COMPOUND
    z         = Z_ENZYME if role == "modifier" else Z_NODE
    edge_tmpl = TMPL_EDGE_MOD if role == "modifier" else TMPL_EDGE

    ann = ET.Element(_q("annotation"))
    wrapper = ET.SubElement(ann, _q("location", PW_NS))
    wrapper.set(f"{{{PW_NS}}}location_type", loc_type)

    nle = ET.SubElement(wrapper, _q("location_element", PW_NS))
    nle.set(f"{{{PW_NS}}}element_type",  el_type)
    nle.set(f"{{{PW_NS}}}element_id",    sid)
    nle.set(f"{{{PW_NS}}}location_id",   _next_lid())
    nle.set(f"{{{PW_NS}}}x",     f"{nd.x:.1f}")
    nle.set(f"{{{PW_NS}}}y",     f"{nd.y:.1f}")
    nle.set(f"{{{PW_NS}}}visualization_template_id", tmpl)
    nle.set(f"{{{PW_NS}}}width",  f"{nd.w:.1f}")
    nle.set(f"{{{PW_NS}}}height", f"{nd.h:.1f}")
    nle.set(f"{{{PW_NS}}}zindex", str(z))
    nle.set(f"{{{PW_NS}}}hidden", "false")

    ele = ET.SubElement(wrapper, _q("location_element", PW_NS))
    ele.set(f"{{{PW_NS}}}element_type", "edge")
    ele.set(f"{{{PW_NS}}}path", edge_path)
    ele.set(f"{{{PW_NS}}}visualization_template_id", edge_tmpl)
    ele.set(f"{{{PW_NS}}}zindex", str(Z_EDGE))
    ele.set(f"{{{PW_NS}}}hidden", "false")

    return ann


def _add_edge_le(parent: ET.Element, eid: str,
                 path: str, tmpl: str) -> None:
    le = ET.SubElement(parent, _q("location_element", PW_NS))
    le.set(f"{{{PW_NS}}}element_type", "edge")
    le.set(f"{{{PW_NS}}}element_id",   eid)
    le.set(f"{{{PW_NS}}}x", "0"); le.set(f"{{{PW_NS}}}y", "0")
    le.set(f"{{{PW_NS}}}width", "0"); le.set(f"{{{PW_NS}}}height", "0")
    le.set(f"{{{PW_NS}}}zindex", "30")
    le.set(f"{{{PW_NS}}}hidden", "false")
    le.set(f"{{{PW_NS}}}path", path)
    le.set(f"{{{PW_NS}}}visualization_template_id", tmpl)


# ── Main ──────────────────────────────────────────────────────────────────────
def add_pathwhiz_layout(in_path: str, out_path: str) -> None:
    _lid[0] = 0   # reset ID counter for each call

    tree = ET.parse(in_path)
    root = tree.getroot()
    model = root.find("sbml:model", NS)
    if model is None:
        raise ValueError("No <model> element found.")

    # Compartment names
    comp_names: Dict[str, str] = {}
    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        cid = c.get("id") or ""
        if cid:
            comp_names[cid] = c.get("name") or cid

    # Species: sid -> (display_name, is_protein, compartment_id)
    sid_info: Dict[str, Tuple[str, bool, str]] = {}
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid  = sp.get("id") or ""
        name = (sp.get("name") or sid).strip()
        comp = sp.get("compartment") or ""
        if sid:
            sid_info[sid] = (name, _is_protein(sid, name), comp)

    # Canonical nodes: one per normalised name
    canon_nodes: Dict[str, CanonNode] = {}
    sid_to_canon: Dict[str, str] = {}
    for sid, (name, is_prot, _comp) in sid_info.items():
        cn = _norm(name)
        if not cn:
            continue
        if cn not in canon_nodes:
            canon_nodes[cn] = CanonNode(
                canon_name=cn, display_name=name, is_protein=is_prot)
        if len(name) > len(canon_nodes[cn].display_name):
            canon_nodes[cn].display_name = name
        canon_nodes[cn].sids.append(sid)
        sid_to_canon[sid] = cn

    # Collect reactions
    rxns: List[RxnNode] = []
    for doc_idx, rxn_el in enumerate(
            root.findall(".//sbml:listOfReactions/sbml:reaction", NS)):
        rid   = rxn_el.get("id") or ""

        def _sids(xpath: str) -> List[str]:
            return [r.get("species") for r in rxn_el.findall(xpath, NS)
                    if r.get("species")]

        rsids = _sids("./sbml:listOfReactants/sbml:speciesReference")
        psids = _sids("./sbml:listOfProducts/sbml:speciesReference")
        msids = _sids("./sbml:listOfModifiers/sbml:modifierSpeciesReference")

        def _uniq_canons(sids: List[str]) -> List[str]:
            seen: Set[str] = set()
            out: List[str] = []
            for s in sids:
                cn = sid_to_canon.get(s)
                if cn and cn not in seen:
                    seen.add(cn)
                    out.append(cn)
            return out

        rxns.append(RxnNode(
            rid=rid, doc_order=doc_idx,
            reactant_canons=_uniq_canons(rsids),
            product_canons=_uniq_canons(psids),
            modifier_canons=_uniq_canons(msids),
            reactant_sids=rsids,
            product_sids=psids,
            modifier_sids=msids,
        ))

    # Layout
    _assign_ranks(rxns)
    _place_reactions(rxns)
    _place_nodes(rxns, canon_nodes)

    # ── Annotate compartments ─────────────────────────────────────────────────
    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        ann = c.find("sbml:annotation", NS)
        if ann is None:
            ann = ET.SubElement(c, _q("annotation"))
        for old in [e for e in list(ann) if PW_NS in e.tag]:
            ann.remove(old)
        pw = ET.SubElement(ann, _q("compartment", PW_NS))
        pw.set(f"{{{PW_NS}}}compartment_type", "biological_state")

    # ── Annotate species ──────────────────────────────────────────────────────
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp.get("id") or ""
        info = sid_info.get(sid)
        if info is None:
            continue
        ann = sp.find("sbml:annotation", NS)
        if ann is None:
            ann = ET.SubElement(sp, _q("annotation"))
        for old in [e for e in list(ann) if PW_NS in e.tag]:
            ann.remove(old)
        pw = ET.SubElement(ann, _q("species", PW_NS))
        pw.set(f"{{{PW_NS}}}species_type",
               "protein" if info[1] else "compound")

    # ── Annotate reactions + speciesReferences ────────────────────────────────
    rxn_by_id = {r.rid: r for r in rxns}
    for rxn_el in root.findall(".//sbml:listOfReactions/sbml:reaction", NS):
        rid = rxn_el.get("id") or ""
        r = rxn_by_id.get(rid)
        if r is None:
            continue

        ann = rxn_el.find("sbml:annotation", NS)
        if ann is None:
            ann = ET.SubElement(rxn_el, _q("annotation"))
        for old in [e for e in list(ann) if PW_NS in e.tag]:
            ann.remove(old)
        rtype = "transport" if "transport" in rid.lower() else "reaction"
        pw = ET.SubElement(ann, _q("reaction", PW_NS))
        pw.set(f"{{{PW_NS}}}reaction_type", rtype)

        for ref in rxn_el.findall(
                "./sbml:listOfReactants/sbml:speciesReference", NS):
            sid = ref.get("species") or ""
            nd  = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd is None:
                continue
            for old in [ref.find("sbml:annotation", NS)]:
                if old is not None:
                    ref.remove(old)
            ref.insert(0, _make_specref_ann(
                nd, sid, _path_reactant(nd, r.cx, r.cy), "reactant"))

        for ref in rxn_el.findall(
                "./sbml:listOfProducts/sbml:speciesReference", NS):
            sid = ref.get("species") or ""
            nd  = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd is None:
                continue
            for old in [ref.find("sbml:annotation", NS)]:
                if old is not None:
                    ref.remove(old)
            ref.insert(0, _make_specref_ann(
                nd, sid, _path_product(r.cx, r.cy, nd), "product"))

        for ref in rxn_el.findall(
                "./sbml:listOfModifiers/sbml:modifierSpeciesReference", NS):
            sid = ref.get("species") or ""
            nd  = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd is None:
                continue
            for old in [ref.find("sbml:annotation", NS)]:
                if old is not None:
                    ref.remove(old)
            ref.insert(0, _make_specref_ann(
                nd, sid, _path_modifier(nd, r.cx, r.cy), "modifier"))

    # ── Model annotation: canvas + backward-compat elements ───────────────────
    all_x = [nd.x + nd.w for nd in canon_nodes.values() if nd.placed]
    all_y = [nd.y + nd.h for nd in canon_nodes.values() if nd.placed]
    for r in rxns:
        all_x.append(r.cx + PRODUCT_OFFSET + COMPOUND_W)
        all_y.append(r.cy + MODIFIER_ABOVE + PROTEIN_H)

    canvas_w = max(all_x, default=1200.0) + MARGIN
    canvas_h = max(all_y, default=800.0)  + MARGIN

    model_ann = model.find("sbml:annotation", NS)
    if model_ann is None:
        model_ann = ET.SubElement(model, _q("annotation"))
    for old in [e for e in list(model_ann) if PW_NS in e.tag]:
        model_ann.remove(old)

    dims = ET.SubElement(model_ann, _q("dimensions", PW_NS))
    dims.set(f"{{{PW_NS}}}width",  f"{canvas_w:.0f}")
    dims.set(f"{{{PW_NS}}}height", f"{canvas_h:.0f}")

    # One location_element per sid (all sids sharing a name → same position)
    for nd in canon_nodes.values():
        if not nd.placed:
            continue
        etype = "protein_location" if nd.is_protein else "compound_location"
        for sid in nd.sids:
            le = ET.SubElement(model_ann, _q("location_element", PW_NS))
            le.set(f"{{{PW_NS}}}element_type",  etype)
            le.set(f"{{{PW_NS}}}element_id",    sid)
            le.set(f"{{{PW_NS}}}x",     f"{nd.x:.2f}")
            le.set(f"{{{PW_NS}}}y",     f"{nd.y:.2f}")
            le.set(f"{{{PW_NS}}}width", f"{nd.w:.2f}")
            le.set(f"{{{PW_NS}}}height",f"{nd.h:.2f}")
            le.set(f"{{{PW_NS}}}zindex","50")
            le.set(f"{{{PW_NS}}}hidden","false")

    # Compartment bounding boxes
    for cid in comp_names:
        members = [
            nd for nd in canon_nodes.values()
            if nd.placed and any(
                sid_info.get(s, ("", False, ""))[2] == cid
                for s in nd.sids)
        ]
        if not members:
            continue
        x1 = min(nd.x for nd in members) - COMPARTMENT_PAD
        y1 = min(nd.y for nd in members) - COMPARTMENT_PAD
        x2 = max(nd.x + nd.w for nd in members) + COMPARTMENT_PAD
        y2 = max(nd.y + nd.h for nd in members) + COMPARTMENT_PAD
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type",  "element_collection_location")
        le.set(f"{{{PW_NS}}}element_id",    cid)
        le.set(f"{{{PW_NS}}}x",      f"{x1:.2f}")
        le.set(f"{{{PW_NS}}}y",      f"{y1:.2f}")
        le.set(f"{{{PW_NS}}}width",  f"{x2-x1:.2f}")
        le.set(f"{{{PW_NS}}}height", f"{y2-y1:.2f}")
        le.set(f"{{{PW_NS}}}zindex", str(Z_COMP))
        le.set(f"{{{PW_NS}}}hidden", "false")

    # Reaction centers
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

    # Edges (backward-compat for sbml_render_pathwhiz_like.py)
    for r in rxns:
        for sid in r.reactant_sids:
            nd = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd:
                _add_edge_le(model_ann, f"{sid}__to__{r.rid}",
                             _path_reactant(nd, r.cx, r.cy), TMPL_EDGE)
        for sid in r.product_sids:
            nd = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd:
                _add_edge_le(model_ann, f"{r.rid}__to__{sid}",
                             _path_product(r.cx, r.cy, nd), TMPL_EDGE)
        for sid in r.modifier_sids:
            nd = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd:
                _add_edge_le(model_ann, f"{sid}__mod__{r.rid}",
                             _path_modifier(nd, r.cx, r.cy), TMPL_EDGE_MOD)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Add connected pathway layout to an SBML file.")
    ap.add_argument("--in",  dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    add_pathwhiz_layout(args.inp, args.out)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
