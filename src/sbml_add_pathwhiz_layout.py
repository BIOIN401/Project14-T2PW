#!/usr/bin/env python3
"""sbml_add_pathwhiz_layout.py  (v3 — name-deduped connected pathway layout)

Root cause of the disconnected diagram:
  The same metabolite (e.g. "citrate") appears in the SBML as multiple species
  with different IDs because each compartment instance gets its own ID:
      Compound20005_c_cell
      Compound20005_c_mitochondrial_matrix
  The previous layout placed each sid independently, so "citrate" was drawn
  multiple times with no visual link between the reactions that share it.

Fix:
  Group all species by their *normalised name*.  Each unique name gets exactly
  ONE canvas position.  Every reaction that references any sid whose name maps
  to that canonical name points its edge at that shared position.
  This is how KEGG and PathWhiz diagrams look — shared metabolites visually
  connect consecutive reactions.

Layout algorithm:
  1. Build name -> sids mapping.
  2. Topological BFS on reactions via shared names to assign left-right ranks.
  3. Stack same-rank reactions into vertical lanes.
  4. Place one node per canonical name (first use wins).
  5. Emit PathWhiz-style annotations on every speciesReference.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Namespaces
SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
PW_NS   = "http://www.spmdb.ca/pathwhiz"

ET.register_namespace("",         SBML_NS)
ET.register_namespace("pathwhiz", PW_NS)
ET.register_namespace("bqbiol",   "http://biomodels.net/biology-qualifiers/")
ET.register_namespace("rdf",      "http://www.w3.org/1999/02/22-rdf-syntax-ns#")

NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}

# Layout constants
MARGIN          = 120.0
RXN_STEP_X      = 400.0
LANE_H          = 300.0
BASE_CY         = 280.0

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
TMPL_EDGE_MOD   = "83"
Z_NODE          = 10
Z_ENZYME        = 8
Z_EDGE          = 18
Z_COMP          = 5


def _q(tag: str, ns: str = SBML_NS) -> str:
    return f"{{{ns}}}{tag}"


def _norm(name: str) -> str:
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", s)


def _is_protein(sid: str, name: str) -> bool:
    if sid.lower().startswith("p_"):
        return True
    kw = [
        "enzyme", "protein", "kinase", "peroxidase", "transporter", "atpase",
        "phosphatase", "deiodinase", "symporter", "receptor", "ligase",
        "reductase", "synthase", "dehydrogenase", "complex", "transferase",
        "isomerase", "mutase", "carboxylase", "oxidase", "hydrolase",
    ]
    nl = name.lower()
    return any(k in nl for k in kw)


_lid = [0]


def _next_lid() -> str:
    _lid[0] += 1
    return str(_lid[0])


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
    name: str
    reactant_canons: List[str] = field(default_factory=list)
    product_canons: List[str] = field(default_factory=list)
    modifier_canons: List[str] = field(default_factory=list)
    reactant_sids: List[str] = field(default_factory=list)
    product_sids: List[str] = field(default_factory=list)
    modifier_sids: List[str] = field(default_factory=list)
    cx: float = 0.0
    cy: float = 0.0
    rank: int = 0
    lane: int = 0


def _topological_layout(rxns: List[RxnNode]) -> None:
    produces: Dict[str, List[int]] = defaultdict(list)
    consumes: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rxns):
        for cn in r.product_canons:
            produces[cn].append(i)
        for cn in r.reactant_canons:
            consumes[cn].append(i)

    successors: Dict[int, Set[int]] = defaultdict(set)
    in_degree = [0] * len(rxns)
    for cn in produces:
        for pi in produces[cn]:
            for ci in consumes.get(cn, []):
                if pi != ci and ci not in successors[pi]:
                    successors[pi].add(ci)
                    in_degree[ci] += 1

    queue: deque = deque(i for i in range(len(rxns)) if in_degree[i] == 0)
    rank = 0
    visited: Set[int] = set()
    while queue:
        next_q: deque = deque()
        for i in list(queue):
            if i in visited:
                continue
            visited.add(i)
            rxns[i].rank = rank
            for j in successors[i]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    next_q.append(j)
        queue = next_q
        rank += 1

    for i, r in enumerate(rxns):
        if i not in visited:
            r.rank = rank

    rank_count: Dict[int, int] = defaultdict(int)
    for r in rxns:
        r.lane = rank_count[r.rank]
        rank_count[r.rank] += 1


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

    fx, fy = MARGIN, BASE_CY + len(rxns) * LANE_H + 100.0
    for nd in nodes.values():
        if not nd.placed:
            nd.x, nd.y = fx, fy
            nd.placed = True
            fx += nd.w + 20.0


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


def _make_specref_ann(nd: CanonNode, sid: str, edge_path: str,
                      role: str) -> ET.Element:
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


def _add_edge_le(parent: ET.Element, eid: str, path: str, tmpl: str) -> None:
    le = ET.SubElement(parent, _q("location_element", PW_NS))
    le.set(f"{{{PW_NS}}}element_type", "edge")
    le.set(f"{{{PW_NS}}}element_id",   eid)
    le.set(f"{{{PW_NS}}}x", "0"); le.set(f"{{{PW_NS}}}y", "0")
    le.set(f"{{{PW_NS}}}width", "0"); le.set(f"{{{PW_NS}}}height", "0")
    le.set(f"{{{PW_NS}}}zindex", "30")
    le.set(f"{{{PW_NS}}}hidden", "false")
    le.set(f"{{{PW_NS}}}path", path)
    le.set(f"{{{PW_NS}}}visualization_template_id", tmpl)


def add_pathwhiz_layout(in_path: str, out_path: str) -> None:
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

    # Species info: sid -> (display_name, is_protein, compartment_id)
    sid_info: Dict[str, Tuple[str, bool, str]] = {}
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid  = sp.get("id") or ""
        name = (sp.get("name") or sid).strip()
        comp = sp.get("compartment") or ""
        if sid:
            sid_info[sid] = (name, _is_protein(sid, name), comp)

    # Build canonical nodes keyed by normalised name
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
    for rxn_el in root.findall(".//sbml:listOfReactions/sbml:reaction", NS):
        rid   = rxn_el.get("id") or ""
        rname = rxn_el.get("name") or rid

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
            rid=rid, name=rname,
            reactant_canons=_uniq_canons(rsids),
            product_canons=_uniq_canons(psids),
            modifier_canons=_uniq_canons(msids),
            reactant_sids=rsids,
            product_sids=psids,
            modifier_sids=msids,
        ))

    # Layout
    _topological_layout(rxns)
    _place_reactions(rxns)
    _place_nodes(rxns, canon_nodes)

    # Annotate compartments
    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        ann = c.find("sbml:annotation", NS)
        if ann is None:
            ann = ET.SubElement(c, _q("annotation"))
        for old in [e for e in list(ann) if PW_NS in e.tag]:
            ann.remove(old)
        pw = ET.SubElement(ann, _q("compartment", PW_NS))
        pw.set(f"{{{PW_NS}}}compartment_type", "biological_state")

    # Annotate species
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
        pw.set(f"{{{PW_NS}}}species_type", "protein" if info[1] else "compound")

    # Annotate reactions + speciesReferences
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

        for ref in rxn_el.findall("./sbml:listOfReactants/sbml:speciesReference", NS):
            sid = ref.get("species") or ""
            nd  = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd is None:
                continue
            old = ref.find("sbml:annotation", NS)
            if old is not None:
                ref.remove(old)
            ref.insert(0, _make_specref_ann(nd, sid,
                            _path_reactant(nd, r.cx, r.cy), "reactant"))

        for ref in rxn_el.findall("./sbml:listOfProducts/sbml:speciesReference", NS):
            sid = ref.get("species") or ""
            nd  = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd is None:
                continue
            old = ref.find("sbml:annotation", NS)
            if old is not None:
                ref.remove(old)
            ref.insert(0, _make_specref_ann(nd, sid,
                            _path_product(r.cx, r.cy, nd), "product"))

        for ref in rxn_el.findall("./sbml:listOfModifiers/sbml:modifierSpeciesReference", NS):
            sid = ref.get("species") or ""
            nd  = canon_nodes.get(sid_to_canon.get(sid, ""))
            if nd is None:
                continue
            old = ref.find("sbml:annotation", NS)
            if old is not None:
                ref.remove(old)
            ref.insert(0, _make_specref_ann(nd, sid,
                            _path_modifier(nd, r.cx, r.cy), "modifier"))

    # Model annotation: canvas + backward-compat elements
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

    # Species location_elements (one per sid, all pointing to the same position)
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
    for cid, cname in comp_names.items():
        members = [
            nd for nd in canon_nodes.values()
            if nd.placed and any(
                sid_info.get(s, ("", False, ""))[2] == cid
                for s in nd.sids
            )
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

    # Edges
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
