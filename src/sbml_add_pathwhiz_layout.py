#!/usr/bin/env python3
"""sbml_add_pathwhiz_layout.py

Take a *core* SBML Level-3 file and add PathWhiz-compatible layout annotations.

Structure matches real PathWhiz SBML exports:

  Model annotation:
    <pathwhiz:dimensions pathwhiz:width="..." pathwhiz:height="..."/>

  Compartment annotation:
    <pathwhiz:compartment pathwhiz:compartment_type="biological_state"/>

  Reaction annotation:
    <pathwhiz:reaction pathwhiz:reaction_id="..." pathwhiz:reaction_type="reaction|transport"/>

  speciesReference annotation (reactants & products):
    <pathwhiz:location pathwhiz:location_type="compound|protein|element_collection">
      <pathwhiz:location_element pathwhiz:element_type="compound_location|protein_location|..."
                                  pathwhiz:element_id="..." pathwhiz:location_id="..."
                                  pathwhiz:x="..." pathwhiz:y="..."
                                  pathwhiz:visualization_template_id="..."
                                  pathwhiz:width="..." pathwhiz:height="..."
                                  pathwhiz:zindex="10" pathwhiz:hidden="false"/>
      <pathwhiz:location_element pathwhiz:element_type="edge"
                                  pathwhiz:path="M x y L x y"
                                  pathwhiz:visualization_template_id="5"
                                  pathwhiz:zindex="18" pathwhiz:hidden="false"/>
    </pathwhiz:location>

  modifierSpeciesReference annotation:
    <pathwhiz:location pathwhiz:location_type="protein">
      <pathwhiz:location_element pathwhiz:element_type="protein_location" .../>
      <pathwhiz:location_element pathwhiz:element_type="edge" .../>
    </pathwhiz:location>

Also keeps backward-compat model-level <pathwhiz:location_element> blocks for the
local renderer (sbml_render_pathwhiz_like.py).

Usage:
  python sbml_add_pathwhiz_layout.py --in core.sbml --out core_with_layout.sbml
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
PW_NS = "http://www.spmdb.ca/pathwhiz"

ET.register_namespace("", SBML_NS)
ET.register_namespace("pathwhiz", PW_NS)
ET.register_namespace("bqbiol", "http://biomodels.net/biology-qualifiers/")
ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")

NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}

# ---------------------------------------------------------------------------
# Layout constants (pixels)
# ---------------------------------------------------------------------------
CANVAS_MARGIN = 80.0
NODE_OFFSET_X = 160.0      # horizontal distance from reaction center to species node center
NODE_SPACING_Y = 85.0      # vertical spacing between stacked reactants/products
MODIFIER_OFFSET_Y = 110.0  # distance above reaction center for enzyme nodes
REACTION_GAP_X = 420.0     # horizontal gap between consecutive reaction centers
CENTER_Y_BASE = 300.0      # baseline vertical center for all reactions

# Node sizes (pixels) — match PathWhiz defaults
COMPOUND_W, COMPOUND_H = 78.0, 78.0
PROTEIN_W, PROTEIN_H = 150.0, 70.0
EC_W, EC_H = 150.0, 70.0  # element_collection
REACTION_CENTER_SIZE = 8.0  # small filled square at each reaction center

# PathWhiz visualization_template_id values
TMPL_COMPOUND_REACTANT = "49"
TMPL_COMPOUND_PRODUCT = "55"
TMPL_ELEMENT_COLLECTION = "37"
TMPL_PROTEIN = "99"
TMPL_EDGE = "5"

# sboTerm Z-index
Z_NODE = 10
Z_EDGE = 18
Z_BOX = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(tag: str, ns: str = SBML_NS) -> str:
    return f"{{{ns}}}{tag}"


def _guess_species_type(sid: str) -> str:
    if sid.startswith("p_"):
        return "protein"
    if sid.startswith("m_"):
        return "compound"
    return "compound"


def _node_size(stype: str) -> Tuple[float, float]:
    if stype == "protein":
        return PROTEIN_W, PROTEIN_H
    if stype == "element_collection":
        return EC_W, EC_H
    return COMPOUND_W, COMPOUND_H


def _location_type(stype: str) -> str:
    """PathWhiz pathwhiz:location_type value for a given species type."""
    if stype == "protein":
        return "protein"
    if stype == "element_collection":
        return "element_collection"
    return "compound"


def _element_type(stype: str) -> str:
    """PathWhiz pathwhiz:element_type value for the node location_element."""
    if stype == "protein":
        return "protein_location"
    if stype == "element_collection":
        return "element_collection_location"
    return "compound_location"


def _template_id(stype: str, role: str) -> str:
    """PathWhiz visualization_template_id for a species in a given role."""
    if stype == "protein":
        return TMPL_PROTEIN
    if stype == "element_collection":
        return TMPL_ELEMENT_COLLECTION
    if role == "reactant":
        return TMPL_COMPOUND_REACTANT
    return TMPL_COMPOUND_PRODUCT


def _next_loc_id(counter: List[int]) -> str:
    counter[0] += 1
    return str(counter[0])


def _build_speciesref_annotation(
    sid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    edge_path: str,
    stype: str,
    role: str,
    loc_id: str,
    edge_loc_id: str,
) -> ET.Element:
    """Build annotation exactly matching PathWhiz's speciesReference format."""
    ann = ET.Element(_q("annotation"))
    loc_wrapper = ET.SubElement(ann, _q("location", PW_NS))
    loc_wrapper.set(f"{{{PW_NS}}}location_type", _location_type(stype))

    # Node location_element
    node_le = ET.SubElement(loc_wrapper, _q("location_element", PW_NS))
    node_le.set(f"{{{PW_NS}}}element_type", _element_type(stype))
    node_le.set(f"{{{PW_NS}}}element_id", sid)
    node_le.set(f"{{{PW_NS}}}location_id", loc_id)
    node_le.set(f"{{{PW_NS}}}x", f"{x:.1f}")
    node_le.set(f"{{{PW_NS}}}y", f"{y:.1f}")
    node_le.set(f"{{{PW_NS}}}visualization_template_id", _template_id(stype, role))
    node_le.set(f"{{{PW_NS}}}width", f"{w:.1f}")
    node_le.set(f"{{{PW_NS}}}height", f"{h:.1f}")
    node_le.set(f"{{{PW_NS}}}zindex", str(Z_NODE))
    node_le.set(f"{{{PW_NS}}}hidden", "false")

    # Edge location_element
    edge_le = ET.SubElement(loc_wrapper, _q("location_element", PW_NS))
    edge_le.set(f"{{{PW_NS}}}element_type", "edge")
    edge_le.set(f"{{{PW_NS}}}path", edge_path)
    edge_le.set(f"{{{PW_NS}}}visualization_template_id", TMPL_EDGE)
    edge_le.set(f"{{{PW_NS}}}zindex", str(Z_EDGE))
    edge_le.set(f"{{{PW_NS}}}hidden", "false")

    return ann


# ---------------------------------------------------------------------------
# Graphviz / fallback layout (for model-level renderer compat)
# ---------------------------------------------------------------------------

@dataclass
class _Species:
    sid: str
    name: str
    compartment: str
    stype: str


def _run_dot(dot_src: str) -> str:
    proc = subprocess.run(
        ["dot", "-Tplain"],
        input=dot_src.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.decode("utf-8", errors="replace")


def _parse_dot_plain(plain: str) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str, List[Tuple[float, float]]]]]:
    pos: Dict[str, Tuple[float, float]] = {}
    edges: List[Tuple[str, str, List[Tuple[float, float]]]] = []
    for line in plain.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "node" and len(parts) >= 4:
            pos[parts[1]] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "edge" and len(parts) >= 4:
            tail, head, n = parts[1], parts[2], int(parts[3])
            coords = [(float(parts[4 + 2 * i]), float(parts[5 + 2 * i])) for i in range(n)]
            edges.append((tail, head, coords))
    return pos, edges


def _build_fallback_layout(
    species_nodes: Dict[str, _Species],
    comp_names: Dict[str, str],
    edge_pairs: List[Tuple[str, str]],
) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str, List[Tuple[float, float]]]]]:
    pos: Dict[str, Tuple[float, float]] = {}
    edges: List[Tuple[str, str, List[Tuple[float, float]]]] = []
    box_pad_x, box_pad_y = 0.9, 0.9
    col_gap, row_gap, comp_gap = 1.8, 1.3, 1.8

    species_by_comp: Dict[str, List[_Species]] = {}
    for sid in sorted(species_nodes):
        s = species_nodes[sid]
        species_by_comp.setdefault(s.compartment, []).append(s)
        comp_names.setdefault(s.compartment, s.compartment)

    current_y = 0.0
    for cid in sorted(species_by_comp):
        members = sorted(species_by_comp[cid], key=lambda s: (s.stype != "protein", s.name.lower(), s.sid))
        if not members:
            continue
        cols = max(1, min(4, math.ceil(math.sqrt(len(members)))))
        rows = max(1, math.ceil(len(members) / cols))
        for idx, s in enumerate(members):
            pos[s.sid] = (box_pad_x + (idx % cols) * col_gap, current_y + box_pad_y + (idx // cols) * row_gap)
        current_y += ((rows - 1) * row_gap) + (2 * box_pad_y) + comp_gap

    for tail, head in edge_pairs:
        tp, hp = pos.get(tail), pos.get(head)
        if tp is None or hp is None:
            continue
        x1, y1 = tp; x2, y2 = hp
        if abs(y1 - y2) < 0.01:
            coords = [(x1, y1), (x2, y2)]
        else:
            mx = round((x1 + x2) / 2.0, 4)
            coords = [(x1, y1), (mx, y1), (mx, y2), (x2, y2)]
        edges.append((tail, head, coords))
    return pos, edges


def _scale_and_flip(x: float, y: float, *, scale: float, yflip_max: float, margin: float) -> Tuple[float, float]:
    return (x * scale + margin, (yflip_max - y) * scale + margin)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_pathwhiz_layout(in_path: str, out_path: str) -> None:
    tree = ET.parse(in_path)
    root = tree.getroot()

    model = root.find("sbml:model", NS)
    if model is None:
        raise ValueError("No <model> found")

    # ------------------------------------------------------------------
    # Collect species and compartment info
    # ------------------------------------------------------------------
    species_type: Dict[str, str] = {}
    species_nodes: Dict[str, _Species] = {}
    comp_names: Dict[str, str] = {}

    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp.get("id") or ""
        if not sid:
            continue
        stype = _guess_species_type(sid)
        species_type[sid] = stype
        species_nodes[sid] = _Species(
            sid=sid,
            name=sp.get("name") or sid,
            compartment=sp.get("compartment") or "c_cell",
            stype=stype,
        )

    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        cid = c.get("id") or ""
        if cid:
            comp_names[cid] = c.get("name") or cid

    # ------------------------------------------------------------------
    # Annotate compartments: <pathwhiz:compartment compartment_type="biological_state"/>
    # ------------------------------------------------------------------
    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        cid = c.get("id") or ""
        if not cid:
            continue
        c_ann = c.find("sbml:annotation", NS)
        if c_ann is None:
            c_ann = ET.SubElement(c, _q("annotation"))
        existing_comp_id = None
        for old in list(c_ann.findall(f"{{{PW_NS}}}compartment")):
            existing_comp_id = old.get(f"{{{PW_NS}}}compartment_id")
            c_ann.remove(old)
        for old in list(c_ann.findall(f"{{{PW_NS}}}pathwhiz")):
            c_ann.remove(old)
        pw_comp = ET.SubElement(c_ann, _q("compartment", PW_NS))
        pw_comp.set(f"{{{PW_NS}}}compartment_type", "biological_state")
        if existing_comp_id is not None:
            pw_comp.set(f"{{{PW_NS}}}compartment_id", existing_comp_id)

    # ------------------------------------------------------------------
    # Collect reactions
    # ------------------------------------------------------------------
    reactions = root.findall(".//sbml:listOfReactions/sbml:reaction", NS)

    # Edge pairs routed through reaction center nodes (avoids reactant→product X-crossings)
    rxn_center_ids: List[str] = []
    edge_pairs: List[Tuple[str, str]] = []
    for rxn in reactions:
        rxn_id_str = rxn.get("id") or ""
        if rxn_id_str:
            rxn_center_ids.append(rxn_id_str)
        reactants = [r.get("species") for r in rxn.findall("./sbml:listOfReactants/sbml:speciesReference", NS) if r.get("species")]
        products = [p.get("species") for p in rxn.findall("./sbml:listOfProducts/sbml:speciesReference", NS) if p.get("species")]
        for r in reactants:
            if r and rxn_id_str:
                edge_pairs.append((r, rxn_id_str))  # type: ignore[arg-type]
        for p in products:
            if p and rxn_id_str:
                edge_pairs.append((rxn_id_str, p))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Compute reaction-centered positions
    # ------------------------------------------------------------------
    first_cx = CANVAS_MARGIN + NODE_OFFSET_X + 20.0
    rxn_layouts: List[Dict] = []
    for ridx, rxn in enumerate(reactions):
        reactant_sids = [r.get("species") or "" for r in rxn.findall("./sbml:listOfReactants/sbml:speciesReference", NS)]
        product_sids = [p.get("species") or "" for p in rxn.findall("./sbml:listOfProducts/sbml:speciesReference", NS)]
        modifier_sids = [m.get("species") or "" for m in rxn.findall("./sbml:listOfModifiers/sbml:modifierSpeciesReference", NS)]
        n_mod = len(modifier_sids)
        cx = first_cx + ridx * REACTION_GAP_X
        cy = CENTER_Y_BASE + (MODIFIER_OFFSET_Y / 2 if n_mod > 0 else 0.0)
        rxn_layouts.append({
            "cx": cx, "cy": cy,
            "reactant_sids": reactant_sids,
            "product_sids": product_sids,
            "modifier_sids": modifier_sids,
        })

    # Canvas dimensions
    if rxn_layouts:
        last_cx = rxn_layouts[-1]["cx"]
        canvas_w = last_cx + NODE_OFFSET_X + CANVAS_MARGIN + 20.0
    else:
        canvas_w = 800.0
    max_stack = max(
        (max(len(l["reactant_sids"]), len(l["product_sids"]), 1) for l in rxn_layouts),
        default=1,
    )
    canvas_h = CENTER_Y_BASE + (max_stack / 2) * NODE_SPACING_Y + CANVAS_MARGIN + MODIFIER_OFFSET_Y

    # Location ID counter (auto-increment to give each location_element a unique id)
    loc_id_counter = [1000]

    # ------------------------------------------------------------------
    # Write per-reaction and per-speciesReference annotations (PathWhiz format)
    # ------------------------------------------------------------------
    for rxn, lay in zip(reactions, rxn_layouts):
        cx, cy = lay["cx"], lay["cy"]
        rxn_id_str = rxn.get("id") or ""
        rtype = "transport" if "transport" in rxn_id_str.lower() else "reaction"

        # Reaction annotation: <pathwhiz:reaction reaction_id="..." reaction_type="..."/>
        rxn_ann = rxn.find("sbml:annotation", NS)
        if rxn_ann is None:
            rxn_ann = ET.SubElement(rxn, _q("annotation"))
        existing_rxn_id = None
        for old in list(rxn_ann.findall(f"{{{PW_NS}}}reaction")):
            existing_rxn_id = old.get(f"{{{PW_NS}}}reaction_id")
            rxn_ann.remove(old)
        for old in list(rxn_ann.findall(f"{{{PW_NS}}}pathwhiz")):
            rxn_ann.remove(old)
        rxn_pw = ET.SubElement(rxn_ann, _q("reaction", PW_NS))
        rxn_pw.set(f"{{{PW_NS}}}reaction_type", rtype)
        if existing_rxn_id is not None:
            rxn_pw.set(f"{{{PW_NS}}}reaction_id", existing_rxn_id)

        # Reactants
        reactant_refs = rxn.findall("./sbml:listOfReactants/sbml:speciesReference", NS)
        n_react = len(reactant_refs)
        for j, ref in enumerate(reactant_refs):
            sid = ref.get("species") or ""
            stype = species_type.get(sid, "compound")
            w, h = _node_size(stype)
            y_off = (j - (n_react - 1) / 2.0) * NODE_SPACING_Y
            node_cx = cx - NODE_OFFSET_X
            node_cy = cy + y_off
            node_x = node_cx - w / 2
            node_y = node_cy - h / 2
            # Edge: right edge of node → reaction center
            edge_path = f"M{node_cx + w/2:.0f} {node_cy:.0f} L{cx:.0f} {cy:.0f}"
            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _build_speciesref_annotation(
                sid, node_x, node_y, w, h, edge_path, stype, "reactant",
                _next_loc_id(loc_id_counter), _next_loc_id(loc_id_counter),
            ))

        # Products
        product_refs = rxn.findall("./sbml:listOfProducts/sbml:speciesReference", NS)
        n_prod = len(product_refs)
        for j, ref in enumerate(product_refs):
            sid = ref.get("species") or ""
            stype = species_type.get(sid, "compound")
            w, h = _node_size(stype)
            y_off = (j - (n_prod - 1) / 2.0) * NODE_SPACING_Y
            node_cx = cx + NODE_OFFSET_X
            node_cy = cy + y_off
            node_x = node_cx - w / 2
            node_y = node_cy - h / 2
            # Edge: reaction center → left edge of node
            edge_path = f"M{cx:.0f} {cy:.0f} L{node_cx - w/2:.0f} {node_cy:.0f}"
            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _build_speciesref_annotation(
                sid, node_x, node_y, w, h, edge_path, stype, "product",
                _next_loc_id(loc_id_counter), _next_loc_id(loc_id_counter),
            ))

        # Modifiers (enzymes)
        modifier_refs = rxn.findall("./sbml:listOfModifiers/sbml:modifierSpeciesReference", NS)
        n_mod = len(modifier_refs)
        for j, ref in enumerate(modifier_refs):
            sid = ref.get("species") or ""
            stype = species_type.get(sid, "protein")
            w, h = _node_size(stype)
            x_off = (j - (n_mod - 1) / 2.0) * (w + 20.0)
            node_cx = cx + x_off
            node_cy = cy - MODIFIER_OFFSET_Y
            node_x = node_cx - w / 2
            node_y = node_cy - h / 2
            # Edge: bottom of modifier → reaction center
            edge_path = f"M{node_cx:.0f} {node_cy + h/2:.0f} L{cx:.0f} {cy:.0f}"
            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _build_speciesref_annotation(
                sid, node_x, node_y, w, h, edge_path, stype, "modifier",
                _next_loc_id(loc_id_counter), _next_loc_id(loc_id_counter),
            ))

    # ------------------------------------------------------------------
    # Model annotation: canvas dimensions + backward-compat location_elements
    # ------------------------------------------------------------------
    model_ann = model.find("sbml:annotation", NS)
    if model_ann is None:
        model_ann = ET.SubElement(model, _q("annotation"))
    for old in list(model_ann.findall(f"{{{PW_NS}}}dimensions")):
        model_ann.remove(old)
    for old in list(model_ann.findall(f"{{{PW_NS}}}pathwhiz")):
        model_ann.remove(old)
    for old in list(model_ann.findall(f"{{{PW_NS}}}location_element")):
        model_ann.remove(old)

    # <pathwhiz:dimensions .../>  (directly in model annotation, no wrapper)
    dims = ET.SubElement(model_ann, _q("dimensions", PW_NS))
    dims.set(f"{{{PW_NS}}}width", f"{canvas_w:.0f}")
    dims.set(f"{{{PW_NS}}}height", f"{canvas_h:.0f}")

    # ------------------------------------------------------------------
    # Graphviz / fallback layout for model-level location_elements
    # (keeps sbml_render_pathwhiz_like.py working)
    # ------------------------------------------------------------------
    dot_lines = ["digraph G {", "rankdir=LR;", "splines=true;", "overlap=false;"]
    for cid, cname in comp_names.items():
        members = [s.sid for s in species_nodes.values() if s.compartment == cid]
        if not members:
            continue
        dot_lines += [f"subgraph cluster_{cid} {{", "style=rounded;", f'label="{cname}";']
        for sid in members:
            dot_lines.append(f'"{sid}";')
        dot_lines.append("}")
    for sid in species_nodes:
        dot_lines.append(f'"{sid}";')
    # Reaction center nodes (point-shaped so graphviz places them between reactants/products)
    for rc_id in rxn_center_ids:
        dot_lines.append(f'"{rc_id}" [shape=point,width=0.1];')
    for a, b in edge_pairs:
        dot_lines.append(f'"{a}" -> "{b}";')
    dot_lines.append("}")
    dot_src = "\n".join(dot_lines)

    if shutil.which("dot"):
        try:
            plain = _run_dot(dot_src)
            pos, dot_edges = _parse_dot_plain(plain)
        except subprocess.CalledProcessError:
            pos, dot_edges = _build_fallback_layout(species_nodes, comp_names, edge_pairs)
    else:
        pos, dot_edges = _build_fallback_layout(species_nodes, comp_names, edge_pairs)

    if not pos:
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return

    # Ensure reaction center positions exist in pos.
    # Graphviz places them natively; fallback only positions species nodes, so compute
    # missing ones as the centroid of their reactant + product species positions.
    for rxn, lay in zip(reactions, rxn_layouts):
        rc_id = rxn.get("id") or ""
        if not rc_id or rc_id in pos:
            continue
        r_pts = [pos[s] for s in lay["reactant_sids"] if s in pos]
        p_pts = [pos[s] for s in lay["product_sids"] if s in pos]
        all_pts = r_pts + p_pts
        if not all_pts:
            continue
        pos[rc_id] = (
            sum(p[0] for p in all_pts) / len(all_pts),
            sum(p[1] for p in all_pts) / len(all_pts),
        )

    # Synthesize any edges that the fallback layout skipped (it ignores unknown nodes)
    existing_edge_set = {(t, h) for t, h, _ in dot_edges}
    for a, b in edge_pairs:
        if (a, b) not in existing_edge_set and a in pos and b in pos:
            dot_edges.append((a, b, [pos[a], pos[b]]))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    y_max = max(ys)
    scale, margin = 140.0, 80.0
    compound_r = 13.0
    protein_w_px, protein_h_px = 92.0, 30.0

    # <pathwhiz:species species_type="..."/> on each species annotation (renderer uses this)
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp.get("id") or ""
        if sid not in species_nodes:
            continue
        sp_ann = sp.find("sbml:annotation", NS)
        if sp_ann is None:
            sp_ann = ET.SubElement(sp, _q("annotation"))
        existing_sp_id = None
        for old in list(sp_ann.findall(f"{{{PW_NS}}}species")):
            existing_sp_id = old.get(f"{{{PW_NS}}}species_id")
            sp_ann.remove(old)
        pw_sp = ET.SubElement(sp_ann, _q("species", PW_NS))
        pw_sp.set(f"{{{PW_NS}}}species_type", species_nodes[sid].stype)
        if existing_sp_id is not None:
            pw_sp.set(f"{{{PW_NS}}}species_id", existing_sp_id)

    for sid, s in species_nodes.items():
        if sid not in pos:
            continue
        xd, yd = pos[sid]
        x, y = _scale_and_flip(xd, yd, scale=scale, yflip_max=y_max, margin=margin)
        if s.stype == "protein":
            w_px, h_px = protein_w_px, protein_h_px
            etype = "protein_location"
            x0, y0 = x - w_px / 2, y - h_px / 2
        else:
            w_px = h_px = compound_r * 2
            etype = "compound_location"
            x0, y0 = x - compound_r, y - compound_r
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", etype)
        le.set(f"{{{PW_NS}}}element_id", sid)
        le.set(f"{{{PW_NS}}}x", f"{x0:.2f}")
        le.set(f"{{{PW_NS}}}y", f"{y0:.2f}")
        le.set(f"{{{PW_NS}}}width", f"{w_px:.2f}")
        le.set(f"{{{PW_NS}}}height", f"{h_px:.2f}")
        le.set(f"{{{PW_NS}}}zindex", "50")
        le.set(f"{{{PW_NS}}}hidden", "false")

    pad = 60.0
    for cid in comp_names:
        members = [s.sid for s in species_nodes.values() if s.compartment == cid and s.sid in pos]
        if not members:
            continue
        pts = [pos[sid] for sid in members]
        xds = [p[0] for p in pts]; yds = [p[1] for p in pts]
        x1, y1 = _scale_and_flip(min(xds), max(yds), scale=scale, yflip_max=y_max, margin=margin)
        x2, y2 = _scale_and_flip(max(xds), min(yds), scale=scale, yflip_max=y_max, margin=margin)
        left, top = min(x1, x2) - pad, min(y1, y2) - pad
        w_box, h_box = abs(x2 - x1) + 2 * pad, abs(y2 - y1) + 2 * pad
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", "element_collection_location")
        le.set(f"{{{PW_NS}}}element_id", cid)
        le.set(f"{{{PW_NS}}}x", f"{left:.2f}")
        le.set(f"{{{PW_NS}}}y", f"{top:.2f}")
        le.set(f"{{{PW_NS}}}width", f"{w_box:.2f}")
        le.set(f"{{{PW_NS}}}height", f"{h_box:.2f}")
        le.set(f"{{{PW_NS}}}zindex", "10")
        le.set(f"{{{PW_NS}}}hidden", "false")

    for tail, head, coords in dot_edges:
        if tail not in pos or head not in pos or not coords:
            continue
        pts_px = [_scale_and_flip(x, y, scale=scale, yflip_max=y_max, margin=margin) for (x, y) in coords]
        d = [f"M {pts_px[0][0]:.2f} {pts_px[0][1]:.2f}"]
        for x, y in pts_px[1:]:
            d.append(f"L {x:.2f} {y:.2f}")
        tail_comp = species_nodes[tail].compartment if tail in species_nodes else None
        head_comp = species_nodes[head].compartment if head in species_nodes else None
        dotted = tail_comp is not None and head_comp is not None and tail_comp != head_comp
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", "edge")
        le.set(f"{{{PW_NS}}}element_id", f"{tail}__to__{head}")
        le.set(f"{{{PW_NS}}}x", "0"); le.set(f"{{{PW_NS}}}y", "0")
        le.set(f"{{{PW_NS}}}width", "0"); le.set(f"{{{PW_NS}}}height", "0")
        le.set(f"{{{PW_NS}}}zindex", "30")
        le.set(f"{{{PW_NS}}}hidden", "false")
        le.set(f"{{{PW_NS}}}path", " ".join(d))
        if dotted:
            le.set(f"{{{PW_NS}}}visualization_template_id", "83")

    # Reaction center nodes: small filled square at the layout position for each reaction.
    # pos[rxn_id] is set by graphviz (natively) or computed as centroid above (fallback).
    rc = REACTION_CENTER_SIZE
    for rxn in reactions:
        rxn_id = rxn.get("id") or ""
        if not rxn_id or rxn_id not in pos:
            continue
        mid_xd, mid_yd = pos[rxn_id]
        cx_px, cy_px = _scale_and_flip(mid_xd, mid_yd, scale=scale, yflip_max=y_max, margin=margin)
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", "reaction_center")
        le.set(f"{{{PW_NS}}}element_id", rxn_id)
        le.set(f"{{{PW_NS}}}x", f"{cx_px - rc / 2:.2f}")
        le.set(f"{{{PW_NS}}}y", f"{cy_px - rc / 2:.2f}")
        le.set(f"{{{PW_NS}}}width", f"{rc:.2f}")
        le.set(f"{{{PW_NS}}}height", f"{rc:.2f}")
        le.set(f"{{{PW_NS}}}zindex", "35")
        le.set(f"{{{PW_NS}}}hidden", "false")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    add_pathwhiz_layout(args.inp, args.out)


if __name__ == "__main__":
    main()
