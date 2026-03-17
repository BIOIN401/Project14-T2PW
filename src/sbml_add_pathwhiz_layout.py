#!/usr/bin/env python3
"""sbml_add_pathwhiz_layout.py

Take a *core* SBML Level-3 file (compartments/species/reactions only) and add
**PathWhiz-compatible layout annotations**.

What it produces:

1. Per-<speciesReference> annotations (what PathWhiz actually reads):
     <speciesReference>
       <annotation>
         <pathwhiz:pathwhiz pathwhiz:visualization_template_id="49">
           <pathwhiz:location pathwhiz:x="..." pathwhiz:y="..." .../>
           <pathwhiz:edge pathwhiz:d="M x1 y1 L x2 y2"/>
         </pathwhiz:pathwhiz>
       </annotation>
     </speciesReference>

2. Per-<reaction> annotations with the reaction center node and type:
     <reaction>
       <annotation>
         <pathwhiz:pathwhiz pathwhiz:reaction_type="reaction">
           <pathwhiz:location pathwhiz:x="..." pathwhiz:y="..." .../>
         </pathwhiz:pathwhiz>
       </annotation>
     </reaction>

3. Model-level canvas dimensions:
     <model>
       <annotation>
         <pathwhiz:pathwhiz>
           <pathwhiz:dimensions pathwhiz:width="..." pathwhiz:height="..."/>
         </pathwhiz:pathwhiz>
       </annotation>
     </model>

4. Model-level <pathwhiz:location_element> blocks (backward-compat for our
   local renderer, sbml_render_pathwhiz_like.py).

Layout algorithm:
- Reactions are placed left-to-right in a chain.
- Each reaction has a small center node (diamond stand-in).
- Reactants are placed to the left of the center; products to the right.
- Modifier enzymes are placed above the center.
- Edges route from node edge → reaction center (reactants/modifiers) or
  reaction center → node edge (products).

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
from typing import Dict, List, Optional, Tuple

SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
PW_NS = "http://www.spmdb.ca/pathwhiz"

ET.register_namespace("", SBML_NS)
ET.register_namespace("pathwhiz", PW_NS)

NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CANVAS_MARGIN = 80.0
NODE_OFFSET_X = 160.0      # horizontal distance from reaction center to species node center
NODE_SPACING_Y = 85.0      # vertical spacing between stacked reactants/products
MODIFIER_OFFSET_Y = 110.0  # distance above reaction center for enzyme nodes
REACTION_GAP_X = 420.0     # horizontal gap between consecutive reaction centers
CENTER_Y_BASE = 300.0      # baseline vertical center for all reactions

# Node sizes (pixels)
COMPOUND_W, COMPOUND_H = 26.0, 26.0
PROTEIN_W, PROTEIN_H = 92.0, 30.0
EC_W, EC_H = 92.0, 30.0   # element_collection same as protein

# PathWhiz visualization_template_id values
TMPL_COMPOUND_REACTANT = "49"
TMPL_COMPOUND_PRODUCT = "55"
TMPL_ELEMENT_COLLECTION = "37"
TMPL_PROTEIN = "99"


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


def _template_id(stype: str, role: str) -> str:
    """PathWhiz visualization_template_id for a species in a given role."""
    if stype == "protein":
        return TMPL_PROTEIN
    if stype == "element_collection":
        return TMPL_ELEMENT_COLLECTION
    if role == "reactant":
        return TMPL_COMPOUND_REACTANT
    return TMPL_COMPOUND_PRODUCT  # product or modifier compounds


def _build_speciesref_annotation(
    x: float,
    y: float,
    w: float,
    h: float,
    edge_d: str,
    template_id: str,
) -> ET.Element:
    """Build <annotation><pathwhiz:pathwhiz ...><pathwhiz:location/><pathwhiz:edge/></pathwhiz:pathwhiz></annotation>."""
    ann = ET.Element(_q("annotation"))
    pw = ET.SubElement(ann, _q("pathwhiz", PW_NS))
    pw.set(f"{{{PW_NS}}}visualization_template_id", template_id)
    loc = ET.SubElement(pw, _q("location", PW_NS))
    loc.set(f"{{{PW_NS}}}x", f"{x:.2f}")
    loc.set(f"{{{PW_NS}}}y", f"{y:.2f}")
    loc.set(f"{{{PW_NS}}}width", f"{w:.2f}")
    loc.set(f"{{{PW_NS}}}height", f"{h:.2f}")
    edge = ET.SubElement(pw, _q("edge", PW_NS))
    edge.set(f"{{{PW_NS}}}d", edge_d)
    return ann


# ---------------------------------------------------------------------------
# Graphviz fallback (kept for model-level renderer positions)
# ---------------------------------------------------------------------------

@dataclass
class _Species:
    sid: str
    name: str
    compartment: str
    stype: str  # compound / protein / element_collection


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
        kind = parts[0]
        if kind == "node" and len(parts) >= 6:
            pos[parts[1]] = (float(parts[2]), float(parts[3]))
        elif kind == "edge" and len(parts) >= 4:
            tail, head, n = parts[1], parts[2], int(parts[3])
            coords = [(float(parts[4 + 2 * i]), float(parts[4 + 2 * i + 1])) for i in range(n)]
            edges.append((tail, head, coords))
    return pos, edges


def _build_fallback_layout(
    species_nodes: Dict[str, _Species],
    comp_names: Dict[str, str],
    edge_pairs: List[Tuple[str, str]],
) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str, List[Tuple[float, float]]]]]:
    """Deterministic compartment-grouped grid layout (no Graphviz required)."""
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


def _scale_and_flip(
    x: float,
    y: float,
    *,
    scale: float,
    yflip_max: float,
    margin: float,
) -> Tuple[float, float]:
    return (x * scale + margin, (yflip_max - y) * scale + margin)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_pathwhiz_layout(in_path: str, out_path: str) -> None:
    tree = ET.parse(in_path)
    root = tree.getroot()

    model = root.find("sbml:model", NS)
    if model is None:
        raise ValueError("No <model> found")

    # ------------------------------------------------------------------
    # 1. Collect species info
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
    # 2. Collect reactions
    # ------------------------------------------------------------------
    reactions = root.findall(".//sbml:listOfReactions/sbml:reaction", NS)

    # Build edge_pairs for Graphviz/fallback (reactant → product)
    edge_pairs: List[Tuple[str, str]] = []
    for rxn in reactions:
        reactants = [r.get("species") for r in rxn.findall("./sbml:listOfReactants/sbml:speciesReference", NS) if r.get("species")]
        products = [p.get("species") for p in rxn.findall("./sbml:listOfProducts/sbml:speciesReference", NS) if p.get("species")]
        for r in reactants:
            for p in products:
                if r != p:
                    edge_pairs.append((r, p))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 3. Compute reaction-centered layout positions
    # ------------------------------------------------------------------
    # First reaction center x: leave room for node + margin on the left
    first_cx = CANVAS_MARGIN + NODE_OFFSET_X + 20.0

    rxn_layouts: List[Dict] = []
    for ridx, rxn in enumerate(reactions):
        reactant_sids = [r.get("species") or "" for r in rxn.findall("./sbml:listOfReactants/sbml:speciesReference", NS)]
        product_sids = [p.get("species") or "" for p in rxn.findall("./sbml:listOfProducts/sbml:speciesReference", NS)]
        modifier_sids = [m.get("species") or "" for m in rxn.findall("./sbml:listOfModifiers/sbml:modifierSpeciesReference", NS)]

        n_mod = len(modifier_sids)
        cx = first_cx + ridx * REACTION_GAP_X
        # Push center down slightly when modifiers exist so they fit above
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
        (max(len(lay["reactant_sids"]), len(lay["product_sids"]), 1) for lay in rxn_layouts),
        default=1,
    )
    canvas_h = CENTER_Y_BASE + (max_stack / 2) * NODE_SPACING_Y + CANVAS_MARGIN + MODIFIER_OFFSET_Y

    # ------------------------------------------------------------------
    # 4. Write per-speciesReference + per-reaction annotations (PathWhiz)
    # ------------------------------------------------------------------
    for rxn, lay in zip(reactions, rxn_layouts):
        cx, cy = lay["cx"], lay["cy"]
        rxn_id = rxn.get("id") or ""
        rtype = "transport" if "transport" in rxn_id else "reaction"

        # -- Reaction annotation: center node + reaction_type --
        rxn_ann = rxn.find("sbml:annotation", NS)
        if rxn_ann is None:
            rxn_ann = ET.SubElement(rxn, _q("annotation"))
        for old in list(rxn_ann.findall(f"{{{PW_NS}}}pathwhiz")):
            rxn_ann.remove(old)
        rxn_pw = ET.SubElement(rxn_ann, _q("pathwhiz", PW_NS))
        rxn_pw.set(f"{{{PW_NS}}}reaction_type", rtype)
        rxn_center_loc = ET.SubElement(rxn_pw, _q("location", PW_NS))
        rxn_center_loc.set(f"{{{PW_NS}}}x", f"{cx - 5:.2f}")
        rxn_center_loc.set(f"{{{PW_NS}}}y", f"{cy - 5:.2f}")
        rxn_center_loc.set(f"{{{PW_NS}}}width", "10.00")
        rxn_center_loc.set(f"{{{PW_NS}}}height", "10.00")

        # -- Reactants --
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

            # Edge from right edge of node → reaction center
            edge_d = f"M {node_cx + w / 2:.2f} {node_cy:.2f} L {cx:.2f} {cy:.2f}"
            tmpl = _template_id(stype, "reactant")

            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _build_speciesref_annotation(node_x, node_y, w, h, edge_d, tmpl))

        # -- Products --
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

            # Edge from reaction center → left edge of node
            edge_d = f"M {cx:.2f} {cy:.2f} L {node_cx - w / 2:.2f} {node_cy:.2f}"
            tmpl = _template_id(stype, "product")

            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _build_speciesref_annotation(node_x, node_y, w, h, edge_d, tmpl))

        # -- Modifiers (enzymes) --
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

            # Edge from bottom of modifier → reaction center
            edge_d = f"M {node_cx:.2f} {node_cy + h / 2:.2f} L {cx:.2f} {cy:.2f}"
            tmpl = _template_id(stype, "modifier")

            old_ann = ref.find("sbml:annotation", NS)
            if old_ann is not None:
                ref.remove(old_ann)
            ref.insert(0, _build_speciesref_annotation(node_x, node_y, w, h, edge_d, tmpl))

    # ------------------------------------------------------------------
    # 5. Model annotation: canvas dimensions + backward-compat
    #    location_elements for the local renderer
    # ------------------------------------------------------------------
    model_ann = model.find("sbml:annotation", NS)
    if model_ann is None:
        model_ann = ET.SubElement(model, _q("annotation"))

    # Remove stale pathwhiz:pathwhiz and old location_elements
    for old in list(model_ann.findall(f"{{{PW_NS}}}pathwhiz")):
        model_ann.remove(old)
    for old in list(model_ann.findall(f"{{{PW_NS}}}location_element")):
        model_ann.remove(old)

    # Canvas dimensions block
    model_pw = ET.SubElement(model_ann, _q("pathwhiz", PW_NS))
    dims = ET.SubElement(model_pw, _q("dimensions", PW_NS))
    dims.set(f"{{{PW_NS}}}width", f"{canvas_w:.0f}")
    dims.set(f"{{{PW_NS}}}height", f"{canvas_h:.0f}")

    # ------------------------------------------------------------------
    # 6. Graphviz / fallback layout for model-level location_elements
    #    (used by sbml_render_pathwhiz_like.py)
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
        # No species at all — still valid, skip location_elements
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    y_max = max(ys)
    scale, margin = 140.0, 80.0
    compound_r = 13.0
    protein_w_px, protein_h_px = 92.0, 30.0

    # species_type annotation on each species element (kept for renderer)
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp.get("id") or ""
        if sid not in species_nodes:
            continue
        s = species_nodes[sid]
        sp_ann = sp.find("sbml:annotation", NS)
        if sp_ann is None:
            sp_ann = ET.SubElement(sp, _q("annotation"))
        for old in list(sp_ann.findall(f"{{{PW_NS}}}species")):
            sp_ann.remove(old)
        pw_sp = ET.SubElement(sp_ann, _q("species", PW_NS))
        pw_sp.set(f"{{{PW_NS}}}species_type", s.stype)

    z_node, z_box, z_edge = 50, 10, 30
    pad = 60.0

    # Species node location_elements
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
        le.set(f"{{{PW_NS}}}zindex", str(z_node))
        le.set(f"{{{PW_NS}}}hidden", "false")

    # Compartment boxes
    for cid, cname in comp_names.items():
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
        le.set(f"{{{PW_NS}}}zindex", str(z_box))
        le.set(f"{{{PW_NS}}}hidden", "false")

    # Edges
    for tail, head, coords in dot_edges:
        if tail not in species_nodes or head not in species_nodes:
            continue
        if not coords:
            continue
        pts_px = [_scale_and_flip(x, y, scale=scale, yflip_max=y_max, margin=margin) for (x, y) in coords]
        d = [f"M {pts_px[0][0]:.2f} {pts_px[0][1]:.2f}"]
        for x, y in pts_px[1:]:
            d.append(f"L {x:.2f} {y:.2f}")
        dotted = species_nodes[tail].compartment != species_nodes[head].compartment
        le = ET.SubElement(model_ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", "edge")
        le.set(f"{{{PW_NS}}}element_id", f"{tail}__to__{head}")
        le.set(f"{{{PW_NS}}}x", "0"); le.set(f"{{{PW_NS}}}y", "0")
        le.set(f"{{{PW_NS}}}width", "0"); le.set(f"{{{PW_NS}}}height", "0")
        le.set(f"{{{PW_NS}}}zindex", str(z_edge))
        le.set(f"{{{PW_NS}}}hidden", "false")
        le.set(f"{{{PW_NS}}}path", " ".join(d))
        if dotted:
            le.set(f"{{{PW_NS}}}visualization_template_id", "83")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input SBML (core)")
    ap.add_argument("--out", dest="out", required=True, help="Output SBML with PathWhiz layout")
    args = ap.parse_args()
    add_pathwhiz_layout(args.inp, args.out)


if __name__ == "__main__":
    main()
