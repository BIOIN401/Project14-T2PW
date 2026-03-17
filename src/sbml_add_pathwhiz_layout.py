#!/usr/bin/env python3
"""sbml_add_pathwhiz_layout.py

Take a *core* SBML Level-3 file (compartments/species/reactions only) and add
**PathWhiz-style layout annotations** so it can be rendered by
`sbml_render_pathwhiz_like.py` (and similarly by PathWhiz-like tooling).

What it adds:
- <pathwhiz:species pathwhiz:species_type="..."> under each species annotation
- <pathwhiz:location_element ...> entries for:
    * compartments (as element_collection_location boxes)
    * species glyphs (compound_location / protein_location)
    * reaction edges (edge) with SVG `M ... L ...` paths

Layout strategy:
- Uses Graphviz `dot` to compute a tidy directed layout from reactions when available.
- Falls back to a deterministic compartment-grouped grid when Graphviz is unavailable.
- Separates nodes by compartment into rounded boxes.

This is not an exact clone of PathWhiz's editor layout, but it produces a clean,
non-overlapping KEGG-like diagram using embedded geometry.

Usage:
  python sbml_add_pathwhiz_layout.py --in core.sbml --out core_with_pwlayout.sbml
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

NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}


@dataclass
class Species:
    sid: str
    name: str
    compartment: str
    stype: str  # compound/protein


def _q(tag: str, ns: str = SBML_NS) -> str:
    return f"{{{ns}}}{tag}"


def _guess_species_type(sid: str) -> str:
    # Matches your json_to_sbml naming: metabolites start with m_, proteins with p_
    if sid.startswith("p_"):
        return "protein"
    if sid.startswith("m_"):
        return "compound"
    return "compound"


def _run_dot(dot_src: str) -> str:
    # Use dot plain output for deterministic parsing
    proc = subprocess.run(
        ["dot", "-Tplain"],
        input=dot_src.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.decode("utf-8", errors="replace")


def _build_fallback_layout(
    species_nodes: Dict[str, Species],
    comp_names: Dict[str, str],
    edge_pairs: List[Tuple[str, str]],
) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str, List[Tuple[float, float]]]]]:
    """Return a deterministic compartment-grouped layout when Graphviz is unavailable."""
    pos: Dict[str, Tuple[float, float]] = {}
    edges: List[Tuple[str, str, List[Tuple[float, float]]]] = []

    # Use dot-like coordinate units so the existing scale/flip path can remain unchanged.
    box_pad_x = 0.9
    box_pad_y = 0.9
    col_gap = 1.8
    row_gap = 1.3
    comp_gap = 1.8

    species_by_compartment: Dict[str, List[Species]] = {}
    for sid in sorted(species_nodes):
        species = species_nodes[sid]
        species_by_compartment.setdefault(species.compartment, []).append(species)
        comp_names.setdefault(species.compartment, species.compartment)

    current_y = 0.0
    for cid in sorted(species_by_compartment):
        members = sorted(
            species_by_compartment[cid],
            key=lambda item: (item.stype != "protein", item.name.lower(), item.sid),
        )
        if not members:
            continue

        cols = max(1, min(4, math.ceil(math.sqrt(len(members)))))
        rows = max(1, math.ceil(len(members) / cols))

        for idx, species in enumerate(members):
            row = idx // cols
            col = idx % cols
            x = box_pad_x + (col * col_gap)
            y = current_y + box_pad_y + (row * row_gap)
            pos[species.sid] = (x, y)

        box_height = ((rows - 1) * row_gap) + (2 * box_pad_y)
        current_y += box_height + comp_gap

    for tail, head in edge_pairs:
        tail_pos = pos.get(tail)
        head_pos = pos.get(head)
        if tail_pos is None or head_pos is None:
            continue
        x1, y1 = tail_pos
        x2, y2 = head_pos
        if abs(y1 - y2) < 0.01:
            coords = [(x1, y1), (x2, y2)]
        else:
            mid_x = round((x1 + x2) / 2.0, 4)
            coords = [(x1, y1), (mid_x, y1), (mid_x, y2), (x2, y2)]
        edges.append((tail, head, coords))

    return pos, edges


def _parse_dot_plain(plain: str) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str, List[Tuple[float, float]]]]]:
    """Return node positions and edges as polylines."""
    pos: Dict[str, Tuple[float, float]] = {}
    edges: List[Tuple[str, str, List[Tuple[float, float]]]] = []

    for line in plain.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        kind = parts[0]
        if kind == "node" and len(parts) >= 6:
            name = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            pos[name] = (x, y)
        elif kind == "edge" and len(parts) >= 4:
            tail = parts[1]
            head = parts[2]
            n = int(parts[3])
            coords = []
            # coords are pairs starting at index 4
            for i in range(n):
                x = float(parts[4 + 2 * i])
                y = float(parts[4 + 2 * i + 1])
                coords.append((x, y))
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
    # dot uses y-up; PathWhiz uses y-down. flip around yflip_max.
    return (x * scale + margin, (yflip_max - y) * scale + margin)


def add_pathwhiz_layout(in_path: str, out_path: str) -> None:
    tree = ET.parse(in_path)
    root = tree.getroot()

    model = root.find("sbml:model", NS)
    if model is None:
        raise ValueError("No <model> found")

    # Collect species
    species_nodes: Dict[str, Species] = {}
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp.get("id") or ""
        if not sid:
            continue
        species_nodes[sid] = Species(
            sid=sid,
            name=sp.get("name") or sid,
            compartment=sp.get("compartment") or "c_cell",
            stype=_guess_species_type(sid),
        )

    # Collect compartments
    comp_names: Dict[str, str] = {}
    for c in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        cid = c.get("id") or ""
        if not cid:
            continue
        comp_names[cid] = c.get("name") or cid

    # Build edges from reactions: connect each reactant->product
    # Keep a simple structure; dot handles routing.
    edge_pairs: List[Tuple[str, str]] = []
    for rxn in root.findall(".//sbml:listOfReactions/sbml:reaction", NS):
        reactants = [r.get("species") for r in rxn.findall("./sbml:listOfReactants/sbml:speciesReference", NS)]
        products = [p.get("species") for p in rxn.findall("./sbml:listOfProducts/sbml:speciesReference", NS)]
        reactants = [s for s in reactants if s]
        products = [s for s in products if s]
        for r in reactants:
            for p in products:
                if r != p:
                    edge_pairs.append((r, p))

    # DOT source
    # Group nodes by compartment using subgraphs to hint at clusters.
    dot_lines = ["digraph G {", "rankdir=LR;", "splines=true;", "overlap=false;"]

    for cid, cname in comp_names.items():
        # Only include compartments that actually have nodes
        members = [s.sid for s in species_nodes.values() if s.compartment == cid]
        if not members:
            continue
        dot_lines.append(f"subgraph cluster_{cid} {{")
        dot_lines.append("style=rounded;")
        dot_lines.append(f"label=\"{cname}\";")
        for sid in members:
            dot_lines.append(f"\"{sid}\";")
        dot_lines.append("}")

    # Ensure isolated nodes exist
    for sid in species_nodes:
        dot_lines.append(f"\"{sid}\";")

    for a, b in edge_pairs:
        dot_lines.append(f"\"{a}\" -> \"{b}\";")

    dot_lines.append("}")
    dot_src = "\n".join(dot_lines)

    if shutil.which("dot"):
        try:
            plain = _run_dot(dot_src)
            pos, edges = _parse_dot_plain(plain)
        except subprocess.CalledProcessError:
            pos, edges = _build_fallback_layout(species_nodes, comp_names, edge_pairs)
    else:
        pos, edges = _build_fallback_layout(species_nodes, comp_names, edge_pairs)

    if not pos:
        raise RuntimeError("Graphviz produced no node positions")

    # Determine dot coordinate bounds
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    y_max = max(ys)

    # Map dot units -> PathWhiz pixels
    scale = 140.0
    margin = 80.0

    # Species glyph sizing
    compound_r = 13.0
    protein_w, protein_h = 92.0, 30.0

    # Prepare / find model annotation to hold location_elements
    ann = model.find("sbml:annotation", NS)
    if ann is None:
        ann = ET.SubElement(model, _q("annotation"))

    # Remove existing location_elements if any (avoid duplicates)
    for le in list(ann.findall("pathwhiz:location_element", NS)):
        ann.remove(le)

    # Ensure each species has pathwhiz:species_type annotation
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp.get("id") or ""
        if sid not in species_nodes:
            continue
        s = species_nodes[sid]
        sp_ann = sp.find("sbml:annotation", NS)
        if sp_ann is None:
            sp_ann = ET.SubElement(sp, _q("annotation"))

        # remove existing pathwhiz:species
        for old in list(sp_ann.findall("pathwhiz:species", NS)):
            sp_ann.remove(old)

        pw_species = ET.SubElement(sp_ann, _q("species", PW_NS))
        pw_species.set(f"{{{PW_NS}}}species_type", s.stype)

    # Create location elements for species
    z_node = 50
    for sid, s in species_nodes.items():
        if sid not in pos:
            continue
        x_dot, y_dot = pos[sid]
        x, y = _scale_and_flip(x_dot, y_dot, scale=scale, yflip_max=y_max, margin=margin)

        if s.stype == "protein":
            w, h = protein_w, protein_h
            etype = "protein_location"
            x0, y0 = x - w / 2, y - h / 2
        else:
            w, h = compound_r * 2, compound_r * 2
            etype = "compound_location"
            x0, y0 = x - compound_r, y - compound_r

        le = ET.SubElement(ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", etype)
        le.set(f"{{{PW_NS}}}element_id", sid)
        le.set(f"{{{PW_NS}}}x", f"{x0:.2f}")
        le.set(f"{{{PW_NS}}}y", f"{y0:.2f}")
        le.set(f"{{{PW_NS}}}width", f"{w:.2f}")
        le.set(f"{{{PW_NS}}}height", f"{h:.2f}")
        le.set(f"{{{PW_NS}}}zindex", str(z_node))
        le.set(f"{{{PW_NS}}}hidden", "false")

    # Compute compartment boxes from member bounds
    z_box = 10
    pad = 60.0
    for cid, cname in comp_names.items():
        members = [s.sid for s in species_nodes.values() if s.compartment == cid and s.sid in pos]
        if not members:
            continue
        pts = [pos[sid] for sid in members]
        xds = [p[0] for p in pts]
        yds = [p[1] for p in pts]
        x_min, x_max = min(xds), max(xds)
        y_min, y_max_local = min(yds), max(yds)

        # Convert corners
        x1, y1 = _scale_and_flip(x_min, y_max_local, scale=scale, yflip_max=y_max, margin=margin)
        x2, y2 = _scale_and_flip(x_max, y_min, scale=scale, yflip_max=y_max, margin=margin)

        left = min(x1, x2) - pad
        top = min(y1, y2) - pad
        right = max(x1, x2) + pad
        bottom = max(y1, y2) + pad

        w = right - left
        h = bottom - top

        le = ET.SubElement(ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", "element_collection_location")
        le.set(f"{{{PW_NS}}}element_id", cid)
        le.set(f"{{{PW_NS}}}x", f"{left:.2f}")
        le.set(f"{{{PW_NS}}}y", f"{top:.2f}")
        le.set(f"{{{PW_NS}}}width", f"{w:.2f}")
        le.set(f"{{{PW_NS}}}height", f"{h:.2f}")
        le.set(f"{{{PW_NS}}}zindex", str(z_box))
        le.set(f"{{{PW_NS}}}hidden", "false")

    # Edges: convert polylines to SVG M/L path
    # Put edges above boxes but under nodes.
    z_edge = 30
    for tail, head, coords in edges:
        if tail not in species_nodes or head not in species_nodes:
            continue
        if not coords:
            continue

        pts = [_scale_and_flip(x, y, scale=scale, yflip_max=y_max, margin=margin) for (x, y) in coords]

        # SVG path
        d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        for (x, y) in pts[1:]:
            d.append(f"L {x:.2f} {y:.2f}")
        path_d = " ".join(d)

        # Dotted if crossing compartments
        dotted = species_nodes[tail].compartment != species_nodes[head].compartment

        le = ET.SubElement(ann, _q("location_element", PW_NS))
        le.set(f"{{{PW_NS}}}element_type", "edge")
        le.set(f"{{{PW_NS}}}element_id", f"{tail}__to__{head}")
        le.set(f"{{{PW_NS}}}x", "0")
        le.set(f"{{{PW_NS}}}y", "0")
        le.set(f"{{{PW_NS}}}width", "0")
        le.set(f"{{{PW_NS}}}height", "0")
        le.set(f"{{{PW_NS}}}zindex", str(z_edge))
        le.set(f"{{{PW_NS}}}hidden", "false")
        le.set(f"{{{PW_NS}}}path", path_d)
        if dotted:
            le.set(f"{{{PW_NS}}}visualization_template_id", "83")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input SBML (core)")
    ap.add_argument("--out", dest="out", required=True, help="Output SBML with PathWhiz-like layout")
    args = ap.parse_args()

    add_pathwhiz_layout(args.inp, args.out)


if __name__ == "__main__":
    main()
