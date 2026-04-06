#!/usr/bin/env python3
"""
sbml_render_pathwhiz_like.py  — FIXED VERSION

Renders a PathWhiz-style SBML to a PNG by reading the per-speciesReference
<pathwhiz:location> annotations that json_to_sbml.py embeds.

Key fix: reads compound_location x/y/w/h and edge path from
  <speciesReference><annotation><pathwhiz:location><pathwhiz:location_element .../>
instead of from the model-level annotation block.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
PW_NS   = "http://www.spmdb.ca/pathwhiz"
NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class NodeInfo:
    sid:   str
    name:  str
    stype: str          # compound | protein | protein_complex
    x: float; y: float
    w: float; h: float
    tmpl:  str = "3"


@dataclass
class EdgeInfo:
    path:            str
    has_start_arrow: bool
    role:            str   # reactant | product | modifier


# ── XML helpers ──────────────────────────────────────────────────────────────

def _attr(el: ET.Element, local: str) -> str:
    return el.get(f"{{{PW_NS}}}{local}", "")


def _parse_species(root: ET.Element) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for sp in root.findall(f".//{{{SBML_NS}}}species"):
        sid  = sp.get("id", "")
        name = sp.get("name", sid)
        stype = "compound"
        ann = sp.find(f".//{{{PW_NS}}}species")
        if ann is not None:
            stype = ann.get(f"{{{PW_NS}}}species_type", "compound")
        out[sid] = {"name": name, "type": stype}
    return out


def _collect_le(ref: ET.Element, role: str, species: Dict[str, Dict],
                nodes: Dict[str, NodeInfo], edges: List[EdgeInfo]) -> None:
    """Walk all pathwhiz:location_element children of a speciesReference."""
    sid = ref.get("species", "")
    for le in ref.findall(f".//{{{PW_NS}}}location_element"):
        etype  = _attr(le, "element_type")
        hidden = _attr(le, "hidden") == "true"
        if hidden:
            continue
        eid = _attr(le, "element_id")

        if etype in ("compound_location", "protein_location"):
            x = float(_attr(le, "x") or 0)
            y = float(_attr(le, "y") or 0)
            w = float(_attr(le, "width") or 78)
            h = float(_attr(le, "height") or 78)
            if eid not in nodes:
                sp_info = species.get(eid, {"name": eid, "type": "compound"})
                nodes[eid] = NodeInfo(
                    sid=eid, name=sp_info["name"], stype=sp_info["type"],
                    x=x, y=y, w=w, h=h, tmpl=_attr(le, "visualization_template_id"),
                )

        elif etype == "edge":
            path     = _attr(le, "path")
            opts_raw = _attr(le, "options")
            opts: Dict = {}
            if opts_raw:
                try:
                    opts = json.loads(html.unescape(opts_raw))
                except Exception:
                    pass
            edges.append(EdgeInfo(
                path=path,
                has_start_arrow=bool(opts.get("start_arrow", False)),
                role=role,
            ))


def parse_sbml(sbml_path: str):
    tree = ET.parse(sbml_path)
    root = tree.getroot()
    species = _parse_species(root)

    nodes: Dict[str, NodeInfo] = {}
    edges: List[EdgeInfo]      = []

    for rxn in root.findall(f".//{{{SBML_NS}}}reaction"):
        for ref in rxn.findall(f".//{{{SBML_NS}}}listOfReactants/{{{SBML_NS}}}speciesReference"):
            _collect_le(ref, "reactant", species, nodes, edges)
        for ref in rxn.findall(f".//{{{SBML_NS}}}listOfProducts/{{{SBML_NS}}}speciesReference"):
            _collect_le(ref, "product", species, nodes, edges)
        for ref in rxn.findall(f".//{{{SBML_NS}}}listOfModifiers/{{{SBML_NS}}}modifierSpeciesReference"):
            _collect_le(ref, "modifier", species, nodes, edges)

    # Reaction centres: median of all edge endpoints per reaction
    import numpy as np
    rxn_centers: Dict[str, Tuple[float, float]] = {}
    for rxn in root.findall(f".//{{{SBML_NS}}}reaction"):
        rid = rxn.get("id", "")
        pts = []
        for ref in (rxn.findall(f".//{{{SBML_NS}}}speciesReference") +
                    rxn.findall(f".//{{{SBML_NS}}}modifierSpeciesReference")):
            for le in ref.findall(f".//{{{PW_NS}}}location_element"):
                if _attr(le, "element_type") == "edge" and _attr(le, "hidden") != "true":
                    nums = re.findall(r"[-+]?\d*\.?\d+", _attr(le, "path"))
                    if len(nums) >= 2:
                        pts.append((float(nums[-2]), float(nums[-1])))
        if pts:
            rxn_centers[rid] = (
                float(np.median([p[0] for p in pts])),
                float(np.median([p[1] for p in pts])),
            )

    return nodes, edges, rxn_centers


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_svg_path(ax, d: str, color: str, lw: float, zorder: int = 2):
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    tokens = re.findall(r"[MLCZmlcz]|[-+]?\d*\.?\d+", d.strip())
    pts = []; codes = []; i = 0; cmd = None
    while i < len(tokens):
        t = tokens[i]
        if t.upper() in "MLCZ":
            cmd = t.upper(); i += 1
            if cmd == "Z" and pts:
                pts.append(pts[0]); codes.append(MplPath.CLOSEPOLY)
            continue
        try:
            if cmd == "M":
                x, y = float(tokens[i]), float(tokens[i+1]); i += 2
                pts.append((x, y)); codes.append(MplPath.MOVETO); cmd = "L"
            elif cmd == "L":
                x, y = float(tokens[i]), float(tokens[i+1]); i += 2
                pts.append((x, y)); codes.append(MplPath.LINETO)
            elif cmd == "C":
                x1,y1 = float(tokens[i]),float(tokens[i+1]); i+=2
                x2,y2 = float(tokens[i]),float(tokens[i+1]); i+=2
                x3,y3 = float(tokens[i]),float(tokens[i+1]); i+=2
                pts.extend([(x1,y1),(x2,y2),(x3,y3)])
                codes.extend([MplPath.CURVE4]*3)
            else:
                i += 1
        except Exception:
            i += 1
    if len(pts) >= 2 and len(pts) == len(codes):
        ax.add_patch(PathPatch(MplPath(pts, codes), fill=False, edgecolor=color,
                               linewidth=lw, zorder=zorder, capstyle="round"))
        return pts
    return []


def _arrowhead(ax, pts, color: str, size: float = 11,
               at_start: bool = False, zorder: int = 5):
    import matplotlib.pyplot as plt
    if len(pts) < 2:
        return
    tip  = pts[0]  if at_start else pts[-1]
    rest = pts[1:] if at_start else reversed(pts[:-1])
    base = next((p for p in rest if math.hypot(p[0]-tip[0], p[1]-tip[1]) > 2), None)
    if base is None:
        return
    dx, dy = tip[0]-base[0], tip[1]-base[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return
    ux, uy = dx/L, dy/L
    hw = size * 0.45
    p1 = tip
    p2 = (tip[0]-ux*size + (-uy)*hw, tip[1]-uy*size + ux*hw)
    p3 = (tip[0]-ux*size - (-uy)*hw, tip[1]-uy*size - ux*hw)
    ax.add_patch(plt.Polygon([p1,p2,p3], closed=True, facecolor=color,
                              edgecolor=color, linewidth=0, zorder=zorder))


def _wrap(text: str, n: int = 14) -> str:
    words = text.replace("\n", " ").split()
    lines = []; cur = ""
    for w in words:
        if cur and len(cur)+1+len(w) > n:
            lines.append(cur); cur = w
        else:
            cur = (cur+" "+w).strip() if cur else w
    if cur:
        lines.append(cur)
    return "\n".join(lines[:5])


def render(sbml_path: str, out_png: str, dpi: int = 180) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Ellipse
    import numpy as np

    nodes, edges, rxn_centers = parse_sbml(sbml_path)

    if not nodes:
        raise RuntimeError(f"No layout nodes found in {sbml_path}. "
                           "Ensure json_to_sbml.py emits pathwhiz:location annotations.")

    # Canvas bounds
    all_xs, all_ys = [], []
    for n in nodes.values():
        all_xs += [n.x, n.x+n.w]; all_ys += [n.y, n.y+n.h]
    for e in edges:
        nums = re.findall(r"[-+]?\d*\.?\d+", e.path)
        for i in range(0, len(nums)-1, 2):
            try:
                all_xs.append(float(nums[i])); all_ys.append(float(nums[i+1]))
            except Exception:
                pass

    pad = 80
    xmin, xmax = min(all_xs)-pad, max(all_xs)+pad
    ymin, ymax = min(all_ys)-pad, max(all_ys)+pad

    fw = 20
    fh = fw*(ymax-ymin)/(xmax-xmin)
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)   # invert Y → PathWhiz screen coords

    # Edges
    EDGE_COLOR = "#1a1a1a"
    for e in edges:
        pts = _draw_svg_path(ax, e.path, EDGE_COLOR, lw=1.6, zorder=3)
        if pts:
            if e.has_start_arrow:
                _arrowhead(ax, pts, EDGE_COLOR, size=11, at_start=True, zorder=6)
            elif e.role == "product":
                _arrowhead(ax, pts, EDGE_COLOR, size=11, at_start=False, zorder=6)

    # Reaction centre squares
    for rid, (cx, cy) in rxn_centers.items():
        ax.add_patch(plt.Rectangle((cx-7, cy-7), 14, 14,
                                   facecolor="black", edgecolor="black", zorder=9))

    # Nodes
    for n in nodes.values():
        if n.stype in ("protein", "protein_complex"):
            ax.add_patch(FancyBboxPatch(
                (n.x, n.y), n.w, n.h,
                boxstyle="round,pad=2,rounding_size=5",
                facecolor="#ddeeff", edgecolor="#1a5fa8", linewidth=1.5, zorder=7,
            ))
            label = _wrap(n.name, 20)
            fs = max(4.5, min(7.0, 110/max(len(n.name), 1)))
            ax.text(n.x+n.w/2, n.y+n.h/2, label,
                    ha="center", va="center", fontsize=fs,
                    fontweight="bold", color="#0a2a5a", zorder=10,
                    multialignment="center", linespacing=1.2)
        else:
            cx_e, cy_e = n.x+n.w/2, n.y+n.h/2
            ax.add_patch(Ellipse((cx_e, cy_e), n.w, n.h,
                                 facecolor="white", edgecolor="#333333",
                                 linewidth=2.0, zorder=7))
            label = _wrap(n.name, 11)
            fs = max(4.5, min(7.5, 80/max(len(n.name), 1)))
            ax.text(cx_e, cy_e, label,
                    ha="center", va="center", fontsize=fs, color="#111111",
                    zorder=10, multialignment="center", linespacing=1.2)

    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Rendered: {out_png}")


def build_render_artifacts(sbml_path: str, dpi: int = 180) -> Dict[str, Any]:
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".png")
    try:
        render(sbml_path, tmp, dpi=dpi)
        png_bytes = Path(tmp).read_bytes()
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
    return {
        "png_bytes":               png_bytes,
        "render_ready_sbml_bytes": Path(sbml_path).read_bytes(),
        "layout_summary": {
            "geometry_source":          "embedded_per_speciesref",
            "has_drawable_geometry":    True,
            "visible_location_element_count": 0,
            "edge_count":               0,
        },
    }


def summarize_layout_geometry(sbml_path: str) -> Dict[str, Any]:
    nodes, edges, rxn_centers = parse_sbml(sbml_path)
    return {
        "has_pathwhiz_layout":           bool(nodes),
        "visible_location_element_count": len(nodes) + len(edges),
        "edge_count":                    sum(1 for e in edges if e.role != "modifier"),
        "node_count":                    len(nodes),
        "has_drawable_geometry":         bool(nodes and edges),
        "geometry_source":               "per_speciesref_annotation",
    }


def render_to_png_bytes(sbml_path: str, dpi: int = 180) -> bytes:
    return build_render_artifacts(sbml_path, dpi)["png_bytes"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a PathWhiz-style SBML to PNG.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default="sbml_diagram.png")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()
    render(args.inp, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
