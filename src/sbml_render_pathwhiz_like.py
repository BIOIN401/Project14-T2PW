#!/usr/bin/env python3
"""
sbml_render_pathwhiz_like.py

Render a PathWhiz-exported SBML (Level 3 core + PathWhiz annotation namespace)
into a KEGG/PathWhiz-like pathway diagram by using the layout geometry embedded
in <pathwhiz:location_element> annotations (x, y, width, height and SVG edge paths).

This renderer does NOT compute a graph layout. It preserves the original spatial
design authored in PathWhiz, including compartment boxes and curved reaction arrows.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, PathPatch, Rectangle
from matplotlib.path import Path


SBML_NS = "http://www.sbml.org/sbml/level3/version1/core"
PW_NS = "http://www.spmdb.ca/pathwhiz"
NS = {"sbml": SBML_NS, "pathwhiz": PW_NS}


@dataclass
class SpeciesInfo:
    sid: str
    name: str
    species_type: str  # compound / protein / element_collection / etc.


@dataclass
class LocationElement:
    element_type: str
    element_id: str
    x: float
    y: float
    w: float
    h: float
    z: int
    hidden: bool
    template_id: Optional[str] = None
    path: Optional[str] = None
    options: Optional[dict] = None


# ----------------------------
# SVG path parsing (M/L/C/Q/Z)
# ----------------------------
_CMD_RE = re.compile(r"([MLCQZmlcqz])|(-?\d+(?:\.\d+)?)")

def _parse_svg_path(d: str) -> Path:
    """
    Parse a *subset* of SVG path data used by PathWhiz:
    M, L, C, Q, Z (absolute only in this exporter).
    """
    d = d.strip()
    tokens = _CMD_RE.findall(d)
    stream: List[str] = []
    for cmd, num in tokens:
        stream.append(cmd or num)

    verts: List[Tuple[float, float]] = []
    codes: List[int] = []
    i = 0
    cmd = None

    def next_float() -> float:
        nonlocal i
        v = float(stream[i]); i += 1
        return v

    while i < len(stream):
        tok = stream[i]
        if tok.isalpha():
            cmd = tok
            i += 1
        if cmd is None:
            raise ValueError("SVG path missing initial command")

        if cmd in ("M", "m"):
            x, y = next_float(), next_float()
            verts.append((x, y))
            codes.append(Path.MOVETO)
            cmd = "L" if cmd == "M" else "l"  # subsequent pairs treated as lineto
        elif cmd in ("L", "l"):
            x, y = next_float(), next_float()
            verts.append((x, y))
            codes.append(Path.LINETO)
        elif cmd in ("C", "c"):
            x1, y1 = next_float(), next_float()
            x2, y2 = next_float(), next_float()
            x3, y3 = next_float(), next_float()
            verts.extend([(x1, y1), (x2, y2), (x3, y3)])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
        elif cmd in ("Q", "q"):
            x1, y1 = next_float(), next_float()
            x2, y2 = next_float(), next_float()
            verts.extend([(x1, y1), (x2, y2)])
            codes.extend([Path.CURVE3, Path.CURVE3])
        elif cmd in ("Z", "z"):
            verts.append((0.0, 0.0))
            codes.append(Path.CLOSEPOLY)
        else:
            raise ValueError(f"Unsupported SVG command: {cmd}")

    return Path(verts, codes)


def _safe_json_from_options(opt_raw: Optional[str]) -> Optional[dict]:
    if not opt_raw:
        return None
    # PathWhiz stores JSON with XML entities (&quot;)
    try:
        s = html.unescape(opt_raw)
        return json.loads(s)
    except Exception:
        return None


# ----------------------------
# SBML + PathWhiz parsing
# ----------------------------
def parse_sbml(sbml_file: str) -> Tuple[Dict[str, SpeciesInfo], List[LocationElement]]:
    tree = ET.parse(sbml_file)
    root = tree.getroot()

    species: Dict[str, SpeciesInfo] = {}
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        sid = sp.get("id", "")
        name = sp.get("name", sid)

        # PathWhiz species_type is stored under annotation: <pathwhiz:species ... pathwhiz:species_type="..."/>
        stype = "unknown"
        ann = sp.find("sbml:annotation/pathwhiz:species", NS)
        if ann is not None:
            stype = ann.get(f"{{{PW_NS}}}species_type", "unknown")

        species[sid] = SpeciesInfo(sid=sid, name=name, species_type=stype)

    loc_elems: List[LocationElement] = []
    for le in root.findall(".//pathwhiz:location_element", NS):
        etype = le.get(f"{{{PW_NS}}}element_type", "")
        eid = le.get(f"{{{PW_NS}}}element_id", "")
        hidden = le.get(f"{{{PW_NS}}}hidden", "false").lower() == "true"
        if hidden:
            continue

        x = float(le.get(f"{{{PW_NS}}}x", "0"))
        y = float(le.get(f"{{{PW_NS}}}y", "0"))
        w = float(le.get(f"{{{PW_NS}}}width", "0"))
        h = float(le.get(f"{{{PW_NS}}}height", "0"))
        z = int(float(le.get(f"{{{PW_NS}}}zindex", "0")))
        tid = le.get(f"{{{PW_NS}}}visualization_template_id")

        path = le.get(f"{{{PW_NS}}}path")
        opts = _safe_json_from_options(le.get(f"{{{PW_NS}}}options"))

        loc_elems.append(
            LocationElement(
                element_type=etype,
                element_id=eid,
                x=x,
                y=y,
                w=w,
                h=h,
                z=z,
                hidden=False,
                template_id=tid,
                path=path,
                options=opts,
            )
        )

    return species, loc_elems


# ----------------------------
# Rendering
# ----------------------------
def _bounds_from_elements(elems: List[LocationElement]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for e in elems:
        if e.element_type == "edge" and e.path:
            try:
                p = _parse_svg_path(e.path)
                for (vx, vy) in p.vertices:
                    xs.append(vx); ys.append(vy)
            except Exception:
                pass
        else:
            xs.extend([e.x, e.x + e.w])
            ys.extend([e.y, e.y + e.h])
    if not xs or not ys:
        return (0, 0, 1000, 800)
    pad = 40
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def render(sbml_file: str, out_png: str, dpi: int = 180, show: bool = False) -> None:
    species, elems = parse_sbml(sbml_file)

    # Sort by z-index so boxes draw before nodes and edges can go on top
    elems_sorted = sorted(elems, key=lambda e: e.z)

    xmin, ymin, xmax, ymax = _bounds_from_elements(elems_sorted)
    width, height = (xmax - xmin), (ymax - ymin)

    fig_w = max(6.0, width / 120.0)
    fig_h = max(4.0, height / 120.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # invert Y to match PathWhiz screen coordinates
    ax.set_aspect("equal")
    ax.axis("off")

    # First draw non-edge shapes (compartments and nodes)
    for e in elems_sorted:
        if e.element_type == "edge":
            continue

        label = species.get(e.element_id, SpeciesInfo(e.element_id, e.element_id, "unknown")).name
        stype = species.get(e.element_id, SpeciesInfo(e.element_id, "", "unknown")).species_type

        # Compartment-like boxes / collections
        if e.element_type in ("element_collection_location", "sub_pathway"):
            box = FancyBboxPatch(
                (e.x, e.y),
                e.w,
                e.h,
                boxstyle="round,pad=0.02,rounding_size=18",
                linewidth=2.4,
                facecolor="none",
                edgecolor="black",
                zorder=e.z,
            )
            ax.add_patch(box)
            # Label near top-left
            ax.text(
                e.x + 10,
                e.y + 22,
                label,
                fontsize=9,
                ha="left",
                va="bottom",
                zorder=e.z + 1,
            )
            continue

        # Proteins: rounded rectangles
        if e.element_type in ("protein_location", "protein_complex_visualization") or stype in ("protein", "protein_complex"):
            rect = FancyBboxPatch(
                (e.x, e.y),
                e.w,
                e.h,
                boxstyle="round,pad=0.02,rounding_size=10",
                linewidth=2.0,
                facecolor="white",
                edgecolor="black",
                zorder=e.z,
            )
            ax.add_patch(rect)
            ax.text(
                e.x + e.w / 2,
                e.y + e.h + 12,
                label,
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=e.z + 1,
            )
            continue

        # Compounds: circles
        if e.element_type == "compound_location" or stype == "compound":
            r = min(e.w, e.h) / 2
            circ = Circle(
                (e.x + e.w / 2, e.y + e.h / 2),
                radius=r,
                linewidth=2.0,
                edgecolor="black",
                facecolor="white",
                zorder=e.z,
            )
            ax.add_patch(circ)
            ax.text(
                e.x + e.w / 2,
                e.y + e.h + 12,
                label,
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=e.z + 1,
            )
            continue

        # Fallback: plain rectangle
        rect = Rectangle((e.x, e.y), e.w, e.h, fill=False, linewidth=1.8, edgecolor="black", zorder=e.z)
        ax.add_patch(rect)
        ax.text(e.x + e.w / 2, e.y + e.h + 12, label, fontsize=8, ha="center", va="bottom", zorder=e.z + 1)

    # Then draw edges + arrowheads
    for e in elems_sorted:
        if e.element_type != "edge" or not e.path:
            continue

        try:
            path = _parse_svg_path(e.path)
        except Exception:
            continue

        is_dotted = (e.template_id == "83")
        patch = PathPatch(
            path,
            fill=False,
            linewidth=2.0,
            edgecolor="black",
            linestyle=(0, (2.5, 6.0)) if is_dotted else "solid",
            zorder=e.z,
        )
        ax.add_patch(patch)

        # Draw arrowheads if provided
        if e.options:
            for key in ("end_arrow_path", "start_arrow_path", "end_flat_arrow_path", "start_flat_arrow_path"):
                ap = e.options.get(key)
                if not ap:
                    continue
                try:
                    arrow_path = _parse_svg_path(ap)
                    arrow_patch = PathPatch(
                        arrow_path,
                        fill=True,
                        linewidth=1.0,
                        edgecolor="black",
                        facecolor="black",
                        zorder=e.z + 1,
                    )
                    ax.add_patch(arrow_patch)
                except Exception:
                    pass

    fig.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a PathWhiz SBML file using embedded PathWhiz layout geometry.")
    ap.add_argument("--in", dest="inp", required=True, help="Input SBML file (PathWhiz-exported).")
    ap.add_argument("--out", dest="out", default="sbml_diagram.png", help="Output PNG filename.")
    ap.add_argument("--dpi", type=int, default=180, help="PNG DPI (default 180).")
    ap.add_argument("--show", action="store_true", help="Show interactive window.")
    args = ap.parse_args()
    render(args.inp, args.out, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()
