"""
This script parses a PathWhiz PWML file and generates a pathway diagram
using the original layout coordinates stored in the PWML.

Key features:
- Reads node positions (x, y, width, height) from PWML location tags
- Renders compounds and proteins at their predefined coordinates
- Draws reaction edges using the exact path geometry from PWML
- Skips hidden elements to prevent unwanted artifacts
- Supports dashed/transport edges based on visualization template IDs
- Renders membrane boundaries from membrane-visualization paths
- Exports the final diagram as a PNG image

"""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Ellipse, PathPatch, Rectangle


# ----------------------------
# XML repairs + helpers
# ----------------------------

_TAG_PREFIX_RE = re.compile(r"<\/?\s*([A-Za-z_][\w.-]*)\:")
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_BARE_AMP_RE = re.compile(r"&(?!(amp|lt|gt|apos|quot|#\d+|#x[0-9A-Fa-f]+);)")


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _find_child(el: ET.Element, name: str) -> Optional[ET.Element]:
    for ch in el:
        if _strip_ns(ch.tag) == name:
            return ch
    return None


def _child_text(el: ET.Element, name: str) -> Optional[str]:
    ch = _find_child(el, name)
    if ch is None:
        return None
    return (ch.text or "").strip()


def _child_int(el: ET.Element, name: str) -> Optional[int]:
    s = _child_text(el, name)
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _child_float(el: ET.Element, name: str) -> Optional[float]:
    s = _child_text(el, name)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_xml_with_repairs(path: str) -> ET.Element:
    """
    Fix common PathWhiz PWML issues:
    - illegal control chars
    - bare ampersands
    - unbound tag prefixes like <option:end_arrow>
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = _ILLEGAL_XML_RE.sub("", text)
    text = _BARE_AMP_RE.sub("&amp;", text)

    prefixes = sorted(set(_TAG_PREFIX_RE.findall(text)))
    if prefixes:
        m = re.search(r"<super-pathway-visualization(\s[^>]*)?>", text)
        if m:
            root_open = m.group(0)
            for pfx in prefixes:
                if f"xmlns:{pfx}=" not in root_open:
                    if root_open.endswith("/>"):
                        root_open = root_open[:-2] + f' xmlns:{pfx}="urn:autofix:{pfx}"/>'
                    else:
                        root_open = root_open[:-1] + f' xmlns:{pfx}="urn:autofix:{pfx}">'
            text = text[: m.start()] + root_open + text[m.end() :]

    return ET.fromstring(text)


# ----------------------------
# SVG path -> matplotlib Path
# Supports: M, L, C, Z (that's what PWML uses)
# ----------------------------

_CMD_RE = re.compile(r"([MLCZmlcz])|(-?\d+(?:\.\d+)?)")


def svg_path_to_mpl_path(d: str) -> Optional[MplPath]:
    if not d or not d.strip():
        return None

    tokens: List[str] = [m.group(0) for m in _CMD_RE.finditer(d)]
    verts: List[Tuple[float, float]] = []
    codes: List[int] = []

    i = 0
    cmd: Optional[str] = None

    def read_num() -> float:
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    while i < len(tokens):
        t = tokens[i]
        if re.fullmatch(r"[MLCZmlcz]", t):
            cmd = t
            i += 1
            if cmd in ("Z", "z"):
                codes.append(MplPath.CLOSEPOLY)
                verts.append((0.0, 0.0))
            continue

        if cmd is None:
            return None

        if cmd in ("M", "m"):
            x, y = read_num(), read_num()
            verts.append((x, y))
            codes.append(MplPath.MOVETO)
            cmd = "L" if cmd == "M" else "l"

        elif cmd in ("L", "l"):
            x, y = read_num(), read_num()
            verts.append((x, y))
            codes.append(MplPath.LINETO)

        elif cmd in ("C", "c"):
            x1, y1 = read_num(), read_num()
            x2, y2 = read_num(), read_num()
            x3, y3 = read_num(), read_num()
            verts.extend([(x1, y1), (x2, y2), (x3, y3)])
            codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

        else:
            return None

    return MplPath(verts, codes)


# ----------------------------
# Model extraction
# ----------------------------

@dataclass
class Node:
    loc_id: int
    element_type: str
    element_id: int
    label: str
    x: float
    y: float
    w: float
    h: float
    hidden: bool
    z: int


@dataclass
class EdgeGeom:
    edge_id: int
    path: str
    template_id: str
    hidden: bool
    z: int
    end_arrow_path: Optional[str] = None
    start_arrow_path: Optional[str] = None
    end_flat_arrow_path: Optional[str] = None
    start_flat_arrow_path: Optional[str] = None


def build_id_to_name(root_all: ET.Element) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for tag in ("compound", "protein", "element-collection", "nucleic-acid", "protein-complex"):
        for el in root_all.iter():
            if _strip_ns(el.tag) != tag:
                continue
            _id = _child_int(el, "id")
            name = _child_text(el, "name")
            if _id is not None and name:
                out[_id] = name
    return out


def find_first_visualization(root_all: ET.Element) -> ET.Element:
    """
    PW000001 keeps <pathway-visualization> nested inside <pathway-visualization-contexts>.
    We pick the biggest one by width*height (usually the “main” layout).
    """
    best = None
    best_area = -1.0
    for el in root_all.iter():
        if _strip_ns(el.tag) == "pathway-visualization":
            w = _child_float(el, "width") or 0.0
            h = _child_float(el, "height") or 0.0
            area = w * h
            if area > best_area:
                best_area = area
                best = el
    return best if best is not None else root_all


def parse_nodes(root_vis: ET.Element, id_to_name: Dict[int, str]) -> Dict[int, Node]:
    allowed = {
        "compound-location",
        "protein-location",
        "element-collection-location",
        "nucleic-acid-location",
        "protein-complex-location",
    }

    nodes: Dict[int, Node] = {}

    for loc in root_vis.iter():
        tag = _strip_ns(loc.tag)
        if tag not in allowed:
            continue

        loc_id = _child_int(loc, "id")
        if loc_id is None:
            continue

        hidden = (_child_text(loc, "hidden") or "false").lower() == "true"
        z = _child_int(loc, "zindex") or 0

        element_id = None
        element_type = None
        for id_field, etype in [
            ("compound-id", "Compound"),
            ("protein-id", "Protein"),
            ("element-collection-id", "ElementCollection"),
            ("nucleic-acid-id", "NucleicAcid"),
            ("protein-complex-id", "ProteinComplex"),
        ]:
            v = _child_int(loc, id_field)
            if v is not None:
                element_id = v
                element_type = etype
                break

        if element_id is None or element_type is None:
            continue

        x = _child_float(loc, "x")
        y = _child_float(loc, "y")
        w = _child_float(loc, "width") or 40.0
        h = _child_float(loc, "height") or 24.0
        if x is None or y is None:
            continue

        # PWML locations are TOP-LEFT; convert to CENTER coords
        x = x + w / 2.0
        y = y + h / 2.0

        label = id_to_name.get(element_id, f"{element_type}:{element_id}")

        nodes[loc_id] = Node(
            loc_id=loc_id,
            element_type=element_type,
            element_id=element_id,
            label=label,
            x=x,
            y=y,
            w=w,
            h=h,
            hidden=hidden,
            z=z,
        )

    return nodes


def parse_membranes(root_vis: ET.Element) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for el in root_vis.iter():
        if _strip_ns(el.tag) != "membrane-visualization":
            continue
        d = _child_text(el, "path")
        if not d:
            continue
        z = _child_int(el, "zindex") or 0
        out.append((z, d))
    return out


def parse_edges(root_vis: ET.Element) -> List[EdgeGeom]:
    out: List[EdgeGeom] = []
    for el in root_vis.iter():
        if _strip_ns(el.tag) != "edge":
            continue
        d = _child_text(el, "path")
        if not d:
            continue

        hidden = (_child_text(el, "hidden") or "false").lower() == "true"
        z = _child_int(el, "zindex") or 0
        template = _child_text(el, "visualization-template-id") or ""

        out.append(
            EdgeGeom(
                edge_id=_child_int(el, "id") or -1,
                path=d,
                template_id=template,
                hidden=hidden,
                z=z,
                start_arrow_path=_child_text(el, "start_arrow_path"),
                end_arrow_path=_child_text(el, "end_arrow_path"),
                start_flat_arrow_path=_child_text(el, "start_flat_arrow_path"),
                end_flat_arrow_path=_child_text(el, "end_flat_arrow_path"),
            )
        )
    return out


# ----------------------------
# Rendering
# ----------------------------

def _bounds_from_paths(paths: List[str]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for d in paths:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", d)]
        xs.extend(nums[0::2])
        ys.extend(nums[1::2])
    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_label(s: str, width: int, max_lines: int = 3) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        # add ellipsis on last line
        if len(lines[-1]) >= 1:
            lines[-1] = lines[-1][:-1] + "…"
        else:
            lines[-1] = "…"
    return "\n".join(lines)


def render_pwml(pwml_path: str, out_png: str, show: bool = False) -> None:
    root_all = read_xml_with_repairs(pwml_path)
    root_vis = find_first_visualization(root_all)

    id_to_name = build_id_to_name(root_all)
    nodes = parse_nodes(root_vis, id_to_name)
    membranes = parse_membranes(root_vis)
    edges = parse_edges(root_vis)

    # Canvas bounds from all visible geometry
    geometry_paths: List[str] = [d for _, d in membranes] + [e.path for e in edges if not e.hidden]
    b = _bounds_from_paths(geometry_paths)
    if b is None:
        # fallback to node bounds
        xs: List[float] = []
        ys: List[float] = []
        for n in nodes.values():
            if n.hidden:
                continue
            xs.extend([n.x - n.w / 2, n.x + n.w / 2])
            ys.extend([n.y - n.h / 2, n.y + n.h / 2])
        if not xs:
            raise RuntimeError("No drawable geometry found in PWML.")
        b = (min(xs), max(xs), min(ys), max(ys))

    minx, maxx, miny, maxy = b
    canvas_w = maxx - minx
    canvas_h = maxy - miny

    # Figure size scales with canvas
    fig_w = max(6, canvas_w / 250)
    fig_h = max(6, canvas_h / 250)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_aspect("equal", adjustable="box")

    # Membranes (compartments)
    for z, d in sorted(membranes, key=lambda t: t[0]):
        mp = svg_path_to_mpl_path(d)
        if mp is None:
            continue
        ax.add_patch(PathPatch(mp, fill=False, linewidth=2.2, zorder=z))

    # Edges
    for e in sorted(edges, key=lambda e: e.z):
        if e.hidden:
            continue
        mp = svg_path_to_mpl_path(e.path)
        if mp is None:
            continue

        # This file uses template 83 for transport-like edges, which should be dotted.
        linestyle = "solid"
        linewidth = 1.1
        if e.template_id == "83":
            linestyle = (0, (2.2, 2.2))
            linewidth = 1.4

        ax.add_patch(PathPatch(mp, fill=False, linewidth=linewidth, linestyle=linestyle, zorder=e.z))

        # Arrowheads are provided as their own SVG paths in PWML
        for apath in (e.start_arrow_path, e.end_arrow_path, e.start_flat_arrow_path, e.end_flat_arrow_path):
            if not apath:
                continue
            amp = svg_path_to_mpl_path(apath)
            if amp is None:
                continue
            ax.add_patch(PathPatch(amp, facecolor="black", edgecolor="black", linewidth=0.0, zorder=e.z + 1))

    # Nodes (KEGG-ish)
    for n in sorted(nodes.values(), key=lambda n: n.z):
        if n.hidden:
            continue
        cx, cy = n.x, n.y

        if n.element_type in ("Compound", "ElementCollection", "NucleicAcid"):
            vis_w = _clamp(n.w, 26, 60)
            vis_h = _clamp(n.h, 26, 60)
            ax.add_patch(Ellipse((cx, cy), width=vis_w, height=vis_h, fill=False, linewidth=1.6, zorder=n.z + 5))
            ax.text(cx, cy, _wrap_label(n.label, 12, 3), ha="center", va="center", fontsize=3, zorder=n.z + 6)
        else:
            vis_w = _clamp(n.w, 50, 140)
            vis_h = _clamp(n.h, 18, 45)
            ax.add_patch(
                Rectangle((cx - vis_w / 2, cy - vis_h / 2), vis_w, vis_h, fill=False, linewidth=1.6, zorder=n.z + 5)
            )
            ax.text(cx, cy + (vis_h / 2 + 10), _wrap_label(n.label, 18, 3), ha="center", va="top", fontsize=3, zorder=n.z + 6)

    # View limits + padding
    pad = 80
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # PWML y increases downward (screen coords), so invert for matplotlib
    ax.invert_yaxis()
    ax.axis("off")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Input .pwml file")
    p.add_argument("--out", dest="out_path", required=True, help="Output PNG path")
    p.add_argument("--show", action="store_true", help="Show interactive window")
    args = p.parse_args()

    render_pwml(args.in_path, args.out_path, show=args.show)
    print(f"Wrote diagram to: {args.out_path}")


if __name__ == "__main__":
    main()
