#!/usr/bin/env python3
"""
sbml_render_pathwhiz_like.py

Render an SBML file into a KEGG/PathWhiz-like pathway diagram.

If the SBML already contains PathWhiz-style geometry, the renderer preserves the
embedded spatial design. For core SBML files without geometry, it first synthesizes
PathWhiz-style layout annotations and then renders the result.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sbml_add_pathwhiz_layout import add_pathwhiz_layout


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

def _load_matplotlib(*, show: bool) -> Tuple[Any, Any, Any, Any, Any, Any]:
    try:
        if not show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
        from matplotlib.path import Path as MplPath
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "matplotlib is required to render SBML diagrams. Install matplotlib in the active environment."
        ) from exc
    return plt, FancyBboxPatch, Circle, PathPatch, Rectangle, MplPath


def _arrowhead_patch(verts: list, path_cls: Any, patch_cls: Any, size: float = 10.0, zorder: int = 5) -> Any:
    """
    Build a filled triangular arrowhead patch pointing from verts[-2] → verts[-1].
    *verts* is the list of (x, y) tuples from the edge path (last vertex is the tip).
    Returns a PathPatch or None if fewer than 2 usable points.
    """
    import math
    # Walk backwards to find a point that is far enough from the tip to give a direction.
    tip = None
    base_pt = None
    for v in reversed(verts):
        if tip is None:
            tip = v
            continue
        dx, dy = v[0] - tip[0], v[1] - tip[1]
        if math.hypot(dx, dy) > 1e-3:
            base_pt = v
            break
    if tip is None or base_pt is None:
        return None

    dx = tip[0] - base_pt[0]
    dy = tip[1] - base_pt[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    ux, uy = dx / length, dy / length          # unit vector along edge direction
    px, py = -uy, ux                            # perpendicular

    half_w = size * 0.45
    depth  = size

    p1 = (tip[0], tip[1])                                           # tip
    p2 = (tip[0] - ux * depth + px * half_w,
          tip[1] - uy * depth + py * half_w)                        # left base
    p3 = (tip[0] - ux * depth - px * half_w,
          tip[1] - uy * depth - py * half_w)                        # right base

    arrow_verts = [p1, p2, p3, p1]
    arrow_codes = [path_cls.MOVETO, path_cls.LINETO, path_cls.LINETO, path_cls.CLOSEPOLY]
    arrow_path  = path_cls(arrow_verts, arrow_codes)
    return patch_cls(arrow_path, fill=True, facecolor="black", edgecolor="black", linewidth=0.5, zorder=zorder)


def _parse_svg_path(d: str, path_cls: Any) -> Any:
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
            codes.append(path_cls.MOVETO)
            cmd = "L" if cmd == "M" else "l"  # subsequent pairs treated as lineto
        elif cmd in ("L", "l"):
            x, y = next_float(), next_float()
            verts.append((x, y))
            codes.append(path_cls.LINETO)
        elif cmd in ("C", "c"):
            x1, y1 = next_float(), next_float()
            x2, y2 = next_float(), next_float()
            x3, y3 = next_float(), next_float()
            verts.extend([(x1, y1), (x2, y2), (x3, y3)])
            codes.extend([path_cls.CURVE4, path_cls.CURVE4, path_cls.CURVE4])
        elif cmd in ("Q", "q"):
            x1, y1 = next_float(), next_float()
            x2, y2 = next_float(), next_float()
            verts.extend([(x1, y1), (x2, y2)])
            codes.extend([path_cls.CURVE3, path_cls.CURVE3])
        elif cmd in ("Z", "z"):
            verts.append((0.0, 0.0))
            codes.append(path_cls.CLOSEPOLY)
        else:
            raise ValueError(f"Unsupported SVG command: {cmd}")

    return path_cls(verts, codes)


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
def parse_sbml(sbml_file: str) -> Tuple[Dict[str, SpeciesInfo], Dict[str, str], List[LocationElement]]:
    tree = ET.parse(sbml_file)
    root = tree.getroot()

    compartments: Dict[str, str] = {}
    for compartment in root.findall(".//sbml:listOfCompartments/sbml:compartment", NS):
        cid = compartment.get("id", "")
        if cid:
            compartments[cid] = compartment.get("name", cid)

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

    return species, compartments, loc_elems


def _has_pathwhiz_layout(sbml_file: str) -> bool:
    tree = ET.parse(sbml_file)
    root = tree.getroot()
    # Accept either the legacy model-level location_element blocks or the
    # new per-speciesReference pathwhiz:pathwhiz annotations.
    if root.find(".//pathwhiz:location_element", NS) is not None:
        return True
    if root.find(f".//{{{PW_NS}}}pathwhiz", NS) is not None:
        return True
    return False


def summarize_layout_geometry(sbml_file: str) -> Dict[str, Any]:
    tree = ET.parse(sbml_file)
    root = tree.getroot()

    location_elements = root.findall(".//pathwhiz:location_element", NS)
    visible_elements = [
        elem
        for elem in location_elements
        if elem.get(f"{{{PW_NS}}}hidden", "false").strip().lower() != "true"
    ]

    edge_count = 0
    node_count = 0
    compartment_count = 0
    for elem in visible_elements:
        element_type = (elem.get(f"{{{PW_NS}}}element_type", "") or "").strip()
        if element_type == "edge":
            edge_count += 1
        elif element_type in {"element_collection_location", "sub_pathway"}:
            compartment_count += 1
        elif element_type:
            node_count += 1

    species_annotation_count = 0
    for sp in root.findall(".//sbml:listOfSpecies/sbml:species", NS):
        if sp.find("sbml:annotation/pathwhiz:species", NS) is not None:
            species_annotation_count += 1

    return {
        "has_pathwhiz_layout": bool(location_elements),
        "location_element_count": len(location_elements),
        "visible_location_element_count": len(visible_elements),
        "edge_count": edge_count,
        "node_count": node_count,
        "compartment_count": compartment_count,
        "species_annotation_count": species_annotation_count,
        "has_drawable_geometry": bool(visible_elements) and (edge_count > 0 or node_count > 0),
    }


def _make_local_temp_dir() -> Path:
    temp_root = Path(__file__).resolve().parent.parent / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"sbml_render_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _prepare_render_input(sbml_file: str) -> Tuple[str, Optional[Path]]:
    if _has_pathwhiz_layout(sbml_file):
        return sbml_file, None

    tmp_dir = _make_local_temp_dir()
    source_path = Path(sbml_file)
    laid_out_path = tmp_dir / f"{source_path.stem}.with_layout.sbml"
    add_pathwhiz_layout(str(source_path), str(laid_out_path))
    return str(laid_out_path), tmp_dir


# ----------------------------
# Rendering
# ----------------------------
def _bounds_from_elements(elems: List[LocationElement], path_cls: Any) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for e in elems:
        if e.element_type == "edge" and e.path:
            try:
                p = _parse_svg_path(e.path, path_cls)
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


def _render_prepared_input(render_input: str, out_png: str, dpi: int = 180, show: bool = False) -> None:
    plt, FancyBboxPatch, Circle, PathPatch, Rectangle, MplPath = _load_matplotlib(show=show)
    species, compartments, elems = parse_sbml(render_input)

    # Sort by z-index so boxes draw before nodes and edges can go on top
    elems_sorted = sorted(elems, key=lambda e: e.z)

    xmin, ymin, xmax, ymax = _bounds_from_elements(elems_sorted, MplPath)
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

        label = compartments.get(e.element_id) or species.get(
            e.element_id,
            SpeciesInfo(e.element_id, e.element_id, "unknown"),
        ).name
        stype = species.get(e.element_id, SpeciesInfo(e.element_id, "", "unknown")).species_type

        # Reaction center: small filled black square
        if e.element_type == "reaction_center":
            sq = Rectangle((e.x, e.y), e.w, e.h, linewidth=0, facecolor="black", edgecolor="none", zorder=e.z)
            ax.add_patch(sq)
            continue

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
                e.y + e.h / 2,
                label,
                fontsize=8,
                ha="center",
                va="center",
                wrap=True,
                zorder=e.z + 1,
            )
            continue

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
                e.y + e.h / 2,
                label,
                fontsize=8,
                ha="center",
                va="center",
                wrap=True,
                zorder=e.z + 1,
            )
            continue

        rect = Rectangle((e.x, e.y), e.w, e.h, fill=False, linewidth=1.8, edgecolor="black", zorder=e.z)
        ax.add_patch(rect)
        ax.text(e.x + e.w / 2, e.y + e.h / 2, label, fontsize=8, ha="center", va="center", wrap=True, zorder=e.z + 1)

    for e in elems_sorted:
        if e.element_type != "edge" or not e.path:
            continue

        try:
            path = _parse_svg_path(e.path, MplPath)
        except Exception:
            continue

        is_dotted = e.template_id == "83"
        patch = PathPatch(
            path,
            fill=False,
            linewidth=2.0,
            edgecolor="black",
            linestyle=(0, (2.5, 6.0)) if is_dotted else "solid",
            zorder=e.z,
        )
        ax.add_patch(patch)

        # Draw a geometric arrowhead on solid (non-modifier) edges regardless of
        # whether e.options carries pre-built SVG arrow paths.
        if not is_dotted:
            arrow_patch = _arrowhead_patch(path.vertices.tolist(), MplPath, PathPatch, size=10.0, zorder=e.z + 1)
            if arrow_patch is not None:
                ax.add_patch(arrow_patch)

        if e.options:
            for key in ("end_arrow_path", "start_arrow_path", "end_flat_arrow_path", "start_flat_arrow_path"):
                ap = e.options.get(key)
                if not ap:
                    continue
                try:
                    arrow_path = _parse_svg_path(ap, MplPath)
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


def build_render_artifacts(sbml_file: str, dpi: int = 180) -> Dict[str, Any]:
    render_input, layout_tmp_dir = _prepare_render_input(sbml_file)
    output_tmp_dir = _make_local_temp_dir()
    try:
        layout_summary = summarize_layout_geometry(render_input)
        layout_summary.update(
            {
                "geometry_source": "embedded" if Path(render_input).resolve() == Path(sbml_file).resolve() else "synthesized",
                "original_has_pathwhiz_layout": _has_pathwhiz_layout(sbml_file),
                "render_input_has_pathwhiz_layout": _has_pathwhiz_layout(render_input),
            }
        )

        out_path = output_tmp_dir / "sbml_diagram.png"
        _render_prepared_input(render_input, str(out_path), dpi=dpi, show=False)
        return {
            "png_bytes": out_path.read_bytes(),
            "render_ready_sbml_bytes": Path(render_input).read_bytes(),
            "layout_summary": layout_summary,
        }
    finally:
        shutil.rmtree(output_tmp_dir, ignore_errors=True)
        if layout_tmp_dir is not None:
            shutil.rmtree(layout_tmp_dir, ignore_errors=True)


def render(sbml_file: str, out_png: str, dpi: int = 180, show: bool = False) -> None:
    render_input, tmp_dir = _prepare_render_input(sbml_file)
    try:
        _render_prepared_input(render_input, out_png, dpi=dpi, show=show)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def render_to_png_bytes(sbml_file: str, dpi: int = 180) -> bytes:
    return build_render_artifacts(sbml_file, dpi=dpi)["png_bytes"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Render SBML to PNG, auto-synthesizing PathWhiz-style layout when needed.")
    ap.add_argument("--in", dest="inp", required=True, help="Input SBML file.")
    ap.add_argument("--out", dest="out", default="sbml_diagram.png", help="Output PNG filename.")
    ap.add_argument("--dpi", type=int, default=180, help="PNG DPI (default 180).")
    ap.add_argument("--show", action="store_true", help="Show interactive window.")
    args = ap.parse_args()
    render(args.inp, args.out, dpi=args.dpi, show=args.show)


if __name__ == "__main__":
    main()
