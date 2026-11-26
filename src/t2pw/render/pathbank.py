"""PathBank-style picture of a PWML, drawn from the PWML's own geometry (VIS-3).

Nothing is laid out here. Edge and membrane paths are drawn exactly as the PWML gives them, nodes at
their x/y/width/height, PathBank's organelle and cell art (``drawable-element-location``) from
``pathbank_art/``, zoom wedges and location labels where the PWML has them. The styles per
visualization template (colours, stroke widths, shapes) were mined from 16 curated PathBank pathways
and PathBank's own SVG of each (docs/webserver/results/VIS-3.md). Unknown templates and unknown
image ids fall back to the default look or are skipped; they never raise.

``render_pwml(path, "png" | "svg") -> bytes``. One matplotlib figure gives both formats.
"""

from __future__ import annotations

import io
import json
import math
import re
import textwrap
from functools import lru_cache
from importlib import resources
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from matplotlib import image as mpimg
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D

from t2pw.pwml.render import find_first_visualization, read_xml_with_repairs, svg_path_to_mpl_path
from t2pw.render import structures as _structures

BACKGROUND = "#000099"
GOLD = "#ffc500"
# edge template -> (colour, stroke); template 5 is the plain reaction edge, 83 transport
EDGE_STYLE = {"5": (GOLD, 9), "83": ("#44ee22", 9), "148": ("#FF0000", 9), "149": ("#2AC5E5", 9),
              "195": ("#000000", 2), "218": ("#000000", 5)}
# membrane template -> (colour, band width); PathWhiz draws an unknown template like 1 (Plasma Membrane)
MEMBRANE_STYLE = {"1": ("#4BC8F2", 50), "84": ("#FF0000", 50), "91": ("#EE9189", 50), "92": ("#AED02A", 50),
                  "93": ("#FEF935", 50), "94": ("#DDB0FD", 50), "95": ("#92130A", 50), "146": ("#FF7000", 50),
                  "220": ("#000000", 6), "237": ("#389F3C", 50), "238": ("#7CC4B7", 50), "291": ("#DDB0FD", 15),
                  "304": ("#ABF7B1", 50)}
ARROW_LEG, ARROW_DEG, FLAT_HALF = 30.0, 30.0, 15.0  # PathBank chevron: two 30-unit legs at +-30 degrees
ARROW_KEYS = ("start_arrow", "end_arrow", "start_flat_arrow", "end_flat_arrow")
COMPOUND_BLUE, PROTEIN_GREEN = "#1E9EFF", "#44EE22"
PROTEIN_FILL = {"76": "#AC58FA", "77": "#AC58FA", "97": "#FE9A2E", "113": "#FE9A2E", "164": "#FE9A2E",
                "98": "#FE642E", "99": "#298A08"}
PROTEIN_STACK = {"6": 2, "77": 2, "113": 2, "16": 3, "7": 4, "8": 4, "17": 4, "164": 4}  # stacked ellipses
PROTEIN_RECT = {"76", "80"}  # transporters and receptors: rounded rectangles
PILL_FILL = {"45": "#8E7BFF", "46": "#8E7BFF", "85": "#E05050", "137": "#E05050", "138": "#E05050",
             "139": "#E05050", "140": "#E05050"}
PILL_DEFAULT = "#D8C25A"  # PathBank's ATP-style radial gradient, flattened
ICON_TEMPLATES = {"47", "49", "51", "52", "55", "56", "63", "64", "65"}  # 78x78 small-molecule icons
NAMED = ("compound", "protein", "nucleic-acid", "element-collection", "sub-pathway")
_ART = "pathbank_art"
TARGET_PX, MIN_SCALE, MAX_PX = 2400, 0.85, 8000  # see output_scale
MARGIN = 40.0  # PWML units of background kept around the drawing


def _local(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(tag, str) else ""


def _kids(el) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in el:
        out.setdefault(_local(c.tag), c)
    return out


def _txt(k: Dict[str, Any], name: str) -> str:
    c = k.get(name)
    return (c.text or "").strip() if c is not None else ""


def _num(k: Dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(_txt(k, name))
    except ValueError:
        return default


# ponytail: unbounded cache; ceiling ~82 MiB (all 41 images as float32 RGBA, measured). Fine for a per-job worker
# child; bound it (maxsize=8) if the worker becomes long-lived.
@lru_cache(maxsize=None)
def _art(image_id: str):
    """PathBank's raster of one drawable image, or ``None`` for an id we do not have."""
    if not image_id.isdigit():
        return None
    try:
        with resources.files("t2pw.render").joinpath(_ART, f"{image_id}.png").open("rb") as fh:
            return mpimg.imread(fh, format="png")
    except (FileNotFoundError, OSError):
        return None


def _fit(text: str, w: float, h: float, size: float, min_size: float = 6.0, max_lines: int = 4,
         bold: bool = False) -> Tuple[str, float]:
    """Wrap ``text`` into a ``w`` x ``h`` box at spaces and hyphens, shrinking the font until it fits (or reaches
    ``min_size``, where a word too long for the line is cut as a last resort)."""
    s = size
    while True:
        per = max(1, int(w / ((0.68 if bold else 0.6) * s)))
        lines = _wrap_chemical(text, per) or [""]
        whole = max(len(t) for t in lines) <= per
        if (whole and len(lines) <= max_lines and len(lines) * 1.2 * s <= h) or s <= min_size:
            if not whole:
                lines = textwrap.wrap(text, per) or [""]
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = lines[-1][:-1] + "…"
            return "\n".join(lines), s
        s -= 1


def _label(ax, cx: float, cy: float, text: str, w: float, h: float, size: float, z: float, *,
           color: str = "black", bold: bool = False, min_size: float = 6.0, max_lines: int = 4, transform=None):
    if not text:
        return
    t, s = _fit(text, w, h, size, min_size, max_lines, bold)
    ax.text(cx, cy, t, ha="center", va="center", fontsize=s, color=color, zorder=z + 0.5, linespacing=1.05,
            fontweight="bold" if bold else "normal", transform=transform or ax.transData, clip_on=False,
            parse_math=False)


def _rbox(x: float, y: float, w: float, h: float, r: float, **kw) -> FancyBboxPatch:
    r = max(0.0, min(r, w / 2, h / 2))
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw)


def _ends(mp: MPath) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """(start point, outward direction at start, end point, outward direction at end) of a path, or ``None``."""
    pts = [tuple(v) for v, c in zip(mp.vertices, mp.codes) if c != MPath.CLOSEPOLY]
    if len(pts) < 2:
        return None
    p0, pn = pts[0], pts[-1]
    q0 = next((p for p in pts[1:] if p != p0), None)
    qn = next((p for p in reversed(pts[:-1]) if p != pn), None)
    if q0 is None or qn is None:
        return None  # a zero-length edge has no direction
    return p0, (p0[0] - q0[0], p0[1] - q0[1]), pn, (pn[0] - qn[0], pn[1] - qn[1])


def chevron(tip: Tuple[float, float], direction: Tuple[float, float], flat: bool = False) -> List[Tuple[float, float]]:
    """The three points of an open arrowhead at ``tip`` pointing along ``direction`` (or a flat bar)."""
    n = math.hypot(*direction) or 1.0
    ux, uy = direction[0] / n, direction[1] / n
    if flat:
        return [(tip[0] - uy * FLAT_HALF, tip[1] + ux * FLAT_HALF), tip, (tip[0] + uy * FLAT_HALF, tip[1] - ux * FLAT_HALF)]
    a = math.radians(ARROW_DEG)
    legs = []
    for sgn in (1, -1):
        c, s = math.cos(sgn * a), math.sin(sgn * a)
        legs.append((tip[0] - ARROW_LEG * (ux * c - uy * s), tip[1] - ARROW_LEG * (ux * s + uy * c)))
    return [legs[0], tip, legs[1]]


def _arrow_options(edge) -> Dict[str, bool]:
    """Arrow flags from ``option:*`` children (PathBank) and/or a JSON ``options`` child (T2PW)."""
    out: Dict[str, bool] = {}
    for c in edge:
        name = _local(c.tag)
        if name in ARROW_KEYS:
            out[name] = (c.text or "").strip().lower() == "true"
        elif name == "options" and c.text:
            try:
                d = json.loads(c.text)
            except ValueError:
                continue
            if isinstance(d, dict):
                for key in ARROW_KEYS:
                    if key in d and key not in out:
                        out[key] = bool(d[key])
    return out


# -- node shapes --------------------------------------------------------------------------------------

def _is_structure_box(w: float, h: float) -> bool:
    return w >= 90 and h >= 80


def _picture(data: bytes):
    """A structure PNG as RGBA, its flat background made transparent and cropped to the drawing; ``None`` if unusable."""
    try:
        a = mpimg.imread(io.BytesIO(data), format="png")
    except Exception:  # noqa: BLE001 - a broken cache file just means an empty box
        return None
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        return None
    if a.dtype != np.float32:
        a = a.astype(np.float32) / (255.0 if a.dtype == np.uint8 else 1.0)
    rgba = np.ones(a.shape[:2] + (4,), dtype=np.float32)
    rgba[..., :a.shape[2]] = a
    rgba[..., 3] = np.where(rgba[..., :3].min(axis=2) >= 0.94, 0.0, rgba[..., 3])  # PubChem's grey 245 backdrop
    ys, xs = np.nonzero(rgba[..., 3] > 0.05)
    if len(xs) == 0:
        return None
    return rgba[max(0, ys.min() - 3):ys.max() + 4, max(0, xs.min() - 3):xs.max() + 4]


def _draw_picture(ax, pic, x, y, w, h, z) -> None:
    """``pic`` fitted into the box (x, y, w, h), aspect kept, centred. PubChem draws at about 1.4 px per bond-length
    unit of PathBank's; enlarging small molecules past 1.1x makes them heavier than PathBank's drawings."""
    ph, pw = pic.shape[:2]
    s = min(w / pw, h / ph, 1.1)
    dw, dh = pw * s, ph * s
    x0, y0 = x + (w - dw) / 2, y + (h - dh) / 2
    ax.imshow(pic, extent=(x0, x0 + dw, y0 + dh, y0), origin="upper", aspect="auto", zorder=z,
              interpolation="antialiased")


def _wrap_chemical(text: str, per: int) -> List[str]:
    """Greedy wrap at spaces and after any hyphen ("2,3-dihydro-" / "2,3-dihydroxybenzoate"), which textwrap does
    not break when a digit follows; a part longer than ``per`` stays whole (the caller shrinks the font)."""
    lines: List[str] = []
    cur = ""
    for word in text.split():
        for i, part in enumerate(re.findall(r"[^-]*-|[^-]+", word)):
            sep = "" if i else " "
            if cur and len(cur) + len(sep) + len(part) > per:
                lines.append(cur)
                cur = part
            else:
                cur = cur + sep + part if cur else part
    if cur:
        lines.append(cur)
    return lines


def _header_lines(name: str, w: float) -> Tuple[List[str], float]:
    """PathBank's compound header: Verdana 20, wrapped at word breaks, extra lines running down into the white
    body. We shrink only when a word is wider than the box, and stop at three lines."""
    s = 20.0
    while True:
        per = max(1, int((w - 10) / (0.55 * s)))  # DejaVu Sans, mixed case: ~0.55 em per character
        lines = _wrap_chemical(name, per) or [""]
        if (len(lines) <= 3 and max(len(t) for t in lines) <= per) or s <= 9:
            if len(lines) > 3:
                lines = lines[:3]
                lines[-1] = lines[-1][:-1] + "…"
            return lines, s
        s -= 1


def _compound(ax, x, y, w, h, tpl, name, short, z, pic=None):
    if _is_structure_box(w, h):  # structure box: blue header with the name, white body with the structure
        ax.add_patch(_rbox(x, y, w, h, 10, fc=COMPOUND_BLUE, ec="black", lw=1, zorder=z))
        ax.add_patch(_rbox(x + 5, y + 35, w - 10, h - 40, 10, fc="white", ec="black", lw=1, zorder=z + 0.1))
        lines, s = _header_lines(name, w)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + 18 + i * s, line, ha="center", va="center", fontsize=s, color="black",
                    zorder=z + 0.5, clip_on=False, parse_math=False)
        if pic is not None:  # PathBank: translate(25,40) scale(0.75) of a 200-unit drawing, below any extra lines
            top = y + 40 + (len(lines) - 1) * s
            _draw_picture(ax, pic, x + 20, top + 4, w - 40, y + h - 12 - top, z + 0.2)
    elif tpl == "9":  # small cofactor tag under an enzyme
        ax.add_patch(_rbox(x, y, w, h, 10, fc="#5AAEFF", ec="black", lw=1, zorder=z))
        _label(ax, x + w / 2, y + h / 2, name or short, w - 6, h - 2, 12, z, min_size=5, max_lines=3)
    elif tpl in ICON_TEMPLATES or (w >= 70 and h >= 70):
        ax.add_patch(_rbox(x + 2, y + 2, w - 4, h - 4, 12, fc="#0000DE", ec="#5577FF", lw=2, zorder=z))
        _label(ax, x + w / 2, y + h / 2, short or name, w - 10, h - 10, 14, z, color="white", bold=True, min_size=5)
    else:  # currency pill
        ax.add_patch(_rbox(x + 1, y + 1, w - 2, h - 2, 12, fc=PILL_FILL.get(tpl, PILL_DEFAULT), ec="black", lw=1, zorder=z))
        _label(ax, x + w / 2, y + h / 2, short or name, w - 4, h - 2, 12, z, bold=True, min_size=4, max_lines=2)


def _protein(ax, x, y, w, h, tpl, name, z):
    fill = PROTEIN_FILL.get(tpl, PROTEIN_GREEN)
    if tpl in PROTEIN_RECT:
        ax.add_patch(_rbox(x, y, w, h, 10, fc=fill, ec="black", lw=1, zorder=z))
    elif tpl == "136":  # multi-enzyme complex: PathBank's cloud of subunits
        for k, (fx, fy) in enumerate(((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5))):
            ax.add_patch(Ellipse((x + fx * w, y + fy * h), w * 0.6, h * 0.6, fc=fill, ec="black", lw=1,
                                 zorder=z + k * 0.01))
    else:
        n = PROTEIN_STACK.get(tpl, 1)
        ew, eh = (min(150.0, w - 20), min(70.0, h - 15)) if n > 1 else (w, h)
        dx, dy = ((w - ew) / (n - 1), (h - eh) / (n - 1)) if n > 1 else (0, 0)
        for k in range(n):
            ax.add_patch(Ellipse((x + k * dx + ew / 2, y + k * dy + eh / 2), ew, eh, fc=fill, ec="black", lw=1,
                                 zorder=z + k * 0.01))
    _label(ax, x + w / 2, y + h / 2, name, w * 0.92, h * 0.9, 15, z, min_size=8)  # PathBank: Verdana 15, wrapped


def _nucleic_acid(ax, x, y, w, h, name, z):
    kw = dict(fc="#FFFF00", ec="#8A8A00", lw=1.5, zorder=z)
    if h >= 100:  # PathBank's tRNA-like shape: a head with a short stem
        hh = h * 0.55
        ax.add_patch(Rectangle((x + w / 2 - w * 0.09, y + hh - 2), w * 0.18, h - hh - 2, **kw))
        ax.add_patch(_rbox(x + 5, y + 5, w - 10, hh - 5, 6, **kw))
        _label(ax, x + w / 2, y + 5 + (hh - 5) / 2, name, w - 16, hh - 8, 12, z)
    else:
        ax.add_patch(_rbox(x, y, w, h, min(h / 2, 20), **kw))
        _label(ax, x + w / 2, y + h / 2, name, w - 12, h - 6, 14, z)


def _collection(ax, x, y, w, h, tpl, name, z):
    if tpl == "144" or h < 80:
        ax.add_patch(Ellipse((x + w / 2, y + h / 2), w, h, fc="#C8C8C8", ec="black", lw=1, hatch="//", zorder=z))
        _label(ax, x + w / 2, y + h / 2, name, w * 0.8, h * 0.85, 15, z)
    else:
        ax.add_patch(_rbox(x, y, w, h, 10, fc="#9A9A9A", ec="black", lw=1, zorder=z))
        ax.add_patch(_rbox(x + 5, y + 35, w - 10, h - 40, 10, fc="white", ec="black", lw=1, hatch="//", zorder=z + 0.1))
        _label(ax, x + w / 2, y + 18, name, w - 10, 34, 16, z, max_lines=2)


def _sub_pathway(ax, x, y, tpl, name, z):
    if tpl == "14":  # cyan ellipse; 217 and others: white rounded box
        ax.add_patch(Ellipse((x + 75, y + 35), 150, 70, fc="#01DFD7", ec="black", lw=1, zorder=z))
    else:
        ax.add_patch(_rbox(x, y, 150, 70, 20, fc="white", ec="black", lw=3, zorder=z))
    _label(ax, x + 75, y + 35, name, 130, 62, 16, z)


# -- the picture ------------------------------------------------------------------------------------------

def _definitions(root) -> Dict[Tuple[str, str], Tuple[str, str]]:
    names: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for el in root.iter():
        t = _local(el.tag)
        if t in NAMED:
            k = _kids(el)
            i = _txt(k, "id")
            name = _txt(k, "altext" if t == "sub-pathway" else "name")  # a sub-pathway's name is its altext
            if i and name:
                names.setdefault((t, i), (name, _txt(k, "short-name") or _txt(k, "gene-name")))
    return names


def _cids(root) -> Dict[str, str]:
    """compound id -> structure key (``structures.key_for``: ChEBI id, else PubChem CID), for those that have one."""
    out: Dict[str, str] = {}
    for el in root.iter():
        if _local(el.tag) == "compound":
            k = _kids(el)
            i, key = _txt(k, "id"), _structures.key_for(_txt(k, "chebi-id"), _txt(k, "pubchem-cid"))
            if i and key:
                out.setdefault(i, key)
    return out


def _structure_cids(root, vis) -> List[str]:
    """Structure keys of the compounds drawn as structure boxes (visible, large enough)."""
    cids = _cids(root)
    want = set()
    for el in vis.iter():
        if _local(el.tag) == "compound-location":
            k = _kids(el)
            if _txt(k, "hidden").lower() != "true" and _is_structure_box(_num(k, "width"), _num(k, "height")):
                cid = cids.get(_txt(k, "compound-id"))
                if cid:
                    want.add(cid)
    return sorted(want)


def _items(root, vis, pics: Optional[Dict[str, Any]] = None, extent: Optional[List[float]] = None
           ) -> Tuple[List[Tuple[float, int, Callable[[Any], None]]], float, float]:
    """The draw calls in PathBank's order, and the drawn extent's right and bottom. ``pics``: structure key ->
    structure picture. ``extent``, when given, receives [left, top, right, bottom] of everything drawn."""
    names = _definitions(root)
    cids = _cids(root) if pics else {}
    used_edges = {(c.text or "").strip() for c in vis.iter() if _local(c.tag) == "edge-id"}
    items: List[Tuple[float, int, Callable[[Any], None]]] = []
    ext = [math.inf, math.inf, 0.0, 0.0]

    def add(z: float, fn: Callable[[Any], None], x2: float = 0.0, y2: float = 0.0,
            x1: Optional[float] = None, y1: Optional[float] = None) -> None:
        items.append((z, len(items), fn))
        ext[0], ext[1] = min(ext[0], x2 if x1 is None else x1), min(ext[1], y2 if y1 is None else y1)
        ext[2], ext[3] = max(ext[2], x2), max(ext[3], y2)

    for el in vis.iter():
        t = _local(el.tag)
        k = _kids(el) if t else {}
        if _txt(k, "hidden").lower() == "true":
            continue
        z = _num(k, "zindex")
        tpl = _txt(k, "visualization-template-id")
        x, y, w, h = _num(k, "x"), _num(k, "y"), _num(k, "width"), _num(k, "height")

        if t == "edge" and _txt(k, "id") not in used_edges:
            continue  # an orphan: no reaction, transport or interaction draws it, and PathWhiz's viewer drops it
        if t in ("membrane-visualization", "edge"):
            mp = svg_path_to_mpl_path(_txt(k, "path"))
            if mp is None or len(mp.vertices) == 0:
                continue
            bx, bn = mp.vertices.max(axis=0), mp.vertices.min(axis=0)
            if t == "membrane-visualization":
                colour, width = MEMBRANE_STYLE.get(tpl, MEMBRANE_STYLE["1"])
                add(z, lambda ax, mp=mp, c=colour, lw=width, z=z: ax.add_patch(
                    PathPatch(mp, fill=False, ec=c, lw=lw, zorder=z, capstyle="butt")),
                    bx[0] + width / 2, bx[1] + width / 2, bn[0] - width / 2, bn[1] - width / 2)
                continue
            colour, width = EDGE_STYLE.get(tpl, EDGE_STYLE["5"])  # strictly by template, as PathWhiz draws it
            add(z, lambda ax, mp=mp, c=colour, lw=width, opts=_arrow_options(el), z=z:
                _draw_edge(ax, mp, c, lw, opts, z), bx[0] + ARROW_LEG, bx[1] + ARROW_LEG,
                bn[0] - ARROW_LEG, bn[1] - ARROW_LEG)
        elif t == "compound-location":
            cid = _txt(k, "compound-id")
            name, short = names.get(("compound", cid), ("", ""))
            pic = (pics or {}).get(cids.get(cid, ""))
            add(z, lambda ax, a=(x, y, w, h, tpl, name, short, z, pic): _compound(ax, *a), x + w, y + h, x, y)
        elif t == "protein-location":
            full, gene = names.get(("protein", _txt(k, "protein-id")), ("", ""))
            lt = _txt(k, "label-type").lower()
            name = "" if lt == "none" else (gene or full) if lt == "gene" else full
            add(z, lambda ax, a=(x, y, w, h, tpl, name, z): _protein(ax, *a), x + w, y + h, x, y)
        elif t == "nucleic-acid-location":
            name = names.get(("nucleic-acid", _txt(k, "nucleic-acid-id")), ("", ""))[0]
            add(z, lambda ax, a=(x, y, w, h, name, z): _nucleic_acid(ax, *a), x + w, y + h, x, y)
        elif t == "element-collection-location":
            name = names.get(("element-collection", _txt(k, "element-collection-id")), ("", ""))[0]
            add(z, lambda ax, a=(x, y, w, h, tpl, name, z): _collection(ax, *a), x + w, y + h, x, y)
        elif t == "sub-pathway-visualization":
            name = names.get(("sub-pathway", _txt(k, "sub-pathway-id")), ("Sub-pathway", ""))[0]
            add(z, lambda ax, a=(x, y, tpl, name, z): _sub_pathway(ax, *a), x + 150, y + 70, x, y)
        elif t == "drawable-element-location":
            art = _art(_txt(k, "drawable-image-id"))
            if art is None:
                continue  # an image we do not have: draw nothing
            sx, sy = _num(k, "scale-x", 1.0), _num(k, "scale-y", 1.0)
            w = w or art.shape[1]
            h = h or art.shape[0]
            tr = Affine2D().scale(sx, sy).rotate_deg(_num(k, "degree")).translate(x, y)
            corners = tr.transform([(0, 0), (w, 0), (0, h), (w, h)])
            (cx1, cy1), (cx2, cy2) = corners.min(axis=0), corners.max(axis=0)
            add(z, lambda ax, art=art, w=w, h=h, tr=tr, z=z: ax.imshow(
                art, extent=(0, w, h, 0), origin="upper", aspect="auto", transform=tr + ax.transData,
                zorder=z, interpolation="antialiased"), cx2, cy2, cx1, cy1)
        elif t == "label-location":
            text = _txt(k, "label-text")
            sx, sy = _num(k, "scale-x", 1.0), _num(k, "scale-y", 1.0)
            add(z, lambda ax, a=(x, y, w or 200, sx, sy, text, tpl, z): _draw_label(ax, *a),
                x + ((w or 200) + 5) * sx, y + 56 * sy, x + 5 * sx, y + 25 * sy)
        elif t == "zoom-visualization":
            p = [_num(k, n) for n in ("p1x", "p1y", "p2x", "p2y", "p3x", "p3y")]
            add(z, lambda ax, p=p, c=_txt(k, "color"), z=z: _draw_zoom(ax, p, c, z),
                max(p[0] + 25, p[4]), max(p[1] + 20, p[5]), min(p[0], p[2]), min(p[1], p[3]))
    if extent is not None:
        extent[:] = ext if items else [0.0, 0.0, 0.0, 0.0]
    return items, ext[2], ext[3]


def _draw_edge(ax, mp: MPath, colour: str, lw: float, opts: Dict[str, bool], z: float) -> None:
    ax.add_patch(PathPatch(mp, fill=False, ec=colour, lw=lw, zorder=z, capstyle="butt", joinstyle="miter"))
    ends = _ends(mp)
    if ends is None:
        return
    p0, d0, pn, dn = ends
    for key, tip, d, flat in (("start_arrow", p0, d0, False), ("end_arrow", pn, dn, False),
                              ("start_flat_arrow", p0, d0, True), ("end_flat_arrow", pn, dn, True)):
        if opts.get(key):
            ax.add_patch(PathPatch(MPath(chevron(tip, d, flat)), fill=False, ec=colour, lw=lw, zorder=z,
                                   capstyle="butt", joinstyle="miter"))


def _draw_label(ax, x, y, w, sx, sy, text, tpl, z) -> None:
    tr = Affine2D().scale(sx, sy).translate(x, y) + ax.transData
    ax.add_patch(_rbox(5, 25, w, 31, 10, fc="white", ec="black", lw=(1 if tpl == "15" else 3) * sx, zorder=z, transform=tr))
    t, s = _fit(text, w - 10, 30, 16, 6, 2)
    ax.text(5 + w / 2, 25 + 15.5, t, ha="center", va="center", fontsize=s * sx, zorder=z + 0.5, transform=tr,
            linespacing=1.0, clip_on=False, parse_math=False)


def _draw_zoom(ax, p: List[float], colour: str, z: float) -> None:
    """PathBank's zoom box: a 25x20 marker at p1, a translucent wedge, and the large box p2..p3."""
    sx, sy, lx, ly, rx, ry = p
    small_r, small_b = sx + 25, sy + 20
    if ly >= small_b:            # box below the marker
        quad = [(sx, small_b), (lx, ly), (rx, ly), (small_r, small_b)]
    elif ry <= sy:               # box above
        quad = [(sx, sy), (lx, ry), (rx, ry), (small_r, sy)]
    elif lx >= small_r:          # box to the right
        quad = [(small_r, sy), (lx, ly), (lx, ry), (small_r, small_b)]
    elif rx <= sx:               # box to the left
        quad = [(sx, sy), (rx, ly), (rx, ry), (sx, small_b)]
    else:
        quad = [(sx, small_b), (lx, ly), (rx, ly), (small_r, small_b)]
    fill = colour if colour.startswith("#") else "none"
    ax.add_patch(Rectangle((lx, ly), rx - lx, ry - ly, fc=fill, ec="white", lw=2, zorder=z))
    ax.add_patch(Polygon(quad, closed=True, fc="#aabbff", ec="white", lw=2, alpha=0.5, zorder=z))
    ax.add_patch(Rectangle((sx, sy), 25, 20, fill=False, ec="white", lw=2, zorder=z))


def output_scale(width: float, height: float, max_px: Optional[int] = None) -> float:
    """Pixels per PWML unit. Aim at ``TARGET_PX`` on the longest side, but never below ``MIN_SCALE`` (PathBank's
    12-unit pill text stays >= 10 px, 15-unit enzyme text >= 12 px) and never above 1; ``max_px`` (default ``MAX_PX``) is a hard cap on the longest side."""
    longest = max(width, height)
    scale = min(1.0, max(TARGET_PX / longest, MIN_SCALE))
    return min(scale, (max_px or MAX_PX) / longest)


def render_pwml(pwml_path, fmt: str = "png", max_px: Optional[int] = None, *,
                structure_budget_s: Optional[float] = None) -> bytes:
    """Render the PWML at ``pwml_path`` as PNG or SVG bytes, cropped to what is drawn.

    The scale is ``output_scale``: labels stay readable on wide pathways, and the longest side is at most ``max_px``
    pixels (default ``MAX_PX``). Compound boxes show PubChem's structure picture of the compound (by its ChEBI
    id, else its ``pubchem-cid``; ``structures.key_for``) from the on-disk cache, fetching what is missing for at most ``structure_budget_s`` seconds (default
    ``$T2PW_STRUCTURE_BUDGET_S`` or 6; 0 = cache only); a box without a picture stays empty."""
    root = read_xml_with_repairs(str(pwml_path))
    vis = find_first_visualization(root)
    pics = {}
    for cid, data in _structures.structures(_structure_cids(root, vis), structure_budget_s).items():
        pic = _picture(data)
        if pic is not None:
            pics[cid] = pic
    extent: List[float] = []
    items, _, _ = _items(root, vis, pics, extent)
    k = _kids(vis)
    if items:  # crop to the drawing, with a margin
        x0, y0 = extent[0] - MARGIN, extent[1] - MARGIN
        width, height = extent[2] + MARGIN - x0, extent[3] + MARGIN - y0
    else:
        x0 = y0 = 0.0
        width, height = _num(k, "width"), _num(k, "height")
    if width <= 0 or height <= 0:
        raise ValueError("the PWML has no drawable extent")
    fig = Figure(figsize=(width / 72, height / 72), dpi=72, facecolor=BACKGROUND)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_autoscale_on(False)
    ax.axis("off")
    for _, _, draw in sorted(items, key=lambda it: (it[0], it[1])):
        draw(ax)
    ax.set_xlim(x0, x0 + width)
    ax.set_ylim(y0 + height, y0)
    buf = io.BytesIO()
    with rc_context({"svg.fonttype": "none"}):
        fig.savefig(buf, format=fmt, dpi=72 * output_scale(width, height, max_px), facecolor=BACKGROUND)
    return buf.getvalue()
