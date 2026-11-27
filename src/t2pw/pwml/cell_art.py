"""PathBank's cell and organelle art for PWML layout v2 (VIS-2): ``drawable-element-location`` images and
``zoom-visualization`` wedges. Visualization records only; no biological record is added or changed.

Convention mined from the 16 curated PathBank reference PWMLs (docs/webserver/results/VIS-2.md section 1):
- Bacteria (PW000779, PW000921, PW000923): the bacterium image (146, 512 x 512) at scale 1.0 just left of the cell, its
  centre on the cell's centre line; a left-oriented zoom box (template 39, no colour, zindex 4) from the bacterium
  (small 25 x 20 box at 28 % / 53 % of the image) to a large box 27 units outside the cell's outer membrane; the
  bacterial chromosome (92, 479 x 481, scale 1.5-1.6) in a free part of the cytoplasm. Images: zindex 2, degree 0.
- Gram-negative envelope (PW000779): outer membrane (template 1), peptidoglycan (template 291) 120 units in, inner
  membrane (template 1) 250 units in; the periplasm is the space between.
- Eukaryotes (PW000001, PW000017, PW012911): an organelle image or PathBank's organelle cluster (90 animal: nucleus,
  rough ER, mitochondrion; 91 plant: chloroplast, nucleus, rough ER, mitochondrion) in or over the cell, and a
  top-oriented zoom box (template 4, the organelle's colour) from that organelle in the image to the box that holds
  its reactions.
PathWhiz draws the image itself from ``drawable-image-id`` (``drawable_images`` row); width / height are the image's
native size (PathWhiz's own export writes them the same way).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from t2pw.pwml.vocabulary_map import organism_kind

Rect = Tuple[int, int, int, int]
BACTERIUM, CHROMOSOME, CLUSTER_ANIMAL, CLUSTER_PLANT = 146, 92, 90, 91
IMAGE_SIZE = {BACTERIUM: (512, 512), CHROMOSOME: (479, 481), CLUSTER_ANIMAL: (484, 497), CLUSTER_PLANT: (543, 497)}
ZOOM_TOP, ZOOM_LEFT = 4, 39  # PathWhiz "Top Oriented Zoom Box" / "Left Oriented Zoom Box"
ZOOM_SMALL = (25, 20)  # the small box PathWhiz draws at p1
ZOOM_OUT = 27  # the large box sits this far outside the membrane line (PathBank: 27 / 28 / 27)
IMAGE_Z, ZOOM_Z = 2, 4
# where each organelle sits in a cluster image (fraction of width, height), read off the images
CLUSTER_SPOTS = {
    CLUSTER_ANIMAL: {"nucleus": (0.32, 0.48), "er": (0.68, 0.25), "mito": (0.60, 0.83)},
    CLUSTER_PLANT: {"plastid": (0.18, 0.20), "nucleus": (0.40, 0.50), "er": (0.70, 0.30), "mito": (0.65, 0.82)},
}
BACTERIUM_SPOT = (0.28, 0.53)
# PathWhiz's zoom background colours per organelle (zoom_visualization.rb COLOR_MAPPING)
ZOOM_COLOUR = {"mito": "#FFEBEB", "er": "#FFEEDE", "nucleus": "#FFF9C9", "plastid": "#EBFFEB"}
# compartment kinds (layout.compartment_kind) -> organelle family shown in a cluster image
FAMILY = {"mito": "mito", "mito_inner": "mito", "mito_matrix": "mito", "er": "er", "nucleus": "nucleus",
          "plastid": "plastid", "thylakoid": "plastid"}
PEPTIDOGLYCAN_TEMPLATE, ENVELOPE_GAP = 291, 250  # outer to inner membrane line (PathBank: 248-250); peptidoglycan at 0.48
GRID = 20
CHROMOSOME_ROOM = (560, 560)  # space layout v2 keeps free in a bacterial cytoplasm (VIS-5: the chromosome at 1.1 and more)
# VIS-5 (STRUCTURE_DIFF G4): PathBank draws the chromosome at scale 1.6 as the cytoplasm's centre piece, named by a label
CHROMOSOME_SCALES = (1.6, 1.4, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
CHROMOSOME_LABEL, LABEL_TEMPLATE, LABEL_SCALE = "Bacterial Chromosome", 235, 2.0


def cell_kind(items: Dict[str, List[Dict[str, Any]]], states: Iterable[Any]) -> Optional[str]:
    """"bacterium" or "eukaryote" when every given state's species is one (a state without a species counts the pathway's
    species, the writer's first species row); None otherwise (archaea, mixed or undetermined organisms: no art)."""
    species = {s.get("id"): s for s in items.get("species", [])}
    first = (items.get("species") or [{}])[0]
    by_state = {s.get("id"): s for s in items.get("biological-states", [])}
    kinds = set()
    for st in states:
        sp = species.get((by_state.get(st) or {}).get("species-id"), first)
        kinds.add(organism_kind(sp.get("name"), sp.get("classification")))
    return next(iter(kinds)) if len(kinds) == 1 and kinds <= {"bacterium", "eukaryote"} else None


def gram_negative(location_kinds: Iterable[str]) -> bool:
    """A bacterium whose PWML names an outer membrane or a periplasm (no genus table: an unnamed envelope stays unknown)."""
    return any(k in ("outer_membrane", "periplasm") for k in location_kinds)


def _drawable(new_id, image: int, x: float, y: float, scale: float) -> Dict[str, Any]:
    w, h = IMAGE_SIZE[image]
    return {"id": new_id(), "visualization-template-id": None, "x": int(round(x)), "y": int(round(y)),
            "scale-x": f"{scale:.2f}", "scale-y": f"{scale:.2f}", "degree": 0, "zindex": IMAGE_Z,
            "drawable-image-id": image, "width": str(w), "height": str(h)}


def _zoom(new_id, template: int, p1: Tuple[float, float], big: Rect, colour: Optional[str]) -> Dict[str, Any]:
    p = [int(round(v)) for v in (p1[0], p1[1], big[0], big[1], big[2], big[3])]
    xs, ys = p[0::2], p[1::2]
    return {"id": new_id(), "p1x": p[0], "p1y": p[1], "p2x": p[2], "p2y": p[3], "p3x": p[4], "p3y": p[5],
            "visualization-template-id": template, "color": colour, "zindex": ZOOM_Z, "label": None,
            "width": str(max(xs) - min(xs)), "height": str(max(ys) - min(ys))}


def free_spot(region: Rect, size: Tuple[int, int], scales: Sequence[float], rects: List[Rect],
              lines: List[List[Tuple[float, float]]], prefer: Tuple[float, float]) -> Optional[Tuple[int, int, float]]:
    """(x, y, scale) of the largest image (``size`` x scale, the first scale that fits) inside ``region`` clear of
    ``rects`` (padded 15) and ``lines`` (polylines), nearest ``prefer``; None when even the smallest scale does not fit.
    An occupancy grid of GRID units with a summed-area table."""
    x0, y0, x1, y1 = region
    gw, gh = (x1 - x0) // GRID, (y1 - y0) // GRID
    if gw <= 0 or gh <= 0:
        return None
    occ = [[0] * gw for _ in range(gh)]

    def mark(a0, b0, a1, b1):
        for gy in range(max(0, int((b0 - y0) // GRID)), min(gh, int((b1 - y0) // GRID) + 1)):
            row = occ[gy]
            for gx in range(max(0, int((a0 - x0) // GRID)), min(gw, int((a1 - x0) // GRID) + 1)):
                row[gx] = 1

    for r in rects:
        mark(r[0] - 15, r[1] - 15, r[2] + 15, r[3] + 15)
    for pts in lines:
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            n = max(1, int(max(abs(bx - ax), abs(by - ay)) // (GRID / 2)))
            for i in range(n + 1):
                px, py = ax + (bx - ax) * i / n, ay + (by - ay) * i / n
                mark(px - 10, py - 10, px + 10, py + 10)
    sat = [[0] * (gw + 1) for _ in range(gh + 1)]
    for gy in range(gh):
        acc, row, prev, cur = 0, occ[gy], sat[gy], sat[gy + 1]
        for gx in range(gw):
            acc += row[gx]
            cur[gx + 1] = prev[gx + 1] + acc
    for s in scales:
        cw, ch = -(-int(size[0] * s) // GRID), -(-int(size[1] * s) // GRID)
        best = None
        for gy in range(0, gh - ch + 1):
            for gx in range(0, gw - cw + 1):
                if sat[gy + ch][gx + cw] - sat[gy][gx + cw] - sat[gy + ch][gx] + sat[gy][gx]:
                    continue
                cx, cy = x0 + (gx + cw / 2) * GRID, y0 + (gy + ch / 2) * GRID
                d = (cx - prefer[0]) ** 2 + (cy - prefer[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, x0 + gx * GRID, y0 + gy * GRID)
        if best is not None:
            return best[1], best[2], s
    return None


def _wedge(p1: Tuple[float, float], big: Rect, template: int) -> List[Tuple[float, float]]:
    """The translucent quad PathWhiz draws (left_zoom / top_zoom.jst.eco)."""
    sw, sh = ZOOM_SMALL
    if template == ZOOM_LEFT:
        return [(p1[0] + sw, p1[1]), (big[0], big[1]), (big[0], big[3]), (p1[0] + sw, p1[1] + sh)]
    return [(p1[0], p1[1] + sh), (big[0], big[1]), (big[2], big[1]), (p1[0] + sw, p1[1] + sh)]


def wedge_crossings(p1: Tuple[float, float], big: Rect, template: int, rects: List[Rect]) -> int:
    """How many of ``rects`` outside the large box the zoom wedge covers (a rect corner or centre inside the quad)."""
    quad = _wedge(p1, big, template)

    def inside(p):
        sign = 0
        for (x1, y1), (x2, y2) in zip(quad, quad[1:] + quad[:1]):
            c = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if c:
                if sign and (c > 0) != (sign > 0):
                    return False
                sign = sign or (1 if c > 0 else -1)
        return True

    n = 0
    for r in rects:
        if not (r[2] <= big[0] or big[2] <= r[0] or r[3] <= big[1] or big[3] <= r[1]):
            continue
        n += any(inside(p) for p in ((r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3]), ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)))
    return n


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _shift_path(d: str, dx: int, dy: int) -> str:
    nums = _NUM.findall(d)
    if len(nums) % 2:
        return d
    it = iter(range(len(nums)))

    def sub(m):
        v = float(m.group(0)) + (dx if next(it) % 2 == 0 else dy)
        return str(int(v)) if v == int(v) else repr(v)
    return _NUM.sub(sub, d)


def translate(items: Dict[str, List[Dict[str, Any]]], dx: int, dy: int, loc_sections: Sequence[str], set_arrow_paths) -> None:
    """Move every drawn record by (dx, dy): locations, bound boxes, labels, membrane and edge paths (arrow paths in the
    edge options recomputed from the moved path)."""
    for sec in list(loc_sections) + ["bound-visualizations", "label-locations"]:
        for it in items.get(sec, []):
            for f, d in (("x", dx), ("y", dy)):
                if it.get(f) is not None:
                    it[f] = int(float(it[f])) + d
    for it in items.get("membrane-visualizations", []):
        it["path"] = _shift_path(str(it.get("path") or ""), dx, dy)
    for e in items.get("edges", []):
        if not e.get("path"):
            continue
        e["path"] = _shift_path(str(e["path"]), dx, dy)
        n = [int(v) if v.lstrip("-").isdigit() else float(v) for v in _NUM.findall(e["path"])]
        pts = list(zip(n[0::2], n[1::2]))
        if len(pts) >= 2:
            set_arrow_paths(e, pts[0], pts[1], pts[-1], pts[-2])


def decorate(items, new_id, width: int, height: int, plan: Dict[str, Any], obstacles: List[Rect],
             lines: List[List[Tuple[float, float]]], loc_sections: Sequence[str], set_arrow_paths, margin: int) -> Tuple[int, int]:
    """Add the art ``plan`` asks for (built by layout.relayout); returns the new canvas size."""
    out_d: List[Dict[str, Any]] = []
    out_z: List[Dict[str, Any]] = []
    if plan.get("cell") == "bacterium" and plan.get("cell_rect"):
        cx0, cy0, cx1, cy1 = plan["cell_rect"]
        host = plan.get("chromosome_region")
        if host:
            w, h = IMAGE_SIZE[CHROMOSOME]
            spot = free_spot(host, (w, h), CHROMOSOME_SCALES, obstacles + plan.get("boxes_inside", []), lines,
                             ((host[0] + host[2]) / 2, (host[1] + host[3]) / 2))
            if spot:
                out_d.append(_drawable(new_id, CHROMOSOME, spot[0], spot[1], spot[2]))
                cx, cy = spot[0] + w * spot[2] / 2, spot[1] + h * spot[2] / 2  # its label at its centre (PW000779)
                items.setdefault("label-locations", []).append({
                    "id": new_id(), "visualization-template-id": LABEL_TEMPLATE, "label-text": CHROMOSOME_LABEL,
                    "x": int(cx - 105 * LABEL_SCALE), "y": int(cy - 40.5 * LABEL_SCALE), "zindex": 20,
                    "scale-x": f"{LABEL_SCALE}", "scale-y": f"{LABEL_SCALE}", "width": "200", "height": "15"})
        bw, bh = IMAGE_SIZE[BACTERIUM]
        s = max(0.5, min(1.0, round(0.2 * (cy1 - cy0) / bh, 1)))
        iw, ih = bw * s, bh * s
        dx = max(0, int(margin + iw + 20 - (cx0 - ZOOM_OUT)))
        if dx:
            translate(items, dx, 0, loc_sections, set_arrow_paths)
            for d in out_d:
                d["x"] += dx  # (the chromosome's label is in items and moved by translate)
            cx0, cx1, width = cx0 + dx, cx1 + dx, width + dx
        ix, iy = cx0 - ZOOM_OUT - 20 - iw, max(0, (cy0 + cy1) / 2 - ih / 2)
        out_d.append(_drawable(new_id, BACTERIUM, ix, iy, s))
        p1 = (ix + BACTERIUM_SPOT[0] * iw - ZOOM_SMALL[0] / 2, iy + BACTERIUM_SPOT[1] * ih - ZOOM_SMALL[1] / 2)
        big = (max(0, cx0 - ZOOM_OUT), max(0, cy0 - ZOOM_OUT), cx1 + ZOOM_OUT, cy1 + ZOOM_OUT)
        out_z.append(_zoom(new_id, ZOOM_LEFT, p1, big, None))
        height = max(height, int(iy + ih) + margin, big[3] + margin)
    elif plan.get("cell") == "eukaryote" and plan.get("target_rect"):
        image = CLUSTER_PLANT if plan.get("plant") else CLUSTER_ANIMAL
        fam = plan["family"]
        rel = CLUSTER_SPOTS[image].get(fam)
        tx0, ty0, tx1, ty1 = plan["target_rect"]
        if rel is not None:
            w, h = IMAGE_SIZE[image]
            big_w = tx1 - tx0
            s_max = max(0.6, min(2.0, round(0.4 * big_w / w, 1)))  # VIS-5 (G5): PathBank 1.6-2.4
            scales = [round(s_max - 0.1 * i, 1) for i in range(int(round((s_max - 0.6) / 0.1)) + 1)]
            # a free spot above the target box (in the cell, as PathBank draws it), else a strip above the picture
            region = (margin, margin, width - margin, ty0 - ZOOM_OUT - 60)
            prefer = ((tx0 + tx1) / 2 - (rel[0] - 0.5) * w * s_max, ty0 - ZOOM_OUT - 60 - h * s_max / 2)
            spot = free_spot(region, (w, h), scales, obstacles + plan.get("membrane_strips", []), lines, prefer) \
                if region[3] - region[1] > h * 0.6 else None
            template = ZOOM_TOP
            s = s_max
            iw, ih = w * s, h * s
            big0 = (tx0 - ZOOM_OUT, ty0 - ZOOM_OUT, tx1 + ZOOM_OUT, ty1 + ZOOM_OUT)
            dy, dx = int(ih + 80), int(iw + 80)
            top_x = int(max(margin, min(width - margin - iw, (tx0 + tx1) / 2 - rel[0] * iw)))
            left_y = int(max(margin, (ty0 + ty1) / 2 - rel[1] * ih))
            p_top = (top_x + rel[0] * iw - ZOOM_SMALL[0] / 2, margin - dy + rel[1] * ih - ZOOM_SMALL[1] / 2)
            p_left = (margin - dx + rel[0] * iw - ZOOM_SMALL[0] / 2, left_y + rel[1] * ih - ZOOM_SMALL[1] / 2)
            n_top = wedge_crossings(p_top, big0, ZOOM_TOP, obstacles)
            n_left = wedge_crossings(p_left, big0, ZOOM_LEFT, obstacles)
            if spot is not None:  # the free spot in the picture only when its wedge covers no more nodes than a strip
                p_in = (spot[0] + rel[0] * w * spot[2] - ZOOM_SMALL[0] / 2, spot[1] + rel[1] * h * spot[2] - ZOOM_SMALL[1] / 2)
                if wedge_crossings(p_in, big0, ZOOM_TOP, obstacles) > min(n_top, n_left):
                    spot = None
            if spot is None:  # a strip above the picture or left of it, whichever wedge covers fewer nodes
                if n_left < n_top:
                    translate(items, dx, 0, loc_sections, set_arrow_paths)
                    tx0, tx1, width, template = tx0 + dx, tx1 + dx, width + dx, ZOOM_LEFT
                    spot = (margin, left_y, s)
                    height = max(height, int(left_y + ih) + margin)
                else:
                    translate(items, 0, dy, loc_sections, set_arrow_paths)
                    ty0, ty1, height = ty0 + dy, ty1 + dy, height + dy
                    spot = (top_x, margin, s)
            x, y, s = spot
            out_d.append(_drawable(new_id, image, x, y, s))
            p1 = (x + rel[0] * w * s - ZOOM_SMALL[0] / 2, y + rel[1] * h * s - ZOOM_SMALL[1] / 2)
            big = (max(0, tx0 - ZOOM_OUT), max(0, ty0 - ZOOM_OUT), tx1 + ZOOM_OUT, ty1 + ZOOM_OUT)
            out_z.append(_zoom(new_id, template, p1, big, ZOOM_COLOUR.get(fam)))
            width = max(width, big[2] + margin)
            height = max(height, big[3] + margin)
    items.setdefault("drawable-element-locations", []).extend(out_d)
    items.setdefault("zoom-visualizations", []).extend(out_z)
    return int(width), int(height)
