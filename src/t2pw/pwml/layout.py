"""PWML layout v2: recompute the geometry of a built PWML (``T2PW_PWML_LAYOUT=v2``; default ``v1`` = unchanged).

Runs on the writer's ``section_items`` after every element and id exists, and changes geometry only: location
x/y, the stored width/height of locations whose template size is known (PathWhiz draws a location at its
template's size and keeps no size of its own), edge paths and the arrow paths in their options, compartment
boxes (bound-visualizations when present, plus one membrane-visualization and one label-location per
compartment) and the canvas size. Ids, references, reaction-member ``side`` (PathWhiz reads it as the reaction
side) and every other field are left as v1 writes them.

Layout: each compartment (biological state) that holds a node PathWhiz draws is a box with a label (the state's
name, never an internal one; PathBank's label template and scale); PathWhiz's viewer drops every location no process
refers to, so such nodes do not make a box of their own; inside it, each connected group of nodes gets a layered
(Sugiyama-style) left-to-right layout (substrate -> enzyme, transporter or reaction point -> product), groups are
shelf-packed in the box, and boxes are packed in rows. Edges are straight when the straight line is clear of
node boxes, otherwise routed around them on a grid.
"""
from __future__ import annotations

import heapq
import json
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from t2pw.pwml.compound_templates import TEMPLATE_DIMS

LOC_SECTIONS = ("compound-locations", "protein-locations", "nucleic-acid-locations", "element-collection-locations")
MARGIN, BAND_GAP, BAND_PAD_X, BAND_PAD_TOP, BAND_PAD_BOT = 40, 40, 40, 60, 40
HGAP, VGAP, MEMBER_GAP, MEMBERS_PER_ROW = 110, 40, 10, 3
JUNCTION_W, JUNCTION_H = 40, 20
OBSTACLE_PAD, CELL = 8, 10
ASPECT = 2.0  # target width:height of the packed picture
MEMBRANE_TEMPLATE = 220  # PathWhiz "Simple Membrane"
# Labels as in PathBank's own files (PW000001: template 235 "Simple Label - Large", scale 1.9). PathWhiz draws it as a
# rounded box at (x + 5s, y + 25s) of 200s x (15 + 16 lines)s, text wrapped at 28 characters (the template's text_length).
LABEL_TEMPLATE, LABEL_SCALE, LABEL_TEXT_LENGTH = 235, 2.0, 28
BOX_PAD_TOP_NO_LABEL, LABEL_GAP = 40, 24
# PathWhiz template sizes of non-compound location templates met in T2PW output (template 2 "Enzyme Monomer I")
OTHER_TEMPLATE_DIMS = {2: (150, 70)}


def _label_lines(text: str, length: int = LABEL_TEXT_LENGTH) -> int:
    """Line count of PathWhiz's divide_label_text (visualization_template.js.coffee): words, long ones split, greedy join."""
    words = []
    for w in text.split():
        words += [w[i:i + length] for i in range(0, len(w), length)] if len(w) > length else [w]
    lines, cur = 0, ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > length:
            lines, cur = lines + 1, w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines + (1 if cur else 0)


def label_text(state: Dict[str, Any]) -> str:
    """The state's name without the trailing " state"; nothing for an internal name such as ``__auto_state__``."""
    name = str(state.get("name") or "").strip()
    if name.endswith(" state"):
        name = name[: -len(" state")].strip()
    return "" if not name or name.startswith("__") else name


def layout_version() -> str:
    return "v2" if os.environ.get("T2PW_PWML_LAYOUT", "").strip().casefold() == "v2" else "v1"


class _Block:
    """A placed unit: one location, the protein locations of one complex, or a reaction/transport point."""

    def __init__(self, key: Any, state: Any, members: List[Dict[str, Any]], w: int, h: int, offsets: List[Tuple[int, int]]):
        self.key, self.state, self.members, self.w, self.h, self.offsets = key, state, members, w, h, offsets
        self.x = self.y = 0

    @property
    def virtual(self) -> bool:
        return not self.members

    def rect(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


def _dim(value: Any) -> int:
    return int(float(value))


def _pack_members(locs: List[Dict[str, Any]]) -> Tuple[int, int, List[Tuple[int, int]]]:
    rows = [locs[i:i + MEMBERS_PER_ROW] for i in range(0, len(locs), MEMBERS_PER_ROW)]
    offsets, y, w = [], 0, 0
    for row in rows:
        x = 0
        for loc in row:
            offsets.append((x, y))
            x += _dim(loc["width"]) + MEMBER_GAP
        w = max(w, x - MEMBER_GAP)
        y += max(_dim(loc["height"]) for loc in row) + MEMBER_GAP
    return w, y - MEMBER_GAP, offsets


def _pava(desired: List[float], seps: List[float]) -> List[float]:
    """Positions closest (least squares) to ``desired`` with pos[i] - pos[i-1] >= seps[i] (pool adjacent violators)."""
    off, acc = [], 0.0
    for i, s in enumerate(seps):
        acc += s if i else 0.0
        off.append(acc)
    blocks: List[List[float]] = []  # [mean, weight]
    for d, o in zip(desired, off):
        blocks.append([d - o, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2 = blocks.pop()
            m1, w1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    z: List[float] = []
    for m, w in blocks:
        z += [m] * int(w)
    return [a + b for a, b in zip(z, off)]


def _anchor(rect: Tuple[int, int, int, int], tx: float, ty: float) -> Tuple[int, int]:
    """Point on the rectangle border toward (tx, ty); side middles when the target lies beside the box."""
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if tx >= x1:
        return x1, int(round(cy))
    if tx <= x0:
        return x0, int(round(cy))
    return int(round(cx)), (y1 if ty >= cy else y0)


def _seg_hits(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = r
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq <= 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 >= t1:
                return False
    return True


class _Router:
    """Orthogonal A* on a CELL grid with node boxes (padded) as obstacles."""

    def __init__(self, width: int, height: int, rects: List[Tuple[int, int, int, int]]):
        self.width, self.height = width, height
        self.gw, self.gh = width // CELL + 1, height // CELL + 1
        self.block = bytearray(self.gw * self.gh)
        for r in rects:
            for gx, gy in self._cells(r, OBSTACLE_PAD):
                self.block[gy * self.gw + gx] = 1

    def _cells(self, r: Tuple[int, int, int, int], pad: int):
        x0, y0, x1, y1 = r
        for gx in range(max(0, (x0 - pad) // CELL), min(self.gw, (x1 + pad) // CELL + 1)):
            for gy in range(max(0, (y0 - pad) // CELL), min(self.gh, (y1 + pad) // CELL + 1)):
                yield gx, gy

    def route(self, a, b, own: List[Tuple[int, int, int, int]]) -> Optional[List[Tuple[int, int]]]:
        allowed = {c for r in own for c in self._cells(r, OBSTACLE_PAD)}
        s = (min(self.gw - 1, a[0] // CELL), min(self.gh - 1, a[1] // CELL))
        t = (min(self.gw - 1, b[0] // CELL), min(self.gh - 1, b[1] // CELL))
        start = (s[0], s[1], -1)
        best, prev, pq = {start: 0}, {}, [(0, 0, start)]
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        goal, expanded = None, 0
        while pq:
            _, g, st = heapq.heappop(pq)
            if g > best.get(st, 1 << 60):
                continue
            x, y, d = st
            if (x, y) == t:
                goal = st
                break
            expanded += 1
            if expanded > 250000:  # ponytail: hard cap; an unroutable edge falls back to a straight line
                return None
            for nd, (dx, dy) in enumerate(dirs):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.gw and 0 <= ny < self.gh):
                    continue
                if self.block[ny * self.gw + nx] and (nx, ny) not in allowed:
                    continue
                ng = g + 1 + (6 if d not in (-1, nd) else 0)
                ns = (nx, ny, nd)
                if ng < best.get(ns, 1 << 60):
                    best[ns], prev[ns] = ng, st
                    heapq.heappush(pq, (ng + abs(nx - t[0]) + abs(ny - t[1]), ng, ns))
        if goal is None:
            return None
        cells = [goal]
        while cells[-1] in prev:
            cells.append(prev[cells[-1]])
        cells.reverse()
        pts = [(min(c[0] * CELL + CELL // 2, self.width), min(c[1] * CELL + CELL // 2, self.height)) for c in cells]
        corners = [pts[0]] + [q for p, q, r in zip(pts, pts[1:], pts[2:]) if (p[0] == q[0]) != (q[0] == r[0])] + [pts[-1]]
        # replace the grid ends by the exact anchors, keeping the first and last legs straight
        if len(corners) >= 3:
            c1, cl = corners[1], corners[-2]
            corners[1] = (a[0], c1[1]) if corners[0][0] == c1[0] else (c1[0], a[1])
            corners[-2] = (b[0], cl[1]) if corners[-1][0] == cl[0] else (cl[0], b[1])
        return [a] + corners[1:-1] + [b]


def _path(points: List[Tuple[int, int]]) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    """Straight cubic segments (the form PathWhiz edits and draws arrows from); returns path, first cp1, last cp2."""
    pts = [points[0]] + [p for p, q in zip(points[1:], points) if p != q]
    if len(pts) == 1:
        pts = pts * 2
    parts = [f"M{pts[0][0]} {pts[0][1]}"]
    cp1 = cp2 = pts[0]
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts, pts[1:])):
        c1 = (x1 + (x2 - x1) // 3, y1 + (y2 - y1) // 3)
        c2 = (x1 + 2 * (x2 - x1) // 3, y1 + 2 * (y2 - y1) // 3)
        parts.append(f"C{c1[0]} {c1[1]} {c2[0]} {c2[1]} {x2} {y2}")
        if i == 0:
            cp1 = c1
        cp2 = c2
    return " ".join(parts), cp1, cp2


def _options(raw: Any) -> Dict[str, Any]:
    try:
        o = json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        return {}
    return o if isinstance(o, dict) else {}


def _set_arrow_paths(edge: Dict[str, Any], start, cp1, end, cp2) -> None:
    from t2pw.pwml.writer import _compute_arrow_path, _compute_flat_arrow_path

    raw = edge.get("options")
    if not raw or not isinstance(raw, str):
        return
    try:
        opts = json.loads(raw)
    except ValueError:
        return
    if not isinstance(opts, dict):
        return
    changed = False
    for key, fn, tip, frm in (
        ("start_arrow_path", _compute_arrow_path, start, cp1),
        ("start_flat_arrow_path", _compute_flat_arrow_path, start, cp1),
        ("end_arrow_path", _compute_arrow_path, end, cp2),
        ("end_flat_arrow_path", _compute_flat_arrow_path, end, cp2),
    ):
        if opts.get(key) is not None:
            opts[key] = fn(tip[0], tip[1], frm[0], frm[1])
            changed = True
    if changed:
        edge["options"] = json.dumps(opts)


def _path_points(d: str) -> List[Tuple[float, float]]:
    nums = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", d)]
    return list(zip(nums[0::2], nums[1::2]))


def _rounded_rect(x0: int, y0: int, x1: int, y1: int, r: int = 30) -> str:
    """Closed membrane path in PathBank's form: M then C segments only, ending where it started."""
    def seg(a, b):
        return f"C{a[0] + (b[0] - a[0]) // 3} {a[1] + (b[1] - a[1]) // 3} {a[0] + 2 * (b[0] - a[0]) // 3} {a[1] + 2 * (b[1] - a[1]) // 3} {b[0]} {b[1]}"

    def corner(a, c, b):
        return f"C{(a[0] + c[0]) // 2} {(a[1] + c[1]) // 2} {(c[0] + b[0]) // 2} {(c[1] + b[1]) // 2} {b[0]} {b[1]}"

    p = [(x0, y0 + r), (x0 + r, y0), (x1 - r, y0), (x1, y0 + r), (x1, y1 - r), (x1 - r, y1), (x0 + r, y1), (x0, y1 - r)]
    out = [f"M{p[0][0]} {p[0][1]}", corner(p[0], (x0, y0), p[1]), seg(p[1], p[2]), corner(p[2], (x1, y0), p[3]),
           seg(p[3], p[4]), corner(p[4], (x1, y1), p[5]), seg(p[5], p[6]), corner(p[6], (x0, y1), p[7]), seg(p[7], p[0])]
    return " ".join(out) + " "


def _layered(blocks: Dict[Any, "_Block"], keys: List[Any], flow: List[Tuple[Any, Any]]) -> Tuple[int, int]:
    """Sugiyama-style layout of one connected group: columns by longest path (back edges dropped), order by
    barycenter sweeps, y by least-squares alignment with neighbours (no overlap). Sets block x/y relative to
    the group's top-left corner; returns the group's (width, height)."""
    inside = set(keys)
    succ: Dict[Any, List[Any]] = defaultdict(list)
    for u, v in flow:
        if u in inside and v in inside and u != v and v not in succ[u]:
            succ[u].append(v)
    mark: Dict[Any, int] = {}
    dag: Dict[Any, List[Any]] = defaultdict(list)
    for start in keys:
        if start in mark:
            continue
        mark[start] = 1
        stack = [(start, iter(succ[start]))]
        while stack:
            u, it = stack[-1]
            for v in it:
                if mark.get(v) == 1:
                    continue  # back edge (a cycle): not used for layering
                dag[u].append(v)
                if v not in mark:
                    mark[v] = 1
                    stack.append((v, iter(succ[v])))
                    break
            else:
                mark[u] = 2
                stack.pop()
    pred: Dict[Any, List[Any]] = defaultdict(list)
    for u in keys:
        for v in dag[u]:
            pred[v].append(u)
    order: List[Any] = []
    indeg = {k: len(pred[k]) for k in keys}
    queue = [k for k in keys if not indeg[k]]
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in dag[u]:
            indeg[v] -= 1
            if not indeg[v]:
                queue.append(v)
    layer = {k: 0 for k in keys}
    for u in order:
        for v in dag[u]:
            layer[v] = max(layer[v], layer[u] + 1)
    for u in reversed(order):  # a pure source sits just before its first successor
        if not pred[u] and dag[u]:
            layer[u] = min(layer[v] for v in dag[u]) - 1
    low = min(layer.values())
    n_layers = max(layer.values()) - low + 1
    cells: List[List[Any]] = [[] for _ in range(n_layers)]
    for k in order:
        cells[layer[k] - low].append(k)
    pos: Dict[Any, int] = {}

    def renumber():
        for cell in cells:
            for i, k in enumerate(cell):
                pos[k] = i

    renumber()
    for sweep in range(8):
        nb = pred if sweep % 2 == 0 else dag
        for cell in cells:
            bary = {k: (sum(pos[n] for n in nb[k]) / len(nb[k]) if nb[k] else pos[k]) for k in cell}
            cell.sort(key=lambda k: bary[k])
        renumber()
    yc: Dict[Any, float] = {}
    for cell in cells:
        cur = 0.0
        for k in cell:
            yc[k] = cur + blocks[k].h / 2
            cur += blocks[k].h + VGAP
    for sweep in range(8):
        nb = pred if sweep % 2 == 0 else dag
        for cell in (cells if sweep % 2 == 0 else cells[::-1]):
            if not cell:
                continue
            desired = [(sum(yc[n] for n in nb[k]) / len(nb[k]) if nb[k] else yc[k]) for k in cell]
            seps = [0.0] + [(blocks[a].h + blocks[c].h) / 2 + VGAP for a, c in zip(cell, cell[1:])]
            for k, y in zip(cell, _pava(desired, seps)):
                yc[k] = y
    colw = [max((blocks[k].w for k in cell), default=0) for cell in cells]
    colx, x = [], 0
    for w in colw:
        colx.append(x)
        x += w + HGAP
    lo = min(yc[k] - blocks[k].h / 2 for k in keys)
    for k in keys:
        bl, l = blocks[k], layer[k] - low
        bl.x = colx[l] + (colw[l] - bl.w) // 2
        bl.y = int(round(yc[k] - bl.h / 2 - lo))
    for k in order:  # a reaction point with no enzyme sits right next to its substrates
        ins = [blocks[p] for p in pred[k] if layer[p] == layer[k] - 1]
        if blocks[k].virtual and ins:
            blocks[k].x = max(p.x + p.w for p in ins) + 12
    return x - HGAP, int(round(max(blocks[k].y + blocks[k].h for k in keys)))


def relayout(items: Dict[str, List[Dict[str, Any]]], new_id: Callable[[], int], ir: Optional[Dict[str, Any]] = None,
             key_ids: Optional[Dict[str, Dict[str, int]]] = None) -> Tuple[int, int]:
    """Re-place every visual element of ``items`` in place; returns the new canvas (width, height).

    ``ir`` and ``key_ids`` (the writer's IR and its IR-key -> PWML-id map) only serve to place edges that no
    visualization member references; an edge still tied to nothing drawn keeps its v1 shape, moved into the canvas."""
    locs: Dict[int, Dict[str, Any]] = {}
    hidden_locs: List[Dict[str, Any]] = []
    for sec in LOC_SECTIONS:
        for it in items.get(sec, []):
            dims = (TEMPLATE_DIMS if sec == "compound-locations" else OTHER_TEMPLATE_DIMS).get(it.get("visualization-template-id"))
            if dims:  # PathWhiz draws a location at its template's size and ignores the stored one
                it["width"], it["height"] = str(dims[0]), str(dims[1])
            (hidden_locs.append(it) if it.get("hidden") else locs.__setitem__(int(it["id"]), it))
    edges = {int(e["id"]): e for e in items.get("edges", [])}

    # ---- blocks: one per visible location, protein locations of one complex visualization grouped
    parent = {lid: lid for lid in locs}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pcv_locs: Dict[int, List[int]] = {}
    for pcv in items.get("protein-complex-visualizations", []):
        ids = [int(m["protein-location-id"]) for m in pcv.get("protein_complex_protein_visualizations", [])
               if m.get("protein-location-id") is not None and int(m["protein-location-id"]) in locs]
        pcv_locs[int(pcv["id"])] = ids
        for other in ids[1:]:
            ra, rb = find(ids[0]), find(other)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: Dict[int, List[int]] = defaultdict(list)
    for lid in sorted(locs):
        groups[find(lid)].append(lid)
    blocks: Dict[Any, _Block] = {}
    block_of_loc: Dict[int, Any] = {}
    for root, lids in groups.items():
        members = [locs[i] for i in lids]
        w, h, offs = _pack_members(members)
        blocks[root] = _Block(root, members[0].get("biological-state-id"), members, w, h, offs)
        for i in lids:
            block_of_loc[i] = root

    def loc_block(lid):
        return block_of_loc.get(int(lid)) if lid is not None else None

    def pcv_block(pid):
        ids = pcv_locs.get(int(pid)) if pid is not None else None
        return block_of_loc.get(ids[0]) if ids else None

    def junction(state):
        key = ("j", len(blocks))
        blocks[key] = _Block(key, state, [], JUNCTION_W, JUNCTION_H, [])
        return key

    # ---- process units: layering edges and, per edge, (start block, end block)
    flow: List[Tuple[Any, Any]] = []
    edge_ends: Dict[int, Tuple[Any, Any]] = {}
    attached: set = set()  # blocks a process visualization refers to; PathWhiz's viewer drops the rest (isOrphan)
    hub_of: Dict[Tuple[str, Any], Any] = {}
    for rv in items.get("reaction-visualizations", []):
        hubs = []
        for ev in rv.get("reaction_enzyme_visualizations", []):
            k = pcv_block(ev.get("protein-complex-visualization-id")) or loc_block(ev.get("protein-location-id"))
            if k is not None and k not in hubs:
                hubs.append(k)
        attached.update(hubs)
        members = []
        for sec, fld in (("reaction_compound_visualizations", "compound-location-id"),
                         ("reaction_element_collection_visualizations", "element-collection-location-id"),
                         ("reaction_nucleic_acid_visualizations", "nucleic-acid-location-id")):
            members += [(m, loc_block(m.get(fld))) for m in rv.get(sec, [])]
        if not members:
            continue
        hub = hubs[0] if hubs else junction(rv.get("biological-state-id"))
        hub_of[("r", rv.get("reaction-id"))] = hub
        for m, k in members:
            e = edges.get(int(m["edge-id"])) if m.get("edge-id") is not None else None
            if e is not None:
                edge_ends[int(e["id"])] = (k if k is not None else hub, hub)
            if k is None:
                continue
            opts = e.get("options") if e is not None else None
            product = _options(opts).get("start_arrow") is True
            for h in hubs or [hub]:
                flow.append((h, k) if product else (k, h))
    for tv in items.get("transport-visualizations", []):
        hubs = []
        for tt in tv.get("transport_transporter_visualizations", []):
            k = pcv_block(tt.get("protein-complex-visualization-id")) or loc_block(tt.get("protein-location-id"))
            if k is not None and k not in hubs:
                hubs.append(k)
        attached.update(hubs)
        members = [(m, loc_block(m.get("compound-location-id"))) for m in tv.get("transport_compound_visualizations", [])]
        if not members:
            continue
        first = next((k for m, k in members if k is not None), None)
        hub = hubs[0] if hubs else junction(blocks[first].state if first is not None else None)
        hub_of[("t", tv.get("transport-id"))] = hub
        for m, k in members:
            left = m.get("side") == "Left"
            if m.get("edge-id") is not None and int(m["edge-id"]) in edges:
                kk = k if k is not None else hub
                edge_ends[int(m["edge-id"])] = (kk, hub) if left else (hub, kk)
            if k is not None:
                flow.append((k, hub) if left else (hub, k))
    for iv in items.get("interaction-visualizations", []):
        ends: Dict[str, List[Tuple[Dict[str, Any], Any]]] = {"Left": [], "Right": []}
        for sec, fld in (("interaction_compound_visualizations", "compound-location-id"),
                         ("interaction_nucleic_acid_visualizations", "nucleic-acid-location-id"),
                         ("interaction_element_collection_visualizations", "element-collection-location-id")):
            for m in iv.get(sec, []):
                ends.setdefault(m.get("side"), []).append((m, loc_block(m.get(fld))))
        for m in iv.get("interaction_protein_complex_visualizations", []):
            ends.setdefault(m.get("side"), []).append((m, pcv_block(m.get("protein-complex-visualization-id"))))
        attached.update(k for side in ends.values() for _, k in side if k is not None)
        lk = next((k for _, k in ends["Left"] if k is not None), None)
        rk = next((k for _, k in ends["Right"] if k is not None), None)
        if (lk is None) != (rk is None) and ends["Left"] and ends["Right"]:
            # one side has nothing drawable (e.g. a complex visualization with no protein): a point beside the other
            stub = junction(blocks[lk if lk is not None else rk].state)
            lk, rk = (lk, stub) if lk is not None else (stub, rk)
        if lk is not None and rk is not None:
            flow.append((lk, rk))
        for side, own, other in (("Left", lk, rk), ("Right", rk, lk)):
            for m, k in ends.get(side, []):
                if m.get("edge-id") is not None and int(m["edge-id"]) in edges and (k or own) is not None:
                    edge_ends[int(m["edge-id"])] = (k or own, other if other is not None else (k or own))

    # Edges no visualization member points at: the IR's enzyme/transporter connectors. v2 draws a reaction
    # through its enzyme, so such a connector runs from its member to the reaction's hub (a point when equal).
    ids = key_ids or {}
    for viz in (ir or {}).get("process_visualizations") or []:
        kind = {"reaction_visualization": "r", "transport_visualization": "t"}.get(str(viz.get("type")))
        hub = hub_of.get((kind, ids.get("processes", {}).get(str(viz.get("process_key")))))
        for m in viz.get("members") or []:
            eid = ids.get("edges", {}).get(str(m.get("edge_key")))
            if hub is None or eid is None or eid not in edges or eid in edge_ends:
                continue
            lk = str(m.get("location_key"))
            own = loc_block(ids.get("locations", {}).get(lk)) or pcv_block(ids.get("protein_complex_visualizations", {}).get(lk))
            edge_ends[eid] = (own if own is not None else hub, hub)

    # ---- compartments: a box per state holding a node PathWhiz draws; each connected group laid out in layers
    attached.update(k for pair in flow for k in pair)
    keys = list(blocks)
    boxed = {blocks[k].state for k in keys if k in attached and not blocks[k].virtual}
    nbrs: Dict[Any, List[Any]] = defaultdict(list)
    for u, v in flow:
        nbrs[u].append(v)
        nbrs[v].append(u)
    for k in keys:  # a reaction/transport point joins the box of a drawn neighbour when its own state has none
        if blocks[k].virtual and blocks[k].state not in boxed:
            st = next((blocks[n].state for n in nbrs[k] if blocks[n].state in boxed), None)
            if st is not None or boxed:
                blocks[k].state = st if st is not None else sorted(boxed, key=str)[0]
    loose_key = ("loose",)  # unattached nodes of a state without a box: packed after the boxes, no outline
    by_state: Dict[Any, List[Any]] = defaultdict(list)
    for k in keys:
        st = blocks[k].state
        by_state[st if st in boxed else loose_key].append(k)
    link: Dict[Tuple[Any, Any], int] = defaultdict(int)
    for u, v in flow:
        su, sv = blocks[u].state, blocks[v].state
        if su != sv:
            link[(su, sv)] += 1
            link[(sv, su)] += 1
    remaining = sorted((s for s in by_state if s != loose_key), key=lambda s: (-len(by_state[s]), str(s)))
    bands = [remaining.pop(0)] if remaining else []
    while remaining:  # heavily linked compartments next to each other
        nxt = max(remaining, key=lambda s: (link[(bands[-1], s)], len(by_state[s]), -remaining.index(s)))
        remaining.remove(nxt)
        bands.append(nxt)

    comps_by_state: Dict[Any, List[Tuple[List[Any], int, int]]] = {}
    for s in bands:
        ks = set(by_state[s])
        und: Dict[Any, List[Any]] = defaultdict(list)
        for u, v in flow:
            if u in ks and v in ks and u != v:
                und[u].append(v)
                und[v].append(u)
        seen, comps = set(), []
        for k in by_state[s]:
            if k in seen:
                continue
            comp, todo = [], [k]
            seen.add(k)
            while todo:
                u = todo.pop()
                comp.append(u)
                for v in und[u]:
                    if v not in seen:
                        seen.add(v)
                        todo.append(v)
            comp = [c for c in by_state[s] if c in set(comp)]
            w, h = _layered(blocks, comp, flow)
            comps.append((comp, w, h))
        comps.sort(key=lambda c: (not any(k in attached for k in c[0]), -(c[1] * c[2])))  # drawn groups first
        comps_by_state[s] = comps
    if loose_key in by_state:
        comps_by_state[loose_key] = sorted(((([k], *_layered(blocks, [k], flow))) for k in by_state[loose_key]),
                                           key=lambda c: -(c[1] * c[2]))

    area = sum((w + HGAP) * (h + VGAP) for cs in comps_by_state.values() for _, w, h in cs)
    target = max([int((area * ASPECT) ** 0.5)] + [w for cs in comps_by_state.values() for _, w, _ in cs])
    # inside each compartment its groups are shelf-packed; the compartment boxes are then packed in rows
    states = {s.get("id"): s for s in items.get("biological-states", [])}
    labels = {st: label_text(states.get(st, {})) for st in bands}
    label_w = int((5 + 200) * LABEL_SCALE)

    def pad_top(st):
        text = labels.get(st, "")
        return int(4 + (40 + 16 * _label_lines(text)) * LABEL_SCALE) + LABEL_GAP if text else BOX_PAD_TOP_NO_LABEL

    boxes = []
    for st in bands + ([loose_key] if loose_key in comps_by_state else []):
        placed, cx, cy, shelf_h, used_w = [], 0, 0, 0, 0
        for comp, w, h in comps_by_state[st]:
            if cx > 0 and cx + w > target:
                cx, cy, shelf_h = 0, cy + shelf_h + 2 * VGAP, 0
            placed.append((comp, cx, cy))
            used_w, shelf_h, cx = max(used_w, cx + w), max(shelf_h, h), cx + w + HGAP
        if labels.get(st):
            used_w = max(used_w, label_w)
        boxes.append((st, used_w + 2 * BAND_PAD_X, cy + shelf_h + pad_top(st) + BAND_PAD_BOT, placed))
    row_limit = max([target + 2 * BAND_PAD_X] + [bw for _, bw, _, _ in boxes])
    rows: List[List[Any]] = []
    x = row_w = 0
    for box in boxes:
        if not rows or (x > 0 and x + box[1] > row_limit):
            rows.append([])
            x = 0
        rows[-1].append((x, box))
        x += box[1] + BAND_GAP
    band_rects: List[Tuple[Any, Tuple[int, int, int, int]]] = []
    top = MARGIN
    for row in rows:
        row_h = max(box[2] for _, box in row)
        for bx, (st, bw, bh, placed) in row:
            for comp, dx, dy in placed:
                for k in comp:
                    blocks[k].x += MARGIN + bx + BAND_PAD_X + dx
                    blocks[k].y += top + pad_top(st) + dy
            if st != loose_key:
                band_rects.append((st, (MARGIN + bx, top, MARGIN + bx + bw, top + row_h)))
            row_w = max(row_w, bx + bw)
        top += row_h + BAND_GAP
    width = MARGIN + row_w + MARGIN
    for bl in blocks.values():
        for loc, (dx, dy) in zip(bl.members, bl.offsets):
            loc["x"], loc["y"] = bl.x + dx, bl.y + dy

    # ---- compartments: bound-visualizations (when written), membranes and labels
    rect_of_state = dict(band_rects)
    extra_x, extra_y = MARGIN, top
    for bv in items.get("bound-visualizations", []):
        r = rect_of_state.get(bv.get("biological-state-id"))
        if r is None:  # a state with nothing drawn in it: small box in a row under the bands
            r = (extra_x, extra_y, extra_x + 200, extra_y + 80)
            extra_x += 240
        bv["x"], bv["y"], bv["width"], bv["height"] = r[0], r[1], str(r[2] - r[0]), str(r[3] - r[1])
    height = (extra_y + 80 + MARGIN) if extra_x > MARGIN else (top - BAND_GAP + MARGIN)
    width = max(width, extra_x + MARGIN)
    for s, (x0, y0, x1, y1) in band_rects:
        items.setdefault("membrane-visualizations", []).append({
            "id": new_id(), "path": _rounded_rect(x0, y0, x1, y1), "visualization-template-id": MEMBRANE_TEMPLATE,
            "complete-membrane": True, "zindex": 6, "width": f"{x1 - x0}.0", "height": f"{y1 - y0}.0"})
        if labels.get(s):
            items.setdefault("label-locations", []).append({
                "id": new_id(), "visualization-template-id": LABEL_TEMPLATE, "label-text": labels[s],
                "x": x0 + 12, "y": y0 + 4, "zindex": 20, "scale-x": f"{LABEL_SCALE}", "scale-y": f"{LABEL_SCALE}",
                "width": "200", "height": "15"})
    for loc in hidden_locs:  # not drawn; kept inside the canvas
        loc["x"] = max(0, min(int(loc.get("x") or 0), width - _dim(loc["width"])))
        loc["y"] = max(0, min(int(loc.get("y") or 0), height - _dim(loc["height"])))

    # ---- edges
    member_rects = {k: [(int(l["x"]), int(l["y"]), int(l["x"]) + _dim(l["width"]), int(l["y"]) + _dim(l["height"]))
                        for l in bl.members] for k, bl in blocks.items()}
    all_rects = [r for rs in member_rects.values() for r in rs]
    router = _Router(width, height, all_rects)
    routed = 0
    for eid, (a, b) in edge_ends.items():
        e, ba, bb = edges[eid], blocks[a], blocks[b]
        ca, cb = ba.center(), bb.center()
        pa = (int(ca[0]), int(ca[1])) if ba.virtual else _anchor(ba.rect(), *cb)
        pb = (int(cb[0]), int(cb[1])) if bb.virtual else _anchor(bb.rect(), *ca)
        pts = [pa, pb]
        if a == b:
            pts = [pa]
        elif e.get("hidden") and bb.virtual:
            pts = [pb]  # v1's hidden substrate edge of a reaction without enzyme: a point at the reaction
        elif any(_seg_hits(pa, pb, r) for r in all_rects if r not in member_rects[a] and r not in member_rects[b]):
            route = router.route(pa, pb, member_rects[a] + member_rects[b])
            if route:
                pts, routed = route, routed + 1
        path, cp1, cp2 = _path(pts)
        e["path"] = path
        _set_arrow_paths(e, pts[0], cp1, pts[-1], cp2)
    # An edge that neither a visualization member nor the IR ties to anything drawn keeps its v1 shape (PathWhiz's
    # viewer drops it as an orphan; other renderers draw it). Lines move together, unchanged, into a strip under the
    # content and the canvas grows to hold them; a zero-length one becomes a point in the bottom margin.
    orphans = [(e, _path_points(str(e.get("path") or ""))) for eid, e in edges.items() if eid not in edge_ends]
    lines = [(e, pts) for e, pts in orphans if len(set(pts)) > 1]
    for e, pts in orphans:
        if len(set(pts)) <= 1:
            e["path"] = _path([(MARGIN, height - MARGIN // 2)])[0]
    if lines:
        x0 = min(px for _, pts in lines for px, _ in pts)
        y0 = min(py for _, pts in lines for _, py in pts)
        dx, dy = int(MARGIN + BAND_PAD_X - x0), int(height - y0)
        for e, pts in lines:
            moved = iter([int(v) + (dx if i % 2 == 0 else dy) for i, v in enumerate(c for p in pts for c in p)])
            e["path"] = re.sub(r"-?\d+(?:\.\d+)?", lambda _m: str(next(moved)), str(e["path"]))
            new = _path_points(e["path"])
            _set_arrow_paths(e, new[0], new[1], new[-1], new[-2])
            for px, py in new:
                width, height = max(width, int(px) + MARGIN), max(height, int(py) + MARGIN)
    return int(width), int(height)
