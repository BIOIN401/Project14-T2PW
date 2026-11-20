"""PWML layout v2: recompute the geometry of a built PWML (the DEFAULT since D-W23; ``T2PW_PWML_LAYOUT=v1`` keeps the old geometry unchanged).

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
MARGIN, BAND_GAP, BAND_PAD_X, BAND_PAD_BOT = 40, 80, 40, 40
HGAP, VGAP, MEMBER_GAP, MEMBERS_PER_ROW = 160, 60, 10, 3  # PathBank substrate-to-product step: median 667
JUNCTION_W, JUNCTION_H = 40, 20
OBSTACLE_PAD, CELL = 8, 10
ASPECT = 1.5  # target width:height of the packed picture (PathBank content: median 1.4)
MEMBRANE_TEMPLATE = 220  # PathWhiz "Simple Membrane"
# Labels as in PathBank's own files (PW000001: template 235 "Simple Label - Large", scale 1.9). PathWhiz draws it as a
# rounded box at (x + 5s, y + 25s) of 200s x (15 + 16 lines)s, text wrapped at 28 characters (the template's text_length).
LABEL_TEMPLATE, LABEL_SCALE, LABEL_TEXT_LENGTH = 235, 2.0, 28
BOX_PAD_TOP_NO_LABEL, LABEL_GAP = 40, 24
INTERNAL_NAME_PREFIXES = ("__", "AutoState_")
# PathWhiz template sizes of non-compound location templates met in T2PW output (template 2 "Enzyme Monomer I")
OTHER_TEMPLATE_DIMS = {2: (150, 70)}

# ---- compartment drawing conventions mined from curated PathBank diagrams (docs/pwml_output/PATHBANK-CONVENTIONS.md)
# kind of a compartment, from its subcellular location name (else its state name); first match wins
_KIND_RULES = [(k, re.compile(p)) for k, p in (
    ("leaflet", r"leaflet"),
    ("thylakoid", r"thylakoid"),
    ("plastid", r"chloroplast|plastid|etioplast|amyloplast|stroma"),
    ("mito_inner", r"mitochondri\w* inner membrane|inner mitochondrial membrane|intermembrane"),
    ("mito_matrix", r"mitochondri\w* matrix"),
    ("mito", r"mitochondri"),
    ("nucleus", r"\bnucle(us|i|ar|oplasm|olus)\b"),
    ("er", r"endoplasmic reticulum|\ber\b|microsom|sarcoplasmic reticulum"),
    ("golgi", r"golgi"),
    ("lysosome", r"lysosom"),
    ("peroxisome", r"peroxisom|glyoxysom"),
    ("melanosome", r"melanosom"),
    ("vacuole", r"vacuol|tonoplast"),
    ("vesicle", r"vesic|endosom|granule"),
    ("periplasm", r"periplasm"),
    ("outer_membrane", r"outer membrane"),
    ("cell_wall", r"cell wall|peptidoglycan"),
    ("extracellular", r"extracellular|secret|saliva|synaptic cleft|blood|\bplasma\b(?! membrane)|urine|leaf surface|environment"),
    ("membrane", r"membrane|apical|basolateral"),
    ("cytoplasm", r"cytoplasm|cytosol|intracellular|\bcell\b|cellular"),
)]
# PathBank membrane visualization templates (visualization_templates, type MembraneVisualization)
KIND_TEMPLATE = {"mito": 84, "mito_inner": 84, "mito_matrix": 84, "nucleus": 93, "er": 91, "golgi": 146, "lysosome": 92,
                 "peroxisome": 94, "melanosome": 95, "vacuole": 238, "vesicle": 238, "plastid": 237, "thylakoid": 237,
                 "cell_wall": 304, "membrane": 1, "outer_membrane": 1, "periplasm": 1, "cytoplasm": 1}
THIN_TEMPLATES = {MEMBRANE_TEMPLATE}  # template 220 draws a 6-unit line; the others a 50-unit band
BAND_STROKE = 50
# double membranes, drawn as two concentric lines: centre line to centre line (curated median: 70, 74, 94)
DOUBLE_GAP = {"nucleus": 70, "mito": 74, "plastid": 94}
# nesting: depth from the outside in; a compartment sits inside the deepest present compartment its chain allows
_DEPTH = {"cell_wall": 1, "outer_membrane": 1, "periplasm": 2, "membrane": 3, "leaflet": 3.5, "cytoplasm": 4,
          "mito": 5, "nucleus": 5, "er": 5, "golgi": 5, "lysosome": 5, "peroxisome": 5, "melanosome": 5, "vacuole": 5,
          "vesicle": 5, "plastid": 5, "mito_inner": 6, "thylakoid": 6, "mito_matrix": 7}
_CHAIN = {"cell_wall", "outer_membrane", "periplasm", "membrane", "cytoplasm"}
_OWN_CHAIN = {"mito_inner": {"mito"}, "mito_matrix": {"mito", "mito_inner"}, "thylakoid": {"plastid"}, "leaflet": {"membrane"}}


def compartment_kind(location_name: str, state_name: str = "") -> str:
    """Kind of a compartment (``other`` when nothing matches), from its location name, else its state name."""
    for text in (location_name, state_name):
        t = str(text or "").casefold().replace("_", " ")
        for kind, rx in _KIND_RULES:
            if rx.search(t):
                return kind
    return "other"


def compartment_parents(kinds: Dict[Any, str]) -> Dict[Any, Any]:
    """Parent compartment of each compartment (``None`` at the top): the deepest present compartment with a smaller
    depth on its chain (the cell envelope, then its own organelle). When two or more compartments share that depth
    (two tissues' cytosols, apical and basolateral membranes) the parent is ambiguous and none is claimed."""
    parent: Dict[Any, Any] = {}
    for s, k in kinds.items():
        if k not in _DEPTH:
            parent[s] = None
            continue
        allowed = _CHAIN | _OWN_CHAIN.get(k, set())
        cands = [o for o, ko in kinds.items() if o != s and ko in allowed and _DEPTH[ko] < _DEPTH[k]]
        if k == "leaflet":  # a leaflet lies inside the membrane it belongs to, never inside a compartment
            cands = [o for o in cands if kinds[o] == "membrane"]
        top = max((_DEPTH[kinds[o]] for o in cands), default=None)
        best = [o for o in cands if _DEPTH[kinds[o]] == top]
        parent[s] = best[0] if len(best) == 1 else None
    return parent


def _label_chunks(text: str, length: int = LABEL_TEXT_LENGTH) -> List[str]:
    """Lines of PathWhiz's divide_label_text (visualization_template.js.coffee): words, long ones split, greedy join."""
    words = []
    for w in text.split():
        words += [w[i:i + length] for i in range(0, len(w), length)] if len(w) > length else [w]
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > length:
            lines, cur = lines + [cur], w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines + ([cur] if cur else [])


def _label_lines(text: str, length: int = LABEL_TEXT_LENGTH) -> int:
    return len(_label_chunks(text, length))


# Verdana advance widths per 1000 em (the label font); other characters count 650. The template wraps at 28 characters
# but the 200-unit label rectangle holds about 20 wide ones, so PathWhiz draws a line like "Bacillus subtilis cytoplasm"
# running out of its box.
_VERDANA = {c: w for w, cs in (
    (269, "'"), (274, "il"), (344, "j"), (352, " f"), (364, ",."), (394, "!t"), (421, "I"), (427, "r"), (454, "()-/:;[\\]|"),
    (455, "J"), (459, '"'), (521, "cs"), (525, "z"), (545, "?"), (557, "L"), (575, "F"), (592, "kvxy"), (596, "e"), (601, "a"),
    (603, "P"), (607, "o"), (615, "Y"), (616, "T"), (623, "bdgpq"), (632, "E"), (633, "hnu"), (635, "{}"),
    (636, "$*0123456789_`"), (684, "ASV"), (685, "XZ"), (686, "B"), (693, "K"), (695, "R"), (698, "C"), (727, "&"),
    (732, "U"), (748, "N"), (751, "H"), (771, "D"), (775, "G"), (787, "OQ"), (818, "#+<=>^w~"), (843, "M"), (973, "m"),
    (989, "W"), (1000, "@"), (1076, "%")) for c in cs}
LABEL_FONT, LABEL_FIT = 16, 180  # font size of the template; widest line that sits inside its 200-unit rectangle with a margin
LABEL_STEP = (15 + LABEL_FONT) * 2 + 6  # stacked one-line labels: rectangle height in canvas units (scale 2) plus a gap


def _text_w(text: str) -> float:
    return sum(_VERDANA.get(c, 650) for c in text) * LABEL_FONT / 1000


def label_parts(text: str) -> List[str]:
    """The label texts to draw for ``text``: the text itself when PathWhiz's wrapping keeps every line inside the label
    rectangle, else one-line pieces (each its own label, stacked) cut by drawn width. The wrap length is a template
    field, so a line cannot be shortened from here."""
    if all(_text_w(c) <= LABEL_FIT for c in _label_chunks(text)):
        return [text]
    parts, cur = [], ""
    for w in text.split():
        while _text_w(w) > LABEL_FIT or len(w) > LABEL_TEXT_LENGTH:  # a word wider than a line: cut it
            k = max(1, max(i for i in range(1, len(w) + 1) if _text_w(w[:i]) <= LABEL_FIT and i <= LABEL_TEXT_LENGTH))
            if cur:
                parts, cur = parts + [cur], ""
            parts, w = parts + [w[:k]], w[k:]
        if cur and _text_w(f"{cur} {w}") > LABEL_FIT:
            parts, cur = parts + [cur], w
        else:
            cur = f"{cur} {w}" if cur else w
    return parts + ([cur] if cur else [])


def _label_pad(parts: List[str]) -> int:
    """Height from the top of a compartment box to the bottom of its label."""
    if len(parts) == 1:
        return int(4 + (40 + 16 * _label_lines(parts[0])) * LABEL_SCALE)
    return int(4 + (len(parts) - 1) * LABEL_STEP + (40 + 16) * LABEL_SCALE)


def _label_rect(lb: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """The white box PathWhiz draws for a template-235 label (simple_label.jst.eco), in canvas coordinates."""
    x, y, sc = int(lb["x"]), int(lb["y"]), float(lb["scale-x"])
    return (int(x + 5 * sc), int(y + 25 * sc), int(x + 205 * sc), int(y + (40 + 16 * _label_lines(str(lb["label-text"]))) * sc))


def label_text(state: Dict[str, Any]) -> str:
    """The state's name without the trailing " state"; nothing for a generated name (``__auto_state__``, ``AutoState_*``)."""
    name = str(state.get("name") or "").strip()
    if name.endswith(" state"):
        name = name[: -len(" state")].strip()
    return "" if not name or name.startswith(INTERNAL_NAME_PREFIXES) else name


def layout_version() -> str:
    """``v1`` only when ``T2PW_PWML_LAYOUT`` is ``v1`` (stripped, any case); unset, empty, ``v2`` and every other value, including ``0``, ``off`` and ``false``, give the default ``v2`` (D-W23 b)."""
    return "v1" if os.environ.get("T2PW_PWML_LAYOUT", "").strip().casefold() == "v1" else "v2"


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


def _layered(blocks: Dict[Any, "_Block"], keys: List[Any], flow: List[Tuple[Any, Any]],
             rank: Optional[Dict[Any, int]] = None) -> Tuple[int, int]:
    """Sugiyama-style layout of one connected group: columns by longest path (back edges dropped), order by
    barycenter sweeps, y by least-squares alignment with neighbours (no overlap). Sets block x/y relative to
    the group's top-left corner; returns the group's (width, height). ``rank`` (the index of the first flow pair
    a block is in; flow follows the pathway's order of processes) orders the depth-first search, so a cycle is broken at the edge that returns to the earlier
    process and the chain reads left to right in pathway order."""
    if rank:
        keys = sorted(keys, key=lambda k: rank.get(k, 1 << 30))  # stable: unranked keep their order, last
    inside = set(keys)
    succ: Dict[Any, List[Any]] = defaultdict(list)
    for u, v in flow:
        if u in inside and v in inside and u != v and v not in succ[u]:
            succ[u].append(v)
    if rank:
        for u in succ:
            succ[u].sort(key=lambda k: rank.get(k, 1 << 30))
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
    pairs = [(u, v) for u in keys for v in dag[u]]

    def crossings():
        n = 0
        for i, (a, b) in enumerate(pairs):
            for c, d in pairs[i + 1:]:
                if layer[a] == layer[c] and layer[b] == layer[d] and (pos[a] - pos[c]) * (pos[b] - pos[d]) < 0:
                    n += 1
        return n

    best = (crossings(), [list(c) for c in cells])
    for sweep in range(24):  # barycenter sweeps; the order with the fewest crossings wins
        nb = pred if sweep % 2 == 0 else dag
        for cell in cells:
            bary = {k: (sum(pos[n] for n in nb[k]) / len(nb[k]) if nb[k] else pos[k]) for k in cell}
            cell.sort(key=lambda k: bary[k])
        renumber()
        if best[0] and len(pairs) <= 400:
            n = crossings()
            if n < best[0]:
                best = (n, [list(c) for c in cells])
    if len(pairs) <= 400:
        cells[:] = best[1]
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
    n_flow_before_transports = len(flow)
    transports: List[Tuple[Any, List[Tuple[Any, Any]]]] = []  # (hub, [(cargo block, side)])
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
        transports.append((hub, [(k, m.get("side")) for m, k in members if k is not None]))
        for m, k in members:
            left = m.get("side") == "Left"
            if m.get("edge-id") is not None and int(m["edge-id"]) in edges:
                kk = k if k is not None else hub
                edge_ends[int(m["edge-id"])] = (kk, hub) if left else (hub, kk)
            if k is not None:
                flow.append((k, hub) if left else (hub, k))
    n_flow_after_transports = len(flow)
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

    # rank: index of the first flow pair a block is in (flow lists reactions, transports, interactions in pathway order)
    rank: Dict[Any, int] = {}
    for i, (u, v) in enumerate(flow):
        rank.setdefault(u, i)
        rank.setdefault(v, i)
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
            w, h = _layered(blocks, comp, flow, rank)
            comps.append((comp, w, h))
        comps.sort(key=lambda c: (not any(k in attached for k in c[0]), -(c[1] * c[2])))  # drawn groups first
        comps_by_state[s] = comps
    if loose_key in by_state:
        comps_by_state[loose_key] = sorted(((([k], *_layered(blocks, [k], flow))) for k in by_state[loose_key]),
                                           key=lambda c: -(c[1] * c[2]))

    area = sum((w + HGAP) * (h + VGAP) for cs in comps_by_state.values() for _, w, h in cs)
    target = max([int((area * ASPECT) ** 0.5)] + [w for cs in comps_by_state.values() for _, w, _ in cs])
    # compartment kinds (PathBank membrane template and colour per kind) and nesting (organelles inside the cell,
    # the cell inside its envelope); each box holds its own groups (shelf-packed), then its child boxes in rows
    states = {s.get("id"): s for s in items.get("biological-states", [])}
    loc_names = {l.get("id"): l.get("name") for l in items.get("subcellular-locations", [])}
    labels = {st: label_text(states.get(st, {})) for st in bands}
    label_w = int((5 + 200) * LABEL_SCALE)
    kinds = {st: compartment_kind(loc_names.get(states.get(st, {}).get("subcellular-location-id")) or "",
                                  str(states.get(st, {}).get("name") or "")) for st in bands}
    parent = compartment_parents(kinds)
    children: Dict[Any, List[Any]] = defaultdict(list)
    for st in bands:
        children[parent[st]].append(st)

    def style(st):
        """(membrane template or None for a label only, gap of a double line or 0) of a compartment box."""
        if st == loose_key:
            return None, 0
        k, pk = kinds[st], kinds.get(parent.get(st))
        if k == "extracellular":
            return None, 0  # PathBank draws no outline around the outside of a cell
        if (k == "cytoplasm" and pk == "membrane") or (k == "periplasm" and pk == "outer_membrane") \
                or (k in ("mito_inner", "mito_matrix") and pk in ("mito", "mito_inner")) or (k == "thylakoid" and pk == "plastid"):
            return None, 0  # the enclosing box already draws this compartment's membrane
        if k == "mito_matrix":
            return KIND_TEMPLATE[k], DOUBLE_GAP["mito"]  # the matrix alone: inside the mitochondrion's two membranes
        return KIND_TEMPLATE.get(k, MEMBRANE_TEMPLATE), DOUBLE_GAP.get(k, 0)

    def inset(st):
        tpl, gap = style(st)
        return (0 if tpl is None else 3 if tpl in THIN_TEMPLATES else BAND_STROKE // 2) + gap

    def pad_top(st):
        text = labels.get(st, "")
        return inset(st) + (_label_pad(label_parts(text)) + LABEL_GAP if text else BOX_PAD_TOP_NO_LABEL)

    def pad_x(st):
        return BAND_PAD_X + inset(st)

    def pad_bot(st):
        return BAND_PAD_BOT + inset(st)

    def pack_rows(sizes, limit):
        """Row packing of (w, h) boxes: offsets, total width, total height."""
        offs, x, y, row_h, used = [], 0, 0, 0, 0
        for w, h in sizes:
            if x > 0 and x + w > limit:
                x, y, row_h = 0, y + row_h + BAND_GAP, 0
            offs.append((x, y))
            used, row_h, x = max(used, x + w), max(row_h, h), x + w + BAND_GAP
        return offs, used, (y + row_h) if sizes else 0

    sizes: Dict[Any, Tuple[int, int]] = {}
    layouts: Dict[Any, Any] = {}

    def measure(st):
        placed, cx, cy, shelf_h, own_w = [], 0, 0, 0, 0
        for comp, w, h in comps_by_state.get(st, []):
            if cx > 0 and cx + w > target:
                cx, cy, shelf_h = 0, cy + shelf_h + 2 * VGAP, 0
            placed.append((comp, cx, cy))
            own_w, shelf_h, cx = max(own_w, cx + w), max(shelf_h, h), cx + w + HGAP
        own_h = cy + shelf_h
        kids = children.get(st, []) if st != loose_key else []
        for c in kids:
            measure(c)
        offs, kw, kh = pack_rows([sizes[c] for c in kids], max(target, own_w))
        inner_w = max(own_w, kw, label_w if labels.get(st) else 0)
        inner_h = own_h + (2 * VGAP if placed and kids else 0) + kh
        kid_top = own_h + (2 * VGAP if placed and kids else 0)
        layouts[st] = (placed, [(c, dx, kid_top + dy) for c, (dx, dy) in zip(kids, offs)])
        sizes[st] = (inner_w + 2 * pad_x(st), pad_top(st) + inner_h + pad_bot(st))

    band_rects: List[Tuple[Any, Tuple[int, int, int, int]]] = []

    def place(st, x, y, w, h):
        placed, kids = layouts[st]
        for comp, dx, dy in placed:
            for k in comp:
                blocks[k].x += x + pad_x(st) + dx
                blocks[k].y += y + pad_top(st) + dy
        if st != loose_key:
            band_rects.append((st, (x, y, x + w, y + h)))
        # kids in one row share the row's height
        row_h: Dict[int, int] = defaultdict(int)
        for c, dx, dy in kids:
            row_h[dy] = max(row_h[dy], sizes[c][1])
        for c, dx, dy in kids:
            place(c, x + pad_x(st) + dx, y + pad_top(st) + dy, sizes[c][0], row_h[dy])

    roots = list(children.get(None, [])) + ([loose_key] if loose_key in comps_by_state else [])
    for st in roots:
        measure(st)
    row_limit = max([target + 2 * BAND_PAD_X] + [sizes[st][0] for st in roots])
    rows: List[List[Any]] = []
    x = row_w = 0
    for st in roots:
        if not rows or (x > 0 and x + sizes[st][0] > row_limit):
            rows.append([])
            x = 0
        rows[-1].append((x, st))
        x += sizes[st][0] + BAND_GAP
    top = MARGIN
    for row in rows:
        row_h = max(sizes[st][1] for _, st in row)
        for bx, st in row:
            place(st, MARGIN + bx, top, sizes[st][0], row_h)
            row_w = max(row_w, bx + sizes[st][0])
        top += row_h + BAND_GAP
    width = MARGIN + row_w + MARGIN

    # ---- compartments: bound-visualizations (when written), membranes and labels
    rect_of_state = dict(band_rects)
    extra_x, extra_y = MARGIN, top
    for bv in items.get("bound-visualizations", []):
        r = rect_of_state.get(bv.get("biological-state-id"))
        if r is None:  # a state with nothing drawn in it: small box in a row under the bands
            r = (extra_x, extra_y, extra_x + 200, extra_y + 80)
            extra_x += 240
        bv["x"], bv["y"], bv["width"], bv["height"] = r[0], r[1], str(r[2] - r[0]), str(r[3] - r[1])
    height = (extra_y + 80 + MARGIN) if extra_x > MARGIN else ((top - BAND_GAP if rows else top) + MARGIN)
    width = max(width, extra_x + MARGIN)
    for s, (x0, y0, x1, y1) in band_rects:
        tpl, gap = style(s)
        lines = [] if tpl is None else [(x0, y0, x1, y1)] + ([(x0 + gap, y0 + gap, x1 - gap, y1 - gap)] if gap else [])
        for (a0, b0, a1, b1) in lines:
            items.setdefault("membrane-visualizations", []).append({
                "id": new_id(), "path": _rounded_rect(a0, b0, a1, b1), "visualization-template-id": tpl,
                "complete-membrane": True, "zindex": 6, "width": f"{a1 - a0}.0", "height": f"{b1 - b0}.0"})
        d = inset(s)
        for j, part in enumerate(label_parts(labels[s]) if labels.get(s) else []):
            items.setdefault("label-locations", []).append({
                "id": new_id(), "visualization-template-id": LABEL_TEMPLATE, "label-text": part,
                "x": x0 + d + 12, "y": y0 + d + 4 + j * LABEL_STEP, "zindex": 20, "scale-x": f"{LABEL_SCALE}", "scale-y": f"{LABEL_SCALE}",
                "width": "200", "height": "15"})

    # ---- transports across their membrane: a transporter (or a transport point) that serves transports only sits on
    # the membrane line between its cargo compartments, on the side facing the outer cargo (PathBank's convention)
    other_use = {k for pair in flow[:n_flow_before_transports] + flow[n_flow_after_transports:] for k in pair}
    anc: Dict[Any, List[Any]] = {}
    for st in bands:
        chain, cur = [], st
        while cur is not None:
            chain.append(cur)
            cur = parent.get(cur)
        anc[st] = chain
    fixed = [(bl.x, bl.y, bl.x + bl.w, bl.y + bl.h) for k, bl in blocks.items() if not bl.virtual]
    fixed += [_label_rect(lb) for lb in items.get("label-locations", [])]
    def crossing(si, so):
        """The membrane a cargo in ``si`` crosses toward ``so``: the highest outlined box around ``si`` not holding ``so``."""
        m = next((s for s in reversed(anc[si]) if s not in anc[so]), None)
        return m if m in rect_of_state and style(m)[0] is not None else None

    plans: Dict[Any, List[Any]] = defaultdict(list)  # hub -> [(membrane, top side, inner cargo block)]
    for hub, cargo in transports:
        sts = [(blocks[k].state, blocks[k], side) for k, side in cargo if blocks[k].state in anc]
        lefts = [c for c in sts if c[2] == "Left"]
        rights = [c for c in sts if c[2] != "Left"]
        if not lefts or not rights or lefts[0][0] == rights[0][0]:
            plans[hub].append(None)
            continue
        (sa, ba, _), (sb, bb, _) = lefts[0], rights[0]
        opts = [(crossing(sa, sb), ba, bb), (crossing(sb, sa), bb, ba)]
        mem, inner, outer = next(((m, i, o) for m, i, o in opts if m is not None), (None, None, None))
        if mem is None:
            plans[hub].append(None)
            continue
        r = rect_of_state[mem]
        plans[hub].append((mem, outer.center()[1] < (r[1] + r[3]) / 2, inner))
    for hub, plan in plans.items():
        # a transporter shared by transports across different membranes stays where its group put it
        if hub in other_use or None in plan or len({(m, t) for m, t, _ in plan}) != 1:
            continue
        mem, on_top, ba = plan[0]
        x0, y0, x1, y1 = rect_of_state[mem]
        gap = style(mem)[1]
        hb = blocks[hub]
        line_y = y0 + gap // 2 if on_top else y1 - gap // 2  # a double membrane: between its two lines
        ny = int(line_y - hb.h / 2)
        want = int(ba.center()[0] - hb.w / 2)
        lo, hi = x0 + inset(mem) + 20, x1 - inset(mem) - 20 - hb.w
        if hi < lo:
            continue
        own = (hb.x, hb.y, hb.x + hb.w, hb.y + hb.h)
        for step in range(0, x1 - x0, 20):
            hit = None
            for nx in (want + step, want - step):
                nx = max(lo, min(hi, nx))
                r = (nx - 15, ny - 15, nx + hb.w + 15, ny + hb.h + 15)
                if not any(not (r[2] <= f[0] or f[2] <= r[0] or r[3] <= f[1] or f[3] <= r[1]) for f in fixed if f != own):
                    hit = nx
                    break
            if hit is not None:
                if not hb.virtual:
                    fixed.remove(own)
                hb.x, hb.y = hit, ny
                fixed.append((hb.x, hb.y, hb.x + hb.w, hb.y + hb.h))
                break
    for bl in blocks.values():
        for loc, (dx, dy) in zip(bl.members, bl.offsets):
            loc["x"], loc["y"] = bl.x + dx, bl.y + dy
    for loc in hidden_locs:  # not drawn; kept inside the canvas
        loc["x"] = max(0, min(int(loc.get("x") or 0), width - _dim(loc["width"])))
        loc["y"] = max(0, min(int(loc.get("y") or 0), height - _dim(loc["height"])))

    # ---- edges
    member_rects = {k: [(int(l["x"]), int(l["y"]), int(l["x"]) + _dim(l["width"]), int(l["y"]) + _dim(l["height"]))
                        for l in bl.members] for k, bl in blocks.items()}
    label_rects = [_label_rect(lb) for lb in items.get("label-locations", [])]
    all_rects = [r for rs in member_rects.values() for r in rs] + label_rects
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
    lines = [(e, pts) for e, pts in orphans if len(set(pts)) > 1 and re.fullmatch(r"[MLC0-9.\s,-]*", str(e.get("path") or ""))
             and len(re.findall(r"-?\d+(?:\.\d+)?", str(e.get("path") or ""))) % 2 == 0]
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
