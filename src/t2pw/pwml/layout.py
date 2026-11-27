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
shelf-packed in the box, and boxes are packed in rows. Edges are PathBank's cubic curves (VIS-1): they leave a node
at right angles from the middle of a side and meet the enzyme at its substrate or product side; currency
co-substrates and co-products are small side nodes beside the reaction's shaft; an edge whose curve would cross a
node box is routed around them on a grid.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from t2pw.pwml import cell_art
from t2pw.pwml.compound_templates import COMPOUND_ID_TO_TEMPLATE, TEMPLATE_DIMS

LOC_SECTIONS = ("compound-locations", "protein-locations", "nucleic-acid-locations", "element-collection-locations")
MARGIN, BAND_GAP, BAND_PAD_X, BAND_PAD_BOT = 40, 80, 40, 40
HGAP, VGAP, MEMBER_GAP, MEMBERS_PER_ROW = 160, 60, 10, 3  # PathBank substrate-to-product step: median 667
JUNCTION_W, JUNCTION_H = 40, 20
STACK_DX, STACK_DY = 10, 38  # VIS-5: offset of each further enzyme in a reaction's stack (PathBank PW000779: ~5, ~22)
OBSTACLE_PAD, CELL = 8, 10
SNAKE_MIN, SNAKE_OVER = 1800, 1.5  # VIS-5: a group folds into rows when wider than SNAKE_OVER x the picture's square-ish width (VIS-8: was 1.8)
ASPECT = 1.5  # target width:height of the packed picture (PathBank content: median 1.4)
MEMBRANE_TEMPLATE = 220  # PathWhiz "Simple Membrane"
# Labels as in PathBank's own files (PW000001: template 235 "Simple Label - Large", scale 1.9). PathWhiz draws it as a
# rounded box at (x + 5s, y + 25s) of 200s x (15 + 16 lines)s, text wrapped at 28 characters (the template's text_length).
LABEL_TEMPLATE, LABEL_SCALE, LABEL_TEXT_LENGTH = 235, 2.0, 28
BOX_PAD_TOP_NO_LABEL, LABEL_GAP = 40, 24
LINE_LABEL_ROOM, LINE_LABEL_X = 45, 80  # VIS-5: label on a membrane line: room above a box, offset from its corner
LOOSE_GAP = 40  # VIS-5: gap between unattached nodes in a box
OUTSIDE_GAP = 110  # VIS-5: outside cargo row distance from the cell's membrane
INTERNAL_NAME_PREFIXES = ("__", "AutoState_")
# PathWhiz template sizes of non-compound location templates met in T2PW output (template 2 "Enzyme Monomer I")
OTHER_TEMPLATE_DIMS = {2: (150, 70), 12: (150, 70)}
# VIS-5 (STRUCTURE_DIFF G6): the writer gives nucleic-acid locations template 2 ("Enzyme Monomer I", a ProteinLocation
# template); PathBank uses NucleicAcidLocation templates. 12 is PathWhiz's "Nucleic Acid" (NucleicAcidLocation, 150x70).
PROTEIN_TEMPLATE_ON_NUCLEIC, NUCLEIC_ACID_TEMPLATE = 2, 12

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


def _side_words(text: str) -> set:
    """VIS-5: which side of a cell envelope a compartment name names: inner / outer / plasma / cytoplasmic."""
    words = set(re.findall(r"[a-z]+", text.casefold().replace("_", " ")))
    out = {w for w in ("inner", "outer", "plasma") if w in words}
    out |= {"inner"} if "im" in words else set()
    out |= {"outer"} if "om" in words else set()
    if "cytoplasmic" in words and "membrane" in words and "leaflet" not in words:
        out.add("cytoplasmic")
    return out


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
        self.dir = 1  # VIS-5: -1 in a right-to-left row of a snaked chain (substrates enter from the right)
        self.layer, self.col = 0, (0, 0)  # column index and (x, width) of its column in the layered group

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


EDGE_PATH_MAX = 255  # VIS-5: PathWhiz's edges.path is varchar(255); a longer path fails the edge's import
ROUTE_PATH_MAX = 220  # a routed path's budget before the cell art may shift the picture (a digit more per number)
CORNER_R = 24  # VIS-5: radius of a routed edge's rounded corner


def _round_corners(points: List[Tuple[int, int]], r: int = CORNER_R) -> List[Tuple[int, int]]:
    """VIS-5: a right-angle route with each corner cut by a short diagonal of up to ``r`` on either side (PathBank's
    connectors bend; a grid route's hard corners read as loops). Ends unchanged."""
    pts = [points[0]] + [q for p, q in zip(points, points[1:]) if q != p]
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for p, q, n in zip(pts, pts[1:], pts[2:]):
        d1 = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
        d2 = ((n[0] - q[0]) ** 2 + (n[1] - q[1]) ** 2) ** 0.5
        k = min(r, d1 / 2, d2 / 2)
        if k < 4:
            out.append(q)
            continue
        out.append((int(round(q[0] - (q[0] - p[0]) * k / d1)), int(round(q[1] - (q[1] - p[1]) * k / d1))))
        out.append((int(round(q[0] + (n[0] - q[0]) * k / d2)), int(round(q[1] + (n[1] - q[1]) * k / d2))))
    return out + [pts[-1]]


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


# ---- edge drawing conventions mined from the 16 curated PathBank reference PWMLs (VIS-1, docs/webserver/results/VIS-1.md):
# every edge is one cubic "M p0 C c1 c2 p3"; it leaves a node from the middle of a side at right angles (83 % / 93 %)
# and meets the enzyme at the middle of its side along the reaction's shaft (85 % / 88 %); all substrates of a reaction
# share one point, all products another (80 % / 85 %); control points sit about a quarter of the chord out (median 30
# for a 140 chord). Co-substrates and co-products (currency compounds) are small nodes 75 along the shaft from the
# reaction point and 73 off it (medians), on short branches (chord median 106-113).
SIDES = {"L": (-1, 0), "R": (1, 0), "T": (0, -1), "B": (0, 1)}
PILL_ALONG, PILL_PERP, PILL_GAP, PILL_MAX = 75, 73, 20, 100  # PILL_MAX: larger nodes stay on the main chain
TRANSPORTER_REACH = 300  # how far a transporter slides along its membrane for a clear line to its inner cargo
SIDE_TEMPLATES = frozenset(COMPOUND_ID_TO_TEMPLATE.values())  # PathBank's small-molecule icons (ATP, NAD+, H2O, ...)
# Compound box templates (both 200 x 190): PathBank draws 157 ("Drug", pink header) only for the drug a drug pathway is
# about (15 of 271 boxes in the 16 reference PWMLs: fluconazole, caffeine, zidovudine, buprenorphine, benzethonium, each
# in its own pathway) and 81 for every other compound, DrugBank id or not (144 of the 256 81-boxes have one: L-alanine,
# pyruvic acid, ...). compound_templates.select_compound_template_id gives 157 to any compound with a DrugBank id, so v2
# draws those as 81. ponytail: the drug of a drug-action pathway loses its pink header too; keep 157 for the compound
# named in the pathway once the pathway name reaches relayout.
DRUG_TEMPLATE, COMPOUND_TEMPLATE = 157, 81
# Edge templates: PathBank draws a transport's edges with template 83 (151 of 152 transport member edges in the 16
# reference PWMLs; 690 of 691 reaction member edges use 5). The writer gives transport edges 5.
TRANSPORT_EDGE_TEMPLATE = 83


def _side_point(r: Tuple[int, int, int, int], s: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Middle of side ``s`` of rectangle ``r`` and that side's outward normal."""
    x0, y0, x1, y1 = r
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return {"L": (x0, cy), "R": (x1, cy), "T": (cx, y0), "B": (cx, y1)}[s], SIDES[s]


def _sides_toward(r: Tuple[int, int, int, int], tx: float, ty: float) -> List[str]:
    """The four sides of ``r``, the one facing (tx, ty) first (distances scaled by the box's half sizes)."""
    x0, y0, x1, y1 = r
    dx = (tx - (x0 + x1) / 2) / max(1.0, (x1 - x0) / 2)
    dy = (ty - (y0 + y1) / 2) / max(1.0, (y1 - y0) / 2)
    return sorted(SIDES, key=lambda s: -(SIDES[s][0] * dx + SIDES[s][1] * dy))


def _facing_sides(r: Tuple[int, int, int, int], tx: float, ty: float) -> List[str]:
    """The sides of ``r`` that face (tx, ty) at all, best first (always at least one): leaving from a side that faces away
    makes a loop."""
    x0, y0, x1, y1 = r
    dx, dy = tx - (x0 + x1) / 2, ty - (y0 + y1) / 2
    sides = _sides_toward(r, tx, ty)
    return sides[:1] + [s for s in sides[1:3] if SIDES[s][0] * dx + SIDES[s][1] * dy > 0]


def _curve(p0, n0, p3, n3, ratio: float = 0.25, bound: Optional[Tuple[int, int]] = None, short: Tuple[bool, bool] = (False, False)):
    """One cubic from p0 (leaving along normal n0) to p3 (arriving against normal n3): path, c1, c2. The control points
    sit ``ratio`` of the chord out (at least 30, or a third of a short chord; at most 200), kept inside ``bound``; an end
    marked in ``short`` (a reaction point, where several edges meet) keeps PathBank's 30, so the edges meeting there
    turn into it late instead of running together."""
    chord = ((p3[0] - p0[0]) ** 2 + (p3[1] - p0[1]) ** 2) ** 0.5
    k = max(1, int(max(min(30.0, chord / 3), min(200.0, ratio * chord))))
    k0, k3 = (max(1, int(min(30.0, chord / 3))) if s else k for s in short)
    c1 = [p0[0] + k0 * n0[0], p0[1] + k0 * n0[1]]
    c2 = [p3[0] + k3 * n3[0], p3[1] + k3 * n3[1]]
    if bound:
        for c in (c1, c2):
            c[0], c[1] = max(0, min(bound[0], c[0])), max(0, min(bound[1], c[1]))
    c1, c2 = tuple(c1), tuple(c2)
    return f"M{p0[0]} {p0[1]} C{c1[0]} {c1[1]} {c2[0]} {c2[1]} {p3[0]} {p3[1]}", c1, c2


BEND_OFFSETS = (90, -90, 180, -180, 300, -300, 450, -450)  # VIS-5: how far a bent edge's middle leaves the chord
VIA_PAD = 60  # VIS-5: a bent edge passes this far off the side of the box it goes round


CROSS_COST = 300  # VIS-8: length an interaction edge may add to avoid crossing one other edge
ARC_HEIGHTS = (60, 110, 170, 250, 350, 480)  # VIS-7: how far an arc's control points stand off the boxes' top or bottom


def _plen(pts) -> float:
    return sum(((q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2) ** 0.5 for o, q in zip(pts, pts[1:]))


def _arc(p0, n0, p3, n3, h: float, bound: Optional[Tuple[int, int]] = None):
    """VIS-7: one cubic leaving p0 along n0 and arriving at p3 against n3, both control points ``h`` out along their
    normals (a flat arch over or under a row of boxes): path, c1, c2."""
    c1 = [p0[0] + h * n0[0], p0[1] + h * n0[1]]
    c2 = [p3[0] + h * n3[0], p3[1] + h * n3[1]]
    if bound:
        for c in (c1, c2):
            c[0], c[1] = max(0, min(bound[0], c[0])), max(0, min(bound[1], c[1]))
    c1, c2 = (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1]))
    return f"M{p0[0]} {p0[1]} C{c1[0]} {c1[1]} {c2[0]} {c2[1]} {p3[0]} {p3[1]}", c1, c2


def _bend(p0, n0, p3, n3, off: float, bound: Optional[Tuple[int, int]] = None, via: Optional[Tuple[int, int]] = None):
    """VIS-5: an edge as two cubics through a point ``off`` units off the middle of the chord (left of p0->p3 for
    positive), smooth there (the tangent runs along the chord): path, first point, first control point, last point,
    last control point, sampled points; ``None`` for a zero chord."""
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    chord = (dx * dx + dy * dy) ** 0.5
    if chord < 1:
        return None
    ux, uy = dx / chord, dy / chord
    w = via or (int(round((p0[0] + p3[0]) / 2 - uy * off)), int(round((p0[1] + p3[1]) / 2 + ux * off)))
    k = max(30.0, min(200.0, chord / 4))
    h = max(20.0, chord / 5)
    c1 = (int(p0[0] + k * n0[0]), int(p0[1] + k * n0[1]))
    c4 = (int(p3[0] + k * n3[0]), int(p3[1] + k * n3[1]))
    c2 = (int(w[0] - h * ux), int(w[1] - h * uy))
    c3 = (int(w[0] + h * ux), int(w[1] + h * uy))
    if bound:
        c1, c2, c3, c4, w = [(max(0, min(bound[0], c[0])), max(0, min(bound[1], c[1]))) for c in (c1, c2, c3, c4, w)]
    path = f"M{p0[0]} {p0[1]} C{c1[0]} {c1[1]} {c2[0]} {c2[1]} {w[0]} {w[1]} C{c3[0]} {c3[1]} {c4[0]} {c4[1]} {p3[0]} {p3[1]}"
    pts = _bezier_points(p0, c1, c2, w) + _bezier_points(w, c3, c4, p3)[1:]
    return path, p0, c1, p3, c4, pts


def _bezier_points(p0, c1, c2, p3, n: int = 16) -> List[Tuple[float, float]]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p3[0],
                    u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p3[1]))
    return out


def _clear(pts: List[Tuple[float, float]], rects: List[Tuple[int, int, int, int]]) -> bool:
    return not any(_seg_hits(p, q, r) for p, q in zip(pts, pts[1:]) for r in rects)


def _crossings(pts: List[Tuple[float, float]], others: List[List[Tuple[float, float]]]) -> int:
    """How many of ``others`` (polylines) cross ``pts`` (a proper crossing; touching at a shared end does not count)."""
    def o(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 1e-9) - (v < -1e-9)

    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    n = 0
    for b in others:
        if not b or max(p[0] for p in b) < x0 or min(p[0] for p in b) > x1 or max(p[1] for p in b) < y0 or min(p[1] for p in b) > y1:
            continue
        n += any(o(p, q, r) * o(p, q, s) < 0 and o(r, s, p) * o(r, s, q) < 0
                 for p, q in zip(pts, pts[1:]) for r, s in zip(b, b[1:]))
    return n


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
    for u in reversed(order):  # VIS-5: a node sits just before its first successor (a side chain joins where it is
        # used instead of running a long edge from the left edge of the group; was: pure sources only)
        if dag[u]:
            layer[u] = max(layer[u], min(layer[v] for v in dag[u]) - 1)
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
        bl.layer, bl.col, bl.dir = l, (colx[l], colw[l]), 1
    for k in order:  # a reaction point with no enzyme sits on its substrates' side (the writer hides the substrate edge,
        # so the visible product edge starts at the substrate; VIS-5: at the substrates' centre height, not wherever the
        # alignment put the point, where the edge began in empty space)
        ins = [blocks[p] for p in pred[k] if layer[p] == layer[k] - 1]
        if blocks[k].virtual and ins:
            blocks[k].x = max(p.x + p.w for p in ins) - JUNCTION_W // 2 + 2
            cy = sum(p.y + p.h / 2 for p in ins) / len(ins)
            if any(layer[o] == layer[k] - 1 and blocks[o] not in ins and blocks[o].y <= cy <= blocks[o].y + blocks[o].h
                   for o in keys):  # the substrates' middle is another node: start at the first substrate instead
                cy = ins[0].y + ins[0].h / 2
            blocks[k].y = int(round(cy - blocks[k].h / 2))
    return x - HGAP, int(round(max(blocks[k].y + blocks[k].h for k in keys)))


ROW_GAP = 160  # VIS-5: vertical gap between the rows of a snaked chain


def _snake(blocks: Dict[Any, "_Block"], keys: List[Any], limit: int, hubs: set,
           free_out: Optional[Dict[int, Tuple[int, int, int, int]]] = None) -> Tuple[int, int]:
    """VIS-5: fold a layered group wider than ``limit`` into rows read alternately left to right and right to left
    (PathBank's long chains turn back instead of running off the page). Rows end after a node column rather than a
    process column where possible, so the turn is a short vertical step from a node to its next process. Sets block
    x/y (relative to the group) and ``dir``; returns the group's (width, height)."""
    cols: Dict[int, Tuple[int, int]] = {}
    for k in keys:
        cols[blocks[k].layer] = blocks[k].col
    order = sorted(cols)
    rows: List[List[int]] = [[]]
    start = 0
    for l in order:
        cx, cw = cols[l]
        if rows[-1] and cx + cw - start > limit:
            last = rows[-1][-1]
            carry = [last] if len(rows[-1]) > 2 and all(k in hubs for k in keys if blocks[k].layer == last) else []
            if carry:
                rows[-1].pop()
            rows.append(carry)
            start = cols[carry[0]][0] if carry else cx
        if not rows[-1]:
            start = cx
        rows[-1].append(l)
    if len(rows) < 2:
        return max(cx + cw for cx, cw in cols.values()), max(blocks[k].y + blocks[k].h for k in keys)
    row_of = {l: i for i, r in enumerate(rows) for l in r}
    top, anchor = 0, 0.0  # each row starts right under the previous row's last column
    for i, r in enumerate(rows):
        ks = [k for k in keys if row_of[blocks[k].layer] == i]
        y0 = min(blocks[k].y for k in ks)
        y1 = max(blocks[k].y + blocks[k].h for k in ks)
        a, cw0 = cols[r[0]]
        sgn = 1 if i % 2 == 0 else -1
        base = 0.0 if i == 0 else anchor - sgn * cw0 / 2  # where the row's first column begins (its far side when sgn<0)
        for k in ks:
            bl = blocks[k]
            rel = bl.x - a
            bl.x = int(round(base + rel)) if sgn > 0 else int(round(base - rel - bl.w))
            bl.y = bl.y - y0 + top
            bl.dir = sgn
        lx, lw = cols[r[-1]]
        anchor = base + sgn * (lx - a + lw / 2)
        top += y1 - y0 + ROW_GAP
    x0 = min(blocks[k].x for k in keys)
    for k in keys:
        blocks[k].x -= x0
    width, height = max(blocks[k].x + blocks[k].w for k in keys), top - ROW_GAP
    if free_out is not None:  # VIS-5: the empty part beside the last (short) row, for other groups of the box
        last = [k for k in keys if row_of[blocks[k].layer] == len(rows) - 1]
        lx0, lx1 = min(blocks[k].x for k in last), max(blocks[k].x + blocks[k].w for k in last)
        ly0 = min(blocks[k].y for k in last)
        right, left = (lx1 + HGAP, ly0, width, height), (0, ly0, lx0 - HGAP, height)
        free_out[id(keys)] = max(right, left, key=lambda r: r[2] - r[0])
    return width, height


def relayout(items: Dict[str, List[Dict[str, Any]]], new_id: Callable[[], int], ir: Optional[Dict[str, Any]] = None,
             key_ids: Optional[Dict[str, Dict[str, int]]] = None) -> Tuple[int, int]:
    """Re-place every visual element of ``items`` in place; returns the new canvas (width, height).

    ``ir`` and ``key_ids`` (the writer's IR and its IR-key -> PWML-id map) only serve to place edges that no
    visualization member references; an edge still tied to nothing drawn keeps its v1 shape, moved into the canvas."""
    locs: Dict[int, Dict[str, Any]] = {}
    hidden_locs: List[Dict[str, Any]] = []
    for sec in LOC_SECTIONS:
        for it in items.get(sec, []):
            if sec == "compound-locations" and str(it.get("visualization-template-id")) == str(DRUG_TEMPLATE):
                it["visualization-template-id"] = COMPOUND_TEMPLATE
            if sec == "nucleic-acid-locations" and str(it.get("visualization-template-id")) == str(PROTEIN_TEMPLATE_ON_NUCLEIC):
                it["visualization-template-id"] = NUCLEIC_ACID_TEMPLATE
            dims = (TEMPLATE_DIMS if sec == "compound-locations" else OTHER_TEMPLATE_DIMS).get(it.get("visualization-template-id"))
            if dims:  # PathWhiz draws a location at its template's size and ignores the stored one
                it["width"], it["height"] = str(dims[0]), str(dims[1])
            (hidden_locs.append(it) if it.get("hidden") else locs.__setitem__(int(it["id"]), it))
    edges = {int(e["id"]): e for e in items.get("edges", [])}
    # VIS-5: PathBank labels each subunit of a complex with its own protein name (label-type "subunit"); the writer's
    # label-type "protein" makes PathWhiz print the complex name on every subunit (TcyJKLMN five times). A complex
    # with two or more drawn subunits, none of them PathBank's Unknown placeholder, gets PathBank's labelling; a
    # one-subunit complex and an Unknown subunit keep the complex name (the enzyme name the paper gives).
    prot_name = {str(pr.get("id")): str(pr.get("name") or "").strip().casefold() for pr in items.get("proteins", [])}
    loc_by_id = {int(it["id"]): it for it in items.get("protein-locations", [])}
    for pcv in items.get("protein-complex-visualizations", []):
        subs = [loc_by_id.get(int(m["protein-location-id"])) for m in pcv.get("protein_complex_protein_visualizations", [])
                if m.get("protein-location-id") is not None]
        subs = [l for l in subs if l is not None and not l.get("hidden")]
        if len(subs) >= 2 and all(prot_name.get(str(l.get("protein-id")), "unknown") not in ("", "unknown") for l in subs):
            for l in subs:
                if l.get("label-type") == "protein":
                    l["label-type"] = "subunit"

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
    edge_role: Dict[int, Tuple[str, Any, str]] = {}  # edge -> (process kind "r"/"t", hub block, "in" or "out")
    attached: set = set()  # blocks a process visualization refers to; PathWhiz's viewer drops the rest (isOrphan)
    hub_of: Dict[Tuple[str, Any], Any] = {}
    # side compounds (PathBank's co-substrate / co-product pills): a small currency compound used by one process only,
    # beside a reaction that has a main (non-currency) member on the same side in the same compartment
    uses: Dict[Any, int] = defaultdict(int)
    for sec in ("reaction-visualizations", "transport-visualizations", "interaction-visualizations"):
        for viz in items.get(sec, []):
            for val in viz.values():
                for m in (val if isinstance(val, list) else []):
                    for f, x in (m.items() if isinstance(m, dict) else []):
                        k = pcv_block(x) if f == "protein-complex-visualization-id" else loc_block(x) if f.endswith("location-id") else None
                        if k is not None:
                            uses[k] += 1
    compounds = {c.get("id"): c for c in items.get("compounds", [])}
    from t2pw.pwml.writer import _is_chain_hub_compound_record

    def is_currency(k):
        bl = blocks.get(k)
        if bl is None or len(bl.members) != 1 or "compound-id" not in bl.members[0] or bl.w > PILL_MAX or bl.h > PILL_MAX:
            return False
        loc = bl.members[0]
        return loc.get("visualization-template-id") in SIDE_TEMPLATES or _is_chain_hub_compound_record(compounds.get(loc.get("compound-id")) or {})

    pills: Dict[Any, List[Tuple[Any, str]]] = defaultdict(list)  # hub -> [(side compound block, "in"/"out")]
    pill_of: Dict[Any, Any] = {}
    pending: List[Tuple[Any, List[Any], List[Tuple[Any, Any, Any, str]]]] = []  # (hub, enzymes, members with roles)
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
        roles = []
        for m, k in members:
            e = edges.get(int(m["edge-id"])) if m.get("edge-id") is not None else None
            product = _options(e.get("options") if e is not None else None).get("start_arrow") is True
            roles.append((m, k, e, "out" if product else "in"))
        pending.append((hub, hubs, roles))
        for m, k, e, role in roles:
            if e is not None:
                edge_ends[int(e["id"])] = (k if k is not None else hub, hub)
                edge_role[int(e["id"])] = ("r", hub, role)
    # VIS-5: PathBank draws the other enzymes of one reaction (isozymes, subunits listed separately) as a stack of
    # overlapping ellipses at the reaction point; an enzyme no other process refers to joins its first enzyme's block
    firsts = {hubs[0] for _, hubs, _ in pending if hubs}
    for _, hubs, _ in pending:
        for k in list(hubs[1:]):
            if uses[k] != 1 or k in firsts or k not in blocks or hubs[0] not in blocks:
                continue
            hb, kb = blocks[hubs[0]], blocks.pop(k)
            n = getattr(hb, "stacked", 0) + 1
            hb.stacked = n
            for loc, (dx, dy) in zip(kb.members, kb.offsets):
                hb.members.append(loc)
                hb.offsets.append((dx + STACK_DX * n, dy + STACK_DY * n))
                block_of_loc[int(loc["id"])] = hubs[0]
            hb.w = max(ox + _dim(l["width"]) for l, (ox, _) in zip(hb.members, hb.offsets))
            hb.h = max(oy + _dim(l["height"]) for l, (_, oy) in zip(hb.members, hb.offsets))
            hubs.remove(k)
            attached.discard(k)
    # PathBank's fork: one main member on the shaft, side compounds beside it. With two or more main members meeting at
    # one side of an enzyme (one reaction or several sharing the enzyme) their edges fan in through the side compounds'
    # places, so that side keeps its compounds in the layered layout.
    def main(k):
        return k is not None and not (uses[k] == 1 and is_currency(k))

    mains: Dict[Tuple[Any, str], set] = defaultdict(set)
    for hub, _, roles in pending:
        for _, k, _, r in roles:
            if main(k):
                mains[(hub, r)].add(k)
    for hub, hubs, roles in pending:
        for m, k, e, role in roles:
            if k is None:
                continue
            ms = mains[(hub, role)]
            if (not main(k) and k not in pill_of and len(ms) == 1 and blocks[next(iter(ms))].state == blocks[k].state
                    and any(k2 in ms for _, k2, _, r2 in roles if r2 == role)):
                pills[hub].append((k, role))
                pill_of[k] = hub
                attached.add(k)
                continue
            for h in hubs or [hub]:
                flow.append((h, k) if role == "out" else (k, h))
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
                edge_role[int(m["edge-id"])] = ("t", hub, "in" if left else "out")
                edges[int(m["edge-id"])]["visualization-template-id"] = TRANSPORT_EDGE_TEMPLATE
            if k is not None:
                flow.append((k, hub) if left else (hub, k))
    n_flow_after_transports = len(flow)
    hub_set = set(hub_of.values())
    interaction_edges: set = set()  # VIS-8: edges of interactions (no hub end): drawn last, shortest first
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
                    interaction_edges.add(int(m["edge-id"]))

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
    # a location no process refers to (PathWhiz's viewer drops it) that has a drawn twin of the same entity (the IR's
    # transporter location in its own membrane state next to the one the transport uses): drawn on top of its twin,
    # never as a stray box of its own
    def entity(loc):
        f = next((f for f in ("protein-id", "nucleic-acid-id", "element-collection-id", "compound-id") if f in loc), None)
        return (f, loc[f]) if f else None

    drawn_by_entity: Dict[Any, Tuple[Any, int]] = {}  # entity -> (drawn block, index of its location in the block)
    for k, bl in blocks.items():
        if k in attached:
            for i, loc in enumerate(bl.members):
                if entity(loc) is not None:
                    drawn_by_entity.setdefault(entity(loc), (k, i))
    twin_of = {k: drawn_by_entity[entity(bl.members[0])] for k, bl in blocks.items()
               if k not in attached and len(bl.members) == 1 and entity(bl.members[0]) in drawn_by_entity}
    # a reaction with side compounds reserves room above and below its enzyme (the block grows evenly, so the shaft
    # stays on the block's centre line); each side compound gets (hub, role, above -1 / below +1, distance to the shaft)
    pill_at: Dict[Any, Tuple[Any, str, int, int]] = {}
    for hub, lst in pills.items():
        hb, reach = blocks[hub], 0
        for role in ("in", "out"):
            level = {-1: PILL_PERP, 1: PILL_PERP}
            for i, (k, _) in enumerate([p for p in lst if p[1] == role]):
                s = -1 if i % 2 == 0 else 1
                pill_at[k] = (hub, role, s, level[s])
                level[s] += blocks[k].h + PILL_GAP
                reach = max(reach, level[s] - PILL_GAP)
        pad = max(0, reach - hb.h // 2)
        hb.h += 2 * pad
        hb.offsets = [(dx, dy + pad) for dx, dy in hb.offsets]
    keys = [k for k in blocks if k not in twin_of and k not in pill_of]
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
    # VIS-5: a group much wider than the whole picture's square-ish width folds into rows (a snake)
    limit = max(SNAKE_MIN, int((area * ASPECT) ** 0.5))
    snaked_states = set()
    snake_free: Dict[int, Tuple[int, int, int, int]] = {}
    for st, cs in comps_by_state.items():
        for i, (comp, w, h) in enumerate(cs):
            if w > SNAKE_OVER * limit and st != loose_key:
                cs[i] = (comp, *_snake(blocks, comp, limit, hub_set, snake_free))
                snaked_states.add(st)
        cs.sort(key=lambda c: (not any(k in attached for k in c[0]), -(c[1] * c[2])))
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
    # VIS-5 (STRUCTURE_DIFF G13): PathBank never draws a membrane leaflet as a box of its own. Where the depth rule
    # leaves a leaflet (or the cytoplasm) with two candidate membranes, the names decide when exactly one candidate
    # shares the side word ("inner"/"IM", "outer"/"OM", "plasma", "cytoplasmic membrane")
    names = {st: str(states.get(st, {}).get("name") or "") + " " + str(loc_names.get(states.get(st, {}).get("subcellular-location-id")) or "")
             for st in bands}
    for st in bands:
        if parent.get(st) is not None or kinds[st] not in ("leaflet", "cytoplasm", "thylakoid"):
            continue
        if kinds[st] == "thylakoid":  # thylakoids belong to a chloroplast, not to an etio- or amyloplast
            cands = [o for o in bands if kinds[o] == "plastid"]
            hit = [o for o in cands if not re.search(r"etioplast|amyloplast|leucoplast|chromoplast", names[o], re.I)]
        else:
            cands = [o for o in bands if o != st and kinds[o] == "membrane"]
            want = _side_words(names[st]) if kinds[st] == "leaflet" else {"inner", "plasma", "cytoplasmic"}
            hit = [o for o in cands if _side_words(names[o]) & want]
        if len(cands) > 1 and len(hit) == 1:
            parent[st] = hit[0]
    children: Dict[Any, List[Any]] = defaultdict(list)
    for st in bands:
        children[parent[st]].append(st)
    # cell art (VIS-2): the organism of the drawn compartments; a Gram-negative bacterium whose boxes do not already nest
    # an outer membrane / periplasm gets PathBank's envelope on its cell box (outer and inner membrane lines 250 apart,
    # peptidoglycan between): membrane visualizations only
    cell = cell_art.cell_kind(items, bands)
    gram_neg = cell == "bacterium" and cell_art.gram_negative(
        [compartment_kind(str(l.get("name") or "")) for l in items.get("subcellular-locations", [])],
        [sp.get("name") for sp in items.get("species", [])])
    flat_envelope = gram_neg and not any(kinds[s] in ("outer_membrane", "periplasm") for s in bands)
    cyto = [s for s in bands if kinds[s] == "cytoplasm"]
    if cell == "bacterium" and cyto:  # room for the chromosome: an empty group after the cytoplasm's own groups
        host = max(cyto, key=lambda s: (len(by_state[s]), -bands.index(s)))
        if host not in snaked_states:  # VIS-5: a snake's short last row leaves room for it already
            comps_by_state[host].append(([], *cell_art.CHROMOSOME_ROOM))

    def style(st):
        """(membrane template or None for a label only, gap of a double line or 0) of a compartment box."""
        if st == loose_key:
            return None, 0
        k, pk = kinds[st], kinds.get(parent.get(st))
        if flat_envelope and parent.get(st) is None and k in ("cytoplasm", "membrane"):
            return KIND_TEMPLATE[k], cell_art.ENVELOPE_GAP
        if k == "extracellular":
            return None, 0  # PathBank draws no outline around the outside of a cell
        if k == "leaflet" and pk == "membrane":
            return None, 0  # VIS-5 (G13): a leaflet is a region of its membrane's box, not a box of its own
        if (k == "cytoplasm" and pk == "membrane") or (k == "periplasm" and pk == "outer_membrane") \
                or (k in ("mito_inner", "mito_matrix") and pk in ("mito", "mito_inner")) or (k == "thylakoid" and pk == "plastid"):
            return None, 0  # the enclosing box already draws this compartment's membrane
        if k == "mito_matrix":
            return KIND_TEMPLATE[k], DOUBLE_GAP["mito"]  # the matrix alone: inside the mitochondrion's two membranes
        return KIND_TEMPLATE.get(k, MEMBRANE_TEMPLATE), DOUBLE_GAP.get(k, 0)

    def inset(st):
        tpl, gap = style(st)
        return (0 if tpl is None else 3 if tpl in THIN_TEMPLATES else BAND_STROKE // 2) + gap

    def envelope(st):
        """The cell box that draws a Gram-negative envelope (outer and inner membrane lines)."""
        return flat_envelope and parent.get(st) is None and kinds.get(st) in ("cytoplasm", "membrane")

    def line_label(st):
        """VIS-5 (STRUCTURE_DIFF G2): PathBank writes the name of a compartment with its own membrane ON that membrane
        line; the inside of the cell (cytoplasm) keeps its label inside the box."""
        text = labels.get(st, "")
        return bool(text) and style(st)[0] is not None and kinds.get(st) != "cytoplasm" and not envelope(st) \
            and len(label_parts(text)) == 1

    def pad_top(st):
        text = labels.get(st, "")
        if line_label(st):
            return inset(st) + BOX_PAD_TOP_NO_LABEL + 10
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
        free: List[List[int]] = []  # VIS-5: empty corners of snaked groups already placed in this box
        for comp, w, h in comps_by_state.get(st, []):
            spot = next((f for f in free if f[2] - f[0] >= w and f[3] - f[1] >= h), None)
            if spot is not None:  # into the empty end of a snake's last row
                placed.append((comp, spot[0], spot[1]))
                spot[0] += w + (HGAP if any(k in attached for k in comp) else LOOSE_GAP)
                continue
            if cx > 0 and cx + w > target:
                cx, cy, shelf_h = 0, cy + shelf_h + 2 * VGAP, 0
            placed.append((comp, cx, cy))
            # VIS-5: nodes no process refers to (PathWhiz's viewer drops them) pack tight instead of a column gap apart
            gap = HGAP if any(k in attached for k in comp) else LOOSE_GAP
            if id(comp) in snake_free:
                fx0, fy0, fx1, fy1 = snake_free[id(comp)]
                free.append([cx + fx0, cy + fy0, cx + fx1, cy + fy1])
            own_w, shelf_h, cx = max(own_w, cx + w), max(shelf_h, h), cx + w + gap
        own_h = cy + shelf_h
        kids = children.get(st, []) if st != loose_key else []
        for c in kids:
            measure(c)
        offs, kw, kh = pack_rows([sizes[c] for c in kids], max(target, own_w))
        inner_w = max(own_w, kw, label_w if labels.get(st) else 0)
        inner_h = own_h + (2 * VGAP if placed and kids else 0) + kh
        kid_top = own_h + (2 * VGAP if placed and kids else 0)
        if any(line_label(c) for c in kids):  # room for a child's label drawn on its top line
            inner_h += LINE_LABEL_ROOM
            kid_top += LINE_LABEL_ROOM
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
    spots: List[Tuple[int, int, Any, int]] = []  # (x, y, state, height) of each root box, relative to the first
    y = 0
    for row in rows:
        row_h = max(sizes[st][1] for _, st in row)
        spots += [(bx, y, st, row_h) for bx, st in row]
        y += row_h + BAND_GAP
    # VIS-5: when the first box is tall and the other boxes are outlined compartments, stacking them in columns
    # beside it (PathBank places organelle boxes around the main one) can give a squarer picture than rows under it
    if len(roots) > 1 and all(style(st)[0] is not None for st in roots[1:]):
        w0, h0 = sizes[roots[0]]
        cols = [(0, 0, roots[0], h0)]
        cx, cy, cw = w0 + BAND_GAP, 0, 0
        for st in roots[1:]:
            w, h = sizes[st]
            if cy > 0 and cy + h > h0:
                cx, cy, cw = cx + cw + BAND_GAP, 0, 0
            cols.append((cx, cy, st, h))
            cy, cw = cy + h + BAND_GAP, max(cw, w)

        def badness(sp):
            bw = max(x0 + sizes[st][0] for x0, _, st, _ in sp)
            bh = max(y0 + h for _, y0, _, h in sp)
            return bw * bh * (1 + abs(math.log(bw / bh / ASPECT)))
        if badness(cols) < badness(spots):
            spots = cols
    top = MARGIN + (LINE_LABEL_ROOM if spots else 0)  # VIS-5: room above the first row for labels on its top line
    for bx, by, st, h in spots:
        place(st, MARGIN + bx, top + by, sizes[st][0], h)
        row_w = max(row_w, bx + sizes[st][0])
    top += max(by + h for _, by, _, h in spots) + BAND_GAP if spots else 0
    width = MARGIN + row_w + MARGIN

    def own_rect(k) -> Tuple[int, int, int, int]:
        """What is drawn of block ``k`` (its locations; a reaction point is a small box round its centre)."""
        bl = blocks[k]
        if bl.virtual:
            cx, cy = bl.center()
            return int(cx) - JUNCTION_W // 2, int(cy) - JUNCTION_H // 2, int(cx) + JUNCTION_W // 2, int(cy) + JUNCTION_H // 2
        rs = [(bl.x + dx, bl.y + dy, bl.x + dx + _dim(l["width"]), bl.y + dy + _dim(l["height"])) for l, (dx, dy) in zip(bl.members, bl.offsets)]
        return min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs)

    for k, (hub, role, s, near) in pill_at.items():  # side compounds beside their reaction's shaft
        x0, y0, x1, y1 = own_rect(hub)
        cy, bl = (y0 + y1) // 2, blocks[k]
        d = blocks[hub].dir  # VIS-5: mirrored in a right-to-left row
        if blocks[hub].virtual:  # a reaction point: side compounds before and after it along the shaft
            cx = (x0 + x1) // 2 + d * (PILL_ALONG // 2 if role == "in" else PILL_ALONG + PILL_ALONG // 2)
        elif (role == "in") == (d > 0):
            cx = x0 - PILL_ALONG
        else:
            cx = x1 + PILL_ALONG
        bl.x, bl.y = cx - bl.w // 2, (cy - near - bl.h) if s < 0 else (cy + near)

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
    label_items: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    line_labels: List[Tuple[Dict[str, Any], int, int, int]] = []  # VIS-5: (label on a membrane line, box x0, x1, y1)

    def membrane_name(kind, default):
        """The PWML's own name of its one membrane location of ``kind`` (apical / basolateral sides do not name a whole
        cell membrane), else ``default``."""
        found = {str(l.get("name") or "").strip() for l in items.get("subcellular-locations", [])}
        found = sorted(n for n in found if n and compartment_kind(n) == kind and not re.search(r"apical|basolateral", n, re.I))
        return found[0][:1].upper() + found[0][1:] if len(found) == 1 else default

    for s, (x0, y0, x1, y1) in band_rects:
        tpl, gap = style(s)
        lines = [] if tpl is None else [(x0, y0, x1, y1)] + ([(x0 + gap, y0 + gap, x1 - gap, y1 - gap)] if gap else [])
        for (a0, b0, a1, b1) in lines:
            items.setdefault("membrane-visualizations", []).append({
                "id": new_id(), "path": _rounded_rect(a0, b0, a1, b1), "visualization-template-id": tpl,
                "complete-membrane": True, "zindex": 6, "width": f"{a1 - a0}.0", "height": f"{b1 - b0}.0"})
        d = inset(s)

        def line_lab(text, ly, x=x0 + LINE_LABEL_X, st=s):
            label_items[st].append({
                "id": new_id(), "visualization-template-id": LABEL_TEMPLATE, "label-text": text,
                "x": x, "y": int(ly - (25 + 15.5) * LABEL_SCALE), "zindex": 20, "scale-x": f"{LABEL_SCALE}",
                "scale-y": f"{LABEL_SCALE}", "width": "200", "height": "15"})
            items.setdefault("label-locations", []).append(label_items[st][-1])
            line_labels.append((label_items[st][-1], x0, x1, y1))

        if envelope(s) and gap:  # PathBank PW000779: each envelope line named on the line
            for text, ly in ((membrane_name("outer_membrane", "Outer Membrane"), y0),
                             ("Peptidoglycan", y0 + int(gap * 0.48)), (membrane_name("membrane", "Inner Membrane"), y0 + gap)):
                line_lab(text, ly)
        elif tpl is not None and parent.get(s) is None and kinds[s] == "cytoplasm" and membrane_name("membrane", ""):
            line_lab(membrane_name("membrane", ""), y0)
        if line_label(s):
            line_lab(labels[s], y0)
        for j, part in enumerate(label_parts(labels[s]) if labels.get(s) and not line_label(s) else []):
            label_items[s].append({
                "id": new_id(), "visualization-template-id": LABEL_TEMPLATE, "label-text": part,
                "x": x0 + d + 12, "y": y0 + d + 4 + j * LABEL_STEP, "zindex": 20, "scale-x": f"{LABEL_SCALE}", "scale-y": f"{LABEL_SCALE}",
                "width": "200", "height": "15"})
            items.setdefault("label-locations", []).append(label_items[s][-1])
    # peptidoglycan (VIS-2): between the outer and inner membrane lines of a Gram-negative cell (after the labels, so
    # every other record keeps its id)
    pg: List[Tuple[int, int, int, int]] = []
    if gram_neg:
        drawn_line = {s for s, _ in band_rects if style(s)[0] is not None}
        for s, (x0, y0, x1, y1) in band_rects:
            if s not in drawn_line or parent.get(s) is not None:
                continue
            if flat_envelope and kinds[s] in ("cytoplasm", "membrane"):
                g = int(style(s)[1] * 0.48)
                pg.append((x0 + g, y0 + g, x1 - g, y1 - g))
            elif kinds[s] in ("outer_membrane", "periplasm"):
                inner = [c for c in children.get(s, []) if c in drawn_line] or \
                        [g for c in children.get(s, []) if c not in drawn_line for g in children.get(c, []) if g in drawn_line]
                if len(inner) == 1 and kinds[inner[0]] in ("membrane", "cytoplasm"):
                    i0, j0, i1, j1 = rect_of_state[inner[0]]
                    pg.append(((x0 + i0) // 2, (y0 + j0) // 2, (x1 + i1) // 2, (y1 + j1) // 2))
    for (a0, b0, a1, b1) in pg:
        items.setdefault("membrane-visualizations", []).append({
            "id": new_id(), "path": _rounded_rect(a0, b0, a1, b1), "visualization-template-id": cell_art.PEPTIDOGLYCAN_TEMPLATE,
            "complete-membrane": True, "zindex": 6, "width": f"{a1 - a0}.0", "height": f"{b1 - b0}.0"})

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
    fixed = [(bl.x, bl.y, bl.x + bl.w, bl.y + bl.h) for k, bl in blocks.items() if not bl.virtual and k not in twin_of]
    on_line = {id(lb) for lb, _, _, _ in line_labels}  # a transporter may take a line label's place; the label moves after
    fixed += [_label_rect(lb) for lb in items.get("label-locations", []) if id(lb) not in on_line]
    def crossing(si, so):
        """The membrane a cargo in ``si`` crosses toward ``so``: the highest outlined box around ``si`` not holding ``so``."""
        m = next((s for s in reversed(anc[si]) if s not in anc[so]), None)
        return m if m in rect_of_state and style(m)[0] is not None else None

    on_membrane: Dict[Any, Tuple[set, bool]] = {}  # moved hub -> (states inside its membrane, membrane is the box top)
    hub_mem: Dict[Any, Any] = {}  # VIS-5: the membrane box a moved transporter sits on
    placed_cols: List[Tuple[Any, int, int, int, int]] = []  # VIS-5: (membrane box, rect) of each moved transporter

    def anc_set(mem):
        return {st for st, chain in anc.items() if mem in chain}

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
    all_states = {st.get("id"): st for st in items.get("biological-states", [])}
    twin_kinds: Dict[Any, set] = defaultdict(set)
    for k, (twin, _) in twin_of.items():
        st = all_states.get(blocks[k].state, {})
        twin_kinds[twin].add(compartment_kind(loc_names.get(st.get("subcellular-location-id")) or "", str(st.get("name") or "")))
    for hub, plan in plans.items():
        # a transporter shared by transports across different membranes stays where its group put it
        if hub in other_use or None in plan or len({(m, t) for m, t, _ in plan}) != 1:
            continue
        mem, on_top, ba = plan[0]
        x0, y0, x1, y1 = rect_of_state[mem]
        gap = style(mem)[1]
        hb = blocks[hub]
        line_y = y0 + gap // 2 if on_top else y1 - gap // 2  # a double membrane: between its two lines
        tk = twin_kinds.get(hub, set())  # VIS-5: the transporter's own membrane (its undrawn twin location) picks the line
        if gap and "outer_membrane" in tk and "membrane" not in tk:
            line_y = y0 if on_top else y1
        elif gap and "membrane" in tk and "outer_membrane" not in tk:
            line_y = y0 + gap if on_top else y1 - gap
        ny = int(line_y - hb.h / 2)
        want = int(ba.center()[0] - hb.w / 2)
        lo, hi = x0 + inset(mem) + 20, x1 - inset(mem) - 20 - hb.w
        if hi < lo:
            continue
        own = (hb.x, hb.y, hb.x + hb.w, hb.y + hb.h)
        # the first free spot from the inner cargo's x outward, preferring one the inner cargo reaches on a clear line
        cargo_r = ba.rect()
        others = [f for f in fixed if f != own]
        # VIS-5: a transporter on the other line of the same envelope keeps its column free (TolC under TonB crossed)
        others += [(c0 - 40, min(c1, ny - 400), c2 + 40, max(c3, ny + hb.h + 400)) for m, c0, c1, c2, c3 in placed_cols
                   if m == mem and abs(c1 - ny) > 20]
        inner_side = "B" if on_top else "T"
        line_rects = [_label_rect(lb) for lb, _, _, _ in line_labels]
        for others in (others + line_rects, others):  # VIS-5: keep off a line label when the line has room
            first = hit = None
            for step in range(0, x1 - x0, 20):
                for nx in dict.fromkeys(max(lo, min(hi, v)) for v in (want + step, want - step)):
                    r = (nx - 15, ny - 15, nx + hb.w + 15, ny + hb.h + 15)
                    if any(not (r[2] <= f[0] or f[2] <= r[0] or r[3] <= f[1] or f[3] <= r[1]) for f in others):
                        continue
                    first = nx if first is None else first
                    if step > TRANSPORTER_REACH:  # past reach: the first free spot
                        break
                    p3, n3 = _side_point((nx, ny, nx + hb.w, ny + hb.h), inner_side)
                    for s in _facing_sides(cargo_r, *p3):  # the sides the edge drawing will try
                        p0, n0 = _side_point(cargo_r, s)
                        _, c1, c2 = _curve(p0, n0, p3, n3)
                        if _clear(_bezier_points(p0, c1, c2, p3), [f for f in others if f != cargo_r]):
                            hit = nx
                            break
                    if hit is not None:
                        break
                if hit is not None or (first is not None and step > TRANSPORTER_REACH):
                    break
            if first is not None:
                break
        hit = first if hit is None else hit
        if hit is not None:
            if not hb.virtual:
                fixed.remove(own)
            hb.x, hb.y = hit, ny
            fixed.append((hb.x, hb.y, hb.x + hb.w, hb.y + hb.h))
            placed_cols.append((mem, hb.x, hb.y, hb.x + hb.w, hb.y + hb.h))
            on_membrane[hub] = (anc_set(mem), on_top)
            hub_mem[hub] = mem
    # VIS-5: the outside cargo of transports through a membrane sits right under (or over) its transporter instead of
    # wherever the outside box's own packing put it, when the outside compartment has no outline and holds nothing else
    hub_x = {h: blocks[h].center()[0] for h in on_membrane}
    by_outer: Dict[Any, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
    hubs_of_outer: Dict[Any, set] = defaultdict(set)
    for hub, cargo in transports:
        if hub not in on_membrane:
            continue
        inside, _ = on_membrane[hub]
        for k, _side in cargo:
            st = blocks[k].state
            if st not in inside and st in rect_of_state:
                by_outer[st][k].append(hub_x[hub])
                hubs_of_outer[st].add(hub)
    for st, want in by_outer.items():
        ks = by_state.get(st, [])
        t_hubs = {h for h, _ in transports}
        if style(st)[0] is not None or parent.get(st) is not None or any(n not in t_hubs for k in ks for n in nbrs[k]):
            continue
        for k in ks:  # outside cargo of a transport whose transporter stays put: keeps its own x in the row
            want.setdefault(k, [blocks[k].center()[0]])
            continue
        sx0, sy0, sx1, sy1 = rect_of_state[st]
        mems = [rect_of_state[m] for m in rect_of_state if style(m)[0] is not None]
        if any(not (sy0 >= r[3] or sy1 <= r[1]) and not (sx1 <= r[0] or sx0 >= r[2]) for r in mems):
            continue  # beside a box, not above or below it: moving under the transporter would land inside the box
        order_k = sorted(ks, key=lambda k: sum(want[k]) / len(want[k]))
        desired = [sum(want[k]) / len(want[k]) - blocks[k].w / 2 for k in order_k]
        seps = [0.0] + [blocks[a].w + 60 for a in order_k[:-1]]
        nh = max(blocks[k].h for k in ks)
        hs = hubs_of_outer[st]
        tops = {on_membrane[h][1] for h in hs}
        if len(tops) != 1:
            continue
        below = not tops.pop()
        # one row just outside the cell (clear of the zoom frame drawn 27 outside it), its label beside the row
        edge_y = max(rect_of_state[hub_mem[h]][3] for h in hs) if below else min(rect_of_state[hub_mem[h]][1] for h in hs)
        if (below and sy0 < edge_y) or (not below and sy1 > edge_y):
            continue
        row_y = edge_y + OUTSIDE_GAP if below else edge_y - OUTSIDE_GAP - nh
        if row_y < MARGIN:
            continue  # no room above the cell
        xs = [max(MARGIN, int(round(x))) for x in _pava(desired, seps)]
        new = [(x, row_y, x + blocks[k].w, row_y + blocks[k].h) for k, x in zip(order_k, xs)]
        if any(not (n[2] <= r[0] or r[2] <= n[0] or n[3] <= r[1] or r[3] <= n[1])
               for o, r in rect_of_state.items() if o != st for n in new):
            continue  # another compartment's box is in the way
        for k, x in zip(order_k, xs):
            blocks[k].x, blocks[k].y = x, row_y
        right = max(blocks[k].x + blocks[k].w for k in ks)
        for j, lb in enumerate(label_items.get(st, [])):
            lb["x"], lb["y"] = right + 40, row_y + nh // 2 - int(40 * LABEL_SCALE) + j * LABEL_STEP
    # VIS-5: a membrane-line label a transporter landed on slides along its line to the first free place
    hubs_r = [(blocks[h].x - 20, blocks[h].y - 20, blocks[h].x + blocks[h].w + 20, blocks[h].y + blocks[h].h + 20)
              for h in on_membrane]
    for lb, bx0, bx1, by1 in line_labels:
        def hits(x, y, lb=lb):
            r = _label_rect(dict(lb, x=x, y=y))
            return any(not (r[2] <= q[0] or q[2] <= r[0] or r[3] <= q[1] or q[3] <= r[1]) for q in hubs_r)
        if hits(lb["x"], lb["y"]):
            w = int((205 - 5) * float(lb["scale-x"]))
            bottom = int(by1 - (25 + 15.5) * float(lb["scale-y"]))  # else the box's bottom line
            free = [(abs(x - lb["x"]) + (0 if y == lb["y"] else 10000), x, y) for y in (lb["y"], bottom)
                    for x in range(bx0 + 40, bx1 - 40 - w, 20) if not hits(x, y)]
            if free:
                _, lb["x"], lb["y"] = min(free)
    for k, (twin, i) in twin_of.items():
        tb = blocks[twin]
        blocks[k].x, blocks[k].y = tb.x + tb.offsets[i][0], tb.y + tb.offsets[i][1]
    for bl in blocks.values():
        for loc, (dx, dy) in zip(bl.members, bl.offsets):
            loc["x"], loc["y"] = bl.x + dx, bl.y + dy
    for loc in hidden_locs:  # not drawn; kept inside the canvas
        loc["x"] = max(0, min(int(loc.get("x") or 0), width - _dim(loc["width"])))
        loc["y"] = max(0, min(int(loc.get("y") or 0), height - _dim(loc["height"])))

    # ---- edges: PathBank's cubic per edge (see SIDES above). A reaction's substrates meet at the middle of its enzyme's
    # left side, its products leave from the right side (top or bottom when the node lies behind the enzyme); a
    # transporter on a membrane takes its inner cargo on the side facing in and its outer cargo on the side facing out
    member_rects = {k: [(int(l["x"]), int(l["y"]), int(l["x"]) + _dim(l["width"]), int(l["y"]) + _dim(l["height"]))
                        for l in bl.members] for k, bl in blocks.items()}
    label_rects = [_label_rect(lb) for lb in items.get("label-locations", [])]
    all_rects = [r for rs in member_rects.values() for r in rs] + label_rects
    router = _Router(width, height, all_rects)

    def hub_end(eid, k, other_r):
        """(point, outward normal) where edge ``eid`` meets block ``k`` when ``k`` is its process's hub, else None."""
        kind, hub, role = edge_role.get(eid, (None, None, None))
        if hub != k:
            return None
        r = own_rect(k)
        ox, oy = (other_r[0] + other_r[2]) / 2, (other_r[1] + other_r[3]) / 2
        if kind == "t" and k in on_membrane:
            inside, on_top = on_membrane[k]
            cargo = edge_ends[eid][0] if edge_ends[eid][1] == k else edge_ends[eid][1]
            inner = blocks[cargo].state in inside
            return _side_point(r, ("B" if inner else "T") if on_top else ("T" if inner else "B"))
        if blocks[k].virtual:  # a point: met from the side facing the other end
            return ((r[0] + r[2]) // 2, (r[1] + r[3]) // 2), SIDES[_sides_toward(r, ox, oy)[0]]
        enter_left = (role == "in") == (blocks[k].dir > 0)  # VIS-5: a right-to-left row enters on the right
        s = "L" if enter_left else "R"
        if (enter_left and ox >= r[0]) or (not enter_left and ox <= r[2]):  # behind the enzyme: top or bottom
            s = "T" if oy < (r[1] + r[3]) / 2 else "B"
        return _side_point(r, s)

    band_strips: Dict[Any, List[Tuple[int, int, int, int]]] = {}
    for st, (x0, y0, x1, y1) in band_rects:
        tpl, gap = style(st)
        if tpl is not None:
            t = (3 if tpl in THIN_TEMPLATES else BAND_STROKE // 2) + 5
            band_strips[st] = [(x0 - t, y0 - t, x1 + t, y0 + t + gap), (x0 - t, y1 - t - gap, x1 + t, y1 + t),
                               (x0 - t, y0 - t, x0 + t + gap, y1 + t), (x1 - t - gap, y0 - t, x1 + t, y1 + t)]
    bound = (width, height)
    drawn: List[List[Tuple[float, float]]] = []  # visible edges drawn so far, as sampled polylines
    plain: Dict[Tuple[Any, Any], Any] = {}  # (start block, end block) -> chosen ends of a curve with no hub end
    plain_h: Dict[Tuple[Any, Any], float] = {}  # VIS-8: (start block, end block) -> height of the arch chosen for the pair

    def span(item):
        ra_, rb_ = own_rect(item[1][0]), own_rect(item[1][1])
        return abs((ra_[0] + ra_[2]) - (rb_[0] + rb_[2])) + abs((ra_[1] + ra_[3]) - (rb_[1] + rb_[3]))

    # VIS-8: reaction and transport edges first (as before), then the interactions shortest first, so the arches of one
    # hub nest (a longer span stands higher) and an interaction never changes where a process edge goes
    order_edges = [it for it in edge_ends.items() if it[0] not in interaction_edges]
    order_edges += sorted((it for it in edge_ends.items() if it[0] in interaction_edges), key=span)
    for eid, (a, b) in order_edges:
        e = edges[eid]
        ra, rb = own_rect(a), own_rect(b)
        if a == b:  # a connector from an enzyme to its own reaction: the shaft through the enzyme (its first location)
            r = member_rects[a][0] if member_rects[a] else ra
            p0, p3 = ((r[0], (r[1] + r[3]) // 2), (r[2], (r[1] + r[3]) // 2))
            path, c1, c2 = _curve(p0, (1, 0), p3, (-1, 0), bound=bound)
            e["path"] = path
            _set_arrow_paths(e, p0, c1, p3, c2)
            continue
        obstacles = [r for r in all_rects if r not in member_rects[a] and r not in member_rects[b]]
        # VIS-5: an edge between two nodes inside a box does not leave it through its membrane band
        common = set(anc.get(blocks[a].state, [])) & set(anc.get(blocks[b].state, [])) \
            if edge_role.get(eid, ("",))[0] != "t" and a not in on_membrane and b not in on_membrane else set()
        obstacles = obstacles + [q for st in common for q in band_strips.get(st, [])]
        fa, fb = hub_end(eid, a, rb), hub_end(eid, b, ra)
        ends_a = [fa] if fa else [_side_point(ra, s) for s in _facing_sides(ra, *(fb[0] if fb else ((rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2)))]
        ends_b = [fb] if fb else [_side_point(rb, s) for s in _facing_sides(rb, *(ends_a[0][0]))]
        if fa is None and fb is None and (b, a) in plain:  # the reverse of a curve already drawn between the two nodes
            ends_b, ends_a = [plain[(b, a)][0]], [plain[(b, a)][1]]  # (an interaction's hidden Left edge)
        ratio = 0.4 if a in pill_of or b in pill_of else 0.25  # side compounds: shorter, rounder branches
        # the first candidate (preferred sides first) clear of node boxes, fewest crossings with the edges drawn so far
        best, score, pick = None, None, None
        for i, (p0, n0) in enumerate(ends_a):
            for j, (p3, n3) in enumerate(ends_b):
                if p0 == p3:
                    continue
                path, c1, c2 = _curve(p0, n0, p3, n3, ratio, bound, (fa is not None, fb is not None))
                pts = _bezier_points(p0, c1, c2, p3)
                sc = (not _clear(pts, obstacles), _crossings(pts, drawn), i + j)
                if score is None or sc < score:
                    best, score, pick = (path, p0, c1, p3, c2, pts), sc, ((p0, n0), (p3, n3))
                if sc[:2] == (False, 0):
                    break
            if score is not None and score[:2] == (False, 0):
                break
        if fa is None and fb is None and score is not None and (score[0] or score[1] or (b, a) in plain_h):
            # VIS-7/VIS-8: an interaction between two nodes far apart (enzyme to enzyme): besides the straight curve, flat
            # arches round the boxes between them; the cheapest clear one wins (a crossing with another edge costs
            # CROSS_COST units of length). An arch leaves the sides that face along the pair's longer axis: top or bottom
            # for a pair side by side, left or right for a pair one above the other (VIS-7 review F3). The pair's second
            # edge takes the first one's arch (review F2).
            horizontal = abs(ra[0] + ra[2] - rb[0] - rb[2]) >= abs(ra[1] + ra[3] - rb[1] - rb[3])
            if (b, a) in plain_h:
                arc_ends, heights = [(ends_a[0], ends_b[0])], [plain_h[(b, a)]]
            else:
                arc_ends = [(_side_point(ra, s), _side_point(rb, s)) for s in (("T", "B") if horizontal else ("L", "R"))]
                chord = ((rb[0] + rb[2] - ra[0] - ra[2]) ** 2 + (rb[1] + rb[3] - ra[1] - ra[3]) ** 2) ** 0.5 / 2
                heights = [h for h in ARC_HEIGHTS if h <= 0.4 * chord + 60]
            cands = [] if score[0] or (b, a) in plain_h else [(CROSS_COST * score[1] + _plen(best[5]), 0, best, pick, None)]
            for h in heights:
                for (p0, n0), (p3, n3) in arc_ends:
                    if p0 == p3:
                        continue
                    path, c1, c2 = _arc(p0, n0, p3, n3, h, bound)
                    pts = _bezier_points(p0, c1, c2, p3)
                    if _clear(pts, obstacles):
                        cands.append((CROSS_COST * _crossings(pts, drawn) + _plen(pts), 1, (path, p0, c1, p3, c2, pts), ((p0, n0), (p3, n3)), h))
            if cands:
                cost, _, cb, cpick, h = min(cands, key=lambda c: c[:2])
                best, score, pick = cb, (False, 0, 0), cpick
                if h is not None:
                    plain_h[(a, b)] = h
        if score is not None and score[0]:
            # VIS-5 (STRUCTURE_DIFF G1): PathBank keeps an edge one cubic (rarely two). Before a grid route: the same
            # ends with the control points further out, then one smooth bend round the boxes (two cubics through a
            # point off the chord)
            short = (fa is not None, fb is not None)
            for kr in (0.5, 0.8):
                for i, (p0, n0) in enumerate(ends_a):
                    for j, (p3, n3) in enumerate(ends_b):
                        if p0 == p3:
                            continue
                        path, c1, c2 = _curve(p0, n0, p3, n3, kr, bound, short)
                        pts = _bezier_points(p0, c1, c2, p3)
                        sc = (not _clear(pts, obstacles), _crossings(pts, drawn), i + j)
                        if sc < score:
                            best, score, pick = (path, p0, c1, p3, c2, pts), sc, ((p0, n0), (p3, n3))
            if score[0]:
                for (p0, n0) in ends_a[:2]:
                    for (p3, n3) in ends_b[:2]:
                        if p0 == p3:
                            continue
                        chord = ((p3[0] - p0[0]) ** 2 + (p3[1] - p0[1]) ** 2) ** 0.5
                        for off in BEND_OFFSETS:
                            if abs(off) > 0.6 * chord:
                                continue  # a bend deeper than the edge is long loops round
                            cand = _bend(p0, n0, p3, n3, off, bound)
                            if cand is None:
                                continue
                            sc = (not _clear(cand[5], obstacles), _crossings(cand[5], drawn), abs(off))
                            if not sc[0] and (score[0] or sc[1:] < score[1:]):
                                best, score, pick = cand, sc, ((p0, n0), (p3, n3))
                        if not score[0]:
                            break
                    if not score[0]:
                        break
            if score[0]:  # round the boxes the curve hits: through a point just off one of their sides or corners
                (p0, n0), (p3, n3) = pick if pick else (ends_a[0], ends_b[0])
                hit = [r for r in obstacles if not _clear(best[5], [r])]
                vias = []
                for r in hit[:3]:
                    cx, cy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
                    vias += [(r[0] - VIA_PAD, cy), (r[2] + VIA_PAD, cy), (cx, r[1] - VIA_PAD), (cx, r[3] + VIA_PAD),
                             (r[0] - VIA_PAD, r[1] - VIA_PAD), (r[2] + VIA_PAD, r[1] - VIA_PAD),
                             (r[0] - VIA_PAD, r[3] + VIA_PAD), (r[2] + VIA_PAD, r[3] + VIA_PAD)]
                found = []
                cl2 = float((p3[0] - p0[0]) ** 2 + (p3[1] - p0[1]) ** 2) or 1.0
                for v in vias:
                    t = ((v[0] - p0[0]) * (p3[0] - p0[0]) + (v[1] - p0[1]) * (p3[1] - p0[1])) / cl2
                    if not 0.15 <= t <= 0.85:
                        continue  # a point behind either end makes the edge hook back
                    cand = _bend(p0, n0, p3, n3, 0, bound, via=v)
                    if cand is not None and _clear(cand[5], obstacles):
                        ln = sum(((q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2) ** 0.5 for o, q in zip(cand[5], cand[5][1:]))
                        found.append((_crossings(cand[5], drawn), ln, cand))
                if found:
                    cr, _, best = min(found, key=lambda f: (f[0], f[1]))
                    score = (False, cr, 0)
        if score is not None and score[0]:  # no clear curve: a right-angle route round the boxes, as PathBank's connectors
            pa, pb = ends_a[0][0], ends_b[0][0]
            ends = [(q[0] - 20, q[1] - 20, q[0] + 20, q[1] + 20) for q in (pa, pb)]  # VIS-5: an end under a label
            route = router.route(pa, pb, member_rects[a] + member_rects[b] + ends) if pa != pb else None
            if route and len(set(route)) > 1:
                # PathWhiz stores an edge path in a varchar(255): cut corners only while the path fits, never longer
                path, c1, c2 = _path(_round_corners(route))
                if len(path) > ROUTE_PATH_MAX:
                    path, c1, c2 = _path(route)
                if len(path) <= ROUTE_PATH_MAX:
                    best, pick = (path, route[0], c1, route[-1], c2, [tuple(q) for q in route]), None
        if fa is None and fb is None and best is not None and pick is not None:
            plain[(a, b)] = pick
        if best is None:  # both ends on one point (a reaction point on its own substrate): a short stub
            p0 = ends_a[0][0]
            best = _curve(p0, (1, 0), (p0[0] + 12, p0[1]), (-1, 0), bound=bound)
            best = (best[0], p0, best[1], (p0[0] + 12, p0[1]), best[2], [])
        if not e.get("hidden"):
            drawn.append(best[5])
        e["path"] = best[0]
        _set_arrow_paths(e, best[1], best[2], best[3], best[4])
    # An edge that neither a visualization member nor the IR ties to anything drawn keeps its v1 shape (PathWhiz's
    # viewer drops it as an orphan; other renderers draw it). Lines move together, unchanged, into a strip under the
    # content and the canvas grows to hold them; a zero-length one becomes a short stub in the bottom margin.
    orphans = [(e, _path_points(str(e.get("path") or ""))) for eid, e in edges.items() if eid not in edge_ends]
    lines = [(e, pts) for e, pts in orphans if len(set(pts)) > 1 and re.fullmatch(r"[MLC0-9.\s,-]*", str(e.get("path") or ""))
             and len(re.findall(r"-?\d+(?:\.\d+)?", str(e.get("path") or ""))) % 2 == 0]
    for e, pts in orphans:
        if len(set(pts)) <= 1:
            p0, p3 = (MARGIN, height - MARGIN // 2), (MARGIN + 30, height - MARGIN // 2)
            e["path"], c1, c2 = _curve(p0, (1, 0), p3, (-1, 0))
            _set_arrow_paths(e, p0, c1, p3, c2)
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

    # ---- cell and organelle art (VIS-2, pwml/cell_art.py): images and zoom boxes over the finished layout; the picture
    # moves right (bacterium image) or down (organelle cluster without a free spot) to make room
    plan: Dict[str, Any] = {"cell": cell}
    lined = [s for s, _ in band_rects if style(s)[0] is not None]

    def nodes_in(st):
        return sum(len(by_state.get(s, [])) for s, chain in anc.items() if st in chain)

    if cell == "bacterium":
        cells = [s for s in lined if parent.get(s) is None and kinds[s] in _CHAIN]
        if cells:
            c = max(cells, key=lambda s: (rect_of_state[s][2] - rect_of_state[s][0]) * (rect_of_state[s][3] - rect_of_state[s][1]))
            plan["cell_rect"] = rect_of_state[c]
            hosts = [s for s in bands if kinds[s] == "cytoplasm" and c in anc[s]]
            if hosts:
                h = max(hosts, key=lambda s: (nodes_in(s), -bands.index(s)))
                x0, y0, x1, y1 = rect_of_state[h]
                plan["chromosome_region"] = (x0 + inset(h) + 20, y0 + pad_top(h), x1 - inset(h) - 20, y1 - inset(h) - 20)
                plan["boxes_inside"] = [rect_of_state[s] for s in bands if s != h and h in anc[s]]
    elif cell == "eukaryote":
        plan["plant"] = any(kinds[s] in ("plastid", "thylakoid") for s in bands)
        fam = cell_art.FAMILY
        cands = [s for s in lined if fam.get(kinds[s]) and fam.get(kinds.get(parent.get(s))) != fam.get(kinds[s])]
        if cands:
            t = max(cands, key=lambda s: (nodes_in(s), -bands.index(s)))
            plan["target_rect"], plan["family"] = rect_of_state[t], fam[kinds[t]]
            strips = []
            for s in lined:  # the target's enclosing boxes: only their membrane bands are in the way; other boxes whole
                x0, y0, x1, y1 = rect_of_state[s]
                if s in anc[t]:
                    strips += [(x0 - 30, y0 - 30, x1 + 30, y0 + 30), (x0 - 30, y1 - 30, x1 + 30, y1 + 30),
                               (x0 - 30, y0, x0 + 30, y1), (x1 - 30, y0, x1 + 30, y1)]
                else:
                    strips.append((x0, y0, x1, y1))
            plan["membrane_strips"] = strips
    obstacles = [r for rs in member_rects.values() for r in rs] + [_label_rect(lb) for lb in items.get("label-locations", [])]
    width, height = cell_art.decorate(items, new_id, int(width), int(height), plan, obstacles, drawn, LOC_SECTIONS,
                                      _set_arrow_paths, MARGIN)
    # VIS-5: the canvas ends a margin below what is drawn (PathBank: canvas = drawing + margin); an outside row moved up
    # to its transporter left an empty strip at the bottom
    _shorten_long_paths(items)
    bottom = _drawn_bottom(items)
    height = min(height, int(bottom) + MARGIN) if bottom else height
    for loc in hidden_locs:  # not drawn; kept inside the (possibly shorter) canvas
        loc["y"] = max(0, min(int(loc.get("y") or 0), height - _dim(loc["height"])))
    return width, height


def _shorten_long_paths(items: Dict[str, List[Dict[str, Any]]]) -> None:
    """Last guard for PathWhiz's varchar(255): an edge path over EDGE_PATH_MAX is redrawn as one cubic between the same ends."""
    for e in items.get("edges", []):
        pts = _path_points(str(e.get("path") or ""))
        if len(str(e.get("path") or "")) > EDGE_PATH_MAX and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            p0, p3 = (int(x0), int(y0)), (int(x1), int(y1))
            dx, dy = p3[0] - p0[0], p3[1] - p0[1]
            n0 = (1 if dx >= 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy >= 0 else -1)
            e["path"], c1, c2 = _curve(p0, n0, p3, (-n0[0], -n0[1]))
            _set_arrow_paths(e, p0, c1, p3, c2)


def _drawn_bottom(items: Dict[str, List[Dict[str, Any]]]) -> float:
    """Lowest point of every drawn record (locations, labels, membranes, edges, cell art, zoom boxes); 0 if none."""
    ys: List[float] = []
    for sec in LOC_SECTIONS:
        ys += [float(it["y"]) + _dim(it["height"]) for it in items.get(sec, []) if not it.get("hidden")]
    ys += [float(lb["y"]) + 56 * float(lb.get("scale-y") or 1) for lb in items.get("label-locations", [])]
    for sec in ("membrane-visualizations", "edges"):
        for it in items.get(sec, []):
            ys += [py for _, py in _path_points(str(it.get("path") or ""))]
    ys += [float(d["y"]) + float(d["height"]) * float(d.get("scale-y") or 1) for d in items.get("drawable-element-locations", [])]
    ys += [float(z["p3y"]) for z in items.get("zoom-visualizations", [])] + [float(z["p1y"]) + 20 for z in items.get("zoom-visualizations", [])]
    return max(ys, default=0.0)
