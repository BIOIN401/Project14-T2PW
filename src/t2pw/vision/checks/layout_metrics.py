"""Attractiveness metrics (docs/vision/PLAN.md row V7): how readable a drawing is, measured, not judged.

``pwml_layout_metrics(pwml_bytes)``  on the PWML drawing coordinates (what PathWhiz draws): element overlaps, edge
crossings, reading-order violations, label fit, duplicated compounds, enzyme distance.
``svg_layout_metrics(svg)``          on our renderer's SVG: the same where it applies, using ``diagram.metrics``
(crossings) and the same enzyme-link definition, so it equals ``diagram.diagram_metrics`` of the IR it was drawn from.

Definitions are the ones the PWML layout and correctness lanes used (docs/pwml_output/evidence/layout/scripts/
layout_metrics.py, C:/t/corr-assess/lanes/L3scripts/l3_metrics.py, l3_labels.py), so the totals over the 97 stored
runs can be compared with docs/pwml_output/CORRECTNESS-FIXES.md (reading order 20 of 253 pairs; SVG crossings 425,
enzyme links over 200 px 6). Pure and offline; PIL (label widths) is imported only when a label is measured.

    python -m t2pw.vision.checks.layout_metrics --runlist runlist.json --idx ok_idx.txt --scope default --out m.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Compounds every metabolic reaction shares; a chain that "runs backwards" through one of them is not a reading-order claim.
CURRENCY = {"atp", "adp", "amp", "nadh", "nad+", "nad", "nadph", "nadp+", "nadp", "h2o", "water", "coa", "coa-sh", "coenzyme a",
            "co2", "carbon dioxide", "h+", "hydrogen ion", "phosphate", "pi", "ppi", "pyrophosphate", "diphosphate", "sam",
            "s-adenosyl-l-methionine", "s-adenosylmethionine", "sah", "s-adenosylhomocysteine", "o2", "oxygen", "nh3", "ammonia",
            "udp", "gtp", "gdp", "fad", "fadh2"}
LOCS = ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location")
LABEL_GEOM = {"15": (160, 10), "235": (200, 16)}  # label template -> (box width, px per line), label_locations/*.jst.eco
LABEL_MARGIN = 10  # 5 px each side inside the box
FAR_ENZYME = 200  # px, centre of the enzyme to the centre of its reaction (the SVG metric's threshold)
_NUM = re.compile(r"[MLCZmlcz]|-?\d+(?:\.\d+)?")
Rect = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- small helpers
def _txt(el, tag: str, default: str = "") -> str:
    c = el.find(tag)
    return (c.text or "").strip() if c is not None and c.text else default


def _num(el, tag: str, default: float = 0.0) -> float:
    try:
        return float(_txt(el, tag))
    except ValueError:
        return default


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _kids(pv, tag: str) -> list:
    e = pv.find(tag)
    return list(e) if e is not None else []


def _overlap(a: Rect, b: Rect) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _centre(r: Rect) -> Tuple[float, float]:
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


def _mean(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) if pts else None


def path_points(d: str, steps: int = 8) -> List[Tuple[float, float]]:
    """Polyline approximation of an M/L/C path (cubic segments sampled)."""
    toks, pts, i, cmd = _NUM.findall(d or ""), [], 0, None
    while i < len(toks):
        t = toks[i]
        if t in "MLCZmlcz":
            cmd, i = t.upper(), i + 1
        elif cmd in ("M", "L") and i + 1 < len(toks):
            pts.append((float(toks[i]), float(toks[i + 1])))
            i += 2
        elif cmd == "C" and i + 5 < len(toks):
            c1, c2, p = [(float(toks[i + k]), float(toks[i + k + 1])) for k in (0, 2, 4)]
            p0 = pts[-1] if pts else c1
            for s in range(1, steps + 1):
                u = s / steps
                a, b, c, e = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
                pts.append((a * p0[0] + b * c1[0] + c * c2[0] + e * p[0], a * p0[1] + b * c1[1] + c * c2[1] + e * p[1]))
            i += 6
        else:
            i += 1
    return pts


def _segs_cross(a, b, c, d) -> bool:
    def o(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 1e-9) - (v < -1e-9)
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def label_lines(text: str, length: int = 28) -> List[str]:
    """The lines PathWhiz's label template wraps a label into (words longer than ``length`` are cut)."""
    words: List[str] = []
    for w in (text or "").split():
        words += [w[i:i + length] for i in range(0, len(w), length)] if len(w) > length else [w]
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > length:
            lines.append(cur)
            cur = w
        else:
            cur = "%s %s" % (cur, w) if cur else w
    return lines + ([cur] if cur else [])


def _text_width(line: str) -> float:
    """Width of one label line in the diagram renderer's measuring font (its own 12 px font and 6% safety)."""
    from t2pw.vision.diagram import render as R
    return R.tw(line)


# --------------------------------------------------------------------------- PWML
def pwml_layout_metrics(pwml: bytes | str, *, pathwhiz_sizes: bool = False) -> Dict[str, Any]:
    """Drawing metrics of one PWML, read from its stored coordinates.

    ``pathwhiz_sizes``: size compound locations by their visualization template (what PathWhiz draws) instead of the
    stored width/height. Keys (all counts unless noted):

    nodes, edges, labels, canvas_w, canvas_h
    node_overlaps            pairs of visible location boxes (compounds, proteins, nucleic acids, collections) that intersect
    label_node_overlaps      drawn label box x node box pairs that intersect
    edge_crossings           pairs of visible edges whose paths cross (a shared end point is not a crossing)
    precedence_pairs         same-state producer -> consumer reaction pairs through a non-currency compound
    reading_order_violations those pairs whose consumer is drawn neither to the right nor 100 px or more below the producer
    labels_too_wide          labels whose widest wrapped line is wider than the label box
    labels_outside_box       labels whose drawn box is not inside a membrane box
    internal_labels          labels that show a generated state name (starting ``__`` or ``AutoState_``)
    compound_locations, used_compound_locations, distinct_compound_state
    duplicated_compound_locations   drawn extra copies: used locations beyond one per (compound, biological state)
    duplicated_compound_keys        (compound, state) pairs drawn more than once
    enzymes_drawn, enzyme_distance_max (px), enzyme_distance_gt200
                             centre of the enzyme (its protein locations) to the centre of the reaction (its drawn compounds)
    examples                 up to three of each offender, for a gallery caption
    """
    root = ET.fromstring(pwml)
    pv = root if root.tag == "pathway-visualization" else root.find(".//pathway-visualization")
    if pv is None:
        raise ValueError("no <pathway-visualization> in the PWML")
    dims: Dict[int, Tuple[int, int]] = {}
    if pathwhiz_sizes:
        from t2pw.pwml.compound_templates import TEMPLATE_DIMS
        dims = dict(TEMPLATE_DIMS)
    nm = {_txt(e, "id"): _txt(e, "name") for e in _kids(pv, "compounds")}

    # ---- boxes
    nodes: List[Dict[str, Any]] = []
    box: Dict[str, Rect] = {}  # location id -> box, hidden ones too (a reaction may still reference them)
    for tag in LOCS:
        for el in pv.iter(tag):
            x, y, w, h = _num(el, "x"), _num(el, "y"), _num(el, "width"), _num(el, "height")
            t = _txt(el, "visualization-template-id")
            if tag == "compound-location" and t.isdigit() and int(t) in dims:
                w, h = dims[int(t)]
            r = (x, y, x + w, y + h)
            box[_txt(el, "id")] = r
            if _txt(el, "hidden", "false") != "true":
                nodes.append({"id": _txt(el, "id"), "r": r})
    overlaps = [(a["id"], b["id"]) for a, b in combinations(nodes, 2) if _overlap(a["r"], b["r"])]

    # ---- edges
    edges = []
    for el in pv.iter("edge"):
        if _txt(el, "hidden", "false") == "true":
            continue
        pts = path_points(_txt(el, "path"))
        if len(pts) >= 2:
            edges.append((_txt(el, "id"), list(zip(pts, pts[1:])), (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))))
    crossing = [(a[0], b[0]) for a, b in combinations(edges, 2)  # boxes first: most pairs are far apart
                if min(a[2][2], b[2][2]) >= max(a[2][0], b[2][0]) and min(a[2][3], b[2][3]) >= max(a[2][1], b[2][1])
                and any(_segs_cross(p, q, r, s) for p, q in a[1] for r, s in b[1])]

    # ---- labels
    membranes = []
    for m in pv.iter("membrane-visualization"):
        pts = path_points(_txt(m, "path"), steps=1)
        if pts:
            membranes.append((min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)))
    labels = 0
    too_wide, outside, internal, lab_over = [], 0, 0, 0
    for lb in pv.iter("label-location"):
        text, sc = _txt(lb, "label-text"), _num(lb, "scale-x", 1.0) or 1.0
        bw, lh = LABEL_GEOM.get(_txt(lb, "visualization-template-id"), (160, 10))
        lines = label_lines(text)
        x, y = _num(lb, "x"), _num(lb, "y")
        r = (x + 5 * sc, y + 25 * sc, x + (5 + bw) * sc, y + (40 + lh * len(lines)) * sc)
        labels += 1
        internal += text.strip().startswith(("__", "AutoState_"))
        outside += not any(m[0] <= r[0] and m[1] <= r[1] and r[2] <= m[2] and r[3] <= m[3] for m in membranes)
        lab_over += sum(_overlap(r, n["r"]) for n in nodes)
        if lines and max(_text_width(ln) for ln in lines) * sc > (bw - LABEL_MARGIN) * sc:
            too_wide.append(text)

    # ---- reactions: drawn compounds, enzymes, reading order, duplicates, enzyme distance
    rx = []
    for r in _kids(pv, "reactions"):
        side = lambda tag: [_txt(e, "element-id") for e in _kids(r, tag) if _txt(e, "element-type") == "Compound"]
        rx.append({"id": _txt(r, "id"), "left": side("reaction-left-elements"), "right": side("reaction-right-elements")})
    rviz = {}
    for v in _kids(pv, "reaction-visualizations"):
        rviz[_txt(v, "reaction-id")] = {
            "state": _txt(v, "biological-state-id"),
            "cvis": [_txt(c, "compound-location-id") for c in v.iter("reaction-compound-visualization")]
            + [_txt(c, "nucleic-acid-location-id") for c in v.iter("reaction-nucleic-acid-visualization")]
            + [_txt(c, "element-collection-location-id") for c in v.iter("reaction-element-collection-visualization")],
            "evis": [_txt(c, "protein-complex-visualization-id") for c in v.iter("reaction-enzyme-visualization")]}
    cstate = {_txt(l, "id"): (_txt(l, "compound-id"), _txt(l, "biological-state-id")) for l in pv.iter("compound-location")}
    pcv = {_txt(v, "id"): [_txt(x, "protein-location-id") for x in v.iter("protein-complex-protein-visualization")]
           for v in _kids(pv, "protein-complex-visualizations")}
    used = {l for v in rviz.values() for l in v["cvis"]}
    per = defaultdict(list)
    for lid in cstate:
        if lid in used:
            per[cstate[lid]].append(lid)
    dup_keys = {k: v for k, v in per.items() if len(v) > 1}

    def where(r) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """(enzyme centre, reaction centre) of one reaction visualization; either may be None."""
        v = rviz.get(r["id"])
        if not v:
            return None, None
        enz = _mean([_centre(box[p]) for e in v["evis"] for p in pcv.get(e, []) if p in box])
        return enz, _mean([_centre(box[l]) for l in v["cvis"] if l in cstate and l in box])  # compound locations, as the L3 lane

    far, dmax, n_enz = [], 0.0, 0
    for r in rx:
        enz, rc = where(r)
        if enz and rc:
            n_enz += 1
            d = math.dist(enz, rc)
            dmax = max(dmax, d)
            if d > FAR_ENZYME:
                far.append((r["id"], round(d)))

    prod, cons = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rx):
        for e in r["right"]:
            prod[e].append(i)
        for e in r["left"]:
            cons[e].append(i)
    prec, back = 0, []
    for c in sorted(set(prod) & set(cons)):
        if _norm(nm.get(c, "")) in CURRENCY:
            continue
        for a in prod[c]:
            for b in cons[c]:
                if a == b or rviz.get(rx[a]["id"], {}).get("state") != rviz.get(rx[b]["id"], {}).get("state"):
                    continue
                ea, ra = where(rx[a])
                eb, rb = where(rx[b])
                pa, pb = ea or ra, eb or rb  # the enzyme's place when it has one, else the drawn compounds'
                if pa is None or pb is None:
                    continue
                prec += 1
                if not (pb[0] > pa[0] or pb[1] > pa[1] + 100):  # reading order: left to right, or a wrap to a row 100 px lower
                    back.append((nm.get(c, c), round(pa[0]), round(pb[0])))

    return {
        "nodes": len(nodes), "edges": len(edges), "labels": labels,
        "canvas_w": _num(pv, "width"), "canvas_h": _num(pv, "height"),
        "node_overlaps": len(overlaps), "label_node_overlaps": lab_over, "edge_crossings": len(crossing),
        "precedence_pairs": prec, "reading_order_violations": len(back),
        "labels_too_wide": len(too_wide), "labels_outside_box": outside, "internal_labels": internal,
        "compound_locations": len(cstate), "used_compound_locations": len(used & set(cstate)), "distinct_compound_state": len(per),
        "duplicated_compound_locations": sum(len(v) - 1 for v in dup_keys.values()), "duplicated_compound_keys": len(dup_keys),
        "enzymes_drawn": n_enz, "enzyme_distance_max": round(dmax), "enzyme_distance_gt200": len(far),
        "examples": {"overlaps": overlaps[:3], "crossings": crossing[:3], "reading_order": back[:3],
                     "labels_too_wide": too_wide[:3], "far_enzymes": far[:3],
                     "duplicated": [[nm.get(k[0], k[0]), len(v)] for k, v in list(dup_keys.items())[:3]]},
    }


# --------------------------------------------------------------------------- our SVG
class _E:  # the two attributes diagram.metrics reads
    def __init__(self, pts, kind):
        self.pts, self.kind = pts, kind


def _local(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _classes(el) -> List[str]:
    return (el.get("class") or "").split()


def _svg_path_points(d: str) -> List[Tuple[float, float]]:
    """The vertices render.path_d rounded: M p0, then per bend ``L a Q vertex b``, then ``L last``."""
    cmds = [(c, [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", v)]) for c, v in re.findall(r"([MLQ])([^MLQ]*)", d or "")]
    pts: List[Tuple[float, float]] = []
    for i, (c, v) in enumerate(cmds):
        if c == "L" and i + 1 < len(cmds) and cmds[i + 1][0] == "Q":
            continue  # the approach to a rounded corner; the corner's vertex is the Q control point
        if len(v) >= 2:
            pts.append((v[0], v[1]))
    return pts


def _svg_box(g) -> Optional[Rect]:
    for s in g:
        if _local(s) == "rect":
            x, y, w, h = (float(s.get(k) or 0) for k in ("x", "y", "width", "height"))
            return (x, y, x + w, y + h)
        if _local(s) == "polygon":
            p = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", s.get("points") or "")]
            return (min(p[0::2]), min(p[1::2]), max(p[0::2]), max(p[1::2])) if p else None
    return None


def _dist_to_rect(p, r: Rect) -> float:
    return math.hypot(max(r[0] - p[0], 0, p[0] - r[2]), max(r[1] - p[1], 0, p[1] - r[3]))


def svg_layout_metrics(svg: str) -> Dict[str, Any]:
    """Metrics of the default (core) view of one of our SVG diagrams, from the SVG alone.

    Same keys and definitions as ``diagram.metrics.model_metrics`` (crossings, bends, enzyme_links,
    enzyme_links_gt200, enzyme_link_max, edges, drawings; ``crossings`` is that module's own function) plus
    ``node_overlaps`` (pairs of drawn node boxes that intersect, junctions included) and ``labels_truncated``
    (labels the renderer cut with an ellipsis: its label-fit check). Reaction enzymes only; transporter pills
    (a catalysis edge into a transport junction) are not enzyme links, as in model_metrics.
    """
    from t2pw.vision.diagram import metrics as M
    root = ET.fromstring(svg)
    view = next((g for g in root.iter() if _local(g) == "g" and "view-core" in _classes(g)), root)
    edges, nodes = [], []  # nodes: (class list, attrs, box, true centre)
    for g in view.iter():
        if _local(g) != "g":
            continue
        cl = _classes(g)
        if "edges" in cl:
            for p in g:
                if _local(p) == "path" and "e" in _classes(p):
                    pts = _svg_path_points(p.get("d") or "")
                    if len(pts) > 1:
                        e = _E(pts, next((c for c in _classes(p) if c != "e"), ""))
                        e.attr = p.attrib
                        edges.append(e)
        elif "nodes" in cl:
            for n in g:
                b = _svg_box(n) if _local(n) == "g" else None
                if b:
                    cx, cy = _centre(b)
                    if any("chip" in _classes(t) for t in n if _local(t) == "text"):
                        cy += 7  # the renderer draws a compartment chip under the body: the node's centre is 7 px lower
                    nodes.append((_classes(n), n.attrib, b, (cx, cy)))
    ents = [n for n in nodes if "n" in n[0]]
    jns = [n for n in nodes if "jn" in n[0]]
    d = []
    for e in edges:
        if e.kind != "cat":
            continue
        jx = [j for j in jns if j[1].get("data-p") == e.attr.get("data-p")]
        if any("transport" in j[0] for j in jx):
            continue  # transporter pill, not a reaction enzyme
        cand = [n for n in ents if n[1].get("data-ent") == e.attr.get("data-e")]
        ends = (e.pts[0], e.pts[-1])
        if not (cand and jx):
            d.append(math.dist(*ends))
            continue
        k = min((0, 1), key=lambda i: min(_dist_to_rect(ends[i], n[2]) for n in cand))  # the end that sits on the enzyme
        en = min(cand, key=lambda n: _dist_to_rect(ends[k], n[2]))
        jn = min(jx, key=lambda j: _dist_to_rect(ends[1 - k], j[2]))
        d.append(math.dist(en[3], jn[3]))
    return dict(crossings=M.crossings(edges), bends=sum(max(0, len(e.pts) - 2) for e in edges),
                enzyme_links=len(d), enzyme_links_gt200=sum(1 for x in d if x > FAR_ENZYME), enzyme_link_max=round(max(d, default=0)),
                edges=len(edges), drawings=len(ents),
                node_overlaps=sum(1 for a, b in combinations(nodes, 2) if _overlap(a[2], b[2])),
                labels_truncated=sum(1 for t in view.iter() if _local(t) == "text" and (t.text or "").endswith("\u2026")
                                     and "chip" not in _classes(t)))


# --------------------------------------------------------------------------- the 97 runs
def _sum(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ("canvas_w", "canvas_h", "enzyme_distance_max", "enzyme_link_max")] if rows else []
    t = {k: sum(r[k] for r in rows) for k in keys}
    t["runs"] = len(rows)
    for k in ("node_overlaps", "edge_crossings", "reading_order_violations", "labels_too_wide", "duplicated_compound_locations", "enzyme_distance_gt200"):
        if k in keys:
            t["runs_with_" + k] = sum(1 for r in rows if r[k])
    return t


def measure_run(run_dir, scope: str = "default") -> Dict[str, Any]:
    """PWML and SVG metrics of one stored run under ``scope`` (generate() builds both from one view)."""
    from t2pw.vision import generate as G
    from t2pw.vision.payload import big_from_run
    big = big_from_run(run_dir)
    sc, opts = G.parse_scope(scope), G.export_options_from_run(run_dir)
    try:
        pw = pwml_layout_metrics(G.generate(big, sc, "pwml", export=opts))
        sv = svg_layout_metrics(G.generate(big, sc, "svg", export=opts).decode("utf-8"))
    finally:
        G._MEMO.clear()
        G._KEPT.clear()
    return {"run": str(run_dir), "pwml": pw, "svg": sv}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m t2pw.vision.checks.layout_metrics", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="*", help="run directories (final_mapped.json)")
    ap.add_argument("--runlist", help="runlist.json (entries with q=true), with --idx")
    ap.add_argument("--idx", help="comma-separated indexes into the q=true entries, or a file holding them")
    ap.add_argument("--scope", default="default", help="paper | default | full | hops:N")
    ap.add_argument("--out", required=True)
    ap.add_argument("--totals-out", default=None, help="also write only the totals")
    a = ap.parse_args(argv)
    todo = [{"run": d, "idx": None} for d in a.run_dirs]
    if a.runlist:
        rl = [x for x in json.loads(Path(a.runlist).read_text(encoding="utf-8")) if x.get("q")]
        idx = [int(i) for i in (Path(a.idx).read_text() if a.idx and Path(a.idx).is_file() else a.idx or "").replace("\n", ",").split(",") if i.strip()]
        todo += [{"run": rl[i]["dir"], "idx": i} for i in (idx or range(len(rl)))]
    runs = []
    for n, item in enumerate(todo):
        try:
            rec = measure_run(item["run"], a.scope)
        except Exception:
            rec = {"run": item["run"], "crash": traceback.format_exc()[-600:]}
        rec["idx"] = item["idx"]
        runs.append(rec)
        print("%d/%d %s %s" % (n + 1, len(todo), item["idx"], "CRASH" if "crash" in rec else "ok"), flush=True)
    ok = [r for r in runs if "crash" not in r]
    totals = {"scope": a.scope, "pwml": _sum([r["pwml"] for r in ok]), "svg": _sum([r["svg"] for r in ok]),
              "crashed": [r["run"] for r in runs if "crash" in r]}
    Path(a.out).write_text(json.dumps({"totals": totals, "runs": runs}, indent=1, default=str), encoding="utf-8")
    if a.totals_out:
        Path(a.totals_out).write_text(json.dumps(totals, indent=1), encoding="utf-8")
    print(json.dumps(totals, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
