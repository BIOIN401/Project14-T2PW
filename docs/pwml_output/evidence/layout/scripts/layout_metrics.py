"""Layout quality of PWML files, read from their stored coordinates (what PathWhiz draws).

usage: layout_metrics.py [--pathwhiz-sizes] OUT.json FILE_OR_DIR [FILE_OR_DIR ...]

--pathwhiz-sizes: size compound locations by their template (t2pw TEMPLATE_DIMS) and template-2 protein /
nucleic-acid locations at 150x70, as PathWhiz draws them;
PathWhiz keeps no width/height of its own for a location. Default: the stored width/height.

Per file: node-node overlaps, nodes outside the canvas, edges passing through a node box that is
not one of their two ends, edge points outside the canvas, edge crossings, compartment containment,
canvas size and fill. Nodes = visible compound/protein/nucleic-acid/element-collection locations.
Compartment boxes = bound-visualizations (have a biological-state-id) and closed membrane-visualizations.
"Attached" nodes are the ones PathWhiz's viewer keeps: a location a process visualization (reaction, transport,
interaction) refers to, directly or through a protein-complex visualization such a process refers to; the viewer drops
every other location (`isOrphan`, pathway_visualization.js.coffee). Compartment containment is measured on attached
nodes; empty boxes = closed membranes (what PathWhiz and render.py draw as compartment outlines) holding no attached
node; bound-visualizations count as boxes for containment only (neither viewer draws them); internal labels = label
text starting with "__" or "AutoState_" (generated state names); edges under label = visible edges crossing a drawn label box;
label overlaps = drawn label box (template 15 / 235 geometry, scale) intersecting a node box.
"""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

from lxml import etree

LOCS = ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location")
NUM = re.compile(r"[MLCZmlcz]|-?\d+(?:\.\d+)?")


def txt(el, tag):
    c = el.find(tag)
    return None if c is None or c.text is None else c.text.strip()


def num(el, tag, default=0.0):
    t = txt(el, tag)
    try:
        return float(t)
    except (TypeError, ValueError):
        return default


def path_points(d, steps=8):
    """Polyline approximation of an M/L/C path (cubic segments sampled)."""
    toks = NUM.findall(d or "")
    pts, i, cmd = [], 0, None
    while i < len(toks):
        t = toks[i]
        if t in "MLCZmlcz":
            cmd = t.upper()
            i += 1
            continue
        if cmd in ("M", "L"):
            pts.append((float(toks[i]), float(toks[i + 1])))
            i += 2
        elif cmd == "C":
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


def seg_hits_rect(p, q, r):
    """Liang-Barsky: does segment p-q enter the open rect r=(x0,y0,x1,y1)?"""
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


def inside(pt, r, tol=0.0):
    return r[0] - tol <= pt[0] <= r[2] + tol and r[1] - tol <= pt[1] <= r[3] + tol


def segs_cross(a, b, c, d):
    def o(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (v > 1e-9) - (v < -1e-9)
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


PW_DIMS = {}
OTHER_DIMS = {}
LABEL_GEOM = {"15": (160, 10), "235": (200, 16)}  # template: box width, px per line (label_locations/*.jst.eco)


def label_lines(text, length=28):
    words = []
    for w in (text or "").split():
        words += [w[i:i + length] for i in range(0, len(w), length)] if len(w) > length else [w]
    lines, cur = 0, ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > length:
            lines, cur = lines + 1, w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines + (1 if cur else 0)


def attached_ids(pv):
    used_pcv, ids = set(), set()
    procs = [v for t in ("reaction-visualization", "transport-visualization", "interaction-visualization") for v in pv.iter(t)]
    for v in procs:
        for el in v.iter():
            if not isinstance(el.tag, str):
                continue
            if el.tag.endswith("location-id"):
                ids.add(el.text)
            elif el.tag == "protein-complex-visualization-id":
                used_pcv.add(el.text)
    for pcv in pv.iter("protein-complex-visualization"):
        if pcv.findtext("id") in used_pcv:
            ids.update(e.text for e in pcv.iter("protein-location-id"))
    return ids


def measure(path):
    root = etree.parse(str(path)).getroot()
    pv = root.find(".//pathway-visualization")
    W, H = num(pv, "width"), num(pv, "height")
    nodes = []
    att = attached_ids(pv)
    for tag in LOCS:
        for el in pv.iter(tag):
            if (txt(el, "hidden") or "false") == "true":
                continue
            x, y, w, h = num(el, "x"), num(el, "y"), num(el, "width"), num(el, "height")
            t = txt(el, "visualization-template-id")
            if tag == "compound-location" and t and t.isdigit() and int(t) in PW_DIMS:
                w, h = PW_DIMS[int(t)]
            elif tag != "compound-location" and t and t.isdigit() and int(t) in OTHER_DIMS:
                w, h = OTHER_DIMS[int(t)]
            nodes.append({"id": txt(el, "id"), "state": txt(el, "biological-state-id"), "r": (x, y, x + w, y + h),
                          "attached": txt(el, "id") in att})
    edges = []
    for el in pv.iter("edge"):
        if (txt(el, "hidden") or "false") == "true":
            continue
        pts = path_points(txt(el, "path"))
        if len(pts) >= 2:
            edges.append({"id": txt(el, "id"), "pts": pts})
    boxes, membranes = [], []
    for el in pv.iter("bound-visualization"):
        if (txt(el, "hidden") or "false") != "true":
            x, y = num(el, "x"), num(el, "y")
            boxes.append({"state": txt(el, "biological-state-id"), "r": (x, y, x + num(el, "width"), y + num(el, "height"))})
    for el in pv.iter("membrane-visualization"):
        d = txt(el, "path") or ""
        pts = path_points(d, steps=1)
        if pts and (d.strip().upper().endswith("Z") or (len(pts) > 2 and pts[0] == pts[-1])):
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            r = (min(xs), min(ys), max(xs), max(ys))
            membranes.append(r)
            if not any(b["r"] == r for b in boxes):  # a membrane drawn over a bound box is one compartment
                boxes.append({"state": None, "r": r})

    overlaps = sum(1 for a, b in combinations(nodes, 2)
                   if min(a["r"][2], b["r"][2]) > max(a["r"][0], b["r"][0]) and min(a["r"][3], b["r"][3]) > max(a["r"][1], b["r"][1]))
    out_nodes = sum(1 for n in nodes if n["r"][0] < 0 or n["r"][1] < 0 or n["r"][2] > W or n["r"][3] > H)
    through, through_pairs, out_edges, long_edges = 0, 0, 0, 0
    for e in edges:
        p0, p1 = e["pts"][0], e["pts"][-1]
        ends = {n["id"] for n in nodes if inside(p0, n["r"], 3) or inside(p1, n["r"], 3)}
        hit = 0
        for n in nodes:
            if n["id"] in ends:
                continue
            r = (n["r"][0] + 2, n["r"][1] + 2, n["r"][2] - 2, n["r"][3] - 2)
            if any(seg_hits_rect(a, b, r) for a, b in zip(e["pts"], e["pts"][1:])):
                hit += 1
        through += hit > 0
        through_pairs += hit
        out_edges += any(p[0] < 0 or p[1] < 0 or p[0] > W or p[1] > H for p in e["pts"])
        length = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(e["pts"], e["pts"][1:]))
        long_edges += length > 0.5 * max(W, H, 1)
    crossings = 0
    segs = [[(a, b) for a, b in zip(e["pts"], e["pts"][1:])] for e in edges]
    for i, j in combinations(range(len(segs)), 2):
        if any(segs_cross(a, b, c, d) for a, b in segs[i] for c, d in segs[j]):
            crossings += 1
    # compartments: every attached node in a box; a box holds one biological state only; bound boxes hold their own state
    not_in_box, mixed, own_out = 0, 0, 0
    empty_boxes = sum(1 for m in membranes if not any(n["attached"] and inside((n["r"][0], n["r"][1]), m)
                                                   and inside((n["r"][2], n["r"][3]), m) for n in nodes))
    labels = []
    for el in pv.iter("label-location"):
        t, sc = txt(el, "visualization-template-id"), num(el, "scale-x", 1.0)
        bw, lh = LABEL_GEOM.get(t, (160, 10))
        x, y = num(el, "x"), num(el, "y")
        labels.append({"text": txt(el, "label-text") or "", "r": (x + 5 * sc, y + 25 * sc, x + (5 + bw) * sc,
                                                                 y + (40 + lh * label_lines(txt(el, "label-text"))) * sc)})
    internal_labels = sum(1 for lb in labels if lb["text"].strip().startswith(("__", "AutoState_")))
    edges_under_label = sum(1 for e in edges if any(seg_hits_rect(a, b, lb["r"]) for lb in labels for a, b in zip(e["pts"], e["pts"][1:])))
    label_overlaps = sum(1 for lb in labels for n in nodes
                         if min(lb["r"][2], n["r"][2]) > max(lb["r"][0], n["r"][0]) and min(lb["r"][3], n["r"][3]) > max(lb["r"][1], n["r"][1]))
    if boxes:
        members = [[] for _ in boxes]
        for n in nodes:
            if not n["attached"]:
                continue
            hold = [k for k, b in enumerate(boxes) if inside((n["r"][0], n["r"][1]), b["r"]) and inside((n["r"][2], n["r"][3]), b["r"])]
            if not hold:
                not_in_box += 1
                continue
            k = min(hold, key=lambda k: (boxes[k]["r"][2] - boxes[k]["r"][0]) * (boxes[k]["r"][3] - boxes[k]["r"][1]))
            members[k].append(n["state"])
            own = [b for b in boxes if b["state"] is not None and b["state"] == n["state"]]
            if own and not any(inside((n["r"][0], n["r"][1]), b["r"]) and inside((n["r"][2], n["r"][3]), b["r"]) for b in own):
                own_out += 1
        mixed = sum(1 for m in members if len(set(m)) > 1)
    box_overlaps = sum(1 for a, b in combinations(boxes, 2)
                       if min(a["r"][2], b["r"][2]) > max(a["r"][0], b["r"][0]) and min(a["r"][3], b["r"][3]) > max(a["r"][1], b["r"][1]))
    if nodes:
        bx = (min(n["r"][0] for n in nodes), min(n["r"][1] for n in nodes), max(n["r"][2] for n in nodes), max(n["r"][3] for n in nodes))
    else:
        bx = (0, 0, 0, 0)
    area = sum((n["r"][2] - n["r"][0]) * (n["r"][3] - n["r"][1]) for n in nodes)
    return {
        "file": str(path), "nodes": len(nodes), "edges": len(edges), "canvas_w": W, "canvas_h": H,
        "node_overlaps": overlaps, "nodes_out_of_canvas": out_nodes,
        "edges_through_node": through, "edge_node_hits": through_pairs,
        "edges_out_of_canvas": out_edges, "long_edges": long_edges, "edge_crossings": crossings,
        "compartment_boxes": len(boxes), "nodes_not_in_box": not_in_box, "mixed_state_boxes": mixed,
        "nodes_outside_own_bound": own_out, "box_overlaps": box_overlaps, "empty_boxes": empty_boxes,
        "labels": len(labels), "internal_labels": internal_labels, "label_node_overlaps": label_overlaps,
        "edges_under_label": edges_under_label, "membranes": len(membranes),
        "unattached_nodes": sum(1 for n in nodes if not n["attached"]),
        "content_w": bx[2] - bx[0], "content_h": bx[3] - bx[1],
        "fill_pct": round(100.0 * area / (W * H), 2) if W and H else 0.0,
    }


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--pathwhiz-sizes":
        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))
        from t2pw.pwml.compound_templates import TEMPLATE_DIMS
        PW_DIMS.update(TEMPLATE_DIMS)
        OTHER_DIMS.update({2: (150, 70)})
        argv = argv[1:]
    out, args = argv[0], argv[1:]
    files = []
    for a in args:
        p = Path(a)
        files += sorted(p.glob("*.pwml"), key=lambda q: (len(q.stem), q.stem)) if p.is_dir() else [p]
    rows = [measure(f) for f in files]
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, float)) and k not in ("canvas_w", "canvas_h", "fill_pct", "content_w", "content_h")} if rows else {}
    tot["files"] = len(rows)
    for k in ("node_overlaps", "nodes_out_of_canvas", "edges_through_node", "edges_out_of_canvas"):
        tot["files_with_" + k] = sum(1 for r in rows if r[k])
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else 0
    tot["median_canvas_w"] = med([r["canvas_w"] for r in rows])
    tot["median_canvas_h"] = med([r["canvas_h"] for r in rows])
    tot["max_canvas_w"] = max((r["canvas_w"] for r in rows), default=0)
    tot["max_canvas_h"] = max((r["canvas_h"] for r in rows), default=0)
    tot["median_fill_pct"] = med([r["fill_pct"] for r in rows])
    tot["median_content_over_canvas_area_pct"] = med([round(100 * r["content_w"] * r["content_h"] / (r["canvas_w"] * r["canvas_h"]), 1) if r["canvas_w"] and r["canvas_h"] else 0 for r in rows])
    Path(out).write_text(json.dumps({"totals": tot, "files": rows}, indent=1), encoding="utf-8")
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
