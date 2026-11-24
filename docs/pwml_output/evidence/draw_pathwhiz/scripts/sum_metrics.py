"""Sum layout_metrics.py outputs of two arms. usage: sum_metrics.py BASE.json AFTER.json OUT.json"""
import json
import sys

KEYS = ["node_overlaps", "nodes_out_of_canvas", "edges_through_node", "edges_out_of_canvas", "long_edges", "edge_crossings",
        "empty_boxes", "label_node_overlaps", "edges_under_label", "internal_labels", "membranes", "compartment_boxes",
        "nodes_not_in_box"]


def load(p):
    d = json.load(open(p))
    return d.get("files") or d.get("rows") or d if isinstance(d, (dict, list)) else d


out = {}
for arm, p in (("base", sys.argv[1]), ("after", sys.argv[2])):
    rows = load(p)
    rows = rows if isinstance(rows, list) else list(rows.values())
    s = {k: sum(int(r.get(k) or 0) for r in rows) for k in KEYS}
    asp = sorted(r["content_w"] / max(1, r["content_h"]) for r in rows if r.get("content_h"))
    s["files"] = len(rows)
    s["aspect_median"] = round(asp[len(asp) // 2], 2) if asp else None
    s["aspect_over_4"] = sum(1 for a in asp if a > 4)
    s["canvas_area_Mpx_median"] = sorted(round(r["canvas_w"] * r["canvas_h"] / 1e6, 1) for r in rows)[len(rows) // 2]
    out[arm] = s
json.dump(out, open(sys.argv[3], "w"), indent=1)
for k in out["base"]:
    print("%-24s %10s %10s" % (k, out["base"][k], out["after"][k]))
