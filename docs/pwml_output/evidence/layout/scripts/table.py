"""Markdown table of the metric totals. usage: table.py METRICS_DIR"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
cols = [("off", "v1", ""), ("off", "v2", ""), ("on", "v1", ""), ("on", "v2", ""),
        ("off", "v1", "_pwsizes"), ("off", "v2", "_pwsizes"), ("on", "v1", "_pwsizes"), ("on", "v2", "_pwsizes")]
T = {c: json.loads((d / f"metrics_{c[0]}_{c[1]}{c[2]}.json").read_text(encoding="utf-8"))["totals"] for c in cols}
rows = ["node_overlaps", "files_with_node_overlaps", "nodes_out_of_canvas", "edges_through_node", "edge_node_hits",
        "files_with_edges_through_node", "edges_out_of_canvas", "long_edges", "edge_crossings", "compartment_boxes",
        "nodes_not_in_box", "mixed_state_boxes", "nodes_outside_own_bound", "box_overlaps", "membranes", "empty_boxes", "labels",
        "internal_labels", "label_node_overlaps", "edges_under_label", "unattached_nodes", "median_canvas_w",
        "median_canvas_h", "max_canvas_w", "max_canvas_h", "median_fill_pct", "median_content_over_canvas_area_pct"]
print("| metric (97 files) | " + " | ".join(f"{m} {v}{' PW sizes' if s else ''}" for m, v, s in cols) + " |")
print("|---|" + "---:|" * len(cols))
for r in rows:
    print(f"| {r} | " + " | ".join(str(T[c].get(r)) for c in cols) + " |")
