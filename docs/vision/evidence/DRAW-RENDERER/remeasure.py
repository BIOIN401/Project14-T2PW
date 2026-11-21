"""Re-measure saved SVGs with the metrics of <tree>. usage: remeasure.py <tree> <svg glob>"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(sys.argv[1], "src"))
from t2pw.vision.checks import layout_metrics as LM  # noqa: E402

tot = {}
for p in sorted(glob.glob(sys.argv[2])):
    m = LM.svg_layout_metrics(open(p, encoding="utf-8").read())
    for k in ("crossings", "long_edges", "edge_overlaps", "score"):
        tot[k] = round(tot.get(k, 0) + m[k], 2)
    if m["edge_overlaps"]:
        print(os.path.basename(p), m["edge_overlaps"])
print(tot)
