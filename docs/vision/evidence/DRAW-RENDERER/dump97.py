"""Dump the IR each of the 97 stored runs draws its default-scope SVG from (same path as layout_metrics.measure_run).
usage: dump97.py <tree> <out_dir>"""
import json
import os
import sys

TREE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(TREE, "src"))
os.environ["T2PW_OFFLINE_CURATOR"] = "1"
from t2pw.vision import generate as G  # noqa: E402
from t2pw.vision.payload import big_from_run  # noqa: E402

os.makedirs(OUT, exist_ok=True)
rl = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x.get("q")]
idx = [int(i) for i in open("C:/t/vis-runs/scratch/generate/ok_idx.txt").read().replace("\n", ",").split(",") if i.strip()]
for n, i in enumerate(idx):
    d = rl[i]["dir"]
    big = big_from_run(d)
    sc, opts = G.parse_scope("default"), G.export_options_from_run(d)
    G._MEMO.clear()
    try:
        G.generate(big, sc, "svg", export=opts)
        ir = next(iter(G._MEMO.values()))[1]
    finally:
        G._MEMO.clear()
        G._KEPT.clear()
    json.dump(ir, open(os.path.join(OUT, "%03d.json" % i), "w", encoding="utf-8"))
    print(n + 1, i, flush=True)
print("DUMP_DONE", len(idx))
