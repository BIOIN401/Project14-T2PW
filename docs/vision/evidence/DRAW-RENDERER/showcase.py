"""Render the 8 showcase IRs with the renderer in <tree>, measure, selfcheck. Also the dumped stored-run IRs if given.
usage: showcase.py <tree> <out_dir> [ir_dir_glob ...]"""
import glob
import json
import os
import sys
import time

TREE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(TREE, "src"))
from t2pw.vision import diagram as dg  # noqa: E402
from t2pw.vision.checks import layout_metrics as LM  # noqa: E402

SLUGS = ["menaquinone", "enterobactin", "coenzyme-a", "lipid-a", "chlorophyll", "ceramide-sphingolipid", "furochromone", "sour-taste"]
os.makedirs(OUT, exist_ok=True)
items = [] if os.environ.get("NOSHOW") else [(s, "C:/t/showcase2/%s/pwml_ir.json" % s) for s in SLUGS]
for pat in sys.argv[3:]:
    items += [(os.path.splitext(os.path.basename(p))[0], p) for p in sorted(glob.glob(pat))]
rows = {}
for name, p in items:
    ir = json.load(open(p, encoding="utf-8"))
    t0 = time.time()
    svg = dg.render_svg(ir)
    dt = time.time() - t0
    m = LM.svg_layout_metrics(svg)
    m["selfcheck_failures"] = len(dg.selfcheck(ir, svg)["failures"])
    m["secs"] = round(dt, 2)
    import re
    m["style"] = re.search(r'data-style="(\w+)"', svg).group(1)
    rows[name] = m
    open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8").write(svg)
    print(name, m["style"], "score", m["score"], "x", m["crossings"], "long", m["long_edges"], "ar", m["aspect_ratio"],
          "nodeedge", m["node_edge_overlaps"], "lab", m["label_collisions"], "bands", m["bands"], "sc", m["selfcheck_failures"], "t", m["secs"], flush=True)
keys = [k for k, v in next(iter(rows.values())).items() if isinstance(v, (int, float)) and k not in ("canvas_w", "canvas_h", "aspect_ratio", "enzyme_link_max", "secs")]
tot = {k: round(sum(r[k] for r in rows.values()), 2) for k in keys}
ar = sorted(r["aspect_ratio"] for r in rows.values() if r["aspect_ratio"])
tot.update(n=len(rows), runs_aspect_gt_4=sum(a > 4 for a in ar), aspect_median=ar[len(ar) // 2] if ar else None, aspect_max=ar[-1] if ar else None)
json.dump({"totals": tot, "rows": rows}, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
print(json.dumps(tot))
