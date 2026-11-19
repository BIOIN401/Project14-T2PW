"""Time render_svg on the largest IRs (by drawn molecule count). usage: timing.py <tree> <n> <ir glob>..."""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(sys.argv[1], "src"))
from t2pw.vision import diagram as dg  # noqa: E402
from t2pw.vision.diagram import render as R  # noqa: E402

files = [p for g in sys.argv[3:] for p in glob.glob(g)]
size = []
for p in files:
    ir = json.load(open(p, encoding="utf-8"))
    m = R.build(ir)
    size.append((len(m.nodes), p))
for n, p in sorted(size, reverse=True)[: int(sys.argv[2])]:
    ir = json.load(open(p, encoding="utf-8"))
    t0 = time.perf_counter()
    svg = dg.render_svg(ir)
    dt = time.perf_counter() - t0
    import re
    print("%-60s nodes=%d secs=%.2f style=%s tiers=%s selfcheck=%d" % (p[-60:], n, dt, re.search(r'data-style="(\w+)"', svg).group(1),
          re.search(r'data-tiers="(\d+)"', svg).group(1), len(dg.selfcheck(ir, svg)["failures"])), flush=True)
