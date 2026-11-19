"""Dev sanity (not a test run): every style x tiers x order on the fixtures and a few dumped IRs passes selfcheck.
usage: devcheck.py <tree> [ir glob]"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(sys.argv[1], "src"))
from t2pw.vision import diagram as dg  # noqa: E402

files = sorted(glob.glob(os.path.join(sys.argv[1], "tests/data/vision/diagram/*.json")))
if len(sys.argv) > 2:
    files += sorted(glob.glob(sys.argv[2]))
bad = 0
for p in files:
    ir = json.load(open(p, encoding="utf-8"))
    for style in ("bands", "cell", "spine"):
        for tiers in (1, 2, 3):
            for order in ("flow", "anat"):
                try:
                    f = dg.selfcheck(ir, dg._R.render(ir, style, "b", tiers=tiers, order=order)[0])["failures"]
                except Exception as e:  # noqa: BLE001
                    f = ["CRASH %r" % e]
                if f:
                    bad += 1
                    print(os.path.basename(p), style, tiers, order, f[:2], flush=True)
print("DEVCHECK bad", bad, "files", len(files))
