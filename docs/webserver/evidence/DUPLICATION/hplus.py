import json, re
from collections import Counter
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl", encoding="utf-8")]
H = {"h+", "h(+)", "h", "proton", "protons", "hydrogen ion", "hydrogen ions", "hydron", "h⁺", "hydrogen cation"}
W = {"water", "h2o", "h₂o"}
multiH = multiW = hasH = 0; names = Counter(); ex = []
for r in rows:
    if r["quarantine_ok"] is not True: continue
    p = json.load(open(r["path"], encoding="utf-8"))
    cs = [c.get("name", "").strip().casefold() for c in p["entities"].get("compounds", []) if isinstance(c, dict)]
    h = [c for c in cs if c in H]; w = [c for c in cs if c in W]
    hasH += bool(h)
    for c in h: names[c] += 1
    if len(h) > 1: multiH += 1; ex.append((r["path"][-60:], h))
    if len(w) > 1: multiW += 1
print("canonical runs with an H+ compound:", hasH, "with >1 H+-alias records:", multiH, "with >1 water records:", multiW)
print(names); print(ex[:5])
