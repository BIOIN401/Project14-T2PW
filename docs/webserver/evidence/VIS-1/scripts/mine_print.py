"""Compact print of mined_edges.json summary."""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
for g, s in d["summary"].items():
    print(f"== {g}  n={s['n']}")
    for k, v in s.items():
        if k == "n":
            continue
        if isinstance(v, list) and v and isinstance(v[0], list):
            v = " ".join(f"{a}:{b}" for a, b in v)
        print(f"   {k}: {v}")
