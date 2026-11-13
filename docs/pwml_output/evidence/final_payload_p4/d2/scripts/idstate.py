"""Count biological states sharing an identical (species,tissue,location,cell-type) tuple within a file.
Usage: idstate.py label=dir ...   (dir holds pwml/<idx>.pwml)"""
import re, sys, collections
from pathlib import Path
B = re.compile(r"<biological-state>(.*?)</biological-state>", re.S)
def f(s, tag):
    m = re.search(r"<%s[^>]*?(?:/>|>([^<]*)</%s>)" % (tag, tag), s)
    return (m.group(1) or "").strip() if m else ""
res = {}
for a in sys.argv[1:]:
    name, d = a.split("=", 1)
    per = {}
    for p in sorted(Path(d, "pwml").glob("*.pwml"), key=lambda x: int(x.stem)):
        c = collections.Counter()
        for b in B.findall(p.read_text(encoding="utf-8")):
            c[tuple(f(b, t) for t in ("species-id", "tissue-id", "subcellular-location-id", "cell-type-id"))] += 1
        per[int(p.stem)] = (sum(c.values()), sum(v - 1 for v in c.values() if v > 1), sum(1 for v in c.values() if v > 1))
    res[name] = per
    print(name, "files", len(per), "states", sum(v[0] for v in per.values()),
          "surplus-duplicate states", sum(v[1] for v in per.values()),
          "states in a dup group", sum(v[1] + v[2] for v in per.values()),
          "dup groups", sum(v[2] for v in per.values()),
          "files with dups", sum(1 for v in per.values() if v[1]))
names = list(res)
for i in sorted(res[names[0]]):
    row = [res[n].get(i, (0, 0, 0)) for n in names]
    if len({r[1] for r in row}) > 1: print("  idx", i, {n: r for n, r in zip(names, row)})
