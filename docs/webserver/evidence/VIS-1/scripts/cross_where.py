"""Classify crossing pairs: do the two edges share an end point (converging fork), and where do they cross?
usage: cross_where.py DIR"""
import collections
import sys
from itertools import combinations
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from layout_metrics import path_points, segs_cross  # noqa: E402

c = collections.Counter()
for f in sorted(Path(sys.argv[1]).glob("*.pwml")):
    pv = etree.parse(str(f)).getroot().find(".//pathway-visualization")
    es = [path_points(e.findtext("path") or "") for e in pv.iter("edge") if e.findtext("hidden") != "true"]
    es = [p for p in es if len(p) >= 2]
    for a, b in combinations(es, 2):
        if any(segs_cross(p, q, r, s) for p, q in zip(a, a[1:]) for r, s in zip(b, b[1:])):
            shared = bool({a[0], a[-1]} & {b[0], b[-1]})
            c["shared end" if shared else "no shared end"] += 1
print(sys.argv[1], dict(c))
