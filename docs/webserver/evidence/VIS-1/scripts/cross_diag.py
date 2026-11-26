"""Crossing pairs and edges-through-node by edge kind (what refers to the edge), per arm.
usage: cross_diag.py DIR [DIR ...]"""
import collections
import sys
from itertools import combinations
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from layout_metrics import path_points, segs_cross, seg_hits_rect, inside  # noqa: E402

KINDS = {"reaction-visualization": "rxn", "transport-visualization": "tr", "interaction-visualization": "int"}
for d in sys.argv[1:]:
    pairs, per_file, through = collections.Counter(), collections.Counter(), collections.Counter()
    for f in sorted(Path(d).glob("*.pwml")):
        pv = etree.parse(str(f)).getroot().find(".//pathway-visualization")
        kind = {}
        for tag, k in KINDS.items():
            for v in pv.iter(tag):
                for e in v.iter("edge-id"):
                    kind.setdefault(e.text, k)
        es = []
        for e in pv.iter("edge"):
            if e.findtext("hidden") == "true":
                continue
            pts = path_points(e.findtext("path") or "")
            if len(pts) >= 2:
                es.append((kind.get(e.findtext("id"), "orphan"), pts))
        for (ka, a), (kb, b) in combinations(es, 2):
            if any(segs_cross(p, q, r, s) for p, q in zip(a, a[1:]) for r, s in zip(b, b[1:])):
                pairs[tuple(sorted((ka, kb)))] += 1
                per_file[f.name] += 1
    print(d, dict(pairs), "top files", per_file.most_common(8))
