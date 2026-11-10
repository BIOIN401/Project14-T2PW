import sys, os
sys.path.insert(0, "C:/t/ev-dedup")
import pwml_dup as P
from collections import Counter, defaultdict
for f in ["PW012926", "PW000001", "PW176888", "PW741182", "PW687775"]:
    root = P.load("reference/%s.pwml" % f)
    names = {}
    for tag in ("compound", "protein", "protein-complex", "nucleic-acid"):
        for e in root.iter(tag):
            if e.find("id") is not None and e.find("name") is not None:
                names[(tag, P.t(e, "id"))] = P.t(e, "name")
    loc = defaultdict(list)
    for tag, kind, idtag in [("compound-location", "compound", "compound-id"), ("protein-location", "protein", "protein-id"), ("protein-complex-location", "protein-complex", "protein-complex-id")]:
        for e in root.iter(tag):
            loc[(kind, P.t(e, idtag))].append(P.t(e, "biological-state-id"))
    for k, v in loc.items():
        c = Counter(v)
        if any(x > 1 for x in c.values()) or len(set(v)) > 1:
            print(f, names.get(k), dict(c))
