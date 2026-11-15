import json, os
from lxml import etree
P = etree.XMLParser(recover=True)
def load(p):
    d = {}
    for l in open(p, encoding="utf-8").read().splitlines():
        if l.strip():
            r = json.loads(l); d[r["file"].replace("\\", "/")] = r
    return d
B = load("C:/t/ev-fp/p5/pwpatch2/before/results.jsonl"); A = load("C:/t/ev-fp/p5/pwpatch2/after/results.jsonl")
bad = 0
for k in B:
    root = etree.parse(k, P).getroot(); sts = []
    for st in root.findall(".//biological-states/biological-state"):
        g = lambda t: (st.findtext(t) or "").strip() or None
        sts.append(((g("species-id"), g("tissue-id"), g("cell-type-id"), g("subcellular-location-id")), g("pwbs-id")))
    # a state omitting a component that has an earlier, strictly more specific sibling (same given components, same pwbs when present)
    pred = False
    for i, (s, ps) in enumerate(sts):
        if None in s[1:] or s[0] is None:
            for t, pt in sts[:i]:
                if t != s and all(x is None or x == y for x, y in zip(s, t)) and (ps is None or ps == pt):
                    pred = True
    changed = (B[k]["success"], B[k]["errors"], B[k]["created"]) != (A[k]["success"], A[k]["errors"], A[k]["created"])
    flag = "" if pred == changed else "  <-- MISMATCH"
    bad += pred != changed
    print("%-14s predicted_wildcard_bind=%-5s changed=%-5s%s" % (os.path.basename(k), pred, changed, flag))
print("mismatches", bad)
