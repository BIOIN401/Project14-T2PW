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
print("%-14s %6s %8s %7s %6s  note" % ("file", "states", "distinct", "before", "after"))
for k in B:
    root = etree.parse(k, P).getroot()
    tup = []
    for st in root.findall(".//biological-states/biological-state"):
        g = lambda t: (st.findtext(t) or "").strip() or None
        tup.append((g("species-id"), g("tissue-id"), g("cell-type-id"), g("subcellular-location-id")))
    d = len(set(tup))
    b = B[k]["created"].get("biological_states", 0); a = A[k]["created"].get("biological_states", 0)
    note = "" if (b == a) else ("after==distinct" if a == d else "AFTER!=distinct")
    if b == a and a != d: note = "unchanged, != distinct"
    print("%-14s %6d %8d %7d %6d  %s" % (os.path.basename(k), len(tup), d, b, a, note))
