"""usage: overlaps.py PWML... -- visible location boxes that overlap another one by more than 10 units in both axes (excluding stacked isozymes)"""
import sys, pathlib
from lxml import etree
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
def d(e): return {L(k): (k.text or "").strip() for k in e}
tot = 0
for p in sys.argv[1:]:
    r = etree.parse(p, etree.XMLParser(recover=True, huge_tree=True)).getroot()
    rs = []
    for e in r.iter():
        if L(e) in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
            k = d(e)
            if k.get("hidden") == "true": continue
            x, y, w, h = (float(k.get(a) or 0) for a in ("x", "y", "width", "height"))
            rs.append((x, y, x + w, y + h, L(e)[:4] + k["id"]))
    bad = []
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            ox, oy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            if ox > 10 and oy > 10 and not (a[4].startswith("prot") and b[4].startswith("prot")): bad.append((a[4], b[4], int(ox), int(oy)))
    tot += len(bad)
    if bad: print(pathlib.Path(p).stem, len(bad), bad[:4])
print("TOTAL overlaps", tot)
