"""usage: dangling.py PWML... -- visible edges with an end that touches no location box (within 14 units): the end is in empty space"""
import sys, re, pathlib
from lxml import etree
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
def d(e): return {L(k): (k.text or "").strip() for k in e}
tot = 0
for p in sys.argv[1:]:
    r = etree.parse(p, etree.XMLParser(recover=True, huge_tree=True)).getroot()
    rects = []
    for e in r.iter():
        if L(e) in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
            k = d(e)
            if k.get("hidden") == "true": continue
            x, y, w, h = (float(k.get(a) or 0) for a in ("x", "y", "width", "height"))
            rects.append((x, y, x + w, y + h))
    used = {(x.text or "").strip() for x in r.iter() if L(x) == "edge-id"}
    n = 0; bad = []
    for e in r.iter():
        if L(e) != "edge": continue
        k = d(e)
        if k.get("hidden") == "true" or k.get("id") not in used: continue
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", k.get("path", ""))]
        if len(nums) < 8: continue
        for q in ((nums[0], nums[1]), (nums[-2], nums[-1])):
            if not any(a[0] - 14 <= q[0] <= a[2] + 14 and a[1] - 14 <= q[1] <= a[3] + 14 for a in rects):
                bad.append((k["id"], q)); 
    print(pathlib.Path(p).stem, "visible edge ends in empty space:", len(bad), bad[:6])
    tot += len(bad)
print("TOTAL", tot)
