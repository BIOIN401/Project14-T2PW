"""usage: intx.py PWML -- for each visible interaction edge: length, segments, crossings with other visible edges"""
import sys, re, math
sys.path.insert(0, "C:/t/vis-pw8-work/scripts")
from lxml import etree
from refint import L, d, bez
from m8 import seg_x
r = etree.parse(sys.argv[1], etree.XMLParser(recover=True, huge_tree=True)).getroot()
inter = set()
for iv in r.iter():
    if L(iv) == "interaction-visualization":
        for x in iv.iter():
            if L(x) == "edge-id": inter.add(x.text.strip())
used = {(x.text or "").strip() for x in r.iter() if L(x) == "edge-id"}
polys = {}
for e in r.iter():
    if L(e) == "edge":
        k = d(e)
        if k.get("hidden") == "true" or k["id"] not in used: continue
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", k.get("path", ""))]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(pts) < 4: continue
        samp = []
        for i in range((len(pts)-1)//3): samp += bez(*pts[3*i:3*i+4], n=16)
        polys[k["id"]] = (samp, pts, k.get("visualization-template-id"))
for eid, (sp, pts, t) in polys.items():
    if eid not in inter: continue
    n = [o for o, (s2, _, _) in polys.items() if o != eid and any(seg_x(a, b, c, e2) for a, b in zip(sp, sp[1:]) for c, e2 in zip(s2, s2[1:]))]
    ni = sum(o in inter for o in n)
    print(eid, "tpl", t, "len", int(sum(math.dist(a, b) for a, b in zip(sp, sp[1:]))), "ctrl", [tuple(int(v) for v in p) for p in pts[:4]], "cross", len(n), "(with interactions", ni, ")")
