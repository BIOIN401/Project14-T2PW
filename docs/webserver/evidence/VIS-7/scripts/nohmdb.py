import sys, glob, os
from lxml import etree
for p in sorted(glob.glob("C:/t/vis-pw7-work/cc_show/a3/*.pwml")) + ["C:/t/vis-pw7-work/cc_item6/a3/ent-item6.pwml"]:
    r = etree.parse(p).getroot()
    n = m = 0
    for c in r.iter("compound"):
        n += 1
        h = c.find("hmdb-id")
        if h is None or not (h.text or "").strip() or h.get("nil") == "true":
            m += 1
    print(f"{os.path.basename(p):26s} compounds={n:3d} without hmdb-id={m}")
