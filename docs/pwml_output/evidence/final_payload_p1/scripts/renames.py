import json, glob, os, re, sys
from lxml import etree
off_dir = "C:/t/ev-fp/replay/off4/pwml"
on_dir = "C:/t/ev-fp/replay/on4/pwml"
tot_c = 0
ren = []
files = 0
def names(path):
    t = etree.parse(path)
    out = {}
    for c in t.iter("compound"):
        i = c.findtext("id")
        n = c.findtext("name")
        if i is not None and n is not None and c.getparent() is not None and c.getparent().tag == "compounds":
            out[i] = n
    return out
for f in sorted(glob.glob(on_dir + "/*.pwml")):
    b = os.path.basename(f)
    o = os.path.join(off_dir, b)
    if not os.path.exists(o):
        continue
    files += 1
    a, c = names(o), names(f)
    for k in a:
        tot_c += 1
        if k in c and a[k] != c[k]:
            ren.append((b, k, a[k], c[k]))
print("files", files, "compounds", tot_c, "renamed", len(ren))
for r in ren[:60]:
    print(r)
