import sys, re
sys.path.insert(0, "C:/t/vis-pw5/src")
from t2pw.pwml.render import read_xml_with_repairs
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
r = read_xml_with_repairs(sys.argv[1])
for e in r.iter():
    if L(e) == "pathway-visualization":
        d = {L(k): (k.text or "").strip() for k in e if not len(k)}; print("canvas", d.get("width"), d.get("height"))
    if L(e) == "membrane-visualization":
        d = {L(k): (k.text or "").strip() for k in e}
        n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d["path"])]
        print("membrane tpl", d.get("visualization-template-id"), "bbox", min(n[0::2]), min(n[1::2]), max(n[0::2]), max(n[1::2]))
    if L(e) == "label-location":
        d = {L(k): (k.text or "").strip() for k in e}
        print("label", d.get("label-text"), d.get("x"), d.get("y"), d.get("scale-x"))
    if L(e) == "drawable-element-location":
        d = {L(k): (k.text or "").strip() for k in e}
        print("art", d.get("drawable-image-id"), d.get("x"), d.get("y"), d.get("scale-x"))
xs=[];ys=[]
for e in r.iter():
    if L(e) in ("compound-location","protein-location"):
        d = {L(k): (k.text or "").strip() for k in e}
        xs += [float(d["x"]), float(d["x"])+float(d["width"])]; ys += [float(d["y"]), float(d["y"])+float(d["height"])]
print("node bbox", min(xs), min(ys), max(xs), max(ys))
