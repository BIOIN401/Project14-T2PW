"""VIS-2: mine PathBank's organelle / cell art convention from the 16 reference PWMLs.
Per file: species, canvas, every drawable-element-location (catalog name, drawn box), every zoom-visualization (which
drawable holds p1, which membrane box p2/p3 matches), membranes (template, box). usage: mine_art.py REF_DIR CATALOG OUT.json"""
import json
import re
import sys
from pathlib import Path

from lxml import etree as ET

PARSER = ET.XMLParser(recover=True, huge_tree=True)
ref, cat, out = Path(sys.argv[1]), json.load(open(sys.argv[2], encoding="utf-8")), sys.argv[3]


def num(e, t, d=0.0):
    v = e.findtext(t)
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def bbox(path):
    n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path or "")]
    xs, ys = n[0::2], n[1::2]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


res = {}
for f in sorted(ref.glob("*.pwml")):
    t = ET.parse(str(f), PARSER).getroot()
    pv = t.find(".//pathway-visualization")
    sp = [(s.findtext("name"), s.findtext("taxonomy-id")) for s in t.iter("species")]
    cts = [c.findtext("name") for c in t.iter("cell-type")]
    locs = [s.findtext("name") for s in t.iter("subcellular-location")]
    states = [s.findtext("name") for s in t.iter("biological-state")]
    dr = []
    for d in t.iter("drawable-element-location"):
        i = d.findtext("drawable-image-id")
        w, h = num(d, "width"), num(d, "height")
        sx, sy = num(d, "scale-x", 1), num(d, "scale-y", 1)
        x, y = num(d, "x"), num(d, "y")
        dr.append({"id": d.findtext("id"), "image": i, "name": cat.get(i, {}).get("name"), "x": x, "y": y, "sx": sx, "sy": sy,
                   "deg": d.findtext("degree"), "z": d.findtext("zindex"), "tpl": d.findtext("visualization-template-id"),
                   "w": w, "h": h, "box": [x, y, x + w * sx, y + h * sy]})
    mem = []
    for m in t.iter("membrane-visualization"):
        mem.append({"tpl": m.findtext("visualization-template-id"), "box": bbox(m.findtext("path")), "z": m.findtext("zindex"),
                    "complete": m.findtext("complete-membrane")})
    zo = []
    for z in t.iter("zoom-visualization"):
        p = {k: num(z, k) for k in ("p1x", "p1y", "p2x", "p2y", "p3x", "p3y")}
        holder = [d["name"] or d["image"] for d in dr if d["box"][0] <= p["p1x"] <= d["box"][2] and d["box"][1] <= p["p1y"] <= d["box"][3]]
        big = (min(p["p2x"], p["p3x"]), min(p["p2y"], p["p3y"]), max(p["p2x"], p["p3x"]), max(p["p2y"], p["p3y"]))
        near = sorted(mem, key=lambda m: sum(abs(a - b) for a, b in zip(m["box"] or (0, 0, 0, 0), big)))[:1]
        zo.append({**p, "tpl": z.findtext("visualization-template-id"), "color": z.findtext("color"), "z": z.findtext("zindex"),
                   "label": z.findtext("label"), "p1_in": holder, "big": big,
                   "nearest_membrane": [(m["tpl"], m["box"]) for m in near]})
    res[f.stem] = {"species": sp, "cell_types": cts, "locations": locs, "states": states,
                   "canvas": (pv.findtext("width"), pv.findtext("height")) if pv is not None else None,
                   "drawables": dr, "zooms": zo, "membranes": mem}
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
for k, r in res.items():
    print(f"== {k} species={r['species']} cells={r['cell_types']} canvas={r['canvas']}")
    print("   locations:", r["locations"])
    for d in r["drawables"]:
        print(f"   D img {d['image']:>4} {str(d['name'])[:28]:28} x{d['x']:.0f} y{d['y']:.0f} s{d['sx']}/{d['sy']} deg{d['deg']} z{d['z']} tpl{d['tpl']} box{[int(v) for v in d['box']]}")
    for z in r["zooms"]:
        print(f"   Z p1({z['p1x']:.0f},{z['p1y']:.0f}) p2({z['p2x']:.0f},{z['p2y']:.0f}) p3({z['p3x']:.0f},{z['p3y']:.0f}) tpl{z['tpl']} {z['color']} z{z['z']} label={z['label']} p1_in={z['p1_in']} near={z['nearest_membrane']}")
    for m in r["membranes"]:
        print(f"   M tpl{m['tpl']} z{m['z']} box{[int(v) for v in (m['box'] or ())]}")
