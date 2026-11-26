"""Build the drawable-image-id -> art catalog from the 16 reference PWML/SVG pairs.

For each PWML drawable-element-location, find the SVG <g id="location"> whose transform starts with
translate(x,y) and holds an <image> of the same width/height; decode its data URI. Writes the distinct
images to OUT/<id>.<ext> and prints a catalog (id, size, uses, source pathways, consistency)."""
import base64
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from lxml import etree as ET

REF = Path("C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
P = ET.XMLParser(recover=True, huge_tree=True)
S = "{http://www.w3.org/2000/svg}"
XL = "{http://www.w3.org/1999/xlink}href"

def num(s):
    return float(s) if s not in (None, "") else None

found = collections.defaultdict(dict)      # id -> sha -> (mime, bytes)
meta = collections.defaultdict(lambda: {"uses": 0, "pathways": set(), "sizes": set(), "missing": 0})
for f in sorted((REF / "pwml_markups").glob("*.pwml")):
    pw = ET.parse(str(f), P).getroot()
    sv = ET.parse(str(REF / "full_vector_images" / (f.stem + ".svg")), P).getroot()
    imgs = []
    for g in sv.iter(S + "g"):
        if g.get("id") != "location":
            continue
        m = re.match(r"\s*translate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\)", g.get("transform") or "")
        im = g.find(S + "image")
        if m and im is not None:
            imgs.append((float(m.group(1)), float(m.group(2)), num(im.get("width")), num(im.get("height")), im.get(XL) or ""))
    for d in pw.iter("drawable-element-location"):
        i = d.findtext("drawable-image-id")
        x, y, w, h = (num(d.findtext(k)) for k in ("x", "y", "width", "height"))
        md = meta[i]
        md["uses"] += 1
        md["pathways"].add(f.stem)
        md["sizes"].add((w, h))
        hit = [t for t in imgs if abs(t[0] - x) < 0.6 and abs(t[1] - y) < 0.6 and (w is None or t[2] == w) and (h is None or t[3] == h)]
        if not hit:
            md["missing"] += 1
            continue
        href = hit[0][4]
        mm = re.match(r"data:([^;,]+);base64,(.*)", href, re.S)
        if not mm:
            md["missing"] += 1
            continue
        raw = base64.b64decode(re.sub(r"\s+", "", mm.group(2)))
        found[i][hashlib.sha256(raw).hexdigest()[:12]] = (mm.group(1), raw)

cat = {}
for i in sorted(meta, key=lambda s: int(s)):
    md = meta[i]
    vs = found.get(i, {})
    row = {"uses": md["uses"], "pathways": sorted(md["pathways"]), "sizes": sorted(md["sizes"]),
           "unmatched": md["missing"], "variants": len(vs)}
    if vs:
        mime, raw = next(iter(vs.values()))
        ext = {"image/svg+xml": "svg", "image/png": "png", "image/jpeg": "jpg"}.get(mime, "bin")
        (OUT / f"{i}.{ext}").write_bytes(raw)
        row.update(mime=mime, bytes=len(raw), file=f"{i}.{ext}")
    cat[i] = row
    print(i, json.dumps(row))
(OUT / "_catalog.json").write_text(json.dumps(cat, indent=1))
