"""Crude PNG preview of a PWML's drawing (membranes in their template colour and stroke, nodes, edges, labels).
usage: preview.py FILE.pwml OUT.png [scale]"""
import re
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

HEX = {1: "#4BC8F2", 84: "#FF0000", 91: "#EE9189", 92: "#AED02A", 93: "#FEF935", 94: "#DDB0FD", 95: "#92130A",
       146: "#FF7000", 220: "#000000", 237: "#389F3C", 238: "#7CC4B7", 304: "#ABF7B1"}
root = ET.parse(sys.argv[1]).getroot()
pv = root.find(".//pathway-visualization")
s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
W, H = int(pv.findtext("width")), int(pv.findtext("height"))
img = Image.new("RGB", (max(1, int(W * s)), max(1, int(H * s))), "white")
d = ImageDraw.Draw(img)


def pts(p):
    n = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", p or "")]
    return [(n[i] * s, n[i + 1] * s) for i in range(0, len(n) - 1, 2)]


for m in pv.iter("membrane-visualization"):
    t = int(m.findtext("visualization-template-id"))
    q = pts(m.findtext("path"))
    d.line(q, fill=HEX.get(t, "#888888"), width=max(1, int((6 if t == 220 else 50) * s)))
for e in pv.iter("edge"):
    if e.findtext("hidden") == "true":
        continue
    q = pts(e.findtext("path"))
    if len(q) > 1:
        d.line([q[0]] + q[3::3] if len(q) > 3 else q, fill=(230, 160, 0), width=2)
names = {}
for kind in ("compound", "protein", "nucleic-acid", "element-collection"):
    for el in root.iter(kind):
        names[(kind, el.findtext("id"))] = el.findtext("name")
for kind, col in (("compound", (120, 180, 255)), ("protein", (90, 220, 90)), ("nucleic-acid", (255, 170, 80)), ("element-collection", (200, 200, 200))):
    for l in pv.iter(kind + "-location"):
        if l.findtext("hidden") == "true":
            continue
        x, y, w, h = (float(l.findtext(k)) * s for k in ("x", "y", "width", "height"))
        d.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), fill=col)
        d.text((x + 2, y + 2), (names.get((kind, l.findtext(kind + "-id"))) or "")[:14], fill=(0, 0, 0))
for lb in pv.iter("label-location"):
    d.text((float(lb.findtext("x")) * s + 10 * s, float(lb.findtext("y")) * s + 50 * s), lb.findtext("label-text") or "", fill=(0, 0, 160))
img.save(sys.argv[2])
print(sys.argv[2], W, H)
