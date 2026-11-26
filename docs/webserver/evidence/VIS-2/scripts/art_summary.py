"""Per PWML: species, art images, zoom boxes, peptidoglycan lines, canvas, and how many drawn nodes an art image
covers or a zoom wedge crosses (outside its large box). usage: art_summary.py OUT.json DIR..."""
import glob
import json
import os
import sys

from lxml import etree

NODE = ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location")


def num(e, t, d=0.0):
    try:
        return float(e.findtext(t))
    except (TypeError, ValueError):
        return d


def inter(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def tri_hits(quad, r):
    """does the convex quad overlap rect r (sample the rect's corners/centre and the quad's points)"""
    def inside(p):
        s = 0
        for (x1, y1), (x2, y2) in zip(quad, quad[1:] + quad[:1]):
            c = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if c != 0:
                if s == 0:
                    s = 1 if c > 0 else -1
                elif (c > 0) != (s > 0):
                    return False
        return True
    pts = [(r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3]), ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)]
    return any(inside(p) for p in pts)


out, tot = {}, {"files": 0, "with_art": 0, "images": {}, "zooms": 0, "pg": 0, "image_node_overlaps": 0, "wedge_node_crossings": 0}
for d in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(d, "*.pwml"))):
        t = etree.parse(f).getroot()
        sp = [(s.findtext("name"), s.findtext("classification")) for s in t.iter("species")][:1]
        nodes = []
        for tag in NODE:
            for e in t.iter(tag):
                if (e.findtext("hidden") or "").lower() == "true":
                    continue
                x, y = num(e, "x"), num(e, "y")
                nodes.append((x, y, x + num(e, "width"), y + num(e, "height")))
        imgs, ov = [], 0
        for e in t.iter("drawable-element-location"):
            s = num(e, "scale-x", 1)
            r = (num(e, "x"), num(e, "y"), num(e, "x") + num(e, "width") * s, num(e, "y") + num(e, "height") * s)
            imgs.append(e.findtext("drawable-image-id"))
            ov += sum(inter(r, n) for n in nodes)
        wz = 0
        zs = list(t.iter("zoom-visualization"))
        for z in zs:
            p = [num(z, k) for k in ("p1x", "p1y", "p2x", "p2y", "p3x", "p3y")]
            big = (p[2], p[3], p[4], p[5])
            if z.findtext("visualization-template-id") == "39":
                quad = [(p[0] + 25, p[1]), (p[2], p[3]), (p[2], p[5]), (p[0] + 25, p[1] + 20)]
            else:
                quad = [(p[0], p[1] + 20), (p[2], p[3]), (p[4], p[3]), (p[0] + 25, p[1] + 20)]
            wz += sum(1 for n in nodes if not inter(n, big) and tri_hits(quad, n))
        pg = sum(1 for m in t.iter("membrane-visualization") if m.findtext("visualization-template-id") == "291")
        pv = t.find(".//pathway-visualization")
        key = os.path.basename(os.path.dirname(f.rstrip("/"))) + "/" + os.path.basename(f)
        out[key] = {"species": sp, "images": imgs, "zooms": len(zs), "pg": pg, "image_node_overlaps": ov,
                    "wedge_node_crossings": wz, "canvas": [pv.findtext("width"), pv.findtext("height")]}
        tot["files"] += 1
        tot["with_art"] += bool(imgs)
        for i in imgs:
            tot["images"][i] = tot["images"].get(i, 0) + 1
        tot["zooms"] += len(zs)
        tot["pg"] += pg
        tot["image_node_overlaps"] += ov
        tot["wedge_node_crossings"] += wz
json.dump({"totals": tot, "files": out}, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print(json.dumps(tot))
for k, v in out.items():
    if v["images"] or v["image_node_overlaps"] or v["wedge_node_crossings"]:
        print(k, v["species"], v["images"], "z", v["zooms"], "pg", v["pg"], "ov", v["image_node_overlaps"], "wedge", v["wedge_node_crossings"])
