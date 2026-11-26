"""List visible edges passing through a node box that is not one of their ends (layout_metrics rule), with what they hit.
usage: through.py DIR"""
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from layout_metrics import path_points, seg_hits_rect, inside, attached_ids  # noqa: E402

for f in sorted(Path(sys.argv[1]).glob("*.pwml")):
    pv = etree.parse(str(f)).getroot().find(".//pathway-visualization")
    att = attached_ids(pv)
    nodes = []
    for tag in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
        for el in pv.iter(tag):
            if el.findtext("hidden") == "true":
                continue
            x, y = float(el.findtext("x")), float(el.findtext("y"))
            nodes.append((tag[:4] + el.findtext("id") + ("" if el.findtext("id") in att else "(orphan)"),
                          (x, y, x + float(el.findtext("width")), y + float(el.findtext("height")))))
    for e in pv.iter("edge"):
        if e.findtext("hidden") == "true":
            continue
        pts = path_points(e.findtext("path") or "")
        if len(pts) < 2:
            continue
        ends = {n for n, r in nodes if inside(pts[0], r, 3) or inside(pts[-1], r, 3)}
        hits = [n for n, r in nodes if n not in ends and any(seg_hits_rect(a, b, (r[0] + 2, r[1] + 2, r[2] - 2, r[3] - 2)) for a, b in zip(pts, pts[1:]))]
        if hits:
            print(f.name, "edge", e.findtext("id"), "tpl", e.findtext("visualization-template-id"), "hits", hits, "ends", sorted(ends))
