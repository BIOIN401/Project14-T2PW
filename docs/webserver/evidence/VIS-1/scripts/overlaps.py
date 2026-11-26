"""List attached node overlaps per file. usage: overlaps.py DIR"""
import sys
from itertools import combinations
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from layout_metrics import attached_ids  # noqa: E402

for f in sorted(Path(sys.argv[1]).glob("*.pwml")):
    pv = etree.parse(str(f)).getroot().find(".//pathway-visualization")
    att = attached_ids(pv)
    rs = []
    for tag in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
        for el in pv.iter(tag):
            if el.findtext("hidden") == "true" or el.findtext("id") not in att:
                continue
            x, y = float(el.findtext("x")), float(el.findtext("y"))
            rs.append((tag[:4] + el.findtext("id") + "/t" + (el.findtext("visualization-template-id") or ""), (x, y, x + float(el.findtext("width")), y + float(el.findtext("height")))))
    for (na, a), (nb, b) in combinations(rs, 2):
        if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
            print(f.name, na, a, nb, b)
