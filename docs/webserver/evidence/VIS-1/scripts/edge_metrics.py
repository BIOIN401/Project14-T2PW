"""Edge-shape metrics of PWML sets (VIS-1), from stored geometry.
usage: edge_metrics.py OUT.json LABEL=DIR [LABEL=DIR ...]   (DIR: *.pwml, or */pathway.pwml for showcase dirs)
Per set: edges, visible edges, zero-length edges (all points equal), curved visible edges (a control point more than
1 unit off the chord), single-cubic edges, edges carrying an <options> element, arrow-flag edges, arrow edges whose
arrow tangent is degenerate (end point == its control point: PathWhiz draws a NaN head), and node overlaps between
attached nodes only (the nodes PathWhiz's viewer keeps)."""
import json
import re
import sys
from itertools import combinations
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from layout_metrics import attached_ids  # noqa: E402

LOCS = ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location")


def off_chord(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = (dx * dx + dy * dy) ** 0.5
    if n == 0:
        return ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5
    return abs(dy * (p[0] - a[0]) - dx * (p[1] - a[1])) / n


def measure(f):
    root = etree.parse(str(f)).getroot()
    pv = root.find(".//pathway-visualization")
    r = dict(edges=0, visible=0, zero_length=0, curved=0, single_cubic=0, with_options=0, arrow_edges=0, degenerate_arrow=0,
             attached_overlaps=0)
    for e in pv.iter("edge"):
        r["edges"] += 1
        d = e.findtext("path") or ""
        nums = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", d)]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(set(pts)) <= 1:
            r["zero_length"] += 1
        hidden = (e.findtext("hidden") or "false") == "true"
        cmds = re.findall(r"[A-Za-z]", d)
        o = e.find("options")
        flags = {}
        if o is not None and o.text:
            r["with_options"] += 1
            try:
                flags = json.loads(o.text)
            except ValueError:
                flags = {}
        if any(flags.get(k) is True for k in ("start_arrow", "end_arrow", "start_flat_arrow", "end_flat_arrow")):
            r["arrow_edges"] += 1
            if len(pts) >= 2 and ((flags.get("start_arrow") or flags.get("start_flat_arrow")) and pts[0] == pts[1]
                                  or (flags.get("end_arrow") or flags.get("end_flat_arrow")) and pts[-1] == pts[-2]):
                r["degenerate_arrow"] += 1
        if hidden:
            continue
        r["visible"] += 1
        if cmds == ["M", "C"]:
            r["single_cubic"] += 1
        if len(pts) >= 4 and any(off_chord(p, pts[0], pts[-1]) > 1 for p in pts[1:-1]):
            r["curved"] += 1
    att = attached_ids(pv)
    rects = []
    for tag in LOCS:
        for el in pv.iter(tag):
            if (el.findtext("hidden") or "false") == "true" or el.findtext("id") not in att:
                continue
            x, y = float(el.findtext("x")), float(el.findtext("y"))
            rects.append((x, y, x + float(el.findtext("width")), y + float(el.findtext("height"))))
    r["attached_overlaps"] = sum(1 for a, b in combinations(rects, 2) if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))
    return r


out = {}
for arg in sys.argv[2:]:
    label, d = arg.split("=", 1)
    files = sorted(Path(d).glob("*.pwml")) or sorted(Path(d).glob("*/pathway.pwml"))
    tot = {}
    for f in files:
        for k, v in measure(f).items():
            tot[k] = tot.get(k, 0) + v
    tot["files"] = len(files)
    tot["pct_curved_of_visible"] = round(100.0 * tot["curved"] / max(1, tot["visible"]), 1)
    tot["pct_single_cubic_of_visible"] = round(100.0 * tot["single_cubic"] / max(1, tot["visible"]), 1)
    tot["pct_with_options"] = round(100.0 * tot["with_options"] / max(1, tot["edges"]), 1)
    out[label] = tot
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
for label, t in out.items():
    print(label, json.dumps(t))
