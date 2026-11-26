"""Scratch renderer (VIS-1): t2pw.pwml.render with arrowheads. The repo renderer reads arrowheads only from option:*_path
child elements; our PWML stores them in the <options> JSON, which PathWhiz's importer reads. This wrapper recomputes each
head the way PathWhiz draws it (edge.js.coffee drawArrow: 30 units, +-30 degrees, from the path's end point toward its
nearest control point) for every arrow flag that is true, then calls the repo renderer unchanged.
usage: render_arrows.py OUT_DIR PWML [PWML ...]   (PNG named after the PWML's parent dir + stem)"""
import json
import math
import re
import sys
from pathlib import Path

from t2pw.pwml import render as R


def head(tip, frm, angle=math.pi / 6, d=30.0):
    dx, dy = frm[0] - tip[0], frm[1] - tip[1]
    n = math.hypot(dx, dy)
    if n == 0:
        return None
    r = d / n
    x4 = (math.cos(angle) * dx - math.sin(angle) * dy) * r + tip[0]
    y4 = (math.sin(angle) * dx + math.cos(angle) * dy) * r + tip[1]
    x5 = (math.cos(-angle) * dx - math.sin(-angle) * dy) * r + tip[0]
    y5 = (math.sin(-angle) * dx + math.cos(-angle) * dy) * r + tip[1]
    return f"M {x5} {y5} L {tip[0]} {tip[1]} L {x4} {y4}"


def parse_edges(root_vis):
    out = []
    for el in root_vis.iter():
        if R._strip_ns(el.tag) != "edge":
            continue
        d = R._child_text(el, "path")
        if not d:
            continue
        flags = {}
        raw = R._child_text(el, "options")
        if raw:
            try:
                flags = {k: v for k, v in json.loads(raw).items() if not k.endswith("_path")}
            except ValueError:
                flags = {}
        for c in el:
            t = R._strip_ns(c.tag)
            if t in ("start_arrow", "end_arrow", "start_flat_arrow", "end_flat_arrow"):
                flags[t] = (c.text or "").strip() == "true"
        nums = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", d)]
        pts = list(zip(nums[0::2], nums[1::2]))
        g = R.EdgeGeom(edge_id=R._child_int(el, "id") or -1, path=d, template_id=R._child_text(el, "visualization-template-id") or "",
                       hidden=(R._child_text(el, "hidden") or "false").lower() == "true", z=R._child_int(el, "zindex") or 0)
        if len(pts) >= 2:
            if flags.get("start_arrow") is True:
                g.start_arrow_path = head(pts[0], pts[1])
            if flags.get("end_arrow") is True:
                g.end_arrow_path = head(pts[-1], pts[-2])
            if flags.get("start_flat_arrow") is True:
                g.start_flat_arrow_path = head(pts[0], pts[1], math.pi / 2)
            if flags.get("end_flat_arrow") is True:
                g.end_flat_arrow_path = head(pts[-1], pts[-2], math.pi / 2)
        out.append(g)
    return out


R.parse_edges = parse_edges
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for p in sys.argv[2:]:
    p = Path(p)
    png = out / f"{p.parent.name}_{p.stem}.png"
    try:
        R.render_pwml(str(p), str(png))
        print("OK", png)
    except Exception as e:  # noqa: BLE001
        print("FAIL", p, type(e).__name__, e)
