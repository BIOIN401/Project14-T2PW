"""Clarity metrics for one rendered model (the default view): crossings, bends, enzyme links longer than 200 px.
Moved from the v2 prototype; the batch CLI was dropped (see docs/vision/results/V3.json)."""
import math


def _segs(e):
    return list(zip(e.pts, e.pts[1:])) if e.pts and len(e.pts) > 1 else []


def crossings(edges):
    H, V = [], []
    for i, e in enumerate(edges):
        for (x0, y0), (x1, y1) in _segs(e):
            if abs(y0 - y1) < 0.5 and abs(x0 - x1) > 0.5:
                H.append((min(x0, x1), max(x0, x1), y0, i))
            elif abs(x0 - x1) < 0.5 and abs(y0 - y1) > 0.5:
                V.append((min(y0, y1), max(y0, y1), x0, i))
    n = 0
    for hx0, hx1, hy, i in H:
        for vy0, vy1, vx, j in V:
            if i != j and hx0 + 1 < vx < hx1 - 1 and vy0 + 1 < hy < vy1 - 1:
                n += 1
    return n


def model_metrics(m):
    cat = [e for e in m.edges if e.kind == "cat" and not hasattr(e.a, "attached")]  # reaction enzymes, not transporter pills
    d = [math.dist((e.a.x, e.a.y), (e.b.x, e.b.y)) for e in cat]
    return dict(crossings=crossings(m.edges), bends=sum(max(0, len(e.pts) - 2) for e in m.edges if e.pts),
                enzyme_links=len(cat), enzyme_links_gt200=sum(1 for x in d if x > 200), enzyme_link_max=round(max(d, default=0)),
                edges=len(m.edges), drawings=sum(1 for n in m.nodes.values() if n.kind != "junction"))
