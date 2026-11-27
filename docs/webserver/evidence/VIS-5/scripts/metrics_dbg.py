"""Per PWML: visible edges, routed (multi-segment) edges, edges whose curve crosses a node box (not its own ends),
canvas size and drawn extent."""
import sys, re, json, pathlib
from lxml import etree
L = lambda e: etree.QName(e).localname if isinstance(e.tag, str) else ""
def d(e): return {L(k): (k.text or "").strip() for k in e}
def bez(p0, c1, c2, p3, n=24):
    out = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        out.append((u**3*p0[0]+3*u*u*t*c1[0]+3*u*t*t*c2[0]+t**3*p3[0], u**3*p0[1]+3*u*u*t*c1[1]+3*u*t*t*c2[1]+t**3*p3[1]))
    return out
tot = {}
for p in sys.argv[1:]:
    r = etree.parse(p).getroot()
    used = {(x.text or "").strip() for x in r.iter() if L(x) == "edge-id"}
    rects = []
    for e in r.iter():
        if L(e) in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
            k = d(e)
            if k.get("hidden") == "true": continue
            x, y, w, h = (float(k.get(a) or 0) for a in ("x", "y", "width", "height"))
            rects.append((x, y, x + w, y + h))
    n = routed = cross = bent = 0
    bad = []
    for e in r.iter():
        if L(e) != "edge": continue
        k = d(e)
        if k.get("hidden") == "true" or k.get("id") not in used: continue
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", k.get("path", ""))]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(pts) < 4: continue
        n += 1
        segs = (len(pts) - 1) // 3
        routed += segs > 2; bent = locals().get("bent", 0) + (segs == 2)
        samp = []
        for i in range(segs):
            samp += bez(*pts[3*i:3*i+4])
        a, b = pts[0], pts[-1]
        def inside(q, rc, m=4): return rc[0]-m <= q[0] <= rc[2]+m and rc[1]-m <= q[1] <= rc[3]+m
        own = [rc for rc in rects if inside(a, rc) or inside(b, rc)]
        hit = any(rc[0]+6 < q[0] < rc[2]-6 and rc[1]+6 < q[1] < rc[3]-6 for q in samp for rc in rects if rc not in own)
        cross += hit
        if hit: print("   HIT edge", k.get("id"), k.get("path")[:120])
    vis = d(r.find(".//{*}pathway-visualization") if r.find(".//{*}pathway-visualization") is not None else r)
    tot[pathlib.Path(p).stem] = dict(edges=n, routed=routed, through_node=cross)
    print(f"{pathlib.Path(p).stem:24s} edges={n:3d} bent2={bent:3d} routed={routed:3d} through_node={cross:3d}")
print("TOTAL routed", sum(v["routed"] for v in tot.values()), "through_node", sum(v["through_node"] for v in tot.values()))
