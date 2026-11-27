"""usage: ehit.py PWML [PAD] -- per visible edge: tpl, length, chord, boxes hit (padded by PAD) other than its own ends"""
import sys, re, pathlib
from lxml import etree
L = lambda e: etree.QName(e).localname if isinstance(e.tag, str) else ""
def d(e): return {L(k): (k.text or "").strip() for k in e}
def bez(p0, c1, c2, p3, n=24):
    return [((1-t)**3*p0[0]+3*(1-t)**2*t*c1[0]+3*(1-t)*t*t*c2[0]+t**3*p3[0], (1-t)**3*p0[1]+3*(1-t)**2*t*c1[1]+3*(1-t)*t*t*c2[1]+t**3*p3[1]) for t in [i/n for i in range(n+1)]]
def edges_of(p, pad=0, verbose=True):
    r = etree.parse(p).getroot()
    used = {(x.text or "").strip() for x in r.iter() if L(x) == "edge-id"}
    rects = []
    for e in r.iter():
        if L(e) in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
            k = d(e)
            if k.get("hidden") == "true": continue
            x, y, w, h = (float(k.get(a) or 0) for a in ("x", "y", "width", "height"))
            rects.append((x, y, x + w, y + h, L(e)[:4]+k.get("id","")))
    out = []
    for e in r.iter():
        if L(e) != "edge": continue
        k = d(e)
        if k.get("hidden") == "true" or k.get("id") not in used: continue
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", k.get("path", ""))]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(pts) < 4: continue
        segs = (len(pts)-1)//3
        samp = []
        for i in range(segs): samp += bez(*pts[3*i:3*i+4], n=30)
        ln = sum(((a[0]-b[0])**2+(a[1]-b[1])**2)**.5 for a,b in zip(samp, samp[1:]))
        a, b = pts[0], pts[-1]
        chord = ((a[0]-b[0])**2+(a[1]-b[1])**2)**.5
        def inside(q, rc, m=4): return rc[0]-m <= q[0] <= rc[2]+m and rc[1]-m <= q[1] <= rc[3]+m
        hits = [rc[4] for rc in rects if not (inside(a, rc) or inside(b, rc)) and any(rc[0]-pad < q[0] < rc[2]+pad and rc[1]-pad < q[1] < rc[3]+pad for q in samp)]
        out.append((k.get("id"), k.get("visualization-template-id"), int(ln), int(chord), segs, hits))
    return out
if __name__ == "__main__":
    pad = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    for t in edges_of(sys.argv[1], pad):
        flag = "LOOP" if t[3] and t[2] > 1.8*t[3] and t[2] > 400 else ""
        if t[5] or flag or t[1] in ("148","195","149","219","218"): print(*t, flag)
