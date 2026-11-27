"""usage: m8.py OUT.json PWML... -- per PWML metrics: canvas w/h/aspect, visible edge count/total/median length,
interaction edge count/total/median/max length, visible-edge crossings, edges through a node box, smallest label px at 1100 wide
(smallest PathWhiz label is 12 units: PathBank PW000779.svg uses font sizes 12/15/16/20)."""
import sys, re, json, math, statistics as st, pathlib
sys.path.insert(0, "C:/t/vis-pw8-work/scripts")
from lxml import etree
from refint import survey, pathinfo, bez
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
def d(e): return {L(k): (k.text or "").strip() for k in e}
def seg_x(p, q, r, s):
    def o(a, b, c):
        v = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]); return (v > 1e-9) - (v < -1e-9)
    return o(p, q, r)*o(p, q, s) < 0 and o(r, s, p)*o(r, s, q) < 0

def one(p):
    r = etree.parse(p, etree.XMLParser(recover=True, huge_tree=True)).getroot()
    used = {(x.text or "").strip() for x in r.iter() if L(x) == "edge-id"}
    rects = []
    for e in r.iter():
        if L(e) in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
            k = d(e)
            if k.get("hidden") == "true": continue
            x, y, w, h = (float(k.get(a) or 0) for a in ("x", "y", "width", "height"))
            rects.append((x, y, x+w, y+h))
    polys, lens, thru = [], [], 0
    for e in r.iter():
        if L(e) != "edge": continue
        k = d(e)
        if k.get("hidden") == "true" or k.get("id") not in used: continue
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", k.get("path", ""))]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(pts) < 4: continue
        segs = (len(pts)-1)//3
        samp = []
        for i in range(segs): samp += bez(*pts[3*i:3*i+4], n=16)
        polys.append(samp); lens.append(sum(math.dist(a, b) for a, b in zip(samp, samp[1:])))
        a, b = pts[0], pts[-1]
        ins = lambda q, rc, m=4: rc[0]-m <= q[0] <= rc[2]+m and rc[1]-m <= q[1] <= rc[3]+m
        thru += any(rc[0]+6 < q[0] < rc[2]-6 and rc[1]+6 < q[1] < rc[3]-6 for q in samp for rc in rects if not (ins(a, rc) or ins(b, rc)))
    cross = 0
    for i in range(len(polys)):
        bi = polys[i]
        for j in range(i+1, len(polys)):
            bj = polys[j]
            if max(q[0] for q in bi) < min(q[0] for q in bj) or max(q[0] for q in bj) < min(q[0] for q in bi) \
               or max(q[1] for q in bi) < min(q[1] for q in bj) or max(q[1] for q in bj) < min(q[1] for q in bi): continue
            cross += any(seg_x(a, b, c, e2) for a, b in zip(bi, bi[1:]) for c, e2 in zip(bj, bj[1:]))
    s = open(p, encoding="utf-8").read()
    m = re.search(r"<pathway-visualization>.*?<height[^>]*>(\d+)</height>\s*<width[^>]*>(\d+)</width>", s, re.S)
    H, W = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    out, summ = survey(p)
    il = [m_["pi"][2] for iv in out for m_ in iv["mem"] if m_["hidden"] != "true" and m_["pi"]]
    return {"w": W, "h": H, "aspect": round(W/H, 2) if H else 0, "edges": len(lens), "edge_total": int(sum(lens)),
            "edge_median": int(st.median(lens)) if lens else 0, "edge_cross": cross, "through_node": thru,
            "int_n": len(il), "int_total": int(sum(il)), "int_median": int(st.median(il)) if il else 0, "int_max": int(max(il)) if il else 0,
            "label_px_1100": round(12*1100/W, 2) if W else 0}

if __name__ == "__main__":
    res = {pathlib.Path(p).stem: one(p) for p in sys.argv[2:]}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    keys = ["w", "h", "aspect", "edges", "edge_total", "edge_cross", "through_node", "int_n", "int_total", "int_max", "label_px_1100"]
    print("%-24s" % "pwml", *["%9s" % k for k in keys])
    for n, v in res.items(): print("%-24s" % n[:24], *["%9s" % v[k] for k in keys])
    if len(res) > 12:
        print("MEDIAN", {k: st.median(v[k] for v in res.values()) for k in keys})
        print("SUM edge_cross", sum(v["edge_cross"] for v in res.values()), "int_total", sum(v["int_total"] for v in res.values()),
              "int_n", sum(v["int_n"] for v in res.values()), "int_max_max", max(v["int_max"] for v in res.values()))
