"""usage: refint.py PWML... -- per interaction visualization: type, partner kinds, node centres/gap, per-edge hidden/template/length/ends.
Also counts protein complexes with >1 subunit (bound complexes) and interactions. Prints a line per interaction viz and a summary."""
import sys, re, math, json
from collections import Counter
from lxml import etree
L = lambda e: e.tag.rsplit("}", 1)[-1].split(":")[-1] if isinstance(e.tag, str) else ""
def d(e): return {L(k): (k.text or "").strip() for k in e}
def num(x):
    try: return float(x)
    except Exception: return 0.0
def bez(p0, c1, c2, p3, n=24):
    return [((1-t)**3*p0[0]+3*(1-t)**2*t*c1[0]+3*(1-t)*t*t*c2[0]+t**3*p3[0], (1-t)**3*p0[1]+3*(1-t)**2*t*c1[1]+3*(1-t)*t*t*c2[1]+t**3*p3[1]) for t in [i/n for i in range(n+1)]]
def pathinfo(path):
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path or "")]
    pts = list(zip(nums[0::2], nums[1::2]))
    if len(pts) < 4: return None
    segs = (len(pts)-1)//3
    samp = []
    for i in range(segs): samp += bez(*pts[3*i:3*i+4])
    ln = sum(math.dist(a, b) for a, b in zip(samp, samp[1:]))
    return pts[0], pts[-1], ln, segs
def side_of(p, r):
    x0, y0, x1, y1 = r
    ds = {"L": abs(p[0]-x0), "R": abs(p[0]-x1), "T": abs(p[1]-y0), "B": abs(p[1]-y1)}
    s = min(ds, key=ds.get)
    return s if ds[s] < 25 else "in" if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 else "out"

def survey(path):
    r = etree.parse(path, etree.XMLParser(recover=True, huge_tree=True)).getroot()
    locs = {}
    for e in r.iter():
        if L(e) in ("compound-location", "protein-location", "nucleic-acid-location", "element-collection-location"):
            k = d(e)
            if k.get("id"):
                locs[(L(e), k["id"])] = (num(k.get("x")), num(k.get("y")), num(k.get("x"))+num(k.get("width")), num(k.get("y"))+num(k.get("height")), k.get("hidden") == "true")
    pcv = {}
    for e in r.iter():
        if L(e) == "protein-complex-visualization":
            ids = [x.text.strip() for x in e.iter() if L(x) == "protein-location-id" and x.text]
            rs = [locs[("protein-location", i)] for i in ids if ("protein-location", i) in locs and not locs[("protein-location", i)][4]]
            if rs:
                pcv[d(e)["id"]] = (min(q[0] for q in rs), min(q[1] for q in rs), max(q[2] for q in rs), max(q[3] for q in rs), False)
    edges = {}
    for e in r.iter():
        if L(e) == "edge":
            k = d(e); edges[k["id"]] = k
    itype = {}
    iparts = {}
    for e in r.iter():
        if L(e) == "interaction" and e.find("interaction-type") is not None:
            k = d(e); itype[k["id"]] = k["interaction-type"]
            lt = [d(x).get("element-type") for x in e.iter() if L(x) == "interaction-left-element"]
            rt = [d(x).get("element-type") for x in e.iter() if L(x) == "interaction-right-element"]
            iparts[k["id"]] = (lt, rt)
    out = []
    for iv in r.iter():
        if L(iv) != "interaction-visualization": continue
        iid = (iv.findtext("interaction-id") or "").strip()
        mem = []
        for sec in iv:
            if not L(sec).startswith("interaction_"): continue
            for m in sec:
                k = d(m)
                kind = L(m).replace("interaction-", "").replace("-visualization", "")
                if "protein-complex-visualization-id" in k: rect = pcv.get(k["protein-complex-visualization-id"])
                else:
                    f = next((f for f in k if f.endswith("location-id")), None)
                    rect = locs.get((f[:-3], k[f])) if f else None
                ed = edges.get(k.get("edge-id"), {})
                mem.append({"kind": kind, "side": k.get("side"), "rect": rect, "hidden": ed.get("hidden"), "tpl": ed.get("visualization-template-id"), "pi": pathinfo(ed.get("path")), "z": ed.get("zindex")})
        out.append({"iid": iid, "type": itype.get(iid, "?"), "parts": iparts.get(iid), "mem": mem})
    n_pc = sum(1 for e in r.iter() if L(e) == "protein-complex")
    multi = 0
    for e in r.iter():
        if L(e) == "protein-complex":
            if len([x for x in e.iter() if L(x) == "protein-complex-protein"]) > 1: multi += 1
    return out, {"interactions": len(itype), "types": dict(Counter(itype.values())), "protein_complexes": n_pc, "multi_subunit": multi,
                 "bound": sum(1 for e in r.iter() if L(e) == "bound")}

if __name__ == "__main__":
    for p in sys.argv[1:]:
        out, summ = survey(p)
        print("==", p.replace("\\", "/").rsplit("/", 1)[-1], json.dumps(summ))
        for iv in out:
            print(f" I{iv['iid']} {iv['type']} parts={iv['parts']}")
            rs = [m["rect"] for m in iv["mem"] if m["rect"]]
            for m in iv["mem"]:
                rc = m["rect"]; pi = m["pi"]
                cen = ((rc[0]+rc[2])/2, (rc[1]+rc[3])/2) if rc else None
                ends = ""
                if pi and rc:
                    ends = f"start@{side_of(pi[0], rc[:4])} end@{side_of(pi[1], rc[:4])}"
                print(f"    {m['kind'][:22]:22s} {m['side']:5s} hid={m['hidden']} tpl={m['tpl']} z={m['z']} len={int(pi[2]) if pi else None} segs={pi[3] if pi else None} {ends} rect={[int(v) for v in rc[:4]] if rc else None}")
            if len(rs) >= 2:
                a, b = rs[0], rs[1]
                ca, cb = ((a[0]+a[2])/2, (a[1]+a[3])/2), ((b[0]+b[2])/2, (b[1]+b[3])/2)
                gx = max(a[0]-b[2], b[0]-a[2], 0); gy = max(a[1]-b[3], b[1]-a[3], 0)
                print(f"    partners: centre dist {int(math.dist(ca, cb))} gap {int(math.hypot(gx, gy))} (dx {int(abs(ca[0]-cb[0]))} dy {int(abs(ca[1]-cb[1]))})")
