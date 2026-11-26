"""Mine PathBank edge conventions from the reference PWMLs (read-only).
usage: mine_edges.py PWML_DIR OUT.json
Per reaction / transport member edge: which end touches the member box and which the enzyme (transporter) box, the side
of the box (and offset from its middle), tangent direction at both ends (H / V / other), control-point offsets, chord,
segments, arrow options, side (Left/Right), member template and size, currency name."""
import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

from lxml import etree as ET

P = ET.XMLParser(recover=True, huge_tree=True)
CUR = {"atp", "adp", "amp", "gtp", "gdp", "nadh", "nad", "nadph", "nadp", "fad", "fadh2", "water", "hydrogen ion", "carbon dioxide",
       "oxygen", "ammonia", "phosphate", "pyrophosphate", "coenzyme a", "adenosine triphosphate", "adenosine diphosphate",
       "adenosine monophosphate", "hydrogen", "nad+", "nadp+", "h2o", "co2", "pi", "ppi", "h+", "ammonium", "magnesium",
       "s-adenosylmethionine", "s-adenosylhomocysteine", "trna", "guanosine triphosphate", "guanosine diphosphate"}


def num(s):
    return [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", s or "")]


def rect(el):
    x, y = float(el.findtext("x") or 0), float(el.findtext("y") or 0)
    return (x, y, x + float(el.findtext("width") or 0), y + float(el.findtext("height") or 0))


def side_of(pt, r, tol=12):
    x, y = pt
    x0, y0, x1, y1 = r
    d = {"L": abs(x - x0), "R": abs(x - x1), "T": abs(y - y0), "B": abs(y - y1)}
    s = min(d, key=d.get)
    inside = x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol
    if not inside:
        return "out", min(d.values())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    off = (y - cy) / max(1, y1 - y0) if s in "LR" else (x - cx) / max(1, x1 - x0)
    if abs(x - cx) < 6 and abs(y - cy) < 6:
        return "C", 0
    return s, round(off, 2)


def dirn(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return "0"
    ang = math.degrees(math.atan2(dy, dx))
    if abs(dy) <= abs(dx) * math.tan(math.radians(8)):
        return "H"
    if abs(dx) <= abs(dy) * math.tan(math.radians(8)):
        return "V"
    return "D"


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


rows = []
for f in sorted(Path(sys.argv[1]).glob("*.pwml")):
    t = ET.parse(str(f), P).getroot()
    comp_name = {c.findtext("id"): (c.findtext("name") or "") for c in t.iter("compound")}
    loc = {}
    for sec, kind in (("compound-location", "c"), ("protein-location", "p"), ("nucleic-acid-location", "n"),
                      ("element-collection-location", "e")):
        for el in t.iter(sec):
            nm = comp_name.get(el.findtext("compound-id"), "") if kind == "c" else ""
            loc[(kind, el.findtext("id"))] = {"r": rect(el), "tpl": el.findtext("visualization-template-id"), "name": nm}
    pcv = {}
    for el in t.iter("protein-complex-visualization"):
        rs = [loc[("p", m.findtext("protein-location-id"))]["r"] for m in el.iter("protein-complex-protein-visualization")
              if ("p", m.findtext("protein-location-id")) in loc]
        if rs:
            pcv[el.findtext("id")] = (min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs))
    edges = {}
    for e in t.iter("edge"):
        o = {}
        for c in e:
            if isinstance(c.tag, str) and c.tag.startswith("option:"):
                o[c.tag[7:]] = (c.text or "").strip()
        edges[e.findtext("id")] = {"path": e.findtext("path") or "", "tpl": e.findtext("visualization-template-id"),
                                   "hidden": e.findtext("hidden"), "opts": o}
    for proc, mem_tags, hub_tags in (
            ("reaction", (("reaction-compound-visualization", "compound-location-id", "c"),
                          ("reaction-nucleic-acid-visualization", "nucleic-acid-location-id", "n"),
                          ("reaction-element-collection-visualization", "element-collection-location-id", "e")),
             ("reaction-enzyme-visualization",)),
            ("transport", (("transport-compound-visualization", "compound-location-id", "c"),),
             ("transport-transporter-visualization",))):
        for pv in t.iter(f"{proc}-visualization"):
            hubs = []
            for ht in hub_tags:
                for h in pv.iter(ht):
                    k = h.findtext("protein-complex-visualization-id")
                    if k in pcv:
                        hubs.append(pcv[k])
                    elif ("p", h.findtext("protein-location-id")) in loc:
                        hubs.append(loc[("p", h.findtext("protein-location-id"))]["r"])
            members = []
            for tag, fld, kind in mem_tags:
                for m in pv.iter(tag):
                    members.append((m, loc.get((kind, m.findtext(fld)))))
            nsub = sum(1 for m, l in members if m.findtext("side") == "Left")
            for m, l in members:
                e = edges.get(m.findtext("edge-id"))
                if e is None or l is None:
                    continue
                n = num(e["path"])
                if len(n) < 4:
                    continue
                pts = list(zip(n[0::2], n[1::2]))
                cmds = re.findall(r"[A-Za-z]", e["path"])
                s, end = pts[0], pts[-1]
                cp1 = pts[1] if len(pts) > 1 else s
                cp2 = pts[-2] if len(pts) > 2 else s
                r = l["r"]
                ds, de = side_of(s, r), side_of(end, r)
                start_at_member = ds[0] != "out" and (de[0] == "out" or dist(s, ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)) <= dist(end, ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)))
                mp, op = (s, end) if start_at_member else (end, s)
                mcp, ocp = (cp1, cp2) if start_at_member else (cp2, cp1)
                hub_side = None
                if hubs:
                    hr = min(hubs, key=lambda h: dist(op, ((h[0] + h[2]) / 2, (h[1] + h[3]) / 2)))
                    hub_side = side_of(op, hr)
                nm = (l["name"] or "").strip().lower()
                rows.append({
                    "file": f.stem, "rx": f.stem + ":" + proc + ":" + (pv.findtext("id") or ""), "hub_pt": [round(op[0]), round(op[1])], "mem_pt": [round(mp[0]), round(mp[1])], "proc": proc, "side": m.findtext("side"), "edge_tpl": e["tpl"], "hidden": e["hidden"],
                    "segments": sum(1 for c in cmds if c.upper() in "CLQ"), "cmds": "".join(sorted(set(c.upper() for c in cmds))),
                    "start_at_member": start_at_member, "member_side": side_of(mp, r)[0], "member_off": side_of(mp, r)[1],
                    "hub_side": hub_side[0] if hub_side else None, "hub_off": hub_side[1] if hub_side else None,
                    "member_dir": dirn(mp, mcp), "hub_dir": dirn(ocp, op), "member_cp": round(dist(mp, mcp)),
                    "hub_cp": round(dist(op, ocp)), "chord": round(dist(s, end)), "dx": round(op[0] - mp[0]), "dy": round(op[1] - mp[1]),
                    "w": round(r[2] - r[0]), "h": round(r[3] - r[1]), "tpl": l["tpl"], "name": l["name"],
                    "currency": nm in CUR or nm.split(" ")[0] in CUR,
                    "opts": e["opts"], "n_sub": nsub, "path": e["path"]})
out = {"n": len(rows), "rows": rows}


def summary(rs, label):
    c = collections.Counter
    res = {"n": len(rs)}
    for k in ("cmds", "segments", "member_side", "hub_side", "member_dir", "hub_dir", "start_at_member"):
        res[k] = c(str(r[k]) for r in rs).most_common(8)
    for k in ("member_cp", "hub_cp", "chord"):
        v = [r[k] for r in rs if r["segments"] == 1]
        if v:
            res[k + "_median_1seg"] = st.median(v)
            res[k + "_q"] = [round(q) for q in st.quantiles(v, n=4)] if len(v) > 3 else v
    v = [r["member_cp"] / r["chord"] for r in rs if r["segments"] == 1 and r["chord"]]
    if v:
        res["member_cp_over_chord"] = round(st.median(v), 2)
    v = [r["hub_cp"] / r["chord"] for r in rs if r["segments"] == 1 and r["chord"]]
    if v:
        res["hub_cp_over_chord"] = round(st.median(v), 2)
    res["arrows"] = c(json.dumps({k: v for k, v in r["opts"].items() if not k.endswith("_path")}, sort_keys=True) for r in rs).most_common(6)
    res["opt_keysets"] = c(",".join(sorted(r["opts"])) for r in rs).most_common(4)
    res["tpl"] = c(r["tpl"] for r in rs).most_common(8)
    res["size"] = c(f'{r["w"]}x{r["h"]}' for r in rs).most_common(6)
    return res


groups = collections.defaultdict(list)
for r in rows:
    groups[(r["proc"], r["side"], "cur" if r["currency"] else "main")].append(r)
out["summary"] = {"/".join(map(str, k)): summary(v, k) for k, v in sorted(groups.items())}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["summary"], indent=1))
