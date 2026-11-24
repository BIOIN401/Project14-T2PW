"""Convention statistics from db2_sample.py output (no DB access). Weighted by recency of the visualization:
updated 2023 or later = 1.0, 2019-2022 = 0.5, older = 0.25.
usage: db3_stats.py DB2.json OUT.json"""
import collections
import json
import math
import re
import statistics as stx
import sys

D = json.load(open(sys.argv[1]))
TPL = {int(t["id"]): t for t in D["templates"]}
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def weight(meta):
    y = int(str(meta.get("updated_at") or "2000")[:4])
    return 1.0 if y >= 2023 else 0.5 if y >= 2019 else 0.25


def pts(path, steps=6):
    """Polyline of an M/C/L path (cubic segments sampled)."""
    toks = re.findall(r"[MLCZmlcz]|-?\d+(?:\.\d+)?", path or "")
    out, cmd, nums = [], None, []

    def flush():
        if cmd in ("M", "L"):
            for i in range(0, len(nums) - 1, 2):
                out.append((nums[i], nums[i + 1]))
        elif cmd == "C" and out:
            for i in range(0, len(nums) - 5, 6):
                p0 = out[-1]
                c1, c2, p = (nums[i], nums[i + 1]), (nums[i + 2], nums[i + 3]), (nums[i + 4], nums[i + 5])
                for s in range(1, steps + 1):
                    t = s / steps
                    a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
                    out.append((a * p0[0] + b * c1[0] + c * c2[0] + d * p[0], a * p0[1] + b * c1[1] + c * c2[1] + d * p[1]))
    for tk in toks:
        if tk.isalpha():
            flush()
            cmd, nums = tk.upper(), []
        else:
            nums.append(float(tk))
    flush()
    return out


def bbox(p):
    xs, ys = [a for a, _ in p], [b for _, b in p]
    return min(xs), min(ys), max(xs), max(ys)


def inside(pt, r):
    return r[0] <= pt[0] <= r[2] and r[1] <= pt[1] <= r[3]


def dist_poly(pt, poly):
    best = 1e18
    for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        t = 0 if L == 0 else max(0, min(1, ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / L))
        best = min(best, math.hypot(pt[0] - x1 - t * dx, pt[1] - y1 - t * dy))
    return best


def wmed(vals):
    """Weighted median of (value, weight)."""
    vals = sorted(vals)
    if not vals:
        return None
    tot, acc = sum(w for _, w in vals), 0
    for v, w in vals:
        acc += w
        if acc >= tot / 2:
            return round(v, 1)


def wq(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    tot, acc = sum(w for _, w in vals), 0
    for v, w in vals:
        acc += w
        if acc >= tot * q:
            return round(v, 1)


tdesc = lambda t: (TPL.get(int(t)) or {}).get("description", "?") if t is not None else "?"
S = collections.defaultdict(list)
C = collections.defaultdict(collections.Counter)
years = collections.Counter()
for rec in D["pvs"]:
    w = weight(rec["meta"])
    years[str(rec["meta"].get("updated_at"))[:4]] += 1
    W, H = float(rec["meta"].get("width") or 0), float(rec["meta"].get("height") or 0)

    def size(t):
        tp = TPL.get(int(t)) if t is not None else None
        return (float(tp["width"] or 80), float(tp["height"] or 40)) if tp else (80.0, 40.0)

    cl = {c["id"]: c for c in rec["compounds"] if not c["hidden"]}
    pl = {p["id"]: p for p in rec["proteins"] if not p["hidden"]}

    def center(loc):
        w_, h_ = size(loc["t"])
        return float(loc["x"]) + w_ / 2, float(loc["y"]) + h_ / 2

    # ---- membranes
    mems = []
    for m in rec["membranes"]:
        p = pts(m["path"])
        if len(p) < 2:
            continue
        closed = math.hypot(p[0][0] - p[-1][0], p[0][1] - p[-1][1]) < 5
        mems.append({"t": m["t"], "poly": p, "bb": bbox(p), "closed": closed, "c": m["c"], "opt": m["options"]})
        C["membrane_template"][tdesc(m["t"])] += w
        C["membrane_closed_by_template"][(tdesc(m["t"]), closed)] += w
        if m["options"]:
            C["membrane_options"][str(m["options"])[:80]] += 1
    for m in mems:
        bb = m["bb"]
        if m["closed"]:
            S["mem_w:" + tdesc(m["t"])].append((bb[2] - bb[0], w))
            S["mem_h:" + tdesc(m["t"])].append((bb[3] - bb[1], w))
        # labels within 250 units of the membrane line: which words name it
        for lb in rec["labels"]:
            if dist_poly((float(lb["x"]), float(lb["y"])), m["poly"]) < 250:
                C["labels_near:" + tdesc(m["t"])][(lb["label_text"] or "").strip().lower()[:40]] += w
        # enclosing membrane (smallest closed one containing this bbox)
        enc = [o for o in mems if o is not m and o["closed"] and o["bb"][0] <= bb[0] and o["bb"][1] <= bb[1]
               and bb[2] <= o["bb"][2] and bb[3] <= o["bb"][3]]
        if enc:
            o = min(enc, key=lambda o: (o["bb"][2] - o["bb"][0]) * (o["bb"][3] - o["bb"][1]))
            C["nested_in"][(tdesc(m["t"]), tdesc(o["t"]))] += w
            if o["t"] == m["t"]:
                gap = min(bb[0] - o["bb"][0], bb[1] - o["bb"][1], o["bb"][2] - bb[2], o["bb"][3] - bb[3])
                S["double_gap:" + tdesc(m["t"])].append((gap, w))
                C["double_pairs"][tdesc(m["t"])] += w
        else:
            C["nested_in"][(tdesc(m["t"]), "(top)")] += w
    # ---- drawables
    for d in rec["drawables"]:
        C["drawables"][d["name"]] += w
    # ---- transports: transporter on a membrane? cargo on both sides?
    pcv_pl = collections.defaultdict(list)
    for r in rec["pcv_prot"]:
        pcv_pl[r["pcv"]].append(r["pl"])
    tr_cmp = collections.defaultdict(list)
    for r in rec["tr_cmp"]:
        tr_cmp[r["tv"]].append(r)
    tr_tr = collections.defaultdict(list)
    for r in rec["tr_tr"]:
        tr_tr[r["tv"]].append(r["pcv"])
    for tv, ms in tr_cmp.items():
        C["transport"]["n"] += w
        cargo = [(m["side"], center(cl[m["cl"]])) for m in ms if m["cl"] in cl]
        tps = [center(pl[p]) for pcv in tr_tr.get(tv, []) for p in pcv_pl.get(pcv, []) if p in pl]
        if tps:
            C["transport"]["with_transporter"] += w
            tc = tps[0]
            if mems:
                dm = min(dist_poly(tc, m["poly"]) for m in mems)
                S["transporter_to_membrane"].append((dm, w))
                near = min(mems, key=lambda m: dist_poly(tc, m["poly"]))
                C["transporter_membrane_template"][tdesc(near["t"])] += w
                if near["closed"]:
                    sides = {(s, inside(p, near["bb"])) for s, p in cargo}
                    ins = {i for _, i in sides}
                    C["transport"]["cargo_on_both_sides_of_its_membrane" if len(ins) == 2 else "cargo_same_side"] += w
            lefts = [p for s, p in cargo if s == "Left"]
            rights = [p for s, p in cargo if s == "Right"]
            if lefts and rights:
                a, b = lefts[0], rights[0]
                S["transport_cargo_dist"].append((math.hypot(a[0] - b[0], a[1] - b[1]), w))
                # is the transporter between the two cargos?
                C["transport"]["transporter_between_cargo"
                             if min(a[0], b[0]) - 80 <= tc[0] <= max(a[0], b[0]) + 80 and min(a[1], b[1]) - 80 <= tc[1] <= max(a[1], b[1]) + 80
                             else "transporter_off_axis"] += w
                C["transport_axis"]["vertical" if abs(a[0] - b[0]) < abs(a[1] - b[1]) else "horizontal"] += w
        else:
            C["transport"]["no_transporter"] += w
    # ---- reactions: spacing, alignment, cofactor stubs
    rx = collections.defaultdict(list)
    for r in rec["rxn_cmp"]:
        rx[r["rv"]].append(r)
    enz = collections.defaultdict(list)
    for r in rec["rxn_enz"]:
        enz[r["rv"]].append(r["pcv"])
    for rv, ms in rx.items():
        mem = [(m["side"], cl[m["cl"]]) for m in ms if m["cl"] in cl]
        big = [(s, l) for s, l in mem if size(l["t"])[0] >= 80]
        small = [(s, l) for s, l in mem if size(l["t"])[0] < 80]
        C["reaction_members"]["main"] += w * len(big)
        C["reaction_members"]["small_template"] += w * len(small)
        for _, l in mem:
            C["compound_template"][tdesc(l["t"])] += w
        L = [center(l) for s, l in big if s == "Left"]
        R = [center(l) for s, l in big if s == "Right"]
        if L and R:
            a, b = L[0], R[0]
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            S["main_step_dist"].append((math.hypot(dx, dy), w))
            C["main_step_axis"]["horizontal(<20)" if dy < 20 else "vertical(<20)" if dx < 20 else "diagonal"] += w
            ec = [center(pl[p]) for pcv in enz.get(rv, []) for p in pcv_pl.get(pcv, []) if p in pl]
            if ec:
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                S["enzyme_off_midpoint"].append((math.hypot(ec[0][0] - mid[0], ec[0][1] - mid[1]), w))
        if enz.get(rv):
            ec = [center(pl[p]) for pcv in enz[rv] for p in pcv_pl.get(pcv, []) if p in pl]
            if ec:
                for s, l in small:
                    c = center(l)
                    S["cofactor_to_enzyme"].append((math.hypot(c[0] - ec[0][0], c[1] - ec[0][1]), w))
    # ---- edges
    for e in rec["edges"]:
        if e["hidden"]:
            continue
        p = [(float(a), float(b)) for a, b in zip(*[iter(NUM.findall(e["path"] or ""))] * 2)]
        if len(p) < 2:
            continue
        a, b = p[0], p[-1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1:
            continue
        dev = max(abs((b[0] - a[0]) * (a[1] - q[1]) - (a[0] - q[0]) * (b[1] - a[1])) / L for q in p)
        C["edge_shape"]["straight(dev<5)" if dev < 5 else "curved"] += w
        S["edge_len"].append((L, w))
        C["edge_template"][tdesc(e["t"])] += w
    # ---- whole diagram
    allc = [center(l) for l in list(cl.values()) + list(pl.values())]
    if allc:
        bb = bbox(allc)
        if bb[3] > bb[1]:
            S["content_aspect"].append(((bb[2] - bb[0]) / max(1, bb[3] - bb[1]), w))
        S["canvas_w"].append((W, w))
        S["canvas_h"].append((H, w))
        S["nodes_per_mpx"].append((len(allc) / max(1, (W * H) / 1e6), w))
    names = collections.Counter(c["id"] for c in rec["compounds"])
    for c in rec["labels"]:
        C["label_template"][tdesc(c["t"])] += w
        if c["scale_x"]:
            C["label_scale"][str(c["scale_x"])] += w

out = {"n_pvs": len(D["pvs"]), "years": dict(years), "weighting": "updated>=2023: 1.0, 2019-2022: 0.5, older 0.25",
       "medians": {k: {"median": wmed(v), "p25": wq(v, .25), "p75": wq(v, .75), "n": len(v)} for k, v in sorted(S.items())},
       "counts": {k: {str(kk): round(vv, 2) for kk, vv in v.most_common(25)} for k, v in sorted(C.items())}}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1)[:12000])
