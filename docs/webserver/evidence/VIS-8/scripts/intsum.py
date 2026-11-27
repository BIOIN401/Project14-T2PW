"""usage: intsum.py PWML... -- per file summary of interaction edges: n, visible edge length median/max, gap median, single-segment share."""
import sys, math, statistics as st, json
sys.path.insert(0, "C:/t/vis-pw8-work/scripts")
from refint import survey


def rows(path):
    out, summ = survey(path)
    vis, gaps, types = [], [], []
    for iv in out:
        rs = [m["rect"] for m in iv["mem"] if m["rect"]]
        gap = None
        if len(rs) >= 2:
            a, b = rs[0], rs[1]
            gx = max(a[0] - b[2], b[0] - a[2], 0)
            gy = max(a[1] - b[3], b[1] - a[3], 0)
            gap = math.hypot(gx, gy)
            gaps.append(gap)
        for m in iv["mem"]:
            if m["hidden"] != "true" and m["pi"]:
                vis.append((m["pi"][2], m["pi"][3], m["tpl"]))
        types.append(iv["type"])
    return summ, vis, gaps, len(out)


def fmt(path):
    summ, vis, gaps, n = rows(path)
    if not vis:
        return f"{path.replace(chr(92), '/').rsplit('/', 1)[-1]}: no interaction edges (interactions={summ['interactions']}, bound={summ['bound']})"
    ls = [v[0] for v in vis]
    return (f"{path.replace(chr(92), '/').rsplit('/', 1)[-1]}: interactions={n} visible_edges={len(vis)} "
            f"len median={int(st.median(ls))} p90={int(sorted(ls)[int(0.9 * (len(ls) - 1))])} max={int(max(ls))} total={int(sum(ls))} "
            f"single_seg={sum(v[1] == 1 for v in vis)}/{len(vis)} gap median={int(st.median(gaps)) if gaps else None} "
            f">600={sum(l > 600 for l in ls)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(fmt(p))
