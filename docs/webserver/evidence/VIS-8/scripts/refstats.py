import sys, math, statistics as st
sys.path.insert(0, "C:/t/vis-pw8-work/scripts")
from refint import survey
R = "C:/t/vis-ref/super_pathway_visualizations/super_pathway_visualizations/pwml_markups/"
rows = []
tot = dict(n=0, pair=0, aligned=0, gap200=0, one_vis=0, both_vis=0, both_hid=0, bound=0, kinds={})
for f in ("PW126614", "PW126843", "PW127049", "PW127096"):
    out, summ = survey(R + f + ".pwml")
    for iv in out:
        tot["n"] += 1
        k = tuple(sorted(m["kind"].replace("interaction-", "").replace("-visualization", "")[:12] for m in iv["mem"]))
        tot["kinds"][k] = tot["kinds"].get(k, 0) + 1
        vis = [m for m in iv["mem"] if m["hidden"] != "true"]
        tot["one_vis"] += len(vis) == 1; tot["both_vis"] += len(vis) == 2; tot["both_hid"] += len(vis) == 0
        rs = [m["rect"] for m in iv["mem"] if m["rect"]]
        if len(rs) < 2:
            tot["bound"] += 1; continue
        a, b = rs[0], rs[1]
        ca, cb = ((a[0]+a[2])/2, (a[1]+a[3])/2), ((b[0]+b[2])/2, (b[1]+b[3])/2)
        gx = max(a[0]-b[2], b[0]-a[2], 0); gy = max(a[1]-b[3], b[1]-a[3], 0)
        gap = math.hypot(gx, gy)
        tot["pair"] += 1
        tot["aligned"] += abs(ca[0]-cb[0]) <= 40 or abs(ca[1]-cb[1]) <= 40
        tot["gap200"] += gap <= 200
        rows.append(gap)
print(tot)
print("gap median", st.median(rows), "p90", sorted(rows)[int(.9*(len(rows)-1))], "n", len(rows))
