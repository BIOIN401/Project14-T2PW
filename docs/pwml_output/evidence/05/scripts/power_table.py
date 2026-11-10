import math, csv
# MDE = (z_a/2 + z_b) * s_w * sqrt(2/(n*K)); 80% power, alpha .05 two-sided => 2.80
rows=[]
metrics=[("gold reaction recall, per-run SD 0.12 (abs points)",0.12,None),("entity count (cpd+prot+cplx), SD 2.82 on mean 13.3",2.82,13.29),("reaction count, SD 0.76 on mean 3.16",0.76,3.16),("all process count, SD 1.73 on mean 5.55",1.73,5.55),("interaction count, SD 1.67 on mean 2.39",1.67,2.39),("transport count, SD 0.83 on mean 0.96",0.83,0.96)]
designs=[(1,3),(1,10),(3,3),(5,3),(7,3),(7,5),(10,3),(10,5),(15,3),(20,3),(20,5)]
with open("C:/t/ev-oa/accvar/mde_table.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["metric","K_papers","n_runs_per_arm","MDE_abs","MDE_relative_to_mean"])
    for name,s,m in metrics:
        for K,n in designs:
            mde=2.8*s*math.sqrt(2/(n*K)); w.writerow([name,K,n,round(mde,3),round(mde/m,3) if m else ""])
print(open("C:/t/ev-oa/accvar/mde_table.csv").read()[:1800])
# n needed for target MDE
for name,s,m in metrics[:3]:
    for tgt in ((0.05,0.10) if m is None else (0.10,0.20)):
        t=tgt if m is None else tgt*m
        # K*n = 2*(2.8*s/t)^2
        print(name[:30],"target",tgt,"K*n runs per arm =",math.ceil(2*(2.8*s/t)**2))
