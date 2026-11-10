import json, statistics as st, collections, math, csv
g=[json.loads(l) for l in open("C:/t/oa/docs/pwml_output/evidence/01/gold_recall.jsonl")]
g=[x for x in g if x["qok"]]
grp=collections.defaultdict(list)
for x in g: grp[(x["paper"],x["mode"])].append(x)
print("gold runs qok",len(g))
pooled=[];rows=[]
for (p,m),v in sorted(grp.items()):
    for st_ in ("s1","f"):
        r=[x[st_]/x["n_gold"] for x in v]
        if len(r)>=2: rows.append((p,m,st_,len(r),x["n_gold"],round(st.mean(r),3),round(st.stdev(r),3),min(r),max(r)))
for r in rows: print(r)
# pooled within SD of per-run recall (df-weighted), stage f and s1
for st_ in ("s1","f"):
    ss=0;df=0
    for (p,m),v in grp.items():
        if len(v)>=2:
            r=[x[st_]/x["n_gold"] for x in v]; ss+=sum((a-st.mean(r))**2 for a in r); df+=len(r)-1
    sw=math.sqrt(ss/df); print(st_,"pooled within SD recall",round(sw,3),"df",df)
# pooled within-group CV of total counts from var_runs
runs=json.load(open("C:/t/ev-oa/accvar/var_runs.json"))
G=collections.defaultdict(list)
for r in runs: G[(r["paper"],r["mode"],r["era"])].append(r)
def tot(r,keys,stage="F"): return sum(len(r[stage][k]) for k in keys)
for name,keys in (("entities(cpd+prot+cplx)",["compounds","proteins","protein_complexes"]),("reactions",["reactions"]),("transports",["transports"]),("interactions",["interactions"]),("all processes",["reactions","transports","interactions"])):
    ss=0;df=0;cvs=[];ms=[]
    for k,v in G.items():
        if len(v)<4: continue
        x=[tot(r,keys) for r in v]; m=st.mean(x)
        if m==0: continue
        ss+=sum((a-m)**2 for a in x); df+=len(x)-1; cvs.append(st.stdev(x)/m); ms.append(m)
    sw=math.sqrt(ss/df); cv=sw/st.mean(ms)
    print(name,"groups",len(cvs),"pooled within SD",round(sw,2),"mean count",round(st.mean(ms),2),"pooled CV",round(cv,3),"median CV",round(st.median(cvs),3))
    # MDE relative for n runs/arm, K papers, paired(same papers) : SE=sw*sqrt(2/(n*K)) ; MDE=2.8*SE ; rel to mean
    for K,n in ((1,3),(1,10),(5,3),(7,3),(7,5),(10,3),(10,5),(20,3)):
        mde=2.8*sw*math.sqrt(2/(n*K))
        print("   K",K,"n/arm",n,"MDE rel",round(mde/st.mean(ms),3))
