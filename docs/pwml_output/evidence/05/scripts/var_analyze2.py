import json, itertools, statistics as st, collections, csv, math
runs = json.load(open("C:/t/ev-oa/accvar/var_runs.json"))
KINDS=["compounds","proteins","protein_complexes","reactions","transports","interactions"]
def jac(a,b):
    u=a|b; return 1.0 if not u else len(a&b)/len(u)
def sd(x): return st.stdev(x) if len(x)>1 else 0.0
groups=collections.defaultdict(list)
for r in runs: groups[(r["paper"],r["mode"])].append(r)
rows=[]
for (p,mode),rs in groups.items():
    if len(rs)<4: continue
    # same-era restriction: largest era subgroup
    ec=collections.Counter(r["era"] for r in rs); e0,c0=ec.most_common(1)[0]
    for scope,sub in (("all",rs),("sameEra",[r for r in rs if r["era"]==e0])):
        if len(sub)<4: continue
        for stage in ("S","F","C"):
            sub2=[r for r in sub if r.get(stage)]
            for k in KINDS:
                sets=[set(r[stage][k]) for r in sub2]; n=len(sets)
                if n<4 or max(len(s) for s in sets)==0: continue
                cnt=[len(s) for s in sets]; c=collections.Counter(x for s in sets for x in s)
                u=len(c); st80=sum(1 for v in c.values() if v>=0.8*n)
                js=[jac(a,b) for a,b in itertools.combinations(sets,2)]
                mean=st.mean(cnt)
                # share of a typical run's items that are stable (>=80%)
                shares=[sum(1 for x in s if c[x]>=0.8*n)/len(s) for s in sets if s]
                rows.append(dict(paper=p,mode=mode,scope=scope,era=e0 if scope=="sameEra" else "mixed",stage=stage,kind=k,n=n,mean=round(mean,2),sd=round(sd(cnt),2),
                    cv=round(sd(cnt)/mean,3),min=min(cnt),max=max(cnt),jaccard=round(st.mean(js),3),union=u,stable80=st80,
                    run_items_stable_share=round(st.mean(shares),3),singleton_share=round(sum(1 for v in c.values() if v==1)/u,3)))
with open("C:/t/ev-oa/accvar/var_summary_by_paper.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
# aggregate: median over papers by stage x kind (sameEra scope)
print("papers x modes used:", sorted({(r['paper'],r['mode']) for r in rows}))
agg=[]
for scope in ("all","sameEra"):
  for stage in ("S","F","C"):
    for k in KINDS:
        sel=[r for r in rows if r["scope"]==scope and r["stage"]==stage and r["kind"]==k]
        if not sel: continue
        agg.append(dict(scope=scope,stage=stage,kind=k,groups=len(sel),med_jaccard=round(st.median(r["jaccard"] for r in sel),3),
          med_cv=round(st.median(r["cv"] for r in sel),3),med_stable_run_share=round(st.median(r["run_items_stable_share"] for r in sel),3),
          med_singleton_share=round(st.median(r["singleton_share"] for r in sel),3)))
with open("C:/t/ev-oa/accvar/var_agg.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(agg[0])); w.writeheader(); [w.writerow(r) for r in agg]
for a in agg: print(a)
