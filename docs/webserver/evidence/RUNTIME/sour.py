import json,glob,collections,re,os
base="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
pats=[base+"/.claude/worktrees/*/runs_papers/ORCH-*/legs/sour_taste*/*/papers/*/strict/LEG_TRACE.jsonl",
      base+"/runs_papers/ORCH-*/legs/sour_taste*/*/papers/*/strict/LEG_TRACE.jsonl"]
files=[]
for p in pats: files+=glob.glob(p)
seen=set()
for f in sorted(files):
    f=f.replace("\\","/")
    ev=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
    key=(ev[0].get("run_dir",""))
    if key in seen: continue
    seen.add(key)
    prev=0;stg=collections.defaultdict(lambda:[0,0.0,0.0,0])
    for e in ev:
        if e["kind"]=="model_attempt":
            d=e["elapsed_seconds"]-prev; s=stg[e["stage"]]; s[0]+=1; s[1]+=d; s[2]=max(s[2],d)
            if e["status"]!="ok": s[3]+=1
        prev=e["elapsed_seconds"]
    orch=re.search(r"ORCH-\d+p?",f).group(0); stamp=re.search(r"/(\d{4}-\d\d-\d\d_\d{4})/",f).group(1)
    nm=re.search(r"/legs/([^/]+)/",f).group(1)
    n=sum(1 for e in ev if e["kind"]=="model_attempt")
    print(f"{orch} {nm} {stamp} total={prev/60:5.1f}m calls={n}  "+"  ".join(f"{k}:{v[0]}c/{v[1]/60:.1f}m/max{v[2]/60:.1f}/bad{v[3]}" for k,v in sorted(stg.items(),key=lambda kv:-kv[1][1])[:6]))
