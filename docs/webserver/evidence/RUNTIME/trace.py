import json,glob,os,collections,re
root="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers"
rows=[]
kinds=collections.Counter()
for f in sorted(glob.glob(root+"/*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")):
    parts=f.replace("\\","/").split("/")
    orch=parts[parts.index("runs_papers")+1]; leg=parts[parts.index("legs")+1]; stamp=parts[parts.index("legs")+2]
    ev=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
    for e in ev: kinds[e["kind"]]+=1
    end=[e for e in ev if e["kind"]=="leg_end"]
    last=ev[-1]["elapsed_seconds"]
    prev=0.0
    stg=collections.defaultdict(lambda:[0,0.0,0.0,0,0]) # n, sum, max, nonstop, errors
    for e in ev:
        if e["kind"]=="model_attempt":
            d=e["elapsed_seconds"]-prev
            s=stg[e["stage"]]
            s[0]+=1; s[1]+=d; s[2]=max(s[2],d)
            if e.get("finish_reason") not in ("stop",""): s[3]+=1
            if e.get("status")!="ok": s[4]+=1
        prev=e["elapsed_seconds"]
    n=sum(1 for e in ev if e["kind"]=="model_attempt")
    rows.append((orch,leg,stamp,last,n,dict(stg)))
print(kinds)
for r in rows:
    print(f"{r[0]:9s} {r[1]:28s} {r[2]} total={r[3]/60:6.1f}min calls={r[4]}")
    for k,v in sorted(r[5].items(), key=lambda kv:-kv[1][1])[:6]:
        print(f"      {k:32s} n={v[0]:3d} sum={v[1]/60:6.1f}m max={v[2]/60:5.1f}m nonstop={v[3]} nonok={v[4]}")
