import json,glob,collections,statistics as st,re
base="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
files=glob.glob(base+"/.claude/worktrees/orch7*run/runs_papers/ORCH-*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")+glob.glob(base+"/.claude/worktrees/orch8*run/runs_papers/ORCH-*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")+glob.glob(base+"/runs_papers/ORCH-759*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")
seen=set(); legs=0
stg=collections.defaultdict(lambda:{"ok":[], "bad":[]})
legbad=[]
for f in files:
    ev=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
    if not ev or ev[0]["kind"]!="leg_begin": continue
    k=ev[0].get("run_dir","")
    if k in seen: continue
    seen.add(k); legs+=1; prev=0; badmin=0; tot=0
    for e in ev:
        if e["kind"]=="model_attempt":
            d=(e["elapsed_seconds"]-prev)/60
            stg[e["stage"]]["ok" if e["status"]=="ok" else "bad"].append(d)
            if e["status"]!="ok": badmin+=d
        prev=e["elapsed_seconds"]
    legbad.append((badmin,prev/60))
print("legs",legs)
for s,v in sorted(stg.items(), key=lambda kv:-(sum(kv[1]["ok"])+sum(kv[1]["bad"]))):
    o,b=v["ok"],v["bad"]
    f=lambda a,q: sorted(a)[min(len(a)-1,int(q*len(a)))] if a else float('nan')
    print(f"{s:22s} ok n={len(o):4d} p50={f(o,.5):5.2f} p90={f(o,.9):5.2f} max={max(o) if o else 0:5.2f} | bad n={len(b):3d} p50={f(b,.5):5.2f} max={max(b) if b else 0:5.2f} sum={sum(b):6.1f}m")
tb=sum(b for b,_ in legbad); tt=sum(t for _,t in legbad)
print(f"time inside non-ok calls: {tb:.0f} of {tt:.0f} traced min = {tb/tt:.0%}; legs with >=10 min in bad calls: {sum(1 for b,_ in legbad if b>=10)}/{legs}")
