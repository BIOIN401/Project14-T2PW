import json,glob
base="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
files=glob.glob(base+"/.claude/worktrees/orch7*run/runs_papers/ORCH-*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")+glob.glob(base+"/.claude/worktrees/orch8*run/runs_papers/ORCH-*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")+glob.glob(base+"/runs_papers/ORCH-759*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")
seen=set(); calls=[]; legs=0
for f in files:
    ev=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
    if not ev or ev[0]["kind"]!="leg_begin": continue
    k=ev[0].get("run_dir","")
    if k in seen: continue
    seen.add(k); legs+=1; prev=0
    for e in ev:
        if e["kind"]=="model_attempt": calls.append((e["stage"],(e["elapsed_seconds"]-prev)/60,e["status"]))
        prev=e["elapsed_seconds"]
print("legs",legs,"calls",len(calls))
for cap in (8,10,12,15,20):
    bad=[c for c in calls if c[2]!="ok" and c[1]>cap]; ok=[c for c in calls if c[2]=="ok" and c[1]>cap]
    print(f"cap {cap:2d} min: saves {sum(c[1]-cap for c in bad):6.1f} min on {len(bad)} failed calls; would also cut {len(ok)} successful calls ({sum(c[1]-cap for c in ok):.1f} min) {[ (c[0],round(c[1],1)) for c in ok][:6]}")
print("per leg avg saving at cap 12 (failed only):", round(sum(c[1]-12 for c in calls if c[2]!='ok' and c[1]>12)/legs,1))
