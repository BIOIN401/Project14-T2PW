import json,glob,re,statistics as st
base="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
files=glob.glob(base+"/.claude/worktrees/*/runs_papers/ORCH-*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")+glob.glob(base+"/runs_papers/ORCH-*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")
seen=set(); rows=[]
for f in files:
    ev=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
    if not ev or ev[0]["kind"]!="leg_begin": continue
    k=ev[0].get("run_dir","")
    if k in seen: continue
    seen.add(k)
    end=[e for e in ev if e["kind"]=="leg_end"]
    ma=[e for e in ev if e["kind"]=="model_attempt"]
    if not end or not ma: continue
    # last non-rag, non-chat? take last model_attempt of any kind
    last=ma[-1]["elapsed_seconds"]; tot=end[-1]["elapsed_seconds"]
    nm=re.search(r"/legs/([^/]+)/",f.replace("\\","/")).group(1)
    rows.append((nm,tot/60,(tot-last)/60,ma[-1]["stage"],end[-1].get("status")))
print("legs with leg_end:",len(rows))
t=[r[2] for r in rows]
print("untraced tail after last model call (min): median %.1f p90 %.1f max %.1f"%(st.median(t),sorted(t)[int(.9*len(t))],max(t)))
sour=[r for r in rows if r[0].startswith("sour")]
for r in sour: print(" ",r)
