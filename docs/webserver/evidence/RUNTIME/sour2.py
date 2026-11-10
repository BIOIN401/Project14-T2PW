import json,glob,re
base="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
files=glob.glob(base+"/.claude/worktrees/*/runs_papers/ORCH-*/legs/sour_taste_v2*/*/papers/*/strict/LEG_TRACE.jsonl")
seen=set()
for f in sorted(files):
    ev=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
    k=ev[0].get("run_dir","")
    if k in seen: continue
    seen.add(k); prev=0
    orch=re.search(r"ORCH-\d+",f).group(0)
    for e in ev:
        if e["kind"]=="model_attempt":
            d=e["elapsed_seconds"]-prev
            if d>240 or e["status"]!="ok":
                print(f"{orch} {e['stage']:20s} {d/60:5.1f}m status={e['status']} fr={e.get('finish_reason')} chars={e.get('content_chars')} att={e['attempt']} {(e.get('reason') or '')[:80]!r}")
        prev=e["elapsed_seconds"]
