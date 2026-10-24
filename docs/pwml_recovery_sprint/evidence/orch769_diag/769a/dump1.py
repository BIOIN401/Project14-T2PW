import json,sys
L=r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/enterobactin-strict/2026-10-23_2224/papers/PMC12096016/strict/"
for fn in ["final_mapped.json","merged_payload.json","stage1_payload.json"]:
    d=json.load(open(L+fn,encoding="utf-8"))
    print("=====",fn, list(d.keys())[:30])
    ents=d.get("entities") or {}
    print("entity keys", list(ents.keys()))
    for i,c in enumerate(ents.get("protein_complexes",[]) or []):
        print("PC",i,json.dumps(c,ensure_ascii=False)[:3000])
    for i,p in enumerate(ents.get("proteins",[]) or []):
        print("P",i,p.get("name"), json.dumps({k:v for k,v in p.items() if k not in ("name",)},ensure_ascii=False)[:600])
