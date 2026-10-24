import json
L=r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/enterobactin-strict/2026-10-23_2224/papers/PMC12096016/strict/"
d=json.load(open(L+"final_mapped.json",encoding="utf-8"))
P=d["entities"]["proteins"]
for i in (8,):
    print("P",i,json.dumps(P[i],ensure_ascii=False,indent=1)[:300])
pc=d["entities"]["protein_complexes"][0]
print(json.dumps({k:v for k,v in pc.items() if k!="provenance_lineage"},ensure_ascii=False,indent=1))
print("lineage stages:",[ (x["stage"],x["origin"]) for x in pc.get("provenance_lineage",[])])
# where are complex names referenced
s=json.dumps(d,ensure_ascii=False)
import re
for name in ["enterobactin synthase complex","enterobactin synthase"]:
    print(name, s.count('"'+name+'"'))
for proc_k,proc in d["processes"].items():
    for j,r in enumerate(proc or []):
        rs=json.dumps(r,ensure_ascii=False)
        if "enterobactin synthase" in rs:
            print(proc_k,j,rs[:1500])
