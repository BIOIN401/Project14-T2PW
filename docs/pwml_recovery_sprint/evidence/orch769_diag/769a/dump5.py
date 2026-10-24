import json,re
L=r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/enterobactin-strict/2026-10-23_2224/papers/PMC12096016/strict/"
d=json.load(open(L+"gap_iteration_summary.json",encoding="utf-8"))
def walk(o,path=""):
    if isinstance(o,dict):
        for k,v in o.items(): yield from walk(v,path+"/"+str(k))
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from walk(v,path+"/"+str(i))
    else:
        yield path,o
print(list(d.keys()) if isinstance(d,dict) else type(d))
for p,v in walk(d):
    s=str(v)
    if "protein_complexes/0" in s or "member_identity" in s or "stoichiometry" in p or "stoichiometry" in s or ("enterobactin synthase" in s and "complex" not in p):
        print(p,"=",s[:300])
