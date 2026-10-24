import json
L=r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/enterobactin-strict/2026-10-23_2224/papers/PMC12096016/strict/"
d=json.load(open(L+"final_mapped.json",encoding="utf-8"))
r=d["processes"]["reactions"][4]
print(json.dumps({k:v for k,v in r.items() if k!="provenance_lineage"},ensure_ascii=False,indent=1)[:5000])
p=d["entities"]["proteins"][8]
print("synonyms:",p.get("synonyms"),"aliases:",p.get("aliases"),"evidence:",p.get("evidence"),"source_refs:",p.get("source_refs"), "key:",p.get("key"))
print("EntE row mapped:",d["entities"]["proteins"][3].get("mapped_ids"), d["entities"]["proteins"][3].get("synonyms"))
print("EntF row mapped:",d["entities"]["proteins"][4].get("mapped_ids"), d["entities"]["proteins"][4].get("synonyms"))
for k,v in d["element_locations"].items():
    for row in v or []:
        if "enterobactin synthase" in json.dumps(row,ensure_ascii=False):
            print("LOC",k,json.dumps(row,ensure_ascii=False)[:300])
for k in ("interactions","transports"):
    for i,row in enumerate(d["processes"].get(k) or []):
        if "enterobactin synthase" in json.dumps(row,ensure_ascii=False):
            print(k,i,json.dumps(row,ensure_ascii=False)[:300])
print("reaction_order", d.get("reaction_order"))
print("protein_export_policy", json.dumps(d.get("protein_export_policy"),ensure_ascii=False)[:600])
