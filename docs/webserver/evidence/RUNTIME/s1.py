import json
for d in ["e2e2","e2e3c-run2","e2e4-run"]:
    o=f"C:/t/{d}/work/item-1/attempt-1/out/"
    for f in ["stage1_payload","merged_payload","extraction_boundary_report","cleaning_report","rag_admission_report","scope_specialization","coverage_summary"]:
        j=json.load(open(o+f+".json"))
        def ent(j):
            e=j.get("entities") or {}
            return {k:len(v) for k,v in e.items() if isinstance(v,list)}
        print(d,f,"keys",list(j.keys())[:25])
        if "entities" in j: print("   ents",ent(j),"reactions",len(j.get("reactions",[]) or []),"transports",len(j.get("transports",[]) or []),"interactions",len(j.get("interactions",[]) or []))
