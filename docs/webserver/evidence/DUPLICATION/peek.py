import json,sys
d=sys.argv[1]
for f in ["stage1_payload.json","merged_payload.json","final_mapped.json"]:
    p=json.load(open(d+"/"+f,encoding="utf-8"))
    print("==",f)
    ents=p["entities"]
    print(" entities type", type(ents).__name__, list(ents.keys()) if isinstance(ents,dict) else len(ents))
    procs=p["processes"]
    print(" processes type", type(procs).__name__, list(procs.keys()) if isinstance(procs,dict) else len(procs))
    if isinstance(ents,dict):
        for k,v in ents.items():
            for e in v:
                n=e.get("name","") if isinstance(e,dict) else str(e)
                if n in("H+",) or "proton" in n.lower() or "OTOP1" in n or n.lower().startswith("hydrogen"):
                    print("  ENT",k,json.dumps(e)[:350])
    if isinstance(procs,dict):
        for k,v in procs.items():
            for e in v:
                if "OTOP1" in json.dumps(e):
                    print("  PROC",k,json.dumps(e)[:1000])
