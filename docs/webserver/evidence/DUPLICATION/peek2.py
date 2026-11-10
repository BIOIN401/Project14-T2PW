import json,sys
d=sys.argv[1]
p=json.load(open(d+"/final_mapped.json",encoding="utf-8"))
for e in p["processes"]["transports"]:
    if e["name"].startswith("H+ influx into"):
        e=dict(e); e.pop("provenance_lineage",None); print(json.dumps(e,indent=1)[:2500])
print([ (b.get("name") if isinstance(b,dict) else b) for b in p["biological_states"]])
print(json.dumps(p["biological_states"][:8])[:1500])
for e in p["entities"]["compounds"]:
    if e["name"]=="H+": 
        e=dict(e); e.pop("provenance_lineage",None); print(json.dumps(e)[:800])
print(json.dumps(p["element_locations"])[:1500] if not isinstance(p["element_locations"],dict) else list(p["element_locations"].keys()))
ir=json.load(open(d+"/pwml_ir.json",encoding="utf-8"))
print(list(ir.keys()))
