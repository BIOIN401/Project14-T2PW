import json,sys
p=json.load(open(sys.argv[1]+"/final_mapped.json",encoding="utf-8"))
ent=p["entities"]
for k,v in ent.items():
    print(k, len(v) if isinstance(v,list) else type(v))
c=ent["compounds"][0]; print({k:(v if k in("name","mapped_ids","class") else "...") for k,v in c.items()})
pr=ent["proteins"][0]; print({k:(v if k in("name","mapped_ids","class") else "...") for k,v in pr.items()})
print(json.dumps(p["element_locations"])[:600])
print([ (k, type(v).__name__) for k,v in p.items()])
