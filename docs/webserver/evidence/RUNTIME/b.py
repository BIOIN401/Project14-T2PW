import json
for d in ["e2e2","e2e3c-run2","e2e4-run"]:
    o=f"C:/t/{d}/work/item-1/attempt-1/out/"
    j=json.load(open(o+"extraction_boundary_report.json"))
    print("==",d,j["boundary_count"],j["outcomes"])
    for b in j["boundaries"]:
        print({k:(v if not isinstance(v,(dict,list,str)) or len(str(v))<90 else str(v)[:90]) for k,v in b.items() if k not in("raw_preview","context_counts","detail")})
