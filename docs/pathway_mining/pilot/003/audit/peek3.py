import json, sys
r = json.load(open("C:/t/mining-pilot-runs3/audit/pilot_003_audit.json", encoding="utf-8"))
for k in sys.argv[1:]:
    v = r
    for part in k.split("."):
        v = v[int(part)] if isinstance(v, list) else v[part]
    print("=====", k)
    print(json.dumps(v, indent=1, ensure_ascii=False)[:9000])
