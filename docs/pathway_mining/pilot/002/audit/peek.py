import json, sys
r = json.load(open("C:/t/mining-pilot-runs2/audit/pilot_002_audit.json", encoding="utf-8"))
for k in sys.argv[1:]:
    v = r
    for part in k.split("."):
        v = v[int(part)] if isinstance(v, list) else v[part]
    print("=====", k)
    print(json.dumps(v, indent=1, ensure_ascii=False)[:12000])
