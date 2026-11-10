import json,sys
for d in ["e2e2","e2e3c-run2","e2e4-run"]:
    p=f"C:/t/{d}/work/item-1/attempt-1/"
    r=json.load(open(p+"result.json"))
    print("==",d,[k for k in r if k not in("warnings","report_paths")])
    for k in r:
        if k in ("stage_timings","timings","elapsed","duration","llm"): print(k,json.dumps(r[k])[:1500])
    o=json.load(open(p+"out/run_outcome.json"))
    print("outcome keys",list(o.keys()))
    for k,v in o.items():
        if any(t in k.lower() for t in ("time","stage","llm","elapsed","dur","call","usage","token")): print(k,json.dumps(v)[:2500])
