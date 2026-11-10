import json,sys
for w,r in [("ws-E2E4","run-733acf326b98"),("ws-E2E3c","run-c4d6293e764b"),("ws-e2e","run-3b9c541c09aa")]:
    p=f"C:/t/{w}/tmp/extraction_diagnostics/{r}/stage1_ensemble_report.json"
    j=json.load(open(p))
    print("==",w,type(j),len(j))
    for i,x in enumerate(j[:6]):
        print(i,{k:(json.dumps(v)[:100]) for k,v in x.items()} if isinstance(x,dict) else str(x)[:200])
    print(open(p.replace(".json",".txt")).read()[:1500] if False else "")
