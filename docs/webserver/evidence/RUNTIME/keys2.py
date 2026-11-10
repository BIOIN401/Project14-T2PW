import json
for d in ["e2e2","e2e3c-run2","e2e4-run"]:
    p=f"C:/t/{d}/work/item-1/attempt-1/"
    r=json.load(open(p+"result.json"))
    print("==",d,"seconds",json.load(open(p+"out/run_outcome.json"))["seconds"])
    print(json.dumps(r["attempt_log"],indent=0)[:3000])
    print("counts",r["counts"])
    print(json.dumps(json.load(open(p+"out/stage0_attempts.json")))[:900])
