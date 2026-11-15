import json
for tag, d in (("p3", "../../final_payload_p3/import"), ("p4", ".")):
    L = [l.strip() for l in open(d + "/list_controls.txt")]
    for p, r in zip(L, [json.loads(l) for l in open(d + "/controls/results.jsonl", encoding="utf-8")]):
        print(tag, p.rsplit("/", 1)[1], "success", r["success"], "msgs", sum(len(v) for v in r["errors"].values()))
