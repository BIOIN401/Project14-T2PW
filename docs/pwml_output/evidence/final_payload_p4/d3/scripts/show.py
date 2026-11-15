import json, sys
idx = [int(l.strip().rsplit("/",1)[1][:-5]) for l in open("../../d2/import/list_on59.txt")]
for tag, p in (("p3", "../../../final_payload_p3/import/on/results.jsonl"), ("d2", "../../d2/import/on59_results.jsonl"), ("d3", "on59_results.jsonl")):
    R = dict(zip(idx, [json.loads(l) for l in open(p, encoding="utf-8")]))
    for i in map(int, sys.argv[1:]):
        r = R[i]; print(tag, i, "created", {k: v for k, v in (r.get("created") or {}).items() if k in ("transports","biological_states","reactions")}, "expected", {k: v for k, v in (r.get("expected") or {}).items() if k in ("transports","biological_states","reactions")})
        for k, v in r["errors"].items(): print("    ", k, v)
