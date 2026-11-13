import json
def L(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
idx = [int(l.strip().rsplit("/",1)[1][:-5]) for l in open("../../d2/import/list_on59.txt")]
S = {"p3": L("../../../final_payload_p3/import/on/results.jsonl"), "d2": L("../../d2/import/on59_results.jsonl"), "d3": L("on59_results.jsonl")}
S["p3"] = dict(zip(idx, S["p3"])) if len(S["p3"]) == 59 else None
for k in ("d2", "d3"): S[k] = dict(zip(idx, S[k]))
m = lambda r: sum(len(v) for v in r["errors"].values())
for i in idx:
    a, b, c = S["p3"][i], S["d2"][i], S["d3"][i]
    if len({m(a), m(b), m(c)}) > 1:
        print(i, "msgs p3/d2/d3", m(a), m(b), m(c), "transports cr/ex", [(x.get("created") or {}).get("transports") for x in (a, b, c)], (c.get("expected") or {}).get("transports"))
print("transports created", {k: sum((r.get("created") or {}).get("transports", 0) for r in v.values()) for k, v in S.items()}, "expected", sum((r.get("expected") or {}).get("transports", 0) for r in S["d3"].values()))
