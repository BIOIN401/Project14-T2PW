import json
for tag, d in (("p3", "../../../final_payload_p3/import"), ("p4first", "../../import"), ("d2", ".")):
    L = [l.strip() for l in open(d + "/list_on.txt")]
    rs = [json.loads(l) for l in open(d + "/on/results.jsonl", encoding="utf-8")]
    print(tag, "n", len(L), len(rs))
    for p, r in zip(L, rs):
        e = r["errors"]
        lr = sum(1 for k, v in e.items() for m in v if "unique biological states" in m)
        if e: print("  ", p.rsplit("/", 1)[1], "msgs", sum(len(v) for v in e.values()), "l=r", lr, "transports", (r.get("created") or {}).get("transports"), "/", (r.get("expected") or {}).get("transports"), "|", sorted({k.rsplit('_',1)[0] if k.rsplit('_',1)[-1].isdigit() else k for k in e})[:4])
