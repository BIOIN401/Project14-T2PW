import json, collections
for tag, d in (("p3", "../../../final_payload_p3/import"), ("p4first", "../../import"), ("d2", ".")):
    L = [l.strip().rsplit("/",1)[1] for l in open(d + "/list_on.txt")]
    rs = [json.loads(l) for l in open(d + "/on/results.jsonl", encoding="utf-8")]
    c = collections.Counter(); voc = collections.Counter()
    for p, r in zip(L, rs):
        for k in ("transports", "interactions"):
            c[k+"_cr"] += (r.get("created") or {}).get(k, 0); c[k+"_ex"] += (r.get("expected") or {}).get(k, 0)
        for k, v in r["errors"].items():
            b = k.rsplit("_",1)[0] if k.rsplit("_",1)[-1].isdigit() else k
            if b in ("subcellular_location","cell_type","tissue"): voc[b] += len(v)
    print(tag, dict(c), dict(voc), "success", sum(1 for r in rs if r["success"]))
    for i in ("10","76","150","68","3","4","7"):
        for p, r in zip(L, rs):
            if p == i + ".pwml":
                print("   ", i, "msgs", sum(len(v) for v in r["errors"].values()), "transports", (r.get("created") or {}).get("transports"), "/", (r.get("expected") or {}).get("transports"), "l=r", sum(1 for v in r["errors"].values() for m in v if "unique biological states" in m))
