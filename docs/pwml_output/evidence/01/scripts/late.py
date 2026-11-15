import json, collections
rows = [json.loads(l) for l in open("C:/t/ev-oa/funnel/out/runs_funnel.jsonl")]
RE = {json.loads(l)["path"]: json.loads(l) for l in open("C:/t/ev-oa/funnel/out/re_reports.jsonl")}
K = ["compounds", "proteins", "protein_complexes", "nucleic_acids", "reactions", "transports", "interactions"]
PW = {"protein_complexes": "protein-complexes", "nucleic_acids": "nucleic-acids"}
late = [r for r in rows if r["qok"] is True and (r["date"] >= "2026-10-19" or not r["date"])]
for lab, rs in (("late pathwhiz", [r for r in late if r["mode"] == "pathwhiz"]), ("late research", [r for r in late if r["mode"] == "research"])):
    print("\n##", lab, len(rs), "runs", len({r['paper'] for r in rs}), "papers", "PWML", sum(1 for r in rs if r.get("pwml")))
    for st in ("s1", "m", "f"):
        print(st, {k: sum(r[st][k] for r in rs) for k in K})
    pr = [r for r in rs if r.get("pwml")]
    if pr: print("pwml", {k: sum(r["pwml"].get(PW.get(k, k), 0) for r in pr) for k in K}, "F on PWML runs", {k: sum(r["f"][k] for r in pr) for k in K})
    qr = collections.Counter()
    for r in rs:
        for k, v in (r.get("q_reasons") or {}).items(): qr[k] += v
    print(qr.most_common(8))
    om = collections.Counter()
    for r in rs:
        for o in RE[r["path"]].get("omitted") or []: om[(o["kind"], o["reason"], o["ki"])] += 1
    print(om.most_common(10))
