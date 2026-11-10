import json
for l in open("C:/t/ev-oa/funnel/out/re_reports.jsonl"):
    x = json.loads(l)
    if "e2e5" in x["path"] or "e2e4" in x["path"]:
        print(x["path"][:20], "stage6:", x.get("omitted_stage6_proteins"), "| entities:", [(o["bucket"], o["name"], o["reason"]) for o in x.get("omitted_entities") or []], "| actors:", x.get("omitted_actors"))
d = "C:/t/e2e5-run/work/item-1/attempt-1/out/"
for fn in ("gap_iteration_summary.json", "audit_iteration_summary.json", "cleaning_report.json"):
    try:
        t = open(d + fn, encoding="utf-8").read()
        print(fn, len(t), t[:600].replace("\n", " "))
    except Exception as e:
        print(fn, e)
g = json.load(open(d + "final_stage3_gate_report.json", encoding="utf-8"))
print({k: v for k, v in (g.get("normalization_stats") or {}).items() if v})
ig = json.load(open(d + "initial_stage3_gate_report.json", encoding="utf-8"))
print({k: v for k, v in (ig.get("normalization_stats") or {}).items() if v})
