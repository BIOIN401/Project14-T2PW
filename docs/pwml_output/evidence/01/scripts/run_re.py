"""Run the repo's own reaction_evidence.build_report on every run dir (pure stdlib, read-only). Output: re_reports.jsonl"""
import sys, json, os
sys.path.insert(0, "C:/t/oa/src")
from t2pw.app import reaction_evidence as re_
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
rows = [json.loads(l) for l in open("C:/t/ev-dedup/out/runs.jsonl")]
out = open("C:/t/ev-oa/funnel/out/re_reports.jsonl", "w", encoding="utf-8")
for r in rows:
    p = r["path"]
    d = os.path.dirname(p if os.path.isabs(p) else R + p)
    rec = {"path": p}
    try:
        rep = re_.build_report(re_.load_run_artifacts(d))
        rec["pwml_coverage"] = rep.get("pwml_coverage")
        rec["omitted"] = [{k: o.get(k) for k in ("kind", "name", "reason", "ki", "group", "counted", "merged_duplicate", "uncarried", "omission")} for o in (rep.get("omitted") or [])]
        rec["omitted_entities"] = rep.get("omitted_entities")
        rec["omitted_actors"] = rep.get("omitted_actors")
        rec["omitted_stage6_proteins"] = rep.get("omitted_stage6_proteins")
        rec["release_status"] = rep.get("release_status")
        rec["n_interactions_withheld"] = sum(1 for i in (rep.get("interactions") or []) if i.get("withheld_reason"))
        rec["withheld_reasons"] = [i.get("withheld_reason") for i in (rep.get("interactions") or []) if i.get("withheld_reason")]
        rec["n_interactions"] = len(rep.get("interactions") or [])
        rec["n_reactions_surviving"] = len(rep.get("reactions") or [])
    except Exception as ex:
        rec["error"] = repr(ex)[:200]
    out.write(json.dumps(rec, default=str) + "\n")
out.close()
