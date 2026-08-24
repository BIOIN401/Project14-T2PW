import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
from t2pw.pipeline.release_status import RELEASE_READY, classify_release_status  # noqa: E402
CHECK = "no_real_id_or_name_conflict"
RUNS = ROOT / "runs_verify"
A = ("quarantine_report.json", "coverage_summary.json", "final_mapped.json")
legs = []
for run in sorted(RUNS.iterdir()):
    p = run / "papers"
    if not p.is_dir():
        continue
    for paper in sorted(p.iterdir()):
        if not paper.is_dir():
            continue
        for leg in sorted(paper.iterdir()):
            if leg.is_dir():
                legs.append((f"{run.name}/papers/{paper.name}/{leg.name}", leg,
                             {a: (leg / a).is_file() for a in A}))
def n(pred):
    return sum(1 for _, _, h in legs if pred(h))
print("TOTAL leg dirs                       :", len(legs))
print("all three artifacts                  :", n(lambda h: all(h.values())))
print("quarantine_report.json present       :", n(lambda h: h[A[0]]))
print("qr + coverage (final_mapped NOT req) :", n(lambda h: h[A[0]] and h[A[1]]))
print("qr present but coverage MISSING      :", n(lambda h: h[A[0]] and not h[A[1]]))
print("qr present but final_mapped MISSING  :", n(lambda h: h[A[0]] and not h[A[2]]))
print("final_mapped present                 :", n(lambda h: h[A[2]]))
print("final_mapped but NO qr               :", n(lambda h: h[A[2]] and not h[A[0]]))
print()
# the two diagnostic_only carriers, forced green
for name, leg, h in legs:
    if not h[A[0]]:
        continue
    rel = (json.loads((leg / A[0]).read_text(encoding="utf-8")).get("release") or {})
    if CHECK not in (rel.get("semantic_failed_checks") or ()):
        continue
    cov = json.loads((leg / A[1]).read_text(encoding="utf-8")) if h[A[1]] else None
    failed = [x for x in (rel.get("semantic_failed_checks") or ()) if x != CHECK]
    ev = "passed" if not failed else "failed"
    out = classify_release_status(
        coverage=cov, pipeline_executed=True, strict_gates_passed=True,
        semantic_evaluation=ev, semantic_not_evaluated_reason="",
        semantic_failed_checks=failed,
        semantic_check_evaluability=rel.get("semantic_check_evaluability") or [],
        retrieval_attempts=rel.get("retrieval_attempts"),
        expansion_blocked_reason=rel.get("expansion_blocked_reason") or "",
    ).to_dict()
    print("CARRIER", name)
    print("   recorded status         :", rel.get("status"), rel.get("reasons"))
    print("   id-dropped + chain GREEN:", out["status"], out["reasons"])
    print("   strict_acceptance_elig  :", out["strict_acceptance_eligible"])
    print("   coverage present        :", h[A[1]],
          {k: cov.get(k) for k in ("requested_core_declared", "coverage_ratio",
                                   "minimum_core_satisfied", "unmatched_terms",
                                   "surviving_processes", "requested_core_source")} if cov else None)
