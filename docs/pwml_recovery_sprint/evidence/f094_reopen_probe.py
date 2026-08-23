"""Would correcting the within-kind rule in semantic_production.py flip a leg to
release_ready, and thereby REOPEN F-094 on PMC12452463/strict?

Deterministic offline replay of classify_release_status using the leg's OWN
recorded coverage verdict (coverage_summary.json) and its OWN recorded semantic
fields (quarantine_report.json -> release), toggling ONLY whether
no_real_id_or_name_conflict is among the failed checks.

No network, no model call, no pipeline execution.
"""
import json
import os

from t2pw.pipeline.release_status import classify_release_status

LEGS = [
    ("runs_verify/2026-08-22_2147", "PMC12452463/strict"),
    ("runs_verify/2026-08-22_2147", "PMC12096016/research"),
    ("runs_verify/2026-08-21_2239", "PMC12856317/research"),
]


def replay(rd, leg):
    qp = os.path.join(rd, "papers", leg, "quarantine_report.json")
    cp = os.path.join(rd, "papers", leg, "coverage_summary.json")
    rel = json.load(open(qp, encoding="utf-8"))["release"]
    cov = json.load(open(cp, encoding="utf-8"))

    failed = list(rel.get("semantic_failed_checks") or [])
    evab = rel.get("semantic_check_evaluability") or []

    common = dict(
        coverage=cov,
        pipeline_executed=bool(rel.get("pipeline_executed")),
        strict_gates_passed=bool(rel.get("strict_gates_passed")),
        semantic_check_evaluability=evab,
        retrieval_attempts=rel.get("retrieval_attempts"),
        expansion_blocked_reason=rel.get("expansion_blocked_reason") or "",
    )

    print("==== %s   (%s)" % (leg, rd))
    print("   recorded status              : %s" % rel.get("status"))
    print("   recorded strict_accept_elig  : %s" % rel.get("strict_acceptance_eligible"))
    print("   recorded failed checks       : %s" % failed)
    print("   coverage.minimum_core_satisfied : %s" % cov.get("minimum_core_satisfied"))
    print("   coverage.coverage_ratio         : %s" % cov.get("coverage_ratio"))
    print("   coverage.surviving_processes    : %s" % cov.get("surviving_processes"))
    print("   coverage.requested_core_declared: %s" % cov.get("requested_core_declared"))
    print("   coverage.reasons                : %s" % (cov.get("reasons") or []))

    # 1. Faithful replay of what was recorded -- proves the harness reproduces.
    got = classify_release_status(
        semantic_evaluation=rel.get("semantic_evaluation") or "not_evaluated",
        semantic_not_evaluated_reason=rel.get("semantic_not_evaluated_reason") or "",
        semantic_failed_checks=failed,
        **common
    ).to_dict()
    match = "MATCHES" if got.get("status") == rel.get("status") else "!!! DIVERGES !!!"
    print("   replay as recorded           : status=%-16s elig=%-6s  %s"
          % (got.get("status"), got.get("strict_acceptance_eligible"), match))

    # 2. The counterfactual: the within-kind rule corrected, so that check passes.
    remaining = [f for f in failed if f != "no_real_id_or_name_conflict"]
    fixed = classify_release_status(
        semantic_evaluation=("passed" if not remaining else "failed"),
        semantic_not_evaluated_reason="",
        semantic_failed_checks=remaining,
        **common
    ).to_dict()
    print("   within-kind rule CORRECTED   : status=%-16s elig=%-6s"
          % (fixed.get("status"), fixed.get("strict_acceptance_eligible")))
    if fixed.get("status") == "release_ready":
        print("   *** THIS LEG WOULD BECOME release_ready ***")
    print("   reasons after correction     : %s" % (fixed.get("reasons") or []))
    print()
    return rel.get("status"), fixed.get("status")


results = []
for rd, leg in LEGS:
    results.append((leg,) + replay(rd, leg))

print("SUMMARY")
for leg, before, after in results:
    flag = "  <== FLIPS TO RELEASE_READY" if after == "release_ready" and before != after else ""
    print("  %-24s %-16s -> %-16s%s" % (leg, before, after, flag))
