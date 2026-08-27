"""Decompose the affected-paper cohort's outcome into the two factors the run
ledger requires reported SEPARATELY:

  1. did C-090 clear the semantic blocker `actor_named_in_its_own_cited_span`, and
  2. did the anchor / coverage cap surface behind it?

Only (1) is attributable to merged code. (2) is Stage 0's non-deterministic
`key_compounds`/`key_proteins` draw, whose prior distribution is pinned in
`LEDGER.md` under COHORT REFERENCE DISTRIBUTION. Without this split, a favourable
draw reads as a fix.

Derived from the committed `f094_reopen_probe.py`, which does exactly this replay
for a different check. The ONLY substantive change is the toggled check name:
`bench.semantic.CHECK_ACTOR_EVIDENCE` (`semantic.py:112`), mirrored in
`release_status.py:122`, which is the blocker C-090 removes.

**The control arm is load-bearing and BLOCKING here.** Quoting
`c080_release_flip_probe.py`: *"A counterfactual whose base measurement does not
reproduce the record is not trustworthy, so this is reported first."* If the
replay does not return the RECORDED status for a leg, that leg's counterfactual is
suppressed rather than printed -- an unreproduced harness must not be allowed to
produce an attribution.

Deliberately NOT used: `rev080_census.py`'s `drop_id_check=True, force_pipeline=True`
arm. It forces `pipeline_executed` as well, which is a different counterfactual and
would overstate the result.

No network, no model call, no pipeline execution. Usage:

    <py> orch703_cohort_decompose.py <repo-root> <run-dir> [paper/mode ...]
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(sys.argv[1])
RUN_DIR = sys.argv[2]
sys.path.insert(0, os.path.join(ROOT, "src"))

from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path  # noqa: E402
from t2pw.bench.semantic import CHECK_ACTOR_EVIDENCE  # noqa: E402
from t2pw.pipeline.release_status import classify_release_status  # noqa: E402

LEGS = sys.argv[3:] or ["PMC12096016/strict", "PMC12782028/strict"]
GOLD = load_gold_set(pinned_gold_set_path())

#: Pinned in LEDGER.md before the run so a draw cannot present as a fix.
PRIOR = {
    "PMC12096016/strict": (0.7059, 0.8571),
    "PMC12782028/strict": (0.2222, 0.6923),
}


def replay(leg: str):
    base = os.path.join(ROOT, RUN_DIR, "papers", leg)
    qp, cp = os.path.join(base, "quarantine_report.json"), os.path.join(base, "coverage_summary.json")
    if not (os.path.isfile(qp) and os.path.isfile(cp)):
        print("==== %s\n   NO ARTIFACTS -- leg did not complete\n" % leg)
        return leg, None, None, None
    rel = json.load(open(qp, encoding="utf-8"))["release"]
    cov = json.load(open(cp, encoding="utf-8"))
    failed = list(rel.get("semantic_failed_checks") or [])

    common = dict(
        coverage=cov,
        pipeline_executed=bool(rel.get("pipeline_executed")),
        strict_gates_passed=bool(rel.get("strict_gates_passed")),
        semantic_check_evaluability=rel.get("semantic_check_evaluability") or [],
        retrieval_attempts=rel.get("retrieval_attempts"),
        expansion_blocked_reason=rel.get("expansion_blocked_reason") or "",
    )

    print("==== %s   (%s)" % (leg, RUN_DIR))
    print("   recorded status / eligible   : %s / %s"
          % (rel.get("status"), rel.get("strict_acceptance_eligible")))
    print("   recorded semantic_evaluation : %s" % rel.get("semantic_evaluation"))
    print("   recorded failed checks       : %s" % failed)
    print("   recorded reasons             : %s" % (rel.get("reasons") or []))

    # -- F-132 / anchor-draw context, so attribution does not need a second run --
    ratio = cov.get("coverage_ratio")
    unmatched = cov.get("unmatched_terms") or []
    case = GOLD.by_id(leg.split("/")[0])
    forbidden = []
    for term in unmatched:
        hit = case.forbidden_match(term) if case else None
        if hit is None and case:
            hit = case.forbidden_match(str(term).split("(")[0].strip())
        if hit is not None:
            forbidden.append((term, hit.kind))
    lo, hi = PRIOR.get(leg, (None, None))
    where = ""
    if lo is not None and isinstance(ratio, (int, float)):
        where = ("INSIDE the prior range" if lo <= ratio <= hi
                 else "OUTSIDE the prior range %.4f-%.4f" % (lo, hi))
    print("   coverage_ratio               : %s   [prior %.4f-%.4f]  %s"
          % (ratio, lo if lo is not None else -1, hi if hi is not None else -1, where))
    print("   minimum_core_satisfied       : %s" % cov.get("minimum_core_satisfied"))
    print("   unmatched_terms              : %s" % unmatched)
    print("   ...of which GOLD-FORBIDDEN   : %d  %s   <-- F-132" % (len(forbidden), forbidden))

    # -- 1. CONTROL ARM. Blocking: an unreproduced harness attributes nothing. --
    got = classify_release_status(
        semantic_evaluation=rel.get("semantic_evaluation") or "not_evaluated",
        semantic_not_evaluated_reason=rel.get("semantic_not_evaluated_reason") or "",
        semantic_failed_checks=failed,
        **common
    ).to_dict()
    reproduced = got.get("status") == rel.get("status")
    print("   CONTROL replay as recorded   : status=%-16s elig=%-6s  %s"
          % (got.get("status"), got.get("strict_acceptance_eligible"),
             "REPRODUCES" if reproduced else "!!! DIVERGES -- counterfactual SUPPRESSED !!!"))
    if not reproduced:
        print("   The harness does not reproduce this leg's record. Nothing is attributed.\n")
        return leg, rel.get("status"), None, False

    # -- 2. COUNTERFACTUAL: C-090's blocker removed, verdict recomputed from the rest --
    remaining = [f for f in failed if f != CHECK_ACTOR_EVIDENCE]
    if remaining == failed:
        print("   COUNTERFACTUAL               : n/a -- %r was not among the failed checks"
              % CHECK_ACTOR_EVIDENCE)
        print("   FACTOR 1 (C-090 semantic)    : nothing to clear on this leg")
        print("   FACTOR 2 (draw / cap)        : whatever is in 'recorded reasons' above\n")
        return leg, rel.get("status"), rel.get("status"), True
    fixed = classify_release_status(
        semantic_evaluation=("passed" if not remaining else "failed"),
        semantic_not_evaluated_reason="",
        semantic_failed_checks=remaining,
        **common
    ).to_dict()
    print("   COUNTERFACTUAL, C-090 applied: status=%-16s elig=%-6s"
          % (fixed.get("status"), fixed.get("strict_acceptance_eligible")))
    print("   reasons once semantics clear : %s" % (fixed.get("reasons") or []))
    print("   FACTOR 1 (C-090 semantic)    : blocker %s"
          % ("REMOVED" if not remaining else "still failing on %s" % remaining))
    print("   FACTOR 2 (draw / cap)        : %s"
          % ("NONE -- leg would export" if fixed.get("status") == "release_ready"
             else "surfaces: %s" % (fixed.get("reasons") or [])))
    if fixed.get("status") == "release_ready":
        print("   *** WOULD REACH release_ready -- check FACTOR 2 and the prior range"
              " before calling this a fix ***")
    print()
    return leg, rel.get("status"), fixed.get("status"), True


rows = [replay(leg) for leg in LEGS]
print("SUMMARY   recorded -> with C-090's semantic blocker removed")
for leg, before, after, ok in rows:
    if before is None:
        print("  %-24s LEG DID NOT COMPLETE" % leg)
    elif not ok:
        print("  %-24s %-16s -> SUPPRESSED (control did not reproduce)" % (leg, before))
    else:
        flag = "  <== reaches release_ready" if after == "release_ready" and before != after else ""
        print("  %-24s %-16s -> %-16s%s" % (leg, before, after, flag))
print("\nAttribution rule (LEDGER, COHORT REFERENCE DISTRIBUTION): a Priority-5 move needs")
print("FACTOR 1 attributable to merged code AND FACTOR 2 absent. A move driven by FACTOR 2")
print("alone, or by gold-forbidden terms simply not being drawn, is draw luck, not a fix.")
