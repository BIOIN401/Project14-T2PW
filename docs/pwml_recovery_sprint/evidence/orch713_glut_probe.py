"""Read-only classification probe for the Glutathione strict-failure red.

Answers, for every case in tests/fixtures/strict_failures/cases.json:
  - what quarantine_and_close actually returns now
  - what the fixture says it should return
  - for the failing case, WHY: the coverage verdict, the admission states,
    and whether requested-core terms were declared at all.

Pure and offline. No network, no DB, no batch run.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve()
# tree passed as argv[1]
TREE = Path(sys.argv[1]).resolve()
SRC = TREE / "src"
sys.path.insert(0, str(SRC))

from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    quarantine_and_close,
    evaluate_core_coverage,
    collect_requested_core_terms,
    DEFAULT_MIN_CORE_PROCESSES,
    DEFAULT_MIN_CORE_COVERAGE,
)

print("MEASURED_TREE t2pw =", sys.modules["t2pw"].__file__)

FIX = TREE / "tests" / "fixtures" / "strict_failures" / "cases.json"
cases = json.loads(FIX.read_text(encoding="utf-8"))["cases"]

print("=" * 72)
print("PER-CASE ok VS FIXTURE recovers")
print("=" * 72)
mismatches = []
for c in cases:
    cid = c["id"]
    exp = c["expect"]
    res = quarantine_and_close(deepcopy(c["payload"]), strict_db=True)
    surviving = len((res.payload.get("processes", {}) or {}).get("reactions") or [])
    ok_match = (res.ok is exp["recovers"])
    sr_exp = exp.get("surviving_reactions")
    sr_match = (sr_exp is None) or (surviving == sr_exp)
    flag = "OK " if (ok_match and sr_match) else "RED"
    print(f"{flag} {cid}: ok={res.ok} expect_recovers={exp['recovers']} "
          f"surviving_reactions={surviving} expect={sr_exp}")
    if not (ok_match and sr_match):
        mismatches.append(cid)

print()
print("=" * 72)
print("DEEP DIVE on the mismatching case(s)")
print("=" * 72)
for cid in mismatches or ["only_unrelated_reactions_survive"]:
    c = next(x for x in cases if x["id"] == cid)
    payload = c["payload"]
    print(f"\n--- {cid} ---")
    terms = collect_requested_core_terms(payload, requested_core=None, pathway_context=None)
    print("collect_requested_core_terms(payload) =", terms)
    print("declared =", bool(terms))
    print("DEFAULT_MIN_CORE_PROCESSES =", DEFAULT_MIN_CORE_PROCESSES,
          " DEFAULT_MIN_CORE_COVERAGE =", DEFAULT_MIN_CORE_COVERAGE)
    res = quarantine_and_close(deepcopy(payload), strict_db=True)
    print("result.ok =", res.ok)
    cov = res.coverage or {}
    for k in ("requested_core_terms", "requested_core_declared", "requested_core_source",
              "matched_terms", "unmatched_terms", "coverage_ratio",
              "core_accepted_processes", "auxiliary_accepted_processes",
              "surviving_processes", "quarantined_processes",
              "minimum_core_satisfied", "reasons", "thresholds"):
        if k in cov:
            print(f"  coverage.{k} = {json.dumps(cov[k])}")
    adm = getattr(res, "admissions", None)
    if adm:
        print("  ADMISSIONS:")
        for row in adm:
            print("   ", json.dumps({kk: row.get(kk) for kk in
                                     ("name", "kind", "state", "reason", "core_terms")}))
    print("  surviving reaction names =",
          [r.get("name") for r in (res.payload.get("processes", {}) or {}).get("reactions") or []])
    # What would the verdict be if the context were passed explicitly, as production must?
    if adm is not None:
        v = evaluate_core_coverage(
            payload, adm,
            requested_core=None,
            pathway_context=payload.get("metadata"),
        )
        print("  RE-EVALUATED with pathway_context=payload['metadata']:")
        print("    declared =", v.get("requested_core_declared"),
              " reasons =", json.dumps(v.get("reasons")),
              " core_accepted =", v.get("core_accepted_processes"))

print()
print("MISMATCHES:", json.dumps(mismatches))
