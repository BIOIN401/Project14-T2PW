"""Reviewer's INDEPENDENT leg census + flip probe for C-080.

Deliberately WIDER than the branch test:
  * enumerates every leg directory under runs_verify/<run>/papers/<paper>/<leg>
    with NO artifact precondition, and reports what each is missing;
  * replays the classifier for every leg that has quarantine_report.json,
    substituting a MAXIMALLY PERMISSIVE coverage verdict when
    coverage_summary.json is absent or unreadable;
  * does NOT require final_mapped.json (the classifier never reads it);
  * reports every hard-coded classifier input.
"""
import json, os, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))
from t2pw.pipeline.release_status import (  # noqa: E402
    RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY,
    classify_release_status, coverage_verdict,
)

CHECK = "no_real_id_or_name_conflict"
RUNS = ROOT / "runs_verify"

# The most permissive coverage verdict that is still well formed: nothing missing,
# ratio 1.0, threshold met, a stated pathway. Used ONLY where the real one is absent.
PERMISSIVE = {
    "requested_core_declared": True,
    "requested_core_source": "explicit_argument",
    "coverage_ratio": 1.0,
    "minimum_core_satisfied": True,
    "surviving_processes": 99,
    "unmatched_terms": [],
    "reasons": [],
}


def read(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def classify(release, coverage, drop_id_check, force_pipeline=False):
    recorded = [str(n) for n in (release.get("semantic_failed_checks") or ())]
    evaluation = str(release.get("semantic_evaluation") or "not_evaluated")
    failed = [n for n in recorded if n != CHECK] if drop_id_check else recorded
    if evaluation == "failed" and not failed:
        evaluation = "passed"
    return classify_release_status(
        coverage=coverage,
        pipeline_executed=True if force_pipeline else bool(release.get("pipeline_executed")),
        strict_gates_passed=True if force_pipeline else bool(release.get("strict_gates_passed")),
        semantic_evaluation=evaluation,
        semantic_not_evaluated_reason=release.get("semantic_not_evaluated_reason") or "",
        semantic_failed_checks=failed,
        semantic_check_evaluability=release.get("semantic_check_evaluability") or [],
        retrieval_attempts=release.get("retrieval_attempts"),
        expansion_blocked_reason=release.get("expansion_blocked_reason") or "",
    ).to_dict()


all_legs, missing_art, no_qr = [], [], []
for run in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
    papers = run / "papers"
    if not papers.is_dir():
        print("RUN WITHOUT papers/ :", run.name)
        continue
    for paper in sorted(papers.iterdir()):
        if not paper.is_dir():
            continue
        for leg in sorted(paper.iterdir()):
            if not leg.is_dir():
                continue
            name = f"{run.name}/papers/{paper.name}/{leg.name}"
            have = {a: (leg / a).is_file() for a in
                    ("quarantine_report.json", "coverage_summary.json", "final_mapped.json")}
            all_legs.append((name, leg, have))
            if not all(have.values()):
                missing_art.append((name, [a for a, ok in have.items() if not ok]))
            if not have["quarantine_report.json"]:
                no_qr.append(name)

print("TOTAL leg directories                :", len(all_legs))
print("legs with ALL THREE artifacts        :",
      sum(1 for _, _, h in all_legs if all(h.values())))
print("legs with quarantine_report.json     :",
      sum(1 for _, _, h in all_legs if h["quarantine_report.json"]))
print("legs with qr + coverage (no fm need) :",
      sum(1 for _, _, h in all_legs if h["quarantine_report.json"] and h["coverage_summary.json"]))
print("legs MISSING at least one artifact   :", len(missing_art))
for n, miss in missing_art:
    print("   MISSING", n, "->", ",".join(miss))
print()

recorded_status = {}
fidelity_mismatch, flips, flips_permissive, unreplayable = [], [], [], []
status_hist = {}
for name, leg, have in all_legs:
    if not have["quarantine_report.json"]:
        unreplayable.append((name, "no quarantine_report.json"))
        continue
    qr = read(leg / "quarantine_report.json")
    if "__error__" in qr:
        unreplayable.append((name, qr["__error__"]))
        continue
    release = qr.get("release") or {}
    rec = release.get("status")
    recorded_status[name] = rec
    status_hist[rec] = status_hist.get(rec, 0) + 1
    if have["coverage_summary.json"]:
        cov = read(leg / "coverage_summary.json")
        cov_src = "recorded"
    else:
        cov, cov_src = PERMISSIVE, "PERMISSIVE-SUBSTITUTE"
    before = classify(release, cov, drop_id_check=False)
    after = classify(release, cov, drop_id_check=True)
    # widest possible: also force the technical chain green
    after_forced = classify(release, cov, drop_id_check=True, force_pipeline=True)
    if before["status"] != rec:
        fidelity_mismatch.append((name, rec, before["status"], cov_src,
                                  before["reasons"], list(release.get("reasons") or ())))
    if after["status"] == RELEASE_READY and rec != RELEASE_READY:
        flips.append((name, rec, after["reasons"], cov_src))
    if after_forced["status"] == RELEASE_READY and rec != RELEASE_READY:
        flips_permissive.append((name, rec, cov_src))

print("recorded status histogram            :", status_hist)
print("legs replayed                        :", len(recorded_status))
print("legs UNREPLAYABLE                    :", len(unreplayable), unreplayable)
print()
print("FIDELITY: replay(before) != recorded  :", len(fidelity_mismatch))
for row in fidelity_mismatch:
    print("   MISMATCH", row)
print()
print("LEGS FLIPPING TO release_ready (upper bound):", len(flips))
for row in flips:
    print("   FLIP", row)
print()
print("LEGS FLIPPING with technical chain ALSO FORCED GREEN:", len(flips_permissive))
for row in flips_permissive:
    print("   FORCED-FLIP", row)
print()
# which legs recorded the id check as a failing gating check at all
carriers = []
for name, leg, have in all_legs:
    if not have["quarantine_report.json"]:
        continue
    release = (read(leg / "quarantine_report.json").get("release") or {})
    if CHECK in (release.get("semantic_failed_checks") or ()):
        carriers.append((name, release.get("status"),
                         list(release.get("semantic_failed_checks") or ()),
                         list(release.get("reasons") or ())))
print("legs RECORDING", CHECK, "as failed:", len(carriers))
for row in carriers:
    print("   CARRIER", row)
