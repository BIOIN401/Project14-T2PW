"""C-056b section 0: MEASURE, before any edit, the four claims this card rests on.

D-033 forbids inheriting a claim that has not been re-measured in the card's own tree,
and every number below was hand-derived by someone else on another SHA.

Claim 1 -- the "1-of-4" hazard. C-056a's reviewer reported that under
``pathway_context=None`` the seam answers ``semantic_evaluation="passed"`` on the strength
of ``CHECK_ID_CONFLICT`` alone. Measured here under BOTH derivations, per leg.

Claim 2 -- ``CHECK_SOURCE_CARRIER`` and ``CHECK_CONNECTED_CORE`` live exposure, and the
counterfactual: how many committed legs would be NEWLY demoted if either gated (D-039
section 3, revisited by this card with measured evidence).

Claim 3 -- the committed corpus carries no runtime semantic verdict at all, so nothing
this card does to the benchmark can move a historical figure.

Claim 4 -- the classifier invariant "semantic_evaluation == failed" implies
"strict_acceptance_eligible is False", exhaustively over the technical inputs. This is
what makes a runtime semantic FAILURE unable to inflate ``strict_ok``, and it is the half
of the hazard that already holds at base.

Reads only. Writes JSON to ``--out``. Imports no C-056b symbol, so it runs unchanged at
the base SHA and at the tip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from t2pw.bench import semantic as _s  # noqa: E402
from t2pw.bench import semantic_production as sp  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    RELEASE_READY,
    SEMANTIC_FAILED,
    SEMANTIC_GATING_CHECKS,
    classify_release_status,
    semantic_verdict,
)
from t2pw.pipeline.strict_quarantine import DEFAULT_MIN_CORE_PROCESSES  # noqa: E402

RUN_ROOTS = ("runs", "runs_verify")
PAYLOAD_NAME = "final_mapped.json"


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _legs(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_root in RUN_ROOTS:
        base = root / run_root
        if not base.is_dir():
            continue
        for payload_path in sorted(base.glob("*/papers/*/*/" + PAYLOAD_NAME)):
            payload = _load(payload_path)
            if not isinstance(payload, dict) or not payload:
                continue
            out.append(
                {
                    "leg": str(payload_path.relative_to(root).parent).replace("\\", "/"),
                    "paper_id": payload_path.parent.parent.name.split("__")[0],
                    "mode": payload_path.parent.name,
                    "payload": payload,
                }
            )
    return out


def _applicability(report: Any) -> Dict[str, Dict[str, bool]]:
    return {
        name: {"applicable": bool(r.applicable), "ok": bool(r.ok)}
        for name, r in report.checks.items()
    }


def _verdict_over(report: Any, gating: Any) -> str:
    """The ``semantic_verdict`` rule, restated over an ARBITRARY gating set.

    The live function hard-codes SEMANTIC_GATING_CHECKS; a counterfactual needs the same
    rule over a widened set. The two are kept in step by
    ``counterfactual_rule_agrees_with_live`` below, which re-derives the LIVE set through
    this function and compares it to what the real ``semantic_verdict`` answered.
    """

    if report is None or not getattr(report, "evaluated", False):
        return "not_evaluated"
    failed = []
    evaluable = 0
    for name in gating:
        result = report.checks.get(name)
        if result is None or not result.applicable:
            continue
        evaluable += 1
        if not result.ok:
            failed.append(name)
    if failed:
        return "failed"
    return "passed" if evaluable else "not_evaluated"


def _manifest_census(root: Path) -> Dict[str, Any]:
    rows = 0
    with_record = 0
    semantic: Dict[str, int] = {}
    eligible: Dict[str, int] = {}
    for run_root in RUN_ROOTS:
        base = root / run_root
        if not base.is_dir():
            continue
        for manifest in sorted(base.glob("*/manifest.jsonl")):
            for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                rows += 1
                record = row.get("release_status")
                if isinstance(record, dict) and record:
                    with_record += 1
                    key = str(record.get("semantic_evaluation") or "(absent)")
                    semantic[key] = semantic.get(key, 0) + 1
                    flag = repr(record.get("strict_acceptance_eligible"))
                    eligible[flag] = eligible.get(flag, 0) + 1
    return {
        "manifest_rows": rows,
        "rows_carrying_release_status": with_record,
        "semantic_evaluation_distribution": semantic,
        "strict_acceptance_eligible_distribution": eligible,
    }


def _classifier_invariant() -> Dict[str, Any]:
    """Exhaustive over the technical inputs: can a FAILED semantic verdict ever leave a
    run strict-acceptance-eligible?"""

    coverages = (
        None,
        {"surviving_processes": 3, "declared_terms": 2, "matched_terms": 2,
         "coverage_ratio": 1.0, "reasons": []},
        {"surviving_processes": 0, "declared_terms": 2, "matched_terms": 0,
         "coverage_ratio": 0.0, "reasons": ["no_surviving_process:0"]},
        {"surviving_processes": 3, "declared_terms": 2, "matched_terms": 0,
         "coverage_ratio": 0.0, "reasons": ["requested_core_coverage_below_minimum:0.000<0.500"]},
    )
    violations: List[Dict[str, Any]] = []
    cases = 0
    for coverage in coverages:
        for executed in (True, False):
            for gates in (True, False):
                for serializable in (True, False):
                    cases += 1
                    status = classify_release_status(
                        coverage,
                        pipeline_executed=executed,
                        strict_gates_passed=gates,
                        serializable_without_invention=serializable,
                        semantic_evaluation=SEMANTIC_FAILED,
                        semantic_failed_checks=[SEMANTIC_GATING_CHECKS[0]],
                    )
                    if status.strict_acceptance_eligible:
                        violations.append({"coverage": coverage, "executed": executed,
                                           "gates": gates, "serializable": serializable})
    return {
        "cases": cases,
        "violations": violations,
        "failed_semantics_never_strict_eligible": not violations,
        "note": "the half of the hazard that already holds at base: a FAILED verdict caps "
                "at review_required, and strict_acceptance_eligible == (status == release_ready)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    import t2pw
    print("T2PW: " + str(Path(t2pw.__file__).resolve()), flush=True)

    gold = load_gold_set()
    by_id = {c.paper_id.casefold(): c for c in gold}

    legs = _legs(root)
    per_leg: List[Dict[str, Any]] = []
    for entry in legs:
        payload = entry["payload"]
        case = by_id.get(entry["paper_id"].casefold())
        arms: Dict[str, Any] = {}
        for arm, ctx in (
            ("pathway_context_none", None),
            ("gold_derived", {"pathway_name": case.requested_pathway if case else "",
                              "likely_organism": case.requested_organism if case else ""}),
        ):
            requested = pathway_context_from_stage_zero(ctx)
            report = sp.evaluate_production_semantics(
                payload,
                requested_pathway=requested.requested_pathway,
                requested_organism=requested.organism,
                mode=entry["mode"],
                min_connected_reactions=DEFAULT_MIN_CORE_PROCESSES,
            )
            checks = _applicability(report)
            live_state, _reason, live_failed = semantic_verdict(report)
            evaluable = [n for n in SEMANTIC_GATING_CHECKS if checks.get(n, {}).get("applicable")]
            arms[arm] = {
                "gating_evaluable": evaluable,
                "gating_evaluable_count": len(evaluable),
                "live_verdict": live_state,
                "live_failed_checks": list(live_failed),
                "counterfactual_rule_agrees_with_live":
                    _verdict_over(report, SEMANTIC_GATING_CHECKS) == live_state,
                "source_carrier_ok": checks.get(_s.CHECK_SOURCE_CARRIER, {}).get("ok"),
                "connected_core_ok": checks.get(_s.CHECK_CONNECTED_CORE, {}).get("ok"),
                "verdict_if_source_carrier_gated":
                    _verdict_over(report, tuple(SEMANTIC_GATING_CHECKS) + (_s.CHECK_SOURCE_CARRIER,)),
                "verdict_if_connected_core_gated":
                    _verdict_over(report, tuple(SEMANTIC_GATING_CHECKS) + (_s.CHECK_CONNECTED_CORE,)),
            }
        per_leg.append({"leg": entry["leg"], "paper_id": entry["paper_id"],
                        "mode": entry["mode"], "arms": arms})

    def _tally(arm: str) -> Dict[str, Any]:
        rows = [p["arms"][arm] for p in per_leg]
        newly_carrier = [
            p["leg"] for p in per_leg
            if p["arms"][arm]["live_verdict"] != "failed"
            and p["arms"][arm]["verdict_if_source_carrier_gated"] == "failed"
        ]
        newly_core = [
            p["leg"] for p in per_leg
            if p["arms"][arm]["live_verdict"] != "failed"
            and p["arms"][arm]["verdict_if_connected_core_gated"] == "failed"
        ]
        counts: Dict[str, int] = {}
        for r in rows:
            counts[r["live_verdict"]] = counts.get(r["live_verdict"], 0) + 1
        evaluable_counts: Dict[str, int] = {}
        for r in rows:
            key = str(r["gating_evaluable_count"])
            evaluable_counts[key] = evaluable_counts.get(key, 0) + 1
        return {
            "legs": len(rows),
            "live_verdict_distribution": counts,
            "gating_evaluable_count_distribution": evaluable_counts,
            "rule_restatement_agrees_everywhere":
                all(r["counterfactual_rule_agrees_with_live"] for r in rows),
            "source_carrier_failing_legs": sum(1 for r in rows if r["source_carrier_ok"] is False),
            "connected_core_failing_legs": sum(1 for r in rows if r["connected_core_ok"] is False),
            "newly_demoted_if_source_carrier_gated": newly_carrier,
            "newly_demoted_if_connected_core_gated": newly_core,
        }

    report = {
        "task": "C-056b",
        "purpose": "section 0 measurement, before any edit (D-033)",
        "root": str(root),
        "payload_legs_measured": len(per_leg),
        "min_core_processes_used": DEFAULT_MIN_CORE_PROCESSES,
        "gating_set_measured": list(SEMANTIC_GATING_CHECKS),
        "claim_3_manifest_census": _manifest_census(root),
        "claim_4_classifier_invariant": _classifier_invariant(),
        "claim_1_and_2_tally": {arm: _tally(arm) for arm in ("pathway_context_none", "gold_derived")},
        "per_leg": per_leg,
        "release_ready_constant": RELEASE_READY,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("WROTE " + str(out_path))
    print(json.dumps({k: v for k, v in report.items() if k != "per_leg"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
