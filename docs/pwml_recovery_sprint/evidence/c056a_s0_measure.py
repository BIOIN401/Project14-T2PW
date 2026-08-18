"""C-056a section 0: MEASURE the two claims the design rests on. D-033 forbids
inheriting an unre-measured claim, and both were hand-evaluated from source predicates.

Claim 1 -- the organism false positive is real: requested ``Escherichia coli`` /
observed ``E. coli`` emits a finding under the LIVE predicate
(``semantic_production._check_requested_organism``).

Claim 2 -- the three-state reachability assertion in
``tests/test_strict_quarantine_release_seam.py`` really does break once semantics gate.
This is measured under BOTH candidate gating rules, because D-039 section 4 attributes the
break to ``_check_source_carrier`` while D-039 section 3 makes that check NON-gating; only
a measurement can say whether the break survives the closed set of four.

Reads only. Writes a JSON report to ``--out``. Imports no C-056a code -- it runs
unchanged at the base SHA and at the tip.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from t2pw.bench import semantic as _s  # noqa: E402
from t2pw.bench import semantic_production as sp  # noqa: E402
from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.rag.admission import ORGANISM_MATCH, compare_organism  # noqa: E402

from test_strict_quarantine_contract_alignment import _base  # noqa: E402

#: The closed gating set D-039 section 3 fixes at exactly four.
GATING = (_s.CHECK_ANCHORS, _s.CHECK_ORGANISM, _s.CHECK_ID_CONFLICT, _s.CHECK_RAG_REINTRODUCTION)

#: Every pair A0-C3 names, plus the two D-039 section 2 says must NOT move.
PAIRS = (
    ("Escherichia coli", "Escherichia coli", "positive: identical"),
    ("Escherichia coli", "E. coli", "ABBREVIATION -- the alleged false positive"),
    ("Escherichia coli", "  Escherichia   coli  ", "whitespace"),
    ("Escherichia coli", "Escherichia coli.", "punctuation"),
    ("Escherichia coli", "Escherichia coli K-12", "strain suffix"),
    ("Escherichia coli", "Escherichia", "bare genus -- already tolerated, must NOT move"),
    ("Escherichia coli", "Escherichia fergusonii", "same genus, other species -- must STAY a finding"),
    ("Escherichia coli", "Listeria monocytogenes", "negative: unrelated organism"),
    ("Escherichia coli", "", "blank observed is never a violation"),
    ("Homo sapiens", "H. sapiens", "abbreviation, second lineage"),
    ("Saccharomyces cerevisiae", "S. cerevisiae", "abbreviation, third lineage"),
)


def _emits_finding(requested: str, observed: str) -> bool:
    """Run the LIVE predicate through its real entry point -- never restated here."""

    check, count = sp._check_requested_organism(
        requested, {"reactions": [{"name": "r", "organism": observed}]}
    )
    return bool(check.findings) or bool(count)


def measure_claim_one() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for requested, observed, note in PAIRS:
        verdict = compare_organism(requested, observed)
        rows.append({
            "requested": requested,
            "observed": observed,
            "note": note,
            "live_predicate_emits_finding": _emits_finding(requested, observed),
            "compare_organism": verdict,
            "widened_would_emit_finding": (
                _emits_finding(requested, observed) and verdict != ORGANISM_MATCH
            ),
        })
    abbreviation = next(r for r in rows if r["observed"] == "E. coli")
    return {
        "claim": "requested 'Escherichia coli' / observed 'E. coli' emits a finding today",
        "holds": abbreviation["live_predicate_emits_finding"] is True,
        "abbreviation_row": abbreviation,
        "table": rows,
        "monotone": all(
            (not r["widened_would_emit_finding"]) or r["live_predicate_emits_finding"] for r in rows
        ),
        "rows_removed_by_widening": [
            f"{r['requested']} / {r['observed']}"
            for r in rows
            if r["live_predicate_emits_finding"] and not r["widened_would_emit_finding"]
        ],
    }


def _reference_payload() -> Dict[str, Any]:
    payload = _base()
    payload["metadata"].pop("key_compounds", None)
    return payload


def _unexportable_payload() -> Dict[str, Any]:
    payload = _base()
    payload["entities"]["proteins"].append({"name": "GCLM"})
    payload["entities"]["protein_complexes"].append({
        "name": "glutamate-cysteine ligase complex",
        "species": "Homo sapiens",
        "components": [
            {"name": "glutamate-cysteine ligase", "stoichiometry": 1},
            {"name": "GCLM", "stoichiometry": 1},
        ],
    })
    payload["processes"]["reactions"][0]["enzymes"] = ["glutamate-cysteine ligase complex"]
    return payload


_SIX_ANCHORS = ["L-glutamate", "ornithine", "citrulline", "argininosuccinate", "arginine", "fumarate"]
_REFERENCE_CONTEXT = {"key_compounds": list(_SIX_ANCHORS)}


def _request_context_only(context: Any, payload: Any) -> Any:
    """Derivation A -- the Stage-0 context and nothing else."""

    return pathway_context_from_stage_zero(deepcopy(context) if context else None)


def _request_metadata_fallback(context: Any, payload: Any) -> Any:
    """Derivation B -- Stage-0 context first, then the payload's own ``metadata``.

    Measured as well as A so the verdict on claim 2 cannot be an artifact of one
    arbitrary wiring choice: B is the derivation that makes CHECK_ANCHORS and
    CHECK_ORGANISM evaluable for ``_base()`` at all.
    """

    merged: Dict[str, Any] = {}
    for container in (payload.get("metadata") if isinstance(payload, dict) else None, context):
        if isinstance(container, dict):
            for key in ("pathway_name", "likely_organism", "organism"):
                if not merged.get(key) and container.get(key):
                    merged[key] = container[key]
    return pathway_context_from_stage_zero(merged)


def _score(label: str, build: Any, context: Any, derive: Any) -> Dict[str, Any]:
    result = quarantine_and_close(build(), strict_db=True, pathway_context=context)
    release = dict(result.quarantine_report["release"])
    request = derive(context, result.payload)
    report = sp.evaluate_production_semantics(
        result.payload,
        requested_pathway=request.requested_pathway,
        requested_organism=request.organism,
        mode=str(result.quarantine_report.get("export_mode") or ""),
    )
    failed_gating = [
        name for name in GATING
        if name in report.checks and report.checks[name].applicable and not report.checks[name].ok
    ]
    return {
        "member": label,
        "requested_pathway": request.requested_pathway,
        "requested_organism": request.organism,
        "status_today": release["status"],
        "semantic_evaluated": report.evaluated,
        "semantic_ok_all_checks": report.ok,
        "failed_checks_all": list(report.failed_checks),
        "inapplicable_checks": list(report.inapplicable_checks),
        "failed_checks_in_closed_four": failed_gating,
        "demotes_under_report_ok_gating": (
            release["status"] == "release_ready" and report.evaluated and not report.ok
        ),
        "demotes_under_closed_four_gating": (
            release["status"] == "release_ready" and bool(failed_gating)
        ),
    }


def measure_claim_two() -> Dict[str, Any]:
    """Replay the three reachability members under both gating rules AND both request
    derivations. Claim 2 is only refuted if it fails under every combination."""

    members = (
        ("_base", _base, None),
        ("_reference_payload", _reference_payload, _REFERENCE_CONTEXT),
        ("_unexportable_payload", _unexportable_payload, None),
    )
    derivations = (
        ("A_context_only", _request_context_only),
        ("B_metadata_fallback", _request_metadata_fallback),
    )
    out: Dict[str, Any] = {
        "claim": "states == {release_ready, review_required, diagnostic_only} breaks once semantics gate",
        "by_derivation": {},
    }
    for dname, derive in derivations:
        rows = [_score(label, build, context, derive) for label, build, context in members]
        out["by_derivation"][dname] = {
            "holds_under_report_ok_gating": any(r["demotes_under_report_ok_gating"] for r in rows),
            "holds_under_closed_four_gating": any(r["demotes_under_closed_four_gating"] for r in rows),
            "members": rows,
        }
    out["holds_under_report_ok_gating"] = any(
        v["holds_under_report_ok_gating"] for v in out["by_derivation"].values()
    )
    out["holds_under_closed_four_gating"] = any(
        v["holds_under_closed_four_gating"] for v in out["by_derivation"].values()
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import t2pw

    report = {
        "task": "C-056a",
        "purpose": "section 0 pre-edit measurement of the two premises",
        "t2pw_file": t2pw.__file__,
        "claim_one_organism_false_positive": measure_claim_one(),
        "claim_two_reachability_break": measure_claim_two(),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"t2pw.__file__ = {t2pw.__file__}")
    print(f"WROTE {out}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
