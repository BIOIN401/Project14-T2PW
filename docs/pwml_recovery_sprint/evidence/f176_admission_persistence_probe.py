"""F-176 verification probe -- is rejected-RAG evidence available to OFFLINE batch evaluation?

READ-ONLY. Opens archived run directories, writes nothing into them, and produces
no acceptance verdict. **This is not a re-score of T-107/T-108/T-109.** It reports
per-check semantic verdicts against artifacts that already exist on disk; the
milestones' dispositions are untouched and unreferenced by anything here.

WHY IT EXISTS. A prior investigation reported that
``no_rejected_rag_reaction_reintroduced`` is not evaluable in batch because
"``AdmissionReport.rejected`` is not persisted by the batch driver". That claim is
testable four ways and this probe tests all four rather than restating it:

  1. does ``rag_admission_report.json`` exist in real benchmark leg directories?
  2. does it carry a populated ``rejected`` list with the fields the check reads?
  3. does the OFFLINE evaluator (``bench.acceptance`` -> ``bench.semantic``) load it
     and reach a verdict?
  4. is the reported ``inapplicable_reason`` -- which is what a human reads -- true?

THE A/B IS THE POINT. Each leg is evaluated twice against the same payload and gold
case: once with the admission report loaded exactly as ``acceptance.py:1372`` loads
it, and once with ``admission=None``. If arm A is applicable and arm B is not, the
check is evaluable in batch and the persistence claim is false. If both arms are
inapplicable, the claim survives. A single arm cannot distinguish those.

APPLICABLE IS NOT PASSED. Every check is reported as one of PASSED / FAILED /
UNEVALUABLE, never as a bare ``applicable`` boolean -- ``CheckResult.ok`` and
``CheckResult.inapplicable_reason`` are different fields and conflating them is the
error this probe was written to make impossible.

Usage:  python f176_admission_persistence_probe.py <repo-root> <run-dir> [<run-dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# The fields ``semantic._check_rag_reintroduction`` actually reads off a rejected
# row, plus the provenance fields an offline RAG evaluation needs. Named here so
# "the artifact exists" is never mistaken for "the artifact is sufficient".
_CLAIM_FIELDS = ("inputs", "outputs", "enzymes", "reversible")
_PROVENANCE_FIELDS = ("gap_id", "name", "reasons", "evidence", "source_paper")


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _verdict(check: Any) -> str:
    """PASSED / FAILED / UNEVALUABLE for one CheckResult, never a bare boolean."""

    reason = getattr(check, "inapplicable_reason", "") or ""
    if reason:
        return "UNEVALUABLE"
    ok = getattr(check, "ok", None)
    if ok is None:
        return "UNEVALUABLE"
    return "PASSED" if ok else "FAILED"


def _describe_admission(document: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(document, dict):
        return {"present": False}
    rejected = document.get("rejected")
    rows = [r for r in (rejected or ()) if isinstance(r, dict)]
    coverage: Dict[str, int] = {}
    for field in _CLAIM_FIELDS + _PROVENANCE_FIELDS:
        coverage[field] = sum(1 for r in rows if r.get(field) not in (None, "", [], {}))
    return {
        "present": True,
        "top_keys": sorted(document.keys()),
        "has_rejected_key": "rejected" in document,
        "rejected_rows": len(rows),
        "accepted_rows": len([r for r in (document.get("accepted") or ()) if isinstance(r, dict)]),
        "field_coverage": coverage,
    }


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2

    repo = Path(argv[1]).resolve()
    sys.path.insert(0, str(repo / "src"))

    from t2pw.bench.acceptance import (  # noqa: E402  -- after the path pin, deliberately
        _ADMISSION_FILES,
        _PAYLOAD_FILES,
        _QUARANTINE_FILES,
        _first_existing,
        _paper_text,
    )
    from t2pw.bench.goldset import load_gold_set  # noqa: E402
    from t2pw.bench.semantic import validate_semantic_coverage  # noqa: E402

    print(f"MEASURED_TREE t2pw = {Path(__import__('t2pw').__file__).resolve()}")

    gold = load_gold_set()
    cases = {case.paper_id: case for case in gold.cases}
    print(f"gold cases = {len(cases)}")

    totals = {
        "legs": 0,
        "admission_present": 0,
        "admission_with_rejected": 0,
        "rejected_rows": 0,
        "A_applicable": 0,
        "B_applicable": 0,
        "A_passed": 0,
        "A_failed": 0,
        "canonical_payload_legs": 0,
        "fallback_payload_legs": 0,
        "A_failed_on_canonical": 0,
        "A_failed_on_fallback": 0,
    }
    check_matrix: Dict[str, Dict[str, Dict[str, int]]] = {}

    for run_arg in argv[2:]:
        run = Path(run_arg)
        if not run.is_absolute():
            run = (repo / run_arg).resolve()
        print("\n" + "=" * 78)
        print(f"RUN = {run}")
        print("=" * 78)
        papers = run / "papers"
        if not papers.is_dir():
            print("  no papers/ directory -- skipped")
            continue

        for paper_dir in sorted(papers.iterdir()):
            if not paper_dir.is_dir():
                continue
            slug = paper_dir.name
            case = cases.get(slug)
            paper_text = _paper_text(run, slug)
            for mode_dir in sorted(paper_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                mode = mode_dir.name
                totals["legs"] += 1

                payload, payload_source, _ = _first_existing(mode_dir, _PAYLOAD_FILES)
                admission, _, admission_path = _first_existing(mode_dir, _ADMISSION_FILES)
                quarantine, _, _ = _first_existing(mode_dir, _QUARANTINE_FILES)

                shape = _describe_admission(admission if isinstance(admission, dict) else None)
                if shape.get("present"):
                    totals["admission_present"] += 1
                if shape.get("has_rejected_key"):
                    totals["admission_with_rejected"] += 1
                totals["rejected_rows"] += int(shape.get("rejected_rows") or 0)

                print(f"\n  {slug}/{mode}")
                print(f"    payload            : {'yes' if isinstance(payload, dict) and payload else 'NO'}")
                print(f"    admission on disk  : {shape.get('present')}  "
                      f"rejected_rows={shape.get('rejected_rows')}  "
                      f"accepted_rows={shape.get('accepted_rows')}")
                if shape.get("present"):
                    print(f"    admission path     : {admission_path}")
                    print(f"    field coverage     : {shape.get('field_coverage')}")
                if case is None:
                    print("    NOT A GOLD CASE -- semantic evaluation skipped")
                    continue

                # Arm A: exactly what acceptance.py does.
                arm_a = validate_semantic_coverage(
                    case,
                    payload if isinstance(payload, dict) else None,
                    mode=mode,
                    admission=admission if isinstance(admission, dict) else None,
                    quarantine_report=quarantine if isinstance(quarantine, dict) else None,
                    paper_text=paper_text,
                )
                # Arm B: the counterfactual in which the artifact is absent.
                arm_b = validate_semantic_coverage(
                    case,
                    payload if isinstance(payload, dict) else None,
                    mode=mode,
                    admission=None,
                    quarantine_report=quarantine if isinstance(quarantine, dict) else None,
                    paper_text=paper_text,
                )

                if not arm_a.evaluated:
                    print(f"    semantic           : NOT EVALUATED -- {arm_a.not_evaluated_reason}")
                    continue

                # WHICH payload was scored is load-bearing and was previously
                # invisible. ``_PAYLOAD_FILES`` falls back from ``final_mapped.json``
                # -- the canonical, gate-bound, export-eligible payload -- to
                # ``merged_payload.json``, which is PRE-mapping and PRE-quarantine and
                # belongs to a leg that exported nothing. Several check reason strings
                # say "the exported payload"; on a fallback leg no export exists, so a
                # verdict there is a statement about a draft, not about a product.
                # Tallying the two together produces a corpus number nobody can read.
                canonical = payload_source == "final_mapped.json"
                bucket = "canonical" if canonical else "fallback"
                totals["canonical_payload_legs" if canonical else "fallback_payload_legs"] += 1
                print(f"    payload source     : {payload_source}  ({bucket})")

                print("    check                                    ARM A (on-disk)   ARM B (none)")
                for name, check in arm_a.checks.items():
                    va = _verdict(check)
                    vb = _verdict(arm_b.checks.get(name)) if name in arm_b.checks else "ABSENT"
                    row = check_matrix.setdefault(name, {}).setdefault(bucket, {})
                    row[va] = row.get(va, 0) + 1
                    print(f"      {name:<38} {va:<17} {vb}")
                    if va == "FAILED":
                        for finding in (getattr(check, "findings", None) or ())[:3]:
                            print(f"          finding: {json.dumps(finding)[:220]}")

                rag = "no_rejected_rag_reaction_reintroduced"
                ca, cb = arm_a.checks.get(rag), arm_b.checks.get(rag)
                if ca is not None:
                    if _verdict(ca) != "UNEVALUABLE":
                        totals["A_applicable"] += 1
                        if _verdict(ca) == "PASSED":
                            totals["A_passed"] += 1
                        else:
                            totals["A_failed"] += 1
                            totals["A_failed_on_canonical" if canonical
                                   else "A_failed_on_fallback"] += 1
                    else:
                        print(f"      RAG-reintroduction UNEVALUABLE reason: {ca.inapplicable_reason}")
                if cb is not None and _verdict(cb) != "UNEVALUABLE":
                    totals["B_applicable"] += 1

    print("\n" + "=" * 78)
    print("TOTALS")
    print("=" * 78)
    for key, value in totals.items():
        print(f"  {key:<26} = {value}")
    print("\n  per-check verdict tally (ARM A), SPLIT BY PAYLOAD SOURCE.")
    print("    canonical = final_mapped.json -- gate-bound and export-eligible.")
    print("    fallback  = merged_payload.json -- PRE-mapping, PRE-quarantine, nothing exported.")
    print("    These are NOT the same measurement. Do not sum them into one corpus number.")
    for name in sorted(check_matrix):
        print(f"    {name}")
        for bucket in ("canonical", "fallback"):
            if bucket in check_matrix[name]:
                print(f"        {bucket:<10} {check_matrix[name][bucket]}")

    print("\n  RAG-REINTRODUCTION FAILURES BY PAYLOAD SOURCE: "
          f"canonical={totals['A_failed_on_canonical']}  fallback={totals['A_failed_on_fallback']}")
    print("    A failure on a FALLBACK payload is NOT evidence that a rejected claim reached an")
    print("    export -- that leg produced none. It is a statement about a pre-quarantine draft,")
    print("    and the check's own reason string says 'the exported payload', which is wrong there.")

    print("\n  VERDICT ON THE F-176 CLAIM "
          "('AdmissionReport.rejected is not persisted by the batch driver'):")
    if totals["admission_with_rejected"] and totals["A_applicable"] and not totals["B_applicable"]:
        print("    REFUTED. The artifact is on disk in real benchmark legs, it carries")
        print("    populated rejected rows, and the offline evaluator reaches a verdict with")
        print("    it and cannot without it.")
    elif not totals["admission_present"]:
        print("    CONFIRMED. No admission artifact reached any leg directory.")
    else:
        print("    MIXED -- read the per-leg rows above; do not summarise this line alone.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
