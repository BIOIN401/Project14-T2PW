"""C-056a at its TIP: the obligation D-042 section 1 substituted for D-039 section 4's
struck repair.

D-039 section 4 predicted the three-state reachability assertion breaks on wiring and
prescribed a ``_sourced_base()`` repair. D-042 struck that as measurably false and replaced
it with a stronger duty: **verify by measurement that all three PRODUCT_CONTRACT section 4
states remain reachable through the seam at the tip**, and if any gating check demotes a
member, stop for a fresh ruling.

This script is that measurement. It also records the per-check EVALUABILITY that D-042
section 4 requires be stated rather than smoothed over -- distinguishing *passed*,
*not_evaluated* and *unevaluable-by-signature*, which a bare "no failures" summary hides.

Reads only. Writes JSON to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from t2pw.bench import semantic as _s  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    RELEASE_READY,
    REVIEW_REQUIRED,
    SEMANTIC_GATING_CHECKS,
)
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402

from test_strict_quarantine_contract_alignment import _base  # noqa: E402

_SIX_ANCHORS = ["L-glutamate", "ornithine", "citrulline", "argininosuccinate", "arginine", "fumarate"]
_REFERENCE_CONTEXT = {"key_compounds": list(_SIX_ANCHORS)}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import t2pw

    members = (
        ("_base", _base, None),
        ("_reference_payload", _reference_payload, _REFERENCE_CONTEXT),
        ("_unexportable_payload", _unexportable_payload, None),
    )
    rows: List[Dict[str, Any]] = []
    for label, build, context in members:
        result = quarantine_and_close(build(), strict_db=True, pathway_context=context)
        release = dict(result.quarantine_report["release"])
        rows.append({
            "member": label,
            "status_at_tip": release["status"],
            "semantic_evaluation": release["semantic_evaluation"],
            "semantic_not_evaluated_reason": release["semantic_not_evaluated_reason"],
            "strict_acceptance_eligible": release["strict_acceptance_eligible"],
            "reasons": list(release["reasons"]),
        })

    states = {row["status_at_tip"] for row in rows}
    report = {
        "task": "C-056a",
        "purpose": "D-042 section 1: all three states still reachable at the tip",
        "t2pw_file": t2pw.__file__,
        "gating_set": list(SEMANTIC_GATING_CHECKS),
        "gating_set_size": len(SEMANTIC_GATING_CHECKS),
        "members": rows,
        "states_reachable": sorted(states),
        "all_three_states_reachable": states == {RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY},
        "any_member_demoted_by_semantics": any(
            "semantic_evaluation_failed" in r for row in rows for r in row["reasons"]
        ),
        # D-042 section 4: state evaluability, never "four live gates".
        "evaluability_at_this_seam": {
            _s.CHECK_ANCHORS: "conditional -- evaluable only when the pinned derivation supplies a pathway_name",
            _s.CHECK_ORGANISM: "conditional -- evaluable only when the pinned derivation supplies an organism",
            _s.CHECK_ID_CONFLICT: "unconditional -- the only always-evaluable member at this seam",
            _s.CHECK_RAG_REINTRODUCTION: (
                "structurally unevaluable -- quarantine_and_close takes no admission parameter; "
                "granting one is NOT C-056a's (D-042 section 4)"
            ),
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"t2pw.__file__ = {t2pw.__file__}")
    print(f"WROTE {out}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_three_states_reachable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
