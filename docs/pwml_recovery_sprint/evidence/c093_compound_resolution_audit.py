"""C-093: audit the COMPOUND RESOLUTION of each newly committed leg fixture.

C-093 must decide, per leg, GOLDEN or EXCLUDED -- and must refuse to pin a leg
whose resolution is WRONG rather than merely uncovered. That decision needs the
resolution measured, not assumed. For every leg named on the command line this
reports:

* the quarantine verdict and its refusal trigger (the C-068 exclusion criterion);
* the accession corroborants C-068 used (``enrichment`` / ``ec_number`` counts,
  ``prefreeze_db_resolution``);
* every compound row's ``mapping_meta.resolution.status``, and which rows export
  an identity while their own resolution says ``ambiguous`` or ``fallback``;
* ``_norm`` collisions among compound names -- the shape that makes
  ``build_pwml_ir`` refuse post-freeze (``PWML_IR_DUPLICATE_NAMED_ROW``);
* compound rows carrying no resolution verdict at all, which C-051 made
  ``build_pwml_ir`` refuse.

Read-only. Uses ``t2pw.pwml.ir`` as an oracle and mutates nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src")]

from t2pw.pwml import ir  # noqa: E402

COMPOUND_KEYS = ["pathbank_compound_id", "pw_compound_id", "pathwhiz_id"]


def _audit(relative: str) -> Dict[str, Any]:
    text = (ROOT / relative).read_text(encoding="utf-8")
    payload = json.loads(text)
    compounds = ((payload.get("entities") or {}).get("compounds")) or []

    statuses: Counter = Counter()
    exported_from_soft: List[Dict[str, Any]] = []
    no_verdict: List[str] = []
    norms: Dict[str, List[str]] = {}

    for row in compounds:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        norms.setdefault(ir._norm(name), []).append(name)
        meta = row.get("mapping_meta") or {}
        resolution = meta.get("resolution") or {}
        status = resolution.get("status")
        statuses[status if status is not None else "<none>"] += 1
        if not resolution:
            no_verdict.append(name)
        identity = ir._db_id(row, COMPOUND_KEYS)
        if status in ("ambiguous", "fallback") and identity not in (None, ""):
            exported_from_soft.append({
                "name": name, "status": status,
                "issue": resolution.get("issue"),
                "exported_identity": identity,
                "confidence": meta.get("confidence"),
            })

    report = (ROOT / relative).parent / "quarantine_report.json"
    quarantine: Dict[str, Any] = {"present": report.is_file()}
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        quarantine.update(
            ok=data.get("ok"),
            refusal_reasons=data.get("refusal_reasons"),
            degree_zero_exports=(data.get("strict_invariants") or {}).get(
                "degree_zero_exports"),
            unexportable=(data.get("strict_invariants") or {}).get(
                "unexportable_entities"),
        )

    return {
        "leg": relative,
        "quarantine": quarantine,
        "enrichment_occurrences": text.count('"enrichment"'),
        "ec_number_occurrences": text.count('"ec_number"'),
        "prefreeze_db_resolution": payload.get("prefreeze_db_resolution"),
        "compound_rows": len(compounds),
        "resolution_status_counts": dict(sorted(statuses.items(),
                                                key=lambda kv: str(kv[0]))),
        "compounds_with_no_resolution_verdict": no_verdict,
        "identity_exported_from_a_soft_resolution": exported_from_soft,
        "norm_collisions": {k: v for k, v in sorted(norms.items()) if len(v) > 1},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("legs", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = [_audit(leg) for leg in args.legs]
    summary = {
        "legs": len(rows),
        "quarantine_not_ok": [r["leg"] for r in rows
                              if r["quarantine"].get("ok") is not True],
        "legs_with_norm_collisions": [r["leg"] for r in rows if r["norm_collisions"]],
        "legs_with_a_compound_missing_a_verdict": [
            r["leg"] for r in rows if r["compounds_with_no_resolution_verdict"]],
        "legs_exporting_an_identity_from_a_soft_resolution": [
            r["leg"] for r in rows if r["identity_exported_from_a_soft_resolution"]],
        "per_leg": rows,
    }
    blob = json.dumps(summary, indent=1, sort_keys=False, default=repr)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
        print(f"wrote {args.out} ({len(blob)} bytes)")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
