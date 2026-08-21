"""C-057 A/B probe: what the quarantine lineage pass moved, over the real cohort.

Run the SAME file against a base tree and the card tree and diff the two JSON
documents. It quarantines every leg of the frozen baseline cohort -- the same
legs ``test_c011_freeze_seam_golden_equivalence`` and
``test_strict_quarantine_real_artifact_replay`` use -- and records, per leg:

* ``canonical_graph_sha256`` -- PRODUCT_CONTRACT section 6, bound by exporters.
  **Must be identical.** ``PRODUCT_CONTRACT:178``: "Lineage must not change graph
  equivalence, but lineage changes must remain detectable."
* ``canonical_payload_sha256`` and ``payload_sha256`` -- bound by
  persisted-artifact integrity. Moves only where a lineage-carrying row reaches
  the saved payload, which on this stage is the retained row of a quarantined
  LOCKED reaction and nothing else.
* ``surviving`` -- every row that survives, by bucket, with its name. **Must be
  identical**: changing it is a merge rule 6 / rule 7 violation.
* ``admissions`` / ``removed`` / ``report`` -- the decisions and the report
  fields. **Must be identical.** Lineage describes decisions, it never causes
  them, and ``detail_row_keys`` in particular pins that no ``row_digest`` moved.
* ``lineage_*`` -- where a ``stage="quarantine"`` entry actually landed. The only
  field expected to differ; at base every one of them is empty.

Orchestration tooling. It imports ``t2pw`` from the tree named by ``--src``
BEFORE it imports the cohort helper out of ``tests/`` -- that helper inserts its
own tree's ``src`` on the path, and importing it first would silently measure the
card tree twice. The assertion on ``strict_quarantine.__file__`` is the proof
that did not happen, and it is the F-051 hazard in its own shape: the invalid
control is the one that gives the comfortable answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


def _lineage_census(document: Any, path: str = "") -> List[Dict[str, str]]:
    """Every ``provenance_lineage`` entry in ``document``, with where it sits."""

    found: List[Dict[str, str]] = []
    if isinstance(document, dict):
        for entry in document.get("provenance_lineage") or []:
            if isinstance(entry, dict):
                found.append({
                    "at": path or "/",
                    "stage": str(entry.get("stage") or ""),
                    "origin": str(entry.get("origin") or ""),
                    "support": str(entry.get("support") or ""),
                    "paper_explicit": str(entry.get("paper_explicit") or ""),
                    "review_required": str(entry.get("review_required")),
                    "reason": str(entry.get("reason") or ""),
                    "uncertainty": str(entry.get("uncertainty") or ""),
                    "sources": str(len(entry.get("sources") or [])),
                })
        for key, value in document.items():
            if key != "provenance_lineage":
                found.extend(_lineage_census(value, f"{path}/{key}"))
    elif isinstance(document, list):
        for index, item in enumerate(document):
            found.extend(_lineage_census(item, f"{path}/{index}"))
    return found


def _survivors(payload: Any) -> Dict[str, List[str]]:
    """Every surviving row, by bucket and name. The rule 6 / rule 7 control."""

    out: Dict[str, List[str]] = {}
    payload = payload if isinstance(payload, dict) else {}

    def label(row: Any) -> str:
        if not isinstance(row, dict):
            return str(row)
        for key in ("name", "compound", "protein", "entity", "locked_reaction_id"):
            if row.get(key):
                return str(row[key])
        return "<row>"

    for section in ("entities", "processes", "element_locations"):
        buckets = payload.get(section)
        if isinstance(buckets, dict):
            for bucket, rows in sorted(buckets.items()):
                if isinstance(rows, list):
                    out[f"{section}/{bucket}"] = [label(row) for row in rows]
    for key in ("biological_states", "quarantined_locked_reactions"):
        rows = payload.get(key)
        if isinstance(rows, list):
            out[key] = [label(row) for row in rows]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="the tree's src/ directory")
    parser.add_argument("--tests", required=True, help="the tree's tests/ directory")
    parser.add_argument("--out", required=True, help="where to write the JSON census")
    args = parser.parse_args()

    src = str(Path(args.src).resolve())
    sys.path.insert(0, src)
    from t2pw.pipeline import strict_quarantine  # noqa: E402
    from t2pw.pipeline.canonical_hash import (  # noqa: E402
        canonical_graph_sha256, canonical_payload_sha256)
    from t2pw.pipeline.extraction_diagnostics import payload_hash  # noqa: E402

    measured = Path(strict_quarantine.__file__).resolve()
    assert measured.is_relative_to(Path(src)), (
        f"measuring {measured}, not the tree at {src} -- the control is invalid")

    sys.path.insert(0, str(Path(args.tests).resolve()))
    import test_strict_quarantine_real_artifact_replay as replay  # noqa: E402

    document: Dict[str, Any] = {
        "measured_module": str(measured),
        "lineage_writer_present": hasattr(strict_quarantine, "_attribute_quarantined_processes"),
        "legs": {},
    }
    for leg in replay._legs():
        source, raw = replay._payload_for(leg)
        normalized = replay._normalized(raw)
        for mode in ("strict", "research"):
            given = deepcopy(normalized)
            result = strict_quarantine.quarantine_and_close(
                given, strict_db=True, mode=mode)
            report = result.quarantine_report
            key = f"{leg.parts[-4]}/{leg.parts[-2][:24]}/{leg.parts[-1]}::{mode}"
            document["legs"][key] = {
                "payload_source": source,
                "canonical_graph_sha256": canonical_graph_sha256(result.payload),
                "canonical_payload_sha256": canonical_payload_sha256(result.payload),
                "payload_sha256": payload_hash(result.payload),
                "input_payload_mutated": given != normalized,
                "surviving": _survivors(result.payload),
                "admissions": [
                    {
                        "pointer": str(row.get("pointer")),
                        "name": str(row.get("name")),
                        "state": str(row.get("state")),
                        "reason": str(row.get("reason")),
                        "iteration": row.get("iteration"),
                        "detail_row_keys": sorted((row.get("detail") or {}).get("row") or {}),
                    }
                    for row in report.get("admissions") or []
                ],
                "removed": result.removed_entity_report,
                "report": {
                    key: report.get(key)
                    for key in ("schema_version", "ok", "admitted_payload_hash",
                                "resulting_payload_hash", "decision_id",
                                "decision_input_hash", "counts", "refusal_reasons",
                                "review_reasons", "policy_version")
                },
                "release": report.get("release"),
                "closure": result.closure_report,
                "lineage_in_saved_payload": _lineage_census(result.payload),
                "lineage_in_quarantine_report": _lineage_census(report),
                "lineage_in_removed_entity_report": _lineage_census(
                    result.removed_entity_report),
                "lineage_in_closure_report": _lineage_census(result.closure_report),
            }

    Path(args.out).write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8")
    print(f"wrote {args.out}: {len(document['legs'])} leg runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
