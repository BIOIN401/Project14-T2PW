"""The exporter mutates biological identity after the canonical freeze.

WHAT IT SHOWS
-------------
Two independent measurements of the same violation.

**A. Committed-artifact diff.** For a leg that PASSED and shipped a PWML, compare
the compound rows in ``final_mapped.json`` (the canonical payload, hash-bound to
the final Stage-3 gate report) against the same compounds in ``pwml_ir.json``
(what the exporter actually built). On committed
``runs_verify/2026-08-04_1647/papers/PMC12856317/strict`` the exporter adds
``pathwhiz_id`` and ``db_id`` to all four compounds and gives Glycine NINE
external identifiers that do not exist in the canonical payload -- ``drugbank
DB00145``, ``hmdb HMDB0000123``, ``kegg C00037``, ``chebi 15428``, ``pubchem
5257127``, ``chemspider 730``, ``cas 56-40-6``, ``pathbank_compound_id 78`` --
plus a ``CHEBI:`` prefix normalization on three others.

**B. Live re-derivation.** Build the IR from a quarantined payload and print the
``db_resolution`` block, showing resolution happening at export time and its
``ambiguous`` outcomes.

WHY IT IS COMMITTED
-------------------
PRODUCT_CONTRACT section 5 requires that reloading ``final_mapped.json`` and
exporting again produce a biologically equivalent pathway, and that exporters not
"add, remove, resolve or reinterpret biological content after the canonical graph
is frozen". Measurement A is the falsification of that today. It is the
acceptance target for C-050/C-051 and for milestone T-102: after those branches
the diff in measurement A must be EMPTY.

INVOCATION
----------
    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/probe_exporter_identity_mutation.py

Measurement A is pure file comparison -- no network, no database, always
reproducible. Measurement B calls ``build_pwml_ir``, which may attempt a PathBank
connection; where the database is unreachable it falls back to the offline name
index. Either way the point stands, and the fact that the OUTCOME DEPENDS ON
DATABASE REACHABILITY is itself part of the finding.

ARTIFACT DEPENDENCY
-------------------
Both legs used here (1647 and 1207) are already committed. This script needs
nothing from INIT-001.
"""

from __future__ import annotations

import copy
import json

from _repo_root import add_src_to_path, require

add_src_to_path()

import t2pw.pipeline.strict_quarantine as SQ  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir  # noqa: E402

PASSING_LEG = "runs_verify/2026-08-04_1647/papers/PMC12856317/strict"
REFUSED_LEG = "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json"

IDENTITY_FIELDS = ("pathwhiz_id", "db_id", "db_status", "chosen_rule", "pathbank_compound_id")


def measurement_a() -> None:
    print("=== A. canonical payload vs shipped IR (committed artifacts) ===")
    leg = require(PASSING_LEG)
    if leg is None:
        return
    canonical_path = leg / "final_mapped.json"
    ir_path = leg / "pwml_ir.json"
    if not canonical_path.is_file() or not ir_path.is_file():
        print("  [skip] leg is missing final_mapped.json or pwml_ir.json")
        return

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    canonical_rows = {row["name"]: row for row in canonical["entities"]["compounds"]}
    ir_rows = {row["name"]: row for row in ir["entities"]["compounds"]}

    total_added = 0
    for name, ir_row in ir_rows.items():
        canonical_row = canonical_rows.get(name, {})
        field_diffs = {
            field: (canonical_row.get(field), ir_row.get(field))
            for field in IDENTITY_FIELDS
            if canonical_row.get(field) != ir_row.get(field)
        }
        canonical_ids = canonical_row.get("mapped_ids", {}) or {}
        ir_ids = ir_row.get("mapped_ids", {}) or {}
        id_diffs = {
            key: (canonical_ids.get(key), ir_ids.get(key))
            for key in set(canonical_ids) | set(ir_ids)
            if canonical_ids.get(key) != ir_ids.get(key)
        }
        added = [key for key, (before, _after) in id_diffs.items() if before is None]
        total_added += len(added)
        print(f"  {name}")
        print(f"      identity fields changed : {field_diffs}")
        print(f"      mapped_ids changed      : {id_diffs}")
        if added:
            print(f"      *** {len(added)} identifier(s) ADDED by the exporter: {sorted(added)}")
    print()
    print(f"  TOTAL identifiers added post-freeze: {total_added}")
    print("  ACCEPTANCE after C-050/C-051 (milestone T-102): this must be 0 and every")
    print("  'identity fields changed' entry must be empty.")
    print()


def measurement_b() -> None:
    print("=== B. resolution observed live at export time ===")
    path = require(REFUSED_LEG)
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = SQ.quarantine_and_close(copy.deepcopy(payload), strict_db=True)
    gate_payload = copy.deepcopy(result.payload)
    gate_payload.setdefault("metadata", {}).update({
        "pathway_name": "Enterobactin biosynthesis", "name": "Enterobactin biosynthesis",
        "pathway_subject": "Metabolic", "subject": "Metabolic",
    })
    try:
        _ir, report = build_pwml_ir(gate_payload, db_resolver=None)
    except Exception as exc:  # noqa: BLE001
        print(f"  build_pwml_ir raised {type(exc).__name__}: {str(exc)[:200]}")
        return
    print("  ir report ok :", report.get("ok"), "| errors:", len(report.get("errors") or []))
    resolution = report.get("db_resolution") or {}
    for row in (resolution.get("compounds") or [])[:8]:
        print("    %-42s status=%-24s rule=%s" % (
            str(row.get("raw_name"))[:42], row.get("status"), row.get("chosen_rule")))
    for err in (report.get("errors") or [])[:3]:
        code = err.get("code") if isinstance(err, dict) else ""
        print(f"    ERROR {code}: {str(err)[:140]}")
    print()
    print("  Resolution is happening HERE, after the payload was frozen and hashed.")


def main() -> int:
    measurement_a()
    measurement_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
