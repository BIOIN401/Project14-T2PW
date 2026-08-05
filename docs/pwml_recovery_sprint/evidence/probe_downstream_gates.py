"""How far a leg gets AFTER the stale-index fix -- the honest limit of C-010.

WHAT IT SHOWS
-------------
Fixing the index defect (C-010) is not the same as producing PWML. This probe
runs the fixed payload through the next three gates in order:

  1. ``run_strict_post_normalization_gates``  -- the final Stage-3 strict gate
  2. ``validate_pre_export``                  -- the required-field gate
  3. ``build_pwml_ir``                        -- the IR build

On ``ORIGIN_SHA``, PMC12452463 passes 1 and 2 and then FAILS 3 with
``compound_db_resolution_failed`` -- because ``build_pwml_ir`` performs live
PathBank compound resolution AFTER the canonical graph is frozen. That is a
separate defect (C-040/C-050/C-051/C-052), not a C-010 shortfall.

WHY IT IS COMMITTED
-------------------
It is the guard against overclaiming. Anyone reading "C-010 makes PMC12452463
pass quarantine" could reasonably assume it therefore exports; this script shows
exactly where it still stops and why the exporter-purity branches exist. It also
gives T-100's acceptance criterion its evidence base.

INVOCATION
----------
    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/probe_downstream_gates.py

No network and no LLM. NOTE: ``build_pwml_ir`` may attempt a PathBank DB
connection via ``PathBankDbResolver.from_env()``. Where the database is
unreachable it degrades to the offline name index and reports the reason -- which
is itself part of the finding, since a reachable-or-not database changing the
exported identifiers is precisely the canonical-payload violation being measured.

ARTIFACT DEPENDENCY
-------------------
The 1754 leg is committed by INIT-001, not by the control-plane setup; it prints
``[skip]`` until then. The 1207 leg is already committed and always runs.
"""

from __future__ import annotations

import copy
import json

from _repo_root import add_src_to_path, require

add_src_to_path()

import t2pw.pipeline.strict_quarantine as SQ  # noqa: E402
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.stage_contracts import StageContractError, validate_pre_export  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir  # noqa: E402

CONTEXT = {
    "pathway": "enterobactin biosynthesis",
    "organism": "Escherichia coli",
    "key_compounds": ["enterobactin"],
    "key_proteins": ["EntC", "EntB", "EntE", "EntD", "EntF"],
}

LEGS = (
    ("committed 1207 PMC12452463/strict",
     "runs_verify/2026-08-04_1207/papers/PMC12452463/strict/final_mapped.json"),
    ("INIT-001 1754 PMC12452463/strict",
     "runs_verify/2026-08-04_1754/papers/PMC12452463/strict/final_mapped.json"),
)


def _patched_degree_zero(snapshot: dict):
    def patched(payload, admissions):
        referenced = SQ._referenced_entity_norms(snapshot["pre"], admissions)
        complexes = SQ._safe_list(SQ._safe_dict(payload.get("entities")).get("protein_complexes"))
        surviving = {
            SQ._normalize(SQ._row_name(row))
            for row in complexes
            if isinstance(row, dict) and SQ._normalize(SQ._row_name(row)) in referenced
        }
        exempt = SQ._complex_component_norms(payload, surviving)
        out = []
        for bucket in SQ._DEGREE_ZERO_BUCKETS:
            for row in SQ._safe_list(SQ._safe_dict(payload.get("entities")).get(bucket)):
                if not isinstance(row, dict):
                    continue
                name = SQ._row_name(row)
                norm = SQ._normalize(name)
                if not norm or norm in referenced or norm in exempt:
                    continue
                out.append({"bucket": bucket, "name": name})
        return out

    return patched


def probe(label: str, relative: str) -> None:
    path = require(relative)
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))

    original_drop = SQ._drop_quarantined_processes
    original_dz = SQ._degree_zero_exports
    snapshot: dict = {}

    def snapshotting_drop(working, admissions):
        snapshot["pre"] = copy.deepcopy(working)
        return original_drop(working, admissions)

    try:
        SQ._drop_quarantined_processes = snapshotting_drop
        SQ._degree_zero_exports = _patched_degree_zero(snapshot)
        result = SQ.quarantine_and_close(
            copy.deepcopy(payload), strict_db=True, pathway_context=CONTEXT
        )
    finally:
        SQ._drop_quarantined_processes = original_drop
        SQ._degree_zero_exports = original_dz

    print(f"--- {label} ---")
    print(f"  1. quarantine (index-fixed) : ok={result.ok}")
    if not result.ok:
        print("     refusals:", result.refusal_reasons)
        print()
        return

    gate_payload = copy.deepcopy(result.payload)
    try:
        run_strict_post_normalization_gates(
            copy.deepcopy(gate_payload), enforce_all_proteins_connected=True
        )
        print("  2. final Stage-3 strict gate: PASS")
    except GateValidationError as exc:
        errors = (exc.details or {}).get("errors") or []
        print(f"  2. final Stage-3 strict gate: FAIL ({len(errors)} errors)")
        for row in errors[:6]:
            print("       ", str(row)[:150])
        print()
        return

    metadata = gate_payload.setdefault("metadata", {})
    metadata.update({
        "pathway_name": "Enterobactin biosynthesis", "name": "Enterobactin biosynthesis",
        "pathway_subject": "Metabolic", "subject": "Metabolic",
        "description": "", "width": 3200, "height": 1400,
    })

    try:
        contract = validate_pre_export(gate_payload, strict_db=True)
        report = contract.get("pwml_contract_report") or {}
        print(f"  3. required-field gate      : ok={report.get('ok')}")
    except StageContractError as exc:
        report = (exc.report or {}).get("pwml_contract_report") or {}
        print(f"  3. required-field gate      : FAIL ok={report.get('ok')}")
        for row in (report.get("errors") or [])[:6]:
            print("       ", str(row)[:150])
        print()
        return

    try:
        ir, ir_report = build_pwml_ir(gate_payload, db_resolver=None)
        errors = ir_report.get("errors") or []
        print(f"  4. IR build                 : ok={ir_report.get('ok')} ({len(errors)} errors)")
        counts = ir.get("processes", {})
        print("     reactions=%d transports=%d interactions=%d" % (
            len(counts.get("reactions", [])), len(counts.get("transports", [])),
            len(counts.get("interactions", [])),
        ))
        for row in errors[:4]:
            code = row.get("code") if isinstance(row, dict) else ""
            print(f"       {code}: {str(row)[:130]}")
        if errors:
            print("     ^ this is the EXPORTER-SIDE blocker: build_pwml_ir resolves compound")
            print("       identity AFTER the freeze. C-040/050/051/052, not C-010.")
    except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
        print(f"  4. IR build                 : RAISED {type(exc).__name__}: {str(exc)[:200]}")
    print()


def main() -> int:
    for label, relative in LEGS:
        probe(label, relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
