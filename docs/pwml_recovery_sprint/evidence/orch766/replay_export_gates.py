"""ORCH-766 G9 replay: the menaquinone legs through the real freeze and export gates.

Run through bounded_run.py with PYTHONPATH=<tree>/src, once on the base tree and once
on the tip. Prints only; reads the two legs (untracked, primary checkout) read-only.

For each leg's own ``final_mapped.json`` -- the frozen quarantine output -- it runs, in
the order the app does (``freeze_canonical_payload`` then ``run_pwml_export``):

1. ``quarantine_and_close`` with the leg's own export_mode / strict_db / context;
2. the final pre-export Stage-3 gate, ``run_strict_post_normalization_gates(...,
   enforce_all_proteins_connected=True)``, as ``freeze_canonical_payload`` calls it;
3. the required-field gate, ``validate_pre_export`` on the payload with
   ``run_pwml_export``'s metadata defaults (pathway name from the leg's context);
4. ``build_pwml_ir`` + ``validate_pwml_ir`` + ``blocking_pwml_ir_errors`` (no DB
   resolver), the exporter's next step.

No LLM, no network, no prefreeze DB pass (``run_pwml_export`` re-runs
``run_prefreeze_resolution`` before the IR; that pass is DB-backed and not replayed).
"""
import json
from copy import deepcopy
from pathlib import Path

from t2pw.pipeline.process_normalizer import GateValidationError, run_strict_post_normalization_gates
from t2pw.pipeline.stage_contracts import StageContractError, validate_pre_export
from t2pw.pipeline.strict_quarantine import admitted_payload_hash, quarantine_and_close
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir
from t2pw.pwml.writer import blocking_pwml_ir_errors

PRIMARY = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
LEGS = (
    "runs_papers/ORCH-759p/legs/menaquinone-strict/2026-10-22_2154/papers/PMC12421875/strict",
    "runs_papers/ORCH-759/legs/menaquinone-strict/2026-10-22_1148/papers/PMC12421875/strict",
)

for leg in LEGS:
    d = PRIMARY / leg
    fm = json.loads((d / "final_mapped.json").read_text(encoding="utf-8"))
    q = json.loads((d / "quarantine_report.json").read_text(encoding="utf-8"))
    ctx = q["coverage"]["requested_context"]
    print(f"== {leg}")
    print(f"   input hash {admitted_payload_hash(fm)} | leg resulting_payload_hash {q['resulting_payload_hash']}")
    res = quarantine_and_close(deepcopy(fm), strict_db=bool(q["strict_db"]), pathway_context=ctx, mode=q["export_mode"])
    out = res.payload
    print(f"1. quarantine: ok={res.ok} release={res.quarantine_report['release']['status']} "
          f"refusals={res.refusal_reasons} result hash {admitted_payload_hash(out)}")
    print(f"   species rows: {[r.get('name') for r in out['entities']['species']]}")
    print(f"   removed_entity_report.removed_entities: {res.removed_entity_report['removed_entities']}")
    print(f"   closure iterations: {[(i['iteration'], i['removed_entities']) for i in res.closure_report['iterations']]}")
    try:
        run_strict_post_normalization_gates(deepcopy(out), enforce_all_proteins_connected=True)
        print("2. final pre-export Stage-3 gate: PASS")
    except GateValidationError as exc:
        print(f"2. final pre-export Stage-3 gate: FAIL {[e.get('reason') for e in exc.details.get('errors') or []]}")
    gate_payload = deepcopy(out)
    md = gate_payload.setdefault("metadata", {})
    for k, v in (("pathway_name", ctx["pathway_name"]), ("name", ctx["pathway_name"]),
                 ("pathway_subject", "Metabolic"), ("subject", "Metabolic"), ("description", ""),
                 ("width", 6400), ("height", 1400)):
        md.setdefault(k, v)
    try:
        report = validate_pre_export(gate_payload, strict_db=bool(q["strict_db"]))
    except StageContractError as exc:
        report = exc.report
    rf = report.get("pwml_contract_report") or {}
    print(f"3. required-field gate: ok={rf.get('ok')} errors={[(e['code'], e.get('pointer')) for e in rf.get('errors') or []]}"
          f" outer_errors={[e.get('code') for e in report.get('errors') or []]}")
    try:
        ir, ir_report = build_pwml_ir(deepcopy(gate_payload), pathway_name=ctx["pathway_name"], strict_db=bool(q["strict_db"]))
        val = validate_pwml_ir(ir)
        print(f"4. IR: species={[s.get('name') for s in ir['species']]} blocking={blocking_pwml_ir_errors(ir_report)}"
              f" validate_errors={[e.get('code') for e in val.get('errors') or []]}")
    except Exception as exc:  # noqa: BLE001
        print(f"4. IR build raised {type(exc).__name__}: {str(exc)[:300]}")
print("REPLAY DONE")
