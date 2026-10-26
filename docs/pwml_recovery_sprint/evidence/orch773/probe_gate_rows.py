"""ORCH-773 diagnosis: for each captured leg, every downstream gate error with its row pointer.

Usage: python probe_gate_rows.py --src <tree>/src --cap <capture dir> [--legs <slug> ...]

For each ``<cap>/<slug>/freeze_input.json`` (written by capture_freeze_inputs.py): quarantine_and_close
(strict, the leg's pathway_context), then on the resulting payload the final gate
(run_strict_post_normalization_gates, enforce_all_proteins_connected=True), the Stage-8 contract
(validate_post_normalization), the required-field gate (validate_pre_export with the app's metadata
defaults), and the IR (run_prefreeze_resolution on a copy, build_pwml_ir, validate_pwml_ir). Prints each
error's pointer, code and message, and for every row pointer the row's name and, for actors, the actor.
Read-only; prints only. Offline: socket connect/DNS raise.
"""

import argparse
import json
import re
import socket
import sys
from copy import deepcopy
from pathlib import Path


def _block(*_a, **_k):
    raise OSError("ORCH773 probe: network blocked")


socket.socket.connect = _block
socket.socket.connect_ex = _block
socket.create_connection = _block
socket.getaddrinfo = _block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--legs", nargs="*")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    import t2pw

    print("t2pw:", Path(t2pw.__file__).resolve().parent)
    from t2pw.pipeline.strict_quarantine import quarantine_and_close
    from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates, GateValidationError
    from t2pw.pipeline.stage_contracts import validate_post_normalization, validate_pre_export, StageContractError
    from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir
    from t2pw.pwml.writer import blocking_pwml_ir_errors
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    def row_at(payload, pointer):
        m = re.match(r"^/(processes|entities)/([a-z_]+)/(\d+)(?:/([a-z_]+)/(\d+))?", pointer or "")
        if not m:
            return ""
        group, bucket, i = m.group(1), m.group(2), int(m.group(3))
        rows = (payload.get(group) or {}).get(bucket) or []
        if i >= len(rows) or not isinstance(rows[i], dict):
            return f"<no row {i}>"
        row = rows[i]
        s = f"row name={row.get('name')!r}"
        if m.group(4):
            sub = row.get(m.group(4)) or []
            j = int(m.group(5))
            if j < len(sub):
                s += f" {m.group(4)}[{j}]={json.dumps(sub[j], ensure_ascii=False, default=str)[:220]}"
        return s

    caps = sorted(p for p in Path(args.cap).iterdir() if (p / "freeze_input.json").exists())
    for d in caps:
        if args.legs and d.name not in args.legs:
            continue
        doc = json.loads((d / "freeze_input.json").read_text(encoding="utf-8"))
        res = quarantine_and_close(doc["payload"], strict_db=True, pathway_context=doc["pathway_context"], mode="pathwhiz")
        p = res.payload
        print(f"\n## {d.name}  quarantine ok={res.ok} refusal={res.refusal_reasons}")
        for key in ("unexportable_entities", "entity_type_overlaps", "degree_zero_exports"):
            for item in (res.quarantine_report.get("strict_invariants") or {}).get(key) or []:
                print(f"   invariant {key}: {item}")
        try:
            g = run_strict_post_normalization_gates(deepcopy(p), enforce_all_proteins_connected=True)
            g = {"ok": True, **g}
        except GateValidationError as exc:
            g = {"ok": False, **(exc.details or {})}
        for e in g.get("errors") or []:
            print(f"   G1 final  {e.get('path')} | {str(e.get('reason'))[:150]} | {row_at(p, e.get('path'))}")
        c = validate_post_normalization(deepcopy(p), g)
        for e in c.get("errors") or []:
            if e.get("code") == "actor_schema_not_canonical":
                print(f"   G2 stage8 {e.get('pointer') or e.get('path')} | {e.get('code')} | {row_at(p, e.get('pointer') or e.get('path'))}")
        gp = deepcopy(p)
        md = gp.setdefault("metadata", {})
        if isinstance(md, dict):
            for k, v in (("pathway_name", "Generated Pathway"), ("name", "Generated Pathway"), ("pathway_subject", "Metabolic"),
                         ("subject", "Metabolic"), ("description", ""), ("width", 3200), ("height", 1400)):
                md.setdefault(k, v)
        try:
            pre = validate_pre_export(gp, strict_db=True)
        except StageContractError as exc:
            pre = exc.report or {}
        rq = (pre or {}).get("pwml_contract_report") or {}
        for e in rq.get("errors") or []:
            print(f"   G3 req    {e.get('pointer')} | {e.get('code')} | {str(e.get('message'))[:120]} | {row_at(p, e.get('pointer'))}")
        ip = deepcopy(p)
        try:
            run_prefreeze_resolution(ip, strict_db=True)
            ir, rep = build_pwml_ir(ip, pathway_name="Generated Pathway", pathway_subject="Metabolic", strict_db=True,
                                    width=3200, height=1400)
            val = validate_pwml_ir(ir)
            for e in blocking_pwml_ir_errors(rep):
                print(f"   G4 irrep  {e.get('pointer') or e.get('path')} | {e.get('code')} | {str(e.get('message'))[:160]}"
                      f" | {json.dumps({k: v for k, v in e.items() if k not in ('code', 'message')}, default=str)[:200]}")
            for e in val.get("errors") or []:
                print(f"   G4 irval  {e.get('pointer') or e.get('path')} | {e.get('code')} | {str(e.get('message'))[:160]}")
        except Exception as exc:  # noqa: BLE001
            print(f"   G4 exception {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
