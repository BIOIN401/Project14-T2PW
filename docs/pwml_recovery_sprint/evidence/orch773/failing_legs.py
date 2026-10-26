"""ORCH-773: list every failing corpus leg and its failing step, from a corpus run's JSON.

Usage: python failing_legs.py <corpus.json> <out-dir of that run>
Read-only. Prints, per replayable failure: the failure stage, the quarantine refusal and its
strict_invariants rows, the final-gate errors, the export error, the Stage-8 contract codes,
the required-field gate codes (with the error rows) and the IR blocking codes.
"""

import json
import re
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    doc = load(sys.argv[1])
    out = Path(sys.argv[2])
    n = 0
    for row in doc["legs"]:
        r = row.get("replay")
        if not r or r.get("error") or r.get("pwml_written"):
            continue
        n += 1
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", row["id"])
        work = out / slug
        q = r.get("quarantine") or {}
        fg = r.get("final_gate") or {}
        ex = r.get("export") or {}
        print(f"\n## {n}. {row['id']}  stage={r.get('failure_stage')}  checkpoint={r.get('checkpoint')}")
        print(f"   quarantine ok={q.get('ok')} refusal={q.get('refusal')} release={q.get('release')} counts={q.get('counts')}")
        qr = work / "outputs" / "quarantine_report.json"
        if qr.exists():
            inv = load(qr).get("strict_invariants") or {}
            for key in ("entity_type_overlaps", "degree_zero_exports", "unexportable_entities"):
                for item in inv.get(key) or []:
                    print(f"   invariant {key}: {item}")
        for e in fg.get("errors") or []:
            print(f"   final gate: {e}")
        if ex:
            print(f"   export ok={ex.get('ok')} error={ex.get('error')!r}")
            print(f"   stage3_contract_codes={ex.get('stage3_contract_codes')} required_gate_codes={ex.get('required_gate_codes')}"
                  f" ir_blocking={ex.get('ir_blocking')} ir_validation={ex.get('ir_validation')}")
        rg = work / "outputs" / "pwml_required_field_gate_report.json"
        if rg.exists() and ex.get("required_gate_ok") is False:
            for e in (load(rg).get("errors") or [])[:12]:
                print(f"   required: {e.get('code')} {e.get('path')} {str(e.get('message'))[:140]}")
        sg = work / "outputs" / "pwml_stage3_gate_report.json"
        if sg.exists():
            d = load(sg)
            for e in ((d.get("contract_report") or {}).get("errors") or [])[:12]:
                print(f"   stage8 contract: {e.get('code')} {e.get('path')} {str(e.get('message'))[:140]}")
            for e in ((d.get("gate_report") or {}).get("errors") or [])[:12]:
                print(f"   stage8 gate: {e.get('path')} {str(e.get('reason'))[:140]}")
    print(f"\n{n} failing legs with a checkpoint")


if __name__ == "__main__":
    main()
