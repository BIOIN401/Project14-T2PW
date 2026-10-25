"""ORCH-773: does the IR's verdict on protein and complex rows depend on the export's pre-freeze seam?

Usage: python ir_prefreeze_equivalence.py --src <tree>/src --cap <capture dir>

The per-row pass builds the IR on the candidate WITHOUT ``run_prefreeze_resolution`` (it renames compounds
and species and may consult the PathBank DB); ``run_pwml_export`` builds it after that seam. The pass maps
only IR errors on ``proteins`` / ``protein_complexes`` rows back to the payload. For every captured leg this
quarantines the freeze input with the tree's quarantine, then builds the IR both ways on the result and
compares the (code, row name) set of blocking errors on those two buckets. Offline. Read-only.
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
socket.create_connection = _block
socket.getaddrinfo = _block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cap", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    import t2pw

    print("t2pw:", Path(t2pw.__file__).resolve().parent)
    from t2pw.pipeline.strict_quarantine import quarantine_and_close
    from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution
    from t2pw.pwml.writer import blocking_pwml_ir_errors

    pat = re.compile(r"^/entities/(proteins|protein_complexes)/(\d+)")

    def entity_errors(payload, prefreeze):
        p = deepcopy(payload)
        if prefreeze:
            run_prefreeze_resolution(p, strict_db=True)
        ir, rep = build_pwml_ir(p, pathway_name="Generated Pathway", pathway_subject="Metabolic", strict_db=True,
                                width=3200, height=1400)
        out = set()
        for e in [*blocking_pwml_ir_errors(rep), *(validate_pwml_ir(ir).get("errors") or [])]:
            m = pat.match(str(e.get("pointer") or ""))
            if m:
                rows = (ir.get("entities") or {}).get(m.group(1)) or []
                i = int(m.group(2))
                out.add((e.get("code"), m.group(1), rows[i].get("name") if i < len(rows) else None))
        return out

    n = same = 0
    for d in sorted(p for p in Path(args.cap).iterdir() if (p / "freeze_input.json").exists()):
        doc = json.loads((d / "freeze_input.json").read_text(encoding="utf-8"))
        res = quarantine_and_close(doc["payload"], strict_db=True, pathway_context=doc["pathway_context"], mode="pathwhiz")
        try:
            a, b = entity_errors(res.payload, False), entity_errors(res.payload, True)
        except Exception as exc:  # noqa: BLE001
            print(f"{d.name}: exception {type(exc).__name__}: {str(exc)[:160]}")
            continue
        n += 1
        same += a == b
        if a or b:
            print(f"{d.name}: without={sorted(a, key=str)} with={sorted(b, key=str)} {'SAME' if a == b else 'DIFFERENT'}")
    print(f"{same}/{n} legs: identical protein/complex IR verdicts with and without the pre-freeze seam")
    return 0


if __name__ == "__main__":
    sys.exit(main())
