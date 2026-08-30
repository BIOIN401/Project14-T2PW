"""ORCH-716: does the SHIPPED payload still fail the gates the run failed on?

Orchestration tooling. Nothing under ``src/`` imports it.

The question this settles
-------------------------
T-107 failed `PMC12452463/strict` and `PMC12180156/strict` with `failure_kind=contract`,
citing protein identity/species gates. But in **both** legs
``final_stage3_gate_report.json`` says ``ok: true, errors: []``, and the only failing report
in ``contract_reports.json`` is ``post_normalization_contract_report``, stamped
``phase: audit_round`` -- a snapshot taken INSIDE the bounded audit loop, which
``streamlit_app.py`` itself documents as *"not a verdict about what shipped -- the remap
below moves the payload again"*.

Two readings fit that evidence and they demand opposite responses:

* **STALE** -- the audit round's payload no longer exists, the remap and the pre-export
  quarantine settled the entities afterwards, and the run was failed on a superseded
  report. PRODUCT_CONTRACT section 1 lists *"a missing or stale gate report"* and *"an
  irrelevant degree-zero entity"* among the **unacceptable terminal blockers**.
* **KEY MISMATCH** -- the gate reads a field the final payload does not populate, so the
  gate would fire on the shipped payload too and ``final_stage3_gate_report`` is the report
  that is wrong. That is far more serious: it would mean the final gate is blind.

These are distinguished by ONE measurement: run the **real production predicates** against
the **final payload** and see whether they still object.

Why the predicates and not the reports
--------------------------------------
Re-reading ``final_stage3_gate_report.json`` would only restate what that report already
says. F-144's rule is the reason: asserting that *a* report is clean is not evidence that
*the payload* is clean. So this probe imports the same functions
``process_normalizer`` calls -- ``protein_external_identity``, ``protein_species_context``
-- and applies them row by row to ``final_mapped.json``.

It also prints, for every protein, WHICH key carried the identity, because
``PMC12180156``'s ALAS2 row has ``uniprot_id: None`` while carrying a verified
``uniprot: P22557``, and a reader who checks only ``uniprot_id`` would conclude the
identifier is missing. A zero from a key that does not exist looks exactly like a zero from
a measurement (F-144).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

LEGS: Tuple[Tuple[str, str], ...] = (
    ("PMC12452463", "strict"),
    ("PMC12180156", "strict"),
    ("PMC12856317", "strict"),   # control: passed T-107
    ("PMC12782028", "strict"),   # control: passed T-107
)

#: Keys an identity can arrive under. Printed per row so the reader can see whether an
#: identity is absent or merely under a key some other reader does not consult.
IDENTITY_KEYS = ("uniprot", "uniprot_id", "drugbank", "drugbank_id", "mapped_ids")


def load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = args[0] if args else "."
    run = args[1] if len(args) > 1 else "runs_verify/2026-08-28_1816"

    sys.path.insert(0, os.path.join(root, "src"))
    from t2pw.pipeline.process_normalizer import (  # noqa: E402
        protein_external_identity,
        protein_species_context,
    )

    print("=" * 88)
    print("ORCH-716 -- production identity/species predicates applied to the SHIPPED payload")
    print("=" * 88)
    print(f"root : {root}")
    print(f"run  : {run}")
    print()

    verdict: Dict[str, str] = {}

    for paper, mode in LEGS:
        base = os.path.join(root, run, "papers", paper, mode)
        payload = load(os.path.join(base, "final_mapped.json"))
        gate = load(os.path.join(base, "final_stage3_gate_report.json"))
        reports = load(os.path.join(base, "contract_reports.json"))

        print("#" * 88)
        print(f"{paper}/{mode}")
        if payload is None:
            print("  no final_mapped.json -- leg produced no canonical payload")
            verdict[f"{paper}/{mode}"] = "no_payload"
            print()
            continue

        if gate is not None:
            print(f"  final_stage3_gate_report : ok={gate.get('ok')} "
                  f"errors={len(gate.get('errors') or [])} phase={gate.get('phase')}")
        if reports:
            for key, value in reports.items():
                if not isinstance(value, dict) or key.endswith("runtime_schema_report"):
                    continue
                if value.get("ok") is False or value.get("errors"):
                    print(f"  FAILING CONTRACT REPORT  : {key} "
                          f"phase={value.get('phase')} stage={value.get('stage')} "
                          f"errors={len(value.get('errors') or [])}")
                    for err in (value.get("errors") or []):
                        detail = err if isinstance(err, str) else (
                            err.get("message") or err.get("reason") or json.dumps(err)[:160]
                        )
                        pointer = "" if isinstance(err, str) else (
                            err.get("pointer") or err.get("path") or "")
                        print(f"      - {pointer} {detail}")

        proteins = ((payload.get("entities") or {}).get("proteins") or [])
        print(f"  proteins in shipped payload: {len(proteins)}")
        still_failing = 0
        for idx, row in enumerate(proteins):
            name = row.get("name")
            ext = protein_external_identity(row)
            species = protein_species_context(row)
            present = [k for k in IDENTITY_KEYS if row.get(k) not in (None, "", {}, [])]
            flag = ""
            if not ext:
                flag += "  <-- STILL MISSING IDENTIFIER"
                still_failing += 1
            if not species:
                flag += "  <-- STILL MISSING SPECIES"
                still_failing += 1
            print(f"    [{idx}] {str(name):<26} identity={str(ext)[:38]:<40} "
                  f"species={str(species)[:22]:<24} keys={present}{flag}")

        complexes = ((payload.get("entities") or {}).get("protein_complexes") or [])
        if complexes:
            print(f"  protein_complexes: "
                  f"{[c.get('name') for c in complexes]}")

        verdict[f"{paper}/{mode}"] = (
            "shipped payload STILL fails" if still_failing else "shipped payload PASSES"
        )
        print(f"  => {verdict[f'{paper}/{mode}']} "
              f"({still_failing} row-level objection(s) on the final payload)")
        print()

    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    for leg, value in verdict.items():
        print(f"  {leg:<28} {value}")
    print()
    print("READING THE RESULT:")
    print("  'shipped payload PASSES' on a leg the run FAILED means the failure came from a")
    print("  superseded audit_round report, not from the payload that would have been")
    print("  exported -- the STALE reading, and PRODUCT_CONTRACT section 1 names a stale gate")
    print("  report an unacceptable terminal blocker.")
    print("  'shipped payload STILL fails' would mean the gate is right and")
    print("  final_stage3_gate_report is blind -- the KEY MISMATCH reading, a worse defect.")
    print()
    print("  NEITHER reading is on its own a licence to make these legs pass. Whether they")
    print("  SHOULD export is a separate question answered by the gold set, not by this probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
