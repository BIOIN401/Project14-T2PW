"""C-121 orchestrator measurement spike -- READ ONLY, base tree.

Confirms, against production functions rather than committed text:
  1. what the required-field gate actually says on the two archived payloads;
  2. whether re-establishing the auto-state alone clears it;
  3. whether a PWML IR can then be built at all.
Nothing is written into the repository. No LLM, no network.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
sys.path.insert(0, str(REPO / "src"))

import t2pw
print("RESOLVED t2pw:", t2pw.__file__)

from t2pw.pwml.ir import validate_required_pwml_contract, build_pwml_ir
from t2pw.pipeline.process_normalizer import ensure_autostates

LEGS = [
    ("PMC11961743", REPO / "runs_smoke/2026-09-08_1528/papers/PMC11961743/strict/final_mapped.json"),
    ("PMC4471609",  REPO / "runs_smoke/2026-09-08_1528/papers/PMC4471609/strict/final_mapped.json"),
    ("PMC9544450",  REPO / "runs_smoke/2026-09-08_1528/papers/PMC9544450/strict/final_mapped.json"),
]

def codes(rep):
    return sorted({e.get("code") for e in (rep.get("errors") or []) if isinstance(e, dict)})

for name, path in LEGS:
    if not path.exists():
        print(f"\n### {name}: MISSING {path}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    el = payload.get("element_locations") or {}
    print(f"\n### {name}")
    print("  element_locations:", {k: len(v) for k, v in el.items() if isinstance(v, list)})
    print("  biological_states:", [r.get("name") for r in (payload.get("biological_states") or []) if isinstance(r, dict)])
    rx = ((payload.get("processes") or {}).get("reactions") or [])
    print("  reactions:", len(rx))

    base = validate_required_pwml_contract(deepcopy(payload), strict_db=True)
    print("  BASE gate ok:", base.get("ok"), "error codes:", codes(base))

    # Hypothetical repair: re-run the existing production pass, nothing else.
    repaired = deepcopy(payload)
    rep = ensure_autostates(repaired)
    print("  after ensure_autostates -> states:",
          [r.get("name") for r in (repaired.get("biological_states") or []) if isinstance(r, dict)])
    after = validate_required_pwml_contract(deepcopy(repaired), strict_db=True)
    print("  REPAIRED gate ok:", after.get("ok"), "error codes:", codes(after))

    if after.get("ok"):
        try:
            ir, irrep = build_pwml_ir(deepcopy(repaired), pathway_name=f"C121 spike {name}", strict_db=True)
            errs = sorted({e.get("code") for e in (irrep.get("errors") or []) if isinstance(e, dict)})
            print("  IR build errors:", errs or "none")
            print("  IR compound_locations:", len(ir.get("compound_locations") or []),
                  "protein_locations:", len(ir.get("protein_locations") or []),
                  "reactions:", len(ir.get("reactions") or []))
        except Exception as exc:
            print("  IR build RAISED:", type(exc).__name__, str(exc)[:300])
print("\nSPIKE DONE")
