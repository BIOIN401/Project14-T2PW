"""C-121 pre-dispatch end-to-end replay -- READ ONLY on the repo.

Question: with the auto-state lifecycle repaired, do the five F-192-positive
archived payloads actually SERIALIZE, or does the fix only clear a predicate?
Mirrors streamlit_app.py's export sequence: metadata inject -> required-field
gate -> build_pwml_ir -> validate_pwml_ir -> DeterministicPwmlBuilder -> bytes.
Writes only into a scratch directory outside the repository.
"""
import json, sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "c121_replay"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
import t2pw
print("RESOLVED t2pw:", t2pw.__file__)

from lxml import etree
from t2pw.pwml.ir import (build_pwml_ir, validate_pwml_ir,
                          validate_required_pwml_contract)
from t2pw.pwml.writer import (DeterministicPwmlBuilder, blocking_pwml_ir_errors)
from t2pw.pwml.validate import (discover_structure_signature, repair_tree,
                                validate_generated_tree)
from t2pw.pipeline.process_normalizer import ensure_autostates

REF = REPO / "reference" / "PW000001.pwml"
LOC_BUCKETS = ("compound_locations", "protein_locations",
               "nucleic_acid_locations", "element_collection_locations")

LEGS = {
    "PMC11961743": "runs_smoke/2026-09-08_1528/papers/PMC11961743/strict/final_mapped.json",
    "PMC4471609":  "runs_smoke/2026-09-08_1528/papers/PMC4471609/strict/final_mapped.json",
    "PMC9544450_c120": "runs_validation/c120/2026-09-08_1240/papers/PMC9544450/strict/final_mapped.json",
    "PMC12312563": "runs_verify/2026-08-21_2014/papers/PMC12312563/strict/final_mapped.json",
    "PMC13231680": "runs_verify/2026-08-24_1203/papers/PMC13231680/strict/final_mapped.json",
    # controls: currently-passing legs that must NOT change
    "CTRL_PMC9544450_734": "runs_smoke/2026-09-08_1528/papers/PMC9544450/strict/final_mapped.json",
    "CTRL_PMC10269868":    "runs_smoke/2026-09-08_1528/papers/PMC10269868/strict/final_mapped.json",
}

def guard_fires(p):
    states = [r for r in (p.get("biological_states") or []) if isinstance(r, dict)]
    el = p.get("element_locations") or {}
    rows = [r for b in LOC_BUCKETS for r in (el.get(b) or []) if isinstance(r, dict)]
    missing = [r for r in rows if not str(r.get("biological_state") or "").strip()]
    pr = p.get("processes") or {}
    content = bool((pr.get("reactions") or []) or (pr.get("transports") or [])
                   or (pr.get("interactions") or []))
    return content and (not states or bool(missing))

def inject_metadata(p, name):
    md = p.setdefault("metadata", {})
    md.setdefault("pathway_name", f"C121 replay {name}")
    md.setdefault("name", f"C121 replay {name}")
    md.setdefault("pathway_subject", "Metabolic")
    md.setdefault("subject", "Metabolic")
    md.setdefault("width", 3200)
    md.setdefault("height", 1400)
    return p

def codes(rep):
    return sorted({e.get("code") for e in (rep.get("errors") or []) if isinstance(e, dict)})

signature = discover_structure_signature(REF)

for name, rel in LEGS.items():
    path = REPO / rel
    print(f"\n{'='*70}\n### {name}  ({rel})")
    if not path.exists():
        print("  MISSING"); continue
    original = json.loads(path.read_text(encoding="utf-8"))
    fires = guard_fires(original)
    print(f"  guard fires: {fires}")

    for arm in ("base", "repaired"):
        p = deepcopy(original)
        if arm == "repaired":
            if not fires:
                print(f"  [{arm}] guard quiet -> payload untouched (inert by construction)")
            else:
                ensure_autostates(p)
        inject_metadata(p, name)
        gate = validate_required_pwml_contract(deepcopy(p), strict_db=True)
        line = f"  [{arm}] gate ok={gate.get('ok')} codes={codes(gate) or 'none'}"
        if not gate.get("ok"):
            print(line); continue
        try:
            ir, irrep = build_pwml_ir(p, pathway_name=f"C121 replay {name}",
                                      pathway_subject="Metabolic", strict_db=True,
                                      width=3200, height=1400)
            blocking = blocking_pwml_ir_errors(irrep)
            irval = validate_pwml_ir(ir)
            if blocking or irval.get("errors"):
                print(line + f" | IR BLOCKED blocking={len(blocking)} val_errors={len(irval.get('errors') or [])}")
                print("      blocking codes:", sorted({e.get('code') for e in blocking}))
                continue
            args = SimpleNamespace(name=f"C121 replay {name}", description="",
                                   subject="Metabolic", pw_id="PW000000",
                                   height=1400, width=3200,
                                   background_color="#FFFFFF", ref=str(REF))
            builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args)
            build_result = builder.build()
            tree = etree.ElementTree(build_result.root)
            repaired_tree = repair_tree(tree, signature)
            vreport = validate_generated_tree(repaired_tree, signature)
            xml = etree.tostring(repaired_tree.getroot(), encoding="utf-8",
                                 xml_declaration=True, pretty_print=True)
            outp = OUT / f"{name}.{arm}.pwml"
            outp.write_bytes(xml)
            print(line + f" | PWML {len(xml)} bytes -> {outp.name}"
                  f" | tree_errors={len(vreport.get('errors') or [])}"
                  f" | counts={ {k:v for k,v in (build_result.counts or {}).items() if v} }")
        except Exception as exc:
            print(line + f" | RAISED {type(exc).__name__}: {str(exc)[:200]}")
print("\nREPLAY DONE")
