"""Stdlib-only: full CPO/CPO1 rows (proteins + complexes) in final_mapped.json, mapping_meta keys listed."""
import json
from pathlib import Path
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
p = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
for b in ("proteins", "protein_complexes", "nucleic_acids", "compounds"):
    for i, r in enumerate(p["entities"].get(b) or []):
        if isinstance(r, dict) and str(r.get("name")) in {"CPO", "CPO1", "Unknown"}:
            mm = r.get("mapping_meta") or {}
            print(b, i, "keys:", sorted(r.keys()))
            print("  mm keys:", sorted(mm.keys()))
            slim = {k: v for k, v in r.items() if k != "mapping_meta"}
            print("  row:", json.dumps(slim, ensure_ascii=False)[:1500])
            for k in ("query", "provider", "source", "chosen_rule", "resolution", "ambiguity", "rejected_candidates", "identity_status", "species_resolution", "real_mapping_failure_reason", "placeholder_target_organisms"):
                if k in mm:
                    print("   mm.%s =" % k, json.dumps(mm[k], ensure_ascii=False)[:600])
print("quarantined_proteins:", [q.get("name") for q in p.get("quarantined_proteins") or []])
