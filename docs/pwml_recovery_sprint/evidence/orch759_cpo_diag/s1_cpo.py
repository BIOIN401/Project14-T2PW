"""Stdlib-only: show every CPO-bearing row in stage1/merged/final_mapped of the chlorophyll leg."""
import json, sys
from pathlib import Path
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
def walk(p, label):
    print("=====", label)
    ents = p.get("entities", {})
    for b, rows in ents.items():
        if not isinstance(rows, list):
            continue
        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                continue
            s = json.dumps({k: r.get(k) for k in ("name", "synonyms", "components", "gene_name")}, ensure_ascii=False)
            if "CPO" in s or "oprogen" in s.lower() and "oxidase" in s.lower():
                print(b, i, json.dumps({k: r.get(k) for k in ("name", "synonyms", "components", "species", "uniprot_id", "mapped_ids", "generated", "generation_reason", "gene_name", "protein_name")}, ensure_ascii=False)[:1200])
                mm = r.get("mapping_meta")
                if mm:
                    print("   mm:", json.dumps(mm, ensure_ascii=False)[:2500])
    procs = p.get("processes", {})
    for b, rows in procs.items():
        if not isinstance(rows, list):
            continue
        for i, r in enumerate(rows):
            s = json.dumps(r, ensure_ascii=False)
            if "CPO" in s:
                print("PROC", b, i, s[:2500])
for fn in sys.argv[1:]:
    walk(json.loads((LEG / fn).read_text(encoding="utf-8")), fn)
