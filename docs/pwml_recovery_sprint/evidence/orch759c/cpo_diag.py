"""Scratch: why is CPO a bare protein enzyme in the chlorophyll leg's final_mapped?"""
import gzip
import json
from pathlib import Path

FIX = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch759c/tests/fixtures/orch759c")
p = json.loads(gzip.decompress((FIX / "chlorophyll_PMC5130987_final_mapped.json.gz").read_bytes()))
for b in ("proteins", "protein_complexes"):
    for i, r in enumerate(p["entities"][b]):
        if "CPO" in json.dumps({k: r.get(k) for k in ("name", "synonyms", "components")}):
            print(b, i, json.dumps({k: r.get(k) for k in ("name", "synonyms", "components", "species", "uniprot_id", "mapped_ids", "generated", "generation_reason", "issues")}, ensure_ascii=False)[:900])
            print("   mapping_meta:", json.dumps(r.get("mapping_meta"), ensure_ascii=False)[:900])
for i, r in enumerate(p["processes"]["reactions"]):
    if "CPO" in json.dumps(r.get("enzymes")) or "CPO" in json.dumps(r.get("modifiers")):
        print("reaction", i, r.get("name"), json.dumps(r.get("enzymes"))[:300], json.dumps(r.get("modifiers"))[:300])
