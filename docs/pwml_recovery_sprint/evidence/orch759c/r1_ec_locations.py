"""ORCH-759c round 1: where do the leg's element_collection_locations rows naming complexes come from?

Stdlib only, on the committed fixture (the leg's final_mapped.json, verbatim) plus
the leg's stage1/merged payloads when present. Prints only.
"""
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FIX = ROOT / "tests" / "fixtures" / "orch759c"
LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
NAMES = ["Mg-chelatase complex", "Mg-Proto ME cyclase complex"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def report(tag, d):
    print(f"######## {tag}")
    ents = d.get("entities") or {}
    locs = d.get("element_locations") or {}
    print("  element_collections:", [r.get("name") for r in ents.get("element_collections") or []])
    for bucket in ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections"):
        for r in ents.get(bucket) or []:
            if norm(r.get("name")) in {norm(n) for n in NAMES}:
                print(f"  entity {bucket}: {r.get('name')!r} class={r.get('class')!r} components={json.dumps(r.get('components'), ensure_ascii=False)[:200]}")
    for bucket, rows in locs.items():
        for r in rows or []:
            if isinstance(r, dict) and any(norm(v) in {norm(n) for n in NAMES} for v in r.values() if isinstance(v, str)):
                print(f"  location {bucket}: {json.dumps(r, ensure_ascii=False)[:300]}")


report("final_mapped.json (fixture)", json.loads(gzip.decompress((FIX / "chlorophyll_PMC5130987_final_mapped.json.gz").read_bytes())))
for name in ("merged_payload.json", "stage1_payload.json"):
    if (LEG / name).is_file():
        report(name, json.loads((LEG / name).read_text(encoding="utf-8")))
