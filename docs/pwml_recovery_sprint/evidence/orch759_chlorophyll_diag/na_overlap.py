"""Stdlib-only: where do the proteins+nucleic_acids name overlaps come from, per artifact."""
import json, re, sys
from pathlib import Path

LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
NAMES = ["CAO", "HEMA1", "GUN4", "PORB", "ALAD1/HEMB1", "ALAD1", "HEMB1"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}/{i}")
    else:
        yield path, o


def show(fname):
    p = LEG / fname
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n######## {fname}")
    ents = d.get("entities") or {}
    print("buckets:", {k: len(v) for k, v in ents.items() if isinstance(v, list)})
    for row_i, row in enumerate(ents.get("nucleic_acids") or []):
        print(f"  NA[{row_i}]", json.dumps({k: row.get(k) for k in ("name", "type", "species", "provenance", "confidence", "evidence", "source", "rag_provenance", "generation_reason", "external_ids", "aliases")}, ensure_ascii=False)[:700])
    targets = {norm(n) for n in NAMES}
    for row_i, row in enumerate(ents.get("proteins") or []):
        if norm(row.get("name")) in targets:
            print(f"  PROT[{row_i}]", row.get("name"), "| uniprot:", row.get("uniprot_id") or (row.get("external_ids") or {}).get("uniprot"), "| aliases:", row.get("aliases"))
    # references to NA names inside processes
    procs = d.get("processes") or {}
    na_names = {norm(r.get("name")) for r in ents.get("nucleic_acids") or [] if isinstance(r, dict)}
    for bucket, rows in procs.items():
        if not isinstance(rows, list):
            continue
        for i, row in enumerate(rows):
            hits = []
            for path, val in walk(row):
                if isinstance(val, str) and norm(val) in na_names and norm(val):
                    hits.append(f"{path}={val}")
            if hits:
                print(f"  proc {bucket}[{i}] {row.get('name')!r}: {hits[:8]}")


for f in sys.argv[1:] or ["stage1_payload.json", "merged_payload.json", "final_mapped.json"]:
    show(f)
