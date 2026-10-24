"""Compare the five failing reactions across stage1/merged/final by name. Read-only."""
import json, sys
from pathlib import Path

LEG = Path(sys.argv[1])
NAMES = ["GluTR (HEMA1) reaction", "ALAD (HEMB1) reaction", "Glu-tRNAGlu reduction to GSA",
         "Coprogen III to Protogen IX", "Protogen IX to Proto IX"]


def actor(m):
    if not isinstance(m, dict):
        return m
    n = m.get("entity") or m.get("protein") or m.get("name")
    return f"{n}<{m.get('entity_type') or m.get('type') or ''}:{m.get('role') or ''}>"


for f in ("stage1_payload.json", "merged_payload.json", "final_mapped.json"):
    d = json.loads((LEG / f).read_text(encoding="utf-8"))
    rx = d["processes"]["reactions"]
    print(f"\n===== {f} ({len(rx)} reactions, {len(d['processes'].get('interactions') or [])} interactions)")
    for nm in NAMES:
        for i, r in enumerate(rx):
            if r.get("name") == nm:
                print(f"  [{i}] {nm} lock={r.get('locked_reaction_id')}")
                print("     enz:", [actor(m) for m in r.get("enzymes") or []])
                print("     mod:", [actor(m) for m in r.get("modifiers") or []])
                ks = sorted(set(r.keys()) - {"inputs", "outputs", "enzymes", "modifiers", "evidence", "provenance_lineage"})
                print("     keys:", ks)
