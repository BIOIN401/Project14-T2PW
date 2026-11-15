"""A4: Unknown-only protein complexes per sample pathway, with the recorded mapping failure reason."""
import json, collections
from pathlib import Path

ROOT = Path("C:/t/corr-assess")
sample = json.loads((ROOT / "sample.json").read_text(encoding="utf-8"))
tot = unk = 0
reasons = collections.Counter()
per = {}
for s in sample:
    d = json.loads((ROOT / "base" / str(s["idx"]) / "final_payload.json").read_text(encoding="utf-8"))
    rows = []
    for pc in d["entities"].get("protein_complexes") or []:
        tot += 1
        comps = [c.get("name") if isinstance(c, dict) else c for c in pc.get("components") or []]
        if comps and all(str(c) == "Unknown" for c in comps):
            unk += 1
            mm = pc.get("mapping_meta") or {}
            r = mm.get("real_mapping_failure_reason")
            reasons[str(r)] += 1
            rows.append([pc.get("name"), r])
    sp = sorted({p.get("species") for p in d["entities"].get("proteins") or [] if p.get("name") == "Unknown"})
    per[s["idx"]] = {"organism": s["organism"], "unknown_only_complexes": rows, "unknown_protein_species": sp,
                     "complexes_total": len(d["entities"].get("protein_complexes") or [])}
print(json.dumps({"complexes_total": tot, "unknown_only": unk, "failure_reasons": reasons, "per": per}, indent=1, ensure_ascii=False))
