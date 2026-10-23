"""Stdlib-only: what keeps the NA rows CAO/HEMA1/GUN4/PORB alive through closure."""
import json, re
from pathlib import Path

LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
T = {"cao", "hema1", "gun4", "porb"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
q = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))
accepted = {(r["bucket"], r["index"]) for r in q["admissions"] if r["state"] in ("core_accepted", "auxiliary_accepted")}
cx_with = {}
for c in fm["entities"]["protein_complexes"]:
    comps = [cc.get("name") if isinstance(cc, dict) else cc for cc in c.get("components") or []]
    hits = [x for x in comps if norm(x) in T]
    if hits:
        cx_with[norm(c["name"])] = (c["name"], comps, c.get("generation_reason"))
for k, v in cx_with.items():
    print("complex", v)
for bucket, rows in fm["processes"].items():
    for i, row in enumerate(rows or []):
        s = json.dumps(row)
        refs = [nm for k, (nm, _, _) in cx_with.items() if f'"{nm}"' in s]
        direct = [t for t in ("CAO", "HEMA1", "GUN4", "PORB") if f'"{t}"' in s]
        if refs or direct:
            print(f"{bucket}[{i}] accepted={(bucket, i) in accepted} {row.get('name')!r} complex_refs={refs} direct={direct}")
# the 4 NA rows' lineage: where did they come from
for na in fm["entities"]["nucleic_acids"]:
    if norm(na["name"]) in T:
        print("NA", na["name"], [(e.get("stage"), e.get("origin")) for e in na.get("provenance_lineage") or []])
