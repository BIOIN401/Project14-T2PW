"""Stdlib-only: quarantine admission records touching the overlap names; interaction row shapes."""
import json, re
from collections import Counter
from pathlib import Path

LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")
NAMES = {"cao", "hema1", "gun4", "porb", "alad1hemb1", "chlh", "pora", "porc"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


q = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))
print("counts", q["counts"])
print("reasons over all admissions:", Counter((r["state"], r["reason"].split(":")[0]) for r in q["admissions"]))
for r in q["admissions"]:
    terms = {norm(t) for t in r.get("core_terms") or []} | {norm(r.get("essential_participant"))}
    if terms & NAMES or "ambiguous" in r["reason"]:
        print(r["pointer"], "|", r["name"], "|", r["state"], "|", r["reason"], "| ess:", r.get("essential_participant"), "| it", r.get("iteration"))

fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
ints = fm["processes"].get("interactions") or []
print("\ninteraction row keys:", sorted(ints[0].keys()) if ints else None)
for i, row in enumerate(ints[:3]):
    print(i, json.dumps(row, ensure_ascii=False)[:900])
rx = fm["processes"]["reactions"]
for i, row in enumerate(rx):
    if norm(row.get("name")).startswith("glutr") or "HEMA1" in json.dumps(row):
        print("RX", i, json.dumps(row, ensure_ascii=False)[:900])
        break
prots = fm["entities"]["proteins"]
for row in prots:
    if norm(row.get("name")) in {"hema1", "cao", "gun4", "porb"}:
        print("\nPROT", row.get("name"), "keys:", sorted(row.keys()))
        print("  ids:", {k: row.get(k) for k in row if "id" in k.lower() or k in ("species", "uniprot", "accession")})
        mm = row.get("mapping_meta") or {}
        print("  mapping_meta:", json.dumps(mm, ensure_ascii=False)[:1500])
na = fm["entities"]["nucleic_acids"]
print("\nNA row full keys:", sorted(na[3].keys()), json.dumps(na[3], ensure_ascii=False)[:1200])
