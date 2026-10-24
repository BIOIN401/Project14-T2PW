"""Show final_mapped quarantined_proteins / protein_export_policy and reactions 17,18,21,22,23. Read-only."""
import json, sys
from pathlib import Path

LEG = Path(sys.argv[1])
fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
NAMES = {"pif1", "gbp", "far1", "cpo1", "ppo1"}


def short(o, n=400):
    s = json.dumps(o, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "...(" + str(len(s)) + ")"


qp = fm.get("quarantined_proteins")
print("quarantined_proteins type:", type(qp).__name__, "len", len(qp) if hasattr(qp, "__len__") else "")
if isinstance(qp, list):
    for r in qp:
        nm = str((r or {}).get("name", "")).lower() if isinstance(r, dict) else str(r).lower()
        if nm in NAMES:
            print("  QP:", short(r, 1500))
    print("  all names:", [ (r.get("name") if isinstance(r, dict) else r) for r in qp])
elif isinstance(qp, dict):
    print("  keys:", list(qp.keys())[:40])
    print(short(qp, 4000))
print("\nprotein_export_policy:", short(fm.get("protein_export_policy"), 3000))
print("\nentity_admission_report keys:", list((fm.get("entity_admission_report") or {}).keys()) if isinstance(fm.get("entity_admission_report"), dict) else type(fm.get("entity_admission_report")))

rx = fm["processes"]["reactions"]
for i in (17, 18, 21, 22, 23):
    r = rx[i]
    print(f"\n--- reaction {i}: id={r.get('id')} name={r.get('name')}")
    print("  inputs:", [x.get("entity") if isinstance(x, dict) else x for x in r.get("inputs", [])])
    print("  outputs:", [x.get("entity") if isinstance(x, dict) else x for x in r.get("outputs", [])])
    for k in ("enzymes", "modifiers"):
        for j, m in enumerate(r.get(k, []) or []):
            print(f"  {k}[{j}]:", short({kk: vv for kk, vv in m.items() if kk != 'evidence'}, 600))
    other = {k: v for k, v in r.items() if k not in ("inputs", "outputs", "enzymes", "modifiers", "evidence")}
    print("  other:", short(other, 1500))

ents = fm["entities"]
for b, rows in ents.items():
    for i, e in enumerate(rows or []):
        if isinstance(e, dict) and str(e.get("name", "")).lower() in NAMES:
            print(f"\nENTITY {b}[{i}]:", short(e, 1500))
print("\nentity buckets:", {b: len(v or []) for b, v in ents.items()})
