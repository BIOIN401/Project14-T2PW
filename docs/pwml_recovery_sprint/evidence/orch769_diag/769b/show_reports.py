"""Dump report structures relevant to the 5 names. Read-only."""
import json, sys
from pathlib import Path

LEG = Path(sys.argv[1])
NAMES = {"pif1", "gbp", "far1", "cpo1", "ppo1"}


def short(o, n=600):
    s = json.dumps(o, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "...(" + str(len(s)) + ")"


fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
ear = fm.get("entity_admission_report") or {}
print("EAR counts:", short(ear.get("counts"), 1000))
print("EAR removed len", len(ear.get("removed") or []), "demoted len", len(ear.get("demoted") or []))
for k in ("removed", "demoted"):
    for r in ear.get(k) or []:
        print(f"  {k}:", short(r, 300))

for f in ("cleaning_report.json", "graph_closure_iterations.json", "audit_iteration_summary.json",
          "coverage_summary.json", "extraction_boundary_report.json"):
    d = json.loads((LEG / f).read_text(encoding="utf-8"))
    print(f"\n==== {f}: keys", list(d.keys()) if isinstance(d, dict) else type(d).__name__)
    print(short(d, 2500))

q = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))
print("\n==== quarantine counts", short(q.get("counts"), 1500))
print("release", short(q.get("release"), 1500))
print("closure keys", list((q.get("closure") or {}).keys()))
print("closure", short(q.get("closure"), 3000))
print("quarantined len", len(q.get("quarantined") or []))
for r in (q.get("quarantined") or [])[:3]:
    print("  Q:", short(r, 700))
print("strict_invariants", short(q.get("strict_invariants"), 1500))
print("review_reasons", short(q.get("review_reasons"), 1500))
