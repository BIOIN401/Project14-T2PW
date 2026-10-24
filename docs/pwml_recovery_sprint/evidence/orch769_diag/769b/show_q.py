"""Quarantine admissions for reactions naming the 5 actors. Read-only."""
import json, sys
from pathlib import Path

LEG = Path(sys.argv[1])
q = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))


def short(o, n=600):
    s = json.dumps(o, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "...(" + str(len(s)) + ")"


print("admissions type", type(q.get("admissions")).__name__, len(q.get("admissions") or []))
a0 = (q.get("admissions") or [None])[0]
print("admission[0] keys", list(a0.keys()) if isinstance(a0, dict) else a0)
for a in q.get("admissions") or []:
    if a.get("bucket") == "reactions":
        print(" R", a.get("index"), a.get("state"), a.get("reason"), "|", a.get("name"))
print("accepted[0]:", short((q.get("accepted") or [None])[0], 800))
print("quarantined reasons:")
from collections import Counter
print(Counter((r.get("bucket"), r.get("reason")) for r in q.get("quarantined") or []))
for r in q.get("quarantined") or []:
    print("  ", r.get("bucket"), r.get("index"), r.get("name"), "|", r.get("reason"), "|", r.get("essential_participant"))
print("decision_inputs:", short(q.get("decision_inputs"), 1500))
