"""Removed-entity rows and quarantine iterations for the referenced proteins. Read-only, stdlib."""
import json
from collections import Counter
D = r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict"
r = json.load(open(f"{D}/removed_entity_report.json", encoding="utf-8"))
print("counts", r.get("counts"))
want = {"c-jun", "mrlc", "profilin", "jnk", "rock", "dia1", "par6", "smurf", "frizzled", "frizzled (fz)"}
for it in r.get("removed_entities", []):
    if str(it.get("name", "")).lower() in want:
        print(json.dumps(it, ensure_ascii=False)[:400])
q = json.load(open(f"{D}/quarantine_report.json", encoding="utf-8"))
print("iterations", Counter((i.get("iteration"), i.get("reason")) for i in q["quarantined"]))
adm = q.get("admissions")
print("admissions type", type(adm).__name__, (list(adm.keys())[:10] if isinstance(adm, dict) else len(adm)))
