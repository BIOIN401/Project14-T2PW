"""Why were the referenced proteins removed before quarantine? Read-only, stdlib."""
import json
D = r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict"
f = json.load(open(f"{D}/final_mapped.json", encoding="utf-8"))
qp = f.get("quarantined_proteins")
print(type(qp).__name__, len(qp) if hasattr(qp, "__len__") else qp)
rows = qp if isinstance(qp, list) else (qp or {}).get("rows", []) if isinstance(qp, dict) else []
if isinstance(qp, dict) and not rows:
    print(list(qp.keys())[:10])
want = {"c-jun", "mrlc", "profilin", "jnk", "rock", "dia1", "par6", "smurf", "frizzled", "frizzled (fz)"}
for r in rows:
    if isinstance(r, dict) and str(r.get("name", "")).lower() in want:
        print(json.dumps({k: r.get(k) for k in ("name", "reason", "quarantine_reason", "status", "uniprot_id", "mapping_status")}, ensure_ascii=False)[:300])
pol = f.get("protein_export_policy")
print("protein_export_policy", json.dumps(pol, ensure_ascii=False)[:600])
r = json.load(open(f"{D}/removed_entity_report.json", encoding="utf-8"))
print("removed_entity_report keys", list(r.keys())[:15])
for it in (r.get("removed") or r.get("entities") or [])[:60]:
    if isinstance(it, dict) and str(it.get("name", "")).lower() in want:
        print("  removed", json.dumps(it, ensure_ascii=False)[:300])
