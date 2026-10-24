"""Are the participants quarantine called undeclared present in merged_payload.json? Read-only, stdlib."""
import json

D = r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict"
merged = json.load(open(f"{D}/merged_payload.json", encoding="utf-8"))
final = json.load(open(f"{D}/final_mapped.json", encoding="utf-8"))
q = json.load(open(f"{D}/quarantine_report.json", encoding="utf-8"))


def names(p):
    out = {}
    for bucket, rows in (p.get("entities") or {}).items():
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict):
                for k in ("name", "raw_name", "original_name"):
                    if r.get(k):
                        out[str(r[k]).lower()] = bucket
                for a in r.get("aliases") or r.get("synonyms") or []:
                    out[str(a).lower()] = bucket
    return out


mn = names(merged)
print("quarantine admitted_payload_hash", q.get("admitted_payload_hash"))
for it in q["quarantined"]:
    if not str(it.get("reason", "")).startswith("undeclared"):
        continue
    part = (it.get("detail") or {}).get("participant") or it.get("essential_participant")
    print(f"{it['process_kind']:11s} {it['reason']:32s} {part!r:45s} in merged: {mn.get(str(part).lower(), '-')}")
