import json
d = r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict"
q = json.load(open(f"{d}/quarantine_report.json", encoding="utf-8"))
qs = q["quarantined"]
print(json.dumps(qs[0], indent=1)[:1500])
for it in qs:
    k = it.get("kind") or it.get("process_kind")
    if k in ("reaction", "transport"):
        print(k, it.get("reason"), "|", json.dumps({kk: it.get(kk) for kk in ("name", "label", "detail", "entity", "missing", "offending")}, ensure_ascii=False)[:300])
