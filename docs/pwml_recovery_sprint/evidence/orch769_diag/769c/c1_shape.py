import json, sys
base = {
 "767": r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict",
 "759": r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/wnt_pcp-strict/2026-10-22_1432/papers/PMC10425814/strict",
}
for tag, d in base.items():
    print("=====", tag)
    for f in ["final_mapped.json", "merged_payload.json", "stage1_payload.json"]:
        try:
            x = json.load(open(f"{d}/{f}", encoding="utf-8"))
        except Exception as e:
            print(f, "ERR", e); continue
        p = x.get("processes", {}) or {}
        e = x.get("entities", {}) or {}
        print(f, {k: len(v) for k, v in p.items() if isinstance(v, list)}, {k: len(v) for k, v in e.items() if isinstance(v, list)})
    q = json.load(open(f"{d}/quarantine_report.json", encoding="utf-8"))
    r = q.get("release", {})
    print("quarantine ok", q.get("ok"), "counts", json.dumps(q.get("counts"))[:600])
    print("release", r.get("status"), r.get("completeness"), r.get("missing_anchors"), "strict_gates_passed", r.get("strict_gates_passed"))
    print("review_reasons", json.dumps(q.get("review_reasons"))[:800])
    qs = q.get("quarantined", [])
    from collections import Counter
    c = Counter()
    for it in qs:
        if isinstance(it, dict):
            c[(it.get("kind") or it.get("process_kind") or it.get("collection"), it.get("reason") or it.get("code"))] += 1
    print("quarantined", len(qs), c.most_common(15))
    g = json.load(open(f"{d}/final_stage3_gate_report.json", encoding="utf-8"))
    print("final gate ok", g.get("ok"), g.get("connectivity"))
