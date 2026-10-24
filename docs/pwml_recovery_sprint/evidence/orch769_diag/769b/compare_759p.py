"""Compare the ORCH-759p (PWML written) and ORCH-767 (failed) chlorophyll strict legs. Read-only."""
import json, sys
from pathlib import Path

ROOT = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
LEGS = {
    "759p": ROOT / "runs_papers/ORCH-759p/legs/chlorophyll-strict/2026-10-22_2301/papers/PMC5130987/strict",
    "767": ROOT / ".claude/worktrees/orch767/runs_papers/ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict",
}
NAMES = ["PIF1", "GBP", "FAR1", "CPO1", "PPO1"]


def load(leg, n):
    p = LEGS[leg] / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def buckets_with(payload, name):
    out = []
    for b, rows in (payload.get("entities") or {}).items():
        for r in rows or []:
            if isinstance(r, dict) and str(r.get("name", "")).strip().lower() == name.lower():
                out.append(b)
    return out


def actor_refs(payload, name):
    out = []
    for i, r in enumerate((payload.get("processes") or {}).get("reactions") or []):
        for k in ("enzymes", "modifiers"):
            for j, m in enumerate(r.get(k) or []):
                if isinstance(m, dict):
                    n = m.get("entity") or m.get("protein") or m.get("name")
                    if isinstance(n, str) and n.strip().lower() == name.lower():
                        out.append(f"r{i}.{k}[{j}]")
    return out


for leg in LEGS:
    print(f"\n######## {leg}: {LEGS[leg]}")
    print("files:", sorted(p.name for p in LEGS[leg].iterdir()))
    res = (LEGS[leg] / "RESULT.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(l for l in res[:12]))
    for f in ("stage1_payload.json", "merged_payload.json", "final_mapped.json"):
        d = load(leg, f)
        if d is None:
            print(f, "missing"); continue
        p = d["processes"]
        print(f"  {f}: rx={len(p.get('reactions') or [])} ix={len(p.get('interactions') or [])} prot={len(d['entities'].get('proteins') or [])} na={len(d['entities'].get('nucleic_acids') or [])}")
        for n in NAMES:
            print(f"     {n}: buckets={buckets_with(d, n)} actor_refs={actor_refs(d, n)}")
    cr = load(leg, "contract_reports.json") or {}
    for key in ("post_normalization_contract_report",):
        rep = cr.get(key) or {}
        print(f"  {key} phase={rep.get('phase')} ok={rep.get('ok')}")
        for e in rep.get("errors") or []:
            if isinstance(e, dict) and e.get("path"):
                print("     ", e.get("path"), "|", str(e.get("reason"))[:90])
    au = load(leg, "audit_iteration_summary.json") or {}
    for r in au.get("rounds") or []:
        print("  audit round", r.get("round"), "patches", r.get("patch_count"), "acc", r.get("accepted_patch_count"),
              "rej", r.get("rejected_patch_count"), "|", str(r.get("llm_repair_rationale"))[:300])
    q = load(leg, "quarantine_report.json") or {}
    print("  quarantine counts", q.get("counts"), "release", (q.get("release") or {}).get("status"))
    g = load(leg, "final_stage3_gate_report.json") or {}
    print("  final gate ok", g.get("ok"), "errors", len(g.get("errors") or []))
    ex = load(leg, "extraction_boundary_report.json") or {}
    print("  stage1 response hashes:", [b.get("response_hash") for b in ex.get("boundaries") or [] if b.get("boundary") == "stage1_extraction"])
