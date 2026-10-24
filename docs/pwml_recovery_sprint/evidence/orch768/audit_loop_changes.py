"""Did a post-merged_payload LLM stage (audit loop, gap resolver) change each tier-A leg? Stdlib, read-only."""
import json
from pathlib import Path

P = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
M = P / ".claude/worktrees/orch768/docs/pwml_recovery_sprint/evidence/orch768_corpus_manifest.json"
m = json.load(open(M, encoding="utf-8"))
changed = 0
n = 0
for leg in m["legs"]:
    if leg.get("tier") != "A" or not leg.get("checkpoint"):
        continue
    d = P / leg["leg_dir"]
    rounds = (json.load(open(d / "audit_iteration_summary.json", encoding="utf-8")).get("rounds") or []) \
        if (d / "audit_iteration_summary.json").is_file() else []
    acc = sum(int(r.get("accepted_patch_count") or 0) for r in rounds)
    flags = {k: any(bool(r.get(k)) for r in rounds) for k in ("audit_payload_changed", "gap_payload_changed", "settled_payload_changed")}
    n += 1
    changed += any(flags.values()) or acc > 0
    print(f"{leg['id']:50s} rounds={len(rounds)} accepted_patches={acc} {flags}")
print(f"legs where the audit/gap loop changed the payload: {changed}/{n} (curator effect is not recorded in the leg)")
