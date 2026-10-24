"""ORCH-769b: does apply_audit_patch's referential-integrity guard let an audit remove a protein row
that a reaction modifier names, when a same-named nucleic_acids row survives? Uses the leg's
final_mapped.json with a PIF1/CPO1 protein row put back (the pre-audit shape). No LLM, no writes."""
import json
import sys
from copy import deepcopy
from pathlib import Path

LEG = Path(sys.argv[1] if len(sys.argv) > 1 else
           "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/"
           "ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict")

import t2pw.curation.apply_audit_patch as ap

fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
for name in ("PIF1", "CPO1"):
    before = deepcopy(fm)
    before["entities"]["proteins"].append({"name": name, "class": "protein"})
    idx = len(before["entities"]["proteins"]) - 1
    op = {"op": "remove", "path": f"/entities/proteins/{idx}", "confidence": 0.99}
    after = deepcopy(before)
    after["entities"]["proteins"].pop(idx)
    print(f"{name}: remove protein row, gene row kept   ->", ap._referential_integrity_rejection(before, after, op, set()))
    after2 = deepcopy(after)
    after2["entities"]["nucleic_acids"] = [r for r in after2["entities"]["nucleic_acids"] if r.get("name") != name]
    print(f"{name}: remove protein row AND gene row    ->", ap._referential_integrity_rejection(before, after2, op, set()))
    # Full apply path, one op, so the batch look-ahead also runs.
    out, rep = ap.apply_patch_with_policy(before, [op])
    still = [r.get("name") for r in out["entities"]["proteins"] if r.get("name") == name]
    print(f"{name}: apply_patch_with_policy accepted={rep.get('accepted_count', rep.get('summary'))!s:.120} protein row survives={bool(still)}")
