"""ORCH-769b: counterfactual F1' -- _actor_verdict uses the final gate's exact actor predicate
(normalized actor name in name+synonym norms of proteins | protein_complexes), instead of the
resolved bucket. Replays on the leg's final_mapped.json. No LLM, no writes."""
import json
import sys
from copy import deepcopy
from pathlib import Path

LEG = Path(sys.argv[1] if len(sys.argv) > 1 else
           "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/"
           "ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict")

import t2pw.pipeline.strict_quarantine as sq
import t2pw.pipeline.process_normalizer as pn

fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
qrep = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))
ctx = qrep["coverage"]["requested_context"]

_orig = sq._actor_verdict
STATE = {}


def actor_verdict_gate_pred(bucket, process, registry, *, strict_db):
    v = _orig(bucket, process, registry, strict_db=strict_db)
    if v is not None:
        return v
    norms = STATE["protein_norms"]
    for slot in sq._ACTOR_SLOTS.get(bucket, ()):
        for actor in sq._safe_list(process.get(slot)):
            if not isinstance(actor, dict):
                continue
            name = pn._actor_name_from_row(actor)
            if name and pn._normalize(name) not in norms:
                return (sq.QUARANTINED_BROKEN_REFERENCE, f"actor_not_in_protein_registry:{slot}", name)
    return None


_orig_admit = sq._admit_processes if hasattr(sq, "_admit_processes") else None


def set_norms(payload):
    ents = payload.get("entities") or {}
    STATE["protein_norms"] = pn._entity_name_norms(ents.get("proteins") or []) | pn._entity_name_norms(ents.get("protein_complexes") or [])


set_norms(fm)
sq._actor_verdict = actor_verdict_gate_pred
try:
    res = sq.quarantine_and_close(deepcopy(fm), strict_db=True, pathway_context=ctx, mode="pathwhiz")
finally:
    sq._actor_verdict = _orig
r = res.quarantine_report
print("counts:", r["counts"], "| ok:", r["ok"], "| release:", (r.get("release") or {}).get("status"))
print("quarantined reactions:", [(a["index"], a["reason"], a["essential_participant"]) for a in res.admissions
                                 if a["bucket"] == "reactions" and not str(a["state"]).endswith("accepted")])
try:
    pn.run_strict_post_normalization_gates(deepcopy(res.payload), enforce_all_proteins_connected=True)
    print("final gate: PASS")
except Exception as exc:
    errs = (getattr(exc, "details", {}) or {}).get("errors") or []
    print("final gate: FAIL", len(errs), [(e.get("path"), e.get("reason")) for e in errs][:10])
# NOTE: protein_norms is computed once from the input payload; closure revalidation sees a pruned
# registry, so a production fix must compute it from the registry the verdict is handed.
