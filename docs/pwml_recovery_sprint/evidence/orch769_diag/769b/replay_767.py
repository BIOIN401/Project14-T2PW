"""ORCH-769b offline replay of the ORCH-767 chlorophyll-strict quarantine + final gate. No LLM, no writes.

A. baseline: quarantine_and_close(final_mapped) with current src, then the final pre-export gate
   (enforce_all_proteins_connected=True, as freeze_canonical_payload does for strict).
A2. same again (determinism).
B. counterfactual = fix candidate F1: strict_quarantine._actor_verdict additionally requires an
   enzymes/modifiers/transporters actor to resolve into proteins/protein_complexes -- the exact
   registry the gate uses (process_normalizer.run_strict_post_normalization_gates
   protein_registry_norms = proteins | protein_complexes).
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

LEG = Path(sys.argv[1] if len(sys.argv) > 1 else
           "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/"
           "ORCH-767/legs/chlorophyll-strict/2026-10-23_2328/papers/PMC5130987/strict")

import t2pw.pipeline.strict_quarantine as sq
from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates

fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
qrep = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))
fgate = json.loads((LEG / "final_stage3_gate_report.json").read_text(encoding="utf-8"))
ctx = qrep["coverage"]["requested_context"]

print("hash(final_mapped)       =", sq.admitted_payload_hash(fm))
print("leg admitted_payload_hash =", qrep["admitted_payload_hash"])
print("leg resulting_payload_hash=", qrep["resulting_payload_hash"])
print("leg decision_input_hash   =", qrep["decision_input_hash"])
print("leg final gate payload_sha256 =", fgate.get("payload_sha256"))


def gate(payload):
    try:
        run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
        return []
    except Exception as exc:  # GateValidationError
        det = getattr(exc, "details", {}) or {}
        return [(e.get("path"), e.get("reason")) for e in det.get("errors") or []]


def run(tag):
    res = sq.quarantine_and_close(deepcopy(fm), strict_db=True, pathway_context=ctx, mode="pathwhiz")
    r = res.quarantine_report
    print(f"\n== {tag}")
    for k in ("admitted_payload_hash", "decision_input_hash", "resulting_payload_hash", "ok", "refusal_reasons", "counts"):
        print(f"  {k}: {r.get(k)}")
    rel = r.get("release") or {}
    print("  release:", rel.get("status"), "| strict_gates_passed:", rel.get("strict_gates_passed"),
          "| completeness:", rel.get("completeness"), "| missing_anchors:", rel.get("missing_anchors"))
    print("  strict_invariants.ok:", (r.get("strict_invariants") or {}).get("ok"))
    q_rx = [(a["index"], a["name"], a["reason"], a.get("essential_participant")) for a in res.admissions
            if a["bucket"] == "reactions" and not str(a["state"]).endswith("accepted")]
    print("  quarantined reactions:", q_rx)
    p = res.payload or {}
    print("  surviving:", {b: len(v) for b, v in (p.get("entities") or {}).items() if isinstance(v, list)},
          {b: len(v) for b, v in (p.get("processes") or {}).items() if isinstance(v, list)})
    errs = gate(p) if res.ok else ["(quarantine refused; final gate not run)"]
    print(f"  final gate errors: {len(errs)}")
    for e in errs[:12]:
        print("    ", e)
    return res, errs


base, base_errs = run("A baseline (current src)")
leg_errs = [(e.get("path"), e.get("reason")) for e in fgate.get("errors") or []]
print("  A errors == leg final_stage3_gate_report errors:", base_errs == leg_errs)
base2, base2_errs = run("A2 baseline re-run")
print("  A2 identical to A (hashes, errors):",
      base.quarantine_report["resulting_payload_hash"] == base2.quarantine_report["resulting_payload_hash"]
      and base_errs == base2_errs)
print("  gate on final_mapped directly:", len(gate(fm)), "errors")

# ---- B: actor must resolve into the gate's protein registry ----
_orig = sq._actor_verdict
_PROTEIN_BUCKETS = ("proteins", "protein_complexes")


def actor_verdict_bucket_aware(bucket, process, registry, *, strict_db):
    v = _orig(bucket, process, registry, strict_db=strict_db)
    if v is not None:
        return v
    for slot in sq._ACTOR_SLOTS.get(bucket, ()):
        for actor in sq._safe_list(process.get(slot)):
            name = sq._actor_name(actor) if not isinstance(actor, str) else sq._canonical(actor)
            resolved = registry.resolve(name) if name else None
            if resolved is not None and resolved[0] not in _PROTEIN_BUCKETS:
                return (sq.QUARANTINED_BROKEN_REFERENCE, f"actor_not_a_protein_in_{slot}:{resolved[0]}", name)
    return None


sq._actor_verdict = actor_verdict_bucket_aware
try:
    cf, cf_errs = run("B counterfactual F1 (actor must resolve to proteins/protein_complexes)")
finally:
    sq._actor_verdict = _orig

# Does each reaction B quarantines have a surviving same-chemistry twin?
def sig(r):
    f = lambda xs: tuple(sorted(str(x.get("entity") if isinstance(x, dict) else x) for x in xs or []))
    return f(r.get("inputs")), f(r.get("outputs"))

surv = cf.payload["processes"]["reactions"]
for a in cf.admissions:
    if a["bucket"] == "reactions" and not str(a["state"]).endswith("accepted"):
        r = fm["processes"]["reactions"][a["index"]]
        twins = [s.get("name") for s in surv if sig(s) == sig(r)]
        print(f"  quarantined r{a['index']} '{r.get('name')}' ({a['core_terms'][:3]}...) twin survivors: {twins}")
removed_b = {(x["bucket"], x["name"]) for x in cf.removed_entity_report.get("removed_entities") or []}
removed_a = {(x["bucket"], x["name"]) for x in base.removed_entity_report.get("removed_entities") or []}
print("  entities removed in B not A:", sorted(removed_b - removed_a))
print("  entities removed in A not B:", sorted(removed_a - removed_b))
cov_a = base.quarantine_report.get("coverage") or {}
cov_b = cf.quarantine_report.get("coverage") or {}
print("  coverage A:", cov_a.get("coverage_ratio"), cov_a.get("unmatched_terms"), "| B:", cov_b.get("coverage_ratio"), cov_b.get("unmatched_terms"))
sys.stdout.flush()
