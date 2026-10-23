"""Replay the chlorophyll-strict quarantine on the leg's own final_mapped.json (no LLM, no writes).

A. baseline at f392d447 code: must reproduce admitted/decision/resulting hashes and
   refusal_reasons ["entity_type_overlap:4"].
B. counterfactual = the narrow fix candidate: a protein-complex COMPONENT name keeps
   only rows in the ``proteins`` bucket alive during closure; every other bucket
   survives only if a surviving process references it. Then run the final
   pre-export gate exactly as freeze_canonical_payload does
   (``enforce_all_proteins_connected=True`` for strict).
Prints only; writes nothing.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict")

import t2pw.pipeline.strict_quarantine as sq
from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates

fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
qrep = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))
ctx = qrep["coverage"]["requested_context"]

print("admitted_payload_hash(final_mapped) =", sq.admitted_payload_hash(fm), "| leg:", qrep["admitted_payload_hash"])


def summarize(tag, res):
    r = res.quarantine_report
    print(f"\n== {tag}")
    for k in ("admitted_payload_hash", "decision_input_hash", "resulting_payload_hash", "ok", "refusal_reasons", "review_reasons", "counts"):
        print(f"  {k}: {r.get(k)}")
    print("  strict_invariants:", json.dumps(r.get("strict_invariants")))
    rel = r.get("release") or {}
    print("  release.status:", rel.get("status"), "| reasons:", rel.get("reasons"), "| semantic_failed:", rel.get("semantic_failed_checks"), "| review_reasons:", rel.get("review_reasons"))
    removed = res.removed_entity_report.get("removed_entities") or []
    print("  removed entities:", len(removed))
    p = res.payload or {}
    print("  surviving buckets:", {b: len(v) for b, v in (p.get("entities") or {}).items() if isinstance(v, list)},
          {b: len(v) for b, v in (p.get("processes") or {}).items() if isinstance(v, list)})
    return removed


base = sq.quarantine_and_close(deepcopy(fm), strict_db=True, pathway_context=ctx, mode="pathwhiz")
base_removed = summarize("A baseline (f392d447 code)", base)

# ---- B: bucket-aware closure (component names keep proteins only) ----
REC = {}
_orig_ref = sq._referenced_entity_norms
_orig_prune = sq._prune_entities


def rec_ref(payload, admissions):
    out = _orig_ref(payload, admissions)
    REC["referenced"] = set(out)
    return out


def bucket_aware_prune(payload, keep_norms):
    referenced = REC.get("referenced", set(keep_norms))
    ents = payload.get("entities") or {}
    held = {}
    # Run the original on everything; then re-prune non-protein buckets against
    # process references only (a complex component is a protein by construction).
    removed = _orig_prune(payload, keep_norms)
    for bucket in sq._GRAPH_ENTITY_BUCKETS:
        if bucket in ("proteins", "protein_complexes"):
            continue
        rows = ents.get(bucket)
        if not isinstance(rows, list):
            continue
        kept = []
        for row in rows:
            norms = {n for n in sq._entity_name_norms([row]) if n}
            if norms & referenced:
                kept.append(row)
            else:
                removed.append({"bucket": bucket, "name": sq._row_name(row), "reason": "unreferenced_after_quarantine", "had_external_identity": None, "_counterfactual": True})
        if len(kept) != len(rows):
            ents[bucket] = kept
    return removed


sq._referenced_entity_norms = rec_ref
sq._prune_entities = bucket_aware_prune
try:
    cf = sq.quarantine_and_close(deepcopy(fm), strict_db=True, pathway_context=ctx, mode="pathwhiz")
finally:
    sq._referenced_entity_norms = _orig_ref
    sq._prune_entities = _orig_prune
cf_removed = summarize("B counterfactual (bucket-aware component keep)", cf)
extra = [r for r in cf_removed if r.get("_counterfactual")]
print("  rows removed ONLY by the counterfactual:", [(r["bucket"], r["name"]) for r in extra])
base_set = {(r["bucket"], r["name"]) for r in base_removed}
cf_set = {(r["bucket"], r["name"]) for r in cf_removed}
print("  removed in A not in B:", sorted(base_set - cf_set))
print("  removed in B not in A:", sorted(cf_set - base_set))
acc = lambda res: sorted((r["pointer"], r["state"]) for r in res.admissions)
print("  admissions identical A vs B:", acc(base) == acc(cf))

if cf.ok:
    try:
        details = run_strict_post_normalization_gates(deepcopy(cf.payload), enforce_all_proteins_connected=True)
        print("\n== B final pre-export gate: PASS; keys:", sorted(details.keys())[:12])
        print("   connectivity:", json.dumps(details.get("connectivity"))[:600])
    except Exception as exc:  # GateValidationError carries .details
        det = getattr(exc, "details", {}) or {}
        errs = det.get("errors") or []
        print(f"\n== B final pre-export gate: FAIL {type(exc).__name__}: {exc}; {len(errs)} error(s)")
        for e in errs[:25]:
            print("   ", json.dumps(e, ensure_ascii=False)[:400])
else:
    print("\n== B still refused; final gate not run")
sys.stdout.flush()
