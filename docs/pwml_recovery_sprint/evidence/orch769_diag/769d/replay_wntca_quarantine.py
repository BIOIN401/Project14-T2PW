"""ORCH-769d: replay the Wnt/Ca2+ strict quarantine on the leg's own final_mapped.json.

No LLM, no network, no writes. Prints only.

A. baseline x2 on current src (== d6849c9e for src/): reproduce hashes + refusal, determinism.
   Also: which keep-norm keeps 'phosphorylated CAMKII' alive through closure.
B. counterfactual: drop every protein synonym that is EXACTLY another protein row's name
   (the cross-row contamination an index-shifted audit patch leaves). Then quarantine and,
   if ok, the final pre-export gate exactly as freeze_canonical_payload runs it (strict).
C. narrowest counterfactual: drop only 'calcineurin B' from 'phosphorylated CAMKII'.
D. synthetic G9 probe of apply_patch_with_policy: remove-then-index-add shifts the add.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

LEG = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_ca-strict/2026-10-24_0314/papers/PMC10425814/strict")

import t2pw.pipeline.strict_quarantine as sq
from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates
from t2pw.curation.apply_audit_patch import apply_patch_with_policy

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
    p = res.payload or {}
    print("  surviving entities:", {b: len(v) for b, v in (p.get("entities") or {}).items() if isinstance(v, list)})
    print("  surviving processes:", {b: len(v) for b, v in (p.get("processes") or {}).items() if isinstance(v, list)})
    print("  surviving proteins:", [x.get("name") for x in (p.get("entities") or {}).get("proteins", [])])


def final_gate(tag, res):
    if not res.ok:
        print(f"\n== {tag} final gate: not run (quarantine refused)")
        return
    try:
        details = run_strict_post_normalization_gates(deepcopy(res.payload), enforce_all_proteins_connected=True)
        print(f"\n== {tag} final pre-export gate: PASS; connectivity:", json.dumps(details.get("connectivity"))[:500])
    except Exception as exc:
        det = getattr(exc, "details", {}) or {}
        errs = det.get("errors") or []
        print(f"\n== {tag} final pre-export gate: FAIL {type(exc).__name__}: {exc}; {len(errs)} error(s)")
        for e in errs[:25]:
            print("   ", json.dumps({k: e.get(k) for k in ("path", "reason", "code")}, ensure_ascii=False)[:300])


# ---- A: baseline x2 + keep-norm attribution ----
REC = {}
_orig_ref = sq._referenced_entity_norms


def rec_ref(payload, admissions):
    out = _orig_ref(payload, admissions)
    REC.setdefault("referenced", []).append(set(out))
    return out


sq._referenced_entity_norms = rec_ref
try:
    a1 = sq.quarantine_and_close(deepcopy(fm), strict_db=True, pathway_context=ctx, mode="pathwhiz")
finally:
    sq._referenced_entity_norms = _orig_ref
a2 = sq.quarantine_and_close(deepcopy(fm), strict_db=True, pathway_context=ctx, mode="pathwhiz")
summarize("A1 baseline", a1)
r1, r2 = a1.quarantine_report, a2.quarantine_report
same = all(r1.get(k) == r2.get(k) for k in ("decision_id", "resulting_payload_hash", "refusal_reasons", "review_reasons", "counts"))
print("\n  A1 == A2 (decision_id, resulting hash, reasons, counts):", same)
print("  matches leg: decision_id", r1.get("decision_id") == qrep.get("decision_id"),
      "| resulting_payload_hash", r1.get("resulting_payload_hash") == qrep.get("resulting_payload_hash"),
      "| refusal_reasons", r1.get("refusal_reasons") == qrep.get("refusal_reasons"))

row = next(x for x in fm["entities"]["proteins"] if x.get("name") == "phosphorylated CAMKII")
norms = sq._entity_name_norms([row])
for i, ref in enumerate(REC.get("referenced", []), 1):
    print(f"  iter {i}: norms(phosphorylated CAMKII)={sorted(norms)} & referenced = {sorted(norms & ref)}")
pc = [x for x in fm["entities"]["protein_complexes"] if any(sq._normalize(sq._component_name(c)) in norms for c in (x.get("components") or []))]
print("  complexes naming it as a component:", [x.get("name") for x in pc])

# ---- B: drop cross-row synonyms ----
fb = deepcopy(fm)
names = {sq._normalize(x.get("name")): x.get("name") for x in fb["entities"]["proteins"] if isinstance(x, dict)}
dropped = []
for x in fb["entities"]["proteins"]:
    syns = x.get("synonyms")
    if isinstance(syns, list):
        keep = [s for s in syns if not (sq._normalize(s) in names and sq._normalize(s) != sq._normalize(x.get("name")))]
        if keep != syns:
            dropped.append((x.get("name"), [s for s in syns if s not in keep]))
            x["synonyms"] = keep
print("\n  B dropped cross-row synonyms:", dropped)
b = sq.quarantine_and_close(fb, strict_db=True, pathway_context=ctx, mode="pathwhiz")
summarize("B cross-row synonyms dropped", b)
acc = lambda res: sorted((r["pointer"], r["state"]) for r in res.admissions)
print("  admissions identical A vs B:", acc(a1) == acc(b))
final_gate("B", b)

# ---- C: drop only the one synonym ----
fc = deepcopy(fm)
for x in fc["entities"]["proteins"]:
    if x.get("name") == "phosphorylated CAMKII":
        x["synonyms"] = [s for s in x.get("synonyms") or [] if s != "calcineurin B"]
c = sq.quarantine_and_close(fc, strict_db=True, pathway_context=ctx, mode="pathwhiz")
summarize("C only 'calcineurin B' dropped from phosphorylated CAMKII", c)
print("  admissions identical A vs C:", acc(a1) == acc(c))
final_gate("C", c)

# ---- D: synthetic applier probe ----
payload = {
    "entities": {
        "proteins": [{"name": "A"}, {"name": "dup"}, {"name": "C"}, {"name": "D"}],
        "protein_complexes": [{"name": "dup"}],
    },
    "processes": {},
}
ops = [
    {"op": "remove", "path": "/entities/proteins/1", "confidence": 0.97, "evidence": "dup declared as protein and complex"},
    {"op": "add", "path": "/entities/proteins/2/synonyms", "value": ["C-short"], "confidence": 0.9, "evidence": "C is called C-short"},
    {"op": "add", "path": "/entities/proteins/0/synonyms/-", "value": "A-short", "confidence": 0.9, "evidence": "A is called A-short"},
]
try:
    out, rep = apply_patch_with_policy(payload, ops)
    print("\n== D synthetic: summary", rep.get("summary"), "| committed", rep.get("transaction", {}).get("committed"))
    print("   rejected:", [(r.get("index"), r.get("reason")) for r in rep.get("rejected", [])])
    print("   proteins after:", json.dumps(out["entities"]["proteins"]))
except Exception as exc:
    print("\n== D synthetic raised", type(exc).__name__, exc)
sys.stdout.flush()
