"""ORCH-759c G9 evidence: replay the chlorophyll-strict quarantine on the leg's own input.

Reads the committed fixture (the leg's final_mapped.json verbatim + its Stage-0
context), runs ``quarantine_and_close`` with the code on PYTHONPATH, and prints:

1. the run's hashes, counts, refusal reasons and release status;
2. a BASE EMULATION in the same process -- ``_prune_entities`` handed the old
   single union set instead of the per-bucket mapping -- and checks it reproduces
   the real base run's ``resulting_payload_hash`` (01-base-facts.log), so the
   row-level diff below is a diff against base, not against a guess;
3. every entity / location / state / process row that differs between the two;
4. the strict final pre-export gate on the result, as freeze_canonical_payload runs it.

Prints only; writes nothing. Run at the base SHA it prints the base result twice.
"""
import gzip
import json
from copy import deepcopy
from pathlib import Path

import t2pw.pipeline.strict_quarantine as sq
from t2pw.pipeline.process_normalizer import GateValidationError, run_strict_post_normalization_gates

FIX = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "orch759c"
payload = json.loads(gzip.decompress((FIX / "chlorophyll_PMC5130987_final_mapped.json.gz").read_bytes()))
context = json.loads((FIX / "chlorophyll_PMC5130987_requested_context.json").read_text(encoding="utf-8"))
BASE_RESULTING = "sha256:e1f3bec8621d8069"  # 01-base-facts.log, real base code
COUNTERFACTUAL_RESULTING = "sha256:be6568f9391467dc"  # orch759_chlorophyll_diag/replay.out.txt, B


def run():
    return sq.quarantine_and_close(deepcopy(payload), strict_db=True, pathway_context=context, mode="pathwhiz")


def show(tag, res):
    r = res.quarantine_report
    print(f"== {tag}")
    for k in ("admitted_payload_hash", "decision_input_hash", "resulting_payload_hash", "ok", "refusal_reasons", "review_reasons", "counts"):
        print(f"  {k}: {r.get(k)}")
    print("  entity_type_overlaps:", r["strict_invariants"]["entity_type_overlaps"])
    rel = r.get("release") or {}
    print("  release.status:", rel.get("status"), "| reasons:", rel.get("reasons"))
    print("  removed entities:", len(res.removed_entity_report["removed_entities"]))


tip = run()
show("TIP (code on PYTHONPATH)", tip)

real_prune = sq._prune_entities


def union_prune(p, keep):
    if isinstance(keep, dict):
        keep = set().union(*keep.values())
    return real_prune(p, keep)


sq._prune_entities = union_prune
try:
    base = run()
finally:
    sq._prune_entities = real_prune
show("BASE EMULATION (one union set for every bucket)", base)
print("  base emulation == real base resulting hash:", base.quarantine_report["resulting_payload_hash"] == BASE_RESULTING)
print("  tip == diagnosis counterfactual resulting hash:", tip.quarantine_report["resulting_payload_hash"] == COUNTERFACTUAL_RESULTING)


def rows(res):
    out = set()
    p = res.payload
    for group in ("entities", "processes", "element_locations"):
        for bucket, items in (p.get(group) or {}).items():
            for item in items if isinstance(items, list) else []:
                out.add((group, bucket, json.dumps(item, sort_keys=True, ensure_ascii=False)))
    for item in p.get("biological_states") or []:
        out.add(("biological_states", "", json.dumps(item, sort_keys=True, ensure_ascii=False)))
    return out


def short(row):
    group, bucket, blob = row
    d = json.loads(blob)
    return f"{group}/{bucket}: {d.get('name') if isinstance(d, dict) else d!r}"


b, t = rows(base), rows(tip)
print("== ROW DIFF base -> tip")
print("  only in base:", sorted(short(x) for x in b - t))
print("  only in tip :", sorted(short(x) for x in t - b))
rb = {(e["bucket"], e["name"], e["reason"]) for e in base.removed_entity_report["removed_entities"]}
rt = {(e["bucket"], e["name"], e["reason"]) for e in tip.removed_entity_report["removed_entities"]}
print("  removed only at tip :", sorted(rt - rb))
print("  removed only at base:", sorted(rb - rt))
print("  removed_locations base/tip:", len(base.removed_entity_report["removed_locations"]), len(tip.removed_entity_report["removed_locations"]))
print("  removed_biological_states base/tip:", len(base.removed_entity_report["removed_biological_states"]), len(tip.removed_entity_report["removed_biological_states"]))
acc = lambda res: sorted((a["pointer"], a["state"], a.get("reason")) for a in res.admissions)
print("  admissions identical:", acc(base) == acc(tip))

print("== STRICT FINAL GATE on tip payload (enforce_all_proteins_connected=True)")
if tip.ok:
    try:
        details = run_strict_post_normalization_gates(deepcopy(tip.payload), enforce_all_proteins_connected=True)
        print("  PASS; connectivity:", json.dumps(details.get("connectivity")))
    except GateValidationError as exc:
        print("  FAIL:", exc, json.dumps((exc.details or {}).get("errors"), ensure_ascii=False)[:2000])
else:
    print("  not run: quarantine refused (freeze_canonical_payload would file final_gate_report_missing)")
