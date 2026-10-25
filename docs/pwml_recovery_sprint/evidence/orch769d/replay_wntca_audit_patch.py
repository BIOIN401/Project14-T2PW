"""ORCH-769d: replay a RECONSTRUCTION of the ORCH-767 Wnt/Ca2+ audit patch through
apply_patch_with_policy, then the strict quarantine, on whichever ``t2pw`` PYTHONPATH
names (run once with base ``src``, once with tip ``src``).

Why a reconstruction: the leg persisted neither the audit patch nor the audit's input
payload. It did persist final_mapped.json, whose rows carry an ``audit_repair`` lineage
entry on exactly the rows an accepted patch op wrote. This script:

1. rebuilds the audit input from final_mapped.json: strips the audit-written synonyms
   (and their audit_repair lineage entry) from those rows, and puts the protein row
   'calcineurin' back at index 10 (the audit rationale says it removed that row;
   stage1_payload.json puts it right after SERCA);
2. rebuilds the patch: ``remove /entities/proteins/10``, then one ``add .../synonyms``
   per audit-written protein row AT THE INDEX THE SYNONYM WAS RECORDED ON (on base that
   is the index the auditor wrote, because base applies it unrebased), plus the
   recorded ``add /entities/compounds/3/synonyms/- "Ca2"`` (the second defect);
3. applies it, checks the base output reproduces the recording, and runs the strict
   quarantine and the final pre-export gate the way ``replay_wntca_quarantine.py``
   (ORCH-769 diagnosis) does.

The other four audit-written rows (compound 4, reaction 3, transports 9 and 10) are
left as recorded: no index in that patch region moves them.

No LLM, no network, no writes except ``--out``.
"""
import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import t2pw
import t2pw.pipeline.strict_quarantine as sq
from t2pw.curation import apply_audit_patch as aap
from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates

ap = argparse.ArgumentParser()
ap.add_argument("--leg", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
LEG = Path(args.leg)

fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
ctx = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))["coverage"]["requested_context"]
applier = Path(aap.__file__)
out = {
    "t2pw": str(Path(t2pw.__file__).parent),
    "applier_sha256": hashlib.sha256(applier.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    "leg": str(LEG),
}


def audit_written(row):
    return any(isinstance(e, dict) and e.get("stage") == "audit_repair" for e in row.get("provenance_lineage") or [])


def strip_audit(row):
    row.pop("synonyms", None)
    row["provenance_lineage"] = [e for e in row["provenance_lineage"] if e.get("stage") != "audit_repair"]


proteins = fm["entities"]["proteins"]
recorded_rows = [i for i, r in enumerate(proteins) if audit_written(r)]
assert recorded_rows == [2, 3, 4, 6, 7, 8, 9, 13, 14, 17, 18, 19, 20], recorded_rows
assert proteins[18]["name"] == "phosphorylated CAMKII" and proteins[18]["synonyms"] == ["calcineurin B"]
assert fm["entities"]["compounds"][3]["synonyms"] == {"-": "Ca2"}
assert "calcineurin" not in {r["name"] for r in proteins}

audit_input = deepcopy(fm)
for i in recorded_rows:
    strip_audit(audit_input["entities"]["proteins"][i])
strip_audit(audit_input["entities"]["compounds"][3])
audit_input["entities"]["proteins"].insert(10, {"name": "calcineurin"})

patch = [{
    "op": "remove", "path": "/entities/proteins/10", "confidence": 0.97,
    "evidence": "Entity 'calcineurin' is declared as both a protein and a protein_complex.",
}]
patch += [
    {"op": "add", "path": f"/entities/proteins/{i}/synonyms", "value": proteins[i]["synonyms"],
     "confidence": 0.9, "evidence": "reconstructed (ORCH-769d)"}
    for i in recorded_rows
]
patch.append({"op": "add", "path": "/entities/compounds/3/synonyms/-", "value": "Ca2",
              "confidence": 0.9, "evidence": "reconstructed (ORCH-769d)"})

patched, report = aap.apply_patch_with_policy(audit_input, patch)
out["patch"] = patch
out["apply_summary"] = report["summary"]
out["apply_committed"] = report["transaction"]["committed"]
out["apply_rejected"] = [{"index": r["index"], "reason": r["reason"]} for r in report["rejected"]]
out["applied_paths"] = {r["index"]: r["applied_path"] for r in report["accepted"] if "applied_path" in r}


def no_lineage(row):
    return {k: v for k, v in row.items() if k != "provenance_lineage"}


def stages(row):
    return sorted(e.get("stage") for e in row.get("provenance_lineage") or [])


got = patched["entities"]["proteins"]
out["protein_synonyms_after_apply"] = {r["name"]: r.get("synonyms") for r in got if r.get("synonyms") is not None}
out["compound_3_synonyms_after_apply"] = patched["entities"]["compounds"][3].get("synonyms")
out["reproduces_recording"] = {
    "protein_rows_equal_ignoring_lineage": [no_lineage(a) for a in got] == [no_lineage(b) for b in proteins],
    "protein_lineage_stage_multisets_equal": [stages(a) for a in got] == [stages(b) for b in proteins],
    "compounds_equal_ignoring_lineage": [no_lineage(a) for a in patched["entities"]["compounds"]]
    == [no_lineage(b) for b in fm["entities"]["compounds"]],
    "rows_differing": [
        {"name": b["name"], "recorded": b.get("synonyms"), "replayed": a.get("synonyms")}
        for a, b in zip(got, proteins) if no_lineage(a) != no_lineage(b)
    ],
}
names = {sq._normalize(r.get("name")) for r in got}
out["cross_row_synonyms"] = [
    (r["name"], s) for r in got for s in (r.get("synonyms") or [])
    if isinstance(s, str) and sq._normalize(s) in names and sq._normalize(s) != sq._normalize(r["name"])
]

res = sq.quarantine_and_close(deepcopy(patched), strict_db=True, pathway_context=ctx, mode="pathwhiz")
q = res.quarantine_report
rel = q.get("release") or {}
p = res.payload or {}
out["quarantine"] = {
    "ok": q.get("ok"),
    "refusal_reasons": q.get("refusal_reasons"),
    "review_reasons": q.get("review_reasons"),
    "unexportable_entities": (q.get("strict_invariants") or {}).get("unexportable_entities"),
    "release_status": rel.get("status"),
    "release_reasons": rel.get("reasons"),
    "counts": q.get("counts"),
    "surviving_processes": {b: len(v) for b, v in (p.get("processes") or {}).items() if isinstance(v, list)},
    "surviving_proteins": [x.get("name") for x in (p.get("entities") or {}).get("proteins", [])],
}
if res.ok:
    try:
        details = run_strict_post_normalization_gates(deepcopy(res.payload), enforce_all_proteins_connected=True)
        conn = details.get("connectivity") or {}
        out["final_gate"] = {"result": "PASS", **{k: conn.get(k) for k in ("n_reactions", "n_transports", "n_interactions")}}
    except Exception as exc:  # noqa: BLE001
        out["final_gate"] = {"result": "FAIL", "error": f"{type(exc).__name__}: {exc}"[:400]}
else:
    out["final_gate"] = {"result": "not run (quarantine refused)"}

Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
print(json.dumps({k: out[k] for k in out if k != "patch"}, indent=2, ensure_ascii=True))
