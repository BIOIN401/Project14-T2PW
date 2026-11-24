"""FIX-NONAME replay: run stored payloads + stored patches through apply_patch_with_policy.

usage: python -I replay.py <src dir to import t2pw from> <out.json>

Run once against the 1a52fb55 tree and once against the fix tree; compare.py diffs them.
Cases:
  llm_patch   : every stored audit_patch*.json (real model output) applied to its round input
  gap_patch   : stored gap-resolver enrichment ops (accepted + rejected, original order)
  det_patch   : the deterministic audit patch of every stored leg payload
  gemma_shape : two synthetic Gemma-shaped ops on every stored leg payload
"""

import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, sys.argv[1])

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402
from t2pw.curation.audit_json_llm import audit_payload  # noqa: E402
from t2pw.pipeline.stage_contracts import StageContractError, validate_post_audit  # noqa: E402

PP_LIST = sys.argv[3] if len(sys.argv) > 3 else None


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


def verdict(payload):
    try:
        validate_post_audit(payload)
        return "ok"
    except StageContractError as exc:
        return "fail:" + ",".join(sorted({str(e.get("code")) for e in exc.errors}))
    except Exception as exc:  # noqa: BLE001
        return "error:" + type(exc).__name__


def run_case(results, key, payload, ops, recorded=None):
    try:
        out, report = apply_patch_with_policy(payload, ops)
        row = {
            "payload": digest(out),
            "report": digest(report),
            "post_audit": verdict(out),
            "accepted": len(report.get("accepted", [])),
            "rejected": len(report.get("rejected", [])),
            "name_rejections": sum(1 for r in report.get("rejected", []) if str(r.get("reason", "")).startswith("entity_name_required")),
        }
        if recorded is not None:
            row["matches_recorded_output"] = digest(out) == digest(recorded)
    except Exception as exc:  # noqa: BLE001
        row = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
    results[key] = row


def round_input(d, rnd):
    if rnd == 1:
        return os.path.join(d, "final.json")
    for name in (f"final.resolved.round{rnd - 1}.json", f"final.audited.round{rnd - 1}.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


results = {}
pp_dirs = [line.split()[0] for line in open(PP_LIST, encoding="utf-8") if "post_pipeline_" in line]
for d in pp_dirs:
    for patch_path in sorted(glob.glob(os.path.join(d, "audit_patch.round*.cand*.json"))):
        base = os.path.basename(patch_path)  # audit_patch.round1.cand2.json
        rnd = int(base.split(".round")[1].split(".")[0])
        cand = base.split(".cand")[1].split(".")[0]
        src = round_input(d, rnd)
        if not src or not os.path.exists(src):
            continue
        recorded_path = os.path.join(d, f"final.audited.round{rnd}.cand{cand}.json")
        recorded = load(recorded_path) if os.path.exists(recorded_path) else None
        run_case(results, f"llm_patch|{patch_path}", load(src), load(patch_path), recorded)
    gap = os.path.join(d, "gap_resolution_report.json")
    audited = os.path.join(d, "final.audited.json")
    if os.path.exists(gap) and os.path.exists(audited):
        app = (load(gap).get("enrichment") or {}).get("patch_application") or {}
        rows = [r for r in (app.get("accepted") or []) + (app.get("rejected") or []) if isinstance(r, dict)]
        ops = [r["op"] for r in sorted(rows, key=lambda r: r.get("index", 0)) if isinstance(r.get("op"), dict)]
        if ops:
            run_case(results, f"gap_patch|{gap}", load(audited), ops)

leg_payloads = []
for root in ("C:/t/acc-runs/legs3", "C:/t/acc-runs/legs4", "C:/t/acc-runs/legs4g", "C:/t/acc-runs/legs5g"):
    for name in ("merged_payload.json", "final_mapped.json"):
        leg_payloads += glob.glob(os.path.join(root, "**", name), recursive=True)

for path in sorted(leg_payloads):
    payload = load(path)
    try:
        det_ops = audit_payload(payload, use_llm=False)["patch"]
    except Exception as exc:  # noqa: BLE001
        results[f"det_patch|{path}"] = {"exception": f"audit: {type(exc).__name__}: {exc}"[:300]}
        det_ops = None
    if det_ops is not None:
        run_case(results, f"det_patch|{path}", payload, det_ops)
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    shapes = []
    if ents.get("species"):
        shapes.append({"op": "replace", "path": "/entities/species/0", "value": {"pathbank_species_id": "1"},
                       "confidence": 0.97, "evidence": "species lookup returned id 1"})
    shapes.append({"op": "add", "path": "/entities/protein_complexes/-",
                   "value": {"complex_name": "synthetic complex", "components": []},
                   "confidence": 0.97, "evidence": "synthetic"})
    if not isinstance(ents.get("protein_complexes"), list):
        shapes.insert(0, {"op": "add", "path": "/entities/protein_complexes", "value": [],
                          "confidence": 0.99, "evidence": "synthetic"})
    run_case(results, f"gemma_shape|{path}", payload, shapes)
    results[f"input|{path}"] = {"post_audit": verdict(payload)}

with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, sort_keys=True)
print("cases", len(results))
