"""ORCH-759 enterobactin no-PWML diagnosis. Read-only: loads leg artifacts, runs pure
gate / normalizer / component-reconcile functions on in-memory copies. Writes nothing."""
import json
from copy import deepcopy

from t2pw.pipeline.process_normalizer import (
    run_strict_post_normalization_gates,
    normalize_process_payload,
    GateValidationError,
)
from t2pw.pipeline.gate_reports import payload_sha256
from t2pw.mapping.map_ids import _reconcile_components_against_local_proteins

ROOT = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
L759 = ROOT + "/runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/papers/PMC12096016/strict"
L756 = ROOT + "/.claude/worktrees/orch756/runs_papers/ORCH-756/legs/enterobactin-strict/2026-10-21_2020/papers/PMC12096016/strict"


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def gate(label, payload):
    try:
        d = run_strict_post_normalization_gates(deepcopy(payload), enforce_all_proteins_connected=True)
        errs = d.get("errors") or []
        ok = True
    except GateValidationError as exc:
        errs = (exc.details or {}).get("errors") or []
        ok = False
    print(f"[gate] {label}: ok={ok} errors={len(errs)}")
    for e in errs:
        print("    ", e.get("path"), "|", e.get("reason"))


fm = load(L759 + "/final_mapped.json")
rep = load(L759 + "/final_stage3_gate_report.json")
print("[hash] final_mapped sha256 =", payload_sha256(fm))
print("[hash] gate report payload_sha256 =", rep.get("payload_sha256"))

gate("759 final_mapped as shipped", fm)

a = deepcopy(fm)
a["entities"]["protein_complexes"][1]["components"][0]["name"] = "enterobactin synthase"
gate("759 variant A: wrapper component named back to 'enterobactin synthase'", a)

b = deepcopy(fm)
b["entities"]["proteins"] = [p for p in b["entities"]["proteins"] if p.get("name") != "enterobactin synthase"]
gate("759 variant B: duplicate protein row 'enterobactin synthase' (P11454 == EntF) removed", b)

gate("756 final_mapped (control)", load(L756 + "/final_mapped.json"))

prots = fm["entities"]["proteins"]
print("[ids] protein rows carrying P11454:",
      [(i, p.get("name")) for i, p in enumerate(prots) if (p.get("mapped_ids") or {}).get("uniprot") == "P11454"])
print("[ids] EntF synonyms:", next(p for p in prots if p.get("name") == "EntF").get("synonyms"))
for label, comp in (
    ("component name only", {"name": "enterobactin synthase", "stoichiometry": 1}),
    ("component + uniprot P11454", {"name": "enterobactin synthase", "stoichiometry": 1, "mapped_ids": {"uniprot": "P11454"}}),
):
    out = _reconcile_components_against_local_proteins([comp], prots)
    print(f"[reconcile] {label} -> name={out[0].get('name')!r}")

# Normalization of the 759 merged (pre-mapping) payload, strict mode: does the
# relocate pass produce the wrapper + bare subunit protein the initial gate saw?
merged = load(L759 + "/merged_payload.json")
norm, nrep = normalize_process_payload(merged)
ents = norm.get("entities", {})
print("[norm] proteins:", [p.get("name") for p in ents.get("proteins", [])])
for c in ents.get("protein_complexes", []):
    print("[norm] complex:", c.get("name"), c.get("generation_reason"),
          [x.get("name") if isinstance(x, dict) else x for x in c.get("components") or []])
print("[norm] gate ok:", (nrep.get("gate") or {}).get("ok"),
      [e.get("reason") for e in (nrep.get("gate") or {}).get("errors") or []])
s = nrep.get("summary") or {}
print("[norm] complex_named_proteins_relocated:", s.get("complex_named_proteins_relocated"),
      "entities_added_as_proteins:", s.get("entities_added_as_proteins"))
