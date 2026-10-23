"""ORCH-759 chlorophyll-strict: is `CPO reaction` the only blocker past 759c's quarantine fix?

Read-only diagnosis at the checked-out code (45aa9c28). Prints only; writes nothing in any repo.
  1. strict quarantine (759c's fix included) on the leg's own final_mapped.json
  2. the final pre-export gate exactly as freeze_canonical_payload runs it
  3. required-field contract + validate_pre_export + IR (Chunk E's exporter approximation), every issue
  4. post-freeze counterfactuals (diagnostic only, NOT a fix): wrap CPO / drop bare CPO / drop the reaction
  5. pre-freeze mapping-seam counterfactuals: re-run Stage 6's enzyme-wrapper rewrite now that CPO
     carries a verified accession (M1: after the Unknown fallback; M2: before it, CPO1 still a bare
     unresolved protein), then quarantine + gate + contract + IR again.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
LEG = ROOT / "runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict"
sys.path.insert(0, str(ROOT / "tests"))

from test_strict_quarantine_real_artifact_replay import _as_export_payload  # noqa: E402
from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.pipeline.process_normalizer import run_strict_post_normalization_gates, GateValidationError  # noqa: E402
from t2pw.pipeline.stage_contracts import validate_pre_export  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir, validate_required_pwml_contract  # noqa: E402
from t2pw.pwml.writer import blocking_pwml_ir_errors  # noqa: E402
import t2pw.mapping.map_ids as M  # noqa: E402

fm = json.loads((LEG / "final_mapped.json").read_text(encoding="utf-8"))
ctx = json.loads((LEG / "quarantine_report.json").read_text(encoding="utf-8"))["coverage"]["requested_context"]


def rxn_name(payload, pointer):
    parts = (pointer or "").split("/")
    try:
        if parts[1] == "processes":
            return payload["processes"][parts[2]][int(parts[3])].get("name")
    except Exception:
        return None
    return None


def export_checks(tag, payload):
    out = {}
    rep = validate_required_pwml_contract(_as_export_payload(payload), strict_db=True)
    errs = rep.get("errors") or []
    out["contract"] = [e.get("code") for e in errs]
    print(f"  [{tag}] required-field contract ok={rep.get('ok')} errors={len(errs)}")
    for e in errs:
        print("     C", e.get("code"), e.get("pointer"), "| rxn:", rxn_name(payload, e.get("pointer")), "|", str(e.get("message"))[:150])
    try:
        validate_pre_export(_as_export_payload(payload), strict_db=True)
        print(f"  [{tag}] validate_pre_export: ok")
        out["pre_export"] = "ok"
    except Exception as exc:  # StageContractError
        det = getattr(exc, "report", None) or getattr(exc, "details", None) or {}
        codes = [i.get("code") for i in (det.get("errors") or [])] if isinstance(det, dict) else []
        print(f"  [{tag}] validate_pre_export RAISED {type(exc).__name__}: codes={codes} msg={str(exc)[:200]}")
        out["pre_export"] = codes or str(exc)[:200]
    try:
        ir, ir_report = build_pwml_ir(
            prefrozen_when_compounded(deepcopy(payload)),
            pathway_name="Replayed pathway", pathway_subject="Metabolic", strict_db=True, width=3200, height=1400,
        )
        blocking = blocking_pwml_ir_errors(ir_report)
        val = validate_pwml_ir(ir).get("errors", [])
        out["ir"] = [i.get("code") for i in blocking] + [i.get("code") for i in val]
        print(f"  [{tag}] IR blocking={len(blocking)} validate_pwml_ir={len(val)}")
        for i in blocking:
            print("     IRb", i.get("code"), i.get("pointer"), "|", str(i.get("message"))[:160])
        for i in val:
            print("     IRv", i.get("code"), i.get("pointer"), "|", str(i.get("message"))[:160])
    except Exception as exc:  # noqa: BLE001
        out["ir"] = [f"ir_build_crashed:{type(exc).__name__}:{exc}"]
        print(f"  [{tag}] IR build crashed: {type(exc).__name__}: {exc}")
    return out


def quarantine_and_gate(tag, payload):
    res = quarantine_and_close(deepcopy(payload), strict_db=True, pathway_context=ctx, mode="pathwhiz")
    r = res.quarantine_report
    rel = r.get("release") or {}
    print(f"\n== {tag}")
    print("  quarantine ok:", res.ok, "| refusal:", r.get("refusal_reasons"), "| counts:", r.get("counts"))
    print("  release:", rel.get("status"), "| reasons:", rel.get("reasons"), "| review_reasons:", rel.get("review_reasons"))
    gate = None
    if res.ok:
        try:
            run_strict_post_normalization_gates(deepcopy(res.payload), enforce_all_proteins_connected=True)
            gate = "ok"
        except GateValidationError as exc:
            d = exc.details if isinstance(exc.details, dict) else {}
            gate = f"FAIL {str(exc)[:200]} | {json.dumps(d.get('errors') or d, default=str)[:600]}"
        print("  final pre-export gate:", gate)
    p = res.payload or {}
    for i, rx in enumerate(p.get("processes", {}).get("reactions") or []):
        acts = [(a.get("entity"), a.get("entity_type")) for a in (rx.get("enzymes") or []) if isinstance(a, dict)]
        if any("CPO" in str(n) for n, _ in acts):
            mods = [(a.get("entity"), a.get("entity_type")) for a in (rx.get("modifiers") or []) if isinstance(a, dict)]
            print(f"  surviving /processes/reactions/{i} {rx.get('name')!r} {rx.get('inputs') or rx.get('left')}->{rx.get('outputs') or rx.get('right')} enzymes={acts} modifiers={mods}")
    names = [c.get("name") for c in p.get("entities", {}).get("protein_complexes") or []]
    print("  CPO-ish complexes:", [n for n in names if "CPO" in str(n)], "| proteins:", [x.get("name") for x in p.get("entities", {}).get("proteins") or [] if "CPO" in str(x.get("name"))])
    return res


# ---------------------------------------------------------------- 1-3. live payload
base = quarantine_and_gate("LIVE final_mapped @ tip code", fm)
live = export_checks("LIVE", base.payload)
for adm in (base.quarantine_report.get("admissions") or base.quarantine_report.get("process_admissions") or []):
    if isinstance(adm, dict) and "CPO" in json.dumps(adm):
        print("  admission:", json.dumps(adm, ensure_ascii=False)[:400])

# ---------------------------------------------------------------- 4. post-freeze counterfactuals
def cpo_rxn(payload):
    return next(i for i, r in enumerate(payload["processes"]["reactions"]) if r.get("name") == "CPO reaction")

wrapped = deepcopy(base.payload)
ri = cpo_rxn(wrapped)
cpo_row = next(r for r in wrapped["entities"]["proteins"] if r.get("name") == "CPO")
wrapped["entities"]["protein_complexes"].append({
    "name": "CPO complex", "species": "Arabidopsis thaliana", "organism": "Arabidopsis thaliana",
    "species_id": cpo_row.get("species_id"), "pathbank_species_id": cpo_row.get("pathbank_species_id"),
    "generated": True, "generation_reason": "single_protein_pathwhiz_wrapper",
    "components": [{"name": "CPO", "stoichiometry": 1, "mapped_ids": {"uniprot": "Q9LR75"}}],
    "mapping_meta": {"provider": "none", "source": "none", "chosen_rule": "novel_enzyme_single_component_complex",
                     "resolution": {"status": "novel", "issue": "diagnostic", "order_step": "novel_enzyme_single_component_complex"}},
})
for key in ("enzymes", "modifiers"):
    for a in wrapped["processes"]["reactions"][ri].get(key) or []:
        if isinstance(a, dict) and a.get("entity") == "CPO":
            a["entity"], a["entity_type"] = "CPO complex", "protein_complex"
print("\n== X1 post-freeze: CPO actor -> generated 'CPO complex' wrapper (diagnostic only)")
x1 = export_checks("X1", wrapped)

dropped = deepcopy(base.payload)
ri = cpo_rxn(dropped)
for key in ("enzymes", "modifiers"):
    dropped["processes"]["reactions"][ri][key] = [a for a in dropped["processes"]["reactions"][ri].get(key) or [] if not (isinstance(a, dict) and a.get("entity") == "CPO")]
print("\n== X2 post-freeze: bare CPO actor removed, CPO1 placeholder kept (diagnostic only)")
x2 = export_checks("X2", dropped)

# ---------------------------------------------------------------- 5. mapping-seam counterfactuals
cache = M.MappingCache(Path(r"C:/t/bt-cpo759/never_written_cache.json"), enabled=False)

m1 = deepcopy(fm)
rep1 = M._rewrite_reaction_protein_enzymes_to_complexes(m1, db=None, cache=cache, global_organism="Arabidopsis thaliana")
print("\n== M1 mapping seam: wrapper rewrite re-run on final_mapped (CPO verified, CPO1 already the placeholder wrapper)")
print("  summary:", json.dumps(rep1.get("summary")))
for a in rep1.get("actions") or []:
    if "CPO" in json.dumps(a):
        print("  action:", json.dumps(a, ensure_ascii=False)[:300])
r1 = quarantine_and_gate("M1 quarantine", m1)
m1x = export_checks("M1", r1.payload)

# M2: CPO1 back to what it was at the rewrite point (a bare protein, no identity) and the
# placeholder complex not yet built; CPO keeps its verified accession. Then rewrite -> Unknown fallback.
m2 = deepcopy(fm)
ents, rxs = m2["entities"], m2["processes"]["reactions"]
ents["protein_complexes"] = [c for c in ents["protein_complexes"] if c.get("name") != "CPO1"]
ents["proteins"].append({"name": "CPO1", "species": "Arabidopsis thaliana", "organism": "Arabidopsis thaliana",
                         "species_id": 4, "pathbank_species_id": 4, "taxonomy_id": "3702",
                         "mapping_meta": {"provider": "PathBankDB", "source": "db", "chosen_rule": "novel_protein",
                                          "real_mapping_failure_reason": "species_mismatch",
                                          "resolution": {"status": "novel", "issue": "species_mismatch", "order_step": "novel_protein"}}})
for rx in rxs:
    for key in ("enzymes", "modifiers"):
        for a in rx.get(key) or []:
            if isinstance(a, dict) and a.get("entity") == "CPO1":
                a["entity_type"] = "protein"
rep2 = M._rewrite_reaction_protein_enzymes_to_complexes(m2, db=None, cache=cache, global_organism="Arabidopsis thaliana")
print("\n== M2 mapping seam: rewrite AFTER the UniProt fallback, BEFORE the Unknown fallback")
print("  summary:", json.dumps(rep2.get("summary")))
for a in rep2.get("actions") or []:
    if "CPO" in json.dumps(a):
        print("  action:", json.dumps(a, ensure_ascii=False)[:300])
unk = M._apply_pathbank_unknown_enzyme_fallback(m2)
print("  unknown fallback summary:", json.dumps(unk.get("summary")))
for a in unk.get("actions") or []:
    if "CPO" in json.dumps(a):
        print("  ufb action:", json.dumps(a, ensure_ascii=False)[:300])
r2 = quarantine_and_gate("M2 quarantine", m2)
m2x = export_checks("M2", r2.payload)

print("\nSUMMARY")
for tag, o in (("LIVE", live), ("X1", x1), ("X2", x2), ("M1", m1x), ("M2", m2x)):
    print(f"  {tag}: contract={o.get('contract')} pre_export={o.get('pre_export')} ir={o.get('ir')}")
