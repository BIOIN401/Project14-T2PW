"""ORCH-761: pin the identity records of the failing legs, so the RESULT doc cites committed text.

The leg directories are uncommitted run output. This script reads them (plain json, no t2pw import,
no network, no DB), prints the sha256 of every file it read, and prints only the identity records
the ORCH-761 root-cause table relies on. A changed sha256 means the leg directory moved.

Run from the primary checkout root:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -I docs/pwml_recovery_sprint/evidence/orch761_leg_identity_extract.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MK756 = ROOT / ".claude/worktrees/orch756/runs_papers/ORCH-756/legs/menaquinone-strict/2026-10-21_2048/papers/PMC12421875/strict"
MK755 = ROOT / "tmp/post_pipeline_3ded14a704d24183a8ab5b8d85ac864e"
EN755 = ROOT / "runs_papers/ORCH-755/legs/enterobactin-strict/2026-10-20_2201/papers/PMC12096016/strict"
EN759 = ROOT / "runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/papers/PMC12096016/strict"
CACHE756 = ROOT / ".claude/worktrees/orch756/data/id_mapping_cache.json"


def load(path):
    data = path.read_bytes()
    print(f"  read {path.relative_to(ROOT).as_posix()}  sha256={hashlib.sha256(data).hexdigest()}")
    return json.loads(data.decode("utf-8"))


def cand(c):
    if not isinstance(c, dict):
        return str(c)
    acc = c.get("accession") or c.get("uniprot")
    return f"{acc}|{c.get('organism')}|tax={c.get('taxonomy_id')}|score={c.get('score')}"


def protein_gate_rows(report, label):
    print(f"\n== {label}: {report.get('stage')} / {report.get('phase')} -- {len(report.get('errors', []))} error(s)")
    for err in report.get("errors", []):
        det = err.get("detail", {}) or {}
        mm = det.get("mapping_meta", {}) or {}
        iv = mm.get("identity_verdict", {}) or {}
        print(f"- {err.get('path')} {det.get('name')!r}: {err.get('reason')}")
        if mm:
            print(f"    organism={det.get('organism')!r} provider={mm.get('provider')} resolution={mm.get('resolution')}")
            print(f"    queries_tried={mm.get('queries_tried')}")
            print(f"    verdict={iv.get('reason')} identity={iv.get('identity')} score={iv.get('score')} margin={iv.get('margin')}")
            print(f"    judged_candidate={cand(iv.get('judged_candidate'))}")


print("# A. menaquinone, ORCH-756 1-draw (organism string: Lactococcus lactis subsp. lactis KF147)")
protein_gate_rows(load(MK756 / "initial_stage3_gate_report.json"), "initial gate")
fm = load(MK756 / "final_mapped.json")
print("  species rows:", [(s.get("name"), s.get("taxonomy_id"), s.get("raw_name")) for s in fm["entities"].get("species", [])])
print("  protein_export_policy.summary:", fm.get("protein_export_policy", {}).get("summary"))
print("  quarantined_locked_reactions:", [(q.get("reaction_name"), q.get("reason")) for q in fm.get("quarantined_locked_reactions", [])])
rem = load(MK756 / "removed_entity_report.json")
print("  removed_entities:", rem.get("removed_entities"))
qr = load(MK756 / "quarantine_report.json")
rel = qr.get("release") or {}
print("  quarantine ok:", qr.get("ok"), "| release.status:", rel.get("status"), "| missing_anchors:", rel.get("missing_anchors"))
for a in qr.get("admissions", []):
    if a.get("essential_participant") == "MenD":
        print(f"  admission: {a.get('process_kind')} {a.get('name')!r} -> {a.get('state')} ({a.get('reason')})")
cache = load(CACHE756)
for key, val in sorted(cache.get("proteins", {}).items()):
    if "kf147" in key and key.startswith(("db::", "api-v8::")):
        top = [cand(c) for c in (val.get("candidates") or [])[:3]]
        print(f"  cache {key}: status={val.get('status')} reason={val.get('reason')} mapped={val.get('mapped_ids')} top={top}")

print("\n# B. menaquinone, ORCH-755 night (organism string: Lactococcus lactis)")
protein_gate_rows(load(MK755 / "gate_fail_report.json"), "initial gate")

print("\n# C. enterobactin, ORCH-755 night (D1: holo-EntB)")
fm = load(EN755 / "final_mapped.json")
c16 = fm["entities"]["compounds"][16]
mm = c16["mapping_meta"]
top = (mm.get("candidates") or [{}])[0]
print(f"  compounds/16 {c16['name']!r} confidence={c16.get('confidence')} provenance={c16.get('provenance')} route={mm.get('route')} "
      f"route_reason={mm.get('route_reason')} chosen_rule={mm.get('chosen_rule')} resolution={mm.get('resolution')}")
print(f"    top candidate={cand(top)} matched_alias={top.get('matched_alias')} alias_source={top.get('alias_source')}")
adm = mm.get("identity_admission", {})
print(f"    identity_admission rules={adm.get('rules')} withheld={adm.get('withheld')}")
p1 = fm["entities"]["proteins"][1]
pm = p1["mapping_meta"]
print(f"  proteins/1 {p1['name']!r} mapped_ids={p1.get('mapped_ids')} rejected_mapped_ids={pm.get('rejected_mapped_ids')} "
      f"admission_rules={pm.get('identity_admission', {}).get('rules')} verdict={pm.get('identity_verdict', {}).get('reason')}")
qr = load(EN755 / "quarantine_report.json")
print("  quarantine ok:", qr.get("ok"), "| refusal_reasons:", qr.get("refusal_reasons"), "| release.status:", (qr.get("release") or {}).get("status"))

print("\n# D. enterobactin, ORCH-759 3-draw")
protein_gate_rows(load(EN759 / "initial_stage3_gate_report.json"), "initial gate")
fm = load(EN759 / "final_mapped.json")
p7 = fm["entities"]["proteins"][7]
print(f"  proteins/7 {p7['name']!r} uniprot={p7.get('uniprot')} pathbank_protein_id={p7.get('pathbank_protein_id')} synonyms={p7.get('synonyms')}")
pc1 = fm["entities"]["protein_complexes"][1]
print(f"  protein_complexes/1 {pc1['name']!r} generated={pc1.get('generated')} generation_reason={pc1.get('generation_reason')} "
      f"components={[c.get('name') for c in pc1.get('components', [])]}")
cands = [(c.get("pathbank_complex_id"), c.get("name"), c.get("score")) for c in pc1["mapping_meta"].get("candidates", [])]
print(f"    mapping_meta.resolution={pc1['mapping_meta'].get('resolution')} candidates={cands}")
print(f"    lineage={[(x.get('stage'), x.get('origin')) for x in pc1.get('provenance_lineage', [])]}")
fg = load(EN759 / "final_stage3_gate_report.json")
print("  final gate:", fg.get("stage"), fg.get("ok"), [e.get("reason") for e in fg.get("errors", [])])
au = load(EN759 / "audit_iteration_summary.json")
r1 = au["rounds"][0]
print("  audit round 1: accepted_patch_count=", r1.get("accepted_patch_count"), "| rationale:", r1.get("llm_repair_rationale"))
