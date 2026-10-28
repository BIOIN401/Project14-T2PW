"""ORCH-786 offline replay of the ORCH-784 wnt_ca strict leg through mapping + post_remap.

usage: python replay.py <tree_root> <workdir>
Reuses tests/orch762_harness.py (Doors: no PathBank DB, no LLM, every HTTP request
answered 404 and logged). Stage-2 cache = a copy of the leg run's id_mapping_cache.json.
"""
import json
import shutil
import sys
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(tree / "src"), str(tree / "tests")]

import orch762_harness as h  # noqa: E402
from t2pw.pipeline.stage_contracts import StageContractError, validate_post_remap  # noqa: E402

LEG = Path(
    "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch784run/runs_papers/"
    "ORCH-784/legs/wnt_ca-strict/2026-10-28_0044/papers/PMC10425814/strict/merged_payload.json"
)
CACHE = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch784run/data/id_mapping_cache.json")

work.mkdir(parents=True, exist_ok=True)
shutil.copyfile(CACHE, work / "id_mapping_cache.json")
payload = json.loads(LEG.read_text(encoding="utf-8"))
http = h.ReplayHttp({})
# The leg's Stage-6 live UniProt answer for STIM2 was not persisted (use_cache=False);
# it is injected here as the reviewed human entry, the ORCH-759a resolution technique.
STIM2 = {"STIM2": {"status": "mapped", "mapped_ids": {"uniprot": "Q9P246"}, "provider": "UniProt",
         "source": "api", "confidence": 1.0, "chosen_rule": "uniprot_exact_gene",
         "candidates": [], "resolution": {"status": "matched"}, "identity_verdict": None, "resolved_name": "STIM2"}}
with h.Doors(http, resolutions=STIM2) as doors:
    out = h.replay(payload, work)
final = out["stage6"]["payload"]
print("tree:", tree)
print("http requests answered 404 (offline):", len(http.calls), "| ai calls:", len(doors.ai_calls))
ents = final["entities"]
print("entity counts:", {k: len(v) for k, v in ents.items() if isinstance(v, list)})
for c in ents.get("protein_complexes", []):
    if c.get("generated"):
        fam = (c.get("mapping_meta") or {}).get("family_complex")
        print("  generated complex:", c.get("name"), "| components:",
              [x.get("name") if isinstance(x, dict) else x for x in c.get("components", [])],
              "| family:", bool(fam))
rep = (out["stage6"]["report"] or {})
for a in (rep.get("enzyme_complex_report") or {}).get("actions", []) if isinstance(rep.get("enzyme_complex_report"), dict) else []:
    if "family" in str(a.get("type")) or "omit" in str(a.get("type")):
        print("  action:", json.dumps(a, ensure_ascii=False)[:300])
adm = final.get("identity_admission_report") or rep.get("identity_admission") or {}
try:
    r = validate_post_remap(final)
    print("POST_REMAP: PASS  errors=", len(r.get("errors", [])))
except StageContractError as exc:
    r = getattr(exc, "report", {}) or {}
    print("POST_REMAP: STOP (StageContractError)")
for e in r.get("errors", []):
    print("  error:", e.get("code"), e.get("json_pointer") or e.get("pointer"), e.get("protein") or e.get("component"))
for c in ents.get("protein_complexes", []):
    fam = (c.get("mapping_meta") or {}).get("family_complex")
    if fam:
        print("  family note:", c.get("name"), "->", fam.get("note"))
adm = (out["stage6"]["report"] or {}).get("identity_admission") or {}
print("admission withheld:", [(w.get("name"), w.get("rule")) for w in adm.get("withheld", [])])
