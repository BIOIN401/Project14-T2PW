"""ORCH-792 / ORCH-793 offline leg replay: Stage 2 -> normalise -> Stage 6 -> strict quarantine, base vs tip.

usage: python orch792_replay.py <tree_root> <workdir> [--no-taxa] [leg ...]
--no-taxa: leave the species rows as merged (no taxonomy id): the taxon-scoped UniProt answers are then
not keyed, every lookup misses offline, and the run reproduces the "nothing resolves" Stage-6 shape.
Reuses tests/orch762_harness.py (no PathBank DB, no LLM, every live HTTP request answered 404 and
counted). The recorded UniProt/PathBank answers are the leg run's own ``data/id_mapping_cache.json``
(a copy), read by Stage 2 AND Stage 6: the live Stage 6 ran uncached, so this replays the answers the
same run recorded for the same queries. The audit and gap rounds are not run (harness limit).
Prints, per leg: the target rows' fate, every quarantined process naming a target, and the kept count.
Read-only on the legs; writes only under <workdir>.
"""
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(tree / "src"), str(tree / "tests")]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import orch762_harness as h  # noqa: E402
from t2pw.mapping import map_ids  # noqa: E402
from t2pw.pipeline.process_normalizer import normalize_process_payload  # noqa: E402
from t2pw.pipeline.strict_quarantine import ACCEPTED_STATES, quarantine_and_close  # noqa: E402

W = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees")
LEGS = {
    "wnt_ca": ("orch785run", "ORCH-785/legs/wnt_ca-strict/2026-10-28_1910/papers/PMC10425814/strict",
               ["PKC", "protein kinase C", "ORAI2", "Orai2", "TCF", "TAK1"]),
    "chlorophyll": ("orch785run", "ORCH-785/legs/chlorophyll-strict/2026-10-28_1743/papers/PMC5130987/strict",
                    ["GBP", "CHL27", "LPOR", "ferrochelatase", "Chl synthase", "EXECUTER proteins", "PIF1"]),
    "wnt_pcp": ("orch784run", "ORCH-784/legs/wnt_pcp-strict/2026-10-27_2309/papers/PMC10425814/strict",
                ["Frizzled", "Wnt", "Smurf", "DIA1", "Profilin", "MRLC", "G-actin", "c-Jun"]),
    "menaquinone": ("orch784run", "ORCH-784/legs/menaquinone-strict/2026-10-28_0208/papers/PMC12421875/strict",
                    ["MenA"]),
    "enterobactin": ("orch784run", "ORCH-784/legs/enterobactin-strict/2026-10-27_1931/papers/PMC12096016/strict",
                     ["Fur"]),
    "sour_taste_v2": ("orch785run", "ORCH-785/legs/sour_taste_v2-strict/2026-10-28_2058/papers/PMC7059786/strict",
                      ["Kir2.1"]),
}


def norm(s):
    return map_ids._normalize_name(str(s or ""))


def fate(payload, name):
    n = norm(name)
    ents = payload.get("entities") or {}
    for row in ents.get("proteins") or []:
        if isinstance(row, dict) and norm(row.get("name")) == n:
            ids = row.get("mapped_ids") or {}
            return f"protein uniprot={ids.get('uniprot') or '-'}"
    for row in ents.get("protein_complexes") or []:
        if isinstance(row, dict) and norm(row.get("name")) == n:
            comps = [c.get("name") if isinstance(c, dict) else c for c in row.get("components") or []]
            return f"complex components={comps}"
    for row in payload.get("quarantined_proteins") or []:
        if isinstance(row, dict) and norm(row.get("name")) == n:
            return f"quarantined_proteins reason={row.get('reason')}"
    return "ABSENT (no row, no record)"


NO_TAXA = "--no-taxa" in sys.argv[3:]
for leg in [a for a in sys.argv[3:] if a != "--no-taxa"] or list(LEGS):
    run, rel, targets = LEGS[leg]
    d = W / run / "runs_papers" / rel
    lw = work / leg
    lw.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(W / run / "data" / "id_mapping_cache.json", lw / "id_mapping_cache.json")
    shutil.copyfile(W / run / "data" / "id_mapping_cache.json", lw / "id_mapping_cache_stage6.json")
    merged = json.loads((d / "merged_payload.json").read_text(encoding="utf-8"))
    # The harness technique: NCBI backfill is off offline, so the species rows carry the taxonomy
    # id the leg's own backfill wrote (final_mapped.json). That is what keys the recorded
    # taxon-scoped UniProt answers (``api-v9::<name>::tax:<id>>...``) the run cached.
    final = json.loads((d / "final_mapped.json").read_text(encoding="utf-8"))
    taxa = {norm(s.get("name")): s.get("taxonomy_id") for s in (final.get("entities") or {}).get("species") or []
            if isinstance(s, dict) and s.get("taxonomy_id")}
    for s in [] if NO_TAXA else (merged.get("entities") or {}).get("species") or []:
        if isinstance(s, dict) and not s.get("taxonomy_id") and taxa.get(norm(s.get("name"))):
            s["taxonomy_id"] = taxa[norm(s.get("name"))]
    http = h.ReplayHttp({})
    with h.Doors(http) as doors:
        first = h.stage2(merged, lw)
        normalized, _ = normalize_process_payload(first["payload"])
        st6 = map_ids.map_payload(deepcopy(normalized), cache_path=lw / "id_mapping_cache_stage6.json",
                                  id_source="hybrid", use_cache=True, allow_complex_wrapper_creation=True,
                                  allow_structural_cleanup=True, merge_identity_duplicates=True)["payload"]
    q = quarantine_and_close(st6, strict_db=True, pathway_context=None)
    (lw / "stage6.json").write_text(json.dumps(st6, ensure_ascii=False, indent=1), encoding="utf-8")
    (lw / "quarantine_report.json").write_text(json.dumps(q.quarantine_report, ensure_ascii=False, indent=1, default=str),
                                               encoding="utf-8")
    print(f"\n######## {leg}  tree={tree.name}  live-http-404={len(http.calls)} ai={len(doors.ai_calls)}")
    for t in targets:
        print(f"  target {t!r}: stage6 -> {fate(st6, t)}")
    tnorms = {norm(t) for t in targets}
    kept = [a for a in q.admissions if a.get("state") in ACCEPTED_STATES]
    omitted = [a for a in q.admissions if a.get("state") not in ACCEPTED_STATES]
    for a in omitted:
        text = json.dumps(a.get("detail") or {}, ensure_ascii=False) + " " + str(a.get("name"))
        if any(t and t in norm(text) for t in tnorms) or norm(a.get("essential_participant")) in tnorms:
            print(f"  omitted {a.get('process_kind')} {a.get('name')!r}: {a.get('reason')}")
    for e in q.removed_entity_report.get("removed_entities") or []:
        drops = [(a.get("process_name"), a.get("decision")) for a in (e.get("detail") or {}).get("actor_removed_from") or []]
        if norm(e.get("name")) in tnorms or drops:
            print(f"  removed_entity {e.get('bucket')} {e.get('name')!r}: {e.get('reason')}"
                  + (f"  actor dropped from kept process: {drops}" if drops else ""))
    print(f"  processes: accepted={len(kept)} omitted={len(omitted)}")
