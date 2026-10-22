"""ORCH-758: Stage-1 boundary records and ensemble side files for every ORCH-755 leg. Stdlib only, reads only.

Usage: python orch758_stage1_boundaries.py <runs_papers/ORCH-755/night.log>

The night log names, per leg, the tmp/extraction_diagnostics/run-<id>/ directory the review
script picked by time window. From it this prints, in order, every Stage-1 boundary record
(one per extraction call, ensemble and ladder termination) and, per ensemble (one per chunk),
the draws, the chunk's character count, pooled reactions, merges by rule, kept-apart reasons
and flagged merges.
"""
import collections
import json
import os
import sys

night = sys.argv[1]
for line in open(night, encoding="utf-8"):
    if not line.startswith("{"):
        continue
    leg = json.loads(line)
    side = [p for p in leg.get("ensemble_side_files") or [] if p.endswith(".json")]
    print(f"\n=== {leg['cond']}/{leg['mode']}  exit={leg['exit']}  {leg['minutes']} min  result={leg['result']}")
    if not side:
        print("   no side file")
        continue
    run_dir = os.path.dirname(side[0])
    print("   dir", run_dir)
    bpath = os.path.join(run_dir, "extraction_boundary_report.json")
    if os.path.exists(bpath):
        rep = json.load(open(bpath, encoding="utf-8"))
        print(f"   boundaries={rep.get('boundary_count')} outcomes={rep.get('outcomes')}")
        for b in rep.get("boundaries") or []:
            if b.get("boundary") == "stage1_ensemble":
                e = b.get("ensemble") or {}
                print(f"   [ensemble] draws {e.get('draws_succeeded')}/{e.get('draws_requested')} failed={e.get('failed_draws')} "
                      f"reactions={((e.get('by_bucket') or {}).get('/processes/reactions') or {}).get('merged')}")
            elif b.get("boundary") == "stage1_extraction_ladder_termination":
                lad = b.get("extraction_ladder") or {}
                print(f"   [ladder end] outcome={b.get('outcome')} issued={lad.get('attempts_issued')}/{lad.get('attempt_cap')} "
                      f"rungs={[a.get('rung') for a in lad.get('attempts') or []]} "
                      f"skipped={[(s.get('rung'), s.get('skip_cause')) for s in lad.get('skipped_steps') or []]}")
            else:
                pc = (b.get("raw_process_counts") or {})
                ec = (b.get("raw_entity_counts") or {})
                print(f"   [{b.get('boundary')}] outcome={b.get('outcome')} finish={b.get('finish_reason')} "
                      f"model={b.get('model')} chars={b.get('raw_chars')} processes={pc.get('total')} "
                      f"(rx {pc.get('reactions')}) entities={ec.get('total')} req={str(b.get('request_hash'))[-6:]} "
                      f"note={b.get('note', '')} attempts={[(a.get('status'), a.get('finish_reason'), a.get('content_chars')) for a in b.get('attempt_log') or []]}")
    for i, r in enumerate(json.load(open(side[0], encoding="utf-8"))):
        rx = (r.get("buckets") or {}).get("/processes/reactions") or {}
        rows = rx.get("rows") or []
        rules = collections.Counter(rule for row in rows for m in row.get("merges") or [] for rule in m.get("bridged_by") or [])
        apart = [(row.get("name"), k.get("name"), k.get("reason")) for row in rows for k in row.get("kept_apart_from") or []]
        flagged = sum(1 for row in rows for m in row.get("merges") or [] if m.get("flag") or m.get("typo_flag"))
        lex = r.get("paper_lexicon") or {}
        print(f"   ensemble #{i + 1}: draws {r.get('draws_succeeded')}/{r.get('draws_requested')} failed={r.get('failed_draws')} "
              f"chunk_chars={lex.get('paper_chars')} abbrev_pairs={lex.get('abbreviation_pairs')} "
              f"reactions={len(rows)} per_draw={rx.get('rows_per_draw')} one_draw={rx.get('unique_to_one_draw')} "
              f"all_draws={rx.get('in_every_successful_draw')} merge_rules={dict(rules)} flagged={flagged} "
              f"names_rewritten={len(r.get('names_rewritten') or [])} typo_snapped={len(r.get('typo_snapped') or [])}")
        for a in apart:
            print(f"      kept apart: {a[0]!r} vs {a[1]!r}: {a[2]}")
