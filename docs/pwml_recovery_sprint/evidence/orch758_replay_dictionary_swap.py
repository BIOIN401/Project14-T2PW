"""ORCH-758: does the dictionary text (full paper vs the app's chunk 1) explain 29 vs 41? Offline, no model.

Usage: PYTHONPATH=src python orch758_replay_dictionary_swap.py <PMC5130987 text>

Pools the three ORCH-751 chlorophyll draws (orch752_chlorophyll_payloads/, the ones ORCH-754a's
replay pooled to 29) with the production merge_draws at this SHA, once with the full paper as
the dictionary (what the replay did) and once with chunk 1 only (what the ORCH-755 app leg
gave Stage 1). If both give 29, the ORCH-755 41 comes from the draws, not from chunking.
"""
import json
import sys
from pathlib import Path

from t2pw.pipeline.pipeline import chunk_text
from t2pw.pipeline.stage_one_ensemble import merge_draws

HERE = Path(__file__).resolve().parent
draws = [json.loads((HERE / "orch752_chlorophyll_payloads" / f"control__run{i}__payload.json").read_text(encoding="utf-8"))
         for i in (1, 2, 3)]
full = Path(sys.argv[1]).read_text(encoding="utf-8")
chunk1 = chunk_text(full, 10000, 1600)[0]["text"]
for label, text in (("full paper", full), ("chunk 1 only", chunk1)):
    _, report = merge_draws(draws, paper_text=text)
    rx = report["buckets"]["/processes/reactions"]
    flagged = sum(1 for r in rx["rows"] for m in r.get("merges") or [] if m.get("flag") or m.get("typo_flag"))
    print(f"{label:<13} chars={len(text):>6} pairs={report['paper_lexicon']['abbreviation_pairs']:>3} "
          f"per_draw={rx['rows_per_draw']} pooled={rx['merged']} one_draw={rx['unique_to_one_draw']} flagged={flagged}")
