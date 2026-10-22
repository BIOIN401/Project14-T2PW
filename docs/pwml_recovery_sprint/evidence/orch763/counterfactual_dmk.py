"""ORCH-763 (b) COUNTERFACTUAL, NOT PRODUCTION CODE: the menaquinone ORCH-759 pool if the paper
dictionary held the two pairs the paper writes but Schwartz-Hearst cannot read.

The paper writes "menaquinone (MK, vitamin K2)" and "demethylmenaquinone (DMK)"; SH rejects both
because "K" occurs in neither long form (diag_dmk.py). This script adds exactly those two pairs to
the lexicon the merge builds (a monkeypatch of PaperLexicon.__init__ in this process only), re-pools
the rebuilt draws and scores them. It answers "what would a ruling buy", nothing else.

Usage (through bounded_run.py): PYTHONPATH=<tip>/src python counterfactual_dmk.py <repo root> <leg 01_source_text.txt>
"""
import json
import sys
from pathlib import Path

import t2pw.pipeline.paper_lexicon as PL
import t2pw.pipeline.stage_one_ensemble as E
from t2pw.bench.eval_v2 import reactions as RX
from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement
from t2pw.pipeline.pipeline import chunk_text

REPO = Path(sys.argv[1])
fixture = json.loads((REPO / "tests/fixtures/orch763/menaquinone_orch759_draws.json").read_text(encoding="utf-8"))
text = next(c["text"] for c in chunk_text(Path(sys.argv[2]).read_text(encoding="utf-8"), 10000, 1600)
            if len(c["text"]) == fixture["source"]["side_file_paper_chars"])
case = load_supplement()["PMC12421875"]
paper = load_fulltext(case, REPO / "docs/pwml_recovery_sprint/evidence/orch754a_inputs/papers")


def pool(label):
    merged, report = E.merge_draws(fixture["draws"], paper_text=text)
    rows = report["buckets"]["/processes/reactions"]["rows"]
    t1 = RX.score_leg(case, merged, [], paper, expected_species=case.species)["table1_paper_extraction"]
    print(f"{label}: pooled {len(rows)}; gold matched {len(t1['gold_reactions_matched'])}/{t1['gold_reactions']} "
          f"P {t1['precision']:.3f}; names rewritten {len(report['names_rewritten'])}")
    for r in rows:
        if r["name"].startswith(("MenA", "MenG")):
            print(f"   {r['name']!r} draws {r['draws']} key {r['key']!r} merges "
                  f"{[(m['draw'], m['bridged_by'], (m.get('location_flag') or {}).get('locations')) for m in r.get('merges') or []]}")


pool("production lexicon (tip)")
real_init = PL.PaperLexicon.__init__


def with_two_pairs(self, text):
    real_init(self, text)
    self.pairs.setdefault("DMK", ("DMK", "demethylmenaquinone", "counterfactual"))
    self.pairs.setdefault("MK", ("MK", "menaquinone", "counterfactual"))


PL.PaperLexicon.__init__ = with_two_pairs  # the class merge_draws builds (E imports the same object)
pool("COUNTERFACTUAL lexicon (+ DMK, + MK)")
