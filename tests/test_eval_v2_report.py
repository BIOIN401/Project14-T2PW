"""NEW ACCEPTANCE TEST (ORCH-742c) -- the committed evaluation-v2 sample report is
reproducible from git-tracked artifacts alone.

Re-runs the offline evaluator over the committed run trees and requires the result to
equal ``docs/evaluation_v2/sample_report.json`` exactly, and the rendered markdown to
equal ``SAMPLE_REPORT.md``. The live-retrieval section is read from its own committed
JSON (it needs an embedding endpoint, so it is not re-measured here). The mapped-ID
reference tables are read as committed at HEAD (git blobs), never from the working tree,
so the dirty runtime caches of an integration checkout cannot move a number (review
finding B-R2-1, repair 2). New capability: no base-SHA failure applies (G9).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
from t2pw.bench.eval_v2.report import evaluate, render_markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evaluation_v2"


def test_committed_sample_report_is_reproducible_from_tracked_artifacts():
    committed = json.loads((OUT / "sample_report.json").read_text(encoding="utf-8"))
    recorded = committed["mapped_id_correctness"]["reference_tables"]
    heads = {rel: subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip() for rel in recorded}
    assert recorded == heads, "a reference table was committed after the report was generated: regenerate it"
    live = json.loads((OUT / "retrieval_live.json").read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(evaluate(ROOT, live), ensure_ascii=False))
    for section in committed:
        assert fresh[section] == committed[section], section
    assert set(fresh) == set(committed)

    md = (OUT / "SAMPLE_REPORT.md").read_text(encoding="utf-8")
    commit = re.search(r"commit `([0-9a-f]{40})`", md).group(1)
    assert render_markdown(committed, commit) == md


def test_committed_report_scores_no_leg_of_the_uncertified_pmc12657337():
    """D-103 (ORCH-744): PMC12657337 is labelled but NOT certified complete (bio-audit
    B-R3-1), so none of its legs feeds reaction precision, recall, the unsupported rate
    or the archived RAG split. Deliberately moved pin (merge rule 4): this replaces
    test_committed_report_no_longer_credits_heptanoyl_hepp, whose row left the report
    with its case; the B2 check on that committed row is kept in the next test."""
    committed = json.loads((OUT / "sample_report.json").read_text(encoding="utf-8"))
    assert committed["gold_supplement"]["complete_cases"] == ["PMC12096016", "PMC12312563", "PMC12421875"]
    assert [l["leg_dir"] for l in committed["reaction_metrics"]["legs"] if l["target_paper"] == "PMC12657337"] == []
    assert [s["leg_dir"] for s in committed["rag_archived_failure_split"]["legs"]
            if s["target_paper"] == "PMC12657337"] == []
    assert [c for l in committed["reaction_metrics"]["legs"] for c in l["metrics"]["name_conflicts_for_hand_review"]] == []


def test_committed_heptanoyl_hepp_row_is_still_a_name_conflict():
    """Review finding B2 (ORCH-742c repair 1) on the committed artifact, scored directly
    now that the report no longer carries the case (D-103): row 17 of the
    runs/2026-07-28_2122 PMC12657337 fallback leg is a name conflict for hand review,
    not a true positive for R1."""
    leg = ROOT / "runs/2026-07-28_2122/papers/PMC12657337__high-level-production-of-vitamin-k2-in-lt-i-gt/research"
    payload = json.loads((leg / "merged_payload.json").read_text(encoding="utf-8"))
    res = R.score_leg(load_supplement()["PMC12657337"], payload, [], None)
    assert [(c["row_index"], c["conflicting_names"]) for c in res["name_conflicts_for_hand_review"]] == [
        (17, ["heptanoyl pyrophosphate (HepPP)"])]
    assert res["rows"][17]["class"] == R.CONFLICT and res["rows"][17]["gold_ids"] == ["R1"]
