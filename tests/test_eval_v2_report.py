"""NEW ACCEPTANCE TEST (ORCH-742c) -- the committed evaluation-v2 sample report is
reproducible from git-tracked artifacts alone.

Re-runs the offline evaluator over the committed run trees and requires the result to
equal ``docs/evaluation_v2/sample_report.json`` exactly, and the rendered markdown to
equal ``SAMPLE_REPORT.md``. The live-retrieval section is read from its own committed
JSON (it needs an embedding endpoint, so it is not re-measured here). New capability:
no base-SHA failure applies (G9).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from t2pw.bench.eval_v2.report import evaluate, render_markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evaluation_v2"


def test_committed_sample_report_is_reproducible_from_tracked_artifacts():
    committed = json.loads((OUT / "sample_report.json").read_text(encoding="utf-8"))
    live = json.loads((OUT / "retrieval_live.json").read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(evaluate(ROOT, live), ensure_ascii=False))
    for section in committed:
        assert fresh[section] == committed[section], section
    assert set(fresh) == set(committed)

    md = (OUT / "SAMPLE_REPORT.md").read_text(encoding="utf-8")
    commit = re.search(r"commit `([0-9a-f]{40})`", md).group(1)
    assert render_markdown(committed, commit) == md


def test_committed_report_no_longer_credits_heptanoyl_hepp():
    """Review finding B2 (ORCH-742c repair 1) on the committed artifact: row 17 of the
    runs/2026-07-28_2122 PMC12657337 fallback leg is a name conflict for hand review,
    not a true positive for R1."""
    committed = json.loads((OUT / "sample_report.json").read_text(encoding="utf-8"))
    legs = [l for l in committed["reaction_metrics"]["legs"]
            if l["leg_dir"].startswith("runs/2026-07-28_2122/papers/PMC12657337")]
    assert legs
    conflicts = [c for l in legs for c in l["metrics"]["name_conflicts_for_hand_review"]]
    assert [(c["row_index"], c["conflicting_names"]) for c in conflicts] == [
        (17, ["heptanoyl pyrophosphate (HepPP)"])]
    for l in legs:
        assert all(d["row_index"] != 17 or d["v2_class"] != "target_paper_supported"
                   for d in l["metrics"]["old_lenient_matcher_same_rows"]["rows_where_old_and_v2_disagree"])
