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
