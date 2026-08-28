"""C-102 / REV-102 F3 and F5: what the reconciliation actually costs a report.

Serializes ``score_run(run).to_dict()`` for a few committed run directories and
prints the byte size. Run at the base SHA and at the tip, the difference is the
cost of the D-072 keys -- including on a run with NO leg carrying a coverage
block, which is the case the first round's TEST_MATRIX note got wrong.

Usage::

    <python> c102_report_size.py <repo-root> [run-dir-name ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
NAMES = [a for a in sys.argv[2:] if not a.startswith("--")] or ["2026-08-04_1207", "2026-08-24_1428"]

from t2pw.bench.acceptance import AcceptanceReport, score_run  # noqa: E402

# F5 was a size decision, so the size it avoided is measured here rather than
# quoted. Restoring the pre-F5 shape -- the priority entries carrying the corpus
# record WHOLE -- is a property swap, not a file edit, so nothing on disk moves.
if "--as-if-uncompacted" in sys.argv:
    AcceptanceReport.coverage_reconciliation_summary = (
        AcceptanceReport.coverage_reconciliation_corpus
    )
    print("MEASURING THE PRE-F5 SHAPE: priority entries carry the corpus record whole")

for name in NAMES:
    run = ROOT / "runs_verify" / name
    report = score_run(run)
    data = report.to_dict()
    blob = json.dumps(data, sort_keys=True)
    legs_with_coverage = sum(
        1
        for paper in data["papers"]
        for leg in paper["legs"].values()
        if leg.get("coverage_reconciliation")
    )
    corpus = data.get("coverage_reconciliation_corpus")
    per_entry = [
        len(json.dumps(entry.get("requested_core_coverage"), sort_keys=True))
        for entry in data["acceptance_priorities"]
        if entry.get("requested_core_coverage") is not None
    ]
    print(
        f"{name:18s} bytes={len(blob):>7}  legs_with_coverage_block={legs_with_coverage:>2}"
        f"  corpus_key={'present' if corpus is not None else 'ABSENT':>7}"
        f"  corpus_bytes={len(json.dumps(corpus, sort_keys=True)) if corpus is not None else 0:>6}"
        f"  priority_copies={per_entry}"
    )
