"""REV-110 -- B9: dump the serialized acceptance report for one FIXED run.

Run once against the BASE tree and once against the TIP tree, then diff. That
is the strongest form of "no threshold, denominator or D-073 band moved": not a
promise, and not the author's own in-tree walk, but the same input scored by
both trees with C-110's own two keys stripped out.

If anything OTHER than the two new keys differs, the card moved a count.

Usage:  rev110_b9_ab_dump.py <src-dir> <out-json>
No committed run is opened; the run directory is built here from literal rows.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(SRC))

from t2pw.bench.acceptance import MODES, score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402

GOLD = load_gold_set()
DIAG = [{"name": "extraction_diagnostics.json", "bytes": 812}]

#: A deliberately MIXED manifest: every gold case gets both legs, cycling
#: through a clean decline, a timeout, a crash and a pass-shaped row, so the
#: A/B covers denominators, blockers, boundaries and priorities -- not just the
#: two negative-control papers.
SHAPES = [
    dict(status="fail", stage="stage1", failure_kind="no_reactions",
         message="extraction produced no reactions", issue_codes=[],
         counts={"reactions": 0}, files=list(DIAG)),
    dict(status="timeout", stage="unknown", failure_kind="timeout",
         termination_reason="budget_exhausted", operational_failure=True,
         message="killed at the wall-clock ceiling", issue_codes=[],
         counts={}, files=[]),
    dict(status="error", stage="unknown", failure_kind="crash",
         message="the child exited without printing a result line",
         issue_codes=[], counts={}, files=[]),
    dict(status="fail", stage="stage2", failure_kind="contract",
         message="a required field was missing", issue_codes=["gate.required_field"],
         counts={"reactions": 0}, files=list(DIAG)),
]

rows = []
for index, case in enumerate(GOLD):
    for offset, mode in enumerate(MODES):
        shape = dict(SHAPES[(index + offset) % len(SHAPES)])
        shape.update(paper_id=case.paper_id, slug=case.paper_id, mode=mode)
        rows.append(shape)

tmp = Path(tempfile.mkdtemp(prefix="rev110_ab_"))
try:
    run = tmp / "2026-01-01_0000"
    (run / "papers").mkdir(parents=True)
    (run / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    report = score_run(run, GOLD)
    data = report.to_dict()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

NEW_KEYS = {"negative_control", "negative_control_outcomes"}


def strip(node):
    """Remove C-110's own two keys. Everything left must be identical A vs B."""
    if isinstance(node, dict):
        return {k: strip(v) for k, v in node.items() if k not in NEW_KEYS}
    if isinstance(node, list):
        return [strip(x) for x in node]
    return node


stripped = strip(data)
# run_dir is a temp path and differs by construction; it is not a count.
stripped.pop("run_dir", None)
OUT.write_text(json.dumps(stripped, indent=2, sort_keys=True, default=str),
               encoding="utf-8")

removed = sorted(
    k for k in NEW_KEYS
    if json.dumps(data, default=str).find(f'"{k}"') >= 0
)
print(f"src            : {SRC}")
print(f"rows scored    : {len(rows)}   papers: {len(report.papers)}")
print(f"C-110 keys seen: {removed or '(none -- this is the BASE tree)'}")
print(f"stripped bytes : {len(OUT.read_text(encoding='utf-8'))}")
print(f"written        : {OUT}")
