"""C-110 -- what a reader actually sees. Rendered from a CONSTRUCTED run.

REV-110 B5 asks for the raw outcome and the rejection reason both present in the
output. This prints the page. Two legs on the declared negative control -- one
that declined correctly, one killed by the clock -- plus the second arm's
``context_only`` case, so the two are side by side on one report.

**The run directory is built here, in a temp dir, and deleted. No committed run
is scored and T-107 is not opened.**

Usage::

    <venv-python> c110_rendered_sample.py <worktree-root>
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.acceptance import score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.bench.render import render_text  # noqa: E402

ROWS = [
    {
        "paper_id": "PMC13231680",
        "slug": "PMC13231680",
        "mode": "strict",
        "status": "fail",
        "stage": "stage1",
        "failure_kind": "no_reactions",
        "message": (
            "extraction produced no reactions: nothing lipid-A-related is present in "
            "this paper at any level of partiality, so no pathway was exported"
        ),
        "issue_codes": [],
        "counts": {"reactions": 0, "transports": 0, "entities": 0},
        "files": [{"name": "extraction_diagnostics.json", "bytes": 812}],
    },
    {
        "paper_id": "PMC13231680",
        "slug": "PMC13231680",
        "mode": "research",
        "status": "timeout",
        "stage": "unknown",
        "failure_kind": "timeout",
        "termination_reason": "budget_exhausted",
        "operational_failure": True,
        "message": (
            "the child process was still running after 1800s and was killed, so this "
            "paper+mode produced nothing (budget_exhausted)"
        ),
        "issue_codes": [],
        "counts": {},
        "files": [],
    },
    {
        "paper_id": "PMC12180156",
        "slug": "PMC12180156",
        "mode": "strict",
        "status": "fail",
        "stage": "stage1",
        "failure_kind": "no_reactions",
        "message": "no heme biosynthesis chemistry was extractable from this paper",
        "issue_codes": [],
        "counts": {"reactions": 0, "transports": 0, "entities": 0},
        "files": [{"name": "extraction_diagnostics.json", "bytes": 401}],
    },
]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="c110_render_"))
    try:
        run_dir = tmp / "2026-01-01_0000"
        (run_dir / "papers").mkdir(parents=True)
        (run_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8"
        )
        report = score_run(run_dir, load_gold_set())
        text = render_text(report)

        print("=" * 78)
        print("THE NEW SECTION")
        print("=" * 78)
        # `_header` is [RULE, title, RULE], so a naive "stop at the next rule"
        # stops on the header's own closing rule and prints one line. Capture
        # from the title and stop at the NEXT header title instead.
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if "NEGATIVE CONTROLS" in line)
        end = next(
            (
                i
                for i in range(start + 3, len(lines))
                if lines[i].startswith("=" * 20)
            ),
            len(lines),
        )
        for line in lines[start - 1 : end]:
            print(line)

        print()
        print("=" * 78)
        print("THE PER-LEG LINES -- raw verdict ABOVE, new status BESIDE it")
        print("=" * 78)
        for block in ("PMC13231680", "PMC12180156"):
            printing = False
            for line in text.splitlines():
                if line.startswith(block):
                    printing = True
                elif printing and line.startswith("-" * 20):
                    break
                if printing:
                    print(line)
            print()

        print("=" * 78)
        print("SERIALIZED, per leg -- the raw record travels inside the new one")
        print("=" * 78)
        data = report.to_dict()
        for paper in data["papers"]:
            for mode, leg in paper["legs"].items():
                if "negative_control" not in leg:
                    continue
                print(f"{paper['paper_id']}:{mode}")
                print(f"  leg status (RAW, untouched) : {leg['status']!r}")
                print(f"  negative_control            : {json.dumps(leg['negative_control'], indent=4)}")
                print()
        print(f"report-level roll-up: {data.get('negative_control_outcomes')}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
