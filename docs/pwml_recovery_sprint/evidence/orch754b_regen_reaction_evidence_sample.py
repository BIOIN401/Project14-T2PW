"""ORCH-754b repair round 1 -- regenerate ``docs/reaction_evidence_sample.txt``.

The sample is, by its own test (``tests/test_reaction_evidence.py`` test 2), this module run
over the committed leg ``runs_verify/2026-09-01_1612/papers/PMC12096016/strict``. ORCH-754b
adds the INTERACTIONS section (and ``pwml_ir.json`` to the artifacts read), so the sample moves
deliberately (merge rule 4); this is the exact command that produced it. Prints a unified diff
of old -> new so the move is reviewable from the run's own output.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEG = REPO / "runs_verify" / "2026-09-01_1612" / "papers" / "PMC12096016" / "strict"
SAMPLE = REPO / "docs" / "reaction_evidence_sample.txt"


def main() -> int:
    from t2pw.app.reaction_evidence import build_report, load_run_artifacts, render_text

    new = render_text(build_report(load_run_artifacts(LEG)))
    old = SAMPLE.read_text(encoding="utf-8")
    sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "sample (old)", "sample (new)"))
    SAMPLE.write_bytes(new.replace("\n", "\r\n").encode("utf-8"))  # the checkout's own line endings
    return 0


if __name__ == "__main__":
    sys.exit(main())
