"""C-056c / G9: is a 1-of-4 semantic pass distinguishable from a 4-of-4 pass?

ONE script, EITHER tree, taken as ``argv[1]``. That is the point: the proof is a
BEHAVIOURAL comparison of two serialized artifacts, never the presence of a symbol.
``_classify`` adapts to either arity of ``semantic_verdict`` -- three values at the base
SHA, four at the tip -- so the script runs unchanged on both sides and the only thing that
differs is what comes out.

Pure functions only: no database, no subprocess, no pytest, no network. Run it as::

    <py> bounded_run.py --timeout 120 --label g9 --json <report> -- \
         <py> c056c_g9_probe.py <tree-root> [--out <capture.json>]

At the base the two serializations are byte-identical and ``distinguishable`` is False.
At the tip they differ by exactly one key and ``distinguishable`` is True.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve() if "--out" in sys.argv else None
sys.path.insert(0, str(ROOT / "src"))

import t2pw  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    SEMANTIC_GATING_CHECKS,
    classify_release_status,
    semantic_verdict,
)

#: Clears the coverage threshold, so nothing but the semantic record is under test.
COVERAGE = {"requested_core_declared": True, "surviving_processes": 3,
            "minimum_core_satisfied": True, "reasons": []}


class _Check:
    def __init__(self, reason: str = "") -> None:
        self.ok, self.inapplicable_reason = True, reason

    @property
    def applicable(self) -> bool:
        return not self.inapplicable_reason


class _Report:
    evaluated, not_evaluated_reason = True, ""

    def __init__(self, inapplicable=()) -> None:
        self.checks = {
            name: _Check(f"{name} could not be evaluated here" if name in inapplicable else "")
            for name in SEMANTIC_GATING_CHECKS
        }


def _classify(report):
    """The arity-tolerant shim -- identical call shape at base and tip."""

    verdict = semantic_verdict(report)
    extra = {"semantic_check_evaluability": verdict[3]} if len(verdict) > 3 else {}
    return classify_release_status(
        COVERAGE, strict_gates_passed=True, semantic_evaluation=verdict[0],
        semantic_not_evaluated_reason=verdict[1], semantic_failed_checks=verdict[2], **extra)


def _serialized(report) -> str:
    return json.dumps(_classify(report).to_dict(), sort_keys=True)


one, four = _Report(SEMANTIC_GATING_CHECKS[1:]), _Report()
record = {
    "tree": str(ROOT),
    "t2pw_file": t2pw.__file__,
    "verdict_arity": len(semantic_verdict(four)),
    "verdict_one_of_four": _classify(one).semantic_evaluation,
    "verdict_four_of_four": _classify(four).semantic_evaluation,
    "serialized_one_of_four": _serialized(one),
    "serialized_four_of_four": _serialized(four),
    "distinguishable": _serialized(one) != _serialized(four),
    "distinct_serializations_across_widths_0_to_4": len(
        {_serialized(_Report(SEMANTIC_GATING_CHECKS[w:])) for w in range(5)}
    ),
}
for key, value in record.items():
    print(f"{key:44}: {value}")
if OUT is not None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT} (exists={OUT.is_file()})")
