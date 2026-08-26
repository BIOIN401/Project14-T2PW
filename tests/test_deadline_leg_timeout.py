"""D-005 leg-timeout classification and guaranteed child cleanup. C-032.

The three ``g9`` tests here fail BEHAVIOURALLY on the dispatch base ``cbf30f6``:
it records every timeout as one undifferentiated ``failure_kind="timeout"``, so
``budget_exhausted`` and ``operation_timeout`` cannot be told apart in either the
manifest row or the driver outcome, and ``launch_child`` returns with the child
still alive when the wait raises anything but ``TimeoutExpired`` /
``KeyboardInterrupt``. The base has no ``t2pw.pipeline.deadline`` at all, so the
vocabulary is shimmed below and every assertion then fails on CLASSIFICATION or
ROW CONTENT rather than on an import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.batch import driver, runner  # noqa: E402

try:  # the base SHA has no deadline module: shim the vocabulary, never the verdict
    from t2pw.pipeline.deadline import (  # noqa: E402
        BUDGET_EXHAUSTED,
        BUDGET_SPENT_MARKER,
        OPERATION_TIMEOUT,
        OPERATIONAL_TERMINATION_REASONS,
    )
except ImportError:  # pragma: no cover - reached only on the base SHA
    BUDGET_EXHAUSTED = "budget_exhausted"
    OPERATION_TIMEOUT = "operation_timeout"
    BUDGET_SPENT_MARKER = "whole-run budget of"
    OPERATIONAL_TERMINATION_REASONS = frozenset({BUDGET_EXHAUSTED, OPERATION_TIMEOUT})

PAPER: Dict[str, Any] = {"paper_id": "PMC1", "title": "a paper"}
#: The four reasons a stopped clock may never be relabelled as.
NEVER = ("retrieval_exhausted", "no_new_claims", "identical_empty_response",
         "scientifically_unrecoverable")


def _row(seconds: float, timeout: float = 3600.0) -> Dict[str, Any]:
    """A killed-child row through the production signature only (base-compatible)."""

    return runner._timeout_row(slug="PMC1__a", mode="strict", paper=PAPER,
                               seconds=seconds, timeout=timeout, tail="")


def _timed_out(detail: str) -> driver.RunOutcome:
    outcome = driver.RunOutcome(paper_id="PMC1", mode="strict")
    driver._finalize_timeout(outcome, message="did not finish", detail=detail)
    return outcome


# --------------------------------------------------------------------------
# G9 regression
# --------------------------------------------------------------------------
def test_g9_a_killed_child_row_says_which_clock_ran_out() -> None:
    """G9 regression: the base row cannot distinguish the two D-005 outcomes."""

    spent = _row(3600.4)
    assert spent["termination_reason"] == BUDGET_EXHAUSTED
    assert spent["operational_failure"] is True
    assert spent["budget"]["elapsed_seconds"] == 3600.4
    assert spent["budget"]["remaining_seconds"] == -0.4
    assert spent["budget"]["leg_timeout_seconds"] == 3600.0
    assert spent["budget"]["child_deadline_seconds"] == 3480.0
    assert spent["budget"]["leg_timeout_overridden"] is False
    # Killed while leg budget remained: a tighter bound fired, not the leg's.
    early = _row(400.0)
    assert early["termination_reason"] == OPERATION_TIMEOUT
    assert early["budget"]["remaining_seconds"] == 3200.0


def test_g9_a_driver_timeout_records_which_clock_ran_out() -> None:
    """G9 regression: on the base both details produce the same undifferentiated row."""

    spent = _timed_out(f"{BUDGET_SPENT_MARKER} 3600s was already spent")
    assert getattr(spent, "termination_reason", None) == BUDGET_EXHAUSTED
    assert getattr(spent, "termination_is_operational", None) is True
    overran = _timed_out("app.run() timed out after 45s")
    assert getattr(overran, "termination_reason", None) == OPERATION_TIMEOUT
    assert getattr(overran, "termination_is_operational", None) is True


def test_g9_launch_child_kills_the_child_when_the_wait_raises(monkeypatch: Any) -> None:
    """G9 regression: on the base only TimeoutExpired/KeyboardInterrupt clean up."""

    killed: List[Any] = []

    class FakePopen:
        pid, returncode = 5151, None

        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def communicate(self, timeout: Any = None) -> Any:
            raise MemoryError("the drain died")

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner, "_kill_tree", lambda proc: killed.append(proc.pid))
    with pytest.raises(MemoryError):
        runner.launch_child([sys.executable, "-c", "pass"], 1.0)
    assert killed == [5151], "the owned child outlived the job"


# --------------------------------------------------------------------------
# Preservation
# --------------------------------------------------------------------------
def test_the_killed_child_row_keeps_every_field_the_aggregators_rank_on() -> None:
    """Preservation: status, failure_kind and the timeout number are unmoved."""

    row = _row(1234.5, timeout=1234.0)
    assert row["status"] == "timeout" and row["failure_kind"] == "timeout"
    assert row["stage"] == "unknown" and row["seconds"] == 1234.5
    assert "1234" in row["message"]
    assert row["issue_codes"] == [] and row["counts"] == {} and row["files"] == []
    assert row["slug"] == "PMC1__a" and row["paper_id"] == "PMC1"


def test_a_driver_timeout_row_carries_the_classification_and_nothing_else() -> None:
    """**RE-BASELINED by C-083 under merge rule 4.** Was
    ``test_a_driver_timeout_keeps_its_row_byte_identical``.

    **What this test used to pin, and why that was interim.** It asserted
    ``"termination_reason" not in row``, on the reasoning in its own docstring:
    "the classification is an attribute, never a manifest field", because "the
    golden driver diff hashes ``RunOutcome.to_dict()``; a new key here would move
    the pinned ``input_timeout`` leg". Both halves were accurate statements of
    C-032's ownership boundary -- ``_finalize_timeout``'s docstring recorded the
    same limit -- and neither was a product decision. F-092 defect 3 measured the
    consequence: the verdict was computed and then dropped by the serializer, so
    both INNER timeout rows on disk (``runs_verify/2026-08-21_1822`` research,
    ``runs_verify/2026-08-22_2147`` strict) reached the manifest with
    ``failure_kind="timeout"`` and no stop reason, against ``PRODUCT_CONTRACT.md``
    section 9. D-004 gave the row an owner and C-083 is that card.

    **The obligation is not deleted, it is inverted and tightened.** The old
    assertion could only catch a key being added. This one catches a key being
    added *that was not intended*, a key being dropped to stabilise a digest, and
    the intended keys going missing -- as a set difference against a fresh
    ``RunOutcome``, so it cannot rot into prose.

    **The exact golden delta this authorizes**, in
    ``tests/test_batch_driver_seam_golden.py``: slot 2 of ``input_timeout`` alone.
    Slots 0 and 1 do not move on any leg, and the other six legs never time out, so
    their digests do not move either. Captures are committed beside the golden.
    """

    before = driver.RunOutcome(paper_id="PMC1", mode="strict").to_dict()
    row = _timed_out("app.run() timed out after 45s").to_dict()

    # Unmoved: everything the aggregators rank on.
    assert row["status"] == "timeout" and row["failure_kind"] == "timeout"
    assert row["message"] == "did not finish"
    assert "release_status" not in row

    # Moved, deliberately and by exactly this much.
    assert row["termination_reason"] == OPERATION_TIMEOUT
    assert row["operational_failure"] is True
    assert sorted(set(row) - set(before)) == [
        "budget_unrecorded", "operational_failure", "termination_reason",
    ], f"an unintended key reached the timeout row: {sorted(set(row) - set(before))}"
    assert set(before) - set(row) == set(), "no key was dropped to stabilise a digest"


def test_the_drivers_own_budget_signal_still_carries_the_marker() -> None:
    """Preservation: ``_run_app``'s exhausted-budget wording and the marker agree.

    The classifier reads that wording, so the two must not drift apart silently.
    """

    budget = driver._Budget(3600.0, 45.0)
    budget.started -= 7200.0  # monotonic start pushed back: the leg is spent
    timed_out, detail = driver._run_app(None, budget)
    assert timed_out is True and BUDGET_SPENT_MARKER in detail
    assert getattr(_timed_out(detail), "termination_reason", None) == BUDGET_EXHAUSTED


def test_a_timeout_is_an_operational_fact_never_a_biological_verdict() -> None:
    """Preservation of the sharpest invariant, on both timeout paths."""

    for reason in (_row(3600.4)["termination_reason"], _row(9.0)["termination_reason"],
                   getattr(_timed_out("timed out"), "termination_reason", None)):
        assert reason in OPERATIONAL_TERMINATION_REASONS and reason not in NEVER
