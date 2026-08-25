"""F-092 defect 3: the INNER timeout path must write down the reason it computed.

``driver._finalize_timeout`` has classified correctly since C-032 -- it asks
``deadline.classify_interaction_timeout`` which of D-005's two operational clocks
ran out, and stores the answer on the outcome. ``RunOutcome`` then threw it away:
neither ``termination_reason`` nor ``termination_is_operational`` was a declared
field, and ``RunOutcome.to_dict`` emitted neither, so the verdict never reached
the manifest row that is the leg's only durable record.

**Measured on the committed runs, not inferred.** Across the four ``runs_verify``
manifests that contain a timeout row there are eight such rows. The four written
by the OUTER path (``runner._timeout_row``, "The parent kills a stuck child...")
that post-date C-032 all carry ``termination_reason``. Both rows written by the
INNER path -- ``PMC12444477`` research at ``2026-08-21_1822`` and
``PMC12444477`` strict at ``2026-08-22_2147``, whose ``detail`` strings are
replayed verbatim below -- carry none, on runs where the classifier had already
computed ``operation_timeout``. ``grep -ric operation_timeout`` over
``runs_verify/2026-08-21_2239`` and ``runs_verify/2026-08-22_2147`` finds nothing.

That violates ``PRODUCT_CONTRACT.md`` section 9: a terminal record preserves "the
exact stop reason". ``failure_kind="timeout"`` says the clock was involved, never
WHICH clock, and D-005 exists because those are different outcomes with different
fixes.

**These are G9 base-failing proofs, not symbol-absence checks.** Every assertion
below feeds a real stored ``detail`` string through the production seam and reads
the emitted ROW. On the base SHA the classification is computed exactly as it is
at the tip and the row still does not carry it, so each fails on ROW CONTENT.

Offline by construction: no live leg, no LLM-backed call, no reproduced timeout.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.batch import driver, runner  # noqa: E402

try:  # shimmed so a base run fails on ROW CONTENT, never on an import
    from t2pw.pipeline.deadline import (  # noqa: E402
        BUDGET_EXHAUSTED,
        BUDGET_SPENT_MARKER,
        OPERATION_TIMEOUT,
        OPERATIONAL_TERMINATION_REASONS,
    )
except ImportError:  # pragma: no cover - reached only on a base without C-032
    BUDGET_EXHAUSTED = "budget_exhausted"
    OPERATION_TIMEOUT = "operation_timeout"
    BUDGET_SPENT_MARKER = "whole-run budget of"
    OPERATIONAL_TERMINATION_REASONS = frozenset({BUDGET_EXHAUSTED, OPERATION_TIMEOUT})

#: Verbatim from ``runs_verify/2026-08-22_2147/manifest.jsonl`` -- PMC12444477,
#: strict, ``stage="stage1"``. The leg the T-105 strict verify lost.
T105_STRICT_DETAIL = "AppTest script run timed out after 471.9942160999999(s)"
#: Verbatim from ``runs_verify/2026-08-21_1822/manifest.jsonl`` -- PMC12444477,
#: research, ``stage="input"``. The other INNER timeout on disk.
T105_RESEARCH_DETAIL = "AppTest script run timed out after 1672.8577338(s)"

PAPER: Dict[str, Any] = {"paper_id": "PMC12444477", "title": "a paper"}

#: The four reasons a stopped clock may never be relabelled as. D-005.
NEVER = ("retrieval_exhausted", "no_new_claims", "identical_empty_response",
         "scientifically_unrecoverable")


def _inner_row(detail: str, *, reason: str = "") -> Dict[str, Any]:
    """The manifest row an INNER timeout produces, through the production seam."""

    outcome = driver.RunOutcome(paper_id="PMC12444477", mode="strict")
    driver._finalize_timeout(
        outcome,
        message="extraction did not finish inside the time budget",
        detail=detail,
        reason=reason,
    )
    return outcome.to_dict()


# ---------------------------------------------------------------------------
# G9. Base-failing on ROW CONTENT.
# ---------------------------------------------------------------------------
def test_g9_the_lost_t105_strict_leg_now_says_which_clock_ran_out() -> None:
    """G9: replaying the stored T-105 strict ``detail``, the row states the reason.

    On the base the same call classifies ``operation_timeout`` onto the outcome
    and ``to_dict()`` returns a row with no such key -- exactly what is on disk at
    ``runs_verify/2026-08-22_2147``. Fails there with ``'termination_reason' not
    in row``, which is a behavioural failure of the record, not a missing symbol.
    """

    row = _inner_row(T105_STRICT_DETAIL)

    assert "termination_reason" in row, (
        "the INNER timeout row does not carry the stop reason it computed "
        f"(keys: {sorted(row)})"
    )
    assert row["termination_reason"] == OPERATION_TIMEOUT
    assert row["operational_failure"] is True
    # The facts the aggregators already rank on are untouched beside it.
    assert row["status"] == "timeout" and row["failure_kind"] == "timeout"
    assert row["detail"] == T105_STRICT_DETAIL


def test_g9_the_other_stored_inner_timeout_is_classified_the_same_way() -> None:
    """G9: the research leg at ``2026-08-21_1822`` -- same defect, same repair."""

    row = _inner_row(T105_RESEARCH_DETAIL)
    assert row["termination_reason"] == OPERATION_TIMEOUT
    assert row["operational_failure"] is True


def test_g9_a_spent_leg_budget_reaches_the_row_as_budget_exhausted() -> None:
    """G9: the OTHER inner verdict survives too, and is not collapsed into one.

    ``_run_app`` refuses to start an interaction once the whole-run clock is gone
    and says so in ``detail``. The classifier reads that marker. Both verdicts
    must reach the row, or the row has merely traded one undifferentiated label
    for another.
    """

    spent = _inner_row(f"{BUDGET_SPENT_MARKER} 3600s was already spent")
    overran = _inner_row(T105_STRICT_DETAIL)

    assert spent["termination_reason"] == BUDGET_EXHAUSTED
    assert overran["termination_reason"] == OPERATION_TIMEOUT
    assert spent["termination_reason"] != overran["termination_reason"]
    assert spent["operational_failure"] is overran["operational_failure"] is True


def test_g9_an_explicit_caller_supplied_reason_reaches_the_row() -> None:
    """G9: the ``reason=`` channel C-042 will use is serialized, not just stored."""

    row = _inner_row("anything at all", reason=BUDGET_EXHAUSTED)
    assert row["termination_reason"] == BUDGET_EXHAUSTED
    assert row["operational_failure"] is True


# ---------------------------------------------------------------------------
# The absence of a budget is RECORDED, never invented.
# ---------------------------------------------------------------------------
def test_the_inner_row_records_that_it_has_no_budget_rather_than_inventing_one() -> None:
    """PRODUCT_CONTRACT section 9 wants the budget too; this seam cannot state it.

    ``_finalize_timeout`` is handed the message, the detail and the codes -- never
    the ``_Budget`` -- and the ceiling it would need is ``run_one``'s ``timeout=``
    argument, which no ``RunOutcome`` field records. Defaulting it to
    ``LEG_TIMEOUT_SECONDS`` would write "3600s" into rows whose leg ran with a
    different ceiling, so the gap is stated instead.
    """

    row = _inner_row(T105_STRICT_DETAIL)

    assert "budget" not in row, (
        "the INNER row must not publish a 'budget' key: the OUTER row's 'budget' "
        "is a measured record and the two shapes must not collide under one name"
    )
    note = row["budget_unrecorded"]
    assert isinstance(note, str) and note.strip(), "the absence must say something"
    assert note == driver.INNER_BUDGET_UNRECORDED
    assert "not recorded" in note and "_timeout_row" in note


def test_the_outer_row_still_carries_the_measured_budget_it_always_did() -> None:
    """The OUTER path is untouched: defect 2 is refuted and its record is complete."""

    row = runner._timeout_row(slug="PMC12444477__a", mode="strict", paper=PAPER,
                              seconds=3600.4, timeout=3600.0, tail="")
    assert row["budget"]["elapsed_seconds"] == 3600.4
    assert row["budget"]["remaining_seconds"] == -0.4
    assert "budget_unrecorded" not in row


# ---------------------------------------------------------------------------
# One vocabulary across the process boundary.
# ---------------------------------------------------------------------------
def test_both_timeout_paths_publish_the_same_two_key_names() -> None:
    """A consumer must not need two names for one fact.

    ``runner._timeout_row`` and ``deadline.Admission.to_dict`` already say
    ``termination_reason`` / ``operational_failure``. The INNER row joins them
    rather than introducing ``termination_is_operational`` as a third spelling --
    that stays the ATTRIBUTE name, which is what the classifier writes.
    """

    inner = _inner_row(T105_STRICT_DETAIL)
    outer = runner._timeout_row(slug="PMC12444477__a", mode="strict", paper=PAPER,
                                seconds=400.0, timeout=3600.0, tail="")

    shared = {"termination_reason", "operational_failure"}
    assert shared <= set(inner) and shared <= set(outer)
    assert inner["termination_reason"] == outer["termination_reason"] == OPERATION_TIMEOUT
    assert "termination_is_operational" not in inner
    # The attribute keeps the classifier's name, so nothing upstream is renamed.
    outcome = driver.RunOutcome()
    driver._finalize_timeout(outcome, message="m", detail=T105_STRICT_DETAIL)
    assert outcome.termination_is_operational is True


def test_a_stopped_clock_in_the_row_is_never_a_biological_verdict() -> None:
    """D-005's sharpest invariant, now enforceable from the manifest alone."""

    for detail in (T105_STRICT_DETAIL, T105_RESEARCH_DETAIL,
                   f"{BUDGET_SPENT_MARKER} 3600s was already spent", "timed out"):
        row = _inner_row(detail)
        assert row["termination_reason"] in OPERATIONAL_TERMINATION_REASONS
        assert row["termination_reason"] not in NEVER
        assert row["operational_failure"] is True


# ---------------------------------------------------------------------------
# Preservation: only a timeout row moves.
# ---------------------------------------------------------------------------
def test_a_run_that_never_timed_out_writes_exactly_the_row_it_wrote_before() -> None:
    """The three keys are conditional, so every non-timeout row is byte-identical."""

    row = driver.RunOutcome(paper_id="PMC1", mode="strict").to_dict()
    for key in ("termination_reason", "operational_failure", "budget_unrecorded"):
        assert key not in row
    assert set(row) == {
        "paper_id", "mode", "status", "stage", "seconds", "failure_kind",
        "message", "detail", "issue_codes", "counts", "files", "warnings",
    }


def test_the_timeout_row_adds_exactly_three_keys_and_removes_none() -> None:
    """The re-baselined pin, stated as a set difference so it cannot rot.

    This is what makes the golden move auditable: an unintended fourth key fails
    here, and a key silently dropped to stabilise a digest fails here too.
    """

    before = copy.deepcopy(driver.RunOutcome(paper_id="PMC12444477", mode="strict").to_dict())
    row = _inner_row(T105_STRICT_DETAIL)

    assert sorted(set(row) - set(before)) == [
        "budget_unrecorded", "operational_failure", "termination_reason",
    ], f"exactly three keys are added, no more and no fewer (got {sorted(row)})"
    assert set(before) - set(row) == set(), "and none is taken away"
    assert "release_status" not in row and "pwml_artifact" not in row


def test_the_row_survives_json_round_tripping() -> None:
    """A manifest row is JSON Lines: the three new values must serialize."""

    import json

    row = _inner_row(T105_STRICT_DETAIL)
    rehydrated = json.loads(json.dumps(row))
    assert rehydrated["termination_reason"] == OPERATION_TIMEOUT
    assert rehydrated["operational_failure"] is True
    assert rehydrated["budget_unrecorded"] == driver.INNER_BUDGET_UNRECORDED


def test_the_manifest_row_consumers_tolerate_the_widened_row() -> None:
    """``batch/report.py`` reads the row by ``.get``; three more keys change nothing.

    Charter step 3.4: confirm the row consumers are not strict about the key set.
    ``report._to_run`` builds its leg record field by field with ``row.get(...)``
    and ``runner._relocate_files`` rewrites ``files`` alone, so neither reads the
    new keys and neither can be surprised by them.
    """

    from t2pw.batch import report

    row = runner._identify(_inner_row(T105_STRICT_DETAIL), PAPER, "PMC12444477__a", "strict")

    run = report._to_run(row)
    assert run.status == "timeout" and run.failure_kind == "timeout"
    assert run.mode == "strict"
    triage = report.group_papers([row])
    assert len(triage) == 1

    relocated = runner._relocate_files(row, "pmc12444477", "strict")
    assert relocated["termination_reason"] == OPERATION_TIMEOUT
    assert relocated["files"] == []
