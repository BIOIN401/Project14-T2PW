"""New-capability acceptance for ``t2pw.pipeline.deadline``. C-032 / D-005.

Everything here is NEW capability -- the module does not exist on the dispatch
base, so no base failure is claimed for it. The two tests marked *preservation*
assert that nothing already in the tree moved: D-005's 3600 / 120 / 3480 values,
and the argv ``child_command`` has always emitted.

Nothing here spawns a process, touches the network, or writes outside ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.batch import driver, runner  # noqa: E402
from t2pw.pipeline import deadline as dl  # noqa: E402


def _ticking(*marks: float) -> Any:
    """A clock returning each mark in turn, then repeating the last one."""

    seen = list(marks)
    return lambda: seen.pop(0) if len(seen) > 1 else seen[0]


# --------------------------------------------------------------------------
# Preservation
# --------------------------------------------------------------------------
def test_the_d005_values_have_not_moved() -> None:
    """Preservation: 3600 s leg, 120 s grace, 3480 s child, one authority each."""

    assert dl.LEG_TIMEOUT_SECONDS == 3600.0
    assert dl.PARENT_CHILD_GRACE_SECONDS == 120.0
    assert dl.CHILD_DEADLINE_SECONDS == 3480.0
    assert runner.DEFAULT_PAPER_TIMEOUT == dl.LEG_TIMEOUT_SECONDS
    assert runner._CHILD_GRACE == dl.PARENT_CHILD_GRACE_SECONDS
    assert dl.child_deadline_seconds(3600.0) == 3480.0
    assert dl.child_deadline_seconds(100.0) == 60.0  # the runner's long-standing floor


def test_child_command_emits_the_same_argv_it_always_did(tmp_path: Path) -> None:
    """Preservation: routing the subtraction through the module changed no output."""

    cmd = runner.child_command(tmp_path / "cli.py", tmp_path, "PMC1__a", "strict", 3600.0)
    assert cmd[cmd.index("--timeout") + 1] == "3480"
    assert cmd[cmd.index("--slug") + 1] == "PMC1__a"
    assert runner.child_command(tmp_path / "c.py", tmp_path, "s", "strict", 2700.0)[-1] == "2580"
    assert runner.child_command(tmp_path / "c.py", tmp_path, "s", "strict", 100.0)[-1] == "60"


# --------------------------------------------------------------------------
# New acceptance -- the monotonic budget and the reserve
# --------------------------------------------------------------------------
def test_one_monotonic_deadline_records_elapsed_and_remaining() -> None:
    """New acceptance: one clock per leg, readable at any stage, never negative."""

    leg = dl.LegDeadline(3600.0, reserve_seconds=120.0, clock=_ticking(1000.0, 1600.0))
    assert leg.elapsed == 600.0 and leg.remaining == 3000.0
    assert leg.spendable == 2880.0 and leg.expired is False
    snap = leg.snapshot()
    assert snap["elapsed_seconds"] == 600.0 and snap["remaining_seconds"] == 3000.0
    assert snap["finalization_reserve_seconds"] == 120.0
    # A clock stepping BACKWARDS cannot hand the leg budget it never had.
    back = dl.LegDeadline(60.0, clock=_ticking(9000.0, 1.0))
    assert back.elapsed == 0.0 and back.remaining == 60.0
    over = dl.LegDeadline(60.0, clock=_ticking(0.0, 90.0))
    assert over.remaining == -30.0 and over.expired is True


def test_no_call_starts_unless_its_maximum_plus_the_reserve_fits() -> None:
    """New acceptance: D-005's admission precondition, at the exact boundary."""

    leg = dl.LegDeadline(1000.0, reserve_seconds=200.0, clock=_ticking(0.0, 100.0))
    assert leg.remaining == 900.0
    assert leg.can_start(700.0) is True  # 700 + 200 == 900 exactly
    assert leg.can_start(700.01) is False
    refused = leg.admit(700.01, label="stage1 extraction")
    assert refused.allowed is False and refused.reason == dl.BUDGET_EXHAUSTED
    assert refused.required_seconds == pytest.approx(900.01)
    assert refused.to_dict()["operational_failure"] is True
    with pytest.raises(dl.BudgetExhausted) as caught:
        refused.raise_if_refused()
    assert caught.value.terminal_reason == dl.BUDGET_EXHAUSTED
    leg.admit(700.0).raise_if_refused()  # an allowed call raises nothing
    # Only the reserve left: work stops, finalization does not.
    tight = dl.LegDeadline(1000.0, reserve_seconds=200.0, clock=_ticking(0.0, 800.0))
    assert tight.expired is False and tight.reserve_breached is True


def test_an_unpriceable_llm_call_fails_closed(monkeypatch: Any) -> None:
    """New acceptance: an unknown call price is infinite, never zero."""

    priced = dl.price_llm_call(timeout=10.0, max_retries=1, base_sleep=0.0,
                               max_sleep=0.0, spacing=0.0)
    assert 10.0 <= priced < float("inf")  # a priceable call is a finite bound
    monkeypatch.setitem(sys.modules, "t2pw.llm.client", None)
    assert dl.price_llm_call() == float("inf")
    _, admission = dl.LegDeadline(3600.0, clock=_ticking(0.0)).before_long_call(label="stage1")
    assert admission.allowed is False and admission.reason == dl.BUDGET_EXHAUSTED


# --------------------------------------------------------------------------
# New acceptance -- checkpoints
# --------------------------------------------------------------------------
def test_the_checkpoint_is_written_before_the_call_is_even_priced() -> None:
    """New acceptance: D-005's order -- checkpoint, then decide, then call."""

    order: List[str] = []
    leg = dl.LegDeadline(1000.0, reserve_seconds=200.0, clock=_ticking(0.0, 100.0))
    record, admission = leg.before_long_call(
        label="rag retrieval", call_seconds=5000.0, stage="stage1",
        state={"claims": 3}, persist=lambda cp: order.append(f"persist:{cp.label}"),
    )
    order.append("decided")
    assert order == ["persist:rag retrieval", "decided"]
    assert admission.allowed is False  # the refused call still left a checkpoint
    assert record.sequence == 1 and record.stage == "stage1"
    assert record.to_dict()["state"] == {"claims": 3}
    assert record.to_dict()["remaining_seconds"] == 900.0
    assert leg.snapshot()["checkpoints"] == 1


def test_a_failing_checkpoint_writer_never_kills_the_leg() -> None:
    """New acceptance: a checkpoint makes a stopped leg reportable; it cannot end it."""

    def explode(_cp: Any) -> None:
        raise OSError("the disk is full")

    assert dl.LegDeadline(1000.0, clock=_ticking(0.0)).checkpoint(
        "stage1", persist=explode).label == "stage1"


# --------------------------------------------------------------------------
# New acceptance -- overrides, vocabulary, wiring
# --------------------------------------------------------------------------
def test_a_per_leg_override_must_be_explicit_and_is_recorded() -> None:
    """New acceptance: no silent extension of a difficult benchmark leg."""

    assert dl.resolve_leg_timeout().to_dict() == {
        "leg_timeout_seconds": 3600.0,
        "leg_timeout_default_seconds": 3600.0,
        "leg_timeout_overridden": False,
    }
    for bad in ({}, {"reason": "it is a hard one"}, {"source": "me"}):
        with pytest.raises(ValueError):
            dl.resolve_leg_timeout(5400.0, **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dl.resolve_leg_timeout(0.0, reason="r", source="s")
    recorded = dl.resolve_leg_timeout(5400.0, reason="PMC12444477 rerun", source="D-005")
    assert recorded.overridden is True
    assert recorded.to_dict()["leg_timeout_override_reason"] == "PMC12444477 rerun"
    assert recorded.to_dict()["leg_timeout_override_source"] == "D-005"
    # The override reaches the manifest row the parent writes for a killed child.
    row = runner._timeout_row(slug="s", mode="strict", paper={}, seconds=5400.5,
                              timeout=5400.0, tail="", leg_timeout=recorded)
    assert row["budget"]["leg_timeout_override_reason"] == "PMC12444477 rerun"
    assert row["termination_reason"] == dl.BUDGET_EXHAUSTED
    # ``--timeout 2700`` is recorded against the D-005 default, never as "not an override".
    bare = runner._timeout_row(slug="s", mode="strict", paper={}, seconds=2700.5,
                               timeout=2700.0, tail="")["budget"]
    assert bare["leg_timeout_seconds"] == 2700.0 and bare["leg_timeout_default_seconds"] == 3600.0
    assert bare["leg_timeout_overridden"] is True
    assert dl.LegDeadline(2700.0, clock=_ticking(0.0)).snapshot()["leg_timeout_overridden"] is True


def test_a_timeout_may_name_only_an_operational_reason() -> None:
    """A timeout cannot borrow a semantic or retrieval label, and refusing one must
    not abandon the terminal row."""

    assert dl.require_operational_reason("operation_timeout") == dl.OPERATION_TIMEOUT
    for bad in ("scientifically_unrecoverable", "retrieval_exhausted", "no_new_claims"):
        with pytest.raises(ValueError):
            dl.require_operational_reason(bad)
        with pytest.raises(ValueError):
            runner._timeout_row(slug="s", mode="strict", paper={}, seconds=3600.4,
                                timeout=3600.0, tail="", reason=bad)
        outcome = driver.RunOutcome(paper_id="PMC1", mode="strict")
        with pytest.raises(ValueError):
            driver._finalize_timeout(outcome, message="m", detail="d", reason=bad)
        # The leg really did time out: the terminal row survives the refusal.
        assert outcome.status == "timeout" and outcome.failure_kind == "timeout"
        assert outcome.message == "m" and outcome.detail == "d"
    row = runner._timeout_row(slug="s", mode="strict", paper={}, seconds=1.0,
                              timeout=3600.0, tail="", reason=dl.BUDGET_EXHAUSTED)
    assert row["termination_reason"] == dl.BUDGET_EXHAUSTED and row["operational_failure"] is True


def test_the_seven_reasons_are_a_closed_never_conflated_vocabulary() -> None:
    """Exactly seven, and STILL only two of them are operational.

    BASELINE MOVE, C-042a / D-024, merge rule 4. This pinned 6 and the six D-005
    strings; it now pins 7 and those same six PLUS ``attempt_cap_reached``. Exact
    delta: +1 member, no member removed, no member renamed. The
    ``OPERATIONAL_TERMINATION_REASONS`` assertion below is deliberately UNCHANGED --
    D-024 does not widen the strict-success denominator, and that line is what
    proves the two moves stayed independent.
    """

    assert len(dl.TERMINATION_REASONS) == len(set(dl.TERMINATION_REASONS)) == 7
    assert set(dl.TERMINATION_REASONS) == {
        "retrieval_exhausted", "no_new_claims", "budget_exhausted",
        "operation_timeout", "identical_empty_response", "scientifically_unrecoverable",
        "attempt_cap_reached",
    }
    assert dl.OPERATIONAL_TERMINATION_REASONS == {"budget_exhausted", "operation_timeout"}
    for reason in dl.TERMINATION_REASONS:
        assert dl.require_reason(reason) == reason
        assert dl.is_operational(reason) is (reason in dl.OPERATIONAL_TERMINATION_REASONS)
    for invented in ("semantic_failure", "timeout", "", None):
        with pytest.raises(ValueError):
            dl.require_reason(invented)


def test_retrieval_exhausted_needs_a_ladder_that_actually_completed() -> None:
    """New acceptance: a short ladder is a budget outcome, never 'nothing is there'."""

    assert dl.claim_retrieval_exhausted(
        rungs_configured=3, rungs_completed=3, ladder_completed=True
    ) == dl.RETRIEVAL_EXHAUSTED
    for kwargs in ({"rungs_configured": 3, "rungs_completed": 1, "ladder_completed": True},
                   {"rungs_configured": 3, "rungs_completed": 3, "ladder_completed": False},
                   {"rungs_configured": 0, "rungs_completed": 0, "ladder_completed": True}):
        with pytest.raises(ValueError):
            dl.claim_retrieval_exhausted(**kwargs)  # type: ignore[arg-type]


def test_launch_child_never_waits_past_the_leg_deadline(monkeypatch: Any) -> None:
    """New acceptance: the parent's wait is bounded by the leg's own budget."""

    waited: List[float] = []

    class FakePopen:
        pid, returncode = 7, 0

        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def communicate(self, timeout: Any = None) -> Any:
            waited.append(float(timeout))
            return "", ""

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    runner.launch_child(["x"], 3600.0, deadline=dl.LegDeadline(3600.0, clock=_ticking(0.0, 3400.0)))
    runner.launch_child(["x"], 3600.0)
    assert waited == [200.0, 3600.0]


def test_the_extraction_escalation_ladder_is_not_implemented_here() -> None:
    """New acceptance: C-032 builds the budget, C-042 spends it. D-005's order.

    BASELINE MOVE, C-042a / D-024, merge rule 4. Exact delta: one name,
    ``ATTEMPT_CAP_REACHED``, is exempted by an explicit allowlist. It is a
    termination-reason STRING CONSTANT in D-005's closed vocabulary, not ladder
    machinery; the ladder still lives entirely in ``extraction_ladder`` and this
    module still counts nothing, retries nothing and holds no ceiling. The guard
    keeps its full force against everything else -- ``MAX_TOTAL_ATTEMPTS``,
    ``attempts_used``, ``attempts_remaining``, any ``*retry*`` -- because the
    exemption is one exact name and not a relaxed pattern. Asserted below rather
    than merely subtracted, so the exemption cannot silently cover a second name.
    """

    names = [name for name in dir(dl) if not name.startswith("_")]
    matched = [n for n in names if "retry" in n.lower() or "attempt" in n.lower()]
    assert matched == ["ATTEMPT_CAP_REACHED"]
    assert dl.ATTEMPT_CAP_REACHED == "attempt_cap_reached"   # a string, not a counter
    assert not any(callable(getattr(dl, n)) for n in matched)
    assert not hasattr(dl, "MAX_TOTAL_ATTEMPTS")
