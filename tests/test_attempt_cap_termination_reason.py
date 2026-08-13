"""C-042a / D-024 — a spent attempt cap ends a leg, so it must SAY it ended the leg.

C-042 built the ladder and refused the rung past § 9's ceiling with a skip cause and
``termination_reason=""``, because none of D-005's six reasons truthfully described the
stop. D-024 adds the seventh, ``attempt_cap_reached``, and ranks it: below a real
deadline, a timeout, a measured budget exhaustion and any explicit refusal, and above
the two reasons that CLAIM retrieval finished — which a capped leg's did not.

The tests import the MODULES, not the new constants, on purpose: this file must be
loadable at the base SHA so ``test_g9_...`` below can fail there *behaviourally*
(observing ``""``) rather than failing on an absent symbol, which would prove nothing.
"""

from __future__ import annotations

import json

import pytest

from t2pw.pipeline import deadline as dl
from t2pw.pipeline import extraction_ladder as el
from t2pw.pipeline.extraction_diagnostics import (
    BOUNDARY_STAGE1_LADDER_TERMINATION, ExtractionDiagnostics,
)
from t2pw.rag import loop_policy as lp

#: The one literal both vocabularies use, spelled out rather than imported so a
#: rename of either constant cannot quietly follow the test along with it.
CAP = "attempt_cap_reached"

#: A healthy mid-loop state: budget left, the round completed, the graph grew, the
#: ladder did NOT complete. Mirrors ``test_rag_loop_policy.BASE``.
BASE = dict(rounds_completed=1, max_rounds=3, now=100.0, deadline=1000.0,
            round_retrieval_completed=True, new_admissible_claims=2, graph_delta=2)


def state(**overrides) -> lp.LoopState:
    return lp.LoopState(**{**BASE, **overrides})


def _ticking(*readings: float):
    """A monotonic clock returning ``readings`` in order, then repeating the last."""
    seq = list(readings)
    return lambda: seq.pop(0) if len(seq) > 1 else seq[0]


def spent_ladder(*, cap: int = 1, deadline=None) -> el.ExtractionLadder:
    """A ladder whose ceiling is exactly consumed and nothing else is wrong."""
    ladder = el.ExtractionLadder(
        stage="extraction", max_total_attempts=cap, deadline=deadline)
    for i in range(cap):
        ladder.record(rung=el.RUNG_NORMAL, model="model-a", request_hash=f"req-{i}",
                      response_hash=f"resp-{i}")
    return ladder


# --------------------------------------------------------------------- A9 / G9
def test_g9_a_capped_ladder_names_its_stop_where_it_used_to_report_nothing() -> None:
    """G9 BEHAVIOURAL BASE PROOF — fails at 472293c, passes at the tip.

    At base this exact call returns ``skip_cause="attempt_cap_reached"`` and
    ``termination_reason=""``: the ladder knew the leg was over and had no vocabulary
    to say so. Written against the literal, never against ``dl.ATTEMPT_CAP_REACHED``,
    so at base it fails on the OBSERVED VALUE and not on an import.
    """
    decision = spent_ladder(cap=1).admit(
        rung=el.RUNG_EMPTY_REPAIR, model="model-a", request_hash="req-next")

    assert decision.allowed is False
    assert decision.skip_cause == CAP
    assert decision.termination_reason != ""      # the base behaviour being corrected
    assert decision.termination_reason == CAP


# ------------------------------------------------------------------- A1 success
def test_a1_success_before_the_cap_never_emits_the_new_reason() -> None:
    """Attempts remain and the rung is admitted: no skip cause, no termination reason."""
    ladder = el.ExtractionLadder(stage="extraction", max_total_attempts=3, deadline=None)
    ladder.record(rung=el.RUNG_NORMAL, model="model-a", request_hash="req-0")

    decision = ladder.admit(rung=el.RUNG_EMPTY_REPAIR, model="model-a",
                            request_hash="req-1")

    assert (decision.allowed, decision.skip_cause, decision.termination_reason) == (
        True, "", "")
    assert not any(d.termination_reason == CAP for d in ladder.decisions)
    assert "termination_reason" not in decision.to_dict()
    # And the loop policy: a state that is going round again names no reason at all.
    assert decide_of(state(attempt_cap_reached=False)) == (True, None, ())


def decide_of(st: lp.LoopState):
    d = lp.decide(st)
    return d.should_continue, d.reason, d.also_true


# ------------------------------------------- A2 refusal · A3 timeout · A4 budget
@pytest.mark.parametrize("overrides,winner", [
    # A2 — an explicit refusal outranks the cap even though the cap is also spent.
    (dict(evidence_sources_exhausted=True, defensible_core=False),
     "scientifically_unrecoverable"),
    # A2 — identical-empty is a stronger, more specific description of the same stop.
    (dict(identical_empty_responses=2), "identical_empty_response"),
    # A3 — a real operation timeout is retained even with the cap spent.
    (dict(operation_timed_out=True), "operation_timeout"),
    # A4 — a MEASURED budget exhaustion is retained even with the cap spent.
    (dict(now=1000.0), "budget_exhausted"),
])
def test_a2_a3_a4_a_stronger_stop_outranks_a_spent_cap(overrides, winner) -> None:
    """Every clause-2/3/4 signal beats clause 5, and the cap is still REPORTED."""
    should_continue, reason, also_true = decide_of(state(attempt_cap_reached=True,
                                                         **overrides))

    assert (should_continue, reason) == (False, winner)
    assert CAP in also_true          # never conflated, never silently dropped
    assert reason != CAP


def test_a4_the_ladders_budget_refusal_is_not_shadowed_by_the_cap() -> None:
    """``extraction_ladder``'s ``reason=BUDGET_EXHAUSTED`` branch still fires.

    The cap check runs FIRST in ``admit``, so this proves the new reason cannot reach
    a rung that budget refused: with attempts remaining, the cap branch is not taken.
    """
    leg = dl.LegDeadline(1000.0, reserve_seconds=200.0, clock=_ticking(0.0, 900.0))
    ladder = el.ExtractionLadder(stage="extraction", max_total_attempts=3, deadline=leg,
                                 price_fn=lambda: 500.0)

    decision = ladder.admit(rung=el.RUNG_DIFFERENT_STRATEGY, model="model-b",
                            request_hash="req-3", budget_conditional=True)

    assert decision.allowed is False
    assert decision.skip_cause == el.SKIP_BUDGET_REFUSED
    assert decision.termination_reason == dl.BUDGET_EXHAUSTED
    assert decision.termination_reason != CAP


def test_a2_an_earlier_identical_prompt_refusal_survives_a_later_cap_refusal() -> None:
    """A cap refusal appends; it never rewrites the refusal already on the record."""
    ladder = el.ExtractionLadder(stage="extraction", max_total_attempts=2, deadline=None)
    ladder.record(rung=el.RUNG_NORMAL, model="model-a", request_hash="req-0",
                  response_hash="empty", structurally_empty=True)
    inert = ladder.admit(rung=el.RUNG_EMPTY_REPAIR, model="model-a",
                         request_hash="req-0")
    ladder.record(rung=el.RUNG_EMPTY_REPAIR, model="model-a", request_hash="req-1")
    capped = ladder.admit(rung=el.RUNG_DIFFERENT_STRATEGY, model="model-b",
                          request_hash="req-2")

    assert inert.termination_reason == dl.IDENTICAL_EMPTY_RESPONSE
    assert capped.termination_reason == CAP
    assert ladder.inert_extraction_observed() is True   # still the stronger fact
    assert [d.termination_reason for d in ladder.decisions] == [
        dl.IDENTICAL_EMPTY_RESPONSE, CAP]


# ---------------------------------------------------------------- A5 the cap alone
def test_a5_the_cap_alone_yields_the_new_reason_and_keeps_its_skip_cause() -> None:
    """Two fields, two vocabularies, one shared literal — both populated, neither lost."""
    ladder = spent_ladder(cap=3)
    decision = ladder.admit(rung=el.RUNG_DIFFERENT_STRATEGY, model="model-b",
                            request_hash="req-next", budget_conditional=True)
    record = decision.to_dict()

    assert record["skip_cause"] == CAP and record["termination_reason"] == CAP
    assert record["allowed"] is False
    assert ladder.skipped[-1]["termination_reason"] == CAP
    # The loop policy agrees when the cap is the ONLY thing that stopped the loop.
    assert decide_of(state(attempt_cap_reached=True)) == (False, CAP, ())


def test_a5_the_cap_does_not_claim_a_ladder_that_actually_completed() -> None:
    """D-005: ``retrieval_exhausted`` needs a COMPLETED ladder — and then it wins.

    The guard, not the ranking, keeps these two mutually exclusive: a ladder that ran
    to completion was ended by its completion, not by the ceiling.
    """
    assert decide_of(state(attempt_cap_reached=True, ladder_completed=True,
                           new_admissible_claims=0, graph_delta=0)) == (
        False, "retrieval_exhausted", ())


def test_a5_a_capped_leg_is_never_reported_as_no_new_claims() -> None:
    """``no_new_claims`` says retrieval finished and found nothing; a capped leg's did not."""
    should_continue, reason, also_true = decide_of(
        state(attempt_cap_reached=True, new_admissible_claims=0, graph_delta=0))

    assert (should_continue, reason) == (False, CAP)
    assert also_true == ("no_new_claims",)


# ------------------------------------------------------- A6 serialization / report
def test_a6_the_value_round_trips_byte_stable_through_report_and_require_reason() -> None:
    """``to_dict`` -> JSON -> ``require_reason`` -> boundary record, unchanged."""
    ladder = spent_ladder(cap=2)
    decision = ladder.admit(rung=el.RUNG_DIFFERENT_STRATEGY, model="model-b",
                            request_hash="req-next")

    # extraction_ladder.RungDecision.to_dict
    rehydrated = json.loads(json.dumps(decision.to_dict()))
    assert dl.require_reason(rehydrated["termination_reason"]) == CAP
    assert el.require_skip_cause(rehydrated["skip_cause"]) == CAP

    # extraction_ladder.preservation_record — the § 9 preservation set, which puts
    # the value through require_reason on the way in.
    preserved = ladder.preservation_record(
        last_completed_stage="preprocess", payload={},
        termination_reason=decision.termination_reason)
    assert preserved["termination_reason"] == CAP
    assert preserved["skipped_next_step"]["termination_reason"] == CAP
    assert json.loads(json.dumps(preserved))["termination_reason"] == CAP

    # extraction_diagnostics.record_boundary — the reporting leg.
    row = ExtractionDiagnostics(run_id="c042a").record_boundary(
        stage="extraction", boundary=BOUNDARY_STAGE1_LADDER_TERMINATION,
        attempts=ladder.attempts_used, terminal_reason=decision.termination_reason,
        extraction_ladder=preserved)
    assert row["terminal_reason"] == CAP
    assert json.loads(json.dumps(row))["terminal_reason"] == CAP


# ------------------------------------------------------------------ A7 precedence
def test_a7_two_three_and_four_signals_at_once_resolve_to_the_ruled_winner() -> None:
    """The clause that proves § 2's ordering is IMPLEMENTED, not merely asserted."""
    two = decide_of(state(attempt_cap_reached=True, identical_empty_responses=2))
    three = decide_of(state(attempt_cap_reached=True, identical_empty_responses=2,
                            operation_timed_out=True))
    four = decide_of(state(attempt_cap_reached=True, identical_empty_responses=2,
                           operation_timed_out=True, now=1000.0))
    five = decide_of(state(attempt_cap_reached=True, identical_empty_responses=2,
                           operation_timed_out=True, now=1000.0,
                           evidence_sources_exhausted=True, defensible_core=False))

    assert two == (False, "identical_empty_response", (CAP,))
    assert three == (False, "operation_timeout", ("identical_empty_response", CAP))
    assert four == (False, "budget_exhausted",
                    ("operation_timeout", "identical_empty_response", CAP))
    assert five == (False, "budget_exhausted",
                    ("operation_timeout", "identical_empty_response",
                     "scientifically_unrecoverable", CAP))
    # In every one of them the cap is last: it never outranks a stronger stop, and
    # it is never dropped either.
    for _, _, also in (two, three, four, five):
        assert also[-1] == CAP


def test_a7_the_ranking_is_exactly_the_ruled_order_and_decide_cannot_key_error() -> None:
    """Every precedence member has a condition key, so ``decide`` is total."""
    assert lp.TERMINATION_PRECEDENCE.index(CAP) == 4
    assert lp.TERMINATION_PRECEDENCE == (
        "budget_exhausted", "operation_timeout", "identical_empty_response",
        "scientifically_unrecoverable", CAP, "retrieval_exhausted", "no_new_claims")
    held = lp._conditions(state(attempt_cap_reached=True))
    assert set(held) == set(lp.TERMINATION_PRECEDENCE)      # no KeyError is possible
    assert lp.TERMINATION_REASONS == frozenset(lp.TERMINATION_PRECEDENCE)


# ------------------------------------------- A8 the two vocabularies stay separate
def test_a8_skip_causes_and_termination_reasons_share_exactly_one_literal() -> None:
    """The sharpest hazard in this card, proven member by member.

    ``attempt_cap_reached`` is the ONE deliberately shared literal: both vocabularies
    name the same event. Every other member of either set is refused by the other's
    validator, so the two remain independently closed.
    """
    shared = set(el.SKIP_CAUSES) & set(dl.TERMINATION_REASONS)
    assert shared == {CAP}                       # intentional, and the only one

    for cause in el.SKIP_CAUSES:
        assert el.require_skip_cause(cause) == cause
        if cause == CAP:
            assert dl.require_reason(cause) == cause
            continue
        with pytest.raises(ValueError):
            dl.require_reason(cause)

    for reason in dl.TERMINATION_REASONS:
        assert dl.require_reason(reason) == reason
        if reason == CAP:
            assert el.require_skip_cause(reason) == reason
            continue
        with pytest.raises(ValueError):
            el.require_skip_cause(reason)

    assert len(el.SKIP_CAUSES) == len(set(el.SKIP_CAUSES)) == 6
    assert len(dl.TERMINATION_REASONS) == len(set(dl.TERMINATION_REASONS)) == 7


def test_a8_the_operational_denominator_is_not_widened() -> None:
    """§ 4. A safety ceiling is not an operational failure, and D-024 does not say it is."""
    assert dl.OPERATIONAL_TERMINATION_REASONS == frozenset(
        {"budget_exhausted", "operation_timeout"})
    assert dl.ATTEMPT_CAP_REACHED not in dl.OPERATIONAL_TERMINATION_REASONS
    assert dl.is_operational(CAP) is False
    with pytest.raises(ValueError):
        dl.require_operational_reason(CAP)


def test_a8_the_llm_clients_third_vocabulary_is_untouched() -> None:
    """``llm.client``'s TERMINAL_* strings are a third vocabulary. Read, never merged."""
    from t2pw.llm import client as llm_client

    assert llm_client.TERMINAL_CONTENT_FILTER == "content_filter"
    assert llm_client.TERMINAL_EMPTY_AFTER_RETRIES == "empty_after_retries"
    assert llm_client.TERMINAL_OPERATION_TIMEOUT == "operation_timeout"
    # Only the timeout string is shared with D-005, exactly as it was before D-024.
    assert CAP not in {llm_client.TERMINAL_CONTENT_FILTER,
                       llm_client.TERMINAL_EMPTY_AFTER_RETRIES,
                       llm_client.TERMINAL_OPERATION_TIMEOUT}


# ------------------------------------------------------- A10 nothing admitted moves
def test_a10_naming_the_stop_changes_no_admission_and_raises_no_new_exception() -> None:
    """This card renames a silence. It admits nothing, refuses nothing new, raises nothing.

    ``raise_if_refused`` must still be a no-op for a cap refusal: turning the cap into
    an exception would report a stopped clock where there was none, and would end runs
    that today keep going with the payload they already have.
    """
    ladder = spent_ladder(cap=1)
    decision = ladder.admit(rung=el.RUNG_EMPTY_REPAIR, model="model-a",
                            request_hash="req-next")

    assert decision.allowed is False              # unchanged: it was already refused
    assert decision.admission is None and decision.checkpoint is None
    assert ladder.checkpoints == []               # no new checkpoint, no new I/O
    assert decision.raise_if_refused() is None    # still not an exception path
    assert ladder.attempts_used == 1 and ladder.attempts_remaining == 0
    assert ladder.repair_budget(2) == 0           # the ceiling arithmetic is untouched


# ===========================================================================
# BOUNDARY EXTENSION — the LEG-level termination reason in
# ``pipeline._run_json_stage``. Same discipline: modules imported, not the new
# constants, so this whole file still loads at 472293c.
# ===========================================================================
import inspect  # noqa: E402

import t2pw.pipeline.extraction_diagnostics as diag_mod  # noqa: E402
import t2pw.pipeline.pipeline as pipeline_mod  # noqa: E402
from t2pw.llm.client import (  # noqa: E402
    CompletionDiagnostics, CompletionResult, LLMOperationTimeout,
)
from t2pw.pipeline.pipeline import PipelineFailure, _run_json_stage  # noqa: E402

#: Two structurally empty payloads with DIFFERENT bytes, so they are empty
#: without being the identical-empty signature. That separation is what lets a
#: run reach the cap branch with nothing stronger true.
EMPTY_A = '{"entities": {}, "processes": []}'
EMPTY_B = '{"entities": {"proteins": []}, "processes": []}'


class Chat:
    """A scripted ``chat_detailed``. Offline; never reaches a provider."""

    def __init__(self, *texts: str, raises=None) -> None:
        self._texts, self._raises, self.prompts = list(texts), raises, []

    def __call__(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        if self._raises is not None:
            raise self._raises
        text = self._texts[min(len(self.prompts) - 1, len(self._texts) - 1)]
        return CompletionResult(text, CompletionDiagnostics(
            model="test-model", stage="extraction", attempts=1))

    @property
    def calls(self) -> int:
        return len(self.prompts)


@pytest.fixture
def recorder():
    rec = diag_mod.activate(run_id="c042a-leg")
    yield rec
    diag_mod.deactivate()


def run_stage(chat, *, monkeypatch, ladder, max_attempts: int = 3):
    monkeypatch.setattr(pipeline_mod, "chat_detailed", chat)
    return _run_json_stage(
        stage_name="extraction", system_prompt="SYSTEM",
        build_user_prompt=lambda prev, err: pipeline_mod._build_extraction_prompt(
            "A paper about a pathway.", prev, err),
        max_attempts=max_attempts, temperature=0.0, max_tokens=100,
        repair_json=False, retry_on_empty_payload=True, ladder=ladder)


def terminations(rec):
    return [r for r in rec.boundaries
            if r.get("boundary") == "stage1_extraction_ladder_termination"]


def seed_cap_refusal(ladder) -> None:
    """Put a REAL cap refusal on the ladder without spending its ceiling.

    Requirement 1 asks for the cap holding *simultaneously* with a budget refusal
    and with an inert ladder. Inside one ``admit`` those cannot co-occur naturally
    -- the ceiling is checked before the budget gate, so a ladder that refused on
    budget still had attempts left -- which is itself a structural guarantee, and
    it is asserted in ``test_l6``. Seeding the row is how the ORDERING of the
    leg-level ``elif`` chain gets tested directly, which is what the branch does.
    ``skipped`` is public state and a caller-supplied ladder is a supported input.
    """
    ladder.skipped.append({
        "rung": el.RUNG_DIFFERENT_STRATEGY, "attempt": 99, "allowed": False,
        "skip_cause": el.SKIP_ATTEMPT_CAP, "termination_reason": CAP,
    })


# ------------------------------------------------- L1 / A9 at the leg level
def test_l1_g9_a_capped_leg_reports_the_reason_where_it_used_to_report_nothing(
        monkeypatch, recorder) -> None:
    """G9 BEHAVIOURAL BASE PROOF, LEG LEVEL — fails at 472293c, passes at tip.

    At base ``PipelineFailure.terminal_reason`` and the boundary record are ``""``
    for a leg the ceiling ended: the block had no branch for it. Asserted against
    the literal, so at base this fails on the OBSERVED VALUE.
    """
    ladder = el.ExtractionLadder(stage="extraction", max_total_attempts=2,
                                 deadline=None)
    with pytest.raises(PipelineFailure) as excinfo:
        run_stage(Chat(EMPTY_A, EMPTY_B), monkeypatch=monkeypatch, ladder=ladder)

    assert any(r.get("skip_cause") == el.SKIP_ATTEMPT_CAP for r in ladder.skipped)
    assert excinfo.value.terminal_reason != ""      # the base behaviour corrected
    assert excinfo.value.terminal_reason == CAP
    # ...and it reached the artifact, not only the exception.
    row = terminations(recorder)[-1]
    assert row["terminal_reason"] == CAP
    assert row["extraction_ladder"]["termination_reason"] == CAP
    assert excinfo.value.ladder["termination_reason"] == CAP


# -------------------------------------------------------- L2/L3 precedence
def test_l2_a_refused_budget_still_wins_when_the_cap_also_holds(
        monkeypatch, recorder) -> None:
    """Clause 4 over clause 5, at the leg level. ``budget_exhausted`` is retained."""
    # 700 s of leg, a 120 s reserve, calls priced at 300 s and each taking 500 s.
    # Draw 1 needs 420 of 700 and starts; draw 2 needs 420 of 200 and is refused.
    # Deterministic, and it does not depend on the machine's speed.
    clock = {"t": 0.0}
    deadline = dl.LegDeadline(700.0, clock=lambda: clock["t"])
    ladder = el.ExtractionLadder(stage="extraction", deadline=deadline,
                                 price_fn=lambda **_kw: 300.0)
    seed_cap_refusal(ladder)

    class Spending(Chat):
        def __call__(self, messages, **kwargs):
            result = super().__call__(messages, **kwargs)
            clock["t"] += 500.0
            return result

    with pytest.raises(PipelineFailure) as excinfo:
        run_stage(Spending(EMPTY_A, EMPTY_B), monkeypatch=monkeypatch,
                  ladder=ladder, max_attempts=3)

    assert any(r.get("skip_cause") == el.SKIP_BUDGET_REFUSED for r in ladder.skipped)
    assert any(r.get("skip_cause") == el.SKIP_ATTEMPT_CAP for r in ladder.skipped)
    assert excinfo.value.terminal_reason == dl.BUDGET_EXHAUSTED
    assert excinfo.value.terminal_reason != CAP
    assert terminations(recorder)[-1]["terminal_reason"] == dl.BUDGET_EXHAUSTED


def test_l3_an_inert_ladder_still_wins_when_the_cap_also_holds(
        monkeypatch, recorder) -> None:
    """Clause 2 over clause 5. ``identical_empty_response`` is retained."""
    ladder = el.ExtractionLadder(stage="extraction", max_total_attempts=3,
                                 deadline=None)
    seed_cap_refusal(ladder)

    with pytest.raises(PipelineFailure) as excinfo:
        run_stage(Chat(EMPTY_A, EMPTY_A), monkeypatch=monkeypatch, ladder=ladder,
                  max_attempts=2)

    assert ladder.identical_empty_hash() or ladder.inert_extraction_observed()
    assert any(r.get("skip_cause") == el.SKIP_ATTEMPT_CAP for r in ladder.skipped)
    assert excinfo.value.terminal_reason == dl.IDENTICAL_EMPTY_RESPONSE
    assert excinfo.value.terminal_reason != CAP


# ------------------------------------------------------------- L4 timeout
def test_l4_an_operation_timeout_is_never_displaced_by_the_cap(
        monkeypatch, recorder) -> None:
    """The cap cannot shadow ``operation_timeout`` because it never meets it.

    ``_issue`` records ``operation_timeout`` and RE-RAISES, so a timed-out leg
    leaves ``_run_json_stage`` by that exception and the termination block -- the
    only place the cap branch lives -- is never executed on that path. Two disjoint
    exits, proven by the exception type that actually escapes.
    """
    ladder = el.ExtractionLadder(stage="extraction", max_total_attempts=3,
                                 deadline=None)
    seed_cap_refusal(ladder)                      # the cap holds, and still loses

    with pytest.raises(LLMOperationTimeout):
        run_stage(Chat(EMPTY_A, raises=LLMOperationTimeout("overran")),
                  monkeypatch=monkeypatch, ladder=ladder)

    rows = terminations(recorder)
    assert rows[-1]["terminal_reason"] == dl.OPERATION_TIMEOUT
    assert not any(r.get("terminal_reason") == CAP for r in rows)


# ----------------------------------------------------- L5 success unreachable
def test_l5_a_successful_leg_can_never_acquire_this_reason(
        monkeypatch, recorder) -> None:
    """Clause 1. The block is unreachable on every success path, structurally.

    Both successful exits (``return parsed, attempts`` and ``return parsed3,
    attempts``) precede the termination block in source order, so a leg that
    produced a payload has already returned before ``reason`` is ever computed.
    Pinned both ways: by source order, and by a run that succeeds.
    """
    source = inspect.getsource(_run_json_stage)
    block = source.index('    reason = ""')
    returns = [i for i in range(len(source))
               if source.startswith("            return parsed", i)]
    assert returns and all(i < block for i in returns)

    ladder = el.ExtractionLadder(stage="extraction", max_total_attempts=3,
                                 deadline=None)
    parsed, _attempts = run_stage(
        Chat('{"entities": {"proteins": [{"name": "P"}]}, "processes": []}'),
        monkeypatch=monkeypatch, ladder=ladder)

    assert parsed["entities"]["proteins"][0]["name"] == "P"
    assert terminations(recorder) == []           # no termination record at all
    assert not any(r.get("skip_cause") == el.SKIP_ATTEMPT_CAP
                   for r in ladder.skipped)


# --------------------------------------- L6 the structural guarantee, recorded
def test_l6_the_ceiling_is_checked_before_the_budget_gate_so_they_cannot_race() -> None:
    """Why L2 seeds rather than co-produces: ``admit`` orders the two checks.

    A ladder refused on budget necessarily still had attempts remaining, so within
    one ``admit`` a budget refusal and a cap refusal cannot arise from the same
    call. The leg-level ``elif`` chain is ordered anyway -- defence in depth, and
    the ordering is what L2 and L3 test. If this assertion ever fails the ordering
    guarantee has moved and L2's seeding is no longer merely a convenience.
    """
    source = inspect.getsource(el.ExtractionLadder.admit)
    assert source.index("attempts_used >= self.max_total_attempts") < source.index(
        "if self.deadline is None")

    ladder = spent_ladder(cap=1)
    decision = ladder.admit(rung=el.RUNG_NORMAL, model="m", request_hash="h")
    assert decision.skip_cause == el.SKIP_ATTEMPT_CAP
    assert decision.termination_reason == CAP
    assert decision.admission is None            # the budget gate never ran
