"""C-064 / F-070 — a configured round ceiling is not an exhausted budget.

SELF-CONTAINED and loadable at base ``ac77668`` on purpose. It imports the MODULES,
never the new constant, and spells every reason literal out, so the G9 arm below fails
there BEHAVIOURALLY — observing ``budget_exhausted`` where the tip observes
``round_cap_reached`` — rather than failing on an absent symbol, which would prove
nothing. Nothing here imports another test module either, so a reviewer can drop this
one file onto the base SHA and watch it go red without carrying anything else across.

F-074: the HIGH half of F-070 is a claim about PRODUCTION output, so ``A1`` drives
``run_rag_loop`` itself — the default ``max_rounds=1``, a round that produced
integrable claims, and a deadline nowhere near expiring — not ``decide`` alone. A
``LoopState`` unit arm would be the replica trap one level down: the controller, not
``decide``, is what ``streamlit_app.py:1426`` calls.

WHAT EACH ARM DOES AT THE BASE — MEASURED, not inferred, against the real ``ac77668``
modules (G11 report ``14-g9-base-ac77668-remeasured``): **7 failed, 4 passed.** Read
this before interpreting a base run; most of these reds are the point of the file.

======================================  ======  ====================================
arm                                     base    what it is
======================================  ======  ====================================
``a1``  default ceiling, run_rag_loop    RED     G9. Observes ``budget_exhausted``
                                                where the tip observes the ceiling.
``a2``  spent clock  (2 arms)            GREEN   true non-regression pins: the time
                                                bound is untouched at both ends.
``a3``  both bounds at once              RED     its REASON half is genuinely
                                                unchanged; the ``also_true ==
                                                (ROUND_CAP,)`` half cannot hold at
                                                the base, where the key does not
                                                exist. Half pin, half tip-only.
``a4``  boundedness                      GREEN   green at BOTH ends BY DESIGN — it
                                                is the invariant that must SURVIVE.
``a5``  a4 non-vacuity (RULING 13)       RED     tip-only: at the base there is no
                                                round-cap key to neutralize.
``a6``  precedence deltas                RED     G9. Observes the reason strings.
``a7``  ranked last / appended           RED     tip-only: the base tuple has 7.
``a7``  two vocabularies differ          RED     tip-only: SET-EQUAL at the base.
``a7``  budget_exhausted is the clock     RED     G9-grade behavioural: at the base it
                                                fires with 900 s of clock left.
``a7``  exemption carried, not widened   GREEN   preservation pin, both ends.
======================================  ======  ====================================

So the G9 base-failure evidence is ``a1``, ``a6`` and the third ``a7``, all of which
observe a reason STRING. The others are either preservation pins that must be green
at both ends, or tip-only invariants — never dress a tip-only red as G9 evidence.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import pytest

from t2pw.pipeline import deadline as dl
from t2pw.pipeline.lineage import LINEAGE_KEY
from t2pw.rag import loop_policy as lp
from t2pw.rag.admission import SCOPE_ADMITTED
from t2pw.rag.controller import RoundContext, RoundResult, RoundStages, run_rag_loop
from t2pw.rag.graph_delta import REENTRY_STAGES

#: The eight reason literals, spelled out. A rename of any constant must not be able
#: to follow this file along, or the G9 arms would silently stop observing anything.
BUDGET = "budget_exhausted"
TIMEOUT = "operation_timeout"
EMPTY = "identical_empty_response"
UNRECOVERABLE = "scientifically_unrecoverable"
ATTEMPT_CAP = "attempt_cap_reached"
EXHAUSTED = "retrieval_exhausted"
NO_NEW = "no_new_claims"
ROUND_CAP = "round_cap_reached"


# ------------------------------------------------------- A1 fixtures (production path)
# Deliberately a private copy of ``test_rag_loop_controller``'s minimal fixture set:
# this file must stand alone at the base SHA.

GAP = "gap-missing_step-1a2b3c4d"
SPAN = "LpxC deacetylates UDP-3-O-acyl-GlcNAc."
PAPER = "PMC11046580"
ENTRY = {"stage": "rag_admission", "origin": "rag_literature", "support": "direct",
         "paper_explicit": "not_explicit", "reason": "fills the detected gap",
         "review_required": False, "sources": [{"source_id": PAPER}]}
CORE = {"name": "core step", "inputs": ["x", "w"], "outputs": ["y"], "enzymes": ["LpxA"]}
HEALTHY = dict(detected_gap_ids=frozenset({GAP}), retrieval_completed=True)


def base_graph() -> dict:
    return {"entities": {"compounds": [{"name": "x"}]},
            "processes": {"reactions": [dict(CORE)]}}


def row(name: str = "deacetylation") -> dict:
    """One admissible RAG-added reaction against the one detected gap."""
    return {"name": name, "inputs": ["a"], "outputs": ["b"], "enzymes": ["LpxC"],
            "scope_membership": SCOPE_ADMITTED, "evidence": SPAN,
            "source_papers": [{"source_id": PAPER}], LINEAGE_KEY: [dict(ENTRY)],
            "rag_provenance": {"source_id": PAPER, "chunk_id": "c1", "gap_id": GAP}}


def claim() -> dict:
    """The claim the ``row`` above came from — the same identity key."""
    return {"inputs": ["a"], "outputs": ["b"], "enzymes": ["LpxC"], "evidence": SPAN,
            "source_papers": [{"source_id": PAPER}],
            "rag_provenance": {"source_id": PAPER, "chunk_id": "c1"}}


def productive_round():
    """A round that retrieved, admitted and merged one genuinely new reaction."""
    rows = [row()]
    claims = [claim()]

    def run(ctx: RoundContext) -> RoundResult:
        run.calls.append(ctx)
        after = deepcopy(ctx.graph)
        after["processes"]["reactions"].extend(deepcopy(r) for r in rows)
        return RoundResult(graph=after, **dict(HEALTHY, claims=claims,
                                               admitted_claims=claims))
    run.calls = []
    return run


def stages() -> RoundStages:
    return RoundStages(**{name: (lambda g: g) for name in REENTRY_STAGES})


def clock_from(*ticks):
    it, held = iter(ticks), [0.0]

    def clock():
        held[0] = next(it, held[0])
        return held[0]
    return clock


# ----------------------------------------------------------------- A1 (G9, F-074)
def test_a1_g9_the_default_round_ceiling_is_named_a_ceiling_not_an_exhausted_budget():
    """G9 BEHAVIOURAL BASE PROOF — fails at ``ac77668``, passes at the tip.

    F-070's default path, driven through the PRODUCTION entry point. ``max_rounds=1``
    is ``controller.py``'s default and ``streamlit_app.py``'s production value, the
    round produced an integrable claim, and 10 000 s of the deadline are untouched.
    At the base this settles on ``budget_exhausted`` — D-005's operational-failure
    denominator — for a loop that did exactly what it was configured to do.
    """
    runner = productive_round()
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=runner,
                       deadline=10_000.0, max_rounds=1, clock=lambda: 0.0)

    # The premise: a real round ran, it was admissible, it produced a claim a further
    # round would have integrated, and the clock is nowhere near expiring.
    assert len(runner.calls) == 1
    assert out.rounds[0].accepted is True
    assert out.rounds[0].new_admissible_claims == 1
    assert len(out.graph["processes"]["reactions"]) == 2
    assert out.decision.counts["seconds_remaining"] == 10_000.0
    assert out.decision.counts["rounds_completed"] == 1
    assert out.decision.counts["max_rounds"] == 1

    # The observation. BASE ac77668 reports "budget_exhausted" on this exact run.
    assert out.reason == ROUND_CAP
    assert BUDGET not in (out.reason,) + out.decision.also_true


# ------------------------------------------------------------- A2 non-regression pin
def test_a2_a_genuinely_spent_clock_still_reports_the_operational_failure():
    """Constraint 1, through the production entry point. Green at the base AND the tip.

    The clock, not a ceiling, refused this run: ``max_rounds=5`` with four rounds
    still allowed. ``budget_exhausted`` keeps its D-005 meaning exactly.
    """
    runner = productive_round()
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=runner,
                       deadline=100.0, max_rounds=5, clock=clock_from(0.0, 100.0))
    assert len(runner.calls) == 1                    # the second round never started
    assert out.decision.counts["seconds_remaining"] == 0.0
    assert out.reason == BUDGET
    assert ROUND_CAP not in (out.reason,) + out.decision.also_true


def test_a2_a_reserve_that_does_not_fit_is_still_an_exhausted_budget():
    """The other half of the time bound: the finalization reserve, not the deadline."""
    runner = productive_round()
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=runner,
                       deadline=100.0, max_rounds=5, next_round_reserve_seconds=90.0,
                       clock=clock_from(20.0))
    assert runner.calls == []                        # 80 s left, 90 s needed
    assert out.reason == BUDGET
    assert ROUND_CAP not in (out.reason,) + out.decision.also_true


# ---------------------------------------------------------- A3 both bounds at once
def test_a3_both_bounds_at_once_still_report_the_operational_failure_first():
    """Charter § 3: where both bounds hold, ``budget_exhausted`` must still WIN, with
    the ceiling demoted to ``also_true``. The reason half of this must not regress."""
    runner = productive_round()
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=runner,
                       deadline=100.0, max_rounds=1, clock=clock_from(0.0, 100.0))
    assert out.reason == BUDGET                      # unchanged from the base
    assert out.decision.also_true == (ROUND_CAP,)    # and the ceiling is not lost


# --------------------------------------------- the decide-level state space (A4-A7)
#: Every axis of ``_conditions``, as the LoopState overrides that select it. The
#: product is the whole reachable condition space of the policy: 3 072 states.
AXES = (
    # the round bound: not reached, reached
    (dict(rounds_completed=1, max_rounds=3), dict(rounds_completed=3, max_rounds=3)),
    # the time bound: 900 s left, nothing left
    (dict(now=100.0, deadline=1000.0), dict(now=1000.0, deadline=1000.0)),
    (dict(ladder_completed=False), dict(ladder_completed=True)),
    (dict(round_retrieval_completed=False), dict(round_retrieval_completed=True)),
    (dict(new_admissible_claims=0), dict(new_admissible_claims=2)),
    (dict(graph_delta=0), dict(graph_delta=2)),
    (dict(operation_timed_out=False), dict(operation_timed_out=True)),
    (dict(identical_empty_responses=0), dict(identical_empty_responses=1),
     dict(identical_empty_responses=2)),
    (dict(evidence_sources_exhausted=False), dict(evidence_sources_exhausted=True)),
    (dict(defensible_core=True), dict(defensible_core=False)),
    (dict(attempt_cap_reached=False), dict(attempt_cap_reached=True)),
)


def all_states():
    for combo in product(*AXES):
        over = {}
        for part in combo:
            over.update(part)
        yield lp.LoopState(**over)


def _round_bound(s: lp.LoopState) -> bool:
    """The round bound, computed from the raw fields so this file stays loadable at
    the base SHA where ``LoopState`` has no property for it."""
    return s.rounds_completed >= s.max_rounds


def _time_bound(s: lp.LoopState) -> bool:
    return s.deadline - s.now <= s.next_round_reserve_seconds


def _base_budget_predicate(s: lp.LoopState) -> bool:
    """``budget_exhausted``'s condition EXACTLY as it stood at ``ac77668``:
    ``state.out_of_budget and not exhausted``, with both bounds folded together."""
    return (_round_bound(s) or _time_bound(s)) and not (
        s.ladder_completed and s.new_admissible_claims <= 0)


def _audit(conditions):
    """Two censuses over the whole state space, driven by ``conditions``.

    ``unbounded`` — states that reached a bound and yielded NO stop reason. Any
    member is a loop that spins forever, which is a far worse defect than a
    mislabelled one.

    ``drifted`` — states where the base's single ``budget_exhausted`` predicate is
    not exactly reproduced by the disjunction of the two reasons that replaced it.
    Any member means the split moved a stop condition rather than partitioning it,
    which would be a widened (or narrowed) ``and not exhausted`` exemption.
    """
    unbounded, drifted = [], []
    for s in all_states():
        held = conditions(s)
        reasons = tuple(r for r in lp.TERMINATION_PRECEDENCE if held.get(r))
        if (_round_bound(s) or _time_bound(s)) and not reasons:
            unbounded.append(s)
        if _base_budget_predicate(s) != (held.get(BUDGET, False)
                                         or held.get(ROUND_CAP, False)):
            drifted.append(s)
    return unbounded, drifted


def _split_done_wrong(state: lp.LoopState):
    """``_conditions`` with the round-cap key neutralized — the split done WRONG:
    ``budget_exhausted`` narrowed to the time bound and nothing put in its place."""
    held = dict(lp._conditions(state))
    held[ROUND_CAP] = False
    return held


# ------------------------------------------------------------------ A4 boundedness
def test_a4_boundedness_survives_the_split_and_the_two_reasons_partition_the_old_one():
    """Charter § 3.3 — reaching EITHER bound still always yields a stop reason.

    Green at the base and at the tip, which is the point: this is the invariant that
    must SURVIVE, not the behaviour that must change. The second census is the
    stronger statement — the two predicates do not merely both exist, together they
    are the base's predicate exactly, state for state, so no state gained a stop
    condition and none lost one.
    """
    states = list(all_states())
    assert len(states) == 3072
    # The enumeration actually reaches the case the card is about, and reaches it
    # while the clock is untouched — otherwise the census below proves nothing.
    only_round = [s for s in states if _round_bound(s) and not _time_bound(s)]
    assert len(only_round) == 768

    for s in states:
        held = lp._conditions(s)
        # Constraint 4: a precedence member with no key here is a KeyError in
        # ``decide``, not a silent omission.
        assert set(held) == set(lp.TERMINATION_PRECEDENCE)
        decision = lp.decide(s)
        if _round_bound(s) or _time_bound(s):
            assert decision.should_continue is False
            assert decision.reason in lp.TERMINATION_REASONS

    assert _audit(lp._conditions) == ([], [])


def test_a5_the_boundedness_audit_is_not_vacuous_it_reddens_on_the_split_done_wrong():
    """RULING 13. Neutralize what the audit audits and it must find the hole.

    Without this, ``_audit`` returning ``([], [])`` is indistinguishable from an
    audit that stopped auditing.
    """
    unbounded, drifted = _audit(_split_done_wrong)
    assert unbounded, "the audit cannot see an unbounded loop — it proves nothing"
    assert drifted, "the audit cannot see a lost stop condition — it proves nothing"
    # And it is specific, not merely noisy: every state it flags is one the round
    # bound alone stopped, and none of them is one the clock stopped.
    assert all(_round_bound(s) and not _time_bound(s) for s in unbounded)
    assert all(_round_bound(s) for s in drifted)
    # The control: the same audit against the real mapping is clean.
    assert _audit(lp._conditions) == ([], [])


# ------------------------------------------- A6 the enumerated precedence deltas (G9)
#: The round ceiling reached, the clock nowhere near it, the round productive.
CEILING = dict(rounds_completed=3, max_rounds=3, now=100.0, deadline=1000.0,
               round_retrieval_completed=True, new_admissible_claims=2, graph_delta=2)

#: ``(overrides, base reason at ac77668, tip reason, tip also_true)`` — EVERY
#: combination of the round bound with another condition, and what each reports.
PRECEDENCE_DELTA = (
    # 1. the fix itself: only the ceiling holds.                    CHANGES
    ({}, BUDGET, ROUND_CAP, ()),
    # 2. both bounds: the operational failure still wins.           reason UNCHANGED
    (dict(now=1000.0), BUDGET, BUDGET, (ROUND_CAP,)),
    # 3. an operation genuinely timed out — that IS operational.    CHANGES
    (dict(operation_timed_out=True), BUDGET, TIMEOUT, (ROUND_CAP,)),
    # 4. an inert mechanism was observed during the round.          CHANGES
    (dict(identical_empty_responses=2), BUDGET, EMPTY, (ROUND_CAP,)),
    # 5. an explicit scientific refusal was made.                   CHANGES
    (dict(evidence_sources_exhausted=True, defensible_core=False),
     BUDGET, UNRECOVERABLE, (ROUND_CAP,)),
    # 6. a ladder was cut off MID-FLIGHT by D-024's attempt cap.    CHANGES
    (dict(attempt_cap_reached=True), BUDGET, ATTEMPT_CAP, (ROUND_CAP,)),
    # 7. the ladder finished and produced nothing new — the shared
    #    ``not exhausted`` carve-out suppresses BOTH ceiling reasons. UNCHANGED
    (dict(ladder_completed=True, new_admissible_claims=0, graph_delta=0),
     EXHAUSTED, EXHAUSTED, ()),
    # 8. the finished round observed nothing new while rungs remain. CHANGES
    (dict(new_admissible_claims=0, graph_delta=0), BUDGET, NO_NEW, (ROUND_CAP,)),
)


def test_a6_g9_every_round_ceiling_combination_reports_its_ruled_winner():
    """G9 BEHAVIOURAL BASE PROOF, and charter § 7.2's enumeration.

    Six of the eight rows report a DIFFERENT reason at the tip than at the base.
    That is the deliberate consequence of ranking the ceiling last instead of first.
    FIVE of the six surface a reason the base already held in ``also_true`` and was
    merely suppressing: a real timeout, an inert mechanism, an explicit refusal, a
    mid-flight attempt cap, or a round that observed nothing new. THE SIXTH — row 1,
    the fix itself — is the new reason, which could not have been in ``also_true`` at
    the base because ``_conditions`` had no key for it there at all. Row 1 is not a
    corner case: the census puts 90 of the 576 changed states in that class. The
    assertion below carves it out explicitly rather than letting the docstring
    generalize over it.
    """
    changed = []
    for overrides, base_reason, tip_reason, tip_also in PRECEDENCE_DELTA:
        decision = lp.decide(lp.LoopState(**{**CEILING, **overrides}))
        assert decision.should_continue is False
        assert (decision.reason, decision.also_true) == (tip_reason, tip_also), overrides
        if base_reason != tip_reason:
            changed.append((base_reason, tip_reason, tuple(sorted(overrides))))
            # Only row 1 may report the NEW reason. For every other changed row the
            # winner was already true at the base and already sat in ``also_true``
            # there, so the re-ranking surfaced it rather than inventing it.
            assert tip_reason != ROUND_CAP or overrides == {}
    assert len(changed) == 6
    assert {c[1] for c in changed} == {ROUND_CAP, TIMEOUT, EMPTY, UNRECOVERABLE,
                                       ATTEMPT_CAP, NO_NEW}


# ----------------------------------------------- A7 the shape of the reason vocabulary
def test_a7_the_ceiling_is_ranked_last_and_was_appended_never_reordered():
    """The chosen rank, pinned. The ceiling is the RESIDUAL reason: it wins only when
    nothing else describes the stop. It holds on every loop that runs its full
    allowance — and the default allowance is one round — so any higher rank would
    mask exactly the reasons that tell one run from another."""
    assert lp.TERMINATION_PRECEDENCE[-1] == ROUND_CAP
    assert lp.TERMINATION_PRECEDENCE == (
        BUDGET, TIMEOUT, EMPTY, UNRECOVERABLE, ATTEMPT_CAP, EXHAUSTED, NO_NEW,
        ROUND_CAP)
    assert lp.TERMINATION_REASONS == frozenset(lp.TERMINATION_PRECEDENCE)
    assert len(lp.TERMINATION_REASONS) == len(lp.TERMINATION_PRECEDENCE) == 8
    # The member was APPENDED, so D-024's ranking is untouched and the first seven
    # slots are identical to the base, member for member and position for position.
    # This is the whole delta ``tests/test_attempt_cap_termination_reason.py:249``
    # observes, and it is the ONLY thing that moved in that tuple.
    assert lp.TERMINATION_PRECEDENCE.index(ATTEMPT_CAP) == 4
    assert lp.TERMINATION_PRECEDENCE[:7] == (
        BUDGET, TIMEOUT, EMPTY, UNRECOVERABLE, ATTEMPT_CAP, EXHAUSTED, NO_NEW)


def test_a7_the_two_termination_vocabularies_differ_by_exactly_the_new_member():
    """The CROSS-SEAM relationship, pinned rather than left to a one-shot probe.

    ``pipeline.deadline`` carries its OWN termination-reason vocabulary. At the base
    the two were SET-EQUAL; at the tip ``loop_policy`` is a strict SUPERSET by exactly
    ``round_cap_reached``, and ``deadline``'s closed vocabulary REFUSES that member
    with a ``ValueError``.

    That is safe only because nothing routes a ``loop_policy`` reason into a
    ``deadline`` validator today — ``outcome.reason`` reaches
    ``streamlit_app.py:1467`` and has no readers. Whoever next wires a RAG
    termination reason into a denominator should meet THIS test rather than a
    runtime surprise, so the relationship is asserted, not merely observed once.

    ``deadline`` is imported READ-ONLY. ``src/t2pw/pipeline/`` is a different seam and
    is not this card's to change; that it is untouched is the point being pinned.
    """
    assert set(lp.TERMINATION_REASONS) - set(dl.TERMINATION_REASONS) == {ROUND_CAP}
    assert set(dl.TERMINATION_REASONS) - set(lp.TERMINATION_REASONS) == set()
    # The pipeline vocabulary did not grow, and its operational denominator -- the
    # thing F-070 is ultimately about -- did not grow either.
    assert len(dl.TERMINATION_REASONS) == len(set(dl.TERMINATION_REASONS)) == 7
    assert dl.OPERATIONAL_TERMINATION_REASONS == frozenset({BUDGET, TIMEOUT})
    assert ROUND_CAP not in dl.OPERATIONAL_TERMINATION_REASONS
    with pytest.raises(ValueError):
        dl.require_reason(ROUND_CAP)


def test_a7_budget_exhausted_fires_for_the_clock_and_for_nothing_else():
    """Constraint 1, exhaustively: after the split ``budget_exhausted`` is TRUE only
    where the wall-clock bound is spent. A round ceiling can never raise it again."""
    for s in all_states():
        held = lp._conditions(s)
        if held[BUDGET]:
            assert _time_bound(s), s
        assert held.get(ROUND_CAP, False) is (_round_bound(s) and not (
            s.ladder_completed and s.new_admissible_claims <= 0)), s


def test_a7_the_exemption_is_not_widened_only_carried():
    """Constraint 2 / F-070's named trap. ``and not exhausted`` still suppresses the
    ceiling reasons on EXACTLY the states it suppressed ``budget_exhausted`` on at the
    base — no state where the clock genuinely ran out lost its reason."""
    suppressed_at_base, suppressed_at_tip = [], []
    for s in all_states():
        held = lp._conditions(s)
        if (_round_bound(s) or _time_bound(s)) and not _base_budget_predicate(s):
            suppressed_at_base.append(s)
        if (_round_bound(s) or _time_bound(s)) and not (
                held[BUDGET] or held.get(ROUND_CAP, False)):
            suppressed_at_tip.append(s)
    assert suppressed_at_base == suppressed_at_tip
    assert suppressed_at_base, "the exemption never fires — this pin is vacuous"
    # Every suppressed state is one where the ladder finished with nothing new, and
    # every one of them still stops, on the more specific scientific verdict.
    for s in suppressed_at_tip:
        assert s.ladder_completed and s.new_admissible_claims <= 0
        decision = lp.decide(s)
        assert decision.should_continue is False
        # It still stops, and on a reason at least as specific: ``retrieval_exhausted``
        # always holds for these, and only ranks 1-4 can outrank it.
        assert EXHAUSTED in (decision.reason,) + decision.also_true
