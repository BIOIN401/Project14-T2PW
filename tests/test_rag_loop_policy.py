"""C-016 — the RAG loop stopping policy: PRODUCT_CONTRACT.md §10 and D-005."""

from __future__ import annotations

import ast
import inspect

import pytest

from t2pw.rag.loop_policy import (
    ATTEMPT_CAP_REACHED, BUDGET_EXHAUSTED, IDENTICAL_EMPTY_RESPONSE, NO_NEW_CLAIMS,
    OPERATION_TIMEOUT, RETRIEVAL_EXHAUSTED, ROUND_CAP_REACHED,
    SCIENTIFICALLY_UNRECOVERABLE, TERMINATION_PRECEDENCE, TERMINATION_REASONS,
    LoopState, SeenClaims, claim_identity_key, decide,
)

#: A healthy mid-loop state: budget left, the round completed, the graph grew.
BASE = dict(rounds_completed=1, max_rounds=3, now=100.0, deadline=1000.0,
            round_retrieval_completed=True, new_admissible_claims=2, graph_delta=2)

ONLY = {  # per reason, the overrides that make exactly that one reason true
    BUDGET_EXHAUSTED: dict(now=1000.0),
    OPERATION_TIMEOUT: dict(operation_timed_out=True),
    IDENTICAL_EMPTY_RESPONSE: dict(identical_empty_responses=2),
    SCIENTIFICALLY_UNRECOVERABLE: dict(evidence_sources_exhausted=True,
                                       defensible_core=False),
    RETRIEVAL_EXHAUSTED: dict(ladder_completed=True, new_admissible_claims=0,
                              graph_delta=0),
    NO_NEW_CLAIMS: dict(new_admissible_claims=0, graph_delta=0),
    # D-024: budget left, the round produced claims, the ladder did not complete —
    # so the ONLY thing that stopped this loop is the spent attempt ceiling.
    ATTEMPT_CAP_REACHED: dict(attempt_cap_reached=True),
    # C-064: 900 s of clock left, the round produced claims, nothing else fired —
    # so the ONLY thing that stopped this loop is the configured round ceiling.
    ROUND_CAP_REACHED: dict(rounds_completed=3, max_rounds=3),
}


def state(**overrides) -> LoopState:
    return LoopState(**{**BASE, **overrides})


def claim(reactant: str, name: str = "step", gap: str = "gap-1") -> dict:
    return {"gap_id": gap, "name": name, "inputs": [reactant], "outputs": ["b"]}


@pytest.mark.parametrize("reason", TERMINATION_PRECEDENCE)
def test_each_reason_is_produced_by_a_state_that_yields_only_it(reason):
    # BASELINE MOVE, C-042a / D-024: 6 -> 7. Exact delta: ``attempt_cap_reached``
    # added, nothing removed or renamed. Still one precedence slot per reason.
    # BASELINE MOVE, C-064 / F-070: 7 -> 8. Exact delta: ``round_cap_reached`` added
    # in the LAST slot, nothing removed or renamed. The round bound moved out of
    # ``budget_exhausted`` into it; ``budget_exhausted`` is now the time bound alone.
    assert len(TERMINATION_REASONS) == len(TERMINATION_PRECEDENCE) == 8  # all distinct
    decision = decide(state(**ONLY[reason]))
    assert (decision.should_continue, decision.reason, decision.also_true) == (
        False, reason, ())
    assert decide(state(identical_empty_responses=1,  # 1 empty is not 2; core defensible
                        evidence_sources_exhausted=True)).should_continue is True


def test_retrieval_exhausted_unreachable_when_the_deadline_cut_the_ladder_short():
    cut = decide(state(now=1000.0, ladder_completed=False,
                       new_admissible_claims=0, graph_delta=0))
    assert cut.reason == BUDGET_EXHAUSTED
    assert RETRIEVAL_EXHAUSTED not in (cut.reason,) + cut.also_true
    done = decide(state(ladder_completed=True, new_admissible_claims=0, graph_delta=0))
    assert done.reason == RETRIEVAL_EXHAUSTED
    assert BUDGET_EXHAUSTED not in (done.reason,) + done.also_true
    spent = decide(state(now=1000.0, ladder_completed=True,  # clock spent AND ladder DONE
                         new_admissible_claims=0, graph_delta=0))
    assert (spent.reason, spent.also_true) == (RETRIEVAL_EXHAUSTED, ())


def test_a_rejected_claim_is_not_new_in_a_later_round_so_the_loop_converges():
    seen, novel = SeenClaims().observe([claim("a"), claim("c")])  # both REJECTED
    assert len(novel) == 2
    # Re-offered against a DIFFERENT gap, under a DIFFERENT paper's wording: same claim.
    again, new2 = seen.observe([claim("a", name="R4 demethylation", gap="gap-9"),
                                claim("c", gap="gap-9")])
    assert new2 == () and again.keys == seen.keys
    assert claim_identity_key(claim("a")) in seen.keys  # a later stage refuses it
    decision = decide(state(new_admissible_claims=len(new2), graph_delta=0))
    assert (decision.should_continue, decision.reason) == (False, NO_NEW_CLAIMS)


def test_precedence_prefers_the_documented_winner_when_two_conditions_hold():
    operational = decide(state(now=1000.0, operation_timed_out=True))
    assert operational.reason == BUDGET_EXHAUSTED
    assert OPERATION_TIMEOUT in operational.also_true
    mechanism = decide(state(identical_empty_responses=2,
                             evidence_sources_exhausted=True, defensible_core=False))
    assert mechanism.reason == IDENTICAL_EMPTY_RESPONSE
    assert SCIENTIFICALLY_UNRECOVERABLE in mechanism.also_true


@pytest.mark.parametrize("bound, reason", [
    (dict(rounds_completed=3, max_rounds=3), ROUND_CAP_REACHED),
    (dict(now=999.5, next_round_reserve_seconds=60.0), BUDGET_EXHAUSTED)])
def test_each_bound_stops_the_loop_and_now_says_which_bound_it_was(bound, reason):
    """BASELINE MOVED, C-064 / F-070. At ``ac77668`` BOTH arms asserted
    ``budget_exhausted``, which reported a configured ceiling as D-005's operational
    failure. Exact delta: the round-bound arm's expected reason, and nothing else —
    the time-bound arm is untouched, and both still STOP, which is boundedness."""
    assert decide(state()).should_continue is True  # same state, neither bound spent
    decision = decide(state(**bound))
    assert (decision.should_continue, decision.reason) == (False, reason)
    assert decision.also_true == ()                 # exactly one bound is spent


def test_both_bounds_at_once_keep_reporting_the_operational_failure_first():
    """The ceiling never displaces a genuinely spent clock: ``budget_exhausted`` is
    rank 1 and the ceiling is rank 8, so a run that hit both is still an operational
    failure — and the ceiling is recorded rather than lost."""
    both = decide(state(rounds_completed=3, max_rounds=3, now=1000.0))
    assert (both.should_continue, both.reason) == (False, BUDGET_EXHAUSTED)
    assert both.also_true == (ROUND_CAP_REACHED,)


def test_decide_is_pure_and_the_module_imports_no_clock_no_io():
    tree = ast.parse(inspect.getsource(inspect.getmodule(decide)))
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                 for a in n.names}
    assert imported == {"__future__", "dataclasses", "typing", "hashlib"}  # no clock/IO
    before = state(operation_timed_out=True)
    assert decide(before) == decide(before)           # same input, same output
    assert before == state(operation_timed_out=True)  # the input was not mutated


class _Audited:  # RagReactionCandidate-shaped: the object itself collapses its synonyms
    outputs, enzymes, reversible = ["b"], [], False
    def __init__(self, wording): self.inputs = [wording]
    def claim_identity(self): return ("acetyl-coenzyme-a",), ("b",), (), False


def test_the_audited_claim_identity_decides_the_key_not_the_mapping_fallback():
    """Pins the ``claim_identity()`` branch: delete it and the raw wording is read, so
    ONE chemistry offered under two synonyms reads as two claims and the loop spins."""
    key = claim_identity_key(_Audited("acetyl-CoA"))
    assert key == claim_identity_key(_Audited("Acetyl coenzyme A"))
    assert key != claim_identity_key({"inputs": ["acetyl-CoA"], "outputs": ["b"]})


def test_internal_whitespace_does_not_split_one_chemistry_into_two_claims():
    """§10 convergence: another paper's spacing is not another claim."""
    seen, _ = SeenClaims().observe([{"inputs": ["acetyl CoA"], "outputs": ["b"]}])
    assert seen.observe([{"inputs": ["acetyl  CoA"], "outputs": ["b"]}])[1] == ()


def test_a_scalar_string_field_never_collides_with_its_own_characters():
    """Iterating ``"abc"`` yields ``'a','b','c'``: two chemistries sharing one key."""
    assert (claim_identity_key({"inputs": "abc", "outputs": ["b"]})
            != claim_identity_key({"inputs": ["a", "b", "c"], "outputs": ["b"]}))
