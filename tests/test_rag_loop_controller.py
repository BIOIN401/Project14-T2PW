"""C-043 — the RAG loop controller (PRODUCT_CONTRACT §10/§9, DECISIONS D-008).

NEW acceptance tests, every one of them. ``t2pw.rag.controller`` does not exist at the
C-043 base SHA `e4eeef4`, so these replace NO prior observable behaviour and are NOT G9
regression evidence. The module is additive and UNWIRED — no production code calls it —
so there is no pre-existing behaviour for it to correct and no base-SHA failure to show.
No fabricated base failure was created for this card, and no test here asserts a bare
``ImportError``: symbol absence is not behavioural proof.

The controller decides nothing the two merged policies already decide, so these tests
check what the *controller* adds: it drives all five D-008 stages itself, it checkpoints
before each round, it refuses to start a round the budget cannot afford, it never
advances the canonical graph on a delta the validator refused, and it never converts an
operational failure into a scientific verdict.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from copy import deepcopy

import pytest

from t2pw.pipeline.lineage import LINEAGE_KEY
from t2pw.rag import controller
from t2pw.rag.admission import SCOPE_ADMITTED, SCOPE_REJECTED
from t2pw.rag.controller import (
    STAGE_FIELDS, Checkpoint, LoopOutcome, RoundAborted, RoundContext, RoundOutcome,
    RoundResult, RoundStages, run_rag_loop,
)
from t2pw.rag.graph_delta import (
    FALSE, NOT_EVALUATED, REENTRY_STAGES, RULE_ADMISSION_VERDICT, RULE_DUPLICATE_CLAIM,
    RULE_EVIDENCE_RETENTION, RULE_GAP_ATTRIBUTION, RULE_GAP_NOT_DETECTED,
    RULE_MUTATED_ELEMENT, RULE_PROVENANCE, RULE_REJECTED_CLAIM, RULE_REMOVED_ELEMENT,
    RULES, TRUE, RoundRecord, validate_graph_delta,
)
from t2pw.rag.loop_policy import (
    BUDGET_EXHAUSTED, IDENTICAL_EMPTY_RESPONSE, OPERATION_TIMEOUT, RETRIEVAL_EXHAUSTED,
    TERMINATION_REASONS, claim_identity_key,
)

GAP = "gap-missing_step-1a2b3c4d"
SPAN = "LpxC deacetylates UDP-3-O-acyl-GlcNAc."
PAPER = "PMC11046580"
#: A valid C-015 lineage entry — the same shape ``test_rag_graph_delta`` validates.
ENTRY = {"stage": "rag_admission", "origin": "rag_literature", "support": "direct",
         "paper_explicit": "not_explicit", "reason": "fills the detected gap",
         "review_required": False, "sources": [{"source_id": PAPER}]}
CORE = {"name": "core step", "inputs": ["x", "w"], "outputs": ["y"], "enzymes": ["LpxA"]}
#: A round that retrieved normally against the one detected gap.
HEALTHY = dict(detected_gap_ids=frozenset({GAP}), retrieval_completed=True)


def base_graph() -> dict:
    return {"entities": {"compounds": [{"name": "x"}]},
            "processes": {"reactions": [dict(CORE)]}}


def row(name="deacetylation", inputs=("a",), outputs=("b",), **over) -> dict:
    """One admissible RAG-added reaction; ``over`` set to ``None`` DROPS that key."""
    r = {"name": name, "inputs": list(inputs), "outputs": list(outputs),
         "enzymes": ["LpxC"], "scope_membership": SCOPE_ADMITTED, "evidence": SPAN,
         "source_papers": [{"source_id": PAPER}], LINEAGE_KEY: [dict(ENTRY)],
         "rag_provenance": {"source_id": PAPER, "chunk_id": "c1", "gap_id": GAP}}
    r.update(over)
    return {k: v for k, v in r.items() if v is not None}


def claim(inputs=("a",), outputs=("b",)) -> dict:
    """The claim a ``row`` of the same chemistry came from — same identity key."""
    return {"inputs": list(inputs), "outputs": list(outputs), "enzymes": ["LpxC"],
            "evidence": SPAN, "source_papers": [{"source_id": PAPER}],
            "rag_provenance": {"source_id": PAPER, "chunk_id": "c1"}}


def ok(rows, **kw):
    """A healthy round spec that merges ``rows`` and admits a claim for each."""
    claims = [claim(r["inputs"], r["outputs"]) for r in rows]
    return rows, dict(HEALTHY, claims=claims, admitted_claims=claims, **kw)


def stages(log=None, fail=None, **override) -> RoundStages:
    """The five D-008 stages as identity transforms, recording their call order."""
    def make(name):
        def stage(graph):
            if log is not None:
                log.append(name)
            if name == fail:
                raise RuntimeError(f"{name} exploded")
            return graph
        return stage
    return RoundStages(**{**{n: make(n) for n in REENTRY_STAGES}, **override})


def scripted(*specs):
    """One ``RoundResult`` per round from ``(rows, kwargs)`` specs; the last repeats."""
    def run(ctx: RoundContext) -> RoundResult:
        rows, kw = specs[min(ctx.round_index, len(specs) - 1)]
        run.calls.append(ctx)
        after = deepcopy(ctx.graph)
        after["processes"]["reactions"].extend(deepcopy(r) for r in rows)
        return RoundResult(graph=after, **kw)
    run.calls = []
    return run


def clock_from(*ticks):
    """A monotonic clock returning each tick once, then holding the last."""
    it, held = iter(ticks), [0.0]

    def clock():
        held[0] = next(it, held[0])
        return held[0]
    return clock


# --------------------------------------------------------------------------- D-008

def test_the_controller_drives_all_five_d008_stages_in_order_every_round():
    """NEW acceptance. Catches a controller that runs the stages out of D-008's order,
    runs only some of them, or runs them once for a multi-round loop."""
    log = []
    runner = scripted(ok([row()]), ok([row("second", ("b",), ("c",))]))
    out = run_rag_loop(base_graph(), stages=stages(log), round_runner=runner,
                       deadline=1000.0, max_rounds=2, clock=lambda: 0.0)
    assert log == list(REENTRY_STAGES) * 2
    assert [r.verdict.reentry for r in out.rounds] == [
        dict.fromkeys(REENTRY_STAGES, TRUE)] * 2
    assert [r.accepted for r in out.rounds] == [True, True]
    assert len(out.graph["processes"]["reactions"]) == 3


def test_a_round_that_omits_a_d008_stage_cannot_be_constructed():
    """NEW acceptance. Catches an optional stage — the structural half of D-008: a caller
    that skips one cannot build the argument, so the round is unrunnable, not refused."""
    assert STAGE_FIELDS == REENTRY_STAGES  # one definition of the five, imported
    complete = {name: (lambda g: g) for name in REENTRY_STAGES}
    for missing in REENTRY_STAGES:
        with pytest.raises(TypeError):
            RoundStages(**{k: v for k, v in complete.items() if k != missing})
    assert dataclasses.is_dataclass(RoundStages(**complete))


def test_no_input_lets_a_caller_assert_that_a_stage_re_entered():
    """NEW acceptance. Catches a re-entry flag on the round's own report, which would let
    a caller claim a stage ran that never did. The map is the controller's observation."""
    surface = {f.name for cls in (RoundResult, RoundContext, Checkpoint, RoundOutcome,
                                  LoopOutcome) for f in dataclasses.fields(cls)}
    surface |= set(inspect.signature(run_rag_loop).parameters)
    assert not surface & set(REENTRY_STAGES)
    assert not {n for n in surface if "reenter" in n or "reentry" in n}


def test_a_failing_stage_aborts_the_round_and_manufactures_no_termination_reason():
    """NEW acceptance. Catches a controller that swallows a stage failure and merges
    anyway, that loses the last good state, or that relabels an infrastructure failure as
    one of D-005's six scientific/operational reasons."""
    saved, log = [], []
    with pytest.raises(RoundAborted) as raised:
        run_rag_loop(base_graph(), stages=stages(log, fail="gates"),
                     round_runner=scripted(ok([row()])), deadline=1000.0, max_rounds=3,
                     clock=lambda: 0.0, checkpoint=saved.append)
    aborted = raised.value
    assert (aborted.stage, log) == ("gates", ["normalization", "mapping", "gates"])
    assert aborted.reentered == {"normalization": True, "mapping": True, "gates": False}
    # Stages after the failure are ABSENT, so the validator reports not_evaluated (§11).
    reentry = validate_graph_delta(base_graph(), base_graph(),
                                   RoundRecord(reentered=aborted.reentered)).reentry
    assert reentry["gates"] == FALSE
    assert reentry["persistence"] == reentry["classification"] == NOT_EVALUATED
    # No D-005 reason was manufactured, and the pre-round checkpoint holds the graph.
    assert not hasattr(aborted, "reason")
    assert (aborted.checkpoint.decision.should_continue,
            aborted.checkpoint.decision.reason) == (True, None)
    assert saved[0].graph == aborted.checkpoint.graph == base_graph()


def test_a_stage_that_drops_a_pre_existing_row_is_refused_and_never_merged():
    """NEW acceptance. Catches a controller that lets a late stage delete established
    biology — the delta is additive by construction and a removal is reported, not kept."""
    def dropper(graph):
        graph = deepcopy(graph)
        graph["processes"]["reactions"] = [
            r for r in graph["processes"]["reactions"] if r.get("name") != CORE["name"]]
        return graph
    out = run_rag_loop(base_graph(), stages=stages(persistence=dropper),
                       round_runner=scripted(ok([row()])), deadline=1000.0,
                       max_rounds=1, clock=lambda: 0.0)
    assert RULE_REMOVED_ELEMENT in out.rounds[0].verdict.rules()
    assert out.rounds[0].accepted is False
    assert out.graph == base_graph()


def test_a_stage_that_mutates_a_pre_existing_row_in_place_is_refused():
    """NEW acceptance. Catches the composition-level fail-open: handing the validator the
    LIVE graph as ``graph_before``. A round may return a payload that SHARES row objects
    with its input; a stage then editing one in place mutates ``graph_before`` too, so
    ``mutated_element``, the pre-existing-row ``evidence_retention`` check and the lineage
    comparison all compare the change against itself and the round is admitted with zero
    violations. That would disarm at composition exactly what H-008 merged to close."""
    extra = row()
    _, kwargs = ok([extra])

    def aliasing(ctx):  # the additions envelope reuses the caller's own row objects
        return RoundResult(graph={
            "entities": ctx.graph["entities"],
            "processes": {"reactions": list(ctx.graph["processes"]["reactions"])
                          + [deepcopy(extra)]}}, **kwargs)

    def rewrite(graph):  # a stage editing an established reaction in place
        graph["processes"]["reactions"][0]["outputs"] = ["WRONG"]
        return graph

    out = run_rag_loop(base_graph(), stages=stages(gates=rewrite), round_runner=aliasing,
                       deadline=1000.0, max_rounds=1, clock=lambda: 0.0)
    assert RULE_MUTATED_ELEMENT in out.rounds[0].verdict.rules()
    assert out.rounds[0].accepted is False
    # The caller's own object was corrupted by its own stage; the pre-round checkpoint is
    # the recovery path and is a deep copy, so it still holds the last good state.
    assert out.checkpoints[0].graph == base_graph()


# ------------------------------------------------------- checkpoint and the deadline

def test_a_checkpoint_is_written_before_every_round_and_holds_the_last_good_graph():
    """NEW acceptance. Catches a checkpoint written after the round (useless to a round
    that never returns), a missing one on later rounds, and a checkpoint that aliases the
    live graph so a stage mutating in place corrupts the state it exists to preserve."""
    order, saved = [], []
    runner = scripted(ok([row()]), ok([row("second", ("b",), ("c",))]))

    def sink(cp: Checkpoint):
        order.append(f"checkpoint-{cp.round_index}")
        saved.append(cp)

    def logged(ctx):
        order.append(f"round-{ctx.round_index}")
        return runner(ctx)

    original = base_graph()
    out = run_rag_loop(original, stages=stages(), round_runner=logged, deadline=1000.0,
                       max_rounds=2, clock=lambda: 0.0, checkpoint=sink)
    assert order == ["checkpoint-0", "round-0", "checkpoint-1", "round-1"]
    assert saved[0].graph == base_graph() and saved[0].graph is not original
    original["processes"]["reactions"].clear()  # the caller's object, mutated afterwards
    assert saved[0].graph == base_graph()       # the checkpoint is unaffected
    assert saved[1].graph == out.rounds[0].graph
    assert len(saved[1].graph["processes"]["reactions"]) == 2


def test_the_deadline_stops_the_loop_without_losing_the_last_good_state():
    """NEW acceptance. Catches a loop that starts a round after the deadline, and one
    that drops the graph it had already earned when it stops."""
    runner = scripted(ok([row()]))
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=runner,
                       deadline=100.0, max_rounds=5, clock=clock_from(0.0, 100.0))
    assert len(runner.calls) == 1                      # the second round never started
    assert out.reason == BUDGET_EXHAUSTED
    assert out.decision.counts["seconds_remaining"] == 0.0
    assert out.graph == out.rounds[0].graph
    assert len(out.graph["processes"]["reactions"]) == 2


def test_no_round_starts_unless_the_finalization_reserve_still_fits():
    """NEW acceptance. Catches a loop that spends the reserve §9 keeps for checkpoint
    persistence, validation and classification, leaving no budget to finish cleanly."""
    runner = scripted(ok([row()]))
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=runner, deadline=100.0,
                       max_rounds=5, next_round_reserve_seconds=90.0,
                       clock=clock_from(20.0))
    assert runner.calls == []                          # 80 s left, 90 s needed
    assert out.reason == BUDGET_EXHAUSTED
    assert (out.graph, out.checkpoints, out.rounds) == (base_graph(), (), ())


# --------------------------------------------------------------------- §10 dedupe

def test_deduplication_is_against_all_claims_ever_seen_not_only_admitted_ones():
    """NEW acceptance. Catches a controller that records only ADMITTED claims, which lets
    a claim the judge discarded return every round so the loop never converges."""
    ghost = claim()                       # round 0 saw it; the judge admitted nothing
    first = row("first", ("p",), ("q",))
    runner = scripted(([first], dict(HEALTHY, claims=[claim(("p",), ("q",)), ghost],
                                     admitted_claims=[claim(("p",), ("q",))])),
                      ok([row()]))        # round 1 re-offers the ghost's chemistry
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=runner,
                       deadline=1000.0, max_rounds=2, clock=lambda: 0.0)
    assert claim_identity_key(ghost) in out.seen_claims.keys
    assert out.rounds[0].accepted is True
    assert out.rounds[1].new_admissible_claims == 0   # newly admitted, but not novel
    assert out.rounds[1].accepted is False
    assert RULE_DUPLICATE_CLAIM in out.rounds[1].verdict.rules()
    assert len(out.graph["processes"]["reactions"]) == 2  # only round 0's addition


def test_a_rejected_claim_reintroduced_by_a_later_stage_is_refused_not_merged():
    """NEW acceptance. Catches a controller that forgets the gate's rejections between
    rounds, and one that lets the merge inside the SAME round reinstate what the gate
    just rejected — §10's "later stage" includes the merge that follows admission."""
    bad, first = claim(), row("first", ("p",), ("q",))
    later = run_rag_loop(base_graph(), stages=stages(), deadline=1000.0, max_rounds=2,
                         clock=lambda: 0.0, round_runner=scripted(
                             ([first], dict(HEALTHY, claims=[claim(("p",), ("q",)), bad],
                                            admitted_claims=[claim(("p",), ("q",))],
                                            rejected_claims=[bad])),
                             ok([row()])))
    assert claim_identity_key(bad) in later.rejected_claim_keys
    assert RULE_REJECTED_CLAIM in later.rounds[1].verdict.rules()
    assert len(later.graph["processes"]["reactions"]) == 2
    same = run_rag_loop(base_graph(), stages=stages(), deadline=1000.0, max_rounds=1,
                        clock=lambda: 0.0, round_runner=scripted(
                            ([row()], dict(HEALTHY, claims=[claim()],
                                           rejected_claims=[claim()]))))
    assert RULE_REJECTED_CLAIM in same.rounds[0].verdict.rules()
    assert same.graph == base_graph()


@pytest.mark.parametrize("over, rule", [
    ({"rag_provenance": None}, RULE_GAP_ATTRIBUTION),
    ({"rag_provenance": {"source_id": PAPER, "chunk_id": "c1", "gap_id": "gap-other"}},
     RULE_GAP_NOT_DETECTED),
    ({"scope_membership": SCOPE_REJECTED}, RULE_ADMISSION_VERDICT),
    ({LINEAGE_KEY: None}, RULE_PROVENANCE),
])
def test_an_inadmissible_delta_never_advances_the_canonical_graph(over, rule):
    """NEW acceptance. Catches a controller that merges a round the validator refused —
    the one way an unattributed, unadmitted or unattributable reaction reaches export."""
    out = run_rag_loop(base_graph(), stages=stages(), round_runner=scripted(
        ok([row(**over)])), deadline=1000.0, max_rounds=1, clock=lambda: 0.0)
    assert rule in out.rounds[0].verdict.rules()
    assert (out.rounds[0].accepted, out.rounds[0].graph_delta) == (False, 0)
    assert out.graph == base_graph()                    # the last good state is kept
    assert out.rounds[0].graph != base_graph()          # the refused round is preserved
    assert out.refused == (out.rounds[0],)


def test_a_row_that_dropped_the_source_identifier_of_its_claim_is_refused():
    """NEW acceptance. Catches a controller that never hands the round's claims to the
    validator: §10 retains supporting passages and source identifiers through final
    export, and the comparison is impossible if the claims are not passed through."""
    other = "PMC99999999"
    lost = row(source_papers=[{"source_id": other}],
               rag_provenance={"source_id": other, "chunk_id": "c1", "gap_id": GAP})
    out = run_rag_loop(base_graph(), stages=stages(), deadline=1000.0, max_rounds=1,
                       clock=lambda: 0.0, round_runner=scripted(ok([lost])))
    assert RULE_EVIDENCE_RETENTION in out.rounds[0].verdict.rules()
    assert PAPER in "".join(v.observed for v in out.rounds[0].verdict.violations)
    assert out.graph == base_graph()


# ------------------------------------------------------- stop reasons, never conflated

def test_a_retrieval_timeout_is_never_recorded_as_a_biological_rejection():
    """NEW acceptance. Catches a controller that marks the claims of a timed-out round
    rejected — a lookup or network failure is not evidence that a claim is false, and a
    claim so marked could never be readmitted by any later round."""
    lost = claim()
    out = run_rag_loop(base_graph(), stages=stages(), deadline=1000.0, max_rounds=3,
                       clock=lambda: 0.0, round_runner=scripted(
                           ([], dict(HEALTHY, claims=[lost], retrieval_completed=False,
                                     operation_timed_out=True))))
    assert out.reason == OPERATION_TIMEOUT
    assert out.rejected_claim_keys == frozenset()          # nothing marked rejected
    assert claim_identity_key(lost) in out.seen_claims.keys  # but it WAS seen
    assert len(out.rounds) == 1


def test_a_completed_ladder_reports_exhaustion_and_a_cut_short_one_reports_budget():
    """NEW acceptance. Catches a controller that feeds ``ladder_completed`` dishonestly,
    which is what turns D-005's operational denominator into a scientific verdict."""
    spent = dict(HEALTHY, claims=[], admitted_claims=[])
    done = run_rag_loop(base_graph(), stages=stages(), deadline=1000.0, max_rounds=3,
                        clock=lambda: 0.0,
                        round_runner=scripted(([], dict(spent, ladder_completed=True))))
    assert done.reason == RETRIEVAL_EXHAUSTED
    assert BUDGET_EXHAUSTED not in (done.reason,) + done.decision.also_true
    cut = run_rag_loop(base_graph(), stages=stages(), deadline=100.0, max_rounds=3,
                       clock=clock_from(0.0, 100.0),
                       round_runner=scripted(([], dict(spent, ladder_completed=False))))
    assert cut.reason == BUDGET_EXHAUSTED
    assert RETRIEVAL_EXHAUSTED not in (cut.reason,) + cut.decision.also_true


def test_two_identical_empty_responses_stop_the_loop_but_two_different_ones_do_not():
    """NEW acceptance. Catches the cross-round bookkeeping: a controller that never
    counts the streak reissues the dead prompt, and one that counts any two empty
    responses stops a loop that was still producing genuinely different responses."""
    empty = dict(HEALTHY, claims=[], retrieval_completed=False)
    out = run_rag_loop(base_graph(), stages=stages(), deadline=1000.0, max_rounds=5,
                       clock=lambda: 0.0, round_runner=scripted(
                           ([], dict(empty, empty_response_hash="h1"))))
    assert out.reason == IDENTICAL_EMPTY_RESPONSE
    assert len(out.rounds) == 2                        # never a third attempt
    varied = run_rag_loop(base_graph(), stages=stages(), deadline=1000.0, max_rounds=3,
                          clock=lambda: 0.0, round_runner=scripted(
                              ([], dict(empty, empty_response_hash="h1")),
                              ([], dict(empty, empty_response_hash="h2")),
                              ([], dict(empty, empty_response_hash="h3"))))
    assert varied.reason == BUDGET_EXHAUSTED
    assert len(varied.rounds) == 3


# ------------------------------------------------------------------ composition proof

def test_the_controller_states_no_policy_and_imports_only_the_two_it_composes():
    """NEW acceptance. Catches a fork of either policy: a controller that names a
    termination reason, a delta rule or the D-008 stage list has restated a decision that
    is already made elsewhere, and the two copies will drift."""
    tree = ast.parse(inspect.getsource(controller))
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not literals & TERMINATION_REASONS
    assert not literals & set(RULES)
    assert not literals & set(REENTRY_STAGES)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"decide", "validate_graph_delta", "claim_identity_key"} <= called
    assert {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} == {
        "__future__", "copy", "dataclasses", "itertools", "typing",
        "t2pw.rag.graph_delta", "t2pw.rag.loop_policy"}
    # No completeness or coverage input: a loop that retrieved until it hit a target
    # number would be inventing biology to satisfy it.
    surface = set(inspect.signature(run_rag_loop).parameters)
    surface |= {f.name for f in dataclasses.fields(RoundResult)}
    surface -= {"retrieval_completed", "ladder_completed"}  # honest retrieval facts
    assert not {n for n in surface if "coverage" in n or "complete" in n}
