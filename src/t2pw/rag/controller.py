"""Drive the bounded RAG loop -- PRODUCT_CONTRACT.md §10 and §9, DECISIONS.md D-008.

CONTROL FLOW ONLY. Every *policy* question is imported and delegated, never restated
here: "go round again, and if not, exactly why not?" is
:func:`t2pw.rag.loop_policy.decide`; "is this round's delta admissible?" is
:func:`t2pw.rag.graph_delta.validate_graph_delta`; "is this the same claim?" is
:func:`t2pw.rag.loop_policy.claim_identity_key`. This module owns the loop, the
checkpoint and the D-008 stage sequencing, and **no rule** -- it names none of the six
termination reasons and none of the eleven delta rules. No network, no LLM, no database
and no clock of its own (``clock`` is an input), so a test drives a whole loop without
RAG. **UNWIRED**: nothing in production calls it; wiring is C-055's.

D-008 (LOCKED) and §10 -- "every RAG round must re-enter normalization, mapping, gates,
persistence and classification". Enforced STRUCTURALLY, never by asking the caller:

* :class:`RoundStages` has one REQUIRED field per stage, so a caller that omits one
  cannot construct the argument at all -- the round is unrunnable, not merely refused.
* Those fields are pinned to :data:`~t2pw.rag.graph_delta.REENTRY_STAGES` at import
  (:data:`STAGE_FIELDS`) and the loop iterates THAT tuple, so the five stages and their
  order have exactly one definition in the subsystem and cannot drift apart.
* The CONTROLLER invokes them. No input carries a re-entry claim, so ``reentered``
  records only stages that actually ran and returned; a caller cannot assert one did.
* A stage that raises aborts the round with :class:`RoundAborted`, which carries the
  partial record and the checkpoint: the canonical graph does not advance, and stages
  after the failure stay ABSENT, which ``validate_graph_delta`` reports
  ``not_evaluated`` -- never ``false`` (§11).

§9 -- a checkpoint is persisted BEFORE the round's retrieval, so a deadline, an abort or
a crash still leaves the last good graph, the whole claim history and the stop reason
recoverable. The loop is deadline-aware by REFUSING TO START a round whose finalization
reserve does not fit; whether it fits is ``LoopState.out_of_budget``'s question, asked
there and not answered here.

An operational failure is NEVER converted into a scientific verdict. :class:`RoundAborted`
carries no termination reason; a round the validator refuses reports a ``graph_delta`` of
0 because the canonical graph genuinely did not change; and a claim enters
``rejected_claim_keys`` only when the caller reports the GATE rejected it -- never
because retrieval timed out, a stage failed or a delta was inadmissible. Relabelling an
operational stop as a scientific one is the misreport D-005 forbids by name, and a
retrieval that ran out of time is not evidence that its claims are false.

The loop has no completeness input and no coverage input, deliberately: a controller that
kept retrieving until it reached a target number would be inventing biology to satisfy it.
It continues only while :func:`decide` finds no reason to stop.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, fields
from itertools import chain
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple,
)

from t2pw.rag.graph_delta import (
    REENTRY_STAGES, DeltaVerdict, RoundRecord, validate_graph_delta,
)
from t2pw.rag.loop_policy import (
    LoopDecision, LoopState, SeenClaims, claim_identity_key, decide,
)


@dataclass(frozen=True)
class RoundStages:
    """The five stages D-008 requires EVERY round to re-enter, as ``graph -> graph``.

    Every field is REQUIRED: omitting one is a :class:`TypeError` at construction, so
    "a round that skips a stage" is not a round this controller can be asked to run.
    The names and their order are :data:`~t2pw.rag.graph_delta.REENTRY_STAGES`, checked
    at import by :data:`STAGE_FIELDS` -- the list of five is imported, never copied.
    """

    normalization: Callable[[Any], Any]
    mapping: Callable[[Any], Any]
    gates: Callable[[Any], Any]
    persistence: Callable[[Any], Any]
    classification: Callable[[Any], Any]


#: The stage fields in declaration order. Pinned to ``graph_delta``'s tuple below, so a
#: change to D-008's five breaks this module loudly instead of silently skipping one.
STAGE_FIELDS: Tuple[str, ...] = tuple(f.name for f in fields(RoundStages))
if STAGE_FIELDS != REENTRY_STAGES:  # pragma: no cover - import-time structural guard
    raise RuntimeError(
        f"RoundStages {STAGE_FIELDS} must be exactly the D-008 stages {REENTRY_STAGES}")


@dataclass(frozen=True)
class RoundContext:
    """What the round is given: the last good graph, and what the loop already knows.

    ``seconds_remaining`` is the caller's own budget for this round, read off the one
    monotonic per-leg deadline (§9), so retrieval can bound itself rather than discover
    the deadline by overrunning it.
    """

    graph: Any
    round_index: int
    seconds_remaining: float
    seen_claims: SeenClaims
    rejected_claim_keys: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class RoundResult:
    """What one round retrieved, admitted and merged -- observations, never verdicts.

    ``graph`` is the CANDIDATE payload (``conform`` builds the additions
    ``pipeline.merge_additions`` injects; this controller neither conforms nor merges).
    ``claims`` is EVERY claim of the round whatever the judge said -- §10 deduplicates
    against all claims ever seen -- and ``admitted_claims`` / ``rejected_claims`` are the
    gate's verdicts on them. Only ``rejected_claims`` marks a claim rejected; a claim lost
    to a timeout belongs in ``claims`` alone, because a lookup failure is not a
    biological rejection.

    Unpopulated defaults stop rather than spin, and ``defensible_core`` defaults TRUE, so
    the controller never manufactures scientific unrecoverability it did not observe.
    """

    graph: Any
    claims: Sequence[Any] = ()
    admitted_claims: Sequence[Any] = ()
    rejected_claims: Sequence[Any] = ()
    detected_gap_ids: FrozenSet[str] = frozenset()
    retrieval_completed: bool = False
    ladder_completed: bool = False
    operation_timed_out: bool = False
    evidence_sources_exhausted: bool = False
    defensible_core: bool = True
    #: Hash of a structurally empty response; two CONSECUTIVE identical ones are what
    #: D-005 counts. The count is bookkeeping here; the threshold stays in the policy.
    empty_response_hash: Optional[str] = None


@dataclass(frozen=True)
class Checkpoint:
    """The state as of BEFORE a round -- §9's "checkpoint persisted before any
    potentially long LLM or retrieval call", and what a deadline or an abort falls back
    to. ``graph`` is a deep copy, so a stage that mutates its input in place cannot
    corrupt the last good state the checkpoint exists to preserve. ``decision.counts``
    carries the elapsed/remaining budget §9 requires preserved."""

    round_index: int
    graph: Any
    seen_claims: SeenClaims
    rejected_claim_keys: FrozenSet[str]
    decision: LoopDecision


@dataclass(frozen=True)
class RoundOutcome:
    """One round: the validator's verdict, and whether the canonical graph advanced.

    ``graph`` is what the round produced even when ``accepted`` is false -- §9 preserves
    the retrieved evidence of a round that did not land, so it can be diagnosed instead
    of silently discarded.
    """

    index: int
    accepted: bool
    verdict: DeltaVerdict
    graph: Any
    novel_claim_keys: FrozenSet[str] = frozenset()
    new_admissible_claims: int = 0
    graph_delta: int = 0


@dataclass(frozen=True)
class LoopOutcome:
    """The last GOOD graph, the policy's stop decision, and every round's record."""

    graph: Any
    decision: LoopDecision
    rounds: Tuple[RoundOutcome, ...] = ()
    checkpoints: Tuple[Checkpoint, ...] = ()
    seen_claims: SeenClaims = SeenClaims()
    rejected_claim_keys: FrozenSet[str] = frozenset()

    @property
    def reason(self) -> Optional[str]:
        """The policy's termination reason -- read from the decision, never re-derived."""
        return self.decision.reason

    @property
    def refused(self) -> Tuple[RoundOutcome, ...]:
        """Rounds whose delta the validator refused; their graphs never merged."""
        return tuple(r for r in self.rounds if not r.accepted)


class RoundAborted(RuntimeError):
    """A D-008 stage failed, so the round cannot be completed or validated.

    Carries NO termination reason on purpose: an infrastructure failure is not one of
    D-005's six, and presenting it as one is the misreport D-005 forbids by name. What it
    does carry is the record -- the failing ``stage``, the partial ``reentered`` map, and
    the ``checkpoint`` written before the round -- so the round is recorded and the last
    good state is not lost. Stages after the failure are ABSENT from the map, which
    ``validate_graph_delta`` reports ``not_evaluated``; §11 forbids collapsing that into
    ``false``.
    """

    def __init__(self, stage: str, reentered: Dict[str, bool],
                 checkpoint: Checkpoint) -> None:
        super().__init__(f"D-008 stage {stage!r} failed; round "
                         f"{checkpoint.round_index} aborted and not merged")
        self.stage = stage
        self.reentered = dict(reentered)
        self.checkpoint = checkpoint


#: Runs one round -- retrieve, admit, merge -- and reports what it observed.
RoundRunner = Callable[[RoundContext], RoundResult]


def _round_claims(result: RoundResult) -> Iterable[Any]:
    """Every claim the round touched. The verdict lists are included as well as
    ``claims`` so that a claim can never escape the history by being reported only as
    admitted or only as rejected -- §10 deduplicates against ALL claims ever seen, and
    ``SeenClaims.observe`` collapses the repeats by identity anyway."""
    return chain(result.claims, result.admitted_claims, result.rejected_claims)


def _empty_streak(result: RoundResult, streak: int,
                  previous: Optional[str]) -> Tuple[int, Optional[str]]:
    """Consecutive structurally-empty responses carrying the SAME hash. Counting across
    rounds is the loop's job; ``decide`` owns what the count means and when it fires."""
    if not result.empty_response_hash:
        return 0, None
    if result.empty_response_hash == previous:
        return streak + 1, result.empty_response_hash
    return 1, result.empty_response_hash


def run_rag_loop(graph: Any, *, stages: RoundStages, round_runner: RoundRunner,
                 deadline: float, max_rounds: int = 1,
                 next_round_reserve_seconds: float = 0.0,
                 clock: Callable[[], float] = time.monotonic,
                 checkpoint: Optional[Callable[[Checkpoint], None]] = None,
                 seen_claims: SeenClaims = SeenClaims(),
                 rejected_claim_keys: Iterable[str] = ()) -> LoopOutcome:
    """Run the bounded RAG loop over ``graph`` and return the last GOOD state.

    ``deadline`` is required and is a reading of ``clock``'s scale -- the one monotonic
    per-leg deadline of §9. Each iteration asks :func:`decide` BEFORE spending anything,
    so no round starts unless a further round plus ``next_round_reserve_seconds`` fits;
    then it checkpoints, runs the round, drives all five D-008 stages, and validates the
    delta. A delta the validator refuses does not advance ``graph``.

    Nothing is repaired and no rule is softened: an inadmissible delta is reported to the
    caller through :attr:`LoopOutcome.refused`, exactly as ``graph_delta`` reports it.
    """
    seen = seen_claims
    rejected = frozenset(rejected_claim_keys)
    rounds: List[RoundOutcome] = []
    checkpoints: List[Checkpoint] = []
    observed: Dict[str, Any] = {}  # nothing observed before round 1; LoopState's defaults
    streak, previous_hash = 0, None
    while True:
        now = clock()
        state = LoopState(rounds_completed=len(rounds), max_rounds=max_rounds, now=now,
                          deadline=deadline,
                          next_round_reserve_seconds=next_round_reserve_seconds,
                          **observed)
        decision = decide(state)
        if not decision.should_continue:
            break
        mark = Checkpoint(len(rounds), deepcopy(graph), seen, rejected, decision)
        checkpoints.append(mark)
        if checkpoint is not None:
            checkpoint(mark)  # BEFORE retrieval: §9's pre-call checkpoint
        result = round_runner(RoundContext(graph, len(rounds), deadline - now, seen,
                                           rejected))
        started = seen  # the history as of the START of the round, for the validator
        seen, novel = seen.observe(_round_claims(result))
        novel_keys = frozenset(novel)
        # The gate's verdict, and only the gate's verdict, makes a claim rejected. Applied
        # BEFORE validation so a claim this round rejected cannot be reintroduced by the
        # merge that follows it -- §10's "later stage".
        rejected |= frozenset(claim_identity_key(c) for c in result.rejected_claims)
        candidate = result.graph
        reentered: Dict[str, bool] = {}
        for stage in REENTRY_STAGES:  # the order is graph_delta's, not this module's
            try:
                candidate = getattr(stages, stage)(candidate)
            except Exception as exc:
                reentered[stage] = False  # ran and failed; the rest stay not_evaluated
                raise RoundAborted(stage, reentered, mark) from exc
            reentered[stage] = True
        verdict = validate_graph_delta(graph, candidate, RoundRecord(
            detected_gap_ids=frozenset(result.detected_gap_ids),
            claims=tuple(result.claims), seen_claims=started,
            rejected_claim_keys=rejected, reentered=reentered))
        admitted = sum(1 for c in result.admitted_claims
                       if claim_identity_key(c) in novel_keys)
        # A refused round changed nothing, so it reports no growth. That is the literal
        # truth about the canonical graph, not a verdict about the claims.
        growth = len(verdict.added) - len(verdict.removed) if verdict.admissible else 0
        if verdict.admissible:
            graph = candidate
        rounds.append(RoundOutcome(len(rounds), verdict.admissible, verdict, candidate,
                                   novel_keys, admitted, growth))
        streak, previous_hash = _empty_streak(result, streak, previous_hash)
        observed = dict(round_retrieval_completed=result.retrieval_completed,
                        ladder_completed=result.ladder_completed,
                        new_admissible_claims=admitted, graph_delta=growth,
                        operation_timed_out=result.operation_timed_out,
                        identical_empty_responses=streak,
                        evidence_sources_exhausted=result.evidence_sources_exhausted,
                        defensible_core=result.defensible_core)
    return LoopOutcome(graph, decision, tuple(rounds), tuple(checkpoints), seen, rejected)


__all__ = ["Checkpoint", "LoopOutcome", "RoundAborted", "RoundContext", "RoundOutcome",
           "RoundResult", "RoundRunner", "RoundStages", "STAGE_FIELDS", "run_rag_loop"]
