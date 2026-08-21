"""RAG loop stopping policy — PRODUCT_CONTRACT.md §10 and DECISIONS.md D-005.

POLICY ONLY: for a *supplied* loop state, "go round again, and if not, exactly why
not?", answered with one of the eight termination reasons -- D-005's six, D-024's
``attempt_cap_reached`` and C-064's ``round_cap_reached``, both of which EXTEND D-005
rather than reopening it. No loop, no I/O, no clock read (``now``/``deadline`` are
inputs), so a controller drives it deterministically and it is testable without RAG.
Leaf: no controller/pipeline/app.

``retrieval_exhausted`` is claimable ONLY when the configured ladder ACTUALLY COMPLETED
(:attr:`LoopState.ladder_completed`); a ladder cut short by the wall-clock is
``budget_exhausted``, which D-005 counts as an OPERATIONAL failure in pipeline-completion
and strict-success denominators while the other does not. Relabelling either as the other
is the misreport D-005 forbids by name.

``budget_exhausted`` is the WALL-CLOCK bound and nothing else (C-064, closing F-070).
The loop is bounded by two independent things -- a clock and a configured round ceiling
-- and only the first is a resource that can be exhausted. Folding the ceiling into the
operational reason reported a policy success as a malfunction, and did so on the DEFAULT
path, since ``controller.run_rag_loop`` defaults to ``max_rounds=1``. The two bounds are
:attr:`LoopState.time_budget_spent` and :attr:`LoopState.round_cap_reached`, and
:attr:`LoopState.out_of_budget` is still exactly their disjunction, which is what keeps
the loop bounded.

PRECEDENCE, total, fixed by :data:`TERMINATION_PRECEDENCE`; two reasons holding at once is
normal and ranking loses nothing, since losers are reported in ``also_true``.

1-2 ``budget_exhausted``, ``operation_timeout`` — an operational stop is never reported as
    a scientific one, and a spent wall-clock binds even when an operation also timed out.
    D-005 names ``budget_exhausted`` as THE operational-failure denominator.
3   ``identical_empty_response`` — an inert MECHANISM (D-005: never reissue that prompt to
    that model) says nothing about the state of the evidence base.
4   ``scientifically_unrecoverable`` — an explicit refusal, and the strongest verdict a
    stopped loop can make about the evidence itself.
5   ``attempt_cap_reached`` (D-024) — ranked BELOW every one of the four above, exactly as
    D-024's precedence rules: a real deadline, a timeout, a measured budget exhaustion or
    an explicit refusal each describe the stop better than "we ran out of tries". Ranked
    ABOVE the two below it because both of those are claims that retrieval finished, and a
    leg cut off by the cap did NOT finish: D-005 permits ``retrieval_exhausted`` "only when
    the configured ladder actually completed", so reporting it — or ``no_new_claims`` — for
    a capped leg is the conflation D-005 forbids by name.
6-7 ``retrieval_exhausted``, ``no_new_claims`` — most specific scientific verdict first, so
    a run never understates what it established.
8   ``round_cap_reached`` (C-064) — the configured ROUND ceiling was honoured. Ranked
    LAST, i.e. BELOW the two reasons D-024 deliberately ranks ``attempt_cap_reached``
    ABOVE, because the two ceilings stop the loop at different moments and D-024's
    argument for the higher rank is unavailable here. The attempt cap cuts a ladder off
    MID-FLIGHT, so ``retrieval_exhausted``/``no_new_claims`` would be FALSE if reported
    for it; the round cap only refuses to START round N+1, after round N ran to
    completion, so every verdict the finished round earned is still true and strictly
    more informative than "the operator allowed one round". It is also the reason that
    holds on EVERY loop which runs its full allowance -- and the default allowance is
    ONE round -- so a higher rank would mask exactly the reasons that tell one run from
    another. Last means RESIDUAL: it wins only when nothing else describes the stop.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

RETRIEVAL_EXHAUSTED = "retrieval_exhausted"
NO_NEW_CLAIMS = "no_new_claims"
BUDGET_EXHAUSTED = "budget_exhausted"
OPERATION_TIMEOUT = "operation_timeout"
IDENTICAL_EMPTY_RESPONSE = "identical_empty_response"
SCIENTIFICALLY_UNRECOVERABLE = "scientifically_unrecoverable"
ATTEMPT_CAP_REACHED = "attempt_cap_reached"
ROUND_CAP_REACHED = "round_cap_reached"

TERMINATION_PRECEDENCE: Tuple[str, ...] = (
    BUDGET_EXHAUSTED, OPERATION_TIMEOUT, IDENTICAL_EMPTY_RESPONSE,
    SCIENTIFICALLY_UNRECOVERABLE, ATTEMPT_CAP_REACHED, RETRIEVAL_EXHAUSTED,
    NO_NEW_CLAIMS, ROUND_CAP_REACHED,
)
TERMINATION_REASONS: FrozenSet[str] = frozenset(TERMINATION_PRECEDENCE)


def _field(value: Any) -> Tuple[str, ...]:
    """One chemistry field, order-free, one key per chemistry (§10, or no convergence):
    internal whitespace collapses so ``"acetyl  CoA"`` IS ``"acetyl CoA"``, and a scalar
    ``str``/``bytes`` is ONE element, so ``"abc"`` never collides with ``list("abc")``."""
    items = () if not value else (value,) if isinstance(value, (str, bytes)) else value
    return tuple(sorted(" ".join(str(i).split()).casefold() for i in items))


def claim_identity_key(claim: Any) -> str:
    """Verdict-, gap- and name-independent identity of ONE claim: its chemistry.

    Prefers the audited ``RagReactionCandidate.claim_identity()`` when the object has it
    (synonyms collapsed via ``canonical_name``), else the same (inputs, outputs, enzymes,
    reversible) shape for mappings. Name and gap are excluded exactly as at
    ``admission.py:1485``: the same chemistry under other wording or against another gap
    is the SAME claim and must never read as new.
    """
    audited = getattr(claim, "claim_identity", None)
    if callable(audited):
        return hashlib.sha1(repr(audited()).encode("utf-8")).hexdigest()[:16]
    read = claim.get if isinstance(claim, dict) else lambda k: getattr(claim, k, None)
    identity = (*(_field(read(f)) for f in ("inputs", "outputs", "enzymes")),
                bool(read("reversible")))
    return hashlib.sha1(repr(identity).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SeenClaims:
    """Every claim key the loop has EVER seen — admitted, rejected or discarded.

    PRODUCT_CONTRACT §10: deduplication is against ALL claims ever seen, not only the
    admitted ones, or judge-rejected claims recur every round and the loop never
    converges. Keys are :func:`claim_identity_key`, so a re-offer under another gap or
    another paper's wording is recognised; a later stage refuses any key already here.
    """

    keys: FrozenSet[str] = frozenset()

    def observe(self, claims: Iterable[Any]) -> Tuple["SeenClaims", Tuple[str, ...]]:
        """Return (history including ``claims``, the keys never seen before). Call with
        EVERY claim of the round whatever the judge said, or the rejected ones recur."""
        keys = set(self.keys)
        novel: List[str] = []
        for claim in claims:
            key = claim_identity_key(claim)
            if key not in keys:
                keys.add(key)
                novel.append(key)
        return SeenClaims(frozenset(keys)), tuple(novel)


@dataclass(frozen=True)
class LoopState:
    """What the controller observed — the only input to :func:`decide`.

    ``now`` is the caller's monotonic reading against the one per-leg monotonic
    ``deadline`` (D-005); ``next_round_reserve_seconds`` is what one more round plus the
    finalization reserve needs. ``ladder_completed`` is true only when every configured
    rung ACTUALLY ran; ``new_admissible_claims`` counts claims novel against
    :class:`SeenClaims` AND admitted this round. Unpopulated defaults stop, never spin.
    """

    rounds_completed: int = 0
    max_rounds: int = 1
    now: float = 0.0
    deadline: float = 0.0
    next_round_reserve_seconds: float = 0.0
    ladder_completed: bool = False
    round_retrieval_completed: bool = False
    new_admissible_claims: int = 0
    graph_delta: int = 0
    operation_timed_out: bool = False
    identical_empty_responses: int = 0
    evidence_sources_exhausted: bool = False
    defensible_core: bool = True
    #: D-024. The configured attempt ceiling is spent and the operation did not
    #: succeed. Supplied by the controller because the count lives in the ladder, not
    #: here; ``False`` by default, so an unpopulated state never invents the reason.
    attempt_cap_reached: bool = False

    @property
    def round_cap_reached(self) -> bool:
        """The configured round ceiling is spent: ``max_rounds`` rounds have run.

        A POLICY bound, not a resource one. Nothing was exhausted -- the loop was told
        how many rounds it may have and it took them all, which is a configuration
        being honoured. Reported as ``round_cap_reached``, never as
        ``budget_exhausted`` (C-064, F-070).
        """
        return self.rounds_completed >= self.max_rounds

    @property
    def time_budget_spent(self) -> bool:
        """The wall-clock refuses another round: what is left will not cover one more
        round plus the finalization reserve. THE resource bound, and the only thing
        ``budget_exhausted`` -- D-005's operational-failure denominator -- reports."""
        return self.deadline - self.now <= self.next_round_reserve_seconds

    @property
    def out_of_budget(self) -> bool:
        """No further round is affordable. The round bound and the time bound each
        stop the loop independently, which is what makes the loop bounded.

        Kept as the disjunction of the two named bounds above: splitting the REASON
        did not split the boundedness invariant, and this stays the single place that
        states it. ``controller.run_rag_loop`` asks this question and no other.
        """
        return self.round_cap_reached or self.time_budget_spent


@dataclass(frozen=True)
class LoopDecision:
    """Go round again, or stop with exactly one D-005 reason plus what was counted."""

    should_continue: bool
    reason: Optional[str] = None
    also_true: Tuple[str, ...] = ()  # also held, lost on precedence; never conflated
    #: Budget accounting reads THIS, not ``also_true`` (see _conditions' carve-out).
    counts: Dict[str, Any] = field(default_factory=dict)


def _conditions(state: LoopState) -> Dict[str, bool]:
    """Which of the eight reasons are TRUE for ``state``. Several may hold at once.

    EVERY member of :data:`TERMINATION_PRECEDENCE` gets a key here, because
    :func:`decide` indexes this mapping by precedence member: a reason added to the
    tuple without a key here would be a ``KeyError``, not a silent omission.
    """
    exhausted = state.ladder_completed and state.new_admissible_claims <= 0
    return {
        # budget_exhausted asserts D-005's "another recovery step MIGHT HAVE HELPED": a
        # rung never ran, or the round produced claims a further round would integrate.
        # Ladder complete with nothing new means no such step exists, and the honest
        # reason is retrieval_exhausted — unreachable while the ladder did not complete.
        # C-064: the TIME bound alone. The round bound moved to ROUND_CAP_REACHED below
        # carrying this exact ``not exhausted`` carve-out with it, so the two together
        # are still precisely the one predicate they replaced -- no state gained a stop
        # condition and none lost one, and the exemption is carried, never widened.
        BUDGET_EXHAUSTED: state.time_budget_spent and not exhausted,
        OPERATION_TIMEOUT: state.operation_timed_out,
        IDENTICAL_EMPTY_RESPONSE: state.identical_empty_responses >= 2,
        SCIENTIFICALLY_UNRECOVERABLE: (
            state.evidence_sources_exhausted and not state.defensible_core),
        # D-024. Guarded on ``not ladder_completed`` because the reason asserts the
        # CAP is what ended processing: a ladder that ran to completion was ended by
        # its own completion, and the honest reason for that is the more specific
        # retrieval_exhausted below. The guard is what keeps the two mutually
        # exclusive rather than leaving the ranking to decide a contradiction.
        ATTEMPT_CAP_REACHED: state.attempt_cap_reached and not state.ladder_completed,
        RETRIEVAL_EXHAUSTED: exhausted,
        # The round-level observation while rungs remain; a completed ladder reports
        # the stronger, more specific reason above instead.
        NO_NEW_CLAIMS: (state.round_retrieval_completed and not state.ladder_completed
                        and state.new_admissible_claims <= 0 and state.graph_delta <= 0),
        # C-064. Guarded on ``not exhausted`` for the same reason budget_exhausted is,
        # and with the SAME predicate the round bound already carried inside it: a
        # ceiling denied the loop nothing when the ladder finished and produced nothing
        # new, so the honest reason there is still retrieval_exhausted. Like D-024's
        # guard, this keeps the two mutually exclusive rather than leaving the ranking
        # to decide a contradiction.
        # THE PREDICATE IS FORCED, NOT MERELY TIDY. Copying D-024's ``not
        # ladder_completed`` verbatim would LOSE BOUNDEDNESS: a state with the ceiling
        # reached, the clock fine, ladder_completed=True and new_admissible_claims > 0
        # is NOT ``exhausted``, so retrieval_exhausted does not fire either, and the
        # loop would yield no reason at all and spin. Carry ``not exhausted``; do not
        # substitute the other guard.
        ROUND_CAP_REACHED: state.round_cap_reached and not exhausted,
    }


def decide(state: LoopState) -> LoopDecision:
    """Continue, or stop with the highest-precedence reason that holds.

    Pure and total: no I/O, no clock, no mutation; a state in which no condition holds
    continues, and ``out_of_budget`` always yields one, so the loop stays bounded.
    """
    held = _conditions(state)
    reasons = tuple(reason for reason in TERMINATION_PRECEDENCE if held[reason])
    counts = dict(asdict(state), seconds_remaining=state.deadline - state.now)
    return LoopDecision(not reasons, reasons[0] if reasons else None,
                        reasons[1:], counts)
