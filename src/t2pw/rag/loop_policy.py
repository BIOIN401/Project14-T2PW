"""RAG loop stopping policy — PRODUCT_CONTRACT.md §10 and DECISIONS.md D-005.

POLICY ONLY: for a *supplied* loop state, "go round again, and if not, exactly why
not?", answered with one of the seven termination reasons -- D-005's six plus D-024's
``attempt_cap_reached``, which EXTENDS D-005 rather than reopening it. No loop, no I/O, no
clock read (``now``/``deadline`` are inputs), so a controller drives it
deterministically and it is testable without RAG. Leaf: no controller/pipeline/app.

``retrieval_exhausted`` is claimable ONLY when the configured ladder ACTUALLY COMPLETED
(:attr:`LoopState.ladder_completed`); a ladder cut short by the wall-clock is
``budget_exhausted``, which D-005 counts as an OPERATIONAL failure in pipeline-completion
and strict-success denominators while the other does not. Relabelling either as the other
is the misreport D-005 forbids by name.

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

TERMINATION_PRECEDENCE: Tuple[str, ...] = (
    BUDGET_EXHAUSTED, OPERATION_TIMEOUT, IDENTICAL_EMPTY_RESPONSE,
    SCIENTIFICALLY_UNRECOVERABLE, ATTEMPT_CAP_REACHED, RETRIEVAL_EXHAUSTED,
    NO_NEW_CLAIMS,
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
    def out_of_budget(self) -> bool:
        """No further round is affordable. The round bound and the time bound each
        stop the loop independently, which is what makes the loop bounded."""
        return (self.rounds_completed >= self.max_rounds
                or self.deadline - self.now <= self.next_round_reserve_seconds)


@dataclass(frozen=True)
class LoopDecision:
    """Go round again, or stop with exactly one D-005 reason plus what was counted."""

    should_continue: bool
    reason: Optional[str] = None
    also_true: Tuple[str, ...] = ()  # also held, lost on precedence; never conflated
    #: Budget accounting reads THIS, not ``also_true`` (see _conditions' carve-out).
    counts: Dict[str, Any] = field(default_factory=dict)


def _conditions(state: LoopState) -> Dict[str, bool]:
    """Which of the seven reasons are TRUE for ``state``. Several may hold at once.

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
        BUDGET_EXHAUSTED: state.out_of_budget and not exhausted,
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
