"""RAG loop stopping policy — PRODUCT_CONTRACT.md §10 and DECISIONS.md D-005.

POLICY ONLY: for a *supplied* loop state, "go round again, and if not, exactly why
not?", answered with one of the six D-005 termination reasons. No loop, no I/O, no
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
4-6 ``scientifically_unrecoverable``, ``retrieval_exhausted``, ``no_new_claims`` — most
    specific scientific verdict first, so a run never understates what it established.
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

TERMINATION_PRECEDENCE: Tuple[str, ...] = (
    BUDGET_EXHAUSTED, OPERATION_TIMEOUT, IDENTICAL_EMPTY_RESPONSE,
    SCIENTIFICALLY_UNRECOVERABLE, RETRIEVAL_EXHAUSTED, NO_NEW_CLAIMS,
)
TERMINATION_REASONS: FrozenSet[str] = frozenset(TERMINATION_PRECEDENCE)


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
    identity = (*(tuple(sorted(str(i).strip().casefold() for i in read(f) or ()))
                  for f in ("inputs", "outputs", "enzymes")), bool(read("reversible")))
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
    """Which of the six reasons are TRUE for ``state``. Several may hold at once."""
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
