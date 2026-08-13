"""One monotonic per-leg deadline with a finalization reserve. D-005.

A leg is one *(paper, mode)* pair. Before this module the wall clock was spent
optimistically: a stage started whatever the remaining budget was, and when the
parent's stopwatch expired the child was killed mid-write, leaving no checkpoint,
no diagnostics and no classification. The night then reported a paper that had
"failed" when what happened is that it was cut off. ``MASTER_PLAN.md`` 1.5 records
that shape as the most likely mechanism behind PMC12444477 timing out in both modes.

**A budget readable before it is spent.** :class:`LegDeadline` is one monotonic
clock per leg, shared across every stage, answering ``elapsed`` and ``remaining``
at any point. ``time.monotonic`` and not ``time.time``: an NTP correction or a DST
step must not be able to hand a leg an extra hour.

**A reserve subtracted first, not last.** ``remaining`` is not spendable. Its last
stretch belongs to checkpoint persistence, validation, status classification and
diagnostic-artifact writing -- the work that turns a stopped leg into a *reported*
leg. :meth:`LegDeadline.admit` refuses a call whose own configured maximum plus
that reserve does not fit, so the leg stops with its evidence written instead of
being killed while writing it. Calls are priced by C-014's
:func:`t2pw.llm.client.worst_case_call_seconds`, an upper bound on purpose.

**Six termination reasons, never conflated.** ``budget_exhausted`` and
``operation_timeout`` are OPERATIONAL facts -- not ``no_new_claims``, not
``identical_empty_response``, not ``scientifically_unrecoverable``, and above all
not ``retrieval_exhausted``, which :func:`claim_retrieval_exhausted` refuses to
issue unless the configured ladder actually completed. A leg that ran out of clock
has said nothing about the biology in the paper, and every denominator downstream
has to see that. **A timeout is an operational fact, never a biological verdict.**

What this module deliberately does NOT contain: the extraction escalation ladder.
D-005 fixes the order -- this is the budget and checkpoint infrastructure, C-042 is
what spends it. There are no retries here, unconditional or otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# D-005 fixed values. These do not move.
# ---------------------------------------------------------------------------
#: Per-leg wall-clock ceiling enforced by the batch parent. Mirrors
#: ``batch.runner.DEFAULT_PAPER_TIMEOUT``; the two are pinned equal by test.
LEG_TIMEOUT_SECONDS = 3600.0

#: Head start the child gets over the parent's kill so it can finalize rather than
#: be killed one second before finishing. Mirrors ``batch.runner._CHILD_GRACE``.
PARENT_CHILD_GRACE_SECONDS = 120.0

#: 3600 - 120. Stated here so the arithmetic has one authority instead of being
#: re-derived at each call site.
CHILD_DEADLINE_SECONDS = LEG_TIMEOUT_SECONDS - PARENT_CHILD_GRACE_SECONDS

#: Floor on a child deadline, preserving the batch runner's long-standing one.
MIN_CHILD_DEADLINE_SECONDS = 60.0

#: Default finalization reserve: checkpoint persistence, validation, status
#: classification and diagnostic-artifact writing -- the work between "the last
#: call returned" and "the leg is on disk and classified". Configurable per
#: :class:`LegDeadline`, and D-005 expects it calibrated from recorded stage
#: runtimes later; until then it is the grace period, the one number here that has
#: actually been sized against finalization.
DEFAULT_FINALIZATION_RESERVE_SECONDS = PARENT_CHILD_GRACE_SECONDS

# ---------------------------------------------------------------------------
# The six termination reasons. Never conflated, never invented.
# ---------------------------------------------------------------------------
#: The configured retrieval ladder ran to completion and found nothing further.
RETRIEVAL_EXHAUSTED = "retrieval_exhausted"
#: A further pass returned nothing the graph did not already hold.
NO_NEW_CLAIMS = "no_new_claims"
#: The leg's wall-clock budget ran out with work still to do. OPERATIONAL.
BUDGET_EXHAUSTED = "budget_exhausted"
#: One operation exceeded its own configured maximum duration. OPERATIONAL.
OPERATION_TIMEOUT = "operation_timeout"
#: Two attempts returned byte-identical empty responses; the strategy is inert.
IDENTICAL_EMPTY_RESPONSE = "identical_empty_response"
#: The source does not support a defensible pathway. A scientific statement.
SCIENTIFICALLY_UNRECOVERABLE = "scientifically_unrecoverable"

TERMINATION_REASONS: Tuple[str, ...] = (
    RETRIEVAL_EXHAUSTED,
    NO_NEW_CLAIMS,
    BUDGET_EXHAUSTED,
    OPERATION_TIMEOUT,
    IDENTICAL_EMPTY_RESPONSE,
    SCIENTIFICALLY_UNRECOVERABLE,
)

#: D-005's denominator rule: these two count as failures of pipeline completion
#: and of end-to-end strict success, and are never relabelled semantic failure,
#: scientific insufficiency, or retrieval exhaustion. A closed set precisely so no
#: caller can widen it.
OPERATIONAL_TERMINATION_REASONS = frozenset({BUDGET_EXHAUSTED, OPERATION_TIMEOUT})

#: What ``driver._run_app`` says when a leg's whole wall clock is already spent
#: before an interaction begins -- the one signal separating budget exhaustion
#: from an interaction that overran its slice. ``tests/test_deadline_leg_timeout.py``
#: pins the two strings together so they cannot drift apart silently.
BUDGET_SPENT_MARKER = "whole-run budget of"


class BudgetExhausted(RuntimeError):
    """The remaining budget could not cover the next call plus the reserve.

    Carries the D-005 reason so a handler never re-derives it from prose, and so
    it cannot be caught and re-reported as a semantic failure.
    """

    terminal_reason = BUDGET_EXHAUSTED

    def __init__(self, message: str, admission: Optional["Admission"] = None) -> None:
        super().__init__(message)
        self.admission = admission


def require_reason(reason: Any) -> str:
    """Return ``reason`` if it is one of the six, else raise.

    A closed vocabulary is what stops the conflation D-005 forbids: an invented or
    misspelled reason fails loudly here rather than reaching a denominator as an
    unrecognised string some later reader buckets by hand.
    """

    token = str(reason or "").strip()
    if token not in TERMINATION_REASONS:
        raise ValueError(
            f"{reason!r} is not one of D-005's six termination reasons: "
            + ", ".join(TERMINATION_REASONS)
        )
    return token


def is_operational(reason: Any) -> bool:
    """Whether ``reason`` is an operational failure rather than a finding."""

    return str(reason or "").strip() in OPERATIONAL_TERMINATION_REASONS


def claim_retrieval_exhausted(
    *, rungs_configured: int, rungs_completed: int, ladder_completed: bool
) -> str:
    """``retrieval_exhausted``, but only when the ladder actually completed.

    Claiming it because the clock ran out would turn an operational fact into "we
    looked everywhere and there was nothing", the most misleading relabelling
    available here. A short ladder is a budget outcome, reported as
    :data:`BUDGET_EXHAUSTED` by the caller.
    """

    configured, completed = int(rungs_configured), int(rungs_completed)
    if not ladder_completed or configured <= 0 or completed < configured:
        raise ValueError(
            "retrieval_exhausted may be claimed only when the configured ladder "
            f"completed; {completed}/{configured} rung(s) ran "
            f"(ladder_completed={bool(ladder_completed)})"
        )
    return RETRIEVAL_EXHAUSTED


# ---------------------------------------------------------------------------
# Per-leg timeout: explicit overrides only.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LegTimeout:
    """The wall-clock ceiling for one leg, plus why it is that number."""

    seconds: float
    default_seconds: float = LEG_TIMEOUT_SECONDS
    overridden: bool = False
    reason: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """The run-manifest record. Present on every leg, override or not."""

        record: Dict[str, Any] = {
            "leg_timeout_seconds": round(float(self.seconds), 2),
            "leg_timeout_default_seconds": round(float(self.default_seconds), 2),
            "leg_timeout_overridden": bool(self.overridden),
        }
        if self.overridden:
            record["leg_timeout_override_reason"] = self.reason
            record["leg_timeout_override_source"] = self.source
        return record


def resolve_leg_timeout(
    override: Optional[float] = None,
    *,
    reason: str = "",
    source: str = "",
    default: float = LEG_TIMEOUT_SECONDS,
) -> LegTimeout:
    """The leg ceiling, refusing any override that is not explained.

    D-005: a per-leg override must be explicit and recorded in the run manifest,
    with no silent extension of difficult benchmark legs. "Explicit" is enforced
    here -- an override with no reason or no source raises -- and "recorded" is
    :meth:`LegTimeout.to_dict`, which the manifest row carries.
    """

    if override is None:
        return LegTimeout(seconds=float(default), default_seconds=float(default))
    seconds = float(override)
    if seconds <= 0:
        raise ValueError(f"leg timeout override must be > 0 seconds, got {override!r}")
    if not str(reason).strip() or not str(source).strip():
        raise ValueError(
            "a per-leg timeout override must be explicit: pass both reason= and "
            f"source= so the manifest records why this leg is not {float(default):.0f}s"
        )
    return LegTimeout(
        seconds=seconds,
        default_seconds=float(default),
        overridden=seconds != float(default),
        reason=str(reason).strip(),
        source=str(source).strip(),
    )


def child_deadline_seconds(
    leg_timeout: float = LEG_TIMEOUT_SECONDS,
    *,
    grace: float = PARENT_CHILD_GRACE_SECONDS,
    floor: float = MIN_CHILD_DEADLINE_SECONDS,
) -> float:
    """``leg_timeout - grace``, floored. 3600 - 120 = 3480 by default."""

    return max(float(floor), float(leg_timeout) - float(grace))


# ---------------------------------------------------------------------------
# Admission and checkpoints.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Admission:
    """Whether one call may start, and the arithmetic that decided it."""

    allowed: bool
    label: str = ""
    call_seconds: float = 0.0
    reserve_seconds: float = 0.0
    remaining_seconds: float = 0.0
    reason: str = ""

    @property
    def required_seconds(self) -> float:
        return float(self.call_seconds) + float(self.reserve_seconds)

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "label": self.label,
            "allowed": bool(self.allowed),
            "call_seconds": round(float(self.call_seconds), 2),
            "reserve_seconds": round(float(self.reserve_seconds), 2),
            "required_seconds": round(self.required_seconds, 2),
            "remaining_seconds": round(float(self.remaining_seconds), 2),
        }
        if self.reason:
            record["termination_reason"] = self.reason
            record["operational_failure"] = is_operational(self.reason)
        return record

    def raise_if_refused(self) -> None:
        """D-005: insufficient budget for the next rung -> stop, record the reason."""

        if self.allowed:
            return
        raise BudgetExhausted(
            f"{self.label or 'call'} needs {self.required_seconds:.0f}s "
            f"({float(self.call_seconds):.0f}s of call + "
            f"{float(self.reserve_seconds):.0f}s of finalization reserve) but only "
            f"{float(self.remaining_seconds):.0f}s of the leg budget remains",
            self,
        )


@dataclass(frozen=True)
class Checkpoint:
    """State captured *before* a long call, with the budget at that instant."""

    label: str
    sequence: int = 0
    stage: str = ""
    elapsed_seconds: float = 0.0
    remaining_seconds: float = 0.0
    reserve_seconds: float = 0.0
    state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "sequence": int(self.sequence),
            "stage": self.stage,
            "elapsed_seconds": round(float(self.elapsed_seconds), 2),
            "remaining_seconds": round(float(self.remaining_seconds), 2),
            "reserve_seconds": round(float(self.reserve_seconds), 2),
            "state": dict(self.state),
        }


def price_llm_call(**kwargs: Any) -> float:
    """Upper bound on one LLM call, from C-014's ``worst_case_call_seconds``.

    Imported lazily on purpose: ``t2pw.llm.client`` raises at IMPORT time when
    ``OPENROUTER_API_KEY`` is absent or malformed, and a budget module needing
    provider credentials to import would be unusable in exactly the offline
    contexts that need it. An unknown price is infinite, never zero: an admission
    check must fail closed.
    """

    try:
        from t2pw.llm.client import worst_case_call_seconds
    except BaseException:  # noqa: BLE001 - the client raises RuntimeError at import
        return float("inf")
    return float(worst_case_call_seconds(**kwargs))


class LegDeadline:
    """One monotonic wall-clock budget, shared by every stage of one leg.

    Constructed once per leg and passed down, never re-created per stage: a
    per-stage clock cannot say how much of the *leg* is left, which is the only
    number an admission decision can be made against.

    Pass ``leg_timeout=resolve_leg_timeout(...)`` in production -- that is where
    "an override must be explicit" is enforced. A bare ``total_seconds`` is the
    convenience form; it records the ceiling it was handed and claims no override,
    so it can never smuggle an unexplained extension into a manifest.
    """

    def __init__(
        self,
        total_seconds: float = LEG_TIMEOUT_SECONDS,
        *,
        reserve_seconds: float = DEFAULT_FINALIZATION_RESERVE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        leg_timeout: Optional[LegTimeout] = None,
    ) -> None:
        self._clock = clock
        self.started = float(clock())
        self.leg_timeout = leg_timeout or LegTimeout(
            seconds=float(total_seconds), default_seconds=float(total_seconds)
        )
        self.total = max(0.0, float(self.leg_timeout.seconds))
        self.reserve = max(0.0, float(reserve_seconds))
        self._checkpoints = 0

    @property
    def elapsed(self) -> float:
        """Seconds since the leg started. Monotonic, so never negative."""

        return max(0.0, float(self._clock()) - self.started)

    @property
    def remaining(self) -> float:
        """Seconds of leg budget left. Negative once the leg has overrun."""

        return self.total - self.elapsed

    @property
    def spendable(self) -> float:
        """What may be spent on work: ``remaining`` minus the reserve."""

        return self.remaining - self.reserve

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    @property
    def reserve_breached(self) -> bool:
        """True once only the finalization reserve is left: stop and finalize."""

        return self.spendable <= 0.0

    def can_start(self, call_seconds: float) -> bool:
        """D-005's precondition: call maximum + reserve must fit in ``remaining``."""

        return float(call_seconds) + self.reserve <= self.remaining

    def admit(self, call_seconds: float, *, label: str = "") -> Admission:
        """Decide whether one call may start, recording the arithmetic.

        A refusal is :data:`BUDGET_EXHAUSTED`, an operational fact. It never says
        the retrieval ladder finished or the source was scientifically
        unrecoverable, and nothing here may turn it into either.
        """

        remaining = self.remaining
        allowed = float(call_seconds) + self.reserve <= remaining
        return Admission(
            allowed=allowed,
            label=label,
            call_seconds=float(call_seconds),
            reserve_seconds=self.reserve,
            remaining_seconds=remaining,
            reason="" if allowed else BUDGET_EXHAUSTED,
        )

    def checkpoint(
        self,
        label: str,
        *,
        stage: str = "",
        state: Optional[Dict[str, Any]] = None,
        persist: Optional[Callable[[Checkpoint], Any]] = None,
    ) -> Checkpoint:
        """Capture the leg's state and budget at this instant.

        ``persist`` is whatever actually writes it, and a failure to persist is
        swallowed: a checkpoint exists to make a killed leg reportable, so it must
        never be the thing that kills the leg. The record comes back complete
        either way, so a caller that wants to know can check.
        """

        self._checkpoints += 1
        record = Checkpoint(
            label=str(label),
            sequence=self._checkpoints,
            stage=str(stage),
            elapsed_seconds=self.elapsed,
            remaining_seconds=self.remaining,
            reserve_seconds=self.reserve,
            state=dict(state or {}),
        )
        if persist is not None:
            try:
                persist(record)
            except Exception:  # noqa: BLE001 - see docstring
                pass
        return record

    def before_long_call(
        self,
        *,
        label: str,
        call_seconds: Optional[float] = None,
        stage: str = "",
        state: Optional[Dict[str, Any]] = None,
        persist: Optional[Callable[[Checkpoint], Any]] = None,
        price: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Checkpoint, Admission]:
        """Checkpoint first, then decide. The order is D-005's, not a preference.

        Checkpointing after the admission check would leave the refused case
        without a record, and checkpointing after the *call* would leave the killed
        case without one -- the state that produced legs with nothing on disk. So
        the state is captured before the call is even priced. ``call_seconds``
        defaults to :func:`price_llm_call`, an upper bound.
        """

        record = self.checkpoint(label, stage=stage, state=state, persist=persist)
        seconds = price_llm_call(**(price or {})) if call_seconds is None else float(call_seconds)
        return record, self.admit(seconds, label=label)

    def snapshot(self) -> Dict[str, Any]:
        """The budget record a manifest row or diagnostic artifact carries."""

        record = dict(self.leg_timeout.to_dict())
        record.update(
            {
                "elapsed_seconds": round(self.elapsed, 2),
                "remaining_seconds": round(self.remaining, 2),
                "finalization_reserve_seconds": round(self.reserve, 2),
                "checkpoints": self._checkpoints,
            }
        )
        return record


# ---------------------------------------------------------------------------
# Classification. Two consumers, one vocabulary.
# ---------------------------------------------------------------------------
def classify_interaction_timeout(detail: Any = "", *, explicit: str = "") -> str:
    """Why an in-process interaction stopped: budget exhaustion or an overrun.

    ``explicit`` wins when a caller already knows. Otherwise the driver's own
    budget check is the evidence: it emits :data:`BUDGET_SPENT_MARKER` when the
    leg's whole wall clock was gone before the interaction could begin, which is
    budget exhaustion; anything else is one operation overrunning its configured
    maximum. Both answers are operational, so the D-005 denominator rule holds
    either way -- neither can be read as a semantic or scientific verdict, and
    neither is ever ``retrieval_exhausted``.
    """

    if explicit:
        return require_reason(explicit)
    text = detail if isinstance(detail, str) else ("" if detail is None else str(detail))
    return BUDGET_EXHAUSTED if BUDGET_SPENT_MARKER in text else OPERATION_TIMEOUT


def classify_child_kill(
    *, elapsed_seconds: float, leg_timeout_seconds: float, tolerance: float = 1.0
) -> str:
    """Why the batch parent killed a child: it spent the leg, or overran a shorter bound.

    Reaching the leg ceiling is :data:`BUDGET_EXHAUSTED`; being killed while leg
    budget remained means a tighter per-operation bound applied, which is
    :data:`OPERATION_TIMEOUT`. ``tolerance`` absorbs the sub-second gap between the
    parent's stopwatch and the timeout it was given.
    """

    if float(elapsed_seconds) + float(tolerance) >= float(leg_timeout_seconds):
        return BUDGET_EXHAUSTED
    return OPERATION_TIMEOUT


def budget_record(
    *,
    elapsed_seconds: float,
    leg_timeout: Any = LEG_TIMEOUT_SECONDS,
    reserve_seconds: float = DEFAULT_FINALIZATION_RESERVE_SECONDS,
) -> Dict[str, Any]:
    """Elapsed and remaining budget, for a row written without a live deadline.

    ``leg_timeout`` accepts a :class:`LegTimeout` -- so an explicit override travels
    into the manifest with its reason attached -- or a plain number, recorded as the
    unmodified ceiling.
    """

    ceiling = leg_timeout if isinstance(leg_timeout, LegTimeout) else LegTimeout(
        seconds=float(leg_timeout), default_seconds=float(leg_timeout)
    )
    elapsed = float(elapsed_seconds)
    record = dict(ceiling.to_dict())
    record.update(
        {
            "elapsed_seconds": round(elapsed, 2),
            "remaining_seconds": round(float(ceiling.seconds) - elapsed, 2),
            "finalization_reserve_seconds": round(float(reserve_seconds), 2),
            "child_deadline_seconds": round(child_deadline_seconds(float(ceiling.seconds)), 2),
        }
    )
    return record
