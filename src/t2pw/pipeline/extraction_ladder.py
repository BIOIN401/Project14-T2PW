"""Stage-1 extraction escalation ladder: three rungs, budget-gated, never inert.

C-042. ``PRODUCT_CONTRACT.md`` § 9 allows Stage-1 extraction **at most three
total model attempts, including the first**: 1 normal, 2 structurally-empty
repair, 3 a materially different strategy -- narrower section-based extraction
or an alternate model -- *only if budget remains*.

WHAT WAS WRONG. The empty-extraction retry existed and did nothing. Measured on
``runs_verify/2026-08-04_1754/papers/PMC12782028/strict``: two boundaries,
``attempts:1`` then ``attempts:2``, **identical** ``response_hash
sha256:44136fa355b3678a`` and **different** ``request_hash``. The prompt was
tweaked, the same model was asked again, and it returned the same nothing. A
tweaked prompt to the same model is not an escalation, and the same paper's
research leg passed, so the paper was fine and the strategy was inert.

WHAT THIS MODULE IS. The *policy* half of the ladder: how many model calls have
been issued, which prompt went to which model, whether two of them came back
byte-identically empty, and whether the next rung may start at all. It issues no
call and builds no prompt -- :func:`t2pw.pipeline.pipeline._run_json_stage` does
both -- so every decision here is testable offline with no provider.

THE CAP COUNTS EVERY MODEL CALL, not every loop iteration. The localized JSON
repair draw is a model call and sits inside the cap; before C-042 it sat outside
``max_attempts`` entirely, so an extraction configured for 2 attempts could
issue 2 + 2 x ``MAX_JSON_REPAIR_ATTEMPTS`` = up to 8 calls against a § 9 ceiling
of three.

FAIL CLOSED, AND ONLY WHERE THAT IS HONEST. Rung 3 is the rung § 9 conditions on
"only if budget remains", so it is the rung that refuses to run when budget,
model identity or attempt history cannot be established -- with the cause
recorded, never a silent fall-through. Rungs 1 and 2 and the localized repair
are pre-existing behaviour and are **not** newly gated: with no
:class:`~t2pw.pipeline.deadline.LegDeadline` in scope they run exactly as they
did before this module existed. Refusing them on an undeterminable budget would
turn a run that works today into a run that stops, which is not a correction.

A SKIP CAUSE IS NOT A TERMINATION REASON. :data:`SKIP_CAUSES` says why one rung
did not start; the six D-005 termination reasons say why the *leg* stopped. They
are recorded in different fields on purpose -- ``budget_refused`` and
``identical_prompt_same_model`` describe different events even when both end a
run, and § 9 forbids conflating what the leg reports.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from t2pw.pipeline.deadline import (
    BUDGET_EXHAUSTED,
    IDENTICAL_EMPTY_RESPONSE,
    OPERATION_TIMEOUT,
    Admission,
    BudgetExhausted,
    Checkpoint,
    LegDeadline,
    price_llm_call,
    require_reason,
)

__all__ = [
    # Re-exported from ``deadline`` so a caller that only speaks to the ladder
    # does not have to import the budget module to name what it can raise.
    "BudgetExhausted",
    "LegDeadline",
    "BUDGET_EXHAUSTED",
    "IDENTICAL_EMPTY_RESPONSE",
    "OPERATION_TIMEOUT",
    "MAX_TOTAL_MODEL_ATTEMPTS",
    "ALTERNATE_MODEL_ENV_VAR",
    "RUNG_NORMAL",
    "RUNG_EMPTY_REPAIR",
    "RUNG_DIFFERENT_STRATEGY",
    "RUNG_JSON_REPAIR",
    "RUNGS",
    "SKIP_ATTEMPT_CAP",
    "SKIP_BUDGET_REFUSED",
    "SKIP_BUDGET_INDETERMINATE",
    "SKIP_IDENTICAL_PROMPT",
    "SKIP_NOT_MATERIALLY_DIFFERENT",
    "SKIP_MODEL_IDENTITY_UNKNOWN",
    "SKIP_CAUSES",
    "SCOPE_FULL_TEXT",
    "ModelAttempt",
    "RungDecision",
    "ExtractionLadder",
    "alternate_model_env_var",
    "is_operation_timeout",
    "current_leg_deadline",
    "activate_leg_deadline",
    "deactivate_leg_deadline",
]

#: § 9's safety ceiling. "At most three total model attempts, including the
#: first" -- and a ceiling, not a quota: a successful first draw issues one call
#: and stops.
MAX_TOTAL_MODEL_ATTEMPTS = 3

#: Env var naming the alternate model for rung 3. Consulted, never required:
#: when it is unset the third rung's material difference has to come from
#: narrower section scoping instead, and when neither is available rung 3 does
#: not run. A fallback model that silently resolves to the model that already
#: returned nothing twice would re-create the exact defect being fixed.
ALTERNATE_MODEL_ENV_VAR = "OPENROUTER_EXTRACTION_FALLBACK_MODEL"

# --- the rungs -------------------------------------------------------------
RUNG_NORMAL = "normal"
RUNG_EMPTY_REPAIR = "structurally_empty_repair"
RUNG_DIFFERENT_STRATEGY = "materially_different_strategy"
#: The localized JSON syntax repair draw. Not one of § 9's three named rungs but
#: unquestionably a model attempt, so it is counted against the same ceiling.
RUNG_JSON_REPAIR = "localized_json_repair"

RUNGS: Tuple[str, ...] = (
    RUNG_NORMAL,
    RUNG_EMPTY_REPAIR,
    RUNG_DIFFERENT_STRATEGY,
    RUNG_JSON_REPAIR,
)

# --- why a rung did not start ----------------------------------------------
#: The § 9 ceiling of three model attempts is already spent.
SKIP_ATTEMPT_CAP = "attempt_cap_reached"
#: A live deadline priced the call and refused it. Pairs with BUDGET_EXHAUSTED.
SKIP_BUDGET_REFUSED = "budget_refused"
#: No leg deadline was in scope, so "budget remains" could not be established.
#: Distinct from a refusal: nobody said there was no budget, nobody could say.
SKIP_BUDGET_INDETERMINATE = "budget_indeterminate"
#: This exact prompt already went to this exact model, and two attempts came
#: back identically empty. § 9: it must never go a third time.
SKIP_IDENTICAL_PROMPT = "identical_prompt_same_model"
#: The proposed third rung was not observably different from what already ran --
#: same request hash, or same model with no narrower scope. A prompt tweak is
#: not a rung, so it does not get to be issued as one.
SKIP_NOT_MATERIALLY_DIFFERENT = "strategy_not_materially_different"
#: The model that would serve the rung could not be named.
SKIP_MODEL_IDENTITY_UNKNOWN = "model_identity_unknown"

SKIP_CAUSES: Tuple[str, ...] = (
    SKIP_ATTEMPT_CAP,
    SKIP_BUDGET_REFUSED,
    SKIP_BUDGET_INDETERMINATE,
    SKIP_IDENTICAL_PROMPT,
    SKIP_NOT_MATERIALLY_DIFFERENT,
    SKIP_MODEL_IDENTITY_UNKNOWN,
)

#: The scope label of an unnarrowed extraction.
SCOPE_FULL_TEXT = "full_text"


def require_skip_cause(cause: Any) -> str:
    """``cause`` if it is one of :data:`SKIP_CAUSES`, else raise.

    Same closed-vocabulary argument as ``deadline.require_reason``: an invented
    skip cause reaches a report as a string somebody has to bucket by hand.
    """

    token = str(cause or "").strip()
    if token not in SKIP_CAUSES:
        raise ValueError(
            f"{cause!r} is not one of the ladder's skip causes: " + ", ".join(SKIP_CAUSES)
        )
    return token


def alternate_model_env_var() -> str:
    """:data:`ALTERNATE_MODEL_ENV_VAR` if it names a model, else ``""``.

    Returns the variable *name* because that is what ``chat_detailed`` takes.
    An empty or whitespace value counts as unset: ``_resolve_model`` would fall
    through it to the same model as rung 1, which is the thing rung 3 exists not
    to be.
    """

    return ALTERNATE_MODEL_ENV_VAR if (os.getenv(ALTERNATE_MODEL_ENV_VAR) or "").strip() else ""


def is_operation_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` is C-014's per-operation timeout.

    Imported lazily: ``t2pw.llm.client`` raises at IMPORT time without provider
    credentials, and the ladder has to be diagnosable offline. A client that
    cannot be imported cannot have raised this, so ``False`` is correct and not
    a guess.
    """

    try:
        from t2pw.llm.client import LLMOperationTimeout
    except BaseException:  # noqa: BLE001 - the client raises RuntimeError at import
        return False
    return isinstance(exc, LLMOperationTimeout)


# ---------------------------------------------------------------------------
# The ambient leg deadline.
# ---------------------------------------------------------------------------
#: Set by whoever owns the leg, read by any stage beneath it. A ContextVar and
#: not a module global for the same reason ``extraction_diagnostics`` uses one:
#: the batch driver runs legs in threads and two legs must never share a clock.
_ACTIVE_DEADLINE: ContextVar[Optional[LegDeadline]] = ContextVar(
    "t2pw_leg_deadline", default=None
)


def current_leg_deadline() -> Optional[LegDeadline]:
    """The leg deadline in scope, or ``None``.

    ``None`` is a real answer and is never papered over with a freshly
    constructed deadline: a clock started here would measure this *stage*, and
    an admission decision can only be made against what is left of the *leg*.
    """

    return _ACTIVE_DEADLINE.get()


def activate_leg_deadline(deadline: Optional[LegDeadline]) -> Token:
    """Install ``deadline`` for the current context; returns the reset token."""

    return _ACTIVE_DEADLINE.set(deadline)


def deactivate_leg_deadline(token: Token) -> None:
    """Undo :func:`activate_leg_deadline`."""

    _ACTIVE_DEADLINE.reset(token)


# ---------------------------------------------------------------------------
# Records.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelAttempt:
    """One issued model attempt: which rung, which model, which prompt, what came back.

    ``request_hash`` is computed by the caller over what it is about to send, not
    read back from the provider's diagnostics: the rule in § 9 is about whether a
    prompt is *re-sent*, which has to be answerable before the call, and an
    offline test has no provider to read it from.
    """

    number: int
    rung: str
    model: str
    request_hash: str
    scope: str = SCOPE_FULL_TEXT
    response_hash: str = ""
    structurally_empty: bool = False
    calls: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": int(self.number),
            "rung": self.rung,
            "model": self.model,
            "scope": self.scope,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "structurally_empty": bool(self.structurally_empty),
            "model_calls": int(self.calls),
        }


@dataclass(frozen=True)
class RungDecision:
    """Whether one rung may start, and the arithmetic and rules that decided it."""

    allowed: bool
    rung: str
    number: int
    budget_conditional: bool = False
    skip_cause: str = ""
    termination_reason: str = ""
    admission: Optional[Admission] = None
    checkpoint: Optional[Checkpoint] = None
    detail: str = ""

    def raise_if_refused(self) -> None:
        """Surface a *budget* refusal as :class:`BudgetExhausted`.

        Only a budget refusal raises. The other skip causes are decisions the
        caller has to keep going after -- ``attempt_cap_reached`` ends the ladder
        with the payload it already has, and turning that into an exception would
        report a stopped clock where there was none.
        """

        if self.allowed:
            return
        if self.admission is not None and not self.admission.allowed:
            self.admission.raise_if_refused()

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "rung": self.rung,
            "attempt": int(self.number),
            "allowed": bool(self.allowed),
            "budget_conditional": bool(self.budget_conditional),
        }
        if self.skip_cause:
            record["skip_cause"] = self.skip_cause
        if self.termination_reason:
            record["termination_reason"] = self.termination_reason
        if self.detail:
            record["detail"] = self.detail
        if self.admission is not None:
            record["admission"] = self.admission.to_dict()
        return record


# ---------------------------------------------------------------------------
# The ladder.
# ---------------------------------------------------------------------------
class ExtractionLadder:
    """Attempt history, the § 9 ceiling, the identical-empty rule, the budget gate.

    One instance per ``_run_json_stage`` call. It answers three questions and
    nothing else: *may this rung start*, *what has already been sent to whom*,
    and *what does the § 9 preservation set look like right now*.
    """

    def __init__(
        self,
        *,
        stage: str = "extraction",
        deadline: Optional[LegDeadline] = None,
        max_total_attempts: int = MAX_TOTAL_MODEL_ATTEMPTS,
        price_fn: Callable[..., float] = price_llm_call,
        persist: Optional[Callable[[Checkpoint], Any]] = None,
    ) -> None:
        self.stage = str(stage or "")
        #: ``None`` means "no budget notion in scope", which rung 3 treats as a
        #: refusal and rungs 1-2 treat as "behave as you did before C-042".
        self.deadline = deadline if deadline is not None else current_leg_deadline()
        self.max_total_attempts = max(0, int(max_total_attempts))
        self._price_fn = price_fn
        #: Whatever actually writes a checkpoint. Public so the stage that owns
        #: the artifact directory can attach its writer to a ladder it was
        #: handed rather than one it built.
        self.persist = persist
        self.issued: List[ModelAttempt] = []
        self.decisions: List[RungDecision] = []
        self.skipped: List[Dict[str, Any]] = []
        self.checkpoints: List[Checkpoint] = []

    # -- history -----------------------------------------------------------
    @property
    def attempts_used(self) -> int:
        """Model calls issued so far, repair draws included."""

        return sum(int(a.calls) for a in self.issued)

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_total_attempts - self.attempts_used)

    @property
    def budget_determinable(self) -> bool:
        return self.deadline is not None

    def prompt_already_sent(self, *, model: str, request_hash: str) -> bool:
        """Whether this exact prompt already went to this exact model."""

        if not request_hash:
            return False
        return any(
            a.request_hash == request_hash and a.model == model for a in self.issued
        )

    def identical_empty_hash(self) -> str:
        """The response hash two attempts returned identically empty, or ``""``.

        This is the PMC12782028 signature, pinned by hash rather than prose: two
        *structurally empty* replies whose bodies were byte-identical. Two empty
        replies that differ are not this -- the model did move, it just moved to
        another nothing -- and § 9's prohibition is specifically about the
        strategy having been proven inert.
        """

        empties = [a for a in self.issued if a.structurally_empty and a.response_hash]
        for i in range(len(empties) - 1):
            if empties[i].response_hash == empties[i + 1].response_hash:
                return empties[i].response_hash
        return ""

    # -- the gate ----------------------------------------------------------
    def admit(
        self,
        *,
        rung: str,
        model: str,
        request_hash: str,
        scope: str = SCOPE_FULL_TEXT,
        budget_conditional: bool = False,
        call_seconds: Optional[float] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> RungDecision:
        """Decide whether one model attempt may be issued, and record why.

        Order matters. The ceiling is checked first because it holds regardless
        of budget; the identical-prompt rule next because § 9 states it as an
        absolute ("must never"), not as something budget can buy past; and the
        budget gate last, because it is the only test whose answer is allowed to
        depend on how long the leg has already taken.
        """

        number = self.attempts_used + 1

        def refuse(cause: str, *, reason: str = "", detail: str = "",
                   admission: Optional[Admission] = None,
                   checkpoint: Optional[Checkpoint] = None) -> RungDecision:
            decision = RungDecision(
                allowed=False,
                rung=rung,
                number=number,
                budget_conditional=budget_conditional,
                skip_cause=require_skip_cause(cause),
                termination_reason=require_reason(reason) if reason else "",
                admission=admission,
                checkpoint=checkpoint,
                detail=detail,
            )
            self.decisions.append(decision)
            self.skipped.append(decision.to_dict())
            return decision

        if self.attempts_used >= self.max_total_attempts:
            return refuse(
                SKIP_ATTEMPT_CAP,
                detail=(
                    f"{self.attempts_used} of {self.max_total_attempts} permitted model "
                    "attempts already issued"
                ),
            )

        if not str(model or "").strip():
            # A11: an unnamed model cannot be checked against the
            # identical-prompt rule, so issuing the call would mean issuing it
            # without knowing whether § 9 forbids it.
            return refuse(
                SKIP_MODEL_IDENTITY_UNKNOWN,
                detail="the model serving this rung could not be named",
            )

        repeated = self.identical_empty_hash()
        if repeated and self.prompt_already_sent(model=model, request_hash=request_hash):
            return refuse(
                SKIP_IDENTICAL_PROMPT,
                reason=IDENTICAL_EMPTY_RESPONSE,
                detail=(
                    f"two attempts already returned {repeated} from {model}; the same "
                    "prompt must not go to the same model again"
                ),
            )

        if self.deadline is None:
            if budget_conditional:
                # A11, and the honest answer: § 9 permits this rung "only if
                # budget remains", and with no leg clock in scope nothing here
                # can establish that it does.
                return refuse(
                    SKIP_BUDGET_INDETERMINATE,
                    detail=(
                        "no LegDeadline is in scope, so remaining budget could not be "
                        "established for a rung § 9 conditions on it"
                    ),
                )
            decision = RungDecision(
                allowed=True, rung=rung, number=number,
                budget_conditional=budget_conditional,
            )
            self.decisions.append(decision)
            return decision

        seconds = self._price_fn() if call_seconds is None else float(call_seconds)
        checkpoint, admission = self.deadline.before_long_call(
            label=f"{self.stage}:{rung}",
            call_seconds=seconds,
            stage=self.stage,
            state=dict(state or {}, attempt=number, rung=rung, model=model),
            persist=self.persist,
        )
        self.checkpoints.append(checkpoint)
        if not admission.allowed:
            return refuse(
                SKIP_BUDGET_REFUSED,
                reason=BUDGET_EXHAUSTED,
                admission=admission,
                checkpoint=checkpoint,
                detail=(
                    f"{admission.required_seconds:.0f}s needed, "
                    f"{admission.remaining_seconds:.0f}s of leg budget left"
                ),
            )
        decision = RungDecision(
            allowed=True, rung=rung, number=number,
            budget_conditional=budget_conditional,
            admission=admission, checkpoint=checkpoint,
        )
        self.decisions.append(decision)
        return decision

    def materially_differs(
        self, *, model: str, request_hash: str, scope: str
    ) -> Tuple[bool, str]:
        """Whether a proposed rung is observably different from what already ran.

        § 9's third rung is "a materially different strategy", and A2 makes that
        checkable: a different ``request_hash`` **and** either a different model
        or a narrower scope. Verified against the attempts actually issued rather
        than assumed from the fact that a different prompt builder was called --
        a builder that ignored the request produces a prompt that hashes to
        something new while still being the same strategy in a different order of
        words, and that is the tweak this card exists to stop counting as a rung.
        """

        if not request_hash:
            return False, "the proposed rung's request could not be hashed"
        if any(a.request_hash == request_hash for a in self.issued):
            return False, "the proposed rung re-sends an already-issued request"
        if scope and scope != SCOPE_FULL_TEXT:
            return True, ""
        if all(a.model != model for a in self.issued):
            return True, ""
        return False, (
            "same model as an earlier attempt and no narrower scope: a prompt "
            "tweak is not a rung"
        )

    def refuse_not_materially_different(self, *, rung: str, detail: str) -> RungDecision:
        """Record a rung refused by :meth:`materially_differs`."""

        decision = RungDecision(
            allowed=False,
            rung=rung,
            number=self.attempts_used + 1,
            budget_conditional=True,
            skip_cause=SKIP_NOT_MATERIALLY_DIFFERENT,
            detail=detail,
        )
        self.decisions.append(decision)
        self.skipped.append(decision.to_dict())
        return decision

    # -- recording ---------------------------------------------------------
    def record(
        self,
        *,
        rung: str,
        model: str,
        request_hash: str,
        response_hash: str = "",
        scope: str = SCOPE_FULL_TEXT,
        structurally_empty: bool = False,
        calls: int = 1,
    ) -> ModelAttempt:
        """Record model calls that were actually issued.

        ``calls`` is not always 1: ``localized_repair.repair_json_text`` can draw
        more than once inside a single invocation and reports how many it used,
        and every one of those is a model attempt under the § 9 ceiling.
        """

        attempt = ModelAttempt(
            number=self.attempts_used + 1,
            rung=rung,
            model=str(model or ""),
            request_hash=str(request_hash or ""),
            scope=str(scope or SCOPE_FULL_TEXT),
            response_hash=str(response_hash or ""),
            structurally_empty=bool(structurally_empty),
            calls=max(0, int(calls)),
        )
        self.issued.append(attempt)
        return attempt

    def repair_budget(self, configured: int) -> int:
        """How many repair draws fit under the ceiling. Never more than configured.

        ``repair_json_text`` already treats 0 as "not attempted" and returns
        without calling a model, so a spent ceiling needs no special case here.
        """

        return max(0, min(int(configured), self.attempts_remaining))

    # -- the § 9 preservation set -----------------------------------------
    def preservation_record(
        self,
        *,
        last_completed_stage: str = "",
        payload: Any = None,
        termination_reason: str = "",
    ) -> Dict[str, Any]:
        """What § 9 says must survive a stop: everything needed to resume or judge.

        Last completed stage, the current structured payload, attempt numbers
        with their prompts/models and response hashes, elapsed and remaining
        budget, the next recovery step that was skipped, and the exact stop
        reason. Assembled here so no caller has to remember the list.
        """

        record: Dict[str, Any] = {
            "attempt_cap": int(self.max_total_attempts),
            "attempts_issued": self.attempts_used,
            "attempts_remaining": self.attempts_remaining,
            "attempts": [a.to_dict() for a in self.issued],
            "identical_empty_response_hash": self.identical_empty_hash(),
            "budget_determinable": self.budget_determinable,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }
        if last_completed_stage:
            record["last_completed_stage"] = str(last_completed_stage)
        if payload is not None:
            record["current_structured_payload"] = payload
        record["budget"] = (
            self.deadline.snapshot()
            if self.deadline is not None
            else {"budget_determinable": False}
        )
        # "The next recovery step that was skipped" -- the most recent refusal,
        # which is the one a resume would retry.
        record["skipped_next_step"] = self.skipped[-1] if self.skipped else None
        record["skipped_steps"] = list(self.skipped)
        if termination_reason:
            record["termination_reason"] = require_reason(termination_reason)
        return record


# Re-exported so a caller importing the ladder does not also have to import the
# deadline module to name the two operational reasons it can produce.
OPERATIONAL_REASONS: Tuple[str, ...] = (BUDGET_EXHAUSTED, OPERATION_TIMEOUT)
