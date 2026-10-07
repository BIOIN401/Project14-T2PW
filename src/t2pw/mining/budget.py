"""Budget accounting and stop conditions.

One :class:`BudgetLedger` owns every counter a run can exhaust. The controller
asks it two questions and nothing else:

* ``check_before_round()`` / ``check_before_call()`` -- may I start this?
* ``stop_reason`` -- if not, exactly which limit stopped me?

A run that stops always names the limit. "The run ended" with no reason is a bug,
so :data:`STOP_REASONS` is closed and :meth:`BudgetLedger.stop` refuses an
unknown one.

Cost is an **estimate** computed from configured per-million-token rates. It is
never a billed amount and every artifact that prints it says so.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from t2pw.mining.config import MiningBudgets

STOP_MAX_ROUNDS = "max_rounds_reached"
STOP_MAX_PAPERS_DISCOVERED = "max_papers_discovered_reached"
STOP_MAX_PAPERS_PROCESSED = "max_papers_processed_reached"
STOP_MAX_PRIMARY_REACTIONS = "max_primary_reactions_reached"
STOP_MAX_CANDIDATE_REACTIONS = "max_candidate_reactions_reached"
STOP_MAX_CANDIDATE_COMPONENTS = "max_candidate_components_reached"
STOP_MAX_NEW_REACTIONS_PER_ROUND = "max_new_reactions_per_round_reached"
STOP_MAX_LLM_CALLS = "max_llm_calls_reached"
STOP_MAX_INPUT_TOKENS = "max_input_tokens_reached"
STOP_MAX_OUTPUT_TOKENS = "max_output_tokens_reached"
STOP_MAX_COST = "max_estimated_cost_reached"
STOP_MAX_WALL_CLOCK = "max_wall_clock_reached"
STOP_CONVERGED = "converged_no_useful_additions"
STOP_FRONTIER_EXHAUSTED = "frontier_exhausted"
STOP_CANCELLED = "cancelled_by_user"
STOP_PREFLIGHT_FAILED = "preflight_failed"
STOP_ERROR = "unrecoverable_error"
STOP_COMPLETED = "completed_all_rounds"

STOP_REASONS = frozenset(
    {
        STOP_MAX_ROUNDS,
        STOP_MAX_PAPERS_DISCOVERED,
        STOP_MAX_PAPERS_PROCESSED,
        STOP_MAX_PRIMARY_REACTIONS,
        STOP_MAX_CANDIDATE_REACTIONS,
        STOP_MAX_CANDIDATE_COMPONENTS,
        STOP_MAX_NEW_REACTIONS_PER_ROUND,
        STOP_MAX_LLM_CALLS,
        STOP_MAX_INPUT_TOKENS,
        STOP_MAX_OUTPUT_TOKENS,
        STOP_MAX_COST,
        STOP_MAX_WALL_CLOCK,
        STOP_CONVERGED,
        STOP_FRONTIER_EXHAUSTED,
        STOP_CANCELLED,
        STOP_PREFLIGHT_FAILED,
        STOP_ERROR,
        STOP_COMPLETED,
    }
)


class BudgetExhausted(RuntimeError):
    """A limit was reached. Carries the exact stop reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class RunCancelled(RuntimeError):
    """Cooperative cancellation. The run stops after the current round's
    artifacts are flushed, and is NEVER reported as a normal completion."""


@dataclass
class Usage:
    """Everything consumed so far."""

    rounds: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0
    papers_discovered: int = 0
    papers_processed: int = 0
    passages_extracted: int = 0
    claims_extracted: int = 0
    elapsed_seconds: float = 0.0
    unproductive_streak: int = 0
    #: Reactions newly ADMITTED to the target graph in the last round. Quarantined
    #: candidates and leads do not count (ORCH-745 B1): in pilot-003 one off-species
    #: review gave 23 of 35 never-admissible candidates and ended the run after round 1.
    last_round_new_reactions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BudgetLedger:
    """Enforces :class:`MiningBudgets` and records why a run stopped."""

    def __init__(self, budgets: MiningBudgets, *, clock: Optional[Any] = None) -> None:
        self.budgets = budgets
        self.usage = Usage()
        self._clock = clock or time.monotonic
        self._started = float(self._clock())
        self.stop_reason: str = ""
        self.stop_detail: str = ""
        self.warnings: List[str] = []
        self._cancelled = False

    # -- time ---------------------------------------------------------------
    def elapsed(self) -> float:
        return max(0.0, float(self._clock()) - self._started)

    def refresh(self) -> None:
        self.usage.elapsed_seconds = round(self.elapsed(), 3)

    # -- cancellation -------------------------------------------------------
    def cancel(self) -> None:
        """Request a graceful stop. Idempotent; safe from another thread."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # -- recording ----------------------------------------------------------
    def price(self, input_tokens: int, output_tokens: int) -> float:
        """Estimated USD for one call, from the configured rates."""
        b = self.budgets
        return (
            (max(0, int(input_tokens)) / 1_000_000.0) * float(b.price_input_per_mtok_usd)
            + (max(0, int(output_tokens)) / 1_000_000.0) * float(b.price_output_per_mtok_usd)
        )

    def record_call(
        self, *, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0
    ) -> float:
        cost = self.price(input_tokens, output_tokens)
        self.usage.llm_calls += 1
        self.usage.input_tokens += max(0, int(input_tokens))
        self.usage.output_tokens += max(0, int(output_tokens))
        self.usage.cached_tokens += max(0, int(cached_tokens))
        self.usage.estimated_cost_usd = round(self.usage.estimated_cost_usd + cost, 6)
        self.refresh()
        return cost

    def record_round(self, *, useful: bool, new_reactions: int = 0) -> None:
        self.usage.rounds += 1
        self.usage.last_round_new_reactions = max(0, int(new_reactions))
        self.usage.unproductive_streak = 0 if useful else self.usage.unproductive_streak + 1
        self.refresh()

    def record_papers_discovered(self, count: int) -> None:
        self.usage.papers_discovered += max(0, int(count))

    def record_paper_processed(self) -> None:
        self.usage.papers_processed += 1

    def record_passages(self, count: int) -> None:
        self.usage.passages_extracted += max(0, int(count))

    def record_claims(self, count: int) -> None:
        self.usage.claims_extracted += max(0, int(count))

    def warn(self, message: str) -> None:
        text = str(message or "").strip()
        if text and text not in self.warnings:
            self.warnings.append(text)

    # -- limits -------------------------------------------------------------
    def stop(self, reason: str, detail: str = "") -> None:
        if reason not in STOP_REASONS:
            raise ValueError(f"unknown stop reason {reason!r}; STOP_REASONS is closed")
        if not self.stop_reason:
            self.stop_reason = reason
            self.stop_detail = detail
        self.refresh()

    def _limit(self, reason: str, used: Any, cap: Any, label: str) -> Optional[str]:
        if cap and used >= cap:
            self.stop(reason, f"{label}: {used} >= limit {cap}")
            return reason
        return None

    def check_before_round(
        self,
        *,
        primary_reactions: int = 0,
        candidate_reactions: int = 0,
        candidate_components: int = 0,
    ) -> Optional[str]:
        """The stop reason that forbids starting another round, or ``None``.

        A stop already recorded mid-round (the discovery budget, the call budget)
        forbids the next round too: pilot-002 started a round 3 that could not
        search (ORCH-745 A5).
        """
        self.refresh()
        if self._cancelled:
            self.stop(STOP_CANCELLED, "cancellation requested")
            return STOP_CANCELLED
        if self.stop_reason:
            return self.stop_reason
        b = self.budgets
        for reason in (
            self._limit(STOP_MAX_ROUNDS, self.usage.rounds, b.max_rounds, "rounds"),
            self._limit(STOP_MAX_WALL_CLOCK, self.elapsed(), b.max_wall_clock_seconds, "seconds"),
            self._limit(STOP_MAX_LLM_CALLS, self.usage.llm_calls, b.max_llm_calls, "llm calls"),
            self._limit(STOP_MAX_INPUT_TOKENS, self.usage.input_tokens, b.max_input_tokens, "input tokens"),
            self._limit(STOP_MAX_OUTPUT_TOKENS, self.usage.output_tokens, b.max_output_tokens, "output tokens"),
            self._limit(STOP_MAX_COST, self.usage.estimated_cost_usd, b.max_cost_usd, "estimated usd"),
            self._limit(STOP_MAX_PAPERS_DISCOVERED, self.usage.papers_discovered, b.max_papers_discovered, "papers discovered"),
            self._limit(STOP_MAX_PAPERS_PROCESSED, self.usage.papers_processed, b.max_papers_processed, "papers processed"),
            self._limit(STOP_MAX_PRIMARY_REACTIONS, primary_reactions, b.max_primary_reactions, "primary reactions"),
            self._limit(STOP_MAX_CANDIDATE_REACTIONS, candidate_reactions, b.max_candidate_reactions, "candidate reactions"),
            self._limit(STOP_MAX_CANDIDATE_COMPONENTS, candidate_components, b.max_candidate_components, "candidate components"),
            self._limit(
                STOP_MAX_NEW_REACTIONS_PER_ROUND,
                self.usage.last_round_new_reactions,
                b.max_new_reactions_per_round,
                f"newly admitted reactions in round {self.usage.rounds}",
            ),
        ):
            if reason:
                return reason
        if (
            b.stop_after_unproductive_rounds
            and self.usage.unproductive_streak >= b.stop_after_unproductive_rounds
        ):
            self.stop(
                STOP_CONVERGED,
                f"{self.usage.unproductive_streak} consecutive rounds added nothing useful",
            )
            return STOP_CONVERGED
        return None

    def check_before_call(self) -> Optional[str]:
        """The stop reason that forbids another model call, or ``None``."""
        self.refresh()
        if self._cancelled:
            self.stop(STOP_CANCELLED, "cancellation requested")
            return STOP_CANCELLED
        b = self.budgets
        for reason in (
            self._limit(STOP_MAX_LLM_CALLS, self.usage.llm_calls, b.max_llm_calls, "llm calls"),
            self._limit(STOP_MAX_INPUT_TOKENS, self.usage.input_tokens, b.max_input_tokens, "input tokens"),
            self._limit(STOP_MAX_OUTPUT_TOKENS, self.usage.output_tokens, b.max_output_tokens, "output tokens"),
            self._limit(STOP_MAX_COST, self.usage.estimated_cost_usd, b.max_cost_usd, "estimated usd"),
            self._limit(STOP_MAX_WALL_CLOCK, self.elapsed(), b.max_wall_clock_seconds, "seconds"),
        ):
            if reason:
                return reason
        return None

    def check_papers_processed(self) -> Optional[str]:
        return self._limit(
            STOP_MAX_PAPERS_PROCESSED,
            self.usage.papers_processed,
            self.budgets.max_papers_processed,
            "papers processed",
        )

    def check_papers_discovered(self) -> Optional[str]:
        return self._limit(
            STOP_MAX_PAPERS_DISCOVERED,
            self.usage.papers_discovered,
            self.budgets.max_papers_discovered,
            "papers discovered",
        )

    # -- reporting ----------------------------------------------------------
    def costs_to_dict(self) -> Dict[str, Any]:
        self.refresh()
        b = self.budgets
        return {
            "estimate_only": True,
            "note": (
                "Estimated from configured per-million-token rates. This is NOT a billed "
                "amount and does not reflect provider discounts, cached-token pricing or "
                "minimum charges."
            ),
            "price_input_per_mtok_usd": b.price_input_per_mtok_usd,
            "price_output_per_mtok_usd": b.price_output_per_mtok_usd,
            "llm_calls": self.usage.llm_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cached_tokens": self.usage.cached_tokens,
            "estimated_cost_usd": round(self.usage.estimated_cost_usd, 6),
            "max_cost_usd": b.max_cost_usd,
            "elapsed_seconds": self.usage.elapsed_seconds,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.refresh()
        return {
            "budgets": self.budgets.to_dict(),
            "usage": self.usage.to_dict(),
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "cancelled": self._cancelled,
            "warnings": list(self.warnings),
        }

    def restore(self, usage: Dict[str, Any]) -> None:
        """Re-seed counters from a checkpoint so a resume respects the budget.

        Wall clock is deliberately NOT restored: a resumed run gets a fresh
        wall-clock allowance, because the elapsed time of a run that was
        interrupted yesterday is not a resource this process is consuming. The
        token, call and cost budgets ARE cumulative across the resume.
        """
        data = dict(usage or {})
        for key in (
            "rounds", "llm_calls", "input_tokens", "output_tokens", "cached_tokens",
            "papers_discovered", "papers_processed", "passages_extracted",
            "claims_extracted", "unproductive_streak", "last_round_new_reactions",
        ):
            if key in data:
                setattr(self.usage, key, int(data.get(key) or 0))
        if "estimated_cost_usd" in data:
            self.usage.estimated_cost_usd = float(data.get("estimated_cost_usd") or 0.0)
        self._started = float(self._clock())
        self.refresh()


__all__ = [
    "STOP_REASONS",
    "STOP_MAX_ROUNDS",
    "STOP_MAX_PAPERS_DISCOVERED",
    "STOP_MAX_PAPERS_PROCESSED",
    "STOP_MAX_PRIMARY_REACTIONS",
    "STOP_MAX_CANDIDATE_REACTIONS",
    "STOP_MAX_CANDIDATE_COMPONENTS",
    "STOP_MAX_NEW_REACTIONS_PER_ROUND",
    "STOP_MAX_LLM_CALLS",
    "STOP_MAX_INPUT_TOKENS",
    "STOP_MAX_OUTPUT_TOKENS",
    "STOP_MAX_COST",
    "STOP_MAX_WALL_CLOCK",
    "STOP_CONVERGED",
    "STOP_FRONTIER_EXHAUSTED",
    "STOP_CANCELLED",
    "STOP_PREFLIGHT_FAILED",
    "STOP_ERROR",
    "STOP_COMPLETED",
    "BudgetExhausted",
    "RunCancelled",
    "Usage",
    "BudgetLedger",
]
