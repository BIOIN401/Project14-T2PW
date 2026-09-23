"""Budgets, stop conditions and cancellation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.budget import (  # noqa: E402
    STOP_CANCELLED,
    STOP_CONVERGED,
    STOP_MAX_CANDIDATE_REACTIONS,
    STOP_MAX_COST,
    STOP_MAX_INPUT_TOKENS,
    STOP_MAX_LLM_CALLS,
    STOP_MAX_NEW_REACTIONS_PER_ROUND,
    STOP_MAX_OUTPUT_TOKENS,
    STOP_MAX_PAPERS_DISCOVERED,
    STOP_MAX_PAPERS_PROCESSED,
    STOP_MAX_PRIMARY_REACTIONS,
    STOP_MAX_ROUNDS,
    STOP_MAX_WALL_CLOCK,
    STOP_REASONS,
    BudgetLedger,
)
from t2pw.mining.config import MiningBudgets  # noqa: E402


def ledger(**kwargs):
    defaults = dict(
        max_rounds=8, max_papers_discovered=100, max_papers_processed=30,
        max_llm_calls=50, max_input_tokens=100_000, max_output_tokens=20_000,
        max_cost_usd=1.0, max_wall_clock_seconds=3600.0,
        max_primary_reactions=200, max_candidate_reactions=300,
        max_candidate_components=40, stop_after_unproductive_rounds=2,
    )
    defaults.update(kwargs)
    return BudgetLedger(MiningBudgets(**defaults))


def test_the_stop_reason_set_is_closed():
    led = ledger()
    with pytest.raises(ValueError, match="STOP_REASONS is closed"):
        led.stop("the run just sort of ended")


def test_max_rounds_stops_the_loop():
    led = ledger(max_rounds=2)
    led.record_round(useful=True)
    led.record_round(useful=True)
    assert led.check_before_round() == STOP_MAX_ROUNDS
    assert "rounds: 2 >= limit 2" in led.stop_detail


def test_two_unproductive_rounds_converge_the_run():
    led = ledger(max_rounds=10, stop_after_unproductive_rounds=2)
    led.record_round(useful=True)
    assert led.check_before_round() is None
    led.record_round(useful=False)
    assert led.check_before_round() is None
    led.record_round(useful=False)
    assert led.check_before_round() == STOP_CONVERGED
    assert "consecutive rounds added nothing useful" in led.stop_detail


def test_a_useful_round_resets_the_unproductive_streak():
    led = ledger(stop_after_unproductive_rounds=2)
    led.record_round(useful=False)
    led.record_round(useful=True)
    assert led.usage.unproductive_streak == 0
    assert led.check_before_round() is None


@pytest.mark.parametrize(
    "kwargs,record,expected",
    [
        ({"max_llm_calls": 1}, lambda l: l.record_call(input_tokens=1, output_tokens=1),
         STOP_MAX_LLM_CALLS),
        ({"max_input_tokens": 10}, lambda l: l.record_call(input_tokens=10),
         STOP_MAX_INPUT_TOKENS),
        ({"max_output_tokens": 10}, lambda l: l.record_call(output_tokens=10),
         STOP_MAX_OUTPUT_TOKENS),
    ],
)
def test_call_budgets_block_the_next_call(kwargs, record, expected):
    led = ledger(**kwargs)
    assert led.check_before_call() is None
    record(led)
    assert led.check_before_call() == expected


def test_cost_is_estimated_from_the_configured_rates_and_capped():
    led = ledger(max_cost_usd=0.001, price_input_per_mtok_usd=1.0,
                 price_output_per_mtok_usd=2.0)
    cost = led.record_call(input_tokens=1000, output_tokens=1000)
    assert cost == pytest.approx(0.001 + 0.002)
    assert led.check_before_call() == STOP_MAX_COST


def test_costs_report_says_it_is_an_estimate_not_a_bill():
    led = ledger()
    led.record_call(input_tokens=1000, output_tokens=500)
    payload = led.costs_to_dict()
    assert payload["estimate_only"] is True
    assert "NOT a billed amount" in payload["note"]
    assert payload["input_tokens"] == 1000
    assert payload["output_tokens"] == 500


def test_wall_clock_is_enforced_with_an_injected_clock():
    ticks = iter([0.0, 0.0, 61.0, 61.0, 61.0, 61.0])
    led = BudgetLedger(MiningBudgets(max_wall_clock_seconds=60.0), clock=lambda: next(ticks))
    assert led.check_before_call() == STOP_MAX_WALL_CLOCK


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("primary_reactions", 200, STOP_MAX_PRIMARY_REACTIONS),
        ("candidate_reactions", 300, STOP_MAX_CANDIDATE_REACTIONS),
    ],
)
def test_graph_size_limits_stop_the_loop(field, value, expected):
    led = ledger()
    assert led.check_before_round(**{field: value}) == expected


def test_new_reactions_per_round_cap_is_checked_before_the_next_round():
    """Unit-level pin of the boundary. The G9 proof that the cap is enforced in a
    real run is test_mining_stability.py."""
    led = ledger(max_new_reactions_per_round=3)
    led.record_round(useful=True, new_reactions=2)
    assert led.check_before_round() is None
    led.record_round(useful=True, new_reactions=3)
    assert led.check_before_round() == STOP_MAX_NEW_REACTIONS_PER_ROUND
    assert "new reactions in round 2: 3 >= limit 3" in led.stop_detail


def test_paper_limits_are_tracked_separately():
    led = ledger(max_papers_discovered=3, max_papers_processed=2)
    led.record_papers_discovered(3)
    assert led.check_papers_discovered() == STOP_MAX_PAPERS_DISCOVERED
    led2 = ledger(max_papers_processed=1)
    led2.record_paper_processed()
    assert led2.check_before_round() == STOP_MAX_PAPERS_PROCESSED


def test_cancellation_is_cooperative_and_named():
    led = ledger()
    assert led.cancelled is False
    led.cancel()
    assert led.cancelled is True
    assert led.check_before_round() == STOP_CANCELLED
    assert led.check_before_call() == STOP_CANCELLED


def test_the_first_limit_reached_is_the_one_recorded():
    led = ledger(max_rounds=1, max_llm_calls=1)
    led.record_round(useful=True)
    assert led.check_before_round() == STOP_MAX_ROUNDS
    led.record_call(input_tokens=1)
    # A later limit does not overwrite the reason the run actually stopped.
    led.check_before_call()
    assert led.stop_reason == STOP_MAX_ROUNDS


def test_restore_makes_token_budgets_cumulative_across_a_resume():
    led = ledger(max_llm_calls=10)
    led.restore({"llm_calls": 8, "input_tokens": 500, "estimated_cost_usd": 0.25, "rounds": 3})
    assert led.usage.llm_calls == 8
    assert led.usage.rounds == 3
    assert led.usage.estimated_cost_usd == 0.25
    led.record_call(input_tokens=1)
    led.record_call(input_tokens=1)
    assert led.check_before_call() == STOP_MAX_LLM_CALLS


def test_restore_gives_a_fresh_wall_clock():
    """Yesterday's elapsed time is not a resource this process is consuming."""
    led = ledger(max_wall_clock_seconds=60.0)
    led.restore({"elapsed_seconds": 999999.0})
    assert led.check_before_call() is None


def test_every_stop_reason_constant_is_in_the_closed_set():
    for reason in (
        STOP_MAX_ROUNDS, STOP_MAX_PAPERS_DISCOVERED, STOP_MAX_PAPERS_PROCESSED,
        STOP_MAX_PRIMARY_REACTIONS, STOP_MAX_CANDIDATE_REACTIONS, STOP_MAX_LLM_CALLS,
        STOP_MAX_INPUT_TOKENS, STOP_MAX_OUTPUT_TOKENS, STOP_MAX_COST,
        STOP_MAX_WALL_CLOCK, STOP_CONVERGED, STOP_CANCELLED,
        STOP_MAX_NEW_REACTIONS_PER_ROUND,
    ):
        assert reason in STOP_REASONS


def test_budgets_resolve_from_the_environment(monkeypatch):
    monkeypatch.setenv("PATHWAY_MINING_MAX_ROUNDS", "4")
    monkeypatch.setenv("PATHWAY_MINING_MAX_COST_USD", "0.25")
    budgets = MiningBudgets.from_env()
    assert budgets.max_rounds == 4
    assert budgets.max_cost_usd == 0.25
    explicit = MiningBudgets.from_env({"max_rounds": 9})
    assert explicit.max_rounds == 9
