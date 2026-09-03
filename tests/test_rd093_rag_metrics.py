"""D-093 s.5.7 -- core RAG metrics. NEW ACCEPTANCE TEST, explicitly labelled.

**G9.** ``rd093_rag_metrics.py`` is a new evaluation-only module; it repairs no
pre-existing observable behaviour, so this is an explicitly labelled NEW acceptance test
with no base-SHA failure proof.

WHAT THESE DEFEND, and each was a real defect in this module before it shipped:

  * **Structural zeros.** The first revision printed ``retrieval_did_not_find_it: 0``,
    ``found_but_not_admitted: 0`` and ``rejected_candidate_reintroduced: 0`` -- not
    because they were measured zero but because nothing ever assigned them. A gold
    signature that no candidate matches produces NO GAP, so a per-gap counter can never
    see it. That is this project's standing "missing key read as zero" defect wearing a
    dashboard, and the fix was to count that outcome in its own unit.
  * **Units that differ.** Three units share one table (gold signature, gap, reaction
    row). A reader who adds them gets a meaningless number, so every row names its unit.
  * **Tie handling that flatters.** Rank is DERIVED from a chunk score that many
    candidates share. Best-rank tie handling would report Recall@1 = 1.0 for a retriever
    that expressed no preference at all.
  * **Truncation.** ``max_report_entries`` silently drops candidates, so a truncated
    leg's ranks assume the missing candidates were irrelevant. Truncated legs are a
    separate population and are never summed with clean ones.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INSTRUMENT = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "rd093_rag_metrics.py"


@pytest.fixture(scope="module")
def rm() -> Any:
    if not INSTRUMENT.is_file():
        pytest.skip(f"instrument not present: {INSTRUMENT}")
    spec = importlib.util.spec_from_file_location("rd093_rag_metrics", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rd093_rag_metrics"] = module
    spec.loader.exec_module(module)
    return module


def _c(score: float, relevant: bool, accepted: bool = False) -> Dict[str, Any]:
    return {"score": score, "relevant": relevant, "accepted": accepted, "name": "c"}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_ties_take_the_worst_rank_not_the_best(rm: Any) -> None:
    """Best-rank handling would score a no-preference retriever as perfect."""

    assert rm.pessimistic_ranks([0.9, 0.9, 0.9]) == [3, 3, 3]
    assert rm.pessimistic_ranks([0.9, 0.8, 0.7]) == [1, 2, 3]
    assert rm.pessimistic_ranks([0.9, 0.9, 0.5]) == [2, 2, 3]


def test_a_missing_score_sorts_last_not_first(rm: Any) -> None:
    """An absent score is not evidence of a good match."""

    ranks = rm.pessimistic_ranks([None, 0.5])
    assert ranks[1] < ranks[0]


def test_an_all_tied_gap_cannot_report_a_top1_hit(rm: Any) -> None:
    g = rm.score_gap([_c(0.9, False), _c(0.9, False), _c(0.9, True)])
    assert g["best_relevant_rank"] == 3
    assert g["hit_at_1"] is False and g["hit_at_5"] is True
    assert g["ties_in_top5"] == 3


# ---------------------------------------------------------------------------
# Positive and negative queries
# ---------------------------------------------------------------------------

def test_a_gap_with_no_relevant_candidate_is_a_negative_query_not_a_recall_miss(rm: Any) -> None:
    g = rm.score_gap([_c(0.9, False), _c(0.8, False)])
    assert g["is_positive_query"] is False
    assert "hit_at_1" not in g, "a negative query was scored as a recall failure"
    assert g["correctly_rejected_all"] is True


def test_a_negative_query_that_admitted_something_is_not_correctly_rejected(rm: Any) -> None:
    g = rm.score_gap([_c(0.9, False, accepted=True)])
    assert g["is_positive_query"] is False
    assert g["correctly_rejected_all"] is False


def test_reciprocal_rank_uses_the_best_relevant_candidate(rm: Any) -> None:
    g = rm.score_gap([_c(0.9, False), _c(0.8, True), _c(0.7, True)])
    assert g["best_relevant_rank"] == 2
    assert g["reciprocal_rank"] == 0.5


# ---------------------------------------------------------------------------
# The seven-way taxonomy -- no structural zeros, units kept apart
# ---------------------------------------------------------------------------

def test_retrieval_did_not_find_it_is_counted_per_signature_not_per_gap(rm: Any) -> None:
    """The regression this module shipped and then fixed.

    A gold signature no candidate matched produces NO GAP at all, so a per-gap counter
    reports a structural zero for the one outcome that means "retrieval failed
    entirely".
    """

    c = rm.classify_outcomes([], signatures_never_retrieved=4, reintroduced=0)
    assert c["retrieval_did_not_find_it"] == 4


def test_reintroduction_is_counted_per_reaction_row(rm: Any) -> None:
    c = rm.classify_outcomes([], signatures_never_retrieved=0, reintroduced=3)
    assert c["rejected_candidate_reintroduced"] == 3


def test_every_outcome_declares_its_unit(rm: Any) -> None:
    assert set(rm.OUTCOME_UNIT) == set(rm.OUTCOMES)
    assert len(set(rm.OUTCOME_UNIT.values())) == 3, "the three units collapsed into fewer"


def test_a_passed_over_relevant_candidate_is_distinct_from_a_rejected_one(rm: Any) -> None:
    """"found but the LLM ignored it" is not "correct candidate rejected"."""

    passed_over = rm.score_gap([_c(0.9, True), _c(0.8, False, accepted=True)])
    rejected = rm.score_gap([_c(0.9, True), _c(0.8, False)])
    a = rm.classify_outcomes([passed_over], 0, 0)
    b = rm.classify_outcomes([rejected], 0, 0)
    assert a["found_but_not_admitted"] == 1 and a["correct_candidate_rejected"] == 0
    assert b["correct_candidate_rejected"] == 1 and b["found_but_not_admitted"] == 0


def test_a_relevant_candidate_outside_top5_is_a_ranking_failure(rm: Any) -> None:
    gap = rm.score_gap([_c(1.0 - i / 100, False) for i in range(6)] + [_c(0.1, True)])
    c = rm.classify_outcomes([gap], 0, 0)
    assert c["found_but_ranked_poorly"] == 1
    assert c["correct_candidate_rejected"] == 0


def test_an_admitted_relevant_candidate_is_not_counted_as_any_failure(rm: Any) -> None:
    gap = rm.score_gap([_c(0.9, True, accepted=True)])
    c = rm.classify_outcomes([gap], 0, 0)
    assert sum(c[o] for o in rm.OUTCOMES) == 0


# ---------------------------------------------------------------------------
# Populations and denominators
# ---------------------------------------------------------------------------

def test_truncated_legs_are_never_summed_with_clean_ones(rm: Any) -> None:
    clean = {"leg_dir": "a", "target_paper": "P", "population": rm.CLEAN,
             "gaps": [rm.score_gap([_c(0.9, True)])], "outcomes": {}}
    trunc = {"leg_dir": "b", "target_paper": "P", "population": rm.TRUNCATED,
             "gaps": [rm.score_gap([_c(0.9, False), _c(0.8, False), _c(0.7, False),
                                    _c(0.6, False), _c(0.5, False), _c(0.4, True)])],
             "outcomes": {}}
    agg = rm.aggregate([clean, trunc])
    assert agg[rm.CLEAN]["recall_at_1"] == 1.0
    assert agg[rm.TRUNCATED]["recall_at_1"] == 0.0
    assert agg[rm.CLEAN]["legs"] == 1 and agg[rm.TRUNCATED]["legs"] == 1


def test_an_empty_population_reports_none_rather_than_zero(rm: Any) -> None:
    agg = rm.aggregate([])
    for pop in rm.POPULATION_ORDER:
        assert agg[pop]["recall_at_1"] is None, "an unmeasured population read as 0%"
        assert agg[pop]["mrr"] is None
        assert agg[pop]["negative_query_rejection_rate"] is None


def test_render_names_a_denominator_for_every_rate(rm: Any) -> None:
    agg = rm.aggregate([{"leg_dir": "a", "target_paper": "P", "population": rm.CLEAN,
                         "gaps": [rm.score_gap([_c(0.9, True)]),
                                  rm.score_gap([_c(0.9, False)])],
                         "outcomes": {}}])
    text = rm.render(agg, {"legs_total": 1, "truncated_legs": 0,
                           "candidates_counted_not_persisted": 0})
    for line in text.splitlines():
        if "%" in line and "n/a" not in line and "NARROW" not in line:
            assert "of " in line or "rejection" in line, f"rate without denominator: {line!r}"
    assert "NOT ADDABLE" in text
    assert "truncation" in text
