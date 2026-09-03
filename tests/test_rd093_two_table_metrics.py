"""D-093 s.5.5 -- the two tables must stay two. NEW ACCEPTANCE TEST, explicitly labelled.

**G9.** ``rd093_two_table_metrics.py`` is a new evaluation-only module. It corrects no
pre-existing observable behaviour, so under merge gate G9 this file is an explicitly
labelled NEW acceptance test and carries no base-SHA failure proof.

THE ONE PROPERTY THESE TESTS EXIST TO DEFEND. D-093 section 1: *"Stage-1
paper-extraction recall and final-system biological support are SEPARATE METRICS and
must never be summed."* An ``external_rag_supported`` row is SUPPORTED in Table 2 and
ABSENT from Table 1's denominator, and no single number may stand in for both. That
distinction is what D-091 lacked, and every test below breaks an implementation that
loses it.
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

INSTRUMENT = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "rd093_two_table_metrics.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tt() -> Any:
    if not INSTRUMENT.is_file():
        pytest.skip(f"instrument not present: {INSTRUMENT}")
    return _load(INSTRUMENT, "rd093_two_table_metrics")


@pytest.fixture(scope="module")
def rd(tt: Any) -> Any:
    return _load(ROOT / "docs" / "pwml_recovery_sprint" / "evidence"
                 / "rd092_1_reaction_lineage.py", "rd092_1_reaction_lineage")


class _Term:
    def __init__(self, name: str) -> None:
        self.name = name


class _Sig:
    """A gold signature stand-in. ``quote`` drives the verification path."""

    def __init__(self, quote: str) -> None:
        self.quote = quote


def _fold(v: Any) -> str:
    return (v or "").strip().lower()


def _records(rd: Any, classes: List[str]) -> List[Dict[str, Any]]:
    return [{"row_index": i, "support_class": c} for i, c in enumerate(classes)]


# ---------------------------------------------------------------------------
# The separation itself
# ---------------------------------------------------------------------------

def test_external_rag_row_is_supported_in_table2_and_absent_from_table1(tt: Any, rd: Any) -> None:
    """The whole ruling in one assertion.

    One target-paper row that matches the gold signature, one externally supported row
    that does not. Table 1 must not charge the external row as a paper-extraction false
    positive; Table 2 must not call it unsupported.
    """

    reactions = [
        {"name": "paper row", "inputs": ["a"], "outputs": ["b"]},
        {"name": "external row", "inputs": ["x"], "outputs": ["y"]},
    ]
    records = _records(rd, [rd.TARGET_PAPER_SUPPORTED, rd.EXTERNAL_RAG_SUPPORTED])

    class Case:
        supported_reactions = (_Sig("the paper states a to b"),)

    def matches(sig: Any, row: Dict[str, Any]) -> bool:
        return row["name"] == "paper row"

    out = tt.score_leg(rd, records, Case(), {"processes": {"reactions": reactions}},
                       "The paper states A to B", _fold, matches)
    t1 = out["table1_paper_extraction"]
    t2 = out["table2_final_support"]

    # Table 1: the external row is NOT in the denominator.
    assert t1["precision_denominator_rows_claimed_target_paper_supported"] == 1
    assert t1["precision"] == 1.0, "the external row was charged as a paper-extraction miss"
    assert t1["recall"] == 1.0

    # Table 2: a different denominator, and the external row counts as SUPPORTED.
    assert t2["retained_rows"] == 2
    assert t2["external_rag_supported"] == 1
    assert t2["unsupported_rows"] == 0
    assert t2["unsupported_rate"] == 0.0


def test_the_two_denominators_are_different_objects(tt: Any, rd: Any) -> None:
    """If these ever coincide by construction, the separation has been lost."""

    reactions = [{"name": f"r{i}", "inputs": [], "outputs": []} for i in range(3)]
    records = _records(rd, [rd.TARGET_PAPER_SUPPORTED, rd.EXTERNAL_RAG_SUPPORTED,
                            rd.INDETERMINATE])

    class Case:
        supported_reactions = (_Sig("q"),)

    out = tt.score_leg(rd, records, Case(), {"processes": {"reactions": reactions}},
                       "q", _fold, lambda s, r: False)
    assert out["table1_paper_extraction"]["precision_denominator_rows_claimed_target_paper_supported"] == 1
    assert out["table2_final_support"]["retained_rows"] == 3


def test_unsupported_rate_uses_every_retained_row_not_only_attributed_ones(tt: Any, rd: Any) -> None:
    reactions = [{"name": f"r{i}", "inputs": [], "outputs": []} for i in range(4)]
    records = _records(rd, [rd.UNSUPPORTED, rd.INDETERMINATE,
                            rd.TARGET_PAPER_SUPPORTED, rd.EXTERNAL_RAG_SUPPORTED])

    class Case:
        supported_reactions = ()

    out = tt.score_leg(rd, records, Case(), {"processes": {"reactions": reactions}},
                       "text", _fold, lambda s, r: False)
    t2 = out["table2_final_support"]
    assert t2["retained_rows"] == 4 and t2["unsupported_rows"] == 1
    assert t2["unsupported_rate"] == 0.25


# ---------------------------------------------------------------------------
# Gold defects are excluded, never charged to the run
# ---------------------------------------------------------------------------

def test_unverifiable_gold_quote_is_excluded_and_counted_not_charged(tt: Any, rd: Any) -> None:
    """A gold quote absent from the stored text is a GOLD defect (semantic.py's rule)."""

    class Case:
        supported_reactions = (_Sig("present in the text"), _Sig("absent from the text"))

    out = tt.score_leg(rd, _records(rd, [rd.TARGET_PAPER_SUPPORTED]), Case(),
                       {"processes": {"reactions": [{"name": "r", "inputs": [], "outputs": []}]}},
                       "PRESENT IN THE TEXT", _fold, lambda s, r: True)
    t1 = out["table1_paper_extraction"]
    assert t1["gold_signatures_stated"] == 2
    assert t1["gold_signatures_unverifiable_excluded"] == 1
    assert t1["recall_denominator_verified_signatures"] == 1, (
        "an unverifiable gold quote was scored against the run instead of excluded")
    assert t1["recall"] == 1.0


def test_missing_paper_text_is_inapplicable_and_never_a_silent_pass(tt: Any, rd: Any) -> None:
    class Case:
        supported_reactions = (_Sig("q"),)

    out = tt.score_leg(rd, _records(rd, [rd.TARGET_PAPER_SUPPORTED]), Case(),
                       {"processes": {"reactions": [{"name": "r", "inputs": [], "outputs": []}]}},
                       None, _fold, lambda s, r: True)
    t1 = out["table1_paper_extraction"]
    assert t1["evaluable"] is False
    assert "no stored paper text" in t1["inapplicable_reason"]


def test_a_case_with_no_signatures_is_inapplicable_not_zero_recall(tt: Any, rd: Any) -> None:
    class Case:
        supported_reactions = ()

    out = tt.score_leg(rd, _records(rd, [rd.TARGET_PAPER_SUPPORTED]), Case(),
                       {"processes": {"reactions": [{"name": "r", "inputs": [], "outputs": []}]}},
                       "text", _fold, lambda s, r: True)
    t1 = out["table1_paper_extraction"]
    assert t1["evaluable"] is False
    assert t1["recall"] is None, "an unscoreable case was reported as 0% recall"


# ---------------------------------------------------------------------------
# Reporting discipline
# ---------------------------------------------------------------------------

def test_f1_is_none_rather_than_zero_when_a_side_is_unscoreable(tt: Any) -> None:
    assert tt.f1(None, 1.0) is None
    assert tt.f1(0.0, 0.0) is None
    assert tt.f1(1.0, 1.0) == 1.0


def test_render_never_prints_a_rate_without_naming_its_denominator(tt: Any, rd: Any) -> None:
    legs = [{
        "leg_dir": "runs/R/papers/PMC1/strict", "population": "canonical",
        "target_paper": "PMC1",
        "scores": {
            "table1_paper_extraction": {
                "gold_signatures_stated": 2, "gold_signatures_unverifiable_excluded": 0,
                "recall_numerator_matched_signatures": 1,
                "recall_denominator_verified_signatures": 2, "recall": 0.5,
                "precision_numerator_matched_rows": 1,
                "precision_denominator_rows_claimed_target_paper_supported": 2,
                "precision": 0.5, "evaluable": True, "inapplicable_reason": "",
            },
            "table2_final_support": {
                "retained_rows": 4, "unsupported_rows": 1, "unsupported_rate": 0.25,
                "target_paper_supported": 2, "external_rag_supported": 1, "indeterminate": 0,
            },
        },
    }]
    agg = tt.aggregate(legs)
    text = tt.render(agg)
    for line in text.splitlines():
        if "%" in line and "n/a" not in line:
            assert "=" in line or "F1" in line, f"rate printed with no denominator: {line!r}"
    # The distinct-signature count must be stated so the micro-averaged denominator
    # cannot be misread as a count of distinct gold signatures.
    assert "distinct signatures" in text
    assert "not addable" in text.lower() or "NOT addable" in text


def test_aggregate_micro_averages_and_reports_both_units(tt: Any) -> None:
    """Two legs of unequal evidence must not be averaged as equal rates."""

    def leg(paper: str, rn: int, rd_: int, pn: int, pd_: int) -> Dict[str, Any]:
        return {
            "leg_dir": f"runs/R/papers/{paper}/strict", "population": "canonical",
            "target_paper": paper,
            "scores": {
                "table1_paper_extraction": {
                    "gold_signatures_stated": rd_, "gold_signatures_unverifiable_excluded": 0,
                    "recall_numerator_matched_signatures": rn,
                    "recall_denominator_verified_signatures": rd_,
                    "recall": rn / rd_, "precision_numerator_matched_rows": pn,
                    "precision_denominator_rows_claimed_target_paper_supported": pd_,
                    "precision": pn / pd_, "evaluable": True, "inapplicable_reason": "",
                },
                "table2_final_support": {
                    "retained_rows": pd_, "unsupported_rows": 0, "unsupported_rate": 0.0,
                    "target_paper_supported": pd_, "external_rag_supported": 0,
                    "indeterminate": 0,
                },
            },
        }

    agg = tt.aggregate([leg("PMC1", 1, 1, 1, 1), leg("PMC2", 1, 3, 1, 3)])
    t1 = agg["canonical"]["table1_paper_extraction"]
    # Micro: 2/4 = 0.5. A macro average of the rates would be (1.0 + 0.333)/2 = 0.667.
    assert t1["recall"] == 0.5, "rates were macro-averaged, weighting unequal evidence equally"
    assert t1["recall_denominator_unit"] == "verified (signature, leg) pairs"
    assert t1["distinct_papers_scored"] == 2
