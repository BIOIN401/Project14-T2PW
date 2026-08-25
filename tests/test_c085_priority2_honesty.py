"""C-085 -- acceptance priority 2 must not report PASS on evidence it never gathered.

The defect
----------
``supported_reactions_complete`` is ``False`` on every case in the pinned gold set.
``semantic._check_supported_reactions`` deletes every ``unsupported_retained_reaction``
finding when the flag is false, and returns a hard ``0``. So the count feeding acceptance
priority 2 -- declared ABSOLUTE -- was structurally incapable of being non-zero on 8 of
the 10 pinned papers, and the run reported ``[PASS] observed: 0``.

The suppression is CORRECT and stays: the signature list is a hand-read subset, and a
cross-paper RAG addition cannot match a seed-paper signature by construction. Calling
those rows fabrications would report hundreds of hallucinations that did not happen.

What was wrong is that "not evaluated" was collapsed into "PASS 0", which
``PRODUCT_CONTRACT.md`` § 11 forbids: the system must distinguish passed / failed /
not-performed "without collapsing any into another", and "``not_evaluated`` is never
``false``".

Base-vs-tip (G9)
----------------
All 11 pass at the C-085 tip. On base ``2972c34``: **9 failed, 2 passed**.

Three of those failures are BEHAVIOURAL and carry the G9 proof -- they assert on values
base computes, never on a symbol base lacks:

* ``..._reports_not_evaluated_not_a_pass`` -- base returns
  ``CheckResult(ok=True, inapplicable_reason='')`` for a payload with two unattributed
  retained reactions, i.e. ``applicable`` is ``True``;
* ``..._does_not_report_pass`` -- base's priority-2 entry is ``ok=True``;
* ``..._says_not_eval_rather_than_pass`` -- base renders
  ``2. [PASS] zero unsupported retained reactions``.

The other six fail on base with ``KeyError`` for the new ``unsupported_verdict_evaluated``
/ ``evaluated`` keys. That is symbol absence, which G9 does not accept as proof; they are
here to hold the behaviour, not to demonstrate it.

The two that PASS on base are the preservation tests -- the subset suppression and the
attribution/recall reporting. They are green on both sides on purpose: this card must not
change them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from t2pw.bench.acceptance import score_run
from t2pw.bench.goldset import GoldCase, GoldTerm, SupportedReaction, load_gold_set
from t2pw.bench.render import render_text
from t2pw.bench.semantic import (
    CHECK_SUPPORTED_REACTIONS,
    ERR_UNSUPPORTED_REACTIONS,
    validate_semantic_coverage,
)

QUOTE = "MenF catalyzes the conversion of chorismate to isochorismate"
PAPER = f"In this organism {QUOTE}, the first step of the pathway."


def _case(**overrides) -> GoldCase:
    """A gold case shaped like the pinned ones: signatures, and NO completeness flag."""

    base = dict(
        paper_id="PMC_C085",
        title="c085 probe",
        requested_pathway="menaquinone biosynthesis",
        requested_organism="Escherichia coli",
        expected_pathway_anchors=(GoldTerm(name="menaquinone"),),
        supported_reactions=(
            SupportedReaction(
                inputs=(GoldTerm(name="chorismate"),),
                outputs=(GoldTerm(name="isochorismate"),),
                quote=QUOTE,
                label="MenF step",
            ),
        ),
    )
    base.update(overrides)
    return GoldCase(**base)


def _payload(reactions):
    return {
        "entities": {"compounds": [], "proteins": []},
        "processes": {"reactions": list(reactions)},
    }


_MATCHED = {
    "name": "menaquinone biosynthesis",
    "inputs": ["chorismate"],
    "outputs": ["isochorismate"],
    "evidence": QUOTE,
}
_UNATTRIBUTED = {
    "name": "from somewhere else",
    "inputs": ["isochorismate"],
    "outputs": ["SEPHCHC"],
    "evidence": "q",
}


# ---------------------------------------------------------------------------
# 1. The verdict is withheld, not passed.
# ---------------------------------------------------------------------------
def test_a_subset_set_with_unmatched_rows_reports_not_evaluated_not_a_pass():
    """THE G9 PROOF. On base this check returns ``ok=True`` with an empty
    ``inapplicable_reason``, so a paper whose unsupported-reaction verdict was never
    reached reads as a clean result."""

    report = validate_semantic_coverage(
        _case(), _payload([_MATCHED, _UNATTRIBUTED, dict(_UNATTRIBUTED, name="and another")]),
        paper_text=PAPER,
    )
    check = report.checks[CHECK_SUPPORTED_REACTIONS]

    assert check.applicable is False, "the unsupported-reaction verdict was NOT reached"
    assert "NOT EVALUATED" in check.inapplicable_reason
    assert "supported_reactions_complete" in check.inapplicable_reason
    # Not evaluated is never a failure either -- no manufactured FAIL.
    assert check.ok is not False
    assert CHECK_SUPPORTED_REACTIONS in report.inapplicable_checks
    assert CHECK_SUPPORTED_REACTIONS not in report.failed_checks
    assert report.support["unsupported_verdict_evaluated"] is False
    # The count itself stays a zero; what changes is that the zero now says so.
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 0


def test_a_run_whose_priority_2_was_never_measured_does_not_report_pass(tmp_path):
    """THE G9 PROOF at the acceptance layer, on the real pinned gold set."""

    gold = load_gold_set()
    case = _subset_case(gold)
    run_dir = _stage(tmp_path, case, [_row_for(case.supported_reactions[0]), _UNATTRIBUTED])

    report = score_run(run_dir, gold)
    entry = next(e for e in report.priorities() if e["rank"] == 2)

    assert entry["ok"] is not True, "priority 2 must not PASS on evidence never gathered"
    assert entry["evaluated"] is False
    assert entry["ok"] is None, "and must not be turned into a manufactured FAIL"
    assert "NOT EVALUATED" in str(entry["observed"])
    assert case.paper_id in entry["not_evaluated_papers"]
    assert entry["counted"] == 0
    # ok=None is falsy, so `all(entry["ok"] ...)` refuses the run.
    assert not all(e["ok"] for e in report.priorities() if e["rank"] <= 3)


def test_the_rendered_report_says_not_eval_rather_than_pass(tmp_path):
    gold = load_gold_set()
    case = _subset_case(gold)
    rows = [_row_for(case.supported_reactions[0]), _UNATTRIBUTED]
    text = render_text(score_run(_stage(tmp_path, case, rows), gold))

    line = next(ln for ln in text.splitlines() if "zero unsupported retained reactions" in ln)
    assert "[NOT EVAL]" in line, line
    assert "PASS" not in line, line


# ---------------------------------------------------------------------------
# 2. What must NOT change.
# ---------------------------------------------------------------------------
def test_the_subset_suppression_is_intact_unattributed_rows_are_not_fabrications():
    """Merge-rule check: this card must not manufacture a FAIL to replace a false PASS."""

    report = validate_semantic_coverage(
        _case(), _payload([_MATCHED, _UNATTRIBUTED]), paper_text=PAPER
    )
    check = report.checks[CHECK_SUPPORTED_REACTIONS]
    assert not [f for f in check.findings if f.get("kind") == "unsupported_retained_reaction"]
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 0
    assert report.support["precision"] is None, "precision needs an exhaustive set"


def test_attribution_rate_and_recall_are_still_reported():
    report = validate_semantic_coverage(
        _case(), _payload([_MATCHED, _UNATTRIBUTED]), paper_text=PAPER
    )
    assert report.support["retained_reactions"] == 2
    assert report.support["true_positives"] == 1
    assert report.support["unattributed_reactions"] == 1
    assert report.support["attribution_rate"] == 0.5
    assert report.support["recall"] == 1.0
    summary = report.checks[CHECK_SUPPORTED_REACTIONS].summary
    assert "attribution rate" in summary and "recall 1/1" in summary


def test_every_row_matching_a_signature_is_a_measured_zero_not_a_withheld_one():
    """A subset set is not automatically unevaluable. When every retained row matched a
    quote-verified signature there ARE no unattributed rows, so "zero unsupported" is a
    measurement and the check stays applicable."""

    report = validate_semantic_coverage(_case(), _payload([_MATCHED]), paper_text=PAPER)
    check = report.checks[CHECK_SUPPORTED_REACTIONS]
    assert check.applicable is True
    assert check.ok is True
    assert report.support["unsupported_verdict_evaluated"] is True
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 0


def test_an_exhaustive_signature_set_still_charges_unmatched_rows():
    report = validate_semantic_coverage(
        _case(supported_reactions_complete=True),
        _payload([_MATCHED, _UNATTRIBUTED]),
        paper_text=PAPER,
    )
    check = report.checks[CHECK_SUPPORTED_REACTIONS]
    assert check.applicable is True
    assert check.ok is False
    assert report.support["unsupported_verdict_evaluated"] is True
    assert report.scientific_errors[ERR_UNSUPPORTED_REACTIONS] == 1


@pytest.mark.parametrize("paper_id", ["PMC13231680", "PMC12180156"])
def test_the_negative_control_ceiling_still_counts_and_still_measures(tmp_path, paper_id):
    """The ONLY route by which priority 2 has ever returned a non-zero count. It needs no
    signature set, so a case carrying a ceiling is measurable even where the signature
    check is inapplicable -- and it must still fail loudly."""

    gold = load_gold_set()
    case = next(c for c in gold if c.paper_id == paper_id)
    assert case.max_retained_reactions is not None
    over = case.max_retained_reactions + 3

    run_dir = _stage(
        tmp_path,
        case,
        [dict(_UNATTRIBUTED, name=f"invented {i}", inputs=[f"a{i}"], outputs=[f"b{i}"])
         for i in range(over)],
    )
    report = score_run(run_dir, gold)
    entry = next(e for e in report.priorities() if e["rank"] == 2)

    assert report.errors.totals[ERR_UNSUPPORTED_REACTIONS] == over - case.max_retained_reactions
    assert paper_id in report.errors.papers_affected[ERR_UNSUPPORTED_REACTIONS]
    assert entry["ok"] is False, "a real violation still FAILS priority 2"
    assert entry["evaluated"] is True
    assert paper_id not in entry["not_evaluated_papers"], "the ceiling measured this paper"


def test_a_ceiling_case_within_its_ceiling_is_a_measured_zero(tmp_path):
    """The ceiling is what makes the negative controls measurable at all -- an
    at-or-under-ceiling control must be a real PASS, not a withheld verdict."""

    gold = load_gold_set()
    case = next(c for c in gold if c.paper_id == "PMC13231680")  # ceiling 0
    run_dir = _stage(tmp_path, case, [])

    report = score_run(run_dir, gold)
    result = next(p for p in report.papers if p.paper_id == case.paper_id)
    assert result.legs["strict"].semantic.support["unsupported_verdict_evaluated"] is True
    assert f"{case.paper_id}:strict" not in report.unmeasured_unsupported
    entry = next(e for e in report.priorities() if e["rank"] == 2)
    assert case.paper_id not in entry["not_evaluated_papers"]


def test_a_run_with_nothing_unmeasured_still_reports_a_real_pass(tmp_path):
    gold = load_gold_set()
    case = _subset_case(gold)
    # EVERY retained row matches a quote-verified signature, so the zero is measured.
    run_dir = _stage(tmp_path, case, [_row_for(s) for s in case.supported_reactions])

    entry = next(e for e in score_run(run_dir, gold).priorities() if e["rank"] == 2)
    assert entry["ok"] is True
    assert entry["evaluated"] is True
    assert entry["not_evaluated_papers"] == []


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _stage(tmp_path: Path, case: GoldCase, reactions) -> Path:
    """A one-leg run directory, as the batch driver writes it."""

    run_dir = tmp_path / "2026-01-01_0000"
    leg = run_dir / "papers" / case.paper_id / "strict"
    leg.mkdir(parents=True)
    (leg / "final_mapped.json").write_text(json.dumps(_payload(reactions)), encoding="utf-8")
    text = PAPER if case.paper_id == "PMC_C085" else _cached_text(case)
    (run_dir / "papers" / case.paper_id / "01_source_text.txt").write_text(text, encoding="utf-8")
    (run_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "paper_id": case.paper_id,
                "slug": case.paper_id,
                "mode": "strict",
                "status": "fail",
                "stage": "post_pipeline",
                "failure_kind": "contract",
                "issue_codes": [],
                "files": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _subset_case(gold) -> GoldCase:
    """A pinned case with signatures and NO ceiling -- the population priority 2 was blind
    to. All eight of them qualify; the first by id keeps the test deterministic."""

    cases = [c for c in gold if c.supported_reactions and c.max_retained_reactions is None]
    assert cases, "the pinned gold set no longer has a subset-signature case"
    return sorted(cases, key=lambda c: c.paper_id)[0]


def _row_for(signature) -> dict:
    """A retained reaction that matches ``signature`` -- a genuine true positive."""

    return {
        "name": signature.label or "matched",
        "inputs": [t.name for t in signature.inputs],
        "outputs": [t.name for t in signature.outputs],
        "evidence": signature.quote,
    }


def _cached_text(case: GoldCase) -> str:
    """Enough stored text for every gold quote of this case to verify."""

    return "\n".join(s.quote for s in case.supported_reactions) or "no supported reactions"
