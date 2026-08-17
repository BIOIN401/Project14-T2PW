"""C-041: release status as a first-class runtime classification.

Two kinds of test live here and they are labelled individually, because the merge
gate treats them differently:

**G9 REGRESSION** -- a pre-existing observable defect. Both reports rendered a
technical ``PASS`` with nothing beside it, so a reader took a gate result for a
release decision: PRODUCT_CONTRACT 11 forbids collapsing "pipeline execution
succeeded", "strict technical gates passed" and "semantic evaluation passed" into
one another, and printing one token for all three is exactly that collapse. These
fail on the dispatch base ``e4eeef4`` on RENDERED TEXT, not on a missing symbol --
which is why the expected wording is spelled out here instead of imported from
``release_status``, a module the base does not have.

**NEW ACCEPTANCE** -- ``t2pw.pipeline.release_status`` does not exist at the base,
so no base failure is claimed for these. Every import of it is function-local so
the whole file still COLLECTS at the base and the G9 tests above can run there.
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.batch.driver import RunOutcome, _finalize_gate_failure  # noqa: E402
from t2pw.batch.report import build_summary  # noqa: E402
from t2pw.bench.acceptance import (  # noqa: E402
    MODE_STRICT,
    AcceptanceReport,
    ModeResult,
    PaperResult,
)
from t2pw.bench.goldset import GoldCase  # noqa: E402
from t2pw.bench.render import render_text  # noqa: E402
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    DEFAULT_MIN_CORE_COVERAGE,
    evaluate_core_coverage,
)

#: The exact wording both reports must carry. Hard-coded, not imported: an
#: imported constant would turn the base run into an ImportError, and symbol
#: absence is not behavioural proof.
_ANTI_COLLAPSE = "a technical PASS is a GATE result, not a release decision"

#: D-002's reference case. Six requested anchors, one survives -> 0.167, below the
#: 0.5 floor. Its correct outcome is review_required PWML, never a refusal.
_PMC12096016_CORE = [
    "menaquinone",
    "chorismate",
    "isochorismate",
    "SEPHCHC",
    "DHNA-CoA",
    "1,4-dihydroxy-2-naphthoate",
]


def _pass_rows() -> List[Dict[str, Any]]:
    """A clean strict PASS + research PASS pair -- the case that used to read as
    "released" with nothing qualifying it."""

    return [
        {
            "paper_id": "PMC_CLEAN", "mode": mode, "status": "pass",
            "stage": "complete", "seconds": 1.0, "files": [], "counts": {},
        }
        for mode in ("strict", "research")
    ]


def _acceptance_report(**leg_extra: Any) -> AcceptanceReport:
    """A one-paper, one-leg acceptance report whose strict leg PASSED.

    PMC12452463 on purpose (TRAP-1 / PRODUCT_CONTRACT 13): the gold
    ``export_rationale`` records its route as chemically broken, so a strict PASS
    printed unqualified beside it is the most misleading line the report can emit.
    """

    case = GoldCase(
        paper_id="PMC12452463", title="enterobactin biosynthesis",
        requested_pathway="enterobactin biosynthesis",
        requested_organism="Escherichia coli", source_uri="u",
    )
    leg = ModeResult(
        paper_id=case.paper_id, mode=MODE_STRICT, attempted=True,
        status="pass", stage="pwml_export",
    )
    for name, value in leg_extra.items():
        setattr(leg, name, value)
    return AcceptanceReport(
        run_dir="r", gold_version="v", gold_path="p",
        papers=[PaperResult(case=case, slug="s", legs={MODE_STRICT: leg})],
    )


def _coverage(core: List[str], matched: List[str]):
    """A real coverage verdict from the real evaluator, never a hand-built dict."""

    admissions = [
        {"state": "core_accepted", "core_terms": [term]} for term in matched
    ] or [{"state": "auxiliary_accepted", "core_terms": []}]
    return evaluate_core_coverage({}, admissions, requested_core=core)


# ---------------------------------------------------------------------------
# G9 REGRESSION -- pre-existing collapse of PRODUCT_CONTRACT 11's five states.
# ---------------------------------------------------------------------------
def test_the_batch_summary_never_lets_a_technical_pass_read_as_release_ready() -> None:
    """**G9 REGRESSION.** Catches: ``build_summary`` printing ``PASS`` for a run
    whose semantic evaluation was never performed, with nothing recording either
    fact -- so the strict gate result, the pipeline result and the (absent)
    semantic result are one token. Fails on ``e4eeef4``: the per-paper block
    contains no release line at all."""

    block = build_summary(_pass_rows()).split("PER-PAPER BREAKDOWN", 1)[1]

    assert "release      :" in block
    assert _ANTI_COLLAPSE in block
    # Absent is reported as absent. It is never inferred from the PASS, from the
    # counts, or from which artifacts are on disk.
    assert "not recorded" in block
    assert block.count("release      :") == 2, "one per mode, strict and research"


def test_the_acceptance_report_never_lets_a_strict_pass_read_as_release_ready() -> None:
    """**G9 REGRESSION.** Catches: ``render_text`` printing ``strict : PASS`` as
    the leg's whole verdict, which is how a chemically broken route (PMC12452463)
    reads as released. Fails on ``e4eeef4``: no release line is emitted."""

    text = render_text(_acceptance_report())

    assert "release :" in text
    assert _ANTI_COLLAPSE in text
    assert "not recorded" in text


def test_both_reports_stay_inside_their_column_budget() -> None:
    """**G9 REGRESSION.** The line added above is long. Catches a release line
    that overflows the 100-column budget both reports are built to. Fails on
    ``e4eeef4`` only in the sense that the line does not exist -- so it is
    asserted here together with the line's presence, never alone."""

    from t2pw.batch.report import WIDTH

    summary = build_summary(_pass_rows())
    assert _ANTI_COLLAPSE in summary
    for line in summary.splitlines():
        assert len(line) <= WIDTH, line
    # Only the lines this card emits: ``_blockers`` already prints a 125-column
    # label from ``bench.metrics.BLOCKER_SCOPE_LABELS``, which is a pre-existing
    # defect in a section C-041 does not own and must not be silently adopted.
    rendered = [line for line in render_text(_acceptance_report()).splitlines() if "release" in line]
    assert rendered, "no release line was emitted"
    for line in rendered:
        assert len(line) <= 100, line


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- pipeline/release_status.py, which the base does not have.
# ---------------------------------------------------------------------------
def test_the_coverage_verdict_is_serialization_identical_to_the_dict_it_replaces() -> None:
    """**NEW ACCEPTANCE.** The return-shape change must be invisible on the wire:
    ``quarantine_coverage`` is pinned in ``evidence/c011_freeze_seam_before.json``
    over 39 legs and in the replay baseline, and neither may move."""

    from t2pw.pipeline.release_status import CoverageVerdict

    verdict = _coverage(_PMC12096016_CORE, ["menaquinone"])
    plain = {key: value for key, value in verdict.items()}

    assert isinstance(verdict, CoverageVerdict) and isinstance(verdict, dict)
    assert verdict == plain
    assert list(verdict) == list(plain), "key order too, not just membership"
    assert json.dumps(verdict, sort_keys=True) == json.dumps(plain, sort_keys=True)
    assert json.dumps(verdict) == json.dumps(plain)
    # Survives every trip a coverage report actually takes.
    assert copy.deepcopy(verdict) == plain
    assert pickle.loads(pickle.dumps(verdict)) == plain
    assert json.loads(json.dumps(verdict)) == plain


def test_the_reference_case_below_the_coverage_floor_is_review_required() -> None:
    """**NEW ACCEPTANCE.** D-002's reference case, PMC12096016 at 0.167 after the
    C-010 fix. Below the floor is not a refusal: the threshold blocks
    RELEASE-READY status, not PWML production."""

    from t2pw.pipeline.release_status import REVIEW_REQUIRED, classify_release_status

    verdict = _coverage(_PMC12096016_CORE, ["menaquinone"])
    assert round(verdict.coverage_ratio, 3) == 0.167
    assert verdict.below_coverage_minimum is True
    assert verdict.minimum_core_satisfied is False

    status = classify_release_status(verdict, strict_gates_passed=True)

    assert status.status == REVIEW_REQUIRED
    assert status.produced_pwml is True, "the fragment is exported, not dropped"
    assert status.strict_acceptance_eligible is False, "and never strict success"
    # D-002's required record: completeness, missing anchors, and why.
    assert status.completeness == verdict.coverage_ratio
    assert len(status.missing_anchors) == 5
    assert any("requested_core_coverage_below_minimum" in r for r in status.reasons)


def test_review_required_is_never_eligible_for_the_strict_denominator() -> None:
    """**NEW ACCEPTANCE.** TRAP-1 / PRODUCT_CONTRACT 13: PMC12452463's route is
    chemically broken, so its correct outcome is ``review_required`` with
    ``strict_acceptance_eligible=false``. No input combination may produce a
    ``review_required`` that is strict-eligible."""

    from t2pw.pipeline.release_status import (
        RELEASE_READY,
        REVIEW_REQUIRED,
        classify_release_status,
    )

    cases = [
        classify_release_status(_coverage(_PMC12096016_CORE, ["menaquinone"]), strict_gates_passed=True),
        classify_release_status(None, strict_gates_passed=True),
        classify_release_status(_coverage([], []), strict_gates_passed=True),
        classify_release_status(_coverage([], []), strict_gates_passed=False),
        classify_release_status(_coverage([], []), strict_gates_passed=True, pipeline_executed=False),
        classify_release_status(
            _coverage([], []), strict_gates_passed=True, serializable_without_invention=False
        ),
    ]
    for status in cases:
        assert status.strict_acceptance_eligible == (status.status == RELEASE_READY)
        if status.status == REVIEW_REQUIRED:
            assert status.strict_acceptance_eligible is False


def test_semantic_evaluation_that_was_not_performed_is_never_false() -> None:
    """**NEW ACCEPTANCE.** PRODUCT_CONTRACT 11: ``not_evaluated`` is never
    ``False``. It is a third value with a stated reason, it is not a pass, and it
    does not by itself refuse the run."""

    from t2pw.pipeline.release_status import (
        SEMANTIC_NOT_EVALUATED,
        classify_release_status,
    )

    status = classify_release_status(_coverage([], []), strict_gates_passed=True)

    assert status.semantic_evaluation == SEMANTIC_NOT_EVALUATED
    assert status.semantic_evaluation is not False
    assert status.semantic_confirmed is False, "not evaluated is not a pass"
    assert status.semantic_not_evaluated_reason
    # The four other states of section 11 are separately readable, never merged.
    assert status.pipeline_executed is True
    assert status.strict_gates_passed is True
    assert status.to_dict()["semantic_evaluation"] == SEMANTIC_NOT_EVALUATED


def test_only_a_graph_with_nothing_left_is_diagnostic_only() -> None:
    """**NEW ACCEPTANCE.** The product invariant: an incomplete-but-correct
    pathway is exported as ``review_required``, never deleted. ``diagnostic_only``
    is reserved for no defensible connected core."""

    from t2pw.pipeline.release_status import (
        DIAGNOSTIC_ONLY,
        REVIEW_REQUIRED,
        classify_release_status,
    )

    empty = evaluate_core_coverage({}, [])
    assert empty.has_surviving_core is False
    assert classify_release_status(empty, strict_gates_passed=True).status == DIAGNOSTIC_ONLY

    # One surviving process and zero matched anchors is still a fragment.
    shallow = _coverage(_PMC12096016_CORE, [])
    assert shallow.surviving_processes == 1 and shallow.coverage_ratio == 0.0
    assert shallow.has_surviving_core is True
    assert classify_release_status(shallow, strict_gates_passed=True).status == REVIEW_REQUIRED


def test_an_undeclared_core_reports_no_completeness_rather_than_zero() -> None:
    """**NEW ACCEPTANCE.** "nothing requested was found" and "nothing was
    requested" are opposite facts that the serialized ``coverage_ratio`` reports
    identically as ``0.0``. The wire value cannot change (pinned); the verdict
    draws the distinction instead."""

    undeclared = evaluate_core_coverage({}, [{"state": "core_accepted"}])
    declared_miss = _coverage(_PMC12096016_CORE, [])

    assert undeclared.declared is False
    assert undeclared["coverage_ratio"] == 0.0, "the WIRE value is unchanged"
    assert undeclared.completeness is None, "but unjudgeable is not zero"
    assert declared_miss.declared is True
    assert declared_miss.completeness == 0.0


def test_the_coverage_threshold_value_is_not_moved() -> None:
    """**NEW ACCEPTANCE.** D-002: "The threshold value does not move." Pinned so a
    later card cannot raise the release rate by lowering the bar."""

    assert DEFAULT_MIN_CORE_COVERAGE == 0.5
    verdict = _coverage(_PMC12096016_CORE, ["menaquinone"])
    assert verdict.min_core_coverage == 0.5
    assert verdict["thresholds"]["min_core_coverage"] == 0.5


def test_a_gate_blocked_leg_is_classified_and_the_row_now_carries_it() -> None:
    """**NEW ACCEPTANCE** (C-041) for the classification; **RE-PIN** (C-053) for the row.

    ``_finalize_gate_failure`` answers two questions -- "did the leg complete?"
    and "is anything releasable?" -- instead of leaving the second to be
    re-derived from a missing ``pathway.pwml``. That half is unchanged.

    **RE-PIN, merge rule 4, exact delta.** This test asserted
    ``"release_status" not in row`` and ``sorted(row) == sorted(before)``. Both
    pinned an INTERIM state that C-041 recorded as interim in the source it
    tests: ``_finalize_gate_failure``'s docstring says the classification "is NOT
    added to ``RunOutcome.to_dict``" because that "would edit ``RunOutcome``,
    which is outside C-041's boundary", and that "promoting it into the row
    belongs to the card that owns the row and the artifact names (D-004 /
    C-053)". D-004 is that decision and this is that card.

    The delta is **exactly one key**, ``release_status``, holding the serialized
    form of the classification this test already asserts field by field. That
    "and nothing else" is asserted below as a set difference rather than
    described in prose, so it cannot rot: no other key appears, none disappears,
    and ``pwml_artifact`` stays absent because this leg wrote no artifact.
    """

    from t2pw.pipeline.release_status import (
        DIAGNOSTIC_ONLY,
        REASON_STRICT_GATES_BLOCKED,
        SEMANTIC_NOT_EVALUATED,
    )

    outcome = RunOutcome(paper_id="PMC1", mode="strict")
    before = copy.deepcopy(outcome.to_dict())
    _finalize_gate_failure(
        outcome,
        verdict=types.SimpleNamespace(
            stage="post_normalization_hard_gates", reason="r", source="gate", failed=True
        ),
        blocking_issues=[object()], codes=["gate.forbidden_complex_reference"],
        code_lines=[], joined="joined", crashed=False, stage3_gate={}, gate_fail={},
    )
    status = outcome.release_status

    assert status.status == DIAGNOSTIC_ONLY
    assert status.reasons == (REASON_STRICT_GATES_BLOCKED,)
    # A technical block, stated as one. It does not claim "no defensible core".
    assert status.pipeline_executed is True
    assert status.strict_gates_passed is False
    assert status.semantic_evaluation == SEMANTIC_NOT_EVALUATED
    assert status.strict_acceptance_eligible is False

    row = outcome.to_dict()
    assert row["release_status"] == status.to_dict()
    assert sorted(set(row) - set(before)) == ["release_status"], "exactly one key is added"
    assert set(before) - set(row) == set(), "and none is taken away"
    # No PWML was written, so D-004's second key stays absent: the artifact name
    # is recorded only when an artifact was actually named.
    assert "pwml_artifact" not in row


def test_both_reports_print_a_classification_the_run_actually_carries() -> None:
    """**NEW ACCEPTANCE.** The rendering is not decorative: when the row or the leg
    carries a classification it is printed verbatim, with section 11's states
    still apart -- and it is read from the structured status, never from a
    filename."""

    from t2pw.pipeline.release_status import REVIEW_REQUIRED, classify_release_status

    status = classify_release_status(
        _coverage(_PMC12096016_CORE, ["menaquinone"]), strict_gates_passed=True
    )
    rows = _pass_rows()
    rows[0]["release_status"] = status.to_dict()
    rows[1]["release_status"] = "diagnostic_only"

    summary = build_summary(rows)
    assert f"release      : {REVIEW_REQUIRED}" in summary
    assert "semantic evaluation NOT" in summary
    assert "release      : diagnostic_only" in summary
    assert _ANTI_COLLAPSE not in summary, "the fallback note is for absent statuses"

    text = render_text(_acceptance_report(release_status=status.to_dict()))
    assert f"release : {REVIEW_REQUIRED}" in text
    assert "strict gates passed" in text
