"""C-056b: the runtime semantic verdict reaches the benchmark, in ONE direction.

**Every module-level import here exists at the base SHA `01bb7ef`**, deliberately, so
this file COLLECTS AND RUNS at base. That is what makes the G9 arm below a behavioural
proof rather than a collection error: the base measurement records a real assertion
failure with the new code absent, not an ``ImportError``. Where a test exercises a symbol
that does not exist at base it says so in its own header and claims no base failure.

The hazard this file exists to hold shut
----------------------------------------
``release_status.semantic_evaluation`` is now populated at the quarantine seam, and its
``passed`` value is **not** evidence that a pathway is semantically right. The gating set
is closed, but ``CHECK_RAG_REINTRODUCTION`` is structurally unevaluable there
(``quarantine_and_close`` takes no ``admission`` parameter, D-042 section 4) and
``CHECK_ANCHORS`` / ``CHECK_ORGANISM`` evaluate only under a derivation that supplies
them. **Measured on all 32 committed payload legs**
(``docs/pwml_recovery_sprint/evidence/c056b_s0_measured.json``): under the seam's own
``pathway_context`` derivation every one of the 32 had exactly ONE evaluable gating check,
and 25 of them answered ``passed`` on that single check. Under a request-carrying
derivation 31 of 32 reached three; **none ever reached the whole set**. Those counts
were measured BEFORE C-071 took the set from four to five and are left as recorded.

So the rule this file locks is: the runtime verdict may REMOVE a semantic confirmation
and may never ADD one. ``bench.semantic``'s ``confirmed`` -- which already requires that
every check was evaluable -- stays the sole source of a semantic pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.bench.acceptance import (  # noqa: E402
    MODE_RESEARCH,
    MODE_STRICT,
    AcceptanceReport,
    ModeResult,
    PaperResult,
    _build_denominators,
)
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.bench.metrics import DENOM_RESEARCH_CONFIRMED, DENOM_SEMANTIC, DENOM_STRICT  # noqa: E402
from t2pw.bench.semantic import (  # noqa: E402
    CHECK_ID_CONFLICT,
    CHECK_SOURCE_CARRIER,
    CheckResult,
    SemanticReport,
)
from t2pw.bench import semantic_production as sp  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    REVIEW_REQUIRED,
    SEMANTIC_FAILED,
    SEMANTIC_GATING_CHECKS,
    SEMANTIC_NOT_EVALUATED,
    SEMANTIC_PASSED,
    classify_release_status,
    semantic_verdict,
)

GOLD = load_gold_set()

#: Three ``core``-relevance cases. **`_STRICT_A` and `_STRICT_B` are NO LONGER
#: `strict_exportable`.** D-065 (LOCKED, 2026-08-25) reconciled PMC12421875 and
#: PMC12657337 to ``partial_only``: both are deliberate ORGANISM TRAPS whose Stage-0
#: conflict D-062 forbids from ever exporting strict, so the gold field that put them
#: in the strict denominator contradicted the ruling. They remain ``core``-relevance
#: cases and remain in the SEMANTIC denominator, which is why every test below that
#: uses them still holds -- only the STRICT population changed.
_CASES = {case.paper_id: case for case in GOLD}
_STRICT_A = "PMC12657337"
_STRICT_B = "PMC12421875"
_STRICT_C = "PMC12096016"
_PARTIAL = "PMC12444477"

#: The two papers that ARE still ``strict_exportable`` after D-065, and therefore the
#: only ones the STRICT denominator can contain. Used by the strict guard alone.
_STRICT_EXPORTABLE_A = "PMC12096016"
_STRICT_EXPORTABLE_B = "PMC12782028"


def _confirmed_semantic(paper_id: str, mode: str, *, confirmed: bool = True) -> SemanticReport:
    """A gold-path report whose ``confirmed`` is exactly what the caller asked for.

    Built from the public dataclass rather than by running the validator on purpose: the
    GOLD predicate (``bench/semantic.py``) is out of this card's boundary and must be held
    FIXED while the runtime input varies. An empty applicable-check set makes ``ok`` and
    ``confirmed`` both True by the definitions at ``semantic.py:284-299``; one failing
    check makes both False.
    """

    checks: Dict[str, CheckResult] = {}
    if not confirmed:
        checks[CHECK_ID_CONFLICT] = CheckResult(
            name=CHECK_ID_CONFLICT, ok=False, summary="held FALSE by this fixture")
    report = SemanticReport(paper_id=paper_id, mode=mode, checks=checks)
    report.graph = {"n_reactions": 3}
    return report


def _record(evaluation: str, failed: Optional[List[str]] = None,
            *, eligible: bool = False) -> Dict[str, Any]:
    """A manifest-shaped ``release_status`` row, written by hand.

    Hand-written because the point is to feed the SCORER the bytes a manifest carries.
    Only keys that exist in schema 4 are used unless a test says otherwise, so this helper
    itself is base-compatible.
    """

    row: Dict[str, Any] = {
        "status": REVIEW_REQUIRED,
        "pipeline_executed": True,
        "strict_gates_passed": True,
        "semantic_evaluation": evaluation,
        "semantic_not_evaluated_reason": "" if evaluation != SEMANTIC_NOT_EVALUATED else "unwired",
        "strict_acceptance_eligible": eligible,
    }
    if failed is not None:
        row["semantic_failed_checks"] = list(failed)
    return row


def _leg(paper_id: str, mode: str, *, confirmed: bool = True,
         record: Optional[Dict[str, Any]] = None) -> ModeResult:
    leg = ModeResult(paper_id=paper_id, mode=mode, attempted=True)
    leg.status = "pass"
    leg.deliverable = True
    leg.semantic = _confirmed_semantic(paper_id, mode, confirmed=confirmed)
    leg.release_status = record
    return leg


def _scored(legs: Dict[str, Dict[str, ModeResult]]) -> AcceptanceReport:
    """Run the real ``_build_denominators`` over hand-built legs.

    No run directory and no disk fixture: the function under test takes an
    ``AcceptanceReport`` and a ``GoldSet``, and building the report by hand is what keeps
    this measurement about the denominator rule instead of about file layout.
    """

    report = AcceptanceReport(run_dir="", gold_version=GOLD.version, gold_path=GOLD.source_path)
    for case in GOLD:
        paper = PaperResult(case=case, slug=case.paper_id)
        paper.legs = dict(legs.get(case.paper_id, {}))
        report.papers.append(paper)
    _build_denominators(report, GOLD)
    return report


def _numerator(report: AcceptanceReport, key: str) -> List[str]:
    return sorted(report.denominators[key].numerator_names)


def _denominator(report: AcceptanceReport, key: str) -> List[str]:
    return sorted(report.denominators[key].denominator_names)


# ---------------------------------------------------------------------------
# G9 CORRECTION PROOF — this table FAILS BEHAVIOURALLY at base 01bb7ef.
# ---------------------------------------------------------------------------
def test_g9_a_runtime_refuted_leg_is_not_a_semantic_success() -> None:
    """G9 CORRECTION PROOF -- **fails at base `01bb7ef`, passes at the tip.**

    At base ``_build_denominators`` reads ``release_status`` for the STRICT flag only and
    never for semantics, so a leg the pipeline itself classified ``semantic_evaluation =
    "failed"`` is counted as a semantic success on the strength of a re-scoring that
    disagrees with the product's own record. PRODUCT_CONTRACT 11 puts the semantic states
    on the runtime classification; a benchmark that overrides it UPWARD scores a pathway
    the product declined to vouch for.

    The failure at base is an assertion failure with every symbol present -- ``.semantic``,
    ``release_status``, ``_build_denominators`` and ``DENOM_SEMANTIC`` all exist there --
    not a collection error. Symbol absence is never proof and none is offered.
    """

    refuted = _record(SEMANTIC_FAILED, [SEMANTIC_GATING_CHECKS[0], SEMANTIC_GATING_CHECKS[2]])
    report = _scored({_STRICT_A: {MODE_STRICT: _leg(_STRICT_A, MODE_STRICT, record=refuted)}})

    # It is still in the POPULATION: a refusal is a miss, not an exclusion. Removing it
    # from the denominator would raise the rate by deleting the evidence against it.
    assert _STRICT_A in _denominator(report, DENOM_SEMANTIC)
    assert _STRICT_A not in _numerator(report, DENOM_SEMANTIC)


def test_g9_a_runtime_refuted_research_leg_is_not_scientifically_confirmed() -> None:
    """G9 CORRECTION PROOF -- **fails at base `01bb7ef`, passes at the tip.**

    The research-confirmed denominator carried the identical defect and is corrected on
    the identical grounds. Held separate from the semantic denominator because the two are
    different questions over different populations and the sprint's standing rule is that
    a rate is never merged into another.
    """

    refuted = _record(SEMANTIC_FAILED, [SEMANTIC_GATING_CHECKS[1]])
    report = _scored({_PARTIAL: {MODE_RESEARCH: _leg(_PARTIAL, MODE_RESEARCH, record=refuted)}})

    assert _PARTIAL in _denominator(report, DENOM_RESEARCH_CONFIRMED)
    assert _PARTIAL not in _numerator(report, DENOM_RESEARCH_CONFIRMED)


# ---------------------------------------------------------------------------
# REGRESSION GUARDS — these pass at base AND at tip. Not G9 proofs.
# ---------------------------------------------------------------------------
def test_regression_guard_a_runtime_pass_never_creates_a_semantic_success() -> None:
    """REGRESSION GUARD (passes at base and tip) -- not a G9 proof.

    The half of the rule that must NOT change, and the half a later edit is most likely to
    break: a runtime ``passed`` adds nothing. Measured, ``passed`` at this seam means "the
    one gating check that could run, ran clean" on 32 of 32 committed legs; ORing it into
    the numerator would turn a one-of-four pass into a benchmark success.

    Self-discriminating: the leg's GOLD verdict is deliberately NOT confirmed, so any edit
    that lets the runtime pass contribute turns this red, at base or at tip.
    """

    passing = _record(SEMANTIC_PASSED, [])
    report = _scored({_STRICT_B: {MODE_STRICT: _leg(_STRICT_B, MODE_STRICT,
                                                    confirmed=False, record=passing)}})

    assert _STRICT_B in _denominator(report, DENOM_SEMANTIC)
    assert _STRICT_B not in _numerator(report, DENOM_SEMANTIC)


def test_regression_guard_not_evaluated_and_an_absent_record_never_demote() -> None:
    """REGRESSION GUARD (passes at base and tip) -- not a G9 proof.

    PRODUCT_CONTRACT 11: ``not_evaluated`` is never ``false``. The over-firing guard for
    the correction above -- a subtraction that fired on anything except an affirmative
    ``failed`` would silently delete successes, which is the same defect as inflation with
    its sign flipped. Three non-firing inputs, all of which MUST still count:
    ``not_evaluated``, a ``passed`` record, and no record at all (every historical run --
    measured, 0 of 143 committed manifest rows carry a ``release_status``).
    """

    report = _scored({
        _STRICT_A: {MODE_STRICT: _leg(_STRICT_A, MODE_STRICT,
                                      record=_record(SEMANTIC_NOT_EVALUATED))},
        _STRICT_B: {MODE_STRICT: _leg(_STRICT_B, MODE_STRICT,
                                      record=_record(SEMANTIC_PASSED, []))},
        _STRICT_C: {MODE_STRICT: _leg(_STRICT_C, MODE_STRICT, record=None)},
    })

    numerator = _numerator(report, DENOM_SEMANTIC)
    for paper_id in (_STRICT_A, _STRICT_B, _STRICT_C):
        assert paper_id in numerator, paper_id


def test_regression_guard_the_strict_denominator_stays_affirmative_and_unmoved() -> None:
    """REGRESSION GUARD (passes at base and tip) -- not a G9 proof.

    The strict rate is gated on an AFFIRMATIVE ``strict_acceptance_eligible is True``
    (D-038 section 3) and this card does not touch that. It is asserted here because the
    strict numerator is the one figure T-100 quotes: a runtime semantic verdict must not
    reach it in EITHER direction, and the classifier already guarantees the failing
    direction by capping at ``review_required`` -- see the invariant test below.
    """

    passing = _record(SEMANTIC_PASSED, [], eligible=False)
    eligible = _record(SEMANTIC_NOT_EVALUATED, eligible=True)
    report = _scored({
        _STRICT_EXPORTABLE_A: {
            MODE_STRICT: _leg(_STRICT_EXPORTABLE_A, MODE_STRICT, record=passing)},
        _STRICT_EXPORTABLE_B: {
            MODE_STRICT: _leg(_STRICT_EXPORTABLE_B, MODE_STRICT, record=eligible)},
    })

    # A runtime semantic pass is not a strict success; an affirmative flag is.
    assert _STRICT_EXPORTABLE_A not in _numerator(report, DENOM_STRICT)
    assert _STRICT_EXPORTABLE_B in _numerator(report, DENOM_STRICT)

    # ANTI-VACUITY, added with the D-065 repair: the guard is only meaningful if the
    # strict population is non-empty and holds exactly the post-D-065 pair. Without
    # this, a future gold edit that emptied the denominator would make both
    # assertions above vacuously true rather than red.
    strict_population = _denominator(report, DENOM_STRICT)
    assert strict_population == [_STRICT_EXPORTABLE_A, _STRICT_EXPORTABLE_B], (
        strict_population)


def test_the_d065_trap_papers_are_excluded_by_expected_export_not_by_absence() -> None:
    """D-065 (LOCKED) behavioural pin. **Red at `91b5c50`, green at `116c8fa`.**

    Replaces two assertions that looked like this guard and were VACUOUS. They
    asserted the trap papers were absent from the strict population while giving them
    no legs -- so ``acceptance.py:977`` excluded them via ``strict_leg is None``
    whatever their ``expected_export`` said, and they passed before D-065 as well as
    after. REV-089 proved that by running the repaired file at ``91b5c50``.

    Here both trap papers get ATTEMPTED, ELIGIBLE strict legs -- identical in every
    respect to the control's -- so the ONLY thing that can exclude them is
    ``acceptance.py:969``'s ``expected_export != EXPORT_STRICT``. Before D-065 both
    were ``strict_exportable`` and this is red; after, both are ``partial_only``.

    D-062 forbids a Stage-0 organism conflict from ever exporting strict, and D-065
    reconciled the gold to match. This is the behavioural lock on that reconciliation.
    """

    eligible = _record(SEMANTIC_NOT_EVALUATED, eligible=True)
    report = _scored({
        _STRICT_A: {MODE_STRICT: _leg(_STRICT_A, MODE_STRICT, record=eligible)},
        _STRICT_B: {MODE_STRICT: _leg(_STRICT_B, MODE_STRICT, record=eligible)},
        _STRICT_EXPORTABLE_A: {
            MODE_STRICT: _leg(_STRICT_EXPORTABLE_A, MODE_STRICT, record=eligible)},
    })
    population = _denominator(report, DENOM_STRICT)

    # Attempted, eligible, deliverable -- and still excluded. Only expected_export
    # can do that, which is exactly what D-065 changed.
    assert _STRICT_A not in population, population
    assert _STRICT_B not in population, population

    # THE CONTROL that makes the two above non-vacuous: a paper that IS still
    # strict_exportable, given a byte-identical leg, IS in the population. Without
    # this, a rule that excluded everything would satisfy the assertions above.
    assert _STRICT_EXPORTABLE_A in population, population


def test_regression_guard_a_non_gating_check_still_cannot_demote() -> None:
    """REGRESSION GUARD (passes at base and tip) -- not a G9 proof.

    **The recorded ruling on D-039 section 3's two revisited checks: both STAY
    NON-GATING**, and this is the behavioural lock on that decision rather than a comment.

    Measured over the 32 committed payload legs
    (``docs/pwml_recovery_sprint/evidence/c056b_s0_measured.json``):

    * ``CHECK_SOURCE_CARRIER`` fails on **1 of 32**, and that one is a RESEARCH leg
      (``runs_verify/2026-08-04_1358/papers/PMC12096016/research``) -- the mode where the
      release record is a finding that is applied to nothing. It also documents itself as
      *"Hygiene only ... Deliberately does NOT claim the reaction is supported"*, so
      blocking a biological release on it would use it against its own stated meaning.
    * ``CHECK_CONNECTED_CORE`` fails on **0 of 32**, and duplicates a floor
      ``evaluate_core_coverage`` already enforces at this same seam
      (``core_process_count_below_minimum``), so gating it would double-count a refusal
      the coverage verdict has already made.

    Gating either buys **at most one demotion, in research mode, on a hygiene check** --
    no measured biological benefit against a real misuse cost. They stay recorded and
    non-gating.
    """

    payload = {
        "entities": {"compounds": [{"name": "chorismate"}, {"name": "isochorismate"}]},
        "processes": {"reactions": [
            {"name": "chorismate to isochorismate", "inputs": [{"name": "chorismate"}],
             "outputs": [{"name": "isochorismate"}]},
        ]},
    }
    report = sp.evaluate_production_semantics(payload, min_connected_reactions=1)

    # The premise: the hygiene check really does fail on this payload.
    assert report.checks[CHECK_SOURCE_CARRIER].ok is False
    assert CHECK_SOURCE_CARRIER in report.failed_checks
    # ...and the verdict is untouched by it, because it does not gate.
    # C-056c widened this return from three values to four. Arity only: every
    # assertion below is C-056b's, unchanged, and still about the SUBTRACTIVE
    # design -- a non-gating check cannot demote the verdict.
    state, _reason, failed, _evaluability = semantic_verdict(report)
    assert state != SEMANTIC_FAILED, (state, failed)
    assert CHECK_SOURCE_CARRIER not in SEMANTIC_GATING_CHECKS
    assert not failed


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE — ``semantic_failed_checks`` does not exist at base.
# ---------------------------------------------------------------------------
def test_new_acceptance_the_failed_gating_checks_are_persisted_by_name() -> None:
    """NEW ACCEPTANCE -- ``semantic_failed_checks`` is absent from ``ReleaseStatus`` at
    base, so no base failure is claimed and none is offered as a G9 proof.

    D-039 section 5 deferred this here. Before it the names survived only inside a
    ``reasons`` entry -- ``semantic_evaluation_failed:a,b`` -- so any consumer wanting them
    had to split a prose string on two separators. The schema bump 4 -> 5 exists for
    exactly this key.
    """

    names = [SEMANTIC_GATING_CHECKS[0], SEMANTIC_GATING_CHECKS[3]]
    status = classify_release_status(
        {"surviving_processes": 3, "declared_terms": 1, "matched_terms": 1,
         "coverage_ratio": 1.0, "reasons": []},
        strict_gates_passed=True,
        semantic_evaluation=SEMANTIC_FAILED,
        semantic_failed_checks=names,
    )

    assert status.semantic_failed_checks == tuple(names)
    assert status.to_dict()["semantic_failed_checks"] == names
    # The cap fired, and the pre-existing prose reason is unchanged beside the new field.
    assert status.status == REVIEW_REQUIRED
    assert status.strict_acceptance_eligible is False


def test_new_acceptance_the_names_are_coherent_with_the_verdict() -> None:
    """NEW ACCEPTANCE -- see the header above; no base failure claimed.

    Non-empty IFF the verdict is ``failed``. Self-discriminating in both directions:

    * a ``passed`` or ``not_evaluated`` verdict handed check names anyway records NONE, so
      a reader never sees a failing check beside "passed";
    * a ``failed`` verdict whose cap did NOT fire -- the run was already
      ``review_required`` on technical grounds -- still records them, so the names are
      tied to the verdict and not to the demotion.
    """

    names = list(SEMANTIC_GATING_CHECKS[:2])
    for evaluation in (SEMANTIC_PASSED, SEMANTIC_NOT_EVALUATED):
        incoherent = classify_release_status(
            None, strict_gates_passed=True,
            semantic_evaluation=evaluation, semantic_failed_checks=names,
        )
        assert incoherent.semantic_failed_checks == (), evaluation
        assert incoherent.to_dict()["semantic_failed_checks"] == [], evaluation

    # Already review_required on coverage grounds, so the cap has nothing to move.
    uncapped = classify_release_status(
        None, strict_gates_passed=True,
        semantic_evaluation=SEMANTIC_FAILED, semantic_failed_checks=names,
    )
    assert uncapped.status == REVIEW_REQUIRED
    assert uncapped.semantic_failed_checks == tuple(names)


def test_new_acceptance_the_scorer_reads_the_persisted_names() -> None:
    """NEW ACCEPTANCE -- ``ModeResult.runtime_semantic_*`` do not exist at base.

    The accessors, and the deliberate ASYMMETRY between them: there is a
    ``runtime_semantic_refuted`` and there is no affirmative twin. That absence is the
    structural half of the safeguard -- a later reader cannot turn a one-of-four pass into
    a numerator by reaching for the convenient property, because the property is not
    there.
    """

    names = [SEMANTIC_GATING_CHECKS[2]]
    leg = _leg(_STRICT_A, MODE_STRICT, record=_record(SEMANTIC_FAILED, names))
    assert leg.runtime_semantic_evaluation == SEMANTIC_FAILED
    assert leg.runtime_semantic_failed_checks == names
    assert leg.runtime_semantic_refuted is True

    for evaluation in (SEMANTIC_PASSED, SEMANTIC_NOT_EVALUATED):
        other = _leg(_STRICT_A, MODE_STRICT, record=_record(evaluation, []))
        assert other.runtime_semantic_refuted is False, evaluation

    absent = _leg(_STRICT_A, MODE_STRICT, record=None)
    assert absent.runtime_semantic_evaluation == ""
    assert absent.runtime_semantic_failed_checks == []
    assert absent.runtime_semantic_refuted is False

    assert not hasattr(ModeResult, "runtime_semantic_confirmed")
    assert not hasattr(ModeResult, "runtime_semantic_passed")
