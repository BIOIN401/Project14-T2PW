"""C-072 / F-094 -- a declared core with UNMATCHED ANCHORS is never ``release_ready``.

At T-104 (``runs_verify/2026-08-21_2239``, leg 17/20) ``PMC12452463/strict`` finished
``release_ready`` with ``strict_acceptance_eligible=true`` and emitted the reserved bare
``pathway.pwml`` -- while its own committed record said, in its own words, that three
requested-core anchors matched no admitted process and that admitting them "would require
unsupported biology". PRODUCT_CONTRACT 13 names the required outcome for that paper:
``review_required`` with ``strict_acceptance_eligible=false``, **never strict success**.

Why the coverage verdict could not catch it: ``evaluate_core_coverage``'s
``minimum_core_satisfied`` is ``not reasons``, and all three of its reasons are THRESHOLD
questions. The leg cleared every one of them -- coverage 0.8 against a pinned floor of 0.5,
seven surviving core processes against a floor of 1, a non-empty graph -- so the three
unmatched anchors were RECORDED (``unmatched_terms``) and never asked about.

**What these tests deliberately do NOT rely on.** At T-103 the same leg reached
``review_required`` only because C-071's semantic check ``actor_named_in_its_own_cited_span``
fired; at T-104 it did not, and nothing else held the leg. F-094's whole content is that the
contractual outcome was resting on one predicate. So every assertion below is made with
``semantic_evaluation == "passed"``, no failed checks, and every structural gate green. If
the protection here could only hold when the semantic gate fires, every test in this module
would fail.

**No import of the new reason constant at module scope, on purpose.** G9 requires the proof
to fail BEHAVIOURALLY on the base SHA, and a module-level import of a symbol that does not
exist there would abort collection instead -- symbol absence dressed up as a behavioural
failure. The prefix is restated as a literal here and reconciled with the module constant in
one separate test, which is the only test in this file that mentions the symbol at all.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    RELEASE_READY,
    REVIEW_REQUIRED,
    SEMANTIC_FAILED,
    SEMANTIC_PASSED,
    classify_release_status,
)

#: The reason prefix the demotion must append. Restated as a LITERAL rather than
#: imported -- see the module docstring.
REASON_PREFIX = "requested_core_anchors_unmatched"

#: The committed T-104 leg. Not a fixture: a real artifact from a real run.
T104_REPORT = (
    ROOT
    / "runs_verify"
    / "2026-08-21_2239"
    / "papers"
    / "PMC12452463"
    / "strict"
    / "quarantine_report.json"
)

#: PMC12452463's twelve matched anchors and the three that matched nothing, exactly as
#: ``coverage`` recorded them at T-104.
T104_MATCHED = [
    "chorismate",
    "isochorismate",
    "2,3-dihydro-2,3-dihydroxybenzoate",
    "2,3-dihydroxybenzoate",
    "enterobactin",
    "ferric-enterobactin",
    "EntC",
    "EntB",
    "EntE",
    "EntD",
    "EntF",
    "FepA",
]
T104_UNMATCHED = ["DHB-AMP", "Fur", "RyhB"]

T104_EXPANSION_BLOCKED = (
    "3 requested-core anchor(s) matched no admitted process: DHB-AMP, Fur, RyhB; "
    "candidate processes withheld by strict admission (quarantined_broken_reference:2); "
    "admitting them would require unsupported biology"
)


def _t104_coverage() -> Dict[str, Any]:
    """The T-104 coverage verdict, key for key, as ``evaluate_core_coverage`` wrote it.

    Every threshold clause is SATISFIED here and that is the point: 0.8 >= 0.5, seven
    core processes >= 1, a non-empty graph, therefore ``reasons == []`` and
    ``minimum_core_satisfied is True``. Nothing in this dict can demote the run except the
    three unmatched anchors.
    """

    return {
        "schema_version": 1,
        "requested_core_terms": T104_MATCHED[:4] + ["DHB-AMP"] + T104_MATCHED[4:] + ["Fur", "RyhB"],
        "requested_core_declared": True,
        "requested_core_source": "pathway_context",
        "matched_terms": list(T104_MATCHED),
        "unmatched_terms": list(T104_UNMATCHED),
        "coverage_ratio": 0.8,
        "core_accepted_processes": 7,
        "auxiliary_accepted_processes": 0,
        "surviving_processes": 7,
        "quarantined_processes": 2,
        "thresholds": {"min_core_processes": 1, "min_core_coverage": 0.5},
        "minimum_core_satisfied": True,
        "reasons": [],
    }


def _complete_coverage() -> Dict[str, Any]:
    """The same shape with NOTHING unmatched -- the control arm."""

    coverage = _t104_coverage()
    coverage["requested_core_terms"] = list(T104_MATCHED)
    coverage["matched_terms"] = list(T104_MATCHED)
    coverage["unmatched_terms"] = []
    coverage["coverage_ratio"] = 1.0
    return coverage


def _classify(coverage: Any, **overrides: Any) -> Any:
    """The production classifier with the T-104 technical outcome: everything green.

    ``semantic_evaluation="passed"`` is the DEFAULT here, not an option, because a default
    of anything else would let a test pass for the wrong reason.
    """

    kwargs: Dict[str, Any] = dict(
        pipeline_executed=True,
        strict_gates_passed=True,
        serializable_without_invention=True,
        retrieval_attempts=0,
        expansion_blocked_reason=T104_EXPANSION_BLOCKED,
        semantic_evaluation=SEMANTIC_PASSED,
        semantic_not_evaluated_reason="",
        semantic_failed_checks=(),
    )
    kwargs.update(overrides)
    return classify_release_status(coverage, **kwargs)


def _anchor_reasons(status: Any) -> list:
    return [reason for reason in status.reasons if reason.split(":", 1)[0] == REASON_PREFIX]


# ── 1. G9 behavioural proof ─────────────────────────────────────────────────


def test_the_t104_pmc12452463_shape_is_never_release_ready() -> None:
    """**G9 PROOF -- fails on base SHA 20e6b68, where this returns ``release_ready``.**

    The exact T-104 shape: pipeline executed, strict gates passed, serializable, seven
    surviving core processes, ``minimum_core_satisfied=True``, coverage 0.8, a DECLARED
    requested core, three unmatched anchors, semantics PASSED.

    PRODUCT_CONTRACT 13 for this paper: ``review_required`` with
    ``strict_acceptance_eligible=false``. Never strict success.
    """

    status = _classify(_t104_coverage())

    # The premise, asserted so a future failure cannot be misread as a premise drift.
    assert status.semantic_evaluation == SEMANTIC_PASSED
    assert status.semantic_failed_checks == ()
    assert status.pipeline_executed is True
    assert status.strict_gates_passed is True
    assert status.completeness == 0.8
    assert list(status.missing_anchors) == T104_UNMATCHED

    # The contract.
    assert status.status == REVIEW_REQUIRED
    assert status.strict_acceptance_eligible is False

    # And the record says WHICH anchors, not merely that something is missing.
    assert _anchor_reasons(status) == [f"{REASON_PREFIX}:DHB-AMP,Fur,RyhB"]


# ── 2. Non-vacuity: independent of the semantic gate, and not a blanket demotion ──


def test_the_demotion_does_not_depend_on_the_semantic_gate() -> None:
    """The adversarial arm. C-071's check is the single point of failure F-094 is about.

    Semantics passed, no failed checks, nothing in the record even mentions semantics --
    and the leg is still demoted. If the fix had been made by re-arming
    ``actor_named_in_its_own_cited_span``, this test could not pass.
    """

    status = _classify(_t104_coverage())

    assert status.semantic_evaluation == SEMANTIC_PASSED
    assert status.semantic_confirmed is True
    assert status.semantic_failed_checks == ()
    assert not [r for r in status.reasons if r.startswith("semantic_evaluation_failed")]
    assert status.status == REVIEW_REQUIRED
    assert status.strict_acceptance_eligible is False


def test_a_declared_core_with_every_anchor_matched_is_still_release_ready() -> None:
    """The converse. The new clause is a targeted cap, NOT a blanket demotion.

    Without this arm the module would pass just as happily against a classifier that had
    stopped returning ``release_ready`` at all -- which would destroy the strict
    denominator instead of correcting one leg.
    """

    status = _classify(_complete_coverage())

    assert status.status == RELEASE_READY
    assert status.strict_acceptance_eligible is True
    assert status.missing_anchors == ()
    assert _anchor_reasons(status) == []
    assert status.completeness == 1.0


def test_the_new_reason_is_a_named_module_constant_with_this_exact_value() -> None:
    """The greppable vocabulary entry, reconciled with the literal used above.

    The ONLY test here that touches the symbol. It is not the proof of anything
    behavioural -- G9 says symbol absence is not proof -- it exists so the reason string
    cannot drift away from the constant a consumer greps for.
    """

    from t2pw.pipeline import release_status as RS

    assert RS.REASON_REQUESTED_CORE_ANCHORS_UNMATCHED == REASON_PREFIX
    # Distinct from every other reason in the vocabulary, including the coverage-ratio
    # shortfall it must never be confused with.
    assert REASON_PREFIX != RS.COVERAGE_REASON_BELOW_MINIMUM
    assert REASON_PREFIX not in {
        RS.REASON_PIPELINE_DID_NOT_EXECUTE,
        RS.REASON_STRICT_GATES_BLOCKED,
        RS.REASON_SERIALIZATION_REQUIRES_INVENTION,
        RS.REASON_NO_DEFENSIBLE_CONNECTED_CORE,
        RS.REASON_COVERAGE_NOT_EVALUATED,
        RS.REASON_SEMANTIC_EVALUATION_FAILED,
    }


# ── 3. Preservation, not dropping (merge rule 7) ────────────────────────────


def test_the_demoted_run_is_preserved_and_never_discarded() -> None:
    """Merge rule 7. An incomplete-but-correct pathway is EXPORTED AND FLAGGED.

    ``review_required`` is "valid, needs review". The seven surviving processes, the
    completeness figure, the anchor list and the expansion reason all survive the
    demotion, and ``produced_pwml`` stays true -- ``diagnostic_only`` is the one state
    with no final PWML, and this must never reach it.
    """

    coverage = _t104_coverage()
    before = deepcopy(coverage)
    status = _classify(coverage)

    assert status.status == REVIEW_REQUIRED
    assert status.status != DIAGNOSTIC_ONLY
    assert status.produced_pwml is True

    # D-002's four items, all still on the record after the demotion.
    assert status.completeness == 0.8
    assert list(status.missing_anchors) == T104_UNMATCHED
    assert status.retrieval_attempts == 0
    assert status.expansion_blocked_reason == T104_EXPANSION_BLOCKED
    assert status.coverage_evaluated is True

    # The surviving processes are a fact about the graph, not about the verdict: the
    # classifier read the coverage verdict and did not edit it.
    assert coverage == before
    assert coverage["surviving_processes"] == 7
    assert coverage["core_accepted_processes"] == 7
    assert coverage["minimum_core_satisfied"] is True


# ── 4. One step only ────────────────────────────────────────────────────────


def test_an_already_diagnostic_only_run_is_untouched() -> None:
    """The cap never deepens a refusal the technical chain already reached."""

    for override in (
        {"pipeline_executed": False},
        {"strict_gates_passed": False},
        {"serializable_without_invention": False},
    ):
        status = _classify(_t104_coverage(), **override)
        assert status.status == DIAGNOSTIC_ONLY, override
        assert status.strict_acceptance_eligible is False
        assert _anchor_reasons(status) == [], override


def test_an_already_review_required_run_is_unchanged_and_gains_no_duplicate_reason() -> None:
    """A run the coverage threshold already demoted keeps exactly the record it had.

    The unmatched anchors are already implied there by the ratio shortfall; restating
    them would tell a reader two things happened when one did.
    """

    coverage = _t104_coverage()
    coverage["coverage_ratio"] = 0.2
    coverage["matched_terms"] = T104_MATCHED[:3]
    coverage["unmatched_terms"] = T104_MATCHED[3:] + T104_UNMATCHED
    coverage["minimum_core_satisfied"] = False
    coverage["reasons"] = ["requested_core_coverage_below_minimum:0.200<0.500"]

    status = _classify(coverage)

    assert status.status == REVIEW_REQUIRED
    assert status.strict_acceptance_eligible is False
    assert status.reasons == ("requested_core_coverage_below_minimum:0.200<0.500",)
    assert _anchor_reasons(status) == []
    # No reason is ever recorded twice, whatever path got here.
    assert len(set(status.reasons)) == len(status.reasons)


def test_a_semantic_failure_and_unmatched_anchors_together_still_move_one_step() -> None:
    """Two independent caps, one destination. Never ``diagnostic_only``, never doubled.

    The semantic cap runs first and takes the single step; the incomplete-core cap then
    sees a status that is already ``review_required`` and does nothing, so the record
    names the semantic failure once and does not restate it as a completeness one.
    """

    status = _classify(
        _t104_coverage(),
        semantic_evaluation=SEMANTIC_FAILED,
        semantic_failed_checks=("actor_named_in_its_own_cited_span",),
    )

    assert status.status == REVIEW_REQUIRED
    assert status.status != DIAGNOSTIC_ONLY
    assert status.strict_acceptance_eligible is False
    assert (
        "semantic_evaluation_failed:actor_named_in_its_own_cited_span" in status.reasons
    )
    assert _anchor_reasons(status) == []
    assert len(set(status.reasons)) == len(status.reasons)


# ── 5. The undeclared regime is out of scope by construction ────────────────


def test_an_undeclared_core_never_triggers_the_new_clause() -> None:
    """With nothing requested, relevance is unjudgeable and no anchor can be missing.

    ``completeness`` is ``None`` in this regime by design, and the adversarial part is the
    second case: a hand-built verdict that carries ``unmatched_terms`` while declaring no
    core must STILL not fire the clause, because the guard is the DECLARATION, not the
    emptiness of a list that ``evaluate_core_coverage`` happens to leave empty there.
    """

    undeclared = _t104_coverage()
    undeclared["requested_core_declared"] = False
    undeclared["requested_core_terms"] = []
    undeclared["matched_terms"] = []
    undeclared["unmatched_terms"] = []
    undeclared["coverage_ratio"] = 0.0

    status = _classify(undeclared)
    assert status.status == RELEASE_READY
    assert status.strict_acceptance_eligible is True
    assert status.completeness is None
    assert _anchor_reasons(status) == []

    contradictory = dict(undeclared, unmatched_terms=list(T104_UNMATCHED))
    contradicted = _classify(contradictory)
    assert contradicted.status == RELEASE_READY
    assert contradicted.completeness is None
    assert _anchor_reasons(contradicted) == []


def test_no_coverage_verdict_at_all_is_unaffected() -> None:
    """Coverage never evaluated is its own reason and stays exactly that."""

    status = _classify(None)

    assert status.status == REVIEW_REQUIRED
    assert status.coverage_evaluated is False
    assert "requested_core_coverage_not_evaluated" in status.reasons
    assert _anchor_reasons(status) == []


# ── 6. Real-artifact replay ─────────────────────────────────────────────────


@pytest.mark.skipif(
    not T104_REPORT.is_file(), reason="the T-104 verify run is not in this checkout"
)
def test_the_committed_t104_leg_replays_to_the_contract_outcome() -> None:
    """**G9 PROOF (second) -- the real committed artifact, not a hand-written shape.**

    The coverage verdict and the gate facts are read from
    ``runs_verify/2026-08-21_2239/papers/PMC12452463/strict/quarantine_report.json`` and
    pushed back through the PRODUCTION classifier. On the base SHA this replay reproduces
    the ``release_ready`` the file records. It must now reach PRODUCT_CONTRACT 13's
    outcome instead.
    """

    report = json.loads(T104_REPORT.read_text(encoding="utf-8"))
    coverage = report["coverage"]
    recorded = report["release"]

    # WHAT WENT WRONG, quoted from the artifact so the defect cannot quietly vanish.
    assert recorded["status"] == RELEASE_READY
    assert recorded["strict_acceptance_eligible"] is True
    assert recorded["missing_anchors"] == T104_UNMATCHED
    assert "would require unsupported biology" in recorded["expansion_blocked_reason"]

    # WHY NOTHING ELSE COULD HAVE CAUGHT IT: every existing gate is green.
    assert coverage["requested_core_declared"] is True
    assert coverage["minimum_core_satisfied"] is True
    assert coverage["reasons"] == []
    assert float(coverage["coverage_ratio"]) >= float(
        coverage["thresholds"]["min_core_coverage"]
    )
    assert recorded["semantic_evaluation"] == SEMANTIC_PASSED
    assert recorded["semantic_failed_checks"] == []
    assert recorded["pipeline_executed"] is True
    assert recorded["strict_gates_passed"] is True

    replayed = classify_release_status(
        coverage,
        pipeline_executed=bool(recorded["pipeline_executed"]),
        strict_gates_passed=bool(recorded["strict_gates_passed"]),
        # The recorded status was ``release_ready``, which the chain reaches only when
        # serialization needed no invention. Derived from the record, not assumed.
        serializable_without_invention=True,
        retrieval_attempts=recorded.get("retrieval_attempts"),
        expansion_blocked_reason=recorded.get("expansion_blocked_reason", ""),
        semantic_evaluation=recorded.get("semantic_evaluation"),
        semantic_not_evaluated_reason=recorded.get("semantic_not_evaluated_reason", ""),
        semantic_failed_checks=recorded.get("semantic_failed_checks", ()),
        semantic_check_evaluability=recorded.get("semantic_check_evaluability", ()),
    )

    assert replayed.status == REVIEW_REQUIRED
    assert replayed.strict_acceptance_eligible is False
    assert replayed.semantic_evaluation == SEMANTIC_PASSED
    assert list(replayed.missing_anchors) == T104_UNMATCHED
    assert _anchor_reasons(replayed) == [f"{REASON_PREFIX}:DHB-AMP,Fur,RyhB"]
    # Preserved, not dropped: there is still a PWML for a human to review.
    assert replayed.produced_pwml is True
