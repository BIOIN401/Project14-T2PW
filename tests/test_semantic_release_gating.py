"""C-056a: semantic evaluation reaches the runtime ``release_status`` (D-037/D-039/D-042).

PRODUCT_CONTRACT 11: *"Semantic checks must affect the runtime ``release_status``. Wiring
them only into benchmark denominators is insufficient."* Before this card
``SEMANTIC_FAILED`` was defined in ``release_status.py`` and **produced nowhere in
``src/``**, ``semantic_evaluation`` was hardcoded to ``not_evaluated``, and
``bench/semantic_production.py`` had **zero ``src/`` consumers** -- so a run could ship
``release_ready`` having never been semantically evaluated at all.

**This is NEW CAPABILITY, not a correction.** There is no runtime semantic verdict at the
base to regress, so per G9 this file carries an explicitly labelled new acceptance test and
**no fabricated base failure**. The preservation invariants are written as
**self-discriminating tables**: each holds both a case the new branch must fire on and a
case it must not, in one assertion loop, so deleting the gate *or* over-firing it turns the
same guard red.

D-042 struck D-039 section 4's predicted breakage as measurably false, so
``tests/test_strict_quarantine_release_seam.py`` is **not edited by this card at all** and
no baseline moves. The obligation survives in stronger form and is discharged by
``test_new_acceptance_all_three_states_remain_reachable_at_the_tip``.
"""

from __future__ import annotations

import ast
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench import semantic as _s  # noqa: E402
from t2pw.bench import semantic_production as sp  # noqa: E402
from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    REASON_SEMANTIC_EVALUATION_FAILED,
    RELEASE_READY,
    REVIEW_REQUIRED,
    SEMANTIC_FAILED,
    SEMANTIC_GATING_CHECKS,
    SEMANTIC_NO_GATING_CHECK_EVALUABLE,
    SEMANTIC_NOT_EVALUATED,
    SEMANTIC_PASSED,
    classify_release_status,
    semantic_verdict,
)
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.rag.admission import ORGANISM_MATCH, compare_organism  # noqa: E402

from test_strict_quarantine_contract_alignment import _base  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────────────


def _release(payload: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context)
    return dict(result.quarantine_report["release"])


def _checks(payload: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The semantic report the seam itself would compute, through the SAME pinned
    derivation, so this file can never assert about a request the seam does not use."""

    from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero

    result = quarantine_and_close(payload, strict_db=True, pathway_context=context)
    requested = pathway_context_from_stage_zero(context)
    return sp.evaluate_production_semantics(
        result.payload,
        requested_pathway=requested.requested_pathway,
        requested_organism=requested.organism,
        min_connected_reactions=1,
    ).checks


def _unexportable_payload() -> Dict[str, Any]:
    payload = _base()
    payload["entities"]["proteins"].append({"name": "GCLM"})
    payload["entities"]["protein_complexes"].append({
        "name": "glutamate-cysteine ligase complex",
        "species": "Homo sapiens",
        "components": [
            {"name": "glutamate-cysteine ligase", "stoichiometry": 1},
            {"name": "GCLM", "stoichiometry": 1},
        ],
    })
    payload["processes"]["reactions"][0]["enzymes"] = ["glutamate-cysteine ligase complex"]
    return payload


def _coverage_shortfall_payload() -> Dict[str, Any]:
    payload = _base()
    payload["metadata"].pop("key_compounds", None)
    return payload


#: Six anchors, one of which a surviving core process touches -> review_required.
_SHORTFALL_CONTEXT: Dict[str, Any] = {
    "key_compounds": [
        "L-glutamate", "ornithine", "citrulline", "argininosuccinate", "arginine", "fumarate",
    ]
}

#: A legitimate Stage-0 request for a pathway this payload is NOT about. The pinned
#: derivation reads ``pathway_name``, so this makes CHECK_ANCHORS -- a GATING check --
#: fail without touching a single technical gate.
_WRONG_PATHWAY_CONTEXT: Dict[str, Any] = {"pathway_name": "menaquinone biosynthesis"}

#: The same shape, naming the pathway the payload really is about.
_RIGHT_PATHWAY_CONTEXT: Dict[str, Any] = {"pathway_name": "glutathione biosynthesis"}


# ── NEW ACCEPTANCE ──────────────────────────────────────────────────────────


# NEW ACCEPTANCE — no runtime semantic verdict exists at the base
def test_new_acceptance_a_failing_gating_check_reaches_the_runtime_release_status() -> None:
    """PRODUCT_CONTRACT 11, the whole point of the card.

    A payload that passes every technical gate but is **not the pathway that was asked
    for** must no longer ship as ``release_ready``. At the base this was impossible to
    express: ``semantic_evaluation`` was the constant ``not_evaluated`` on every path and
    ``SEMANTIC_FAILED`` was produced nowhere in ``src/``.

    Self-discriminating: the right-request row must stay ``release_ready``, so a gate that
    over-fires fails this test just as surely as one that never fires.
    """

    wrong = _release(_base(), _WRONG_PATHWAY_CONTEXT)
    right = _release(_base(), _RIGHT_PATHWAY_CONTEXT)

    # The request the payload does not answer: semantics fail and the status is capped.
    assert wrong["semantic_evaluation"] == SEMANTIC_FAILED
    assert wrong["status"] == REVIEW_REQUIRED
    # TRAP-1 / PRODUCT_CONTRACT 13: a capped run is never strict success.
    assert wrong["strict_acceptance_eligible"] is False
    # The reason names the check, so a reviewer is told WHICH semantic fact failed.
    assert any(
        reason.startswith(REASON_SEMANTIC_EVALUATION_FAILED) for reason in wrong["reasons"]
    ), wrong["reasons"]
    assert _s.CHECK_ANCHORS in " ".join(wrong["reasons"])
    # ...and a semantic FAILURE carries no "not evaluated" reason. The two are different
    # states and PRODUCT_CONTRACT 11 forbids collapsing either into the other.
    assert wrong["semantic_not_evaluated_reason"] == ""

    # The discriminator: same payload, same seam, a request it DOES answer.
    assert right["semantic_evaluation"] == SEMANTIC_PASSED
    assert right["status"] == RELEASE_READY
    assert right["strict_acceptance_eligible"] is True
    assert not any(
        reason.startswith(REASON_SEMANTIC_EVALUATION_FAILED) for reason in right["reasons"]
    )


# NEW ACCEPTANCE — no runtime semantic verdict exists at the base
def test_new_acceptance_all_three_states_remain_reachable_at_the_tip() -> None:
    """**D-042 section 1**, which replaced D-039 section 4's struck ``_sourced_base()``.

    D-039 section 4 predicted wiring would demote the ``release_ready`` member of
    ``test_strict_quarantine_release_seam.py``'s reachability set, because ``_base()``'s
    reactions carry no source field. Measured, that mechanism cannot reach the gate:
    ``CHECK_SOURCE_CARRIER`` is NOT in :data:`SEMANTIC_GATING_CHECKS`, and a check that
    does not gate cannot demote. The prediction was struck as false and the seam test is
    **not edited by this card**.

    The duty that replaced it is discharged here: all three PRODUCT_CONTRACT 4 states are
    still reachable **through the wired seam**, and none of them got there by way of a
    semantic demotion.
    """

    outcomes = [
        _release(_base(), None),
        _release(_coverage_shortfall_payload(), _SHORTFALL_CONTEXT),
        _release(_unexportable_payload(), None),
    ]

    assert {row["status"] for row in outcomes} == {RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY}
    # And none of the three reached its state through the semantic cap -- otherwise the set
    # would be whole by coincidence rather than because the gate left it alone.
    for row in outcomes:
        assert not any(
            reason.startswith(REASON_SEMANTIC_EVALUATION_FAILED) for reason in row["reasons"]
        ), row


# NEW ACCEPTANCE — no runtime semantic verdict exists at the base
def test_new_acceptance_the_gating_set_is_closed_at_exactly_four() -> None:
    """**D-039 section 3.** The set is closed, and closure is asserted, not commented.

    ``release_status.py`` is ``t2pw.pipeline`` and restates the four names as literals
    rather than importing ``t2pw.bench`` at module level, which would invert the layering
    for every importer of that module. This is the test that keeps the restatement honest.
    """

    assert SEMANTIC_GATING_CHECKS == (
        _s.CHECK_ANCHORS,
        _s.CHECK_ORGANISM,
        _s.CHECK_ID_CONFLICT,
        _s.CHECK_RAG_REINTRODUCTION,
    )
    assert len(SEMANTIC_GATING_CHECKS) == 4
    assert len(set(SEMANTIC_GATING_CHECKS)) == 4

    # CLOSED: every other check ``evaluate_production_semantics`` populates is excluded,
    # by name, so adding a fifth gate silently is impossible.
    for excluded in (
        _s.CHECK_PLACEHOLDER_IDENTITY,   # PRODUCT_CONTRACT 13 / TRAP-3: non-adjudicating
        _s.CHECK_SOURCE_CARRIER,         # hygiene only; disclaims biological meaning
        _s.CHECK_CONNECTED_CORE,         # duplicates the coverage floor at this same seam
        _s.CHECK_SUPPORTED_REACTIONS,    # gold-only; always inapplicable in production
    ):
        assert excluded not in SEMANTIC_GATING_CHECKS, excluded

    # The gating set is a subset of the vocabulary, never a private dialect.
    for name in SEMANTIC_GATING_CHECKS:
        assert name in _s.CHECK_ORDER


# NEW ACCEPTANCE — no runtime semantic verdict exists at the base
def test_new_acceptance_the_seam_pins_exactly_one_request_derivation() -> None:
    """**D-042 section 2.** The verdict is a property of the DERIVATION, not the payload.

    The seam reads the request through ``entity_admission.pathway_context_from_stage_zero``
    -- ``pathway_name`` / ``likely_organism``|``organism`` -- and through nothing else.
    ``metadata.pathway_subject`` is a PathWhiz **category** ("Metabolic"), not a pathway
    name; deriving anchors from it fails CHECK_ANCHORS on essentially every payload and
    would demote ``_base`` for a reason that is a category label. This locks the choice so
    a later edit cannot quietly switch derivations.
    """

    payload = _base()
    assert payload["metadata"]["pathway_subject"] == "Metabolic"
    assert payload["metadata"]["pathway_name"] == "Glutathione biosynthesis"

    # No Stage-0 context: the derivation yields an EMPTY request, so the two
    # request-dependent checks report not-evaluated. It does NOT fall back to the
    # payload's own metadata -- a request read off the result is not a request.
    checks = _checks(payload, None)
    assert checks[_s.CHECK_ANCHORS].applicable is False
    assert checks[_s.CHECK_ORGANISM].applicable is False
    assert _release(_base(), None)["status"] == RELEASE_READY

    # The pinned keys are honoured...
    assert _checks(_base(), {"pathway_name": "glutathione biosynthesis"})[_s.CHECK_ANCHORS].ok is True
    # ...including the ``likely_organism`` alias the factory prefers.
    organism_check = _checks(_base(), {"likely_organism": "Homo sapiens"})[_s.CHECK_ORGANISM]
    assert organism_check.applicable is True

    # ...and ``pathway_subject`` is NOT one of them: a context carrying only a category
    # leaves the anchors check unevaluated rather than failing the run on a category.
    assert _checks(_base(), {"pathway_subject": "Metabolic"})[_s.CHECK_ANCHORS].applicable is False
    assert _release(_base(), {"pathway_subject": "Metabolic"})["status"] == RELEASE_READY


# NEW ACCEPTANCE — no runtime semantic verdict exists at the base
@pytest.mark.parametrize(
    "check, evaluability",
    [
        (_s.CHECK_ID_CONFLICT, "unconditional"),
        (_s.CHECK_ANCHORS, "conditional"),
        (_s.CHECK_ORGANISM, "conditional"),
        (_s.CHECK_RAG_REINTRODUCTION, "structurally_unevaluable"),
    ],
)
def test_new_acceptance_measured_evaluability_of_each_gating_check(check, evaluability) -> None:
    """**D-042 section 4: state it, do not ship "four live gates".**

    Three of the four are conditionally or structurally unevaluable at this seam, and a
    report that said only "no failures" would conceal the difference between *passed*,
    *not evaluated*, and *unevaluable by signature*. This pins each one.

    ``CHECK_RAG_REINTRODUCTION`` is the strong case: ``quarantine_and_close`` has no
    ``admission`` parameter at all, so reintroduction is undetectable here by construction.
    Giving it one is explicitly NOT granted to this card -- it is a signature change on a
    seam C-057 also touches. This is contract-compliant because ``not_evaluated`` is never
    ``false`` and produces no status change.
    """

    bare = _checks(_base(), None)
    supplied = _checks(_base(), {"pathway_name": "glutathione biosynthesis", "organism": "Homo sapiens"})

    if evaluability == "unconditional":
        # Evaluable with or without a Stage-0 context: it asks only about the payload.
        assert bare[check].applicable is True
        assert supplied[check].applicable is True
    elif evaluability == "conditional":
        # Not evaluable without the request; evaluable once the derivation supplies it.
        assert bare[check].applicable is False
        assert supplied[check].applicable is True
    else:
        # Unevaluable on BOTH, because no context can conjure an admission report.
        assert bare[check].applicable is False
        assert supplied[check].applicable is False
        assert "admission" in bare[check].inapplicable_reason.lower()


# NEW ACCEPTANCE — no runtime semantic verdict exists at the base
def test_new_acceptance_demotion_is_a_one_step_cap_never_a_move_to_diagnostic_only() -> None:
    """**D-042 section 3.** A failing gating check CAPS at ``review_required``.

    It never produces ``diagnostic_only`` -- PRODUCT_CONTRACT 13 calls ``review_required``
    "valid, needs review", exactly the state of a pathway whose semantics did not confirm,
    and merge rule 7 preserves incomplete-but-correct work rather than dropping it. It also
    never deepens a status the technical rules already reached, so a technical refusal is
    never restated as a biological one.

    Because a cap is monotone it can only ever REMOVE strict successes, never create one.
    """

    # release_ready -> review_required, exactly one step.
    assert _release(_base(), _WRONG_PATHWAY_CONTEXT)["status"] == REVIEW_REQUIRED

    # Already review_required: the semantic failure is recorded, the status does not deepen.
    shortfall = _coverage_shortfall_payload()
    shortfall_context = dict(_SHORTFALL_CONTEXT, pathway_name="menaquinone biosynthesis")
    capped = _release(shortfall, shortfall_context)
    assert capped["semantic_evaluation"] == SEMANTIC_FAILED
    assert capped["status"] == REVIEW_REQUIRED
    assert capped["status"] != DIAGNOSTIC_ONLY

    # Already diagnostic_only: a semantic failure cannot make it "more" refused, and the
    # technical reason stays the one on the record.
    broken = _release(_unexportable_payload(), _WRONG_PATHWAY_CONTEXT)
    assert broken["status"] == DIAGNOSTIC_ONLY
    assert "serialization_would_require_invention" in broken["reasons"]

    # No input of any kind can drive the classifier to diagnostic_only on semantics alone.
    for failed in ([], [_s.CHECK_ANCHORS], list(SEMANTIC_GATING_CHECKS)):
        status = classify_release_status(
            {"requested_core_declared": True, "surviving_processes": 3,
             "minimum_core_satisfied": True, "reasons": []},
            strict_gates_passed=True,
            semantic_evaluation=SEMANTIC_FAILED,
            semantic_failed_checks=failed,
        )
        assert status.status == REVIEW_REQUIRED
        assert status.strict_acceptance_eligible is False


# ── REGRESSION GUARDS ───────────────────────────────────────────────────────


# REGRESSION GUARD (passes at base and tip) — not a G9 proof
def test_regression_guard_a_non_gating_check_failure_never_demotes() -> None:
    """**D-039 section 3.** ``CHECK_SOURCE_CARRIER`` is recorded but must NOT gate.

    It documents itself as *"Hygiene only ... Deliberately does NOT claim the reaction is
    supported"*, so blocking a **biological** release on it would misuse it -- and it
    demonstrably over-fires: it is the single check that fails on all three members of the
    reachability set, which is exactly why D-039 section 4 mistakenly predicted a breakage.

    SELF-DISCRIMINATING, and the first row is asserted NON-VACUOUS: the check really does
    fail on this payload. If a later edit adds ``CHECK_SOURCE_CARRIER`` to the gating set,
    row 1 goes red; if it deletes the gate entirely, row 2 goes red.
    """

    table = (
        # payload,     context,                  failing check,          gates?  expected
        (_base(), None, _s.CHECK_SOURCE_CARRIER, False, RELEASE_READY),
        (_base(), _WRONG_PATHWAY_CONTEXT, _s.CHECK_ANCHORS, True, REVIEW_REQUIRED),
    )
    for payload, context, failing, gates, expected in table:
        checks = _checks(deepcopy(payload), context)
        # Non-vacuous: the named check is applicable AND actually failing on this row.
        assert checks[failing].applicable is True, failing
        assert checks[failing].ok is False, failing
        assert (failing in SEMANTIC_GATING_CHECKS) is gates, failing

        release = _release(deepcopy(payload), context)
        assert release["status"] == expected, (failing, release)
        assert (release["semantic_evaluation"] == SEMANTIC_FAILED) is gates, (failing, release)


# REGRESSION GUARD (passes at base and tip) — not a G9 proof
def test_regression_guard_not_evaluated_is_never_false_and_never_demotes() -> None:
    """**PRODUCT_CONTRACT 11** (*"``not_evaluated`` is never ``false``"*) and merge rule 7.

    An unevaluable check produces NO status change -- never a demotion, never
    ``diagnostic_only``. The three semantic states stay distinct: a run that could not be
    evaluated is not a run that failed, and neither is a pass.
    """

    # The default, unwired call: unchanged from the base, reason stated, never False.
    unwired = classify_release_status(
        {"requested_core_declared": True, "surviving_processes": 3,
         "minimum_core_satisfied": True, "reasons": []},
        strict_gates_passed=True,
    )
    assert unwired.semantic_evaluation == SEMANTIC_NOT_EVALUATED
    assert unwired.semantic_evaluation is not False
    assert unwired.semantic_not_evaluated_reason
    assert unwired.status == RELEASE_READY
    assert unwired.strict_acceptance_eligible is True

    # MEASURED: an empty payload still yields a PASS, because CHECK_ID_CONFLICT is
    # unconditionally applicable -- it asks only about the payload and an empty payload
    # has no conflicts. So "every gating check inapplicable" is UNREACHABLE from
    # ``evaluate_production_semantics`` today, and the guard below is defensive rather
    # than a live path. Stated plainly instead of dressed up as a live branch.
    empty = sp.evaluate_production_semantics({"entities": {}, "processes": {}})
    assert empty.checks[_s.CHECK_ID_CONFLICT].applicable is True
    assert semantic_verdict(empty)[0] == SEMANTIC_PASSED

    # The defensive branch, reached through the helper's own duck-typed contract: if a
    # producer ever hands over a report in which every gating check is inapplicable, the
    # answer is not_evaluated, NOT passed. A pass with nothing behind it would let an
    # unevaluable run count as semantically confirmed.
    class _Check:
        def __init__(self) -> None:
            self.ok, self.applicable = True, False

    class _AllInapplicable:
        evaluated = True
        not_evaluated_reason = ""
        checks = {name: _Check() for name in SEMANTIC_GATING_CHECKS}

    state, reason, failed = semantic_verdict(_AllInapplicable())
    assert (state, failed) == (SEMANTIC_NOT_EVALUATED, ())
    assert reason == SEMANTIC_NO_GATING_CHECK_EVALUABLE

    # No payload at all -> not_evaluated with the payload reason, never a failure.
    absent_state, absent_reason, _ = semantic_verdict(sp.evaluate_production_semantics(None))
    assert absent_state == SEMANTIC_NOT_EVALUATED
    assert absent_reason == sp.NO_PAYLOAD
    # ...and None itself is tolerated rather than raising.
    assert semantic_verdict(None)[0] == SEMANTIC_NOT_EVALUATED

    # The three states are mutually exclusive, never collapsed.
    assert len({SEMANTIC_PASSED, SEMANTIC_FAILED, SEMANTIC_NOT_EVALUATED}) == 3


# REGRESSION GUARD (passes at base and tip) — not a G9 proof
def test_regression_guard_the_rag_import_is_function_local() -> None:
    """**D-039 section 1.** A module-level ``rag`` import is a REJECT, not a re-pin request.

    ``tests/test_semantic_production_no_gold.py::test_e`` pins this module's import graph as
    ``set(sys.modules) - package`` -- i.e. EVERY module including stdlib -- so a top-level
    ``from t2pw.rag...`` would drag ``math`` and others into a pinned exact set. That test
    stays byte-identical because the import below is deferred to call time. Asserted
    structurally, off the AST, so it cannot pass by luck of import ordering.
    """

    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    module_level: List[str] = []
    for node in tree.body:  # top level ONLY -- nested imports are function-local
        if isinstance(node, ast.Import):
            module_level.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_level.append(node.module or "")

    assert not any(name.startswith("t2pw.rag") for name in module_level), module_level
    # The sanctioned module-level t2pw import is still the single bench one C-017 declared.
    assert [n for n in module_level if n.startswith("t2pw")] == ["t2pw.bench"]

    # And the deferred import is really there, inside the organism check.
    organism = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_check_requested_organism"
    )
    nested = [
        node.module for node in ast.walk(organism)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert "t2pw.rag.admission" in nested


# ── A0-C3: the organism comparison, in its MEASURED form ────────────────────


#: requested, observed, emits a finding AFTER the widening, why.
#:
#: Written in the MEASURED form D-042 section 6 requires. The naive "all abbreviations
#: match" assertion is FALSE and would fail: ``eligibility._canonical_organism`` returns
#: ``"Escherichia coli"`` for ``"E. coli"`` but ``""`` for ``"H. sapiens"``, because the
#: human alias set carries no abbreviated binomial. Fixing that alias set is NOT this
#: card's -- ``rag/admission.py`` and ``rag/eligibility.py`` are import-only.
_ORGANISM_TABLE = (
    ("Escherichia coli", "Escherichia coli", False, "positive: identical"),
    ("Escherichia coli", "E. coli", False, "ABBREVIATION: the false positive this card removes"),
    ("Saccharomyces cerevisiae", "S. cerevisiae", False, "abbreviation, second lineage: also cleared"),
    ("Homo sapiens", "H. sapiens", True, "abbreviation NOT cleared: no human abbreviated binomial"),
    ("Escherichia coli", "  Escherichia   coli  ", False, "whitespace"),
    ("Escherichia coli", "Escherichia coli.", False, "punctuation"),
    ("Escherichia coli", "Escherichia coli K-12", False, "strain suffix"),
    ("Escherichia coli", "Escherichia", False, "bare genus: already tolerated, UNMOVED"),
    ("Escherichia coli", "Escherichia fergusonii", True, "same genus other species: STAYS a finding"),
    ("Escherichia coli", "Listeria monocytogenes", True, "negative: unrelated organism"),
    ("Escherichia coli", "", False, "blank observed is never a violation"),
)


@pytest.mark.parametrize("requested, observed, expected_finding, why", _ORGANISM_TABLE)
def test_organism_comparison_matches_the_measured_verdict_table(
    requested, observed, expected_finding, why
) -> None:
    """**A0-C3.** Positive, negative, abbreviation, whitespace, punctuation -- measured.

    ``E. coli`` is the case no existing assertion covered and the one the card exists to
    fix. The two rows that must NOT move are asserted beside it, in the same table, so a
    widening that over-reaches is as red as one that under-reaches.
    """

    check, count = sp._check_requested_organism(
        requested, {"reactions": [{"name": "r", "organism": observed}]}
    )
    assert bool(check.findings) is expected_finding, why
    assert (count > 0) is expected_finding, why
    assert check.ok is (not expected_finding), why


def test_organism_widening_is_strictly_monotone_and_reuses_the_public_wrapper() -> None:
    """**D-039 section 2.** The change may only REMOVE findings, never add one.

    A row is compatible iff the observed organism is blank, OR the pre-existing literal
    disjunction holds, OR ``compare_organism`` returns an exact ``match``. Only the third
    disjunct is new, and adding a disjunct to an ``or`` cannot make a previously-clean row
    dirty. Asserted rather than argued, over the whole table.

    ``genus_level`` deliberately does NOT clear: that is what keeps *Escherichia
    fergusonii* a finding while a bare genus stays tolerated by the pre-existing prefix
    rule -- two OPPOSITE existing behaviours, both unchanged.
    """

    for requested, observed, expected_finding, why in _ORGANISM_TABLE:
        verdict = compare_organism(requested, observed)
        if verdict == ORGANISM_MATCH:
            # Every exact match is cleared -- that is the whole widening.
            assert expected_finding is False, why
        if expected_finding:
            # Nothing still reported is an exact match: no finding survives that the
            # public wrapper calls the same organism.
            assert verdict != ORGANISM_MATCH, why

    # genus_level is neither newly tolerated nor newly penalised.
    assert compare_organism("Escherichia coli", "Escherichia fergusonii") != ORGANISM_MATCH
    assert compare_organism("Escherichia coli", "Escherichia") != ORGANISM_MATCH

    # Deterministic: the same inputs give the same verdict every time. No network, no LLM.
    for _ in range(3):
        assert compare_organism("Escherichia coli", "E. coli") == ORGANISM_MATCH


def test_the_organism_check_still_ignores_entity_rows_and_blank_organisms() -> None:
    """**PRODUCT_CONTRACT 13 / TRAP-3**, preserved: reactions only.

    A protein row legitimately carries the PathBank ``Unknown`` sentinel species, and
    grading it here would take a side in the standing ``placeholder_backed_proteins``
    disagreement. D-042 section 6 measured 8 such Arabidopsis rows in the corpus and
    records that extending this check to entities would newly fail 7 of 32 legs on a
    sentinel -- a merge-rule-6 violation. This card does not extend it.
    """

    report = sp.evaluate_production_semantics(
        {
            "entities": {"proteins": [{"name": "Unknown", "species": "Listeria monocytogenes"}]},
            "processes": {"reactions": [{"name": "r", "inputs": ["a"], "outputs": ["b"]}]},
        },
        requested_organism="Escherichia coli",
    )

    # The entity's contradicting species is not a cross-organism REACTION finding...
    assert report.checks[_s.CHECK_ORGANISM].ok is True
    assert report.scientific_errors[_s.ERR_CROSS_ORGANISM_REACTIONS] == 0
    # ...and a reaction with no organism at all is still not a violation.
    assert report.checks[_s.CHECK_ORGANISM].findings == []
