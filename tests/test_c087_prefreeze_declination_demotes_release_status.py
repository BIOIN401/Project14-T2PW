"""C-087 / F-123 / D-068 -- a pre-freeze declination must demote the release status.

**The defect.** ``prefreeze_report["ok"] = False`` demoted nothing. Both consuming
seams say so in terms (``writer.py:2724``, ``streamlit_app.py:4952``), because D-029
as split by **D-040 section 8** assigned *"acting on ``review_required``"* to no card
and registered it as backlog ``BL-004``. ``classify_release_status`` took no prefreeze
parameter, so an ambiguous-species-rename declination was **release-status-neutral**
and such a leg could reach ``release_ready`` -- with D-035 clause 8's *"must not
become a successful export"* enforced only by the OTHER gates and never by this
channel. D-068 assigns ``BL-004`` and rules the demotion.

**G9 -- and what the base failure is, precisely.** This is a correction of
pre-existing observable behaviour, so the proof must fail BEHAVIOURALLY on base
``91b5c50``, and **symbol absence is not proof**. That constrains where the proof can
live: calling ``classify_release_status(..., prefreeze_review_required=...)`` at base
raises ``TypeError`` for an unexpected keyword, which is symbol absence wearing a
behavioural costume, and F-122 registered exactly that mistake on C-082.

So the load-bearing proof is at ``batch/driver.py``'s seam, whose input is a plain
``pwml_result`` dict of **identical shape at base and at tip**: every key it reads is
a key production already publishes (``quarantine_report.release`` since D-004;
``prefreeze_resolution_report`` / ``prefreeze_review_required`` since C-052). The
tests marked ``G9`` below construct that dict, and on base ``91b5c50`` they fail
with ``release_ready`` / ``pathway.pwml`` where the tip answers ``review_required`` /
``pathway.review_required.pwml``. No symbol this file imports is new at base except
through ``t2pw.pipeline.release_status``, and the G9 tests import nothing new from
there -- they read the STATUS STRINGS, which are contract vocabulary and predate the
card.

The classifier-level cap is a genuinely NEW input surface, so its tests are labelled
``NEW ACCEPTANCE`` rather than pretending to a base failure, exactly as G9's second
arm requires.

**Non-vacuity.** Every behavioural test carries its own control arm -- the same
fixture WITHOUT a declination, asserted to reach ``release_ready`` -- so a demotion
assertion can never pass on a leg that could not have been release-ready anyway. On
top of that, ``test_every_demotion_assertion_in_this_file_is_non_vacuous`` neuters the
single normalizer the whole mechanism reads through and re-runs each assertion,
proving each one flips.

This file is in **no chunk and not in SMOKE**, deliberately and by the same reasoning
``test_c082_post_pipeline_seam.py`` records: ``chunk_d_gate.py`` hard-codes its file
list and its component counts are ENFORCED, so adding to a chunk member would move a
sprint-gate baseline under merge rule 4 for a reason unrelated to the baseline. It is
run as a NAMED FOCUSED obligation.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.driver import (  # noqa: E402
    MODE_STRICT,
    PWML_RELEASE_READY_NAME,
    PWML_REVIEW_REQUIRED_NAME,
    RunOutcome,
    _add_strict_artifacts,
    _finalize_pwml_export,
    _frozen_release_record,
)

#: Read as STRINGS, not imported as symbols. These three are PRODUCT_CONTRACT 4
#: vocabulary and exist unchanged at base; spelling them out is what keeps the G9
#: tests below free of any dependency on a name this card introduced.
RELEASE_READY = "release_ready"
REVIEW_REQUIRED = "review_required"
DIAGNOSTIC_ONLY = "diagnostic_only"

#: The exact reason ``prefreeze_resolution`` publishes for the declination C-082
#: shipped, written out rather than imported for the same reason C-082's own seam
#: test writes it out: an ``ImportError`` at base would be proving symbol absence.
DECLINED = "species_rename_declined:AMBIGUOUS_RENAME_TARGET"
#: D-029's channel-mate. Same channel, different cause, same disposition.
DB_UNAVAILABLE = "resolution_report_not_ok:db_unavailable"


# ---------------------------------------------------------------------------
# Fixtures -- the shapes production actually publishes.
# ---------------------------------------------------------------------------
def _release_ready_record() -> Dict[str, Any]:
    """A frozen boundary record that IS release-ready, in its serialized shape.

    Copied in structure from ``ReleaseStatus.to_dict`` rather than built by calling
    the classifier, so the fixture is the same object at base and tip and the test
    is measuring the seam rather than the constructor.
    """

    return {
        "status": RELEASE_READY,
        "pipeline_executed": True,
        "strict_gates_passed": True,
        "semantic_evaluation": "passed",
        "semantic_not_evaluated_reason": "",
        "semantic_failed_checks": [],
        "semantic_check_evaluability": [
            {"check": "organism_compatible", "applicable": True, "inapplicable_reason": ""},
        ],
        "strict_acceptance_eligible": True,
        "completeness": 1.0,
        "missing_anchors": [],
        "retrieval_attempts": 3,
        "expansion_blocked_reason": "no further supported content remained at the freeze seam",
        "coverage_evaluated": True,
        "reasons": [],
    }


def _prefreeze_report(review: Dict[str, str], *, ok: bool) -> Dict[str, Any]:
    """The whole report shape ``run_prefreeze_resolution`` returns.

    ``failures`` is carried because the real report carries it and because the
    asymmetry test needs a report that is ``ok=False`` with failures that are NOT
    review-required -- which is exactly a non-empty ``failures`` beside an empty
    ``review_required``.
    """

    return {
        "stage": "prefreeze_resolution",
        "canonicalizers": ["compounds", "species"],
        "compounds": {"resolution_report": {"ok": True}},
        "species": {"applied": True},
        "ok": ok,
        "failures": dict(review) if review else ({} if ok else {"compounds": "resolution_report_not_ok"}),
        "review_required": dict(review),
    }


def _pwml_result(
    *,
    release: Dict[str, Any],
    prefreeze: Any = None,
    review_only: Any = None,
    payload_rows: Tuple[str, ...] = ("Escherichia coli K-12", "Escherichia coli"),
) -> Dict[str, Any]:
    """One strict export result, exactly the keys ``driver`` reads off it.

    ``canonical_payload`` rides along so the preservation test can assert the graph
    is untouched by the demotion on the same object the seam saw.
    """

    result: Dict[str, Any] = {
        "ok": True,
        "xml_bytes": b"<?xml version='1.0'?><pathway/>",
        "counts": {"reactions": 4, "entities": 11},
        "quarantine_report": {"release": release},
        "canonical_payload": {
            "entities": {
                "species": [
                    {"name": name, "taxonomy_id": tax}
                    for name, tax in zip(payload_rows, ("83333", "562"))
                ]
            }
        },
    }
    if prefreeze is not None:
        result["prefreeze_resolution_report"] = prefreeze
    if review_only is not None:
        result["prefreeze_review_required"] = review_only
    return result


def _drive(pwml_result: Dict[str, Any]) -> RunOutcome:
    """Run the production terminal path for a passing strict leg."""

    outcome = RunOutcome(paper_id="PMC12444477", mode=MODE_STRICT)
    _finalize_pwml_export(
        outcome,
        xml=pwml_result["xml_bytes"],
        pwml_result=pwml_result,
        joined="",
        codes=[],
    )
    return outcome


# ---------------------------------------------------------------------------
# G9 -- the base-failing behavioural proofs.
# ---------------------------------------------------------------------------
def test_g9_a_declined_species_rename_stops_a_release_ready_leg() -> None:
    """G9. Otherwise-release-ready leg + ``ok=False`` + a review reason -> demoted.

    **Base ``91b5c50``:** ``release_ready``. **Tip:** ``review_required``.

    The CONTROL arm runs first and is what makes the assertion non-vacuous: the
    identical fixture with an ``ok=True`` report reaches ``release_ready``, so the
    demotion below is caused by the declination and by nothing else about the
    fixture.
    """

    control = _drive(
        _pwml_result(
            release=_release_ready_record(),
            prefreeze=_prefreeze_report({}, ok=True),
            review_only={},
        )
    )
    assert control.release_status["status"] == RELEASE_READY, control.release_status
    assert control.release_status["strict_acceptance_eligible"] is True

    demoted = _drive(
        _pwml_result(
            release=_release_ready_record(),
            prefreeze=_prefreeze_report({"species": DECLINED}, ok=False),
            review_only={"species": DECLINED},
        )
    )
    assert demoted.release_status["status"] == REVIEW_REQUIRED, demoted.release_status
    # The record says WHICH, so an operator never has to re-open the prefreeze
    # report to learn why the leg is not release-ready.
    assert any(
        reason.startswith("prefreeze_resolution_review_required:")
        and DECLINED in reason
        for reason in demoted.release_status["reasons"]
    ), demoted.release_status["reasons"]


def test_g9_a_declined_rename_also_closes_the_artifact_naming_channel() -> None:
    """G9. The demoted leg must not ship a bare ``pathway.pwml``.

    PRODUCT_CONTRACT 13 reads that filename as *"ship it, no review needed"*. Capping
    only the manifest row would leave D-035 clause 8's *"must not become a successful
    export"* enforced by coincidence on this channel -- which D-068 forbids by name.

    **Base ``91b5c50``:** ``pathway.pwml``. **Tip:** ``pathway.review_required.pwml``.
    """

    control_files: Dict[str, Any] = {}
    control_name = _add_strict_artifacts(
        {},
        _pwml_result(release=_release_ready_record(), prefreeze=_prefreeze_report({}, ok=True)),
        control_files,
    )
    assert control_name == PWML_RELEASE_READY_NAME
    assert PWML_RELEASE_READY_NAME in control_files

    files: Dict[str, Any] = {}
    result = _pwml_result(
        release=_release_ready_record(),
        prefreeze=_prefreeze_report({"species": DECLINED}, ok=False),
    )
    name = _add_strict_artifacts({}, result, files)
    assert name == PWML_REVIEW_REQUIRED_NAME, name
    assert PWML_RELEASE_READY_NAME not in files
    # Merge rule 7: the bytes are PRESERVED under the honest name, never dropped.
    assert files[PWML_REVIEW_REQUIRED_NAME] == result["xml_bytes"]
    # And the two channels agree, which is the property capping one alone destroys.
    assert _drive(result).pwml_artifact == PWML_REVIEW_REQUIRED_NAME


def test_g9_the_db_unavailable_declination_demotes_on_the_same_channel() -> None:
    """G9. D-029's cause, C-082's cause, one disposition.

    D-068 rules on the review-required CHANNEL, not on the species rename alone.
    ``resolution_report_not_ok:db_unavailable`` is the channel's other member
    (``prefreeze_resolution._REVIEW_REQUIRED_REASONS``) and must demote identically,
    or the invariant is a special case for one canonicalizer.

    **Base ``91b5c50``:** ``release_ready``. **Tip:** ``review_required``.
    """

    demoted = _drive(
        _pwml_result(
            release=_release_ready_record(),
            prefreeze=_prefreeze_report({"compounds": DB_UNAVAILABLE}, ok=False),
        )
    )
    assert demoted.release_status["status"] == REVIEW_REQUIRED, demoted.release_status
    assert any(DB_UNAVAILABLE in reason for reason in demoted.release_status["reasons"])


def test_g9_a_demoted_run_is_never_strict_acceptance_eligible() -> None:
    """G9 + TRAP-1 / PRODUCT_CONTRACT 13. ``review_required`` never counts as strict.

    The frozen record arrives carrying ``strict_acceptance_eligible: True`` -- it was
    release-ready when it was frozen -- so a demotion that only rewrote ``status``
    would leave a ``review_required`` run in the STRICT benchmark denominator. That
    is the precise shape TRAP-1 exists to stop.

    **Base ``91b5c50``:** ``True``. **Tip:** ``False``.
    """

    for review in ({"species": DECLINED}, {"compounds": DB_UNAVAILABLE},
                   {"species": DECLINED, "compounds": DB_UNAVAILABLE}):
        outcome = _drive(
            _pwml_result(
                release=_release_ready_record(),
                prefreeze=_prefreeze_report(review, ok=False),
            )
        )
        assert outcome.release_status["strict_acceptance_eligible"] is False, review
        assert outcome.release_status["status"] == REVIEW_REQUIRED, review
        # The invariant FINDINGS M-8 pins, restated on the demoted record.
        assert (
            outcome.release_status["strict_acceptance_eligible"]
            == (outcome.release_status["status"] == RELEASE_READY)
        )


def test_g9_the_payload_and_every_other_recorded_fact_survive_the_demotion() -> None:
    """G9 + merge rule 7 + D-068's *"useful intact biology remains available"*.

    Four things are asserted because four different things could have been lost:
    the two organism rows (unmerged, under their own names, with their own taxonomy
    ids -- the rename stays DECLINED and nothing is guessed), the PWML bytes, the
    counts, and every field of the frozen record other than the two the cap is
    allowed to move.

    **Base ``91b5c50``:** the leg is ``release_ready`` and the assertions on
    ``status`` / ``strict_acceptance_eligible`` fail. Everything else already held --
    which is the point: this test proves the correction costs nothing.
    """

    frozen = _release_ready_record()
    result = _pwml_result(
        release=dict(frozen),
        prefreeze=_prefreeze_report({"species": DECLINED}, ok=False),
    )
    before_species = [dict(row) for row in result["canonical_payload"]["entities"]["species"]]

    outcome = _drive(result)
    record = outcome.release_status

    # 1. The biology. Two rows, unmerged, names and taxonomy ids intact.
    after_species = result["canonical_payload"]["entities"]["species"]
    assert after_species == before_species
    names = [row["name"] for row in after_species]
    assert names == ["Escherichia coli K-12", "Escherichia coli"]
    assert len(names) == len(set(names)), "two organisms were merged onto one name"

    # 2. The bytes, and 3. the counts the seam records off them.
    assert result["xml_bytes"] == b"<?xml version='1.0'?><pathway/>"
    assert outcome.counts["pwml"] == {"reactions": 4, "entities": 11}
    assert outcome.counts["pwml_bytes"] == len(result["xml_bytes"])
    assert outcome.status == "pass", "the leg is still a PASS; only the release claim moved"

    # 4. Every field of the record except the two the cap may move, plus ``reasons``
    #    which may only GROW.
    moved = {"status", "strict_acceptance_eligible", "reasons"}
    for key, value in frozen.items():
        if key in moved:
            continue
        assert record[key] == value, key
    assert record["reasons"][: len(frozen["reasons"])] == frozen["reasons"]
    assert len(record["reasons"]) == len(frozen["reasons"]) + 1

    # 5. The frozen object handed in was NOT mutated: a caller still holding it
    #    still holds the boundary's record.
    assert result["quarantine_report"]["release"] == frozen


def test_g9_a_diagnostic_only_run_is_never_promoted_by_this_path() -> None:
    """G9-adjacent, and the one direction a cap must never move.

    ``diagnostic_only`` means no PWML exists. Promoting it to ``review_required``
    because a rename was declined would have this seam invent a deliverable the
    chain said does not exist -- and would raise a strict-benchmark denominator.
    Asserted at BOTH the record and the FILENAME, because ``_pwml_artifact_name``
    reads ``diagnostic_only`` as "write no final PWML" and a promotion there would
    silently start emitting one.

    Passes at base AND at tip (nothing promoted at base either); it is the guard
    that keeps the correction one-directional, and it is non-vacuous because the
    tests above prove this same seam DOES move a release-ready record.
    """

    for review in ({"species": DECLINED}, {"compounds": DB_UNAVAILABLE}):
        record = dict(_release_ready_record())
        record["status"] = DIAGNOSTIC_ONLY
        record["strict_acceptance_eligible"] = False
        record["reasons"] = ["strict_technical_gates_blocked_export"]
        result = _pwml_result(
            release=record, prefreeze=_prefreeze_report(review, ok=False)
        )

        outcome = _drive(result)
        assert outcome.release_status["status"] == DIAGNOSTIC_ONLY, review
        assert outcome.release_status["reasons"] == ["strict_technical_gates_blocked_export"]
        assert outcome.pwml_artifact == "", "a diagnostic_only leg emitted a final PWML"

        files: Dict[str, Any] = {}
        assert _add_strict_artifacts({}, result, files) == ""
        assert files == {} or PWML_RELEASE_READY_NAME not in files

    # A record already ``review_required`` is likewise not moved, and no reason is
    # accumulated twice -- the same "never restate a refusal" rule the four caps in
    # ``classify_release_status`` follow.
    already = dict(_release_ready_record())
    already["status"] = REVIEW_REQUIRED
    already["strict_acceptance_eligible"] = False
    already["reasons"] = ["requested_core_coverage_below_minimum"]
    outcome = _drive(
        _pwml_result(
            release=already, prefreeze=_prefreeze_report({"species": DECLINED}, ok=False)
        )
    )
    assert outcome.release_status["status"] == REVIEW_REQUIRED
    assert outcome.release_status["reasons"] == ["requested_core_coverage_below_minimum"]


# ---------------------------------------------------------------------------
# The documented asymmetry (D-068 requires it decided, not silently collapsed).
# ---------------------------------------------------------------------------
def test_an_ok_false_with_no_review_required_reason_stays_status_neutral() -> None:
    """The asymmetry, asserted rather than assumed.

    ``prefreeze_resolution._REVIEW_REQUIRED_REASONS`` is documented as *"verdicts
    that mean 'identity was not established', not 'the payload is wrong'"*. An
    ``ok=False`` outside that set -- ``resolution_report_not_ok`` with the DB
    REACHABLE, a non-benign skip, ``summary_not_a_mapping`` -- is the opposite kind
    of fact, and D-068 rules only on the review-required channel. It is left exactly
    as base leaves it, and this test is what stops a later reader "tidying" the two
    into one rule.

    Passes at base AND at tip, by construction -- that is the claim.
    """

    for report in (
        _prefreeze_report({}, ok=False),
        {"stage": "prefreeze_resolution", "ok": False,
         "failures": {"compounds": "summary_not_a_mapping"}, "review_required": {}},
        # The shape an entry point writes when a canonicalizer RAISED: ok=False and
        # NO ``review_required`` key at all. Must not be mined for reasons out of
        # its own bookkeeping keys.
        {"stage": "prefreeze_resolution", "ok": False, "raised": True,
         "code": "AMBIGUOUS_REFERENCE", "message": "boom", "details": {}},
    ):
        outcome = _drive(_pwml_result(release=_release_ready_record(), prefreeze=report))
        assert outcome.release_status["status"] == RELEASE_READY, report
        assert outcome.release_status["strict_acceptance_eligible"] is True
        assert outcome.release_status["reasons"] == []


def test_an_ok_true_report_never_demotes_whatever_else_it_carries() -> None:
    """``ok=True`` is a hard gate on the whole-report shape.

    A report that finished ``ok=True`` cannot, by ``run_prefreeze_resolution``'s own
    construction, carry a non-empty ``review_required`` -- but if one ever did, D-068
    demotes on ``ok=False`` WITH a reason, so ``ok=True`` must win. Asserted so the
    literal reading is the shipped one.
    """

    contradictory = _prefreeze_report({"species": DECLINED}, ok=True)
    outcome = _drive(_pwml_result(release=_release_ready_record(), prefreeze=contradictory))
    assert outcome.release_status["status"] == RELEASE_READY
    assert outcome.release_status["reasons"] == []


def test_both_published_shapes_are_read_and_the_bare_mapping_needs_no_ok() -> None:
    """The sub-mapping alone demotes; the empty sub-mapping does not.

    Three seams publish ``prefreeze_review_required`` -- the ``review_required``
    sub-mapping with no ``ok`` beside it. It needs none: ``run_prefreeze_resolution``
    builds ``ok = not failures`` and filters ``review_required`` out of the SAME
    ``failures`` dict, so a non-empty sub-mapping implies ``ok is False``.
    """

    # Sub-mapping only, no whole report present at all.
    demoted = _drive(
        _pwml_result(release=_release_ready_record(), review_only={"species": DECLINED})
    )
    assert demoted.release_status["status"] == REVIEW_REQUIRED

    # Empty sub-mapping is the ok=True case and demotes nothing.
    kept = _drive(_pwml_result(release=_release_ready_record(), review_only={}))
    assert kept.release_status["status"] == RELEASE_READY

    # The CLI publishes a PATH string under the report key; the fallback answers.
    both = _pwml_result(release=_release_ready_record(), review_only={"species": DECLINED})
    both["prefreeze_resolution_report"] = "outputs/pwml_prefreeze_resolution_report.json"
    assert _drive(both).release_status["status"] == REVIEW_REQUIRED

    # No prefreeze key at all -- every pre-C-052 result shape -- is untouched.
    assert _frozen_release_record(
        {"quarantine_report": {"release": _release_ready_record()}}
    )["status"] == RELEASE_READY


def test_the_stage_label_is_the_one_the_real_prefreeze_module_writes() -> None:
    """``PREFREEZE_RESOLUTION_STAGE`` is kept in step BY TEST, not by comment.

    The whole-report / sub-mapping discrimination hangs on this one string, and
    ``release_status`` cannot import ``t2pw.pwml`` to learn it without inverting the
    layering for every importer. So the real orchestrator is run and its report's
    own label is compared -- behaviour, not a symbol copied between two files.
    """

    from t2pw.pipeline.release_status import PREFREEZE_RESOLUTION_STAGE
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    report = run_prefreeze_resolution({"entities": {}, "processes": {}}, strict_db=False)
    assert report["stage"] == PREFREEZE_RESOLUTION_STAGE
    assert report["ok"] is True
    assert report["review_required"] == {}


# ---------------------------------------------------------------------------
# NEW ACCEPTANCE -- the classifier-level cap. A new input surface, labelled as one.
# ---------------------------------------------------------------------------
#: The documented input surface of ``classify_release_status``, as its own docstring
#: and signature define it. C-077 swept this surface with ``strict_gates_passed``
#: pinned False and measured ``diagnostic_only`` 196,800 times and ``review_required``
#: zero times; the sweep below re-measures that invariant with every prefreeze arm
#: attached, so the new input cannot have perturbed it.
_COVERAGES: Tuple[Any, ...] = (
    None,
    {"requested_core_declared": False, "surviving_processes": 3,
     "minimum_core_satisfied": True, "reasons": [], "coverage_ratio": 0.0},
    {"requested_core_declared": True, "surviving_processes": 3, "coverage_ratio": 1.0,
     "minimum_core_satisfied": True, "reasons": [], "unmatched_terms": []},
    {"requested_core_declared": True, "surviving_processes": 3, "coverage_ratio": 0.8,
     "minimum_core_satisfied": True, "reasons": [], "unmatched_terms": ["EntA"]},
    {"requested_core_declared": True, "surviving_processes": 0, "coverage_ratio": 0.0,
     "minimum_core_satisfied": False, "reasons": ["no_surviving_process"],
     "unmatched_terms": ["EntA"]},
    {"requested_core_declared": True, "surviving_processes": 2, "coverage_ratio": 0.25,
     "minimum_core_satisfied": False,
     "reasons": ["requested_core_coverage_below_minimum"], "unmatched_terms": ["EntA", "EntB"]},
)

#: Prefreeze inputs that must ALL classify identically to passing nothing.
_NEUTRAL_PREFREEZE: Tuple[Any, ...] = (
    None,
    {},
    _prefreeze_report({}, ok=True),
    _prefreeze_report({"species": DECLINED}, ok=True),
    {"stage": "prefreeze_resolution", "ok": False, "failures": {"c": "x"}, "review_required": {}},
    "outputs/pwml_prefreeze_resolution_report.json",
    42,
)


def _surface() -> List[Dict[str, Any]]:
    """Every documented keyword crossed with every other. 9,216 combinations."""

    grid = itertools.product(
        _COVERAGES,                        # 6
        (True, False),                     # pipeline_executed
        (True, False),                     # strict_gates_passed
        (True, False),                     # serializable_without_invention
        ("passed", "failed", "not_evaluated"),
        (None, 0, 1, 3),                   # connected_core_reactions
        (2, 3),                            # min_connected_core_reactions
        (True, False),                     # single_reaction_scope_requested
        (None, 5),                         # retrieval_attempts
        ((), ("stage0_scope_conflict",)),   # extra_reasons
    )
    return [
        {
            "coverage": coverage,
            "pipeline_executed": ran,
            "strict_gates_passed": gates,
            "serializable_without_invention": serializable,
            "semantic_evaluation": semantic,
            "semantic_failed_checks": ("organism_compatible",) if semantic == "failed" else (),
            "connected_core_reactions": core,
            "min_connected_core_reactions": floor,
            "single_reaction_scope_requested": single,
            "retrieval_attempts": retrieval,
            "extra_reasons": extra,
            "expansion_blocked_reason": "measured",
        }
        for (coverage, ran, gates, serializable, semantic, core, floor, single, retrieval, extra)
        in grid
    ]


def test_new_acceptance_a_neutral_prefreeze_input_changes_nothing_on_the_surface() -> None:
    """Required test 3. ``ok=True`` leaves classification byte-identical, everywhere.

    9,216 documented-surface combinations x 7 neutral prefreeze inputs = 64,512
    classifications, each compared field-by-field against the SAME combination
    classified with the parameter omitted entirely. Any divergence is a
    perturbation of a surface C-077 measured and D-068 did not authorize moving.
    """

    from t2pw.pipeline.release_status import classify_release_status

    surface = _surface()
    assert len(surface) == 9216, len(surface)

    for kwargs in surface:
        coverage = kwargs.pop("coverage")
        baseline = classify_release_status(coverage, **kwargs).to_dict()
        for prefreeze in _NEUTRAL_PREFREEZE:
            got = classify_release_status(
                coverage, prefreeze_review_required=prefreeze, **kwargs
            ).to_dict()
            assert got == baseline, (prefreeze, kwargs, got, baseline)
        kwargs["coverage"] = coverage


def test_new_acceptance_c077s_sweep_invariant_survives_every_prefreeze_arm() -> None:
    """Required test 4, at C-077's own scale and in C-077's own terms.

    With ``strict_gates_passed`` pinned False the surface is ``diagnostic_only``
    every time and ``review_required`` never -- C-077's measured result. Re-measured
    here with a DECLINING report attached to every combination: the cap must not
    reach a status the technical chain already lowered, in either direction.
    """

    from t2pw.pipeline.release_status import classify_release_status

    declining = _prefreeze_report({"species": DECLINED}, ok=False)
    seen: Dict[str, int] = {}
    for kwargs in _surface():
        if not kwargs["strict_gates_passed"]:
            coverage = kwargs.pop("coverage")
            status = classify_release_status(
                coverage, prefreeze_review_required=declining, **kwargs
            ).status
            seen[status] = seen.get(status, 0) + 1
            kwargs["coverage"] = coverage

    assert seen.get(REVIEW_REQUIRED, 0) == 0, seen
    assert seen.get(RELEASE_READY, 0) == 0, seen
    assert seen == {DIAGNOSTIC_ONLY: 4608}, seen


def test_new_acceptance_the_classifier_cap_demotes_and_records_which() -> None:
    """Required test 1, at the classifier. NEW input surface, so labelled as new.

    The same rule as the seam proof above, reached through the parameter rather than
    through a frozen record, so the two entry points cannot drift into two readings
    of one ruling.
    """

    from t2pw.pipeline.release_status import (
        REASON_PREFREEZE_REVIEW_REQUIRED,
        classify_release_status,
    )

    coverage = _COVERAGES[2]
    control = classify_release_status(coverage, strict_gates_passed=True)
    assert control.status == RELEASE_READY
    assert control.strict_acceptance_eligible is True

    capped = classify_release_status(
        coverage,
        strict_gates_passed=True,
        prefreeze_review_required=_prefreeze_report(
            {"species": DECLINED, "compounds": DB_UNAVAILABLE}, ok=False
        ),
    )
    assert capped.status == REVIEW_REQUIRED
    assert capped.strict_acceptance_eligible is False
    # Sorted, so the line does not depend on canonicalizer or dict ordering.
    assert capped.reasons == (
        f"{REASON_PREFREEZE_REVIEW_REQUIRED}:"
        f"compounds:{DB_UNAVAILABLE},species:{DECLINED}",
    )
    # Nothing else moved: the coverage record is carried through untouched.
    assert capped.completeness == control.completeness
    assert capped.missing_anchors == control.missing_anchors
    assert capped.coverage_evaluated is True


def test_new_acceptance_the_normalizer_is_the_single_reading_of_the_ruling() -> None:
    """Both entry points read one function, so one ruling cannot become two.

    Cross-checked over every shape this file uses: whatever
    ``prefreeze_review_reasons`` answers, the classifier cap and the frozen-record
    cap agree about whether a release-ready result survives.
    """

    from t2pw.pipeline.release_status import (
        cap_release_for_prefreeze_declination,
        classify_release_status,
        prefreeze_review_reasons,
    )

    shapes: Tuple[Any, ...] = _NEUTRAL_PREFREEZE + (
        _prefreeze_report({"species": DECLINED}, ok=False),
        _prefreeze_report({"compounds": DB_UNAVAILABLE}, ok=False),
        {"species": DECLINED},
        {"species": ""},
        {"stage": "prefreeze_resolution", "ok": False, "raised": True, "code": "X"},
    )
    for shape in shapes:
        expected_demotion = bool(prefreeze_review_reasons(shape))
        classified = classify_release_status(
            _COVERAGES[2], strict_gates_passed=True, prefreeze_review_required=shape
        )
        capped = cap_release_for_prefreeze_declination(_release_ready_record(), shape)
        assert (classified.status == REVIEW_REQUIRED) is expected_demotion, shape
        assert (capped["status"] == REVIEW_REQUIRED) is expected_demotion, shape
        assert capped["strict_acceptance_eligible"] is not expected_demotion, shape


# ---------------------------------------------------------------------------
# Non-vacuity.
# ---------------------------------------------------------------------------
def test_every_demotion_assertion_in_this_file_is_non_vacuous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neuter the one function the whole mechanism reads through; every arm flips.

    The mechanism has exactly one input predicate --
    ``release_status.prefreeze_review_reasons``. Stubbing it to ``()`` is the
    strongest available mutation: it disables the classifier cap AND the frozen
    record cap at once, which is precisely the base behaviour. Each assertion below
    is the negation of a demotion asserted above, so a test that passed for a reason
    other than the cap would keep passing here and fail this.
    """

    from t2pw.pipeline import release_status as rs

    demoting = _pwml_result(
        release=_release_ready_record(),
        prefreeze=_prefreeze_report({"species": DECLINED}, ok=False),
        review_only={"species": DECLINED},
    )
    # Armed: the mechanism is live and every arm demotes.
    assert _drive(demoting).release_status["status"] == REVIEW_REQUIRED
    assert _drive(demoting).release_status["strict_acceptance_eligible"] is False
    files: Dict[str, Any] = {}
    assert _add_strict_artifacts({}, demoting, files) == PWML_REVIEW_REQUIRED_NAME
    assert rs.classify_release_status(
        _COVERAGES[2], strict_gates_passed=True,
        prefreeze_review_required=_prefreeze_report({"species": DECLINED}, ok=False),
    ).status == REVIEW_REQUIRED

    monkeypatch.setattr(rs, "prefreeze_review_reasons", lambda _prefreeze: ())

    # Disarmed: every one of the four flips back to the base answer.
    assert _drive(demoting).release_status["status"] == RELEASE_READY
    assert _drive(demoting).release_status["strict_acceptance_eligible"] is True
    files = {}
    assert _add_strict_artifacts({}, demoting, files) == PWML_RELEASE_READY_NAME
    assert rs.classify_release_status(
        _COVERAGES[2], strict_gates_passed=True,
        prefreeze_review_required=_prefreeze_report({"species": DECLINED}, ok=False),
    ).status == RELEASE_READY


def test_the_control_arms_are_capable_of_reaching_release_ready() -> None:
    """The second non-vacuity risk: a fixture that could never be release-ready.

    Every demotion test above compares against a control built from the SAME
    ``_release_ready_record``. If that record could not produce ``release_ready``
    through the seam, every demotion assertion would be trivially true. Asserted
    here once, on the bare seam, with no prefreeze input in play at all.
    """

    bare = {"quarantine_report": {"release": _release_ready_record()}}
    assert _frozen_release_record(bare) == _release_ready_record()
    assert _frozen_release_record(bare)["strict_acceptance_eligible"] is True

    # And the guard that predates this card still fires: an uninterpretable status
    # is discarded rather than demoted, capped or trusted.
    unknown = {"quarantine_report": {"release": {"status": "core_release_ready"}}}
    assert _frozen_release_record(unknown) == {}
