"""D-002 at the release/refusal seam: a coverage shortfall is not a refusal.

**NEW ACCEPTANCE for the ``review_required`` outcome; BEHAVIOURAL CORRECTION for
``ok``.** Both halves are here on purpose, because the card is exactly the shape
merge gate G6 exists to catch and the two halves have to be read together:

* the correction -- a subthreshold but structurally valid, internally connected,
  serializable-without-guessing graph used to end the run with **no PWML at
  all**, which is PRODUCT_CONTRACT 1's "a valid pathway core suppressed because
  optional peripheral material is unresolved". It is now ``review_required``
  PWML. This half FAILS at the base SHA; see
  ``test_the_reference_case_is_no_longer_refused``.
* the preservation -- the other five refusal reasons say the graph is WRONG or
  UNSERIALIZABLE, and every one of them still refuses. Five separate tests, one
  per reason, plus the fail-open case where a subthreshold graph is ALSO
  structurally broken. **No biological gate is weakened and no content is
  admitted**: the exported graph is byte for byte the graph the strict rules
  already produced, which is asserted directly rather than argued.

The threshold value does not move and nothing here can move it -- there is no new
argument, no new knob, and the two thresholds are read straight off the coverage
verdict.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    RELEASE_READY,
    REVIEW_REQUIRED,
)
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    CORE_ACCEPTED,
    StrictQuarantineResult,
    quarantine_and_close,
)

from test_strict_quarantine_contract_alignment import _base  # noqa: E402


# ── The reference case ──────────────────────────────────────────────────────
#
# D-002's reference case is PMC12096016 at 0.167 coverage after the C-010 fix:
# one of six requested anchors is covered by a surviving core process, and the
# graph that survives is otherwise clean. Reproduced here as a shaped payload
# rather than replayed, because no archived leg carries a Stage-0 context (see
# ``test_strict_quarantine_real_artifact_replay``'s module docstring) and without
# one the run is in the undeclared regime, where 0.167 cannot arise at all.

#: Six anchors, exactly one of which any surviving core process touches.
#: 1/6 = 0.1666.. -> the pinned 0.167.
_SIX_ANCHORS: List[str] = [
    "L-glutamate",
    "ornithine",
    "citrulline",
    "argininosuccinate",
    "arginine",
    "fumarate",
]
_REFERENCE_CONTEXT: Dict[str, Any] = {"key_compounds": list(_SIX_ANCHORS)}


def _reference_payload() -> Dict[str, Any]:
    """A clean, connected, serializable pathway judged against six anchors.

    ``metadata.key_compounds`` is dropped so the request is the handed-over
    Stage-0 context and nothing else -- payload-derived anchors would be anchors
    taken from the survivors, which is not a request.
    """

    payload = _base()
    payload["metadata"].pop("key_compounds", None)
    return payload


def _report(result: StrictQuarantineResult) -> Dict[str, Any]:
    return result.quarantine_report


def _release(result: StrictQuarantineResult) -> Dict[str, Any]:
    return dict(_report(result)["release"])


def _reference_result() -> StrictQuarantineResult:
    return quarantine_and_close(
        _reference_payload(), strict_db=True, pathway_context=_REFERENCE_CONTEXT
    )


def test_the_reference_case_measures_the_pinned_coverage() -> None:
    """0.167, pinned. If this moves, every claim below is about a different run."""

    result = _reference_result()
    coverage = result.coverage

    assert coverage["requested_core_declared"] is True
    assert coverage["requested_core_source"] == "pathway_context"
    assert len(coverage["requested_core_terms"]) == 6
    assert coverage["matched_terms"] == ["L-glutamate"]
    assert round(float(coverage["coverage_ratio"]), 3) == 0.167
    assert coverage["minimum_core_satisfied"] is False
    # The shortfall is COVERAGE, not emptiness and not a process-count floor:
    # something defensible survived, and that is the whole distinction.
    assert coverage["reasons"] == ["requested_core_coverage_below_minimum:0.167<0.500"]
    assert coverage["core_accepted_processes"] >= 1
    assert coverage["surviving_processes"] >= 1
    # The threshold itself, unmoved.
    assert coverage["thresholds"] == {"min_core_processes": 1, "min_core_coverage": 0.5}


def test_the_reference_case_is_no_longer_refused() -> None:
    """**A1.** BEHAVIOURAL CORRECTION -- fails at the base SHA, where this run
    ends with ``ok=False``, ``minimum_core:...`` in ``refusal_reasons`` and no
    PWML produced at all."""

    result = _reference_result()

    assert result.ok is True
    assert result.refusal_reasons == []
    assert _report(result)["review_reasons"] == [
        "minimum_core:requested_core_coverage_below_minimum:0.167<0.500"
    ]
    assert _release(result)["status"] == REVIEW_REQUIRED
    # There IS a graph to export, which is the point: the run is not blocked.
    assert result.payload["processes"]["reactions"]


def test_the_reference_case_is_never_release_ready() -> None:
    """**A2.** The threshold blocks release-ready status, and it still does."""

    release = _release(_reference_result())

    assert release["status"] != RELEASE_READY
    assert release["status"] == REVIEW_REQUIRED
    assert release["strict_acceptance_eligible"] is False


def test_the_review_required_record_carries_all_four_d002_items() -> None:
    """**A6.** completeness, missing anchors, retrieval attempts, and why further
    supported expansion failed."""

    result = _reference_result()
    release = _release(result)

    # completeness -- the real ratio, and never None in the declared regime.
    assert release["completeness"] is not None
    assert round(float(release["completeness"]), 3) == 0.167
    # missing anchors -- the five nothing surviving touches, non-empty.
    assert release["missing_anchors"] == [
        "ornithine",
        "citrulline",
        "argininosuccinate",
        "arginine",
        "fumarate",
    ]
    # retrieval attempts -- present, an int, and labelled by its source so it can
    # never be misread as a retrieval-round counter.
    assert isinstance(release["retrieval_attempts"], int)
    assert release["retrieval_attempts_source"] == "surviving_rows_carrying_rag_provenance"
    # why further supported expansion failed -- present and non-empty, and it
    # names the unmatched anchors rather than asserting a bare failure.
    assert release["expansion_blocked_reason"]
    assert "requested-core anchor(s) matched no admitted process" in (
        release["expansion_blocked_reason"]
    )
    assert "ornithine" in release["expansion_blocked_reason"]
    # And the shortfall itself is on the record, not merely implied by a ratio.
    assert release["reasons"] == [
        "requested_core_coverage_below_minimum:0.167<0.500"
    ]
    assert release["coverage_evaluated"] is True


def test_the_seam_admits_nothing_and_invents_nothing() -> None:
    """**A10.** The graph exported under the relief is byte for byte the graph
    the strict rules produced when the same run was refused.

    Asserted against the payload the SAME code path builds, so it cannot drift:
    quarantine's decisions are all taken above the seam, and the seam is reached
    with ``working``, ``admissions`` and ``coverage`` already final. Compared as
    serialized JSON rather than by ``==`` so an added key of any kind shows up.
    """

    relieved = _reference_result()
    # The same payload judged with the coverage shortfall made irrelevant -- one
    # anchor instead of six -- so coverage passes and no relief is in play. The
    # exported graph must be identical: coverage is a VERDICT about the graph and
    # must never be an input to what the graph contains.
    satisfied = quarantine_and_close(
        _reference_payload(), strict_db=True, pathway_context={"key_compounds": ["L-glutamate"]}
    )

    dump = lambda payload: json.dumps(payload, sort_keys=True, ensure_ascii=False)
    assert dump(relieved.payload) == dump(satisfied.payload)
    assert (
        _report(relieved)["resulting_payload_hash"]
        == _report(satisfied)["resulting_payload_hash"]
    )
    # Coverage moved; the counted survivors did not.
    assert relieved.coverage["minimum_core_satisfied"] is False
    assert satisfied.coverage["minimum_core_satisfied"] is True
    assert relieved.coverage["surviving_processes"] == satisfied.coverage["surviving_processes"]
    assert (
        _report(relieved)["counts"][CORE_ACCEPTED]
        == _report(satisfied)["counts"][CORE_ACCEPTED]
    )


def test_the_threshold_value_is_not_moved_and_cannot_be_moved_by_this_seam() -> None:
    """**A10.** No new knob, and the shipped thresholds are what they were."""

    from t2pw.pipeline import strict_quarantine as SQ

    assert SQ.DEFAULT_MIN_CORE_COVERAGE == 0.5
    assert SQ.DEFAULT_MIN_CORE_PROCESSES == 1
    # A lower threshold still has to be asked for explicitly, and asking for one
    # changes the RECORDED verdict rather than being applied silently.
    tightened = quarantine_and_close(
        _reference_payload(),
        strict_db=True,
        pathway_context=_REFERENCE_CONTEXT,
        min_core_coverage=0.9,
    )
    assert tightened.coverage["thresholds"]["min_core_coverage"] == 0.9
    assert "0.167<0.900" in tightened.coverage["reasons"][0]
    assert tightened.quarantine_report["decision_inputs"]["min_core_coverage"] == 0.9


# ── A3: the other five still refuse. Five separate assertions. ──────────────


def _assert_refuses(result: StrictQuarantineResult, prefix: str) -> None:
    """One structural reason: still a refusal, still no PWML, still diagnostic."""

    assert result.ok is False, f"{prefix} stopped refusing"
    assert any(reason.startswith(prefix) for reason in result.refusal_reasons), (
        result.refusal_reasons
    )
    release = _release(result)
    assert release["status"] == DIAGNOSTIC_ONLY
    assert release["strict_acceptance_eligible"] is False
    # No PWML: ``ok`` is what the freeze seam reads, and a diagnostic_only run
    # has no final PWML by PRODUCT_CONTRACT 4.
    assert result.quarantine_report["ok"] is False


# Two of the five detectors cannot be made to fire from a payload without also
# emptying the graph, and that is a PROPERTY of the merged code rather than a gap
# in these tests:
#
# * ``_entity_type_overlaps`` needs one name declared in two surviving buckets.
#   The registry calls such a name ambiguous, so every process that reaches it is
#   quarantined, and closure then prunes both declarations -- the overlap
#   disappears along with the graph.
# * ``_degree_zero_exports`` is a residual guard after C-010: closure prunes
#   every row nothing references, so a payload that reaches it with a degree-zero
#   row has already lost the reference that made it degree-zero.
#
# What A3 is about is the SEAM's treatment of a reason, not a detector's trigger
# conditions -- both detectors belong to C-010/C-041 and are untouched here. So
# for these two the detector is made to fire directly, against the real
# ``_base()`` graph, and the seam's answer is what is asserted. The substitution
# returns the detector's own row shape.


def _with_detector(name: str, rows: List[Dict[str, str]], **kwargs: Any) -> StrictQuarantineResult:
    """Run the real seam over the real ``_base()`` graph with ``name`` firing."""

    from t2pw.pipeline import strict_quarantine as SQ

    shipped = getattr(SQ, name)
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(SQ, name, lambda *a, **k: list(rows))
        return quarantine_and_close(_base(), strict_db=True, **kwargs)
    finally:
        monkey.undo()
        assert getattr(SQ, name) is shipped


_OVERLAP_ROWS = [{"name": "GPX1", "buckets": "compounds+proteins"}]
_DEGREE_ZERO_ROWS = [{"bucket": "proteins", "name": "catalase"}]


def test_entity_type_overlap_still_refuses() -> None:
    """**A3.1.** A structural contradiction, untouched by D-002."""

    _assert_refuses(
        _with_detector("_entity_type_overlaps", _OVERLAP_ROWS), "entity_type_overlap:"
    )


def test_degree_zero_export_still_refuses() -> None:
    """**A3.2.** An entity with no connectivity, untouched by D-002."""

    _assert_refuses(
        _with_detector("_degree_zero_exports", _DEGREE_ZERO_ROWS), "degree_zero_export:"
    )


def _unexportable_payload() -> Dict[str, Any]:
    payload = _base()
    # A complex that a surviving reaction uses as its enzyme, one of whose
    # components carries no species and no accession -- the row the required
    # field contract rejects and that nothing upstream can fix without inventing
    # the missing identity.
    payload["entities"]["proteins"].append({"name": "GCLM"})
    payload["entities"]["protein_complexes"].append(
        {
            "name": "glutamate-cysteine ligase complex",
            "species": "Homo sapiens",
            "components": [
                {"name": "glutamate-cysteine ligase", "stoichiometry": 1},
                {"name": "GCLM", "stoichiometry": 1},
            ],
        }
    )
    payload["processes"]["reactions"][0]["enzymes"] = ["glutamate-cysteine ligase complex"]
    return payload


def test_unexportable_entity_still_refuses() -> None:
    """**A3.3.** Serialization would require inventing an identity."""

    result = quarantine_and_close(_unexportable_payload(), strict_db=True)
    _assert_refuses(result, "unexportable_entity:")
    # And it is recorded as a SERIALIZATION failure, not as a generic gate one.
    assert "serialization_would_require_invention" in _release(result)["reasons"]


def _unaccounted_lock_payload() -> Dict[str, Any]:
    payload = _base()
    payload["processes"]["reactions"][0]["locked_reaction_id"] = "rxn_lock_001"
    payload["locked_reaction_filter_report"] = {
        "locked_reactions_found": 4,
        "exported_locked_reactions": 1,
        "quarantined_locked_reactions": 0,
        "unaccounted_locked_reactions": 3,
    }
    return payload


def test_unaccounted_locked_reactions_still_refuses() -> None:
    """**A3.4.** A stranded lock, untouched by D-002."""

    _assert_refuses(
        quarantine_and_close(_unaccounted_lock_payload(), strict_db=True),
        "unaccounted_locked_reactions:",
    )


def test_closure_not_converged_still_refuses() -> None:
    """**A3.5.** The closure loop did not settle, untouched by D-002."""

    _assert_refuses(
        quarantine_and_close(_base(), strict_db=True, max_iterations=0),
        "closure_not_converged:",
    )


# ── A4: the fail-open clause ────────────────────────────────────────────────


def _assert_fails_open(result: StrictQuarantineResult, prefix: str) -> None:
    """Subthreshold AND structurally broken: refuses, and says both things."""

    assert result.coverage["minimum_core_satisfied"] is False, "not subthreshold; test is void"
    assert result.ok is False
    assert any(reason.startswith(prefix) for reason in result.refusal_reasons), (
        result.refusal_reasons
    )
    # The shortfall is still RECORDED -- relief and refusal are not exclusive --
    # but it is recorded where it cannot shorten the blocking list.
    assert any(r.startswith("minimum_core:") for r in result.quarantine_report["review_reasons"])
    assert _release(result)["status"] == DIAGNOSTIC_ONLY
    assert _release(result)["strict_acceptance_eligible"] is False


@pytest.mark.parametrize(
    "name, build",
    [
        ("unexportable_entity", _unexportable_payload),
        ("unaccounted_locked_reactions", _unaccounted_lock_payload),
    ],
)
def test_coverage_relief_never_masks_a_structural_failure(name: str, build) -> None:
    """**A4.** Subthreshold AND structurally broken refuses, every time.

    The failure mode this card could plausibly have: relieving the coverage
    reason and, with the refusal list now shorter, letting a run through that a
    structural gate was already blocking. Proved per reason rather than once,
    because "one combination refuses" would not exclude it for the others.
    """

    _assert_fails_open(
        quarantine_and_close(build(), strict_db=True, pathway_context=_REFERENCE_CONTEXT), name
    )


def test_closure_failure_plus_a_shortfall_still_refuses() -> None:
    """**A4**, third reason. Separate because it is set by an argument."""

    _assert_fails_open(
        quarantine_and_close(
            _reference_payload(),
            strict_db=True,
            pathway_context=_REFERENCE_CONTEXT,
            max_iterations=0,
        ),
        "closure_not_converged:",
    )


@pytest.mark.parametrize(
    "detector, rows, prefix",
    [
        ("_entity_type_overlaps", _OVERLAP_ROWS, "entity_type_overlap:"),
        ("_degree_zero_exports", _DEGREE_ZERO_ROWS, "degree_zero_export:"),
    ],
)
def test_coverage_relief_never_masks_an_injected_structural_failure(
    detector: str, rows: List[Dict[str, str]], prefix: str
) -> None:
    """**A4**, fourth and fifth reasons, with the detector made to fire."""

    _assert_fails_open(
        _with_detector(detector, rows, pathway_context=_REFERENCE_CONTEXT), prefix
    )


def test_coverage_relief_never_produces_a_strict_acceptance_eligible_run() -> None:
    """**A4/A7**, stated as the one invariant that bounds this whole card.

    Whatever else moved, a run that was relieved of a coverage shortfall may
    never be counted as strict success. Asserted over every shape in this module
    and both regimes at once, so a future payload class cannot slip past it.
    """

    for build in (
        _reference_payload,
        _base,
        _unexportable_payload,
        _unaccounted_lock_payload,
    ):
        for context in (None, _REFERENCE_CONTEXT):
            result = quarantine_and_close(build(), strict_db=True, pathway_context=context)
            if result.quarantine_report["review_reasons"]:
                assert _release(result)["strict_acceptance_eligible"] is False, (
                    build.__name__,
                    context,
                )


# ── A5: diagnostic_only is preserved ────────────────────────────────────────


def test_an_empty_graph_is_still_diagnostic_only_and_still_refuses() -> None:
    """**A5.** ``no_surviving_process`` is not a shortfall. It is an absence.

    ``release_status.py`` draws the distinction and it is consumed, not
    re-derived: an empty core has no defensible connected core, so the coverage
    reason stays a REFUSAL reason here rather than becoming a review reason.
    """

    payload = _base()
    payload.pop("metadata", None)
    for reaction in payload["processes"]["reactions"]:
        reaction["scope_membership"] = "out_of_scope"

    result = quarantine_and_close(payload, strict_db=True)

    assert result.ok is False
    assert "minimum_core:no_surviving_process" in result.refusal_reasons
    assert result.quarantine_report["review_reasons"] == []
    release = _release(result)
    assert release["status"] == DIAGNOSTIC_ONLY
    assert "no_defensible_connected_core" in release["reasons"]
    assert release["strict_acceptance_eligible"] is False


def test_serialization_requiring_invention_is_still_diagnostic_only() -> None:
    """**A5**, second half: no PWML when writing one would invent biology."""

    release = _release(quarantine_and_close(_unexportable_payload(), strict_db=True))

    assert release["status"] == DIAGNOSTIC_ONLY
    assert release["reasons"][0] == "serialization_would_require_invention"


# ── A7: TRAP-1, and the invariant that carries it ───────────────────────────


_TRAP1_LEG = (
    ROOT / "runs" / "2026-08-02_2130" / "papers" / "PMC12452463" / "research" / "final_mapped.json"
)

#: The route the gold ``export_rationale`` describes, anchor by anchor. Taken
#: from the rationale rather than chosen: the whole point is that the leg cannot
#: cover it, because EntA is absent and nothing converts
#: 2,3-dihydro-2,3-dihydroxybenzoate onward.
_TRAP1_CONTEXT = {
    "key_compounds": [
        "enterobactin",
        "2,3-dihydroxybenzoate",
        "2,3-dihydro-2,3-dihydroxybenzoate",
        "chorismate",
        "isochorismate",
        "L-serine",
    ]
}


@pytest.mark.skipif(not _TRAP1_LEG.is_file(), reason="archived leg not in this checkout")
def test_pmc12452463_is_review_required_and_never_strict_success() -> None:
    """**A7.** PRODUCT_CONTRACT 13: never strict success.

    A real archived PMC12452463 payload, judged against the route its gold
    ``export_rationale`` describes. It covers one anchor of six, so the outcome
    D-002 and PRODUCT_CONTRACT 13 both name is ``review_required`` with
    ``strict_acceptance_eligible=false`` -- exported and flagged, never counted.

    What this does NOT claim: that the seam can detect the chemical break on its
    own. It cannot -- that is a semantic fact and ``release_status`` carries
    ``semantic_evaluation="not_evaluated"`` on every path by construction until
    C-056a wires the input. What is asserted is that the coverage relief this
    card adds never makes this leg eligible.
    """

    payload = json.loads(_TRAP1_LEG.read_text(encoding="utf-8"))
    result = quarantine_and_close(payload, strict_db=True, pathway_context=_TRAP1_CONTEXT)
    release = _release(result)

    assert round(float(result.coverage["coverage_ratio"]), 3) == 0.167
    assert result.coverage["minimum_core_satisfied"] is False
    assert result.ok is True, "the fragment is still exported, not dropped"
    assert release["status"] == REVIEW_REQUIRED
    assert release["strict_acceptance_eligible"] is False
    # And the missing anchors are recorded, so the reviewer is told WHICH part of
    # the route is unsupported rather than only that something is.
    assert "2,3-dihydro-2,3-dihydroxybenzoate" in release["missing_anchors"]


@pytest.mark.parametrize(
    "build, context",
    [
        (_base, None),
        (_base, _REFERENCE_CONTEXT),
        (_reference_payload, _REFERENCE_CONTEXT),
        (_unexportable_payload, _REFERENCE_CONTEXT),
        (_unaccounted_lock_payload, None),
    ],
)
def test_strict_acceptance_eligibility_equals_release_ready_everywhere(build, context) -> None:
    """**A7.** The invariant, through the seam, on every branch it has.

    ``ReleaseStatus`` has no ``__post_init__`` (FINDINGS M-8), so this invariant
    lives only in ``classify_release_status``. This asserts the seam went through
    that factory rather than building the record beside it.
    """

    release = _release(quarantine_and_close(build(), strict_db=True, pathway_context=context))

    assert release["strict_acceptance_eligible"] == (release["status"] == RELEASE_READY)


def test_ok_and_the_release_status_can_never_disagree() -> None:
    """``ok`` says the graph may be frozen; ``release`` says what the run is.

    Two answers to one question is how a consumer ends up exporting a run the
    classifier called diagnostic_only. Asserted over every payload shape in this
    module at once.
    """

    for build in (
        _base,
        _unexportable_payload,
        _unaccounted_lock_payload,
        _reference_payload,
    ):
        for context in (None, _REFERENCE_CONTEXT):
            result = quarantine_and_close(build(), strict_db=True, pathway_context=context)
            produces_pwml = _release(result)["status"] in (RELEASE_READY, REVIEW_REQUIRED)
            assert result.ok is produces_pwml, (build.__name__, context, result.refusal_reasons)
    for detector, rows in (
        ("_entity_type_overlaps", _OVERLAP_ROWS),
        ("_degree_zero_exports", _DEGREE_ZERO_ROWS),
    ):
        result = _with_detector(detector, rows)
        assert result.ok is False
        assert _release(result)["status"] == DIAGNOSTIC_ONLY


# ── A8: research mode is untouched ──────────────────────────────────────────


def test_research_mode_is_byte_for_byte_unchanged() -> None:
    """**A8.** Research decides and applies nothing; that is still true."""

    payload = _reference_payload()
    before = json.dumps(payload, sort_keys=True, ensure_ascii=False)

    result = quarantine_and_close(
        deepcopy(payload), strict_db=True, pathway_context=_REFERENCE_CONTEXT, mode="research"
    )

    # The candidate comes back byte for byte, and the input was not mutated.
    assert json.dumps(result.payload, sort_keys=True, ensure_ascii=False) == before
    assert json.dumps(payload, sort_keys=True, ensure_ascii=False) == before
    assert result.ok is True
    assert result.refusal_reasons == []
    assert result.quarantine_report["review_reasons"] == []
    # ``would_have_refused`` still reports what the STRICT rules would have
    # refused, in the pre-D-002 order, including the coverage shortfall that no
    # longer refuses a strict run.
    assert result.quarantine_report["research_mode"]["would_have_refused"] == [
        "minimum_core:requested_core_coverage_below_minimum:0.167<0.500"
    ]
    assert result.quarantine_report["research_mode"]["applied"] is False
    # And the classification is labelled a finding there, never a verdict acted on.
    assert _release(result)["applied"] is False


def test_research_mode_would_have_refused_still_lists_every_structural_reason() -> None:
    """The pre-D-002 list, unshortened and in the pre-D-002 order.

    Coverage first, then the structural classes -- the order ``refusal_reasons``
    itself had before this card. A research report that lost the coverage entry,
    or reordered it, would be a change to a path this card must not touch.
    """

    result = quarantine_and_close(
        _unexportable_payload(),
        strict_db=True,
        pathway_context=_REFERENCE_CONTEXT,
        mode="research",
    )
    would = result.quarantine_report["research_mode"]["would_have_refused"]

    assert would == [
        "minimum_core:requested_core_coverage_below_minimum:0.286<0.500",
        "unexportable_entity:1",
    ]


# ── A9: artifact naming stays a consumer's decision ─────────────────────────


def test_the_seam_names_no_artifact_and_writes_no_pwml(tmp_path: Path) -> None:
    """**A9.** PRODUCT_CONTRACT 13's naming is keyed off the STRUCTURED status.

    This seam produces the status and writes nothing: ``pathway.pwml`` versus
    ``pathway.review_required.pwml`` versus no final PWML is the batch artifact
    set's decision (D-004 / C-053), the interactive download and ``outputs/`` are
    unchanged, and ``outputs/pathway.pwml`` is never touched from here. What is
    asserted is that the three states a namer needs are distinguishable from the
    record alone.
    """

    monitored = list(tmp_path.rglob("*"))
    assert monitored == []

    states = {
        _release(quarantine_and_close(build(), strict_db=True, pathway_context=context))["status"]
        for build, context in (
            (_base, None),
            (_reference_payload, _REFERENCE_CONTEXT),
            (_unexportable_payload, None),
        )
    }

    assert states == {RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY}
    # Nothing was written anywhere by the seam itself.
    assert list(tmp_path.rglob("*")) == []
    assert not list(ROOT.glob("*.pwml"))


def test_the_report_schema_is_additive_over_schema_three() -> None:
    """Every schema-3 key keeps its name, its type and its meaning."""

    report = _report(_reference_result())

    assert report["schema_version"] == 4
    for key in (
        "stage", "strict_db", "export_mode", "policy_version", "admitted_payload_hash",
        "resulting_payload_hash", "decision_inputs", "decision_input_hash", "ok",
        "counts", "admissions", "quarantined", "accepted", "strict_invariants",
        "refusal_reasons", "locked_reactions", "coverage", "closure", "decision_id",
    ):
        assert key in report, key
    assert set(report) >= {"review_reasons", "release"}
