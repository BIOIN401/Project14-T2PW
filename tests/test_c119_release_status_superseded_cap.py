"""C-119 seam 3 -- **AN EXPLICITLY LABELLED NEW ACCEPTANCE TEST** (G9 class B).

**THIS IS A NEW CAPABILITY, NOT A REGRESSION FIX, AND NO BASE FAILURE IS CLAIMED
FOR IT.** ``classify_release_status`` had no sixth cap and no
``superseded_intermediate_report`` parameter at base SHA
``6746a8d31e993d8dc6968ad46534b4e48b9c7bee``; there is no pre-existing observable
behaviour here to correct, so G9's regression arm does not apply and nothing in
this file should be read as a base-failure proof. The base-failure proof belongs to
seams 1-2 and lives in ``tests/test_c119_g9_base_failure_proof.py``, which says so
in its own docstring and which is the file that was actually run against the base
tree. (``tests/test_c119_superseded_intermediate_report.py`` is NOT that proof and
cannot be: it imports ``REASON_SUPERSEDED_INTERMEDIATE_REPORT``, a C-119 symbol, so
it does not even import at base -- REV-119 finding 2.) Mislabelling either
direction is a reject, so the two files never share a claim.

WHAT IS BEING ACCEPTED. A sixth cap of exactly the shape of the five that already
exist -- semantic (D-042), incomplete-core (F-094), connected-pathway (F-101),
unstated-request (F-100) and pre-freeze (C-087 / **D-068**) -- together with the
frozen-record twin ``cap_release_for_superseded_intermediate_report``, which is the
same arrangement C-087 already uses for the pre-freeze declination and for the same
ordering reason. The four restrictions every cap carries are asserted individually
below:

1. it may move ONLY ``release_ready`` -> ``review_required``;
2. exactly ONE step -- never to ``diagnostic_only``;
3. it may NEVER create ``release_ready``, and never lowers a status the chain
   already lowered;
4. its new parameter defaults to ``None`` / not-recorded, so every existing caller
   stays byte-identical.

THE ONE DELIBERATE DIFFERENCE from the five, asserted here so a reviewer sees it
stated rather than discovers it: the REASON is recorded whenever the carrier is
present, cap or no cap, the way ``semantic_failed_checks`` already is in this same
function. ORCH-728 requires that a superseded error never silently vanish from
review metadata, and on every leg the rule was derived from the chain had ALREADY
lowered the status on ordinary caps -- so recording "from ``release_ready`` only"
would have dropped the finding in exactly the cases it exists for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    REASON_SUPERSEDED_INTERMEDIATE_REPORT,
    RELEASE_READY,
    REVIEW_REQUIRED,
    cap_release_for_superseded_intermediate_report,
    classify_release_status,
    superseded_report_reasons,
)

#: A payload whose coverage clears every existing gate, so ``classify_release_status``
#: reaches ``release_ready`` and the new cap has something to act on. Anything less
#: and the test would be asserting the OTHER caps' behaviour.
CLEAN_COVERAGE = {
    "schema_version": 1,
    "requested_core_declared": True,
    "requested_core_source": "pathway_context",
    "requested_core_terms": ["a", "b"],
    "matched_terms": ["a", "b"],
    "unmatched_terms": [],
    "coverage_ratio": 1.0,
    "core_accepted_processes": 4,
    "auxiliary_accepted_processes": 0,
    "surviving_processes": 4,
    "quarantined_processes": 0,
    "thresholds": {"min_core_processes": 1, "min_core_coverage": 0.5},
    "minimum_core_satisfied": True,
    "reasons": [],
    "requested_context": {"pathway_name": "a demonstrative pathway"},
}

SUPERSEDED = [
    {"report": "post_normalization_contract_report", "phase": "audit_round", "errors": 2}
]


def _classify(**kwargs):
    return classify_release_status(
        CLEAN_COVERAGE,
        pipeline_executed=True,
        strict_gates_passed=True,
        **kwargs,
    )


# --- restriction 4: the default is "not recorded", and it changes nothing -------


def test_new_acceptance_the_default_leaves_every_existing_caller_byte_identical() -> None:
    """NEW CAPABILITY. Omitting the parameter must be indistinguishable from base."""

    without = _classify().to_dict()
    explicit_none = _classify(superseded_intermediate_report=None).to_dict()
    empty = _classify(superseded_intermediate_report=[]).to_dict()

    assert without == explicit_none == empty
    assert without["status"] == RELEASE_READY
    assert not any(
        str(reason).startswith(REASON_SUPERSEDED_INTERMEDIATE_REPORT)
        for reason in without["reasons"]
    )


# --- restrictions 1-3: the cap itself -------------------------------------------


def test_new_acceptance_the_sixth_cap_moves_release_ready_to_review_required() -> None:
    """NEW CAPABILITY. One step down, and the reason names the superseded report."""

    capped = _classify(superseded_intermediate_report=SUPERSEDED).to_dict()

    assert capped["status"] == REVIEW_REQUIRED
    assert capped["strict_acceptance_eligible"] is False
    assert (
        f"{REASON_SUPERSEDED_INTERMEDIATE_REPORT}:"
        "post_normalization_contract_report@audit_round=2"
    ) in capped["reasons"]


def test_new_acceptance_the_cap_never_reaches_diagnostic_only() -> None:
    """NEW CAPABILITY, restriction 2. Exactly one step; the payload is preserved."""

    capped = _classify(superseded_intermediate_report=SUPERSEDED)
    assert capped.status == REVIEW_REQUIRED
    assert capped.status != DIAGNOSTIC_ONLY
    # Nothing about the measured coverage moved: a cap reads a status, not biology.
    assert capped.completeness == 1.0
    assert capped.missing_anchors == ()


def test_new_acceptance_the_cap_can_never_create_release_ready() -> None:
    """NEW CAPABILITY, restriction 3. Monotone -- it can only ever REMOVE a success.

    A run the technical chain already refused stays refused. The carrier is present
    and the reason is recorded, and the status does not move a millimetre upward.
    """

    refused = classify_release_status(
        CLEAN_COVERAGE,
        pipeline_executed=True,
        strict_gates_passed=False,
        superseded_intermediate_report=SUPERSEDED,
    )
    assert refused.status == DIAGNOSTIC_ONLY
    assert refused.strict_acceptance_eligible is False


def test_new_acceptance_the_reason_is_recorded_even_on_an_already_lowered_status() -> None:
    """NEW CAPABILITY. The ONE deliberate difference from the five existing caps.

    ORCH-728: a superseded error must never silently vanish from review metadata.
    On a status the chain already lowered, the cap performs no transition -- and it
    still states what was superseded, because that is the fact a human needs to be
    able to open afterwards.
    """

    lowered = classify_release_status(
        {**CLEAN_COVERAGE, "unmatched_terms": ["b"], "matched_terms": ["a"]},
        pipeline_executed=True,
        strict_gates_passed=True,
        superseded_intermediate_report=SUPERSEDED,
    )
    assert lowered.status == REVIEW_REQUIRED
    assert any(
        reason.startswith(REASON_SUPERSEDED_INTERMEDIATE_REPORT) for reason in lowered.reasons
    )


# --- the normalizer -------------------------------------------------------------


@pytest.mark.parametrize(
    "carrier, expected",
    [
        (None, ()),
        ([], ()),
        ({}, ()),
        ("", ()),
        (17, ()),
        ([{"phase": "audit_round", "errors": 2}], ()),  # no report name: not a finding
        ({"report": "r", "phase": "audit_round", "errors": 2}, ("r@audit_round=2",)),
        ([{"report": "r", "errors": 2}], ("r@(no phase)=2",)),
        ([{"report": "r", "phase": "audit_round"}], ("r@audit_round=?",)),
        ([{"report": "r", "phase": "audit_round", "errors": "x"}], ("r@audit_round=?",)),
        ("already:a reason", ("already:a reason",)),
        (
            [
                {"report": "b", "phase": "audit_round", "errors": 1},
                {"report": "a", "phase": "audit_round", "errors": 3},
                {"report": "a", "phase": "audit_round", "errors": 3},
            ],
            ("a@audit_round=3", "b@audit_round=1"),
        ),
    ],
)
def test_new_acceptance_the_normalizer_is_total_and_sorted(carrier, expected) -> None:
    """NEW CAPABILITY. Not recorded is not a failure, and order never varies."""

    assert superseded_report_reasons(carrier) == expected


# --- the frozen-record twin -----------------------------------------------------


def test_new_acceptance_the_frozen_record_cap_matches_the_classifier_cap() -> None:
    """NEW CAPABILITY. The same rule applied to an ALREADY-FROZEN record.

    It exists for the ordering fact C-087 already documented for the pre-freeze
    declination: the quarantine boundary freezes the classification inside the app,
    while "which contract boundary spoke last?" is a question only the batch driver
    can answer. So the verdict reaches the record, not the classifier call.
    """

    frozen = {"status": RELEASE_READY, "strict_acceptance_eligible": True, "reasons": ["x"]}
    capped = cap_release_for_superseded_intermediate_report(frozen, SUPERSEDED)

    assert capped["status"] == REVIEW_REQUIRED
    assert capped["strict_acceptance_eligible"] is False
    assert capped["reasons"][0] == "x"
    assert any(
        reason.startswith(REASON_SUPERSEDED_INTERMEDIATE_REPORT) for reason in capped["reasons"]
    )
    # The record handed in is never mutated.
    assert frozen == {"status": RELEASE_READY, "strict_acceptance_eligible": True, "reasons": ["x"]}


def test_new_acceptance_the_frozen_record_cap_never_promotes_and_never_invents() -> None:
    """NEW CAPABILITY. ``diagnostic_only`` keeps its status; an alien record is copied."""

    diagnostic = cap_release_for_superseded_intermediate_report(
        {"status": DIAGNOSTIC_ONLY}, SUPERSEDED
    )
    assert diagnostic["status"] == DIAGNOSTIC_ONLY
    # The finding is still stated -- it just changes nothing.
    assert any(
        reason.startswith(REASON_SUPERSEDED_INTERMEDIATE_REPORT)
        for reason in diagnostic["reasons"]
    )

    # A status this module cannot interpret is not interpreted: no cap, no reason,
    # no invented classification.
    alien = cap_release_for_superseded_intermediate_report({"status": "something_else"}, SUPERSEDED)
    assert alien == {"status": "something_else"}
    assert cap_release_for_superseded_intermediate_report(None, SUPERSEDED) == {}

    # And with no carrier at all the record is returned untouched.
    unchanged = cap_release_for_superseded_intermediate_report(
        {"status": RELEASE_READY, "strict_acceptance_eligible": True}
    )
    assert unchanged == {"status": RELEASE_READY, "strict_acceptance_eligible": True}
