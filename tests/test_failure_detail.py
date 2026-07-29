"""The failure-detail channel: gate and contract errors carry what they rejected.

Two properties are load-bearing and are what this suite pins:

1. A failure says what the payload actually held, not only which rule fired.
2. It does so without ever emitting raw evidence, whose observed peak was
   139,576 characters in one field -- enough to wedge the browser tab the detail
   is meant to inform.

They pull against each other, which is why both are tested on the same payloads.
"""

from __future__ import annotations

import json

import pytest

from t2pw.pipeline.failure_detail import (
    MAX_VALUE_CHARS,
    census,
    clip,
    closest_names,
    detail as build_detail,
    headline,
    row_digest,
    scrub_detail,
)
from t2pw.pipeline.process_normalizer import (
    GateValidationError,
    run_strict_post_normalization_gates,
)
from t2pw.pipeline.stage_contracts import StageContractError, validate_post_extraction


#: The size actually observed in a run's evidence field. Used verbatim so the
#: test fails for the reason the guard exists rather than for a round number.
OBSERVED_EVIDENCE_CHARS = 139_576


def _evidence_blob(size: int = OBSERVED_EVIDENCE_CHARS) -> str:
    text = "Lipid A biosynthesis proceeds via the LpxC deacetylase. "
    return (text * (size // len(text) + 1))[:size]


def _payload_with_evidence() -> dict:
    blob = _evidence_blob()
    return {
        "entities": {
            "proteins": [
                {
                    "name": "LpxC",
                    "species": "Escherichia coli",
                    "uniprot": "P0A725",
                    "evidence": [{"text": blob, "section": "Results"}],
                    "source_refs": [blob],
                }
            ],
            "protein_complexes": [],
            "compounds": [],
        },
        "processes": {
            "reactions": [
                {
                    "inputs": ["UDP-GlcNAc"],
                    "outputs": ["lipid X"],
                    "enzymes": [
                        {"entity": "LpxC2", "entity_type": "protein", "evidence": blob}
                    ],
                }
            ]
        },
    }


def _gate_errors(payload: dict) -> list[dict]:
    with pytest.raises(GateValidationError) as exc:
        run_strict_post_normalization_gates(payload)
    return exc.value.details["errors"]


# ---------------------------------------------------------------------------
# The evidence bound.
# ---------------------------------------------------------------------------
def test_raw_evidence_never_reaches_a_gate_error() -> None:
    blob = _evidence_blob()
    serialized = json.dumps(_gate_errors(_payload_with_evidence()), ensure_ascii=False)

    # Not "the whole blob is absent" -- no recognisable RUN of it may survive,
    # which is what a naive per-field clip would still leak.
    assert blob[:200] not in serialized
    assert "proceeds via the LpxC deacetylase. Lipid A" not in serialized


def test_the_elided_size_is_reported_rather_than_the_field_being_dropped() -> None:
    """A dropped key reads as "there was no evidence", which is a different bug."""

    errors = _gate_errors(_payload_with_evidence())
    actor_error = next(e for e in errors if "Unknown protein/modifier" in e["reason"])

    assert actor_error["detail"]["actor_row"]["evidence"] == (
        f"{OBSERVED_EVIDENCE_CHARS} chars elided"
    )


def test_the_whole_error_list_stays_orders_of_magnitude_under_the_payload() -> None:
    payload = _payload_with_evidence()
    raw = len(json.dumps(payload, ensure_ascii=False))
    serialized = len(json.dumps(_gate_errors(payload), ensure_ascii=False))

    assert raw > 250_000, "fixture no longer reproduces the size problem"
    assert serialized < 5_000, f"detail channel grew to {serialized} chars"


def test_censusing_is_idempotent_so_the_count_survives_nested_building() -> None:
    """``build_detail(row=row_digest(row))`` then ``_add_error`` scrubs 3 times.

    Without idempotence each pass measures the previous census string instead of
    the data: "139576 chars elided" -> "19 chars elided" -> "15 chars elided".
    The number silently becomes a fact about itself.
    """

    blob = _evidence_blob()
    row = {"name": "LpxC", "evidence": blob}

    once = row_digest(row)
    thrice = scrub_detail(build_detail(row=row_digest(row)))

    assert once["evidence"] == f"{OBSERVED_EVIDENCE_CHARS} chars elided"
    assert thrice["row"]["evidence"] == once["evidence"]
    assert census(census(census(blob))) == census(blob)


def test_a_long_scalar_is_clipped_with_its_dropped_count_stated() -> None:
    clipped = clip("z" * 5_000)

    assert clipped.startswith("z" * MAX_VALUE_CHARS)
    assert clipped.endswith(f"(+{5_000 - MAX_VALUE_CHARS} chars elided)")


# ---------------------------------------------------------------------------
# What the detail is for: naming the value that was rejected.
# ---------------------------------------------------------------------------
def test_an_unknown_actor_names_the_registry_entry_it_nearly_matched() -> None:
    errors = _gate_errors(_payload_with_evidence())
    actor_error = next(e for e in errors if "Unknown protein/modifier" in e["reason"])
    detail = actor_error["detail"]

    assert detail["actor_name"] == "LpxC2"
    assert detail["did_you_mean"] == ["LpxC"]
    # An empty registry and a registry with no near match are different upstream
    # bugs, so the size is recorded even when a suggestion is found.
    assert detail["registry_size"] == 1


def test_a_genuinely_undeclared_actor_is_distinguishable_from_a_near_miss() -> None:
    payload = _payload_with_evidence()
    payload["processes"]["reactions"][0]["enzymes"][0]["entity"] = "ZZZ_not_a_protein"

    errors = _gate_errors(payload)
    detail = next(e for e in errors if "Unknown protein/modifier" in e["reason"])["detail"]

    assert "did_you_mean" not in detail  # empty keys are dropped, not sent as []
    assert detail["registry_sample"] == ["LpxC"]


@pytest.mark.parametrize(
    "payload, expected_type, key_present",
    [
        ({"entities": {}, "processes": []}, "list", True),
        ({"entities": {}, "processes": None}, "NoneType", True),
        ({"entities": {}}, "NoneType", False),
    ],
)
def test_one_message_three_upstream_bugs_now_separated(
    payload: dict, expected_type: str, key_present: bool
) -> None:
    """"Payload must include a processes object" covers wrong-type, null and absent."""

    with pytest.raises(StageContractError) as exc:
        validate_post_extraction(payload)
    issue = next(e for e in exc.value.report["errors"] if e["code"] == "processes_required")

    assert issue["detail"]["found_type"] == expected_type
    assert issue["detail"]["key_present"] is key_present
    # The sibling keys say which stage produced the payload.
    assert "entities" in issue["detail"]["payload_keys"]


def test_a_nameless_entity_row_carries_the_row_that_has_no_name() -> None:
    with pytest.raises(StageContractError) as exc:
        validate_post_extraction(
            {
                "entities": {"proteins": [{"uniprot": "P0A725", "species": "E. coli"}]},
                "processes": {},
            }
        )
    issue = next(e for e in exc.value.report["errors"] if e["code"] == "entity_missing_name")

    assert issue["detail"]["uniprot"] == "P0A725"


# ---------------------------------------------------------------------------
# Back-compatibility: consumers that read the old two-key shape.
# ---------------------------------------------------------------------------
def test_an_error_without_detail_keeps_its_original_two_key_shape() -> None:
    """``batch.driver`` and the audit-retry prompts project to {path, reason}.

    An always-present ``detail`` key -- even an empty one -- would show up in
    ``review_flags.json`` diffs and in the JSON handed to the repair LLM for
    every error that has nothing to add.
    """

    payload = _payload_with_evidence()
    payload["locked_reaction_filter_report"] = "not an object"

    errors = _gate_errors(payload)
    bare = next(e for e in errors if e["path"] == "/processes")  # validator exception

    assert set(bare) >= {"path", "reason"}
    assert all(set(e) <= {"path", "reason", "detail"} for e in errors)


def test_every_gate_error_still_has_a_string_path_and_reason() -> None:
    for error in _gate_errors(_payload_with_evidence()):
        assert isinstance(error["path"], str) and error["path"]
        assert isinstance(error["reason"], str) and error["reason"]


# ---------------------------------------------------------------------------
# The one-line summary shared by the app caption and the batch flag rows.
# ---------------------------------------------------------------------------
def test_headline_leads_with_the_field_that_resolves_the_failure() -> None:
    summary = headline(
        {"registry_size": 40, "did_you_mean": ["LpxC", "LpxD"], "actor_name": "LpxC2"}
    )

    assert summary.startswith("did_you_mean=LpxC, LpxD")
    assert "actor_name=LpxC2" in summary


def test_headline_is_empty_when_nothing_is_worth_a_caption() -> None:
    assert headline({}) == ""
    assert headline(None) == ""
    assert headline({"unrecognised_key": "value"}) == ""


def test_headline_reports_falsey_findings_that_are_real() -> None:
    """``registry_size=0`` is the most informative case, not an absent one."""

    assert "registry_size=0" in headline({"registry_size": 0})


def test_build_detail_drops_blanks_but_keeps_zero_and_false() -> None:
    built = build_detail(
        name="", missing=None, empty=[], degree=0, generated=False, kept="yes"
    )

    assert built == {"degree": 0, "generated": False, "kept": "yes"}


def test_closest_names_ranks_the_best_match_first() -> None:
    # Several candidates may clear the cutoff -- a short protein name is close to
    # its whole family -- so this pins the ORDER, not the count. The reviewer
    # reads left to right.
    assert closest_names("LpxC2", ["LpxD", "LpxC"])[0] == "LpxC"


def test_closest_names_returns_nothing_rather_than_a_bad_guess() -> None:
    """An empty list is a finding: the reference is undeclared, not misspelled."""

    assert closest_names("ZZZZZZ", ["LpxC", "LpxD"]) == []
    assert closest_names("", ["LpxC"]) == []
    assert closest_names("LpxC", []) == []


def test_closest_names_is_capped_so_a_suggestion_list_stays_readable() -> None:
    family = [f"Lpx{letter}" for letter in "ABCDEFGHIJ"]

    assert len(closest_names("LpxC2", family)) <= 3
