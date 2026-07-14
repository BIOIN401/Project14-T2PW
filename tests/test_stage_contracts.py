from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.stage_contracts import (  # noqa: E402
    StageContractError,
    validate_post_extraction,
    validate_post_mapping,
    validate_post_normalization,
    validate_pre_export,
)


def _raw_payload() -> dict:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "compounds": [{"name": "Glucose"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "Glucose phosphorylation",
                    "inputs": ["Glucose"],
                    "outputs": ["Glucose 6-phosphate"],
                }
            ],
        },
    }


def test_post_extraction_aborts_when_entity_name_is_missing() -> None:
    payload = _raw_payload()
    payload["entities"]["compounds"][0].pop("name")

    with pytest.raises(StageContractError) as exc_info:
        validate_post_extraction(payload)

    assert exc_info.value.stage == "post_extraction"
    assert {error["code"] for error in exc_info.value.errors} == {"entity_missing_name"}


def test_post_extraction_does_not_weaken_malformed_reaction_detection() -> None:
    payload = _raw_payload()
    payload["processes"]["reactions"][0]["inputs"] = [" ", {}]
    payload["processes"]["reactions"][0]["outputs"] = []

    with pytest.raises(StageContractError) as exc_info:
        validate_post_extraction(payload)

    assert {error["code"] for error in exc_info.value.errors} == {
        "reaction_missing_participants"
    }
    assert exc_info.value.errors[0]["pointer"] == "/processes/reactions/0"
    assert exc_info.value.errors[0]["required_any"] == ["inputs", "outputs"]


@pytest.mark.parametrize(
    ("bucket", "row"),
    [
        ("reactions", {"name": "R1", "inputs": [{"compound": "ATP"}]}),
        ("reactions", {"name": "R2", "outputs": ["ADP"]}),
        ("transports", {"name": "T1", "cargo": "Glucose"}),
        (
            "transports",
            {"name": "T2", "elements_with_states": [{"element": "Glucose", "side": "left"}]},
        ),
        (
            "interactions",
            {"name": "I1", "entity_1": "Hexokinase", "entity_2": "Glucose"},
        ),
        (
            "interactions",
            {
                "name": "I2",
                "participants": [{"entity": "Hexokinase"}],
            },
        ),
        (
            "reaction_coupled_transports",
            {"name": "RCT1", "reaction": "R1", "transport": "T1"},
        ),
        (
            "reaction_coupled_transports",
            {"name": "RCT2", "elements_with_states": [{"element": "Proton"}]},
        ),
        ("sub_pathways", {"name": "Glycolysis"}),
        ("sub_pathways", {"reference_pathway_id": 123}),
        ("sub_pathways", {"alttext": "Related glycolysis pathway"}),
    ],
    ids=[
        "reaction-input",
        "reaction-output",
        "transport-cargo",
        "transport-elements",
        "interaction-endpoints",
        "interaction-participants",
        "reaction-coupled-references",
        "reaction-coupled-elements",
        "sub-pathway-name",
        "sub-pathway-reference",
        "sub-pathway-alttext",
    ],
)
def test_post_extraction_accepts_valid_bucket_specific_process_shapes(
    bucket: str, row: dict
) -> None:
    payload = _raw_payload()
    payload["processes"] = {bucket: [row]}

    report = validate_post_extraction(payload)

    assert report["ok"] is True


def test_post_extraction_accepts_valid_interaction_without_reaction_fields() -> None:
    payload = _raw_payload()
    payload["processes"] = {
        "interactions": [
            {
                "name": "Hexokinase catalysis",
                "entity_1": "Hexokinase",
                "entity_2": "Glucose phosphorylation",
                "relationship": "catalyzes",
            }
        ]
    }

    report = validate_post_extraction(payload)

    assert report["ok"] is True


@pytest.mark.parametrize(
    ("bucket", "row", "expected_code", "required_any"),
    [
        (
            "reactions",
            {"name": "R1", "inputs": ["", {}], "outputs": []},
            "reaction_missing_participants",
            ["inputs", "outputs"],
        ),
        (
            "transports",
            {"name": "T1", "cargo": " ", "elements_with_states": [{"element": ""}]},
            "transport_missing_cargo",
            ["cargo", "elements_with_states[].element"],
        ),
        (
            "interactions",
            {"name": "I1", "entity_1": "Hexokinase", "participants": [{"entity": " "}]},
            "interaction_missing_participants",
            [
                ["entity_1", "entity_2"],
                "participants[].{entity,protein,protein_complex,name}",
            ],
        ),
        (
            "reaction_coupled_transports",
            {"name": "RCT1", "reaction": "R1", "elements_with_states": [{}]},
            "reaction_coupled_transport_missing_structure",
            [["reaction", "transport"], "elements_with_states[].element"],
        ),
        (
            "sub_pathways",
            {"name": " ", "reference_pathway_id": 0, "alttext": ""},
            "sub_pathway_missing_identity",
            ["name", "reference_pathway_id", "alttext"],
        ),
    ],
    ids=[
        "reaction",
        "transport",
        "interaction",
        "reaction-coupled-transport",
        "sub-pathway",
    ],
)
def test_post_extraction_rejects_invalid_bucket_specific_process_shapes(
    bucket: str,
    row: dict,
    expected_code: str,
    required_any: list,
) -> None:
    payload = _raw_payload()
    payload["processes"] = {bucket: [row]}

    with pytest.raises(StageContractError) as exc_info:
        validate_post_extraction(payload)

    error = next(error for error in exc_info.value.errors if error["code"] == expected_code)
    assert error["pointer"] == f"/processes/{bucket}/0"
    assert error["bucket"] == bucket
    assert error["required_any"] == required_any


def test_post_extraction_allows_additive_process_bucket_without_reaction_fields() -> None:
    payload = _raw_payload()
    payload["processes"] = {"custom_events": [{"name": "Custom event", "subject": "ATP"}]}

    report = validate_post_extraction(payload)

    assert report["ok"] is True


def test_post_extraction_requires_object_rows_in_additive_process_buckets() -> None:
    payload = _raw_payload()
    payload["processes"] = {"custom/events": ["not an object"]}

    with pytest.raises(StageContractError) as exc_info:
        validate_post_extraction(payload)

    error = next(error for error in exc_info.value.errors if error["code"] == "process_not_object")
    assert error["pointer"] == "/processes/custom~1events/0"
    assert error["bucket"] == "custom/events"


def test_post_mapping_requires_species_and_mapping_meta_for_named_entities() -> None:
    payload = _raw_payload()
    payload["entities"]["species"] = []

    with pytest.raises(StageContractError) as exc_info:
        validate_post_mapping(payload)

    codes = {error["code"] for error in exc_info.value.errors}
    assert "species_required" in codes
    assert "entity_missing_mapping_meta" in codes


def test_post_mapping_requires_structured_mapping_resolution() -> None:
    payload = _raw_payload()
    payload["entities"]["species"][0]["mapping_meta"] = {
        "resolution": {
            "status": "matched",
            "issue": 42,
        }
    }
    payload["entities"]["compounds"][0]["mapping_meta"] = {
        "resolution": {"status": ""}
    }

    with pytest.raises(StageContractError) as exc_info:
        validate_post_mapping(payload)

    codes = {error["code"] for error in exc_info.value.errors}
    assert "mapping_resolution_field_invalid" in codes
    assert "mapping_resolution_status_required" in codes


def test_post_normalization_semantic_gate_failure_feeds_audit_without_raising() -> None:
    gate_report = {
        "ok": False,
        "errors": [
            {
                "code": "protein_missing_external_identity",
                "message": "Protein has no UniProt or DrugBank identity.",
                "pointer": "/entities/proteins/0",
            }
        ],
    }

    report = validate_post_normalization(_raw_payload(), gate_report)

    assert report["ok"] is False
    assert report["effect_on_failure"] == "feed_audit"
    assert report["errors"] == gate_report["errors"]


def test_pre_export_wraps_required_pwml_contract_failures() -> None:
    payload = {"entities": {}, "processes": {}}

    with pytest.raises(StageContractError) as exc_info:
        validate_pre_export(payload, strict_db=False)

    assert exc_info.value.stage == "pre_export"
    assert exc_info.value.report["pwml_contract_report"]["ok"] is False
    assert "pathway_missing_name" in {error["code"] for error in exc_info.value.errors}
