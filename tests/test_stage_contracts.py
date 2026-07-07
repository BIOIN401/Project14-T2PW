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


def test_post_extraction_aborts_when_process_has_no_boundary_participants() -> None:
    payload = _raw_payload()
    payload["processes"]["reactions"][0].pop("inputs")
    payload["processes"]["reactions"][0].pop("outputs")

    with pytest.raises(StageContractError) as exc_info:
        validate_post_extraction(payload)

    assert {error["code"] for error in exc_info.value.errors} == {
        "process_missing_participants"
    }


def test_post_mapping_requires_species_and_mapping_meta_for_named_entities() -> None:
    payload = _raw_payload()
    payload["entities"]["species"] = []

    with pytest.raises(StageContractError) as exc_info:
        validate_post_mapping(payload)

    codes = {error["code"] for error in exc_info.value.errors}
    assert "species_required" in codes
    assert "entity_missing_mapping_meta" in codes


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
