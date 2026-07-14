from __future__ import annotations

import json
from copy import deepcopy

import pytest

from t2pw.pipeline.payload_models import (
    RuntimeSchemaValidationError,
    validate_payload_shape,
)
from t2pw.pipeline.stage_contracts import (
    StageContractError,
    validate_post_remap,
    validate_runtime_payload_contract,
)


def _mapping_meta(status: str = "matched") -> dict:
    return {
        "provider": "fixture",
        "candidates": [],
        "resolution": {
            "status": status,
            "issue": "no_match" if status == "unresolved" else "",
            "order_step": "fixture",
        },
    }


def _payload() -> dict:
    return {
        "metadata": {
            "pathway_name": "Example pathway",
            "organism": "Homo sapiens",
            "width": 3200,
            "provider_extension": {"reviewed_by": "fixture"},
        },
        "entities": {
            "species": [
                {
                    "name": "Homo sapiens",
                    "taxonomy_id": "9606",
                    "mapping_meta": _mapping_meta(),
                }
            ],
            "subcellular_locations": [
                {"name": "cytosol", "mapping_meta": _mapping_meta("unresolved")}
            ],
            "compounds": [
                {"name": "A", "mapping_meta": _mapping_meta()},
                {"name": "B", "mapping_meta": _mapping_meta("unresolved")},
            ],
            "proteins": [
                {
                    "name": "Enzyme",
                    "species": "Homo sapiens",
                    "species_ref": {"name": "Homo sapiens", "pathbank_species_id": 1},
                    "mapped_ids": {"uniprot": "P12345"},
                    "mapping_meta": _mapping_meta(),
                }
            ],
            "protein_complexes": [
                {
                    "name": "Enzyme complex",
                    "species": "Homo sapiens",
                    "generated": True,
                    "generation_reason": "single_protein_pathwhiz_wrapper",
                    "components": [{"name": "Enzyme", "stoichiometry": 1}],
                    "mapping_meta": _mapping_meta("unresolved"),
                }
            ],
            "provider_bucket": [{"opaque": [1, "two", {"three": True}]}],
        },
        "biological_states": [
            {
                "name": "cytosol",
                "species": "Homo sapiens",
                "subcellular_location": "cytosol",
            }
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "A", "biological_state": "cytosol"},
                {"compound": "B", "biological_state": "cytosol"},
            ],
            "protein_locations": [
                {"protein": "Enzyme", "biological_state": "cytosol"}
            ],
        },
        "processes": {
            "reactions": [
                {
                    "name": "A to B",
                    "inputs": ["A"],
                    "outputs": [{"compound": "B", "stoichiometry": 1}],
                    "enzymes": [
                        {
                            "entity": "Enzyme complex",
                            "entity_type": "protein_complex",
                            "role": "catalyst",
                        }
                    ],
                    "modifiers": [],
                    "spontaneous": False,
                }
            ],
            "transports": [],
            "interactions": [],
        },
        "enrichment_extension": {"provider": "fixture", "rows": [1, 2]},
    }


@pytest.mark.parametrize(
    "boundary",
    [
        "post_extraction",
        "post_mapping",
        "post_normalization",
        "post_audit",
        "post_remap",
        "post_enrichment",
        "pre_export",
    ],
)
def test_representative_payload_passes_each_runtime_boundary(boundary: str) -> None:
    report = validate_payload_shape(_payload(), boundary=boundary)  # type: ignore[arg-type]

    assert report["ok"] is True
    assert report["summary"] == {"error_count": 0, "warning_count": 0}
    json.dumps(report)


def test_nested_wrong_type_has_precise_json_pointer() -> None:
    payload = _payload()
    payload["processes"]["reactions"][0]["enzymes"][0]["entity"] = ["Enzyme"]

    report = validate_payload_shape(payload, boundary="post_normalization")

    assert any(
        error["code"] == "runtime_schema_type_error"
        and error["pointer"] == "/processes/reactions/0/enzymes/0/entity"
        and error["expected"] == "string"
        and error["received"] == "array"
        for error in report["errors"]
    )


def test_spontaneous_string_is_not_coerced_to_boolean() -> None:
    payload = _payload()
    payload["processes"]["reactions"][0]["spontaneous"] = "true"

    report = validate_payload_shape(payload, boundary="post_extraction")

    assert any(
        error["pointer"] == "/processes/reactions/0/spontaneous"
        and error["expected"] == "boolean"
        for error in report["errors"]
    )


def test_actor_item_must_be_an_object_at_the_exact_actor_pointer() -> None:
    payload = _payload()
    payload["processes"]["reactions"][0]["enzymes"][0] = "Enzyme complex"

    report = validate_payload_shape(payload, boundary="post_normalization")

    assert any(
        error["pointer"] == "/processes/reactions/0/enzymes/0"
        and error["expected"] == "object"
        and error["received"] == "string"
        for error in report["errors"]
    )


def test_unknown_additive_metadata_is_allowed_and_payload_is_unchanged() -> None:
    payload = _payload()
    before = deepcopy(payload)

    report = validate_payload_shape(payload, boundary="post_enrichment")

    assert report["ok"] is True
    assert payload == before


def test_report_mode_returns_errors_without_raising() -> None:
    payload = _payload()
    payload["entities"]["compounds"] = "not a list"

    report = validate_payload_shape(payload, boundary="post_extraction", mode="report")

    assert report["ok"] is False
    assert report["mode"] == "report"
    assert report["summary"]["error_count"] >= 1


def test_enforce_mode_raises_with_the_structured_report() -> None:
    payload = _payload()
    payload["entities"]["compounds"] = "not a list"

    with pytest.raises(RuntimeSchemaValidationError) as exc_info:
        validate_payload_shape(payload, boundary="post_extraction", mode="enforce")

    report = exc_info.value.report
    assert report["ok"] is False
    assert report["mode"] == "enforce"
    assert exc_info.value.errors == report["errors"]


def test_post_mapping_requires_metadata_but_allows_unmapped_resolution() -> None:
    payload = _payload()
    payload["entities"]["compounds"][0].pop("mapping_meta")

    report = validate_payload_shape(payload, boundary="post_mapping")

    assert any(
        error["code"] == "entity_missing_mapping_meta"
        and error["pointer"] == "/entities/compounds/0/mapping_meta"
        for error in report["errors"]
    )
    payload["entities"]["compounds"][0]["mapping_meta"] = _mapping_meta("unresolved")
    assert validate_payload_shape(payload, boundary="post_mapping")["ok"] is True


def test_post_normalization_requires_canonical_actor_fields() -> None:
    payload = _payload()
    payload["processes"]["reactions"][0]["enzymes"][0] = {
        "protein_complex": "Enzyme complex"
    }

    report = validate_payload_shape(payload, boundary="post_normalization")

    assert any(
        error["code"] == "actor_schema_not_canonical"
        and error["pointer"] == "/processes/reactions/0/enzymes/0"
        for error in report["errors"]
    )


def test_post_remap_requires_structured_generated_wrapper_components() -> None:
    payload = _payload()
    payload["entities"]["protein_complexes"][0]["components"] = ["Enzyme"]

    report = validate_payload_shape(payload, boundary="post_remap")

    assert any(
        error["code"] == "generated_wrapper_component_not_structured"
        and error["pointer"] == "/entities/protein_complexes/0/components/0"
        for error in report["errors"]
    )


def test_runtime_shape_leaves_cross_record_wrapper_identity_to_semantic_contract() -> None:
    payload = _payload()
    payload["entities"]["protein_complexes"][0]["components"][0]["name"] = "Missing protein"

    assert validate_payload_shape(payload, boundary="post_remap")["ok"] is True
    with pytest.raises(StageContractError) as exc_info:
        validate_post_remap(payload)
    assert any(
        error["code"] == "generated_wrapper_component_protein_unresolved"
        for error in exc_info.value.errors
    )


def test_stage_contract_adapter_is_report_first_and_deduplicates() -> None:
    payload = _payload()
    payload["processes"]["reactions"][0]["enzymes"][0] = {
        "protein_complex": "Enzyme complex"
    }
    contract_report = {
        "ok": True,
        "stage": "post_normalization",
        "contract_type": "semantic",
        "effect_on_failure": "feed_audit",
        "errors": [],
        "warnings": [
            {
                "code": "actor_schema_not_canonical",
                "pointer": "/processes/reactions/0/enzymes/0",
                "message": "existing",
            }
        ],
        "summary": {"error_count": 0, "warning_count": 1},
    }

    runtime_report = validate_runtime_payload_contract(
        payload,
        boundary="post_normalization",
        contract_report=contract_report,
    )

    assert runtime_report["ok"] is False
    identities = [
        (warning["code"], warning.get("pointer"))
        for warning in contract_report["warnings"]
    ]
    assert identities.count(
        ("actor_schema_not_canonical", "/processes/reactions/0/enzymes/0")
    ) == 1
    assert contract_report["ok"] is True


def test_stage_contract_adapter_enforce_mode_preserves_stage_error_semantics() -> None:
    payload = _payload()
    payload["processes"]["reactions"][0]["spontaneous"] = "true"

    with pytest.raises(StageContractError) as exc_info:
        validate_runtime_payload_contract(
            payload,
            boundary="post_normalization",
            mode="enforce",
        )

    assert exc_info.value.stage == "post_normalization"
    assert exc_info.value.report["contract_type"] == "structural"
    assert exc_info.value.report["effect_on_failure"] == "abort"
    assert exc_info.value.report["runtime_schema_report"]["mode"] == "enforce"
