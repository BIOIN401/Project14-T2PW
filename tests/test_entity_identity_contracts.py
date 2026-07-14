from __future__ import annotations

from unittest.mock import patch

import pytest

from t2pw.mapping.map_ids import map_payload
from t2pw.pipeline.entity_identity import (
    has_protein_external_identity,
    is_generated_complex_wrapper,
    protein_species_context,
    route_entity_for_mapping,
)
from t2pw.pipeline.process_normalizer import normalize_process_payload
from t2pw.pipeline.stage_contracts import (
    StageContractError,
    validate_post_normalization,
    validate_post_remap,
)


def test_shared_identity_helpers_cover_current_and_legacy_wrapper_rows() -> None:
    assert is_generated_complex_wrapper({"generated": True})
    assert is_generated_complex_wrapper(
        {"mapping_meta": {"resolution": {"order_step": "novel_enzyme_single_component_complex"}}}
    )
    assert not is_generated_complex_wrapper({"generation_reason": "manual_complex"})
    protein = {"mapped_ids": {"uniprot": "P12345"}, "species_ref": {"name": "Homo sapiens"}}
    assert has_protein_external_identity(protein)
    assert protein_species_context(protein) == "Homo sapiens"


def test_shared_mapping_classifier_preserves_biochemical_colon_routes() -> None:
    assert route_entity_for_mapping("OPC-6:0-CoA", "compound")["route"] == "compound"
    assert route_entity_for_mapping("receptor:kinase", "compound")["route"] == "complex"


def test_map_payload_can_disable_wrapper_creation(tmp_path) -> None:
    payload = {"entities": {"species": [], "proteins": [], "compounds": []}, "processes": {"reactions": []}}
    with patch("t2pw.mapping.map_ids._rewrite_reaction_protein_enzymes_to_complexes") as rewrite:
        result = map_payload(
            payload,
            cache_path=tmp_path / "mapping-cache.json",
            id_source="api",
            allow_complex_wrapper_creation=False,
        )
    rewrite.assert_not_called()
    assert result["report"]["enzyme_complex_conversion"]["skipped"] is True


def test_normalizer_emits_canonical_actor_fields() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [{"name": "Enzyme", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P12345"}}],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [{"name": "A to B", "inputs": ["A"], "outputs": ["B"], "enzymes": [{"protein": "Enzyme"}]}],
            "transports": [],
            "interactions": [],
        },
    }
    normalized, _ = normalize_process_payload(payload)
    actor = normalized["processes"]["reactions"][0]["enzymes"][0]
    assert actor["entity"] == "Enzyme"
    assert actor["entity_type"] == "protein"


def test_post_normalization_reports_noncanonical_actor_rows() -> None:
    payload = {
        "entities": {},
        "processes": {"reactions": [{"enzymes": [{"protein": "Legacy enzyme"}]}]},
    }
    report = validate_post_normalization(payload, {"ok": True, "errors": []})
    assert report["ok"] is False
    assert report["errors"][0]["code"] == "actor_schema_not_canonical"


def test_post_remap_requires_generated_wrapper_component_identity_and_species() -> None:
    payload = {
        "entities": {
            "proteins": [{"name": "Enzyme", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P12345"}}],
            "protein_complexes": [{"name": "Enzyme complex", "generated": True, "components": [{"name": "Enzyme"}]}],
        },
        "processes": {},
    }
    assert validate_post_remap(payload)["ok"] is True

    del payload["entities"]["proteins"][0]["mapped_ids"]
    with pytest.raises(StageContractError) as exc_info:
        validate_post_remap(payload)
    assert "generated_wrapper_component_missing_external_identity" in {
        error["code"] for error in exc_info.value.errors
    }
