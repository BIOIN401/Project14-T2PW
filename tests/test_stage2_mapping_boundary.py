from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from t2pw.mapping.map_ids import map_payload
from t2pw.pipeline.stage_contracts import validate_post_mapping


def _paper_like_payload() -> dict[str, Any]:
    return {
        "metadata": {"pathway_name": "Paper pathway", "organism": "Arabidopsis thaliana"},
        "entities": {
            "species": [{"name": "Arabidopsis thaliana", "taxonomy_id": "3702"}],
            "cell_types": [{"name": "mesophyll cell"}],
            "tissues": [{"name": "leaf"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [{"name": "substrate"}, {"name": "product"}],
            "element_collections": [{"name": "electron pool"}],
            "nucleic_acids": [{"name": "regulatory RNA"}],
            "proteins": [
                {"name": "resolved enzyme", "species": "Arabidopsis thaliana"},
                {"name": "unresolved enzyme", "species": "Arabidopsis thaliana"},
            ],
            "protein_complexes": [
                {
                    "name": "mixed complex",
                    "species": "Arabidopsis thaliana",
                    "components": ["resolved enzyme", "unresolved enzyme"],
                }
            ],
            "bounds": [{"name": "pathway boundary"}],
        },
        "biological_states": [
            {
                "name": "leaf cytosol",
                "species": "Arabidopsis thaliana",
                "subcellular_location": "cytosol",
            }
        ],
        "processes": {
            "reactions": [
                {
                    "name": "paper reaction",
                    "inputs": ["substrate"],
                    "outputs": ["product"],
                    "enzymes": [
                        {
                            "entity": "unresolved enzyme",
                            "entity_type": "protein",
                            "role": "catalyst",
                        }
                    ],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _unmapped_result() -> dict[str, Any]:
    return {
        "status": "unmapped",
        "reason": "no_match",
        "provider": "test",
        "source": "test",
        "confidence": 0.0,
        "candidates": [],
        "resolution": {
            "status": "unresolved",
            "issue": "no_match",
            "order_step": "test_lookup",
        },
    }


def _best_effort_protein_result() -> dict[str, Any]:
    return {
        "status": "mapped",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.42,
        "chosen_rule": "best_effort_fallback",
        "best_effort": True,
        "method": "fuzzy_search",
        "mapped_ids": {"uniprot": "Q9TEST"},
        "candidates": [
            {
                "accession": "Q9TEST",
                "score": 0.42,
                "organism": "Arabidopsis thaliana",
            }
        ],
        "resolution": {
            "status": "matched",
            "issue": "best_effort",
            "order_step": "api_uniprot",
        },
    }


def _run_mapping(
    payload: dict[str, Any],
    tmp_path: Path,
    *,
    allow_complex_wrapper_creation: bool,
    allow_structural_cleanup: bool | None,
) -> dict[str, Any]:
    def protein_result(*_: Any, name: str, **__: Any) -> dict[str, Any]:
        if name == "resolved enzyme":
            return _best_effort_protein_result()
        return _unmapped_result()

    kwargs: dict[str, Any] = {
        "cache_path": tmp_path / "mapping-cache.json",
        "id_source": "api",
        "use_cache": False,
        "allow_complex_wrapper_creation": allow_complex_wrapper_creation,
    }
    if allow_structural_cleanup is not None:
        kwargs["allow_structural_cleanup"] = allow_structural_cleanup

    with (
        patch(
            "t2pw.mapping.map_ids.hydrate_species_references",
            return_value={"hydrated": 0, "matched": 0, "novel": 0},
        ),
        patch("t2pw.mapping.map_ids._map_protein_with_strategy", side_effect=protein_result),
        patch("t2pw.mapping.map_ids._map_compound_with_strategy", return_value=_unmapped_result()),
        patch("t2pw.mapping.map_ids._map_complex_with_strategy", return_value=_unmapped_result()),
        patch.dict("os.environ", {"T2PW_LLM_PROTEIN_FALLBACK": "0"}),
    ):
        return map_payload(payload, **kwargs)


def test_stage2_annotation_only_preserves_processes_and_entity_inventory(tmp_path: Path) -> None:
    payload = _paper_like_payload()
    original = deepcopy(payload)

    result = _run_mapping(
        payload,
        tmp_path,
        allow_complex_wrapper_creation=False,
        allow_structural_cleanup=False,
    )
    mapped = result["payload"]

    assert payload == original
    assert mapped["processes"] == original["processes"]
    for bucket, rows in original["entities"].items():
        assert [row["name"] for row in mapped["entities"][bucket]] == [
            row["name"] for row in rows
        ]
    assert mapped["entities"]["protein_complexes"][0]["components"] == [
        "resolved enzyme",
        "unresolved enzyme",
    ]
    assert not any(
        row.get("generated") is True
        for row in mapped["entities"]["protein_complexes"]
    )
    assert all(row.get("name") != "Unknown" for row in mapped["entities"]["proteins"])
    assert result["report"]["summary"]["structural_cleanup_enabled"] is False
    assert result["report"]["summary"]["generated_wrappers_created"] == 0


def test_compatibility_default_keeps_stage6_structural_cleanup(tmp_path: Path) -> None:
    result = _run_mapping(
        _paper_like_payload(),
        tmp_path,
        allow_complex_wrapper_creation=False,
        allow_structural_cleanup=None,
    )

    assert [row["name"] for row in result["payload"]["entities"]["proteins"]] == [
        "resolved enzyme"
    ]
    assert result["payload"]["entities"]["protein_complexes"][0]["components"] == [
        "resolved enzyme"
    ]
    assert result["report"]["summary"]["structural_cleanup_enabled"] is True


def test_mapping_metadata_policy_and_uncertainty_are_explicit(tmp_path: Path) -> None:
    result = _run_mapping(
        _paper_like_payload(),
        tmp_path,
        allow_complex_wrapper_creation=False,
        allow_structural_cleanup=False,
    )
    mapped = result["payload"]

    assert validate_post_mapping(mapped)["ok"] is True
    for bucket in (
        "cell_types",
        "tissues",
        "subcellular_locations",
        "element_collections",
        "nucleic_acids",
        "bounds",
    ):
        resolution = mapped["entities"][bucket][0]["mapping_meta"]["resolution"]
        assert resolution == {
            "status": "not_applicable",
            "issue": "entity_bucket_not_mapped",
            "order_step": "not_applicable",
        }

    protein = mapped["entities"]["proteins"][0]
    assert protein["mapping_meta"]["best_effort"] is True
    assert protein["mapping_meta"]["method"] == "fuzzy_search"
    assert protein["mapping_meta"]["confidence"] == 0.42
    assert protein["mapping_meta"]["candidates"][0]["accession"] == "Q9TEST"
    protein_log = next(
        row
        for row in result["report"]["entities"]
        if row["entity_type"] == "protein" and row["name"] == "resolved enzyme"
    )
    assert protein_log["best_effort"] is True
    assert protein_log["confidence"] == 0.42
    assert protein_log["chosen_rule"] == "best_effort_fallback"
    assert result["report"]["summary"]["low_confidence_mappings"] == 1
    assert result["report"]["summary"]["best_effort_mappings"] == 1
    assert result["report"]["summary"]["entities_mapped"] == 1
    assert result["report"]["summary"]["entities_ambiguous"] == 0
    assert result["report"]["summary"]["entities_unmapped"] == 4
    assert result["report"]["mapping_metadata_policy"]["not_applicable_rows"] == 6


def test_unknown_fallback_remains_wrapper_enabled_stage6_only(tmp_path: Path) -> None:
    payload = _paper_like_payload()
    payload["entities"]["proteins"] = [
        {"name": "unresolved enzyme", "species": "Arabidopsis thaliana"}
    ]
    payload["entities"]["protein_complexes"] = []

    stage2 = _run_mapping(
        payload,
        tmp_path,
        allow_complex_wrapper_creation=False,
        allow_structural_cleanup=False,
    )
    assert [row["name"] for row in stage2["payload"]["entities"]["proteins"]] == [
        "unresolved enzyme"
    ]
    assert stage2["payload"]["entities"]["protein_complexes"] == []

    stage6 = _run_mapping(
        stage2["payload"],
        tmp_path,
        allow_complex_wrapper_creation=True,
        allow_structural_cleanup=True,
    )
    assert [row["name"] for row in stage6["payload"]["entities"]["proteins"]] == [
        "Unknown"
    ]
    assert stage6["payload"]["entities"]["proteins"][0]["pathbank_protein_id"] == 9659
    assert stage6["payload"]["entities"]["protein_complexes"][0]["generated"] is True
    assert stage6["report"]["summary"]["generated_wrappers_created"] == 1
    assert stage6["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 1
