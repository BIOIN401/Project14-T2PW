from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from t2pw.mapping.enrich_entities import enrich_payload
from t2pw.mapping.map_ids import map_payload
from t2pw.pipeline.process_normalizer import normalize_process_payload
from t2pw.pipeline.stage_contracts import validate_post_normalization, validate_post_remap
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir, validate_required_pwml_contract
from t2pw.pwml.validate import discover_structure_signature
from t2pw.pwml.writer import DeterministicPwmlBuilder


ROOT = Path(__file__).resolve().parents[1]


def _payload(*, protein_id: str = "", include_enzyme: bool = True, reaction_count: int = 1) -> dict:
    protein = {
        "name": "N-methyl nucleosidase",
        "species": "Camellia sinensis",
        "organism": "Camellia sinensis",
    }
    if protein_id:
        protein["mapped_ids"] = {"uniprot": protein_id}
    reactions = []
    for idx in range(reaction_count):
        reactions.append(
            {
                "name": f"caffeine demethylation {idx + 1}",
                "inputs": ["caffeine"],
                "outputs": ["theobromine"],
                "biological_state": "leaf cytosol",
                "enzymes": (
                    [
                        {
                            "entity": "N-methyl nucleosidase",
                            "entity_type": "protein",
                            "role": "catalyst",
                        }
                    ]
                    if include_enzyme
                    else []
                ),
            }
        )
    return {
        "metadata": {
            "pathway_name": "Caffeine degradation",
            "pathway_subject": "Metabolic",
            "organism": "Camellia sinensis",
            "width": 3200,
            "height": 1400,
        },
        "entities": {
            "species": [
                {
                    "name": "Camellia sinensis",
                    "taxonomy_id": "4442",
                    "pathbank_species_id": 99,
                }
            ],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [{"name": "caffeine"}, {"name": "theobromine"}],
            "proteins": [protein],
            "protein_complexes": [],
        },
        "biological_states": [
            {
                "name": "leaf cytosol",
                "species": "Camellia sinensis",
                "subcellular_location": "cytosol",
            }
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "caffeine", "biological_state": "leaf cytosol"},
                {"compound": "theobromine", "biological_state": "leaf cytosol"},
            ],
            "protein_locations": [
                {"protein": "N-methyl nucleosidase", "biological_state": "leaf cytosol"}
            ],
        },
        "processes": {"reactions": reactions, "transports": [], "interactions": []},
    }


def _transporter_payload(*, transporter_name: str = "ABCG-116", add_interaction_ref: bool = False) -> dict:
    payload = {
        "metadata": {
            "pathway_name": "Caffeine export",
            "pathway_subject": "Metabolic",
            "organism": "Camellia sinensis",
            "width": 3200,
            "height": 1400,
        },
        "entities": {
            "species": [
                {
                    "name": "Camellia sinensis",
                    "taxonomy_id": "4442",
                    "pathbank_species_id": 99,
                }
            ],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [{"name": "caffeine"}],
            "proteins": [
                {
                    "name": transporter_name,
                    "species": "Camellia sinensis",
                    "organism": "Camellia sinensis",
                }
            ],
            "protein_complexes": [],
        },
        "biological_states": [
            {
                "name": "leaf cytosol",
                "species": "Camellia sinensis",
                "subcellular_location": "cytosol",
            }
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "caffeine", "biological_state": "leaf cytosol"},
            ],
            "protein_locations": [
                {"protein": transporter_name, "biological_state": "leaf cytosol"}
            ],
        },
        "processes": {
            "reactions": [],
            "transports": [
                {
                    "name": "caffeine export",
                    "cargo": "caffeine",
                    "from_biological_state": "leaf cytosol",
                    "to_biological_state": "leaf cytosol",
                    "transporters": [
                        {
                            "entity": transporter_name,
                            "entity_type": "protein",
                            "role": "transporter",
                        }
                    ],
                }
            ],
            "interactions": [],
        },
    }
    if add_interaction_ref:
        payload["processes"]["interactions"] = [
            {
                "name": "transporter observation",
                "entity_1": transporter_name,
                "entity_2": "caffeine",
                "relationship": "binds",
            }
        ]
    return payload


def _unmapped_result() -> dict:
    return {
        "status": "unmapped",
        "reason": "no_match",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.0,
        "candidates": [],
        "resolution": {
            "status": "unresolved",
            "issue": "no_match",
            "order_step": "api_uniprot",
        },
    }


def _mapped_result(accession: str = "Q9TEST") -> dict:
    """A real match, with the candidate row that describes it.

    The candidate is not decoration: the identity ladder fails closed, so an
    accession that nothing in the result describes is unverified evidence rather
    than a verified mapping. A resolver that really matched this protein returns
    this row.
    """

    return {
        "status": "mapped",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.99,
        "chosen_rule": "reviewed_exact",
        "resolved_name": "N-methyl nucleosidase",
        "mapped_ids": {"uniprot": accession},
        "candidates": [
            {
                "accession": accession,
                "uniprot": accession,
                "protein_name": "N-methyl nucleosidase",
                "organism": "Camellia sinensis",
                "score": 0.99,
            }
        ],
        "resolution": {"status": "matched", "order_step": "api_uniprot"},
    }


def _map(
    payload: dict,
    tmp_path: Path,
    *,
    protein_result: dict,
    allow_complex_wrapper_creation: bool = True,
) -> dict:
    with (
        patch(
            "t2pw.mapping.map_ids.hydrate_species_references",
            return_value={"hydrated": 0, "matched": 0, "novel": 0},
        ),
        patch("t2pw.mapping.map_ids._map_protein_with_strategy", return_value=protein_result),
        patch("t2pw.mapping.map_ids.map_protein_uniprot", return_value=protein_result),
        patch("t2pw.mapping.map_ids._map_compound_with_strategy", return_value=_unmapped_result()),
    ):
        return map_payload(
            payload,
            cache_path=tmp_path / "mapping-cache.json",
            id_source="api",
            use_cache=False,
            allow_complex_wrapper_creation=allow_complex_wrapper_creation,
        )


def test_unknown_fallback_activates_only_after_mapping_failure(tmp_path: Path) -> None:
    result = _map(_payload(), tmp_path, protein_result=_unmapped_result())
    mapped = result["payload"]

    assert [row["name"] for row in mapped["entities"]["proteins"]] == ["Unknown"]
    unknown = mapped["entities"]["proteins"][0]
    assert unknown["pathbank_protein_id"] == 9659
    assert unknown["pw_protein_id"] == 9659
    assert unknown["mapped_ids"]["uniprot"] == "Unknown"
    assert unknown["species"] == "Arabidopsis thaliana"
    assert unknown["species_id"] == 4
    assert unknown["taxonomy_id"] == "3702"
    assert unknown["mapping_meta"]["cross_species_placeholder"] is True
    assert unknown["mapping_meta"]["placeholder_target_organisms"] == ["Camellia sinensis"]

    complex_row = mapped["entities"]["protein_complexes"][0]
    assert complex_row["name"] == "N-methyl nucleosidase"
    assert complex_row["components"] == [
        {
            "name": "Unknown",
            "stoichiometry": 1,
            "pathbank_protein_id": 9659,
            "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        }
    ]
    assert complex_row["mapping_meta"]["chosen_rule"] == "pathbank_unknown_protein_fallback"
    assert complex_row["mapping_meta"]["target_organism"] == "Camellia sinensis"
    assert mapped["processes"]["reactions"][0]["enzymes"][0]["entity_type"] == "protein_complex"
    assert mapped["element_locations"]["protein_locations"] == []
    assert result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 1


def test_stage2_wrapper_disabled_does_not_create_unknown_fallback(tmp_path: Path) -> None:
    result = _map(
        _payload(),
        tmp_path,
        protein_result=_unmapped_result(),
        allow_complex_wrapper_creation=False,
    )

    assert [row["name"] for row in result["payload"]["entities"]["proteins"]] == [
        "N-methyl nucleosidase"
    ]
    assert result["payload"]["entities"]["protein_complexes"] == []
    assert result["payload"]["processes"]["reactions"][0]["enzymes"][0]["entity_type"] == "protein"
    assert "reaction_enzyme_unknown_fallbacks" not in result["report"]["summary"]


def test_unknown_fallback_does_not_override_real_mapping(tmp_path: Path) -> None:
    result = _map(_payload(protein_id="Q9TEST"), tmp_path, protein_result=_mapped_result())

    assert all(row.get("name") != "Unknown" for row in result["payload"]["entities"]["proteins"])
    assert result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 0
    complex_rows = result["payload"]["entities"]["protein_complexes"]
    assert complex_rows[0]["components"][0]["mapped_ids"]["uniprot"] == "Q9TEST"


def test_unknown_fallback_excludes_non_enzyme_proteins(tmp_path: Path) -> None:
    payload = _payload(include_enzyme=False)
    payload["processes"]["interactions"] = [
        {
            "name": "protein observation",
            "entity_1": "N-methyl nucleosidase",
            "entity_2": "caffeine",
            "relationship": "binds",
        }
    ]
    result = _map(payload, tmp_path, protein_result=_unmapped_result())

    assert [row["name"] for row in result["payload"]["entities"]["proteins"]] == [
        "N-methyl nucleosidase"
    ]
    assert result["payload"]["entities"]["protein_complexes"] == []
    assert result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 0


def test_unknown_fallback_wraps_transporter_only_protein(tmp_path: Path) -> None:
    result = _map(_transporter_payload(), tmp_path, protein_result=_unmapped_result())
    mapped = result["payload"]

    assert [row["name"] for row in mapped["entities"]["proteins"]] == ["Unknown"]
    unknown = mapped["entities"]["proteins"][0]
    assert unknown["pathbank_protein_id"] == 9659
    assert unknown["mapped_ids"]["uniprot"] == "Unknown"

    complex_row = mapped["entities"]["protein_complexes"][0]
    assert complex_row["name"] == "ABCG-116"
    assert complex_row["components"] == [
        {
            "name": "Unknown",
            "stoichiometry": 1,
            "pathbank_protein_id": 9659,
            "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        }
    ]
    assert complex_row["mapping_meta"]["chosen_rule"] == "pathbank_unknown_protein_fallback"

    transporter = mapped["processes"]["transports"][0]["transporters"][0]
    assert transporter["entity"] == "ABCG-116"
    assert transporter["entity_type"] == "protein_complex"

    assert result["report"]["summary"]["transporter_unknown_fallbacks"] == 1
    assert result["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 0


def test_unknown_fallback_excludes_transporter_referenced_elsewhere(tmp_path: Path) -> None:
    payload = _transporter_payload(add_interaction_ref=True)
    result = _map(payload, tmp_path, protein_result=_unmapped_result())

    assert [row["name"] for row in result["payload"]["entities"]["proteins"]] == [
        "ABCG-116"
    ]
    assert result["payload"]["entities"]["protein_complexes"] == []
    assert result["report"]["summary"]["transporter_unknown_fallbacks"] == 0
    transporter = result["payload"]["processes"]["transports"][0]["transporters"][0]
    assert transporter.get("entity_type") != "protein_complex"


def test_catalyst_modifier_is_promoted_before_unknown_fallback(tmp_path: Path) -> None:
    payload = _payload(include_enzyme=False)
    payload["processes"]["reactions"][0]["modifiers"] = [
        {
            "entity": "N-methyl nucleosidase",
            "entity_type": "protein",
            "role": "catalyst",
        }
    ]
    normalized, _ = normalize_process_payload(payload)
    reaction = normalized["processes"]["reactions"][0]
    assert reaction["modifiers"][0]["role"] == "catalyst"
    assert reaction["enzymes"][0]["entity"] == "N-methyl nucleosidase"

    result = _map(normalized, tmp_path, protein_result=_unmapped_result())
    mapped_reaction = result["payload"]["processes"]["reactions"][0]
    assert mapped_reaction["enzymes"][0]["entity"] == "N-methyl nucleosidase"
    assert mapped_reaction["enzymes"][0]["entity_type"] == "protein_complex"
    assert mapped_reaction["enzymes"][0]["role"] == "catalyst"
    assert mapped_reaction["modifiers"][0]["entity"] == "N-methyl nucleosidase"
    assert mapped_reaction["modifiers"][0]["entity_type"] == "protein_complex"
    assert result["payload"]["entities"]["protein_complexes"][0]["components"][0]["name"] == "Unknown"


def test_unknown_fallback_reuses_one_sentinel_and_functional_complex(tmp_path: Path) -> None:
    first = _map(_payload(reaction_count=2), tmp_path, protein_result=_unmapped_result())
    mapped = first["payload"]

    assert [row["name"] for row in mapped["entities"]["proteins"]].count("Unknown") == 1
    assert [row["name"] for row in mapped["entities"]["protein_complexes"]].count(
        "N-methyl nucleosidase"
    ) == 1
    assert {reaction["enzymes"][0]["entity"] for reaction in mapped["processes"]["reactions"]} == {
        "N-methyl nucleosidase"
    }

    second = _map(deepcopy(mapped), tmp_path, protein_result=_unmapped_result())
    assert [row["name"] for row in second["payload"]["entities"]["proteins"]].count("Unknown") == 1
    assert [row["name"] for row in second["payload"]["entities"]["protein_complexes"]].count(
        "N-methyl nucleosidase"
    ) == 1
    assert second["report"]["summary"]["reaction_enzyme_unknown_fallbacks"] == 0


def test_unknown_sentinel_is_not_sent_to_uniprot_enrichment(tmp_path: Path) -> None:
    mapped = _map(_payload(), tmp_path, protein_result=_unmapped_result())["payload"]

    with patch("t2pw.mapping.enrich_entities._fetch_uniprot_enrichment") as fetch:
        result = enrich_payload(mapped, cache_path=tmp_path / "enrichment-cache.json")

    fetch.assert_not_called()
    assert result["report"]["summary"]["proteins_skipped_pathbank_unknown"] == 1
    assert result["report"]["calls"]["uniprot"] == 0


def test_unknown_fallback_passes_stage3_and_serializes_exact_pathbank_sentinel(tmp_path: Path) -> None:
    mapped = _map(_payload(), tmp_path, protein_result=_unmapped_result())["payload"]
    validate_post_remap(mapped)

    normalized, normalization_report = normalize_process_payload(mapped)
    stage3_report = validate_post_normalization(normalized, normalization_report["gate"])
    assert stage3_report["ok"] is True
    assert validate_required_pwml_contract(normalized, strict_db=False)["ok"] is True

    ir, ir_report = build_pwml_ir(normalized, strict_db=False)
    assert ir_report["errors"] == []
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(
        extraction=ir,
        signature=signature,
        args=SimpleNamespace(
            name="Caffeine degradation",
            description="",
            subject="Metabolic",
            pw_id="PW000000",
            height=1400,
            width=3200,
            background_color="#FFFFFF",
            ref=str(ROOT / "reference" / "PW000001.pwml"),
        ),
    )
    root = builder.build().root

    protein = root.find(".//proteins/protein")
    assert protein is not None
    assert protein.findtext("id") == "9659"
    assert protein.findtext("name") == "Unknown"
    assert protein.findtext("species-id") == "4"
    assert protein.findtext("uniprot-id") == "Unknown"
    member = root.find(".//protein-complex-protein")
    assert member is not None, [element.tag for element in root.iter() if "protein" in str(element.tag)]
    assert member.findtext("protein-id") == "9659"


def test_orphan_named_complex_referenced_as_enzyme_gets_wrapped(tmp_path: Path) -> None:
    payload = _payload(include_enzyme=False)
    payload["entities"]["protein_complexes"] = [
        {"name": "oxoglutarate dehydrogenase complex", "components": []}
    ]
    payload["processes"]["reactions"][0]["enzymes"] = [
        {
            "entity": "oxoglutarate dehydrogenase complex",
            "entity_type": "protein_complex",
            "role": "catalyst",
        }
    ]
    result = _map(payload, tmp_path, protein_result=_unmapped_result())

    complex_row = result["payload"]["entities"]["protein_complexes"][0]
    assert complex_row["components"] == [
        {
            "name": "Unknown",
            "stoichiometry": 1,
            "pathbank_protein_id": 9659,
            "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        }
    ]
    assert complex_row["mapping_meta"]["chosen_rule"] == "pathbank_unknown_protein_fallback"
    assert result["report"]["summary"]["complex_missing_components_unknown_fallbacks"] == 0


def test_orphan_named_complex_not_referenced_anywhere_still_gets_wrapped(tmp_path: Path) -> None:
    payload = _payload(include_enzyme=False)
    payload["entities"]["protein_complexes"] = [
        {"name": "oxoglutarate dehydrogenase complex", "components": []}
    ]
    result = _map(payload, tmp_path, protein_result=_unmapped_result())

    complex_row = result["payload"]["entities"]["protein_complexes"][0]
    assert complex_row["components"] == [
        {
            "name": "Unknown",
            "stoichiometry": 1,
            "pathbank_protein_id": 9659,
            "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
        }
    ]
    assert complex_row["mapping_meta"]["chosen_rule"] == "pathbank_unknown_protein_fallback"
    assert complex_row["mapping_meta"]["fallback_reason"] == "complex_has_no_resolvable_components"
    assert result["report"]["summary"]["complex_missing_components_unknown_fallbacks"] == 1


def test_stage2_wrapper_disabled_does_not_wrap_orphan_complex(tmp_path: Path) -> None:
    payload = _payload(include_enzyme=False)
    payload["entities"]["protein_complexes"] = [
        {"name": "oxoglutarate dehydrogenase complex", "components": []}
    ]
    result = _map(
        payload,
        tmp_path,
        protein_result=_unmapped_result(),
        allow_complex_wrapper_creation=False,
    )

    assert result["payload"]["entities"]["protein_complexes"][0]["components"] == []
    assert "complex_missing_components_unknown_fallbacks" not in result["report"]["summary"]


def test_complex_with_real_pathbank_id_is_not_wrapped_despite_empty_components(tmp_path: Path) -> None:
    payload = _payload(include_enzyme=False)
    payload["entities"]["protein_complexes"] = [
        {
            "name": "oxoglutarate dehydrogenase complex",
            "components": [],
            "pathbank_complex_id": 555,
        }
    ]
    result = _map(payload, tmp_path, protein_result=_unmapped_result())

    complex_row = result["payload"]["entities"]["protein_complexes"][0]
    assert complex_row["components"] == []
    assert result["report"]["summary"]["complex_missing_components_unknown_fallbacks"] == 0


def test_real_pathbank_complex_with_no_listed_components_exports_with_empty_members(tmp_path: Path) -> None:
    """A complex with a confirmed PathBank identity may have zero listed subunits.

    reference/PW1.pwml's "alanine aminotransferase (ALT)" complex (pwp-id
    PW_P000036) is a real PathBank export with exactly this shape: a genuine
    complex-level identity and a self-closing, empty
    <protein_complex-proteins/>. This does not go through map_payload's
    Unknown-sentinel fallback at all -- it proves the fallback isn't the only
    thing keeping this shape exportable; the writer and validate_pwml_ir must
    also accept it directly.
    """
    payload = _payload(include_enzyme=False)
    payload["entities"]["proteins"] = []
    payload["entities"]["protein_complexes"] = [
        {
            "name": "alanine aminotransferase (ALT)",
            "species": "Camellia sinensis",
            "organism": "Camellia sinensis",
            "pathbank_complex_id": 36,
            "components": [],
        }
    ]
    payload["processes"]["reactions"][0]["enzymes"] = [
        {
            "entity": "alanine aminotransferase (ALT)",
            "entity_type": "protein_complex",
            "role": "catalyst",
        }
    ]
    payload["element_locations"]["protein_locations"] = []

    normalized, normalization_report = normalize_process_payload(payload)
    stage3_report = validate_post_normalization(normalized, normalization_report["gate"])
    assert stage3_report["ok"] is True
    assert validate_required_pwml_contract(normalized, strict_db=False)["ok"] is True

    ir, ir_report = build_pwml_ir(normalized, strict_db=False)
    assert ir_report["errors"] == []

    validation = validate_pwml_ir(ir)
    assert validation["ok"] is True
    assert any(w["code"] == "protein_complex_missing_components" for w in validation["warnings"])

    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(
        extraction=ir,
        signature=signature,
        args=SimpleNamespace(
            name="Caffeine degradation",
            description="",
            subject="Metabolic",
            pw_id="PW000000",
            height=1400,
            width=3200,
            background_color="#FFFFFF",
            ref=str(ROOT / "reference" / "PW000001.pwml"),
        ),
    )
    root = builder.build().root

    complex_el = root.find(".//protein-complexes/protein-complex")
    assert complex_el is not None
    assert complex_el.findtext("name") == "alanine aminotransferase (ALT)"
    members_el = complex_el.find("protein_complex-proteins")
    assert members_el is not None
    assert list(members_el) == []
