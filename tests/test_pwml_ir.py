from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
from types import SimpleNamespace

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# C-051 / D-015 (LOCKED): compound resolution finishes BEFORE the canonical
# freeze, so ``build_pwml_ir`` now refuses a compound row that carries no
# resolution verdict instead of resolving it late. Every fixture below is a raw
# extraction payload, which production no longer hands the exporter, so each
# call site is taken through the stage that now does the work. The helper
# asserts the stage actually ruled on every row -- see helpers_prefreeze.
from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir, validate_required_pwml_contract  # noqa: E402
from t2pw.pwml.qa import run_pwml_qa  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature, repair_tree, validate_generated_tree  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402


def _base_payload() -> dict:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {"name": "Glucose", "pathbank_compound_id": 101, "mapped_ids": {"chebi": "CHEBI:17234"}},
                {"name": "Glucose 6-phosphate", "pathbank_compound_id": 102, "mapped_ids": {"chebi": "CHEBI:4170"}},
            ],
            "proteins": [
                {"name": "Hexokinase", "species": "Homo sapiens", "pathbank_protein_id": 201, "mapped_ids": {"uniprot": "P19367"}},
            ],
            "protein_complexes": [{"name": "Hexokinase complex", "species": "Homo sapiens", "components": ["Hexokinase"]}],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"},
        ],
        "processes": {
            "reactions": [
                {
                    "name": "Glucose phosphorylation",
                    "inputs": ["Glucose"],
                    "outputs": ["Glucose 6-phosphate"],
                    "biological_state": "cytosol",
                    "enzymes": [{"entity": "Hexokinase complex", "entity_type": "protein_complex"}],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _obag_plp_payload() -> dict:
    return {
        "entities": {
            "species": [{"name": "Pseudomonas fluorescens"}],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "L-Thr"},
                {"name": "glycine enolate"},
                {"name": "acetaldehyde"},
                {"name": "pyridoxal-phosphate"},
            ],
            "proteins": [{"name": "ObaG"}],
            "protein_complexes": [],
        },
        "biological_states": [
            {
                "name": "Pseudomonas fluorescens cytosol",
                "species": "Pseudomonas fluorescens",
                "subcellular_location": "cytosol",
            }
        ],
        "processes": {
            "reactions": [
                {
                    "name": "L-Thr cleavage",
                    "inputs": ["L-Thr"],
                    "outputs": ["glycine enolate", "acetaldehyde"],
                    "biological_state": "Pseudomonas fluorescens cytosol",
                    "modifiers": [
                        {"entity": "ObaG", "entity_type": "protein", "role": "catalyst"},
                        {"entity": "pyridoxal-phosphate", "entity_type": "compound", "role": "catalyst"},
                    ],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def test_reaction_ir_construction_refs_resolve() -> None:
    ir, report = build_pwml_ir(prefrozen_when_compounded(_base_payload()), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not report["errors"]
    assert validation["ok"], validation["errors"]
    reaction = ir["processes"]["reactions"][0]
    assert reaction["left"][0]["entity_key"] == ir["entities"]["compounds"][0]["key"]
    assert reaction["right"][0]["entity_key"] == ir["entities"]["compounds"][1]["key"]
    assert reaction["enzymes"][0]["entity_type"] == "protein_complex"
    assert reaction["enzymes"][0]["entity_key"] == ir["entities"]["protein_complexes"][0]["key"]
    assert ir["entities"]["protein_complexes"][0]["components"] == [
        {"protein_key": ir["entities"]["proteins"][0]["key"]}
    ]
    complex_viz = ir["protein_complex_visualizations"][0]
    assert complex_viz["entity_key"] == reaction["enzymes"][0]["entity_key"]
    assert complex_viz["biological_state_key"] == reaction["biological_state_key"]
    assert complex_viz["hidden"] is False
    assert {"x", "y", "zindex", "visualization_template_id"} <= set(complex_viz)

    viz = ir["process_visualizations"][0]
    assert viz["type"] == "reaction_visualization"
    assert {m["process_member_key"] for m in viz["members"]} >= {
        reaction["left"][0]["key"],
        reaction["right"][0]["key"],
        reaction["enzymes"][0]["key"],
    }


def test_validate_pwml_ir_errors_on_protein_complex_missing_components() -> None:
    payload = _base_payload()
    payload["entities"]["protein_complexes"].append(
        {"name": "oxoglutarate dehydrogenase complex", "species": "Homo sapiens", "components": []}
    )

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    # build_pwml_ir mirrors validate_pwml_ir's own errors into its report, so
    # this surfaces here too -- catching the gap even earlier than export time.
    assert "protein_complex_missing_components" in {
        err["code"] for err in report["errors"]
    }

    validation = validate_pwml_ir(ir)

    assert not validation["ok"]
    assert "protein_complex_missing_components" in {
        err["code"] for err in validation["errors"]
    }


def test_required_contract_validates_protein_complex_enzyme_in_direct_ir() -> None:
    ir, report = build_pwml_ir(prefrozen_when_compounded(_base_payload()), strict_db=True)
    assert not report["errors"]

    valid_contract = validate_required_pwml_contract(ir, strict_db=True)
    assert "reaction_enzyme_must_be_protein_complex" not in {
        error["code"] for error in valid_contract["errors"]
    }

    bare_ir = deepcopy(ir)
    protein = bare_ir["entities"]["proteins"][0]
    bare_ir["processes"]["reactions"][0]["enzymes"][0].update(
        {"entity_key": protein["key"], "entity_type": "protein"}
    )

    invalid_contract = validate_required_pwml_contract(bare_ir, strict_db=True)
    wrapper_errors = [
        error
        for error in invalid_contract["errors"]
        if error["code"] == "reaction_enzyme_must_be_protein_complex"
    ]
    assert len(wrapper_errors) == 1
    assert wrapper_errors[0]["pointer"] == "/processes/reactions/0/enzymes/0"


def test_required_contract_accepts_canonical_complex_modifier_mirror() -> None:
    payload = _base_payload()
    payload["processes"]["reactions"][0]["modifiers"] = [
        {
            "entity": "Hexokinase complex",
            "entity_type": "protein_complex",
            "role": "catalyst",
        }
    ]

    contract = validate_required_pwml_contract(payload, strict_db=True)

    assert not {
        "reaction_enzyme_must_be_protein_complex",
        "duplicate_reaction_enzyme_complex",
    } & {error["code"] for error in contract["errors"]}

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    assert not report["errors"]
    assert len(ir["processes"]["reactions"][0]["enzymes"]) == 1


def test_compound_catalyst_modifier_is_not_exported_as_reaction_enzyme() -> None:
    payload = _base_payload()
    payload["entities"]["compounds"].append({"name": "pyridoxal-phosphate", "pathbank_compound_id": 103})
    payload["entities"]["protein_complexes"] = []
    payload["processes"]["reactions"][0]["enzymes"] = []
    payload["processes"]["reactions"][0]["modifiers"] = [
        {"entity": "pyridoxal-phosphate", "entity_type": "compound", "role": "catalyst"}
    ]

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert validation["ok"], validation["errors"]
    assert ir["processes"]["reactions"][0]["enzymes"] == []
    assert ir["entities"]["protein_complexes"] == []
    assert "non_protein_catalyst_dropped" in {warning["code"] for warning in report["warnings"]}


def test_bare_protein_catalyst_fails_instead_of_being_wrapped_during_export() -> None:
    payload = _base_payload()
    payload["entities"]["protein_complexes"] = []
    payload["processes"]["reactions"][0]["enzymes"] = []
    payload["processes"]["reactions"][0]["modifiers"] = [
        {"entity": "Hexokinase", "entity_type": "protein", "role": "catalyst"}
    ]

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not validation["ok"]
    reaction = ir["processes"]["reactions"][0]
    assert reaction["enzymes"][0]["entity_type"] == "protein"
    assert ir["entities"]["protein_complexes"] == []
    assert "reaction_enzyme_must_be_protein_complex" in {error["code"] for error in report["errors"]}


def test_obag_export_drops_plp_catalyst_but_rejects_bare_protein_enzyme() -> None:
    ir, report = build_pwml_ir(prefrozen_when_compounded(_obag_plp_payload()), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not validation["ok"]
    assert "reaction_enzyme_must_be_protein_complex" in {
        error.get("code") for error in validation["errors"]
    }
    reaction = ir["processes"]["reactions"][0]
    assert len(reaction["enzymes"]) == 1
    assert reaction["enzymes"][0]["entity_type"] == "protein"
    assert {warning["code"] for warning in report["warnings"]} >= {
        "non_protein_catalyst_dropped",
    }
    assert any(
        warning.get("code") == "non_protein_catalyst_dropped"
        and warning.get("name") == "pyridoxal-phosphate"
        for warning in report["warnings"]
    )


def test_legacy_reaction_enzyme_rows_continue_exporting() -> None:
    cases = [
        ({"protein_complex": "Hexokinase complex"}, [{"name": "Hexokinase complex", "species": "Homo sapiens", "components": ["Hexokinase"]}]),
    ]

    for enzyme_row, protein_complexes in cases:
        payload = _base_payload()
        payload["entities"]["protein_complexes"] = protein_complexes
        payload["processes"]["reactions"][0]["enzymes"] = [enzyme_row]

        ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
        validation = validate_pwml_ir(ir)

        assert not report["errors"]
        assert validation["ok"], validation["errors"]
        assert ir["processes"]["reactions"][0]["enzymes"][0]["entity_type"] == "protein_complex"


def test_create_defaults_fill_unmatched_species_and_cell_location() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Narcissus sp. aff. pseudonarcissus"}],
            "subcellular_locations": [{"name": "cell"}],
            "compounds": [],
            "proteins": [],
        },
        "biological_states": [
            {
                "name": "__auto_state__",
                "species": "Narcissus sp. aff. pseudonarcissus",
                "subcellular_location": "cell",
            }
        ],
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not report["errors"]
    assert validation["ok"], validation["errors"]
    assert ir["species"] == [
        {
            "key": "sp_1",
            "name": "Narcissus aff. pseudonarcissus MK-2014",
            "aliases": [
                "Narcissus sp. aff. pseudonarcissus",
                "Narcissus aff. pseudonarcissus MK-2014",
            ],
            "taxonomy_id": "1540222",
            "classification": "Eukaryote",
            "common_name": "Daffodil",
        }
    ]
    assert ir["subcellular_locations"] == [
        {"key": "scl_1", "name": "cell", "aliases": ["cell"], "ontology_id": "GO:0005623"}
    ]
    assert ir["biological_states"][0]["species_key"] == "sp_1"
    assert ir["biological_states"][0]["subcellular_location_key"] == "scl_1"


def test_biological_state_locations_are_hydrated_into_component_registry() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Pseudomonas fluorescens"}],
            "subcellular_locations": [{"name": "cell"}],
            "compounds": [{"name": "obafluorin"}],
            "proteins": [],
        },
        "biological_states": [
            {
                "name": "AutoState_pseudomonas_fluorescenscytosol",
                "species": "Pseudomonas fluorescens",
                "subcellular_location": "Cytosol",
            },
            {
                "name": "AutoState_pseudomonas_fluorescensperiplasmic_space",
                "species": "Pseudomonas fluorescens",
                "subcellular_location": "Periplasmic Space",
            },
            {
                "name": "AutoState_pseudomonas_fluorescensmitochondrial_membrane",
                "species": "Pseudomonas fluorescens",
                "subcellular_location": "Mitochondrial membrane",
            },
        ],
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=False)
    validation = validate_pwml_ir(ir)

    assert not any(error["code"].startswith("biological_state_") for error in report["errors"])
    assert validation["ok"], validation["errors"]
    assert {row["name"] for row in ir["subcellular_locations"]} >= {
        "Cytosol",
        "Periplasmic Space",
        "Mitochondrial membrane",
    }
    assert all(state["subcellular_location_key"] for state in ir["biological_states"])


def test_transport_ir_construction_has_state_and_visual_refs() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [
                {"name": "extracellular", "pathwhiz_id": 2},
                {"name": "cytosol", "pathwhiz_id": 3},
            ],
            "compounds": [{"name": "Glucose", "pathbank_compound_id": 101}],
            "proteins": [{"name": "GLUT1", "pathbank_protein_id": 301}],
        },
        "biological_states": [
            {"name": "extracellular", "species": "Homo sapiens", "subcellular_location": "extracellular"},
            {"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"},
        ],
        "processes": {
            "reactions": [],
            "transports": [
                {
                    "name": "Glucose import",
                    "cargo": "Glucose",
                    "from_biological_state": "extracellular",
                    "to_biological_state": "cytosol",
                    "transporters": [{"protein": "GLUT1", "biological_state": "cytosol"}],
                }
            ],
            "interactions": [],
        },
    }

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not report["errors"]
    assert validation["ok"], validation["errors"]
    transport = ir["processes"]["transports"][0]
    element = transport["transport_elements"][0]
    assert element["left_biological_state_key"] != element["right_biological_state_key"]

    viz = [v for v in ir["process_visualizations"] if v["type"] == "transport_visualization"][0]
    cargo_members = [m for m in viz["members"] if m["process_member_key"] == element["key"]]
    assert {m["role"] for m in cargo_members} == {"left", "right"}
    assert all(m["location_key"] and m["edge_key"] for m in cargo_members)


def test_name_only_compound_is_exportable_without_db_identity_class_or_type() -> None:
    payload = _base_payload()
    payload["entities"]["compounds"][0].pop("pathbank_compound_id")
    payload["entities"]["compounds"][0].pop("mapped_ids")

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not any(err["code"] == "missing_db_identity" and err.get("entity_type") == "compound" for err in report["errors"])
    assert validation["ok"], validation["errors"]


def test_required_contract_protein_needs_species_and_uniprot_or_drugbank() -> None:
    payload = _base_payload()
    payload["entities"]["proteins"][0].pop("pathbank_protein_id")
    payload["entities"]["proteins"][0].pop("mapped_ids")
    payload["entities"]["proteins"][0]["species"] = "Homo sapiens"

    report = validate_required_pwml_contract(payload, strict_db=True)

    codes = {err["code"] for err in report["errors"]}
    assert "protein_missing_db_identity" not in codes
    assert "protein_species_missing_db_identity" not in codes
    assert "protein_missing_external_identity" in codes


def test_required_contract_accepts_protein_with_species_and_drugbank() -> None:
    payload = _base_payload()
    payload["entities"]["proteins"][0].pop("pathbank_protein_id")
    payload["entities"]["proteins"][0].pop("mapped_ids")
    payload["entities"]["proteins"][0]["species"] = "Homo sapiens"
    payload["entities"]["proteins"][0]["drugbank_id"] = "DB00000"

    report = validate_required_pwml_contract(payload, strict_db=True)

    codes = {err["code"] for err in report["errors"]}
    assert "protein_missing_db_identity" not in codes
    assert "protein_species_missing_db_identity" not in codes
    assert "protein_missing_external_identity" not in codes


def test_name_only_biological_state_is_not_exportable() -> None:
    payload = _base_payload()
    payload["biological_states"] = [{"name": "Generic state"}]
    payload["processes"]["reactions"][0]["biological_state"] = "Generic state"

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    error_codes = {err["code"] for err in report["errors"]}
    assert "biological_state_missing_species" in error_codes
    assert "biological_state_missing_subcellular_location" in error_codes
    assert not validation["ok"]


def test_required_contract_validator_does_not_mutate_payload() -> None:
    payload = _base_payload()
    before = deepcopy(payload)

    report = validate_required_pwml_contract(payload, strict_db=True)

    assert payload == before
    assert report["summary"]["checked_as"] == "payload"


def test_required_contract_validator_indexes_raw_payload_entity_reference_tables() -> None:
    payload = _base_payload()

    report = validate_required_pwml_contract(payload, strict_db=True)

    codes = {err["code"] for err in report["errors"]}
    assert "biological_state_species_missing_db_identity" not in codes
    assert "biological_state_subcellular_location_missing_db_identity" not in codes


def test_required_contract_allows_named_biological_state_context_without_db_ids() -> None:
    payload = _base_payload()
    payload["entities"]["species"] = [{"name": "Homo sapiens"}]
    payload["entities"]["subcellular_locations"] = [{"name": "cytosol"}]
    payload["entities"]["proteins"][0]["species"] = "Homo sapiens"
    payload["entities"]["proteins"][0]["pathbank_species_id"] = 1

    report = validate_required_pwml_contract(payload, strict_db=True)

    codes = {err["code"] for err in report["errors"]}
    assert "biological_state_species_missing_db_identity" not in codes
    assert "biological_state_subcellular_location_missing_db_identity" not in codes


def test_novel_protein_complex_with_resolved_components_does_not_need_db_identity() -> None:
    payload = _base_payload()
    payload["entities"]["protein_complexes"] = [
        {
            "name": "Hexokinase complex",
            "species": "Homo sapiens",
            "components": ["Hexokinase"],
        }
    ]
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "Hexokinase complex", "entity_type": "protein_complex"}
    ]

    contract = validate_required_pwml_contract(payload, strict_db=True)
    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert "protein_complex_missing_db_identity" not in {err["code"] for err in contract["errors"]}
    assert not any(
        err["code"] == "missing_db_identity" and err.get("entity_type") == "protein_complex"
        for err in report["errors"]
    )
    assert validation["ok"], validation["errors"]


def test_generated_single_protein_complex_requires_valid_component_not_complex_identity() -> None:
    payload = _base_payload()
    payload["entities"]["proteins"][0]["species"] = "Homo sapiens"
    payload["entities"]["protein_complexes"] = [
        {
            "name": "Hexokinase complex",
            "species": "Homo sapiens",
            "generated": True,
            "generation_reason": "single_protein_pathwhiz_wrapper",
            "components": [
                {"name": "Hexokinase", "stoichiometry": 1, "mapped_ids": {"uniprot": "P19367"}},
            ],
        }
    ]
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "Hexokinase complex", "entity_type": "protein_complex"}
    ]

    contract = validate_required_pwml_contract(payload, strict_db=True)
    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    codes = {err["code"] for err in contract["errors"]}
    assert "protein_complex_missing_db_identity" not in codes
    assert "generated_complex_component_missing_external_identity" not in codes
    assert not report["errors"]
    assert validation["ok"], validation["errors"]


def test_generated_single_protein_complex_errors_when_component_protein_lacks_external_identity() -> None:
    payload = _base_payload()
    payload["entities"]["proteins"][0]["species"] = "Homo sapiens"
    payload["entities"]["proteins"][0].pop("mapped_ids")
    payload["entities"]["protein_complexes"] = [
        {
            "name": "Hexokinase complex",
            "species": "Homo sapiens",
            "generated": True,
            "generation_reason": "single_protein_pathwhiz_wrapper",
            "components": [{"name": "Hexokinase", "stoichiometry": 1}],
        }
    ]
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "Hexokinase complex", "entity_type": "protein_complex"}
    ]

    contract = validate_required_pwml_contract(payload, strict_db=True)

    codes = {err["code"] for err in contract["errors"]}
    assert "generated_complex_component_missing_external_identity" in codes


def test_protein_complex_components_hydrate_and_export_as_protein_refs() -> None:
    payload = _base_payload()
    payload["entities"]["protein_complexes"] = [
        {
            "name": "Hexokinase complex",
            "pathbank_complex_id": 401,
            "species": "Homo sapiens",
            "components": ["Hexokinase"],
        }
    ]

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not report["errors"]
    assert validation["ok"], validation["errors"]
    complex_record = ir["entities"]["protein_complexes"][0]
    # A bare-string component states no count, so stoichiometry stays absent
    # rather than being assumed; PathWhiz accepts a nil coefficient.
    assert complex_record["components"] == [{"protein_key": "prot_1"}]

    ref_path = ROOT / "reference" / "PW000001.pwml"
    signature = discover_structure_signature(ref_path)
    args = SimpleNamespace(
        name="Generated Pathway",
        description="",
        subject="Metabolic",
        pw_id="PW000000",
        height=1400,
        width=3200,
        background_color="#FFFFFF",
        ref=str(ref_path),
    )
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args)
    builder.build()

    protein_id = builder.section_items["proteins"][0]["id"]
    complex_members = builder.section_items["protein-complexes"][0]["protein_complex-proteins"]
    assert complex_members == [
        {"id": complex_members[0]["id"], "protein-id": protein_id, "stoichiometry": None}
    ]


def test_protein_complex_component_links_by_matching_uniprot_or_pathbank_id() -> None:
    payload = _base_payload()
    payload["entities"]["proteins"].append(
        {
            "name": "acyl-CoA oxidase 1 (ACX1)",
            "species": "Arabidopsis thaliana",
            "pathbank_protein_id": 11618,
            "mapped_ids": {"uniprot": "O65202", "pathbank_protein_id": "11618"},
        }
    )
    payload["entities"]["protein_complexes"] = [
        {
            "name": "acyl coenzyme A oxidase",
            "species": "Arabidopsis thaliana",
            "components": [
                {
                    "name": "acyl coenzyme A oxidase",
                    "stoichiometry": 1,
                    "mapped_ids": {"uniprot": "O65202", "pathbank_protein_id": "11618"},
                    "pathbank_protein_id": 11618,
                    "gene_name": "ACX1",
                }
            ],
        }
    ]

    contract = validate_required_pwml_contract(payload, strict_db=True)
    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not any(err["code"] == "component_protein_unresolved" for err in contract["errors"])
    assert not any(err["code"] == "component_protein_unresolved" for err in report["errors"])
    assert ir["entities"]["protein_complexes"][0]["components"] == [
        {"protein_key": "prot_2", "stoichiometry": 1}
    ]
    assert validation["ok"], validation["errors"]


def test_protein_complex_unresolved_component_is_exportable_with_warnings() -> None:
    payload = _base_payload()
    payload["entities"]["protein_complexes"] = [
        {
            "name": "Broken complex",
            "pathbank_complex_id": 401,
            "species": "Homo sapiens",
            "components": ["Missing protein"],
        }
    ]

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert any(w["code"] == "component_protein_unresolved" for w in report["warnings"])
    assert not any(err["code"] == "component_protein_unresolved" for err in report["errors"])
    # A complex with a real PathBank identity (a non-generated row) is allowed
    # to end up with zero components at export time -- real PathBank exports
    # (e.g. reference/PW1.pwml's "alanine aminotransferase (ALT)" complex)
    # carry a genuine pwp-id with an empty <protein_complex-proteins/>, so
    # this is a warning, not an error. Only a pipeline-generated wrapper
    # (see test_validate_pwml_ir_errors_on_protein_complex_missing_components)
    # must always have at least one member.
    assert any(w["code"] == "protein_complex_missing_components" for w in validation["warnings"])
    assert not any(err["code"] == "protein_complex_missing_components" for err in validation["errors"])
    assert validation["ok"]
    assert ir["entities"]["protein_complexes"][0]["components"] == []


def test_dangling_process_visualization_references_fail_validation() -> None:
    ir, report = build_pwml_ir(prefrozen_when_compounded(_base_payload()), strict_db=True)
    assert not report["errors"]

    ir["process_visualizations"][0]["process_key"] = "missing_process"
    ir["process_visualizations"][0]["members"][0]["location_key"] = "missing_location"

    validation = validate_pwml_ir(ir)
    codes = {err["code"] for err in validation["errors"]}

    assert "visualization_unknown_process" in codes
    assert "visualization_unknown_location" in codes
    assert not validation["ok"]


def test_same_compound_two_states_is_one_entity_two_locations() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}, {"name": "mitochondria", "pathwhiz_id": 3}],
            "compounds": [{"name": "Pyruvate", "pathbank_compound_id": 150}],
            "proteins": [],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"},
            {"name": "mitochondria", "species": "Homo sapiens", "subcellular_location": "mitochondria"},
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": "Pyruvate", "biological_state": "cytosol"},
                {"compound": "Pyruvate", "biological_state": "mitochondria"},
            ],
            "protein_locations": [
                {"protein": "KT378599", "biological_state": "cytosol"},
            ],
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    validation = validate_pwml_ir(ir)
    pyruvate_key = ir["entities"]["compounds"][0]["key"]
    pyruvate_locations = [loc for loc in ir["locations"] if loc["entity_key"] == pyruvate_key]

    assert not report["errors"]
    assert any(w["code"] == "location_entity_not_found" for w in report["warnings"])
    assert validation["ok"], validation["errors"]
    assert len(ir["entities"]["compounds"]) == 1
    assert len(pyruvate_locations) == 2
    assert len({loc["biological_state_key"] for loc in pyruvate_locations}) == 2


def test_ir_writer_integration_validates_and_has_no_fatal_qa_errors() -> None:
    ir, report = build_pwml_ir(prefrozen_when_compounded(_base_payload()), strict_db=True)
    assert not report["errors"]

    ref_path = ROOT / "reference" / "PW000001.pwml"
    signature = discover_structure_signature(ref_path)
    args = SimpleNamespace(
        name="Generated Pathway",
        description="",
        subject="Metabolic",
        pw_id="PW000000",
        height=1400,
        width=3200,
        background_color="#FFFFFF",
        ref=str(ref_path),
    )
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args)
    build_result = builder.build()
    repaired = repair_tree(etree.ElementTree(build_result.root), signature)
    validation = validate_generated_tree(repaired, signature)
    xml_bytes = etree.tostring(repaired.getroot(), encoding="utf-8", xml_declaration=True, pretty_print=True)
    qa = run_pwml_qa(xml_bytes)

    complex_viz_id = builder.section_items["protein-complex-visualizations"][0]["id"]
    complex_member_viz = builder.section_items["protein-complex-visualizations"][0]["protein_complex_protein_visualizations"]
    enzyme_viz = builder.section_items["reaction-visualizations"][0]["reaction_enzyme_visualizations"][0]
    assert len(builder.section_items["reactions"][0]["reaction-enzymes"]) == 1
    assert complex_member_viz
    assert "protein-location-id" in complex_member_viz[0]
    assert enzyme_viz["protein-complex-visualization-id"] == complex_viz_id
    assert "protein-location-id" not in enzyme_viz
    assert validation["ok"], validation["issues"][:3]
    assert qa["ok"], qa["errors"]


def test_spontaneous_field_is_always_forced_false_on_export() -> None:
    payload = _base_payload()
    payload["processes"]["reactions"][0].update({"enzymes": [], "spontaneous": True})
    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    assert not report["errors"]
    assert ir["processes"]["reactions"][0]["spontaneous"] is False

    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000", height=1400, width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    root = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args).build().root
    assert root.findtext(".//reactions/reaction/spontaneous") == "false"
    assert run_pwml_qa(etree.tostring(root))["ok"] is True


def test_pre_export_and_qa_reject_duplicate_enzyme_complex_even_when_spontaneous_set() -> None:
    payload = _base_payload()
    enzyme = payload["processes"]["reactions"][0]["enzymes"][0]
    payload["processes"]["reactions"][0].update({"enzymes": [enzyme, dict(enzyme)], "spontaneous": True})
    contract = validate_required_pwml_contract(payload, strict_db=True)
    assert "duplicate_reaction_enzyme_complex" in {
        error["code"] for error in contract["errors"]
    }

    ir, _ = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000", height=1400, width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    root = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args).build().root
    qa = run_pwml_qa(etree.tostring(root))
    assert any("more than once as an enzyme" in error for error in qa["errors"])


def test_pre_export_rejects_duplicate_legacy_protein_complex_enzyme_rows() -> None:
    payload = _base_payload()
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"protein_complex": "Hexokinase complex"},
        {"protein_complex": "Hexokinase complex"},
    ]

    contract = validate_required_pwml_contract(payload, strict_db=True)

    assert "duplicate_reaction_enzyme_complex" in {
        error["code"] for error in contract["errors"]
    }


def test_writer_serializes_each_proteins_resolved_species() -> None:
    payload = _base_payload()
    payload["entities"]["species"].append({"name": "Mus musculus", "pathwhiz_id": 9})
    payload["entities"]["proteins"][0]["species"] = "Mus musculus"
    ir, report = build_pwml_ir(prefrozen_when_compounded(payload), strict_db=True)
    assert not report["errors"]
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000", height=1400, width=3200, background_color="#FFFFFF", ref=str(ROOT / "reference" / "PW000001.pwml"))
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args)
    builder.build()
    assert builder.section_items["proteins"][0]["species-id"] == 9


# ---------------------------------------------------------------------------
# C-050k -- F-048's alias-index ambiguity is recorded. Ruling: D-041 (a); scope,
# adjudication and constraints: D-043, which is LOCKED and is the authority --
# cited here, not transcribed.
#
# G9: **CORRECTION of pre-existing observable behaviour, NOT new capability.** The
# base records nothing at all on this path. Measured over the 32 committed legs:
# **61 live ``resolve_entity`` consultations through an ambiguous alias key across
# 8 legs**, 20 of them same-type, with ``ambiguous_entity_reference`` **empty on
# all 8** (``evidence/c050k_census_prechange.json``). Every arm below therefore
# fails at base ``15f36b4`` and passes at the tip, proved by
# ``probe_c050k_alias_ambiguity.py --mode g9`` run in a real git worktree at the
# base SHA, never an export (**F-042**).
#
# **All 20 adjudicated bindings were biologically CORRECT** (D-043 section 1).
# Nothing here claims a mis-binding and nothing rebinds: the violation is
# ``PRODUCT_CONTRACT`` section 3 traceability, so the obligation is to record the
# choice, not to have been lucky.
#
# The DB is pinned DOWN in every arm, which is load-bearing rather than tidy:
# ``PathBankDbResolver.from_env`` calls ``ensure_dotenv_loaded``, so an unpinned arm
# reads the developer's real ``.env`` and stops being deterministic.
# ``helpers_prefreeze.prefrozen``'s own docstring asks callers to do exactly this.
# ---------------------------------------------------------------------------


class _DownDb:
    """A DB that is reachable-but-down, so no arm below depends on the ambient
    environment. Same shape as the golden sweep's ``B`` configuration."""

    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


#: The regression fixture ``pwml-bio-auditor`` specified in **D-043**, transcribed
#: exactly: the real ``PMC12312563/strict`` collision, minimal and DB-free. The
#: frozen leg carries ``{"name": "1,4-dihydroxy-2-naphthoic acid", "synonyms":
#: ["DHNA"], "pathbank_compound_id": 40747}`` and a bare ``{"name": "DHNA"}``, and
#: reaction 6 (``DHNA-CoA thioesterase activity``, MenI) declares output ``"DHNA"``.
_AUDITOR_ROWS = [
    {"name": "1,4-dihydroxy-2-naphthoic acid", "synonyms": ["DHNA"],
     "pathbank_compound_id": 40747},
    {"name": "DHNA"},
]


def _alias_payload(compounds: list, outputs: list = None, extra_entities: dict = None) -> dict:
    """A minimal payload whose only interesting property is its alias index."""

    entities = {"compounds": compounds, "proteins": []}
    entities.update(extra_entities or {})
    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": entities,
        "processes": {"reactions": [{
            "name": "R1", "inputs": ["DHNA-CoA"],
            "outputs": ["DHNA"] if outputs is None else outputs}]},
    }


def _alias_build(payload: dict) -> tuple:
    down = _DownDb()
    staged = prefrozen_when_compounded(
        deepcopy(payload), db_resolver=down, strict_db=False, name_index=None)
    return build_pwml_ir(staged, pathway_name="P", pathway_subject="Metabolic",
                         db_resolver=down, strict_db=False, name_index=None)


def _row_issues(report: dict) -> list:
    return [issue for bucket in ("errors", "warnings")
            for issue in report[bucket]
            if issue.get("code") == "ambiguous_entity_row_reference"]


def test_c050k_alias_ambiguity_is_recorded_and_the_binding_does_not_move() -> None:
    """NEW ARM, G9 CORRECTION. D-043's regression fixture, both halves.

    (a) the binding is **still the first row** -- unchanged from base;
    (b) the report now carries an issue naming **both** candidate keys.
    """

    ir, report = _alias_build(_alias_payload(deepcopy(_AUDITOR_ROWS)))

    # (a) UNCHANGED. Payload row order stays authoritative (D-043 section 4): the
    # output "DHNA" binds the row that came FIRST -- the alias holder -- as at base.
    bound_key = ir["processes"]["reactions"][0]["right"][0]["entity_key"]
    bound = next(r for r in ir["entities"]["compounds"] if r["key"] == bound_key)
    assert bound_key == "cmp_1"
    assert bound["name"] == "1,4-dihydroxy-2-naphthoic acid"

    # (b) RECORDED. This is what fails at base, where the list is empty.
    issues = _row_issues(report)
    assert len(issues) == 1, issues
    issue = issues[0]
    assert issue["entity_keys"] == ["cmp_1", "cmp_2"], issue
    assert issue["bound_entity_key"] == "cmp_1"
    assert issue["name"] == "DHNA"
    assert issue["role"] == "reaction_member"
    assert issue["pointer"] == "/processes/reactions/0/outputs/0"
    # The ids are carried, NEVER compared: F-043 forbids inferring sameness from
    # identifier equality and D-043 section 4 forbids this diagnostic making that call.
    assert issue["entity_names"] == ["1,4-dihydroxy-2-naphthoic acid", "DHNA"]
    assert len(issue["entity_pathwhiz_ids"]) == 2


def test_c050k_alias_ambiguity_is_a_warning_and_never_moves_report_ok() -> None:
    """NEW ARM. **D-041 section 2 limit 1.** ``error`` would set
    ``report["ok"] = False`` and flip legs between exporting and not; the diagnostic
    must never do that."""

    _ir, report = _alias_build(_alias_payload(deepcopy(_AUDITOR_ROWS)))

    codes = {issue["code"] for issue in report["errors"]}
    assert "ambiguous_entity_row_reference" not in codes, report["errors"]
    assert any(issue["code"] == "ambiguous_entity_row_reference"
               for issue in report["warnings"])

    # And on a payload whose only new finding is the ambiguity, ``ok`` stays exactly
    # where the base left it: the diagnostic contributes nothing to the verdict.
    clean = _base_payload()
    clean["entities"]["compounds"][0]["synonyms"] = ["Glucose 6-phosphate"]
    down = _DownDb()
    _clean_ir, clean_report = build_pwml_ir(
        prefrozen_when_compounded(clean, db_resolver=down, strict_db=False,
                                  name_index=None),
        db_resolver=down, strict_db=False, name_index=None)
    assert _row_issues(clean_report), "arm is vacuous unless the ambiguity fired"
    assert clean_report["ok"] is True


def test_c050k_slot_duplicated_single_entity_raises_no_ambiguity() -> None:
    """NEW ARM, and the arm that makes the guard NON-VACUOUS.

    **D-041 section 5 / D-043 section 4.** ``entity_by_name`` is appended to once
    per alias slot with no dedupe, so ONE row whose ``name``, ``raw_name`` and
    ``synonyms`` all normalize alike occupies several slots under one key and
    ``len(candidates) > 1`` is true for a single entity. Measured on the committed
    corpus that shape holds for **209** ``_norm`` keys. A length-based test would
    raise 209 false alarms; the test is over distinct ``entity_key``s, so this
    must stay silent.
    """

    ir, report = _alias_build(_alias_payload([
        {"name": "DHNA", "raw_name": "dhna", "short_name": "DHNA",
         "common_name": "D.H.N.A.", "synonyms": ["dhna", "DHNA"]},
    ]))

    assert len(ir["entities"]["compounds"]) == 1
    assert ir["processes"]["reactions"][0]["right"][0]["entity_key"] == "cmp_1"
    assert _row_issues(report) == []


def test_c050k_alias_ambiguity_is_recorded_in_both_payload_row_orders() -> None:
    """NEW ARM. One order alone cannot tell "first wins" from "last wins", and the
    diagnostic must name both candidates whichever row happens to be first."""

    for rows, expected_name in (
        (deepcopy(_AUDITOR_ROWS), "1,4-dihydroxy-2-naphthoic acid"),
        (list(reversed(deepcopy(_AUDITOR_ROWS))), "DHNA"),
    ):
        ir, report = _alias_build(_alias_payload(rows))
        bound_key = ir["processes"]["reactions"][0]["right"][0]["entity_key"]
        bound = next(r for r in ir["entities"]["compounds"] if r["key"] == bound_key)
        assert bound_key == "cmp_1"
        assert bound["name"] == expected_name
        issues = _row_issues(report)
        assert len(issues) == 1
        assert sorted(issues[0]["entity_keys"]) == ["cmp_1", "cmp_2"]
        assert issues[0]["bound_entity_key"] == "cmp_1"
        assert issues[0]["bound_entity_name"] == expected_name


def test_c050k_no_overlap_emits_no_ambiguity_at_all() -> None:
    """NEW ARM, control. Two rows that share nothing must stay silent, so a green
    arm above is not just "the diagnostic fires on everything"."""

    _ir, report = _alias_build(_alias_payload([
        {"name": "1,4-dihydroxy-2-naphthoic acid", "pathbank_compound_id": 40747},
        {"name": "DHNA"},
    ]))

    assert _row_issues(report) == []


def test_c050k_cross_type_ambiguity_is_recorded_and_preference_still_decides() -> None:
    """NEW ARM. 41 of the census's 61 consultations were CROSS-type, where
    ``preferred_order`` does disambiguate deterministically. The choice is still an
    arbitrary-looking one from the reader's side, so section 3 still wants it
    recorded -- and the type preference must keep deciding exactly as at base."""

    payload = _alias_payload(
        [{"name": "enterobactin synthase", "pathbank_compound_id": 1}],
        outputs=["enterobactin synthase"],
        extra_entities={
            "proteins": [{"name": "enterobactin synthase", "species": "Homo sapiens",
                          "pathbank_protein_id": 6301,
                          "mapped_ids": {"uniprot": "P10378"}}],
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}]})

    ir, report = _alias_build(payload)

    # ``preferred_order["reaction_member"]`` puts ``compound`` first, and it still does.
    member = ir["processes"]["reactions"][0]["right"][0]
    assert member["entity_type"] == "compound"
    issues = _row_issues(report)
    assert len(issues) == 1
    assert sorted(issues[0]["entity_types"]) == ["compound", "protein"]
    assert issues[0]["bound_entity_type"] == "compound"


def test_c050k_row_code_is_distinct_from_the_entity_type_code() -> None:
    """NEW ARM. **D-043 section 4 forbids folding the two codes.** On the same-type
    early-return path the pre-existing ``ambiguous_entity_reference`` -- which claims
    "multiple entity TYPES" -- must stay silent, or the census's own distinction is
    destroyed and 20 same-type cases get described as type ambiguity."""

    _ir, report = _alias_build(_alias_payload(deepcopy(_AUDITOR_ROWS)))

    assert _row_issues(report)
    assert not any(issue.get("code") == "ambiguous_entity_reference"
                   for issue in report["errors"] + report["warnings"])


def test_c050k_unresolved_reference_path_is_unchanged() -> None:
    """NEW ARM, boundary. A name matching nothing must still produce exactly the
    pre-existing ``unresolved_entity_reference`` error and no row diagnostic: this
    card added a branch, it did not move the ones already there."""

    _ir, report = _alias_build(_alias_payload([{"name": "DHNA"}],
                                              outputs=["nothing here"]))

    assert any(issue["code"] == "unresolved_entity_reference"
               for issue in report["errors"])
    assert _row_issues(report) == []
