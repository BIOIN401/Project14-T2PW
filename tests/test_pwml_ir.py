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
                {"name": "Hexokinase", "pathbank_protein_id": 201, "mapped_ids": {"uniprot": "P19367"}},
            ],
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
                    "enzymes": [{"protein": "Hexokinase"}],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


def test_reaction_ir_construction_refs_resolve() -> None:
    ir, report = build_pwml_ir(_base_payload(), strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not report["errors"]
    assert validation["ok"], validation["errors"]
    reaction = ir["processes"]["reactions"][0]
    assert reaction["left"][0]["entity_key"] == ir["entities"]["compounds"][0]["key"]
    assert reaction["right"][0]["entity_key"] == ir["entities"]["compounds"][1]["key"]
    assert reaction["enzymes"][0]["entity_type"] == "protein_complex"
    assert reaction["enzymes"][0]["entity_key"] == ir["entities"]["protein_complexes"][0]["key"]
    assert ir["entities"]["protein_complexes"][0]["components"] == [{"protein_key": ir["entities"]["proteins"][0]["key"], "stoichiometry": 1}]
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

    ir, report = build_pwml_ir(payload, strict_db=True)
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

    ir, report = build_pwml_ir(payload, strict_db=True)
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

    ir, report = build_pwml_ir(payload, strict_db=True)
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

    ir, report = build_pwml_ir(payload, strict_db=True)
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
    ir, report = build_pwml_ir(payload, strict_db=True)
    validation = validate_pwml_ir(ir)

    assert "protein_complex_missing_db_identity" not in {err["code"] for err in contract["errors"]}
    assert not any(
        err["code"] == "missing_db_identity" and err.get("entity_type") == "protein_complex"
        for err in report["errors"]
    )
    assert validation["ok"], validation["errors"]


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

    ir, report = build_pwml_ir(payload, strict_db=True)
    validation = validate_pwml_ir(ir)

    assert not report["errors"]
    assert validation["ok"], validation["errors"]
    complex_record = ir["entities"]["protein_complexes"][0]
    assert complex_record["components"] == [{"protein_key": "prot_1", "stoichiometry": 1}]

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
    assert complex_members == [{"id": complex_members[0]["id"], "protein-id": protein_id, "stoichiometry": 1}]


def test_protein_complex_unresolved_component_is_not_exportable() -> None:
    payload = _base_payload()
    payload["entities"]["protein_complexes"] = [
        {
            "name": "Broken complex",
            "pathbank_complex_id": 401,
            "species": "Homo sapiens",
            "components": ["Missing protein"],
        }
    ]

    ir, report = build_pwml_ir(payload, strict_db=True)
    validation = validate_pwml_ir(ir)

    assert any(err["code"] == "component_protein_unresolved" for err in report["errors"])
    assert any(err["code"] == "protein_complex_missing_components" for err in validation["errors"])
    assert not validation["ok"]


def test_dangling_process_visualization_references_fail_validation() -> None:
    ir, report = build_pwml_ir(_base_payload(), strict_db=True)
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
            ]
        },
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    ir, report = build_pwml_ir(payload, strict_db=True)
    pyruvate_key = ir["entities"]["compounds"][0]["key"]
    pyruvate_locations = [loc for loc in ir["locations"] if loc["entity_key"] == pyruvate_key]

    assert not report["errors"]
    assert len(ir["entities"]["compounds"]) == 1
    assert len(pyruvate_locations) == 2
    assert len({loc["biological_state_key"] for loc in pyruvate_locations}) == 2


def test_ir_writer_integration_validates_and_has_no_fatal_qa_errors() -> None:
    ir, report = build_pwml_ir(_base_payload(), strict_db=True)
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
