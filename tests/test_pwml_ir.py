from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir  # noqa: E402
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
    assert reaction["enzymes"][0]["entity_key"] == ir["entities"]["proteins"][0]["key"]

    viz = ir["process_visualizations"][0]
    assert viz["type"] == "reaction_visualization"
    assert {m["process_member_key"] for m in viz["members"]} >= {
        reaction["left"][0]["key"],
        reaction["right"][0]["key"],
        reaction["enzymes"][0]["key"],
    }


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


def test_strict_db_missing_compound_identity_is_error() -> None:
    payload = _base_payload()
    payload["entities"]["compounds"][0].pop("pathbank_compound_id")

    _ir, report = build_pwml_ir(payload, strict_db=True)

    assert any(err["code"] == "missing_db_identity" and err["entity_type"] == "compound" for err in report["errors"])


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

    assert validation["ok"], validation["issues"][:3]
    assert qa["ok"], qa["errors"]
