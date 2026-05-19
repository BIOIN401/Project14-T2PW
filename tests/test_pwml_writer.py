from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.ir import build_pwml_ir  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402


def _payload_with_complex_enzyme() -> dict:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {"name": "Glucose", "pathbank_compound_id": 101},
                {"name": "Glucose 6-phosphate", "pathbank_compound_id": 102},
            ],
            "proteins": [{"name": "Hexokinase", "pathbank_protein_id": 201}],
        },
        "biological_states": [{"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"}],
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


def test_writer_emits_visible_complex_and_reaction_enzyme_visualization() -> None:
    ir, report = build_pwml_ir(_payload_with_complex_enzyme(), strict_db=True)
    assert not report["errors"]

    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    args = SimpleNamespace(
        name="Generated Pathway",
        description="",
        subject="Metabolic",
        pw_id="PW000000",
        height=1400,
        width=3200,
        background_color="#FFFFFF",
        ref=str(ROOT / "reference" / "PW000001.pwml"),
    )
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args)
    builder.build()

    complex_visualizations = builder.section_items["protein-complex-visualizations"]
    reaction_visualizations = builder.section_items["reaction-visualizations"]

    assert len(complex_visualizations) == 1
    assert builder.section_items["compounds"][0]["id"] == 101
    assert builder.section_items["proteins"][0]["id"] != 201
    assert builder.section_items["protein-locations"][0]["visualization-template-id"] == 2
    assert builder.section_items["protein-locations"][0]["label-type"] == "subunit"
    enzyme_viz = reaction_visualizations[0]["reaction_enzyme_visualizations"][0]
    assert enzyme_viz["protein-complex-visualization-id"] == complex_visualizations[0]["id"]
    assert "protein-location-id" not in enzyme_viz
