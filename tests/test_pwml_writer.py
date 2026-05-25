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
from t2pw.pwml.writer import DeterministicPwmlBuilder, blocking_pwml_ir_errors  # noqa: E402


HEXOKINASE_COMPOUND_ROWS = [
    {
        "id": 77,
        "name": "D-Glucose",
        "short_name": "D-Glc",
        "hmdb_id": "HMDB0000122",
        "kegg_id": "C00031",
        "chebi_id": "4167",
        "pubchem_cid": "5793",
        "pwc_id": "PW_C000077",
        "cas": "",
        "biocyc_id": "",
        "chemspider_id": "",
        "drugbank_id": "",
        "description": "DB glucose row",
    },
    {
        "id": 414,
        "name": "Adenosine triphosphate",
        "short_name": "ATP",
        "hmdb_id": "HMDB0000538",
        "kegg_id": "C00002",
        "chebi_id": "15422",
        "pubchem_cid": "5957",
        "pwc_id": "PW_C000414",
        "cas": "",
        "biocyc_id": "",
        "chemspider_id": "",
        "drugbank_id": "",
        "description": "",
    },
    {
        "id": 1083,
        "name": "Glucose 6-phosphate",
        "short_name": "Gluc-6P",
        "hmdb_id": "HMDB0001401",
        "kegg_id": "C00092",
        "chebi_id": "4170",
        "pubchem_cid": "5958",
        "pwc_id": "PW_C001083",
        "cas": "",
        "biocyc_id": "",
        "chemspider_id": "",
        "drugbank_id": "",
        "description": "",
    },
    {
        "id": 1034,
        "name": "Adenosine diphosphate",
        "short_name": "ADP",
        "hmdb_id": "HMDB0001341",
        "kegg_id": "C00008",
        "chebi_id": "16761",
        "pubchem_cid": "6022",
        "pwc_id": "PW_C001034",
        "cas": "",
        "biocyc_id": "",
        "chemspider_id": "",
        "drugbank_id": "",
        "description": "",
    },
    {
        "id": 40034,
        "name": "Hydrogen Ion",
        "short_name": "H+",
        "hmdb_id": "HMDB0059597",
        "kegg_id": "C00080",
        "chebi_id": "15378",
        "pubchem_cid": "1038",
        "pwc_id": "PW_C040034",
        "cas": "",
        "biocyc_id": "",
        "chemspider_id": "",
        "drugbank_id": "",
        "description": "",
    },
]


class _CompoundDb:
    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: tuple) -> list[dict]:
        value = str(params[0]).strip().lower()
        column = ""
        for candidate in ["pwc_id", "hmdb_id", "kegg_id", "chebi_id", "pubchem_cid", "id"]:
            if f"{candidate})" in sql or f" {candidate}=" in sql:
                column = candidate
                break
        if column == "id":
            return [row for row in HEXOKINASE_COMPOUND_ROWS if str(row["id"]).lower() == value]
        if column:
            return [row for row in HEXOKINASE_COMPOUND_ROWS if str(row.get(column) or "").lower() == value]
        return []


class _EmptyCompoundDb:
    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: tuple) -> list[dict]:
        return []


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


def test_compound_db_resolution_failures_are_non_blocking_for_pwml_build() -> None:
    payload = _payload_with_complex_enzyme()
    payload["entities"]["compounds"] = [
        {"name": "norbelladine"},
        {"name": "Schiff-base intermediate"},
    ]
    payload["processes"]["reactions"][0]["inputs"] = ["norbelladine"]
    payload["processes"]["reactions"][0]["outputs"] = ["Schiff-base intermediate"]

    ir, report = build_pwml_ir(payload, strict_db=True, db_resolver=_EmptyCompoundDb())

    assert report["errors"]
    assert {err["code"] for err in report["errors"]} == {"compound_db_resolution_failed"}
    assert blocking_pwml_ir_errors(report) == []

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

    assert {item["name"] for item in builder.section_items["compounds"]} == {
        "norbelladine",
        "Schiff-base intermediate",
    }


def test_structural_ir_errors_still_block_pwml_export_policy() -> None:
    report = {
        "errors": [
            {
                "code": "biological_state_missing_species",
                "message": "Biological state has no resolved species reference.",
            }
        ]
    }

    assert blocking_pwml_ir_errors(report) == report["errors"]


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


def test_pwml_uses_db_exact_compound_rows_and_ids_for_hexokinase() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {"name": "Glucose", "kegg_id": "C00031", "hmdb_id": "HMDB0000122", "chebi_id": "CHEBI:4167"},
                {"name": "ATP", "pwc_id": "PW_C000414", "kegg_id": "C00002", "hmdb_id": "HMDB0000538"},
                {"name": "Glucose-6-phosphate", "kegg_id": "C00092", "hmdb_id": "HMDB0001401", "chebi_id": "CHEBI:4170"},
                {"name": "ADP", "pwc_id": "PW_C001034", "kegg_id": "C00008", "hmdb_id": "HMDB0001341"},
                {"name": "H+", "kegg_id": "C00080", "hmdb_id": "HMDB0059597", "chebi_id": "CHEBI:15378"},
            ],
            "proteins": [{"name": "Hexokinase", "pathbank_protein_id": 201}],
        },
        "biological_states": [{"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"}],
        "processes": {
            "reactions": [
                {
                    "name": "Glucose phosphorylation",
                    "inputs": ["Glucose", "ATP"],
                    "outputs": ["Glucose-6-phosphate", "ADP", "H+"],
                    "biological_state": "cytosol",
                    "enzymes": [{"protein": "Hexokinase"}],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }

    ir, report = build_pwml_ir(payload, strict_db=True, db_resolver=_CompoundDb())
    assert not report["errors"]
    assert all(item["status"] == "matched" for item in report["db_resolution"]["compounds"])
    assert {compound["name"] for compound in ir["entities"]["compounds"]} >= {
        "D-Glucose",
        "Glucose 6-phosphate",
        "Hydrogen Ion",
    }

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

    compounds = {item["id"]: item for item in builder.section_items["compounds"]}
    assert compounds[77]["name"] == "D-Glucose"
    assert compounds[77]["short-name"] == "D-Glc"
    assert compounds[77]["hmdb-id"] == "HMDB0000122"
    assert compounds[77]["kegg-id"] == "C00031"
    assert compounds[77]["chebi-id"] == "4167"
    assert compounds[77]["pubchem-cid"] == "5793"
    assert compounds[77]["pwc-id"] == "PW_C000077"
    assert compounds[1083]["name"] == "Glucose 6-phosphate"
    assert compounds[40034]["name"] == "Hydrogen Ion"
    assert compounds[40034]["chebi-id"] == "15378"

    reaction = builder.section_items["reactions"][0]
    element_ids = {
        item["element-id"]
        for side in ["reaction-left-elements", "reaction-right-elements"]
        for item in reaction[side]
    }
    assert element_ids == {77, 414, 1083, 1034, 40034}
    assert {loc["compound-id"] for loc in builder.section_items["compound-locations"]} == element_ids
    assert all(
        isinstance(loc["visualization-template-id"], int)
        and loc["visualization-template-id"] > 0
        for loc in builder.section_items["compound-locations"]
    )
    reaction_viz = builder.section_items["reaction-visualizations"][0]
    location_ids = {loc["id"] for loc in builder.section_items["compound-locations"]}
    edge_ids = {edge["id"] for edge in builder.section_items["edges"]}
    rcvs = reaction_viz["reaction_compound_visualizations"]
    assert len(rcvs) == 5
    assert all(rcv["compound-location-id"] in location_ids for rcv in rcvs)
    assert all(rcv["edge-id"] in edge_ids for rcv in rcvs)
    assert {rcv["side"] for rcv in rcvs} == {"Left", "Right"}
