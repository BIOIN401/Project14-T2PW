from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.ir import build_pwml_ir  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature, repair_tree  # noqa: E402
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


def _writer_args() -> SimpleNamespace:
    return SimpleNamespace(
        name="Generated Pathway",
        description="",
        subject="Metabolic",
        pw_id="PW000000",
        height=1400,
        width=3200,
        background_color="#FFFFFF",
        ref=str(ROOT / "reference" / "PW000001.pwml"),
    )


def _minimal_pwml_ir_with_compounds(compounds: list[dict]) -> dict:
    return {
        "pathway": {
            "key": "pathway_1",
            "name": "Generated Pathway",
            "subject": "Metabolic",
            "width": 3200,
            "height": 1400,
        },
        "species": [],
        "subcellular_locations": [],
        "cell_types": [],
        "tissues": [],
        "biological_states": [],
        "entities": {
            "compounds": compounds,
            "proteins": [],
            "nucleic_acids": [],
            "element_collections": [],
            "protein_complexes": [],
            "bounds": [],
        },
        "processes": {
            "reactions": [],
            "reaction_coupled_transports": [],
            "transports": [],
            "interactions": [],
            "sub_pathways": [],
        },
        "locations": [],
        "protein_complex_visualizations": [],
        "bound_visualizations": [],
        "edges": [],
        "process_visualizations": [],
    }


def _minimal_pwml_ir_with_biological_states(biological_states: list[dict]) -> dict:
    ir = _minimal_pwml_ir_with_compounds([])
    ir["species"] = [{"key": "sp_1", "name": "Homo sapiens", "pathwhiz_id": 1}]
    ir["subcellular_locations"] = [{"key": "scl_1", "name": "cytosol", "pathwhiz_id": 2}]
    ir["biological_states"] = biological_states
    return ir


def _build_pwml_for_ir(ir: dict) -> tuple[DeterministicPwmlBuilder, object]:
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=_writer_args())
    build = builder.build()
    return builder, build.root


def _compound_xml(root: object) -> bytes:
    return etree.tostring(root, encoding="utf-8")


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


def test_db_matched_compound_emits_trusted_pwc_id_and_short_name() -> None:
    ir = _minimal_pwml_ir_with_compounds(
        [
            {
                "key": "cmp_1",
                "name": "Extracted glucose",
                "pathwhiz_id": 77,
                "db_status": "matched",
                "db_row": {
                    "id": 77,
                    "name": "D-Glucose",
                    "pwc_id": "PW_C000077",
                    "short_name": "D-Glc",
                },
            }
        ]
    )

    builder, root = _build_pwml_for_ir(ir)

    compound = builder.section_items["compounds"][0]
    assert compound["id"] == 77
    assert compound["pwc-id"] == "PW_C000077"
    assert compound["short-name"] == "D-Glc"

    compound_node = root.find(".//compounds/compound")
    assert compound_node is not None
    assert compound_node.findtext("id") == "77"
    assert compound_node.findtext("pwc-id") == "PW_C000077"
    assert compound_node.findtext("short-name") == "D-Glc"


def test_structured_ir_novel_compound_omits_synthetic_pwc_id() -> None:
    ir = _minimal_pwml_ir_with_compounds([{"key": "cmp_1", "name": "Novel compound"}])

    builder, root = _build_pwml_for_ir(ir)

    compound = builder.section_items["compounds"][0]
    assert compound["id"] == 20000
    assert "pwc-id" not in compound
    assert root.find(".//compounds/compound/pwc-id") is None
    repaired_root = repair_tree(etree.ElementTree(root), builder.signature).getroot()
    assert repaired_root.find(".//compounds/compound/pwc-id") is None
    assert b"PW_C020000" not in _compound_xml(root)


def test_structured_ir_novel_compound_does_not_trust_record_pwc_id() -> None:
    ir = _minimal_pwml_ir_with_compounds(
        [{"key": "cmp_1", "name": "Novel compound", "pwc_id": "PW_C020001"}]
    )

    builder, root = _build_pwml_for_ir(ir)

    compound = builder.section_items["compounds"][0]
    assert "pwc-id" not in compound
    assert root.find(".//compounds/compound/pwc-id") is None
    assert b"PW_C020001" not in _compound_xml(root)


def test_structured_ir_novel_compound_does_not_trust_mapped_pwc_id() -> None:
    ir = _minimal_pwml_ir_with_compounds(
        [{"key": "cmp_1", "name": "Novel compound", "mapped_ids": {"pwc_id": "PW_C020001"}}]
    )

    builder, root = _build_pwml_for_ir(ir)

    compound = builder.section_items["compounds"][0]
    assert "pwc-id" not in compound
    assert root.find(".//compounds/compound/pwc-id") is None
    assert b"PW_C020001" not in _compound_xml(root)


def test_structured_ir_novel_compound_omits_unsafe_short_name() -> None:
    long_name = "N10-demethylated synthetic intermediate with unresolved PathWhiz identity"
    ir = _minimal_pwml_ir_with_compounds(
        [{"key": "cmp_1", "name": long_name, "short_name": long_name}]
    )

    builder, root = _build_pwml_for_ir(ir)

    compound = builder.section_items["compounds"][0]
    assert "short-name" not in compound
    assert root.find(".//compounds/compound/short-name") is None
    assert b"<short-name>" not in _compound_xml(root)


def test_fallback_novel_compound_omits_synthetic_pwc_id_and_short_name() -> None:
    payload = {"entities": {"compounds": [{"name": "Novel fallback compound"}]}, "processes": {}}
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(extraction=payload, signature=signature, args=_writer_args())
    build = builder.build()

    compound = builder.section_items["compounds"][0]
    assert compound["id"] == 20000
    assert "pwc-id" not in compound
    assert "short-name" not in compound
    assert build.root.find(".//compounds/compound/pwc-id") is None
    assert build.root.find(".//compounds/compound/short-name") is None
    repaired_root = repair_tree(etree.ElementTree(build.root), builder.signature).getroot()
    assert repaired_root.find(".//compounds/compound/pwc-id") is None
    assert repaired_root.find(".//compounds/compound/short-name") is None
    assert b"PW_C020000" not in _compound_xml(build.root)


def test_fallback_generated_biological_state_omits_pwbs_id() -> None:
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
        },
        "biological_states": [{"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"}],
        "processes": {},
    }
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(extraction=payload, signature=signature, args=_writer_args())
    build = builder.build()

    state = builder.section_items["biological-states"][0]
    assert isinstance(state["id"], int)
    assert state["species-id"] == 1
    assert state["subcellular-location-id"] == 2
    assert "pwbs-id" not in state
    assert build.root.find(".//biological-states/biological-state/pwbs-id") is None
    repaired_root = repair_tree(etree.ElementTree(build.root), builder.signature).getroot()
    assert repaired_root.find(".//biological-states/biological-state/pwbs-id") is None
    assert b"<pwbs-id>" not in _compound_xml(build.root)


def test_fallback_transport_visualization_populates_edges_and_transporter() -> None:
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
            {
                "name": "extracellular",
                "species": "Homo sapiens",
                "subcellular_location": "extracellular",
                "compartment_canonical": "extracellular",
            },
            {
                "name": "cytosol",
                "species": "Homo sapiens",
                "subcellular_location": "cytosol",
                "compartment_canonical": "cytosol",
            },
        ],
        "processes": {
            "reactions": [],
            "transports": [
                {
                    "name": "Glucose import",
                    "cargo": "Glucose",
                    "from_biological_state": "extracellular",
                    "to_biological_state": "cytosol",
                    "transporters": [{"protein": "GLUT1"}],
                }
            ],
            "interactions": [],
        },
    }
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    builder = DeterministicPwmlBuilder(extraction=payload, signature=signature, args=_writer_args())
    builder.build()

    transport_viz = builder.section_items["transport-visualizations"][0]
    compound_viz = transport_viz["transport_compound_visualizations"]
    transporter_viz = transport_viz["transport_transporter_visualizations"]
    assert {item["side"] for item in compound_viz} == {"Left", "Right"}
    assert len(compound_viz) == 2
    assert len(transporter_viz) == 1
    assert "edge-id" not in transporter_viz[0]

    edge_by_id = {edge["id"]: edge for edge in builder.section_items["edges"]}
    loc_by_id = {loc["id"]: loc for loc in builder.section_items["compound-locations"]}
    protein_loc = builder.section_items["protein-locations"][0]
    transporter_top = int(protein_loc["x"]) + int(protein_loc["width"]) // 2, int(protein_loc["y"])
    transporter_bottom = transporter_top[0], int(protein_loc["y"]) + int(protein_loc["height"])
    assert int(protein_loc["y"]) == int(builder.args.height) // 2
    assert transporter_viz[0]["protein-location-id"] == protein_loc["id"]

    left_viz = next(item for item in compound_viz if item["side"] == "Left")
    left_loc = loc_by_id[left_viz["compound-location-id"]]
    left_edge = edge_by_id[left_viz["edge-id"]]
    left_anchor = int(left_loc["x"]) + int(left_loc["width"]) // 2, int(left_loc["y"]) + int(left_loc["height"])
    assert left_edge["path"] == f"M{left_anchor[0]} {left_anchor[1]} L{transporter_top[0]} {transporter_top[1]}"
    assert left_edge["visualization-template-id"] == 83
    assert left_edge["option:end_arrow"] is True
    assert "option:start_arrow" not in left_edge

    right_viz = next(item for item in compound_viz if item["side"] == "Right")
    right_loc = loc_by_id[right_viz["compound-location-id"]]
    right_edge = edge_by_id[right_viz["edge-id"]]
    right_anchor = int(right_loc["x"]) + int(right_loc["width"]) // 2, int(right_loc["y"])
    assert right_edge["path"] == (
        f"M{right_anchor[0]} {right_anchor[1]} L{transporter_bottom[0]} {transporter_bottom[1]}"
    )
    assert right_edge["visualization-template-id"] == 83
    assert right_edge["option:start_arrow"] is True
    assert "option:end_arrow" not in right_edge


def test_structured_ir_generated_biological_state_omits_pwbs_id_but_keeps_local_context_and_refs() -> None:
    ir, report = build_pwml_ir(_payload_with_complex_enzyme(), strict_db=True)
    assert not report["errors"]

    builder, root = _build_pwml_for_ir(ir)

    state = builder.section_items["biological-states"][0]
    state_id = state["id"]
    assert isinstance(state_id, int)
    assert "pwbs-id" not in state
    assert state["species-id"] == 1
    assert state["subcellular-location-id"] == 2

    state_node = root.find(".//biological-states/biological-state")
    assert state_node is not None
    assert state_node.findtext("id") == str(state_id)
    assert state_node.findtext("species-id") == "1"
    assert state_node.findtext("subcellular-location-id") == "2"
    assert state_node.find("pwbs-id") is None

    assert root.findtext(".//compound-locations/compound-location/biological-state-id") == str(state_id)
    assert root.findtext(".//reaction-visualizations/reaction-visualization/biological-state-id") == str(state_id)
    assert root.findtext(".//protein-locations/protein-location/biological-state-id") == str(state_id)

    repaired_root = repair_tree(etree.ElementTree(root), builder.signature).getroot()
    assert repaired_root.find(".//biological-states/biological-state/pwbs-id") is None
    assert b"<pwbs-id>" not in _compound_xml(root)
    assert b"PW_BS000003" not in _compound_xml(root)


def test_structured_ir_db_backed_biological_state_emits_real_pwbs_id() -> None:
    ir = _minimal_pwml_ir_with_biological_states(
        [
            {
                "key": "bs_1",
                "name": "cytosol",
                "species_key": "sp_1",
                "subcellular_location_key": "scl_1",
                "db_status": "matched",
                "pwbs_id": "PW_BS000123",
            }
        ]
    )

    builder, root = _build_pwml_for_ir(ir)

    state = builder.section_items["biological-states"][0]
    assert state["pwbs-id"] == "PW_BS000123"
    assert root.findtext(".//biological-states/biological-state/pwbs-id") == "PW_BS000123"


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


def test_writer_places_reaction_elements_around_enzyme_center() -> None:
    ir, report = build_pwml_ir(_payload_with_complex_enzyme(), strict_db=True)
    assert not report["errors"]

    builder, _root = _build_pwml_for_ir(ir)

    enzyme = builder.section_items["protein-locations"][0]
    enzyme_x = int(enzyme["x"])
    enzyme_y = int(enzyme["y"])
    enzyme_w = int(enzyme["width"])
    enzyme_h = int(enzyme["height"])
    enzyme_cy = enzyme_y + enzyme_h // 2
    assert (enzyme_x, enzyme_y, enzyme_w, enzyme_h) == (1525, 665, 150, 70)

    compounds = {loc["compound-id"]: loc for loc in builder.section_items["compound-locations"]}
    substrate = compounds[101]
    product = compounds[102]
    substrate_right = int(substrate["x"]) + int(substrate["width"])
    product_left = int(product["x"])
    substrate_cy = int(substrate["y"]) + int(substrate["height"]) // 2
    product_cy = int(product["y"]) + int(product["height"]) // 2

    assert enzyme_x - substrate_right == 46
    assert product_left - (enzyme_x + enzyme_w) == 51
    assert substrate_cy == enzyme_cy - 139
    assert product_cy == enzyme_cy - 139

    reaction_viz = builder.section_items["reaction-visualizations"][0]
    edge_by_side = {
        rcv["side"]: next(
            edge for edge in builder.section_items["edges"] if edge["id"] == rcv["edge-id"]
        )
        for rcv in reaction_viz["reaction_compound_visualizations"]
    }
    assert edge_by_side["Left"]["path"] == f"M{substrate_right} {substrate_cy} L{enzyme_x} {enzyme_cy}"
    assert edge_by_side["Left"]["visualization-template-id"] == 5
    assert edge_by_side["Left"]["option:end_arrow"] is True
    assert "option:start_arrow" not in edge_by_side["Left"]
    assert edge_by_side["Right"]["path"] == (
        f"M{product_left} {product_cy} L{enzyme_x + enzyme_w} {enzyme_cy}"
    )
    assert edge_by_side["Right"]["visualization-template-id"] == 5
    assert edge_by_side["Right"]["option:start_arrow"] is True
    assert "option:end_arrow" not in edge_by_side["Right"]


def test_writer_uses_virtual_edges_for_reactions_without_enzymes() -> None:
    payload = _payload_with_complex_enzyme()
    payload["entities"].pop("proteins")
    payload["processes"]["reactions"][0]["enzymes"] = []
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert not report["errors"]

    builder, _root = _build_pwml_for_ir(ir)

    compounds = {loc["compound-id"]: loc for loc in builder.section_items["compound-locations"]}
    substrate = compounds[101]
    product = compounds[102]
    substrate_right = int(substrate["x"]) + int(substrate["width"])
    product_left = int(product["x"])
    center_y = int(builder.args.height) // 2
    virtual_left = substrate_right + 46
    virtual_right = virtual_left + 70

    reaction_viz = builder.section_items["reaction-visualizations"][0]
    edge_by_side = {
        rcv["side"]: next(
            edge for edge in builder.section_items["edges"] if edge["id"] == rcv["edge-id"]
        )
        for rcv in reaction_viz["reaction_compound_visualizations"]
    }

    left_edge = edge_by_side["Left"]
    assert left_edge["path"] == f"M{virtual_right} {center_y} L{virtual_right} {center_y}"
    assert left_edge["visualization-template-id"] == 5
    assert left_edge["hidden"] is True
    assert left_edge["option:end_arrow"] is True
    assert "option:start_arrow" not in left_edge

    right_edge = edge_by_side["Right"]
    assert right_edge["path"] == f"M{product_left} {center_y} L{virtual_left} {center_y}"
    assert right_edge["visualization-template-id"] == 5
    assert right_edge["hidden"] is False
    assert right_edge["option:start_arrow"] is True
    assert "option:end_arrow" not in right_edge


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
