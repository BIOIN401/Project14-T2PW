"""PWML import fixes (docs/pwml_output/IMPORT-FIXES.md): defects that made PathWhiz refuse parts of our pathways.

F1 a transport side that names a location, not a state (run B2 "extracellular"), and a side that names nothing: one bad
   transport never refuses the whole export. F1b an exact repeated state row never splits references over two states.
F2 the same transporter twice on one transport is written once. F3 a promoter/enhancer/locus/plasmid/allele row is DNA.
"""
from __future__ import annotations

from copy import deepcopy

from lxml import etree

from test_pwml_writer import ROOT, _writer_args, build_pwml_ir
from t2pw.pwml.validate import discover_structure_signature
from t2pw.pwml.writer import DeterministicPwmlBuilder, blocking_pwml_ir_errors, nucleotide_type_for


def _payload(transports, states=None):
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [
                {"name": "extracellular", "pathwhiz_id": 2},
                {"name": "cytosol", "pathwhiz_id": 3},
            ],
            "compounds": [{"name": "Glucose", "pathbank_compound_id": 101}, {"name": "Lactate", "pathbank_compound_id": 102}],
            "proteins": [{"name": "GLUT1", "species": "Homo sapiens", "uniprot": "P11166", "pathbank_protein_id": 301}],
        },
        "biological_states": states if states is not None else [
            {"name": "cytosolic state", "species": "Homo sapiens", "subcellular_location": "cytosol"},
            {"name": "extracellular state", "species": "Homo sapiens", "subcellular_location": "extracellular"},
        ],
        "processes": {"reactions": [], "transports": transports, "interactions": []},
    }


def _tr(name, cargo, frm, to, transporters=None):
    row = {"name": name, "cargo": cargo, "from_biological_state": frm, "to_biological_state": to}
    if transporters is not None:
        row["transporters"] = transporters
    return row


def _codes(report, bucket):
    return [issue["code"] for issue in report.get(bucket, [])]


def _state_names(ir, transport):
    names = {s["key"]: s["name"] for s in ir["biological_states"]}
    return [(names[e["left_biological_state_key"]], names[e["right_biological_state_key"]]) for e in transport["transport_elements"]]


def test_b2_side_naming_a_location_resolves_to_the_one_state_there() -> None:
    ir, report = build_pwml_ir(_payload([_tr("Glucose secretion", "Glucose", "cytosolic state", "extracellular")]), strict_db=True)
    assert blocking_pwml_ir_errors(report) == []
    [transport] = ir["processes"]["transports"]
    assert _state_names(ir, transport) == [("cytosolic state", "extracellular state")]
    assert "biological_state_resolved_by_location" in _codes(report, "warnings")


def test_side_naming_no_state_withholds_only_that_transport() -> None:
    payload = _payload([
        _tr("Glucose to nowhere", "Glucose", "cytosolic state", "vacuole"),
        _tr("Lactate export", "Lactate", "cytosolic state", "extracellular state"),
    ])
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert blocking_pwml_ir_errors(report) == []  # before: unresolved_biological_state error refused the whole export
    assert [t["name"] for t in ir["processes"]["transports"]] == ["Lactate export"]
    assert [t["key"] for t in ir["processes"]["transports"]] == ["tr_1"]
    assert report["withheld_transports"] == [{
        "name": "Glucose to nowhere", "pointer": "/processes/transports/0", "reason": "unresolved_biological_state",
        "from_biological_state": "cytosolic state", "to_biological_state": "vacuole"}]
    assert {"transport_withheld_from_pwml", "unresolved_biological_state"} <= set(_codes(report, "warnings"))
    viz = [v for v in ir["process_visualizations"] if v["type"] == "transport_visualization"]
    assert [v["process_key"] for v in viz] == ["tr_1"]


def test_location_shared_by_two_states_is_ambiguous_and_withheld() -> None:
    states = [
        {"name": "cytosolic state", "species": "Homo sapiens", "subcellular_location": "cytosol"},
        {"name": "hepatocyte cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol", "cell_type": "hepatocyte"},
        {"name": "extracellular state", "species": "Homo sapiens", "subcellular_location": "extracellular"},
    ]
    payload = _payload([_tr("Glucose import", "Glucose", "extracellular state", "cytosol")], states)
    payload["entities"]["cell_types"] = [{"name": "hepatocyte", "pathwhiz_id": 9}]
    ir, report = build_pwml_ir(payload, strict_db=True)
    assert ir["processes"]["transports"] == []
    assert report["withheld_transports"][0]["to_biological_state"] == "cytosol"
    assert blocking_pwml_ir_errors(report) == []


def test_transporter_naming_no_state_loses_the_transporter_not_the_transport() -> None:
    tr = _tr("Glucose import", "Glucose", "extracellular state", "cytosolic state",
             [{"protein": "GLUT1", "biological_state": "nowhere"}])
    ir, report = build_pwml_ir(_payload([tr]), strict_db=True)
    assert blocking_pwml_ir_errors(report) == []
    [transport] = ir["processes"]["transports"]
    assert transport["transporters"] == [] and len(transport["transport_elements"]) == 1


def test_exact_repeated_state_row_resolves_to_the_first_and_its_location_stays_unambiguous() -> None:
    states = [
        {"name": "cytosolic state", "species": "Homo sapiens", "subcellular_location": "cytosol"},
        {"name": "extracellular state", "species": "Homo sapiens", "subcellular_location": "extracellular"},
        {"name": "cytosolic state", "species": "Homo sapiens", "subcellular_location": "cytosol", "evidence": "other"},
    ]
    ir, report = build_pwml_ir(_payload([
        _tr("Glucose import", "Glucose", "extracellular state", "cytosolic state"),
        _tr("Lactate import", "Lactate", "extracellular state", "cytosol"),
    ], states), strict_db=True)
    assert blocking_pwml_ir_errors(report) == []
    first = ir["biological_states"][0]["key"]
    for transport in ir["processes"]["transports"]:
        assert transport["transport_elements"][0]["right_biological_state_key"] == first


def test_same_transporter_twice_is_written_once() -> None:
    tr = _tr("Glucose import", "Glucose", "extracellular state", "cytosolic state",
             [{"protein": "GLUT1", "biological_state": "cytosolic state"},
              {"protein": "GLUT1", "biological_state": "cytosolic state", "provenance": "inferred"}])
    ir, report = build_pwml_ir(_payload([tr]), strict_db=True)
    assert len(ir["processes"]["transports"][0]["transporters"]) == 1
    assert "duplicate_transporter_written_once" in _codes(report, "warnings")
    signature = discover_structure_signature(ROOT / "reference" / "PW000001.pwml")
    result = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=_writer_args()).build()
    xml = etree.tostring(result.root).decode()
    assert xml.count("<transport-transporter>") == 1


def test_distinct_transporters_are_all_kept() -> None:
    payload = _payload([_tr("Glucose import", "Glucose", "extracellular state", "cytosolic state",
                            [{"protein": "GLUT1", "biological_state": "cytosolic state"},
                             {"protein": "GLUT1", "biological_state": "extracellular state"}])])
    ir, _ = build_pwml_ir(payload, strict_db=True)
    assert len(ir["processes"]["transports"][0]["transporters"]) == 2


def test_resolved_payload_is_unchanged_by_the_fallback() -> None:
    payload = _payload([_tr("Glucose import", "Glucose", "extracellular state", "cytosolic state")])
    ir, report = build_pwml_ir(deepcopy(payload), strict_db=True)
    assert "withheld_transports" not in report
    assert not {"biological_state_resolved_by_location", "transport_withheld_from_pwml"} & set(_codes(report, "warnings"))
    assert _state_names(ir, ir["processes"]["transports"][0]) == [("extracellular state", "cytosolic state")]


def test_dna_region_names_are_dna_and_protein_names_stay_untyped() -> None:
    for name in ("CHLH promoter", "HEMA1 promoter", "ALAD1/HEMB1 promoter", "SV40 enhancer", "HY5 locus", "pUC19 plasmid",
                 "mutant alleles"):
        assert nucleotide_type_for({"name": name}) == "DNA", name
    for name in ("promoter binding factor", "CHL27", "LHCB1.1", "ALAD1/HEMB1", "RyhB"):
        assert nucleotide_type_for({"name": name}) is None, name
    assert nucleotide_type_for({"name": "promoter-associated RNA"}) is None  # both kinds: unsettled
