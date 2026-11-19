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
    held = {"name": "Glucose to nowhere", "pointer": "/processes/transports/0", "payload_index": 0,
            "reason": "unresolved_biological_state",
            "failed_sides": [{"element": "Glucose", "side": "right", "biological_state": "vacuole"}]}
    assert report["withheld_transports"] == [held] and ir["withheld_transports"] == [held]
    assert {"transport_withheld_from_pwml", "unresolved_biological_state"} <= set(_codes(report, "warnings"))
    viz = [v for v in ir["process_visualizations"] if v["type"] == "transport_visualization"]
    assert [v["process_key"] for v in viz] == ["tr_1"]


def test_withheld_record_names_the_cargo_level_side_that_failed() -> None:
    tr = {"name": "Mixed", "from_biological_state": "cytosolic state", "to_biological_state": "extracellular state",
          "transport_elements": [{"element": "Glucose"},
                                 {"element": "Lactate", "left_biological_state": "lysosome lumen"}]}
    ir, report = build_pwml_ir(_payload([tr]), strict_db=True)
    assert ir["processes"]["transports"] == []
    assert report["withheld_transports"][0]["failed_sides"] == [
        {"element": "Lactate", "side": "left", "biological_state": "lysosome lumen"}]


def test_location_fallback_is_for_transport_sides_only() -> None:
    """Review finding 3: a reaction or a placement that names a location keeps base behaviour (default state)."""
    payload = _payload([])
    payload["element_locations"] = {"compound_locations": [{"compound": "Lactate", "biological_state": "extracellular"}]}
    ir, report = build_pwml_ir(payload, strict_db=False)
    assert "biological_state_resolved_by_location" not in _codes(report, "warnings")
    assert "unresolved_biological_state" in _codes(report, "warnings") + _codes(report, "errors")


def test_location_fallback_needs_the_sides_own_qualifiers_to_agree() -> None:
    states = [
        {"name": "cytosolic state", "species": "Homo sapiens", "subcellular_location": "cytosol"},
        {"name": "liver extracellular", "species": "Homo sapiens", "subcellular_location": "extracellular", "tissue": "liver"},
    ]
    def run(tissue):
        tr = _tr("Glucose export", "Glucose", "cytosolic state", "extracellular")
        if tissue:
            tr["tissue"] = tissue
        payload = _payload([tr], deepcopy(states))
        payload["entities"]["tissues"] = [{"name": "liver", "pathwhiz_id": 5}, {"name": "kidney", "pathwhiz_id": 6}]
        return build_pwml_ir(payload, strict_db=True)
    ir, _ = run("kidney")  # the side says kidney; the only extracellular state is the liver's: no match
    assert ir["processes"]["transports"] == []
    ir, _ = run("liver")
    assert _state_names(ir, ir["processes"]["transports"][0]) == [("cytosolic state", "liver extracellular")]
    ir, _ = run(None)  # no qualifier on the side: the one state of that location
    assert len(ir["processes"]["transports"]) == 1


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
    assert report["withheld_transports"][0]["failed_sides"][0]["biological_state"] == "cytosol"
    assert blocking_pwml_ir_errors(report) == []


def test_transporter_naming_no_state_loses_the_transporter_not_the_transport() -> None:
    tr = _tr("Glucose import", "Glucose", "extracellular state", "cytosolic state",
             [{"protein": "GLUT1", "biological_state": "nowhere"}])
    ir, report = build_pwml_ir(_payload([tr]), strict_db=True)
    assert blocking_pwml_ir_errors(report) == []
    [transport] = ir["processes"]["transports"]
    assert transport["transporters"] == [] and len(transport["transport_elements"]) == 1
    assert [h["transporter"] for h in ir["withheld_transporters"]] == ["GLUT1"]
    assert "transporter_withheld_from_pwml" in _codes(report, "warnings")


def test_strict_quarantine_omits_the_withheld_transport_and_caps_the_release() -> None:
    """D-114 s2/s3: an IR-withheld transport is a per-row omission in strict mode: listed, review_required."""
    from test_orch773_omit_per_row import _leg, _omitted, _quarantine, _refusals, _surviving

    payload, context = _leg("runs_verify_2026-08-21_2057_PMC12096016")
    names = [t["name"] for t in payload["processes"]["transports"]]
    payload["processes"]["transports"][0]["to_biological_state"] = "vacuole lumen"
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    assert [o[0] for o in _omitted(result)] == [names[0]]
    assert _omitted(result)[0][2] == "transport_withheld_from_pwml", _omitted(result)
    assert [t["name"] for t in _surviving(result, "transports")] == names[1:]
    release = result.quarantine_report["release"]
    assert release["status"] == "review_required" and release["strict_acceptance_eligible"] is False
    assert any(r.startswith("omitted_from_pwml:") for r in release["reasons"])


def test_strict_quarantine_drops_only_the_withheld_transporter_and_caps_the_release() -> None:
    """D-114 AMENDMENT 1 s2: a transporter whose state names nothing loses the transporter, never the transport, in
    strict mode; the drop is recorded and the release is review_required (review round 2, finding a)."""
    from test_orch773_omit_per_row import _leg, _quarantine, _refusals, _surviving

    payload, context = _leg("runs_verify_2026-08-21_2057_PMC12096016")
    protein = next(p["name"] for p in payload["entities"]["proteins"] if isinstance(p, dict))
    transport = payload["processes"]["transports"][0]
    good = {"entity": protein, "entity_type": "protein", "biological_state": transport["from_biological_state"]}
    bad = {"entity": protein, "entity_type": "protein", "biological_state": "nowhere lumen"}
    transport["transporters"] = [good, bad]
    result = _quarantine(payload, context)
    assert _refusals(result) == []
    kept = [t for t in _surviving(result, "transports") if t["name"] == transport["name"]]
    assert len(kept) == 1
    assert [a.get("biological_state") for a in kept[0]["transporters"]] == [transport["from_biological_state"]]
    dropped = [e for e in result.removed_entity_report["removed_entities"] if e["bucket"] == "transporters"]
    assert [(e["name"], e["reason"], e["detail"]["actor"]["biological_state"]) for e in dropped] == [
        (protein, "transporter_withheld_from_pwml", "nowhere lumen")]
    release = result.quarantine_report["release"]
    assert release["status"] == "review_required"
    assert any(r.startswith("omitted_from_pwml:") for r in release["reasons"])


def test_text_record_lists_an_ir_withheld_transport_and_transporter_as_omitted(tmp_path) -> None:
    from t2pw.app.reaction_evidence import build_report, render_text

    payload = _payload([
        _tr("Glucose to nowhere", "Glucose", "cytosolic state", "vacuole"),
        _tr("Lactate export", "Lactate", "cytosolic state", "extracellular state",
            [{"protein": "GLUT1", "biological_state": "nowhere"}]),
    ])
    ir, _ = build_pwml_ir(deepcopy(payload), strict_db=True)
    report = build_report({"final_mapped.json": payload, "pwml_ir.json": ir}, pwml_written=True)
    omitted = [e for e in report["omitted"] if e["group"] == "ir"]
    assert [(e["kind"], e["name"]) for e in omitted] == [("transport", "Glucose to nowhere")]
    assert report["pwml_coverage"]["by_kind"]["transport"]["drawn"] == 1
    assert [a["actor"] for a in report["omitted_actors"]] == ["GLUT1"]
    text = render_text(report)
    assert "Glucose to nowhere" in text and "OMITTED ACTOR 1" in text


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
    # Review finding 1: an RNA word wins over promoter/enhancer/locus/plasmid/allele (base types these RNA)
    for name in ("promoter-associated RNA", "enhancer RNA", "plasmid-encoded sRNA", "allele-specific mRNA"):
        assert nucleotide_type_for({"name": name}) == "RNA", name
    assert nucleotide_type_for({"name": "gene mRNA"}) is None  # gene/dna unchanged: still both kinds, unsettled
