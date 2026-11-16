"""L4 (D-W24 items 1, 4, 6; D-W26): regulation and interactions in the PWML.

1. An activator / inhibitor modifier is never the step's enzyme: it becomes an Activation /
   Inhibition interaction on the step (right element: the step's enzyme, else its substrate).
2. A signed interaction with a valid endpoint reaches the PWML; Stage 6 wraps the protein
   endpoints of every exportable type.
3. PathWhiz t2pw/05 stores Binding, Regulation, Catalysis, Modification and Unspecified too,
   each with its own edge template; a repeat of the same type and elements is written once.
4. Evidence travels with every interaction; a row without it is flagged, never dropped.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helpers_prefreeze import prefrozen  # noqa: E402
from t2pw.mapping.map_ids import MappingCache, _rewrite_reaction_protein_enzymes_to_complexes  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir  # noqa: E402
from t2pw.pwml.qa import run_pwml_qa  # noqa: E402
from t2pw.pwml.validate import discover_structure_signature, repair_tree, validate_generated_tree  # noqa: E402
from t2pw.pwml.writer import DeterministicPwmlBuilder  # noqa: E402

REFERENCE = ROOT / "reference" / "PW000001.pwml"
STATE = "cytosol"


def _complex(name: str, protein: str) -> dict:
    return {"name": name, "species": "Escherichia coli", "components": [protein], "generated": True,
            "generation_reason": "single_protein_pathwhiz_wrapper"}


def _protein(name: str, uniprot: str) -> dict:
    return {"name": name, "species": "Escherichia coli", "organism": "Escherichia coli", "species_id": 3,
            "pathbank_protein_id": 0, "mapped_ids": {"uniprot": uniprot}}


def _base() -> dict:
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "pathwhiz_id": 3}],
            "subcellular_locations": [{"name": STATE, "pathwhiz_id": 2}],
            "compounds": [
                {"name": "protoporphyrin IX", "pathbank_compound_id": 1001, "mapped_ids": {"chebi": "CHEBI:15430"}},
                {"name": "Mg-protoporphyrin IX", "pathbank_compound_id": 1002, "mapped_ids": {"chebi": "CHEBI:50152"}},
                {"name": "OSB-CoA", "pathbank_compound_id": 1003, "mapped_ids": {"chebi": "CHEBI:57364"}},
                {"name": "DHNA-CoA", "pathbank_compound_id": 1004, "mapped_ids": {"chebi": "CHEBI:58897"}},
            ],
            "proteins": [
                _protein("ChlH", "P00001"), _protein("GUN4", "P00002"), _protein("MenD", "P00003"),
                _protein("AraC", "P0A9E0"),
            ],
            "protein_complexes": [
                _complex("Mg-chelatase complex", "ChlH"), _complex("GUN4 complex", "GUN4"),
                _complex("MenD complex", "MenD"), _complex("AraC complex", "AraC"),
            ],
            "nucleic_acids": [{"name": "araBAD promoter", "class": "nucleic_acid"},
                              {"name": "Pc promoter", "class": "nucleic_acid"}],
            "element_collections": [{"name": "ara regulatory elements", "class": "element_collection"}],
        },
        "biological_states": [
            {"name": STATE, "species": "Escherichia coli", "subcellular_location": STATE, "compartment_canonical": STATE},
        ],
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _modifier_payload() -> dict:
    payload = _base()
    payload["processes"]["reactions"] = [
        {
            "name": "Mg-chelatase reaction",
            "inputs": [{"name": "protoporphyrin IX"}],
            "outputs": [{"name": "Mg-protoporphyrin IX"}],
            "enzymes": [{"entity": "Mg-chelatase complex", "entity_type": "protein_complex", "role": "catalyst"}],
            "modifiers": [
                {"entity": "Mg-chelatase complex", "entity_type": "protein_complex", "role": "catalyst",
                 "evidence": "Mg-chelatase inserts Mg2+"},
                {"entity": "GUN4 complex", "entity_type": "protein_complex", "role": "activator",
                 "evidence": "GUN4 stabilizes the Mg-chelatase complex and stimulates its activity"},
            ],
            "biological_state": STATE,
        },
        {
            # The idx 9 shape: an inhibitor named on a step the quote does not mention.
            "name": "MenB-catalyzed conversion of OSB-CoA to DHNA-CoA",
            "inputs": [{"name": "OSB-CoA"}],
            "outputs": [{"name": "DHNA-CoA"}],
            "enzymes": [],
            "modifiers": [
                {"entity": "MenD complex", "entity_type": "protein_complex", "role": "inhibitor",
                 "evidence": "kcatD [MenD] [B] (1+[DHNA]/Ki)"},
            ],
            "biological_state": STATE,
        },
    ]
    return payload


def _export(payload: dict):
    ir, report = build_pwml_ir(prefrozen(payload), strict_db=True)
    assert report["errors"] == [], report["errors"]
    validation = validate_pwml_ir(ir)
    assert validation["ok"], validation["errors"]
    signature = discover_structure_signature(REFERENCE)
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF", ref=str(REFERENCE))
    build = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args).build()
    repaired = repair_tree(etree.ElementTree(build.root), signature)
    assert validate_generated_tree(repaired, signature)["ok"]
    xml = etree.tostring(repaired.getroot(), encoding="utf-8", xml_declaration=True)
    return ir, report, build, repaired.getroot(), run_pwml_qa(xml)


def _all_interactions(ir: dict) -> list:
    return [*ir["processes"]["interactions"], *ir["processes"].get("modifier_interactions", [])]


def _names(ir: dict) -> dict:
    return {e["key"]: e["name"] for rows in ir["entities"].values() for e in rows}


def _edge_templates(root) -> dict:
    """interaction-type -> sorted template ids of the edges of its visualization."""
    edges = {e.findtext("id"): e for e in root.iter("edge")}
    types = {i.findtext("id"): i.findtext("interaction-type") for i in root.iter("interaction")}
    out: dict = {}
    for viz in root.iter("interaction-visualization"):
        ids = [eid.text for eid in viz.iter("edge-id")]
        out.setdefault(types[viz.findtext("interaction-id")], []).extend(
            int(edges[i].findtext("visualization-template-id")) for i in ids)
    return out


# ── 1. activators / inhibitors are interactions on the step, not enzymes ─────────────


def test_activator_and_inhibitor_modifiers_are_interactions_on_the_step_not_enzymes() -> None:
    ir, report, _build, root, qa = _export(_modifier_payload())
    names = _names(ir)
    rx = {r["name"]: [names[e["entity_key"]] for e in r["enzymes"]] for r in ir["processes"]["reactions"]}
    assert rx["Mg-chelatase reaction"] == ["Mg-chelatase complex"]  # GUN4 is not an enzyme of the step
    assert rx["MenB-catalyzed conversion of OSB-CoA to DHNA-CoA"] == []  # MenD is not a co-catalyst

    mods = {names[i["left"]["entity_key"]]: i for i in ir["processes"]["modifier_interactions"]}
    gun4, mend = mods["GUN4 complex"], mods["MenD complex"]
    assert gun4["interaction_type"] == "Activation" and gun4["export_status"] == "exportable"
    assert names[gun4["right"]["entity_key"]] == "Mg-chelatase complex" and gun4["step_anchor"] == "enzyme"
    assert gun4["on_reaction"]["name"] == "Mg-chelatase reaction"
    assert gun4["evidence"].startswith("GUN4 stabilizes") and "review_required" not in gun4
    # No enzyme on the step: the step's substrate stands for it.
    assert mend["interaction_type"] == "Inhibition" and mend["export_status"] == "exportable"
    assert names[mend["right"]["entity_key"]] == "OSB-CoA" and mend["step_anchor"] == "substrate"
    # The quote names MenD but nothing of the MenB step: flagged for review, still drawn (D-W24 6).
    assert mend["review_required"] is True and mend["review_reasons"] == ["evidence_does_not_name_step"]
    assert any(w["code"] == "interaction_review_required" for w in report["warnings"])
    # The payload's own interaction list still lines up one-to-one (app.reaction_evidence).
    assert ir["processes"]["interactions"] == []

    assert sorted(i.findtext("interaction-type") for i in root.iter("interaction")) == ["Activation", "Inhibition"]
    assert _edge_templates(root) == {"Activation": [149, 149], "Inhibition": [148, 148]}
    assert qa["ok"], qa["errors"]


def test_a_modifier_without_evidence_is_flagged_not_dropped() -> None:
    payload = _modifier_payload()
    payload["processes"]["reactions"][0]["modifiers"][1]["evidence"] = ""
    ir, _report, _build, root, qa = _export(payload)
    gun4 = next(i for i in ir["processes"]["modifier_interactions"] if i["modifier_role"] == "activator")
    assert gun4["export_status"] == "exportable"
    assert gun4["review_required"] is True and gun4["review_reasons"][0] == "no_evidence"
    assert len(list(root.iter("interaction"))) == 2 and qa["ok"]


# ── 3. the five new PathWhiz types, their edge templates, and merge-once ────────────


def _typed_payload() -> dict:
    payload = _base()
    rows = [
        ("AraC binds araBAD", "AraC complex", "binds", "araBAD promoter"),
        ("AraC regulates Pc", "AraC complex", "regulates", "Pc promoter"),
        ("AraC catalyzes on elements", "AraC complex", "catalyzes", "ara regulatory elements"),
        ("GUN4 modifies ChlH", "GUN4 complex", "modifies", "Mg-chelatase complex"),
        ("AraC interacts with GUN4", "AraC complex", "interacts with", "GUN4 complex"),
        ("MenD competes with AraC", "MenD complex", "competes_with", "AraC complex"),
        ("AraC activates araBAD", "AraC complex", "activates", "araBAD promoter"),
        ("AraC activation of araBAD", "AraC complex", "activation", "araBAD promoter"),  # same type + elements
        ("AraC is part of elements", "AraC complex", "component_of", "ara regulatory elements"),
    ]
    payload["processes"]["interactions"] = [
        {"name": n, "entity_1": a, "relationship": w, "entity_2": b, "biological_state": STATE, "evidence": f"quote: {n}"}
        for n, a, w, b in rows
    ]
    return payload


def test_new_pathwhiz_types_are_written_with_their_edge_templates() -> None:
    ir, _report, build, root, qa = _export(_typed_payload())
    assert qa["ok"], qa["errors"]
    written = sorted(i.findtext("interaction-type") for i in root.iter("interaction"))
    assert written == ["Activation", "Binding", "Catalysis", "Modification", "Regulation", "Unspecified", "Unspecified"]
    templates = _edge_templates(root)
    assert templates == {
        "Activation": [149, 149], "Binding": [195, 195], "Regulation": [219, 219], "Catalysis": [5, 5],
        "Modification": [218, 218], "Unspecified": [195, 195, 195, 195],
    }
    # Binding / Unspecified are undirected: no arrow at either end.
    edges = {e.findtext("id"): e for e in root.iter("edge")}
    types = {i.findtext("id"): i.findtext("interaction-type") for i in root.iter("interaction")}
    for viz in root.iter("interaction-visualization"):
        if types[viz.findtext("interaction-id")] in ("Binding", "Unspecified"):
            for eid in viz.iter("edge-id"):
                opts = edges[eid.text].findtext("options") or ""
                assert '"start_arrow": true' not in opts and '"start_flat_arrow": true' not in opts
    # Every row keeps its evidence; component_of has no PathWhiz type and stays withheld with its word.
    by_name = {r["name"]: r for r in ir["processes"]["interactions"]}
    assert all(r["evidence"].startswith("quote: ") for r in by_name.values())
    assert by_name["AraC is part of elements"]["withheld"]["reason"] == "untyped_interaction"
    # The second Activation between the same two elements is written once, and says so.
    dup = by_name["AraC activation of araBAD"]
    assert dup["export_status"] == "withheld" and dup["withheld"]["reason"] == "duplicate_interaction"
    assert dup["withheld"]["duplicate_of"] == by_name["AraC activates araBAD"]["key"]
    assert {w["name"] for w in build.withheld_interactions} == {"AraC is part of elements", "AraC activation of araBAD"}


def test_qa_accepts_every_stored_type_and_refuses_others() -> None:
    _ir, _report, _build, root, _qa = _export(_typed_payload())
    node = next(i for i in root.iter("interaction") if i.findtext("interaction-type") == "Binding")
    node.find("interaction-type").text = "binds"
    qa = run_pwml_qa(etree.tostring(root))
    assert [e for e in qa["errors"] if "interaction-type 'binds'" in e]
    assert not [e for e in qa["errors"] if "'Binding'" in e or "'Regulation'" in e]


def test_an_interaction_without_evidence_is_flagged_not_dropped() -> None:
    payload = _typed_payload()
    payload["processes"]["interactions"][0]["evidence"] = ""
    ir, _report, _build, root, _qa = _export(payload)
    row = ir["processes"]["interactions"][0]
    assert row["export_status"] == "exportable" and row["review_required"] is True
    assert row["review_reasons"] == ["no_evidence"]
    assert len(list(root.iter("interaction"))) == 7


# ── 2. Stage 6 wraps protein endpoints of every exportable type ──────────────────


def test_stage6_wraps_protein_endpoints_of_every_exportable_type() -> None:
    payload = _base()
    payload["entities"]["proteins"] += [_protein("EntB", "P0ADI4"), _protein("EntE", "P10378")]
    payload["entities"]["compounds"].append(
        {"name": "2,3-dihydroxybenzoic acid", "pathbank_compound_id": 1005, "mapped_ids": {"chebi": "CHEBI:36654"}})
    payload["processes"]["interactions"] = [
        {"name": "DHB inhibits EntB", "entity_1": "2,3-dihydroxybenzoic acid", "relationship": "inhibits",
         "entity_2": "EntB", "biological_state": STATE, "evidence": "2,3-DHB is a competitive inhibitor of EntB"},
        {"name": "DHB binds EntE", "entity_1": "2,3-dihydroxybenzoic acid", "relationship": "binds to",
         "entity_2": "EntE", "biological_state": STATE, "evidence": "2,3-DHB bound to EntE"},
        {"name": "DHB part of EntE", "entity_1": "2,3-dihydroxybenzoic acid", "relationship": "component_of",
         "entity_2": "EntE", "biological_state": STATE, "evidence": "q"},
    ]
    before = deepcopy(payload["processes"]["interactions"][2])
    _rewrite_reaction_protein_enzymes_to_complexes(
        payload, db=None, cache=MappingCache(Path("unused.json"), enabled=False), global_organism="Escherichia coli")
    rows = payload["processes"]["interactions"]
    assert rows[0]["entity_2"] == "EntB complex"
    assert rows[1]["entity_2"] == "EntE complex"
    assert rows[2] == before  # no PathWhiz type: left as written

    ir, _report, _build, root, qa = _export(payload)
    assert qa["ok"], qa["errors"]
    assert sorted(i.findtext("interaction-type") for i in root.iter("interaction")) == ["Binding", "Inhibition"]
    assert [r["export_status"] for r in ir["processes"]["interactions"]] == ["exportable", "exportable", "withheld"]


def test_an_endpoint_named_exactly_like_a_complex_is_drawn_as_that_complex_not_a_bare_protein() -> None:
    """idx 6: the endpoint is a complex's own name and also a synonym of a protein; the
    resolver used to pick the protein, which can never be drawn, so the Activation vanished."""
    payload = _base()
    payload["entities"]["proteins"].append(dict(_protein("Celsr1", "Q9NYQ6"), synonyms=["Cadherin receptor 1"]))
    payload["entities"]["protein_complexes"].append(_complex("Cadherin receptor 1", "Celsr1"))
    payload["processes"]["interactions"] = [
        {"name": "Celsr1 activates araBAD", "entity_1": "Cadherin receptor 1", "relationship": "activates",
         "entity_2": "araBAD promoter", "biological_state": STATE, "evidence": "Celsr1 stimulates it"},
    ]
    ir, _report, _build, root, qa = _export(payload)
    row = ir["processes"]["interactions"][0]
    assert row["export_status"] == "exportable"
    assert _names(ir)[row["left"]["entity_key"]] == "Cadherin receptor 1" and row["left"]["entity_type"] == "protein_complex"
    assert [i.findtext("interaction-type") for i in root.iter("interaction")] == ["Activation"] and qa["ok"]


def test_stage6_reuses_a_one_protein_wrapper_whose_component_is_the_same_protein_by_identity() -> None:
    """idx 7: "GluTR1 complex" wraps "HEMA1" (UniProt P42804, the protein GluTR1's own); the
    interaction pass refused it as another complex and left GluTR1 bare."""
    payload = _base()
    payload["entities"]["proteins"] += [_protein("FLU", "Q940U6"), _protein("GluTR1", "P42804")]
    payload["entities"]["protein_complexes"].append(
        dict(_complex("GluTR1 complex", "x"), components=[{"name": "HEMA1", "mapped_ids": {"uniprot": "P42804"}}]))
    payload["processes"]["interactions"] = [
        {"name": "FLU inhibits GluTR1", "entity_1": "FLU", "relationship": "inhibits", "entity_2": "GluTR1",
         "biological_state": STATE, "evidence": "FLU negatively regulates GluTR"},
    ]
    _rewrite_reaction_protein_enzymes_to_complexes(
        payload, db=None, cache=MappingCache(Path("unused.json"), enabled=False), global_organism="Escherichia coli")
    row = payload["processes"]["interactions"][0]
    assert (row["entity_1"], row["entity_2"]) == ("FLU complex", "GluTR1 complex")
    assert [c["name"] for c in payload["entities"]["protein_complexes"]].count("GluTR1 complex") == 1


def test_an_enzymes_list_entry_with_an_inhibitor_role_is_not_drawn_as_an_enzyme() -> None:
    """PathWhiz's reaction_enzymes has no inhibitor column: the writer's ``<inhibitor>`` flag
    is not imported, so an inhibitor kept as an enzyme reads as a catalyst there."""
    payload = _modifier_payload()
    rx = payload["processes"]["reactions"][1]
    rx["enzymes"] = [rx.pop("modifiers")[0]]
    ir, _report, _build, root, qa = _export(payload)
    assert ir["processes"]["reactions"][1]["enzymes"] == []
    assert [i["interaction_type"] for i in ir["processes"]["modifier_interactions"]] == ["Activation", "Inhibition"]
    assert root.find(".//inhibitor") is None and qa["ok"]


def test_the_same_interaction_in_another_state_is_still_drawn_there() -> None:
    """idx 151: PathWhiz maps a repeat onto the existing interaction and keeps its drawing, so
    only a repeat in the same state (same nodes, same line) is written once."""
    payload = _typed_payload()
    payload["entities"]["subcellular_locations"].append({"name": "membrane", "pathwhiz_id": 5})
    payload["biological_states"].append(
        {"name": "membrane", "species": "Escherichia coli", "subcellular_location": "membrane", "compartment_canonical": "membrane"})
    payload["processes"]["interactions"][7]["biological_state"] = "membrane"
    ir, _report, _build, root, qa = _export(payload)
    by_name = {r["name"]: r for r in ir["processes"]["interactions"]}
    assert by_name["AraC activation of araBAD"]["export_status"] == "exportable"
    assert [i.findtext("interaction-type") for i in root.iter("interaction")].count("Activation") == 2 and qa["ok"]


def test_a_row_with_no_relationship_word_is_drawn_unspecified_only_with_evidence() -> None:
    """L4 ruling (orchestrator, 2026-11-15): no stated word -> Unspecified, drawn only with paper
    evidence and flagged relationship_type_not_stated; with no evidence it stays withheld, listed."""
    payload = _base()
    payload["processes"]["interactions"] = [
        {"name": "AraC and araBAD", "entity_1": "AraC complex", "entity_2": "araBAD promoter",
         "biological_state": STATE, "evidence": "AraC acts at araBAD"},
        {"name": "AraC and Pc", "entity_1": "AraC complex", "entity_2": "Pc promoter", "biological_state": STATE},
    ]
    ir, _report, build, root, qa = _export(payload)
    stated_by_evidence, bare = ir["processes"]["interactions"]
    assert stated_by_evidence["export_status"] == "exportable" and stated_by_evidence["interaction_type"] == "Unspecified"
    assert stated_by_evidence["review_reasons"] == ["relationship_type_not_stated"]
    assert bare["export_status"] == "withheld" and bare["withheld"]["reason"] == "relationship_type_not_stated"
    assert bare["review_reasons"] == ["no_evidence"]
    assert [w["name"] for w in build.withheld_interactions] == ["AraC and Pc"]
    assert [i.findtext("interaction-type") for i in root.iter("interaction")] == ["Unspecified"] and qa["ok"]


# ── review fixes F1, F2, F4 ───────────────────────────────────────────────────


def test_the_required_contract_accepts_a_drawn_modifier_interaction() -> None:
    """F1: validate_required_pwml_contract collected process keys from PROCESS_BUCKETS only, so a
    drawn modifier interaction's visualization was a false visualization_missing_process."""
    from t2pw.pwml.ir import validate_required_pwml_contract

    ir, _report = build_pwml_ir(prefrozen(_modifier_payload()), strict_db=True)
    assert ir["processes"]["modifier_interactions"]
    codes = {e["code"] for e in validate_required_pwml_contract(ir)["errors"]}
    assert "visualization_missing_process" not in codes


def test_the_text_record_glosses_the_new_withheld_reasons_truthfully() -> None:
    """F2: no gloss may still say Activation/Inhibition are the only stored types."""
    from t2pw.app.reaction_evidence import _pwml_verdict

    for reason in ("untyped_interaction", "duplicate_interaction", "relationship_type_not_stated",
                   "same_element_both_sides", "step_has_no_element_to_anchor"):
        text = _pwml_verdict({"export_status": "withheld", "withheld": {"reason": reason}})
        assert text.startswith(f"withheld -- {reason}: ") and len(text) > len(reason) + 20
        assert "the only interaction types" not in text
    assert "Binding" in _pwml_verdict({"export_status": "withheld", "withheld": {"reason": "untyped_interaction"}})


def _atp_payload(inputs: list, outputs: list) -> dict:
    payload = _modifier_payload()
    payload["entities"]["compounds"].append(
        {"name": "ATP", "pathbank_compound_id": 1006, "kegg_id": "C00002", "mapped_ids": {"chebi": "CHEBI:30616", "kegg": "C00002"}})
    payload["entities"]["compounds"].append(
        {"name": "ADP", "pathbank_compound_id": 1007, "kegg_id": "C00008", "mapped_ids": {"chebi": "CHEBI:16761", "kegg": "C00008"}})
    rx = payload["processes"]["reactions"][1]
    rx["inputs"], rx["outputs"] = [{"name": n} for n in inputs], [{"name": n} for n in outputs]
    return payload


def test_with_no_enzyme_the_step_is_anchored_on_a_non_currency_participant_before_a_currency_one() -> None:
    """F4: GTP/ATP stood for the step in 7 of 17 modifier rows; currency is the fallback only."""
    for inputs, outputs, expected, side in ((["ATP", "OSB-CoA"], ["DHNA-CoA"], "OSB-CoA", "substrate"),
                                            (["ATP"], ["DHNA-CoA"], "DHNA-CoA", "product"),
                                            (["ATP"], ["ADP"], "Adenosine triphosphate", "substrate")):  # only currency: used (pre-freeze names ATP)
        ir, _report, _build, _root, qa = _export(_atp_payload(inputs, outputs))
        mend = next(i for i in ir["processes"]["modifier_interactions"] if i["modifier_role"] == "inhibitor")
        assert (_names(ir)[mend["right"]["entity_key"]], mend["step_anchor"]) == (expected, side) and qa["ok"]
