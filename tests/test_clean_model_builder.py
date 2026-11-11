"""Builder and round trip over a synthetic payload that has every object type, plus tracked real payloads."""

import copy
import glob
import json
import os

import pytest

from t2pw.clean_model import build_from_payload, build_from_run, summary, to_payload

PROV = [{"stage": "paper_extraction", "origin": "paper_stated"}]


def _payload():
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "taxonomy_id": "562", "provenance_lineage": PROV}],
            "subcellular_locations": [{"name": "cytoplasm", "canonical_id": "cytosol", "evidence": "in the cytoplasm"},
                                      {"name": "periplasm", "mapping_meta": {"resolution": {"status": "x"}}}],
            "compounds": [
                {"name": "chorismate", "class": "compound", "provenance": "extracted", "kegg_id": "C00251",
                 "chebi_id": "29748", "synonyms": ["CHA"], "mapping_meta": {"provider": "PathBankDB", "chosen_rule": "r"},
                 "enrichment": {"x": 1}, "element_states": [1]},
                {"name": "isochorismate", "provenance": "extracted"},
            ],
            "nucleic_acids": [{"name": "araO2", "provenance": "extracted"}],
            "proteins": [
                {"name": "EntC", "provenance": "extracted", "uniprot": "P0AEJ2", "gene_name": "entC",
                 "synonyms": ["IC synthase"], "organism": "Escherichia coli", "taxonomy_id": "562",
                 "mapping_meta": {"provider": "PathBankDB", "chosen_rule": "exact", "identity_status": "verified",
                                  "resolution": {"status": "ok"}}},
                {"name": "Unknown", "identity_status": "placeholder", "uniprot_id": "Unknown",
                 "pathbank_protein_id": 9659, "mapped_ids": {"uniprot": "Unknown"}},
            ],
            "protein_complexes": [
                {"name": "EntC dimer", "components": ["EntC", {"name": "EntC", "stoichiometry": 2}],
                 "generated": False, "mapped_ids": {"keep": "me"}},
                {"name": "mystery complex", "components": ["Unknown"], "generated": True},
                {"name": "dict unknown", "components": [{"name": "Whatever", "pathbank_protein_id": 9659,
                                                         "stoichiometry": 1}]},
            ],
            "cell_types": [{"name": "epithelial"}],
        },
        "biological_states": [
            {"name": "E. coli cytoplasm", "species": "Escherichia coli", "subcellular_location": "cytoplasm"},
            {"name": "E. coli periplasm", "species": "Escherichia coli", "subcellular_location": "periplasm"},
        ],
        "element_locations": {
            "compound_locations": [{"compound": "chorismate", "biological_state": "E. coli cytoplasm", "evidence": "q"}],
            "protein_locations": [{"protein": "EntC", "biological_state": "E. coli cytoplasm"}],
            "nucleic_acid_locations": [],
        },
        "processes": {
            "reactions": [
                {"name": "EntC reaction", "inputs": ["chorismate"], "outputs": ["isochorismate"],
                 "enzymes": [{"entity": "EntC", "entity_type": "protein", "role": "catalyst", "evidence": "EntC acts"},
                             {"entity": "Unknown", "entity_type": "protein", "role": "catalyst"}],
                 "modifiers": [], "evidence": "EntC converts chorismate to isochorismate.",
                 "biological_state": "E. coli cytoplasm", "spontaneous": False, "locked_reaction_id": "rxn_lock_001"},
                {"name": "second", "inputs": ["isochorismate"], "outputs": ["chorismate"], "enzymes": [],
                 "evidence": "back again", "biological_state": "nowhere"},
            ],
            "transports": [
                {"name": "export", "cargo": "chorismate", "from_biological_state": "E. coli cytoplasm",
                 "to_biological_state": "E. coli periplasm",
                 "transporters": [{"entity": "EntC", "entity_type": "protein", "protein": "EntC"}, "TolC"],
                 "elements_with_states": [{"element": "chorismate", "side": "left"}], "evidence": "exported"},
            ],
            "interactions": [
                {"entity_1": "EntC", "entity_2": "chorismate", "name": "i1", "relationship": "inhibits", "evidence": "q1"},
                {"entity_1": "EntC", "entity_2": "araO2", "name": "i2", "relationship": "regulates", "evidence": "q2"},
                {"entity_1": "EntC", "entity_2": "ghost", "name": "i3", "relationship": "binds"},
                {"entity_1": "EntC", "entity_2": "araO2", "name": "i4", "relationship": "ACTIVATES", "evidence": "q4"},
            ],
        },
        "quarantined_proteins": [{"name": "Fur", "reason": "not_referenced_by_any_surviving_process",
                                  "original_row": {"name": "Fur"}}],
        "quarantined_process_claims": [{"actor": "channel", "reason": "role"}],
        "entity_admission_report": {"removed": [{"entity": "X", "reason": "r"}], "demoted": []},
        "reaction_order": ["second", "EntC reaction", "EntC reaction"],
        "source_text_index": {"length": 3},
        "brand_new_top_level_key": {"nested": [1, {"a": None}]},
    }


def test_roundtrip_exact_and_input_not_mutated():
    p = _payload()
    before = copy.deepcopy(p)
    m = build_from_payload(p)
    assert p == before  # building never mutates the input
    assert to_payload(m) == before
    # mutating the model does not reach back into the input
    m.compounds[0].synonyms.value.append("zzz")
    assert p == before


def test_roundtrip_empty_and_partial_payloads():
    for p in ({}, {"entities": {}}, {"entities": {"compounds": []}, "processes": {"reactions": []}},
              {"entities": {"compounds": "not a list"}}, {"entities": {"proteins": [{"name": "A"}]}}):
        assert to_payload(build_from_payload(p)) == p


def test_every_object_type_present_and_extras_keep_the_rest():
    m = build_from_payload(_payload())
    assert (len(m.species), len(m.locations), len(m.compounds), len(m.nucleic_acids), len(m.proteins),
            len(m.complexes), len(m.states), len(m.reactions), len(m.transports), len(m.interactions)) == (1, 2, 2, 1, 2, 3, 2, 2, 1, 4)
    assert sum(len(v) for v in m.placements.values()) == 2
    assert m.extras["entities"]["cell_types"] == [{"name": "epithelial"}]
    assert m.extras["top"]["brand_new_top_level_key"] == {"nested": [1, {"a": None}]}
    assert m.compounds[0].extras["enrichment"] == {"x": 1}
    assert m.complexes[0].extras["mapped_ids"] == {"keep": "me"}  # a complex keeps its own ids only


def test_unknown_sentinel_is_a_flagged_unknown_member_not_an_entity():
    m = build_from_payload(_payload())
    assert "unknown_sentinel" in m.proteins[1].flags and not m.proteins[0].flags
    c1, c2, c3 = m.complexes[1], m.complexes[2], m.complexes[0]
    for c in (c1, c2):
        assert c.members[0].unknown and c.members[0].entity.prov.label == "unknown"
        assert "all_members_unknown" in c.flags and "unknown_member" in c.flags
    assert not any(x.unknown for x in c3.members) and not c3.flags
    assert m.reactions[0].catalysts[1].unknown and not m.reactions[0].catalysts[0].unknown
    s = summary(m)
    assert (s.complexes_with_unknown_member, s.complexes_all_unknown) == (2, 2)
    assert to_payload(m)["entities"]["protein_complexes"][1]["components"] == ["Unknown"]  # round trip keeps the text


def test_interactions_kind_sign_declared_and_raw_text():
    m = build_from_payload(_payload())
    got = [(i.relationship.value, i.kind.value, i.sign.value, i.endpoints_declared) for i in m.interactions]
    assert got == [("inhibits", "inhibition", "negative", True), ("regulates", "modulation", "unsigned", True),
                   ("binds", "binding", "unsigned", False), ("ACTIVATES", "activation", "positive", True)]
    assert m.interactions[2].flags == ["undeclared_endpoint"]
    assert m.interactions[0].kind.prov.label == "inferred" and not m.interactions[0].kind.prov.frozen
    assert m.interactions[0].evidence.prov.quote == "q1"
    s = summary(m)
    assert s.interactions_by_sign == {"negative": 1, "unsigned": 2, "positive": 1}
    assert (s.interactions_with_quote, s.reactions_with_quote, s.transports_with_quote) == (3, 2, 1)


def test_reaction_provenance_locations_and_step_index():
    m = build_from_payload(_payload())
    r = m.reactions[0]
    assert r.direction.absent and r.reactants[0].stoichiometry.absent
    assert r.reactants[0].location.value == "cytoplasm" and r.reactants[0].location.prov.label == "inferred"
    assert m.reactions[1].reactants[0].location.absent  # unknown state: nothing invented
    assert m.reactions[1].step_index.value == 0  # unique name in reaction_order
    assert r.step_index.absent  # name listed twice: ambiguous, left unknown
    t = m.transports[0]
    assert (t.from_location.value, t.to_location.value) == ("cytoplasm", "periplasm")
    assert m.transports[0].transporters[1].bare and m.transports[0].transporters[1].entity.value == "TolC"


def test_removed_and_mapping_records():
    m = build_from_payload(_payload())
    got = {(x.kind, x.name, x.reason) for x in m.removed}
    assert ("protein", "Fur", "not_referenced_by_any_surviving_process") in got
    assert ("process_claim", "channel", "role") in got
    assert ("removed", "X", "r") in got
    by = {(r.kind, r.name): r for r in m.mapping_records}
    assert by[("protein", "EntC")].rule == "exact" and by[("protein", "EntC")].flag == "verified"
    assert by[("compound", "chorismate")].rule == "r"
    assert "quarantined_proteins" in m.extras["top"]  # the view does not remove the payload key


def test_pathway_meta_from_run_folder(tmp_path):
    run = tmp_path / "legs" / "2026-10-21_0709" / "papers" / "PMC12096016" / "strict"
    run.mkdir(parents=True)
    (run / "final_mapped.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (run.parent / "01_source_text.txt").write_text(
        "507 prosci Protein Sci Wiley PMC12096016 12096016 12096016 40400396 10.1002/pro.70160 The title and authors",
        encoding="utf-8")
    (run.parent / "00_PAPER.txt").write_text("PAPER\ntitle      : (unknown)\nyear       : (unknown)\n", encoding="utf-8")
    m = build_from_run(str(run))
    pw = m.pathway
    assert pw.pmcid.value == "PMC12096016" and pw.pmid.value == "40400396" and pw.doi.value == "10.1002/pro.70160"
    assert pw.pmid.prov.label == "inferred" and pw.pmid.prov.frozen and pw.pmid.prov.source == "PMC text header"  # token guess
    assert pw.pmcid.prov.label == "paper" and pw.doi.prov.label == "paper"
    assert pw.title.absent and pw.title.prov.label == "unknown"  # not guessed from the text
    assert pw.organism.value == "Escherichia coli" and pw.taxon_id.value == "562"
    assert pw.mode.value == "strict" and pw.run_date.value == "2026-10-21_0709"
    assert to_payload(m) == _payload()


def test_pmid_from_structured_request_stays_paper(tmp_path):
    run = tmp_path / "PMC12096016" / "strict"
    run.mkdir(parents=True)
    (run / "final_mapped.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (run.parent / "request.json").write_text(json.dumps({"pmids": [40400396]}), encoding="utf-8")
    (run.parent / "01_source_text.txt").write_text("PMC12096016 12096016 11111111 10.1002/pro.70160 x", encoding="utf-8")
    m = build_from_run(str(run))
    assert m.pathway.pmid.value == "40400396" and m.pathway.pmid.prov.label == "paper"  # header token never overrides it
    assert to_payload(m) == _payload()


def test_only_pmcid_known(tmp_path):
    run = tmp_path / "PMC999" / "strict"
    run.mkdir(parents=True)
    (run / "final_mapped.json").write_text("{}", encoding="utf-8")
    pw = build_from_run(str(run)).pathway
    assert pw.pmcid.value == "PMC999" and pw.pmid.absent and pw.title.absent


def test_tracked_real_payloads_roundtrip():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    files = sorted(glob.glob(os.path.join(root, "runs", "**", "final_mapped.json"), recursive=True))[:3]
    if not files:
        pytest.skip("no tracked runs/**/final_mapped.json")
    for f in files:
        with open(f, encoding="utf-8") as fh:
            p = json.load(fh)
        assert to_payload(build_from_payload(p)) == p
