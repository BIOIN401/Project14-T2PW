"""Clean model dataclasses, provenance labels and the interaction vocabulary (no network, no pipeline)."""

import copy

import pytest

from t2pw.clean_model import UNSET, Field, Prov
from t2pw.clean_model.model import Compound, Interaction, Reaction, Ref
from t2pw.clean_model.vocab import KINDS, classify


def test_field_defaults_are_unknown_and_absent():
    f = Field()
    assert f.absent and f.prov.label == "unknown" and not f.prov.frozen
    assert f.value is UNSET
    assert copy.deepcopy(f).value is UNSET


def test_prov_slots():
    p = Prov(label="db", source="ChEBI", accession="CHEBI:1", retrieved="2026-11-10", match_type="exact label",
             quote=None, location=None, frozen=False)
    assert p.label == "db" and p.match_type == "exact label"


def test_compound_labels_and_extras():
    raw = {"name": "ATP", "chebi_id": "15422", "kegg_id": "C00002", "confidence": 1.0, "provenance": "extracted",
           "mapping_meta": {"provider": "PathBankDB"}, "chosen_rule": "r1", "odd_key": {"a": [1, 2]}}
    c = Compound.from_raw(raw)
    assert c.name.prov.label == "paper" and c.name.prov.frozen
    assert c.chebi_id.prov.label == "db" and c.chebi_id.prov.accession == "15422" and c.chebi_id.prov.frozen
    assert c.chebi_id.prov.source == "PathBankDB"
    assert c.confidence.prov.label == "inferred"
    assert c.extras == {"odd_key": {"a": [1, 2]}}
    assert c.inchikey.absent and c.formula.absent  # online-layer slots stay empty
    assert c.to_raw() == raw


def test_reaction_silent_fields_unknown_and_defaults_labelled():
    raw = {"name": "r", "inputs": ["A"], "outputs": ["B"], "enzymes": [], "evidence": "A becomes B.",
           "spontaneous": False, "biological_state": "s"}
    r = Reaction.from_raw(raw)
    for slot in (r.direction, r.reversible, r.rhea_id, r.ec_number, r.kegg_reaction_id, r.go_mf, r.step_index):
        assert slot.absent and slot.prov.label == "unknown"
    assert r.reactants[0].stoichiometry.absent  # bare strings carry no stoichiometry; none is invented
    assert r.evidence.prov.label == "paper" and r.evidence.prov.quote == "A becomes B."
    assert r.spontaneous.prov.label == "default"
    assert Reaction.from_raw({**raw, "spontaneous": True}).spontaneous.prov.label == "inferred"
    assert r.to_raw() == raw


def test_ref_bare_dict_and_entity_key_roundtrip():
    for raw in ("EntB", {"name": "EntF", "stoichiometry": 1, "mapped_ids": {"uniprot": "P1"}},
                {"entity": "TolC", "entity_type": "protein", "protein": "TolC", "evidence": "q"}, 5):
        assert Ref.from_raw(raw).to_raw() == raw
    r = Ref.from_raw({"name": "EntF", "stoichiometry": 1})
    assert r.stoichiometry.prov.label == "default" and r.entity.value == "EntF"
    assert Ref.from_raw({"name": "X", "stoichiometry": 2}).stoichiometry.prov.label == "inferred"


@pytest.mark.parametrize("word,kind,sign", [
    ("regulates", "modulation", "unsigned"),
    ("REGULATES", "modulation", "unsigned"),
    ("modulates", "modulation", "unsigned"),
    ("inhibits", "inhibition", "negative"),
    ("INHIBITS", "inhibition", "negative"),
    ("inhibition", "inhibition", "negative"),
    ("down-regulates", "inhibition", "negative"),
    ("activates", "activation", "positive"),
    ("stimulates", "activation", "positive"),
    ("binds", "binding", "unsigned"),
    ("binds to", "binding", "unsigned"),
    ("BINDS_TO", "binding", "unsigned"),
    ("interacts_with", "association", "unsigned"),
    ("competes_with", "competition", "unsigned"),
    ("degrades", "degradation", "unsigned"),
    ("phosphorylates", "phosphorylation", "unsigned"),
    ("allosteric_inhibitor_of", "inhibition", "negative"),
    ("SAME_AS", "other_unsigned", "unsigned"),
    ("substrate of", "other_unsigned", "unsigned"),       # a substrate does not catalyse
    ("provides substrate for", "other_unsigned", "unsigned"),
    ("catalyzes", "catalysis", "unsigned"),
    ("some brand new verb", "other_unsigned", "unsigned"),
    (None, "other_unsigned", "unsigned"),
])
def test_vocab(word, kind, sign):
    assert classify(word) == (kind, sign)
    assert classify(word)[0] in KINDS


def test_interaction_keeps_raw_text_and_has_sbo_slot():
    raw = {"entity_1": "A", "entity_2": "B", "name": "A BINDS B", "relationship": "BINDS", "evidence": "q"}
    i = Interaction.from_raw(raw)
    assert i.relationship.value == "BINDS" and i.sbo.absent and i.kind.absent
    assert i.to_raw() == raw


def test_compound_drugbank_and_class_flags_are_typed_and_roundtrip_exact():
    raw = {"name": "L-serine", "class": "cofactor", "compound_class": "cofactor", "compound_type": "Compound", "type": "Compound",
           "mapped_ids": {"drugbank": "DB00133", "kegg": "C00065"}, "mapping_meta": {"provider": "PathBankDB"}, "chosen_rule": "r1"}
    c = Compound.from_raw(raw)
    assert c.drugbank_mapped.value == "DB00133" and c.drugbank_mapped.prov.label == "db" and c.drugbank_mapped.prov.frozen
    assert c.drugbank_mapped.prov.source == "PathBankDB" and c.drugbank_mapped.prov.match_type == "r1"
    assert c.drugbank_id.absent  # the payload key itself stays unset
    assert (c.class_.value, c.compound_class.value, c.compound_type.value, c.type_.value) == ("cofactor", "cofactor", "Compound", "Compound")
    assert "class" not in c.extras and "compound_type" not in c.extras and "mapped_ids" in c.extras
    assert c.to_raw() == raw  # nothing added, nothing lost
    top = Compound.from_raw({"name": "cAMP", "drugbank_id": "DB02527", "mapped_ids": {"drugbank": "DB09999"}, "class": None})
    assert top.drugbank_mapped.absent and top.drugbank_id.value == "DB02527" and top.class_.value is None
    assert top.to_raw()["class"] is None
