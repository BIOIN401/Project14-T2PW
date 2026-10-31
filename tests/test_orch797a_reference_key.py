"""ORCH-797a (D-120 §§ 1, 4): names equal up to format are one entity; a tRNA is a nucleic acid.

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR.** ``test_g9_*`` import only names present at base ``aefa5b30`` and
fail there on VALUES:

* the ORCH-796 chlorophyll shape with no second kind: one nucleic-acid row ``tRNA^Glu``, a reaction input and a
  location row spelled ``tRNAGlu``. Base: the quarantine's key (``trnaglu``) admits it, the IR's (``trna glu``)
  does not resolve it, and the whole pathway is refused (``ir:unresolved_entity_reference``);
* ``5-HT 3 receptor`` against the row ``5-HT3 receptor``. Base: the interaction is omitted as an unknown entity;
* a tRNA declared as a compound WITH a KEGG id and as a nucleic acid: base keeps the compound (``db_identity``);
* a tRNA declared as a nucleic acid and a protein: base renames the nucleic acid ``tRNAGlu gene``.

``test_preservation_*`` pass at base and tip. ``test_new_acceptance_*`` are an **explicitly labelled NEW
ACCEPTANCE TEST** (no base failure is claimed): the new ``name_keys.reference_key``, and the D-120 § 1 fold of two
spellings of one row (agreeing identities -> one row; conflicting -> per-row omission), new behaviour by decision.
"""

from __future__ import annotations

from helpers_prefreeze import prefrozen
from test_orch773_omit_per_row import _quarantine, _refusals
from test_strict_quarantine_contract_alignment import _base

DROP = "same_name_declared_as_another_kind:"


def _reasons(result):
    return {(a["bucket"], a["index"]): a["reason"] for a in result.quarantine_report["admissions"]}


def _trna_leg_shape():
    payload = _base()
    payload["entities"]["nucleic_acids"] = [{"name": "tRNA^Glu", "class": "nucleic_acid"}]
    payload["element_locations"]["nucleic_acid_locations"] = [
        {"nucleic_acid": "tRNA^Glu", "biological_state": "cytosol"},
        {"nucleic_acid": "tRNAGlu", "biological_state": "cytosol"}]
    payload["processes"]["reactions"][0]["inputs"].append("tRNAGlu")
    payload["processes"]["reactions"][0]["evidence"] = "tRNAGlu"  # free text: never rewritten
    return payload


def test_g9_a_reference_spelled_without_the_caret_resolves_to_the_single_row_and_the_pwml_is_written():
    result = _quarantine(prefrozen(_trna_leg_shape()), None)
    assert _refusals(result) == []  # base: ["ir:unresolved_entity_reference"]
    reaction = result.payload["processes"]["reactions"][0]
    assert "tRNA^Glu" in reaction["inputs"] and "tRNAGlu" not in reaction["inputs"]
    assert reaction["evidence"] == "tRNAGlu"
    assert result.payload["element_locations"]["nucleic_acid_locations"] == [
        {"nucleic_acid": "tRNA^Glu", "biological_state": "cytosol"}]


def _receptor_shape(reference: str):
    payload = _base()
    payload["entities"]["compounds"].append({"name": "serotonin"})
    payload["element_locations"]["compound_locations"].append({"compound": "serotonin", "biological_state": "cytosol"})
    payload["entities"]["proteins"].append(
        {"name": "5-HT3 receptor", "species": "Homo sapiens", "mapped_ids": {"uniprot": "P46098"}})
    payload["element_locations"]["protein_locations"].append(
        {"protein": "5-HT3 receptor", "biological_state": "cytosol"})
    payload["processes"]["interactions"] = [
        {"name": "serotonin activates the 5-HT3 receptor", "entity_1": "serotonin", "entity_2": reference,
         "relationship": "activation", "evidence": f"serotonin activates the {reference}"}]
    return payload


def test_g9_a_receptor_written_with_a_space_before_its_digit_is_the_same_protein():
    result = _quarantine(prefrozen(_receptor_shape("5-HT 3 receptor")), None)
    assert _reasons(result)[("interactions", 0)] == "admitted"  # base: an unknown entity, the row omitted
    assert result.payload["processes"]["interactions"][0]["entity_2"] == "5-HT3 receptor"
    assert _refusals(result) == []


def test_preservation_the_receptor_spelled_as_its_row_is_drawn():
    result = _quarantine(prefrozen(_receptor_shape("5-HT3 receptor")), None)
    assert _reasons(result)[("interactions", 0)] == "admitted"
    assert _refusals(result) == []


def test_preservation_a_reference_naming_no_row_is_omitted_per_row_never_the_pathway():
    payload = _trna_leg_shape()
    payload["processes"]["reactions"][1]["inputs"].append("tRNAAsp")  # no row of any kind
    result = _quarantine(prefrozen(payload), None)
    assert _reasons(result)[("reactions", 1)] != "admitted"
    assert result.ok


def _dropped(result):
    return {(e["bucket"], e["name"]): e["reason"] for e in result.removed_entity_report["removed_entities"]
            if e["reason"].startswith(DROP)}


def test_g9_a_charged_trna_with_a_compound_identity_keeps_its_nucleic_acid_row():
    payload = _base()
    payload["entities"]["compounds"].append({"name": "L-glutamyl-tRNA(Glu)", "kegg_id": "C02987"})
    payload["entities"]["nucleic_acids"] = [{"name": "L-glutamyl-tRNA(Glu)", "class": "nucleic_acid"}]
    payload["processes"]["reactions"][0]["outputs"].append("L-glutamyl-tRNA(Glu)")
    result = _quarantine(payload, None)
    assert _dropped(result) == {("compounds", "L-glutamyl-tRNA(Glu)"): DROP + "nucleic_acids"}  # base: compound kept
    assert [r["name"] for r in result.payload["entities"]["nucleic_acids"]] == ["L-glutamyl-tRNA(Glu)"]


def test_g9_a_trna_declared_as_a_protein_too_is_never_renamed_a_gene():
    payload = _base()
    payload["entities"]["proteins"].append({"name": "tRNAGlu", "species": "Homo sapiens"})
    payload["entities"]["nucleic_acids"] = [{"name": "tRNAGlu", "class": "nucleic_acid"}]
    payload["processes"]["reactions"][0]["inputs"].append("tRNAGlu")
    result = _quarantine(payload, None)
    assert [r["name"] for r in result.payload["entities"]["nucleic_acids"]] == ["tRNAGlu"]  # base: "tRNAGlu gene"
    assert _dropped(result) == {("proteins", "tRNAGlu"): DROP + "nucleic_acids"}
    assert _reasons(result)[("reactions", 0)] == "admitted"


def test_new_acceptance_reference_key_folds_format_and_keeps_identity():
    from t2pw.pipeline.name_keys import reference_key as k

    same = [("tRNA^Glu", "tRNAGlu", "tRNA-Glu", "trna glu"), ("5-HT 3 receptor", "5-HT3 receptor"),
            ("α-ketoglutarate", "alpha-ketoglutarate", "Α-KETOGLUTARATE"), ("Ca²⁺", "Ca2+", "Ca 2+"),
            ("IκB", "Ikappa B"), ("λ-phage", "lambda phage"), ("Glu–tRNA", "Glu-tRNA"), ("ζ-carotene", "zeta-carotene")]
    for group in same:
        assert len({k(n) for n in group}) == 1, group
    different = [("MK-4", "MK-7"), ("Ca2+", "Ca"), ("α-ketoglutarate", "β-ketoglutarate"),
                 ("Glu-tRNAGlu", "tRNAGlu"), ("OPC-8:0", "OPC-80"), ("ζ-carotene", "carotene")]
    for a, b in different:
        assert k(a) != k(b), (a, b)


def test_new_acceptance_every_reference_resolving_stage_uses_the_one_key():
    from t2pw.app import reaction_evidence
    from t2pw.pipeline import entity_admission, process_normalizer, reference_repair
    from t2pw.pipeline.name_keys import reference_key
    from t2pw.pwml import compound_resolution, ir, prefreeze_resolution

    keys = [process_normalizer._normalize, reference_repair._normalize, entity_admission._norm, ir._norm,
            prefreeze_resolution._norm, compound_resolution._norm, reaction_evidence._ir_norm]
    for name in ("tRNA^Glu", "5-HT 3 receptor", "β-D-glucose", "Ca2+", "OPC-8:0-CoA", "1,2-diacylglycerol"):
        assert {key(name) for key in keys} == {reference_key(name)}, name


def test_new_acceptance_identity_routing_reads_either_fold():
    from t2pw.pipeline.entity_identity import compound_name_block_rule
    from t2pw.pipeline.name_keys import reference_key

    assert compound_name_block_rule("coenzyme A", compound_names={reference_key("coenzyme A")}) == "compound_registry"
    assert compound_name_block_rule("coenzyme A", compound_names={"coenzyme a"}) == "compound_registry"  # map_ids fold


def _two_spellings(first_ids, second_ids):
    payload = _base()
    payload["entities"]["compounds"] += [{"name": "lipid IVA", **first_ids}, {"name": "lipid IV A", **second_ids}]
    payload["element_locations"]["compound_locations"] += [
        {"compound": "lipid IVA", "biological_state": "cytosol"}, {"compound": "lipid IV A", "biological_state": "cytosol"}]
    payload["processes"]["reactions"][0]["outputs"].append("lipid IVA")
    payload["processes"]["reactions"][1]["inputs"].append("lipid IV A")
    return payload


def test_new_acceptance_two_spellings_of_one_compound_fold_into_one_row_and_the_pwml_is_written():
    """NEW ACCEPTANCE (D-120 § 1; base wrote both rows, as two compounds of one name, and the PWML too)."""

    result = _quarantine(prefrozen(_two_spellings({"chebi_id": "CHEBI:9"}, {})), None)
    assert _refusals(result) == []  # without the fold the IR refuses the pathway (DuplicateNamedRowError: one key, two rows)
    assert [r["name"] for r in result.payload["entities"]["compounds"] if "lipid" in r["name"]] == ["lipid IVA"]
    assert result.payload["processes"]["reactions"][1]["inputs"][-1] == "lipid IVA"
    removed = {(e["bucket"], e["name"]): e["reason"] for e in result.removed_entity_report["removed_entities"]}
    assert removed[("compounds", "lipid IV A")] == "same_entity_name_form:compounds"


def test_new_acceptance_two_spellings_with_conflicting_identities_are_omitted_per_row_never_picked():
    """NEW ACCEPTANCE (a deliberate change: base exported both rows as two compounds of one name, D-120 § 1)."""

    payload = _two_spellings({"chebi_id": "CHEBI:9"}, {"chebi_id": "CHEBI:10"})
    payload["processes"]["reactions"][0]["outputs"].remove("lipid IVA")
    payload["processes"]["reactions"][1]["outputs"].append("lipid IVA")
    result = _quarantine(prefrozen(payload), None)
    assert _reasons(result)[("reactions", 0)] == "admitted"
    assert _reasons(result)[("reactions", 1)] == "duplicate_canonical_row_identity_not_proven"
    assert result.ok and _refusals(result) == []
    assert not [r for r in result.payload["entities"]["compounds"] if "lipid" in r["name"]]
