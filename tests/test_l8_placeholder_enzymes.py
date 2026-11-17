"""L8 (D-W24 items 7b, 7c): an enzyme with no database identity keeps its reaction and its name; the Unknown placeholder
protein carries no species it was not given.

7b. Case: A4 finding rank 1 (idx 5 MenD, idx 121 HS41.20 / HS41.17, idx 68 LpxL / LpxM): the paper-explicit step left the
PWML because its enzyme has no UniProt/DrugBank identity (before lane L5 the reaction went; after L5 the enzyme went). Now the
enzyme is written as the Unknown-backed placeholder complex the Stage-6 fallback ships elsewhere (idx 9 MenI), named after the
paper's enzyme, and the reaction and the run are ``review_required``.

7c. PathBank's Unknown protein (9659) was written with species Arabidopsis thaliana in every payload (20 of 26 A1-A4 payloads,
human ones included). It is now written with no species: PathWhiz binds its existing record 9659 by name and UniProt value
(``pwml_parser.store_object``: an omitted column is not part of ``find_by``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from t2pw.pipeline.strict_quarantine import quarantine_and_close

from test_l5_quarantine_keeps_reactions import ENZYME, _actor_drops, _omitted_names, _reaction, _sphingolipid
from test_pwml_l4_interactions import _export

REACTION = "ceramide synthase reaction"


def _sentinel(species: bool = True) -> Dict[str, Any]:
    row = {"name": "Unknown", "pathbank_protein_id": 9659, "pw_protein_id": 9659, "uniprot_id": "Unknown",
           "mapped_ids": {"uniprot": "Unknown", "pathbank_protein_id": 9659},
           "identity_status": "placeholder",
           "mapping_meta": {"chosen_rule": "pathbank_unknown_protein_fallback", "pathbank_protein_id": 9659,
                            "cross_species_placeholder": True, "fallback_used": True, "identity_status": "placeholder"}}
    if species:
        row.update(species="Arabidopsis thaliana", organism="Arabidopsis thaliana", species_id=4,
                   pathbank_species_id=4, taxonomy_id="3702")
    return row


def _protein_species(root, name: str) -> list:
    species = {s.findtext("id"): s.findtext("name") for s in root.iter("species") if s.findtext("id")}
    return [species.get(p.findtext("species-id") or "", p.findtext("species-id") or None)
            for p in root.iter("protein") if p.findtext("name") == name]


def test_l8_an_enzyme_without_identity_keeps_its_reaction_as_an_unknown_backed_placeholder() -> None:
    """Base (L5): the reaction stays but loses its enzyme (actor drop, protein_missing_external_identity)."""
    payload = _sphingolipid(uniprot="")
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert result.ok and _omitted_names(result) == []
    assert _actor_drops(result) == []
    reaction = _reaction(result, REACTION)
    for slot in ("enzymes", "modifiers"):
        assert [(a["entity"], a["entity_type"]) for a in reaction[slot]] == [(ENZYME, "protein_complex")]
        assert reaction[slot][0]["evidence"] == "Ceramide Synthases in Biomedical Research"
    assert reaction["review_required"] is True
    assert reaction["review_reasons"][0].startswith(f"placeholder_enzyme_without_identity: {ENZYME}:")
    (placeholder,) = [c for c in result.payload["entities"]["protein_complexes"] if c["name"] == ENZYME]
    assert [c["name"] for c in placeholder["components"]] == ["Unknown"]
    assert placeholder["components"][0]["pathbank_protein_id"] == 9659
    assert placeholder["species"] == "Homo sapiens" and placeholder["review_required"] is True
    proteins = {p["name"]: p for p in result.payload["entities"]["proteins"]}
    assert ENZYME not in proteins and "species" not in proteins["Unknown"]
    report = result.quarantine_report
    assert "placeholder_enzyme_without_identity:1" in report["review_reasons"]
    assert report["release"]["status"] == "review_required"
    (record,) = [w for w in report["enzyme_wrappers"] if w["decision"] == "D-W24 7b"]
    assert (record["protein"], record["protein_complex"], record["reactions"]) == (ENZYME, ENZYME, [REACTION])
    assert payload == _sphingolipid(uniprot="")  # the input is never mutated


def test_l8_the_placeholder_reaches_the_pwml_with_no_species_on_the_unknown_protein() -> None:
    result = quarantine_and_close(_sphingolipid(uniprot=""), strict_db=True, mode="pathwhiz")
    ir, report, build, root, qa = _export(result.payload)
    enzymes = {r["name"]: [ir_e["entity_key"] for ir_e in r["enzymes"]] for r in ir["processes"]["reactions"]}
    assert len(enzymes[REACTION]) == 1
    assert [c.findtext("name") for c in root.iter("protein-complex")] == [ENZYME]
    assert [p.findtext("id") for p in root.iter("protein") if p.findtext("name") == "Unknown"] == ["9659"]
    assert _protein_species(root, "Unknown") == [None]
    assert "Arabidopsis thaliana" not in {s.findtext("name") for s in root.iter("species")}


def test_l8_a_stage6_sentinel_loses_its_arabidopsis_species_in_the_final_payload_step() -> None:
    """The stored payloads carry the Stage-6 sentinel with Arabidopsis; the final payload step takes it off (flag on)."""
    from t2pw.service.final_payload_evidence import strip_placeholder_species

    payload = _sphingolipid()
    payload["entities"]["proteins"].append(_sentinel())
    payload["entities"]["protein_complexes"].append(
        {"name": "dihydroceramide desaturase", "species": "Homo sapiens", "generated": True,
         "generation_reason": "single_protein_pathwhiz_wrapper", "components": [{"name": "Unknown", "stoichiometry": 1}]})
    payload["entities"]["species"].append({"name": "Arabidopsis thaliana", "taxonomy_id": "3702", "pathwhiz_id": 4})
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "dihydroceramide desaturase", "entity_type": "protein_complex", "role": "catalyst"}]
    assert strip_placeholder_species(payload) == ["Arabidopsis thaliana"]
    sentinel = payload["entities"]["proteins"][-1]
    assert not {"species", "organism", "species_id", "pathbank_species_id", "taxonomy_id"} & set(sentinel)
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert result.ok and _omitted_names(result) == []
    assert "Arabidopsis thaliana" not in {s["name"] for s in result.payload["entities"]["species"]}
    _ir, _report, _build, root, _qa = _export(result.payload)
    assert _protein_species(root, "Unknown") == [None]


def test_l8_a_sentinel_given_a_species_still_writes_it() -> None:
    """Flag off (no final payload step): the stored sentinel row keeps the species it has; bytes do not move."""
    payload = _sphingolipid()
    payload["entities"]["proteins"].append(_sentinel())
    payload["entities"]["protein_complexes"].append(
        {"name": "dihydroceramide desaturase", "species": "Homo sapiens", "generated": True,
         "generation_reason": "single_protein_pathwhiz_wrapper", "components": [{"name": "Unknown", "stoichiometry": 1}]})
    payload["entities"]["species"].append({"name": "Arabidopsis thaliana", "taxonomy_id": "3702", "pathwhiz_id": 4})
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "dihydroceramide desaturase", "entity_type": "protein_complex", "role": "catalyst"}]
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    _ir, _report, _build, root, _qa = _export(result.payload)
    assert _protein_species(root, "Unknown") == ["Arabidopsis thaliana"]


def test_l8_an_enzyme_an_interaction_also_names_gets_a_complex_of_its_own_and_the_interaction_is_judged_as_before() -> None:
    """idx 5 MenD: 'DHNA inhibits MenD' named the protein too, which is why Stage 6 left MenD bare. The reaction keeps
    MenD as 'MenD complex'; the interaction still names the protein and is quarantined as at base (no new admission)."""
    payload = _sphingolipid(uniprot="")
    payload["processes"]["interactions"] = [
        {"name": "ceramide inhibits ceramide synthase", "entity_1": "ceramide", "entity_2": ENZYME,
         "relationship": "inhibition", "biological_state": "ER state", "evidence": "ceramide inhibits ceramide synthase"}]
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert result.ok
    assert _omitted_names(result) == ["ceramide inhibits ceramide synthase"]
    assert result.quarantine_report["quarantined"][0]["reason"] == "protein_missing_external_identity"
    reaction = _reaction(result, REACTION)
    assert [(a["entity"], a["entity_type"]) for a in reaction["enzymes"]] == [(f"{ENZYME} complex", "protein_complex")]
    (placeholder,) = [c for c in result.payload["entities"]["protein_complexes"] if c["name"] == f"{ENZYME} complex"]
    assert [c["pathbank_protein_id"] for c in placeholder["components"]] == [9659]


def test_l8_a_protein_that_is_also_a_substrate_is_not_made_a_placeholder() -> None:
    payload = _sphingolipid(uniprot="")
    payload["processes"]["reactions"][0]["inputs"] = ["Sphinganine", ENZYME]
    from t2pw.pipeline.strict_quarantine import placeholder_enzymes

    assert placeholder_enzymes(payload) == {}


def test_l8_research_mode_is_untouched() -> None:
    payload = _sphingolipid(uniprot="")
    result = quarantine_and_close(deepcopy(payload), strict_db=True, mode="research")
    assert result.payload == payload
