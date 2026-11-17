"""L5 (D-W24 item 3): the export quarantine never deletes a correct reaction for its enzyme.

Case: run idx 116 (Homo sapiens sphingolipid, PMC12376012 strict). ``sphinganine -> dihydroceramide``
(ceramide synthase) is in the frozen payload, and the quarantine omitted it whole with
``reaction_enzyme_must_be_protein_complex``: its enzyme was still a bare protein. Now the bare protein takes
the Stage-6 single-protein wrapper and the reaction stays; an enzyme that cannot be written at all leaves the
reaction, which stays and is flagged (``review_required``, the actor recorded verbatim).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from t2pw.pipeline.strict_quarantine import quarantine_and_close

from test_orch773_omit_per_row import _leg, _refusals

ENZYME = "ceramide synthase"


def _sphingolipid(*, uniprot: str = "Q71RH2") -> Dict[str, Any]:
    """Shaped like idx 116: one reaction catalysed by a bare protein, one enzyme-less reaction beside it."""

    protein: Dict[str, Any] = {"name": ENZYME, "species": "Homo sapiens", "organism": "Homo sapiens"}
    if uniprot:
        protein["mapped_ids"] = {"uniprot": uniprot}
    actor = {"entity": ENZYME, "entity_type": "protein", "role": "catalyst", "provenance": "extracted",
             "confidence": 1.0, "evidence": "Ceramide Synthases in Biomedical Research"}
    compounds = ("Sphinganine", "dihydroceramide", "ceramide")
    return {
        "metadata": {"pw_id": "PW000000", "pathway_name": "sphingolipid biosynthesis",
                     "name": "sphingolipid biosynthesis", "pathway_subject": "Metabolic", "subject": "Metabolic",
                     "organism": "Homo sapiens", "width": 3200, "height": 1400,
                     "key_compounds": ["Sphinganine", "dihydroceramide"]},
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606", "classification": "Eukaryote"}],
            "subcellular_locations": [{"name": "endoplasmic reticulum"}],
            "compounds": [{"name": name} for name in compounds],
            "proteins": [protein],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "biological_states": [{"name": "ER state", "species": "Homo sapiens",
                               "subcellular_location": "endoplasmic reticulum"}],
        "element_locations": {
            "compound_locations": [{"compound": name, "biological_state": "ER state"} for name in compounds],
            "protein_locations": [],
        },
        "processes": {
            "reactions": [
                {"name": "Ceramide synthase reaction", "inputs": ["Sphinganine"], "outputs": ["ceramide"],
                 "enzymes": [], "biological_state": "ER state"},
                {"name": "ceramide synthase reaction", "inputs": ["Sphinganine"], "outputs": ["dihydroceramide"],
                 "enzymes": [deepcopy(actor)], "modifiers": [deepcopy(actor)], "biological_state": "ER state"},
            ],
            "transports": [],
            "interactions": [],
        },
    }


def _reaction(result, name: str) -> Dict[str, Any]:
    (row,) = [r for r in result.payload["processes"]["reactions"] if r["name"] == name]
    return row


def _omitted_names(result) -> List[str]:
    return [row["name"] for row in result.quarantine_report["quarantined"]]


def _actor_drops(result) -> List[tuple]:
    return [
        (row["bucket"], row["name"], row["reason"], row["detail"]["actor_removed_from"][0]["process_name"])
        for row in result.removed_entity_report["removed_entities"]
        if row["bucket"] in ("enzymes", "modifiers")
    ]


def test_l5_a_bare_protein_enzyme_is_wrapped_and_its_reaction_is_kept() -> None:
    """idx 116. Base: 'ceramide synthase reaction' omitted, reaction_enzyme_must_be_protein_complex."""

    payload = _sphingolipid()
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert result.ok
    assert _omitted_names(result) == []
    assert [r["name"] for r in result.payload["processes"]["reactions"]] == [
        "Ceramide synthase reaction", "ceramide synthase reaction"
    ]
    reaction = _reaction(result, "ceramide synthase reaction")
    for slot in ("enzymes", "modifiers"):
        assert [(a["entity"], a["entity_type"], a["role"]) for a in reaction[slot]] == [
            (f"{ENZYME} complex", "protein_complex", "catalyst")
        ]
        assert reaction[slot][0]["evidence"] == "Ceramide Synthases in Biomedical Research"
    (wrapper,) = result.payload["entities"]["protein_complexes"]
    assert wrapper["name"] == f"{ENZYME} complex"
    assert wrapper["generated"] is True and wrapper["generation_reason"] == "single_protein_pathwhiz_wrapper"
    assert [c["name"] for c in wrapper["components"]] == [ENZYME]
    assert ENZYME in {p["name"] for p in result.payload["entities"]["proteins"]}
    # The enzyme-less step beside it is untouched (an extraction question, not this lane's).
    assert _reaction(result, "Ceramide synthase reaction")["enzymes"] == []
    assert _actor_drops(result) == []
    assert _refusals(result) == []
    assert payload == _sphingolipid()  # the input is never mutated


def test_l5_an_enzyme_that_cannot_be_written_leaves_the_reaction_and_the_reaction_stays_flagged() -> None:
    """No database identity (strict): the wrapper refuses it and PathWhiz cannot take it. Base: the reaction
    was omitted whole (protein_missing_external_identity). Now it stays without that enzyme, review_required.
    L8 (D-W24 7b): an enzyme with no identity but a species is now an Unknown-backed placeholder
    (test_l8_placeholder_enzymes); with no species either it still cannot be written, so this case drops its species."""

    payload = _sphingolipid(uniprot="")
    for key in ("species", "organism"):
        payload["entities"]["proteins"][0].pop(key)
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert result.ok
    assert _omitted_names(result) == []
    reaction = _reaction(result, "ceramide synthase reaction")
    assert reaction["enzymes"] == [] and reaction["modifiers"] == []
    assert reaction["inputs"] == ["Sphinganine"] and reaction["outputs"] == ["dihydroceramide"]
    assert _actor_drops(result) == [
        ("enzymes", ENZYME, "protein_missing_species", "ceramide synthase reaction"),
        ("modifiers", ENZYME, "protein_missing_species", "ceramide synthase reaction"),
    ]
    dropped = [r for r in result.removed_entity_report["removed_entities"] if r["bucket"] == "enzymes"]
    assert dropped[0]["detail"]["actor"]["entity"] == ENZYME  # verbatim, for a reviewer to restore
    release = result.quarantine_report["release"]
    assert release["status"] == "review_required"
    assert "omitted_from_pwml:0_processes+2_actors" in release["reasons"]
    assert _refusals(result) == []


def test_l5_a_non_protein_modifier_leaves_the_reaction_not_the_reaction_itself() -> None:
    """idx 44 shape: a cofactor (a compound) listed as a reaction modifier. Base: the reaction was omitted
    (non_protein_actor_in_modifiers). Now only the modifier goes."""

    payload = _sphingolipid()
    payload["entities"]["compounds"].append({"name": "pyridoxal 5'-phosphate"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "pyridoxal 5'-phosphate", "biological_state": "ER state"})
    payload["processes"]["reactions"][1]["modifiers"].append({"entity": "pyridoxal 5'-phosphate", "role": "cofactor"})
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    assert _omitted_names(result) == []
    assert ("modifiers", "pyridoxal 5'-phosphate", "non_protein_actor_in_modifiers", "ceramide synthase reaction") in (
        _actor_drops(result))
    assert [a["entity"] for a in _reaction(result, "ceramide synthase reaction")["enzymes"]] == [f"{ENZYME} complex"]
    assert _refusals(result) == []


def test_l5_the_ldh_leg_keeps_its_reaction_with_a_wrapped_enzyme() -> None:
    """runs_verify 2026-08-21_2057 PMC12096016 (KI-006, also replay idx 27): 'lactate dehydrogenase' is a bare
    protein enzyme with a UniProt id. Base: 'LDH coupled reaction' omitted whole."""

    payload, context = _leg("runs_verify_2026-08-21_2057_PMC12096016")
    names_in = [r["name"] for r in payload["processes"]["reactions"]]
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    assert "LDH coupled reaction" not in _omitted_names(result)
    assert [r["name"] for r in result.payload["processes"]["reactions"]] == names_in
    assert [(a["entity"], a["entity_type"]) for a in _reaction(result, "LDH coupled reaction")["enzymes"]] == [
        ("lactate dehydrogenase complex", "protein_complex")
    ]
    assert _refusals(result) == []


def test_l5_research_mode_is_untouched() -> None:
    payload = _sphingolipid()
    result = quarantine_and_close(payload, strict_db=True, mode="research")
    assert result.payload == payload
    assert result.removed_entity_report["removed_entities"] == []


def test_l5_an_interaction_whose_endpoint_complex_cannot_be_written_is_kept_as_a_withheld_row() -> None:
    """Ruling 2 (idx 76/77 shape). runs_verify 2026-08-25_1216 PMC12856317: 'LONP1' is a complex with no
    resolvable component, so 'LONP1 degrades ALAS1' cannot be drawn. Base: the interaction left the payload
    and nothing downstream kept it. Now it is listed as a withheld row with its reason (pwml_ir.json and the
    withheld side file), as L4 withholds rows; it is still not drawn."""

    from types import SimpleNamespace

    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution
    from t2pw.pwml.validate import discover_structure_signature
    from t2pw.pwml.writer import DeterministicPwmlBuilder
    from test_pwml_l4_interactions import REFERENCE

    payload, context = _leg("runs_verify_2026-08-25_1216_PMC12856317")
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    assert _refusals(result) == []
    (held,) = result.payload["withheld_interactions"]
    assert (held["name"], held["reason"], held["endpoint"]) == (
        "LONP1 degrades ALAS1", "protein_complex_missing_components", "LONP1")
    assert held["row"]["name"] == "LONP1 degrades ALAS1"  # the row verbatim
    export = deepcopy(result.payload)
    run_prefreeze_resolution(export, strict_db=True)  # as _refusals and the export do
    ir, _report = build_pwml_ir(export, strict_db=True, width=3200, height=1400)
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF", ref=str(REFERENCE))
    build = DeterministicPwmlBuilder(extraction=ir, signature=discover_structure_signature(REFERENCE), args=args).build()
    (row,) = ir["processes"]["quarantine_withheld_interactions"]
    assert row["export_status"] == "withheld"
    assert row["withheld"]["reason"] == "protein_complex_missing_components"
    assert row["origin"] == "export_quarantine"
    assert "LONP1 degrades ALAS1" in {w["name"] for w in build.withheld_interactions}
    assert "LONP1 degrades ALAS1" not in {r["name"] for r in ir["processes"]["interactions"]}


# ── Review round 2 ──────────────────────────────────────────────────────────


def _drop_on_first_prune(monkeypatch, bucket: str, name: str) -> None:
    """Remove ``name`` from ``bucket`` behind closure's back, once (as test_strict_quarantine's backstop test)."""

    from t2pw.pipeline import strict_quarantine as sq

    real = sq._prune_entities
    calls = {"n": 0}

    def prune(payload, keep_norms):
        removed = real(payload, keep_norms)
        calls["n"] += 1
        if calls["n"] == 1:
            payload["entities"][bucket] = [r for r in payload["entities"][bucket] if r.get("name") != name]
        return removed

    monkeypatch.setattr(sq, "_prune_entities", prune)


def test_l5_an_actor_lost_during_closure_leaves_the_reaction_not_the_reaction_itself(monkeypatch) -> None:
    """Review F6. Closure's backstop quarantined a surviving reaction whole (``stranded_by_closure``) when only
    its enzyme stopped resolving. Now, as at admission, the reaction stays and loses only that actor."""

    _drop_on_first_prune(monkeypatch, "protein_complexes", f"{ENZYME} complex")
    result = quarantine_and_close(_sphingolipid(), strict_db=True, mode="pathwhiz")
    assert _omitted_names(result) == []
    reaction = _reaction(result, "ceramide synthase reaction")
    assert reaction["enzymes"] == [] and reaction["modifiers"] == []
    assert reaction["inputs"] == ["Sphinganine"] and reaction["outputs"] == ["dihydroceramide"]
    drops = [(r["bucket"], r["name"], r["reason"], r["detail"]["gate"])
             for r in result.removed_entity_report["removed_entities"] if r["bucket"] in ("enzymes", "modifiers")]
    assert drops == [
        ("enzymes", f"{ENZYME} complex", "unresolved_actor_in_enzymes", "strict_quarantine_closure"),
        ("modifiers", f"{ENZYME} complex", "unresolved_actor_in_modifiers", "strict_quarantine_closure"),
    ]
    assert "omitted_from_pwml:0_processes+2_actors" in result.quarantine_report["release"]["reasons"]


def _with_unidentified_regulator() -> Dict[str, Any]:
    payload = _sphingolipid()
    payload["entities"]["proteins"].append({"name": "unidentified regulator", "species": "Homo sapiens"})
    payload["processes"]["reactions"][1]["modifiers"].append(
        {"entity": "unidentified regulator", "entity_type": "protein", "role": "inhibitor"})
    return payload


def test_l5_a_dropped_actor_stays_in_the_frozen_payload_as_a_withheld_record() -> None:
    """Review F4 (idx 44 PLP): an actor a kept reaction loses is recorded in the payload itself
    (``withheld_reaction_actors``), so final_mapped.json and the clean model / SBML keep it, not only
    removed_entity_report.json."""

    from t2pw.clean_model.builder import build_from_payload

    payload = _sphingolipid()
    payload["entities"]["compounds"].append({"name": "pyridoxal 5'-phosphate"})
    payload["element_locations"]["compound_locations"].append(
        {"compound": "pyridoxal 5'-phosphate", "biological_state": "ER state"})
    plp = {"entity": "pyridoxal 5'-phosphate", "role": "cofactor"}
    payload["processes"]["reactions"][1]["modifiers"].append(deepcopy(plp))
    result = quarantine_and_close(payload, strict_db=True, mode="pathwhiz")
    (held,) = result.payload["withheld_reaction_actors"]
    assert (held["reaction"], held["slots"], held["name"], held["reason"], held["actor"], held["decision"]) == (
        "ceramide synthase reaction", ["modifiers"], "pyridoxal 5'-phosphate", "non_protein_actor_in_modifiers",
        plp, "D-W24 3")
    model = build_from_payload(result.payload)
    assert ("reaction_actor", "pyridoxal 5'-phosphate", "non_protein_actor_in_modifiers") in {
        (r.kind, r.name, r.reason) for r in model.removed}
    # A payload that lost no actor is unchanged by this field.
    assert "withheld_reaction_actors" not in quarantine_and_close(_sphingolipid(), strict_db=True).payload


def test_l5_a_reaction_that_leaves_after_losing_an_actor_keeps_that_actor_in_its_original(monkeypatch) -> None:
    """Review F5. Admission drops an actor from an accepted reaction; if the reaction later leaves anyway, its
    ``original_process`` is the row as it arrived (actor included), not the edited one. A reaction that left
    is not listed as having withheld an actor (F4)."""

    _drop_on_first_prune(monkeypatch, "compounds", "dihydroceramide")
    result = quarantine_and_close(_with_unidentified_regulator(), strict_db=True, mode="pathwhiz")
    (record,) = result.quarantine_report["quarantined"]
    assert record["name"] == "ceramide synthase reaction"
    assert record["reason"].startswith("stranded_by_closure:")
    assert "unidentified regulator" in [a["entity"] for a in record["original_process"]["modifiers"]]
    assert "withheld_reaction_actors" not in result.payload


def test_l5_the_quarantine_report_records_the_wrappers_it_built() -> None:
    """Review F7 (PRODUCT_CONTRACT section 3): which bare enzyme the quarantine wrapped, into what, for which
    reactions, and on what basis."""

    result = quarantine_and_close(_sphingolipid(), strict_db=True, mode="pathwhiz")
    assert result.quarantine_report["schema_version"] == 7
    (row,) = result.quarantine_report["enzyme_wrappers"]
    assert (row["protein"], row["protein_complex"], row["built_here"], row["reactions"], row["decision"]) == (
        ENZYME, f"{ENZYME} complex", True, ["ceramide synthase reaction"], "D-W24 3")
    assert (row["stage"], row["origin"], row["paper_explicit"]) == ("quarantine", "inferred", "not_evaluated")
    # No lineage written on the row: that would switch on the support gate's archival guard.
    (wrapper,) = result.payload["entities"]["protein_complexes"]
    assert not wrapper.get("provenance_lineage")
    assert quarantine_and_close(_sphingolipid(), strict_db=True, mode="research").quarantine_report[
        "enzyme_wrappers"] == []
