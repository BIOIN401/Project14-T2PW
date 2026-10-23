"""ORCH-759e: a protein whose identity only the Phase-2 UniProt fallback finds is wrapped like any other.

Stage 6 (``map_payload``) builds the single-protein ProteinComplex wrappers that PathWhiz needs
for reaction enzymes and signed-interaction endpoints (D-104 § 4) in
``_rewrite_reaction_protein_enzymes_to_complexes``. That rewrite ran BEFORE the Phase-2
UniProt fallback, so a protein that had no identity at the rewrite and was resolved by the
fallback afterwards shipped as a bare protein: ``reaction_enzyme_must_be_protein_complex``
on the ORCH-759 chlorophyll leg (``CPO reaction``), and on PMC12096016 / PMC11405693 in
``runs_verify``. The Unknown fallback did not catch it either, because by then the protein
had a real identity. ``product_contract_violation``, merge rule 7.

**G9 -- CORRECTION of observable behaviour** (``test_g9_*``). Each drives ``map_payload``
itself, offline (the main resolver and the Phase-2 resolver are stubbed per name; no
network, no database, no model), imports nothing this lane adds, and on base ``366b9395``
fails on the shipped payload: the enzyme or endpoint is still a bare protein.

**PRESERVATION** (``test_preservation_*``): pass at base and at tip.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from t2pw.curation.interactive_curator import compact_mapping_misses
from t2pw.mapping import map_ids
from t2pw.pwml.ir import build_pwml_ir, validate_required_pwml_contract

ORGANISM = "Arabidopsis thaliana"
#: The Phase-2 answer the ORCH-759 chlorophyll leg recorded for CPO (its identity_verdict's
#: judged candidate): accepted on the gene symbol, ``exact_symbol_identity``.
CPO_CANDIDATE = {
    "accession": "Q9LR75",
    "uniprot": "Q9LR75",
    "protein_name": "Coproporphyrinogen-III oxidase 1, chloroplastic",
    "gene_names": ["CPO", "CPX1", "HEMF", "HEMF1", "LIN2"],
    "primary_gene_names": ["CPX1"],
    "organism": ORGANISM,
    "taxonomy_id": "3702",
    "reviewed": True,
    "score": 1.0,
}


def _unmapped() -> Dict[str, Any]:
    return {
        "status": "unmapped",
        "reason": "no_match",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.0,
        "candidates": [],
        "resolution": {"status": "unresolved", "issue": "no_match", "order_step": "api_uniprot"},
    }


def _mapped(name: str, accession: str, candidate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = candidate or {"accession": accession, "uniprot": accession, "protein_name": name,
                        "organism": ORGANISM, "reviewed": True, "score": 0.99}
    return {
        "status": "mapped",
        "provider": "UniProt",
        "source": "api",
        "confidence": 0.99,
        "chosen_rule": "reviewed_exact",
        "resolved_name": row["protein_name"],
        "mapped_ids": {"uniprot": accession},
        "candidates": [dict(row)],
        "resolution": {"status": "matched", "order_step": "api_uniprot"},
    }


def _protein(name: str) -> Dict[str, Any]:
    return {"name": name, "species": ORGANISM, "organism": ORGANISM, "species_id": 4, "pathbank_species_id": 4}


def _actor(name: str) -> Dict[str, Any]:
    return {"entity": name, "entity_type": "protein", "role": "catalyst", "evidence": f"quote naming {name}",
            "provenance": "extracted"}


def _payload(proteins: List[str], enzymes: List[str], *, mirror_modifiers: bool = False,
             interactions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    reaction: Dict[str, Any] = {
        "name": "coprogen III oxidation",
        "inputs": ["coproporphyrinogen III"],
        "outputs": ["protoporphyrinogen IX"],
        "biological_state": "chloroplast",
        "enzymes": [_actor(n) for n in enzymes],
    }
    if mirror_modifiers:
        reaction["modifiers"] = [_actor(n) for n in enzymes]
    return {
        "metadata": {"pathway_name": "Tetrapyrrole synthesis", "pathway_subject": "Metabolic",
                     "organism": ORGANISM, "width": 3200, "height": 1400},
        "entities": {
            "species": [{"name": ORGANISM, "pathbank_species_id": 4}],
            "subcellular_locations": [{"name": "chloroplast", "pathwhiz_id": 8}],
            "compounds": [{"name": "coproporphyrinogen III"}, {"name": "protoporphyrinogen IX"}],
            "proteins": [_protein(n) for n in proteins],
            "protein_complexes": [],
        },
        "biological_states": [{"name": "chloroplast", "species": ORGANISM, "subcellular_location": "chloroplast",
                               "compartment_canonical": "chloroplast"}],
        "element_locations": {
            "compound_locations": [{"compound": c, "biological_state": "chloroplast"}
                                   for c in ("coproporphyrinogen III", "protoporphyrinogen IX")],
            "protein_locations": [{"protein": n, "biological_state": "chloroplast"} for n in proteins],
        },
        "processes": {"reactions": [reaction] if enzymes else [], "transports": [],
                      "interactions": interactions or []},
    }


class _NoNetwork:
    """Stands in for ``HttpClient``: every request is recorded, none is sent."""

    calls: List[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
        _NoNetwork.calls.append(url)
        raise ConnectionError("offline test")

    post = get


def _run(payload: Dict[str, Any], tmp_path: Path, *, main: Dict[str, Dict[str, Any]],
         phase2: Dict[str, Dict[str, Any]], wrappers: bool = True) -> Dict[str, Any]:
    """``map_payload`` as Stage 6 calls it, with each resolver answering per protein name.

    ``main`` answers the main protein loop (``_map_protein_with_strategy``), ``phase2`` the
    Phase-2 UniProt fallback (``map_protein_uniprot``). A name a table does not hold is
    unmapped there.
    """

    _NoNetwork.calls = []

    def main_loop(**kwargs: Any) -> Dict[str, Any]:
        return deepcopy(main.get(kwargs["name"], _unmapped()))

    def fallback(_client: Any, name: str, _organism: str, **_kwargs: Any) -> Dict[str, Any]:
        return deepcopy(phase2.get(name, _unmapped()))

    with (
        patch.object(map_ids, "HttpClient", _NoNetwork),
        patch.object(map_ids, "hydrate_species_references", return_value={"hydrated": 0, "matched": 0, "novel": 0}),
        patch.object(map_ids, "_map_protein_with_strategy", side_effect=main_loop),
        patch.object(map_ids, "map_protein_uniprot", side_effect=fallback),
        patch.object(map_ids, "_map_compound_with_strategy", return_value=_unmapped()),
    ):
        result = map_ids.map_payload(
            deepcopy(payload), cache_path=tmp_path / "mapping-cache.json", id_source="api", use_cache=False,
            allow_complex_wrapper_creation=wrappers, allow_structural_cleanup=wrappers,
            merge_identity_duplicates=True,
        )
    assert _NoNetwork.calls == [], _NoNetwork.calls
    return result


def _reaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload["processes"]["reactions"][0]


def _actors(payload: Dict[str, Any], key: str = "enzymes") -> List[tuple]:
    return [(a["entity"], a["entity_type"]) for a in _reaction(payload).get(key) or []]


def _complex(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(r for r in payload["entities"]["protein_complexes"] if r["name"] == name)


def _codes(report: Dict[str, Any]) -> List[str]:
    return [str(e.get("code")) for e in report.get("errors") or []]


def _contract_and_ir_codes(payload: Dict[str, Any]) -> List[str]:
    contract = validate_required_pwml_contract(deepcopy(payload), strict_db=False)
    _ir, ir_report = build_pwml_ir(prefrozen_when_compounded(deepcopy(payload)), strict_db=False)
    return _codes(contract) + _codes(ir_report)


# ── G9 ──────────────────────────────────────────────────────────────────────────────────


def test_g9_an_enzyme_resolved_by_the_phase2_fallback_ships_wrapped(tmp_path: Path) -> None:
    """The shape of PMC12096016 (lactate dehydrogenase) and PMC11405693 (AtCYP73A5): one
    enzyme, no identity in the main loop, an accession from the Phase-2 fallback."""

    name = "lactate dehydrogenase"
    result = _run(_payload([name], [name]), tmp_path, main={}, phase2={name: _mapped(name, "P52643")})
    payload = result["payload"]

    protein = next(r for r in payload["entities"]["proteins"] if r["name"] == name)
    assert protein["mapped_ids"]["uniprot"] == "P52643"
    assert protein["mapping_meta"]["chosen_rule"] == "api_uniprot_fallback"
    # Base: [("lactate dehydrogenase", "protein")] -- the bare enzyme the importer refuses.
    assert _actors(payload) == [("lactate dehydrogenase complex", "protein_complex")]
    wrapper = _complex(payload, "lactate dehydrogenase complex")
    assert wrapper["generated"] is True and wrapper["generation_reason"] == "single_protein_pathwhiz_wrapper"
    assert [(c["name"], c["mapped_ids"]["uniprot"]) for c in wrapper["components"]] == [(name, "P52643")]
    assert "reaction_enzyme_must_be_protein_complex" not in _contract_and_ir_codes(payload)
    # Through the existing path: the wrapper and the actor row are the ones an enzyme the
    # main loop resolved gets, byte for byte (the complex-mapping pass included).
    main = _run(_payload([name], [name]), tmp_path, main={name: _mapped(name, "P52643")}, phase2={})["payload"]
    assert wrapper == _complex(main, "lactate dehydrogenase complex")
    assert _reaction(payload)["enzymes"] == _reaction(main)["enzymes"]


def test_g9_the_cpo_shape_wraps_cpo_and_keeps_the_cpo1_placeholder(tmp_path: Path) -> None:
    """The ORCH-759 chlorophyll leg: ``CPO reaction`` names CPO and CPO1 as enzymes and as
    catalyst modifiers. The main loop leaves both without an identity, the Phase-2 fallback
    finds Q9LR75 for CPO only, and CPO1 takes the Unknown fallback as before."""

    result = _run(
        _payload(["CPO", "CPO1"], ["CPO", "CPO1"], mirror_modifiers=True),
        tmp_path,
        main={},
        phase2={"CPO": _mapped("CPO", "Q9LR75", CPO_CANDIDATE)},
    )
    payload = result["payload"]

    # Base: [("CPO", "protein"), ("CPO1", "protein_complex")] in both lists.
    expected = [("CPO complex", "protein_complex"), ("CPO1", "protein_complex")]
    assert _actors(payload, "enzymes") == expected
    assert _actors(payload, "modifiers") == expected
    wrapper = _complex(payload, "CPO complex")
    assert [(c["name"], c["mapped_ids"]["uniprot"]) for c in wrapper["components"]] == [("CPO", "Q9LR75")]
    placeholder = _complex(payload, "CPO1")
    assert placeholder["mapping_meta"]["chosen_rule"] == "pathbank_unknown_protein_fallback"
    assert [c["name"] for c in placeholder["components"]] == ["Unknown"]
    assert "reaction_enzyme_must_be_protein_complex" not in _contract_and_ir_codes(payload)


def test_g9_an_interaction_endpoint_resolved_by_the_phase2_fallback_is_exported(tmp_path: Path) -> None:
    """D-104 § 4 for interactions: both endpoints of a signed row must be wrappable, so a
    Phase-2-resolved endpoint used to leave the whole row bare and the IR withheld it."""

    interaction = {"name": "SFXN4 inhibits ALAS2", "entity_1": "SFXN4", "relationship": "inhibits",
                   "entity_2": "ALAS2", "biological_state": "chloroplast", "evidence": "quote"}
    result = _run(
        _payload(["SFXN4", "ALAS2"], [], interactions=[interaction]),
        tmp_path,
        main={"ALAS2": _mapped("ALAS2", "P22557")},
        phase2={"SFXN4": _mapped("SFXN4", "Q6P4A7")},
    )
    payload = result["payload"]
    row = payload["processes"]["interactions"][0]

    # Base: ("SFXN4", "ALAS2") and the IR withholds the row as ``bare_protein_endpoint``.
    assert (row["entity_1"], row["entity_2"]) == ("SFXN4 complex", "ALAS2 complex")
    ir, _report = build_pwml_ir(prefrozen_when_compounded(deepcopy(payload)), strict_db=False)
    exported = ir["processes"]["interactions"][0]
    assert (exported["export_status"], exported["interaction_type"]) == ("exportable", "Inhibition")


# ── PRESERVATION ──────────────────────────────────────────────────────────────────────────


def test_preservation_the_late_wrap_writes_no_report_record_of_its_own(tmp_path: Path) -> None:
    """D-104 amendment 2. The mapping report reaches the interactive curator's prompt
    (``compact_mapping_misses`` walks it whole), so the late wrap adds no key, counter or
    action to it: the report has exactly the keys of a run whose main loop resolved the
    enzyme, and the actions of the run as the base wrote them."""

    name = "lactate dehydrogenase"
    late = _run(_payload([name], [name]), tmp_path, main={}, phase2={name: _mapped(name, "P52643")})["report"]
    main = _run(_payload([name], [name]), tmp_path, main={name: _mapped(name, "P52643")}, phase2={})["report"]

    assert sorted(late) == sorted(main)
    assert sorted(late["summary"]) == sorted(main["summary"])
    assert [a["type"] for a in late["enzyme_complex_conversion"]["actions"]] == [
        "reaction_enzyme_complex_wrapper_skipped_invalid_component"]
    assert late["summary"]["reaction_protein_enzymes_rewritten_to_complexes"] == 0
    assert compact_mapping_misses(late)  # the walker still sees the report; nothing new is in it


def test_preservation_an_enzyme_resolved_in_the_main_loop_is_wrapped_by_the_one_rewrite(tmp_path: Path) -> None:
    name = "lactate dehydrogenase"
    with patch.object(map_ids, "_rewrite_reaction_protein_enzymes_to_complexes",
                      wraps=map_ids._rewrite_reaction_protein_enzymes_to_complexes) as rewrite:
        result = _run(_payload([name], [name]), tmp_path, main={name: _mapped(name, "P52643")},
                      phase2={name: _mapped(name, "P00000")})
    payload = result["payload"]
    assert rewrite.call_count == 1
    assert _actors(payload) == [("lactate dehydrogenase complex", "protein_complex")]
    protein = next(r for r in payload["entities"]["proteins"] if r["name"] == name)
    assert protein["mapped_ids"] == {"uniprot": "P52643"}
    assert protein["mapping_meta"]["chosen_rule"] == "reviewed_exact"
    summary = result["report"]["summary"]
    assert summary["reaction_protein_enzymes_rewritten_to_complexes"] == 1
    assert summary["reaction_enzyme_complexes_skipped_invalid_component"] == 0


def test_preservation_an_enzyme_still_unresolved_after_phase2_takes_the_unknown_fallback(tmp_path: Path) -> None:
    name = "lactate dehydrogenase"
    with patch.object(map_ids, "_rewrite_reaction_protein_enzymes_to_complexes",
                      wraps=map_ids._rewrite_reaction_protein_enzymes_to_complexes) as rewrite:
        payload = _run(_payload([name], [name]), tmp_path, main={}, phase2={})["payload"]
    assert rewrite.call_count == 1
    assert _actors(payload) == [(name, "protein_complex")]
    assert [r["name"] for r in payload["entities"]["protein_complexes"]] == [name]
    assert _complex(payload, name)["mapping_meta"]["chosen_rule"] == "pathbank_unknown_protein_fallback"


def test_preservation_research_mode_wraps_nothing(tmp_path: Path) -> None:
    name = "lactate dehydrogenase"
    payload = _run(_payload([name], [name]), tmp_path, main={}, phase2={name: _mapped(name, "P52643")},
                   wrappers=False)["payload"]
    assert _actors(payload) == [(name, "protein")]
    assert payload["entities"]["protein_complexes"] == []


def test_preservation_the_drop_rule_still_runs_before_phase2(tmp_path: Path) -> None:
    """Not this lane's rule, pinned so the boundary is visible: an enzyme with no identity at
    the rewrite, beside one that has an identity, is dropped there ("other enzymes remain")
    and its row removed, so the Phase-2 fallback never sees it. Unchanged; recorded in
    ``ORCH-759e-RESULT.md`` under "Not settled". The resolved sibling is wrapped exactly as
    when it stands alone."""

    names = ["HEMA1", "CPO"]
    alone = _run(_payload(["HEMA1"], ["HEMA1"]), tmp_path, main={"HEMA1": _mapped("HEMA1", "P42804")}, phase2={})
    both = _run(_payload(names, names), tmp_path, main={"HEMA1": _mapped("HEMA1", "P42804")},
                phase2={"CPO": _mapped("CPO", "Q9LR75", CPO_CANDIDATE)})
    assert _actors(both["payload"]) == [("HEMA1 complex", "protein_complex")]
    assert [r["name"] for r in both["payload"]["entities"]["proteins"]] == ["HEMA1"]
    assert _reaction(both["payload"])["enzymes"] == _reaction(alone["payload"])["enzymes"]
    assert _complex(both["payload"], "HEMA1 complex") == _complex(alone["payload"], "HEMA1 complex")
