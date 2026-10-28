"""ORCH-782 (KI-011): a gene and a protein that share one bare symbol.

**G9: A CORRECTION OF OBSERVABLE BEHAVIOUR, NOT NEW WORK.** Only names that exist at
the base SHA ``5e1edb3f`` are imported, so this module imports cleanly there, and every
``test_g9_*`` test fails there on VALUES: base quarantines every process naming the
symbol as ``ambiguous_entity_type:nucleic_acids+proteins``. ``test_preservation_*``
passes at base and tip by design and fails only if the split reaches an unrelated row.

The product-owner ruling: the gene row becomes ``"<SYMBOL> gene"``; an interaction
target whose wording is expression / promoter / transcription resolves to the gene;
an enzyme (and every other non-target role) keeps the protein; an undecidable target
omits only that one row under the KI-011 reason code.

Real shape: the ORCH-759 chlorophyll-strict leg's quarantine input, already committed
verbatim as ``tests/fixtures/orch759c/`` (see its README).
"""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

from t2pw.pipeline.strict_quarantine import quarantine_and_close

from test_strict_quarantine_contract_alignment import _base

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "orch759c"
KI011 = "ambiguous_entity_type:nucleic_acids+proteins"


def _synthetic():
    """_base() plus a gene GCLC beside the protein GCLC (the first reaction's enzyme)."""

    payload = _base()
    entities = payload["entities"]
    entities["proteins"][0]["name"] = "GCLC"
    entities["protein_complexes"][0]["components"] = [{"name": "GCLC", "stoichiometry": 1}]
    entities["proteins"].append(
        {"name": "NRF2", "species": "Homo sapiens", "mapped_ids": {"uniprot": "Q16236"}}
    )
    entities["nucleic_acids"] = [{"name": "GCLC", "class": "nucleic_acid"}]
    payload["element_locations"]["nucleic_acid_locations"] = [
        {"nucleic_acid": "GCLC", "biological_state": "cytosol"}
    ]
    payload["element_locations"]["protein_locations"] = [
        {"protein": "NRF2", "biological_state": "cytosol"}
    ]
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "GCLC", "entity_type": "protein", "role": "catalyst"}
    ]
    payload["processes"]["interactions"] = [
        {  # 0: gene role (promoter / expression)
            "name": "NRF2 activates GCLC",
            "entity_1": "NRF2",
            "entity_2": "GCLC",
            "relationship": "activation",
            "evidence": "NRF2 binds the GCLC promoter and induces its expression",
        },
        {  # 1: undecidable
            "name": "NRF2 binds GCLC",
            "entity_1": "NRF2",
            "entity_2": "GCLC",
            "relationship": "binding",
            "evidence": "NRF2 associates with GCLC",
        },
        {  # 2: unrelated (no clashing symbol)
            "name": "NRF2 activates glutathione synthetase",
            "entity_1": "NRF2",
            "entity_2": "glutathione synthetase",
            "relationship": "activation",
            "evidence": "NRF2 induces glutathione synthetase expression",
        },
    ]
    return payload


def _reasons(result):
    return {(a["bucket"], a["index"]): a["reason"] for a in result.quarantine_report["admissions"]}


def _interaction(result, name):
    return next(row for row in result.payload["processes"]["interactions"] if row["name"] == name)


def test_g9_gene_role_target_resolves_to_gene():
    result = quarantine_and_close(_synthetic(), strict_db=True)
    assert _reasons(result)[("interactions", 0)] == "admitted"
    assert _interaction(result, "NRF2 activates GCLC")["entity_2"] == "GCLC gene"
    assert [row["name"] for row in result.payload["entities"]["nucleic_acids"]] == ["GCLC gene"]
    assert result.payload["element_locations"]["nucleic_acid_locations"][0]["nucleic_acid"] == "GCLC gene"


def test_g9_enzyme_role_resolves_to_protein():
    result = quarantine_and_close(_synthetic(), strict_db=True)
    assert _reasons(result)[("reactions", 0)] == "admitted"
    assert result.payload["processes"]["reactions"][0]["enzymes"][0]["entity"] == "GCLC"
    assert "GCLC" in [row["name"] for row in result.payload["entities"]["proteins"]]


def test_g9_ambiguous_target_omits_only_that_row():
    result = quarantine_and_close(_synthetic(), strict_db=True)
    reasons = _reasons(result)
    assert reasons[("interactions", 1)] == KI011
    assert reasons[("interactions", 0)] == reasons[("reactions", 0)] == "admitted"
    assert result.ok and result.refusal_reasons == []


def test_preservation_unrelated_rows_untouched():
    payload = _synthetic()
    result = quarantine_and_close(payload, strict_db=True)
    assert _reasons(result)[("interactions", 2)] == "admitted"
    assert _interaction(result, "NRF2 activates glutathione synthetase") == payload["processes"]["interactions"][2]
    kept = [row for row in result.payload["processes"]["reactions"] if row["name"] == "glutathione synthesis"]
    assert kept == [payload["processes"]["reactions"][1]]


def test_g9_chlorophyll_leg_recovers_gene_regulation_by_row():
    payload = json.load(
        gzip.open(FIXTURES / "chlorophyll_PMC5130987_final_mapped.json.gz", "rt", encoding="utf-8")
    )
    context = json.loads((FIXTURES / "chlorophyll_PMC5130987_requested_context.json").read_text(encoding="utf-8"))
    original = deepcopy(payload)
    result = quarantine_and_close(payload, strict_db=True, pathway_context=context, mode="pathwhiz")
    assert payload == original  # input never mutated
    by_name = {}
    for record in result.quarantine_report["admissions"]:
        by_name.setdefault(record["name"], record["reason"])

    # gene role: expression / promoter wording -> "<SYMBOL> gene"
    for name, gene in (("HY5 activates HEMA1", "HEMA1 gene"), ("PIF3 inhibits GUN4", "GUN4 gene"),
                       ("TOC1 inhibits CHLH", "CHLH gene")):
        assert by_name[name] == "admitted", name
        assert _interaction(result, name)["entity_2"] == gene
    # regulator / enzyme role -> the protein (GLK1 is a gene too; ALAD1/HEMB1 is the enzyme,
    # now judged as the protein by the identity gate instead of by ambiguity)
    assert by_name["GLK1 activates HEMA1 promoter"] == "admitted"
    assert _reasons(result)[("reactions", 17)] == "protein_missing_external_identity"  # 2nd "ALA condensation to PBG"
    # undecidable (protein-level wording) -> only that row, under KI-011
    assert by_name["FLU inhibits HEMA1"] == KI011
    assert result.ok and result.refusal_reasons == []
    assert sum(r == KI011 for r in _reasons(result).values()) == 9
