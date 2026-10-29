"""ORCH-790 (family path, D-119 § 1) and ORCH-791 (Stage-6 step order, ORCH-789 C1).

G9 labels
- G9 BEHAVIOURAL PROOF (``test_g9_*``; each fails on base f3a80e09 on a value, passes at tip):
  * "G protein" never takes EIF3G (ORCH-789 D1): the UniProt phrase match on
    ``family:"G family"`` returned "eIF-3 subunit G family".
  * the D-119 family terms are detected, spacing/hyphen/plural-insensitive.
  * a family name gets no best-effort UniProt guess (ORCH-789 D3: PKC -> Par-3).
  * a guessed accession on a family row does not stop the family conversion (C3/PKC).
  * the family complex keeps the row's synonyms (C8-i: NF-ATp).
  * member choice does not depend on the order UniProt lists its hits.
  * ORCH-791: an identified regulator of a gene-role target is wrapped (C1, chlorophyll).
  * ORCH-791: the partner of a converted family is wrapped (C1, "Fzd2 activates G protein").
  * ORCH-791: a family modifier is converted, not dropped as a redundant id-less actor (C3/PKC).
- PRESERVATION (``test_preservation_*``): single proteins stay single; a clash target the
  split cannot decide is still left alone.

No test touches the network. The UniProt answers below are the live answers recorded in
``docs/pwml_recovery_sprint/evidence/orch790/probe_family_fields.json`` (trimmed).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from t2pw.mapping import map_ids
from t2pw.pipeline.entity_identity import is_pathbank_unknown_protein
from t2pw.pipeline.strict_quarantine import _split_gene_protein_names

HUMAN = {"name": "Homo sapiens", "taxonomy_id": "9606"}


class _UniProtTsv:
    """Serves recorded TSV rows (accession, gene, taxon, families) per family query."""

    def __init__(self, answers):
        self.answers = answers

    def get(self, url, params=None, **_):
        rows = self.answers.get(params["query"].split(" AND organism_id:")[0], [])
        text = "Entry\tGene Names (primary)\tOrganism (ID)\tProtein families\n" + "".join(
            "\t".join(row) + "\n" for row in rows
        )
        return SimpleNamespace(status_code=200, text=text)


def _payload(proteins, interactions=(), species=HUMAN):
    return {
        "entities": {
            "species": [dict(species)],
            "proteins": [p if isinstance(p, dict) else {"name": p, "organism": species["name"]} for p in proteins],
            "compounds": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "transports": [], "interactions": list(interactions)},
    }


def _complex(mapped, name):
    return next(c for c in mapped["entities"]["protein_complexes"] if c["name"] == name)


def _protein(mapped, name):
    return next((p for p in mapped["entities"]["proteins"] if p["name"] == name), None)


# ── ORCH-790 ─────────────────────────────────────────────────────────────────


def test_g9_g_protein_never_takes_eif3g():
    client = _UniProtTsv({'family:"G family"': [("O75821", "EIF3G", "9606", "EIF-3 subunit G family")]})
    mapped = _payload(["G protein"])
    map_ids._apply_family_complexes(mapped, client=client, db=None)

    pc = _complex(mapped, "G protein")
    assert [c["name"] for c in pc["components"]] == [], "EIF3G is not a G protein"
    assert all(p.get("uniprot_id") != "O75821" for p in mapped["entities"]["proteins"])
    assert "no members found" in pc["mapping_meta"]["family_complex"]["note"]


@pytest.mark.parametrize(
    "name", ["JNK", "ROCK", "GDI", "RhoGDI", "Rho GDI", "DELLA", "DELLAs", "PIFs", "Orai",
             "5-HT 3 receptors", "5-HT3 receptor", "G proteins", "VGCCs"]
)
def test_g9_d119_family_terms_detected(name):
    assert map_ids.is_family_or_complex_term(name)


@pytest.mark.parametrize("name", ["NLK", "Cdc42", "TCF4", "PIF1", "PIF3", "Orai1", "GDI1", "ROCK1", "JNK1", "RhoA"])
def test_preservation_single_proteins_not_caught(name):
    assert not map_ids.is_family_or_complex_term(name)


def test_g9_rhogdi_members_are_the_rho_gdis_not_the_rab_gdis():
    rho = [("P52565", "ARHGDIA", "9606", "Rho GDI family"), ("P52566", "ARHGDIB", "9606", "Rho GDI family"),
           ("Q99819", "ARHGDIG", "9606", "Rho GDI family")]
    rab = [("P31150", "GDI1", "9606", "Rab GDI family"), ("P50395", "GDI2", "9606", "Rab GDI family")]
    client = _UniProtTsv({'family:"GDI family"': rho + rab, 'family:"Rho GDI family"': rho})
    mapped = _payload(["GDI"])
    map_ids._apply_family_complexes(mapped, client=client, db=None)
    assert [c["name"] for c in _complex(mapped, "GDI")["components"]] == ["ARHGDIA", "ARHGDIB", "ARHGDIG"]


_PAR3 = {
    "results": [{
        "primaryAccession": "Q8TEW0",
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Partitioning defective 3 homolog"}},
            "alternativeNames": [{"fullName": {"value": "Atypical PKC isotype-specific-interacting protein"}}],
        },
        "genes": [{"geneName": {"value": "PARD3"}}],
        "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
    }]
}


def test_g9_family_name_gets_no_best_effort_guess():
    client = SimpleNamespace(get=lambda url, params=None, headers=None: SimpleNamespace(
        status_code=200, json=lambda: _PAR3, text=""))
    with patch.object(map_ids, "_ai_protein_synonym_lookup", return_value=[]), patch.object(
        map_ids, "lookup_literature_protein_aliases", return_value={"aliases": []}
    ):
        result = map_ids.map_protein_uniprot(client, "PKC", "Homo sapiens")
        single = map_ids.map_protein_uniprot(client, "PARD3x", "Homo sapiens")
    assert result["status"] == "unmapped", result.get("mapped_ids")
    assert result["reason"] == "family_name_has_no_single_protein"
    assert not result.get("best_effort")
    # A single-protein name keeps today's best-effort path (not this lane's to change).
    assert single.get("best_effort") is True


@pytest.mark.parametrize(
    "meta",
    [
        {"chosen_rule": "best_effort_fallback", "best_effort": True, "resolution": {"status": "matched"}},
        {"chosen_rule": "stage3_api_background_candidate", "resolution": {"status": "unresolved"}},
    ],
)
def test_g9_guessed_accession_does_not_block_family_conversion(meta):
    pkc = {"name": "PKC", "organism": "Homo sapiens", "species": "Homo sapiens",
           "uniprot_id": "Q8TEW0", "mapped_ids": {"uniprot": "Q8TEW0"}, "mapping_meta": dict(meta)}
    mapped = _payload([pkc])
    map_ids._apply_family_complexes(mapped, client=None, db=None)
    assert _protein(mapped, "PKC") is None, "a guessed Par-3 accession must not keep PKC a single protein"
    assert _complex(mapped, "PKC")["mapping_meta"]["family_complex"]["family"] == "PKC"


def test_preservation_matched_accession_still_keeps_a_family_named_row():
    row = {"name": "PLC", "organism": "Homo sapiens", "species": "Homo sapiens", "uniprot_id": "P19174",
           "mapped_ids": {"uniprot": "P19174"},
           "mapping_meta": {"chosen_rule": "pathbank_protein_id", "resolution": {"status": "matched"}}}
    mapped = _payload([row])
    map_ids._apply_family_complexes(mapped, client=None, db=None)
    assert _protein(mapped, "PLC") is not None and mapped["entities"]["protein_complexes"] == []


def test_g9_family_complex_keeps_synonyms():
    mapped = _payload([{"name": "NFAT", "organism": "Homo sapiens", "synonyms": ["NF-ATp"]}])
    map_ids._apply_family_complexes(mapped, client=None, db=None)
    assert _complex(mapped, "NFAT").get("synonyms") == ["NF-ATp"]


def test_g9_members_do_not_depend_on_uniprot_hit_order():
    hits = [("Q00001", "AAA2", "9606", "Toy family"), ("Q00002", "BBB1", "9606", "Toy family"),
            ("Q00003", "AAA1", "9606", "Toy family"), ("Q00004", "BBB2", "9606", "Toy family")]
    chosen = []
    for order in (hits, list(reversed(hits))):
        mapped = _payload(["Toy family"])
        map_ids._apply_family_complexes(mapped, client=_UniProtTsv({'family:"Toy family"': order}), db=None)
        chosen.append([c["name"] for c in _complex(mapped, "Toy family")["components"]])
    assert chosen[0] == chosen[1] == ["AAA1", "AAA2"]


# ── ORCH-791 ─────────────────────────────────────────────────────────────────

ARABIDOPSIS = {"name": "Arabidopsis thaliana", "taxonomy_id": "3702"}


def _plant_protein(name, uniprot):
    return {"name": name, "species": ARABIDOPSIS["name"], "organism": ARABIDOPSIS["name"], "species_id": 4,
            "uniprot_id": uniprot, "mapped_ids": {"uniprot": uniprot}}


def _chlorophyll_shape():
    """The ORCH-785 chlorophyll shape: CHLH is declared both as a gene and as a protein."""
    payload = _payload([_plant_protein("HY5", "O24646"), _plant_protein("CHLH", "Q9FNB0")], [
        {"name": "HY5 activates CHLH expression", "entity_1": "HY5", "relationship": "activates",
         "entity_2": "CHLH", "evidence": "HY5 activates the expression of CHLH"},
        {"name": "HY5 regulates CHLH", "entity_1": "HY5", "relationship": "activates",
         "entity_2": "CHLH", "evidence": "HY5 acts on CHLH"},
    ], species=ARABIDOPSIS)
    payload["entities"]["nucleic_acids"] = [{"name": "CHLH"}]
    return payload


def _stage6_wrap(payload, **kwargs):
    return map_ids._rewrite_reaction_protein_enzymes_to_complexes(
        payload, db=None, cache=map_ids.MappingCache(Path("unused.json"), enabled=False),
        global_organism=payload["entities"]["species"][0]["name"], **kwargs,
    )


def test_g9_regulator_of_a_gene_role_target_is_wrapped():
    payload = _chlorophyll_shape()
    _stage6_wrap(payload)
    expression = payload["processes"]["interactions"][0]
    assert expression["entity_1"] == "HY5 complex", "HY5 has species and UniProt; the gene target must not strand it"
    assert expression["entity_2"] == "CHLH"  # the gene; quarantine's split renames it
    forced = _split_gene_protein_names(payload)
    assert expression["entity_2"] == "CHLH gene"
    # The undecidable row: still omitted by the split, still not wrapped here.
    assert ("interactions", 1) in forced
    assert payload["processes"]["interactions"][1]["entity_1"] == "HY5"


def _resolved(name, gene, accession):
    return {"status": "mapped", "mapped_ids": {"uniprot": accession}, "provider": "UniProt", "source": "api",
            "confidence": 1.0, "chosen_rule": "uniprot_exact_gene", "resolved_name": name,
            "candidates": [{"name": name, "gene_name": gene, "uniprot": accession,
                            "organism": "Homo sapiens", "taxonomy_id": "9606", "score": 1.0}],
            "resolution": {"status": "matched"}, "identity_verdict": None}


def _stage6_offline(payload, tmp_path, resolved):
    """``map_payload`` as Stage 6 calls it, offline; ``resolved`` answers named rows."""

    real = map_ids._map_protein_with_strategy

    def strategy(**kwargs):
        hit = resolved.get(kwargs.get("name"))
        return dict(hit) if hit else real(**kwargs)

    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0", "T2PW_LLM_PROTEIN_FALLBACK": "0"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(map_ids, "_ai_protein_synonym_lookup", return_value=[]), patch.object(
        map_ids, "_map_protein_with_strategy", side_effect=strategy
    ), patch.object(map_ids.HttpClient, "get", side_effect=RuntimeError("network")):
        return map_ids.map_payload(payload, cache_path=tmp_path / "c.json", id_source="db", use_cache=False,
                                   allow_complex_wrapper_creation=True)["payload"]


def test_g9_family_modifier_is_converted_not_dropped(tmp_path):
    """wnt_ca: the id-less "PKC" modifier sat beside an identified enzyme, and the first
    rewrite dropped it as redundant (KI-007) before the family conversion could run."""

    payload = _payload(["PLCB1", "PKC", "CAMK2A"], [
        {"name": "DAG activates PKC", "entity_1": "DAG", "relationship": "activates", "entity_2": "PKC"},
    ])
    payload["entities"]["compounds"] = [{"name": "PIP2"}, {"name": "DAG"}]
    payload["processes"]["reactions"] = [{
        "name": "PLC reaction", "inputs": ["PIP2"], "outputs": ["DAG"],
        "enzymes": [{"protein": "PLCB1"}], "modifiers": [{"protein": "PKC"}, {"protein": "CAMK2A"}],
    }]
    mapped = _stage6_offline(payload, tmp_path, {
        "PLCB1": _resolved("1-phosphatidylinositol 4,5-bisphosphate phosphodiesterase beta-1", "PLCB1", "Q9NQ66"),
        "CAMK2A": _resolved("Calcium/calmodulin-dependent protein kinase type II subunit alpha", "CAMK2A", "Q9UQM7"),
    })
    assert any(c["name"] == "PKC" for c in mapped["entities"]["protein_complexes"]), \
        "the PKC family must become a complex, not be dropped as a redundant id-less modifier"
    assert mapped["processes"]["interactions"][0]["entity_2"] == "PKC"


def test_g9_partner_of_a_converted_family_is_wrapped(tmp_path):
    payload = _payload(["Fzd2", "G protein"], [
        {"name": "Fzd2 activates G protein", "entity_1": "Fzd2", "relationship": "activates", "entity_2": "G protein"},
    ])
    mapped = _stage6_offline(payload, tmp_path, {"Fzd2": _resolved("Frizzled-2", "FZD2", "Q14332")})

    [row] = mapped["processes"]["interactions"]
    assert row["entity_2"] == "G protein" and _complex(mapped, "G protein")
    assert row["entity_1"] == "Fzd2 complex", "Fzd2 has species and UniProt; it must not stay a bare endpoint"
    assert is_pathbank_unknown_protein(_protein(mapped, _complex(mapped, "G protein")["components"][0]["name"]))
