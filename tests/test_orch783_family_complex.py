"""ORCH-783 (KI-001, family/complex subset): a paper-named protein family becomes a
protein complex named after it.

G9 labels
- NEW ACCEPTANCE TESTS (new capability, no base failure claimed): the member-source
  ladder (paper -> PathBank -> UniProt family search), the detector, and the
  identity-admission route for family members.
- G9 BEHAVIOURAL PROOF (pre-existing behaviour changed; fails on base 5e1edb3f):
  ``test_g9_family_interaction_endpoint_gets_unknown_backed_complex`` -- at base the
  Unknown-backed complex served reaction enzymes only, so "DAG activates PKC" kept a
  bare, identity-less "PKC" protein and was omitted.

No test touches the network: ``HttpClient.get`` is faked or made to raise.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from t2pw.mapping import identity_admission, map_ids
from t2pw.pipeline.entity_identity import has_protein_external_identity, is_pathbank_unknown_protein

HUMAN = {"name": "Homo sapiens", "taxonomy_id": "9606"}


class _NoNetwork(RuntimeError):
    pass


def _payload(protein_names, interactions=()):
    return {
        "entities": {
            "species": [dict(HUMAN)],
            "proteins": [{"name": n, "organism": "Homo sapiens"} for n in protein_names],
            "compounds": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": [], "transports": [], "interactions": list(interactions)},
    }


class _FakeUniProt:
    """Answers TSV for the queries it knows, empty TSV for the rest; records calls."""

    def __init__(self, answers):
        self.answers = answers
        self.queries = []

    def get(self, url, params=None, **_):
        query = params["query"].split(" AND organism_id:")[0]
        self.queries.append(query)
        rows = self.answers.get(query, [])
        text = "Entry\tGene Names (primary)\tOrganism (ID)\n" + "".join(f"{a}\t{g}\t{t}\n" for a, g, t in rows)
        return SimpleNamespace(status_code=200, text=text)


class _Refuse:
    def get(self, *a, **k):
        raise _NoNetwork("network must not be reached")


def _complex(mapped, name):
    return next(c for c in mapped["entities"]["protein_complexes"] if c["name"] == name)


def _protein(mapped, name):
    return next((p for p in mapped["entities"]["proteins"] if p["name"] == name), None)


# ── NEW ACCEPTANCE: the detector ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["PKC", "Orai family proteins", "STIM1/2", "P2X2/P2X3", "VGCC", "5-HT3 receptor", "G protein"]
)
def test_new_acceptance_detector_catches_families(name):
    assert map_ids.is_family_or_complex_term(name)


@pytest.mark.parametrize("name", ["NLK", "Cdc42", "TCF4", "PIF1", "PI3K/AKT", "Ca2+/calmodulin", "Orai1"])
def test_new_acceptance_detector_leaves_single_proteins(name):
    assert not map_ids.is_family_or_complex_term(name)


# ── NEW ACCEPTANCE: the member-source ladder ─────────────────────────────────


def test_new_acceptance_paper_named_members():
    client = _FakeUniProt({
        "gene_exact:STIM1": [("Q13586", "STIM1", "9606")],
        "gene_exact:STIM2": [("Q9P246", "STIM2", "9606")],
    })
    mapped = _payload(["STIM1/2"])
    map_ids._apply_family_complexes(mapped, client=client, db=None)

    assert _protein(mapped, "STIM1/2") is None
    pc = _complex(mapped, "STIM1/2")
    assert [c["name"] for c in pc["components"]] == ["STIM1", "STIM2"]
    assert pc["generated"] is True and pc["species"] == "Homo sapiens"
    assert pc["mapping_meta"]["family_complex"]["note"] == "members named by the paper"
    for name in ("STIM1", "STIM2"):
        row = _protein(mapped, name)
        assert has_protein_external_identity(row) and row["taxonomy_id"] == "9606"
    assert all(not q.startswith("family:") for q in client.queries)  # stopped at source 1


class _FakePathBank:
    def __init__(self):
        self._species_meta = {}

    def _find_species_ids(self, organism):
        self._species_meta = {1: {"name": "Homo sapiens", "taxonomy_id": "9606"},
                              2: {"name": "Mus musculus", "taxonomy_id": "10090"}}
        return [1]

    def map_protein_complex(self, name, species):
        comp = lambda n, u, sid: {"name": n, "gene_name": n, "uniprot": u, "species_id": sid, "pathbank_protein_id": 7}
        return {
            "status": "mapped",
            "pathbank_complex_id": 42,
            "candidates": [{"name": "PKC"}],
            "components": [comp("PRKCA", "P17252", 1), comp("Prkcb", "P68404", 2),
                           comp("PRKCB", "P05771", 1), comp("PRKCG", "P05129", 1), comp("PRKCD", "Q05655", 1)],
        }


def test_new_acceptance_pathbank_members_capped_and_species_checked():
    mapped = _payload(["PKC"])
    map_ids._apply_family_complexes(mapped, client=_Refuse(), db=_FakePathBank())

    pc = _complex(mapped, "PKC")
    assert [c["name"] for c in pc["components"]] == ["PRKCA", "PRKCB", "PRKCG"]  # mouse Prkcb dropped, cap 3
    assert "PathBank complex 42" in pc["mapping_meta"]["family_complex"]["note"]
    assert "not the paper" in pc["mapping_meta"]["family_complex"]["note"]


def test_new_acceptance_uniprot_family_capped_prefix_filtered_species_checked():
    pkc = [("P17252", "PRKCA", "9606"), ("P41743", "PRKCI", "9606"), ("Q02156", "PRKCE", "9606"),
           ("Q16512", "PKN1", "9606"), ("Q05655", "PRKCD", "9606"), ("P05771", "PRKCB", "9606"),
           ("P20444", "Prkca", "10090"), ("Q16513", "PKN2", "9606"), ("P05129", "PRKCG", "9606")]
    client = _FakeUniProt({'family:"PKC subfamily"': pkc})
    mapped = _payload(["PKC"])
    report = map_ids._apply_family_complexes(mapped, client=client, db=None)

    pc = _complex(mapped, "PKC")
    assert [c["name"] for c in pc["components"]] == ["PRKCA", "PRKCB", "PRKCD"]
    assert client.queries == ['family:"PKC family"', 'family:"PKC subfamily"']
    note = pc["mapping_meta"]["family_complex"]["note"]
    assert note.startswith("members from UniProt family search") and note.endswith("not the paper")
    assert "first 3 of 6" in note  # 6 same-taxon PRKC hits; PKN and mouse dropped
    row = _protein(mapped, "PRKCA")
    assert row["uniprot_id"] == "P17252" and row[map_ids.FAMILY_COMPLEX_PROVENANCE_KEY]["family"] == "PKC"
    assert report["summary"]["family_complexes_with_members"] == 1


def test_new_acceptance_family_member_identity_abstains_in_admission():
    index = identity_admission.read_source_index(
        {identity_admission.SOURCE_INDEX_KEY: identity_admission.build_source_index("DAG activates PKC.")}
    )
    row = {"name": "PRKCA", "mapped_ids": {"uniprot": "P17252"},
           map_ids.FAMILY_COMPLEX_PROVENANCE_KEY: {"family": "PKC", "source": "uniprot"}}
    assert identity_admission.identity_support(row, index)["status"] == identity_admission.STATUS_NOT_EVALUATED
    row.pop(map_ids.FAMILY_COMPLEX_PROVENANCE_KEY)
    assert identity_admission.identity_support(row, index)["status"] == identity_admission.STATUS_UNSUPPORTED


def test_new_acceptance_single_named_protein_left_untouched():
    mapped = _payload(["NLK"], [{"entity_1": "TAK1", "entity_2": "NLK", "relationship": "activates"}])
    map_ids._apply_family_complexes(mapped, client=_Refuse(), db=None)
    assert _protein(mapped, "NLK") is not None
    assert mapped["entities"]["protein_complexes"] == []


# ── G9 BEHAVIOURAL PROOF: fails on base 5e1edb3f, passes at tip ──────────────


def test_g9_family_interaction_endpoint_gets_unknown_backed_complex(tmp_path):
    payload = _payload(["PKC"], [{"entity_1": "diacylglycerol", "entity_2": "PKC",
                                  "relationship": "activates", "name": "DAG activates PKC"}])
    payload["entities"]["compounds"] = [{"name": "diacylglycerol"}]
    env = {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}
    with patch.dict(os.environ, env), patch.object(
        map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)
    ), patch.object(map_ids, "_ai_protein_synonym_lookup", return_value=[]), patch.object(
        map_ids.HttpClient, "get", side_effect=_NoNetwork("network")
    ):
        mapped = map_ids.map_payload(payload, cache_path=tmp_path / "c.json", id_source="db", use_cache=False)["payload"]

    assert _protein(mapped, "PKC") is None, "the family must not stay a bare, identity-less protein endpoint"
    pc = _complex(mapped, "PKC")
    [component] = pc["components"]
    assert is_pathbank_unknown_protein(_protein(mapped, component["name"]))
