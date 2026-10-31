"""ORCH-797b (D-120 §§ 2-3): alias identity from UniProt's protein names; more family terms.

UniProt answers: ``tests/fixtures/orch797b/uniprot.json``, trimmed verbatim (``docs/pwml_recovery_sprint/evidence/
orch797x/build_fixture.py``) from the answers recorded once, live, for the ORCH-796 leg replays
(``evidence/orch797x/orch797b_uniprot_responses.json``). No test touches the network.

G9 -- CORRECTION of observable behaviour (``test_g9_*``), each fails on base ``aefa5b30`` on a value:
  * c-Jun -> P05412 (JUN): "c-Jun" is only in UniProt's protein name "Proto-oncogene c-Jun".
  * Kir2.1 -> P35561 (Kcnj2, mouse): only in the alternative name "Inward rectifier K(+) channel Kir2.1".
  * NARF -> Q8WVD3 (RNF138) when the paper says "NLK-associated RING finger protein (NARF)": the paper's
    wording separates it from Q9UHQ1 (prelamin A recognition factor, primary gene NARF).
  * TAK1 -> O43318 (MAP3K7) when the paper says "TGF-beta-activated protein kinase 1 (TAK1)".
  * InsP3R, SERCA, calmodulin, Dvl become family complexes with UniProt members.
PRESERVATION (``test_preservation_*``): passes at base and tip.
  * NARF never ships Q9UHQ1, and without separating wording it stays omitted.
  * Wnt5a, Fzd2 are not families; a plural with no UniProt family stays a protein.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from t2pw.mapping import map_ids

FIX = json.loads((Path(__file__).parent / "fixtures" / "orch797b" / "uniprot.json").read_text(encoding="utf-8"))
HS, MM = "Homo sapiens", "Mus musculus"


class _Search:
    """UniProt search stand-in: every search for ``name`` answers the given recorded entries."""

    def __init__(self, answers):
        self.answers = answers  # name (as queried) -> [accession, ...]

    def get(self, url, params=None, **_):
        query = str((params or {}).get("query") or "")
        if "uniprotkb/search" in url and str((params or {}).get("format")) == "json":
            for name, accessions in self.answers.items():
                if f'"{name}"' in query:
                    return SimpleNamespace(status_code=200, json=lambda a=accessions: {
                        "results": [json.loads(json.dumps(FIX["entries"][acc])) for acc in a]}, text="")
            return SimpleNamespace(status_code=200, json=lambda: {"results": []}, text="")
        return SimpleNamespace(status_code=404, json=lambda: {}, text="")


def _lookup(name, organism, accessions):
    client = _Search({name: accessions})
    with patch.object(map_ids, "lookup_literature_protein_aliases", return_value={"aliases": []}):
        result = map_ids.map_protein_uniprot(client, name, organism)
    return map_ids._apply_name_plausibility_gate(result, name, kind="protein", organism=organism)


def _payload(protein, quotes, organism=HS):
    return {
        "entities": {"species": [{"name": organism}], "proteins": [{"name": protein, "organism": organism}],
                     "compounds": [], "protein_complexes": []},
        "processes": {"reactions": [], "transports": [], "interactions": [
            {"entity_1": "X", "entity_2": protein, "name": f"X binds {protein}", "relationship": "binding",
             "evidence": quote} for quote in quotes]},
    }


def _map(payload, answers, tmp_path):
    with patch.dict(os.environ, {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}), \
            patch.object(map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)), \
            patch.object(map_ids, "_ai_protein_synonym_lookup", return_value=[]), \
            patch.object(map_ids, "lookup_literature_protein_aliases", return_value={"aliases": []}), \
            patch.object(map_ids.HttpClient, "get", lambda _s, url, params=None, headers=None: _Search(answers).get(url, params)):
        out = map_ids.map_payload(payload, cache_path=tmp_path / "cache.json", id_source="api", use_cache=False)
    return {p["name"]: (p.get("mapped_ids") or {}).get("uniprot") for p in out["payload"]["entities"]["proteins"]}


# ── § 2 alias identity ──────────────────────────────────────────────────────────


def test_g9_c_jun_is_jun_by_its_uniprot_protein_name():
    out = _lookup("c-Jun", HS, ["P05412"])
    assert out.get("status") == "mapped" and out["mapped_ids"]["uniprot"] == "P05412", out.get("identity_verdict", {}).get("reason")


def test_g9_kir2_1_is_kcnj2_by_its_alternative_name():
    out = _lookup("Kir2.1", MM, ["P35561"])
    assert out.get("status") == "mapped" and out["mapped_ids"]["uniprot"] == "P35561", out.get("identity_verdict", {}).get("reason")


NARF_SAYS = ["NLK can also associate with NLK-associated RING finger protein (NARF)",
             "NARF can ubiquitinate TCF/LEF for degradation by the proteasome"]


def test_g9_narf_is_rnf138_when_the_paper_says_ring_finger(tmp_path):
    ids = _map(_payload("NARF", NARF_SAYS), {"NARF": ["Q9UHQ1", "Q8WVD3"]}, tmp_path)
    assert ids["NARF"] == "Q8WVD3"


def test_g9_tak1_is_map3k7_when_the_paper_says_kinase(tmp_path):
    quotes = ["CAMKII stimulates TGF-β-activated protein kinase 1 (TAK1)"]
    ids = _map(_payload("TAK1", quotes), {"TAK1": ["P49116", "O43318"]}, tmp_path)
    assert ids["TAK1"] == "O43318"


def test_preservation_narf_never_ships_prelamin_a_recognition_factor(tmp_path):
    for quotes in (NARF_SAYS, [], ["NARF inhibits TCF/LEF"]):
        for order in (["Q9UHQ1", "Q8WVD3"], ["Q8WVD3", "Q9UHQ1"]):
            ids = _map(_payload("NARF", quotes), {"NARF": order}, tmp_path)
            assert ids["NARF"] != "Q9UHQ1"


def test_preservation_narf_without_separating_wording_stays_omitted(tmp_path):
    ids = _map(_payload("NARF", ["NARF inhibits TCF/LEF"]), {"NARF": ["Q9UHQ1", "Q8WVD3"]}, tmp_path)
    assert not ids["NARF"]


def test_preservation_a_verb_never_separates_two_genes(tmp_path):
    # "activated" is in MAP3K7's names only, but a verb is not support (lead ruling).
    ids = _map(_payload("TAK1", ["TAK1 is activated by CAMKII"]), {"TAK1": ["P49116", "O43318"]}, tmp_path)
    assert not ids["TAK1"]


def test_new_acceptance_the_paper_name_must_head_the_protein_name():
    """NEW ACCEPTANCE (helper added by ORCH-797b). From the ORCH-797b new-ids audit: a whole-word match
    anywhere shipped CAMKII -> CAMK2N1 ("CaMKII inhibitory protein alpha"), RhoA -> LAMTOR1
    ("p27Kip1-releasing factor from RhoA") and Frizzled -> SFRP4 ("Frizzled protein, human endometrium")."""

    yes = [("c-Jun", "Proto-oncogene c-Jun"), ("Kir2.1", "Inward rectifier K(+) channel Kir2.1"),
           ("TAK1", "Orphan nuclear receptor TAK1"), ("Inturned", "Protein inturned")]
    no = [("CaMKII", "CaMKII inhibitory protein alpha"), ("RhoA", "p27Kip1-releasing factor from RhoA"),
          ("Frizzled", "Frizzled protein, human endometrium"), ("Prickle", "Prickle-like protein 3"),
          ("actin", "Alpha-cardiac actin"),
          ("c-Jun", "C-Jun-amino-terminal kinase-interacting protein 1"), ("Kir2.1", "Inward rectifier K(+) channel Kir2.10")]
    assert all(map_ids._name_in_protein_names(q, [n]) for q, n in yes)
    assert not any(map_ids._name_in_protein_names(q, [n]) for q, n in no)


def test_preservation_only_the_papers_definition_separates_two_genes(tmp_path):
    """Words elsewhere in a sentence never decide (the ORCH-797b audit: RhoA -> LAMTOR1 via "factor")."""

    quotes = ["NARF is a RING finger E3 ligase that binds NLK"]  # the right words, but no "long form (NARF)"
    ids = _map(_payload("NARF", quotes), {"NARF": ["Q9UHQ1", "Q8WVD3"]}, tmp_path)
    assert not ids["NARF"]


# ── § 3 family terms ────────────────────────────────────────────────────────────


class _Tsv:
    """Recorded TSV answers by exact query; an unrecorded query answers no rows."""

    def get(self, url, params=None, **_):
        empty = "Entry\tGene Names (primary)\tOrganism (ID)\tProtein families\n"
        return SimpleNamespace(status_code=200, text=FIX["tsv"].get(params["query"], empty))


def _families(names):
    mapped = {"entities": {"species": [{"name": HS, "taxonomy_id": "9606"}],
                           "proteins": [{"name": n, "organism": HS} for n in names], "compounds": [],
                           "protein_complexes": []},
              "processes": {"reactions": [], "transports": [], "interactions": []}}
    map_ids._apply_family_complexes(mapped, client=_Tsv(), db=None)
    return ({c["name"]: [m["name"] for m in c["components"]] for c in mapped["entities"]["protein_complexes"]},
            {p["name"] for p in mapped["entities"]["proteins"]})


@pytest.mark.parametrize("name,members", [
    ("InsP3R", ["ITPR1", "ITPR2", "ITPR3"]),
    ("SERCA", ["ATP2A1", "ATP2A2", "ATP2A3"]),
    ("calmodulin", ["CALM1", "CALM2", "CALM3"]),
    ("Dvl", ["DVL1", "DVL2", "DVL3"]),
    # wnt_ca's own spelling: the paper's "long (SHORT)" form names the family twice.
    ("inositol-1,4,5-trisphosphate receptor (InsP3R)", ["ITPR1", "ITPR2", "ITPR3"]),
])
def test_g9_d120_family_terms_become_complexes_with_uniprot_members(name, members):
    complexes, _ = _families([name])
    assert complexes.get(name) == members


def test_g9_a_family_name_a_uniprot_search_narrowed_to_one_member_still_becomes_the_family():
    """wnt_ca: "inositol-1,4,5-trisphosphate receptor (InsP3R)" matched ITPR1 (Q14643) alone by UniProt name
    search (``top_unique_alias_candidate``); admission then withheld it, and the row was lost (KI-001)."""

    row = {"name": "calmodulin", "organism": HS, "species": HS, "uniprot_id": "P0DP23",
           "mapped_ids": {"uniprot": "P0DP23"},
           "mapping_meta": {"provider": "UniProt", "chosen_rule": "top_unique_alias_candidate",
                            "resolution": {"status": "matched", "order_step": "api_uniprot"}}}
    mapped = {"entities": {"species": [{"name": HS, "taxonomy_id": "9606"}], "proteins": [row], "compounds": [],
                           "protein_complexes": []},
              "processes": {"reactions": [], "transports": [], "interactions": []}}
    map_ids._apply_family_complexes(mapped, client=_Tsv(), db=None)
    complexes = {c["name"]: [m["name"] for m in c["components"]] for c in mapped["entities"]["protein_complexes"]}
    assert complexes.get("calmodulin") == ["CALM1", "CALM2", "CALM3"]


def test_g9_pkc_delta_is_one_protein_not_the_pkc_family():
    """At base "PKCδ" keyed as "pkc" (the Greek letter was deleted) and became the PKC family."""

    assert not map_ids.is_family_or_complex_term("PKCδ")


def test_preservation_single_named_proteins_are_not_families():
    for name in ("Wnt5a", "Fzd2", "TAK1", "c-Jun"):
        assert not map_ids.is_family_or_complex_term(name), name
    complexes, proteins = _families(["Wnt5a", "Fzd2", "Smads"])
    assert complexes == {} and proteins == {"Wnt5a", "Fzd2", "Smads"}
