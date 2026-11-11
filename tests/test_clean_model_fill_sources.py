"""Each source client against canned responses (real shapes)."""

import pathlib

from clean_model_fill_fakes import (PMC, RHEA_MANY, RHEA_ONE, TAX, FakeFetcher, chebi, ols, qg_anc, qg_search, qg_terms,
                                    uniprot, uniprot_search)
from t2pw.clean_model.fill import sources as S


def test_uniprot_entry_and_gene_search():
    f = FakeFetcher([("/P0AEJ2.json", None, uniprot())])
    e = S.uniprot_entry(f, "P0AEJ2")
    assert e["accession"] == "P0AEJ2" and e["gene_names"] == ["entC"] and e["ec_numbers"] == ["5.4.4.2"]
    assert e["taxon_id"] == "83333" and e["xrefs"]["kegg_gene"] == ["eco:b0593"] and e["xrefs"]["ensembl"] == ["ENSP00000269305.4", "AAC73696"]
    assert e["xrefs"]["ensembl_gene"] == ["ENST00000269305.9", "ENSG00000141510.18", "AAC73696"]  # transcript/gene ids apart from protein ids
    assert "Isochorismate mutase" in e["synonyms"] and "entC2" in e["synonyms"] and e["reviewed"]
    assert S.uniprot_entry(f, "P99999") is None
    f = FakeFetcher([("/search", None, uniprot_search(uniprot(), uniprot("A0A1", reviewed=False)))])
    r = S.uniprot_by_gene(f, "entC", "562")
    assert [x["accession"] for x in r] == ["P0AEJ2", "A0A1"]
    assert f.calls[0][1]["query"] == "gene_exact:entC AND taxonomy_id:562"
    assert S.uniprot_by_gene(f, "bad symbol) OR (x", "562") == [] and len(f.calls) == 1  # no injection, no call


def test_chebi_compound_shape():
    c = S.chebi_compound(FakeFetcher([("/15422/", None, chebi())]), "CHEBI:15422")
    assert c["formula"] == "C10H16N5O13P3" and c["inchikey"].startswith("ZKHQ") and c["smiles"]
    assert c["xrefs"]["kegg_compound"] == ["C00002"] and c["xrefs"]["hmdb"] == ["HMDB0000538"]
    assert c["xrefs"]["cas"] == ["56-65-5"]
    assert "iupac" not in c["synonyms"] and S.chebi_compound(FakeFetcher([]), "1") is None


def test_chebi_by_name_exact_only_and_charge_flag():
    f = FakeFetcher([("ols4", None, ols((29748, "chorismate(2-)", ["chorismate"]), (1, "chorismic acid", ["other thing"]),
                                          (2, "chorismate", []), (3, "x", ["Chorismate"])))])
    got = {h["id"]: h for h in S.chebi_by_name(f, "chorismate")}
    assert set(got) == {"29748", "2", "3"}  # CHEBI:1 matches neither label nor synonym exactly
    assert got["29748"]["match_type"] == "exact_synonym" and got["29748"]["charged"]
    assert got["2"]["match_type"] == "exact_label" and not got["2"]["charged"]
    assert f.calls[0][1]["exact"] == "true"


def test_go_cc_by_name_and_ancestors():
    f = FakeFetcher([("/search", None, qg_search(("GO:0042597", "periplasmic space", "cellular_component"),
                                                   ("GO:0043580", "periplasmic space organization", "biological_process"))),
                     ("/terms/GO:0042597", None, qg_terms(("GO:0042597", "periplasmic space", ["periplasm"])))])
    assert S.go_cc_by_name(f, "periplasm") == [{"id": "GO:0042597", "name": "periplasmic space", "match_type": "exact_synonym"}]
    assert S.go_cc_by_name(f, "periplasmic spac") == []
    f = FakeFetcher([("/ancestors", {"relations": "part_of"}, qg_anc(GO_0005829=["GO:0005737", "GO:0005622"]))])  # part_of only, not is_a
    assert S.go_ancestors(f, ["GO:0005829"]) == {"GO:0005829": ["GO:0005737", "GO:0005622"]}
    assert f.calls[0][1] == {"relations": "part_of"}


def test_ncbi_clients_and_rhea():
    assert S.ncbi_taxids(FakeFetcher([("esearch", None, TAX)]), "Escherichia coli") == ["562"]
    a = S.ncbi_article(FakeFetcher([("esummary", {"db": "pmc"}, PMC)]), pmcid="PMC11983282")
    assert a["year"] == 2025 and a["doi"] == "10.1093/nar/gkaf224" and a["pmid"] == "40210244"
    assert a["authors"][0] == "Kim NM"
    assert S.ncbi_article(FakeFetcher([]), pmcid="PMC1") is None and S.ncbi_article(FakeFetcher([])) is None
    assert S.rhea_by_ec(FakeFetcher([("rhea-db", None, RHEA_ONE)]), "5.4.4.2") == ["RHEA:18985"]
    assert len(S.rhea_by_ec(FakeFetcher([("rhea-db", None, RHEA_MANY)]), "5.4.4.2")) == 2
    assert S.rhea_by_ec(FakeFetcher([]), "not an ec") == []


def test_no_kegg_request_target_in_package():
    for p in pathlib.Path(S.__file__).parent.glob("*.py"):
        assert "kegg.jp" not in p.read_text(encoding="utf-8")
