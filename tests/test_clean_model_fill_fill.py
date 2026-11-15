"""fill_model rules: provenance, never overwrite, refusals, suggestions-not-ids, to_payload invariant."""

import copy

from clean_model_fill_fakes import (PMC, RHEA_MANY, RHEA_ONE, TAX, FakeFetcher, chebi, ols, qg_anc, qg_search, qg_terms,
                                    uniprot, uniprot_search)
from t2pw.clean_model import build_from_payload, to_payload
from t2pw.clean_model.fill import fill_model
from t2pw.clean_model.fill.cache import Cache
from t2pw.clean_model.fill.http import FetchError

TODAY = "2026-11-10"
PROV = [{"stage": "paper_extraction", "origin": "paper_stated"}]


def _payload():
    return {
        "entities": {
            "species": [{"name": "Escherichia coli", "provenance_lineage": PROV}],
            "subcellular_locations": [{"name": "cytosol"}, {"name": "cytoplasm"}, {"name": "plasma membrane"}],
            "compounds": [
                {"name": "ATP", "chebi_id": "15422", "kegg_id": "C00002", "provenance": "extracted"},
                {"name": "ADP", "chebi_id": "16761", "kegg_id": "C99999", "provenance": "extracted"},
                {"name": "chorismate", "provenance": "extracted"},
                {"name": "zinc", "provenance": "extracted"},
                {"name": "DMAPP", "chebi_id": "15377", "provenance": "extracted"},
                {"name": "prenyl diphosphate", "provenance": "extracted"},
                {"name": "mystery", "provenance": "extracted"},
                {"name": "has kegg", "kegg_id": "C00001", "provenance": "extracted"},
            ],
            "proteins": [
                {"name": "EntC", "uniprot": "P0AEJ2", "gene_name": "entC", "taxonomy_id": "83333", "provenance": "extracted"},
                {"name": "EntX", "uniprot": "P12345", "gene_name": "entX", "ec_number": "1.1.1.1", "provenance": "extracted"},
                {"name": "FooA", "gene_name": "fooA", "taxonomy_id": "562", "provenance": "extracted"},
                {"name": "Kinase family", "gene_name": "kin", "taxonomy_id": "562", "provenance": "extracted"},
            ],
            "protein_complexes": [{"name": "EntC dimer", "components": ["EntC", "EntC"]}],
        },
        "biological_states": [],
        "processes": {"reactions": [
            {"name": "EntC rxn", "inputs": ["chorismate"], "outputs": ["ADP"],
             "enzymes": [{"entity": "EntC", "entity_type": "protein", "role": "catalyst"}]},
            {"name": "No catalyst", "inputs": ["ATP"], "outputs": ["ADP"], "enzymes": []},
        ]},
    }


def _routes(rhea=RHEA_ONE):
    adp = chebi(16761, "ADP", kegg="C00008", hmdb="HMDB0001341")
    return [
        ("/P0AEJ2.json", None, uniprot()),
        ("/P12345.json", None, uniprot("P12345", gene="entX", taxon=562, ec="2.2.2.2")),
        ("/search", {"query": "gene_exact:fooA AND taxonomy_id:562"},
         uniprot_search(uniprot("Q1", gene="fooA", reviewed=False), uniprot("Q2", gene="fooA"))),
        ("/compound/15422/", None, chebi()),
        ("/compound/16761/", None, adp),
        ("/compound/29748/", None, chebi(29748, "chorismate(2-)", "C10H8O6", "WTFXTQVDAKGDEY-HTQZYQBOSA-L", "C00251", "HMDB0000975")),
        ("ols4", {"q": "chorismate"}, ols((29748, "chorismate(2-)", ["chorismate"]))),
        ("ols4", {"q": "zinc"}, ols((30185, "zinc(0)", ["zinc"]), (27363, "zinc atom", ["zinc"]))),
        ("ols4", {"q": "prenyl diphosphate"}, ols((15377, "prenyl diphosphate", []))),
        ("ols4", {"q": "mystery"}, ols()),
        ("QuickGO/services/ontology/go/search", {"query": "cytosol"}, qg_search(("GO:0005829", "cytosol", "cellular_component"))),
        ("QuickGO/services/ontology/go/search", {"query": "cytoplasm"}, qg_search(("GO:0005737", "cytoplasm", "cellular_component"))),
        ("QuickGO/services/ontology/go/search", {"query": "plasma membrane"}, qg_search(("GO:0005886", "plasma membrane", "cellular_component"))),
        ("/ancestors", {"relations": "part_of"}, qg_anc(GO_0005829=["GO:0005737", "GO:0005622"], GO_0005737=["GO:0005622"], GO_0005886=["GO:0016020"])),
        ("/terms/GO:0005829", None, qg_terms(("GO:0005829", "cytosol", []))),
        ("/terms/GO:0005737", None, qg_terms(("GO:0005737", "cytoplasm", []))),
        ("/terms/GO:0005886", None, qg_terms(("GO:0005886", "plasma membrane", []))),
        ("esummary", {"db": "pmc"}, PMC),
        ("esearch", None, TAX),
        ("rhea-db", {"query": "ec:5.4.4.2"}, rhea),
    ]


def _run(rhea=RHEA_ONE, payload=None, **kw):
    p = payload or _payload()
    m = build_from_payload(p, {"pmcid": "PMC11983282", "title": "A different title", "source": "t"})
    rep = fill_model(m, FakeFetcher(_routes(rhea)), Cache(None), TODAY, **kw)
    return p, m, rep


def _items(m, **kw):
    return [i for i in m.db_layer["items"] if all(i.get(k) == v for k, v in kw.items())]


def test_to_payload_unchanged_after_fill():
    p, m, rep = _run()
    assert to_payload(m) == p and rep.fields_filled > 10
    before = copy.deepcopy(to_payload(m))
    fill_model(m, FakeFetcher(_routes()), Cache(None), TODAY)  # a second fill changes nothing either
    assert to_payload(m) == before == p


def test_every_filled_value_has_db_provenance_and_date():
    _, m, _ = _run()
    for i in m.db_layer["items"]:
        pv = i["prov"]
        assert pv["label"] == "db" or pv["label"] == "inferred"
        assert pv["source"] and pv["retrieved"] == TODAY and pv["match_type"]
        assert pv["match_type"].split("+")[0] in {"accession", "exact_label", "exact_synonym", "cross_reference",
                                                  "inchikey", "gene_symbol_taxon", "catalyst_ec"}
    cm = {c.name.value: c for c in m.compounds}
    assert cm["ATP"].inchikey.value.startswith("ZKHQ") and cm["ATP"].inchikey.prov.label == "db"
    assert cm["ATP"].formula.prov.accession == "CHEBI:15422" and cm["ATP"].formula.prov.match_type == "accession"


def test_frozen_values_never_overwritten_conflict_is_recorded():
    _, m, _ = _run()
    adp = next(c for c in m.compounds if c.name.value == "ADP")
    assert adp.kegg_id.value == "C99999" and adp.kegg_id.prov.frozen
    cf = [c for c in m.db_layer["conflicts"] if c["entity"] == "ADP" and c["field"] == "kegg_id"]
    assert cf and cf[0]["frozen"] == "C99999" and cf[0]["db"] == ["C00008"]
    assert not _items(m, entity="ADP", field="kegg_id")
    atp = next(c for c in m.compounds if c.name.value == "ATP")  # equal xref: nothing added, no conflict
    assert not _items(m, entity="ATP", field="kegg_id") and atp.kegg_id.value == "C00002"
    assert _items(m, entity="ATP", field="hmdb_id")[0]["value"] == "HMDB0000538"
    ex = next(p for p in m.proteins if p.name.value == "EntX")  # EC differs from UniProt
    assert ex.ec_number.value == "1.1.1.1"
    assert any(c["entity"] == "EntX" and c["field"] == "ec_number" for c in m.db_layer["conflicts"])
    assert not _items(m, entity="EntX", field="ec_number")


def test_name_lookup_single_match_notes_charged_form():
    _, m, _ = _run()
    it = _items(m, entity="chorismate", field="chebi_id")
    assert len(it) == 1 and it[0]["value"] == "CHEBI:29748" and it[0]["prov"]["match_type"] == "exact_synonym+charged_form"
    ch = next(c for c in m.compounds if c.name.value == "chorismate")
    assert ch.chebi_id.absent and ch.inchikey.value.startswith("WTFX")  # id stays out of the payload slot


def test_ambiguous_name_is_a_suggestion_not_an_id():
    _, m, rep = _run()
    assert not _items(m, entity="zinc", field="chebi_id")
    s = [x for x in m.db_layer["suggestions"] if x["entity"] == "zinc"]
    assert s and {c["id"] for c in s[0]["candidates"]} == {"CHEBI:30185", "CHEBI:27363"} and s[0]["prov"]["label"] == "inferred"
    assert rep.lookups_ambiguous >= 1
    zn = next(c for c in m.compounds if c.name.value == "zinc")
    assert zn.chebi_id.absent


def test_collision_with_other_entity_is_refused():
    _, m, rep = _run()
    assert not _items(m, entity="prenyl diphosphate", field="chebi_id")
    r = [x for x in m.db_layer["refused"] if x["entity"] == "prenyl diphosphate"]
    assert r and r[0]["reason"] == "collision" and r[0]["detail"]["used_by"] == ["DMAPP"] and rep.refused["collision"] == 1


def test_no_match_and_compounds_with_other_ids_are_not_looked_up():
    _, m, rep = _run()
    assert not _items(m, entity="mystery") and not _items(m, entity="has kegg")
    f = FakeFetcher(_routes())
    fill_model(build_from_payload(_payload(), {}), f, Cache(None), TODAY)
    assert not any(c[1].get("q") == "has kegg" for c in f.calls)
    assert rep.lookups_no_match.get("chebi", 0) >= 1


def test_protein_with_accession_adds_names_ec_taxon_and_carried_xrefs():
    _, m, _ = _run()
    fld = {i["field"] for i in _items(m, entity="EntC")}
    assert {"protein_name", "synonym", "ec_number", "taxon_id", "kegg_gene", "ensembl", "ensembl_gene"} <= fld
    assert _items(m, entity="EntC", field="ensembl")[0]["value"] == "ENSP00000269305.4"
    kg = _items(m, entity="EntC", field="kegg_gene")[0]
    assert kg["value"] == "eco:b0593" and kg["prov"]["source"] == "UniProt" and kg["prov"]["match_type"] == "cross_reference"
    assert not _items(m, entity="EntC", field="gene_name")  # frozen gene equals UniProt: nothing to add


def test_protein_without_accession_gets_suggestions_only():
    _, m, rep = _run()
    foo = next(p for p in m.proteins if p.name.value == "FooA")
    assert foo.uniprot.absent and not _items(m, entity="FooA", field="uniprot")
    assert [c["accession"] for c in foo.inferred_suggestion.value] == ["Q2", "Q1"]  # reviewed first
    assert foo.inferred_suggestion.prov.label == "inferred" and foo.inferred_suggestion.prov.match_type == "gene_symbol_taxon"
    kin = next(p for p in m.proteins if p.name.value == "Kinase family")
    assert kin.inferred_suggestion.absent and kin.uniprot.absent  # class names are skipped
    for c in m.complexes:
        assert not _items(m, entity=c.name.value)


def test_locations_go_ids_parent_and_membrane():
    _, m, _ = _run()
    loc = {x.name.value: x for x in m.locations}
    assert loc["cytosol"].go_id.value == "GO:0005829" and loc["cytosol"].go_id.prov.match_type == "exact_label"
    assert loc["cytosol"].parent.value == "cytoplasm"  # nearest model ancestor
    assert loc["cytoplasm"].parent.absent and loc["plasma membrane"].parent.absent  # unknown pairs stay un-nested
    assert loc["plasma membrane"].kind.value == "membrane" and loc["plasma membrane"].kind.prov.label == "inferred"
    assert loc["cytosol"].kind.absent


def test_reaction_ec_and_rhea_not_inferred_by_default():
    _, m, _ = _run()
    assert all(r.ec_number.absent and r.rhea_id.absent for r in m.reactions)


def test_reaction_ec_and_rhea_only_when_unique():
    _, m, _ = _run(infer_reaction_ec=True)
    rx = next(r for r in m.reactions if r.name.value == "EntC rxn")
    assert rx.ec_number.value == "5.4.4.2" and rx.ec_number.prov.label == "inferred" and rx.ec_number.prov.match_type == "catalyst_ec"
    assert rx.rhea_id.value == "RHEA:18985" and rx.rhea_id.prov.label == "inferred"
    assert next(r for r in m.reactions if r.name.value == "No catalyst").ec_number.absent
    _, m2, _ = _run(RHEA_MANY, infer_reaction_ec=True)
    rx2 = next(r for r in m2.reactions if r.name.value == "EntC rxn")
    assert rx2.ec_number.value == "5.4.4.2" and rx2.rhea_id.absent


def test_pathway_citation_filled_conflict_noted_and_taxon():
    _, m, _ = _run()
    pw = m.pathway
    assert pw.doi.value == "10.1093/nar/gkaf224" and pw.doi.prov.label == "db" and pw.year.value == 2025
    assert pw.pmid.value == "40210244" and pw.authors.value[0] == "Kim NM"
    assert pw.title.value == "A different title" and pw.title.prov.frozen  # frozen title kept
    assert any(c["kind"] == "pathway" and c["field"] == "title" for c in m.db_layer["conflicts"])
    assert _items(m, kind="species", field="taxonomy_id")[0]["value"] == "562"
    assert pw.taxon_id.value == "562"


def test_licences_for_sources_used():
    _, m, _ = _run(infer_reaction_ec=True)
    text = " ".join(f.value for f in m.pathway.licences)
    for name in ("UniProt", "ChEBI", "Rhea", "Gene Ontology", "NCBI"):
        assert name in text
    n = len(m.pathway.licences)
    fill_model(m, FakeFetcher(_routes()), Cache(None), TODAY, infer_reaction_ec=True)
    assert len(m.pathway.licences) == n  # not duplicated


def test_report_counts_and_is_json_serialisable():
    import json
    _, m, rep = _run()
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["fields_filled"] == len(m.db_layer["items"]) and d["ids_filled"] > 0
    assert sum(d["by_source"].values()) == d["fields_filled"] and sum(d["by_match_type"].values()) == d["fields_filled"]
    assert d["conflicts"] == len(m.db_layer["conflicts"]) and d["requests"] > 0 and d["cache_hits"] >= 0


def test_cache_hits_on_second_fill_and_fetch_failure_counted():
    m = build_from_payload(_payload(), {"pmcid": "PMC11983282"})
    cache = Cache(None)
    r1 = fill_model(m, FakeFetcher(_routes()), cache, TODAY)
    f2 = FakeFetcher([])
    r2 = fill_model(build_from_payload(_payload(), {"pmcid": "PMC11983282"}), f2, cache, TODAY)
    assert f2.calls == [] and r2.requests == 0 and r2.cache_hits == r1.requests + r1.cache_hits
    bad = FakeFetcher([("uniprotkb", None, FetchError("HTTP 500")), ("chebi", None, FetchError("x"))])
    r3 = fill_model(build_from_payload(_payload(), {}), bad, Cache(None), TODAY, sources=("uniprot", "chebi"))
    assert r3.lookups_failed.get("uniprot", 0) > 0 and r3.fields_filled == 0


def test_cached_entry_keeps_its_own_retrieval_date_in_provenance():
    cache = Cache(None)
    fill_model(build_from_payload(_payload(), {"pmcid": "PMC11983282"}), FakeFetcher(_routes()), cache, "2026-01-02")
    assert all(e["retrieved"] == "2026-01-02" for e in cache._mem.values())
    m = build_from_payload(_payload(), {"pmcid": "PMC11983282"})
    rep = fill_model(m, FakeFetcher([]), cache, TODAY)  # everything served from the old cache
    assert rep.requests == 0 and rep.cache_hits > 0 and m.db_layer["items"]
    for i in m.db_layer["items"]:
        assert i["prov"]["retrieved"] == "2026-01-02", i
    cm = {c.name.value: c for c in m.compounds}
    assert cm["ATP"].inchikey.prov.retrieved == "2026-01-02"
    # a live fetch is stamped with today's date; a mixed lookup takes the oldest date it used
    m2 = build_from_payload(_payload(), {})
    fill_model(m2, FakeFetcher(_routes()), Cache(None), TODAY)
    assert {i["prov"]["retrieved"] for i in m2.db_layer["items"]} == {TODAY}


def test_licence_order_is_sorted_not_set_order():
    _, m, _ = _run(infer_reaction_ec=True)
    keys = [f.prov.source for f in m.pathway.licences]
    assert len(keys) >= 4 and keys == sorted(keys)


def test_non_json_200_is_a_counted_failure_not_an_abort():
    from t2pw.clean_model.fill.http import Throttle, UrllibFetcher
    uf = UrllibFetcher(Throttle(0, sleep=lambda s: None), retries=0, opener=lambda u, h, t: (200, {}, b"<html>busy</html>"))
    m = build_from_payload(_payload(), {})
    rep = fill_model(m, uf, Cache(None), TODAY, sources=("uniprot", "chebi", "go"))
    assert rep.lookups_failed.get("uniprot", 0) > 0 and rep.lookups_failed.get("chebi", 0) > 0 and rep.fields_filled == 0
