"""D-W21(d): unsettled DNA/RNA type is looked up (NCBI Gene, RNAcentral) or flagged unknown. Fakes only: no network."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from test_final_payload_writer_fixes import _build, _na_types, _payload  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.clean_model.fill.http import FetchError  # noqa: E402
from t2pw.service import final_payload as fp  # noqa: E402
from t2pw.service import final_payload_nucleic as fn  # noqa: E402

TODAY = "2026-11-13"


def G(gid, name, typ, aliases="", taxid=511145, org="Escherichia coli str. K-12 substr. MG1655", status=None):
    return {"id": str(gid), "name": name, "aliases": aliases, "type": typ, "taxid": taxid, "org": org, "status": status}


class Fake:
    """NCBI Gene (esearch / esummary / efetch) and the EBI search for RNAcentral, built from plain dicts."""

    def __init__(self, genes=(), rnac=None, fail=None):
        self.genes, self.rnac, self.fail, self.calls = list(genes), dict(rnac or {}), fail, []

    def _go(self, url, params):
        self.calls.append((url, dict(params or {})))
        if self.fail == url.split("/")[-1].split("?")[0] or self.fail == "all":
            raise FetchError("boom")

    def get_json(self, url, params=None):
        self._go(url, params)
        if url.endswith("esearch.fcgi"):
            name = re.match(r'"(.*)"\[Gene Name\]', params["term"]).group(1).casefold()
            ids = [g["id"] for g in self.genes if name in [g["name"].casefold()] + [a.strip().casefold() for a in g["aliases"].split(",")]]
            return {"esearchresult": {"count": str(len(ids)), "idlist": ids}}
        if url.endswith("esummary.fcgi"):
            return {"result": {g["id"]: {"uid": g["id"], "name": g["name"], "otheraliases": g["aliases"],
                                         "organism": {"scientificname": g["org"], "taxid": g["taxid"]}}
                               for g in self.genes if g["id"] in params["id"].split(",")}}
        if url == fn.EBI_SEARCH:
            name = re.match(r'gene:"(.*)" AND TAXONOMY:(\d+)', params["query"])
            return {"hitCount": 1, "entries": [e for e in self.rnac.get((name.group(1).casefold(), name.group(2)), [])]}
        raise AssertionError(url)

    def get_text(self, url, params=None):
        self._go(url, params)
        g = next(g for g in self.genes if g["id"] == params["id"])
        st = '<Gene-track_status value="%s">1</Gene-track_status>' % g["status"] if g["status"] else ""
        return '<Entrezgene><Entrezgene_track-info><Gene-track>%s</Gene-track></Entrezgene_track-info><Entrezgene_type value="%s"/></Entrezgene>' % (st, g["type"])


def E(urs, taxid, genes, rna_type="ncRNA", desc="x RNA"):
    return {"id": f"{urs}_{taxid}", "source": "rnacentral", "fields": {"gene": genes, "rna_type": [rna_type], "description": [desc], "active": ["Active"]}}


def P(*rows, taxid="562", **extra):
    p = {"entities": {"species": [{"name": "Escherichia coli", "taxonomy_id": taxid}] if taxid else [],
                      "nucleic_acids": [dict(r) for r in rows]}}
    p.update(extra)
    return p


def run(p, fake, text="", **kw):
    return fn.apply_nucleic_types(p, text, fetcher=fake, cache=Cache(None), today=TODAY, **kw)


def row(p, i=0):
    return p["entities"]["nucleic_acids"][i]


RYHB = [G(1, "ryhB", "ncRNA", "b4451, ECK3426"), G(2, "ryhB", "miscRNA", "Y75_s0048", 316407, "E. coli W3110", status="discontinued"),
        G(3, "ryhB", "miscRNA", "", 316407, "E. coli W3110", status="discontinued")]


def test_ncbi_gene_rna_gene_resolves_with_provenance() -> None:
    p = P({"name": "RyhB"})
    rep = run(p, Fake(RYHB))
    r = row(p)
    assert r["nucleotide_type"] == "RNA" and r["nucleotide_type_status"] == "resolved"
    s = r["nucleotide_type_source"]
    assert s["source"] == "NCBI Gene" and s["accession"] == "GeneID:1" and s["match_type"] == "exact_symbol"
    assert s["retrieved"] == TODAY and s["species_taxid"] == "562" and "gene type ncRNA" in s["quote"]
    assert s["candidates"] == ["GeneID:1"] and s["confirmed_by"] == []  # the two discontinued strain records are dropped
    e = p["final_payload_fill"]["entries"][0]
    assert (e["field"], e["value"], e["accession"], e["source"]) == ("nucleotide_type", "RNA", "GeneID:1", "NCBI Gene")
    assert rep["lookup_resolved"] == 1 and rep["lookup_by_source"] == {"NCBI Gene": 1} and rep["unknown"] == 0
    assert p["final_payload_fill"]["by_field"] == {"nucleic_acid.nucleotide_type": 1}


def test_protein_coding_alias_is_dna_and_symbol_beats_alias() -> None:
    genes = [G(7, "CRD1", "protein-coding", "AT3G56940, CHL27", 3702, "Arabidopsis thaliana")]
    p = P({"name": "CHL27"}, taxid="3702")
    run(p, Fake(genes))
    assert row(p)["nucleotide_type"] == "DNA" and row(p)["nucleotide_type_source"]["match_type"] == "exact_alias"


def test_name_is_exact_not_a_prefix_or_substring() -> None:
    p = P({"name": "ryh"})
    rep = run(p, Fake(RYHB))
    assert row(p)["nucleotide_type_status"] == "unknown" and rep["lookup_codes"] == {"not_found": 1} and "nucleotide_type" not in row(p)


def test_two_live_records_are_ambiguous_even_when_they_agree() -> None:
    p = P({"name": "RyhB"})
    rep = run(p, Fake([G(1, "ryhB", "ncRNA"), G(2, "ryhB", "ncRNA", taxid=316407)]))
    assert row(p)["nucleotide_type_unknown_lookup"] == "ambiguous" and rep["unknown"] == 1 and "nucleotide_type" not in row(p)
    p = P({"name": "Foo1"}, taxid="3702")  # an alias shared by two genes
    run(p, Fake([G(1, "AAA1", "protein-coding", "Foo1", 3702), G(2, "BBB2", "protein-coding", "Foo1", 3702)]))
    assert row(p)["nucleotide_type_unknown_lookup"] == "ambiguous"


@pytest.mark.parametrize("typ", ["pseudo", "other", "unknown", "biological-region", None])
def test_undecisive_gene_type_is_unresolved(typ) -> None:
    p = P({"name": "Foo1"}, taxid="3702")
    run(p, Fake([G(10, "Foo1", typ, taxid=3702)]))
    assert row(p)["nucleotide_type_unknown_lookup"] == "gene_type_not_decisive" and "nucleotide_type" not in row(p)


@pytest.mark.parametrize("typ,want", [("protein-coding", "DNA"), ("ncRNA", "RNA"), ("tRNA", "RNA"), ("rRNA", "RNA"), ("snRNA", "RNA"),
                                      ("snoRNA", "RNA"), ("scRNA", "RNA"), ("miscRNA", "RNA")])
def test_gene_type_map(typ, want) -> None:
    p = P({"name": "Foo1"}, taxid="3702")
    run(p, Fake([G(10, "Foo1", typ, taxid=3702)]))
    assert row(p)["nucleotide_type"] == want


def test_rnacentral_resolves_when_ncbi_has_nothing_and_needs_the_exact_taxid() -> None:
    rn = {("mir9", "562"): [E("URS0000AAAA", 562, ["MIR9", "miR9"])]}
    p = P({"name": "miR9"})
    rep = run(p, Fake(rnac=rn))
    s = row(p)["nucleotide_type_source"]
    assert row(p)["nucleotide_type"] == "RNA" and s["source"] == "RNAcentral" and s["accession"] == "URS0000AAAA"
    assert s["match_type"] == "exact_gene_name" and "rna_type ncRNA" in s["quote"] and rep["lookup_by_source"] == {"RNAcentral": 1}
    # a record of another taxid, or whose gene field only contains the name inside a longer one, is no hit
    rn = {("mir9", "562"): [E("URS0000AAAA", 511145, ["miR9"]), E("URS0000BBBB", 562, ["miR9-5p"])]}
    p = P({"name": "miR9"})
    run(p, Fake(rnac=rn))
    assert row(p)["nucleotide_type_unknown_lookup"] == "not_found"


def test_rnacentral_several_sequences_of_one_gene_name_are_still_rna() -> None:
    rn = {("mir9", "562"): [E("URS0000AAAA", 562, ["miR9"]), E("URS0000BBBB", 562, ["miR9"])]}
    p = P({"name": "miR9"})
    run(p, Fake(rnac=rn))
    assert row(p)["nucleotide_type"] == "RNA" and row(p)["nucleotide_type_source"]["candidates"] == ["URS0000AAAA", "URS0000BBBB"]


def test_sources_that_disagree_stay_unknown() -> None:
    genes = [G(5, "Foo1", "protein-coding", taxid=562, org="Escherichia coli")]
    p = P({"name": "Foo1"})
    run(p, Fake(genes, {("foo1", "562"): [E("URS0000AAAA", 562, ["Foo1"])]}))
    assert row(p)["nucleotide_type_unknown_lookup"] == "sources_disagree" and "nucleotide_type" not in row(p)


def test_both_sources_agreeing_use_ncbi_first() -> None:
    p = P({"name": "RyhB"})
    run(p, Fake(RYHB, {("ryhb", "562"): [E("URS0000AAAA", 562, ["ryhB"])]}))
    s = row(p)["nucleotide_type_source"]
    assert s["source"] == "NCBI Gene" and row(p)["nucleotide_type"] == "RNA" and s["confirmed_by"] == ["RNAcentral URS0000AAAA"]


@pytest.mark.parametrize("taxid,code", [(None, "species_unknown"), ("not-a-number", "species_unknown")])
def test_species_unknown_skips_the_lookup(taxid, code) -> None:
    fake = Fake(RYHB)
    p = P({"name": "RyhB"}, taxid=taxid)
    run(p, fake)
    assert fake.calls == [] and row(p)["nucleotide_type_unknown_lookup"] == code


def test_two_species_skips_the_lookup() -> None:
    fake = Fake(RYHB)
    p = P({"name": "RyhB"})
    p["entities"]["species"].append({"name": "Homo sapiens", "taxonomy_id": "9606"})
    run(p, fake)
    assert fake.calls == [] and row(p)["nucleotide_type_unknown_lookup"] == "species_ambiguous"
    p["entities"]["species"].append({"name": "Mus", "taxonomy_id": "562"})  # the same id twice is one species
    p["entities"]["species"].pop(1)
    run(p, fake)
    assert fake.calls and row(p)["nucleotide_type"] == "RNA"


def test_names_that_are_not_one_token_symbols_are_not_queried() -> None:
    fake = Fake(RYHB)
    rows = [{"name": "PORA promoter"}, {"name": "ALAD1/HEMB1"}, {"name": "-35 box"}, {"name": "ab"}]
    p = P(*rows)
    run(p, fake, text="nothing")
    assert fake.calls == [] and [r["nucleotide_type_unknown_lookup"] for r in p["entities"]["nucleic_acids"]] == ["no_symbol_name"] * 4
    p = P({"name": "Long name", "synonyms": ["RyhB"]})  # a synonym that is a symbol is queried
    run(p, fake)
    assert row(p)["nucleotide_type"] == "RNA"


def test_fetch_error_flags_this_and_later_rows_but_never_raises() -> None:
    fake = Fake(RYHB, fail="esearch.fcgi")
    p = P({"name": "RyhB"}, {"name": "Other1"})
    rep = run(p, fake)
    assert [r["nucleotide_type_unknown_lookup"] for r in p["entities"]["nucleic_acids"]] == ["lookup_error", "lookup_skipped_after_error"]
    assert rep["unknown"] == 2 and rep["error"] is None and len(fake.calls) == 1  # the first failure stopped the lookups


def test_unknown_flag_reasons_and_never_a_type() -> None:
    p = P({"name": "Aaa1"}, {"name": "Bbb"}, {"name": "Zzz9"}, {"name": "DNA polymerase"}, {"name": "Ddd"})
    text = "The Aaa1 gene mRNA rose. Bbb was measured. The DNA polymerase gene. The Ddd transcript rose."
    rep = run(p, Fake(), text=text)
    rows = p["entities"]["nucleic_acids"]
    assert [r["nucleotide_type_unknown_reason"] for r in rows[:4]] == ["conflicting", "no_cue", "not_mentioned", "protein_like"]
    assert all(r["nucleotide_type_status"] == "unknown" and "nucleotide_type" not in r and "nucleotide_type_paper" not in r for r in rows[:4])
    assert rows[3]["nucleotide_type_unknown_lookup"] == "not_attempted" and rows[0]["nucleotide_type_unknown_lookup"] == "not_found"
    assert rows[4]["nucleotide_type_paper"] == "RNA" and rows[4]["nucleotide_type_status"] == "resolved"
    assert rep["unknown"] == 4 and rep["unknown_by_reason"] == {"conflicting": 1, "no_cue": 1, "not_mentioned": 1, "protein_like": 1}
    assert rep["resolved"] == 1 and rep["lookup_resolved"] == 0
    blk = p["final_payload_fill"]
    assert blk["by_field"] == {"nucleic_acid.nucleotide_type_paper": 1, "nucleic_acid.nucleotide_type_status": 4}
    unk = [e for e in blk["entries"] if e["field"] == "nucleotide_type_status"]
    assert len(unk) == 4 and {e["value"] for e in unk} == {"unknown"} and {e["match_type"] for e in unk} == \
        {"conflicting", "no_cue", "not_mentioned", "protein_like"}


def test_no_paper_text_still_looks_up_and_flags() -> None:
    p = P({"name": "RyhB"}, {"name": "Zzz9"})
    rep = run(p, Fake(RYHB), text=None)
    assert row(p)["nucleotide_type"] == "RNA" and row(p, 1)["nucleotide_type_unknown_reason"] == "no_paper_text"
    assert rep["no_paper_text"] == 2


def test_paper_resolved_ids_and_name_settled_rows_are_never_looked_up() -> None:
    fake = Fake(RYHB)
    p = P({"name": "RyhB", "pathbank_nucleic_acid_id": 4}, {"name": "ent mRNA"}, {"name": "RyhB2"})
    run(p, fake, text="The RyhB2 gene.")
    rows = p["entities"]["nucleic_acids"]
    assert fake.calls == [] and rows[2]["nucleotide_type_paper"] == "DNA" and rows[2]["nucleotide_type_status"] == "resolved"
    assert "nucleotide_type_status" not in rows[0] and "nucleotide_type_status" not in rows[1]
    assert "nucleotide_type_source" not in rows[2]  # the paper has its own entry


def test_existing_nucleotide_type_is_not_overwritten_and_writer_must_honour_it() -> None:
    p = P({"name": "RyhB", "nucleotide_type": "unknown"}, {"name": "Foo gene mRNA", "synonyms": ["RyhB"]})
    rep = run(p, Fake(RYHB))
    assert row(p)["nucleotide_type"] == "unknown" and row(p)["nucleotide_type_unknown_lookup"] == "slot_filled"
    # the second row's name tokens (gene + mRNA) make the writer return None, so a looked-up type would never be written
    assert "nucleotide_type" not in row(p, 1) and row(p, 1)["nucleotide_type_unknown_lookup"] == "writer_ignores" and rep["lookup_resolved"] == 0


def test_second_run_is_served_from_the_cache() -> None:
    cache = Cache(None)
    fake = Fake(RYHB)
    fn.apply_nucleic_types(P({"name": "RyhB"}), "", fetcher=fake, cache=cache, today=TODAY)
    n = len(fake.calls)
    rep = fn.apply_nucleic_types(P({"name": "RyhB"}), "", fetcher=fake, cache=cache, today=TODAY)
    assert len(fake.calls) == n and rep["lookup_requests"] == 0 and rep["lookup_cache_hits"] >= 1


def test_no_fetcher_means_no_lookup_but_the_flag() -> None:
    p = P({"name": "RyhB"})
    rep = fn.apply_nucleic_types(p, "", today=TODAY)
    assert row(p)["nucleotide_type_unknown_lookup"] == "no_fetcher" and rep["unknown"] == 1


def test_lookup_failure_rolls_back_only_the_lookup_step(monkeypatch) -> None:
    p = P({"name": "RyhB"}, {"name": "Pp"})
    monkeypatch.setattr(fn._Lookup, "resolve", lambda self, r: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = run(p, Fake(RYHB), text="The Pp gene.")
    assert rep["error"].startswith("RuntimeError") and rep["resolved"] == 1 and rep["unknown"] == 0 and rep["lookup_resolved"] == 0
    assert row(p, 1)["nucleotide_type_paper"] == "DNA" and "nucleotide_type_status" not in row(p, 1)  # the paper step stands
    assert "nucleotide_type_status" not in row(p, 0) and "nucleotide_type" not in row(p, 0)


def test_hook_flag_off_untouched_and_flag_on_runs_the_lookup(monkeypatch) -> None:
    p = P({"name": "RyhB"})
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
    assert fp.maybe_apply_final_id_fill(p, {}, {}) == {} and p == P({"name": "RyhB"})
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    rep = fp.maybe_apply_paper_nucleic_types(p, {"final_payload_paper_text": "x"}, fetcher=Fake(RYHB), cache=Cache(None), today=TODAY)
    assert rep["lookup_resolved"] == 1 and row(p)["nucleotide_type"] == "RNA"


def test_writer_writes_a_looked_up_type_flag_on_only_and_unknown_rows_stay_untyped(monkeypatch) -> None:
    def build(flag):
        pl = _payload()
        pl["entities"]["species"][0]["taxonomy_id"] = "9606"  # Homo sapiens: the fake ignores the taxid
        pl["entities"]["nucleic_acids"] = [{"name": "RyhB"}, {"name": "Zzz9"}, {"name": "CHL27", "pathbank_nucleic_acid_id": 7}]
        run(pl, Fake(RYHB), text="")
        if flag:
            monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
        else:
            monkeypatch.delenv("T2PW_FINAL_PAYLOAD", raising=False)
        return pl, _build(pl)

    pl, (builder, root) = build(True)
    assert pl["entities"]["nucleic_acids"][0]["nucleotide_type"] == "RNA"
    assert _na_types(root) == {"RyhB": "RNA", "Zzz9": "", "CHL27": ""}
    assert builder.layout_debug_counts["nucleic_acids_type_unset"] == 1  # only the unknown row; as before for an unsettled row
    _, (_, root) = build(False)  # flag off: the writer ignores the key
    assert _na_types(root) == {"RyhB": "", "Zzz9": "", "CHL27": ""}


def test_unknown_flag_does_not_change_what_the_writer_does(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")

    def types(extra):
        pl = _payload()
        pl["entities"]["nucleic_acids"] = [{"name": "Zzz9", **extra}]
        b, root = _build(pl)
        return _na_types(root), dict(b.layout_debug_counts)

    plain = types({})
    flagged = types({"nucleotide_type_status": "unknown", "nucleotide_type_unknown_reason": "no_cue",
                     "nucleotide_type_unknown_lookup": "not_found"})
    assert plain == flagged and plain[0] == {"Zzz9": ""}
