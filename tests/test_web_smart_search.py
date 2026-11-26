"""NEW ACCEPTANCE TEST (W10, new capability): the one-box smart search.

Router classification (regexes, organism split, LLM mocked), the PathBank index on a small
fixture (exact, synonym, organism filter, fuzzy, no match) and the web routes (PathBank hit,
multi-match, fallback to a pipeline job). No network, no database, no LLM.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from t2pw.service import organisms as organism_data
from t2pw.web import pathbank, smart_search as ss
from t2pw.web.app import create_app
from t2pw.web.jobs import JobStore

ORGS = organism_data.load_organisms()
SID_RX = re.compile(r"/job/([0-9a-f]{32})$")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'

ROWS = [  # species: 1 Homo sapiens, 3 Escherichia coli, 12 Mus musculus, 18 Saccharomyces cerevisiae
    dict(pw_id="PW000001", smp_id="SMP0000055", name="Alanine Metabolism", species_id=1, subject="Metabolic"),
    dict(pw_id="PW000005", smp_id="SMP0000057", name="Citric Acid Cycle", species_id=1, subject="Metabolic"),
    dict(pw_id="PW000146", smp_id="SMP0000040", name="Glycolysis", species_id=1, subject="Metabolic"),
    dict(pw_id="PW000779", smp_id="SMP0000802", name="TCA Cycle", species_id=3, subject="Metabolic"),
    dict(pw_id="PW000785", smp_id="SMP0000999", name="Glycolysis and Pyruvate Dehydrogenase", species_id=1,
         subject="Metabolic"),
    dict(pw_id="PW002048", smp_id="SMP0002062", name="Enterobactin Biosynthesis", species_id=3, subject="Metabolic"),
    dict(pw_id="PW002386", smp_id="SMP0002312", name="Glycolysis I", species_id=18, subject="Metabolic"),
    dict(pw_id="PW100001", smp_id="SMP0100001", name="Alanine metabolism", species_id=12, subject="Metabolic"),
    dict(pw_id="bad", smp_id="", name="skipped: not a PW id", species_id=1, subject=""),
]


@pytest.fixture()
def index(tmp_path) -> pathbank.PathBankIndex:
    path = tmp_path / "pb.sqlite"
    assert pathbank.build_index(ROWS, path) == 8
    return pathbank.PathBankIndex(path)


# -- router ------------------------------------------------------------------------------


@pytest.mark.parametrize("text, kind, pmids, pmcids, dois", [
    ("PMID: 12345678", "paper", ("12345678",), (), ()),
    ("12345678, 23456789", "paper", ("12345678", "23456789"), (), ()),
    ("PMC1234567", "paper", (), ("PMC1234567",), ()),
    ("see doi 10.1038/nature12373.", "paper", (), (), ("10.1038/nature12373",)),
    ("https://pubmed.ncbi.nlm.nih.gov/31234567/", "paper", ("31234567",), (), ()),
    ("https://doi.org/10.1093/nar/gkad1041", "paper", (), (), ("10.1093/nar/gkad1041",)),
    ("https://example.org/some/page", "url", (), (), ()),
    ("   ", "empty", (), (), ()),
])
def test_identifiers_by_regex(text, kind, pmids, pmcids, dois):
    r = ss.classify(text, ORGS)
    assert (r.kind, r.pmids, r.pmcids, r.dois) == (kind, pmids, pmcids, dois)


def test_long_passage_is_text_even_with_a_citation():
    para = ("Glucose is phosphorylated by hexokinase to glucose 6-phosphate. " * 8) + "(PMID 12345678)."
    assert ss.classify(para, ORGS).kind == "text"
    assert ss.classify("glucose " * 80, ORGS).kind == "text"  # long without sentences: still pasted text


@pytest.mark.parametrize("text, name, organism", [
    ("TCA cycle", "TCA cycle", None),
    ("glycolysis in Saccharomyces cerevisiae", "glycolysis", "Saccharomyces cerevisiae"),
    ("enterobactin biosynthesis E. coli", "enterobactin biosynthesis", "Escherichia coli"),
    ("alanine metabolism (human)", "alanine metabolism", "Homo sapiens"),
    ("ergosterol biosynthesis in yeast", "ergosterol biosynthesis", "Saccharomyces cerevisiae"),
    ("metabolism of serine", "metabolism of serine", None),  # "serine" is no organism: the phrase stays whole
])
def test_pathway_phrase_and_organism(text, name, organism):
    r = ss.classify(text, ORGS)
    assert r.kind == "pathway" and r.pathway_name == name
    assert (r.organism.label if r.organism else None) == organism


def test_llm_only_for_sentences():
    assert not ss.wants_llm(ss.classify("TCA cycle", ORGS))
    assert not ss.wants_llm(ss.classify("glycolysis in Saccharomyces cerevisiae", ORGS))  # organism split off
    assert ss.wants_llm(ss.classify("how does yeast make ergosterol", ORGS))
    assert ss.wants_llm(ss.classify("ergosterol?", ORGS))
    assert not ss.wants_llm(ss.classify("PMID 1234 and the rest of it", ORGS))


def test_llm_extract_parses_and_checks_the_answer():
    route = ss.classify("how does baker's yeast make ergosterol", ORGS)
    seen = []

    def chat(messages):
        seen.append(messages)
        return json.dumps({"kind": "pathway", "pathway_name": "  Ergosterol   Biosynthesis ",
                           "organism": "Saccharomyces cerevisiae", "identifiers": []})

    got = ss.llm_extract(route, ORGS, chat)
    assert got.used_llm and got.pathway_name == "Ergosterol Biosynthesis"
    assert got.organism.label == "Saccharomyces cerevisiae"
    assert seen[0][0]["content"] == ss.ROUTER_PROMPT and seen[0][1]["content"] == route.query


def test_llm_extract_keeps_only_real_identifiers_and_known_organisms():
    route = ss.classify("the paper about how bacteria build siderophores", ORGS)
    invented = lambda m: json.dumps({"kind": "paper", "identifiers": ["the siderophore paper"],  # noqa: E731
                                     "pathway_name": "", "organism": "Martian bacterium"})
    assert ss.llm_extract(route, ORGS, invented) is None
    real = lambda m: json.dumps({"kind": "paper", "identifiers": ["PMID 7654321"]})  # noqa: E731
    assert ss.llm_extract(route, ORGS, real).pmids == ("7654321",)
    unknown_org = lambda m: json.dumps({"kind": "pathway", "pathway_name": "Enterobactin Biosynthesis",  # noqa: E731
                                        "organism": "Martian bacterium"})
    got = ss.llm_extract(route, ORGS, unknown_org)
    assert got.organism is None and got.organism_text == "Martian bacterium"


@pytest.mark.parametrize("reply", ["not json", "[]", json.dumps({"kind": "other"}), None])
def test_llm_extract_bad_answers_are_none(reply):
    route = ss.classify("show me something about energy please", ORGS)
    assert ss.llm_extract(route, ORGS, lambda m: reply) is None


def test_llm_extract_failure_and_timeout_are_none():
    route = ss.classify("show me something about energy please", ORGS)

    def boom(m):
        raise RuntimeError("provider down")

    assert ss.llm_extract(route, ORGS, boom) is None
    t = time.monotonic()
    assert ss.llm_extract(route, ORGS, lambda m: time.sleep(2) or "{}", timeout_s=0.2) is None
    assert time.monotonic() - t < 1.5


def test_to_pmids_uses_the_id_converter_and_never_raises():
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, records):
            self._r = records

        def json(self):
            return {"records": self._r}

    def get(url, params):
        calls.append(params["idtype"])
        if params["idtype"] == "pmcid":
            return Resp([{"requested-id": "PMC1234567", "pmid": "11111111"}])
        return Resp([{"requested-id": "10.1/x", "status": "error"}])

    assert ss.to_pmids(("PMC1234567",), ("10.1/X",), get) == {"PMC1234567": "11111111", "10.1/X": ""}
    assert calls == ["pmcid", "doi"]
    assert ss.to_pmids(("PMC1",), (), lambda u, params: 1 / 0) == {}


# -- PathBank index ----------------------------------------------------------------------


def test_exact_name_is_strong(index):
    hits = index.search("tca  CYCLE")
    assert hits[0].pw_id == "PW000779" and hits[0].score == 1.0 and pathbank.is_strong(hits)
    assert hits[0].url == "https://pathbank.org/view/SMP0000802"


def test_synonym_finds_both_names(index):
    hits = index.search("Krebs cycle")
    assert {h.pw_id for h in hits} == {"PW000005", "PW000779"} and pathbank.is_strong(hits)


def test_organism_filter(index):
    hits = index.search("alanine metabolism", species_ids=[12])
    assert [h.pw_id for h in hits] == ["PW100001"] and hits[0].organisms == 2
    assert index.search("alanine metabolism")[0].pw_id == "PW000001"  # lowest PW id when unfiltered
    assert index.search("enterobactin biosynthesis", species_ids=[1]) == []


def test_partial_and_fuzzy_are_not_strong(index):
    yeast = index.search("glycolysis", species_ids=[18])
    assert [h.name for h in yeast] == ["Glycolysis I"] and not pathbank.is_strong(yeast)
    typo = index.search("glycolisis")
    assert typo[0].name == "Glycolysis" and not pathbank.is_strong(typo)


def test_no_match_and_missing_index(index, tmp_path):
    assert index.search("zzzz qqqq") == []
    assert index.get("PW000779").name == "TCA Cycle" and index.get("PW999999") is None
    assert index.get("../etc") is None
    missing = pathbank.PathBankIndex(tmp_path / "nope.sqlite")
    assert not missing.available and missing.search("TCA cycle") == [] and missing.get("PW000779") is None


def test_local_files_only_by_valid_id(tmp_path):
    (tmp_path / "pwml_markups").mkdir()
    (tmp_path / "pwml_markups" / "PW000001.pwml").write_text("<x/>")
    assert pathbank.local_file(tmp_path, "pwml", "PW000001") is not None
    assert pathbank.local_file(tmp_path, "svg", "PW000001") is None
    assert pathbank.local_file(tmp_path, "pwml", "../PW000001") is None
    assert pathbank.local_file(None, "pwml", "PW000001") is None
    for sub in ("simple_vector_images", "full_vector_images"):  # the full-colour drawing wins
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "PW000001.svg").write_text("<svg/>")
        assert pathbank.local_file(tmp_path, "svg", "PW000001").parent.name == sub


# -- web routes --------------------------------------------------------------------------


class Site:
    def __init__(self, tmp: Path, index: pathbank.PathBankIndex, llm=False, **kw):
        files = tmp / "pbfiles"
        (files / "simple_vector_images").mkdir(parents=True)
        (files / "pwml_markups").mkdir()
        (files / "simple_vector_images" / "PW000779.svg").write_bytes(SVG)
        (files / "pwml_markups" / "PW002048.pwml").write_text("<super-pathway-visualization/>")
        self.db = tmp / "jobs.sqlite"
        self.app = create_app(self.db, tmp / "work", "https://t2pw.test", rate_secret=b"s", rate_limit=100,
                              pathbank_index=index.path, pathbank_files=files, search_llm=llm, **kw)
        self.client = TestClient(self.app, follow_redirects=False)

    def jobs(self) -> int:
        st = JobStore(self.db)
        try:
            return st._db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        finally:
            st.close()


def test_front_page_has_the_search_box(tmp_path, index):
    html = Site(tmp_path, index).client.get("/").text
    assert '<form method="post" action="/search" role="search">' in html and 'name="q"' in html


def test_known_pathway_shows_pathbank_without_a_job(tmp_path, index):
    site = Site(tmp_path, index)
    r = site.client.post("/search", data={"q": "TCA cycle"})
    assert r.status_code == 200 and "<h1>TCA Cycle</h1>" in r.text
    assert "https://pathbank.org/view/SMP0000802" in r.text and "PathBank (Wishart Lab" in r.text
    assert 'src="/pathbank/PW000779/diagram"' in r.text and "PathBank&#39;s own diagram" in r.text
    assert "Citric Acid Cycle" in r.text  # the synonym hit is offered as another pathway
    assert "Generate from literature instead" in r.text and "/search/run?kind=name&amp;name=TCA+cycle" in r.text
    assert site.jobs() == 0
    d = site.client.get("/pathbank/PW000779/diagram")
    assert d.status_code == 200 and d.headers["content-type"].startswith("image/svg+xml")
    assert d.headers["content-security-policy"].startswith("sandbox")
    assert site.client.get("/pathbank/PW000779/pwml").status_code == 404  # no local PWML for this one


def test_get_search_and_organism_scoped_hit(tmp_path, index):
    site = Site(tmp_path, index)
    r = site.client.get("/search", params={"q": "enterobactin biosynthesis E. coli"})
    assert r.status_code == 200 and "<h1>Enterobactin Biosynthesis</h1>" in r.text
    assert "Escherichia coli" in r.text and "Drawn by T2PW from PathBank&#39;s PWML" in r.text
    assert 'href="/pathbank/PW002048/pwml"' in r.text


def test_pwml_only_pathway_is_rendered_once(tmp_path, index, monkeypatch):
    import t2pw.render.pathbank as render

    calls = []
    monkeypatch.setattr(render, "render_pwml", lambda p, fmt="png", **k: calls.append(Path(p).name) or PNG)
    site = Site(tmp_path, index)
    for _ in range(2):
        r = site.client.get("/pathbank/PW002048/diagram")
        assert r.status_code == 200 and r.content == PNG and r.headers["content-type"] == "image/png"
    assert calls == ["PW002048.pwml"]
    p = site.client.get("/pathbank/PW002048/pwml")
    assert p.status_code == 200 and "attachment" in p.headers["content-disposition"]


def test_unknown_pathbank_ids_are_404(tmp_path, index):
    c = Site(tmp_path, index).client
    for url in ("/pathbank/PW999999", "/pathbank/xyz", "/pathbank/PW999999/diagram", "/pathbank/PW000001/diagram"):
        assert c.get(url).status_code == 404, url


def test_several_matches_offer_a_choice(tmp_path, index):
    site = Site(tmp_path, index)
    r = site.client.post("/search", data={"q": "glycolysis in Saccharomyces cerevisiae"})
    assert r.status_code == 200 and "PathBank pathways for" in r.text
    assert r.text.count('href="/pathbank/PW002386?q=') == 1 and "Glycolysis I" in r.text  # not listed twice
    assert "Drawn by PathBank for other organisms" in r.text and "/pathbank/PW000146?q=" in r.text
    assert site.jobs() == 0


def test_no_match_falls_back_to_a_literature_run(tmp_path, index):
    site = Site(tmp_path, index)
    r = site.client.post("/search", data={"q": "staphyloxanthin biosynthesis"})
    assert r.status_code == 200 and "No curated PathBank pathway matched" in r.text
    assert 'name="name" value="staphyloxanthin biosynthesis"' in r.text and 'value="name"' in r.text
    org = organism_data.get_by_label("Staphylococcus aureus") or ORGS[0]
    q = site.client.post("/search/run", data={"kind": "name", "name": "staphyloxanthin biosynthesis",
                                              "organism_1": org.label, "consent": "1"})
    assert q.status_code == 303 and SID_RX.search(q.headers["location"])
    st = JobStore(site.db)
    try:
        req = json.loads(st._db.execute("SELECT request_json FROM items").fetchone()[0])
    finally:
        st.close()
    assert req["scope"]["requested_pathway"] == "staphyloxanthin biosynthesis" and not req["pmids"]


def test_sentence_calls_the_llm_once_then_finds_pathbank(tmp_path, index):
    calls = []

    def chat(messages):
        calls.append(1)
        return json.dumps({"kind": "pathway", "pathway_name": "TCA Cycle", "organism": "", "identifiers": []})

    site = Site(tmp_path, index, llm=chat)
    r = site.client.post("/search", data={"q": "how do cells burn acetyl-CoA for energy?"})
    assert "<h1>TCA Cycle</h1>" in r.text and calls == [1]
    site.client.post("/search", data={"q": "TCA cycle"})  # deterministic hit: no LLM
    assert calls == [1]


def test_llm_off_or_failing_still_answers(tmp_path, index):
    r = Site(tmp_path, index, llm=False).client.post("/search", data={"q": "how do cells burn acetyl-CoA for energy?"})
    assert r.status_code == 200 and "No curated PathBank pathway matched" in r.text
    r = Site(tmp_path / "b", index, llm=lambda m: 1 / 0).client.post(
        "/search", data={"q": "how do cells burn acetyl-CoA for energy?"})
    assert r.status_code == 200 and "No curated PathBank pathway matched" in r.text


def test_paper_ids_prefill_the_run_and_queue_one_job(tmp_path, index):
    class Resp:
        status_code = 200

        def json(self):
            return {"records": [{"requested-id": "PMC1234567", "pmid": "11111111"}]}

    site = Site(tmp_path, index, search_idconv=lambda url, params: Resp())
    r = site.client.post("/search", data={"q": "PMID 12345678 and PMC1234567"})
    assert r.status_code == 200 and "Build a pathway from these papers" in r.text
    assert "12345678, 11111111" in r.text and site.jobs() == 0
    q = site.client.post("/search/run", data={"kind": "paper", "pmids": "12345678, 11111111", "consent": "1",
                                              "name": "glycolysis", "organism_1": "Homo sapiens"})
    assert q.status_code == 303 and site.jobs() == 1


def test_unresolved_doi_is_reported(tmp_path, index):
    class Resp:
        status_code = 200

        def json(self):
            return {"records": []}

    r = Site(tmp_path, index, search_idconv=lambda url, params: Resp()).client.post(
        "/search", data={"q": "10.1000/xyz123"})
    assert "10.1000/xyz123: no PubMed record was found" in r.text and "did not answer" in r.text


def test_pasted_text_prefills_and_queues(tmp_path, index):
    site = Site(tmp_path, index)
    para = "Hexokinase phosphorylates glucose. Phosphoglucose isomerase makes fructose 6-phosphate. " * 6
    r = site.client.post("/search", data={"q": para})
    assert r.status_code == 200 and "Build a pathway from your text" in r.text and "Hexokinase" in r.text
    bare = site.client.post("/search/run", data={"kind": "text", "text": para})
    assert bare.status_code == 400 and "Enter the pathway name this text" in bare.text  # PO ruling Q-BR-1
    q = site.client.post("/search/run", data={"kind": "text", "text": para, "name": "glycolysis",
                                              "organism_1": "Homo sapiens"})
    assert q.status_code == 303 and site.jobs() == 1


def test_run_form_validation(tmp_path, index):
    site = Site(tmp_path, index)
    c = site.client
    assert "Choose an organism." in c.post("/search/run", data={"kind": "name", "name": "x"}).text
    assert "Enter at least one PubMed ID." in c.post("/search/run", data={"kind": "paper"}).text
    assert "not PubMed IDs" in c.post("/search/run", data={"kind": "paper", "pmids": "abc"}).text
    r = c.post("/search/run", data={"kind": "text", "text": "x", "organism_1": "Homo sapiens"})
    assert r.status_code == 400 and "needed to use the organism" in r.text
    assert "not in the organism list" in c.post("/search/run", data={"kind": "text", "text": "x",
                                                                       "name": "n", "organism_1": "Martians"}).text
    assert site.jobs() == 0


def test_cross_site_posts_are_refused(tmp_path, index):
    c = Site(tmp_path, index).client
    h = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    assert c.post("/search", data={"q": "TCA cycle"}, headers=h).status_code == 403
    assert c.post("/search/run", data={"kind": "text", "text": "x"}, headers=h).status_code == 403


def test_url_and_empty_queries(tmp_path, index):
    c = Site(tmp_path, index).client
    assert "cannot read a web page directly" in c.post("/search", data={"q": "https://example.org/x"}).text
    assert c.post("/search", data={"q": " "}).status_code == 303


def test_hostile_query_is_escaped(tmp_path, index):
    c = Site(tmp_path, index).client
    r = c.get("/search", params={"q": '<script>alert(1)</script> cycle'})
    assert r.status_code == 200 and "<script>alert(1)" not in r.text


def test_missing_index_offers_the_literature_run(tmp_path):
    site = Site(tmp_path, pathbank.PathBankIndex(tmp_path / "none.sqlite"))
    r = site.client.post("/search", data={"q": "TCA cycle"})
    assert r.status_code == 200 and "PathBank lookup is not available" in r.text
    assert site.client.get("/pathbank/PW000779").status_code == 404
