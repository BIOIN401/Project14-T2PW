"""ORCH-745 change set A: evidence hygiene (F-2, F-4, A4, A7, IT2-N3).

The F-1 change set will make the species gate read each claim's own supporting
sentences. That is only safe once a review is graded as a review and a reference list
or a journal's front matter can never be cited as evidence. Everything here runs on
what the pilots actually held: ``tests/fixtures/mining/orch745a_pilot_texts.json``
carries pilot-003's four included papers with the full texts it chunked and all 73
evidence spans it minted, and pilot-002's abstract-only records.

Each ``test_g9_745a_*`` node fails behaviourally at f9556288 (a wrong value through a
code path that exists there); its docstring says what base returns. The
``test_new_acceptance_745a_*`` nodes pin new capabilities (the grading basis, the
metadata lookup, the exclusion record, the grade on the reaction node). No network and
no model: every HTTP client and model is scripted.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.acquisition import (  # noqa: E402
    Acquirer,
    LoadedPaper,
    OfflinePaperSource,
    PaperSource,
    classify_evidence_type,
    make_paper_starting_input,
    record_from_candidate,
    record_from_upload,
)
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.mining.models import LiteratureQuery, PaperRecord  # noqa: E402
from t2pw.rag import acquire as rag_acquire  # noqa: E402
from t2pw.rag.acquire import CandidatePaper  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "mining"
PILOT = json.loads((FIXTURES / "orch745a_pilot_texts.json").read_text(encoding="utf-8"))
P3 = {p["paper_id"]: p for p in PILOT["pilot003"]["papers"]}
P2 = {p["paper_id"]: p for p in PILOT["pilot002"]["papers"]}
BRYOPHYTE, GH3_10, LOX13, PEPPER = "PMC:PMC13364333", "PMC:PMC12096059", "PMC:PMC8508710", "PMC:PMC12787732"
RECORD_FIELDS = ("paper_id", "pmid", "pmcid", "doi", "title", "abstract", "url", "acquisition_source",
                 "year", "observed_organisms", "observed_pathways")
_EVIDENCE = re.compile(r"\[evidence_id:\s*([A-Za-z0-9_]+)\]")


def _norm(text):
    return " ".join(str(text or "").split())


def _record(paper):
    return PaperRecord(**{k: paper[k] for k in RECORD_FIELDS})


def _loaded(paper):
    return LoadedPaper(record=_record(paper), text=paper["full_text"])


def _candidate(paper):
    return CandidatePaper(id=paper["pmcid"], source="europepmc",
                          title=paper["title"], abstract=paper["abstract"], full_text=paper["full_text"],
                          source_uri=paper["url"], year=paper["year"])


def _acquire(tmp_path, *candidates):
    acquirer = Acquirer(H.make_config(tmp_path), source=OfflinePaperSource(list(candidates)))
    return acquirer.acquire(LiteratureQuery(query_id="q", query_text="q"), {}, round_index=1, max_papers=10)


# ---------------------------------------------------------------------------
# F-2: a review is graded as a review
# ---------------------------------------------------------------------------
def test_g9_745a_f2_the_bryophyte_review_is_a_review_once_its_full_text_is_read(tmp_path):
    """PMC13364333 declares nothing the search returned and its title and abstract carry no
    review marker; its introduction says "this review makes four specific contributions".
    Base grades it at discovery from the title and abstract only: ``unknown``."""
    result = _acquire(tmp_path, _candidate(P3[BRYOPHYTE]))
    record = result.papers[0].record
    assert record.evidence_type == "review"
    assert "this review" in record.evidence_type_basis and "full text" in record.evidence_type_basis


def test_g9_745a_f2_an_explicit_pmcid_review_is_graded_on_its_full_text(tmp_path):
    """A user-supplied PMCID has no title or abstract until its text is fetched. Base: ``unknown``."""
    class Fetches(OfflinePaperSource):
        def full_text(self, candidate):  # noqa: ARG002
            return P3[BRYOPHYTE]["full_text"]

    acquirer = Acquirer(H.make_config(tmp_path), source=Fetches())
    loaded, errors = acquirer.load_explicit([make_paper_starting_input("PMC13364333")])
    assert not errors
    assert loaded[0].record.evidence_type == "review"


def test_g9_745a_f2_an_uploaded_review_is_graded_on_its_text():
    """An upload was graded from its title alone. Base: ``unknown``."""
    paper = P3[BRYOPHYTE]
    loaded = record_from_upload(data=paper["full_text"].encode("utf-8"), file_name="bryophyte.txt",
                                title="Jasmonate Biosynthesis Across Bryophyte Lineages")
    assert loaded.record.evidence_type == "review"


class _Resp:
    def __init__(self, status=200, text="", payload=None):
        self.status_code, self.text, self._payload = status, text, payload

    def json(self):
        return self._payload


def _fake_network(monkeypatch, tmp_path, *, pmid, efetch_xml, europepmc):
    """rag's HttpClient replaced: NCBI esearch returns ``pmid``, efetch ``efetch_xml``, and every
    EuropePMC search goes to ``europepmc(query)`` (which may raise). Returns the EuropePMC queries."""
    asked = []

    class FakeHttp:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        def get(self, url, params=None, headers=None):  # noqa: ARG002
            params = params or {}
            if "esearch.fcgi" in url:
                return _Resp(payload={"esearchresult": {"idlist": [pmid]}})
            if "efetch.fcgi" in url:
                return _Resp(text=efetch_xml)
            if url.endswith("/search"):
                asked.append(str(params.get("query")))
                return europepmc(str(params.get("query")))
            return _Resp(status=404)

    monkeypatch.setattr(rag_acquire, "HttpClient", FakeHttp)
    monkeypatch.setattr(rag_acquire, "_ncbi_throttle", lambda: None)
    monkeypatch.setattr(rag_acquire, "_default_cache_dir", lambda: tmp_path / "acquire_cache")
    return asked


def _context(term):
    return {"pathway_name": "", "likely_organism": "Marchantia polymorpha", "key_compounds": [term],
            "key_proteins": [], "gap_terms": []}


def _epmc(*items):
    return _Resp(payload={"resultList": {"result": list(items)}})


def test_g9_745a_f2_an_ncbi_publication_type_review_gives_a_review_record(tmp_path, monkeypatch):
    """The efetch XML ``PaperSource._ncbi`` already parses carries PublicationTypeList. This
    record is assembled from pilot-003's record of the bryophyte review (PMID 42452298, from
    its own front matter) with the two types PubMed gives a review. Base drops the types and
    grades the title and abstract: ``unknown``."""
    paper = P3[BRYOPHYTE]
    title = "Jasmonate Biosynthesis Across Bryophyte Lineages: Lessons from Marchantia polymorpha and Beyond."
    xml = (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42452298</PMID><Article>"
        "<Journal><Title>Plants (Basel, Switzerland)</Title></Journal>"
        f"<ArticleTitle>{escape(title)}</ArticleTitle>"
        f"<Abstract><AbstractText>{escape(re.sub(r'<[^>]+>', '', paper['abstract']))}</AbstractText></Abstract>"
        "<PublicationTypeList><PublicationType UI=\"D016428\">Journal Article</PublicationType>"
        "<PublicationType UI=\"D016454\">Review</PublicationType></PublicationTypeList>"
        "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")
    _fake_network(monkeypatch, tmp_path, pmid="42452298", efetch_xml=xml,
                  europepmc=lambda query: _epmc({"source": "MED", "pmid": "42452298", "pmcid": "PMC13364333"}))
    found = PaperSource().search(_context("OPDA"), sources=["ncbi"], max_papers=5)
    record = record_from_candidate(found[0], round_index=1)
    assert record.evidence_type == "review"
    assert (record.paper_id, record.pmid) == ("PMC:PMC13364333", "42452298")
    assert record.article_type == "Journal Article; Review"
    assert record.evidence_type_basis == "declared article type 'Review'"


def test_g9_745a_f2_pilot001_reviews_the_old_markers_missed():
    """The bio-audit called PMC12865809 and PMC13493544 "general oxylipin review"; each says so
    beyond the first 400 abstract characters base read ("In the present review we focus",
    "This review synthesizes"). Base: ``unknown`` for both. Every other pilot-001 paper but the
    bryophyte review (whose abstract does not say it) stays a non-review."""
    papers = json.loads((FIXTURES / "pilot001_screened_papers.json").read_text(encoding="utf-8"))["papers"]
    reviews = {"PMC:PMC12865809", "PMC:PMC13493544", "PMC:PMC13490264"}
    graded = {p["paper_id"]: classify_evidence_type(title=p["title"], abstract=p["abstract"]) for p in papers}
    assert {pid for pid, grade in graded.items() if grade == "review"} == reviews


@pytest.mark.parametrize("abstract", [
    "Jasmonates regulate defence, as reviewed in [3]. Here LOX2 converts 18:3 to 13-HPOT.",
    "Jasmonates regulate defence (for a review, see Wasternack 2013). LOX2 converts 18:3 to 13-HPOT.",
    "Jasmonates regulate defence (reviewed by Wasternack and Hause, 2013). LOX2 converts 18:3.",
    "Jasmonate biosynthesis was recently reviewed. We show that LOX2 converts 18:3 to 13-HPOT.",
    "Arabidopsis thaliana jasmonate metabolism, extensively reviewed elsewhere, is extended here.",
])
def test_g9_745a_f2_citing_someone_elses_review_is_not_being_one(abstract):
    """Base matched the bare word "review" in the first 400 characters: ``review`` for each."""
    assert classify_evidence_type(abstract=abstract) != "review"


def _review_claims(messages, **kwargs):  # noqa: ARG001
    """A model that reports one reaction from the first passage it is shown."""
    user = "".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
    ids = _EVIDENCE.findall(user)
    claims = []
    if ids:
        eid = ids[0]
        claims.append({
            "substrates": [{"name": "dn-cis-OPDA", "entity_type": "metabolite", "evidence_ids": [eid]}],
            "products": [{"name": "dn-iso-OPDA", "entity_type": "metabolite", "evidence_ids": [eid]}],
            "evidence_ids": [eid], "direction": "forward", "direction_evidence_ids": [eid],
            "species": "Marchantia polymorpha", "species_evidence_ids": [eid], "hypothesis": False,
        })
    return SimpleNamespace(text=json.dumps({"claims": claims}), diagnostics=None)


def _run_on_the_review(tmp_path):
    config = H.make_config(tmp_path, target_species="Marchantia polymorpha", budgets={"max_rounds": 1})
    controller = MiningController(config, source=OfflinePaperSource([_candidate(P3[BRYOPHYTE])]),
                                  chat_fn=_review_claims, index=H.offline_index(config, tmp_path),
                                  enabled_override=True)
    return controller.run(probe_model=False, probe_embeddings=False)


def test_g9_745a_f2_a_reaction_only_a_review_supports_is_review_article_only(tmp_path):
    """``admission.evidence_level`` already maps a review to ``review_article_only``; the
    paper was never graded one. Base: ``single_paper``."""
    outcome = _run_on_the_review(tmp_path)
    decisions = [d for d in outcome.state.store.decisions() if d.claim_identity]
    assert decisions
    assert {d.evidence_level for d in decisions} == {"review_article_only"}


def test_new_acceptance_745a_f2_the_grade_reaches_the_reaction_node_view_and_summary(tmp_path):
    from t2pw.mining import viz

    outcome = _run_on_the_review(tmp_path)
    reactions = list(outcome.state.graph.reactions.values())
    assert reactions and {r.evidence_level for r in reactions} == {"review_article_only"}
    assert reactions[0].to_dict()["evidence_level"] == "review_article_only"
    view = viz.reaction_network(outcome.state.graph, papers=outcome.state.papers)
    assert {n["evidence_level"] for n in view["nodes"] if n["type"] == "reaction"} == {"review_article_only"}
    provenance = viz.provenance_network(graph=outcome.state.graph, papers=outcome.state.papers,
                                        claims=outcome.state.store.claims, evidence=outcome.state.evidence)
    paper = next(n for n in provenance["nodes"] if n["type"] == "paper")
    assert paper["evidence_type"] == "review" and "this review" in paper["evidence_type_basis"]
    assert outcome.summary["graph"]["evidence_levels"] == {"review_article_only": len(reactions)}
    report = outcome.run_dir.path("report.md").read_text(encoding="utf-8")
    assert "`review_article_only`" in report


def test_new_acceptance_745a_f2_primaries_stay_primary(tmp_path, monkeypatch):
    """GH3.10 (a Wiley "Research Article"), 13-LOX and the pepper paper cite reviews, and their
    front matter says "Writing - review & editing" and "Institutional Review Board": none of it
    is a self-reference. With no declared type they are ``unknown``; with EuropePMC's types,
    ``primary``, and the lookup fills the DOI and journal pilot-003 left empty (A14)."""
    offline = _acquire(tmp_path, *(_candidate(P3[pid]) for pid in (GH3_10, LOX13, PEPPER)))
    assert {p.record.evidence_type for p in offline.papers} == {"unknown"}

    def europepmc(query):
        return _epmc({"pmcid": query.split(":", 1)[1], "doi": "10.1111/plb.70001",
                      "journalInfo": {"journal": {"title": "Plant biology (Stuttgart, Germany)"}},
                      "pubTypeList": {"pubType": ["research-article", "Journal Article"]}})

    asked = _fake_network(monkeypatch, tmp_path, pmid="", efetch_xml="", europepmc=europepmc)

    class Scripted(PaperSource):
        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            return [_candidate(P3[GH3_10])]

        def full_text(self, candidate):
            return candidate.full_text

    acquirer = Acquirer(H.make_config(tmp_path, offline=False), source=Scripted())
    record = acquirer.acquire(LiteratureQuery(query_id="q", query_text="q"), {}, round_index=1,
                              max_papers=1).papers[0].record
    assert asked == ["PMCID:PMC12096059"]
    assert record.evidence_type == "primary"
    assert record.evidence_type_basis == "declared article type 'research-article'"
    assert (record.doi, record.journal) == ("10.1111/plb.70001", "Plant biology (Stuttgart, Germany)")
    assert record.article_type == "research-article; Journal Article"


@pytest.mark.parametrize("phrase", [
    "In this review we trace the pathway.", "This review is organized by step.",
    "The present review covers bryophytes.", "In this mini-review, we discuss LOX.",
    "Here we review jasmonate biosynthesis.", "We briefly review the literature.",
    "This article reviews recent work.", "We provide a comprehensive overview of oxylipins.",
])
def test_new_acceptance_745a_f2_self_referential_phrases_mark_a_review(phrase):
    from t2pw.mining.acquisition import grade_evidence_type

    filler = "Lipoxygenase converts linolenic acid in chloroplasts. " * 20
    grade, basis = grade_evidence_type(full_text=filler + phrase)
    assert grade == "review" and basis.startswith("full text says")
    assert classify_evidence_type(abstract=phrase) == "review"


@pytest.mark.parametrize("pid", [GH3_10, LOX13, PEPPER])
def test_new_acceptance_745a_f2_the_primaries_full_texts_say_no_review(pid):
    from t2pw.mining.acquisition import grade_evidence_type
    from t2pw.mining.retrieval import body_text

    paper = P3[pid]
    assert grade_evidence_type(title=paper["title"], abstract=paper["abstract"],
                               full_text=body_text(paper["full_text"])) == ("unknown", "")


def test_new_acceptance_745a_f2_one_metadata_lookup_per_included_paper(tmp_path, monkeypatch):
    """A review declared by EuropePMC; an excluded paper (no text) is never looked up."""
    asked = _fake_network(monkeypatch, tmp_path, pmid="", efetch_xml="", europepmc=lambda query: _epmc(
        {"pmcid": query.split(":", 1)[1], "doi": "10.3390/plants15132102", "journalTitle": "Plants (Basel)",
         "pubTypeList": {"pubType": ["review-article", "Review", "Journal Article"]}}))
    empty = CandidatePaper(id="PMC1", source="europepmc", title="No text at all")

    class Scripted(PaperSource):
        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            return [_candidate(P3[BRYOPHYTE]), empty, _candidate(P3[LOX13])]

        def full_text(self, candidate):
            return candidate.full_text

    result = Acquirer(H.make_config(tmp_path, offline=False), source=Scripted()).acquire(
        LiteratureQuery(query_id="q", query_text="q"), {}, round_index=1, max_papers=5)
    assert sorted(asked) == ["PMCID:PMC13364333", "PMCID:PMC8508710"]
    assert [p.paper_id for p in result.excluded] == ["PMC:PMC1"]
    review = result.papers[0].record
    assert (review.evidence_type, review.evidence_type_basis) == ("review", "declared article type 'review-article'")
    assert (review.doi, review.journal) == ("10.3390/plants15132102", "Plants (Basel)")


def test_new_acceptance_745a_f2_a_failed_metadata_lookup_is_a_note_not_an_error(tmp_path, monkeypatch):
    def down(query):
        raise RuntimeError(f"HTTP request failed after retries: {query}")

    _fake_network(monkeypatch, tmp_path, pmid="", efetch_xml="", europepmc=down)

    class Scripted(PaperSource):
        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            return [_candidate(P3[BRYOPHYTE])]

        def full_text(self, candidate):
            return candidate.full_text

    record = Acquirer(H.make_config(tmp_path, offline=False), source=Scripted()).acquire(
        LiteratureQuery(query_id="q", query_text="q"), {}, round_index=1, max_papers=5).papers[0].record
    assert any("metadata lookup failed" in note for note in record.notes)
    assert record.evidence_type == "review" and record.evidence_type_basis.startswith("full text says")


# ---------------------------------------------------------------------------
# F-4: reference lists, front matter and publisher notes are never evidence
# ---------------------------------------------------------------------------
def _index_pilot003(tmp_path, **budgets):
    config = H.make_config(tmp_path, budgets=budgets)
    index = H.offline_index(config, tmp_path)
    index.loaded = {pid: _loaded(paper) for pid, paper in P3.items()}
    for loaded in index.loaded.values():
        index.add_paper(loaded)
    return index


def _span_text(span):
    paper = P3[span["paper_id"]]
    if span["char_start"] is None:
        return _norm(paper["abstract"])
    return _norm(paper["full_text"][span["char_start"]:span["char_end"]])


def test_g9_745a_f4_pilot003_bibliography_front_matter_and_disclaimers_never_become_spans(tmp_path):
    """Base indexes all twelve: seven GH3.10 reference-list chunks labelled ``results``, four
    JATS front-matter ``body#0`` chunks and the MDPI disclaimer tail labelled ``methods``.
    Every other pilot-003 span stays retrievable: by its id, or, where its chunk was trimmed
    or was a duplicate (a figure snippet inside its section), by its text."""
    index = _index_pilot003(tmp_path, max_passages_per_paper=500)
    spans = [s for pid in P3 for s in index.retrieve("jasmonate", top_k=1000, paper_ids=[pid])]
    ids = {s.evidence_id for s in spans}
    texts = [_norm(s.text) for s in spans]
    pilot = PILOT["pilot003"]["spans"]

    bad = [s for s in pilot if s["role"] != "prose"]
    assert len(bad) == 12
    assert sorted(s["evidence_id"] for s in bad if s["evidence_id"] in ids) == []
    for span in bad:
        text = _span_text(span)
        text = text[text.index("References"):] if "References" in text else text  # ev_ae0d92db16547d2c
        assert not [probe for probe in (text[:200], text[-200:]) if any(probe in t for t in texts)], span

    lost = []
    for span in (s for s in pilot if s["role"] == "prose"):
        probe = _span_text(span).split(" References ")[0]  # ev_8d7685e98100893f runs into the list
        if span["evidence_id"] not in ids and not any(probe in t for t in texts):
            lost.append(span["evidence_id"])
    assert lost == []
    assert not [s.evidence_id for s in spans if "pmc-prop-" in s.text or "is-retracted" in s.text]


def test_g9_745a_f4_front_matter_no_longer_takes_the_results_passage_slot(tmp_path):
    """With ``max_passages_per_paper`` 6, base indexes GH3.10's abstract, its front matter,
    the introduction and three methods chunks: the RESULTS passage is cut by the cap."""
    index = _index_pilot003(tmp_path, max_passages_per_paper=6)
    spans = index.retrieve("jasmonate", top_k=6, paper_ids=[GH3_10])
    assert any(s.text.startswith("RESULTS In Arabidopsis") for s in spans)
    assert not any(s.text.startswith("pmc Plant Biol") for s in spans)


def test_new_acceptance_745a_f4_what_was_kept_out_is_recorded_per_paper(tmp_path):
    index = _index_pilot003(tmp_path)
    excluded = index.papers[GH3_10].excluded
    assert excluded["front_matter"] == {"count": 1, "chars": 4576}
    assert excluded["reference_list"] == {"count": 1, "chars": 25174}
    assert excluded["chunk_inside_excluded_text"]["count"] == 7
    assert index.papers[BRYOPHYTE].excluded["publisher_note"]["count"] == 1
    stats = index.stats()
    assert stats["excluded"]["reference_list"]["count"] == 4
    assert set(stats["excluded_by_paper"]) == set(P3)
    assert index.papers[GH3_10].to_dict()["excluded"] == excluded
    notes = index.loaded[GH3_10].record.notes  # the paper record carries it into the round artifacts
    assert [n for n in notes if n.startswith("indexing kept out:")] == [
        "indexing kept out: front_matter x1 (4576 chars), reference_list x1 (25174 chars), "
        f"chunk_inside_excluded_text x7 ({excluded['chunk_inside_excluded_text']['chars']} chars), "
        f"duplicate_chunk x{excluded['duplicate_chunk']['count']} ({excluded['duplicate_chunk']['chars']} chars)"]


def test_new_acceptance_745a_f4_legends_after_a_reference_list_are_prose():
    """PMC8508710 prints its figure and table legends after the reference list."""
    from t2pw.mining.retrieval import body_text, citation_dense, text_regions

    text = _norm(P3[LOX13]["full_text"])
    regions = text_regions(text)
    assert [r for _, _, r in regions] == ["front_matter", "publisher_note", "reference_list"]
    start, end, _ = regions[-1]
    assert text[start:].startswith("References 1. Horn T.") and text[end:].startswith("Figure 1 In silico")
    body = body_text(P3[LOX13]["full_text"])
    assert "Figure 1 In silico analysis of At-13-LOXs" in body and "Horn T. Adel S." not in body
    assert body.startswith("Introduction Lipoxygenases") and not citation_dense(body)
    assert text_regions(_norm(P2["PMID:12777050"]["full_text"])) == []


# ---------------------------------------------------------------------------
# A4: one passage, one span
# ---------------------------------------------------------------------------
def test_g9_745a_a4_an_abstract_only_paper_is_one_span(tmp_path):
    """Pilot-002 indexed PMID 12777050's abstract twice, as ev_220364b31b76704e (abstract#0) and
    ev_5015a845c75ff216 (body#0). Base returns both."""
    paper = P2["PMID:12777050"]
    index = H.offline_index(H.make_config(tmp_path), tmp_path)
    index.add_paper(_loaded(paper))
    spans = index.retrieve("allene oxide cyclase", top_k=10, paper_ids=[paper["paper_id"]])
    assert [s.evidence_id for s in spans] == ["ev_220364b31b76704e"]
    assert {s["evidence_id"] for s in PILOT["pilot002"]["spans"]} == {"ev_220364b31b76704e", "ev_5015a845c75ff216"}


# ---------------------------------------------------------------------------
# A7: per-paper retrieval ranks that paper's own chunks
# ---------------------------------------------------------------------------
def test_g9_745a_a7_a_paper_whose_chunks_rank_low_still_gets_its_passages(tmp_path):
    """Pilot-002's PMC5104190 (grape PAT1, abstract only) got "no passages retrieved": the
    store-wide top 3k went to other papers' chunks. Base: 0 passages."""
    grape = P2["PMC:PMC5104190"]
    index = _index_pilot003(tmp_path)
    index.add_paper(_loaded(grape))
    query = "13-lipoxygenase LOX2 LOX3 LOX4 LOX6 alpha-linolenic acid 13-HPOT 12-OPDA Arabidopsis thaliana"
    spans = index.spans_for_paper(grape["paper_id"], query, top_k=2)
    assert len(spans) == min(2, len(index.papers[grape["paper_id"]].chunk_ids)) > 0
    assert {s.paper_id for s in spans} == {grape["paper_id"]}


# ---------------------------------------------------------------------------
# IT2-N3: a failed PMCID lookup is not "no PMCID"
# ---------------------------------------------------------------------------
LOX13_XML = (FIXTURES / "pubmed_pmid34638573.xml").read_text(encoding="utf-8")
LOX13_EPMC = {"source": "MED", "id": "34638573", "pmid": "34638573", "pmcid": "PMC8508710"}


def test_g9_745a_it2n3_a_failed_pmcid_lookup_is_asked_again(tmp_path, monkeypatch):
    """Base caches the failure as "no PMCID" for the run: the second search still returns
    the PMID, so the article is a ``PMID:`` paper here and a ``PMC:`` paper from EuropePMC."""
    answers = iter([RuntimeError("HTTP request failed after retries"), _epmc(LOX13_EPMC)])

    def europepmc(query):  # noqa: ARG001
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    _fake_network(monkeypatch, tmp_path, pmid="34638573", efetch_xml=LOX13_XML, europepmc=europepmc)
    source = PaperSource()
    first = source.search(_context("13-lipoxygenase"), sources=["ncbi"], max_papers=5)
    second = source.search(_context("13-lipoxygenase"), sources=["ncbi"], max_papers=5)
    assert first[0].id == "34638573"
    assert second[0].id == "PMC8508710"


def test_new_acceptance_745a_it2n3_an_answer_without_a_pmcid_is_asked_once(tmp_path, monkeypatch):
    asked = _fake_network(monkeypatch, tmp_path, pmid="34638573", efetch_xml=LOX13_XML,
                          europepmc=lambda query: _epmc())
    source = PaperSource()
    for _ in range(2):
        assert source.search(_context("13-lipoxygenase"), sources=["ncbi"], max_papers=5)[0].id == "34638573"
    assert asked == ["EXT_ID:34638573 AND SRC:MED"]
    assert source.paper_metadata(record_from_candidate(source.search(
        _context("13-lipoxygenase"), sources=["ncbi"], max_papers=5)[0], round_index=1)) == {}
    assert asked == ["EXT_ID:34638573 AND SRC:MED"]  # the inclusion lookup is the same, cached, query


def test_new_acceptance_745a_it2n3_a_pmcid_learned_at_inclusion_makes_one_paper(tmp_path, monkeypatch):
    """The discovery lookup failed, so the paper is ``PMID:34638573``; the inclusion lookup
    succeeds, and EuropePMC's own record of the same article is then the same paper."""
    answers = iter([RuntimeError("down"), _epmc(LOX13_EPMC)])

    def europepmc(query):  # noqa: ARG001
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    _fake_network(monkeypatch, tmp_path, pmid="34638573", efetch_xml=LOX13_XML, europepmc=europepmc)

    class TwoSources(PaperSource):
        calls = 0

        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return super().search(context, sources=["ncbi"], max_papers=max_papers)
            return [CandidatePaper(id="PMC8508710", source="europepmc", title=P3[LOX13]["title"],
                                   abstract=P3[LOX13]["abstract"])]

        def full_text(self, candidate):  # noqa: ARG002
            return P3[LOX13]["full_text"]

    acquirer = Acquirer(H.make_config(tmp_path, offline=False), source=TwoSources())
    first = acquirer.acquire(LiteratureQuery(query_id="q1", query_text="q1"), _context("13-lipoxygenase"),
                             round_index=1, max_papers=5)
    second = acquirer.acquire(LiteratureQuery(query_id="q2", query_text="q2"), _context("LOX6"),
                              round_index=1, max_papers=5)
    assert [p.record.paper_id for p in first.papers] == ["PMID:34638573"]
    assert first.papers[0].record.pmcid == "PMC8508710"
    assert second.papers == [] and second.excluded == []
