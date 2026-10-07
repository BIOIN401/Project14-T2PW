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


#: EuropePMC ``core`` records as the live service returned them (orchestrator probe, ORCH-745
#: CS-A repair 1; ``journalTitle`` is null in the core shape, the journal is nested).
LIVE_EPMC = {
    "PMC13364333": {"source": "MED", "pmid": "42452298", "pmcid": "PMC13364333",
                    "doi": "10.3390/plants15132102", "journalTitle": None,
                    "journalInfo": {"journal": {"title": "Plants (Basel, Switzerland)"}},
                    "pubTypeList": {"pubType": ["review-article", "Review", "Journal Article"]}},
    "PMC12096059": {"source": "MED", "pmid": "40095511", "pmcid": "PMC12096059",
                    "doi": "10.1111/plb.70001", "journalTitle": None,
                    "journalInfo": {"journal": {"title": "Plant biology (Stuttgart, Germany)"}},
                    "pubTypeList": {"pubType": ["research-article", "Journal Article"]}},
    "PMC8508710": {"source": "MED", "pmid": "34638573", "pmcid": "PMC8508710",
                   "doi": "10.3390/ijms221910237", "journalTitle": None,
                   "journalInfo": {"journal": {"title": "International journal of molecular sciences"}},
                   "pubTypeList": {"pubType": ["research-article", "Journal Article"]}},
}


def _live(query):
    """EuropePMC search by ``PMCID:...`` or ``EXT_ID:<pmid> AND SRC:MED``, from LIVE_EPMC."""
    key = query.split(":", 1)[1].split(" ")[0]
    return _epmc(*[r for r in LIVE_EPMC.values() if key in (r["pmcid"], r["pmid"])])


def test_new_acceptance_745a_f2_one_metadata_lookup_per_included_paper(tmp_path, monkeypatch):
    """A review declared by EuropePMC; an excluded paper (no text) is never looked up."""
    asked = _fake_network(monkeypatch, tmp_path, pmid="", efetch_xml="", europepmc=_live)
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
    assert (review.doi, review.journal) == ("10.3390/plants15132102", "Plants (Basel, Switzerland)")
    assert review.evidence_type_verified is True
    primary = result.papers[1].record
    assert (primary.evidence_type, primary.evidence_type_basis) == ("primary", "declared article type 'research-article'")


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


# ===========================================================================
# Repair round 1 (reviews of 6ceef736). Each ``test_g9_745a_r1_*`` node fails
# behaviourally at 6ceef736 (this whole file runs there: every symbol it imports at
# module level exists at 6ceef736); its docstring says what 6ceef736 returns.
# ``tests/fixtures/mining/orch745a_repair1_texts.json``: EuropePMC full texts a PWML run
# committed under runs/2026-07-27_*/papers/ at f9556288, byte for byte.
# ===========================================================================
REPAIR1 = {p["pmcid"]: p for p in json.loads(
    (FIXTURES / "orch745a_repair1_texts.json").read_text(encoding="utf-8"))["papers"]}


def _bare(pmcid):
    """A EuropePMC candidate as rag's search returns it: no abstract, no declared type."""
    paper = REPAIR1[pmcid]
    return CandidatePaper(id=pmcid, source="europepmc", title=paper["title"], full_text=paper["full_text"],
                          source_uri=paper["source_uri"], year=paper["year"])


def _answers(*replies):
    """A scripted EuropePMC: each call takes the next reply; an exception reply is raised."""
    queue = list(replies)

    def europepmc(query):  # noqa: ARG001
        reply = queue.pop(0) if queue else RuntimeError("no further reply scripted")
        if isinstance(reply, Exception):
            raise reply
        return reply

    return europepmc


# --- A1: review recall without a declared type ---------------------------------
@pytest.mark.parametrize("pmcid", ["PMC12180156", "PMC12444477"])
def test_g9_745a_r1_a1_a_review_named_only_in_its_subject_heading_is_a_review(tmp_path, pmcid):
    """"... 779 1 Review Update of the sideroflexin ..." and "... 1 JBC Reviews The regulation
    of lipid A biosynthesis": the JATS subject heading is the only place either says it is a
    review, and it sits in the front matter that grading stripped. 6ceef736: ``unknown``."""
    record = _acquire(tmp_path, _bare(pmcid)).papers[0].record
    assert record.evidence_type == "review"
    assert "subject heading" in record.evidence_type_basis


def test_g9_745a_r1_a1_a_journal_article_record_with_a_failed_lookup_is_not_primary(tmp_path, monkeypatch):
    """The JBC review as NCBI returns it (PMID 40752575, from its own front matter), its
    PubMed record listing only "Journal Article", and every EuropePMC lookup failing.
    6ceef736 grades it ``primary`` on "declared article type 'Journal Article'"."""
    paper = REPAIR1["PMC12444477"]
    xml = ("<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>40752575</PMID><Article>"
           f"<ArticleTitle>{escape(paper['title'])}</ArticleTitle><PublicationTypeList>"
           "<PublicationType UI=\"D016428\">Journal Article</PublicationType></PublicationTypeList>"
           "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")
    _fake_network(monkeypatch, tmp_path, pmid="40752575", efetch_xml=xml,
                  europepmc=_answers(*[RuntimeError("HTTP request failed after retries")] * 3))

    class NcbiWithText(PaperSource):
        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            return super().search(context, sources=["ncbi"], max_papers=max_papers)

        def full_text(self, candidate):  # noqa: ARG002
            return paper["full_text"]

    result = Acquirer(H.make_config(tmp_path, offline=False), source=NcbiWithText()).acquire(
        LiteratureQuery(query_id="q", query_text="q"), _context("lipid A"), round_index=1, max_papers=5)
    record = result.papers[0].record
    assert (record.paper_id, record.article_type) == ("PMID:40752575", "Journal Article")
    assert record.evidence_type == "review"
    assert record.evidence_type_basis.endswith("type unverified: EuropePMC lookup failed")
    assert record.evidence_type_verified is False


def test_g9_745a_r1_a1_a_failed_lookup_is_asked_once_more(tmp_path, monkeypatch):
    """The first lookup fails, the second answers with the live record. 6ceef736 asks once:
    no declared types, no DOI."""
    asked = _fake_network(monkeypatch, tmp_path, pmid="", efetch_xml="",
                          europepmc=_answers(RuntimeError("down"), _epmc(LIVE_EPMC["PMC12096059"])))

    class Scripted(PaperSource):
        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            return [_candidate(P3[GH3_10])]

        def full_text(self, candidate):
            return candidate.full_text

    record = Acquirer(H.make_config(tmp_path, offline=False), source=Scripted()).acquire(
        LiteratureQuery(query_id="q", query_text="q"), {}, round_index=1, max_papers=5).papers[0].record
    assert asked == ["PMCID:PMC12096059", "PMCID:PMC12096059"]
    assert (record.article_type, record.doi) == ("research-article; Journal Article", "10.1111/plb.70001")
    assert (record.evidence_type, record.evidence_type_verified) == ("primary", True)


def test_new_acceptance_745a_r1_a1_the_subject_heading_needs_the_title_and_review_words():
    from t2pw.mining.acquisition import grade_evidence_type

    review = REPAIR1["PMC12180156"]
    assert grade_evidence_type(title="", full_text=review["full_text"]) == ("unknown", "")
    for pid in (GH3_10, LOX13, PEPPER, BRYOPHYTE):  # Research Article / Article / CRediT / none
        paper = P3[pid]
        assert "subject heading" not in grade_evidence_type(title=paper["title"], full_text=paper["full_text"])[1]
    assert classify_evidence_type(article_type="research-article; Journal Article") == "primary"
    assert classify_evidence_type(article_type="review-article; Review; Journal Article") == "review"


# --- A2: a reference list with no heading ---------------------------------------
def test_g9_745a_r1_a2_a_reference_list_with_no_heading_never_becomes_a_span(tmp_path):
    """PMC12624714 (ACS Infect Dis): EuropePMC's flattening prints no "References" heading,
    so 6ceef736 kept the chunk that runs from the Abbreviations into the bibliography
    (Andrade et al., 10.3390/microorganisms8111716, PMC7692639) and its ACS processing
    flags ("production-flag-...") in the first chunk."""
    paper = REPAIR1["PMC12624714"]
    index = H.offline_index(H.make_config(tmp_path, budgets={"max_passages_per_paper": 500}), tmp_path)
    index.add_paper(LoadedPaper(record=PaperRecord(paper_id="PMC:PMC12624714", pmcid="PMC12624714",
                                                   title=paper["title"], acquisition_source="europepmc"),
                                text=paper["full_text"]))
    texts = [_norm(s.text) for s in index.retrieve("colistin", top_k=1000, paper_ids=["PMC:PMC12624714"])]
    for leak in ("10.3390/microorganisms8111716", "Tambadou", "PMC7692639", "production-flag-",
                 "document-id-alt-64"):
        assert not [t for t in texts if leak in t], leak
    for prose in ("Colistin is currently used as a last resort antibiotic against infections by "
                  "multidrug-resistant Gram-negative bacilli",
                  "Survival analyses of larvae were carried out with the time to event model Kaplan"):
        assert any(prose in t for t in texts), prose


def test_g9_745a_r1_a2_a_mentioned_table_does_not_end_a_reference_list():
    """The pepper paper's list runs into "(Supplemental Table S3) , and the raw read data
    (FASTQ) have been deposited ...": a mention, not a legend. 6ceef736 ended the list there."""
    from t2pw.mining.retrieval import text_regions

    text = _norm(P3[PEPPER]["full_text"])
    assert [(r, e) for _, e, r in text_regions(text) if r == "reference_list"] == [("reference_list", len(text))]


def test_g9_745a_r1_a7_a_fig_label_after_a_reference_list_is_a_legend():
    """PMC8508710 with its first legend label printed as "FIG. 1" (ASM style), and as "Fig 1":
    6ceef736 knew only "Figure"/"Fig." and cut every legend after the list."""
    from t2pw.mining.retrieval import text_regions

    text = _norm(P3[LOX13]["full_text"])
    for label in ("FIG. 1", "Fig 1"):
        variant = text.replace("Figure 1 In silico", f"{label} In silico")
        ends = [e for _, e, r in text_regions(variant) if r == "reference_list"]
        assert [variant[e:].startswith(f"{label} In silico") for e in ends] == [True]


def test_new_acceptance_745a_r1_a2_the_unheaded_rule_finds_every_headed_list():
    """On every pilot-003 text the density rule alone starts within 20 characters of, or in
    the back matter just before, the headed list; a text with no list gets none."""
    from t2pw.mining.retrieval import _unheaded_list_start, text_regions

    for paper in P3.values():
        text = _norm(paper["full_text"])
        start, _, _ = next(r for r in text_regions(text) if r[2] == "reference_list")
        found = _unheaded_list_start(text, len(text) // 3)
        assert found is not None and -1500 < found - start < 20, (paper["paper_id"], start, found)
    assert _unheaded_list_start(_norm(P2["PMID:12777050"]["full_text"]), 0) is None


# --- non-blocking ---------------------------------------------------------------
@pytest.mark.parametrize("pmc_copy_lookup", ["answers", "fails"])
def test_g9_745a_r1_a4_the_pmc_copy_first_then_its_pmid_record_is_one_paper(tmp_path, monkeypatch,
                                                                             pmc_copy_lookup):
    """EuropePMC's copy of 13-LOX is included first; then NCBI returns PMID 34638573 and its
    discovery PMCID lookup fails, so it arrives as ``PMID:34638573``. Whether the PMC copy's own
    lookup named the PMID (answers) or not (fails), it is one paper. 6ceef736: two."""
    live = _epmc(LIVE_EPMC["PMC8508710"])
    down = RuntimeError("HTTP request failed after retries")
    pmc_copy = [live] if pmc_copy_lookup == "answers" else [down, down]
    _fake_network(monkeypatch, tmp_path, pmid="34638573", efetch_xml=LOX13_XML,
                  europepmc=_answers(*pmc_copy, down, live))

    class EuropePmcThenNcbi(PaperSource):
        calls = 0

        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return [CandidatePaper(id="PMC8508710", source="europepmc", title=P3[LOX13]["title"],
                                       abstract=P3[LOX13]["abstract"])]
            return super().search(context, sources=["ncbi"], max_papers=max_papers)

        def full_text(self, candidate):  # noqa: ARG002
            return P3[LOX13]["full_text"]

    acquirer = Acquirer(H.make_config(tmp_path, offline=False), source=EuropePmcThenNcbi())
    runs = [acquirer.acquire(LiteratureQuery(query_id=q, query_text=q), _context("13-lipoxygenase"),
                             round_index=1, max_papers=5) for q in ("q1", "q2")]
    assert [p.record.paper_id for r in runs for p in r.papers] == ["PMC:PMC8508710"]
    if pmc_copy_lookup == "fails":
        assert [(p.paper_id, p.exclusion_reasons) for p in runs[1].excluded] == [
            ("PMID:34638573", ["same article as PMC:PMC8508710, already included"])]


def test_g9_745a_r1_a4_a_resumed_run_remembers_an_included_papers_pmcid(tmp_path):
    """A paper included as ``PMID:34638573`` whose PMCID became known; after a resume its
    ``PMC:`` id must still be the same paper. 6ceef736 restores only the paper ids."""
    config = H.make_config(tmp_path, budgets={"max_rounds": 1})
    candidate = CandidatePaper(id="34638573", source="ncbi", title=P3[LOX13]["title"],
                               abstract=P3[LOX13]["abstract"])
    controller = MiningController(config, source=OfflinePaperSource([candidate]),
                                  chat_fn=lambda *a, **k: SimpleNamespace(text='{"claims": []}', diagnostics=None),
                                  index=H.offline_index(config, tmp_path), enabled_override=True)
    outcome = controller.run(probe_model=False, probe_embeddings=False)
    outcome.state.papers["PMID:34638573"].pmcid = "PMC8508710"
    outcome.state.paper_texts = {}  # nothing to re-index on resume
    outcome.run_dir.write_state(outcome.state.to_dict())
    resumed = MiningController.resume(outcome.run_dir.root, source=OfflinePaperSource([]),
                                      enabled_override=True)
    assert "PMC:PMC8508710" in resumed.acquirer.seen_paper_ids


def test_g9_745a_r1_a5_an_abstractless_paper_keeps_its_front_matter_abstract(tmp_path):
    """GH3.10 supplied as a bare PMCID, offline: no abstract and no lookup. Its abstract lies
    only in the front matter, whose decisive sentence 6ceef736 lost with the rest of it."""
    class Fetches(OfflinePaperSource):
        def full_text(self, candidate):  # noqa: ARG002
            return P3[GH3_10]["full_text"]

    config = H.make_config(tmp_path, budgets={"max_passages_per_paper": 500})
    loaded, _ = Acquirer(config, source=Fetches()).load_explicit([make_paper_starting_input("PMC12096059")])
    assert loaded[0].record.abstract == ""
    index = H.offline_index(config, tmp_path)
    index.add_paper(loaded[0])
    texts = [_norm(s.text) for s in index.retrieve("JA-Ile", top_k=1000, paper_ids=["PMC:PMC12096059"])]
    assert any("wounded leaves of gh3.10 jar1 knockout lines" in t for t in texts)
    assert not [t for t in texts if "pmc-prop-" in t or "ifeussn@" in t]


def test_g9_745a_r1_r4_the_europepmc_abstract_is_stored_as_plain_text(tmp_path, monkeypatch):
    """PMC13490264's abstract as EuropePMC serves it opens "<h4>Main conclusion</h4>This
    review highlights ..." (pilot-001). 6ceef736 stored it raw."""
    abstract = ("<h4>Main conclusion</h4>This review highlights recent advances revealing auxin "
                "homeostasis as an integrated network of <i>metabolism</i> and transport.")
    _fake_network(monkeypatch, tmp_path, pmid="", efetch_xml="",
                  europepmc=lambda query: _epmc({"pmcid": "PMC13490264", "abstractText": abstract}))

    class Scripted(PaperSource):
        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            return [CandidatePaper(id="PMC13490264", source="europepmc",
                                   title="Metabolism meets transport: a new perspective on auxin homeostasis",
                                   full_text="Results. Auxin conjugates are hydrolysed in the ER. " * 20)]

        def full_text(self, candidate):
            return candidate.full_text

    record = Acquirer(H.make_config(tmp_path, offline=False), source=Scripted()).acquire(
        LiteratureQuery(query_id="q", query_text="q"), {}, round_index=1, max_papers=1).papers[0].record
    assert record.abstract == ("Main conclusion This review highlights recent advances revealing auxin "
                               "homeostasis as an integrated network of metabolism and transport.")


def test_g9_745a_r1_a6_the_admitted_reactions_table_shows_the_evidence_level(tmp_path):
    """6ceef736's report table had no evidence-level column."""
    outcome = H.run_fixture(tmp_path)
    report = outcome.outcome.run_dir.path("report.md").read_text(encoding="utf-8")
    assert "| reaction | status | evidence level | round | papers | enzyme | compartment |" in report
    assert "| `accepted` | `multiple_independent_papers` |" in report


@pytest.mark.parametrize("phrase", [
    "This article provides a comprehensive review of jasmonate signalling.",
    "In this minireview we discuss the lipoxygenases.",
    "The aim of this paper is to review oxylipin biosynthesis.",
    "This work reviews recent findings on JA-Ile.",
    "We summarize and review the literature on 12-OPDA.",
    "A review of the literature on jasmonate conjugates is presented.",
])
def test_g9_745a_r1_r1_more_ways_a_review_says_it_is_one(phrase):
    """6ceef736's phrase list missed each of these: ``unknown``."""
    filler = "Lipoxygenase converts linolenic acid in chloroplasts. " * 20
    assert classify_evidence_type(full_text=filler + phrase) == "review"
    assert classify_evidence_type(abstract=phrase) == "review"


@pytest.mark.parametrize("declared,grade", [("reviewed-preprint", "preprint"),
                                            ("Peer Review; research-article", "primary")])
def test_g9_745a_r1_r5_reviewed_and_peer_review_are_not_review_types(declared, grade):
    """6ceef736 matched the substring "review": ``review`` for both."""
    assert classify_evidence_type(article_type=declared) == grade
