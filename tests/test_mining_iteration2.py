"""ORCH-743 iteration 2: the pilot-002 audit's five defects inside the ORCH-743 areas.

Each ``test_g9_*`` node fails behaviourally at ecf6b9a0, the tip pilot-002 ran on. The
other tests pin the new behaviour's edges. No network and no model: every provider, HTTP
client and model is scripted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402
from t2pw.mining.models import EvidenceSpan  # noqa: E402

PER_FIELD_LISTS = ("enzyme_evidence_ids", "gene_evidence_ids", "species_evidence_ids",
                   "compartment_evidence_ids", "direction_evidence_ids", "reversible_evidence_ids")


# ---------------------------------------------------------------------------
# IT2-1: extraction in passage batches, with an output budget that fits one
# ---------------------------------------------------------------------------
def _twelve_passages(paper_id, query_text, **kwargs):  # noqa: ARG001
    tag = paper_id.split(":")[-1].lower()
    return [EvidenceSpan(evidence_id=f"ev_{tag}_{n:02d}", paper_id=paper_id, section="results",
                         text=f"Passage {n}: alpha-linolenic acid is oxygenated in Arabidopsis thaliana.")
            for n in range(12)]


def test_g9_it2_1_a_paper_is_extracted_in_batches_with_the_extraction_budget(tmp_path):
    """Pilot-002 sent 20 passages in one call with max_tokens 4000 and a 120 s timeout;
    the hidden reasoning spent the budget and the call came back empty (evidence 032)."""
    calls = []

    def chat(messages, **kwargs):
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        calls.append({"passages": user.count("[evidence_id: "), "max_tokens": kwargs.get("max_tokens"),
                      "timeout": kwargs.get("timeout"), "system": system})
        return SimpleNamespace(text='{"claims": []}', diagnostics=None)

    config = H.make_config(tmp_path, budgets={"max_rounds": 1})
    index = H.offline_index(config, tmp_path)
    index.spans_for_paper = _twelve_passages
    controller = MiningController(config, source=H.ScriptedPaperSource({1: ["P1"]}), chat_fn=chat,
                                  index=index, enabled_override=True)
    outcome = controller.run(probe_model=False, probe_embeddings=False)

    assert [c["passages"] for c in calls] == [5, 5, 2]
    assert {c["max_tokens"] for c in calls} == {12000}
    assert {c["timeout"] for c in calls} == {360.0}
    assert len([r for r in outcome.state.papers.values() if r.processed_round == 1]) == 1
    assert outcome.ledger.usage.papers_processed == 1
    # The prompt is unchanged: every per-field citation list is still demanded.
    for call in calls:
        assert all(key in call["system"] for key in PER_FIELD_LISTS)


def test_the_extraction_settings_and_prompt_shape_are_pinned():
    from t2pw.mining.config import MiningBudgets, MiningModelConfig
    from t2pw.mining.extraction import SYSTEM_PROMPT

    model = MiningModelConfig()
    assert (model.max_tokens, model.request_timeout_seconds, model.prompt_version) == (
        12000, 360.0, "mining-extract/1.2.0")  # ORCH-745 N-5 (base: 1.1.0)
    assert MiningBudgets().max_passages_per_call == 5
    assert all(key in SYSTEM_PROMPT for key in PER_FIELD_LISTS)
    assert "a filled\n   field whose own list is missing or empty will be discarded" in SYSTEM_PROMPT


def test_the_batch_knob_is_honoured(tmp_path):
    seen = []

    def chat(messages, **kwargs):  # noqa: ARG001
        seen.append(next(m["content"] for m in messages if m["role"] == "user").count("[evidence_id: "))
        return SimpleNamespace(text='{"claims": []}', diagnostics=None)

    config = H.make_config(tmp_path, budgets={"max_rounds": 1, "max_passages_per_call": 20})
    index = H.offline_index(config, tmp_path)
    index.spans_for_paper = _twelve_passages
    MiningController(config, source=H.ScriptedPaperSource({1: ["P1"]}), chat_fn=chat, index=index,
                     enabled_override=True).run(probe_model=False, probe_embeddings=False)
    assert seen == [12]


# ---------------------------------------------------------------------------
# IT2-5: every ladder rung, with its fraction and passage count, in model_raw.jsonl
# ---------------------------------------------------------------------------
def _diag(finish_reason, requests):
    return SimpleNamespace(
        finish_reason=finish_reason, attempts=requests, model="scripted", request_hash="req",
        response_hash="res", attempt_log=[{"attempt": n + 1, "finish_reason": finish_reason,
                                           "status": "empty" if finish_reason == "length" else "ok"}
                                          for n in range(requests)])


def test_g9_it2_5_every_ladder_rung_is_recorded_with_its_fraction_and_passages(tmp_path):
    """Pilot-002 overwrote attempt_log per rung (3 of _0002's 9 requests survived) and
    model_raw.jsonl carried no shrink fraction."""
    from t2pw.mining.artifacts import read_jsonl

    empties = []

    def chat(messages, **kwargs):  # noqa: ARG001
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "Lipoxygenase LOX2 oxygenates" in user and len(empties) < 2:
            empties.append(1)
            return SimpleNamespace(text="", diagnostics=_diag("length", 3))
        return SimpleNamespace(text='{"claims": []}', diagnostics=_diag("stop", 1))

    config = H.make_config(tmp_path, budgets={"max_rounds": 1})
    outcome = MiningController(config, source=H.ScriptedPaperSource({1: ["P1"]}), chat_fn=chat,
                               index=H.offline_index(config, tmp_path),
                               enabled_override=True).run(probe_model=False, probe_embeddings=False)
    root = outcome.run_dir.root
    rows = [r for r in read_jsonl(root / "rounds" / "round_001" / "model_raw.jsonl")
            if r.get("paper_id") == "PMC:PMC1000001"]
    call_ids = {r["call_id"] for r in rows}
    assert len(call_ids) == 1 and len(rows) == 3
    assert [r.get("shrink_fraction") for r in rows] == [1.0, 0.6, 0.35]
    sent = [r.get("passages_sent") for r in rows]
    assert sent[0] >= 1 and sent[0] >= sent[1] >= sent[2]
    assert [r.get("schema") for r in rows] == ["full", "full", "tight"]
    assert [len(r.get("provider_requests") or []) for r in rows] == [3, 3, 1]

    record = next(r for r in read_jsonl(root / "model_calls.jsonl") if r["call_id"] in call_ids)
    assert [e.get("rung") for e in record["attempt_log"]] == [1, 1, 1, 2, 2, 2, 3]


# ---------------------------------------------------------------------------
# IT2-4: the grape host
# ---------------------------------------------------------------------------
def _screen_config(tmp_path):
    from t2pw.mining.config import MiningConfig, StartingInput

    seeds = [StartingInput(kind="seed_metabolite", raw="12-OPDA", value="12-OPDA"),
             StartingInput(kind="seed_enzyme", raw="AOS", value="AOS")]
    return MiningConfig.build(
        target_species="Arabidopsis thaliana", topic="jasmonate biosynthesis", starting_inputs=seeds,
        overrides={"runs_dir": str(tmp_path / "runs"), "offline": True, "screen_papers": True,
                   "enabled_via": "test"},
    )


def test_g9_it2_4_route1_does_not_rescue_a_grape_host_paper(tmp_path):
    """Pilot-002's PMC5104190: grape VaPAT1, 'ectopic expression ... in Arabidopsis'. The
    title says 'grape', which neither title detector knew."""
    from t2pw.mining.acquisition import Acquirer, OfflinePaperSource
    from t2pw.mining.models import LiteratureQuery
    from t2pw.rag.acquire import CandidatePaper

    paper = json.loads((ROOT / "tests" / "fixtures" / "mining" / "pilot002_grape_host.json")
                       .read_text(encoding="utf-8"))["papers"][0]
    candidate = CandidatePaper(id=paper["candidate_id"], source="fixture", title=paper["title"],
                               abstract=paper["abstract"], full_text=paper["abstract"])
    config = _screen_config(tmp_path)
    acquirer = Acquirer(config, source=OfflinePaperSource([candidate]))
    decision = acquirer.screen(candidate)
    assert (decision.classification, decision.organism_match, decision.score) == ("omics_only", "match", 3.25)
    assert acquirer.rescue(candidate, decision, ["12-OPDA", "AOS"]) == ""
    result = acquirer.acquire(LiteratureQuery(query_id="qr_fixture", query_text="fixture"), {},
                              round_index=2, max_papers=1, run_terms=["12-OPDA", "AOS"])
    assert [p.record.paper_id for p in result.papers] == []


# ---------------------------------------------------------------------------
# IT2-3: a window the discovery budget cut short never exhausts an item
# ---------------------------------------------------------------------------
def _search_run(tmp_path, papers, budgets):
    from test_mining_search import BooleanSource

    from t2pw.mining.config import StartingInput

    config = H.make_config(tmp_path, budgets={"max_rounds": 1, "max_llm_calls": 200,
                                              "max_papers_processed": 100, **budgets},
                           starting_inputs=[StartingInput(kind="seed_metabolite", raw="12-OPDA",
                                                          value="12-OPDA")])
    source = BooleanSource(papers)
    controller = MiningController(config, source=source, chat_fn=H.ScriptedModel(),
                                  index=H.offline_index(config, tmp_path), enabled_override=True)
    outcome = controller.run(probe_model=False, probe_embeddings=False)
    items = {i["target"]: i["frontier_item_id"] for i in outcome.state.round_records[0]["input_frontier"]}
    return outcome, source, items


def _jasmonate_papers(count):
    return [(f"PMC{2000000 + n}", f"Study {n}: 12-OPDA and jasmonate biosynthesis in Arabidopsis thaliana.")
            for n in range(count)]


def test_g9_it2_3_a_budget_cut_window_does_not_exhaust_the_item(tmp_path):
    """The seed takes 25 of a 26-paper discovery budget. The topic item's window is then cut
    to 1, and that 1 is an already-seen paper. At ecf6b9a0 the topic item was marked
    exhausted, which looks the same as a real search that found nothing."""
    outcome, _, items = _search_run(tmp_path, _jasmonate_papers(30), {"max_papers_discovered": 26})
    topic = items["jasmonate biosynthesis"]
    asked = [q for q in outcome.state.queries if q.frontier_item_id == topic]
    assert asked and asked[0].result_count == 1  # the cut window, filled by a seen paper
    assert topic not in outcome.state.exhausted_frontier
    assert any(i.frontier_item_id == topic and not i.exhausted for i in outcome.state.frontier)


def test_a_spent_discovery_budget_stops_the_round_and_is_the_stop_reason(tmp_path):
    outcome, source, items = _search_run(tmp_path, _jasmonate_papers(30), {"max_papers_discovered": 25})
    assert outcome.ledger.usage.papers_discovered == 25
    assert [q.frontier_item_id for q in outcome.state.queries] == [items["12-OPDA"]]  # nothing after
    assert outcome.stop_reason == "max_papers_discovered_reached"
    assert items["jasmonate biosynthesis"] not in outcome.state.exhausted_frontier


def test_a_full_window_that_finds_nothing_new_still_exhausts(tmp_path):
    from test_mining_search import _run

    known = [(f"PMC{2000000 + n}", f"Study {n}: 12-OPDA and jasmonate biosynthesis in Arabidopsis thaliana.")
             for n in range(3)]
    outcome, _, queries = _run(tmp_path, known, explicit=[i for i, _ in known])
    assert [q.attempt for q in queries] == [1, 2]
    assert queries[0].frontier_item_id in outcome.state.exhausted_frontier


# ---------------------------------------------------------------------------
# IT2-2: NCBI identity from the PMID only; one paper across sources; full-text title check
# ---------------------------------------------------------------------------
PUBMED_XML = ROOT / "tests" / "fixtures" / "mining" / "pubmed_pmid34638573.xml"
TITLE_13LOX = "Biochemical Characterization of 13-Lipoxygenases of Arabidopsis thaliana."


class _Resp:
    def __init__(self, status=200, text="", payload=None):
        self.status_code, self.text, self._payload = status, text, payload

    def json(self):
        return self._payload


def _fake_network(monkeypatch, tmp_path, *, ncbi_for=lambda term: ["34638573"], epmc_for=lambda query: []):
    """Stand-in for rag's HttpClient: NCBI esearch / efetch from the fixture, EuropePMC search and
    the by-PMID lookup from the arguments, 404 for any full-text download."""
    from t2pw.rag import acquire as rag_acquire

    xml = PUBMED_XML.read_text(encoding="utf-8")

    class FakeHttp:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        def get(self, url, params=None, headers=None):  # noqa: ARG002
            params = params or {}
            if "esearch.fcgi" in url:
                return _Resp(payload={"esearchresult": {"idlist": ncbi_for(str(params.get("term")))}})
            if "efetch.fcgi" in url:
                return _Resp(text=xml)
            if url.endswith("/search"):
                query = str(params.get("query"))
                if query.startswith("EXT_ID:34638573"):
                    return _Resp(payload={"resultList": {"result": [
                        {"source": "MED", "id": "34638573", "pmid": "34638573", "pmcid": "PMC8508710"}]}})
                return _Resp(payload={"resultList": {"result": epmc_for(query)}})
            return _Resp(status=404)

    monkeypatch.setattr(rag_acquire, "HttpClient", FakeHttp)
    monkeypatch.setattr(rag_acquire, "_ncbi_throttle", lambda: None)
    monkeypatch.setattr(rag_acquire, "_default_cache_dir", lambda: tmp_path / "acquire_cache")


def _context(term):
    return {"pathway_name": "", "likely_organism": "Arabidopsis thaliana", "key_compounds": [term],
            "key_proteins": [], "gap_terms": []}


def test_g9_it2_2_an_ncbi_record_never_carries_a_cited_reference_identity(tmp_path, monkeypatch):
    """Pilot-002 recorded PMID 34638573 (2021) as PMC411572 with doi 10.1073/pnas.76.9.4350,
    the Towbin 1979 paper its reference list cites."""
    from t2pw.mining.acquisition import PaperSource, record_from_candidate

    _fake_network(monkeypatch, tmp_path)
    found = PaperSource().search(_context("13-lipoxygenase"), sources=["ncbi"], max_papers=5)
    assert len(found) == 1
    paper = found[0]
    assert "PMC411572" not in paper.id and "pnas" not in paper.source_uri
    assert paper.id == "PMC8508710"  # the article's own, from EuropePMC by PMID
    record = record_from_candidate(paper, round_index=1)
    assert (record.paper_id, record.pmid, record.title) == ("PMC:PMC8508710", "34638573", TITLE_13LOX)
    assert record.url == "https://pubmed.ncbi.nlm.nih.gov/34638573/"


def test_g9_it2_2_the_same_article_from_two_sources_is_one_paper(tmp_path, monkeypatch):
    """Pilot-002 discovered 6 articles twice, once per source, under two ids."""
    from t2pw.mining.acquisition import Acquirer, PaperSource
    from t2pw.mining.models import LiteratureQuery

    epmc_item = {"id": "PMC8508710", "source": "PMC", "pmid": "34638573", "pmcid": "PMC8508710",
                 "title": TITLE_13LOX, "abstractText": "13-LOX enzymes of A. thaliana.", "pubYear": "2021"}
    _fake_network(monkeypatch, tmp_path,
                  ncbi_for=lambda term: ["34638573"] if "LOX6" in term else [],
                  epmc_for=lambda query: [epmc_item] if "13-lipoxygenase" in query else [])
    acquirer = Acquirer(H.make_config(tmp_path), source=PaperSource())
    first = acquirer.acquire(LiteratureQuery(query_id="q1", query_text="q1"), _context("13-lipoxygenase"),
                             round_index=1, max_papers=5)
    second = acquirer.acquire(LiteratureQuery(query_id="q2", query_text="q2"), _context("LOX6"),
                              round_index=1, max_papers=5)
    included = [p.record.paper_id for p in [*first.papers, *second.papers]]
    assert included == ["PMC:PMC8508710"]


NAD_TITLE = ("NAD&lt;sup&gt;+&lt;/sup&gt; deficiency primes defense metabolism via "
             "&lt;sup&gt;1&lt;/sup&gt;O&lt;sub&gt;2&lt;/sub&gt;-escalated jasmonate biosynthesis in plants.")
NAD_ABSTRACT = "NAD+ deficiency in the qs-2 mutant primes jasmonate biosynthesis through 13-LOX."
#: The header of what pilot-002 fetched for PMC6628814 (audit A1), and of the real article
#: (the opening of C:/t/mining-pilot-runs2/bio-audit/nad_fulltext.txt).
WRONG_FULL_TEXT = ("An effector from cotton bollworm oral secretion impairs host plant defense signaling "
                   "Chen Chun-Yu a b 1 Liu Yao-Qian ... Results. The effector HARP1 binds JAZ proteins.")
RIGHT_FULL_TEXT = ("2873 ncomms Nature Communications Nat Commun Nature Publishing Group PMC11300881 11300881 "
                   "39103368 10.1038/s41467-024-51114-1 NAD + deficiency primes defense metabolism via 1 O 2 "
                   "-escalated jasmonate biosynthesis in plants Hong Yechun 1 2 # Yu Zongjun ... Results.")


def _fetched(full_text, tmp_path):
    from t2pw.mining.acquisition import Acquirer, PaperSource
    from t2pw.mining.models import LiteratureQuery
    from t2pw.rag.acquire import CandidatePaper

    class OneArticle(PaperSource):
        def search(self, context, *, sources, max_papers):  # noqa: ARG002
            return [CandidatePaper(id="PMC6628814", source="ncbi", title=NAD_TITLE, abstract=NAD_ABSTRACT)]

        def full_text(self, candidate):  # noqa: ARG002
            return full_text

    result = Acquirer(H.make_config(tmp_path), source=OneArticle()).acquire(
        LiteratureQuery(query_id="q", query_text="q"), {}, round_index=1, max_papers=1)
    return result.papers[0]


def test_g9_it2_2_a_full_text_of_another_article_is_refused(tmp_path):
    """Pilot-002 extracted the cotton-bollworm paper under the NAD+ paper's title."""
    loaded = _fetched(WRONG_FULL_TEXT, tmp_path)
    assert loaded.text == NAD_ABSTRACT
    assert any("full text refused" in note for note in loaded.record.notes)


def test_the_article_own_full_text_is_kept(tmp_path):
    loaded = _fetched(RIGHT_FULL_TEXT, tmp_path)
    assert loaded.text == RIGHT_FULL_TEXT
    assert not any("full text refused" in note for note in loaded.record.notes)
