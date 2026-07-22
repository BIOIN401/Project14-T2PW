"""WP1 acquisition tests for ``t2pw.rag.acquire``.

All tests run offline with a mocked ``HttpClient`` (canned EuropePMC JSON / NCBI
eutils JSON+XML / full-text XML) so results are deterministic and no live network
is touched. They cover: query construction, EuropePMC + NCBI normalization to
``CandidatePaper``, dedup against the seed and among candidates, the on-disk cache
(a cache hit avoids a second network call), full-text fetch via the reused core
helper, and graceful offline degradation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.rag.acquire import (  # noqa: E402
    AcquireCache,
    CandidatePaper,
    build_query,
    fetch_full_text,
    search_candidates,
)


# ---------------------------------------------------------------------------
# Canned payloads + a mock HttpClient that records every call.
# ---------------------------------------------------------------------------
_EUROPEPMC_JSON: Dict[str, Any] = {
    "resultList": {
        "result": [
            {
                "title": "Caffeine degradation by Ndm demethylases",
                "abstractText": "N-demethylation of caffeine in Pseudomonas.",
                "source": "MED",
                "id": "111",
                "pmid": "111",
                "pmcid": "PMC111",
                "doi": "10.1/aaa",
                "pubYear": "2019",
            },
            {
                "title": "Theobromine oxidation pathway",
                "abstractText": "Downstream oxidation steps.",
                "source": "MED",
                "id": "222",
                "pmid": "222",
                "doi": "10.2/bbb",
                "pubYear": "2021",
            },
        ]
    }
}

_NCBI_ESEARCH_JSON: Dict[str, Any] = {
    "esearchresult": {"idlist": ["333", "222"]}
}

_NCBI_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>333</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Xanthine methyltransferase in caffeine biosynthesis</ArticleTitle>
        <Abstract><AbstractText>Methyl transfer steps.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">333</ArticleId>
        <ArticleId IdType="doi">10.3/ccc</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article>
        <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Theobromine oxidation pathway</ArticleTitle>
        <Abstract><AbstractText>Downstream oxidation steps.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.2/bbb</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""

_FULLTEXT_XML = (
    "<article><body><sec><title>Results</title>"
    "<p>The Ndm enzymes demethylate caffeine to xanthine.</p>"
    "</sec></body></article>"
)


class _FakeResponse:
    def __init__(
        self, payload: Optional[Dict[str, Any]] = None, *, status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeClient:
    """Deterministic HttpClient stand-in that records every request URL."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def get(
        self, url: str, *, params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> _FakeResponse:
        self.calls.append(url)
        if "fullTextXML" in url:
            return _FakeResponse(text=_FULLTEXT_XML)
        if "europepmc" in url and "search" in url:
            return _FakeResponse(_EUROPEPMC_JSON)
        if "esearch.fcgi" in url:
            return _FakeResponse(_NCBI_ESEARCH_JSON)
        if "efetch.fcgi" in url:
            return _FakeResponse(text=_NCBI_EFETCH_XML)
        return _FakeResponse(status_code=404)


class _OfflineClient:
    """Every request raises, as the reused HttpClient does after retries."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def get(self, url: str, **_: Any) -> _FakeResponse:
        self.calls.append(url)
        raise RuntimeError("HTTP request failed after retries")


_SEED_CONTEXT: Dict[str, Any] = {
    "pathway_name": "caffeine degradation",
    "likely_organism": "Pseudomonas putida",
    "key_compounds": ["caffeine", "theobromine", "xanthine"],
    "key_proteins": ["NdmA", "NdmB"],
    "gap_terms": ["demethylation"],
}


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("RAG_ACQUIRE_MAX_PAPERS", "NCBI_API_KEY", "NCBI_EMAIL"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
def test_build_query_is_organism_scoped() -> None:
    query = build_query(_SEED_CONTEXT)
    assert query.endswith('AND "Pseudomonas putida"')
    assert '"caffeine degradation"' in query
    assert '"demethylation"' in query  # gap terms included


def test_build_query_empty_context_is_empty() -> None:
    assert build_query({}) == ""


# ---------------------------------------------------------------------------
# EuropePMC normalization
# ---------------------------------------------------------------------------
def test_search_europepmc_returns_candidate_papers(
    clean_env: None, tmp_path: Path
) -> None:
    client = _FakeClient()
    papers = search_candidates(
        _SEED_CONTEXT,
        sources=("europepmc",),
        client=client,
        cache_dir=tmp_path,
    )
    assert all(isinstance(p, CandidatePaper) for p in papers)
    ids = [p.id for p in papers]
    assert "PMC111" in ids  # pmcid preferred as the full-text handle
    assert "222" in ids  # no pmcid -> falls back to pmid
    first = next(p for p in papers if p.id == "PMC111")
    assert first.source == "europepmc"
    assert first.title == "Caffeine degradation by Ndm demethylases"
    assert first.organism == "Pseudomonas putida"
    assert first.year == "2019"
    assert first.full_text == ""  # search does not fetch full text


# ---------------------------------------------------------------------------
# Cache: a hit avoids a second network call
# ---------------------------------------------------------------------------
def test_cache_hit_avoids_second_network_call(
    clean_env: None, tmp_path: Path
) -> None:
    client = _FakeClient()
    first = search_candidates(
        _SEED_CONTEXT, sources=("europepmc",), client=client, cache_dir=tmp_path
    )
    calls_after_first = len(client.calls)
    assert calls_after_first >= 1

    client2 = _FakeClient()
    second = search_candidates(
        _SEED_CONTEXT, sources=("europepmc",), client=client2, cache_dir=tmp_path
    )
    assert client2.calls == []  # served entirely from disk cache
    assert [p.id for p in second] == [p.id for p in first]


class _EmptyResultClient(_FakeClient):
    """Reachable APIs that legitimately return zero results."""

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse({"resultList": {"result": []}})


def test_empty_query_short_circuits_without_network_or_cache(
    clean_env: None, tmp_path: Path
) -> None:
    # An empty Stage-0 context builds no query at all. That is an upstream error,
    # not a search: no HTTP call, no cache file, and nothing to memoize.
    client = _FakeClient()
    papers = search_candidates(
        {}, sources=("europepmc", "ncbi"), client=client, cache_dir=tmp_path
    )
    assert papers == []
    assert client.calls == []
    assert list(tmp_path.glob("*.json")) == []


def test_zero_result_search_is_not_cached(clean_env: None, tmp_path: Path) -> None:
    # A search that returns nothing must be retried next run, never memoized.
    client = _EmptyResultClient()
    assert (
        search_candidates(
            _SEED_CONTEXT, sources=("europepmc",), client=client, cache_dir=tmp_path
        )
        == []
    )
    assert client.calls  # the network was actually queried
    assert list(tmp_path.glob("*.json")) == []  # nothing persisted

    client2 = _FakeClient()
    papers = search_candidates(
        _SEED_CONTEXT, sources=("europepmc",), client=client2, cache_dir=tmp_path
    )
    assert client2.calls  # re-queried rather than served an empty cache entry
    assert [p.id for p in papers]  # and the retry now yields real candidates


def test_non_empty_result_is_still_cached(clean_env: None, tmp_path: Path) -> None:
    # Regression guard: legitimate caching of non-empty results is unchanged.
    client = _FakeClient()
    first = search_candidates(
        _SEED_CONTEXT, sources=("europepmc",), client=client, cache_dir=tmp_path
    )
    assert first
    assert len(list(tmp_path.glob("*.json"))) == 1

    client2 = _FakeClient()
    second = search_candidates(
        _SEED_CONTEXT, sources=("europepmc",), client=client2, cache_dir=tmp_path
    )
    assert client2.calls == []
    assert [p.id for p in second] == [p.id for p in first]


def test_offline_zero_result_is_not_cached(clean_env: None, tmp_path: Path) -> None:
    # The poisoning case: an offline run must not freeze a zero into the cache.
    search_candidates(
        _SEED_CONTEXT,
        sources=("europepmc",),
        client=_OfflineClient(),
        cache_dir=tmp_path,
    )
    assert list(tmp_path.glob("*.json")) == []


def test_refresh_bypasses_cache(clean_env: None, tmp_path: Path) -> None:
    client = _FakeClient()
    search_candidates(
        _SEED_CONTEXT, sources=("europepmc",), client=client, cache_dir=tmp_path
    )
    client2 = _FakeClient()
    search_candidates(
        _SEED_CONTEXT,
        sources=("europepmc",),
        client=client2,
        cache_dir=tmp_path,
        refresh=True,
    )
    assert client2.calls  # refresh re-hit the network


# ---------------------------------------------------------------------------
# NCBI normalization
# ---------------------------------------------------------------------------
def test_search_ncbi_parses_pubmed_xml(clean_env: None, tmp_path: Path) -> None:
    client = _FakeClient()
    papers = search_candidates(
        _SEED_CONTEXT, sources=("ncbi",), client=client, cache_dir=tmp_path
    )
    by_id = {p.id: p for p in papers}
    assert "333" in by_id
    paper = by_id["333"]
    assert paper.source == "ncbi"
    assert paper.title == "Xanthine methyltransferase in caffeine biosynthesis"
    assert paper.abstract == "Methyl transfer steps."
    assert paper.year == "2020"
    assert paper.organism == "Pseudomonas putida"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
def test_dedup_across_sources_by_identity(clean_env: None, tmp_path: Path) -> None:
    # EuropePMC id 222 and NCBI PMID 222 are the same paper -> kept once.
    client = _FakeClient()
    papers = search_candidates(
        _SEED_CONTEXT,
        sources=("europepmc", "ncbi"),
        client=client,
        cache_dir=tmp_path,
    )
    assert [p.id for p in papers].count("222") == 1


def test_dedup_against_seed(clean_env: None, tmp_path: Path) -> None:
    context = dict(_SEED_CONTEXT)
    context["seed_paper"] = {"pmcid": "PMC111"}
    client = _FakeClient()
    papers = search_candidates(
        context, sources=("europepmc",), client=client, cache_dir=tmp_path
    )
    assert all(p.id != "PMC111" for p in papers)  # the seed itself is excluded


# ---------------------------------------------------------------------------
# max_papers cap
# ---------------------------------------------------------------------------
def test_max_papers_caps_results(clean_env: None, tmp_path: Path) -> None:
    client = _FakeClient()
    papers = search_candidates(
        _SEED_CONTEXT,
        sources=("europepmc",),
        max_papers=1,
        client=client,
        cache_dir=tmp_path,
    )
    assert len(papers) == 1


# ---------------------------------------------------------------------------
# Full text
# ---------------------------------------------------------------------------
def test_fetch_full_text_reuses_core_helper(
    clean_env: None, tmp_path: Path
) -> None:
    candidate = CandidatePaper(id="PMC111", source="europepmc", title="x")
    client = _FakeClient()
    text = fetch_full_text(candidate, client=client, cache_dir=tmp_path)
    assert "demethylate caffeine to xanthine" in text
    assert any("fullTextXML" in url for url in client.calls)

    # Second fetch is served from cache with no network call.
    client2 = _FakeClient()
    text2 = fetch_full_text(candidate, client=client2, cache_dir=tmp_path)
    assert text2 == text
    assert client2.calls == []


def test_fetch_full_text_resolves_pmid_to_pmcid(
    clean_env: None, tmp_path: Path
) -> None:
    # A bare PMID first hits the EuropePMC search to resolve a PMCID, then the
    # full-text XML endpoint.
    candidate = CandidatePaper(id="111", source="ncbi", title="x")
    client = _FakeClient()
    text = fetch_full_text(candidate, client=client, cache_dir=tmp_path)
    assert "demethylate caffeine to xanthine" in text
    assert any("search" in url for url in client.calls)
    assert any("fullTextXML" in url for url in client.calls)


# ---------------------------------------------------------------------------
# Offline-first: missing network degrades gracefully
# ---------------------------------------------------------------------------
def test_offline_returns_empty_without_raising(
    clean_env: None, tmp_path: Path
) -> None:
    client = _OfflineClient()
    papers = search_candidates(
        _SEED_CONTEXT,
        sources=("europepmc", "ncbi"),
        client=client,
        cache_dir=tmp_path,
    )
    assert papers == []


def test_offline_full_text_returns_empty(clean_env: None, tmp_path: Path) -> None:
    candidate = CandidatePaper(id="PMC111", source="europepmc")
    text = fetch_full_text(candidate, client=_OfflineClient(), cache_dir=tmp_path)
    assert text == ""


# ---------------------------------------------------------------------------
# Cache is fail-safe
# ---------------------------------------------------------------------------
def test_acquire_cache_corrupt_file_is_a_miss(tmp_path: Path) -> None:
    cache = AcquireCache(tmp_path)
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert cache.get("bad") is None
