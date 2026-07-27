"""Tests for ``t2pw.batch.fetch`` -- the batch runner's paper acquisition.

Everything here is offline: ``search_candidates`` / ``fetch_full_text`` are
replaced by injected fakes (the ``search_fn`` / ``fetch_text_fn`` parameters), so
no network is touched and results are deterministic. Coverage: topics-file
parsing (all three line shapes plus comments/blanks), pinned-id detection,
"empty full text is a skip, not a paper", Windows-safe and length-bounded slugs,
cross-topic de-duplication, the total ``limit`` cap, and the hard guarantee that
``fetch_papers`` never raises.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.fetch import (  # noqa: E402
    DEFAULT_PER_TOPIC,
    BatchPaper,
    TopicSpec,
    fetch_papers,
    make_slug,
    parse_topics,
)
from t2pw.rag.acquire import CandidatePaper  # noqa: E402

_WINDOWS_ILLEGAL = '<>:"/\\|?*'


def _candidate(ident: str, title: str = "", **kwargs: Any) -> CandidatePaper:
    return CandidatePaper(
        id=ident,
        source=kwargs.pop("source", "europepmc"),
        title=title,
        **kwargs,
    )


def _searcher(mapping: Dict[str, List[CandidatePaper]]):
    """Fake ``search_candidates`` keyed by the topic (pathway_name) in context."""

    calls: List[Dict[str, Any]] = []

    def _search(context: Dict[str, Any], **kwargs: Any) -> List[CandidatePaper]:
        calls.append({"context": context, "kwargs": kwargs})
        status = kwargs.get("status")
        if isinstance(status, dict):
            status["query"] = f'"{context.get("pathway_name")}"'
        return list(mapping.get(str(context.get("pathway_name") or ""), []))

    _search.calls = calls  # type: ignore[attr-defined]
    return _search


def _texter(texts: Dict[str, str]):
    def _fetch(candidate: CandidatePaper, **_: Any) -> str:
        return texts.get(candidate.id, "")

    return _fetch


# ---------------------------------------------------------------------------
# parse_topics
# ---------------------------------------------------------------------------
def test_parse_topics_all_three_shapes_with_comments_and_blanks() -> None:
    text = """
    # overnight batch topics

    lipid A biosynthesis      | Escherichia coli    | 3
    menaquinone biosynthesis  | Lactococcus lactis
    caffeine degradation
    PMC4412817

    #  a trailing comment-only line
    """
    specs = parse_topics(text)
    assert len(specs) == 4

    three = specs[0]
    assert (three.topic, three.organism, three.count) == (
        "lipid A biosynthesis",
        "Escherichia coli",
        3,
    )
    assert not three.is_pinned

    two = specs[1]
    assert (two.topic, two.organism, two.count) == (
        "menaquinone biosynthesis",
        "Lactococcus lactis",
        DEFAULT_PER_TOPIC,
    )

    bare = specs[2]
    assert bare.topic == "caffeine degradation"
    assert bare.organism == ""
    assert not bare.is_pinned

    pinned = specs[3]
    assert pinned.is_pinned and pinned.pinned_id == "PMC4412817"
    assert pinned.topic == ""


def test_parse_topics_ignores_bad_count_and_empty_topic() -> None:
    specs = parse_topics("pathway X | Escherichia coli | not-a-number\n| organism | 2\n")
    assert len(specs) == 1
    assert specs[0].count == DEFAULT_PER_TOPIC


@pytest.mark.parametrize(
    "token",
    ["PMC4412817", "pmc123", "12345678", "10.1038/nature12345"],
)
def test_pinned_id_detection_positive(token: str) -> None:
    specs = parse_topics(token)
    assert specs[0].is_pinned, token
    assert specs[0].pinned_id == token


@pytest.mark.parametrize(
    "token",
    ["lipid A biosynthesis", "menaquinone", "2019", "PMC", "glycolysis 10.1"],
)
def test_pinned_id_detection_negative(token: str) -> None:
    specs = parse_topics(token)
    assert not specs[0].is_pinned, token
    assert specs[0].topic == token


# ---------------------------------------------------------------------------
# slugs
# ---------------------------------------------------------------------------
def test_slug_strips_windows_illegal_characters() -> None:
    slug = make_slug(
        "PMC999", 'Lipid A: a review / part 2 <draft> "final" | v3?*', set()
    )
    assert not any(ch in slug for ch in _WINDOWS_ILLEGAL)
    assert slug.startswith("PMC999__")
    assert " " not in slug
    assert not slug.endswith(".")
    assert not slug.endswith(" ")


def test_slug_is_length_bounded_for_a_very_long_title() -> None:
    long_title = " ".join(["biosynthesis"] * 40)
    slug = make_slug("PMC4412817", long_title, set())
    assert len(slug) <= 60
    assert slug.startswith("PMC4412817__")
    assert not slug.endswith("-")


def test_slug_uniqueness_on_colliding_titles() -> None:
    used: set = set()
    first = make_slug("PMC1", "Same Title", used)
    second = make_slug("PMC1", "Same Title", used)
    assert first != second
    assert len(second) <= 60
    assert {first, second} == used


def test_slug_survives_a_doi_id_and_an_empty_title() -> None:
    slug = make_slug("10.1038/nature12345", "", set())
    assert not any(ch in slug for ch in _WINDOWS_ILLEGAL)
    assert slug and not slug.endswith(".")


# ---------------------------------------------------------------------------
# fetch_papers
# ---------------------------------------------------------------------------
def test_empty_full_text_becomes_a_skip_not_a_paper() -> None:
    search = _searcher(
        {"topic A": [_candidate("PMC1", "Open access"), _candidate("PMC2", "Paywalled")]}
    )
    papers, skipped = fetch_papers(
        [TopicSpec(topic="topic A", organism="Escherichia coli", count=3)],
        search_fn=search,
        fetch_text_fn=_texter({"PMC1": "full body text"}),
    )
    assert [p.paper_id for p in papers] == ["PMC1"]
    assert papers[0].full_text == "full body text"
    assert [(s["paper_id"], s["reason"]) for s in skipped] == [("PMC2", "no_full_text")]


def test_pinned_id_skips_search_and_fetches_directly() -> None:
    search = _searcher({})
    papers, skipped = fetch_papers(
        [TopicSpec(pinned_id="PMC4412817")],
        search_fn=search,
        fetch_text_fn=_texter({"PMC4412817": "pinned body"}),
    )
    assert search.calls == []  # type: ignore[attr-defined]
    assert len(papers) == 1
    assert papers[0].paper_id == "PMC4412817"
    assert papers[0].topic == "" and papers[0].query == ""
    assert skipped == []


def test_pinned_id_without_full_text_is_skipped() -> None:
    papers, skipped = fetch_papers(
        [TopicSpec(pinned_id="PMC404")],
        search_fn=_searcher({}),
        fetch_text_fn=_texter({}),
    )
    assert papers == []
    assert len(skipped) == 1
    record = skipped[0]
    assert record["paper_id"] == "PMC404"
    assert record["reason"] == "no_full_text"
    assert record["pinned"] is True


def test_deduplicates_the_same_paper_across_topics() -> None:
    shared = _candidate("PMC1", "Shared hit")
    search = _searcher({"topic A": [shared], "topic B": [shared]})
    papers, skipped = fetch_papers(
        [TopicSpec(topic="topic A", count=2), TopicSpec(topic="topic B", count=2)],
        search_fn=search,
        fetch_text_fn=_texter({"PMC1": "body"}),
    )
    assert [p.paper_id for p in papers] == ["PMC1"]
    assert papers[0].topic == "topic A"
    assert [s["reason"] for s in skipped] == ["duplicate"]
    assert len({p.slug for p in papers}) == len(papers)


def test_limit_caps_total_papers_across_topics() -> None:
    search = _searcher(
        {
            "topic A": [_candidate(f"PMCA{i}", f"A {i}") for i in range(4)],
            "topic B": [_candidate(f"PMCB{i}", f"B {i}") for i in range(4)],
        }
    )
    texts = {f"PMCA{i}": "body" for i in range(4)}
    texts.update({f"PMCB{i}": "body" for i in range(4)})
    papers, _ = fetch_papers(
        [TopicSpec(topic="topic A", count=4), TopicSpec(topic="topic B", count=4)],
        limit=3,
        search_fn=search,
        fetch_text_fn=_texter(texts),
    )
    assert len(papers) == 3
    assert len({p.slug for p in papers}) == 3


def test_per_topic_count_is_respected() -> None:
    search = _searcher(
        {"topic A": [_candidate(f"PMC{i}", f"T {i}") for i in range(5)]}
    )
    texts = {f"PMC{i}": "body" for i in range(5)}
    papers, _ = fetch_papers(
        [TopicSpec(topic="topic A", count=2)],
        search_fn=search,
        fetch_text_fn=_texter(texts),
    )
    assert len(papers) == 2


def test_never_raises_when_the_fetcher_returns_nothing() -> None:
    papers, skipped = fetch_papers(
        parse_topics("topic A | Escherichia coli | 3\nPMC1\n"),
        search_fn=lambda context, **kwargs: [],
        fetch_text_fn=lambda candidate, **kwargs: "",
    )
    assert papers == []
    reasons = sorted(s["reason"] for s in skipped)
    assert reasons == ["no_candidates", "no_full_text"]


def test_never_raises_when_search_and_fetch_blow_up() -> None:
    def _boom_search(context: Dict[str, Any], **kwargs: Any) -> List[CandidatePaper]:
        raise RuntimeError("network on fire")

    def _boom_text(candidate: CandidatePaper, **kwargs: Any) -> str:
        raise OSError("socket closed")

    papers, skipped = fetch_papers(
        [TopicSpec(topic="topic A", count=2), TopicSpec(pinned_id="PMC1")],
        search_fn=_boom_search,
        fetch_text_fn=_boom_text,
    )
    assert papers == []
    assert sorted(s["reason"] for s in skipped) == ["fetch_failed", "search_failed"]
    assert any("RuntimeError" in s["detail"] for s in skipped)


def test_fetch_papers_accepts_a_raw_topics_string_and_records_the_query() -> None:
    search = _searcher({"topic A": [_candidate("PMC1", "Hit")]})
    papers, _ = fetch_papers(
        "# comment\ntopic A | Escherichia coli | 1\n",
        search_fn=search,
        fetch_text_fn=_texter({"PMC1": "body"}),
    )
    assert len(papers) == 1
    assert papers[0].query == '"topic A"'
    assert papers[0].topic == "topic A"
    assert papers[0].organism == "Escherichia coli"
    ctx = search.calls[0]["context"]  # type: ignore[attr-defined]
    assert ctx["pathway_name"] == "topic A"
    assert ctx["likely_organism"] == "Escherichia coli"


def test_monkeypatched_module_defaults_are_used_when_no_fakes_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No network: patch the module-level Stage R1 helpers instead of injecting."""
    import t2pw.batch.fetch as batch_fetch

    monkeypatch.setattr(
        batch_fetch,
        "search_candidates",
        lambda context, **kwargs: [_candidate("PMC7", "Patched hit")],
    )
    monkeypatch.setattr(
        batch_fetch, "fetch_full_text", lambda candidate, **kwargs: "patched body"
    )
    papers, skipped = fetch_papers([TopicSpec(topic="topic A", count=1)])
    assert [p.paper_id for p in papers] == ["PMC7"]
    assert skipped == []


def test_describe_is_readable_and_excludes_the_full_text() -> None:
    paper = BatchPaper(
        paper_id="PMC1",
        title="Lipid A biosynthesis",
        source_uri="https://europepmc.org/article/PMC/PMC1",
        year="2019",
        organism="Escherichia coli",
        topic="lipid A biosynthesis",
        query='"lipid A biosynthesis" AND "Escherichia coli"',
        full_text="beta-hydroxymyristoyl body text",
        slug="PMC1__lipid-a-biosynthesis",
    )
    body = paper.describe()
    assert "PMC1" in body
    assert "Lipid A biosynthesis" in body
    assert "Escherichia coli" in body
    assert "characters : 31" in body
    assert "beta-hydroxymyristoyl body text" not in body
    assert paper.to_dict()["slug"] == "PMC1__lipid-a-biosynthesis"


def test_no_full_text_candidate_uses_prefetched_text_when_present() -> None:
    """A candidate that already carries full_text needs no second fetch."""
    prefetched = _candidate("PMC5", "Already loaded")
    prefetched.full_text = "inline body"

    def _never(candidate: CandidatePaper, **_: Any) -> str:
        raise AssertionError("should not re-fetch an already-loaded candidate")

    papers, skipped = fetch_papers(
        [TopicSpec(topic="topic A", count=1)],
        search_fn=_searcher({"topic A": [prefetched]}),
        fetch_text_fn=_never,
    )
    assert [p.full_text for p in papers] == ["inline body"]
    assert skipped == []
