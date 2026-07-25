"""WP2 selection tests for ``t2pw.rag.select``.

All tests are deterministic and offline: the reused preprocessor
(``t2pw.rag.select.preprocess``) is monkeypatched with a canned, per-text lookup
so no network / LLM call is made. They cover: score component breakdown, organism
match / mismatch, seed-term overlap, that a ``multi_example_review`` with no
seed-matching example is dropped (and ranked below on-topic primary research),
that a review whose example DOES match the seed is example-scoped and ranked
below primary research, dedup, the ``max_papers`` cap, and that the
``selection_report`` explains every drop.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ``t2pw.rag.select`` imports the reused preprocessor, which pulls in
# ``t2pw.llm.client`` (``from openai import OpenAI``). Per the repo convention,
# any test that pulls in the llm client stubs ``openai`` itself so it runs in
# isolation (not only when an earlier test happens to have stubbed it first).
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

from t2pw.rag.acquire import CandidatePaper  # noqa: E402
from t2pw.rag import select as select_mod  # noqa: E402
from t2pw.rag.select import (  # noqa: E402
    Selection,
    SelectionScore,
    score_candidate,
    select,
)


# ---------------------------------------------------------------------------
# Canned preprocessor: maps a substring of the input text -> a fixed context.
# This replaces the LLM call so the whole module is deterministic.
# ---------------------------------------------------------------------------
def _empty_ctx() -> Dict[str, Any]:
    return {
        "document_type": "primary_research",
        "pathway_name": "",
        "likely_organism": "",
        "key_compounds": [],
        "key_proteins": [],
        "main_subprocesses": [],
        "candidate_examples": [],
        "selected_example": "",
        "pathway_relevance_score": 0.0,
    }


# Marker token -> preprocessor context returned for any text containing it.
_PREPROCESS_TABLE: Dict[str, Dict[str, Any]] = {
    "CAFFEINE_PRIMARY": {
        **_empty_ctx(),
        "document_type": "primary_research",
        "pathway_name": "caffeine degradation",
        "likely_organism": "Pseudomonas putida",
        "key_compounds": ["caffeine", "theobromine", "xanthine"],
        "key_proteins": ["NdmA", "NdmB"],
        "pathway_relevance_score": 0.95,
    },
    "CAFFEINE_WRONG_ORG": {
        **_empty_ctx(),
        "document_type": "primary_research",
        "pathway_name": "caffeine degradation",
        "likely_organism": "Homo sapiens",
        "key_compounds": ["caffeine", "theobromine"],
        "key_proteins": ["CYP1A2"],
        "pathway_relevance_score": 0.6,
    },
    "OFFTOPIC_PRIMARY": {
        **_empty_ctx(),
        "document_type": "primary_research",
        "pathway_name": "glycolysis",
        "likely_organism": "Saccharomyces cerevisiae",
        "key_compounds": ["glucose", "pyruvate"],
        "key_proteins": ["hexokinase"],
        "pathway_relevance_score": 0.9,
    },
    "REVIEW_NO_MATCH": {
        **_empty_ctx(),
        "document_type": "multi_example_review",
        "pathway_name": "",
        "likely_organism": "",
        "selected_example": "",
        "pathway_relevance_score": 0.9,
        "candidate_examples": [
            {
                "name": "pentalenolactone biosynthesis",
                "organism": "Streptomyces exfoliatus",
                "key_compounds": ["pentalenolactone", "pentalenene"],
                "key_proteins": ["PtlD"],
            },
            {
                "name": "obafluorin biosynthesis",
                "organism": "Pseudomonas fluorescens",
                "key_compounds": ["obafluorin", "AHBA"],
                "key_proteins": ["ObaG"],
            },
        ],
    },
    "REVIEW_MATCH": {
        **_empty_ctx(),
        "document_type": "multi_example_review",
        "pathway_name": "",
        "likely_organism": "",
        "selected_example": "",
        "pathway_relevance_score": 0.9,
        "candidate_examples": [
            {
                "name": "pentalenolactone biosynthesis",
                "organism": "Streptomyces exfoliatus",
                "key_compounds": ["pentalenolactone"],
                "key_proteins": ["PtlD"],
            },
            {
                "name": "caffeine degradation",
                "organism": "Pseudomonas putida",
                "key_compounds": ["caffeine", "theobromine"],
                "key_proteins": ["NdmA"],
            },
        ],
    },
}


def _fake_preprocess(text: str, **_: Any) -> Dict[str, Any]:
    for marker, ctx in _PREPROCESS_TABLE.items():
        if marker in text:
            return dict(ctx)
    return _empty_ctx()


@pytest.fixture(autouse=True)
def _patch_preprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the name as looked up inside t2pw.rag.select — no network / LLM.
    monkeypatch.setattr(select_mod, "preprocess", _fake_preprocess)


# ---------------------------------------------------------------------------
# Seed context + candidate builders.
# ---------------------------------------------------------------------------
_SEED = {
    "pathway_name": "caffeine degradation",
    "likely_organism": "Pseudomonas putida",
    "key_compounds": ["caffeine", "theobromine", "xanthine"],
    "key_proteins": ["NdmA", "NdmB"],
    "gap_terms": ["7-methylxanthine"],
}

# A novel/ambiguous seed: Stage 0 could name no pathway, compounds, proteins,
# gaps OR organism -> total_terms == 0 and seed_org == "". Such a seed carries
# no signal to prove a candidate off-topic (C2).
_SPARSE_SEED = {
    "pathway_name": "",
    "likely_organism": "",
    "key_compounds": [],
    "key_proteins": [],
    "gap_terms": [],
}


def _paper(pid: str, marker: str, *, title: str = "", organism: str = "") -> CandidatePaper:
    return CandidatePaper(
        id=pid,
        source="europepmc",
        title=title or pid,
        abstract=f"Abstract marker {marker}.",
        organism=organism,
    )


# ---------------------------------------------------------------------------
# score_candidate.
# ---------------------------------------------------------------------------
def test_score_on_topic_primary_is_high() -> None:
    score = score_candidate(_paper("PMC1", "CAFFEINE_PRIMARY"), _SEED)
    assert isinstance(score, SelectionScore)
    assert score.organism_match == 1.0
    assert score.entity_overlap > 0.5
    assert score.pathway_relevance == pytest.approx(0.95)
    assert score.review_penalty == 0.0
    assert score.score > 0.7
    assert "caffeine" in score.matched_terms


def test_score_is_deterministic() -> None:
    paper = _paper("PMC1", "CAFFEINE_PRIMARY")
    a = score_candidate(paper, _SEED)
    b = score_candidate(paper, _SEED)
    assert a.score == b.score
    assert a.matched_terms == b.matched_terms


def test_organism_mismatch_lowers_score() -> None:
    right = score_candidate(_paper("PMC1", "CAFFEINE_PRIMARY"), _SEED)
    wrong = score_candidate(_paper("PMC2", "CAFFEINE_WRONG_ORG"), _SEED)
    assert wrong.organism_match == 0.0
    assert wrong.score < right.score
    assert any("organism mismatch" in n for n in wrong.notes)


def test_offtopic_primary_has_no_overlap() -> None:
    score = score_candidate(_paper("PMC9", "OFFTOPIC_PRIMARY"), _SEED)
    assert score.entity_overlap == 0.0
    assert score.matched_terms == []


def test_review_no_matching_example_is_penalized_and_negative() -> None:
    score = score_candidate(_paper("PMC5", "REVIEW_NO_MATCH"), _SEED)
    assert score.document_type == "multi_example_review"
    assert score.review_penalty == pytest.approx(select_mod._REVIEW_PENALTY_NO_MATCH)
    assert score.score < 0.0  # driven below the drop floor
    assert any("multi_example_review" in n for n in score.notes)


def test_review_with_matching_example_is_example_scoped() -> None:
    score = score_candidate(_paper("PMC6", "REVIEW_MATCH"), _SEED)
    assert score.review_penalty == pytest.approx(select_mod._REVIEW_PENALTY_MATCH)
    # Penalized but not necessarily negative — the point is it ranks below
    # on-topic primary research (asserted in the select-level test below).
    assert any("example-scoped" in n for n in score.notes)


# ---------------------------------------------------------------------------
# select: ranking, drop of non-matching review, report explains every drop.
# ---------------------------------------------------------------------------
def test_non_matching_review_dropped_below_primary() -> None:
    candidates = [
        _paper("PMC_REVIEW", "REVIEW_NO_MATCH"),
        _paper("PMC_PRIMARY", "CAFFEINE_PRIMARY"),
    ]
    result = select(candidates, _SEED, max_papers=8)
    assert isinstance(result, Selection)
    kept_ids = [p.id for p in result.selected]
    assert "PMC_PRIMARY" in kept_ids
    assert "PMC_REVIEW" not in kept_ids  # non-matching review is dropped

    drops = {e.paper_id: e.reason for e in result.dropped_entries()}
    assert "PMC_REVIEW" in drops
    assert "multi_example_review" in drops["PMC_REVIEW"]


def test_matching_review_ranks_below_primary() -> None:
    candidates = [
        _paper("PMC_REVIEW", "REVIEW_MATCH"),
        _paper("PMC_PRIMARY", "CAFFEINE_PRIMARY"),
    ]
    result = select(candidates, _SEED, max_papers=8)
    kept_ids = [p.id for p in result.selected]
    # If the matching review survives the floor it must rank below the primary
    # paper; the primary is always first.
    assert kept_ids[0] == "PMC_PRIMARY"
    if "PMC_REVIEW" in kept_ids:
        assert kept_ids.index("PMC_REVIEW") > kept_ids.index("PMC_PRIMARY")


def test_every_drop_is_explained() -> None:
    candidates = [
        _paper("PMC_PRIMARY", "CAFFEINE_PRIMARY"),
        _paper("PMC_REVIEW", "REVIEW_NO_MATCH"),
        _paper("PMC_OFFTOPIC", "OFFTOPIC_PRIMARY"),
    ]
    result = select(candidates, _SEED, max_papers=8)
    # One report entry per candidate, and every dropped entry has a non-empty
    # reason.
    assert len(result.selection_report) == len(candidates)
    for entry in result.dropped_entries():
        assert entry.reason.startswith("dropped:")
        assert entry.reason.strip() != "dropped:"


def test_dedup_by_identity() -> None:
    dup_a = _paper("PMC777", "CAFFEINE_PRIMARY", title="Caffeine degradation study")
    dup_b = _paper("PMC777", "CAFFEINE_PRIMARY", title="Caffeine degradation study")
    result = select([dup_a, dup_b], _SEED, max_papers=8)
    assert [p.id for p in result.selected].count("PMC777") == 1
    dup_drops = [
        e for e in result.dropped_entries() if "duplicate" in e.reason
    ]
    assert len(dup_drops) == 1


def test_cap_limits_selection_and_explains_overflow() -> None:
    candidates = [
        _paper(f"PMC{i}", "CAFFEINE_PRIMARY", title=f"Caffeine paper {i}")
        for i in range(5)
    ]
    result = select(candidates, _SEED, max_papers=2)
    assert len(result.selected) == 2
    over_cap = [e for e in result.dropped_entries() if "exceeds cap" in e.reason]
    assert len(over_cap) == 3


def test_cap_defaults_to_rag_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SELECT_MAX_PAPERS", "1")
    candidates = [
        _paper(f"PMC{i}", "CAFFEINE_PRIMARY", title=f"Caffeine paper {i}")
        for i in range(3)
    ]
    result = select(candidates, _SEED)  # no max_papers -> uses rag_config()
    assert len(result.selected) == 1


def test_empty_candidates() -> None:
    result = select([], _SEED, max_papers=8)
    assert result.selected == []
    assert result.selection_report == []
    assert result.scores == {}


def test_preprocess_called_once_per_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def _counting(text: str, **_: Any) -> Dict[str, Any]:
        calls.append(text)
        return _fake_preprocess(text)

    monkeypatch.setattr(select_mod, "preprocess", _counting)
    candidates = [
        _paper("PMC1", "CAFFEINE_PRIMARY"),
        _paper("PMC2", "OFFTOPIC_PRIMARY"),
    ]
    select(candidates, _SEED, max_papers=8)
    assert len(calls) == 2  # exactly once per candidate, not twice


def test_select_survives_a_failing_preprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: a flaky preprocess (rate limit / timeout) on one candidate must
    # not abort the whole selection (which would silently disable RAG). The
    # failing candidate degrades to an organism-only context; the rest still score.
    def _boom(text: str, **_: Any) -> Dict[str, Any]:
        if "BOOM" in text:
            raise RuntimeError("rate limited")
        return _fake_preprocess(text)

    monkeypatch.setattr(select_mod, "preprocess", _boom)
    candidates = [
        _paper("PMC1", "CAFFEINE_PRIMARY"),
        _paper("PMCBOOM", "BOOM", organism="Pseudomonas putida"),
    ]
    result = select(candidates, _SEED, max_papers=8)  # must not raise
    assert "PMC1" in result.scores and "PMCBOOM" in result.scores
    assert any(e.paper_id == "PMC1" and e.kept for e in result.selection_report)


# ---------------------------------------------------------------------------
# Issue C — related papers must be selectable (C1 scope threading, C2 sparse
# seed penalty downgrade).
# ---------------------------------------------------------------------------
def test_select_threads_seed_scope_into_candidate_preprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # C1: select() must thread the seed's scope (selected_example, else
    # pathway_name) into each candidate's preprocess. A review about the seed
    # pathway then reaches the matched-example case (Case B, 0.25 penalty) and
    # survives the floor, instead of collapsing to a blank-scope no-match review
    # (Case C, 1.0 penalty) and being force-dropped.
    seen_ctx: List[Any] = []

    def _scope_aware(text: str, *, user_task_context: Any = None, **_: Any) -> Dict[str, Any]:
        if "REVIEW_SCOPED" in text:
            seen_ctx.append(user_task_context)
            focus = str(user_task_context or "").casefold()
            # With the seed's focus threaded in, the review resolves to a
            # seed-matching example; without it, it is a blank-scope no-match.
            if "caffeine" in focus:
                return dict(_PREPROCESS_TABLE["REVIEW_MATCH"])
            return dict(_PREPROCESS_TABLE["REVIEW_NO_MATCH"])
        return _fake_preprocess(text)

    monkeypatch.setattr(select_mod, "preprocess", _scope_aware)
    candidates = [
        _paper("PMC_PRIMARY", "CAFFEINE_PRIMARY"),
        _paper("PMC_REVIEW", "REVIEW_SCOPED"),
    ]
    result = select(candidates, _SEED, max_papers=8)

    # The seed's pathway_name was threaded into the candidate preprocess call.
    assert any("caffeine" in str(c or "").casefold() for c in seen_ctx)

    kept_ids = [p.id for p in result.selected]
    assert "PMC_REVIEW" in kept_ids  # example-scoped, kept (not force-dropped)
    review_score = result.scores["PMC_REVIEW"]
    assert review_score.review_penalty == pytest.approx(
        select_mod._REVIEW_PENALTY_MATCH
    )
    assert review_score.score >= select_mod._MIN_SCORE


def test_sparse_seed_downgrades_no_match_review_and_keeps_it() -> None:
    # C2: a sparse seed (blank scope AND blank organism -> total_terms == 0,
    # seed_org == "") cannot prove a review off-topic, so the force-drop no-match
    # penalty (1.0) is downgraded to the rank-below penalty (0.25); the review
    # then survives the floor instead of being force-dropped.
    score = score_candidate(_paper("PMC_REVIEW", "REVIEW_NO_MATCH"), _SPARSE_SEED)
    assert score.document_type == "multi_example_review"
    assert score.review_penalty == pytest.approx(select_mod._REVIEW_PENALTY_MATCH)
    assert score.score >= select_mod._MIN_SCORE

    # And at the select level the sparse-seed review is kept, not dropped.
    result = select([_paper("PMC_REVIEW", "REVIEW_NO_MATCH")], _SPARSE_SEED, max_papers=8)
    assert "PMC_REVIEW" in [p.id for p in result.selected]


def test_signal_bearing_seed_keeps_full_no_match_penalty() -> None:
    # Guard the C2 gate: when the seed HAS signal (the norm), a no-match review
    # keeps the full 1.0 penalty and is still force-dropped below the floor.
    score = score_candidate(_paper("PMC_REVIEW", "REVIEW_NO_MATCH"), _SEED)
    assert score.review_penalty == pytest.approx(select_mod._REVIEW_PENALTY_NO_MATCH)
    assert score.score < select_mod._MIN_SCORE
