"""WP3 ingest & index tests — memory backend, stubbed (offline) embedder.

Self-contained and offline: no chromadb, no live network, no live LLM. The
guarded ``openai`` stub mirrors tests/test_audit_json_llm_payload.py because
importing ``t2pw.rag.select`` (for :class:`Selection`) pulls in
``t2pw.pipeline.preprocessor`` -> ``t2pw.llm.client`` -> ``openai`` at import time.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from t2pw.rag.embed import Embedder  # noqa: E402
from t2pw.rag.ingest import (  # noqa: E402
    SEED_SOURCE_ID,
    build_hybrid_scorer,
    chunk_corpus,
    chunk_db_reactions,
    chunk_paper,
    ingest,
    seed_candidate,
)
from t2pw.rag.select import Selection  # noqa: E402
from t2pw.rag.store import Chunk, MemoryVectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------
_FULL_TEXT = (
    "Introduction. Caffeine is a purine alkaloid widely consumed by humans. "
    "Background on xanthine metabolism motivates this work. "
    "Methods. Bacterial strains were cultured overnight and assayed for activity. "
    "Results. The enzyme NdmA catalyzes the initial N1-demethylation of caffeine, "
    "yielding theobromine and formaldehyde as products. NdmA is a Rieske "
    "oxygenase requiring a reductase partner. "
    "Discussion. These findings clarify the bacterial caffeine catabolism route. "
    "Figure 1. NdmA reaction scheme depicting caffeine converted to theobromine."
)


def _canned_paper() -> CandidatePaper:
    return CandidatePaper(
        id="PMC999001",
        source="europepmc",
        title="Bacterial caffeine N-demethylation by NdmA",
        abstract="This study examines caffeine catabolism in soil bacteria.",
        organism="Pseudomonas putida",
        full_text=_FULL_TEXT,
        source_uri="https://europepmc.org/article/PMC/PMC999001",
        year="2021",
    )


def _offline_embedder(tmp_path: Path) -> Embedder:
    # online=False -> deterministic lexical fallback vectors; no network / LLM.
    return Embedder(
        model="test-embed",
        dim=64,
        cache_path=tmp_path / "embeddings_cache.json",
        online=False,
    )


# ---------------------------------------------------------------------------
# chunk_paper — section-aware + stable ids.
# ---------------------------------------------------------------------------
def test_chunk_paper_is_section_aware_and_carries_provenance() -> None:
    chunks = chunk_paper(_canned_paper())
    sections = {c.section for c in chunks}
    # abstract + detected body sections + figure caption.
    assert "abstract" in sections
    assert "results" in sections
    assert "methods" in sections
    assert "figure" in sections
    for chunk in chunks:
        assert chunk.source_id == "PMC999001"
        assert chunk.source_uri == "https://europepmc.org/article/PMC/PMC999001"
        assert chunk.source_type == "paper"
        assert chunk.organism == "Pseudomonas putida"
    results_chunks = [c for c in chunks if c.section == "results"]
    assert results_chunks
    assert any("ndma" in c.text.lower() for c in results_chunks)


def test_chunk_paper_ids_are_stable_across_calls() -> None:
    ids_a = [c.id for c in chunk_paper(_canned_paper())]
    ids_b = [c.id for c in chunk_paper(_canned_paper())]
    assert ids_a == ids_b
    assert len(ids_a) == len(set(ids_a))  # unique within a paper


# ---------------------------------------------------------------------------
# ingest — end-to-end on the memory backend with a stubbed embedder.
# ---------------------------------------------------------------------------
def test_ingest_end_to_end_query_returns_right_chunk_with_provenance(tmp_path) -> None:
    embedder = _offline_embedder(tmp_path)
    store = MemoryVectorStore(embed_fn=embedder.embed)
    selection = Selection(selected=[_canned_paper()])

    report = ingest(
        selection, store=store, embedder=embedder, index_corpus=False, persist=False
    )
    assert report.papers == 1
    assert report.chunks > 0
    assert report.upserted == report.chunks
    assert report.embedded == report.chunks  # first run: every chunk embedded
    assert report.store_total == report.chunks

    hits = store.query(
        "theobromine and formaldehyde produced by a Rieske oxygenase reductase",
        top_k=1,
    )
    assert hits
    top = hits[0].chunk
    assert top.section == "results"
    assert top.source_id == "PMC999001"
    assert top.source_uri == "https://europepmc.org/article/PMC/PMC999001"
    assert "ndma" in top.text.lower()


def test_reingesting_unchanged_paper_is_a_no_op(tmp_path) -> None:
    embedder = _offline_embedder(tmp_path)
    store = MemoryVectorStore(embed_fn=embedder.embed)
    selection = Selection(selected=[_canned_paper()])

    first = ingest(
        selection, store=store, embedder=embedder, index_corpus=False, persist=False
    )
    assert first.embedded == first.chunks
    assert first.cache_hits == 0

    second = ingest(
        selection, store=store, embedder=embedder, index_corpus=False, persist=False
    )
    # No new embeddings; every chunk served from the cache; store unchanged.
    assert second.embedded == 0
    assert second.cache_hits == second.chunks
    assert second.store_total == first.store_total


# ---------------------------------------------------------------------------
# Exact-symbol retrieval via the lexical half when embeddings are unavailable.
# ---------------------------------------------------------------------------
def test_exact_symbol_retrieves_via_lexical_half_when_embeddings_unavailable() -> None:
    # Embeddings unavailable: no embed_fn, chunks carry no vectors.
    store = MemoryVectorStore(embed_fn=None)
    chunks = chunk_paper(_canned_paper())
    # Add unrelated distractor chunks so an exact-symbol hit must be earned.
    distractors = [
        Chunk(
            id=f"distract-{i}",
            text=text,
            source_id=f"D{i}",
            source_type="paper",
            section="results",
        )
        for i, text in enumerate(
            [
                "Glycolysis converts glucose to pyruvate across ten steps.",
                "The citric acid cycle oxidizes acetyl-CoA in the mitochondrion.",
                "Photosystem II splits water during the light reactions.",
            ]
        )
    ]
    store.upsert(chunks + distractors)

    scorer = build_hybrid_scorer(store)
    hits = scorer("NdmA", top_k=3)
    assert hits
    top = hits[0].chunk
    assert "ndma" in top.text.lower()
    assert top.source_id == "PMC999001"


def test_hybrid_scorer_blends_semantic_and_lexical(tmp_path) -> None:
    embedder = _offline_embedder(tmp_path)
    store = MemoryVectorStore(embed_fn=embedder.embed)
    ingest(
        Selection(selected=[_canned_paper()]),
        store=store,
        embedder=embedder,
        index_corpus=False,
        persist=False,
    )
    scorer = build_hybrid_scorer(store, semantic_weight=0.7, lexical_weight=0.3)
    hits = scorer("NdmA theobromine formaldehyde", top_k=2)
    assert hits
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert "ndma" in hits[0].chunk.text.lower()


# ---------------------------------------------------------------------------
# chunk_db_reactions — one chunk per record, source_type pathbank/kegg.
# ---------------------------------------------------------------------------
def test_chunk_db_reactions_one_per_record_with_source_type() -> None:
    records = [
        {
            "id": "PW_RXN_1",
            "source": "pathbank",
            "name": "caffeine demethylation",
            "equation": "caffeine -> theobromine + formaldehyde",
            "enzymes": ["NdmA"],
            "organism": "Pseudomonas putida",
        },
        {
            "entry": "R01234",
            "source_type": "kegg",
            "definition": "ATP + D-glucose -> ADP + D-glucose 6-phosphate",
        },
    ]
    chunks = chunk_db_reactions(records)
    assert len(chunks) == 2
    by_type = {c.source_type for c in chunks}
    assert by_type == {"pathbank", "kegg"}
    pathbank = next(c for c in chunks if c.source_type == "pathbank")
    assert pathbank.source_id == "PW_RXN_1"
    assert pathbank.section == "reaction"
    assert "ndma" in pathbank.text.lower()
    # Idempotent id: same record -> same id.
    assert chunk_db_reactions(records)[0].id == chunks[0].id


# ---------------------------------------------------------------------------
# chunk_corpus — reference example corpus tagged pwml_example.
# ---------------------------------------------------------------------------
def test_chunk_corpus_tags_pwml_example(tmp_path) -> None:
    pwml = tmp_path / "PWTEST.pwml"
    pwml.write_text(
        "<?xml version='1.0' encoding='utf-8'?>"
        "<super-pathway-visualization>"
        "<cached-name>Test Pathway</cached-name>"
        "<cached-subject>Metabolic</cached-subject>"
        "<name>caffeine</name><name>theobromine</name><name>NdmA</name>"
        "</super-pathway-visualization>",
        encoding="utf-8",
    )
    chunks = chunk_corpus(tmp_path)
    assert chunks
    assert all(c.source_type == "pwml_example" for c in chunks)
    assert all(c.section == "example" for c in chunks)
    assert any("ndma" in c.text.lower() for c in chunks)
    assert chunks[0].source_id == "PWTEST.pwml"


# ---------------------------------------------------------------------------
# The uploaded seed paper is indexed like any other source.
# ---------------------------------------------------------------------------
_SEED_TEXT = """Lipid A biosynthesis (Raetz pathway).
Introduction
Lipid A anchors lipopolysaccharide in Gram-negative bacteria.
Results
LpxH cleaves most of the UDP moiety to leave lipid X.
References
1. Raetz CRH et al. Annu Rev Biochem.
"""


def test_seed_candidate_is_none_for_empty_text() -> None:
    """Callers pass it unconditionally, so empty input must not fabricate a paper."""

    assert seed_candidate("") is None
    assert seed_candidate("   \n  ") is None


def test_seed_candidate_uses_the_provenance_sentinel() -> None:
    candidate = seed_candidate(_SEED_TEXT, title="lipid A biosynthesis")

    # Must match the source_id seed-derived provenance already carries, or the
    # seed's own passages would not join to its own reactions.
    assert candidate.id == SEED_SOURCE_ID
    assert candidate.title == "lipid A biosynthesis"


def test_seed_candidate_falls_back_to_a_usable_title() -> None:
    assert seed_candidate(_SEED_TEXT).title == "uploaded seed paper"


def test_seed_is_chunked_with_sections_and_stable_ids() -> None:
    chunks = chunk_paper(seed_candidate(_SEED_TEXT, title="lipid A biosynthesis"))

    assert chunks, "the seed must produce retrievable passages"
    assert {c.source_id for c in chunks} == {SEED_SOURCE_ID}
    assert "results" in {c.section for c in chunks}
    assert all(c.id for c in chunks), "every seed chunk needs a chunk id to be citable"
    # The reference list is back matter, not evidence -- same rule as any paper.
    assert "references" not in {c.section for c in chunks}


def test_ingest_indexes_the_seed_alongside_the_selected_papers() -> None:
    store = MemoryVectorStore()
    report = ingest(
        Selection(selected=[_canned_paper()]),
        store=store,
        embedder=Embedder(config={"embedding_provider": "offline"}),
        index_corpus=False,
        persist=False,
        seed=seed_candidate(_SEED_TEXT, title="lipid A biosynthesis"),
    )

    indexed = {c.source_id for c in store.all_chunks()} if hasattr(store, "all_chunks") else set()
    assert report.papers == 2, "the seed counts as an indexed paper"
    assert report.chunks > 0
    if indexed:
        assert SEED_SOURCE_ID in indexed


def test_ingest_without_a_seed_is_unchanged() -> None:
    """The default path must be byte-identical to before seed indexing existed."""

    store = MemoryVectorStore()
    report = ingest(
        Selection(selected=[_canned_paper()]),
        store=store,
        embedder=Embedder(config={"embedding_provider": "offline"}),
        index_corpus=False,
        persist=False,
    )

    assert report.papers == 1
