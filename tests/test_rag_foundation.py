"""WP0 foundation tests for the RAG subsystem (``t2pw.rag``).

Covers: config defaults, the in-memory vector-store backend
(upsert/query/persist), the embedding cache hit/miss + lexical fallback, and
that the guarded ``chromadb`` import never breaks importing the package.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.config import rag_config, rag_enabled  # noqa: E402
from t2pw.rag.embed import Embedder, _lexical_embedding  # noqa: E402
from t2pw.rag.provenance import RAG_ADDITIVE_KEYS  # noqa: E402
from t2pw.rag.store import (  # noqa: E402
    Chunk,
    MemoryVectorStore,
    UpsertReport,
    VectorStore,
    get_vector_store,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
_RAG_ENV_VARS = [
    "RAG_ENABLED",
    "RAG_VECTOR_BACKEND",
    "RAG_INDEX_DIR",
    "RAG_EMBEDDING_PROVIDER",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_BASE_URL",
    "RAG_EMBEDDING_API_KEY",
    "RAG_EMBEDDING_DIM",
    "RAG_ACQUIRE_MAX_PAPERS",
    "RAG_SELECT_MAX_PAPERS",
    "RAG_RETRIEVE_TOP_K",
]


@pytest.fixture()
def clean_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neutralize the developer's .env so config defaults are tested in isolation:
    # mark dotenv already-loaded so rag_config()'s ensure_dotenv_loaded() is a
    # no-op and cannot repopulate the RAG_* vars we delete below.
    import t2pw.config as _config

    monkeypatch.setattr(_config, "_ENV_LOADED", True, raising=False)
    for var in _RAG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "local")


def test_rag_config_defaults(clean_rag_env: None) -> None:
    cfg = rag_config()
    assert cfg["enabled"] is False
    assert rag_enabled() is False
    assert cfg["vector_backend"] == "chroma"
    assert cfg["index_dir"] == "data/rag_index"
    assert cfg["embedding_model"] == ""
    assert cfg["embedding_base_url"] == ""
    assert cfg["embedding_api_key"] == ""
    assert cfg["embedding_dim"] == 0
    assert cfg["acquire_max_papers"] == 20
    assert cfg["select_max_papers"] == 8
    assert cfg["retrieve_top_k"] == 8
    # Blank provider falls back to LLM_PROVIDER.
    assert cfg["embedding_provider"] == "local"


def test_rag_config_env_and_overrides(
    clean_rag_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "MEMORY")
    monkeypatch.setenv("RAG_RETRIEVE_TOP_K", "5")
    cfg = rag_config()
    assert cfg["enabled"] is True
    assert rag_enabled() is True
    assert cfg["vector_backend"] == "memory"  # normalized to lowercase
    assert cfg["retrieve_top_k"] == 5
    # Overrides win over the environment.
    over = rag_config({"enabled": "false", "retrieve_top_k": 9})
    assert over["enabled"] is False
    assert over["retrieve_top_k"] == 9


def test_rag_config_bad_int_falls_back(
    clean_rag_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_ACQUIRE_MAX_PAPERS", "not-a-number")
    assert rag_config()["acquire_max_papers"] == 20


def test_embedder_routes_to_dedicated_endpoint(clean_rag_env: None, tmp_path: Path) -> None:
    """With embedding_base_url set, the embedder uses a dedicated client (e.g. LM
    Studio) rather than the shared chat client, and returns its vectors."""

    class _FakeResp:
        def __init__(self, vec: list) -> None:
            self.data = [type("_D", (), {"embedding": vec})()]

    class _FakeEmbeddings:
        def __init__(self) -> None:
            self.calls: list = []

        def create(self, *, model: str, input: str) -> "_FakeResp":  # noqa: A002
            self.calls.append((model, input))
            return _FakeResp([0.1, 0.2, 0.3])

    class _FakeClient:
        def __init__(self) -> None:
            self.embeddings = _FakeEmbeddings()

    cfg = rag_config(
        {
            "embedding_base_url": "http://127.0.0.1:1234/v1",
            "embedding_api_key": "lm-studio",
            "embedding_model": "test-embed-model",
        }
    )
    emb = Embedder(config=cfg, cache_path=tmp_path / "cache.json")
    assert emb.base_url == "http://127.0.0.1:1234/v1"
    emb._api_client = _FakeClient()  # inject to avoid constructing a real client
    vec = emb.embed_one("hello world")
    assert vec == [0.1, 0.2, 0.3]
    assert emb.stats["api"] == 1
    assert emb.stats["lexical"] == 0
    assert emb._api_client.embeddings.calls == [("test-embed-model", "hello world")]


# ---------------------------------------------------------------------------
# Memory backend: upsert / query / persist
# ---------------------------------------------------------------------------
def _chunk(cid: str, text: str, **meta: str) -> Chunk:
    return Chunk(id=cid, text=text, **meta)


def test_memory_backend_upsert_and_query() -> None:
    store = MemoryVectorStore()
    assert isinstance(store, VectorStore)  # runtime_checkable Protocol
    report = store.upsert(
        [
            _chunk("c1", "glucose is phosphorylated to glucose-6-phosphate", organism="human"),
            _chunk("c2", "the citric acid cycle oxidizes acetyl-CoA", organism="human"),
            _chunk("c3", "photosynthesis fixes carbon dioxide in plants", organism="plant"),
        ]
    )
    assert isinstance(report, UpsertReport)
    assert report.upserted == 3
    assert report.total == 3

    hits = store.query("glucose phosphate", top_k=2)
    assert hits, "expected at least one lexical hit"
    assert hits[0].chunk.id == "c1"
    assert hits[0].score > 0.0
    assert len(hits) <= 2


def test_memory_backend_upsert_skips_bad_and_dedupes() -> None:
    store = MemoryVectorStore()
    report = store.upsert([_chunk("", "no id"), _chunk("c1", "first"), _chunk("c1", "updated")])
    assert report.skipped == 1
    assert report.total == 1  # same id upserted, not duplicated
    hits = store.query("updated")
    assert hits[0].chunk.text == "updated"


def test_memory_backend_filters() -> None:
    store = MemoryVectorStore()
    store.upsert(
        [
            _chunk("c1", "carbon dioxide fixation", organism="plant"),
            _chunk("c2", "carbon dioxide transport", organism="human"),
        ]
    )
    hits = store.query("carbon dioxide", filters={"organism": "human"})
    assert [h.chunk.id for h in hits] == ["c2"]


def test_memory_backend_persist_roundtrip(tmp_path: Path) -> None:
    persist = tmp_path / "mem_store.json"
    store = MemoryVectorStore(persist_path=persist)
    store.upsert([_chunk("c1", "alpha ketoglutarate dehydrogenase"), _chunk("c2", "malate")])
    store.persist()
    assert persist.exists()

    reloaded = MemoryVectorStore(persist_path=persist)
    assert reloaded.stats()["count"] == 2
    hits = reloaded.query("malate")
    assert hits[0].chunk.id == "c2"


def test_memory_backend_uses_embeddings_when_available() -> None:
    # A deterministic 3-dim embed_fn: query "a" should match the "a"-heavy chunk.
    def embed_fn(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            out.append([float(t.count("a")), float(t.count("b")), float(t.count("c"))])
        return out

    store = MemoryVectorStore(embed_fn=embed_fn)
    store.upsert([_chunk("ca", "aaaa"), _chunk("cb", "bbbb")])
    hits = store.query("aaaa")
    assert hits[0].chunk.id == "ca"


# ---------------------------------------------------------------------------
# Embedding cache hit / miss + lexical fallback
# ---------------------------------------------------------------------------
def test_embedding_cache_hit_miss(tmp_path: Path) -> None:
    cache = tmp_path / "embeddings_cache.json"
    emb = Embedder(model="", dim=32, cache_path=cache, online=False)

    v1 = emb.embed_one("acetyl-CoA")
    assert emb.stats["misses"] == 1
    assert emb.stats["hits"] == 0
    assert emb.stats["lexical"] == 1
    assert len(v1) == 32

    v2 = emb.embed_one("acetyl-CoA")  # identical text -> cache hit
    assert emb.stats["hits"] == 1
    assert emb.stats["misses"] == 1
    assert v2 == v1

    emb.embed_one("pyruvate")  # different text -> another miss
    assert emb.stats["misses"] == 2


def test_embedding_cache_persists_across_instances(tmp_path: Path) -> None:
    cache = tmp_path / "embeddings_cache.json"
    first = Embedder(model="", dim=16, cache_path=cache, online=False)
    original = first.embed_one("citrate")
    first.save()
    assert cache.exists()

    second = Embedder(model="", dim=16, cache_path=cache, online=False)
    restored = second.embed_one("citrate")  # served from disk cache
    assert second.stats["hits"] == 1
    assert second.stats["misses"] == 0
    assert restored == original


def test_lexical_embedding_is_deterministic_and_normalized() -> None:
    a = _lexical_embedding("succinate dehydrogenase", 64)
    b = _lexical_embedding("succinate dehydrogenase", 64)
    assert a == b
    norm = sum(x * x for x in a) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Guarded import: importing the package must not require chromadb
# ---------------------------------------------------------------------------
def test_import_rag_without_chromadb() -> None:
    # chromadb is not installed in the test environment; importing the package
    # and the store module must still succeed (lazy backend import).
    assert importlib.import_module("t2pw.rag") is not None
    assert importlib.import_module("t2pw.rag.store") is not None
    # The memory backend is always available with no optional dependency.
    assert isinstance(get_vector_store(backend="memory"), VectorStore)


def test_chroma_backend_missing_dep_raises_clear_error() -> None:
    pytest.importorskip  # keep import local intent explicit
    try:
        import chromadb  # noqa: F401,PLC0415
    except Exception:
        with pytest.raises(RuntimeError, match="chromadb"):
            get_vector_store(backend="chroma")
    else:
        pytest.skip("chromadb is installed; guarded-error path not exercised")


def test_provenance_additive_keys() -> None:
    # The primary pointer key is namespaced ``rag_provenance`` (never the bare
    # ``provenance``) so it can never collide with the core string field.
    assert RAG_ADDITIVE_KEYS == ("rag_provenance", "evidence", "source_papers", "rag_confidence")
    assert "provenance" not in RAG_ADDITIVE_KEYS


def test_memory_query_dim_mismatch_falls_back_to_lexical() -> None:
    # Regression: a chunk embedded at one width (e.g. a lexical-fallback vector
    # cached while the embeddings endpoint was down) sitting beside a query
    # vector of another width must NOT score a silent 0.0 via cosine — it falls
    # back to lexical token overlap so retrieval still ranks by real relevance.
    store = MemoryVectorStore(embed_fn=lambda texts: [[0.1] * 768 for _ in texts])
    hit_chunk = Chunk(id="c1", text="caffeine is demethylated to theobromine by NdmA")
    hit_chunk.embedding = [0.2] * 256  # different width than the 768-dim query
    miss_chunk = Chunk(id="c2", text="photosystem II splits water in the thylakoid")
    miss_chunk.embedding = [0.2] * 256
    store._chunks["c1"] = hit_chunk
    store._chunks["c2"] = miss_chunk

    hits = store.query("caffeine theobromine", top_k=2)
    assert hits and hits[0].chunk.id == "c1"
    assert hits[0].score > 0.0  # was a silent 0.0 before the dim guard
