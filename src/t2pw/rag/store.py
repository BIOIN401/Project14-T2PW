"""Vector store interface and backends for the RAG subsystem (WP0).

This module owns the ``VectorStore`` ``Protocol``, the ``Chunk`` / ``Retrieved``
record dataclasses, and two backends:

* ``MemoryVectorStore`` — a dependency-free in-memory backend used by tests and
  by any offline run. It supports optional JSON persistence.
* ``ChromaVectorStore`` — the default persistent backend, backed by ``chromadb``.

The ``chromadb`` import is **lazy and guarded**: importing this module (or the
``t2pw.rag`` package) never imports ``chromadb``. A missing ``chromadb`` only
surfaces as a clear ``RuntimeError`` when a caller actually constructs the
chroma backend, so the core pipeline and existing tests never depend on it.

Per the separation invariant, nothing here is imported by any core stage; the
dependency arrow points RAG -> core only.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from t2pw.config import rag_config
from t2pw.paths import PROJECT_ROOT

# An embedding function maps a batch of texts to a batch of vectors. It is
# supplied by ``t2pw.rag.embed`` in real runs and may be omitted in tests, in
# which case the backends fall back to lexical token-overlap scoring.
EmbedFn = Callable[[List[str]], List[List[float]]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.casefold())


def _lexical_overlap(query: str, text: str) -> float:
    """Jaccard-style token overlap in [0, 1]; the offline query fallback."""
    q = set(_tokenize(query))
    d = set(_tokenize(text))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Chunk:
    """A single embedded passage of a paper or database record.

    Mirrors the schema in docs/rag/02_vector_store.md. ``embedding`` is ``None``
    until the embedder fills it. ``source_id`` + ``source_uri`` are what WP5/WP6
    later attach as provenance to synthesized elements.
    """

    id: str
    text: str
    source_id: str = ""
    source_title: str = ""
    source_type: str = ""  # "paper" | "pathbank" | "kegg" | "pwml_example"
    source_uri: str = ""
    organism: str = ""
    section: str = ""  # "abstract" | "results" | "methods" | "figure" | ...
    pathway_tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def metadata(self) -> Dict[str, Any]:
        """Flat metadata dict used for filtering and provenance carry-through."""
        return {
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "organism": self.organism,
            "section": self.section,
            "pathway_tags": list(self.pathway_tags),
        }


@dataclass
class Retrieved:
    """One retrieval hit: the stored chunk plus its similarity score."""

    chunk: Chunk
    score: float


@dataclass
class UpsertReport:
    """Summary returned by ``VectorStore.upsert``."""

    upserted: int = 0
    skipped: int = 0
    total: int = 0
    backend: str = ""


@runtime_checkable
class VectorStore(Protocol):
    """The pluggable vector-store contract (chroma | faiss | memory)."""

    def upsert(self, chunks: List[Chunk]) -> UpsertReport: ...

    def query(
        self, text: str, *, top_k: int = 8, filters: Optional[dict] = None
    ) -> List[Retrieved]: ...

    def persist(self) -> None: ...

    def stats(self) -> dict: ...


def _matches_filters(meta: Dict[str, Any], filters: Optional[dict]) -> bool:
    if not filters:
        return True
    for key, want in filters.items():
        have = meta.get(key)
        if isinstance(have, list):
            if want not in have:
                return False
        elif have != want:
            return False
    return True


class MemoryVectorStore:
    """In-memory ``VectorStore`` with optional JSON persistence.

    Query ranks by cosine similarity when an ``embed_fn`` is available (and the
    chunks carry embeddings), otherwise by lexical token overlap so it always
    returns *something* offline. When ``persist_path`` is given, ``persist``
    writes the chunks to JSON and construction reloads them.
    """

    def __init__(
        self,
        *,
        embed_fn: Optional[EmbedFn] = None,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._persist_path = Path(persist_path) if persist_path else None
        self._chunks: Dict[str, Chunk] = {}
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if self._embed_fn is None:
            return None
        try:
            vectors = self._embed_fn(texts)
        except Exception:  # noqa: BLE001 - embedding must never hard-fail a query
            return None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            return None
        return vectors

    def upsert(self, chunks: List[Chunk]) -> UpsertReport:
        upserted = 0
        skipped = 0
        to_embed: List[Chunk] = []
        for chunk in chunks:
            if not isinstance(chunk, Chunk) or not chunk.id:
                skipped += 1
                continue
            self._chunks[chunk.id] = chunk
            upserted += 1
            if chunk.embedding is None:
                to_embed.append(chunk)
        if to_embed:
            vectors = self._embed([c.text for c in to_embed])
            if vectors is not None:
                for chunk, vector in zip(to_embed, vectors):
                    chunk.embedding = list(vector)
        return UpsertReport(
            upserted=upserted, skipped=skipped, total=len(self._chunks), backend="memory"
        )

    def query(
        self, text: str, *, top_k: int = 8, filters: Optional[dict] = None
    ) -> List[Retrieved]:
        candidates = [
            c for c in self._chunks.values() if _matches_filters(c.metadata(), filters)
        ]
        if not candidates:
            return []

        query_vec: Optional[List[float]] = None
        embedded = self._embed([text])
        if embedded is not None:
            query_vec = embedded[0]

        scored: List[Retrieved] = []
        for chunk in candidates:
            if (
                query_vec is not None
                and chunk.embedding is not None
                and len(query_vec) == len(chunk.embedding)
            ):
                score = _cosine(query_vec, chunk.embedding)
            else:
                # No usable vectors, or a dimension mismatch — e.g. a lexical
                # fallback vector cached while the embeddings endpoint was down,
                # now sitting beside real API vectors of a different width. Fall
                # back to lexical overlap so retrieval degrades gracefully
                # instead of every mismatched pair scoring a silent 0.0.
                score = _lexical_overlap(text, chunk.text)
            scored.append(Retrieved(chunk=chunk, score=round(float(score), 6)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(1, int(top_k))]

    def persist(self) -> None:
        if self._persist_path is None:
            return
        import json

        payload = {
            "version": 1,
            "backend": "memory",
            "chunks": [asdict(c) for c in self._chunks.values()],
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        import json

        assert self._persist_path is not None
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken cache must never break startup
            return
        for row in data.get("chunks", []) if isinstance(data, dict) else []:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            self._chunks[str(row["id"])] = Chunk(
                id=str(row.get("id", "")),
                text=str(row.get("text", "")),
                source_id=str(row.get("source_id", "")),
                source_title=str(row.get("source_title", "")),
                source_type=str(row.get("source_type", "")),
                source_uri=str(row.get("source_uri", "")),
                organism=str(row.get("organism", "")),
                section=str(row.get("section", "")),
                pathway_tags=list(row.get("pathway_tags", []) or []),
                embedding=row.get("embedding"),
            )

    def stats(self) -> dict:
        embedded = sum(1 for c in self._chunks.values() if c.embedding is not None)
        return {
            "backend": "memory",
            "count": len(self._chunks),
            "embedded": embedded,
            "persist_path": str(self._persist_path) if self._persist_path else "",
        }


def _import_chromadb() -> Any:
    """Import ``chromadb`` lazily, raising a clear error when it is absent."""
    try:
        import chromadb  # noqa: PLC0415 - intentionally lazy/guarded
    except Exception as exc:  # noqa: BLE001 - any import failure -> actionable message
        raise RuntimeError(
            "The 'chroma' vector backend requires the optional 'chromadb' package, "
            "which is not installed. Install it (see the '# RAG (optional)' block in "
            "requirements.txt) or set RAG_VECTOR_BACKEND=memory."
        ) from exc
    return chromadb


class ChromaVectorStore:
    """Persistent ``VectorStore`` backed by ``chromadb`` (default backend).

    ``chromadb`` is imported only inside ``__init__`` (via ``_import_chromadb``),
    so importing this module never requires the dependency.
    """

    def __init__(
        self,
        *,
        index_dir: Path,
        collection: str = "t2pw_rag",
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        chromadb = _import_chromadb()
        self._embed_fn = embed_fn
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._index_dir))
        self._collection = self._client.get_or_create_collection(collection)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self._embed_fn is None:
            raise RuntimeError(
                "ChromaVectorStore requires an embed_fn to upsert/query. Wire the "
                "embedder from t2pw.rag.embed via get_vector_store()."
            )
        return self._embed_fn(texts)

    def upsert(self, chunks: List[Chunk]) -> UpsertReport:
        rows = [c for c in chunks if isinstance(c, Chunk) and c.id]
        skipped = len(chunks) - len(rows)
        if rows:
            missing = [c for c in rows if c.embedding is None]
            if missing:
                vectors = self._embed([c.text for c in missing])
                for chunk, vector in zip(missing, vectors):
                    chunk.embedding = list(vector)
            self._collection.upsert(
                ids=[c.id for c in rows],
                embeddings=[c.embedding for c in rows],
                documents=[c.text for c in rows],
                metadatas=[_chroma_metadata(c) for c in rows],
            )
        return UpsertReport(
            upserted=len(rows),
            skipped=skipped,
            total=self._collection.count(),
            backend="chroma",
        )

    def query(
        self, text: str, *, top_k: int = 8, filters: Optional[dict] = None
    ) -> List[Retrieved]:
        vector = self._embed([text])[0]
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=max(1, int(top_k)),
            where=filters or None,
        )
        out: List[Retrieved] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 0.0
            out.append(
                Retrieved(
                    chunk=Chunk(
                        id=str(cid),
                        text=str(docs[i]) if i < len(docs) else "",
                        source_id=str(meta.get("source_id", "")),
                        source_title=str(meta.get("source_title", "")),
                        source_type=str(meta.get("source_type", "")),
                        source_uri=str(meta.get("source_uri", "")),
                        organism=str(meta.get("organism", "")),
                        section=str(meta.get("section", "")),
                        pathway_tags=_split_tags(meta.get("pathway_tags", "")),
                    ),
                    score=round(1.0 - float(dist), 6),
                )
            )
        return out

    def persist(self) -> None:
        # PersistentClient flushes to disk on write; nothing extra to do.
        return None

    def stats(self) -> dict:
        return {
            "backend": "chroma",
            "count": self._collection.count(),
            "index_dir": str(self._index_dir),
        }


def _chroma_metadata(chunk: Chunk) -> Dict[str, Any]:
    # Chroma metadata values must be scalars; join list tags into a string.
    meta = chunk.metadata()
    meta["pathway_tags"] = "|".join(str(t) for t in chunk.pathway_tags)
    return meta


def _split_tags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [t for t in value.split("|") if t]
    return []


def resolve_index_dir(config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve the configured index dir to an absolute path under PROJECT_ROOT."""
    cfg = config or rag_config()
    raw = Path(str(cfg.get("index_dir", "data/rag_index")))
    return raw if raw.is_absolute() else (PROJECT_ROOT / raw)


def get_vector_store(
    *,
    backend: Optional[str] = None,
    embed_fn: Optional[EmbedFn] = None,
    config: Optional[Dict[str, Any]] = None,
) -> VectorStore:
    """Construct the configured backend (``RAG_VECTOR_BACKEND``, default chroma).

    ``memory`` is always available; ``chroma`` lazily imports ``chromadb`` and
    raises a clear error if it is missing; ``faiss`` is reserved for a later WP.
    """
    cfg = config or rag_config()
    name = (backend or str(cfg.get("vector_backend", "chroma"))).strip().lower()
    if name == "memory":
        return MemoryVectorStore(embed_fn=embed_fn)
    if name == "chroma":
        return ChromaVectorStore(index_dir=resolve_index_dir(cfg), embed_fn=embed_fn)
    if name == "faiss":
        raise NotImplementedError(
            "The 'faiss' backend is not implemented in WP0; use 'chroma' or 'memory'."
        )
    raise ValueError(f"Unknown RAG_VECTOR_BACKEND: {name!r}")


__all__ = [
    "Chunk",
    "Retrieved",
    "UpsertReport",
    "VectorStore",
    "MemoryVectorStore",
    "ChromaVectorStore",
    "EmbedFn",
    "get_vector_store",
    "resolve_index_dir",
]
