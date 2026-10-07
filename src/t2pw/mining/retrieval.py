"""Passage indexing and retrieval for a mining run.

Reuses the existing RAG infrastructure -- ``t2pw.rag.ingest.chunk_paper`` for
section-aware chunking, ``t2pw.rag.embed.Embedder`` for cached embeddings and
``t2pw.rag.store`` for the vector index -- and adds the two mining-specific
things:

1. every retrieved chunk becomes an :class:`~t2pw.mining.models.EvidenceSpan`
   carrying the query and frontier item that surfaced it, so provenance runs
   backwards from a claim all the way to the question that asked it;
2. the index is **per-run** and defaults to the in-memory backend, so mining
   cannot write into the production RAG index.

Embeddings honesty
------------------
``Embedder`` degrades to deterministic lexical vectors when no embedding
endpoint answers. That degradation is legitimate for offline work but must never
be invisible: :attr:`PassageIndex.embedding_mode` records which mode a run was
actually in, the preflight reports it, and a run configured to require real
embeddings refuses to start without them. "RAG added nothing" is not an
acceptable description of an embeddings outage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from t2pw.mining.acquisition import LoadedPaper
from t2pw.mining.config import MiningConfig
from t2pw.mining.ids import evidence_id as make_evidence_id
from t2pw.mining.models import EvidenceSpan

EMBEDDING_API = "api"
EMBEDDING_LEXICAL = "lexical_fallback"
EMBEDDING_UNKNOWN = "unknown"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EmbeddingsUnavailable(RuntimeError):
    """The configured embeddings service did not answer and was required."""


@dataclass
class IndexedPaper:
    """Bookkeeping for one paper's passages."""

    paper_id: str
    chunk_ids: List[str] = field(default_factory=list)
    chars: int = 0
    sections: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "chunks": len(self.chunk_ids),
            "chars": self.chars,
            "sections": dict(self.sections),
        }


class PassageIndex:
    """A per-run passage store with provenance-carrying retrieval."""

    def __init__(
        self,
        config: MiningConfig,
        *,
        embedder: Optional[Any] = None,
        store: Optional[Any] = None,
        require_embeddings: bool = False,
    ) -> None:
        self.config = config
        self.require_embeddings = bool(require_embeddings)
        self.papers: Dict[str, IndexedPaper] = {}
        self.texts: Dict[str, str] = {}
        self._chunk_meta: Dict[str, Dict[str, Any]] = {}
        self._embedder = embedder
        self._store = store
        self.embedding_mode = EMBEDDING_UNKNOWN
        self.warnings: List[str] = []

    # -- lazy construction --------------------------------------------------
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from t2pw.rag.embed import Embedder

            self._embedder = Embedder()
        return self._embedder

    @property
    def store(self) -> Any:
        if self._store is None:
            from t2pw.rag.store import get_vector_store

            self._store = get_vector_store(
                backend=self.config.vector_backend, embed_fn=self.embedder.embed
            )
        return self._store

    def _record_embedding_mode(self) -> None:
        stats = getattr(self.embedder, "stats", None)
        if not isinstance(stats, dict):
            return
        api = int(stats.get("api", 0) or 0)
        lexical = int(stats.get("lexical", 0) or 0)
        if api and not lexical:
            self.embedding_mode = EMBEDDING_API
        elif lexical:
            self.embedding_mode = EMBEDDING_LEXICAL
            message = (
                f"embeddings degraded to the deterministic lexical fallback for {lexical} "
                "passage(s): the configured embedding endpoint did not answer. Retrieval "
                "quality is materially lower; this is an infrastructure state, not a "
                "finding about the literature."
            )
            if message not in self.warnings:
                self.warnings.append(message)
            if self.require_embeddings:
                raise EmbeddingsUnavailable(message)

    # -- indexing -----------------------------------------------------------
    def add_paper(self, paper: LoadedPaper) -> IndexedPaper:
        """Chunk, embed and index one paper. Idempotent per paper id."""
        record = paper.record
        if record.paper_id in self.papers:
            return self.papers[record.paper_id]

        chunks = self._chunk(paper)
        cap = max(1, int(self.config.budgets.max_passages_per_paper or len(chunks) or 1))
        chunks = chunks[:cap]

        indexed = IndexedPaper(paper_id=record.paper_id, chars=len(paper.text))
        ordinals: Dict[str, int] = {}
        for chunk in chunks:
            section = str(getattr(chunk, "section", "") or "body")
            ordinal = ordinals.get(section, 0)
            ordinals[section] = ordinal + 1
            indexed.sections[section] = ordinals[section]
            indexed.chunk_ids.append(chunk.id)
            start = paper.text.find(chunk.text) if paper.text else -1
            self._chunk_meta[chunk.id] = {
                "paper_id": record.paper_id,
                "section": section,
                "ordinal": ordinal,
                "char_start": start if start >= 0 else None,
                "char_end": (start + len(chunk.text)) if start >= 0 else None,
            }

        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = list(vector)
            self.store.upsert(chunks)
            self._record_embedding_mode()

        self.papers[record.paper_id] = indexed
        self.texts[record.paper_id] = paper.text
        record.full_text_chars = len(paper.text)
        return indexed

    def _chunk(self, paper: LoadedPaper) -> List[Any]:
        """Section-aware chunking through the existing RAG chunker."""
        from t2pw.rag.acquire import CandidatePaper
        from t2pw.rag.ingest import chunk_paper

        record = paper.record
        candidate = CandidatePaper(
            id=record.paper_id,
            source=record.acquisition_source or "mining",
            title=record.title,
            abstract=record.abstract,
            full_text=paper.text,
            source_uri=record.url,
            year=record.year,
            observed_organisms=list(record.observed_organisms),
            observed_pathways=list(record.observed_pathways),
        )
        chunks = chunk_paper(candidate)
        if chunks:
            return chunks
        # A very short document (an abstract-only record, a tiny fixture) can
        # fall through the section splitter. Index it whole rather than losing it.
        from t2pw.rag.store import Chunk

        body = paper.text or record.abstract
        if not body.strip():
            return []
        return [
            Chunk(
                id=f"{record.paper_id}::whole",
                text=body,
                source_id=record.paper_id,
                source_title=record.title,
                source_type="paper",
                source_uri=record.url,
                section="body",
            )
        ]

    # -- retrieval ----------------------------------------------------------
    def retrieve(
        self,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        round_index: int = 0,
        query_id: str = "",
        frontier_item_id: str = "",
        paper_ids: Optional[Sequence[str]] = None,
    ) -> List[EvidenceSpan]:
        """Retrieve passages and return them as evidence spans.

        ``paper_ids`` restricts retrieval to a subset -- used when extracting
        from one specific paper, so passages from a different paper can never be
        cited as that paper's evidence. Each requested paper's OWN chunks are
        ranked (the store's ``source_id`` filter): a store-wide top ``3k`` filtered
        afterwards gave a paper whose chunks rank below other papers' nothing at
        all (A7: pilot-002's PMC5104190, "no passages retrieved").
        """
        if not self.papers:
            return []
        k = int(top_k or self.config.retrieve_top_k or 8)
        query = str(query_text or "")
        try:
            if paper_ids:
                hits = sorted(
                    (hit for pid in dict.fromkeys(paper_ids)
                     for hit in self.store.query(query, top_k=k, filters={"source_id": pid})),
                    key=lambda hit: float(getattr(hit, "score", 0.0) or 0.0), reverse=True)
            else:
                hits = self.store.query(query, top_k=max(k * 3, k))
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"retrieval failed: {type(exc).__name__}: {exc}")
            return []
        self._record_embedding_mode()

        allowed = set(paper_ids) if paper_ids else None
        spans: List[EvidenceSpan] = []
        for hit in hits:
            chunk = getattr(hit, "chunk", hit)
            source = str(getattr(chunk, "source_id", "") or "")
            if allowed is not None and source not in allowed:
                continue
            meta = self._chunk_meta.get(str(getattr(chunk, "id", "")), {})
            text = str(getattr(chunk, "text", "") or "").strip()
            if not text:
                continue
            section = str(meta.get("section") or getattr(chunk, "section", "") or "body")
            ordinal = meta.get("ordinal")
            spans.append(
                EvidenceSpan(
                    evidence_id=make_evidence_id(source, str(getattr(chunk, "id", "")), text),
                    paper_id=source,
                    text=text,
                    section=section,
                    paragraph=(int(ordinal) + 1) if isinstance(ordinal, int) else None,
                    locator=f"{section}#{ordinal}" if ordinal is not None else section,
                    chunk_id=str(getattr(chunk, "id", "")),
                    char_start=meta.get("char_start"),
                    char_end=meta.get("char_end"),
                    context_ref=str(getattr(chunk, "source_title", "") or ""),
                    extraction_round=int(round_index),
                    retrieval_query_id=query_id,
                    frontier_item_id=frontier_item_id,
                    retrieval_score=float(getattr(hit, "score", 0.0) or 0.0),
                    retrieved_at=_utcnow(),
                )
            )
            if len(spans) >= k:
                break
        return spans

    def spans_for_paper(
        self,
        paper_id: str,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        round_index: int = 0,
        query_id: str = "",
        frontier_item_id: str = "",
    ) -> List[EvidenceSpan]:
        return self.retrieve(
            query_text,
            top_k=top_k,
            round_index=round_index,
            query_id=query_id,
            frontier_item_id=frontier_item_id,
            paper_ids=[paper_id],
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "papers_indexed": len(self.papers),
            "chunks": sum(len(p.chunk_ids) for p in self.papers.values()),
            "embedding_mode": self.embedding_mode,
            "vector_backend": self.config.vector_backend,
            "warnings": list(self.warnings),
        }


__all__ = [
    "EMBEDDING_API",
    "EMBEDDING_LEXICAL",
    "EMBEDDING_UNKNOWN",
    "EmbeddingsUnavailable",
    "IndexedPaper",
    "PassageIndex",
]
