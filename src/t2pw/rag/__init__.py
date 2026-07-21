"""``t2pw.rag`` — the RAG subsystem package (WP0 foundation).

This package is an **optional, opt-in** evidence layer. Per the separation
invariant (docs/rag/03_separation_invariant.md), all RAG code lives here and
touches the core pipeline only through named seams; no core module imports
``t2pw.rag``, and the dependency arrow points RAG -> core only.

Importing this package is cheap and dependency-light: it does **not** import
``chromadb`` (the chroma backend imports it lazily on construction) and does
**not** construct the LLM client (the embedder imports it lazily on first API
call). With ``RAG_ENABLED`` unset, nothing here changes pipeline behavior.
"""

from __future__ import annotations

from t2pw.config import rag_config, rag_enabled
from t2pw.rag.embed import Embedder, embed_texts
from t2pw.rag.provenance import (
    RagAdditiveMetadata,
    RagEvidence,
    RagProvenance,
    RagSourcePaper,
)
from t2pw.rag.store import (
    Chunk,
    MemoryVectorStore,
    Retrieved,
    UpsertReport,
    VectorStore,
    get_vector_store,
)

__all__ = [
    "rag_config",
    "rag_enabled",
    "Chunk",
    "Retrieved",
    "UpsertReport",
    "VectorStore",
    "MemoryVectorStore",
    "get_vector_store",
    "Embedder",
    "embed_texts",
    "RagProvenance",
    "RagEvidence",
    "RagSourcePaper",
    "RagAdditiveMetadata",
]
