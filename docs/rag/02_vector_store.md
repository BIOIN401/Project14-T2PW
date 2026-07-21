# Vector Store — Design Spec

Owned by WP0 (interface + backend) and populated by WP3 (ingest). Everything here
follows the pipeline's **offline-first** precedent (`data/pathwhiz_id_db.json`, the
id/enrichment caches).

---

## Where it lives

- Persisted under **`data/rag_index/`** (git-ignored like the other caches).
- A new package **`t2pw/rag/store.py`** owns the interface and the default backend.
- **Separate from the PathBank MySQL DB.** That DB is the read-only canonical name
  source; the RAG store is a local, rebuildable evidence index. Do not put RAG
  vectors in MySQL.

---

## Interface (WP0 defines this)

```python
# t2pw/rag/store.py
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> UpsertReport: ...
    def query(self, text: str, *, top_k: int = 8,
              filters: dict | None = None) -> list[Retrieved]: ...
    def persist(self) -> None: ...
    def stats(self) -> dict: ...
```

- `filters` supports metadata filtering (e.g. `{"organism": "...", "source_type": "paper"}`).
- Backend is chosen by env `RAG_VECTOR_BACKEND` (default `chroma`). Keep it
  pluggable so `faiss` or an in-memory test backend can drop in.

### Chunk record schema

```python
@dataclass
class Chunk:
    id: str                 # stable hash of (source_id, section, offset)
    text: str
    source_id: str          # PMID / PMCID / DOI / pathbank id / reference filename
    source_title: str
    source_type: str        # "paper" | "pathbank" | "kegg" | "pwml_example"
    source_uri: str         # resolvable provenance pointer
    organism: str
    section: str            # "abstract" | "results" | "methods" | "figure" | ...
    pathway_tags: list[str]
    embedding: list[float] | None   # filled by the embedder; None until embedded
```

`source_id` + `source_uri` are what WP5/WP6 attach as **provenance** to synthesized
elements. Every retrieval carries them through.

---

## Embeddings

- Reuse the existing OpenAI-compatible client (`t2pw.llm.client`) against an
  embeddings endpoint — OpenRouter and LM Studio both expose one. Config:
  - `RAG_EMBEDDING_PROVIDER` (default: same provider as `LLM_PROVIDER`)
  - `RAG_EMBEDDING_MODEL`
  - `RAG_EMBEDDING_DIM` (validated on upsert)
- **Cache embeddings** in `data/rag_index/embeddings_cache.json`, mirroring
  `data/id_mapping_cache.json`. Never re-embed an unchanged chunk.
- **Offline fallback:** if no embedding endpoint is reachable, WP3 falls back to the
  existing lexical motif scoring (`t2pw.sbml.examples`) so retrieval still returns
  *something*. RAG never hard-fails on a missing embedder.

---

## Hybrid retrieval

WP4 queries by combining:

- **Semantic** score from `VectorStore.query`, and
- **Lexical** score from the existing `_score_entry` token-overlap in
  `t2pw.sbml.examples`.

Blend (starting weights, tune later): `0.7 * semantic + 0.3 * lexical`. The lexical
half guarantees exact gene/compound symbol matches (e.g. `NdmA`, `OPC-8:0-CoA`) are
never lost to embedding fuzz — the same class of tokens the pipeline already treats
carefully.

---

## Config summary (WP0 adds these; all optional, all default-safe)

```text
RAG_ENABLED=false                 # master switch; off = today's behavior
RAG_VECTOR_BACKEND=chroma         # chroma | faiss | memory
RAG_INDEX_DIR=data/rag_index
RAG_EMBEDDING_PROVIDER=           # defaults to LLM_PROVIDER
RAG_EMBEDDING_MODEL=
RAG_EMBEDDING_DIM=
RAG_ACQUIRE_MAX_PAPERS=20         # WP1 cap
RAG_SELECT_MAX_PAPERS=8           # WP2 cap
RAG_RETRIEVE_TOP_K=8              # WP4
```

All read through `t2pw.config` (extend it, do not scatter `os.getenv`). Nothing
hardcoded.

---

## Dependencies to add (WP0)

- `chromadb` (default backend) **or** `faiss-cpu` + a metadata sidecar. `numpy` is
  already present.
- Import them **lazily/guarded** inside `t2pw/rag/store.py` so the base pipeline
  never requires them. With `RAG_ENABLED=false` and RAG never imported, a missing
  `chromadb` must not break any existing test.
- Add to `requirements.txt` under a clearly commented `# RAG (optional)` block.
