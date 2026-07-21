# WP0 — Foundation

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core pipeline only through named seams (S1 context params, S2
> `retrieval_context`, S3 `Payload` boundary, S4 read-only reports, S5
> orchestration). No RAG logic inside any stage module. Additive metadata only.
> Full text: [`../03_separation_invariant.md`](../03_separation_invariant.md).

**This WP lands first. Everything else blocks on it.** Keep it small, correct, and
fully green before any other agent starts.

---

## Purpose

Create the shared scaffolding every other WP builds against: the package, the
`VectorStore` interface + a working local backend, the embedding client, config +
feature flag, and the additive provenance field definitions. No pipeline behavior
changes in this WP.

## Background you need

- Offline-index precedent: `data/pathwhiz_id_db.json`, `data/id_mapping_cache.json`,
  `data/enrichment_cache.json`. Follow this pattern.
- Config precedent: `t2pw.config` (`ensure_dotenv_loaded`, `resolution_db_config`).
  Extend it; do not scatter `os.getenv`.
- Vector store spec: [`../02_vector_store.md`](../02_vector_store.md).
- Existing lexical retrieval you will later wrap: `t2pw.sbml.examples`.

## What to build

1. **Package skeleton** — `src/t2pw/rag/__init__.py` and empty module stubs:
   `store.py`, `acquire.py`, `select.py`, `ingest.py`, `retrieve.py`,
   `synthesize.py`, `provenance.py`, `triage.py`. (Later WPs fill these.)
2. **Config** — extend `t2pw.config` with a `rag_config()` reader for every
   `RAG_*` var in [`../02_vector_store.md`](../02_vector_store.md). All optional,
   all default-safe. `RAG_ENABLED` defaults `false`.
3. **`VectorStore`** — the `Protocol` + `Chunk`/`Retrieved` dataclasses + a default
   backend (`chroma`) and an in-memory backend (`memory`, for tests). Guarded lazy
   import of `chromadb` so a missing dep never breaks the base pipeline.
4. **Embedding client** — `t2pw/rag/embed.py`: OpenAI-compatible embeddings via
   `t2pw.llm.client`, with `data/rag_index/embeddings_cache.json` and the lexical
   offline fallback.
5. **Provenance fields** — define the optional additive keys (`rag_provenance`,
   `evidence`, `source_papers`, `rag_confidence`) as `TypedDict`s in
   `t2pw/rag/provenance.py`. **Do not** edit `t2pw.schema`; reference it. These are
   optional keys existing stages ignore. The source pointer is `rag_provenance`, not
   `provenance` — `provenance` is a core-owned string field, so a RAG key of that name
   would collide with it (see [`../03_separation_invariant.md`](../03_separation_invariant.md)).
6. **requirements.txt** — add a commented `# RAG (optional)` block.

## Depends on / blocks

- Depends on: nothing.
- Blocks: **all** other WPs.

## Acceptance criteria

- `RAG_ENABLED=false` (default) → `pytest -q`, `ruff check src tests scripts`, and a
  reference PWML export are identical to pre-change `main`.
- `grep -rn "t2pw.rag" src/t2pw/{pipeline,mapping,curation,pwml}` returns nothing.
- Unit tests: config defaults; `memory` backend `upsert`/`query`/`persist`;
  embedding cache hit/miss; guarded import works with `chromadb` absent.
- New `docs/change_log.md` entry (what/why/how per the repo rule).

## Out of scope

Fetching papers, chunking, retrieval logic, synthesis, UI — those are WP1–WP7.
