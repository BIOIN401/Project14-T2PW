# WP3 — Ingest & Index

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core only through named seams. No RAG logic inside a stage module.
> Additive metadata only. Full text: [`../03_separation_invariant.md`](../03_separation_invariant.md).

## Purpose

Turn selected papers + structured DB records + the existing example corpus into a
populated, persisted vector store, and provide the hybrid (semantic + lexical)
scorer WP4 will call.

## Background you need

- Store interface + `Chunk` schema + embeddings + config: [`../02_vector_store.md`](../02_vector_store.md).
- **Existing lexical retrieval to wrap, not replace:** `t2pw.sbml.examples`
  (`build_motif_index`, `retrieve_motif_examples`, `_score_entry`). WP4's hybrid
  score blends this with semantic score.
- **Example corpus already on disk:** `reference/*.pwml`, `reference/*.sbml`.
- Structured reactions: PathBank MySQL (via existing mapping helpers) and KEGG
  (`rest.kegg.jp`, already used in `map_ids.py` / `stoich/agent.py`).

## What to build

- `src/t2pw/rag/ingest.py`:
  - `chunk_paper(candidate) -> list[Chunk]` — **section-aware** splitting (abstract,
    intro, results, methods, figure captions). Keep chunks ~500–1000 tokens with
    overlap; tag `section` and carry `source_uri` for provenance.
  - `chunk_db_reactions(records) -> list[Chunk]` — one chunk per reaction/record.
  - `ingest(selection) -> IngestReport` — chunk → embed (WP0 embedder) → `upsert`
    to the `VectorStore` → `persist()`.
  - `build_hybrid_scorer(store)` → callable used by WP4 that blends
    `store.query(...)` with `t2pw.sbml.examples._score_entry`
    (`0.7*semantic + 0.3*lexical`, tunable).
- Also index the `reference/` example corpus once, tagged `source_type="pwml_example"`,
  so example-pathway retrieval becomes semantic too.

## Depends on / blocks

- Depends on: WP0, WP2. (Corpus-half can start against WP0 in parallel with WP1–2.)
- Blocks: WP4.

## Acceptance criteria

- `memory` backend end-to-end test: chunk a canned paper → embed (stubbed embedder)
  → query returns the right chunk with provenance intact.
- Re-ingesting an unchanged paper is a no-op (embedding cache + stable chunk ids).
- Exact-symbol query (e.g. `NdmA`) still retrieves via the lexical half when
  embeddings are unavailable.
- `docs/change_log.md` entry.

## Out of scope

Gap detection and query formulation (WP4), synthesis (WP5).
