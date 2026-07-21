# WP2 — Selection

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core only through named seams. No RAG logic inside a stage module.
> Additive metadata only. Full text: [`../03_separation_invariant.md`](../03_separation_invariant.md).

## Purpose

Rank, filter, dedupe, and cap the candidate papers from WP1 so only the papers
genuinely relevant to the target pathway + organism reach the (expensive) embedding
step. Prevents unrelated review examples from bleeding into the corpus.

## Background you need

- **Reuse the preprocessor.** `t2pw.pipeline.preprocessor.preprocess(text)` already
  returns `pathway_relevance_score`, `scope_clarity_score`, `document_type`,
  `likely_organism`, `key_compounds/proteins`. Run it per candidate (on abstract, or
  a truncated full text) — do not build a new classifier.
- **The locality discipline matters.** See `preprocess_system.txt` STEP 3: entities
  belong to an example only when locally anchored. A `multi_example_review` candidate
  should be down-weighted or example-scoped, not ingested wholesale.

## What to build

- `src/t2pw/rag/select.py`:
  - `score_candidate(candidate, seed_context) -> SelectionScore` combining:
    organism match, compound/protein overlap with seed + gaps,
    `pathway_relevance_score`, and a penalty for `multi_example_review` with no
    matching example.
  - `select(candidates, seed_context, *, max_papers=RAG_SELECT_MAX_PAPERS) -> Selection`
    → ranked, deduped, capped subset + a `selection_report` (kept/dropped + reasons).

## Depends on / blocks

- Depends on: WP0, WP1.
- Blocks: WP3.

## Acceptance criteria

- Deterministic given fixed inputs (mock the preprocessor LLM call).
- A `multi_example_review` candidate whose examples don't match the seed is dropped
  or ranked below on-topic primary research.
- `selection_report` explains every drop.
- No import of `t2pw.rag` by `preprocessor.py`; dependency points RAG → core only.
- `docs/change_log.md` entry.

## Out of scope

Chunking/embedding (WP3), retrieval (WP4). Selection only.
