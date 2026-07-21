# WP4 — Gap Retrieval

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core only through named seams. No RAG logic inside a stage module.
> Additive metadata only. Full text: [`../03_separation_invariant.md`](../03_separation_invariant.md).

## Purpose

Find the specific gaps in the current pathway, turn each into a query, retrieve
evidence from the vector store, and hand that evidence to the core pipeline **only
through the existing seams S1 and S2**.

## Background you need

- **Gap signals already computed by the core (read-only, seam S4):**
  - `qa_graph.py` — connectivity graph, degree checks (dangling nodes, orphans).
  - Stage 3 strict gate report (`run_strict_post_normalization_gates`) — unresolved
    references, missing participants.
  - Mapping reports — `status="unmapped"` entities (missing enzyme/compound IDs).
- **Injection seams (do not edit stage bodies):**
  - **S1** — `run_extraction_pipeline(..., pathway_context=, user_task_context=)`.
    Fold retrieved evidence into these existing params for the seed extraction.
  - **S2** — `run_audit(..., retrieval_context="")` (already exists,
    `audit_json_llm.py:1489`). Pass gap-targeted evidence text here during the loop.
- Hybrid scorer from WP3.

## What to build

- `src/t2pw/rag/retrieve.py`:
  - `detect_gaps(payload, reports) -> list[Gap]` — read `qa_graph` + gate + mapping
    reports; classify each gap: `dangling_reaction`, `orphan_metabolite`,
    `unmapped_enzyme`, `missing_precursor`, `missing_compartment`.
  - `query_for_gap(gap, seed_context) -> str` — natural-language + symbol query.
  - `retrieve_evidence(gap, store, *, top_k=RAG_RETRIEVE_TOP_K) -> EvidenceBundle`
    — hybrid retrieval; each hit keeps `source_id`/`source_uri` provenance.
  - `format_retrieval_context(bundles) -> str` — render to the plain-text shape the
    audit/extraction prompts already expect (mirror
    `t2pw.sbml.examples.build_retrieval_context`).

## Depends on / blocks

- Depends on: WP0, WP3.
- Blocks: WP5.

## Acceptance criteria

- Given a payload with one dangling reaction + a store containing the missing step,
  `detect_gaps` finds it and `retrieve_evidence` returns the right chunk with
  provenance.
- The formatted context is a **string** consumed via S1/S2 — no new parameter is
  added to `pipeline.py` or the body of `run_audit`.
- `grep` confirms `retrieve.py` only *reads* report artifacts; it never mutates them.
- `docs/change_log.md` entry.

## Out of scope

Merging evidence into a final payload (WP5), gate enforcement (WP6).
