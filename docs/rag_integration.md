# RAG Integration — Master Plan

**Status:** planning / not yet implemented
**Owner doc set:** this file + [`docs/rag/`](rag/)
**Goal:** let T2PW generate **novel pathways** by pulling evidence from *multiple*
papers, not just the one a user uploads.

---

## 1. Why this exists

Today T2PW is a **single-paper → single-pathway** pipeline. One document goes in,
one PWML file comes out (see [`docs/pipeline.md`](pipeline.md)). That is correct
for a paper that fully describes its pathway.

It is **not** enough when a user uploads one paper and says *"this pathway is
unknown / incomplete."* In that case the pathway cannot be assembled from the
single document — the missing steps, precursors, enzymes, and compartments live
in **other** papers and in structured databases.

RAG (Retrieval-Augmented Generation) closes that gap. When triggered, the system:

1. detects that the seed pathway is incomplete or flagged as novel,
2. autonomously **fetches related papers** through literature APIs,
3. **selects** the relevant ones,
4. **embeds** them into a vector store (a real semantic index, not the current
   keyword index),
5. **retrieves** evidence targeted at each specific gap, and
6. **synthesizes** one connected, provenance-tagged pathway from the combined
   evidence — which then flows through the *existing* Stage 2B→8 unchanged.

The result is a **novel pathway = a novel *connection* of evidence-backed steps.**
It is never invented chemistry. Every reaction and entity must trace back to a
retrieved source passage. This is the same discipline the pipeline already applies
when it forbids the audit from inventing stoichiometry.

---

## 2. The one rule that must not be broken

> **RAG plugs into the pipeline. It does not merge with it.**

The eight existing stages (0–8) are the **core pipeline**. RAG (stages R0–R5) is a
**separate subsystem** in a new `t2pw.rag` package. RAG touches the core **only**
through a small set of pre-existing seams, and adds **only additive metadata**. With
RAG disabled, the pipeline must behave exactly as it does today.

This rule is important enough that it has its own document —
[`docs/rag/03_separation_invariant.md`](rag/03_separation_invariant.md) — and a
condensed copy of it is pinned at the top of **every** sub-agent brief. Read it
before writing any code.

---

## 3. Where RAG sits in the ordered pipeline

```
Seed paper + user "this pathway is unknown/incomplete" flag
    │
    ▼
[R0 Triage]        extends Stage 0. Novel/incomplete? (explicit flag OR auto-detect)
    │              If not novel → run the existing single-paper pipeline untouched.
    ▼
[R1 Acquire]       query EuropePMC / NCBI (already wired) for related papers; download
    │              full text.
    ▼
[R2 Select]        rank / dedupe / cap candidates by relevance to target + organism.
    │
    ▼
[R3 Ingest/Embed]  chunk selected papers + DB reactions → embed → vector store.
    │              Upgrades the lexical motif index to hybrid semantic+lexical.
    ▼
[R4 Gap-Retrieve]  per pathway gap, form a query, retrieve top-k evidence; feed it into
    │              the core via existing context/retrieval_context seams.
    ▼
[R5 Synthesize]    merge multi-paper evidence into ONE connected Payload with provenance.
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  EXISTING CORE PIPELINE — UNCHANGED                                    │
│  Stage 2B Map → 3 Normalize → 4 Audit/4a Gap → 5 Curate → 6 Remap →   │
│  7 Enrich → 8 Export (PWML)                                            │
└──────────────────────────────────────────────────────────────────────┘
```

Full detail: [`docs/rag/01_pipeline_placement.md`](rag/01_pipeline_placement.md).

---

## 4. Implementation is decomposed into eight work packages

Each work package (WP) is a **self-contained brief** that a single Claude Code
sub-agent can own. Briefs live in [`docs/rag/agents/`](rag/agents/). They are
ordered by dependency so agents can run in parallel where the graph allows.

| WP | Brief | Builds |
|----|-------|--------|
| WP0 | [Foundation](rag/agents/wp0_foundation.md) | `t2pw.rag` package skeleton, `VectorStore` interface + local backend, embedding client, config/feature flag, additive provenance fields |
| WP1 | [Acquisition](rag/agents/wp1_acquisition.md) | multi-source paper fetch + full-text download (extends existing EuropePMC/NCBI plumbing) |
| WP2 | [Selection](rag/agents/wp2_selection.md) | relevance scoring / ranking / capping of candidate papers |
| WP3 | [Ingest & Index](rag/agents/wp3_ingest_index.md) | section-aware chunker, embed + upsert, hybrid upgrade of the motif index |
| WP4 | [Gap Retrieval](rag/agents/wp4_gap_retrieval.md) | gap detector + per-gap query formulation + retrieval, fed through existing seams |
| WP5 | [Synthesis](rag/agents/wp5_synthesis.md) | multi-paper merge into one connected Payload with provenance |
| WP6 | [Provenance & Gates](rag/agents/wp6_provenance_gates.md) | "no evidence → no element" enforcement + provenance validation + tests |
| WP7 | [Orchestration, UI & Triage](rag/agents/wp7_orchestration_ui.md) | R0 trigger, wire R0–R5 in the orchestrator (no logic), provenance viewer UI |

**Dependency graph**

```
WP0 ──┬─> WP1 ─> WP2 ─┐
      │                ├─> WP3 ─> WP4 ─> WP5 ─> WP6 ─> WP7
      └────────────────┘
```

WP0 must land and be green before anything else starts. WP1 and (the store half
of) WP3 can proceed in parallel after WP0. WP7 lands last.

---

## 5. Shared background (read before any WP)

- [`docs/rag/00_overview.md`](rag/00_overview.md) — goal, the definition of "novel
  pathway," guardrails, glossary.
- [`docs/rag/01_pipeline_placement.md`](rag/01_pipeline_placement.md) — the R-stages
  and the exact seams they attach to.
- [`docs/rag/02_vector_store.md`](rag/02_vector_store.md) — store choice, record
  schema, embeddings, config, offline behavior.
- [`docs/rag/03_separation_invariant.md`](rag/03_separation_invariant.md) — the
  non-negotiable "do not bleed into the stages" rule.
- [`docs/rag/MASTER_AGENT_PROMPT.md`](rag/MASTER_AGENT_PROMPT.md) — the prompt for
  the coordinating master agent that deploys the sub-agents.

---

## 6. Definition of done for the whole initiative

1. With `RAG_ENABLED=false` (default), `pytest -q`, `ruff check`, and PWML export
   produce **identical** results to `main` before this work. RAG is invisible when off.
2. With RAG on, a seed paper flagged "incomplete" produces a connected pathway whose
   every reaction/entity carries provenance to a retrieved source.
3. No existing stage module (`process_normalizer.py`, `map_ids.py`,
   `audit_json_llm.py`, `pwml/*`, …) contains RAG logic. All RAG code lives under
   `t2pw.rag`.
4. Every code change has a [`docs/change_log.md`](change_log.md) entry per the repo
   rule.
