# WP7 — Orchestration, UI & Triage

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core only through named seams. The orchestrator wires stages and RAG
> the same way: **wiring only, no logic.** Additive metadata only. Full text:
> [`../03_separation_invariant.md`](../03_separation_invariant.md).

**Lands last.** Ties R0–R5 together and exposes them, without moving any logic into
the orchestrator.

## Purpose

1. Implement R0 triage (the on/off decision).
2. Wire R0→R1→R2→R3→R4→R5→(Stage 2B) in the orchestrator behind `RAG_ENABLED`.
3. Add UI: the incomplete/novel flag, fetched-paper list, and a provenance viewer.

## Background you need

- **Orchestrator rule:** `streamlit_app.py` "orchestrates... calls stage functions
  and wires results together but does not own logic" (`docs/pipeline.md`). RAG wiring
  obeys the same rule — call `t2pw.rag` functions; put no RAG logic in the app.
- **Existing retrieval wiring to mirror:** `streamlit_app.py` around lines
  1327–1414 already builds `retrieval_context` from the motif index and injects it.
  Follow that shape for the RAG seams (S1/S2).
- **Stage 0 output** to read for auto-triage: `preprocessor.preprocess` fields
  (`scope_clarity_score`, `document_type`).
- **Payload entry (S3):** hand R5's `Payload` to the same post-pipeline entry the app
  already uses after extraction/inference.

## What to build

- `src/t2pw/rag/triage.py`:
  - `should_run_rag(context, user_flag, reports=None) -> TriageDecision` — explicit
    flag OR auto (low `scope_clarity_score`, dangling reactions, orphan metabolites,
    unmapped enzymes). Returns decision + reason.
- Orchestration in `streamlit_app.py` (**wiring only**):
  - If `RAG_ENABLED` and `should_run_rag(...)`: run `acquire → select → ingest →
    retrieve`, feed evidence via S1/S2, run `synthesize`, then hand the payload to
    the existing post-pipeline path (S3). Else: today's flow, untouched.
- UI:
  - A checkbox/flag: "This pathway is unknown / incomplete (enable multi-paper RAG)."
  - A panel listing fetched + selected papers (from WP1/WP2 reports).
  - A provenance viewer: for each reaction/entity, show its source papers (from WP5).

## Depends on / blocks

- Depends on: WP0–WP6.
- Blocks: nothing (final WP).

## Acceptance criteria

- `RAG_ENABLED=false` → the app path and outputs are identical to today.
- With RAG on and a flagged incomplete seed, the app fetches papers, synthesizes a
  connected pathway, and the provenance viewer shows sources per element.
- `streamlit_app.py` gains **no** normalization/mapping/synthesis logic — only calls
  into `t2pw.rag` and existing stage functions.
- `docs/change_log.md` entry.

## Out of scope

Any change to the core stage internals. If you feel the urge to edit a stage body,
re-read the separation invariant.
