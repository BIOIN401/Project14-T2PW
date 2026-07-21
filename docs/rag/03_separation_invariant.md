# The Separation Invariant — Do Not Bleed Into the Stages

**This is the most important rule in the RAG initiative.** Every sub-agent brief
pins a condensed copy of it. If a change would violate it, the change is wrong —
redesign it, do not weaken the rule.

---

## Statement

> The existing pipeline stages stay separate. RAG plugs into them; it does not
> merge with them. All RAG code lives in the new `t2pw.rag` package and touches the
> core pipeline only through named seams, adding only additive metadata.

This is the RAG-scoped extension of a rule the pipeline already lives by
([`docs/pipeline.md`](../pipeline.md), "Design principle"):

> *"Stages are independent. Each stage function takes an input object and returns
> an output object. Logic that spans two stages belongs in the orchestrator, not
> inside either stage function."*

RAG adds: **RAG is independent of the stages.** Its logic belongs in `t2pw.rag`,
never inside a stage module, and it reaches the stages only through the seams below.

---

## Why this matters

The core pipeline is heavily gated and documented precisely to prevent bloat and
cross-stage coupling (see the change-log requirement in `docs/pipeline.md`). If RAG
code leaks into `process_normalizer.py`, `map_ids.py`, `audit_json_llm.py`, or
`pwml/`, three things break:

1. **Regressions become untraceable** — a mapping bug could now be a RAG bug, and
   the stage's contract no longer means what it says.
2. **RAG-off is no longer safe** — the guarantee that disabling RAG restores today's
   behavior depends on RAG code being physically absent from the stage's hot path.
3. **The gates stop being authoritative** — the whole point of Stage 3/8 gates is
   that they validate *any* payload. If RAG special-cases them, they validate less.

---

## The only allowed seams

RAG may interact with the core pipeline **only** at these points:

| # | Seam | Direction | Contract |
|---|------|-----------|----------|
| S1 | **Context injection (Stage 1)** — `run_extraction_pipeline(..., pathway_context=, pathway_scope=, user_task_context=)` | RAG → core | RAG supplies retrieved evidence through these **existing** parameters (e.g. an augmented `user_task_context`). It does **not** add a new parameter to, or edit the body of, `pipeline.py`. |
| S2 | **Audit retrieval (Stage 4)** — `run_audit(..., retrieval_context="")` | RAG → core | Already exists. RAG passes gap-targeted evidence text here. No edit to the audit body. |
| S3 | **Payload entry (Stage 2B)** — the `Payload` handed to `map_payload` / the post-pipeline orchestrator | RAG → core | RAG synthesis emits a **standard `Payload`** (the `TypedDict` shapes in `t2pw.schema`). From here the core runs exactly as today. RAG never edits `map_ids.py`. |
| S4 | **Report reads** — gate reports, `qa_graph` output, mapping reports | core → RAG | **Read-only.** RAG reads these artifacts to detect gaps. It never mutates them. |
| S5 | **Orchestration** — `streamlit_app.py` | wiring | The orchestrator calls RAG functions the same way it calls stage functions: wiring only, no logic (existing rule). |

Anything not on this list is off-limits.

---

## Additive-metadata rule

RAG may attach new keys — `rag_provenance`, `evidence`, `source_papers`,
`rag_confidence` — to entities and processes. It may **not**:

- rename, repurpose, or change the type of any field an existing stage owns;
- remove fields;
- reorder or restructure processes/entities in a way that changes their meaning.

New keys must be optional and ignored by every stage that does not know about them —
exactly how the pipeline already treats "additive metadata" at each boundary
(`docs/pipeline.md`, runtime shape validation).

> **A new additive key must not collide with a name a core stage already owns.**
> The RAG source pointer is `rag_provenance`, **not** `provenance`: the core payload
> already owns `provenance` as a *string* (`PayloadProvenance =
> Literal["extracted","inferred","curated","enriched"]` in `t2pw.schema`), present on
> every entity/process via `PayloadCommonRecord`. Writing a dict under `provenance`
> is a repurpose/retype of an existing field — forbidden by the rule above — and it
> trips the runtime shape validator ("Expected a string"). The additive set is
> deliberately namespaced (`rag_*`) so a RAG key can never shadow a core one. (See the
> 2026-07-21 change-log entry.)

---

## Concrete do / don't

**Do**

- Create `t2pw/rag/acquire.py`, `t2pw/rag/store.py`, `t2pw/rag/synthesize.py`, etc.
- Feed evidence in through S1/S2; hand a finished `Payload` in through S3.
- Read `qa_graph` / gate reports (S4) to find gaps.
- Guard every RAG entry with the `RAG_ENABLED` flag in the orchestrator (S5).

**Don't**

- Add a branch inside `normalize_process_payload`, `map_payload`, `run_audit`'s
  body, or the PWML writer that "does something different when RAG is on."
- Import `t2pw.rag` from any stage module. Dependencies point **RAG → core**, never
  core → RAG.
- Weaken a gate to let synthesized content through. If synthesized content fails a
  gate, fix the synthesis (WP5) or report the gap — never the gate.

---

## How to verify you did not break it

1. `grep -rn "t2pw.rag" src/t2pw/pipeline src/t2pw/mapping src/t2pw/curation src/t2pw/pwml`
   returns **nothing**. (No stage imports RAG.)
2. With `RAG_ENABLED=false`: full test suite and a reference PWML export are
   identical to pre-change `main`.
3. Every new field RAG adds is optional and absent when RAG is off.

---

## Sanctioned exceptions

The invariant is the default and the strong preference. Where it has been
**deliberately** relaxed by explicit decision, it is recorded here so the
verification above stays honest (i.e. so a reader knows the "no core stage edited"
claim has a named, bounded exception rather than silent drift):

- **2026-07-21 — `process_normalizer.py` pathway-metadata-blob guard.** As
  defense-in-depth against `" ; "`-joined pathway-metadata garbage names (see the
  change-log entry of that date), a narrow, additive guard
  (`_quarantine_pathway_metadata_blobs`) was added to the core normalizer. It is gated
  to fire only on the garbage signature, changes no existing behavior (zero test
  regressions), and does **not** import `t2pw.rag` (verification #1 still holds). The
  *primary* fix for that bug lives inside `t2pw.rag` (`synthesize.py`); this core edit
  is a redundant second line of defense, not the mechanism RAG relies on.

Anything not listed here is still bound by the full invariant above.
