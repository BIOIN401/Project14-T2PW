# WP5 — Synthesis

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core only through named seams. No RAG logic inside a stage module.
> Additive metadata only. Full text: [`../03_separation_invariant.md`](../03_separation_invariant.md).

## Purpose

Merge the seed extraction plus per-gap evidence bundles into **one connected
pathway** and emit it as a **standard `Payload`** that enters the core at the Stage
2B boundary (seam S3). This is where the "novel pathway" is actually assembled.

## Background you need

- **Payload shape:** the `TypedDict`s in `t2pw.schema` (entities, processes, actors).
  The synthesizer's output must satisfy the Stage 1 structural contract
  (`validate_post_extraction`) so Stage 2B accepts it — see `docs/pipeline.md`.
- **Actor schema:** `docs/pipeline.md` "Actor field schema" (`entity`, `entity_type`,
  `role`, `evidence`, `confidence`, `provenance`). The RAG source pointer goes in the
  additive `rag_provenance` key (**not** the core `provenance` string field — see the
  additive-metadata rule in [`../03_separation_invariant.md`](../03_separation_invariant.md)).
- **The non-invention rule:** `docs/pipeline.md` Stage 4 — never invent
  stoichiometry/participants without evidence. Same rule applies here, harder: no
  reaction/entity without a retrieved source.

## What to build

- `src/t2pw/rag/synthesize.py`:
  - `synthesize(seed_payload, evidence_bundles, seed_context) -> Payload`:
    1. **Stitch** — connect reactions so a product feeds the next reaction's input
       across papers (close dangling ends only where evidence supports it).
    2. **Reconcile synonyms** — unify cross-paper names (reuse
       `BIOCHEMICAL_ALIAS_MAP` and same-as logic patterns from
       `process_normalizer.py`; do not import RAG into it — copy/adapt read-only).
    3. **Resolve conflicts** — when papers disagree (direction, stoichiometry,
       compartment), pick by evidence weight; record the alternatives.
    4. **Attach provenance** — every synthesized entity/process/actor gets
       `rag_provenance` + `evidence` keyed to `source_id`/`source_uri`.
  - `to_payload(...)` returning the standard shape (no RAG-only required keys;
    provenance is additive/optional).
  - **Carry forward the seed's contextual scaffolding** — synthesis rebuilds
    `compounds`/`proteins`/`protein_complexes` from evidence, but the seed's
    non-reaction scaffolding entities (`species`, `subcellular_locations`,
    `cell_types`, `tissues`, and top-level `biological_states`) are **not**
    reaction-derived and are copied forward from `seed_payload` unchanged. They are
    contextual, evidence-exempt (see `_EVIDENCE_ENTITY_BUCKETS` in
    `provenance.py`), and required downstream — without `entities.species` the Stage
    2B post-mapping gate aborts with `species_required` (see the 2026-07-21
    change-log entry).

## Hard constraints

- Output must pass `validate_post_extraction` before it is handed to Stage 2B.
- Every reaction and non-cofactor entity must carry at least one provenance pointer.
  Elements without evidence are **omitted and reported as unresolved gaps**, never
  fabricated.
- Do not pre-run normalization, mapping, or audit — that is the core's job. Emit the
  payload and let Stage 2B→8 do their work (S3).

## Depends on / blocks

- Depends on: WP0, WP4.
- Blocks: WP6, WP7.

## Acceptance criteria

- Two canned papers (reactions 1–3 and 4–6) → one connected payload where reaction 3's
  product is reaction 4's input, every element has provenance, and it passes
  `validate_post_extraction`.
- A gap with no supporting evidence stays unfilled and appears in the unresolved
  report — nothing invented.
- `docs/change_log.md` entry.

## Out of scope

Gate/provenance *enforcement* wiring (WP6), orchestration/UI (WP7).
