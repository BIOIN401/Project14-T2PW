# WP6 — Provenance & Gates

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core only through named seams. No RAG logic inside a stage module.
> Additive metadata only. Full text: [`../03_separation_invariant.md`](../03_separation_invariant.md).

## Purpose

Guarantee the initiative's core promise: **no element without evidence.** Validate
that every synthesized element carries usable provenance, and prove the synthesized
payload survives the *existing* Stage 3/8 gates unmodified.

## Background you need

- The existing gates you must **not weaken**: `run_strict_post_normalization_gates`
  (Stage 3) and `validate_required_pwml_contract` (Stage 8, `pwml/ir.py`). See
  `docs/pipeline.md` "Stage contract summary."
- Provenance field defs from WP0 (`t2pw/rag/provenance.py`).
- The additive-metadata rule: new keys optional; existing gates ignore them.

## What to build

- `src/t2pw/rag/provenance.py` (extend WP0 stub):
  - `validate_provenance(payload) -> ProvenanceReport` — every reaction + non-cofactor
    entity has ≥1 resolvable `source_id`/`source_uri`; flag any that don't.
  - `strip_provenance(payload) -> Payload` — return a provenance-free copy, used to
    prove the core gates pass on the plain payload too.
- Tests that assert:
  - The synthesized payload passes the **unmodified** Stage 3 gate and Stage 8
    contract (call them directly; do not fork them).
  - `validate_provenance` fails a payload with an unsourced reaction.
  - Provenance keys are absent/ignored when `RAG_ENABLED=false`.

## The critical guardrail

If synthesized content fails a core gate, the fix goes in **WP5 (synthesis)** or is
reported as an unresolved gap — **never** by editing the gate. This WP's tests are
the tripwire that catches anyone loosening a gate to push RAG output through.

## Depends on / blocks

- Depends on: WP0, WP5.
- Blocks: WP7.

## Acceptance criteria

- Direct-call tests: Stage 3 gate + Stage 8 contract pass on a good synthesized
  payload and fail on a bad one — using the real functions, no RAG-specific variant.
- `validate_provenance` catches unsourced elements.
- `grep -rn "t2pw.rag" src/t2pw/{pipeline,pwml,mapping,curation}` still returns
  nothing.
- `docs/change_log.md` entry.

## Out of scope

Triage trigger and UI (WP7).
