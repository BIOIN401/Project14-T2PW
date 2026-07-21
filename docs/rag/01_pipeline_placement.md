# Pipeline Placement — The R-Stages and Their Seams

This maps each RAG stage (R0–R5) onto the existing pipeline and names the exact
seam it uses. Seams are defined in [`03_separation_invariant.md`](03_separation_invariant.md).

---

## Ordered flow

```
Seed paper + optional "incomplete/novel" flag
        │
        ▼
   [Stage 0: Preprocess]  ← existing (preprocessor.py)
        │
        ▼
   [R0: Triage]           ← NEW. Reads Stage 0 output + user flag. Decides RAG on/off.
        │  RAG off ────────────────────────► existing single-paper pipeline (unchanged)
        │  RAG on
        ▼
   [R1: Acquire] → [R2: Select] → [R3: Ingest/Embed]      ← NEW corpus build
        │
        ▼
   [Stage 1: Extract seed]  ← existing; RAG injects retrieved context via seam S1
        │
        ▼
   [R4: Gap-Retrieve]     ← NEW. Detect gaps from reports (S4), retrieve evidence.
        │
        ▼
   [R5: Synthesize]       ← NEW. Merge seed + evidence into ONE Payload.
        │  (emits a standard Payload via seam S3)
        ▼
   [Stage 2B → 3 → 4 → 4a → 5 → 6 → 7 → 8]  ← existing core, UNCHANGED
                                             (R4 also feeds Stage 4 via seam S2)
```

Note the two feed directions for R4: it primes the **seed extraction** (S1) and,
during the audit loop, supplies gap-targeted evidence to **Stage 4** (S2). R5's
output enters at the **Stage 2B boundary** (S3).

---

## Stage-by-stage

### R0 — Triage  (WP7)
- **Input:** Stage 0 context dict + the user's flag.
- **Decision:** is the pathway novel/incomplete? (explicit flag, or auto: low
  `scope_clarity_score`, dangling reactions, orphan metabolites, unmapped enzymes).
- **Output:** a boolean + a reason, consumed by the orchestrator (S5). If false,
  RAG never runs.
- **Must not:** modify Stage 0. It *reads* Stage 0's output.

### R1 — Acquire  (WP1)
- **Input:** seed pathway context (name, organism, key compounds/proteins, gap terms).
- **Action:** query EuropePMC / NCBI (reuse existing `HttpClient` + endpoints in
  `map_ids.py`), optionally Crossref / Semantic Scholar / bioRxiv. Download full text.
- **Output:** list of candidate papers (id, title, organism, full text, source URI).

### R2 — Select  (WP2)
- **Input:** candidate papers + seed context.
- **Action:** score relevance to target pathway + organism (reuse `preprocessor.py`
  per candidate), dedupe vs seed, cap count.
- **Output:** the selected subset + a selection report.

### R3 — Ingest / Embed  (WP3)
- **Input:** selected papers + structured DB reaction records + the existing
  `reference/` example corpus.
- **Action:** section-aware chunk → embed → upsert to the vector store
  ([`02_vector_store.md`](02_vector_store.md)). Also produce the hybrid upgrade of
  the lexical motif index.
- **Output:** a populated, persisted vector store keyed by provenance.

### R4 — Gap-Retrieve  (WP4)
- **Input:** the current pathway payload + gate/`qa_graph` reports (read-only, S4).
- **Action:** detect each gap, form a query, retrieve top-k evidence chunks
  (hybrid semantic + lexical). Hand evidence to the core via:
  - **S1** for the seed extraction (through `pathway_context` / `user_task_context`), and
  - **S2** for the audit loop (`run_audit(..., retrieval_context=...)`).
- **Output:** per-gap evidence bundles (text + provenance).

### R5 — Synthesize  (WP5)
- **Input:** seed extraction + per-gap evidence bundles.
- **Action:** merge into one connected pathway — stitch reactions end-to-end,
  reconcile cross-paper synonyms, resolve conflicts by evidence weight, attach
  provenance to every element.
- **Output:** a **standard `Payload`** (S3). From here the existing Stage 2B→8 runs
  with no change.

---

## What the core pipeline never learns

The core stages do not know RAG exists. They receive:

- a possibly richer `user_task_context` / `pathway_context` (S1) — already a string/dict they accept;
- a `retrieval_context` string at audit (S2) — already a parameter;
- a `Payload` at Stage 2B (S3) — the shape they already consume, plus optional
  additive provenance keys they ignore.

That is the whole contact surface. Everything else about RAG is internal to
`t2pw.rag`.
