# Architecture

## Design constraints that shaped it

1. **Additive and reversible.** Everything lives in a new package,
   `src/t2pw/mining/`. The only edit to an existing file is a 13-line guarded
   block in `src/t2pw/app/streamlit_app.py` that is a no-op when the flag is off.
2. **Reuse, do not fork.** Acquisition, eligibility screening, chunking,
   embeddings, the vector store and the LLM client are the *existing* modules,
   reached through thin adapters. Copying `t2pw.rag` to change a few policies
   would create two implementations that drift.
3. **Separate the layers.** Paper evidence, normalized entities, model
   interpretation, admission decision, graph connectivity and exploratory
   hypothesis are six different things with six different records. Nothing
   collapses them into one model response.
4. **One controller, two front ends.** The CLI and the Streamlit page both call
   `MiningController`. Neither reimplements a round loop; a test asserts this.

---

## Module map

```
src/t2pw/mining/
    flags.py          master gate. The ONLY thing __init__ imports.
    config.py         MiningConfig, MiningBudgets, MiningModelConfig, StartingInput
    ids.py            content-addressed, counter-free, clock-free id minting
    models.py         PaperRecord, EvidenceSpan, ExtractedClaim, ClaimDecision,
                      ConflictRecord, FrontierItem, LiteratureQuery, GraphDelta,
                      ModelCallRecord, RoundCheckpoint, provenance_chain_complete
    species.py        the guardrail: 7 states, ortholog barrier, local normalization
    normalize.py      EntityRegistry: within-run entity folding + alias provenance
    graph.py          ResearchGraph, ResearchReaction, CandidateSubgraph,
                      connection_between  <- the only connectivity oracle
    claims.py         ClaimStore: dedup, conflicts, versioned decision history
    admission.py      the ordered admission gates; claim -> decision -> reaction
    resolver.py       the DEFERRED external-database seam (no-op in v1)
    budget.py         BudgetLedger, the closed STOP_REASONS set, cancellation
    modelclient.py    the model boundary: empty/truncated/schema handling + pricing
    extraction.py     prompts, JSON schema, per-field citation checking
    frontier.py       expansion-driven frontier generation (12 kinds)
    queries.py        traceable literature queries (required terms; rendered by rag.acquire.build_query)
    acquisition.py    adapter over rag.acquire + explicit refs + uploads
    retrieval.py      adapter over rag.ingest/embed/store; per-run index
    preflight.py      visible service checks before anything expensive
    artifacts.py      RunDirectory, RunManifest, per-round flush
    report.py         summary.json + report.md
    regenerate.py     rebuild report/summary/views from artifacts alone
    viz.py            the two view payloads + DOT + the flat evidence table
    controller.py     the round loop, checkpointing, resume, provenance audit
    cli.py            python -m t2pw.mining {run,resume,status,report,preflight,list}
    ui/
        entry.py      the gated hook streamlit_app.py calls
        page.py       the twelve-section Streamlit page
        network.py    self-contained SVG renderer (no CDN, offline)
        app.py        standalone `streamlit run` entry point
```

---

## Seams into the existing stack

Every one of these is a **read** of an existing module with no behaviour change
to it. No existing module imports `t2pw.mining`; the dependency arrow points one
way, exactly as the RAG separation invariant requires of `t2pw.rag`.

| Existing component | Used for | Adapter |
|---|---|---|
| `t2pw.rag.acquire.search_candidates` / `fetch_full_text` | paper discovery and full text | `acquisition.PaperSource` |
| `t2pw.rag.acquire.build_query` | the boolean literature query grammar | `queries.build_query` |
| `t2pw.rag.acquire.CandidatePaper` | the acquisition record shape | `acquisition.record_from_candidate` |
| `t2pw.rag.eligibility.screen_candidate` | relevance / organism screening | `acquisition.Acquirer.screen` |
| `t2pw.rag.ingest.chunk_paper` | section-aware chunking with provenance | `retrieval.PassageIndex._chunk` |
| `t2pw.rag.embed.Embedder` | cached embeddings + offline lexical fallback | `retrieval.PassageIndex` |
| `t2pw.rag.store.get_vector_store` | the vector index | `retrieval.PassageIndex` |
| `t2pw.llm.client.chat_detailed` | the model boundary and its diagnostics | `modelclient.MiningModelClient` |
| `t2pw.llm.client.get_token_stats` | per-call token deltas | `modelclient` |
| `t2pw.extraction.pdf_parser` | uploaded PDF text | `acquisition._extract_pdf_text` |
| `t2pw.config.ensure_dotenv_loaded` / `rag_config` | environment resolution | `flags`, `preflight` |

**Optional hooks added to existing components: none.** Nothing needed one. If a
future change does, it must be a backward-compatible optional parameter whose
default preserves today's behaviour exactly.

### Deliberate isolation

Mining-specific policy lives **only** in the mining package and is never routed
through the production path:

* admission (`admission.py`) — the PWML pipeline has its own, unchanged;
* frontier generation (`frontier.py`) — the production RAG controller is
  gap-driven and stays that way;
* graph management (`graph.py`), provenance (`models.py`), visualization
  (`viz.py`) and stopping policy (`budget.py`).

The run's vector index defaults to the **in-memory** backend so mining cannot
write into the production RAG index.

---

## Why a separate controller

`t2pw.rag.controller.run_rag_loop` is **gap-driven**: it detects a fixed
vocabulary of gaps in a pathway it is trying to complete, retrieves evidence for
those gaps, and stops when they are filled or the budget runs out. That is the
right shape for completing a known pathway.

Mining is **expansion-driven**: there is no target pathway. Each round's
questions are derived from what the previous round just discovered, so the
network grows outward. Reconfiguring the gap detector to do that would mean
teaching it twelve new gap kinds, a candidate-quarantine concept, a bridge
search and a promotion rule — i.e. writing this controller inside that one, while
changing behaviour the production path depends on.

**What is freed is the exploration strategy. The evidence requirements are not
loosened at all** — they are stricter than production in one respect: every
material field must cite the passage that supports it.

---

## Anatomy of one round

`MiningController._run_round(index)`:

| # | Step | Module |
|---|---|---|
| 1 | select frontier items (priority-ordered, unexhausted) | `frontier` |
| 2 | build one traceable query per item: its own terms AND the species; a second formulation (+ topic) if the first finds no new paper | `queries` |
| 3 | acquire, screen, cache; record inclusions and exclusions | `acquisition` |
| 4 | index new papers; retrieve passages as evidence spans | `retrieval` |
| 5 | extract claims; drop every uncited field (an enzyme / gene / species name found verbatim in the claim's own cited participant span is kept, recorded as `verbatim_cited_span`) | `extraction` |
| 6 | normalize entities within the run | `normalize` |
| 7 | register claims: dedup, keep independent evidence | `claims` |
| 8 | judge: accepted / candidate / conflicting / incomplete / rejected / hypothesis | `admission` |
| 9 | place admitted reactions **by chemistry**; quarantine the rest | `graph` |
| 10 | record the graph delta | `models.GraphDelta` |
| 11 | generate the next frontier from the delta | `frontier` |
| 12 | write the round dir, the checkpoint and the state; flush artifacts | `artifacts` |

Steps 8 and 9 are deliberately separate: **admission decides whether the biology
is defensible; placement decides where it connects.** Neither can do the other's
job, which is why a wrong-species reaction that shares a metabolite with the
primary graph stays quarantined.

### Concurrency

None. Every heavy step is sequential: one model call at a time, one paper
indexed at a time, one budget ledger. The existing architecture provides no
proven safe concurrency mechanism around the shared caches and the vector store,
so none is used.

---

## The six separated layers, concretely

| Layer | Record | Example field |
|---|---|---|
| factual paper evidence | `PaperRecord`, `EvidenceSpan` | `text`, `pmcid`, `locator` |
| normalized entity | `ResearchEntity` | `key`, `aliases`, `alias_provenance` |
| model interpretation | `ExtractedClaim` | `enzyme`, `field_evidence`, `model` |
| admission decision | `ClaimDecision` | `status`, `reason`, `evidence_level` |
| graph connectivity | `ResearchReaction`, `ConnectionEvidence` | `graph_status`, `via_entity_id` |
| exploratory hypothesis | `ExtractedClaim.hypothesis`, `hypothesis_only` | — |

A reader of any artifact can always tell which layer a statement came from.
