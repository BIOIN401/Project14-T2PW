# RAG — Overview, Goal & Guardrails

Read this first. It defines the terms every other RAG doc uses.

---

## The goal in one sentence

When a user uploads a single paper for a pathway that is **not fully described in
that paper**, T2PW should gather evidence from other papers and databases and
assemble one connected, exportable pathway — with every part traceable to a source.

---

## What "novel pathway" means here (and does not)

**It means:** a novel *connection* of steps that are each individually supported by
retrieved evidence. Paper A describes reactions 1–3, paper B describes reactions
4–6, a KEGG record supplies the cofactor for reaction 4 — RAG stitches them into
one pathway that no single source stated in full.

**It does not mean:** inventing chemistry. The synthesizer may not emit a reaction,
enzyme, metabolite, stoichiometry, or compartment that no retrieved passage
supports. This mirrors the pipeline's existing rule that the audit must never
invent a stoichiometric ratio ([`docs/pipeline.md`](../pipeline.md), Stage 4). If
the evidence is not there, the element stays out and the gap is reported — it is
not filled with a guess.

---

## The trigger

RAG runs only when the pathway is **novel/incomplete**. That state is reached two ways:

1. **Explicit** — the user flags the upload as "unknown / incomplete pathway."
2. **Automatic** — Stage 0 / early gates signal incompleteness: low
   `scope_clarity_score`, dangling reactions (a product that is no reaction's
   input and no pathway output), orphan metabolites, or enzymes that never map.

If neither fires, RAG stays off and the existing single-paper pipeline runs
unchanged. See WP0 (feature flag) and WP7 (triage).

---

## Guardrails (non-negotiable)

1. **Separation.** RAG is a separate subsystem. It never adds logic inside an
   existing stage module. See [`03_separation_invariant.md`](03_separation_invariant.md).
2. **Additive-only metadata.** RAG may attach `rag_provenance`/`evidence` fields to
   entities and processes. It may never change the meaning or shape of a field
   owned by an existing stage — the same boundary rule the pipeline already enforces.
   (The source pointer is `rag_provenance`, not `provenance`: the core already owns
   `provenance` as a string. See [`03_separation_invariant.md`](03_separation_invariant.md).)
3. **Evidence-bound generation.** No element without a retrieved source. Provenance
   is mandatory, not decorative (enforced in WP6).
4. **Offline-first.** Like the rest of the pipeline (`data/pathwhiz_id_db.json`,
   the id/enrichment caches), RAG must degrade gracefully with no network and no
   external embedding endpoint — it falls back to what is already indexed.
5. **Opt-in.** With `RAG_ENABLED=false`, behavior is byte-for-byte today's behavior.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Core pipeline** | The existing Stages 0–8 in `t2pw.pipeline`, `t2pw.mapping`, `t2pw.curation`, `t2pw.pwml`. |
| **RAG subsystem** | New `t2pw.rag` package implementing Stages R0–R5. |
| **Seam** | One of the few pre-existing contact points where RAG may hand data to the core (context params, `retrieval_context`, the `Payload` boundary, read-only reports). |
| **Seed paper** | The single document the user uploaded. |
| **Candidate paper** | A paper fetched by the acquisition stage, before selection. |
| **Chunk** | A passage of a paper/DB record, embedded and stored in the vector store. |
| **Gap** | A specific missing piece in the pathway graph (dangling reaction, orphan metabolite, unmapped enzyme, missing precursor). |
| **Provenance** | The source identity (paper id, section, DB record) backing an entity/process. |
| **Motif index** | The existing lexical example-retrieval index in `t2pw.sbml.examples`; WP3 upgrades it to hybrid. |

---

## What already exists that we build on

- **Literature APIs are wired.** EuropePMC search + full-text XML fetch
  (`_europepmc_full_text`, `map_ids.py`) and NCBI eutils (`map_ids.py`), over a
  shared `HttpClient`. WP1 extends these; it does not reinvent HTTP.
- **A lexical retrieval layer exists.** `t2pw.sbml.examples`
  (`build_retrieval_context`, `retrieve_motif_examples`) already retrieves example
  pathways by token overlap and injects them into Stage 1. WP3 makes it semantic;
  WP4 reuses its injection path.
- **An injection seam exists at Stage 4.** `run_audit(..., retrieval_context="")`
  already accepts retrieved text (`audit_json_llm.py`).
- **Offline-index precedent exists.** `data/pathwhiz_id_db.json` and the id/
  enrichment caches show the pattern the vector store should follow.
