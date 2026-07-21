# WP1 — Acquisition

> **PINNED RULE — Separation Invariant.** All RAG code lives in `t2pw.rag` and
> touches the core only through named seams. No RAG logic inside a stage module.
> Additive metadata only. Full text: [`../03_separation_invariant.md`](../03_separation_invariant.md).

## Purpose

Given the seed pathway context and its gap terms, fetch candidate papers from
literature APIs and download their full text. Output feeds WP2 (selection).

## Background you need

- **The HTTP plumbing already exists.** Reuse the shared `HttpClient` and the
  patterns in `src/t2pw/mapping/map_ids.py`:
  - EuropePMC search: `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
    (`lookup_literature_protein_aliases`, ~line 293).
  - EuropePMC full text: `_europepmc_full_text` (~line 245) →
    `.../{id}/fullTextXML`, already converts XML to plain text.
  - NCBI eutils: `_NCBI_EUTILS_BASE` (~line 3055), `esearch`/`efetch`.
- **PDF fallback:** `src/t2pw/extraction/pdf_parser.py` for user-uploaded PDFs.
- Do **not** reinvent HTTP, retries, or rate limiting — extend what map_ids uses.

## What to build

- `src/t2pw/rag/acquire.py`:
  - `search_candidates(context, *, sources=("europepmc","ncbi"), max_papers=RAG_ACQUIRE_MAX_PAPERS) -> list[CandidatePaper]`
  - `fetch_full_text(candidate) -> str` (reuse `_europepmc_full_text`; add PMC/DOI
    resolution as needed).
  - Optional extra sources behind flags: Crossref, Semantic Scholar, bioRxiv.
  - Dedupe against the seed (by DOI/PMID/title-normalize).
  - On-disk cache under `data/rag_index/acquire_cache/` keyed by query hash
    (offline-first; re-runs must not re-hit the network).
- `CandidatePaper` dataclass: `{id, source, title, abstract, organism, full_text, source_uri, year}`.

## Query construction

Build queries from the seed context: `pathway_name`, `likely_organism`,
`key_compounds`, `key_proteins`, and the gap terms WP4 will later supply. Prefer
organism-scoped queries (`... AND "<organism>"`), mirroring how map_ids scopes
UniProt/EuropePMC lookups.

## Depends on / blocks

- Depends on: WP0.
- Blocks: WP2, WP3.

## Acceptance criteria

- Offline test: mocked `HttpClient` returns canned EuropePMC JSON → deterministic
  `list[CandidatePaper]`; no live network in tests.
- Cache hit avoids a second network call.
- `grep` shows `acquire.py` imports from `t2pw.mapping` helpers or a shared client —
  it does **not** duplicate URL/retry logic, and nothing in `t2pw.mapping` imports
  `t2pw.rag`.
- `docs/change_log.md` entry.

## Out of scope

Ranking/selection (WP2), embedding (WP3). Just fetch and normalize to text.
