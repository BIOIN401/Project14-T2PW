# ORCH-743 — pathway-mining iteration: result

2026-09-24 · Lead Orchestrator. **Nothing merged. Nothing pushed.** Production is still frozen at `5f6a7aa3`; every
change is inside `src/t2pw/mining/` on `experiment/pathway-mining`.

## Bottom line

**Pathway mining is still BLOCKED for evaluation. The blockers left are not the four it was asked to fix.**
Two iterations fixed everything in the four ORCH-743 areas, and pilots confirmed each fix live:
- lineage;
- uncited fields;
- search collapse;
- the screen;
- empty extractions;
- NCBI paper identity.

The target graph is still empty because of problems **outside** the scope the owner set:
- the species gate (`species.py`, which the owner asked to leave untouched);
- a run-stopping cap policy;
- the frontier cap;
- provider routing (production `llm/`).

These need product-owner decisions, below. Per the pre-registration, iterations stopped at two.

## The three pilots (same config; only the code changed)

| | pilot-001 (`23e1ae83`) | pilot-002 (`ecf6b9a0`) | pilot-003 (`f9556288`) |
|---|---|---|---|
| rounds / stop | 2 / frontier exhausted | 3 / 150 papers discovered | 1 / new-reaction cap |
| discovered / included / processed | 21 / 1 / 1 | 150 / 11 / 10 | 64 / 4 / 4 |
| uncited-field drops | 59 | 0 | 0 |
| empty provider requests | 0 | 21 | 2 (retried) |
| wrong NCBI ids | not reached | 7 of 8 | 0 of 28 |
| target (A. thaliana) reactions | 0 | 1 (in vitro, wrong id) | 0 |
| correct A. thaliana-tied candidates | 0 | 1 | 8 |
| bio verdict | NOT_PLAUSIBLE | PLAUSIBLE_BUT_THIN | NOT_PLAUSIBLE |
| est. cost | $0.0034 | $0.0589 | $0.0356 |

The records are on `experiment/pathway-mining-pilot-001` (`dcb64fc4`), `-002` (`d6cb9868`) and `-003` (`b7e26d8d`).
Each holds the full run, the wrapper reports (0 survivors everywhere), the artifact audit and the bio-audit.

## What changed on `experiment/pathway-mining` (`23e1ae83` → `f9556288`)

| round | fixes | review |
|---|---|---|
| iteration 1 (items 1–4) | raw lineage; per-field citation prompt plus a `verbatim_cited_span` fallback that **never** counts as species evidence; the search requires the item's own terms, has a 25-result window, 2 attempts plus a wider attempt, NCBI, and a hard processed cap; a mining-only screen rescue (2 routes, a foreign-host title guard, a title-organism check) | 2 repair rounds; 7 blocking findings fixed (bio B-1 to B-4, review B1 to B3); final `APPROVE_WITH_FINDINGS` / `SOUND_WITH_FINDINGS` |
| iteration 2 (IT2-1 to IT2-5) | 5-passage extraction batches at 12k tokens; NCBI identity from the PMID only, with the PMCID resolved via EuropePMC and mismatched full texts refused; budget-cut windows no longer exhaust items; grape/vitis hosts; per-rung lineage | `APPROVE_WITH_FINDINGS` / `SOUND_WITH_FINDINGS`, no blocking findings |

- **Tests:** mining suite 226 → **281**; SMOKE **508** with the feature off at every tip.
- **G9 base arms** (all behavioural, run by the orchestrator in detached base worktrees): `012`, `020`–`022`, `025`–`028`, `031`, `035`.
- **Live diagnostic spend:** about $0.033 across 7 calls.

## Decisions for the product owner (none is taken here)

1. **The species gate (`species.py`, F-1).** Markers are tested against whole multi-kilobyte spans, not the claim's sentence, so 26 of 35 labels are artefacts. It held back a correct in-planta reaction (JA → JA-Ile, GH3.10) and admitted pilot-002's in-vitro 13-LOX step.
   - **Fixing it means touching `species.py`.**
   - Order matters: land the review grading (F-2) and bibliography/front-matter exclusion (F-4) first, or review legends become admitted "primary" evidence.
2. **The `max_new_reactions_per_round` semantics.** It is a run-stopper (ORCH-742a) that counts quarantined, non-admissible leads. In pilot-003, one off-species review stopped the run after round 1. Options: throttle within a round, or count only admitted reactions.
3. **The frontier cap (60).** It silently cut 137 items in pilot-003, including every `missing_species_evidence` query, so target-species evidence is never sought.
4. **Provider routing (`llm/client.py`, production).** Some answers come from a route that does no reasoning (thin, placeholder-filled), and the provider is not recorded. Pinning the provider, or passing the reasoning setting through, needs production code.
5. **The production PubMed parser (`rag/acquire.py:536`).** It takes a cited reference's PMCID/DOI. Mining now works around it; production RAG may have the same exposure (a hypothesis). It needs its own card.

## Other registered findings (not fixed)

- **Latent contract violations:** IT2-N1 / F-5, participants and tissue not citation-gated; IT2-N2 / F-6, last-writer-wins species merge; F-2, reviews never graded.
- **Report defects:** report.md misstatements; F9 (the mining CLI does not load `.env`; all three pilots used a launcher).
- **Frontier and identity:** order-dependent bridge anchors; lexical entity splits.
- **Host detection:** binomial hosts still undetected (N-9).
- Full lists are in each pilot's `PILOT-00N-RESULT.md` and audit.

## Left on disk

- **Frozen pilot trees:** `C:/t/mining-pilot`, `-pilot2`, `-pilot3` (each on its record branch).
- **Base trees:** `C:/t/mining-base2` (`23e1ae83`) and `C:/t/mining-rep-base` (`ecf6b9a0`).
- **Run and audit scratch:** `C:/t/mining-pilot-runs{,2,3}`; `C:/t/mining-iter` (the G9 manifest and diagnostics); `C:/t/bt/mining-iter` (reviewer probes).

Nothing was pruned.
