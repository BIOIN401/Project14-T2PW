# Pathway mining run `pilot-001-jasmonate-ath`

> **This run stopped at a configured limit rather than finishing its work.** Stop reason: `frontier_exhausted` - no unexhausted frontier items remain.
> The graph below is what the budget bought, not a complete answer.

- **Target species:** Arabidopsis thaliana (normalized `arabidopsis thaliana`)
- **Topic:** jasmonate biosynthesis
- **Rounds completed:** 2
- **Stop reason:** `frontier_exhausted`

> **This run produced a research graph, not PWML.** The PWML exporter was not invoked, no PathWhiz identifier was required, and no PWML file was written.

> External biochemical database validation (PathBank, PWC, HMDB, KEGG, ChEBI, PubChem, UniProt) is DEFERRED in version 1. Every entity and reaction carries status 'not_evaluated'. Admission was on paper evidence alone; no identifier in this run has been checked against any external database.

## Literature

| papers | count |
|---|---|
| discovered | 21 |
| screened | 21 |
| included | 1 |
| excluded | 20 |
| fully processed | 1 |

Evidence type of included papers: `unknown` x1

Exclusion reasons:
- 6 x off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract
- 4 x context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it
- 4 x omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract
- 2 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomic)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: enriched)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: upregulated, rna seq)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: multi omics)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: sequencing)

## Claims and decisions

- Claims extracted: **19** across **19** distinct reaction identities
- Passages retrieved: **20**

Latest decision per reaction identity:

- `candidate`: 19

## Research graph

- Primary graph: **0** reactions, **10** chemical connections, **30** entities
- Candidate subgraphs: **14** components holding **19** reactions
- Reactions supported by more than one paper: **0**

Species state of every reaction node:
- `in_vitro_or_cell_free`: 1
- `inferred_only`: 10
- `species_not_stated`: 8

## Conflicts

No conflicting evidence was recorded.

## Growth by round

| round | added | candidates | promoted | updated | conflicts | useful |
|---|---|---|---|---|---|---|
| 1 | 0 | 19 | 0 | 2 | 0 | yes |
| 2 | 0 | 0 | 0 | 0 | 0 | no |

## Cost and usage

- Model calls: **1**
- Input tokens: **22,843**; output tokens: **3,629**; cached: **0**
- Estimated cost: **$0.003373** (limit $5)
- Elapsed: **303.7s**

> Estimated from configured per-million-token rates. This is NOT a billed amount and does not reflect provider discounts, cached-token pricing or minimum charges.

## Retrieval

- Papers indexed: 1; chunks: 20
- Embedding mode: `api`

## Warnings, partial failures and unresolved questions

- ERROR: claim[0].enzyme='MpFAD5' (no valid citation)
- ERROR: claim[0].gene='Mp3g10660' (no valid citation)
- ERROR: claim[0].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[1].enzyme='MpFAD6' (no valid citation)
- ERROR: claim[1].gene='Mp4g24020' (no valid citation)
- ERROR: claim[1].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[2].enzyme='MpFAD7' (no valid citation)
- ERROR: claim[2].gene='Mp5g20700' (no valid citation)
- ERROR: claim[2].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[3].enzyme='MpFAD7' (no valid citation)
- ERROR: claim[3].gene='Mp5g20700' (no valid citation)
- ERROR: claim[3].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[4].enzyme='MpFAD3' (no valid citation)
- ERROR: claim[4].gene='Mp3g17690' (no valid citation)
- ERROR: claim[4].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[5].enzyme='MpDES6' (no valid citation)
- ERROR: claim[5].gene='Mp6g17330' (no valid citation)
- ERROR: claim[5].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[6].enzyme='MpDES6' (no valid citation)
- ERROR: claim[6].gene='Mp6g17330' (no valid citation)
- ERROR: claim[6].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[7].enzyme='MpELO1' (no valid citation)
- ERROR: claim[7].gene='Mp6g02350' (no valid citation)
- ERROR: claim[7].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[8].enzyme='MpELO1' (no valid citation)
- ERROR: claim[8].gene='Mp6g02350' (no valid citation)
- ERROR: claim[8].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[9].enzyme='MpDES5' (no valid citation)
- ERROR: claim[9].gene='Mp3g13250' (no valid citation)
- ERROR: claim[9].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[10].enzyme='MpDES5' (no valid citation)
- ERROR: claim[10].gene='Mp3g13250' (no valid citation)
- ERROR: claim[10].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[11].enzyme='MpAOS1' (no valid citation)
- ERROR: claim[11].gene='Mp3g21350' (no valid citation)
- ERROR: claim[11].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[12].enzyme='MpAOS1' (no valid citation)
- ERROR: claim[12].gene='Mp3g21350' (no valid citation)
- ERROR: claim[12].species='Marchantia polymorpha' (no valid citation)
- ERROR: claim[13].enzyme='MpAOS1' (no valid citation)

