# Pathway mining run `pilot-002-jasmonate-ath`

> **This run stopped at a configured limit rather than finishing its work.** Stop reason: `max_papers_discovered_reached` - papers discovered: 150 >= limit 150.
> The graph below is what the budget bought, not a complete answer.

- **Target species:** Arabidopsis thaliana (normalized `arabidopsis thaliana`)
- **Topic:** jasmonate biosynthesis
- **Rounds completed:** 3
- **Stop reason:** `max_papers_discovered_reached`

> **This run produced a research graph, not PWML.** The PWML exporter was not invoked, no PathWhiz identifier was required, and no PWML file was written.

> External biochemical database validation (PathBank, PWC, HMDB, KEGG, ChEBI, PubChem, UniProt) is DEFERRED in version 1. Every entity and reaction carries status 'not_evaluated'. Admission was on paper evidence alone; no identifier in this run has been checked against any external database.

## Literature

| papers | count |
|---|---|
| discovered | 150 |
| screened | 150 |
| included | 11 |
| excluded | 139 |
| fully processed | 10 |

Evidence type of included papers: `unknown` x11

Exclusion reasons:
- 49 x off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract
- 45 x omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract
- 17 x context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it
- 3 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomic)
- 2 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: enriched)
- 2 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomics)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: multi omics)
- 1 x omics_only: 'jasmonate pathway' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptome, enriched)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: enrichment analysis, rna seq, genome wide)
- 1 x observed organism (Pseudomonas syringae) does not match the requested organism (Arabidopsis thaliana)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: bioinformatics)
- 1 x observed organism (Homo sapiens) does not match the requested organism (Arabidopsis thaliana)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: up regulated, rna sequencing, sequencing)
- 1 x off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title
- 1 x context_only: 'jasmonate biosynthetic pathway, jasmonate biosynthetic' appears in the title/abstract but no strong reaction/enzyme evidence is local to it
- 1 x omics_only: 'jasmonate biosynthesis, jasmonate pathway' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: up regulated)
- 1 x decisive negative evidence: software_only -- not a mechanistic pathway paper
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomic, proteomic, metabolomics)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomic, transcriptome, differentially expressed, up regulated)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptome, upregulated, enriched)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: up regulated)
- 1 x decisive negative evidence: epidemiology_survey -- not a mechanistic pathway paper
- 1 x omics_only: 'jasmonate biosynthetic' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomic, gene family, enriched, sequencing)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: upregulated, rna seq)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: genome wide)
- 1 x context_only: 'jasmonate biosynthesis, jasmonate pathway' appears in the title/abstract but no strong reaction/enzyme evidence is local to it
- 1 x omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title

## Claims and decisions

- Claims extracted: **11** across **11** distinct reaction identities
- Passages retrieved: **80**

Latest decision per reaction identity:

- `accepted`: 1
- `candidate`: 9
- `hypothesis_only`: 1

## Research graph

- Primary graph: **1** reactions, **11** chemical connections, **20** entities
- Candidate subgraphs: **3** components holding **10** reactions
- Reactions supported by more than one paper: **0**
- Primary component: `cmp_98656adf98c1` - selected as the primary component: evidence-weighted topic relevance 1.0000 over 1 reaction(s) and 1 distinct paper(s); 0 other admitted component(s) retained as candidate subgraphs (quarantined components were never eligible)

Species state of every reaction node:
- `in_vitro_or_cell_free`: 9
- `species_not_stated`: 1
- `target_species_supported`: 1 (admitted to the target-species graph)

### Admitted reactions

| reaction | status | round | papers | enzyme | compartment |
|---|---|---|---|---|---|
| α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) | `accepted` | 1 | 1 | 13-lipoxygenase (13-LOX) | - |

## Conflicts

No conflicting evidence was recorded.

## Growth by round

| round | added | candidates | promoted | updated | conflicts | useful |
|---|---|---|---|---|---|---|
| 1 | 1 | 10 | 0 | 5 | 0 | yes |
| 2 | 0 | 0 | 0 | 0 | 0 | no |
| 3 | 0 | 0 | 0 | 0 | 0 | no |

## Cost and usage

- Model calls: **15**
- Input tokens: **302,829**; output tokens: **95,417**; cached: **0**
- Estimated cost: **$0.058908** (limit $5)
- Elapsed: **2039.2s**

> Estimated from configured per-million-token rates. This is NOT a billed amount and does not reflect provider discounts, cached-token pricing or minimum charges.

### Model-call failures

These calls produced no admissible claim. None of them added a reaction.

- `mc_001_extract_claims_0002` round 1: `empty_completion` (finish_reason `length`) empty completion (finish_reason=length)
- `mc_001_extract_claims_0003` round 1: `empty_completion` (finish_reason `length`) empty completion (finish_reason=length)

## Retrieval

- Papers indexed: 11; chunks: 112
- Embedding mode: `api`

## Frontier left open at the stop

These questions were generated but never asked. They are where a longer run would go next.

- `seed_expansion` -> 12-OPDA (user-supplied seed metabolite)
- `seed_expansion` -> AOS (user-supplied seed enzyme)
- `seed_expansion` -> jasmonate biosynthesis (run topic)
- `branch_reaction` -> OPC-8 (newly discovered metabolite; look for other reactions it participates in)
- `branch_reaction` -> α-linolenic acid (newly discovered metabolite; look for other reactions it participates in)
- `branch_reaction` -> α-linolenic acid (LeA) (newly discovered metabolite; look for other reactions it participates in)
- `enzyme_reactions` -> 13-lipoxygenase (13-LOX) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `enzyme_reactions` -> 13-lipoxygenase (LOX2) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `enzyme_reactions` -> 13S-lipoxygenase (13S-LOX) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `enzyme_reactions` -> acyl-CoA oxidase (ACX), multifunctional protein (MFP), 3-ketoacyl-CoA thiolase (KAT) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `enzyme_reactions` -> allene oxide cyclase (AOC) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `enzyme_reactions` -> allene oxide synthase (AOS) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `enzyme_reactions` -> jasmonate resistant 1 (JAR1) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `enzyme_reactions` -> OPDA reductase 3 (OPR3) (newly discovered enzyme; look for other reactions it catalyses (a shared enzyme is a search term, never a chemical connection))
- `missing_direction` -> α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) (direction and reversibility are unsupported; retrieve evidence for them)
- `missing_direction` -> α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) (direction and reversibility are unsupported; retrieve evidence for them)
- `missing_direction` -> α-linolenic acid -> 13-hydroperoxy-α-linolenic acid (direction and reversibility are unsupported; retrieve evidence for them)
- `missing_compartment` -> α-linolenic acid (LeA) -> 13-S-hydroperoxyoctadeca-9,11,15-trienoic acid (13-HPOT) (no compartment evidence; retrieve subcellular localization)
- `missing_compartment` -> α-linolenic acid -> (13S)-hydroperoxy-octadecatrienoic acid (13-HPOT) (no compartment evidence; retrieve subcellular localization)

