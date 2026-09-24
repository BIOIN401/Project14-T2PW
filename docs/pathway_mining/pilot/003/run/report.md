# Pathway mining run `pilot-003-jasmonate-ath`

> **This run stopped at a configured limit rather than finishing its work.** Stop reason: `max_new_reactions_per_round_reached` - new reactions in round 1: 35 >= limit 25.
> The graph below is what the budget bought, not a complete answer.

- **Target species:** Arabidopsis thaliana (normalized `arabidopsis thaliana`)
- **Topic:** jasmonate biosynthesis
- **Rounds completed:** 1
- **Stop reason:** `max_new_reactions_per_round_reached`

> **This run produced a research graph, not PWML.** The PWML exporter was not invoked, no PathWhiz identifier was required, and no PWML file was written.

> External biochemical database validation (PathBank, PWC, HMDB, KEGG, ChEBI, PubChem, UniProt) is DEFERRED in version 1. Every entity and reaction carries status 'not_evaluated'. Admission was on paper evidence alone; no identifier in this run has been checked against any external database.

## Literature

| papers | count |
|---|---|
| discovered | 64 |
| screened | 64 |
| included | 4 |
| excluded | 60 |
| fully processed | 4 |

Evidence type of included papers: `unknown` x4

Exclusion reasons:
- 22 x omics_only: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract
- 19 x off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title/abstract
- 7 x context_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it
- 2 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomic)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: multi omics)
- 1 x omics_only: 'jasmonate pathway' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptome, enriched)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: enriched)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: enrichment analysis, rna seq, genome wide)
- 1 x observed organism (Pseudomonas syringae) does not match the requested organism (Arabidopsis thaliana)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: bioinformatics)
- 1 x observed organism (Homo sapiens) does not match the requested organism (Arabidopsis thaliana)
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: up regulated, rna sequencing, sequencing)
- 1 x off_topic: no mechanistic term for 'jasmonate biosynthesis' in the title
- 1 x omics_only: 'jasmonate biosynthesis' appears in the title/abstract but no strong reaction/enzyme evidence is local to it (it sits in screening context: transcriptomics)

## Claims and decisions

- Claims extracted: **37** across **37** distinct reaction identities
- Passages retrieved: **73**

Latest decision per reaction identity:

- `candidate`: 34
- `incomplete`: 2
- `hypothesis_only`: 1

## Research graph

- Primary graph: **0** reactions, **26** chemical connections, **67** entities
- Candidate subgraphs: **21** components holding **35** reactions
- Reactions supported by more than one paper: **0**

Species state of every reaction node:
- `in_vitro_or_cell_free`: 15
- `inferred_only`: 17
- `multiple_species`: 2
- `species_not_stated`: 1

## Conflicts

No conflicting evidence was recorded.

## Growth by round

| round | added | candidates | promoted | updated | conflicts | useful |
|---|---|---|---|---|---|---|
| 1 | 0 | 35 | 0 | 1 | 0 | yes |

## Cost and usage

- Model calls: **16**
- Input tokens: **104,173**; output tokens: **83,802**; cached: **0**
- Estimated cost: **$0.035558** (limit $5)
- Elapsed: **1115.3s**

> Estimated from configured per-million-token rates. This is NOT a billed amount and does not reflect provider discounts, cached-token pricing or minimum charges.

## Retrieval

- Papers indexed: 4; chunks: 76
- Embedding mode: `api`

## Frontier left open at the stop

These questions were generated but never asked. They are where a longer run would go next.

- `bridge_search` -> ALA (18:3n3) | jasmonate biosynthesis (candidate component cmp_a8c6dac117f6 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `seed_expansion` -> 12-OPDA (user-supplied seed metabolite)
- `seed_expansion` -> AOS (user-supplied seed enzyme)
- `seed_expansion` -> jasmonate biosynthesis (run topic)
- `bridge_search` -> jasmonoyl-isoleucine (JA-Ile) | jasmonate biosynthesis (candidate component cmp_9abd34f16007 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> 13-S-hydroperoxyoctadeca-9, 11, 15-trienoic acid (13-HPOT) | jasmonate biosynthesis (candidate component cmp_bc76b9cc43fd is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> allene oxide | jasmonate biosynthesis (candidate component cmp_5c1f4c272b39 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> dn-cis-OPDA | jasmonate biosynthesis (candidate component cmp_35d0b739cdec is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> JA-Ile | jasmonate biosynthesis (candidate component cmp_dea944368da1 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> LeA | jasmonate biosynthesis (candidate component cmp_8117f67d1867 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> 12-hydroxy jasmonoyl-isoleucine | jasmonate biosynthesis (candidate component cmp_dd8384a54e53 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> 12-hydroxy-JA conjugate | jasmonate biosynthesis (candidate component cmp_cfd9264eb103 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> 16:0 | jasmonate biosynthesis (candidate component cmp_db257e922c58 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> 16:2(n6) | jasmonate biosynthesis (candidate component cmp_00119fa52993 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> C18-allene oxide | jasmonate biosynthesis (candidate component cmp_7e72499580bf is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> C20-allene oxide derived from 12-HPETE | jasmonate biosynthesis (candidate component cmp_d6aa0a82726e is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> dn-OPDA-Gln | jasmonate biosynthesis (candidate component cmp_14e7f709c875 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> EPA (20:5n3) | jasmonate biosynthesis (candidate component cmp_af4bce6bf0a5 is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> HDA (16:2n6) | jasmonate biosynthesis (candidate component cmp_e4fbeae15dea is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> OPDA | jasmonate biosynthesis (candidate component cmp_f63e6cc91e7b is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> α-linolenic acid | jasmonate biosynthesis (candidate component cmp_c312598eda0c is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `bridge_search` -> Δ4-dn-cis-OPDA | jasmonate biosynthesis (candidate component cmp_5782c1e049fd is paper-supported but not chemically connected to the primary graph; look for a biochemical step linking them)
- `downstream_expansion` -> (9S,13S)-dn-cis-OPDA (newly supported product of allene oxide -> (9S,13S)-dn-cis-OPDA with no known consuming reaction; expand downstream)
- `downstream_expansion` -> (9S,13S)-OPDA (newly supported product of allene oxide -> (9S,13S)-OPDA with no known consuming reaction; expand downstream)
- `downstream_expansion` -> 11-oxo-prostatrienoic acid (11-OPTA) (newly supported product of C20-allene oxide derived from 12-HPETE -> 11-oxo-prostatrienoic acid (11-OPTA) with no known consuming reaction; expand downstream)
- `downstream_expansion` -> 12-hydroxy jasmonoyl-isoleucine (newly supported product of 12-hydroxy jasmonate -> 12-hydroxy jasmonoyl-isoleucine with no known consuming reaction; expand downstream)
- `downstream_expansion` -> 12-hydroxy-JA conjugate (newly supported product of 12-hydroxy-JA -> 12-hydroxy-JA conjugate with no known consuming reaction; expand downstream)
- `downstream_expansion` -> 12-OH-JA-Ile (newly supported product of 12-OH-JA + Ile + ATP -> 12-OH-JA-Ile with no known consuming reaction; expand downstream)
- `downstream_expansion` -> 12-oxo-phytodienoic acid (12-OPDA) (newly supported product of α-linolenic acid (LeA) -> 13(S)-hydroperoxy-9(Z),11(E),15(Z)-octadecatrienoic acid (13-HPOT) + 12-oxo-phytodienoic acid (12-OPDA) with no known consuming reaction; expand downstream)
- `downstream_expansion` -> 13(S)-hydroperoxy-9(Z),11(E),15(Z)-octadecatrienoic acid (13-HPOT) (newly supported product of α-linolenic acid (LeA) -> 13(S)-hydroperoxy-9(Z),11(E),15(Z)-octadecatrienoic acid (13-HPOT) + 12-oxo-phytodienoic acid (12-OPDA) with no known consuming reaction; expand downstream)

