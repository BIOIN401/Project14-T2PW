# 02 Mapping failures and sequence homology

Read-only assessment. Nothing in `src/` changed; no pipeline run, no LLM call, no test run. Scripts and tables:
`C:/t/ev-oa/mapping/` (full) and `docs/pwml_output/evidence/02/` (small copy). Claim labels: M measured by me,
L log or doc evidence, G guess.

## NOT ESTABLISHED

1. **Era.** 150 of the 152 canonical runs are dated before 2026-10-28; only e2e4/e2e5 (2026-11-09/10) postdate D-117 to
   D-120 (M, `run_dates.json`). Family-term detection (D-117, D-119 s1), alias identity (D-119 s2), the ORCH-793
   strain-duplicate collapse and tRNA-as-nucleic-acid (D-120) are in HEAD but not in those runs. Every "still lost"
   figure below is therefore an upper bound for HEAD. I did not re-run the pipeline (not allowed), so I cannot say how
   many of the 51 name-recoverable units HEAD already recovers. Only one check was possible: HEAD's lexical
   `is_family_or_complex_term` (`map_ids.py:10554`) flags 10 of the 197 failed protein units (M).
2. **Skew.** 5 papers supply 82 of 152 runs (PMC12096016 20, PMC12452463 19, PMC12782028 16, PMC12856317 16,
   PMC12180156 11). I therefore count at the unit level (paper, normalized name), 53 paper ids, not at run level.
3. **No gold for the unmapped entities.** "Recovered" means a unique plausible record by public lookup, not verified
   truth. The only truth anchor is a 22-gene positive control (section 4).
4. **Entities removed before `final_mapped.json`.** In 64 runs the closure drops entities and their
   mapping metadata with them. 29 protein units are known only from the quarantine report (name, no cause field).
5. **Function-seeded homology** (EC, Rhea or KEGG reaction to the organism's proteome), **Ensembl compara** and
   **reciprocal best hit by BLAST** were not run. My homology test is name-seeded, UniProt-only, with a local
   alignment (simple scoring, so identity values are indicative, +-3 points, and it finds no alignment below about 30%,
   e.g. human POR vs *E. coli* CysJ). The ceiling in section 4 is a guess (G).
6. Whether a given ChEBI/KEGG name hit is the molecule the paper means was not checked by a chemist. Only ID-to-name
   agreement was checked (section 5).
7. PathBank MySQL was not read. Whether `data/pathwhiz_id_db.json` or the live DB already holds records for the
   unmapped names is a gap; a read-only DB route would close it.

## 1. How mapping works (L, file:line)

Ladder per entity, `src/t2pw/mapping/map_ids.py`:

- Compound: `map_compound_row` (1832): PathBank id, external id, exact name, synonym, fuzzy (`map_compound`, 2760,
  SQL `LIKE` on the PathBank `compounds` table), else `novel_compound` (1887-1894). **When the DB answer is `novel`
  or `ambiguous`, `_map_compound_with_strategy` (7690) returns at 7732 and the external-name API fallback
  `map_compound_all` (5766: ChEBI, KEGG, HMDB name search) is never called.** The API runs only when the DB is
  unavailable or `id_source=api`. Also called from `curation/gap_resolver.py:1829,1868,2423`.
- ChEBI name search `_query_chebi` (5520) targets `https://www.ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity`.
  `enrich_entities.py` header records HTTP 500 on that service (all 230 cached ChEBI entries were errors). **I
  re-tested it on 2026-11-10: still HTTP 500 (M).** The current ChEBI API (`/chebi/backend/api/public/`) and OLS4 work.
- Protein: PathBank DB first (`pathbank_protein_id`, 545 of 744 non-sentinel protein rows = 73%), then UniProt API
  (`map_protein_uniprot`, 5214; query `(protein_name:"X" OR gene:"X") AND organism_name:"..."`, then an unscoped
  query), then the identity verifier: name gate (`_apply_name_plausibility_gate`, 7156), species check (6877), and a
  **0.1 score margin over rivals (`_REAL_PROTEIN_MIN_MARGIN`, 6392; reject at 6984-6986 as
  `ambiguous_insufficient_margin`)**.
- Failure ends as `novel_protein` (unmapped) or, for an enzyme actor, the **Unknown-backed placeholder complex**
  (D-070; `_PATHBANK_UNKNOWN_FALLBACK_RULE` 9222; failure reason stored by `_placeholder_failure_reason` 9568 as
  `real_mapping_failure_reason`, written at 9892 and 10121).
- Gate: a protein without UniProt or DrugBank id is quarantined as `protein_missing_external_identity`
  (`pipeline/strict_quarantine.py:765`, `pipeline/entity_identity.py:84-110`). `_entity_representability` has no
  identity check for compounds, so an id-less compound is not quarantined by that module (the PWML-side check was
  not read).

## 2. Corpus and headline counts (152 canonical runs, 53 paper ids)

Rows in `final_mapped.json` (M): 744 protein rows (excl. 39 `Unknown` sentinel rows), 415 complex rows, 1095 compound
rows, 209 species rows, 39 nucleic-acid rows.

| Unit (paper, name), canonical | n | mapped | placeholder | unmapped |
|---|---|---|---|---|
| protein or complex | 464 | 267 (57.5%) | 99 (21.3%) | 98 (21.1%) |
| compound | 517 | 370 (71.6%) | 0 | 147 (28.4%) |

- Unmapped protein units include 29 known only from the quarantine report (cause metadata lost).
- 106 of 152 runs (70%) carry at least one id-less compound; 39 runs (26%) at least one placeholder complex; 43 of 53
  papers have at least one failed protein unit and 26 of 53 papers fail at least half of their protein units.
- 71 further protein units are mapped, but their `X complex` wrapper is unresolved (43 `component_proteins_unresolved`,
  11 placeholder, 8 `species_not_found`, 6 ambiguous name, 3 refused). Not investigated (G: wrapper artifact).
- Quarantine: 294 quarantined process rows in 52 of 152 runs: `protein_missing_external_identity` 196 (67%),
  `ambiguous_entity_type` 54, `undeclared_entity_*` 41, other 3. 98 unique (run, protein) pairs in 44 runs. Closure then
  removed 304 entities in 64 runs (179 unreferenced compounds or nucleic acids, 125 degree-zero proteins or complexes);
  those removals are consequences of quarantine, not mapping failures.
- Organism not in the PathBank species table: 13 of 53 papers (status `novel`, `no_db_match`), among them *L. lactis*
  KF147, *S. aureus* N315, *Paris polyphylla*, *Catharanthus*, *Petunia*, *Moringa*, *Rhizopus*. In 5 papers, 15 wrapper
  complexes fail as `species_not_found`.

**By organism (M).** Protein-unit failure rate: model organisms (human, mouse, *E. coli*, *Arabidopsis*, *B. subtilis*)
101/334 = 30.2%; all others 96/130 = 73.8%. 15 papers fail every protein unit (n>=3). Compound failure is NOT organism
driven: 71/268 = 26.5% model vs 76/249 = 30.5% other.

## 3. Cause of failure

### 3a. Proteins and enzymes (197 failed units: 98 unmapped + 99 placeholder)

Cause by name shape (my classifier plus 17 manual overrides; M):

| Name shape | units | Example |
|---|---|---|
| single gene symbol or paper-coined enzyme name | 147 (75%) | LapB, LONP1, EasF, UGT703R3, SdOMT |
| family or class term | 18 | P2X receptor, ceramidases, Frizzled |
| multi-entity complex or system | 16 | SPT/ORMDL, Lpt system |
| descriptive enzyme name | 8 | glucosylceramide synthase |
| modified form | 6 | holo-Fur, apo-EntB, phosphorylated c-Jun |
| RNA, not a protein | 2 | RyhB, GcvB |

Cause as recorded by the pipeline (placeholder complexes, 133 rows, `real_mapping_failure_reason`; unmapped rows,
`failed_check`):

| Recorded reason | rows | What it means |
|---|---|---|
| `species_mismatch` | 54 placeholders (44 units, 19 papers) | a candidate was found, but in another species |
| `no_match` | 27 | UniProt returned nothing for the name |
| `implausible_name_match` | 17 | name gate refused the hit (paper name and record share no token) |
| `ambiguous_insufficient_margin` | 16 placeholders, 33 unmapped rows | correct entry plus strain or paralog rivals within 0.1 |
| `identity_evidence_missing` | 11 + 12 | identifier claimed but not verifiable |

Name-variant misses are visible in the judged candidates (L): `LpxL` found as `htrB` P0ACV0, `Kir2.1` as `KCNJ2`
P63252, `glucosylceramide synthase` as `UGCG` Q16739, all refused as `no_shared_meaningful_token`. `FtsH`, `LapB`, `LONP1`,
`PvdA` were found and refused only because K12/O157/SE11 duplicate entries tied (margin 0.00).

Organism effect on cause (M, `org_stats.json`): 24 failed units belong to organisms with under 150 UniProt entries in
total (*Saposhnikovia* 3, *Moringa* 53, *Paris* 99, *Astragalus* 79, KF147 2), so no lookup of any kind can succeed.
26 more belong to species with at most 15 reviewed entries but 1800-16000 unreviewed (*A. japonicus* 11 reviewed and
12042 unreviewed, *S. tubercidicus* 0 and 7016): names fail there, homology to a well-annotated relative might work.

### 3b. Compounds (147 unmapped units, 144 unique names; classifier on name shape, M)

| Name shape | units | Examples |
|---|---|---|
| specific named compound (trivial or systematic) | 94 | SEPHCHC, chanoclavine-I, UDP-3-O-(R-3-hydroxymyristoyl)-... |
| abbreviation (with or without parentheses) | 25 | DHB-AMP, SHCHC, cPMP, DMAPP |
| class or generic term | 12 | oxysterols, glycosphingolipids, prenyl diphosphate, "pyoverdine backbone" |
| ion or formula | 5 | Zn2+, Ca++, FeII, MgCl2 |
| nucleic acid, tRNA conjugate, ACP, protein-bound or protein name filed as compound | 10 | araBAD mRNA, apo-EntB, PvdA |
| misspelling | 1 | lansterol |

The recorded cause is `novel_compound` (no PathBank record) for 216 of 261 unmapped rows and `name_plausibility_gate` for
45. Because of section 1 (line 7732), nothing outside PathBank was ever asked.

### 3c. Mapped but WRONG (distinct from unmapped)

- **Shared accession** (M, 152 runs): 16 unique accession-name groups in 18 runs. By hand: about 9 are true aliases
  (CoA-SH and coenzyme A, serotonin and 5-HT, ChlM, ChlP, S1P lyase, Dishevelled, D-ribulose 5-phosphate, L-serine and
  serine, ferric enterobactin and its "complex"), 1 is a gene/protein pair (HEMA1 and GluTR1), and about 6 are wrong or
  narrowing: holo-EntB onto EntB (P0ADI4), the "enterobactin synthase" complex onto subunits EntE (P10378) and EntF
  (P11454), DAAM onto DAAM1, sphingosine kinase onto SPHK1, hordatine A and C sharing one id (two molecules).
- **Ambiguous-status compounds keep an id** (`legacy_pathwhiz_id_unverified`): 118 rows / 69 units. I checked the 12
  with a ChEBI id against ChEBI names: 4 wrong (MPT is tetrahydromethanopterin; "compound 2", "compound 3" are "Haloaromatic
  compound"; "compound 5" is an unrelated isoxazole), 3 partial, 5 agree. Baseline: 30 random compounds matched by
  PathBank id: 23 exact, 4 partial, 3 no agreement, of which 1 is clearly wrong (UMP carries ChEBI 17277, a UDP-MurNAc
  peptide). n is tiny; this says wrong ids exist, not their rate. Related, L: F-096 (7 false real identifiers), and
  ORCH-789 D3 (a `best_effort_fallback` gave PKC the accession of Par-3, score 0.39).

## 4. Measurement: what better lookup and homology would add

**Sample frame (M).** All 197 failed protein units were enumerated. 93 were tested: all 50 non-gene-symbol units
(census) plus 43 gene-symbol units stratified by organism class (model or not) and state (unmapped or placeholder),
seed 20261110. Population figures reweight the gene-symbol stratum. UniProt cap about 300 requests (used about 305,
cached under `ev-oa/mapping/cache/`); generic client name; no email; names and accessions only.

**Methods.** (a) UniProt `gene_exact:SYMBOL AND taxonomy_id:T` (T from the run's species), then
`protein_name:"..." AND taxonomy_id:T`. "Unique" = one gene group among reviewed hits (strain duplicates collapsed).
(b) Homology: seed = reviewed UniProt entry with the same gene symbol in any organism; target pool = entries in taxon T
whose name equals the seed's recommended name (or same EC); each pool member aligned locally to the seed (semi-global).
Identity = matches over aligned columns, coverage = aligned columns over seed length.

| Outcome, 93 tested units | tested | population estimate of 197 |
|---|---|---|
| A. Name lookup in the paper's taxon returns one record | 17 (15 reviewed) | about 51 (26%) |
| B. No name hit; homology gives a hit at identity >=50%, coverage >=80% | 3 | about 10 (5%) |
| C. Neither: no named seed anywhere | 21 | about 60 (30%) |
| C. Neither: seed exists, organism has no matching record | 8 | about 27 (14%) |
| Family or class term, not one protein | 17 | 17 |
| Multi-entity complex, needs splitting first | 16 | 16 |
| Modified form (base protein only 3, unresolved 3) | 6 | 6 |
| Ambiguous paralogs or family | 3 | about 7 |
| Not a protein (sRNA) | 2 | 2 |

Gene-symbol stratum is the only reweighted one (n=43); expect roughly +-8 units on A and B. The model-organism units
account for nearly all of A: 13 of 63 tested, versus 4 of 30 in other organisms.

**A, in detail.** 11 of the 17 recovered accessions were already the pipeline's own judged candidate and were refused:
8 for margin ties (LapA, LapB, LONP1, ORMDL1, ORMDL2, PvdA, PvdJ, PvdL), 2 by the name gate (Kir2.1, glucosylceramide
synthase), 1 on score (sigma 70). Those are lookup-policy failures, not missing database records. A 2026-10 change
(ORCH-793) already collapses strain duplicates when the paper name is the primary gene symbol of both, so the margin
share may be fixed at HEAD (not established, item 1). The other 6 are plain name or synonym misses (AraC, CRP, TMLD to
TMLHE, PPO1 to PPOX1, SGMS1, IlvC). 3 modified-form hits (holo-Fur, apo-EntB, phosphorylated CapZIP) return the base
protein; exporting that accession for the modified form is the F-096 false-identifier pattern, so I do not count them.

**B, in detail.** SGD (*C. roseus*, 75% identity, 99% coverage to a *Rauvolfia* seed), ChoC (*Rhizopus*, 62%, 93%), EasF
(*A. japonicus*, 54%, 99%). All three are single unreviewed TrEMBL entries with no gene name and no second hit.

**Why B is small.** Of the 29 gene-symbol units not recovered by name (B 3, ambiguous 2), 16 have no
reviewed UniProt entry with that symbol in any organism and 8 have a seed but no matching record in the organism:
paper-coined names (EasF, TubF, SdOMT, UGT703R3, NIT-9G, HS41.21) give no seed; a name seed that exists
can be wrong (SUR1, TGG5 seeded from *Arabidopsis* into *Moringa*: pool empty; `tubC` seed from *Dictyostelium* for a
*Streptomyces* gene). Papers give names, not sequences, so name-seeded homology starts from a guess.

**Positive control (22 genes, M).** Hide a mapped protein, seed from a close species by symbol, search the true taxon
by the seed's name, align. Same gene as top hit: 21 of 22. Exact recorded accession: 17 of 22 (the other 4 are
strain duplicates of the same gene, e.g. *E. coli* `lpxD` P21645 vs P65323 at 95.9% vs 95.6%). One wrong gene: *Arabidopsis*
`FC1` seeded from rice returned `CAD7` at 58% identity. Correct orthologs spanned 58-100% (human-mouse 69-100%,
*E. coli*-*Salmonella* 77-97%, *Arabidopsis*-rice 58-78%).

**Paralogs inside one genome (M, same aligner).** ORMDL1 vs ORMDL3 84.3% identity at 100% coverage; SMURF1 vs SMURF2
73%; KCNJ2 vs KCNJ4 65%; SHMT1 vs SHMT2 64%; SPHK1 vs SPHK2 39%; *E. coli* LpxL vs LpxM 32%; versus human-mouse KCNJ2 98.6%.
OrthoDB search for KCNJ2 returns a Vertebrata group with 3587 genes (paralogs included). UniRef is no shortcut: mouse
Kcnj2 is not in the human seed's UniRef50 cluster (it sits in `UniRef50_Q64273`).

### What a high identity does and does not prove

- Proves: the two proteins share ancestry and fold. Does not prove: the same gene (paralogs reach 84% inside one
  genome), the same function (cross-species transfer; a change at a few residues can move substrate), the same
  isoform, or that a domain-only match (coverage <80%) is the full protein.
- Identity alone separates nothing at 50-80%: true orthologs (58-78%) and the one false hit (58%) overlap with paralogs
  (64-84%).
- A confident tier would need all of: unique best hit with a gap of at least 10 identity points over the second;
  coverage >=90% both ways; name or EC agreement with the seed; reciprocal best hit (not tested). Under that rule the
  tested sample yields **0** (SGD fails on the no-gene-name, unreviewed, single-hit test).

### Bounds on what homology could add (197 failed protein units)

- **Lower: 0 units** (confident tier above, 0 of 93 tested, 0 of 43 in the sampled stratum; rule-of-three upper limit
  about 7% of that stratum, roughly 10 units).
- **Upper, name-seeded, any hit at identity >=50% and coverage >=80%: about 10 units (5%)** (M, estimate). Function-seeded
  homology into annotation-poor proteomes could add part of the 26 units in *Aspergillus*, *Streptomyces*, *Apium*,
  *Penicillium*, *Fusarium*, *Rhizopus* (ceiling about 13% of 197, G; not measured, item 5).
- Name lookup (A) is the larger, cheaper lever: about 51 units (26%), and 11 of 17 sampled were already in hand.

### Locked decisions homology would touch

- **D-106**: "The external-identity requirement for proteins stays exactly as strict as it is." Remedy is "upstream: find
  the identity correctly (organism and strain-aware lookup...)". An inferred id is still a real UniProt id, so the gate
  holds, but the identity verifier would need a new admission rung.
- **D-108** s2: "Never across a subspecies boundary, and never across species." All three B hits are inside the paper's
  own species (its own TrEMBL record), so they are allowed by D-108 only if "same species, other protein name" counts
  as lookup. A congeneric ortholog (the DmaW case, *A. fumigatus* for *A. japonicus*) is forbidden.
- **D-117** s2: "the protein-id requirement is not relaxed (D-106)" and every family member's "species must match the
  pathway species". Same wall for members found by homology.
- **D-109** s2 (compounds): "mapping never narrows a class name to a specific molecule."
- **Open owner question already on file** (L, ORCH-739 and LEDGER): "whether a congeneric orthologue of the right
  function may stand in for an organism whose own proteome is not in UniProt ... biological policy decision for the
  product owner"; trigger to revisit was a second independent refusal. Count now: 44 placeholder units in 19 papers
  fail on `species_mismatch`, 16 rows in *A. japonicus* alone. Whether each is a congeneric ortholog I did not verify (G).

## 5. Compounds: structure-based identity instead of homology (147 unmapped units, all tested; M)

| Route | units resolved | note |
|---|---|---|
| ChEBI exact label or synonym (OLS4, 2 name variants) | 52 (35%) | 15 return several ids (neutral and ionic forms) |
| KEGG `find/compound` exact name | 30 (20%) | 24 also in ChEBI |
| Either | 58 (39%) | |
| ChEBI id that carries an InChIKey | 44 | 8 ChEBI hits are classes or generic (no structure) |
| OPSIN name-to-structure for those with no db hit | 3 of 67 tried | trivial and systematic names only |
| Any InChIKey (ChEBI or OPSIN) | 49 (33%) | |
| Nothing: no db name, no parsed structure | 86 (59%) | |

By shape: ions 5 of 5, class terms 4 of 4 (class-level ChEBI ids exist: oxysterol, glycerophospholipid, glycosphingolipid,
sphingoid), abbreviations 11 of 17, specific named 33 of 94, nucleic-acid, protein-bound and misfiled protein names 0 of 10.
The 86 unresolved are mostly true pathway intermediates with no public record: lipid A and Kdo precursors, chlorophyll
intermediates, ergot and pyoverdine intermediates, CDP-sugar variants.

Cautions: ChEBI exact synonyms land on charged forms ("chorismate" gives `chorismate(2-)`, CHEBI:29748). The class term
"prenyl diphosphate" resolves to CHEBI:16057, the same id that DMAPP resolves to, exactly the narrowing D-109 bans.
"Zn2+" returns both `zinc(2+)` and `zinc atom`. Abbreviations resolve only when ChEBI lists them as synonyms
(SEPHCHC, SHCHC yes; DHB-AMP no).

## 6. VERDICT on sequence homology

**Not recommended as an identity source. Recommended, if at all, only as a flagged suggestion.**

1. It adds at most about 5% of the failed protein units (0 confident, about 10 upper), because papers give names not
   sequences and the failing names are mostly paper-coined, in organisms with few annotated records.
2. A cheaper lever is larger: about 26% of failed units have a unique record by plain name lookup in the paper's own
   taxon, and most were already found and refused.
3. Identity does not discriminate paralogs from cross-species orthologs in the 50-85% range, and a false hit appeared
   in 1 of 22 controls.
4. It conflicts with D-106, D-108 and D-117 unless the owner rules otherwise.

If the owner wants it: output only in the human-facing review list, labelled "inferred by homology, not paper-stated",
never in the PWML id fields, never in LLM-bound text, with identity, coverage, rival gap and source species recorded.

## 7. WHAT TO DO

1. **Re-measure at HEAD first.** Re-run only the 51 name-recoverable units and the 44 `species_mismatch` units through the
   current mapper (bounded wrapper, no heavy LLM) to see what D-117 to D-120 already fixed. Cost: low. Risk: none.
2. **Name-lookup tie-break (no new gate).** For `ambiguous_insufficient_margin` where all tied rivals are curated entries
   of the same gene in the pathway species, pick the reference strain (K12 for *E. coli*). Expected gain: up to about 8 of
   the 51 sampled-A units (G for HEAD). No-regression risk: needs a D-entry (touches `map_ids.py:6984`), and D-108 s3 flag
   applies.
3. **Gene-synonym rung for name-gate refusals** (`LpxL`/`htrB`, `Kir2.1`/`KCNJ2`, `UGCG`): D-119 s2 already covers UniProt
   gene synonyms; verify it also covers PathBank-judged candidates. Gain: 2-3 of 17.
4. **Compounds: call the external name API when PathBank says `novel`** (line 7732), with ChEBI moved to the current
   API (OLS4 or `/chebi/backend/api/public/`). Expected: up to 58 of 147 unmapped compound units (39%, an upper bound
   before charge-form and class-narrowing rules). Risk: false ids (section 3c) and the D-109 narrowing; require exact
   label or synonym match, neutral form preferred, class terms only to class entries, and a name-gate pass.
5. **Do not narrow**: keep class terms ("prenyl diphosphate", "oxysterols") class-level or id-less.
6. **Wrong ids**: audit the 118 `legacy_pathwhiz_id_unverified` compound rows; 4 of the 12 checked were wrong.

## Questions for the owner

- Q1: Is a same-species, different-name TrEMBL record found by homology acceptable as a flagged suggestion (D-106, D-108)?
- Q2: Is a congeneric ortholog ever acceptable for annotation-poor organisms (ORCH-739)? 44 units in 19 papers now meet the
  stated trigger by count.
- Q3: Should compounds keep a PathBank-only identity policy, or may a public ChEBI/KEGG exact-name hit supply the id?


## RECONCILIATION (added by orchestrator-dispatched check)

Scripts: `C:/t/ev-oa/reconcile/recon.py`, `org.py` (run on the same 152 canonical runs, frame rules from `units.py`). All M.

**Why 464/197 != 308/88 (and 572/298).** Both counts are exact under their own rules; they differ only in definition.
Running the frame's own rules on the proteins bucket alone, keeping the 43 `Unknown` sentinel rows as placeholders,
reproduces the verifier exactly: 308 units, 88 fail (53 unmapped + 35 placeholder).

| Source (frame units, 464 total) | units | mapped | placeholder | unmapped | failed |
|---|---|---|---|---|---|
| proteins bucket only | 182 | 137 | 0 | 45 | 45 |
| proteins + same-stem `X complex` wrapper | 91 | 84 | 4 | 3 | 7 |
| complexes only (no protein row of that name) | 162 | 46 | 95 | 21 | 116 |
| quarantine-report-only (not in final_mapped.json) | 29 | 0 | 0 | 29 | 29 |
| total | 464 | 267 | 99 | 98 | 197 |
| `Unknown` sentinel rows (43 rows; excluded from frame) | 35 per-paper units (verifier) | | 35 | | 35 |

The frame merges a protein and its "X complex" wrapper into one unit (name normalisation drops "complex"), judges
the unit mapped if any row is mapped, counts complex wrappers (133 rows are the Unknown-backed placeholder complex),
excludes `Unknown`, and adds 29 quarantine-only units. The verifier counts proteins rows only, keeps `Unknown`
(35 units), and had a cruder complex rule (572/298; the frame rules with separate keys give 561/277). "Enzymes
referenced in reactions/transports" are not a separate source: they appear only as bucket rows or as quarantine rows
(`undeclared_entity_*`, 41 rows), which are not units in either count.

**Best definition for "share of protein/enzyme entities named in the final output lacking a database accession".**
Final-output units only (drop the 29 quarantine-removed ones, which are not in the output), proteins and complexes,
`Unknown` excluded: **168 of 435 units = 38.6%**. The placeholder complex is the output's stand-in for a named
enzyme with no accession, so complexes belong in the denominator. Other definitions: published 197/464 = 42.5%
(includes 29 entities absent from the final output); proteins bucket only with `Unknown` 88/308 = 28.6%; proteins
bucket only, named, no `Unknown` 53/273 = 19.4%. The 42% headline overstates the final output by 4 points.

**Correction to the organism split (section 2).** The 73.8% vs 30.2% numbers reproduce, but they include the 29
quarantine-only units. On final-output units (435): model 82/315 = 26.0%, other 86/120 = 71.7%. The gap lives in the
complex wrappers: among units with a proteins-bucket row (273), failure is 46/233 = 19.7% model vs 6/40 = 15.0% other
(no gap). So say "non-model organisms fail mostly through placeholder complex wrappers (`species_not_found`), not
through protein-level lookup". Use 168/435 = 38.6%, 71.7% and 26.0% in place of 197/464, 73.8% and 30.2%; the
unit counts in section 3a stay valid as the quarantine-inclusive set. Compounds are unaffected (147/517 vs 148/520).
