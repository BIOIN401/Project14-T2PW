# 04 Accuracy against gold: is "more output" also "more wrong output"?

Read-only assessment. No `src/` change, no LLM call, no pipeline run. Scores below were computed by me by calling the
repo's own scorers (`t2pw.bench.eval_v2.reactions.score_leg`, `id_correctness`, `bench.semantic._cover`) on existing run
folders. Scripts: `docs/pwml_output/evidence/04/scripts/`; tables: `docs/pwml_output/evidence/04/`; full scratch
`C:/t/ev-oa/accvar/`. Labels: **M** measured by me, **L** from a log or document, **G** guess (including my chemistry
reading of single rows). CIs are Wilson 95%.

## 0. NOT ESTABLISHED

1. **Entity, transport and interaction precision and recall.** No instrument scores them. The evaluator "measures two-sided
   conversions only" and "no precision or recall is claimed for interactions" (L: `docs/evaluation_v2/PAPER_FINDINGS.md` s3,
   s6.5). Gold has no compound/protein/transport/interaction denominator (see 01 s0). I give only proxies: gold-term
   coverage (a recall proxy), verbatim-quote rates, and mapped-identifier correctness.
2. **Reaction precision rests on 3 papers.** Only PMC12096016, PMC12312563, PMC12421875 are certified complete
   (`gold_v2_supported_reactions.json`; PMC12657337 uncertified per D-103). Of my 24 runs, 20 are one paper. Effective n is
   3 papers, all bacterial, all metabolic. Nothing here transfers to signalling papers.
3. **Era.** 19 of the 24 runs are Aug-Sep 2026 (before D-104 and later changes); 5 are 22 Oct. Two runs are web-era
   (e2e4/e2e5), neither of a gold paper. HEAD numbers are not measured.
4. **Whether a flagged wrong identifier is wrong.** `id_correctness` is a name-vs-reference-record test, not truth. I
   hand-adjudicated 36 of 133 flagged units from my chemistry reading (G). No second reviewer.
5. **Precision of rows that quarantine removes, outside the 3 gold papers.** Not measured. The only direct figure is 7 rows
   (s3.1). No released-quarantine run exists.
6. **Homology precision inside the pipeline.** Never run in the pipeline. The 02 positive control (22 genes) is the only
   figure.
7. UniProt/ChEBI truth for mapped ids was not re-fetched (no online access this task); the reference is the committed
   snapshot (`reactions.py:418` reads blobs at HEAD).

## 1. What each instrument counts (L, file:line)

| metric | instrument | what a "match" or "error" is | gold population |
|---|---|---|---|
| Reaction recall | `eval_v2/reactions.py:104 match_reaction` | every *defining* participant of an in-scope gold reaction present on the right side (reversible gold may match backwards); cofactors scored apart | 4 labelled cases, 3 certified complete, 15 in-scope reactions; pinned_v1 has 10 cases, 23 supported reactions in 7 papers |
| Reaction precision | `reactions.py:198 score_leg`, row classes in `:198-300`; table 1 = true-positive rows / rows claimed as paper extraction (precision computed before `:300`); aggregated by `report.py:61 _micro` | a row is TP if it matches an in-scope gold reaction; **off-pathway, summary, partial, name-conflict and unsupported rows are all non-TP**. D-093: an unmatched row is never by itself "invented" | the 3 complete cases only (`report.py` skips cases whose flag is false) |
| Name equality | `eval_v2/matcher.py:55 strict_key`, `:66 candidate_keys`, `:105 matches` | equality of a normalised key; no containment either way; a parenthetical's inside never credits (`:87 name_verdict`, conflict). `menaquinone` does not match `menaquinone-7`, `CoA` does not match `acetyl-CoA` | n/a |
| Entity coverage (recall proxy) | `bench/semantic.py:488 _cover`, vocabulary `:435 _payload_vocabulary`; `goldset.GoldTerm.match` (lenient, pinned-v1 rules, includes containment) | gold enzyme/substrate/product name found anywhere in entity, process or participant names | 10 pinned cases (`expected_enzymes/substrates/products`) |
| Mapped-id correctness | `reactions.py:562 id_correctness`, reference `:430 build_reference_index` | per identifier, the entity NAME must hit a name recorded for that id (PathWhiz table, enrichment cache, or the mapper's own candidate cache). `wrong_identity` = any id contradicted | all entities with ids in all canonical runs |
| Strict/lexical bench | `bench/metrics.py:363 Denominator` | carries its own population and exclusions; "0 of 0" is `None`, not 0 | per acceptance report |

Consequences for reading "false positive" (L/G): the reaction matcher is *exact on normalised names*, so a paraphrase or
a synonym absent from the gold alias list is a miss; and the id test is *name-in-reference-list*, so a correct id whose
reference record lacks the paper's spelling is flagged wrong. Both inflate apparent error (s4).

## 2. Current numbers

### 2.1 Reactions (M, 24 canonical runs, 3 certified-complete papers, 129 unique retained rows, 120 gold slots)

| slice | runs | precision (TP / rows claimed) | recall (gold matched / gold slots) |
|---|---|---|---|
| all | 24 | 114/129 = 88.4% (81.7-92.8) | 110/120 = 91.7% (85.3-95.4) |
| PathWhiz mode (post-quarantine) | 13 | 67/75 = 89.3% (80.3-94.5) | 64/69 = 92.8% |
| research mode (no quarantine applied) | 11 | 47/54 = 87.0% (75.6-93.6) | 46/51 = 90.2% |
| Aug-Sep runs | 19 | 81/95 = 85.3% (76.8-91.0) | 79/87 = 90.8% |
| 22 Oct runs | 5 | 33/34 = 97.1% (85.1-99.5) | 31/33 = 93.9% |
| per paper | PMC12096016 20 runs 95/106 = 89.6%; PMC12421875 2 runs 17/18; PMC12312563 2 runs 2/5 | recall 92/100, 16/18, 2/2 | |

Species on the 129 rows: 123 correct, 6 unknown, 0 wrong (M). Old committed report for comparison (L,
`docs/evaluation_v2/READINESS.md`, older corpus, 26 canonical legs): P 86.0% (117/136), R 96.6% (114/118), F1 91.0%,
unsupported 2.8%, substrate 96.5%, product 97.9%, enzyme 100% of matched. Same order as mine. The Oct-22 gain (97.1% vs
85.3%) is confounded with paper (4 of 5 runs are PMC12421875/PMC12096016) and era; I do not claim an improvement (n=5).

Recall over all 7 gold-case papers with any reaction gold (M, from 01): Stage 1 88.4% (220/249 slots in 79 runs), final
88.4%; the union over repeated runs reaches 23/23.

### 2.2 Entities (recall proxy only, M, canonical runs of 9 pinned papers, 94 runs)

Gold term found anywhere in the payload vocabulary, final payload (stage 1 in brackets): enzymes 273/287 = 95.1% (100%),
substrates 212/243 = 87.2% (96.3%), products 204/220 = 92.7% (96.4%). So the S1-to-final loss on gold entities is 14
enzymes, 22 substrates, 8 products, concentrated in PMC12096016 substrates (98 to 59 of 80) and PMC12444477 substrates
(9 of 16). Cause is closure/quarantine (see 01 s2.5, 3 #4). Precision of entities: not measured.

### 2.3 Transports and interactions

No gold, no scorer (s0.1). Proxies (L/M from 01 and eval_v2): a verbatim evidence quote supports 77.3% of retained
reactions (canonical, READINESS), 71% of reactions and 66% of interactions *added after Stage 1* (01 s2.4, 116 runs).
Transports 95% of Stage-1 rows survive to final (01 s2.5). A quote is not correctness.

### 2.4 Mapped identifiers (M, 152 canonical runs, 2,297 compound/protein/complex rows)

| verdict | rows |
|---|---|
| correct identity | 1,405 |
| unresolved or novel | 657 |
| `wrong_identity` flag | 192 (133 unique paper x name units, 39 papers) |
| no reference | 43 |

Flag rate 192/1,597 evaluable = 12.0%. By bucket: compounds 151, proteins 33, complexes 8. By mapping rule (M,
`evidence/04/id_by_rule.csv`): compounds with an id are all labelled `legacy_pathwhiz_id_unverified` (805 rows, 151 flagged =
18.8%, CI 16.3-21.7); proteins by `pathbank_protein_id` 19/545 flagged = 3.5% (2.2-5.4); `api_uniprot_fallback` 5/16 = 31%
(14-56); `direct_id_match:uniprot_id` 7/15; `top_unique_alias_candidate` 2/3; `top_unique_candidate` 0/84. Small cells.
Older corpus (L, READINESS): 115 legs, 1,184 correct, 119 flagged, hand precision of the flag 5/21.

## 3. Where the wrong rows come from

### 3.1 Reactions: the 15 non-TP rows of 129 (M classification, G reading)

| class | rows | what they are | wrong? |
|---|---|---|---|
| `paper_stated_off_pathway` | 4 | LDH assay (3), Fe-enterobactin complex | true statements, outside scope. Not errors (D-093) |
| `paper_stated_summary` | 1 | chorismate to DHB-intermediate plus pyruvate, multi-step | true |
| `unsupported` | 6 | MenD ThDP intermediates I/II (3), apo-EntB to holo-EntB (2), DHB + ATP to **AMP** by EntE (1) | 5 are paper-stated protein-bound or covalent states that gold deliberately excludes (D-091 rule); 1 is wrong chemistry (AMP should be DHB-AMP + PPi) |
| `partial_match` | 4 | DHNA + DMAPP to DMK (gold: "prenyl diphosphate"); enterobactin from DHB+ATP without L-serine; from L-serine only; from DHB-pantetheinyl-EntB | 1 is a narrowing of a class term, 3 are lumped or incomplete rows. Real, minor |

So apparent false positives are 15/129 = 11.6%, but at most 4 to 5 rows (3.1% to 3.9%) look like real errors and 10 to 11
are scope or matcher artefacts (G, n=15 rows from 3 papers). Name conflicts: 0. Duplicates collapsed: not counted (unique rows).

Which stage creates them (M, same 24 runs; `evidence/04/stage_precision.json`):

| stage | PathWhiz rows / TP / precision / recall | research rows / TP / precision / recall |
|---|---|---|
| Stage 1 pool | 78 / 71 / 91.0% / 94.2% | 45 / 44 / 97.8% / 86.3% |
| after Stage-2 merge | 82 / 74 / 90.2% / 95.7% | 54 / 47 / 87.0% / 90.2% |
| final (after quarantine for PathWhiz) | 75 / 67 / 89.3% / 92.8% | 54 / 47 / 87.0% / 90.2% |

- **Stage 2 buys recall with precision.** The 13 rows added after Stage 1 on these papers: 6 TP, 4 unsupported, 1 partial, 1
  summary, 1 off-pathway = 46% TP (CI 23-71) against 93.5% (115/123) for Stage-1 rows. All 4 "unsupported" are the gold-excluded
  protein-bound states above, so none is clearly invented, and 3 of them come from one paper. Small n.
- **Quarantine removed no false rows here.** PathWhiz M to F dropped 7 rows on the 13 PathWhiz runs, and all 7 were TP
  (7/7, CI 65-100); recall fell 95.7% to 92.8%. Research mode (nothing dropped) has precision 87.0% vs 89.3%: a 2.3-point gap
  with overlapping CIs and different draws.

### 3.2 Identifiers: the 192 `wrong_identity` flags (G, 36-unit hand sample, seed 20261110, `evidence/04/wr_sample.txt`)

| class | units | share (CI) | examples |
|---|---|---|---|
| **matcher/reference artefact** (correct entity, id correct, spelling not in reference list) | 19 | 52.8% (37-68) | methionine vs "L-Methionine"; laurate vs "Dodecanoic acid"; phenylalanine; tryptophan; HMG-CoA; 11-deoxycortisol vs cortexolone; UDP-GlcNAc; Kir2.1; S1P lyase vs SGPL1; DHBP vs 1-deoxy-L-glycero-tetrulose 4-phosphate |
| **real narrowing** (class or complex mapped to one member/form) | 9 | 25% | anthranilate synthase to component II; SCRIB-VANGL2 complex to VANGL2; sphingosine kinase to SPHK1; "enterobactin synthase" to EntE; Dishevelled to DVL1; holo-EntB to EntB; cardiolipin to one species; DMK to DMK-8 |
| **real wrong molecule or protein** | 8 | 22% | menaquinone to menadione; MPT to 5-Methyl-THMPT; hexadecenal to phytal; PMEA to adefovir dipivoxil; sphingomyelin synthase to SAMD8; S1P phosphatase to HM13; "intermediate I/II" placeholders given real compounds |

By bucket in the sample: compounds 16/24 artefacts (67%), proteins and complexes 3/12 artefacts (25%). So compounds are
mostly name-list artefacts and proteins are mostly real narrowings.
Implied real wrong-or-narrowed rate among evaluable mapped rows: 192 x 47% = about 91 of 1,597 = 5.7% (sample CI 32-63% of
flags gives 3.8-7.6%). This agrees with the earlier tiny samples: 4 of 12 ambiguous-status compounds wrong (33%, CI 14-61), 1
of 30 baseline compounds wrong (3%, CI 0.6-17), 5 of 21 flags confirmed in the older corpus (24%).
**Shared accession is not identity** (L, DUPLICATION-ASSESSMENT.md; 02 s3c): of 16 shared-accession groups in 152 runs about 9
are true aliases, 1 a gene/protein pair and about 6 wrong or narrowing (hordatine A and C, "compound 1/2", holo/apo-EntB,
"enterobactin synthase" with EntE, DAAM with DAAM1). Do not use a shared accession as proof of a merge or of correctness.

### 3.3 Other sources (L)

- **Wrong species:** 0 of 123 known in the 24-run set (M); the older 115-leg corpus was 100% of known (L). Not a measured problem.
- **Over-merging:** DUPLICATION-ASSESSMENT found 0 same-name duplicate entities and 25 records sharing an accession in 19 runs; it
  advises against merging on accession. Not a precision problem on the evidence.
- **Gold defects (L):** `docs/evaluation_v2/PAPER_FINDINGS.md` s5.6: the curated araC record has a sign disagreement, an edge for a
  superseded model and CAP typed as a nucleic acid. F-150 (`FINDINGS.md:6324`, class `gold_data_defect`): a missing spelling
  makes a false accession score zero. PMC12657337 omitted a stated conversion (B-1), then lost certification (D-103). Gold is
  not ground truth; both directions occur.
- **Real errors the gold cannot see (L):** F-096, 7 false real identifiers emitted on PASS legs (cofactor as protein, regulator
  as metabolite, heading as protein). The E2E-2 hand audit of PMC7059786 (1 run, no gold) found 1 false anchor and 1 real
  loss; its 3 of 6 transports are one event drawn three times.

## 4. Share of apparent false positives that are artefacts (M counts, G judgement)

| population | apparent FP | artefact | real | basis |
|---|---|---|---|---|
| reactions, 24 runs | 15 rows (11.6%) | 10-11 (67-73%) | 4-5 (3.1-3.9% of all rows) | s3.1 |
| mapped-id flags, 36-unit sample | 36 | 19 (53%) | 17 (47%) | s3.2 |
| entities, transports, interactions | unknown | unknown | unknown | no instrument |

Both runs of counting say the same thing: raw "FP" overstates real error by about 2-3 times, and the real part is dominated by
**narrowing** (a class, complex or modified form given the id of one member) rather than by random wrong molecules.

## 5. What MORE output would risk (precision cost per added row class)

Every figure is a proxy from the measurements above, not a trial of the feature. Cost = extra wrong rows per added row.

| added class | size at stake (01/02) | evidence on correctness | estimated cost | label |
|---|---|---|---|---|
| **Release rows quarantined for `protein_missing_external_identity`** (101 processes P1+P0; 97 in research) | 12% of text-record processes | rows themselves: 7/7 dropped rows were TP on gold papers; research vs PathWhiz precision 87.0 vs 89.3. The risk is the *entity* (no id), not the reaction | reaction precision about -2 points (CI wide, +-8). Identity-less proteins stay unmapped, which D-106 forbids shipping | M (n=7 rows) |
| **Mapped via fallback lookup** (name/alias/UniProt API when DB fails) | 51 of 197 failed protein units recoverable by name (02) | `api_uniprot_fallback` 5/16 flagged (31%), `top_unique_alias_candidate` 2/3, `direct_id_match` 7/15; DB-exact `pathbank_protein_id` 19/545 (3.5%) | 3.5% best case, about 20-30% for fallback rules after ~75% of protein flags prove real | M (tiny n), G (adjudication) |
| **External compound name lookup for `novel`** (147 unmapped units, up to 58 recoverable, 02) | `legacy_pathwhiz_id_unverified` compounds: 18.8% flagged, real about 33% of flags (8/24) | baseline real wrong 3-10%. External fuzzy ChEBI/KEGG names are not expected to beat the PathBank-exact baseline (G) | 5-15% wrong or narrowed ids on added compounds (G). Narrowing dominates, so exact-label and neutral-form rules matter more than the source | M/G |
| **Homology / ortholog accessions** | 0 to about 10 of 197 units by 02; confident tier 0 of 93 tested | positive control: same gene 21/22, exact accession 17/22, one wrong gene; paralogs reach 84% identity | 5-25%+ of added ids wrong or paralog (G); would put a *different* protein's accession on a paper-named protein (F-096 pattern) | L/G |
| **Second-pass (Stage-2) additions** | +143 reactions, +281 interactions after Stage 1 (01) | 6 TP of 13 on gold papers (46%), none clearly invented; 71%/66% carry a verbatim quote | precision of added rows about 46% vs 93% for Stage-1 rows on gold papers; 4 of 7 misses are gold-scope exclusions | M (n=13) |
| **Unsigned interactions** (122/346 Stage-1 interactions untyped) | export for non-PWML formats only | no scorer; interaction content is the most unstable (assessment 05: Jaccard 0.25 between runs) | endpoints carry the same id risk as above; sign cannot be wrong if absent, but 34% of added interactions have no verbatim quote (G: paraphrase or inference) | L/G |
| **Recover quarantined gene/protein symbol clashes, undeclared endpoints** | 37 + 30 interactions (01) | clash class was one run, fixed by ORCH-782 (L); undeclared endpoint = extraction inconsistency | not measurable | L |

Reading: reaction rows (the unit gold covers) are cheap to release, because the dropped rows were true. The expensive
class is anything that attaches an **identifier** to a name by lookup: 5-30% wrong or narrowed, with the worst cases
being exactly the ones that look plausible (SPHK1 for "sphingosine kinase", DVL1 for "Dishevelled"). Adding unmapped
compounds as id-less entities (the present behaviour for compounds, 24% of PWML compounds) costs no precision and no
accession risk.

## 6. WHAT TO DO

| id | proposal | gain bound | no-regression risk |
|---|---|---|---|
| A1 | Treat **narrowing** as the primary id risk: never export a member accession for a class/complex/modified-form name (extend the D-109/F-096 rule to proteins). Measure on the 9 narrowing units in s3.2 | removes about 25% of flags (about 48 rows of 192) | Low: it can only downgrade ids to class-level or id-less; needs a D-entry (touches mapping) |
| A2 | Fix the evaluator before using it as the success metric: add the L-/D-, `acid`/`ate` and short-name aliases that produced 19 of 36 artefact flags; report `wrong_identity` split into artefact/narrowing/wrong | makes the flag precision about 47% to 70+%; no product change | None (bench code only; needs product-owner decision per D-087, `bench/` gold edits) |
| A3 | Extend gold before claiming improvement on anything but reactions: certify 3 more papers (a signalling paper, a plant paper, a eukaryotic metabolic paper) and add entity/transport/interaction gold rows | gives the first non-reaction denominators | None; costs hand time |
| A4 | Keep reaction rows from quarantine when the only reason is identity of an *enzyme* and the row is verbatim-supported; ship them as `review_required` text-record rows, not PWML | up to 7 of 75 rows (9%) on the gold set; no precision loss measured (7/7 TP) | Medium: D-106 forbids shipping unmapped proteins in PWML, so this is a text-record/other-format decision (QUESTION for owner) |
| A5 | For any new lookup class, gate on a measured wrong-id rate on a pre-registered sample of at least 100 added ids, per rule label, using `id_by_rule.csv`'s cell structure | stops 5-30% wrong ids shipping | None |

Questions for the owner: (1) Is a narrowed accession (class given one member's id) acceptable as a flagged suggestion or must
it be id-less (D-106, D-109)? (2) May the evaluator's wrong-identity flag be changed (bench edit, D-087)? (3) Is a reaction row
whose enzyme lacks an id publishable in non-PathWhiz formats?
