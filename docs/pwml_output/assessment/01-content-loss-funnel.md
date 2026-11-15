# Assessment 01 - Where content is lost from paper to PWML (funnel)

Branch `pwml/output-assessment` (from `85b5f9d4`). Read-only analysis; no product code changed, no LLM call, no pipeline run.
Claim labels: **M** measured by me from run artifacts, **L** from log/doc evidence, **G** guess.
Tidy per-run table: `docs/pwml_output/evidence/01/tidy_runs.csv` (192 rows). Scripts: `evidence/01/scripts/`. Full scratch: `C:/t/ev-oa/funnel/out/`.

## 0. NOT ESTABLISHED

1. **What the paper states, in absolute terms.** No entity-level gold exists. Gold covers *reactions only*: 7 papers, 23 reactions (3 papers / 15 reactions certified complete; `gold_v2_supported_reactions.json`, `pinned_v1.json`). Compounds, proteins, complexes, transports and interactions have no paper-stated denominator. Section 2.3 gives a reaction-only estimate; everything else is measured from Stage 1 onward.
2. **The current code.** 91 of the 101 PathWhiz-mode canonical runs date from Aug-Sep 2026, before D-104 (Oct 19), D-114 per-row omission (Oct 25), ORCH-749 interaction typing and D-117. Only 10 runs (8 from Oct 21-22 and the 2 web runs e2e4/e2e5; 7 papers; 6 with PWML) are recent. **The funnel describes the pipeline as it ran, not as it is at HEAD `85b5f9d4`.** I could not re-run (no pipeline runs allowed).
3. **Why Stage 1 misses content** (reading/prompt limits versus paper absence) and which of the 143 post-Stage-1 reaction additions are correct biology: only whether their evidence quote appears verbatim in the paper (71%, section 2.4).
4. **Attribution of ~23% of PathWhiz-mode protein rows** (168 of 741 merged rows absent from final_mapped with no entry in `removed_entity_report.json`): 132 are recorded as KI-013 orphan drops by `reaction_evidence`, 6 as KI-015, the rest unattributed (G: alias merges and wrapper rewrites).
5. **Whether the 40 no-PWML runs would write a PWML today** (D-114 was built to prevent it; no post-D-114 corpus of comparable size exists; only e2e4/e2e5 and 2 Oct-22 legs).
6. Process-name matching across stages is by normalized names and participant sets, not identity; "kept" can hide a rewrite and "dropped" can hide a merge. Where I could separate these I did (2.5).

## 1. Corpus and method

| group | definition | runs | papers | notes |
|---|---|---|---|---|
| ALL152 | `quarantine_ok=true` final_mapped.json | 152 | 51 | headline corpus (M) |
| P1 | ALL152, export_mode `pathwhiz`, PWML written | 61 | 37 | 57 `pathway.review_required.pwml` + 4 `pathway.pwml` |
| P0 | ALL152, `pathwhiz`, **no PWML** | 40 | 26 | every one has `RESULT: FAIL` |
| RS | ALL152, export_mode `research` | 51 | 14 | no PWML by design; quarantine is diagnostic (`research_mode.applied=false`) |
| OTHER40 | `quarantine_ok` false/absent | 40 | 12 | older or failed-quarantine runs; reported separately, never in headline |

Sources per run: `stage1_payload.json` (S1), `merged_payload.json` (M, after Stage-2 inference merge + entity-admission gate), `final_mapped.json` (F), `quarantine_report.json`, `removed_entity_report.json`, `graph_closure_iterations.json`, `initial_stage3_gate_report.json` (`normalization_stats`), `pwml_ir*.json`, the PWML itself, and the repo's own D-114 record builder `app/reaction_evidence.py:build_report` (run read-only on all 192 dirs; stdlib only).
**Order of stages matters (M):** in `pathwhiz` mode `final_mapped.json` is *after* strict quarantine (process count F = quarantine "accepted" in 101/101 runs); in `research` mode it is *before* (quarantine reported but not applied, F = accepted + quarantined in 19/51, nothing quarantined in 32/51). So F is the post-drop payload for P1/P0 and the pre-drop payload for RS.
Counting caveat (M): S1 is a pooled union of draws (D-104), so it contains duplicates; "dropped" counts below split those out.

## 2. Numbers

### 2.1 Run-level: whole pathways lost (M, 101 PathWhiz-mode canonical runs)

| outcome | runs | share |
|---|---|---|
| PWML written | 61 | 60% |
| **no PWML** | **40** | **40%** |

The 40: 24 stopped at `post_pipeline` (`final_pre_export_stage3_gates`); across the 40 runs' `gate_fail_report.json` the errors are 119 protein-identity, 14 missing species, 7+7 generated-complex species/identity, 3 gene/protein type overlaps, 8 registry (interaction endpoint) errors; 14 stopped at the PWML required-field gate (species taxonomy/classification 4, `no_defensible_reaction_support` 3, `no_biological_states` 3, `visible_entity_missing_location_state` 3, `reaction_enzyme_must_be_protein_complex` 2); 1 pre-export revalidation; 1 IR validation. By month the no-PWML rate is Aug 13/35, Sep 23/56, Oct 4/8 (M). These 40 runs hold 183 reactions + 32 transports + 61 interactions in F (276 processes, 40% of all F processes in PathWhiz mode) and deliver none. The 40 are all pre-D-114 (L: D-114 AMENDMENT 2 reports 56 of 91 legs wrote a PWML at main on Oct 25, so ~38% still did not).

### 2.2 Entity and process counts by stage

PathWhiz-mode canonical (P1+P0, 101 runs, 37+26 papers; "PWML" is P1 only, 61 runs):

| kind | S1 | M | F (post-quarantine) | PWML (P1) | P1: F |
|---|---|---|---|---|---|
| compounds | 860 | 875 | 719 | 431 | 431 |
| proteins | 737 | 741 | 386 | 241 | 241 |
| protein complexes | 32 | 37 | 390 | 234 | 234 |
| nucleic acids | 39 | 39 | 15 | 12 | 12 |
| reactions | 402 | 500 | 428 | 245 | 245 |
| transports | 75 | 79 | 74 | 42 | 42 |
| interactions | 234 | 558 | 188 | 87 | 127 |

Notes (M): complexes jump 37 to 390 because 239 of 415 F complexes are `generated` single-protein PathWhiz wrappers (`complex_named_source_entity_wrapper` 14 more); they are structure, not paper content. M interactions include 210 `SAME_AS` alias rows (all canonicalised away by `process_normalizer.py:2353`) and 281 added after Stage 1 (no lineage). **P1: PWML equals F for every kind except interactions (127 to 87): the PWML writer itself is lossless except for interactions.**

Research mode (RS, 51 runs): S1/M/F = compounds 368/374/376, proteins 399/402/401, complexes 19/15/25, nucleic 23/25/24, reactions 159/205/193, transports 32/32/32, interactions 112/279/164. Nothing is removed in RS; the diagnostic quarantine would have removed 35 reactions, 4 transports, 74 interactions (113 of 389 F processes, 29%), 97 of them (86%) for `protein_missing_external_identity` (M).
OTHER40 (separate): S1 rx/tr/ia 162/38/125, F 228/42/208; 6 of 40 wrote a PWML (54 rx, 3 tr, 18 ia).
Late-era subset (Oct 20 onward, PathWhiz, 10 runs / 7 papers, 6 PWML): S1 rx/tr/ia 84/31/161, M 98/32/248, F 72/29/63, PWML 31/18/17 on the 6 PWML runs. Same shape: interactions lose most (M, n small).

### 2.3 What the paper states (reaction-only estimate)

Method (M): for each canonical run of a gold-case paper, count gold reactions (both sides' defining participants matched, `bench/eval_v2/reactions.py:104 match_reaction`) present in S1, M, F. Sample: **79 canonical runs, 7 papers, 249 (run x gold reaction) slots** (heavily repeated papers: PMC12096016 x20, PMC12452463 x19, PMC12782028 x16, PMC12856317 x16; effective n is 7 papers). Gold lists the clearly stated reactions, not every one (only 3 papers are certified complete).

| corpus | S1 | M | F | in PWML |
|---|---|---|---|---|
| all 79 runs | 88.4% (220/249) | 90.0% | 88.4% | n/a |
| certified-complete papers (3 papers, 24 runs, 120 slots) | 90.8% | 93.3% | 91.7% | n/a |
| PathWhiz mode (41 runs, 135 slots) | 88.9% | 90.4% | 87.4% | **49.6% (67/135)**: 23 of 41 runs wrote a PWML; given a PWML, 90.5% (67/74) |
| research mode (38 runs, 114 slots) | 87.7% | 89.5% | 89.5% | no PWML |

Union over runs of the same paper reaches 23/23 gold reactions at S1: every gold reaction is extractable, no single run reliably gets it (L: RUNTIME-VARIANCE.md). Per-paper S1 recall ranges from 71.9% (PMC12452463) to 100% (M). Stage-2 merge adds 4 slots (+1.6 pts), F loses 4 (M to F: PMC12421875 x2, PMC12452463, PMC12444477).
Entity level (G, not a recall bound): across 11-20 repeated runs of the same paper the union of distinct normalized names is 1.4-6x the per-run median (e.g. PMC12096016 compounds 25 vs 11, reactions 30 vs 5), so a single Stage-1 draw sees a minority of the names that all draws see together; name-spelling variants inflate this.

### 2.4 What is added after Stage 1 (M, ALL152)

Reactions 561 (S1) to 705 (M): +143 with no lineage record (+2 `rag_literature`); transports +4; interactions +281 non-alias rows (+210 `SAME_AS`). In the 116 runs where the merged payload carries the paper text, the evidence quote of the added row appears verbatim in the paper for 84/118 reactions (71%), 137/208 interactions (66%), 3/4 transports. So most additions are second-pass recoveries of paper content, a minority are not verbatim (G: paraphrase or inference). Entity-admission gate `pipeline/entity_admission.py` removed only 10 rows in 152 runs and demoted 95 currency compounds (kept, not dropped).

### 2.5 Fate of Stage-1 rows in PathWhiz mode (M, 101 runs, rows from S1)

Interactions (234 S1 rows, alias-aware endpoint-pair match):

| fate | rows | share |
|---|---|---|
| kept in F (76 same pair, 10 rewritten) | 86 | 37% |
| quarantined: `protein_missing_external_identity` 44, gene/protein `ambiguous_entity_type` 35, `undeclared_entity_in_entity_1/2` 15 | 94 | 40% |
| duplicate of another S1 row (same alias-resolved pair) | 27 | 12% |
| an endpoint entity is gone from F | 18 | 8% |
| both endpoints present, row absent (not attributed) | 9 | 4% |

Then 40 of the 127 F interactions in P1 are withheld at export (below).
Reactions (402 S1 rows): kept as-is 269 (67%); duplicate of an S1 row 52 (13%); participants rewritten so the set no longer matches 47 (12%; in a second check 59 of 63 unmatched rows still have a same-named F reaction, so mostly rewrites, not losses); quarantined 25 (6%: identity 15, `ambiguous_entity_type` 10); locked-reaction record (coarse-grained no-op 6, duplicate 3) 9 (2%).
Transports: F paper-stated 71 of S1 75 (95%).
Entities (merged rows, PathWhiz mode): compounds 875: kept 75%, removed `unreferenced_after_quarantine` 14%, absent unattributed 7%, duplicate 5%. Proteins 741: kept 63%, `degree_zero_after_quarantine` 12%, absent 23% (132 recorded as KI-013 `not_referenced_by_any_surviving_process`), duplicate 2%. RS for comparison: compounds 91% / proteins 97% kept.

### 2.6 DROPPED vs KEPT-UNMAPPED vs KEPT-MAPPED

| | P1 PWML (61 runs) | RS output (51 runs) |
|---|---|---|
| compounds kept and mapped (external id) | 327 of 431 (76%) | 298 of 376 (79%) |
| compounds **kept but unmapped** (no id; emitted as new/unmatched compound) | **104 (24%)**; IR `db_status` unmatched/ambiguous/identity-refused = 113 | 78 (21%) |
| proteins kept and mapped | 241 of 241 (100%; 18 are the PathBank `Unknown` sentinel) | 326 of 401 (81%) |
| proteins **kept but unmapped** | 0 (D-106 gate: they cannot ship) | **75 (19%)** |
| complexes | 234: 95 matched, 68 Unknown-backed fallback, 71 other; 134 are generated wrappers | 25 |
| **dropped** instead of kept unmapped | proteins' processes: see 3, R3 | none (flagged only) |

Reading (M): the PWML never carries an unmapped protein; a protein without UniProt/DrugBank id costs its processes (dropped), while a compound without an id is kept as a placeholder entity. 120 of the 431 P1 compounds have mapping status `novel` (`no_db_candidates`/`implausible_name_match`, `mapping/map_ids.py:1894,5999`), 32 `ambiguous`.

### 2.7 PWML export step itself (M, P1)

Entities, reactions and transports pass unchanged (431/241/234/12; 245; 42). Interactions: 127 in F, 87 in the PWML; the recorded withholds are `untyped_interaction` 38 and `bare_protein_endpoint` 2. Caveat: 70 of the 87 drawn interactions were written with raw words (`inhibits` 21, `modifies` 11, `catalyzes` 9, `binds` ...) as `interaction-type` by Aug-Sep writers; since ORCH-749 only `Activation`/`Inhibition` export (the 17 Oct/Nov-era interactions are all typed). Under today's rule, 86 of the 127 F interactions in P1 carry words outside `PWML_INTERACTION_TYPES` (`interaction_types.py:33`) (binding 27, modifies 11, catalyzes 9, binds 5, regulates 5 ...), i.e. about two thirds would be withheld (G for HEAD behaviour).

## 3. Ranked loss reasons

Size = rows in the 101 PathWhiz-mode runs unless stated. "Correct?" = under PRODUCT_CONTRACT (smaller defensible beats larger contaminated; D-106 identity gate locked; D-114 per-row omission).

| # | reason | size (M) | code | correct under contract? |
|---|---|---|---|---|
| 1 | **Run writes no PWML** (stage-3 gates / required-field gate) | 40 runs, 276 F processes (40%) | `service/postpipeline.py:453` (`final_pre_export_stage3_gates`); `pwml/ir.py:2947,2955` (species); `pipeline/reaction_support.py:125` | **No.** Contract section 1: a recoverable problem must not end a run without PWML; D-114 section 2 replaced it by per-row omission (KI-016, status "built by ORCH-773" per D-114; not measured here) |
| 2 | Interaction word not Activation/Inhibition (KI-008) | P1: 40 withheld (38 recorded) of 127; ~86 of 127 under HEAD rule (G); 122 of 346 S1 interactions untyped in ALL152 | `pwml/ir.py:62`; `pwml/interaction_types.py:33` | **By design for PWML** (D-104 section 4; PathBank has 2 types). Not a loss for other formats (BioPAX/SBGN/SBML-qual can hold binding, modifies): see section 5 |
| 3 | `protein_missing_external_identity` (KI-001) | 101 processes (23 rx, 3 tr, 75 ia) in P1+P0; RS would lose 97 | `pipeline/strict_quarantine.py:765` | **Yes, locked** (D-106). Recoverable only by a better lookup (D-106 section 2): RS shows 19% of proteins have no id |
| 4 | Consequential entity closure: compounds `unreferenced_after_quarantine`, proteins `degree_zero_after_quarantine`, KI-013 orphans, complexes, nucleic acids | compounds 154, proteins 101 + 132, complexes 24, nucleic 24 | `strict_quarantine.py:2250`; `process_normalizer.py:2199` | Yes (degree-zero entities are not allowed); derived from #3, #5, #6 not independent |
| 5 | Gene/protein same-symbol clash `ambiguous_entity_type:nucleic_acids+proteins` (KI-011) | 37 interactions, **all in one run** (chlorophyll PMC5130987, Oct 22) | `strict_quarantine.py:1370,1421,1798` | Correct at the time; **L: fixed by ORCH-782 (D-117 section 1)**, 28 to 10 on fixture; not re-measured |
| 6 | Undeclared endpoint (`undeclared_entity_in_entity_1/2`, KI-015) | 30 interactions (P1 22, P0 8) | `strict_quarantine.py:1375` | Correct (no invented entity), but it is an extraction inconsistency: the endpoint is named in the row and never declared (recoverable, G) |
| 7 | Reaction no-op: inputs equal outputs (`coarse_grained_same_entity_transformation`, KI-015) | 18 reactions (P1 15, P0 3); RS 10 | `pipeline/process_normalizer.py:4170` | Correct drop of a mis-typed row; the content is usually a transport (sour taste: "H+ permeation through OTOP1: H to H" duplicates the kept transport) |
| 8 | Duplicates: S1 pooled draws (52 rx, 27 ia), `duplicate_locked_reaction` KI-014 (15), `SAME_AS` alias rows (210) | 52 + 27 + 15 + 210 | `process_normalizer.py:2353,4088`; `pipeline/pipeline.py:1370,2640` | **Correct, not a loss** |
| 9 | Ambiguous type reaction endpoints (`protein_complexes+proteins` 11, `compounds+nucleic_acids` 3) | 14 reactions | `strict_quarantine.py:1370` | Yes |
| 10 | Stage-1 under-extraction | S1 gold reaction recall 88.4%; +143 reactions found by the second pass | `pipeline/pipeline.py:1171 merge_additions` (Stage-2 merge) | Not a drop: a recall gap (2.3, 2.4) |
| 11 | Entity-admission gate removals / currency demotion | 10 removed (hallucination gate, 6 reagent forms) / 95 demoted (kept) | `pipeline/entity_admission.py:103` | Yes |

By content kind (section 2.5 shares): interactions lose most (kept 37%), reactions least (kept 67% + 12% rewritten; 13% were duplicates), transports 95% kept. Details by paper class are in assessment 03.

## 4. Cause notes (correlation vs cause)

- **#1 is run-level, #3-#9 are row-level.** The P0 failures cluster in Aug-Sep (36 of 40); era and cause are confounded: the same gates now omit a row instead of failing the pathway (L: D-114, KI-016 text). I do not claim the rate has fallen.
- **#3 (identity)** is the dominant row-level cause in every group (P1+P0: 101 of 181 applied quarantines; RS 97 of 113 diagnostic). Cause is the lookup, not the gate (L: D-106; KI-001 cause text). The 75 unmapped proteins in RS output (19%) are a direct sample of the lookup failure rate (M).
- **Signalling interactions** are lost because three independent rules stack (#2 word, #3 identity, #5/#6 type/declaration): in signalling-class papers 141 S1 interactions leave 36 paper-stated in F and 11 in the PWML (assessment 03).
- Gold shows the cleanly stated reactions survive (M to F loses 1.6 pts); the loss is concentrated in content that gold does not list (interactions, transports, peripheral reactions), so the gold-based recall (88-91%) overstates overall yield (G).

## 5. WHAT TO DO

Proposed, each with gain bound and no-regression risk. None changes a locked decision; items touching one are marked QUESTION.

| id | proposal | expected gain (bound) | no-regression risk |
|---|---|---|---|
| A | Keep the **text record + non-PWML exports** for interactions PathWhiz cannot type (#2): emit them in SBML/BioPAX/SBGN-ML/KEGG-style outputs from the final payload, with sign "unspecified" where unsigned | up to 86 of 127 P1 interactions and 122/346 S1 interactions (35%) become representable in at least one format; PWML unchanged | Low: new exporters read `final_mapped.json` after freeze and add nothing biological; PWML bytes unchanged |
| B | Re-measure the 40-run P0 failure on a post-D-114 replay (the Oct-25 corpus ORCH-768/772) before any work on #1 | Upper bound 40% of PathWhiz runs, 276 processes; realistic gain unknown | None (measurement) |
| C | Extraction: require every interaction endpoint be declared as an entity row (or auto-declare when the paper names it) (#6) | up to 30 interactions (13% of S1 pathwhiz interactions) | Low-medium: must not create entities with no paper support (contract hard limit); gate on a verbatim mention |
| D | Protein lookup recall (D-106 section 2, existing ORCH lanes): organism/strain-aware lookup | up to 101 processes in P1+P0 (12% of the 871 processes in the text record) and 75 RS proteins; bound = share that truly exist in UniProt | Medium: identity false positives; D-106 forbids relaxing the gate; measure with ID-correctness (`eval_v2/reactions.py:562`) |
| E | Re-type no-op reactions as transports when the evidence says "permeation/influx" (#7) | up to 18 reactions; only where no transport already exists | Medium: biology reinterpretation after S1; do upstream, not in the exporter (merge rule 8). QUESTION for owner |
| F | Report mapping state of kept compounds in the PWML text record (#2.6: 24% placeholders) | none in count; improves honesty and review | None |
| G | Raise Stage-1 recall (second pass already recovers +143 reactions, 71% quoted): promote the second-pass finding as a first-class stage with provenance | up to ~20% more reactions (143/705) of which ~71% verbatim-supported | Medium: more rows also means more contamination; gate on verbatim span as `entity_admission` does |

**Questions for the owner:** (1) Is a 24% share of compounds emitted without a database id acceptable in a "release_ready" PWML? (2) Is re-typing a no-op reaction as a transport (E) allowed, or must it stay an omission? (3) Do non-PathWhiz exporters (A) fall under "no biology after freeze" (they only re-serialise), so interactions PathWhiz cannot type may appear there? (4) Which era's numbers should be the baseline for the paper: Aug-Sep legacy or a fresh post-D-114 replay?
