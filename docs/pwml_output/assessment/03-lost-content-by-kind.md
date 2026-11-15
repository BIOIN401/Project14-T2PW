# Assessment 03 - Which kinds of content are lost most; is "0 reactions" correct for sour taste

Branch `pwml/output-assessment` (from `85b5f9d4`). Read-only. Labels: **M** measured, **L** log/doc evidence, **G** guess.
Data: `docs/pwml_output/evidence/01/` (`tidy_runs.csv`, `paper_classes.csv`, `class_kind.json`); scripts in `evidence/01/scripts/`. Companion: assessment 01 (stage funnel, loss reasons with file:line).

## 0. NOT ESTABLISHED

1. **What each paper states, by kind.** Only reactions have gold (7 papers, 23 reactions). Transport and interaction denominators are the Stage-1 extraction itself (S1), which is also what the paper classes are derived from; the classification is therefore circular (class = what the extractor found).
2. **Non-metabolic classes are tiny.** Of 51 canonical papers: 35 metabolic, 6 signalling, 2 transport-heavy, 8 sparse. Signalling and transport-heavy results rest on 6 and 2 papers (7 and 6 runs). Treat them as case studies, not rates.
3. **Era.** 91 of 101 PathWhiz-mode runs predate D-104/D-114/ORCH-749/D-117 (see assessment 01, section 0). The HEAD behaviour for interactions in particular is known only from code and decision text (L), not from re-runs.
4. **Whether the sour-taste review's non-sour content** (taste cell types, Type II cascades, salt) is in scope for the requested "Taste transduction: sour"; I read only the sour paragraphs and the Type III transmission sentences.
5. **My count of what the sour paragraph states** (about 5-6 processes, section 3.1) is a reading of the text by me, not a certified gold (G).

## 1. Sample, classification rule, paper table

Corpus: 152 canonical runs (`quarantine_ok=true`), 51 distinct papers (the two web runs e2e4/e2e5, PMID 32185015, are the same review as PMC7059786). Split: 101 PathWhiz-mode runs (61 wrote a PWML, 40 did not), 51 research-mode runs. The other 40 runs (12 papers) are excluded from every number here.

**Rule (stated before use; M):** for each paper take the median over its canonical runs of Stage-1 counts R (reactions), T (transports), I (interactions); total = R+T+I.
1. total < 3: `other (sparse)`.
2. T >= 2 and T/total >= 0.4: `transport-heavy`.
3. R/total >= 0.5: `metabolic`.
4. I/total >= 0.5: `signalling`.
5. otherwise `other (mixed)` (no paper fell here).
Cross-check: the Stage-0 requested pathway name (keywords biosynthesis/metabolism vs signal/transduction/operon vs transport/taste) agrees for 41 of 51 papers (all 35 metabolic, both transport-heavy); it disagrees on 4 `signalling` and 6 `sparse` papers, all named "... biosynthesis" but extracted as interactions-only or near-empty (flagged in `paper_classes.csv`, column `name_keyword_class`).

| class | papers | canonical runs | paper id (runs) |
|---|---|---|---|
| metabolic | 35 | 101 | PMC10031235 (2), PMC10055903, PMC10269868, PMC11016064, PMC11405693 (2), PMC11487621, PMC11961743, PMC12051542, PMC12059215, **PMC12096016 (20)**, PMC12369002, PMC12376012 (2), PMC12421875 (2), PMC12444477 (4), **PMC12452463 (19)**, PMC12505894, PMC12542839, PMC12707518, **PMC12782028 (16)**, PMC12914822, PMC12983300, PMC13084691, PMC13089919, PMC13123502 (2), PMC13184244, PMC13364408, PMC13438895, PMC13474689, PMC13486683, PMC4471609 (2), PMC4725005, PMC6112128, PMC7232280 (2), PMC7402084, PMC9544450 (5) |
| signalling | 6 | 7 | PMC10425814 (WNT/PCP), PMC11172790 (2, pyoverdine), PMC11983282 (Ara operon), PMC13405594 (steroid), PMC5130987 (chlorophyll), PMC8510960 (terpenoid indole alkaloid) |
| transport-heavy | 2 | 6 | PMC7059786 (5, sour taste), PMC12071552 (wall teichoic acid D-alanylation) |
| other (sparse) | 8 | 38 | **PMC12856317 (16)**, PMC12180156 (11), PMC13231680 (4), PMC12312563 (2), PMC13488460 (2), PMC13170813, PMC13474940, PMC8211424 |

Heavy repetition (bold) means runs, not papers, dominate counts: PMC12096016 + PMC12452463 + PMC12782028 + PMC12856317 are 71 of 152 runs. Per-class tables below are pooled by run; I also state paper counts.

## 2. Loss by kind (M)

### 2.1 Pooled, PathWhiz-mode canonical (101 runs, S1 rows = paper-stated by lineage)

| kind | S1 | in F, paper-stated | kept | drawn in PWML (61 runs; 40 deliver none) | drawn / S1 |
|---|---|---|---|---|---|
| reactions | 402 | 351 | 87% | 245 | 61% |
| transports | 75 | 71 | 95% | 42 | 56% |
| interactions | 234 | 94 | **40%** | 87 | 37% (70 of the 87 are Aug-Sep raw-word interactions) |

"Drawn / S1" mixes retention with the 40-run failure and with rows added after Stage 1, so use "kept" for the extraction-to-final step and the PWML column for delivery. **Interactions are lost far more than reactions or transports at every step**; reactions and transports lose mainly through whole-run failure (40% of runs write nothing: assessment 01, section 2.1) rather than row-level drops.

### 2.2 By paper class (PathWhiz mode; sample sizes in the header)

| class (runs / papers) | kind | S1 | F paper-stated (kept) | PWML drawn |
|---|---|---|---|---|
| metabolic (69 / 34) | reactions | 341 | 307 (90%) | 198 |
| | transports | 45 | 43 (96%) | 23 |
| | interactions | 53 | 35 (66%) | 50 |
| signalling (5 / 5) | reactions | 40 | 27 (68%) | 26 |
| | transports | 7 | 6 (86%) | 2 |
| | interactions | 141 | 36 (**26%**) | **11** |
| transport-heavy (5 / 2) | reactions | 6 | 2 (33%) | 2 |
| | transports | 23 | 22 (96%) | 17 |
| | interactions | 19 | 5 (26%) | 5 |
| sparse (22 / 8) | reactions | 15 | 15 (100%) | 19 |
| | interactions | 21 | 18 (86%) | 21 |

PWML-write rate by class: metabolic 43 of 69 runs (62%), signalling 2 of 5, transport-heavy 4 of 5, sparse 12 of 22 (M; n tiny for the middle two). In research mode (51 runs) retention is higher and flatter: reactions 95%, transports 100%, interactions 72% (metabolic), 94% (signalling, 2 papers), because nothing is removed (assessment 01, section 1).
In the signalling class 141 S1 interactions yield 36 paper-stated in F and 11 in a PWML: 8% end to end (M, n = 5 papers; one run, chlorophyll PMC5130987 (Oct 22), accounts for 81 of the 141 S1 interactions and for all 37 gene/protein-clash quarantines in the corpus, the clash D-117 section 1 fixed later, L).

### 2.3 Interactions by relationship sign (PathWhiz mode, 207 unique S1 rows after removing 27 duplicates; M)

| relationship | rows | in F | quarantined | other lost |
|---|---|---|---|---|
| Inhibition (typed) | 73 | 32 (44%) | 22 (30%) | 19 |
| Activation (typed) | 75 | 8 (11%) | 50 (67%) | 17 |
| binding / interacts / associates (unsigned) | 53 | 31 (58%) | 21 (40%) | 1 |
| other unsigned (modifies, regulates, degrades) | 6 | 5 | 1 | 0 |

Only the 40 signed rows that reach F can be drawn by today's rule; the 36 unsigned rows that reach F are withheld at export (KI-008, `pwml/ir.py:62`, `pwml/interaction_types.py:33`). So **about 40 of 207 (19%) unique Stage-1 interactions are deliverable in a PathWhiz PWML**; the 11% Activation survival is dominated by 35 gene/protein clash quarantines from one run (assessment 01 loss #5) and should not be read as a general rule (G: confounded with that paper).
Regulation and binding are lost mostly at three gates: protein identity (44 rows), gene/protein type (35), undeclared endpoint (15), then the sign table at export.

### 2.4 Hidden conversion: interactions absorbed by `promote_interaction_enzymes`

`process_normalizer.py:3789` rewrites interactions whose verb contains `catalyz`/`enzyme`/`activat`/`promotes`/`facilitate` into reaction enzymes and deletes the interaction. Counted in `normalization_stats.interaction_enzymes_promoted`: 51 across the 152 canonical runs (P1 23, P0 7, RS 21) (M). If the receiving reaction is later dropped (no-op, duplicate, quarantine) the interaction is gone with no record of its own. Worked example in section 3.2 (e2e5).

## 3. Sour taste (OTOP1): is 0 reactions correct?

### 3.1 What the paper states (M for quotes, G for the count)

The five sour-taste runs are all the same paper: "Recent advances in taste transduction and signaling" (Kinnamon and Finger, F1000Research 2019; PMC7059786 = PMID 32185015), a review, not a primary OTOP1 paper. The sour passage (source text, "Sour" paragraph) states, in order: PKD2L1 marks sour cells (2006); PKD2L1 itself is not required (Horio 2011); "transduction of sour involves permeation of H+ through an apical ion channel subsequently identified as OTOP1"; "The entering H+ ions not only directly depolarize the taste cells but also block Kir2.1 K+ channels, thereby amplifying the depolarization"; depolarization "triggers voltage-gated Na+ channels (SCN2A) to generate action potentials that activate voltage-gated Ca++ channels triggering the release of synaptic vesicles"; 5-HT acting on 5-HT3 receptors transmits sour; optogenetic driving of PKD2L1 cells evokes aversion. The CALHM1/CALHM3 ATP release belongs to Type II (sweet/bitter/umami) cells, not sour.
No sentence names a substrate converted to a product. My reading: about 5-6 processes, all of kind **transport** (H+ through OTOP1; Ca2+ entry through Cav; 5-HT vesicle release) or **interaction/regulation** (H+ blocks Kir2.1; depolarization activates SCN2A; Cav activation; 5-HT activates 5-HT3), and **zero reactions** (G).

### 3.2 What the five runs emitted (M; `evidence/01/scripts/sour.py`)

| run | mode | S1 rx / tr / ia | F rx / tr / ia | PWML rx / tr / ia | distinct processes drawn |
|---|---|---|---|---|---|
| ORCH-755 research, Oct 21 | research | 0 / 4 / 3 | 0 / 4 / 3 | none (research) | 3 transports + 3 interactions in the record, none drawn |
| ORCH-755 strict, Oct 21 | pathwhiz | 0 / 7 / 5 | 0 / 6 / 1 | **none: export failed** (pre-export Stage 3 revalidation) | 0 |
| ORCH-759p v2 strict, Oct 22 | pathwhiz | 0 / 6 / 5 | 0 / 7 / 0 | 0 / 7 / 0 | 3 (H+ influx x3 rows, 5-HT release x3, Ca2+ influx x1); 4 interactions omitted `protein_missing_external_identity` |
| e2e4, Nov | pathwhiz | 1 / 5 / 3 | 0 / 5 / 2 | 0 / 5 / 2 | 4 (H+ influx x3 rows, 5-HT release x2, H+ blocks Kir2.1 complex, 5-HT activates 5-HT3) |
| e2e5, Nov | pathwhiz | 3 / 3 / 6 | 0 / 3 / 3 | 0 / 3 / 3 | 3 (H+ influx x2 rows, serotonin exocytosis, 5-HT activates 5-HT3 x3 spellings) |

Findings:
1. **0 reactions in the PWML is the correct count.** Every reaction the model produced (e2e4: "H+ permeation through OTOP1", "action potential generation by SCN2A", "voltage-gated Ca++ influx", "Kir2.1 K+ efflux"; e2e5: "H+ permeation through OTOP1", "Kir2.1 block by H+", "ATP release via CALHM1/CALHM3") has inputs equal to outputs (H to H) and was dropped as `coarse_grained_same_entity_transformation` (`pipeline/process_normalizer.py:4170`, KI-015). These are channel events mis-modelled as reactions; dropping them is right.
2. **The contract allows it.** PRODUCT_CONTRACT section 1: smaller supported pathway beats larger contaminated one; D-113 section 3 (interaction-only pathways export as `review_required`); D-114 AMENDMENT 2 section 3 (even a PWML drawing zero processes is written). The `no_defensible_reaction_support` gate (`pipeline/reaction_support.py:125`) fires only when reactions exist and none is supported; e2e4/e2e5 passed it with 0 reactions (M). Strict quarantine requires only `min_core_processes: 1` (`quarantine_report.decision_inputs`). The gold set's `min_connected_reactions` is a metabolic notion with no transport-only counterpart (L).
3. **But content was still lost, and the losses are in transports and interactions, not reactions.** Against my reading of 5-6 stated processes, the best PWML (e2e4) draws 4 distinct (H+/OTOP1 transport, 5-HT release, H+ blocks Kir2.1, 5-HT activates 5-HT3); none draws SCN2A firing or Cav activation, and the Ca2+ entry through Cav appears in only one run (759p v2). Each PWML carries 3-4 duplicate rows (spellings "H+ influx through OTOP1", "H+ permeation through OTOP1", "H+ influx into Type III cell" are one event; 10 duplicate rows across the three PWMLs).
4. **Hidden chain in e2e5 (M facts, G link):** `interaction_enzymes_promoted=3`, `no_op_quarantined_count=3`, `orphan_proteins_dropped=6` in `initial_stage3_gate_report.json`. "H+ inhibits Kir2.1" and "ATP activates P2X receptor(s)" vanish between M and F with no quarantine entry; Kir2.1, P2X, SCN2A, PKD2L1 and the Cav channel are then dropped as unreferenced proteins (KI-013). The most plausible path is promote-into-the-no-op-reaction then drop (section 2.4). Result: the sour-defining Kir2.1 block is absent from the e2e5 PWML.
5. Run-to-run variance is large: the same paper gives a failed export (755 strict), 0 interactions (759p), and 2-3 interactions (e2e4/e2e5), with 1 vs 3 reactions at S1 (L: RUNTIME-VARIANCE.md).
6. Scope: ORCH-755 requested *Homo sapiens* for a mouse-dominated review; the request was corrected to *Mus musculus* in the later runs (L: KI-026, D-110 section 2).

Verdict: 0 reactions is a **correct output**; 3-4 of ~5-6 stated processes in the best case is the real shortfall, caused by identity gating (Kir2.1, 5-HT3, P2X are receptors/channels with no accession found: runs 759p and 755), by the no-op/promotion chain, and by pooled-draw duplicates.

## 4. WHAT TO DO

| id | proposal | expected gain (bound) | no-regression risk |
|---|---|---|---|
| 1 | Add a **transport/channel-gating process type** so that "permeation", "influx", "channel opens" statements enter as a transport at Stage 1 instead of an H-to-H reaction (prompt/schema, not exporter) | removes the source of 18 no-op reactions in PathWhiz mode and 7 in the sour-taste runs; recovers up to the interactions absorbed by them (e2e5: 2 of 3) | Medium: touches `llm/prompts/` (locked from this track; owner ruling needed) |
| 2 | Log interactions removed by `promote_interaction_enzymes` and the follow-on orphan drop in the omission record (`reaction_evidence`), so that nothing disappears unrecorded | 0 content, but makes the 51 promotions auditable; today e2e5 loses two sour-defining interactions silently | None: reporting only |
| 3 | Do not delete a promoted interaction if its receiving reaction is later quarantined as a no-op; restore the interaction | up to the promoted subset of 51 (unknown share; measure first) | Medium: changes row survival; needs a replay on gold cases |
| 4 | Export unsigned interactions (binding, modifies, regulates; 36 of 76 F survivors in PathWhiz mode) to formats that carry them (SBML-qual, BioPAX, SBGN-ML, GPML) from the frozen payload | up to 122 of 346 S1 interactions across ALL152 (35%) become representable; PWML unchanged | Low: exporters add no biology after freeze |
| 5 | Receptors/channel identity: organism-aware UniProt lookup for named channels and receptors (Kir2.1, 5-HT3, P2X) under D-106 | sour taste: 3 interactions in 759p; across PathWhiz mode up to 44 interactions + 23 reactions + 3 transports | Medium: identity false positives; D-106 forbids relaxing the gate |
| 6 | Classify papers at intake (metabolic / signalling / transport) and report a class-appropriate completeness measure; do not apply "connected reactions" floors to non-metabolic classes | no count change; avoids reading 0 reactions as failure | None |
| 7 | Collapse near-identical transports/interactions after pooling (assessment DUPLICATION covers) | removes about 10 duplicate rows across the 3 sour-taste PWMLs | Low, but D-034/D-035/D-036 govern duplicate consolidation: QUESTION for the owner, do not change silently |

**Questions for the owner:** (a) May Stage-1 prompting introduce a channel-gating/translocation type, given `llm/prompts/` is outside this track? (b) Should a transport-only or interaction-only pathway carry its own completeness definition (0 reactions is then the expected value)? (c) Is an interaction removed by `promote_interaction_enzymes` and then orphaned acceptable, or must it be restored (proposal 3)? (d) Same era question as assessment 01: which baseline for the paper?
