# PLAN: more correct PWML content, then better diagrams and more file types

Status: PROPOSAL for the product owner. Nothing here has been implemented. Written 2026-11-10 from the seven assessments in `assessment/` (01 to 07b), `VERIFICATION.md`, `DECISIONS-PROPOSED.md`. Evidence tables are in `evidence/`.
Rule applied to every item: **no regression** (never worsen text-to-PWML output; if no loss cannot be proved, defer). "Add-only" means an item adds fields or reports and leaves existing PWML bytes unchanged.

## 0. What could not be established (read this first)

1. **Whether any generated file imports into PathWhiz.** No result is recorded for the current pipeline; D-099 section 19 is open (07). Everything below about "valid output" is offline.
2. **Behaviour at HEAD.** 150 of the 152 canonical runs predate D-117 to D-120 (2026-10-28); 91 of 101 PathWhiz-mode runs predate D-104/D-114 (01, 02). Every "still lost" figure is an upper bound for HEAD until a replay is done. The 40-run no-PWML failure has no post-D-114 measurement.
3. **Entity-level gold.** Only reactions have gold (7 papers, 23 reactions; 3 certified papers, 20 of 24 scored runs are one paper). There is no precision or recall for entities, transports, interactions (04).
4. **Gain from any lookup.** "Recovered" in 02 means a unique plausible database record, not verified truth. Wrong-id rates rest on tiny samples (4 of 12 ambiguous compounds wrong; 1 of 30 baseline compounds).
5. **Homology via orthology tables / BLAST** was not run; the ceiling for annotation-poor organisms (about 26 units) is a guess.
6. **KEGG:** whether a free public server counts as an "academic user" or "service provider", and whether KEGG ids alone in a file are redistribution (07b). Legal conclusions are not made here.
7. **Noise:** no same-commit repeat set exists, so run-to-run figures are upper bounds on pure LLM noise (05).

## 1. Findings that drive the plan

| # | Finding | Number (corpus, denominator) | Source |
|---|---|---|---|
| F1 | Whole-run failure is the biggest single loss: PathWhiz-mode canonical runs that wrote **no .pwml** | 40 of 101 runs; 276 processes (183 reactions, 32 transports, 61 interactions); all 40 predate D-114. **Independently reproduced.** | 01, VERIFICATION |
| F2 | Of what is delivered, interactions are lost most | ~40 of 207 unique Stage-1 interactions (19%) are drawable in PWML (Activation/Inhibition only); 40% of Stage-1 interactions quarantined | 01, 03 |
| F3 | Reactions are extracted well; delivery is the loss | gold reaction recall 88.4% at Stage 1 and final (249 slots, 7 papers, 79 runs); only 49.6% (67/135) in delivered PWMLs | 01 |
| F4 | Protein/enzyme units without a database accession | 168 of 435 (38.6%) final-output units, 152 canonical runs (definition reconciled, VERIFICATION); the model/non-model gap (26.0% vs 71.7%) sits in complex wrappers, not in the proteins bucket (19.7% vs 15.0%) | 02 reconciliation |
| F5 | Compounds without an id | 147 of 517 units (28%), independent count 148 of 520; 58 of 147 (39%) have an exact ChEBI/KEGG name hit, 49 (33%) an InChIKey | 02 |
| F6 | Name lookup in the paper's own taxon would recover about 51 of 197 failed protein/complex units (17 of 93 tested); 11 of those 17 were the pipeline's own refused candidate | 02 |
| F7 | Homology adds almost nothing safe | lower bound 0, upper about 10 units (5%); paralogs reach 84% identity; one cross-species false hit scored 58% | 02 |
| F8 | Mapped-id accuracy | about 5.7% (3.8 to 7.6) of evaluable ids wrong or narrowed (17 real of 36 hand-checked flags; 19 were matcher artefacts) | 04 |
| F9 | Added rows are riskier than Stage-1 rows | Stage-2 rows 6 of 13 true positives (46%) vs 93.5% for Stage-1 rows; quarantine dropped 7 rows on 13 runs, all true positives | 04 |
| F10 | PWML invents defaults and drops detail | every one of 61 PWMLs says "Generated Pathway"/"Metabolic", empty references; stoichiometry 1 on 623/623, direction ">" on 246/246; evidence quotes 621/621 reactions lost in the IR; gene names never written | 06 |
| F11 | Today's offline QA fails a third of delivered files | 36 pass, 25 of 61 fail, all on interaction shape; **reproduced** | 07, VERIFICATION |
| F12 | Run-to-run noise is large for reactions | Jaccard between two runs: proteins 0.75, compounds 0.62, reactions 0.38, transports 0.33, interactions 0.25; per-run recall SD 0.11 to 0.13 | 05 |
| F13 | Sour-taste paper: 0 reactions is correct | the paper states H+ permeation and channel interactions only; the best PWML draws 4 of about 5 to 6 stated processes | 03 |

Correction note: the 02 document's first headline (197 of 464, 42.5%) mixed definitions; use F4. Compound figures were reproduced.

## 2. Ranked improvements

Ranking = expected gain times safety, per cost. Group A is add-only or measurement; group B changes pipeline logic and needs an owner ruling and a D-entry.

### Group A. No behaviour change in existing PWML (do first)

| Rank | Item | Evidence | Expected gain (bound) | Regression risk | Cost | Needs owner ruling? |
|---|---|---|---|---|---|---|
| A1 | **Replay at HEAD** the 51 name-recoverable units, the 44 `species_mismatch` units, and re-measure the 40-run no-PWML failure on a fresh post-D-114 set | F1, F6; 01, 02 | turns upper bounds into real numbers; may show F1 is already fixed | none (measurement; bounded wrapper, one heavy job) | low to medium (LLM spend for the fresh set) | no |
| A2 | **Metadata block** (pathway title, PMID, organism) carried into the payload before the freeze so the writer stops emitting "Generated Pathway" and empty references | F10; `ir.py:1352` already reads `payload["metadata"]` | fixes title and references in 61 of 61 files; no biology | low: text fields only; run the golden-seam tests | low | no (no biology); confirm title wording rule |
| A3 | **Keep process evidence and "unstated" markers in the IR** (quotes for 621 reactions; stoichiometry/direction/spontaneous marked unstated for non-PWML exports; PWML defaults byte-identical) | F10 | gives every non-PWML export provenance; no change to PWML | none if PWML path is untouched | low to medium | no |
| A4 | **Stronger read-only offline validator** (importer rules from D-099 docs: UniProt/DrugBank on proteins, spontaneous when enzyme-less, id references resolve, species and compound ids vs `data/pathwhiz_id_db.json`); fix qa.py false positives (bare interaction stubs, the `option:` namespace) | F11, 07 | closes the "stale QA" gap; makes 25 old failures visible before any import | none if not wired into the writer; qa.py is shared so add tests with `PW1.pwml` and `pathway (39).pwml` | low | possibly a D-entry if it touches a locked gate |
| A5 | **Instrument lineage**: per-row draw id at pooling; log what `promote_interaction_enzymes` absorbs (51 promotions in corpus) | DUPLICATION-ASSESSMENT option 4; 01 | attribution of loss and near-copies; unblocks later fixes | none | low | no |
| A6 | **Manual PathWhiz import test** of 3 files (positive control `reference/pathway (39).pwml`, one Oct canonical run that passes QA, one Aug-Sep failing run e.g. PMC9544450) | 07 | the only way to establish import validity | none | owner time, staging PathWhiz | **yes: owner action** |
| A7 | **Evaluation protocol**: certify 3 more gold papers (a signalling, a plant, a eukaryotic metabolic paper); replay frozen `stage1_payload.json` for deterministic changes; for LLM-affecting changes use at least 10 papers x 5 runs per arm | F12; 04, 05 | 10 x 5 detects about 7 recall points; +5 points needs about 91 runs per arm, +10 needs about 23 | none | medium | budget ruling |
| A8 | **Format-neutral clean model, then SBML export first** (see `CLEAN-MODEL-REQUIREMENTS.md`). Unsigned interactions and binding relations go to non-PWML formats only | F2, F10; 06 | interactions: up to 35% of Stage-1 interactions (122 of 346) that PWML cannot hold; SBML fillable rows 9 of 30 today, 19 of 30 after pre-freeze work | low: reads the frozen payload | medium to high | yes: D-023 "no SBML implementation" boundary and D-104 Amendment 1 (unsigned relations) must be ruled |

### Group B. Changes pipeline logic or biology-adjacent behaviour (owner ruling, D-entry, no-regression gate)

| Rank | Item | Evidence | Expected gain (bound) | Regression risk | Cost |
|---|---|---|---|---|---|
| B1 | **External exact-name lookup for compounds** when PathBank returns "novel" (`map_ids.py:7732` returns before the ChEBI/KEGG name API is asked); move ChEBI call to its current API (the targeted service returned HTTP 500 on 2026-11-10). Require exact label or synonym match, neutral form, class terms to class entries only, name-gate pass | F5 | up to 58 of 147 compound units (39%, upper bound); InChIKey for 49 (33%) | real: expected 3.5% wrong for exact matches, 20 to 30% for fallback rules (G). Gate on a pre-registered sample of at least 100 added ids per rule | medium |
| B2 | **Reference-strain tie-break** for `ambiguous_insufficient_margin` (`map_ids.py:6984`) | F6; 11 of 17 name-recoverable were the pipeline's own refused candidate | up to about 8 units (G) | medium: strain false positives (D-106) | low |
| B3 | **Never export a member accession for a class, complex or modified-form name** | F8; 48 of 192 id flags | removes about 25% of wrong-id flags; can only downgrade ids | low; needs D-entry | low |
| B4 | **Require every interaction endpoint to be a declared entity** | 01: up to 30 interactions (13%) | recovers quarantined rows | low to medium | low |
| B5 | **Sort set iteration in `build_qa_feedback`** (`pipeline.py:618-627`); pin `PYTHONHASHSEED` for evaluation | F12; 05: 8 of 15 payloads gave different Stage-2 feedback under 6 seeds | reduces noise; effect on outputs not established | changes Stage-2 prompt bytes; D-entry | low |
| B6 | **Write gene names and synonyms to PWML** | 06: up to 652 of 787 proteins | richer PWML | changes PWML output | low; owner yes required |
| B7 | **Promote second-pass reaction recovery** | 01: up to 20% more reactions, 71% verbatim-quoted | gain bounded | **high, defer**: F9 shows Stage-2 rows 46% true positives; only with a gold replay proving no loss | medium |
| B8 | **Channel-gating transport type in Stage-1 prompt** | 03: sour-taste shortfall | up to 1 to 2 processes per transport paper | touches `llm/prompts/`, outside the web-server track; biology change | medium; owner ruling |

### Group C. Recommend NOT doing, or leaving alone

| Item | Recommendation | Why |
|---|---|---|
| **Sequence homology as an identity source** | **Not recommended.** If the owner wants it, only as a flagged "inferred, not paper-stated" suggestion outside PWML id fields | 3 of 93 tested passed identity >= 50% and coverage >= 80%; bound is 0 to about 10 units (5%); 3 passers were single unreviewed TrEMBL hits with no gene name; 87 of 197 units are recovered by neither route; identity cannot discriminate paralogs (84% within one genome; 58% for a wrong cross-species hit); conflicts with D-106, D-108, D-117 unless ruled |
| **Near-copy transport merge** (the H+ duplicate question) | **Option 1 of the duplication assessment: no code change; treat tiles as a presentation matter.** Do A5 (lineage) first. Never use an accession-keyed entity merge | 9% of runs, 5.5% of rows; no file models individual ions, so reuse of one H+ record is already true; merging conflicts with D-034/D-035/D-036/F-043; evidence loss cannot be proved absent, so defer. Extending the D-104/D-109 compatibility rule to transports (option 2) stays an owner choice with an N>1-draw gate and re-baselined goldens |
| **Live KEGG API calls from the public server** | Do not do until a written licence answer exists | HIGH risk (07b) |

## 3. Recommendation on the new formats (sequence)

1. A2, A3, A4 first (cheap, add-only, remove invention from the data).
2. A8: clean model, then **SBML** (its rows are the best covered; compartments need nesting stored before the freeze, B-free if GO parents are tabulated: 15 names tabulated, 56 observed in the corpus).
3. **KGML** with option A of 07b: KEGG ids only when user-supplied, ChEBI/UniProt by default, attribution lines. 49% of compounds have no KEGG id; the owner must choose omit, placeholder or refuse (06 Q2). Do not call KEGG live.
4. BioPAX, SBGN-ML, GPML: after SBML; rows in 06 section 5.
5. Diagrams: depend on step order (derivable from shared compounds for 188 of 246 reactions), signed interactions, nesting and membrane-located transporters (1 of 22 transporters lies in a membrane location; the fix is upstream, an exporter fix would break D-015).

## 4. Questions the owner must rule on

1. Pipeline spend: approve a post-D-114 replay (A1) and an evaluation budget of about 100 runs per LLM-affecting change (A7).
2. D-023 "no SBML implementation": does it still bind? (06 Q1)
3. D-104 Amendment 1: may non-PWML formats carry unsigned relations? (06 Q3) Also: unmapped entities in KGML (omit, placeholder, refuse; 06 Q2), SBML compartment granularity (06 Q4), Unknown-backed complexes in other formats (06 Q5; 138 of 415 complexes are all-Unknown).
4. Protein ids: may a same-species, other-name TrEMBL record be offered as a flagged suggestion (D-106, D-108)? May a congeneric ortholog ever be used for annotation-poor organisms (44 `species_mismatch` units in 19 papers meet ORCH-739's revisit trigger by count)? (02 Q1, Q2) Recommendation: no to both for PWML; yes to showing a suggestion in a separate report.
5. Compound ids: may a public ChEBI/KEGG exact-name hit supply compound ids, or do compound ids stay PathBank-only? (02 Q3; B1)
6. May a narrowed accession ship as a flagged suggestion, or must the entity be id-less? (04 Q1; B3)
7. May the evaluator's `wrong_identity` flag change (bench edit under D-087)? 53% of 36 hand-checked flags were matcher artefacts. (04 Q2)
8. May Stage-1 prompting add a channel-gating transport type (B8), and may a no-op reaction be re-typed as a transport? (01)
9. Is 24% of PWML compounds without a database id acceptable? (01)
10. KEGG choice: option A, B (paid academic service provider licence) or C (skip). Ask Pathway Solutions through their inquiry form. (07b)
11. Gene names in PWML (B6): yes or no.
12. Paper baseline: legacy Aug-Sep runs, or a fresh post-D-114 replay (recommended).
13. Is the stale `pwml_qa.json` (ok in 61 of 61, current QA 36 of 61) a concern before the web server ships?

## 5. Expected combined effect (honest bounds)

- Recoverable without touching biology: the 40-run failure if still present (F1: up to 40% of PathWhiz-mode runs and 276 processes), title/reference fidelity (all files), non-PWML interactions (up to 35%).
- Recoverable with an owner ruling: compound ids up to about 39% of the 28% that lack them (about 11% of all compound units), protein ids about 5% of failed units by name-tie-break plus about 26% by in-taxon name search where the pipeline already judged a candidate.
- Not recoverable by lookup: 87 of 197 failed protein units (44%) by either route.
- Detectability: a gain below about 7 recall points or 12% entity count is not distinguishable from noise at 10 papers x 5 runs per arm.

## 6. Verification record

Three headline numbers were recomputed from raw files by an independent agent (`VERIFICATION.md`): 40 of 101 and 276 processes (agree); compound 147/517 vs 148/520 (agree); today's QA 36 pass of 61 (agree). The protein figure did not reproduce at first; the cause was a definition difference, now reconciled (F4). Other numbers in the assessments are single-agent and labelled M, L or G in their documents.
