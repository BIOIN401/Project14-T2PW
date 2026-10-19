# ORCH-747 — paper gold, interaction plumbing, and the prompt grid

**Status:** work delivered, NOT committed and NOT merged. Four product-owner decisions are
open (§ 7). Every run below went through the bounded wrapper with zero surviving
processes; reports are in `evidence/g11/ORCH-747/` (sequences 01–42).

**Base:** `e4348aaf`. Working tree only; nothing pushed.

---

## 1. Which prompt variant is best

**`v03_resolution_precedence`**, on the evidence available, and the margin over the other
new variants is one row. Read § 6 before relying on that.

It is the defensible pick because it keeps the only change with a decisive measured
effect — the relationship-vocabulary rule, which v02 introduced and every later variant
inherits — and is the only variant that removed the restatement v02 introduced, while
producing the leanest output of the ten with full evidence coverage.

What the grid does NOT support: picking between `v05`, `v08`, `v09` and `v10`. Their
differences are inside one draw's noise. And two variants were aimed at the wrong stage
(§ 5), so their instructions are dead weight at extraction time.

`v06` and `v07` are not ranked. They test opposite answers to a product question — is a
transcription step a reaction — and both did exactly what they were designed to do. That
decision is D-open-2 in § 7.

---

## 2. The prompt grid, measured

One draw per variant, same paper (PMC319880, `C:/t/eval-fulltext/PMC319880.r3.txt`,
sha256 `ca9e1d60…`), same model, temperature 0, 64,000 output tokens. Outputs in
`C:/t/prompt-grid/final/`; wrapper report `42-grid-64k.json`.

| variant | interactions | bad vocabulary | reactions | DNA entities | rows with no evidence |
|---|---|---|---|---|---|
| v01 committed baseline | 13 | **13** | 0 | 10 | 0 |
| v02 working tree (today's edit) | 13 | 0 | 0 | 16 | 0 |
| **v03 resolution precedence** | **11** | 0 | 0 | 9 | 0 |
| v04 alias rows banned | 15 | 0 | 0 | 8 | 0 |
| v05 one canonical name | 16 | 0 | 0 | 7 | 0 |
| v06 transcription as reaction | 23 | 0 | **2** | 13 | 0 |
| v07 transcription excluded | 18 | 0 | 0 | 13 | 0 |
| v08 regulatory example | 16 | 0 | 0 | 10 | 0 |
| v09 vocabulary by construction | 21 | 0 | 0 | 8 | 0 |
| v10 pre-output self-check | 21 | 0 | 0 | 11 | 0 |

**The one decisive result.** The committed prompt put 13 of 13 relationship values outside
the allowed four, writing `activates`, `binds`, `inhibits`, `competes with`. Every variant
carrying today's vocabulary rule produced zero violations. Prohibition was enough;
`v09`'s derive-from-class construction added nothing measurable on top of it.

**Nothing traded evidence for coverage.** No variant produced a single interaction without
an evidence quote, including those that tripled the row count. Raising resolution did not
loosen evidence discipline anywhere.

**Restatement.** Counting pairs that share a source and a relationship where one target's
name contains the other's: v02 produced 1, v03 produced 0, v06–v10 produced 1 each. v03
removed what v02 introduced, which is its design intent — but n=1.

---

## 3. The reason three grid attempts failed, and the production finding in it

**Your configured extraction budget is too small for this model.**
`OPENROUTER_EXTRACTION_MAX_TOKENS=16000` in `.env`; `streamlit_app.py:5243` defaults to
**64000**. This model spends output tokens on reasoning before emitting content, so at
16,000 it often exhausts the allowance and returns an empty body with
`finish_reason=length`. The pipeline then fails the leg after its retries.

One variant, three budgets, same paper:

| budget | outcome |
|---|---|
| 12,000 | empty, failed on all 3 attempts (`39-v04-probe.json`) |
| 16,000 | empty, failed on all 3 attempts (`40-grid-16k.json`, 9 of 10 variants failed) |
| 64,000 | succeeded: 21 interactions, 10 DNA entities, 0 rows without evidence (`41-probe-64k.json`) |

**This closes an open question in your own records.** Card `PF-746-02` documents the
identical signature from pilot-002 — calls billing their full budget across 9 requests,
every one ending `finish_reason=length` with empty content — and states that the
attribution to reasoning was inferred and that *"whether sending a reasoning setting would
have prevented it has not been tested."* The budget half is now tested: raising it fixes
the failures with no production change. Whether a reasoning setting would be cheaper than
a 64,000 budget is still untested.

**Consequence for you:** every app run at 16,000 carries this risk. It surfaces as a failed
extraction, not a silent degradation, so it is loud — but today's successful run was margin,
not safety. See D-open-1.

**Two wrong theories of mine, recorded so they are not re-run.** First, rate limiting,
because the first three calls succeeded and the rest failed; disproved by running the first
failing variant alone, where it failed identically. Second, that 12,000 was simply too
small; disproved when 16,000 made things *worse*, 9 of 10 failing including the baseline
that had passed at 12,000.

---

## 4. Paper gold for PW701720

Two golds, per the product owner's 2026-10-13 decision: the lab's curated PathBank record,
and a paper gold drafted from one pinned full text with a verbatim quote per row.
Artifacts and commands: `docs/evaluation_v2/pathbank_gold/README.md`.

**New code.** `src/t2pw/bench/eval_v2/pathbank_gold.py` reads a pathway read-only and emits
both the structural extract and a draft whose quotes are empty by construction.
`src/t2pw/bench/eval_v2/paper_gold.py` assembles a draft into a loadable supplement,
verifies every quote with the repository's own loader, and compares a blind draft against a
hint-given one. `report.evaluate` and `run_live` take an optional `supplement_path` so a
second gold is scored without touching the pinned four-case file.

**The accession trap.** `PW701720` is `super_pathway_visualizations.pw_id`, not
`pathways.id`. Looking it up as a pathway id silently returns an unrelated cardiolipin
pathway. All ten resolve; six are `private_pathway=1` and nine have `status NULL`, so they
can still change, which is why each extract records `updated_at` and a content hash.

**The curated pathway has no chemistry.** PW701720 holds 1 reaction and 7 interactions, and
its single reaction is the araC gene producing the AraC protein — expression, not a
metabolite conversion. The eval-v2 matcher scores two-sided metabolite conversions only, so
the reaction metrics have almost nothing to measure here. See D-open-3.

**Two curated edges are contradicted by the paper.** `PW_I016132` says the araO1 operator
*activates* the araC promoter; the paper's central result is that occupancy of that site
*represses* it by competing with RNA polymerase. The sign is a deliberate claim, not a
drawing convention: the same pathway uses `Inhibition` on `PW_I016136`. `PW_I016133`
encodes the model the paper's Discussion argues against. `PW_I016134` is left `not_stated`
rather than contradicted, because the paper is silent rather than opposed. Separately, CAP
is typed as a nucleic acid (`nucleic_acids` 275) although it is a protein and PathBank holds
`proteins` 10248, UniProt P0ACJ8. Classification is D-open-4.

**Blind versus hint-given drafting.** 15 interactions against 11, 8 matched, 7 quotes
verbatim-identical across two passes that never saw each other's work. The contamination
check is clean: no in-scope reaction in the hinted pass is unrelated to anything the blind
pass found.

---

## 5. Findings that changed what we built

**F-747-1 — The text extraction was wrong twice, silently.** The source is a scanned
two-column 1981 PNAS paper. The production PDF reader groups words into lines by vertical
position, and on a two-column page the columns share a baseline, so its output interleaves
them: every sentence breaks mid-word and continues with text from the other column. Quote
verification folds whitespace away, so a shredded text still hashes and still "verifies"
quotes that are themselves shredded. Three revisions were needed; `r3` is authoritative.
The decisive measurement: **every quote in all four drafts verifies against r3, including
the two round-0 drafts whose quotes failed against r1 and r2.** The drafters had been
quoting real sentences throughout; the extraction was wrong, not the drafts. Script
`evidence/orch747_pdf_columns.py`; revisions and reasons in
`pathbank_gold/PMC319880__source.json`. The production reader is still defective for any
two-column PDF, which is most papers.

**F-747-2 — The graph builder mis-typed every DNA entity (FIXED).**
`draft_graph._resolve_actor_kind` was given name sets for three of the five entity
registries, so a participant naming a nucleic acid or element collection took the
`compound` default. Because a node id carries its kind, that minted a second, mis-typed
node, and every edge attached to it while the correctly typed node sat with no edges.
Measured on a real run: all 8 DNA entities existed twice, 8 nucleic-acid nodes were
orphans, 8 of 15 compound nodes were DNA. Fixed by making the resolver consult the same map
that creates entity nodes. Proof: `evidence/orch747_graph_kind_base_proof.py`, verdict
satisfied. Tests: `tests/test_draft_graph_entity_kinds.py`.

**F-747-3 — The report hid most of the output (FIXED).** `reaction_summary.py` rendered
reactions and transports only, so a run carrying 1 reaction and 12 interactions displayed
the reaction alone. For a regulatory pathway that is a fraction of what the pipeline found,
and it reads as a recall failure that did not happen. Both text outputs now carry a
regulation section with per-row evidence, and the summary leads with a process census. The
research report's JSON gained a top-level `interactions` key; that pinned key set moved
deliberately, with the reason recorded in the test.

**F-747-4 — Interactions never reached the PWML (FIXED, but inert).** See § 6.

**F-747-5 — Two targeted defects originate after stage one.** The alias rows
(`cAMP is identical to cyclic AMP`, `relationship: SAME_AS`) and the split-name problem
(`CAP` beside `catabolite gene-activator protein`) do **not** come from extraction. Stage 1
produced zero alias rows in the measured run and zero in all ten grid draws; all four appear
only after the retrieval and merge stage, which also introduces the short forms. So
`v04_alias_rows_banned` and `v05_one_canonical_name` put their instructions in a prompt that
runs before the defect exists. The curator's own `SAME_AS` cleanup
(`pathway_curator_system.txt:19-21`) only catches rows whose endpoints are identical
strings, so it misses all four.

**F-747-6 — The prompt contradicted itself.** The extraction prompt's worked Example 2
emits `"relationship": "inhibits"` and `"activates"`, the exact values its own enumeration
forbids, each paired with a correct `class`. Today's edit added the prohibition and left the
counterexample. `v09` repairs it.

**F-747-7 — Five fields the prompt mandates are deleted before any reader sees them.**
`_clean_processes` rebuilds every process row from a key set carrying neither `class`,
`confidence`, `provenance`, `source_refs` nor `spontaneous` (`pipeline.py:2229-2265`
reactions, `:2383-2404` interactions). Consequence: `stage_one_boundary._paper_entry` tests
`row.get("provenance")` to set `review_required`, so merge gate 7 cannot fire on process
rows. Not fixed; out of scope here.

---

## 6. Interaction export to PWML

**Correct and currently inert.** `ir.py`, `writer.py` and `qa.py` now emit interactions in
PathBank's shape — `pwi-id`, both element blocks, the type, no spurious `name` — with a
visualization carrying left and right members, the arrowhead for Activation and the T-bar
for Inhibition. Interactions are validated for the first time. A real export with two
PathBank-shaped interactions and DNA endpoints is at `C:/t/orch747-pwml/pathway.pwml`, with
the withheld rows beside it.

Facts established from the live database rather than guessed:

| question | answer |
|---|---|
| how Inhibition is drawn | the right edge's `options` JSON: `start_flat_arrow: true` with a path, `start_arrow: false`. Activation is the exact inverse. Sampled 40 of each; 39/40 consistent |
| the member container's id column | protein complexes carry a **visualization** id; compound, nucleic acid and element collection carry a **location** id; bound carries a visualization id |
| whether `<interaction>` has a name | it does not; children are `id`, `interaction-type`, `pwi-id`, and the two element blocks |

The reference files contain only protein-complex and compound examples and no Inhibition at
all, so working from them alone would have produced the wrong field for the nucleic-acid
case, which is the one this pathway needs.

**Why inert.** Across the committed runs there are 271 interactions and **not one** is
spelled `activation` or `inhibition`. They are verb forms: `inhibits` 60, `represses` 10,
`activates` 6. The exporter accepts only the two canonical nouns, so on today's data it
withholds essentially everything. See D-open-1b.

**A side effect that reaches other sections.** A section whose reference template holds no
item now falls back to name-based typing. Measured on an interaction-free payload, base
versus tip: 30 lines differ, all in `bound-visualizations`, where five fields gain
`type="integer"`. PathBank's own files do type these fields, so this moves toward its shape
— but it changes existing output, which was not asked for. See D-open-1c.

**Two pinned digest tests moved** because of this change. The delta
(`evidence/orch747_ir_digest_delta.json`) shows 89 legs, 73 moved, every changed pointer
under the interaction annotations, with no error, pass or stop verdict moving anywhere.
Re-pinning was outside the implementer's file boundary and is still owed.

---

## 7. Open for the product owner

1. **Raise `OPENROUTER_EXTRACTION_MAX_TOKENS`** from 16,000 toward the 64,000 code default.
   Fixes the empty-extraction failures; changes what every run costs.
   1b. **Should the exporter map unambiguous verb forms** (`inhibits`, `represses` →
   Inhibition; `activates` → Activation)? Without it, interactions stay out of the PWML on
   all existing data. Mapping `inhibits` to Inhibition is the same word, not a guessed sign.
   1c. **Keep or narrow the `type="integer"` side effect** that reaches five other sections.
2. **Is a transcription or expression step a reaction?** `v06` says yes when both ends are
   named, `v07` says no and zero is the right answer. Both demonstrated.
3. **Scoring rule for pathways with no metabolite chemistry.** Six of the ten are
   signaling, physiological or disease. Interactions are recorded but not scored. Current
   default: report them as recall-only counts, no precision or F1.
4. **Classify the two contradicted curated edges** as `gold_data_defect`,
   `policy_disagreement`, or a legitimate difference between this paper and the two Schleif
   reviews the curator cited (neither of which has retrievable full text).

---

## 8. Gates

- Integration smoke: **508 passed**, measured 6 times across the day, zero survivors each.
- New tests: `test_eval_v2_pathbank_gold.py` (15), `test_eval_v2_paper_gold.py` (13),
  `test_draft_graph_entity_kinds.py` (5), `test_pwml_interaction_export.py` (10).
- Eval-v2 set: 89 passed + 1 pre-existing xfail. Graph consumers: 159 passed.
- PWML/export chunk: 771 passed, 8 skipped, 7 failed — 5 environmental (3 assert the DB is
  unreachable, 2 list untracked `runs_verify` legs), 2 are the digest pins above.
- G9 proofs: graph kind resolution, satisfied; interaction export, satisfied.
- One pinned baseline moved deliberately: the research report's JSON key set, reason in the
  test.

## 9. Cost

Roughly 30 billed extraction calls across four grid attempts, three of which failed on
configuration rather than on the prompts. The budget finding was worth the spend; the grid
itself cost about three times what it should have.

---

## 10. Round 2 — 16 generalizable variants across 3 papers, scored against gold

Added 2026-10-15 (late). Round 1 ran one paper and counted rows. Round 2 runs three
contrasting papers and scores reaction precision, recall and F1 through
`reactions.score_leg`, the evaluator's own matcher, on two papers whose gold is
**certified exhaustive** with a verbatim quote per reaction.

Variants: `docs/evaluation_v2/prompt_variants_r2/` (`r2_01`–`r2_16`, plus `AUDIT_R2.md`,
`VARIANTS_R2.md`, `SCHEMA_CHECK_R2.md`). All 16 verified schema-identical to the baseline,
86 declared keys, checked independently of the author. All 16 derive from `v03`, round 1's
winner, which is carried as the control `r2_01`.

Papers, chosen to be unlike each other rather than for convenience:

| paper | kind | gold |
|---|---|---|
| PMC12096016 | enterobactin biosynthesis, *E. coli* | 5 in-scope reactions, **certified** |
| PMC12421875 | menaquinone biosynthesis, *L. lactis* | 9 in-scope reactions, **certified** |
| PMC319880 | araC autoregulation, DNA footprinting | 1 reaction + 15 interactions, NOT certified |

48 draws, one per variant per paper, 64,000 output tokens, temperature 0. Evidence
`43-r2-*.json`, `44-r2-*.json`, `45-r2-*.json`. Outputs `C:/t/prompt-grid/r2/`; scores
`C:/t/prompt-grid/r2/scores.json`.

### 10.1 The result that reframes the problem: precision is already perfect

**Every variant, on both certified papers, claimed only reactions that matched gold.**
4 claimed / 4 matched, 9 claimed / 9 matched, 3 claimed / 3 matched, and so on for all 32
metabolic draws. Not one false reaction in the entire grid.

So the entire spread in F1 is **recall**. The model is not inventing chemistry; it is
missing chemistry the paper states. Every prompt rule aimed at suppressing bad rows is
aimed at a problem that does not exist on this evidence, and the only rules that can help
are ones that make the model find more of what is there without starting to invent.

This also means the 0 evidence-free rows across all 48 draws is not the whole story: the
rows that exist are sound, and the loss is upstream of the writing.

### 10.2 F1 by paper, ranked by the WORST paper

Ranked by worst paper on purpose: the goal is one prompt that serves every kind, and an
average rewards a variant that wins on one type by losing another.

| variant | PMC12096016 | PMC12421875 | matched | worst F1 |
|---|---|---|---|---|
| **`r2_01_control_v03`** (control) | 0.889 (4/5) | **1.000 (9/9)** | 13/14 | **0.889** |
| **`r2_02_classify_then_branch`** | 0.889 (4/5) | **1.000 (9/9)** | 13/14 | **0.889** |
| **`r2_14_pre_output_self_check`** | 0.889 (4/5) | **1.000 (9/9)** | 13/14 | **0.889** |
| `r2_04_kind_recheck` | 0.889 (4/5) | 0.875 (7/9) | 11/14 | 0.875 |
| `r2_05_mixed_case_procedure` | **1.000 (5/5)** | 0.875 (7/9) | 12/14 | 0.875 |
| `r2_13_model_stated_example` | **1.000 (5/5)** | 0.875 (7/9) | 12/14 | 0.875 |
| `r2_15_core_plus_typed_overlay` | **1.000 (5/5)** | 0.875 (7/9) | 12/14 | 0.875 |
| `r2_09_requested_pathway_boundary` | 0.750 (3/5) | 1.000 (9/9) | 12/14 | 0.750 |
| `r2_10_legend_vs_body` | 0.750 (3/5) | 1.000 (9/9) | 12/14 | 0.750 |
| `r2_03_conditional_routing` | 0.750 (3/5) | 0.875 (7/9) | 10/14 | 0.750 |
| `r2_08_summary_not_a_step` | 0.750 (3/5) | 0.875 (7/9) | 10/14 | 0.750 |
| `r2_16_integrated` | 0.750 (3/5) | 0.875 (7/9) | 10/14 | 0.750 |
| `r2_11_example2_repair` | 0.889 (4/5) | 0.615 (4/9) | 8/14 | 0.615 |
| `r2_12_regulatory_example` | 0.750 (3/5) | 0.615 (4/9) | 7/14 | 0.615 |
| `r2_06_bucket_two_sided_gate` | 0.889 (4/5) | **0.364 (2/9)** | 6/14 | **0.364** |
| `r2_07_conversion_sweep` | 0.889 (4/5) | **0.364 (2/9)** | 6/14 | **0.364** |

**No variant beat the control.** Three tie it; twelve are worse. Fifteen new ideas, and the
best outcome is parity with round 1's winner.

**No variant gets both papers right.** The three at the top take 4 of 5 and 9 of 9; the
three below take 5 of 5 and 7 of 9. That is a real tradeoff, not noise in one direction.

**Two variants are actively harmful**, losing 7 of 9 reactions on the menaquinone paper.
`r2_06` (gate buckets by grammar) and `r2_07` (sweep for indirectly stated conversions)
are the two the author predicted would be the biggest winners. See § 10.4.

### 10.3 The araC paper discriminates nothing, and that is itself the answer

**All 16 variants scored recall 0 on its single gold reaction.** None extracted the
transcription step. F1 is undefined rather than zero, because no variant claimed any
reaction at all, so precision has an empty denominator — which is why that column is blank
in the generated table rather than showing zeros.

Interactions ranged 11–21 with **zero** rows lacking evidence in every variant, and
restatements of 0 or 1. So the regulatory paper is reproducing round 1's picture and adding
nothing new: until **D-open-2** (is a transcription step a reaction) is ruled, this paper
cannot separate prompts on the scored metric. No r2 variant legislates transcription in
either direction, deliberately, because the question is the product owner's.

### 10.4 A confident prediction, measured and wrong

`AUDIT_R2.md` predicted routing would not be the main lever and that the gains would come
from kind-neutral shape rules, naming `r2_06` as *"the only variant expected positive on
all three"* and `r2_07` as the only rule reaching PMC12421875's rate-law steps.

Measured: `r2_06` and `r2_07` are the two **worst** variants in the grid, and the paper they
were predicted to help is the one where they collapse, 2 of 9. Meanwhile the router
`r2_02_classify_then_branch` ties for first. Routing did not lose; the shape rules did.

Recorded because the reasoning was good and the result contradicts it. An argument from how
a rule ought to read is not evidence about how a model behaves.

### 10.5 What this round does NOT settle

One draw per variant per paper. The top group is separated by a single reaction, and the
run-to-run variance already measured on this model is larger than that: the same prompt on
the same paper produced 2 DNA entities in one draw and 9 in another. **The top seven are
tied until repeated.** The 2-of-9 and 4-of-9 collapses are large enough to trust.

**Round 3 is running**: three draws of the six-variant shortlist (`r2_01`, `r2_02`,
`r2_05`, `r2_13`, `r2_14`, `r2_15`) on the two certified papers, 36 draws, evidence
`46-r3-*.json` / `47-r3-*.json`, outputs `C:/t/prompt-grid/r3/`. araC is dropped from it
because § 10.3 shows it discriminates nothing on the scored metric.

### 10.6 What to do with this

1. **Keep `v03` / `r2_01` as the production candidate** unless round 3 separates
   `r2_02` or `r2_14` from it. Nothing has earned a change.
2. **Reject `r2_06` and `r2_07`.** Large, directional recall loss on a certified paper.
3. **Stop writing rules that suppress rows.** Precision is 1.000 everywhere; the headroom
   is entirely recall, and the two suppression-shaped variants did the most damage.
4. **Rule D-open-2.** It is now blocking: one of the three test papers cannot contribute
   to the scored comparison until transcription is decided.
5. The recall ceiling is ~13 of 14 across both papers. Finding the last reactions is a
   different problem from the ones round 2 tested, and may not be a prompt problem at all.

---

## 11. Round 3 — three draws each, and why round 2's ranking did not survive

36 draws: 6 shortlisted variants x 2 certified papers x 3 draws, 64,000 tokens,
temperature 0. Evidence `46-r3-*.json`, `47-r3-*.json`. Outputs `C:/t/prompt-grid/r3/`.

### 11.1 The dominant effect is run-to-run variance, not the prompt

Matched reactions, per draw, same prompt and same paper each time:

| variant | PMC12096016 (5 gold) | PMC12421875 (9 gold) | mean F1 | worst draw |
|---|---|---|---|---|
| **`r2_13_model_stated_example`** | 5 / 4 / 4 | 7 / 7 / 9 | **0.921** | **0.875** |
| `r2_02_classify_then_branch` | 5 / 2 / 5 | 9 / 7 / 7 | 0.887 | 0.571 |
| `r2_01_control_v03` (round 2's joint winner) | 3 / 5 / 4 | 9 / 7 / 4 | 0.855 | 0.615 |
| `r2_14_pre_output_self_check` | 5 / 2 / 5 | 9 / 2 / 9 | 0.823 | 0.364 |
| `r2_15_core_plus_typed_overlay` | 3 / 4 / 3 | 9 / 4 / 7 | 0.806 | 0.571 |
| `r2_05_mixed_case_procedure` | 2 / 4 / 5 | 2 / 9 / 9 | 0.789 | 0.364 |

**One prompt, one paper, three draws: 9 then 7 then 4 matched reactions.** Another gives
2, 9, 9. The within-variant spread is as large as the entire between-variant spread, which
means **round 2's one-draw ranking was noise**, as § 10.5 warned it might be. Round 2's
joint winner is third here, and two variants that looked respectable there have single
draws at F1 0.364.

This is the same instability recorded in § 10.5 and in CLAUDE.md's standing warning that
identical legs give materially different Stage-1 draws at temperature 0. It is now
quantified on a scored metric: **the noise floor is roughly ±0.25 F1 per draw.**

### 11.2 One variant is genuinely more stable

`r2_13_model_stated_example` leads on the mean (0.921) and is the only variant whose
**worst** single draw stays above 0.87; every other has a draw at or below 0.615. Its
per-draw counts, 5/4/4 and 7/7/9, are the tightest in the grid.

Its mechanism is coherent rather than lucky. Every worked example in the prompt shows a
conversion the paper states plainly; this variant adds one for conversions stated
indirectly — through letter-assigned equations and rate laws, and through products named
only inside an enzyme name. PMC12421875 states MenH through MenI exactly that way, through
its steady-state model's equations (5)–(11), which `docs/evaluation_v2/README.md` § 1 records
for that case. So the variant targets the specific reading failure those papers exhibit.

### 11.3 A correction to § 10.1

Round 2 reported precision 1.000 on every metabolic draw. Over 36 draws that does not
hold: **2 of 36 fell below 1.000, minimum 0.800.** False reactions are rare but real, so
"the model never invents chemistry" was an overstatement from too few draws. The direction
of § 10.1 stands — recall is where the headroom is, by a wide margin — but the absolute
claim was wrong and is withdrawn here.

**What does hold, now over 84 draws (48 + 36): zero rows without an evidence quote.**
Not one, in any variant, on any paper. That is the strongest result in the whole
experiment.

### 11.4 Round 4 is running

Five draws each of `r2_13_model_stated_example` against the control `r2_01_control_v03`,
both certified papers, 20 draws. Evidence `48-r4-*.json`, `49-r4-*.json`; outputs
`C:/t/prompt-grid/r4/`. With a ±0.25 noise floor, three draws cannot separate a 0.07
difference in means; five per arm on a head-to-head is the cheapest test that can.

### 11.5 Standing recommendation

1. **`r2_13_model_stated_example` is the candidate to adopt**, pending round 4. It leads
   on the mean and is the only variant that does not have a bad draw.
2. **Do not read any single-draw comparison again.** The measured noise floor is ±0.25 F1.
   Any future prompt claim needs at least three draws per arm, and five to separate
   near-neighbours.
3. **§ 10.6 item 1 is superseded**: the control is no longer the best candidate.
4. **§ 10.6 items 2–5 stand**, including that `r2_06` and `r2_07` are rejected and that
   `D-open-2` is blocking.

---

## 12. Round 4 — the head-to-head, and the finding that beats every prompt

20 draws: 2 variants x 2 certified papers x 5 draws. Evidence `48-r4-*.json`,
`49-r4-*.json`. Outputs `C:/t/prompt-grid/r4/`.

### 12.1 No prompt is demonstrably better than the control

| | enterobactin (5 gold) | menaquinone (9 gold) | overall mean F1 |
|---|---|---|---|
| `r2_01_control_v03` | 5,5,4,5,5 → F1 0.978 | 4,7,4,9,7 → F1 0.796 | **0.887** |
| `r2_13_model_stated_example` | 4,5,4,4,5 → F1 0.933 | 9,9,4,9,7 → F1 0.898 | **0.916** |

Difference in means: **0.029**, against a pooled standard deviation of **0.135**. That is
**0.21 of one standard deviation**, with 10 draws per arm — nowhere near a real effect.
Both arms share the same worst draw, F1 0.615.

It is also a trade rather than a win: the **control is better** on the enterobactin paper
(0.978 vs 0.933) and worse on the menaquinone paper (0.796 vs 0.898). The direction matches
`r2_13`'s mechanism, since the menaquinone paper is the one stating steps through
letter-assigned equations, but the overall difference does not survive the noise.

**So the answer to "which prompt is best" is: on this evidence, none of them.** After 68
scored draws over four rounds, no variant separates from round 1's winner. § 11.5 item 1,
which named `r2_13` the candidate to adopt, is **withdrawn**.

### 12.2 Pooling draws reaches recall 1.000 at precision 1.000

The variance is not only a measurement nuisance. Because precision is near-perfect, each
draw misses *different* reactions, so the misses are recoverable. Taking the UNION of the
gold reactions matched across the 5 draws:

| variant | paper | per draw | union | gold | union recall |
|---|---|---|---|---|---|
| `r2_01_control_v03` | PMC12096016 | 5,5,4,5,5 | **5** | 5 | **1.000** |
| `r2_01_control_v03` | PMC12421875 | 4,7,4,9,7 | **9** | 9 | **1.000** |
| `r2_13_model_stated_example` | PMC12096016 | 4,5,4,4,5 | **5** | 5 | **1.000** |
| `r2_13_model_stated_example` | PMC12421875 | 9,9,4,9,7 | **9** | 9 | **1.000** |

And pooling costs nothing in precision on this evidence: over the 115 pooled rows across
all four arms, **true positives 115, claimed 115, precision 1.000**. Not one false reaction
entered the pool.

**Expected recall against the number of pooled draws**, averaged over every combination of
that size:

| arm | N=1 | N=2 | N=3 | N=4 | N=5 |
|---|---|---|---|---|---|
| control, enterobactin | 0.960 | 1.000 | 1.000 | 1.000 | 1.000 |
| control, menaquinone | 0.689 | 0.922 | **1.000** | 1.000 | 1.000 |
| r2_13, enterobactin | 0.880 | 0.940 | 0.980 | 1.000 | 1.000 |
| r2_13, menaquinone | 0.844 | 1.000 | 1.000 | 1.000 | 1.000 |

**Three draws reaches recall 1.000 in three of four arms and 0.98 in the fourth.** The
weakest single-draw arm, the control on menaquinone at 0.689, rises to 1.000 by N=3 — a
gain of 0.31 recall. The best prompt difference available was 0.03.

**So the engineering lever is ensembling, not prompt wording, by an order of magnitude.**

### 12.3 Caveats, stated plainly

- Two papers, both metabolic, both with certified gold. araC is excluded because no variant
  extracts its single reaction at all (§ 10.3), so nothing here says ensembling helps a
  regulatory paper.
- Precision was 1.000 on all 20 draws here, but § 11.3 recorded 2 of 36 round-3 draws at
  0.800. Pooling more draws will eventually pool a false row. The cost is small and
  one-directional, and it should be re-measured at the N actually adopted.
- Union is the most permissive pooling rule. A majority rule would trade some of the recall
  gain for protection against the rare false row; it was not tested.
- Three draws triples the cost and the wall-clock of every extraction. That is the real
  price, and it is a product decision.

### 12.4 Recommendation after four rounds

1. **Do not change the extraction prompt on this evidence.** Keep `v03` /
   `r2_01_control_v03`. Supersedes § 10.6 item 1 and § 11.5 item 1.
2. **Pursue ensembling instead: run Stage 1 three times and union the reactions.**
   Measured gain up to +0.31 recall on the weakest arm, versus +0.03 for the best prompt
   change. This is a pipeline change, not a prompt change, and it is the single largest
   available improvement found in four rounds.
3. **Keep rejecting `r2_06` and `r2_07`** (§ 10.6 item 2). Their losses, 7 of 9 reactions,
   are far outside the noise floor.
4. **Any future prompt claim needs ≥3 draws per arm, and 5 to separate near-neighbours.**
   Measured noise floor: ±0.25 F1 on a single draw.
5. **`D-open-2` is still blocking** one of the three papers (§ 10.3).

### 12.5 The standing result across all 104 draws

**Not one row without a verbatim evidence quote**, in any variant, on any paper, in any
round. Raising extraction resolution, adding routers, adding examples and adding
self-checks never once bought coverage by loosening evidence. That is the one thing this
experiment established beyond argument.
