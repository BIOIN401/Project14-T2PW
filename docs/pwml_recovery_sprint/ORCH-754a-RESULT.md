# ORCH-754a — the paper as the dictionary of a pooled extraction (D-104), plus findings 1 and 4

**Status:** delivered on `orch754/a-ensemble-merge` (base `f9892cdf`), not merged, not pushed.

- Round 1: code `772bd3df` and `cf84626b`, docs and evidence `6c75112f`.
- Round 2, which implements the product owner's rulings of 2026-10-20 on questions 1 and 2: code `1cfae4d3`, docs and evidence in the commit after it.
- Bounded-run reports `evidence/g11/ORCH-754a/01`–`86`, pins `evidence/g11/pin/ORCH-754a/`.
- Offline only: **no LLM call and no pipeline run anywhere in this task.**

**Headline (current tip).**

- Chlorophyll pools **41 → 29** reactions: 11 at 3/3 support, 2 at 2/3, 16 at 1/3 (all kept).
  - 11 merges are flagged `undefined_state_treated_as_absent`, naming `'plastidic state'`.
  - Two steps stay split, on enzymes the paper dictionary cannot bridge: `PPO`, and the ambiguous `POR`.
- Gold is unchanged: enterobactin 4/5 at precision 0.8, menaquinone 9/9 at precision 1.0. That holds at base, before the rulings, and now.
- The chimera fix (finding 1), the `.env` isolation (finding 4) and the human-only support marks (§ 1.6) all still hold.

**Read § 0 first.** It records the round-2 rulings and what they changed. §§ 1–8 are the round-1 record, still accurate except where § 0 supersedes them: § 1.5's state and enzyme rules, the § 2.1 numbers, and § 7 items 1–3.

---

## 0. Round 2: the rulings of 2026-10-20

### 0.1 Q1 (product owner): an undefined biological state counts as no state stated

**Ruling.**

- A row whose `biological_state` names a state its own draw never defines is treated as having no state, for compatibility.
- A defined state against an undefined one is therefore compatible. Two defined states that differ still conflict.
- Every merge that relied on this is flagged.
- The survivor keeps a defined state. The row with the defined state is preferred as survivor.

**Implementation** (`stage_one_ensemble._conflict`, `_undefined_state`, `_prefers`, `_fold`):

- *Which states are compared.* `_conflict` compares only states that resolve to a `biological_states` row of the same payload (`_defined`). Defined states are compared by compartment, cell type, tissue and species, as before.
- *The flag.* `_undefined_state` names the undefined label(s) whenever both rows state something different and at least one is undefined. The merge entry then carries `flag: {reason: "undefined_state_treated_as_absent", undefined_state: "'plastidic state'"}`. The side-file `.txt` prints `FLAG (review): undefined_state_treated_as_absent -- 'plastidic state' is not defined in its draw's biological_states`. Rule-3 merges carry and print the same flag.
- *Survivor choice.* `_prefers` makes the joining row the survivor when it has a defined state and the current survivor does not. It takes the first row's place in the pooled list, and the previous survivor is folded into it (`became_survivor`, `previous_survivor`). If the survivor is kept but lacks a defined state, `_fold` copies the other row's defined label (`state_taken`).
- *Tests:*
  - `test_ruling_q1_an_undefined_state_counts_as_absent_and_the_merge_is_flagged`: the defined-state row survives with its own quote, the flag names the state, and a third row in a different defined compartment stays apart;
  - `test_ruling_q1_a_defined_state_is_carried_onto_a_survivor_without_one`;
  - `test_two_defined_states_are_compared_by_compartment`.

**Note.** The survivor preference also applies when the other row states no state at all. That is the same rule, "the row that states more survives". It moved two menaquinone survivors (MenH and MenE rows of draw 3 over draw 2's stateless rows), and the gold score is unchanged.

### 0.2 Q2 (product owner): enzyme supersets merge, and the larger set is kept

**Ruling.**

- Two enzyme sets, one a subset of the other after the paper dictionary and the identity rules, are compatible.
- The survivor carries the larger set, but each extra enzyme only on a verbatim quote that supports it.
- The larger-set row is preferred as survivor.
- Overlapping sets that do not nest still conflict.

**Implementation.**

- `_conflict` refuses only non-nested enzyme sets: A+B vs A+C, and A vs B.
- `_prefers` makes the larger-set row the survivor when both rows' states are equally defined. A defined state outranks a larger enzyme set.
- When the survivor is kept, `_fold` copies each extra enzyme entry only if the merged row's own quote, or that enzyme entry's own `evidence`, names it. The quote is recorded with it (`enzymes_added: [{enzyme, supporting_quote}]`). Otherwise the enzyme is listed in `enzymes_withheld` with the donor quote.
- The record's claimed profile now carries the union of enzyme sets, so a third row is judged against everything the record already absorbed.
- Rule 3 uses the same mechanics, with enzymes compared by identity.
- *Tests:*
  - `test_ruling_q2_mg_chelatase_and_gun4_merge_and_keep_gun4`: the GUN4 row survives, keeping its own quote that names GUN4;
  - `test_ruling_q2_an_extra_enzyme_no_quote_names_is_withheld`;
  - `test_ruling_q2_overlapping_enzyme_sets_that_do_not_nest_stay_apart`;
  - `test_rule3_an_enzyme_superset_by_identity_merges_and_keeps_the_larger_set`.

### 0.3 Orchestrator rulings on Q3 to Q7

- **Q3, `shared_locant`: accepted provisionally, pending independent review.**
  - *The exact rule.* Let Schwartz–Hearst accept a pair from the `long form (SF)` pattern; the swapped `SF (long)` pattern never qualifies. If the text right after the closing parenthesis matches `\s*([IVX]{1,4}|[A-Za-z])(?![\w-])`, a second pair `(SF + " " + T, long + " " + T)` is registered with pattern `shared_locant`. `T` is a roman numeral of 1–4 characters from `I`, `V`, `X`, or one letter, not followed by a word character or hyphen. Only the locant the paper actually writes after that parenthesis is registered.
  - *Negative test.* `test_shared_locant_never_bridges_a_different_locant`: `Urogen III` → `uroporphyrinogen III`, but `Urogen I` and `Urogen IV` are untouched, and a `Urogen I` row never pools with `uroporphyrinogen III`.
- **Q4** (reaction-evidence UI reads the side file): deferred, not in this lane.
- **Q5** (1/N drives `release_status`): no, not in this lane.
- **Q6** (rule-3 lock record in `quarantined_locked_reactions`): accepted. It matches `dedupe_processes`.
- **Q7** (menaquinone `DHNA` interaction endpoint renamed): noted for integration with ORCH-754b.
- **Instruction 7's withdrawal** is recorded by the orchestrator in `ORCH-754-RESULT.md`. It needs no DECISIONS entry, so the item in § 7.12 is withdrawn.

### 0.4 Measurements (`evidence/orch754a_ensemble_merge.{py,json,txt}`, run `g11/ORCH-754a/84`)

`strict_before_rulings` is the merge as committed at `cf84626b`, loaded from `git show`.

| pooling | chlorophyll | enterobactin (5 gold) | menaquinone (9 gold) |
|---|---|---|---|
| base `f9892cdf` | 41 | 4/5, P 0.8 | 9/9, P 1.0 |
| `cf84626b` (strict, before the rulings) | 41 | 4/5, P 0.8 | 9/9, P 1.0 |
| **tip, delivered** | **29** | **4/5, P 0.8** | **9/9, P 1.0** |
| ORCH-751 union (reference) | -- | 4/5 | 9/9 |

The round-1 counterfactual (an undefined state treated as no state, Q1 alone) gave 30. Q2 merges the Mg-chelatase/GUN4 pair as well, which gives 29.

**Chlorophyll per-merge table** (the full table, naming the bridging rule of each row, is in `orch754a_ensemble_merge.txt`; the hand-comparison text is `orch754a_chlorophyll_side_file.txt`):

| rows | support | merges, draw → rule (`*` = flagged undefined state) |
|---|---|---|
| Glu-tRNAGlu → GSA, GSA → ALA, ALA → PBG, HMB → Urogen III, Urogen III → Coprogen III, Coprogen III → Protogen IX, Mg-Proto IX → Mg-Proto ME, Chlide a → Chlide b | 3/3 | d2 → abbreviation\*, d3 → exact |
| PBG → HMB | 3/3 | d2 → abbreviation+typo\*, d3 → exact |
| Proto IX → Mg-Proto IX (Mg-chelatase) | 3/3 | d2 → abbreviation\*, d3 → exact; **d3 became the survivor** (larger enzyme set, GUN4, its own quote kept) |
| Mg-Proto ME → DV-Pchlide | 3/3 | d2 → abbreviation\*, d3 → abbreviation |
| Protogen IX → Proto IX | 2/3 + 1/3 | d3 → exact; d2 kept apart: `different enzymes {ppo} vs {protogen ix oxidase ppo}` |
| DV-Pchlide → DV-Chlide | 2/3 + 1/3 | d3 → abbreviation; d2 kept apart: `different enzymes {por} vs {pchlide oxidoreductase por}` |
| 14 single-draw rows (heme b/protoheme variants, MV-Chlide a variants, GGPP/phytyl chain, Chl a → Chl b, etc.) | 1/3 | none; kept and flagged `review_required` |

**Totals:** 24 merges, 11 flagged, 1 survivor replaced, 2 kept apart. No enzyme, cofactor or state had to be carried, because every survivor already named what it absorbed.

**Gold, before and after.** Enterobactin is 4/5 at P 0.8, with 5 rows claimed: 4 target plus 1 `paper_stated_off_pathway`, the LDH row. Menaquinone is 9/9 at P 1.0. Both identical in all three poolings. No recall or precision drop.

### 0.5 Gates, round 2 (tip `1cfae4d3`; the battery ran on exactly that content before it was committed)

Every run was bounded, with zero survivors. G11 check: **83 artifacts, 0 non-compliant**, each with a pin verdict, none refused.

| gate | default env | `T2PW_EXTRACTION_DRAWS=3` |
|---|---|---|
| focused (`test_stage_one_ensemble.py` + `test_stage_one_paper_dictionary.py`) | **87 passed** (`evidence/g11/ORCH-754a/70`) | **87 passed** (`/76`) |
| affected, pipeline and summary (the 12 files of § 3.1) | **203 passed** (`/71`) | **203 passed** (`/77`) |
| affected, mapping (the 17 files of § 3.1) | 397 passed, 1 skipped, **1 failed = pre-existing c075** (`/72`) | same (`/78`) |
| AppTest `test_c055_rag_loop_wiring.py`, alone | **14 passed** (`/73`) | **14 passed** (`/79`) |
| AppTest `test_streamlit_quarantine_boundary.py`, alone | **23 passed** (`/74`) | **23 passed** (`/80`) |
| **SMOKE** (22 files) | **508 passed** (`/75`) | **508 passed** (`/81`) |
| without conftest, the reviewer's condition | -- | pipeline: 24 failed, all `test_early_failure_replay` (`/82`); mapping: only c075 (`/83`) |

| proof | report | result |
|---|---|---|
| G9 base pre-run, final test file, src modules restored to `f9892cdf` | `evidence/g11/ORCH-754a/69` | **6 failed, 22 passed**: the chlorophyll steps, IDH ×2, ATP/ADP + Mg/Mn, enzymes/defined states, unsupported donation |
| N=1 `run_extraction_pipeline` identical to base | `/85`, `evidence/orch754a_n1_identity_verdict.json` | **N=1 IDENTICAL to base** |
| model-bound text: base-rendered goldens | `/86` (and `/65`) | N=1 `d171a3b8…`, pooled `ab01b0dd…`, identical in both runs |
| gold replays and chlorophyll measurement | `/84` (and `/67`) | § 0.4 |

**TEST_MATRIX delta, round 2.** The focused selection is now 87 tests: `test_stage_one_ensemble.py` 28, `test_stage_one_paper_dictionary.py` 59. SMOKE stays 508. Two goldens in `tests/fixtures/orch754a/` were regenerated, because the delivered pooled payload changed: N=1 draw sha256 `d171a3b8…` (unchanged), pooled `ab01b0dd…` (was `7258fe0a…`). Both were rendered by the base modules.

### 0.6 Still not settled (round 2)

1. **Survivor replacement rewrites the first row's place.** When a later row wins on Q1 or Q2, the earlier survivor's quote is no longer the row's quote. It survives only through the merge record in the side file. That is what the ruling asks ("its own evidence is kept"), recorded here because it changes "first occurrence wins".
2. **A defined state outranks a larger enzyme set** when they point to different rows. The other row's extra enzymes then need a quote (`enzymes_withheld` otherwise). The PO may prefer the opposite order.
3. **Enzyme names that are not dictionary pairs still split** (`PPO` vs `Protogen IX oxidase (PPO)`, and the ambiguous `POR`), as in round 1 § 7.2.
4. Round-1 § 7 items 4–11 are unchanged, except item 6 (Q6, accepted) and item 12 (withdrawn).

---

# Round 1 record (tip `cf84626b`)

**Round-1 headline (superseded by § 0 where they differ).**

- The D-104 merge rules are built and pinned: Schwartz–Hearst abbreviation pairs, typo snapping, the long-form rewrite, support counts, a bridging rule recorded on every merge, and the post-mapping database-identity merge.
- On chlorophyll the pool stays at **41 → 41**, and the reason is now a deliberate fail-closed rule, not the dictionary. Finding 1's compatibility rule keeps draw 2 apart: every one of draw 2's reactions names a biological state (`plastidic state`) that its payload never defines, because the draw wrote the bucket as `biogical_states`. With that one rule relaxed the pool is **30** (§ 2.1, counterfactual). That relaxation is an open question for the product owner (§ 7.1).
- Enterobactin and menaquinone are unchanged: **4/5 at precision 0.8**, and **9/9 at precision 1.0**.
- The reviewer's IDH chimera is refused, as is any merge across a cofactor-role, enzyme or state conflict. No support mark or merge note reaches any text a model reads (§ 1.6).

---

## 1. What changed

### 1.1 Files

| file | change |
|---|---|
| `src/t2pw/pipeline/paper_lexicon.py` | **new**. Rules 1 and 2, standard library only |
| `src/t2pw/pipeline/stage_one_ensemble.py` | rules 1, 2, 4, 5, 6 in `merge_draws`; finding 1's compatibility rule; the side-file text; rule 3 (`merge_on_mapped_identity`) |
| `src/t2pw/pipeline/pipeline.py` | **one line**: `run_extraction_pipeline` passes its input text to `run_ensemble(..., paper_text=)` |
| `src/t2pw/mapping/map_ids.py` | **two adjacent lines** at the end of `map_payload`: a function-local import and the rule-3 call. Nothing else in that file |
| `tests/conftest.py` | **new** (finding 4) |
| `tests/test_stage_one_ensemble.py` | corrections with base-module proofs; one pin moved deliberately (§ 4) |
| `tests/test_stage_one_paper_dictionary.py` | **new acceptance tests** (G9 label in the module docstring) |
| `tests/fixtures/orch754a/` | fixture payloads and goldens rendered by the base modules (§ 1.6) |
| `docs/pwml_recovery_sprint/evidence/orch754a_inputs/` | the measurement and test inputs: the enterobactin and menaquinone draws plus `index.json`, and all three paper texts. Each paper's sha256 is checked against its draws' `paper_sha256` |

Two files were changed by this task and then restored to their base contents under the 2026-10-20 ruling (§ 1.6): `reaction_summary.py`, and the `_clean_processes` carrier and `build_and_save_draft_graph` line in `pipeline.py`. Commit `772bd3df` still contains them, and `cf84626b` reverts them. Measured: `git diff f9892cdf -- src/t2pw/pipeline/reaction_summary.py` is empty.

### 1.2 Rule 1: abbreviations (`paper_lexicon.abbreviation_pairs`, `PaperLexicon.long_form`)

The algorithm is Schwartz–Hearst, restated from its reference implementation:

- the `long form (SF)` pattern, and the `SF (long form)` swap when the parenthetical has more than two words;
- the right-to-left character match whose first character must start a word;
- the shortest long form, and the validity limits (long-form words ≤ min(2·|SF|, |SF|+5), |SF| ≤ 10).

Four deliberate choices:

1. **Short forms match case-sensitively.** PMC12096016 defines `PPI` as "protein–protein interaction", and a case-insensitive match would have renamed the draws' `PPi` (pyrophosphate) to it. NFKC, dash style and whitespace are folded; nothing else is.
2. **Ambiguous short forms bridge nothing.** A short form the paper defines with two long forms (by strict key) bridges nothing. On chlorophyll these are `POR` (Pchlide oxidoreductase / protochlorophyllide oxidoreductase) and `TPB`. Enterobactin and menaquinone have none.
3. **One labelled extension, `shared_locant`.** The paper writes `uroporphyrinogen (Urogen) III`, so `Urogen III` ↔ `uroporphyrinogen III` is registered: a roman numeral or a single letter right after the parenthesis extends both forms. This is the only thing that bridges `Urogen III`, `Coprogen III`, `Protogen IX` and `Proto IX`. Every record carries the pattern, so the product owner can veto this extension alone (§ 7.3).
4. **Long forms are written as the paper spells them, with ASCII dashes.** A U+2010 hyphen in a name can defeat a database lookup by name.

A draw-written `X (Y)` bridges only when the paper defines that pair. `protoheme (heme b)` therefore stays as written, because the paper's `protoheme (heme b )` is not an abbreviation.

### 1.3 Rule 2: typos (`PaperLexicon.snap`)

A name snaps only when all of these hold:

- it does not occur in the paper (whole-token match on strict keys);
- each of its words that is not in the paper is ≥ 6 characters and has **exactly one** paper word within 1 edit (2 edits from 10 characters up);
- that paper word keeps the name word's digits, Greek letters, first 2 and last 3 characters, and oxidation-state or `ethyl` morphemes;
- the snapped name then occurs in the paper as written.

Only long words are ever replaced, so the short tokens that carry the chemistry can never change: `1`/`6`, `alpha`/`beta`, `D`/`L`, `R`/`S`, `cis`/`trans`, `a`/`b`, and roman numerals. Every other character, charge signs included, is kept.

Measured snaps on chlorophyll: `porphobilogen`, `hydroxymethybilane`, `bilverdin`, `geranyllgeraniol`. All four are typos ORCH-752 listed.

**What I believed and what disproved it.** I first had no morpheme guard. Writing the safety tests showed that `ferricyanide` → `ferrocyanide` (a different oxidation state) passes every other rule: 1 edit, the same prefix and the same suffix. That is why `ferr[oi]|cupr[oi]|(?<!m)ethyl` exists. A broader guard on every chemical morpheme would also have blocked the real typo `hydroxymethybilane`, which drops the `l` of `methyl`, so the guard stays narrow (ceiling in § 7.10).

### 1.4 Rule 4 (long form only) and rule 6 (audit)

**One rewrite map, built from all draws.** It is built from every reaction participant any draw wrote. It is then applied to every row that names the participant:

- the reaction sides;
- every entity registry row;
- element locations and transport cargo;
- enzyme and interaction endpoints that carry that exact name.

A renamed entity is never left dangling.

**What is never rewritten:**

- Evidence is never edited.
- Currency names (`CURRENCY_KEYS`) are never rewritten, so `ATP` stays `ATP` and the cofactor-blind key still sees it.
- Enzyme names are compared through the dictionary (finding 1) but written as the draw wrote them.

Every rewrite is listed in `names_rewritten` (draw, from, to, rules, pattern), so no spelling is lost.

**Why the original spellings are not kept on the entity row.** The ruling allowed a side file or an alias field. I chose the side file only: `aliases` and `synonyms` feed database name lookup, and putting `ALA` there could ground 5-aminolevulinate as alanine.

**Rule 6.** Each collapse is recorded on its side-file row as `merges: [{draw, name, bridged_by}]`. `bridged_by` is `exact` (which includes the cofactor-blind key), `abbreviation`, `typo` or `database-id`. It names exactly the rewrites that made two different original spellings equal.

### 1.5 Finding 1 (orchestrator, blocking): chimeric reactions, fixed

**What the reviewer found.** The ORCH-752 key ignored cofactors, enzymes and `biological_state`, and the merge unioned cofactors.

- An NAD-dependent IDH row (IDH3) and an NADP-dependent IDH row (IDH1) became one row: `isocitrate + NAD+ + NADP+ -> 2-oxoglutarate + NADH + CO2 + NADPH`.
- That row carried only IDH3 and draw 1's quote. IDH1's reaction vanished with no flag.
- The same happened within one draw.
- Probe: `C:/t/bt-orch754r0/probe/probe_chimera.py`.
- The base pre-run reproduces exactly that row (`g11/ORCH-754a/06`, and the final ``/46``).

**The fix.** Brief instruction 7 is withdrawn. The cofactor-blind key now only *proposes* a merge, and `_conflict` decides. The same test applies within one draw, across draws, and in rule 3.

**Compatible:** one row names a cofactor, enzyme or state the other omits. The rows merge.

**Conflict:** the rows never merge, and the kept-apart row records why in `kept_apart_from`. A conflict is any of these:

- *Cofactors.* Both rows name different cofactors in one role on one side:
  - nicotinamide: NAD+ / NADH / NADP+ / NADPH;
  - adenine nucleotide: ATP / ADP / AMP;
  - flavin: FAD / FADH2;
  - metal ion: Mg2+, Fe2+, Fe3+, Zn2+, Mn2+, Ca2+, Cu, K+, Na+.

  Spellings of one molecule (`CoA`/`coenzyme A`, `Pi`/`phosphate`, `SAM` ×3, …) are folded first. A cofactor the code cannot classify (marked only by `role: cofactor`) is a conflict unless one row's cofactor set contains the other's. That is the fail-closed case.
- *Enzymes.* Both rows name enzymes and the sets differ after the paper dictionary. In rule 3 the comparison is by identity.
- *Biological state.* Both rows state one and they differ. A state label is resolved through that payload's own `biological_states` row: its compartment (`compartment_canonical`, else `subcellular_location` with a plural `-s` folded), cell type, tissue and species. **A label its payload never defines cannot be compared, and that is a conflict.**

Two further rules:

- **A survivor is judged on everything it absorbed.** It accumulates every cofactor a merged row claimed. A plain row that took in an NADP row can no longer take in an NAD row, so a row compatible with each of two conflicting rows cannot join them (pinned).
- **Donation: I chose "do not donate".** A cofactor is donated only when the **survivor's own quote names it**. Otherwise it is listed in `cofactors_withheld` with the donor's verbatim quote. The reasons:
  - quotes may not be edited or joined;
  - a side-file-only provenance would be invisible to every payload consumer (the evaluator's `evidence_support_rate`, the reaction-evidence UI);
  - leaving the cofactor off keeps the row self-consistent. A smaller supported row beats a larger one.

**Regression tests.** They are in `tests/test_stage_one_ensemble.py`, and they fail behaviourally on the delivered module (§ 4):

- the IDH probe, across draws and within one draw;
- ATP vs ADP, and Mg2+ vs Mn2+;
- different enzymes, and different defined states;
- an unsupported donation.

**Effect on the acceptance cases.** This is the part that moved the numbers:

- Chlorophyll stays at 41. Draw 2's undefined state keeps all of its rows apart.
- Three enzyme pairs also stay apart:
  - `Mg-chelatase` vs `Mg-chelatase + GUN4`, twice;
  - `PPO` vs `Protogen IX oxidase (PPO)`. Schwartz–Hearst cannot derive PPO from that long form;
  - `POR` vs `Pchlide oxidoreductase (POR)`. POR is ambiguous.
- Enterobactin and menaquinone are unaffected (§ 2).

### 1.6 Product-owner ruling of 2026-10-20: support marks to human-facing outputs only

**What I had built first.** Commit `772bd3df` did three things:

- it put `ensemble_support` on the pooled reaction rows;
- it carried that key through `_clean_processes`;
- it added a `Support:` line to `reaction_summary.txt`, through `build_and_save_draft_graph`.

**Why that was wrong.** Every model input traced from those two outputs:

| output | read by | verdict |
|---|---|---|
| `reaction_summary.txt` / `generate_reaction_summary` | the curator (`streamlit_app.py:4232-4236` → `pathway_curator.py:299-300`), the enrichment agent (`gap_resolver.py:2727-2728`), and the session summary that is passed back into the next pipeline call (`streamlit_app.py:5693 → 5698 → 6718/7175`) | **fed to models: not human-only** |
| the pooled Stage-1 payload | `run_inference_pipeline`: `json.dumps(stage_one)` is the Stage-2 prompt (`pipeline.py:404`) | **fed to a model** |
| the mapping report | `interactive_curator.compact_mapping_misses` walks it into the interactive curator's prompt (`interactive_curator.py:270`) | **fed to a model**, so rule 3's record is not put there |
| reaction-evidence `.txt` (`reaction_evidence.render_text`) | `reaction_evidence_view` download only | human-only, but it reads `final_mapped.json`, which no longer carries the count |
| research report | `streamlit_app.py:2425-2523` downloads only | human-only; not used |
| `stage1_ensemble_report.json` / **`.txt`**, `ensemble_identity_merge_report.json` | nothing in `src/` reads them back (grep) | **human-only, and this is where the marks go** |

**What ships now:**

- **The payload gains no key.** The schema is frozen again, as ORCH-750 designed it. The original `test_merge_adds_no_key_the_draws_did_not_carry` is back, byte-identical to base.
- **The support marks live in the side file.** Every side-file row carries `support` (`2/3`) and, at 1/N, `review_required: found by one draw only`. A 1/N row is **kept** in the payload: the contract says review it, never drop it.
- **The hand-comparison text is a new side-file rendering.** `render_report_text` writes it to `stage1_ensemble_report.txt` beside the JSON. Per reaction it shows support with draw numbers, the REVIEW flag, each merge's bridging rule, why a row was kept apart, and withheld cofactors with the donor quote. It then lists every rename. Rule-3 passes append their own block.
- **Rule 3's record goes to side files only.** It is written to `ensemble_identity_merge_report.json` (and the `.txt`), never into the mapping report.
- **Proof that model input is unchanged.** `test_what_a_model_reads_is_the_bases_bytes` compares the tip's `generate_reaction_summary` with goldens that the **base** `draft_graph.py` and `reaction_summary.py` rendered (loaded with `git show f9892cdf`; `evidence/orch754a_llm_input_base_golden.py`, run `g11/ORCH-754a/43`). Two fixture payloads are used: one saved draw (N=1) and the delivered N=3 pooled merge. Both are byte-identical.
- **Proof that the payload carries no mark.** `test_the_pooled_payload_a_model_reads_carries_no_mark` checks the pooled payload against the delivered merge and finds none of `support`, `review_required`, `merges`, `bridged_by`, `kept_apart_from`, `cofactors_withheld`, `names_rewritten` or `ensemble`.
- **Proof that the human output does carry the marks.** `test_the_human_side_file_text_carries_the_marks` runs `run_extraction_pipeline` at N=3 and checks the `.txt` it writes.
- **Lane 754b.** None of the text exports' interaction rows were touched. The goldens leave interactions out so that lane's merge cannot couple to mine.

### 1.7 Rule 3: database identity (`merge_on_mapped_identity`)

It is called once, at the end of `map_payload`. That is after `_admit_identities`, and before stoichiometry, enrichment, pre-freeze resolution and the freeze (merge rule 8). `map_payload` runs at three call sites: the RAG-round mapping, the early Stage-2 mapping, and Stage 6. Rule 3 is idempotent, so running at all three is harmless. No pre-freeze seam needs `streamlit_app.py`.

**Admitted identities.** A participant's identity is the accessions `identity_admission.external_accessions` reports for its entity row (`mapped_ids` after admission withheld the unsupported ones). They count only when the mapping record has all of these:

- `confidence ≥ 0.78`, the threshold `map_payload` counts as `low_confidence_mappings`;
- not `best_effort`;
- not a placeholder, `unresolved`, or Unknown-backed.

Candidates, withheld ids and an unrecorded `mapped_ids` never count (7 parametrised tests).

**How identities are grouped.**

- Rows of one kind that share an accession are one identity (union-find).
- A group whose accessions disagree in any namespace is no identity.
- Without an admitted identity, the identity is the name. At N>1, Stage 1 has already written that name as the long form (test `…long_form_stage1_wrote`).

**The key.** It is the cofactor-blind key over identities. A participant is currency by name, or by sharing an identity with a currency-named entity (`magnesium` ≡ `Mg2+`). Compatibility is finding 1's test with enzymes compared by identity, so the chelatase metal still separates Fe from Mg (test).

**What happens to the merged-away row.**

- A locked one is filed in `quarantined_locked_reactions` with reason `ensemble_identity_duplicate`, exactly as `dedupe_processes` files `duplicate_locked_reaction`. `locked_reaction_filter_report` is recomputed by the same rule, so no lock goes `unaccounted` (test).
- A row a `reaction_coupled_transport` names is never merged away (test).

**Activation.** Rule 3 runs only when THIS run's Stage 1 pooled: `T2PW_EXTRACTION_DRAWS > 1` **and** the run's diagnostics hold the `stage1_ensemble` boundary of 2+ successful draws.

- The payload can carry no marker (§ 1.6).
- The variable alone is not enough: it would merge a saved N=1 payload that is re-mapped under a `.env` saying 3. Pinned in 5 cases.
- **What I believed and what disproved it.** I first activated on the variable alone. Running the mapping chunk under `T2PW_EXTRACTION_DRAWS=3` with conftest disabled (`g11/ORCH-754a/26`) failed `test_map_ids_mapping_lineage::…matches_the_base_golden`. That is the re-map exposure, so I moved activation to the payload. The ruling then forbade a payload marker, which gave the variable-plus-diagnostics rule.

### 1.8 Finding 4 (orchestrator, blocking): the suite read `.env`, fixed

**What the reviewer found.**

- `llm/client.py:22` runs `load_dotenv(override=True)` at import, and the primary checkout's `.env` has `T2PW_EXTRACTION_DRAWS=3`.
- So every Stage-1 call in the suite issued 3 draws.
- `tests/test_early_failure_replay.py` failed 24 cases: "replay made 4 provider call(s) but the case scripted only 3".

**Reproduced here** with the variable set in the bounded child and conftest disabled: **24 failed, 1 passed** (`g11/ORCH-754a/04`). With the fix: **25 passed** (`/05`).

**The fix: `tests/conftest.py`.** An autouse fixture unsets the variable for every test. A test that wants an ensemble sets it itself, as `test_stage_one_ensemble.py` does. The fixture also wraps `dotenv.load_dotenv` for the test's duration, so a **mid-test** reload cannot change the variable. That reload happens: `test_llm_client_request_timeout.py` re-executes `client.py`, and any first import of the client inside a test body reloads too. Every other variable still loads, and `client.py`'s override semantics are unchanged (test `test_a_mid_test_dotenv_load_cannot_switch_ensembling_on`).

**Other files that change under `T2PW_EXTRACTION_DRAWS=3` without the conftest** (`g11/ORCH-754a/59`, `60`):

- `test_early_failure_replay.py` (24). This is the only one in the affected pipeline chunk.
- In the mapping chunk, nothing besides the pre-existing c075 failure, now that rule 3 does not read the variable alone.
- SMOKE: none (`/24`, 508 passed; run on the pre-ruling tip, whose SMOKE-relevant code is unchanged since).

**For the record (requested).** In the primary checkout a shell `T2PW_EXTRACTION_DRAWS=1` **cannot** switch ensembling off, because `client.py` reloads `.env` with `override=True` at import. The only way to switch it off there is to edit `.env`.

---

## 2. Measurements

*Round-1 numbers. They are in commit `6c75112f`'s copies of the files named below; those files now hold the round-2 run (§ 0.4), whose `strict_before_rulings` column reproduces round 1's delivered 41.*

Script `evidence/orch754a_ensemble_merge.py` → `.json`, `.txt`, and the side-file text `orch754a_{chlorophyll,enterobactin,menaquinone}_side_file.txt`. Run bounded at `g11/ORCH-754a/61`.

The "before" is `merge_draws` **as committed at `f9892cdf`**, loaded from `git show`. Paper sha256 values were checked against `index.json`: chlorophyll `8f7ff62b…`, enterobactin `b7d583c0…`, menaquinone `3c563565…`.

### 2.1 Chlorophyll (PMC5130987; draws of 16, 19, 18 reactions)

| pooling | pooled reactions |
|---|---|
| base `f9892cdf` (ORCH-752 key) | **41** |
| tip, compatibility rule only (no paper) | 43 |
| **tip, delivered (paper dictionary + compatibility)** | **41** |
| counterfactual: an undefined state is no state (NOT delivered) | 30 |

**Delivered tip in detail.**

- **Support:** 12 rows at 2/3 and 29 at 1/3. All 29 are kept and flagged.
- **Merges:** 12. Ten were `exact`: draws 1 and 3 wrote the same abbreviation, both rewritten to the same long form. Two were `abbreviation`: `DV-Pchlide` and `DV-Chlide` against draw 1's `3,8-divinyl-…`, which base keeps as three rows each.
- **Kept apart:** 14 rows, with 15 recorded conflicts. Eleven conflict on biological state: draw 2's `plastidic state` is undefined in its payload. Four conflict on enzymes: GUN4 ×2, PPO, POR.
- **Renames:** 31, every one listed with its rule and pattern.
- The full per-merge table, naming the bridging rule of each row, is the `.txt`.

**Against base, row by row:**

- Two DV steps go 3 → 2.
- Mg-chelatase goes 2 → 3: GUN4 is a different enzyme set.
- `Chlide a → Chlide b`, shared by all three draws at base, goes 1 → 2: draw 2's state.

The test `test_chlorophyll_pools_each_paper_step_once` pins 11 paper steps at two rows each: draws 1+3, and draw 2. On base, the two DV steps are 3.

### 2.2 No true reaction lost (certified gold, `reactions.score_leg`)

| case | per draw (ORCH-751) | ORCH-751 union | base pool | **tip pool** | tip precision | rows claimed |
|---|---|---|---|---|---|---|
| enterobactin PMC12096016 (5 gold) | 4, 4, 4 | 4/5 | 4/5, P 0.8 | **4/5, P 0.8** | 0.8 | 5 (4 target, 1 `paper_stated_off_pathway`: pyruvate → lactate, LDH) |
| menaquinone PMC12421875 (9 gold) | 2, 9, 7 | 9/9 | 9/9, P 1.0 | **9/9, P 1.0** | 1.0 | 9 |

- **Recall** is at ORCH-751's level on both cases. **Precision** did not drop, and no row became a `name_conflict`.
- **Menaquinone renames:** `SEPHCHC` → `2-succinyl-5-enolpyruvyl-6-hydroxy-3-cyclohexene-1-carboxylate`, and `DHNA` → `1,4-dihydroxy-2-naphthoic acid`. Both are gold aliases, so both still match.
- **Menaquinone 1/3 rows:** two gold reactions (`DHNA + prenyl diphosphate → DMK`, `DMK → MK`) are 1/3 and are kept with the REVIEW flag. This is a live example of why a 1/N row must never be dropped.
- **Enterobactin** has no renames. Its names are long forms already.

---

## 3. Gates

Every run used `bounded_run.py` with `--heavy-lock ORCH-754a`, `pinned_pytest.py` with no `--`, `PYTHONPATH=<worktree>/src`, `T2PW_OFFLINE_CURATOR=1`, and a fresh `--basetemp` under `C:/t/bt-orch754a/`. Every cleanup report shows `final_surviving_count: 0`. `g11_evidence.py check --task ORCH-754a`: **60 artifacts, 0 non-compliant**, every report with a pin verdict, none refused.

### 3.1 Final gates (tip `cf84626b`; the battery ran on that exact content before it was committed)

| gate | default env | `T2PW_EXTRACTION_DRAWS=3` in the child |
|---|---|---|
| focused: `test_stage_one_ensemble.py` + `test_stage_one_paper_dictionary.py` | **80 passed** (`evidence/g11/ORCH-754a/47`) | **80 passed** (`/53`) |
| affected, pipeline and summary: 12 files (`test_curator_offline_mode`, `test_pipeline_cleanup`, `test_streamlit_stage2_orchestration`, `test_early_failure_replay`, `test_empty_extraction_payload`, `test_extraction_diagnostics`, `test_pipeline_lineage_carrier`, `test_pipeline_reaction_rag_provenance`, `test_preprocessor`, `test_scope_membership_filter_end_to_end`, `test_stage_one_boundary`, `test_draft_graph_entity_kinds`) | **203 passed** (`/48`) | **203 passed** (`/54`) |
| affected, mapping: 17 files (every test file calling `map_payload`/`run_mapping` except the AppTest one) | 397 passed, 1 skipped, **1 failed = pre-existing c075** (`/49`) | same (`/55`) |
| AppTest `test_c055_rag_loop_wiring.py`, alone | **14 passed** (`/50`) | **14 passed** (`/56`) |
| AppTest `test_streamlit_quarantine_boundary.py`, alone | **23 passed** (`/51`) | **23 passed** (`/57`) |
| **SMOKE** (22 files, TEST_MATRIX § Commands) | **508 passed** (`/52`) | **508 passed** (`/58`) |
| without conftest (`--noconftest`), the reviewer's condition | -- | pipeline chunk 24 failed = all `test_early_failure_replay` (`/59`); mapping chunk only c075 (`/60`); SMOKE 508 (`/24`) |

### 3.2 Behavioural proofs

| proof | report | result |
|---|---|---|
| G9 base pre-run, final test file, src modules restored to `f9892cdf` | `evidence/g11/ORCH-754a/46` | **6 failed, 22 passed**: the chlorophyll steps, IDH ×2, ATP/ADP+Mg/Mn, enzyme/state, unsupported donation |
| earlier base pre-runs | `/01` (first chlorophyll expectation), `/06` | 1 failed / 6 failed, as recorded in § 4 |
| finding 4 reproduced, then fixed | `/04` no conftest, `/05` with conftest (`T2PW_EXTRACTION_DRAWS=3` in the child) | 24 failed, 1 passed → 25 passed |
| c075 failure is pre-existing | `/16` (src restored to base, `--noconftest`) | the same 1 failure, same message |
| N=1 `run_extraction_pipeline` identical to base | `/62`, `evidence/orch754a_n1_identity_verdict.json` (`/10`, `/42` earlier) | **N=1 IDENTICAL to base**, 4/4 cases |
| model-bound text: base goldens | `/63` (and `/43`): sha256 `d171a3b8…` (N=1 draw), `7258fe0a…` (pooled), identical across both runs | goldens committed in `tests/fixtures/orch754a/` |
| measurement | `/61` (and `/09`, `/41` earlier, identical numbers) | § 2 |

### 3.3 The one failing test, pre-existing

`test_c075_source_support_armed.py::test_corpus_the_armed_pass_over_every_committed_artifact_and_its_own_paper` fails in the mapping chunk ("unexpected: PMC12096016 strict Pantetheine 4'-phosphate").

- It fails identically **on the base module** with `--noconftest` (`g11/ORCH-754a/16`).
- It is listed as pre-existing in the LEDGER's ORCH-744 close.
- It is not in SMOKE. Not mine, and not touched.

---

## 4. G9

- **Corrections.** These fail behaviourally on the base module as delivered and pass at the tip. The method follows ORCH-752 `01-focused-pre`: for the run, the src modules were set to the base's bytes, then the tip's restored and diffed.
  - The chlorophyll inflation.
  - Finding 1's chimeras (IDH across draws and within one draw, ATP/ADP, Mg/Mn, enzymes, states).
  - Unsupported donation.
- **Pre-runs, in order:**
  - `/01` (first chlorophyll expectation, 1 row per step; superseded once finding 1 changed the expected counts);
  - `/06` (6 failed);
  - final ``/46``.
- **Base failures are assertion failures, not import errors.** The IDH failure prints the reviewer's exact chimeric row.
- **Preservation proofs:**
  - N=1 `run_extraction_pipeline` is identical to base (`orch750_n1_base_identity_proof.py --rev f9892cdf`, `/62`, verdict `evidence/orch754a_n1_identity_verdict.json`).
  - Model-bound text is the base's bytes (§ 1.6).
- **New capabilities** are labelled new acceptance tests in `tests/test_stage_one_paper_dictionary.py`.
- **Pin moved deliberately:** `test_collapsed_duplicate_donates_its_cofactors_to_the_survivor`. Its survivor quote now names the three cofactors it is donated, because donation requires that. The opposite case has its own test.

---

## 5. TEST_MATRIX delta (for the orchestrator to apply)

- SMOKE stays **508**. No SMOKE file changed.
- **New suite-wide fixture:** `tests/conftest.py` unsets `T2PW_EXTRACTION_DRAWS` for every test. A test that needs N>1 sets it. This adds no tests.
- **New focused selection:** `tests/test_stage_one_ensemble.py` (28 tests, was 22) and `tests/test_stage_one_paper_dictionary.py` (52, new). 80 together.
- **New fixture directory:** `tests/fixtures/orch754a/`. Regenerate it with `evidence/orch754a_llm_input_base_golden.py` whenever the merge's output for chlorophyll changes deliberately. `test_the_pooled_payload_a_model_reads_carries_no_mark` fails until you do.

---

## 6. Boundary notes for the reviewer

- **`pipeline.py`:** one line, inside `run_extraction_pipeline`.
- **`map_ids.py`:** two adjacent lines at the end of `map_payload` (an import and the call). The brief said "exactly one call". The import is function-local for layering, so it is a second line.
- **Interaction rows (lane 754b's).** The Stage-1 merge rewrites an interaction endpoint only when it is exactly a renamed reaction participant, so the renamed entity is not left dangling (§ 1.4). On the measured draws exactly one was renamed: menaquinone's `DHNA` → `1,4-dihydroxy-2-naphthoic acid` in the `DHNA` → `MenD` interaction, which follows the reaction participant of the same name. Chlorophyll and enterobactin had none (probe of the pooled payloads). No interaction text output and no `ir.py`/`writer.py` line was touched. Flagged as a possible overlap.
- **Not edited:** `streamlit_app.py`, `reaction_summary.py` (net), `reaction_evidence.py`, `research_report.py`.

---

## 7. Not settled: for the product owner

1. **An undefined biological state fails closed. This is the whole chlorophyll result.**
   - *(a) Keep.* 41, and draw 2's rows stay as 1/3 duplicates flagged for review.
   - *(b) Treat a state its payload never defines as no state.* 30, measured as the counterfactual. Recall and precision on the gold cases are unchanged.
   - Cost of (b): two rows in genuinely different, undefined compartments would merge.
2. **Enzyme-set strictness.**
   - `{Mg-chelatase}` vs `{Mg-chelatase, GUN4}` is a conflict under "the enzymes differ".
   - The alternative is to treat a superset as compatible, like an omitted cofactor. Cost: the merged row keeps only the survivor's enzymes. That loses GUN4, though it is not chimeric.
   - Separately, `PPO` vs `Protogen IX oxidase (PPO)` cannot be bridged. The paper writes that pair but Schwartz–Hearst cannot derive it.
   - The option is to let a draw-written `X (Y)` whose `X (Y)` occurs verbatim in the paper bridge enzyme names. That widens rule 1 beyond what was ruled.
3. **The `shared_locant` extension** (§ 1.2) bridges four chlorophyll intermediates. Veto it and they stay apart. It is one branch in `abbreviation_pairs`.
4. **Support counts in the reaction-evidence UI.** They now live in the side files only. The evidence `.txt` reads `final_mapped.json`, which may not carry them. The option is to have `reaction_evidence_view` load `stage1_ensemble_report.json` from the run's diagnostics directory and match rows by reaction name. Cost: the names change after Stage 1 (normalisation, curator, rule 3), so the matching is approximate.
5. **The 1/N review flag is in the side file, not in `release_status`.** Wiring it to make a run `review_required` is a contract-level decision.
6. **The rule-3 lock record.** A merged-away locked reaction is added to `quarantined_locked_reactions` in the payload, which a later model call may see. That follows `dedupe_processes` and is required for lock accounting, but it is new payload content at N>1. The alternative is never merging locked rows, which would disable rule 3, because Stage-1 rows are locked.
7. **Rule 3 has no paper.** It runs at mapping time, so it cannot expand an abbreviation that Stage 2, RAG or the curator introduces after Stage 1. Such a name is keyed as written. That is the safe direction: it keeps a duplicate.
8. **Rule 3 activation ceiling.** A detached diagnostics recorder persists in one context until the next `activate()`. A library caller that pools and then maps an unrelated payload in the same context, with the variable at N>1, would apply rule 3 to it. Production calls `activate()` per run.
9. **Chunked extraction.** `run_extraction_pipeline` runs once per chunk, so the dictionary is that chunk's text. An abbreviation defined in another chunk does not bridge.
10. **Typo-snap ceiling.** A one-letter in-word edit that changes the molecule in a way no rule names still snaps, but only onto a word the paper uses and only from a name the paper never uses.
11. **Entity unification by identifier is not done.** `protoheme` and `heme b` stay two entity rows even when rule 3 merges their reactions. That was pre-existing at N=1 too.
12. **Instruction 7 withdrawn.** Recorded here for DECISIONS.md; the product owner's entry, not mine.

---

## 8. Not done

- LEDGER, DECISIONS.md and TEST_MATRIX.md were not edited (orchestrator and product owner).
- No pipeline run, no LLM call, no network.
- Rule 3 was measured only on synthetic payloads (20 tests). It needs Stage 6.
- Survival of the merged Stage-1 payload beyond normalisation was probed only for the support key, now removed. It survived cleaning, `merge_additions` and `normalize_process_payload`.
