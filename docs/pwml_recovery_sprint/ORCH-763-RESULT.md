# ORCH-763: nested locations are merge-compatible (D-109 § 1); the DMK split

**Status:** implemented on `orch763/merge-compat`, awaiting independent review. Not merged, not pushed.
**Charter:** D-109 § 1 (product owner, 2026-10-22). **Base:** `5d1ea64c` (= `sprint/pwml-recovery`, carries D-109).
**Commits:** `56ca400c` tests and fixtures first, on base code · `fc00253b` the code · docs commit (this file, evidence).
No LLM call, no pipeline run, no network. Every job ran through `bounded_run.py`; `g11_evidence.py check --task ORCH-763`
reports every artifact compliant, each with `FINAL SURVIVING COUNT : 0` (§ 7).

## 1. Summary

| | before (base `5d1ea64c`) | after (tip `fc00253b`) | source |
|---|---|---|---|
| menaquinone ORCH-759 leg, pooled Stage-1 reactions | **16** (9/9 gold, P 1.000) | **11** (9/9 gold, P 1.000) | `evidence/orch763/08-…base…log`, `09-…tip…log` |
| chlorophyll ORCH-759 leg | **33** | **28** | same |
| enterobactin ORCH-759 leg | 6 (5/5) | 6 (5/5), byte-identical report | same |
| menaquinone / enterobactin / chlorophyll ORCH-751 draws | 9 (9/9) / 5 (4/5) / 29 | identical | same |
| pairwise conflicts between draw rows inside one pooled row | menaquinone-759 **2** (MenF, MenD), others 0 | **0** everywhere | same |
| nested-location merges flagged | none | menaquinone 7, chlorophyll-759 6 | same |

- **(a) done.** Two locations where one contains the other are compatible. The pooled row keeps the more specific
  location, and the merge is flagged `nested_location_merged` in the side file and its `.txt`, never in the pooled
  payload. Pairs that do not nest, unknown pairs included, still conflict. This applies at Stage 1 and in rule 3.
  The 5 MenH–MenI duplicates merge, and each survivor carries draw 2's cytosol state.
- **(b) is a policy question, not fixed.** The paper defines DMK and MK. Schwartz-Hearst cannot read either pair
  because the letter "K" does not occur in either long form. This is the same implementation ceiling as ORCH-758 D6
  (`MV-Chlide a`), which ORCH-758 § 9 item 9 already put to the product owner as "a ruling is needed: it loosens
  rule 1". Case and location played no part. A counterfactual with the two pairs takes menaquinone from 11 to
  **9 rows for 9 gold** (§ 4).

## 2. Where the per-draw payloads are, and how the leg's draws were rebuilt

**The per-draw Stage-1 payloads of a paper run are not kept anywhere.** That is ORCH-758 § 8, re-confirmed here:
- the menaquinone leg directory holds only the POOLED `stage1_payload.json`;
- `tmp/extraction_diagnostics/run-4c8c1399c816/` holds the side file and the boundary reports, no draws;
- `tmp/stage1_raw_extraction.json` is dated 14:51, i.e. a later leg's pooled payload (the menaquinone leg ran
  11:48–12:31).

ORCH-758 § 5 had complete payloads only for the **ORCH-751** chlorophyll draws (`evidence/orch752_chlorophyll_payloads/`).
For ORCH-755 it had only the side file.

So the leg's three draws were **rebuilt** by `evidence/orch763/build_fixture.py`, from two inputs:
- the pooled payload, whose survivor rows are verbatim draw rows;
- the side file, which names, for every pooled row, the draw that wrote it and every row merged into it (draw,
  name, quote and entry quotes).

The rules are written into each fixture under `reconstruction`:
1. Survivor rows are verbatim, minus the post-Stage-1 keys `provenance_lineage`, `locked_reaction_id` and
   `scope_membership`.
2. A merged-in row takes its own name and quotes, plus the survivor's participants and enzyme names. The merge had
   bridged their keys.
3. A merged-in row's `biological_state` is not recorded anywhere, so it is **inferred** and listed (menaquinone: 6
   rows).
4. Names the side file lists as rewritten go back to the draw's own spelling.
5. One field the pooled payload lost was patched, with its reason. The pooled payload carries no
   `compartment_canonical` at all. The side file kept draw 3's MenH–MenI rows apart on
   `biological_state location 'cytosol' vs 'intracellular'`, with the candidate (draw 2) printed first. Draw 2's
   `intracellular state` has `subcellular_location: intracellular`, so only `compartment_canonical: cytosol` can
   have produced "cytosol" (`_state_profile` reads `compartment_canonical` first).

**Validation.** Re-pooled at base with the leg's own dictionary text, each rebuilt leg reproduces its leg's side
file exactly: every row's key, draws and kept-apart reasons, plus the rename count. The dictionary text is chunk 1,
re-chunked from the leg's `01_source_text.txt` with the app's 10,000/1,600 defaults; its length equals the side
file's `paper_chars`.
- menaquinone: 16 rows, 6 renames;
- enterobactin: 6 rows, 4 renames;
- chlorophyll: 33 rows, 33 renames.

The checker prints `RECONSTRUCTION CHECK … IDENTICAL` for all three (`08-repool-base-5d1ea64c.log`). Base's 16 rows,
9/9 and "dups collapsed 5" equal `orch759_gold_accuracy.txt` Stage 1.

**What the inference touches.** Only merged-in rows. For menaquinone these are draw 2's MenF, MenD, MenA and MenG
and draw 3's MenF and MenD, each given its draw's one intracellular state. The 5 MenH–MenI pairs are verbatim on
both sides, so the G9 proof does not depend on any inference (§ 5).

## 3. (a) The change

`src/t2pw/pipeline/stage_one_ensemble.py` only, +110/−15 (`git diff --numstat 5d1ea64c fc00253b`). No other src file
changed. `matcher.py`, `name_keys.py`, `streamlit_app.py`, `map_ids.py`, `extraction_diagnostics.py` and
`strict_quarantine.py` are untouched.

| where (tip) | what |
|---|---|
| `:132` | `NESTED_LOCATION_FLAG = "nested_location_merged"` |
| `:266` `_location` | the existing compare form (strict key, plural `-s` folded), lifted out of `_state_profile` so the table is keyed the same way. Behaviour is identical. |
| `:281` `_LOCATION_PARENT` | the containment table (§ 3.1) |
| `:302-318` `_containers`, `_nests`, `_narrower` | chain walk; nested-or-equal; the more specific of two |
| `:379` `_nested_location` | the flag text `'inner' within 'outer'`: two differently named DEFINED states whose locations differ and nest |
| `:430` `_conflict` | a location pair that nests is skipped (`continue`). Everything else is unchanged, including cofactors, enzymes, cell type, tissue, species, and unknown or non-nested locations |
| `:511` `_fold` | the survivor takes the joining row's state when that row's location lies strictly inside the survivor's own (`state_taken`, already rendered as "Biological state taken from a merged row") |
| `:625` `_absorb` | of two defined states, the record CLAIMS the narrower location, so later rows are judged against it (§ 3.2) |
| `:710` `_union`, `:1443` rule 3 | `location_flag: {reason: nested_location_merged, locations: …}` on the merge entry |
| `:984` `_flag_lines`, `:1031` | the `.txt` line `FLAG (review): nested_location_merged -- 'cytosol' within 'intracellular' (…)`, counted in "merge(s) flagged for review" |

**LLM-bound text (D-104 AMENDMENT 2).**
- The flag lives only in the side file JSON and in `stage1_ensemble_report.txt` / the identity side file, which
  nothing reads back.
- The pooled payload, which is the Stage-2 prompt, gains no key. Two tests check this: `test_preserved_the_pooled_payload_carries_no_merge_mark`
  and `test_new_rule3_…` (`FLAG not in json.dumps(payload)`).
- **What does change for a model is the D-109 effect itself:** fewer pooled rows, and a survivor's
  `biological_state` value may now be the more specific label. No renderer or prompt changed.

### 3.1 The containment source, and its review

**What the table is.** A small committed table, `_LOCATION_PARENT`, with 17 entries. Each location names ONE
container, so the containers of any location form a single chain. Nothing is looked up at run time, and there is no
network call.

**Where it comes from.** It is written from GO cellular component part_of / is_a. The GO id sits beside each row
for the reviewer:

| chain | rows |
|---|---|
| cell | intracellular (GO:0005622) |
| intracellular | cytoplasm (GO:0005737), nucleus (GO:0005634) |
| cytoplasm | cytosol (GO:0005829), mitochondrion (GO:0005739), endoplasmic reticulum (GO:0005783), golgi (GO:0005794), lysosome (GO:0005764), peroxisome (GO:0005777), endosome (GO:0005768), plastid (GO:0009536) |
| cell | plasma membrane (GO:0005886) |
| mitochondrion | mitochondrial matrix (GO:0005759) |
| plastid | chloroplast (GO:0009507, is_a), etioplast (GO:0009513, is_a) |
| chloroplast | thylakoid (GO:0009534 chloroplast thylakoid) |
| thylakoid | thylakoid membrane (GO:0042651) |

**Why these entries.** They are the Stage-1 prompt's canonical compartment vocabulary (`pwml_system.txt:170-172`:
cytosol, nucleus, mitochondrion, mitochondrial_matrix, endoplasmic_reticulum, golgi, lysosome, peroxisome,
plasma_membrane, endosome; extracellular is deliberately left out of the table), the locations observed in the
ORCH-759/751 draws (intracellular, cell, plastid, chloroplast, etioplast, thylakoid membrane, cytoplasm), and the
chain nodes between them.

**Review done here:**
- `test_new_the_containment_table_is_a_tree_of_folded_names`: every key is in compare form, there is no cycle, the
  D-109 example nests, and `cytosol` vs `membrane`, `cytosol` vs `nucleus` and `cytosol` vs `periplasm` (unknown)
  do not.
- **The GO ids were written from knowledge, not fetched.** No network was used. A reviewer should spot-check
  them against QuickGO (NOT settled, § 9).

**Known ceilings, stated in the code:**
- *thylakoid ⊂ chloroplast* holds for plants and algae. A cyanobacterial thylakoid lies in no plastid, and a pool
  that names both would be misread.
- Synonyms are absent. `mitochondria`, `Golgi apparatus`, `cytoplasmic` and `periplasm` are unknown, so they fail
  closed as before.

### 3.2 One correction this needed: the claimed location

**What changed.** `_absorb` used to keep the first defined state as the record's claim. A pooled row that took a
row with no location first therefore let **any** two locations join: at base, a row with no location bridged
cytosol and mitochondrial matrix into one row (`test_g9_a_row_without_location…`, which fails at base).

**Why it was needed.** With nesting this would get worse. If intracellular is claimed, then cytosol joins, and then
nucleus joins too, two locations that do not nest would sit in one row. Claiming the narrowest location closes
both.

**Real data, at base.** In the menaquinone leg, MenF and MenD joined draw 2 (cytosol) and draw 3 (intracellular)
through draw 1's unlocated state: 2 pairwise conflicts by base's own rule (`08-…base…log`, "PAIRWISE CONFLICT
inside R1/R2"). At the tip those pairs nest legitimately, and the merges are flagged.

**What I did not change.** I applied the survivor rule ("keeps the more specific") only when the survivor itself
states a location. A survivor with NO location keeps its own state label, as at base; the record's claim still
narrows. So menaquinone's MenF survivor keeps `Lactococcus lactis cells`. See § 9 Q3.

## 4. (b) DMK vs "demethylmenaquinone (DMK)": defined in the paper, blocked by the abbreviation matcher

Evidence: `evidence/orch763/diag_dmk.py`, output `01-diag-dmk.log`.

- **The paper defines both.** "…MenA joins DHNA and prenyl diphosphate to produce demethylmenaquinone (DMK)" (chunk
  offset 6333) and "Quinones, such as menaquinone (MK, vitamin K2)" (5322).
- **Blocked by the dictionary.** The leg's lexicon holds 15 pairs (= the side file's 15), and neither DMK nor MK is
  among them. Schwartz-Hearst's `_best_long_form` returns `None` for both. The short-form character K is absent
  from the whole long-form candidate (`short-form characters absent … ['K']`). The K comes from "vitamin K", not
  from the name.
- **Not case.** `spelling_key` is exact (`DMK`), and a case-folded search of the pairs finds nothing.
- **Not location.** The MenA/MenG rows were never compared. Their keys differ (`… -> demethylmenaquinone dmk` vs
  `… -> dmk`), so no `kept_apart_from` exists. At base, draw 1's unlocated state is compatible with draw 3's
  intracellular one: MenF/MenD of the same draws merged.
- **Classification: `policy_disagreement`, not a defect fixed here.** D-104 § 3(a)'s authority is "the paper
  defines", and the paper does. The chosen algorithm is narrower than the rule. That is exactly ORCH-758 D6, and
  ORCH-758 § 9 item 9 asked for a ruling because a fix "loosens rule 1". D-109 does not rule on it. So no code.
- **What a ruling would buy** (`counterfactual_dmk.py`, `13-counterfactual-dmk.log`; a monkeypatch in-process,
  not production code): adding exactly the two pairs gives **11 → 9 pooled rows, 9/9 gold, P 1.000, 9 rows for 9
  reactions**. Draw 3's MenA and MenG join by `abbreviation`, and the merges are flagged nested
  (`'cytosol' within 'intracellular'`).

## 5. G9 proof

| test (`tests/test_orch763_nested_locations.py`) | class | base `56ca400c` (`05-g9-red`) | tip `fc00253b` (`07-g9-green`) |
|---|---|---|---|
| `test_g9_leg_menh_to_meni_duplicates_merge_keeping_cytosol`: the leg's verbatim MenH–MenI rows of draws 2 and 3 | G9 correction | FAIL: 10 rows ("Left contains 5 more items") | pass: 5 rows, survivor `intracellular state` (= cytosol), 5 flags `'cytosol' within 'intracellular'` |
| `test_g9_leg_pool_16_to_11`: whole rebuilt leg, gold 9/9 | G9 correction | FAIL: `assert 16 == 11` | pass |
| `test_g9_a_row_without_location_does_not_join_two_non_nested_locations` | G9 correction (§ 3.2) | FAIL: `['plain'] == ['plain', 'matrix']` | pass |
| `test_new_*` (4): survivor takes the narrower state; no join through a common container; rule 3 merges and flags; the table | new acceptance (D-109), labelled | fail (by construction; one `AttributeError` on the new table) | pass |
| `test_preserved_*` (12): unknown pair, cytosol/matrix, cytosol/membrane apart; chimeric pairs (cofactor, enzyme, species) with nested locations apart; equal locations merge unflagged; payload carries no mark; rule 3 keeps non-nested apart; enterobactin 5/5; chlorophyll 759 and 751 re-pool with no pairwise conflict | preservation | **all 12 pass** | all pass |

Base totals: 7 failed, 12 passed. Tip: 19 passed. The file uses string literals, not new symbols, so it collects at
base and its base failures are behaviour.

## 6. Every remaining split after the re-pool (tip)

"Split" here means pooled rows that look alike but stay apart:
- (a) one reaction key, kept apart with a recorded reason;
- (b) different keys, same enzyme set;
- (c) one gold reaction matched by several rows.

Source: `09-repool-tip-fc00253b.log` and `14-chl-rows-tip.log`.

**menaquinone-759 (11 rows, 9 reactions):**
- MenA: R3 (draws 1, 2; `demethylmenaquinone (DMK)`) vs R10 (draw 3; `DMK`).
- MenG: R4 (draws 1, 2; `demethylmenaquinone (DMK) -> menaquinone (MK)`) vs R11 (draw 3; `DMK -> MK`).

Reason: there is no dictionary pair (§ 4, Q1). No (a) split remains.

**chlorophyll-759 (28 rows, about 19 reactions; 9 surplus, none of them on location any more):**

| # | reaction | rows (draws) | reason kept apart |
|---|---|---|---|
| 1 | GluTR | R1 d1 `Glu-tRNA`→GSA {HEMA1}; R23 d2 `Glu-tRNAGlu` {HEMA1}; R24 d3 `Glu-tRNAGlu` {GluTR} | spelling `Glu-tRNA`/`Glu-tRNAGlu`: no pair, and `Glu-tRNA` occurs in the paper so rule 2 never snaps (ORCH-758 § 5, rule-3 territory). Also gene vs enzyme, {hema1} vs {glu trna reductase}: non-nested sets (AMENDMENT 3 § 2) |
| 2 | ALAD | R3 d1,d3 {ALAD → ALA dehydratase} vs R4 d2 {ALAD1/HEMB1} | gene vs enzyme name: non-nested sets |
| 3 | CPO | R8 d1,d2 {CPO1} vs R9 d3 {CPO → coprogen III oxidase} | isoform vs enzyme (ORCH-758 § 9 item 7, open) |
| 4 | PPO | R10 d1,d2 {PPO1} vs R11 d3 {PPO} | isoform vs enzyme |
| 5 | Mg-chelatase | R12 d1,d3 {GUN4, Mg-chelatase} vs R13 d2 {Mg-chelatase complex} | complex vs subunit names |
| 6 | Mg-Proto ME cyclase | R15 d1,d3 {Mg-Proto ME cyclase} vs R16 d2 {… complex} | complex vs enzyme name |
| 7 | POR | R17 d1,d3 {POR → Pchlide oxidoreductase} vs R18 d2 {PORB} | enzyme vs isoform |
| 8 | ferrochelatase | R25 d2 Proto IX → `heme b` {FC2} vs R28 d3 → `protoheme` {ferrochelatase} | synonym products, which Schwartz-Hearst cannot pair (rule-3 territory), plus isoform vs enzyme |
| — | Chl synthase | R21 (3/3) geranylgeraniol + MV-Chlide a vs R26 d3 MV-Chlide a + phytol | **genuinely different substrates; correct** |
| — | CAO | R22 (3/3) Chl a → Chl b vs R27 d3 Chlide a → Chlide b | **genuinely different reactions; correct** |

The 5 `thylakoid membrane` vs `plastid` splits of base (Mg-chelatase, cyclase, POR, Chl synthase, CAO) all merged,
each flagged (6 flags; Chl synthase took two).

**chlorophyll-751 (29, unchanged):**
- PPO {ppo} vs {protogen ix oxidase ppo} and POR {por} vs {pchlide oxidoreductase por}: draw-written `X (Y)` enzyme
  names that are not bridged.
- Different keys: ferrochelatase `protoheme` / `protoheme (heme b)` / `heme b`; DVR `3-vinyl Chlide a` vs
  `MV-Chlide a` (D6); Chl synthase, three spellings (D5/D6).

None is on location. Every reaction of this pool is in plastid: draw 1 `plastid`, draw 3 `plastid state`, and draw
2's `plastidic state`, which its payload never defines, so it counts as no state (11 flags). Draw 3's
`nucleus state` names no reaction.

**enterobactin-759 (6):** EntF R5 (draw 1, 2,3-dihydroxybenzoyl-AMP + L-serine) vs R6 (draws 2, 3, 2,3-DHB +
L-serine). A different defining participant, so it is never merged (D-104 § 3(f)). The gold accepts both.

**enterobactin-751, menaquinone-751:** none. Enterobactin-751's EntF is unmatched because no draw found it: the
known ORCH-751 4/5.

**ORCH-755 chlorophyll (key-level prediction only; its draws are not kept).** Its side file
(`evidence/orch758_inputs/…`) has exactly one location split: Mg-Proto ME cyclase, `thylakoid membrane` vs `plastid`.
The enzyme comparison precedes the state comparison in `_conflict`, so its enzymes were compatible. Expected
**41 → 40**. This cannot be verified without the draws.

## 7. Gates (G11)

All reports `evidence/g11/ORCH-763/`, logs `evidence/orch763/`, pin verdicts `evidence/g11/pin/ORCH-763/`. Every
report shows `FINAL SURVIVING COUNT : 0`.

| gate | report | tree | result |
|---|---|---|---|
| G9 RED | `05-g9-red` | `56ca400c` (tests on base src), clean | 7 failed, 12 passed (§ 5) |
| G9 GREEN | `07-g9-green` | `fc00253b`, clean | **19 passed** |
| affected: new file, `test_stage_one_ensemble`, `test_stage_one_paper_dictionary`, `test_name_keys_match_evaluation_matcher`, `test_orch758b_empty_chunk`, `test_early_failure_replay`, `test_c089_participant_schema` (the brief's grep plus the named files) | `10-affected` | `fc00253b` | **228 passed, 2 skipped**. The 2 skips are `test_c089_participant_schema.py:224/357/385` "committed run artifact/corpus not present", independent of `src` |
| SMOKE | `11-smoke` | `fc00253b` | **508 passed** |
| `tests/test_eval_v2_*.py` (6 files) | `12-eval-v2` | `fc00253b` | **89 passed, 1 xfailed** (the strict xfail ORCH-742c in `test_eval_v2_embedding_cache.py:53`, known: TEST_MATRIX § ORCH-754) |
| re-pool, base / tip | `08` / `09` | exported `5d1ea64c` src / `fc00253b` | § 1, § 6 |
| `g11_evidence.py check --task ORCH-763` | — | — | every artifact `ok`, 0 non-compliant; 6 direct-pytest reports, all with pin verdicts |

`name_keys.py` = `matcher.py` is untouched, and the pin test is in the affected set (green).

## 8. Failures and dead ends (in order)

1. **Per-draw payloads are not kept.** I expected them beside the side file (the brief: "find where the 3 per-draw
   payloads are kept"). They are nowhere (§ 2). The workaround is the rebuild, validated row for row at base.
2. **`02-build-fixtures` exit 1.** `bash` on the wrapper's PATH resolved to WSL (`execvpe(/bin/bash) failed`).
   Re-run with Git Bash's full path as `03`. This was infrastructure, not a result (0 survivors).
3. **`04-repool-base` mis-numbered splits (b)/(c).** `repool.py` zipped the side-file records, which are grouped by
   key, with the pooled payload, which is in creation order. Fixed by pairing them k-th to k-th per key, as the
   builder does. Re-run on an exported base tree as `08`. The `04` summary counts and the reconstruction check were
   right; only the per-row labels in (b)/(c) were wrong.
4. **`06-g9-green-draft`** ran on a dirty tree before the code commit. It is kept for the record; `07` is the GREEN
   of record.
5. **A theory I ruled out: that DMK was blocked by location or case.** Both were disproved (§ 4). The rows were
   never compared, and the lookup is exact-case with no case-folded hit.
6. **A design I rejected: flagging only when the survivor's OWN location nests with the joiner's.** It would hide a
   merge that only the record's claimed location admitted, as with MenF draw 3. The flag is computed on the claim,
   which is what `_conflict` judged.

## 9. NOT settled: for the product owner

**Q1. DMK/MK, the Schwartz-Hearst ceiling (ORCH-758 D6 class).** The paper defines the pair, but the matcher's
character match fails.

| option | what it costs |
|---|---|
| (a) accept the ceiling | menaquinone keeps 2 surplus rows (11 for 9); chlorophyll keeps the `MV-Chlide a` rows. The evaluator does not collapse them (keys differ), so they stay as duplicate review rows. No risk |
| (b) allow a paper `long (SF)` pair whose short form misses at most one letter, when its first letter starts the long form's word | catches DMK, MK and probably `MV-Chlide a`. It loosens rule 1 for every paper, so a false pair such as `protein (PDB)` must still fail. Needs its own lane, G9 and red-team; about half a day |
| (c) accept a pair only when a draw writes `X (Y)` AND the paper contains that exact phrase (first item of a comma list) | narrowest: the boundary comes from the draw, the proof from the paper. It catches `demethylmenaquinone (DMK)`, and `menaquinone (MK)` via the `(MK, vitamin K2)` first-item rule. About 2-3 h plus review |

A counterfactual of any option that yields the two pairs gives 11 → 9 (§ 4).

**Q2. Two pre-existing ways a pooled row can still span incompatible states, outside D-109 and NOT fixed here**
(`probe_open_holes.py`, `15`/`16`, identical at base and tip):
- **H1.** Claims chain through an empty species, cell type or tissue. A row with no species lets an E. coli row
  and a B. subtilis row join (1 pooled row).
- **H2.** One state label defined differently by two draws merges without comparison, because `_conflict` compares
  only differently named states. Cytosol and mitochondrial matrix joined.
- **Measured occurrence of H2** in the complete ORCH-751 draws: 0 labels defined two ways in menaquinone,
  enterobactin and chlorophyll (`17-probe-h2-occurrence`).

| option | what it costs |
|---|---|
| (a) one lane: claim field-wise in `_absorb` (H1) and compare defined states even under one label (H2) | stricter, so it can only split rows. G9 for each; about 2 h plus review |
| (b) accept | the risk as measured: not observed in kept draws |

I left them alone deliberately: D-109 rules on locations only. The no-location chain (§ 3.2) I did fix, because
nesting would otherwise have made it worse.

**Q3. A survivor that states no location keeps its own state label** (MenF/MenD keep `Lactococcus lactis cells`)
while the record claims cytosol. D-109's "the survivor keeps the more specific location" read literally would
give the survivor draw 2's state.

| option | what it costs |
|---|---|
| (a) as now | the base behaviour for no-location merges is unchanged |
| (b) take the narrowest located state onto an unlocated survivor too | changes merges base already made silently, with no flag today |

Product call.

**Q4. The GO ids in § 3.1 were written without a lookup.** A reviewer should spot-check about 17 ids. If thylakoid ⊂
chloroplast is not wanted (cyanobacteria), removing two rows restores chlorophyll-759's 5 splits.

**Q5. Isoform vs enzyme, complex vs subunit, gene vs enzyme names** (chlorophyll 7 of 9 surplus rows) and
synonym products (rule-3 territory). These are still ORCH-758 § 9 item 7, part 1; D-109 ruled only on locations.

**Q6. Keep per-draw payloads.** A future lane re-running this analysis needs them. Writing the N draw payloads into
the side-file directory would cost about 3× one payload per leg (about 15-60 KB). There is no LLM input impact.

**Not observed:** a 3-draw app leg at the tip. The re-pool is offline. A real leg after merge (with D-107's rule-3
recorder) is the confirming run.

## 10. TEST_MATRIX delta (exact text, for the orchestrator to append; not edited by this lane)

```
## ORCH-763: nested locations merge-compatible (D-109 § 1) — appended end-of-file by rule · 2026-10-22

New focused file `tests/test_orch763_nested_locations.py` (**19**, ~0.5 s; fixtures `tests/fixtures/orch763/`,
the three ORCH-759 legs' draws rebuilt by `evidence/orch763/build_fixture.py`). Run it with the `754a` ensemble
pair for any change to `_conflict`, `_absorb`, `_fold`, `_state_profile`, `_LOCATION_PARENT` or the merge flags.
`754a` ensemble merge + ORCH-763: 28 + 94 + 19 = **141**. Measured at `fc00253b`: G9 RED at `56ca400c` 7 failed /
12 passed (`g11/ORCH-763/05`), GREEN 19 passed (`07`); affected set 228 passed, 2 skipped (`10`); SMOKE **508**
(`11`, unchanged: the new file is not in SMOKE); `tests/test_eval_v2_*.py` 89 passed, 1 xfailed (`12`).
```

## 11. Evidence index

| file | what |
|---|---|
| `evidence/orch763/diag_dmk.py` → `01-diag-dmk.log` | (b) diagnosis |
| `evidence/orch763/build_fixture.py` → `03-build-fixtures-gitbash.log` | the rebuild, with every inference printed |
| `evidence/orch763/repool.py` → `08-repool-base-5d1ea64c.log`, `09-repool-tip-fc00253b.log` | before/after, splits, gold, pairwise chimera check, reconstruction check |
| `evidence/orch763/counterfactual_dmk.py` → `13-counterfactual-dmk.log` | Q1's value |
| `evidence/orch763/chl_rows.py` → `14-chl-rows-tip.log` | chlorophyll-759 tip rows with enzymes. Its draw list repeats the first record for the two same-name ALAD rows; § 6 uses the side-file reasons |
| `evidence/orch763/probe_open_holes.py` → `15`, `16`, `17` | Q2 |
| `tests/fixtures/orch763/*_orch759_draws.json` | rebuilt draws; `source` (input sha256s) and `reconstruction` inside each |
