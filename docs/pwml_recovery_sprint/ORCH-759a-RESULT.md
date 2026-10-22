# ORCH-759a — an unreferenced duplicate protein row no longer blocks a `review_required` pathway (finding E1)

**Status:** implemented and gated on the lane branch. **Not reviewed and not merged.** An independent
reviewer judges the actual diff next.

| | |
|---|---|
| branch | `orch759a/fold-duplicate-protein` (worktree `.claude/worktrees/orch759a`) |
| base | `b06b86b5` (= `sprint/pwml-recovery`; code identical to `f392d447`) |
| test-first commit (base code) | `30942f32` |
| code commit | `4a6d9a6c` |
| test correction (app-path test only, see § 6 F2) | `e346bb6e` |
| docs/evidence commit | the commit that adds this file |
| charter | `ORCH-759-RESULT.md` § 4 finding 2 (E1), the orchestrator ruling approved by `main` 2026-10-22 (five conditions), PRODUCT_CONTRACT § 1 ("an irrelevant degree-zero entity" may not end a run), D-104 § 3(c)-(f), AMENDMENTS 2, 4, 5, D-105, D-106 |

No LLM call and no pipeline run was made. Every number below names its G11 report under
`evidence/g11/ORCH-759a/` and its log under `evidence/orch759a/` (same number). Every report shows
`FINAL SURVIVING COUNT : 0`; `g11_evidence.py check --task ORCH-759a`: 27 artifacts, 0 non-compliant.

---

## 1. The defect

Leg `runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/papers/PMC12096016/strict/` (primary
checkout, read-only), `final_mapped.json` file sha256 `b2ab0c94…`, canonical `payload_sha256` `e72d3a65…`.

1. Stage 2 inferred the enzyme `enterobactin synthase complex`; normalization made a generated wrapper
   (`relocate_complex_named_proteins`, `process_normalizer.py:5140-5241`) whose one component is protein row
   `enterobactin synthase`.
2. The audit tied that row and the component to EntF's UniProt P11454 — after Stage 3.
3. Stage 6 `_reconcile_components_against_local_proteins` (`map_ids.py:6985-7078`) matches a component on UniProt
   first, first row wins: the component became `EntF`, and row `enterobactin synthase` was referenced by nothing.
4. The final pre-export gate (`streamlit_app.py:3299-3307` → `process_normalizer.py:5092-5108`) raised
   "Protein has degree 0 after normalization: enterobactin synthase", and no PWML was written for a pathway
   quarantine had released as `review_required` with strict gates passed.

**Found by this lane, and needed to reproduce E1 at all: why quarantine did not delete the orphan.** The strict
quarantine pruner keeps a row whose name **or any synonym** is referenced, and it exempts the components of
surviving complexes (`strict_quarantine.py` `_prune_entities` :1798-1850 via `_entity_name_norms`, and
`_degree_zero_exports` :2231-2300). Row `enterobactin synthase` carried its record's names as synonyms: `entF`
(normalizes to `EntF`, the component of two surviving complexes) and `enterobactin synthase complex` (a referenced
enzyme). So the pruner kept it. The final gate tests the protein's **name** only. A synonym-less orphan is deleted
by quarantine and never reaches the gate: the first version of the app-path test showed exactly that at base
(§ 6 F2, report `06`). E1 needs both the Stage-6 rename and this name-versus-synonym disagreement.

## 2. The diff (`git diff b06b86b5 e346bb6e -- src`: 3 files, +186 −1)

- **`src/t2pw/mapping/map_ids.py`**
  - New block before `map_payload` (`:10115-10278`): `_fold_unreferenced_duplicate_proteins(payload, report)` and
    its helpers `_fold_key`, `_exact_key`, `_strings`, `_row_name_variants`, `_entity_quotes`, `_move_protein_pointers`.
  - One call in `map_payload` (`:11293`), inside the existing `if merge_identity_duplicates:` and before rule 3's
    `merge_on_mapped_identity`. It is after every resolution loop, after component reconciliation, after
    `_admit_identities`, and after the report is built.
  - Module import of `DUPLICATE_PROTEIN_FOLDS_KEY`, and two sentences added to the `map_payload` docstring.
- **`src/t2pw/curation/prompt_payload_keys.py`** (+9): `DUPLICATE_PROTEIN_FOLDS_KEY = "duplicate_protein_folds"`,
  added to `PROMPT_OMITTED_PAYLOAD_KEYS`, with the reason.
- **`src/t2pw/curation/interactive_curator.py`** (+3): `compact_mapping_misses`' walker skips every key in
  `PROMPT_OMITTED_PAYLOAD_KEYS`.

What the fold does. For each protein row R:
- R must not be referenced. Every string in `processes`, in `element_locations`, in every complex's `components`
  and in element collections' `element_states` is keyed case-, space- and punctuation-blind. R counts as referenced
  if any of these match:
  - its name;
  - its `key`;
  - any alias that no other protein or complex row holds. Aliases are compared exactly here: an alias another row
    holds resolves to that row after the fold. In the leg, `entF` resolves to EntF.
- R needs at least one admitted accession (`stage_one_ensemble._admitted_ids`, rule 3's own definition: what
  `map_payload` left in `mapped_ids` after `_admit_identities`, from a confident, non-best-effort, non-placeholder,
  non-unresolved mapping).
- R needs a known species (`_species_hint_from_row`).

R is then folded into the first other row S that meets all of these:
- S is referenced;
- S shares at least one admitted accession with R;
- no admitted namespace of theirs disagrees;
- S's species hint is equal to R's.

The fold does five things:
- R's name is appended to S's `synonyms` (if S does not already hold it);
- R's `evidence` and `source_refs` quotes are appended to S's `source_refs` (those S does not hold);
- R is removed from `entities.proteins`;
- every `/entities/proteins/<i>` `json_pointer` in the report is moved (R's pointer names S);
- one entry goes into `report["duplicate_protein_folds"]`.

Every other row, including every other degree-0 row, is untouched.

## 3. The five conditions, and the tests that hold each

All `test_new_*` tests are **NEW ACCEPTANCE TESTS** and are labelled so in their module docstrings.

| condition | how it is met | tests (`tests/test_orch759a_duplicate_protein_fold.py` unless named) |
|---|---|---|
| 1. same **admitted** accession, never candidate/provisional/withheld | `_admitted_ids` (rule 3's definition), read after `_admit_identities`; any disagreeing namespace refuses | `test_new_a_candidate_withheld_or_provisional_accession_never_folds[candidate_only, withheld, low_confidence, best_effort, placeholder, unresolved]`; `test_new_an_identity_admission_withholding_blocks_the_fold` (the REAL pass A withholds P11454 into `rejected_mapped_ids`, so there is no fold and the gate still blocks); `test_new_rows_whose_admitted_accessions_differ_never_fold[another_accession, one_shared_one_conflicting]` |
| 1. same species | equal `_species_hint_from_row`, and a species must be known | `test_new_a_different_or_unknown_species_never_folds[different_species, no_species]` |
| 1. folded row referenced by nothing; survivor referenced | the reference scan above | `test_new_a_referenced_duplicate_is_never_folded[reaction_enzyme, reaction_modifier, transport, interaction, location, complex_component, only_alias]`; `test_new_an_alias_another_row_holds_does_not_count_as_a_reference_to_the_duplicate` (the leg's own synonym shape); `test_new_two_unreferenced_rows_never_fold_into_each_other` |
| 1. alias + every quote kept, fold recorded | survivor `synonyms` and `source_refs`, record entry, pointers moved | `test_new_the_fold_keeps_the_name_and_every_quote_and_records_them`; `test_new_the_report_pointers_follow_the_fold`; G9 cases |
| 1. unreferenced row with no shared admitted accession: untouched, the gate still catches it | no admitted accession, or none shared, means no fold | `test_new_an_unreferenced_row_with_no_admitted_accession_is_untouched_and_the_gate_still_catches_it` (gate result `["Protein has degree 0 …: enterobactin synthase"]`); the withholding test above asserts the same |
| 2. fires in the REAL app path, at 1 and 3 draws, never gated on `_pooled_draws()` / rule 3 / the recorder / the draw count | gated only on `merge_identity_duplicates=True`, which the protected app already passes at its Stage-6 call (`streamlit_app.py:4248-4262`) | `tests/test_orch759a_fold_app_path.py::test_new_the_fold_fires_through_the_apps_own_stage6_call[one_draw, three_draws]` (see below); `test_new_the_fold_fires_at_any_draw_count[draws_unset, draws_1, draws_3_not_pooled]`; `test_new_only_the_stage6_pass_folds[default, explicit_false]` |
| 3. identity lookup, admission and reconciliation untouched | no change to `_reconcile_components_against_local_proteins`, the resolvers, `identity_admission.py` or `_admit_identities` (diff above); the fold runs after them | G9 cases assert the wrapper's component is still `EntF` after the pass; the app test asserts it is still `glutathione synthetase` |
| 4. the record is human-facing and reaches no LLM | § 4 | `test_new_the_fold_record_reaches_no_llm_input`; `test_new_the_record_key_is_skipped_whatever_it_holds`; the app test's `refinement_mapping_misses` assertions |
| 5. `streamlit_app.py` untouched; no gate or prompt change; no LLM or pipeline run | `git diff --quiet b06b86b5 HEAD -- src/t2pw/app/streamlit_app.py src/t2pw/llm/prompts` passes; `process_normalizer.py`, `strict_quarantine.py` and `identity_admission.py` are not in the diff | the LLM-input goldens `tests/test_orch757_llm_token_budgets.py` and `tests/test_stage_one_paper_dictionary.py::test_what_a_model_reads_is_the_bases_bytes` pass in `16` |

**How condition 2 was proven (real app path).** `test_orch759a_fold_app_path.py` drives the protected
`streamlit_app.py` with `AppTest`, the way `batch/driver.py` does:
1. one script run loads the app;
2. a finished pipeline is seeded;
3. `pwml_generate_btn` is clicked, and the post-pipeline runs in a script run of its own.

It uses the harness of `test_streamlit_quarantine_boundary.py` (`_run_post_pipeline`, `real_streamlit`,
`_app_exceptions`).

What the test keeps real and what it replaces:
- **Real:** `map_payload`, at Stage 2B and at Stage 6. Nothing in the app is replaced.
- **Lookups:** protein and compound lookups are answered from a name-keyed table, with no database or network
  (`HttpClient.get` raises), and `MappingCache.save` is a no-op.
- **Enrichment:** a copy of its input.
- **Curator (offline in every deterministic run):** replaced by the audit's stand-in. After Stage 3 it gives the
  wrapper's component the survivor's UniProt, which is ORCH-759 E1 step 3.

The payload is the boundary suite's glutathione core plus the E1 shape:
- a second GSS row, `glutathione synthase`, sharing P48637 and carrying its record's names as synonyms, as row 7
  did;
- a generated wrapper whose one component is that row.

A spy on `stage_one_ensemble.merge_on_mapped_identity` records `_pooled_draws()` at the moment the app's Stage-6
pass called it: `[0]` at both `T2PW_EXTRACTION_DRAWS=1` and `=3`. So rule 3 was off (ORCH-758 D4) and the fold fired
anyway. Asserted on the app's own session state:
- `refinement_gate_errors == []` and `final_stage3_gate_report.ok is True`;
- `mapping_report["duplicate_protein_folds"]` = `[(glutathione synthetase, glutathione synthase, P48637)]`;
- the canonical payload has no `glutathione synthase` row, and the survivor lists it as a synonym;
- `refinement_mapping_misses`, from which the interactive curator's prompt is built, equals
  `compact_mapping_misses` of the report without the key.

| run | report |
|---|---|
| tip, each node alone | `08` (one_draw) and `09` (three_draws): 1 passed each, at `e346bb6e` |
| base `src/` (b06b86b5's three files swapped into the worktree) | `10` and `11`: 1 failed each, on the app's own verdict, `refinement_gate_errors` = `[{path: /entities/proteins/2/name, reason: "Protein has degree 0 after normalization: glutathione synthase"}]` |

So the app-level reproduction is real at base, not just at the tip. The orchestrator's live enterobactin 3-draw
re-run after merge remains the end-to-end proof.

## 4. Where the fold is recorded, and why it reaches no LLM

**Location:** the Stage-6 mapping report, key `duplicate_protein_folds` (a list), present only when a fold happened.
It is visible in the app's `tmp/mapping_report.json`, in the "Download mapping_report.json" button, and in
`st.session_state["post_pipeline_artifacts"]["mapping_report"]`.

Each entry holds:
- `rule` (`unreferenced_protein_row_shares_an_admitted_accession`);
- `survivor`;
- `folded`;
- `shared_accessions` (e.g. `{uniprot: P11454, pathbank_protein_id: 6312}`);
- `species` (`{name, taxonomy_id, pathbank_species_id}`);
- `alias_added`;
- `quotes_moved`;
- `folded_row_names` (every spelling the folded row carried, so none is lost).

The alias on the survivor is data, not a record. The payload carries no record.

**The only LLM that reads the mapping report** is the interactive curator. The app stores
`compact_mapping_misses(mapping_report)` as `refinement_mapping_misses` (`streamlit_app.py:1883-1884`).
`build_interactive_curator_messages` walks it again (`interactive_curator.py:273`), and so does
`run_interactive_curator_round` (`streamlit_app.py:2832`). Every other consumer checked does not reach a model:
- the gap-resolution UI (DB only, `status == "unmapped"` logs);
- the debug expander;
- the RAG gap detector, which runs before Stage 6 and reads only `status == "unmapped"` entries;
- the batch driver, which does not persist the mapping report.

Two independent closures:
1. **The walker never descends into the key.** It is a member of `PROMPT_OMITTED_PAYLOAD_KEYS` and the new skip in
   `walk` (`interactive_curator.py:215-217`). This holds whatever the entries contain:
   `test_new_the_record_key_is_skipped_whatever_it_holds` puts entries shaped exactly like misses (`status:
   unmapped`, `reason: ambiguous`, `confidence: 0.1`) under it, top-level and nested, and they are not collected.
   Mutation M1 (remove the skip) turns that test red (`14`).
2. **If an enclosing dict were ever copied as a record,** `strip_payload_for_interactive_context` already drops
   `PROMPT_OMITTED_PAYLOAD_KEYS` at every depth.

`test_new_the_fold_record_reaches_no_llm_input` replays the leg and builds the curator's whole message list from the
report, with and without the record. The two are byte-identical, and so are the two `compact_mapping_misses`
outputs. No report or payload produced before this lane carries the key, so no existing prompt changes by one byte
(D-104 AMENDMENT 2): the c079 allowlist suite, `test_interactive_curator.py` and the LLM-input goldens pass (`15`, `16`).

**The Reaction Evidence report cannot show the fold.** It reads only a run's own artifacts: `final_mapped.json`,
the quarantine set, `pwml_ir.json` and `LEG_TRACE.jsonl` (`app/reaction_evidence.py:1-25`). None of them carries
the mapping report, and the batch driver does not write one. Showing the fold there needs a decision (§ 7 item 1).
I did not widen this lane to make it.

## 5. Proofs and gates

### 5.1 G9: RED at the test-first commit, GREEN at the tip

The G9 proof is `test_g9_the_stage6_pass_leaves_a_payload_the_final_gate_passes`. It feeds the leg's own payload
through `map_payload`'s real Stage-6 entry (`merge_identity_duplicates=True`). Each protein row's recorded
resolution is replayed as the resolver's answer (`_replayed_lookup`); every other door is shut. It has two inputs:
- `leg_final_mapped_as_shipped`;
- `stage6_input_reconstructed`: the wrapper's component named back to `enterobactin synthase` with P11454, the
  diagnosis's variant A. Here map_payload's own reconciliation performs the rename.

The fixture is `tests/fixtures/orch759a/enterobactin_final_mapped.excerpt.json`. It is the leg's file with the
`enrichment` blocks and `source_text_index` removed (`evidence/orch759a/make_fixture.py`; recipe and hashes in the
fixture README). Precondition test `test_the_excerpt_carries_the_legs_defect` holds at base and at the tip: the
excerpt fails the gate exactly as the leg did, `["Protein has degree 0 after normalization: enterobactin synthase"]`.

| | report | result |
|---|---|---|
| RED at `30942f32` (test-first, base code) | `01-g9-red` | 2 failed, 1 passed. Both G9 cases fail on the gate assertion: `Left contains one more item: 'Protein has degree 0 after normalization: enterobactin synthase'` |
| GREEN at `4a6d9a6c` | `02-g9-green` | 33 passed (the whole file, G9 + new acceptance) |

At the tip, the fold record names survivor `EntF`, folded `enterobactin synthase` and shared uniprot `P11454`.
Species is `Escherichia coli`.

### 5.2 Non-vacuity: nine mutations of the fix, all killed (`14-mutations`, `evidence/orch759a/mutations.py`)

Each mutation turns its target tests red:

| # | mutation | result |
|---|---|---|
| M1 | the walker reads the record key | killed |
| M2 | the survivor need not be referenced | killed |
| M3 | species ignored | killed |
| M4 | references ignored | 7 red |
| M5 | every alias counts as a reference | killed, including both G9 cases |
| M6 | raw `mapped_ids` instead of admitted ids | 4 red: low_confidence, best_effort, placeholder, unresolved. Candidate_only and withheld carry no `mapped_ids`; the real-pass-A test covers withholding |
| M7 | gated on rule 3's `_pooled_draws()` | killed: the 3 draw-count tests and the app-path test |
| M8 | quotes not moved | killed |
| M9 | pointers not moved | killed |

Every source file is restored byte for byte; the script checks this with `git diff --quiet`.

### 5.3 Gates

| gate | report | result |
|---|---|---|
| G9 RED / GREEN | `01` / `02` | 2 failed / 33 passed |
| new acceptance, app path, each node alone, tip | `08`, `09` | 1 + 1 passed |
| same, base `src/` | `10`, `11` | 1 + 1 failed, on the degree-0 gate error (base behaviour recorded, not claimed as G9) |
| new acceptance file, base `src/` | `07` | 28 failed, 5 passed. The 5 are the preservation tests and the precondition, which hold at base too |
| mutations | `14` | 9/9 killed |
| affected sweep A (29 files) | `15` | 710 passed, 4 failed, 3 skipped |
| affected sweep B (28 files) | `16` | 776 passed, 1 failed |
| SMOKE (22 files) | `17` | **508 passed** |
| `tests/test_eval_v2_*.py` (6 files) | `18` | 89 passed, 1 xfailed |
| each alone: `test_streamlit_quarantine_boundary.py` | `19` | 23 passed |
| each alone: `test_c052_prefreeze_report_at_the_streamlit_seams.py` | `20` | 13 passed |
| each alone: `test_c055_rag_loop_wiring.py` | `21` | 14 passed |
| each alone: `test_reaction_evidence_view.py` | `22` | 11 passed |
| each alone: `test_reaction_evidence_app_hook.py` | `23` | 1 passed |
| each alone: `test_batch_preflight.py` | `24` | 32 passed, 1 failed, 4 skipped |
| affected AppTest files, each alone: `test_batch_research_gate_fail_open.py`, `test_streamlit_stage8_export_contract.py` | `25`, `26` | 14 / 4 passed |
| the 6 failing nodes on base `src/` | `27` | all 6 fail at base |

The affected set is `grep -l -E "map_payload|map_ids|identity_admission|reconcile|protein_has_degree" tests/`. To it I
added every file touching `interactive_curator` / `compact_mapping_misses` / `prompt_payload_keys`, plus
`test_stage_one_ensemble.py`, `test_stage_one_paper_dictionary.py`, `test_reaction_evidence.py` and `test_pwml_ir.py`:
63 files in all. The AppTest files ran alone; `test_batch_driver.py` ran in SMOKE. No gap-resolver file
(`test_gap_resolver*.py`) is in the set. The exact file lists are on each report's command line.

The 6 failing nodes, all in the known-failure lists and all red on base `src/` in `27`:
- `test_c075_source_support_armed.py::test_corpus_the_armed_pass_over_every_committed_artifact_and_its_own_paper`;
- `test_c100_tolerance_scope.py::test_5b_the_bare_pathbank_sentinel_name_is_not_a_tolerated_entity`;
- `test_compound_resolution_extraction.py::test_the_golden_covers_every_committed_leg_fixture`;
- `test_compound_resolution_extraction.py::test_the_repaired_coverage_and_digest_pins_themselves_go_red`;
- `test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform`.

  These five are in TEST_MATRIX § ORCH-754 "Known failures".
- `test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure`: worktree-only
  (no `.venv`), § ORCH-757 / § ORCH-757i.

The 3 skips in `15` are the c081/c089 corpus tests, which skip any path containing `worktrees` (§ ORCH-754
"Path-sensitive skips"). Chunk D was not run (the orchestrator runs it). No `--heavy-lock` was used, and jobs ran one
at a time.

## 6. Findings, failures and dead ends

- **F1 — an offline replay with every door shut cannot test this.** I first replayed the leg through `map_payload`
  with DB `None` and network raising. Every protein then came back `unresolved` with `mapped_ids {}`, and the gate
  still showed only the degree-0 error. Without identities there is nothing admitted to fold on. Hence the G9
  harness replays each row's own recorded resolution instead. Scratch run, not committed:
  `C:/t/bt-orch759a/replay1.{json,log}`, 0 survivors.
- **F2 — the first app-path test did not reproduce E1 at base.** Its duplicate row had no synonyms. At base the
  run passed the gate (`06`: the test failed only on the absent record, `KeyError: 'duplicate_protein_folds'`).
  Debugging that run showed the orphan in `final_mapped_db` but gone from the canonical payload: quarantine had
  pruned it. That led to the name-versus-synonym finding in § 1. The test was corrected in `e346bb6e` and now
  reproduces E1 at base (`10`, `11`). Reports `03`-`05` are the earlier version at the tip, all green. In `03` both
  nodes shared one process, and the second printed the known `FragmentThreadState not initialized` harness fault
  while rendering later UI. `_app_exceptions` filters it, as the boundary suite documents; the assertions had
  already passed. Every later app run is one node per process.
- **F3 — the first app-path lookup stub reused `_replayed_lookup`.** For rows with no recorded candidates, the
  real name gate rejected its answer and stripped the seeded UniProt at Stage 2B, so every protein became an
  Unknown placeholder. I replaced it with a name-keyed table that returns a candidate record. Found with scratch
  debug runs, not committed.
- **F4 — two mutation-harness attempts were invalid, not results.** In `12`, the multi-line patterns did not match
  the CRLF working copy: the worktree's `core.autocrlf` rewrote the sources on `git checkout`. In `13`, `--basetemp`'s
  parent did not exist, so `tmp_path` tests ERRORed and M9 was mis-scored "survived". `14` is the corrected harness.
- **F5 — the mapping report's protein census is computed before the fold, as rule 3's is.** `summary.proteins_total`,
  `proteins_mapped` and `protein_export_policy.summary.verified_real_proteins`, in both the report and the payload,
  still count the folded row. On the leg that is 8, while 7 rows ship. I did not change it (§ 7 item 4).
- **F6 — what was ruled out.** The fold does not depend on the record. G9 and every acceptance test check the
  payload effect (the row is gone, the alias is present, the gate passes) separately from the record.

## 7. Not settled — for the product owner

1. **The fold record is not persisted in batch leg directories.** The batch driver writes `final_mapped.json` but
   no `mapping_report.json`. A batch leg therefore shows the effect (the row is gone and the alias is on the
   survivor) but not the record.
   - (a) The driver writes `mapping_report.json` into the leg. This is about 3 lines plus a test; the mapping
     report is not an LLM input in batch.
   - (b) Additionally, the Reaction Evidence report renders a "duplicate protein rows folded" section when that
     file is present. This is presentation only, and the lane needs (a) for batch legs.
   - (c) Leave it as is: the record is in the app's mapping report only.
2. **Research mode also folds.** The app passes `merge_identity_duplicates=True` in research mode too, and the
   ruling gates the fold on that keyword only. Research mode otherwise keeps unused proteins and flags them for
   review.
   - (a) Keep, as now: the row is the same protein, its name and quotes are kept, and the fold is recorded.
   - (b) Skip the fold in research mode: one condition and one test.
3. **The quarantine pruner and the final gate disagree about references** (§ 1). The pruner uses name ∪ synonyms;
   the final gate's degree test uses the name only. This lane removes the E1 case. Any other synonym-matched orphan
   that shares no admitted accession, or has another species, still passes quarantine and stops the run at the
   final gate. Correctly, it is a degree-0 row, but it is late.
   - (a) Leave it.
   - (b) Make the gate's degree test synonym-aware. That is a gate change, and D-106 says no gate change without a
     ruling.
   - (c) Make the pruner name-only for proteins. That deletes such rows earlier; it risks undoing F-083's
     synonym-aware complex handling.
4. **The census is not recomputed after the fold (F5).**
   - (a) Recompute the protein counts after the fold: small, but it touches the report summary the curator may read.
   - (b) Leave it, as rule 3 does.
5. **The folded row's `provenance_lineage` is not carried anywhere.** The record keeps its names, accessions,
   species and quotes.
   - (a) Add the lineage to the record entry (presentation only).
   - (b) Leave it.
6. **Moved quotes go to the survivor's `source_refs`.** On a RAG-imported survivor that field holds source ids,
   not quotes (`identity_admission.py:252-260`). Identity admission does not read it as a route, so nothing
   changes there; but the field mixes two kinds of value.
   - (a) Accept.
   - (b) Use a dedicated field. It would be in the payload, so it needs the same LLM-input argument as § 4.

## 8. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

Exact text to append at end of file:

```markdown
## ORCH-759a (E1): test obligations — appended end-of-file by rule · 2026-10-22

Lane `orch759a/fold-duplicate-protein` (`30942f32` test-first, `4a6d9a6c` code, `e346bb6e` app-path test
correction). Reports `evidence/g11/ORCH-759a/01`–`27`.

**SMOKE is unchanged at 508** (`g11/ORCH-759a/17`). No pin moved.

### New focused selection

- `tests/test_orch759a_duplicate_protein_fold.py` (**33**, <1 s), fixture
  `tests/fixtures/orch759a/enterobactin_final_mapped.excerpt.json`. G9: 2 `test_g9_*` red at base (`01`), 33 green
  at tip (`02`). Run it for any change to `map_payload`'s Stage-6 tail, `_reconcile_components_against_local_proteins`,
  `stage_one_ensemble._admitted_ids`, `_species_hint_from_row`, `compact_mapping_misses` or
  `PROMPT_OMITTED_PAYLOAD_KEYS`.
- `tests/test_orch759a_fold_app_path.py` (**2**, AppTest, ~3 s per node): the fold through the real app's
  Stage-6 call at 1 and 3 draws. **Run each node in its own process** (two in one process trip the known
  `FragmentThreadState` harness fault in later UI rendering; filtered, but run them apart). Imports the harness of
  `test_streamlit_quarantine_boundary.py`. Red on base `src/` (`10`, `11`).

Neither file is in a chunk or in SMOKE; Chunk D's partition is unchanged.

### Known failures

Unchanged from § ORCH-754, § ORCH-757 and § ORCH-757i; all six seen here are red on base `src/` (`27`).
```

## 9. Files changed

| file | commit | what |
|---|---|---|
| `src/t2pw/mapping/map_ids.py` | `4a6d9a6c` | the fold and its call (§ 2) |
| `src/t2pw/curation/prompt_payload_keys.py` | `4a6d9a6c` | `DUPLICATE_PROTEIN_FOLDS_KEY`, in `PROMPT_OMITTED_PAYLOAD_KEYS` |
| `src/t2pw/curation/interactive_curator.py` | `4a6d9a6c` | the walker skip |
| `tests/test_orch759a_duplicate_protein_fold.py` | `30942f32` (G9), `4a6d9a6c` (acceptance) | G9 + new acceptance tests |
| `tests/test_orch759a_fold_app_path.py` | `4a6d9a6c`, `e346bb6e` | real-app proof |
| `tests/fixtures/orch759a/` (excerpt + README) | `30942f32`, README hash fixed in `4a6d9a6c` | leg excerpt |
| `docs/pwml_recovery_sprint/evidence/orch759a/make_fixture.py` | `30942f32` | excerpt recipe |
| `docs/pwml_recovery_sprint/evidence/orch759a/mutations.py`, logs `01`-`27`, `evidence/g11/ORCH-759a/`, `evidence/g11/pin/ORCH-759a/` | docs commit | evidence |
| this file | docs commit | record |

## 10. Repair round 1

Review round 0 **rejected** the lane on two blocking findings under standard item 1: "the fold can fire on a
referenced row, or drop a quote". It also asked for two additions to the record. Nothing else changed.

| | |
|---|---|
| tests first, on the round-0 code | `03673599` |
| code | `f7790501` (`map_ids.py` only, +36 −9 against `493c1a10`) |
| docs/evidence | the commit that adds this section |

### The findings and the fix

- **B1 — a referenced row could fold.** In round 0, `referenced()` excused an alias whenever ANY other protein or
  complex row also held it, including an unrelated protein. The reviewer's probe 34:
  - the duplicate (P11454) and an unrelated protein (P99999) both list `shared alias`, and a reaction enzyme names it;
  - `strict_quarantine._build_registry().resolve("shared alias")` returns the duplicate, yet the fold fired;
  - after the fold the alias resolved to P99999, so the reaction's enzyme had moved to another protein.

  **Fix:** an alias is excused only when, after the fold, it still resolves to the same thing:
  - it is another protein's or complex's own **name**, or
  - every other protein holding it shares this row's admitted identity (`same_identity`: at least one admitted
    accession shared, none disagreeing).

  The name and the `key` always count. I added one condition beyond the reviewer's rule: a complex holding the
  alias as a synonym (not its name) also keeps the reference. A registry lookup could otherwise move the reference
  to that complex. This only blocks folds; it never permits one.
- **B2 — quotes were dropped silently.** `_entity_quotes` read only plain-string `evidence` / `source_refs`, so
  `evidence_quote` and `source_evidence` vanished (probe 33, `quotes_moved: []`).

  **Fix:** it reads `_ENTITY_QUOTE_KEYS = ("evidence", "source_refs", "evidence_quote", "source_evidence",
  "source_text")`. These are `entity_admission._spans`' keys plus `apply_audit_patch._ACTOR_ROW_EVIDENCE_KEYS`.
  Each may be a string or a list, and a dict entry contributes its `span`, `text` or `quote`. `quote` is one key
  beyond the request; it is the dict key `app/reaction_evidence._quotes` also reads.
- **N1:** each fold record carries `folded_mapped_ids`, the folded row's whole `mapped_ids`.
- **N5:** each fold record carries `folded_provenance_lineage`, `folded_rag_provenance` and `folded_source_papers`.
  The record stays under the skipped key, so § 4's LLM argument is unchanged.

These answer § 7 item 5 (lineage) with option (a). The other § 7 items stand.

### New tests (new acceptance, in `tests/test_orch759a_duplicate_protein_fold.py`, now 37)

- `test_new_an_alias_held_by_an_unrelated_row_still_names_the_duplicate[unrelated_protein, complex_alias]` (B1).
  It asserts the registry resolves `shared alias` to the duplicate before the fold, then that nothing folds and the
  payload is byte-identical.
- `test_new_an_alias_held_only_by_rows_of_the_same_identity_is_not_a_reference` (B1's other side: EntF holds the
  alias, so the fold fires).
- `test_new_every_quote_field_moves_with_the_fold` (B2: all five keys and dict `span`/`text` entries; a quote the
  survivor already holds does not move twice).
- `test_new_the_fold_keeps_the_name_and_every_quote_and_records_them` now also asserts the N1/N5 fields.

### Gates (round 1)

| gate | report | result |
|---|---|---|
| repair tests on round-0 code (`03673599`) | `28-r1-red` | **4 failed**, 33 passed. The failures: both B1 cases, the B2 case, and the record test (N1/N5). The same-identity case passes there too, as the preservation side should |
| G9 + acceptance at `f7790501` | `29-r1-g9-green` | **37 passed** |
| app path, each node alone | `30`, `31` | 1 + 1 passed |
| SMOKE | `32-r1-smoke` | **508 passed** |
| affected sweep A (29 files) | `33` | 710 passed, 4 failed, 3 skipped |
| affected sweep B (28 files) | `34` | 780 passed, 1 failed |
| mutations (M5 updated to the new line; M10 and M11 added: round 0's B1 and B2 behaviour) | `35-r1-mutations` | **11/11 killed**, sources restored |

The 5 failures in `33` and `34` are the same nodes as round 0, all red on base `src/` in `27`. Every report shows
`FINAL SURVIVING COUNT : 0`.

### TEST_MATRIX delta: supersedes § 8's numbers

In the § 8 text, `tests/test_orch759a_duplicate_protein_fold.py` is **37** tests (was 33). Its G9 line becomes:
"G9: 2 `test_g9_*` red at base (`01`), 33 green at round 0 (`02`), 37 green at repair round 1 (`29`); the four
repair-round tests red on the round-0 code (`28`)". Reports run to `35`. Everything else in § 8 stands.
