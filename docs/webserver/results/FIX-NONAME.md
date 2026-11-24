# FIX-NONAME: Gemma runs abort at post_audit with `entity_missing_name`

Branch `fix/post-audit-empty-name`, cut from `main` 1a52fb55. Biology-code fix approved by the PO.
Code commit `336f9179` (code and tests). This record is a separate docs-only commit. Evidence: `docs/webserver/evidence/FIX-NONAME/`.

## 1. Symptom

Under Gemma 4 31B, 2 of 35 legs (0 of 19 deepseek) stopped at `stage_contracts.validate_post_audit`
with "Every entity row must have a non-empty name.":

| leg | flags | rounds | nameless row produced in |
|---|---|---|---|
| legs4g `vt1-PMC12096016` (enterobactin) | verifier on | 1 | audit round 1 (audit patch or gap enrichment) |
| legs4g `base1-PMC13231680` (lipid A) | all off | 2 | audit round 2 (audit patch or gap enrichment) |

Stage-1 and merged payloads have no nameless rows. The run folders did not record which row.

## 2. Root cause

`src/t2pw/curation/apply_audit_patch.py`, `apply_patch_with_policy` (1a52fb55 line 2755; the op is
applied at line 2821 and committed at line 2852) had no rule that an entity row keeps a name.
Every writer of model output into `/entities` inside the audit loop goes through it: the audit
patch (`run_apply`) and the gap-resolver enrichment agent (`gap_resolver.py:3341`). An op that
touches only an entity no process references (an orphan compound, the species row) also passes
the referential-integrity guard, which only protects names a process uses. So these
Gemma-shaped ops were accepted and handed a nameless row to post_audit:

- `replace /entities/species/0` with `{"pathbank_species_id": ...}` (an id-only row replaces the row);
- `add /entities/species/0 {...}` (an insert at an index meant as a field merge);
- `replace|remove /entities/<bucket>/<i>/name` with `""` or `null`;
- `add /entities/protein_complexes/- {"complex_name": ..., ...}` (a row under a non-`name` key).

How the step was pinned down, from the saved artifacts:

- PMC13231680 round 1: audit `error_count` 0. The deterministic audit raises an error for any
  nameless entity (`audit_json_llm.py:560-569`), so the round-2 input had none. Round 2: audit 0
  errors and 3 accepted patches; gap resolver `mapped_ids_added`, `locations_added`,
  `location_states_filled` and `organisms_added` all 0, `enrichment_patches_accepted` 1. The
  only changes in round 2 were patch ops, all through `apply_patch_with_policy`.
- PMC13231680 entity counts are unchanged between merged and post_audit apart from the
  normalizer's own `cell` location, so a row was blanked or replaced in place, not added.
- The raw model responses were not kept (`audit_iteration_summary.json` holds only the
  rationale; the tmp dir with `audit_report.json` was cleaned; the usage logs hold token counts
  only). The exact op is unrecoverable. The unit test reproduces each shape above on 1a52fb55.

## 3. Fix

`apply_audit_patch.py`:

- `_nameless_entity_counts` / `_nameless_entity_rejection` (lines 1262-1300): per bucket, rows whose
  name fails the post_audit test (`str(name or "").strip()` empty).
- `apply_patch_with_policy` judges the batch it would hand back (line 2925): if the committed
  result has a nameless entity row the source did not, the batch is re-run from pristine copies
  of the ops (line 2812) with a per-op guard (line 2890) that refuses the offending op with reason
  `entity_name_required: ...`, recorded in `rejected` and the rejected patch log.
- A batch that ends with every row named (a blank-then-rename pair included) or that rolls back
  anyway is applied exactly as before. A refused op keeps the row as it was: a blanked name keeps
  its prior name, a nameless row is never added. No row with a valid name is removed or renamed.

`service/postpipeline.py`: `_record_mapped_failure_snapshot` adds `failing_issues` (code, pointer,
bucket, the contract's bounded row digest; first 10 errors) to `mapped_failure_snapshot.json`, so
post_audit (and every other post-mapping contract abort) names bucket and index. The exception
message is unchanged.

## 4. Evidence

| check | result | log |
|---|---|---|
| new tests on 1a52fb55 (src stashed, same test file) | 7 failed, 2 passed: 6 shapes raise `StageContractError: Every entity row must have a non-empty name.`, snapshot test `KeyError: 'failing_issues'` | `unit-base-1a52fb55.log` |
| new tests + 5 patch-module test files on fix | 70 passed, 0 survivors | `unit-fix.log` |
| offline replay (section 5) | 0 not-allowed diffs | `compare.txt`, `stats.txt` |
| suite2 on HEAD 336f9179, clean tree | SMOKE 511 passed; golden batch 2, c011 8, stage8 4, quarantine 24; equivalence run 8, app 14, default 6; vision 370 passed 1 skipped; web 83; 0 survivors | `suite2.log` |
| 47 test files importing a changed module or a caller (`cover2.sh`: 9 groups, each its own process under `timeout 600`) | 1,269 passed, 3 failed; 0 survivors | `cover2.log` |
| the 3 failures, one per process, on HEAD and on 1a52fb55 `src/` (temporarily checked out, then restored) | identical on both trees: pre-existing, not this change | `three-head.log`, `three-base-1a52fb55.log` |

The 3 covering-set failures, all present on 1a52fb55:

- `test_c075_source_support_armed.py::test_corpus_the_armed_pass_over_every_committed_artifact_and_its_own_paper`:
  the corpus (`git ls-files *final_mapped.json`, 118 artifacts) now includes a committed
  PMC12096016 artifact, giving one unexpected catch, `Pantetheine 4'-phosphate`. Fails the same on base.
- `test_curator_offline_mode.py::test_a7_flag_is_read_in_exactly_one_module`: `T2PW_OFFLINE_CURATOR`
  is also read in `service/final_payload_compartments.py` and `vision/{generate,scope,serve}.py`,
  none of them touched here. Fails the same on base.
- `test_orch762_protein_identity.py::test_new_the_flag_never_reaches_an_llm_input`: failed only
  inside its batched group. It passes alone, on both trees, so the failure depends on test order.

A first covering run that put 44 files in one process stalled silently (the known
AppTest/curator stall) and held the heavy lock. It was stopped by killing my own `cover.sh` tree
by PID (`taskkill //PID 270680 //T //F`). The wrapper released the lock cleanly with 0 survivors.

No golden-seam file was edited.

## 5. No-regression replay

`replay.py` runs `apply_patch_with_policy` on the 1a52fb55 tree and on the fix tree over the same
inputs, then `post_audit`; `compare.py` diffs payload hash, report hash and post_audit verdict.

| case | source | n | identical | differs |
|---|---|---|---|---|
| `llm_patch` | every stored real audit patch `audit_patch.round*.cand*.json` in 38 `post_pipeline_*` tmp dirs, applied to its round input. Base reproduces the recorded `final.audited.round*.cand*.json` for 80 of 80 | 80 | 80 | 0 |
| `gap_patch` | stored gap-resolver enrichment ops (`gap_resolution_report.json`) | 6 | 6 | 0 |
| `det_patch` | deterministic audit patch of every `merged_payload.json` and `final_mapped.json` in legs3, legs4, legs4g, legs5g | 119 | 119 | 0 |
| `gemma_shape` | the same 119 payloads with an id-only species replace and a nameless complex append | 119 | 0 | 119, all allowed: base fails post_audit with `entity_missing_name`, fix passes |

No stored real patch triggers the new guard (`name_rejections` 0 on all 205 real/deterministic
cases). All 119 stored input payloads pass post_audit on both trees. The `post_pipeline_*` dir
inside `C:/t/t2pw-server` was not read (live server tree).

## 6. Bug B (secondary): chlorophyll `AMBIGUOUS_REFERENCE "tRNA Glu"` -- diagnosed, deferred

Leg `legs4g/base1-PMC5130987` (flags off) crashed in `t2pw.pwml.prefreeze_resolution`
(`prefreeze_resolution.py:886-899`, compound stage): the compound `tRNA Glu` was to be renamed to the
DB name `tRNA(Glu)`, but `nucleic_acids#0` is also named `tRNA Glu`, so a reference to the name
cannot be redirected unambiguously.

Cause: the merged payload declares `tRNA Glu` only as a nucleic acid; Stage-2 reaction 20
("Glu to Glu-tRNA Glu") uses it as an input. The deterministic audit's "add missing compound
referenced by a process" (`audit_json_llm.py:680-684`, adds at 1068-1093) counts a reference as
declared only if it matches a compound, element collection, protein or complex, not a nucleic
acid, so it adds a duplicate compound row `tRNA Glu`. `bugb_probe.py` confirms this on the stored
merged payload; `bugb_count.py`: 1 of 62 stored merged payloads gets such a duplicate (this one).

Deferred, not fixed. Adding `nucleic_acids` to that set is one line, but no-loss cannot be shown:
today the duplicate compound row is what gets DB-mapped and what the reaction input resolves to
in the export. Without it the input resolves to an unmapped nucleic acid, which strict mode can
quarantine. The prefreeze refusal itself is D-034/D-035 fail-closed policy and was not touched.

## 7. Dead ends

- Usage logs (`logs4g/*.usage.jsonl`) have token counts and costs only, no response text.
- `gap_iteration_summary.json` keeps `summary`, `db`, `stage3`; the `enrichment` block with the
  proposed ops is not recorded, so the enrichment op could not be told apart from the audit op.
- First draft of the guard was per-op, always on. Dropped: it would change a batch that blanks
  a name and then renames the row, and could change whether a batch rolls back. The final shape
  only runs when the committed result would carry a new nameless row.
- First draft of the re-run reused the caller's ops. Applied values are inserted by reference and
  then stamped with lineage, so the re-run would have double-stamped them; it now starts from a
  deep copy taken before the first pass.
- No live LLM calls. Spend $0.

## 8. Open questions (product owner)

1. Keep raw audit and enrichment responses? Neither is persisted to the run folder, which is why
   the exact Gemma op is unknown. Options: (a) write `audit_report.json` and the enrichment
   `patch_application` into the leg folder (some 10-100 KB per leg, all models); (b) keep only the
   rejected ops (small); (c) leave as is (the new `entity_name_required` rejections live only in
   the tmp dir).
2. Bug B: allow the one-line `nucleic_acids` change in `audit_json_llm.py` behind a flag and measure
   it on the nucleic-acid papers first? Cost: one A/B leg set. Risk: reaction inputs that today
   resolve to a DB-mapped compound would resolve to an unmapped nucleic acid.
3. A refused id-only species replace also drops the id the model found. A merge (keep the name,
   add the fields) would keep it, but that is a new repair, not a guard, and is not part of this fix.
