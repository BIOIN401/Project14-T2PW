# ORCH-769d — audit-patch index rebasing (KI-018, D-113 orchestrator ruling (a))

Not to be confused with `ORCH-759d-RESULT.md`, an older lane.

| | |
|---|---|
| Branch | `orch769d/audit-rebase`, worktree `.claude/worktrees/orch769d` |
| Base | `3ef3fa9c` (sprint tip; `src/` = `main`) |
| Commits | `96848d43` tests (6 G9) · `2a791f69` **the fix** · `2b0ea179` one pinned test moved (merge rule 4) · `c03ab59f` 7th G9 test · docs commit (this file + evidence) |
| Diff vs base | `src/t2pw/curation/apply_audit_patch.py` (+129/−17), `tests/test_orch769d_audit_patch_rebase.py` (new, 7 tests), `tests/test_apply_audit_patch_referential_integrity.py` (one test's expectations). Nothing else. `streamlit_app.py`, `strict_quarantine.py`, `reaction_evidence.py`, `evidence/orch768_replay_corpus.py`: untouched (`git diff --stat 3ef3fa9c HEAD`). |
| G11 | `evidence/g11/ORCH-769d/01`–`17`, pins `evidence/g11/pin/ORCH-769d/`; `g11_evidence.py check --task ORCH-769d` exit 0, 17 artifacts, 0 non-compliant, every pytest run pinned. Pytest stdout of each run: `evidence/orch769d/<SEQ>-<label>.out.txt` |
| Status | implemented, all gates green; awaiting independent review |

## 1. The defect, and where the leg pins it

`apply_patch_with_policy` applied every op at its written index. After an accepted `remove` of an array row (or an
insert), each later op on that array landed one row off. ORCH-769 § (d) diagnosed this on ORCH-767 Wnt/Ca2+
(PMC10425814) by counterfactual. This lane found the **recorded** evidence in the leg:

`final_mapped.json` carries an `audit_repair` lineage entry on exactly the rows an accepted patch op wrote
(G11 `02`, `evidence/orch769d/02-peek-lineage.out.txt`). Protein rows 2, 3, 4, 6, 7, 8, 9 carry a synonym that is their
own (short) name. Rows 13, 14, 17, 18, 19, 20 each carry **the name of the row before them**:

| row | name | recorded synonym | meant for |
|---|---|---|---|
| 13 | β-catenin | `TCF` | row 12, TCF |
| 14 | calmodulin | `β-catenin` | row 13 |
| 17 | calcineurin B | `calcineurin A` | row 16 |
| 18 | **phosphorylated CAMKII** | **`calcineurin B`** | row 17 |
| 19 | phosphorylated TCF4 | `phosphorylated CAMKII` | row 18 |
| 20 | NF-ATp | `phosphorylated TCF4` | row 19 |

Rows up to 9 are right and rows from 13 on are shifted, so the removed row sat at pre-patch index 10..13. The audit
rationale says it removed the protein `calcineurin` (also declared as a complex; `audit_iteration_summary.json`), and
`stage1_payload.json` puts `calcineurin` directly after SERCA (G11 `01`). So the auditor removed
`/entities/proteins/10` and wrote every later op against the pre-patch payload. The synonym `calcineurin B` kept the
unidentified `phosphorylated CAMKII` referenced, and the quarantine refused the pathway (`unexportable_entity:1`).

**What the leg does not record:** the patch itself, its op order, the op confidences, and the audit's input payload.
The audit summary counts 19 ops, 18 accepted, 1 rejected. The lineage shows 18 written rows (13 protein synonyms,
compound 3, compound 4, reaction 3, transports 9 and 10); with the remove, that is 19 accepted ops if each op wrote
one row. The two do not reconcile exactly (one op may have written two rows, or the gap resolver, which uses the same
applier and the same lineage stage, wrote one). Not resolved.

## 2. Design

**Semantics (ruling (a)).** Every op in one patch addresses the payload its author saw: the pre-patch snapshot.
Evidence in the code that this is what callers assume: the curator and the gap resolver dedupe ops on `(op, path)`
(`pathway_curator.py:334-341`, `gap_resolver.py:2753-2768` at `3ef3fa9c`), and `audit_json_llm.score_audit_candidate`
reads remove paths as snapshot indices against the payload's locked indices. The deterministic auditor emits its own
removals in descending index order, which means the same rows under either reading. All patch producers in `src/`
are LLM outputs plus that deterministic auditor; I found no producer that relies on sequential (RFC 6902) index
semantics.

**Mechanism** (`apply_audit_patch.py`, block comment "Index rebasing"):

* `shifts`: per array, keyed by the array's path **in pre-patch coordinates**, the list of `(remove|add, live index)`
  events made by accepted ops so far.
* `_rebase_op(op, shifts)`: walks each array-index token of the op's path through that array's events in order
  (`remove at r`: an index above `r` drops by one; `add at k`: an index at or above `k` rises by one). If a token
  equals a removed row, the op is rejected with `removed_target: ...`. The last index of an `add` is an insertion
  position, not a row, so a removed row there is not a rejection. **When nothing moves it returns the same op
  object**, so the rest of the loop sees exactly what base saw.
* `_record_index_shift`: after an op is accepted and committed to `working`, records an event when the op removed
  or inserted an array element (judged on the pre-op payload). Any other accepted op rewrites the node at its path
  wholesale, so shifts recorded for arrays under that node are dropped (see § 6 item 3).
* In the loop: the **policy** (`_should_accept`) is handed the op **as written**, with `source_payload`, exactly as at
  base, so no verdict it gives can change. The lock validators, `_apply_single_op`, the referential-integrity guard
  and lineage get the rebased op (`live_op`). A removed-target op is rejected **after** the policy, so an op the policy
  refuses keeps the policy's reason. It goes to `rejected` and the rejected log like any other rejection.
* Records: `op` stays as written; a rebased op's record and applied-log entry gain `applied_path` (only when it
  differs).

Two stale comments that described index drift as expected behaviour were corrected (`:826-834` at `3ef3fa9c`, and
the last comment in `_should_accept`).

**Not done, deliberately:** the `/-` on a missing key defect (§ 5, F-2). It lives in `_resolve_parent`/`_set_value`,
not in `apply_patch_with_policy`, and any fix changes behaviour for a patch with no index-shifting op, which the
lane's severity standard makes blocking. Recorded, with options, in § 7.

## 3. G9 proofs (all corrections; no test imports a symbol the fix adds)

`tests/test_orch769d_audit_patch_rebase.py`:

| test | base `3ef3fa9c` (behavioural failure) |
|---|---|
| `test_g9_a_later_op_hits_the_row_it_was_written_for_after_a_remove` — `[remove /x/1, replace /x/2/note]` | the note lands on D, not C |
| `test_g9_an_op_aimed_at_the_removed_row_is_rejected_not_redirected` | the note meant for `dup` lands on C |
| `test_g9_a_repeated_remove_does_not_delete_the_neighbour` | C is deleted too: `['A', 'D']` |
| `test_g9_an_op_aimed_at_the_removed_row_cannot_compensate_the_removal` | commits with `X` as a synonym of B |
| `test_g9_ops_before_the_removed_row_and_inserts_are_rebased_correctly` | the insert appends after D; D's note lands on the new row |
| `test_g9_the_integrity_guard_judges_the_rebased_row` | base removes and refuses `hydrolyzed meropenem` instead of `meropenem` |
| `test_g9_wnt_ca_reconstruction_every_synonym_lands_on_its_own_row` (labelled reconstruction, § 4) | `phosphorylated CAMKII` gets `['calcineurin B']`, as on the leg |

* **Base red:** G11 `03` (6 failed, the first six) and `16` (all 7 failed), both on a `git archive 3ef3fa9c src
  pyproject.toml pytest.ini tests/conftest.py` export plus the test file, run with `--expect-tree C:/t/orch769d/base`;
  the pin verdicts resolve `t2pw` to `C:\t\orch769d\base\src\t2pw`. Failure lines in
  `evidence/orch769d/03-g9-red-base.out.txt`, `16-g9-red-base-7.out.txt`.
* **Tip green:** G11 `04` (6 passed at `2a791f69`) and `17` (660 passed at `c03ab59f`, which includes all 7).

## 4. Wnt/Ca2+: the applier output for that leg, base vs tip

The corpus cannot measure this fix: the audit loop is step 2 of `run_post_pipeline_sbml_artifacts`, before the
`final_mapped.json` checkpoint the replay corpus starts from (ORCH-768-RESULT § 1, § 9 item 1; same class as its
F-6). So I replayed the applier itself on the leg.

`evidence/orch769d/replay_wntca_audit_patch.py` (**a reconstruction, labelled as such**): it rebuilds the audit input
from the leg's `final_mapped.json` (strips the audit-written synonyms and their lineage entry from the 13 protein rows
and compound 3, and puts `{"name": "calcineurin"}` back at protein index 10), rebuilds the patch (the remove, one
`add .../synonyms` per recorded row at the index it was recorded on, and the recorded `add
/entities/compounds/3/synonyms/- "Ca2"`), applies it, then runs the strict quarantine and the final pre-export gate the
way the ORCH-769 diagnosis did. The four other audit-written rows are left as recorded.

| | base `src` (G11 `10`) | tip `src` (G11 `11`) |
|---|---|---|
| applier | 15/15 accepted, committed | 15/15 accepted, committed; 6 ops rebased (`applied_paths` 12, 13, 16, 17, 18, 19) |
| reproduces the recorded `final_mapped.json` | **yes**: protein rows and compounds equal ignoring lineage; lineage stage multisets equal | no: 8 rows differ, every synonym on its own row |
| cross-row synonyms | 6 (the table in § 1) | **0** |
| `phosphorylated CAMKII` synonyms | `['calcineurin B']` | `['phosphorylated CAMKII']` |
| quarantine | **refused** `unexportable_entity:1` (`phosphorylated CAMKII`, `protein_missing_external_identity`), release `diagnostic_only` | **ok**, release `review_required` (`requested_core_coverage_below_minimum:0.429<0.500`) |
| surviving proteins / processes | 11 / 0 reactions, 9 transports, 13 interactions | 10 (`phosphorylated CAMKII` gone) / 0, 9, 13 |
| final pre-export gate | not run | **PASS** |

Results: `evidence/orch769d/10-wntca-replay-base.result.json`, `11-wntca-replay-tip.result.json`. The base row matches
the diagnosis's baseline A1 (same refusal, same counts) and the tip row matches its counterfactual B
(`evidence/orch769_diag/769d/2-diag.out.txt`). Limits: Stage 6 and the gap resolver ran on the mis-patched payload in
the real leg, and their output is reused here; only a live re-run measures the whole tip pipeline (needs the LLM; not
authorised for this lane).

## 5. Findings

1. **F-1. The leg records which rows the audit wrote.** The `audit_repair` lineage entries localise the shift
   exactly (§ 1). This is how the reconstruction was checked: on base it reproduces the recorded rows (G11 `10`).
2. **F-2. `/-` on a missing key creates a dict, and it hit this leg too.** Compound 3 `Ca2+` carries
   `synonyms: {"-": "Ca2"}` in the recorded `final_mapped.json` (G11 `02`), reproduced at base and unchanged at tip
   (`compound_3_synonyms_after_apply` in both replay results). Cause: `_resolve_parent` creates `{}` for every missing
   dict key on the way down, and `_set_value` then stores the leaf `-` as a dict key. **Not fixed** (§ 2). Whether a
   dict-valued `synonyms` misleads a later stage (e.g. a registry that iterates it and reads the key `-`) was not
   investigated.
3. **F-3. The same auto-create reaches further than `/-`.** Any op through a missing key builds intermediate dicts:
   after `remove /entities/proteins`, an op `add /entities/proteins/2/synonyms` writes
   `{"proteins": {"2": {"synonyms": ...}}}` instead of failing. By reading `_resolve_parent`; not measured. Same
   decision as F-2.
4. **F-4. One existing test pinned the defect.** `test_apply_audit_patch_referential_integrity.py::
   test_index_drift_across_two_removals_is_judged_against_the_live_payload` asserted the unrebased result. At the tip
   with it unchanged: 1 failed, 619 passed (G11 `05`); the same 13 files at base: 620 passed (G11 `06`). Moved
   deliberately in `2b0ea179` (§ 6).
5. **F-5. A rejected removed-target op can roll back a whole batch.** The referential-integrity look-ahead
   (`_pending_entity_name_norms`, computed once from the ops as written) still credits a later op aimed at the removed
   row. That op is now rejected, so the removed name is orphaned and the end-of-batch check rolls everything back. At
   base the batch committed with the synonym on the neighbour. Pinned by
   `test_g9_an_op_aimed_at_the_removed_row_cannot_compensate_the_removal` (base red G11 `16`, tip green `17`). Correct
   under the ruling (no redirect), but the blast radius is the whole batch; see § 7 item 2.
6. **F-6. Collection-order hazard, one more victim.** G11 `08` stopped at collection: `test_orch758b_empty_chunk.py`
   cannot import `t2pw.llm` after the gap-resolver test files stub `t2pw.llm.client` (the known hazard,
   `TEST_MATRIX.md` § ORCH-757 "Collection-order hazard"). Rerun with the gap-resolver files last: 283 passed (G11
   `09`). Not caused by this lane.
7. **F-7. Heavy-lock contention, no kills.** Two first attempts exited 95 (lock held by other lanes; the first named
   `ORCH-773-rev`); no child ran, nothing was removed. I waited for `holder.json` to disappear and retried
   (`evidence/orch769d/run_when_free.sh`: waits for the lock and >= 6 GB free RAM, retries on 95). Orphan sweeps after
   every job found no sprint `python.exe` outside a running wrapper; **no process was killed**. Every report shows
   `FINAL SURVIVING COUNT: 0`.
8. **F-8. A dead end.** I first expected the leg to hold the audit patch (an `audit_report.json` or the response in
   `LEG_TRACE.jsonl`). It holds neither: the trace records request hashes only, and `merged_payload.json` is the
   post-pipeline input (46 proteins, no synonyms), not the audit input. Hence the reconstruction.

## 6. Pins and gates

**Pin moved deliberately (merge rule 4), count unchanged:**
`test_apply_audit_patch_referential_integrity.py::test_index_drift_across_two_removals_is_judged_against_the_live_payload`.
Same input `[remove /entities/compounds/0, remove /entities/compounds/1]`:

| | before | after |
|---|---|---|
| summary | accepted 1, rejected 1 (`referential_integrity`, meropenem) | accepted 2, rejected 0 |
| compounds left | spectator, meropenem, hydrolyzed meropenem | meropenem, hydrolyzed meropenem |
| added asserts | | 2nd accepted record `applied_path == "/entities/compounds/0"`; `validate_registry_references(patched)` |

The property it used to pin (the guard judges the live row) is now pinned by
`test_g9_the_integrity_guard_judges_the_rebased_row`.

**Severity item (2), no change for a patch with no index-shifting op:**
* By construction: with no accepted array remove or insert, `shifts` is empty and `_rebase_op` returns the op object
  itself, so every call after it gets the identical object, and `_should_accept` always gets the op as written.
* Measured: `evidence/orch769d/ab_no_shift_corpus.py`, seeded, run on base `src` (G11 `13`) and tip `src` (`14`),
  compared (`15`). **3000 / 3000 no-shift patches byte-identical** (payload and full report). Of 3000 shift patches,
  1014 have an accepted remove or insert at base; 307 differ, all 307 among those 1014, none outside.
* Every existing applier test file passes at the tip except the one moved pin (G11 `05` vs `06`, then `07`/`17`).

**Tests run (all through `bounded_run.py --heavy-lock ORCH-769d` and `pinned_pytest.py`, 0 survivors):**

| G11 | tree | selection | result |
|---|---|---|---|
| `03` | base export | new file (6) | **6 failed** (G9 red) |
| `04` | `2a791f69` | new file (6) | 6 passed |
| `05` | `2a791f69` | 13 existing applier files, unchanged | 1 failed (F-4), 619 passed |
| `06` | base export (`basefull`) | same 13 files | 620 passed |
| `07` | `2b0ea179` | new file + 13 applier files + `test_completeness_audit.py` | 659 passed |
| `08` | `2b0ea179` | 14 caller files (focused repair, gap resolver x5, curator, identity, species, DB resolver) | collection error (F-6) |
| `09` | `2b0ea179` | same, gap-resolver files last | 283 passed |
| `12` | `2b0ea179` | **SMOKE**, 22 files | **511 passed** (unchanged) |
| `16` | base export | new file (7) | **7 failed** (G9 red) |
| `17` | `c03ab59f` | new file (7) + 13 applier files + `test_completeness_audit.py` | **660 passed** |

The 13 existing applier files: `test_apply_audit_patch_{lineage,lock_policy,referential_integrity}.py`,
`test_audit_json_llm_payload.py`, `test_c105_actor_role_evidence.py`, `test_c107_actor_cue_calibration.py`,
`test_c108_f155_class.py`, `test_curation_patch_transaction.py`, `test_curator_offline_mode.py`,
`test_interactive_curator.py`, `test_orch757_llm_token_budgets.py`, `test_orch759a_duplicate_protein_fold.py`,
`test_streamlit_stage2_orchestration.py` (every test file that names `apply_audit_patch` or
`apply_patch_with_policy`). `src/` is identical from `2a791f69` to `c03ab59f`, so `09` and `12` hold for the tip.
No known-failure node was hit; nothing was red except F-4 and F-6. Not run: Chunk D, AppTest files alone, any
pipeline or LLM.

Other evidence runs: `01`, `02` (leg inspection), `10`, `11` (Wnt/Ca2+ replay), `13`–`15` (A/B corpus).

## 7. Not settled (for the product owner / orchestrator)

1. **The `/-` defect (F-2, F-3).** Options:
   (a) make it fail: the op is rejected and recorded ("Application failed"). Cost: a behaviour change for patches with
   no index shift (this leg's `Ca2` synonym would be dropped, not mangled).
   (b) create the list: `synonyms: ["Ca2"]`. Cost: the same class of change, and the name then counts in the registry,
   so it can keep a row referenced.
   (c) leave it. Cost: dict-valued fields keep flowing downstream (seen on this leg).
   Either (a) or (b) needs its own lane, because this lane's severity standard blocks any no-shift change.
2. **Whole-batch rollback (F-5).** Options: keep it (strict, simple); or drop a removed-target op from the look-ahead
   so the removal is refused on its own and the rest of the batch commits (smaller blast radius; a second change to
   the integrity guard).
3. **An op into an array that an earlier op rewrote wholesale** (e.g. `remove /x/1`, `replace /x` with a new list,
   then `add /x/2/...`). It is applied at its written index, as at base. A snapshot op cannot name a row of content
   that did not exist. Option: reject it instead. Cost: one more rejection rule. No producer is known to emit this.
4. **Insert, then address the new row by its index** (`add /x/1 {...}`, then `replace /x/1/name`). Under ruling (a)
   the second op means pre-patch row 1, now at 2; a model that wrote the patch sequentially meant the new row. This is
   the inherent cost of (a); the prompts do not say which reading is meant. Option: say "all paths address the payload
   as shown" in the audit, curator and gap-resolver prompts. Cost: prompt text changes (D-104 governs LLM-bound text).
5. **Live confirmation.** The replay (§ 4) reuses Stage 6 and gap-resolver output computed on the mis-patched payload.
   A live Wnt/Ca2+ leg on the merged tip would measure the whole pipeline; it needs LLM spend.
6. **`applied_path` is a new key** on rebased ops' records and applied-log entries. No consumer in `src/` validates
   those records' keys (grepped); readers outside `src/` were not checked.

## 8. LEDGER row (for the orchestrator to append)

`ORCH-769d | implemented, awaiting review | orch769d/audit-rebase @ <docs tip> (code 2a791f69, tests 96848d43/2b0ea179/c03ab59f) | KI-018: audit-patch ops rebased after an accepted array remove/insert (D-113 ruling (a)); op at a removed row rejected "removed_target", recorded. G9 7 tests base-red G11 03/16, tip-green 04/17. Wnt/Ca2+ applier replay (reconstruction): base refused unexportable_entity:1, tip review_required, gate PASS (G11 10/11). No-shift A/B 3000/3000 identical (13-15). SMOKE 511 (12). One pin moved (test_index_drift..., same input, exact delta in RESULT § 6). /- defect recorded, not fixed (RESULT § 7.1).`

## 9. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

* **New focused selection:** `tests/test_orch769d_audit_patch_rebase.py`, 7 tests, in no chunk and not in SMOKE. Run
  for any change to `apply_audit_patch.apply_patch_with_policy` or its pointer helpers.
* **Pin moved (merge rule 4), count unchanged:** `tests/test_apply_audit_patch_referential_integrity.py::
  test_index_drift_across_two_removals_is_judged_against_the_live_payload`, delta in § 6.
* **SMOKE unchanged at 511** (G11 `12`).
* **Collection-order hazard (§ ORCH-757):** add `tests/test_orch758b_empty_chunk.py` to the files that fail with
  `ImportError` when collected after the gap-resolver files (G11 `08`).
* **Applier coverage list** (13 files, § 6) for any future applier change: 620 tests at base, 627 at this tip
  including the new file (G11 `17` minus `test_completeness_audit.py`).
