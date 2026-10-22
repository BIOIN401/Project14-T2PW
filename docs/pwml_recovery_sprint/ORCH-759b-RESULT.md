# ORCH-759b — the Stage-1 recorder is re-attached in the post-pipeline run (D-107)

**Status:** implemented and gated on the lane branch; **not reviewed, not merged.** An independent
reviewer judges the actual diff next. The live 3-draw proof (an app leg) is the orchestrator's.

| | |
|---|---|
| branch | `orch759b/recorder-reattach` (worktree `.claude/worktrees/orch759b`) |
| base | `4f750030` (= `sprint/pwml-recovery`) |
| test-first commit (base code) | `f9ea4a9a` |
| test amendment (base code) | `a2a804dc` (§ 8 F2) |
| code commit | `81a5c556` |
| docs/evidence commit | the commit that adds this file |
| charter | `DECISIONS.md` D-107 (option 1); D-104 § 3(c), AMENDMENTS 2 and 5; `ORCH-758-RESULT.md` § 4 D4 |

No LLM call and no pipeline run was made. Every number below names its G11 report under
`evidence/g11/ORCH-759b/` and its log under `evidence/orch759b/`. Line numbers in
`streamlit_app.py` are base `4f750030`'s unless stated (the tip adds one line at `:99`, one at `:5707`,
one at `:6692`).

---

## 1. The defect (ORCH-758 D4, reproduced here)

`merge_on_mapped_identity` (rule 3) applies only when `_pooled_draws()` > 1
(`stage_one_ensemble.py:1154`), which reads `current_diagnostics().boundaries`, and it files its
record through `_write_identity_side_file(current_diagnostics().artifact_dir, …)` (`:1374`). That
recorder is a ContextVar (`extraction_diagnostics.py:872`). "Run pipeline" activates it
(`streamlit_app.py:2042`, `_start_extraction_diagnostics`); "Run audit and DB mapping" is a later
script run on a new `ScriptRunner` thread, which starts from the ContextVar default. So in the
post-pipeline run:

- `_pooled_draws()` was 0 and rule 3 returned `{}` at the Stage-6 `map_payload(...,
  merge_identity_duplicates=True)` call: D-104 § 3(c) was not in effect in the app;
- the audit and gap rounds (`_record_audit_round` / `_record_gap_round`, `:2111-2125`, called at
  `:4131`, `:4143`) went into a fresh **detached** recorder that has no artifact directory, so they
  were never written under `tmp/extraction_diagnostics/run-*`;
- and (new, § 8 F1) the `finally:` `_sync_extraction_diagnostics()` of the post-pipeline handler then
  published that detached recorder's artifacts into session state, **replacing** Stage 0/1's.

Reproduced behaviourally at base by `test_g9_*` (§ 5.1).

## 2. The change

### 2.1 Helper, non-protected (`src/t2pw/pipeline/extraction_diagnostics.py`, +19 lines)

```python
def reenter(recorder: Any) -> None:
    _ACTIVE.set(recorder if isinstance(recorder, ExtractionDiagnostics) else None)
```

(plus its docstring). A recorder becomes current in this execution context; anything else (no
pipeline run in this session) clears the context, so `current()` builds a detached recorder exactly
as a fresh thread did. No snapshot/copy function: the session keeps the recorder **object** itself,
which already carries everything the readers need (`boundaries`, `artifact_dir`, `run_id`, Stage 0,
cleaning, iterations, its own lock). A copy was rejected because the readers append to it (§ 7 Q3).

### 2.2 The PROTECTED `src/t2pw/app/streamlit_app.py`: three lines, nothing else

Exact diff (`git diff 4f750030 81a5c556 -- src/t2pw/app/streamlit_app.py`):

```diff
@@ -96,6 +96,7 @@ from t2pw.pipeline.extraction_diagnostics import (
     count_processes as diagnostic_process_counts,
     current as current_extraction_diagnostics,
     payload_hash as diagnostic_payload_hash,
+    reenter as reenter_extraction_diagnostics,
 )
 from t2pw.pipeline.stage_one_boundary import settle_stage_one
 from t2pw.pipeline.gate_reports import (
@@ -5703,6 +5704,7 @@ if submit:
     st.session_state["pathway_context"] = pathway_context
     st.session_state["user_task_context"] = user_task_context
     st.session_state["stage_one"] = stage_one
+    st.session_state["extraction_diagnostics_recorder"] = current_extraction_diagnostics()
     st.session_state["out_of_scope_removed_reactions"] = out_of_scope_removed_reactions
     st.session_state["chunk_details"] = chunk_details
     st.session_state["stage_two"] = stage_two
@@ -6687,6 +6689,7 @@ if st.session_state.get("pipeline_ready"):
             # never the thing that has to remember to invalidate.
             reset_quarantine_state(PROJECT_ROOT / "outputs")
             post_artifacts = {}
+            reenter_extraction_diagnostics(st.session_state.get("extraction_diagnostics_recorder"))
             try:
                 with st.spinner("Running audit and DB mapping..."):
                     final_payload = propagate_context_organism(
```

1. **the import** — one line in the existing `extraction_diagnostics` import block;
2. **the store** — no import needed (`current_extraction_diagnostics` is already imported, `:97`);
3. **the re-entry** — before the handler's `try:`, so the audit rounds, the gap rounds, the Stage-6
   call, the except-branch `_record_mapped_failure_snapshot` and the `finally:` publish all see it.

No comment, no whitespace, no other byte. Made by `C:/t/bt-orch759b/edit_app.py` on raw bytes with
`\r\n` in every pattern. Verified: 7,281 lines, 7,281 CRLF, **0 bare LF**; a normalized line diff
against `4f750030` shows exactly the three `+` lines above and no `-` line.

**Fingerprints** (sha256 unless stated):

| | working tree (CRLF, what the app runs) | git blob content (LF) | git blob id |
|---|---|---|---|
| before (`4f750030`) | `28c091171f18a700dea952723f5451f012090640712c0821ac66d58134f892ae` | `52e016d7e823122a1762dd0e26c475950ce58331598c79aa5e15385aa765dd7c` | `12bf51735565a02f632734cdff475a97d44240a0` |
| after (`81a5c556`) | `1e0986a538216c23cd0ea79002e916a8189e8b36ce81c2d77d83a03268b04136` | `eef4d80452766e8ebb29936c17566a932650ebbbc95cce268f1de11eb6045621` | `ed5d55367d6adaac8c7b7c613a783d80254ee895` |

The "before" values equal `ORCH-757-RESULT.md` § 4's "after". No test hashes this file (searched
`tests/`, `evidence/*.py`).

### 2.3 Where the store sits, and why not literally "after Stage 1"

D-107 § 1 says "when Stage 1 finishes". The store is in the block where the `if submit:` run commits
Stage 1's results to session state (`stage_one`, `chunk_details`, ..., `final_payload`), i.e. the end of
the script run in which Stage 1 ran, not the line right after `settle_stage_one`. Reason: between those
two points the run can still `st.stop()` (scope conflict, empty pathway, Stage-2 failure, ...), and
`pipeline_ready` is never cleared on that path (`:5431` only clears it for the ambiguous-review case).
A store placed right after Stage 1 would then pair **this** run's recorder with the **previous** run's
`final_payload`, which the post-pipeline button still maps: rule 3 would be gated on the wrong run's
draws (merging a 1-draw payload is exactly the hazard `_pooled_draws`' docstring names), and the rounds
would be filed in the wrong run's directory. In the commit block the recorder and the payload are
written together or not at all. **This is a placement call the reviewer should check against D-107
§ 2's wording** (§ 7 Q1).

## 3. Session and run isolation

- **Across sessions.** The recorder lives in `st.session_state`, which is per browser session
  (`AppSession._session_state`, handed to each of that session's `ScriptRunner`s,
  `streamlit/runtime/app_session.py:543-548`, Streamlit 1.62.0). No code path reads another session's
  state, so no session can re-enter another's recorder.
- **Across runs in one session.** Every "Run pipeline" activates a fresh recorder (`activate`, unchanged)
  and, once it reaches the commit block, replaces the stored one. `test_g9_a_new_pipeline_run_replaces_
  the_stored_recorder` runs a 3-draw pipeline, then a 1-draw pipeline in the same session, then the
  post-pipeline with `T2PW_EXTRACTION_DRAWS=3`: rule 3 stays off (run 2 did not pool), the rounds land in
  run 2's directory and not run 1's.
- **No stored state.** `reenter(None)` clears the context, so the post-pipeline run is exactly today's:
  `test_still_no_stage1_run_in_the_session_means_todays_post_pipeline` (passes at base and tip).
- **Live UI threads.** A live Streamlit session runs each script run in `ScriptRunner._run_script_thread`
  on a `threading.Thread` it starts (`script_runner.py:341`); a rerun requested while one is running is
  served on that same thread (the `while request.type == ScriptRequestType.RERUN` loop, `:412`), otherwise the next interaction gets a new `ScriptRunner` and
  a new thread. Either way the thread belongs to one session. AppTest's `LocalScriptRunner` subclasses
  `ScriptRunner` (`testing/v1/local_script_runner.py:39`) and starts its thread the same way, which is why
  the AppTest proof covers the mechanism. A thread the same session reuses may still hold the recorder
  after the post-pipeline run; every recording path that matters either activates (`submit`) or
  re-enters (post-pipeline) first, and `reenter` overwrites, never reads, what the thread held.
  `test_new_reenter_crosses_threads_and_drops_anything_else` pins this at the helper. **The live UI was
  not driven** (no browser here); see § 7 Q5.

## 4. Where each file lands

`<root>` is the app's `PROJECT_ROOT` (the primary checkout in a live leg); `run-<hex>` is the directory
`_start_extraction_diagnostics` creates for that "Run pipeline" (`activate(..., unique=True)`), also in
session state as `extraction_diagnostics_dir`.

| file | written by | lands at | leg run dir (`runs_*/…/papers/<id>/<mode>/`)? | review pack (`orch756_review_pack.py`)? |
|---|---|---|---|---|
| `stage1_ensemble_report.json` / `.txt` | Stage 1 (unchanged) | `<root>/tmp/extraction_diagnostics/run-<hex>/` | no (not in the driver allowlist) | yes (`SIDE_FILES`, mtime window) |
| `ensemble_identity_merge_report.json` | rule 3 at Stage 6 — **new in the app** | same `run-<hex>/` | no | **yes** (`SIDE_FILES`) |
| rule-3 text block `DATABASE-IDENTITY MERGE …` | rule 3 — **new in the app** | appended to `run-<hex>/stage1_ensemble_report.txt` (its mtime moves into the post-pipeline window) | no | yes, via that `.txt` |
| `audit_iteration_summary.json`, `gap_iteration_summary.json` | each audit/gap round — **new on disk** | same `run-<hex>/`, rewritten every round | **yes**, now with the run's `run_id` (at base: `run_id ""`, § 8 F1) | **no**: not in `SIDE_FILES` |
| `stage0_attempts.json`, `extraction_boundary_report.json`, `cleaning_report.json` | Stage 0/1 (unchanged) | same `run-<hex>/` | **now yes** for legs that reach the post-pipeline (at base: lost, § 8 F1) | no |
| `mapped_failure_snapshot.json` | a post-pipeline failure | same `run-<hex>/` — **new on disk** | yes | no |

`SIDE_FILES` also lists `ensemble_identity_merge_report.txt`, which nothing writes (the text goes into
`stage1_ensemble_report.txt`); harmless, noted for whoever owns the pack (§ 7 Q4).

## 5. Proofs

All in `tests/test_orch759b_recorder_reattach.py` (an AppTest file: **run it alone**). The real app,
driven as `batch/driver.py` drives it (paste, "Run pipeline", "Run audit and DB mapping"), offline:
`t2pw.llm.client._client` is a fake transport that records every request (Stage 0 gets a context, every
Stage-1 draw gets the same two reactions `protoheme formation` / `heme b formation`, everything else
`{}`); `map_ids._map_compound_with_strategy` answers from a table (`protoheme`, `heme b` -> CHEBI:17627);
no PathBank DB; `map_ids.HttpClient.get` and `requests.Session.request` raise; Stage 7 enrichment copies
its input; `t2pw.paths.PROJECT_ROOT`, `pipeline.TMP_DIR` and `render_pathwhiz_like.TMP_DIR` point into
`tmp_path`, so nothing in the checkout moves (`git status --porcelain` clean after every run). Inference
(Stage 2) is unchecked; the post-pipeline widgets keep their shipped defaults (LLM audit, gap resolver,
LLM gap resolver on).

### 5.1 G9: RED at the test-first commit, GREEN at the tip

| test | base | tip | what fails at base |
|---|---|---|---|
| `test_g9_rule3_runs_at_stage6_and_files_its_record_beside_stage1` | FAIL | pass | `final_mapped_db` keeps both rows: `['heme b formation', 'protoheme formation'] != ['protoheme formation']` (the premise holds: `stage1_ensemble_report.json` exists, `final_audited` has both rows) |
| `test_g9_audit_and_gap_rounds_reach_the_stage1_directory_and_the_driver` | FAIL | pass | `FileNotFoundError: …/run-<hex>/audit_iteration_summary.json` |
| `test_g9_a_new_pipeline_run_replaces_the_stored_recorder` | FAIL | pass | `(second / 'audit_iteration_summary.json').is_file()` is False |

- RED at `f9ea4a9a`: `05-g9-red` — **3 failed, 3 passed** (the 3 G9; preservation and LLM pass).
- RED of the final test file on base code (`a2a804dc`, src at `4f750030`): `07-g9-red-final` —
  **4 failed, 3 passed** (the 3 G9 at the same assertions, plus the labelled new acceptance test, which
  fails on the missing `reenter` and is not offered as proof).
- GREEN at `81a5c556`: `08-g9-green` — **7 passed**.

At the tip the rule-3 record is `[("protoheme formation", "heme b formation", "database-id")]`, written
once, in the Stage-1 `run-<hex>/` and nowhere else under `tmp/extraction_diagnostics`.

### 5.2 Preservation (pass at base and tip)

- `test_still_one_draw_never_runs_rule3`: Stage 1 at 1 draw, post-pipeline with `.env`-style `3`: both
  rows survive Stage 6, no identity report anywhere (D-104 § 2: code default 1 behaves as before).
- `test_still_no_stage1_run_in_the_session_means_todays_post_pipeline`: a seeded payload, no Stage-1 run:
  nothing merges, nothing under `tmp/extraction_diagnostics`, and the published diagnostics are exactly
  the detached recorder's audit/gap summaries with `run_id ""`, as at base.

### 5.3 LLM inputs unchanged (D-104 AMENDMENT 2)

`test_llm_inputs_are_the_bases_bytes_and_never_carry_the_merge_record` digests every request the fake
transport receives (sha256 of the request minus `max_tokens`/`timeout`, the ORCH-757 digest) across both
script runs and compares the list to `GOLDEN`, **measured on base** (`03-dev-base-golden2`; it then passed
on base in a different process and `tmp_path`, `04-dev-base-request-kinds`, `05-g9-red`, `07-g9-red-final`,
and at the tip, `08-g9-green`). The 15 requests (`04-dev-base-request-kinds.log`, `PROBE` lines):

| # | request | script run |
|---|---|---|
| 0 | Stage 0 preprocessor | Run pipeline |
| 1-3 | Stage 1 extraction, 3 draws | Run pipeline |
| 4-6, 9-11 | LLM audit candidates, 3 per round, 2 rounds | post-pipeline |
| 7, 12 | gap resolver planner, per round | post-pipeline |
| 8, 13 | gap enrichment agent (tool call), per round | post-pipeline |
| 14 | pathway curator (tool call) | post-pipeline |

Every one is byte-identical to base. Every post-pipeline model call in this path runs **before** the
Stage-6 `map_payload`, so none can see rule 3's output; on the "Run audit and DB mapping" path nothing
after Stage 6 calls a model (overwatch needs `build_legacy_sbml`, which this button passes as `False`;
the stoich agent is off by default). The same test asserts that none of `ensemble_identity_merge`,
`DATABASE-IDENTITY MERGE`, `bridged_by`, `database-id` occurs in any request, in the returned
`mapping_report` (which `compact_mapping_misses` feeds the interactive curator), or in the session's
`reaction_summary` (the curator's and enrichment agent's text). The record stays in the side files
because `map_payload` discards `merge_on_mapped_identity`'s return value (`map_ids.py:11121`, unchanged).

`tests/test_orch757_llm_token_budgets.py` (the per-stage request golden) passes unchanged
(`09-affected-core`). What rule 3 may change is the payload **after** Stage 6, its approved purpose
(D-104 § 3(c)); later model calls a human starts on that payload (interactive curator, completeness
audit) will see the merged rows, never the record.

### 5.4 New acceptance test (labelled)

`test_new_reenter_crosses_threads_and_drops_anything_else`: a recorder activated on the test thread is
absent on a new thread (the D4 mechanism itself), present after `reenter`, and `None`, `{}` or a string
clears it. Fails at base only because `reenter` does not exist.

## 6. Gates

Each run bounded (`bounded_run.py`, no `--heavy-lock`), `pinned_pytest.py` with `--basetemp`, **FINAL
SURVIVING COUNT 0 in every report**; `g11_evidence.py check --task ORCH-759b` exit 0, 19 ok.

| gate | G11 report | tree | result |
|---|---|---|---|
| G9 RED | `05-g9-red` | `f9ea4a9a` | 3 failed (G9), 3 passed |
| G9 RED, final test file | `07-g9-red-final` | `a2a804dc` + base src | 4 failed (3 G9 + new acceptance), 3 passed |
| G9 GREEN + preservation + LLM inputs | `08-g9-green` | `81a5c556` | **7 passed** |
| affected: 22 files (`test_extraction_diagnostics`, `test_adversarial_diagnostics_isolation`, `test_adversarial_json_repair_guard`, `test_adversarial_row_repair_deletion`, `test_attempt_cap_termination_reason`, `test_c110_negative_control_status`, `test_early_failure_replay`, `test_extraction_escalation_ladder`, `test_localized_repair`, `test_orch758b_empty_chunk`, `test_stage_one_boundary`, `test_stage_one_boundary_lineage`, `test_stage_one_ensemble`, `test_stage_one_paper_dictionary`, `test_streamlit_stage2_orchestration`, `test_orch757_llm_token_budgets`, `test_batch_driver_quarantine_artifacts`, `test_batch_driver_seam_golden`, `test_batch_pwml_artifact_naming`, `test_batch_report`, `test_batch_run`, `test_batch_fetch`) | `09-affected-core` | tip | **637 passed** |
| `tests/test_batch_driver.py` (AppTest, alone) | `10-batch-driver` | tip | **58 passed** |
| `tests/test_batch_research_gate_fail_open.py` (AppTest, alone) | `11-batch-research-gate` | tip | **14 passed** |
| SMOKE (22 files, `TEST_MATRIX.md` § Commands) | `12-smoke` | tip | **508 passed** |
| `tests/test_eval_v2_*.py` (6 files) | `13-eval-v2` | tip | **89 passed, 1 xfailed** |
| alone: `test_batch_preflight.py` | `14-alone-batch-preflight` | tip | 32 passed, **1 failed**, 4 skipped — `test_the_message_names_the_problem_the_interpreter_and_the_cure` ("this project ships a .venv"): worktree-only, listed in `TEST_MATRIX.md` (§ ORCH-757 known failures, `:1044`, `:1095`; red on base `ORCH-757/22`, `ORCH-758b/10`) |
| alone: `test_c055_rag_loop_wiring.py` | `15-alone-c055` | tip | **14 passed** |
| alone: `test_streamlit_quarantine_boundary.py` | `16-alone-quarantine-boundary` | tip | **23 passed** |
| alone: `test_c052_prefreeze_report_at_the_streamlit_seams.py` | `17-alone-c052` | tip | **13 passed** |
| alone: `test_reaction_evidence_view.py` | `18-alone-reaction-evidence-view` | tip | **11 passed** |
| alone: `test_reaction_evidence_app_hook.py` | `19-alone-reaction-evidence-hook` | tip | **1 passed** |
| Chunk D | — | — | not run: the orchestrator's (charter) |

Development runs, kept because they are part of the record: `01-dev-base-probe` (first draft on base,
5 failed 1 passed: the expected G9 failures plus two wrong expectations, § 8 F1/F3), `02-dev-base-golden`
and `03-dev-base-golden2` (digest capture), `04-dev-base-request-kinds` (temporary `PROBE` print, then
the file was restored from a byte copy), `06-g9-green-dev` (tip code uncommitted: 1 failed, § 8 F2).

## 7. Not settled: for the product owner / orchestrator

1. **Store placement (§ 2.3).** The store is in the commit block of the run that ran Stage 1, not on the
   line after `settle_stage_one`. Options: (a) keep it (recorder and payload always paired; one line);
   (b) move it right after Stage 1 as D-107 § 1 literally reads, accepting that a later `st.stop()` in the
   same run pairs the new recorder with the old payload (rule 3 on the wrong run's draws); (c) both a
   store after Stage 1 and a clear at run start (more protected lines). Recommendation: (a).
2. **The legacy SBML export button** (`:7148`) also calls `run_post_pipeline_sbml_artifacts`, so also
   reaches the Stage-6 call, and does **not** re-enter: rule 3 and the rounds stay off there, exactly as
   at base. Fixing it is one more protected line before that call; not authorized by D-107 § 2 as I read
   it ("re-enter before the Stage-6 call" in the post-pipeline run). The batch never presses it.
3. **Repeated "Run audit and DB mapping" clicks in one session** re-enter the same recorder, so a second
   click's rounds are appended to the first's (`round_count` grows; round numbers restart at 1), and a
   second rule-3 record is appended to the identity report (its writer already appends). Nothing is lost;
   the alternative (a copy per click) would overwrite click 1's rounds on disk. Say if a per-click record
   is wanted.
4. **Review pack.** `orch756_review_pack.py` `SIDE_FILES` will copy the identity report and the
   `.txt`, but not `audit_iteration_summary.json` / `gap_iteration_summary.json`; they are in the leg's
   run directory (driver) and in `run-<hex>/`. Adding them to `SIDE_FILES` is the orchestrator's call; it
   also lists an `ensemble_identity_merge_report.txt` nothing writes.
5. **Live UI** was not driven in a browser. The mechanism is the same `ScriptRunner` thread start AppTest
   uses (§ 3); the 3-draw app leg the orchestrator runs is the live proof.
6. **ORCH-758 D4's corroboration needs one correction** (§ 8 F1): leg run directories did hold
   audit/gap summaries at base (186 of them), from the detached recorder, with `run_id ""`, and none of
   those 186 holds Stage 0/1's `stage0_attempts.json` / `extraction_boundary_report.json`. The defect is
   real; its symptom in the leg directories was different from "no summaries".

## 8. Findings, failures and dead ends

- **F1 — the post-pipeline publish replaced Stage 0/1's diagnostics at base.** I believed (from ORCH-758
  D4: "0 of 312 runs hold their summaries") that the session hand-off after the post-pipeline was empty.
  `01-dev-base-probe` disproved it: at base `extraction_diagnostics` after the post-pipeline holds
  `audit_iteration_summary.json` and `gap_iteration_summary.json` from the detached recorder (`run_id ""`),
  and nothing from Stage 0/1. Read-only count over the primary checkout's `runs*/` (2026-10-22): 186 leg
  directories hold `audit_iteration_summary.json`, **all 186 with `"run_id": ""` and 0 of them with
  `stage0_attempts.json`**; the 83 directories that hold `stage0_attempts.json` are legs that ended before
  the post-pipeline. So `t2pw.batch.driver` has been copying the detached recorder's rounds and dropping
  Stage 0/1's evidence for every leg that reached audit. At the tip the hand-off carries both, under the
  run's id (`test_g9_audit_and_gap_rounds_…`). The preservation test's expectation was corrected to the
  measured base behaviour before the test-first commit.
- **F2 — an over-strict "no network attempted" assertion.** The first draft asserted that the offline
  guard was never hit. At the tip it was (`06-g9-green-dev`): 52 refused requests to ChEBI's web service
  (`ebi.ac.uk/webservices/chebi/2.0/test/getLiteEntity`) from lookups the table stub does not cover. They
  are refused at base too (base never reached that line because rule 3 failed first). The assertion was
  removed in `a2a804dc` (test-only, base code), and the final test file re-run RED on base (`07`) to keep
  the G9 claim on the file that ships.
- **F3 — `GOLDEN` capture.** `-vv` did not untruncate pytest's diff through `pinned_pytest -q`; the
  measured list is now the assertion message. A first parse script failed on the `E` prefixes; the
  second extracted the 64-hex digests in order. Determinism: the golden from `03` passed in four later
  processes with different `tmp_path`s.
- **F4 — process slips (no effect, recorded because they broke a hard rule).** Twice I issued a Bash
  command containing an empty `python - <<'EOF'` heredoc (no code in it); the harness left `python -`
  waiting on stdin and backgrounded the shell. Both were stopped with `TaskStop` (`b6gyydcbr`,
  `bau7kueew`); no Python code ran, nothing was written, and `git status` was clean afterwards. No
  `taskkill`/`pkill` was used.
- **Dead end not taken:** putting the re-entry inside `run_post_pipeline_sbml_artifacts` (it would also
  cover the legacy button, § 7 Q2) — rejected: `tests/test_streamlit_stage2_orchestration.py` executes that
  function from source in a hand-built namespace, so a new global there would break that harness, and it
  would tie a function that tests call directly to Streamlit session state.

## 9. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

Add under the per-branch focused selections / AppTest "each alone" list:

```
ORCH-759b (D-107): tests/test_orch759b_recorder_reattach.py — AppTest, RUN ALONE (one process),
7 tests, ~16 s: 3 G9 (red on base 4f750030), 2 preservation, 1 LLM-input golden (15 request
digests measured on 4f750030), 1 labelled new acceptance (reenter helper). Expect 7 passed.
Known failure unchanged: test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_
and_the_cure in a worktree with no .venv.
```

SMOKE stays **508** (the new file is not in it); Chunk D's partition is untouched (its file lists are
fixed and do not include the new file; importing `real_streamlit` / `_app_exceptions` from
`test_streamlit_quarantine_boundary.py` does not change that file's collected count).

## 10. Files changed

| file | change |
|---|---|
| `src/t2pw/app/streamlit_app.py` (PROTECTED) | +3 lines (§ 2.2) |
| `src/t2pw/pipeline/extraction_diagnostics.py` | +19: `reenter` |
| `tests/test_orch759b_recorder_reattach.py` | new, 7 tests |
| `docs/pwml_recovery_sprint/ORCH-759b-RESULT.md` | this file |
| `docs/pwml_recovery_sprint/evidence/g11/ORCH-759b/`, `…/g11/pin/ORCH-759b/`, `…/evidence/orch759b/` | 19 G11 reports, 19 pin verdicts, 19 logs |

Not touched: `map_ids.py` (parallel lanes), any prompt, any gate, the merge rules, `TEST_MATRIX.md`,
`LEDGER.md`, `DECISIONS.md`, caches, `out/`, `outputs/`, `tmp/`, `runs_*/`.
