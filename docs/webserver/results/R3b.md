# R3b - the submit block's Stage 0 moves to `service.extraction.run_stage_zero`

- **Branch:** `ws/R3b-stage-zero` from `webserver/refactor` @ `220b1621` (R3a merged).
- **Code commits:**
  - `b7830ed2`: the move and the new test file.
  - `1df7b6b2`: the new tests also pin the post-Stage-0 sync (section 6, item 3).
  - `623ab02b`: test path repoint under PO ruling Q-R0-1 (section 4).
  - `d6b903b1`: review round 1, F1 and F2 (section 8).
- **Docs commits:** `b747cf17` (needs-decision version), `bdd8aaac`, and the round-1 update.
- **Status:** done, pending independent review. The first report was needs-decision, because 4 source-reading tests in `tests/test_preprocessor.py` failed. The orchestrator authorised a path-only repoint of those tests under PO ruling Q-R0-1, made in `623ab02b` (section 4). After it, every chunk's failing-node set equals base.
- **Behaviour change:** none intended. Two invisible differences are listed in section 3, items 4 and 5.
- **NEW TESTS (R3b, new module):** `tests/test_service_extraction_stage0.py`, 19 tests (20 after review round 1).
- **Scope:** R3b only, from the scout's split (`results/R3-scout.md` section 7). Stage 1 onwards is untouched.
- **Deviation from the brief:** the ambiguous-review early exit **stays in the app**. `tests/test_stage0_case_c_retry_clobber.py` lifts that exact `if` block out of `streamlit_app.py` by AST and execs it. See section 3, item 1, and Open question 1.

## 1. What moved where

APP = `src/t2pw/app/streamlit_app.py`, EXT = `src/t2pw/service/extraction.py`.

| what | base `220b1621` APP | tip EXT (`1df7b6b2`) |
|---|---|---|
| `_PREPROCESS_RETRY_CHARS`, `_has_usable_context`, `_seed_context_from_user_scope`, `_stage0_context_rank`, `_run_stage_zero_with_retry` | 217-395 | 164-347 |
| the `EXTRACTION_DIAGNOSTICS_KEY` doc comment (it sat above `logger`) | 756-760 | 350-354 |
| `EXTRACTION_DIAGNOSTICS_KEY`, `EXTRACTION_DIAGNOSTICS_DIR_KEY`, `DIAGNOSTICS_DIRNAME`, `_start_extraction_diagnostics`, `_sync_extraction_diagnostics` | 786-846 | 355-415 |
| submit block: the diagnostics start, the `Running preprocessor...` spinner, Stage 0 with its retry, the seeded / failed / recovered messages, and the post-Stage-0 `_sync_extraction_diagnostics()` | 2105-2166 | `run_stage_zero` body, 443-505 |
| new: `StageZeroResult(pathway_context, diagnostics_dir)` and the `run_stage_zero(...)` signature | - | 418-441, 507 |

**Stays in the app:**

- `reset_refinement_state()` (APP 1886). It is the first statement, directly before the call, so the order is unchanged. It is the refinement panel's UI state; `REFINEMENT_STATE_DEFAULTS` is read by `test_c055:722` on the app module.
- The ambiguous-review gate and its `st.stop()` (APP 1902-1926). It is byte-identical to base 2168-2192.
- `_last_cleaning_report` (Stage 1, R3c).

**The app call site** (APP 1886-1900):

```python
reset_refinement_state()
_stage_zero = run_stage_zero(text, temperature=temperature, user_task_context=user_task_context,
                             state=st.session_state, reporter=StreamlitReporter(st),
                             preprocess_fn=preprocess)
pathway_context = _stage_zero.pathway_context
_diagnostics_dir = _stage_zero.diagnostics_dir
```

`_diagnostics_dir` is still read by the Stage-1 boundary message (APP ~1981).

**Names kept reachable on the app module (APP 191-203, 229-240, 627-629):**

- Re-imported from EXT: `_PREPROCESS_RETRY_CHARS`, `_has_usable_context`, `_seed_context_from_user_scope`, `_stage0_context_rank`, `EXTRACTION_DIAGNOSTICS_KEY`, `EXTRACTION_DIAGNOSTICS_DIR_KEY`, `DIAGNOSTICS_DIRNAME`.
- **Shim** `_run_stage_zero_with_retry(text, *, temperature, user_task_context)`: the old signature. It calls the EXT function with `preprocess_fn=preprocess`, where `preprocess` is the app module's own global, read at call time.
- **Shim** `_sync_extraction_diagnostics()`: the old no-arg signature. It calls the EXT function with `state=st.session_state`. The app's four other call sites (Stage 1 pass/fail paths, post-pipeline) still use it unchanged.
- `_start_extraction_diagnostics` is **not** kept on the app. Its only caller moved, and no test or module references it (grep over `tests/` and `src/`).
- The app keeps its now-unused imports `describe_preprocess_status`, `preprocess_was_recovered`, `activate_extraction_diagnostics` and `diagnostic_payload_hash`, so those names still resolve on the module.

**The service API:**

- `run_stage_zero(text, *, temperature, user_task_context, state, reporter, preprocess_fn=None)`.
  - `state` and `reporter` are required and keyword-only, with no default (R2 F1). Nothing here calls `default_reporter`, `default_state` or `install_app_defaults`.
  - It never stops: no `st.stop()` was inside the moved range (the only one is the gate's, which stayed).
- `_start_extraction_diagnostics(text, *, state)` and `_sync_extraction_diagnostics(*, state)`: `state` is required and keyword-only.
- `_run_stage_zero_with_retry(..., preprocess_fn=None)`: `None` means `t2pw.pipeline.preprocessor.preprocess`, looked up **at call time**. The app always passes its own binding.
- `PROJECT_ROOT` is read as `t2pw.paths.PROJECT_ROOT` at call time. EXT holds no `PROJECT_ROOT`/`TMP_DIR` attribute, so the harness's `isolate` reaches it through `paths` (a test pins this).
- The recorder is still the single `t2pw.pipeline.extraction_diagnostics` context-var recorder: `activate(..., unique=True)` and `current()`. No second instance exists.
- Imports: `t2pw.paths` (module), `t2pw.pipeline.preprocessor` (the module, plus three names nobody patches) and `t2pw.pipeline.extraction_diagnostics` (the module only, as `_diag`, since review round 1; section 8). EXT imports neither streamlit nor `t2pw.app`; R3a's subprocess test `test_importing_the_module_loads_neither_streamlit_nor_the_app` passes.

**Verbatim proof:** `evidence/R3b/verbatim-diff.txt` diffs the three base ranges against EXT (CR stripped, block dedented). The only changed lines are:

- the `preprocess_fn` parameter, its docstring note, and the one line that resolves it;
- `state` replaces `st.session_state` in the two diagnostics helpers, whose signatures gained `*, state`;
- `paths.PROJECT_ROOT` replaces `PROJECT_ROOT` (twice);
- `reporter.spinner` / `reporter.info` / `reporter.warning` replace `st.spinner` / `st.info` / `st.warning`;
- the Stage-0 call passes `preprocess_fn=preprocess_fn`, and the two helper calls pass `state=state`.

No message text, order or condition changed. `reporter.py` is unchanged: `info`, `warning` and `spinner` already exist.

## 2. Test hazards checked (scout section 6 and my own grep)

| hazard | what keeps it working | proof |
|---|---|---|
| `test_stage0_case_c_retry_clobber.py` calls `app._run_stage_zero_with_retry` 8 times, reads `app._PREPROCESS_RETRY_CHARS`, `app._has_usable_context` and `app._stage0_context_rank`, and **patches `app.preprocess`** | the shim passes the app's `preprocess` global at call time, so the patch lands. The three other names are re-imported | file unedited, 8 passed (solo). Mutation `app-shim-ignores-app-preprocess` (the shim passes a stub instead) fails 7 of its 8 tests |
| the same file **lifts the ambiguous-review `if` block out of APP by AST** (`_ambiguous_review_gate`: exactly one `If` whose test is `is_ambiguous_multi_example_review_context(pathway_context)` and whose source contains `ambiguous_review_scope`), then execs it with only `st`, `pathway_context` and that predicate in scope | the gate stays in APP, byte-identical | the same run |
| **`test_preprocessor.py`: four tests read the app SOURCE** for the two `preprocess(` calls, the transient-warning guard, `describe_preprocess_status(pathway_context)` and `preprocess_was_recovered(pathway_context)` | they failed at `1df7b6b2`. I found them only in the 2e run. **Repointed to `service/extraction.py` in `623ab02b` (Q-R0-1)** | section 4; 2e: 44 passed at `623ab02b` |
| `test_stage0_scope_fallback.py` does `from t2pw.app.streamlit_app import _has_usable_context, _seed_context_from_user_scope` | re-imported | unedited, passes solo |
| `test_c055:722` reads `app_module.REFINEMENT_STATE_DEFAULTS` | not moved | 2b-c055 |
| `test_orch759b` patches `paths.PROJECT_ROOT` (and `postpipeline.PROJECT_ROOT`) before AppTest, then reads `extraction_diagnostics_dir` from session state | EXT reads `paths.PROJECT_ROOT` at call time, so the diagnostics land under the patched root as before | 2e-759b |
| equivalence harness `isolate` patches `paths.PROJECT_ROOT`/`TMP_DIR` and every `t2pw*` module that has those attributes | EXT has neither attribute and reads through `paths` | `eq` 14 passed. Unit test `test_the_service_has_no_project_root_copy`. Mutation `diagnostics-root-not-paths` is killed |
| AppTest re-execs the script each run, so `from t2pw.pipeline.preprocessor import preprocess` rebinds every run | the app passes that binding as `preprocess_fn`, so the source is the same as base | new real-app test patches `preprocessor.preprocess` and sees exactly one call |
| other `monkeypatch.setattr` on a moved name | **CORRECTED in review round 1.** The original claim ("no test patches the diagnostics helpers") was **wrong**: R3a's `tests/test_service_extraction.py:316` patches `extraction_diagnostics.payload_hash` on its source module, and the import-time binding in `b7830ed2` silently disarmed that tripwire. I grepped for the helpers' own names and missed the callees' source-module names. The full per-name check is in section 8 | section 8 (F1) |
| `install_app_defaults` | the new code never resolves a default. The app passes `StreamlitReporter(st)` and `st.session_state` explicitly | unit test `test_state_and_reporter_are_required_keywords`. Mutation `state-and-reporter-defaulted` is killed |
| `user_task_context` value and order | unchanged: APP 1868 strips it before `assemble_inputs`, and it is passed straight through | ivm scenarios; unit tests assert the kwargs of every `preprocess` call |

## 3. Things the reviewer must look at

1. **The ambiguous-review gate was not moved.** The brief asked for it to move, with its `session_state` writes and its stop. Moving it breaks `test_stage0_case_c_retry_clobber.py::test_case_c_then_empty_draw_still_ends_in_ambiguous_review_scope` unless that test is edited. The test finds the gate in APP by AST and runs it in a namespace that holds only `st`, `pathway_context` and the predicate. So even a thin app gate that calls the service fails: the service name would be a `NameError` in that namespace.
   - The one workaround that keeps the test green is an in-gate `from t2pw.service.extraction import ...`, with `ambiguous_review_scope` left in a comment so the AST filter still matches. That games the test, so I did not do it.
   - Open question 1 has the options.
2. **The display inside the gate** (`st.error`, `st.subheader("Candidate examples")`, `st.json`, `st.warning`) therefore still goes to `st` directly, not through the Reporter. No `subheader` primitive was added to `reporter.py`.
3. **`preprocess_fn`** is a new optional parameter on `_run_stage_zero_with_retry` and `run_stage_zero`. It exists only to keep `streamlit_app.preprocess` the patch target the clobber test uses, and to keep the app calling exactly the `preprocess` binding it called before. A web worker can leave it as `None`.
4. **Invisible difference A: Stage 0 raising.** If `preprocess` raises, base had already bound the script global `_diagnostics_dir` before the exception; tip binds it only from the returned result.
   - Everything else in that path is identical: the `st` calls, the state writes, the recorder activation, the exception.
   - The exception ends the script run, and Streamlit re-executes the script in a fresh namespace on the next run, so nothing can read the global.
   - The ivm script reports this as `SAME-EXCEPT-UNBOUND-GLOBAL` in 2 scenarios and fails on any other difference.
   - The same holds for the dead script globals `seeded_context`, `_seed_org`, `first_context`, `retry_context`: nothing in the app reads them (grep).
5. **Invisible difference B: logger name.** `_sync_extraction_diagnostics` logs at DEBUG level only when `artifacts()` raises. Its logger is now `t2pw.service.extraction`, not the app module's `logging.getLogger(__name__)`. No test reads it.
6. **No `InputFailure` / kind** was added. The moved range contains no stop point.

## 4. Test edits (Q-R0-1)

**One edit: commit `623ab02b`, `tests/test_preprocessor.py`, 4 lines changed (`git show --numstat`: `4 4`).**

- **Authority:** PO ruling **Q-R0-1** (`docs/webserver/results/PO-RULINGS-2026-11-04.md`, line 9): "Pointing a source-reading test (or a patch target) at the new file counts as an import-path edit." The orchestrator authorised it for R3b. The reviewer must confirm that the edit changes only paths.
- **The change, in each of four tests:** `app_path = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"` becomes `app_path = ROOT / "src" / "t2pw" / "service" / "extraction.py"`. The tests are:

  | line | test |
  |---|---|
  | 149 | `test_streamlit_forwards_user_task_context_to_every_preprocess_call` |
  | 171 | `test_streamlit_suppresses_transient_warning_for_ambiguous_reviews` |
  | 344 | `test_streamlit_stage_zero_warning_reports_the_reason` |
  | 603 | `test_streamlit_surfaces_a_recovered_stage_zero_context` |

- Nothing else changed: no assertion, no test name and not the `ROOT` line. The proof copy in `evidence/R3b/repoint/` pins `ROOT` to the tree; that line is **not** in the commit. `test_streamlit_strips_diagnostics_before_the_completeness_audit_prompt` still reads `streamlit_app.py`, because that code did not move.
- **Before/after:**

  | chunk 2e (`test_preprocessor.py`) | result |
  |---|---|
  | base `220b1621` | 44 passed |
  | tip `1df7b6b2` (before the edit) | 40 passed, 4 failed (the four above) |
  | tip `623ab02b` (after the edit) | 44 passed |

  Survivors were 0 on all three runs.
- **How it was missed:** `test_preprocessor.py` **was in my pre-move grep hit list** (it matched `describe_preprocess_status`), but I did not open it. I took it for a unit-test file of the preprocessor module. Every other hit was opened and checked (section 2).
- No other existing test was edited: `git diff 220b1621..HEAD --stat -- tests` shows this file and the new `test_service_extraction_stage0.py`.

## 5. Tests: tip vs base

All runs used `bounded_run.py` in the foreground, with the venv python for both the wrapper and the child.

- Environment: `PYTHONPATH=<tree>/src`, `pinned_pytest --expect-tree <tree>` with no `--`, `T2PW_OFFLINE_CURATOR=1`.
- Each run had a fresh `--basetemp=C:/t/wsR3btmp/<pfx>-<chunk>` whose parent was created first.
- `--heavy-lock ws-R3b` was passed on every chunk except `web`. Every heavy run was gated (section 6, item 7).
- Helpers: `evidence/R3b/scripts/run.sh` and `chunks.sh` (the chunk lists: R2 2a-2g with the AppTest files split out, as in R3a), and `compare.sh EVDIR [prefix]`. The full table is in `compare.txt`.

**Trees and survivors:**

- **Tip `t-`:** `C:/t/ws-R3b` @ `1df7b6b2`, the code as reviewed. Every chunk ran.
- **Tip `t2-`:** the same tree @ `623ab02b`, after the test repoint. 2e, the golden files, eq, the new file and the solo Stage-0 files were re-run.
  - `623ab02b` differs from `1df7b6b2` only in `tests/test_preprocessor.py`, so the other `t-` results stand for it.
- **Base `b-`:** detached `C:/t/ws-R3b-base` @ `220b1621`. **Every chunk ran.**
- Every report shows `worktree dirty=False`.
- **Survivors were 0 on every run:** all 113 `final_surviving_count` fields (91 run reports plus 22 mutation reports) are 0.

| chunk | tip | base @ 220b1621 | failing-node set |
|---|---|---|---|
| g1 batch_driver_seam_golden | 2 passed (t, t2) | 2 passed | identical (none) |
| g2 c011 | 8 passed (t, t2) | 8 passed | identical |
| g3 stage8_export_contract | 4 passed (t, t2) | 4 passed | identical |
| g4 quarantine_boundary | 24 passed (t, t2) | 24 passed | identical |
| **eq** equivalence, alone | **14 passed** (t, t2) | 14 passed | identical |
| 2a | 78 passed, 1 failed | same | identical (c052) |
| 2b (incl. c060a) | 134 passed | 134 passed | identical |
| 2b-c055 | 14 passed (t, t2) | 14 passed | identical |
| 2c | 101 passed, 3 skipped, 1 failed | same | identical (prefreeze_third) |
| 2c-c126 | 2 passed | 2 passed | identical |
| 2d (incl. clobber, scope_fallback, 759a) | 413 passed, 1 failed | same | identical (rag_seed) |
| 2d-stage3 | 28 passed | 28 passed | identical |
| 2d-c119g9 | 3 passed | 3 passed | identical |
| **2e test_preprocessor** | t: 40 passed, 4 failed. **t2: 44 passed** | 44 passed | **t2: identical**. t: the 4 Q-R0-1 tests, section 4 |
| 2e-759b | 7 passed (t, t2) | 7 passed | identical |
| 2e-hook | 1 passed | 1 passed | identical |
| 2e-759afold | 2 passed (t, t2) | 2 passed | identical |
| 2f | 32 passed, 4 skipped, 1 failed | same | identical (batch_preflight) |
| 2f-driver | 61 passed | 61 passed | identical |
| 2f-gate | 14 passed | 14 passed | identical |
| 2g | 162 passed, 8 skipped | same | identical |
| 2g-c082pp / c056d / c119sup / revview | 1 / 14 / 37 / 11 passed | same | identical |
| 2g-759d | 4 passed (t, t2) | 4 passed | identical |
| at-c052 (alone) | 12 passed, 1 failed | same | identical (c052) |
| at-c082sr / at-c083 / at-bridge (alone) | 8 / 12 / 37 passed | same | identical |
| at-pf3 (alone) | 11 passed, 1 failed | same | identical |
| web (13 files) | 301 passed | 301 passed | identical |
| ext: test_service_extraction + test_service_reporter | 37 passed | 37 passed | identical |
| solo clobber / scope_fallback / c060a / 759a duplicate_protein_fold | 8 / 5 / 11 / 37 passed (clobber, c060a, 759a also t2) | same | identical |
| **new** test_service_extraction_stage0 | **19 passed** (t, t2) | n/a (file absent) | - |

- **Golden 38 passed on tip and base.** The golden files are unchanged.
- The four known pre-existing failures (c052, prefreeze_third, rag_seed, batch_preflight) have identical node ids on tip and base.
- **SMOKE was not run.** That is the independent runner's job.

### 5a. How the four `test_preprocessor.py` failures were found and proven path-only

| test | what it reads | why it failed at `1df7b6b2` |
|---|---|---|
| `..._forwards_user_task_context_to_every_preprocess_call` | AST: exactly 2 `preprocess(...)` Name-calls, each with `user_task_context=` | both calls moved with `_run_stage_zero_with_retry` |
| `..._suppresses_transient_warning_for_ambiguous_reviews` | AST: an `if` containing "transient LLM failure" whose test names `is_ambiguous_multi_example_review_context` and `_has_usable_context` | that `if` moved into `run_stage_zero` |
| `..._stage_zero_warning_reports_the_reason` | text `describe_preprocess_status(pathway_context)` and `Stage 0 failed:` | moved |
| `..._surfaces_a_recovered_stage_zero_context` | text `preprocess_was_recovered(pathway_context)` and `Stage 0 partially recovered` | moved |

- **The proof before the ruling:** `evidence/R3b/repoint/repointed_preprocessor_check.py` is an untracked-by-the-suite copy (no `test_` prefix), with `repoint-proof.diff` showing the change.
  - It has the same 4 path lines changed, plus `ROOT` pinned to the tree, because the copy lives outside `tests/`.
  - Run explicitly through the wrapper at `1df7b6b2`, it gave 44 passed (`t-repoint.*`).
- The committed edit (`623ab02b`) has only the 4 path lines.

## 6. What failed, dead ends, incidents

1. **I ran `python -` once (stdin interpreter), against the standing rule.** It was an inspection command (`python.exe -I - <<'EOF' ... EOF`, an empty heredoc), before any test ran. It hung, the harness moved it to the background at 120 s, and I stopped it with TaskStop. A `Get-CimInstance` check found no surviving process of it. Afterwards I used only script files.
2. **Off-by-one boundaries in my first edit script.** I misread `sed` output, so its asserted line ranges were wrong. The script's own asserts caught it before any write. The second run, with ranges from `grep -n`, wrote the files.
3. **Mutation round 1 was invalid, and round 2 found a weak test.**
   - First, the mutation runner's `--basetemp` parent (`C:/t/wsR3btmp/mut-tmp`) did not exist, so 15 of 19 tests errored on every mutation. That is the documented trap. I also captured only stdout, which drops the wrapper's survivor count. I fixed both and discarded the results.
   - Then, on the valid round, `post-stage0-sync-dropped` **survived**: the fake `preprocess` recorded nothing, so `{}` from the start looked the same as the sync's `{}`.
   - I strengthened the unit test and the real-app test: the fake now records a stage-0 attempt, and both tests assert `stage0_attempts.json` reached `state`. That is commit `1df7b6b2`. I tried to amend it into the first commit; the amend was denied, so it is a second code commit.
   - The final round killed 22 of 22. The superseded logs are in `evidence/R3b/dev-superseded/mutation-v1-tests/`.
4. A `sed` with `#` as delimiter broke on a `#` in its replacement and did nothing. I redid the edit with the Edit tool.
5. The first ivm run failed its own sanity assert: `"run_stage_zero" not in old` also matches `_run_stage_zero_with_retry`. I tightened it to `"= run_stage_zero("`.
6. The first tip golden/eq batch took more than 180 s, and the harness moved the foreground Bash call to the background. The wrapper was not detached: it was the same process tree, the call was awaited to completion, and survivors were 0. After that I kept each batch under 180 s.

7. **Heavy-run gate and waits** (`scripts/waitfree.py`, log in `evidence/R3b/waits.log`): 111 gate observations, 97 of them free.
   - **E2E-2 (orchestrator live job), about 33 minutes.** At 20:51 the gate before `t-2a` found PIDs 941404/940044: `bounded_run --timeout 7200 --label e2e-2-live`, in `c:/t/ws-e2e` / `c:/t/e2e2`. The batch stopped (exit 3). A foreground waiter on those PIDs logged `wait-start 20:53:47` and `wait-end 21:26:50`. I used the time for this doc and the compare script.
   - **W4b runs in `c:/t/ws-w4b`:** busy at the first look before `t2-new` (21:48), `b-2b` (21:53), `b-2f-driver` (21:57) and `b-solo-clobber` (21:59). Each cleared within one 55 s poll.
   - **`c:/t/ws-w4b-trial` (with `c:/t/bt`):**
     - before `b-g1`: busy 22:02-22:04, the gate gave up (exit 3); the retry was free at 22:04:18 after one more busy look;
     - before `b-eq`: busy 22:07-22:09, the gate gave up; the retry was free.
   - No run started while another worktree's pytest or bounded_run was alive. No run exited 95.

## 7. Open questions and carry-forward

0. **Resolved:** the four `test_preprocessor.py` source readers. The orchestrator authorised the repoint under PO ruling Q-R0-1, made in `623ab02b` (section 4).

**Carry-forward to R3c/R3d** (orchestrator decision: keep these as they are in R3b):

1. **The ambiguous-review gate stays in the app** (APP 1902-1926, byte-identical to base 2168-2192).
   - It writes `pipeline_ready`, `pathway_context` and `pipeline_error`, displays through `st` (`error` / `subheader` / `json` / `warning`) rather than the Reporter, and calls `st.stop()`.
   - `test_stage0_case_c_retry_clobber.py` lifts it from `streamlit_app.py` by AST and execs it with only `st`, `pathway_context` and the predicate in scope. So moving it needs more than a Q-R0-1 path repoint: the exec namespace must change too.
   - R3d/R5 must decide how a headless worker gets this refusal. The options:
     - a separate task with an approved test edit, adding a Reporter `subheader` and returning a failure of kind `ambiguous_review_scope`;
     - or a service helper that builds the `pipeline_error` dict.
2. **`reset_refinement_state()` stays in the app**, as the refinement panel's UI state (`REFINEMENT_STATE_DEFAULTS` is read on the app module by `test_c055:722`). The web worker has no refinement panel. R5 decides whether it needs an equivalent.
3. **`preprocess_fn`** stays as implemented, for the reviewer to judge.
   - It is an optional parameter on `_run_stage_zero_with_retry` and `run_stage_zero`. `None` means `t2pw.pipeline.preprocessor.preprocess`, looked up at call time.
   - The app passes its own `preprocess` binding, so the clobber test's patch of `streamlit_app.preprocess` still lands and the app calls exactly the binding it called before.
   - When the clobber test is next touched, it could patch `t2pw.service.extraction` instead (a Q-R0-1 patch-target edit), and the parameter could go.
4. **For R3c:** the app keeps `_sync_extraction_diagnostics()` as a no-arg shim over the moved function. Its 4 remaining call sites are in Stage 1 and post-pipeline. `_last_cleaning_report` is still in the app.

## 8. Review round 1: F1, F2

The review of `bdd8aaac` returned CHANGES_REQUESTED with one blocking finding (F1) and one note (F2). Both are fixed in code commit `d6b903b1`. No existing test was edited.

### F1 (blocking): diagnostics names bound at import disarmed R3a's tripwire

- **What was wrong:** `service/extraction.py` did `from t2pw.pipeline.extraction_diagnostics import activate, current, payload_hash` once, at import.
  - Before R3b, the app script ran that `from ... import` on every AppTest run, so `monkeypatch.setattr(diag, "payload_hash", ...)` reached the call.
  - After R3b it did not. R3a's `test_real_app_warns_and_stops_before_stage_zero` uses that patch as the tripwire behind the input-assembly `st.stop()`. With the patch unreachable, deleting the stop no longer failed the test deterministically.
- **The fix:** `from t2pw.pipeline import extraction_diagnostics as _diag`, then `_diag.payload_hash(...)`, `_diag.activate(...)` and `_diag.current()` at call time. This is the same pattern as `paths.PROJECT_ROOT` and `_preprocessor.preprocess`.
  - The three `from`-import aliases are gone from EXT. The app keeps its own imports of them.
  - My new tests and the ivm script now patch `diag.*` instead of the removed EXT aliases.
- **Every `from X import name` in EXT, checked against `tests/`:** I grepped for `setattr(` / `patch(` / `patch.object(` with the name as a string attribute, and for direct `X.name =` assignment. I also checked the dynamic `setattr(module, name, ...)` sites (`equivalence/harness.py` spies on postpipeline functions only; the others are unrelated).

  | module X | name | patched on X by a test? | action |
  |---|---|---|---|
  | `t2pw.pipeline.extraction_diagnostics` | `payload_hash` | **yes**, `test_service_extraction.py:316` | now `_diag.payload_hash` at call time |
  | same | `activate` | no | converted with it (reviewer's request; same module) |
  | same | `current` | no | converted with it |
  | `t2pw.pipeline.preprocessor` | `preprocess` | `app.preprocess` (clobber) and source-module patches in my tests | already call-time via `_preprocessor.preprocess`, or the app's `preprocess_fn` |
  | same | `describe_preprocess_status`, `is_ambiguous_multi_example_review_context`, `preprocess_was_recovered` | no | left bound |
  | `t2pw.extraction.pdf_parser` | `parse_pdf`, `SKIP_SECTIONS` | no. `parse_pdf` is patched **on EXT** (`test_service_extraction.py:80,212`), which needs the EXT-level binding | left bound; converting it would break those R3a tests |
  | `t2pw.service.reporter` | `Reporter` | no (a type) | left |

- **Probe, before/after** (`evidence/R3b/probe-f1.txt`, script `scripts/probe_f1.py`, run through the wrapper, survivors 0). It patches each name on `extraction_diagnostics` and calls `_start_extraction_diagnostics` and `_sync_extraction_diagnostics`:

  | name | `bdd8aaac` (before) | `d6b903b1` (after) |
  |---|---|---|
  | `diag.payload_hash` patch reaches service | **False** | **True** |
  | `diag.activate` patch reaches service | False | True |
  | `diag.current` patch reaches service | False | True |

- **New unit test** `test_patches_on_the_source_modules_reach_run_stage_zero`. It patches `diag.payload_hash`, `diag.activate`, `diag.current` and `preprocessor.preprocess`, then asserts all four are called in order (`payload_hash`, `activate`, `preprocess`, `current`), and that the recorder carries the patched run id.
- **Mutations** (`evidence/R3b/mutation-r1/`, survivors 0 on each):

  | mutation | what it does | result |
  |---|---|---|
  | `f1-bound-payload_hash` | reverts that one name to an import-time `from` binding | 1 failed |
  | `f1-bound-activate` | same, for `activate` | 3 failed |
  | `f1-bound-current` | same, for `current` | 2 failed |
  | `f1-app-input-stop-dropped` | removes the app's input-assembly `st.stop()` (`if _assembled.failure is not None:`), with **R3a's tripwire** as the test | **2 failed** (both parametrizations) with "the run went past the input-assembly stop" |

  The tripwire is armed again. Afterwards `git status -- src tests` was clean.

### F2 (note): the focus box reaches Stage 0 in the real app

`test_real_app_case_c_opens_diagnostics_then_stops_at_the_ambiguous_gate` now types `"  lipid A biosynthesis  "` into "Optional extraction focus / task context". It asserts that the one `preprocess` call received `user_task_context == "lipid A biosynthesis"` (stripped, as APP 1868 does).

### Round-1 runs on `d6b903b1`

All runs went through the bounded wrapper, with the heavy lock, gated, each chunk in its own process, `worktree dirty=False`, survivors 0. Reports are `evidence/R3b/r1-*`, and the comparison is in `compare.txt` under "review round 1".

| chunk | tip @ d6b903b1 | base @ 220b1621 |
|---|---|---|
| svc3: `test_service_extraction` + `test_service_extraction_stage0` + `test_service_reporter` | 57 passed (29 + 20 + 8) | n/a (the stage0 file is new) |
| tripwire: `test_service_extraction.py::test_real_app_warns_and_stops_before_stage_zero` | 2 passed. With the stop removed: 2 failed | - |
| g1 / g2 / g3 / g4, own processes | 2 / 8 / 4 / 24 (38) | same |
| eq, alone | 14 passed | 14 |
| 2e `test_preprocessor` | 44 passed | 44 |
| clobber / c060a / c055, own processes | 8 / 11 / 14 | same |
| inline-vs-moved on the committed tip | 24/24 (22 identical, 2 SAME-EXCEPT-UNBOUND-GLOBAL), 0 unexplained | - |

- The 2a-2g set was not repeated: only these lookups changed (the brief's condition).
- Gate: 126 observations in `waits.log`. The independent runner in `C:/t/ws-R3b-trial` and other worktrees (`ws-int`, `bt`) were seen busy before `d3-svc3` and `r1-svc3`, and each cleared within one poll. No run exited 95.
- Incident this round: once more, a `sed` with `#` as delimiter broke on a `#` in its replacement. Nothing was written; I redid the edit with the Edit tool.

## 9. Independent run, review and merge (added by the orchestrator)

**Review round 0:** ws-reviewer (opus/high) reviewed the diff `220b1621..bdd8aaac` and returned **CHANGES_REQUESTED**.

- **F1 (blocking).** The diagnostics names were bound at import, which disarmed R3a's tripwire patch. See § 8 for the fix.
- **F2 (note, done).** The `user_task_context` hand-off had no assertion. One was added.
- **F3-F7 (notes).** All accepted:
  - the unbound `_diagnostics_dir` on raise;
  - the logger name;
  - the `preprocess_fn` default, which is dependency injection and not a state fallback;
  - the reviewer's own inline-vs-moved re-run, 24/24;
  - import without streamlit.

  The Q-R0-1 edit to `test_preprocessor.py` was confirmed to be path-only (4/4 lines). The 4 tests still guard what they guarded at base.

**Review round 1:** the same reviewer reviewed the diff `bdd8aaac..e9f9820c` and returned **APPROVE WITH NOTES**.

- **F1 resolved.** The reviewer's own probe on `e9f9820c` showed that patches of `diag.payload_hash`, `activate` and `current` are each reached. With the stop removed, the tripwire fails again. The 4 new mutations were all killed.
- **F2 resolved.**
- **F3 (note).** The audit of bound names is complete. `parse_pdf` and `SKIP_SECTIONS` stay bound, because R3a's tests patch `parse_pdf` on the service module itself. The rest are never patched.
- **F4 (note).** The only test changes are the new file and the Q-R0-1 4-line repoint.
- **F5 (note).** `test_service_extraction_stage0.py:250-256` patches by direct assignment and restores in `finally`. This is tidiness only.
- **F7 (carried to R3d/R5).** The ambiguous-review gate and `reset_refinement_state` stay in the app, and the gate's display still goes to `st`.

**Independent run:** ws-test-runner (sonnet/medium) ran on the trial merge `0586a3fb` of `e9f9820c` onto `68470cfc`, in `C:/t/ws-R3b-trial`. All 17 runs exited 0 with 0 survivors and a gate wait of 0 s. Results:

| run | passed |
|---|---|
| golden | 2 + 8 + 4 + 24 = 38 |
| equivalence (alone) | 14 |
| `test_preprocessor` | 44 |
| service | 57 |
| web+bridge | 301 |
| clobber | 8 |
| c060a | 11 |
| c055 | 14 |
| 759d | 4 |
| 759b | 7 |
| 759a protein | 37 |
| 759a fold | 2 |
| scope_fallback | 5 |
| SMOKE | 511 |

The census count is 83. The reports are in `docs/webserver/evidence/R3b/independent/`. An earlier, unfinished run on the stale `bdd8aaac` is kept under `independent/old-bdd8aaac/`. It was interrupted by a machine restart.

**Merge:** `4ea3524a` (`--no-ff` into `webserver/refactor`). Its code tree is byte-identical to the trial merge that was tested. The post-merge result is recorded in the LEDGER.
