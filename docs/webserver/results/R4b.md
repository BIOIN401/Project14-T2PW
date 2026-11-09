# R4b - the "Run audit and DB mapping" handler moves to `service.post_run.run_post_pipeline`

- **Branch:** `ws/R4b-post-pipeline` from `72d309e9`.
- **Commits:** `08c011dd` (code and new tests), `7916bf2f` and `ffeeb6e0` (more new tests, test file only), `7170a9a7` (the authorised test edit, test only), then the docs commits.
- **Status: done, pending independent review.**
- **Behaviour change:** none intended. See section 3 for the two invisible differences.
- **NEW TESTS (R4b, new module):** `tests/test_service_post_run.py`, 42 tests, labelled in its docstring. `Reporter.exception` and `write` are additive.

## 1. What moved

APP = `src/t2pw/app/streamlit_app.py` at base, PR = `src/t2pw/service/post_run.py`.

| what | base APP | tip |
|---|---|---|
| `reenter`, the audit and mapping call, the verdict presentation, both handlers, the `finally` sync | 2892-3082 | PR `run_post_pipeline` |
| new: `PostRunRequest` (21 fields, no defaults, `db_password` excluded from `repr`), `PostRunResult(final_payload, failure)` | - | PR |

`verbatim-diff.txt` shows only `st.*` to `reporter.*`, session state to `state`, `request.x` for widget locals, source-module lookups, and the callback names. Every comment is unchanged. The message text is byte-identical, including both em dashes.

**Stays in the app:** the button, the not-a-dict guard and its message, the `.pop`, `reset_quarantine_state(PROJECT_ROOT / "outputs")`, `post_artifacts = {}` in that order, then the call and `final_payload = _post_run.final_payload`. Also `_render_stage_contract_failure`, `_render_issue_detail`, `initialize_refinement_review_state` and the post-block render. No `out_dir` parameter (Q-W0-1).

## 2. Patch sites and source readers

- **Call-time lookups** (all on the source module; each reverted to import time as a mutation, all killed): `postpipeline.run_post_pipeline_sbml_artifacts` (the harness spy), `postpipeline._record_mapped_failure_snapshot`, `pipeline.propagate_context_organism`, `extraction_diagnostics.reenter`, `extraction._sync_extraction_diagnostics`, `config.rag_config`, `export_mode.coerce_mode`, `export_mode.is_research`, `paths.PROJECT_ROOT`.
- No test patches any moved name on `streamlit_app`. `_safe_dict` and `_safe_list` are bound at import; nothing patches them.
- **Source readers of the moved range:** only `test_stage2_artifacts_are_visible_and_both_ui_paths_share_the_orchestrator` needed an edit. `test_streamlit_quarantine_boundary.py:1464-1476` still sees both reset calls in APP and is unedited. `:778` is unaffected because PR does not call `quarantine_and_close`.

## 3. For the reviewer

1. **`render_contract_failure` is a callback, and the exception is also returned.** The brief says the service returns the `StageContractError`, and also that the sync stays in `finally`, after the UI messages. A render after the return would put the sync before the messages on this path. The callback keeps base's order exactly, and `PostRunResult.failure` still carries the exception for headless callers. Open question 1 has the alternative.
2. **`open_review` and `render_issue_detail` are callbacks** for the same reason: both run inside the try. `open_review` runs before the "ready for review" message, and an exception in it lands in the generic handler.
3. **Invisible difference, dead globals.** Base bound about 13 script globals inside the handler (`artifacts`, `_pa`, `final_gate`...); the tip binds `_post_run`. A static check shows none is read after the handler.
4. **Invisible difference, the rebind on a raise.** If the contract renderer itself raises, base had already rebound `final_payload` and the tip has not. The exception ends the script run, so nothing can read that global. `inline_vs_moved.py` excludes this one field only when both arms raise the same exception.
5. **`reset_quarantine_state` before `reenter`** is the same order as base: the reset is the app's last line before the call, and `reenter` is the service's first.

## 4. Tests, tip vs base

Bounded wrapper, heavy lock R4b, fresh basetemp, survivors 0, `worktree dirty=False`. Tip chunks ran at `7170a9a7`; base at `72d309e9` (`C:/t/ws-R4-base`). `ffeeb6e0` changes only `tests/test_service_post_run.py`, so `new` and SMOKE were re-run on it and the other chunks stand. Full table: `evidence/R4b/proof/compare.txt`, 38 of 38 failing sets identical.

| run | tip | base |
|---|---|---|
| golden g1/g2/g3/g4, each alone | 2 / 8 / 4 / 24 = 38 | same |
| equivalence, alone | 14 | 14 |
| 2a / 2c / 2d / 2f | 78+1f (c052) / 101+1f+3s (prefreeze_third) / 413+1f (rag_seed) / 32+1f+4s (batch_preflight) | same, same nodes |
| 2b (includes the edited test) | 134 | 134 |
| 2b-c055, 2c-c126, 2d-stage3, 2d-c119g9, 2e x4, 2f-driver, 2f-gate, 2g x6 | identical | same |
| at-c052 / c082sr / c083 / pf3 / bridge | 12+1f / 8 / 12 / 11+1f / 37 | same |
| web+bridge | 301 | 301 |
| svc / stage1 / stage2 | 57 / 22 / 28 | same |
| c060a / s2orch, alone | 11 / 12 | same |
| **new** `test_service_post_run` | **42** at `ffeeb6e0` | n/a |
| **SMOKE** | **511** at `ffeeb6e0` | 511 |

## 5. Proof

- **Inline vs moved** (`scripts/inline_vs_moved.py`, `proof/ivm/`): **36/36, 0 unexplained, static check OK**, on the committed tip `7170a9a7`. It execs the real base and tip handler text against one recording `st`, and compares st calls and state writes in one combined order, callee arguments, the outcome, the final state and the globals. It covers every path in R4-scout section 3 plus blank widgets, RAG on and off, `reenter` raising, the review raising, the renderer raising and diagnostics raising.
- **Non-vacuity** (`scripts/ivm_sanity.py`, `proof/ivm-sanity/`): **17/17 detected**, on a scratch copy.
- **Mutations** (`scripts/mutate.py`, `proof/mutation/`, scratch copy only, tally in `tally.txt`): **97/98 killed**, including the 9 call-time reversions. The one survivor is `eq-mapping-cache` against the equivalence test: on this fixture the cache path does not change the output, and the unit tests kill the same fault (`cache-default`, and the real-app widget check).
- **F5 gap at this call site:** `test_real_app_passes_every_widget_value_through_to_the_pipeline` sets all 20 widgets to non-default values and asserts each reaches the pipeline. `app-swapped-*` mutations are killed.

## 6. The authorised test edit (Q-R4-1)

`tests/test_streamlit_stage2_orchestration.py`, that one function only, +43/-17, commit `7170a9a7`. It now reads `post_run.py` for the moved call, its try/except and the message, and adds a check that the button's branch reaches `run_post_pipeline` with a callback calling `_render_stage_contract_failure`. The Q-R3d-1 order test is untouched.

| deliberate break (scratch copy) | expected | got | first assertion |
|---|---|---|---|
| none, untouched tip | pass | 1 passed | |
| handler's call missing | fail | 1 failed | `assert 0 == 1` |
| wrong `build_legacy_sbml` | fail | 1 failed | `{True} == {False, True}` |
| `StageContractError` handling removed | fail | 1 failed | `'Exception' == 'StageContractError'` |
| generic handler removed | fail | 1 failed | `IndexError` |
| `.pop` removed | fail | 1 failed | `assert False` |
| failure message removed | fail | 1 failed | message not in source |
| app's render callback not wired | fail | 1 failed | `assert False` |
| button does not call the handler | fail | 1 failed | `StopIteration` |
| service does not call the render callback | fail | 1 failed | `assert False` |
| legacy button's `.pop` removed | fail | 1 failed | `assert False` |

**11/11 as expected** (`proof/vacuity/summary.txt`).

## 7. What failed, dead ends, incidents

1. **The first new tests missed coercions.** 7 mutations survived because `3307.0 == 3307` and `0 == False`, plus a non-dict context and the app's `.pop`. Fixed with strict type checks and an order check (`ffeeb6e0`); those 9 re-ran and were killed. Seeding stale artifacts through AppTest did not work: the page failed to reach the button, so I used an AST order check instead.
2. **A sanity-script bug.** `endswith("0 unexplained")` also matched "30 unexplained" and reported `app-swapped-widget` as MISSED. Fixed to parse the number and re-ran, 17/17. R3d's `ivm_sanity.py` has the same flaw; it can only produce false MISSED, never false DETECTED.
3. **`git commit --amend` was denied** by the permission system, on a compound command, so `7916bf2f` and `ffeeb6e0` are separate commits rather than folded into `08c011dd`. Code and new tests therefore span three commits, not the one the brief asked for. I did not retry the amend.
4. **Heavy-lock contention.** The R26 live recording held the lock for long stretches. Several runs waited and exited 95, and were re-run. No wrapper was killed.
5. **Usage limit.** The session stopped mid-run and resumed at `7170a9a7`; finished proofs were reused, not repeated.

## 8. Open questions

1. **Stage-contract render: callback or return-then-render?** A (shipped): callback plus returned failure; exact order; headless callers pass their own. B: return only, and the app renders then syncs; literal to the brief, but the service then skips its sync on this path, so every headless caller must remember it. Cost of switching: about 6 lines and one test.
2. **R4c and R5 must supply `open_review`.** `run_export` reads `refinement_working_json`, which only `initialize_refinement_review_state` writes, and that lives in the app. Options: move it to the service in R4c/R5 with attribute-to-item conversion (about 15 lines, AppTest re-proof), or give the headless path its own writer (a second copy, which drifts). I recommend the move.
3. **`PostRunRequest` has no source of defaults** for a headless caller, the same gap as audit F2. Carry to R5.
4. **Carried, not done here:** the audit's F1 quarantine clear for `run_extraction`, and R3d F2/F5, belong to R4c/R5.

## Evidence

`docs/webserver/evidence/R4b/`: `runs/` (`t-` tip, `b-` base, `f-` final tip; logs and JSON), `proof/`, `scripts/`.

## Batched test, review and merge (added by the orchestrator, 2026-11-08)

To conserve usage, W6b, W6c and R26 were tested as one trial merge (`98d7d641`), and R4a and R4b on top of it (`5a17423f`). Reviews and reports: `docs/webserver/evidence/batch-2026-11-08/`. Merged: W6b, W6c, R26 ending `f74ca936`; R4a, R4b ending `786f8a43`. Results on `5a17423f`: golden 38, equivalence 14, second fixture 8, service 57, stage1 22, stage2 28, export 30, post_run 42, c060a 11, s2orch 12, web 307, reasons 10, browser fixes 15, SMOKE 511, census 83, 0 survivors. No separate post-merge run: `786f8a43` has the same git tree hash (`e4422562`) as the tested trial, so it is byte-identical to what passed.
