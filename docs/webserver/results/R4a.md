# R4a - the "Generate PWML" step moves to `service.export.run_export`

- **Branch:** `ws/R4a-export-step` from `webserver/refactor` @ `72d309e9`.
- **Code commits:** `2be3333c` (the move and the new tests), `bc712202` (one added test case, test only).
- **Status:** done, pending independent review.
- **Behaviour change:** none.
- **New tests (R4a, new module):** `tests/test_service_export.py`, 30 tests. `run_export` is a new public interface over moved code, and the module docstring labels it as such.

## 1. What moved

APP = `src/t2pw/app/streamlit_app.py`. EXP = `src/t2pw/service/export.py`.

| what | base APP | tip EXP |
|---|---|---|
| `quarantine_session_state` (now `*, state`) | 504-510 | 24-30 |
| `_pwml_reference_path` | 975-980 | 33-38 |
| `_write_reviewed_payload_snapshot` (gains `*, state`) | 983-1000 | 41-60 |
| `_generate_pwml_from_refinement_working_json`, now `run_export(*, state, reporter, work_dir, project_root)` | 1003-1061 | 63-133 |
| `st.spinner("Building PWML from reviewed pathway JSON...")` around the call | 1335 | `reporter.spinner`, wraps the `run_export` body |

**Stays in the app:** the button, the try/except and its failure dict, the `pwml_export_result` write, and the success and error text. The text is byte-identical.

- `state`, `reporter`, `work_dir` and `project_root` are all required and keyword-only. There is no `out_dir` (Q-W0-1).
- Attribute writes became item writes: `state["refinement_pwml_ready"] = True`, still before the export.
- The spinner moved because it is the only thing `reporter` is for. The brief lists what the app keeps, and the spinner is not on that list. If it had stayed, `reporter` would be an unused parameter.
- `project_root` is the app's `PROJECT_ROOT`, now read before the early returns instead of after them. It is a plain name read, so nothing observable changes.

**Call-time lookups:** `export.py` holds module objects and reads `_postpipeline.run_pwml_export`, `_safe_dict`, `_safe_list` and `_export_mode.coerce_mode` on each call. The app script's per-run `from ... import` did the same thing. Each one has a mutation back to an import-time binding, and all four are killed.

**Names kept on the app module:**

- `_pwml_reference_path` is re-imported, because `orch729_generate_pwml.py`, `orch768_replay_corpus.py` and `orch787_replay.py` under `docs/pwml_recovery_sprint/evidence/` read it through the app.
- `run_pwml_export` stays imported, because tests such as c052 and prefreeze_third import it from the app.
- `quarantine_session_state`, `_write_reviewed_payload_snapshot` and `_generate_pwml_from_refinement_working_json` are not kept. No test or script names them, and their signatures changed (R3b and R3d precedent).

**Not touched:** the dead locals at APP 2834-2839 and the "Run audit" handler (R4b).

## 2. Patch sites and source readers

| reader or patcher | what it depends on | result |
|---|---|---|
| `tests/equivalence/harness.py:128-143` spy on `postpipeline.run_pwml_export` | call-time lookup | reached; equivalence 14 passed |
| `test_streamlit_quarantine_boundary.py:778` scans the app and `service/*.py` | `run_export` must not call `quarantine_and_close` | passes, callers unchanged |
| `test_service_reporter.py:41` imports every service module, no streamlit | `export.py` imports none | passes |
| `test_batch_driver_seam_golden.py:294` message strings | app text unchanged | passes |
| No test reads the moved functions from the app | no test edit needed | **zero existing tests edited** |

Stale mentions left alone, because they are outside the file boundary: `postpipeline.py:110` (docstring), `test_streamlit_quarantine_boundary.py:7` (docstring), `harness.py:46,357` (comments).

## 3. Tests, tip against base

Tip is the code at `bc712202` (run at `18a9a15d`, which adds only proof scripts). Base is `72d309e9`, in `C:/t/ws-R4-base`. Every run went through `bounded_run` + `pinned_pytest` with the heavy lock, and every run had **0 survivors**. Full table: `docs/webserver/evidence/R4a/compare-tip-vs-base.txt`. Reports: `docs/webserver/evidence/R4a/runs/`.

| selection | tip | base | failing sets |
|---|---|---|---|
| golden g1 / g2 / g3 / g4, each alone | 2 / 8 / 4 / 24 passed (38) | same | equal |
| equivalence, alone | 14 passed | 14 passed | equal |
| new `test_service_export.py` | 30 passed | n/a | n/a |
| 2a | 1 failed, 78 passed | same | equal (c052) |
| 2b, 2b-c055 | 134, 14 passed | same | equal |
| 2c, 2c-c126 | 1 failed, 101 passed, 3 skipped; 2 passed | same | equal (prefreeze_third) |
| 2d, stage3, c119g9 | 1 failed, 413 passed; 28; 3 passed | same | equal (rag_seed) |
| 2e, 759b, hook, 759afold | 44, 7, 1, 2 passed | same | equal |
| 2f, driver, gate | 1 failed, 32 passed, 4 skipped; 61; 14 passed | same | equal (batch_preflight) |
| 2g and its five splits | 162 passed, 8 skipped; 1, 14, 37, 4, 11 passed | same | equal |
| AppTest splits c052, c082sr, c083, pf3, bridge | 1F+12, 8, 12, 1F+11, 37 | same | equal (c052, prefreeze_third) |
| web + bridge | 301 passed | 301 passed | equal |
| svc / stage1 / stage2 | 57 / 22 / 28 passed | same | equal |
| c060a / s2orch, each alone | 11 / 12 passed | same | equal |
| service-tree scans (5 files) | 1 failed, 88 passed | same | equal (dependency_declarations) |
| **SMOKE** | **511 passed** | **511 passed** | equal |

39 chunks compared, 39 identical, 0 different. The scans chunk was added here because those five test files walk `src/t2pw/` and the task adds a module.

**Base reports reused.** R4b ran the same base tree and commit. 38 of the 39 base logs are R4b's (`C:/t/wsR4btmp/ev2`), copied after checking each one is complete, ran in `C:/t/ws-R4-base`, and names `72d309e9`. Only base `scans` was run here.

## 4. Proofs

**Inline against moved** (`scripts/inline_vs_moved.py`, `proof/`): the base app's helpers and button block, cut out by AST, against the tip's button block plus `run_export` plus `StreamlitReporter`. It compares one ordered stream of st calls, session writes and the export call with all its arguments, then the final session and the files on disk.

- **30/30 scenarios the same**, including the failure dict when the work folder cannot be created.
- **Non-vacuity 27/27 sabotages detected.** These are in-memory edits of the tip's source.

**Mutations** (`scripts/mutate.py`, `mutations/`), on a scratch `git archive` copy: **51/51 killed**, control green. That includes the four call-time lookups reverted to import-time bindings and the app's call site.

## 5. What failed, and dead ends

1. **The first proof's 27/27 was vacuous.** My path normaliser missed the temp-folder name inside an exception message, so `work-dir-is-a-file` differed under every sabotage and made each one look detected. With the normaliser fixed, the true count was 26/27. The miss was `grounding-default`. Adding the scenario `grounding-absent-with-dict` made it 27/27.
2. **The first mutation pass was 50/51.** The same `grounding-default` survived. `bc712202` adds the absent-key case, and the rerun on the final tests is 51/51.
3. **Code and tests are two commits, not the one the brief asked for.** I tried to fold the test fix into `2be3333c` with `--amend`, in one compound command that also had an `rm -rf` of a stale proof folder. The permission system denied it. I did not retry either action, and committed the fix separately. The stale folder's only file (`proof/ivm.log`) was overwritten by the final run.
4. **The mutation batch ran inside one `bounded_run --heavy-lock R4a`**, with each pytest as a `pinned_pytest` child and its own `--basetemp`. That is not one wrapper per pytest. With several lanes queueing on the lock, 52 separate acquires could each have waited out another lane's run. The wrapper report shows 0 survivors. Evidence: `mutations/mutations2.log`.
5. **The lock queue.** R26's recording segments held the lock for about 10 minutes each. The harness also moved one foreground call to the background at 180 s. After that I kept each call under that limit and retried on exit 95. The lock directory was never touched.
6. **Usage limit.** The run stopped partway through the tip chunks and resumed. No finished run was repeated.

## 6. Open questions

1. **`test_dependency_declarations` fails on base and tip** with `assert not ['starlette']`. It is not on the known-failures list. It predates this task and is not caused by it. Options: declare `starlette` in the dependencies (a one-line change, outside this boundary); or list it as a known failure (no code cost, but the test stays red).
2. **Stale wording** in `postpipeline.py:110` and the harness comments still names the old function. Options: fix it in R4c, which already edits the harness (near zero cost); or leave it (a misleading name in a docstring).

## Batched test, review and merge (added by the orchestrator, 2026-11-08)

To conserve usage, W6b, W6c and R26 were tested as one trial merge (`98d7d641`), and R4a and R4b on top of it (`5a17423f`). Reviews and reports: `docs/webserver/evidence/batch-2026-11-08/`. Merged: W6b, W6c, R26 ending `f74ca936`; R4a, R4b ending `786f8a43`. Results on `5a17423f`: golden 38, equivalence 14, second fixture 8, service 57, stage1 22, stage2 28, export 30, post_run 42, c060a 11, s2orch 12, web 307, reasons 10, browser fixes 15, SMOKE 511, census 83, 0 survivors. No separate post-merge run: `786f8a43` has the same git tree hash (`e4422562`) as the tested trial, so it is byte-identical to what passed.
