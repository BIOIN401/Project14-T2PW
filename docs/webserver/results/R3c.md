# R3c - the submit block's Stage 1 and RAG seed move to `service.extraction.run_stage_one`

- **Branch:** `ws/R3c-stage-one` from `webserver/refactor` @ `4ea3524a` (R3a and R3b merged).
- **Code commit:** `f1238da7`, the move and the new test file.
- **Docs commit:** this file and `docs/webserver/evidence/R3c/`.
- **Status: needs-decision.** One existing source-reading test fails at the tip and cannot be fixed by a Q-R0-1 path-only repoint: `tests/test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference` (section 4, Open question 1). **No existing test was edited.** Every other chunk's failing-node set equals base.
- **Behaviour change:** none intended. One invisible difference (dead script globals on an early exit), listed in section 3, item 3.
- **NEW TESTS (R3c, new module):** `tests/test_service_extraction_stage1.py`, 22 tests.

## 1. What moved where

APP = `src/t2pw/app/streamlit_app.py`, EXT = `src/t2pw/service/extraction.py`.

| what | base `4ea3524a` APP | tip `f1238da7` EXT |
|---|---|---|
| `render_attempts` (now takes `reporter=`) | 566-577 | 521-532 |
| `_last_cleaning_report` | 632-648 | 535-553 |
| submit block: the `# Stage 1` comment, the "Running Stage 1 extraction..." spinner, `run_stage_one_with_chunking`, `settle_stage_one`, the `stage_one_boundary` write, the `PipelineFailure` / `StageContractError` handlers, the post-try diagnostics sync, the incomplete-boundary exit, the reconstruction message, `filter_out_of_scope_reactions` and its message, the RAG seed (`rag_config`, `build_draft_graph`, `generate_qa_report`, `maybe_run_rag`), the `rag_result` / `rag_evidence_context` writes and the `user_task_context` fold | 1928-2039 | `run_stage_one` body, 607-723 |
| new: `KIND_STAGE_ONE_FAILED` / `_BOUNDARY_ERROR` / `_INCOMPLETE`, `StageOneResult`, the `run_stage_one(...)` signature and its return | - | 516-518, 556-605, 725-732 |
| new Reporter primitives `markdown(body)` and `code(body, **kwargs)`: Protocol, `StreamlitReporter` (forwards unchanged), `LoggingReporter` (logs at INFO) | - | `service/reporter.py`, +17 lines, additive only |

**The app call site** (APP 1904-1929) replaces base 1928-2039:

```python
_stage_one = run_stage_one(text, pathway_context=pathway_context, user_task_context=user_task_context,
                           diagnostics_dir=_diagnostics_dir, enable_chunking=enable_chunking,
                           chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                           extract_attempts=extract_attempts, temperature=temperature,
                           extract_tokens=extract_tokens, rag_incomplete_flag=rag_incomplete_flag,
                           state=st.session_state, reporter=StreamlitReporter(st))
if _stage_one.failure is not None:
    st.stop()
stage_one = _stage_one.stage_one
chunk_details = _stage_one.chunk_details
stage_one_in_scope = _stage_one.stage_one_in_scope
out_of_scope_removed_reactions = _stage_one.out_of_scope_removed_reactions
rag_result = _stage_one.rag_result
user_task_context = _stage_one.user_task_context
```

These six names are everything the rest of the block reads from the moved range. I grepped APP from 2040 to the end for each moved local: `_boundary`, `_rag_seed_graph` and `failure` are not read after it (`failure` is rebound by Stage 2's own `except`).

**Stays in the app:** Stage 2 and both `merge_additions` sites (R3d; c060a lifts them, and c060a passes unchanged), the ambiguous-review gate (R3b), `reset_refinement_state()`.

**Names kept reachable on the app module:**

- `render_attempts(label, attempts)` stays, as a **shim** over the EXT function with `reporter=StreamlitReporter(st)` (APP 569-571). Stage 2 and the chunk listings (APP 1958, 2151, 2184) still call it. That is one function body instead of two copies; its output is proven identical (section 5).
- `_last_cleaning_report` is re-imported from EXT. Nothing in the app calls it now, and no test references it.
- The app keeps its now-unused imports (`run_stage_one_with_chunking`, `settle_stage_one`, `filter_out_of_scope_reactions`, `coerce_mode`, `build_draft_graph`, `generate_qa_report`, `rag_config`, `maybe_run_rag`, `CLEANING_REPORT_NAME`, `current_extraction_diagnostics`), so those names still resolve on the module. `app.maybe_run_rag` is read by `test_rag_triage_orchestration` (it is the same function object).

**The service API:**

- `run_stage_one(text, *, pathway_context, user_task_context, diagnostics_dir, enable_chunking, chunk_size, chunk_overlap, extract_attempts, temperature, extract_tokens, rag_incomplete_flag, state, reporter) -> StageOneResult`.
  - Every parameter after `text` is keyword-only with **no default**, `state` and `reporter` included (R2 F1). Nothing calls `default_reporter`, `default_state` or `install_app_defaults`.
  - The `int(...)` / `bool(...)` conversions stay inside, verbatim, so the app passes the raw widget values.
- **The three `st.stop()` exits** become returned failures at the same points, after the same messages: `InputFailure(kind, message)`, where `message` is the `st.error` text.

  | base exit | kind | message |
  |---|---|---|
  | `except PipelineFailure` | `stage_one_failed` | `Extraction failed: {failure}` |
  | `except StageContractError` | `stage_one_boundary_error` | `Extraction boundary failed: {failure}` |
  | `if not _boundary.ok` | `stage_one_incomplete` | `Extraction boundary failed and could not be recovered: {reason}` |

  `InputFailure` (R3a's dataclass) is reused rather than adding a second identical one. Open question 3 covers the name.
- **Display goes through the reporter:** `spinner`, `error`, `info`, `json`, and inside `render_attempts`, `expander` / `markdown` / `code` / `caption`. `maybe_run_rag` receives the same `reporter`. It has no `state` parameter (R2's signature), so none is passed; the app's `rag_result` / `rag_evidence_context` writes go into `state`.
- `export_mode` is read as `state.get("export_mode")` (base: `st.session_state.get(...)`), through `coerce_mode`, unchanged. `pwml_strict_db` is not read in this range.
- `PROJECT_ROOT` is `paths.PROJECT_ROOT`, read at call time. EXT still has no `PROJECT_ROOT` / `TMP_DIR` attribute (R3b's test `test_the_service_has_no_project_root_copy` passes).
- **Imports are function-local, as module objects** (EXT 599-605, plus `rag_orchestration` inside the RAG branch, and `postpipeline._safe_*` inside `_last_cleaning_report`). This has two effects:
  - every callee is looked up **at call time on its source module**, as the app script's per-run `from ... import` was (section 2);
  - importing EXT loads exactly what it loaded before R3c: 13 `t2pw` modules, measured before and after. Module-level imports would have added 24, including `postpipeline`, whose cached / not-cached state c055's fixtures branch on. The new test `test_importing_the_service_loads_no_stage_one_module` pins this. EXT still imports neither streamlit nor `t2pw.app` (R3a's subprocess test passes).

**Verbatim proof:** `evidence/R3c/verbatim-diff.txt` diffs the three base ranges against EXT (CR stripped, dedented). The only changed lines are:

- `st.*` replaced by `reporter.*`, and `st.session_state[...]` by `state[...]`;
- bare callee names replaced by `_module.name` (`_pipeline.`, `_stage_one_boundary.`, `_export_mode.`, `_config.`, `_draft_graph.`, `_qa_graph.`, `_rag_orchestration.`, `_diag.`);
- `PROJECT_ROOT` replaced by `paths.PROJECT_ROOT`, and `_diagnostics_dir` by `diagnostics_dir`;
- each `st.stop()` replaced by `return StageOneResult(failure=...)`, with the error text bound to `message` first;
- `_sync_extraction_diagnostics()` replaced by `_sync_extraction_diagnostics(state=state)`, and `render_attempts(...)` gains `reporter=reporter`;
- the two function-local import lines.

No message text, order or condition changed.

## 2. Patch sites (checked before moving anything)

I grepped `tests/` and `tests/equivalence/` for `setattr(` / `patch(` / `patch.object(` / `setitem(` with each name as a string, and for direct `X.name =` assignment. I did this for every name the moved code calls, on every module: the app, the service and the source modules. I opened every hit.

| name the moved code calls | source module | who patches or injects it | how the tip still sees it | proof |
|---|---|---|---|---|
| `run_stage_one_with_chunking` | `t2pw.pipeline.pipeline` | **c055:839** (`offline_pipeline`, on the source module, before AppTest) | `_pipeline.run_stage_one_with_chunking` at call time | c055 14 passed; mutation `bound-run_stage_one_with_chunking`: 20 failed |
| `settle_stage_one` | `t2pw.pipeline.stage_one_boundary` | **c055:853** | `_stage_one_boundary.settle_stage_one` at call time | mutation `bound-settle_stage_one`: 17 failed |
| `filter_out_of_scope_reactions` | `t2pw.pipeline.pipeline` | none (s2orch:971 is an AST reader, section 4) | call time | mutation: 4 failed |
| `build_draft_graph` | `t2pw.pipeline.draft_graph` | `test_interactive_curator:347,431` (source module, other caller); s2orch:421 injects it into a lifted postpipeline namespace | call time | mutation: 2 failed |
| `generate_qa_report` | `t2pw.pipeline.qa_graph` | `test_interactive_curator:349,433`; s2orch:422 (namespace) | call time | mutation: 8 failed |
| `maybe_run_rag` | `t2pw.service.rag_orchestration` | c055:479 (on rag_orchestration, for `run_rag_rounds`); `test_rag_triage_orchestration` calls `app.maybe_run_rag` and `setitem`s `rag_config` into **its** `__globals__` (rag_orchestration's, untouched) | `_rag_orchestration.maybe_run_rag` at call time | triage 14 passed; mutation `bound-maybe_run_rag`: 9 failed |
| `rag_config` | `t2pw.config` | nobody patches it on `t2pw.config`. `test_service_reporter:199` and triage:195/216/237 patch rag_orchestration's own binding | `_config.rag_config()` at call time | mutation `bound-rag_config`: 8 failed |
| `extraction_diagnostics.current` (in `_last_cleaning_report`; `_sync` was already call-time, R3b) | `t2pw.pipeline.extraction_diagnostics` | `test_service_extraction_stage0` (current); s2orch:444 (namespace) | `_diag.current()` at call time | mutation `bound-diag-current-cleaning`: 4 failed |
| `coerce_mode`, `CLEANING_REPORT_NAME` | `export_mode`, `extraction_diagnostics` | only namespace injections for lifted postpipeline functions (c011:68, s2orch:434) | module attribute at call time | no patch to preserve |
| `_safe_dict`, `_safe_list` | `t2pw.service.postpipeline` | namespace injections only (c011:70, s2orch:397, stage8:43); c060a:347 lifts `_safe_dict` by AST from APP + postpipeline + rag_orchestration | function-local import | c060a 11 passed |
| `PipelineFailure`, `StageContractError` | classes | none | bound | - |
| `PROJECT_ROOT` | `t2pw.paths` | equivalence `isolate` (harness:186-192), 759b:168 | `paths.PROJECT_ROOT` at call time | eq 14, 759b 7; mutation `artifact-dir-not-paths`: 1 failed |
| `render_attempts`, `_sync_extraction_diagnostics`, `StreamlitReporter` | app / EXT / reporter | none | the app passes `reporter`; EXT calls its own functions | ivm (section 5) |

**On the app module:** no test patches any of these names on `streamlit_app` (grep for `setattr(app...` with each name: no hit). AppTest re-executes the script, so a patch on the imported app module never reached the submit block, before or after.

**c055 alone does not guard the bound-import mistake.** Run alone, c055 imports EXT for the first time inside AppTest, *after* its fixtures have patched the source modules. So even a module-level `from ... import` binds the patched function, and c055 stays green: `evidence/R3c/mutation/c055-alone/`, 14 passed for both bound mutations.

- I tried pairing c055 with a file that imports EXT first. Unmutated, that pair already gives `1 failed, 30 passed, 12 errors`, on **base** and on tip (`mutation/c055-combo/`). That is a pre-existing cross-file isolation fault, not mine, so it is not a usable control.
- The guard is therefore the new unit test `test_patches_on_the_source_modules_reach_run_stage_one`, which patches after EXT is imported, plus every other unit test, since they all patch the source modules. Each bound-import mutation is killed (section 5).

## 3. Things the reviewer must look at

1. **The one failing source reader** (section 4). This is the needs-decision.
2. **`render_attempts` became a shim in the app.** The brief moved the Stage-1 call. Keeping the app's own body as a second copy would have duplicated it, so the app function now forwards to EXT.
   - The Stage-2 and chunk listings therefore render through `StreamlitReporter`, which forwards `expander(label, expanded=False)`, `markdown(body)`, `code(raw, language="json")` and `caption(body)` unchanged.
   - ivm compares the base function against the tip shim directly (3 cases, identical). The real-app test asserts the Stage-2 expander and its markdown. Mutation `app-shim-silent` is killed.
3. **Invisible difference: dead script globals on an early exit.** When the run ends early, base had already bound some script globals before the exit; the tip binds them only from a returned result.
   - It applies to `st.stop()` after the boundary settled or failed, and to `maybe_run_rag` raising. The globals are `stage_one` and `chunk_details`, plus `stage_one_in_scope`, `out_of_scope_removed_reactions` and `rag_result` when the RAG seed raised.
   - The script run is over either way, and Streamlit re-executes the script in a fresh namespace, so nothing can read them. This is the same class as R3b's `_diagnostics_dir` case.
   - ivm reports it as `SAME-EXCEPT-UNBOUND-GLOBALS` in 6 scenarios. It allows the difference only when the outcome is the same early exit on both arms, every other field is identical, and each differing value is unbound on the tip.
   - Also dead: base's `_boundary` and `_rag_seed_graph`, and the tip's new `_stage_one`. None of them is read anywhere else in the app (grep).
4. **The reporter instance passed to `maybe_run_rag`.** Base built a new `StreamlitReporter(st)` for that call; the tip passes the one the app built for `run_stage_one`. `StreamlitReporter` is a stateless forwarder over the same `st`, so this makes no difference. ivm records the reporter's type and `_st is st` for both arms.
5. **Function-local imports** (section 1). Open question 4.

## 4. The failing source reader (needs-decision; not edited)

`tests/test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference` (lines 958-996, in chunk 2b and solo-s2orch). It parses **APP only** and asserts:

1. exactly one `ast.Name` call each of `run_stage_one_with_chunking`, `filter_out_of_scope_reactions` and `run_stage_two_with_feedback_loop`;
2. line order: extraction < filter < inference;
3. Stage 2's `args[1]` is a Name other than `stage_one`;
4. the first `merge_additions` call's base is a Name other than `stage_one`.

At the tip, assertion 1 fails: `assert len(extraction_calls) == 1` gives `assert 0 == 1` (`evidence/R3c/t-solo-s2orch.log`). The base passes, 12/12.

**Why a Q-R0-1 path-only repoint cannot fix it:**

- Stage 2 and `merge_additions` stay in APP (R3d), and the extraction and filter are now in EXT. No single file holds all four calls, so repointing to EXT fails at `run_stage_two_with_feedback_loop`, and repointing to APP fails as now.
- In EXT the two calls are **attribute** calls (`_pipeline.run_stage_one_with_chunking(...)`), which the call-time lookup rule (section 2) requires. The test matches only `ast.Name`.

**Why I did not keep the code in the app:**

- The only way to leave Name calls of `run_stage_one_with_chunking` and `filter_out_of_scope_reactions` at the right APP lines is to keep the whole try / spinner / handlers / filter in the app. Then Stage 1 is not moved, which is the task.
- The alternative is thunks (`extract=lambda: run_stage_one_with_chunking(...)`). That games the test, the same workaround R3b refused for the ambiguous gate.

**What the test protects is still pinned, behaviourally:**

- unit test `test_out_of_scope_reactions_are_removed_and_reported` checks the filter runs after the boundary, on the settled payload;
- `test_rag_on_seeds_from_the_in_scope_payload...` checks the RAG seed gets the filtered payload, by identity;
- the real-app test `..._hands_stage_two_the_filtered_payload_and_the_folded_focus` checks Stage 2 receives the filtered payload;
- mutations `seed-from-unfiltered` and `app-in-scope-is-raw` are both killed.

Options are in Open question 1.

## 5. Proof

**Inline vs moved** (`scripts/inline_vs_moved.py`; run on the committed tip `f1238da7` through `bounded_run` with the heavy lock; `inline-vs-moved.log` / `.json`, `worktree dirty=False`, survivors 0): **30/30, 0 unexplained.**

- The script cuts the real base range (APP from `# Stage 1: strict extraction with auto-repair` to `final_payload = stage_one_in_scope`) and the real tip call site.
- It lifts base `render_attempts` / `_last_cleaning_report` / `_sync_extraction_diagnostics` by AST, and execs both arms at module level against the same recording fake `st`.
- The fakes reach the base through its namespace (its `from ... import` names) and the tip through the source modules.
- Per scenario it compares the ordered `st.*` calls (with spinner and expander enter/exit), the stop point or the raise, every `session_state` write and its value, every callee call with its arguments (the reporter recorded by type and target), and the six values read afterwards, `rag_result` identity included.

| scenarios | result |
|---|---|
| success; chunked (float widget values, `int()`); research mode; unknown mode; cleaning report last pass; diagnostics `current()` raising; reconstruction info (with and without `shell_count`); `PipelineFailure` (attempts with error/phase/note, and none); `StageContractError` from Stage 1; other exception from Stage 1; out-of-scope removal (alone, and with reconstruction); RAG off; RAG on: skip (None), no evidence, fold, fold with no focus, fold with an empty focus, error-degraded result | 24 IDENTICAL |
| `StageContractError` / `PipelineFailure` raised in `settle`; incomplete boundary with and without a failure report; incomplete boundary with RAG on; `maybe_run_rag` raising | 6 SAME-EXCEPT-UNBOUND-GLOBALS (section 3, item 3) |
| `render_attempts`: base function vs tip shim (Stage 2 attempts, an empty list, an empty error with a note) | 3 IDENTICAL (counted in the 30) |

**Non-vacuity** (`scripts/ivm_sanity.py`, `ivm-sanity/`). ivm detects each of 6 single mutations of the tip: `fold-order-swapped`, `spinner-text`, `contract-json-dropped`, `seed-from-unfiltered`, `app-focus-not-rebound` and `app-no-stop-on-failure`. Each run went through the wrapper, survivors 0.

**Mutations** (`scripts/mutate.py`, `mutation/`, **43/43 killed** by `tests/test_service_extraction_stage1.py`, survivors 0 on every run, `git status -- src tests` clean afterwards).

- **8 patch-target conversions reverted to a module-level bound import:** `run_stage_one_with_chunking`, `settle_stage_one`, `filter_out_of_scope_reactions`, `build_draft_graph`, `generate_qa_report`, `maybe_run_rag`, `rag_config`, `diag.current`. Each one is killed.
- **30 behaviour mutations of the moved code:** the spinner text; the artifact dir; `int()`; mode from state; the cleaning report dropped or using the first pass; the settled payload; the boundary write; each sync; attempts not rendered; failure kind; contract json; contract not a failure; the incomplete json, fall-through and message; reconstruction and out-of-scope info; seed from the unfiltered payload; `bool(flag)`; fold order; both RAG state writes; the unfolded return; render_attempts' `language` and caption; defaulted state and reporter.
- **5 app call-site / shim mutations:** no stop; focus not rebound; in-scope bound to the raw payload; `state={}`; silent shim.
- **2 reporter mutations:** `markdown` forwarded as `text`; `code` dropping its kwargs.

**Unit tests** (`tests/test_service_extraction_stage1.py`, 22, with a recording fake reporter).

- 20 offline tests: every branch above, the call-time patch reach, the import graph, the keyword-only signature, and the two new Reporter primitives on both reporters.
- 2 real-app AppTest tests:
  - a Stage-1 `PipelineFailure` gives exactly the one error and the "Stage 1 attempts" expander, and Stage 2 never runs;
  - with RAG on and a Stage-2 failure as the stop, Stage 2 receives the filtered payload, the chunk details and `"menaquinone\n\nPMC1: ..."` (the stripped focus plus the folded evidence), and session state carries `rag_evidence_context` and `stage_one_boundary`.

## 6. Tests: tip vs base

All runs went through `bounded_run.py` in the foreground, with the venv python for both the wrapper and the child.

- Environment: `PYTHONPATH=<tree>/src`, `pinned_pytest --expect-tree <tree>` with no `--`, `T2PW_OFFLINE_CURATOR=1`.
- Each run had a fresh `--basetemp=C:/t/wsR3ctmp/<pfx>-<chunk>`; the parent was created first.
- `--heavy-lock ws-R3c` on every chunk except `web`, and every heavy run was gated (section 7, item 6).
- Helpers: `evidence/R3c/scripts/run.sh`, `chunks.sh` (R2 2a-2g with the AppTest files split out, as R3b; plus `svc`, `new` and the `solo-*` RAG / s2orch files) and `compare.sh`. The full table is in `compare.txt`.

**Trees:** tip `t-` is `C:/t/ws-R3c` @ `f1238da7`; base `b-` is the detached `C:/t/ws-R3c-base` @ `4ea3524a`.

- All 41 tip and 41 base reports show `worktree dirty=False`.
- Survivors were 0 on every run.

| chunk | tip @ f1238da7 | base @ 4ea3524a | failing-node set |
|---|---|---|---|
| g1 / g2 / g3 / g4 (own processes) | 2 / 8 / 4 / 24 = **38** | same | identical (none) |
| **eq** equivalence, alone | **14 passed** | 14 | identical |
| 2a | 78 passed, 1 failed | same | identical (c052) |
| **2b** (incl. c060a, s2orch) | **133 passed, 1 failed** | 134 passed | **DIFFERENT: the s2orch reader, section 4** |
| 2b-c055 | 14 | 14 | identical |
| 2c | 101 passed, 3 skipped, 1 failed | same | identical (prefreeze_third) |
| 2c-c126 | 2 | 2 | identical |
| 2d (incl. clobber, rag_seed, triage, admission, 759a) | 413 passed, 1 failed | same | identical (rag_seed) |
| 2d-stage3 / 2d-c119g9 | 28 / 3 | same | identical |
| 2e test_preprocessor | 44 | 44 | identical |
| 2e-759b / 2e-hook / 2e-759afold | 7 / 1 / 2 | same | identical |
| 2f | 32 passed, 4 skipped, 1 failed | same | identical (batch_preflight) |
| 2f-driver / 2f-gate | 61 / 14 | same | identical |
| 2g | 162 passed, 8 skipped | same | identical |
| 2g-c082pp / c056d / c119sup / 759d / revview | 1 / 14 / 37 / 4 / 11 | same | identical |
| at-c052 (alone) | 12 passed, 1 failed | same | identical (c052) |
| at-c082sr / at-c083 / at-bridge | 8 / 12 / 37 | same | identical |
| at-pf3 (alone) | 11 passed, 1 failed | same | identical |
| web (13 files) | 301 | 301 | identical |
| svc: test_service_extraction + _stage0 + test_service_reporter | 57 | 57 | identical |
| **new** test_service_extraction_stage1 | **22 passed** | n/a (file absent) | - |
| solo c060a / clobber / scope_fallback / 759a | 11 / 8 / 5 / 37 | same | identical |
| solo test_rag_triage_orchestration / test_rag_admission_production_path | 14 / 9 | same | identical |
| solo test_rag_seed_entity_reimport | 13 passed, 1 failed | same | identical (the known rag_seed failure) |
| **solo test_streamlit_stage2_orchestration** | **11 passed, 1 failed** | 12 passed | **DIFFERENT: section 4** |

- Golden 38 passed on tip and base, and the golden files are unchanged.
- The four known failures (c052, prefreeze_third, rag_seed, batch_preflight) have identical node ids on tip and base.
- SMOKE was not run; that is the independent runner's job.

## 7. What failed, dead ends, incidents

1. **The source reader** (section 4). Found by a discovery run on the uncommitted tree (`C:/t/wsR3ctmp/dev`, not committed) and confirmed at `f1238da7`. I made no test edit.
2. **Module-level imports, then function-local.** The first draft imported the Stage-1 modules at EXT top level. A load check showed `import t2pw.service.extraction` went from 13 to 37 `t2pw` modules. I moved the imports into the function so the import graph is unchanged (section 1). The RAG module is imported only on the RAG branch, as base never touched it with RAG off.
3. **c055 as a bound-import guard** (section 2): it does not catch the mistake alone, and the pair with `test_service_extraction` is broken on base too. Those runs are kept under `mutation/c055-alone/` and `mutation/c055-combo/`, out of the 43.
4. **Tooling slips, none of which wrote a wrong file:**
   - A bash heredoc containing apostrophes failed to parse, and nothing was appended. I wrote the block to a scratch file and appended that.
   - Two `sed` edits did not match because of escaping, and nothing changed; I redid them with the Edit tool.
   - A python heredoc dropped a backslash, and its own assert stopped it before writing.
   - One mutation string was not unique. The script's assert stopped it with the file restored; I fixed the string and re-ran that batch.
5. **Near-miss on the `python -` rule.** One composed command contained `python.exe - 2>/dev/null </dev/null`, a stdin interpreter. **It did not execute:** it sat after a failing `mv` in an `&&` chain. No process was started. Every Python run in this task used a script file or `-c`.
6. **Heavy-run gate** (`scripts/waitfree.py`, `evidence/R3c/waits.log`): 131 observations, all free. No other worktree's pytest or bounded_run was seen alive before any heavy run, and no run exited 95.
   - The first tip batch (12 chunks) ran past 180 s, and the harness moved the foreground call to the background. It was the same process tree, awaited to completion, with survivors 0. Later batches were kept under 180 s.

## 8. Open questions

1. **`test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference` (blocking).** Options:
   - **A (recommended): a test edit beyond Q-R0-1.** Parse EXT's `run_stage_one` and APP's submit block.
     - In EXT: exactly one call (Name or Attribute) each of `run_stage_one_with_chunking` and `filter_out_of_scope_reactions`, in that order, and the RAG seed and the return use the filtered name.
     - In APP: exactly one `run_stage_one(` call before the one `run_stage_two_with_feedback_loop` call; Stage 2's `args[1]` is still not `stage_one`; the `merge_additions` base is still not `stage_one`.
     - Cost: about 15 lines in one test, and a PO ruling, because Q-R0-1 covers path-only edits. It keeps the test's intent, and R3d will need the same file touched again when Stage 2 moves.
   - **B: keep Stage 1 extraction and the filter in the app,** and move only the RAG seed. Cost: R3c shrinks to about 20 lines, and the headless worker still cannot run Stage 1. The same decision comes back in R3d.
   - **C: thunks from the app** (`extract=lambda: run_stage_one_with_chunking(...)`). It passes unedited but games the test, and adds two parameters only a test needs. Not recommended.
   - **D: retire the test,** relying on the new behavioural tests and the killed mutations `seed-from-unfiltered` and `app-in-scope-is-raw`. Cost: loses the structural pin; needs a PO ruling.
2. **`render_attempts` shim** (carry-forward to R3d). When Stage 2 and the chunk listings move, the app shim can go.
3. **`InputFailure` reused for Stage-1 failures.** It is the right shape, but the name says "input". Options:
   - leave it;
   - add `StageFailure = InputFailure` as an alias (one line);
   - rename it in R5, which touches R3a's API and tests.
4. **Function-local imports in `run_stage_one` and `_last_cleaning_report`.** They keep the EXT import graph unchanged and make every lookup call-time. If the reviewer prefers module-level imports, the cost is 24 extra modules on import, and a change in when `postpipeline` is first loaded, which c055's fixtures branch on.
5. **Observation, not caused here:** `tests/test_service_extraction.py` followed by `tests/test_c055_rag_loop_wiring.py` in one process gives `1 failed, 30 passed, 12 errors` on base `4ea3524a` and on the tip. Each file passes alone. A test-isolation fault, worth a ticket if SMOKE ever co-locates them.
