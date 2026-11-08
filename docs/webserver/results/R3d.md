# R3d - Stage 2 through the session-state publish moves to `service.extraction.run_stage_two`; `run_extraction` added

- **Branch:** `ws/R3d-stage-two` from `webserver/refactor` @ `5af91e07` (R3a, R3b and R3c merged).
- **Code commit:** `590563ed`. It holds the move, `run_extraction`, the equivalence harness's service arm, and the new test file.
- **Test commit:** `3ade39e3`, the three authorised test edits, test only (section 9).
- **Docs commits:** `bdd75f05` (proof scripts and the two proposed test edits, as out-of-suite copies), `5f3fa309` (the needs-decision version of this file and the run evidence), and the post-ruling update.
- **Status: done, pending independent review.** The rulings are in section 9. After `3ade39e3`, every chunk that was re-run equals base. Sections 4 and 8 keep the needs-decision record as first reported.
- **First report, needs-decision.** The code was done, and every proof was complete on the committed tip. Two existing source-reading tests failed at the tip, and both needed a test edit:
  - `tests/test_c060a_requested_pathway_wiring.py`: 7 of 11 fail. The fix is a path and line-locator repoint, which needs orchestrator authorisation under Q-R0-1 (Open question 1).
  - `tests/test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_stage1_extraction_and_stage2_inference`: the Q-R3c-1 test. It needs a rewrite beyond Q-R0-1, so it needs a PO ruling (Open question 2).

  I edited neither test. Both proposed edits exist as proof copies outside the suite, and they pass at the tip; their non-vacuity checks came out 11/11 as expected (section 4). The headless ambiguous-review gate is also left as an open question, because building the app's `pipeline_error` dict in the service would duplicate logic (Open question 3).
- **Behaviour change:** none intended. One invisible difference, dead script globals, is described in section 3, item 2.
- **NEW TESTS (R3d, new module):** `tests/test_service_extraction_stage2.py`, 28 tests. `run_extraction` and `ExtractionRequest` are new capability, and they are labelled as such in the module docstring.

## 1. What moved where

APP = `src/t2pw/app/streamlit_app.py`, EXT = `src/t2pw/service/extraction.py`.

| what | base `5af91e07` APP | tip `590563ed` EXT |
|---|---|---|
| `emit_rag_loop_messages` (now takes `reporter=`) and `rag_loop_session_record` | 246-272 | 747-777 |
| submit block: R3c's six rebinding lines, then `final_payload = stage_one_in_scope` through the publish's last line, `st.session_state.pop("post_pipeline_artifacts", None)`. Covers Stage 2 and its `PipelineFailure` exit, the Stage-2 `merge_additions`, `_rag_merge_site` with the RAG `merge_additions`, `run_rag_rounds`, the loop messages and the `rag_loop` record, `_write_reaction_preservation_report`, `build_and_save_draft_graph`, and all 22 session-state writes and pops | 1924-2083 | `run_stage_two` body, 790-993 |
| new: `KIND_STAGE_TWO_FAILED`, `ExtractionResult`, the `run_stage_two(...)` signature | - | 744, 779-788, 790-814 |
| new: `ExtractionRequest` and `run_extraction(request, *, reporter, state)` | - | 995-1098 |

**The app call site** (APP 1897-1918) replaces base 1924-2083:

```python
_stage_two = run_stage_two(text, _stage_one, pathway_context=pathway_context, run_inference=run_inference,
                           enable_chunking=enable_chunking, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                           infer_rounds=infer_rounds, infer_attempts=infer_attempts, temperature=temperature,
                           infer_tokens=infer_tokens, rag_incomplete_flag=rag_incomplete_flag,
                           export_mode=export_mode, state=st.session_state, reporter=StreamlitReporter(st))
if _stage_two.failure is not None:
    st.stop()
```

**Stays in the app**, unchanged:

- `reset_quarantine_state(PROJECT_ROOT / "outputs")` and `st.session_state["token_stats"] = ...`, directly after the call and in the same order as before. They are base 2084-2090.
  - The reason: the golden-seam test `test_streamlit_quarantine_boundary.py::test_the_reset_is_wired_into_the_start_of_a_pipeline_run` requires at least two `reset_quarantine_state(...)` Name calls in APP. Moving this one would leave one.
  - The token stats stay with it, so the order is unchanged (the reset first, then the token stats).
- The ambiguous-review gate, `reset_refinement_state()`, and `if submit:` as the caller (the brief; R3b carry-forwards 1-2).
- The `render_attempts` shim (APP 542). This is R3c carry-forward Q2. The shim cannot go: two app callers remain, both in the `pipeline_ready` display panel (the Stage-1 and Stage-2 chunk listings). That panel is not part of R3.

**Names kept reachable on the app module:** `rag_loop_session_record` is re-imported from EXT, because c055:720 calls `app_module.rag_loop_session_record`. `emit_rag_loop_messages` is **not** kept: its only caller moved, nothing in `tests/` or `src/` references it, and its signature gained `reporter=`. R3b set the precedent with `_start_extraction_diagnostics`. The app keeps its now-unused imports (`merge_additions`, `run_stage_two_with_feedback_loop`, `build_and_save_draft_graph`, `run_rag_rounds`, `_write_reaction_preservation_report`, `current_extraction_diagnostics`), so those names still resolve on the module. c055 uses `app_module.run_rag_rounds`.

**The service API:**

- `run_stage_two(text, stage_one_result, *, pathway_context, run_inference, enable_chunking, chunk_size, chunk_overlap, infer_rounds, infer_attempts, temperature, infer_tokens, rag_incomplete_flag, export_mode, state, reporter) -> ExtractionResult`.
  - Every parameter after the first two is keyword-only, with no default. That includes `state` and `reporter`.
  - It takes R3c's `StageOneResult` whole and unpacks it in its first six lines. Those are R3c's app rebinding lines, moved.
  - The `int(...)` and `bool(...)` conversions stay inside, verbatim.
- `export_mode` is the app's local value, which is what base passed to `run_rag_rounds`. `pwml_strict_db` is read as `state.get("pwml_strict_db", True)`, so it still defaults to True when the checkbox never rendered (scout hazard).
- **The one `st.stop()`** becomes `return ExtractionResult(failure=InputFailure("stage_two_failed", "Inference stage failed: {failure}"))`. It returns after the same `error` and the same attempts expander, in the same order. Nothing is published on that path, as in base.
- `ExtractionResult(pathway_context, final_payload, failure)`. Everything else the run produced is in `state`, under the app's session keys. Nothing in the app reads the result beyond `failure`.
- **Lookups at call time.** The imports are function-local: `pipeline` and `postpipeline` as module objects, and `rag_orchestration` only inside the RAG branch, as in R3c.
  - `merge_additions` is imported **by name** inside the function. C-060a lifts the two merge statements by AST and execs them with `merge_additions` as a plain name, so it has to stay a Name. The import still runs on every call, so a patch of `pipeline.merge_additions` reaches it. The mutation `bound-merge_additions`, which moves it to module level, is killed.
  - The two `pathway_context_from_stage_zero` imports stay inside the block, verbatim (scout hazard).
  - Importing EXT still loads 13 `t2pw` modules, and R3c's `test_importing_the_service_loads_no_stage_one_module` passes.
- `run_extraction(request, *, reporter, state)` composes `assemble_inputs` → `run_stage_zero` → `run_stage_one` → `run_stage_two`, in the app's order, and returns each step's failure as it is. It adds only two lines, each a copy of an app line:
  - `state["export_mode"] = request.export_mode`. It is the copy of APP 1643, the radio's session write that Stage 1 reads. Without it, Stage 1 and Stage 2 could read two different modes.
  - `user_task_context = (request.user_task_context or "").strip() or None`. It is the copy of APP 1844.

  It has **no ambiguous-review gate**: Open question 3 covers it, and the unit test `test_run_extraction_refuses_an_ambiguous_review_in_stage_one_before_any_llm_call` pins the interim behaviour. It does not call `reset_refinement_state`, the quarantine reset or the token stats, which stay in the app.
- `ExtractionRequest` is a frozen dataclass of the form's 20 inputs, raw, with **no defaults**. That means no new config.

**Verbatim proof:** `evidence/R3d/verbatim-diff.txt` (from `scripts/verbatim_diff.py`; CR stripped, dedented). The only changed lines are:

- `st.*` replaced by `reporter.*`, `st.session_state[...]` and `.pop` replaced by `state[...]`, and `StreamlitReporter(st)` replaced by `reporter`;
- `_stage_one.x` replaced by `stage_one_result.x`;
- bare callees replaced by `_pipeline.`, `_postpipeline.`, `_rag_orchestration.` and `_diag.current()`; `PipelineFailure` became `_pipeline.PipelineFailure`;
- `st.stop()` replaced by `return ExtractionResult(failure=...)`, with the error text bound to `message` first;
- `render_attempts(...)` and `emit_rag_loop_messages(...)` gain `reporter=reporter`;
- the function-local imports, and a two-line R3d note under the C-060a comment.

No message text, order or condition changed. The moved comments are verbatim, so their line references (`:5514`, `OWNED_FLOOR = 4640`) are as stale as they were in base.

## 2. Patch sites, AST and exec readers (mapped before moving anything)

I grepped `tests/` for every name the moved code calls:

- `setattr(` / `patch(` / `patch.object(` / `setitem(` with the name as a string, including multi-line calls;
- direct `X.name =` assignments;
- `_source_tree`, `ast.parse`, `inspect.getsource`, `read_text(` and `APP_PATH` in the 34 test files that read source and mention the app;
- the moved range's string literals (`after_stage2`, `run_inference_enabled`, `Inference stage failed` and so on) outside `session_state` reads.

I opened every hit.

**Patch targets:**

| name the moved code calls | source module | who patches it | how the tip still sees it | proof |
|---|---|---|---|---|
| `run_stage_two_with_feedback_loop` | `t2pw.pipeline.pipeline` | **c055:843** (`offline_pipeline`, source module, before AppTest); `test_service_extraction_stage1:427,457` | `_pipeline.run_stage_two_with_feedback_loop` at call time | c055 14, stage1 22; mutation `bound-run_stage_two_with_feedback_loop`: 18 failed |
| `build_and_save_draft_graph` | `t2pw.pipeline.pipeline` | **c055:336,549** on the source module (the `_persist` redirect keeps AppTest from writing the checkout's `tmp/draft_graph.json`); c055:354,381,550 patch rag_orchestration's own binding (untouched) | `_pipeline.build_and_save_draft_graph` at call time | c055 14 (`persistence == [None]`); mutation `bound-...`: 5 failed |
| `merge_additions` | `t2pw.pipeline.pipeline` | no patch; **c060a lifts both statements by AST** (below) | function-local `from ... import merge_additions` (a Name, at call time) | mutation `bound-merge_additions`: 7 failed |
| `pathway_context_from_stage_zero` | `t2pw.pipeline.entity_admission` | none; c060a execs the in-block imports | the in-block imports, verbatim | c060a proof copy |
| `_write_reaction_preservation_report` | `t2pw.service.postpipeline` | none; s2orch:401 injects it into a lifted postpipeline namespace | `_postpipeline._write_...` at call time | mutation: 5 failed |
| `run_rag_rounds` | `t2pw.service.rag_orchestration` | none; c055 calls `app_module.run_rag_rounds` directly (still re-imported) | `_rag_orchestration.run_rag_rounds` at call time | c055 14; mutation: 5 failed |
| `extraction_diagnostics.current` (the publish's recorder) | `t2pw.pipeline.extraction_diagnostics` | stage0 and stage1 tests; s2orch:444 (namespace) | `_diag.current()` at call time | mutation `bound-diag-current-publish`: 4 failed |
| `PipelineFailure` | class | none | `_pipeline.PipelineFailure` | - |
| `reset_quarantine_state`, `llm_client_module.get_token_stats` | app | none | not moved | - |

No test patches any of these names on `streamlit_app` (`setattr(app...` / `patch("t2pw.app.streamlit_app...` with each name: no hit).

**AST and exec readers of the moved range** (this is the risk the brief named):

| test | what it reads | tip |
|---|---|---|
| `test_c060a_requested_pathway_wiring.py` (`_app_sites`, `_run_app_site`, lines 66-104) | **APP only**: exactly two `X = merge_additions(...)` Name-call assignments at or below `OWNED_FLOOR = 1440`. It execs each one, with the `entity_admission` imports between the floor and the site, in a namespace holding `merge_additions`, `stage_one_in_scope` / `final_payload`, `stage_two` / `rag_envelope`, `pathway_context` and `text` | **7 of 11 fail** (`expected two owned app merge sites: []`). The 4 that pass do not read the app. Open question 1 |
| `test_streamlit_stage2_orchestration.py::test_out_of_scope_filter_runs_between_...` (Q-R3c-1) | EXT `run_stage_one`, plus APP: exactly one `run_stage_two_with_feedback_loop` Name call after `run_stage_one(`; its `args[1]` is not `stage_one`; the first app `merge_additions` base is not `stage_one` | **fails** (`assert len(inference_calls) == 1`). Open question 2 |
| `test_streamlit_quarantine_boundary.py:1453` (golden seam) | APP: at least two `reset_quarantine_state(...)` Name calls, each with args | passes, because the reset **stays in the app** (section 1) |
| `test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform` | APP: one `conform_rag_additions_for_merge` call | already failing at base: `conform` left the app in R2. Same node and the same first failing assertion on tip and base |
| `test_c055_rag_loop_wiring.py:518-530` | the AST of `maybe_run_rag`, `run_rag_rounds` and `build_rag_round_stages` in `rag_orchestration.py` | untouched; passes |
| `test_streamlit_stage2_orchestration.py` `_load_function` (lines 79-99) | APP and postpipeline function bodies (`_json_artifact_entries` and others) | none of them moved; passes |
| `test_service_driver_bridge.py::test_app_still_joins_text_parts_that_way` | EXT, for the R3a join lines | unchanged; passes |
| `test_preprocessor.py` (4 tests repointed in R3b) | EXT: exactly two `preprocess(` calls, among others | no `preprocess(` was added; passes |

The other 26 source-reading files read code outside the moved range: the post-pipeline, the PWML step, `_safe_dict`, the widgets and so on.

## 3. Things the reviewer must look at

1. **The two failing source readers** (section 2; Open questions 1 and 2). This is the needs-decision.
2. **Invisible difference: dead script globals.**
   - Base bound about 19 script globals inside this range: `stage_one`, `chunk_details`, `stage_one_in_scope`, `out_of_scope_removed_reactions`, `rag_result`, the folded `user_task_context`, `final_payload`, `stage_two*`, `qa_hints`, `label`, `failure`, `rag_loop_record`, `stage2_preservation_report`, `draft_graph`, `qa_report`, `reaction_summary` and `_rag_merge_site`. The tip binds `_stage_two`. `user_task_context` keeps its stripped, pre-fold value.
   - Unlike R3c, this applies to the **success path too**. Nothing after the block reads any of these names without first re-assigning it. The `pipeline_ready` panel re-assigns `stage_one`, `chunk_details`, `stage_two*`, `qa_hints` and `final_payload` from session state at its top. The others are never read again.
   - The ivm script checks this **statically**: it parses the tip app and requires that no such name is loaded after `if submit:` before a store. That check is OK.
   - Every ivm scenario therefore reports `SAME-EXCEPT-DEAD-GLOBALS`, and it allows only that difference: every st call, state write and pop, callee call, outcome and final state must be identical.
3. **The reporter instance passed to `run_rag_rounds`** is the app's single `StreamlitReporter(st)` from the call, where base made a new one. `emit_rag_loop_messages` uses the same instance where base built its own. It is a stateless forwarder, the same class as R3c item 4, and ivm records the reporter type and `_st is st`.
4. **`run_extraction` is not called by the app** (by design: the gate). It is exercised by the unit tests and by the equivalence harness. Its two copied app lines are in section 1.
5. **Equivalence harness service arm** (section 5). It now starts from `case["full_text"]`, so it reads the request off the real form, and it reads the run's own products from the service `state`.
6. **One survivor among the eq mutations** (section 5): `eq-extraction-one-qa-round`.

## 4. The two failing source readers and the proposed edits (not applied)

Neither test can be fixed by moving less code. Both merge statements, Stage 2 and the publish are the task. Keeping the merge statements in APP as callbacks would leave headless `run_extraction` with no merge sites. It would then need a second copy in EXT, out of C-060a's reach, which is exactly what the app's own C-060a comment warns against.

**Proof copies** are in `evidence/R3d/proposed/`. They have no `test_` prefix, so the suite never collects them, and their `ROOT` is resolved from their own location. `scripts/proof_runs.py` ran each one on the committed tip, untouched and under temporary single mutations of the tip. Every run went through the bounded wrapper with the heavy lock, with survivors 0, and `git status -- src tests` was clean afterwards. **11/11 as expected** (`proposed-test-edits/summary.txt`):

| run | expected | got |
|---|---|---|
| c060a copy, untouched tip | pass | 11 passed |
| c060a copy, Stage-2 site passes the Stage-0 dict instead of the factory | fail | 3 failed (currency rows survive) |
| c060a copy, RAG site passes the dict | fail | 2 failed |
| c060a copy, RAG site wires `seed_text` | fail | 1 failed (`'seed_text' not in ...`) |
| order test (proposed), untouched tip | pass | 1 passed |
| order test, filter removed | fail | 1 failed (`0 == 1`) |
| order test, `run_stage_one` returns the raw extraction | fail | 1 failed |
| order test, Stage 2 gets the raw `stage_one` (in EXT) | fail | 1 failed |
| order test, `run_stage_two` binds `stage_one_in_scope` from `.stage_one` (review F1) | fail | 1 failed |
| order test, first merge base is the raw `stage_one` | fail | 1 failed |
| order test, the app hands `run_stage_two` something other than the `run_stage_one` result | fail | 1 failed |

- **c060a, the proposed edit** (`proposed/c060a-proposed.diff`, 3 code lines, 1 constant, 3 comment lines):
  - `_app_sites` and `_run_app_site` parse `EXTRACTION_PATH` (`src/t2pw/service/extraction.py`) instead of `APP_PATH`.
  - `OWNED_FLOOR` goes from 1440 to 790, the line of `def run_stage_two`.

  No assertion, namespace or selection rule changes. It is the same edit R1 and R2 already made to this file: R1 moved the floor 4640 → 2571, and R2 moved it 2571 → 1440, both as "line locator only". Every one of the 11 tests then passes, and the behavioural G9 proofs still bite (rows 2-4).
- **Q-R3c-1 order test, the proposed rewrite** (`proposed/s2orch_order_proposed.py`, the function only). It guards the same rule one level further in:
  - in EXT `run_stage_one`: unchanged;
  - in EXT `run_stage_two`: exactly one Stage-2 inference call (Name or Attribute); its `args[1]` is the name bound from `stage_one_result.stage_one_in_scope` (this closes review F1's gap); the first `merge_additions` base is not `stage_one`;
  - in APP: exactly one `run_stage_one(` and one `run_stage_two(`, in that order, and `run_stage_two` is handed the Name `run_stage_one`'s result was assigned to.

  That is about 20 changed lines in one function, beyond Q-R0-1, so it needs a PO ruling like Q-R3c-1.

## 5. Proof

**Inline vs moved** (`scripts/inline_vs_moved.py`; run on the committed tip `590563ed` through `bounded_run` with the heavy lock; `ivm.log` / `.json`, `worktree dirty=False`, survivors 0): **22/22, 0 unexplained, static dead-globals check OK.**

- The script cuts the real base range, from R3c's rebinding to the token-stats line, and the real tip range, from the R3d call site to the token-stats line. So the unchanged reset and token stats, and their order after the publish, are compared too.
- It lifts base `render_attempts`, `emit_rag_loop_messages` and `rag_loop_session_record` by AST. It execs both arms at module level against one recording fake `st`, with the real `pathway_context_from_stage_zero`.
- Per scenario it compares: the ordered st calls, with spinner and expander enter/exit; every session-state write **and pop**, with its value; every callee call with its arguments, including both merge sites' `PathwayContext`, recorded by field; the reporter's type and target; `state is session_state`; the outcome; the final state; and `rag_result.loop`.
- Scenarios:
  - inference: success, a non-dict result, a dict with no hints, inference off (False and 0), float widget values;
  - Stage-2 failures: `PipelineFailure` with a stage, without a stage, and without attempts; any other exception;
  - stale keys popped;
  - RAG rounds: all message kinds, `strict_db` off, research mode, inference off, a frozen result;
  - no rounds: not synthesized, no payload, an empty payload;
  - raises: the rounds raise, the draft graph raises;
  - the preservation report returns None.
- All 22 report `SAME-EXCEPT-DEAD-GLOBALS` (section 3, item 2).

**ivm non-vacuity** (`scripts/ivm_sanity.py`, `ivm-sanity/`): **9/9 single tip mutations detected.** They are: the spinner text; the publish order swapped; the `strict_db` default; the merge without `source_text`; the attempts not rendered; the loop messages dropped; the RAG merge without a context; the app not stopping on a failure; and the app's quarantine reset dropped.

**Mutations** (`scripts/mutate.py`, `mutation/`; every run through the wrapper with the heavy lock, survivors 0, `git status -- src tests` clean afterwards):

- **Against `tests/test_service_extraction_stage2.py`: 41/41 killed.**
  - 6 call-time lookups reverted to module-level bound imports: `run_stage_two_with_feedback_loop`, `merge_additions`, `build_and_save_draft_graph`, `_write_reaction_preservation_report`, `run_rag_rounds` and `diag.current`.
  - 27 behaviour mutations of the moved code:
    - Stage 2: the spinner; `int()`; the raw payload passed to Stage 2; the unfolded focus;
    - the failure exit: its message, its kind, falling through, the label fallback, the attempts not rendered;
    - `qa_hints`: its default;
    - the merges: the raw base, no `source_text`, a dict context, the RAG site without a context;
    - the RAG rounds: the `strict_db` default, `bool(flag)`, the messages dropped, the payload not taken, the unguarded loop attach;
    - the helpers: the session record keeping the payload, `emit` dropping json;
    - the publish: the order, a pop, the inference flag, the snapshot, the preservation stage name;
    - defaults on `state` and `reporter`.
  - 4 `run_extraction` mutations: the focus not stripped, the mode not written to state, a Stage-1 failure ignored, an input failure ignored.
  - 4 app call-site mutations, killed by the real-app AppTest tests: no stop, the reset dropped, the token stats dropped, `state={}`.
- **Against the equivalence test: 2/3 killed.**
  - `eq-extraction-no-inference` and `eq-merge-no-source-text` were killed.
  - **`eq-extraction-one-qa-round` survived:** `infer_rounds=1` instead of the form's 2 gives the same canonical payload and the same PWML on this one fixture case.
  - The service arm does not check for unused cassette entries (the app arms do, in `test_a`), so a skipped Stage-2 QA round is invisible when its output changes nothing.
  - This is an equivalent mutant **for this case**, not a harness defect. The request takes `infer_rounds` from the real form widget, not from a constant. Open question 4 has a cheap tightening.

**Unit tests** (`tests/test_service_extraction_stage2.py`, 28):

- 26 offline tests: every branch above, the exact state-write order, the call-time patch reach, the keyword-only signatures, no defaults on `ExtractionRequest`, the composition and failure returns of `run_extraction`, the ambiguous-review interim, and one `run_extraction` end to end with only the LLM stages stubbed.
- 2 real-app AppTest tests:
  - a successful submit publishes every key, then the app's reset clears a seeded `quarantine_report` and `post_pipeline_artifacts`, then `token_stats` appears;
  - a Stage-2 `PipelineFailure` shows exactly the one error and the attempts expander, and publishes nothing; the seeded `quarantine_report` survives, because the reset sits after the stop, as before.

**Equivalence harness** (`tests/equivalence/harness.py`, the `TODO(R3)` arm and the module docstring line naming it; nothing else changed): `run_service_arm` now runs the whole case on the service.

1. `new_root`; then the cassette, `isolate` and the clock freeze, as before.
2. It loads the **real app form** in `AppTest`, never submitted. It sets the export and input modes the way the driver does, and reads every widget the submit block reads, by label. With the driver's `_paper_text` and `_extraction_focus` that gives the `ExtractionRequest`, so the request carries no copied defaults.
3. `run_extraction(request, reporter=LoggingReporter(), state={})`.
4. `run_post_pipeline_sbml_artifacts(propagate_context_organism(final_payload, state["pathway_context"]), **settings)`. The settings are the app's recorded sidebar settings, with the run's own `qa_report`, `reaction_summary`, `rag_evidence_context`, `pathway_context`, `outputs_dir` and `state`.
5. `run_pwml_export`, likewise.

The test file is unedited, and `test_c` keeps its assertions: the same canonical hash and payload, the same verdict, and the same PWML bytes as the app. **The equivalence test passed 14/14, alone, on tip and base.** A wrong request value fails it loudly: the Stage-0/1/2 LLM keys include `max_tokens` and `temperature`, and a cassette miss fails `test_c`.

## 6. Tests: tip vs base

All runs went through `bounded_run.py` in the foreground, with the venv python for both the wrapper and the child.

- Environment: `PYTHONPATH=<tree>/src`, `pinned_pytest --expect-tree <tree>` with no `--`, and `T2PW_OFFLINE_CURATOR=1`.
- Each run had a fresh `--basetemp=C:/t/wsR3dtmp/<pfx>-<chunk>`, whose parent was created first.
- `--heavy-lock R3d` on every chunk except `web`. Every heavy run was gated (`scripts/waitfree.py`): 97 observations in `waits.log`, all free.
- Helpers: `scripts/run.sh`, `chunks.sh` (R3c's list, plus `stage1` and `new` = the stage2 file) and `compare.sh`. The full table is in `compare.txt`.

**Trees:** tip `t-` is `C:/t/ws-R3d` @ `590563ed`; base `b-` is the detached `C:/t/ws-R3d-base` @ `5af91e07`. Every report shows `worktree dirty=False`, and survivors were 0 on every run.

| chunk | tip @ 590563ed | base @ 5af91e07 | failing-node set |
|---|---|---|---|
| g1 / g2 / g3 / g4 (own processes) | 2 / 8 / 4 / 24 = **38** | same | identical (none) |
| **eq** equivalence, alone | **14 passed** | 14 | identical |
| 2a | 78 passed, 1 failed | same | identical (c052) |
| **2b** (c060a, s2orch, research, paper dictionary) | **126 passed, 8 failed** | 134 passed | **DIFFERENT**: the 7 c060a + 1 Q-R3c-1 (section 4) |
| 2b-c055 | 14 | 14 | identical |
| 2c / 2c-c126 | 101 passed, 3 skipped, 1 failed / 2 | same | identical (prefreeze_third) |
| 2d / 2d-stage3 / 2d-c119g9 | 413 passed, 1 failed / 28 / 3 | same | identical (rag_seed) |
| 2e preprocessor / 759b / hook / 759afold | 44 / 7 / 1 / 2 | same | identical |
| 2f / 2f-driver / 2f-gate | 32 passed, 4 skipped, 1 failed / 61 / 14 | same | identical (batch_preflight) |
| 2g / c082pp / c056d / c119sup / 759d / revview | 162 + 8 skipped / 1 / 14 / 37 / 4 / 11 | same | identical |
| at-c052 / at-c082sr / at-c083 / at-pf3 / at-bridge | 12+1f / 8 / 12 / 11+1f / 37 | same | identical |
| web (13 files, incl. bridge) | 301 | 301 | identical |
| svc (extraction + stage0 + reporter) / stage1 | 57 / 22 | 57 / 22 | identical |
| **new** test_service_extraction_stage2 | **28 passed** | n/a | - |
| solo c060a | **4 passed, 7 failed** | 11 | DIFFERENT (Open question 1) |
| solo s2orch | **11 passed, 1 failed** | 12 | DIFFERENT (Open question 2) |
| solo clobber / scope / 759a / triage / admission | 8 / 5 / 37 / 14 / 9 | same | identical |
| solo rag_seed | 13 passed, 1 failed | same | identical (known) |
| **SMOKE** (TEST_MATRIX list, 22 files) on the tip `bdd75f05` (code = `590563ed`) | **511 passed** | - | census 83 after |

- Golden: 38 passed on tip and base, and the golden files are unchanged.
- Known failures: the four (c052, prefreeze_third, rag_seed, batch_preflight) have identical node ids on tip and base.
- The proposed edits make solo c060a 11 and the order test pass (section 4). With both applied, 2b would be 134, equal to base.

## 7. What failed, dead ends, incidents

1. **The two source readers** (section 4). I found them in the pre-move map and confirmed them in the discovery runs (`C:/t/wsR3dtmp/dev`, dirty tree, not committed). I made no test edit.
2. **A mutation wrote the worktree's scratch files.** `bound-build_and_save_draft_graph` disarmed the test's fake, so the real function wrote `tmp/draft_graph.json`, `tmp/qa_report.json` and `tmp/reaction_summary.txt` **in `C:/t/ws-R3d`** (tracked files; `pipeline.TMP_DIR` is module-bound).
   - I saw it in `git status` after that batch and restored the three files with `git checkout --`.
   - Nothing was committed, and the primary checkout was not touched. Later batches showed a clean `git status`.
   - That is the mutation doing its job: the patch no longer landed.
3. **Machine restart mid-run.** The first base batch for 2a and onwards died (`exit 107`) when the machine restarted, and it left no reports. Every base chunk was re-run from scratch afterwards. The 48 reports written before the restart (43 tip, 5 base) were checked: each is complete, with `final_surviving_count: 0`.
4. **The proof copies were first placed outside the tree.** `pinned_pytest` refused them (`SELECTION_OUTSIDE_EXPECTED_TREE`, exit 98, no test run). I moved them under `docs/webserver/evidence/R3d/proposed/` with a self-locating `ROOT`, committed them (docs only, `bdd75f05`), and re-ran on the committed tree. The refused attempt is not counted.
5. **Tooling slips, none of which wrote a wrong file:**
   - A bash heredoc containing an apostrophe failed to parse, and nothing was appended. I made that edit with the Edit tool instead.
   - Two `sed` edits did not match, and nothing changed. I redid them with the Edit tool.
6. **Batches over 180 s.** One base batch (g1-g4 + eq) ran past 180 s, and the harness moved the foreground call to the background. It was the same process tree, not detached, and I waited for its completion notification; survivors were 0. Later batches stayed under 180 s.
7. I did not run `python -` or any stdin heredoc. Every Python run used a script file or `-c`.

## 8. Open questions

1. **c060a repoint.** It needs orchestrator authorisation under Q-R0-1. **A (recommended):** apply `proposed/c060a-proposed.diff`:
   - **Change:** `_app_sites` and `_run_app_site` parse EXT; `OWNED_FLOOR` goes 1440 → 790.
   - **Cost:** 3 code lines, 1 constant and 3 comment lines in one test commit, the same kind of edit as R1 and R2.
   - **Reviewer check:** that only a path and the line locator changed.

   B: keep both merge statements in the app as callbacks passed to `run_stage_two`. c060a stays unedited, but `run_extraction` then has no merge sites, or needs an unguarded second copy in EXT. That defeats R3 and C-060a. Not recommended.
2. **The Q-R3c-1 order test.** It needs a PO ruling like Q-R3c-1, because the edit goes beyond a path. **A (recommended):** apply `proposed/s2orch_order_proposed.py`:
   - **Change:** about 20 lines in one function.
   - **What it guards:** the same rule, and it closes review F1's gap.
   - **Proof:** 6/6 vacuity mutations fail it, and the untouched tip passes it.

   B: retire the test, and rely on the behavioural pins: the new tests' `stage-two-raw-payload`, `merge-base-raw` and the stage1 test's filter checks, all killed. Cost: it loses the structural pin.
3. **The headless ambiguous-review gate.** **STOPPED, as the brief requires.** A pure predicate exists (`preprocessor.is_ambiguous_multi_example_review_context`). But the app's `pipeline_error` dict, with its status, message and `candidate_examples`, exists only inline in the app's gate. Building "the same dict" in the service means a second copy.
   - **Interim** (what is committed): `run_extraction` has no gate. Stage 1's own entry check (`pipeline.run_stage_one_with_chunking` → `_ambiguous_review_scope_failure`) refuses before any LLM call, as `InputFailure("stage_one_failed", "Extraction failed: multi_example_review detected with no selected_example. Extraction skipped to prevent mixed-pathway output.")`. That message text is the same as the app dict's `message`. A unit test pins this. Stage 0's diagnostics are still written; nothing is published.
   - **Options:**
     - (a) a service helper `ambiguous_review_error(pathway_context) -> dict`, which `run_extraction` calls, returning kind `ambiguous_review_scope`, with the app gate left byte-identical. Cost: about 12 lines, two copies of one dict literal, and a new test that lifts the app gate's dict by AST and asserts equality, to stop drift.
     - (b) move the gate into EXT and make the app call the helper. Cost: one copy, but it edits `test_stage0_case_c_retry_clobber.py`'s exec namespace, which needs a PO ruling. It also needs a Reporter `subheader` primitive.
     - (c) keep the interim and map `stage_one_failed` whose message starts with `multi_example_review detected` to `ambiguous_review_scope` in R5's error mapping. Cost: nothing now, but it is a string match.

     I recommend (b) in R5, when the clobber test is touched anyway, and (c) until then.
4. **Equivalence tightening** (optional, test-only). The service arm could also return `cassette.unused()` and be required to consume exactly what the app consumed. That would kill `eq-extraction-one-qa-round`, but it needs an assertion added to `test_equivalence_app_vs_service.py`, an existing test, so it needs authorisation. Cost: 2 lines.
5. **Carried, unchanged:**
   - R3c Q3: `InputFailure` names Stage-1 and Stage-2 failures too; rename it in R5.
   - R3c Q5: the isolation fault between `test_service_extraction.py` and c055, a separate ticket.
   - R3a N3 (PDF paths AppTest cannot drive): **already covered** by the R3a unit tests on `assemble_inputs` in `tests/test_service_extraction.py`. They cover: the PDF-only reporter sequence (109), the caption (136), the preview (146), texts before PDFs (155), upload with no PDF (164), Text+PDF with no PDF (170), an empty PDF (184), an unreadable PDF (193), the temp-file cleanup (205), the page range (227), and `skip_refs`/OCR (237). So no new test was added.

## 9. Rulings, the three test edits and the re-runs

**Authority:** `docs/webserver/results/PO-RULINGS-2026-11-04.md`, Addendum 5 (on `webserver/refactor` @ `3bda46dd`), plus the orchestrator note there for c060a.

| ref | ruling | what was done |
|---|---|---|
| Q-R0-1 (orchestrator) | c060a: point its two parse sites at `service/extraction.py` and move `OWNED_FLOOR` from 1440 to 790. The reviewer must confirm it selects the same statements as at base | Applied exactly `proposed/c060a-proposed.diff` |
| Q-R3d-1 (PO) | Rewrite the order test again, this test only | Applied `proposed/s2orch_order_proposed.py` to that one function |
| Q-R3d-2 (PO) | Keep what ships for the headless ambiguous-review refusal; map it in R5 and move the gate then. No service copy of `pipeline_error` now | **Deferred to R5.** No code change. Stage 1 refuses before any LLM call, as `stage_one_failed`, pinned by `test_run_extraction_refuses_an_ambiguous_review_in_stage_one_before_any_llm_call` |
| Q-R3d-3 (PO) | Add the check that every recorded LLM reply is consumed | 2 lines in `test_c`, reading a new `unused` field of the service arm |

**The test commit `3ade39e3`** (test only; `git show --numstat`; made by `scripts/apply_test_edits.py`):

| file | +/- | what |
|---|---|---|
| `tests/test_c060a_requested_pathway_wiring.py` | +7/-3 | the two parse sites, `OWNED_FLOOR`, the `EXTRACTION_PATH` constant, 3 comment lines |
| `tests/test_streamlit_stage2_orchestration.py` | +24/-11 | the one function only |
| `tests/test_equivalence_app_vs_service.py` | +2/-0 | a comment line and the assertion, in `test_c` after the `misses` / `refused` line |
| `tests/equivalence/harness.py` | +3/-1 | `"unused": cassette.unused()` on the service arm's two return shapes, which the assertion reads |

The assertion: `assert [row for row in service.get("unused", []) if row[0] == "llm"] == [], service.get("unused")`. It is LLM-only, as the ruling words it.

**The base run proves nothing about this assertion, and I chose to record that rather than change base.** The assertion reads `service.get("unused", [])`, and the `"unused"` keys were added to the harness in the same commit. Base's harness has no such key, so on base the assertion compares `[]` to `[]`. The base result (14 passed) shows only that the added lines do not break the file on base. I did not run base with the harness change: base's service arm starts from the app's Stage-2 payload and never makes the Stage-0-2 calls, so I expect those replies to show as unused and the assertion to fail there. That expectation is reasoning, not a measured result. The proof that the check works is on the tip: it kills the one-QA-round mutation (below).

**Re-runs on the tip `3ade39e3`** (`evidence/R3d/post-rulings/`; bounded, heavy lock R3d, gated, fresh basetemp, foreground, `worktree dirty=False`, survivors 0):

| run | result | required |
|---|---|---|
| solo c060a | **11 passed** | 11 |
| solo s2orch | **12 passed** | 12 |
| chunk 2b | **134 passed** (first report: 126 + 8 failed) | 134, equal to base |
| golden g1 / g2 / g3 / g4, each alone | 2 / 8 / 4 / 24 = **38** | 38 |
| equivalence, alone, tip | **14 passed** | 14 |
| equivalence, alone, base `5af91e07`, edited test as a scratch copy | **14 passed** (vacuous for the new line, see above) | 14 |
| new `test_service_extraction_stage2` | **28 passed** | 28 |
| SMOKE | **511 passed**, census 83 | 511, 83 |

`3ade39e3` changes only those four test files against `590563ed` (plus docs), so section 6's other chunks stand for it. No other chunk runs these four files, except `2b`, the solo runs and the equivalence run above.

The base scratch copy was the committed test with the two differences shown in `post-rulings/eq_check_copy.diff`: the `sys.path` line, and an import of the suite's `conftest` autouse fixture, because the copy lives outside `tests/`. The full copy is not committed: it would only repeat the test, including the fake key shapes `test_fixture_holds_no_secrets` plants, which trip the secret grep. It sat in `C:/t/ws-R3d-base/docs/webserver/evidence/R3d/` for that run and was deleted afterwards; base is clean at `5af91e07`. The wrapper's `worktree dirty=False` on that report does not count an untracked file, so that field says nothing about the copy.

**Vacuity.** Each edit was a temporary change to the committed tip, restored after its run; `git status` was clean afterwards. Each run went through the wrapper with the heavy lock, survivors 0.

| check | temporary edit | expected | result | failing assertion |
|---|---|---|---|---|
| order test (`post-rulings/order-test-vacuity/`, **8/8 as expected**) | none, untouched tip | pass | 1 passed | - |
| | filter removed (required) | fail | 1 failed | `assert 0 == 1` (`len(filter_calls) == 1`) |
| | filter reordered, before the extraction (required) | fail | 1 failed | `assert 612 < 608` |
| | `run_stage_one` returns the raw extraction (required) | fail | 1 failed | `returned[0].id == filtered.elts[0].id` |
| | Stage 2 fed the raw `stage_one` (required) | fail | 1 failed | `stage_one_arg.id != "stage_one"` |
| | first `merge_additions` fed the raw `stage_one` (required) | fail | 1 failed | `merge_base_arg.id != "stage_one"` |
| | `run_stage_two` binds `stage_one_in_scope` from `.stage_one` (R3c review F1) | fail | 1 failed | `in_scope_from[0].attr == "stage_one_in_scope"` |
| | the app hands `run_stage_two` a result other than `run_stage_one`'s | fail | 1 failed | `args[1].id == stage_one_result.id` |
| equivalence (Q-R3d-3, `post-rulings/mutation-rerun/`) | `run_extraction` passes `infer_rounds=1` (the earlier survivor) | fail | **killed: 1 failed, 13 passed** | `assert [('llm', 5)] == []` |

The equivalence mutation count is now **3/3 killed** (2/3 in the first report).

**c060a selects the same statements** (the orchestrator's condition; `scripts/c060a_selection.py`, `post-rulings/c060a-selection.log`, through the wrapper, survivors 0). It applies c060a's own selection rule to base APP with floor 1440 and tip EXT with floor 790, and compares `ast.unparse` of what it selects:

| site | base lines (import, statement) | tip lines | |
|---|---|---|---|
| Stage-2 merge and its gate import | 1969, 1971 | 872, 874 | IDENTICAL |
| RAG merge and its gate import | 2024, 2026 | 929, 931 | IDENTICAL |

The behavioural bite of the repointed c060a was shown on the proof copy (section 4, rows 2-4).

**Open questions after the rulings:** 1 and 2 are resolved. 3 is deferred to R5 (Q-R3d-2). 4 is done (Q-R3d-3). 5 is unchanged.

**Incidents in this round:**

1. **Machine restart during SMOKE** (12:32). The run `R3d-r-smoke` wrote no JSON report, so it counts for nothing. Its partial log was discarded. The orchestrator verified the stranded lock's holder was dead and cleared the lock; I did not touch it. SMOKE was re-run with a fresh basetemp as `r2-smoke`. Every other post-ruling report was checked first: each is complete, with `final_surviving_count: 0`, at `3ade39e3` (base check at `5af91e07`), so they stand.
2. The order-vacuity script was first written into the tip's evidence directory, which would have left an untracked file during the runs. I moved it to the scratch directory before running it, and copied it back for this commit.
3. The secret grep before this docs commit hit one line, in the staged full copy of the equivalence test: the fake key shape its `test_fixture_holds_no_secrets` plants on purpose, not a real value. The gate held the commit. I dropped the full copy from the evidence (the diff stays) and re-ran the grep, which was clean.

## 10. Independent run, review and merge (added by the orchestrator)

**Independent run.** ws-test-runner (sonnet/medium) ran on trial merge `ee35bc4f` (`21417b44` onto `7dd5b1d1`, tree `62cb27ef`) in `C:/t/ws-R3d-trial`. Every run had 0 survivors, and every gate wait was 0 or 1 s. Every run exited 0, except `rag_seed`, which has a known failure on base too.

| run | passed | expected |
|---|---|---|
| golden | 2 + 8 + 4 + 24 = 38 | 38 |
| equivalence (alone) | 14 | 14 |
| service | 57 | 57 |
| `test_service_extraction_stage1` | 22 | 22 |
| `test_service_extraction_stage2` (new) | 28 | 28 |
| c060a (alone) | 11 | 11 |
| `test_streamlit_stage2_orchestration` (alone) | 12 | 12 |
| chunk 2b | 134 | 134 |
| `test_preprocessor` | 44 | 44 |
| c055 | 14 | 14 |
| clobber, 759d, 759b, 759a protein, 759a fold, scope_fallback | 8, 4, 7, 37, 2, 5 | same |
| rag triage, rag admission | 14, 9 | same |
| rag_seed | 13 + 1 known failure (`test_the_orchestrator_passes_the_merge_base_to_conform`) | same as base |
| web+bridge | 301 | 301 |
| SMOKE | 511 | 511 |

The census count is 83. The reports are in `docs/webserver/evidence/R3d/independent/`.

**Review round 0.** ws-reviewer (opus/high) reviewed the diff `5af91e07..21417b44` and returned **APPROVE WITH NOTES**. All gates hold: boundary, no biology change, behaviour preserved, tests, evidence, secrets, RESULT doc.

Test edits:

- **c060a, confirmed.** The edit changes only the two parse sites, the `OWNED_FLOOR` line locator, the `EXTRACTION_PATH` constant those sites need, and comments. c060a's own selector picks base 1971/2026 (imports 1969/2024) and tip 874/931 (imports 872/929), and `ast.unparse` of the selected statements is identical. The test still fails under mutation of the moved statements.
- **Q-R3d-1 order test, confirmed.** It guards the same rule. All 5 required mutations fail it on the reviewer's scratch copy.
- **Q-R3d-3, confirmed.** It only adds the check. The one-QA-round mutation is killed. The base run is vacuous, as § 9 states.

Checks on the move:

- **The harness is not weakened.** The `test_c` assertions are intact, each arm has its own cassette and root, and every run product now comes from the service `state`. The service arm now covers Stages 0-2, so the test is stricter than at base.
- **State writes.** A normalized diff of base 1924-2083 against the moved body shows only the expected changes. All 22 writes and pops are unchanged in order.
- **What stayed in the app.** Publish, then reset, then token stats is unchanged. A Stage 2 failure stops before the reset, as at base. The `render_attempts` shim has two remaining callers, both in the display panel, so it stays.
- **Patch sites.** Every patched name is looked up at call time. No test patches a moved name on the app module.
- **Secrets and census.** The scan of 470 evidence files found no key shape. It is a shape scan, not a comparison against `.env` values.

Findings, all notes:

- **F1, for the PO.** The recorded fixture cannot see Stage 2 additions, the out-of-scope filter, RAG or a non-default export mode (`additions` 0, `out_of_scope` empty, mode `pathwhiz`, `RAG_ENABLED=0`). Three mutations pass the equivalence test 14/14: additions replaced by `{}`, the first merge fed raw `stage_one`, and `state["export_mode"]=None`. R3d did not weaken anything here, because base's service arm started after Stage 2. The additions and raw-merge-base mutations are killed by unit tests. Equivalence must not be counted as covering these paths for R4-R6. A second fixture is a product call.
- **F2.** `service.get("unused", [])` at `tests/test_equivalence_app_vs_service.py:207` passes silently if the key is dropped. It was needed for the base run. Use `service["unused"]` from R4.
- **F3.** Four mutations still pass the order test: `final_payload = stage_one`, rebinding `stage_one_in_scope` after the filter, an aliased merge base, and `run_extraction` passing a replaced result. All four are killed by the stage1 and stage2 suites.
- **F4, for R5.** "Before any LLM call" means before any Stage 1 call; Stage 0 has already called the LLM. The refusal is the first statement of `run_stage_one_with_chunking`. The app gate is byte-identical to base. The headless path leaves `pipeline_ready` unset where the app sets False, so a reused `state` could stay stale. Handle this in R5 with the Q-R3d-2 mapping.
- **F5.** Stale wording only (`test_equivalence_app_vs_service.py:13,195`, `extraction.py:923`).

The reviewer's files are in `docs/webserver/evidence/R3d/review/`. Its scratch source copy and pytest temp directories are not committed. The one secret-scan hit in that folder is the regular expression inside the reviewer's own `probes/evidence_check.py`.

**Merge:** `2e3d6a28` (`--no-ff` into `webserver/refactor`). Its tree differs from the trial merge only in `docs/webserver/LEDGER.md`. The post-merge result is recorded in the LEDGER.

**Carried forward:**

- R4: use `service["unused"]` (F2), and fix the stale wording (F5).
- R5: the Q-R3d-2 mapping, plus the `pipeline_ready` reset for reused state (F4).
- PO: a second equivalence fixture (F1).
