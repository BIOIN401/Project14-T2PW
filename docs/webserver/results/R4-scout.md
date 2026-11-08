# R4 scout: "Run audit and DB mapping" handler and PWML step
Measured at `2e3d6a28` in `C:/t/ws-R3d-post`. APP = `src/t2pw/app/streamlit_app.py` (3483 lines). Nothing was run. The PLAN's old line ranges are stale; use the ones below.

## Headline
1. **Handler:** APP:2877-3082 (206 lines). It sits inside `if st.session_state.get("pipeline_ready"):` (APP:1927) and is plain script code. It has **no `st.stop()`, `return` or `raise`**, so R3's "stop becomes a returned failure" is not needed. It has a try/except/finally instead.
2. **PWML step:** it is not in the handler. "Generate PWML" is `refinement_generate_pwml` (APP:1333-1351) in `_render_review_refine_section`. The export logic is `_generate_pwml_from_refinement_working_json` (APP:1003-1061) plus `_write_reviewed_payload_snapshot` (APP:983-1000) and `_pwml_reference_path` (APP:975-980). Together that is about 87 lines. It returns dicts and has no `st.stop()`.
3. **Four traps, none of them in the Streamlit calls:**
   - **Reset call:** the golden `test_streamlit_quarantine_boundary.py:1464-1476` needs at least 2 `reset_quarantine_state(...)` Name calls in APP with arguments. Only APP:1924 and APP:2890 qualify, so the call at APP:2890 **must stay in the app**.
   - **Artifacts-call count:** `test_streamlit_stage2_orchestration.py:1055` demands exactly 2 `run_post_pipeline_sbml_artifacts` Name calls in APP (the other is the legacy SBML button, `build_legacy_sbml=True`). Moving this call breaks it. It needs a repoint, so a PO ruling like Q-R0-1 and Q-R3d-1.
   - **Missing helper:** the `Reporter` protocol lacks `exception` and `write`.
   - **`st.session_state.refinement_* = ...`:** attribute-style writes need item assignment on a plain mapping.
4. **Suggested split:** R4a (PWML export step, low risk), R4b (handler, with test repoints needing rulings), R4c (equivalence harness and carry-forwards).

## 1. Line ranges and helpers (APP)
| range | what | used by |
|---|---|---|
| 2877-3082 | handler: `if st.button("Run audit and DB mapping", key="pwml_generate_btn")` | |
| 2834-2875 | widgets above it (name, subject, size, bg, strict_db, grounding). Stay in the app as UI. | |
| 2834-2839 | `_project_root_pwml`, `_ref_candidates`, `_ref_path_pwml` are **dead locals** (`_ref_path_pwml` is read nowhere else) | none |
| 3084-3090 | post-block render, `_render_research_mode_panels(...)`. Not part of the move. | |
| 975-1061 | `_pwml_reference_path`, `_write_reviewed_payload_snapshot`, `_generate_pwml_from_refinement_working_json`. Used only at APP:1336 and by each other (975/1050, 983/1036). | PWML step only |
| 1333-1351 | button, try/except, `pwml_export_result` write, success/error | PWML step |
| 504-510 | `quarantine_session_state`. Used at APP:1033 and APP:1060 only, so only the PWML step uses it. | PWML step |
| 513-534 | `initialize_refinement_review_state`. Used at APP:3036 and APP:3062 only, so only the handler uses it. | handler |
| 482-502 | `reset_quarantine_state` is also called at APP:1924. Stays. | |
| 599-668 | `_render_stage_contract_failure` is used at APP:3065 and APP:3388 (legacy). `_render_issue_detail` (671) is used at 658, 2372, 3054 and 3204. Both are shared, and tests load them from APP. | shared |

## 2. `session_state` in order
**Handler:**
| line | op | key |
|---|---|---|
| 2881 | pop | `post_pipeline_artifacts` |
| 2890 | pop x5 plus disk clear | `quarantine_report`, `quarantine_artifacts`, `quarantine_coverage`, `quarantine_ok`, `quarantine_decision_controls` (APP:473-479, 495-499) |
| 2892 | get | `extraction_diagnostics_recorder` |
| 2897, 2934 | get | `pathway_context` |
| 2921 / 2922 / 2925 / 2929 / 2933 | get | `qa_report` / `reaction_summary` / `rag_evidence_context` (only if `rag_config()["enabled"]`) / `export_mode` / `pwml_strict_db` (default True) |
| 2936 | `state=st.session_state` handed to the callee, which writes the 5 quarantine keys (`postpipeline.py:150-154`). Other writes: unverified. | |
| 2938 | set | `post_pipeline_artifacts` |
| 2941, 2943 | set (conditional) | `qa_report`, `reaction_summary` |
| 2944 | get (default `final_payload`) | `final_payload_snapshot` |
| 2976 | set | `refinement_gate_errors` (attribute style) |
| 2996, 3036, 3062 | get `export_mode`; `initialize_refinement_review_state` sets 13 keys: `refinement_working_json`, `_graph_dict`, `_graph_bytes`, `_qa_report`, `_mapping_report`, `_mapping_misses`, `_history`, `_round`, `_pwml_ready`, `_last_error`, `_last_warnings`, `_gate_errors`, `_checkpoints` (APP:522-534) | |
| 3013, 3038, 3061 | set | `refinement_gate_errors` |
| 3082 (finally) | `_sync_extraction_diagnostics` reads and writes the extraction-diagnostics keys (service version, `state=`). Exact keys: unverified. | |

**PWML step (reads, in order):** `refinement_working_json` (1004), `refinement_gate_errors` (1016), `post_pipeline_artifacts` (992), `pwml_grounding` (1041), `pwml_grounding_dict` (1042), `pwml_name`, `pwml_description`, `pwml_subject`, `pwml_width`, `pwml_height`, `pwml_bg`, `pwml_strict_db` (1046-1055), `pathway_context` (1058), `export_mode` (1059), `quarantine_report` and `quarantine_artifacts` (1033, 1060).
**PWML step (writes):** `post_pipeline_artifacts` (998, only if it is a dict; also mutates its `final_mapped_db`, `final_export_input` and `reviewed_final_mapped_path`), `refinement_pwml_ready` (1037, attribute style), `pwml_export_result` (1347, in the button).

**Read after the handler:**
- `post_pipeline_artifacts` at 3090 and 3341.
- Local `post_artifacts` at 3092. This is `{}` after a click (APP:2891), so the refinement cache path never comes from the fresh run on the click pass. **Preserve this.**
- Local `final_payload` at 3343. It is rebound at 2895 by `propagate_context_organism`. The `final_payload` session key is **not** rewritten.
- `pwml_export_result` at 3114.
- The refinement keys are read by `_render_review_refine_section` at 3106.

## 3. Stops and exits
**Handler:** no `st.stop()`, `return` or `raise`. The paths are:
- 2878: not a non-empty dict. Shows `st.error("No pipeline output in session state. Run the pipeline first.")` and skips everything.
- 2945: `gate_failed`. Shows `st.warning("Post-pipeline stopped at hard-gate validation. Review gate_fail_report.json.")`.
- 2972 quarantine refused. Shows `st.error("Quarantine could not produce a viable requested-pathway core: ...")`, then `st.caption("Requested-core coverage")`, `st.json(...)`, `st.caption("Strict invariants")`, `st.json(...)`. Review is not opened.
- 2996 research mode, Stage 3 failed. Shows `st.warning("Research mode: the pre-export Stage 3 revalidation FAILED ...")`, `st.caption`, a markdown list, `st.json`. Review is opened (2036).
- 3037 strict, Stage 3 failed. Shows `st.error("Mapped pathway still fails the pre-export Stage 3 revalidation. Refinement review was not opened.")` and an expander "Why Stage 3 failed (N error(s))" (`expanded=True`).
- 3060 pass. Calls `initialize_refinement_review_state` and shows `st.info("Mapped pathway is ready for review. PWML generation is paused until approval.")`.
- 3064 `except StageContractError`. Calls `_render_stage_contract_failure(..., operation_label="Audit and DB mapping", download_key="dl_audit_mapping_stage_contract_error")`. It does **not** call `_record_mapped_failure_snapshot`.
- 3070 `except Exception`. Calls `_record_mapped_failure_snapshot(final_payload, stage="post_pipeline", reason=...)`, then `st.error(f"Post-pipeline conversion failed: {exc}")` and `st.exception(exc)`. `final_payload` here is the **rebound** value if `propagate_context_organism` succeeded.
- 3078: `finally: _sync_extraction_diagnostics()`.

**PWML step:**
- 1005: no working JSON. Returns a failure dict, "No reviewed mapped pathway is available for PWML export.".
- 1017: gate errors present. Returns a failure dict, "PWML export stopped by pre-export Stage 3 revalidation. ...", with `**quarantine_session_state()`.
- 1337: the button catches `Exception` and builds a failure dict.
- 1348-1351: shows `st.success(f"PWML generated from reviewed JSON: ...")` or `st.error(f"PWML export failed: ...")`. The batch driver and the golden strings depend on this text (`test_batch_driver_seam_golden.py:294`).

## 4. Streamlit calls
- **Handler:** `st.button` (the trigger, stays in the app), `spinner`, `error`, `warning`, `info`, `json`, `caption`, `markdown`, `expander`, `write` (via `_render_stage_contract_failure` and `_render_issue_detail`), `exception`, `download_button` (via `_render_stage_contract_failure`).
- **PWML step:** `st.spinner`, `st.success`, `st.error`.
- **`Reporter` has** `info`, `warning`, `error`, `json`, `progress`, `success`, `caption`, `text`, `spinner`, `expander`, `markdown`, `code` (`service/reporter.py:39-64`).
- **Missing:** `exception` and `write`. `download_button` is also missing, but it only occurs inside `_render_stage_contract_failure`. If that function stays in the app, the service returns the failure and the app renders it, as R3 did. The `Reporter` Protocol and both implementations need two small additions, or the handler's `st.exception` and `st.write` stay in an app-side wrapper.

## 5. Callees
- `reset_quarantine_state`, `initialize_refinement_review_state`, `_render_stage_contract_failure`, `_render_issue_detail`, `_sync_extraction_diagnostics` (wraps `service.extraction._sync_extraction_diagnostics`, APP:594): APP.
- `reenter` (as `reenter_extraction_diagnostics`): `pipeline.extraction_diagnostics:928`.
- `propagate_context_organism`: `pipeline.pipeline:2434`.
- `run_post_pipeline_sbml_artifacts`, `_record_mapped_failure_snapshot`, `_safe_dict`, `_safe_list`, `run_pwml_export`: `service.postpipeline`.
- `rag_config`: `t2pw.config`.
- `coerce_mode`, `is_research`: `pipeline.export_mode`.
- `CANONICAL_PAYLOAD_KEY`, `FINAL_GATE_REPORT_KEY`: `pipeline.gate_reports`.
- `StageContractError`: `pipeline.stage_contracts`.
- `build_draft_graph`, `render_draft_graph_to_png_bytes`, `generate_qa_report` (inside `initialize_refinement_review_state`): `pipeline.draft_graph`, `pipeline.draft_graph_render`, `pipeline.qa_graph`.
- `PROJECT_ROOT`: `t2pw.paths`.
- Widget locals the handler reads as bare names (about 21): `use_llm_audit`, `use_sbml_overwatch`, `default_compartment`, `mapping_cache_text`, `id_source_mode`, `db_host`, `db_port`, `db_user`, `db_password`, `db_schema`, `audit_max_rounds`, `audit_timeout_seconds`, `audit_candidate_count`, `use_example_retrieval`, `example_index_path`, `example_top_k`, `use_gap_resolver`, `use_llm_gap_resolver`, `gap_resolver_max_items`, `use_stoich_agent`, plus `final_payload` (APP:1935). These are the request fields (APP:2137-2260).

## 6. Test coupling
**Reads APP source:**
- `test_streamlit_stage2_orchestration.py::test_stage2_artifacts_are_visible_and_both_ui_paths_share_the_orchestrator` (:1027-1119), AST over APP:
  - exactly 2 Name calls to `run_post_pipeline_sbml_artifacts` with `build_legacy_sbml` in `{False, True}` (:1055-1063). **Breaks if the handler call moves.**
  - exactly 2 `ast.Try` nodes whose body holds that call, with handler[0] `StageContractError` calling `_render_stage_contract_failure`, and handler[1] `Exception` (:1065-1089). **Breaks.**
  - the `If` whose test is `st.button("Run audit and DB mapping")` has a `.pop("post_pipeline_artifacts")` inside (:1091-1110). **Breaks** if the button branch is gutted.
  - `"Mapped pathway still fails the pre-export Stage 3 revalidation."` is a substring of APP source (:1113). **Breaks** if that text moves. It also expects `Download stage2.mapped.json` and similar strings in APP (:1114-1119, the artifact viewer, not moving).
- `test_streamlit_stage2_orchestration.py:861,862,944` `_load_function("_render_stage_contract_failure" / "_render_issue_detail")` from APP. Fine while they stay in APP. `_load_function` at :79-99 also reads service ASTs. `:1029` loads `_json_artifact_entries`.
- **Golden:** `test_streamlit_quarantine_boundary.py:1464-1476` needs at least 2 `reset_quarantine_state(<args>)` Name calls in APP (see Headline). At :778 the app AND `service/*.py` are scanned and callers of `quarantine_and_close` must equal `{run_quarantine_boundary, run_pwml_export}`. So a new service function must not call it directly. At :729 `run_pwml_export` is read from `postpipeline.py` (unchanged).
- The other golden tests read **service** source (`SERVICE_PATH`/`SERVICE_REL`): `test_c011_freeze_seam_golden_equivalence.py` (:98, APP only via `git show` at base refs), `test_streamlit_stage8_export_contract.py` (:21, :200), `test_c052_...` (:299, :699 service). `test_batch_driver_seam_golden.py` has no APP reads. It drives the real app through AppTest and the driver, with message strings pinned (:294-296). So the **message text must be identical**.

**Patches:**
- No test patches `run_post_pipeline_sbml_artifacts`, `run_pwml_export`, `propagate_context_organism`, `_record_mapped_failure_snapshot`, `initialize_refinement_review_state`, `reenter` or `quarantine_session_state` on `streamlit_app` (grep: no hit).
- The harness spies on the SOURCE module: `tests/equivalence/harness.py:93-143` wraps `postpipeline.run_post_pipeline_sbml_artifacts` and `postpipeline.run_pwml_export` and rebinds **every module** holding the name. The new service functions must look the callee up **at call time** on `postpipeline`, or via a module-level name that `install_spies` can reach. The harness then also sees the spy in the new module.
- Source-module patches that reach callees through the unchanged `postpipeline`: `test_streamlit_quarantine_boundary.py:344-373,1135-1138` (`run_enrichment`, `map_payload`) and `test_orch759b_recorder_reattach.py:168-182` (`PROJECT_ROOT`, `run_enrichment`). These are patches on `postpipeline` or `enrich_entities` and are unaffected.
- Tests with copy-pasted APP-like scripts that click the two buttons: `test_batch_driver.py`, `test_batch_research_gate_fail_open.py`, `test_batch_driver_quarantine_artifacts.py`, `test_c119_*`, `test_c126_*`, `test_c119_superseded...`, `test_stage3_gate_report_lifecycle.py`. They carry their own scripts and do not read the APP.
- AppTest on the real app clicks `pwml_generate_btn`: `test_orch759b`, `test_streamlit_quarantine_boundary`, `test_c082_post_pipeline_seam.py:99`, `test_c077`. They depend on behaviour only.

## 7. Hazards
1. **Rebinding:** `final_payload` is rebound at 2895 inside the spinner. The failure snapshot at 3071 and the `final_payload_snapshot` default at 2944 read the rebound value. If `propagate_context_organism` itself raises, the pre-rebind value is used. The session key `final_payload` is not updated. The service must return the rebound payload, or the app must keep the viewer fallback at 3343 working.
2. **Order:** pop `post_pipeline_artifacts`, then `reset_quarantine_state` (which also clears `outputs/`), then `post_artifacts = {}`, then `reenter`, then the try. `reset_quarantine_state` is cleared BEFORE the try on purpose (comment at 2882-2889). `reenter` is outside the try, so an exception there is not caught.
3. **Aliasing:** `_pa = _safe_dict(artifacts)` is the same object as `st.session_state["post_pipeline_artifacts"]` (`_safe_dict` returns the same dict, `postpipeline.py:212-213`). The later writes to `_pa[...]` (2944, 3011, 3012) mutate the stored artifacts after storage. Preserve this.
4. **`finally` ordering:** `_sync_extraction_diagnostics` runs on pass, `StageContractError` and `Exception`, after the UI messages. In the service this stays in a `finally`.
5. **Research branch:** at 3013 `refinement_gate_errors` is set to `[]` before `initialize_refinement_review_state(...)`, which sets it to `[]` again at :533. The refusal branch sets it to a quarantine list. The Stage-3 branch sets it to the gate errors. The order against `_pa[...]` is observable. The message text contains an em dash and a non-ASCII "—" in the markdown (`f"- \`{path}\` — ..."`).
6. **Defaults and coercions:** `(default_compartment or "cell").strip() or "cell"`, `mapping_cache_text.strip() or "id_mapping_cache.json"`, `(id_source_mode or "hybrid").strip().lower()`, `int(db_port)`, `bool(...)` and others at 2899-2937, plus `rag_config()["enabled"]` read at call time (:2926). Keep them in one place. `st.session_state.get("pwml_strict_db", True)` appears at 2933 and 1055. The PWML step has `or`-fallbacks for name, subject, width (3200), height (1400), bg.
7. **PWML step mutates the artifacts:** `_write_reviewed_payload_snapshot` writes `final.mapped.json` into `work_dir` (`PROJECT_ROOT/"tmp"`, APP:3108) and mutates `post_pipeline_artifacts`. `refinement_pwml_ready = True` is set before the export, so it stays True on an export failure. The quarantine decision is read from session at call time, not the pipeline's return value.
8. **Attribute-style writes:** `st.session_state.refinement_gate_errors = ...` and `.refinement_pwml_ready`. Item assignment on a `dict` state is needed. `initialize_refinement_review_state` uses attribute style on 13 keys, so it moves with attribute-to-item conversion, or it stays in the app and is passed in as a callback (this keeps AppTest identical).
9. **Freeze and quarantine sequencing is inside the callee:** the boundary and the Stage 3 gate run inside `run_post_pipeline_sbml_artifacts` (comment 2948-2957). The handler only presents their verdicts. So moving the handler does not change freeze ordering as long as the callee call and `state=` are unchanged.
10. **Tests that pin the app text:** the "Mapped pathway still fails ..." string (:1113) must remain in APP. The artifact-count and Try-node tests above.

## 8. Suggested split
- **R4a: PWML export step (about 90 lines out of the app).** Move `_pwml_reference_path`, `_write_reviewed_payload_snapshot`, `_generate_pwml_from_refinement_working_json` and `quarantine_session_state` to `service/export.py` as `run_export(*, state, reporter, work_dir, project_root)`. It returns the result dict, with no stops. The app keeps the button, try/except, `pwml_export_result` write and success/error text at 1333-1351 (or passes the reporter). It needs no test repoints (no test reads these functions from APP; `streamlit_app._generate_pwml...` is used by AppTest only). It uses call-time `postpipeline.run_pwml_export` so the harness spy sees it. Also delete the dead locals at 2834-2839 (or leave; behaviour-neutral).
- **R4b: handler (about 200 lines).** `run_post_pipeline(request, *, state, reporter)` takes the 21 widget values as a request dataclass (as `ExtractionRequest`), plus `final_payload`, and returns the rebound payload plus a failure object. Keep in the app: the button, the `reset_quarantine_state(PROJECT_ROOT/"outputs")` call at 2890 (golden), the post-block render, and `_render_stage_contract_failure` (shared). **Needs PO rulings before it can merge:**
  - `test_streamlit_stage2_orchestration.py:1055-1110` must be repointed. This is a test edit like Q-R3d-1, so the reviewer must confirm it is only a repoint.
  - The "Mapped pathway still fails" string must stay in the app or the test must read the service.
  - Add `exception` and `write` (and optionally `download_button`) to `Reporter`, `StreamlitReporter` and `LoggingReporter`.
  - The `StageContractError` rendering stays app-side: the service returns the exception.
  - Two sub-steps are possible: R4b1, pure presentation of the verdict (about 100 lines of lines 2958-3063, `present_post_pipeline_verdict(artifacts, ...)`); R4b2, the call and try/except/finally.
- **R4c: harness** (below).

## 9. Equivalence harness changes
- **Service arm:** `harness.py:338-373` currently inlines the handler's three steps for `propagate_context_organism`, the kwargs and the `run_post_pipeline_sbml_artifacts` call, and builds the `run_pwml_export` call by hand (:362-369). Once `run_post_pipeline` and `run_export` exist, replace these with `run_post_pipeline(request, state=state, reporter=LoggingReporter())` and `run_export(...)`. The `post_kwargs`/`export_kwargs` merging (which comes from `app["post_call"]` and `app["export_call"]`) must still reach the service functions' call into `postpipeline`. `install_spies` (`harness.py:128-143`) then spies the same two functions, still on `postpipeline`, where they are called by the new service module.
- The `real_function(...)` unwrapping (:350, :367) must go through the new functions or keep calling the callee directly. Decide before R4b. Otherwise the arm's calls are recorded twice (spy `arm["post_call"]` set by the first call only, `harness.py:100-110`, so it is probably harmless, but unverified).
- The harness must also stop reading the app's `post_call` args to build the service request: it reads the widget values. Today the post kwargs come from `app["post_call"]` (`harness.py` `post_args`/`post_kwargs`, :330-345 region).
- **F2 and F5:** I could not find the review text defining F2 and F5 (no match in `docs/webserver/results/R3d.md` or LEDGER). So these are **inferred**.
  - F2, inferred: `test_equivalence_app_vs_service.py:207` uses `service.get("unused", [])`, which silently hides a missing key. Use `service["unused"]`. The harness always sets the key (`harness.py:301,382`).
  - F5, inferred: stale wording at `test_equivalence_app_vs_service.py:13` (says the service post-pipeline is "called directly on the Stage-2 payload the app reached"; since R3d the arm starts from the text), and at `:195` ("The Stage-2 payload the service started from is the one the app's session held"). At `extraction.py:923` the comment says "so C-060a has to read this file instead of streamlit_app.py", but the same comment block at `extraction.py:919` still says "OWNED_FLOOR = 4640" while the floor is now 790 (R3d.md:310) and the tests point at `extraction.py`. Confirm with the R3d reviewer text before editing.
- Harness changes are test-only edits. Per the merge gates they need the reviewer's confirmation that they only change the arm, as in R3d (`tests/equivalence/harness.py` +3/-1 there).

## Could not determine
- The exact keys `run_post_pipeline_sbml_artifacts` and `_sync_extraction_diagnostics` write into `state` beyond the 5 quarantine keys (not read in full).
- The body of `_render_issue_detail` (APP:671-688), which was not read.
- Whether recording the spy calls twice would matter in the harness (unverified).
- The wording of the R3d reviewer's F2 and F5 (see above).
