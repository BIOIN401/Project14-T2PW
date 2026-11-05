# Scout map for R3

- **Agent:** ws-scout (sonnet/medium), read-only, 2026-11-05. Nothing was executed.
- **Tree:** `C:/t/ws-int` @ `82e0267a`: R1, R2, R2-F2 and BR merged; R2.5 not merged.
- APP = `src/t2pw/app/streamlit_app.py` (4085 lines).
- Line numbers refer to that tree. Items marked *unverified* are the scout's own inference.

## Headline

1. PLAN's "~5284-5727" range is stale. The R3 target is `if submit:` at APP:2083-2527, about 445 lines.
   - It is inline script code. `submit` is the return value of `st.form_submit_button("Run pipeline")` at APP:2081.
2. The block contains **no** audit, DB mapping, post-pipeline or export step. Those run from separate buttons after `if pipeline_ready:` (APP:2529), so they belong to R4.
   - R3 ends at `reset_quarantine_state` / `token_stats` (APP:2526-2527).
3. Streamlit coupling in the block:
   - 4 `st.spinner`, 8 `st.stop`, and about 29 other `st.*` display calls.
   - 35 `session_state` lines.
   - No `st.rerun` or `st.cache*`.
4. About 11 widget values are read as bare locals. These are the real inputs.
5. The largest hazard is `test_c060a_requested_pathway_wiring.py`. It lifts both `merge_additions` statements out of APP by AST, so moving either one breaks it.

## 1. Target flow

| step | APP lines |
|---|---|
| PDF parse | 2088-2157 |
| Text assembly | 2159-2162 |
| `reset_refinement_state` | 2164 |
| `_start_extraction_diagnostics` | 2170 |
| Stage 0 via `_run_stage_zero_with_retry` | 2180 |
| Usable-context, seeding and recovered branches | 2187-2225 |
| Ambiguous-review early exit | 2229-2253 |
| Stage 1: `run_stage_one_with_chunking`, `settle_stage_one` | 2258, 2277 |
| Stage 1 failure handlers | 2284-2313 |
| `filter_out_of_scope_reactions` | 2327 |
| `maybe_run_rag` (`user_task_context` folded at 2361-2366) | 2350 |
| Stage 2: `run_stage_two_with_feedback_loop` | 2378 |
| `merge_additions` (Stage 2) | 2408 |
| `_rag_merge_site` closure | 2456 |
| `run_rag_rounds` | 2471 |
| `build_and_save_draft_graph` | 2497 |
| `session_state` writes | 2498-2527 |
| `reset_quarantine_state(PROJECT_ROOT/"outputs")` | 2526 |

Callees that are still in APP:

- `_run_stage_zero_with_retry` (316), `_has_usable_context` (223), `_seed_context_from_user_scope` (243)
- `_start_extraction_diagnostics` (801), `_sync_extraction_diagnostics` (829), `_last_cleaning_report` (848)
- `reset_refinement_state` (645), `reset_quarantine_state` (660)
- `render_attempts` (720), `emit_rag_loop_messages` (397), `rag_loop_session_record` (412)
- `_write_reaction_preservation_report` and `build_and_save_draft_graph`: not located (*unverified*)

## 2. Coupling

**Progress (goes to the Reporter):**

- Spinners: 2117, 2173, 2257, 2377. The Reporter must keep these stage boundaries.
- Info, warning and success messages:
  - PDF: 2107, 2132-2157
  - Stage 0: 2198, 2207, 2220
  - Others: 2315, 2329
- Stage failures: errors at 2286, 2291, 2299, 2393; `render_attempts` at 2287, 2395; `st.json` at 2292, 2310, 2312.
- Ambiguous-review display: 2241-2252.
- `st.stop` at 2092, 2133, 2162, 2253, 2288, 2293, 2313, 2396. Each must become a returned failure at the same point, with a kind and a message.

**Inputs (bare locals):**

- `text_entries`, `uploaded_pdfs`, `pdf_page_range`, `pdf_skip_refs`, `pdf_ocr`
- `user_task_context`, `input_mode`
- `run_inference`, `enable_chunking`, `rag_incomplete_flag`
- `extract_attempts`, `infer_attempts`, `temperature`, `infer_rounds`
- `extract_tokens`, `infer_tokens`, `chunk_size`, `chunk_overlap`
- `export_mode`: a local set at 1882 and also `session_state["export_mode"]` at 1883. The block reads the session copy at 2279 and the local at 2478.
- `st.session_state.get("pwml_strict_db", True)` at 2479. The checkbox is defined below the block (3452), so the key is absent on the first run.

**Outputs (`session_state` writes):**

- Ambiguous path, 2231-2233: `pipeline_ready`, `pathway_context`, `pipeline_error`
- 2283: `stage_one_boundary`
- 2359: `rag_result`
- 2366: `rag_evidence_context`
- 2489: `rag_loop`
- 2498-2502: `draft_graph`, `qa_report`, `reaction_summary`; pops `draft_graph_png_bytes` and `draft_graph_render_error`
- 2504-2519:
  - `pipeline_ready`, `run_inference_enabled`
  - `pathway_context`, `user_task_context`
  - `stage_one`, `extraction_diagnostics_recorder`, `out_of_scope_removed_reactions`, `chunk_details`
  - `stage_two`, `stage_two_chunks`, `stage_two_rounds`, `qa_hints`
  - `final_payload`, `final_payload_snapshot`, `pipeline_source_text`, `reaction_preservation_after_stage2`
- 2520: pops `post_pipeline_artifacts`
- 2527: `token_stats`

Hidden writers:

- `_start_extraction_diagnostics` (diagnostics keys, about APP:824)
- `reset_refinement_state` and `reset_quarantine_state`
- `maybe_run_rag` and `run_rag_rounds` (`state=st.session_state`, 2481)
- The module-global extraction recorder (`current_extraction_diagnostics()`, 2509)

## 3. Field map

- **Already in `PathwayRequest`:** `texts`, `pdf_paths`, `scope` (name, organism), `export_mode`, `out_dir`, `deadline`.
- **No field yet:**
  - run_inference, enable_chunking, the rag flag
  - extract/infer attempts, infer_rounds, temperature, extract/infer tokens, chunk_size, chunk_overlap
  - user_task_context
  - the PDF options (page range, skip_refs, ocr)
  - strict_db

  The batch driver probably relies on the app defaults for all of these (*unverified*), which makes them R5 decisions. R3 can carry them on a separate `ExtractionSettings` dataclass.
- **No home in `PathwayResult`** for the extraction outputs: `final_payload`, `stage_one`, `stage_two*`, `pathway_context`, `chunk_details`, `qa_hints`, `rag_result`, `token_stats`, `draft_graph`, `qa_report`. PLAN names an `ExtractionResult` for them. It does not exist yet (*unverified*).

## 4. Tests

**Source and AST readers:**

- `test_c060a_requested_pathway_wiring.py:42,57,80` lifts the `merge_additions` sites at 2408 (Stage 2) and 2463 (RAG) by AST. It requires exactly 2 at or below `OWNED_FLOOR = 1440` and execs them against the APP path.
  - Moving them breaks the test. The simplest fix is to keep both statements in APP and pass them to the moved code as callbacks.
- The c055 AST tests read `maybe_run_rag`, `run_rag_rounds` and `build_rag_round_stages`, and forbid `while` loops. They stay green if those functions are untouched.
- In total, 62 test files mention `streamlit_app`; most of them were not read.

**Callers and patches:**

- `test_stage0_case_c_retry_clobber.py:250` calls `app._run_stage_zero_with_retry`. It needs an alias or import in APP.
- `test_c055:839-853` patches `run_stage_one_with_chunking`, `run_stage_two_with_feedback_loop` and `settle_stage_one` on their source modules. Keep those names importable from the same modules.

**AppTest users that drive submit:**

- `test_c055`, `test_c052`, `test_c082`, `test_c083`
- `test_orch759a_fold_app_path`, `test_orch759b_recorder_reattach`, `test_orch759d_app_timeout`
- `test_stage3_gate_report_lifecycle`, `test_prefreeze_third_export_seam`, `test_reaction_evidence_app_hook`
- `test_batch_driver`, `test_batch_preflight`, `test_service_driver_bridge`, `test_streamlit_quarantine_boundary`

**Golden-seam tests:** none of them reads the `if submit:` block (inferred from greps).

- c011 exec-extracts `run_quarantine_boundary` from postpipeline.
- stage8 reads postpipeline's AST for `run_pwml_export`.
- quarantine_boundary is AppTest-driven.
- batch_driver_seam: no APP read found (*unverified*).

**Equivalence harness** (`ws/R25-equivalence`, `tests/equivalence/harness.py`):

- The app arm (161) patches `AppTest.from_file` (176/196) and spies on `postpipeline.run_post_pipeline_sbml_artifacts` / `run_pwml_export` (177-178/197-198). It also sets up the LLM cassette and the frozen clock (69).
- `isolate` (125-134) patches `paths.PROJECT_ROOT` and `paths.TMP_DIR`, plus the copies held by other modules.
- The service arm (227-275) replays the recorded post-pipeline and export arguments. Its `TODO(R3)` at 231 says it should start from `case["full_text"]` through `run_extraction`.
- **So R3 must add the extraction half of the service arm** and keep `AppTest.from_file` and the postpipeline names unchanged.

## 5. Other callers

- `batch/driver.py` presses "Run pipeline" by label (`LABEL_RUN_PIPELINE`, driver.py:2808). It reads `pipeline_error` and `pathway_context` (464, 676) and drives `KEY_POST_PIPELINE` (2901) and `KEY_GENERATE_PWML` (3149).
  - The `session_state` keys written at 2504-2527 must therefore still appear. A thin publisher in APP can do this.
- `service/driver_bridge.py` goes through `driver.run_one` only.

## 6. Hazards

- **`install_app_defaults` (APP:194) is process-global.** Moved code must keep the same defaults or pass the reporter and state explicitly. R2 review F1 says the web worker must always pass them.
- **The extraction recorder is module-global** (`unique=True`).
- **Fixed paths:** `PROJECT_ROOT / "tmp"` (2262, 801-826) and `PROJECT_ROOT / "outputs"` (2526).
  - The harness isolates only through the `t2pw.paths` module attributes, so moved code must read `PROJECT_ROOT` the same way.
- **Behaviour traps:**
  - `st.stop()` ends the script. Moved code must return at the same points.
  - The two `export_mode` reads agree only because line 1883 sets both.
  - `user_task_context` is rebound by the RAG fold at 2361 before Stage 2 uses it. Keep that order.
  - `pwml_strict_db` is absent on the first run, so it defaults to True.
  - The `pathway_context_from_stage_zero` imports sit inside the block (2406, 2461).

## 7. Suggested split

Each step keeps the `session_state` keys byte-identical through a thin publisher in APP, and keeps `if submit:` as the caller. After each step, run c060a, c055, orch759d, the AppTest files, the golden tests and the equivalence harness.

| step | scope | size | note |
|---|---|---|---|
| R3a | Input assembly 2084-2162 | ~110 lines | `assemble_inputs`; warnings go through the Reporter; `st.stop` becomes a returned failure. Low risk |
| R3b | Stage 0 with the Case-C and seed fallbacks, 2164-2253, plus `_run_stage_zero_with_retry` and the diagnostics helpers | ~100 lines | Alias `app._run_stage_zero_with_retry` |
| R3c | Stage 1, the boundary, and the RAG seed, 2255-2366 | ~130 lines | Keep the merge callbacks in APP |
| R3d | Stage 2, both `merge_additions` sites as callables, `run_rag_rounds`, the draft graph and the publish, 2368-2527 | ~130 lines | Also the harness `run_extraction` arm (`TODO(R3)`) |
