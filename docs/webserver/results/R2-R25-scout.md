# Scout map for R2 and R2.5

- **Agent:** ws-scout (sonnet/medium), read-only, run 2026-11-05.
- **Tree:** `C:/t/ws-R1-tip2` at `c626a9b0`, which is the R1 tip that is about to merge. Line numbers are relative to that tree. APP is `src/t2pw/app/streamlit_app.py`. PP is `src/t2pw/service/postpipeline.py`.
- Nothing in this map was executed. Rows marked *unverified* are the scout's inference.

## Headline

1. Across the three R2 functions, only 3 `st.*` call sites and 5 `session_state` writes need to be abstracted.
2. `maybe_run_rag` never touches `session_state`. Its UI is one `st.status` container (APP:471) plus one `st.warning` (APP:531).
3. `run_rag_rounds` makes no `st` or `session_state` calls. It returns `messages` as `(kind, body)` tuples, and `emit_rag_loop_messages` (APP:1530) renders them with `st.info/warning/error/json`.
4. `run_quarantine_boundary` (PP:97-163) only writes, five keys in this order: `quarantine_report` (146), `quarantine_artifacts` (147), `quarantine_coverage` (148), `quarantine_ok` (149), `quarantine_decision_controls` (150). It binds `import streamlit as st` at PP:12.
5. `build_rag_round_stages` (APP:1032) saves these five keys and restores them in a `finally` block (1148-1162; `QUARANTINE_SESSION_KEYS` at APP:~929). Passing an explicit state dict makes this trivial.
6. **No LLM replay fixture exists.** The batch driver uses the live LLM (`driver.py:2339`, `AppTest.from_file` at `:2756`). Individual tests stub `chat_detailed` and similar functions one at a time.

## R2: callers and tests

| target | callers |
|---|---|
| `maybe_run_rag` (APP:435-~860) | APP:1375 (inside `run_rag_rounds`), APP:3482 (main flow, then `st.session_state["rag_result"]` at 3489) |
| `run_rag_rounds` (APP:1263-1529) | APP:3602 |
| `run_quarantine_boundary` (PP:97-163) | APP:1150 (`build_rag_round_stages`), PP:411 (`freeze_canonical_payload`) |

Tests that touch these functions. **Golden** means a merge-gate golden-seam test.

| test | what it does |
|---|---|
| `test_c055_rag_loop_wiring.py` | calls `app_module.run_rag_rounds`; patches `app_module.maybe_run_rag` (464); reads the AST of `maybe_run_rag`, `run_rag_rounds` and `build_rag_round_stages` and forbids `while` (503-511); stubs `streamlit` and re-imports PP (245-252) |
| `test_rag_triage_orchestration.py:196,217,238`, `test_rag_admission_production_path.py:341` | call `app.maybe_run_rag` |
| `test_c060a_requested_pathway_wiring.py:341-345` | reads the AST of `maybe_run_rag` and `_safe_dict` |
| **golden** `test_c011_freeze_seam_golden_equivalence.py:140` | exec-extracts `run_quarantine_boundary` from source at git refs |
| `test_c030a_object_sharing_at_the_freeze_seam.py:267`, `test_streamlit_stage2_orchestration.py:470` | inject a stub `run_quarantine_boundary` into the namespace or globals |
| **golden** `test_streamlit_quarantine_boundary.py:788` | asserts the callers of `quarantine_and_close` are exactly {`run_quarantine_boundary`, `run_pwml_export`}, across APP and PP since R1 F1 |

Functions that are still in APP and only serve the RAG-round code: `_post_extraction_new_error_keys` (239), `rag_loop_max_rounds`/`rag_loop_deadline_seconds` (936/947), `rag_round_scratch_dir` (968), `rag_round_mapping_cache_path` (979), `rag_round_reports` (1016), `build_rag_round_stages` (1032), `rag_round_merge` (1176), `_rag_round_claims` (1222), `_rag_round_gap_ids` (1235) and `_rag_reentry_map` (1245). Three helpers are shared with code that stays in the app: `RagOrchestrationResult` (193), `_count_reactions` (206), and `_safe_dict` (PP:208).

Streamlit behaviour a Reporter must express:

- `st.status` with live label updates and a final state of complete/error (wrapped in a try/except).
- `st.warning`, called partway through a function.
- Deferred `(kind, body)` messages, including a **json** kind.

`st.stop`, `st.rerun`, `st.cache`, spinners and widgets do **not** appear in these three functions; they belong to the R3 handler.

## R2.5: equivalence harness

- **LLM boundary.** All calls go through `src/t2pw/llm/client.py`: `chat` (634), `chat_detailed` (666), `chat_with_tools` (900), and the module `_client`. The switch between OpenRouter and a local model is at 71-105. But importers bind the names directly (`pipeline.py:9`, and also `localized_repair`, `preprocessor`, the curator and `gap_resolver`), so a recorder has to patch either `client._client` (one place) or every importer's name.
- **Nearest existing model.** `tests/test_orch768_replay_corpus.py` with `tests/fixtures/orch768/` (saved legs, gzipped) runs `freeze_canonical_payload` → `run_pwml_export` offline, with sockets blocked. It does not go through AppTest. `T2PW_OFFLINE_CURATOR` (`curation/pathway_curator.py:62`) covers only the curator. Other replay tests exist (`test_rag_cached_replay`, `test_early_failure_replay`, `test_strict_failure_replay`), but their contents are *unverified*.
- **Hash and bytes.** `canonical_payload_sha256` is at `pipeline/canonical_hash.py:240` (`HASH_SCHEMA_VERSION=2`) and is computed at PP:428. The PWML is written at PP:2144 to the **fixed** path `outputs/pathway.pwml`, which every run shares. So the harness needs separate project roots, or must copy the bytes out between runs.
- **Nondeterminism.**
  - A uuid4 temp dir at PP:590 (removed after use).
  - `time.time()` timings at PP:920-1331, which appear in the reports. Whether they also reach the PWML is *unverified*.
  - No datetime or uuid in `pwml/`, per grep; the writer itself was not read.
  - Stage 1 gives different results across runs even at temperature 0, so a recorded fixture is mandatory.
  - Also needed: `T2PW_OFFLINE_CURATOR=1`, `T2PW_SPECIES_LLM=0`, `T2PW_SPECIES_NCBI=0` (as in `test_c095:202`).
- **Shortest app path.** `driver.run_one` (`driver.py:2339`) uses AppTest and ends at PWML export with a live LLM. 17 test files use AppTest. Which of them reaches export with a stubbed LLM, and how long that takes, is *unverified*.

## Risks (scout)

**R2**
1. Moving `maybe_run_rag` or `run_rag_rounds` breaks the AST and exec readers above (c055:503-511, c060a:341, golden c011:140). Repointing them is allowed only as a path change under Q-R0-1.
2. PP binds `st` at import. The state-dict parameter must default to something that keeps `st.session_state` on the app path, so that the stubbed-streamlit tests (c055:245) still see the writes.
3. Keep the order and labels of `st.status` updates exactly as they are, or the `test_rag_*` fakes may drift.

**R2.5**
1. There is no replay mechanism, so a recorder is new code that needs several patch points or the single `client._client` point.
2. Both runs write the fixed `outputs/pathway.pwml` and `tmp/`. They must be isolated per run.
3. The Stage 0-2 handler stays inline in the app until R3. Before R3, the harness can prove the *post-pipeline* half equal (from a captured Stage-2 payload through to the PWML), and that the app is deterministic under the fixture. Full app-vs-`run_pathway` equality only becomes testable from R3 onwards.
