# RT1 - per-call logging and a 12-minute cap on LLM requests in a website job

- **Branch:** `ws/RT1-llm-timing` from `307a06af`.
- **Status: implemented, awaiting independent review.** Not merged. Open questions in section 7.
- **Code commits:** `723b6acd` (part 1, logging) and `07ee8421` (part 2, the cap). This doc and the evidence are a separate docs-only commit after them.
- **Authority:** PO ruling Q-RT-1 (`PO-RULINGS-2026-11-04.md`, addendum 8): "yeah sure do that". Diagnosis: `RUNTIME-VARIANCE.md` sections 5.1 and 6.
- **New tests (new capability, labelled in their docstring):** `tests/test_llm_call_logging.py` (13), `tests/test_llm_call_cap.py` (28). No existing test was edited.
- **Files changed:** `src/t2pw/llm/client.py`, `src/t2pw/service/pathway.py`, the two test files, this doc, `evidence/RT1/`. `batch/leg_trace.py` is **not** changed (see 1.3).

## 1. What was built

### 1.1 One switch: the web-run scope

`llm/client.py` gets a module flag `_WEB_RUN` (False) and a context manager `web_run_scope()`. `service.pathway.run_pathway` (the web entry) wraps `run_paper` in it, and in a `finally` it is restored. Everything new below is gated on it, so the Streamlit app and a batch leg (which calls `run_paper` directly, never `run_pathway`) take exactly today's path. A process-wide flag rather than a `ContextVar`, so a call issued from a helper thread is also covered; the web worker runs one job per process, so no second job can see it. If one process ever served two jobs at once the flag would need to become per-job (Q-RT1-C).

### 1.2 Part 1, logging

| requirement | how |
|---|---|
| Trace active for a web job | `run_pathway` calls `leg_trace.activate(out_dir / "LEG_TRACE.jsonl")` after the refusal checks and before `run_paper`, records `job_begin` / `job_end` (status, stage, seconds; no text), and deactivates in `finally`. If a trace is already active (a batch leg, which does its own at `batch/runner.py:1151`) it is left alone: not replaced, not deactivated. `run_paper` is untouched, so batch cannot double-activate |
| Survives the job; not a download | It sits in the job's `out/` folder beside `run_outcome.json`. Downloads are resolved by key from the stored result's `pwml_path`, `image_path` and `report_paths` (`web/views.py:resolve_download`), and `report_paths` comes only from the artifacts the run produced. The trace is in none of them, so the web layer needed no change. Test: `test_run_pathway_writes_a_trace_that_is_not_a_download` |
| `started_at`, `duration_s` per attempt | `CompletionDiagnostics.begin_attempt()` stamps wall and monotonic clocks before each provider request; `note()` publishes them with the row. Failed attempts get a duration too |
| Per-call tokens, provider, generation id | `observe(resp)` reads `usage.prompt_tokens`, `completion_tokens`, `completion_tokens_details.reasoning_tokens`, `resp.provider` (also `model_extra`), `resp.id`, `resp.model`. Only plain ints and non-empty strs are kept; an odd response adds nothing |
| Curator calls visible | `chat_with_tools._call_once` builds a private `CompletionDiagnostics` (web run only) and notes ok / empty / content_filter / error / timeout per attempt, per round, under the caller's `stage_name` (the curator's is `"curator"`; the gap resolver's tool loop is `"gap resolver"`) |

Where `CompletionDiagnostics` data goes, checked before adding anything: `to_dict()` feeds `chat_detailed`'s callers, which pass `attempt_log` into `extraction_diagnostics.record_boundary` (the `extraction_boundary_report.json` rows) and the Stage 0 attempts file. So **none of the new data is added to a diagnostics field, `to_dict()` or an `attempt_log` row**. It lives in underscore-named scratch fields that `to_dict()` does not read, and goes to the trace only. `chat_with_tools` returns a plain string and keeps no diagnostics, so its private object is read by nothing but the trace. Tests: `test_result_and_diagnostics_are_identical_with_and_without_the_scope` (text and `to_dict()` equal, no `duration_s` / `started_at` / token key anywhere in it).

### 1.3 Trace row, and why `leg_trace.py` is untouched

`LegTrace.model_attempt` takes a fixed set of keyword arguments. Rather than widen it, a web run publishes the same row through `leg_trace.record_event("model_attempt", ...)` with the same keys plus the new ones, and `summarize()` / `read_events()` ignore the extras. A batch trace keeps `record_model_attempt` and gets exactly the old row (`test_a_batch_leg_trace_gets_exactly_the_old_row`). The leg trace redacts credential shapes from every string, and holds no prompt or reply text (the test asserts the reply and the paper text are absent).

### 1.4 Part 2, the cap

| requirement | how |
|---|---|
| Value in one named constant | `LLM_CALL_CAP_SECONDS = 720.0` (the longest good Stage-1 call measured was 11.97 min = 718 s). A test checks the literal appears once and exceeds 11.97 x 60 |
| Allowlist | `LLM_CALL_CAP_STAGES` = the `stage_name` strings the four measured stages pass to `chat_detailed`: **`preprocessor`** (Stage 0, `pipeline/preprocessor.py:165`), **`Stage 1 extraction`** and **`Stage 2 inference`** (`pipeline._run_json_stage` maps `extraction` / `inference` to these labels at `pipeline.py:3808`, and passes the label to `chat_detailed`), **`rag_prose_extraction`** (`rag/extract.py:144`). Not capped: `audit loop`, `final_completeness_audit`, `gap resolver`, `curator`, `overwatch`, `focused_repair`, `<stage> json repair`, `<stage> row repair`, the unnamed `chat` (alias lookups, stoich, interactive curator) |
| Mechanism | `_create(kwargs, stage)`. Not a web run, or a stage off the list: the plain `_client.chat.completions.create(**kwargs)` on the caller's thread, as before. Otherwise the request runs on a daemon thread (`t2pw-llm-call`) and the caller waits `Event.wait(cap)`. Streaming was not needed |
| A capped call is a failed attempt in the existing loop | `chat_detailed` gets one new `except _CallCapExceeded` clause: `last_err = e`, `diagnostics.note(status="call_cap")`, the same backoff sleep, and the loop continues. No new retry, backoff, budget or model choice. If every attempt is capped the call ends as `LLMOperationTimeout` with `terminal_reason="operation_timeout"` (the existing outcome for "every attempt timed out"), the message naming the cap |
| Distinct status in the trace | `call_cap`, with the reason text `<stage> request still running after the 720s wall-clock cap; abandoned` and the real `duration_s` |
| Late result discarded | The worker only makes the request and puts the answer in a box local to that one call. `_record_usage`, `diagnostics.observe`, the spacing sleep and every other effect happen on the caller's thread, on a response the caller actually received. An abandoned worker therefore cannot write token totals, diagnostics, the trace or anything shared. Test: `test_a_late_result_is_discarded_and_touches_no_totals_or_records` releases the held request (it answers with 1000 / 2000 tokens) and checks the token totals, `to_dict()` and the trace are byte-identical before and after |
| Process exit | The worker is a daemon. Test `..._worker_is_a_daemon...` |
| Context that does not cross threads | The worker runs only `create(**request)`, which reads no `current_diagnostics()`, deadline or scope state (those are read on the caller's thread before and after). `request` is a copy of the kwargs, which the caller edits between attempts |
| Off for batch and the app | `_WEB_RUN` is set only by `run_pathway`. `run_paper` and the app never set it (`test_run_pathway_caps_but_run_paper_does_not`, `test_without_the_web_scope_a_slow_allowlisted_call_is_never_capped`) |
| `chat_with_tools` not capped | It does not call `_create` at all. Its curator stage is not measured, and the PO's ruling excludes it |

What the abandoned request does afterwards: it is still an open HTTP request until the SDK's own timeouts (300 s read, no keep-alive reset) or the server end it. It holds a connection and a thread, not memory the run needs. At most a handful can exist per job (8 attempts x 3 stages x a few calls), and the provider may still bill the abandoned generation (Q-RT1-B).

## 2. Requirements checklist

| requirement | where |
|---|---|
| Trace active for a web job, not double-activated | 1.2; `test_run_pathway_leaves_a_batch_trace_that_is_already_active_alone` |
| Trace not a public download, `web/` untouched | 1.2; `git diff 307a06af --stat` shows no `web/` file |
| `started_at`, `duration_s`, tokens, provider, generation id | 1.2; `test_a_web_call_records_...`, `..._retried_call_has_its_own_timing`, `..._failed_request_still_gets_a_duration` |
| Curator visible | `test_curator_rounds_are_visible_in_a_web_trace`, `..._retries_and_empty_replies_are_visible`, `..._silent_outside_a_web_job` |
| No change to any report, result or verdict | to_dict and attempt_log unchanged; the four golden seams, three equivalence tests and SMOKE pass unedited (section 3) |
| Cap: past-cap call = failed attempt, existing retry | `test_a_request_past_the_cap_is_a_failed_attempt_and_the_existing_retry_follows` (2 provider calls, statuses `call_cap`, `ok`); `test_every_attempt_capped_ends_as_the_distinct_operation_timeout` (exactly `LLM_MAX_RETRIES` calls) |
| Allowlist only | `test_the_allowlist_is_exactly_the_four_measured_stages`, a parametrised "is capped" test x 4 and "is not capped, runs on the caller's thread" x 9 |
| Off for `run_paper` and the app | 1.4 |
| Normal call unchanged | `test_a_call_that_finishes_normally_is_identical_with_the_cap_on_or_off` x 4 (text, `to_dict()`, token totals) |
| Tiny monkeypatched cap | `CAP = 0.15 s` |

## 3. Tests

Every run through the bounded wrapper (`evidence/RT1/scripts/run.sh`: heavy lock `RT1`, fresh `--basetemp` under `C:/t/bt/RT1/`, `PYTHONPATH` the worktree, `T2PW_OFFLINE_CURATOR=1`, `pinned_pytest.py --expect-tree`). Each report: **0 surviving processes**. Reports and outputs in `evidence/RT1/runs/`. Run on the tip of the code commits (`07ee8421`).

| run | result |
|---|---|
| `test_batch_driver_seam_golden.py` | 2 passed |
| `test_c011_freeze_seam_golden_equivalence.py` | 8 passed |
| `test_streamlit_stage8_export_contract.py` | 4 passed |
| `test_streamlit_quarantine_boundary.py` | 24 passed |
| `test_equivalence_app_vs_service.py` | 14 passed |
| `test_equivalence_second_fixture.py` | 8 passed |
| `test_equivalence_run_pathway.py` | 8 passed |
| new: `test_llm_call_logging.py` + `test_llm_call_cap.py` | **41 passed** (13 + 28) |
| `test_service_run_pathway.py` | **107 passed** (same as base) |
| `test_batch_driver_run_pathway.py` | **25 passed** (same as base) |
| `test_llm_client_empty_completion_retry.py` (alone) | 29 passed |
| `test_batch_preflight`, `test_c111_timeout_observability`, `test_llm_client_request_timeout` | 61 passed, 4 skipped, **1 failed**: `test_batch_preflight::test_the_message_names_the_problem_the_interpreter_and_the_cure` ("this project ships a .venv"; a worktree has none). Same failure as on R6's base (`results/R6.md` section 3), so not caused by this task |
| SMOKE (`smoke.lst`) | **511 passed** |

The equivalence tests matter here: `run_pathway` is now in the web scope, so Stage 0/1/2 requests in the replay run through the worker thread. All three pass unchanged, which shows the replay harness and the pipeline are indifferent to the thread.

## 4. Revert proofs

On a scratch copy of the tip at `C:/t/ev-RT1/rev/base` (never the worktree), each revert applied alone and restored, running both new test files (41 tests). Outputs in `evidence/RT1/revert/`, the script in `evidence/RT1/scripts/mutate.py`. **Every revert fails at least one test**:

| revert | failing tests |
|---|---|
| M01 web branch of `_publish_attempt` removed (no new fields) | 6 |
| M02 curator rounds not traced | 2 |
| M03 `run_pathway` does not activate the trace | 1 |
| M04 trace added to `report_paths` (a download) | 1 |
| M05 cap removed (every call plain) | 10 (and the run took 187 s: the held requests are no longer abandoned) |
| M06 abandoned worker records token usage | 5 (late-result test and the 4 normal-call tests) |
| M07 every stage capped | 9 |
| M08 cap not gated on the web scope | 2 (app/batch tests) |
| M09 a capped call does not retry (`break`) | 19 |
| M10 worker not a daemon | 2 |
| M11 scope not restored on exit | 8 |
| M12 scope on by default | 8 |
| M13 cap constant 600 instead of 720 | 1 |
| M14 `to_dict()` gains a `duration_s` key | 1 |
| M15 `run_paper` enters the scope | 7 |

One dead end: my first M12 (force the trace branch in `_publish_attempt` on) **survived**, because with the scope off no attempt clock is stamped, so the branch publishes the same row. It is an equivalent mutant, not a gap; the output is kept (`M12_batch_gets_web_fields.out`) and M12 was replaced by "scope on by default".

## 5. What failed and dead ends

- **A first draft broke an existing test** (`test_llm_client_empty_completion_retry.py`, 16 failures): that test replaces `client.time` with a sleep recorder, and my `begin_attempt` called `time.time()`. Fixed by binding the clocks at import (`_wall_clock`, `_mono_clock`) and by stamping only in a web run. The 29 tests pass untouched.
- **Pre-existing test-order dependence, not mine:** running `test_llm_client_empty_completion_retry.py` and `test_llm_client_request_timeout.py` in one process fails 3 of the latter on base `307a06af` too (the first file stubs `openai` with `RuntimeError` for every error class). Alone, each passes (29 and 6), and SMOKE passes 511. My new tests patch the error classes themselves so they hold in either order.
- I started one heredoc-to-stdin `python -` by mistake while creating the scripts; it hung and was stopped by task id (no process was killed by name). No other slip.
- Streaming was not tried: the worker thread did what was needed.

## 6. Biology and boundary check

No file under `pipeline/`, `pwml/`, `mapping/`, `rag/`, `curation/`, `llm/prompts/`, `web/`, `app/`, `data/` is touched. For any call that finishes under the cap nothing changes in the return value, diagnostics or token totals (tests above). The one behaviour change is deliberate and PO-ruled: in a web job, a Stage 0 / 1 / 2 / RAG request still running at 12 minutes is abandoned and retried through the existing loop instead of waited for.

## 7. Open questions (for the product owner)

- **Q-RT1-A. The cap is per request; the job can still run long.** A request that hits the cap is retried by the existing loop (up to `LLM_MAX_RETRIES` = 8, 1 to 20 s backoff), so a stage whose every request hangs could spend 8 x 12 = 96 min, past the 90-minute job cap (`RUN_TIMEOUT_S`). The cap removes the single 14 to 28 minute stalls the diagnosis found; it does not bound a run whose provider is down for an hour. Options: leave it (the job cap ends the run, no biology change), or cap the attempts for web jobs (an `LLM_MAX_RETRIES` change, which is a behaviour change you did not ask for). Cost of the second: fewer chances to recover in a bad hour.
- **Q-RT1-B. The abandoned request may still be billed.** The provider may finish and charge for a generation whose answer we discard. Not measurable here (no live call). The new `generation_id` in the trace lets the cost be checked on OpenRouter after a real job. If it matters, the fix is streaming with a real cancel, a larger change.
- **Q-RT1-C. A process-wide flag.** Fine for one job per worker process (the W3 rule, Q-W0-1). If a worker ever ran two jobs in one process, the flag and the process-global leg trace would have to become per-job.
- **Q-RT1-D. `call_cap` appears in the diagnostics `attempt_log` (so in `extraction_boundary_report.json` and Stage 0 attempts) when it fires.** That is the existing `status` field with a new value, only on the path that did not exist before, and consumers treat statuses as opaque text. I kept it so a retried capped call is explained in the same place as other failures. If you want those reports to show only the old vocabulary, it can be mapped to `timeout` there and kept distinct in the trace only.
- **Unverified:** no live OpenRouter call was made (no keys, no network by the brief). That a real OpenRouter response carries `provider` as a top-level field, and `completion_tokens_details.reasoning_tokens`, is from OpenRouter's documented shape and the SDK's types, tested against fakes only. If a field is absent the trace simply omits it. The first real job after merge should be read to confirm.

## 8. Independent test run, review and merge (orchestrator)

- **Merged** into `webserver/refactor` as `9ddeb374` (`--no-ff`). `src/` and `tests/` trees of the merge are identical to the tested `9afa3c14` (`c248a922`, `ecb4a124`). The only other difference is the two LEDGER lines already on the integration branch, so no separate post-merge run was made.
- **Independent run** on `9afa3c14` (`ws-test-runner`, haiku/medium), through the bounded wrapper, 0 survivors on all 15 runs, outputs in `evidence/RT1/independent/`: golden 2, 8, 4, 24; equivalence 14, 8, 8; new 41; client 29 and 6 (each alone); `run_pathway` 107; driver 25; batch set 201 passed, 4 skipped, 1 failed (the known preflight failure, same as base); web 306; SMOKE 511; census 83.
- **Review** on the actual diff (`ws-reviewer`, opus/medium): APPROVE WITH NOTES, no blocking finding, report in `evidence/RT1/review/`. The reviewer ran nothing and made no live call.
- **Review notes**
  - F1: the cap cannot tell a hung request from a long healthy one. The margin over the longest good call (718 s, from 38 legs) is 1.8 s. If generation time grows with paper size (not established), a slightly larger paper could hit the cap on every attempt and end as a timeout where it used to finish in about 13 minutes. Section 7 Q-RT1-A understates this by describing only a provider outage. Goes to the PO, read with the first real jobs.
  - F2: with `use_stoich_agent` on, `service/postpipeline.py:1417` and `stoich/agent.py` call `_client` directly and are absent from the trace. Fixing it is outside this task's files.
  - F3: `test_llm_call_cap.py:283` compares a literal with a literal, and line 190 compares the constant with a copy of itself. No test ties the four allowlist strings to the real call sites; a renamed label would silently switch the cap off. The reviewer confirmed the four strings match by reading.
  - F4: on ok and empty attempts `duration_s` includes the 0.35 s spacing sleep.
  - F5: an abandoned request can live as long as the provider keeps streaming, so section 1.4's "ends on the SDK's own timeouts" is too strong. Section 1.4 also says 3 stages; there are 4. Abandoned requests may count against provider concurrency as well as cost.
  - F6: "on all N attempts" in the final timeout message describes only the last error when attempts were mixed. The older timeout path has the same flaw.
- **Open questions, settled by the orchestrator on the reviewer's advice:** Q-RT1-B, check `generation_id` cost after the first real job and escalate only if material. Q-RT1-C, the one-job-per-process rule stands. Q-RT1-D, keep `call_cap`; the reviewer found no source reader that branches on attempt status.
- **For the product owner:** Q-RT1-A, read together with F1. The no-regression rule applies.
- **Still unverified after merge:** real provider field shapes and real abandonment behaviour. The first real job is the check.
