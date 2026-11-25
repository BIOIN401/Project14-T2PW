# OBS-RAW: keep the raw LLM replies of a run

Branch `obs/raw-llm-responses`, cut from main `5bf9d552`. Code commits `2c81ef19` (hook) and `355c1963` (tool calls, see below).

## What
`T2PW_KEEP_RAW_LLM=1` (default off, read per call like `T2PW_REACTION_VERIFIER`) makes every provider
attempt append one JSON line to `llm_raw_responses.jsonl` in the folder of the run's leg trace. With the
flag off nothing is written and nothing else changes.

## Where the hook is
`src/t2pw/llm/client.py`: `_keep_raw()`, called from `CompletionDiagnostics.note()`. `note()` is already
called for every attempt (ok, empty, content filter, rate limit, timeout, error) by `chat`,
`chat_detailed` and by the `chat_with_tools` loop, so one hook covers audit (`audit loop`), the gap resolver
(`gap resolver`, including its agent loop), the curator, and the enrichment alias call in `mapping/map_ids.py`.
It also covers every other stage (extraction, preprocessor, verifier, ...); filter on `stage`.
Only the `note()` signature (new optional `raw=`) and the OK paths of `chat_detailed` and `chat_with_tools`
changed. In `chat_with_tools` outside a web run, a private diagnostics object with `_raw_only=True` is made
only when the flag is on, so the leg trace gets no extra rows.

## Where the file lands
The folder is `leg_trace.active().path.parent`. `run_pathway` activates the trace at `out_dir/LEG_TRACE.jsonl`
(server job `out/`), the batch runner at `papers/<id>/<mode>/LEG_TRACE.jsonl`. So the new file sits beside
`LEG_TRACE.jsonl`. It is not in `report_paths`, so it is not a public download. With no active trace
(plain Streamlit use) nothing is written.

## Fresh per run
The first write after each trace activation opens the file with `"w"`, later writes append. A new job or
leg activates a new trace, so one run's replies never carry into another's folder. Limit: a run that makes
zero LLM calls leaves an old file in a reused folder untouched (job folders are unique, so not seen in practice).

## Format (one line per attempt)
`ts` (UTC ISO), `stage`, `model`, `attempt`, `status` (`ok`/`empty`/`error`/`timeout`/...), `finish_reason`,
`error` (credential-redacted by `leg_trace.redact`), `request_hash`, and for an `ok` attempt `response_chars`
and `response` (the reply exactly as returned, not stripped). A tool-call turn also has `tool_calls`:
`[{"name", "arguments"}]`, each function name and its raw arguments string exactly as returned (the gap resolver's and
curator's edit ops are in the arguments). The reply and all arguments share one 200 000-char budget per line; a
string that does not fit is cut and `"\n[TRUNCATED: N chars, first K kept]"` is appended, with `truncated: true`. No headers, key or provider
config. `request_hash` is the existing `sha256:` prefix (16 hex) of the messages, the same value as in
`LEG_TRACE.jsonl`, so lines can be joined to the trace. Failed attempts have no reply, only status/error.
Write errors are logged as a warning and swallowed. A lock serialises writes from concurrent draws.

## Tool calls (added after review)
`chat_with_tools` passes `(name, arguments)` of each `message.tool_calls` entry to `note()`. They are read only when the
flag is on, inside a try, so a malformed tool call cannot fail the run. 2 new tests: a tool-call turn with odd-spaced and
non-JSON arguments is kept verbatim; the shared cap (reply 8 + arguments 8 + 5 under a cap of 10).
Gate for this change (new file + client files only, `suite2` skipped by instruction, `obsraw-final.log`): new test file
10 passed, `test_llm_client_request_timeout` 6 passed, `test_llm_client_empty_completion_retry` 29 passed, surviving 0.

## Review fixes (F1, F9, F3)
- F3: the per-trace restart flag is set only after a good write, so a failed first write cannot make later lines
  append to an old run's file in a reused folder. Test: first write fails, old file appears, next line replaces it.
- F1: a test calls `_keep_raw` with an error holding a key-shaped string and asserts `[redacted:openai_style_key]`
  is in the line and the key is not (it fails if `redact()` is removed).
- F9: the fake key is built at runtime (`"sk-" + "or-v1-" + ...`), no key-shaped literal in the source.
Gate (`obsraw-final2.log`, own processes): new file 12 passed, `test_llm_call_logging` 13, `test_llm_call_cap` 28,
`test_llm_client_request_timeout` 6, `test_llm_client_empty_completion_retry` 29; surviving 0.

## Tests
`tests/test_obs_raw_llm_responses.py`, 12 tests (8 below plus 2 tool-call tests and 2 from the review fixes): flag off writes nothing; flag on one line per call with the
fields and the unstripped reply; tool loop covered and leg trace unchanged; empty attempt line; no secret or
provider field (key in env not in file); truncation marker; fresh file per activation; failing write does not raise.

## Gates (all through the queue with the heavy lock; every wrapper report: FINAL SURVIVING COUNT 0)
- new file: 8 passed.
- 44 files importing the llm client or `leg_trace` (core group 692 tests): 691 passed, 1 failed (see below).
  AppTest files one per process: `test_orch759b` 7, `test_orch759d` 4, `test_streamlit_stage2_orchestration` 12 passed;
  `test_batch_preflight` 32 passed, 4 skipped, 1 failed (see below).
- Order-dependent pair, own process, timeout first: `test_llm_client_request_timeout` 6 passed,
  `test_llm_client_empty_completion_retry` 29 passed.
- Full gate `suite2.sh`: SMOKE 511, golden-seam 2/8/4/24, equivalence 8/14/6, vision 370 + 1 skipped, web 83: all passed.
  Golden tests not edited.

Two failures, both pre-existing: re-run on the stashed base `5bf9d552` they fail identically
(`obsraw-base.log`): `test_curator_offline_mode.py::test_a7_flag_is_read_in_exactly_one_module` (vision modules
read `T2PW_OFFLINE_CURATOR`) and `test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure`
(the worktree has no `.venv`). Neither is in SMOKE.

## Dead ends / notes
- First coverage run: collection errors from a `\n` escaped wrongly by my edit script (syntax error in client.py);
  fixed before any other result was taken. Evidence logs are of the re-run.
- Waited about 2 h for the heavy lock held by another session.
- Scope is wider than audit and enrichment (all stages), by design of the single hook. A stage filter is a one-liner if wanted.

## Open questions
None.
