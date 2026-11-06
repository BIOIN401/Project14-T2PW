# W4b: one 90-minute try, a timeout is not retried (PO ruling Q-E2E-1)

Branch `ws/W4b-time-budget`, base `6b108aa9`. Ruling: `PO-RULINGS-2026-11-04.md` Addendum 3.

## What changed
- `src/t2pw/web/retry.py`: `RUN_TIMEOUT_S = 90*60`; `PER_ITEM_WALL_CAP_S = RUN_TIMEOUT_S`;
  `KIND_TIMEOUT` removed from `RETRY_KINDS` (crash, network, LLM still retry inside the same
  90 min cap); comments rewritten with the evidence (E2E-1/E2E-2, median 18 / p90 43 min,
  15 of 280 legs over 60 min, timeout = LLM-latency variance). `MAX_RETRIES`,
  `MIN_RETRY_BUDGET_S` unchanged. `worker.py` imports the constants (default `--timeout`,
  `timeout_s`, `wall_cap_s`), so it follows with no edit.
- `help.html`, `submit.html`: "most take 20 to 60 minutes, stopped after 90"; help no longer says
  timeouts are retried. The phrase "20 to 60 minutes" is kept (test_web_app pins it).

## Tests
- Web+bridge 13 files: 301 passed (base 301), 0 survivors. `evidence/W4b/web.json`.
- SMOKE: 510 passed, 1 failed, 0 survivors. `evidence/W4b/smoke.json`.
  The failure, `test_c106_mutation_harness_executable.py::test_08_the_census_pins_still_match_the_committed_corpus`,
  was caused by the orchestrator's own E2E-2 evidence commit (it added a quarantine_report.json) and is fixed on `webserver/refactor` @ `1062ce0d`; it is NOT from this change: it counts `git ls-files *quarantine_report.json` (84 on base `6b108aa9`)
  against the c102 pin (83); W4b touches no such file. Pre-existing on `webserver/refactor`
  (some committed run added a report). Needs the product owner or a census re-pin (F-151).

## Test edits
- `test_web_retry.py::test_constants_are_the_decided_defaults`: pins 90 min, cap == attempt, timeout not in RETRY_KINDS (old pinned 3600/7200).
- `test_web_retry.py`: added `test_a_timeout_is_final`; removed timeout from the retrying-kinds parametrization; the wall-cap test's last line (timeout retried at full length) replaced by "a full attempt of run time leaves no retry".
- `test_web_worker.py::test_wall_cap_shapes_the_retry_deadline...`: 7000/7200 s became 5200/5400 s (same shape: ~200 s left).
- `test_service_driver_bridge.py`: two assertions flipped from "timeout is retryable" to final (passed-deadline test, renamed; parametrize row timeout `True` -> `False`).

## Extra edit (authorised by the orchestrator, wording only)
- `views.py` (two job-page strings) now say a run is stopped after 90 minutes; comment-only wording in `jobs.py:41` and `limits.py:60`. No logic changed; no test pins that wording.
- Web+bridge 13 files re-run after this edit: 301 passed, 0 survivors (`evidence/W4b/web2.json`). SMOKE not re-run.

## Open questions / reported
- `src/t2pw/web/views.py` (outside the boundary): job-page copy still says "about 20 to 60 minutes" (two lines), `_kind_text` for timeout says "the run took too long" which is fine, and `MAX_ATTEMPTS`/"Retrying automatically" only ever show for crash/network/LLM now. Suggest the same "stopped after 90 minutes" clause there.
- `jobs.py:41` and `limits.py:60` comments say "20-60 min" (comment-only, outside boundary).
- `driver_bridge.py` derives the driver timeout from the request deadline (`deadline - GRACE_S`), no separate budget; nothing shorter than 90 min hard-coded. `app.py` `Retry-After: 3600` is an unrelated HTTP header.
