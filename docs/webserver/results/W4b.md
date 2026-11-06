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
  was caused by the orchestrator's own E2E-2 evidence commit `55f6d9d4`, which added a
  `quarantine_report.json`. That raised the `git ls-files '*quarantine_report.json'` census to 84 against the
  c102 pin of 83. It is fixed on `webserver/refactor` @ `1062ce0d`. W4b touches no such file. The trial-merge
  run gave SMOKE 511 (see the orchestrator section below).

## Test edits
- `test_web_retry.py::test_constants_are_the_decided_defaults`: pins 90 min, cap == attempt, timeout not in RETRY_KINDS (old pinned 3600/7200).
- `test_web_retry.py`: added `test_a_timeout_is_final`; removed timeout from the retrying-kinds parametrization; the wall-cap test's last line (timeout retried at full length) replaced by "a full attempt of run time leaves no retry".
- `test_web_worker.py::test_wall_cap_shapes_the_retry_deadline...`: 7000/7200 s became 5200/5400 s (same shape: ~200 s left).
- `test_service_driver_bridge.py`: two assertions flipped from "timeout is retryable" to final (passed-deadline test, renamed; parametrize row timeout `True` -> `False`).

## Extra edit (authorised by the orchestrator, wording only)
- `views.py` (two job-page strings) now say a run is stopped after 90 minutes; comment-only wording in `jobs.py:41` and `limits.py:60`. No logic changed; no test pins that wording.
- Web+bridge 13 files re-run after this edit: 301 passed, 0 survivors (`evidence/W4b/web2.json`). SMOKE not re-run.

## Open questions / reported (rewritten by the orchestrator after review F1)

The earlier items about the `views.py` copy and the `jobs.py`/`limits.py` comments were resolved by commits
`30ae2bac` and `30c43d59`.

- `driver_bridge.py` derives the driver timeout from the request deadline (`deadline - GRACE_S`). There is no
  separate budget, and nothing shorter than 90 min is hard-coded. `app.py` `Retry-After: 3600` is an
  unrelated HTTP header.
- **For the PO:** D-W05 in DECISIONS.md ("a run takes 20-60 minutes") needs the same update as the copy.
- Dead ends: none.

## Independent run, review and merge (added by the orchestrator)

**Independent run:** ws-test-runner (sonnet/medium). It ran on a trial merge `e6a0d199` of `d7b3b7e3` onto
`718f3c07`, in `C:/t/ws-W4b-trial`. Results: web+bridge 301, c106 alone 16 (census 83), golden
2+8+4+24 = 38, equivalence 14 alone, SMOKE 511. All exited 0 with 0 survivors. The gate waited about
3 min, for R3b. Reports: `docs/webserver/evidence/W4b/independent/`.

**Review:** ws-reviewer (opus/high), on the actual diff `6b108aa9..d7b3b7e3`. Verdict: **APPROVE WITH NOTES**.
A timeout is final on every path: the worker kill (`worker.py:291-293`), the bridge's passed-deadline
refusal (`driver_bridge.py:295`), the driver's inner timeout (`driver.py:2461-2463`) and crash recovery
through `_wall_used`. A crash after minute 75 is not retried (`MIN_RETRY_BUDGET_S`). The bridge's
deadline wins over the driver's `DEFAULT_TIMEOUT=7200`.

- F1 (should-fix): this section was stale. It was fixed here by the orchestrator.
- F2 (note): SMOKE 510/1 on the branch. Answered by SMOKE 511 on the trial merge.
- F3 (note, carried): `jobs.py:41-42` says "20 items is roughly 7-20 hours". Under the new cap the worst
  case is 30 h. Comment only.
- F4 (note, carried): `retry.py:50-51` says "the fastest runs take ~18 min", which contradicts "median 18 min".
  A 15-16 min retry budget is below the median run time. This is a data point for the PO.
- F5 (note): the final-timeout job-page text is accurate. Naming the 90-minute limit there is optional.
- F6 (note, carried): `service/rag_orchestration.py:647` keeps the RAG loop's own `CHILD_DEADLINE_SECONDS`
  (3480 s). It bounds the loop, not the run. It is out of the W4b boundary.
- F7 (note, carried): `help.html:17,19` has a comma splice and should say "90-minute limit". Cosmetic.

**Merge:** `23644199` (`--no-ff` into `webserver/refactor`). Its code tree is byte-identical to the tested
trial merge. The post-merge result is in the LEDGER.
