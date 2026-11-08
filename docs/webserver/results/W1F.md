# W1F - follow-ups on the W1 input adapters

Branch `ws/W1F-followups` from `webserver/refactor` @ `71cfe7d3`. Code commit `192bc108`. Files: `src/t2pw/service/inputs.py`,
new tests appended to `tests/test_service_inputs.py` (no existing test edited). No behaviour change outside
the cases below; nothing under `pipeline/`, `pwml/`, `mapping/`, `rag/`, `curation/`, `llm/prompts/`.

## Follow-ups

1. **idconv errmsg (fixed).** A per-record `status: "error"` now maps to `NO_PMCID` only when `errmsg`
   (stripped, lower-cased) is in `_IDCONV_NOT_FOUND = {"identifier not found in pmc", "invalid article id"}`.
   The first is the live wording from `evidence/W1/testrunner/06-idconv-raw.json`; the second is kept
   because an existing W1 test uses it. Any other per-record error (rate limit, missing errmsg, new wording)
   is left out of the mapping, so the PMID is `LOOKUP_FAILED` (detail `absent_from_reply`), never a false
   "please upload the PDF". Match is exact on the whole message, not a substring.
2. **Unicode tests (added).** `test_w1f_mixed_ascii_and_unicode_digits_rejected`: `1٢٣`, `12٣`, `१२३`, `1２3`.
   These pass on the base too (the W1 regex was already ASCII-only); they close the coverage gap the reviewer named.
3. **Path comment (added).** Comment on `PdfResult.path`: internal key, never rendered to users.
4. **Per-chunk throttle (fixed).** W1 called `_ncbi_throttle()` once before `_idconv`, which then issued
   one request per 100 ids with no wait between them (250 ids = 3 back-to-back requests). `_idconv` now takes
   an optional `throttle` callable run before every chunk; `resolve_pmids` passes `_ncbi_throttle` for the default
   client and `None` for an injected one (same rule as before: an injected client is the caller's concern).
   The request was clear and stays inside `inputs.py`, so nothing is deferred.

## Tests (bounded wrapper, `--heavy-lock W1F`, `PYTHONPATH=C:/t/ws-W1F/src`, `T2PW_OFFLINE_CURATOR=1`)

New tests: 3 functions, 6 cases (4 Unicode, errmsg, throttle). All mocked, no network.

- `tests/test_service_inputs.py`: 39 passed (33 + 6), exit 0, survivors 0 - `evidence/W1F/01-focused.json/.log`.
- New tests against the base `inputs.py`: 2 failed (errmsg, throttle), 4 passed (Unicode), survivors 0 -
  `evidence/W1F/02-new-tests-on-base.json/.log`. `inputs.py` was restored afterwards.
- web+bridge list (13 files, expected 301 + 6 new): 307 passed, exit 0, survivors 0 - `evidence/W1F/03-web-bridge.json/.log`.

SMOKE was not run (not requested; diff touches one module that nothing outside its own test imports yet).

## Dead ends / incidents

- First run exit 95 (heavy lock held by another job); retried in a loop, passed on the second try.
- I ran one empty `python3 - <<EOF` heredoc by mistake while orienting (the pattern the brief forbids). It
  exited at once, ran no code, left no process.

## Open questions

None.
