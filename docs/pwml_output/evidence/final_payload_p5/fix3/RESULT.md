# Final payload P5, fix round 3 (review 3: R3-1, R3-2, R3-3, R3-8)

Code: `src/t2pw/service/final_payload.py` only (postpipeline.py and pathway.py untouched). New tests: `tests/test_final_payload_default_seam.py` (19 cases).

## Per finding

- **R3-1 (major) fixed.**
  - `_Budget` now opens a circuit after `CIRCUIT_BREAKER_N = 5` consecutive `FetchError`s. A success (including a 404 `None`) resets the count. Once open, every request is refused without touching the network. The old `exhausted` is now a property (`refused > 0`).
  - `_cap_to_job_deadline(fetcher, state)` shortens the id-fill/nucleic fetcher and the title fetcher to `state["pathway_run_deadline"] - time.time() - DEADLINE_MARGIN_S (60)`, floored at 0. The key is an absolute `time.time()` epoch second (it is `deadline - GRACE_S` already, see `extraction.DEADLINE_STATE_KEY`); a test pins the constant equal to extraction's. No deadline, a non-number or nan: unchanged. A caller-given fetcher (fakes) is never touched.
  - Addition beyond the brief: under a deadline the live inner fetcher is also set to `timeout=10`, `retries=2` (worst request about 33 s instead of about 127 s). Without it one blackholed in-flight request could overrun the 60 s margin on its own. Only applies when a deadline is in state, so Streamlit and flag-on-without-deadline behave as before.
  - `circuit_open` is in the report of the id fill, the nucleic step and the title step.
- **R3-2 (major) addressed with the limits below.** New tests set `T2PW_FINAL_PAYLOAD=1`, replace only the inner `UrllibFetcher` (so the real `_live_fetcher`, `_Budget`, deadline cap, resolver and `freeze_canonical_payload` run, called the way postpipeline calls it: no `fetcher=`, no `cache=`; the disk cache read is patched out):
  - dead network: payload equal to the flag-off run, no fill/title key, `written 0`, `error None`, `circuit_open` and `budget_exhausted` True, exactly 5 inner calls, freeze verdict equal to flag-off;
  - empty answers: returns in under 30 s, payload unchanged, no circuit;
  - deadline nearly spent (60 s margin minus 5 s) and deadline already past: 0 inner calls, including the title step;
  - plus circuit unit tests, per-step `budget_exhausted`, budget env tests.
- **R3-3 (minor) fixed.** Each step's `budget_exhausted` is `refused after > refused before` for that step; the module docstring now says no new request starts after about 360 s but an in-flight one can run past it, and describes the new bounds.
- **R3-8 (minor) fixed.** `T2PW_FINAL_PAYLOAD_BUDGET_S` sets the id-fill budget and caps the title budget (`min(60, value)`); nan, inf, negative, non-numeric and empty fall back to the defaults. `=0` makes zero network calls in all three steps (the comparison is now `>=`), documented as the offline switch in the module docstring.

## Tests

`tests/test_final_payload_*.py`, `tests/test_pwml_writer*.py`, `tests/test_pwml_ir*.py` (reports `tests/fix3b.*`): 323 passed, 1 failed, surviving processes 0. The one failure, `test_pwml_writer.py::test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance`, asserts the PathBank DB is unreachable (`report["db_resolution"]["available"] is False`) and it is reachable from this worktree; it failed the same way in fix rounds 1 and 2 (`tests/fix1a`, `fix1c`) and is unrelated to this change. Focused first run `tests/fix3a.*`: 1 failed (my own title test assumed 5 title requests; the title makes fewer, assertion corrected), 44 passed.

## Not established

- No test runs the real `run_post_pipeline_sbml_artifacts` end to end (it is one very large function needing a full pipeline state); the seam is `prefreeze_with_final_fill` with the real resolver, plus `freeze_canonical_payload`, which is what the other final-payload tests use too. The app and `run_pathway` default path as a whole is still not exercised with the flag on; the orchestrator's SMOKE runs with the pin at 0.
- No live network run: how a real dead or blackholed host behaves (timeout 10 s, 2 retries under a deadline) is reasoned from `UrllibFetcher`, not measured.
- The 60 s margin and 5-error breaker are the brief's numbers; the remaining pipeline time after the hook (freeze, audit tail, export) was not measured against them.
- With the flag on and no deadline (Streamlit), the worst case is still the 300 s plus 60 s budgets plus one in-flight request.
