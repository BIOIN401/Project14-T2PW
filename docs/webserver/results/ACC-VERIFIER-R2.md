# ACC-VERIFIER-R2: reaction verifier rounds 2-5 merged; verifier ON for web jobs

2026-11-24. Branch `acc/verifier-r2` (final `30a77331`, code head `e72c2759`). Product track, outside the
R/W merge flow; PO approved both the main merge and enabling the verifier.

## Outcome

- **main** = `1a52fb55` (merge of `acc/verifier-r2` into `e17b78ba`). All four lane flags default OFF.
- **webserver/refactor** = `fb339129` (main merged in).
- **deploy/local** (local server deployment) = `9d44d705`, on top of the merge `102f1d26`. `deploy/local/start.sh`
  sets `T2PW_REACTION_VERIFIER=1` and pins `T2PW_LLM_MODEL=google/gemma-4-31b-it`, and unsets
  `T2PW_INTERACTION_TYPING`, `T2PW_VD18_SPAN_TEMPLATES` and `T2PW_VERIFIER_GRANULARITY`. CLI, batch and Streamlit runs
  stay off. The server is stopped; the setting applies the next time it starts.

## What changed since the first default-off merge (`e17b78ba`)

Every change below is reachable only with a flag on:

- R3 P1-P8: whole-word effect matching; sentence splitting at E. coli, et al. and Fig.; a wall-clock 600 s cap (optional
  `max_retries` on `llm/client.chat_detailed`, default unchanged); no second spelling on rename; no cofactor folding;
  off-pathway rows are flagged only; typing reads `interaction_type` before `relationship`.
- R4 B1: abbreviation guard. N1: "inhibitor", "inhibitory", "activator" and "repressor" count as effect words.
- Measurement: the verifier report records calls, seconds and model.
- F-G1: a failed typing step drops only that step, and the lock bookkeeping stays consistent.
- F-G2: a summary row is dropped only if every substrate reaches every product.
- F-G3: no rename onto another entity's name (exporter key `reference_key`); additions are deduplicated.

Lane record: `docs/pwml_output/ACCURACY.md` §11-§14d.

## Evidence (`docs/webserver/evidence/ACC-VERIFIER-R2/`)

- Reviews: R4 (`REVIEW-R4.md`) and R5 (`REVIEW-R5.md`, APPROVE). R5 verdicts:
  - merge default-off: approve;
  - enable the verifier: approve, conditional (below);
  - typing: merge off, don't enable on Gemma;
  - VD-18: stays off.
- Live, Gemma 4 31B in both arms:
  - **36-leg run** on `82bda3b6` (`score4g-gemma36.json`): 6 papers × 3 draws × 2 arms. It found F-G1, F-G2 and F-G3.
  - **12-leg re-measure** on `4d29c4bf` (`score5g-remeasure.json`):
    - enterobactin: verifier arm 15 gold matches vs base 14, precision 0.869 vs 0.730, PWML 3/3 in both arms;
    - MenD: equal in both arms;
    - verifier: at most 2 calls and 10 s per job, 0 errors.
- Offline paired application on every stored BASE payload (deepseek and Gemma answers): 0 gold lost.
- Gates on main `1a52fb55` (`main-merge-gates.log`):
  - passed: SMOKE 511, golden 2/8/4/24, equivalence 8/14/6, vision 370 (+1 skip), web 83, focused 1157;
  - focused failure (1): the stale conform test, fixed on webserver/refactor by WS-FIX-CONFORM;
  - LLM-client failures (3): order-dependent. The same 3 fail on `e17b78ba` in the same order, and all pass in the other
    order (`llm-client-order-check.log`).
- Gates on webserver/refactor `fb339129` (`webserver-merge-gates.log`): all green, including focused 1158 and LLM client 35.
- deploy/local: SMOKE 511 on the merged tree (`C:/t/vis-runs/logs/deploy-smoke.log`); the start.sh diff was independently
  reviewed (see LEDGER row).

## Conditions on the verifier being on (R5)

- Only with Gemma 4 31B. A model change means re-running the re-measure first.
- Code at or after `e72c2759`; typing and VD-18 off; granularity unset.
- Single server worker: lock-manifest pruning relies on the shared `tmp/` (R3 P8).
- `reaction_verifier_report.json` is kept in every job folder. The orchestrator spot-checks the first server jobs.

### Deploy review (ws-reviewer, 9d44d705): APPROVE WITH NOTES

- **Model pin cannot be overridden.** `llm/client.py:30-38` keeps `T2PW_LLM_MODEL` across `load_dotenv(override=True)`.
- **Verifier flag has no such guard.** `T2PW_REACTION_VERIFIER` is not in that per-run list, so a `.env` entry would
  override the export.
  - Check on the first server jobs: `reaction_verifier_report.json` exists in each job's `out/`, and every
    `stage0_attempts.json` model reads `google/gemma-4-31b-it`.
  - If the report is missing, `.env` is overriding the flag. The fix is then a `src/` hardening task.
- **Single worker confirmed.** `worker.py:217-229` takes an exclusive worker lock.
- **Untested combination.** The server runs 3 extraction draws (`T2PW_EXTRACTION_DRAWS=3`). The lane measured the
  verifier on single-draw legs. The first-job spot-check covers this.

## Dead ends and failures

- The first live 3×3 ran on deepseek and stalled on the host memory leak. Superseded by the Gemma run (PO: users run Gemma).
- The 36-leg run on `82bda3b6` failed two criteria: MenD gold 2 < 3 and enterobactin no-PWML 2 > 0. The causes were
  F-G1/2/3, fixed and re-measured.
- The kernel process leak paused the runs three times (each needed a reboot). Vanguard `vgk` is still loaded.
- Agent process slips: two `python -` stdin starts; both exited at once with no survivors.

## Open questions and follow-ups

- **PO, pipeline bug (outside the lane):** `entity_missing_name` abort at post_audit. It hit 2 of 35 Gemma legs (one with
  all flags off), 0 of 19 on deepseek. The same ticket covers the flags-off chlorophyll `AMBIGUOUS_REFERENCE "tRNA Glu"`
  abort. Ticket: `ACCURACY.md` §14b. Fixing it needs a biology-code change, so it waits for PO approval.
- Test isolation: `test_llm_client_empty_completion_retry.py` before `test_llm_client_request_timeout.py` fails 3 tests
  (leaked client state; same on old main). Gate scripts run them in a separate process, timeout file first.
- The main merge carries the stale conform test until webserver/refactor next merges into main.
- **The deploy tree fails two golden tests; this is a test problem, not a product bug.** The full gate run on
  `C:/t/t2pw-server` had 2 failures in `test_streamlit_quarantine_boundary.py`, the same 2 tests each time:
  `test_a_refinement_that_adds_an_unexportable_process_still_exports_the_core` and
  `test_research_mode_keeps_the_unmapped_candidate_and_does_not_block`. They pass at the same source in the clean
  worktree.
  - **Cause:** the server tree's `.env` makes the PathBank DB reachable. `PathBankDbResolver.from_env()` refills the
    host the test blanks (test line 437). The DB then maps `OPC-8:0` to its IUPAC name, which collides with
    `compounds#6` (`AMBIGUOUS_REFERENCE`).
  - **Bisection:** copying every outputs/, tmp/, data/ cache, `clean_model_cache` and `rag_index` file into the clean
    tree did not reproduce the failure. The culprit files are identical at `c889d336`, so it predates this merge.
  - **Follow-up:** give these tests an offline-DB guard (monkeypatch the `PATHBANK_DB_*` variables away).
  - **Real jobs:** they canonicalize against the DB by design. `run_pathway` clears quarantine artifacts at the start
    of each job (`service/pathway.py:256`, `:344`), and it only logs a warning if clearing fails.
- Worktrees for the PO to remove: `C:/t/acc-old`, `C:/t/acc-ver5`, `C:/t/merge-acc2`, `C:/t/ws-acc-check`.
