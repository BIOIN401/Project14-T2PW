# ORCH-759d: the per-interaction app cap follows the leg budget

Lane `orch759d/app-timeout`, base `5d1ea64c` (`sprint/pwml-recovery`). Implementer report, 2026-10-22.
Not reviewed, not merged. Commits:

| commit | what |
|---|---|
| `30702e2f` | test-first: `tests/test_orch759d_app_timeout.py` on base source |
| `96d686c7` | code: `src/t2pw/batch/driver.py` |
| (this commit) | docs: this file, `docs/batch_runner.md` § 6, evidence |

## 1. Defect

`driver.run_one` (the brief called it `run_paper`; the function is `run_one`, `driver.py:2341` at base)
defaulted `app_timeout` to `DEFAULT_APP_TIMEOUT = 3600.0` (`driver.py:120`). `_Budget.slice()`
(`driver.py:1719`) offers each `at.run()` `min(per_run, remaining)`. The only production caller,
`runner.run_single` (`runner.py:1195`), calls `runner(paper, mode, timeout=max(60.0, float(timeout)))`
and nothing else; `scripts/batch_run.py` has no flag for it. So under the 10,800 s operating ruling
(`ORCH-759-RESULT.md` § 1, `ORCH-756-RESULT.md` § 2) the child got a 10,680 s budget, but any single
interaction still died at 3,600 s.

Measured, ORCH-759 Wnt/Ca2+ strict,
`runs_papers/ORCH-759/legs/wnt_ca-strict/2026-10-22_1528/papers/PMC10425814/strict/`:

- `RESULT.txt`: `message: extraction did not finish inside the time budget`,
  `DETAIL: AppTest script run timed out after 3600.0(s)`, wall time 5,295.9 s.
- `LEG_TERMINAL.json`: `timeout_source: in_process_deadline`, `leg_timeout_seconds: 10800.0`,
  `child_deadline_seconds: 10680.0`, reserve `left_seconds: 120.0`, `exhausted: false`.
- `LEG_TRACE.jsonl`: Stage 2 model attempts still returning `ok` up to `elapsed_seconds: 5295.828`;
  driver `stage_end` at 5,295.847.

The detail line is the direct proof: the interaction was offered 3,600 s, not the 10,680 s the leg had.

## 2. Diff (`96d686c7`, `driver.py` only, +7 −5)

- `run_one(..., app_timeout: Optional[float] = None)`; `None` means the leg's own `timeout`:
  `budget = _Budget(timeout, timeout if app_timeout is None else app_timeout)`. With per-run equal to
  the total, `slice()` is `max(1.0, remaining)`: the interaction gets whatever the leg has left.
- A caller that passes `app_timeout` gets exactly what it passed, as before. Every test that calls
  `run_one` passes it explicitly (`grep -n "run_one(" tests/`), so none of their budgets move.
- `DEFAULT_APP_TIMEOUT` and its `__all__` entry are deleted: nothing in the repo imports it
  (`git grep DEFAULT_APP_TIMEOUT` finds only `docs/batch_runner.md`, updated here, and this lane's
  test docstring).
- No `--app-timeout` CLI flag. Base had none, so no CLI user could set a smaller cap before; the API
  argument still exists.

`_Budget`, `_run_app`, `runner.py`, `scripts/batch_run.py`, `streamlit_app.py` and every pipeline,
mapping and quarantine module are untouched.

## 3. Why the leg deadline and the finalization reserve still hold

1. **The driver never offers time past the child deadline.** `slice()` is still `min(per_run, remaining)`
   and `remaining = total - elapsed`, where `total` is the child budget the runner passes
   (`child_deadline_seconds(10800, grace=120) = 10680`). `_run_app` still refuses to start an
   interaction once `remaining <= 0`. Proved for both a 3,000 s leg and the 10,680 s leg
   (§ 4, preservation 1): every `start + timeout <= deadline`.
2. **The 120 s reserve sits outside the child budget.** It is the gap between the parent's kill
   (`launch_child(..., timeout)` at 10,800 s) and the child deadline (10,680 s), and this change
   touches neither. The proof computes `leg_trace.finalization_reserve_record` from the leg's
   elapsed time: `used_seconds == 0.0`, `exhausted is False`.
3. **A hung interaction is still bounded.** `AppTest.run(timeout=...)` only *reports* a timeout: it
   requests a stop and joins the script thread with no limit (`docs/batch_runner.md` § 7; streamlit
   `local_script_runner.require_widgets_deltas`). A blocked call is therefore bounded, as at base, by
   the **parent's process-tree kill at the leg timeout (10,800 s)** and, past that, by the **outer
   `bounded_run.py` wrapper** (11,400 s for ORCH-759 paper legs). This change does not weaken either.
   The fake clock in the proof models a *cooperative* interaction only, and says so in its docstring.

## 4. G9 proof

`tests/test_orch759d_app_timeout.py`, 4 tests. **A regression fix, not a new capability**: every
name it uses exists at `5d1ea64c`, it imports cleanly there and fails on a value.

Harness: the `tests/test_batch_driver.py` fixture app (strict-pass body), `driver.time` replaced by a
fake clock, and `AppTest.run` wrapped so each interaction runs the instant fixture script and then
costs `fake_cost` fake seconds (set by the script; the submit costs N, everything else 1 s). A cost
above the offered timeout advances the clock by the timeout and raises AppTest's own
`RuntimeError('AppTest script run timed out after ...')`.

| test | path | base `5d1ea64c` (`01-g9-red`) | tip (`02-g9-green`) |
|---|---|---|---|
| `test_g9_a_4000s_submit_finishes_inside_a_10800s_leg` | `runner.run_single(timeout=10680)` -> real `driver.run_one`, no `app_timeout` | **FAIL**: `'timeout' != 'pass'`, `extraction did not finish inside the time budget`, submit offered `3600.0` | pass: submit offered `10678.0` (all remaining), leg `pass` at `pwml_export` |
| `test_preservation_the_leg_deadline_still_caps_an_interaction[3000-4000]` | same, 3,000 s child deadline | pass | pass |
| `test_preservation_the_leg_deadline_still_caps_an_interaction[10680-20000]` | same, 10,680 s, submit needs 20,000 s | pass (stopped by the 3,600 s cap) | pass (stopped by the leg: every `start + timeout <= 10680`, reserve unused) |
| `test_preservation_an_explicit_app_timeout_still_caps_each_interaction` | `run_one(..., app_timeout=3600)` | pass | pass: offered `[3600, 3600, 3600]` |

The RED log (`evidence/orch759d/01-g9-red.log`) shows the call record at base:
`[{'cost': 1.0, 'start': 0.0, 'timeout': 3600.0}, {'cost': 1.0, 'start': 1.0, 'timeout': 3600.0},
{'cost': 4000.0, 'start': 2.0, 'timeout': 3600.0}]`.

## 5. Gates

All runs through `bounded_run.py` + `pinned_pytest.py`, `--basetemp` under `C:/t/bt-orch759d2/`, one
job at a time, no heavy lock. Reports `evidence/g11/ORCH-759d/`, logs `evidence/orch759d/`.

| gate | report | result | survivors |
|---|---|---|---|
| G9 RED, test commit on base source | `01-g9-red` | 1 failed, 3 passed | 0 |
| G9 GREEN, tip | `02-g9-green` | 4 passed | 0 |
| Affected, 31 files (`evidence/orch759d/affected_run.txt`) | `03-affected` | 840 passed, 1 skipped, **2 failed** | 0 |
| The 2 failures on base `driver.py` | `04-affected-fails-at-base` | same 2 failed, same assertions | 0 |
| `tests/test_batch_preflight.py` alone | `05-preflight` | 32 passed, 4 skipped, 1 failed (known) | 0 |
| SMOKE, 22 files (`evidence/orch759d/smoke.txt`) | `06-smoke` | **508 passed** | 0 |

Affected set: `tests/test_batch_*.py`, `tests/test_early_failure_replay.py`, and every file matching
`grep -l -E "app_timeout|_Budget|DEFAULT_APP_TIMEOUT|batch_run|runner" tests/`, minus
`test_batch_preflight.py` (run alone) and `test_c055_rag_loop_wiring.py` (Chunk D AppTest file; it
matched only on the string `round_runner` and never calls the driver; not run, per the brief).

The failures, all unrelated to the driver:

- `test_c075_source_support_armed.py::test_corpus_the_armed_pass_over_every_committed_artifact_and_its_own_paper`:
  `unexpected: [('PMC12096016', 'strict', "Pantetheine 4'-phosphate")]`. Identical with base `driver.py`
  (`/04`); already on the known list (`TEST_MATRIX.md`, lanes 754a/754b).
- `test_paper_eligibility_corrections.py::test_the_2122_evaluation_with_cached_abstracts_beats_title_only`:
  `{'derived_from_stored_full_text': 28} != {'derived_from_stored_full_text': 1}` (no
  `acquisition_cache` in a worktree). Identical with base `driver.py` (`/04`). Not on the known list
  I found; it reads a cache, not the driver.
- `test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure`:
  `this project ships a .venv; the test assumes it`. Worktree-only, known (`TEST_MATRIX.md` ORCH-757 row;
  `ORCH-757/22`, `ORCH-758b/10`).

`/04` ran with `git checkout 5d1ea64c -- src/t2pw/batch/driver.py` in this worktree, then
`git checkout HEAD -- src/t2pw/batch/driver.py`; `git diff HEAD` was empty afterwards.

## 6. Findings

1. **A 3,600 s report took 5,296 s.** The Wnt/Ca2+ leg's submit was offered 3,600 s, but the child did
   not record the timeout until 5,296 s: Stage 2 calls kept returning `ok` for ~1,690 s after the cap.
   That is AppTest's cooperative stop (§ 3.3): the script stops only when the pipeline next returns
   control to a Streamlit call. The overrun is the length of the work in flight, not one LLM call.
2. **Dead end ruled out: "the runner drops `app_timeout`".** The runner never had it: `run_single`
   passes only `timeout`, and `run_one`'s default was the cap. Fixing the default fixes every caller
   that omits it; plumbing the value through the runner would have fixed one.
3. The brief's `run_paper` does not exist; the function is `run_one`. Line numbers otherwise matched.

## 7. NOT settled

**Q1. Finding 1 applies at the new deadline too.** At the tip a slow leg is offered time up to the
child deadline (10,680 s). If the work in flight then runs on for more than the 120 s grace, the parent
kills the child at 10,800 s, and the leg's artifacts are lost (synthesized `timeout` row, empty mode
folder, `timeout_source: outer_parent_kill`). Base had the same exposure for any interaction that
reached the deadline; at base the 3,600 s cap usually ended the leg earlier, with a child-written row.
Options:
(a) accept it: the leg was out of time either way; cost: such legs report via the parent's row;
(b) a larger grace for paper runs; cost: a `runner.py` change plus wall time per timed-out leg;
(c) wire the leg deadline into pipeline admission so Stage 1/2 refuse a call that cannot finish
before the child deadline; cost: a pipeline-lane change, out of this lane's boundary.

**Q2. Deleting `DEFAULT_APP_TIMEOUT` removes a public name.** No importer exists in the repo. If an
out-of-tree script uses it, the reviewer can ask for `DEFAULT_APP_TIMEOUT = None` instead; cost: a
name whose old meaning (3,600 s) no longer applies.

**Q3. No `--app-timeout` flag.** Nothing at base exposed one. Add it only if someone wants a smaller
per-interaction cap from the CLI.

## 8. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

Append end-of-file:

```markdown
## ORCH-759d: test obligations (appended end-of-file by rule) · 2026-10-22

Lane `orch759d/app-timeout` (`30702e2f` test-first, `96d686c7` code). Reports `evidence/g11/ORCH-759d/01`–`06`.

**SMOKE is unchanged at 508** (`g11/ORCH-759d/06`). No pin moved.

### New focused selection

`tests/test_orch759d_app_timeout.py` (4, ~2 s): through `runner.run_single` and the real
`driver.run_one`, with a fake clock and a cost-charging `AppTest.run`, a 4,000 s submit finishes inside
a 10,800 s leg (child deadline 10,680 s); an interaction that outruns the leg still times out at or
before the child deadline with the 120 s finalization reserve unused; an explicit `app_timeout` still
caps every interaction. Run it for any change to `driver.run_one`'s budget arguments, `_Budget`,
`_run_app`, or the `timeout` that `runner.run_single` hands the driver.

### Known failures

In a worktree, `test_paper_eligibility_corrections.py::test_the_2122_evaluation_with_cached_abstracts_beats_title_only`
also fails (`{'derived_from_stored_full_text': 28} != {'derived_from_stored_full_text': 1}`), identically
on base `src/t2pw/batch/driver.py` (`g11/ORCH-759d/04`).
```

## 9. Files changed

Test (`30702e2f`): `tests/test_orch759d_app_timeout.py` (new).
Code (`96d686c7`): `src/t2pw/batch/driver.py`.
Docs/evidence (this commit): this file; `docs/batch_runner.md` § 6 (the "Two clocks" bullet no longer
names `DEFAULT_APP_TIMEOUT`); `evidence/g11/ORCH-759d/01`–`06` and their pin verdicts under
`evidence/g11/pin/ORCH-759d/`; `evidence/orch759d/` (6 logs, `affected_run.txt`, `smoke.txt`).
