# Test-process evidence for the pathway-mining experiment

Every test and benchmark in this repository runs through the bounded foreground
wrapper (`docs/pwml_recovery_sprint/evidence/bounded_run.py`) with guaranteed
cleanup, and the wrapper's structured report is committed rather than pasted.
This directory holds those reports for the experimental pathway-mining work.

They are **not** filed under `docs/pwml_recovery_sprint/evidence/g11/` on
purpose: that tree is keyed to sprint task ids (`C-xxx`, `D-xxx`, `ORCH-xxx`),
and this is an experimental research feature, not a sprint card. Inventing a task
id to file under would misrepresent what this work is. The process discipline is
the same — bounded, owned, cleanup verified, zero survivors, measured tree
verified, report committed.

| File | |
|---|---|
| `001-smoke-feature-disabled.json` | the production SMOKE gate run on this branch with the feature OFF |
| `001-smoke-feature-disabled.pin.json` | the measured-tree verdict for that run |
| `002-mining-suite.json` | the pathway-mining suite |
| `002-mining-suite.pin.json` | its measured-tree verdict |

## What each report must show, and does

| Field | 001 | 002 |
|---|---|---|
| exit reason | `completed` | `completed` |
| exit code (real) | 0 | 0 |
| **final surviving count** | **0** | **0** |
| cleanup success | `true` | `true` |
| measured tree `violations` | `[]` | `[]` |
| heavy lock | acquired and released (`EXP-MINING`) | acquired and released |
| result | **508 passed** in 44.45 s | **222 passed** in 4.54 s |

508 is the count merge gate 10 expects for SMOKE, unchanged by this branch — the
feature-disabled regression evidence.

Pre-existing processes are reported by the wrapper and were **never** killed.
Cleanup targeted only PIDs the job created; no global `taskkill /IM python.exe`
or `pkill python` was used anywhere.

## Reproducing

Both commands are in [`../testing.md`](../testing.md). `--basetemp` is mandatory
and its parent must already exist.

## Tip measurements

`001` and `002` were taken mid-branch, before the documentation commit's two
report-wording fixes. `003` and `004` re-measure the same two gates at the
branch tip and are the ones to cite.

| File | Gate | Result |
|---|---|---|
| `003-smoke-feature-disabled-tip.json` | production SMOKE, feature OFF | **508 passed** in 45.67 s |
| `004-mining-suite-tip.json` | the pathway-mining suite | **222 passed** in 4.09 s |

Both: exit `completed` / code 0, **final surviving count 0**, cleanup success,
heavy lock `EXP-MINING` acquired and released, measured-tree `violations: []`.

## ORCH-742a: mining stability fixes

Two G9 corrections, committed in `ad4245f5`: the preflight probe budget
(hardcoded `8` → `PROBE_MAX_TOKENS = 64`, override
`PATHWAY_MINING_PROBE_MAX_TOKENS`) and the enforcement of
`max_new_reactions_per_round`. No live call in any run below. Every run: final
surviving count **0**, cleanup success, measured-tree `violations: []`.

| File | Tree / SHA | Selection | Result |
|---|---|---|---|
| `005-mining-suite-untouched-tip` | `C:/t/mining` @ `9c085967` | `tests/test_mining_*.py` | **222 passed** (baseline before any change) |
| `006-g9-base-arm` | `C:/t/mining-base` @ `9c085967` | `test_mining_stability.py` (earlier assertion order) | **3 failed** (behavioural) |
| `007-g9-tip-arm` | `C:/t/mining` working tree, before the commit | `test_mining_stability.py` + `test_mining_budget.py` | **23 passed** |
| `008-g9-base-arm-final` | `C:/t/mining-base` @ `9c085967` | `test_mining_stability.py` exactly as committed | **3 failed** (behavioural) |
| `009-mining-suite-fix-tip` | `C:/t/mining` @ `ad4245f5` | `tests/test_mining_*.py` | **226 passed** |
| `010-smoke-feature-disabled-fix-tip` | `C:/t/mining` @ `ad4245f5` | production SMOKE, `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** |

`008` is the G9 base arm to cite. Its three failures are assertions, not
import errors: the preflight **failed** a scripted reasoning model that needs
more than 8 tokens (`'fail' == 'ok'`); the failure detail named neither the
budget nor `PATHWAY_MINING_PROBE_MAX_TOKENS`; and a run with
`max_new_reactions_per_round=1` went on past round 1 and stopped with
`frontier_exhausted`. `006` is the same file with test 1's assertions in a
different order: it failed 3/3 as well, and is kept because it was run. For
both base arms the test file was copied into the base worktree, run with that
tree's own launcher (`--expect-tree C:/t/mining-base`), then deleted.

`007` measured the uncommitted working tree. `009` and `010` measured the
commit and are the ones to cite for the tip. `005`, `009` and `010` held the
heavy lock `ORCH-742a` (acquired and released). The G9 arms did not take it.
