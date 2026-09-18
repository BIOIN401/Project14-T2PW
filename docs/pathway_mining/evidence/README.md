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
