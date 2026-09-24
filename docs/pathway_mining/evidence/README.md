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

## ORCH-743: pathway-mining iteration 1 (the four pilot-001 blockers)

Base `23e1ae83`. Item commits: `638c8f48` (1, raw completion lineage), `5fd3acad` (2, citation
recovery), `f5e44ec7` (3, search), `6f910144` (4, screen rescue). Every run below ended with
exit reason `completed`, **final surviving count 0**, cleanup success and, where pinned,
measured-tree `violations: []`.

| File | Tree / SHA | What | Result |
|---|---|---|---|
| `011-mining-suite-untouched-tip` | `C:/t/mining` @ `23e1ae83` | `tests/test_mining_*.py`, before any change | **226 passed** (heavy lock `ORCH-743`) |
| `012-g9-base-arm-lineage` | `C:/t/mining-base2` @ `23e1ae83` | item 1 G9 base arm, `test_mining_lineage.py`, run by the orchestrator | **3 failed** (behavioural) |
| `013-item2-replay/` | `C:/t/mining` @ `638c8f48` (+ the item-2 working tree for call 3 and the offline fallback) | item 2 diagnosis: pilot-001's extraction input replayed, 3 live calls, est. $0.018879 | see its [`README.md`](013-item2-replay/README.md) |
| `014-mining-suite-orch743-tip` | `C:/t/mining` @ `6f910144` | `tests/test_mining_*.py` (16 files) | **252 passed** in 6.59 s (heavy lock `ORCH-743`) |
| `015-smoke-feature-disabled-orch743-tip` | `C:/t/mining` @ `6f910144` | production SMOKE (22 files), `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** in 52.58 s (heavy lock `ORCH-743`) |
| `016-item3-search-probe/` | `C:/t/mining` @ `6f910144` | live, read-only EuropePMC + NCBI search. No model call and no cache read or write | see below |
| `017-item4-screen-fixture/` | `C:/t/mining` @ `6f910144` | offline: the mining screen over pilot-001's 21 screened papers | see below |

The G9 base arms for items 2, 3 and 4 are run by the orchestrator. This worker cannot
write into the base worktree.

**016, the search probe.** Pilot-001's three round-1 frontier items (12-OPDA, AOS, the
topic) ran through the new formulation and the real `PaperSource.search`. Attempt 1 gave 3
queries, 75 result slots and **66 unique papers**; pilot-001's round 1 gave 30 slots and 12
unique. NCBI contributed 10, 12 and 12 of the 25 results per query. Before, it contributed
none. Query strings: `"12-OPDA" AND "Arabidopsis thaliana"`, and
`"12-OPDA" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana"` for attempt 2.

**017, the screen fixture.** Each paper is screened with the vocabulary its round had.
Included: PMC9947334 (required; rescued, score 4.0), PMID 35404431 (desired; rescued, score
3.0) and PMC13364333 (the production screen's own inclusion). Excluded: PMC12280011
(desired, not reached), both borderline papers, and **all 15 must-exclude papers**.
