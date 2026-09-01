# T-108 — run-ownership record

**The single authoritative record of who owns the T-108 milestone run.** Opened before the run was
staged; updated at each monitoring interval; closed only after the wrapper exits, cleanup is
verified and the result is committed.

**An unacknowledged background process is a G11 violation.** This file exists so that no state of
this run is ever unowned.

---

## 1. Owner

| Field | Value |
|---|---|
| Session identity | `project14-t2pw-da` `[237fab]` |
| Role | Lead Orchestrator · Integration Authority · **T-108 Run Owner** |
| Repository | `C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW` (**primary checkout**, not a worktree) |
| Integration branch | `sprint/pwml-recovery` |
| Branch tip at claim | `0bbac3fd863d3ff22d1172354c4b367d34a6d1bd` |
| `local = origin/ = git ls-remote` | **verified equal at `0bbac3fd`**, 2026-09-01T22:00Z |
| `main` | local `7531692` / remote `03f1af5` — **neither ref written** |
| Ownership status | **SOLE OWNER — claimed 2026-09-01T22:11Z** |

## 2. Exclusivity — how it was established

`ListAgents` returned 25 peers. **Exactly one is live and in this repository:**
`project14-t2pw-93` (interactive). Every other peer is a cloud or Remote-Control session on
unrelated projects (StockScreener, LoadLink, quant work), idle or offline.

`project14-t2pw-93` was contacted directly and replied with an explicit four-point stand-down:

1. claims **no** Lead Orchestrator or integration authority role;
2. holds **no** branch, worktree, heavy lock, running job or pending edit — has run no Python at all;
3. claims **no** T-108 or benchmark ownership;
4. will **not** launch any `bounded_run.py`, pytest, benchmark or pipeline job, and will **not**
   clear `C:/t/heavylock`, while this run is live.

**Its caveat is recorded rather than smoothed over, because it is the honest shape of the hold.** It
cannot bind its user's future instructions; if later directed to run its RAG experiment it will
**notify this session first and wait**, and it commits **unconditionally** never to clear, steal or
touch the lock and never to launch a heavy job unilaterally. **That is a firm hold and ownership is
unambiguous.**

**One correction to the inherited record, accepted:** previous handoffs describe
`project14-t2pw-93` as *"working for a different user."* **That is wrong** — it is the **same user**
(`angads.chahil@gmail.com`), the same machine and the same account, on a **different task**.
Everything else in the record is right: read-only, unrelated assessment, not authorized for sprint
work. Ownership reasoning depends on this distinction, so it is corrected here rather than repeated.

**Transfer recipients: none.** There is no live peer authorized for sprint work, so **no transfer is
possible** and none is planned. This session therefore holds the run to completion. If an
unavoidable session failure occurs, a transfer is valid **only** to an already-live named peer that
explicitly acknowledges the wrapper PID, task id, output path, heavy-lock state, last observed
progress, and the responsibility to monitor through cleanup. **A transfer nobody has accepted is not
a transfer.**

## 3. The run

| Field | Value |
|---|---|
| Milestone identity | **T-108** — a NEW milestone. Not a re-run, re-score or re-reading of T-107 |
| Run directory | **`runs_verify/2026-09-01_1612`** (staged, verified, continued without `--fresh`) |
| Wrapper | `docs/pwml_recovery_sprint/evidence/bounded_run.py`, build `sha256:83d13954…` (unmodified) |
| Wrapper task id | **`b26hpbb2y`** (Bash tracked background; notifies this session on exit) |
| Wrapper root PID | **586012** (outer `bounded_run.py`) -> **584068** (job root, named in the lock) -> **590540** batch_run |
| Heavy lock | **HELD** - `C:/t/heavylock`, holder `T-108`, token `T-108:584068:06cdcf139f4c2b80`, acquired `2026-09-01T22:14:52Z` |
| Wrapper stdout/stderr log | `C:/t/x108/log/t108_run.log` |
| G11 cleanup report | `docs/pwml_recovery_sprint/evidence/g11/T-108/07-official-run.json` |
| Basetemp root | `C:/t/x108/` (parent pre-created) |
| Start time | **2026-09-01T22:14:52Z** (16:14:52 MDT) |
| Wrapper hard ceiling | **72000 s (20 h)** |
| Internal `--deadline` | **18 h** |
| Per-leg ceiling | **3600 s default, NO override** — `leg_timeout_overridden: false` |

## 4. Monitoring log

Recorded at every interval: wrapper alive · current paper/mode · completed legs · output growth ·
heavy-lock holder · last log mtime · provider errors · retry and timeout telemetry · finalization
reserve · partial-payload preservation.

| # | UTC | Wrapper | Legs | Notes |
|---|---|---|---|---|
| 0 | 2026-09-01T22:11Z | not launched | - | ownership claimed; pre-launch gates green |
| 1 | 2026-09-01T22:14:52Z | **LAUNCHED** | 0/20 | lock acquired. `CONTINUING the incomplete run 2026-09-01_1612 (no --fresh given)` - `already recorded : 0`, `still to do : 20`. Leg `[1/20] PMC12444477 / strict -> starting (timeout 3600s)`. Deadline 18.0h |
| 2 | 2026-09-01T22:22Z | alive | 0/20 | leg 1 in flight. Lock held, run dir 44 MB / 28 files, 9 owned python procs. Quiet log is expected mid-leg, not a stall |
| 3 | 2026-09-01T23:15Z | alive | **1/20** | `[1/20] PMC12444477 / strict -> TIMEOUT (timeout) in 3600.8s (60m00s)`. **Timed out at the FULL 3600 s ceiling** - on T-107 the same leg timed out at 1798.3 s against 1800 s. `budget.leg_timeout_overridden: false` confirmed **in the produced manifest**. C-111 preserved `LEG_TERMINAL.json` 1131 B + `LEG_TRACE.jsonl` 24564 B + `RESULT.txt` 5157 B - where T-107's timed-out legs preserved nothing. `[2/20] PMC12444477 / research -> starting`. Projection revised: T-107's finished legs are unaffected by the higher ceiling, so the estimate is T-107's 5.63 h plus up to 3x1800 s of extra timeout budget = **~7.1 h**, landing ~05:20Z. Well inside the 18 h deadline and 20 h wrapper |
| 4 | 2026-09-01T23:56Z | alive | **2/20** | `[2/20] PMC12444477 / research -> PASS WITH WARNINGS in 2446.5s (40m46s)`, **18 files**. **This is the 3600 s ruling vindicated by measurement.** The same leg on T-107 was `TIMEOUT 1800.5s, 0 files`; 2446.5 s is **136% of T-107's ceiling**, so it could never have finished on that budget. A leg lost to the clock, not to biology - exactly F-148's diagnosis - recovered by restoring the default. Consequence to measure at scoring: **LpxH may become measurable on T-108**, where T-107 had no payload on either `PMC12444477` leg. Not claimed yet. `[3/20] PMC13231680 / strict -> starting` |

## 5. Closure

_Closed at the bottom of this file once the wrapper exits, survivors are verified zero, the lock is
released and the result is committed._

**Status: OPEN - RUN LIVE. This session monitors through wrapper exit and cleanup.**
