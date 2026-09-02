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

**Closing correction from the peer, recorded because the distinction matters.** After the run
finished I told `project14-t2pw-93` it was unblocked. It corrected me: **being unblocked is not
being authorized.** Its user asked for an assessment with edits explicitly excluded and has not
authorized any run, so releasing the lock removed an *external blocker*, not the *requirement*.
It is not launching anything, and would still notify before pushing. Recorded verbatim in effect
so that *"session 93 was unblocked and proceeded"* can never be read out of this file.

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
| 5 | 2026-09-02T00:06Z | alive | 3/20 | `PMC13231680/strict` FAIL(`no_reactions`) 624.8 s, stage1. **Negative control.** `operational_failure=null`, `termination_reason=null` -> the empty result is NOT caused by timeout, crash or infrastructure failure, satisfying **Q1 ruling condition 3**. The runner's `FAIL` token is the known-misleading one (`runner.py:717` has no gold access); `PASS_NEGATIVE_CONTROL` is a **scorer** verdict, verified at scoring, not asserted here. T-107 same leg: FAIL 679.9 s, triaged CORRECT |
| 6 | 2026-09-02T00:12Z | alive | 4/20 | `PMC13231680/research` FAIL(`no_reactions`) 360.4 s, 6 files. **CHANGED from T-107**, which had PASS 795.2 s / 16 files / `stage=research_report`. **Mechanism measured, not inferred:** stage0 ok 1 attempt `finish_reason=stop`; stage1 `outcome=ok` `response_status=ok` `finish_reason=stop` **`attempts=1`**; raw entities species 1 / compounds 3 / proteins 2; **reactions 0, transports 0**. So NOT a timeout, crash, truncation or retry exhaustion -- Stage 1 ran ONCE, succeeded, and returned a well-formed payload asserting no reactions. **What this is NOT: evidence that F-146 is fixed.** F-146 is this leg retaining an invented reaction, and it was Priority 2's single row on T-107. Its absence on ONE DRAW is not a fix -- CLAUDE.md's trap says identical legs give materially different Stage-1 draws at temperature 0 and a single leg must not be called a regression; **the symmetric rule binds, so it must not be called an improvement either.** Also NOT distinguishable from these artifacts: "declined" vs "this draw extracted nothing" -- the payload records zero reactions, not a recorded refusal |
| 7 | 2026-09-02T00:29Z | alive | 5/20 | `PMC12657337/strict` SCOPE_CONFLICT 1011.2 s (T-107 547.9 s). **Expected** -- organism trap, `eligibility_stage0_conflict_aborts=True`. Slower, same outcome. Gate held, nothing exported |
| 8 | 2026-09-02T00:43Z | alive | 6/20 | `PMC12657337/research` SCOPE_CONFLICT 842.9 s. Organism trap 1 of 3 complete, **both** legs aborted at Stage 0 as designed |
| 9 | 2026-09-02T00:57Z | alive | 7/20 | `PMC12421875/strict` SCOPE_CONFLICT 842.1 s. Organism trap 2 of 3. Expected |
| 10 | 2026-09-02T01:12Z | alive | 8/20 | `PMC12421875/research` SCOPE_CONFLICT 949.3 s. Organism trap 2 of 3 complete. **Pace:** 5 legs at 8044 s vs T-107's 5622 s for the same five; the entire +2422 s delta is the one leg that finished instead of timing out. Projection **6.30 h**, landing ~04:33Z -- inside the 18 h deadline and 20 h wrapper |
| 11-16 | 2026-09-02T01:19Z - 02:34Z | alive | 9-14/20 | Organism trap 3 of 3 aborted at Stage 0. `PMC12856317/strict` FAIL(contract) on `gate.protein_clpxp_is_missing_a_uniprot_or_drugbank_identifier` -- **reads as a regression, is not one**: T-107's `final_mapped.json` for that leg holds only `ALAS1`/`ALAS2`, no ClpXP, so the gate had nothing to fire on. `PMC12180156/strict` FAIL(contract) on `ferrochelatase` where T-107 had `ALAS2` -- **F-147 reproducing**, same seam, different protein |
| 17 | 2026-09-02T03:06Z | alive | 15/20 | **`PMC12096016/strict` PASS 1952.9 s**, PWML **74367 B**, gate_errors 0, blocking_issues 0. T-107: TIMEOUT 1800.2 s, 0 files. Needed **152.9 s = 2.5 min** beyond T-107's ceiling; **108.5%** of the old budget, **54.2%** of the new. The sharpest vindication of the § 2.1 ruling |
| 18-21 | 2026-09-02T03:29Z - 04:22Z | alive | 16-19/20 | `PMC12452463/strict` FAIL(contract) with **6** blocking issues vs T-107's 3 and T-106's 7 -- **the oscillation retires the previous wave's "improved 7 to 3" claim as draw variance**. `PMC12782028/strict` PASS 590.6 s vs T-107's 596.6 s, PWML 34931 B vs 35295 B -- **near-deterministic control proving the divergences elsewhere are draw-specific, not run-wide instability** |
| 22 | 2026-09-02T04:37:01Z | **EXITED** | **20/20** | `FAILURES: 12 of 20`. Wrapper exit **1** (`nonzero`) -- the expected code when not every leg passes, not an infrastructure failure. **22929.17 s = 6.37 h** of a 72000 s ceiling. **`FINAL SURVIVING COUNT : 0`**, **`cleanup : success`**, 44 descendants observed and 44 terminated, heavy lock **acquired and released**, 4 pre-existing reported and never killed |

## 5. Closure — CLOSED 2026-09-02T04:45Z

**T-108 ran exactly once and is complete. Verdict `NOT ACCEPTED` — `T108-RESULT.md`.**

| Closure check | Result |
|---|---|
| Wrapper exit | **1** (`nonzero`) — expected when not every leg passes; T-107 exited 1 likewise. **Not** an infrastructure failure |
| Duration | 22929.17 s = **6.37 h** of a 72000 s ceiling (31.8%) |
| `FINAL SURVIVING COUNT` | **0** |
| `cleanup` | **success** |
| survivors | `[]` |
| Heavy lock | `acquired=true released=true`; `C:/t/heavylock` **absent** after |
| Sprint-owned Python after | **zero** — verified by **full command line** |
| G11 | `check --task T-108` → **0 non-compliant** |
| `acceptance.py` | `4bd893ac…` unchanged |
| `streamlit_app.py` | `47e4fafa…` unchanged, still uncommitted |
| Gold | `36f4b7b6…` unchanged before and after |

**One honest deviation from the expected process baseline.** The handoff expects **two**
`ms-python.isort` IDE processes; **three** are present after the run — PIDs `662052`, `703704`,
`713836`, and one runs under system `c:\python313\python.exe` rather than the venv. All three match
`ms-python.isort ... lsp_server.py` **on the command line**, none is a sprint job, and none is a
cleanup target. The IDE spawned an extra language server during the run. Recorded rather than
explained away, because the baseline said two and the count is now three. **PIDs change; the command
line is the identity** — every original PID from launch is gone.

**Ownership held end to end by `project14-t2pw-da` `[237fab]`. No transfer occurred, none was needed,
and no job was ever unowned.** Monitor task `bwxowap41` stopped at completion; wrapper task
`b26hpbb2y` exited on its own.

**Status: CLOSED.**
