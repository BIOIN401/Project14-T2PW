# C-111 — what the probe measured, against the hypotheses committed before it

Hypotheses: `c111_three_readings_hypotheses.md`, commit **`4fde91b3`**, which predates
`c111_deadline_probe.py` and every number below. Probe source and its full redirected output:
`c111_deadline_probe.py` / `c111_deadline_probe.log`.
Cleanup report: `evidence/g11/C-111/02-deadline-probe.json` — `FINAL SURVIVING COUNT : 0`,
`cleanup : success`, exit 0.

**No LLM call, no provider call, no benchmark leg, no T-run, no rerun of any T-107 leg, and neither
`runs/` nor `runs_verify/` was read or written.** Every write went to one `tempfile.mkdtemp`
directory, removed at the end.

**The first run of this probe FAILED its own controls and is kept here beside its correction:**
`evidence/g11/C-111/01-deadline-probe.json`, `CONTROLS_FAILED: M2.control_positive_parent_waits_full_ceiling=False`.
The synthetic `BatchPaper` carried no `slug`, so `plan_pairs` skipped it, the leg loop never ran and
**M2 measured nothing while looking like a clean job** — zero survivors, `cleanup : success`, a
green-looking report. The control is the only reason that was visible. (The second failure in that
run, `M4.control_instant_drive_recorded_seconds`, was a naming bug in the probe's own control
collector, which treats any `control_*` key as a boolean; the value was a float. Renamed.)

---

## The verdicts, by the rules fixed in advance

| Reading | Verdict | Decided by |
|---|---|---|
| **1 — the child never honoured its 1680 s deadline** | **REFUTED** | M3 |
| **2 — the child stopped near its deadline and finalization overran the reserve** | **MECHANISM CONFIRMED, MAGNITUDE NOT MEASURABLE OFFLINE** | M4, M6 |
| **3 — nothing on the outer-kill path invokes finalization at all** | **CONFIRMED** | M5 |

2 and 3 both hold, which the hypotheses allowed for. 1 and 2 are exclusive and 1 fell.

---

## The measurements

**M1 — the arithmetic, and the argv actually built.** `child_deadline_seconds(1800, grace=120)`
= `1680.0`; `runner.child_command(..., timeout=1800)` emits `--timeout 1680`;
`DEFAULT_FINALIZATION_RESERVE_SECONDS` = `120.0`. Controls: positive (`1680` computed *and* in the
argv) and negative (`1800` is **not** in the argv) both held.

**M2 — what the parent waits versus what it tells the child.** Driving the real leg loop through
`run_batch(child_fn=…)`: the parent's wait is **`1800.0`** while the argv it hands the child says
**`1680`**. The manifest row it produced is the T-107 shape exactly — `stage: unknown`,
`files: []`, `budget.child_deadline_seconds: 1680.0`,
`leg_timeout_override_reason: ""`, `leg_timeout_override_source: ""`.

**M3 — reading 1, refuted.** With a 3.0 s in-process total, `driver._run_app` admitted three
interactions and then **refused the fourth**, at `elapsed = 3.001` — a **0.001 s** overrun of a 3 s
budget — with `detail = "whole-run budget of 3s was already spent"`, classified `budget_exhausted`.
`work_admitted_after_budget_gone` = **False**. Hypotheses clause B (a slice reaching past the total)
was measured as a **magnitude**, not a boolean, because `_Budget.slice` has a `max(1.0, …)` floor:
measured **0.001 s**. **A deadline honoured to a millisecond cannot explain a 120 s overrun.**
Second arm: a single interaction that overruns its own slice classifies as `operation_timeout`, not
`budget_exhausted` — the two in-process classifications are genuinely distinguishable.

**M4 — reading 2's mechanism, confirmed.** The real `driver.run_one` with a `_drive` that spends the
whole budget and then finalizes for 1.5 s: work stopped at **2.015 s** of a 2.0 s budget, and
`outcome.seconds` came back **3.51 s** — an overrun of **1.51 s**, which is the finalization, in
full. `run_one` samples `budget.elapsed` **after** `_drive` returns, and **nothing between the last
budget check and that sample consults the budget again.** Control: an instant `_drive` records
**0.0 s**, not the total, so the measurement is not an artifact of the harness.

> **This is also what the `1798.3 s` hint looks like when it is TESTED rather than confirmed.**
> F-148 recorded `PMC12444477/strict` firing an in-process `operation_timeout` at 1798.3 s rather
> than at 1680 s and called it *a hint, not a measurement*. M4 shows the mechanism that produces
> exactly that shape: the leg's recorded `seconds` includes post-deadline finalization. **The hint
> was used to choose the measurement, never as its premise** — M4's verdict rule references neither
> 1798.3 nor that leg, and M4 would have refuted the mechanism just as readily.

**M5 — reading 3, confirmed.** A real child, killed by the real `runner._kill_tree`
(`taskkill /F /T` on this platform): `started.txt` **present**, `finalized.txt` **absent**, though
the child had registered an `atexit` hook *and* SIGTERM/SIGINT/SIGBREAK handlers. The control arm —
the same child exiting before the parent's wait — wrote **both** markers. **The outer kill delivers
nothing the child can act on. A leg gets exactly zero instructions at the moment it dies.**

**M6 — the T-107 shape, scaled.** Leg ceiling 12 s, grace 4 s, child deadline 8 s, real subprocess
through the real `launch_child`:

| finalization | work stopped at | payload on disk | outcome |
|---|---|---|---|
| **6 s (> grace)** | `8.010 s` | **absent** | killed at 12.6 s |
| **1.5 s (< grace)** | `8.011 s` | **present** | exited cleanly at 9.6 s |

Same leg, same work, same stop — **the payload's survival is decided entirely by whether
finalization fits inside the grace.**

---

## What this settles, and what it does not

**Settled:** the child stops its work on time; the work that follows is unbounded and is counted in
the leg's elapsed; and when the ceiling arrives the child is force-killed with no notice.

**NOT settled, and stated as unsettleable in advance:** whether the REAL driver's finalization on
`PMC12096016` or `PMC12444477` actually exceeded 120 s. That needs a real leg, which is forbidden.
**The verdict above says `MECHANISM CONFIRMED, MAGNITUDE NOT MEASURABLE OFFLINE` and must not be
upgraded.**

---

## Registered, NOT repaired

**R-C111-1 — post-deadline finalization is unbounded and unpriced.** `run_one` samples the leg's
elapsed after `_drive` returns; nothing between the last `_run_app` budget check and that sample
consults the budget. A leg can therefore overrun its child deadline by an arbitrary amount, and the
120 s grace is the only thing standing between that and a force kill.

**R-C111-2 — the outer kill is unconditionally hard.** `_kill_tree` is `taskkill /F /T`; no graceful
signal precedes it, so the grace period is only useful to a child that finishes finalizing on its
own. Note this is also what makes the kill *reliable*, which is why it is a finding to weigh and not
an obvious defect to fix.

**R-C111-3 — `launch_child` accepts a `deadline=` argument that the production leg loop never
passes.** `_run_batch` calls `launch(cmd, timeout)` positionally.

**All three are REGISTERED for a follow-on card and repaired NOWHERE in C-111.** Charter § 5 and
REV-111 B4: observability first, and a proven narrow timeout defect is a registered finding, not a
blocking objection to this card. The instrument this card builds is what would let the next timeout
be diagnosed with real numbers instead of a probe.
