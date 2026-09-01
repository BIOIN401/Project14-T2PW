# C-111 — the three readings of F-148 § 3, written BEFORE the probe ran

**This file is committed as its own commit, before `c111_deadline_probe.py` exists and before any
probe result exists, so the ordering is provable from git history rather than asserted.**
REV-111 **B9** requires the three readings to be separated by MEASUREMENT, not argument. What
follows is what each reading predicts, and the rule that decides between them, fixed in advance.

F-148 § 3, verbatim, leaves these three open — all consistent with `elapsed = 1800.4`, none
excluded by the committed artifacts:

1. the child **never honoured** its 1680 s deadline;
2. the child **did** stop near 1680 s and its **finalization itself exceeded** the 120 s reserve;
3. the reserve is real but **nothing on the outer-kill path invokes finalization at all**.

**They are not mutually exclusive.** 2 and 3 can both be true; 1 and 2 cannot.

---

## 0. The recorded hint, and how it is being used

`PMC12444477/strict` fired an **in-process** `operation_timeout` at **1798.3 s**, not at 1680 s.
F-148 records this as *"a hint, not a measurement"*.

**It is used here only to choose which measurements to take (M3 and M4 below), never as a premise.**
The verdict rules in § 3 do not reference 1798.3, and every rule is decidable from probe output
alone. If the probe contradicts the hint, the probe wins and this file says so. (REV-111 **B10**.)

---

## 1. What the probe is allowed to be

Offline, cheap, synthetic. **No LLM spend, no benchmark leg, no T-run, no rerun of any T-107 leg,
no `runs/` or `runs_verify/` mutation.** Every filesystem write goes to a pytest/`tmp` directory
named explicitly by the probe. The run tree under measurement is named in the probe source; the
probe never reads or writes `runs/` or `runs_verify/`.

---

## 2. The six measurements, named before they are taken

| # | Measurement | Production code it drives |
|---|---|---|
| **M1** | The child deadline arithmetic and the argv actually built | `deadline.child_deadline_seconds`, `deadline.DEFAULT_FINALIZATION_RESERVE_SECONDS`, `runner.child_command` |
| **M2** | What the parent actually waits, and whether it hands the child-launch seam a `LegDeadline` | `runner._run_batch` leg loop via the public `run_batch(child_fn=...)` seam |
| **M3** | Whether the in-process budget admits work past its own total | `driver._Budget`, `driver._run_app`, `deadline.classify_interaction_timeout` |
| **M4** | Whether work that happens AFTER the in-process deadline is inside the leg's recorded elapsed, and whether anything bounds it | `driver.run_one` (real), with `_drive` replaced by a synthetic stage/finalize stub |
| **M5** | Whether the outer parent kill gives the child any opportunity to finalize | `runner.launch_child`, `runner._kill_tree`, against a REAL synthetic child process |
| **M6** | Scaled end-to-end reproduction of the T-107 shape: parent ceiling L, child deadline L−G, child finalization F | `runner.launch_child` + `runner.child_command` arithmetic, real subprocess |

### Controls, fixed in advance (a probe that cannot fail has measured nothing)

Every measurement carries **both** a known-positive and a known-negative arm. The probe **fails
loudly and reports nothing** if a control arm does not come out as stated here.

| Measurement | Known-POSITIVE (must be observed) | Known-NEGATIVE (must NOT be observed) |
|---|---|---|
| M1 | `child_deadline_seconds(1800, grace=120) == 1680.0`, and `--timeout 1680` appears in the argv | `--timeout 1800` does **not** appear in the argv |
| M2 | the recording `child_fn` is called with the parent's full ceiling as its wait | the recording `child_fn` is **not** called with the child deadline as its wait |
| M3 | with total T and an interaction that overruns, `_run_app` returns `timed_out=True` | `_run_app` does **not** return `timed_out=False` after the total is spent |
| M4 | a `_drive` stub that returns immediately makes `run_one` record ≈0 s | `run_one` does **not** record ≈ the timeout when nothing was spent |
| M5 | a child that exits BEFORE the parent's timeout writes its `finalized` marker | that same marker is **not** written by a child the parent kills |
| M6 | the arm with F < G leaves a complete payload on disk | the arm with F > G does **not** leave a complete payload on disk |

If M5's known-positive arm did not write `finalized`, the marker mechanism itself is broken and the
kill arm proves nothing. That is the C-108 lesson: the Lead's first verification probe used the
wrong payload envelope, never reached the guard, returned all-permissive at base and *looked*
exactly like a finding. Only contradicting two independent records exposed it.

---

## 3. The verdict rules — fixed before the data exists

**Reading 1 — "the child never honoured its 1680 s deadline" — is CONFIRMED if and only if**
M3 shows the in-process budget **admitting or continuing work** after its total is spent: either
`_run_app` returns `timed_out=False` with `budget.remaining <= 0`, or the interaction it starts is
given a slice that reaches past the total. **REFUTED if** M3 shows `_run_app` refusing to start an
interaction once `remaining <= 0` and capping every slice at `remaining`.

**Reading 2 — "the child stopped near its deadline and finalization overran the reserve" — is
CONFIRMED AS A MECHANISM if and only if** reading 1 is refuted **and** M4 shows that
`run_one` records `outcome.seconds ≈ total + F` for a `_drive` that finalizes for `F` seconds after
the budget is gone — i.e. post-deadline work is inside the leg's recorded elapsed and **nothing
bounds it**. **REFUTED if** `outcome.seconds` is clamped at or near `total` regardless of `F`.

> **Stated in advance, and it is a limit of this card, not a result of it:** offline the probe can
> establish the *mechanism* — that post-deadline finalization is unbounded and counted — and the
> *shape* (M6). It **cannot** establish that the REAL driver's finalization on the REAL
> `PMC12096016` / `PMC12444477` legs actually exceeded 120 s, because that needs a real leg, which
> is forbidden. **If the mechanism is confirmed and the magnitude is not, this file's verdict must
> read `MECHANISM CONFIRMED, MAGNITUDE NOT MEASURABLE OFFLINE` and must not be upgraded.**

**Reading 3 — "nothing on the outer-kill path invokes finalization at all" — is CONFIRMED if and
only if** M5's kill arm leaves the child's `started` marker on disk and **no** `finalized` marker,
while M5's control arm (same child, exits early) leaves both. **REFUTED if** the killed child's
`finalized` marker is present.

**Combination rules.** 1 and 2 are exclusive by construction. 3 is independent of both. If 2 and 3
are both confirmed, the picture is: *the child stops its work on time, then does unbounded
finalization work, and when the parent's ceiling arrives it is force-killed with no notice — so
whatever the finalization had not yet flushed is lost.* That is the picture this card must
instrument against, and it dictates **incremental flush-as-you-go durability** rather than a
write-at-the-end checkpoint, because a force kill gives a leg exactly zero instructions.

If instead reading 1 were confirmed, the instrumentation target would be different: a deadline that
is not enforced needs the *enforcement point* recorded, not the *finalization window*.

---

## 4. What this file is NOT

- It is not a diagnosis of the timeout. **C-111 instruments; if it fixes, it has failed.**
- Whatever the probe shows, **no repair of the finalization seam is made in this card** (charter § 5,
  REV-111 B4). A proven narrow defect is **registered** for a follow-on card.
- No retry behaviour, no leg ceiling, no `leg_timeout_override_*` is touched (B2, B3).
- `stage=unknown` on the outer-kill path is **honest** and is not made to guess (B5).
