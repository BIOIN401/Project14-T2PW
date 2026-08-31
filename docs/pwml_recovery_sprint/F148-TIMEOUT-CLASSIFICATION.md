# F-148 — the three T-107 timeouts, classified from committed artifacts

**Wave ORCH-717. `runs_verify/2026-08-28_1816` was NOT rerun and no leg of it was repeated.**
Everything here is read out of the artifacts that run already produced —
`evidence/orch717_f148_timeout_probe.py` / `.log`, G11 `ORCH-717/03-f148-timeout-probe.json`.

The charter required these to be separated into: ordinary stochastic timeout · wrapper or deadline
behaviour · provider failure · pipeline non-termination · paper-specific pathological expansion ·
retry amplification · absence of a payload caused by **cleanup** rather than by pipeline failure.

**They are not one thing, and the SUMMARY's framing of them is wrong in two ways.**

---

## 0. The answer, in one table

| Candidate cause | Verdict | On what evidence |
|---|---|---|
| Ordinary stochastic timeout | **NO — budget-bound, not unlucky** | the slowest leg that *finished* used **92.1%** of the ceiling |
| **Wrapper / deadline behaviour** | **YES — the primary finding** | the 120 s finalization reserve was consumed to **zero** on both outer-kill legs |
| Provider failure | **NO** | 17 of 20 legs finished; **0** occurrences of retry/backoff/rate-limit/429 in `batch.log` |
| Pipeline non-termination | **NO** for `PMC12096016` · **UNPROVEN** for `PMC12444477` | `PMC12096016/research` finished in 1263.4 s with 5 reactions |
| Paper-specific pathological expansion | **NOT SUPPORTED** for `PMC12096016` · **plausible, unproven** for `PMC12444477` | see § 4 |
| Retry amplification | **CANNOT BE EXCLUDED — and that is itself the finding** | the timed-out legs preserved **no attempt record at all** |
| **Payload absence caused by cleanup, not pipeline failure** | **YES, for the two `budget_exhausted` legs** | the child was killed *while running*, with its preservation window already spent |

---

## 1. There are TWO mechanisms here, and the run reports them as one

| Leg | `termination_reason` | `stage` | elapsed | mechanism |
|---|---|---|---|---|
| `PMC12444477/strict` | **`operation_timeout`** | `input` | 1798.3 s | **in-process** deadline path |
| `PMC12444477/research` | **`budget_exhausted`** | `unknown` | 1800.47 s | **outer parent kill** |
| `PMC12096016/strict` | **`budget_exhausted`** | `unknown` | 1800.16 s | **outer parent kill** |

The in-process path knows where it was (`stage=input`) and declares its missing budget honestly
rather than guessing it — that is **F-092 defect 3 closed**, and it is the one thing in this
picture working as the contract asks.

The outer-kill path records `stage=unknown` **because the parent genuinely does not know**: it
killed a child that never reported back. Two legs, one `stage` field, two entirely different
epistemic situations. Any future reader treating `stage=unknown` as a pipeline stage rather than as
"the parent could not see" will mis-diagnose this.

---

## 2. The timeouts are BUDGET-BOUND, and the budget was halved by someone who left no note

`leg_timeout_overridden: true`, `3600.0 → 1800.0`, with
**`leg_timeout_override_reason: ""` and `leg_timeout_override_source: ""`.**

The distribution of the seventeen legs that *finished*, against that 1800 s ceiling:

```
  1657.8s   92.1%  #######################################################
  1486.2s   82.6%  ##################################################
  1263.4s   70.2%  ##########################################
  1210.4s   67.2%  ########################################
  1118.1s   62.1%  #####################################
  1005.3s   55.9%  ##################################
     median 720.1s
```

**The slowest leg that finished had 142 seconds of headroom.** Against the *default* 3600 s ceiling
that same leg is **46.0%**.

That settles "ordinary stochastic timeout". A timeout is bad luck only when the ceiling is
comfortable for everything else; here the upper tail is pressed flat against it. **At this budget,
three timeouts is the expected outcome, not an anomaly** — and the halving that produced it is
recorded as a *fact* with its *justification* left empty. PRODUCT_CONTRACT § 9 requires per-leg
overrides to be *"explicit and recorded in the run manifest"*. The fact and the value are recorded;
the decision is not.

**The override shortens rather than extends**, so § 9's *"no silent extension of difficult benchmark
legs"* clause is not violated. Half the requirement is met. The half that is missing is the half
that would tell a successor whether 1800 s was a deliberate cost-control decision or an accident —
**and on this evidence that single unrecorded decision is the proximate cause of all three
timeouts.**

---

## 3. The sharpest finding: the preservation window was spent before it could be used

Both `budget_exhausted` legs carry:

```
finalization_reserve_seconds : 120.0
child_deadline_seconds       : 1680.0
elapsed_seconds              : 1800.47   /  1800.16
remaining_seconds            : -0.47     /  -0.16
```

The design is explicit and correct: the parent's leg ceiling is 1800 s, the child is handed
`--timeout 1680` (`runner.child_command` → `deadline.child_deadline_seconds`, 1800 − 120), and the
**120 s difference is a finalization reserve** — the window in which a leg that has run out of time
is supposed to write down what PRODUCT_CONTRACT § 9 requires it to preserve: last completed stage,
current structured payload, retrieved evidence, attempt numbers, elapsed and remaining budget, the
skipped recovery step, the exact stop reason.

**Both legs ran to 1800 s. The reserve was consumed to zero, and each overran the child deadline by
almost exactly the whole 120 s.**

So `files: []` and `counts: {}` **do not mean the pipeline produced nothing.** They mean the child
was killed while still working, with the window in which it would have written its checkpoint
already spent. The SUMMARY says *"the child process was still running after 1800s and was killed,
so this paper+mode produced nothing"* — the first clause is accurate and the second does not follow
from it.

> **This is the charter's "absence of a payload caused by cleanup rather than pipeline failure",
> and it is the correct classification for these two legs.**

### What is NOT yet proven, and the probe that would settle it

Three readings are all consistent with `elapsed = 1800.4` and none is excluded by the artifacts:

1. the child **never honoured** its 1680 s deadline;
2. the child **did** stop near 1680 s and its **finalization itself exceeded** the 120 s reserve;
3. the reserve is real but nothing on the outer-kill path **invokes** finalization at all.

`PMC12444477/strict` is a hint against reading 1: it fired an **in-process** `operation_timeout` at
**1798.3 s**, not at 1680 s, which suggests the in-process deadline is keyed to something other
than the child deadline the parent computed. **It is a hint, not a measurement**, and it is
recorded as one.

Settling this needs a **short, cheap, offline** probe of `deadline.py` and `runner.py` against a
synthetic child that overruns — **no LLM spend, no benchmark leg, no T-run.** It is not chartered
here: the classification the charter asked for is complete without it, and the fix belongs to
whoever owns the deadline seam.

---

## 4. Non-termination and pathological expansion — separated by paper

**`PMC12096016` is not non-terminating and is not pathological.** Its **research** leg finished in
1263.4 s (70.2% of ceiling) and produced 5 reactions, 19 entities, 6 proteins. The paper is
tractable and the pipeline terminates on it; the **strict** leg simply costs more than research and
ran past a halved ceiling. Its gold is `expected_export: strict_exportable`,
`min_connected_reactions: 4` — one of only two strict-denominator papers, **lost to the clock
rather than to biology**, which is part of why Priority 5 reads `0/2`.

**`PMC12444477` is UNPROVEN in both directions.** Both legs timed out, so nothing about its
termination behaviour was observed at all. Its gold notes *"The chemistry lives in Figure 1B, which
is not in the cached text"* — the hardest extraction in the set — which makes a long, expensive
extraction plausible. **Plausible is not measured**, and the triage's `!! RESEARCH-MODE DEFECT !!`
banner on this paper rests on a premise that does not hold: *"research mode is fail-open by design,
therefore any research failure is a code defect"* is a property of the **format-gate** path, and a
child killed by its parent at wall clock **cannot fail open**.

**This is also why `LpxH` is UNVERIFIED on T-107** and must not be reported otherwise: both
`PMC12444477` legs timed out with no payload to inspect. It remains verified at the merged tip on
the pinned run `runs/2026-08-02_2130`. **T-107 does not confirm it.**

---

## 5. Retry amplification — the one that cannot be excluded, and why that matters

`batch.log` (142 lines) contains **zero** occurrences of `retry`, `retrying`, `attempt`, `backoff`,
`rate limit` or `429`. On its face that excludes retry amplification.

**It does not.** The three timed-out legs preserved **no attempt record of any kind** — no
`attempts`, no `retries`, no per-call log. PRODUCT_CONTRACT § 9 requires *"attempt numbers,
prompts/models and response hashes"* to survive a timeout, and none did.

> **The artifact needed to rule out retry amplification is exactly the artifact the kill destroyed.**
> "No evidence of retries" here is not "evidence of no retries" — it is the absence of the
> instrument. Recorded as **unexcluded**, not as excluded.

This is the strongest practical argument for fixing § 3's preservation seam: the missing checkpoint
does not merely lose a payload, it **removes the ability to diagnose the next timeout**, and it did
so on the run whose Priority 5 depended on one of these legs.

---

## 6. Disposition

**No card is chartered from this document.** The charter asked for classification from committed
artifacts first, and that is what this is.

| Item | Disposition |
|---|---|
| The empty `leg_timeout_override_reason` / `_source` | **operational, not code.** The next benchmark run must record why its ceiling differs from 3600 s. Folded into the readiness table, not a card |
| The consumed finalization reserve (§ 3) | **the real F-148 defect and the one worth a card**, once the three readings in § 3 are separated by the cheap offline probe named there. Not chartered here — the deadline seam is owned by nobody in this wave |
| `stage=unknown` on the outer-kill path | **not a defect.** It is honest. Worth a comment so no future reader reads it as a pipeline stage |
| The two SUMMARY mislabels | already recorded in `T107-TRIAGE.md` § 9.5; § 4 above adds the reason the `RESEARCH-MODE DEFECT` premise fails |
| `LpxH` | **UNVERIFIED on T-107.** Verified on `runs/2026-08-02_2130`. Do not report T-107 as confirming it |

**T-107's verdict is untouched by any of this.** It remains `NOT ACCEPTED`. Nothing here rescores
it, and the classification of an operational failure is not a re-reading of a Priority result.
