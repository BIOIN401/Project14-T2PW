# T-108 release-candidate readiness

**Status at last update: `NO-GO`.** Blockers named below. This file is updated as conditions close;
**a `GO` requires every row green, verified at the integration tip, not remembered.**

**T-108 is a NEW milestone identity.** It is not a re-run, re-score or re-reading of T-107.
**T-107's verdict is `NOT ACCEPTED` and is a fact about the artifacts it produced.** No card merged
after it re-accepts it, and no T-108 result may be reported as confirming or overturning it — **a
T-108 result is reported independently.**

---

## 1. The gate

| # | Condition | State | Evidence / blocker |
|---|---|---|---|
| 1 | F-155 merged and independently approved | **BLOCKED** | C-108 dispatched, worktree `C:/t/c108`. REV-108 criteria committed before the diff existed |
| 2 | Mutation harness executable and green | unverified this wave | C-106 restored it (`fa69c57`); re-verify at the merge tip |
| 3 | Census pin is in SMOKE | unverified this wave | C-106 put the file in a gate; re-verify |
| 4 | F-146 remains rejected | **GREEN** | measured by me at `f67e00a`: `F146=REJECTED`. G11 `ORCH-717/13` |
| 5 | 29-case battery at zero mismatches | **GREEN** | measured by me: `battery=0/29`. G11 `ORCH-717/13` |
| 6 | Corpus movers understood in both directions | **BLOCKED** | baseline taken (692 rows / 401 accepted / 291 refused, G11 `ORCH-717/14`); tip pending C-108 |
| 7 | Negative controls scored per the Q1 ruling | **BLOCKED** | C-110 chartered, not dispatched. Scorer **cannot** represent the ruling today — `RULINGS-ORCH717.md` § Q1 |
| 8 | Q2/Q3 decision merged and reviewed | **PARTIAL** | **Q3 ruled: no code change required** (§ Q3). Q2 half 1 pending Wave 4; half 2 is an open product-owner question |
| 9 | Every applied F-150 correction passed its independent A/B | n/a — **none applied** | gold file unmodified |
| 10 | No absolute acceptance priority guaranteed to fail | **AT RISK — see § 2** | Priority 5 read `0/2` on T-107 with one of the two strict-denominator papers lost to the clock |
| 11 | Deterministic SMOKE + gold-reader gates green | unverified this wave | SMOKE **503 / 22 files**; gold readers **456 passed / 8 skipped / exit 0** |
| 12 | `acceptance.py` hashes identically before and after SMOKE | **must be checked every run** | SMOKE writes to the tracked file and restores it in a `finally`. A changed hash is an **infrastructure failure even if pytest is green** |
| 13 | Integration pushed and remotely verified | **GREEN** | `local = origin = git ls-remote` re-verified after every push this wave |
| 14 | Pinned 10-paper / 20-leg plan verifies offline | **GREEN at T-107's plan** | `scripts/bench_acceptance.py --verify-plan` → `verdict: OK`, 10 cases, **0 search calls**, all `[pinned_override]`. **Must be re-verified inside T-108's own staged directory** |
| 15 | Configured provider and pinned model available | **GREEN, with a trap — § 3** | `.env` present; `LLM_PROVIDER=openrouter`, `LLM_TEMPERATURE=0`, full `OPENROUTER_*_MODEL` set |
| 16 | Heavy lock free | transient | free at last check; two cards will contend for it |
| 17 | Zero sprint-owned Python | **NO — by design** | two implementers active. Must be zero at launch |
| 18 | No peer owns an overlapping live job | **GREEN** | `project14-t2pw-93` replied **"no claims"**: no branch, worktree, lock, job or edit |
| 19 | Enough time to monitor or formally transfer the run | assess at launch | T-104–T-107 durations set the wrapper timeout |

---

## 2. The readiness risk nobody has closed — the leg ceiling

**This is the row most likely to make a green table produce a failed run.**

F-148 measured T-107's timeouts as **budget-bound, not stochastic**:

```
leg_timeout_overridden : true    3600.0 -> 1800.0
leg_timeout_override_reason : ""     leg_timeout_override_source : ""
slowest leg that FINISHED : 1657.8s = 92.1% of the 1800s ceiling  (142s headroom)
                                    = 46.0% of the default 3600s ceiling
```

**At that budget three timeouts is the expected outcome, not an anomaly**, and the halving that
produced it is recorded as a fact with its justification left blank. PRODUCT_CONTRACT § 9 requires
per-leg overrides to be *"explicit and recorded in the run manifest"* — the fact and the value are
recorded, **the decision is not**.

It bites Priority 5 directly: `PMC12096016` is **one of only two strict-denominator papers**, and its
strict leg was **lost to the clock rather than to biology**. If T-108 runs at 1800 s and the same leg
times out, Priority 5 can read `0/2` again for a reason that is **operational, not biological** — and
would be misread as a pipeline failure.

**F-148 § 6 disposition: this is operational, not code.** It belongs here, not in a card.

> **Before T-108 launches, the leg ceiling must be chosen deliberately and its reason recorded in the
> run manifest.** Either restore 3600 s, or record explicitly why a shorter ceiling is being accepted
> and that timeouts are therefore an expected cost. **Do not launch at 1800 s with an empty
> `leg_timeout_override_reason` and then classify the timeouts afterwards.**

### 2.1 RULED by the Lead, 2026-09-01 — restore the 3600 s default. This row is now GREEN.

**The decision is made and recorded here BEFORE launch, which is what the row asked for.**

Measured first, decided second. `evidence/orch718_leg_duration_census.py` / `.log`, G11
`ORCH-718/02`: a read-only census of **every** committed leg duration under **both** `runs/` and
`runs_verify/` — named explicitly, because both roots are live and "the pinned run" is ambiguous.
**No leg was run, re-run, re-scored or mutated. T-107 remains immutable and its verdict is
untouched:** a duration is a fact about how long a process ran, never a verdict about what it
produced.

```
POOLED, finished legs only, every tree
  n=192   min=11.0   median=927.9   p90=1609.0   max=3421.4
  slowest finisher anywhere : 3421.4 s in runs/2026-07-28_2122

  1800s ceiling -> the slowest observed finisher needs 190.1% of it
  2400s ceiling ->                                    142.6%
  3000s ceiling ->                                    114.0%
  3600s ceiling ->                                     95.0%   (179 s headroom)
```

**Three findings settle it:**

1. **A leg has demonstrably needed 3421.4 s.** Every ceiling below 3600 s is *known-insufficient by
   direct measurement*, not by extrapolation. 1800 s is **less than half** the observed requirement.
2. **p90 is 1609 s — 89.4% of an 1800 s ceiling.** Roughly a tenth of all legs finish within 11% of
   that budget. **That is not a ceiling that occasionally times out; it is a ceiling that times out
   by construction.**
3. **The 1800 s ceiling has produced timeouts in FOUR separate run trees, not just T-107** — with
   the slowest finisher using 69.4%, 75.8%, 80.8% and 92.1% of budget respectively, and two to three
   timeouts each time. T-107's three timeouts were never an anomaly; they were the fourth
   observation of a repeating pattern.

**Restoring the default also dissolves the contract problem rather than papering over it.** At
3600 s there is **no override**, so `leg_timeout_overridden` is `false` and there is no
`leg_timeout_override_reason` left empty for PRODUCT_CONTRACT § 9 to catch. **The honest way to
satisfy a "record your reason" rule is to stop needing an exception.**

**Two limits recorded with the decision, because a ceiling chosen on observed maxima is a ceiling
chosen on censored data:**

- **Every timed-out leg is CENSORED.** It proves the work needed *more* than the ceiling and never
  *how much more*. The true requirement is **at least** 3421.4 s and may be larger. 3600 s clears
  the slowest *observed* finisher by **179 s — 5%**. That is thin.
- **A timeout at 3600 s is therefore NOT automatically a defect**, and must not be reported as one
  without evidence. Equally, it must not be waved away: if T-108 times out at 3600 s, that is new
  information about the requirement and belongs in the run report as such.

**Registered for `FINDINGS.md` as soon as C-112 releases the file** (C-112 owns `FINDINGS.md` in the
current wave and a concurrent edit would conflict): *the 1800 s ceiling is a recurring
infrastructure cause of leg loss across four runs, and Priority 5's `0/2` on T-107 is partly
attributable to it rather than to biology.* **Nothing is lost by the delay — the measurement, its
probe, its correction and this ruling are all committed here.**

**Probe correction preserved, per the sprint rule.** Attempt 1 classified a **crashed** leg as
"finished" and swept in all 56 legs of `runs/2026-07-27_2135`, every one of which is
`ModuleNotFoundError: No module named 'streamlit'` at 0.0 s — **F-143 itself, preserved in a
committed run tree**: a bare `python` outside the venv. It is an infrastructure failure, not a
duration. Excluding crashes moved **n 250 → 192**, **median 734.1 → 927.9** and **p90 1534.1 →
1609.0**, and left the **max unchanged at 3421.4**. **The distribution was wrong and the ceiling
conclusion was not.** Attempt 1's log is kept beside the corrected one as
`orch718_leg_duration_census.attempt1-crashes-counted-as-finished.log`.

**Launch obligation:** T-108 runs at the **3600 s default with no override**, and the run manifest
must show `leg_timeout_overridden: false`. **Verify that in the staged directory before launch, not
after.** The wrapper timeout is chosen separately, from these same durations plus cleanup headroom.

---

## 3. Traps that have already cost this sprint a false result

**`.env` is untracked, so a WORKTREE silently gets `LLM_PROVIDER=local`** — the call 400s, the
exception is swallowed, and the curator becomes a no-op **by accident** — while the primary checkout
issues real billed calls. **A green cohort obtained in a worktree does NOT certify the same cohort in
the primary.** T-108 runs from the **primary checkout**.

**`T2PW_OFFLINE_CURATOR=1` must be set in the BOUNDED CHILD environment, not just the shell.**
Without it `run_pathway_curator` issues one ungated LLM call per post-pipeline app run at temperature
0.2, whose accepted patches are written back into `audited_json` and flow through mapping into
`final_mapped_db`. **That is the measured root cause of BL-003.**

**`LLM_MAX_RETRIES=3`** — retries are configured, not disabled. This does not prove retry
amplification occurred on T-107, but it is why F-148 records it as **unexcluded rather than
excluded**: the mechanism exists and the artifact that would have measured it was destroyed by the
kill.

**A bare `python` is system 3.13 with no `streamlit`** → 35 spurious import errors that read exactly
like a regression (F-143). Always the explicit venv interpreter.

---

## 4. Launch protocol, once and only once every row is green

1. Fresh **T-108 milestone identity**.
2. The same ratified **10-paper / 20-leg** plan.
3. **Stage-only preflight.**
4. Verify the plan **and the gold** inside that exact staged directory.
5. Require **all pinned overrides and zero search calls**.
6. Continue the verified directory **without `--fresh`**.
7. Configured **pinned OpenRouter models**.
8. **One** run, through the bounded wrapper.
9. Background, with **explicit ownership**.
10. Wrapper timeout chosen from **measured T-104–T-107 durations**, with cleanup headroom.
11. **Monitor the existing wrapper — never launch a duplicate.**
12. Score the **first valid official draw**, honestly.
13. Preserve **raw and contract-adjusted** results separately.
14. **Do not rerun it to improve stochastic composition.**

**There is no OpenRouter usage ceiling. Cost must not restrict justified work.**

### What T-108 must report

20-leg completion and scorable denominator · every timeout and missing payload · Priority 1 raw and
accepted counts **and composition** · Priority 2 eligible denominator and unsupported retained
reactions · Priority 3 referential-integrity failures · Priority 4/5 raw and accepted coverage ·
negative-control outcomes · every applied policy adjustment · whether the result is accepted · exact
evidence paths · model/provider provenance · usage and cost where available.

**If T-108 fails it is preserved as a failed official release candidate and triaged from committed
artifacts. It is NOT rerun.** A later candidate needs a new milestone identity and a separately
recorded readiness decision.

---

## 5. Two numbers that must carry their limits into any T-108 report

**Priority 2 is evaluable only through `max_retained_reactions`**, which is set on exactly two gold
cases, **both negative controls**. `supported_reactions_complete` is `False` on **all ten**.

> **Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry a run
> produced.**

**`LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out with no payload to inspect. It
is verified only on the pinned run `runs/2026-08-02_2130`. **Do not report any T-107 result as
confirming it**, and do not carry the claim into T-108 unmeasured.
