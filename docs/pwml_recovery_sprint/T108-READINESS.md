# T-108 release-candidate readiness

**Status: `NO-GO` — 2026-09-02. See § 7.** The `GO` at `0bbac3fd` (§ 6) was correct and is **spent**: T-108 launched once under it and is scored **`NOT ACCEPTED`**. It is now NO-GO again because the **acceptance instrument itself is being corrected under D-088**, and a candidate scored mid-correction cannot be interpreted. The `NO-GO` at `848bc18b` (§ 5) is preserved as the record of what was true before this session, and it was the correct answer then. Blockers named below. This file is updated as conditions close;
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

> **This value was challenged and it SURVIVED — the challenge was the trap above, firing again.**
> REV-111 reported, in good faith, that `LLM_MAX_RETRIES` is **8** at base and tip and that this row
> was stale. It measured honestly and it measured **the wrong tree**. `client.py:484` reads
> `int(os.getenv("LLM_MAX_RETRIES", "8"))` — **8 is the CODE DEFAULT when the variable is unset** —
> and the primary checkout's `.env:34` sets **`LLM_MAX_RETRIES=3`**. A reviewer's worktree
> (`C:/t/rev111`) has **no `.env` at all**, so it saw the default.
>
> **T-108 runs from the primary checkout, so 3 is the operative value and this row is correct.**
>
> **The lesson is sharper than the number.** This is not a new trap; it is the trap this very
> section documents, and it caught a careful reviewer *reading this file* — because the trap does not
> announce itself. A worktree does not error on a missing `.env`; it silently substitutes defaults
> and every measurement taken in it is internally consistent. **Any claim about configuration
> measured in a worktree is a claim about the code's defaults, not about the run.** Verify config in
> the primary or not at all.
>
> Note the error ran in the *safe* direction here — 8 would mean more amplification headroom than 3,
> so acting on it would have overstated the risk rather than hidden it. **That is luck, not method.**

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

---

## 5. Readiness REBUILT at `848bc18b`, 2026-09-01 — **decision: NO-GO, on one row**

**Rebuilt against the then-current tip, not against § 1's remembered state.** Every row below was
re-derived this session. § 1's table is left standing as the record of what was true before this
wave.

### 5.1 The rows

| # | Condition | State | Evidence at this tip |
|---|---|---|---|
| 1 | F-155 merged and independently approved | **GREEN** | C-108 `2e2a294e`, REV-108 approved. Unchanged this wave |
| 2 | Mutation harness executable and green | **GREEN** | REV-112 ran the **whole** `c107` harness independently: **17 mutations, SURVIVORS 0, all RED, no ABORT**, target sha256 identical before and after. Run with `PYTHONDONTWRITEBYTECODE=1`, which is stronger than a pre-purge — F-160's failure mode is a *false* `MUTATION SURVIVED`, and an all-red zero-survivor result cannot be an F-160 artefact. **C-108 R4's abort is closed** |
| 3 | Census pin is in SMOKE | **GREEN** | `test_c102_coverage_denominator.py` is in the 22-file selection and passes inside 503 |
| 4 | F-146 remains rejected | **GREEN** | re-measured at this tip: **`F146=REJECTED`**. G11 `ORCH-718/12` |
| 5 | 29-case battery at zero mismatches | **GREEN** | re-measured at this tip: **`battery=0/29`**, `C1=0 C2=0 C3=0 C4=0 C5=0 C6=0`. **Unmoved after the gold change AND the production instrumentation** |
| 6 | Corpus movers understood in both directions | **GREEN** | 19 refused / 0 admitted, mover set stable. Closed by C-108 last wave |
| 7 | Negative controls scored per the Q1 ruling | **GREEN** | C-110 merged; `PASS_NEGATIVE_CONTROL` implemented, default-deny |
| 8 | Q2/Q3 decision merged and reviewed | **GREEN** | Q3 ruled, no code change. **Q2 half 1 is now MERGED** (C-113, REV-113 approved). Half 2 is an open product-owner question that is **explicitly not a launch blocker** — see § 5.3 |
| 9 | Every applied F-150 correction passed its independent A/B | **GREEN** | applied at C-113; the four-step A/B ran, and REV-113 re-derived it independently rather than accepting it |
| 10 | No absolute acceptance priority guaranteed to fail | **GREEN, was AT RISK** | Priority 5's risk was **operational**: `PMC12096016`, one of only two strict-denominator papers, was lost to the clock at the 1800 s ceiling. **§ 2.1 restores 3600 s**, which the census shows clears every observed finisher. It is no longer *guaranteed* to fail |
| 11 | Deterministic SMOKE + gold-reader gates green | **GREEN** | at the combined tip: SMOKE **503 / exit 0**, gold-readers **456 / 0 / 8 / 0 / exit 0** against gold `36f4b7b6`. G11 `ORCH-718/10`, `/11` |
| 12 | `acceptance.py` hashes identically before and after | **GREEN** | `4bd893ac410d16d3…` at every sample point all wave. **Note the form**: that is the CRLF working-tree hash; the LF blob is `d9f817e1…`. Quote which one you mean |
| 13 | Integration pushed and remotely verified | **GREEN** | `local = origin/ = git ls-remote` re-verified after **every** push this wave |
| 14 | Pinned 10-paper / 20-leg plan verifies offline | **GREEN, re-verify at launch** | unchanged. **Must still be re-verified inside T-108's own staged directory** — that is a launch step, not a pre-launch blocker |
| 15 | Configured provider and pinned model available | **GREEN** | verified without printing secrets: `LLM_PROVIDER=openrouter`, `LLM_TEMPERATURE=0`, all nine `OPENROUTER_*_MODEL` pinned to `deepseek/deepseek-v4-flash`, key present. **`LLM_MAX_RETRIES=3`** — see § 3's ratified note; the challenge to that value was the `.env` worktree trap |
| 16 | Heavy lock free | **GREEN** | `C:/t/heavylock` absent. Four strands this wave, all recovered; **F-163** |
| 17 | Zero sprint-owned Python | **GREEN** | only the two `ms-python.isort` IDE processes, matched on command line |
| 18 | No peer owns an overlapping live job | **GREEN** | `project14-t2pw-93` confirmed read-only: no branch, worktree, lock, job or edit |
| **19** | **Enough time to monitor or formally transfer the run** | **RED — THE ONLY BLOCKER** | see § 5.2 |

**The three blockers § 6 of the previous handoff named are all CLOSED:** F-150 is resolved and
merged · C-112 and C-111 are dispatched, reviewed and merged · the leg ceiling is deliberately
chosen and recorded.

### 5.2 The one red row, stated exactly

**Row 19 fails, and it fails on the "or" clause as much as the "and".**

The row requires *enough time to monitor* **or** *formally transfer*. Neither holds:

- **Monitoring.** T-105's comparable 20-leg run took **4.85 hours** at a *lower* ceiling. Restoring
  3600 s raises the worst case substantially. **This session cannot see a run of that length
  through.**
- **Transfer.** There is no recipient. The only live peer is `project14-t2pw-93`, which is
  read-only, working for a different user on an unrelated assessment, and has **not** been
  authorized for sprint work. **A transfer nobody has accepted is not a transfer.**

**And the compliance rule decides it independently of judgement.** `TEST_MATRIX` § 0 rule 1 permits
a tracked background job **only** when the orchestrator *"polls rather than launching duplicates"*
and **"no detached or unowned job remains."** A T-108 launched now becomes **unowned** the moment
this session ends. That is not a risk assessment — **it is a G11 violation by construction**, and
G11 is merge gate 11.

**Compounding it, and this is why waiting is cheap while launching is not:** T-108 is **one-shot**.
*"If T-108 fails, preserve it as a failed official release candidate and triage from committed
artifacts. Do NOT rerun it."* A later candidate needs a **new milestone identity** and a separately
recorded readiness decision. **Launching unmonitored risks burning the milestone identity on an
infrastructure failure nobody was watching** — the worst available outcome, and strictly worse than
launching tomorrow.

**Cost is not the constraint and was never considered.** There is no OpenRouter spending, token,
request or model-usage ceiling. **The blocker is ownership and observability, not money.**

### 5.3 Why half 2 is not a blocker, stated so nobody re-derives it

`supported_reactions_complete` is unset on all ten cases and **the status quo is honest**:
`semantic.py` stamps `UNSUPPORTED-REACTION VERDICT NOT EVALUATED` and withholds `false_positives`
rather than reporting a hard zero, which is what `PRODUCT_CONTRACT` § 11 requires. **T-108 can
launch under that.** What T-108 must **not** do is quote a Priority 2 number without the standing
limit attached:

> **Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry a run
> produced.**

The question is preserved unanswered in `DECISION-PACKET-F150-HALF2.md`. **Answering it is not a
prerequisite for launching; quoting Priority 2 without its limit is what would be wrong.**

### 5.4 What the next session must do — in this order

1. **Re-verify rows 4, 5, 11, 16, 17 at the then-current tip.** They are cheap and they are the ones
   that can silently rot. Do not carry these numbers forward from this table — **re-derive them.**
2. **Confirm row 19 is satisfiable for you** before anything else, because it is the only one that
   depends on the operator rather than the tree. **If you cannot monitor ~5–8 hours or hand the run
   to a named owner who has accepted it, T-108 stays NO-GO and that is the correct answer, not a
   failure.**
3. Then the launch protocol in § 4, unchanged, **at the 3600 s ceiling with no override**, verifying
   `leg_timeout_overridden: false` **in the staged directory before launch, not after**.

---

## 6. Readiness RE-DERIVED at `0bbac3fd`, 2026-09-01 — **decision: GO. T-108 LAUNCHED.**

**Row 19 is now GREEN, and it is the only row whose state changed.** Everything else was re-derived
rather than carried forward from § 5, per § 5.4 step 1.

### 6.1 Row 19 — why it closed

§ 5.2 recorded row 19 RED for a reason that was **about the operator, not the tree**: the session
could not see a ~5–8 h run through, and there was no peer authorized to accept a transfer.

**This session was convened specifically to own T-108 through completion** — instructed to remain
active and retain ownership until the wrapper exits, the run is scored, every process is closed, and
the result is committed and pushed. **That is exactly what row 19 asks for, and it is satisfied by
the monitoring arm of the "or", not the transfer arm.** No transfer is needed and none is claimed.

**The compliance objection dissolves with it.** § 5.2's decisive argument was that
`TEST_MATRIX` § 0 rule 1 permits tracked background **only** when *"no detached or unowned job
remains"*, so a T-108 launched by a session that would end mid-run is a G11 violation **by
construction**. A run owned end-to-end by its launching session is the case rule 1 explicitly
authorizes. `T108-RUN-OWNERSHIP.md` is the record.

**Exclusivity was established before anything was claimed**, not assumed: `ListAgents` showed
exactly one live peer in this repository, `project14-t2pw-93`, which stood down on all four points
(no orchestrator role, no branch/worktree/lock/job/edit, no T-108 claim, will not launch or clear the
lock while the run is live). **Its caveat that it cannot bind its user's future instructions is
recorded in the ownership file rather than smoothed away** — it commits to notify first and wait, and
unconditionally never to touch the lock. **One inherited fact was corrected by it: it is the same
user on a different task, not a different user.**

### 6.2 The rows § 5.4 required re-deriving — all re-measured, none carried forward

| # | Condition | State at `0bbac3fd` | Evidence |
|---|---|---|---|
| 4 | F-146 remains rejected | **GREEN** | `F146=REJECTED`, re-measured. G11 `T-108/03` |
| 5 | 29-case battery at zero mismatches | **GREEN** | `battery=0/29  C1..C6 all 0`. G11 `T-108/03` |
| 11 | Deterministic SMOKE + gold-reader gates | **GREEN** | SMOKE **503 / exit 0** (`T-108/01`); gold-readers **456 passed / 0 failed / 8 skipped / 0 errors / exit 0** (`T-108/02`), against gold `36f4b7b6` |
| 16 | Heavy lock free | **GREEN at launch**, now **held by T-108** | absent at every pre-launch sample; acquired `2026-09-01T22:14:52Z`, token `T-108:584068:06cdcf139f4c2b80` |
| 17 | Zero sprint-owned Python | **GREEN** | only the two `ms-python.isort` IDE processes, matched on **command line**, never on name. Independently corroborated by the peer, which flagged the same two PIDs unprompted |

**Also re-derived, because they are cheap and load-bearing:** row 12 — `acceptance.py` is
`sha256:4bd893ac410d16d3…` **before and after** SMOKE, byte-identical · row 13 —
`local = origin/ = git ls-remote` all `0bbac3fd`, `main` untouched at local `7531692` / remote
`03f1af5` · row 15 — provider preflight **OK** (`T-108/04`) · whole-tree G11 — **5032 artifacts, 0
non-compliant**, which reconciles exactly with the previous wave's count because nothing had been
added yet · C-111 / C-112 / C-113 all **ancestors** of the measured tip.

### 6.3 Row 14 discharged inside T-108's own staged directory, as it required

| Step | Result |
|---|---|
| Fresh milestone identity, **stage-only** | `runs_verify/2026-09-01_1612`, 2.19 s, **0 legs executed** |
| `--verify-plan` on that exact directory | **`verdict: OK`** · `cases checked: 10   search calls: 0` · **all 10 `[pinned_override]`** |
| Continuity proof | `find_resumable()` returned **that exact path**; **20 pairs, 20 pending, 0 recorded, 0 leg directories, 0 `RESULT.txt`** |
| Gold identity in the staged tree | working-tree blob **=** HEAD blob **=** pinned `36f4b7b6…`, version `2026-08-01.1`, 10 cases |
| Leg ceiling **before** launch | `_ceiling(3600.0)` → `leg_timeout_overridden: **False**`, **no** `leg_timeout_override_reason` key emitted. Child deadline `3480 s` |
| Continue **without** `--fresh` | runner confirmed `CONTINUING the incomplete run 2026-09-01_1612 (no --fresh given)` · `already recorded : 0` · `still to do : 20` |

**The runner's own hint remains the trap § 4 step 6 exists to defeat** — *"rerun the same command
WITHOUT --stage-only"* still carries `--fresh`, which would have discarded the staging just
certified. The discriminator is `already recorded : 0`, asserted directly rather than through the
`--fresh` proxy.

### 6.4 The one environment decision that is not a repeat of T-107's paperwork

**`T2PW_OFFLINE_CURATOR` is deliberately NOT set for the live run**, against the sprint's blanket
rule, and the exception is documented in `T108-MANIFEST.md` § 5. § 3 of this file requires the flag
on sprint jobs; **that rule governs deterministic test and gate jobs.** `run_pathway_curator` is a
ratified production stage using the pinned `OPENROUTER_CURATOR_MODEL`, and the flag makes it *"an
explicit, deterministic no-op: zero model calls"*. Setting it would disable a ratified stage — the
`LLM_PROVIDER=local` failure mode wearing different clothes.

**It was settled by measurement, not by argument.** T-107's own committed artifact
(`runs_verify/2026-08-28_1816/papers/PMC12096016/strict/RESULT.txt:68`) records a live curator call
on the pinned model. **T-107 ran the curator online; T-108 matches it, so comparability holds.** The
same line reads `on attempt 1/3`, which **independently confirms `LLM_MAX_RETRIES=3` on the real
run** — § 3's `.env` trap closed by observation of the production path rather than by re-reading the
file that documents it.

### 6.5 What is NOT claimed

**This is a launch decision, not a result.** T-108's verdict is whatever its artifacts say. Nothing
here predicts it, and **no T-108 outcome re-opens T-107**, whose `NOT ACCEPTED` verdict is a fact
about the artifacts it produced.

**T-108 is ONE-SHOT and was launched exactly once.** It is not re-run for a timeout, stochastic
composition, an unexpected count, a seven instead of a six, a failed acceptance priority, or missing
model-usage telemetry.

**A timeout at 3600 s is not automatically a defect and must not be waved away either** — § 2.1's
censoring limit stands, and any T-108 timeout is new information about the requirement.

---

## 7. CLOSED, then REOPENED as `NO-GO` — 2026-09-02, after T-108 ran and after D-088

**§ 6's `GO` was correct and it is spent.** T-108 launched once under it, ran 20/20 legs, and is
scored: **`NOT ACCEPTED`** — `T108-RESULT.md`. **T-108 is immutable and is not re-run or re-scored.**

### 7.1 Status now: **`NO-GO` for any further release-candidate launch**

**A new candidate needs a NEW milestone identity and a separately recorded readiness decision.** This
file's § 1, § 5 and § 6 tables are the record of three earlier assessments and are **not** a standing
authorisation for anything.

**The blocker is no longer operational.** Every row § 5.4 flagged as rot-prone was re-derived green at
launch, the run completed cleanly, and run ownership held end to end. **The blocker is that the
acceptance instrument itself is being corrected** — see § 7.2.

### 7.2 The F-167 question is RESOLVED — **D-088** — and the correction is not yet built

**F-167 is no longer open.** It asked whether a requested core legitimately includes cofactors and
regulators; **the product owner has ruled (D-088, 2026-09-02)**:

> **The pipeline's primary goal is to recover the paper's important pathway reactions as correctly as
> possible. It is not required to achieve perfect participant-level biochemical completeness.**

**Hard completeness decisions move to validated reactions and major subprocesses**, and flat Stage-0
`key_compounds` / `key_proteins` stop being automatic hard release requirements. **This supersedes
the assumption that every requested-core entity must match an admitted process for release.**

**Nothing is implemented. No production, scorer, test or gold file has changed.** The correction is
chartered as **one narrow acceptance/release-policy card** in `HANDOFF.md` § 5.2a.

### 7.3 Why that makes T-108-class launches NO-GO right now

**A release candidate scored on an instrument that is mid-correction cannot be interpreted.** Under
D-088 the pass/fail boundary for Priorities 4 and 5 is about to move by design, so a run launched
before the card lands would be measured by a rule already ruled wrong, and a run launched during
would be measured by neither rule cleanly. **`HANDOFF.md` § 5.2a step 9 is explicit: launch only
after the new reaction-focused instrument is reviewed, merged, gated and remotely verified.**

**This is not the row-19 blocker of § 5.2.** That one was about the operator; this one is about the
instrument. **Run ownership is a solved problem here** — `T108-RUN-OWNERSHIP.md` records the pattern
that worked.

### 7.4 What a rebuilt readiness table must additionally prove

Beyond the nineteen rows, D-088 adds obligations that a future `GO` must carry:

| Obligation | Source |
|---|---|
| Expected core reactions / major subprocesses **defined or curated** for all ten papers | D-088 clause 9 — the cap needs a replacement input before the old one is removed |
| Archived-artifact **A/B across all 83 committed legs** | `HANDOFF.md` § 5.2a step 5 |
| `PMC12096016` loses **only** the false entity-anchor cap | D-088 expected consequences |
| `PMC12782028` **remains** a reaction-recall failure | D-088 expected consequences |
| The **60** subprocess-aligned and **90** payload-unwired anchors remain **separately visible** | D-088 clauses 7 and 8 |
| **No gold-forbidden content becomes releasable** because entity anchors were downgraded | D-088 expected consequences |

**Both named consequences must hold simultaneously.** A correction that clears `PMC12096016` **and**
`PMC12782028` has removed the measurement rather than improved recall, and is a reject.

### 7.5 What carries forward unchanged from the T-108 launch

The § 2.1 leg-ceiling ruling **stands at 3600 s with no override** and is now backed by a live
result: timeouts **3 → 1**, scorable denominator **17 → 19**, `PMC12096016/strict` recovered at
**108.5%** of the old ceiling. **F-166** records the limit — one leg consumed the full 3600 s and
still timed out, so the ceiling is sufficient for the corpus and not for every leg in it. **No
ceiling change is proposed on one observation.**
