# T-107 READINESS — assessed 2026-08-28 at integration tip `ad62338`

**VERDICT: ~~NO-GO~~ → GO, as of D-085 (2026-08-28).** The product owner authorized paid models on the pinned configuration, clearing condition 8 — the single failing condition. **The analysis below is preserved unedited as the record of why the question was asked**; only this line and § 6 are added. One condition fails, and it fails for a reason no engineering in this sprint can
clear: **the pinned model is not available on the pinned terms.** Everything else that was blocking
T-107 at the start of this wave is now cleared.

All figures below are **measured at the merged tip**, not quoted from a prior run's record. Evidence:
`evidence/g11/ORCH-714/01`–`03`, scoring `runs_verify/2026-08-24_1428` (the most recent complete
10-paper / 20-leg run) and `runs/2026-08-02_2130` (the pinned run) through the merged
`bench.acceptance`.

---

## 1. The acceptance table

| Priority | Raw | Contract-accepted | Status | Evaluated denominator | Remaining limitation |
|---|---:|---:|---|---|---|
| **1** — zero known false real identifiers | **8** | **8** | **`FAIL`** | 20 / 20 legs | Draw-variable. T-104 = 7, T-105 = 7, this run = 8. **8 is `FAIL` under D-073; 7 would be `PASS_WITHIN_VARIANCE`** |
| **2** — zero unsupported retained reactions | 0 counted | — | **`CONDITIONALLY SATISFIED`** | **9 of 20 legs eligible** | 11 legs `NOT EVALUATED`, all carrying the documented D-067 reason. **May not be reported as full 20-leg biological validation** |
| **3** — zero referential-integrity violations | 0 | — | **`PASS`** | 20 / 20 legs | none |
| **4** — meaningful requested-pathway coverage | `0/8 = 0%` | `0/8 = 0%` | `FAIL` (not a hard gate) | 10 legs carrying coverage | **Ruling A does not move it** — see D-081 |
| **5** — strict PWML pass rate | `0/2 = 0%` | `0/2 = 0%` | `FAIL` (not a hard gate) | 2 eligible legs | as above |

### Priority 1 — complete row composition, raw and accepted

| Paper | Mode | Count | Rows |
|---|---|---:|---|
| PMC12856317 | strict | 1 | `Pyridoxal 5'-phosphate` |
| PMC12856317 | research | 1 | `Pyridoxal 5'-phosphate` |
| PMC12096016 | research | 2 | `NAD+`, `NADH` |
| PMC12782028 | research | 4 | `LBR`, `LIPA`, `SREBF1`, `SREBF2` |
| | | **8** | |

**Accepted == raw == 8**, and that is a **measurement, not a construction** — see **D-077**. Under
D-074 as ruled, no Priority-1 row can be contract-adjusted at all: D-074 licenses only the *bare*
sentinel, and a bare sentinel can never *be* a Priority-1 row because that branch requires an
external accession. **Case-specific tolerances applied: none, because none can apply.**

Note the four `PMC12782028` rows are exactly the four the F-132 bundle named as *simultaneously*
Priority-1 survivors and Priority-4/5 coverage penalties. After C-102 they are **excluded from the
accepted coverage denominator while still counted under Priority 1** — which is precisely what D-072
required, now visible in the data.

### `LpxH` — confirmed still counted

Measured on the pinned run at the merged tip, `PMC12444477/strict`:

```
placeholder findings = 8  ["LpxA","LpxB","LpxD","LpxH","LpxK","LpxL","LpxM","WaaA"]
LpxH present : True        Unknown absent : True
census: sentinel_rows=1  wrappers=8  other=0
```

**9 → 8, never 9 → 7.** D-074's constraint holds at the merged tip.

### Priority 2 — the D-075 determination, checked rather than asserted

* `NOT EVALUATED` legs: **11**, all with a recorded reason; **every one** is the same class —
  `supported_reactions_complete` is false, i.e. exhaustiveness unproven, which is **D-067
  precondition 3** (*"the biological completeness has been independently reviewed"*).
* Legs: `PMC12096016:{research,strict}`, `PMC12421875:{research,strict}`,
  `PMC12444477:{research,strict}`, `PMC12452463:{research,strict}`,
  `PMC12657337:{research,strict}`, `PMC12782028:strict`.
* Eligible legs: **9**. Unsupported reactions counted on them: **0** → every eligible leg passes.

**→ `CONDITIONALLY SATISFIED` under D-075.** The report must state the evaluated denominator and
**may not claim full 20-leg biological validation.**

### Priorities 4/5 — raw and accepted after Ruling A

`legs_with_coverage: 10` · `legs_with_forbidden_terms: 7` · `forbidden_terms_excluded: 19` ·
**`legs_cleared_by_reconciliation: []`** · `legs_still_below_minimum: ["PMC12782028:strict"]` ·
`legs_with_undefined_accepted_rate: []`.

**Neither priority moves**, at base or tip, on any of the 21 run directories. `PMC12782028/strict`
goes `0.2222 → 0.2609` and still fails `0.500`. **D-081** records this against the bundle's explicit
prediction that Priority 4 would move off `0/8`.

---

## 2. Gate conditions

| # | Condition | State |
|---|---|---|
| 1 | C-101 merged and independently approved | ✅ `ee7cb6b`, REV-101 **APPROVE** after 3 correction rounds |
| 2 | Ruling-A card merged and independently approved | ✅ `8e4334f`, REV-102 **APPROVE** after 1 correction round |
| 3 | Rulings B, C, D recorded accurately | ✅ D-073, D-074, D-075 |
| 4 | Glutathione red classified, not a guaranteed acceptance failure | ✅ **F-142** — stale expectation, not a production defect; cannot affect T-107 |
| 5 | No absolute priority guaranteed to fail | ✅ P1 not *guaranteed* (7 achieved twice) · P2 `CONDITIONALLY SATISFIED` · P3 `PASS` |
| 6 | Deterministic gates green | ✅ SMOKE **473** post-merge on the combined tree. Gold-readers carries only the two F-142 reds (**C-103 in flight to clear them**) |
| 7 | Integration pushed and remotely verified | ✅ local = `origin/` = `ls-remote` after every push |
| 8 | **LM Studio and the pinned model available** | ❌ **FAILS — see § 3** |
| 9 | Heavy lock free | ✅ absent |
| 10 | Zero sprint-owned Python | ✅ only the two IDE `isort` servers |
| 11 | No peer session owns an overlapping live job | ✅ sole peer stood down explicitly |
| 12 | Run monitorable through completion | ✅ feasible — ~5 h wall clock, context ample |
| 13 | External spend within **$5** | ⚠ **unverifiable in advance; see § 3** |

**Twelve of thirteen hold. Condition 8 fails and condition 13 cannot be discharged.**

---

## 3. The blocker, exactly

**The pinned model is `deepseek/deepseek-v4-flash` on OpenRouter — not an LM Studio model, and not
free.**

| Fact | Measured |
|---|---|
| `.env` `LLM_PROVIDER` | `openrouter` |
| All **nine** OpenRouter slots | `deepseek/deepseek-v4-flash` |
| LM Studio (`localhost:1234`) serves | `zai-org/glm-4.6v-flash`, `text-embedding-nomic-embed-text-v1.5` |
| `.env` `LOCAL_MODEL` | `meta-llama-3.1-8b-instruct` — **which LM Studio is not serving either** |
| `deepseek/deepseek-v4-flash` pricing (public read-only `GET /api/v1/models`) | **prompt `$0.0868`/M · completion `$0.1736`/M — paid** |

**Why this is a hard NO-GO rather than a budgeting question:**

1. **The ≈$0 cost basis is void.** `T101_T103_AUTHORIZATION.md` and D-057's authorization rested on
   *"every OpenRouter model slot in `.env` is set to `openrouter/free`"*, confirmed at the time by a
   read-only models check. **`.env` no longer matches that**, and `.env` is untracked — the change is
   real and unattributable through git.
2. **LM Studio cannot substitute.** It serves neither the pinned OpenRouter model nor the configured
   `LOCAL_MODEL`. Running T-107 on `glm-4.6v-flash` would be **a fallback model, which this wave
   explicitly forbids**, and would destroy comparability with T-104, T-105 and T-106 — all three ran
   `deepseek-v4-flash`. A T-107 that cannot be compared to its predecessors is not the pinned plan.
3. **Spend cannot be bounded in advance.** T-105 ran **4.85 h** over 20 legs with nine model roles,
   RAG, and extraction ladders on a 1 M-context model. A defensible order-of-magnitude estimate is
   **$1–3**, but there is no spend telemetry to abort on, the variance is real, and the ceiling is
   **hard**. Committing a 5-hour unattended run to an unverifiable projection against a hard ceiling
   is not a risk I may take on the product owner's behalf.
4. **Changing `.env` is product-owner state.** Re-pinning the model or switching provider to make
   T-107 runnable is a product decision, not an orchestration one.

**Nothing in this sprint's engineering can clear condition 8.** It is a configuration and
authorization question.

---

## 4. The smallest next action

**One product-owner decision, with two coherent options:**

* **(A) Authorize paid spend for T-107 on `deepseek/deepseek-v4-flash`**, with an explicit ceiling
  (the current $5 is plausibly adequate but unproven) and, ideally, a spend check partway. This keeps
  T-107 **comparable** to T-104/T-105/T-106, which is the entire point of the pinned plan.
* **(B) Re-pin the plan to a free model** and accept, in writing, that T-107's numbers are **not
  comparable** to its three predecessors — which materially weakens what T-107 is for.

**(A) is the better option** if any paid spend is acceptable at all, because comparability is the
deliverable. (B) produces a number that cannot be read against the baseline it exists to move.

**Do not run T-107 until this is ruled.** A run started on the wrong model cannot be undone —
condition: *"must be launched only once … must not be rerun because of stochastic composition."*

---

## 5. What changed this wave, for the record

T-107 was NO-GO at the start of this wave too, but **for a different and worse reason**: the previous
record held that gate condition 9 was *"not met and not reachable by any engineering in this
sprint"*, because only ask B of the F-132 bundle could clear it and B was unanswered.

**B is now answered (D-073), and so is D (D-075).** Priorities 1, 2 and 3 are all in an admissible
state, both instrument cards are merged with independent approval, and the deterministic gates are
green. **The remaining blocker is not about the pipeline, the instrument, or the acceptance criteria
at all — it is that the model the plan pins is not available on the terms the plan was authorized
under.**

That is a much better place to be blocked, and it is one decision away from resolution rather than a
sprint's worth of engineering.


---

## 6. RESOLVED — D-085 clears condition 8

**The product owner ruled: paid models are fine for T-107.** That is option (A) of § 4, and it is the
better one, because it preserves comparability with T-104/T-105/T-106.

| | |
|---|---|
| Condition 8 — pinned model available | ✅ **cleared by D-085.** Run on `deepseek/deepseek-v4-flash` as `.env` already configures it. **Do not edit `.env`. Do not use LM Studio for T-107.** |
| Condition 13 — spend within **$5** | ⚠ **still binding.** Projected ~$1–3 at T-105's scale, with real variance and no spend telemetry. **Exceed it and stop.** |

**Everything else in § 1's acceptance table stands as measured** and is the pre-run expectation, not
a prediction of T-107's own draw. Priority 1 is genuinely uncertain between `PASS_WITHIN_VARIANCE`
(7) and `FAIL` (8+): T-104 = 7, T-105 = 7, the T-106 artifacts re-scored at the merged tip = 8.
**Score the first valid draw honestly and do not rerun it.**

---

## 7. D-080 RATIFIED — the accepted-coverage definition is locked

**Product-owner ratification, 2026-08-28.** The interpretation D-080 recorded is now the ruling:

```
eligible_anchors     = raw_anchors − case_scoped_forbidden_identifiers
accepted_numerator   = | matched ∩ eligible_anchors |
accepted_denominator = | eligible_anchors |
accepted_coverage    = accepted_numerator / accepted_denominator
```

Raw numerator, denominator and coverage are preserved as **separate raw diagnostic values**.
**C-102 implements this exactly — verified against the shipped code — and no production change was
required.** Priorities 4/5 in § 1's table are already computed on this definition, so **the table
stands unchanged**.
