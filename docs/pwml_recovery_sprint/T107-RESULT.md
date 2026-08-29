# T-107 — OFFICIAL RESULT · `runs_verify/2026-08-28_1816`

**Run once. This is the first valid official draw and it is preserved and scored as-is.** It is not
re-run, and no leg is repeated. Scored at integration tip `e5ee620`, with C-101, C-102 and C-103
merged.

| | |
|---|---|
| Started / finished | 2026-08-28 18:16 → 2026-08-29 02:17 |
| Wall clock | **5.63 h** (T-104 5.44 h, T-105 4.85 h) |
| Model | `deepseek/deepseek-v4-flash`, all nine slots, `LLM_TEMPERATURE=0`, `.env` unmodified |
| Legs | **20/20 attempted, 10/10 papers** · **17 scorable** (3 timeouts produced no payload) |
| Process | every job bounded · `FINAL SURVIVING COUNT : 0` · `cleanup : success` · heavy lock acquired and released |
| Plan | `verdict: OK`, `search calls: 0`, all 10 `[pinned_override]`, verified against this exact staged directory before launch |

---

## 1. The acceptance table

| Priority | Raw | Contract-accepted | Status | Evaluated denominator | Limitation |
|---|---:|---:|---|---|---|
| **1** — zero known false real identifiers | **5** | **5** | **`PASS`** | 17 scorable legs | First result under 6 this sprint |
| **2** — zero unsupported retained reactions | **1** | — | **`FAIL`** | **6 of 17 legs eligible** | An **eligible** leg failed. Not `CONDITIONALLY SATISFIED` |
| **3** — zero referential-integrity violations | **0** | — | **`PASS`** | 17 scorable legs | none |
| **4** — meaningful requested-pathway coverage | `0/7` | `0/7` | `FAIL` (not a hard gate) | 10 legs with coverage | unmoved, as D-081 predicted |
| **5** — strict PWML pass rate | `0/2` | `0/2` | `FAIL` (not a hard gate) | 2 eligible legs | unmoved |

### Overall: **NOT ACCEPTED.** Priority 2 is absolute and it failed.

Priorities 1 and 3 pass. **Priority 2 is an absolute priority with a non-zero count on a leg whose
verdict was actually reached**, so D-075's `CONDITIONALLY SATISFIED` does not apply — that status is
available only when *every eligible leg passes*. D-075 says it plainly: **"An eligible leg that fails
remains a failure."**

---

## 2. Priority 1 — `PASS` at 5, and the composition

| Paper | Mode | Count | Rows |
|---|---|---:|---|
| PMC12856317 | research | 1 | `Pyridoxal 5'-phosphate` |
| PMC12782028 | research | 4 | `LBR`, `LIPA`, `SREBF1`, `SREBF2` |
| | | **5** | |

**Both contributing legs are research. Every strict leg contributed zero.**

**Status `PASS` under D-073** (0–6 `PASS`, 7 `PASS_WITHIN_VARIANCE`, 8+ `FAIL`). The variance band
was not needed: 5 is inside the target, not inside the tolerance.

**`accepted == raw == 5`, and that is structural, not a coincidence** — under D-074 as ruled, no
Priority-1 row can be contract-adjusted at all, because D-074 licenses only the *bare* sentinel and a
bare sentinel can never *be* a Priority-1 row (**D-077**). **Case-specific tolerances applied: none,
because none can apply.**

### Trajectory

| Run | Priority 1 |
|---|---:|
| T-104 | 7 |
| T-105 | 7 |
| T-106 artifacts, re-scored at the merged tip | 8 |
| **T-107** | **5** |

**This is the first Priority-1 result under 6 in the sprint.** It is one draw. The same four
`PMC12782028` rows that appear here also appeared at T-104 and in the T-106 re-score, so the
persistent core is stable; what moved is the surrounding population.

---

## 3. ⚠ `LpxH` could NOT be verified on this run

**Both `PMC12444477` legs TIMED OUT**, so the paper produced no scorable payload and its placeholder
findings do not exist to inspect.

D-074's constraint — *`LpxH` remains a Priority-1 finding; PMC12444477 goes 9 → 8, never 9 → 7* —
is therefore **NOT VERIFIABLE on T-107.** It **is** verified at the merged tip on the pinned run
(`runs/2026-08-02_2130`), where `PMC12444477/strict` yields exactly 8 findings including `LpxH` with
`Unknown` absent. **This is a gap in T-107's coverage of the constraint, not evidence against it, and
it must not be reported as confirmation.**

---

## 4. Priority 2 — the failure, and why it is not `CONDITIONALLY SATISFIED`

* **1 unsupported retained reaction**, on **`PMC13231680`**.
* `NOT EVALUATED` on **11 legs** across 6 papers, every one carrying the documented D-067
  precondition-3 reason (`supported_reactions_complete` false — exhaustiveness unproven).
* **Eligible legs: 6.** One of them failed.

D-075 permits `CONDITIONALLY SATISFIED` only when **every eligible leg passes** and every ineligible
leg carries the exact documented reason. The second condition holds; **the first does not.**

**The report may not claim full 20-leg biological validation**, and it may not claim
`CONDITIONALLY SATISFIED` either. **Priority 2 is a `FAIL` on a measured, eligible leg.**

---

## 5. Legs — and four degradations against T-105

Strict PWML **2 pass / 5 fail**; research **6 pass / 1 fail**; **3 timeouts**; 0 skipped.

**Of the six strict legs that passed in T-105, four degraded and two held:**

| Leg | T-105 | T-107 |
|---|---|---|
| `PMC13231680/strict` | PASS | **FAIL (no_reactions)** |
| `PMC12180156/strict` | PASS | **FAIL (contract)** |
| `PMC12096016/strict` | PASS | **TIMEOUT** |
| `PMC12452463/strict` | PASS | **FAIL (contract)** |
| `PMC12856317/strict` | PASS | PASS ✔ |
| `PMC12782028/strict` | PASS | PASS ✔ |

**Every research leg held.** On three of the four degraded papers the *research* leg passed on the
same draw, which makes those legs paired controls rather than anecdotes.

`FAIL (contract)` appears **twice**, which is one failure mode rather than four unrelated ones.

### What this is NOT yet

**These are leg headlines, not a diagnosis.** Three explanations remain open and the outcomes alone
cannot separate them:

1. **Draw variance** — real here, and the sprint records identical legs giving materially different
   Stage-1 draws at temperature 0.
2. **A C-099/C-100 regression** — **T-107 is the first full run with those production changes in.**
   T-105 (2026-08-22) and T-106's artifacts (2026-08-24) both predate them. C-099 touched
   `map_ids.py`. This wave's three cards are scorer-and-test only and **cannot** move a leg outcome.
3. **A real defect T-105 masked.**

**Two facts constrain any hypothesis**, and one of them killed my own mid-run reading: **`PMC12856317/strict`
and `PMC12782028/strict` both passed.** A mechanism predicting that strict mode is broken is falsified
by two counterexamples. **Those two legs are the most informative in the run** — they are what the four
failures must be distinguished from.

**Required before any code:** classify each as `product_contract_violation`, `gold_data_defect`, or
`policy_disagreement`, citing the gold `relevance_note` / `export_rationale`. **Only the first
justifies a card.** The two `FAIL (contract)` legs are where to start, because that class is the one
that can justify code.

---

## 6. Priorities 4/5 — unmoved, exactly as D-081 predicted

`0/7` and `0/2`. The reconciliation is visible and working — 10 legs with coverage, 5 legs carrying
forbidden terms across 4 papers, **5 forbidden terms excluded**,
**`legs_cleared_by_reconciliation: []`**, `legs_still_below_minimum: []`.

**D-081 predicted this from the previous wave's measurement and it held on a fresh run:** Ruling A
makes the coverage measurement *readable per leg* without moving either priority, because their
numerators are semantic confirmation and the frozen release record, not the requested-core ratio.

The D-080-ratified definition is visible in the artifacts — e.g. `PMC13231680/research` raw
`5/6 = 0.833` → accepted `4/5 = 0.800`, with `Zn2+` (`cofactor_as_protein`) removed from **both**
numerator and denominator because `matched_in_raw: true`. **Under the literal denominator-only
reading that leg would have scored `5/5 = 1.000`** — a matched forbidden term inflating coverage,
which is exactly what ratification prevented.

---

## 7. Cost

**Ceiling lifted by D-086**, so this is recorded rather than enforced. **No token usage is recorded
anywhere by the pipeline**, so actual spend cannot be derived from the artifacts — the observability
gap registered in D-086. The pre-run bound was **$0.62–$3.70**; actual spend must be read from the
provider account. A figure materially outside that range is a finding about the estimate's model, not
a budget event.

---

## 8. What T-107 settles, and what it does not

**Settles:** Priority 1 is **not** structurally stuck at 7–8; it reached **5** with the merged
instrument. Priority 3 is clean. Priorities 4/5 behave exactly as D-081 said they would.

**Does not settle:** whether the four strict degradations are variance, regression, or newly-exposed
defects. Whether `LpxH` holds on this paper under a live run — **it timed out**. Whether Priority 2's
single unsupported reaction is a product-contract violation or a gold-data defect.

**The run is NOT ACCEPTED**, and the reason is a single measured unsupported reaction on one eligible
leg — not a systemic collapse. That is a much narrower gap than any previous milestone reported.
