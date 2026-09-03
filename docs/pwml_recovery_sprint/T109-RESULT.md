# T-109 — official release-candidate result

**Run `runs_verify/2026-09-02_2052`. Launched 2026-09-03T02:54:33Z, exited 07:51:36Z. Scored ONCE,
offline, from committed artifacts. `T109-RUN-OWNERSHIP.md` holds the live record.**

> ## VERDICT: **NOT ACCEPTED**
>
> **No hard gate FAILED. One hard gate could not be EVALUATED, and that is not a pass.**
> Priority 2 is `ok = null` / `evaluated = false`, and `acceptance.py:1057-1061` states the rule this
> turns on: *"`ok=None` is falsy, so a caller gating acceptance on `all(entry["ok"] ...)` refuses the
> run, which is the correct default for an unproven absolute."*
>
> **This is a materially better run than T-108 and it is still not accepted. Both halves are true and
> neither cancels the other.**

---

## 1. Hard gates and diagnostics, reported SEPARATELY as the ruling requires

### 1a. The absolutes — Priorities 1, 2, 3

| # | Gate | Result | |
|---|---|---|---|
| **1** | zero known false real identifiers | **`ok = true`** — raw **0**, accepted **0**, `accepted_status: "PASS"`, target 6 | **PASS** |
| **2** | zero unsupported retained reactions | **`ok = null`, `evaluated = false`** — the verdict was never reached on **13 of 19** scored legs across **7 papers** | **NOT EVALUATED** |
| **3** | zero referential-integrity violations | **`ok = true`** — observed **0** | **PASS** |

**Priority 2's zero is the absence of a measurement, not the absence of unsupported reactions** — the
scorer says so in its own words. `supported_reactions_complete` is `false` on all ten gold cases
(D-087), so the verdict can only be reached where a `max_retained_reactions` ceiling exists, and
**both ceilings sit on negative controls.** This is the standing D-087 limitation, unchanged, and it
is why the run cannot be accepted however well it performs elsewhere.

### 1b. The diagnostics — Priorities 4 and 5, which are NOT hard gates

| # | Diagnostic | Result |
|---|---|---|
| 4 | meaningful requested-pathway coverage | `ok = false` — **0/8 = 0%** |
| 5 | strict PWML pass rate among exportable papers | `ok = false` — **0/2 = 0%** |

**`acceptance.py:1050-1052`: "Priority 4 is a coverage judgement and priority 5 is a rate to
maximise, so neither is a hard gate."** Priority 5's `0/2` is the outcome **`D-089` accepted in
advance**, and § 3 below shows the instrument attributing it precisely.

---

## 2. Completion, and the operational result

| | T-107 | T-108 | **T-109** |
|---|---|---|---|
| completion | 20/20 | 20/20 | **20/20, `complete: true`** |
| scorable legs | 17 | 19 | **19** |
| **timeouts** | 3 | 1 | **0** |
| legs with an empty payload | — | — | **0** |
| duration | 5.63 h | 6.37 h | **4.95 h** (17824 s of a 72000 s ceiling) |

**Zero timeouts is the first in the sprint**, and `2a` records it as *"NONE. No leg hit the 3600 s
ceiling."* The § 2.1 ceiling ruling of 3600 s is now backed by a run that needed none of its top end.

**Status tally:** `pass 7`, `fail 7`, `scope_conflict 6`. **All six organism-trap legs aborted at
Stage 0 and all three trap papers carry zero `.pwml` files.**

**One evidence limit the scorer states rather than glosses**, and it is worth carrying forward:
`leg_timeout_overridden` is written **only on the timeout path**, so with zero timeouts **0 of 20
manifest rows carry a budget block at all**. The no-override claim therefore rests on the pre-launch
resolution (`_ceiling(3600.0) -> overridden False`, G11 `T-109/04`) and not on the manifest. *"An
audit that counted silent legs as confirmations would be reporting its own blind spot as a pass."*

---

## 3. Why Priority 5 reads `0/2` — the instrument's own attribution

**Both legs in the denominator passed the strict technical gates AND semantic evaluation, and both
were held by the incomplete-core cap alone.**

| leg | status | strict gates | semantic | completeness | missing anchors |
|---|---|---|---|---|---|
| `PMC12096016/strict` | `review_required` | **passed** | **passed** | **0.916667** | **`EntD` — one anchor** |
| `PMC12782028/strict` | `review_required` | **passed** | **passed** | 0.571429 | `oxysterol, MSMO1, SQLE, FDFT1, HMGCR, HMGCS1` |

Both carry `strict_acceptance_eligible: false` with the single reason
`requested_core_anchors_unmatched`. **The scorer's own note:**

> A runner `pass` is NOT a Priority-5 point. A leg that executes, clears the strict technical gates
> and passes semantic evaluation can still be held at `review_required` for incomplete requested-core
> coverage — **which is merge rule 7 working as written: incomplete-but-correct pathways are
> PRESERVED as `review_required`, never dropped and never promoted.**

### 3.1 D-088's two required consequences BOTH hold, on one fresh draw

**`PMC12782028` remains a genuine recall failure, and the artifacts name the enzymes.** `HMGCR` and
`HMGCS1` **are** the mevalonate arm; `FDFT1` and `SQLE` sit immediately downstream; `MSMO1` is the
methylsterol demethylation stage. D-088 requires this leg to *"remain incomplete — its upstream
mevalonate reaction arm is genuinely absent."* **It does, by name.**

**`PMC12096016` is held by `EntD` ALONE** — and D-088's expected-consequence table excuses
`ATP`/`NADH`/`Fur` while explicitly **not** excusing `EntD`, which it requires *"remain VISIBLE as an
extracted/supporting entity that is not properly wired."* **On this draw `ATP`, `NADH` and `Fur` all
MATCHED**, where F-169 found all four unmatched on the archived draws.

> **So on this draw the cap held both legs, and in each case for a reason D-088 sanctions. A change
> clearing both would have been a reject; nothing cleared either. This is the discrimination the
> whole wave was built to preserve, holding on data D-089 was not ruled on.**

**This REFINES F-173 and does not retract it.** F-173's claim is that the instrument *cannot
distinguish* the two legs on inputs production may read — still true. What this draw shows is that
the limitation **did not bite here**, because the draw matched the three cofactor-class anchors.

**Candidate A is re-refuted on fresh data.** `min_core_coverage` is 0.5 and `PMC12782028/strict`
reads **0.571 ≥ 0.5** with `minimum_core_satisfied: True` — so relying on the existing thresholds
would have **released the leg whose mevalonate arm is missing.** The packet measured 0.538 on the
archived draw; a second independent draw gives the same answer.

---

## 4. Priority 1 fell 8 → 0, and here is exactly how much that is worth

**The number is real: `raw = 0`, `accepted = 0`, no rows at all.** T-108 scored **8** against the
same gold blob `36f4b7b6…`, so unlike the T-107/T-108 pair (**F-165**: different gold, not
comparable) **these two numbers ARE comparable on gold.**

**What is NOT established, and must not be inferred from it:**

- **That a defect was fixed.** No production code changed between T-108 and T-109 — the whole wave
  was documentation and evidence. **A code change cannot explain this, because there was none.**
- **That the identification pass improved.** Priority 1 counts false real identifiers **among what
  was exported**. T-109 exported a different and partly smaller surface than T-108; a zero over a
  different surface is a different claim from a zero over the same one.
- **That 0 will recur.** The scorer's own `variance_note` warns that seven is *"a one-finding
  stochastic band"* and that T-105's seven was *"composed of almost entirely different rows"* than
  T-104's. **An 8 → 0 move is far outside that band and is therefore evidence about the DRAW, not
  about the pipeline, until a second run reproduces it.**

**The honest sentence: T-109 exported no false real identifiers. That is the best Priority-1 result
the sprint has recorded, and it is one draw.**

---

## 5. Priority 3 reads 0 while a leg produced orphaned references — and that is the gate working

`PMC12856317/strict` failed with `gate.registry_validation_failed`:
`/processes/interactions/1/entity_2 unknown entity: HRM3` and `.../2/entity_2 unknown entity: HRM6`.

**Priority 3 counts referential-integrity violations in what was EXPORTED.** The gate refused the
leg, so the orphans never reached an export. **A zero here is a gate that held, not an absence of
defects in the draw** — and reporting it without this paragraph would overstate it.

---

## 6. LpxH — the question raised at leg 1, answered NO

At leg 1 I recorded that `PMC12444477/strict` **finished** where T-107 and T-108 both lost it to the
clock, and flagged that `LpxH` *might* become measurable — `T108-READINESS.md` § 5 records it as
unverified precisely because both legs had no payload.

**It did not.** Section 9: `research status='fail' findings=0`, `strict status='fail' findings=0` →
**`LpxH` remains UNVERIFIED.** It is verified only on the pinned run `runs/2026-08-02_2130`.
**Recorded because I raised it; a flagged question that turns out negative must be closed as loudly
as one that turns out positive.**

---

## 7. What must travel with any quotation of these numbers

> **Priority 2 = 0 is the absence of a measurement, not the absence of unsupported reactions.**
> D-087 stands; `supported_reactions_complete` is unset on all ten cases and both
> `max_retained_reactions` ceilings are on negative controls.

> **Priority 5's `0/2` is an explicitly accepted conservative limitation (`D-089`), not a pipeline
> capability measurement, and NOT delivery of D-088 clause 2.** Both legs passed strict gates and
> semantic evaluation and were held only by the anchor cap.

> **Priority 1 = 0 is one draw, on a different exported surface from T-108's 8, with no code change
> between them.**

> **The C-116 D-088 diagnostics do not exist for any leg of this run — F-175.** `coverage_summary.json`
> still carries `matched_terms` and `unmatched_terms` verbatim, so the anchors are visible; the
> census classification is not.

---

## 8. Process closure

| Check | Result |
|---|---|
| Wrapper exit | **1** (`nonzero`) — expected when not every leg passes; T-107 and T-108 exited 1 likewise. **Not** an infrastructure failure |
| Duration | **17824.00 s = 4.95 h** of a 72000 s ceiling (24.8%) |
| `FINAL SURVIVING COUNT` | **0** |
| `cleanup` | **success** — 43 descendants observed, 43 terminated |
| Heavy lock | `acquired=true released=true`; `C:/t/heavylock` **absent** after |
| Sprint-owned Python after | **zero** — two `ms-python.isort` `lsp_server.py` only, matched on **full command line** |
| Gold | `36f4b7b690b577f72882c3045ca6728d1ec8d9d1` unchanged |
| `acceptance.py` | `4bd893ac…` unchanged |
| `streamlit_app.py` | `47e4fafa…` unchanged, still uncommitted |
| `main` | `7531692` — never written |

**The `ms-python.isort` PIDs CHANGED across the run** — `177556`/`177596` before, `271968`/`272160`
after. The IDE respawned them. **The count is unchanged at two and the command line is identical,
which is why the command line is the identity and the PID never is.**

**T-109 ran exactly once, is scored exactly once, and is now IMMUTABLE. It is not re-run, not
re-scored and not reinterpreted.** A further candidate needs a new milestone identity and a
separately recorded readiness decision.
