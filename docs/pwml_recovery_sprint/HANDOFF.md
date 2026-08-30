# PWML RECOVERY SPRINT — T-107 TRIAGE AND CLOSE-OUT

You are the **Lead Orchestrator and Integration Authority** for
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, integration branch `sprint/pwml-recovery`.

**Do not merge to `main`.** Work autonomously.

**T-107 has RUN and is SCORED. Your job is to triage its failures — not to re-run it.**

---

## 1. Takeover — verify once

Read `CLAUDE.md`, then `T107-RESULT.md`, `RESUME-NEXT-SESSION.md`, `PRODUCT_CONTRACT.md`,
`LEDGER.md`, `DECISIONS.md`, `TEST_MATRIX.md`. **G1–G11 bind you.**

| Check | Expected |
|---|---|
| local = `origin/` = `git ls-remote` | **all three equal.** Tip at handoff: **`e77ad3d`** — read the invariant, not the number |
| `main` | local `7531692`, remote `03f1af5`. Advanced **outside** this sprint. **Touch neither** |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python | zero |
| IDE processes | two `ms-python.isort` — **never cleanup targets**; **PIDs change on reboot, match on command line** |
| `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` |
| caches + `topics_*.txt` | uncommitted |
| **SMOKE** | **473 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** ← changed by C-103; any charter saying it exits 1 is **stale** |

Run `ListAgents` and coordinate with live peers before claiming the branch or launching a job.
**Prune no worktree.**

---

## 2. T-107 — the official result. Do NOT re-run it.

`runs_verify/2026-08-28_1816` · 20/20 legs · **17 scorable** (3 timeouts produced no payload) ·
5.63 h · every job zero survivors. Full detail in **`T107-RESULT.md`**.

| Priority | Raw | Accepted | Status |
|---|---:|---:|---|
| **1** — false real identifiers | **5** | **5** | **`PASS`** ← first result under 6 in the sprint |
| **2** — unsupported retained reactions | **1** | — | **`FAIL`** (an *eligible* leg failed) |
| **3** — referential integrity | 0 | — | **`PASS`** |
| 4 — requested-pathway coverage | `0/7` | `0/7` | FAIL (not a hard gate) |
| 5 — strict PWML rate | `0/2` | `0/2` | FAIL (not a hard gate) |

**Overall: NOT ACCEPTED**, on Priority 2 alone.

**Run-once still binds. A 5 is not re-drawn, and neither is anything else.** Never repeat a leg
because its draw is unfavourable; something not observed is reported as *"not observed"*.

**⚠ `LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out, so there is no payload. It
**is** verified at the merged tip on the pinned run (`runs/2026-08-02_2130`: 8 findings including
`LpxH`, `Unknown` absent). **Do not report T-107 as confirming it.**

---

## 3. THE JOB — triage, and classify before you touch code

**A benchmark failure does not by itself justify a code change.** Classify each as
`product_contract_violation` · `gold_data_defect` · `policy_disagreement`, citing the gold
`relevance_note` / `export_rationale`. **Only the first justifies a card.** Use `pwml-bio-auditor`
for adjudication; it is read-only.

Four strict legs that passed at T-105 degraded. **Two held** — `PMC12856317/strict` and
`PMC12782028/strict` — and **those two are your controls.** Any hypothesis that predicts strict mode
is broken is already falsified by them.

**T-107 is the first full run with C-099 and C-100 in production.** T-105 (2026-08-22) and T-106
artifacts (2026-08-24) predate both. **C-099 touched `map_ids.py` — species preservation.** This
wave's three cards (C-101/C-102/C-103) are scorer-and-test only and **cannot** move a leg outcome.

### 3a. `PMC12452463/strict` — FAIL (contract). Start here.

```
gate.protein_fur_is_missing_a_uniprot_or_drugbank_identifier
gate.protein_enterobactin_synthase_is_missing_species_organism
gate.generated_protein_complex_enterobactin_synthase_complex_componen...
```

**Three blocking gates, two of them species/identity — exactly C-099 surface.** And **`Fur` is the
protein F-141 classified** as *"candidate does not describe the shipped identifier — withholding
correct"* (both Fur rows). If C-099 changed what survives on those rows, this is where it shows.
**Strongest lead in the run.**

### 3b. `PMC12180156/strict` — FAIL (contract)

```
gate.protein_alas2_is_missing_a_uniprot_or_drugbank_identifier
```

**`ALAS2` is the exact protein at the centre of the O-1 / species dispute** — the previous handoff
records `runs_verify/2026-08-04_1754/.../PMC12856317/strict` shipping PWML with *Arabidopsis* on a
**human ALAS2** wrapper. One blocking issue, same identity-gate family as 3a.

### 3c. `PMC13231680/strict` — FAIL (no_reactions). **This may be CORRECT.**

Message: *"Multi-paper RAG produced no additional usable reactions (no evidence retrieved — check
that the embeddings endpoint is running...)"*.

**That parenthetical is boilerplate, not a diagnosis — I checked.** RAG ran normally: **1294 rejected
candidates across 15 legs**, against T-105 1837 across 18. The embeddings endpoint is fine
(`RAG_EMBEDDING_BASE_URL` points at LM Studio, which serves `text-embedding-nomic-embed-text-v1.5`).

**The real question is the opposite one: `PMC13231680` is a NEGATIVE CONTROL**
(`mechanistic_relevance=context_only`, excluded from `gold_relevance_prevalence`). **A negative
control producing no reactions in strict mode may be exactly right — and T-105 PASS may have been
the wrong outcome.** Check the gold `export_rationale` before calling this a regression. **This is a
strong gold-data-defect / correct-behaviour candidate, not a defect candidate.**

**The same paper carries Priority 2 only failure** (1 unsupported retained reaction). **If the
negative-control reading holds, that finding needs re-examining too** — and Priority 2 `FAIL` with
it. **That single row is the whole reason T-107 is NOT ACCEPTED.** It is worth getting right.

### 3d. `PMC12444477` x2 and `PMC12096016/strict` — TIMEOUTs

`PMC12444477` timed out in **both** modes at 30m00s. The run summary flags it
`!! RESEARCH-MODE DEFECT !!` — *"research mode is fail-open by design, so ANY research failure is a
code defect"* — and `class=broken (strict failed too -- fix the shared cause)`. T-105 also had 2
timeouts, so this is not new; but it is why `LpxH` is unverified, which gives it extra weight.

---

## 4. Also open

* **D-083** — C-102 deep-copy fix has no test (its revert mutation is green); the split-gate driver
  should abort on `errors > 0`. Evidence tooling plus one low-stakes test.
* **F-145** — quote the F-132 population as **92 terms / 47 legs / 7 papers**, not the bundle
  62/32/6.
* **D-086 observability gap** — **the pipeline records no token usage anywhere**, so no run in this
  sprint can be costed after the fact. **Actual T-107 spend must be read from the OpenRouter account
  and recorded**; the pre-run bound was $0.62–$3.70. **The ceiling is lifted (D-086) — record it, do
  not enforce it.** A figure far outside that range is a finding about the estimate model.
* **Unaudited** — whether `test_c074_strict_core_floor.py` / `test_c072_incomplete_core_demotion.py`
  pin their caps **non-vacuously**. F-142 no-coverage-gap claim rests on them.

---

## 5. Process — merge gates, not suggestions

Everything through `evidence/bounded_run.py`. **Pass the explicit venv interpreter** (**F-143**: a
bare `python` resolves to the system 3.13 with no `streamlit` → 35 spurious import errors that read
exactly like a regression; `pinned_pytest` exit-98 check verifies the *tree*, not the *interpreter*).
`--basetemp` under `C:/t/` with the **parent pre-created** (a missing parent once reported **382
instead of 453**). `PYTHONPATH=<tree>/src`, `T2PW_OFFLINE_CURATOR=1`, `--heavy-lock <TASK>`. Allocate
report paths with `g11_evidence.py next` and **capture the output into a variable** (an invalid label
silently becomes your `--json` path). Reports carry **no child stdout** — redirect and grep, never
`head`. Every job: `FINAL SURVIVING COUNT : 0`, `cleanup : success`. **Never** `taskkill /IM
python.exe`. **Never commit** the caches, `topics_*.txt`, `streamlit_app.py`, or any
`cache_snapshot/`.

**A killed job strands the heavy lock** (the kill skips the wrapper `finally`). Before clearing
`C:/t/heavylock`: holder PID **dead**, holder file **byte-identical across samples seconds apart**,
**zero** matching processes. A sample archive is at `evidence/t107_stranded_holder.json`. **Never
clear a lock on one sample.**

**Bash heredocs break on apostrophes here** — write the file with the Write tool, or `cat` from a
scratch file. This cost a commit attempt while writing this handoff.

### Review discipline

Fix pass/fail items in writing **before** the diff exists · record predictions before running · run
selections **split** as well as combined · check the guard that was **removed** · **D-084: mutation
restores replay SAVED BYTES** (`git checkout --` reverts *more*; text-mode reverts *less*) · keep
failed measurements **beside** their corrections · **review-mandated work never charges a ceiling**
(D-076 A1, D-082) · two automatic correction rounds, a third is an explicit authority decision.

**F-144, and it bit four times this sprint including twice in one session and once in a process
rule:** *a non-vacuity guard can be real and still guard the wrong emptiness.* Asserting that **a**
finding was produced is not evidence that **the path under test** produced it. **A non-vacuity guard
is not evidence until a party who did not write it has failed to defeat it.**

**It bit me again while writing this handoff.** I counted RAG admissions with the keys
`admitted`/`candidates`, got a clean `0 of 17`, and nearly wrote *"the embeddings endpoint failed"*
into the triage as fact. **The real keys are `accepted`/`rejected`; the true count is 1294
rejections.** A zero from a key that does not exist looks exactly like a zero from a measurement.
**Check the schema before you believe a null.**

---

## 6. Before you stop

Confirm: no merge in progress · nothing staged · local = origin = `ls-remote` · `main` untouched ·
`streamlit_app.py` intact at 35/2 with the expected hash · caches, `topics_*.txt` and
`cache_snapshot/` uncommitted · G11 0 non-compliant · heavy lock absent · zero sprint-owned Python ·
only the expected IDE `isort` processes · every job `FINAL SURVIVING COUNT : 0` / `cleanup : success` ·
**actual T-107 spend in dollars** · no agent silently stalled.

**Track agent liveness separately from job liveness** — a subagent sat at `running` for twelve hours
this sprint. ~15 min without progress → status request; ~30 min with nothing → stalled, interrupt,
preserve, redispatch.

**Update `RESUME-NEXT-SESSION.md` in place.** Two load-bearing probe outputs survived only by luck in
dead sessions temp directories this sprint and one probe source was lost for good. **A G11 report
certifies a job was clean and preserves nothing about what it found.**
