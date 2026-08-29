# PWML RECOVERY SPRINT — T-107 EXECUTION AND CLOSE-OUT

You are the **Lead Orchestrator and Integration Authority** for:

`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`

Integration branch: `sprint/pwml-recovery`

**Do not merge to `main`.** Work autonomously. Do not ask the product owner about routine
implementation, testing, review or merge decisions.

**Your primary job is to run T-107 once, score it honestly, and close the sprint's acceptance
question.** T-107 is **GO** — the authorization landed at the end of the previous wave.

---

## 1. TAKEOVER — verify once, then move

Read `CLAUDE.md` first, then `docs/pwml_recovery_sprint/RESUME-NEXT-SESSION.md`,
`T107-READINESS.md`, `PRODUCT_CONTRACT.md`, `LEDGER.md`, `DECISIONS.md`, `TEST_MATRIX.md`.
The permanent merge rules **G1–G11** are binding and are not restated here.

| Check | Expected |
|---|---|
| local = `origin/` = `git ls-remote` | **all three equal** — read it, do not recall it |
| `main` | local `7531692`, remote `03f1af5`. **`main` advanced outside this sprint; touch neither ref.** The sprint tip is not an ancestor of remote `main` |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python | zero |
| allowed IDE processes | two `ms-python.isort` `lsp_server.py` — **never cleanup targets**. Their PIDs change when the IDE restarts them; match on command line, not PID |
| product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632` |
| caches + `topics_*.txt` | uncommitted, as found |
| whole-tree G11 | **0 non-compliant** (count is self-referential — a check's own report is committed after it runs) |

**Before claiming the branch, editing shared files, pushing, or launching a live job:** run
`ListAgents`, contact every live peer, and agree explicit ownership. Session identities are not
stable. Do not treat a quiet session as dead.

**Prune no worktree.** `C:/t/c101`, `c101base`, `c102`, `c103`, `rev102base`, `rev102tip`,
`rev102r1`, `rev103base`, `rev103tip`, plus everything earlier waves listed.

---

## 2. DONE — do not rediscover

**Three cards merged, each independently reviewed, each review finding something real:**

* **C-101** `ee7cb6b` — 16/5 metric split · row-aware PathBank sentinel seam · raw/accepted
  Priority 1 with the variance band. Three correction rounds.
* **C-102** `8e4334f` — coverage denominator reconciled against `forbidden_identifiers`. One round.
* **C-103** `511344a` — F-142 replay expectation re-pointed at the seam that carries the verdict.
  **Zero production lines.** One round.

**Rulings:** D-072 (A) · D-073 (B, variance band) · D-074 (C, exact sentinel) · D-075 (D, Priority-2
`NOT EVALUATED`) · D-076/D-079/D-082 (budget) · D-077 · D-078 · **D-080 (interpretation — see § 6)** ·
D-081 · D-083 · D-084 · **D-085 (paid models authorized for T-107)**.

**Findings:** F-142 · F-143 · F-144 · F-145.

**Settled — do not re-derive:** the Glutathione red is a **stale expectation, not a production
defect** (F-142, now fixed by C-103) · Ruling A **does not move Priority 4 or 5** (D-081, reproduced
across all 21 run dirs) · under D-074 **no Priority-1 row can ever be contract-adjusted** (D-077) ·
the F-132 population is **92 terms / 47 legs / 7 papers**, not the bundle's 62/32/6 (F-145).

### The gate numbers

| Gate | Expected |
|---|---|
| **SMOKE** (20 files) | **473 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** |

**⚠ The gold-readers baseline CHANGED.** It exited 1 for most of this sprint, correctly, because
`test_strict_failure_replay.py` carried the two F-142 reds. **C-103 cleared them.** Any older charter
saying *"this selection exits 1 at base, and that is correct"* is **stale**.

---

## 3. T-107 — GO. This is the job.

**D-085 (product-owner ruling): paid external models are authorized.** That cleared condition 8, the
only failing condition. Full pre-run analysis is in `T107-READINESS.md`, preserved unedited with a
new § 6 recording the resolution.

### Binding constraints

* **Run on the pinned configuration exactly as `.env` already holds it** —
  `deepseek/deepseek-v4-flash` on all nine OpenRouter slots, `LLM_PROVIDER=openrouter`,
  `LLM_TEMPERATURE=0`.
* **Do not edit `.env`.** Do not switch provider, do not re-pin, do not set a free variant.
* **Do not use LM Studio for T-107.** It serves `glm-4.6v-flash`, which is neither the pinned model
  nor the configured `LOCAL_MODEL`. Using it is a **fallback model** and destroys comparability with
  T-104/T-105/T-106 — comparability is the entire point of the pinned plan.
* **Launch ONCE.** Score the **first valid official draw**. Do not rerun for stochastic composition.
  **Do not rerun a 7 to chase a 6** (D-073).
* **$5 ceiling still binds** unless the product owner raises it. Measured pricing: `$0.0868`/M
  prompt, `$0.1736`/M completion; projected **$1–3** at T-105's scale, with real variance and **no
  spend telemetry to abort on**. **If projected or actual spend would exceed $5, stop and report.**

### Before launching, verify at launch (not from this document)

heavy lock free · zero sprint-owned Python · no peer session owning an overlapping live job ·
SMOKE and gold-readers green at the tip you are about to measure · enough session time to monitor
through completion or formally transfer the wrapper.

### The run

The pinned plan is **10 papers / 20 legs**, the same set T-104/T-105/T-106 used. Prior launches went
through `scripts/batch_run.py`; **`topics_t104.txt` is the ratified pinned topic set** — the 10
pinned gold cases with scope lines verbatim from `bench/gold/pinned_v1.json`. **Ratify it before
launch** with `scripts/bench_acceptance.py --verify-plan`, which must report `OK` with all 10
`[pinned_override]` and **0 search calls**. If it does not, stop — do not improvise a topic set.

Prior shape, for sizing only (**re-derive, do not copy**):

```
scripts/batch_run.py --topics topics_t104.txt --out runs_verify --modes strict,research \
    --timeout 1800 --deadline 3 --fresh
```

**Wrapper timeout:** T-105 ran **4.85 h**, T-104 **5.44 h**. Set the bounded wrapper's timeout from
the measured ~5 h with a **hard ceiling of 6 h**. Run it **tracked in the background** (D-026 — it
exceeds the interactive cap by construction), record the task id and output path **immediately**,
poll rather than launching duplicates, and **never leave it unowned**.

### Scoring — what the report must say

Produce the full table from `T107-READINESS.md` § 1, measured on the new run:

* **Priority 1** — **raw** count and composition, **accepted** count and composition, status
  `PASS` (0–6) / `PASS_WITHIN_VARIANCE` (7) / `FAIL` (8+), every applied case-scoped tolerance, and
  **confirmation that `LpxH` remains counted** on PMC12444477.
* **Priority 2** — results on eligible legs, the exact `NOT EVALUATED` count, the D-067 precondition-3
  reason, and whether it is `CONDITIONALLY SATISFIED` (D-075). **It may not be reported as full
  20-leg biological validation.**
* **Priorities 4/5** — raw **and** accepted anchor coverage.

**Priority 1 is genuinely uncertain between 7 and 8.** T-104 = 7, T-105 = 7, T-106's artifacts
re-scored at the merged tip = 8. **7 is `PASS_WITHIN_VARIANCE` and clears the gate; do not rerun it.
8+ is an honest acceptance failure and is reported as one.** A new systematic defect is still a
defect even when the total sits inside the band.

---

## 4. AFTER T-107

Classify every failure as `product_contract_violation`, `gold_data_defect`, or
`policy_disagreement`, citing the gold `relevance_note` / `export_rationale`. **Only the first
justifies code.** A benchmark failure does not by itself justify a change.

Charter cards only for `product_contract_violation`s, one narrow card each, dispatched and reviewed
under the process in § 5.

---

## 5. PROCESS — merge gates, not suggestions

**Every** test, probe, scorer, benchmark, pipeline leg and LLM-backed command runs through
`docs/pwml_recovery_sprint/evidence/bounded_run.py`. No detached processes, no `nohup`, never
`pytest -n auto`. One heavy job at a time.

* **Pass the explicit interpreter** `…/.venv/Scripts/python.exe` — **F-143**: `bounded_run.py`
  resolves a bare `python` from the child's PATH → the system 3.13 with no `streamlit`, producing
  **35 spurious import errors that read exactly like a regression**. `pinned_pytest`'s exit-98 check
  verifies the *tree*, not the *interpreter*.
* `--basetemp=<unique short dir under C:/t/>` on every pytest call, **parent pre-created**. Missing
  it errors 83 tests; a missing parent errors every test in setup and once reported **382 instead of
  453**. Neither is a test result.
* `PYTHONPATH=<tree>/src`, `T2PW_OFFLINE_CURATOR=1` unless a specifically authorized live path
  requires otherwise, `--heavy-lock <TASK>` on SMOKE and heavy legs.
* Allocate every report path with `g11_evidence.py next --task <T> --label <l>` and **capture the
  output into a variable**. Labels must match `^[a-z0-9][a-z0-9._-]*$` — an invalid label is rejected
  and **the error text silently becomes your `--json` path**.
* `bounded_run.py`'s report carries **no child stdout**. Redirect to a log and grep it. **Never pipe
  through `head`** — SIGPIPE truncates the log. **Commit probes and their logs**, not just reports.
* Every job: `FINAL SURVIVING COUNT : 0` and `cleanup : success`. Survivors are an **infrastructure
  failure**, not a test result.
* **Never** `taskkill /IM python.exe` or `pkill python`. Terminate only an exact sprint-owned tree.
* **Never commit** `data/enrichment_cache.json`, `data/id_mapping_cache.json`, `topics_*.txt`, or
  `src/t2pw/app/streamlit_app.py`. Stage explicit paths, inspect `git diff --cached`, use
  `git commit -F <file>`.

### Review discipline — this is what made the last wave work

* **Fix pass/fail items in writing BEFORE the diff exists.** An item chosen after seeing the code is
  a rationalisation.
* **Record predictions before running.** Being wrong in writing is productive; being vague is not.
* **Run selections split, not only combined** — a combined total hides a shift between two files.
* **Check the guard that was REMOVED**, not only the one added.
* **F-144, the wave's central lesson: a non-vacuity guard can be real and still guard the wrong
  emptiness.** Asserting that *a* finding was produced is not evidence that *the path under test*
  produced it. Every test whose null result is load-bearing must (a) run the production path, (b)
  assert the specific thing that path emits, and (c) **survive a mutation attack by a non-author**.
  **A non-vacuity guard is not evidence until someone who did not write it has failed to defeat it.**
* **D-084 — mutation restores replay SAVED BYTES.** `git checkout --` reverts *more* than it mutated;
  text-mode restore reverts *less* than byte-exactly. Verify the restore, do not assume it.
* **Keep failed and invalidated measurements beside their corrections.** Never replace them. A
  repository holding only successful measurements has a survivorship-biased evidence trail.
* **Ceilings: seven under-set last wave, all seven the orchestrator's.** Derive from the mandated
  deliverables at the module's measured comment density. **Review-mandated work never charges
  ceiling 1 or 2 (D-076 A1, D-082); kept failed runs are never charged.**
* Up to **two** automatic correction rounds per card; a third is an explicit authority decision with
  reasons (D-078).

---

## 6. OPEN ITEMS — none blocking T-107

* **D-080 wants product-owner ratification.** It is an *interpretation*: D-072 says exclude forbidden
  identifiers from the "denominator"; C-102 excludes from numerator **and** denominator. The literal
  reading makes **nine committed legs report a coverage ratio above 1**, and lets a pipeline that
  exported all three forbidden identifiers on PMC12782028 score **1.0000**. The code is merged on the
  both-sides reading. **Surface this; do not re-litigate it.**
* **D-083** — two follow-ons: C-102's deep-copy fix has no test (its revert mutation is green), and
  the split-gate driver should abort on `errors > 0`. Evidence tooling plus one low-stakes test.
* **F-145** — quote the F-132 population as **92 / 47 / 7**, not the bundle's 62/32/6.
* **Unaudited:** whether `test_c074_strict_core_floor.py` and `test_c072_incomplete_core_demotion.py`
  pin their caps **non-vacuously**. F-142's no-coverage-gap claim rests on them.

---

## 7. BEFORE YOU STOP

Confirm and report: no merge in progress · nothing staged · local = `origin/` = `ls-remote` · `main`
untouched · product-owner `streamlit_app.py` intact at 35/2 with the expected hash · caches and
`topics_*.txt` uncommitted · whole-tree G11 0 non-compliant · heavy lock absent · zero sprint-owned
Python · only the expected IDE `isort` processes · no pytest, scorer, benchmark, paper run or live
wrapper remaining · every completed job `FINAL SURVIVING COUNT : 0` and `cleanup : success` · **total
external spend, stated in dollars** · no dispatched agent silently stalled.

**Track agent liveness separately from job liveness.** A subagent sat at `running` for twelve hours
this sprint. ~15 min without observable progress → request status. ~30 min with no response, no
process, no changing artifact and no commit → treat as stalled, interrupt, preserve the worktree,
record, redispatch from the last verified commit.

**Update `RESUME-NEXT-SESSION.md` in place before you finish.** Two load-bearing probe outputs this
sprint survived only by luck in dead sessions' temp directories, and one probe source was lost for
good. **A G11 report certifies a job was clean and preserves nothing about what it found.**
