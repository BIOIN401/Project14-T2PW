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
| local = `origin/` = `git ls-remote` | **all three equal.** The T-107 launch tip was **`66615a3`**; it advances as run artifacts are committed, so **read the invariant, not the number** |
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

## 3. T-107 — LAUNCHED. Finish it, score it, do not re-draw it.

**D-085 authorized paid models.** T-107 was launched **once**, at `20:39`, into
**`runs_verify/2026-08-28_1816`**, on `deepseek/deepseek-v4-flash` via OpenRouter.

### If it is still running

Leave it. Poll rather than relaunching. **A single `absent` reading of `C:/t/heavylock` is not
evidence the run died** — the lock cycles fast enough to fool one observation.

### If it stopped before all 20 legs

**Continuing is NOT a re-draw.** It is the same run identity resuming the same directory; the
manifest flushes after every pair, so finished legs are preserved and skipped. Rerun exactly:

```
scripts/batch_run.py --topics topics_t104.txt --out runs_verify     --modes strict,research --timeout 1800 --deadline 5.5
```

**No `--fresh`, no `--stage-only`.** Wrap in `bounded_run.py`, `--heavy-lock T-107`, timeout `21600`.

**Prove continuity before relaunching rather than assuming it** — run
`evidence/t107_continuity_check.py`, which calls the runner's own `find_resumable()` and asserts it
returns the verified path. Confirm `pending` and `legs already present` are what you expect.
**The resume window is 24 h (`RESUME_MAX_AGE_HOURS`)**; past that the runner refuses the directory as
stale and you must re-stage under the full procedure below.

**A killed run strands the heavy lock** — the kill skips the wrapper's `finally`. That happened once
already. Before clearing `C:/t/heavylock`, prove it abandoned: holder PID **dead**, holder file
**byte-identical across samples seconds apart**, and **zero** `batch_run`/`bounded_run` processes.
**Never clear a lock on one sample.**

### If you must stage a genuinely new run — the accepted procedure

1. **Fresh milestone identity, stage-only** — `--fresh --stage-only`. Returns *before* the run loop,
   so zero LLM legs are possible regardless of cache speed.
2. **`--verify-plan` against that exact staged directory.** Require **`verdict: OK`**,
   **`search calls: 0`**, **all 10 `[pinned_override]`**.
3. **Prove continuity** with `find_resumable()` — must return the exact verified path, all pairs
   pending, zero legs present.
4. **Continue that directory WITHOUT `--fresh`.**

**The runner's own hint is a trap.** `--stage-only` prints *"rerun the same command WITHOUT
--stage-only"* — and that command **still carries `--fresh`**, creating a new **unverified**
directory and discarding the staging just certified. There is a genuine sibling hazard the other way
(`batch_run.py` silently skips **finished** pairs without `--fresh`, giving a partial cohort that
looks complete), and a peer recommended `--fresh` as a blanket cure — **which here would have been
the defect.** The discriminator is measured, not guessed: `already recorded : 0`. **Assert the
property; do not apply the proxy rule.**

### Cost — the ceiling is LIFTED (D-086); still measure it

**The $5 ceiling no longer binds** — product-owner ruling D-086. **No cost-based abort.** Run-once,
never-chase-a-draw and the pinned configuration all still bind; budget was never why.

Prices `$0.0868`/M prompt, `$0.1736`/M completion. **T-105 recorded no token usage
anywhere**, so the pre-run figure is a **bound from measurable inputs**: 10 papers, 592,813 source
chars, ~14,820 tokens per full text → **$0.62–$3.70**, with $5 reached only at ~162
full-text-equivalent passes per leg. Derivation: `evidence/orch715_t107_cost_bound.py` / `.log`.

**Cheap does not mean unmeasured.** Read **actual** spend from the provider afterwards and record it.
**If it lands materially outside $0.62–$3.70 that is a finding about the instrument** — the bound was
built from source-text volume and would have mis-modelled how the pipeline consumes it. Separately:
**the pipeline records no token usage at all**, so no run in this sprint can be costed after the
fact. That observability hole is registered in D-086 and is worth closing.

### Scoring — what the report must say

Produce the full table from `T107-READINESS.md` § 1, measured on the new run:

* **Priority 1** — **raw** count and composition, **accepted** count and composition, status
  `PASS` (0–6) / `PASS_WITHIN_VARIANCE` (7) / `FAIL` (8+), every applied case-scoped tolerance, and
  **confirmation `LpxH` remains counted** on PMC12444477.
* **Priority 2** — eligible-leg results, the exact `NOT EVALUATED` count, the D-067 precondition-3
  reason, and whether it is `CONDITIONALLY SATISFIED` (D-075). **Never reported as full 20-leg
  biological validation.**
* **Priorities 4/5** — raw **and** accepted coverage on the **D-080-ratified** definition:
  `eligible = raw_anchors − case_scoped_forbidden`, numerator from matched eligible only, denominator
  from eligible only, raw preserved separately.

**Priority 1 is genuinely uncertain between 7 and 8.** T-104 = 7, T-105 = 7, T-106 re-scored at the
merged tip = 8. **7 is `PASS_WITHIN_VARIANCE`, clears the gate, and must NOT be re-drawn** (D-073).
**8+ is an honest acceptance failure and is reported as one.** A new systematic defect is still a
defect even inside the band. **A leg is never repeated because its draw is unfavourable; something
not observed is reported as "not observed", never chased.**

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
