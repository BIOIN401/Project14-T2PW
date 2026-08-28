# RESUME — next session handoff

**Rewritten by the Lead Orchestrator, 2026-08-28, at the close of the acceptance-instrument
reconciliation wave.** Supersedes the previous record; that content is in git history.
**`LEDGER.md` remains the single source of truth for task state.**

> **⚠ Keep this file in the repo and update it in place.** Two load-bearing probe outputs this sprint
> existed only in dead sessions' temp directories. Both were recovered this wave; **one probe source
> is gone for good.** A G11 report certifies a job was clean and preserves **nothing** about what it
> found.

---

## 1. Integration state

| | |
|---|---|
| Branch | `sprint/pwml-recovery` |
| **Do not pin a tip SHA here** | the invariant is **local = `origin/` = `git ls-remote`**. Read it, do not recall it |
| Merges to `main` | **none, and none permitted** |
| `main` | **local `7531692` untouched; remote `03f1af5`.** `main` advanced **outside this sprint**. Verified: the sprint tip is **not** an ancestor of remote `main` — nothing leaked. **Touch neither ref.** A previous handoff reported "main untouched" having checked only the local ref |
| Product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` — verified intact throughout |
| Caches, `topics_*.txt` | uncommitted, untouched (7 entries) |
| Whole-tree G11 | **0 non-compliant.** The count is self-referential — a whole-tree check's own report is committed after it runs, so the recorded number is always one less than the tree containing it. Reconcile, do not panic |

## 2. What this wave did

**Three cards merged, each independently reviewed, each with the reviewer finding something real.**

| Card | Merge | What it did |
|---|---|---|
| **C-101** | `ee7cb6b` | 16/5 metric split · row-aware PathBank sentinel seam · raw/accepted Priority 1 with the variance band. **3 correction rounds** |
| **C-102** | `8e4334f` | Coverage denominator reconciled against `forbidden_identifiers` (Ruling A). **1 correction round** |
| **C-103** | `511344a` | F-142 replay expectation re-pointed at the seam that carries the verdict. **Zero production lines.** 1 correction round |

**Rulings recorded:** D-072 (A) · D-073 (B, the variance band) · D-074 (C, the exact sentinel) ·
D-075 (D, Priority-2 `NOT EVALUATED`) · **D-080** (interpretation: D-072's "denominator" means both
sides — **flagged for product-owner ratification**) · D-081 · D-076/D-079/D-082 (budget) · D-077 ·
D-078 · D-083 · **D-084** (byte-exact mutation restores).

**Findings:** F-142 (Glutathione = stale expectation) · F-143 (bare `python` resolves to the system
interpreter) · F-144 (a non-vacuity guard can guard the wrong emptiness) · F-145 (the F-132
population is 92/47/**7**, not 62/32/6).

## 3. The gate numbers every future charter needs

| Gate | Result on the integration tip |
|---|---|
| **SMOKE** (20 files) | **473 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** |

**⚠ The gold-readers baseline CHANGED this wave.** It exited 1 correctly all sprint because
`test_strict_failure_replay.py` carried the two F-142 reds. **C-103 cleared them.** Any charter still
carrying *"this selection exits 1 at base, and that is correct"* is **stale** — delete that warning.

## 4. T-107 — NO-GO, and the blocker is one product decision

Full assessment, measured at the merged tip: **`docs/pwml_recovery_sprint/T107-READINESS.md`**.

**Twelve of thirteen gate conditions hold.** The failure is **condition 8, the pinned model**, and it
is configuration and authorization, **not engineering**:

* `.env` pins all nine OpenRouter slots to **`deepseek/deepseek-v4-flash`**, which a read-only models
  check prices at **$0.0868/M prompt, $0.1736/M completion — paid**.
* The T-101/T-103 authorization's **≈$0 basis rested on every slot being `openrouter/free`**. It no
  longer is. `.env` is untracked, so the change is unattributable through git.
* **LM Studio cannot substitute** — it serves `glm-4.6v-flash`, which is neither the pinned model nor
  the configured `LOCAL_MODEL`. Using it would be a **fallback model** (forbidden) and would destroy
  comparability with T-104/T-105/T-106, all three of which ran `deepseek-v4-flash`.
* Spend cannot be bounded in advance: ~5 h, 20 legs, nine model roles, no spend telemetry to abort
  on, hard ceiling.

**The decision needed, and only the product owner can make it:**

* **(A)** authorize paid spend on the pinned model with an explicit ceiling — **keeps comparability,
  which is the entire point of the pinned plan**; or
* **(B)** re-pin to a free model and accept **in writing** that T-107 is not comparable to its three
  predecessors.

**(A) is better** if any paid spend is acceptable. **Do not run T-107 until this is ruled** — it may
be launched only once and must not be rerun for composition.

### The acceptance table at the merged tip

| Priority | Raw | Accepted | Status |
|---|---:|---:|---|
| 1 | **8** | **8** | `FAIL` (7 = `PASS_WITHIN_VARIANCE`; T-104 and T-105 both scored 7) |
| 2 | 0 counted | — | **`CONDITIONALLY SATISFIED`** — 9 of 20 eligible, 11 `NOT EVALUATED` on D-067 precondition 3 |
| 3 | 0 | — | `PASS` |
| 4 | `0/8` | `0/8` | `FAIL` (not a hard gate) — **unmoved by Ruling A**, see D-081 |
| 5 | `0/2` | `0/2` | `FAIL` (not a hard gate) |

**`LpxH` confirmed still counted:** `PMC12444477/strict` = 8 findings including `LpxH`, `Unknown`
gone — **9 → 8, never 9 → 7**. `accepted == raw` is a **measurement**, not a construction: under
D-074 no Priority-1 row can be contract-adjusted at all (**D-077**).

## 5. Open items, none blocking

* **D-080 wants product-owner ratification** — the interpretation that D-072's "denominator" means
  numerator *and* denominator. The code is merged on that reading; the measurement forcing it is nine
  legs reporting a ratio above 1 under the literal text.
* **D-083** — two C-103-adjacent follow-ons: F7's deep copy has no test (its revert mutation is
  green), and the split-gate driver should abort on `errors > 0`. Evidence tooling and one low-stakes
  missing test. Route to a housekeeping card.
* **F-145** — the decision bundle's § 2 population figures are an undercount and should be quoted as
  **92 / 47 / 7**.
* REV-103 did not audit whether `test_c074_strict_core_floor.py` / `test_c072_incomplete_core_demotion.py`
  pin their caps **non-vacuously**. If F-142's no-coverage-gap claim needs to be load-bearing, that
  audit is a follow-on.

## 6. Traps this wave paid for — additional to the standing list

* **F-144 — a non-vacuity guard can be real and still guard the wrong emptiness.** Asserting that *a*
  finding was produced is not evidence that *the path under test* produced it. Remedy, and it is the
  rule now: **a non-vacuity guard is not evidence until a party who did not write it has failed to
  defeat it.** Every card from C-102 on was dispatched with a mutation-proof requirement.
* **F-143 — pass the explicit venv interpreter.** `bounded_run.py` resolves a bare `python` from the
  child's PATH → the system 3.13, no `streamlit`, **35 spurious import errors that read exactly like
  a regression**. `pinned_pytest`'s exit-98 check verifies the *tree*, not the *interpreter*.
* **D-084 — restore saved BYTES.** `git checkout --` reverts **more** than it mutated (it takes a
  card's uncommitted edits); text-mode restore reverts **less** than byte-exactly (CRLF→LF). Three
  sessions hit these two opposite failures in one card.
* **Measurement failures wear the costume of results.** A missing `--basetemp` parent reported **382
  instead of 453**. The tell is many failures in an *unrelated* shape — `KeyError`, collection errors
  — not the targeted assertion.
* **A recovered artifact records the tree it ran against, not the tree it lands in.** I committed the
  recovered ORCH-702 log without re-running it; it was stale at the moment of commit.
* **Report byte sizes are worktree-path-dependent** — acceptance reports embed the absolute run-dir
  path. Quote the tree.
* **Ceilings: seven under-set, seven mine.** Derive from the mandated deliverables at the measured
  density of the module. **Review-mandated work never charges ceiling 1 or ceiling 2** (D-076 A1,
  D-082), and **kept failed runs are never charged** — charging for a mandatory artifact is a tax on
  honesty.

## 7. Peer sessions

`project14-t2pw-14` **stood down explicitly** this wave and holds nothing.
**`project14-t2pw-51`, the C-099/C-100 reviewer, is gone**; its method survives only in
`prompts/REV-101.md` § 4. **Run `ListAgents` and contact every peer before treating the branch, the
lock or the worktrees as exclusively yours.**

**Prune no worktree.** Added this wave: `C:/t/c101`, `c101base`, `c102`, `c103`, `rev102base`,
`rev102tip`, `rev102r1`, `rev103base`, `rev103tip` — plus everything the previous waves listed.
