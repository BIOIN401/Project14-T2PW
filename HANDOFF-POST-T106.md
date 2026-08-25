# Handoff — the correction wave is merged and T-106 has run

Written 2026-08-24. Integration tip **`1bbb1a9`** on `sprint/pwml-recovery`, pushed,
`local = origin = git ls-remote`. **Supersedes `HANDOFF-T106.md`.**

---

## 0. Verify once, then work

| check | expected |
|---|---|
| tip | `1bbb1a9`, local = origin = `git ls-remote` |
| merge in progress / staged | none / 0 |
| heavy lock `C:\t\heavylock` | absent |
| sprint-owned Python processes | 0 (two `ms-python.isort` IDE servers — **never kill those**) |
| product-owner edit `src/t2pw/app/streamlit_app.py` | **35 ins / 2 del, uncommitted**, file `sha256:47e4fafa789d359d…` |
| SMOKE | **473** (measured after all five merges) |
| `TEST_MATRIX.md` | 578 lines, line 477 byte-identical |
| `data/*cache*.json` | modified, **never staged, never committed** |

**Correction to the previous handoff.** It recorded the product-owner file hash as
`sha256:e50a248bb7189c22…`. The measured value is `47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632`,
and the file's mtime predates that handoff's own, so the file never moved and the recorded hash was
simply wrong. **Use the measured value.** The load-bearing invariant — 35/2, uncommitted — held
through eleven commits and five merges.

---

## 1. T-106 — the headline

**`runs_verify/2026-08-24_1428`, committed `efca465`. `COMPLETE (10/10 papers, 20/20 legs)`,
5.36 h. STATUS: `MEASURED — NOT ACCEPTED`. Never record as PASS. Do not rerun under this identity.**

| priority | T-105 | **T-106** |
|---|---|---|
| 1 zero false real identifiers | FAIL 7 | **FAIL 8** |
| 2 zero unsupported reactions | FAIL 3 | **PASS 0** |
| 3 zero referential violations | FAIL 2 | **PASS 0** |
| 4 requested-pathway coverage | PASS 1/8 | **PASS 1/8 = 12%** |
| 5 strict PWML pass rate | FAIL 0/4 | **FAIL 0/4** |

Reports: `docs/pwml_recovery_sprint/evidence/t106_acceptance_report.txt`, `t106_verify_plan.txt`.
Full record and the prediction-vs-outcome triage are in `LEDGER.md` under *T-106 — RAN 2026-08-24*.

**Read `docs/pwml_recovery_sprint/T106_PREDICTION.md` before interpreting anything.** It was
committed *before* the run and it is the reason the result is classifiable rather than arguable.

---

## 2. What is done — do not redo

**Five cards merged, each independently reviewed against the actual diff, SMOKE 473 after every one:**

| card | merge | closes |
|---|---|---|
| C-076 | `3b7a7b1` | F-102 — scorer + gold identity |
| C-077 | `26fa809` | F-095 / **D-062** disposition |
| C-078 | `4797f58` | F-099 (amended LOW → HIGH) |
| C-079 | `15a8a15` | F-105 — 64,952 bytes out of every interactive prompt |
| C-080 | `89aaced` | F-108 — production release gate |

**Three live runs, 30 legs, all clean** (`FINAL SURVIVING COUNT 0`, `cleanup success` on every job):
cohort A `2026-08-24_1203` (8 legs), cohort B `2026-08-24_1402` (2), T-106 `2026-08-24_1428` (20).
Everything before them — five merges, six charters, ten findings, two amendments — cost **zero
credits**.

**Findings registered:** F-106 … F-115. **Amended:** F-099 (LOW → HIGH), F-092 (two of three defects
**refuted**, not reclassified).

---

## 3. What the wave actually bought, measured on a release candidate

* **Zero bare `pathway.pwml` across all 20 legs.** Five strict legs emitted
  `pathway.review_required.pwml`, each `strict_acceptance_eligible: false`. F-100 and F-101's class
  is closed.
* **Priorities 2 and 3 moved FAIL → PASS.**
* **C-077 works on real legs.** Six `scope_conflict` legs, all six now carrying `diagnostic_only`,
  `pipeline_executed: true`, and `requested_scope` beside `observed_context`, where T-105 had `null`.
* **C-076's delta realised:** `holo-EntB` is gone from priority 1 on a fresh draw.
* **The source-support pass is live and withholding** — `source_text_index` on every payload; 8 and
  2 identifiers withheld on PMC12856317 in cohort A.
* **C-078 armed and correctly inert** across all 10 cohort legs — non-vacuity measured in
  production, not fixtures.

---

## 4. Why it is NOT ACCEPTED, and what to do about it

**Priority 1 = 8.** `NADH`, `NAD+`, `LIPA`, `LBR`, `SREBF1`, `SREBF2`, `pyridoxal 5-phosphate` ×2.
**Every one is a Stage-1 extraction hallucination handed a real identifier downstream** — the
pre-existing **F-096** class. Not one is an accession-conflict case; not one is a restored-refused
identifier. **No card in this wave was chartered to fix it**, and 7 → 8 is composition churn, not
regression: `SREBF1/2`, `LIPA` and `LBR` are back after vanishing from T-105 by draw variance.

**This is the next real target.** It is a Stage-1 extraction-quality problem, not an identity-layer
one, and the identity layer is now demonstrably doing its job around it.

**Priority 5 = 0/4.** No strict-exportable paper produced a `release_ready` export; five produced
`review_required` instead. That is correct behaviour under merge rule 7, not strict success.

---

## 5. Open work, in priority order

| item | class | blocks | note |
|---|---|---|---|
| **F-096 / priority 1** | `product_contract_violation` | acceptance | Stage-1 hallucination handed real identifiers. The wave's remaining acceptance gap. |
| **F-110** | `product_contract_violation` | no | name gate cannot relate `ferric iron`↔`Fe3+`, `Zn2+`↔`Zinc (II) ion`. **C-078 makes it bite.** `mapping/`. Predicted, did not materialise on these draws — `Zn2+` came back `ambiguous` both times. |
| **F-115** | `product_contract_violation` | no | `AMBIGUOUS_RENAME_TARGET` ends a leg as a *crash* rather than preserving the payload. New from T-106. |
| **F-092 defect 3** | `product_contract_violation` | no | inner deadline path discards a computed `operation_timeout`. **Not observed at T-106** — the timeouts did not reproduce. Still open on T-104/T-105 evidence. Fix lands in `driver.py`. |
| **F-107** | `policy_disagreement` | no | **product ruling needed.** `PRODUCT_CONTRACT` §4 has no state for "defensible core extracted but never serialized". D-062 assumed one existed. |
| **F-109** | `control_plane_contradiction` | no | **doc owner's call.** rule 10 forbids `pythonpath = src`; `pytest.ini:8` has it. Every base proof this wave was pinned, so none is in doubt. |
| **F-113** | `control_plane_gap` | no | the 2026-08-23 identity ruling has no `D-xxx` entry, yet two merged cards rest on it. |
| **F-111 / F-112 / F-114** | tooling / staleness | no | G11 checker blind to a hand-formed path; two pinned censuses stale (T-106 will break them a third time); `--basetemp` parent-missing is a second false-regression mode. |

**Standing rule until F-109 is ruled on:** every base-tree measurement runs through
`pinned_pytest.py` with `--expect-tree` and a committed `--pin-verdict`. An unpinned base run is not
evidence.

---

## 6. State notes for whoever picks this up

* **11 G11 staging reservations are pending** across C-073, C-076, C-078, T-101, T-105 and T-106.
  These are **explained, not stray**: F-071's design is that a job which reserves a path and never
  writes a report leaves nothing in the reports tree. Most predate this session. **Do not delete
  them**; the gaps in the sequence are the intended visible residue.
* **Leftover worktrees** `C:/t/c080` and `C:/t/rev080base` remain — `git worktree remove --force`
  was denied by a permission rule and cleanup is optional (§15). Both are merged and clean.
* **Do not commit** `data/enrichment_cache.json` or `data/id_mapping_cache.json` (39 MB, tracked).
* **`cache_snapshot/` is excluded** from all three run-directory commits (43 MB each).

## 7. Traps paid for this session — do not pay again

* **The Bash tool's 2-minute default timeout kills `bounded_run.py` before its `finally`.** The Job
  Object still kills the children (zero survivors), but the heavy lock is stranded. Pass an explicit
  `timeout` longer than the wrapper's, or run in background.
* **`g11_evidence.py next --task` validates its id.** `--task ORCH` is rejected, and a
  `$(… | tail -1)` silently captures the *error text* as your `--json` path. Check the reserved value
  looks like a path.
* **Never hand-form a `--json` path and promote it afterwards** (F-111) — always pre-allocate.
* **A `--basetemp` whose *parent* is missing errors the run** (F-114), alongside the documented
  omit-it-entirely `PermissionError` mode. Both look like large regressions and are neither.
* **Reviewers must re-measure SMOKE themselves.** The G11 JSON retains no stdout and pin verdicts
  carry no counts, so an implementer's figure is not recoverable from committed evidence. Every
  reviewer this session ran both sides.
* **Charters get corrected by implementers, and that is the process working.** Four were corrected
  this session — C-078's most consequentially, where my "control row" turned out to be the only
  realized hit and my reasoning was exactly inverted.
