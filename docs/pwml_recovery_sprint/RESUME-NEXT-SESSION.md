# RESUME — next session handoff

**Rewritten by the Lead Orchestrator, 2026-08-28, the acceptance-instrument reconciliation wave.**
Supersedes the O-1 ruling wave record, which is in git history and in `LEDGER.md`. **`LEDGER.md`
remains the single source of truth for task state.**

> **⚠ Keep this file in the repo and update it in place.** A prior session wrote its handoff to a
> session-local scratchpad and the next session could not find it. This wave proved the point a third
> time, in the worst way: the **only** record of what the F-132 measurement found was a probe in a
> dead session's temp directory, and the same was true of C-100's `03`/`04` probe stdout. Both were
> recovered this wave. **One of the two probe sources is gone for good.**

---

## 1. Integration state

| | |
|---|---|
| Branch | `sprint/pwml-recovery` |
| **Do not pin a tip SHA here** | the invariant is **local = `origin/` = `git ls-remote`**. Read it, do not recall it |
| Merges to `main` | **none, and none permitted** |
| `main` | **local `7531692`, remote `03f1af5`.** `main` advanced **outside this sprint**; `7531692` is an ancestor of `03f1af5`. Verified: the sprint tip is **not** an ancestor of remote `main` — no sprint work has leaked. **Touch neither ref.** A previous handoff reported "main untouched" having checked only the local ref |
| Product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` — verified intact repeatedly |
| Caches, `topics_*.txt` | uncommitted, untouched |
| Whole-tree G11 | **0 non-compliant.** The count is self-referential — a whole-tree check's own report is committed after it runs, so the recorded number is always one less than the tree that contains it. Reconcile, do not panic |

## 2. What this wave settled

**All four product rulings are recorded: D-072 (A), D-073 (B), D-074 (C), D-075 (D).**

**The T-107 blocker changed character.** The previous wave recorded gate condition 9 as *"not met and
not reachable by any engineering in this sprint"*, because only ask B could clear it and B was
unanswered. **B is now answered by D-073** and condition 2 by D-075. T-107 is still NO-GO, but it is
blocked on **mergeable cards**, not on an unanswerable product question.

**F-142 — the Glutathione red is diagnosed.** Four cards carried it on *"fails at base"*. It is a
**stale expectation, not a production defect**. The coverage gate is entirely correct; C-041a
(`4177fe5`, D-002) deliberately moved a `minimum_core:*` shortfall to `review_reasons`, so `ok`
stopped answering the question the fixture asks it. The protection moved and is richer —
`review_required`, `strict_acceptance_eligible false`, `completeness 0.0`, all three anchors named.
Merge rule 7 and `has_surviving_core`'s own docstring require the current behaviour verbatim.
**Not a T-107 blocker.**

**D-076 ratifies C-101's ceiling 420 → 560** and corrects two errors in my own charter — see § 6.
**D-077 registers, unanswered, whether D-073's "authorized, case-scoped tolerances" is currently
vacuous for Priority 1.** No code depends on the answer.

## 3. Cards

| Card | Branch / worktree | State |
|---|---|---|
| **C-101** — 16/5 split · row-aware sentinel seam · raw/accepted Priority 1 | `card/C-101-o1-metric-split` @ `06a03a7` · `C:/t/c101` | **COMPLETE, UNDER REVIEW.** REV-101 dispatched. Gates: SMOKE 473=473, gold-readers identical, 34 new tests pass, G9 regression passes at base and tip. **Not merged** |
| **C-102** — coverage denominator vs `forbidden_identifiers` (Ruling A) | *(not dispatched)* | **CHARTERED.** Blocked on C-101 merging — serial `bench/` ownership |
| **C-103** — re-point the F-142 replay expectation | *(not dispatched)* | **CHARTERED.** Sequenced last. Not a T-107 blocker |

**C-101 and C-102 are NOT parallel work.** Serial ownership of `src/t2pw/bench/` is a hard
requirement even though they touch different functions.

**I deliberately did not dispatch C-103 in parallel with REV-101.** Its files are disjoint, but both
would want the heavy lock, and "one heavy job at a time" is a hard rule. The cycle saved is not worth
a job dying on `EXIT_HEAVY_LOCK_UNAVAILABLE` (95).

### The gold-readers baseline every card in this wave must be told

The 22-file gold-readers selection **exits 1 at base, correctly** — it contains
`test_strict_failure_replay.py` and its two F-142 reds. **Expect exactly two; a third is the card's.**
**C-103 changes this** and is required to hand back the corrected baseline (expected `0 failed`,
exit 0) so every later charter is updated.

## 4. Evidence rescued this wave — and the standing lesson

| What | Where it is now |
|---|---|
| The F-132 probe + its full output (62/281, 32 legs, 6 papers) | `evidence/orch702_f132_forbidden_anchors.py` / `.log` |
| C-100's `03`/`04` probe stdout | `evidence/c100_0{3,4}-*.RECOVERED.log`, `c100_probe_stdout.RECOVERED.md` |
| C-100's probe **source** | **GONE. Unrecoverable.** Recorded so nobody hunts for it |

**A G11 report certifies that a job was clean and preserves nothing about what it found.** Commit the
probe *and* its log, every time.

**The recovered F-132 log also corrects the bundle's prose:** the population spans **five** mechanism
kinds, not four — `placeholder_product` 19, `heading_or_prose` 19, `regulator_as_metabolite` 16,
`cofactor_as_protein` 6, **`modification_state` 2** (omitted there). 62 total.

**Independently measured and committed:** `evidence/orch713_gold_selfconsistency.py` / `.log` — **zero
overlap** between any gold case's declared positive fields and its own `forbidden_identifiers`, all
ten cases. **This is a right answer to a narrower question than Ruling A asks.** It establishes the
contradiction is introduced **at runtime by the Stage-0 draw**, not authored into the gold — which is
why the bundle's *"Gold: None"* holds and the fix belongs in the scorer. **It is not evidence F-132 is
absent**, and C-102's charter says so explicitly.

## 5. T-107 readiness — live blocker list

| # | Condition | State |
|---|---|---|
| 1 | Priority 1 not guaranteed to fail | **CLEARED** by D-073 |
| 2 | Priority 2 not guaranteed to fail | **CLEARED** by D-075 |
| 3 | C-101 merged + independently approved | **OPEN** — under review |
| 4 | Ruling-A card merged + independently approved | **OPEN** — chartered |
| 5 | Rulings A–D recorded | **DONE** |
| 6 | Glutathione classified, not a guaranteed failure | **DONE** — F-142 |
| 7 | deterministic gates green; integration pushed + remotely verified | rolling |
| 8 | lock free · zero sprint Python · no peer owning an overlapping job | **DONE**, re-verified |
| 9 | LM Studio + pinned model; run monitorable; spend ≤ $5 | **not assessed** — assess only when 3 and 4 close |

**No live model run, no cohort, no benchmark, no paper leg this wave. External spend $0.00 of $5.**

## 6. Traps this wave paid for — additional to the standing list

* **A charter can send a card looking for an object that does not exist.** `AMENDMENT 1` § A4 named
  *"the row used by C-100's accepted A/B"*. **That A/B is a test-node A/B** — 20 SMOKE + 22
  gold-readers = the "42 files" — and contains **no payload row**. The card found the right row by a
  better route (`orch710_pinned21.json` + the LEDGER's "which run is the pinned run" correction) and
  **recorded the discrepancy instead of smoothing it**. Corrected in D-076.
* **Committing an amendment moves the tip out from under the base SHA the amendment names.**
  `AMENDMENT 1` says base `b30193f`; the dispatched worktree carries `d7cf4a4`. **Re-read
  `git rev-parse HEAD` in the worktree; never trust a SHA written in a charter.**
* **A ceiling raised by estimate for tripled scope will be wrong.** Sixth under-set ceiling of this
  sprint. Derive from deliverables, not by feel. And **a ceiling breach with a clean boundary and a
  documented trim is a mis-set ceiling, not scope creep** — check the boundary before deciding which
  it is.
* **`is_pathbank_unknown_protein` requires four clauses.** A "not a sentinel" control that fails all
  four at once cannot tell you which clause is load-bearing. Perturb them **one at a time**.
* **A three-valued status degrades silently into a Boolean at the render site.** Grep every render
  site for `PASS_WITHIN_VARIANCE`; distinct in the model and identical in the output is a fail.
* **Agent liveness is tracked separately from job liveness.** A subagent sat at `running` for twelve
  hours this sprint. 15 min without observable progress → status request; 30 min with no response, no
  process, no artifact and no commit → stalled.

## 7. Peer sessions

`project14-t2pw-14` **stood down explicitly** this wave — owns nothing, holds no lock, runs nothing,
will message before touching the tree. Its worktrees (`c099`, `c100`, `c099base`, `c100base`,
`c099g9`) and the older set are recorded; **prune none of them**, nor `C:/t/c101` / `c101base`.
**`project14-t2pw-51`, the C-099/C-100 reviewer, is gone** and its context with it. Its method is
preserved in `prompts/REV-101.md` § 4 — that is the only place it now exists.

**Run `ListAgents` and contact every peer before treating the branch, the lock or the worktrees as
exclusively yours.** Session identities are not stable.
