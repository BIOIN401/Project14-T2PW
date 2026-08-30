# PWML Recovery Sprint — Ledger

**Single source of truth for task state. Written by the Lead Orchestrator only.**

States: `BLOCKED` → `READY` → `IMPLEMENTING` → `REVIEW` → `CORRECTION` → `INTEGRATION`
→ `BATCHED_VALIDATION` → `ACCEPTED`. Also terminal: `REJECTED`.

**Statuses are updated as one compact batch at each pack closeout** — status, merge SHA and
reviewer-verdict cells only. No essay-length status cells.

Column meanings:

| Column | Meaning |
|---|---|
| Base SHA | integration commit the branch was cut from, recorded at dispatch |
| Worktree | `.claude/worktrees/<name>` — implementers only; reviewers and test runners work in-place |
| Ownership boundary | exact files :: functions. A diff outside is an automatic reject |
| Reviewer | must be a different agent than the implementer |
| Focused | test chunks the branch owes before review (see `TEST_MATRIX.md`) |
| Integration | suite run after merge, on the integration branch |
| Merge SHA | integration SHA after this branch merged |
| Bench delta | milestone benchmark movement attributable to this branch |
| Blockers | what is actually stopping it now — a task ID, or a named gate such as product-owner approval. A dependency that has cleared does not belong here |

---

## Wave 0

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INIT-001 | Sprint init + baseline capture | `ACCEPTED` | — | `721a256` | `sprint/pwml-recovery` | none | `BASELINE.md`, evidence commit, `bounded_run.py` | product owner | full suite ×10 chunks | smoke 457 ✔ | `0c469f7`, `0132cb8` | — | — |
| SPIKE-002 | Compound-resolution extraction scoping | `ACCEPTED` | INIT-001 ✔ | `2b786aa` | none (no code) | none | none — investigation only | `pwml-reviewer` | none | n/a — no code | `cfd0e10` | — | — |
| R-003 | False-identifier triage (10 findings) | `READY` | INIT-001 ✔ | read at dispatch | none (read-only) | none | none | Lead | none | — | — | — | subagent registration |
| R-004 | RAG-reintroduction triage (3 claims, PMC12657337) | `READY` | INIT-001 ✔ | read at dispatch | none (read-only) | none | none | Lead | none | — | — | — | subagent registration |

## Pre-Wave-A0 — test-harness maintenance (D-012, D-013)

**These block C-010 and all of Wave A0.** The replay merge gate must be green, on a
frozen cohort, before any implementation branch is dispatched against it.

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H-001 | Freeze the baseline cohort to a manifest | `ACCEPTED` | — | `2b786aa` | `agent/h01-baseline-manifest` | `.claude/worktrees/agent-a6a00cb7c4bd4286b` | `test_strict_quarantine_real_artifact_replay.py` :: `RUNS`, `_legs`, `_leg_ids`, `FULL_STACK_BASELINE`, `RESIDUAL_CODES_BY_{LEG,ROW}`, `test_the_full_stack_baseline_...`; NEW manifest; `docs/change_log.md` baseline table. **Size deviation (closeout review D-5):** landed **≈637 changed lines** (591 insertions + 46 deletions at `aa0fb0a`) against the **~400-line `[S4]` stop condition**, which obliged the implementer to stop and propose a split. It did not, and **the deviation was not recorded as one at the time**. Mitigation, not excuse: **212 of the 237 excess lines are generated manifest data** — the whole of the new `tests/data/baseline_cohort_manifest.json`, whose 39 entries were verified before freezing. Net of the manifest the branch still changed ~425 lines, i.e. still over the bound. No file outside the stated boundary was touched | `pwml-reviewer` — **APPROVE** | A 123 ✔, E 165p/1f ✔, smoke 457 ✔ | smoke 457 ✔, A 123 ✔, D 177 ✔, E 165p/1f ✔ | `aa0fb0a` | — | — |
| H-002 | Scope the replay assertion to payloads | `ACCEPTED` | — | `aa0fb0a` | `agent/h02-replay-payload-scope` | `.claude/worktrees/agent-aee0bf8fcd24a3886` | **Stated:** `test_strict_quarantine_real_artifact_replay.py` :: `test_no_archived_leg_carries_stage_zero_context` **only**, + granted text-only module-docstring extension. **Landed (closeout review D-2):** the same single file, but **7 new tests, 4 new module-level helpers, 3 new module-level constants, 351 insertions / 12 deletions** at `e5eeb8c`. That is a **departure from the stated one-function boundary** and is recorded as one. It is also true that the new tests were **required by G9** — a regression test that fails on the base SHA is mandatory — and that the correction's restored tripwire landed as its own enforced test (`test_no_unlisted_artifact_quietly_carries_stage_zero_context`), outside the one named function. All 4 helpers and all 3 constants are module-private and are referenced by no file other than this test module. Both facts stand: the G9 obligation explains the departure, it does not retire it | `pwml-reviewer` — **APPROVE** after 1 CORRECTION round | E 173 ✔, A 123 ✔, smoke 457 ✔ | smoke 457 ✔, A 123 ✔, D 177 ✔, E 173p/0f ✔ | `e5eeb8c` | — | — |

Same file, different functions. Whichever merges second rebases and re-runs chunk E
before review; they are not reviewed as one unit.

## Pre-Wave-A0 — G11 infrastructure (D-017, D-018)

Opened 2026-08-05 by product-owner ruling, on findings the closeout review produced.
G11 is a hard merge rule that is currently **unenforceable after the fact**: the wrapper
can be killed by its own output-forwarding and leave no cleanup report, and no report is
committed anywhere, so no G11 claim in this sprint is checkable against an artifact.

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H-003 | Repair the wrapper's Unicode output-forwarding defect | `ACCEPTED` | — | `5a476a5` | `agent/h03-wrapper-unicode-drain` | `.claude/worktrees/agent-a2087a6d4c01172df` | `evidence/bounded_run.py`, `evidence/bounded_run_selftest.py` — **temporary ownership grant, this task only; it expires with this task**. **Size deviation, accepted deliberately:** 441 touched lines (410 insertions / 31 deletions) against the ~400 `[S4]` bound = **10.25% over**. 423 were self-declared by the implementer at first submission; the remaining 18 came from the owner-authorised C1/C2 correction round. Attributable to the G9-mandated tests, which cannot be split from the fix. Code-vs-prose split of the net growth: +213 executable, +166 blank/comment/docstring | `pwml-reviewer` — **APPROVE** (exact) after 1 authorised CORRECTION round; verified by `pwml-test-runner` | selftest 8/8 ✔, smoke 457 ✔, E 173p/0f ✔ | smoke 457 ✔, E 173p/0f ✔ | `aab975a` | — | — |
| H-004 | Durable, version-controlled G11 evidence | `ACCEPTED` | H-003 ✔ | `ede8c32` | `agent/h04-durable-g11-evidence` | `.claude/worktrees/agent-a4e9c764eb0dd0323` | `TEST_MATRIX.md` § 0; `_SHARED_BLOCKS.md` block S8; NEW `evidence/g11/`. **Size deviation, accepted:** 917 changed lines, 272 wrapper-generated, **645 hand-authored = 61% over** the ~400 `[S4]` bound. The implementer committed and self-declared rather than stopping; the reviewer ruled the atomicity argument sound — **neither half of the offered H-004a/H-004b split passes the card's own ACCEPTANCE list**, so the card was scoped past its bound by the orchestrator. A card-scoping defect, not implementer scope creep | `pwml-reviewer` — **APPROVE** (exact); verified by `pwml-test-runner` | selftest 3/0 ✔, check 3 artifacts 0 non-compliant ✔ | smoke 457 ✔ ×2, E 173p/0f ✔ | `a04a0aa` | — | — |
| H-006 | Wrapper report schema version + build identity | `ACCEPTED` | H-003 ✔, H-004 ✔ | `ad917c2` | `agent/h06-wrapper-report-identity` | `.claude/worktrees/agent-a80c4e39b4c6d3f8d` | `evidence/bounded_run.py`, `evidence/bounded_run_selftest.py`, `evidence/g11/README.md` disclaimer — **temporary ownership grant, this task only**; the README sub-grant was widened twice by product-owner authorisation, to repair citations this branch itself invalidated and then to fix the Git-blob verification command. Reviewed at `9553cd2` (three commits: `4afcc6d` implementation, `5e01428` citation/byte-identity correction, `9553cd2` shell-safe blob command). **Procedural fault, NOT cured:** the branch's **original declared estimate was exceeded** — 572 hand-authored lines against the ~400 `[S4]` bound, 43% over — and **a commit was created before renewed authority was obtained**, where `[S4]` obliges the implementer to stop and report *without committing*. The implementer self-declared the overrun rather than concealing it, and the reviewer independently judged the **acceptance-criterion atomicity** argument sound: every split it constructed fails the card's own ACCEPTANCE list, so this is a **card-scoping defect, not implementer scope creep**. Those two findings are separate: atomicity explains the size, it does not excuse committing before renewed authority. **Subsequent technical approval, test-runner verification and acceptance did not retroactively cure the procedural fault** | `pwml-reviewer` — **APPROVE** (exact, unsuffixed), independently derived against the complete diff from `ad917c2`; verified by `pwml-test-runner` | selftest 12/12 ✔ (12/12 reports, 0 proved survivors), `g11_evidence` 3p/0f ✔, check 26 artifacts / 0 non-compliant ✔ | smoke 457 ✔ | `d167f93` | — | — |

**H-003 is the prerequisite for H-004**: requiring a committed report from every job is
only sound once the wrapper reliably writes one. Neither task may touch `src/`, `tests/`
or `batch/runner.py` (C-032's). **H-004 is prospective only — no historical backfill.**

**Findings H-003 hands to H-004, all verified during its review and verification:**

1. **Compliance must key on artifact presence, never on the wrapper's exit code.** When
   `--json` is unwritable the wrapper deliberately returns the *child's* real exit code
   (ruled correct — a synthetic infrastructure code would lose exactly what the card
   protects). An exit-code-based check would therefore pass a job that wrote no report.
2. **Compliance must require `cleanup_success: true`; `final_surviving_count: 0` alone is
   insufficient.** The reviewer's guard-removal experiment produced a report reading
   `final_surviving_count: 0` **and** `cleanup_success: false` — survivor count kept its
   default because verification never ran, while a child was still alive. A checker
   keying on the survivor count alone would have been fooled.
3. **`json_report_written: false` from a direct `run()` caller is not a violation.**
   `emit_json_report` runs only from `main()`, so in-process callers such as
   `baseline_suite.py:127` legitimately show `json_report_path: ""` and
   `json_report_written: false`. Cases 1–6 of the selftest show the same. Those two
   fields are **not applicable** when `--json` was never requested.
4. **`REPORT_DIR` silently overwrites.** The selftest's default report directory is a
   single fixed temp path, so consecutive runs clobber each other's per-case artifacts —
   two reviewers and the implementer each hit it. Left unfixed **deliberately**: naming
   and durability are H-004's ownership, and H-003 pre-empting that design would itself
   be a boundary breach. The existing `BOUNDED_RUN_SELFTEST_REPORTS` override is the
   intended mechanism and every quoted artifact used it.
5. **No artifact identifies the wrapper build that produced it.** None of the report's
   fields names the *wrapper* — `command` names the child. `[S8]`'s self-reference clause
   requires stating which build produced each result, and today the artifact cannot
   answer that. Relocating the directory is not sufficient; H-004 should add a build
   identifier and forbid silent overwrite.

## Wave A0

**ALL NINE WAVE A0 CARDS ARE MERGED.** H-001 and H-002 are merged and ACCEPTED and the
replay merge gate is green on a frozen cohort; every A0 card was subsequently dispatched,
implemented, independently reviewed and merged into `sprint/pwml-recovery`, ending at
integration head **`0182eae`**. The previous text of this paragraph — *"they are blocked
only on **product-owner approval**, which has not been given. Nothing here is
dispatchable"* — is **false for all nine** and is superseded here.

**Reconciled 2026-08-11 under `CONTROL-PLANE-RECONCILE-001` §5.** This is a factual update
of live task state against the verified merge records; it re-litigates no verdict and
rewrites no history. Every value below was derived from the repository: each **Base SHA**
is the **first parent** of its merge commit (`git rev-list --parents -n 1 <merge>`), each
**Reviewed tip** is the **second parent**, and each **Worktree** is the real attached path
read from `git worktree list`. The nine implementer worktrees all remain **attached**, and
each is parked at exactly the reviewed tip that was merged — an independent cross-check
that the merged commit and the reviewed commit are the same object.

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-010 | p01 stale positional index | `MERGED` | INIT-001 ✔ | `9e06360` | `agent/p01-stale-index` | `.claude/worktrees/agent-a931503c554f6eb56` (attached, at `d784747`) | `strict_quarantine.py` :: `_surviving_processes`, `_degree_zero_exports`, `quarantine_and_close`; `tests/test_strict_quarantine.py`; `tests/test_strict_quarantine_real_artifact_replay.py`; `docs/change_log.md` | C-041 impl *(planned)* — **actual: fresh independent reviewer, no authorship role; exact unsuffixed `APPROVE` of `d784747`, the commit reviewed and the commit merged** | A **126** ✔ (123 + 3 new), E **174** ✔ (173 + 1 new) | smoke **460** ✔ pre-merge — deliberate G4 baseline move 457→460, exact delta documented; G11 11 artifacts / 0 non-compliant | `72ee20f` | M1 planned — **not measured** (T-100 `BLOCKED`) | **Cleared** — dispatched and merged. **Clearing authorization not recorded in the repository** — see note ‡ |
| C-011 | p00a canonical-freeze seam | `MERGED` | H-001 ✔, H-002 ✔ | `85fae43` | `agent/p00a-freeze-seam` | `.claude/worktrees/agent-ab6facabd81769da5` (attached, at `066fb6b`) | `streamlit_app.py` :: `run_post_pipeline_sbml_artifacts` | C-012 impl *(planned)* — **actual: fresh independent reviewer of the rebased tip, told earlier verdicts carried no weight; exact unsuffixed `APPROVE` of `066fb6b`** | golden **2** ✔ ×2, orchestration + lifecycle **40** ✔ combined; chunk D **176p / 1f** — ratified environment-specific baseline exception, **D-020** | smoke **460** ✔ pre-merge, no pinned baseline moved; G11 45 artifacts / 0 non-compliant. **Forward Chunk D gate: IN EFFECT.** The split-process definition merged at **`69d4069`** (RECONCILE-B), and its execution partition was superseded by **H-007** (per-node AppTest isolation); `TEST_MATRIX.md` § Chunk D and **D-022** are authoritative. The earlier cell here read *"NOT YET IN EFFECT — that definition has not merged"*, true when written at `0182eae` and **false from `69d4069` onward**; it is corrected, not deleted. The monolithic **176p/1f** recorded in this row remains the historical C-011 result under **D-020** and still licenses nothing forward | `0182eae` | — (no milestone run) | **Cleared by `C-011-CLOSEOUT-001`** (ceiling 475; named contract ordered and delivered; dispatch attribution **inferred** — note †) |
| C-012 | p00b driver seam | `MERGED` | INIT-001 ✔ | `361b158` | `agent/p00b-driver-seam` | `.claude/worktrees/agent-af19fd2e5bc385c1e` (attached, at `def6adb`) | `driver.py` :: `_drive` → `_finalize_*` | C-011 impl *(planned)* — **actual: fresh independent reviewer, no authorship role, told no prior verdict carried weight; exact unsuffixed `APPROVE` of `def6adb`** | B **225** ✔ on the tip, **227** ✔ co-collected with the new golden file | smoke **457** ✔ pre-merge (23.64 s, exit 0, 0 survivors); G11 4 artifacts / 0 non-compliant | `9e06360` | — (no milestone run) | **Cleared** — dispatched and merged. **Clearing authorization not recorded in the repository** — see note ‡ |
| C-013 | p04a two versioned hashes | `MERGED` | INIT-001 ✔ | `72ee20f` | `agent/p04a-hash-module` | `.claude/worktrees/agent-a9b3b47ac8370377e` (attached, at `fb5a75a`) | NEW `pipeline/canonical_hash.py`; `gate_reports.py` :: `payload_sha256`, `stamp_report`, `gate_verdict` | C-020 impl *(planned)* — **actual: independent reviewer that re-derived every gate itself and audited all 32 committed `final_mapped.json`; exact unsuffixed `APPROVE` of `fb5a75a`** | focused ✔; smoke **460** ✔ | smoke **460** ✔ pre-merge; G11 6 artifacts / 0 non-compliant | `09fb40d` | — (no milestone run) | **Cleared by `WAVE-A0-RESUME-002`** (§2 ceiling 400; dispatch attribution **inferred** — note †) |
| C-014 | p03a LLM request timeout | `MERGED` | INIT-001 ✔ | `09fb40d` | `agent/p03a-llm-timeout` | `.claude/worktrees/agent-a3e8fd47a9d62b545` (attached, at `9c09ee8`) | `llm/client.py` :: `OpenAI(...)`, `chat_detailed`, `chat_with_tools` | C-032 impl *(planned)* — **actual: fresh independent reviewer of the rebased tip, told earlier verdicts carried no weight; exact unsuffixed `APPROVE` of `9c09ee8`** | new file **6** ✔, A **126** ✔, C **109** ✔ — all re-run by the reviewer | smoke **460** ✔ pre-merge; G11 22 artifacts / 0 non-compliant | `c832894` | — (no milestone run) | **Cleared by `WAVE-A0-RESUME-002`** (§2 ceiling 350; dispatch attribution **inferred** — note †) |
| C-015 | p20 lineage schema | `MERGED` | INIT-001 ✔ | `c832894` | `agent/p20-lineage-schema` | `.claude/worktrees/agent-aa8808f30d7a0fce7` (attached, at `101d25c`) | NEW `pipeline/lineage.py` | C-038 impl *(planned)* — **actual: fresh independent reviewer of the rebased tip, told earlier verdicts carried no weight; exact unsuffixed `APPROVE` of `101d25c`** | **24** new ✔, C **109** ✔, smoke **460** ✔ — all re-run by the reviewer | smoke **460** ✔ pre-merge; G11 9 artifacts / 0 non-compliant, no backfill | `8b4bc0c` | — (no milestone run) | **Cleared by `WAVE-A0-RESUME-002`** (§2 ceiling 400; dispatch attribution **inferred** — note †) |
| C-016 | p30 RAG stopping policy | `MERGED` | INIT-001 ✔ | `8b4bc0c` | `agent/p30-rag-stop-policy` | `.claude/worktrees/agent-ae97af9a4a0b99bd2` (attached, at `a0aaa56`) | NEW `rag/loop_policy.py` | C-043 impl *(planned)* — **actual: fresh independent reviewer of the rebased tip; decisive review (corrections exhausted); exact unsuffixed `APPROVE` of `a0aaa56`** | **15** new ✔, C **109** ✔, **124** combined ✔, smoke **460** ✔ | smoke **460** ✔ pre-merge; G11 11 artifacts / 0 non-compliant | `729c40e` | — (no milestone run) | **Cleared by `WAVE-A0-RESUME-002`** (§2 ceiling 300; dispatch attribution **inferred** — note †) |
| C-017 | p40 semantic production module | `MERGED` | INIT-001 ✔ | `729c40e` | `agent/p40-semantic-module` | `.claude/worktrees/agent-a45cc6efe5a6033d6` (attached, at `9479c2d`) | NEW `bench/semantic_production.py` | C-056a impl *(planned)* — **actual: fresh independent reviewer of the rebased tip, all 28 of its commands wrapped; exact unsuffixed `APPROVE` of `9479c2d`** | **8** new ✔, B **233** ✔ combined | smoke **460** ✔ pre-merge; G11 13 artifacts / 0 non-compliant | `fc8b059` | — (no milestone run) | **Cleared by `WAVE-A0-RESUME-002`** (§2 ceiling 500; dispatch attribution **inferred** — note †) |
| C-018 | p50 cofactor / assay-reporter policy | `MERGED` | INIT-001 ✔ | `fc8b059` | `agent/p50-cofactor-classifier` | `.claude/worktrees/agent-a9d33ec5778fa464e` (attached, at `bdc006d`) | NEW `pipeline/cofactor_policy.py` | R-003 *(planned)* — **actual, both on this exact tip and both decisive: exact unsuffixed `APPROVE` from a fresh independent code reviewer, and `BIO_APPROVE_WITH_FINDINGS` from a fresh independent biological auditor, of `bdc006d`** | **36** new ✔, C **109** ✔, **145** combined ✔ | smoke **460** ✔ pre-merge; G11 26 artifacts / 0 non-compliant, no backfill | `85fae43` | — (no milestone run) | **Cleared by `WAVE-A0-RESUME-002`** (§2 ceiling 320; dispatch attribution **inferred** — note †) |

**‡ Flagged, not filled in — C-010 and C-012 have no derivable clearing authorization.**
C-013…C-018 name `WAVE-A0-RESUME-002` and C-011 names `C-011-CLOSEOUT-001` in their own
merge messages, so those seven are derivable. **C-010's (`72ee20f`) and C-012's
(`9e06360`) merge messages name no dispatch or unblocking authorization**, and no such ID
exists anywhere under `docs/pwml_recovery_sprint/`. The only authorization their cards
cite is `WAVE-A0-BUDGET-RATIFICATION-001`, which **cannot** be what cleared them: the
cards themselves state it "Ratifies limits ONLY: no implementation approval, no
unblocking, no branch, no dispatch authority" (`prompts/C-010.md:164`,
`prompts/C-012.md:119`). Both cards were demonstrably dispatched, reviewed and merged, so
the blocker **was** cleared; **which authorization cleared it is not recoverable from the
repository** and is left unstated rather than guessed. For the product owner to supply.

**† The dispatch attribution is an inference, and is marked as one.** All six C-013…C-018
merge messages cite `WAVE-A0-RESUME-002` **only for its §2 changed-line ceiling**, never as
dispatch or unblocking authority; `0182eae` cites `C-011-CLOSEOUT-001` the same way, for the
475 ceiling and the named return contract. That is structurally the same citation shape
rejected for `WAVE-A0-BUDGET-RATIFICATION-001` in note ‡. The attribution is nevertheless
kept, on one piece of distinguishing evidence: **only** `WAVE-A0-BUDGET-RATIFICATION-001`
carries an explicit self-limiting disclaimer ("no implementation approval, no unblocking, no
branch, no dispatch authority"), and neither of the other two does. That is grounds to
prefer the attribution, not grounds to assert it — so it is recorded as **inferred**. For
the product owner to confirm or correct.

**Also for the product owner — `G11-EVIDENCE-ACCOUNTING-001` is cited but never defined.**
It is relied on by all nine Wave A0 card budget blocks, by every A0 merge message and by
**D-021 §7**, yet **no document under `docs/pwml_recovery_sprint/` defines it**; it exists
only in merge-message prose. Every A0 card's evidence accounting rests on it. **No
definition is written here — that is the product owner's authority alone.**

**Reviewer column.** The planned reviewer recorded at dispatch is preserved unchanged in
every row; the actual review outcome is appended beside it. Where the two differ, both
stand — the planned assignment is the historical record and is not rewritten. **Every one
of the nine received an exact unsuffixed `APPROVE` from an independent reviewer of the
exact merged tip**, and C-018 additionally received `BIO_APPROVE_WITH_FINDINGS` from an
independent biological auditor of that same tip.

**Bench delta.** No milestone benchmark has run against any A0 card: T-100…T-105 are all
`BLOCKED` (see § Milestone tests). C-010's `M1` is a *planned* attribution, not a measured
one, and is marked as such.

## Post-Wave-A0 — infrastructure closeout (H-007)

The last bounded control-plane task before Wave A1. Authority `CONTROL-PLANE-CLOSEOUT-002`.
Card: `prompts/H-007.md`. Outcome recorded in **D-022**.

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H-007 | G11 unmatched-selector fix · per-node Chunk D isolation · live-record correction | `ACCEPTED` | — | `08d5d07` | `agent/h07-closeout` | `.claude/worktrees/agent-h07-closeout` (new; the existing 27 untouched) | `evidence/g11/g11_evidence.py`; `evidence/chunk_d_gate.py`; NEW `evidence/chunk_d_gate_selftest.py`; NEW `prompts/H-007.md`; `LEDGER.md`, `TEST_MATRIX.md`, `DECISIONS.md` (append). **Zero files under `src/`, `tests/`, `batch/`; `bounded_run.py` unmodified** | independent non-author reviewer of the exact diff — **APPROVE** (exact, unsuffixed) | g11 selftest **4** ✔, chunk-d selftest **10** ✔; six-run base/candidate Chunk D matrix — see D-022 | smoke **460** ✔; whole-tree G11 0 non-compliant | `4a15230` (reviewed tip `12e4898`) | — | — |

## Wave A1

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-020 | p06a biological-equivalence comparator | `ACCEPTED` | C-013 ✔ | — | `agent/p06a-equiv-comparator` | *(not yet created)* | NEW `pipeline/canonical.py` | independent non-author reviewer — **APPROVE** (exact, unsuffixed) after 1 correction round (`2d8eb56` "read PWML identifiers held in a list container") | new, D | smoke + D | `ad45dcf` (reviewed tip `2d8eb56`) | M3 | **product-owner dispatch** (C-013 ✔ merged `09fb40d` — cleared, no longer blocking) |
| C-021 | p31 RAG graph-delta validation | `ACCEPTED` | C-015 ✔ | — | `agent/p31-rag-graph-delta` | *(not yet created)* | NEW `rag/graph_delta.py` | independent non-author reviewer — **APPROVE** (exact, unsuffixed) | new, C | smoke | `d2b2c1f` (reviewed tip `ec43bc9`) | — | **product-owner dispatch** (C-015 ✔ merged `8b4bc0c` — cleared, no longer blocking) |
| H-008 | RAG evidence retention — close two fail-open holes in `validate_graph_delta` | `ACCEPTED` | C-021 ✔ | `ad45dcf` | `agent/h08-rag-evidence-retention` | `.claude/worktrees/agent-h08-retention` | `rag/graph_delta.py` :: `validate_graph_delta` | independent non-author reviewer — **APPROVE** (exact, unsuffixed) after 1 correction round (`57f024d` "compare the passage fact by containment, not as one value") | new, C | smoke | `3bfa7af` (reviewed tip `57f024d`) | — | — |

Both A1 dependencies have merged, so under the Blockers-column legend above ("a dependency
that has cleared does not belong here") they are shown as cleared and the real remaining
blocker — dispatch — is named. Neither card is dispatched: no `agent/p06a-equiv-comparator`
or `agent/p31-rag-graph-delta` worktree exists in `git worktree list`, so the Worktree cells
read *not yet created* rather than naming a path that is not there. **Waves B, C and D carry
no Blockers column at all** — only `Deps` — and those cells are deliberately left untouched
under the D-6 precedent recorded below: *a satisfied dependency is still a true dependency*.

## Wave B

| ID | Task | Status | Deps | Branch | Ownership boundary | Reviewer | Focused | Bench |
|---|---|---|---|---|---|---|---|---|
| C-030 | p04b hash wiring | `ACCEPTED` merged `f3e9fb1`; APPROVE (exact) — rev2; rev1 stopped and escalated 4 blockers, uncommitted. A0-C1 discharged over the measured 49 rows | C-011, C-013 | `agent/p04b-hash-wiring` | `streamlit_app.py` :: `freeze_canonical_payload` | C-052 impl | D | — |
| C-031 | p02 quarantine artifacts | `ACCEPTED` merged `cbf30f6`; APPROVE (exact) — 0 rounds; golden driver diff empty | C-012 | `agent/p02-quarantine-artifacts` | `driver.py` :: `_add_common_artifacts`, `_add_identity_artifacts` | C-053 impl | B | M1 |
| C-032 | p03b deadline + checkpoints | `ACCEPTED` merged `1801298`; APPROVE (exact) after 1 correction round; budget 950→1100 pre-commit | C-012, C-014 | `agent/p03b-deadline-module` | NEW `pipeline/deadline.py`; `runner.py` :: `_timeout_row`, `launch_child`, `child_command`; `_finalize_timeout` | C-042 impl | B | — |
| C-033 | p10 identity hydration | `ACCEPTED` merged `7ff1211`; APPROVE (exact) after 1 correction round; budget 850→1300 pre-commit | H-001 ✔, H-002 ✔ | `agent/p10-identity-hydration` | `src/t2pw/mapping/map_ids.py` :: 2 fns; `src/t2pw/pipeline/entity_identity.py`; NEW `src/t2pw/mapping/uniprot_evidence.py` (**not** `src/map_ids.py`) | C-044 impl | C | M2 |
| C-034 | p21 lineage: extraction | **RECONCILED 2026-08-18 (Git ancestry):** **`ACCEPTED`, MERGED `aa95518`.** The `DISPATCHED` label below is historical. **`DISPATCHED` 2026-08-13** from `bcc0bfe`, **RE-SCOPED** by product-owner decision | C-015 | `agent/p21-lineage-extract` | **`pipeline/stage_one_boundary.py` :: `settle_stage_one` (lineage writes only)**. Former target `extraction/extract.py` is dead demo code and is now **excluded** | rotate | A + `test_stage_one_boundary.py` + `test_early_failure_replay.py` | — |
| C-035 | p22 lineage: RAG | `ACCEPTED` merged `5ce3870`; APPROVE (exact) — 0 rounds; C-061 reserved fns byte-identical | C-015 | `agent/p22-lineage-rag` | `rag/synthesize.py`, `rag/admission.py` — **except `parse_span_relation` and `validate_evidence_span`, reassigned to C-061 on 2026-08-13.** Both were already byte-identical at merge; **C-035 is not reopened** | rotate | C | — |
| C-036 | p23 lineage: audit | `ACCEPTED` merged `20be28d`; APPROVE (exact) — 0 rounds; authorized 2-line pinned-baseline move | C-015 | `agent/p23-lineage-audit` | `curation/apply_audit_patch.py` | rotate | A | — |
| C-037 | p24 lineage: gap resolver | `ACCEPTED` merged `fbedfe6`; APPROVE (exact) after 1 correction round | C-015 | `agent/p24-lineage-gapres` | `curation/gap_resolver.py` | rotate | C | — |
| C-038 | p25 lineage carrier | `ACCEPTED` merged `b2377d7`; APPROVE (exact) — 0 rounds | C-015 | `agent/p25-lineage-carrier` | `pipeline.py` :: `_carry_rag_provenance`, `_RAG_ROW_CARRIER_KEYS` | C-015 impl | A + provenance | — |

## Wave C

| ID | Task | Status | Deps | Branch | Ownership boundary | Reviewer | Focused | Bench |
|---|---|---|---|---|---|---|---|---|
| C-040 | p05a compound-resolution extract | `ACCEPTED` merged `334ad88`; APPROVE (exact) — 0 rounds; qb discharged at pack checkpoint 23/23 | SPIKE-002 | `agent/p05a-resolution-extract` | NEW `pwml/compound_resolution.py`; `ir.py` :: 2 fns | C-051 impl | D | — |
| C-041 | p08 release status + coverage split | `ACCEPTED` merged `b5bbf08`; APPROVE (exact) — 0 rounds; golden driver diff empty | C-010, C-012 | `agent/p08-release-status` | NEW `pipeline/release_status.py`; `strict_quarantine.py` :: `evaluate_core_coverage`; `_finalize_gate_failure`; `batch/report.py`; `bench/render.py` | C-010 impl | A, B | — |
| C-041a | p08a D-002 release/refusal seam | **`ACCEPTED`, merged `eeb576f`, reviewed tip `4177fe5`** — status corrected 2026-08-16; the cell below is retained as historical in-flight detail and its “no implementation commit” clause is **SUPERSEDED**. `4177fe5` (“a coverage shortfall stops being a refusal at the release/refusal seam”) is an ancestor of the integration tip and `defensible_core` is live at `strict_quarantine.py:2022`. **TRAP-1 is therefore LIVE, not hypothetical**: D-002 is applied, a coverage shortfall no longer refuses, and a `review_required` leg can export and enter strict success — which is exactly what C-053/C-056b must gate. Historical detail follows. Branch was at base `b268121` with no implementation commit **at the time of writing**; 302 insertions / 40 deletions live in the worktree. **Ceiling 1500 → 1900 by FINAL product-owner pre-commit re-charter** — valid precisely because no commit exists; a ceiling is **never** raised after one does. The orchestrator's own single narrow re-charter for this card (1100 → 1500) was already spent at Pack 2. Scope, production ownership, acceptance clauses and generated-evidence budget **unchanged**. Remaining: isolated re-run + base-export comparison for `qb` node12/node18/node22 to close D-023 criteria 2 and 3 (**isolated single nodes, not a cohort**), the `executed=22/23 omissions=1` question from run 2, final `check --task C-041a`, then commit. **Self-declared process deviation stands and is for its reviewer to adjudicate, not to wave through:** the 10-min foreground cap killed the first `qb` attempt, and the second used the harness's **tracked** background job with the same `bounded_run.py` wrapper, Job Object, outer timeout and survivor verification — disclosed before review. **F-1 carries into closeout unsoftened:** PMC12452463 classifies `release_ready` and strict-acceptance-eligible with a *chemical* defect and `semantic_evaluation="not_evaluated"` by design until C-056a; **unchanged from base**, so this card neither creates nor worsens it, and **A7's local pass must not be read as TRAP-1 satisfied** — denominator enforcement is C-053/C-056b | C-041, F-006 / D-002 | `agent/p08a-d002-release-seam` | release/refusal seam around `strict_quarantine.py` :: `quarantine_and_close`; **plus exactly its four re-pinned `report["ok"]` assertions and `_payload_with_no_viable_core` in `tests/test_streamlit_quarantine_boundary.py`**, moved under merge rule 4 because D-002 (LOCKED) changes the behaviour they pin. **Guard: where a payload has no genuine surviving core the assertion stays `ok is False`.** **Must not run concurrently with C-057** | independent | A, D (`qb`) | — |
| C-042 | p03c extraction escalation ladder | **`ACCEPTED`, merged `8917349`**, reviewed tip `5b86153`, 1 round, budget **1300/1300 (exhausted with a commit existing)**. Its reviewer's LOW findings 6 and 7 were left undone because a further round would have needed a ceiling raise after commit, which is forbidden. **Finding 6 is carried forward as C-042a — NOT as an unresolved failure of this accepted review.** **CLOSED, not reopened, not amended** | C-032, C-038 | `agent/p03c-extraction-ladder` | `pipeline.py` :: `_run_json_stage`, `_build_extraction_prompt`; `extraction_diagnostics.py` | C-032 impl | A | M2 |
| C-042a | p03d attempt-cap termination reason | **`ACCEPTED`, merged `758b312`**, reviewed tip `9b920c72`, 2 commits, `REV-042` bare **APPROVE**, budget **793/1000**. Owned-diff equivalence byte-identical (`8d0fc3e7`). The second commit is an **orchestrator-authorized boundary extension**: the writer found the new reason reached `RungDecision` and `preservation_record` but **not the leg-level report**, because `pipeline.py`'s termination block had only two branches; that block is inside `_run_json_stage`, **C-042's own owned function**, so the extension was a charter correction and the omission was the orchestrator's (**O-6s**). Dispatched from `472293c` under product-owner **RULING 3**, ceiling 1000. Establishes a **seventh** termination reason `attempt_cap_reached` under new decision **D-024**, used only when the configured attempt count alone ended processing and **no** stronger existing reason truthfully describes the outcome. Precedence: success → refusal → deadline/timeout → separately measured resource/token budget → `attempt_cap_reached`. Today `extraction_ladder.py:478-483` refuses on a spent cap with `termination_reason` **`""`**, while its neighbours at `:500` and `:539` pass `IDENTICAL_EMPTY_RESPONSE` and `BUDGET_EXHAUSTED`. **`OPERATIONAL_TERMINATION_REASONS` stays `{budget_exhausted, operation_timeout}`** — widening the strict-success denominator is a product decision not made. **Hazard named in the charter:** the literal `"attempt_cap_reached"` already exists as `SKIP_ATTEMPT_CAP` (`extraction_ladder.py:125`); the shared string is intentional but `SKIP_CAUSES` and `TERMINATION_REASONS` must stay independently closed vocabularies, and the module's "A SKIP CAUSE IS NOT A TERMINATION REASON" must stay true. `tests/test_deadline_budget.py:199` pins `len(...) == 6` — a deliberate 6 → 7 baseline move under merge rule 4 | C-042 ✔ `8917349` | `agent/p03d-attempt-cap-reason` | `pipeline/deadline.py` :: `TERMINATION_REASONS`, `require_reason`; `rag/loop_policy.py` :: `TERMINATION_PRECEDENCE`; `pipeline/extraction_ladder.py` :: the attempt-cap refusal + `termination_reason` serialization; NEW `DECISIONS.md` **D-024** | independent | C | — |
| C-043 | p32 RAG loop controller | `ACCEPTED` merged `3c04d4b`; APPROVE (exact) after 1 correction round | C-016, C-021 | `agent/p32-rag-controller` | NEW `rag/controller.py` | C-055 impl | C | — |
| C-044 | p26 lineage: mapping | **RECONCILED 2026-08-18 (Git ancestry):** **`ACCEPTED`, MERGED `8f1a692`.** The `DISPATCHED` label below is historical. **`DISPATCHED` 2026-08-13** from `bcc0bfe`; deps satisfied (C-015 `8b4bc0c`, C-033 `7ff1211`) — authorized by the product owner | C-015, C-033 | `agent/p26-lineage-mapping` | `src/t2pw/mapping/map_ids.py` (lineage writes; **not** `src/map_ids.py`) | C-033 impl | C | — |

## Wave D

| ID | Task | Status | Deps | Branch | Ownership boundary | Reviewer | Focused | Bench |
|---|---|---|---|---|---|---|---|---|
| C-050 | p05b pre-freeze resolution call | **RECONCILIATION 2026-08-18 (orchestrator, from Git ancestry):** **`ACCEPTED`, MERGED.** The cell below is retained as historical in-flight detail; its "`COMPLETE, HELD` ... NOT merged" claim is **SUPERSEDED**. C-050 landed inside the twelve-card composite **`beddcdd`** under D-030, after its own integration sync **`0f859d9`**; `768be7507ce5ad08f2a4efe72537347b1a363dec` is an **ancestor of the integration tip** and so is its branch tip `agent/p05b-prefreeze-call` @ **`edf8a0d`**. Verified by `git merge-base --is-ancestor`. **`COMPLETE, HELD` — tip `768be7507ce5ad08f2a4efe72537347b1a363dec`, NOT merged.** Correct on its own terms: five-category exporter-mutation measurement **0/0/0/0/0 `RESULT: MEASURED`** at tip vs **1/1/1/8/16 exit 1** at base; SMOKE 460; Chunk D core+s8 4/4; 59 G11 artifacts / 0 non-compliant. Budget ~1540/1600. Blocked by **B-P2-1** (`qb` node15), **now resolved by product-owner RULING 1 of 2026-08-13**: released to merge **after C-050a merges**, with one complete 23-node `qb` run on the prospective combined state and all 23 nodes classified. **C-050 is not to be faulted merely because its original base lacked C-050a.** `qb` **node06** remains unresolved (passes at base, 1 pass / 2 fails at tip, `KeyError: quarantine_report` with **no C-050 symbol in the traceback**) — re-check after node15 settles | C-040, C-030 | `agent/p05b-prefreeze-call` | `streamlit_app.py` :: enrichment block above the seam | C-052 impl | D | M3 |
| C-050a | p05c node15 quarantine-boundary comparand | **`ACCEPTED`, merged `be7f4c2`**, reviewed tip `d88dcb5e`, `REV-050` bare **APPROVE**, hand-authored **640/800**. Owned-diff equivalence over `tests/` byte-identical (`c1001935`). **GENERATED-EVIDENCE DEVIATION, RECORDED HONESTLY PER RULING 5 — this card is NOT fully budget-compliant.** Actual **41** artifacts against a ceiling of **40**. **Cause:** the charter mandated the split-process Chunk D gate, which **auto-allocated 32 of the 41 reports**, making the ceiling internally inconsistent with the gate the same charter required — a **charter-sizing defect, not a writer defect**. **Impact on code / PWML / biology: none identified**; total size 131 KB against a 2 MB cap, and every artifact is genuine (the writer correctly **refused to delete superseded or failing reports** to reach 40, and committed the **failing** determinism run `05` alongside the passing `04`). **Corrective action: D-025** — future split-process Chunk D cards must budget all auto-generated reports plus review/failure headroom. Accepted code is **not** reopened and the merge is **not** rewritten. Dispatched from `472293c` under product-owner **RULING 1**. **TEST-ONLY — no production code authorized.** Ceiling 800 | — | `agent/p05c-node15-comparand` | `tests/test_streamlit_quarantine_boundary.py` :: `test_research_mode_keeps_the_unmapped_candidate_and_does_not_block` + minimum supporting fixture. Re-points `mapped_in` from the **pre-enrichment** `final_mapped_db` to the **post-enrichment** artifact quarantine is actually handed. The comparand already exists — `streamlit_app.py:3725` publishes `"final_mapped_enriched": final_export_payload`, and `:3605` hands exactly that object to `freeze_canonical_payload` — **so no production change is needed**. Must stay whole-object equality; a partial-field weakening is a reject. Must be made **non-vacuous** by a deterministic, semantically harmless pre/post-enrichment difference. **Merges BEFORE C-050** | independent | D (`qb`) | — |
| C-051 | p05c IR assert-only | **`ACCEPTED`, MERGED** in the twelve-card composite **`beddcdd`** (D-030), reviewed tip **`2b3de80a`** — `REV-051` bare **APPROVE**, 1 correction round. **RECONCILED 2026-08-18:** the previous `BLOCKED` was stale; both dependencies (C-040 `334ad88`, C-050 `beddcdd`) are merged and `2b3de80a` is an ancestor of the integration tip. Detail row below. | C-040, C-050 | `agent/p05c-ir-assert-only` | `ir.py` :: `build_pwml_ir` | C-040 impl | D | — |
| C-052 | p06b freeze enforcement | **RECONCILIATION 2026-08-18 (orchestrator, from Git ancestry):** **`ACCEPTED`, MERGED `c0df0d0`, reviewed tip `c7250d3`** (D-040). The `DISPATCHED` label below is historical. **`DISPATCHED`** 2026-08-17 — base **`a662c3f5ce994a1436fe62429c74a7db1144df14`**, worktree `.claude/worktrees/agent-c052-prefreeze-report`, charter `C-052-charter-v2.md`, ceilings 900 / 90 / 20,000. **This is the authoritative C-052 row; the second row further down is reconciled to it.** Boundary per **D-040** | C-030, C-050, C-020 · **C-050i ✔ merged `509faee`** | `agent/c052-streamlit-prefreeze-report` *(actual; the planned `agent/p06b-freeze-enforce` was never cut)* | `streamlit_app.py` :: `freeze_canonical_payload` (**zero lines**), `run_pwml_export`, SBML bind (`:3638`/`:3650`), **+ key-addition only on the `run_post_pipeline_sbml_artifacts` success return `:3680-3807` (D-040)** | C-030 impl | **`qb` + full Chunk D + SMOKE + named focused set** (5 of 7 changed files are in no chunk) | M3 |
| C-053 | p09 PWML artifact naming | **`RULINGS COMMITTED`** 2026-08-17 — dependency C-041 **merged and re-verified live** (`strict_quarantine.py:846` declares `-> CoverageVerdict`; `release_status.py:234` defines `coverage_verdict`, exported `:379`, consumed `strict_quarantine.py:2021`). Boundary re-measured and widened per **D-038**; charter must be rebuilt from D-038 before dispatch | C-041 ✔ | `agent/p09-pwml-naming` | see `MASTER_PLAN` §9 (re-measured 2026-08-17) | C-031 impl | B + named focused set | — |
| C-054 | p16 gold `expected_export` required | **`DEPENDENCY-READY`, NOT STARTED.** **RECONCILED 2026-08-18:** the previous `BLOCKED` was stale — **both** dependencies are merged (C-041 `b5bbf08`, C-053 `3fde1f1`). No branch, no worktree, no commit. **F-053 caution:** C-054 must not be chartered to consume `semantic_evaluation == "passed"` affirmatively until C-056c discharges F-053. | C-041, **C-053** | `agent/p16-goldset-required` | `bench/goldset.py` | C-056b impl | B | — |
| C-055 | p33 RAG controller wiring | **`ACCEPTED`, MERGED `365c99d`, reviewed tip `84a6e4f`** (2026-08-20, PACK 9) — bare `APPROVE` from REV-055 at `5ad0d47`, two correction rounds, both comment-only. **`run_rag_loop` had zero production callers; it has one now, and every round re-enters all five stages.** Separation invariant holds — `ast.While` count in the app is **0 at base and 0 at tip**. REV-055 AST-hashed every top-level def (54 -> 67, **ADDED 13, REMOVED 0, CHANGED 0**), confirmed `maybe_run_rag` **byte-identical**, re-ran SMOKE itself at 465, and **mutation-tested the guards** (restore disabled -> restoration test red with the boundary's REAL report in session; mapping stubbed -> 3 red including the real-app AppTest). **Trap 11 answered better than chartered:** rounds map against a round-scoped copy, so the fragile unlocked 4.2 MB cache overwrite is multiplied by **ZERO**. **Central obligation discharged** — two `streamlit.testing.v1.AppTest` tests on the RAG path, the first ever. **Two forced deviations ratified**, one contradicting my own charter ruling (classification via `run_quarantine_boundary`, because `test_streamlit_quarantine_boundary.py:761-779` is hotspot 9 and a direct call adds **two** callers — the card complied with the guard and removed both harms the ruling named). **PACK 9 RULING 4:** it tightens a previously-unconditional gate (`validate_graph_delta` before `final_payload` advances) — PRODUCT_CONTRACT §10 enforced for the first time. Budget ratified 1600 -> 1900, measured **1791**. Post-merge gates green (`MERGE-055/01-smoke.json`, 465). **The product owner's uncommitted `streamlit_app.py` edit was preserved through the merge and proved byte-identical.** **T-103 IS UNBLOCKED.** Historical detail follows. **`DEPENDENCY-READY`, NOT STARTED.** — worktree fast-forwarded `1c06918` -> `32f3a57`, one writer, ceilings 1600 / 140 / 6 MB. Historical detail follows. **`DEPENDENCY-READY`, NOT STARTED.** **RECONCILED 2026-08-18:** the previous `BLOCKED` was stale — **all three** dependencies are merged (C-043 `3c04d4b`, C-041 `b5bbf08`, C-032 `1801298`). Branch `agent/p33-rag-wiring` and worktree `.claude/worktrees/agent-c055-rag-wiring` **created 2026-08-18 at `1c06918`** (clean, zero commits of their own, `.env` copied for F-051 control parity). **HIGHEST MILESTONE VALUE OF THE REMAINING QUEUE: it is the sole prerequisite of T-103 (M4), which cannot run until it merges.** | C-043, C-041, C-032 | `agent/p33-rag-wiring` | `streamlit_app.py` :: `maybe_run_rag` + script body | senior | C + AppTest | M4 |
| C-056a | p42a semantic → runtime release_status | **RECONCILIATION 2026-08-18 (orchestrator, from Git ancestry):** **`ACCEPTED`, MERGED `93594aa`, reviewed tip `9c9f94a`** (D-037/D-039/D-042). The `DISPATCHED` label below is historical. **`DISPATCHED`** 2026-08-17 — base **`a662c3f5ce994a1436fe62429c74a7db1144df14`**, worktree `.claude/worktrees/agent-c056a-semantic-gating`, charter `C-056a-charter-v2.md` (rebuilt from D-037+D-039), ceilings 1,100 / 95 / 22,000. Boundary granted by **D-037**, eight rulings issued as **D-039**. **Buildable without moving a single exact-set pin.** The retired scratchpad charter's central blocking finding (F-2) is **measurably false** and its remedy is superseded | C-017 ✔ · C-041 ✔ · D-037 ✔ · D-039 ✔ | `agent/c056a-semantic-release-gating` *(actual; the planned `agent/p42a-semantic-runtime` was never cut)* | see `MASTER_PLAN` §9 (re-measured 2026-08-17) | **independent — not C-017/C-041a impl** | B + `D (qb)` + named focused set | — |
| C-056b | p42b semantic → benchmark denominators | **`ACCEPTED`, MERGED `69928eb`, reviewed tip `8eee549`** (D-039/D-042); D-052 ratifies its two disclosed boundary widenings. **RECONCILED 2026-08-18:** the previous `BLOCKED` was stale — both dependencies merged (C-056a `93594aa`, C-053 `3fde1f1`). It was the last code-card prerequisite of T-100. | C-056a, C-053 | `agent/p42b-semantic-bench` | `acceptance.py` :: `_build_denominators` | C-056a impl | B | — |
| C-057 | p27 lineage: quarantine | **`CHARTERED` 2026-08-20 (PACK 9), not dispatched — two-writer cap.** Charter written against post-C-056c source per D-054 §9. Boundary measured for the first time: the **five module-level functions** `_admit_processes` `:1095`, `_prune_entities` `:1294`, `_prune_locations` `:1349`, `_prune_biological_states` `:1419`, `_drop_quarantined_processes` `:1519` — all *before* `quarantine_and_close` (`:1793-2338`), so **same-file, different-function** with C-056c. **`strict_quarantine.py` contains ZERO lineage writes and no writer of `stage="quarantine"` exists in `src/`** though the value has been in the closed vocabulary since C-015 (`lineage.py:54`). Ceilings 1000 / 110 / 5 MB. Historical detail follows. **`DEPENDENCY-READY`, NOT STARTED.** **RECONCILED 2026-08-18:** the previous `BLOCKED` was stale — **all three** dependencies are merged (C-015 `8b4bc0c`, C-010 `72ee20f`, C-041 `b5bbf08`). No branch, no worktree, no commit. | C-015, C-010, C-041 | `agent/p27-lineage-quarantine` | `strict_quarantine.py` (lineage writes) | C-041 impl | A, E | — |
| C-050j | F-046 component-bucket dedupe residual | **`ACCEPTED`, MERGED `cbeaa84`, reviewed tip `8c062ee`** (2026-08-20, PACK 9) — exact bare `APPROVE` from REV-050j after **zero correction rounds**. **The chartered census ran first and returned ZERO** — 35 legs x both production `strict_db` arms = 70 measurements, 0 manufactured merges, 0 pre-existing component collisions, 0 post-freeze renames — so the card stayed on the **G9 new-capability arm** as D-050 section 3 prescribes, with no fabricated base failure. The probe **wraps** the real `_apply_create_defaults` / `_dedupe_named_rows` rather than reimplementing them, and a synthetic control carrying both `ir.py:230-238` Narcissus spellings makes the same instrument report `BUILD_RAISED`, so the zero is a measurement and not a dead sensor. **REV-050j reproduced that zero two independent ways** (static over all 35 legs; and the real offline species ladder over 35 x 2 with the guard's map recomputed), **proved the boundary by AST comparison** (entity call site and `DuplicateNamedRowError.__init__` AST-identical base vs tip; exactly two top-level defs changed code — the two D-050 section 4 names), and **proved non-vacuity with four mutants** (neuter -> 4 red; force over-fire -> 11 red including all four preservation tests; identifier-equality rebuild -> F-043 arm red; identifier rule atop the real map -> second arm red). **F-043 holds**: the discriminator reads only `name` on both sides of the rename — no `taxonomy_id`, no `pathbank_species_id`, no accession. **Both collapses the pipeline depends on are preserved by construction** — D-016/C-045's convergence arrives under one pre-rename key; hydration placeholders are outside the map's domain, an argument the reviewer confirmed genuinely order-independent. **Merge rules**: 6 — the warning branch is *narrowed*, the only new exit is more refusal; 7 — no pathway dropped, the residual F-046 warning path untouched; 8 — it refuses and never consolidates. D-035/D-036 unamended. **Budgets 836/1100 · 44/120 · 0.17 MB/5 MB.** **Three disclosed deviations, all ratified**: the `PostFreezeComponentMergeError` subclass (in boundary; collapsing it would destroy the one distinction D-050 section 2 says the record lacked — *who authored the collision*); two comments amended because the diff made them literally false, one at the out-of-boundary **entity** call site (AST-proved zero behaviour delta — **registered for C-050k's owner**); and `git stash push -- src tests` as the F-051 control (a *tighter* control than a base worktree; a silent no-op stash is refuted by base 325 vs tip 339 passed and base collect 22 vs tip 36). **Carries three record corrections**: D-050 section 1's `followed_leader` nomination **does not transfer** to the create-defaults path (amended in `DECISIONS.md`); the corpus is **35, not 32** (F-068); and two corpus-pinned tests fail **unconditionally**, not `.env`-conditionally (**F-069**). Historical detail follows. **`DEPENDENCY-READY`, NOT STARTED.** — branch `agent/c050j-component-dedupe` and worktree created at `32f3a57`, `.env` copied for F-051 parity, one writer, ceilings 1100 / 120 / 5 MB. **Its first act is D-050's exposure census; a non-zero result is a chartered STOP.** Historical detail follows. **`DEPENDENCY-READY`, NOT STARTED.** **REGISTERED 2026-08-18** — the card had no ledger row until this reconciliation. No branch, no worktree, no commit; nothing to recover. Its blocking defect was corrected by **D-050**: F-046's proposed discriminator would fire **never** (`prefreeze_resolution.py:1215-1245` stamps *every* named species row), and the correct marker is **`marker["followed_leader"]`** (`:1242`). **First act is to measure the exposure of the `_apply_create_defaults` rename path** (`ir.py:344`, applied `:1085-1096`) immediately before the unguarded dedupe at `:1097` — C-050i's zero-of-32 census structurally could not see it. If exposure is non-zero, **G9 flips from new capability to a correction with a real base-failing proof and the card stops for a fresh ruling**. **F-043 binds: never build the discriminator from identifier equality.** **Re-measure every `ir.py` line number** — five cards merged after the draft charter | **D-050** ✔ | *to be created* | `ir.py` :: `_dedupe_named_rows` component call site | non-author of C-050i / C-050k | D | — |
| C-050k | F-048 ambiguous alias bindings | **`ACCEPTED`, MERGED `d8de94d`, reviewed tip `11844c5`** (D-041/D-043), 1 correction round. **REGISTERED 2026-08-18** — the card had no ledger row until this reconciliation. `resolve_entity`'s silent first-wins on an ambiguous alias index now records the binding. **D-043's bio-adjudication of all 20 bindings is authoritative and must not be re-derived** | F-048 | `agent/c050k-alias-ambiguity` | `resolve_entity` alias index | non-author | D | — |
| C-056c | F-053 semantic evaluability carrier | **`DEPENDENCY-READY`, NOT STARTED.** **REGISTERED 2026-08-18** — the card had no ledger row until this reconciliation. Branch `agent/c056c-semantic-evaluability` and worktree `.claude/worktrees/agent-c056c-evaluability` **created 2026-08-18 at `1c06918`** (clean, zero commits of their own, `.env` copied for F-051 control parity). Both upstream cards are merged (C-056a `93594aa`, C-056b `69928eb`). **It owns F-053: evaluability does not travel beside the verdict.** **BINDING until it is discharged — no card may be chartered to read `semantic_evaluation == "passed"` affirmatively, or build any rate on it.** C-056b's subtractive-only design holds that line by discipline; C-056c is the guarantee. **Do not broaden it into unrelated semantic redesign** | C-056a ✔, C-056b ✔ | *to be created* | the semantic evaluability carrier beside the verdict | non-author of C-056a / C-056b | B | — |

### Wave D — the C-050/C-045/C-051 atomic stack (D-030), opened 2026-08-14

**Subcard ID allocation, recorded before dispatch.** `C-050a` and `C-050b` are merged; `C-050c` was
used as a C-050 correction-round evidence task ID (its G11 reports live on the branch). The next
unused IDs are therefore **`C-050d`** and **`C-050e`**, allocated as below. No ID is reused.

**Build order — strictly serial. C-045 and C-051 must never run concurrently.**
`corrected C-050 (C-050e) → C-050d → C-050f → C-045 → C-045a → C-045b → C-051 → C-051a → C-051c → C-051b → C-051d → ONE composite --no-ff merge.`

**C-051c inserted 2026-08-15, BEFORE C-051b.** Chunk D core is a **BLOCKING** gate standing at **150/152**, and the two failures (`REV-051` **F-5** `raw_name` overwritten by the fixed-point loop, and **F-3** the orphaned non-blocking test whose premise is structurally gone) were no prior card's to fix. It is sequenced before C-051b because a surviving `raw_name` may move the 32 `GOLDEN` digests, and **the golden must move only once more**.

**C-045b inserted 2026-08-15 under D-032 (LOCKED), and it is a HARD PREREQUISITE FOR C-051.**
`REV-045a` measured, through the documented README command, that the tip **loses species
canonicalization entirely on the CLI export path**: `pathway.pwml` emits the strain string where
base emitted the canonical binomial, with **no `raw_name`, no `aliases`, empty rename log and
`preflight: null`** — the organism changes identity and the provenance carrier is gone with it.
`run_prefreeze_resolution` has exactly **one** production caller (`streamlit_app.py:3587`) while
`writer.py :: run_pwml_pipeline_export` reaches `build_pwml_ir` with none, and the repo's own
`pathwhiz_requirements.md` names **"the two production entry points"**. **No single card erred** —
the defect is emergent across four, each link defensible and documented, and the orchestrator's
C-045a charter compounded it by asserting the standalone configuration was "not the production
path". **⚠ The same defect is queued to recur for COMPOUNDS:** `_resolve_compound_rows` is still
called at `ir.py:979`, so the CLI still gets compound resolution from the exporter — and **C-051's
charter is to remove exactly that call site.** C-045b wires the seam once for the whole
`PREFREEZE_CANONICALIZERS` tuple, covering both stages permanently. **C-051 must not be dispatched
until C-045b is approved.**

**C-045a inserted 2026-08-15.** C-045's D-016 move turns C-040's pre-extraction golden red, and that golden is
**C-040's acceptance artifact**, not C-045's to regenerate. The repair is routed as its own owned subcard —
the same handling node15 received as **C-050d** — so C-045 supplies the exact delta under **merge rule 4**
and C-045a performs the baseline move under an independent review that looks at nothing else.

**C-050f was inserted after REV-050e**, which measured **P5-01** (F-2): `_propagate` rewrites by
`_canonical` while `_assert_fully_propagated` audits by `_norm`, so the detection set is strictly
wider than the rewrite set and a case- or punctuation-variant reference is never rewritten and then
always raises `PREFREEZE_RENAME_NOT_PROPAGATED` — a hard abort on a reference that is **not**
dangling. It is **not** C-050e-caused, and it is pre-existing only relative to the branch:
`prefreeze_resolution.py` does not exist in integration at all, so the stack would **introduce** it.
**It lands before C-045** because C-045 renames species through the same `_propagate`, and its
deterministic strain normalization produces exactly the substantive-rename class that triggers the
abort, on names far more prone to case/punctuation variance than compounds.
No intermediate card is merged into integration (**D-030** clause 4). Each card is reviewed against
its **declared direct parent** by an independent non-author reviewer at the **exact tip**.

| ID | Task | Status | Deps | Branch / base | Ownership boundary | Reviewer | Ceilings (D-025) |
|---|---|---|---|---|---|---|---|
| C-050e | C-050 correction round 3 — B-4 discriminating provenance proof; DEF-3 cross-kind rename safety | **`ACCEPTED` (in-stack, NOT merged to integration)** — tip **`edf8a0d1`**, base `0f859d9f`, `REV-050e` bare **APPROVE**, 0 correction rounds. **TEST-ONLY: `git diff 0f859d9f edf8a0d1 -- src/` is EMPTY**; `test_streamlit_quarantine_boundary.py` zero lines; 9 files, 759+/23−; budgets **309/320**, **7/25 artifacts**, **14.3 KB / 1.5 MB**. **B-4 discharged behaviourally** — the comparand is the historical blob `768be75:prefreeze_resolution.py`, proven by `git log --reverse` to be the **only and last** tree without `_snapshot_provenance`/`_authoritative_provenance`/`_restore_provenance` (symbol count 0 there, 7 at `e7f28e7`), loaded under a private module name so it imports the **tip's** `compound_resolution` with C-040a included: `unmatched` → `matched_offline_name_index`. REV-050e strengthened the attribution by disabling **only** `_restore_provenance` on the tip and reproducing `unmatched`, isolating the cause exactly rather than relying on the payload's missing `element_locations`. **DEF-3 closed by proof, no production code** — `_reject_ambiguous_renames` (`:646-690`) raises `AMBIGUOUS_REFERENCE` through `_alias_index`'s **primary** index, and `_connectivity_signature` (`:290-312`, compared `:423`) raises `PREFREEZE_CONNECTIVITY_BROKEN` for the synonym-only case. REV-050e verified **the hazard is real underneath** (with both mechanisms disabled the protein location row really is redirected onto the compound) and that the new test **discriminates against each mechanism separately**. The **arm-3 residual is genuinely identity-preserving**: canonical `entity_locations` and the broken-reference list are identical before and after, checked by execution across 7 further payload shapes the card never constructed. **REV-050e built the counterfactual the card omitted** — a kind filter applied consistently in `_iter_refs` runs clean but creates **three new dangling entity keys and three new broken references**, a `PRODUCT_CONTRACT` §5 and D-015 clause 6 violation — so the no-code completion is **correct**, not merely convenient. **A9 category 5 left RED and untouched** (`identity materialization 1`, `matched_offline_name_index → legacy_id_unverified` on `Glycine`, `C-050 ACCEPTANCE: FAILED`): a **PASS condition** under D-030, discharged later by C-051. Findings: **P5-01 (F-2, HIGH) routed to C-050f**; P5-02 (LOW, docstring precision) and P5-03 (INFORMATIONAL, uncommitted helper scripts) folded, neither blocking | C-040a ✔, C-060a ✔ | `agent/p05b-prefreeze-call` (the existing C-050 branch — **no new branch**), wt `agent-p05b-prefreeze` | `prefreeze_resolution.py` :: the entity-reference traversal (`_LOCATION_MEMBER_FIELDS`, `_iter_refs`, and `_propagate` only if measurement proves it necessary) + `tests/test_prefreeze_compound_resolution.py`. **`tests/test_streamlit_quarantine_boundary.py` is zero lines — it is C-050a's.** **B-1's fifth A9 delta is deliberately left failing** (D-030) | independent, non-author | 320 hand-authored · 25 artifacts · 1.5 MB |
| C-050d | node15 residual `:1111` repair — C-050a's `_without_annotation` harmlessness proof vs C-050's new pre-freeze stage | **`ACCEPTED` (in-stack, NOT merged)** — tip **`a81b1d65`**, base `edf8a0d1`, `REV-050d` bare **APPROVE**, 0 correction rounds. **TEST-ONLY**: `src/` delta empty, one test file **+39/−8**, 39 evidence artifacts, nothing outside `tests/` + own evidence dir. Budgets **47/220**, **39/40**, **115 KB / 2 MB**. **Reproduction confirmed the orchestrator's `:1111`→`:1101` mapping exactly**; the artifact diff is **37 paths, all ADDED** (2 enrichment + 35 pre-freeze, 5 fields × 7 rows), zero changed, zero removed. **Remedy B (semantic harmlessness)**, and remedy A was killed by measurement not argument: `final_mapped_db is final_mapped_enriched` is **False** — `streamlit_app.py:3747` re-parses from disk while `:3748` binds the mutated payload — so the charter's alias hazard is structurally impossible and a "deep pre-enrichment snapshot" *is* `final_mapped_db`. REV-050d additionally found `final_mapped_quarantined is final_mapped_enriched` **False** while `==` **True**, so the boundary assertion is a genuine structural equality, not an identity tautology. **A2 verified byte-identical**: base `:1103-1120` and tip `:1134-1151` share sha256 `517a5f25…`. **23-case adversarial harness**: `row.items() <= row_after.items()` catches modified **dict**-valued keys and removed keys without `TypeError` (`dictitems_contains` compares with `PyObject_RichCompare`, never hashes); rename/reorder/add/remove-row fire at `:1116`; reaction and location mutations at `:1109`. **Non-vacuity adjudicated**: enrichment reverting to a no-op **fires at `:1129`** — and `:1090` alone would **not** have, because the 35 compound paths keep the inequality true, so C-050d's new block is a **real strengthening** that stopped C-050's own stage from silently rescuing a vacuous A3. **Chunk D one cohort, `T2PW_OFFLINE_CURATOR=1`**: `jobs=28 executed=177/177 omissions=0 additions=0 failed=none`, partition core 150 + s8 4 + qb 23, `SETS_EQUAL=True`, overlap 0; **all 23 `qb` ok, node15 ok, node10 ok under BOTH readings** — REV-050d confirmed from the reports' own `command` arrays that the gate numbers AppTest nodes with a shared counter over `sorted(...)` (`chunk_d_gate.py:259-263`), so s8 takes 00-03 and qb 04-26, making gate `node10` = `test_all_four_artifacts_are_written_and_retained` while definition-index 10 is gate `node19`; both green. All 39 reports `final_surviving_count = 0` / `cleanup_success = true`. **D-025 honesty confirmed structurally**: `allocate()` uses `max(used)+1` and never reuses sequences, so a deletion would leave a permanent gap — the directory is **contiguous 01…39**, proving nothing genuine was deleted to fit under 40; the superseded `05-a42-mutation.json` and the failing `01-repro-n15-base.json` were both retained. Findings: **P5-05 (MEDIUM)**, **P5-06 (LOW-MED)**, **P5-07 (LOW)** — none blocking | C-050e | *(branch created at dispatch)* | **C-050a's ownership.** `tests/test_streamlit_quarantine_boundary.py` :: node15 only. **TEST-ONLY** unless measurement proves a production defect, in which case the card **stops and reports** | independent, non-author | set at dispatch |
| C-050f | align the propagation match rule with the propagation audit rule (**P5-01**) | **`ACCEPTED` (in-stack, NOT merged to integration) 2026-08-15** — final tip **`0ec64d2c65f02033cdfece8bbdd8587992de98b4`**, parent `5c30e958`, `REV-050f` **round 2 bare `APPROVE`** at the exact tip, **1 correction round**. Two-round totals: `src/` delta **`prefreeze_resolution.py` only, exactly 2 hunks**; test file **162 additions / 0 deletions**; `tests/test_streamlit_quarantine_boundary.py` **zero lines**; 43 G11 artifacts, all `final_surviving_count = 0` / `cleanup_success = true`, `check --task C-050f` 0 non-compliant. Budgets: round 1 **423/430**, round 2 **180/180**, artifacts **44/55**, size **135.4 KiB / 2.5 MB**. **The B-1 fix: one `_match_key(value) = _norm(value) or "\x00" + _canonical(value)`**, keyed by **both** `_rename_targets`/`_propagate` and `_assert_fully_propagated`, so **the rewrite set and the detection set are one set by construction** — the card's thesis, broken by the original defect in one direction and by round 1 in the other. Namespace disjointness is provable (`_norm` output is confined to `[a-z0-9:+ ]` and can never begin with `\x00`) and **REV-050f brute-forced 75,894 strings over a 44-character alphabet including Greek, superscripts and punctuation: 0 namespace violations**. The skip guard moved from `if not _norm(old)` to `if not _canonical(old)`, so only genuinely nameless sources are dropped. **Round-1 reproducer restored to base exactly**: `---`/`α`/`-`/`??` @ `CHEBI:15428` give `PREFREEZE_CONNECTIVITY_BROKEN` with the payload untouched at base **and** at the final tip, and `_propagate` returns a real rewrite instead of `[]`. **The B-1 guard fails behaviourally on the round-1 shape in all four parametrizations** — not by symbol absence. **⚠ ONE DELIBERATE NARROWING REMAINS, ADJUDICATED AND ACCEPTED (`REV-050f` Q7).** The fix is **not** a strict superset: it removes exactly one class — an empty-`_norm` reference that is **not** the renamed name, in a payload where some *other* empty-`_norm` name was renamed. Base bucketed every empty-`_norm` name under one `""` key, so renaming `---` made an unrelated `α` reference fatal **in a message naming `---`**. REV-050f adjudicated it acceptable **by exhausting the resolvers rather than by argument**: such a reference is unresolvable before *and* after any rename in `_alias_index` (`if key:`), `ir.entity_by_name` (`ir.py:949`/`:953` guard `if alias_norm:`), and `canonical._Graph.resolve` (`canonical.py:254` short-circuits to `unresolved|`); the one registry that *does* bucket them (`process_normalizer._entity_name_norms`) has a `""` bucket that is **structurally un-emptiable by a rename**, because `_preserve_original_names` re-adds the old name as a synonym — measured on the hardest construction, so **no stranding is constructible**. D-015 clause 6 is satisfied: **112 new raises at tip**, and **zero narrowings in the `ref IS the renamed source` class**; what is lost is a fatal base issued with a *false cause*, on a reference the rename never touched, and the condition is still recorded downstream as `unresolved_entity_reference` / `broken`. Differential re-run and **widened to 5,634 audit pairs**: the only narrowings are the 270 named-class cases plus 3 that **cannot reach the audit** (`{"Gly": "gly", "gly": "Glycine"}` is refused by `_propagate` first, and every `_assert_fully_propagated` call site is immediately preceded by `_propagate` on the same map). **No unnamed narrowing survives**, and round 1's 49 silent propagation drops are gone. **The 180/180 landing was audited line-by-line by the reviewer**: all 31 deletions are code being replaced or round-1 prose that round 2 *corrected*; no load-bearing rationale removed, no assertion weakened, net docstring content grew, both guard tests kept per the orchestrator ruling. **New findings, both non-blocking: `REV-050f` F-5** — `test_c050f_the_rewrite_set_and_the_detection_set_are_one_set` is a **forward invariant test, not a regression proof** (it fails on `5c30e958` by `ImportError`, i.e. symbol absence, and *could* not fail behaviourally there because at round 1 the two key sets were equal — both wrong); harmless, since B-1's behavioural proof is the other guard, but its docstring overstates what it would have caught — **fold the wording fix into a later card only if the file is opened for another reason**. **`REV-050f` F-6** — `_propagate` still rewrites in 32 whitespace-normalization cases where the audit is silent (`{"GLY": "gly"}` with ref `"  gly  "`); **identical at round 1 and round 2 so not introduced here**, and unreachable from `resolve_compounds_prefreeze` because stage 2 canonicalizes both sides of every map entry — recorded so a future reader knows the "one set by construction" equality is over **key sets**, not over behaviours. **No regression in what round 1 cleared**: Q1 boundary, DEF-3 unedited and passing, `_iter_refs`/`_LOCATION_MEMBER_FIELDS`/`_canonical`/`_norm` untouched, A1, A5 (rename maps byte-identical across all three legs), idempotence, merge rule 6, and **A8 category 5 = `1` on `Glycine` with `C-050 ACCEPTANCE: FAILED`, identical base-vs-tip** — the D-030 pass condition, left alone. Gates at the final tip: G9 probe 16 arms `mismatches 0`; focused **40 passed**; neighbourhood **41 passed** exit 0 with the `qb` file excluded; Chunk D core **150/150** and s8 **4/4**; the r2 tests on base give **8 failed / 32 passed** with all 26 pre-existing green. **Both reviewer rounds carried over nothing** — after the overnight machine kill the reviewer re-ran its three pre-kill probes to fresh paths and they were byte-identical, and every one of its 35 bounded jobs reported zero survivors. ROUND 1 HISTORY, retained: `REV-050f` round 1 returned **`CORRECTION`** with one **BLOCKING** finding inside the card's own granted symbols. **B-1: an empty-`_norm` rename source is silently neither propagated nor audited.** `_norm` strips everything outside `[a-z0-9:+ ]`, so a name written *entirely* in such characters (`α`, `β`, `Δ`, `---`, `-`, `??`, CJK, placeholder artifacts) normalizes to `""`, and **both** of the card's edits **discard** those keys rather than merely failing to add them — `_rename_targets` via `if not key: continue` and `_assert_fully_propagated` via `if _norm(old)`. `_alias_index` and `_connectivity_signature` skip empty keys too, so nothing downstream catches it. Measured end-to-end through `run_prefreeze_resolution` with real resolution code: entity `---` @ `CHEBI:15428` gives base **`RAISE PREFREEZE_CONNECTIVITY_BROKEN`, payload untouched** and tip **`applied: true`** with the row renamed to `Glycine` while **the reference stays `'---'`**; same for `α`, `-`, `??`; the `_propagate` differential shows **57 base rewrites lost** in this class. Orchestrator confirmed the mechanism independently by inspection of the diff. **Violates `PRODUCT_CONTRACT` § 5** (process-to-entity references are a must-remain-equivalent dimension; the frozen payload now carries one resolving to nothing on reload), **D-015 clause 3** (atomic propagation to *every* participant reference) and **clause 6 / D-029** (`PREFREEZE_RENAME_NOT_PROPAGATED` silenced), and the charter § 3a reject clause. The card's own **A3** and its docstring's "strict superset, never a narrowing" are **false for this class** — measured **132 narrowings over 3,968 pairs**, all from the `if _norm(old) != _norm(new)` → `if _norm(old)` edit, **none** attributable to the added `_canonical` condition, which is provably incapable of suppressing a base raise. **Base was loud and safe; the tip is silent and wrong — worse than the abort the card was chartered to remove.** Correction round dispatched 2026-08-15 with **fresh allowances inheriting nothing** (180 hand-authored this round; totals 55 artifacts / 2.5 MB), restoring the card's own invariant that **the rewrite set and the detection set must be identical**; indicated direction is a `_canonical` fallback for empty-`_norm` sources applied to **both** sides, and the orchestrator **steered the card away from the fail-closed alternative** because refusing such a map would convert a case that *worked at base* into a new hard abort in a card whose purpose is removing one. **Everything else in the card survived independent measurement and is kept**: Q1, Q3, Q5, Q6, A5, A8, idempotence and merge rule 6 all reproduced by the reviewer. Commit `5c30e958` stays in history — no amend/rebase/squash, the findings reference it. Reviewed tip **`5c30e9589ada7b980cca604f5ba796d76e58c7de`**, direct parent `a81b1d65`, worktree clean, 28 files +3288/−17 (bulk G11 JSON). **Orchestrator-verified before routing review:** `src/` delta is `prefreeze_resolution.py` **only**; `tests/test_streamlit_quarantine_boundary.py` **zero lines**; 24 G11 reports committed; primary checkout still 11 protected entries; integration untouched at `96a64d2`; `agent/p05b-prefreeze-call` and `agent/p05d-node15-harmlessness` unmoved. Final budgets **423/430** hand-authored, **25/45** artifacts, **74.7 KiB / 2 MB**. **⚠ ONE NARROW PRE-COMMIT RE-CHARTER GRANTED under S7: 380 → 430.** The card reported at 423/380 and **stopped before committing without self-authorizing**, correctly. The only route to 380 was deleting `test_c050f_a_genuinely_stale_reference_still_raises` (−24) and `test_c050f_a_widened_match_still_stops_at_a_cross_kind_collision` (−19), landing at exactly 380 with zero headroom — the first being the durable guard against a future card "fixing" a false positive by **gagging the auditor** (the one direction the charter calls an automatic reject), the second covering the **novel** hazard this card creates, since C-050e's DEF-3 proved the arm-3 residual safe under `_canonical` matching while the widened set reaches variant spellings DEF-3 never exercised, making **DEF-3 passing unedited necessary but not sufficient**. The ceiling was sized before the second rewrite site was known — a chartering error, not an overrun; 430 leaves 7 lines, enough to commit and not to grow. **⚠ BOUNDARY QUESTION ROUTED TO REV-050f AS ITS FIRST NAMED QUESTION: the charter undercounted the rewrite sites — there are two.** `_propagate` (`:721`) and a **verbatim inline copy of the same match rule** (`rename_map.get(_canonical(ref.get()))`) in `resolve_compounds_prefreeze`'s stage-4 commit loop (`:435-440`), with an undo-log side effect; measured, widening only `_propagate` **still aborts at `:441` and rolls back**, so A1 is undischargeable without it, and the loop now delegates (6 lines → 1). **Orchestrator ruled it in-boundary pending independent adjudication** — the card owns "the match rule", which had two implementations, and the exclusion list names `resolve_compounds_prefreeze`'s **matching/confidence logic** (stages 1–3), whereas stage 4 is propagation; delegating **removes** a duplicate rather than adding a parallel mechanism, per § 3c. **Merge rule 5 means the reviewer settles this, not the orchestrator.** **Charter § 1 corrected by measurement (O-3s):** an exactly-spelled `glycine` **is** rewritten at base, so the genuine silent miss is the **case-variant** `GLYCINE` — load-bearing, because the real A9 payload's rename map is that pure case change `{"glycine": "Glycine"}`. **⚠ CORRECTION — `PREFREEZE_RENAME_MAP_COLLISION` is REACHABLE. The earlier "unreachable" record in this ledger and in commit `b8ad8b0`'s message was wrong and is retracted here rather than rewritten.** The card measured 67 rename maps reaching propagation with 65 single-key and reported the code unreachable end-to-end; **REV-050f built the counterexample the census missed** (`REV-050f` F-2). The stage ordering is real — `_reject_ambiguous_renames` does run before `_propagate` — but it groups by **`_norm(new)`**, so it does **not** fire when two `_norm`-colliding sources resolve to targets differing only in **case**. Measured through `run_prefreeze_resolution`: rows `[gly→"Glycine", Gly→"glycine"]` give base **`applied=true`, refs `["Glycine","glycine"]`** and tip **`RAISE PREFREEZE_RENAME_MAP_COLLISION`**. It is therefore a **reachable new terminal code**, not merely a guard for the next caller. **The guard is kept** — fail-closed, charter § 3b-authorized, and it cannot increase PWML output — but it must be recorded as reachable, and it *does* still guard **C-045's species map** as the next direct caller of `_propagate`. **`REV-050f` F-3: the census is not checkable from the record** — `24-collision-census.json` is a `bounded_run` cleanup report carrying none of the numeric result, and its producer `collision_census.py` is uncommitted scratch, so the reviewer answered from source ordering plus its own end-to-end counterexample. The correction round must commit a census artifact that actually carries its result or drop the numeric claim; an assertion no one can re-derive from the repo is the **P4-02** class of debt. **`REV-050f` F-4, record only:** on the synonym-only cross-kind arm with a *variant* spelling, base `PREFREEZE_RENAME_NOT_PROPAGATED` → tip `PREFREEZE_CONNECTIVITY_BROKEN`; both fatal, payload unchanged, and REV-050e's exact-spelling case is unchanged — noted so the composite landing does not read it as new. **`04-neighbourhood` and `06-quarantine-tip` exit 1 are `_HARNESS_FAULT` non-results, orchestrator-verified**: both run `tests/test_streamlit_quarantine_boundary.py` monolithically, the one-process form `chunk_d_gate.py:20-24` names as the cause; `05-neighbourhood-fast` is the same set minus that file at exit 0 and `03-focused` covers the focused file at 35 passed, so the neighbourhood obligation is discharged by 03 + 05. **`bounded_run.py` captures no child stdout**, so those artifacts cannot self-classify and the label must live in prose or the composite landing will read them as red product results. `22-a9-residual`/`23-a9-residual-detail` exit 1 is A8's **required** state under D-030. All 24 reports `final_surviving_count = 0` / `cleanup_success = true`; `check --task C-050f` 0 non-compliant. **Dispatch record:** base **`a81b1d65`** (the approved C-050d tip), branch created, worktree `agent-p05f-match-rule` clean at base, primary checkout unchanged (11 protected working-tree entries before and after). Dispatched as a **`general-purpose`** agent carrying the implementer rules in-prompt, because `pwml-implementer` is unusable here (its frontmatter pins `isolation: worktree` and the harness refuses, `.claude/worktrees/` sitting inside the primary checkout). **AST verified at dispatch: NO drift** — C-050d was test-only, so `prefreeze_resolution.py` is untouched and `_canonical` `:78`, `_norm` `:82`, `_LOCATION_MEMBER_FIELDS` `:167`, `_iter_refs` `:193`, `_propagate` `:721`, `_assert_fully_propagated` `:737` all sit exactly where the charter records them; the card is still required to re-derive by symbol and report. **Orchestrator ruling issued at dispatch — the `qb` cohort is NOT authorized** (the charter's § 6 told the card to ask): C-050d already ran one complete 23-node cohort **at this exact base** (`executed=177/177`, all 23 `qb` ok), this diff is confined to two functions in `prefreeze_resolution.py`, and the authoritative combined-state cohort belongs to the composite landing under **D-030**. Chunk D **core+s8 (154) remains blocking**; if the card measures a concrete reason to think `qb` is sensitive it must stop and report rather than run it on its own authority. The dispatch also carries three traps explicitly: the **D-029/D-030/D-031 visibility trap** (branch `DECISIONS.md` stops at D-028 — read via `git show sprint/pwml-recovery:docs/pwml_recovery_sprint/DECISIONS.md`), the **stale Pack-2 base override** (PACK2-SHARED line 5 and § S5 name `bcc0bfe3`, which does **not** apply — every base proof is against `a81b1d65`), and the statement that **A9 category 5 red is a PASS condition under D-030**, not a defect, with a *change* in that number being the reportable event | C-050e ✔, C-050d ✔ | `agent/p05f-propagate-match-rule`, wt `agent-p05f-match-rule`, base `a81b1d65` | `prefreeze_resolution.py` :: `_propagate` and `_assert_fully_propagated` — the match rule and the audit rule, plus the rename-map collision refusal **only if measurement proves it necessary**. **NOT** `_iter_refs` / `_LOCATION_MEMBER_FIELDS` (C-050e just closed DEF-3 against them, and REV-050e measured that a kind filter there creates three new dangling references), **NOT** `_canonical`/`_norm` semantics — change which one the two call sites use. Indicated direction: **widen the rewriter to `_norm`, do not narrow the auditor**; narrowing would leave genuinely stale references undetected, which is the case D-015 clause 6 exists for | `REV-050f` — independent, non-author | **430** hand-authored (re-chartered from 380 under S7, **pre-commit**, once) · 45 artifacts · 2 MB |
| C-045 | `_canonicalize_species_offline` moves into the prefreeze sequence (D-016) | **`ACCEPTED` (in-stack, NOT merged to integration) 2026-08-15** — final tip **`e2b336c31eeeb270140fdf8fc00c3599c5003b36`**, `REV-045` **round 2 bare `APPROVE`** at the exact tip, **1 correction round**. **The B-1 fix was cleared by an 18,536-case adversarial sweep that found ZERO holes — and its authority rests on a 5,924-case POSITIVE CONTROL**: the same search space finds the organism-loss defect 428 times (2-row) and 5,496 times (3-row) at the pre-fix tip `0577cee7`, proving the search demonstrably reaches the defect class, then finds it 0 times at `e2b336c3`. The reviewer derived the complete loss condition first — *after the stage, two species rows share a `_norm(name)` they did not share before*, because `_dedupe_named_rows` (`ir.py:343`) keys on exactly the `_norm` the guard uses — rather than testing the exclusions as written. **All four named adversarial constructions measured:** partial vacating with a third claimant **refused**; chain `A→B` while `B→C` **accepted, both survive, identical to base**; cycle `A→B`/`B→A` **accepted as a swap, nothing lost**; and a row that appears to vacate but keeps its name **cannot occur structurally**, because exclusion 1 fires only when `_norm(old)==_norm(new)`, leaving `_norm` unchanged so the occupancy scan still counts it. A `_norm` group holding two organisms with different taxonomy ids **loses one at base too**, so that is pre-existing, not C-045's. **Exclusion 1 is provably incapable of hiding a merge** independent of reachability: a `_norm`-preserving rename cannot change the dedupe key. **Two channels the guard structurally cannot see were checked separately** — hydration suppression yields a row name-identical to the renamed row carrying no taxonomy id (a duplicate placeholder, never a distinct organism), and `_apply_create_defaults` is unreachable from a ladder rename. **Q8b: the card's ladder argument is TRUE, verified mechanically** — every assignment to a `["name"]` key in both files was enumerated by AST; there are exactly six, the three ladder write sites all guard on `_norm` inequality, and the one gap (`extraction_name and` at `ir.py:704`) is unreachable because `_canonicalize_species_rows` skips empty-`_canonical` rows first. So no rung can emit a `_norm`-preserving rename, the card's disclosure is honest, and shape 2 is correctly labelled unreachable-by-construction defensive code rather than papered over. **Over-fire measured, not argued: of 6,140 newly-introduced refusals, ZERO land on a payload the pre-fix tip handled without collapsing a row** (428 + 5,496 were losses, 216 dropped a row without a taxid). The 152-payload census is byte-identical to **both** base and the pre-fix tip, re-derived by the reviewer at the new tip. **The refusal's atomicity was proven stronger than deep equality**: same list object, same row objects, no marker, no `aliases`, no `raw_name` written. **The correction is surgical** — re-running the reviewer's 12-case differential at the corrected tip changed **exactly one** case (`10_rename_onto_other` → `AMBIGUOUS_RENAME_TARGET`); every other case bit-identical. **`ir.py` round-2 delta is provably comment-only**: `ast.dump(parse(0577cee7:ir.py)) == ast.dump(parse(e2b336c3:ir.py))` → `True`. **The one-call Chunk D deviation is ACCEPTED**: the reviewer re-derived the partition itself by `--collect-only` (core 150 + s8 4 + qb 23 = 177, union equals mono, zero pairwise overlap) and **re-ran core and every s8 node with `env -u T2PW_OFFLINE_CURATOR`** (150 passed, 1 passed ×4) plus **node15 alone with no flag (1 passed)**, proving the curator flag masked nothing. **Two new non-blocking findings, neither worth a round: `REV-045` F-7 RE-OPENED** — the A2 residual cite was corrected to `855-860` (its value at `0577cee7`) and then **13 comment lines were added above `build_pwml_ir` in the same commit**, so at the shipping tip it is `868-873`; S9 trap 1 recurring *within one commit*, documentation only, **corrected here in the control plane rather than by another round**. **F-10** — `_reject_ambiguous_species_renames`'s docstring still says *"only the bucket differs"* from `_reject_ambiguous_renames`, but the function now carries a third condition; the 22-line inline comment above the new block documents it in full so no reader is misled. Fold both into a later card only if the file is opened for another reason. **F-8 SHARPENED AND STILL WITH THE PRODUCT OWNER:** among constructed cases, **428 + 5,496 payloads exist where base produced a pathway containing BOTH organisms and the corrected tip produces no PWML at all** — fail-closed, consistent with D-029 and with C-050's approved compound behaviour, and **0 occurrences on the 152 committed payloads**, so no committed leg loses its PWML; the merge-rule-7 / § 1 alternative is `review_required`. Belongs in `DECISIONS.md`, not an implementer's judgement. **PRE-ACCEPTANCE RECORD:** correction round 1 tip **`e2b336c31eeeb270140fdf8fc00c3599c5003b36`**, parent `0577cee7` (preserved as ancestor, no amend/rebase/squash), worktree clean; round diff **190/200** fresh allowance (188+/2−), **92/100** artifacts, **0.284 MB / 3 MB**; this round's `ir.py` delta is **100% comments, zero executable lines** (the F-5/F-6 recordings). **`REV-045` round 1 returned `CORRECTION` with one BLOCKING defect, `B-1`: a species renamed onto a name another DISTINCT species row already carries was not refused, and the exporter's `_dedupe_named_rows` then silently deleted the other organism.** Measured on two genuinely different organisms (`Lactococcus lactis subsp. lactis KF147` tax `1091041` and `Lactococcus lactis` tax `1358`): base kept **both**; C-045's pre-fix tip kept **one**, absorbed the loss into a generic `duplicate_named_record` warning, and left the survivor named `Lactococcus lactis` while carrying the **strain's** id — **a wrong organism identity, not merely a lossy one** — while the operator-facing at-risk list *shrank* 2→1 at the moment of loss. **Orchestrator confirmed the structural cause independently:** `_LOCATION_MEMBER_FIELDS` has no species bucket, `_iter_refs` traverses only `processes` and `element_locations`, and `_connectivity_signature` projects only those two, so **species rows are invisible to all four of C-050's structural guards** — C-045 registered a canonicalizer for the one entity class where none of the propagation safety net applies. **C-045-introduced, not inherited:** at base the rename ran **after** `_dedupe_named_rows`, so dedupe never saw the renamed name. REV-045 also measured that the **compound** stage has no such hole (renaming `glycolate` onto an existing `Glycolic acid` raises `PREFREEZE_CONNECTIVITY_BROKEN` at base), so the case the compound stage fails closed on passed silently for species. Violates `PRODUCT_CONTRACT` **§ 5** (organism context and biological identifiers must remain equivalent), **§ 2** (correct organism context) and **merge rule 7**. **Fixed inside the card's own `_reject_ambiguous_species_renames`**, deliberately narrow so it cannot over-fire — a target inside the source's own `_norm` group is skipped, and an occupant that vacates the target is not a collision — with a **three-leg measurement** the card ran itself (base `BASE-SHAPE` exit 1 · pre-fix `FAILED` exit 1, confirming the finding on its own prior tip rather than taking it on report · corrected tip `PASSED` exit 0, refused before the freeze, payload untouched). **Over-fire proof: the 152-payload census is byte-identical to BOTH the pre-fix tip and base, zero new refusals.** Second **`qb` cohort authorized and run: 23/23 `ok`, node15 green**, via one full split-process gate call (`jobs=28 executed=177/177 omissions=0 additions=0 failed=none`, core 150 + s8 4 + qb 23) — the card flagged choosing one call over three `--only` invocations to spend 33 artifacts instead of 43, which is the same shape C-050d ran and was approved on. **A9 unchanged at `0/0/0/0/1`.** Focused **91 passed / 1 failed**, the failure being the routed golden. **The narrowing is `REV-045` round 2's Q8** — the same shape as C-050f's first B-1 fix, which introduced a fresh silent hole that this reviewer caught. **PRE-CORRECTION RECORD:** tip `0577cee77aa5f30436cc2993fcff51f9b957c935`, direct parent `0ec64d2c`, worktree clean, one commit. **Orchestrator-verified before routing review:** `src/` delta is `ir.py` + `prefreeze_resolution.py` only; **`tests/test_compound_resolution_extraction.py` and `tests/test_streamlit_quarantine_boundary.py` are both ABSENT from the commit**; 53 G11 artifacts committed; hand-authored recount **1101** (1081+/20−). Budgets **1101/1150** after the S7 re-charter, **53/60** artifacts, **0.163 MB / 3 MB**; docstrings not compressed, no probe mode dropped, both hazard tests kept — all per orchestrator ruling. **`qb` cohort AUTHORIZED and run once** (`T2PW_OFFLINE_CURATOR=1`, split-process, `--only qb`): **23/23 `ok`, `jobs=23 executed=23/23 omissions=0 additions=0 failed=none`**, gate exit 0, **node15 GREEN**. *This reversed the orchestrator's C-050f `qb` refusal, deliberately and on the record:* C-050f changed a match rule inside two functions, whereas C-045 **moves a production stage across the freeze boundary** — the largest behavioural change in the stack and the one card whose charter predicted node15 would go red — so a regression found here and attributable to C-045 alone is far cheaper than the same one surfacing at the composite landing on top of C-051. **Hazard 4a did NOT materialise:** measured directly against node15's own `_payload_with_one_bad_peripheral()` fixture — whole-object equality on `metadata`/`processes`/`biological_states`/`element_locations`, identical entity name lists, ADD-only row keys, **`renamed=0`** — because the committed index carries **zero species entries** and `Homo sapiens` (tax 9606) is already its own deterministic form; the only payload change is an ADD-only `species_canonicalization` marker node15 permits. **Hazard 4b foreclosed:** `PREFREEZE_RENAME_MAP_COLLISION` is unreachable from this ladder, since rows are grouped by `_norm` (mirroring `_dedupe_named_rows`) and only the group leader is canonicalized, so at most one target per `_match_key` — proven by test on the live collision case (`Lactococcus lactis subsp. lactis KF147` vs its uppercase twin) and by 0 intra-file `_norm` collisions among species rows across 152 payloads. **A1 found more than the charter asked:** tip `invocations=1` (caller `_canonicalize_species_rows`, exporter **0**), **base 2, both from the exporter** — `build_pwml_ir` ran inside the freeze *and* again after — so the move also removed a genuine pre-existing double execution. **A8/G9 behavioural:** `Lactococcus lactis subsp. lactis KF147` (tax 1091041, rung 4) gives **base exit 1** (frozen payload holds the strain name, exporter emits the binomial) and **tip exit 0** (frozen payload already canonical), with `canonical_graph_sha256` differing `589d1024…` → `3fab9a88…`. **Base-leg pinning is real, not nominal:** `add_src_to_path()` does `sys.path.insert(0, REPO_ROOT/"src")` derived from the **script's own** location, so `PYTHONPATH` alone would have been silently overridden and every base leg would have measured **tip** code and passed; the base legs therefore run a copy of the probe from **inside** a `git archive 0ec64d2c` export at `C:\t\c045\base`, with `--corpus-root` pointed back at the worktree so both legs read byte-identical inputs and only the code differs. **A9 unchanged at `0/0/0/0/1`**, `RESULT: MEASURED`, `C-050 ACCEPTANCE: FAILED` — B-1's `Glycine` residual untouched. Chunk D core+s8 **154 passed**; focused **89 passed / 1 failed**, the failure being the routed golden. **⚠ A3 VACUITY, flagged to `REV-045` as a named question:** the 152-payload census reports byte-identical dumps, but with **0 species renames on both legs**, because the offline index has **no species entries at all** — orchestrator-confirmed independently: `data/pathwhiz_id_db.json`'s top level is `compounds`/`proteins`/`protein_complexes`/`element_collections`/`biological_states`/`reactions`/`subcellular_locations`, with no `species` key. The rename comparison is therefore **empty-vs-empty**; the non-vacuous part is the **43 at-risk entries across 35 files reproducing identically** (the `deterministic` status path). Whether A3 is genuinely discharged is `REV-045`'s to rule — the same vacuity class REV-050f caught in C-050f's collision census. **⚠ BOUNDARY QUESTION ROUTED AS `REV-045` Q1, and the card did NOT flag it:** the `build_pwml_ir` species branch is **not a bare call removal** — the call was replaced with ~30 lines of new **reader** logic (`zip(rows, ir[ir_key])`, reading a `SPECIES_CANONICALIZATION_FIELD` marker off each payload row, appending `entry` dicts to `report["name_canonicalization"]["species"]`, building `species_at_risk` from `marker.get("status")`), plus two new module-level constants and two widened identity reads inside the owned symbol. The charter grants the call site *"strictly to remove/relocate the call"* and forbids otherwise editing `build_pwml_ir`, which is **C-051's**. **A7 forces some reader** (both `_species_at_risk` and the `name_canonicalization` log were produced by the call), so the question is whether a replay is a relocation or a second canonicalizer in disguise — a charter § 1 reject if it can compute a different answer than the pre-freeze stage did. Orchestrator confirmed no hunk lands inside `_entity_record`, `_emit_canonicalization_preflight`, `_resolve_compound_rows` or `_apply_create_defaults` (their only appearances are in comments). **Dispatch record:** base **`0ec64d2c`** (the approved C-050f tip), on a **fresh branch `agent/p45-species-prefreeze`**, wt `agent-p45-species-prefreeze`, clean at base; primary unchanged at its 11 protected entries. **`agent/gate4-d016-c045` @ `96174e8` was NOT reused and NOT pruned** — it is the D-016 ownership-decision commit, an ancestor of integration with zero unique commits, i.e. spent history. Dispatched as a **`general-purpose`** agent (`pwml-implementer` remains unusable — frontmatter pins `isolation: worktree`). **AST verified at dispatch: NO drift** — no card in the stack touched `ir.py`, so `_entity_record` `:438`, `_canonicalize_species_offline` `:617`, `_emit_canonicalization_preflight` `:704`, `build_pwml_ir` `:770`, the species call site `:844`, `_species_at_risk` `:849`/`:732`, and `PREFREEZE_CANONICALIZERS` `:855` all sit exactly where the charter records them; the card must still re-derive by symbol. **Ceilings set at dispatch: 800 hand-authored · 60 artifacts · 3 MB**, sized for a move of an ~85-line function plus its new entry point, nine acceptance clauses and a blocking Chunk D run. **Three hazards carried explicitly.** (1) **node15 WILL go red and that is P5-06, not C-045 being wrong** — C-050d's harmlessness proof asserts whole-object equality plus identical entity name lists, which forbids a pre-freeze stage from renaming an entity at all, and species canonicalization renames from the **offline name index** and **deterministic strain normalization**, neither needing a DB, so this card breaks the no-`.env` assumption that keeps it inert; the card must **stop and report**, never weaken C-050a's function, and the fixture repair routes as a separate C-050a-owned subcard exactly as C-050d did. (2) **C-045 is the first direct caller of `PREFREEZE_RENAME_MAP_COLLISION`, now known REACHABLE** — species names are far more collision-prone than compounds (`E. coli` / `E coli` / `e. coli` share one `_norm`), and `_reject_ambiguous_renames` will not catch it because it groups by `_norm(new)`; the card must measure whether its ladder can emit such a map and **stop and report** rather than weaken the guard or silently de-duplicate. (3) **P4-01** — the offline name-index path is the compliant deterministic vehicle; anything PathBank-dependent launched from a worktree is vacuously clean. Also carried: the **stale Pack-2 base override** (S5 names `bcc0bfe3`; this card's base is `0ec64d2c`), the **D-029/D-030/D-031 visibility trap**, that **A9 category 5 = 1 is a PASS condition** under D-030, that **`qb` must not be run without asking** (node15 lives there, so a red is expected rather than informative), and that `tests/test_streamlit_quarantine_boundary.py` must never go into a monolithic pytest run — C-050f burned two evidence slots on that harness fault | C-050e ✔, C-050d ✔, **C-050f ✔** | **fresh** `agent/p45-species-prefreeze`, wt `agent-p45-species-prefreeze`, base `0ec64d2c`; `agent/gate4-d016-c045` preserved unused | `ir.py` :: `_canonicalize_species_offline` (**exists** ~`:617-701`, **already called** inside `build_pwml_ir` ~`:844`). Extends C-050's `prefreeze_resolution.py`. **Moves** execution pre-freeze — does not create a second canonicalizer and does not retain duplicate pre/post-freeze execution | independent, non-author | set at dispatch |
| C-045a | re-baseline C-040's pre-extraction golden for the D-016 species move | **`ACCEPTED` (in-stack, NOT merged) 2026-08-15** — tip **`d146be48`**, cleared by `REV-045a`'s **combined bare `APPROVE`** covering C-045a and C-045b together, after **one round of `CORRECTION`** whose blocking finding became **D-032**. The objection was never the digests — `REV-045a` independently reproduced **32/32** at both SHAs — but that the golden must not be re-baselined *out from under a live regression* on a false premise. **Discharged by C-045b:** the regression is closed and the reviewer reproduced the closure itself, and the premise is now true because both production entry points run the pre-freeze sequence. **PRE-ACCEPTANCE RECORD:** tip **`d146be48aa935915b19df616a9c3bef80d0a0b11`**, direct parent `e2b336c3`, worktree clean, 39 files. **Orchestrator-verified before routing review:** the test-file diff is **exactly two hunks** (the `#:` comment above `GOLDEN` and the 32 digest entries); **`src/` is EMPTY in the diff**; **`runs/`, `runs_verify/`, `data/`, `out/`, `outputs/`, `tmp/` are ALL EMPTY in the diff**, so the protected leg fixtures stayed read-only; 35 G11 artifacts committed. Budgets **246/280** hand-authored, **37/45** artifacts, **~254 KB / 1.5 MB** — after **one granted pre-commit re-charter (25→45 artifacts, 250→280, 1→1.5 MB)**. **⚠ THAT RE-CHARTER WAS AN ORCHESTRATOR CHARTERING ERROR, THE THIRD OF THIS STACK.** The card stopped before committing and cited **D-025** correctly: the 25-artifact ceiling was **infeasible against the Chunk D gate the same charter mandated as blocking**. Verified against C-050f's committed evidence rather than taken on report — `--only core` self-allocates 6 reports and `--only s8` self-allocates 9, exactly the 15 predicted, so 22 + 15 = 37 against a ceiling of 25 — and there is no cheaper compliant route, since omitting `--only` pulls in the unauthorized `qb` cohort. **The pattern is the orchestrator's, not the cards':** C-050f 380, C-045 800, C-045a 25 artifacts — three ceilings sized for the *code change* while under-counting the *evidence the acceptance clauses demanded*, which is the exact failure mode D-025 (LOCKED) exists to name. **Proofs (to be independently re-derived by `REV-045a`, not inherited): P1** — the **pre-image was pinned first (S9 trap 3)**: under a `git archive` export of `0ec64d2c` (pre-C-045) the script reproduced **all 32 committed GOLDEN digests exactly, 32/32**, establishing that the superseded golden *was* base-code output and the base sweep is its faithful expansion rather than a fresh assertion — the step most re-baseline work skips, and the load-bearing claim of the card. Delta = the five documented paths and no others; **30 legs lose `preflight` entirely, 2 retain it** (they carry at-risk **compounds**) with `species` 1→0; only configs **A, B, E** move, since **C and D report `db_available=True`** and `_emit_canonicalization_preflight` returns early. **P2** — `P2_ir_paths_changed: []`, report keys `['preflight','warnings']`: **no path under `ir` moves, any leg, any config.** **P3, the decisive result and non-vacuous** — the production path (`run_prefreeze_resolution` → `build_pwml_ir`) shows **`legs_changed 0`**, zero difference of any kind, while **95 (leg, config) pairs across all 32 legs carry a non-empty `preflight["species"]` at the tip, identical to base**; had that count been zero the proof would have been empty. The 4 `AMBIGUOUS_RENAME_TARGET` stops (D-015 cl. 6) raise identically at both SHAs. **So the configuration the golden stops covering is provably not the production one — a stale-configuration artifact, not a regression.** **P4** — focused **92 passed / 0 failed** on C-045's exact selection, which C-045 left at 91/1, the +1 being the golden this card closes; `test_the_golden_covers_every_committed_leg_fixture` **unedited and passing**, `GOLDEN` still **32** keys. **P5** — `evidence/c045a_golden_rebaseline.py` re-derives all 32 from the repo alone via the test module's **own** `_leg_digest` over its **own** `GOLDEN` keys (no second implementation to drift), importing `t2pw` **before** the test module to defeat that module's `sys.path.insert`; this closes the **P4-02** debt `REV-045` raised on this stack. **G9 is deliberately NOT the usual shape:** labelled a **merge-rule-4 baseline move**, with the base-fails/tip-passes shape **expressly not offered as evidence** and no fabricated behavioural base proof. Chunk D **core 150/150 + s8 4/4 = 154 BLOCKING, green**, split-process, `qb` correctly not run. **11 superseded or failing G11 reports RETAINED per D-025, not deleted** — `05-prod-base` (exit 1, a real D-015 cl. 6 stop the sweep did not yet guard, since fixed) and `10`/`11` (exit 2, shell quoting) — with the sequence contiguous `01-35` so a deletion would have left a visible gap. **Orchestrator § 2 summary corrected by the card's measurement:** base at-risk species was **1 on 22 of the 30 removed legs and 2 on the other 8**, so `LEN 1 → 0` is exact only for the retained pair; not a wider delta, since those paths vanish with the whole `preflight` block, but the orchestrator's phrasing (inherited from REV-045) was imprecise and the refinement is carried in the commit message for the composite reviewer. **DISPATCH RECORD:** base **`e2b336c3`** (the approved C-045 tip), fresh branch `agent/p45a-golden-rebaseline`, wt `agent-p45a-golden`, clean at base; primary unchanged at its 11 protected entries. Ceilings **250 hand-authored · 25 artifacts · 1 MB**. The target is the inline `GOLDEN` dict at `tests/test_compound_resolution_extraction.py:322` — **32 leg→digest entries**, confirmed at dispatch. **Boundary is deliberately tiny:** the `GOLDEN` entries and their explanatory comment, nothing else. `_leg_digest`, `_configs`, `ROOT` and every other helper stay C-040's; **`test_the_golden_covers_every_committed_leg_fixture` is zero lines** (it is the guard that a new leg must be added deliberately); `ir.py` and `prefreeze_resolution.py` are C-045-approved/C-051's. **⚠ `runs/` and `runs_verify/` are PROTECTED** and hold the 32 committed leg fixtures — read-only, and a missing fixture means **stop and report**, never create one. **The card must prove the delta, not the pass**, because re-baselining is exactly how a real regression gets buried: **P1** reproduce the five changed paths independently at base with the 30/2 leg enumeration (explicitly *not* taking the orchestrator's summary on trust), **P2** the measured explicit negative that **no path under `ir`** and no other report key moves, **P3** production-path preservation, **P4** both golden tests green with the coverage guard unedited, **P5** a re-derivation route a later reader can run without the card — the **P4-02** debt class REV-045 already raised once on this stack. **G9 is NOT the usual shape:** this is a deliberate **merge-rule-4 baseline move**, so the golden trivially fails at base and passes at tip and **that is expressly not the evidence**; the card was told not to fabricate a behavioural base proof, and its G9 obligation is discharged by P1–P3. **Standing instruction: if the delta is wider than the five paths, or anything under `ir` moves, STOP AND DO NOT REGENERATE** — that would mean C-045 carries an undetected regression and the whole stack is affected, which is the far more important result. `qb` **not** authorized (no production code changes, and C-045 ran it twice at 23/23). **ROUTING RECORD:** routed 2026-08-15, held unchartered pending `REV-045` approval of C-045, and dispatched on the approved C-045 tip **before C-051**. Exists because C-045's move turns `tests/test_compound_resolution_extraction.py::test_build_pwml_ir_matches_the_pre_extraction_golden` red, and **that golden is C-040's acceptance artifact**. The implementer **refused to regenerate ~50 hashes in another card's artifact and stopped to ask** — the correct call, and the same shape as node15 being C-050a's and routed to **C-050d**. C-045 therefore commits with the golden **red, undeselected and unweakened**, supplying the delta instead of performing the move, which is what **permanent merge rule 4** contemplates: *a pinned baseline moved deliberately, with an exact documented delta*. **The delta was verified by `REV-045` Q3 before chartering — and the verification caught a wrong count, which is why the conditional existed.** REV-045 re-derived it by running `build_pwml_ir` over all 32 `GOLDEN` legs × all 5 `_configs()` at both SHAs and structurally diffing `(ir, report)`. **Corrected scope: 32 legs, ALL 32 changing** — 30 lose `report["preflight"]` plus its `noncanonical_names_collision_risk` warning, and 2 keep it with `species` 1→0 (`preflight/species` LEN 1→0, `warnings/[]/message` "1 species"→"0 species", `warnings/[]/species` LEN 1→0). **C-045a must re-baseline all 32 digests.** The card's original "36 files / 34+2" was over the **152-payload probe corpus**, not over `GOLDEN` — a different set entirely, and had C-045a been chartered on it the card would have re-baselined against the wrong corpus. **The shape is confirmed exactly as claimed: no other path appears anywhere, and in particular NO path under `ir`** — zero change to `ir["species"]`, to any entity name list, or to any other report key. **Explicit negative: zero change to `ir["species"]`, to any entity name list, or to any other report key — no biology moves.** **Cause:** `build_pwml_ir` called **without** the pre-freeze stage cannot know a species was only deterministically normalized, and restoring that would require running the ladder **twice** — the duplication **D-016** forbids. **Counter-evidence making it a stale-configuration golden rather than a regression:** on the production path (`run_prefreeze_resolution` → `build_pwml_ir`) the preflight is **fully preserved**, all 43 at-risk entries across 35 files reproducing identically (C-045 A3/A7 census) | C-045 (approved) | *(fresh branch at dispatch)* | **C-040's ownership.** `tests/test_compound_resolution_extraction.py` :: the pre-extraction golden only. **Baseline-move card** — it may not change `ir.py`, `prefreeze_resolution.py`, or any other test; if regenerating the golden requires a production change, it **stops and reports** | independent, non-author | set at dispatch |
| C-045b | wire the pre-freeze sequence into the CLI export entry point (**D-032**) | **`ACCEPTED` (in-stack, NOT merged) 2026-08-15** — tip **`328862ab`**, `REV-045a` **combined bare `APPROVE`** with C-045a, **0 correction rounds**. **The reviewer reproduced the closure itself** across three SHAs and caught its own methodological error doing so: its first probe patched module attributes and reported `resolve_*_prefreeze` as **0**, because `run_prefreeze_resolution`'s default argument binds `PREFREEZE_CANONICALIZERS` **at definition time**, so the tuple holds the original function objects and attribute patching is invisible — it redid the measurement with `sys.setprofile` and got `1/1/1/1` at tip against `0/0/0` at base. **That trap is now recorded for every future card measuring canonicalizer execution.** **A6 is STRONGER than the card claimed:** at `d146be48` the collision payload exported **exit 0 with a written `pathway.pwml` whose IR held two rows both named `Glycine`**, only one rename logged — so C-045b closed a **silent duplicate-identity export**, not merely a warning gap. **Q4 ruling on the `chunk_d_gate` count move, and the distinction is worth keeping:** C-040's golden is a **content** pin (opaque digests whose delta *is* the work, and where a regression hides), while `chunk_d_gate`'s `150` is a **cardinality** pin sitting beside an independent, strictly stronger set-based proof **not derived from it** — a silent test gain cannot survive `union == monolithic` with zero overlap/missing/extra. Non-silent, fully attributable, no separate card needed. **New finding `REV-045a` F-2, routed to C-052:** the entry-point asymmetry now runs in the **opposite** direction to the original defect — the **CLI persists and surfaces** `review_required` while **Streamlit still discards** the report (`streamlit_app.py:3587-3591`, D-029's DEF-1). **C-045b is the better behaviour; C-052 must converge Streamlit onto it, not the reverse.** **PRE-ACCEPTANCE RECORD:** tip **`328862abbc2055c1005870e33484da0accc6d781`**, parent `d146be48`, worktree clean, 40 files. **Orchestrator-verified before routing review:** `src/` delta is **`writer.py` only**; **`streamlit_app.py` ZERO diff**; `tests/test_streamlit_quarantine_boundary.py` zero lines; the golden file has a **single hunk at the `#:` comment with NO digest line touched**; `chunk_d_gate.py` shows exactly **one** changed executable line plus three docstring lines; 34 G11 reports committed. Budgets **789/800** hand-authored, **34/50** artifacts, **160 KB / 2 MB** — after a **second S7 re-charter (600 → 800)**, which is **the FOURTH ceiling the orchestrator under-set on this stack** and the one where it had explicitly claimed to size from the mandated gates rather than the diff: the production change is **76 lines**, and the other 694 is evidence the acceptance clauses demanded. **Both alternatives to raising it were refused, and the reasons are precedent:** dropping `c045b_base_tree.py` would have recreated the **P4-02** debt class (an uncommitted measurement script) that `REV-045` raised against this very stack and C-045a closed by committing its own — and that tool re-hashed **all 1731 blobs against `git ls-tree`** to prove the base export, which is what makes A7 real rather than nominal; trimming probe prose is what the orchestrator refused for C-050f, where `REV-050f` later found the surviving rationale was what made the change reviewable. **A1 — the D-032 regression is closed**, measured through `scripts/run_pwml.py`: `KF147` occurrences in the emitted PWML go **1 → 0**, `<name>Lactococcus lactis</name>` replaces the strain string, `aliases:["…KF147"]` and the `deterministic_strain_normalization` log entry and `preflight.species` are all restored. **A1 is PARTIAL and correctly so:** `raw_name` reaches the species *payload* row but stops at `ir._component_record`'s projection — **measured identical on the Streamlit path** through a plain `run_prefreeze_resolution` + `build_pwml_ir` call touching none of the card's code, so it is C-045's consequence and **C-051's** to fix (`REV-045` F-1, already routed), not a C-045b shortfall. **A2 — discharged by EXECUTION TRACE, and this is the evidence C-051 will be held to under D-032 clause 4:** `sys.setprofile` over one CLI export gives `run_prefreeze_resolution` **1**, `resolve_compounds_prefreeze` **1**, `resolve_species_prefreeze` **1**, `build_pwml_ir` **1**, in that order, against **0/0/0** at base where compound resolution came *only* from `ir.py:979`. So when C-051 withdraws that call site the CLI keeps compound resolution. **The seam's placement is MEASURED, not assumed:** placing it *before* `normalize_process_payload` yields `['Glycine', 'pyruvate', 'Pyruvic acid']` with `entities_added_as_compounds=1` — an invented compound and split connectivity — while the shipped order (after the post-normalization gate, before `build_pwml_ir`) yields `['Glycine', 'Pyruvic acid']`. **Error/report semantics per § 2b:** `report["ok"] is False` is persisted to `pwml_prefreeze_resolution_report.json` and surfaced as `prefreeze_review_required` but **deliberately does not abort** (**D-029** rules `db_unavailable` a `review_required` outcome that must not raise by itself; **merge rule 7** keeps the incomplete-but-correct pathway; acting on it is C-052's seam), while a `PrefreezeResolutionError` is caught and returned in the entry point's **existing** failure shape with a named code — making an existing terminal outcome legible rather than inventing an abort, with real recovery since the input file is byte-unchanged and the report names the offending row. **A6 — strictly more refusing:** a collision payload that **exported a PWML silently at base** is now refused. **A3** `streamlit_app.py` zero diff and C-050's five-category probe identical at base and tip. **A4** no double execution. **A5** second pass `renamed=0/0` with the payload byte-unchanged, two exports byte-identical, and a raise leaves the input unchanged with **no PWML written**. **A8** category 5 = exactly `1`, untouched. **⚠ AUTHORIZED CHUNK D BASELINE MOVE:** the charter mandated a test in `tests/test_pwml_writer.py`, which is in `CORE`, so `chunk_d_gate.py:70`'s pinned `("core", CORE, 150)` no longer matched. The orchestrator authorized **150 → 152** (`TOTAL` at `:74` is **derived** and follows to **179**; verified before ruling), plus the comments quoting the counts, on the grounds that the gate's **set-based** proof is the substantive check and passed independently — `SETS_EQUAL=True`, `overlap=0`, `missing=0`, `extra=0`, no deselection — leaving nowhere for a regression to hide, which is what distinguishes it from C-040's ~50-hash acceptance golden that was routed to its own card. The two added tests are named in the commit and in the `:73` comment, which now **records the deliberate move rather than continuing to assert 177** — a stale pin comment being exactly how C-045a's `#:` comment misled the next reader. **Re-run partition proof passes at 152 / 4 / 23 = 179**, `union=179 monolithic=179 missing=0 extra=0 SETS_EQUAL=True`, both gate invocations outer exit 0. **⚠ THE COMPOSITE-LANDING PARTITION IS NOW 152 + 4 + 23 = 179, NOT 177.** New deferred finding registered, not fixed: **`process_normalizer.py:82`** maps `"pyruvic acid" → "pyruvate"` on reaction participant references but **not** on compound entity rows, then adds the now-dangling name as a **new compound**; a control leg with no pre-freeze call anywhere reproduces it at base, so it is **SHA-invariant and not this card's** — relevant to any future replay or round-trip work. **DISPATCH RECORD:** base **`d146be48`** (the C-045a tip), fresh branch `agent/p45b-cli-prefreeze-seam`, wt `agent-p45b-cli-seam`, clean at base. Ceilings **600 hand-authored · 50 artifacts · 2 MB** — **sized from the gates the charter mandates rather than from the diff**, after three ceilings on this stack were under-set by counting only code (D-025's exact failure mode); Chunk D core+s8 alone self-allocates 15. **Closes a MEASURED live biology regression**, not an inferred one: `REV-045a` drove the documented README command over a taxonomy-identified strain at both SHAs and got `<name>Lactococcus lactis</name>` at base against `<name>Lactococcus lactis subsp. lactis KF147</name>` at tip, with `raw_name`/`aliases` provenance and the `deterministic_strain_normalization` log present at base and **absent** at tip, and `preflight` going from populated to `null`. **Owns `writer.py :: run_pwml_pipeline_export`** — the pre-freeze call site and its report/error handling — plus an additive CLI-path species assertion in `tests/test_pwml_writer.py` and **the `#:` comment above `GOLDEN` only** (zero digest lines; `REV-045a` reproduced all 32 at both SHAs and they are correct). **NOT** `build_pwml_ir` or the `ir.py:979` call site (C-051 — explicitly told not to "help"), **NOT** any symbol in `prefreeze_resolution.py` (it is a **caller**, not an editor), **NOT** `streamlit_app.py` (zero diff required). **Must wire the whole `PREFREEZE_CANONICALIZERS` tuple, never per-stage**, so compounds are covered by construction — **A2 requires a measured demonstration that `resolve_compounds_prefreeze` executes on the CLI path**, not an argument from the tuple's contents. Also carried: **D-029** binds the error path (`db_unavailable` is `review_required`, must not raise by itself) and **`PRODUCT_CONTRACT` § 1 / merge rule 7** forbid inventing a new terminal abort where Streamlit has none; **A4** demands an execution trace proving the stage runs **exactly once** (the duplicate-execution trap that produced B-1); **A7** warns that `_repo_root.add_src_to_path()` inserts the containing checkout's `src` at `sys.path[0]`, so `PYTHONPATH` alone is overridden and a base leg will measure **tip** code **and pass** — the base tree must be hash-verified against `git show`; **A8** A9 category 5 = `1` is a **PASS condition**. `qb` **not** authorized (it exercises the Streamlit path, which must not change) | C-045a | fresh `agent/p45b-cli-prefreeze-seam`, wt `agent-p45b-cli-seam`, base `d146be48` | `writer.py` :: `run_pwml_pipeline_export` (prefreeze call site + report/error handling) · `tests/test_pwml_writer.py` (additive CLI-path assertion) · the `GOLDEN` `#:` comment only | independent, non-author | 600 hand-authored · 50 artifacts · 2 MB |
| C-051 | remove/foreclose the post-freeze `_resolve_compound_rows` path; discharge B-1 | **RECONCILIATION 2026-08-18 (orchestrator, from Git ancestry):** **C-051 IS MERGED** — it landed inside the twelve-card composite **`beddcdd`** under D-030. The "(in-stack, **NOT merged**)" qualifier below is **SUPERSEDED**; `2b3de80a` is an ancestor of the integration tip. **`ACCEPTED` (in-stack, NOT merged) 2026-08-15** — tip **`2b3de80a`**, `REV-051` bare **APPROVE**, **1 correction round**. **⚠ B-1 IS DISCHARGED AND INDEPENDENTLY VERIFIED AT BOTH SHAs BY THE REVIEWER:** base `0/0/0/0/1`, `C-050 ACCEPTANCE: FAILED`, exit 1 — tip `0/0/0/0/0`, `PASSED`, exit 0, with both legs **paired 4/4 one-to-one** so the result is MEASURED rather than a lower bound, and **no category moved in the wrong direction**. That is the outcome the entire seven-card stack existed to produce. **The reviewer closed the base-pinning trap ADVERSARIALLY** rather than by construction: it ran the base leg with `PYTHONPATH` pointed at the **tip** `src` and the leg still reported the base path and still failed, because `_repo_root.add_src_to_path()` does `sys.path.insert(0, …)` — so a mis-pinned base run cannot masquerade as a pass here. Base tree materialized from the object DB with **3088/3088 blobs re-hashed against `git ls-tree`**. **The non-vacuity guard was attacked four ways and fired every time it could:** stage deleted → **70 failed**; verdict stripped from all but the first row → 54 failed (fewer only because many fixtures carry a single row). Census: **80 nodes touch the helper, 79 engage with all three assertions live, and exactly ONE no-ops** — `test_create_defaults_fill_unmatched_species_and_cell_location`, whose fixture is `“compounds”: []`, **precisely where `build_pwml_ir`'s own assertion is equally vacuous**. The reviewer located that one case rather than accepting the equivalence, and confirmed `not compound_verdicts(payload)` is falsy only for a zero-length list, so the no-op cannot swallow an unresolved row. The 10 mutation survivors are all in `test_strict_quarantine_real_artifact_replay.py`, whose `_ir_codes` catches exceptions **by pre-existing design** — and its exact-equality pin `test_the_full_stack_baseline_is_exactly_what_was_reported` **is** in the mutation-failure list, with `FULL_STACK_BASELINE` untouched and reproducing at tip. **Test-function census 304 → 304**, and grepping the entire test diff for changed `assert`/comparison lines returns **ZERO** — nothing relaxed to a subset, allowlist or normalized comparand. **TWO THINGS THE REVIEWER FOUND THAT THE CARD DID NOT.** (i) `_entity_record`'s materializing path is **still live in isolation** — handed a row with `pathbank_compound_id: 99` and no `pathwhiz_id` it *does* emit `pathwhiz_id: 99` — but such a row **cannot reach it** (tip REFUSED, base ACCEPTED), which makes the no-code D-027 result **genuine rather than coincidental**. (ii) A **stronger ground** for the carrier being underivable: `available` is computed at `compound_resolution.py:485` from `db_resolver.available()`, where a `None` resolver is first replaced by `PathBankDbResolver.from_env()` — so reproducing it inside `build_pwml_ir` would be **the exporter probing an external service after the freeze, which D-015 forbids outright**. **A4 refined by the reviewer:** **6 of 32** legs actually produce a `pathway.pwml`; those 6 are byte-identical and the other 26 produce none at either SHA — a precision the card's “32/32” phrasing lost. `ok` flags differ on 0 legs, compound names on 0 legs, and the only delta is IR-internal `db_status` on 10 legs where the tip **preserves** `matched_offline_name_index` that the base **overwrote** with `legacy_id_unverified`. **C-040's purity pin was EXECUTED, not asserted:** `ir._resolve_compound_rows is cr._resolve_compound_rows` → `True`. **A2 fail-closed confirmed at runtime** with the live message naming all 10 offending rows by name and key. **Q6 rulings: all three escalations CORRECT**, and the reviewer **endorsed the refusal** to re-point C-045b's CLI test, confirming that re-homing its guard would make the following `preflight` assertion pass in **both** configurations — destroying the property the guard exists to protect. **`REV-051` independently re-derived the D-033 clause 3 enumeration**: `build_pwml_ir` has exactly two callers in `src/`. **New findings: F-1 (MEDIUM)** — the tip emits *“Resolution DB unavailable (db_not_configured)”* when the DB **was** reachable and **was** consulted pre-freeze, a **false statement in product-visible content** under D-032 cl. 6; fail-closed and invisible on the offline corpus, **routed to C-051a**. **F-2 (LOW)** — the persisted `pwml_ir_report.json` delta the card's A4 did not compare: `db_resolution` collapses on 13/13 CLI legs, but the content is **fully relocated** into `pwml_prefreeze_resolution_report.json` with identical counts, so nothing is lost on the CLI; on Streamlit the report is discarded, adding to D-029's C-052 debt. **F-3 (LOW)** — `writer.is_non_blocking_pwml_ir_error`'s tolerated branch now has no producer and no positive test. **F-4 (INFO, and it corrects D-033's framing) — `run_pwml_export` is the ONLY `build_pwml_ir` caller in `streamlit_app.py` and the only writer of `outputs/pathway.pwml`, so C-051a's seam covers ALL Streamlit PWML output, not a “refinement re-export” side path.** Orchestrator-confirmed independently. **F-5 (INFO)** — `_PROVENANCE_FIELDS` omits `raw_name`, pre-existing and identical at base. **The reviewer disclosed its own slot-allocation slip**, in review scratch outside the repository, deleting nothing — the same failure the card disclosed twice. **PRE-ACCEPTANCE RECORD:** tip **`2b3de80a72c2e10bcfffdd105ebe223e68e98cba`**, 3 commits on base `328862ab`, all preserved. **Orchestrator-verified before routing review:** `src/` delta is **`ir.py` only, 64+/8-** across all three commits; **`test_compound_resolution_extraction.py` and `test_streamlit_quarantine_boundary.py` are ZERO lines**, so all 32 `GOLDEN` digests and node15 are untouched; 15 test files (14 modified + 1 new helper); 73 G11 reports. Budgets **1026/1100**, **73/100**, **360 KB / 4 MB** — the card stopped writing code at 1026 rather than approach the ceiling. **ROUND 2 — the 70 refused nodes re-pointed, 70 → 3.** Non-vacuity is centralized in ONE auditable place, `tests/helpers_prefreeze.py`: `prefrozen()` asserts on every call that the payload **has** compound rows, that **no** row arrived with a verdict, that **every** row leaves with one, and that the count is unchanged — so deleting the stage, stubbing it, or letting it skip rows fails every re-pointed test at that line. `prefrozen_when_compounded()` no-ops **only** where there are no compound rows, which the card argues is exactly where `build_pwml_ir`'s own assertion is equally vacuous. `test_pwml_writer.py` extends the file's **existing** shim at `:157`; `test_streamlit_stage8_export_contract.py` wraps at the **fixture** so `assert payload == before` stays intact. **FINDING 2 — ruled NOT DERIVABLE, and the card took the stop branch, changing no production code.** The decisive population is `all_legacy`: every row carrying a `pathbank_compound_id` makes `_resolve_compound_rows` take its legacy branch and `continue` **before the resolver is ever consulted**, so the resolved rows are **byte-identical** while `available` differs — `build_pwml_ir` would be **guessing**, and `_emit_canonicalization_preflight` would then warn or stay silent on a false premise. The card also located where the value **does** exist pre-freeze: `report[“compounds”][“resolution_report”][“db_resolution”][“available”]`, which simply never reaches the exporter. **Routed to C-051a.** **THREE NODES ESCALATED, NONE WEAKENED**, all in `test_pwml_writer.py` and all classified PASS conditions for this card: (1) `test_offline_index_emits_canonical_compound_name` — `_resolve_to_fixed_point`'s `_PROVENANCE_FIELDS` is `(“db_status”,“chosen_rule”,“confidence”,“db_match”)` and **`raw_name` is not in it**, so a second pass overwrites the extraction name; measured **identically at base and tip**, pre-existing in C-050/C-045's module, and it joins `REV-045`'s F-1 (`_component_record` drops `raw_name`) as a **`raw_name` provenance cluster** now spanning two layers. (2) `test_compound_db_resolution_failures_are_non_blocking_for_pwml_build` — its **premise is structurally gone** since the exporter no longer runs resolution, so no input change can reach `compound_db_resolution_failed`; its *intent* should now be satisfied by **D-029**'s `review_required` path. (3) `test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance` — **C-045b's own test**, and the card **refused to re-point it as green-washing**, because doing so turns it green while the defect it guards is live: `db_available` is structurally falsy and the preflight fires unconditionally. **That refusal is the most consequential judgement in the card** and is put to `REV-051` as Q6.3. **Self-correction to its own round-1 report:** `test_strict_failure_replay.py`'s 2 failures are **not this card's** — that module never touches `build_pwml_ir` and both fail identically on the hash-verified base, so the true blast radius was **70, not 72**. **Gates: Chunk D core 149/152** (up from 109; the 3 shortfalls are exactly the 3 escalated nodes) · **s8 4/4 GREEN** · focused **692 passed / 6 failed / 8 skipped** (3 escalated + 1 `GOLDEN` awaiting C-051b + 2 pre-existing) · `qb` **not re-run** · **A10/B-1 re-confirmed `0/0/0/0/0`, `C-050 ACCEPTANCE: PASSED`** · **A4 re-confirmed 32/32 byte-identical** · **A5 re-confirmed: compound path foreclosed, D-027 not invoked**. **Second slot-allocation slip disclosed AND process-fixed:** the s8 `--json` path was hand-written instead of the allocated `62-…`, exactly that placeholder was deleted, the real report retained at `63-chunkd-s8-repointed.json`, and every later job now captures the allocator's output into a variable. **PRE-CORRECTION RECORD:** round 1 stopped at a measured boundary breach at tip `b371cfc0` (2 commits on base `328862ab`), `src/` delta **`ir.py` only, 64+/8-**. **⚠ B-1 IS DISCHARGED — the point of the entire stack. A9 reads `0/0/0/0/0`, `RESULT: MEASURED`, `C-050 ACCEPTANCE: PASSED`** — the first zero on this stack, after `0/0/0/0/1` held as a deliberate D-030 pass condition through six cards. The `Glycine` `matched_offline_name_index -> legacy_id_unverified` residual is gone. **A9/D-032 cl. 4 measured**: `sys.setprofile` over one `run_pwml_pipeline_export` gives `run_prefreeze_resolution 1 / resolve_compounds_prefreeze 1 / resolve_species_prefreeze 1 / _resolve_compound_rows 3` (the pre-freeze fixed-point loop) `/ build_pwml_ir 1`, and **C-045b's own probe still PASSES at this tip** — while the base leg shows `_resolve_compound_rows 4` with the exported verdict **degrading to `unmatched`**, the clearest evidence yet that the redundant pass was *overwriting* the pre-freeze verdict rather than merely duplicating it. **A4**: all 32 committed leg fixtures through the CLI at a hash-verified base and at tip — `pathway.pwml` sha256 identical **32/32**, compound names identical 32/32; the only delta is IR-internal `db_status` on 10 legs where the **tip preserves** what base overwrote. **A5/P2-06 — no-code result, D-027 NOT invoked, D-021's `_entity_record` lock RETAINED**: C-050's `_project_db_identity` runs the identical `_db_id` projection pre-freeze, so `_entity_record:447` only *forwards*, including the mapping-metadata case. **The orchestrator's 2c STOP condition correctly did not trigger** — for compounds the live post-freeze path runs through `_entity_record:447`, while `_component_record:419` serves species, locations, cell types and tissues. **A7** base tree via `git cat-file` with **1767/1767 blobs re-hashed against `git ls-tree`**; base renames and stamps `db_status` *after* the freeze, tip refuses. **AST drift: none** — all nine orchestrator measurements reproduced. **STOPPED on three items outside its grant, all routed:** (1) **~70 failing nodes across 16 files** — 67 are the new refusal correctly firing on tests that hand `build_pwml_ir` a never-pre-resolved payload; **re-pointing AUTHORIZED** with three binding conditions, the load-bearing one being that every re-pointed helper must be **non-vacuous** (it must fail if the pre-freeze stage does not run), following C-045's `_species_canonicalized_payload` precedent that `REV-045` probed and confirmed. (2) **`report[“db_resolution”][“available”]` is no longer populated**, so `_emit_canonicalization_preflight` (`ir.py:797-798`) would fire on a false premise — **ruled MEASURE-FIRST**: if `build_pwml_ir` can derive it faithfully from the pre-freeze verdicts already on the rows, fix it inside that owned symbol and prove the derivation in both directions; if it would be **guessing, stop and report**, because inference dressed as a carrier is worse than a missing one. (3) **A THIRD export path runs no pre-freeze sequence** — see **D-033**; routed to **C-051a**, and `qb` node06's red is a **PASS condition** for C-051, not a defect to chase. **The 32 `GOLDEN` digests are NOT this card's** despite the test file's docstring naming C-051 as the intended mover — that docstring predates C-045a, which already moved the golden once and predicted this second move. Routed to **C-051b**, which lands **last** so the digests reflect the final combined state and move only once more. **Budgets 554/1100 hand-authored, 52 artifacts with the ceiling raised 80 -> 100, ~246 KB/4 MB**; `qb` **not** to be re-run (its one run is done and node06's red is classified). **Two process deviations self-disclosed and correctly handled:** an abandoned `22-chunkd-qb.json` reservation created by hand-writing a `--json` path instead of the allocated slot — **exactly that placeholder deleted and disclosed**, with the real job's complete report at `14-chunkd-qb.json`, which is what PACK2-SHARED requires since the failure mode is *quiet* cleanup; and unbounded design-phase exploration with **every cited result** from a bounded committed run. **Deferred, registered, not to be fixed:** `build_pwml_ir`'s now-dead `name_index`/`resolved_name_index` parameters · `ir.py:16`'s pre-existing unused imports (C-040 residue) · **proteins and species still newly materialize `pathwhiz_id` post-freeze from `mapping_meta.candidates[0]`** — the most interesting of the three, recorded for the C-052 queue. **DISPATCH RECORD:** base **`328862ab`** (the approved C-045b tip), branch `agent/p05c-ir-assert-only` (registered in `MASTER_PLAN` § 9; **not** `agent/p05c-node15-comparand` @ `d88dcb5e`, which shares the prefix and is merged), wt `agent-p05c-ir-assert`, clean at base. Ceilings **1100 hand-authored · 80 artifacts · 4 MB**, sized from the mandated gates after four under-set ceilings on this stack. **THE CHARTER PREDATES THE STACK AND SIX CORRECTIONS WERE ISSUED AT DISPATCH:** (2a) its "NOT DISPATCHABLE pending C-050 merging at `768be750`" status is superseded by **D-030**; (2b) **every line number in it is stale** — measured at base, `_component_record` `:415`, `_entity_record` `:438`, `_canonicalize_species_offline` `:636`, `_emit_canonicalization_preflight` `:750`, `build_pwml_ir` `:816`, `_resolve_compound_rows(` **`:979`** (charter says `:911`), and there is **no ladder call inside `build_pwml_ir` any more** (charter says `:844`) because C-045 moved it, leaving only a marker reader; (2c) **⚠ THE CHARTER UNDERCOUNTS THE `pathwhiz_id` SITES — THERE ARE TWO**, `_component_record:419` as well as the `_entity_record:447` that P2-06 and **D-027** name, and **D-027's carve-out does not reach `_component_record`**, so if the live post-freeze path runs through `:419` the card must **stop and report** rather than proceed — *this is the THIRD time a charter of mine undercounted sites, after C-050f's two rewrite sites and C-045b's two entry points, and the card was told so explicitly*; (2d) the loose earlier routing of the `_component_record` **`raw_name`** omission to C-051 is **formally WITHDRAWN** — outside its grant, non-blocking, and this card is already large; (2e) **SMOKE 460 is NOT run here** — under D-030 it runs once at the composite landing; (2f) **`qb` AUTHORIZED, one run**, `T2PW_OFFLINE_CURATOR=1` mandatory in the bounded child (C-050b's flag **is** merged, `curation/pathway_curator.py:62`), and the partition is now **179**; (2g) C-045 is **accepted, not in flight**, so § 6's concurrency stop does not apply. **Two acceptance clauses added beyond the charter: A9 = D-032 clause 4** — removing `:979` must leave compound resolution intact on **both** entry points, **measured through `scripts/run_pwml.py`**, with C-045b's `sys.setprofile` evidence as the model and its reviewer's attribute-patching trap flagged (default-argument binding makes attribute patches invisible and reports a false zero); **A10 = B-1's discharge** — A9's five categories have read `0/0/0/0/1` through every card of this stack as a D-030 **PASS** condition, and **this is the card that must take category 5 to zero**; if it does not, that is a finding to report, not a probe to adjust. **A no-code D-027 result is explicitly a valid completion** if the reachability proof shows the path already foreclosed | C-045, **C-045b ✔**, never concurrent with C-045. **⚠ MUST NOT BE DISPATCHED UNTIL C-045b IS APPROVED:** `_resolve_compound_rows` is still called at `ir.py:979` inside `build_pwml_ir`, so the CLI export path currently receives compound resolution **from the exporter**; removing that call site while the CLI seam is unwired would strip compound resolution from a documented production entry point, repeating D-032's species defect on the primary biology of a metabolic pathway. **Acceptance extended by D-032 clause 4:** must show that removing the call site leaves compound resolution intact on **both** entry points, **measured through the CLI**, not argued | C-045, **C-045b** | `agent/p05c-ir-assert-only` | `ir.py` :: `build_pwml_ir` incl. its compound-resolution call site. **D-027** authorizes touching `_entity_record`'s `pathwhiz_id` materialization **only if measurement proves the path still reachable**; if unreachable, prove it and leave `_entity_record` unchanged. **Not an IR refactor.** Acceptance must show **all five** C-050 A9 categories zero on the completed stack | final independent composite reviewer, author of none of the stacked cards | set at dispatch |
| C-051a | make the pre-freeze stage's output reach the exporter on EVERY path — the Streamlit seam (**D-033**) **and** the `db_resolution.available` carrier | **`ACCEPTED` (in-stack, NOT merged) 2026-08-15** — final tip **`b8f7902c`**, `REV-051a` **round 2 bare APPROVE**, **1 correction round**. `src/` delta **+170/−0 across `streamlit_app.py`, `ir.py`, `prefreeze_resolution.py`**; **zero deletion lines anywhere in the whole two-round diff**; all five forbidden files absent from the diff. Budgets: round 1 **1100/1100**, round 2 **17/40**, artifacts **76/85**, size **243.9 KB / 3 MB**. **⚠ `REV-051a` ROUND 1 RETURNED `CORRECTION` ON A NARROWED ASSERTION, AND IT MEASURED RATHER THAN ARGUED.** The card had narrowed a whole-marker equality to `[“available”]`, claiming the down-DB test pinned the marker exhaustively. The reviewer injected a stray key **only when `available is True`** and got **`2 failed, 299 passed` — identical to the clean run. Nothing caught it.** Cause: the “exhaustive” pin is parametrized `(None, “db_not_configured”)` and `(_DownDb(), “harvest_db_down”)` — **both legs are `available: False`** — and `ir_report[“db_resolution”]` carries the same asymmetry, so the narrowed line was the **sole whole-marker pin on the reachable leg**. **Not cosmetic:** the marker moves `canonical_payload_sha256` and `admitted_payload_hash` — the fingerprint every quarantine decision binds to — while leaving `canonical_graph_sha256` untouched. **Fixed and PROVEN fixed:** the same mutation now gives `4 failed, 297 passed`, the two new failures being exactly the reachable params, and the all-markers control went from 2 catching nodes to **6**. Assertion census **30 → 30**: one assertion replaced, none added or dropped. **This is the third time on this stack that “coverage moved elsewhere” proved false once someone actually mutated the code.** **F-2 discharged with reasoning the orchestrator had not considered:** the card chose an `assert` over `PrefreezeResolutionError` for the registry invariant because **D-029 reserves that raise for structural payload defects**, and `REV-051a` verified the load-bearing half by reading the whole handler — `writer.py:2696-2721` would **catch** it, write `stage: prefreeze_resolution, ok: False, raised: True` into the biology report, and tell the operator **“Pre-freeze canonicalization failed”**, sending them to hunt a dangling reference in their own payload for what is a code change. The reviewer also checked the `python -O` caveat empirically: **no `-O`/`-OO`/`PYTHONOPTIMIZE` anywhere**, **no `except AssertionError` in `src/`**, and 15 existing production asserts — established idiom, caveat theoretical. It then ran the assert live: silent on the real registry, firing and **naming the offending registry** when a second recorder is appended. **⚠ THE REVIEWER ALSO CORRECTED THE ORCHESTRATOR'S CHUNK D REASONING.** The orchestrator ruled no re-run needed because the new seam file is in none of `CORE`/`S8`/`QB`. **That answers COLLECTION, not BEHAVIOUR** — round 2 also touched `prefreeze_resolution.py`, which `CORE` tests execute through via C-051's `helpers_prefreeze`. It ran the gate anyway: partition **179** intact, core **150/152**, s8 **4/4**. **Process items, both ruled in the card's favour and one correcting the orchestrator:** the **artifact ceiling breach (74/70) is a recorded S7 deviation but NOT a D-025 violation** — no evidence was deleted to fit (the one deletion was an **81,566-byte RULE-5-violating** report replaced by a compliant method, the opposite of hiding evidence), the overage is 4 cleanup JSONs against a size ceiling at **8%** use, and rework would itself require deleting genuine evidence; the reviewer confirmed **56 + 16 > 70 before the card wrote a line**, so the instruction was unsatisfiable. **The exactly-1100 landing is cleared on a stronger basis than either party argued:** the diff has **zero deletion lines anywhere**, so nothing pre-existing could have been lost — the compression trimmed only the card's own uncommitted draft, **a different act** from the one refused for C-050f and C-045b where surviving rationale in *existing* files was at stake; measured **1.8:1 rationale-to-code** with all eight things a future reader needs present. **⚠ ADOPTED AS STANDING PRACTICE — `REV-051a`'s correction to the orchestrator: “the defect is in applying D-025, not in D-025.”** The rule already says to count what the gates allocate. **Six under-set ceilings** (C-050f 380, C-045 800, C-045a 25 artifacts, C-045b 600, C-051a 900, C-051a artifacts 70) all came from sizing at charter time before counting gate self-allocations. **From here every ceiling is computed as `consumed + Σ(gate self-allocation for each ordered gate) + headroom`, at the moment the work is ordered.** The orchestrator declined to raise the breached ceiling retroactively, since S7 forbids raising one after a commit exists and papering over it would hollow out the rule. **Substance verified:** `REV-051`'s F-1 closed on **all three legs at both SHAs** with adversarial base pinning — and **leg B vindicates the orchestrator's `reason` authorization**, since with `available` alone a configured-and-**down** DB would still have reported `db_not_configured`; `harvest_db_down` now appears in the marker, the IR report, the preflight **and** the message. **A5 honoured and sharper than reported:** at C-051 that guard **`KeyError`ed**, so it was not merely red but **inoperative**, and the `preflight[“species”]` assertion beneath it had gone **vacuous**; the carrier made it functional again in its original location with its original text and **zero lines** in `test_pwml_writer.py`. **The placement counterfactual was re-proved at library level:** above `decision_matches` reuse collapses to `False`, and where the card put it the payload hash is **unchanged**, so it cannot invalidate a stored decision. **The batch exporter** (`c051a_base_tree_batch.py`, reusable by C-051b/C-051c) took **806 descendants → 2** and **81,566 bytes → 2,707**, cross-checked at **1789/1789 byte-identical** against the reviewer's independently built tree. **A9 `0/0/0/0/0` `PASSED`** · **A7 32/32** with 6/6 `pathway.pwml` identical · `canonical_graph_sha256` unaffected by the marker · **the focused failure count went DOWN by one, and the one that cleared is C-045b's**. New findings: **F-2** (aggregation policy — closed by the card's assert), **F-3** (a no-op `isinstance` assertion), and a nit that the batch exporter does not clear `--dest`. **DISPATCH RECORD:** base **`2b3de80a`** (the approved C-051 tip), fresh branch `agent/p51a-streamlit-seam-carrier`, wt `agent-p51a-seam-carrier`, clean at base. Ceilings **900 hand-authored · 70 artifacts · 3 MB**, sized from the mandated gates. **⚠ D-033's “refinement re-export” FRAMING IS CORRECTED BY `REV-051` F-4 AND CARRIED INTO THE DISPATCH:** `build_pwml_ir(` appears **exactly once** in `streamlit_app.py`, at `:4052`, and `outputs/pathway.pwml` is written at `:4104` in the same function — orchestrator-confirmed — so `run_pwml_export` is **the only Streamlit PWML producer** and this seam covers **all** Streamlit PWML output rather than a niche path. **A2 requires `qb` node06 to go GREEN without touching node06 or `test_streamlit_quarantine_boundary.py` (zero lines)**; `qb` is authorized for one run with `T2PW_OFFLINE_CURATOR=1` mandatory. **A3 names the discriminating test the card must use** — the `all_legacy` population, where rows are byte-identical while `available` differs, which is the whole reason inference fails. **A5 forbids undoing C-051's refusal:** C-045b's `report[“db_resolution”][“available”] is False` assertion is a deliberate **P4-01 non-vacuity guard**, and it must pass **because the carrier became truthful**, not because the guard moved — `REV-051` measured that relocating it would make the following `preflight` assertion pass in **both** configurations. **A9 makes B-1 a REGRESSION GUARD**: `0/0/0/0/0` must hold, and any category leaving zero is a stop-and-report. Also carried: the attribute-patching trap (default-argument binding makes patches invisible and reports a false zero — use `sys.setprofile`), the adversarial base-pinning method `REV-051` used, the instruction to **capture the allocator's output into a variable from the start** after C-051 slipped twice, and the warning that `run_pwml_export` writes protected `outputs/pathway.pwml` at `:4104`. **PART 2 rationale:** **two parts, both measured by C-051 and both the same class of defect: the pre-freeze stage's output failing to reach the exporter.** **PART 2, added 2026-08-15:** `report[“db_resolution”][“available”]` is no longer populated once `ir.py:979` is removed, so `_emit_canonicalization_preflight` (`ir.py:797-798`) fires on a false premise. C-051 was ruled **measure-first** and proved it **NOT derivable** inside `build_pwml_ir`: for an `all_legacy` population the resolved rows are byte-identical while `available` differs, so the exporter would be guessing. The value exists pre-freeze at `report[“compounds”][“resolution_report”][“db_resolution”][“available”]` and must be **carried**, never inferred — inference dressed as a carrier is worse than a missing one. Closing it also closes **two of C-051's three escalated nodes**. **PART 1:** pending C-051 approval, then dispatched on the approved C-051 tip and **before C-051b**. Exists because **D-032 clause 1 undercounted the entry points**: `run_prefreeze_resolution` sits at `streamlit_app.py:3587` inside `run_post_pipeline_sbml_artifacts` (**2617-3809**), but `build_pwml_ir` is called at **`:4052` inside `run_pwml_export` (3828-4136)** — outside that function — and the **refinement re-export** (`_render_review_refine_section:2231 -> `_generate_pwml_from_refinement_working_json:1898` -> `run_pwml_export`) reaches the exporter without ever crossing the seam. Orchestrator-confirmed independently. At base this was invisible because `ir.py:979` silently repaired those rows **after the freeze**; C-051's removal converts that into a loud refusal, so **`qb` node06 turning red is the defect becoming visible, not a regression**. Owns exactly that seam; `run_pwml_export` and the refinement path are otherwise C-030/C-052 surface | C-051 (approved) | *(fresh branch at dispatch)* | `streamlit_app.py` :: the `run_pwml_export` pre-freeze seam only | independent, non-author | set at dispatch |
| C-051c | preserve `raw_name` across the fixed-point loop, and re-home the orphaned non-blocking test | **`ACCEPTED` (in-stack, NOT merged) 2026-08-16** — tip **`487ac21c`**, `REV-051c` bare **APPROVE**, **0 correction rounds**. **⚠ CHUNK D CORE IS 152/152 FOR THE FIRST TIME IN THIS STACK** — s8 4/4, qb 23/23 with node06 **and** node15 green, partition `SETS_EQUAL=True`, union 179, `jobs=28 executed=179/179 omissions=0 additions=0 failed=none`. Both reds it cleared were raised by this same reviewer. `src/` delta **`prefreeze_resolution.py` only, +46/−5**; `tests/` **`test_pwml_writer.py` only**; `ir.py`, `streamlit_app.py`, `writer.py`, `compound_resolution.py`, node15, the GOLDEN and `chunk_d_gate.py` all **zero lines**. Budgets **498/500**, **49/60**, **151.4 KB / 2 MB**. **THE MECHANISM WAS EARNED, NOT ASSUMED, AND THE REVIEWER VERIFIED IT BY AST:** `_PROVENANCE_FIELDS` is now **exactly** the unconditional block of `db_resolver.apply_compound_db_resolution` `:450-454` — `SET EQUAL: True`, nothing missing in either direction, and **nothing pulled in from the 13 conditional identity writes** after the early return, which is correct because D-015 lets a later pass add identity but not restate provenance. `raw_name` (`:450`) was the one member omitted; the drift is that `PathWhizCompoundResolver.resolve` sets `match[“raw_name”] = row[“name”]` — *the name queried on that pass* — and `:450` prefers it over the row's own. The reviewer also censused every `raw_name` writer (three exist; two are `setdefault`, one is the **species** dedup-follower path) and confirmed no hidden writer the widened set could reach. **The tuple is COMPLETE rather than patched.** **⚠ THE ZERO-DELTA CORPUS IS VACUOUS FOR THE FIX, THE CARD SAID SO UNPROMPTED, AND THE REVIEWER PROVED THE FIX IS GUARDED ANYWAY.** All 32 legs byte-identical — the reviewer made the check *stricter* than the card's, hashing **103/103** persisted artifacts rather than just the pwml. The reason is measured: `_resolve_compound_rows` tests the legacy id **first** and `continue`s (`compound_resolution.py:503-527`), so every corpus rename target, carrying a legacy id, never reaches `apply_compound_db_resolution:450`. **The drifting branch IS reachable in production** — it needs an index-canonicalizable name on a row with **no** PathBank id, i.e. the ordinary “extraction wrote a synonym” case. So the corpus proves **no regression, not validation** — and the reviewer closed the gap by **mutation**: reverting `_PROVENANCE_FIELDS` to the base tuple at runtime, changing nothing else, took 297 passed to **1 failed** with the exact base error `assert 'Glycolic acid' == 'glycolate'`. The production change has a live guard. **A9 INDEPENDENTLY RECOMPUTED, AND IT UNBLOCKS C-051b:** the reviewer recomputed **160 digests (32 legs × 5 configs) at both SHAs — 0 differing**, so **C-051b has no re-baseline to do on account of C-051c**, which is the entire reason this card was sequenced first. **A3/A6 over 96 leg×config measurements and 826 compound rows: `canonical_graph_sha256`, `canonical_payload_sha256` and `admitted_payload_hash` ALL unchanged — no quarantine decision moves.** A4 B-1 holds `0/0/0/0/0`. A5 idempotence holds on the very loop being edited. **A8 G9 has a discriminating control:** base reachable → `'Glycolic acid'` (provenance destroyed), base `db_resolver=None` → `'glycolate'`, tip both → `'glycolate'`, so the drift is the second pass and not the fixture; Part 2 is labelled a **test re-homing** with no fabricated base failure. **A THIRD STALE PREMISE was found hidden behind the first** — the same test also read `name_canonicalization[“compounds”]` off the **IR** report, which moved to the pre-freeze report at C-051 and was masked by the `raw_name` failure two lines above. The reviewer broke **both** re-homings to prove they assert presence at the new location rather than mere relocation. **A7 DISCHARGED, AND THE CEILING-DRIVEN TRIM COST NOTHING A7 REQUIRED.** All 16 raise sites and 8 structural codes byte-identical base vs tip; the diff touches `_PROVENANCE_FIELDS` and prose only — no control flow, no raise added or removed. The card dropped a synthetic `structural` probe section to land at 498/500 and discharged A7 on **real legs** instead; the reviewer ruled real-leg observation **stronger**, since a synthetic section cannot demonstrate reachability on real data, and verified `AMBIGUOUS_RENAME_TARGET` firing identically at both SHAs. **F-2 (INFO, pre-existing, routed):** four codes — `AMBIGUOUS_RENAME_SOURCE`, `PREFREEZE_COMMIT_DIVERGED`, `PREFREEZE_RESOLUTION_UNSTABLE`, `PREFREEZE_ROW_COUNT_CHANGED` — have **no by-name test anywhere**, unchanged at base; the dropped section would have been their first coverage, so it is an **opportunity forgone, not a regression**, and belongs to `prefreeze_resolution.py`'s owner. **F-3 (INFO, refinement in the card's favour):** the card proactively disclosed that `_authoritative_provenance`'s absence semantics now reach `raw_name`, so a row with truthy `db_status` and no `raw_name` has it **popped rather than manufactured** (0 of 282 corpus rows, 0 of 326 incoming). The reviewer found the analogy to `db_match`/`chosen_rule`/`confidence` **imperfect** — `raw_name` is in `_PRIMARY_ALIAS_FIELDS` (`:262`) and feeds `_alias_index`, where the other three feed nothing — but that difference **argues FOR the pop**: retaining a manufactured `raw_name` **adds an alias key the incoming payload never had**, while popping keeps the before/after indices symmetric, and base's retained value would manufacture `raw_name = <canonical name>` on a genuinely renamed row, which is the C-051c defect wearing a different hat. Fail-closed under D-015 clause 7. Focused **709 passed / 3 failed**, down from 5. **DISPATCH RECORD:** base **`b8f7902c`** (the approved C-051a tip), fresh branch `agent/p51c-provenance-and-coverage`, wt `agent-p51c-provenance`, clean at base. Ceilings **500 hand-authored · 60 artifacts · 2 MB**, computed as `0 consumed + 33 (full Chunk D gate self-allocation) + ~15 + headroom` — **the first ceiling on this stack sized by the adopted formula rather than against the diff**. Exists because **Chunk D core stands at 150/152, a BLOCKING gate**, and the two failures were no prior card's to fix. **PART 1 (`REV-051` F-5):** `_PROVENANCE_FIELDS` at `prefreeze_resolution.py:476` is `(“db_status”,“chosen_rule”,“confidence”,“db_match”)` and **omits `raw_name`**, so a second fixed-point pass overwrites the extraction name with the canonical one — measured identical at base and tip by C-051 and re-measured by `REV-051`. It matters beyond the test: `ir.py:433` projects `raw_name` into entity records, so **compounds carry it into the IR**, and losing it makes the frozen payload record the canonical name as though it were what the paper said. **The `_component_record` half of the `raw_name` cluster (`REV-045` F-1, species) is explicitly NOT this card's.** **PART 2 (`REV-051` F-3):** `test_compound_db_resolution_failures_are_non_blocking_for_pwml_build`'s premise is **structurally gone** — `compound_db_resolution_failed` is emitted only by `compound_resolution.py:566` into what is now the **pre-freeze** report, so no input can put it in the IR report — leaving `writer.py :: is_non_blocking_pwml_ir_error` (`:109-115`) with **no producer and no positive test**, only its blocking counterpart covered. `REV-051` confirmed the content is **fully relocated** into `pwml_prefreeze_resolution_report.json` with matching counts, and D-029 keeps the outcome non-fatal, so the property is still true and still worth asserting. **⚠ WIDEST BLAST RADIUS IN THE STACK:** Part 1 edits C-050's approved surface in a loop **every compound row of every payload traverses**, so the card must prove corpus byte-identity or an exact per-file per-key delta (with expected `raw_name` deltas distinguished from unexpected ones), `canonical_graph_sha256` unmoved, **B-1 still `0/0/0/0/0`**, idempotence, rename-map identity, and every structural code plus C-050e's DEF-3 and C-050f's empty-`_norm` sentinel unedited. **Sequenced BEFORE C-051b** because a surviving `raw_name` may move the 32 `GOLDEN` digests, and the golden must move only once more — the card documents that delta precisely as its deliverable and **touches zero digest lines**. `qb` authorized for one run (node06 and node15 must stay green); **core must reach 152/152 or the card names the node and stops** | C-051a ✔ | fresh `agent/p51c-provenance-and-coverage`, wt `agent-p51c-provenance`, base `b8f7902c` | `prefreeze_resolution.py` :: the fixed-point loop's provenance preservation only · `tests/test_pwml_writer.py` :: the two named tests · additive coverage for the tolerated branch | independent, non-author | 500 hand-authored · 60 artifacts · 2 MB |
| C-051b | re-baseline C-040's pre-extraction golden a SECOND time, for C-051's assert-only change | **`ACCEPTED` (in-stack, NOT merged) 2026-08-16** — tip **`164a5a8f`**, `REV-051b` bare **APPROVE**, **0 correction rounds**. **`src/` is COMPLETELY EMPTY in the diff** — zero production change — and `runs/`, `runs_verify/`, `data/`, `out/`, `outputs/`, `tmp/` are all empty; `tests/` is the golden file only; the coverage guard is **unedited** and `GOLDEN` still has **32 keys**, which the reviewer verified equals the git leg set at the pre-image. Budgets **388/500**, **16/30**, **232.9 KB / 2 MB**. Labelled a **deliberate merge-rule-4 baseline move**, with the base-fails/tip-passes shape **expressly not offered as evidence** and no fabricated base proof. **P1 — THE CARD CAUGHT A VACUITY TRAP IN ITS OWN PRE-IMAGE PROOF.** The pre-image is **`328862a`** (C-045b), justified because `git log 328862a..487ac21` is exactly the six C-051* commits and `git diff 328862a 487ac21` over **both** the golden and `c045a_golden_rebaseline.py` is **empty** — so the superseded digests and the tool that regenerates them are genuinely that code's output. But `c045b_base_tree.PATHSPEC` **carries no fixtures**, so a straight export contained **zero `GOLDEN` legs** and the reproduction would have been vacuously perfect. The card wrote `c051b_leg_fixture_export.py`, which **rebinds that shared `PATHSPEC`** to the 32 `final_mapped.json` blobs (discovered by the coverage guard's own rule) **before** the batch exporter's `from … import`, reusing the verified exporter byte-unchanged. **`REV-051b` rebuilt both exports itself** (1767 code blobs + 32 fixtures, both `VERIFIED`), **re-hashed all 32 fixtures a second time independently of the exporter**, confirmed `set(GOLDEN) == set(git leg set)`, and reproduced **32/32** superseded digests. **P2 — ALL 32 DIGESTS MOVE AND EVERY ONE IS ATTRIBUTED.** Four sweeps, **rebuilt from scratch by the reviewer rather than read from the card's JSON**: **D** (raw @ tip) → **160/160** pairs raise `PWML_IR_COMPOUND_VERDICT_MISSING`, which is why A has no successor at the tip; **A→B** routing 3 IR paths; **B→C** the stack's code 1 IR path; **A→C** the whole move **2** IR paths / 4 legs. Every structural number matched; an apparent 84/21-vs-54/12 discrepancy resolved to a counting convention (distinct `(leg, config)` pairs), identical to the pair. **P3 — THE TWO LOAD-BEARING CLAIMS, BOTH CONFIRMED IN THE STRONG FORM.** (i) `A→C` moves exactly `entities/compounds[]/synonyms` and its length, **12 pairs / 4 legs** — the original supported name preserved as a synonym, which **D-015 clause 5 requires** and the raw standalone configuration never exercised — and it **appears in A→B, i.e. at pre-C-051 code, so NO code in this stack produces it**. Prose correction: at least four distinct compounds gain one, not only `α-ketoglutarate`. (ii) **`db_status` is a genuine exact round trip, not two populations cancelling in aggregate** — the **same 54 pairs across the same 18 legs**, `matched_offline_name_index → legacy_id_unverified` in A→B and the **exact inverse** in B→C, 84 leaves each way, **absent from A→C**. The exporter's post-freeze second pass had been overwriting the pre-freeze verdict and deleting it restores it — the same degradation `REV-051` measured from the other side, so it **corroborates C-051 rather than contradicting it**. Nothing else under `ir` moves on any leg or config. **THE FOUR STRUCTURAL STOPS ARE DOUBLE-PINNED, AND THE REVIEWER BROKE BOTH PINS TO CHECK:** `GOLDEN_PREFREEZE_STOPS` pins them **by code** *and* the code is hashed into the digest, so a substituted code fails at `:570` and a stop that **stops raising** fails loudly via `UnresolvedCompoundRowError` propagating — `_leg_digest` catches **only** `PrefreezeResolutionError`. **P4-02 CLOSED AGAIN:** all three tool modes run, and `--mode digest` re-derives **32/32 digests and the exact stops table from the repo alone**. **⚠ THE REVIEWER FOUND A FOURTH, CANCELLING IR PATH NEITHER THE CARD NOR THE ORCHESTRATOR HAD** — and only by rebuilding the sweeps instead of reading the summary. Sweep **A** shares one payload across the five configs (old `_leg_digest` semantics) while **B**/**C** deep-copy per config (new semantics, forced by the pre-freeze in-place rewrite), so **A→B is routing PLUS per-config isolation**. Isolated, the isolation component alone moves `/0/entities/compounds/[]/db_match/reason` on **23 pairs / 23 legs** (config E, all 32 legs) and the routing component moves the **exact inverse** — cancelling, which is why the three-bucket view shows three paths. **No committed digest is affected**, since A→C defines every one of them and contains only the two `synonyms` paths, so this is an **attribution-label gap, not a hidden move** (**F-1, LOW**). The isolation change is itself a **correction**: the old sharing measured config E against a payload the other four had polluted. **DISPATCH RECORD:** base **`487ac21c`** (the approved C-051c tip), fresh branch `agent/p51b-golden-rebaseline-2`, wt `agent-p51b-golden2`, clean at base. Ceilings **500 hand-authored · 30 artifacts · 2 MB**, computed by the adopted formula at the moment the work was ordered. **⚠ SCOPE IS LARGER THAN A DIGEST RE-BASELINE, per `REV-051c` § 7.** The golden no longer fails by digest drift — **it fails by REFUSAL**, because `build_pwml_ir` now raises `UnresolvedCompoundRowError` on the raw payloads it is fed, so no digest can even be computed. **Two of the three modes of C-045a's re-baseline tool are dead, not one:** `--mode digest` at `:161` and `--mode flatten` **without** `--production` at `:105`, both via `build_pwml_ir` on a raw payload. **`--mode flatten --production` (`:94-102`) already implements exactly what is needed** and is the card's template: it deep-copies, runs `run_prefreeze_resolution` **per config with that config's own `strict_db`/`db_resolver`/`name_index`**, and records a D-015 clause 6 stop as a **result** rather than a crash. **⚠ FOUR of 160 leg×config combinations legitimately RAISE `AMBIGUOUS_RENAME_TARGET` identically at both SHAs** — `PMC12444477…/strict` under configs A/C/D (two names canonicalizing to `glycerol 3 phosphate`) and `PMC13278307…/strict` under C (to `glycine`) — **pre-existing and correct**, a structural stop being the system working; they are **not digests and must not be forced into being digests**, and the card must record them so a future change that silently stops raising is caught. Also required: repair both dead modes, since leaving them dead permanently re-opens the **P4-02** debt C-045a closed and C-051 unintentionally re-opened. **G9 is deliberately NOT the usual shape** — a merge-rule-4 baseline move, with the base-fails/tip-passes shape **expressly not evidence** and no fabricated base proof; the obligation is discharged by P1 (pre-image reproduction, **S9 trap 3**), P2 (per-leg delta **attributed to C-051's refusal**) and P3 (the explicit negative). **Standing instruction: if any leg's digest moves for a reason that cannot be attributed to C-051's refusal, STOP AND DO NOT COMMIT** — that is a regression hiding behind a re-baseline and it outranks the card. **Chunk D and `qb` are NOT authorized and NOT required**: the orchestrator verified `test_compound_resolution_extraction.py` is in **none** of `CORE`/`S8`/`QB` and the card touches **no production file**, so neither collection nor behaviour can move — and if the card finds itself needing a production file, that assumption breaks and it stops. **ROUTING RECORD:** **lands LAST**, on the approved C-051a tip, so the 32 digests reflect the **final combined state** and move only once more. C-045a already moved this golden once for D-016 and **predicted this second move in its own deferred finding**; the test file's docstring naming C-051 as the intended mover predates C-045a and is stale. Deliberately **not** folded into C-051's ~70-node test migration: a golden re-baseline needs its own focused review where the reviewer looks at nothing else — that is the review shape that caught the **D-032** regression, and 32 opaque digests would be buried inside a migration. Will reuse C-045a's committed `c045a_golden_rebaseline.py`, which re-derives all 32 from the repo alone | C-051a (approved) | *(fresh branch at dispatch)* | **C-040's ownership.** `tests/test_compound_resolution_extraction.py` :: the `GOLDEN` digests only. Same P1-P5 proof burden as C-045a, including **pre-image pinning (S9 trap 3)** and the explicit negative that no path under `ir` moves | independent, non-author | set at dispatch |
| C-051d | correct two measurably false committed sentences (**`REV-051b` F-1 / F-2**) | **`ACCEPTED` (in-stack, NOT merged) 2026-08-16** — tip **`9cc40286`**, `REV-051d` bare **APPROVE**, **0 correction rounds**. **THE ELEVENTH AND FINAL CARD; THE STACK IS COMPLETE.** `src/` **empty in the diff**; five files total (the golden test +17/−8, `c051b_delta_attribution.py` +14/−2, three G11 reports). Budgets **41/60**, **3/8**, **7.7 KB / 500 KB**. Verified by AST base-vs-tip: `GOLDEN` unchanged (**32 keys**), `GOLDEN_PREFREEZE_STOPS` unchanged (2 keys), `LEAF_HELPERS` unchanged (9), `_leg_digest`, `_configs` and **both** golden tests byte-identical by source, 13 test functions both sides, and `c051b_golden_move_attribution.json` untouched. `GOLDEN` still equals the git leg set at the pre-image (**32 = 32, SET EQUAL**). **The load-bearing sentence survived byte-for-byte** — *“It must never be moved to make an accidental drift go green”* — wording unaltered, one occurrence, only its line-wrapping moved inside the reflowed paragraph. **A4 re-run by the reviewer, not assumed: `--mode digest` still re-derives 32/32 digests and the exact stops table**, golden file 14 passed — so a documentation card did not disturb the **P4-02** guarantee C-051b closed. **`REV-051d`'s Q5 ruling went FURTHER than the card asked, and it is worth keeping:** the card declined to rename the emitted JSON key `step_A_to_B_routing_only_at_base_code` because renaming would desynchronize C-051b's committed artifact; the reviewer ruled renaming would be **actively worse** — the key is **functional output**, the identifier a re-derivation reads by, so renaming it without regenerating the artifact converts a naming imprecision into a **reproducibility break, precisely the P4-02 class this stack just spent a card closing**. Strictly negative trade; if ever renamed, the owner is whoever reruns the sweeps, not a documentation card. It also endorsed the card's **2b** choice of doing *both* — naming the bucket **and** recording the cancelling pair — because *“the key was never the misleading part; the count was”*: a reader who sees `ir_paths_changed_count: 3` and stops is the failure mode, and both records now name the fourth path, size it (23 pairs), scope it (config E only) and state that routing moves it back exactly, which is enough to reproduce the isolation control without the reviewer. **DISPATCH RECORD:** base **`164a5a8f`** (the approved C-051b tip), fresh branch `agent/p51d-stale-docs`, wt `agent-p51d-docs`, clean at base. Ceilings **60 hand-authored · 8 artifacts · 500 KB**, computed by the adopted formula. **Changes no behaviour: `src/` must be EMPTY in the diff**, and the 32 `GOLDEN` digests, `GOLDEN_PREFREEZE_STOPS`, `_leg_digest` and the coverage guard are all **zero lines**. **F-2:** the module docstring at `tests/test_compound_resolution_extraction.py:16-18` says the pin is *“expected to be moved — once, deliberately … by C-051”* and that the digests derive from a sweep at `e4eeef42` *“before the extraction”*. **Both false at this tip** — it has moved **twice** (C-045a for D-016, C-051b for C-051's refusal, exactly as C-045a's own deferred finding predicted), and the digests now derive from a **pre-freeze-routed** sweep at the stack tip. The final sentence — *“It must never be moved to make an accidental drift go green”* — is correct, load-bearing, and must survive **verbatim**. **F-1:** `c051b_delta_attribution.py:24` labels A→B *“the routing alone”*; measured, it is **routing plus per-config isolation**, with a fourth IR path that cancels exactly. **Why an eleventh card for two documentation lines:** this sprint has repeatedly found that stale control-plane text is what misleads the next reader, and **this exact file is where a stale comment already misled one** — C-045a's `#:` block, which C-045b had to correct. Shipping a knowingly false sentence after a reviewer flagged it would be inconsistent with how every other stale comment in this sprint was handled. **A4 requires `--mode digest` to still re-derive all 32 digests and the exact stops**, so the P4-02 guarantee is not disturbed by a docs card. **No G9 and none may be invented** — a documentation correction is neither a behaviour change nor a new capability. Chunk D, `qb`, SMOKE and benchmarks neither authorized nor required; **if the diff touches a production file that assumption breaks and the card stops** | C-051b ✔ | fresh `agent/p51d-stale-docs`, wt `agent-p51d-docs`, base `164a5a8f` | the golden file's **module docstring** and the A→B label · `c051b_delta_attribution.py`'s **bucket label** — documentation only | independent, non-author | 60 hand-authored · 8 artifacts · 500 KB |

### ✅ THE STACK IS COMPLETE — eleven cards accepted, composite landing in progress (2026-08-16)

```
integration 6c98508
  └─ 0f859d9  C-050 sync merge (the ONLY merge in the chain)
       ├─ edf8a0d1  C-050e  ✔      ├─ 328862ab  C-045b  ✔
       ├─ a81b1d65  C-050d  ✔      ├─ 2b3de80a  C-051   ✔  <- B-1 DISCHARGED
       ├─ 0ec64d2c  C-050f  ✔      ├─ b8f7902c  C-051a  ✔
       ├─ e2b336c3  C-045   ✔      ├─ 487ac21c  C-051c  ✔  <- Chunk D core 152/152
       ├─ d146be48  C-045a  ✔      ├─ 164a5a8f  C-051b  ✔
       └─                          └─ 9cc40286  C-051d  ✔  <- TOP TIP
```

**20 commits, exactly one merge.** Production surface **4 files, +1741 / −18**: `prefreeze_resolution.py` +1378 (new module), `ir.py` +199, `streamlit_app.py` +106, `writer.py` +76.

**The composite merge is PRE-VERIFIED CONFLICT-FREE.** `merge-base` = `6c98508`; since then integration touched only `DECISIONS.md`/`FINDINGS.md`/`LEDGER.md` and **other** cards' `evidence/g11/` dirs, while the stack touched `src/`, `tests/` and its own probes. **The intersection of the two changed-file sets is EMPTY.** Provenance spot-checks already done so the composite reviewer need not rediscover them: `tests/test_streamlit_quarantine_boundary.py` was changed by **C-050d alone** (the card chartered for node15, C-050a's function) and `evidence/chunk_d_gate.py` by **C-045b alone** (the authorized `core 150 → 152` move; `TOTAL` is derived).

**Landing gates dispatched to an independent test operator on the top tip**, which is the prospective combined state because `git diff 6c98508 sprint/pwml-recovery -- src/ tests/` is empty: the full Chunk D cohort at **179** with `T2PW_OFFLINE_CURATOR=1`, **SMOKE 460** (its **first** execution — D-030 reserved it for the landing and no card ran it), and whole-tree G11. Full plan, including the amendments this stack produced and the open items that do **not** block, is in `LANDING-PLAN.md` in the session scratchpad.

**⚠ THE CHUNK D PARTITION IS NOW `core 152 + s8 4 + qb 23 = 179`, NOT 177.** C-045b added two tests
to `tests/test_pwml_writer.py` (a `CORE` file) under a charter that mandated them, and the
orchestrator authorized the pinned `chunk_d_gate.py:70` count `150 → 152` as a merge-rule-4 baseline
move; `TOTAL` at `:74` is derived and follows to 179. The gate's set-based proof passed independently
at the new counts (`SETS_EQUAL=True`, `overlap=0`, `missing=0`, `extra=0`). **The composite landing
must expect 179.**

**Landing gates, run ONCE on the top C-051 tip** (not per card): one complete 23-node deterministic
`qb` cohort with `T2PW_OFFLINE_CURATOR=1` in the bounded child, all 23 classified · SMOKE 460 ·
whole-tree G11 zero non-compliant · protected manifest 42/42 · zero surviving owned processes ·
composite reviewer's exact bare `APPROVE`. **node10 is expected GREEN** — its earlier failure was
the `OPDA → Dinor-12-oxo-phytodienoate` rename that D-028/C-040a now refuses; **node10 is not to be
changed to obtain that result.**

### ✅ THE STACK LANDED — `beddcdd6`, twelve cards, one `--no-ff` merge (2026-08-16)

The composite landing above **completed**. C-050g (`47e608c6`) joined the eleven and the whole stack merged
as a single `--no-ff` commit **`beddcdd634ba645ab67362bf3dca779c4ab5f67d`**, second parent **`47e608c6`**
(verified). Nothing merged before it.

**Accepted gates at the merged tip:** C-050 five-category probe `0/0/0/0/0`, `C-050 ACCEPTANCE: PASSED` ·
post-freeze compound identity mutations **13 at integration base, 0 at merged tip** on a real leg ·
Chunk D **179** (core 152 / s8 4 / qb 23) · **SMOKE 460 exact** · whole-tree G11 zero non-compliant ·
protected manifest 42/42 · zero surviving owned processes.

**B-1 is discharged. Merge rule 8 holds.** Do not re-review the twelve cards and do not rerun the 32-leg
base-versus-tip comparison; both are accepted measurements.

**The ratified cost is D-034.** Exactly one committed leg of thirty-two no longer exports —
`runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict`. It exported at
base only because the exporter silently merged `lipid IV_A` with `lipid IV A` **after** the freeze. The
fail-closed result stands until the duplicate-row policy card lands. **Do not restore the post-freeze merge
and do not weaken the guard to recover the leg.**

**All twelve card branches and the eight pre-stack branches are preserved** — not to be reset, deleted,
pruned, reused or rewritten.

## Post-stack cohort — opened 2026-08-16

Takeover verified at `beddcdd6`: local = `origin` = `ls-remote`, no merge in progress, empty index,
protected manifest 42/42, G11 zero non-compliant across the primary checkout and every registered worktree,
zero abandoned `g11_reserved` placeholders in the repository, zero surviving owned processes.

| ID | Task | Status | Deps | Notes |
|---|---|---|---|---|
| C-050h | duplicate canonical rows — **refusal path only** (discharges **D-034 clause 4** and the remaining half of **F-8**) | **`ACCEPTED`, merged `75f8bba`**, reviewed tip **`f804f61`** (verified second parent), **REV-050h bare `APPROVE`** after **1 correction round**. Adds **`PREFREEZE_DUPLICATE_CANONICAL_ROWS`**. **Post-merge gates on the merged state:** SMOKE **460 passed in 39.60 s** exact · whole-tree G11 **1936 / 0 non-compliant**, exit 0 · protected manifest **42/42** · **zero surviving owned processes** · every one of the card's 37 artifacts `final_surviving_count` 0 / `cleanup_success` true. **Golden movement, measured per configuration (160 cells, not the opaque digest): exactly 3 differ**, all `PMC12444477…/strict` under A/C/D, `PREFREEZE_CONNECTIVITY_BROKEN` → the new code; **nothing moved STOP→export or export→STOP**; `PMC13278307…/strict · C_canned` unchanged at `AMBIGUOUS_RENAME_TARGET` per **D-035 clause 7**. **No consolidation path exists — proved by AST walk**, not by test coverage. Final budgets **573/580 hand-authored, 53/56 artifacts, 145,486 B**. Superseded status: `CORRECTION` round 1 of 2 2026-08-16 — **REV-050h did NOT approve.** No BLOCKING issue: no boundary breach, no weakened gate, no post-freeze repair, no consolidation path (proved by **AST walk** — every assignment target a bare local, zero `Subscript`/`Attribute` stores, `_Ref.set` never called), no undocumented golden movement (**160-cell per-config matrix: exactly 3 differing cells**, all the D-034 leg, all code-only; the reviewer refused the opaque digest and measured per config). **The findings are about the record, not the code.** MAJOR `prefreeze_resolution.py:842`: the docstring says the check *"does not fire on a collision the payload already carried"*, but the test at `:875` is **per-row** (`arrived`), so a pre-existing collision renamed **en bloc** onto a new key **does** fire — three synthetic shapes measured EXPORTED at base / REFUSED at tip. The behaviour is correct under D-035 clause 6 and stays; **D-034 clause 2's “known, measured, attributable” standard is what the false docstring breaks.** MAJOR: the test header's unqualified *“nothing here newly refuses a payload that exported at base”* is true only over the committed corpus. Two MINORs. **Ceilings 580 / 56 ratified.** The reviewer also **discharged F-042 itself**, proving the leg's `final_mapped.json` byte-identical at both SHAs and importing it into the base tree to produce the real behavioural base failure the card could not. Superseded status: `REVIEW` tip **`999209e`**, parent `b5fb82a`, single commit, routed to **REV-050h**. Adds **`PREFREEZE_DUPLICATE_CANONICAL_ROWS`**; trigger is ≥ 2 compound rows sharing one `_norm(after)` key where ≥ 1 did not already hold it **and a reference lands on it**. Orchestrator-verified: budget **520/520**, diff is 3 files, and `prefreeze_resolution.py` has **zero deleted lines**, which *proves* `_reject_ambiguous_renames`, `_norm` and `_canonical` untouched rather than asserting it. **No identifier comparison decides anything** — identifiers are diagnostic payload only, so **F-043**'s PathBank-193 trap is unreachable by construction. Golden moves for **one leg, three configs, code only** (`PMC12444477…/strict` under A/C/D, `PREFREEZE_CONNECTIVITY_BROKEN` → the new code); `PMC13278307…/strict` under `C_canned` **unchanged**; 31 other legs identical.

**Scope narrowing ACCEPTED by the orchestrator 2026-08-16.** The card first refused **unreferenced** created duplicates too, measured that this **newly refused two parametrizations** of `test_prefreeze_third_export_seam.py::test_db_reachability_reaches_the_exporter_in_both_directions` — another card's acceptance fixture, subject DB reachability, not duplicate rows — and **withdrew it**, drawing the line at ambiguous connectivity. **Accepted because D-035 clause 6 names *ambiguous connectivity*, and an unreferenced duplicate creates none.** The residue — an unreferenced created duplicate exports and is then coalesced first-wins by `ir._dedupe_named_rows` — is **F-039**, owned by **C-050i**, which **D-036**'s third option explicitly declined to fold in here. The residue is named in the suite rather than left silent. **REV-050h is asked to attack whether the narrowing leaves a hole neither card owns**; if it does, that is a finding. Superseded status: `RE-CHARTERED` under **D-036**: census complete, **zero committed groups clear D-035 clause 3**, so the card builds **clause 6 only** — a named, machine-readable fail-closed reason replacing the opaque diff-string diagnosis. **The consolidation engine (clauses 2–5) is deferred, not cancelled**; D-035 is unamended and reopening is triggered by a payload measured to clear clause 3. Awaiting the heavy-job slot. Superseded status: Product ruling recorded as **D-035** *before* dispatch, so the card does not stop to ask "merge or refuse". A read-only census agent is measuring committed duplicate-canonical groups against D-035 clause 3 **before** any implementation charter is written | D-034 ✔ · D-035 ✔ | Base `beddcdd6`. **A measured finding that no committed group clears the D-035 clause 3 bar is a valid outcome** — the D-034 leg is then correctly still refusing. Evidence must not be stretched to recover it. `_reject_ambiguous_renames` is structurally blind here (D-034 clause 5) and the card must supply its own detection |
| H-010 | measurement-harness source pin — the `.pth` / CWD hole | **`ACCEPTED`, merged `16cd3bd`**, reviewed tip **`5126197`** (verified second parent), **REV-H010 bare `APPROVE`** after **1 correction round**. **Post-merge gates on the merged state:** SMOKE **460 passed in 29.12 s** exact · whole-tree G11 **1898 / 0 non-compliant**, `MERGE-010` 2/0, exit 0 · protected manifest **42/42** · **zero surviving owned processes** (4 `python.exe` live, all VSCode isort LSP, none sprint-owned) · every artifact `final_surviving_count` 0 / `cleanup_success` true. **Disclosed deviation:** the two post-merge smoke runs added `-p no:cacheprovider` to the documented command form to avoid writing `.pytest_cache`; selection and counts are unchanged and both runs reported exactly 460. Superseded detail: **`RE-REVIEW`** 2026-08-16 — correction round 1 delivered at tip **`5126197`**, parent `93e8f3f`; boundary and chain orchestrator-verified, still exactly the 6 granted files. All four findings discharged, the BLOCKING one proved on **real trees** (expecting the primary while `PYTHONPATH` names a genuinely nested worktree → exit 98, with `scripts` and `cwd` staying clean so the refusal isolates the defect). Card found and self-disclosed a **fifth** defect (**F-045**): `--pin-verdict` resolved after the `chdir`, so a relative path wrote evidence into the **audited** checkout — impact contained, primary re-verified clean at G11 **1870/0**. **Ceilings ratified: hand-authored 880 → 907** (the fifth defect was out of scope when 880 was set) and **artifacts 32 → 40, with the Chunk D re-run skip RATIFIED** — 25 + 18 = 43 made the old ceiling internally inconsistent with its own mandated gates, the D-025 failure mode by name, and the card correctly stopped rather than self-authorized. Superseded detail: **`CORRECTION` round 1 of 2** 2026-08-16 — **REV-H010 did NOT approve**, one BLOCKING finding. `tree_pin.py:163`/`:174` accept any `t2pw` satisfying `is_relative_to(expected)`, i.e. **anything nested inside the expected tree** — and every agent worktree in this sprint lives at `<primary>/.claude/worktrees/`. Measured: a run declaring the primary with a stale worktree `PYTHONPATH` returns **exit 0, `1 passed`, `violations: []`**, and the committed `.pin.json` **certifies the wrong tree as clean**. The card's own suite missed it because all five arms mispoint to *siblings*, never to a nested decoy. Plus MAJOR `pinned_pytest.py:77` (`--expect-tree` unvalidated — an ancestor path yields `violations: []`) and two MINORs. **Ceiling raised 794 → 880**, ratified: these are reviewer-mandated corrections to a measurement-integrity defect. **What the review confirmed:** boundary clean, budget reproduced, G9 split correctly labelled per component, `_LEGACY` verified statement-for-statement against the base file (not a strawman), an independent mutation run (guard → `pass`) killed all five arms, SMOKE **460**, Chunk D **179/152/4/23** with the C-045b pin unmoved, and all three declared charter contradictions judged SOUND. Superseded detail: tip **`93e8f3f4`**, direct parent `57295da` (the assigned base), branch `agent/h010-measurement-tree-pin`, worktree `.claude/worktrees/agent-h010-tree-pin`. Boundary clean: exactly the 6 granted hand-authored files, no forbidden file touched. Orchestrator-verified G11 at the tip: **1892/0** whole-worktree, **22/0** for `H-010`, exit 0. Reported gates: SMOKE **460**, Chunk D `--only core` **152** (C-045b baseline unmoved), `--only s8` ok, all 25 artifacts `final_surviving_count` 0 / `cleanup_success` true. **Budget 794 hand-authored RATIFIED** — the 620 ceiling was a chartering error computed from a pre-implementation estimate (**REV-051a**: *the defect is in applying D-025, not in D-025*); the only lever to reach it was dropping adversarial arms A2(b) and A4(c), and A4(c) is the arm that excludes a naive PYTHONPATH-comparing guard. Precedent **C-050f** 380→430. **794 is the figure; further growth needs a fresh decision.** Artifacts 25/32, 77,288 B/150 KB, largest 6,596 B. Routed to **REV-H010**, which must mispoint the environment itself rather than trust the card's own adversarial suite | — | Base `beddcdd6`. Two halves of one defect: `__editable__.t2pw-0.1.0.pth` hard-codes the **primary** checkout's `src`, so a worktree run without a source pin can silently measure **base** code and pass; and `c045_pinned_pytest.py` sets `sys.path[0]` to its own directory where `python -m pytest` sets the CWD (C-050g's `ModuleNotFoundError: scripts`). **`bounded_run.py` is protected** — a companion launcher or preflight is preferred to modifying it. Target invariant: every official measured command **fails fast and visibly** unless the loaded `t2pw` and `scripts` come from the exact tree under measurement |
| C-052 | pre-freeze report at the Streamlit seams (carries **A0-C8** only — see the ownership correction below) | **RECONCILED 2026-08-18 (Git ancestry):** **C-052 is `ACCEPTED` and MERGED `c0df0d0`, reviewed tip `c7250d3`.** **`RULINGS COMMITTED`** 2026-08-17 — ten rulings issued as **D-040**. **Reconciled with the authoritative C-052 row above** (two rows carried different statuses and only one carried a boundary; same self-contradiction class struck for C-050i in `8f7514f`). **Not buildable as chartered** — three blockers, all fixed by D-040 | D-029 ✔ · D-032 ✔ · D-033 ✔ · **D-040 ✔** | **Base = the tip at dispatch, recorded here at dispatch, never in the charter** — `beddcdd6` and the charter's `57295da6` are both stale, and every charter cite into `prefreeze_resolution.py` is off by ~126 lines after C-050h. Both Streamlit seams discard the pre-freeze report while the CLI (C-045b / C-051a) persists it. **The CLI is the reference — converge on its *shape*, never on its *path*: `writer.py:2690`+`:2817` make `outputs/pwml_prefreeze_resolution_report.json` the CLI's own default, so EP3 writes `…report.streamlit.json` (D-040 §2).** D-033 binds. **The `st.form` hazard is NOT a constraint here** — all three C-052 surfaces make zero `st.` calls, so the MagicMock-stub convention suffices and no AppTest is needed |

**C-050i correction round 1, ruled 2026-08-17. R1 is NARROWED; two grants issued.** The card reached
`c5cea52` (parent `8f7514f`), **hit R1's stop condition, and correctly stopped and reported** rather than
weakening the guard or narrowing scope itself. That was the right call and the ruling is the orchestrator's.

* **R1 NARROWED — the guard binds the entity call site (`ir.py:1049`) only; the component call site
  (`ir.py:957`) keeps its pre-existing warning.** R1 as issued was wrong for the component buckets.
  `prefreeze_resolution._canonicalize_species_rows` (`:1180-1197`) **deliberately converges a `_norm` group
  onto its leader because the exporter dedupe collapses it**, and its docstring states that a row which
  *stops* being a duplicate becomes *"a second species in the IR that the exporter never emitted — that is
  inventing biology"*. Converged rows share a `taxonomy_id`: **proven identity**, which D-035 permits, versus
  F-039's compound pair which is **coincident spelling with conflicting identifiers** (PathBank 40738 vs
  40982). Refusing in the component bucket would break C-045/D-016's accepted acceptance criteria
  (`tests/test_prefreeze_species_resolution.py:191`, `:312`) and cause the very harm class the guard exists
  to prevent. **This is not a gate weakened for green:** it confines the guard to the class where harm was
  **measured**, and the residual is named rather than silently absorbed — **F-046**, owned by a new card
  **C-050j**, *not* C-050i. The two species tests go back to green **untouched**.
* **Merge-rule-4 golden move GRANTED**, with the exact delta the card measured: over 32 legs × 5 configs,
  **one leg, two configs** (`…PMC12444477…/strict` under `B_dbdown_noindex_strict` and
  `E_fromenv_raises_emptyindex_lenient`) now raise `PWML_IR_DUPLICATE_NAMED_ROW` instead of producing a
  digest; configs A/C/D on that leg are unchanged (`PREFREEZE_DUPLICATE_CANONICAL_ROWS`) and the other 31
  legs are untouched. This is an **entity-bucket** collision, so it survives the narrowing. Record it in
  `tests/test_compound_resolution_extraction.py` as a **`GOLDEN_IR_REFUSALS`** table in the existing
  `GOLDEN_PREFREEZE_STOPS` idiom (`:504`, asserted `:636`), with a **distinct** digest marker so an IR
  refusal is never conflated with a pre-freeze stop. **Prefer reusing `_leg_digest`'s existing
  `(digest, stops)` return shape — no arity change**, so nothing ripples into
  `evidence/c045a_golden_rebaseline.py:173`.
* **Ceiling 1 raised 900 → 1,050, ratified not charged.** The card landed **919** and reported rather than
  cutting an arm — the required behaviour. **REV-051a governs: the defect is in applying D-025, and the
  mis-set ceiling was the orchestrator's**, as it has been every time. The correction round adds test lines
  for the narrowed bucket contract and the golden table. Ceilings 2 (55/72) and 3 (4,496/20,000) stand.
* **Correction 1 delivered at `6605066` (parent `c5cea52`, chain root `8f7514f`). Orchestrator-verified,
  not accepted on report.** Boundary is **exactly four files** — `ir.py` (**the only production file**),
  the new acceptance suite, the golden file, and the evidence probe; `tests/test_prefreeze_species_resolution.py`
  is **absent from the diff**, i.e. untouched, and green. The component branch is byte-identical in behaviour
  to base (same `"warning"` severity, same `duplicate_named_record` code, same message text, same
  `pointer=f"{pointer_prefix}/{idx}"`, same `idx += 1; continue`), and **`refuse_duplicates` defaults to
  `False`** — i.e. the default **preserves base behaviour**. *(Wording corrected 2026-08-17: this cell first
  called `False` "the safe direction", which conflates two senses. It is safe against **regression** and
  permissive against **the defect**; a fail-closed default would be `True` with an explicit
  `refuse_duplicates=False` at the component site. REV-050i flagged the conflation and confirmed the
  in-diff docstring makes no such claim and is accurate. Left as built — the risk is theoretical with two
  call sites in one private helper — but the record is corrected rather than left overstating.)*
  **`refuse_duplicates=True` appears exactly once**, at
  `ir.py:1197`, on the **entity** call site that populates `entity_by_name`; the component call site does not
  pass it. Gates on the corrected tip: affected suites **88/88** (was 3 failed / 85 passed), focused **22**,
  Chunk D **179/179**, SMOKE **460 exact**, G11 **exit 0** (2019 artifacts, 0 non-compliant), **zero
  surviving owned processes across all 83 jobs**.
* **BOTH D-025 ceilings ratified, not charged to the card.** Ceiling 1 **1,159 / 1,050** and ceiling 2
  **98 / 72**. Ceiling 2's overage is **structural and mine**: 64 of the 98 are **two full Chunk D runs**
  (the gate self-allocates ~32 each), and the second run exists **because I mandated a re-measure on the
  corrected tip** — without it the count is 66, under the ceiling. Ceiling 1 grew by the +240 lines the
  correction itself required (the `refuse_duplicates` split and its rationale, the golden table, the
  component-warn arm). **The card deleted no evidence and cut no arm to fit a number, and refused to do so
  when it would have brought the count under** — that is the required behaviour (D-025), and **REV-051a
  governs: the defect is in applying D-025, and the mis-set ceilings were the orchestrator's, as they have
  been every time.** Ceiling 3 **8,140 / 20,000** ✓.
* **Accepted as measured, not re-run:** §5's live residual is **MEASURED, not DB-unavailable** —
  `residual_count: 0`, `created_collisions: {}` over 32 legs through EP3's second pre-freeze pass with the
  live resolver, run from the primary checkout, read-only, output resolved before `chdir`. A zero result
  does not invalidate the guard (charter §5).

**C-053 ACCEPTED — merged `3fde1f1`, reviewed tip `09f8371`, independent bare `APPROVE` after one correction
round.** Post-merge gates **on the merged state**: SMOKE **460 exact** · whole-tree G11 **exit 0**, 2073
artifacts, 0 non-compliant · protected manifest **42/42** · **zero owned surviving processes** · no cache,
`runs/`, `outputs/`, `data/`, `tmp/`, `writer.py` or `streamlit_app.py` touched. Hand-authored **1193 /
1200** by the corrected **F-050** command.

The re-review **reproduced rather than accepted**: all six return dispositions probed directly against the
tip module, the new pin killed under **three** separate mutations with a hash-verified revert, the G9 base
failure independently reproduced at `8920371`, the rest of `driver.py` proven **byte-identical outside the
granted symbol**, and **319 tests run by name** across the files belonging to no chunk. The `_add_strict_artifacts`
return is now correct and pinned on all six dispositions — the dimension that previously **could not fail**
now fails three ways. The G9 labels are measured (`6 of 8`), and the new return pin is **deliberately not
offered as a base proof**, because at base the function had no `return` at all and G9 holds that symbol
absence is not proof.

**Follow-ups recorded, none blocking:** `_finalize_gate_failure`'s docstring at `driver.py:1755-1757` is now
stale (it says promoting the record into the row "belongs to C-053", which C-053 did) — correctly left
untouched as out of boundary; owner is whichever card next holds that function. The row's `release_status`
carries **15 keys on the PASS path** (the seam record verbatim, per D-038 §2) and **12 on the gate-fail
path**; both carry `status` and `strict_acceptance_eligible`, so `describe()` and the denominator gate are
safe — **flagged for C-056b, which reads these rows**. Historical `runs/` manifests carry no `release_status`
and therefore now score **0 strict successes by design**; no test pins it, so **do not mis-read a future
benchmark as a regression**.

**REV-053 REJECTED `57be026`, 2026-08-17 — two blocking findings, correction round 1 of 2 dispatched.**
The reviewer verified the card's substance independently and found it sound (see the accepted list below),
then found two things the card's own suite structurally could not.

* **BLOCKER 1 — `driver.py:1499-1530`: `_add_strict_artifacts` returns the wrong string on every path.**
  The signature became `-> str` documenting *"the PWML filename"*, but the loop at `:1517` **rebinds `name`**
  over five unconditional iterations, so `return name` yields
  `'pwml_required_field_gate_report.json'` on **all five** dispositions. Probed directly against the tip
  module: naming correct on all five, return wrong on all five. **No live impact** — `_drive` discards it and
  `_finalize_pwml_export` re-derives — but it is **the one dimension of the new seam with no test that can
  fail**, which is why nine mutations and 212 focused tests missed it, and it ships a **measurably false
  documented contract** on §3 hotspot 2 with **C-054 and C-056b queued directly behind on the same seam**.
  Fix: rename the loop target, and **pin the return on every disposition** so the arm can fail.
* **BLOCKER 2 — `tests/test_batch_pwml_artifact_naming.py:13-14` over-claims the G9 correction arm.** It
  states *"every one of them fails at `8920371`"*; measured, **6 of 8** fail — two are **preservation** arms
  that correctly pass at base. **G9's entire content is the accuracy of these labels.** The card's report to
  the orchestrator was accurate (10/2); the **shipped file** is not, and the shipped file is what a future
  auditor re-derives from. Over-claiming rather than dodging, so a correction, not a reject. **Prose only.**
  The NEW ACCEPTANCE label is correct and stands.

**Accepted from the review, not to be re-litigated or regenerated:** boundary PASS with **`_drive` proven
byte-identical** (491 lines at both SHAs) and every tripwire clean · **the G9 base failure is REAL**,
reproduced independently at `8920371` with only the new test file copied in (**10 failed / 2 passed**) ·
**nine single-point mutations all killed**, the golden alone killing three · provenance PASS — `git grep`
finds `artifacts["quarantine_report` only inside a docstring explaining why that source is wrong · no
strict-success inflation (affirmative eligibility gate; the four struck keys not invented) · merge rule 7
PASS (bytes kept; warnings never flip a status; `diagnostic_only` still writes all four JSON artifacts) ·
hotspot 10 growth-only with a **derived** delta, `release_status_absent` **replaced by value** so presence
became the invariant · the merge-rule-4 re-pin judged **stronger** than the original ·
`test_batch_preflight.py`'s two failures re-confirmed **pre-existing and not this card's**.

**Housekeeping owed:** the reviewer's base worktree `C:/t/rv53/basewt` is registered, detached, and clean;
its removal was **permission-denied**. Leave it registered — it blocks nothing.

**C-053 round 0 rulings, 2026-08-17. Ceiling ratified, two test files granted, one instrument corrected.**
The card **stopped before committing** on a ceiling-1 overage and staged the complete patch rather than
trimming it. That is the required behaviour (D-025), and because no commit exists the ceiling can still be
raised cleanly. Orchestrator-verified in the worktree: `HEAD` still at base `8920371`, **zero** commits, no
forbidden path staged (`pwml/writer.py`, `streamlit_app.py`, `outputs/`, `batch/report.py`,
`bench/render.py`, `bench/goldset.py`, caches, `runs/`, `tmp/` — all absent), and **`driver.py :: _drive`
untouched at zero lines** as chartered.

* **Ceiling 1 raised 950 → 1,200, ratified not charged.** Measured hand-authored is **1,105** — `src`+`tests`
  **722** plus `evidence/*.py` **383**. Both overruns are the orchestrator's mis-estimate: the charter
  budgeted **one** base-proof script at ≈120 lines and **three** were needed (the §0 probe, the golden
  capture/delta tool, the preflight attribution), and `driver.py` came in at 176 against ≈90 budgeted.
  **REV-051a governs.** Ceilings 2 (**25**/60) and 3 (**3,640**/18,000) are comfortably under.
* **The budget command itself is defective — F-050.** Its literal output here is **2974**, of which **1869**
  is generated `evidence/c053_*.json`: the command's only exclusion is `evidence/g11/`, so generated evidence
  written anywhere else is counted as hand-authored. The literal figure and the charter's derivation differ
  by ≈2.7×. **Use the corrected command in F-050 from now on**; charters already dispatched carry the old
  one, so **re-measure before concluding anything from an overage**.
* **Two additional test files GRANTED as "directly corresponding tests":** `tests/test_batch_driver.py`
  (Chunk B — its strict-pass fixtures carried no frozen release record and would otherwise have become
  classification-unavailable legs) and `tests/test_batch_driver_quarantine_artifacts.py` (unchunked; asserts
  `warnings == []`). Both are consequences of the chartered change, not scope creep, and both are inside the
  card's named focused gate.
* **`tests/test_batch_preflight.py`'s two failures are PRE-EXISTING and are NOT this card's.** Measured red
  at base `8920371` with `base_missed = ['t2pw.pipeline.strict_quarantine', 't2pw.pipeline.release_status']`
  and **`modules_newly_uncovered_at_tip = []`** — the diff adds no newly-uncovered module, it only repeats an
  already-uncovered name. The cure is `runner.py :: CHILD_IMPORTS`, **outside the boundary**; correctly
  reported and not touched. The second failure (`this project ships a .venv; the test assumes it`) is a
  worktree artefact independent of the diff. **Both belong to F-049's class — another unchunked file.**
* **§0 measurement PASSED:** `pwml_result["quarantine_report"]["release"]` is `PRESENT_AND_NON_EMPTY` at
  runtime through the real production export on **both** boundary dispositions (`fresh`, and the batch-shaped
  `carried` where `decision_matches` reuses the record), 15 keys observed. D-033's re-measure obligation is
  discharged.

**Charters v2 written 2026-08-17, and the v1 charters are RETIRED.** All three scratchpad v1 charters
(`C-052-charter.md`, `C-053-charter.md`, `C-056a-charter.md`) are **not to be implemented from**: each ended
with unrecorded rulings, and re-derivation against live source found that **every one had a false
load-bearing premise**. C-053's mandated a data source that **cannot execute** (a `str` subscripted as a
dict). C-052's cited a table column headed ***"Not owned by"*** as proof of ownership, and its mandated EP3
filename **collides with the CLI's own default**. C-056a's central *"BLOCKING"* finding is **measurably
false** — a public wrapper exists and the file itself proves a function-local import does not move the pin.
Between them, **five rulings were already discharged** while the charters still demanded them.

Replacements, written from the committed rulings and validated against live source, in this session's
scratchpad (`…\3a5a9d9e-0c1e-4421-9b7a-d913b20f0bdb\scratchpad\`):

| Card | Charter v2 | Built from |
|---|---|---|
| C-053 | `C-053-charter-v2.md` | **D-038** + the C-053 rulings block (Q1–Q9) |
| C-056a | `C-056a-charter-v2.md` | **D-037** + **D-039** |
| C-052 | `C-052-charter-v2.md` | **D-040** |

Each carries its own D-025 ceilings computed from its **ordered gates**, an explicit stop-condition list, and
a named focused gate for the files that belong to **no chunk** (**F-049** — the gap that nearly cost C-050i
its merge, and which has now bitten three cards).

**C-053 DISPATCHED 2026-08-17** at base **`8920371`** (the tip including C-050i), branch
`agent/p09-pwml-naming`, worktree `.claude/worktrees/agent-c053-pwml-naming`.

**C-050i correction round 2, ruled 2026-08-17 on REV-050i. One BLOCKING regression; a narrow boundary
extension granted.** The independent reviewer **REJECTED** `6605066` and earned it: it proved, in two trees
differing only in `src/`, that `tests/test_prefreeze_third_export_seam.py::test_db_reachability_reaches_the_exporter_in_both_directions`
**passes at base `8f7514f` and fails at tip** on the `[True-mixed]` and `[False-mixed]` parametrizations.
`CARRIER_POPULATIONS["mixed"]` is `{"name": "Glycine", "pathbank_compound_id": 78}` beside
`{"name": "gly", "kegg_id": "C00037"}`; pre-freeze renames `gly` → `Glycine`, producing two `Glycine` rows in
the **compounds (entity)** bucket, which the new guard refuses. `_reject_ambiguous_renames` is blind to it
(D-034 clause 5) and C-050h's refusal deliberately does not fire (nothing references the rows).

**This is merge rule 4 unsatisfied — existing affected tests fail with no documented delta — and it is R1's
stop condition hit a second time, undetected.** The card is not at fault for missing it: **the file is in no
chunk** (zero occurrences in `TEST_MATRIX.md` and in `chunk_d_gate.py`), so none of the 83 bounded evidence
records could contain it. That gap is **F-049**.

* **The refusal is CORRECT and is not to be weakened.** At base that fixture was **silently collapsing two
  post-freeze `Glycine` rows** — it was unknowingly exercising the exact defect C-050i removes. **Do not
  relax the guard, do not special-case the fixture's shape, do not narrow R1 further.**
* **Narrow boundary extension GRANTED — `tests/test_prefreeze_third_export_seam.py`, fixture only, as a
  fifth file.** Routing the repair to the seam's owner was considered and **rejected as circular**: the
  fixture change only makes sense with the guard present, so a separate card would have nothing to build
  against until C-050i merges, and C-050i cannot merge until it lands.
* **Conditions.** Change **only** `CARRIER_POPULATIONS["mixed"]` so the incidental `_norm` collision
  disappears. **Preserve the mixture-of-carrier-types property** — one row carrying a PathBank identifier
  beside one carrying a KEGG identifier — because that is what makes the value **carried, not inferred**
  (the fixture's own comment: *"Any exporter that tried to infer availability from the rows would have to
  guess here, which is why the value has to be carried"*). **Demonstrate explicitly that D-032's
  carried-not-inferred argument survives.** **Weaken no assertion.** Ship the **exact documented delta**
  (merge rule 4).
* **Add the file to the affected set and re-run it.** It must pass at the corrected tip.
* **Two false records must be corrected in the same round.** (i) The alias-residual docstring at
  `tests/test_pwml_ir_duplicate_row_refusal.py:462-463` claims *last-writer-wins* and *"attributable instead
  of latent"*; **measured, all of it is wrong** — binding is **first-wins by payload row order** and **no
  `ambiguous_entity_reference` warning is emitted at all**, so the residue is exactly as latent as before.
  That is **F-048**, owned by a new card **C-050k**; C-050i's only obligation is to stop misdescribing it.
  (ii) The instrument note above `GOLDEN` repeats **F-047**'s `_nonjson`/`repr` mechanism, which REV-050i
  **falsified** (`_nonjson` fires zero times) while confirming the phenomenon. Reword both to the measured
  claim.
* **Accepted from the review, not to be re-litigated:** boundary PASS · blocks-never-repairs PASS (no
  consolidation, no identifier-equality trigger) · R2 PASS (raise precedes `out.append`; both production
  call sites fail closed) · R3 PASS (base-vs-tip `ir_digest` **and** `report_digest` identical, entity and
  all four component buckets) · guard un-bypassable (reviewer added an **NBSP** variant the suite lacked;
  it refuses) · golden move **exactly** as granted (reviewer hand-computed the moved digest from first
  principles and it equals the committed value) · G9 honesty PASS (mutation kills 16 of 22 arms; the 6
  survivors are exactly the preservation pins) · one collision in the corpus, reviewer's own 32×9 census.

**C-053 rulings, issued 2026-08-17 after a read-only re-derivation against live source.** The C-053 charter
ended with seven owed rulings, **none recorded**. Re-derived, **two were already discharged and the charter
still demanded them**, two were **factually false**, and **two dispatch-blocking rulings the charter never
asked** were found. Product-policy content is **D-038**; the engineering rulings are here.

* **Q1 — STRIKE, already discharged.** The charter says `LEDGER.md:213` records C-041a with no
  implementation commit. It does not: `LEDGER.md:209` records C-041a `ACCEPTED`, merged `eeb576f`, reviewed
  tip `4177fe5`, with the "no implementation commit" clause explicitly **SUPERSEDED**. `:213` is the C-044
  row. **Do not "correct" a row that is already correct.**
* **Q7 — STRIKE, already discharged.** The charter says `TEST_MATRIX` still reads `177`/`core 150`. Zero
  occurrences of either remain; `TEST_MATRIX.md:42`, `:216`, `:298-299`, `:340` all read
  `179 = 152 + 4 + 23`, matching `chunk_d_gate.py:70-71`. **The surviving tripwire is kept:** if the diff
  touches `src/t2pw/pwml/writer.py`, `src/t2pw/app/streamlit_app.py` or `outputs/`, C-053 is **out of
  scope** — D-004 is batch-artifact-set only.
* **Q2 — boundary widened, and one exclusion measured rather than assumed.** `_add_strict_artifacts`
  (`driver.py:1373-1396`) is **the only site that names the strict artifact** (`:1380`
  `out["pathway.pwml"] = xml`) and a sweep of `docs/pwml_recovery_sprint/` finds it in **no card's
  manifest** — C-031's manifest is `_add_common_artifacts`/`_add_identity_artifacts` only. Granted with the
  full symbol list in `MASTER_PLAN` §9. **`driver.py :: _drive` is excluded at exactly zero lines**, and
  that is measured: `pwml_result` is already bound at `:2125` and already passed to both helpers at `:2126`
  and `:2177`, both of which already declare the parameter (`:1373-1376`, `:1673-1679`). No signature and no
  call site moves. **Absolute paths are used** — bare `driver.py`/`runner.py`/`acceptance.py` are ambiguous.
* **Q4 — C-053 strictly before C-054.** C-054 does **not** edit `_build_denominators`; it edits
  `goldset.py:647`, whose value is read at `acceptance.py:626`. The interaction is arithmetic: C-054 moves
  which papers enter `strict_pool` (**denominator**) at the moment C-053 moves `strict_ok` (**numerator**),
  and two simultaneous moves make neither attributable. `MASTER_PLAN` §3 hotspot 7 over-claimed C-054 as a
  branch and is corrected.
* **Q5 — test-function-level ownership of `_observable` and `GOLDEN` granted** as `MASTER_PLAN` §3
  **hotspot 10**, under a **growth-only** guard on `_observable`'s field list and an exact slot-by-slot
  delta for the seven digests.
* **Q8 (NEW — the charter never asked it).** C-053's base proofs **must run in a git worktree checked out
  at the base SHA, never in a `c045b_base_tree.py` / `c051a_base_tree_batch.py` export.** **F-042**:
  `PATHSPEC` omits `scripts/`, and `tests/test_bench_controls.py:344` does `from scripts import batch_run`
  while sitting in **Chunk B** — so Chunk B and SMOKE **cannot be collected** on an exported base tree and
  the card would report a false regression. **C-053 must not widen `PATHSPEC`** — F-042 is unowned and
  belongs with the card that also flips `require_scripts` on.
* **Q9 (NEW — the charter never asked it, and it is budget-affecting).** **Four of the five test files
  C-053 must change are in no chunk**: `test_release_status_classification`, `test_strict_quarantine_release_seam`,
  `test_batch_run`, `test_batch_report` (`test_batch_driver_seam_golden` is the fifth). Chunk B and SMOKE
  would not run them. This is not academic: `tests/test_release_status_classification.py:344-345` asserts
  `"release_status" not in row` **and** `sorted(row) == sorted(before)`, so it **fails either way** once
  `to_dict` emits the key — unconditional trips `:344`, conditional trips `:345`. It is C-053's mandatory
  merge-rule-4 re-pin and **nothing in the mandated gate set would surface it.** A named-file focused gate
  is added. `tests/test_strict_quarantine_release_seam.py:548` and `:661-687` are **must-not-weaken** inputs.
  **The D-025 generated-artifact ceiling must be recomputed with these gates included before dispatch** —
  a charter that omits the figure is a dispatch error (D-025).
* **Pre-implementation measurement, mandatory.** The static chain proving
  `pwml_result["quarantine_report"]["release"]` reaches the strict PASS path is sound, but its **runtime
  value on a live passing leg is unmeasured**. **C-053's first act is to measure it**, before the boundary
  is treated as final. D-033's standing rule forbids inheriting a site count or a "both/the" claim.

**A0-C7 ownership correction, ruled 2026-08-17 by the product owner. F-041 is authoritative.**
**C-052 owns A0-C8. C-052 does NOT own A0-C7, and must not absorb it** — letting a requirement change
owner for convenience is how a discharged-looking row ends up with nobody who ever discharged it.
A0-C7's owner column named **C-030**, which is `ACCEPTED`/merged `f3e9fb1` having recorded only **A0-C1**
discharged. A0-C7 is therefore re-assigned to a **new, narrow follow-up card `C-030a`** — the next unused
suffix in the C-030 family (a sweep of the control plane finds `C-030` and no suffixed variant in use).
**C-052 is not broadened to cover it.**

> **This also resolves a live disagreement between two findings, rather than leaving both standing.**
> **F-008** records A0-C7's *"Owner: **C-052**"* and *"the freeze seam is recorded incomplete pending
> C-052"*. **F-041** records the owner column as C-030 and A0-C7 as orphaned. They cannot both be right.
> **The ruling adopts F-041: the owner is `C-030a`, not C-052.** F-008 is amended in place so the two
> records agree. What survives from F-008 is its *technical* proof, which is load-bearing for the new
> card's boundary and is carried forward below.

| C-030a | A0-C7 — object-sharing across the freeze seam, proved not assumed | **RECONCILIATION 2026-08-18 (orchestrator, from Git ancestry):** **`ACCEPTED`, MERGED `be23905`, reviewed tip `495077d`** (D-051) — test-only, A0-C7 discharged by mutant discrimination. The `CHARTER PENDING` label below is historical. **`CHARTER PENDING`** 2026-08-17. Created solely to give orphaned **A0-C7** a named owner (**F-041**). **Test-only** — LEDGER's carried-requirement table binds A0-C7 and A0-C8 to *"test observable behaviour only"*, and it must not change C-011 lifecycle semantics: the freeze order (gate, hash, stamp, serialize, hand to SBML), the seven-field `CanonicalFreezeResult` contract, the refusal branch, or the `_freeze["canonical_json_path"] or sbml_input_path` fallback | C-011 ✔ · C-030 ✔ (merged `f3e9fb1`) | **Boundary is NOT C-030's.** MASTER_PLAN §9 scopes C-030 to `streamlit_app.py :: freeze_canonical_payload`, and **F-008 proved A0-C7 is undischargeable inside it**: the canonical payload there is a `deepcopy` that nothing in the returned dict aliases, so there is no sharing inside that boundary to break, and the documented share→copy mutant is indistinguishable from the tip on all nine seam observables across all 39 legs. The real sharing sites are in `run_post_pipeline_sbml_artifacts` — **MASTER_PLAN §9 row 1, shared with C-050 and C-052**, so the charter must declare exact functions/tests and **serialize against C-052**. The discriminator F-041 located is live and **re-measured today at `streamlit_app.py:3746`** — `"final_mapped": canonical_export_payload or final_export_payload` — **not `:3748` as F-041 recorded; the line had drifted by two and the charter must use the measured number** |

| C-050i | post-freeze row dedupe fails closed (**F-039**) | **`ACCEPTED`, merged `509faee`, reviewed tip `d5356b9`** — independent bare `APPROVE` after **two** correction rounds (R1 narrowed by the orchestrator in round 1; a reviewer-found regression fixed in round 2). **Post-merge gates on the merged state:** SMOKE **460 exact** · whole-tree G11 **exit 0**, 2056 artifacts, 0 non-compliant · protected manifest **42/42** · **zero owned surviving processes**. Merge introduced **five** files and touched no cache, `runs/`, `runs_verify/`, `outputs/` or `data/`. Superseded status: **`READY`** 2026-08-16, awaiting the heavy-job slot. **Scope addition 2026-08-16 (REV-050h finding 4):** F-039's *unmeasured residual* — EP3's second `run_prefreeze_resolution` creating a `_norm` collision absent from the committed file — is **explicitly in C-050i's scope**, not inherited by proximity. C-050h leaves *unreferenced* created duplicates unrefused by design (D-035 clause 6 names *ambiguous connectivity*, which they do not create); C-050i owns the harm. Measured: that shape exports identically at base and tip, so C-050h changes nothing there. Measured verdict: **merge rule 8 IS violated at EP3** — `run_pwml_export` reaches `ir._dedupe_named_rows` on a deepcopy-of-a-deepcopy of the frozen payload, **after the hash**, and that is the path every committed batch leg takes (`driver.py:2085-2112`). **Live corpus exposure is ZERO of 32 legs**: the only colliding leg is D-034's, which now aborts pre-freeze. Proof it is invented biology rather than a harmless duplicate: on that leg the exporter re-bound reaction 9 from PathBank 40738 / ChEBI 60365 to PathBank 40982 / ChEBI 58603 **after the hash**, with one warning and no error; references do not dangle, they silently repoint, because `entity_by_name` keys on the same `_norm` the dedupe grouped on | F-039 ✓ measured |

**C-050i scope, ruled 2026-08-16.** Promote `duplicate_named_record` (`ir.py:409-415`) from **warning to blocking error**. Chosen over deleting the dedupe because deletion changes IR key numbering (`ir.py:420`) and moves goldens, while promoting severity does not. **Group identity is `_norm`** and nothing else (**F-040 as corrected**) — `entity_by_name` (`ir.py:1105-1117`) and `resolve_entity` (`ir.py:1371`) key on it. **G9: new capability with an explicitly labelled new acceptance test** — there is no base SHA at which a behavioural probe over the tip's corpus fails, because the only leg reaching the code now aborts earlier. Fabricating a base failure here is a reject. **Out of scope:** `process_normalizer._dedupe_named_rows` (**F-044**, pre-freeze, unowned).

> **Correction 2026-08-17 — this paragraph contradicted the row above it.** As first written it closed
> with "*and the unmeasured non-participant-rename residual, which needs a live pre-freeze run*" in its
> **out-of-scope** list, while the C-050i row (`:358`) records the opposite from **REV-050h finding 4**:
> that residual is **explicitly in C-050i's scope, assigned by name so it cannot fall between C-050h and
> C-050i**. The row is the later and correct record; the out-of-scope clause was stale and is struck.
> **The residual is IN SCOPE and the heavy-job slot is granted for it** (charter §5). Recorded rather than
> silently fixed, per D-034 clause 5 — a mis-recorded guard has already cost a later card real time.

**C-050i rulings R1–R5, issued 2026-08-17 by the Lead Orchestrator before dispatch.** No `C-050i-charter.md`
existed in any session scratchpad or in the repo; the charter was reconstructed from F-039, the F-039
measurement, and live source read at `fd5afd8`, and these five rulings — which were **never recorded** —
were issued with it. R1–R4 are **engineering details consistent with locked policy**, decided
autonomously; R5 restates an existing ruling because it is load-bearing.

* **R1 — the guard binds every `_dedupe_named_rows` caller, not compounds only.** One shared function;
  both call sites (`ir.py:957` components, `ir.py:1049` entities) feed a `_norm`-keyed lookup with the
  identical silent-repoint mechanism (`entity_by_name` `:1105-1117`; `component_by_name` `:995-1000`).
  Narrowing to compounds would need a discriminator parameter and would knowingly leave the same latent
  hole in four component and four other entity buckets. **Fail closed** governs. Measured impact of the
  wider scope is **zero** — F-039 replayed first-wins `_norm(_canonical(...))` across all 32 committed
  `final_mapped.json` over all five entity and all four component buckets and found **no component-bucket
  collision in any leg**. *Stop condition:* if an existing test or golden depends on a **silent collapse**
  in any bucket, the card **stops and reports**; it may neither weaken the guard nor narrow scope itself.
* **R2 — refuse by raising, not by report severity alone.** `_add_issue(report, "error", …)` sets
  `report["ok"] = False` (`ir.py:322-325`) but **does not stop IR construction**: the row would still be
  dropped, the reference would still repoint, and an invalid IR would still be returned. Only a raise
  satisfies "fail before invalid IR can be emitted". Model it on the in-file precedent
  `UnresolvedCompoundRowError` (`ir.py:59-75`, raised `:1079`), introduced by C-051 under D-021 for exactly
  this shape of post-freeze refusal. **The structured diagnostic is preserved and must name *both*
  conflicting rows** — survivor and intruder, with names, keys and pointers; the current message names one.
* **R3 — key/row numbering on the non-colliding path stays byte-identical.** `record["key"] =
  f"{key_prefix}_{len(out) + 1}"` (`ir.py:420`) is why deletion is out of scope. A payload with no
  collision must produce a **bit-identical** IR to the base.
* **R4 — the CLI entry point needs no separate adjudication for this card.** The F-039 measurement's open
  question 4 (EP2 "post-freeze in the artifact sense, indeterminate in the process sense") does **not**
  need deciding here: the guard sits in the shared IR builder and so applies uniformly wherever an IR is
  built (EP2 `writer.py:2734`, EP3 `streamlit_app.py:4135`; EP1 builds none). **The card introduces no
  per-entry-point discrimination.** EP2's status **remains an open finding, unowned by C-050i.**
* **R5 — group identity is `ir._norm` and nothing else** (restated; **F-040 as corrected**).
  `prefreeze_resolution._norm` is a **byte-identical duplicate**, not a competitor;
  `process_normalizer._normalize` is **incomparable** (deletes where `_norm` substitutes a space). Do not
  import, reimplement or "reconcile" any other normalizer.

**C-050i D-025 ceilings, computed 2026-08-17 from the ordered gates** (full Chunk D `179 = 152 + 4 + 23` ·
SMOKE 460 · focused runs · the G9 constructed-fixture proof · the §5 live residual run · G11 evidence ·
provision for **at least one failing run** · headroom for **one** correction round):
**(1) hand-authored additions+deletions ≤ 900 · (2) generated artifacts ≤ 72** (Chunk D self-allocates
≈33, SMOKE 1, focused 4, G9 proof 1, live run 3, G11 ≈12, failing-run provision 3, correction ≈12 ⇒ ≈69) **·
(3) generated size ≤ 20,000 lines.** Ceilings, not targets: genuine evidence is never deleted to meet a
number, and **the adversarial bypass arm is not to be cut to fit** (REV-051a).

**ID allocation note.** `C-053`, `C-054`, `C-055`, `C-056a/b` and `C-057` were **already allocated** in
`MASTER_PLAN` §9 to other planned cards and are **not** reused here. The duplicate-row card takes the next
free suffix in its own family, **`C-050h`**; the harness card takes the next free harness number, **`H-010`**
(`H-001`…`H-009` are taken).

**Serialization note.** C-050h and H-010 may both touch shared test infrastructure. If they do, their
**merges serialize** and only the implicated cross-seam checks re-run after the first merge — their complete
reviews are not repeated.

**Still ahead of T-100:** C-053 (dependency C-041 is merged, boundary disjoint) and C-056b (blocked on
C-056a, which was never dispatched — **dispatch C-056a first**). **No strict benchmark-success figure may be
quoted until both C-053 and C-056b are merged.**

## Wave E — placeholders

| ID | Task | Status | Deps | Notes |
|---|---|---|---|---|
| C-060 | p51 false-identifier repairs | **RECONCILIATION 2026-08-18 (orchestrator, from Git ancestry):** **`ACCEPTED`, MERGED `f2f7599`** (`agent/p51-false-id-repairs`, reviewed tip **`7e189ef`**, 1 correction round), and its follow-on **C-060a merged `6c98508`**. The `AUTHORIZED`/dispatch-ready label below is **SUPERSEDED** — the card was built, reviewed and landed on 2026-08-13. Wave E is therefore complete (C-061 + C-061a merged `afcbf1d`). `AUTHORIZED` 2026-08-13; **charter written 2026-08-13, dispatch-ready, held only by the four-lane cap.** Its "serialize behind C-042" constraint is **discharged** — C-042 merged at `8917349`, and the adjacency risk was **measured, not assumed**: `pipeline.py :: merge_additions` is at `:1130` while C-042a's possible consumer touches in the same file are at `:3531`/`:3558`/`:3722`/`:3971`, ~2,400 lines apart, so C-060 may run concurrently with C-042a. **Its first act is the dissolution check** — R-003 measured F1/F3/F4 against `3bfa7af`, not the live tip; if any of the three no longer reproduces at the dispatch base, the card narrows or dissolves and that is reported, not worked around | R-003 ✔ | **Scope accepted (R-003 M1 only):** NEW `pipeline/entity_admission.py` assay-reagent admission gate + minimal call site in `pipeline.py :: merge_additions`. **A0-C5 binds: the hallucination gate runs first and independently of `cofactor_policy`** — R-003 measured C-018's classifier returning `participant`/high for `succinyl-CoA`, so consulting it first would have protected the run's worst fabrication. |
| C-061 | p52 missing-supported-reaction repairs | **RECONCILED 2026-08-18 (Git ancestry):** **`ACCEPTED`, MERGED `afcbf1d`** as the C-061 + C-061a composite. The `DISPATCHED` label below is historical. **`DISPATCHED` 2026-08-13** from `472293c`, ceiling 900. Base for its G9 proof is `472293c` — **not** R-004's `3bfa7af` | R-004 ✔ | **Scope accepted (R-004 B-2 only):** `rag/admission.py` :: `parse_span_relation`, `validate_evidence_span`. The blocking §9 ownership conflict with C-035 **is corrected** in `MASTER_PLAN` §9 and in C-035's row above. |

## Carried Wave A0 requirements — binding on the named owners

Opened 2026-08-10 under `CONTROL-PLANE-RECONCILE-001` §7. Every row below is a **deferred
finding from a merged Wave A0 card**, bound here to the card that must discharge it.
**Control plane only — no production change is authorized by this section, and no merged
card is reopened.** A seam is **not complete** while its row is open.

| # | Requirement | Source | Primary owner | Cross-refs |
|---|---|---|---|---|
| A0-C1 | `canonical_graph_sha256` must hash the normalized identifier fallback values from `mapping_meta` / `candidates` **when those values are consumed to establish canonical entity identity**, and must not hash ranking, transient metadata or provenance noise. Acceptance must cover the **49 committed rows** dependent on that fallback and must prove that changing an identity-relevant fallback **changes** the hash. **Census ratified at 49 by the product owner on 2026-08-13 (F-014).** The former figure **60** did not reproduce: three independent censuses over all 32 committed `final_mapped.json` measure 49 — compounds 38, protein_complexes 11, all at tier 4, across 19 distinct files — with corpus drift excluded. C-030 discharged this requirement against all 49. **Historical evidence recording 60 is not rewritten.** `C-013.md:48-50`'s EXCLUDE line is qualified in place and may not be cited against this. | C-013 merge `09fb40d`, finding 3 | **C-030** | C-013 (qualified), C-052 |
| A0-C2 | **Retry policy for the four call sites C-014 stripped of SDK retries** — `src/t2pw/stoich/agent.py:477`, `:552`, `:580` and `src/t2pw/rag/embed.py:154`. Each owner must **decide and test** its replacement resilience before its seam is considered complete. **Retries must not be restored now.** | C-014 merge `c832894`, finding 4 | **BL-001** (backlog item, H-007 / D-022 §5). No existing card owns either file — re-verified at `08d5d07` against `MASTER_PLAN` §9, §2, this ledger and every `prompts/*.md`, and `C-014.md:145` says so in terms. Binding it to C-032, C-035, C-042, C-043 or C-055 would invent overlapping ownership, which the closeout is forbidden to do | C-032 (consumer of `worst_case_call_seconds()`, cross-ref only) |
| A0-C3 | C-056a must **not** wire the literal organism comparison unchanged; organism comparison must reuse the repository's established synonym/canonicalization behaviour (`rag/eligibility.py:1366-1404` — `_organism_aliases`, `_canonical_organism`, `_taxon`), including `E. coli` ≡ `Escherichia coli`. **No competing synonym table.** Positive, negative, abbreviation, whitespace, punctuation and deterministic-regression assertions all required. | C-017 merge `fc8b059`, finding 1 | **C-056a** | C-045 / D-016 (shared owner if one is later selected); C-056b |
| A0-C4 | C-056a must combine `evaluated` + `ok` + `inapplicable_checks`. **`confirmed` can never be True on a production run**, so gating on it alone would ship nothing, ever. | C-017 merge `fc8b059`, finding 3 | **C-056a** | C-056b |
| A0-C5 | A `participant` verdict must **never** protect an entity from hallucination removal and is **not** evidence of paper support; C-060's hallucination gate runs **before and independently of** any `cofactor_policy` consultation. Pinned against `src/t2pw/bench/gold/pinned_v1.json`. | C-018 merge `85fae43`, item 1 | **C-060** | R-003; D-010 constraint preserved |
| A0-C6 | The R-003 harness must extract `name` and `aliases` from the gold objects it evaluates (they are `{name, quote}` objects, not strings) and must **fail closed** when the intended cohort or assertion count is zero. Pinned against `src/t2pw/bench/gold/pinned_v1.json`. | C-018 merge `85fae43`, item 5 | **R-003** | C-060 |
| A0-C7 | Capture the pre-seam caller-owned payload reference and prove the intended object-sharing relationship after `freeze_canonical_payload`. `final_mapped is result["payload"]` is **tautological** (True 35 / False 4, tracking `quarantine_ok`; a share→copy mutant survives it) and is **not** sufficient. | C-011 merge `0182eae`, item 1 | **C-030a** (re-assigned 2026-08-17; was C-030, which shipped without discharging it — **F-041**) | C-050, C-052 |
| A0-C8 | Capture and assert the **actual `canonical_json_path`** for all **39** cohort legs, **including the path supplied to downstream SBML generation**. `sbml_input_source` alone is insufficient — **nothing in the suite currently pins the SBML input path**. | C-011 merge `0182eae`, item 2 | **C-052** | C-030, C-050 |

**Note on A0-C2 — the owner does not exist, and none is invented here.** A sweep of the
whole control plane (`MASTER_PLAN.md` §9 branch register, `MASTER_PLAN.md` §2, this ledger,
every `prompts/*.md`) finds **no ownership row, register entry or boundary grant naming
`stoich/agent.py` or `rag/embed.py`** — stated as ownership rather than as a raw string
count, because this note itself names both paths.
**No existing card owns either file**, re-verified at `08d5d07` by H-007. A0-C2 is bound to
backlog item **BL-001** (D-022 §5) rather than to C-032, C-035, C-042, C-043 or C-055, none
of which owns those paths; binding it to one of them would invent overlapping ownership.
**This does not block Wave A1.** Until BL-001 is scoped and given an owner, the four call
sites run with **no retry layer and a 300 s per-request bound** where they previously had
the SDK's 2 retries and 600 s — a known, recorded, accepted state, not a defect of C-014.
Note the shape precisely: `llm/client.py` pins `SDK_MAX_RETRIES = 0` and carries its own
8-attempt retry loop, but these four sites call `_client.chat.completions.create(...)`
**directly**, so they bypass that loop and are the only ones left with no retry at all.

**A0-C7 and A0-C8 must test observable behaviour only** and must not change the established
C-011 lifecycle semantics: the freeze order (gate, hash, stamp, serialize, hand to SBML),
the seven-field `CanonicalFreezeResult` contract, the refusal branch, or the
`_freeze["canonical_json_path"] or sbml_input_path` fallback. See **D-021** for the
C-040/C-050/C-051 ownership lock that governs the surrounding seams.

## Milestone tests

| ID | Milestone | Status | After | Legs | Wall clock |
|---|---|---|---|---|---|
| T-100 | M1 | **`RUN, NOT DISCHARGED` 2026-08-18 — full record in D-055.** Four authorized curator-enabled legs ran into `runs_verify/2026-08-18_1328` (committed `8ea52c4`); G11 **0 survivors**. **Acceptance FAILED**: `TEST_MATRIX.md:477` requires `PMC12452463 -> review_required`, and both strict legs classified **`diagnostic_only`** because `release_status.py:414-419` tests `strict_gates_passed` **before** any coverage branch and `strict_quarantine.py:2025-2034` appends structural reasons to `refusal_reasons` unconditionally (**F-062**). **NO FIGURE IS QUOTABLE** — the scorer returns `NOT ACCEPTED: INCOMPLETE (2/10 papers, 4/20 legs)`; `acceptance.py:397` measures completeness against the **gold-set size 10**, so no re-run of four legs can ever qualify. A quotable strict **rate** needs all 20 gold legs = **T-104**. **Both strict legs failed on one shared cause** — a RAG alias-duplicate of EntB sharing UniProt `P0ADI4` (**F-057**) — and **neither leg was an incomplete-but-correct pathway wrongly dropped**: both carry fabricated `EntE` transporters (**F-058**) and leg B carries LDH assay reporters as metabolites. **Ten findings F-055..F-064, seven `product_contract_violation`.** ⚠ **F-062 must not be fixed before F-057 and F-058 — merge rule 6.** | Wave B | PMC12452463 ×2, PMC12096016 ×2 | ~1.5 h |
| T-101 | M2 | **`PREREQUISITE SATISFIED`, not started.** **RECONCILED 2026-08-18:** Wave C is complete — C-040 `334ad88`, C-041 `b5bbf08`, C-042 `8917349`, C-043 `3c04d4b`, C-044 `8f1a692`, C-045 in composite `beddcdd`. Runnable once T-100 discharges and the heavy slot frees. | Wave C | + PMC12444477 ×2, PMC12782028, PMC12312563 | ~2 h |
| T-102 | M3 | **`MEASURED — organism/SBML axis structurally unreachable (F-009)` 2026-08-20 at `32f3a57`. NEVER record this as `PASS`** — the product owner already ruled at `LEDGER.md:821` (PACK 2): *"if T-102's organism dimension is unreachable solely because SBML lacks taxonomy annotation, that exact limitation is recorded truthfully and the other benchmark dimensions continue."* **Measured, all re-verified today, not quoted:** `grep -rn "taxonom" src/t2pw/sbml/` → **0 hits across 7 modules** · `canonical.py:226` `_KINDS` **omits species** · `canonical.py:855-856` — **one `Difference` forces `not_equivalent`** · `find runs/ runs_verify/ -iname '*sbml*'` → **0 SBML artifacts for any leg**. The axis is pinned by a committed passing test: `tests/test_canonical_biological_equivalence.py` **29 passed in 0.19 s**, G11 0 survivors (`evidence/g11/T-102/02-canonical-axis-pythonpath.json`). **REFINEMENT to F-056:** SBML is not absent from the code — `streamlit_app.py:6472` passes `build_legacy_sbml=True` from a **manual UI button** (`:6463-6472`) that the batch driver never clicks; `:6015`, the automated path, passes `False`. Conclusion unchanged, mechanism sharper. **TWO LIMITATIONS REMAIN EXPLICIT AND UNOWNED — the milestone is NOT discharged:** (1) **no `canonical_graph_sha256` baseline exists for PMC12856317**. F-056's *"0 files anywhere"* became **false** — one now exists, `runs_verify/2026-08-18_1328/papers/PMC12096016/research/final_stage3_gate_report.json` = `2597ca91…`, arrived with `8ea52c4` after F-056 was written — but it is the **wrong paper and the wrong mode**; `stamp_report` still has **0 call sites in `pwml/writer.py`**, so a CLI re-export emits none. (2) **nothing drives `biological_equivalence` end to end**: defined `canonical.py:827`, its only repo-wide callers are in its own test file; the ~80-line offline probe is **unowned work needing a grant**. **No unowned production functionality was created to make this green, and none may be.** Acceptance as specified (`TEST_MATRIX.md:479`) additionally requires SBML, which no leg has ever produced. | C-052 | PMC12856317 equivalence | measured ~15 min |
| T-103 | M4 | **`BLOCKED` on C-055 — the only remaining card that gates a milestone.** C-055 is `DEPENDENCY-READY, NOT STARTED` (all three deps merged). | C-055 | 4 RAG legs | ~1.5 h |
| T-104 | M5 first RC | **`RAN 2026-08-21/22` — full record in the TEST_MATRIX T-104 row and D-062/D-063.** 20/20 legs into `runs_verify/2026-08-21_2239` (committed `2673067`); deepseek-v4-flash; 5.44 h; G11 **0 survivors** on every job. **STATUS: `MEASURED — NOT ACCEPTED`. Never record as PASS.** Topics set built this session (`topics_t104.txt`) from the 10 pinned gold cases, scopes verbatim, `--verify-plan` = `OK`, all 10 `[pinned_override]`. **BENCHMARK COMPLETION `COMPLETE (10/10 papers, 20/20 legs)` — the first complete run of the sprint; T-100's `INCOMPLETE (2/10)` bar is cleared and the rates are quotable.** Priorities **1, 4, 5 FAIL** (7 false real identifiers; coverage 0/7; strict PWML 0/4), 2 and 3 PASS. Legs: 10 PASS, 6 `scope_conflict`, 2 `no_reactions`, 2 TIMEOUT. **Four findings F-094..F-097 + F-092 re-confirmed; only F-094 and F-096 are `product_contract_violation` and only those two justify code.** ⚠ **F-094: PMC12452463/strict reached `release_ready`, which `PRODUCT_CONTRACT.md` §13 forbids outright — it is the only strict leg that emitted a bare `pathway.pwml`.** | Wave E | full pinned, 20 legs | ~7 h |
| T-105 | M5 second RC | **`HELD` per D-063 (2026-08-22) — NOT RUN.** The previously recorded blocker ("T-104 has not run") is **discharged**: T-104 ran 2026-08-21/22 into `runs_verify/2026-08-21_2239` (`2673067`). **The blocker is now "F-094 and F-096 corrections not yet merged".** T-105's acceptance is *remaining* failures explained and classified, which has no meaning until corrections exist; and re-running the same 20 legs against unchanged code would **collapse the two release candidates into one**, forbidden since PACK 9. **Prerequisite chain: (1) cards opened for F-094 and F-096; (2) implemented, reviewed against the actual diff, merged; (3) T-105 re-runs the same 20 pinned legs.** **F-095/D-062 is NOT in that chain** and may be sequenced independently — `review_required` is not a strict export, so it does not move the strict rate either way. | **F-094 + F-096 corrections merged** (was: "T-104, then its corrections") | full pinned, 20 legs | ~7 h |

---

## Change log

| Date | Entry |
|---|---|
| 2026-08-05 | Ledger created. `sprint/pwml-recovery` cut from `research-mode` @ `9e1b9ab`. All tasks `BLOCKED` pending product-owner approval of the setup report. |
| 2026-08-05 | Control-plane audit passed. Every `MASTER_PLAN` § 1 / § 2 citation verified against the source; `baseline_acceptance.json` SHA-256 matches `PROVENANCE.md`; all 28 `TEST_MATRIX` files exist; the 7 tracked modifications are exactly as documented. Three citation line-drifts (≤ 6 lines) recorded in the report; all function-level claims exact. |
| 2026-08-05 | INIT-001 Step 0: `evidence/bounded_run.py` created (Windows Job Object, `KILL_ON_JOB_CLOSE`; cleanup in `finally`; graceful→forced; survivor verification). Validated 6/6 synthetic cases, **0 proved survivors**. `batch/runner.py` not modified (owned by C-032). |
| 2026-08-05 | INIT-001 Step 3: `runs_verify/2026-08-04_1754/` committed at `0c469f7` — 152 files, ~6 MB, `cache_snapshot/` (38 MB, 2 files) excluded. `.git` 158 → 159 MB. C-010's last two allowlist legs are now verifiable in an isolated worktree. |
| 2026-08-05 | INIT-001 Step 4: baseline measured. Smoke **457** ✔, chunk D **177** ✔, A **123** ✔, B **225** ✔, C **109** ✔. Full suite **2311 passed / 2 failed / 8 skipped** over 104 files. `bench_acceptance.py` re-run is **byte-identical** to the committed baseline (SHA-256 `d3538f4b…4ec3`). Every heavy job: **0 surviving owned processes** (G11). |
| 2026-08-05 | **BLOCKER RAISED.** The pinned `FULL_STACK_BASELINE` in `test_strict_quarantine_real_artifact_replay.py` fails on `ORIGIN_SHA` — 23 legs pinned, 39 measured — because `runs/2026-08-02_2130` was archived at `5f2cd2f` (2026-08-04) after the pin was written at `404cc8d` (2026-08-01). Proven sprint-independent: no sprint commit touched `runs/` or the test, and the test globs `runs/` only. Affects TRAP-2 and C-010's acceptance. See `BASELINE.md` § 5. Wave A0 not dispatched. |
| 2026-08-05 | **INIT-001 ACCEPTED** by the product owner. |
| 2026-08-05 | Product-owner rulings applied → **D-011** (O-2 closed: narrow `.gitignore` for `runs_verify/*/cache_snapshot/`; 304 MB reported, **nothing deleted**), **D-012** (baseline cohort frozen to a reviewed manifest, not re-pinned), **D-013** (replay assertion scoped to payload files), **D-014** (O-3 closed: the 7 scratch files are protected — never staged, committed, reset, restored, stashed, regenerated or reformatted). O-1 remains the only open question. |
| 2026-08-05 | New pre-Wave-A0 tasks **H-001** and **H-002** created and prompted. Wave A0 and C-010 now block on them, not on INIT-001. Cohort measured and verified: 39 legs, `evidence/baseline_cohort_measured.json` + `baseline_cohort_measure.py`. Exactly **2 of C-010's 6** allowlist legs fall inside the gate's cohort, so the two deltas stay separable. |
| 2026-08-05 | Six missing Wave A0 prompts written from the templates: C-012, C-014, C-015, C-016, C-017, C-018. All ten A0/pre-A0 prompts now instruct the agent to read the base SHA with `git rev-parse sprint/pwml-recovery` **at dispatch**, never from a document. |
| 2026-08-05 | Control-plane corrections: three line-number drifts fixed (`_drop_quarantined_processes` `:1868`→`:1862`; `_revalidate_surviving_processes` `:1449`→`:1448`; TRAP-2's replay pin `:416`→`:384-393` def / `:432` assert) in `MASTER_PLAN`, `_SHARED_BLOCKS` and `_TEMPLATE_IMPLEMENT`. C-033/C-044 ownership path-qualified, and a canonical-path table added to `MASTER_PLAN` § 9 covering the `pipeline.py` / `map_ids.py` / `extract.py` re-export shims. |
| 2026-08-05 | **Awaiting session restart** to register the four project subagents. Nothing dispatched. |
| 2026-08-05 | Subagent registration **confirmed**: all four resolve from `.claude/agents/` (previous session was rooted one directory too high). `pwml-reviewer`, `pwml-test-runner` and `pwml-bio-auditor` are structurally read-only — no `Write`/`Edit`/`NotebookEdit` — so G5 cannot be satisfied by an agent that can also edit. `pwml-implementer` carries `isolation: worktree`. |
| 2026-08-05 | **SPIKE-002 ACCEPTED** at `cfd0e10`. Verdict **`LIFT_WITH_ADAPTER`** — the two `ir.py` functions need no input reshaping; independently reviewed **APPROVE_WITH_CORRECTIONS**, reviewer does not support `REQUIRES_RESHAPE`. Investigation only: no branch, no file modified, `git status` identical before and after for both agents. Blocks C-040/C-050/C-051 only; none is authorized. Three items escalated to the product owner — see § below. |
| 2026-08-05 | **H-001 ACCEPTED**, merged at `aa0fb0a`. The replay merge gate now reads `tests/data/baseline_cohort_manifest.json` **only**; no glob decides cohort membership, and a missing entry raises `BaselineCohortError` naming it rather than shrinking the cohort silently. All 39 entries verified before freezing. Pinned baseline moved deliberately (G4): `legs_examined` 23→39, `exportable` 1→8. **No leg's verdict changed** — the reviewer re-measured the original 23-leg subpopulation with the base `_full_stack` and reproduced the old pin exactly, so the whole movement is the 16 unmeasured legs of `runs/2026-08-02_2130`. C-010's six-leg allowlist preserved separately per D-012; exactly 2 fall inside the cohort, as ordinary members with no delta pre-applied. |
| 2026-08-05 | **H-002 ACCEPTED**, merged at `e5eeb8c`, after **one CORRECTION round**. Round 1 narrowed the assertion correctly but removed a real (if over-broad) tripwire while adding a comment claiming it survived — an unlisted artifact carrying `key_compounds` passed silently. Correction restored the classification as an enforced test that fires on any unlisted carrier within `_MAX_PAYLOAD_BYTES` and names it; the reviewer closed its own finding by re-derivation. A second, text-only round corrected a wrong illustrative number in a rationale docstring (`14 or 37` → the measured `12 or 39`) and qualified four unbounded claims. ~~Net coverage against the base is a strict increase, not a trade.~~ **RETRACTED 2026-08-05 — that sentence was wrong; see the correction entry below (closeout review, D-3).** |
| 2026-08-05 | **Pre-Wave-A0 verification GREEN** on `e5eeb8c`. Chunk E **173 passed / 0 failed** (pre-sprint: 159 passed / 2 failed). Smoke **457**, chunk A **123**, chunk D **177** — all identical to the INIT-001 baseline, zero regressions. Both named gates pass. `src/` **0 files changed** across the whole pre-A0 sequence. ~~Every job across SPIKE-002, H-001, H-002 and all verification ran through `bounded_run.py`; **0 surviving owned processes throughout**.~~ **RETRACTED 2026-08-05 — that universal claim is false; three lapses were disclosed and are enumerated in the correction entry below (closeout review, D-1).** The 7 D-014 protected files were never staged, committed, reset, restored, stashed, regenerated or reformatted. |
| 2026-08-05 | **Boundary extension granted by the Lead Orchestrator** (recorded as a deviation): H-002 was authorized to amend the module docstring of `test_strict_quarantine_real_artifact_replay.py`, **text only**. Reason: `"No Stage-0 context is stored in any archived leg"` is literally false — 9 archived legs store Stage-0 candidates — and at base that falsehood announced itself through a red test, while after the narrowing it would have been silent. The implementer had correctly declined to touch it on its own authority. |
| 2026-08-05 | **Wave A0 and C-010 are no longer blocked on H-001/H-002.** They are blocked on product-owner approval only. Not dispatched. Wave E (C-060/C-061) remains undispatchable pending R-003/R-004 per D-010. |
| 2026-08-05 | **Closeout-record corrections D-1…D-6 merged at `81e65e3`.** Documentation only, two files, 47 insertions / 17 deletions, reviewer verdict **APPROVE**. Ordered by the product owner after the independent closeout review of `2b786aa..1c2dbee` returned `APPROVE_WITH_CORRECTIONS`. The reviewer re-derived every number written into the corrections from the repository and re-read all three corrected line ranges at source. |
| 2026-08-05 | **H-003 and H-004 opened at `5a476a5`** as pre-A0 G11 infrastructure tasks, on findings the closeout review produced. Control plane only. |
| 2026-08-05 | **H-003 ACCEPTED**, merged at `aab975a`, after **one product-owner-authorised CORRECTION round**. The defect: `_drain()` caught `OSError` only, so writing unencodable text to a cp1252 console raised `UnicodeEncodeError` — a `ValueError` — which escaped and skipped the `--json` write on `main()`'s normal return path. The child was still reaped by the `finally`, but the run produced **no cleanup report and was uncertifiable under G11**: a job could be killed by its own wrapper and leave no record because the child printed a character the console cannot encode. It was reaching the real production caller, `baseline_suite.py`'s `_Tee`, not only `main()`. All seven card properties verified independently by reviewer and test-runner. Selftest **8/8 `WRAPPER VALIDATED`** on the repaired build against **6/8 `WRAPPER NOT VALIDATED`** on the base build — the repair is behavioural and non-vacuous, and the Unicode case injects cp1252 explicitly so it does not depend on the developer's console. Smoke **457** and chunk E **173p/0f** exactly on baseline. Zero files under `src/`, `tests/` or `batch/`. |
| 2026-08-05 | **H-003 first review returned `APPROVE_WITH_CORRECTIONS` — a stop condition, and the sequence stopped.** No correction round was dispatched until the product owner authorised one explicitly. That authorisation was scoped to the two findings and stated that it **does not establish authority for automatic future corrections**. The two findings were C1 (`_report_from_render` ignored the rendered `cleanup :` status and `json report :` path, so case 8's persisted artifact recorded a *false* `cleanup_success: false` beside `final_surviving_count: 0` — the exact ambiguous pair that would mislead a G11 checker) and C2 (case 8 aborted at base on `AttributeError` before reaching `_record`, making its regression demonstration symbol-existence rather than behaviour). Both resolved; re-review returned **exact `APPROVE`**. |
| 2026-08-05 | **Reportable documentation drift, not a regression:** `BASELINE.md` § 2 still records chunk E as *"159 passed, 2 failed"* — the pre-sprint figure. The accepted gate value of **173 passed / 0 failed**, reproduced independently by the test runner, currently lives only in this ledger. `BASELINE.md:167` is likewise stale against `docs/change_log.md`, and `BASELINE.md:206-207`'s per-batch breakdown sums to 41 rather than the manifest's 39. All three are queued for the authorised authority-document correction round; none is owned by H-001/H-002/H-003, and none was touched by them. |
| 2026-08-05 | **CLOSEOUT REVIEW — `APPROVE_WITH_CORRECTIONS`.** An independent closeout review of the pre-Wave-A0 sequence returned six findings, D-1 (high) through D-6 (low). The product owner ordered a **documentation-only** repair: no production code, no test, no source file, two documents only (`LEDGER.md`, `SPIKE-002-REPORT.md`). The six entries below record each correction. **No historical entry was deleted** — two false sentences were struck in place, with a pointer to the correction that supersedes them, because what the sprint claimed at the time is itself part of the record. |
| 2026-08-05 | **D-1 CORRECTION (high) — the G11 process-lifecycle record for the pre-A0 sequence.** The struck sentence in the "Pre-Wave-A0 verification GREEN" entry above claimed that *every* job across SPIKE-002, H-001, H-002 and verification ran through `bounded_run.py` with 0 surviving owned processes *throughout*. **That universal claim is false.** Three lapses were disclosed by the agents themselves, and the product owner has ruled on each. **(1) An unwrapped chunk E invocation** — H-001's implementer, while trying to capture a summary line past `tail`. **Ruling: a G11 nonconformance. Its result is not evidence of record; the wrapped rerun supersedes it and independently established zero survivors.** **(2) A discarded wrapper run whose report was lost** — H-002 reviewer round 2, label `h2r_r2_probe2`: stdout was piped into `head`, which closed the pipe early, so the `--json` cleanup report was never written. **Ruling: not evidence of record.** **(3) A read-only formatting command** — SPIKE-002's reviewer ran a sub-second `python -c` one-liner pretty-printing two committed `pwml_ir_report.json` files outside the wrapper, and disclosed it (`SPIKE-002-REPORT.md:298-302`; `:276-280` before this correction commit). **Ruling: it was not itself a test, a benchmark, a pipeline leg or an LLM-backed command, and therefore was not a G11 test-run violation.** **The accurate statement, which is narrower than the retracted one:** the **final evidence of record** for the pre-A0 sequence is wrapped and reported zero surviving owned processes — *not* that the sequence ran perfectly. Procedural nonconformance during a sequence and a later clean evidence-of-record rerun are different things, and only the second is being claimed here. |
| 2026-08-05 | **D-1 CORRECTION, part 2 — what the repository can and cannot support.** **Durable cleanup reports were not committed for the historical jobs of the pre-A0 sequence**, so **their cleanup claims cannot be independently reconstructed from the repository.** This is the closeout reviewer's own position: it could show that the ledger's universal G11 claim was contradicted by another sprint document, but it **could not establish from the repository what actually happened** on any individual job. No retrospective G11 evidence has been generated, re-run or inferred to close that gap, and none should be — a reconstruction produced after the fact would not be evidence of the original run. Standing consequence for the rest of the sprint: a G11 claim is only as good as a committed `--json` cleanup report, and the pre-A0 sequence does not have one per job. |
| 2026-08-05 | **D-1 CORRECTION, part 3 — two retroactive product-owner rulings.** **(a) H-002's initial rejected review should have triggered the stated stop condition.** It did not; a correction round was run instead. The completed correction is **accepted retroactively**, on these grounds only: it remained within H-002, it received final substantive approval, it changed **no production code**, and it passed wrapped verification. **This does not authorize future automatic correction rounds after a stop condition has been reached.** **(b) The narrow, text-only H-002 boundary extension is ratified** — the module-docstring amendment recorded in the "Boundary extension granted" entry above — on these grounds: the executable AST was unchanged, and the edit corrected a **false test description**. **It creates no general authority for the Lead Orchestrator to expand a task's ownership boundary.** |
| 2026-08-05 | **D-2 CORRECTION (medium) — H-002's landed scope.** The H-002 row's Ownership boundary column recorded only the *stated* boundary (one function + a text-only docstring extension). It now records what **actually landed** at `e5eeb8c`: **7 new tests, 4 new module-level helpers, 3 new module-level constants, 351 insertions / 12 deletions**, all inside the one owned file. This is a **departure from the stated one-function boundary** and is recorded as one. It is *also* true that the tests were **required by G9** — a regression test that fails on the base SHA is mandatory. Both facts are recorded; neither excuses the other. |
| 2026-08-05 | **D-3 CORRECTION (medium) — H-002 coverage was a trade, not a strict increase.** The struck sentence in the H-002 entry above ("net coverage against the base is a strict increase, not a trade") is wrong. It **is** a trade: **nine `stage0_attempts.json` files under `runs/2026-08-02_2130` moved from failing a test to failing nothing.** All nine carry `key_compounds`; the base assertion at `aa0fb0a` globbed `runs/**/*.json` under `_MAX_PAYLOAD_BYTES` and asserted `carriers == []`, so all nine were carriers failing it, and the narrowed assertion excludes them **by name** (`_STAGE_ZERO_DIAGNOSTIC_FILENAMES`). That exclusion is a **deliberate, documented exemption of one filename class**, authorized by **D-013** (scope the replay assertion to payload files) and stated in the module docstring. It is **offset**, not cancelled, by the restored unlisted-carrier tripwire and the 7 new tests. Recorded as a documented exemption with an offset — deliberately not re-described as a win. |
| 2026-08-05 | **D-5 CORRECTION (low) — H-001 exceeded the `[S4]` size stop condition, unrecorded.** H-001 landed **≈637 changed lines** (591 insertions + 46 deletions at `aa0fb0a`) against the **~400-line `[S4]` bound**, which obliges an implementer to stop and propose a split. It did not stop, and **the overrun was not recorded as a deviation at the time**. It is recorded as one now, in the H-001 row. Mitigation, stated as mitigation and not as excuse: **212 of the 237 excess lines are generated manifest data** — the whole of the new `tests/data/baseline_cohort_manifest.json`, whose 39 entries were verified before freezing. Net of the manifest the branch still changed ~425 lines, still over the bound. |
| 2026-08-05 | **D-4 / D-6 CORRECTIONS (low).** **D-4:** three imprecise line citations in `SPIKE-002-REPORT.md` corrected in place, each re-verified against the source at `1c2dbee` (`def` line and last line read for all three): `PathWhizCompoundResolver.resolve` `db_resolver.py:279-305` → **`:278-423`**; `resolve_entity` `ir.py:1400-1416` → **`:1400-1433`**; `PathBankDbResolver.from_env()` `map_ids.py:819-873` → **`:820-873`**. A correction table in that report's §9 records the before/after. No other range in that report was touched. **D-6:** the Wave A0 prose said A0 was "no longer blocked" on H-001/H-002 while all nine Blockers cells still read `H-001 ✔, H-002 ✔`. All nine now read **product-owner approval**, with the cleared dependencies shown as cleared; the Blockers column legend now states that a dependency which has cleared does not belong in that column. The Deps columns (C-011 in Wave A0, C-033 in Wave B) are untouched — a satisfied dependency is still a true dependency. |
| 2026-08-07 | **H-006 ACCEPTED**, merged at `d167f93738a423e97894051b340f21cfa2a8b309`. Branch `agent/h06-wrapper-report-identity` carries **three commits above `ad917c2`**: `4afcc6d` (the implementation — report `schema_version` with a stated bump discipline, and a `wrapper_build` identity digesting the **executing** module via `os.path.abspath(__file__)`, defeating cross-checkout execution, rebase/squash and stale-wrapper attribution; repository SHA and dirty state recorded as context, never as a substitute), `5e01428` (repairs three `bounded_run.py` citations the branch itself invalidated — `CleanupReport` `:507-508`, `emit_json_report` `:1016`, the guarded child-log unlink `:1007-1010` — and documents raw-byte/CRLF/`core.autocrlf` byte identity), and **`9553cd28f457a31fbc2411da5e7ca4d531795843`, the reviewed commit** (replaces a Git-blob verification command that was silently wrong under Windows PowerShell 5.1, because the native pipeline re-encodes the stream). Prospective only: no historical H-003/H-004/H-005/O-1 artifact was edited or regenerated, and the 16 pre-H-006 reports remain schema 0 and still validate. **Reviewer verdict: exact unsuffixed `APPROVE`**, from a fresh reviewer against the complete diff from `ad917c2`, which **explicitly did not rely on either of the two earlier verdicts** (`CORRECTION`, `CORRECTION`) and re-derived every claim, including building its own base-SHA regression proof with a shim supplying the missing symbols so cases 9–11 fail on **artifact content, not symbol absence**, and proving item 7 by an AST-level diff showing `run`, `snapshot_processes`, `descendants_of`, `emit_json_report`, `main` and `_JobObject` byte-identical to base. **Test-runner, against that exact SHA:** selftest **12/12** with **12/12 cleanup reports** and **0 proved survivors**; `g11_evidence` selftest **3 passed / 0 failed**; evidence check **26 artifacts / 0 non-compliant**; smoke **457 passed**; every job `cleanup_success: true`, `final_surviving_count: 0`. The README verification recipe was confirmed in **Windows PowerShell 5.1, Git Bash and `cmd.exe`**, all three reproducing the executing wrapper at 46 712 B `sha256:69f9f1b5…baad5` and the Git blob at 45 620 B `sha256:ffd5b424…fd98`. **Procedural fault, recorded and NOT cured:** the branch exceeded its original estimate (572 hand-authored lines vs the ~400 `[S4]` bound, 43% over) and **a commit was created before renewed authority**, which `[S4]` forbids. The overrun was self-declared, and the reviewer judged the acceptance-criterion atomicity argument sound — a card-scoping defect rather than implementer scope creep — but **atomicity explains the size without excusing the failure to stop before committing, and the later approval and acceptance did not retroactively cure it.** |
| 2026-08-07 | **INCIDENT — D-014 and TRAP-5 violated by `42e21db5c57bfcb7f1db6ab0956645ff7b76da7b`** (parent `ad917c23ef667794320d74fb97771ca49c4b847a`), subject *"uipdating the repo after a run"*. **Exact ten-file manifest, re-derived from the commit object:** `data/enrichment_cache.json`, `data/id_mapping_cache.json`, `out/enrichment_dump.json`, `outputs/pathway.pwml`, `tmp/draft_graph.json`, `tmp/qa_report.json`, `tmp/reaction_summary.txt`, `topics_flip_strict.txt`, `topics_regression_research.txt`, `topics_verify_subset.txt`. **All seven D-014-protected tracked scratch files were committed, and all three protected untracked topic files were committed.** `data/enrichment_cache.json` was committed with a **37 993 923-byte (≈37 MB) blob**, violating **TRAP-5** ("no branch may commit a cache modification"). **The commit was pushed to `origin/sprint/pwml-recovery`.** **Containment**, product-owner-authorised and executed in this order: an explicit **full-SHA `--force-with-lease`** move of the remote integration ref from `42e21db…` to `ad917c23ef…` (no plain `--force`, no weakened retry, no backup ref created), verified by `git ls-remote`; then **one authorised `git reset --mixed ad917c23ef…`** for the local branch and index only. A **42-file byte-length and SHA-256 manifest** — the seven tracked scratch files, the three `topics_*.txt`, and all 32 files beneath the sixteen protected `cache_snapshot/` directories — was taken **before** and re-verified **after**: **0 differing, 0 missing**, proving the containment operation itself preserved every protected byte. **Attribution: none can be established.** Git author and committer metadata alone cannot establish who or what executed the commit, and no attribution to a person is made here. **Product-owner rulings:** **D-014 — violation confirmed, contained, NOT waived, and NOT retroactively cured.** **TRAP-5 — violation confirmed, contained, NOT waived, and NOT retroactively cured.** The guarded ref repair and the mixed reset were **one-time, incident-specific containment exceptions and create no general exception** to the standing no-force-push and no-reset rules. **Scope of the accepted recovery:** it concerns **branch ancestry and protected-byte preservation only.** Removing the refs did **not** retroactively cure either violation, and it **does not prove** the Git object was physically deleted from GitHub or from local object storage; **no storage-reclamation claim is made.** |
| 2026-08-07 | **Commit B / D-019 ACCEPTED**, merged at `b063d14d8d5a459aed07b9a3b038b75bc6419813`, parents `a77cc4fdbdf94e134ed4b2a0c1082b87d1d40315` and `b2fc12f4337263cb8f83753a2936f761eea8aab6`, merge tree `f1856434897d34e0704a3fa5bcdd740d5025150c` identical to the reviewed tip `b2fc12f`. Non-fast-forward; pushed with an ordinary non-force push and confirmed on `origin/sprint/pwml-recovery` by `git ls-remote`. Branch `agent/s4-budget-repair`, **two commits** above `a77cc4f` — `999f71e` (the `[S4]`/`_TEMPLATE_IMPLEMENT.md`/`DECISIONS.md` repair) and `b2fc12f` (removal of the third live threshold from `.claude/agents/pwml-implementer.md`, which the first review caught) — neither squashed, rebased, amended, replaced nor cherry-picked. Control-plane only, hand-authored only: **four files, 46 insertions / 3 deletions = 49 changed lines** (`pwml-implementer.md` 10, `DECISIONS.md` 12, `_SHARED_BLOCKS.md` 17, `_TEMPLATE_IMPLEMENT.md` 10), **0 generated artifacts, 0 generated bytes**, no binary change, no rename. **Reviewer verdict: exact unsuffixed `APPROVE`**, from a fourth, fresh reviewer told every prior verdict carried no weight, which ran its own repository-wide sweep including `.claude/` and measured **3** live threshold sites at `a77cc4f`, **1** at `999f71e`, **0** at `b2fc12f`. **D-019 is LOCKED** (`DECISIONS.md:367`) and effective on this merge. Prospective only: it does not cure, waive or reclassify **H-006** or any earlier event, **D-017 is unchanged and not reopened**, and every historical `~400` record stands exactly as written. |
| 2026-08-11 | **Wave A0 status reconciled** under `CONTROL-PLANE-RECONCILE-001` §5, correction round 1. The Wave A0 table recorded all nine merged cards as `BLOCKED` on "product-owner approval, which has not been given", with empty Base SHA, empty Merge SHA and nine worktree paths that do not exist on disk — a factual error in the file `CLAUDE.md` designates the single source of truth for what is dispatched and merged. All nine rows now read `MERGED` with derived values: **Base SHA = the first parent** of each merge (`git rev-list --parents -n 1`), giving C-010 `9e06360`, C-011 `85fae43`, C-012 `361b158`, C-013 `72ee20f`, C-014 `09fb40d`, C-015 `c832894`, C-016 `8b4bc0c`, C-017 `729c40e`, C-018 `fc8b059`; **Merge SHA** `72ee20f`, `0182eae`, `9e06360`, `09fb40d`, `c832894`, `8b4bc0c`, `729c40e`, `fc8b059`, `85fae43`; and **Worktree = the real attached path** from `git worktree list` (hashed `agent-a…` names, all nine still attached). **Cross-check performed, not assumed:** each attached worktree's HEAD equals the **second parent** of its own merge — `d784747`, `066fb6b`, `def6adb`, `fb5a75a`, `9c09ee8`, `101d25c`, `a0aaa56`, `9479c2d`, `bdc006d` — so the reviewed commit and the merged commit are the same object in every case. Focused and Integration cells carry the measured results from each merge record; the planned reviewer is preserved and the actual outcome appended. **Two items flagged rather than filled in:** C-010's and C-012's clearing authorization is named in no merge message and exists nowhere under `docs/pwml_recovery_sprint/` (`WAVE-A0-BUDGET-RATIFICATION-001` cannot be it — the cards state it grants "no implementation approval, no unblocking, no branch, no dispatch authority"), and no milestone benchmark has run, so every Bench delta is planned rather than measured. **Factual reconciliation only:** no verdict re-litigated, no merge message or historical entry rewritten, no production or test file touched. C-011's forward Chunk D gate is recorded as an **obligation, not a citation**: a split-process gate is required to be defined in `TEST_MATRIX.md` by the reconciliation's Chunk D lane per **D-020**, and is **not in effect until that definition merges** — verified at `0182eae` that `TEST_MATRIX.md` contains no "split-process" and `agent/recon-chunkd-gate` carries no commit. |
| 2026-08-11 | **H-007 closeout** under `CONTROL-PLANE-CLOSEOUT-002`, branch `agent/h07-closeout` from `08d5d07`, one new worktree, the existing 27 untouched. **(1) G11 unmatched-selector false green fixed.** `check --task <nonexistent>` resolved to `[]` and `check_many([])` exited **0**, certifying a task with no committed evidence; an explicit path beside it dropped `--task` entirely, so one valid selector concealed an unmatched one. `resolve_selection` now returns artifacts **and** unmatched selectors; `check` names each, classifies `unmatched_task` vs `malformed_task`, and exits nonzero. Whole-tree behaviour, the clean-run summary line, `allocate`'s own malformed-task rejection and every `check_report` rule are unchanged. **G9 is behavioural, not symbol-absence:** at `0182eae` the unmodified CLI exits 0 on both cases; at the candidate both exit 1. **(2) Chunk D execution partition is now per NODE.** Per-**file** isolation was measured insufficient — one process still built 23 `AppTest` objects, the documented cause — and per-**node** isolation had never been tested. `run` executes the 150-test core in one process and each of the **27** AppTest node IDs alone in a fresh process, proving the partition **177 = 150 + 4 + 23** on every invocation and comparing the executed node-ID set to the collected one. The verdict parser was rebuilt: `passed == expected` **and** zero failed/errors/skipped/xfailed/xpassed/deselected **and** exit 0 **and** the wrapper's own report reading `completed` / `cleanup_success` / 0 survivors. New `chunk_d_gate_selftest.py`, **10/10**. **(3) Six scheduled runs, order declared and committed before any ran** — three against a clean export of `0182eae`, three against the candidate: **base 1 green / 2 red, candidate 1 green / 2 red**; core+`s8` green in all six; every failure in the `qb` cohort, three different symptoms, and a declared 18-run per-node diagnostic returning 17 passed / 1 failed. Since `tests/` and `src/` are **byte-identical** at both trees the diff cannot cause a Chunk D outcome in either direction, so this is a **test-infrastructure race**: the **154 deterministic tests block**, the **23-node `qb` cohort is mandatory-to-run and mandatory-to-report but temporarily non-blocking**, and the race is backlog item **BL-003**. **(4) Live record corrected** — the C-011 row's "that definition has not merged" (true at `0182eae`, false from `69d4069`), the obsolete 457 smoke figure, Chunk D timing split into deterministic-core vs complete-gate, and the historical `qb` sample re-cited exactly against committed artifacts, with the parts that rest on no committed report said plainly. **(5) D-022 appended** — D-020 and D-021 are not rewritten — defining `G11-EVIDENCE-ACCOUNTING-001` and `CONTROL-PLANE-RECONCILE-001`, setting the measured ceilings **C-010 = 294**, **C-012 = 321** with no unused headroom, and binding the two retry seams to **BL-001**. **261 G11 artifacts, 0 non-compliant, 0 surviving owned processes throughout.** Zero files under `src/`, `tests/`, `batch/`; `bounded_run.py`, the C-011 fixture and `.gitattributes` untouched. |
| 2026-08-11 | **`CONTROL-PLANE-RECONCILE-001` correction round 2**, after an independent review returned `REJECT` on two blocking findings, both upheld. **(1) D-021 §2 was not exhaustive.** It inherited `MASTER_PLAN` §9 `:367`'s two-function list for C-040 while the authority §2 itself cites — `SPIKE-002-REPORT.md` §3 — enumerates **four** moved functions plus a mandatory reference repair. `_normalize_compound_external_ids` (`ir.py:530-555`), `_compound_external_ids` (`:558-575`) and `_emit_canonicalization_preflight` (`:900-963`) were owned by no row anywhere in the control plane, so a C-040 implementer would have escalated on day one — the dispatch error §5 exists to prevent. All are now assigned in §2; the first two to C-040 as moves, the third to C-040 as a **minimum-surface `:920` re-import repair only**, with the function's behaviour explicitly out of every boundary. `_entity_record` (`:437-449`) recorded as staying, owned by no card. **Nothing frozen** — SPIKE-002 §3's own move/stay classification settles each, with no product choice involved. Every line number re-verified against `src/t2pw/pwml/ir.py` at `0182eae`. **(2) D-020 locked a forward gate that does not exist.** It stated present-tense that the authoritative Chunk D gate "is the split-process Chunk D gate defined in `TEST_MATRIX.md`"; verified at `0182eae`, `TEST_MATRIX.md` contains **zero** occurrences of "split-process" and `agent/recon-chunkd-gate` carries **no commit**, so the gate exists in no ref. Restated as an **obligation with an effective date** — required under `CONTROL-PLANE-RECONCILE-001` §3, **not in effect until that definition merges** — with the merge-order constraint recorded. Same treatment applied to the C-011 row and the round-1 timeline entry. **Also corrected:** D-020 fact 2 now states its provenance — the `85fae43` integration control has **no committed G11 artifact**, its report having been written to the durable checkpoint outside the repository because the C-011 branch manifest was closed; it rests on the merge record and an out-of-repository wrapped report. `C-013.md`'s C-030 binding re-pointed from `streamlit_app.py:3604` to **`:3659`** (C-011 moved the write; `:3604` is now `sbml_overwatch_report`). The A0-C2 and C-014 sweep claims restated as "no card owns" rather than "zero occurrences", which the same diff falsified. The seven `WAVE-A0-RESUME-002` / `C-011-CLOSEOUT-001` blocker cells marked **dispatch attribution inferred** (note †) — those IDs are cited only for ceilings, the same citation shape rejected for `WAVE-A0-BUDGET-RATIFICATION-001`, kept only because that one alone carries an explicit no-dispatch-authority disclaimer. **`G11-EVIDENCE-ACCOUNTING-001` recorded as cited-everywhere / defined-nowhere** and routed to the product owner; no definition written. Wave A1's two Blockers cells cleared to **product-owner dispatch**; Waves B/C/D have no Blockers column and their `Deps` cells are untouched per D-6. Documentation only; no verdict re-litigated, no history rewritten, no `src/`, `tests/`, `TEST_MATRIX.md` or `.gitattributes` touched; no python executed. |
| 2026-08-13 | **PACK 2 OPENED — product-owner authority recorded, four writers dispatched.** Integration tip at dispatch `bcc0bfe3cdbd02f731fda09adbcf655e525a3345`, verified local = `origin` = `ls-remote`; takeover checks all matched (whole-tree G11 **747 artifacts / 0 non-compliant**, protected manifest **42/42**, `outputs/pathway.pwml` **34 878 B**, 11 dirty-but-protected scratch entries, empty index, no merge in progress, 43 attached worktrees). **Decisions ratified by the product owner and applied to the live control plane** (historical evidence NOT rewritten): **(1) A0-C1 census ratified at 49, not 60** — F-014's three independent censuses over all 32 committed `final_mapped.json` (compounds 38, protein_complexes 11, all tier 4, 19 distinct files, corpus drift excluded); C-030 already discharged against all 49. **(2) C-034 re-scoped** off the dead `extraction/extract.py` demo onto `pipeline/stage_one_boundary.py :: settle_stage_one`, lineage writes only, ownership disjoint from C-042; the dead file is now explicitly excluded and is NOT deleted. **(3) C-044 authorized** — its C-033 dependency is satisfied at `7ff1211`. **(4) C-042, C-045, C-050..C-057, C-060, C-061 authorized** subject to real dependencies and function-level ownership. **(5) C-060 scope accepted (R-003 M1 only)** — NEW `pipeline/entity_admission.py` + minimal call site in `pipeline.py :: merge_additions`; **A0-C5 binds — the hallucination gate runs first and independently of `cofactor_policy`**. **(6) C-061 scope accepted (R-004 B-2 only)** — `rag/admission.py :: parse_span_relation`, `validate_evidence_span`; **the stale C-035 claim over both symbols is corrected in `MASTER_PLAN` §9 and in C-035's row. C-035 is NOT reopened** — it merged with both functions byte-identical, verified three times including in the merged tree, precisely to keep them free. **(7) F-006 / D-002 assigned to a new narrow follow-up card `C-041a`**, owning only the release/refusal seam around `strict_quarantine.py :: quarantine_and_close` (incl. the live refusal near the previously reported `:1959`), so a subthreshold but structurally valid pathway becomes `review_required` output rather than an unconditional refusal. **(8) Minimum PWML/JSON-side wiring authorized for C-045** so it is operational before T-102; an unwired helper is not completion. **(9) C-050 and the later freeze-lifecycle cards authorized**, preserving the C-030/C-050/C-051/C-052 serialization. **(10) T-100..T-105 and Gate 5 authorized at their documented milestone boundaries only.** **Still prohibited:** resolving **O-1**; any SBML functionality, extension, refactor or taxonomy implementation (C-052 may read and assert the existing SBML input binding but may not modify SBML generation; if T-102's organism dimension is unreachable solely because SBML lacks taxonomy annotation, that exact limitation is recorded truthfully and the other benchmark dimensions continue); opportunistic cleanup swarms; reopening accepted cards without evidence of a genuine regression. **Dispatched concurrently from `bcc0bfe` into four new isolated worktrees, disjoint function-level ownership, one writer each (O-2s check performed — every created worktree received a writer):** C-034 `agent/p21-lineage-extract` (850) · C-042 `agent/p03c-extraction-ladder` (1300) · C-044 `agent/p26-lineage-mapping` (900) · C-050 `agent/p05b-prefreeze-call` (1200, **`qb` required before merge per D-023**). Ceilings sized per acceptance clause per F-015/O-1s, not anchored on a peer card. Charters and the shared block are durable external records under `scratchpad\sprint-records\`. Control-plane edit only: no `src/`, `tests/`, `TEST_MATRIX.md`, `.gitattributes`, fixture, cache or protected file touched; no verdict re-litigated; no historical entry rewritten. |
| 2026-08-13 | **PACK 3 OPENED — takeover verified, three product-owner rulings applied, four writers dispatched.** Integration tip at takeover `472293c9265fcc45d4af9f87f7ae25707af11f24`, verified **local = `origin` = `ls-remote`**; `b268121` and `bcc0bfe` both confirmed ancestors; `outputs/pathway.pwml` **34 878 B**; protected manifest **42/42** (16 run dirs + 16 `cache_snapshot/` + 7 scratch + 3 `topics_*.txt`) with **zero protected paths staged**; index empty; no merge in progress. **One delta from the handoff's expectation, recorded as P3-01 and judged non-blocking:** whole-tree G11 reports **884 artifacts / 1 non-compliant**, not 0 — `docs/pwml_recovery_sprint/evidence/g11/C-042r/09-r1focused.json` is a bare `{"g11_reserved": true}` reservation stub (`report_never_written`) that the C-042 reviewer allocated and never ran, in **untracked** evidence for an **already-accepted, already-merged** card. It threatens no git history, no protected data, no accepted merge, no integration-branch state and no PWML or biological correctness, so it meets none of the handoff's four stop conditions. **Not deleted** (standing instruction preserves the reviewer evidence dirs) and **not filled** (writing a report for a job that never ran would be fabrication). **Binding consequence: for every Pack 3 merge gate the whole-tree G11 baseline is 1 known non-compliant — this stub — not 0; a gate is clean when the check reports exactly this one entry and no other, and any second non-compliant artifact is a real failure.** Resolve at the next natural closeout with the other Pack 2 evidence dirs (`REV-034/`, `C-042r/`, `C-044r/`), which are preserved and **must not** be retroactively inserted into accepted card merge commits. **Product-owner rulings applied: (1) C-050 released from B-P2-1** via a new narrow **test-only** card **C-050a** owning the node15 quarantine-boundary comparand; C-050a merges first, then C-050 is reviewed against the corrected combined state with one complete 23-node `qb` run and all 23 nodes classified, and **C-050 is not to be faulted merely because its original base lacked C-050a**. The corrected invariant: *the quarantine boundary must forward unchanged the artifact it actually receives after enrichment, rather than equal an earlier pre-enrichment artifact*. **No production code is authorized** — and none is needed: `streamlit_app.py:3725` already publishes `"final_mapped_enriched": final_export_payload`, which is exactly the object `:3605` hands to `freeze_canonical_payload`. **(2) C-041a ceiling 1500 → 1900** by final product-owner **pre-commit** re-charter, valid because the branch still carries **no implementation commit** (verified: `agent/p08a-d002-release-seam` at `b268121`, 302 insertions / 40 deletions uncommitted in its existing worktree). Scope, production ownership, behavioural scope, acceptance clauses and generated-evidence budget **all unchanged**; the existing worktree is **resumed, not recreated**; valid existing bounded evidence is **reused rather than ceremonially re-run**. **(3) C-042 stays merged, unamended and unreopened**, and its accepted review is **not** characterized as having failed; its reviewer's LOW finding 6 is carried forward as **C-042a**, establishing a seventh termination reason `attempt_cap_reached` under new decision **D-024**, with `OPERATIONAL_TERMINATION_REASONS` explicitly **unchanged**. **P2-11 resolved — `tests/test_streamlit_quarantine_boundary.py` now has explicit test-function-level ownership** (`MASTER_PLAN` §3, hotspot 9): C-041a owns its four re-pinned `report["ok"]` assertions and `_payload_with_no_viable_core`; C-050a owns node15 and its minimum supporting fixture. Four standing rules bind every future card touching the file — **no card may add, remove, rename or reorder a test function** (the `qb` gate addresses nodes **positionally**); it is validated only through the 23-node gate and sits behind **BL-003**, so **never read a red as expected**; **two cards owning disjoint functions here must not be merged blind of each other**, and whichever lands second re-runs `qb` on the combined state; and a correction must **preserve or strengthen** the stated invariant — re-pointing a stale comparand is in scope, weakening the comparison is not. **Dispatched concurrently, disjoint function-level ownership, one writer each (O-2s check performed — every created worktree received a writer):** C-061 `agent/p52-missing-reactions` (900, from `472293c`) · C-041a `agent/p08a-d002-release-seam` (**1900**, resumed in its existing worktree from `b268121`) · C-050a `agent/p05c-node15-comparand` (800, from `472293c`) · C-042a `agent/p03d-attempt-cap-reason` (1000, from `472293c`). **C-060's charter is written and dispatch-ready**, held only by the four-lane cap: its "serialize behind C-042" constraint is discharged (C-042 merged `8917349`) and the 8 000-line-file adjacency risk was **measured, not assumed** — `merge_additions` at `:1130` versus C-042a's possible consumer touches at `:3531`/`:3558`/`:3722`/`:3971`. **C-045 remains blocked** on C-050's pre-freeze call site existing; **C-051** follows C-045 and its charter **must** name `ir.py :: _entity_record` (P2-06); **C-052** follows C-050/C-051. Control-plane edit only: no `src/`, `tests/`, `TEST_MATRIX.md`, `.gitattributes`, fixture, cache or protected file touched; no verdict re-litigated; no historical entry rewritten; **D-024 deliberately left unwritten — it belongs to C-042a's diff.** |
| 2026-08-13 | **PACK 3 — two merges, eight product-owner rulings applied, G11 restored to zero, three lanes dispatched.** Integration tip `472293c` → `3197171` (control plane) → **`758b312`** (merge C-042a) → **`be7f4c2`** (merge C-050a), each pushed and verified local = `origin` = `ls-remote`. Takeover re-verified at `be7f4c2`: empty index, no merge in progress, `outputs/pathway.pwml` **34,878 B** blob `9f7110ba`, protected manifest **42/42**, all of `758b312`/`3197171`/`472293c`/`bcc0bfe` confirmed ancestors, 2 pre-existing `python.exe` reported and never killed. **RULING 1 — G11 RETURNED TO ZERO. The prior session's position that merges could baseline whole-tree G11 at one non-compliant artifact is REJECTED; every prospective merge requires `0 non-compliant`.** The single offending object, `evidence/g11/C-042r/09-r1focused.json`, was verified **untracked** two ways (`git ls-files --error-unmatch` errors; `--others` lists it, so it is in no commit and moving it cannot alter history), verified to be **only an unused reservation** (three keys — `g11_reserved`/`task`/`label` — and **none** of the ~28 fields a completed `bounded_run.py` report carries: no `command`, `duration_seconds`, `exit_code`, `descendants_observed`, `final_surviving_count` or `cleanup_success`, because no process was ever launched under it), hashed **`sha256:b34225553e197d1f26a420efa7b61da88e26d6c9d5ac5c8eae6f0098bc4361d8`** (74 bytes), and **moved out of the repository** to the durable Pack 3 scratch record at `scratchpad/sprint-records/p3-01-quarantined-reservation/`, hash verified identical at source before and destination after. **No run was fabricated and no history was deleted.** Full disclosure — original path, hash, reason, and explicit confirmation that **no test ever ran for it** — is recorded in that directory's `RECORD.md`. **No completed G11 report was moved, deleted or altered:** `C-042r/` held 14 files and holds **13**, with `01`–`08` and `10`–`14` byte-identical and in place; `REV-034/`, `C-044r/`, `MERGE-*/`, `REV-041/`, `REV-050/`, `REV-061/` untouched. It was also confirmed the **only** abandoned reservation tree-wide (`grep -rl g11_reserved` returns otherwise only the allocator's own source, README and pycache). **Whole-tree G11 after correction: 989 artifacts, 0 non-compliant.** Standing rule: allocate a G11 path **only when the job is ready to start**; an abandoned reservation must be resolved before closeout and **may never become a permanent non-compliant baseline**. **RULING 5 — C-050a's generated-evidence overrun is recorded as a disclosed deviation, not compliance** (41 against 40; cause was the mandatory gate auto-allocating 32; no code/PWML/biology impact; accepted code not reopened, merge not rewritten) — see its row. **RULING 4 — C-041a proceeds under a one-time, disclosed process waiver.** Its charter omitted the D-019 generated-evidence ceiling entirely, so **D-019 was NOT formally satisfied for that dimension and this is not claimed as compliance**; abandoning an independently approved implementation solely to regenerate the same mandatory reports would add cost without improving code confidence. The waiver holds only while the reviewed tip remains exactly **`4177fe57`** with the approved diff unchanged, hand-authored work within the **1900** ceiling, all evidence genuine and no report discarded to manufacture compliance; the actual generated-evidence count and size are to be recorded before merge. **Not retroactive budget authority and not precedent.** If C-041a changes after `4177fe57`, the changed exact tip goes through independent re-review and the correction receives a **prospective** generated-evidence budget before any new evidence-producing job. **RULING 8 — benchmark order.** `REV-041` established that after C-041a merges a subthreshold strict leg completes, `_finalize_pwml_export` sets `status=pass`, `pathway.pwml` is written, and `bench/acceptance.py` (`_STRICT_DELIVERABLE` `:88`, `ModeResult.passed` `:161`) — which contains **no reference to `release_status`** — would score it as strict success, contradicting D-002 and § 7. **No strict benchmark success figure may be quoted between the C-041a merge and C-053/C-056b landing; C-053/C-056b are prioritized before T-100; diagnostic runs are permitted only if clearly labelled non-acceptance.** **RULING 6 → D-025** (three separate ceilings per charter, with auto-allocated gate reports budgeted before dispatch) and **RULING 7 → D-026** (tracked background execution authorized under eight named conditions), both appended to `DECISIONS.md`; `TEST_MATRIX` § 0 rule 1 amended in the same commit so the C-034 / C-041a / C-050a background-job question is not relitigated a fourth time. **RULING 2 — C-061a authorized** on the rejected C-061 tip `ceb1ab4d`; its charter was updated to record that the monotonicity argument ("a post-filter can only remove readings") is **explicitly rejected as proof by itself**, since removing readings could also delete a legitimately recovered relation — and the correct MenA reading is itself a **superset** of the drop-readings being eliminated, so a slightly wrong filter deletes the very reading the card exists to recover. The safety case must be **empirical**. **C-061's rejected tip will never be merged alone:** the reviewer reviews the **entire combined diff** from C-061's base through C-061a's tip, and the C-061a tip is merged **once** with `--no-ff`. **RULING 3 → C-050b authorized** (deterministic offline curator mode). **Three lanes dispatched, disjoint ownership, one writer each (O-2s check performed — every created worktree received a writer):** C-060 `agent/p51-false-id-repairs` (1200 / 50 artifacts / 3 MB, from `be7f4c2`, **dissolution check is its first act**) · C-061a `agent/p52a-relation-subset-filter` (500 / 25 / 1 MB, from `ceb1ab4d`) · C-050b `agent/p05d-offline-curator` (450 / 20 / 500 KB, from `be7f4c2`). **C-050b was scoped after inspecting the merged source, per ruling:** the switch lives in `curation/pathway_curator.py :: run_pathway_curator` and touches **no** `streamlit_app.py`, which keeps it fully disjoint from unmerged C-050 (whose only `streamlit_app.py` hunk is 23 insertions at `:3569`) and from C-052. It is **forbidden** to run a `qb` cohort, to edit `.env`, to handle the live API key, or to use connection failure or a swallowed exception as the offline mechanism. Control-plane and evidence-hygiene only in this commit: no `src/`, `tests/`, fixture, cache or protected file touched; no verdict re-litigated; no historical entry rewritten; no accepted merge reconstructed. |
| 2026-08-14 | **D-027 recorded; C-051's D-021 conflict ruled on; three lanes relaunched after the quota reset.** Narrow state check re-verified at `e5b83ff`: local = `origin` = `ls-remote`, empty index, no merge in progress, `outputs/pathway.pwml` **34,878 B** blob `9f7110ba`, protected manifest **42/42**, **whole-tree G11 989 artifacts / 0 non-compliant**, **zero abandoned G11 reservations** in the primary or any worktree, and all six card worktrees clean at their recorded tips (`dad0789`, `4177fe5`, `768be75`, `a69d2cc`, `be7f4c2`, `be7f4c2`). No owned surviving processes. **No completed implementation or review work was rerun merely because the quota reset.** **D-027 — conditional C-051 ownership of the post-freeze identity seam.** The conflict was real: **D-021 § 2 (LOCKED) assigns `ir.py :: _entity_record` to "no card — untouched"**, while **P2-06 (HIGH)** measured that `pathwhiz_id` is materialized **there** and not in `_resolve_compound_rows`, so deleting the resolution call — all C-051 is chartered to do — could leave post-freeze materialization reachable, which is exactly what merge rule 8 forbids. D-021 § 2 **remains locked except** for one narrow conditional carve-out: C-051 may **inspect**, and **only when proven necessary** modify, the `pathwhiz_id` materialization logic inside `_entity_record`, amended **only to the extent required to enforce the already-locked rule that identity may not be created or resolved after freeze**. **C-051 stays blocked until C-050 and C-045 merge, and C-045 and C-051 must not run concurrently** because `_canonicalize_species_offline` is already called from **inside** `build_pwml_ir`. Before any implementation commit C-051 must re-derive symbols by AST, trace the four symbols on the combined tip, measure post-freeze reachability, and exercise four named cases (valid pre-freeze identity · absent at freeze · identity only in mapping metadata · a normal compound through the live production chain). **If unreachable: change nothing, retain the D-021 lock, add a focused unreachability guard, and close P2-06 by reachability proof — a no-code result is a valid completion.** If reachable: modify only the `pathwhiz_id` block, forward or serialize a pre-freeze identity but **never resolve, infer, synthesize, hydrate or newly materialize** one, and route a missing pre-freeze identity through the existing missing-identity/review policy rather than inventing an identifier. **D-021's live symbol citations re-derived by AST and recorded in D-027; historical evidence NOT rewritten.** The drift is substantial and would have misled anyone quoting it: `_entity_record` `:437-449` → **`:438-450`**; `_emit_canonicalization_preflight` `:900-963` → **`:704-767`**; the resolution call site `:1106-1114` → **`:911`**; and C-040's four moved functions are no longer in `ir.py` at all — `_normalize_compound_external_ids` → `compound_resolution.py :198-223`, `_compound_external_ids` → `:226-243`, `_canonicalize_compound_offline` → `:246-311`, `_resolve_compound_rows` → `:314-421`. The shared seam is now explicit: inside `build_pwml_ir` (`:770-1811`) the three call sites run in sequence — `_canonicalize_species_offline` `:844` (C-045 moves it pre-freeze) → `_resolve_compound_rows` `:911` (C-051 deletes it) → `_entity_record` `:921` (materializes `pathwhiz_id` at `ir.py:447`). **Lanes relaunched into their EXISTING worktrees, nothing recreated:** `REVA-061` reviewing the **complete combined** C-061+C-061a diff `472293c → dad07895` (the rejected intermediate tip `ceb1ab4d` is **never merged separately**; the composite tip merges once with `--no-ff` on an exact bare `APPROVE`) · **C-060** `agent/p51-false-id-repairs` (1200 / 50 / 3 MB, dissolution check first) · **C-050b** `agent/p05d-offline-curator` (450 / 20 / 500 KB). `C-051-charter.md` written and aligned to D-027; **not dispatchable** until C-050 and C-045 merge. Control-plane only in this commit: no `src/`, `tests/`, fixture, cache or protected file touched; no verdict re-litigated; no historical entry rewritten; no accepted merge reconstructed. |
| 2026-08-14 | **C-061+C-061a composite MERGED `afcbf1d`; C-050b MERGED `1383624`; deterministic `qb` gates are now explicitly offline.** **C-061+C-061a** — `REVA-061` bare **APPROVE** on the **complete combined diff** `472293c → dad07895`; the rejected intermediate `ceb1ab4d` was **never merged separately**, per RULING 2. Owned-diff equivalence byte-identical (`b2a5db2c`); SMOKE **460**, focused 8 files, whole-tree G11 **1016 / 0**, protected **42/42**. The reviewer went past the writer on every load-bearing claim: AST symbol hashes across all three revisions proved `_PROSE_PATTERNS` and `_EXTRA_PROSE_PATTERNS` byte-identical with only `_pattern_matches` and `parse_span_relations` moved; it rebuilt all three goldens itself and its regenerated `472293c` digests came back **byte-identical to the committed base golden**, discharging S9 trap 3 so the comparison is anchored rather than circular; the tip golden is identical to `ceb1ab4d`'s, so **the post-filter cost nothing — 115 gained, 0 lost, 0 churn**; a 32-case probe at all three revisions confirmed **symmetry restored** (first-drop and last-drop both refused, where C-061 admitted last-drop only). Decisively, it **scanned the raw `ceb1ab4d` census and found ZERO participant-subset pairs across 2000 spans, 462 of them multi-reading** — establishing that the filter removes nothing on this corpus **because there is nothing to remove, not because it is broken** — then supplemented with 12 condition probes, 20,000 randomized structural inputs and 8 constructed spans. It also **re-implemented the blanket connector exclusion the writer declined** and confirmed it does lose the legitimate MenG reading, validating that judgment call rather than accepting it; and it verified the catalyst-fragment deferred finding is **not** a live admission widening by measuring at `472293c`, where base already returns the enzyme-less reading and already admits an enzyme-less claim while a *wrong* enzyme is refused at both ends. `candidate_type_cannot_fill_gap` still carried by **70 of the 115** flips — the bio-audit's number, unmoved. **C-050b** — `REV-050b` bare **APPROVE**. Establishes **`T2PW_OFFLINE_CURATOR`**, opt-in and default-off, guarding `run_pathway_curator` after the payload load so that `chat_with_tools` and every per-call model/provider input sit **physically below the return** — structurally unreachable, and explicitly **not** a swallowed exception, forced provider, unreachable host or monkeypatch, all of which RULING 3 forbids as the mechanism. The contract is honoured under the flag: output written **byte-for-byte via `write_bytes(input_path.read_bytes())`** rather than re-serialized, report written, report returned. A new top-level **`"skipped"`** object records the deliberate disable — deliberately **not** the existing `"error"` key — so *off-on-purpose*, *ran-and-found-nothing* and *raised-and-swallowed* are discriminable **by key alone**, all three carrying `patches_accepted == 0`, which is exactly why counts were insufficient. `streamlit_app.py` untouched, keeping it disjoint from unmerged C-050 and from C-052. The reviewer **inspected the complete call path first, as RULING 3 requires**, then built its **own** base export, reproduced the A6 differential exactly, and **wrote a STRICTER criterion than the one shipped** — full ordered `t2pw` frame sequence, guard frames removed, exact equality with base — passing it with `STRICT_FULL_TRACE_EQUIVALENCE: True`, the `+1` frame being exactly `curator_offline_mode_enabled`, once, immediately after entry. It read `config.py` and confirmed the flag's truthy token set is **identical to the existing `_TRUE_TOKENS`**, so gate commands inherit no divergent truthiness rule, and confirmed the zero-call proof patches **the binding actually invoked**, not merely the source module. Gates: SMOKE **460**, focused, whole-tree G11 **1037 / 0**, protected **42/42**, owned-diff equivalence `deb9e1ee`. **`TEST_MATRIX` § Commands amended in this commit: `T2PW_OFFLINE_CURATOR=1` is now MANDATORY on every deterministic `qb` run, set in the bounded CHILD environment**, with the reason recorded inline so nobody re-derives it — and with the explicit note that a live-curator acceptance run is separate, bounded work requiring cost authorization and is **not** a deterministic gate. **Two deferred findings assessed, neither blocking:** `llm/client.py` builds its client **at import** and can **raise at import** on a misconfigured remote provider, so "no provider initialized" cannot mean "no client object exists" — only *no request issued, no per-call model resolution*, which is what the card claims; and the **default zero-patch path is not byte-preserving** (`read_text` + `write_text` + `json.dumps`), but the reviewer established this has **no canonical-hash impact** because curator bytes propagate only when `patches_accepted > 0` and canonical hashes are computed downstream of the freeze. |
| 2026-08-14 | **PACK 4 — control plane restored before any card merge; D-028 recorded; stale prospective merge cleared.** Integration tip `f2f7599`, local = `origin` = `ls-remote`. A prospective `--no-ff --no-commit` merge of C-050's **superseded** tip `768be750` had been held open across a machine crash and a session limit; it was verified narrow before removal (MERGE_HEAD exact; staged 63 = the C-050 contributor diff exactly, **0 extras**; no protected path staged) and aborted with product-owner approval. Post-abort verified: HEAD `f2f7599`, no merge in progress, **index empty**, `outputs/pathway.pwml` **34,878 B** `01c036c2`, protected manifest **42/42**, whole-tree G11 **1212 / 0 non-compliant**, zero owned processes, **zero abandoned G11 reservations**, reviewer evidence `REV-040a/` preserved. **D-028 (LOCKED) appended to `DECISIONS.md`** — DB match admission. An independent `pwml-bio-auditor` adjudicated a **`product_contract_violation`**: `compound_resolution.py` required `confidence >= 0.85`, logged `compound_db_resolution_failed` on failure, then **applied the resolution anyway**, while `db_resolver.py :: apply_compound_db_resolution` read only `status` and never confidence — **the gate decided whether to log a failure, not whether to apply.** Measured over the 124 distinct compound names in committed `runs/**/final_mapped.json`: **24 would be renamed and identifier-stamped, every one below the bar** — including `OPDA → Dinor-12-oxo-phytodienoate` (**not a PathBank synonym at all**; exact-name and synonym lookups both empty; fuzzy tie-break passes by 0.0006), `THF → Tetrahydrofuran`, `CL → Chloride ion`, `G3P → 3-Phosphoglyceric acid`, `PE → O-Phosphoethanolamine`. The gold set already names this class (`PMC13231680` `forbidden_identifiers`, the `PSA` note). Ruling: **fuzzy_name may never rename or stamp**; a unique exact normalized full-name/synonym match may, **except** names of **≤ 4 characters**, which require **corroboration by a matching identifier on the same DB row**; corroboration means **agreement, not presence**; corroborating namespaces are **KEGG, ChEBI, PubChem, HMDB only — DrugBank excluded and fail-closed unless separately ruled**; "record only" means **no rename AND no stamp**; a refused match is **recorded, never raised** (merge rule 7), status `identity_refused_review_required`; the `4` is a named constant citing the decision. **Attribution: the defect is PRE-EXISTING** — C-040 lifted it verbatim from `ir.py` and it remained live post-freeze at `ir.py:911-918`. **C-050 does not create it**; C-050 widens its blast surface by moving false names pre-freeze where they are hashed. **Fix the gate, do not revert C-050.** `test_all_four_artifacts_are_written_and_retained` needs **no** test change and returns to green once the gate lands. **Permanent ownership/boundary records written** for **C-040a** (`prompts/C-040a.md`, tip `7d5a3916`, `REV-040a` bare APPROVE, no blocking findings) and **C-060a** (`prompts/C-060a.md`, tip `07d9d6a7`, `REV-060a` bare APPROVE, no blocking findings) — both cards were dispatched from charters held in session scratch, so these files are the durable record. C-040a's authorized pinned-baseline move is recorded there with its exact delta: **23 of 32 leg digests moved while the compound projection, unresolved-identity counts and `report["ok"]` are identical across 32 legs × 5 pinned offline configs**, from one purely additive `admission` key — **report-shape only, no biology moved**, independently reproduced by `REV-040a`. **Three findings recorded (`FINDINGS.md`): P4-01 (HIGH, evidence integrity) — there is no `.env` in any worktree**, so `PathBankDbResolver.from_env()` returns `None` there and **any PathBank-dependent measurement run from a worktree is vacuously clean**; this is the proven cause of committed `C-050/01-baseproof.json` exiting 0 as a "base proof" (**retained, not deleted; never to be cited** — C-050's genuine G9 base proof is `07-g9base.json`), and it means the bio-auditor's OPDA measurement came from the main checkout and cannot be reproduced in a worktree. Standing rule: harvest read-only from the main checkout and decide offline against the fixture, or label the numbers DB-unavailable; **never edit, copy or shuttle credentials out of `.env`**. C-040a's two-phase shape is the compliant pattern and is *stronger* than a live query, since a reviewer without a DB can re-run the decision phase. **P4-02 (LOW)** — C-040a's measurement scripts are uncommitted, so its committed artifacts are not reproducible from the repo alone; **the approved tip is not to be disturbed to add them.** **P4-03 (INFORMATIONAL)** — the DrugBank exclusion is fail-closed and awaits ratification. Control-plane and documentation only in this commit: no `src/`, no `tests/`, no fixture, no cache, no protected file touched; no verdict re-litigated; no historical entry rewritten; no accepted merge reconstructed. |
| 2026-08-14 | **PACK 5 OPENED — takeover verified, C-050 synchronized and its unauthorized hunk withdrawn, three decisions recorded, the C-050 stack ruled atomic.** Integration tip `6c98508c6ba5dba01d2ab0f9d2c462ad47f4f326`, **local = `origin` = `ls-remote`**. Takeover checks all green: no merge in progress (`MERGE_HEAD`, rebase, cherry-pick and revert all absent), index empty, protected manifest **42/42** (8 `runs/` + 8 `runs_verify/` dirs, 16 `cache_snapshot/` files, 7 protected scratch modifications, exactly **3** untracked `topics_*.txt` — 6 further `topics_*` are **tracked**, which is the counting trap), whole-tree G11 **26,183 artifacts / 0 non-compliant** (primary 1,242 + 53 worktrees 24,941, each scanned through its own `g11_evidence.py` because `check` with no args resolves `REPORTS_ROOT` to its own file's directory), **zero `g11_reserved` stubs tree-wide**, **zero owned surviving processes** (the two VSCode isort LSP servers, PIDs 35456/35500, are pre-existing and were not killed), all nine preserved card branch tips exact, both inert worktrees untouched. **`outputs/pathway.pwml` metadata reconciled, no drift:** the two records name **different objects** — the committed blob is `9f7110ba` and is **byte-identical at `f2f7599` and at `6c98508`** (`git diff` empty), while `01c036c2` @ 34,878 B is the **working-tree protected scratch modification**. `PACK4-STATE.md` had this right; the resume record conflated them and is the stale one. File retained, nothing changed. **C-050 synchronized:** integration merged **into** `agent/p05b-prefreeze-call` at **`0f859d9f`** (parents `7e45f3b5` + `6c98508`) — rebase and cherry-pick are forbidden sprint-wide, so this is the only compliant way to move a card branch onto a moved base. The single predicted conflict (`REV-050c` **B-3**) on `tests/test_streamlit_quarantine_boundary.py` was resolved **to integration's blob verbatim** (`0f00c9f2` on both sides), which also discharges **B-2**, since both offending C-050 hunks — the `PREFREEZE_RESOLUTION_FIELDS` constant and the node15 body rewrite — lived in that one file. Verified after: that file has **zero** diff against integration; `PREFREEZE_RESOLUTION_FIELDS`, `_without_compounds` and the `.get("final_mapped_db") or …` fallback are **absent**; C-050a's `final_mapped_enriched` comparand and its **whole-object** `candidate == mapped_in` **survive unchanged**; integration is now an **ancestor**, so the conflict is dissolved rather than deferred. Resolving in the branch's favour would have silently reverted C-050a — the orchestrator authorized that edit in error and has withdrawn it. Post-sync the delta from integration is **C-050-authorized only**: `streamlit_app.py` +23, `prefreeze_resolution.py` +864 (new), `test_prefreeze_compound_resolution.py` +792 (new), the `C-050`/`C-050c` G11 evidence and the probe script. **Decisions recorded:** **D-029** (`db_unavailable` in prefreeze resolution is `review_required` and must not raise by itself; D-015 clause 6 undisturbed for the four structural codes, which still raise — this writes down `REV-050c`'s **DEF-2**), **D-030** (C-050/C-045/C-051 land as **one atomic stack**, A9 **not** weakened, merge rule 8 **not** waived, one composite `--no-ff` merge preserving individual card commits, and the node15 repair is **C-050a's**, routed as C-050d), **D-031** (D-028's DrugBank exclusion **ratified**, closing **P4-03**). **Findings:** P4-03 **CLOSED**; **P4-02 reaffirmed DEFERRED** — C-040a's measurement scripts stay uncommitted and `7d5a3916`/`734c958` are **not** to be rewritten to add them; the remedy is a later additive evidence commit. **Subcard IDs allocated before dispatch:** `C-050c` was already consumed as a correction-round evidence task ID on the branch, so the node15 follow-up takes **`C-050d`** and the C-050 correction takes **`C-050e`**; no ID reused. **C-050e dispatched** from `0f859d9f` with fresh ceilings 320 / 25 artifacts / 1.5 MB — it does **not** inherit or raise any earlier round's budget. **C-042b stays blocked with proof and its evidence-only branch is not merged**: the measured `worst_case_call_seconds() = 2493.8 s` would arm a refusal that can prevent the **first** model draw having issued **zero** calls, too consequential to wire from one aggregate estimate; a later narrow measurement/design card must reproduce the 2493.8 s decomposition before proposing production wiring. **C-042c remains a measured reachability proof, not a wiring card** — `run_rag_loop` has zero production callers, the Stage-1 ladder already emits `ATTEMPT_CAP_REACHED`, a no-code completion is valid, and no `RoundResult` field is to be added. Control-plane and documentation only in this commit: no `src/`, no `tests/`, no fixture, no cache, no protected file touched; no accepted merge reopened. |

---

## 2026-08-20 — PACK 9: takeover at `32f3a57`, Wave 1 dispatched, T-102 measured, four cards chartered

**Integration** `32f3a57` → **`ee266ce`** (control plane + evidence), pushed, local = `origin` = `ls-remote`.

**Takeover verified once, narrowly.** Empty index · no merge in progress · whole-tree G11 **exit 0, 2408
artifacts, 0 non-compliant** · protected manifest exact (7 scratch + `.claude/settings.json` +
`src/t2pw/app/streamlit_app.py` product-owner edit + 4 untracked `topics_*.txt`) · **zero sprint-owned
processes** (2 VSCode isort LSP servers reported by full command line and **never killed**) · heavy mutex
`C:\t\heavylock` **FREE** · both preserved card branches at their documented tips.

**Environment drift check — the rebuilt `.venv` reproduces the pinned baseline.** Python **3.13.6**, pytest
**9.0.3**, httpx **0.28.1**. SMOKE re-measured on clean unmerged integration: **465 collected**, real run
**exit 0**, 63.82 s, G11 **0 survivors** (`evidence/g11/T-106/01-smoke-drift.json`, `02-smoke-collect.json`).
**The C-054 baseline of 465 holds on the rebuilt environment.** Anything saying 460, 457, 177, 179, 150 or
152 is stale.

### Wave 1 — dispatched concurrently, seams verified disjoint, one writer each

* **C-055** `agent/p33-rag-wiring`, worktree `.claude/worktrees/agent-c055-rag-wiring`, **fast-forwarded
  `1c06918` → `32f3a57`** before dispatch, clean, `.env` present. Owns `streamlit_app.py :: maybe_run_rag`
  + script body. Ceilings **1600 / 140 / 6 MB**. **Sole code-card blocker of T-103.**
* **C-050j** `agent/c050j-component-dedupe`, worktree **created at `32f3a57`**, `.env` copied for F-051
  control parity. Owns `pwml/ir.py :: _dedupe_named_rows` component branch + its call site. Ceilings
  **1100 / 120 / 5 MB**. **Its first act is D-050's exposure census, whose non-zero result is a chartered
  STOP.**

Disjointness verified at file level before dispatch: `streamlit_app.py` vs `pwml/ir.py`, separate worktrees.
Heavy gates still serialize on the mutex. Both received the charter, `_SHARED_EXECUTION_BLOCK.md`, and a new
`_COUNT_CORRECTIONS.md` overriding the shared block § 8's stale SMOKE 460.

### T-102 — `MEASURED`, never `PASS`

Terminal state recorded in its row. F-009's axis re-verified on all four legs rather than quoted; pinned by
`tests/test_canonical_biological_equivalence.py` **29 passed in 0.19 s**, G11 0 survivors. **Both
limitations remain explicit and unowned; no unowned production functionality was created to make it green.**
Two corrections to F-056 recorded there and in `FINDINGS.md`: its *"no `canonical_graph_sha256` anywhere"*
became **false** (one exists, wrong paper and wrong mode, so the blocker survives on narrower grounds), and
**SBML is not absent from the code** — `streamlit_app.py:6472` builds it from a manual UI button the batch
driver never clicks, while `:6015` passes `False`.

### F-066 — NEW, HIGH (process). 21 test files cannot be collected in isolation

Found while running T-102's evidence, on the first isolated run attempted. No `conftest.py` anywhere, `t2pw`
not installed, `pytest.ini` sets no `pythonpath`; each test file inserts `src` into `sys.path` itself and
**21 of 148 do not**. Measured both ways on one file: collection error and exit 2 alone, **29 passed** with
`PYTHONPATH=src`. **SMOKE and every chunk gate are unaffected** — they always include a path-inserting file,
which is why it stayed invisible. **F-054 is what makes it bite**: every card is now told to name and run its
chunkless files explicitly, which is exactly the isolated selection that breaks. **Both live cards were
affected — 4 of C-055's 6 named chunkless files and 4 of C-050j's — and both were notified with the remedy
before reaching their focused runs.** Registered, **not fixed**: every durable fix changes collection
semantics for all 148 files and would force re-pinning SMOKE and all four chunks.

### F-067 — `httpx` undeclared, promoted from a session handoff into the durable record.

### Four cards chartered against live source — none dispatched (two-writer cap)

| Card | Owns | Branch (to create) | Ceilings |
|---|---|---|---|
| **C-056d** | F-055 · `driver.py :: _finalize_gate_failure` + `release_status.py :: SEMANTIC_INPUT_NOT_WIRED` text | `agent/c056d-gate-failure-carry` | 700 / 60 / 3 MB |
| **C-057** | `strict_quarantine.py` lineage writes — the **five module-level functions** | `agent/p27-lineage-quarantine` | 1000 / 110 / 5 MB |
| **C-058** | F-058 · `pipeline.py :: _inject_name_based_modifiers` **transports branch `:2880-2914`** | `agent/c058-transport-actor-guards` | 900 / 90 / 4 MB |
| **C-059** | F-057 · `rag/admission.py` admission decision + `synthesize._dedupe_candidates` | `agent/c059-gap-already-covered` | 1100 / 110 / 5 MB |

**C-056d — the structural fact that shapes it:** `pwml_result` is bound at `driver.py:2294`, **95 lines after
the gate-failure path returns at `:2199`**, so the PASS path's `_frozen_release_record(pwml_result)` source
**does not exist at that seam**. Three candidate sources are enumerated in the charter; session state
(`streamlit_app.py:1143`) is the most promising and the card must confirm or refute it.

**C-057 — the structural fact no record states:** quarantine **destroys** its rows (`strict_quarantine.py:1345`,
`:1542`), while every sibling lineage card writes onto rows that **survive**. "Write lineage onto the row" has
no target for a removed row. `PRODUCT_CONTRACT.md:85-102` binds *"the final pathway"*, which a deleted row is
not in. The card must decide and defend, and **must not change what survives in order to make lineage
attachable**. Also established: the five candidate functions are **module-level defs at `:1095`, `:1294`,
`:1349`, `:1419`, `:1519`** — all *before* `quarantine_and_close` (`:1793-2338`) — so C-057 is **same-file,
different-function** with C-056c, and D-054 §9's collision hazard is structurally avoidable.

### ⚠ BLOCKING SEQUENCING, unchanged and re-affirmed — merge rule 6

**`C-058` and `C-059` must BOTH be ACCEPTED before the F-062 refusal-seam correction is dispatched or
merged.** Repairing the refusal seam first would make both T-100 strict legs exportable and thereby **ship**
the fabricated `EntE` transporters and the LDH-derived NAD+/NADH. **C-058 and C-059 are disjoint at source
(intersection empty) and may run concurrently; they share three test files, so whichever merges second
re-runs all three.**

### Record corrections — five, all measured, code wins each time

1. **F-057 half (i) is FALSE.** The gap detector did **not** mint two ids for one dangling edge. `make_gap_id`
   is `sha1(f"{kind}|{label}")[:8]`; both pre-images were recovered — `EntC reaction` and `EntB
   isochorismatase reaction`. **Two adjacent reactions were each dangling on a different open metabolite**
   (`chorismate` / `pyruvate`). The per-edge dedup **works**, proven by `EntF …` emitting twice and yielding
   one id. The real mechanism is `Gap.target_names()` (`retrieve.py:316-347`) returning
   `_reaction_participants`, so a shared internal metabolite lands in two adjacent gaps' target sets — **a
   structural property of every linear pathway.** A card "fixing the detector" would be fixing correct code.
2. **F-057 half (ii) is confirmed but biologically inert.** No cross-gap dedup exists — but the two
   candidates already collapse to **one** payload row at `synthesize._resolve_reactions`. **Fixing the dedup
   alone yields a byte-identical payload.** The load-bearing defect is that **nothing refuses a gap already
   covered by a locked reaction**, established four ways.
3. **F-057's UNVERIFIED prune/degree-zero asymmetry is RESOLVED and is NOT the cause.** Neither protein row
   carries a `synonyms` key, so both predicates are identical on them. The actual cause is a **third,
   previously unrecorded mechanism**: `map_ids._rewrite_reaction_protein_enzymes_to_complexes` renamed and
   retyped the enzyme reference, leaving the protein row unreferenced. **Registered, not fixed, outside both
   cards.** Alias canonicalization is therefore **STRUCK from C-059** — it is not the cause and would collide
   with both C-050j and C-057.
4. **F-058 is no longer UNVERIFIED, and its scope was off by one.** The site is **inside** `merge_additions`,
   not before it: `pipeline.py :: _inject_name_based_modifiers` transports branch `:2880-2914`, called at
   `:1218`. **Reproduced byte-exactly on 3 legs × 2 papers.** Cause: the bare substring test at `:2884-2886`
   — `"ente"` inside `"enterobactin"`. **The reaction branch's cue window (`:2840-2842`) and exactly-one
   guard (`:2853-2863`) were never extended to the transport branch.** Eight other candidate sites were
   enumerated and excluded with cited bases.
5. **D-054 §9 understates C-056c's footprint.** Measured hunks are `:2130-2138`, `:2152`, `:2228-2240`; §9
   claims `:2130` and `:2132-2144` and **omits the `schema_version` hunk its own §8 authorized**. §9's
   *"exactly one function — `quarantine_and_close`"* claim **is** correct. `git diff 2308ecd 32f3a57 --
   src/t2pw/pipeline/strict_quarantine.py` is **empty**: no drift since the reviewed tip.

### Three more F-054 traps, found live and named so no card inherits them

**`test_rag_admission_adversarial` (54 tests) is NOT Chunk C** — only `test_rag_admission_production_path` is;
a substring on `test_rag_admission` pulls a 54-test file into a chunk documented at 109. **`test_entity_admission`
(23) is NOT Chunk C.** **`test_rag_payload_gate_guardrails` is in NO chunk** while `test_rag_provenance_gates`
is Chunk C. Adding to the two already recorded: `test_batch_driver_quarantine_artifacts` is **not** Chunk B,
and `test_strict_failure_replay` is **not** Chunk E.

### Two untested seams, named so no one assumes coverage

`_admit_for_gap` and `_dedupe_candidates` have **zero direct references anywhere in `tests/`**. And **nothing
asserts on the transport branch of `_inject_name_based_modifiers` at all** — all three of its name-heuristic
tests are on the **reaction** branch.

### Disclosure

Two takeover-baseline G11 reports were allocated under task id **T-106**, which is not a real milestone. The
allocator rejects non-`C`/`H`/`T`-shaped ids and abandoning a created placeholder is a disclosable deviation,
so the id was used rather than discarded. **T-106 = session takeover SMOKE drift check, 2026-08-20.**

**Control plane and evidence only in `ee266ce`:** no `src/`, no `tests/`, no fixture, no cache, no protected
file touched; no verdict re-litigated; no accepted history rewritten; no historical entry rewritten.

### ⚠ MILESTONE SEQUENCING — determined 2026-08-20 (PACK 9), and it bears directly on the T-104 authorization

**Running T-104 before C-058 and C-059 land would burn ~7 h of billed live-curator time on a run that is
guaranteed to fail acceptance for reasons already known.**

The reasoning, from records already accepted:

* **T-104's acceptance is the full matrix vs `BASELINE.md`** (`TEST_MATRIX.md:481`), which includes
  `TEST_MATRIX.md:477`'s requirement that **PMC12452463 classify `review_required`**.
* **T-100 already failed exactly that**, on both strict legs, because `release_status.py:414-419` tests
  `strict_gates_passed` **before** any coverage branch and `strict_quarantine.py:2025-2034` appends
  structural reasons to `refusal_reasons` unconditionally (**F-062**). Nothing about that mechanism is
  leg-count-dependent — **a 20-leg run reproduces it 20 times.**
* **F-062 must not be repaired before C-058 and C-059 are accepted** (merge rule 6): repairing the refusal
  seam first makes both strict legs exportable and thereby **ships** the fabricated `EntE` transporters and
  the LDH-derived NAD+/NADH.

**Therefore the dependency order for a quotable strict rate is:**

```
C-058 (F-058) ─┐
               ├─► F-062 refusal-seam correction ─► T-104 (20 legs, ~7 h) ─► triage ─► T-105 (20 legs, ~7 h)
C-059 (F-057) ─┘
```

**This is a scheduling conclusion, not a new decision** — it follows from D-055, F-062's blocking clause and
merge rule 6, all already locked. It is recorded here so the T-104 authorization is asked for at the point
where the run can actually succeed, rather than twice.

**T-103 is NOT subject to this.** Its acceptance (`TEST_MATRIX.md:480`) is *"every RAG round re-entered
normalization, mapping, gates, persistence, classification"* — a **structural** property of the loop, not a
strict-rate or release-classification claim. **F-062 does not gate it.** T-103 becomes runnable the moment
C-055 merges, and needs only its own live-curator authorization for 4 RAG legs (~1.5 h).

### PACK 9 RULING 1 — where the heavy-mutex line falls, and it is narrower than "every pytest"

C-058 disclosed the line it drew and asked to have it drawn differently if I disagreed: **named sprint gates
(SMOKE, Chunk C, Chunk D) took the mutex; offline replay probes and single-file focused runs did not.** Its
charter listed "replay derivation" as not needing the mutex and it extended that to single-file runs.

**RATIFIED, with one boundary the card did not need but the next card will.**

The mutex exists to stop **contention that manufactures a false result** — F-064's leg died on a concurrent
4.4 MB unlocked cache write, and REV-050j's sampled census timed out *"while C-055's Chunk D contended for
CPU and the DB."* It does not exist to serialize every Python process in the sprint.

**The line, stated so it is not re-litigated a fourth time:**

**TAKE THE MUTEX** for: SMOKE · any chunk gate (A/B/C/D/E, including `--only` narrowings) · any `qb` run ·
any pipeline leg or benchmark · **and any job that reaches the live PathBank DB or writes a shared cache
under `data/`** — regardless of how few tests it runs.

**DO NOT NEED IT** for: a fully offline single-file focused pytest · a read-only replay or census probe ·
`g11_evidence.py next` / `check` · AST or source-hash comparisons.

**The discriminator is the resource, not the test count.** A one-test run that opens the PathBank DB
contends exactly as hard as a chunk; a 226-test offline replay contends with nothing but CPU. `_SHARED_
EXECUTION_BLOCK.md` § 1 should carry this wording; it currently says only *"one heavy job at a time"* and
leaves each card to guess.

**Note for the record:** C-058 disclosed its line rather than assuming it, which is why this ruling exists at
all. That is the behaviour the shared block asks for and it cost the card nothing.

### PACK 9 RULING 2 — three charter line numbers were WRONG, and all three were the orchestrator's error

C-058 verified every line number it was given and found three false. **The code won each time. Recorded here
because the sprint's standing claim — that records are false more often than code — applies to the
orchestrator's own charters, not only to inherited records.**

| Charter said | Live truth at base `6f9b499` | Consequence |
|---|---|---|
| the replay reproduces on **"3 legs across 2 papers"** | **4 legs / 5 transport rows**; corpus-wide **44 rows across 64 legs** | **My summary was imprecise. `FINDINGS.md:1890`'s own correction block lists all four** — I under-counted when compressing it into the charter. |
| `merge_additions` called from `streamlit_app.py:4910` and `:4974` | **`:4877` and `:4941`** — off by 33 | I propagated line numbers measured against **C-055's branch state**, not against the card's own base. Harmless here only because **no signature change was made**. |
| `_attached_actor_names` at `:2777` | the `def` is at **`:2774`**; `:2777` is inside its docstring | cosmetic |

**The second is the instructive one.** A charter's line numbers must be measured **against the card's own
base SHA**, not against whatever tree the preparatory research happened to run on. C-055 and C-058 were
chartered from research done at different tips, and one card inherited the other's addresses. **No harm
resulted because the card verified rather than trusted — which is exactly why every charter carries the
instruction to verify.**

**Also confirmed, and not a defect:** four focused files collect more nodes than they declare `def test_`
(`entity_admission` 23→33, `streamlit_stage2` 11→12, `adversarial_actor` 18→29, `payload_models` 13→19).
**Parameterization, not drift.** Any future census that compares a `def test_` count to a collected count
must expect this — the two are different measurements and neither is wrong.

### PACK 9 RULING 3 — `T2PW_SPECIES_LLM` on the RAG round path is an OPERATOR decision, not a code change

REV-055 asked the orchestrator to decide whether `T2PW_SPECIES_LLM=0` belongs on the per-round mapping path.
Per-round `map_payload` runs `hydrate_species_references(use_llm=…)` (`streamlit_app.py:1063-1076`,
`map_ids.py:8119`) and **the flag defaults to ON**, so each round issues LLM calls on top of the
post-pipeline pass.

**RULING: it does not go in the code, and C-055 was right to refuse.**

C-055 disclosed this as its deviation 4 and explicitly declined to *"flip a resolver flag behind the
operator's back."* That is the correct instinct and it is the same principle F-056 records one layer down,
where `db_resolver=None` **enables** the DB rather than disabling it — a flag whose sense a caller has to
guess is how this project has repeatedly measured the wrong thing. A card silently suppressing a resolver to
make its own loop cheaper would be exactly that defect, authored deliberately.

**Instead it is set at run time, per run, and recorded in that run's config:**

* **T-103 SHALL run with `T2PW_SPECIES_LLM=0` in the bounded child.** Its acceptance
  (`TEST_MATRIX.md:480`) is *"every RAG round re-entered normalization, mapping, gates, persistence,
  classification"* — **mapping still runs and is still observable with the flag off**, so the acceptance
  property is untouched while the cost and the nondeterminism drop. Add it to `T-103-prep.md`'s command.
* **T-104 is a separate decision** and must not inherit this one. A release-candidate benchmark that
  suppresses a production resolver is not measuring production.

**No code change. No card. The flag's default is unchanged.**

### PACK 9 RULING 4 — C-055 tightens a gate that was previously unconditional, and that must not be read as a regression

REV-055 identified a **real behavioural delta on the RAG-on path at defaults**, and ruled it correctly:
after C-055, RAG additions must clear `validate_graph_delta` (`controller.py:306-307`,
`streamlit_app.py:1284-1313`) before `final_payload` advances. **At base they advanced unconditionally**,
subject only to the differential schema gate.

**This is `PRODUCT_CONTRACT` §10 — *"only when it fills a specific detected typed gap"* — being enforced for
the first time.** It is a **tightening**, so **merge rule 6 holds** (no gate weakened to increase PWML
production) and the **NEW CAPABILITY** arm remains correct.

**Binding on how T-103's output is read:** real legs may now **refuse a merge that previously landed**, and a
warning naming the refusing rules is emitted. **That is the contract working, not a regression.** A T-103 leg
showing fewer merged additions than a pre-C-055 leg is **not** evidence of a defect and must not be reported
as one. Anyone comparing T-103 against T-100's committed legs must account for this delta explicitly.

### PACK 9 RULING 5 — a reviewer that cannot take the mutex is an infrastructure gap, not a process violation

REV-055 disclosed that it **could not create `C:\t\heavylock`** — `mkdir` outside the project directory is
blocked by its permission classifier — so its SMOKE run was **unprotected**. It verified the lock was free
immediately before, **did not take it, and did not clear it**, and said so plainly.

**Ratified as a disclosed deviation, and the gap is registered rather than papered over.** The reviewer did
the only correct thing available to it: it disclosed rather than concealing, and it did not touch a lock it
could not properly hold.

**Consequence to fix, unowned:** `pwml-reviewer` agents cannot participate in the sprint's own mutex
protocol. Either reviewers get the permission, or the protocol must give them a project-local lock path.
**Until then, an orchestrator dispatching a reviewer that will run a heavy gate should confirm the mutex is
free and hold it on the reviewer's behalf, or accept a disclosed unprotected run.** This is the second
infrastructure limitation this sprint has hit at the agent-permission layer.

### Two LOW findings registered from REV-055 — neither owned, neither fixed

* **Per-round `map_payload` can conditionally reach the tracked 39 MB `data/enrichment_cache.json`** through
  the UniProt identity-evidence ladder's `cache.save()` (`uniprot_evidence.py:335-364`, `:428-446`) — but
  **only** when the identity-evidence env flag selects *network* mode **and** a fetch misses cache. C-055's
  trap-11 answer covers `data/id_mapping_cache.json` only. **Conditional, pre-existing on the post-pipeline
  path, and both modules are outside C-055's boundary.** A tracked 39 MB file written from a loop would
  dirty the tree — worth an owner before `max_rounds > 1` is ever used in anger.
* **`apply_post_merge_cleanup`'s `quarantine_output_path`** still defaults to the shared
  `tmp/quarantined_rag_reactions.json` (`streamlit_app.py:1275-1279`) and is now written **once per round**.
  Identical to base at the default `max_rounds=1`; at `N > 1` the last round overwrites earlier rounds'
  records. **Not one of the three protected scratch files, and no test covers it.**

### PACK 9 RULING 6 — C-059's ceiling raised 1100 → 1700, and my charter §2 was FALSE

**Measured 1625 hand-authored** (tests 911 · `admission.py` 343 · probe 337 · `synthesize.py` 34), against a
ceiling of 1100. **Ratified prospectively to 1700 / 110 / 5 MB.** The diff **deletes zero lines anywhere**;
nothing was cut to fit.

**The overrun is mine, and it has a specific cause: I told the card something false.**

C-059's charter §2 asserted, in bold, that fixing the cross-gap dedup alone would produce a **byte-identical
payload** — and instructed the card to *"report that, it contradicts §2 and I want to know"* if it turned out
otherwise. **It turned out otherwise, and the card reported it.**

`_merge_into` unions `_Reaction.scores` and `gap_ids`; `_confidence` maxes the scores. Collapsing at
admission therefore cost four things: `rag_confidence` **0.930233 → 0.914815**, the same on the enzyme actor,
`rag_provenance.gap_ids`, and the lineage entry naming the gaps.

**The `gap_ids` loss was a regression against a pinned test** —
`test_rag_admission_adversarial::test_one_claim_admitted_for_two_gaps_keeps_both_attributions`. **It was
caught by running that test, not by reasoning**, which is the whole reason the sprint requires the pinned
suites to be run rather than argued about.

Repairing it required carrying the union with the verdict — ~20 lines in **`synthesize_with_report`'s body**,
a **disclosed boundary extension** beyond the charter-named `_dedupe_candidates`, guarded so uncollapsed
claims never take it. That work exists **only because my §2 was wrong**, and the ceiling was sized before it.

**§2's substantive point survives and is confirmed:** `REASON_ALREADY_COVERED` is the fix that changes the
biology (accepted 2 → **0**, the re-imported row gone), and the cross-gap dedup is hygiene — with the
coverage rule neutralized it takes accepted 2 → 1 and **still emits** the row that went degree-zero. The
card had the two the right way round, which is what §2 existed to enforce. **Only the byte-identical claim
was false.**

**Second charter error of the session** (PACK 9 RULING 2 recorded three wrong line numbers). Both were
caught by the card, not by me. **The instruction to verify rather than trust is the only reason neither cost
anything.**

### PACK 9 RULING 7 — a card that shortens its own text rather than raise a bound gets it right

C-059 edited `tests/test_rag_gap_admission.py` (+6 lines), extending its pinned `ADMISSION_RULES` name list
**additively** for two new reason codes.

It also hit that file's **report-size bound** — and **did not raise it.** It shortened its own rule texts
**575 → 344 bytes** instead, on the stated reasoning that *raising a size bound to fit one's own additions is
what that bound exists to catch.*

**That is exactly right, and it is worth recording as the standard.** A bound a card may raise whenever it
becomes inconvenient is not a bound. The same instinct — fix the thing the guard is complaining about, never
the guard — is what REV-051 protected when it refused to let F-065's assertion be re-pointed, and what
REV-051a protects when it forbids cutting an adversarial arm to fit a budget.

**Standing guidance:** a pinned bound, digest or count may move **only** as a deliberate baseline move under
merge rule 4, **with an exact documented delta**, and never to accommodate the moving card's own additions.
Where the card can instead change its own contribution to fit, **that is the correct fix**.

### C-058 — ACCEPTED, MERGED `c3fd041`, reviewed tip `a908666` (2026-08-20, PACK 9)

Exact bare `APPROVE` from REV-058, **zero correction rounds**. Owns F-058.

`pipeline.py :: _inject_name_based_modifiers`, transports branch only. The cause was one statement
(`:2884-2886`): a bare substring test with no word boundary, no cue window and no exactly-one guard, so
`"ente"` matched inside `"enterobactin"` and `EntE` — an adenylation enzyme — was attached as the transporter
of every transport row on four legs across two papers, on spans naming TolC and TonB.

**Four guards**, three of them the reaction branch's own discipline plus one the charter did not name: a
row's declared **`cargo` is never its transporter**. REV-058 confirmed that fourth guard was **required, not
optional** — without it the card would have left **1 of 24** shipped fabrications in place (`ALAS2` on
`PMC12856317/strict`, which passes guards 1-3).

**Merge rule 7 holds corpus-wide, proved by the reviewer independently:** `rows_losing_a_stage1_transporter:
0` over 44 rows across 64 legs, zero skipped. **24 rows carry a Stage-1 transporter and all 24 survive** —
FepA ×5, TolC ×5, TonB ×5, MsbA ×5, SFXN1 ×3, Lpt ×1. All 24 base injections were fabrications (23 `EntE`,
1 `ALAS2`); nothing correct was traded away.

**G9 CORRECTION with a real base failure**, re-run by the reviewer in a git worktree at `6f9b499` carrying
`.env` (not an export — `PATHSPEC` omits `runs_verify/`, which the proof reads): **`10 failed, 5 passed` ->
`15 passed`**. Zero failures are symbol absence; eight assert on produced payload content, one is a `KeyError`
on the produced dict, one is a crash **inside base production code** at `pipeline.py:2908`. The five passing
at base are exactly the five preservation arms.

**Reaction branch byte-identical, three proofs, all re-derived:** no hunk intersects base `:2818-2878`;
base `:2818-2878` and tip `:2828-2888` hash identically; and the reviewer's own replay digest of every
reaction's `enzymes`/`modifiers` is identical on **64 of 64 legs**.

**The cue-vocabulary decision was a biological judgement and it was made correctly.** The card deliberately
did not reuse `ENZYME_EVIDENCE_CUE_RE`; the reviewer measured it against every whole-token window around
`FepA` and confirmed it does not fire on `PMC12452463/strict` — borrowing it would have **refused the one
correct attachment**. The enzyme vocabulary is also wrong in the other direction: `TolC-dependent` fires it
via `dependent`, which is not a catalysis claim.

**Non-vacuity re-measured at the actual tip** (the card's own mutation evidence predated its final commit):
boundary 3 red, cue 1 red, exactly-one 1 red, cargo 1 red.

**Five behaviour changes, not four** — `_attached_transporter_names` reads five key shapes where base read
two, so a row already naming its transporter under the typed `entity` key is no longer rewritten.
Base-failing-proved, non-destructive, inside the seam. Recorded because the card summary emphasises four.

Budget ratified 900 -> 1200, measured **889 / 73 / 0.39 MB**. Gates: SMOKE **465**, Chunk C **109**,
Chunk D **185/187**, focused **226 collected / 0 failures**, 68 G11 reports **0 non-compliant**, survivors 0
on every one. Post-merge gates green (`MERGE-058/01-smoke.json`, 465).

**One process violation, the reviewer's and not the card's** — F-072, disclosed unprompted.

**⚠ C-058 is one half of the F-062 lock. C-059 must also be ACCEPTED before the refusal seam is touched.**

### PACK 9 RULING 8 — C-059's ceiling raised again, 1700 → 2000, and both raises trace to my charter

**Measured 1875** at `94625bd` after the REV-059 correction round (was 1625 at `c822dd8`; +188 test, +62
probe). **Ratified prospectively to 2000 / 110 / 5 MB.** Zero deletions anywhere in the tree.

**Product code is 377 of the 1875.** The remaining ~1500 is the test suite, the offline probe and their
documentation — for a card that had to build an empirical merge-rule-7 safety case (REV-061 forbids the
monotonicity argument), non-vacuity arms into a seam with **zero prior test references**, both G9 arms, and
then a real-path witness after its replica proved blind.

**Both raises trace to the same root: my charter §2 asserted a byte-identical payload that was false**
(RULING 6). That error created the union-carry work, and the correction round then created the real-path arm.
**The card cut nothing at either point and reported both overruns rather than absorbing them** — which is the
behaviour REV-051a exists to produce, and it worked twice.

### PACK 9 RULING 9 — a card that finds a second instance of its own defect, unprompted, is doing the job

While fixing the false record REV-059 caught, C-059 **found a second one of the same class in its own diff**
and fixed it without being asked: a replica test named `..._is_payload_neutral_once_the_union_is_carried`
asserting full-row equality, blind for exactly the reason the first was.

**What it did with it is the part worth recording.** It did not merely correct the name. The replica now runs
the production lineage append, is renamed `..._changes_nothing_on_the_row_but_the_lineage`, and **asserts the
differing-key set is exactly `{provenance_lineage}`** — so a *second* field cannot begin moving unnoticed.

**That is the correct generalisation of F-074.** The failure was never "this assertion was wrong"; it was
"the instrument could not see the field that moved." An exact-set assertion converts an open blind spot into
a closed one: any new divergence fails the test rather than passing it silently. **Prefer pinning the exact
set of differing keys over asserting that a specific field is unchanged**, wherever a card claims something is
neutral.

The card also **withdrew an argument of its own** that the reviewer had flagged as weaker than stated — that
a two-module `rag` diff cannot reach F-065 or `qb` node15 because neither file references `t2pw.rag`. The
greps hold; the inference does not, since `qb` drives the app through `maybe_run_rag`. Both reds now stand on
pre-charged status alone.

### PACK 9 RULING 10 — C-057's §3 answer overrides my charter's stated preference, and the card was right

My C-057 charter posed the question and said option 1 — **attribute the surviving rows** — was *"the reading
this charter finds most defensible."* **C-057 argued option 2 — attribute what was EXCLUDED, and nothing
kept — and I accept it. My preference was wrong.**

The reasoning, recorded because it is the durable part:

* **Quarantine is a filter, not a source.** It introduces no content, so the only `PRODUCT_CONTRACT` § 3
  category it can populate is the sixth — `CONTRACT_CATEGORIES["unresolved_or_excluded"] = ("unresolved",
  "excluded")` (`lineage.py:79-86`).
* **Option 1 could not be done honestly.** § 3's per-element sentence binds *"Every **externally added**
  entity and process"*, and its stated purpose is attributing **false content** to the stage that introduced
  it. Naming quarantine as the provenance of a row it merely let through runs that backwards. And the closed
  `ORIGINS` tuple has **no member meaning "retained unchanged"**, so option 1 required either a fabricated
  origin or an edit to frozen `lineage.py`. **The card took neither, and correctly did not stop-and-report,
  because § 3 *can* be discharged without changing what survives.**
* **A reader genuinely reaches the artifact.** `test_the_attribution_reaches_quarantine_report_json_on_disk`
  reads it back **off the filesystem**; and quarantined *locked reactions* carry it into `final_mapped.json`
  itself via `_reconcile_locked_reactions:1668` — 125 entries across the cohort.

**The corroboration the card found is the strongest evidence it chose right:**
`tests/test_pipeline_lineage_schema.py:25` already pairs `"excluded": ("quarantine", "unsupported",
"not_evaluated")` as *"what a real writer would pair it with"*. **C-015 wrote down the expected shape before
any writer existed, and C-057 matched it field for field without being pointed at it.**

### PACK 9 RULING 11 — the c011 golden moves via the file's own house helper, never by regenerating the fixture

`tests/test_c011_freeze_seam_golden_equivalence.py` passes at `ca6bf13` and fails ×2 at C-057's tip, because
its golden pins `canonical_payload_sha256` and lineage is part of that hash by design.

**GRANTED: a narrow boundary extension for C-057 to add `_with_c057_lineage_hashes` to that file, in the
established house pattern, and nothing else in it.**

* **The file already carries this pattern twice** — C-030's `_with_c030_hash_keys` (`:236-258`, *"stated here
  instead of being absorbed by a rewritten fixture"*) and C-052's `_with_c052_path_keys` (`:281-302`). This
  is the file's own documented mechanism, not a new precedent.
* **Regenerating the tracked fixture is FORBIDDEN.** The golden is named as a BEFORE document; rewriting it
  destroys the property it exists to hold. Same standing precedent as REV-051's refusal to re-point F-065.
* **The delta is safe and was measured exactly before the grant:** 2 fields on 7 of 39 legs, size identical,
  no top-level key moved, added, removed or renamed — and **`canonical_graph_sha256` moves on ZERO legs.**
  Graph equivalence holds; only `canonical_payload_sha256` moves, which `PRODUCT_CONTRACT.md:178`
  **requires** to remain detectable.

**Generalisation:** where a fixture already carries a documented per-card helper for baseline moves, **that
helper is the sanctioned mechanism and regeneration is not**. The helper keeps the fixture the BEFORE
document it is named for, and keeps every prior card's delta legible instead of folding them all into one
opaque rewrite.

### PACK 9 RULING 12 — C-057's ceiling raised a third time, 1200 → 1250, on an estimate I gave it

**Measured 1208.** Ratified to **1250 / 110 / 5 MB**. **No correction round consumed** — a budget correction
caused by an inaccurate orchestrator estimate is the orchestrator's to make, and both of C-057's rounds are
already spent on substantive work.

**The cause is traceable to a number I quoted.** When granting the perturbation driver I said *"~45 lines →
~1140"*, taken from the card's *scratch* harness. The committed driver is **98 lines**, because my own ruling
— *runnable beats a transcript, and it must be pointable at the next card's own helper* — requires a module
docstring explaining what the two biology arms buy, `argparse`, and an exit-code contract. **Those are
precisely the parts that make it reusable rather than a script with one card's leg names hard-coded.**
1095 + 98 + 15 (comment) = 1208.

**The card did not trim it and was right not to.** Cutting the docstring or the exit-code contract to reach
1200 would degrade exactly what the ruling asked for — and cutting an adversarial arm or its documentation to
fit a number is the one move I have said I would reject for. **It reported and asked. That is the sixth and
last ceiling raise of the session, and every one of the six traced to my estimate rather than to a card's
discipline.**

**Recorded as a pattern, not an apology:** a ceiling quoted from a scratch artifact will under-count the
committed one, because committing something makes it reusable and reusable costs lines. **Size a ceiling
against what the deliverable must be, not against the prototype that proved it possible.**

### PACK 9 RULING 13 — a mutation harness must be able to fail, and C-057 proved its own could

C-057 committed a driver that runs its seven perturbations and exits non-zero if any arm is silently
accepted. **Then it audited the harness itself**: replacing `_with_c057_lineage_hashes` with a no-op returning
the document unchanged makes the driver report **all seven arms GREEN and exit 1**. Measured, not argued.

**That is what makes a green run evidence instead of silence.** A harness that reports success when the guard
it audits has stopped guarding is worse than no harness — it converts an unchecked property into a checked-
looking one. C-051's reviewer found exactly that failure in a centralized guard, and F-074 is the same defect
one level up, in an instrument rather than a guard.

**Standing guidance:** a committed mutation or perturbation harness carries its **own** non-vacuity check —
neutralize the thing it audits and show the harness goes red. **Without it, the harness is an assertion about
an assertion, and nothing has verified the outer one.**

---

# PACK 10 — the post-PACK-9 wave (2026-08-21)

Opened from integration `ac776682d36012b0b583952d78ac8f0cf02115a3` with an **empty card queue**. Takeover
verified clean: local = origin = `ls-remote`, no merge in progress, empty index, whole-tree G11 **exit 0 /
2890 artifacts / 0 non-compliant**, **zero** sprint-owned Python processes, `C:\t\heavylock` absent,
protected state exactly as manifested.

**SMOKE re-measured before any card merged: `465 passed in 38.82s`** at `931c065` (`ac77668` plus five
charter `.md` files, nothing else), under the mutex, zero survivors, cleanup success. Evidence at
`evidence/g11/T-106/03` and `04`. **This is what makes a post-merge red attributable to a card this session
rather than to drift.**

## Card register

| Card | Finding | Branch | Base | Tip | State |
|---|---|---|---|---|---|
| **C-063** | F-071 + F-072 | `agent/c063-runjob-lifecycle` | `ac77668` | `5d54eb5` | reported; **REV-063 dispatched on exact tip** |
| **C-064** | F-070 | `agent/c064-round-cap-reason` | `ac77668` | `b51b6c9` | reported, 1 orchestrator-caused correction; **REV-064 dispatched on exact tip** |
| **C-065** | F-076 | `agent/c065-golden-regen-guard` | `4bfeb06` | — | dispatched |
| **C-066** | F-067 | — | — | — | chartered, ready |
| **C-067** | F-062 | — | — | — | chartered, **GATED** on a per-reason biological ruling |
| **C-068** | F-069 | `agent/c068-golden-coverage` | `5414cda` | — | dispatched |
| **C-069** | F-073 | — | — | — | chartered, ready |
| **C-070** | F-066 | — | — | — | chartered, ready |

**F-077** — LOW, deliberate, pinned by tests, **not a contract violation today**. Both scopes blocked on
separate decisions (a `schema_version` 6→7 bump; a card owning `_revalidate_surviving_processes`).
**Deliberately deferred. That is a decision, not an oversight.**

**Seam disjointness verified before every parallel dispatch**, and re-verified from the merged file lists
after reporting: C-063 touches only `docs/pwml_recovery_sprint/evidence/`; C-064 touches
`src/t2pw/rag/loop_policy.py` plus four test files and its own evidence. **Zero overlap, and neither touches
`src/t2pw/app/streamlit_app.py`** — so the product-owner-edit stash procedure is **not** required for either
merge. Verified, not assumed.

## PACK 10 RULING 1 — a control-plane grep restricted to `*.md` does not search the control plane

**`PRODUCT_CONTRACT.md:341` conditions PMC12452463's required outcome on "after the index fix". Three
consecutive sessions recorded that phrase as having no antecedent, and F-062 — the sprint's highest-value
finding — has been blocked on it throughout.**

All three ran **the same search**, and it is quoted verbatim at `FINDINGS.md:1510` and `DECISIONS.md:3316`:

```
grep -rn "index fix" docs/pwml_recovery_sprint/*.md
```

**The antecedent is in a committed `.py` docstring**, which that glob cannot reach:
`evidence/probe_downstream_gates.py:1` — *"How far a leg gets AFTER the **stale-index fix** -- the honest
limit of **C-010**"*; `:5-6` names *"the index defect (C-010)"*; `:122` prints `quarantine (index-fixed)`.
`LEDGER.md:114` has **C-010 = "p01 stale positional index", MERGED `9e06360`**.

That probe is **about PMC12452463**, and states its own purpose as giving **T-100** its evidence base —
whose acceptance (`TEST_MATRIX.md:477`) **is** the contract row. Probe, contract row and milestone acceptance
are three statements of one thing, and the probe is the one that names the event.

**Registered as F-080, and deliberately scoped as a READING rather than a definition** — no document says
*"the index fix means C-010"* in those words. It does **not** overturn `DECISIONS.md` D-055 §6, which asks
the product owner to name the referent or strike the condition; **it supplies the evidence for the naming.**

**Standing rule: search `docs/` including `.py`, and `tests/` too, before recording that a term has no
antecedent.** `docs/` carries committed evidence code whose docstrings hold load-bearing definitions —
`probe_downstream_gates.py` is 40 lines of prose before its first import. **This habit cost three sessions.**

## PACK 10 RULING 2 — size a ceiling against everything the charter mandates, not against the ownership table

**C-063 reported itself over ceiling and did not self-authorize.** Correct under S4, and the seventh
consecutive raise this sprint traceable to an orchestrator estimate rather than to a card's discipline.

I set 1400 changed lines against a **three-file** ownership table, while the same charter simultaneously
required a base-vs-tip G9 harness, **six** permanent selftest arms and **four** RULING-13 neutralizations.
The harness alone is 715 lines and was **not in the table the ceiling was estimated against**.

**Ratified to 1800 changed lines (including the harness) / 380 docstring+comment / 5 MB.** Measured
1628 / 360 / ~96 KB. **No correction round charged** (RULING 12). **RULING 7 forbids trimming an adversarial
arm or its documentation to fit a number, and the card did not.**

**The generalisation, which RULING 12 half-stated and this completes:** a ceiling must be sized against
**every artifact the charter obliges the card to produce**, including ones that will correctly live outside
the owned-files table. A G9 harness, a census instrument and a neutralization suite are deliverables, not
overhead.

## PACK 10 RULING 3 — a `missed` list may count occurrences, not distinct items

F-073 records `runner.CHILD_IMPORTS` as *"missing six deferred imports"* and says the cure is *"adding all
six"*. **Measured by executing the test module's own helpers against the real driver: 9 deferred imports,
6 missed — and the six resolve to exactly TWO distinct modules**, `t2pw.pipeline.release_status` (4 sites)
and `t2pw.pipeline.strict_quarantine` (2 sites). `CHILD_IMPORTS` goes **5 → 7**, not 5 → 11.

The finding is not wrong — it counts **deferral sites**, and each site is a real place the preflight is blind
— but *"adding all six"* materially overstates the work. **Recorded because it is the fourth drifted count in
one session:** corpus 32 → **35**, test files 148 → **153**, `missed` 6 sites → **2 modules**, G11 artifacts
2889 → **2890**. **Never cite a count from a record. Measure it.**

## Two findings registered from the F-069 biological lane

Both verified unregistered before filing.

* **F-078** (MEDIUM) — an adenylation reaction emits `AMP` as a co-product. One ATP cannot yield both an
  adenylylated product and free AMP; EntE releases pyrophosphate. The row carries
  `(paper_extraction, paper_stated, explicit)` while `AMP` appears in the source **only inside an enzyme
  name**. Same mechanism as F-058's fabricated transporter and the NAD+/NADH rows — **a species derived from
  an enzyme name or an EC number, then stamped explicit as though the span carried it. Three instances now
  share it on one leg.**
* **F-079** (HIGH) — a payload asserting `EntE`-catalyzed **adenylation** outputs **enterobactin**, plus a
  fabricated transporter on a **TolC** span and a reaction whose entire evidence string is its own name, was
  classified **`release_ready`** with **`semantic_evaluation: passed`** and one unrelated review flag.
  `PRODUCT_CONTRACT.md:343` makes structured status authoritative. **Distinct from F-055**, which concerns
  gate-**failed** legs; this leg failed no gate. Nothing shipped — the run wrote no PWML — so the violation
  is in the **classification**.
  **F-079 is evidence the F-053 prohibition should STAY.** A `passed` verdict has now been observed on a
  payload carrying a false product assignment, so a consumer treating `passed` as positive evidence would be
  consuming a value that has demonstrably not earned it.

## Milestone posture

* **T-101** — `TEST_MATRIX.md:478` is a **live 4–6 leg benchmark, ~2 h**, not an offline check. Its third
  acceptance clause, *"`budget_exhausted` distinct from failure"*, **is F-070**, so it is sequenced after
  C-064. **PMC12312563's scope was recorded in no topics file** and was recovered from the successful slugged
  run's `00_PAPER.txt`: **`menaquinone biosynthesis | Bacillus subtilis`**. The *"Stage-0 scope abort"* at
  `topics_verify_subset.txt:13` is what happens **scopeless**.
* **T-103** — **round multiplier resolved: 1×.** `rag_loop_max_rounds()` (`streamlit_app.py:912-920`)
  defaults to 1 and `RAG_LOOP_MAX_ROUNDS` is unset in `.env`, so T-103 ≈ 1.5 h. The same measurement
  independently confirms F-070's claim that `max_rounds=1` is the **default production path**.
* **Cost for both ≈ $0 marginal** — all nine OpenRouter model slots are `openrouter/free`. The binding
  constraint is free-tier rate limiting and wall clock. Package at `T101_T103_AUTHORIZATION.md`.
* **T-104 / T-105** — unchanged: blocked behind F-062, two separate ~7 h runs, **never collapsed into one**.

## PACK 10 RULING 4 — "do not touch the lock" plus an unforbidden heavy job guarantees an unprotected run

**C-068 ran SMOKE (`465 passed in 37.65s`) without the heavy mutex.** It did not touch `C:\t\heavylock` and
did not claim to hold it, so this is a **disclosed unprotected heavy run**, not an F-072-class violation. The
number is independently corroborated by the orchestrator's own protected SMOKE earlier the same session, so
nothing measured is in doubt.

**The cause is a defect in my charter, not in the card's judgement.** I wrote *"Offline and focused — do not
create, touch or remove `C:/t/heavylock`. Never `pytest -n auto`; never run the full suite unchunked."*
I forbade **taking the lock** and never forbade **running a heavy job**. The card then, reasonably, wanted to
show the pinned baseline unmoved by its own change — which is exactly the diligence the sprint asks for — and
the only path my instructions left open was to run it unprotected.

**PACK 9 RULING 1 sets the line by RESOURCE, not by test count**: SMOKE, any chunk, any `qb`, any leg or
benchmark, and anything touching the live DB or writing a shared cache under `data/` all need the mutex.
**PACK 9 RULING 5** then records that subagents **cannot create the lock at all** — a permission classifier
blocks it. **Those two rulings together mean a subagent can never legitimately run a heavy job on its own.**

**Standing rule for every future charter and dispatch.** Telling an agent not to touch the lock is only half
an instruction. Pair it explicitly with one of:

1. **"Do not run SMOKE, any chunk gate, any `qb`, any benchmark or any pipeline leg. If you believe you need
   one, stop and ask the orchestrator, who will hold the lock for you and run it."** — the default; or
2. **"The orchestrator is holding the lock for job X"**, named and time-bounded; or
3. an explicit, recorded acceptance that a specific run will be **unprotected and disclosed as such**.

**Absent one of those three, an agent that decides mid-card that it needs a heavy run has no compliant option
and will produce an unprotected one.** That is a charter bug with an agent-shaped symptom, and it is the
third distinct process defect this sprint rooted in agents being asked to satisfy an infrastructure protocol
the charter did not fully specify — after F-071 and F-072.

**Note for the C-063 merge:** once `bounded_run.py` owns mutex acquisition (`--heavy-lock`), option 1 stops
being a workaround and becomes enforceable — a subagent passing the flag gets a real acquire or a hard stop,
and the "cannot create the lock" limitation is removed from agent hands entirely. **This ruling is the case
for C-063 rather than an argument against it.**

## PACK 10 RULING 5 — a guard can be silently disabled by the transport that writes it, and only a bidirectional test catches that

**While fixing F-082, C-063 first wrote `\b` as a literal backspace byte (`0x08`)** — the shell transport
collapsed `\\` on the way in. **The pattern compiled without error and matched nothing.** A credential
scanner in that state reports every artifact clean, forever, and whole-tree G11 stays green while guarding
nothing.

**The card caught it by inspecting bytes**, then rebuilt both patterns with `chr(92)` and scanned all four
owned files for stray control characters, tabs and over-long lines. **The orchestrator verified the result
independently:** no stray control bytes in the `CRED_PATTERNS` block, five false-positive shapes clean
(`ondisk-`, `task-`, `risk-`, `C:/runs/troughs_…`, `laughs_…`), and all three true-positive shapes still
caught (bare key, assigned key, `ghp_` token).

### Why this is a ruling and not an anecdote

**The charter required the regression arm to be bidirectional**, and that requirement is the only thing
standing between this defect and a silently dead scanner:

* a **false-positive-only** test ("these ordinary labels must come back clean") **passes on a dead pattern**;
* the **true-positive** half ("this key shape must still be rejected") is what fails.

C-063 confirmed the arm is non-vacuous by replacing the openai pattern with one that matches nothing and
watching it go red. **The half that felt redundant when the charter was written is the half that did the
work.**

### The generalisation

RULING 13 says a mutation harness must carry its own non-vacuity check. **This extends it one step further
back: a guard can be neutralised before it ever runs, by the mechanism that writes it.** A regex, a glob, a
schema, a validator — any of them can be transported into a form that parses, imports, compiles and is inert.

**Standing rule.** Whenever a card writes or edits a *pattern* — regex, glob, matcher, validator — its test
must assert **both** directions: that the things it must not match come back clean, **and** that a known
positive is still caught. **A one-directional pattern test is not a weaker test; it is a test that cannot
detect the most likely failure.** Where the pattern is security-adjacent, inspect the committed bytes as
well, because `\b`, `\d`, `\s` and `\\` are all one transport away from a control character that compiles.

### Sizing the risk honestly

C-063 scanned **all 2906 committed artifacts** with every pattern, before and after: **0 hits either way.**
The F-082 hazard was **latent, not active** — precisely because `check` had already refused to let REV-068's
two artifacts be committed. **The gate worked. The cost was that a reviewer had to find the defect instead of
a red tree finding it**, and that is the sentence F-082's closure should carry.

### PACK 10 RULING 5 — AMENDMENT: it happened a second time, from the opposite direction

**Byte inspection is not sufficient. The same afternoon, the docstring written to warn about this defect
reproduced it.**

C-063 was asked to record, in `test_credential_scan_is_word_bounded_and_still_bites`, that the true-positive
half exists because a pattern can be silently neutralised by **transport** rather than by a bad edit. **The
docstring it wrote was not raw**, so Python parsed its own `\b` into a real backspace character. *The text
warning about a parsed escape was itself carrying one.*

**The card's earlier control-byte scan did not catch it**, and the reason is the whole point: that scan reads
**file bytes**, where `\b` is two entirely innocent characters, `\` and `b`. The defect exists only in the
**parsed** value. It was caught by inspecting `__doc__` at runtime, fixed by making the docstring raw and
saying why it is raw, and every docstring in both modules was then audited for parsed control characters —
**none remaining**.

### The sharpened rule

**Two checks, not one, and they find different things:**

| check | reads | catches |
|---|---|---|
| **byte inspection** | the file on disk | a control character written literally by a transport that collapsed `\` |
| **parsed inspection** | `pattern.pattern`, `__doc__`, the live object | an escape the file spells correctly and **Python then interprets** |

**Neither subsumes the other.** A regex whose source reads `r"\bsk-"` is fine in bytes and fine parsed; one
whose source reads `"\bsk-"` is fine in **bytes** and **dead** parsed; one written through a collapsing
transport is **dead in bytes** and dead parsed. **Only running both checks distinguishes the three.**

**Standing rule, extended:** for any security-adjacent pattern, assert the regression arm in both directions
**and** inspect both the committed bytes and the parsed runtime value. **Two instances of one failure class
in one afternoon, from two different directions, is not bad luck — it is the shape of the hazard.**

---

# PACK 11 — the post-PACK-10 wave (2026-08-21)

Opened at integration `e616846de75e2098e3fb76592665955b3cfe3bbc`. Takeover verified once:
branch, local = `origin` = `git ls-remote`, no merge in progress, empty index, G11 **3,096
artifacts / 0 non-compliant / exit 0**, **zero sprint-owned Python processes** (only the two
protected `ms-python.isort` LSP servers), `C:\t\heavylock` absent, and all seven protected
scratch modifications plus the four untracked `topics_*.txt` intact.

## Integration baselines re-measured at tip — not copied from the register

All three through `bounded_run.py`, all `FINAL SURVIVING COUNT: 0` / `cleanup: success`,
heavy mutex acquired and released by the wrapper.

| gate | result at `e616846` | verdict |
|---|---|---|
| SMOKE | exit 0, 53.43 s (`INTEG-069/01`) | green |
| Chunk E | **174 passed** in 116.93 s (`INTEG-069/02`) | matches the pin exactly |
| Chunk D (full split-process gate) | **core 159/160**, **node15 failed**, `jobs=28`, `additions=0` (`INTEG-069/03`, nodes under `INTEG-070`) | matches the documented `.env`-conditional baseline **exactly** |

**Chunk D is the one worth stating carefully.** The primary checkout carries `.env`, so the
conditional baseline applies: core 159/160 (one `.env`-conditional red) plus `qb` node15
(pre-charged whenever PathBank is reachable). **Both failures are documented and expected;
neither is new.** The gate observed **173 descendants and terminated 173, surviving 0** — the
wrapper doing exactly the job G11 exists for.

## Findings registered this session

| id | severity | status |
|---|---|---|
| **F-084** | LOW | **NOT a defect — mechanism disproved offline.** Registered *and closed in the same entry* so it is not re-investigated. No live request made; none required. Carries a real but **unreachable** latent sub-finding, and the reason it is unreachable is a property of `openai`'s internals, not of anything this repo controls. |
| **F-085** | MEDIUM | **F-080 names the wrong SHA for C-010.** `9e06360` is its **base**; the merge is **`72ee20f`**. Caught **before** ratification — `DECISIONS.md` is append-only, so the circulated wording would have made a false fact permanent. |
| **F-086** | MEDIUM | The batch preflight's own detector discards submodule names, hiding a **third** undeclared module. **Assigned to C-069**, ceiling raised 400 → 650. Does not consume that card's correction rounds. |
| **F-087** | LOW | `runner.py:1341`'s stated measurement went stale. Found by C-069, correctly reported rather than fixed. No card of its own. |

## PACK 11 RULING 1 — a SHA lifted from a wide Markdown table must name its column

This is the sibling of PACK 10 RULING 1 (*"a control-plane grep restricted to `*.md` does not
search the control plane"*), and it cost less only because it was caught in time.

The LEDGER's card rows carry **two** SHA columns — `Base SHA` and `Merge SHA` — five columns
apart, on rows thousands of characters wide. F-080 quoted *"`MERGED` at `9e06360`"* from row
C-010 and took the **base** for the **merge**. The takeover brief inherited it, and the
proposed ratification text carried it to the threshold of a LOCKED, append-only entry.

**The standing guard, and it is one line:** before citing any sprint SHA, run
`git log -1 --format="%s" <sha>` and confirm the subject names the card you think it does.

**The correction strengthened the underlying reading rather than weakening it.** F-080's
load-bearing claim was that no competing candidate exists for *"the index fix"*. That was
argued from prose; it is now **measured** — `git log --oneline --all --merges | grep -i index`
returns **exactly one commit in the entire repository**, and it is C-010's merge.

## PACK 11 RULING 2 — RULING 13's non-vacuity duty extends to a guard's INPUT derivation

F-086's mechanism is that `_deferred_imports` loses a submodule name **before** `_covered` is
ever called, so `_covered` is asked about the parent package and answers correctly — about the
wrong thing.

RULING 13 requires a mutation harness to prove it can fail. C-069's charter applied that to
the **assertion** (*"neutralize `_deferred_imports` so it returns nothing"*), and the card
delivered it: at base the neutralized guard passed vacuously, at tip it fails with a distinct
message. **But a guard that finds nothing and a guard that asks the wrong question are
different failure modes, and only the first was covered.**

**The sharpened rule:** when a guard derives its own input, the derivation needs its own
non-vacuity arm. *"I found no problem"* and *"I looked in the wrong place"* are
indistinguishable from every level above, and the second is worse because it survives review.

## F-062 — DISPOSITIONED, no card required

Re-measured at tip. The four-way classification the disposition was asked for:

* **defect corrected** — no. The routing seam at `strict_quarantine.py:2273-2275` is
  **byte-identical**: `structural_reasons` is still appended to `refusal_reasons`
  unconditionally, regardless of `defensible_core`. F-062 read the mechanism correctly and
  still does.
* **original trigger removed** — **YES**, and this is the operative one. C-067 (`bb6bb6d`)
  made `_degree_zero_exports` resolve through `_entity_name_norms` (`:1905, 1925, 1953,
  1969`), so on a converged run `degree_zero_export` is empty by construction.
* **proposed remedy superseded** — **YES**, by F-081, on evidence: routing would have shipped
  a review instruction to delete a **connected** enzyme, and because `classify_release_status`
  encodes the same refusal independently, flipping `ok` alone yields `ok: true` with
  `status: diagnostic_only` — shipping a final PWML on a `diagnostic_only` run, breaching
  `PRODUCT_CONTRACT.md:343`.
* **different residual seam still present** — **YES, but adjudicated correct.** The
  unconditional append survives and is the **right** behaviour for the four remaining
  structural reasons, each ruled `keep_refusing` for its own stated reason.

**So F-062 requires no code card.** Writing one now would re-open a seam an independent
biological adjudication ruled should not move.

**The honest limit, and it is why this is a milestone rather than a card:** whether the two
T-100 legs now actually reach `review_required` **cannot be established offline**. The
quarantine input payload is not persisted (instrumentation gap 1 of 3, `FINDINGS.md:1580`,
UNOWNED) — `admitted_payload_hash` is `sha256:b22521ec9dfc4088` while the committed
`final_mapped.json` gives `sha256:7e22a4662dbe2f61` and `merged_payload.json` gives
`sha256:a88b67690be2da81`, so **neither committed file is the payload quarantine judged.**
The confirming measurement is **T-104**, which is the right home for it.

**Residual risk, stated rather than assumed away:** F-081 holds its own core claim at MEDIUM,
not HIGH, and names what would overturn it — *"If the flagged row's synonym set is disjoint
from `keep_norms`, the theorem is wrong and there is a third divergence not yet found."*
**T-104 triage must carry that possibility explicitly in scope.**

## F-077 — REASSESSED, classification holds, accepted as deliberate residual

Verified against current source rather than re-read from the record:

* `schema_version` is **still 6** (`strict_quarantine.py:2481`). No card bumped it, so the
  report-schema decision F-077 is blocked on remains unmade.
* All three closure prunes still exist — `_prune_entities` (`:1493`), `_prune_locations`
  (`:1548`), `_prune_biological_states` (`:1618`) — and `_revalidate_surviving_processes`
  (`:1656`). *(Line numbers moved from the F-077 record by C-067's insertions; the functions
  are the same.)*
* The house-rule comment survives at `:1133`.
* Both scopes remain pinned by `tests/test_strict_quarantine_lineage.py` (19 tests).
* `PRODUCT_CONTRACT.md:85-102` §3 still binds *"the final pathway"*, verbatim.

**A row deleted by a closure prune is not in the final pathway, so §3 does not bind it.**
F-077 remains LOW, deliberate, pinned, and **not a contract violation today**.

**Recorded as accepted deliberate residual behaviour. No card required, and none was
manufactured to make the queue look empty.**

## Correction to `LEDGER.md:231` — the C-056c row is FALSE

That row reads **`DEPENDENCY-READY`, NOT STARTED**. **C-056c is MERGED.** `61d5473` (merge,
*"Merge C-056c (agent/c056c-semantic-evaluability): evaluability travels beside the …"*) and
`f6c8404` (impl) are both ancestors of `HEAD`, and the F-053 carrier is live in source at
`release_status.py:251-272`, `:302-310`, `:373-380`, `:401-417` and
`strict_quarantine.py:2378-2390`.

**This matters because a charter written from the ledger would conclude the carrier does not
exist.** The row is corrected here rather than rewritten in place: it is a 14-column table row
several thousand characters wide, and editing it in place is how the *next* transcription
error gets introduced.

## F-053 — remains UNDISCHARGED, and F-079 is fresh evidence it should stay

Verified rather than assumed: `DECISIONS.md` ends at D-055, and **no entry after D-054 lifts
the prohibition**. D-054 §7 (`:3179-3184`) says in terms that C-056c *"makes the shortfall
visible; it never closes it"* and that lifting it is a separate product decision. So the bar
at `DECISIONS.md:2939-2941` still stands.

**F-079 is the first observed instance of the harm F-053 was written to prevent.** A `passed`
verdict has now been measured on a payload carrying a fabricated actor — and
`strict_acceptance_eligible` came back **`True`**, which is precisely a denominator-entry
authorisation. Had any card been chartered to build a rate on `passed`, this leg would have
entered a strict-benchmark numerator carrying a reaction the paper does not state.

**Re-measured, not copied from D-054 §5:** `ReleaseStatus.semantic_confirmed`
(`release_status.py:285-288`) still has **zero `src/` consumers** — every live `src/` predicate
on `semantic_evaluation` is either subtractive (`bench/acceptance.py:256`,
`release_status.py:522, 541`) or a three-way membership validation (`batch/driver.py:1803`).

**No question is being put to the product owner about F-053.** It stays in force.

## C-071 chartered — F-079

`prompts/C-071.md`, branch `agent/c071-actor-span-gate`, ceiling 700/140/1 MB.

The narrowest verified boundary is **three files**: a new `CHECK_*` constant and one
`CHECK_ORDER` entry in `bench/semantic.py`; one new `_check_*` function plus two lines in
`bench/semantic_production.py`; and **one appended literal** in
`release_status.py :: SEMANTIC_GATING_CHECKS`. **It is deliberately NOT combined with F-078** —
different package, different stage, zero file overlap.

**The engineering choice was taken by the orchestrator rather than left open**: the new check
is *an actor whose entity name does not appear in the span it cites as its own evidence*. Of
the three candidate defects on that leg it is the only one that is string-decidable, offline,
deterministic **and** generalises — it is exactly the F-058 fabricated-transporter class.
A hard constraint makes this non-negotiable: `tests/test_semantic_production_no_gold.py:20,186`
pins that `openai`, `requests`, `httpx`, `socket`, `ssl`, `sqlite3` and two others are **not
importable** from `semantic_production`.

**The card respects F-053 by construction.** It may not read `passed` affirmatively; it must
make the defect produce `SEMANTIC_FAILED` through a gating check and let the existing
subtractive cap at `release_status.py:541` do the demotion.

**⚠ It moves a pinned baseline by construction, and its merge is expected to be HELD.**
`tests/test_semantic_release_gating.py:192-223` asserts the gating set is *"closed at exactly
four"* with the stated purpose *"adding a fifth gate silently is impossible."* That test is
doing its job and this card is the deliberate act it was built to force —
`SEMANTIC_GATING_CHECKS` **4 → 5**. Under the D-044 / D-052 §1 standing rule the disclosure is
the card's obligation and **the ratification is the product owner's**. The card implements
fully regardless, so a one-line answer unblocks the merge rather than starting the work.

**Ten of twelve consuming test files are in no chunk** — the F-054 hazard live — so the
charter enumerates the focused set explicitly instead of naming a gate.

---

## C-070 — ACCEPTED, MERGED `09f7156`, reviewed tip `5bc600e` (2026-08-21, PACK 11)

Exact bare unsuffixed `APPROVE` from REV-070 after **zero correction rounds**. Closes **F-066** (HIGH).

The remedy is one line — `pythonpath = src` in `pytest.ini` — and the card's real contribution is proving
that it is *allowed* to be one line.

### The measurement that was the whole card

F-066 characterised every candidate remedy as *"a sprint-wide gate change … SMOKE and all four chunks would
have to be re-pinned."* Measured, that is an **over-estimate** for this remedy:

| comparison | result |
|---|---|
| Chunk C | **109 → 109**, per-node outcome lists byte-identical |
| Chunk D-core | **160 → 160**, per-node outcome lists byte-identical |
| collected node IDs across SMOKE + Chunk D + Chunk E | **identical, 834 = 473 + 187 + 174** |

**REV-070 re-derived this by a tighter route than the card's** — holding the tree constant and toggling only
`-o pythonpath=`, isolating the single changed line rather than comparing two commits.

**Orchestrator heavy gates on the branch**, all through `bounded_run.py` with the wrapper-owned mutex, all
zero survivors: SMOKE **473**, Chunk A **134**, Chunk E **174**, Chunk D **executed=187/187, omissions=0,
additions=0, failed=none**. Post-merge at integration, SMOKE + the new file: **475 passed, 1 skipped** — 473
unchanged plus two routine arms, the 94 s sweep correctly skipping behind `T2PW_ISOLATED_COLLECT_ALL=1`.

**Nothing was re-pinned, because nothing moved.** Mechanically it cannot: `src` is already on `sys.path` in
every pinned run from two independent sources — the measured launcher **requires** `PYTHONPATH=<tree>/src`,
and 132 files insert it themselves — so the ini adds a duplicate of an entry that is always present, and
duplicate `sys.path` entries are inert.

### PACK 11 RULING 3 — measure the risky claim, do not argue it

The one thing in C-070 that could have been a REJECT was that `pythonpath` inserts `<rootdir>/src` at
`sys.path[0]` **at config time**, so it now beats a `PYTHONPATH` naming a different tree. In a sprint whose
entire measurement discipline rests on proving *which tree was measured*, that is the question that matters.

**The card disclosed it and named it as the thing to scrutinise, rather than hoping nobody noticed.** The
reviewer then measured it four ways instead of accepting the card's argument — and the finding **inverted**:

```
plain python -m pytest, same tree, foreign PYTHONPATH:
  ini ON : t2pw.__file__ = <rootdir>/src/t2pw/__init__.py    <- rootdir's own tree
  ini OFF: t2pw.__file__ = <foreign>/src/t2pw/__init__.py    <- FOREIGN tree
```

Without the ini, plain pytest silently imported `t2pw` from whatever `PYTHONPATH` named. **The change
enforces the very property `tree_pin.py` exists to enforce.** Under `pinned_pytest` it is a no-op either
way, because `tree_pin.resolve_facts` binds `t2pw` in `sys.modules` before `import pytest` and `check`
refuses a mismatch first — re-confirmed post-merge: foreign `PYTHONPATH` still yields
`T2PW_FROM_WRONG_TREE`, *"REFUSED before collection. No test was run. Exit 98."*

**The rule:** when a card changes something the sprint's own measurement discipline depends on, the reviewer
reproduces the hazard rather than evaluating the card's reasoning about it. An argument that a change is
safe and a measurement that it is safe are not the same evidence, and only the second survives a reader who
disagrees.

### F-066's own record was wrong in two places, annotated rather than deleted

**Its 21-file exposure list is wrong in BOTH directions, and its count is right by coincidence.** 155 files,
132 mention `sys.path`, 23 do not; of those 23, **18 fail and 5 pass**; of the 132, **3 fail**. 18 + 3 = 21.
Four files it names collect alone perfectly well (all reach `src` via `from helpers_prefreeze import ...`);
four genuinely failing files are missing, including one that mentions `sys.path` only in a **docstring** and
two that use it inside a test **body**.

**So no static predicate separates the two sets** — which is why the acceptance test is a real sweep over
every `tests/test_*.py` rather than a name list, and why the card's own first design, an "at-risk subset"
sweep, was discarded: **it would have missed 3 of the 21.** A card that discards its own design on its own
measurement is doing the job.

### The acceptance test, and the cost decision inside it

The routine canary is **generated, not named**, so test file 157 is covered the day it is written. The
complete sweep — 156 interpreter starts, ~94 s — is env-gated and skips by default, because a 94-second
test per merge is one someone later deletes. **Runtime was measured and the placement argued, not assumed.**

Non-vacuity: discovery neutralised identically in two arms, the only difference being the `MIN_TEST_FILES`
floor. Unguarded PASSED, guarded FAILED. **REV-070 re-ran this against the SHIPPED module** rather than the
card's deleted probes, by repointing `TESTS_DIR` at an empty directory — a stricter control.

### Findings carried out of this card

* **F-088** — `tree_pin.py:3-4` cited *"`pytest.ini` sets no `pythonpath`"* as a premise. This merge made it
  false. **Fixed at the merge, docstring only**, because the sentence became false as a direct result of a
  merge the orchestrator performed. C-070 correctly did not touch it — its charter says *"Call them; never
  edit them."*
* **F-089** — `tests/test_c030_canonical_identity_fallback.py:88` shells out to `git ls-files` at **import**
  time, and exported base trees exclude `.git`. This is why REV-070's base sweep reported **22** where the
  card's reported 21 — a one-file discrepancy the reviewer ran down instead of waving away. LOW, unowned.
* **Charter defect, mine not the card's.** `_SHARED_BLOCKS.md` § S4 (D-019) requires machine-generated
  evidence to carry *"max artifact count AND a size limit"*. The C-070 charter gave only the size. **Both
  reviewers found the same omission in the charters they reviewed**, and both correctly attributed it to the
  charter. C-071's charter was corrected in flight rather than disclosed at closeout.

---

## C-069 — ACCEPTED, MERGED `8a93da0`, reviewed tip `b08cdce` (2026-08-21, PACK 11)

Exact bare unsuffixed `APPROVE` from REV-069 after **one correction round**. Closes **F-073** (MEDIUM) and
**F-086** (MEDIUM). Three commits, none amended: `29f71a6` → `86d5807` → `b08cdce`.

### The deliberate baseline move, and the register entry it corrects

**`test_every_import_driver_defers_is_covered_by_the_preflight` is STRUCK from the standing pre-charge
list.** The exact delta, using the card's own corrected numbers:

```
CHILD_IMPORTS entries        5 -> 8
missed, occurrences          7 -> 0
missed, distinct modules     3 -> 0
```

**The card's FIRST report claimed 5→7 / 6→0 / 2→0 and corrected itself against its own interest** once F-086
was folded in: the base guard *reported* 6/2, but what was *actually blind* was 7/3. The difference is
precisely the module the guard could not see. REV-069 verified the right-hand column independently.

**The register entry does not go 2 → 1. It goes 2 → 0, and the 2 was never right for integration.**
Measured before the merge as a prediction, then confirmed after it:

| | worktree | integration, base | integration, post-merge |
|---|---|---|---|
| failed | 2 | **1** | **0** |
| passed | 30 | 35 | **37** |
| skipped | 4 | 0 | 0 |

The second "pre-charged failure" is `test_batch_preflight.py:480`'s
`assert venv is not None, "this project ships a .venv; the test assumes it"`. **`git worktree add` does not
copy the untracked `.venv`**, so it fires in every agent worktree and four further tests gated on
`venv_python() is None` skip there and pass here. **It is a measurement-environment artifact, not a red.**

*Prediction discipline note:* the orchestrator first predicted **36** and was wrong — it subtracted the
closed failure without adding the classifier test the card introduces. Corrected to **37** before the merge
on REV-069's `.venv`-realistic measurement, and the merge produced **37 passed, 0 failed**.

### F-086 — the guard that measures the blindness was itself blind

`_deferred_imports` recorded `inner.module` for an `ast.ImportFrom`, so
`from t2pw.pipeline import deadline` (`driver.py:1718`) recorded the **package** and discarded the submodule;
`_covered` then answered `True` off `t2pw.pipeline.export_mode`. **`t2pw.pipeline.deadline` was undeclared,
unvalidatable in the child, and the guard reported zero problem.**

The classifier uses `find_spec(pkg)` — which imports only **ancestors** — plus `pkgutil.iter_modules`.
**The card declined the orchestrator's suggested `find_spec(f"{pkg}.{name}")` and was right to**, though its
first stated reasons did not survive measurement; the surviving reason is that the chosen form resolves
**once per package** and answers every name on a statement from one cached listing.

**The self-introduced hazard is the sharpest thing in the card.** `_submodule_names` falls back to an empty
set when it cannot answer, which **silently restores F-086 behaviour**. Neutralized: the **guard test
passes** while the **classifier test fails**. A reader tells the two apart *by which test is red* — the guard
going green proves only that the question it asked got answered.

### PACK 11 RULING 4 — a rationale is an assertion, and drifts like one

**Three instances in one session**, which makes it a pattern rather than three accidents:

* **F-087** — `runner.py:1341` cites a 2026-07-28 measurement that no longer holds.
* **F-088** — `tree_pin.py:3-4` cited *"`pytest.ini` sets no `pythonpath`"*, which C-070 falsified.
* **This card's correction round** — two `CHILD_IMPORTS` reason strings asserted a failure mode that
  measurably cannot happen.

The author's own diagnosis is the best statement of it and is adopted here:

> *"The defect was **method, not wording** — the deferred site is only the first loss if nothing on the
> child's module-scope path already needs the module, and I never checked whether anything did."*

That is `_SHARED_BLOCKS.md` § S5 exactly — *"Never assert a runtime behaviour from a static code path
alone."* The card measured everything else and **read** this.

**The rule:** prose that states *why* — a reason string, a rationale docstring, a comment justifying a design
choice — is a claim about behaviour and carries the same evidentiary burden as an assertion. It is **worse**
than a stale assertion, because no test goes red when it drifts. **A rationale that cites a measurement
should cite its date and its method**, so a later reader can distinguish staleness from disagreement.

*Measured consequence, so the rule is not abstract:* both corrected strings were **wrong in the direction of
understating the damage.** `strict_quarantine` claimed the child *"still writes all four reports"* — in fact
`streamlit_app.py:52` imports it at module scope, so the child dies executing the app and writes nothing.
`deadline` claimed *"exactly the night's slowest legs"* — in fact `extraction_ladder.py:61` → `pipeline.py:32`
makes it a module-scope dependency, so **every** leg dies at pipeline import. This is operator-facing text
read at 2am.

### The history deliverable — F-073's stated core, and it corrects F-073

```
9e1b9ab ORIGIN_SHA  missed 0   -- GREEN at the sprint baseline
d179c49 C-041       missed 1   + release_status    _finalize_gate_failure()
f3ab5a9 C-031       missed 2   + strict_quarantine _add_identity_artifacts()
985355f C-032       missed 3   + deadline          _finalize_timeout()      <- F-086
57be026 C-053       missed 5   + release_status x2
7e04a1f C-056d      missed 7   + both x1
```

**F-073's *"for as long as they have existed"* is wrong: nine days, 2026-08-12 → 2026-08-20, entirely inside
this sprint.** `CHILD_IMPORTS` had been frozen at five since `48e3669`, the commit that created both it and
the guard. **The `deadline` site was added by C-032 — the card that owns `CHILD_IMPORTS`** — in the same
commit that created `pipeline/deadline.py`. REV-069 spot-checked all five rows by `git blame` and `git log -S`
and independently re-verified the `ORIGIN_SHA` green.

Why none of it surfaced: `tests/test_batch_preflight.py` is in **no chunk**, certified stem-exactly with zero
substring lookalikes. C-053 edited `runner.py` in the very commit that added two of the sites.

### Ceilings — two raises, both orchestrator underestimates, neither charged

400 → 650 when F-086 was folded in mid-flight; **650 → 800** at the correction round, re-measured with
F-050's corrected command: **741** = 169 `src`+`tests` + 572 across three evidence instruments.
**REV-069's 152 was `src`+`tests` alone, which is not ceiling 1.**

All three instruments were mandated by the orchestrator — two by the charter (the history deliverable, the
behavioural arm) and the third by the correction-round instruction to *measure* the reason strings. **A
charter that mandates three instruments and budgets for none is an under-budgeted charter.** C-053 is the
precedent in both directions. **One correction round remains unspent.**

**The card was right not to trim.** D-025 forbids curing an overage by deleting evidence; stopping for a
ruling rather than self-authorizing is what S4 requires.

### Disclosed against the orchestrator, per D-025

* **The charter omitted the artifact COUNT** that § S4 requires alongside a size limit. **Both reviewers
  found this same omission independently, in both charters they reviewed.** Accepted at 24 artifacts /
  204 KB. C-071's charter was corrected in flight rather than disclosed at closeout.
* **The orchestrator relayed the reviewer's marginal-cost figures without re-measuring them.** The author
  measured different numbers and reported the disagreement rather than echoing. **REV-069 then adjudicated
  that there was never a disagreement** — the figures measure different quantities (a warmed process versus
  a controlled bare one) — and **deferred to the author's 18 modules / 0.057 s as the better-controlled
  measurement**, superseding its own 6 / 0.03 s. Its 483 for the guard's import footprint stands, reproduced
  by the author.

### Three LOW findings, none blocking, explicitly not a correction round

* `tests/test_batch_preflight.py:91-92` pairs a module count from the footprint experiment with a timing from
  the bare-interpreter one. Both numbers are real and committed; one word fixes it at the next incidental
  edit.
* `evidence/c069_child_import_reality.py:66-68` justifies its scope by naming `try`-guarded imports; the app
  has none. The real skip is **7 `if`-nested module-scope imports**, and REV-069 **proved the
  under-approximation is conservative** — none of the seven names a target module, and the two exit-1 rows
  already fail on a strict subset.
* `evidence/c069_child_import_reality.py:190` returns `0` unconditionally, with no `--expect` self-validation
  unlike its sibling instrument. It records rather than asserts, so a future matrix regression would be
  written down rather than caught.

### Gates

SMOKE **473** at the corrected tip (`C-069/24`) and **473** post-merge at integration
(`INTEG-069/10`). `test_batch_preflight.py` post-merge: **37 passed, 0 failed** (`INTEG-069/09`).
`driver.py` untouched at zero lines across the entire range — the import at `:1718` was never un-deferred.
`01…24` contiguous. Every job zero survivors, `cleanup: success`.

---

## C-071 — ACCEPTED, MERGED `e4d92fc`, reviewed tips `38cbbf8` then `bbcaa59` (2026-08-21, PACK 11)

**Two bare unsuffixed `APPROVE`s from REV-071 — one on the card, one on the delta — after ZERO
correction rounds.** Closes **F-079** (HIGH) and **F-091** (LOW). Merged only after the product
owner ratified **D-060** (`SEMANTIC_GATING_CHECKS` 4 → 5); the card was implemented to
completion and held, so the ratification was a one-line unblock rather than a start signal.

The full technical record is in the merge commit. This entry keeps the rulings and the lessons.

### The three rulings I made

1. **Boundary: the four-function decomposition is IN BOUNDS.** The charter said *"one new
   module-level `_check_*` function"*; the card wrote four plus four constants. The constraints
   that carry meaning all hold — no existing function body modified, `evaluate_production_semantics`
   gets exactly the call plus the tuple membership, `release_status.py` is literally `+1 −0`, and
   **no new `import` appears anywhere in `src`**, which is what keeps the import-graph and
   no-network pins structurally safe. **The clause meant one check, not one `def`.** A charter
   that forced a 160-line monolith would be worse engineering than what shipped.
2. **Ceilings 700 → 870 and 140 → 280, ratified not charged.** Final 855 changed (642
   `src`+`tests`, 213 the G9 instrument) and 263 doc+comment. **The card was inside 850 at 819
   before I ordered the F-091 sweep**, so the entire increment is work I asked for. The doc
   overage was ratified only after I read the prose and the reviewer independently **checked its
   numbers against the data**.
3. **The widening past my four named F-091 sites is correct — dating, not restating.** Two
   adjacent clauses in `acceptance.py` and `test_c056b` report a **committed measurement
   artifact**. Re-stating them with new numbers without re-running the measurement would claim a
   figure whose source run is not committed, **which § S5 forbids**. The card dated them
   (*"measured BEFORE C-071 widened the set"*) and de-cardinalised only the clause that was a
   claim about the *current* configuration. It disclosed and offered to revert rather than
   hoping. **That is the minimum honest treatment, not scope creep.**

### PACK 11 RULING 5 — a relayed number is a hypothesis, and this session proved it twice

**Twice in PACK 11 a number the orchestrator passed down without re-measuring was wrong, and
both times the implementer caught it by refusing to take it on trust.**

* **C-069.** I relayed REV-069's marginal-cost figures. The author measured 18 modules / 0.057 s
  against my quoted 6 / 0.03 s and reported the disagreement. REV-069 then adjudicated that
  there had never been a disagreement — the figures measured different quantities — and
  **deferred to the author's as the better-controlled measurement, superseding its own.**
* **C-071.** I relayed REV-071's *"20 rows also carry `source_refs`"* as exact. The author
  measured **19 non-empty**, 20 **keys**; the 20th is a `[""]` row. The sentence is about what
  the check can **read**, so 19 is load-bearing.

**REV-071 diagnosed the second incident exactly, and the diagnosis generalises:** the wrong
number entered the record because it was measured with a **truthiness test** — `[""]` is a
truthy list — and then relayed without re-measuring.

**The rule:** a measurement handed down by a reviewer or an orchestrator carries no more
authority than one handed up by an implementer. **All three are hypotheses until reproduced.**
An implementer that reports a disagreement rather than echoing the number it was given is doing
the job, and should never be treated as arguing with its reviewer.

### The card's own best line

Its diagnosis of *why* the F-091 class of defect happens, which is now PACK 11 RULING 4's
concrete case: a **serialized** constant is worse than a stale comment, because a stale comment
misleads a maintainer reading source while a stale serialized constant misleads a reviewer
reading a run's **output**, who has no reason to suspect the string and no way to check it from
the artifact alone.

### Findings carried out

* **F-090** (MEDIUM, UNOWNED) — `bounded_run.py`'s descendant enumeration vs RULE 5's 64 KiB
  cap. A clean job produced a 149,703-byte non-compliant record and forced a gap into the very
  sequence D-025 uses to detect evidence tampering. **The deletion was accepted because it was
  disclosed**; a silent one would have been a reject.
* **F-091** — closed by this merge.
* **New, LOW, not charged:** REV-071 found my F-091 sweep incomplete — the `"four of four"`
  idiom survives at six further sites. **None ships**, three are in `strict_quarantine.py`
  (out of bounds), and all six use it as C-056c's idiom for *fully evaluated* inside text
  describing a measurement taken when the set had four members — the historical class we just
  agreed to date rather than restate. **`acceptance.py:245` is the one worth a pickup**: it sits
  three lines below an edit the card did make, so one paragraph now dates its measurement in one
  sentence and says "four-of-four" in the next.

### Gates

SMOKE **473** on the branch and **473** post-merge at integration. Chunk D on the branch
**`executed=187/187, omissions=0, additions=0, failed=none`** — the run the author asked for,
covering the 23 `test_streamlit_quarantine_boundary.py` AppTest nodes it flagged as unmeasured
and correctly refused to call proven from reasoning alone.

---

## T-103 (M4) — RAN 2026-08-21, `runs_verify/2026-08-21_2057`

4 legs, **1h13m**, exit 1, `FINAL SURVIVING COUNT: 0`, `cleanup: success`, heavy mutex acquired and
released by the wrapper. `T2PW_SPECIES_LLM=0` (mandatory, PACK 9 RULING 3) and
`T2PW_OFFLINE_CURATOR=1`, both in the bounded child environment via the shell-prefix form D-058
corrected.

| leg | outcome | wall |
|---|---|---|
| PMC12452463 strict | FAIL (contract) | 15m05s |
| PMC12452463 research | PASS WITH WARNINGS | 17m54s |
| PMC12096016 strict | FAIL (contract) | 15m31s |
| PMC12096016 research | PASS WITH WARNINGS | 23m59s |

### Acceptance — SATISFIED for one round, and the qualification is the important half

Acceptance is *"every RAG round re-entered normalization, mapping, gates, persistence,
classification."* Measured on all four legs:

```
PMC12452463 strict    round_count=1  mapped_ids_added=2  locations_added=1   items=5
PMC12452463 research  round_count=1  mapped_ids_added=1  locations_added=11  items=11
PMC12096016 strict    round_count=1  mapped_ids_added=4  locations_added=12  items=13
PMC12096016 research  round_count=1  mapped_ids_added=4  locations_added=3   items=7
```

Every round did real work, and `contract_reports.json` shows the stage chain re-entered, each
report carrying its own `stage`:

| report | stage | proves |
|---|---|---|
| `post_normalization_contract_report` | `post_normalization` | **normalization** |
| `stage2_contract_report` | `post_mapping` | **mapping** |
| **`post_remap_contract_report`** | **`post_remap`** | **mapping RE-ENTERED — a second mapping contract after the audit** |
| `post_audit_contract_report` | `post_audit` | audit |
| `pre_export_runtime_schema_report` | — | **gates** |
| artifacts on disk | — | **persistence** |
| `quarantine_report.json` → `release` | — | **classification** |

**`post_remap` is the load-bearing one.** A single mapping report proves mapping ran; a *second*
one after the audit is what "re-entered" means.

### ⚠ The qualification, stated rather than glossed

**`round_count = 1` on every leg, so "every round" means exactly one round per leg.**
`RAG_LOOP_MAX_ROUNDS` is unset, so `rag_loop_max_rounds()` returns 1 — which is the production
default, and D-058 predicted 1× before the run. **This confirms the prediction.**

**But it means multi-round re-entry is NOT tested by this run.** The acceptance says *"every
round"*; with one round, the universal claim is verified over a set of size one. A second round
re-entering correctly is **unmeasured**, and anyone quoting T-103 as proof that the loop
re-enters on iteration N > 1 would be over-reading it.

**T-103's status is therefore `MEASURED — acceptance satisfied at round_count=1; multi-round
re-entry untested`.** Not PASS without that qualifier.

---

## ⭐ The result that matters most: PMC12452463 reaches the contractually required status

This was not T-103's acceptance criterion. It fell out of the run, and it closes a chain this
sprint has been working on for weeks.

`runs_verify/2026-08-21_2057/papers/PMC12452463/strict/quarantine_report.json`:

```
status                      review_required
strict_acceptance_eligible  false
strict_gates_passed         true
degree_zero_exports         []
closure_converged           true
entity_type_overlaps        []
unexportable_entities       []
minimum_core_satisfied      True        coverage.reasons  []
refusal_reasons             []
semantic_evaluation         failed
semantic_failed_checks      ["actor_named_in_its_own_cited_span"]
reasons                     ["semantic_evaluation_failed:actor_named_in_its_own_cited_span"]
missing_anchors             ["EntA", "Fur"]
```

**`PRODUCT_CONTRACT.md:341` (§13, LOCKED) reads:** *"Correct outcome after the index fix is
`review_required` with `strict_acceptance_eligible=false`. **Never strict success.**"*

**D-056 ratified that "the index fix" is C-010 (merged `72ee20f`), so that row binds today — and
the measured outcome matches it exactly.**

### Four things this settles, each by measurement rather than argument

**1. T-100's acceptance criterion / TRAP-1 is met.** *"PMC12452463 → `review_required`, not strict
success"* (`TEST_MATRIX.md:477`). Measured: `review_required`, `strict_acceptance_eligible=false`.

**2. F-062 is CONFIRMED CLOSED, and the confirmation arrived earlier than expected.**
`F-062-DISPOSITION.md` concluded F-062 needs no code card because C-067 removed the trigger, and
said the confirming measurement belonged to **T-104** because the quarantine input payload is not
persisted. **T-103 produced it incidentally.** `strict_gates_passed = true` and
`refusal_reasons = []` — **no structural reason fires at all**, so the seam F-062 described cannot
be reached on this leg. The disposition is now measured, not inferred.

**3. F-081's own MEDIUM caveat is REFUTED.** F-081 held its theorem at MEDIUM and named what would
overturn it: *"If the flagged row's synonym set is disjoint from `keep_norms`, the theorem is
wrong and there is a third divergence not yet found."* Measured: **`degree_zero_exports = []`**.
C-067's synonym-resolving detector finds nothing on the leg that used to carry the identical
`Isochorismatase (EntB)` flag. **There is no third divergence.** F-081 may be upgraded to HIGH.

**4. C-071 is working in production, on the exact paper the contract names.** The demotion is
carried by `actor_named_in_its_own_cited_span` — the check merged hours earlier — firing on a
real leg, not a fixture. It is the sole reason for the demotion; every structural gate passed.

### One honest qualification on the mechanism

**The leg reaches the right STATUS, but not by the route the contract's rationale describes.** The
gold `export_rationale` calls the route chemically broken because **EntA is absent**. The pipeline
does record that — `missing_anchors: ["EntA", "Fur"]`, `expansion_blocked_reason: "2 requested-core
anchor(s) matched no admitted process: EntA, Fur"`, `completeness: 0.857` — but coverage still
passed (`minimum_core_satisfied: True`), and the demotion is carried by the **semantic** gate
instead.

**So the contract's required outcome and the pipeline's reason for producing it are not the same
fact.** That is worth stating before anyone quotes this as end-to-end vindication. **It is a real
question for T-104 triage**, and it should be carried there explicitly rather than discovered
again.

---

## C-072 - ACCEPTED, MERGED `d7f4f96`, reviewed tip `b86cc41` (2026-08-22, unattended correction phase)

**One `APPROVE` from an independent reviewer after ZERO correction rounds.** Closes **F-094**
(HIGH, `product_contract_violation`) - the first of the two corrections D-063 makes T-105 wait on.

`agent/c072-incomplete-core-demotion`, worktree `C:/t/c072`, base `20e6b68`.
Diff: `release_status.py` +57/-0, `tests/test_c072_incomplete_core_demotion.py` +454/-0.
**Insertions only; zero deleted lines in `src`.**

### The change

One new cap in `classify_release_status`, at `release_status.py:605`:

```python
if status == RELEASE_READY and verdict is not None and verdict.declared and missing:
    status = REVIEW_REQUIRED
    reasons.append(f"{REASON_REQUESTED_CORE_ANCHORS_UNMATCHED}:{','.join(missing)}")
```

A declared requested core with one or more anchors matching no admitted process is incomplete,
so `release_ready` - which asserts no human review is needed - is not available.

### The ruling I made: why NOT in `evaluate_core_coverage`

The obvious placement is a fourth clause in `evaluate_core_coverage.reasons`, which feeds
`minimum_core_satisfied = not reasons` (`strict_quarantine.py:960`). **Rejected.** All three
existing clauses are THRESHOLD questions; adding an unconditional one would make the pinned
`min_core_coverage = 0.5` non-load-bearing and break the two tests that pin it
(`test_release_status_classification.py:301`, `test_strict_quarantine_release_seam.py:222`).
Placing the cap in the classifier leaves the coverage function untouched and the threshold
meaningful. No existing test changed value anywhere in the card.

### Why this is not a re-firing of the semantic gate

F-094's content is that the required outcome had been resting entirely on C-071's
`actor_named_in_its_own_cited_span`: at T-103 that gate fired and the leg reached
`review_required`; at T-104 it did not fire and nothing else held the leg. A correction that
only makes the semantic gate fire again rebuilds the same single point of failure.

The new cap reads **no semantic field** - its three operands are `status`, `verdict.declared`
and `missing`. It sits after the semantic cap and fires when that one did not. Both the
implementer and the reviewer proved it with `semantic_evaluation == "passed"`,
`semantic_failed_checks = ()` and every structural gate green.

### G9 - independently reproduced, not accepted on report

The reviewer did NOT reuse the implementer's stash-based base measurement. It cut a detached
worktree at `20e6b68`, confirmed the new constant was absent, and ran the new test file there
tree-pinned via `pinned_pytest.py --expect-tree`:

```
AssertionError: assert 'release_ready' == 'review_required'
```

`5 failed, 6 passed` on base - collection succeeded and imports resolved, so this is a
**behavioural** failure, not symbol absence. The six that pass on base are the control arms,
which assert UNCHANGED behaviour; that is the correct shape. The constant-name test is
explicitly labelled as not being the proof, and the module deliberately does not import the
constant at module scope, which is why base collection did not error.

### Gates

| gate | result |
|---|---|
| focused (card 9, 6 files) | `99 passed` (implementer) |
| obligation set (card 8, 9 files) | `133 passed` (reviewer) |
| blast radius, 16 files | `543 passed` (reviewer) |
| SMOKE at tip | `473 passed`, pin verdict `violations: []` (reviewer) |
| SMOKE on integration after merge | **`473 passed`** (orchestrator) |
| `test_strict_failure_replay.py` | `2 failed, 37 passed, 8 skipped` - **exactly the pre-charged baseline**, measured by orchestrator AND reviewer |
| process lifecycle | `FINAL SURVIVING COUNT : 0` and `cleanup : success` on **every** job, all parties |

**SMOKE is 473, measured three times this session at three different tips.** `CLAUDE.md` merge
rule 10 still says 465 and is stale by the C-067 delta; `TEST_MATRIX.md` 473 is correct.

### Blast radius - measured by the orchestrator, larger than F-094 described

F-094 says PMC12452463/strict was "the only strict leg producing a bare `pathway.pwml`". True,
and it hid something: **three T-104 legs finished `release_ready`, not one.** The other two are
research legs, which emit no PWML and so were invisible to the artifact-name argument. All three
carry a declared core with unmatched anchors, so C-072 demotes all three:

| leg | T-104 | completeness | unmatched anchors |
|---|---|---|---|
| PMC12452463/strict | `release_ready` | 0.800 | DHB-AMP, Fur, RyhB |
| PMC12096016/research | `release_ready` | 0.667 | NADH, L-serine, ATP, EntH, Fur, MenD |
| PMC12782028/research | `release_ready` | 0.563 | MSMO1, SQLE, FDFT1, HMGCR, HMGCS1, FDPS, MVD |

The two research demotions are a **deliberate, contract-consistent side effect**, not a
surprise: a declared core missing 6 of 18 and 7 of 16 anchors is incomplete on exactly the
reading that condemns PMC12452463. Recorded here so T-105 triage does not rediscover them as a
finding. PMC12856317/strict - completeness 1.0, zero unmatched anchors - is the control that
proves the cap is not a blanket demotion, and must stay unchanged at T-105.

### It fixes a contract violation and moves NO acceptance rate

Measured, not assumed:

- **Strict PWML success stays 0/4.** The four gold `strict_exportable` cases are PMC12657337,
  PMC12421875, PMC12096016, PMC12782028. PMC12452463 is `export=partial_only` and was never in
  that denominator; PMC12096016/strict and PMC12782028/strict were ALREADY `review_required`.
- **Research deliverable stays 4/8.** `acceptance.py:605` computes `deliverable` as a pure
  filename test (`research_pathway_report.txt`) and never reads `release_status`.
- **Strict deliverable is unaffected.** `_STRICT_DELIVERABLES` (`acceptance.py:99`) accepts BOTH
  PWML names by design - D-004 split them precisely so "did an importable file land?" stays
  separate from "may this count as strict success?".

That the correction moves no rate is the correct outcome, not a disappointment: the rates were
already failing for other reasons. **A reviewer or a later session must not read the unchanged
0/4 as evidence that C-072 did nothing.**

### Reviewer's open item, closed here

The reviewer flagged that "the T-105 strict denominator should be expected to move by more than
one leg, and that delta is unmeasured". It is measured above: three legs move status, zero rates
move. No further action.

---

## C-073 — ACCEPTED, MERGED `6373ad1`, reviewed tips `01f2bd3` (REJECT) then `2be4740` (APPROVE) (2026-08-22, unattended correction phase)

**One evidence-backed REJECT, one correction round, then APPROVE.** Closes **F-096** partially —
see "What it does NOT fix", which is the more important half of this entry.

`agent/c073-identity-admission`, worktree `C:/t/c073`, base `20e6b68`.
Diff: `mapping/identity_admission.py` +433 (new), `mapping/map_ids.py` +307,
`pipeline/entity_admission.py` +47, `tests/test_c073_identity_admission.py` +964 (new).
**+1751 / −0. Zero deleted lines in `src`.**

### The rejection, and why it matters more than the approval

The first tip refused a shared accession whenever its claimants had **different names**. My card's
§2 handed the implementer a "zero collateral" figure and told them not to re-derive it. That figure
was measured over the ten T-104 legs only. **Over all 53 committed `final_mapped.json` artifacts the
rule strips 92 claimant-incidences across 36 distinct rows in 10 artifacts, of which exactly ONE is
the F-096 target.** The other 41 pairs are one entity written two ways, both rows legitimately
owning the accession:

```
uniprot:P0ADI4        EntB / Isochorismatase (EntB)                    x6 artifacts
uniprot:P10378        EntE / enterobactin synthase                     x2
8 namespaces          PEtN / Phosphoethanolamine
chebi:16412           LPS / lipopolysaccharide
uniprot:A0A0H3GEM5    LMRG_02730 / MenI          (locus tag vs gene symbol)
uniprot:P0A7A7        PlsB / PlsB glycerol-3-phosphate acyltransferase
```

The reviewer also found it **contradicts D-035 clause 3c**, which rules that a matching stable
external identifier is *proof two differently-named rows are the same biological entity*. The rule
read that identical fact as proof at least one row was false, and stripped it from both.

**The card error was mine.** A measurement handed to an implementer must carry its sample size, not
just its result; "do not re-derive this" removed the only check that would have caught it. The
implementer was entitled to rely on what the card asserted.

### The correction

Refuse a shared `(namespace, accession)` **only when two claimants differ in KIND and in normalized
NAME** (`identity_admission.py:398-401`, `left[0] != right[0] and left[1] != right[1]`):

- **kind** = protein-ish (`proteins`, `protein_complexes`) vs compound-ish (`compounds`). One
  accession cannot denote a protein and a metabolite at once. This is also what F-096 actually
  found — the gold calls the PLP row `cofactor_as_protein`, a **type** error.
- **name** is retained so a routing artefact — one entity landed in both buckets under one name —
  cannot read as a type error.
- **within one kind nothing is touched**, which is exactly D-035 clause 3c.

### Measured, three times independently

| corpus | old predicate | new predicate |
|---|---|---|
| 53 committed artifacts, 547 eligible rows | 42 pairs / 92 incidences / 10 artifacts | **1 pair / 2 rows / 1 artifact** |

Target caught **1/1**, collateral **0**. Derived by the implementer, by the reviewer, and by the
orchestrator, with three separate scripts.

The 53-artifact replay is now a **test**
(`test_the_whole_committed_corpus_yields_one_conflict_and_no_collateral`), not a probe, so the
evidence cannot rot. It is **corpus-pinned**: it will go red the first time a committed run carries
a genuine cross-kind defect. That is a deliberate re-baseline, not a regression.

### What it does NOT fix — read this before quoting the card as closing F-096

**Priority 1 does not move. It stays at 7.**

- **Pass A (source support) is DORMANT in production.** My card's §4a asserted that
  `screen_additions` receives `seed_text` on the batch path, citing `streamlit_app.py:5554/:5674`.
  **False** — those are `maybe_run_rag` and `run_rag_rounds`. `final_payload` is built by
  `merge_additions` at `streamlit_app.py:5606`; the RAG merge is `:5660`; the one EDITABLE merge
  site, `pipeline.py:732`, produces only an inner QA-loop payload. Both merge sites document the
  omission as deliberate. Corroborated: zero `evidence_span_not_locatable` removals across all ten
  T-104 legs. Arming it is two lines, one inside `streamlit_app.py`, which carries the product
  owner's uncommitted edit and must not be touched. **BLOCKED on the product owner, not on code.**
- **The accession conflict is not counted by priority 1.** `semantic.py:908-919` appends
  `accession_claimed_by_multiple_entities` to `findings` but **never increments `false_real`**;
  only `:867` and `:893` do. So the one thing this card fixes in production is invisible to the
  metric it was chartered against.

Pass A itself is sound and measured — 1 catch (`succinyl-CoA`, PMC12180156/research, the gold's
designated HALLUCINATION TEST), 0 collateral over 102 rows, reproduced by the reviewer. It is
correct code waiting on a wiring decision.

**Also unreachable by design, and correctly not attempted:** `Pyridoxal 5'-phosphate` ×2, `SREBF1`,
`SREBF2`, `LIPA`, `LBR`. All are genuinely named in their own papers, carry correct accessions for
the molecule named, and are byte-identical in provenance (`extracted`, confidence 1.0) to the 16
legitimate cholesterol enzymes in the same leg. Their falseness is a biological ROLE judgement. No
heuristic was smuggled in to reach them; the reviewer verified that.

### Gates

| gate | result |
|---|---|
| focused + all affected @ tip | `240 passed` (reviewer) |
| `map_payload` consumers + real-artifact replay | `315 passed` (reviewer) — `FULL_STACK_BASELINE` identical at base, round 1 and round 2 |
| implementer full set incl. smoke | `725 passed` |
| SMOKE at tip / after merge | `473` / **`473 passed`** |
| lineage golden | byte-identical, `md5 ad1827e42faf7807867a3da2a64724de` |
| C-060 A6 | green; narrowing is future-only, pinned by `test_seed_text_alone_never_writes_the_index` |
| G9 | **6** behavioural base failures available, **2** claimed. Underclaimed, the safe direction |
| process lifecycle | `FINAL SURVIVING COUNT : 0` and `cleanup : success` on all 39 G11 jobs |

### Rulings I made

1. **The §5 / §4b inconsistency resolves in the implementer's favour.** §5's table said
   "`map_payload` tail only" while §4b named `_mapping_lineage_facts` as the site to record the
   refusal. The edit is one guarded branch, inert on any payload without the new record
   (`map_ids.py:8805`), and the golden is byte-identical. In bounds.
2. **Conditional emission of `report["identity_admission"]` is honest and stays.** It is emitted
   only when an index is present or something was withheld, which keeps the pinned golden still.
   Crucially "no index offered" reports as silence or `not_evaluated`, **never** as "supported" —
   verified on all 53 index-free artifacts.
3. **Keeping `Kdo-lipid IV_A` / `lipid IV A` is correct.** They are genuinely distinct molecules
   sharing `chebi:60365` — a real mis-resolution — but there is no type error, so an identity-layer
   gate has no basis to pick a victim. D-034 already ratified a loud fail-closed refusal for that
   class; a second silent response could remove the very collision that makes the leg refuse.

### Residues registered, none blocking

- **The composite case.** If one pair has two same-kind claimants AND one cross-kind claimant, all
  three lose the accession, including the two innocent 3c-agreeing rows. Occurrence on the committed
  corpus: **0/53**, asserted explicitly by the replay. Defensible — once an accession provably
  denotes both a protein and a metabolite, no claimant carries evidence it owns it — but
  `identity_admission.py:34-38` overstates the guarantee by saying within-kind rows are left
  "completely alone". Worth a docstring correction, not a code change.
- **F-099** — withholding a PathBank scalar is not durable to pre-freeze resolution.
- Module docstring and the G9 note in the test file still say "collision" where the concept is now
  "kind conflict". Cosmetic.

---

# Final correction wave before T-106 — session of 2026-08-23 (second session)

Opened at integration tip `129d9b2`. Local = origin = `git ls-remote` verified at start; no merge
in progress; 0 staged; heavy lock absent; 0 sprint-owned Python processes (the two
`ms-python.isort` servers are the product owner's IDE and are never touched).

**Correction to the handoff's start-state table.** `HANDOFF-T106.md:16` records the product-owner
edit's file hash as `sha256:e50a248bb7189c22…`. The measured value is
`sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632`, and the file's mtime
(2026-08-23 13:12) **predates the handoff's own** (15:55), so the file has not moved since the
handoff was written and the recorded hash is simply wrong. The load-bearing invariant — **35
insertions / 2 deletions, uncommitted** — holds and has been re-verified after every commit this
session. Use the measured hash from here on.

## Live-run ledger

**No live paper leg, and no LLM-backed command, has been run this session.** Every measurement so
far is current-source inspection or deterministic replay of committed T-104 / T-105 artifacts.
Implementation and measurement lanes are all explicitly forbidden from running live legs; the
orchestrator is the sole owner of live execution.

| # | paper / mode | purpose | status |
|---|---|---|---|
| — | — | — | none run yet |

Preflight facts established for the cohort and T-106, offline:

* LM Studio is up at `http://127.0.0.1:1234/v1` and serves
  `text-embedding-nomic-embed-text-v1.5`, which is `RAG_EMBEDDING_MODEL` in `.env`.
* `.env` pins `deepseek/deepseek-v4-flash` for `OPENROUTER_MODEL`,
  `OPENROUTER_EXTRACTION_MODEL` and `OPENROUTER_CURATOR_MODEL`. No fallback model was enabled.
* PathBank is reachable: all 11 `db_resolution` records in the committed corpus read
  `available: True` with no unavailability reason.
* Cohort topic files written (untracked, deliberately): `topics_cohort_both.txt` (PMC12856317,
  PMC13231680 — both modes) and `topics_cohort_research.txt` (PMC12180156, PMC12782028 — research
  only). Six legs, which is the §12 minimum without paying for two unneeded strict legs.

## Findings this session

| finding | class | severity | blocks T-106 | carded |
|---|---|---|---|---|
| **F-106** — seed provenance mark carries a pathway name where a paper title belongs | `reporting_defect` | LOW | **no** | **no — deliberately not carded** |
| **F-099** — AMENDED, re-measured after the pass was armed | `product_contract_violation` | LOW → **HIGH** | **yes** | **C-078** |

F-106's registered premise was corrected on measurement: the three provenance fields are not
mutually inconsistent, only `source_title` is wrong, because Stage 0's context schema has no
document-title field at all. C-075's verdict is unaffected and this was proved, not asserted — its
route clause tests carrier presence only, and a fabricated tuple, a `{"x": 1}` carrier and the real
tuple give byte-identical outcomes.

## Cards dispatched

| card | finding / ruling | branch | worktree | base | state |
|---|---|---|---|---|---|
| **C-076** | F-102 · identity ruling 2026-08-23 | `agent/c076-alias-holo-apo-identity` | `C:/t/c076` | `129d9b2` | dispatched |
| **C-077** | F-095 · **D-062** (LOCKED, never implemented) | `agent/c077-stage0-conflict-disposition` | `C:/t/c077` | `129d9b2` | dispatched |
| **C-078** | F-099 as amended | `agent/c078-refusal-durable-through-resolution` | — | `9831fc1` | chartered, queued behind a free lane |

### C-077 — the D-062 seam, measured before chartering

`driver.py:2130` calls `_reconcile_stage0_scope` at step 3b, after Stage 1 and **before** the
payload counts, audit, DB mapping, freeze and export. With `stage0_conflict_aborts` true
(`config.py:194`, the default) it sets `outcome.status = _STATUS_SCOPE_CONFLICT` and returns, and
no release classification is ever attached.

All six T-105 `scope_conflict` rows confirm it: `stage=stage1`, `release_status=null`, and every one
had already written `stage1_payload.json` and `merged_payload.json`. The observed organism was read
correctly and is already recorded — PMC12421875/strict carries `observed_organisms:
["Lactococcus lactis"]` against the requested `Bacillus subtilis`.

The detection is correct. The classification is the untruth: `OUTCOME_SCOPE_CONFLICT` sits in
`INELIGIBLE_OUTCOMES` (`eligibility.py:123-131`) whose docstring reads *"nothing was attempted, so
nothing failed"*, and `report.py:49/140` imports that set so `_norm_status` folds the row to
`STATUS_INELIGIBLE`, which repeats the claim. For these six legs it is false on the evidence.

The card is scoped to the disposition only. Driving the run onward into audit and export under a
contradicted organism would be a **new product decision nobody has taken**, and is the one shape
that could accidentally produce the strict export D-062 forbids.

## Sequencing decision — F-099 and F-105 are NOT one card

§10 of the session charter asked whether they form one coherent seam. Measured: they do not.
F-099 is pre-freeze compound identity admission (`pwml/compound_resolution.py`); F-105 is prompt
serialisation in the interactive curator (`curation/interactive_curator.py:164`). They share a
cause — the source-support pass being armed — but not a security or provenance boundary, and the
charter's own rule is to keep them separate unless the boundary is shared. **Separate.**

F-099 is batch-reachable and on the T-106 path, so it is carded now as C-078. **F-105 is not
T-106-reachable** — its own registration says the batch legs T-106 runs do not touch the
interactive curator — so it queues behind the blockers and must close before the interactive app is
used against a real paper, not before T-106.

The F-105 fix shape is already measured and is small: `strip_payload_for_interactive_context`
(`interactive_curator.py:164`) is a blacklist, and the allow-list constant it should consult already
exists as `PROMPT_OMITTED_PAYLOAD_KEYS` (`curation/audit_json_llm.py:41`), keyed off
`identity_admission.SOURCE_INDEX_KEY`.

## Merges — final correction wave

| card | merge | closes | SMOKE after | reviewer |
|---|---|---|---|---|
| **C-077** | `26fa809` | F-095 / **D-062** disposition | **473** (51.83 s) | APPROVE, actual diff |
| **C-076** | `3b7a7b1` | **F-102** scorer/gold identity | **473** (41.93 s) | APPROVE, actual diff |

Both reviewers re-measured rather than quoting the implementer, and both found the committed G11
evidence insufficient to confirm SMOKE on its own — the JSON retains no stdout and the pin verdicts
carry no counts. Each ran both sides itself. That is now a standing expectation for this gate.

### C-077 — what shipped, and the one thing it did not

`diagnostic_only` with an explicit `stage0_scope_conflict_stopped_the_run_before_serialization`
reason, plus `requested_scope` recorded beside `observed_context`. **Not D-062's literal
`review_required`**, deliberately and disclosed.

The reviewer proved that state unreachable two ways: by reading (`classify_release_status` is one
`elif` chain whose second arm pins `diagnostic_only` when the strict gates did not pass, and all
five `REVIEW_REQUIRED` sites sit below it or are guarded on `status == RELEASE_READY`), and by a
**196,800-combination sweep** of the documented input surface with `strict_gates_passed=False`
pinned — `diagnostic_only` 196,800 times, `REVIEW_REQUIRED` zero — against a control arm at
`True` returning 24. The only routes to `review_required` were to fabricate a gate result,
hand-build a `ReleaseStatus` (forbidden by the charter), or edit an out-of-bounds module.

Registered as **F-107** for a product ruling. Two non-blocking residues the reviewer surfaced:
the record inherits `strict_technical_gates_blocked_export` so `describe()` renders *"strict gates
failed"* for a leg whose gates never **ran**; and the `scope_conflict` → `STATUS_INELIGIBLE` fold is
unchanged, so D-062's §2b untruth is corrected in the machine-readable record but not in the
rendered summary. Both belong with F-107's ruling.

### C-076 — the out-of-table test edit, and why it was allowed

`tests/test_bench_goldset_and_semantic.py` is outside the §5 table and in SMOKE. The reviewer proved
the edit was **forced, not convenient**, by running the base payload through the tip scorer:
`MenD`/`MenF` sharing `P80867` now yields `ok=True` with no findings, so the base assertion would
have failed and merge rules 4 and 10 would have broken. The obligation that test encoded is exactly
the rule the product owner's ruling rejects. Node id unchanged (it is a context line), file count
62 → 62, both assertions byte-identical — only the payload moved from within-kind to cross-kind.

Cross-kind safety was established by exhaustive probe, not argument: all **246** combinations of
2–4 rows over 3 buckets × 3 names sharing one accession, zero disagreements with the pipeline's
`find_kind_conflicting_accessions`, zero genuine cross-kind conflicts suppressed. Structurally the
tip branch adds a condition to the base branch, so tip findings ⊆ base findings — the change can
only remove findings, never invent one.

**Delta for T-106's baseline** (do not apply to T-105): `accession_claimed_by_multiple_entities`
2 → 0; priority 1 `false_real_identifiers` 7 → 6; priorities 2/3/4/5 identical; **no leg's gating
`no_real_id_or_name_conflict` flips**; 2 legs moved, 18 unchanged.

### A known gap the ruling leaves open

The ruling says *"biologically unrelated **or** cross-kind"*. Both seams implement **cross-kind
only** — there is no biological-relatedness oracle in either. Two genuinely unrelated same-kind
proteins fused onto one accession by a mapper bug are now invisible to the benchmark. That mirrors
the pipeline's pre-existing blind spot and is what charter §4 Arm A directs, so it is not a
deviation — but the "unrelated within one kind" half is **unmeasured corpus-wide** and is recorded
here as a known gap rather than an assumed non-issue.

### The F-108 flip question — adjudicated

The C-076 reviewer concluded that aligning the production seam would flip `PMC12452463/strict` to
`release_ready`, and flagged its own probe as *"indicative, not definitive"*. **Adjudicated against
that reading**, and both accounts are written into `prompts/C-080.md` §2e so the implementer settles
it rather than inherits it.

`classify_release_status` applies its caps in order, each guarded on `status == RELEASE_READY`. The
semantic cap (`release_status.py:693`) fires first in the recorded run and demotes to
`review_required`, which makes the incomplete-core cap (`:730`) unreachable — so **its reason never
reaches the persisted record**, and that absence is precisely what made the single-element
`semantic_failed_checks` list look conclusive. Remove the semantic failure and the order inverts:
status stays `release_ready` through `:693`, and `:730` fires because the leg has `declared: true`
and `missing: [DHB, EntA, Fur]`. The in-source comment at `:715-730` labels that cap **"A SECOND,
INDEPENDENT"** guard, for F-094 specifically.

The §2d replay executed the real classifier on each leg's own recorded coverage verdict and
reproduced all three recorded statuses **exactly before the toggle**, which is what makes its
counterfactual trustworthy.

### Live-run ledger — still empty

**No live paper leg and no LLM-backed command has run this session.** Everything above is current
source, committed unit/integration tests, or deterministic replay of committed T-104 / T-105
artifacts. Two merges, five charters, four findings, zero credits.

## Merges, continued

| card | merge | closes | SMOKE after | reviewer |
|---|---|---|---|---|
| **C-078** | `4797f58` | **F-099** (as amended) | **473** (52.91 s) | APPROVE |
| **C-079** | `15a8a15` | **F-105** | **473** (50.46 s) | APPROVE |

### C-078 — the charter was wrong four times and the implementer caught all four

Confirmed independently by the reviewer:

* **`Fe3+` is the one REALIZED hit, not the control the charter called it.** Its PathBank scalar is a
  *consequence* of an admitted name-keyed match, proved four ways: `db_match.raw_name` is the
  extracted name, `db_status` is `matched` not `legacy_id_unverified`, `db_row` is present (written
  only on the admitted branch), and `mapped_ids` is byte-equal to a wholesale rebuild from the DB row.
* **5 rows exposed, 0 landed.** All five `2,3-diDHB` rows are `db_status: ambiguous`, so the apply
  took its not-matched early return. Realized blast radius on the committed corpus is **1**.
* **The refusals are not C-073's** — grepping `identity_admission` across both run directories
  returns zero files. 10 of 11 came from the name gate's `no_shared_meaningful_token`.
* **The honest denominator is 130 compound rows / 122 refusal-free**, not 379/368, which required a
  recursive walk reaching 22 `quarantined_proteins` rows this code never sees.

The rename was AST-verified byte-identical, 2354 bytes each side.

**The accepted cost.** The one row this patch changes carries a **biologically wrong** refusal —
ferric iron *is* Fe³⁺ *is* KEGG C14819 — so nine correct identifiers are stripped. It ships because
**D-028 (LOCKED) already decided this exact trade at this exact seam** and named its own casualties
under *Measured effect*: *"legitimate identity lost, 3 of 124 real rows, stated rather than hidden"*,
including `Zn²⁺ → Zinc`. Same case. The locked requirement is that the loss be **stated**, and it is.
Not a new product decision.

And leaving it was strictly worse: the DHB rows refused `pathbank_compound_id: 40770` while the
resolver offers **41128** — a *different* record. The mask hid false restores as well as false
refusals.

### C-079 — non-vacuity proved by mutation, not assertion

64,952 bytes stop entering every interactive curator round (65,777 → 825).

The reviewer ran **two mutants**: deleting the allow-list clause breaks 8 of 16 tests, with the
non-vacuity arm failing on its own guard message (*"neutralizing the clause changed nothing — the
comparison above proves nothing"*); deleting the blacklist clauses instead breaks exactly the three
additivity tests. A silently inert filter is caught either way.

**The regression that would have mattered most did not happen:** `read_source_index` on the caller's
payload still returns non-`None` at tip, and the stripper's two call sites neither feed mapping —
so C-075's refusal of `succinyl-CoA` is intact.

The new leaf module is **in boundary**: the charter pre-authorised a relocation, and every other
destination is foreclosed — `identity_admission` is out of bounds, `interactive_curator` has zero
`t2pw` module-scope imports at base, and `audit_json_llm` pulls in `openai`/`httpx`/`dotenv`/
`t2pw.llm.client`, which **constructs an `OpenAI` client at module scope and raises on a bad
provider**. Reproduced on both trees.

**One finding recorded rather than waved through:** the implementer's reason for skipping the
AppTest cohort rested on a false premise — `streamlit_app.py:1884` calls `compact_mapping_misses` on
every refinement-review init, which *does* reach the changed predicate. The reviewer closed the gap
with evidence rather than argument: full 187-node Chunk D gate at tip, `SETS_EQUAL=True`, 187/187,
`failed=none`.

## Findings registered this session

| finding | class | blocks T-106 | disposition |
|---|---|---|---|
| **F-106** | `reporting_defect` | no | deliberately not carded |
| **F-107** | `policy_disagreement` | no | needs a product ruling |
| **F-108** | `product_contract_violation` | **yes** | **C-080** |
| **F-109** | `control_plane_contradiction` | no | doc owner's call |
| **F-110** | `product_contract_violation` | no — **but predicted** | card after T-106 |
| **F-111** | `process_tooling_gap` | no | card after T-106 |
| **F-112** | `test_accounting_staleness` | no | re-baseline after T-106 |

`F-099` amended LOW → HIGH. `F-092` amended: two of its three defects **refuted**, not merely
reclassified.

## The affected-paper cohort — expanded beyond the handoff minimum, deliberately

`topics_cohort_both.txt` (4 papers × 2 modes) + `topics_cohort_research.txt` (2 × research) =
**10 legs**.

The handoff's minimum was 6, written before C-076/C-078/C-080 existed. Four legs were added because
the correction wave moved their exact seam:

* **`PMC12096016`** (both modes) — carries `uniprot:P0ADI4` `EntB`/`holo-EntB`, the accession
  C-076/C-080 stopped flagging, **and** the `Fe3+` row behind F-110.
* **`PMC12452463`** (both modes) — carries `uniprot:P10378` `EntE`/`enterobactin synthase`, and is
  the leg `T106_PREDICTION.md` §3 singles out as the one that may reach `release_ready` against
  gold's `partial_only`.

Learning that in a 10-leg cohort costs two extra legs. Learning it inside T-106 costs a release
candidate. `--fresh` into a new run directory; without it `batch_run.py` silently skips finished
pairs.

## Live-run ledger

**Still empty at this point.** Four merges, six charters, seven findings, two amendments — all from
current source, committed tests, or deterministic replay of committed artifacts. **Zero credits.**
The cohort is the first live spend of the session and does not start until C-080 merges.

---

# T-106 — RAN 2026-08-24. `MEASURED — NOT ACCEPTED`. Never record as PASS.

`runs_verify/2026-08-24_1428`, committed `efca465`. 19,294 s (5.36 h),
`deepseek/deepseek-v4-flash`, real curator, LM Studio embeddings.
**`COMPLETE (10/10 papers, 20/20 legs)`.** G11: `FINAL SURVIVING COUNT 0`, `cleanup success`, heavy
lock acquired and released. `cache_snapshot` (43 MB) excluded from the commit.

Preflight: stage-only staged 20 pairs, 0 skipped; `--verify-plan` → `verdict: OK`, 10 cases,
**0 search calls**, all 10 `[pinned_override]`, three organism traps still reading
`Bacillus subtilis`. The real run continued that plan without `--fresh`.

## Acceptance, against T-105

| priority | T-105 | **T-106** |
|---|---|---|
| 1 zero false real identifiers | FAIL 7 | **FAIL 8** |
| 2 zero unsupported reactions | FAIL 3 | **PASS 0** |
| 3 zero referential violations | FAIL 2 | **PASS 0** |
| 4 requested-pathway coverage | PASS 1/8 | **PASS 1/8 = 12%** |
| 5 strict PWML pass rate | FAIL 0/4 | **FAIL 0/4 = 0%** |

**Priorities 2 and 3 moved FAIL → PASS.** Separated denominators are unchanged from T-105:
extraction success 8/8 = 100%; semantic pathway success 1/8 = 12% (PMC12421875 again); research
deliverable 4/8 = 50%; research confirmed 0/8.

## Priority 1 is 8, and the composition is the finding — not the count

`NADH`, `NAD+` (PMC12096016/research); `LIPA`, `LBR` as `heading_or_prose` and `SREBF1`, `SREBF2` as
`regulator_as_metabolite` (PMC12782028/research); `pyridoxal 5-phosphate` as `cofactor_as_protein`
(both PMC12856317 legs).

Every one is a **Stage-1 extraction hallucination handed a real identifier downstream** — the
pre-existing **F-096** class. **Not one is an accession-conflict case and not one is a
restored-refused identifier**, so none is attributable to C-076, C-078 or C-080.

**`holo-EntB`, which was in T-105's seven, is gone** — C-076's predicted delta realised on a fresh
run. `SREBF1/2`, `LIPA` and `LBR` are back after vanishing from T-105 by draw variance. **7 → 8 is
composition churn**, precisely what `T106_PREDICTION.md` was written to stop anyone reading as a
regression.

`Lpt system`, `RyhB` and `enterobactin synthase complex` all report
`forbidden_identity_present_unmapped` — flagged by gold, carrying no identifier. Correct outcome.

## What the correction wave bought, measured on a release candidate

* **Zero bare `pathway.pwml` across all 20 legs.** Five strict legs emitted
  `pathway.review_required.pwml`, each with `strict_acceptance_eligible: false`. At T-105 two legs
  shed bare PWML as **F-100** and **F-101**; that class is closed.
* **C-077 validated.** Six `scope_conflict` legs as predicted, and all six carry a real release
  record where T-105 had `null`: `diagnostic_only`, `pipeline_executed: true`, `elig: false`,
  reasons naming `stage0_scope_conflict_stopped_the_run_before_serialization`, and `requested_scope`
  beside `observed_context`. Zero PWML on all six. D-062's disposition, working on real legs.
* **PMC12452463/strict did not reach `release_ready`** — `fail` at `post_pipeline`,
  `diagnostic_only`. The hard gate in `T106_PREDICTION.md` §3 held on the run as well as the cohort.
* **Priorities 2 and 3 at zero.**

## Two predictions did not hold — recorded, not explained away

* **F-092's two PMC12444477 TIMEOUTs did not reproduce.** The strict leg **passed** with a
  64,359-byte `pathway.review_required.pwml`. F-092's surviving defect 3 is **not observed on this
  run** and remains open on the T-104/T-105 evidence rather than being re-confirmed.
* **A new terminal outcome instead:** `PMC12444477/research` ended `error`/`crash` on
  `AMBIGUOUS_RENAME_TARGET`. The guard is pre-existing (C-050h, `999209e`) and fired in neither
  T-104 nor T-105; a draw emitting both `Escherichia coli K-12` and `Escherichia coli` exposed it.
  Nothing in this wave touches species canonicalisation. Registered as **F-115** — the concern is
  that a correct fail-closed guard *terminates the leg* instead of preserving the payload as
  `review_required`.

## Why it is NOT ACCEPTED

Priorities 1 and 5 fail. Priority 1's eight are all F-096-class Stage-1 hallucination, which no card
in this wave was chartered to fix; priority 5 is 0/4 because no strict-exportable paper produced a
`release_ready` export — five produced `review_required` instead, which is correct behaviour and not
strict success. **Do not rerun under the T-106 identity. Do not relabel as PASS.**

## Live-run ledger — closed

| # | run | legs | purpose | result |
|---|---|---|---|---|
| 1 | `runs_verify/2026-08-24_1203` | 8 | cohort A, affected papers both modes | clean, hard gate cleared |
| 2 | `runs_verify/2026-08-24_1402` | 2 | cohort B, research-only | clean, source-support pass withholding |
| 3 | `runs_verify/2026-08-24_1428` | 20 | **T-106 release candidate** | `COMPLETE`, `MEASURED — NOT ACCEPTED` |

Three live runs, 30 legs, all through the bounded wrapper with zero survivors and cleanup success on
every one. Everything before them — five merges, six charters, ten findings, two amendments — cost
no credits at all.

---

# POST-T-106 CORRECTION SESSION — opened at `e648287`

Lead Orchestrator session opened 2026-08-25. Starting state verified in full: tip `e648287`,
local = origin = `git ls-remote`, no merge in progress, 0 staged, heavy lock `C:/t/heavylock`
**absent**, **zero** sprint-owned Python processes (only the two `ms-python.isort` `lsp_server.py`
IDE servers, never killed), product-owner `streamlit_app.py` edit intact at **35 ins / 2 del,
uncommitted**, `sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632`,
`TEST_MATRIX.md` 578 lines with line 477 byte-identical, tracked caches modified and never staged.

## Cards dispatched

| card | finding | branch | worktree | base | lane |
|---|---|---|---|---|---|
| **C-081** | **F-096** — Stage-1 hallucinations receive real identifiers (priority 1 = 8) | `card/C-081-f096` | `C:/t/c081` | `e648287` | writer A |
| **C-082** | **F-115** — `AMBIGUOUS_RENAME_TARGET` crashes the leg instead of preserving the payload | `card/C-082-f115` | `C:/t/c082` | `e648287` | writer B |
| **C-083** | **F-092 defect 3** — the inner deadline path discards the `operation_timeout` it computed | *charter written, not dispatched* | — | — | queued behind C-082 (`driver.py`) |

`prompts/C-083.md` is committed with the card unopened, because its fix necessarily **moves a
pinned golden baseline** (`tests/test_deadline_leg_timeout.py:125-135`
`test_a_driver_timeout_keeps_its_row_byte_identical`, plus the golden driver diff that hashes
`RunOutcome.to_dict()`). That is permitted only under merge rule 4 with an exact documented delta,
and the charter says so in terms: deleting the pin instead of re-baselining it is an automatic
reject.

## Measurements taken before any card returned — offline, zero credits

### SMOKE re-verified independently at the tip

`473 passed in 49.68 s`, exit 0, `FINAL SURVIVING COUNT : 0`, `cleanup : success`.
G11 `evidence/g11/T-107/02-smoke-baseline-e648287.json`. The handoff's figure is confirmed rather
than inherited, so every merge gate this session measures against a number this session took.

### F-110 is NOT REACHABLE on T-106 — no card is justified

The finding predicted that C-078 would make the name gate strip metal ions and formula-named
compounds. **It did not fire even once.** Measured across the entire T-106 run directory including
`batch.log`:

```
no_shared_meaningful_token          0 occurrences
identity_refused_review_required    0 occurrences
shipped_identity_name               0 occurrences
```

Ion- and formula-shaped names **do** occur — but every one sits in a leg that terminated before DB
mapping ever ran, which is why the gate was never consulted:

| name | leg | payload written | why mapping never ran |
|---|---|---|---|
| `Mg2+`, `Mg2+ cofactor of MenD` | PMC12312563/research | `merged_payload.json` only | `scope_conflict`, stops at Stage 1 |
| `Mg2+`, `Mg2+ binding to MenD` | PMC12312563/strict | `merged_payload.json` only | `scope_conflict` |
| `ferric iron (Fe3+)` | PMC12452463/strict | `merged_payload.json` only | failed `stage3_normalization_gate` |
| `Zn2+` | PMC13231680/research | `merged_payload.json` only | negative control, 0 reactions |
| `NAD+` | PMC12096016/research | `final_mapped.json` | mapping DID run — but this is an **F-096** false-identifier row, not an F-110 refusal |

**Consequence:** F-110 is *precisely non-blocking* for T-107, and §15's cohort obligation ("if that
seam affected a real T-106 case") is **not triggered**. No live leg is owed to it.

**Coupling that must be carried forward:** F-110 becomes reachable the moment a ruling on F-107 /
D-062 drives `scope_conflict` legs onward into DB mapping — `PMC12312563` carries `Mg2+` in **both**
modes. Whoever implements that ruling inherits F-110 with it.

Incidental: `ferric iron (Fe3+)` as a *name* would dodge the defect anyway, since it carries the
token `Fe3+` that the gate looks for.

### Priority 5 — the ceiling is 1/4, and three of the four are unreachable by code

Measured from `runs_verify/2026-08-24_1428/manifest.jsonl`, not inferred from the report:

**`PMC12096016/strict`** — `status: pass`, `strict_gates_passed: true`, `release: review_required`
```
semantic_evaluation    : failed
semantic_failed_checks : ['actor_named_in_its_own_cited_span']
completeness           : 0.764706
reasons                : ['semantic_evaluation_failed:actor_named_in_its_own_cited_span']
```

**`PMC12782028/strict`** — `status: pass`, `strict_gates_passed: true`, `release: review_required`
```
semantic_failed_checks : ['requested_pathway_anchors_present', 'actor_named_in_its_own_cited_span']
completeness           : 0.222222
reasons                : ['requested_core_coverage_below_minimum:0.222<0.500']
```

| paper | what actually blocks it | reachable by code? |
|---|---|---|
| PMC12421875 | **D-062, LOCKED**: the correct outcome is `review_required`, and D-062 states in terms that *"a `review_required` artifact is not a strict export"* | **no — product ruling** |
| PMC12657337 | same | **no — product ruling** |
| PMC12782028 | completeness **0.222 < 0.500**, 21 missing anchors including `cholesterol` itself; gold `relevance=partial`, 3 reactions at 33% attribution | **no — correctly blocked** |
| PMC12096016 | **`actor_named_in_its_own_cited_span` alone**; completeness 0.765 is comfortably above the floor | **only if that predicate is defective** |

**The strict denominator contains two papers that locked policy forbids from ever passing.** D-062's
own closing section left the gold-versus-ruling reconciliation explicitly open — *"neither paper
counts as a strict-export success and the strict denominator is unchanged"* — and F-107's
"what a ruling would need to settle" names that same open question as its point 4.

**So the T-107 go/no-go hinges on one predicate.** If `actor_named_in_its_own_cited_span` is firing
correctly on PMC12096016, priority 5 is 0/4 with **no code remedy at all** and §16 forbids the run
until the product owner rules. If it is defective, priority 5 can reach 1/4 and the run is no longer
guaranteed to fail. The check fired on **both** remaining papers, and `T106_PREDICTION.md`'s
amendment §C had already flagged it as one of three draw-sensitive gating production checks.

### C-077 was NOT a silent divergence — recorded so it is not reopened

The T-106 scope-conflict legs carry `diagnostic_only`, not D-062's literal `review_required`. That
was **deliberate and disclosed**: the C-077 reviewer proved `review_required` unreachable at that
seam by reading the `elif` chain and by a 196,800-combination sweep, and the gap was registered as
**F-107** for a product ruling. It is not a defect in C-077 and must not be re-litigated as one.

### T-107 preflight — the configured model is reachable

`provider=openrouter`, model `deepseek/deepseek-v4-flash`, endpoint reachable.
**Session-start usage baseline `161.092487097`; `limit_remaining` $69.44**, recorded so the cohort's
and T-107's real cost can be reported as a delta rather than estimated. G11
`evidence/g11/T-107/01-preflight-model-availability.json`, exit 0, zero survivors, cleanup success.

Note the charter's "LM Studio" wording is stale — `.env` selects OpenRouter. Per §6 the configured
model stands; no provider switch and no new fallback.

---

## CORRECTION to `03f60b0` — the ceiling is 0/4, not 1/4, and priority 5 is guaranteed to fail

**Measured by REV-081 and independently confirmed by the Lead Orchestrator against source.
`03f60b0`'s claim is FALSIFIED and is corrected here rather than quietly amended.**

`03f60b0` recorded: *"That leaves PMC12096016, blocked by `actor_named_in_its_own_cited_span`
alone. The whole T-107 go/no-go now hinges on whether that one predicate is firing correctly."*

**Both halves are wrong.**

### The error

I read `semantic_failed_checks` having exactly one entry as *"one blocker"*. It is not. It is the
**first cap to fire** in a chain of caps that are independent by construction, and the source says so
in terms. `release_status.py:701` introduces the anchor cap as *"The INCOMPLETE-CORE cap (F-094,
PRODUCT_CONTRACT 13). **A SECOND, INDEPENDENT**"* cap:

```python
release_status.py:693   if evaluation == SEMANTIC_FAILED and status == RELEASE_READY:      # cap 1
release_status.py:730   if status == RELEASE_READY and verdict is not None \               # cap 2
                           and verdict.declared and missing:
```

Each is guarded on `status == RELEASE_READY`, so **whichever fires first hides the others**. Cap 1
demoted PMC12096016, so cap 2 never ran — and `semantic_failed_checks` therefore records a single
entry on a leg that has two independent blockers. On PMC12096016 `missing_anchors` is
`['NADH', 'ATP', 'MenD (…)', 'Fur (…)']` and `completeness` is `0.764706`, so `verdict.declared` is
true and `missing` is non-empty: **cap 2 fires regardless of the semantic outcome.**

### The proof

REV-081 replayed the real `classify_release_status` over each leg's own recorded
`coverage_summary.json`. Arm A is the control and reproduced **9 of 9** recorded statuses exactly,
which is what makes arm B admissible. Arm B forces the semantic verdict to `passed`:

| paper | recorded | arm A control | **arm B, semantics forced to `passed`** | arm B reason |
|---|---|---|---|---|
| PMC12096016 | `review_required` | reproduced | **`review_required`** | `requested_core_anchors_unmatched:NADH,ATP,MenD…,Fur…` |
| PMC12782028 | `review_required` | reproduced | **`review_required`** | `requested_core_coverage_below_minimum:0.222<0.500` |
| PMC12421875 | `diagnostic_only` | reproduced | **`diagnostic_only`** | `strict_technical_gates_blocked_export` |
| PMC12657337 | `diagnostic_only` | reproduced | **`diagnostic_only`** | `strict_technical_gates_blocked_export` |

Evidence: `evidence/g11/REV-081/07-rev081-counterfactual.json`.

**Priority 5's ceiling under unchanged policy is 0/4. Not one of the four moves even if every
semantic check passes.**

### And the predicate I flagged as the hinge is a TRUE POSITIVE

`actor_named_in_its_own_cited_span` is firing **correctly** on PMC12096016. It caught a real
biological error, so "fixing" it would be weakening a gate to raise a rate — the exact inversion
this sprint exists to correct.

### Corrected disposition table

| paper | class | what actually blocks it |
|---|---|---|
| PMC12096016 | **`correctly_blocked`**, twice over | cap 1 on a true-positive actor finding; cap 2 (`:730`) fires underneath it regardless |
| PMC12782028 | **`correctly_blocked`** | coverage 0.222 vs a 0.500 floor `PRODUCT_CONTRACT` §7 `:199-200` pins as immovable; retrieval demonstrably ran and was exhausted — `rag_admission_report.json`: 15 typed gaps, **0 accepted, 159 rejected** |
| PMC12421875 | **`policy_disagreement`** | D-062's open reconciliation |
| PMC12657337 | **`policy_disagreement`** | D-062's open reconciliation, **plus** an independently gating `no_rejected_rag_reaction_reintroduced` failure |

**Zero of the four are reachable by code that does not weaken a gate or manufacture bare PWML.
Every route to 1/4 or better runs through a product ruling first.**

### Two further premises of mine that REV-081 corrected

* **The Stage-0 abort never replaced an annotate-only mode.** I inferred from `config.py:193`'s
  comment that it had. `git log -S eligibility_stage0_conflict_aborts` returns exactly one
  introducing commit — `15a7998`, 2026-07-29, before the sprint branch point — with the flag **born
  at `True`** alongside both branches. Annotate-only is a live alternative today
  (`driver.py:669-670`), never a removed default.
* **`_requested_scope`'s `pinned` is a HOMONYM.** `driver.py:517` sets `pinned = not pathway`,
  meaning *"this paper has no topic line at all"* — **not** the gold set's `[pinned_override]`.
  `scope_conflict.json` recording `"pinned": false` is therefore **correct**. Nobody may "fix" this
  by conflating the two senses.

### What stands from `03f60b0`

The F-110 non-reachability measurement, the SMOKE 473 verification, the model preflight, and the
C-077/F-107 note all stand unchanged. Only the priority-5 ceiling claim is corrected.

---

## F-116 — a supported enzyme is replaced by a superset protein complex, injecting catalysts that do not perform the step

- **Severity** HIGH · **Class `product_contract_violation`** (candidate — see the competing reading)
- **Registered 2026-08-25 by the Lead Orchestrator**, measured by REV-081 from committed T-106
  artifacts. **Distinct from F-096**: F-096 is an *unsupported* entity gaining an identifier; this is
  a *supported* entity being *replaced* by a wrong one.

`_rewrite_reaction_protein_enzymes_to_complexes` (`src/t2pw/mapping/map_ids.py:8668`) replaces
bare-protein reaction actors with PathBank `protein_complex` wrappers. It is strict-mode-only, gated
by `allow_complex_wrapper_creation=not research_mode`, and its in-source rationale is that *"the
PathWhiz importer refuses a bare protein as a reaction enzyme."*

On `PMC12096016/strict` it resolved **EntE** onto PathBank complex **3623**, whose components are a
strict **superset**: `EntB P0ADI4, EntD P19925, EntF P11454, EntE` — `chosen_rule
pathbank_protein_complex_id`, `confidence 1.0`, `source db`. Its siblings resolved to
**one-component** wrappers that preserve identity exactly: EntC→1143, EntB→1189, EntA→1190.

**The biological consequence is real.** The 2,3-DHB adenylation step (EC 6.2.1.71, EntE alone) now
carries EntB, EntD and EntF as catalysts, none of which performs it. `reactions[4]`
("EntF-catalyzed enterobactin synthesis") collapsed onto the **same** complex, so two chemically
distinct steps became indistinguishable by actor — destroying the gold's own requirement of *"a
named enzyme per step"*.

This is what `actor_named_in_its_own_cited_span` caught, and why that finding is a true positive.

**Corpus reach: 3 of the 6 strict legs that reached mapping.** The same generator produced
PMC12452463's Stage-3 gate deaths
(`gate.entity_enterobactin_synthase_complex_is_declared_as_both_a_prote`,
`gate.generated_protein_complex_wrapper_enterobactin_synthase_complex_`). The `ALAS2 → ALAS2
complex` shape passes; the `EntE → enterobactin synthase` and `CYP51A1 → Lanosterol 14-alpha
demethylase` shapes fail.

**The competing reading a product owner must weigh:** the enterobactin synthase assembly line
genuinely does perform both steps *in vivo*, so the attribution may be imprecise rather than
chemically false. That judgement is not the orchestrator's.

**It does not unblock the leg.** Arm B proves `:730` fires anyway. Fixing this is contract-grounded
work that *strengthens* biology; it is not a route to a strict export.

**Attribution honesty:** the seam is bracketed by two payload snapshots, not lineage-proven. The
actor substitution left **no lineage carrier of its own** — `reactions[3].provenance_lineage` still
reads `origin: paper_stated`, `paper_explicit: explicit`. A per-actor rewrite record in
`mapping_meta` (source protein → chosen complex → component delta) would make this attributable
rather than inferred. **Unowned; no card opened** — it is out of C-081's boundary and must not be
folded into it.

---

## F-117 — the actor gate cannot relate a DB canonical name to a bare gene symbol

- **Severity** LOW · **Class `product_contract_violation`** · **Blocks nothing; costs nothing today.**
- **Registered 2026-08-25**, measured by REV-081.

On `PMC12782028/strict`, `actor_named_in_its_own_cited_span` fires on entity
*"Lanosterol 14-alpha demethylase"* against the span *"CYP51A1 catalyzes the conversion of lansterol
to …"*. The wrapper is PathBank complex **442, one component, `CYP51A1`, `Q16850`, gene `CYP51A1`` —
**the identity is exact**. The check fires only because the normalized names share no whole token,
aggravated by the paper's own typo `lansterol`.

Proof it is the typo and not the biology: `reactions[2]` carries the **identical** actor with
correctly-spelled `lanosterol` and **passes**. `audit_iteration_summary.json` records the audit LLM
proposing exactly that fix, `accepted_patch_count: 1`.

`semantic_production.py:371-378` documents the loose one-shared-token rule precisely to tolerate
*"the same protein under a DB or wrapper name"* — but it cannot help when the span names **only**
the gene symbol. A live blind spot in a **gating** check.

**It changes no disposition.** PMC12782028's recorded cause is the coverage floor, and the semantic
cap at `:693` never runs because it is guarded on `status == RELEASE_READY`. Recorded so a future
reader does not mistake it for the EntE case, which is the opposite — a true positive.

---

## An observation off the release path, recorded but not classified

PMC12096016/strict scored `retained_reactions_match_supported_signatures` **0/7 attribution, 0/3
recall** while the *same paper's research leg* scored 2/6 on a near-identical payload with identical
5/5 enzyme recall. That 0% vs 33% swing on one paper suggests a brittle signature matcher.

It is **not** in `SEMANTIC_GATING_CHECKS` (`release_status.py:114-120`, a CLOSED set of five:
`requested_pathway_anchors_present`, `organism_compatible`, `no_real_id_or_name_conflict`,
`no_rejected_rag_reaction_reintroduced`, `actor_named_in_its_own_cited_span`), it did not demote the
leg, and acceptance priority 2 measured **0** unsupported retained reactions. Not chased, not
classified — flagged as an observation only.

---

## A live tension the product owner should see

PMC12096016's cap-2 demotion cites unmatched anchor **`MenD`** — which the gold's own
`export_rationale` for that paper says *"Export must **exclude** MenD"*. The pipeline is being
demoted for correctly omitting something the gold forbids. `Fur` is a regulator; `ATP` and `NADH`
are cofactors.

Cap 2's input is Stage 0's `key_compounds` / `key_proteins`, **not a curated core**. The cap itself
is a merged F-094 correction and is not reopened here — but the *quality of its input* is a real and
separate question, and it is currently costing a leg that has 5/5 enzyme recall and 7/8 metabolite
recall.

Stage 0's draw is also non-deterministic: PMC12096016's `missing_anchors` was `[ATP, NADH, NAD+,
EntA, Fur]` at T-104, `[ATP, EntD]` at T-105 and `[NADH, ATP, MenD, Fur]` at T-106 — coverage
0.706 / 0.857 / 0.765 on one paper across three runs.

---

## C-081 — MERGED `b869780`. Priority 1 goes 8 → 6. **It does NOT close F-096.**

| card | merge | closes | SMOKE | review |
|---|---|---|---|---|
| **C-081** | `b869780` | **F-096 partially — 2 of 8** | **473** (52.39 s, post-merge, pinned) | APPROVE, actual diff, 12 bounded jobs |

**The rule.** A row declaring `class:"cofactor"` that no reaction or transport uses as input,
output, enzyme, modifier, cargo or transporter may not ship real accessions. Implemented as PASS C
inside `map_ids._admit_identities`, reusing the existing `_withhold_identity` unmodified.

**It is not a rule against cofactors.** `ATP` declares the same role ten times corpus-wide, is used
by a reaction every time, and is untouched. The sharpest evidence that this is *role consistency*
and not molecule blacklisting: `NAD+`/`NADH` are refused on one leg of `PMC12096016` and **kept ×7
each on other legs of the same paper**, and `Pyridoxal 5'-phosphate` is **kept ×3** on PMC12856317
legs where a reaction does use it. Same molecule, same paper, opposite outcome, decided by the graph.

**Ordering is load-bearing, and proven so by mutation.** PASS C must run **last**. The implementer's
first draft ran it before PASS B and dropped C-073's corpus test from 1 conflict to 0 — refusing the
compound first hides the `drugbank:DB00114` collision and leaves the **protein** `ALAS2` holding a
compound accession. The reviewer built a reorder mutant and it went red **on C-073's own untouched
assertion at `:597`**, which is independent confirmation that the narrowed C-073 test kept its guard.

### What the reviewer measured itself, not inherited

* **SMOKE 473 at base `e648287` (40.79 s) and 473 at tip `b71fe0b` (36.74 s)** — both its own runs.
  I measured 473 post-merge at `b869780` (52.39 s, exit 0) and pinned the collect count at **473**.
* **Blast radius reproduced with its own replay script: 18 refusals, 0 collateral.** It then ran the
  decisive experiment the card did not: replaying with a **schema-complete** participant reader
  (all ten `payload_models.py` name keys plus `elements_with_states` and the legacy fields) rescues
  **0 of the 18**. That is stronger anti-collateral evidence than the card itself offers.
* It checked all 18 against the pinned gold **paper by paper**. 15 exact/alias matches; the three
  "alias-misses" verified molecule-by-molecule — `pyridoxal phosphate`, `thiamine diphosphate
  (ThDP)` and `tetrahydrofolate (THF)` each ship accessions byte-identical to the exact-match row
  for the same paper.
* **G9 reproduced and behavioural**: the base run **collected all 25 tests** — no ImportError, no
  collection error — and failed at `test_c081_cofactor_role_identity.py:214` on
  `AssertionError: a cofactor-role row that no reaction uses shipped real external accessions:
  {'drugbank': 'DB00114', 'hmdb': 'HMDB0001491', 'kegg': 'C00018', 'chebi': '18405', 'pubchem':
  '1051'}`. Symbol absence is not what was measured.
* **Affected sweep: 683 passed, 2 failed — identically on BOTH arms.** Both are the already-registered
  stale corpus-size pins (F-112's class). Not C-081's doing.
* **Chunk E real-artifact replay: 174 passed**, `FULL_STACK_BASELINE` unmoved.

### The scope statement, made plainly by both implementer and reviewer

Reviewer's own replay over the T-106 artifacts, re-scored with `bench.semantic`'s forbidden-identifier
rule:

```
PMC12856317/strict   before=1 after=0   REMOVED  "Pyridoxal 5'-phosphate"
PMC12856317/research before=1 after=0   REMOVED  "Pyridoxal 5'-phosphate"
PMC12096016/research before=2 after=2   REMAINS  'NADH', 'NAD+'
PMC12782028/research before=4 after=4   REMAINS  'LIPA','LBR','SREBF1','SREBF2'
T-106 priority 1: 8 -> 6
```

The six survivors are correctly out of reach of *this* rule: `NADH`/`NAD+` **are** used by a reaction
on that leg, so a role-consistency rule has nothing to say about them; the four proteins declare no
cofactor role at all. **Acceptance priority 1 remains FAIL, and it is absolute.**

### Decision compliance checked by the reviewer

Open **O-1** (`placeholder_backed_proteins`) is **not** resolved by this card and was not improvised
on: `_identity_admission_eligible` returns `{}` for `is_pathbank_unknown_protein` and
`IDENTITY_PLACEHOLDER`, so PASS C never sees a placeholder row. `PRODUCT_CONTRACT` §8 is satisfied —
`cofactor_participation` returns `not_evaluated` for no-role, no-reactions and no-evaluable-name, and
PASS C `continue`s on all three rather than treating `not_evaluated` as `false`.

---

## F-119 — the participant reader is narrower than the schema it reads

- **Severity** MEDIUM · **Class `product_contract_violation`** (latent) · **Measured reach: ZERO.**
- **Registered 2026-08-25 by the C-081 reviewer.** Did not block the merge; needs a follow-up card.

`identity_admission._PARTICIPANT_NAME_KEYS` is narrower than the runtime schema **and narrower than
the codebase's own existing definition of the same notion**:

```python
identity_admission.py:676   ("entity", "name", "ref", "id")
canonical.py:330            ("entity", "compound", "protein", "protein_complex",
                             "nucleic_acid", "element_collection", "name")
```

`payload_models.py` declares `ProcessParticipantModel.{compound, protein, protein_complex, element,
element_collection, nucleic_acid}` and `ActorModel.{protein, protein_complex}` — **none readable by
C-081**. `PARTICIPANT_FIELDS["reaction_coupled_transports"]` lists four fields that **do not exist**
on `ReactionCoupledTransportModel` while omitting `elements_with_states`, the one that does;
`TransportModel.elements_with_states` is omitted too.

The reviewer probed seven schema-legal shapes: **six produce a false refusal of a cofactor a reaction
genuinely uses.** e.g. `modifiers: [{"protein": "Pyridoxal 5'-phosphate", "role": "cofactor"}]` →
`cofactor_withheld: 1`, `mapped_ids` emptied.

**Why it did not block:** measured reach is zero. In all 89 committed `final_mapped.json`, reaction
`inputs`/`outputs` are always bare strings, every `elements_with_states.element` name is redundantly
present in `cargo`/`transporters`, and `reaction_coupled_transports` is empty in **all 235** payload
artifacts scanned. Same precedent as F-110: measured non-reachable → register, do not block.

**Why it must not be forgotten:** the failure direction is **stripping a correct identifier**, and
`{"protein": …}` is the *dominant* actor shape in the corpus — **1,820 occurrences against 615 for
`entity`** across 423 payload files. The margin is thinner than the card's write-up implies, and the
module docstring presents the enumeration as deliberate and complete when it was not derived from
`payload_models.py`. A follow-up card inside the same file should reconcile `_PARTICIPANT_NAME_KEYS`
with `canonical.py:330` and add `elements_with_states`.

---

## F-120 — C-081's anti-collateral arm is weaker than the measurement that licensed it

- **Severity** LOW · **Class `test_accounting_staleness`** · **Registered 2026-08-25 by the reviewer.**

`tests/test_c081_cofactor_role_identity.py:568-599` pins `assert len(refused) >= 15`, not `== 18` —
a silent drop to 15 stays green. And its `alias_of_a_forbidden_molecule` allowlist is a **paper-blind**
set of three bare names, while the gold is **paper-relative**: `tetrahydrofolate (thf)` refused on a
paper whose gold *permits* THF would be silently classed as non-collateral.

C-073's own `test_paper_relativity_the_same_name_is_withheld_here_and_kept_there` establishes exactly
the principle this allowlist breaks. Tighten with the follow-up card for F-119.

### A census miscount in the card write-up — recorded, not load-bearing

C-081 reports "91 committed `final_mapped.json`, 1,790 rows, 898 shipping". At the tip commit it is
**89 / 1,781 / 887** — the extra two files exist only in the main working tree under `.pytest_*`
directories, so the census was taken there rather than in the branch worktree. **The load-bearing
figures reproduce exactly**: 50 cofactor declarations, 18 refusals, 0 collateral.

### A scope note the product owner should ratify rather than inherit

The rule's real class is *"cofactors the extractor never wired into a reaction"*, which is broader
than *"hallucinations"*. Cofactor **binding** — an `interaction` — is the canonical way papers state
cofactor relationships, and this card rules that an interaction endpoint confers **no** participant
role. On this corpus every such row happens to be gold-forbidden, so collateral is 0. But *"an
interaction endpoint does not license a database identity"* is a policy-adjacent judgement that
deserves explicit ratification rather than silent inheritance from a card.

---

## C-084 — STOPPED at the measurement gate, and the stop is the result

**Worktree `C:/t/c084`, branch `card/C-084-assay-context`, cut from C-081's tip `b71fe0b`. No
commit. No production code. Working tree clean.** `g11_evidence.py check --task C-084` → 6
artifacts, 0 non-compliant; every job `FINAL SURVIVING COUNT : 0` / `cleanup : success`.

The card was chartered to catch `NADH`/`NAD+` with an assay-context discriminator, on the reading
that they entered as coupled-assay reporter species. **The charter's premise was wrong, and the
implementer proved it rather than building on it.** That is the outcome I asked for.

### Three formulations measured, all fail the zero-collateral bar

| formulation | rows evaluated | catches | **collateral** |
|---|---|---|---|
| A. assay markers **with** reaction-verb veto (windows 150/250/400/600) | 208 | **0** | 0 — **vacuous** |
| B. readout markers alone (windows 120/200/300/450) | 208 | 4 | **4** — `pyruvate` ×4 |
| C. participant not named in its reaction's own evidence span | 385 | 22 | **60** |
| D. C ∧ reaction enzyme is `inferred` | 385 | 5 | **20** |

* **A is vacuous** because the sentence that *proves* NADH is a reporter — *"LDH-catalyzed conversion
  of pyruvate to lactate is then monitored by loss of OD 340 following oxidation of NADH to NAD+"* —
  contains reaction verbs. The readout **is** a reaction, so the veto fires on every span.
* **B cannot be tuned out.** `pyruvate` is a genuine EntB product, quoted verbatim
  (*"isochorismate is converted to 2,3-diDHB and pyruvate by EntB isochorismatase activity"*), and
  sits in the **same sentence** as the readout prose — its two spans are at offsets 21914 and 21984,
  **70 characters apart inside one clause**. Identical results at every window width from 120 to 450.
* **B also misses the target.** `NAD+` normalizes to `nad`, which substring-matches **"Ca*nad*a"** and
  **"bioshop ca*nad*a"** in the affiliations and reagents text, and those non-readout spans rescue it.
  Production `contains` is substring-first by design (*"the looser test is the safe direction here"*),
  so the very looseness that stops the rule stripping `ATP` makes it blind to `NAD+`.
* **C and D strip the pathway's own chemistry**: `ATP`, `pyruvate`, `enterobactin`,
  `2,3-dihydroxybenzoic acid` (**the paper's titular compound**), `L-serine`, `AMP`, `PPi`, `heme`,
  `iron`. Narrowing on `inferred` still strips `enterobactin`, the pathway's end product.

**Every non-vacuous variant hit the declared red line — `ATP` stripped by C, `pyruvate` by B — so
nothing was implemented.** Priority 1 stays at **6**.

### A boundary correction for the register

`actor_named_in_its_own_cited_span` is **not** F-110. It is a *bench* check
(`bench/semantic.py:112`, `CHECK_ACTOR_EVIDENCE`, mirrored at `release_status.py:119`). **F-110** is
the separate production function `map_ids._enforce_shipped_identity_names` at `map_ids.py:5413`.
There is **no** production predicate comparing participants against their reaction's evidence span —
formulation C would be **new capability, not a repair**.

---

## F-118 — an inferred reaction carries inferred cosubstrates on an enzyme-name-only evidence span

- **Severity** HIGH · **Class: candidate `product_contract_violation`** — *under adjudication*
- **Registered 2026-08-25 by the Lead Orchestrator**, measured by the C-084 lane from committed
  T-106 artifacts. **Supersedes the mechanism half of F-096's `NADH`/`NAD+` rows; not their verdict.**

`runs_verify/2026-08-24_1428/papers/PMC12096016/research`:

```
reaction 5  "EntA dehydrogenase reaction"
  inputs : ["2,3-dihydro-2,3-dihydroxybenzoate", "NAD+"]
  outputs: ["2,3-dihydroxybenzoic acid", "NADH"]
  enzymes: [{entity: "EntA", role: "catalyst", provenance: "inferred"}]
  evidence: "EntA (2,3-dihydro-2,3-dihydroxybenzoate dehydrogenase; EC 1.3.1.28)"
```

**The evidence span is the enzyme's name and EC number. It names neither `NAD+` nor `NADH`.** The
pipeline appears to have inferred the standard redox pair that every EC 1.3.1.28 dehydrogenase uses
and attached it to a reaction whose enzyme provenance is itself `inferred`.

So the mechanism is **"inferred standard chemistry becoming paper-stated evidence"** — one of the
classes F-096's charter enumerated — and **not** the assay-reagent leakage the gold's `reason`
describes (*"Coupled-assay reporter species from the LDH readout"*). The gold may be **right that
these must not carry real accessions and wrong about why**; those are separable and both may need
recording.

**The seam is upstream of mapping.** It is reaction admission — *whether an inferred participant may
be added to a reaction whose cited span does not name it* — not identity admission. No card may be
chartered against `mapping/` for this.

**Open question with wider consequences than the identifiers:** acceptance **priority 2 (unsupported
retained reactions) scored 0** on T-106, so the scorer did not flag this reaction. Either its
supported-signature logic is subset-based and structurally cannot see it (the acceptance report
annotates *"signature set is a SUBSET, so unattributed rows are not counted as fabrications"*), or
the reaction genuinely is supported. **If the former, priority 2's PASS cannot detect fabrication,
which is worse than a FAIL.** Adjudication dispatched (`REV-084`).

**Do not charter a code change until that adjudication returns.** A rule here that strips `ATP` from
the EntE adenylation reaction, or `pyruvate` from the EntB reaction, is disqualifying — both are
genuine verbatim-quoted participants in this same paper.

---

## F-121 — acceptance priority 2's PASS is an instrument reading nothing on 8 of 10 papers

- **Severity CRITICAL** · **Class `product_contract_violation`** (with a scoping caveat below)
- **Registered 2026-08-25.** Measured by the REV-084 bio-auditor, **independently verified by the
  Lead Orchestrator against source and gold before registration.**
- **This corrects a headline claim of the C-076…C-080 correction wave. See the correction below.**

### The defect

Priority 2 is declared **absolute** — `bench/acceptance.py:439-441`: *"Priorities 1-3 are absolute:
any non-zero count fails them however good the rest looks."*

`bench/semantic.py`, in `_check_supported_reactions`:

```python
:705    if not complete:
:706        findings = [f for f in findings if f.get("kind") != "unsupported_retained_reaction"]
:708    ok = not missing_signatures and not unverifiable and (not complete or not unsupported_rows)
:722    false_positives if complete else 0,
```

`complete` is `case.supported_reactions_complete`, declared `bool = False` at `bench/goldset.py:384`
and read at `:714` as `bool(raw.get("supported_reactions_complete", False))`.

**Measured directly against the pinned gold:**

```
cases                             : 10
with supported_reactions_complete : 0        <-- ABSENT FROM EVERY CASE
with max_retained_reactions       : [('PMC13231680', 0), ('PMC12180156', 2)]
```

So on every paper `complete is False`, and therefore:

* every `unsupported_retained_reaction` finding is **deleted at `:706`** before it can be counted;
* `:722` contributes a hard **`0`**;
* `:708`'s `(not complete or not unsupported_rows)` is **unconditionally `True`**.

The only surviving route to a non-zero `ERR_UNSUPPORTED_REACTIONS` is the `max_retained_reactions`
ceiling at `semantic.py:1473-1476`, and that field is set on exactly **two** cases —
**both of which are the negative controls.**

**Priority 2 has only ever measured negative-control ceilings.**

### What it missed on a real leg

Offline re-score of `PMC12096016/research` at T-106, deterministic, no LLM:

```
signature_set_complete    False
retained_reactions        6
true_positives            2          (EntC, EntB)
unattributed_reactions    4          <-- EntA x2, EntE, and the interactions
attribution_rate          0.3333
unsupported_reactions     0          <-- priority 2's entire contribution
```

**Four of six retained reactions unattributed. Priority 2 counted zero.**

### CORRECTION to the sprint record

`LEDGER.md` (T-106 section) and `HANDOFF-POST-T106.md` §3 both record **"Priorities 2 and 3 moved
FAIL → PASS"** as a measured gain of the correction wave, and `TEST_MATRIX.md:482` records T-105's
priority 2 as `FAIL` on 3 unsupported reactions.

**For priority 2 that movement is not evidence of improvement.** `FAIL 3 → PASS 0` is the reading of
an instrument that, on 8 of the 10 papers, cannot return anything but zero. Whether the pipeline
actually stopped retaining unsupported reactions between T-105 and T-106 is **unmeasured**.

The T-105 `FAIL 3` and the T-106 `PASS 0` are both preserved as the official recorded results — they
are what the scorer of the day said. **Neither may be quoted again as evidence that unsupported
reactions were eliminated.**

### The suppression itself is CORRECT and must not be overturned

`semantic.py:700-704` explains it: the signature set is a hand-read **subset**, and a cross-paper RAG
addition cannot match a seed-paper signature by construction. Counting unattributed rows as
hallucinations *"would have reported 227 fabricated reactions in a run that produced far fewer."*

That reasoning is sound. **The defect is not the suppression. It is that the suppressed result is
then reported as a PASS on an absolute priority.**

### Why it is a contract violation

`PRODUCT_CONTRACT.md:309-317` requires the system to distinguish, *"without collapsing any into
another"*, semantic evaluation **passed** / **failed** / **not performed**, and states that
**`not_evaluated` is never `false`**. Priority 2 collapses *not-evaluated* into *PASS 0*.

The machinery already exists and is already used for the adjacent case: `semantic.py:586` emits
`inapplicable_reason="the gold case carries no supported_reactions to audit against"` when a case
declares none. The subset case simply does not use it.

**Scoping caveat, and it is a product-owner call.** §11's grammatical subject is *"The pipeline"*,
not the benchmark scorer. Under a narrow reading this becomes a **`gold_data_defect`** instead, whose
remedy is that no gold case ever sets `supported_reactions_complete` and the fix belongs in
`pinned_v1.json`. **Under either reading the finding is real; only the remedy location moves.**

### Owned

**C-085** (`card/C-085-priority2-honesty`, cut from `2972c34`) is chartered for the **code half
only**: report `not_evaluated` with an explicit reason instead of `PASS 0`, keep the subset
suppression, keep attribution rate and recall, keep the negative-control ceiling path working. It is
forbidden from editing `pinned_v1.json`.

### OPEN — priority 3 has not been checked for the same disease

Priority 3 (zero referential-integrity violations) also moved `FAIL 2 → PASS 0` in the same wave and
is also declared absolute. **Nobody has verified that its PASS is capable of returning non-zero.**
Until someone does, treat its `PASS 0` as unconfirmed. This is a cheap static check and belongs with
C-085's review.

### Consequence for release candidates

Every acceptance run to date inherits this blind spot, and any correction wave measured against
priority 2 will appear to hold when it may not. **A gate that cannot produce a non-zero count is
inert, not absolute.** This is why C-085 is chartered ahead of any further release candidate.

---

## C-082 — MERGED `0242810`. F-115 closed.

| card | merge | closes | SMOKE | Chunk D | review |
|---|---|---|---|---|---|
| **C-082** | `0242810` | **F-115** | **473** (50.62 s, post-merge, pinned) | **187 = 160+4+23**, `SETS_EQUAL=True`, `qb` 23 | APPROVE, actual diff, 54 wrapper reports, 0 non-compliant |

**The guard was always right; the disposition was wrong.** `_reject_ambiguous_species_renames` →
`_screen_ambiguous_species_renames`: the two `AMBIGUOUS_RENAME_TARGET` branches (`:1153`
occupied-target, `:1111` two-sources) now return declination records instead of raising.
`AMBIGUOUS_REFERENCE` still raises. Classification rides the existing **D-029** channel:
`report["ok"] = False`, `report["review_required"]["species"] =
"species_rename_declined:AMBIGUOUS_RENAME_TARGET"`.

**Authority: D-035 clause 8** — *"may convert it into a structured review-required or refusal result
**only if** the graph remains intact and **no invalid PWML is emitted**."* Both conditions verified.

It fixed the **module** rather than the unprotected call site at `streamlit_app.py:4299` — a
forbidden file — and so repairs all three call sites at once (`:4299`, `:4850`, `writer.py:2692`).

### What the reviewer proved rather than read

* **Detection is unweakened, by AST set-difference.** It extracted every `If` test, comprehension
  and `For` iteration from the base and tip functions and differenced them: **zero base expressions
  removed, relaxed or reordered.** All four additions belong to the disposition. `by_target`, the
  `len({_norm(source) for source in sources}) > 1` test, the `target == _norm(old): continue` guard
  and the `occupied` comprehension are verbatim.
* **Untouched helpers proved by function-body hashing**, not by trusting the hunk list —
  `_reject_ambiguous_renames`, `resolve_compounds_prefreeze`, `_norm`, `_canonical`, `_alias_index`
  and `_canonicalize_species_rows` all hash identically at base and tip.
* **The G9 proof is tied to the measured defect.** Its own base run produced a failure string
  compared programmatically against `runs_verify/2026-08-24_1428/manifest.jsonl` row
  `PMC12444477`/`research`: **209 bytes, `sha256 3af32ffe628ff57c9adf3aa6d331469b` on both sides,
  byte-identical.** Not a reconstruction.
* **The two species rows stay distinct** — `['Escherichia coli K-12', 'Escherichia coli']`,
  taxonomies `83333` and `562`, at the stage, in the IR, and on the canonical payload that ships
  through the real app.
* **The restore is byte-level complete.** Fed a row carrying a pre-existing `raw_name` and `aliases`
  the ladder would have overwritten, the delta after declination is a single field
  (`species_canonicalization`); `name`/`raw_name`/`aliases` move back together because the restore
  is a wholesale `deepcopy` of the pristine row, so a partial restore is structurally impossible.
  Rename log trimmed: `name_canonicalization == []`, `rename_map == {}`, `renamed == 0`.
* **The marker is outside the graph projection** — `species_canonicalization` appears nowhere in
  `canonical_hash.py`, so the biological graph hash never sees it.
* **SMOKE 473 three times** — base, tip, and the C-081+C-082 merged tree. **Chunk D authoritative
  split form**, `T2PW_OFFLINE_CURATOR=1`: `union=187 monolithic=187 missing=0 extra=0
  SETS_EQUAL=True`, `executed=187/187 omissions=0 additions=0 failed=none`.
* **Trial merge onto `0879e62` was clean** — zero conflicts, no overlap with C-081's `mapping/`
  territory, 86 tests green across C-073 + C-081 + C-082 together. No rebase needed.

### Two judgement calls that were right

**The relocated seam test.** `tests/test_streamlit_quarantine_boundary.py` **is** the `qb` component,
`ENFORCED` at 23 in `chunk_d_gate.py`; appending would have moved `qb` 23→24 and `TOTAL` 187→188 — a
merge-rule-4 baseline move with no authorization and nothing to do with species canonicalisation. The
implementer relocated to `tests/test_c082_post_pipeline_seam.py` instead. **No coverage was lost**:
the new file imports the real harness (`real_streamlit` is `autouse=True` at its definition, so
importing it arms it), and the proof it is armed rather than asserting against a `MagicMock` is that
it **fails at base with the real crash string**, which a stubbed `AppTest` could not produce. The
boundary file's blob is identical at both SHAs — nothing survived the relocation.

**The disclosed measurement failure was handled correctly.** `g11/C-082/06-affected-tip.json`:
`exit_reason: timeout`, `returned_code: 124`, `exit_code: None` (**no test result produced**),
`graceful_attempted: True` then `forced: True`, `descendants_terminated: [62500, 64056]` **by PID**,
`final_surviving_count: 0`, `cleanup_success: True`, 4 pre-existing reported and never killed. The
report was **retained** with the sequence contiguous at `05 → 06 → 07`, so the failure is visible
rather than erased, and the work was re-run through the sanctioned split forms.

---

## F-122 — one of C-082's base failures is symbol absence, not behaviour

- **Severity** LOW · **Class `evidence_precision`** · **Registered 2026-08-25 by the C-082 reviewer.**

`tests/test_prefreeze_species_resolution.py:373`
`test_new_acceptance_the_existing_row_guard_does_not_over_fire` fails at base with
`ImportError: cannot import name '_screen_ambiguous_species_renames'` — it imports the renamed
private helper by its new name.

It is an **over-fire control, not a G9 proof**, and the five failures in the new file are all genuine
`PrefreezeResolutionError`. But the card's headline *"base 8 failed"* should be recorded as
**7 behavioural + 1 rename artifact**. **G9 is amply discharged without it** — symbol absence is not
proof, and none of the load-bearing arms rest on it.

---

## F-123 — a declination does not demote the release status, and closing that needs a ruling

- **Severity** MEDIUM · **Class `policy_disagreement`** · **Registered 2026-08-25 by the C-082
  reviewer.** Same family as **F-107**.

`report["ok"] = False` does **not** structurally prevent a PWML or `release_ready`. Both consuming
seams say so in terms:

> `writer.py:2731` — *"`prefreeze_report["ok"] is False` deliberately does NOT abort. D-029 (LOCKED)
> … Acting on it is the downstream seam's job."*
> `streamlit_app.py:4930` — *"D-029, as split by **D-040 §8**: this seam PERSISTS and SURFACES the
> verdict. It does not act on it — no branch here changes whether a PWML is produced."*

`classify_release_status` takes no `prefreeze_review_required` parameter, so a declination is
**release-status-neutral**.

**The observable consequence is real and must be recorded honestly.** At base, a strict leg hitting
this shape **crashed**, so it could never be `release_ready`. At the tip it proceeds and *can* reach
`release_ready` while carrying two organism rows the ladder wanted to merge. That is permitted by
D-035 clause 8 (graph intact, no invalid PWML) and **required** by merge rule 7 — but it means clause
8's *"must not become a successful export"* is enforced only by the **other** gates, never by this
channel.

**The card did not stop short.** Closing this requires `release_status.py` or `streamlit_app.py`,
both outside C-082's boundary and one of them forbidden, and would reverse or extend **D-040 §8**,
which is LOCKED. Doing it inside C-082 would have been an improvised product decision.
**Needs a product ruling, with F-107.**

---

## F-124 — the claimed `DuplicateNamedRowError` backstop does not exist for species

- **Severity** LOW · **Class `product_contract_violation`** (latent; **measured non-reachable**)
- **Registered 2026-08-25 by the C-082 reviewer.**

`_screen_ambiguous_species_renames`'s docstring claims the second-order collision *"is not silently
merged — `ir.DuplicateNamedRowError` refuses two species rows sharing an exporter name key."*

**It does not, for species.** Species go through the **component** call site `ir.py:1317`, which does
**not** pass `refuse_duplicates=True` and takes the warning branch at `ir.py:688` — commented *"the
branch the species converger depends on (F-046)"* — which **drops the second row first-wins with only
a warning**. `refuse_post_freeze_merges` does not cover it either: two rows already sharing a `_norm`
before `_apply_create_defaults` yield one origin key, so `manufactured` is `False`. Only the
**entity** call site `ir.py:1420` refuses.

**Measured non-reachable**, three independent ways: the shipped `data/pathwhiz_id_db.json` contains
**0 species entries**, so ladder rung 2 cannot rename a species offline; `_deterministic_species_name`
is **idempotent on all 11 realistic strain forms tested**, so every rename target is a fully-stripped
binomial while every declined row keeps a strain-qualified name — the sets cannot intersect; and a
direct adversarial construction of three colliding strains produced **all three declined,
`rename_map == {}`, no duplicate names**.

**Safe today; the reasoning is wrong.** Registered so the docstring is corrected rather than
inherited by a reader who relies on a guard that would not fire.

---

## C-085 — MERGED `07db68f`. F-121 closed. **The honest T-106 numbers are worse than the recorded ones.**

| card | merge | closes | SMOKE | review |
|---|---|---|---|---|
| **C-085** | `07db68f` | **F-121** | **473** (48.69 s, post-merge, pinned) | APPROVE, actual diff, 13 wrapper reports, 0 non-compliant |

### The re-scored T-106 — offline, nothing re-run, reproduced independently by the reviewer

```
=== recorded (base) ===              === honest (tip) ===
1. [FAIL] false real identifiers     1. [FAIL]  identical
2. [PASS] unsupported reactions      2. [NOT EVAL] unsupported reactions
3. [PASS] referential violations     3. [PASS]  identical
4. [PASS] coverage 1/8 = 12%         4. [FAIL]  coverage 0/8 = 0%
5. [FAIL] strict PWML 0/4            5. [FAIL]  identical
```

Priority 2 is `NOT EVALUATED` on **11 of 20 scored legs covering 6 papers** — PMC12096016,
PMC12421875, PMC12444477, PMC12452463, PMC12657337, PMC12782028. **It is not re-reported as FAIL:**
nothing unsupported was ever counted there, and swapping a false PASS for a false FAIL would be the
same lie in the other direction. All nine scientific-error totals are byte-identical between the two
scorings, and both exit 1.

**Two of the five priorities recorded as PASS were artifacts of one blind spot.**

### The priority-4 flip is the pre-existing requirement finally biting — verified structurally

`SemanticReport.confirmed` and the entire `CheckResult` class are **byte-identical at base and tip**.
On `PMC12421875/strict`: `check.ok` did **not** move (True → True); only `check.applicable` moved
(True → False). `confirmed = self.ok and all(c.applicable …)` was already written that way — the
supported-reactions check simply never declared itself unevaluated, so the requirement had nothing to
bite on.

Corroborating: at base, `PMC12180156` and `PMC13231680` **already** carried
`inapplicable=['retained_reactions_match_supported_signatures']` by the two other routes. C-085 added
a **third route to a mechanism that already existed and already worked.**

### The design decision was scrutinised and holds

`unsupported_verdict_evaluated = complete or not unsupported_rows`.

The reviewer went looking for the hole and reported it is not there. `matched_pointers` is populated
**only inside the `quote_ok` branch** — a signature whose quote is not located in the stored paper
text hits `continue` and can never match a row. So `not unsupported_rows` means literally *"every
retained process row matched at least one gold signature whose quote I verified is in the paper"*.
That is a **positive per-row measurement**, not an inference from exhaustiveness, and it is **strictly
narrower** than `complete` because it demands 100% row attribution.

Decisively, **the matcher's known brittleness runs the safe way.** Under-matching creates
`unsupported_rows` → `uve=False` → `NOT EVALUATED`. It cannot manufacture a false PASS. The five
T-106 legs that still PASS priority 2 hold genuinely measured zeros; marking them unevaluated would
have been the opposite error.

### The `render.py` deviation — disclosed, and ruled FORCED rather than scope creep

`bench/render.py` was not in the card's May-change table. The implementer flagged it rather than
burying it. The reviewer read `_mark` and demonstrated it:

```
_mark(True) -> 'PASS'    _mark(False) -> 'FAIL'    _mark(bool(None)) -> 'FAIL'
```

`_mark` is **binary by signature**, and the base renderer coerces `None` to `False`. **Without those
six lines C-085 would have printed priority 2 as `[FAIL]` on T-106** — a fabricated failure, exactly
the lie the card exists to prevent. Six lines, `_priorities` only, `_mark` itself untouched so no
other caller changes, and the card's own G9 proof cannot pass without it. **Approved as forced.**

### The negative-control ceiling still works — demonstrated

Forcing `PMC12180156/strict` to 5 retained reactions against its ceiling of 2 gives
`unsupported=3, uve=True`. The one path that ever worked is intact.

**F-121's premise confirmed in full:** `supported_reactions_complete` is `False` on **all 10** gold
cases; `max_retained_reactions` is set on exactly **two** — PMC12180156 (2) and PMC13231680 (0).

### The official T-106 result is preserved

`evidence/t106_acceptance_report.txt` blob `29a12994…` is **identical at base and tip**. The re-score
is a new file, `evidence/c085_t106_rescore.txt`. `pinned_v1.json` blob `4b5c0355…` is likewise
identical — **the gold decision was escalated, not absorbed.**

---

## PRIORITY 3 — ADJUDICATED: **CAPABLE, not inert.** The last unexamined absolute priority.

I asked whether priority 3 shared priority 2's disease. **It does not.**

`_orphaned_references(payload)` takes **only `payload`** and reads **no gold field whatsoever** — no
completeness gate, no finding deletion, no ceiling dependency. `ERR_ORPHANED_REFERENCES:
len(orphans)` is unconditional. Demonstrated non-zero:

```
PMC12096016/strict as committed : 0 orphans
  + one undeclared input        : 1 orphan   -> orphaned_references = 1, evaluated = True
  + all compound rows removed   : 13 orphans
```

It is architecturally the **opposite** of priority 2, and **its `PASS 0` on T-106 is a real
measurement.** Recorded so nobody re-opens the question.

---

## F-125 — the referential-integrity gate reads roughly half the participant surface

- **Severity** MEDIUM · **Class `product_contract_violation`** · **Zero exposure on T-106; live in the corpus.**
- **Registered 2026-08-25 by the C-085 reviewer**, `semantic.py:1413-1445` `_orphaned_references`,
  `:216-220` `_enzyme_names`.

Priority 3 reads `inputs`, `outputs` and `_enzyme_names` (`enzymes` / `modifiers` / `catalysts`). It
**never reads `cargo`, `transporters` or `elements_with_states`** — that is, **every participant slot
a `TransportModel` or `ReactionCoupledTransportModel` actually has.**

Measured over the 89 committed `final_mapped.json` (65 transport rows):

```
orphans priority 3 COUNTS        : 3
orphans a WIDER reader would see : 6
invisible to priority 3          : 3
   MISSED: PMC12096016/strict  (runs/2026-08-02_2130)          transports.transporters 'EntE'  x2
   MISSED: PMC12180156/research (runs_verify/2026-08-24_1402)  transports.transporters '/entities/proteins/0'
```

That last one is a **leaked JSON pointer sitting in a name slot** — an unambiguous
referential-integrity violation in a committed artifact that the absolute gate for referential
integrity cannot see.

**On T-106 itself: 0 narrow and 0 wide**, so T-106's priority-3 `PASS 0` is **not** falsified. The
gap is live in the corpus, not in that run. **Unowned.** Note the shape is the same family as
**F-119** — a participant reader narrower than the schema it reads.

---

## F-126 — the new priority-2 PASS route inherits matcher precision, which has never been measured

- **Severity** LOW · **Class `measurement_assumption`** · **Registered 2026-08-25 by the C-085
  reviewer.** Not a defect in C-085; a residual that should be named rather than inherited.

The `not unsupported_rows` route rests entirely on `_signature_matches`. A matcher **false positive**
would turn a fabricated reaction into a "measured zero". The observed failure direction is
**under**-matching, which is safe — but matcher **precision** has never been measured, and the five
T-106 legs that now PASS priority 2 rest on that unmeasured assumption.

The relevant recorded observation: PMC12096016/**strict** scored 0/7 attribution while its
**research** leg scored 2/6 on a near-identical payload with identical 5/5 enzyme recall. That swing
is evidence of brittleness in the safe direction, and it is the only precision datum anyone has.

---

## RETRACTION applied to `TEST_MATRIX.md:482`

The row called `PMC12421875` *"the first semantic confirmation of the sprint"*. That claim is now
measurably wrong and sat in a live document. It is **retracted in place**, naming C-085/F-121, the
leg's real numbers (11 retained, 3 attributed, 8 matching nothing) and the re-scored
`priority 4 = 0/8`. Line count unchanged at **578** — the edit is inline, so no pinned baseline moved.

---

## Validation cohort — `runs_verify/2026-08-25_1216`, 4 legs, 1.96 h, **$0.177**

Run at integration `7d0bc22` (C-081 + C-082 + C-085 all merged), through the bounded wrapper with the
heavy lock: `acquired=True released=True`, `FINAL SURVIVING COUNT 0`, `cleanup success`, 11
descendants terminated by PID, 4 pre-existing reported and never killed.
G11 `evidence/g11/T-107/07-cohort-c081-c082.json`. Credit delta measured against the session
baseline: usage `161.092487097 → 161.26978855`, **$0.177**, `limit_remaining` $69.26.

| leg | T-106 | cohort | why |
|---|---|---|---|
| PMC12856317 / strict | PASS | **FAIL** `contract` @ `pwml_export` | `protein_complex_missing_components`: *"Protein complex 'LONP1' has no protein components."* |
| PMC12856317 / research | PASS | **FAIL** `unknown` @ `stage1` | *"structurally empty payload (no entities and no processes) on all 2 attempts … stop reason: `identical_empty_response`"* |
| PMC12444477 / strict | PASS | **FAIL** `contract` @ `post_pipeline` | 9 × `protein_X_is_missing_a_uniprot_or_drugbank_identifier` + `registry_validation_failed` |
| PMC12444477 / research | **ERROR / crash** (F-115) | **PASS WITH WARNINGS** | completed with a citation report |

### The three properties the cohort was run to settle

**1. Does C-081 refuse on the LIVE path, not just in replay? — VALIDATED.**

`PMC12856317/strict`, `final_mapped.json` `/entities/compounds/4`:

```
name        : 'pyridoxal 5-phosphate'      class : 'cofactor'
mapped_ids  : {}
rejected_mapped_ids : {hmdb HMDB0001491, kegg C00018, chebi CHEBI:18405, pubchem 1051,
                       cas 54-47-7, biocyc PYRIDOXAL_PHOSPHATE, chemspider 1022,
                       drugbank DB00114, pathbank_compound_id 1148}
```

Nine accessions refused, including the `drugbank:DB00114` that was T-106's false identifier. **The
row is preserved** — still present, still named, still `class: cofactor`. Scored: PMC12856317 carried
**2** false real identifiers at T-106 and carries **0** here.

**2. Does C-082's declination work in the batch driver? — INCONCLUSIVE, and I will not claim otherwise.**

`AMBIGUOUS_RENAME_TARGET`: **0 occurrences** in `batch.log`. `declined_rename` /
`species_rename_declined`: **0 occurrences** anywhere in the run.

**The triggering condition did not recur on this draw.** `PMC12444477/research` completing is
therefore **not** evidence that C-082 works — it would have completed at base too, because no
ambiguity arose. C-082's evidence remains its behavioural proof: the AppTest seam driving the real
Streamlit app, failing at base with a string **byte-identical to the T-106 manifest row** (209 bytes,
matching sha256) and passing at tip.

**I did not re-run the leg to chase the condition.** Re-running until a draw exhibits the shape is
the same move §9 forbids for F-096, and the sprint's own standing note applies: identical legs give
materially different Stage-1 draws at temperature 0.

**3. Does C-081 collateral a leg whose identifiers must survive? — VALIDATED, no collateral.**

`cofactor_role_used_by_no_reaction` fired **0 times** on PMC12444477. Its strict leg's protein rows:
**8 of 8 carry `mapped_ids`, 0 have rejected ids.**

### The three PASS → FAIL legs are NOT caused by the merged cards

Each was traced to its own cause before being classified:

* **PMC12856317/strict** — `pwml_ir_validation.json` names one error:
  `protein_complex_missing_components` on `LONP1`, plus the same warning on `CLPXP protease`. That is
  **F-116's class** — the complex-wrapper generator producing a component-less complex — not C-081.
  C-081 behaved correctly on this very leg (property 1 above).
* **PMC12856317/research** — Stage 1 returned an empty payload twice with
  `stop reason: identical_empty_response`. **Pure draw variance, upstream of everything merged.**
* **PMC12444477/strict** — 9 proteins failed the UniProt/DrugBank identifier gate. C-081's rule fired
  **0 times** on this paper and C-082's fired 0 times anywhere. T-106's draw of this leg had 10
  protein rows all resolving; this draw extracted a different set. **Draw variance in extraction and
  resolution.**

**No leg failed for a reason attributable to C-081, C-082 or C-085.**

### C-085 confirmed working on a live scoring

The cohort's acceptance report renders:

```
2. [NOT EVAL] zero unsupported retained reactions
   observed: NOT EVALUATED -- 0 counted, but the unsupported-reaction verdict was never
   reached on 2 of 3 scored leg(s), covering 1 paper(s). This zero is the absence of a
   measurement, not the absence of unsupported reactions.
```

The honesty change is live in production scoring, not only in the T-106 re-score.

### A new F-096 member surfaced by draw variance — record it

Priority 1 on the cohort is **1**, on PMC12444477/strict: **`(p)ppGpp`**, shipping
`hmdb HMDB0060480 / pubchem 38166 / pathbank_compound_id 41212`. Gold kind **`heading_or_prose`**,
reason *"Parenthesised shorthand denoting two compounds (ppGpp and pppGpp); the literal token
resolves to neither."*

PMC12444477 carried **0** false real identifiers at T-106. This one appeared on a fresh draw.

**Consequence, and it matters for scoping the next F-096 card: the F-096 class has more members than
the eight T-106 happened to name, and different draws surface different ones.** `(p)ppGpp` is the
same `heading_or_prose` mechanism as `LIPA`/`LBR` and is out of C-081's reach for the same reason —
it is not a cofactor-declaring row that no reaction uses. Counting the class by any single run's
membership will under-scope it.

### What the cohort settles for T-107

It confirms C-081 works in production and costs no collateral, and it confirms C-085's reporting
change is live. It leaves C-082's production behaviour unproven by draw. **None of that changes the
T-107 verdict**, which rests on priority 1 being unreachable at 0 and priority 5 being 0/4 by proof.

---

## F-127 — priority 1's six survivors are two different defects, and BOTH are unreachable because the discriminating fact is never recorded

- **Severity CRITICAL** (it is the absolute priority-1 blocker) · **Class `product_contract_violation`**
- **Registered 2026-08-25 by the Lead Orchestrator.** Measured offline from committed T-106
  artifacts; **nothing re-run, no live leg, no LLM-backed command.**
- **This is the answer to the product owner's Priority A question**, taken down the branch the
  instruction names: *"If no safe general production rule exists, record the exact missing
  biological information and propose the smallest new representation or evidence field needed."*
- **Supersedes nothing. Reopens nothing.** C-084's rejected formulations are not revisited; this
  explains *why* they had to fail.

### The six survivors are not one class

| # | entity | leg | gold class | mechanism |
|---|---|---|---|---|
| 1–2 | `NAD+`, `NADH` | PMC12096016/research | reporter species | **inferred standard chemistry** (F-118) |
| 3–4 | `LIPA`, `LBR` | PMC12782028/research | `heading_or_prose` | **entity admitted without a supporting span** |
| 5–6 | `SREBF1`, `SREBF2` | PMC12782028/research | `regulator_as_metabolite` | same as 3–4 |

They are in **different payload structures**, reached by **different seams**, and no single predicate
can address both. That alone explains C-084's four failures: every formulation it tried was aimed at
class 1 and was scored against a count containing class 2.

### Class 1 — the metabolite slots have no provenance carrier at all

Corpus census over all **10** T-106 legs that reached mapping
(`evidence/g11/T-107/15-f096-corpus-carrier-census.json`):

```
SLOT                   SHAPES        PROVENANCE-BEARING KEYS
inputs                 str=39        NONE
outputs                str=34        NONE
enzymes                dict=36       provenance=36   (extracted 29 / inferred 7)
modifiers              dict=34       provenance=34   (extracted 26 / inferred 8)
cargo                  str=6         NONE
transporters           dict=4        NONE
elements_with_states   dict=12       NONE
```

**Every metabolite participant in the corpus is a bare string.** The provenance carrier this sprint
already built exists on the two ACTOR slots and on no metabolite slot. The offending reaction:

```
PMC12096016/research  reactions[5]
  inputs : ["2,3-dihydro-2,3-dihydroxybenzoate", "NAD+"]      <- bare strings
  outputs: ["2,3-dihydroxybenzoic acid", "NADH"]              <- bare strings
  enzymes: [{entity: "EntA", ..., provenance: "inferred",
             evidence: "EntA (2,3-dihydro-2,3-dihydroxybenzoate dehydrogenase; EC 1.3.1.28)"}]
  provenance_lineage: null
```

The pipeline **knows** it inferred EntA and writes that down. It supplied `NAD+`/`NADH` from
EC 1.3.1.28's standard redox chemistry in the same act — and has **nowhere to write that down**.

**Why every lexical reconstruction must fail, measured rather than argued**
(`evidence/g11/T-107/14-f096-provenance-probe3.json`) — on the *same leg*:

```
rxn[5].inputs[1]  'NAD+'                      named_in_own_span=False   <- fabricated
rxn[4].outputs[0] 'enterobactin'              named_in_own_span=False   <- LEGITIMATE, end product
rxn[2].outputs[0] '2,3-dihydroxybenzoic acid' named_in_own_span=False   <- LEGITIMATE, titular compound
rxn[3].inputs[0]  '2,3-dihydroxybenzoic acid' named_in_own_span=False   <- LEGITIMATE
```

The fabricated row and the pathway's own end product are **indistinguishable by span membership**.
That is precisely the red line C-084 hit, now explained: span membership is a *proxy* for provenance,
and the proxy is degenerate because the real signal was discarded upstream.

### Class 2 — the entity rows carry no evidence, and provenance is a constant

`PMC12782028/research`, all **20** protein rows
(`evidence/g11/T-107/18-f096-entity-fields.json`, `19-f096-stage1-evidence.json`):

```
LSS CYP51A1 MSMO1 DHCR24 SQLE FDFT1 FDPS MVD HMGCR HMGCS1
MVK IDI1 ACAT2 EBP HSD17B7 NSDHL          -> gold PERMITS   (16)
LIPA LBR SREBF1 SREBF2                    -> gold FORBIDS   (4)

every one of the 20:  provenance = "extracted"   evidence = None
```

**The four forbidden rows are byte-identical in every discriminating field to the sixteen required
ones.** There is no field in the emitted record that separates `SREBF1` from `HMGCR`.

**And a graph-role rule cannot separate them either.** The orphan census
(`evidence/g11/T-107/17-f096-orphan-census.json`) over all 10 legs: 195 entities, 93 with real
accessions, **102 used by no process, 50 of those carrying identifiers.** Those 50 include
`EntA…EntF`, `LpxA…LpxK`, `WaaA`, `FabZ`, `ALAS2`, `ferrochelatase`, `HMGCR`, `CYP51A1`, `LSS` —
essentially every enzyme of every pathway in the corpus. **Extending C-081's role-consistency rule
from `class:"cofactor"` to proteins would strip the identifier from almost every legitimate pathway
enzyme we produce.** It is not a near miss; it is a catastrophic false-positive rate, and it is
recorded here so nobody proposes it a third time.

### Traced to the origin: this was never captured, not captured-and-dropped

`stage1_payload.json` — the first artifact out of extraction — carries protein rows with exactly
`{name, class, confidence, provenance, provenance_lineage}`. **No `evidence` key exists at any
stage**, and `provenance` is the constant `"extracted"` on all 20 rows at Stage 1, at
`merged_payload.json` and at `final_mapped.json`.

**Consequence for G9, and it is load-bearing for whoever gets this card:** closing either half is
**new capability, not a correction of pre-existing observable behaviour.** It therefore carries an
**explicitly labelled new acceptance test** and **must not** carry a fabricated base failure.
Mislabelling it a regression fix is a reject under G9.

### The smallest representation that would close each half

**Class 1 — one field on one existing model.** `payload_models.py:327` `ProcessParticipantModel`
has `name, entity, compound, protein, protein_complex, element, element_collection, nucleic_acid,
biological_state, stoichiometry, coefficient, evidence` — and **no `provenance`**. It is the *only*
participant-bearing model without one: `ActorModel:323`, `ElementWithStateModel:351`,
`ReactionModel:364` and `ReactionCoupledTransportModel:377` all carry it.

`ParticipantLike = str | ProcessParticipantModel` (`:342`) — **the schema already accepts the
structured form in `inputs`/`outputs`**, so no exporter change is forced and nothing needs
inventing. The change is: add `provenance` to `ProcessParticipantModel`, have extraction emit the
structured form for participants it supplied rather than read, and let identity admission refuse
real accessions to an `inferred` participant. `ATP`, `pyruvate`, `enterobactin` and
`2,3-dihydroxybenzoic acid` are untouched because they would carry `extracted`.

**Class 2 — populate the evidence span the entity row already has room for.** The requirement is a
per-entity supporting span, so `heading_or_prose` admission is visible as what it is. This one is
**not** free: it changes what Stage 1 must return, so it needs a prompt change and a live leg to
validate, and it interacts with `paper_explicit` under D-059 (*an unmarked Stage-1 `paper_explicit`
claim is RECORDED, not VERIFIED*).

### What this means for T-107 — state it plainly

**Priority 1 cannot reach 0 in this correction wave.** Both halves need a carrier that does not
exist, one of them needs a Stage-1 contract change, and neither is a lexical filter that can be
tuned. Any predicate written against the current record either misses the six or strips the
pathway's own chemistry — both directions now measured, not predicted.

**No card is chartered against this today.** The product owner's instruction is explicit: *"Do not
force a card merely to reduce the benchmark count."* This is registered as the measured missing
representation, for a ruling.

---

# POST-T-106 RULINGS AND CORRECTION WAVE — session opened at `91b5c50`, 2026-08-25

Lead Orchestrator session. **Starting state verified in full before anything was touched:**
tip = `origin/sprint/pwml-recovery` = `git ls-remote` = **`91b5c50`**; no merge in progress; **0
staged**; heavy lock `C:/t/heavylock` **absent**; **zero** sprint-owned Python processes (only the
two `ms-python.isort` `lsp_server.py` IDE servers, PIDs 26052 / 31504, never touched); product-owner
`streamlit_app.py` edit intact at **35 ins / 2 del, uncommitted**,
`sha256:47e4fafa789d359d8526642cd8e70bf968196a46cd8b02d069c6d76a3c5bb632`; **no cache commit since
the sprint base** (`af5c3d2` and `868a254` are the product owner's own pre-sprint `research-mode`
commits of 2026-08-20, not sprint work); `g11_evidence.py check` → **3793 artifacts, 0
non-compliant**; working tree carrying the same four topics files and nine modified files as at
handoff.

## The seven rulings, recorded

| ruling | where recorded | note |
|---|---|---|
| 1 — `extracted_not_serialized` | **D-065** | bundle's "no production work" claim **corrected** |
| 2 — keep `pythonpath = src` | **D-066** + `TEST_MATRIX` rule 10 | refusal superseded, not forgotten |
| 3 — identity | **D-064** | F-113 closed; reopens nothing |
| 4 — `supported_reactions_complete` | **D-067** | Option C; five preconditions |
| 5 — F-123 demotion | **D-068** | Option A; **Option C refused**; extends D-040 § 8, assigns `BL-004` |
| 6 — interaction identity | **D-069** | C-081's blanket rule **not ratified**; narrowed |
| 7 — F-119 / F-125 | assignment, not a decision | design brief in flight (`REV-086`) |

Recorded in `37b87ef`. `TEST_MATRIX.md` 578 → 597 lines (D-061 permits growth); § 0 also gains
F-114's second infrastructure mode.

## The gold reconciliation — full audit trail

`src/t2pw/bench/gold/pinned_v1.json`, **PMC12421875 and PMC12657337 only**, exactly two fields each.

```
pre  sha256 f4ede3e4e7ce60fde928ae1b72dde6c65c542654688791179d1c47a7532c9166
post sha256 4ef1f51d20aa6bdb3f608f40477a9776ad03bbf6c5e3dbf8563661f09f6cd573
git  4 insertions / 4 deletions, CRLF line terminator preserved
```

Verified **mechanically, not by eye** — the patch script refuses to write unless every
pre-condition holds and re-parses the result afterwards:

* the other **8** cases byte-identical, key sets unchanged;
* `forbidden_organisms` unchanged on both (**8** and **10** entries) — the traps stay exercisable;
* `relevance_note` unchanged on both — the ORGANISM TRAP designation is preserved;
* the pre-existing `export_rationale` text byte-identical, the D-062 sentence appended;
* **no topics file touched.**

`bench_acceptance.py --verify-plan runs_verify/2026-08-24_1428` → **`OK`, 10 cases, all ten
`[pinned_override]`** (`evidence/g11/T-107/10-verify-plan-d065.json`, exit 0, 0 survivors).
SMOKE **473 passed** in 50.78 s post-edit under the heavy lock
(`11-smoke-post-d065-gold.json`). Merge rule 10 discharged.

**Priority 5's denominator is now 2, and the honest reading is 0/2** — not a pass. `PMC12782028`
is correctly blocked on coverage and `PMC12096016` twice over.

## Cards dispatched

| card | finding | branch | worktree | base | lane |
|---|---|---|---|---|---|
| **C-086** | **F-116** — a component match promotes a single enzyme to a superset complex | `card/C-086-f116` | `C:/t/c086` | `91b5c50` | writer A |
| **C-087** | **F-123** — a prefreeze declination does not demote the release status (D-068) | `card/C-087-f123` | `C:/t/c087` | `91b5c50` | writer B |
| **REV-086** | D-069 conformance measurement + F-119/F-125 design brief | — | primary, read-only | `91b5c50` | measurement |

**Ownership collision recorded at dispatch:** C-087 has taken `src/t2pw/batch/driver.py` as its
caller seam. **C-083 (F-092 defect 3) also owns `driver.py`** (`_finalize_timeout`, the `RunOutcome`
dataclass and `to_dict`). The two are almost certainly disjoint by function, but they are not
disjoint by file, so **C-083 serialises behind C-087's merge.** It is not dispatched into a
concurrent lane.

## Live-run ledger

**No live paper leg and no LLM-backed command has been run this session.** Every measurement is
current-source inspection or deterministic replay of committed T-104 / T-105 / T-106 artifacts.
Incremental external-model spend: **$0.00**. Implementation and measurement lanes are all explicitly
forbidden from live legs; the orchestrator remains the sole owner of live execution.

## F-117 — MEASURED, and the narrow route the product owner asked for exists

Priority F asked: *"Measure and charter only if a narrow exact-identity route exists. Do not create
broad fuzzy matching."* Measured offline over the whole T-106 corpus
(`evidence/g11/T-107/20-f117-blast-radius.json`).

`_check_actor_evidence` (`bench/semantic_production.py:395`) takes only `processes`. It reads the
actor sub-entry's `entity`/`protein`/`name` and compares it against that row's own cited span. The
component list — which already carries the exact gene symbol — lives on the **entity** table, which
the function is never passed:

```
entities.protein_complexes["Lanosterol 14-alpha demethylase"]
  pathbank_protein_complex_id 442
  components [{name: "CYP51A1", gene_name: "CYP51A1", uniprot: "Q16850", ...}]   <- 1 component
```

**Nine actor-evidence findings exist corpus-wide. Classified:**

| | class | count | effect of a one-component exact-identity route |
|---|---|---|---|
| **A** | 1-component wrapper, component name **verbatim in the span** | **2** | flips to pass — both are the CYP51A1 false positive, `hit=['CYP51A1']` |
| **B** | **multi-component** wrapper | **3** | **keeps firing** — all three are F-116's `enterobactin synthase` (4 components) |
| **C** | not a wrapper / component not in span | **4** | keeps firing |

**The safety property is the point: the narrow route does NOT excuse F-116.** The EntE superset
keeps failing the gate, because 3623 has four components and the rule is one-component-only. Exactly
2 of 9 findings move, both true false positives, and the fix is **exact string identity against a
symbol the record already holds** — not fuzzy matching.

**It changes no disposition**, as F-117 predicted: `PMC12782028/strict` is blocked on
`requested_core_coverage_below_minimum:0.222<0.500` and would remain so. This is an
acceptance-instrument honesty fix, and it raises no rate. **Chartered as C-090, queued behind the
running lanes.**

## D-067 readiness — the two candidate papers, measured. **Flag NOT set.**

Ruling 4 names `PMC13231680` and `PMC12180156` as the starting pair and imposes five
preconditions. Measured offline (`evidence/g11/T-107/21-d067-seedonly-probe.json`); the credit
instruction forbids spending model credits on the review half, so nothing was dispatched.

### Current gold state — both candidates

```
PMC13231680  lipid A biosynthesis   max_retained_reactions 0   supported_reactions 0   complete ABSENT
PMC12180156  heme biosynthesis      max_retained_reactions 2   supported_reactions 0   complete ABSENT
```

**Both signature sets are already empty.** So the question is not *"write the signatures"* — it is
*"is empty the exhaustive truth?"*

### Precondition 4 — seed-only — **SATISFIED on this run, measured**

`rag_admission_report.json` shows `accepted=[]` on **all four** legs of both papers. No multi-paper
RAG synthesis contributed a row, so `goldset.py:384`'s incompatibility warning is not engaged for
these two papers on T-106. **This does not generalise to a future run** and must be re-checked
whenever the flag is relied on.

### What the flag would actually do — `semantic.py`'s own arithmetic

`unsupported_verdict_evaluated = bool(complete or not unsupported_rows)`. With `complete=True` and
an empty signature set, **every retained row becomes a reported fabrication.** Measured retention:

| leg | retained reactions | effect of setting the flag |
|---|---|---|
| `PMC13231680/strict` | none (no `final_mapped.json`) | priority 2 → **genuinely measured PASS** |
| `PMC13231680/research` | none | priority 2 → **genuinely measured PASS** |
| `PMC12180156/strict` | **1** | priority 2 → **FAIL, 1 fabrication** |
| `PMC12180156/research` | none | measured PASS |

### The two papers are NOT the same risk, and the ruling's pairing hides that

**`PMC13231680` is the low-risk one and the gold already asserts the answer twice.** Its
`relevance_note` records the full-text counts — *'lipid A' 1, 'LpxC' 9, and **UDP 0, GlcNAc 0, Kdo
0, acyl 0, deacetyl 0***. Every lipid-A intermediate term occurs **zero** times. And
`max_retained_reactions: 0` is the product owner's existing ruling that *"the only correct output is
an empty one."* An empty exhaustive set is the same claim those two fields already make; setting the
flag would only let the scorer act on it. Setting it converts a `NOT EVALUATED` into an **honestly
earned PASS** and cannot manufacture a fabrication, because nothing is retained.

**`PMC12180156` is the higher-stakes one and it exposes a tension inside the gold.**
`max_retained_reactions: 2` tolerates up to two retained reactions, while the `relevance_note` says
*"Zero heme-biosynthesis reactions have both sides named."* Those cannot both be the exhaustive
truth. The strict leg retained **1**, under the ceiling and therefore currently invisible — but
under an empty exhaustive set that row is a **fabrication and priority 2 fails**, absolutely.

**That is not a reason to avoid the flag. It is the measurement the flag exists to make.** It is a
reason not to set it without the biological review, because the honest answer changes an absolute
priority.

### Precondition status

| # | precondition | `PMC13231680` | `PMC12180156` |
|---|---|---|---|
| 1 | complete scoped source read | **recorded in gold** (term counts) | asserted in `relevance_note`, not itemised |
| 2 | every supported signature defined | **empty set, twice asserted** | empty set claimed; **conflicts with the ceiling of 2** |
| 3 | **independent biological review** | **NOT DONE** | **NOT DONE** |
| 4 | seed-only compatible | **YES, measured** | **YES, measured** |
| 5 | who established exhaustiveness, and how | pending 3 | pending 3 |

**Precondition 3 is unmet for both, and the orchestrator cannot supply it** — it does not approve
its own work, and the credit instruction forbids spending model credits on establishing
reaction-gold exhaustiveness where manual/source-based review is required.

**Therefore the field stays ABSENT on both and priority 2 continues to report `NOT EVALUATED`**,
exactly as D-067 directs when exhaustiveness cannot be proven. **This blocks no release candidate.**

**Recommended when a reviewer is available:** take `PMC13231680` alone first. It is the one where
the gold already contains the evidence, where the flag cannot manufacture a fabrication, and where
the outcome is a strictly honest gain. `PMC12180156` needs its ceiling-versus-note conflict
adjudicated before anyone sets a boolean on it.

---

## F-128 — production violates D-069 on 12 of 18 refusals, and complying RAISES priority 1

- **Severity HIGH** · **Class: `product_contract_violation` in production, layered over a
  `gold_data_defect` in the acceptance mechanism**
- **Registered 2026-08-25.** Measured by the **REV-086** measurement lane over 92 committed
  artifacts; **the load-bearing claim independently re-verified by the Lead Orchestrator against
  source and payload before registration** (`evidence/g11/T-107/22-verify-d069-violation.json`).
- **D-069 § "Required follow-up" directed this measurement.** It has returned a violation.
- **C-081 is NOT reopened.** Its implementation was correct against the rule it was given; D-069
  changed the rule afterwards.

### The seam

| what | where |
|---|---|
| the refusal | `mapping/map_ids.py:8133` `_admit_identities`, **PASS C at `:8309-8352`** |
| the call | `map_ids.py:8330-8339` → `_withhold_identity(..., rule=RULE_COFACTOR_ROLE_UNUSED)` |
| the predicate | `mapping/identity_admission.py:725` `cofactor_participation` (`unsupported` at `:756`) |
| the slot map | `identity_admission.py:662-668` `PARTICIPANT_FIELDS` |
| **the D-069 hinge** | `identity_admission.py:653-661` — the comment that `interactions` is *deliberately absent* because *"an interaction endpoint is exactly the evidence that does NOT make it one"* |

That comment is precisely the blanket rule **D-069 refused to ratify.**

### The measurement

Corpus reproduced: 92 artifacts, 921 rows shipping real accessions, 48 declaring `class:"cofactor"`,
**18 refusals** (PASS C), plus 2 PASS B kind-conflict refusals, 0 PASS A.

**12 of the 18 violate D-069. 6 are correctly refused. 0 undetermined.**
By paper: `PMC12856317` ×10, `PMC12312563` ×1, `PMC12180156` ×1 (the last is weaker — its
THF↔SHMT2 edge is assembled across two sentences and is flagged as such, not asserted).

All 18 sit in `entities.compounds`, so **the entity kind is valid in every one**. The verdict
therefore turns entirely on whether the paper explicitly supports the interaction.

**The six correct refusals are correct for a reason D-069 endorses:** they are not interaction
endpoints at all — those payloads' interactions name different entities (`DHNA inhibits MenD`,
`2,3-DHB inhibits EntB`, `heme inhibits ALAS2`). An unused row with no interaction to lean on gets
nothing from D-069. Three of them (`NADH`, `NAD+`, `thiamine pyrophosphate` on `PMC12096016`) are
additionally the assay-reporter class `PRODUCT_CONTRACT` § 2 forbids by name.

### Verified independently, not inherited

`runs_verify/2026-08-24_1428/papers/PMC12856317/strict`:

```
entities.compounds  "Pyridoxal 5'-phosphate"  class=cofactor  ids={}        <- stripped
processes.interactions[3]  entity_1 "Pyridoxal 5'-phosphate"  entity_2 "ALAS2"
    evidence: "a PLP-dependent homodimer enzyme that mediates the condensation
               of glycine and succinyl-CoA"
01_source_text.txt: "...aminolevulinic acid synthase (ALAS) (9,10), a PLP-dependent
                     homodimer enzyme that mediates the condensation..."   (PLP x22)
```

Source-supported interaction, valid entity kind, identity refused. **Under D-069 that row should
have retained identity.**

### Why C-081 measured 0 collateral and this measures 12 — the yardstick, not the biology

C-081's zero was scored against the pinned gold's `forbidden_identifiers` list
(`tests/test_c081_cofactor_role_identity.py:551-599`), which is matched **by name and is
bucket-blind** (`goldset.py:428-445`, consumed at `semantic.py:1029-1046`). It condemns PLP whichever
bucket PLP sits in.

But the gold's own `kind` for these entries is **`cofactor_as_protein`**, which `goldset.py:314-316`
defines as *"a small molecule filed under `entities.proteins`"* — **and not one of the 18 is in
`entities.proteins`. All 18 are compounds.** The gold's stated failure mode does not obtain for a
single one of them, and the gold's reason text for PLP — *"The ALAS2 cofactor. Never a substrate,
never a product, never a protein."* — is a statement about **role and kind**, which is exactly what
D-069 protects. It says nothing about identity.

**So both numbers are true.** C-081's 0 is true against the gold as *matched*; D-069's 12 is true
against the gold as *written*. A correction card that uses the gold's forbidden-name list as its
acceptance oracle will measure 0 again and conclude nothing changed.

### The consequence that needs a ruling before anything merges

**Restoring identity to those 12 rows makes them count as `false_real_identifier`
(`semantic.py:1035-1046`) — acceptance priority 1, which is ABSOLUTE. Priority 1 would rise from
6.**

So D-069 compliance and the gold's `forbidden_identifiers` mechanism now point in opposite
directions, and the sprint cannot satisfy both. **This is not the orchestrator's to settle.** Under
the standing classification rule the production half is a `product_contract_violation` and the gold
half is a `gold_data_defect` in the *matching mechanism* rather than in the reason text — but which
one yields is a product decision of exactly the shape Item 1 was, and Item 1's own ruling records
that silently editing gold to move a rate is what this sprint exists to prevent.

**Therefore: the correction is chartered as C-091 per D-069's follow-up clause, and C-091 does NOT
merge until the gold-mechanism question is ruled.** Its charter carries the priority-1 delta as a
declared, required deliverable rather than a merge-gate surprise.

### Sequencing

C-091 and **C-089** (F-119/F-125, Ruling 7) both edit `identity_admission.PARTICIPANT_FIELDS`.
**They must not run concurrently.** C-089 owns the shared participant-schema constants; C-091 takes
a read-only dependency on them.

---

# T-107 READINESS TABLE — 2026-08-25, post-ruling wave

**Verdict: T-107 remains NO-GO.** Priority 1 is **absolute** and is **guaranteed to fail for a known
unresolved reason**, which is the exact condition § 8 forbids running under. Two further absolute or
scored priorities are also guaranteed to fail. Running now would buy another predictably failed score
and teach us nothing new.

| Priority | Reachable? | Remaining blocker | Expected T-107 result |
|---|---|---|---|
| **1** — zero false real identifiers | **NO — and it got worse** | **F-127**: the discriminating fact is never recorded. Metabolite slots have no provenance carrier (`inputs`/`outputs`/`cargo` are bare strings corpus-wide) and entity rows carry `evidence=None` with `provenance` a constant, at Stage 1 and every stage after. **F-128**: complying with D-069 would *restore* identity to 12 refused rows, which the gold's bucket-blind `forbidden_identifiers` list counts as false identifiers. **Unresolved product conflict.** | **FAIL — 6 or HIGHER** |
| **2** — no unsupported retained reactions | **Partially, and honestly** | **D-067 precondition 3** (independent biological review) unmet on both candidate papers; the orchestrator cannot supply it and the credit rule forbids buying it. `PMC12180156` additionally carries a gold-internal conflict: a ceiling of 2 against a `relevance_note` saying zero reactions have both sides named. | **NOT EVALUATED** on 11/20 legs (6 papers); genuinely-measured PASS on the other legs |
| **3** — zero referential-integrity violations | **YES — passes today** | None for T-107. F-125's 3 invisible orphans are real but sit in **other** artifacts; T-106 exposure is **0**. C-089 closes the blind spot without moving this run. | **PASS** |
| **4** — requested-core coverage | **NO** | Nothing in this wave addresses coverage. The honest T-106 figure is **0/8 = 0%** after C-085 removed the blind spot that had reported 1/8. Cap 2's input is Stage 0's `key_compounds`/`key_proteins`, a non-deterministic draw, not a curated core. | **FAIL — 0/8** |
| **5** — strict PWML export | **NO, but the metric is now honest** | Denominator reconciled **4 → 2** by the D-065 gold edit. Both survivors are `correctly_blocked` and measured so: `PMC12782028` on `requested_core_coverage_below_minimum:0.222<0.500`, `PMC12096016` twice over on `requested_core_anchors_unmatched`. C-086 strengthens the biology but **does not unblock the leg** — cap 2 fires regardless. | **FAIL — 0/2** |

## Against § 8's eight gate conditions

| # | condition | state |
|---|---|---|
| 1 | priority 1 has a safe correction **or an explicitly accepted measurement limitation** | **NOT MET.** F-127 states the limitation precisely and F-128 adds a conflict on top. **Acceptance is the product owner's to give; the orchestrator cannot accept a limitation on its own behalf.** |
| 2 | priority 5's denominator reconciled under Ruling 1 | **MET** — gold edited with full audit trail, `--verify-plan` `OK` / ten `[pinned_override]`, SMOKE 473. The *code* half (C-088) is chartered and queued. |
| 3 | the two remaining strict-denominator papers classified honestly | **MET** — both `correctly_blocked`, each with its measured reason. |
| 4 | applicable F-116 / F-123 corrections merged | **NOT MET** — C-086 and C-087 in flight, neither merged. |
| 5 | acceptance instrumentation remains honest | **MET and improving.** C-085 made priority 2 honest; F-117/C-090 removes 2 measured false positives without rescuing F-116's 3. |
| 6 | affected-paper validation passes | **NOT MET** — nothing merged to validate. |
| 7 | all processes closed | **MET at every checkpoint so far** — every job zero survivors, cleanup success, lock free. |
| 8 | integration clean and pushed | **MET** — local = origin = `git ls-remote`, working tree carries only the pre-existing caches/scratch and the product-owner edit. |

**Three of eight unmet. Condition 1 is the one that cannot be cleared by this session's work at all.**

## What would change the verdict

**Not** merging C-086, C-087, C-088, C-089 or C-090. Those are all worth merging on their own
contract-grounded merits, and none of them moves priority 1 to zero. Merging every one of them still
leaves priority 1 failing at 6 or higher.

The verdict changes only when **one of these two happens**:

1. the product owner **explicitly accepts F-127's measurement limitation** — recording that priority 1
   cannot reach 0 until a participant-provenance carrier and an entity evidence span exist, and that
   T-107 may run against a stated non-zero floor; **or**
2. the **F-128 gold-mechanism conflict is ruled** and a participant-provenance representation is
   chartered and merged, so priority 1 has a route to 0 rather than a route upward.

**Until then T-107 stays blocked, and that is the correct outcome rather than a delay.** The last
release candidate was not run for exactly this reason, and nothing measured since has made the
answer different — it has made it better evidenced.

## Offline re-score of T-106 against the reconciled gold — Ruling 1 proven in the scorer

`bench_acceptance.py --run-dir runs_verify/2026-08-24_1428` at `3042256`, through the bounded
wrapper under the heavy lock (`evidence/g11/T-107/23-rescore-post-d065-gold.json`, exit 1 — the
scorer's normal non-zero on a failing priority — lock acquired and released, **0 survivors**).
**Nothing re-run. No live leg. No LLM call.** Stored artifacts only.

```
1. [FAIL]     zero known false real identifiers      observed: 8
              papers: PMC12096016, PMC12782028, PMC12856317
2. [NOT EVAL] zero unsupported retained reactions    NOT EVALUATED on 11 of 20 legs, 6 papers
3. [PASS]     zero referential-integrity violations  observed: 0
4. [FAIL]     meaningful requested-pathway coverage  observed: 0/8 = 0%
5. [FAIL]     strict PWML pass rate                  observed: 0/2 = 0%
```

### Priority 5 — the denominator moved, and only the denominator

```
STRICT PWML SUCCESS
  population : strict_exportable gold cases with an attempted strict leg
  result     : 0/2 = 0%
  not passing: PMC12096016, PMC12782028
  excluded   : PMC12657337 -- expected_export=partial_only: A three-reaction connected chain...
  excluded   : PMC12421875 -- expected_export=partial_only: A fully connected chorismate-to-DHNA...
  (+ six pre-existing partial_only exclusions, unchanged)
```

**This is the ruling working exactly as specified and no further.** The two trap papers leave the
strict denominator by an explicit, recorded decision; their exclusion lines quote their **original**
`export_rationale` text, confirming in the scorer's own output that the pre-existing wording stayed
byte-identical. The denominator is **2**, and it holds precisely the two papers the bundle measured
as `correctly_blocked`.

**The rate did not improve. It went from `0/4` to `0/2`** — the metric became honest, not better.
That was the whole claim, and it is now measured rather than predicted.

### Priority 1 reads 8 here, and that is NOT a regression — read this before quoting it

The stored T-106 artifacts were produced on **2026-08-24**. **C-081 merged on 2026-08-25**
(`b869780`). The artifacts therefore predate the fix, and scoring them reproduces the **pre-C-081**
count of **8**, including the two `Pyridoxal 5'-phosphate` rows on `PMC12856317` that C-081 now
refuses.

**`8` is the historical T-106 figure. `6` is the post-C-081 figure**, established by replay and
recorded against C-081's merge. Both are correct about different trees. **A fresh run is what would
show 6**, and that is the figure the T-107 readiness table uses, because the readiness table is a
prediction about a fresh run.

Quoting the `8` from this re-score as evidence that C-081 regressed would be wrong, and it is
written down here so nobody does.

### Priority 2 confirms C-085 is live

`NOT EVALUATED — 0 counted, but the unsupported-reaction verdict was never reached on 11 of 20
scored legs, covering 6 papers. This zero is the absence of a measurement, not the absence of
unsupported reactions.` C-085's honesty change is working on the real report, and **D-067 leaves it
that way** deliberately.

---

## Workstation crash mid-wave — recovery record

The workstation crashed with both writer lanes running and **both branches uncommitted**. Recovery
state verified before anything was touched:

| check | state |
|---|---|
| tip = origin = `git ls-remote` | **`7621f4c`** — every commit pushed before the crash survived |
| merge in progress | none |
| staged | 0 |
| heavy lock `C:/t/heavylock` | **ABSENT** — the crash did **not** strand it |
| sprint-owned Python | **0** (two `ms-python.isort` IDE servers on fresh post-reboot PIDs, never touched) |
| product-owner edit | intact, **35 ins / 2 del**, `sha256:47e4fafa…` |
| G11 `.staging` reservations | **empty in both card dirs** — no orphaned allocation to explain |

**Nothing was lost.** Both worktrees still held their full working state, including G11 reports and
pin verdicts. The lanes had done the work and simply never reached a commit.

### What I did, and what I did NOT do

I committed each lane's surviving worktree state **verbatim** to its own branch, and ran the one
job C-087's lane had not reached. **No line of either production diff is mine**, and the commit
messages say so. Merge rule 5 is untouched: an independent reviewer (`REV-087`) is reviewing the
actual diffs, and I do not approve work I am going to merge.

### Salvaged

| card | tip | diff | tests | G11 |
|---|---|---|---|---|
| **C-086** | `fb42e62` | `map_ids.py` +280/-11, 4 new private helpers | 574 lines, 9 tests | 10 reports, 7 pin verdicts, G9 pair complete (tip exit 0 / base exit 1), affected sweep exit 0 |
| **C-087** | `6deb55f` | `release_status.py` +241, `driver.py` +40/-2 | 762 lines, 16 tests | 4 reports, 4 pin verdicts, G9 pair complete |

### C-087 deviated from its charter, deliberately and correctly

It was chartered to thread the prefreeze verdict into `classify_release_status`. **It measured that
this is not constructible** — across `src`, no caller of that function holds a prefreeze report and
no holder of one calls it, because pre-freeze canonicalization runs at the **export** seam, after
the boundary already froze the record.

So it added a monotone `release_status.cap_release_for_prefreeze_declination` and applied it at
`driver._frozen_release_record`, whose only transition is `release_ready` → `review_required`. Its
merge-rule-8 argument, which the reviewer is checking rather than me: capping is the **opposite** of
an exporter repairing biology after the freeze — it reads a verdict another stage already reached
and can only **remove** a strict success.

Its choice of `_frozen_release_record` over `_finalize_pwml_export` is argued from the code:
`_add_strict_artifacts` derives the PWML **filename** from it while `_finalize_pwml_export` derives
the **manifest row**, so capping only the row would leave a declined leg shipping a bare
`pathway.pwml` — which `PRODUCT_CONTRACT` § 13 reads as *"ship it, no review needed"*, leaving
D-035 clause 8 enforced only by coincidence on the other channel. That is exactly what D-068
forbids.

**Verified disjoint from C-083:** the `driver.py` hunk touches neither `_finalize_timeout`,
`RunOutcome` nor `to_dict`.

### The affected sweep I ran for C-087, and its classification

```
tip  C:/t/c087      154 passed, 2 failed
base C:/t/c087base   22 passed, 2 failed    <- the SAME two

FAILED test_c074_strict_core_floor.py::test_the_full_corpus_replay_moves_exactly_the_legs_that_are_named
FAILED test_c074_strict_core_floor.py::test_exactly_one_committed_leg_declares_a_core_without_stating_a_pathway
```

Both fail at base with C-087's changes absent, so **neither is C-087's doing**. They are the
already-registered **F-112** class — committed tests red because `runs_verify/**` grew, which
REV-086 independently corroborated this session when the artifact count moved **89 → 92**. Merge
rule 4 holds. Classified rather than assumed: the base arm is its own pinned run with
`--expect-tree C:/t/c087base` and a committed verdict, per D-066.

### Lanes refilled

`C-089` (`C:/t/c089`) and `C-090` (`C:/t/c090`), both cut from `736c1a2`, with base worktrees
`C:/t/c089base` and `C:/t/c090base` **created up front** so neither lane repeats C-086's
base-tree-export detour. Both charters were already written before the crash.

Implementers are now instructed to **commit as they go** rather than at the end. That is the one
process change the crash actually justifies.

---

## C-086 and C-087 — MERGED. REV-087 APPROVED both against the actual diff.

| card | merge | closes | review | SMOKE |
|---|---|---|---|---|
| **C-086** | `0c9705b` | **F-116** | APPROVE, actual diff, 16 bounded reports | **473** together |
| **C-087** | `9e41a11` | **F-123**, discharges **BL-004** | APPROVE, actual diff | (`28-smoke-post-c086-c087.json`, exit 0) |

**The reviewer inherited nothing.** It derived both affected sets independently, reproduced C-086's
7 files exactly and added 7 more (**343 passed**), reproduced C-087's 154/2 exactly and added a
second sweep of 8 more files (**313 passed, 2 failed**), rebuilt both base arms on the git worktrees
rather than trusting the committed verdicts, and compared its pin verdicts byte-for-byte against the
implementers'.

### Two measurements neither implementer produced

The crash cost us both RISK sections, so the reviewer supplied them:

**C-086 blast radius**, over 92 committed `final_mapped.json`: **803** one-component complex actors,
untouched by construction; **89** multi-component actors — the entire population this rule can move
— confined to **3 papers / 18 artifacts**. Realistic DB-matched refusal population ≈ **60 actor
slots** across `PMC12096016`, `PMC12452463`, `PMC12444477`. Nothing else in the corpus can move.

**C-087 monotonicity**, enumerated rather than asserted: **7 status values × 19 prefreeze shapes =
133 pairs**, and exactly **one** moving transition, `release_ready → review_required`.
Non-`release_ready` records moved at all: **none**. Keys ever changed: `status`,
`strict_acceptance_eligible`, `reasons` — **no entity, reaction, identifier, complex, location or
reference is read or written**, which is what settles merge rule 8. The input dict is byte-unchanged
after every call.

### C-087's charter deviation, confirmed correct

It was chartered to thread the verdict into `classify_release_status`. The reviewer independently
reproduced the measurement that **no such seam exists**: the only call that can reach `release_ready`
is `strict_quarantine.py:2477`, at the quarantine boundary, while pre-freeze canonicalization runs
later inside the export seams. And `_frozen_release_record` is the **single** choke point feeding
both consequential channels — `_add_strict_artifacts` derives the PWML **filename**,
`_finalize_pwml_export` derives the **manifest row**. Capping only the row would have left a demoted
leg shipping a bare `pathway.pwml`. D-068's own wording authorises the seam.

### Carried forward, not buried

* **PathBank 3468 `phosphatidylglycerophosphatase`** stands as an actor **14 times** in
  `PMC12444477` and will likely now be refused. Its components read `Phosphatidylglycerophosphatase
  B` and `Pgp phosphatases` — the second looks like a **group row**, not a subunit, and C-086's rule
  cannot tell those apart. **Adjudication dispatched (`R-089`) before any benchmark comparison**, so
  a status move on that paper is not misread either way.
* **`db_unavailable` demotes on the same channel** as the rename declination — D-029 acting as
  ruled. Measured ceiling: **3 legs** across 21 committed manifests / 117 legs, and the T-106 run has
  **zero** `release_ready`, so its recorded statuses cannot move at all.
* **The interactive Streamlit path stays uncapped.** Correct — `streamlit_app.py` is
  product-owner-owned and `PRODUCT_CONTRACT` § 13 scopes the naming rule to the batch artifact set —
  but recorded as a **known ruled gap**, not an oversight.
* `03-base-tree-export.json` on C-086's branch is an abandoned export attempt. **No accepted proof
  rests on it** (verified: `04`, `07`, `09` all pin `C:\t\c086base`). Recorded so a future reader
  does not mistake it for evidence.

### A fourth pre-existing failure, better classified than mine was

I had classed C-087's residual failures as the F-112 stale-corpus-pin class. The reviewer found a
**fourth** and classified it separately and correctly: `test_batch_preflight.py:616` fails with
*"this project ships a .venv; the test assumes it"* — a **worktree environment artefact**, since the
base worktree ships no `.venv`. Not a corpus pin. It fails identically on both trees.

`FULL_STACK_BASELINE` is intact; `test_strict_quarantine_real_artifact_replay.py:416` failed in
neither sweep.

---

## My own regression, found late and recorded in full

**The D-065 gold reconciliation broke `test_c056b_semantic_denominators.py` and I did not catch it.**
SMOKE does not include that file, so **473 green told me nothing about it**. Both implementer lanes
then reported the failure as *pre-existing* — and from their base it was, because they were cut from
`736c1a2`, which already contained my edit. Only an A/B against `91b5c50` exposed it.

```
91b5c50  (before the gold edit)   9 passed
110cffe  (after the gold edit)    1 failed, 8 passed
    assert 'PMC12421875' in []
```

`_STRICT_A` / `_STRICT_B` are `PMC12657337` / `PMC12421875` — exactly the two papers D-065 moved to
`partial_only`. The file's own comment claimed *"the first two are `strict_exportable`"*, which
stopped being true the moment I edited gold.

This is **merge rule 4's second clause**: a pinned baseline moved deliberately, and the delta is now
documented rather than discovered at a gate. Repaired in `0820f5c` with two new constants naming the
papers that **are** still `strict_exportable`, rewiring only the one test that needs the strict
denominator — the other eight exercise the **semantic** denominator, whose population is unchanged,
which is why 8 of 9 passed throughout. An anti-vacuity assertion the original lacked now pins the
strict population to exactly that pair, so a future gold edit that emptied the denominator goes
**red instead of vacuously green** — the failure mode that hid this one.

**That diff is mine and was committed unreviewed.** It is in front of `REV-089` now, with the
question put plainly: whether the fixture rewire preserves the guard's intent, and whether my
anti-vacuity pin is itself a new brittle pin.

**The general lesson, recorded because it will recur:** a control-plane gold edit is a *behavioural*
change to every test that reads the gold, and SMOKE is not a sufficient gate for one. Any future
`expected_export` edit runs the gold-reading test files explicitly, not just SMOKE.

---

## R-089 — PathBank 3468 adjudicated: the refusal is correct, and the "14 slots" figure was wrong

**Dispatched at C-086's merge on REV-087's MED finding. Returned 2026-08-25.** Eight bounded jobs,
all zero survivors, `check --task R-089` → 8 artifacts, 0 non-compliant.

### The verdict

**Refusing the 3468 promotion on `PMC12444477` is biologically correct. No follow-up card, no
special case.** Not a `product_contract_violation`, not a `gold_data_defect`, not a
`policy_disagreement` — the rule is working as chartered.

### The reviewer's premise was wrong, and correcting it makes the refusal MORE clearly right

REV-087 suspected `Pgp phosphatases` was a family/group row. **It is not** — it is one reviewed
accession, **P18200**, *E. coli* K12 `pgpA`, a genuine distinct polypeptide. The plural is a
PathBank naming artifact, and `entities/proteins/17.mapping_meta.resolved_name` reads
`"Phosphatidylglycerophosphatase A"` under `direct_id_match:uniprot_id`.

**But the grouping is real one level up, in the complex rather than the component.** PathBank 3468
pairs **PgpA (P18200)** with **PgpB (P0A924)**, and in *E. coli* PGP → PG + Pi (EC 3.1.3.27) is
served by **three independent, mutually redundant isozymes** — PgpA, PgpB (PAP2 superfamily, broad
specificity) and PgpC (P0AD42). Singles and doubles are viable; only the triple is lethal. The
artifact corroborates the separateness itself: `entities/proteins/16.mapping_meta.candidates` lists
P18200 and P0AD42 as distinct K12 entries while P0A924 sits in the complex row.

So **3468 is an "any-of" isozyme set rendered in PathBank's `protein_complex` table**, and PathWhiz
reads a `protein_complex` as an "all-of" assembly. Emitting it as the catalyst asserts *"PgpA and
PgpB jointly catalyse this step"* — false in both directions: they do not assemble, and either alone
suffices. Reaction 21 is a one-substrate hydrolysis, and the **pre-mapping** payload named exactly
one intended catalyst, `Pgp phosphatases`. `Phosphatidylglycerophosphatase B` is injected purely by
the component match.

**Both readings converge on refuse.** Genuine polypeptide → PgpB is an uncatalysing stranger the
reaction never names. Family row → an all-of complex over a family is meaningless. There is no third
reading in which the promotion is right, so REV-087's structural point — that the rule cannot tell a
group row from an uncatalysing subunit — is **true about the rule's discriminating power and
irrelevant to its output here.** A special case would buy nothing and would bolt a curated
PathBank-ID allowlist onto a rule whose entire value is being identity-driven.

### The chemistry is not in the paper at all

`01_source_text.txt` term counts: `phosphatidylglycerophosphat` **0**, `PgpA`/`PgpB`/`PgpC`
**0/0/0**, `phosphatidylglycerol` **0**, `CDP-DAG` **0**, `cdsA` **0**. Every case-insensitive `pgp`
hit is `(p)ppGpp`, which the gold independently **forbids** as `heading_or_prose`
(`pinned_v1.json:275-278`).

The whole PG/CL branch is **RAG-imported and attributable**: `rag_provenance.source_id
PMC12898747`, *"Essential Role of LapD in the Absence of Cardiolipins"*, confidence 0.86-0.87, on
`PgsA`, `PlsB`, `PlsC`, `cdsA` and compounds 15-40. The evidence spans use `[ 29 ]` bracket markers
where PMC12444477 uses `( 82 )` parentheticals.

### R-089-B — and the "14 slots" figure does not describe the live pipeline

**This is the part that matters for the next comparison.** The 14 is **7 reactions × 2 lists**, and
**13 of the 14 are wrong regardless of what C-086 does**: `phosphatidylglycerophosphatase` stands as
a catalyst on reactions 15-21, and only reaction 21 is chemistry a PGP phosphatase performs.

That is **actor spraying**, not a 3468 problem — every reaction from index 9 on carries 6-15
catalysts with `enzymes` byte-identical to `modifiers`, all sharing one block-level evidence span.
Reaction 10 (`acetyl-CoA → malonyl-CoA`) carries **12** catalysts including `KDO transferase` and
`lipopolysaccharide ABC transporter`.

**But it is confined to `runs/2026-07-28_0919`, a month-old leg predating C-081 and the RAG
admission gate.** The current draws are clean: `runs_verify/2026-08-24_1428` has 3 reactions with
**1 enzyme each**; `runs_verify/2026-08-25_1216` has 7 reactions with **1-2 each**. No spraying, no
PG/CL block, no `Pgp` string, no 3468.

**REV-087's 14 conflated one stale July leg with the live draw.** Counting across all 92 committed
artifacts is the right way to bound a rule's reach and the wrong way to predict a run.

The auditor **declined to open a card** on the spraying and **declined to guess** which admission
stage let the RAG import through, because the deciding artifact — `rag_admission_report.json` — does
not exist for a leg from before that instrumentation. Correct on both counts; a stage named by
inference is not a finding.

### What this means for the next benchmark comparison — read before comparing

1. **Expect `PMC12444477` not to move at all.** Multi-component DB-matched actors on the current
   draw: `2026-08-24_1428` → **0**; `2026-08-25_1216` → **1**, and that one is `FtsH/YciM complex`
   with `pathbank_complex_id: None` — a generated wrapper, so C-086's rule never reaches it.
   **C-086's realistic blast radius on this paper today is zero slots.**
2. **If the PG/CL block reappears** (the RAG draw is non-deterministic) **and 3468 is refused, that
   is an improvement to expect, not a regression to investigate.** Signature: the actor becomes
   `Pgp phosphatases complex` and
   `mapping_meta.reaction_enzyme_complex_superset_promotions_refused` increments with
   `uncovered_components: ["Phosphatidylglycerophosphatase B"]`.
3. **If the paper's STATUS moves, it is not C-086.** That card is Stage-2 mapping and its fallback is
   byte-identical to the pre-existing `novel_enzyme_single_component_complex` shape, so it cannot
   change a gate outcome on the Raetz backbone, which is entirely one-component wrappers 1616-1629.
   The paper already oscillates on draw variance alone — **PASS with warnings** on `2026-08-24_1428`
   against **FAIL** on `2026-08-25_1216`, one day apart, same paper.
4. **No gold change.** The gold correctly expects no PGP phosphatase, so no metric it computes can
   move on this.

### The one caveat the auditor flagged against itself

Its claim that the refusal *will* fire is **static** — read off C-086's merged source against the
committed payload, not demonstrated by a run. It read the token construction but not
`_normalize_name`'s body, and says so. Recorded as static analysis, not as a measured outcome.

---

## C-088 returned: the D-065 population is FOUR legs, not six — and my brief was the thing that was wrong

**Committed `b7bec6d`, not merged. Under review as REV-090.**

I briefed C-088 with the bundle's figure: *"Six legs across three papers end `scope_conflict`."* That
is true of the **scope-conflict** count and **false** as the disposition population. The implementer
measured it and said so rather than following the brief.

| leg | core | gold floor | placed |
|---|---|---|---|
| `PMC12421875` research / strict | 9 / 9 | 7 | **yes** |
| `PMC12657337` research / strict | 3 / 5 | 3 | **yes** |
| `PMC12312563` research / strict | 1 / 1 | 1 | **NO** |

`PMC12312563` clears its own gold floor of 1 and is still excluded, because
`MIN_CONNECTED_CORE_REACTIONS = 2` (C-074 / F-101) says one reaction is not a pathway — and the
gold's own `export_rationale` for that case says it in terms: *"A single reaction cannot form a
connected pathway, and no second reaction anywhere in the text shares a metabolite with it."*

**On that leg `diagnostic_only`'s existing gloss is TRUE.** Placing it would have replaced one
untruth with another, which is the exact failure D-065 exists to stop. I verified the floors and the
rationale independently before accepting the deviation
(`evidence/g11/T-107/29-c088-population-check.json`).

**The consistency check is what makes it convincing:** the four legs are exactly the two papers D-065
named for the gold reconciliation, and exactly the 4 → 2 denominator arithmetic. Three independent
routes to the same pair.

**The two-floor rule (gold floor AND the global minimum) is a narrowing my charter did not specify**,
so it is in front of REV-090 as the thing to scrutinise hardest. If the product owner rules for
gold-floor-only, the implementer reports it is a one-line change.

### What the card did and did not build

`RELEASE_STATES` was **not** extended — there are still exactly three output states, and no
STOP-and-report condition was hit. `release_disposition()` is a **single rule shared by the
classifier and the scorer**, so the two cannot drift into two readings of one ruling — the same shape
C-087 gave `prefreeze_review_reasons`. `to_dict` writes the key **only when set**, which is what
keeps the 7-slot golden digest in `test_batch_driver_seam_golden.py` still.

`classify_release_status` gains one keyword, `required_connected_reactions=None`, and **no production
caller supplies it**, so every existing runtime record is byte-identical.

### The gap I am NOT going to paper over

**Production never populates the runtime field today.** `driver.py::_finalize_scope_conflict` is
outside the card's boundary and has no access to the gold floor, so the disposition is established
**only in the acceptance record**. D-065 says *"the release/acceptance record"*, so I read the letter
as satisfied — but D-065 also says the emitted record must be honest and that a contract state must
not be described as something it is not.

So the implementer's proposed § 4 gloss, which reads *"the record carries an explicit `disposition`
field beside the status"*, **overstates it for a runtime record**, where the field exists and is
always empty. I have put that to REV-090 with a request for corrected text rather than landing the
gloss as offered. **The gloss lands in the merge commit, and it will say where the disposition is
actually established.**

Wiring the runtime seam needs a card that owns `driver.py` and a production-side floor decision.
Registered here as the residual rather than left to be discovered.

Also out of scope and recorded: `bench/render.py` renders through `describe(leg.release_status)` —
the frozen record, which carries no disposition — so the rendered acceptance table does not show it.

### F-112 is now FOUR stale corpus pins

Both C-083 and C-088 hit the same two reds, identical at base and tip, and C-089 hit a third
alongside them. With `test_batch_preflight.py:616` (a worktree `.venv` artefact, a different class
that REV-087 separated correctly), the stale-pin population is now large enough that it is costing
every card a paragraph of classification. **Re-baselining F-112's pins is worth its own small card**
once this wave's merges settle.

---

## CORRECTION to the live-run ledger — this session is NOT $0.00

I reported **"incremental external-model spend: $0.00"** earlier in this session. **That is now
false and I am correcting it rather than leaving it to stand.**

### What happened

C-083's implementer copied `.env` into `C:/t/c083` and `C:/t/c083base` as the F-051 golden control.
`.env` sets `LLM_PROVIDER=openrouter`. It then ran an affected chunk that drives the real Streamlit
app through the post-pipeline path **without** `T2PW_OFFLINE_CURATOR=1`.

`TEST_MATRIX.md:277-284` is explicit about why that guard exists: without it `run_pathway_curator`
issues **one ungated LLM call per post-pipeline app run** at temperature 0.2. It also warns that
because `.env` is untracked, a worktree normally gets `LLM_PROVIDER=local` and the curator becomes a
no-op **by accident** — *"so a green qb cohort obtained in a worktree does NOT certify the same
cohort in the primary."* **Copying `.env` in removed that accidental protection and put the worktree
on the billed path.**

### Measured, not estimated — from the committed log

`docs/pwml_recovery_sprint/evidence/c083_aff_c3.log`, lines 90-93:

```
WARNING t2pw.llm.client:client.py:887 LLM returned an empty completion for curator
  (model deepseek/deepseek-v4-flash, finish_reason=length, tools_sent=True)
  on attempt 1/3; retrying as a transient.        <- invocation A
  on attempt 2/3; retrying as a transient.        <- invocation A
  on attempt 1/3; retrying as a transient.        <- invocation B
  on attempt 2/3; retrying as a transient.        <- invocation B
```

* **2 curator invocations**, each retried — **4 to 6 real API calls** on
  `deepseek/deepseek-v4-flash`.
* Wall clock corroborates: **197.67 s ungated vs 24.76 s with the guard**, same chunk.
* Every completion came back **empty with `finish_reason=length`** — the calls consumed tokens and
  returned nothing usable.

**Cost: well under $0.01.** A handful of flash-model calls that hit the output ceiling. The **$5
incremental ceiling is not remotely at risk**, and no further expansion is warranted or planned.

**The amount is not the point.** The rule was broken, and a rule that only matters when the bill is
large is not a rule.

### The contamination path did NOT fire, and that is checkable

`TEST_MATRIX` names the real hazard above the cost: accepted curator patches are *"written back into
`audited_json` and flow through mapping into `final_mapped_db`"* — the measured root cause of
**BL-003**. **Because every completion was empty, no patch was produced and none was accepted.**

Independently confirmed from the diff: `git diff --name-only 116c8fa..card/C-083-f092-d3` contains
**no** `audited_json`, `final_mapped*`, `runs/` or `runs_verify/` path. `.env` is gitignored
(`.gitignore:3`) so it could not ride in. **No committed artifact is contaminated.**

### A useful by-product for REV-090

The ungated run failed **5**, the gated re-run failed **4**. The one that moved is
`test_c082_post_pipeline_seam::test_an_ambiguous_species_rename_does_not_end_the_post_pipeline_leg`,
which timed out at 120 s **because of the live calls** and passes with the guard.

The other **four — all `test_prefreeze_third_export_seam` — are byte-identical across both runs**, so
they are not curator-induced. That is evidence for, though not proof of, C-083's "pre-existing, DB
reachability" label. REV-090 has been asked to confirm they also fail at base, because *"the database
was unreachable"* is exactly the kind of explanation that is convenient and hard to falsify.

### Process changes, effective now

1. **`T2PW_OFFLINE_CURATOR=1` is exported to every lane in its charter**, not left as a
   `TEST_MATRIX` reference. REV-090 was warned mid-flight and both C-083 worktrees still carry
   `.env`.
2. **Copying `.env` into a worktree is itself a cost decision** and must be declared before it is
   done, not reported afterwards. The F-051 golden control genuinely needs it — so the guard must be
   set in the same breath.
3. The guard goes in the **bounded child environment**, not just the shell, exactly as
   `TEST_MATRIX:284` says.

### Two other process facts the lane surfaced, worth keeping

* **`pinned_pytest.py` swallows `-m pytest`** as a pytest marker expression — `12 deselected`,
  exit 5. Pass test paths directly.
* **The Bash tool caps at 10 minutes**, which kills the wrapper mid-run and strands the heavy lock
  because the `finally` never runs. The lane released it **correctly**: it read `holder.json`,
  confirmed the holder named its own job (`C-083`) and that PID 68020 was dead, and only then
  cleared it. That is the protocol; clearing another holder's lock stays forbidden.

---

## C-083 and C-088 — MERGED. REV-090 APPROVED both; C-088's approval was conditional and the condition was met.

| card | merge | closes | review | SMOKE |
|---|---|---|---|---|
| **C-083** | `bf3fa77` | **F-092 defect 3** | APPROVE, actual diff, 21 bounded reports | **473** together |
| **C-088** | `d3fb884` | **D-065**'s scored half | APPROVE *conditional on the § 4 gloss* | (`30-smoke-post-c083-c088.json`, 0 curator calls) |

**The condition was not waived.** REV-090's corrected `PRODUCT_CONTRACT` § 4 gloss is staged **into
C-088's own merge commit**, so the contract and the emitted record land together — the coupling
D-065's correction to the bundle exists to force.

**The implementer's proposed gloss was rejected.** It read *"the record carries an explicit
`disposition` field beside the status"*, which is true only of the benchmark artifact. The runtime
record declares the field and nothing populates it, so that sentence would have stated as fact the
exact class of untruth D-065 exists to remove.

### The four-legs question, settled by an argument neither the implementer nor I made

I briefed "six legs across three papers" from the bundle. The implementer measured **four** and
excluded `PMC12312563` on `MIN_CONNECTED_CORE_REACTIONS = 2` plus gold's own rationale. I accepted
that as a well-grounded deviation. **REV-090 showed it is not a deviation at all:**

> D-065's ruling section is headed **"For `PMC12421875` and `PMC12657337`"**, and those are the only
> two papers whose `expected_export` it moves. `PMC12312563` appears nowhere in the disposition
> ruling.

So four legs across two papers **is** the chartered population. My "six" was the **scope-conflict**
population — a different set that I carried across without checking. The implementer reached the
right answer by measuring against a locked constant rather than by reading D-065's headings, which
is the stronger route precisely because the two agree.

`_as_measured_int` excluding `bool` is a real catch: `isinstance(True, int)` is `True`, so a flag
passed where a count belongs reads as `1` — exactly the value that would have wrongly placed
`PMC12312563`.

### What REV-090 measured rather than accepted

**C-088's "moves no rate", end to end.** It scored the committed run at base and tip in separate
worktrees and diffed the full report: **13 keys added** (12 leg keys plus the top-level roll-up the
implementer did not count), **zero dropped, zero values changed.** The only residual differences are
`payload_path` worktree prefixes — an artifact of the measurement.

**C-083's golden confinement, proved better than the captures did.** Rather than re-capturing, it
observed that the **six unchanged `GOLDEN` lines still pass at the tip** — so any other leg's
observable moving would have reddened its own untouched pin. One entry moved: `input_timeout` slot 2,
`382cc778…` → `b55b5024…`.

**F-112 confirmed with the actual number:** C-074 pinned a census of **38** committed legs carrying a
release record; the checkout now has **60**.

REV-090 also recorded an infrastructure error of its own rather than letting it read as a finding:
its first SMOKE showed 3 spurious `FileNotFoundError` failures from a scratchpad `--basetemp`
exceeding MAX_PATH. Re-run short → 473.

---

## F-129 — an explicit `db_resolver=None` is silently replaced by the ambient live database

- **Severity MEDIUM** · **Class `product_contract_violation`** (a caller cannot express "no DB")
- **Registered 2026-08-25.** Surfaced by REV-090 while classifying C-083's residual failures;
  **independently re-verified by the Lead Orchestrator before registration**
  (`evidence/g11/T-107/32-f129-db-probe2.json`).
- **Not C-083's doing** — the four failures are identical at base and tip. C-083's *classification*
  of them was wrong, and that is what this records.

### The seam

`src/t2pw/pwml/compound_resolution.py:594-601`:

```python
if db_resolver is None:
    try:
        from t2pw.mapping.map_ids import PathBankDbResolver
        db_resolver = PathBankDbResolver.from_env()      # <- ambient DB substituted
    except Exception as exc:
        db_reason = f"db_resolver_unavailable:{exc}"
        db_resolver = None
```

**Measured in this environment:** `PathBankDbResolver.from_env()` returns a live resolver and
`available()` is **`True`**.

### Why it matters

`None` is the only way a caller can say *"resolve no compounds against a database."* This seam reads
`None` as *"I didn't specify one, go find one"* and conflates the two. A caller that deliberately
disables DB resolution gets the ambient database instead, and nothing in the report says so.

The immediate consequence is in the tests: the four
`test_prefreeze_third_export_seam` failures expect the unreachable arm
(`db_resolver=None` → `{"available": False, "reason": "db_not_configured"}`) and **cannot reach it
while a real PathBank DB is up.** So those four are **green or red depending on whether a
developer's PathBank service happens to be running** — which is why they have been drifting through
this wave being labelled "DB reachability".

**C-083's label was inverted.** *"The database was unreachable"* is exactly the kind of explanation
that is convenient and hard to falsify; the truth is that the database was **reachable and was
substituted**. The conclusion (pre-existing, not C-083's) stands; the reason does not.

**No card is chartered.** Fixing it means distinguishing "unspecified" from "explicitly none" — a
sentinel or a separate flag — which touches a production seam and a test contract at once. Recorded
with the measurement so the next reader does not re-derive it, and so nobody "fixes" those four
tests by making them pass against an ambient DB.

---

## Two follow-ups registered now, before they cost a re-baseline

* **BL-005 — thread the connected-core floor to `batch.driver._finalize_scope_conflict`.** C-088's
  runtime disposition is **structurally unreachable**, not merely unpopulated:
  `bench/acceptance.py:777` is the only supplier of `required_connected_reactions`. Until this lands,
  D-065's *"emitted and scored record must be honest"* is satisfied on the scored half only — which
  the § 4 gloss now says out loud. REV-090's stronger suggestion, which I endorse: pin the
  disposition **at write time** in the runtime record and have the scorer *read* it rather than
  re-derive it, so the benchmark cannot assert a disposition about a run whose code no longer exists.
  Note also that `ReleaseStatus.to_dict` **inserts** `disposition` after `status` rather than
  appending, which will surprise an insertion-order digest the moment anything sets it.
* **BL-006 — thread `_Budget` into `driver._finalize_timeout`.** C-083 recorded the budget's absence
  honestly rather than inventing `LEG_TIMEOUT_SECONDS`, but `budget_unrecorded` is now a permanent
  manifest field and **removing it later is itself a baseline move**. Register the honest fix now, so
  a second card does not have to re-baseline the golden to delete this key. It needs `_drive`'s five
  call sites.

---

## C-089 and C-090 — MERGED. REV-089 APPROVED all three items it was given.

| item | merge | closes | verdict |
|---|---|---|---|
| **C-089** | `3f848b4` | **F-119** + **F-125** | APPROVE |
| **C-090** | `ef3a0d4` | **F-117** | APPROVE |
| **the c056b repair** (mine) | `0820f5c`, corrected in `51b3bb4` | — | APPROVE, **with one correction** |

SMOKE **473** after both merges, 0 curator calls, `FINAL SURVIVING COUNT : 0`. Gate 10 discharged
for all six cards merged this wave.

### The correction to my own work — my anti-vacuity assertions were vacuous

I added two assertions to the c056b repair and presented them in the commit message as the D-065
guard. **They were not.** Both trap papers had **no legs** in that fixture, so `acceptance.py:977`
excluded them via `strict_leg is None` whatever their `expected_export` said — they would have
passed before D-065 as well as after.

REV-089 proved it rather than arguing it: it ran **my repaired file at `91b5c50`**, where both
papers are still `strict_exportable`, and got **9 passed**.

**That is the same failure mode I wrote the assertions to prevent, one level up.** An assertion that
reads as coverage and provides none is worse than none, because it stops anyone looking again.

Replaced in `51b3bb4` with a test that discriminates: both trap papers get **attempted, eligible,
deliverable** strict legs byte-identical to the control's, so the only thing that can exclude them is
`acceptance.py:969`'s `expected_export` check — which runs **before** the leg check at `:977`. It
carries its own control so a rule that excluded everything would not satisfy it.

```
91b5c50 (pre-D-065 gold)   1 failed, 9 passed   <- assert _STRICT_A not in population
116c8fa (post-D-065)       10 passed
```

Red before, green after, **on the gold edit alone**. That is the behavioural lock I claimed to have
written the first time.

The **equality** assertion REV-089 examined separately is kept, and its reasoning for keeping it is
better than mine for writing it: a future gold edit that *adds* a third `strict_exportable` case does
**not** break it, because that paper has no leg in the fixture. It breaks only if one of these two
fixture papers leaves `strict_exportable` — the premise the fixture depends on, and precisely the red
worth having.

It also endorsed the sequencing call: folding this into C-088 would have left the integration branch
knowingly red across two more merges, handed a card a repair whose cause sat outside its boundary,
and let the next lanes keep reporting it as pre-existing — which had already happened twice.

### What REV-089 measured that neither card did

**C-089's gate-6 exposure.** `_names` feeds `_connected_core`, so widening it could have *inflated*
connectivity and let payloads clear `min_connected_reactions`. Census over 92 artifacts, base vs tip:

```
CONNECTED-CORE MOVES : 0        priority 1 : 18 -> 18, refusals identical row-for-row
DECLARED/BUCKET MOVES: 0        priority 3 : 3 -> 6, exactly the three F-125 names, none lost
ORPHAN MOVES         : 2        bench_acceptance over T-106 : TOTAL DIFFS: 0
```

**C-090's anti-widening, from its own census** rather than the card's test: 34 legs across 5 run
directories. On T-106, base 9 findings → tip 7, **ADDED 0**. All three `enterobactin synthase`
findings — the F-116 four-component superset — keep firing. A = 2, B = 3, C = 4 exactly as chartered.

**C-090 resisted a real temptation.** D-064 says in terms that *"`EntE` and `enterobactin synthase`
are the same protein identity"*; read literally that would license rescuing the class-B superset.
The card declined on the ground that a four-component wrapper is not its component, and wrote the
reasoning into a docstring rather than acting on it.

---

## F-130 — C-090's blast radius outside T-106: two entities move, not one, and a semantic verdict flips

- **Severity MEDIUM** · **Class: not a defect — a correctly-applied rule on an unenumerated corpus**
- **Registered 2026-08-26** by REV-089, correcting C-090's own under-report.

C-090 flagged **one** class-A-shaped finding it had not re-scored on
`runs_verify/2026-08-25_1216/PMC12444477/strict`. There are **two**, across four pointers:

| entity | complex | sole component | the row's own cited span |
|---|---|---|---|
| `tetraacyldisaccharide 4'-kinase` | 1621 | `LpxK` / P27300 | *"phosphorylated by **LpxK** to produce lipid IV_A"* |
| `phospholipase A1` | 1185 | `PldA` / P0A921 | *"**PldA** activity has been shown to stabilize LpxC…"* |

Both are genuine class-A shapes by the charter's definition — one component, symbol verbatim on
whole-token boundaries in the row's own span — and both identities are biologically correct: LpxK
**is** tetraacyldisaccharide 4′-kinase, and *E. coli* PldA **is** the outer-membrane phospholipase A.

**The consequence C-090 did not measure:** on that leg `semantic_evaluation` flips **`failed` →
`passed`** and `failed_checks` goes `['actor_named_in_its_own_cited_span'] → []`.

**But the leg does not move.** A different, already-live blocker takes over:

```
base   review_required  strict_eligible False  ['semantic_evaluation_failed:actor_named_in_its_own_cited_span']
tip    review_required  strict_eligible False  ['requested_core_anchors_unmatched:UDP-GlcNAc,...,PldA']
```

Same status, same eligibility, different recorded reason. **No gate weakened, no PWML gained.** That
leg is referenced once (`LEDGER.md:3824`, a validation cohort) and pinned by no test.

**Recorded, not chartered.** A semantic verdict flipping on a committed leg belongs on the record
rather than being discovered during a future comparison.

**A neighbouring gap this exposes, and it is NOT C-090's:** the rescued `phospholipase A1` span is
*regulatory* (*"PldA stabilizes LpxC"*), not catalytic. The actor check only asks whether the span
names the actor, so removing the finding is correct **for that check** — but the finding was doing
accidental double duty for `CHECK_SUPPORTED_REACTIONS`, which has no such guard of its own.

---

## F-131 — `ref` / `id` now reach `bench.semantic._names` for the first time

- **Severity LOW** · **Registered 2026-08-26** by REV-089.

C-089's charter scoped the legacy `ref`/`id` tail to `identity_admission`. Because both readers
consume the single `PARTICIPANT_NAME_KEYS`, `bench.semantic._names` now treats `id` as an
entity-name key, which it never did.

**Corpus impact measured 0** — connected core identical on 92/92, orphan delta exactly the three
F-125 names — and the direction is stricter. But it is a widening the charter did not authorise, and
it is recorded rather than inherited by proximity.

---

## `test_c074_strict_core_floor.py` is RED ON THE INTEGRATION BRANCH, right now

REV-089 flagged it and I confirmed it at `ef3a0d4`: **2 failed, 22 passed.**

```
AssertionError: the corpus is not the measured 38 legs: 60
AssertionError: … Left contains one more item: '2026-08-24_1203/PMC13231680/strict'
```

C-074 pinned a census of **38** committed legs carrying a release record; the checkout now has
**60**, because further `runs_verify/` directories were committed afterwards. Pure F-112
stale-corpus-pin staleness — no behavioural regression, and correctly reported as pre-existing by
four separate lanes this wave.

**But it is live and unrepaired, and it is not in SMOKE — the same blind spot that hid my c056b
regression.** Four lanes have now each spent a paragraph re-classifying it. **Chartered as C-092**
so the fifth does not.

---

## F-130 — RECONCILED 2026-08-27. All four claims confirmed; narration only, no code moves

Reconciled offline by the Lead Orchestrator at integration tip `79faf93`. No live paper run, no
LLM-backed command, no historical artifact rewritten. Measurement: G11 `ORCH-130/01-f130-replay.json`,
exit 0, `FINAL SURVIVING COUNT : 0`, `cleanup : success`.

**This entry is appended, not substituted.** The F-130 registration above stands as REV-089 wrote it.

### The four claims, each with what settles it

| # | Claim | Verdict | What settles it |
|---|---|---|---|
| 1 | **Two** entities move, not one, across four pointers | **CONFIRMED** | Committed: `runs_verify/2026-08-25_1216/papers/PMC12444477/strict/final_mapped.json` carries two one-component `protein_complexes` rows — `tetraacyldisaccharide 4'-kinase` (pathbank complex `1621`, component `LpxK` / `P27300`) and `phospholipase A1` (complex `1185`, component `PldA` / `P0A921`). Four firing pointers: reaction 0 `enzymes/0` + `modifiers/0`, reaction 3 `enzymes/1` + `modifiers/1`. A *third* `tetraacyldisaccharide 4'-kinase` reference (reaction 4, span *"…lipid A 4'-kinase"*) never fired — `normalize_name` reduces punctuation to spaces, so both sides share the token `kinase` and the base check already passed. **Six references, four findings.** |
| 2 | `semantic_evaluation` flips `failed` → `passed` | **CONFIRMED** | Base is committed: `quarantine_report.json` `release.semantic_evaluation = "failed"`, `release.semantic_failed_checks = ["actor_named_in_its_own_cited_span"]`. Tip follows from `semantic_production.py :: _sole_component_symbols` + `_component_named_in_span`, which match on whole-token boundaries, and was measured by REV-089 (`evidence/g11/REV-089/25-actorcensus-tip.json`). |
| 3 | The leg's final disposition does **not** change | **CONFIRMED — now measured, not inferred** | Deterministic replay of the committed base record through the *production* classifier, twice, everything but the semantic verdict read off the report. Both runs: `review_required`, `strict_acceptance_eligible = False`. |
| 4 | A **different live blocker** becomes controlling | **CONFIRMED — with the exact reason string** | Same replay. Reason removed: `semantic_evaluation_failed:actor_named_in_its_own_cited_span`. Reason surfaced: `requested_core_anchors_unmatched:UDP-GlcNAc,R-3-hydroxymyristoyl-ACP,palmitoyl-CoA,LapB (YciM),YejM (PbgA),LpxA,ObgE,PldA`. |

### The controlling blocker, named

`REASON_REQUESTED_CORE_ANCHORS_UNMATCHED` — `src/t2pw/pipeline/release_status.py:142`, applied at
`:1088-1092`. **Semantic verdict and leg disposition are separate fields and this leg is the proof.**
The verdict feeds one of five *caps*; a cap can only remove `release_ready` and can never deepen an
existing `review_required`. C-090 removes the semantic reason and thereby *surfaces* a cap that was
always satisfied and merely unrecorded — the C-072 convention records the anchor cap from
`release_ready` only. Nothing about the leg's biology, eligibility or export changed. **Do not read
"semantic verdict passed" as "the leg improved".**

### Under-reporting: narration, not production or scoring

The counting code is correct and per-pointer: `_check_actor_evidence`
(`bench/semantic_production.py:578-599`) appends one finding per actor pointer and reported `4` on
this leg at base, `0` at tip. **Nothing in `src/` counts "entities moved"** — that figure came from an
uncommitted probe, and C-090's committed census
(`tests/test_c090_wrapper_identity_actor_evidence.py:53`, `test_g`) is scoped by `CORPUS` to
`runs_verify/2026-08-24_1428`, which does not contain the 1216 leg. C-090's own gates commit
`cb10134` says *"C-090 moves two pointers and both are on PMC12782028/strict"* — true **inside
T-106**, false as a global statement, and the merge commit `ef3a0d4` already carried the correction
forward.

**Therefore no production or scoring change is justified, and none is made.** F-130's class as
registered — *"not a defect — a correctly-applied rule on an unenumerated corpus"* — is upheld. A
card that altered production to match a narrower historical summary would be changing behaviour to
fit a report, which the sprint forbids.

**Recorded as available, not dispatched:** a `test_h`-shaped regression over the 1216 leg (assert
`_check_without_entities` yields exactly the four pointers, `_check` yields none, and
`classify_release_status` returns `review_required` / `strict_acceptance_eligible False` with the
anchors reason). No merge gate requires it, because no production logic under-reported. **Minor,
labelled observation, unowned:** `_check_actor_evidence`'s summary string calls pointers `"actor(s)"`,
which is what makes an entity-versus-pointer miscount easy to write in the first place.

### Protected — never regenerate

All 15 committed files under `runs_verify/2026-08-25_1216/papers/PMC12444477/strict/`, and in
particular **`quarantine_report.json`, which is the only copy of the base release record**. Re-running
that paper would destroy it. Also protected: `…/PMC12444477/research/*`, `…/PMC12856317/**`, the
run-level `SUMMARY.txt` / `manifest.jsonl` / `batch.log` / `failures_by_code.txt`, all of
`runs_verify/2026-08-24_1428/**` (which `test_g` pins), and `evidence/g11/REV-089/24…27-*.json`.

**F-130 is CLOSED as a reconciliation.** No branch, no card, no code.

---

# AFFECTED-PAPER VALIDATION COHORT — run ledger, written BEFORE launch

**Derived 2026-08-27** by the Lead Orchestrator from the six merged cards' charters, their own test
modules, T-106 artifacts and the prior cohort records. **Two strict legs. One `batch_run.py`
invocation.** Topics file: `topics_wave_cohort.txt` (untracked, like every other topics file).

**THIS IS NOT A BENCHMARK AND MUST NEVER BE SCORED AS ONE.** A two-paper denominator makes
acceptance priorities 1, 4 and 5 read misleadingly. The only artifact to produce is a **leg-level
comparison against T-106**.

## Why any live run at all

C-086, C-089 and C-090 were each measured by replay over the **same pre-C-086 committed payloads**.
C-086's own preserve arm was proved by a *constructed fixture*, not by a live mapping pass. **The
composition of the three on a freshly mapped payload has never been observed once.** That is an
integration risk in the direction merge rule 6 cares about, and it is the one property no stored
artifact can carry.

## The ledger

| Paper | Mode | Cards validated | Exact property | Run directory | Status | Result | Rerun justified? |
|---|---|---|---|---|---|---|---|
| `PMC12096016` | strict | C-086 (F-116 defect case), C-090 (F-117 class B anti-widening), C-089 (F-125 orphan slot) | on a freshly-mapped `final_mapped.json`: (1) no reaction actor resolves to complex **3623**; (2) `reactions[3]` and `reactions[4]` carry **distinct** actors; (3) `EntC→1143`, `EntB→1189`, `EntA→1190` still present; (4) `actor_named_in_its_own_cited_span` still fires on any surviving multi-component wrapper; (5) `transports.transporters` orphans counted by priority 3 | *(new, `--fresh`)* | PENDING | — | only on a measured production defect |
| `PMC12782028` | strict | C-086 (preserve arm / control), C-090 (F-117 class A) | wrapper **442** regenerated with exactly one component `CYP51A1`/`Q16850`; the two `reactions/1` pointers raise **no** actor-evidence finding; leg stays blocked on `requested_core_coverage_below_minimum`, `strict_acceptance_eligible=False` | *(same run dir)* | PENDING | — | only on a measured production defect |

**Why stored evidence is insufficient, per leg.** `PMC12096016/strict`: every committed artifact for
this paper was produced *with* the promotion applied; no stored payload exists in which C-086's
output is the input to C-090's check and C-089's reader. `PMC12782028/strict`: the corpus cannot show
that C-086 leaves wrapper 442 intact when the database is queried live, and C-090's class-A flip was
measured on a payload C-086 never touched.

**What would require a narrow repair.** `PMC12096016/strict`: payload lost at Stage 3 or
`pwml_export` **and** traced to a bare-protein enzyme actor → repair confined to
`_rewrite_reaction_protein_enzymes_to_complexes`. A *new* actor-evidence finding (`ADDED > 0`) →
C-090 widening, revert-scope. `PMC12782028/strict`: 442 disappears or gains components → C-086
over-fires on its own control set, which its charter § 3 forbids.

**Why neither is duplicated.** `PMC12096016/strict` is the only leg carrying the 3623 superset **and**
the F-125 `transports.transporters` slot **and** a class-B population in one mapping pass.
`PMC12782028/strict` is the only class-A leg in the corpus and the only place C-086's preserve arm
and C-090's rescue arm meet live.

## The drop list — every leg considered and excluded

| Excluded | Card | Why stored evidence settles it |
|---|---|---|
| `PMC12452463/strict` | C-086, C-090 B | second sample of the same rule on the same complex 3623; not among the 4 `strict_exportable` papers, so a status move there moves no rate. **First leg to add if `PMC12096016/strict` is inconclusive.** |
| `PMC12444477/strict` | C-086, C-090 / F-130 | R-089 adjudicated the 3468 refusal correct offline; F-130's claims 3 and 4 are now CONFIRMED by deterministic replay through the production classifier. Most expensive strict leg (1864 s). |
| `PMC12856317/strict` | C-086 ALAS2 control | the one-component preserve arm is already covered by 442 on `PMC12782028/strict`. |
| `PMC12444477/research` | **C-087** | `AMBIGUOUS_RENAME_TARGET` **did not recur** on the 2026-08-25 draw — 0 occurrences anywhere in that run. A live leg cannot be relied on to reproduce the ambiguity, and re-running until it does is the move § 9 forbids. C-087's proof is an AppTest seam failing at base on a **byte-identical 209-byte string** matching the T-106 manifest row's sha256, plus an exhaustive **7 statuses × 19 prefreeze shapes = 133 pairs** with exactly one moving transition. Exhaustive enumeration cannot be improved by one sample. |
| all C-083 timeout legs | **C-083** | historically 30–60 min and forbidden by charter § 6. Pinned by `tests/test_c083_inner_timeout_row.py`, which replays the two stored `detail` strings verbatim through the production seam, plus the re-baselined golden. |
| `PMC12421875` ×2, `PMC12657337` ×2 | **C-088** | the disposition is established **only in the acceptance record**, never at runtime, so `bench_acceptance.py` over any committed run dir reproduces it at zero credit cost. The legs stop at Stage 0 on a deterministic gold organism trap; a live draw could only *un*-establish it and would prove nothing about the code. |
| `PMC12312563` ×2 | C-088 | outside the chartered population — `MIN_CONNECTED_CORE_REACTIONS=2` excludes it. |
| `PMC12180156/research` | C-089 (F-125 orphan 3) | draw-dependent leaked pointer on a `context_only` paper whose T-106 research leg died `no_reactions`; pinned as a committed fixture. |
| any leg for F-119 | C-089 | corpus exposure **0** across 92 artifacts; anti-widening measured base-vs-tip row-for-row. Nothing live can exhibit a shape the corpus never produces. |
| all research legs | all six | C-086 is strict-only by construction (`allow_complex_wrapper_creation=not research_mode`); all 9 T-106 actor-evidence findings are on strict legs; C-088/C-089 settled offline; C-087's trigger is unreproducible. |

## Standing rules for this run

1. **`--fresh` into a NEW run directory.** F-130 protects `runs_verify/2026-08-25_1216/**` and
   C-090's pin protects `runs_verify/2026-08-24_1428/**`. `batch_run.py` also silently skips
   finished pairs without `--fresh`.
2. **Run it ONCE.** No leg is repeated because a draw is unfavourable. If C-087's ambiguity does not
   recur, that is reported as **"not observed"** and never chased.
3. A live run executes the **uncommitted** product-owner `streamlit_app.py` edit (35 ins / 2 del,
   `sha256:47e4fafa…`). The result therefore measures *tip + that edit*, not the committed tree, and
   the record must say so.
4. If one leg fails: classify it first as causally related to a merged card, versus stochastic
   Stage-1 variation, versus an unrelated existing blocker. Repair only a **measured production
   defect**, and rerun only the affected leg unless the repair changes a shared upstream seam.
5. Budget: per-leg T-106 durations are 1597.17 s and 690.57 s = **2287.7 s ≈ 38 min**. Draw variance
   was +36% on the one observed case, so **budget 55 min, `--timeout 5400`**. Estimated credit spend
   **≈ $0.06**, scaled from the 1216 cohort's measured $0.177 / 7067 s — order of magnitude only,
   since spend tracks tokens rather than seconds.

## Cohort preflight — staged, verified, and one G11 process incident recorded

**2026-08-27, integration `46df1e7`.** Staged directory: `runs_verify/2026-08-27_1341`, **untracked**
until the run completes.

```
scripts/batch_run.py --topics topics_wave_cohort.txt --modes strict --out runs_verify --fresh --stage-only
```

`--stage-only` executes **zero Streamlit/LLM legs by construction** — it returns before the run loop,
so no manifest row and no leg directory can be created however warm the paper cache is. Both papers
came from cache. Acquisition funnel: requested 2, examined 2, eligible 2, ineligible 0,
`no_full_text` 0, accepted 2. Planned 2 papers × 1 mode = **2 runs, 0 skipped**.

### The plan verification, and why REFUSED is the right answer

```
scripts/bench_acceptance.py --verify-plan runs_verify/2026-08-27_1341
verdict: REFUSED     cases checked: 2     search calls: 0
  PMC12096016 | enterobactin biosynthesis | Escherichia coli   [pinned_override]
  PMC12782028 | cholesterol biosynthesis  | Homo sapiens       [pinned_override]
  MISSING  PMC12444477, PMC13231680, PMC12657337, PMC12421875,
           PMC12312563, PMC12856317, PMC12180156, PMC12452463
```

**`--verify-plan` is the *acceptance-run* preflight and requires all ten pinned gold cases.** This
cohort is a deliberate two-paper subset, so REFUSED is the correct and expected verdict — the tool
independently refusing to treat this run as acceptance is corroboration of the ledger's own standing
rule that it must never be scored as one.

**What the check did validate, which is what I needed:** both triples resolve `[pinned_override]`
against gold `2026-08-01.1` — the requested pathway and organism match the pinned gold exactly, so
the legs will be scored against the same gold cases T-106 used — and **`search calls: 0`**, so no
accidental live search was staged.

**Therefore `--verify-plan` is NOT the gate for this run.** The gate is: two cases, both
`[pinned_override]`, zero search calls. All three hold. Recorded so nobody later reads REFUSED as a
blocked cohort.

### G11 PROCESS INCIDENT — a job that produced no report

The staging job was allocated under `--task ORCH-COHORT`, which is **not a valid G11 task id** (the
allocator requires the `H-004` / `C-056a` / `INIT-001` shape). `g11_evidence.py next` printed a
`ValueError` instead of a path, and that error string was passed to `--json`. `bounded_run.py`
detected it and reported `BOUNDED_RUN_JSON_REPORT_UNWRITABLE` rather than failing silently, which is
the wrapper behaving exactly as designed.

**Consequence, stated plainly: the staging job has no G11 report and is uncertifiable under G11.** It
is not claimed as certified evidence. What certifies the staged plan is the `--verify-plan` job under
`ORCH-701/01`, which read the staged `plan.json` off disk and reproduced its contents — so the
artifact is verified even though the job that produced it is not.

The staging was **not** re-run. Re-staging with `--fresh` would create a second run directory for a
paperwork defect, and the plan it produced is independently verified above.

Both jobs reported `FINAL SURVIVING COUNT : 0` and `cleanup : success`; the heavy lock was acquired
and released by each.

**Standing correction, third time this trap has been hit in the sprint:** always validate that
`g11_evidence.py next` returned a real path before passing it to `--json`. A malformed `--task`
returns an error message, not a failure, and the message becomes the path.

### WHAT THIS COHORT CANNOT MEASURE — read before scoring it

Raised by the peer sprint session and **confirmed against my own itemization**
(`ORCH-092/01-p1-itemize.json`). It is a correction to how the run must be *reported*, not to the
cohort's design.

**The cohort is two STRICT legs. All six Priority-1 survivors live on RESEARCH legs.**

| Paper | strict | research |
|---|---|---|
| `PMC12096016` | **0 false real identifiers** | `NADH`, `NAD+` |
| `PMC12782028` | **0 false real identifiers** | `LIPA`, `LBR`, `SREBF1`, `SREBF2` |

Those six names do not appear on the strict legs of either paper at all. My own Priority-1 document
already said *"all six on research legs"* and *"after C-081 there is no strict-leg false identifier in
the T-106 corpus at all"* — I had the fact and failed to connect it to a strict-only cohort. The peer
connected it.

**Therefore: if this cohort scores Priority 1 as `0`, that is NOT evidence of improvement. It is
evidence the legs the survivors live on were not run.** The only honest label is
**`not evaluated on this cohort`**. Anyone scoring this run must write that, not a zero.

**The research legs are still deliberately NOT added.** No merged card in this wave touches any of the
six mechanisms — Priority 1's remaining cases need an entity-level role/scope carrier that does not
exist, and the one available correction reaches none of them and is blocked on the D-069 ruling.
Running two research legs would re-observe a known failure at extra cost and prove nothing about the
merged code. Excluded on value, not on convenience.

**Also mute: C-087 / F-123.** The ambiguous-rename declination fired on `PMC12444477`, which is not in
this cohort. F-123 is validated by C-087's own behavioural proof — an AppTest seam failing at base on
a byte-identical 209-byte string, plus an exhaustive 7 × 19 = 133-pair enumeration with exactly one
moving transition — and **not** by this run. The record must not leave F-123 looking cohort-validated.

### What the cohort legitimately does measure

* **Priority 5** — after the D-065 reconciliation these two papers *are* the entire strict
  denominator, so this is precisely the right cohort for it.
* **C-086 / F-116** — the EntE → superset complex 3623 promotion, on `PMC12096016/strict`.
* **C-090 / F-117** — the CYP51A1 one-component wrapper rescue, on `PMC12782028/strict`.
* **C-089 / F-125** — the `transports.transporters` orphan slot, on `PMC12096016/strict`.
* And the property no stored artifact can carry: the **composition** of C-086, C-089 and C-090 on a
  freshly mapped payload, which has never been observed because each was measured by replay over the
  same pre-C-086 payloads.

### One inherited-error warning, declined

The peer's own probe reported `Pyridoxal 5'-phosphate` carrying no identifiers on both `PMC12856317`
legs, and **explicitly warned me not to treat that as corroboration** of the C-081 finding: it read
only top-level accession keys and does not follow `mapped_ids` / `external_ids`. C-081's reviewer
replay recorded `before=1` on both legs, which contradicts it. **Not used.** My own finding rests on
the shipped predicate `cofactor_participation` returning
`status="unsupported", reason="cofactor_role_used_by_no_reaction"` (`ORCH-092/10-p1-passc.json`), which
is independent of that probe.

**Correction adopted from the peer:** C-081's merge is **`b869780`** (2026-08-25 11:23), not `2972c34`,
which is the record commit three minutes later. Pinned in the Priority-1 document.

### COHORT REFERENCE DISTRIBUTION — recorded BEFORE the run, so a draw cannot present as a fix

Raised by the peer sprint session; **measured independently here** over every committed leg of the two
cohort papers (`ORCH-702/02-anchor-draw-variance-2.json`). Cap 2's input is Stage 0's
`key_compounds`/`key_proteins`, a **non-deterministic draw**, not a curated core — so the prior
distribution belongs on the page before scoring, not in an argument afterwards. `CLAUDE.md` already
requires re-running a leg before calling a single-leg change a regression; **the same discipline
applies in the improvement direction.**

#### `PMC12096016/strict` — six committed draws

| run | `coverage_ratio` | `unmatched_terms` | recorded reason |
|---|---|---|---|
| `2026-08-18_1328` | 0.8235 | ATP, EntD, Fur | `strict_technical_gates_blocked_export` |
| `2026-08-21_2057` | 0.8333 | ATP, Fur, **MenD** | `semantic_evaluation_failed:actor_named_in_its_own_cited_span` |
| `2026-08-21_2239` | 0.7059 | ATP, NADH, NAD+, EntA, Fur | same |
| `2026-08-22_2147` | **0.8571** | ATP, EntD | same |
| `2026-08-24_1203` | 0.7895 | ATP, **MenD**, Fur, **LDH** | same |
| `2026-08-24_1428` (T-106) | 0.7647 | NADH, ATP, **MenD**, Fur | same |

**Range 0.7059 – 0.8571, spread 0.151, a different anchor set every single time.** `ATP` is unmatched
in all six; `Fur` in five; `MenD` in three. `minimum_core_satisfied` is **True** in all six, so cap 2
never fires here — the recorded blocker is the *semantic* one.

**That is the confound, and it is sharper than a coverage wobble.** C-090 / F-117 removes exactly the
`actor_named_in_its_own_cited_span` failure — F-130 proved on `PMC12444477` that when it does, the
verdict flips and *a different blocker becomes controlling*. So on this leg a Priority-5 move would
need **two** things: C-090 legitimately clearing the semantic blocker, **and** the anchor cap not
surfacing behind it. Only the first is attributable to merged code. **Both must be reported
separately or draw luck will read as a fix.**

#### `PMC12782028/strict` — four committed draws, and the peer's risk read is wrong here

| run | `coverage_ratio` | `minimum_core_satisfied` | recorded reason |
|---|---|---|---|
| `2026-08-21_1822` | 0.280 | False | `requested_core_coverage_below_minimum:0.280<0.500` |
| `2026-08-21_2239` | **0.6923** | **True** | `semantic_evaluation_failed:…` |
| `2026-08-22_2147` | 0.2963 | False | `requested_core_coverage_below_minimum:0.296<0.500` |
| `2026-08-24_1428` (T-106) | 0.2222 | False | `requested_core_coverage_below_minimum:0.222<0.500` |

The peer judged this leg "safer — 0.278 is a long way to travel on variance". **It is not.** There is a
committed counterexample: on `2026-08-21_2239` this leg drew **0.6923**, comfortably above the 0.500
threshold, with `minimum_core_satisfied=True`. **Range 0.222 – 0.692, spread 0.470.** This leg has
already cleared cap 2 once on draw alone. It is the *more* variance-exposed of the two, not the less.

**Consequence: neither cohort leg's Priority-5 outcome is attributable without comparing its draw to
the table above.** Scoring rule for this run: state the new `coverage_ratio` and `unmatched_terms`, say
whether each sits inside or outside the observed range, and only then discuss attribution.

---

## F-132 — Stage 0 draws requested-core terms the gold set forbids exporting, and coverage penalises the correct omission

**Severity MEDIUM · Class: `gold_data_defect` / instrument tension, NOT a pipeline defect ·
Registered 2026-08-27** by the Lead Orchestrator, from the measurement above. Prompted by the peer
sprint session.

`PMC12096016`'s gold `export_rationale` states verbatim:

> *"A fully connected metabolite chain … **Export must exclude MenD, LDH and the transport mentions.**"*

`MenD` additionally appears in that case's `forbidden_identifiers` with kind `heading_or_prose`:
*"deliberately a COMPETING isochorismate sink for the menaquinone branch. It must never be exported as
an enterobactin biosynthetic step."*

Yet **`MenD` is drawn as a requested-core term in three of six committed draws, and `LDH` in one** —
and each then counts as an **unmatched** term, lowering `coverage_ratio` and feeding cap 2.

**The pipeline is being scored down for correctly omitting exactly what the gold forbids it to
export.** Two instruments on the same case disagree: Stage 0's anchor draw treats `MenD`/`LDH` as
requested biology, while the gold's `export_rationale` and `forbidden_identifiers` treat them as
things whose export is a defect.

**Classification matters here.** Under `CLAUDE.md`'s rule, a benchmark failure must be classified
before it justifies code. This is **not** `product_contract_violation` — production behaved correctly.
It is a **`gold_data_defect`** in the sense the sprint uses: the measuring apparatus contradicts
itself. **No code change is justified and none is proposed.**

**Not fixed here, and deliberately so.** The fix would touch either Stage 0's anchor selection or the
coverage denominator, both of which are production seams outside any current card's ownership, and
either could move Priority 4 and Priority 5 at once. It needs its own card and a product-owner ruling
on which instrument is authoritative — the gold's exclusion list or the drawn anchor set.

**Immediate consequence for this cohort:** if `PMC12096016/strict` shows a higher `coverage_ratio` this
run, check first whether `MenD`/`LDH` were simply not drawn. That would be F-132 resolving by accident,
not a merged card working.

### F-132 — RECLASSIFIED the same day: `gold_data_defect` → **`product_contract_violation`**

**Superseding correction, appended rather than rewritten in place.** The registration above stands as
written; **its `Class:` line is wrong and this entry replaces it.** Challenged by the peer sprint
session within the hour, accepted, and then quantified corpus-wide before adopting.

#### Why the original label was wrong, and why it was dangerous

`CLAUDE.md` makes classification pick the remedy. **`gold_data_defect` points a future card at the
gold** — where the natural "fix" is to drop `MenD` from `forbidden_identifiers` or soften *"Export must
exclude MenD, LDH and the transport mentions."* **That would weaken a correct biological constraint to
make a coverage number move.** Merge rule 6 forbids exactly that, and it is the failure mode this
sprint exists to prevent.

**The gold is the instrument that is right here.** Every exclusion on that case is specific and
biologically sound: `MenD` a competing menaquinone-branch isochorismate sink; `lactate dehydrogenase`
(alias `LDH`) a porcine coupled-assay reporter, not an *E. coli* pathway member; `NADH` (aliases
`NAD+`, `lactate`) coupled-assay reporter species from the LDH readout. **Nothing in the gold needs
changing.**

**My own entry contradicted its own label.** The remedy sentence already said the fix *"would touch
Stage-0 anchor selection or the coverage denominator, both production seams"* — that is the
`product_contract_violation` remedy, not the gold one. The peer caught the label and the remedy
disagreeing; they were right and I have taken it.

#### The corrected shape

Cap 2's input is Stage 0's `key_compounds`/`key_proteins` — **not a curated core**, and
`release_status.py` carries `requested_core_source` for exactly that provenance distinction. So:
**the pipeline's own Stage-0 draw pulls in terms the case's `forbidden_identifiers` prohibits, and the
scorer then penalises the pipeline for obeying the gold.** Anchor selection and the forbidden list are
unaware of each other. That is a production-side contract violation.

#### Quantified corpus-wide before adopting (`ORCH-702/03-f132-forbidden-anchors.json`)

Over every committed `quarantine_report.json`:

| | |
|---|---|
| legs carrying unmatched terms | **52** |
| unmatched terms in total | **281** |
| of those, **gold-forbidden identifiers** | **62 (22%)** |
| legs affected | **32** |
| papers affected | **6** — `PMC12096016`, `PMC12312563`, `PMC12444477`, `PMC12452463`, `PMC12782028`, `PMC12856317` |

**This is systemic, not a `MenD` quirk.** Roughly **one coverage penalty in five, corpus-wide, is
levied for failing to match a term the gold forbids exporting.** It spans four of the gold's own
mechanism kinds — `placeholder_product`, `heading_or_prose`, `regulator_as_metabolite`,
`cofactor_as_protein`.

The peer's sharper reading of the T-106 draw is confirmed: `[NADH, ATP, MenD, Fur]` contains **two**
gold-forbidden terms, not one — `NADH` as an LDH-readout species and `MenD` as a competing-branch sink.

#### The double bind, which is the part worth keeping

On `PMC12782028/strict` the gold-forbidden unmatched terms are `LIPA`, `LBR`, `SREBF1`, `SREBF2` —
**the exact four Priority-1 survivors.** The same four entities are simultaneously:

* a **Priority-1 failure** when the pipeline *does* export them with real identifiers, and
* a **Priority-4/5 coverage penalty** when it *does not* match them.

**The pipeline is penalised either way, by two different instruments, for the same four rows.** No
behaviour available to it scores well on both. That is a stronger argument than F-132 alone: it says
the anchor set and the forbidden list must be reconciled before either Priority 1 or Priority 4 can be
read as a measurement of pipeline quality.

#### Remedy, unchanged in substance and now correctly aimed

Still **no code change proposed and none justified from this alone.** The fix touches Stage-0 anchor
selection or the coverage denominator; both are production seams outside any current card's ownership,
and either could move Priorities 4 and 5 at once. It needs its own card and a product-owner ruling on
which instrument is authoritative.

**Explicit instruction to whoever charters it: the gold is not the thing to change.** Removing a
forbidden identifier to raise coverage is a merge-rule-6 rejection, not a fix.

# T-107 READINESS — 2026-08-27 REVISION

**This EXTENDS the table at `LEDGER.md:4359` ("T-107 READINESS TABLE — 2026-08-25, post-ruling
wave"). It does not replace it and there is not a second table.** The 2026-08-25 rows stand as
written except where a row is explicitly revised below; unrevised rows carry forward unchanged.
Handover from the peer sprint session, which authored the original and is holding the branch.

**Verdict unchanged: T-107 remains NO-GO.** What changed is *why*, and the reason is now larger.

---

## A GOVERNING CAVEAT, which must be read BEFORE the per-priority table

**On the affected papers, Priorities 1 and 4/5 score the same rows in opposite directions, so
neither is currently a measurement of pipeline quality.**

On `PMC12782028/strict`, `LIPA`, `LBR`, `SREBF1` and `SREBF2` are simultaneously:

* the **Priority-1 false real identifiers** when the pipeline exports them carrying accessions, and
* a **Priority-4/5 coverage penalty** when it does not match them.

**No behaviour available to the pipeline scores well on both.** Corpus-wide the pattern is not
isolated: **62 of 281 unmatched terms across 32 legs and 6 papers are gold-forbidden identifiers**
(F-132, `ORCH-702/03`) — roughly one coverage penalty in five is levied for failing to match a term
the gold forbids exporting.

**This is a statement about the instrument, not about the code**, and it governs how rows 1, 4 and 5
below are read. It is deliberately placed above the table rather than added as a sixth row, because a
sixth row would compete with the priorities instead of qualifying them.

---

## REVISED ROWS

| Priority | Reachable? | Revision |
|---|---|---|
| **1** — zero false real identifiers | **NO** | The 2026-08-25 row cited F-127 (no entity provenance carrier) and F-128 (D-069 compliance pushes the count *up*). **Both still stand.** Two corrections: the live count is **6, not 8** — the two PLP rows are already withheld by C-081 (`b869780`), which merged one day after T-106 was committed, now confirmed by replay through the shipped predicate rather than inferred. And the row read as a pure extraction problem; **four of the six survivors are simultaneously a coverage penalty**, per the caveat above. Expected T-107 result: **FAIL at 6 or higher.** |
| **4** — requested-core coverage | **NO** | The 2026-08-25 row attributed the failure entirely to Stage 0's non-curated `key_compounds`/`key_proteins` draw and quoted 0/8. Still true as far as it goes. **A different and larger cause is now measured:** 62 of 281 unmatched terms corpus-wide are gold-forbidden (F-132). The draw is not merely uncurated; it pulls in entities the same gold case prohibits exporting, and the metric then penalises the pipeline for obeying the gold. |
| **5** — strict PWML export | **NO** | The 2026-08-25 row said both survivors are *"`correctly_blocked` and measured so"*. **The word "correctly" is withdrawn for `PMC12782028`.** Its `requested_core_coverage_below_minimum:0.222<0.500` penalty is levied partly for not matching `LIPA`/`LBR`/`SREBF1`/`SREBF2`, which the gold forbids exporting. **The block is real; calling it *correct* asserts the instrument is sound, and F-132 says it is not.** The leg is `blocked`, and whether that block is correct is exactly what F-132 puts in question. |

Rows **2** (NOT EVALUATED on 11/20 legs) and **3** (PASS) carry forward unrevised.

## REVISED GATE CONDITION 1

The 2026-08-25 text: *"priority 1 has a safe correction **or an explicitly accepted measurement
limitation**"*, with the note that only the product owner can grant the acceptance. **Still NOT MET,
and the size of what is being asked has grown.**

* **As written 2026-08-25**, the limitation to accept was: *priority 1 cannot reach 0 until a
  participant-provenance carrier and an entity evidence span exist.*
* **As it now stands**, it is: *priority 1 and priority 4/5 are scoring the same rows in opposite
  directions, so neither is currently a measurement of pipeline quality on the affected papers.*

**That is a materially larger thing to ask a product owner to accept, and it must be put as the
larger thing rather than as a footnote to the smaller one.**

## GATE CONDITIONS — status at `65cc96a`

| # | condition | state |
|---|---|---|
| 1 | priority 1 has a safe correction or an accepted limitation | **NOT MET** — and enlarged, above. Acceptance is the product owner's alone. |
| 2 | priority 5 denominator reconciled | **MET** — carried forward; C-088 merged. |
| 3 | the two strict-denominator papers classified honestly | **PARTIALLY WITHDRAWN** — see row 5. `PMC12096016` stands; `PMC12782028` is `blocked`, not `correctly_blocked`. |
| 4 | applicable F-116 / F-123 corrections merged | **MET** — C-086 and C-087 both merged and independently reviewed. |
| 5 | acceptance instrumentation remains honest | **MET, and improved** — C-085 made priority 2 honest; F-132 now names a contradiction the instrumentation could not previously see. |
| 6 | affected-paper validation passes | **IN FLIGHT** — cohort running, `runs_verify/2026-08-27_1341`, 2 strict legs. **Cannot observe Priority 1: all six survivors are on research legs. A `0` there is `not evaluated on this cohort`, never improvement.** |
| 7 | all processes closed | **MET at every checkpoint** — every job zero survivors, cleanup success. |
| 8 | integration clean and pushed | **MET** — `65cc96a`, local = origin = `ls-remote`. |
| 9 | **deterministic suite genuinely green** | **MET — newly, and for the first time this wave.** C-092 and C-093 merged; 5 failed/252 → 2 failed/263 → **0 failed/273, 0 warnings** on a forced fresh compile; SMOKE **473**. |
| 10 | F-130 reconciled | **MET** — all four claims confirmed, claims 3 and 4 measured through the production classifier. Narration only; no production change justified or made. |

**Two conditions unmet, one partially withdrawn, one in flight. Condition 1 remains the one this
session's work cannot clear at all**, and it has grown rather than shrunk.

## A CAUTION ABOUT THE GREEN SUITE, recorded because it has already bitten this sprint

**273 passed / 0 warnings certifies the four modules it ran, not the tree.** The D-065 gold edit broke
`test_c056b_semantic_denominators.py` while SMOKE stayed 473 throughout, because that module is not in
SMOKE — and two lanes then reported the red as pre-existing because their base already contained the
edit. **If the cohort produces anything gold-adjacent, A/B it against a pre-change SHA rather than
trusting the suite.**

Related near-miss, kept visible: C-093 **excluded** the identity-ladder leg rather than pinning its
digest. Pinning would have written the defect into the golden as expected behaviour — the same class
of error as "fixing" F-129's four tests by making them pass against an ambient live database. **Two
near-misses this sprint with that shape.**

## WHAT WOULD CHANGE THE VERDICT

Unchanged in kind from 2026-08-25, enlarged in content. Either:

1. the product owner **explicitly accepts the enlarged limitation** in gate condition 1 — including
   that Priorities 1 and 4/5 currently contradict each other on the affected papers — and states a
   non-zero Priority-1 floor T-107 may run against; **or**
2. the **F-128 / D-069 conflict is ruled** *and* **F-132's instrument contradiction is ruled**, so
   that a participant-provenance representation and a reconciled anchor set can be chartered.

**Merging more cards will not do it.** C-092 and C-093 were both worth merging and neither moves
Priority 1. **Until one of the two above happens, T-107 stays NO-GO, and that remains the correct
outcome rather than a delay.**

# AFFECTED-PAPER COHORT — RESULT, scored against the ledger written before the run

**Run:** `runs_verify/2026-08-27_1341` · **2026-08-27** · 2 strict legs · **2298.54 s (38.3 min)**,
against a 38-minute estimate · G11 `ORCH-703/01`–`04`, every job `FINAL SURVIVING COUNT : 0`,
`cleanup : success`, heavy lock acquired and released. Ran **once**. No leg repeated.

**This is not a benchmark and is not scored as one.** Per the pre-run ledger, Priority 1 is
**`not evaluated on this cohort`** — all six survivors are on research legs and this cohort is
strict-only.

| Paper | Mode | batch status | release status | eligible | seconds |
|---|---|---|---|---|---|
| `PMC12096016` | strict | fail (`contract`) | `review_required` | False | 1388.3 |
| `PMC12782028` | strict | pass (technical) | `review_required` | False | 907.5 |

## Priority 5 did NOT move — and that is the correct outcome

**Still 0/2.** `PMC12782028/strict` passed its technical gates and wrote
**`pathway.review_required.pwml`, not a bare `pathway.pwml`.** That is exactly right: the leg is
`review_required` with `strict_acceptance_eligible=False`, so it must not ship a bare deliverable.
**F-094's closure and merge rule 7 both held live on a fresh payload** — the pathway was preserved as
`review_required` rather than dropped or shipped.

The peer session predicted this leg was the draw-luck risk. It drew **0.3214**, **inside** the prior
range 0.2222–0.6923, and stayed blocked. **No favourable-draw headline occurred.**

## The two-factor decomposition (`ORCH-703/02`) — control arm reproduces on both legs

| leg | control replay | FACTOR 1 (C-090 semantic) | FACTOR 2 (draw / cap) |
|---|---|---|---|
| `PMC12096016/strict` | **REPRODUCES** | nothing to clear — `semantic_evaluation: passed`, `failed_checks: []` | `requested_core_anchors_unmatched:ATP,phosphopantetheine,EntD,Fur`; `coverage_ratio` **0.7778**, INSIDE prior 0.7059–0.8571 |
| `PMC12782028/strict` | **REPRODUCES** | nothing to clear — same | `requested_core_coverage_below_minimum:0.321<0.500`; **0.3214**, INSIDE prior 0.2222–0.6923 |

`actor_named_in_its_own_cited_span` was **not among the failed checks on either leg**, so the
counterfactual was correctly reported `n/a` rather than fabricated. At T-106 that check was the
controlling blocker on `PMC12096016/strict`; it is absent now. **That is consistent with C-090 working
and is NOT claimed as proof** — the draw also changed, and one run cannot separate those.

## F-132 observed live, and it behaves exactly as predicted

* `PMC12782028/strict` — **4 of 19** unmatched terms are gold-forbidden: `LIPA`, `LBR`, `SREBF1`,
  `SREBF2`, kinds `heading_or_prose` and `regulator_as_metabolite`. **The double bind, live**: the
  same four rows are the Priority-1 survivors on this paper's research leg and the coverage penalty
  on its strict leg, in the same run.
* `PMC12096016/strict` — **0** gold-forbidden terms this draw. No `MenD`, no `LDH`, no `NADH` in the
  anchor set, where T-106 drew `NADH` and `MenD`. **F-132's exposure is itself draw-dependent**, which
  is why the pre-run distribution mattered.

## F-116: C-086 measurably improved this leg, and my own pass-criterion still FAILED

The run ledger's property 1 for this leg was *"no reaction actor resolves to complex 3623"*.
**It failed.** Measured (`ORCH-703/04-reaction-actors`, with `03-wrapper-compare`):

**T-106, pre-C-086** — `rx[3]` and `rx[4]` both carry the *identical* actor `enterobactin synthase`
→ **3623**. That is F-116 exactly: one enzyme promoted to a 4-component superset, two reactions
collapsed onto one row.

**This run, post-C-086** — `rx[3]` is `EntE complex`, `complex_id=None`; `rx[4]` is `EntF complex`
plus a distinct `enterobactin synthase complex`. **Property 2 (distinct actors) HOLDS and is a real
improvement over T-106.** `EntC→1143`, `EntB→1189`, `EntA→1190` all preserved — **property 3 HOLDS**,
no collateral.

**But `EntF complex` and `EntD complex` are `single_protein_pathwhiz_wrapper` rows that still carry
`pathbank_protein_complex_id=3623` with all four superset components (EntB, EntD, EntF, EntE).**
C-086 stopped the component-match promotion; **the same superset attachment survives on the
wrapper-generation path.** F-116's shape is narrowed, not closed.

**Reported as a partial result, not a pass.** This is the honest reading of a criterion I wrote before
the run and which the run did not meet.

---

## F-133 — a generated single-protein wrapper still inherits a superset complex id

**Severity MEDIUM · Class: `product_contract_violation` · Registered 2026-08-27 from the cohort.**

`EntF complex` and `EntD complex` are generated wrappers (`generation_reason:
single_protein_pathwhiz_wrapper`) each carrying `pathbank_protein_complex_id=3623` and the four
components of the enterobactin synthase superset. A wrapper that exists to represent **one** protein
must not carry a **four-protein** complex identity.

C-086 is **not** reopened: its charter was the component-match promotion and its own tests pin that
behaviour, which this run shows working. This is a **different code path with the same outcome**, and
it needs its own card with `src/` ownership over the wrapper-generation seam. **A fix must preserve
the `EntC/EntB/EntA` one-component wrappers measured intact here.**

## F-134 — an Unknown-backed generated wrapper is assigned an unrelated organism

**Severity HIGH · Class: `product_contract_violation` · Registered 2026-08-27 from the cohort.**

On an *Escherichia coli* paper, three rows carry **`species = organism = "Arabidopsis thaliana"`**:
`proteins:Unknown`, `protein_complexes:enterobactin synthase complex`, and
`protein_complexes:porcine lactate dehydrogenase`. All three are `Unknown`-backed
(`components=[{"name": "Unknown"}]`). The remaining 12 rows are correctly `Escherichia coli`.

**Arabidopsis thaliana is a plant.** Nothing in this paper is a plant. The T-106 payload for the same
leg carries **no** Arabidopsis rows, so this is newly observed rather than long-standing — though one
run cannot establish whether it is new *behaviour* or a newly-drawn shape.

Two aggravating details: `porcine lactate dehydrogenase` is a **gold-forbidden identity** on this case
(the coupled-assay reporter), so a forbidden entity is being given an organism as well as a wrapper;
and the Stage-3 gate simultaneously reported these complexes as *"missing species/organism"* at
`post_normalization` while the final payload carries Arabidopsis — so the species is being attached
**after** the gate that checks for it.

**Cross-organism assignment is an acceptance-counted category.** This did not reach export only
because Stage 3 blocked the leg for other reasons. **Needs its own card with `src/` ownership.**
Do not fix by defaulting the species to the requested organism — that would launder an unknown into a
confident answer, which is the F-127 failure mode in a new place.

## The Stage-3 block on `PMC12096016/strict`, classified

`final_pre_export_stage3_gates`, 4 blocking issues. **One of them is the pipeline behaving
correctly:** `gate.protein_porcine_lactate_dehydrogenase_is_missing_a_uniprot_or_dr` refuses the
gold-forbidden LDH for lacking an identifier — the pipeline did **not** forge one, which is
Priority-1-correct. The other three are the species/organism gaps of F-134.

**Not a regression attributable to a merged card.** The same Stage-3 `enterobactin synthase complex`
gate family blocked `PMC12452463/strict` at T-106, so the failure mode pre-dates this wave; it has
appeared on a new paper because the draw produced a bare `enterobactin synthase` enzyme here.

## What was NOT observed, reported honestly

**C-087 / F-123 — `AMBIGUOUS_RENAME_TARGET` did not occur.** The cohort does not contain
`PMC12444477`, where it fired. **Not observed, and not chased.** F-123 rests on C-087's own
behavioural proof — the byte-identical 209-byte base failure and the exhaustive 7 × 19 = 133-pair
enumeration with one moving transition — **not on this run.**

**C-089 / F-125, C-088 / D-065, C-083 / F-092** — no live observation was sought and none is claimed.

## Rerun judgement

**No leg is rerun.** F-133 and F-134 are measured production defects, not stochastic conditions, and
both need cards with `src/` ownership this cohort does not have. Rerunning would re-measure a known
result. Both cohort legs' failures are explained; neither is a repair this session may make without a
charter and an independent review.

---

# SUPERSEDING CORRECTION — C-086 does NOT close F-116. My merge record overstates it.

**Branch handed back at `0beda95`**, verified: local = origin = `git ls-remote`, no merge in progress,
0 staged, heavy lock **absent**, only the two `ms-python.isort` IDE processes, product-owner edit
intact at 35/2 `sha256:47e4fafa…`, whole-tree G11 4056 artifacts / 0 non-compliant.

At `LEDGER.md:4554` I recorded C-086 in a table whose column is **`closes`**, against **F-116**.
**That is wrong, and this supersedes it.** C-086 **narrows** F-116. The finding stays **OPEN**.

### Verified myself on the cohort artifact, not taken from the handover

`runs_verify/2026-08-27_1341/papers/PMC12096016/strict/final_mapped.json`:

```
enterobactin synthase complex   id=None   1 comp ['Unknown']                      generated
Isochorismate synthase          id=1143   1 comp ['EntC']
isochorismatase                 id=1189   1 comp ['EntB']
oxidoreductase (entA)           id=1190   1 comp ['EntA']
EntE complex                    id=None   1 comp ['EntE']                         generated
EntF complex                    id=3623   4 comp ['EntB','EntD','EntF','EntE']    generated  <-- SUPERSET
EntD complex                    id=3623   4 comp ['EntB','EntD','EntF','EntE']    generated  <-- SUPERSET
porcine lactate dehydrogenase   id=None   1 comp ['Unknown']                      generated
```

**What C-086 did, and it is real:** `EntE` no longer resolves onto 3623; the two reactions that
carried the *identical* `enterobactin synthase` actor at T-106 now carry distinct actors; and
`EntC`→1143, `EntB`→1189, `EntA`→1190 are untouched, so the one-component controls hold live on a
fresh payload exactly as its tests predicted.

**What it did not do.** F-116's own words are *"a supported enzyme is replaced by a superset protein
complex, injecting catalysts that do not perform the step."* `EntF complex` and `EntD complex` do
exactly that, on the same paper, after the merge — each a **generated single-protein wrapper**
carrying 3623 and all four components. C-086 closed the **component-match** path; the same attachment
survives on the **wrapper-generation** path.

**C-086 is NOT reopened.** Its charter was the component-match path, its tests pin that path working,
and the reviewer verified the boundary hunk-by-hunk. The card is fine. **The register was wrong to
mark the finding closed on the strength of it**, and that was my entry, not the card's claim.

### The lesson, and it is the second time this wave

**A card's charter and a finding's scope are different objects.** "The card passed its gates" does not
license "the finding is closed", and a `closes` column invites exactly that elision. I made the same
class of error on the c056b repair, where two assertions read as coverage and provided none. **Both
were caught by someone re-measuring rather than re-reading** — REV-089 ran my own file against a
pre-change SHA; this one needed a live leg. Neither would have been caught by reading the diff again.

**Register `closes` only against a finding whose scope has been re-measured, not against the card that
was chartered at it.**

### F-133 and F-134 — verified independently before accepting them

**F-133** confirmed exactly as above: two generated wrappers inherit a superset complex id. The
handover's framing is right — a fix must preserve the one-component wrappers, which this run measured
intact.

**F-134** confirmed, and the correlation is cleaner than a count. On the same artifact, requested
organism *Escherichia coli*:

```
enterobactin synthase complex   organism = species = "Arabidopsis thaliana"   Unknown-backed
porcine lactate dehydrogenase   organism = species = "Arabidopsis thaliana"   Unknown-backed
every other protein_complex     organism = None, species = "Escherichia coli"  not Unknown-backed
```

**Every Unknown-backed wrapper gets Arabidopsis; every non-Unknown-backed one gets *E. coli*.** The
organism is coming from whatever the placeholder record carries, not from the requested or observed
organism. One of the two is the **gold-forbidden porcine lactate dehydrogenase** — a forbidden entity
handed both a wrapper and a confident foreign organism.

*Scope note on my own check:* I measured **two** rows in `entities.protein_complexes` only; the
handover reports **three** across buckets. No contradiction — I did not look at the other buckets.

The handover's instruction not to "fix" this by defaulting species to the requested organism is
correct and worth restating: that would launder an unknown into a confident answer, which is F-127's
failure mode in a new place.

---

# T-107 READINESS TABLE — CORRECTIONS, superseding the 2026-08-25 table at `LEDGER.md:4359`

The table stands except for these rows, which F-132 has invalidated.

**A caveat that governs rows 1, 4 and 5, and should be read before them.** On the affected papers,
**Priority 1 and Priority 4/5 score the same rows in opposite directions.** `LIPA`, `LBR`, `SREBF1`
and `SREBF2` are a Priority-1 failure when exported with real identifiers and a Priority-4/5 coverage
penalty when not matched — measured live on the cohort, 4 of 19 unmatched terms on
`PMC12782028/strict`. Corpus-wide, **62 of 281 unmatched terms (22%) across 32 legs and 6 papers are
gold-forbidden identifiers.** **Neither priority is currently a measurement of pipeline quality on
those papers.** That is a statement about the instrument, not the code.

| row | correction |
|---|---|
| **1** | Stands (`NO — and it got worse`), but the cited cause is incomplete. It reads as a pure extraction problem; **four of the six survivors are simultaneously a coverage penalty** under F-132. |
| **4** | I attributed 0/8 entirely to Stage-0's non-curated draw. **F-132 is a second and larger cause** and my row does not mention it. |
| **5** | **"Both survivors are `correctly_blocked` and measured so" is withdrawn.** For `PMC12782028/strict` the coverage penalty is levied partly for not matching gold-forbidden terms, so "correctly" asserts the instrument is sound and F-132 says it is not. The block is real; the word **correct** is not defensible and is removed. |

**Gate condition 1 is now a larger ask than I wrote.** I framed it as *"priority 1 has a safe
correction **or an explicitly accepted measurement limitation**"*. The limitation to be accepted is no
longer *"priority 1 cannot reach 0 because a carrier is missing"* but **"two priorities score the same
rows in opposite directions, so neither measures pipeline quality on the affected papers."** It should
be put to the product owner as that larger thing, not as a footnote to the smaller one.

**T-107 remains NO-GO**, and the cohort did not change it. F-133 and F-134 add two more `src/`-owning
cards to the queue.

### What the cohort did establish, recorded so it is not lost

**Priority 5 held at 0/2 and the draw-luck risk did not materialise.** `PMC12782028/strict` drew
**0.3214** — inside the prior observed range — and stayed blocked, so there is no favourable-draw
headline to explain away. It passed technical gates and wrote **`pathway.review_required.pwml`, not a
bare `pathway.pwml`**: F-094's closure and merge rule 7 both holding live on a fresh payload.

**The decomposition probe's control arm did its job.** It reproduced both legs' recorded status, and
`actor_named_in_its_own_cited_span` was absent from the failed checks on both, so the counterfactual
was reported **`n/a` rather than fabricated**. That check *was* the controlling blocker on
`PMC12096016/strict` at T-106 and is gone — consistent with C-090 working and **not claimed as proof**,
because the draw also changed. That restraint is the correct reading.

**C-087's `AMBIGUOUS_RENAME_TARGET` did not occur** — the cohort carries no `PMC12444477`. Recorded as
**not observed**, not chased. F-123 rests on C-087's own behavioural proof, not on this run.

---

# WAVE RECORD — C-094 / C-095 / C-096 / C-097 / C-098, and a machine crash

Written by the Lead Orchestrator at integration tip `fc3dd24`. **Nothing merged yet, and that is
deliberate.** Every card below is complete or in progress on its own branch; the integration branch
carries charters, findings and evidence only.

## Branch register for this wave

| Branch | Tip | State |
|---|---|---|
| `card/C-094-f134` | `53eaf24` | **complete**, reviewed, blocked on its companion C-098 |
| `card/C-096-f129` | `19875cc` | **complete**, reviewed → APPROVE WITH CORRECTIONS, one round outstanding |
| `card/C-095-f133` | `0128fa6` | uncommitted partial: `map_ids.py` +187, 801-line test file, 10 G11 jobs |
| `card/C-098-f135` | `5475ebc` | uncommitted partial: `process_normalizer.py` +203, `release_status.py` +112 |

Worktrees `C:/t/c094`, `C:/t/c095`, `C:/t/c096`, `C:/t/c098`, plus the reviewer's base tree
`C:/t/rev096base` (detached @ `7862fcc`). None may be pruned while work is outstanding.

## The machine crashed mid-wave. Nothing was lost.

Three agents were in flight. The host crashed; all three processes died. Verified after restart, not
assumed: integration at `f3ad3d2` with local = origin = `git ls-remote`, no merge in progress,
nothing staged, **heavy lock ABSENT** (not stranded), zero sprint-owned Python, product-owner
`streamlit_app.py` intact at 35/2 `sha256:47e4fafa…`, every card branch and every uncommitted partial
present on disk.

**The peer session `project14-t2pw-60` did not survive.** It had been this wave's independent
adversarial reviewer and its catches changed the work three times — the three-branch component
overwrite in C-095's charter, the `EntE` abstention trap, and the release-classifier measurement that
established F-135. Its evidence is committed under `ORCH-704`, `ORCH-705` and `ORCH-707`.

**Standing instruction for a post-crash resume:** a G11 report whose job was killed is **not
evidence**, even though the file exists. Every resumed agent was told to re-validate each report it
intends to cite for a real `exit_reason`, exit code, `FINAL SURVIVING COUNT : 0` and
`cleanup : success`, and to discard and re-run any that fails that check.

---

## REV-096 — independent adversarial review of C-096. Verdict: **APPROVE WITH CORRECTIONS**

50 G11 reports, 15 pin verdicts, all `0 non-compliant`. The reviewer reproduced every load-bearing
number itself rather than taking the author's: the `4 failed / 8 passed` base failure with the exact
four node ids against a live PathBank, `12 passed` at tip, SMOKE **473**, the affected sweep
`2 failed / 78 passed` identically at base and tip, and `92 passed` / `102 passed` with `.env`
physically removed.

**Three pieces of work went well beyond the brief and are worth keeping.**

**A mutation matrix** (`REV-096/55`) that loads mutated copies of `compound_resolution.py` under the
real module name and replays the load-bearing assertions:

```
M0 unmutated        -> SURVIVES (control)
M1 elif -> if       -> KILLED   available True while reason still claims the caller disabled it
M2 arms swapped     -> SURVIVES
M3 db_reason reset  -> KILLED   reason degrades to db_not_configured
M4 no __deepcopy__  -> SURVIVES
```

**M2 corrects the author and the source comment.** Both claim the mechanism rests on matching the
sentinel *before* the ambient substitution. It does not — the two conditions are identity tests on
distinct objects and are mutually exclusive, so arm order is irrelevant. **The invariant is the
`elif`, not the order.** M1 shows the `elif` is genuinely load-bearing; M4 shows `__deepcopy__` is
redundant because `__reduce__` already routes `deepcopy` through the singleton.

**A 45-scenario `None` differential** (`REV-096/13`, `14`) — 5 populations × 3 resolver selections ×
`strict_db` both ways, through `run_prefreeze_resolution` and `build_pwml_ir`, capturing the full
report, payload marker, compounds, preflight and sorted warning/error codes. Base vs tip:
**byte-identical, sha256 `e750f970b04126…`**. That is a far stronger preservation proof than the
author's single base-form re-run, and it settles PRODUCT_CONTRACT § 8.

**A retraction of its own method.** Jobs `10` and `17` hid the database by exporting empty
`PATHBANK_DB_*`. That is **defeated in-process** by `src/t2pw/llm/client.py:22`
`load_dotenv(dotenv_path=ENV_PATH, override=True)`, which re-applies `.env` over the exported values
for any test that transitively imports the LLM client. The reviewer caught it from a contradictory
assertion, voided both jobs as no-DB evidence, and replaced them with `19`/`20`, which physically
rename `.env` away.

> **Sprint-wide, and it will bite the next agent: exported `PATHBANK_DB_*` variables cannot hide the
> database. Only moving `.env` works.**

### Findings, and their disposition

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | MED | `test_streamlit_quarantine_boundary.py::test_research_mode_keeps_the_unmapped_candidate_and_does_not_block` is a **third** instance of F-129's ambient-dependence class — FAIL with the DB live at base *and* tip, PASS with `.env` removed at both | **Registered as F-136.** Pre-existing, not chargeable to C-096 |
| 2 | MED | `NO_DB_RESOLVER` is absorbed by `_REVIEW_REQUIRED_REASONS` and demotes release status under the string `resolution_report_not_ok:db_unavailable` — false, since nothing was unavailable | **Registered as F-137.** Out of C-096's boundary |
| 3 | LOW | The class docstring claims a non-singleton would "fail **open**, back onto the ambient database". It would not: it is not `None`, so `compound_resolution.py:687` defaults a missing `available` to **True** and produces `available: True` with **no reason at all** — worse than claimed | **Correction required before merge** |
| 4 | LOW | The stated invariant ("matched BEFORE the ambient substitution") is the wrong one; it is the `elif` | **Correction required before merge** |
| 5 | MED | Unused re-export at `prefreeze_resolution.py:54-60` and `:80-85`, at module scope, outside the four authorized function bodies | **Boundary extension granted — see below** |
| 6 | INFO | `__deepcopy__` is dead beside a working `__reduce__` | Recorded, not changed |
| 7 | INFO | A test cites an evidence path not an ancestor of this branch; resolves after merge | Benign |
| 8 | INFO | Every C-096 report records `repo_head 7862fcc` because jobs ran uncommitted | Normal |

### Ruling on FINDING 5 — extension GRANTED, recorded rather than waived silently

The charter narrowed `prefreeze_resolution.py` to "the `db_resolver` parameter threading on the
signatures at `:340`, `:606`, `:1393`, `:1644`". The import and the two `__all__` entries sit at
module scope, outside all four, and I verified the reviewer's claim myself: both test files import
the names from `t2pw.pwml.compound_resolution` directly, and no `from … prefreeze_resolution import *`
exists anywhere in `src/` or `tests/`. **The re-export is genuinely unused.**

I am granting the extension rather than requiring removal, for reasons that should be checkable:

1. The two hunks add **public API surface for the capability this card exists to add**, on the module
   that is the public entry point — `run_prefreeze_resolution` is what callers call, and requiring
   them to import the sentinel from a different module than the function it parameterises is a wart,
   not a safeguard.
2. The boundary's **stated reason** was that C-011's golden fixture pins the freeze-seam logic. These
   hunks touch no freeze-seam logic, and the reviewer ran
   `tests/test_c011_freeze_seam_golden_equivalence.py` **green at tip**.
3. The change is strictly additive and behaviour-free — proved, not asserted, by the 45-scenario
   byte-identical differential.

**Not waived: FINDINGS 3 and 4.** A comment stating the wrong invariant is precisely what misleads a
later reader, and this sprint has paid for that failure twice. C-096 does **not** merge until both
docstring claims are corrected.

**Also settled by the review, honestly:** no signature actually changed, because all four were
already `db_resolver: Any` at base. "Threading" was a no-op by construction. And the author's own
coverage gap was real and contained FINDING 1 — the `s8` partition it ran does not cover the seam,
and the 23 `qb` nodes it skipped are exactly where the residual defect sits. *"`s8` was enough" was
not a safe inference.*

---

## F-136 — a third ambient-dependent test, and F-129's class is narrowed rather than closed

**Severity MEDIUM · pre-existing · registered 2026-08-27 from REV-096.**

`tests/test_streamlit_quarantine_boundary.py::test_research_mode_keeps_the_unmapped_candidate_and_does_not_block`
fails `assert _before[_section] == _after[_section]` — *"processes moved pre-freeze"* — with a live
PathBank at **both** `7862fcc` and `19875cc`, and passes at both with `.env` removed.

Together with the two the C-096 author found
(`test_pwml_writer.py::test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance` and
`test_canonicalization_preflight_and_species.py::test_preflight_warns_when_no_db_and_no_covering_index`),
**at least three tests outside C-096's ownership remain green-or-red on whether a developer's PathBank
happens to be running.**

**Consequence for gating:** the Chunk D gate **cannot go green in this environment**, at base or tip,
with the database up — `core` carries the `test_pwml_writer` failure and `qb` carries this one.
Anyone reading a red Chunk D this wave must classify it against this entry before calling it a
regression.

**F-129 is narrowed, not closed.** C-096 fixes the seam it owns and the four tests it was chartered
to repair. The class survives elsewhere.

## F-137 — the sentinel's honest reason is collapsed one layer up

**Severity MEDIUM · registered 2026-08-27 from REV-096 · outside C-096's boundary.**

Measured at tip (`REV-096/21`):

```
reachable_no_match  strict=True  ok=False  failures={'compounds': 'resolution_report_not_ok'}
NO_DB_RESOLVER      strict=True  ok=False  failures={'compounds': 'resolution_report_not_ok:db_unavailable'}
                                           review_required={'compounds': 'resolution_report_not_ok:db_unavailable'}
```

`prefreeze_resolution.py:1673` classifies on `db_resolution.available is False` alone, so
`_REVIEW_REQUIRED_REASONS` absorbs the sentinel and — via D-068 and `release_status.py:605-660` —
demotes the release status, emitting `db_unavailable` for a run in which **nothing was unavailable**.

C-096 spends `compound_resolution.py:584-592` arguing that reporting the caller's decision as a
failure "would be untrue about the run", and then the distinction is collapsed one layer up. The
direction is **conservative** (a demotion, never a loosening) and **no production caller can reach it
today**, since all three omit `db_resolver`. Disclosed and chartered rather than fixed inside C-096.


---

# T-107 READINESS — 2026-08-27 revision 2, extending the table at `LEDGER.md:4359` and its 2026-08-27 revision

**This extends the existing table. It does not replace it and there is no competing table.** Rows
not mentioned here stand as previously written and revised.

**Verdict: T-107 remains NO-GO.** The blocker is unchanged and none of this wave's engineering
touches it.

## What this wave changed, and what it did not

Five cards were chartered and four implemented. **None of them moves a T-107 priority**, and that was
predictable before they were written — they are correctness repairs in the mapping and resolution
seams, not coverage or identity-provenance work.

| Card | Finding | Moves a T-107 priority? |
|---|---|---|
| C-094 | F-134, placeholder-derived false organism | **No.** Cross-organism assignment is acceptance-counted, but the affected rows never reached export |
| C-095 | F-133 / F-116's open path, superset complex identity | **No.** Priority 2 is `NOT EVALUATED` on the affected legs for an unrelated reason |
| C-096 | F-129, `db_resolver=None` overloaded | **No.** Test-infrastructure honesty; no scored artifact changes |
| C-098a/b | F-135, unresolved placeholder species must not drop the pathway | **No**, but see below — it prevents a *regression* in priority 5 |
| C-097 | F-131, `ref`/`id` in `bench.semantic._names` | **No.** Corpus impact measured 0 |

**C-098 is the one worth stating precisely.** It does not improve priority 5; it prevents C-094 from
*damaging* it. Without the companion, C-094 alone would convert the majority Unknown-backed wrapper
shape — **25 of 31 wrappers, measured** — from a leg that produces a PWML into one that produces
none. Merging C-094 alone would have moved priority 5 in the wrong direction while appearing to fix a
correctness defect.

## Gate condition 1 — unchanged, and unchanged for the same reason

The § 8 gate condition that cannot be cleared by engineering is still condition 1: *"priority 1 has a
safe correction **or an explicitly accepted measurement limitation**."*

The limitation to be accepted remains the larger one recorded in the 2026-08-27 revision:

> **On the affected papers, Priorities 1 and 4/5 score the same rows in opposite directions, so
> neither is currently a measurement of pipeline quality.**

`DECISION-BUNDLE-F132-PRIORITY1.md` (committed `7862fcc`) puts that to the product owner with the
corpus figures — **62 of 281 unmatched terms across 32 legs and 6 papers are gold-forbidden**, and on
`PMC12782028/strict` the four gold-forbidden unmatched terms are **exactly** the four Priority-1
survivors. It asks for two rulings: adopt or amend option A (separate the measurements), and accept
or decline option B (a stated Priority-1 floor of 6). **Neither has been given, and an orchestrator
may not give either.**

## Against § 8's eight conditions, revised

| # | condition | state at `c49562c` |
|---|---|---|
| 1 | priority 1 has a safe correction or an accepted limitation | **NOT MET.** Unchanged. The decision packet is written and waiting; nothing in this wave could clear it |
| 2 | priority 5's denominator reconciled | **MET** — unchanged |
| 3 | the two strict-denominator papers classified honestly | **MET**, with the 2026-08-27 withdrawal of the word *"correctly"* for `PMC12782028` still standing |
| 4 | applicable F-116 / F-123 corrections merged | **STILL NOT MET, and now more precisely.** C-086 narrowed F-116; C-095 addresses its remaining wrapper-generation path but is **committed, not merged** — under independent review at `194d6cd` |
| 5 | acceptance instrumentation remains honest | **MET and improved.** F-132's contradiction is named; F-136 records that at least three tests are green-or-red on ambient infrastructure, so a red Chunk D is now classifiable rather than mysterious |
| 6 | affected-paper validation passes | **NOT MET — and deliberately not attempted.** No live leg was run this wave. Nothing was merged, so there was nothing to validate; running one would have measured the unmerged tip |
| 7 | all processes closed | **MET at every checkpoint**, including across a host crash: heavy lock absent, zero sprint-owned Python, every completed job `FINAL SURVIVING COUNT : 0` / `cleanup : success` |
| 8 | integration clean and pushed | **MET** — local = origin = `git ls-remote`, working tree carries only the pre-existing caches, scratch files and the product-owner edit |

**Three of eight unmet, the same three as before, and condition 1 is still the one no amount of
engineering clears.**

## What would change the verdict — unchanged, restated so it is not softened

Merging C-094, C-095, C-096, C-097 and both C-098 arms **will not** change it. All six are worth
merging on their own contract-grounded merits and **none moves priority 1 toward zero.**

The verdict changes only when the product owner rules on `DECISION-BUNDLE-F132-PRIORITY1.md`:

* **Option A alone** makes priorities 4 and 5 readable for the first time, but leaves condition 1
  unmet — A does not touch priority 1.
* **Option B alone** clears condition 1 by explicit acceptance, but leaves 4 and 5 measuring the
  instrument rather than the pipeline.
* **A and B together** make T-107 a *partial but honest* measurement. Priority 2 would remain
  `NOT EVALUATED` on more than half the legs for an unrelated reason — D-067 precondition 3, an
  independent biological review the orchestrator cannot supply and the credit rule forbids buying.

**Until then T-107 stays NO-GO, and that remains the correct outcome rather than a delay.** No
predicted post-ruling values are given here; predicting them is the class of claim this sprint has
repeatedly had to withdraw.

## One thing that must not be repeated

**The affected-paper cohort at `runs_verify/2026-08-27_1341` was not rerun**, no new cohort was run,
no T-107 leg was run, and no live model credit was spent this wave. The cohort's result is unchanged
and re-running it would re-measure a known answer.


---

# F-135 — REFRAMED, and C-098c REFUSED. The finding is not what I registered.

**Superseding correction, appended rather than rewritten in place.** The F-135 registration above
stands as written; **its central claim is the wrong description of the mechanism** and this entry
replaces it. Established by C-098b's measurement of a second refusal point, then followed into the
exporter by the Lead Orchestrator.

## What C-098b found

`run_pwml_export` has **two** refusal points before any XML exists. C-098b clears the first — the
strict Stage-3 post-normalization gate. The second refuses the same payload on the same fact:

* `stage_contracts.validate_pre_export` → `ir.validate_required_pwml_contract`, code
  **`protein_complex_missing_species`**, returning `{"ok": False, "output_path": ""}`.

So the 25-shape moves from `diagnostic_only` to a **run FAIL**. Both are no-PWML states. C-098's
§ 7.1 target — *"`review_required` **with PWML**"* — is **not reached by the gate arm**, and C-098b
says so rather than claiming success.

**C-098b corrected its own earlier measurement to find this.** Its first probe measured only the
first refusal point and reported `pathway.review_required.pwml`; it had already reported that to me
as met, and I had acted on it. It caught the error itself, committed **both** the wrong measurement
(`C-098b/03-g9-tip`) and the corrected one (`06-g9-tip-corrected`) side by side, and made the suite
**assert the shortfall** (`required_field_gate_ok is False`, `exporter_would_build_xml is False`) so
it fails loudly if anyone later assumes the gap is closed. The clause that caught it was § 7.9,
which required running a test that could contradict the card and reporting what it actually did.

## Why C-098c is refused

`validate_required_pwml_contract`'s own docstring: *"Pre-export contract validator … Checks **every
required field defined by the PWML-ready contract**"*. Species on a protein complex is a **required
field of the export format**, raised as an `err`, not a `warn`.

And the decisive fact, checked one layer further down. `pwml/writer.py` resolves a record's species
from a long candidate chain — `species`, `organism`, `species_id`, `pathbank_species_id`,
`taxonomy_id`, `species_ref.*`, `mapping_meta.*` — and when none resolves it ends at:

```python
return default_species_id
```

**The writer silently assigns a default species.** So a C-098c that punched through the
required-field gate would replace a false *Arabidopsis* stamped at mapping time with a **false
default stamped at export time** — the fix recreating the original defect one stage later, inside
the exporter. That is `PRODUCT_CONTRACT` § 5 and merge rule 8, and it is the single failure pattern
this sprint has caught most often (C-092 shipped it; C-093 shipped it one level up).

**C-098c is not chartered. § 7.1 of `C-098.md` was an unreachable requirement and is withdrawn.**

## The reframing — this is the part that matters

F-135 as registered says C-094 *"turns a leg that would have produced a PWML into one that produces
none"*. True as an observation, **wrong as a description of the mechanism**, and the wrong framing
led me to charter a gate exemption chain.

**Three independent gates require species on a protein complex** — the strict Stage-3 gate, the
PWML-ready contract, and the writer's own resolution. Those 25 wrappers were therefore **never
legitimately exportable**. The only thing that ever carried them through the format contract was the
**fabricated *Arabidopsis* species**.

> **C-094 does not drop pathways. It stops fabricating the field that made unexportable entities
> look exportable.**

`PRODUCT_CONTRACT` § 1 is explicit on both halves: the system *"must **never invent** … identities …
merely to guarantee a PWML file"*, and *"a smaller supported pathway is preferable to a larger
contaminated one."* On that reading C-094 **is the contract being enforced**, and the leg-level
consequence is a product decision rather than a regression C-094 introduces.

## What is left is § 13's standing disagreement, and it is escalated

`PRODUCT_CONTRACT` § 13 makes `placeholder_backed_proteins` a **standing disagreement, not a
defect** — the pipeline treats `Unknown`-backed export as legitimate biology preservation, and **no
agent may "fix" it. Escalate.** Three cards into a gate-exemption chain is exactly the fixing § 13
forbids, and I am stopping.

The question for the product owner: **what should happen to an entity that is preserved as biology
but cannot satisfy the export format without a fabricated field?** Three coherent answers, none of
which an orchestrator may choose:

| | Outcome | Cost |
|---|---|---|
| **Quarantine the entity, export the rest** | the pathway ships smaller and honest — `PRODUCT_CONTRACT` § 1's *"smaller supported pathway"* | the reaction loses its actor, and the wrapper exists precisely because the importer refuses a bare protein as an enzyme, so this may cascade |
| **Block the leg** | no contaminated output; today's behaviour after C-094 | a valid pathway core is suppressed because one peripheral actor is unresolved — § 1's own unacceptable-blocker list |
| **Keep the placeholder species** | today's behaviour before C-094 | a fabricated organism in a released payload; F-134 |

**No option is free, and that is why it is a ruling and not an implementation.**

## Disposition of the three branches

| Branch | Tip | Disposition |
|---|---|---|
| `card/C-094-f134` | `53eaf24` | **Correct on its own merits.** On the measured corpus it costs **zero** PWML — the single PWML-producing leg is in the 6-shape and is *improved*, recovering *Homo sapiens*. Under independent review; merges on its own if approved |
| `card/C-098a-cap` | `8cfa33e` | **Held.** Inert and harmless, but merging it alone adds capability with no producer. Held with C-098b so the pair stays coherent for whatever is ruled |
| `card/C-098b-gate` | `b589821` | **Held, and not merged.** It converts `diagnostic_only` into a run FAIL — no PWML either way — while costing four baseline moves. It does not achieve its goal and its cost is real |

**Four baseline moves are NOT authorized**, since nothing they belong to is merging. C-098b did not
edit any of them. For the record they were: `test_c094_placeholder_species.py::test_the_existing_stage3_gate_observes_a_wrapper_with_no_species`
(C-094's own seam-B test, which asserts the very blocking C-098b removes), and three
`test_c011_freeze_seam_golden_equivalence.py` goldens whose diff is exactly
`Extra items in the left set: 'review_findings'`.

## Two things from C-098a/b that survive the refusal and are worth keeping

**A cross-module contract that armed itself.** C-098a's `test_rule_name_matches_the_gate_arm_once_it_exists`
asserts its restated `GATE_REVIEW_SEVERITY` and rule name equal `process_normalizer`'s. It **skipped**
while the gate arm was absent and went **SKIP → PASS** the moment C-098b landed. The two halves could
not drift apart and nobody had to remember to add the check. That is a pattern worth reusing.

**A G9 harness that runs on both trees.** C-098b's `evidence/c098b_g9_base_vs_tip.py` imports **only
base-existing symbols**, so it executes at base as well as tip. A suite importing the new constants
would have died at base with an `ImportError` — and symbol absence dressed as a base failure is
exactly what G9 forbids.


---

# REV-094 — independent review of C-094. Verdict: **APPROVE WITH CORRECTIONS, do not merge.**
# And it forces a correction to my own reframing above.

The diff is correct, minimal, in-boundary and genuinely subtractive; the reviewer reproduced every
number the author reported, verified the base tree's blob identity itself, and confirmed the seam-B
determination independently. **No change to `src/t2pw/mapping/map_ids.py` is required.**

**But the blast radius was under-measured by three, and the third of them is a pinned product
position.**

## The three baselines nobody ran — verified twice, by the reviewer and then by me

`tests/test_protein_export_policy.py`, measured by me on the hash-verified trees
(`ORCH-709/01`, `/02`, both exit-recorded, survivors 0, cleanup success):

```
base C:/t/c094base @ 14121d5   63 passed
tip  C:/t/c094     @ 53eaf24    3 failed, 60 passed

FAILED ::test_the_sentinel_component_is_not_treated_as_an_unused_protein
FAILED ::test_strict_gates_accept_a_correctly_formed_unknown_backed_complex
FAILED ::test_later_normalization_keeps_the_wrapper_and_drops_the_original
```

**Why it was missed is structural, not careless.** That file is in **neither SMOKE nor Chunk D**. It
is a `BASELINE.md` group-05 file — *the same group as the `test_pathbank_unknown_fallback.py` the
charter singled out as untouchable* — and it is invisible to every gate this sprint runs. The
author's affected-file sweep never selected it because my charter's affected list never named it.
**That is my error, not the author's.**

**So C-094's true delta is FOUR moved baselines, not one.** The commit message's *"that one test is
the only move"* is false outside the charter's six-file list.

## The correction to my own reframing — I resolved O-1 by assertion, and I may not

My reframing above says those 25 wrappers *"were never legitimately exportable"* and that the
species was *"fabricated"*. **Read `tests/test_protein_export_policy.py`'s pinned assertions before
accepting that:**

```python
assert normalization_report["gate"]["ok"] is True
assert validate_post_normalization(normalized, normalization_report["gate"])["ok"] is True
assert validate_required_pwml_contract(normalized, strict_db=False)["ok"] is True
```

under the section header *"strict gates: accept the right shape, reject the poser"*. **The pinned
product position is that a correctly-formed `Unknown`-backed complex passes all three gates,
including the PWML-ready contract.** Under that design the sentinel's species is not a fabrication —
it is part of a coherent *"this row is the PathBank Unknown record"* marker, and the complex is
exportable **by design**.

That is exactly `DECISIONS.md` **O-1**, quoted verbatim:

> | O-1 | `placeholder_backed_proteins` (21 in the pinned run): gold-set error class, or legitimate
> biology preservation? | **any branch that touches protein export policy** | It is a genuine
> disagreement between two intentional designs, not a defect. **TRAP-3 forbids agents from resolving
> it.** |

**My reframing stated one side of O-1 as established fact. It is not, and I withdraw that half of
it.** What survives, and needs no O-1 ruling:

* The measured row is **internally contradictory** — `species: "Arabidopsis thaliana"` and
  `species_id: 4` alongside `species_name: "Escherichia coli"`, `taxonomy_id: "562"` and a
  `species_ref` that resolved *E. coli* at confidence 1.0. A row asserting two organisms is wrong
  under either reading of O-1.
* `PMC12856317/strict` at `runs_verify/2026-08-04_1754` shipped `pathway.pwml` — `release_ready` —
  with *Arabidopsis* on a **human** ALAS2 wrapper whose correct species was present underneath.
* Three gates do require species; the writer's chain does end at `return default_species_id`.

**What does NOT survive without a ruling** is the conclusion that removing the species is the right
repair. C-094 inverts a pinned product statement on the exact surface O-1 blocks — *by consequence,
not in prose*, which is if anything worse, because nothing in the diff announces it.

## Disposition — C-094 is HELD

**C-094 does not merge.** Not because the code is wrong — the reviewer and I both find it correct —
but because merging it decides O-1, and O-1 is TRAP-3 protected. Together with C-098a and C-098b,
the whole F-134/F-135 chain now waits on one product-owner ruling.

**Four baseline moves are NOT authorized.** None has been edited.

## Two further findings from the review, both real

**MEDIUM — the moved baseline silently drops PWML serialization coverage.**
`test_pathbank_unknown_fallback.py:420` short-circuits at `assert stage3_report["ok"] is True`, so
everything after it stops running: the `validate_required_pwml_contract` assertion, `build_pwml_ir`
errors `== []`, and the exact-sentinel PWML assertions (`protein/id == "9659"`, `name == "Unknown"`,
`species-id == "4"`, `uniprot-id == "Unknown"`,
`protein-complex-protein/protein-id == "9659"`). **Nothing in `test_c094_placeholder_species.py`
replaces it** — the new file never imports `build_pwml_ir` or `DeterministicPwmlBuilder`. Whoever
re-authors that baseline must **restore the coverage, not merely relax line 420.**

**MEDIUM — the writer's terminal fallback, independently confirmed.**
`writer.py:1137-1165` `protein_species_id()` ends `return default_species_id`
(`self._ir_pathway_species_id`). A species-less wrapper reaching the writer on a path where the
missing-species finding is not blocking is emitted under the **pathway's** species — the requested or
dominant organism, in the PWML itself. Pre-existing, not chargeable to C-094, and already the reason
C-098c was refused. Recorded here because it is where the card's *"carries nothing"* property stops
holding.

## Process notes worth keeping

* Both `C:/t/c094` and `C:/t/c094base` lack `.env`, so PathBank was **down symmetrically in both
  arms**. Base↔tip deltas are valid; any DB-dependent *naming* is an artifact of both. The reviewer
  said so rather than letting it pass.
* The author's `10-chunkd-core.json` was a five-file pytest selection, not the Chunk D gate. The
  reviewer ran the real `chunk_d_gate.py run --only core`: **160 passed**,
  `executed=160/160 omissions=0 additions=0`. The claim was true; the evidence for it was not.
* **A gate-invisible baseline group is a standing hazard.** `test_protein_export_policy.py` is in no
  chunk, so nothing this sprint runs would ever have caught its move. It is the same blind spot that
  hid the c056b regression behind a green SMOKE, in a new file.


---

# REV-095 — independent review of C-095. Verdict: **APPROVE WITH CORRECTIONS. The corrections are blocking.**

The core F-133 fix is correct and in-boundary, and its preservation obligations are genuinely
measured. The reviewer **re-measured every headline number itself** and only two claims did not
survive. Both are recorded below as F1.

**Verified independently, none of it taken on trust:** the diff is purely additive, 187 insertions
and **zero deletions**; `_superset_complex_promotion_refusal` is **byte-identical** to C-086's, so it
is reused rather than restated; the three no-touch functions
(`_apply_pathbank_unknown_enzyme_fallback`, `_apply_pathbank_unknown_complex_fallback`,
`_rewrite_reaction_protein_enzymes_to_complexes`) are byte-identical base to tip; the control set
holds by value on the artifact; G9 is `6 failed / 5 passed` at base → `11 passed` at tip with
behavioural assertion text; the five base passes are genuine preservation controls; the branch-only
mutant is faithful; SMOKE **473**; the affected set **182**.

**It improved on the author's method in one place.** Rather than accept the in-tree revert, it built
its own base comparator as a **real git worktree** at `0128fa6` — because `test_c075…` shells out to
`git ls-files` and a `git archive` export has no `.git`. Identical `4 failed / 570 passed` with the
identical four node ids at both ends. **Zero blast radius from C-095**, established on a comparator
that could actually run the corpus tests.

## F1 · HIGH · the guard fires on results that confer no identity, and writes a false refusal

Two reproductions, both the reviewer's own.

**A — no database.** `_map_complex_with_strategy:5892` sets the result's components to the row's
*own* components on `db_unavailable`; `:9398-9401` reconciles and **renames** them; the guard then
refuses the row against itself:

```
base:  chosen_rule ''  resolution {"status":"unresolved","issue":"db_unavailable"}  refused_superset null
tip :  resolution {"status":"novel","issue":"generated_wrapper_superset_complex_identity_refused"}
       refused_superset {"protein":"EntF","protein_complex":"",
                         "uncovered_components":["enterobactin synthase component F"]}
```

The "uncovered injected catalyst" **is the wrapper's own protein**, and the record names no complex.

**B — with the resolver, on the card's own § 4 preservation case.** `EntE complex`, whose payload
protein is canonically `2,3-dihydroxybenzoate-AMP ligase` with `EntE` as a synonym:

```
tip: refused_superset {"protein":"EntE",
                       "protein_complex":"ferric enterobactin outer membrane transport complex",
                       "uncovered_components":["2,3-dihydroxybenzoate-AMP ligase"]}
```

The record asserts the row refused a **named PathBank complex it never matched** — `candidates[0]` of
a **ten-way ambiguous** lookup, the one the resolver explicitly declined to choose. The
`candidates[0].name` fallback at `:6432-6438` is right for a `matched` result and wrong by
construction for an `ambiguous` one. **That is exactly the dishonest audit trail
`_wrapper_identity_refused_result`'s own docstring says it exists to avoid.**

**Root cause, and it falsifies a docstring in the diff.** `map_ids.py:6371` claims *"the two seams
cannot disagree about who is who."* They can. `_reconcile_components_against_local_proteins` matches
by uniprot, `pathbank_protein_id`, canonical name **and alias/synonym** and then renames;
`_declared_membership_identity_scope:6387` folds payload rows in **by canonical name only**.

**Nothing is dropped and no id is wrongly conferred** — merge rule 7 holds, no biological gate is
weakened. The damage is a false, biologically-worded provenance claim on rows the guard has no
business touching, plus silent metric drift from `protein_complexes_ambiguous` into `novel` +
`skipped`.

## F2 · MEDIUM · the guard key is broader than the charter binds

C-095 § 5 binds the key to `generation_reason == "single_protein_pathwhiz_wrapper"`. The code keys on
`is_generated_complex_wrapper`, true for **any** `generated is True`. `process_normalizer.py:4999-5005`
emits a second kind — `complex_named_source_entity_wrapper`, created *because the source text named a
complex* — which is closer to the declared row § 5 says must **not** be refused. Not established as a
live regression; the divergence from the charter was not disclosed.

## F3 · LOW · seven of ten G11 reports point at pin-verdict paths that do not exist

All 10 verdicts are valid (`refused: false`, `violations: []`, `foreign_src_entries: []`), but they
sit at `evidence/c095_*.pin.json` rather than `evidence/g11/pin/C-095/`, and reports `01`–`10` record
`--pin-verdict` paths inside `g11/C-095/` that no longer resolve because the files moved afterwards.
**The audit trail's own pointer is dangling.**

## Disposition

**C-095 is HELD pending correction round 1 of 2.** Dispatched: gate the guard on a result that
actually confers an identity; make the two seams agree on identity keys; regression-test both
proofs; narrow the key to the charter's or justify the breadth; relocate the pin verdicts and make
their recorded paths resolve; correct the two false claims in the report.

**F1 is the fourth time this wave a guard has been demonstrated against a case that could not
exercise it, or has fired on one it should not have.** The pattern is now well enough evidenced to
state plainly: *a refusal record is a claim, and a claim needs the same proof as the behaviour it
describes.*

---

# C-095 correction round 1 — `b9b6901`. REV-095's F2 was a **live** regression, and the author proved it against its own card.

All four F1 corrections applied, F2 narrowed, F3 relocated, F4 left. Under re-review.

## F1 — the guard now fires only where an identity is conferred

New `_result_confers_complex_identity(result)`; the call at `:9412` is gated on `status == "mapped"`
**or** a non-zero `pathbank_complex_id`/`pathbank_protein_complex_id`. The id test sits beside the
status test because a result carrying an id confers one whatever it calls its status.

**The coverage claim was verified rather than accepted.** Both branch-3 tests pass unchanged
(`22-rev-tip3`, 14 passed), and the author established independently why the gate reopens nothing:
if the row's components are empty — the only way to reach branch 3 — then every non-`mapped` return
(`:2476`, `:2496`, `:2519`, `:2541`, `5892`) has set `components = input_components`, so the
result's are empty too and **branch 3 does not execute at all**.

## The root cause is closed, and a sibling instance was flagged rather than silently edited

`_declared_membership_identity_scope` now reconciles the declared components through the **same call**
the loop applies to the result's, then looks payload rows up by reconciliation's own keys. The false
docstring at `:6371` is replaced by one that explains why the seams cannot disagree *and* names the
bug it caused.

**The identical false sentence also appears at `:6169`, inside C-086's `_enzyme_actor_identity_tokens`,
where it is imprecise for the same reason** — reconciliation matches aliases and that function's
comment says it cannot. The author **did not edit another card's function**, and flagged it instead.
Registered here as **F-138 (LOW)**: a docstring in `_enzyme_actor_identity_tokens` overstates the
agreement between the two identity seams. Not chargeable to C-086, whose behaviour is unaffected.

## F2 was NOT theoretical — the broader key was stripping real complexes

REV-095 could not establish that a `complex_named_source_entity_wrapper` row re-enters `map_payload`
and correctly declined to call it a live regression. **It is one.** Those rows are emitted with
`"components": []` — exactly the shape that routes to the third assignment branch — so at `194d6cd` a
source-named complex was refused and stripped of its id **and all four of its components**:

```
a source-named complex wrapper was refused:
  {'protein': '', 'protein_complex': 'enterobactin synthase',
   'pathbank_protein_complex_id': 3623, 'component_count': 4,
   'uncovered_components': ['EntB','EntD','EntF','EntE'],
   'declared_component_count': 0}
```

That is § 5's *declared* row — a biological claim, created because the source text named a complex —
being refused by a guard chartered to protect it. The key now requires
`generation_reason == "single_protein_pathwhiz_wrapper"` when a reason is present; a reason-less row
stays in scope as the legacy `novel_enzyme_single_component_complex` marker. **That carve-out is the
open question for re-review.**

The author explicitly declined to argue for the broader key and withdrew its earlier framing: it had
reported the breadth as a deliberate choice with *"bounded"* preservation cost. **The cost was not
bounded.**

## A vacuity failure the author recorded rather than replaced

Its first version of the two F1 regression tests used declared components carrying
`pathbank_protein_id` and `uniprot`. **They passed at `194d6cd`** — vacuous, because an
accession-bearing component is covered by that token whatever reconciliation renames it to. It caught
this itself, added `_bare_component()`, and **committed the vacuous run's report
(`15-rev-nonvacuity-194d6cd`, 13 passed) rather than overwriting it**. The corrected three fail at
`194d6cd` (`23-rev-nonvacuity3`, **3 failed / 11 passed**) and pass at tip, reproducing both REV-095
proofs byte-for-byte.

**This is the second time this wave an author has committed its own wrong measurement beside the right
one** (C-098b was the first). It is the behaviour to want: a quietly corrected probe leaves the record
un-auditable, and in both cases the error was one an orchestrator had already acted on.

## Two claims withdrawn by the author, both false as measured

1. *"The guard is a strict no-op in DB-less tests."* False — `db_unavailable` echoes the row's own
   components back, reconciliation renames them, and the guard fired and wrote a false record. It is
   a no-op **now**, by the F1 gate, not by accident.
2. *"The two seams cannot disagree about who is who."* False — they disagreed on exactly the alias
   case.

## Delta against the committed numbers

| Run | `194d6cd` | `b9b6901` | Δ |
|---|---|---|---|
| focused file | 11 passed | **14 passed** | +3 correction tests |
| G9 base `0128fa6`, in-tree A/B | 6 failed / 5 passed | **6 failed / 8 passed** | same six behavioural; the three new tests pass at base, correctly — base has no guard to write a false record |
| affected set | 182 passed | **182 passed** | 0 |
| blast radius | 4 failed / 570 passed | **4 failed / 570 passed** | 0, identical four pre-existing node ids |
| SMOKE | 473 passed | **473 passed** | 0 |

26 G11 reports, `check --task C-095` **0 non-compliant**, every job `FINAL SURVIVING COUNT : 0` /
`cleanup : success`.

## Ruling on the dangling pin pointers

Reports **04–13** record `--pin-verdict` paths at the pre-move location and no longer resolve.
**They are not re-run.** They are superseded by 17–26, which cover the same ground with resolving
pointers, and `TEST_MATRIX` § 0 is explicit that a successful expensive job is not re-run to repair
paperwork — the incident is recorded instead. This entry is that record.


---

# C-095 MERGED — `13b5696`. Gates discharged on the combined tree.

**REV-095 delta verdict: APPROVE.** Both round-1 findings fixed; the reviewer reproduced every number
itself **in genuine git worktrees rather than in-tree reverts**, plus five independent probes outside
the author's test file, and states plainly that nothing in the delta was taken on trust.

## Gates, measured on the integration branch after the merge

| Gate | Result |
|---|---|
| SMOKE (A+B+C, 20 files) | **473 passed**, 47.32 s, exit 0, survivors 0, cleanup success (`MERGE-095/01`) |
| C-095 affected set + the new file, **combined tree** | **196 passed** = 182 + 14, zero failures (`MERGE-095/02`) |

The 196 is the promise in `C-095.md`'s merge-order note kept: the card's numbers were taken against
`0128fa6`, and I undertook to re-run the affected suite on the combined tree. C-096 is merged in that
tree and touches `pwml/compound_resolution.py` / `prefreeze_resolution.py`, disjoint from
`map_ids.py`; **no interaction, measured rather than assumed.**

## What the reviewer verified from source rather than accepting

The load-bearing claim was that gating the guard on a result which actually confers an identity
reopens nothing. It tabulated **every** return that can produce `result` in this loop — eleven of
them — and established that none is simultaneously non-`mapped`, id-less, and carrying components
other than the row's own; and that branch 3 requires `complex_row["components"]` empty, which forces
`input_components` empty, so **branch 3 does not execute at all**.

It also checked that **neither clause of the gate is redundant**, which nobody had asked for:
`_complex_result_from_row` can return `status == "mapped"` with `cid == 0` — caught by the status
test — and a `use_cache=True` hit can return a stored result carrying an id under any status — caught
by the id test.

## F-138 · LOW · a C-086 docstring overstates the agreement between the two identity seams

`map_ids.py:6169`, inside C-086's `_enzyme_actor_identity_tokens`, carries the identical sentence
C-095 removed from `:6371` — that the two seams "cannot disagree about who is who". It is imprecise
for the same reason: `_reconcile_components_against_local_proteins` also matches aliases.

**C-095's author found this and did not edit another card's function.** The reviewer confirmed
`_enzyme_actor_identity_tokens` is untouched — no hunk falls in `6163-6172` — and called the
restraint correct. **Not chargeable to C-086, whose behaviour is unaffected.** Comment-only.

## F-139 · LOW · the carve-out comment justifies half of what the key admits

C-095's key requires `generation_reason == "single_protein_pathwhiz_wrapper"` **when a reason is
present**; a reason-less row stays in scope. The comment justifies that by the legacy
`novel_enzyme_single_component_complex` marker, and the reviewer confirmed that half: the marker is
emitted at exactly four sites (`:2671/2682`, `:6316/6337`, `:6593/6607`, `:6707/6722`) and **all four
are the single-protein wrapper**.

The comment does not name the *other* reason-less shape — `generated: True` with **no**
`generation_reason` — which also stays in scope and still produces the F2 harm at the merged tip.
**Reachability is nil:** no in-tree producer emits it, and such a row is already a runtime-schema
violation (`payload_models.py:713-735`, `generated_wrapper_reason_invalid`). Malformed input only,
and refusing a row whose kind cannot be determined is the conservative direction.

**Merged with this open**, as the reviewer graded it — requested, not required. The fix is one
sentence at `map_ids.py:6511-6515`.

## Correction to my own F3 ruling — the count was ten, not seven

Reports **04–13** record `--pin-verdict` paths that no longer resolve, not `04–10`: `11`–`13` also
dangle, because they pointed at the `evidence/c095_*` names the correction delta moved. My framing of
the range was right; the round-1 count of seven was low and the reviewer corrected it.

**Still not re-run**, and the reviewer agrees. One mitigating fact worth recording beside the
incident: the verdicts **kept their basenames** under `g11/pin/C-095/`, so `NN-<label>.json` →
`NN-<label>.pin.json` is recoverable by name. **No evidence is lost — only the recorded absolute path
is stale.**


---

# C-097 MERGED — `b35b6a2`. REV-097: **APPROVE**, and the review is stronger than the card.

Every number reproduced independently. **Taken on trust: only REV-089's historical 92-leg figure**,
which the reviewer's own 94-leg measurement supersedes.

## Gates

| Gate | Result |
|---|---|
| focused, tip | **16 passed** |
| G9, real git worktree at `ea688e0` | **4 failed, 12 passed** — behavioural |
| affected (charter's three + the author's seven) | **225 passed** |
| SMOKE, post-merge on the integration branch | **473 passed**, 40.59 s, survivors 0 (`MERGE-097/01`) |

## The census was verified on a strictly larger population than the card measured

The reviewer did not reuse the author's probe. Walking **every dict at every depth** across 94 legs:

```
Pass A   39,542 dicts        1,773 carry a truthy ref/id
Pass A'  of those 1,773, under /processes at any depth:   0
Pass B   500 tracked JSONs carrying "processes", 71,004 dicts:  0 under /processes
Pass C   every bucket, every slot key under /processes, 1,753 rows:  0 legacy, 0 moving
```

All 1,773 legacy-key dicts live in `/entities/*/enrichment/…` and
`/entities/*/mapping_meta/candidates/*`, which `_names` never reaches — the reviewer read **all 13
call sites** and confirmed each takes a slot off a process row, and that `_declared_names` reads
entity rows directly without calling `_names`. **Pass C covers 1,753 rows across every bucket
including `interactions`, against the author's 1,449 restricted to `PARTICIPANT_SLOTS`.** Both zero.
The *by construction* claim holds and is stronger than it was made.

## The zero is a measurement, proved without in-process emulation

The author's control rebound the constant in-process. The reviewer replaced it with a **four-cell,
two-process A/B** — one process importing `bench.semantic` from the base tree, one from the tip —
scoring all 94 legs through the real `_orphaned_references` and `_connected_core`:

| cell | digest | orphans |
|---|---|---|
| tip, corpus as committed | `4c014b7d…` | 6 |
| base, corpus as committed | `4c014b7d…` | 6 |
| tip, one synthetic `{"id": …}` injected per leg | `4c014b7d…` | 6 |
| **base, same injection** | **`ab88e8e6…`** | **100** |

Byte-identical unmodified; **all 94 legs differ under injection, 6 → 100 orphans.** Had one
legacy-key participant existed anywhere in the corpus, this would have caught it.

**And it matters for the live gate, not only the benchmark:** `semantic_production.py:638,642` calls
those exact two functions.

## The no-new-key-list guard was verified by building the impostor

A hard-coded, *correct*, eight-key private tuple in `_names` **satisfies all 15 behavioural
assertions** and is caught **only** by `test_names_is_a_derived_view_of_the_shared_schema_constant`
(`1 failed, 15 passed`). That is what a non-vacuous guard looks like.

## G6 direction, stated rather than glossed

The narrowing moves two gates **in opposite directions**: priority 5's `_connected_core` gets
**stricter**; priority 3's `_orphaned_references` gets **more permissive in principle**. But the names
removed are a CURIE and a JSON pointer, which the entity-name registry can never match — they were
false positives. Both directions measured **exactly 0** across 94/94 legs. No gate weakened.

## The disclosed restore incident — settled by re-running, not by argument

The author disclosed that its restore trap (`git checkout -- <file>`) reverted the fix mid-session
because base == HEAD. **The reviewer could not settle from the artifacts** whether the author's runs
`03` and `04` — both timestamped after the revert — were measured on the base or the re-applied file,
because **pin verdicts record `cwd`, `sys.path`, `t2pw_file` and `selection_paths` but no file content
hash.** It said so plainly and **declined to rely on them**, re-running both itself instead
(225 / 473). It also proved the committed census *could not* have been produced against a reverted
file: the census emulates base by rebinding, which is a no-op on the base file, so
`discriminates` would read **false**; the committed artifact reads `true`.

### F-140 · LOW · a pin verdict cannot prove which source it measured

`pinned_pytest`'s verdict records the resolved tree and selection but **no hash of the source under
test**, so a run cannot be attributed to a specific file state after the fact. Latent everywhere; it
became visible here only because an author disclosed an incident that made the question worth asking.
**The fix is a source-hash field in the verdict — a separate card, not a C-097 correction.**

## Correction to the ledger, caught by the reviewer

**`LEDGER.md:5162` § "`test_c074_strict_core_floor.py` is RED ON THE INTEGRATION BRANCH, right now"
is STALE and is hereby closed.** The reviewer ran it alone at base `ea688e0`: **31 passed**. Green at
base *and* tip. **C-092 (`c2cdb82`) closed it**, and rewrote its pins from equalities to properties —
which is why the count moved from 24 tests to 31. **Nothing to do with C-097.**

## Findings, all LOW, none blocking

* **Observation, not a defect** — a *future* payload carrying a human-readable name under an `id` key
  would escape priority-3 orphan detection. Zero occurrences in 94/94 legs and in all 500
  `processes`-carrying tracked JSONs; `id` is a declared field on no participant model;
  `identity_admission` still reads it. **This is F-131's own LOW residual**, and the direction
  `participant_schema.py:98-100` explicitly rules correct.
* **Process** — the author's `03-c097-affected` selection omitted two charter-named files. Both are
  inside SMOKE and passed there, and the reviewer ran them explicitly in its 225. Effectively covered.
* **Process** — no report markdown committed with the branch; the report was returned as text.
  Cosmetic, no gate affected.

## Housekeeping

`C:/t/c097base` — the reviewer's temporary base worktree — remains on disk; `git worktree remove` was
denied by the permission system in its session. Carries no accepted work; safe to remove.


---

# O-1 RULED — the corrected record · 2026-08-27 · tip `5d3c119`

**D-070 and D-071 are LOCKED.** O-1 is closed and no longer blocks anything. This section is the
controlling record for the facts around it; where an earlier report in this ledger disagrees, **this
supersedes it.** Historical reports are left unedited on purpose — corrections are additive here,
never a rewrite of what an earlier session actually claimed.

## Confirmed by measurement

* The pinned 21 partitions **exactly 16 / 5** — 16 generated functional wrappers in
  `entities.protein_complexes`, 5 PathBank `Unknown` sentinel rows in `entities.proteins`.
* The partition is **exhaustive and mutually exclusive**; overlap **0**.
* The 24 pinned stripped/withheld candidate identities are **entirely outside** the 21.
* The corpus-wide count of that population is **82**.
* **P22557** is withheld in **two** measured cases, through the species check.
* **EntD, EntE and EntF** suffer identity loss in **research** legs.
* EntD/EntE/EntF **separately** suffered superset-complex inheritance in later **strict** legs.
  **These are different defects on different legs** and were previously conflated.
* **PMC12444477**'s Boolean tolerance contradicts its own stated per-entity rationale.
* The false-Arabidopsis wrapper clobber is **orthogonal to placeholder status**.
* Of the six currently measured species-bearing wrappers, **two** are in the pinned 21, and **both**
  carry `source: explicit_entity_species`.

## Refuted — struck from all prospective rulings and summaries

* **"There are four genuine-loss cases."** There are not. The population is 24 pinned / 82
  corpus-wide, it sits on a different seam, and it is F-141.
* **"EC 1.3.1.28 is dropped."** **It is not.** EC 1.3.1.28 is **present in both the research and the
  strict payloads of PMC12096016**. The dropped-EC claim is struck. No card may be justified by it
  and no summary may repeat it.
* **"EntD/EntE/EntF loss and superset inheritance are one conflated event."** Two events, two legs,
  two defects.

### F-141 · MED · a candidate identity survives the identity verdict and is not shipped, and nothing counts it

**Registered 2026-08-27 from `evidence/g11/ORCH-710/04`. This is NOT
`placeholder_backed_proteins` and must never be reported under that name** — D-070 § O-1c.

A candidate identity exists in mapping metadata or in `identity_verdict.identity`, and the shipped
row does not carry it. **24 rows in the pinned run, 82 corpus-wide, 0 of them inside the placeholder-
backed 21.** The pinned 24 is the primary reproducible set.

**It is not a defect count.** The measured breakdown is already known to contain at least two
mechanisms with opposite dispositions:

| Sub-population | n (pinned) | Disposition |
|---|---:|---|
| species remains unknown | **22** | withholding is **contractually correct** — do not ship |
| no candidate describes the shipped identifier (**Fur**, ×2) | **2** | withholding correct; the candidate is not this entity |
| **P22557**, withheld through the species check | 2 occurrences within the above | correct unless species evidence was available and discarded |

Every row must be classified as **exactly one** of: source-supported species available but
discarded · species genuinely unresolved so withholding was correct · conflicting species evidence ·
candidate failed to describe the shipped identifier · identity or species evidence lost across a
stage boundary · other measured mechanism.

**Classify before fixing.** A production correction is authorised **only** for a subclass where
source-supported species evidence exists, the candidate identity matches the entity, the evidence is
lost across a deterministic seam, and restoring it **infers neither species nor identity**. Where the
classification shows multiple seams, they get **narrow disjoint cards** — not one broad repair.
**Do not ship identifiers for genuinely species-unknown rows**, and do not penalise the pipeline
twice for safely withholding an identifier the contract forbids it to ship.

## Cards chartered off this ruling

| Card | Branch | What | Gate |
|---|---|---|---|
| **C-099** | `card/C-099-species-preservation` | preserve already-resolved, source-supported species when an Unknown-backed functional wrapper is built | focused + SMOKE + the O-1 baseline test unmoved |
| **C-100** | `card/C-100-tolerance-scope` | per-entity tolerance scope for `unknown_backed_proteins_acceptable`; gold schema + parser + scorer | full base/tip gold A/B, 14 SMOKE-missed gold readers |
| **C-101** | `card/C-101-o1-metric-split` | the 16/5 split in the acceptance instrument; F-141 registered as its own metric | focused + SMOKE, offline re-score |

**`card/C-094-f134` is NOT relabelled and NOT merged.** C-099 is a new card with a new charter; C-094
remains on disk as diagnostic evidence of the clobber and of the three sites that carry it.

---

# F-141 CLASSIFIED — all 24 pinned rows are correct withholding · ORCH-711 · tip `f7dc223`

D-070 § O-1c required the pinned 24 be classified before any fix. It is done, and the answer is
**no production correction is authorised**. Evidence: `evidence/g11/ORCH-711/01`–`06`, and the probe
sources and logs now committed beside them.

## First, a correction to the record: which run is "the pinned run"

**It is `runs/2026-08-02_2130`, not `runs_verify/2026-08-24_1428`.** The first classification pass of
this session scanned the latter — the T-106 10-paper/20-leg run — found **5** withheld rows where the
ruling says 24, and was about to report the ruling's denominator as unreproducible. It was wrong.
`orch710_probeD_stripped_identity.py:26` pins `PINNED = "runs/2026-08-02_2130"`, a different tree
entirely, and against that tree the count is **exactly 24**, corpus-wide **exactly 82**.

**The measurement was always right; the assumption about which artifacts it named was not.** Recorded
because the next reader will make the same assumption: `runs/` and `runs_verify/` are both live, both
carry `papers/*/*/final_mapped.json`, and nothing in the O-1 packet says which one it meant.

### The reproducibility gap that made this possible, now closed

`bounded_run.py`'s JSON report carries **no child stdout** by design, so the six ORCH-710 artifacts
certify that the jobs ran bounded and clean while preserving **nothing about what they measured**.
The probe sources lived in a session scratchpad under `AppData/Local/Temp`, outside the repository.
They survived only by luck.

All five ORCH-710 probes and their logs are now committed in `evidence/` beside the reports —
`orch710_probeA_pinned21.{py,log}` through `orch710_probeE_gold_tolerance.{py,log}`, plus
`orch710_pinned21.json`, the pointer file the 16/5 partition was computed against. **A certificate
that a job was clean is not a record of what the job found.** This is F-140's class, one level up.

## The classification — measured, per row

| Class | pinned | corpus |
|---|---:|---:|
| no candidate record at all, so the rung could not compare — **withholding correct** | **22** | 23 |
| candidate does not describe the shipped identifier (**both Fur rows**) — **withholding correct** | **2** | 13 |
| candidate carries no species — **withholding correct** | 0 | 20 |
| conflicting species evidence | 0 | 3 |
| other measured mechanism | 0 | 23 |
| **source-supported species available but discarded** | **0** | **0** |
| **entity species silent (a propagation loss)** | **0** | **0** |
| **both sides present and the rung still unknown** | **0** | **0** |
| **total** | **24** | **82** |

Every one of the 24 ships **nothing** (`shipped=(none)`), carries `identity_status=unresolved`, and
records `reason=identity_evidence_missing`. **None forged an identity.** Five papers:
PMC12096016, PMC12180156, PMC12452463, PMC12856317, PMC13231680. **0 of 24 are inside the pinned 21**,
confirming D-070 § O-1c on the same artifacts that produced it.

## The inference that does NOT hold, and why it nearly became a card

An intermediate pass of this classification labelled **22 rows**
`species_evidence_lost_across_stage_boundary` — on the ground that each row carries
`species_ref.source = explicit_entity_species`, `status = matched`, *Escherichia coli* or *Homo
sapiens*, `taxonomy_id` 562/9606, **confidence 1.0**, while the ladder's `species` rung reads
`unknown`. That looks exactly like resolved evidence being dropped at a seam, and § 10 authorises a
production fix for precisely that subclass. **It is wrong**, and the superseded probe and its log are
committed as `orch711_f141_species_seam.SUPERSEDED.{py,log}` rather than deleted, per D-025.

`_candidate_species_verdict` (`map_ids.py:4828`) returns `unknown` when **either** side is silent:

```python
if not requested: return "unknown"      # the ENTITY side
...
if not declared:  return "unknown"      # the CANDIDATE side
```

The entity carrying a species therefore proves nothing on its own. The decisive measurement is
`identity_verdict.organism`, which is what the ladder actually received: on **all 24 rows it is
non-empty** — `Escherichia coli` on **18**, `Homo sapiens` on **6**. `requested` was never silent, so
`unknown` **can only have come from the candidate side**. The entity's species reached the ladder
intact; there was no candidate record carrying a species to compare it against.

**This holds regardless of whether the judged candidate is persisted in the artifact.** The proof is
the rung's own value combined with a non-empty `requested`, not the absence of a stored candidate —
which matters, because `candidate_evidence: ok` is recorded on 22 of these rows while
`mapping_meta.candidates` is empty and `judged_candidate` is `{}`. That the artifact cannot show
*what* was judged is **F-140's defect, not a new one**, and it is why the classification was
deliberately built on a field the artifact does preserve.

## Disposition

**No card. No production change.** § 10's authorised subclass requires all four of: source-supported
species evidence exists · the candidate identity matches the entity · the evidence is lost across a
deterministic seam · restoring it infers neither species nor identity. On the pinned 24 the **second
and third fail on every row** — there is no candidate evidence to have lost. Shipping P0AEJ2 for
EntC because the entity says *E. coli* and the accession "is" *E. coli* EntC would be **inferring an
identity the pipeline never verified**, which is the exact behaviour `PRODUCT_CONTRACT` § 8 and
D-070 forbid. The fail-closed ladder is working.

**F-141 stays OPEN as a measurement obligation, not a defect.** What remains genuinely unexplained is
the corpus-wide **23 `other_measured_mechanism`** rows, whose species rung is neither `ok`,
`unknown`, `mismatch` nor absent. They are outside the pinned set, they are not blocking, and they
are the only part of the 82 that could still hide something. A future card may classify them; none is
chartered now, because nothing measured says one is needed.

**F-141 is not, and must never be reported as, `placeholder_backed_proteins`.**

## F-141 addendum — three challenges from peer review, all answered from evidence

Raised after the classification landed, by the two other sessions on this tree. **None overturns
"no card follows"; two sharpen what it rests on and one caught an arithmetic error of mine.**

### Correction: the organism split is 18 / 6, not 19 / 5

The entry above originally read *"`Escherichia coli` on 19, `Homo sapiens` on 5"*. **It is 18 and 6**
— PMC12096016/research 9 + PMC12452463/research 8 + PMC13231680/strict 1 = **18** *E. coli*;
PMC12180156/research 5 + PMC12856317/research 1 = **6** *H. sapiens*. Counted from
`orch711_f141_which_side.log`, corrected in place. The total, the conclusion and every other figure
are unaffected — but a measurement claim that is wrong is wrong, and I had eyeballed it instead of
counting it.

### Challenge 1 — "are all 24 `unknown`, or are some `mismatch`?"

A fair challenge: `mismatch` is a **different** correct reason — two species that both spoke and
disagreed — and the recorded mechanism covers `unknown` only. If any of the 24 were `mismatch`, the
conclusion would survive but the stated reasoning would not cover it.

Counted from the committed `orch710_probeD_stripped_identity.log`:

| species rung | rows |
|---|---:|
| `unknown` | **22** |
| `mismatch` | **0** |
| `ok` / `genus_level` | **0** |
| rung never reached — `candidate_evidence` returned `no_candidate_describes_the_shipped_identifier` | **2** |

**The mechanism is uniform.** No row is `mismatch`. The two Fur rows never reach the species rung at
all, which is why they classify separately. *"All 24 are correct withholding **for the reason
stated**"* — the load-bearing form of the claim — holds.

### Challenge 2 — "is the requested organism itself missing on any leg?"

Sharper than it looks. `_candidate_species_verdict`'s **first** `unknown` path tests the *requested*
argument, not the candidate:

```python
if not requested: return "unknown"
```

A leg reaching that with an empty requested organism would withhold **every** identity on the leg,
each row classifying as "correct withholding" while the real cause is a missing organism upstream —
a materially different finding wearing the same label.

Measured: `requested=(SILENT)` occurs on **0 of 24**. All 24 carry a requested organism (18 + 6
above). **The first path is not in play**, so the silence is the candidate's on every row.

### Challenge 3 — "`PRODUCT_CONTRACT` § 8 has a preservation half; is it implemented?"

§ 8's `unavailable` row requires **two** things, not one: *"accession preserved as `unverified_claim`;
not promoted; **not erased**"*. The report was that `grep unverified_claim src/` returns nothing, so
the carrier does not exist.

**The carrier exists.** The grep looked for the contract's prose token; the implementation names it
`unverified_identity_claim` — `entity_identity.py:147`, written at `map_ids.py:5555` into
`mapping_meta.unverified_identity_claim`, read at `map_ids.py:8468`, cited to **D-003**, and pinned by
`tests/test_identity_evidence_hydration.py`. Its docstring states the contract obligation verbatim:
*"not in `mapped_ids` (that would be promotion) and not only in `rejected_mapped_ids` (that would read
as a refutation)"*.

Measured behaviour under **current** code, `runs_verify/2026-08-24_1428`, 5 withheld rows:

| `verification_status` | rows | carrier | correct? |
|---|---:|---|---|
| `not_evaluated` | 2 | **present** | yes — preserved, not promoted, not erased |
| `rejected` | 3 | **absent** | yes — § 8 makes `rejected` *"the only case where identifiers may be stripped"* |

**Both halves of § 8 are honoured, including the half that must NOT fire.**

On the pinned 24 the carrier is absent and `verification_status` is **absent entirely** — the field
does not exist on those rows. `runs/2026-08-02_2130` **predates D-003's implementation**, so that
absence is the age of the artifact, not a defect in the code. It is also a standing caution about the
pinned set: **it is a 2026-08-02 artifact tree, and a property measured on it is a property of code
as it was then.** The 24/82 population figures are unaffected — the withholding criterion reads only
fields that existed — but any *behavioural* claim drawn from that run needs re-measuring against a
current one before it is acted on.

### Two lessons worth more than the finding they came from

**The pinned run is safe as a POPULATION and unsafe as a BEHAVIOUR.** `runs/2026-08-02_2130`
predates D-003, so `verification_status` and `unverified_identity_claim` do not exist on those rows
at all. Counting *what is in it* is sound — the withholding criterion reads only fields that existed
then, so 24 and 82 stand. Concluding *what the pipeline does* from it is not, because the code that
produced it is a month stale. Any behavioural claim drawn from that tree gets re-measured against a
current run before it is believed. The O-1 packet drew a population from it and stopped there, which
was the right instinct whether or not it was deliberate.

**A zero-hit grep on a document's vocabulary is not evidence that a mechanism is missing.**
`PRODUCT_CONTRACT.md:244` calls it `unverified_claim`; the code calls it `unverified_identity_claim`.
`grep -rn unverified_claim src/` returns nothing and means nothing. Ask what the **code** calls the
concept before reporting it absent — and prefer a behavioural probe over a name search, which is what
settled this one. The same session made the mirror-image error in the same wave: citing a denominator
that lived only in prose. Both are the same mistake about what counts as evidence.

**And the probe that settles a two-sided obligation must test the side that must NOT fire.** Checking
the carrier is *present* on `not_evaluated` would have passed while a carrier wrongly firing on
`rejected` — the actively dangerous behaviour, since § 8 makes `rejected` the only case where
identifiers may be stripped — went unnoticed. It is absent on all three. That is what makes this a
result rather than a presence check.

**Disposition unchanged: no card.** The challenges strengthen the conclusion rather than weakening
it — the promotion half is correct, the preservation half is implemented and fires exactly where the
contract says, and the mechanism behind all 24 is uniform and measured.


---

# C-098a and C-098b RECONCILED against D-070 — both INVALIDATED, nothing salvaged

Required by the O-1 ruling: inspect the held C-098 work, identify the O-1 premise each assumed, and
decide whether the 16/5 split authorises, narrows or invalidates it. **It invalidates both.** Neither
merges, and no hunk of either is independently justified today.

## They are not standalone branches

Both are **stacked on C-094**, which does not merge:

| Branch | Base | Carries | Its own incremental work |
|---|---|---|---|
| `card/C-098a-cap` | `14121d5` | all of C-094 (`map_ids.py` +97, its 755-line test file) | `release_status.py` +129, `driver.py` +30, `test_c098a_gate_review_cap.py` +365 |
| `card/C-098b-gate` | `14121d5` | all of C-094 **and** all of C-098a | `process_normalizer.py` +204, `test_c098b_...py` +823 |

So neither could merge on its own terms even before the ruling: the merge rule 1 dependency is a
branch that is refused.

## The premise, in the authors' own words

C-098a's docstring states it plainly (`release_status.py`,
`cap_release_for_unresolved_placeholder_species`):

> *"WHY A CAP EXISTS AT ALL, when the gate could simply have kept blocking. **C-094 stops the
> PathBank `Unknown` sentinel lending its own *Arabidopsis thaliana* to the wrapper built around it.
> On the majority wrapper shape nothing else resolved a species, so the wrapper now carries none**,
> and the strict gate's 'Generated protein complex is missing species/organism' rule — which has only
> a blocking channel — would turn a leg that produced a PWML into one that produces none."*

C-098b is the other half of the same mechanism: it defines
`PLACEHOLDER_SPECIES_UNRESOLVED_ISSUE = "protein_complex_missing_species"`, adds a `review_findings`
channel at `REVIEW_SEVERITY = "review_required"`, and grades exactly that one finding review-grade
instead of blocking — so that C-098a's cap has an input.

**Both exist to absorb damage C-094 causes.**

## Why the ruling invalidates them

**C-099 causes no such damage.** It never removes a species; it only declines to *overwrite* one that
is already resolved and source-supported. A wrapper with nothing resolved underneath — the 25 of 31,
and 14 of the pinned 16 — keeps the sentinel species **exactly as today**. The set of wrappers
carrying no species is therefore **unchanged**, and `protein_complex_missing_species` **cannot newly
fire on this population**.

Consequences, in order:

1. C-098b's review-grade channel has **nothing to grade**;
2. C-098a's cap has **no input**, and is inert by its own acceptance criterion;
3. what remains of C-098b is a mechanism whose only effect is to convert a **blocking** biological
   gate finding into a non-blocking one.

Point 3 is why nothing is salvaged. Merge rule 6 forbids weakening a biological gate to increase PWML
production, and the only defence C-098b offered was merge rule 7 — that the demotion is required to
preserve an incomplete-but-correct pathway. **With C-094 gone that necessity is gone**, and a gate
demotion with no pathway to preserve is merge rule 6 with nothing on the other side of the scale.

Landing C-098a alone would be dead code: an inert cap, a forward-declared severity constant with no
producer, and a conditional test that skips until a branch that will not merge introduces the string
it waits for.

## Disposition

* **`card/C-094-f134` — NOT merged, NOT relabelled.** Diagnostic evidence of the clobber and of the
  three sites that carry it. Superseded for production purposes by **C-099**.
* **`card/C-098a-cap` — NOT merged. Invalidated by D-070.** No hunk salvaged.
* **`card/C-098b-gate` — NOT merged. Invalidated by D-070.** No hunk salvaged.
* **C-098c — remains REFUSED** and is not chartered. An export-time `default_species_id` fallback
  replaces a false *Arabidopsis* at mapping time with a false default species at export time: merge
  rule 8, and the defect recreated one stage later.

Branches and worktrees `C:/t/c094`, `C:/t/c098`, `C:/t/c098a`, `C:/t/c098b` stay on disk. They carry
real measurement work and are not pruned for tidiness.

**Card completion is not finding closure.** F-134 is addressed in production only to the extent
C-099 addresses it — the clobber where a resolved species exists. The 25 wrappers with no resolved
species keep the sentinel's *Arabidopsis*, which D-070 § O-1b rules is **not** a forged identity and
**not** to be resolved by deleting wrappers or blocking their serialization. Whether that
representation should change at all is a PathBank-representation question, not an agent's.

---

# T-107 READINESS — **NO-GO**, and this wave could never have changed that

Assessed 2026-08-27 at integration tip `8f342f4`, after D-070 and D-071.

## The priority table

| Priority | Current result | Reachable? | Remaining blocker | Expected T-107 result |
|---|---|---|---|---|
| **1** — zero false real identifiers | **FAIL, 6** | **NO** | F-127: no participant-provenance carrier and no entity evidence span, so the discriminating fact is never recorded. F-128: complying with D-069 would *restore* identity to 12 rows the gold's bucket-blind `forbidden_identifiers` then counts as false. **Neither is ruled.** | **FAIL at 6 or higher** |
| **2** — no unsupported retained reactions | `NOT EVALUATED` on 11/20 legs | **NO** | D-067 precondition 3 needs independent biological review. Unrelated to this wave. | `NOT EVALUATED` on most legs |
| **3** — referential integrity | **PASS** | yes | none | **PASS** |
| **4** — requested-core coverage | **FAIL, 0/8** | **NO** | F-132: 62 of 281 unmatched terms corpus-wide are gold-**forbidden**, so the metric penalises the pipeline for obeying the gold. Unruled. | unreadable — measures the instrument, not the pipeline |
| **5** — strict PWML export | **FAIL, 0/4** | **NO** | F-132 again. `PMC12782028/strict` is `blocked` partly for not matching `LIPA`/`LBR`/`SREBF1`/`SREBF2`, which the same gold forbids exporting. It is **`blocked`, not `correctly_blocked`** | unreadable, same cause |

## Gate conditions

| # | Condition | State at `8f342f4` |
|---|---|---|
| 1 | O-1 recorded and implemented consistently | **PARTIAL** — recorded (D-070). Implementation is C-101, which waits on C-100 |
| 2 | ORCH-710 evidence committed and certified | **MET** — `5d3c119`, 6 artifacts, 0 non-compliant, probes and logs committed too |
| 3 | 16/5 metric split enforced | **NOT MET** — C-101 chartered, not dispatched |
| 4 | species-clobber card merged | **IN FLIGHT** — C-099 |
| 5 | PMC12444477 scoped per-entity tolerance | **IN FLIGHT** — C-100 |
| 6 | gold A/B green but for predicted approved movers | **PENDING** — C-100 owns it |
| 7 | held C-098 work reconciled | **MET** — both invalidated, nothing salvaged |
| 8 | 24/82 no longer conflated with O-1 | **PARTIAL** — registered and classified as F-141; the metric itself is C-101 |
| 9 | **no absolute acceptance priority guaranteed to fail** | **NOT MET — and unreachable by any engineering in this sprint** |
| 10 | all deterministic gates green | **MET so far** — SMOKE 473 at `6effe58`; `test_protein_export_policy.py` 63 passed |
| 11 | integration pushed and remotely verified | **MET** — local = origin = `ls-remote` at every step |
| 12 | LM Studio and model healthy | **NOT CHECKED** — deliberately. Only relevant if 9 clears |
| 13 | heavy lock free | **MET** |
| 14 | zero sprint-owned Python | **MET** between jobs |

## Condition 9 is the whole story, and it is not ours to clear

`DECISION-BUNDLE-F132-PRIORITY1.md` § 9 already settled this, and the ruling that arrived does not
touch it:

> *"condition 1 … is the one that **no engineering in this sprint can clear**. **A does not clear
> it; only B does**, because B *is* the explicit acceptance."*

The bundle asks **two** things. The product owner has ruled the **addendum** — O-1 / F-135, now
D-070 — and PMC12444477, now D-071. **Asks A and B are still open:**

* **A** — reconcile the anchor set against `forbidden_identifiers`, so Priorities 1 and 4/5 stop
  scoring the same rows in opposite directions;
* **B** — accept, or decline, a **Priority-1 floor of 6** for T-107 purposes.

**Only B clears gate condition 1**, because B *is* the acceptance. Nothing in D-070 or D-071 states
a floor, and neither rules F-132 or the F-128 / D-069 conflict. D-071 rules a *different* instrument
defect — `unknown_backed_proteins_acceptable`'s missing scope — and reconciling the anchor and
forbidden sets is not in it.

**So T-107 stays NO-GO, and that is the correct outcome rather than a delay.** Running it now would
spend roughly seven hours to re-measure two priorities that grade the instrument instead of the
pipeline, against an absolute target already known to fail.

**Merging C-099, C-100 and C-101 will not change this**, and none of them was chartered to. C-092 and
C-093 were both worth merging and neither moved Priority 1; this wave is the same. C-099 corrects a
false *species*, not a false *identifier*, so it cannot move Priority 1's count in either direction.

## The one question that goes back to the product owner

This is § 17's permitted escalation — *an absolute acceptance target remains impossible after
implementing this ruling* — and not a routine question.

> **F-132 asks A and B are unanswered. Until B is answered, T-107 cannot be scheduled.** Please rule
> on A, and accept or decline the Priority-1 floor of 6.

Everything else in this wave proceeds without it.

---

# REV-099 — the seventh consecutive review round to find something real

**C-099**, `f7dc223..c932ea0`, reviewed by an independent peer session that did not write the code and
holds no edit tools on the branch. **APPROVE WITH REQUIRED CORRECTIONS** — three findings, all
accepted, one of which needed an orchestrator ruling rather than a code fix.

## What the review confirmed behaviourally, not by inspection

* **`tests/test_protein_export_policy.py` = 63 passed at the tip.** The O-1 statement
  `test_strict_gates_accept_a_correctly_formed_unknown_backed_complex` is green. This is the baseline
  C-094 inverted, and it is in **neither SMOKE nor Chunk D**. Run twice, in two different trees, by
  reviewer and orchestrator independently — the same number from two trees is a stronger statement
  than one run.
* **The boundary holds.** Three hunks: the helper, and the two owned `update()` sites. **Nothing at
  ~7419, ~8140 or ~8191** — the two sentinel-protein builders and
  `_apply_pathbank_unknown_complex_fallback` are untouched, so O-1a's five rows are unchanged and the
  `setdefault` C-094 broke is intact.
* **The non-vacuity case is genuine.** It monkeypatches `_wrapper_species_fields` to return the four
  sentinel fields unconditionally; because the call site is `**_wrapper_species_fields(complex_row)`
  *inside* the same `update()`, that reproduces the pre-C-099 statement **exactly** — the actual
  overwrite, not a neighbour. It then asserts the shipped contradiction reappears. **It exercises the
  guarded path**, which four earlier rounds of this sprint could not say of their guards.
* **The G9 base tree is real.** `pinned_pytest` refused `C:/t/c099base` with
  `SELECTION_OUTSIDE_EXPECTED_TREE` (the new test file does not exist there), so the author
  materialised `C:/t/c099g9` at `f7dc223` via `evidence/c045b_base_tree.py` — 5399 blobs exported,
  5399 re-hash-verified. The reviewer confirmed `map_ids.py` there hashes to blob `c21f7ae`,
  byte-identical to `git ls-tree f7dc223` **and** to the diff's own `index c21f7ae..` pre-image, and
  that the pin verdict records `expected_tree = C:\t\c099g9`, `refused=False`. `c099base` was never
  modified.

## Finding 1 — a ruling, not a defect: `single_pathway_species` is not evidence

The author's `_SOURCE_SUPPORTED_SPECIES_SOURCES` justified itself **by exclusion** — `gap_resolver_llm`
is an LLM inference, `novel_species` records absence. The reviewer applied the same reasoning to the
remaining member and it holds. `_single_pathway_species_hint` (`map_ids.py:3236`) returns a hint only
when the payload declares **exactly one** species, then applies it to a row that never stated its own.
That is **payload-scoped inference** — deterministic rather than an LLM's, but inference. The other
two members are **entity-scoped**.

**Ruled: removed.** The counter-argument is recorded in the charter amendment rather than discarded —
C-099 does not *apply* species, `hydrate_species_references` already did, so declining an overwrite is
arguably not "defaulting to the pathway organism". Ruled against because the charter forbids pathway-
dominant defaulting and this card must not become the precedent cited for it; because **both pinned
examples O-1 names carry `explicit_entity_species`**, so the narrower set loses nothing the ruling
asked for; and because § 4 says to leave uncovered wrappers *"unchanged pending their own evidence
classification"*, which is exactly this population.

**The denominator was confirmed independently before the ruling, not taken from the review.** 24
roots, **71** generated single-protein wrappers, **6** with a resolved species ref, **4**
`explicit_entity_species` / **2** `single_pathway_species`
(`evidence/orch711_wrapper_species_census.{py,log}`). Reviewer and orchestrator agree exactly. The
change moves **2 rows** back to today's behaviour; nothing regresses.

**The reviewer disclosed a prior** — it had proposed promoting `species_ref` into `species` earlier in
the sprint and been refused on this same ground — and flagged that it might be over-fitting the
correction. It was not: it argued from the function's construction rather than from the memory, and
supplied the measured split so the ruling could be weighed. **A previous refusal of the same class of
reasoning on the same ground is evidence, not bias.**

## Finding 2 — the reviewer corrected the author's disclosure *in the author's favour*, then required the change anyway

The author disclosed that `mapping_meta.species_preservation` is written *"on every wrapper row
carrying a species resolution record — including the 25 that have nothing resolved"*. **Measured, that
is not what the code does**: a row with nothing resolved has no `species_ref` and no
`species_resolution`, so the `if ref:` guard is false and no note is attached. Actual reach is **6 of
71**, and **0 of them are the TRAP-3 shape**. The orchestrator's boundary worry was accordingly
overblown.

Required anyway, and correctly: a row with a ref that proves unusable gets `decision:
placeholder_species_applied` with `contradictions: []` — **a note that surfaces nothing**, in gold-
and IR-visible serialized output. § 4 authorises surfacing contradictions; it does not authorise empty
ones. One line moved.

**This is the behaviour to want**: an author disclosing something against itself, and a reviewer
measuring the disclosure rather than accepting it — in the direction that made the author look
*better*, and then requiring the fix on its own merits.

## Finding 3 — an unstated exception to the card's own headline

`if disagree: return _placeholder(...)` means a wrapper **with** source-supported species can still
end up carrying *Arabidopsis*, when its own visible fields disagree with its resolution record.
Refusing to arbitrate there is right and is unchanged from base. But the card's headline is *"a
wrapper that already carries source-supported species keeps it"*, and the next reader will cite the
headline as covering a case it does not. **Documented, not changed.**

## Two charter amendments, because the spec was what moved

After Finding 1 the implementation would have been **deliberately narrower than its own charter**,
which still listed `single_pathway_species` — the shape a later session "fixes" back by comparing code
to spec. So `C-099.md` § 4 is amended in place (`e45bfdb`) with the ruling, its date, its reasoning
and its reversal condition; and the constant's comment names the charter line it departs from. **The
spec explains the code and the code names the spec.**

The same pass amended § 4's preservation paragraph, which was **outcome-correct and
mechanism-misleading**: only four keys were ever in the clobber's path, preservation is **by
omission**, and there is no code preserving `species_ref`/`taxonomy_id`/`species_name` — *and there
should not be*. Without that note the next reviewer spends a round looking for code that should not
exist.

## Orchestrator verification, independent of both author and reviewer

| Check | Result |
|---|---|
| `test_protein_export_policy.py` + the new file at the C-099 tip | **97 passed** |
| golden / freeze-seam surface — the new `mapping_meta` key lands in serialized payloads | **128 passed** across `test_batch_driver_seam_golden.py`, `test_c011_freeze_seam_golden_equivalence.py`, `test_baseline_regression_2026_07_28.py`, `test_c030a_object_sharing_at_the_freeze_seam.py`, `test_c073_identity_admission.py`, `test_c030_canonical_identity_fallback.py` |
| `git merge-tree` dry run against integration | clean, **0 conflicts** |

Checking the serialization surface **before** the note question was settled turned out to be the
right order: it meant Finding 2 only shrinks an already-safe footprint rather than being load-bearing
for safety.

## Disclosed-unmeasured

**Chunk D was not run** — it is not on the charter's gate list — so
`test_streamlit_stage8_export_contract.py`, `test_streamlit_stage2_orchestration.py` and
`test_streamlit_quarantine_boundary.py` are uncovered by this card. **F-136 already records that
Chunk D cannot go green in this environment with the DB up**, so the card is not held on it. Recorded
as disclosed-unmeasured rather than left for a reader to infer coverage.

**Environment:** `C:/t/c099` has no `.env` and no `.venv`; the author *probed* rather than assumed and
ran base and tip in the identical resolver-`None` state. Green there means **"no regression with the
resolver hidden"**. The orchestrator's own verification ran from the integration tree, which does have
`.env`, so the two together cover both states.

---

## The lesson of this wave, and it is one sentence rather than five

**Do not let an aggregate stand in for the thing being claimed.** Every finding in this wave is that
shape, and they were found by five different readers looking at five different artifacts:

| The aggregate | What it stood in for | How it broke |
|---|---|---|
| `placeholder_backed_proteins` = 21 | two populations with opposite answers | D-070: it is 16 + 5, and the question was unanswerable while they were one number |
| "the pinned run" | a named artifact tree | `runs/` and `runs_verify/` are both live; the same criterion gives 24 and 5 |
| a certified G11 report | what the job measured | the report carries no stdout; it proves a job was clean and preserves nothing it found |
| `grep unverified_claim src/` = 0 hits | whether a mechanism exists | the contract's prose token is not the code's identifier |
| `species: unknown` on 22 rows | which side of the comparison was silent | `_candidate_species_verdict` returns `unknown` for either side; the entity's species proved nothing |
| combined `97 passed` | two files, separately | a shift between them nets to zero; the reviewer ran them split |
| census "4 preserved" | live behaviour | the reconstruction agreed with itself by construction; the `disagree` branch could not fire |
| "reachable and did not fire" | a check that ran and agreed | an **absent** field produces no disagreement for the same reason a blind probe does |

**In every one of those rows the aggregate was TRUE.** The 97 was a real 97, the constructed 4 was a
real 4, the zero-hit grep really returned zero, the 22 rows really did read `unknown`. Not one was an
error in the number; each was an error about **what the number was evidence for**. That is why
re-running never caught any of them and re-reading the claim always did — and it is the practical
reason a merge gate built only on "do the tests pass" cannot find this class at all.

The last two are the same defect one level apart, found ten minutes apart, and the second was caught
**only** because the first had just been named. That is the argument for keeping an adversarial
reviewer on a card after it has already been approved once.

### The operational form of it, and the standard that follows

**An aggregate is not evidence until the instrument that produced it has been shown capable of
producing a different one.**

Every round of this wave applied the vacuity question to the **code** — does the guard fire, can it
fire, would it fire on a case that actually reaches it. C-099's author turned it on the
**instrument**: a probe that reports zero must first demonstrate it can report non-zero. Its
`CONTROL_SPECIES = "Xanthomonas campestris"` perturbs a real corpus row, proves
`row_species_fields_disagree_with_resolution` does raise, and makes the probe **exit 1 and refuse to
report any number** if that control ever fails.

That is the same question one level up, and it answers as a class what this sprint had been catching
one instance at a time. **A committed probe that reports a zero, an absence, or a "no occurrences"
carries a positive control, or its number is a ceiling and must be labelled one.** Four of the eight
rows in the table above would have been caught at birth by that rule.

### The same move in the other direction: how a requirement is written

**A design instruction is satisfied by intent. A test case is satisfied by behaviour.**

Item 2 of the C-100 pre-checks was first put to the author as a design instruction — *"an absent
per-entity scope must inherit the case Boolean, not default permissive"*. That form is satisfied by
an author who agrees with it, and it would have sailed past the one construction most likely to
break it:

```python
scope.get(name, True)        # a permissive default: OVERRIDES nine explicit False values
name in scope                # permissive by OMISSION -- and it has no default to inspect
scope.get(name, case.unknown_backed_proteins_acceptable)   # the safe one
```

The middle form reads as obviously correct, is the natural way to write a membership test, and
**contains no default for a reviewer to examine**. The instruction cannot catch it. What catches it is
one test: **an entity deliberately left OUT of the scope, on a case whose Boolean is `False`,
asserting it stays strict.** A scope exercised only with entities that are *in* it never reaches the
absent branch at all — the same shape that has now cost this sprint four rounds.

### The limit of the positive-control standard, found the day after it was written

**A control proves the instrument can report non-zero. It does not prove the instrument is asking the
right question.**

The orchestrator's `lipoprotein` probe reported `Lpp` = **0** across 37 archived files **with a
passing control** — `LpxC` = 1726 in the same texts, so the corpus was live and the matching worked.
By the standard written the previous day that zero was trustworthy. **It was wrong.** The pattern was
case-sensitive `Lpp` and the token in the paper is lowercase `lpp`. The control was sound and
irrelevant: it proved the probe could find *a* token, not that it could find *this* one.

So the standard needs its second half. A control answers *"could this instrument report a different
answer?"* It cannot answer *"is this instrument looking for the right thing?"* — and a probe can pass
the first while failing the second completely. Where a null result is load-bearing, the control should
exercise **the same predicate** on a case known to match it, not merely a different predicate on the
same corpus.

**This and the positive-control standard are one move**: refusing to accept a claim in a form that
cannot fail. Together they are the whole of what this wave learned about evidence — an instrument
must be able to report the other answer, and a requirement must be able to fail.

**Nobody in this wave got it right first time, including the orchestrator.** The corrections that
mattered were each made by someone other than the author of the claim, and in three cases the author
had already disclosed the weakness themselves and been measured anyway. That is the process working,
not failing.

---

## C-099 MERGED — `9e4a28a` · integration `233b26a`

**F-134's production correction, narrowed by D-070 and merged after eight review rounds.**

**Gate:** SMOKE **473** + `test_protein_export_policy.py` **63** + the card's own **34**, run together
on the integration tip = **570 passed**, zero survivors, heavy lock acquired and released
(`evidence/g11/MERGE-099/01`). The export-policy file went into the same selection rather than being
trusted from the card — it is in **neither SMOKE nor Chunk D** and carries the O-1 statement C-094
inverted.

**G9:** base **16 failed / 320 passed**, tip **336 passed**, failing on an observed value
(`assert 'Arabidopsis thaliana' == 'Escherichia coli'`), not a missing symbol. Counts moved from the
first round's 20/316 exactly as predicted when `single_pathway_species` came out of the set; the four
tests that stopped failing at base became **reversal guards** rather than G9 proofs — re-add the
string and they fail.

**Boundary:** three hunks throughout, in `_apply_pathbank_unknown_enzyme_fallback` only. The two
sentinel-protein builders (O-1a) and `_apply_pathbank_unknown_complex_fallback`'s preserving
`setdefault` (which C-094 broke) are untouched. 114 hand-authored lines of a 120 ceiling; 18 generated
artifacts of 25; 9 G11 artifacts, 0 non-compliant.

**Measured effect, through the shipped constant rather than source labels:** 71 generated
single-protein wrappers corpus-wide, 6 with a resolved species ref, **4 preserved** — both pinned
examples (enterobactin synthase complex, ALAS2 homodimer) among them — and 2 returned to today's
behaviour by the Finding-1 ruling. **Nothing regresses**: every row that is not preserved keeps the
sentinel species exactly as at base.

### The round that mattered most was about the probe, not the code

REV-099's delta review found that the preservation census **agreed with itself by construction**: it
rebuilt each row as `{"name": …}` stamped from its own ref, so every field the consistency check reads
either came from that ref or was absent. **The `disagree` branch could not fire**, which made
"4 preserved" a ceiling shaped like a measurement — and the path it could not exercise was exactly the
exception the card had just finished documenting.

Rebuilt to deep-copy the committed row and reset **only** the four clobber-written fields, leaving
`species_name` and `taxonomy_id` as committed. That surfaces a second trap the reviewer named before
it could bite: **an absent field produces no disagreement for the same reason a blind probe does**, so
"reachable and did not fire" is only a measurement if the fields were *present*. Reported per row,
presence and match separately: **6 of 6 live, 0 contradictions, 4 preserved.** Six named
"could have fired, did not" rows rather than a zero from absence.

The author then added what neither the orchestrator nor the reviewer asked for: **a positive control.**
A real corpus row with `species_name` perturbed to *Xanthomonas campestris* **does** raise
`row_species_fields_disagree_with_resolution`, and the probe **exits 1 and refuses to report a number**
if that control ever fails. The zero is a demonstrated negative, not a quiet path. **That is the
vacuity question asked of the measuring instrument rather than of the code**, and it is the standard
this sprint should hold probes to from here.

The reviewer's stop-condition is **encoded rather than checked**: if either pinned example clears the
source gate and fails the contradiction gate, the probe prints `*** STOP CONDITION MET ***`, states
the failure is against **REV-099 Finding 1's premise rather than the code**, forbids adjusting
`map_ids.py`, and exits 2. It did not fire. That condition existed because the orchestrator's original
stop-rule covered *"contradicts production code"* and left *"contradicts the ruling"* open — and round
two was the change that made the contradiction gate reachable for the first time.

**The two 4s are different numbers that coincide** — the charter's **4/2 census split** (refs by
source, read off committed data) is not the **preserved-row count**. Recorded as a do-not-reconcile
note so a later reader does not "correct" a figure that was never wrong.

**Disclosed-unmeasured:** Chunk D. `test_streamlit_stage8_export_contract.py`,
`test_streamlit_stage2_orchestration.py`, `test_streamlit_quarantine_boundary.py` are uncovered by
this card; F-136 already records that Chunk D cannot go green in this environment with the DB up.
Not held on it, but not implied as covered either.

---

# REV-100 — C-100 reviewed, APPROVED, and two registrations required before merge

**C-100**, `f7dc223..9d2c587`, reviewed by the same independent peer session. **APPROVE**, with two
registrations and one boundary note — none of them a code change.

## The three pre-agreed pass/fail items, verified at the branch tip

| Item | Result |
|---|---|
| `unknown_backed_rationale` gains no new readers | **PASS** — exactly 3 hits, all `goldset.py` (`398`, `549`, `805`); **0** in `semantic.py`. The remaining hits are inside `pinned_v1.json`, which is data, not a reader. **The invariant caught two regressions the author had introduced and removed before committing** — it earned its place rather than passing vacuously |
| an absent scope inherits, never defaults permissive | **PASS**, and in a better form than any of the three named in advance |
| `:1417` and `:1453` move together | **PASS** — and the summary is *re-derived*, not re-keyed |

The tolerance decision, which is the whole card in five lines:

```python
inherited = self.unknown_backed_proteins_acceptable
if not self.unknown_backed_tolerated_entities:  return inherited   # no scope -> Boolean governs
if not inherited:                               return False       # a False case can NEVER be widened
return self.unknown_backed_tolerance_match(candidate) is not None  # a declared scope is exhaustive
```

The summary arm now reads a `tolerated` counter incremented in the loop and discloses
*"(each one named by the case)"*, so it reports **what happened** rather than re-reading the Boolean.
That closes the under-disclosure the review predicted: without it, a scoped case would have stopped
saying anything had been tolerated, silently, for exactly the cases this card exists to create.

## The implementer was right against its instruction, and said so

The orchestrator wrote that an entity **absent from a non-empty scope** must *"inherit the case
Boolean"*. Taken literally that returns `True` for every absent name — a scope can only exist on a
case whose Boolean is already `True` — which would make **every scope decorative** and excuse the
seven core enzymes, the exact opposite of the card. The implementer implemented the coherent reading
(**absent scope inherits; a declared scope is exhaustive**), preserved the property the rule actually
protects (`if not inherited: return False`, so none of the nine explicit `false` cases can be
widened), and **named the divergence in its report and its docstring.**

**Second time this wave an implementer was right against its instruction.** Both times it said so
rather than silently complying or silently diverging.

## The A/B — zero movers, run twice by two parties

| Node set | Base | Tip | Delta |
|---|---|---|---|
| SMOKE, 20 files | 473 passed | 473 passed | **0** |
| gold-readers, 22 files | 2 failed / 453 passed / 8 skipped | 2 failed / 453 passed / 8 skipped | **0** |

Run by the author, and **re-run independently by the orchestrator** in separate jobs against the same
two worktrees (`evidence/g11/ORCH-712/06`, `08`, `09`). Identical failing node IDs on every leg. The
author predicted **zero movers in writing before the tip run**, with a reason per named file, and
compared per-file progress lines without `-q` rather than totals.

The 22-file selection covers all 14 charter-named files **plus eight further `goldset` /
`bench.semantic` importers found by grep** — including `test_semantic_release_gating.py` and
`test_c056b_semantic_denominators.py`, the two the review had flagged as highest-risk.

**G9:** 19 of 32 base failures behavioural, not symbol absence — the author reordered its tests after
the first G9 run so scorer behaviour asserts *before* schema symbols, raising the behavioural count
from 7. Base text: `AssertionError: LpxA is an expected core Raetz enzyme … assert 0 == 1`. Tip: 39
passed.

### An open red, registered rather than absorbed

`test_strict_failure_replay.py::{test_every_stored_strict_failure_replays_to_its_recorded_verdict,
test_recovered_cases_are_smaller_and_refused_cases_are_not_claimed}[only_unrelated_reactions_survive]`
fails **on both legs** and is **pre-existing on `f7dc223`**. A Glutathione biosynthesis payload: no
gold case, no PMC12444477, no `unknown_backed` surface. The fixture records `recovers: false` while
`quarantine_and_close` now returns `ok=True`. **Not C-100's**, and not to be carried as background
noise.

## REGISTRATION 1 — the bare `Unknown` sentinel is an F-132-class instrument tension

Scoping makes the bare PathBank `Unknown` sentinel a finding on PMC12444477 — 3 sentinel rows across
archived legs. It is **not** one of the seven the rationale names, and refusing it is defensible: the
tolerance list is about *named entities the paper discusses that will not cleanly resolve*, and a bare
`Unknown` is not a named entity, it is the absence of one.

**But the test that matters is whether the pipeline can clear the finding by doing something
correct — and here it largely cannot.** D-070 § O-1a rules the sentinel is PathBank's own legitimate
representation, so the only way to clear it is to stop emitting it, which may be the wrong behaviour.

> **When a finding cannot be cleared by correct behaviour, it is measuring the representation rather
> than the pipeline.**

That is **F-132's shape exactly**, and this count is to be read as instrument tension, not pipeline
defect. Merge rule 6 is **not** in play — the change makes the scorer *stricter*, never more
permissive. Excusing the sentinel would be an eighth gold entry and needs **D-071 amended**; it is a
product-owner call and no agent may make it.

## REGISTRATION 2 — `lipoprotein`, and the three-way correction that resolved it

The gold tolerates a generic `lipoprotein`. Three readers produced three different accounts and **all
three were wrong in part**. Resolved by reading the source text:

* **The author** reported the token occurs **0 times** in the paper and is an extractor output found
  once in `runs/2026-07-28_2122/.../merged_payload.json`. The zero is right; **the provenance is
  not** — that file does not contain it.
* **The reviewer** found the body contains **`Lpp`** once — *E. coli* murein lipoprotein, specific and
  resolvable — and inferred the tolerance excuses degrading a resolvable name into a generic one.
* **The orchestrator's first probe** found `Lpp` **0** times with a passing control (`LpxC` = 1726),
  contradicting the reviewer. **That probe was wrong**: the pattern was case-sensitive `\bLpp\b`.

**Ground truth.** The token is lowercase **`lpp`** — the *gene*, not the protein — and it occurs
**once**, in `01_source_text.txt` of every archived copy:

> *"…can be suppressed by deleting a protein that tethers the OM to the cell wall, **lpp**, thereby
> elevating OM vesiculation…"*

So the reviewer was right that the paper names a specific lipoprotein, and right that it is
resolvable. **But its inference does not hold.** `lipoprotein` appears in **no** protein or
protein-complex row in **any** payload — 0 across 20 payload files carrying entities. Where it does
appear it is a **reaction participant**: an *input* to `Lnt acyl transfer` (`GPL donor` +
`lipoprotein` → `LPL`, enzyme `Lnt`) inside `rag_admission_report.json`. **Lnt is lipoprotein
N-acyltransferase — a different pathway from Raetz**, and the row is a gap-admission candidate, not a
degraded `lpp`.

**Therefore the `lipoprotein` tolerance entry is INERT.** The scorer applies tolerance to
protein/protein_complex rows; no such row is ever named `lipoprotein`. It can never fire. It is
defensive, and it is not excusing a lost identity.

### Confirmed on a wider population, and the reviewer records its own error

The reviewer re-ran the question over **245 payload files** carrying entities — twelve times the
orchestrator's 20:

```
payload files carrying entities                              245
protein / protein_complex rows named exactly "lipoprotein"     0
payload files with "lipoprotein" anywhere in processes         4
lowercase lpp in 01_source_text.txt                        1
```

**Zero protein rows across 245 files.** Inert is confirmed on a population large enough to settle it,
and the four hits sit in `processes`, consistent with the `Lnt acyl transfer` reading.

**The reviewer asked for its own error to be recorded here under its name**, rather than left as a
near-miss, and it is right that it belongs beside the others: *"'the extractor degraded `Lpp` into a
generic protein row' was my own leap, and there is no such row in the corpus. I inferred a mechanism
from a coincidence of vocabulary and did not check whether the thing I was theorising about existed —
the same error I have spent eight rounds finding in other people's work."*

Three readers, three accounts, each wrong in part, and **none of them wrong about a number**. The
author's zero was right and its provenance was not; the reviewer's `lpp` was right and its mechanism
was not; the orchestrator's control passed and its question was wrong. **That is the wave lesson in
its purest form.**

**Kept, not removed** — removing a gold entry needs its own source evidence, and an inert entry
harms nothing. The `quote: ""` stands: the author **refused to fabricate a span** for a token the
paper does not contain, which is exactly right and is the one thing the quote field exists to
guarantee.

**Residual observation, not a card:** the paper names `lpp` once and the pipeline never emits it as an
entity. It is not in `expected_enzymes` or `acceptable_enzymes`, it is not a Raetz enzyme, and it is
mentioned in passing about OM tethering — so not emitting it is correct. Recorded only so a future
reader does not rediscover the token and mistake it for a miss.

## Boundary note — scope creep, allowed and named

The author added an **unrequested load-time refusal** for a scope declared under a `false` Boolean
(unreachable data). It guards a gold-authoring error that would otherwise be silent, and it matches
the file's existing *"refuse it at load rather than let it quietly mislead"* style. **Allowed, and
recorded here as beyond the charter** so the boundary record stays honest.

## Disclosed, not found

* **The stall.** The agent went idle ~12 hours with everything uncommitted and both tip legs unrun. A
  status check woke it; it committed, ran them, certified. Reports 01–08 are dated 03:44–03:57 and
  09–12 16:25–16:28 — **that gap is a stall, not a long job.**
* **An escaped heredoc.** One `python - <<EOF` ran outside the bounded wrapper, hung on stdin and was
  killed by the shell's 2-minute clock. The author verified **zero survivors** immediately by full
  command line and used no named or global kill; the orchestrator independently observed the two
  `python.exe -` processes from outside and watched them clear. **Reported rather than omitted.**
* **`LpxH` is now a finding** and the list was deliberately **not** widened. `LpxH` is the *E. coli*
  enzyme for the organism-dependent ninth step; `LpxG` is the variant in other organisms. An
  Unknown-backed `LpxH` means a resolvable *E. coli* enzyme failed to resolve — a genuine finding.
  **Widening would be the merge-rule-6 direction**: weakening a gate to reduce a count. Declining to
  widen needs no positive evidence; widening would.

**Net effect on measured legs**, static re-scores of archived artifacts, no benchmark run:
`runs/2026-08-02_2130/.../strict` **0 → 9** tolerance findings (7 core enzymes + `LpxH` + `Unknown`);
`runs_verify/2026-08-24_1428` **0 → 2**.

## And one operational rule this card paid for

**A heavy lock is not stranded until it has been sampled over time.** Diagnosing the stall, the
orchestrator found the lock held by a PID that was not running, with **zero** Python processes — the
textbook stranded signature — and was one step from clearing it as exact-owner cleanup. Re-reading the
holder file first showed it had **changed between reads** (`tip-smoke`/392540 → `tip-goldreaders`/394680).
The agent was alive and cycling through short jobs, and every instantaneous sample landed in a gap.
Sampling every 15 s for two minutes showed it release cleanly.

**Clearing it would have put two heavy jobs on the same trees.** The signatures of *dead* and *cycling
fast* are identical at an instant. The rule against deleting an unfamiliar lock is really a rule
against deleting a lock you have looked at **once**.

The lock also earned its keep in the other direction the same hour: an orchestrator job that would
have duplicated the agent's in-flight leg **refused with exit 95** rather than run.

---

# WAVE — ACCEPTANCE-INSTRUMENT RECONCILIATION AND T-107 READINESS · opened 2026-08-28

Lead Orchestrator session `project14-t2pw-af`. Opened at integration tip **`b7f1bea`**.

## Takeover verification — all checks passed at open

| Check | Expected | Found |
|---|---|---|
| local = `origin/` = `git ls-remote` | `b7f1bea` | **`b7f1bea`** ✔ |
| merge in progress / staged | none / none | none / none ✔ |
| heavy lock `C:/t/heavylock` | absent | absent ✔ |
| sprint-owned Python | zero | zero ✔ |
| allowed IDE processes | two `ms-python.isort` | PIDs 407368, 407420 ✔ |
| product-owner `streamlit_app.py` | 35 ins / 2 del, `sha256:47e4fafa…` | **exact match** ✔ |
| caches + `topics_*.txt` | uncommitted | uncommitted, untouched ✔ |

**One documented discrepancy, pre-existing and not ours.** The brief expected `main` at `7531692`.
**Local `main` is `7531692` and untouched; remote `main` is `03f1af5`.** `7531692` is an ancestor of
`03f1af5`, so `main` advanced **outside this sprint**. The previous orchestrator confirmed it had
checked only the local ref and corrected its own final report. The invariant that matters was
verified independently by that session: **the sprint tip is NOT an ancestor of remote `main`** — no
sprint work has leaked. Neither ref touched.

## Peer coordination — complete, and the tree is exclusively ours

`ListAgents` showed **one** live peer, `project14-t2pw-14` (interactive). Contacted before claiming
the branch. It **stood down explicitly**: owns nothing, holds no lock, runs no Python, will not touch
`semantic.py`, and will message before touching the tree if redirected. Its five worktrees
(`c099`, `c100`, `c099base`, `c100base`, `c099g9`) and the older set are recorded and **none is to be
pruned**. `project14-t2pw-51`, the independent reviewer for C-099/C-100, **is gone** — its context
went with it; C-101 needs a fresh reviewer.

Three method standards inherited from that reviewer, recorded here because they were never in the
ledger as method: **pass/fail items agreed in writing before the diff exists** · **predictions
recorded before the run** · **never let an aggregate stand for the claim** (run split, not combined —
a combined total hides a shift between two files) · **check the guard that was REMOVED, not only the
one added.**

## Rulings recorded — both open asks of the F-132 bundle are now closed

| Ruling | Decision | Substance |
|---|---|---|
| **A** | **D-072** | Coverage anchors reconciled against the same case's `forbidden_identifiers`. Raw preserved, accepted computed separately, exclusion exact and case-scoped, Priority 1 still catches erroneous export. **Instrument reconciliation, not a pipeline fix.** |
| **B** | **D-073** | Priority 1: `0–6 PASS` · `7 PASS_WITHIN_VARIANCE` · `8+ FAIL`, on the contract-adjusted count with raw preserved. Six remains the target. **Clears T-107 gate condition 1.** Do not rerun to chase a six |
| **C** | **D-074** | PMC12444477 tolerates the exact PathBank `Unknown` sentinel only — five conditions, row-predicated, `is_pathbank_unknown_protein(row)` authoritative. **`LpxH` does not move: 9 → 8, not 9 → 7** |
| **D** | **D-075** | A truthful Priority-2 `NOT EVALUATED` from the unmet D-067 precondition 3 is not an automatic T-107 failure. `CONDITIONALLY SATISFIED` may clear the gate; the report may not claim full 20-leg validation |

**Consequence for T-107.** The previous wave recorded gate condition 9 as *"not met and not reachable
by any engineering in this sprint"*, because only ask B could clear it and B was unanswered. **B is
now answered.** That blocker is lifted; the remaining conditions are engineering and are tracked
below.

## F-142 — the Glutathione red is diagnosed, not inherited

Four cards carried this pair as accepted noise on the strength of *"fails at base"*. It is now
classified, with measurement rather than argument: **stale expectation, not a production defect.**

The coverage gate is entirely correct (`core_accepted_processes 0`, `minimum_core_satisfied false`,
both reasons present). **C-041a (`4177fe5`, under D-002) deliberately moved a `minimum_core:*`
shortfall from `refusal_reasons` to `review_reasons`**, so `ok` stopped answering the question the
fixture asks it. The protection did not vanish — this payload gets `review_required`,
`strict_acceptance_eligible false`, `completeness 0.0`, all three anchors named missing. The
empty-graph sibling still hard-refuses to `diagnostic_only`, so the distinction is live.

**Merge rule 7 and `has_surviving_core`'s own docstring both require the current behaviour verbatim.**
Chartered as **C-103**, sequenced last. **Not a T-107 blocker.**

**Structural note — the third instance this sprint.** `test_strict_failure_replay.py` is in **no
chunk**. A red that no gate runs gets carried rather than diagnosed. Another datum for F-049/F-054.

## Evidence rescued — the F-132 population was one cleanup from being lost

`evidence/g11/ORCH-702/03-f132-forbidden-anchors.json` certified a clean job and **preserved nothing
about what it found**; the probe ran from a **session-local scratchpad of a session that has since
ended**. Recovered and committed as `evidence/orch702_f132_forbidden_anchors.py` / `.log`.

Reproduces the bundle exactly — **52 legs with unmatched terms · 281 terms · 62 gold-forbidden · 32
legs affected · 6 papers**. **It also corrects the bundle's prose:** the population spans **five**
mechanism kinds, not four — `placeholder_product` 19, `heading_or_prose` 19,
`regulator_as_metabolite` 16, `cofactor_as_protein` 6, **`modification_state` 2** (omitted there).

Separately measured and committed (`evidence/orch713_gold_selfconsistency.py` / `.log`): **zero
overlap** between any gold case's declared positive fields and its own `forbidden_identifiers`,
across all ten cases. **This is a right answer to a narrower question than Ruling A asks** and is
labelled as such in C-102's charter — it establishes that the contradiction is introduced **at
runtime by the Stage-0 draw** and is not authored into the gold, which is why the bundle's
*"Gold: None"* holds and why the fix belongs in the scorer. It is **not** evidence that F-132 is
absent.

## C-102 implemented — the exact documented delta on Priorities 4 and 5

Measured **offline over the 62 committed `quarantine_report.json` artifacts**. No leg re-run, no
cohort, no live call. `evidence/c102_f132_coverage_ab.py` / `.log`.

| | Pre-change | Post-change |
|---|---|---|
| coverage answers per leg | one, unreconciled | **two — raw preserved verbatim, accepted beside it** |
| legs with a coverage block | 62 | 62 |
| requested-core terms drawn | 860 | 860 |
| gold-forbidden terms withheld | **0 — none was excluded** | **92** |
| legs carrying ≥ 1 such term | not measurable | **47** |
| legs clearing the unchanged 0.500 minimum | 6 below | **6 below — ZERO cleared** |

**Priority 4 does NOT move off `0/8`, and Priority 5 does not move either.** Measured on
`runs_verify/2026-08-24_1428`: both read `0/8 = 0%` and `0/2 = 0%` before and after. Their numerators
are semantic confirmation and the frozen strict release record, not the requested-core ratio, so no
recomputed ratio can move them — and Priority 5 deliberately does **not** promote a leg the runtime
froze as `review_required` even where its coverage block would now clear, because this module scores
runs and does not reclassify them (merge rule 8). What became readable is the coverage measurement
itself, per leg, raw beside accepted.

**`PMC12782028/strict` does NOT clear.** `requested_core_coverage_below_minimum:0.222<0.500` becomes
`6/23 = 0.261`, still below the unchanged 0.500. The four withheld terms — `LIPA`, `LBR`, `SREBF1`,
`SREBF2` — are the exact four Priority-1 survivors on that paper, and all four remain punishable
under Priority 1 and named in the diagnostics.

**Direction, reported as it fell:** of the 47 affected legs, **32 rise, 7 fall, 8 are unchanged.**
A leg falls when the pipeline **matched** a forbidden term: the exclusion is symmetric, so a
forbidden match is withheld from the numerator too. Counting it as a coverage success would score
obeying the gold below breaking it, which is the inversion D-072 exists to remove.

**Two corrections to the recovered ORCH-702 population, both measured not argued.**

1. Its probe replayed unchanged on `bcf9a23` gives **54 legs · 304 unmatched terms · 66
   gold-forbidden**, against the **52 · 281 · 62** in its own committed log at `e9aa5c8`. The
   artifact population grew by two legs; the probe is sound. `evidence/c102_orch702_replay.log`.
2. ORCH-702 counted forbidden terms only among the **unmatched**, so a forbidden term the pipeline
   **matched** — and was given coverage credit for — was invisible to it. There are **26** of those,
   which is why this card measures 92 where the probe measures 66. They surface a **seventh** paper,
   **`PMC13231680`** (3 legs), outside the bundle's six.

**Seam.** `src/t2pw/bench/acceptance.py` and `src/t2pw/bench/render.py` only. **No production
pipeline file and no gold file is touched**; `strict_quarantine.py` is untouched, as § 3 of the
charter requires. Zero forbidden identifiers removed, softened or reworded; the threshold value does
not move.

**G9** is a behavioural proof, not symbol absence: `evidence/c102_g9_denominator_proof.py` asks the
public `score_run(...).to_dict()` which requested-core denominators it states for that leg and gets
`[]` at base against `[23, 27]` at the tip. **All seven mutations in
`evidence/c102_mutation_attack.py` are detected**, including one that leaks the coverage exemption
into Priority 1, one that restores the contradictory denominator outright, and **M7**, which reverts
the numerator half — see the correction round below.

**Gates.** Gold-readers **2 failed / 453 passed / 8 skipped at base AND at tip** — the two F-142
reds, unchanged, no third. SMOKE **473 passed**. Focused **14 passed**. Every job
`FINAL SURVIVING COUNT : 0`, `cleanup : success`.

### C-102 correction round 1 — REV-102, and the deviation that shipped untested

**The escalated line had no assertion behind it.** D-072 says forbidden terms leave the
**denominator**; this card removes them from the **numerator** too. That deviation was escalated
rather than taken silently — and it still shipped with nothing testing it. Reverting the one line
(mutation **M7**) left all eleven tests green. **This is F-144 on my own card**: a claim nobody had
attacked. **M7 is now in the attack set and tests 12 and 13 bite it.**

**The deviation is right, and the corpus says so more sharply than either side argued.** Measured
independently at `evidence/c102_numerator_verify.log`, a denominator-only exclusion reports a
"coverage ratio" **above 1.0 on nine committed legs — eight of them exactly `1.2000`** (`6/5`), and
one at `1.125` (`9/8`). It is not a rate. **23 of the 62 legs carry a matched forbidden term**
— confirmed here after REV-102's own first count of 19 was corrected by the corpus. Under the
literal reading, matching a forbidden identifier is worth exactly as much as matching a legitimate
anchor, so obeying the gold scores **below** breaking it. Removing the term from both sides makes a
forbidden match exactly **neutral**, which is the property test 12 pins.

**PRODUCT_CONTRACT § 7 currently reads "numerator and denominator alike" while LOCKED D-072 reads
"denominator".** D-072 outranks it. **That reconciliation is the product owner's, not mine**, and is
being recorded separately; this card's diff is not an improvised product decision.

**Re-gated on the corrected tree.** Focused **14 passed** · gold-readers **2 failed / 453 passed /
8 skipped combined** and **453 / 2 / 8 split one file per process across all 22** — identical
totals, **zero per-file shift**, both reds isolated to `test_strict_failure_replay.py` · SMOKE
**473 passed** · base `bcf9a23` re-measured: the same **2 / 453 / 8**, no third failure at either
end. All seven mutations RED, tree clean after each. Every job through `c045_pinned_pytest.py`,
which printed the resolved `T2PW` path to this worktree's own `src` on each; `FINAL SURVIVING COUNT : 0` and
`cleanup : success` throughout.

**Two runs kept because they failed.** The split gate's first run had no `--basetemp` parent, so
every test errored in setup and files reported `0 passed` with exit 1 — an infrastructure failure
wearing the costume of a wiped test file, and the driver now aborts on exactly that shape rather
than folding it into a total. The mutation attack's first run would not parse, because a shell
heredoc collapsed an escape in the M7 substitution. Neither is a test result and neither is deleted.

**Also corrected in this round.** The serialization note (the report-level key is unconditional —
a run with zero coverage blocks still grows) · the G9 row, which described an `ImportError` as if it
were the proof · the aggregate/per-leg key-name collision, now `coverage_reconciliation_corpus` ·
a shallow copy in `ModeResult.to_dict` sharing `excluded_terms` by identity · the ~24% report growth,
now ~12.4% · and `render.py`'s 27 lines, which had no test and now have test 14.

## Cards

| Card | Branch / worktree | State |
|---|---|---|
| **C-101** — 16/5 metric split · row-aware sentinel seam · raw/accepted Priority 1 | `card/C-101-o1-metric-split` · `C:/t/c101` | **DISPATCHED** 2026-08-28 on base `d7cf4a4`. Charter amended (AMENDMENT 1) for D-073/D-074 |
| **C-102** — coverage denominator vs `forbidden_identifiers` (Ruling A) | `card/C-102-coverage-denominator` · `C:/t/c102` | **IMPLEMENTED** on base `bcf9a23` (the C-101 integration tip). Awaiting independent review. Delta below |
| **C-103** — re-point the F-142 replay expectation | *(not dispatched)* · `C:/t/c103` at dispatch | **CHARTERED.** Sequenced after C-102. Not a T-107 blocker |

**C-101 and C-102 are explicitly NOT parallel work.** Serial ownership of `src/t2pw/bench/` is a hard
requirement of this wave even though they touch different functions.

### The gold-readers baseline, which every card in this wave must be told

The 22-file gold-readers selection **exits 1 at base, correctly**: it contains
`test_strict_failure_replay.py` and its two F-142 reds. **Expect exactly two; a third is the card's.**
**C-103 changes this** — it is required to hand back the corrected baseline (expected `0 failed`,
exit 0) so later charters are updated.

## Agent liveness register

| Agent | Card | Worktree | Start | Last evidence of progress | State |
|---|---|---|---|---|---|
| `a63b0e93…` | C-101 | `C:/t/c101` | 2026-08-28 ~11:14 | worktree created, base `d7cf4a4` confirmed; no commit or dirty file yet | **RUNNING** — status requested at first checkpoint; reading/A4-archaeology phase is expected to precede any edit |

Protocol in force: **~15 min** without observable progress → request status · **~30 min** with no
response, no process, no changing artifact, no commit and no owned job → treat as stalled, interrupt,
preserve the worktree, record, redispatch from the last verified commit. **Rapid lock cycling is not
an abandoned lock** — read holder metadata at least twice across intervals before declaring one
stranded.

## Run ledger — this wave

| Job | Task | Purpose | Exit | Survivors | Cleanup |
|---|---|---|---|---|---|
| `01-glut-classify` | ORCH-713 | per-case replay vs fixture | 0 | **0** | success |
| `02-glut-release-seam` | ORCH-713 | did the protection move or vanish | 0 | **0** | success |
| `03-glut-release-fields` | ORCH-713 | `quarantine_report.release` fields | 0 | **0** | success |
| `04-f132-overlap` | ORCH-713 | gold self-consistency — **failed on a wrong import name** | 1 | **0** | success |
| `05-f132-overlap-v2` | ORCH-713 | same, corrected | 0 | **0** | success |

`g11_evidence.py check --task ORCH-713`: **5 artifacts, 0 non-compliant.** **The failed run is kept,
not tidied away** — it was a real bounded job, and deleting evidence to make a record look clean is
the thing these rules exist to prevent.

**No live model run, no cohort, no paper leg, no benchmark, no T-107.** External model spend this
wave: **$0.00** against a **$5** ceiling.

## T-107 readiness — live blocker list

| # | Condition | State |
|---|---|---|
| 1 | Priority 1 not guaranteed to fail | **CLEARED by D-073** — `PASS_WITHIN_VARIANCE` at 7 |
| 2 | Priority 2 not guaranteed to fail | **CLEARED by D-075** — `CONDITIONALLY SATISFIED` is admissible |
| 3 | C-101 merged and independently approved | **OPEN** — dispatched |
| 4 | Ruling-A card merged and independently approved | **OPEN** — chartered, blocked on C-101 |
| 5 | Rulings A–D recorded accurately | **DONE** — D-072…D-075 |
| 6 | Glutathione red classified, not a guaranteed acceptance failure | **DONE** — F-142; cannot affect T-107 |
| 7 | deterministic gates green; integration pushed and remotely verified | rolling — verified after every push |
| 8 | heavy lock free · zero sprint-owned Python · no peer owning an overlapping job | **DONE** and re-verified |
| 9 | LM Studio + pinned model available; run monitorable to completion; spend ≤ $5 | **not yet assessed** — assess only when 3 and 4 close |

**T-107 remains NO-GO**, and the reason has changed for the better: it is now blocked on **two
mergeable cards** rather than on an unanswerable product question.

---

## C-101 MERGED — `ee7cb6b`, gate pinned at `bcf9a23` · 2026-08-28

**Branch** `card/C-101-o1-metric-split`, tip `0ff60b4`, base `d7cf4a4`, worktree `C:/t/c101`
(**do not prune**; `C:/t/c101base` likewise). Merged `--no-ff`. **REV-101: APPROVE.**

### What landed

| | |
|---|---|
| **16/5 split** | `placeholder_backed_proteins` keeps its value and meaning; four mutually exclusive, exhaustive categories beside it, with `other` **reported** rather than dropped. Reproduces D-070 exactly on the pinned run: **21 = 5 + 16 + 0**, F-141 **24 / 0 / 0** |
| **Row-aware sentinel seam** | `tolerates_unknown_backed(name, row)`. Licence in its own field `unknown_backed_tolerated_sentinel`, **outside** the name-keyed list, refused at load if declared name-only, returns `False` for `row=None`. Legacy one-arg call returns `False` |
| **Raw + accepted Priority 1** | both counts reported; `0–6 PASS · 7 PASS_WITHIN_VARIANCE · 8+ FAIL`. Nothing anywhere maps `PASS_WITHIN_VARIANCE` onto `PASS` |
| **`LpxH`** | untouched. PMC12444477 goes **9 → 8**, never 9 → 7. `Unknown` is the **only** finding-list delta across all 11 legs |

### Gates

**SMOKE 473 = 473** (base, tip, and **post-merge on the combined tree**, `MERGE-101/01`). Gold-readers
**2 failed / 453 passed / 8 skipped** at base and tip, run **split one-process-per-file as well as
combined**, **zero per-file shifts across all 22**. Focused **38 passed**. G9 pasted: the regression
half runs and asserts at base; `test_9b` `KeyError`s there. Every job `FINAL SURVIVING COUNT : 0` /
`cleanup : success`.

### Three correction rounds, each of which found something real

**Round 1 — an asymmetry the card applied to itself.** It correctly reported `placeholder_other_rows`
rather than folding a remainder into a clean bucket, then folded **F-141's** remainder into *correct
withholding*. Now routed per F-141's own table: unrecognised rungs and `mismatch`/`conflict` →
`withheld_identity_other`; **absent** rungs stay `CORRECT` (the two Fur rows, which the table does
call correct). Verified by execution — both Fur rows carry `checks.species = None`, and the only two
rung values in the pinned corpus are `unknown` and `None`, so **no pinned row can reach the new bucket
at all.**

**Round 2 — a defect the orchestrator introduced.** My round-1 instruction to tighten the
bare-sentinel guard to match D-074 was right, and the guard stays — but `_contract_adjustment`'s only
call site is inside `if ids:` and the tightened guard is `if ids: return ""`. The seam became
unreachable for every input, so `accepted == raw` **by construction**, which the card's own charter
forbids. **The reporting had become untrue**, asserting a measurement over a structurally impossible
quantity — the same failure the same commit had fixed one function away. Answered as **D-077**.

**Round 3 — F-144, and the reason it was worth a third round.** Both tests written in round 2 to
guarantee that honesty were **vacuous**, proven by mutation: reverting the guard left all three
passing, and **deleting the bareness guard entirely left the focused file at 38 passed.** Fixed
test-only — the **git tree object for `src/t2pw` is identical** at `e14ab87` and `0ff60b4`. Both
mutations now fail exactly the two intended tests, for the stated reasons, **re-run independently by
the reviewer.**

### Budget

Ceiling 1 ratified **420 → 560** (**D-076**) because the orchestrator under-set it when AMENDMENT 1
tripled the scope — the sixth under-set ceiling of this sprint. Card measured **541**, having trimmed
`src` **563 → 499** in prose first, and **stopped rather than self-authorizing**. Review-mandated
corrections budgeted separately (**D-076 Amendment 1**): round 1 **59/60**, round 2 **34/25** (9 over,
**taken rather than reflowed**, on standing instruction), round 3 **0/40**. Cumulative base→tip
**600**, of which **93** is review-mandated.

### Conduct worth keeping — recorded because it is the behaviour to want

* **Four failed measurements committed beside their corrections**, never replacing them: `01`/`04`/`07`
  (round 0), the WSL/bash misfire at `22`, the `28` collection error, the wrong `>= 5` guess at `36`.
  REV-101 on the practice: *a repository that keeps only successful measurements has an evidence trail
  that is a survivorship-biased narrative.*
* **The card refused twice to loosen a guard to manufacture a number** — declining to weaken the
  accession guard in round 0, and again in round 3 rather than making the seam reachable.
* **It declined to write to `DECISIONS.md`** although the charter listed it as in scope, because
  `CLAUDE.md` marks it append-only and product-owner-only. REV-101 said it would have **rejected** the
  alternative: a subagent writing there would be improvising product authority.
* **It reproduced its own vacuity as a committed measurement** (`c101_probe_vacuity.py`/`.log`) before
  fixing it, rather than quietly correcting.
* **REV-101 disclosed two failures of its own** — three legs run under the wrong interpreter (**F-143**),
  and an allocator label rejection that silently swallowed a report path — and re-ran rather than
  keeping only the good runs.
* **REV-101 recorded predictions before running and was wrong on three in writing** (P6, P7, P13).

### Two charter errors of mine, corrected in D-076

`AMENDMENT 1` § A4 sent the card after *"the row used by C-100's accepted A/B"* — **that A/B is a
test-node A/B** (20 SMOKE + 22 gold-readers = the "42 files") and contains **no payload row.** And the
amendment's header named base `b30193f` while the dispatched worktree carried `d7cf4a4`, because
committing the amendment advanced the tip before the worktree was moved onto it.

### Not closed by this card

**F-132 remains open** — that is C-102. **D-077 is answered** but the seam it describes stays
unreachable by design until a future licence exists. **F-142's two reds are untouched** and remain the
expected gold-readers baseline until C-103 lands.

---

## C-102 MERGED — `8e4334f`, gate pinned at `ad62338` · 2026-08-28

**Branch** `card/C-102-coverage-denominator`, tip `e213742`, base `bcf9a23`, worktree `C:/t/c102`
(**do not prune**; nor `C:/t/rev102base`, `rev102tip`, `rev102r1`). Merged `--no-ff`.
**REV-102: APPROVE.** One correction round.

### What landed

`contract_accepted_coverage(case, coverage)` reads a leg's **frozen** coverage block and the case's
own `forbidden_identifiers`. Raw is **copied verbatim**, never recomputed — 0/62 drift measured.
Accepted is computed beside it, removing exact case-scoped forbidden terms, alias-aware with the
gloss-head retry. Priorities 4 and 5 carry the reconciliation; their `ok`/`observed` are unchanged.
**No production pipeline file, no gold file, no threshold change** — `min_core_coverage` still `0.5`,
read from each leg's own record rather than hardcoded.

**The seam was settled before dispatch and held:** the exclusion lives in `bench/`, never in
`strict_quarantine.py`, because the forbidden list is gold and the ratio is production, and threading
gold into the pipeline would embed gold-set-only policy into it (`PRODUCT_CONTRACT` § 12). REV-102
confirmed **no production module imports `bench.acceptance`**.

### The deviation that turned out to be the ruling

D-072 says *denominator*; the card excludes from **numerator and denominator alike**, and **stopped
to flag it rather than deciding silently**. The literal text is what is wrong, and it is measurable:
removing a **matched** forbidden term from the denominator alone leaves it in the numerator, so
**nine committed legs report a ratio above 1** — eight at `6/5`, one at `9/8` — which is not a rate.
Synthetically, a pipeline that exported **all three** forbidden identifiers on `PMC12782028` and
matched no legitimate anchor scores **1.0000**, and matching a forbidden identifier is worth exactly
as much as matching a legitimate one. **F-132 with its sign flipped, at full amplitude.**

No counter-perversity: across all 64 match-subsets on a real gold case, un-matching a legitimate
anchor never raises the accepted ratio, toggling a forbidden match is **exactly neutral**, and the
ratio stays bounded by 1. Recorded as **D-080**, an *interpretation* flagged for product-owner
ratification, **corrected to nine by D-080 Amendment 1**.

### The correction round — F-144 again, on the card's most consequential line

The card shipped that deviation **defended by zero assertions**: REV-102 reverted the numerator half
and **all 11 tests stayed green**. Tests 12 and 13 now bite, M7 is in the attack set, and **all seven
mutations go red** with the tree clean after each. Also fixed: two factually wrong `TEST_MATRIX`
statements, an untested `render.py` block, a key-name collision (`coverage_reconciliation_corpus`),
and a shallow-copy aliasing hazard. F5 compaction removed **23,818 bytes — 49% of the growth** — and
the card reported, **without rounding it away**, that the fix makes the zero-coverage report 76 bytes
*larger*.

### Results, both of which contradict expectations

* **D-081 — Ruling A moves neither Priority 4 nor Priority 5.** The bundle predicted Priority 4 would
  move off `0/8`. Reproduced across **all 21 run directories at base and tip**: not one moves,
  `legs_cleared_by_reconciliation` empty on every one. What Ruling A bought is real but smaller —
  the coverage measurement is **readable per leg** for the first time.
* **F-145 — the F-132 population was an undercount: 92 terms / 47 legs / 7 papers**, not 62/32/6. The
  probe only ever iterated `unmatched_terms`, so **26 matched forbidden terms were structurally
  invisible** and the seventh paper (`PMC13231680`) never appeared. **And I committed that probe's
  recovered log without re-running it** against the tree I was committing it into.

### Gates

SMOKE **473** at tip and **post-merge on the combined tree**. Gold-readers **2 failed / 453 passed /
8 skipped** at base and tip, run **split one-process-per-file as well as combined — identical per
file, per outcome**. Focused **14 passed** at tip. G9 behavioural, reproduced by REV-102 in a **real
`bcf9a23` worktree** rather than restore-in-place: denominators `[]` at base → `[23, 27]` at tip.

### Budget

Ceiling 1 ratified **300 → 400** (**D-079**); measured **391**, cumulative **499** across both
rounds. Round allowance **120 → 140** and ceiling 2 **40 → 55** (**D-082**). **Seventh under-set
ceiling of this sprint and the seventh that was mine.** The card's own question exposed a real
instrument defect — the F-050 command and the ceiling sentence contradicted each other, reporting
**1171** against a ceiling meaning 391 — now fixed sprint-wide.

### Conduct

Kept **four** failed or invalidated measurements beside their corrections, including a split-gate run
whose missing `--basetemp` parent made every test error in setup and reported **382 instead of 453** —
*an infrastructure failure wearing the costume of a wiped test file*. It **declined to touch
`DECISIONS.md`** and escalated the `PRODUCT_CONTRACT` § 7 tension instead of resolving it itself.
REV-102 recorded predictions before reading any diff hunk and **was wrong on three of them in
writing**.

---

## C-103 — dispatched and complete, under review · 2026-08-28

**Branch** `card/C-103-f142-replay-expectation`, tip `89afc11`, base `ad62338`, worktree `C:/t/c103`.
**Zero production lines** — `git diff --numstat ad62338 89afc11 -- src` is empty.

**New gold-readers baseline: `0 failed / 456 passed / 8 skipped / exit 0`** (from 2/453/8/exit 1).
**Every charter carrying the "this selection exits 1 at base, and that is correct" warning is now
stale** and must be updated. G9: same file, same selection, **2 failed → 0 failed**.

**It corrects F-142**, subject to REV-103 confirming. F-142 attributes this payload's
`review_required` to C-041a's branch alone; mutation **B1** shows that is true of the **channel** but
not the **status**, because the **F-094 incomplete-core cap (C-072)** independently demotes the leg
on its three unmatched anchors. **Two independent rules, where the finding said one.** B1's report is
kept beside B2's rather than replaced — which is why it was findable at all.

**A new restore trap, worth naming:** `git checkout -- cases.json` in a restore path reverted the
card's own **uncommitted fixture edit** along with the mutation, so mutated production ran against an
uncorrected fixture and produced 20 mostly-`KeyError` failures that looked like a result. **A restore
that reverts more than it mutated is a measurement failure, not a test result** — the same family as
the missing `--basetemp` parent and the wrong interpreter. Invalid run kept at `10`, re-measured at
`11`.

---

## T-107 — NO-GO, and the blocker changed character · 2026-08-28

Full assessment: **`docs/pwml_recovery_sprint/T107-READINESS.md`**, measured at the merged tip.

**Twelve of thirteen gate conditions hold.** The failure is condition 8 — the pinned model —
and it is a configuration and authorization question, **not** engineering.

`.env` pins all nine OpenRouter slots to `deepseek/deepseek-v4-flash`, which a read-only models check
prices at **$0.0868/M prompt, $0.1736/M completion — paid**. The T-101/T-103 authorization's ≈$0
basis rested on every slot being **`openrouter/free`**; `.env` no longer matches, and `.env` is
untracked so the change is unattributable through git. **LM Studio cannot substitute** — it serves
`glm-4.6v-flash`, which is neither the pinned model nor the configured `LOCAL_MODEL`, and using it
would be a **fallback model** (forbidden this wave) that destroys comparability with T-104/T-105/T-106.

**At the merged tip:** Priority 1 raw **8** / accepted **8** → `FAIL` (7 would be
`PASS_WITHIN_VARIANCE`; T-104 and T-105 both scored 7) · Priority 2 **`CONDITIONALLY SATISFIED`**,
9 of 20 legs eligible, all 11 `NOT EVALUATED` carrying the same D-067 precondition-3 reason ·
Priority 3 `PASS` · Priorities 4/5 unmoved at `0/8` and `0/2`. **`LpxH` confirmed still counted:
`PMC12444477/strict` = 8 findings including `LpxH`, `Unknown` gone — 9 → 8, never 9 → 7.**

**What changed:** the wave opened with condition 9 recorded as *"not met and not reachable by any
engineering in this sprint"*. **B and D are now answered**, both instrument cards are merged with
independent approval, and the remaining blocker is **one product decision** rather than a sprint's
worth of work.

---

## D-080 RATIFIED — accepted coverage is formally defined · 2026-08-28

The product owner ratified D-080 as the ruling. **C-102 needs no change; the shipped implementation
already computes exactly the ratified definition**, verified against `contract_accepted_coverage` in
`src/t2pw/bench/acceptance.py` rather than against the card's report:

```
eligible_anchors     = raw_anchors − case_scoped_forbidden_identifiers
accepted_numerator   = | matched ∩ eligible_anchors |
accepted_denominator = | eligible_anchors |
accepted_coverage    = accepted_numerator / accepted_denominator
```

with `raw_ratio` (copied verbatim from the frozen block, never recomputed), `raw_matched` and
`raw_denominator` preserved beside them, and an empty eligible set handled explicitly as
`accepted_ratio = None` with its own state rather than as a coverage success.

**The four reasons the product owner gave, and each is measured rather than argued:** accepted
coverage stays within `[0, 1]` · exporting a forbidden identifier earns **zero** coverage credit ·
the denominator-only behaviour that produced ratios above 1 on **nine committed legs** is avoided ·
all raw evidence and every Priority-1 penalty are preserved.

**Unchanged by ratification:** Priority 1 (`LpxH` still counted, PMC12444477 still 9 → 8) · the
`0.5` threshold · the gold · **D-081** (Ruling A still moves neither Priority 4 nor 5) · the seam
staying in `bench/`.

**Record correction:** `PRODUCT_CONTRACT.md` § 7 already stated the both-sides rule and conflicted
with the letter of locked D-072. REV-102 escalated it rather than letting the diff read as an
improvised product decision. **The conflict now resolves in § 7's favour.**

---

## T-107 LAUNCHED — `runs_verify/2026-08-28_1816` · 2026-08-28 18:17

**Authorized by D-085.** Launched **once**, under the release procedure the product owner specified,
at integration tip `ae66b52` (local = `origin/` = `ls-remote`).

### The procedure, and why each step exists

| Step | Result |
|---|---|
| **1. Fresh milestone identity, stage-only** | `runs_verify/2026-08-28_1816` created in 3.3 s. `--stage-only` returns **before the run loop**, so no manifest row and no leg directory can exist regardless of cache speed. Zero LLM legs by construction |
| **2. `--verify-plan` against that exact staged directory** | **`verdict: OK`** · `cases checked: 10   search calls: 0` · **all 10 `[pinned_override]`** |
| **3. Continuity proof** | `find_resumable()` called directly — returned **the exact verified path**, `20` plan pairs, **`20` pending**, **`0` legs present** |
| **4. Continue that directory WITHOUT `--fresh`** | Runner confirmed: *"CONTINUING the incomplete run 2026-08-28_1816 (no --fresh given) · already recorded : 0 paper+mode run(s) -- these are skipped · still to do : 20"* |

**The runner's own hint is a trap and step 4 exists to defeat it.** `--stage-only` prints
*"then run it: rerun the same command WITHOUT --stage-only"* — and that command **still carries
`--fresh`**, which would create a new, **unverified** directory and silently discard the staging just
certified. A peer session independently flagged the sibling hazard (`batch_run.py` skips finished
pairs without `--fresh`, yielding a partial cohort that looks complete) and recommended `--fresh` as
the fix; **here that would have been the defect, not the cure.** The measured discriminator is
`already recorded : 0` — the skip hazard needs *finished* pairs, and a stage-only directory has none.

### Cost

**T-105 recorded no token usage anywhere in its artifacts**, so no estimate can be built from actual
usage. Stated as a **bound from measurable inputs**, not a prediction:

* 10 papers, **592,813** source characters, mean **59,281**/paper ≈ **14,820** tokens per full text
* prices `$0.0868`/M prompt, `$0.1736`/M completion (read-only `GET /api/v1/models`)

| full-text-equivalent passes per leg | total |
|---:|---:|
| 20 | **$0.62** |
| 40 | **$1.23** |
| 80 | **$2.47** |
| 120 | **$3.70** |
| **162** | **$5.00 — the ceiling** |

**$5 is reached only at ~162 full-text passes per leg (~48M prompt tokens across the run)** — about
18 per model role. Plausible range **$0.62–$3.70**. **Launched inside the ceiling.** No spend
telemetry exists to abort on mid-run; final spend is to be read from the provider and recorded.

### Run parameters

`topics_t104.txt` (the ratified pinned set) · `--modes strict,research` · per-leg `--timeout 1800` ·
internal `--deadline 5.5h` · wrapper hard ceiling **21600 s (6 h)** · `--heavy-lock T-107` ·
`deepseek/deepseek-v4-flash` on all nine slots, `LLM_TEMPERATURE=0`, `.env` **unmodified**.

Measured predecessors: T-104 **5.44 h**, T-105 **4.85 h**. The internal deadline stops *starting* new
pairs rather than killing a running one, so a deadline stop is a clean partial that the same command
can continue — **continuing an interrupted run is the same run identity and is not a re-draw.**

### Standing constraints

**Once only.** The first valid official draw is scored and preserved. **A Priority-1 result of 7 is
`PASS_WITHIN_VARIANCE` and will not be re-drawn** (D-073). A leg is never repeated because its draw
is unfavourable; something not observed is reported as *"not observed"*, never chased.

Peer `project14-t2pw-14` re-confirmed — verified rather than recalled — that it holds no lock, runs
no Python, will not push, and has nothing touching `runs_verify/`, the caches or `.env`.

### The wave's lesson, in its sharpest form — and it recurred during the launch

A peer session flagged the `batch_run.py` skip hazard and prescribed `--fresh`. The hazard is real;
the prescription would have **destroyed the verified staging directory**. Its own diagnosis
afterwards is the best statement of this wave's lesson anyone produced:

> **I let a heuristic stand in for the thing being claimed.** The hazard is that `batch_run.py`
> silently skips *finished* pairs — but the property actually being protected is *"know how many
> pairs are already recorded before you continue."* `--fresh` guards a **proxy**; asserting
> `find_resumable()` returns the verified path with 20 pairs, 20 pending, 0 legs present guards the
> **property**. The ledger line I cited was written for T-106, where the risk was resuming *into* a
> directory with finished pairs; the staged-identity workflow inverts it, and I did not check which
> situation applied before prescribing.

**That is F-144 in a process rule instead of a test.** The same shape as: a guard satisfied by a
different code path · a probe passing its own positive control while asking the wrong question · a
test named for a function it never called · and a scan for the string `1.2000` standing in for the
predicate `> 1`.

**And the three-caps result is the same shape a fourth time, in a finding.** F-142 said **one** rule
held the leg out of `release_ready`; C-103 measured **two**; REV-102/REV-103 measured **at least
three**, each independently sufficient and **none necessary**. **Every one of those counts was true.
Each was wrong about what it was evidence for.** *"How many things are holding this"* reads as
settled and is not.

**The standing rule, stated once:** where a claim is load-bearing, assert **the property**, on the
**production path**, and have someone who did not write the assertion **try to defeat it**.

---

# WAVE ORCH-716 — T-107 triage and close-out · 2026-08-29

**T-107 was NOT rerun and no leg of it was repeated.** The official run
`runs_verify/2026-08-28_1816` stands exactly as scored: **NOT ACCEPTED, on Priority 2 alone.**

Full classification: **`docs/pwml_recovery_sprint/T107-TRIAGE.md`**.

## Takeover verification, all confirmed

| Check | Result |
|---|---|
| local = `origin/` = `git ls-remote` | ✔ all three `36f773c` at takeover |
| `main` local `7531692` / remote `03f1af5` | ✔ untouched, neither ref written |
| merge in progress / staged | ✔ none / none |
| heavy lock `C:/t/heavylock` | ✔ absent at takeover; acquired and released by every job after |
| sprint-owned Python | ✔ zero |
| IDE processes | ✔ exactly two `ms-python.isort`, matched on command line, never targeted |
| `streamlit_app.py` | ✔ uncommitted, 35 ins / 2 del, `sha256:47e4fafa789d359d…` — re-verified after every commit |
| **SMOKE** | ✔ **473 passed** |
| **gold-readers** (22 files) | ✔ **456 passed / 8 skipped / exit 0** |

Peer `project14-t2pw-41` contacted before the branch, the lock or any job was claimed. It stood
down explicitly: read-only reconnaissance in the same repo, holds no lock, no worktree, no
uncommitted work, no intent to push. The dirty tree it observed predates both of us.

## The result of the triage, in one line each

| Item | Classification |
|---|---|
| `PMC13231680/strict` empty pathway | **CORRECT — not a defect.** Gold: *"the correct pipeline outcome is an empty pathway plus a rejection reason."* **T-105's PASS was the false positive**, already registered as **F-100** |
| Priority 2's single row | **`product_contract_violation`** — **F-146**, F-100's open remainder on the research leg |
| `PMC12452463/strict` + `PMC12180156/strict` contract failures | **one shared seam**, **`product_contract_violation`** — **F-147**. **Registered, deliberately NOT chartered** |
| Three timeouts | **`product_contract_violation`** — **F-148**. **F-092 defect 3 CLOSED** by the same measurement |
| Harness scoring a correct negative-control outcome as `FAIL` | **`policy_disagreement`** — decision packet, product owner |
| Non-vacuity audit | **no defect** — **F-149**, both files pin non-vacuously |

**No gold edit is proposed.** The gold was right on every case examined in this wave.

### The C-099 hypothesis is falsified for two of the three legs it was raised about

`PMC12452463/strict` and `PMC13231680/strict` **already failed in the T-106 artifacts
(`runs_verify/2026-08-24_1428`)**, three days before C-099 merged (`9e4a28a`, 2026-08-27) and four
before C-100 (`8e5d549`, 2026-08-28). `PMC12452463` **improved** at T-107, 7 contract errors to 3.
Only `PMC12180156/strict` genuinely turns at T-107, and F-147 shows its mechanism is not species
preservation either.

### The refusal that matters most

**F-147 is a genuine contract violation and no card was chartered for it.** It fails exactly two
legs; fixing it alone would make both pass and export gold-forbidden content — `enterobactin
synthase complex`, `RyhB`, an efflux step the paper never describes, and a `ferrochelatase reaction`
built on `protoporphyrin IX`, which the gold certifies occurs **zero times** in its paper. **The
earliest unsafe seam is Stage-1 extraction, not the driver.** Merge rule 6.

## Cards

| Card | State |
|---|---|
| **C-104** — D-083's two carried follow-ons (prove C-102's deep copy; abort the split gate on `errors > 0`) | **CHARTERED and DISPATCHED.** Worktree `C:/t/c104`, branch `card/C-104-d083-followons`, base `36f773c`. **Changes no production line** |

## Run ledger — this wave

| Job | Task | Purpose | Exit | Survivors | Cleanup |
|---|---|---|---|---|---|
| `01-smoke` | ORCH-716 | takeover gate (unpinned, no lock — superseded by `11`) | 0 | **0** | success |
| `02-goldreaders` | ORCH-716 | takeover gate, pinned, lock held | 0 | **0** | success |
| `03-nv-baseline` | ORCH-716 | non-vacuity baseline, 42 passed | 0 | **0** | success |
| `04-nv-m1-constant` | ORCH-716 | mutation M1, 14 failed | 1 | **0** | success |
| `05-nv-m2-arma-neutered` | ORCH-716 | **mutation M2, the finding**, 13 failed | 1 | **0** | success |
| `06-nv-m3-c072-neutered` | ORCH-716 | mutation M3, 5 failed | 1 | **0** | success |
| `07-nv-restore-verify` | ORCH-716 | byte-exact restore verified, 42 passed | 0 | **0** | success |
| `08-openrouter-usage` | ORCH-716 | D-086, superseded by `09` (unredacted) | 0 | **0** | success |
| `09-openrouter-usage-redacted` | ORCH-716 | D-086 measured usage, key label and account id redacted | 0 | **0** | success |
| `10-stale-verdict-probe` | ORCH-716 | **F-147's decisive measurement** | 0 | **0** | success |
| `11-smoke-pinned` | ORCH-716 | SMOKE re-run pinned, lock held, `violations: []` | 0 | **0** | success |
| *(uncertifiable)* | ORCH-716 | first M2 attempt, invalid label -> empty `--json`, **no artifact** | 1 | **0** | success |

`g11_evidence.py check --task ORCH-716`: **11 artifacts, 0 non-compliant.** The uncertifiable run is
recorded above and **not counted**; it is kept rather than tidied away.

**Two process deviations of my own, corrected and recorded rather than quietly fixed:**

1. **`01-smoke` ran plain `pytest` with no `--pin-verdict` and no `--heavy-lock`.** TEST_MATRIX § 0
   requires both on a gate. Re-run correctly as `11-smoke-pinned` (473 passed, `violations: []`,
   lock acquired and released). The original is kept.
2. **The first M2 run used an uppercase label**, was rejected by `g11_evidence.py next`, and the
   empty capture became `--json ""`. **A job with no G11 artifact is not a passed test**, so it was
   re-run under `05`. This is the charter's named label trap in a variant that produces an *empty*
   path rather than error text.

## D-086 — usage MEASURED, and what it can and cannot say

Read live from the OpenRouter account (`evidence/orch716_openrouter_usage.py`, report `09`):

| Field | Value |
|---|---|
| `usage_weekly` | **1.769355221** |
| `usage_monthly` | 6.483844303 |
| `usage_daily` | 0 |
| `limit` / `limit_remaining` | 75 / 68.516155697, `limit_reset: monthly` |
| `is_free_tier` | **false** — corroborates T107-READINESS condition 8 |
| account `total_credits` / `total_usage` | 933.65 / 894.35836073 |

**These are cumulative ACCOUNT totals. The pipeline sends no run identifier, so the provider cannot
attribute any part of them to T-107 — this is D-086's gap confirmed from the other side, not closed.**
T-107 is the only substantial live model run inside the weekly window, so **`$1.77` is a measured
upper bound on its spend, not a measurement of it**, and it sits inside the `$0.62–$3.70` pre-run
estimate. **Any tighter figure would be an estimate and is not offered.**

The probe redacts the key `label` and `creator_user_id` on the response path, so the committed log
carries no account identifier. The key itself is never printed, written or sent anywhere but
`openrouter.ai`.

**Usage did not constrain this wave at any point.** No analysis was shortened, no model call
declined, and no decision turned on cost.

## Evidence committed

`T107-TRIAGE.md` · `prompts/C-104.md` · `evidence/orch716_nonvacuity_predictions.md` (written
before the mutations, unedited) · `evidence/orch716_nonvacuity_results.md` ·
`evidence/orch716_openrouter_usage.py` + `.log` · `evidence/orch716_stale_verdict_probe.py` +
`.log` · 11 G11 reports · 7 pin verdicts, all `violations: []`.

Also committed: **eight REV-101 G11 reports from the previous wave that were left uncommitted**
(`40..47-r3-*`). They validate clean and I did not produce them. A G11 report that exists only in a
working tree is one session away from being lost, which this sprint has already paid for twice.

---

## Wave ORCH-716 addendum — C-105 chartered after the adjudication

**C-105** (`card/C-105-audit-actor-evidence`, worktree `C:/t/c105`, base `36f773c`) is chartered
and dispatched for **F-146**. It is the only card in this wave that touches production.

It was NOT chartered in the first pass. `T107-TRIAGE.md` § 2 recorded the judgement that the
remedy was too diffuse. The independent adjudication then supplied the audit stage's **written
motive** for attaching the enzyme and the **exact policy hole** in `apply_patch_with_policy` that
admits an `add` to `/processes/reactions/N/enzymes/-` on confidence alone. **Both were verified by
me against the artifacts before the card was written.** The superseded paragraph is left standing
beside its correction rather than rewritten.

**F-150** is registered as a `gold_data_defect` with an exact proposed edit and a four-step A/B
plan, and is **NOT applied** — gold-change authority is the product owner's.

Agent register for this wave:

| Agent | Role | State |
|---|---|---|
| `pwml-bio-auditor` | independent read-only adjudication of items 1-6 | **COMPLETE.** Status requested at the liveness checkpoint; reported in full. Ran no job, took no lock, wrote no file |
| general-purpose (C-104) | implementer, worktree `C:/t/c104` | dispatched |
| general-purpose (C-105) | implementer, worktree `C:/t/c105` | dispatched |

---

## ORCH-716 — C-104 MERGED, and the census fix I proposed was wrong

**C-104 merged at `57e604d` with `--no-ff`.** REV-104 **APPROVE**, 15 G11 reports, 0 non-compliant.
Post-merge **SMOKE 473 passed**, zero survivors, lock acquired and released (merge rule 10).

| Merge gate | Result |
|---|---|
| 1 dependency merged | none |
| 2 diff within boundary | ✔ no `src/` line; `acceptance.py` byte-identical base and tip |
| 3 focused tests | ✔ **delta zero** — 2 failed / 12 passed at base AND tip |
| 4 existing affected tests | ✔ the two reds are **F-151**, pre-existing, not this card |
| 5 independent review of the actual diff | ✔ **APPROVE**, six adversarial mutations by a non-author |
| 6 no biological gate weakened | ✔ the diff contains no production line |
| 7 preserves `review_required` | n/a |
| 8 no exporter repairs biology | n/a |
| 9 G9 behavioural base failure | ✔ both halves |
| 10 SMOKE after merge | ✔ **473 passed** |
| 11 test-process lifecycle | ✔ every job zero survivors, cleanup success |

**The A1 result is the one worth keeping.** REV-104 rewrote the author's identity assertions as
equality, deleted the consequence block, applied R5 — and test 4 went **GREEN**. The equality form
is precisely the vacuous guard F-144 names, and the author did not write it.

**Two boundary notes against my own charters, both raised by the agents rather than by me:**

1. **Neither C-104.md nor C-105.md existed in its own worktree.** I cut both worktrees at
   `36f773c` and then wrote the cards into the primary checkout, committing them later. Both
   implementers read the card from the primary checkout read-only and **both flagged it**. **Commit
   the card before cutting the worktree**, or the worktree cannot see its own charter.
2. **C-104's ownership table named the wrong file for the attack-set entry.** The attack set lives
   in `evidence/c102_mutation_attack.py`, which the table did not list, while D-083's own text and
   the card's prose both required the entry. **My table was the error.** The implementer flagged it
   rather than treating the table as authoritative; the reviewer independently judged it not a
   breach (15 lines, purely additive).

**F-151's proposed fix was wrong and REV-104 corrected it.** I proposed `>= 62`, generalising from
line 325's idiom without checking what lines 347 and 461 are *for*. They pin **derived** quantities
against the census — a `>=` would let a leg join the corpus and go unvisited unnoticed. Revised:
**re-pin to 72 and record why it grew**, in its own card that also makes the mutation harness
runnable and D-084-compliant. **Handed forward, not chartered here.**

**F-152** registered: C-104's widened guard can abort a green file, because the pre-existing count
parse at line 52 reads all of stdout. Inert before this card, fatal after. Outside C-104's boundary
and correctly not fixed in it; no live exposure on the real selection.

---

## ORCH-716 — REV-105: CORRECTION ROUND, and the finding is the one I most feared

**C-105 is NOT merged.** REV-105 returned **CORRECTION ROUND** (round 1 of the two that are
automatic). 16 G11 reports, 0 non-compliant, every job zero survivors.

**The core is sound and the card is not rejected.** B2, B3, B5, B6, B7, R4, R13 and R16 all hold and
the reviewer reproduced each independently. The author's structural purity property survived
adversarial scrutiny: the reviewer verified from the **signature** that
`_unevidenced_actor_role_rejection(op)` cannot reach the payload, then A/B'd the exact
inhibitor-promotion case and got identical refusal whether the protein is present or absent. That
was the right design and it does fix T-107's Priority-2 defect.

### B1 fails, at scale — and it is a citation problem as much as a code problem

`_span_licenses_actor`'s docstring claims *"the same comparison discipline
`bench.semantic_production._actor_named_in_span` uses"* and that *"no substring, per-token union or
edit distance is involved"*. **That function is exactly a per-token union** —
`bool(wanted & seen)`, `semantic_production.py:399-405` — and its own docstring records why:

> *"One SHARED token, not the whole name -- the half that stops it over-firing. … `ALAS2 complex`
> cites *"ALAS2 mediates …"* … **and a whole-name rule demotes five of the 21 legs over that.**"*

**C-105 implemented the whole-name rule and attributed it to the function that measured it wrong.**

| Measurement | Value |
|---|---|
| legitimate evidenced cases refused | **12 of 29** |
| real corpus rows refused on lexical artefacts | **258 of 692 (37.3%)** |
| refused **only** by the whole-name rule the repo's own rule would license | **150** |
| refused because the cue vocabulary is too narrow | 108 |

Refused examples **lifted verbatim from committed payloads**, not hypotheticals:
`MenD complex` ← *"MenD catalyses the first irreversible step"*;
`UDP-N-acetylglucosamine acyltransferase` ← *"LpxA … catalyzes the reversible acylation"*;
and `ALAS2 complex` ← *"ALAS2 mediates …"*, which is the example `_actor_named_in_span`'s **own
docstring** gives. `"NDM-1-catalyzed hydrolysis"` is invisible because `_registry_normalize`
**deletes** hyphens, making it `ndm1catalyzed`.

**This is exactly the B9 trap the review criteria were written to catch**, and it is why B1 was
written as *"the single most important item in this review"*. A guard that refuses more looks like
the safe direction. This one refuses legitimate evidenced enzyme attachment on **three live
production passes**, and it would have surfaced later as fewer reactions across many papers with
nothing to attribute it to.

### Blast radius is wider than the card flagged — four callers, not one

| Caller | Line | Actor-role work |
|---|---|---|
| `focused_repair.py` | 215 | all four repair passes, incl. `modifier_enzyme_repair` |
| `pathway_curator.py` | 348 | prompt: *"propose missing transporters"* |
| `gap_resolver.py` | 3341 | — |
| `pipeline.py` | 82 | the audit stage itself |

Two aggravating facts the reviewer measured: `_run_modifier_enzyme_repair` draws its value names
from the **declared entity registry** — exactly the `MenD complex` spellings — while the model's
`reason` uses paper symbols, which *is* the 150-row bucket; and `pathway_curator`'s schema documents
`evidence` as *"One sentence explaining why this patch is proposed"*, **a rationale, not a quoted
span**. The guard changes that field's required meaning without changing any prompt that fills it.
Prompts are outside C-105's boundary, so this is **routed as a follow-on, not fixed in the card.**

**Six existing test files exercise `apply_patch_with_policy` and none is in SMOKE or gold-readers**,
so merge rule 4's "existing affected tests" were never run by the card's own evidence. The reviewer
ran 13 such files: **145 passed** at tip. No regression, but the gap in the card's evidence is real.

### B9 — the deterministic half passes, the fresh-run half is honestly unmeasured

All 17 committed `PMC12452463` / `PMC12180156` strict artifacts re-scored through
`evaluate_production_semantics` at base and tip: **identical digest `f90111274f84dff6…`**. No
committed leg flips, and an exhaustive caller grep shows the guard is unreachable from any scoring
path. **Whether a fresh end-to-end run of either leg would flip is NOT measured** — it needs live
LLM spend the reviewer had no authority for. Its reasoning that a flip is implausible (the guard
only ever rejects, and both legs fail on entity-identity grounds an enzyme refusal does not satisfy)
is recorded as **a prediction to check before the next T-run, not as a result.**

### Two residual routes recorded so they are not rediscovered as defects

1. `replace /processes/reactions/N/modifiers/M/role = "catalyst"` — in-place promotion of an
   existing inhibitor. **Deliberately excluded**, and the reviewer agrees the exclusion is
   defensible for this card. But it confirmed **by execution** that a model refused on the `add`
   path can take the `role` path instead, because the rejection reason tells it so.
2. `add /processes/reactions/-` carrying an invented enzyme is **accepted**. The card's comment
   calls this "major-topology territory"; it is not protected —
   `enforce_major_topology_threshold` is `True` at exactly one call site
   (`interactive_curator.py:507`) and defaults `False` at every automated caller.

### Correction round 1 dispatched

C1-C8 sent to the implementer: adopt the repo's shared-token rule, map separators before the
boundary test, widen the cue vocabulary, **fix the false citation**, drop the rename over-reach,
correct the major-topology claim, add the four-caller and affected-seam evidence, and replace the
preservation control — which used the one-character name `P`, **the single shape a whole-name rule
always handles, which is why it passed while the class it stood for was broken.**

**Review-mandated work never charges a ceiling** (D-076 A1, D-082). A third round would need
explicit authority.

---

## ORCH-716 — C-105 MERGED. T-107's Priority-2 defect is closed at the seam that caused it.

**Merged `afb0541` with `--no-ff`**, three commits (`28d8443` → `30d46d9` → `29abe85`).
**REV-105 APPROVE after three rounds**, 47 G11 reports, every number reproduced independently by
a reviewer that had no edit tools.

Post-merge: **SMOKE 473 passed** · **gold-readers 456 passed / 8 skipped / exit 0**, both zero
survivors, lock acquired and released.

| Merge gate | Result |
|---|---|
| 1 dependency merged | none |
| 2 diff within boundary | ✔ `apply_audit_patch.py`, its new test, G11 reports — nothing else |
| 3 focused tests | ✔ **35 passed** at tip |
| 4 existing affected tests | ✔ affected seam 13 files **145 passed**; Chunk C **109**; both gates exact |
| 5 independent review of the actual diff | ✔ **APPROVE**, three rounds, adversarial throughout |
| 6 no biological gate weakened | ✔ **441 added, 0 removed.** It *strengthens* a gate |
| 7 preserves `review_required` | ✔ B6: a refused row survives verbatim, payload **byte-identical**, `committed: true`, no `diagnostic_only` |
| 8 no exporter repairs biology after freeze | n/a — this seam is pre-freeze |
| 9 G9 behavioural base failure | ✔ 16 failed at base → 35 passed at tip, on `assert 1 == 0` over `accepted_count` |
| 10 SMOKE after merge | ✔ **473 passed** |
| 11 test-process lifecycle | ✔ every job zero survivors, cleanup success |

**B9, deterministic half:** all 17 committed `PMC12452463` / `PMC12180156` strict artifacts re-score
to digest `f90111274f84dff6…` at base **and** tip. **No committed leg moves.** The fresh-run half is
honestly unmeasured — it needs live LLM spend and is recorded as a prediction to check before the
next T-run, not as a result.

### What the three rounds actually bought

**Round 1 caught the thing the criteria were written for.** The guard implemented **whole-name**
matching while citing, as its authority, `_actor_named_in_span` — which implements a **shared-token**
match, and whose own docstring says a whole-name rule *"demotes five of the 21 legs"*. It refused
**12 of 29** legitimate evidenced cases and **258 of 692** real corpus rows on lexical artefacts,
across **four** production callers, not the one the card named. Fixed by adopting the calibrated
rule and pinning the two implementations against each other; the reviewer's **byte-identical**
battery went **12 refusals → 1**, and the whole-name bucket **150 → 0**.

**Round 3 I authorised deliberately, and the corpus vindicated it.** `mediat` in the catalysis
family let the defect class back in by paraphrase — a span saying the protein was *inhibited*
licensed it as the reaction's *catalyst*. Not synthetic: the reviewer's A/B found **three distinct
committed spans** doing exactly that, one more than the author reported —

* `EntB` ← *"2,3-DHB is a **competitive inhibitor** of apo-EntB isochorismatase activity"*
* `ALAS2 complex` and bare `ALAS2` ← *"**inhibiting** the translation of erythroid
  δ-aminolevulinic acid synthase (ALAS2), a key enzyme"*

The fourth newly-refused span is collateral on a 140 KB discussion section where `mediat` matched
inside *"intermediate"* and `suppress` inside *"suppressor mutations"* — **two false positives
cancelling**, right outcome for the wrong reason, quantified at **0 of 397** accepted rows relying
on that stem alone.

### Routed to a follow-on card — not fixed, and deliberately so

All permissive-direction, **all strictly better than base, which accepts every one unconditionally
on confidence alone**:

1. **Inhibition near-synonyms defeat the contra-cue** — 11 of 12 tested (`blockade`, `impairment`,
   `disruption`, `reduction`, `loss`, `silencing`, `sequestration`, `depletion`, `ablation`,
   `interference`, `quenching`). `reduction of` is worst: it is itself a *catalysis* cue.
2. Passive-with-agent fires when the agent is not the actor.
3. 17 ordinary English `-ase` words over-accept, including a plural bypass
   (`purchases`, `showcases`, `staircases` have no stoplist entry).
4. Transport family has no enzyme-noun rule, so `flippase MsbA` refuses.
5. Role `cofactor` refuses.
6. `mediat` matches inside *"intermediate"*; anchor it.
7. **Pin scope**, recorded in the pin's own docstring: it binds `_identifying_match_tokens` and
   `_match_fold` but **never calls `_span_licenses_actor`**, so a substring regression in the
   consumer leaves it green — the F-079 EntE case is what catches that. A future reader is told not
   to over-trust it.

Also for that card: **62 of 692 spans exceed 5,000 characters, max 176,375.** Oversized
actor-evidence spans are upstream of C-105 and outside its boundary.

### The reviewer corrected its own measurements twice, and kept both

A `FAILED (\S+)` regex split a parametrized test id and reported a mutation as 3 reds instead of 1;
and a first Pgp diagnosis matched the wrong row. **Both preserved beside their corrections**, along
with two staging slots that produced no artifact. That is the practice working.
