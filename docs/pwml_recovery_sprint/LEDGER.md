# PWML Recovery Sprint — Ledger

**Single source of truth for task state. Written by the Lead Orchestrator only.**

States: `BLOCKED` → `READY` → `IMPLEMENTING` → `REVIEW` → `CORRECTION` → `INTEGRATION`
→ `BATCHED_VALIDATION` → `ACCEPTED`. Also terminal: `REJECTED`.

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
| H-006 | Wrapper report schema version + build identity | `READY` | H-003 ✔, H-004 ✔ | read at dispatch | `agent/h06-wrapper-report-identity` | assigned at dispatch | `evidence/bounded_run.py`, `evidence/bounded_run_selftest.py`, `evidence/g11/README.md` disclaimer — **temporary ownership grant, this task only** | `pwml-reviewer`, then `pwml-test-runner` | full synthetic lifecycle suite | smoke | — | — | — |

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

**H-001 and H-002 are merged and ACCEPTED; the replay merge gate is green on a frozen
cohort.** Wave A0 and C-010 are therefore no longer blocked on them — they are blocked
only on **product-owner approval**, which has not been given. Nothing here is dispatchable.

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-010 | p01 stale positional index | `BLOCKED` | INIT-001 | — | `agent/p01-stale-index` | `.claude/worktrees/p01-stale-index` | `strict_quarantine.py` :: `_surviving_processes`, `_degree_zero_exports`, `quarantine_and_close`; `tests/test_strict_quarantine.py`; `tests/test_strict_quarantine_real_artifact_replay.py`; `docs/change_log.md` | C-041 impl | A, E | smoke | — | M1 | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-011 | p00a canonical-freeze seam | `BLOCKED` | H-001 ✔, H-002 ✔ | — | `agent/p00a-freeze-seam` | `.claude/worktrees/p00a-freeze-seam` | `streamlit_app.py` :: `run_post_pipeline_sbml_artifacts` | C-012 impl | D | smoke + D | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-012 | p00b driver seam | `BLOCKED` | INIT-001 | — | `agent/p00b-driver-seam` | `.claude/worktrees/p00b-driver-seam` | `driver.py` :: `_drive` → `_finalize_*` | C-011 impl | B + golden | smoke | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-013 | p04a two versioned hashes | `BLOCKED` | INIT-001 | — | `agent/p04a-hash-module` | `.claude/worktrees/p04a-hash-module` | NEW `pipeline/canonical_hash.py`; `gate_reports.py` :: `payload_sha256`, `stamp_report`, `gate_verdict` | C-020 impl | smoke | smoke | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-014 | p03a LLM request timeout | `BLOCKED` | INIT-001 | — | `agent/p03a-llm-timeout` | `.claude/worktrees/p03a-llm-timeout` | `llm/client.py` :: `OpenAI(...)`, `chat_detailed`, `chat_with_tools` | C-032 impl | A, C | smoke | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-015 | p20 lineage schema | `BLOCKED` | INIT-001 | — | `agent/p20-lineage-schema` | `.claude/worktrees/p20-lineage-schema` | NEW `pipeline/lineage.py` | C-038 impl | new | smoke | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-016 | p30 RAG stopping policy | `BLOCKED` | INIT-001 | — | `agent/p30-rag-stop-policy` | `.claude/worktrees/p30-rag-stop-policy` | NEW `rag/loop_policy.py` | C-043 impl | new, C | smoke | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-017 | p40 semantic production module | `BLOCKED` | INIT-001 | — | `agent/p40-semantic-module` | `.claude/worktrees/p40-semantic-module` | NEW `bench/semantic_production.py` | C-056a impl | B | smoke | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |
| C-018 | p50 cofactor / assay-reporter policy | `BLOCKED` | INIT-001 | — | `agent/p50-cofactor-classifier` | `.claude/worktrees/p50-cofactor-classifier` | NEW `pipeline/cofactor_policy.py` | R-003 | new, C | smoke | — | — | **product-owner approval** (H-001 ✔ / H-002 ✔ — cleared `2026-08-05`, no longer blocking) |

## Wave A1

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-020 | p06a biological-equivalence comparator | `BLOCKED` | C-013 | — | `agent/p06a-equiv-comparator` | `.claude/worktrees/p06a-equiv-comparator` | NEW `pipeline/canonical.py` | C-013 impl | new, D | smoke + D | — | M3 | C-013 |
| C-021 | p31 RAG graph-delta validation | `BLOCKED` | C-015 | — | `agent/p31-rag-graph-delta` | `.claude/worktrees/p31-rag-graph-delta` | NEW `rag/graph_delta.py` | C-016 impl | new, C | smoke | — | — | C-015 |

## Wave B

| ID | Task | Status | Deps | Branch | Ownership boundary | Reviewer | Focused | Bench |
|---|---|---|---|---|---|---|---|---|
| C-030 | p04b hash wiring | `BLOCKED` | C-011, C-013 | `agent/p04b-hash-wiring` | `streamlit_app.py` :: `freeze_canonical_payload` | C-052 impl | D | — |
| C-031 | p02 quarantine artifacts | `BLOCKED` | C-012 | `agent/p02-quarantine-artifacts` | `driver.py` :: `_add_common_artifacts`, `_add_identity_artifacts` | C-053 impl | B | M1 |
| C-032 | p03b deadline + checkpoints | `BLOCKED` | C-012, C-014 | `agent/p03b-deadline-module` | NEW `pipeline/deadline.py`; `runner.py` :: `_timeout_row`, `launch_child`, `child_command`; `_finalize_timeout` | C-042 impl | B | — |
| C-033 | p10 identity hydration | `BLOCKED` | H-001 ✔, H-002 ✔ | `agent/p10-identity-hydration` | `src/t2pw/mapping/map_ids.py` :: 2 fns; `src/t2pw/pipeline/entity_identity.py`; NEW `src/t2pw/mapping/uniprot_evidence.py` (**not** `src/map_ids.py`) | C-044 impl | C | M2 |
| C-034 | p21 lineage: extraction | `BLOCKED` | C-015 | `agent/p21-lineage-extract` | `extraction/extract.py` | rotate | A | — |
| C-035 | p22 lineage: RAG | `BLOCKED` | C-015 | `agent/p22-lineage-rag` | `rag/synthesize.py`, `rag/admission.py` | rotate | C | — |
| C-036 | p23 lineage: audit | `BLOCKED` | C-015 | `agent/p23-lineage-audit` | `curation/apply_audit_patch.py` | rotate | A | — |
| C-037 | p24 lineage: gap resolver | `BLOCKED` | C-015 | `agent/p24-lineage-gapres` | `curation/gap_resolver.py` | rotate | C | — |
| C-038 | p25 lineage carrier | `BLOCKED` | C-015 | `agent/p25-lineage-carrier` | `pipeline.py` :: `_carry_rag_provenance`, `_RAG_ROW_CARRIER_KEYS` | C-015 impl | A + provenance | — |

## Wave C

| ID | Task | Status | Deps | Branch | Ownership boundary | Reviewer | Focused | Bench |
|---|---|---|---|---|---|---|---|---|
| C-040 | p05a compound-resolution extract | `BLOCKED` | SPIKE-002 | `agent/p05a-resolution-extract` | NEW `pwml/compound_resolution.py`; `ir.py` :: 2 fns | C-051 impl | D | — |
| C-041 | p08 release status + coverage split | `BLOCKED` | C-010, C-012 | `agent/p08-release-status` | NEW `pipeline/release_status.py`; `strict_quarantine.py` :: `evaluate_core_coverage`; `_finalize_gate_failure`; `batch/report.py`; `bench/render.py` | C-010 impl | A, B | — |
| C-042 | p03c extraction escalation ladder | `BLOCKED` | C-032, C-038 | `agent/p03c-extraction-ladder` | `pipeline.py` :: `_run_json_stage`, `_build_extraction_prompt`; `extraction_diagnostics.py` | C-032 impl | A | M2 |
| C-043 | p32 RAG loop controller | `BLOCKED` | C-016, C-021 | `agent/p32-rag-controller` | NEW `rag/controller.py` | C-055 impl | C | — |
| C-044 | p26 lineage: mapping | `BLOCKED` | C-015, C-033 | `agent/p26-lineage-mapping` | `src/t2pw/mapping/map_ids.py` (lineage writes; **not** `src/map_ids.py`) | C-033 impl | C | — |

## Wave D

| ID | Task | Status | Deps | Branch | Ownership boundary | Reviewer | Focused | Bench |
|---|---|---|---|---|---|---|---|---|
| C-050 | p05b pre-freeze resolution call | `BLOCKED` | C-040, C-030 | `agent/p05b-prefreeze-call` | `streamlit_app.py` :: enrichment block above the seam | C-052 impl | D | M3 |
| C-051 | p05c IR assert-only | `BLOCKED` | C-040, C-050 | `agent/p05c-ir-assert-only` | `ir.py` :: `build_pwml_ir` | C-040 impl | D | — |
| C-052 | p06b freeze enforcement | `BLOCKED` | C-030, C-050, C-020 | `agent/p06b-freeze-enforce` | `streamlit_app.py` :: `freeze_canonical_payload`, `run_pwml_export`, SBML bind | C-030 impl | D | M3 |
| C-053 | p09 PWML artifact naming | `BLOCKED` | C-041 | `agent/p09-pwml-naming` | `_finalize_pwml_export`; `runner.py` `:116`/`:856`; `acceptance.py` | C-031 impl | B | — |
| C-054 | p16 gold `expected_export` required | `BLOCKED` | C-041 | `agent/p16-goldset-required` | `bench/goldset.py` | C-056b impl | B | — |
| C-055 | p33 RAG controller wiring | `BLOCKED` | C-043, C-041, C-032 | `agent/p33-rag-wiring` | `streamlit_app.py` :: `maybe_run_rag` + script body | senior | C + AppTest | M4 |
| C-056a | p42a semantic → runtime release_status | `BLOCKED` | C-017, C-041 | `agent/p42a-semantic-runtime` | `pipeline/release_status.py` | C-017 impl | B | — |
| C-056b | p42b semantic → benchmark denominators | `BLOCKED` | C-056a, C-053 | `agent/p42b-semantic-bench` | `acceptance.py` :: `_build_denominators` | C-056a impl | B | — |
| C-057 | p27 lineage: quarantine | `BLOCKED` | C-015, C-010, C-041 | `agent/p27-lineage-quarantine` | `strict_quarantine.py` (lineage writes) | C-041 impl | A, E | — |

## Wave E — placeholders

| ID | Task | Status | Deps | Notes |
|---|---|---|---|---|
| C-060 | p51 false-identifier repairs | `BLOCKED` | R-003 | **Not dispatchable.** Prompt body is generated *from* R-003's report: exact findings, affected files, expected corrections, regression fixtures. |
| C-061 | p52 missing-supported-reaction repairs | `BLOCKED` | R-004 | **Not dispatchable.** Same condition, from R-004. |

## Milestone tests

| ID | Milestone | Status | After | Legs | Wall clock |
|---|---|---|---|---|---|
| T-100 | M1 | `BLOCKED` | Wave B | PMC12452463 ×2, PMC12096016 ×2 | ~1.5 h |
| T-101 | M2 | `BLOCKED` | Wave C | + PMC12444477 ×2, PMC12782028, PMC12312563 | ~2 h |
| T-102 | M3 | `BLOCKED` | C-052 | PMC12856317 equivalence | ~25 min |
| T-103 | M4 | `BLOCKED` | C-055 | 4 RAG legs | ~1.5 h |
| T-104 | M5 first RC | `BLOCKED` | Wave E | full pinned, 20 legs | ~7 h |
| T-105 | M5 second RC | `BLOCKED` | Day 6 corrections | full pinned, 20 legs | ~7 h |

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
