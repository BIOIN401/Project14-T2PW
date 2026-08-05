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
| Blockers | what is stopping it, by task ID |

---

## Wave 0

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INIT-001 | Sprint init + baseline capture | `READY` | — | `9e1b9ab` | `sprint/pwml-recovery` | none | `docs/pwml_recovery_sprint/BASELINE.md`, `.gitignore`, evidence commit | Lead | full suite ×10 chunks | — | — | — | awaiting approval |
| SPIKE-002 | Compound-resolution extraction scoping | `BLOCKED` | INIT-001 | — | none (no code) | none | none — investigation only | Lead | none | — | — | — | INIT-001 |
| R-003 | False-identifier triage (10 findings) | `BLOCKED` | INIT-001 | — | none (read-only) | none | none | Lead | none | — | — | — | INIT-001 |
| R-004 | RAG-reintroduction triage (3 claims, PMC12657337) | `BLOCKED` | INIT-001 | — | none (read-only) | none | none | Lead | none | — | — | — | INIT-001 |

## Wave A0

| ID | Task | Status | Deps | Base SHA | Branch | Worktree | Ownership boundary | Reviewer | Focused | Integration | Merge SHA | Bench delta | Blockers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-010 | p01 stale positional index | `BLOCKED` | INIT-001 | — | `agent/p01-stale-index` | `.claude/worktrees/p01-stale-index` | `strict_quarantine.py` :: `_surviving_processes`, `_degree_zero_exports`, `quarantine_and_close`; `tests/test_strict_quarantine.py`; `tests/test_strict_quarantine_real_artifact_replay.py`; `docs/change_log.md` | C-041 impl | A, E | smoke | — | M1 | INIT-001 |
| C-011 | p00a canonical-freeze seam | `BLOCKED` | INIT-001 | — | `agent/p00a-freeze-seam` | `.claude/worktrees/p00a-freeze-seam` | `streamlit_app.py` :: `run_post_pipeline_sbml_artifacts` | C-012 impl | D | smoke + D | — | — | INIT-001 |
| C-012 | p00b driver seam | `BLOCKED` | INIT-001 | — | `agent/p00b-driver-seam` | `.claude/worktrees/p00b-driver-seam` | `driver.py` :: `_drive` → `_finalize_*` | C-011 impl | B + golden | smoke | — | — | INIT-001 |
| C-013 | p04a two versioned hashes | `BLOCKED` | INIT-001 | — | `agent/p04a-hash-module` | `.claude/worktrees/p04a-hash-module` | NEW `pipeline/canonical_hash.py`; `gate_reports.py` :: `payload_sha256`, `stamp_report`, `gate_verdict` | C-020 impl | smoke | smoke | — | — | INIT-001 |
| C-014 | p03a LLM request timeout | `BLOCKED` | INIT-001 | — | `agent/p03a-llm-timeout` | `.claude/worktrees/p03a-llm-timeout` | `llm/client.py` :: `OpenAI(...)`, `chat_detailed`, `chat_with_tools` | C-032 impl | A, C | smoke | — | — | INIT-001 |
| C-015 | p20 lineage schema | `BLOCKED` | INIT-001 | — | `agent/p20-lineage-schema` | `.claude/worktrees/p20-lineage-schema` | NEW `pipeline/lineage.py` | C-038 impl | new | smoke | — | — | INIT-001 |
| C-016 | p30 RAG stopping policy | `BLOCKED` | INIT-001 | — | `agent/p30-rag-stop-policy` | `.claude/worktrees/p30-rag-stop-policy` | NEW `rag/loop_policy.py` | C-043 impl | new, C | smoke | — | — | INIT-001 |
| C-017 | p40 semantic production module | `BLOCKED` | INIT-001 | — | `agent/p40-semantic-module` | `.claude/worktrees/p40-semantic-module` | NEW `bench/semantic_production.py` | C-056a impl | B | smoke | — | — | INIT-001 |
| C-018 | p50 cofactor / assay-reporter policy | `BLOCKED` | INIT-001 | — | `agent/p50-cofactor-classifier` | `.claude/worktrees/p50-cofactor-classifier` | NEW `pipeline/cofactor_policy.py` | R-003 | new, C | smoke | — | — | INIT-001 |

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
| C-033 | p10 identity hydration | `BLOCKED` | INIT-001 | `agent/p10-identity-hydration` | `map_ids.py` :: 2 fns; `entity_identity.py`; NEW `mapping/uniprot_evidence.py` | C-044 impl | C | M2 |
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
| C-044 | p26 lineage: mapping | `BLOCKED` | C-015, C-033 | `agent/p26-lineage-mapping` | `map_ids.py` (lineage writes) | C-033 impl | C | — |

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
