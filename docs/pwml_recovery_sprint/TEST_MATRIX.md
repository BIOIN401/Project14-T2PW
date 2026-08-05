# PWML Recovery Sprint — Test Matrix

**`--basetemp` is mandatory on every invocation.** Without it, 83 tests error with
`PermissionError` and you will report a false regression. Never run the full suite
unchunked — it approaches 16 GB.

Runtimes measured on `ORIGIN_SHA`, Windows, `.venv/Scripts/python.exe` (3.13.6).

---

## Chunks

| Chunk | Files | Tests | Runtime |
|---|---|---|---|
| **A** | `test_reference_repair`, `test_strict_quarantine`, `test_strict_quarantine_contract_alignment`, `test_strict_quarantine_locks_and_scope`, `test_strict_quarantine_versioning`, `test_empty_extraction_payload` | 123 | **12 s** |
| **B** | `test_bench_goldset_and_semantic`, `test_bench_acquisition_and_artifacts`, `test_bench_controls`, `test_completeness_audit`, `test_batch_driver`, `test_stage3_gate_report_lifecycle` | 225 | **25 s** |
| **C** | `test_rag_admission_production_path`, `test_rag_gap_admission`, `test_rag_triage_orchestration`, `test_rag_provenance_gates`, `test_pipeline_reaction_rag_provenance`, `test_research_mode_orchestration`, `test_map_ids_name_gate`, `test_db_candidate_species_evidence` | 109 | **2 s** |
| **D** | `test_process_normalizer`, `test_pwml_ir`, `test_pwml_writer`, `test_stage_contracts`, `test_payload_models`, `test_streamlit_stage8_export_contract`, `test_streamlit_quarantine_boundary` | 177 | **222 s** ⚠ |
| **E** | `test_strict_quarantine_real_artifact_replay` | parameterized over `runs/` | tens of s per leg |

**SMOKE = A + B + C = 457 tests, ~40 s.** Runs after **every** merge, on the integration
branch. Gate G10.

**Chunk D is excluded from the smoke gate** — 222 s is too slow per merge. It is
**mandatory as a focused test** for every branch marked ✔ below, because that is exactly
where their regressions land and none of it appears in the smoke suite.

**Chunk E skips silently when `runs/` is absent.** `runs/` is committed, so a clean clone
has it; `runs_verify/2026-08-04_1754/` is not yet committed and INIT-001 must fix that or
C-010's allowlist is unverifiable in an isolated worktree.

---

## Commands

```bash
# SMOKE (every merge) — expect 457 passed
.venv/Scripts/python.exe -m pytest -q --basetemp=<tmp>/smoke \
  tests/test_reference_repair.py tests/test_strict_quarantine.py \
  tests/test_strict_quarantine_contract_alignment.py \
  tests/test_strict_quarantine_locks_and_scope.py \
  tests/test_strict_quarantine_versioning.py tests/test_empty_extraction_payload.py \
  tests/test_bench_goldset_and_semantic.py tests/test_bench_acquisition_and_artifacts.py \
  tests/test_bench_controls.py tests/test_completeness_audit.py \
  tests/test_batch_driver.py tests/test_stage3_gate_report_lifecycle.py \
  tests/test_rag_admission_production_path.py tests/test_rag_gap_admission.py \
  tests/test_rag_triage_orchestration.py tests/test_rag_provenance_gates.py \
  tests/test_pipeline_reaction_rag_provenance.py tests/test_research_mode_orchestration.py \
  tests/test_map_ids_name_gate.py tests/test_db_candidate_species_evidence.py

# CHUNK D (export / normalization branches) — expect 177 passed
.venv/Scripts/python.exe -m pytest -q --basetemp=<tmp>/d \
  tests/test_process_normalizer.py tests/test_pwml_ir.py tests/test_pwml_writer.py \
  tests/test_stage_contracts.py tests/test_payload_models.py \
  tests/test_streamlit_stage8_export_contract.py tests/test_streamlit_quarantine_boundary.py

# CHUNK E (quarantine artifact replay)
.venv/Scripts/python.exe -m pytest -q --basetemp=<tmp>/e \
  tests/test_strict_quarantine_real_artifact_replay.py
```

---

## Per-branch obligations

| Branch | Focused | Chunk D | Golden diff | Notes |
|---|---|---|---|---|
| C-010 | A, E | — | — | **plus** the exact per-leg allowlist test (6 legs) and a re-pinned `FULL_STACK_BASELINE` |
| C-011 | D | ✔ | ✔ artifacts dict | pure move — golden diff must be EMPTY |
| C-012 | B | — | ✔ full driver-observable behaviour | see below |
| C-013 | smoke | — | — | |
| C-014 | A, C | — | — | |
| C-015 | new only | — | — | |
| C-016 | new, C | — | — | |
| C-017 | B | — | — | |
| C-018 | new, C | — | — | |
| C-020 | new, D | ✔ | — | equivalence proven by parsing JSON + PWML + SBML |
| C-021 | new, C | — | — | |
| C-030 | D | ✔ | — | |
| C-031 | B | — | ✔ driver | |
| C-032 | B | — | ✔ driver | |
| C-033 | C | — | — | |
| C-034…C-037 | A or C | — | — | |
| C-038 | A + `test_pipeline_reaction_rag_provenance` | — | — | |
| C-040 | D | ✔ | — | |
| C-041 | A, B | — | ✔ driver | |
| C-042 | A | — | — | |
| C-043 | C | — | — | |
| C-044 | C | — | — | |
| C-050 | D | ✔ | — | |
| C-051 | D | ✔ | — | |
| C-052 | D | ✔ | — | |
| C-053 | B | — | ✔ driver | |
| C-054 | B | — | — | |
| C-055 | C + AppTest (`test_rag_triage_orchestration`, `test_batch_driver`) | — | — | least-verifiable branch; senior reviewer |
| C-056a | B | — | — | |
| C-056b | B | — | — | |
| C-057 | A, E | — | — | |

---

## Golden behavioural-equivalence diffs

Two seams and every later `driver.py` branch must prove they changed nothing observable.

### C-012 and every `driver.py` branch — full driver-observable behaviour

Payload hash alone is insufficient: a driver refactor can preserve the biological payload
while dropping a quarantine report or misclassifying a timeout. Compare, byte-for-byte,
across every committed leg fixture:

| Field | Source |
|---|---|
| exit classification | `status`, `stage`, `failure_kind` |
| release status | `release_status` (absent pre-C-041 — assert absent) |
| artifact filenames and paths | `sorted(outcome.artifacts.keys())`, `files[].name` after `_relocate_files` |
| persisted diagnostic artifacts | `files[].bytes` per name |
| manifest fields | full `RunOutcome.to_dict()` (`driver.py:714`) |
| failure reasons | `issue_codes`, `detail` |
| messages | `message`, `warnings` |
| canonical payload hash | `canonical_payload_sha256` |

### C-011 — canonical-freeze seam, same protection

A freeze-seam refactor can preserve the payload while dropping a gate report, changing
`sbml_input_source`, or reordering a side effect. Compare the `post_pipeline_artifacts`
dict before and after on: sorted key set · `canonical_payload_sha256` ·
`final_stage3_gate_report` (full dict incl. `phase`, `payload_sha256`) · `quarantine_ok`,
`quarantine_refusal_reasons`, `quarantine_coverage` · `sbml_input_source` ·
`final_mapped` ↔ `final_mapped_quarantined` identity relationship · presence of
`tmp/final.canonical.json`.

"Before" fixtures are captured on `ORIGIN_SHA` and committed under `evidence/`.

---

## Regression-test standard (gate G9)

Every regression test must **fail on the base SHA**. The reviewer checks out the base,
applies only the test, runs it, and pastes the failure. A regression test that passes
without the fix is not a regression test.

---

## Milestone benchmarks

Scheduled by the Lead Orchestrator only. Agents never launch a benchmark.

| ID | Milestone | After | Legs | Wall clock | Acceptance |
|---|---|---|---|---|---|
| T-100 | M1 | Wave B | PMC12452463 ×2, PMC12096016 ×2 | ~1.5 h | both pass the quarantine boundary; **PMC12452463 → `review_required`, not strict success** (TRAP-1) |
| T-101 | M2 | Wave C | + PMC12444477 ×2, PMC12782028, PMC12312563 | ~2 h | no leg reports "produced nothing"; `identical_empty_response` recorded where two draws share a hash; `budget_exhausted` distinct from failure |
| T-102 | M3 | C-052 | PMC12856317 | ~25 min | reload + re-export with **all resolvers disabled** → identical `canonical_graph_sha256`; equivalence proven by parsing and normalizing JSON, PWML and SBML |
| T-103 | M4 | C-055 | 4 RAG legs | ~1.5 h | every RAG round re-entered normalization, mapping, gates, persistence, classification |
| T-104 | M5 first RC | Wave E | full pinned, 20 legs | ~7 h | full acceptance matrix vs `BASELINE.md` |
| T-105 | M5 second RC | Day 6 | full pinned, 20 legs | ~7 h | remaining failures explained and classified |

Measured leg times: 1308 s and 1511 s. A full run is ~7 h; the runner's own
`DEFAULT_DEADLINE_HOURS = 10.0` confirms overnight sizing.

**Never use a full benchmark as a per-patch merge gate.** And before calling any
single-leg change a regression, re-run that leg — identical legs give materially
different Stage-1 draws at temperature 0 in this repository.

---

## Baseline to preserve (filled by INIT-001)

Full suite per-chunk counts · smoke 457 · chunk D 177 · `bench_acceptance.py` on
`runs/2026-08-02_2130` · `FULL_STACK_BASELINE` and `RESIDUAL_CODES_BY_{LEG,ROW}` as
currently pinned. See `BASELINE.md`.
