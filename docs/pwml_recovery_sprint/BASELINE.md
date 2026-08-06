# PWML Recovery Sprint — Baseline

**Status: MEASURED by INIT-001 on 2026-08-05.**

Every command below ran through the bounded foreground wrapper
(`evidence/bounded_run.py`), one heavy job at a time. Cleanup reports are in § 8;
**every job reported a final surviving count of 0.**

---

## Provenance

| Field | Value |
|---|---|
| Source branch | `research-mode` |
| `ORIGIN_SHA` | `9e1b9abe7ba8a1a228558fd03ca6c394cc22c31e` |
| HEAD subject | "Add the gate-lifecycle acceptance run (2026-08-04_1647)" |
| HEAD date | Tue Aug 4 17:38:06 2026 -0600 |
| Ahead of `main` by | 72 commits, behind by 0 |
| Integration branch | `sprint/pwml-recovery` |
| SHA at INIT-001 start | `721a256d3934f936cf48a5369cba15ebe05c1a48` |
| Measured on | 2026-08-05, Windows 11, `.venv/Scripts/python.exe` 3.13.6 |
| Wrapper | `docs/pwml_recovery_sprint/evidence/bounded_run.py`, Windows Job Object with `KILL_ON_JOB_CLOSE` |

---

## 1. Test suite — full, chunked

104 test files, contiguous **alphabetical** partition into 10 chunks (reproducible
and unbiased), unique `--basetemp` per chunk, run strictly sequentially. Exact
membership: `evidence/baseline_suite_result.json`.

| Chunk | Files | Passed | Failed | Skipped | Errors | Runtime |
|---|---|---|---|---|---|---|
| 01 `test_adversarial_actor_entity_type` … `test_batch_preflight` | 11 | 263 | 0 | 0 | 0 | 28.1 s |
| 02 `test_batch_report` … `test_db_candidate_species_evidence` | 11 | 282 | 0 | 0 | 0 | 20.4 s |
| 03 `test_db_resolver_primitives` … `test_gap_resolver` | 11 | 225 | 0 | 0 | 0 | 39.6 s |
| 04 `test_gap_resolver_agent_tools` … `test_paper_eligibility_corrections` | 11 | **284** | 0 | 0 | 0 | 9.3 s |
| 05 `test_pathbank_unknown_fallback` … `test_protein_export_policy` | 10 | 288 | 0 | 0 | 0 | 20.0 s |
| 06 `test_pwml_db_resolver` … `test_rag_extract` | 10 | 159 | 0 | 0 | 0 | 32.9 s |
| 07 `test_rag_foundation` … `test_rag_select` | 10 | 170 | 0 | 0 | 0 | 3.2 s |
| 08 `test_rag_synonym_merge` … `test_research_mode_normalizer` | 10 | 163 | 0 | 0 | 0 | 12.5 s |
| 09 `test_research_mode_orchestration` … `test_stage_contracts` | 10 | 136 | 0 | 0 | 0 | 9.4 s |
| 10 `test_stage_one_boundary` … `test_strict_quarantine_versioning` | 10 | 341 | **2** | 8 | 0 | 302.2 s |
| **TOTAL** | **104** | **2311** | **2** | **8** | **0** | **~478 s** |

### Chunk 04 required a split — a chunking artifact, not a failure

As one group, chunk 04 aborted at collection with
`ImportError: cannot import name '_completion_is_empty' from 't2pw.llm.client'
(unknown location)`, and pytest's `Interrupted: 1 error during collection` meant
**none of its 11 files ran**. Re-run split, everything passes:

| Split | Files | Result |
|---|---|---|
| 04a | `test_llm_client_empty_completion_retry.py` alone | **29 passed** in 0.09 s |
| 04b | the other 10 files | **255 passed** in 7.33 s |

**Cause — a pre-existing test-isolation defect, unrelated to this sprint.**
Bisected: `test_gap_resolver_agent_tools.py` and `test_gap_resolver_stage3_issues.py`
each leave a stub for `t2pw.llm.client` in `sys.modules`. Any later real import in
the same pytest session gets the stub, which has no `__file__` — hence "unknown
location". `test_interactive_curator.py` and
`test_gap_resolver_biological_state_resolution.py` do **not** do this.

```
test_gap_resolver_agent_tools            + llm_client -> 1 error in 0.17s
test_interactive_curator                 + llm_client -> 47 passed in 0.30s
test_gap_resolver_stage3_issues          + llm_client -> 1 error in 0.35s
test_gap_resolver_biological_state_reso. + llm_client -> 31 passed in 0.90s
```

No merge gate is affected: chunks A–E never co-locate these files. Any future
chunking must keep the two gap-resolver files out of a group containing
`test_llm_client_empty_completion_retry.py`.

### Chunk E **RAN — it did not skip**

`runs/` is committed, so the parameterized replay harness had inputs.
Run on its own: **159 passed, 2 failed, 0 skipped** in 31.7 s. The 8 skips in
chunk 10 come from its other nine files.

The 2 failures are **pre-existing on `ORIGIN_SHA`** and are analysed in § 5.

---

## 2. Gate suites

| Suite | Expected | Measured | |
|---|---|---|---|
| Smoke (A+B+C) | 457 passed, ~40 s | **457 passed**, 30.6 s | ✔ |
| Chunk D | 177 passed, ~222 s | **177 passed**, 199.5 s | ✔ |
| Chunk A | 123 passed, ~12 s | **123 passed**, 2.8 s | ✔ |
| Chunk B | 225 passed, ~25 s | **225 passed**, 26.1 s | ✔ |
| Chunk C | 109 passed, ~2 s | **109 passed**, 2.2 s | ✔ |
| Chunk E | 173 passed, ~20 s | **173 passed, 0 failed**, 18.2 s | ✔ |

> **Corrected 2026-08-06 (H-005).** This row originally read "| Chunk E | — | **159
> passed, 2 failed** | see § 5 |", the pre-sprint figure. The accepted gate value after
> H-001 + H-002 is **173 passed / 0 failed**, and it is measured here rather than copied:
> `pytest -q --basetemp=C:/pt/h5e tests/test_strict_quarantine_real_artifact_replay.py`
> under the wrapper → `173 passed in 18.22s`, exit 0, final surviving count 0
> (`evidence/g11/H-005/05-chunk-e.json`; an earlier identical-result run is
> `04-chunk-e.json`). **§ 1's "Chunk E RAN" note and § 5 keep the 159/2 figure and are
> left untouched** — they are the correct historical INIT-001 record of `ORIGIN_SHA`,
> before H-001 froze the cohort manifest and H-002 landed.

---

## 3. Benchmark — `runs/2026-08-02_2130`

```
.venv/Scripts/python.exe scripts/bench_acceptance.py \
  --run-dir runs/2026-08-02_2130 --json <out>.json
```

**The re-run is byte-identical to the committed `evidence/baseline_acceptance.json`:**
SHA-256 `d3538f4b1cefc1f8e7aca933318df13c9967ce1071e48f8e6a3e4bd6830f4ec3`,
242,199 bytes — matching `evidence/PROVENANCE.md` exactly. Nothing has drifted.

Gold set `2026-08-01.1`. 10 papers, 20 legs attempted, 16 scored.

| Metric | Expected | Measured | |
|---|---|---|---|
| False real identifiers | 10 | **10** | ✔ |
| Placeholder-backed proteins | 21 | **21** | ✔ |
| Unsupported reactions | 7 | **7** | ✔ |
| Orphaned references | 2 | **2** | ✔ |
| Missing supported reactions | 3 | **3** | ✔ |
| Missing pathway anchors | 4 | **4** | ✔ |
| Quarantined processes | 3 | **3** | ✔ |
| Semantic pathway success | 0/8 | **0/8** | ✔ |
| Strict PWML success | 0/4 | **0/4** | ✔ |
| Research deliverable produced | 4/8 | **4/8 = 50%** | ✔ |
| Research semantically confirmed | 1/8 | **1/8 = 12%** | ✔ |
| Extraction success | 8/8 | **8/8 = 100%** | ✔ |
| Gold relevance prevalence | — | 8/10 = 80% | |
| Identity: verified / placeholder / unresolved / PathBank-Unknown | 1 / 21 / 108 / 5 | **1 / 21 / 108 / 5** | ✔ |

Strict failures by boundary — expected 6 / 2 / 2:
**`stage3_normalization_gate` 6, `stage1_extraction` 2, `scope_ambiguity` 2** ✔

Payload sources scored — expected 11 / 5:
**`final_mapped.json` 11, `merged_payload.json` 5** ✔

`scripts/bench_acceptance.py` exits **1**; that is its "acceptance not met" signal,
not a crash. Recorded so a future runner does not read it as infrastructure failure.

Two caveats the scorer itself emits, carried forward: 5 legs were scored from
pre-mapping `merged_payload.json`, so their identity and false-identifier counts
are **floors, not measurements**; and 3 checks could not be evaluated at all and are
excluded from `confirmed` rather than counted as passes.

---

## 4. Pinned baselines inside the test suite

From `tests/test_strict_quarantine_real_artifact_replay.py:384-408`, as currently
pinned. **C-010 changes these by design (TRAP-2).** Recorded so the delta is provable.

```python
FULL_STACK_BASELINE = {
    "legs_examined": 23, "quarantine_admitted": 18, "quarantine_refused": 5,
    "stage3_after_pass": 18, "required_contract_pass": 1,
    "reached_ir": 1, "ir_pass": 1, "exportable": 1,
}
RESIDUAL_CODES_BY_LEG = {
    "species_missing_classification": 17, "species_missing_taxonomy": 17,
    "no_biological_states": 4,
}
RESIDUAL_CODES_BY_ROW = {
    "species_missing_classification": 19, "species_missing_taxonomy": 19,
    "no_biological_states": 4,
}
```

> **Corrected 2026-08-06 (H-005).** This line originally read: "`docs/change_log.md:147`
> carries the same '17 legs, 19 rows' figures and agrees." **It is stale in both halves.**
> The line number moved — `:147` is now inside the H-001 re-measurement note — and the
> figures moved with H-001: `docs/change_log.md:165-167` now reads
> **`species_missing_classification` (19 legs, 27 rows), `species_missing_taxonomy`
> (19 legs, 27 rows), `no_biological_states` (4 legs, 4 rows)**, matching
> `tests/test_strict_quarantine_real_artifact_replay.py:850-859` as re-pinned by H-001.
> The 17/19 block above is the INIT-001 record of what was pinned on `ORIGIN_SHA` and is
> deliberately left standing as history; it is **not** the current pin.

---

## 5. ⚠ The pinned replay baseline is ALREADY STALE on `ORIGIN_SHA`

**This is the most consequential finding of INIT-001 and it is not in the plan.**

Two tests in `test_strict_quarantine_real_artifact_replay.py` fail **before any
sprint code change**:

```
FAILED ...::test_no_archived_leg_carries_stage_zero_context
FAILED ...::test_the_full_stack_baseline_is_exactly_what_was_reported
```

### Proof the sprint did not cause them

| Check | Result |
|---|---|
| `git diff ORIGIN_SHA..HEAD -- runs/ tests/test_strict_quarantine_real_artifact_replay.py` | **empty** |
| `git status --porcelain runs/` | **empty** — fully tracked, 754 files, clean |
| Test input scope | `RUNS = ROOT / "runs"` (`:104`) — **`runs_verify/` is never globbed** |

The evidence commit added only `runs_verify/2026-08-04_1754/`, which this test does
not read. Inputs and test are byte-identical to `ORIGIN_SHA`; the failures pre-date
the sprint branch.

### Failure 1 — `FULL_STACK_BASELINE`: 23 legs pinned, 39 measured

```
measured = {'legs_examined': 39, 'quarantine_admitted': 27, 'quarantine_refused': 12,
            'stage3_after_pass': 27, 'required_contract_pass': 8,
            'reached_ir': 8, 'ir_pass': 8, 'exportable': 8}
```

**When it broke.** The pin was written at `404cc8d` (2026-08-01). `runs/2026-08-02_2130`
— 16 further legs — was committed at `5f2cd2f` (2026-08-04), *three commits before the
sprint branch was cut*, and nobody re-measured. Four `runs/` directories contribute legs:
`2026-07-27_1623` (5), `2026-07-28_0919` (**2**), `2026-07-28_2122` (16),
`2026-08-02_2130` (16) = **39**.

> **Corrected 2026-08-06 (H-005).** The breakdown originally gave `2026-07-28_0919` as
> **4**, summing to 41 against a stated 39-leg total. Re-derived directly from the
> committed cohort manifest `tests/data/baseline_cohort_manifest.json`
> (`sha256 086aa65a86b45be0a0bf32113b4ff6fd66599862c5ff2eacc7212d90aeebe013`,
> `cohort_id pre-implementation-baseline-2026-08-05`): declared `leg_count` 39, 39 leg
> entries, 39 unique paths, and by first path segment `2026-07-27_1623` = 5,
> `2026-07-28_0919` = **2**, `2026-07-28_2122` = 16, `2026-08-02_2130` = 16, sum 39.
> Cleanup report `evidence/g11/H-005/06-manifest-derivation.json`. Note the manifest is
> H-001's frozen cohort and is now the sole membership source — the harness no longer
> globs the filesystem — so this breakdown can no longer drift with an archived run.

**Why it matters beyond a stale number.** This test is parameterized over the
*filesystem*. Every milestone benchmark T-100…T-105 archives a new run, so each one
silently re-breaks a merge-gate test. The gate is not stable across the sprint it is
meant to guard.

### Failure 2 — `key_compounds` carriers

9 files under `runs/` contain `key_compounds`. Classified:

| | |
|---|---|
| Files carrying it | 9 — **all `stage0_attempts.json`** |
| **Payload files (`final_mapped.json` / `merged_payload.json`) carrying it** | **0** |
| Run directory | all 9 in `runs/2026-08-02_2130` |

**`MASTER_PLAN` § 1's premise is intact.** The C-010 allowlist was measured with
`pathway_context=None` "matching the archived-leg reality that no leg carries Stage-0
context". No *payload* carries it — that remains true. What changed is that
`runs/2026-08-02_2130` introduced a new **diagnostic** artifact, `stage0_attempts.json`,
which persists Stage-0 context for the first time; the test globs every `*.json`, so it
now trips on a diagnostic rather than a payload. The assertion is broader than its own
docstring, which is specifically about *payload-only discovery*.

**Consequence for C-010.** TRAP-2 tells its implementer that `FULL_STACK_BASELINE`
moves by design and that reverting behaviour to make it pass is a reject. But the pin is
*already* wrong for an unrelated reason, so an implementer cannot distinguish "my fix
moved it" from "it was broken before I arrived" unless the pre-existing delta above is
treated as the true starting point. **This needs a product-owner decision before C-010
is dispatched** — see the report accompanying this file.

---

## 6. C-010 expected per-leg delta allowlist

Unchanged from the diagnosis; reproduced here for the acceptance comparison. Measured on
`ORIGIN_SHA` across every archived leg carrying a `final_mapped.json`, with
`pathway_context=None`.

```
legs examined : 32      unchanged : 26      CHANGED : 6      errors : 0
```

| Leg | degree_zero before → after | ok | refusals before → after |
|---|---|---|---|
| `runs/2026-08-02_2130/papers/PMC12096016/strict` | `[EntA, Unknown, enterobactin synthase complex]` → `[]` | F→**T** | `[degree_zero_export:3]` → `[]` |
| `runs/2026-08-02_2130/papers/PMC12856317/research` | `[ALAS2, ALAS2 homodimer]` → `[]` | F→**F** | `[degree_zero_export:2, unexportable_entity:1]` → `[unexportable_entity:1]` |
| `runs_verify/2026-08-04_1207/papers/PMC12452463/strict` | `[EntE]` → `[]` | F→**T** | `[degree_zero_export:1]` → `[]` |
| `runs_verify/2026-08-04_1234/papers/PMC12096016/strict` | `[EntA]` → `[]` | F→**T** | `[degree_zero_export:1]` → `[]` |
| `runs_verify/2026-08-04_1754/papers/PMC12452463/research` | `[Isochorismatase (EntB)]` → `[]` | F→**T** | `[degree_zero_export:1]` → `[]` |
| `runs_verify/2026-08-04_1754/papers/PMC12452463/strict` | `[EntD, EntF]` → `[]` | F→**T** | `[degree_zero_export:2]` → `[]` |

Three properties the allowlist asserts, all required:

1. every listed leg transitions exactly as tabulated;
2. `set(changed legs) == set(allowlist)` — no unlisted leg changed in any of the three
   observables;
3. every changed leg's `degree_zero_after` is `[]` — the fix never **adds** a
   degree-zero entity.

`PMC12856317/research` remaining `ok=False` on a genuine `unexportable_entity:1` is the
strongest available evidence that the fix removes only the false refusal and does not
weaken a biological gate.

**The last two rows are now verifiable in an isolated worktree** —
`runs_verify/2026-08-04_1754/` was committed by INIT-001 (§ 7).

Note that this allowlist globs `runs/` **and** `runs_verify/` (32 legs), while the
replay test globs `runs/` only (39 legs). Two different populations; do not compare the
counts.

---

## 7. Repository size and cache state

| Item | Size | Tracked | Note |
|---|---|---|---|
| `.git` before INIT-001 | 158 MB | — | |
| `.git` after the evidence commit | **159 MB** | — | +~1 MB (6 MB of JSON compresses well) |
| `data/enrichment_cache.json` | 39.4 MB | **yes**, still modified | untouched — O-3 open (TRAP-5) |
| `data/id_mapping_cache.json` | 4.4 MB | **yes**, still modified | untouched — O-3 open |
| `runs_verify/2026-08-04_1754/` total | 44 MB | partially | |
| └ `papers/` + metadata | ~6 MB | **yes — committed, 152 files** | |
| └ `cache_snapshot/` | 38 MB, 2 files | **no — excluded** | recent `runs_verify` convention |
| All 8 untracked `cache_snapshot/` dirs | 304 MB | no | **O-2 still open** — no `.gitignore` rule added |

Verification required by INIT-001 Step 3:

```
git ls-files runs_verify/2026-08-04_1754 | wc -l              -> 152   (must be > 0) ✔
git ls-files 'runs_verify/2026-08-04_1754/cache_snapshot/*'   -> 0     (must be 0)   ✔
```

---

## 8. Cleanup reports — gate G11

Every heavy job, in execution order. Isolation was a Windows **Job Object with
`KILL_ON_JOB_CLOSE`** in every case; no run fell back to the process-group model.

| Job | Root PID | Exit reason | Exit code | Descendants obs. | Terminated | **Survivors** | Cleanup |
|---|---|---|---|---|---|---|---|
| suite chunk01–10 (10 jobs) | see `baseline_suite_result.json` | completed ×9, nonzero ×1 | — | 1–3 each | all | **0** | ok |
| c04a | 801724 | completed | 0 | 1 | 1 | **0** | ok |
| c04b | 801708 | completed | 0 | 1 | 1 | **0** | ok |
| smoke | 799864 | completed | 0 | 9 | 9 | **0** | ok |
| chunkD | 802844 | completed | 0 | 1 | 1 | **0** | ok |
| chunkA | 806756 | completed | 0 | 1 | 1 | **0** | ok |
| chunkB | 806224 | completed | 0 | 8 | 8 | **0** | ok |
| chunkC | 807404 | completed | 0 | 1 | 1 | **0** | ok |
| chunkE | 806936 | nonzero | 1 | 1 | 1 | **0** | ok |
| bench_acceptance | 804840 | nonzero | 1 | 1 | 1 | **0** | ok |

**Total surviving owned processes across the entire baseline: 0.**

Pre-existing Python processes were detected and **reported, never killed**, per
`[S8]` item 4.

---

## 9. Working-tree state — unchanged by INIT-001

The 7 tracked modifications are exactly as expected and were **left untouched**;
O-3 is a product-owner decision.

```
 M data/enrichment_cache.json     39.4 MB   regenerable lookup cache
 M data/id_mapping_cache.json      4.4 MB   regenerable lookup cache
 M out/enrichment_dump.json         82 KB   run artifact
 M outputs/pathway.pwml             35 KB   run artifact
 M tmp/draft_graph.json            9.5 KB   run artifact
 M tmp/qa_report.json              5.8 KB   run artifact
 M tmp/reaction_summary.txt        1.1 KB   run artifact
```

No other tracked file was modified — the repository matches the state the control
plane was committed against.
