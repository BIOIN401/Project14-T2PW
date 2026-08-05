# PWML Recovery Sprint — Baseline

**Status: TEMPLATE — NOT YET MEASURED.**

INIT-001 fills every `TBD` below and commits the result. Until then, no acceptance claim
in this sprint has a comparison surface, and no branch may be merged.

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
| Measured on | TBD (date, machine, Python 3.13.6, Windows) |

---

## 1. Test suite — full, chunked

`--basetemp` per chunk. Never unchunked.

| Chunk | Passed | Failed | Skipped | Runtime |
|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD |
| … | | | | |
| **TOTAL** | **TBD** | **TBD** | **TBD** | **TBD** |

Any pre-existing failure or skip is recorded here so it is never mistaken for a
sprint-induced regression. Note in particular whether chunk E
(`test_strict_quarantine_real_artifact_replay`) ran or **skipped** — it skips silently
when `runs/` is absent.

---

## 2. Gate suites

| Suite | Expected | Measured |
|---|---|---|
| Smoke (A+B+C) | 457 passed, ~40 s | TBD |
| Chunk D | 177 passed, ~222 s | TBD |
| Chunk A | 123 passed, ~12 s | TBD |
| Chunk B | 225 passed, ~25 s | TBD |
| Chunk C | 109 passed, ~2 s | TBD |

---

## 3. Benchmark — `runs/2026-08-02_2130` (committed, 208 tracked files)

```
.venv/Scripts/python.exe scripts/bench_acceptance.py \
  --run-dir runs/2026-08-02_2130 \
  --json docs/pwml_recovery_sprint/evidence/baseline_acceptance.json
```

Gold set `2026-08-01.1`. **If any value differs from Expected, STOP** — the baseline has
drifted and every downstream acceptance criterion is invalid.

| Metric | Expected | Measured |
|---|---|---|
| False real identifiers | 10 | TBD |
| Placeholder-backed proteins | 21 | TBD |
| Unsupported reactions | 7 | TBD |
| Orphaned references | 2 | TBD |
| Missing supported reactions | 3 | TBD |
| Missing pathway anchors | 4 | TBD |
| Quarantined processes | 3 | TBD |
| Requested-pathway coverage | 0/8 | TBD |
| Semantic confirmed | 0/8 | TBD |
| Strict PWML success | 0/4 | TBD |
| Research deliverable produced | 4/8 | TBD |
| Research semantically confirmed | 1/8 | TBD |
| Extraction success | 8/8 | TBD |
| Identity: verified / placeholder / unresolved / PathBank-Unknown | 1 / 21 / 108 / 5 | TBD |

Strict failures by boundary — expected: 6 `stage3_normalization_gate`,
2 `stage1_extraction`, 2 `scope_ambiguity`.

Payload sources scored — expected: `final_mapped.json` 11, `merged_payload.json` 5.

---

## 4. Pinned baselines inside the test suite

From `tests/test_strict_quarantine_real_artifact_replay.py`. **C-010 changes these by
design (TRAP-2).** Record the pre-change values so the delta is provable.

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

The change log (`docs/change_log.md`) must carry the same table and must never disagree.

---

## 5. C-010 expected per-leg delta allowlist

Measured on `ORIGIN_SHA` across every archived leg carrying a `final_mapped.json`, with
`pathway_context=None` (matching the archived-leg reality that no leg carries Stage-0
context).

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

**The last two rows require `runs_verify/2026-08-04_1754/` to be committed.** Until INIT-001
commits it, C-010's allowlist cannot be verified in an isolated worktree.

---

## 6. Repository size and cache state

| Item | Size | Tracked | Note |
|---|---|---|---|
| `.git` | 158 MB | — | |
| `data/enrichment_cache.json` | 39.4 MB | **yes**, modified | **no branch may commit a cache modification** (TRAP-5) |
| `data/id_mapping_cache.json` | 4.4 MB | **yes**, modified | same |
| `runs_verify/2026-08-04_1754/` total | 44 MB | no | evidence needed by C-010, C-014, C-042 |
| └ `cache_snapshot/` | 38 MB | no | **exclude from the evidence commit** |
| └ `papers/` | 5.7 MB | no | commit |
| └ metadata | ~140 KB | no | commit |
| Evidence-commit size | **~5.9 MB** | | |
| All 8 untracked `cache_snapshot/` dirs | 304 MB | no | see DECISIONS O-2 |

Precedent is mixed: 16 `cache_snapshot` files are tracked under older `runs/`, none under
recent `runs_verify/`. Follow the recent convention.

---

## 7. Working-tree state at branch creation

Preserved exactly. Nothing stashed, reset or discarded.

Tracked modifications carried onto `sprint/pwml-recovery` (all scratch/generated,
**none sprint-related**):

```
 M data/enrichment_cache.json     39.4 MB   regenerable lookup cache
 M data/id_mapping_cache.json      4.4 MB   regenerable lookup cache
 M out/enrichment_dump.json         82 KB   run artifact (dir is gitignored, file tracked)
 M outputs/pathway.pwml             35 KB   run artifact (dir is gitignored, file tracked)
 M tmp/draft_graph.json            9.5 KB   run artifact (dir is gitignored, file tracked)
 M tmp/qa_report.json              5.8 KB   run artifact
 M tmp/reaction_summary.txt        1.1 KB   run artifact
```

Untracked, repo-local:

```
runs_verify/2026-08-04_{1148,1207,1234,1306,1358,1504,1647}/cache_snapshot/   7 × 38 MB
runs_verify/2026-08-04_1754/                                                  44 MB  SPRINT-RELATED
topics_flip_strict.txt  topics_regression_research.txt  topics_verify_subset.txt   run inputs
```
