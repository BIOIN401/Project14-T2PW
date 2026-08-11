# Evidence provenance

Every file in this directory, where it came from, and what it is trusted to prove.

---

## `baseline_acceptance.json`

The pre-sprint benchmark measurement. `BASELINE.md` § 3 compares against it; every
acceptance claim in the sprint is a delta from these numbers.

| Field | Value |
|---|---|
| **Source commit** | `b0b95184f1c5c2693058e4d22ddd128cc8988a27` (`sprint/pwml-recovery`, control-plane setup) |
| **Generated** | 2026-08-05T20:33Z |
| **Command** | `.venv/Scripts/python.exe scripts/bench_acceptance.py --run-dir runs/2026-08-02_2130 --json docs/pwml_recovery_sprint/evidence/baseline_acceptance.json` |
| **Python** | 3.13.6 (Windows, `.venv`) |
| **SHA-256** | `d3538f4b1cefc1f8e7aca933318df13c9967ce1071e48f8e6a3e4bd6830f4ec3` |
| **Size** | 242,199 bytes |

### Inputs

| Input | State |
|---|---|
| `runs/2026-08-02_2130/` | **Committed, clean.** 208 tracked files. `git status --porcelain` on the path returns nothing. |
| `src/t2pw/bench/gold/pinned_v1.json` | **Committed, clean.** Gold version `2026-08-01.1`, 10 cases, last touched at `5f2cd2f` (Tue Aug 4 2026). |
| `scripts/bench_acceptance.py`, `src/t2pw/bench/` | **Committed, clean.** |

`git status --porcelain runs/2026-08-02_2130 src/t2pw/bench scripts/bench_acceptance.py`
returned **0 lines** at generation time.

### Were dirty or uncommitted inputs used?

**No.** Seven tracked files were modified in the working tree when this was generated:

```
data/enrichment_cache.json   data/id_mapping_cache.json   out/enrichment_dump.json
outputs/pathway.pwml   tmp/draft_graph.json   tmp/qa_report.json   tmp/reaction_summary.txt
```

**None of them is an input to this measurement.** `bench_acceptance.py` performs no
network access, no LLM call and no database access: `score_run` reads the run directory's
`manifest.jsonl`, each leg's `final_mapped.json` / `merged_payload.json` /
`rag_admission_report.json` / `quarantine_report.json`, and `01_source_text.txt` for gold
quote verification — plus the gold set. The mapping and enrichment caches are pipeline
inputs, not scoring inputs, and `out/`, `outputs/` and `tmp/` are unrelated run scratch.

`runs_verify/2026-08-04_1754/` was **not** committed at generation time and is **not**
used: this measurement scores `runs/2026-08-02_2130` only.

### Verified reproduction

Re-derived from the committed JSON at generation time and matching `BASELINE.md` § 3:

| Metric | Expected | Reproduced |
|---|---|---|
| False real identifiers | 10 | **10** ✔ |
| Unsupported reactions | 7 | **7** ✔ |
| Orphaned references | 2 | **2** ✔ |
| Missing supported reactions | 3 | **3** ✔ |
| Placeholder-backed proteins | 21 | **21** ✔ |

Top-level keys: `run_dir`, `gold_set`, `legs_attempted`, `legs_scored`, `completion`,
`acceptance_priorities`, `denominators`, `scientific_errors`,
`strict_failures_by_boundary`, `research_failures_by_cause`, `identity`,
`blockers_by_scope`, `papers`, `notes`. 10 papers scored.

### Caveat for INIT-001

INIT-001 is still instructed to run this command itself and confirm the numbers. If its
run disagrees with the SHA-256 above, **stop** — either an input drifted or the scoring
changed, and every downstream acceptance criterion is invalid until that is explained.
This file is a starting point and a cross-check, not a substitute for that step.

---

## Reproduction and probe scripts

All were written during the diagnostic pass and are committed here so a reviewer can
regenerate the evidence rather than trust a number in a document.

**One change was made to every script when committing:** the scratch versions hard-coded
`c:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`. That is replaced by `_repo_root.py`,
which walks up from `__file__` looking for `pyproject.toml` + `src/t2pw`. No other logic
was altered.

**Audited and clear:** no secrets, no API keys, no credentials, no `.env` reads, no
network calls, no writes outside stdout. None imports `llm.client`. The two scripts that
call `build_pwml_ir` may attempt a PathBank connection via `PathBankDbResolver.from_env()`
and degrade to the offline name index when it is unreachable — documented in their
docstrings, because the database-dependence is itself part of the finding.

### Scratch → committed mapping

Six scratch scripts became five committed files. **None was excluded.**

| Scratch | Committed as | Note |
|---|---|---|
| `allowlist.py` | `allowlist_generator.py` | 1:1 |
| `repro_idx.py` | `repro_stale_index_synthetic.py` | 1:1 |
| `replay.py` | `repro_stale_index_real_artifact.py` | **merged** — see below |
| `replay2.py` | `repro_stale_index_real_artifact.py` | **merged** — see below |
| `nextgate.py` | `probe_downstream_gates.py` | 1:1 |
| `irdrop.py` | `probe_exporter_identity_mutation.py` | 1:1, plus measurement A |

**Why `replay.py` and `replay2.py` merged.** Both answered the same question — "does the
stale-index defect change this real leg's verdict?" — with the same
shipped-vs-counterfactual pattern. `replay.py` probed one leg and printed extra detail;
`replay2.py` probed three and printed a summary. Committing both would put two scripts
with one purpose in the evidence directory, and a reviewer would have to work out which
is authoritative. The merged file probes all four legs and prints the fuller output. No
measurement was dropped.

`replay2.py` also had a stale label: it called
`runs_verify/2026-08-04_1754/.../PMC12096016` "UNTRACKED", which was accurate during the
diagnosis but becomes wrong once INIT-001 commits that run. The merged file labels legs
by which step commits them instead.

### The five committed scripts

| File | Proves | Artifact dependency | Runtime |
|---|---|---|---|
| `allowlist_generator.py` | The exact per-leg allowlist in `BASELINE.md` § 5. **C-010's acceptance criterion.** | globs whatever is present; finds 4 changed legs before INIT-001, 6 after | seconds |
| `repro_stale_index_synthetic.py` | The mechanism, with no artifacts. Basis for C-010's first unit test. | **none** — runs in any checkout | < 1 s |
| `repro_stale_index_real_artifact.py` | The same defect on production payloads, and the Fur → index-shift chain. | 2 of 4 legs need INIT-001; prints `[skip]` until then | seconds |
| `probe_downstream_gates.py` | How far a fixed leg gets: Stage-3 PASS, required-field PASS, **IR build FAIL**. The guard against overclaiming C-010. | 1 of 2 legs needs INIT-001 | seconds; may attempt a DB connection |
| `probe_exporter_identity_mutation.py` | Post-freeze mutation on `runs_verify/2026-08-04_1647/…/strict`, in five categories that are never summed: **name changes 1** (`glycine → Glycine`, no provenance record) · **mapped-ID changes 1** (heme `mapped_ids.pubchem`, a within-row re-projection) · **synthetic database rows 1** (a fabricated `db_row`) · **prefix normalization 8** (4 of 4 rows) · **identity materialization 16** (4 of 4 rows). Rows pair on the pre-freeze `mapping_meta.query.name`, strictly one-to-one; a lineage failure exits nonzero as `UNDETERMINED` and can never read as clean. **Acceptance target for C-050/C-051** — all five 0 with `RESULT: MEASURED`. **Necessary but NOT sufficient for T-102** (D-016: T-102 also requires organism/species equivalence across JSON, PWML and SBML; this measures compound rows in the JSON only). Full output: `probe_exporter_identity_mutation_2026-08-06.md`. | **none beyond already-committed legs** | seconds; may attempt a DB connection |

> **Corrected 2026-08-06 (H-005).** The `probe_exporter_identity_mutation.py` row
> previously read: *"Nine identifiers added to Glycine after the freeze. **Acceptance
> target for C-050/C-051 and T-102.**"* Both halves were wrong and are quoted here rather
> than deleted. **(a)** All nine of Glycine's identifiers are present in the canonical
> row; only the `chebi` *value* differs (`CHEBI:15428` → `15428`). The "nine added" figure
> came from the probe pairing rows on the raw `name` while the exporter renamed
> `glycine → Glycine`. **(b)** Naming it an acceptance target for T-102 contradicts
> **D-016 (LOCKED)**, which rules T-102 is not narrowed to compounds. It is an acceptance
> target for C-050/C-051, and only a necessary component of T-102.

### Findings recorded as evidence

These are measured records, not scripts and not fixes. Each states its own status.

| File | Records | Status |
|---|---|---|
| `probe_exporter_identity_mutation_2026-08-06.md` | The corrected probe's full output, the superseded pre-fix output, and the before/after per category. | Measurement |
| `finding_alas2_identity_placeholder_2026-08-06.md` | `runs/2026-08-02_2130/papers/PMC12856317/strict` ships protein `ALAS2` as a literal `"Unknown"` on a cross-species PathBank row, while `P22557` appears **twice** in the supplied paper evidence. | **UNRESOLVED. Not a fix. Closes nothing, including O-1.** |

### Invocation

```bash
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/<script>.py
```

Run from the repository root. `_repo_root.py` is a helper module, not a script.
`probe_exporter_identity_mutation.py` additionally accepts `--selfcheck` (proves every
lineage-failure condition fires) and `--demo-lineage-failure` (proves a lineage failure
exits nonzero rather than presenting as clean).

### Monkeypatching is a measurement device

Three scripts temporarily replace `strict_quarantine._drop_quarantined_processes` and
`_degree_zero_exports` to compute the counterfactual without editing the module under
test. **This is not a proposed patch.** C-010 implements the fix properly inside
`strict_quarantine.py`; these scripts only answer "what would change?". Every one restores
the originals in a `finally` block.

---

## `c011_freeze_seam_before.json`

What the canonical-freeze block **did at BASE `72ee20f`**, before C-011 lifted it out of
`run_post_pipeline_sbml_artifacts` into module-level `freeze_canonical_payload`. Generated by
`tests/test_c011_freeze_seam_golden_equivalence.py` (run it directly with the venv python),
which executes the BASE blob's own statements — lifted from the BASE AST by line span, never
retyped — over **all 39 legs** of `tests/data/baseline_cohort_manifest.json`, cohort
`pre-implementation-baseline-2026-08-05`. 345,053 bytes, SHA-256
`59c3b21254c49e5edb89a13af6266278921c44cac9fddddf102c02e9d6a24553`.

`ORIGIN_SHA` `9e1b9abe7ba8a1a228558fd03ca6c394cc22c31e` is retained as an **equality witness**:
`run_post_pipeline_sbml_artifacts` is AST-identical there and at BASE (`5792b1c5…`), recorded
in the fixture's `source_equivalence`. The fixture proves the extraction changed nothing
observable at the seam. It says nothing about any leg's biology.
