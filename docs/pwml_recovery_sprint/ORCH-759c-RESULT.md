# ORCH-759c: RESULT (strict-quarantine closure is now bucket-aware: the chlorophyll 3-draw blocker)

| | |
|---|---|
| Lane | `orch759c/quarantine-bucket-closure`, worktree `.claude/worktrees/orch759c` |
| Base | `5d1ea64c` (= `sprint/pwml-recovery`) |
| Commits | `58821368` test-first · `a2fc1408` code · `7b30e564` baseline move (merge rule 4) · docs commit (this file + evidence) |
| Status | **Implemented and tested. Not merged, not pushed, not reviewed.** It needs an independent review of the actual diff. |
| Classification | `product_contract_violation`: a correct `review_required` pathway was suppressed (merge rule 7; PRODUCT_CONTRACT "a valid pathway core suppressed …") |
| Reports | `evidence/g11/ORCH-759c/01`–`23`; logs `evidence/orch759c/NN-*.log`; every report `FINAL SURVIVING COUNT : 0` |

## 1. The defect

`quarantine_and_close` built one keep set per iteration, `referenced | _complex_component_norms(...)`
(base `strict_quarantine.py:2454`). `_prune_entities` kept a row in **any** graph bucket whose
name-union-synonyms met it (base `:1829-1831`). On the chlorophyll-strict leg (PMC5130987,
`runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231`), Stage 1 declared gene rows
`HEMA1`, `CAO`, `GUN4` and `PORB` beside proteins of the same names. Mapping wrapped each protein as a
`single_protein_pathwhiz_wrapper` (`HEMA1 complex` → component `HEMA1`). Admission correctly
quarantined all 28 processes naming the bare symbol (`ambiguous_entity_type:nucleic_acids+proteins`),
so nothing surviving referenced the genes. The component name `hema1` still kept the **gene** row.
`_entity_type_overlaps` then found 4 overlaps, giving `refusal_reasons ["entity_type_overlap:4"]`,
`ok false` and `diagnostic_only`. `freeze_canonical_payload` runs the final gate only when quarantine is ok, so the
driver filed `final_gate_report_missing`. The other overlapping gene names (CHLH, GLK1, …; 8 per the read-only diagnosis) were
pruned normally because no component carries them (`evidence/orch759c/01-base-facts.log`).

## 2. The fix (`src/t2pw/pipeline/strict_quarantine.py`, +21 −4)

A component names a protein. `_component_name`'s own docstring says so; `ir.py` resolves a component
against `proteins_by_name` / `proteins_by_key` / `_by_uniprot` / `_by_db_id` only, and `writer.py`
`_protein_complex_members` against protein ids only. Nothing resolves a component to a compound,
nucleic acid, element collection or complex. So:

- **closure loop**: `keep_norms` is now a per-bucket mapping. `proteins` keeps
  `referenced | component_norms`; every other bucket keeps `referenced`.
- **`_prune_entities`**: accepts either one set (applies to every bucket, the old behaviour) or the
  mapping. The set form is kept because two existing tests call or monkeypatch it with two
  positional arguments (`test_strict_quarantine.py:387` and `:1087`, base line numbers); a third parameter would have
  broken the monkeypatch.

Untouched: `_entity_type_overlaps`, the refusal (`:2619` base), every admission decision,
`_complex_component_norms`, `_degree_zero_exports`, `_referenced_entity_norms`. No other file in `src/`.
No prompt and no LLM input changes (see F5).

## 3. G9: the leg's own input, base RED → tip GREEN

Fixture `tests/fixtures/orch759c/`: the leg's `final_mapped.json` **verbatim**, gzipped (sha256
`20c61fd7…1526` asserted by the test), plus the leg's Stage-0 `requested_context` copied out of its
`quarantine_report.json`. Replayed at base it reproduces the live decision exactly:
`decision_input_hash sha256:d2ef87f2ae5da361`, counts 19/36/14/28, `entity_type_overlap:4`,
`diagnostic_only` (`01-base-facts.log`).

| | base `58821368` (test-first, base `src/`) | tip `a2fc1408` / `7b30e564` |
|---|---|---|
| `test_g9_chlorophyll_leg_releases_review_required_once_orphan_genes_close` | **FAIL**: `entity_type_overlaps` = 4 rows (`03-g9-red.log`) | pass (`06`) |
| `test_g9_component_name_no_longer_keeps_an_unreferenced_same_named_gene` (8-row synthetic) | **FAIL**: `[{"name": "HEMA1", "buckets": "proteins+nucleic_acids"}]` (`03`) | pass (`06`) |
| 3 × `test_preservation_*` | pass (`03`) | pass (`06`) |

Both failures are on values. Every name the module uses exists at base with the same signature.

What the leg test pins at tip: same `decision_input_hash`, same counts, same admissions digest as base;
`ok true`, `refusal_reasons []`, `review_required`; exactly the 4 gene rows added to
`removed_entities` (`unreferenced_after_quarantine`) and the other 37 base removals unchanged (count and
digest); 9 surviving gene rows; 30 reactions, 25 interactions; and the strict final gate
(`run_strict_post_normalization_gates(..., enforce_all_proteins_connected=True)`, as
`freeze_canonical_payload` runs it) passes with one connected component.

`evidence/orch759c/g9_leg_replay.py` (`05-g9-leg-tip.log`) adds a row-level diff. In one process it runs the tip
and a base emulation (`_prune_entities` handed the old union set). The emulation reproduces the **real**
base `resulting_payload_hash sha256:e1f3bec8621d8069`. The tip's `sha256:be6568f9391467dc` equals the
read-only diagnosis counterfactual's. Rows only in base: the 4 gene rows; rows only in tip: none;
removed locations 53/53, removed states 7/7, admissions identical. Final gate: 148 nodes, 213 edges,
1 component.

## 4. Preservation (labelled; pass at base and tip by design)

| test | claim |
|---|---|
| `test_preservation_a_referenced_gene_is_kept_and_its_overlap_still_refuses` | a gene a surviving interaction reaches (by its synonym `HEMA1 gene`) is kept, and its overlap with protein `HEMA1` still refuses `entity_type_overlap:1`. No gate is relaxed (D-106). |
| `test_preservation_a_protein_named_only_by_a_component_is_kept` | requirement 3: both subunits of a surviving two-subunit complex stay, in both component row shapes (`"P1"`, `{"protein": "P2"}`) |
| `test_preservation_nothing_changes_when_no_names_overlap` | with no cross-bucket names, exactly one orphan per bucket is removed and nothing else |

## 5. Baseline move (merge rule 4, commit `7b30e564`)

At `a2fc1408` two existing tests went red (`07-affected.log`: 2 failed, 554 passed). Both test the
overlap refusal, and both built their overlap with **the mechanism this lane removes**: a compound
that nothing references, kept alive only because a complex component has the same name.

| node | exact delta | assertions |
|---|---|---|
| `tests/test_strict_quarantine.py::test_an_entity_type_overlap_that_survives_closure_blocks_the_export` | `{"name": "GCLC"}` → `{"name": "GCLC", "synonyms": ["glycine"]}`; docstring rewritten to say why | unchanged |
| `tests/test_strict_quarantine_locks_and_scope.py::test_each_refusal_class_gets_its_own_reason` | `{"name": "GPX1"}` → `{"name": "GPX1", "synonyms": ["glycine"]}`; one comment | unchanged |

The compound now survives because a surviving reaction reaches it. `glycine` is an input of
`glutathione synthesis`, and the registry still resolves `glycine` to the glycine row (first wins).
The moved tests pass on **base** `src/` (`09-moved-base`) and at tip (`08-moved-tip`), so the refusal
itself did not move; only the vehicle did. Both tests date from the module's creation (`404cc8d7`,
2026-08-01). No locked decision covers keeping non-protein rows by component name (`DECISIONS.md` searched).

## 6. Blast radius: every archived strict leg on disk

`evidence/orch759c/census_archived_legs.py` (read-only; tip vs base emulation; the leg's own context;
`strict_db=True`, `mode="pathwhiz"`; research legs skipped because research mode applies nothing):

| roots | report | legs | accepted processes | differ |
|---|---|---|---|---|
| worktree `runs/`, primary `runs_papers/`, `runs_validation/`, `runs_smoke/` | `14-census-a2` | 49 | 380 | **1**: the chlorophyll leg, `ok` False→True, +4 gene rows removed, admissions identical |
| primary `runs_verify/` | `15-census-b2` | 72 | 427 | 0 |

(`12`/`13` are the same census before the accepted-process counter was added: identical verdicts.)

## 7. Gates

| # | gate | report | result |
|---|---|---|---|
| 1 | G9 RED at test-first | `03-g9-red` | 2 failed (the two `test_g9_*`), 3 passed |
| 1 | G9 GREEN at tip | `06-g9-green` | 5 passed |
| 2 | affected suites: every file matching `strict_quarantine\|_prune_entities\|entity_type_overlap` plus every caller of `quarantine_and_close` / `run_quarantine_boundary` / `freeze_canonical_payload` / `write_quarantine_artifacts`, plus `test_stage3_gate_report_lifecycle.py` (27 files, AppTest files excluded → gate 5) | `10-affected-tip` | **556 passed, 8 skipped**, 0 failed. The 8 skips are data-driven `pytest.skip`s in `test_strict_failure_replay.py:234/249`, the same 8 in `07` |
| 2 | Chunk E `test_strict_quarantine_real_artifact_replay.py` | `11-chunk-e` | 175 passed |
| 3 | SMOKE | `16-smoke` | **508 passed** |
| 4 | `tests/test_eval_v2_*.py` (6 files) | `17-eval-v2` | 89 passed, 1 xfailed |
| 5 | `test_streamlit_quarantine_boundary.py` alone | `18` | 23 passed |
| 5 | `test_c052_prefreeze_report_at_the_streamlit_seams.py` alone | `19` | 13 passed |
| 5 | `test_c055_rag_loop_wiring.py` alone | `20` | 14 passed |

Chunk D was not run (as instructed). `streamlit_app.py`, `map_ids.py`, `stage_one_ensemble.py` and
`extraction_diagnostics.py` were not touched. No `--heavy-lock`; one job at a time; clean worktree (no `.env`).

## 8. Findings

- **F1. The defect** (§ 1). Closure applied a protein-only name source (component names) to every
  bucket.
- **F2. Two pinned tests encoded the defect as the way to build a surviving overlap** (§ 5). After the
  fix, a cross-bucket overlap can survive closure only when the non-protein row is reached by a
  surviving process through a **synonym**. A bare-name reference is ambiguous, so admission quarantines
  that process.
- **F3. THE LEG HAS A NEXT BLOCKER, PAST THE FINAL GATE. This lane does not clear it.**
  `evidence/orch759c/export_probe.py` (`21`, detail `22`) runs Chunk E's own exporter
  approximation on the tip's quarantined payload. The required-field contract fails twice with
  `reaction_enzyme_must_be_protein_complex` at `/processes/reactions/21/enzymes/0` and `/modifiers/0`:
  reaction **`CPO reaction`** names the **bare protein `CPO`** (verified, UniProt `Q9LR75`) beside the
  wrapper `CPO1`. `CPO1` is an Unknown-backed cross-species placeholder
  (`pathbank_unknown_protein_fallback`, `real_mapping_failure_reason: species_mismatch`)
  (`23-cpo-diag.log`, `evidence/orch759c/cpo_diag.py`). The IR reports the same code 4 times. The app's
  export runs the same contract on the quarantined payload (`streamlit_app.py:4788`
  `validate_pre_export` → `stage_contracts.py:352-355` `validate_required_pwml_contract`). So on
  this draw the leg would likely now stop at **"PWML required-field gate failed"** instead of
  `final_gate_report_missing`. The diagnosis's "strict final gate passes" was true. It does not mean
  a PWML would be written. The causes sit in mapping (why the verified `CPO` was not wrapped, and why a
  placeholder `CPO1` was created beside it) or in quarantine's contract alignment (it does not check
  enzyme kind). Both are outside this lane.
- **F4. Component kinds.** The schema (`schema.py:246-263`, `payload_models.py:220-235`) gives
  components `name/protein/protein_name/component/entity/gene/gene_name`. Every consumer resolves
  them to proteins only, so no other bucket legitimately takes a component name.
- **F5. No LLM-bound change.** In production, quarantine runs at the freeze seam, in
  `run_pwml_export`, and in the RAG round's `_classification` stage (`streamlit_app.py:1104-1141`).
  That stage writes `channel["classification"]`, which nothing reads (grep over `src/t2pw`), and
  returns the graph unchanged. So a different `ok` cannot change a later LLM call.
- **F6.** The live `admitted_payload_hash` (`d33121c9…`) differs from the replay's (`b43803ad…`)
  because `final_mapped.json` is not the exact in-memory object the app hashed. The decision input hash,
  which the decisions are taken on, is identical. Recorded in the fixture README.

## 9. Failures and dead ends

- `02-draft-base` / `04-draft-tip` are draft runs of the new test file before it was committed. They
  checked that the synthetic fixtures behave as designed (2 fail / 3 pass at base; 5 pass at tip). The
  official runs are `03` and `06`.
- `07-affected` is red (2 failed). That is F2, found by the gate, and fixed by the baseline move rather
  than by narrowing the fix. **Considered and rejected:** letting component names keep
  `protein_complexes` rows as well (the diagnosis counterfactual left that bucket alone). No consumer
  resolves a component to a complex, and neither failing test involved one. The census found no leg
  where it matters. Recorded as open question 1.
- The census was first written without a process counter (`12`, `13`). 72 legs in 2.8 s looked too fast
  to trust, so sizes were checked (median 170 KB; none under 2 KB) and the counter added (`14`, `15`: 807
  accepted processes across 121 legs). The verdicts did not change.

## 10. NOT settled (for the product owner)

1. **A complex named only as a component of another complex is now pruned.** At base it was kept
   at degree zero (`_degree_zero_exports` exempted it). Now it is removed as `degree_zero_after_quarantine`.
   No consumer can represent a nested complex, and the census found no affected leg.
   *Options:* (a) accept (cost: none measured); (b) add `protein_complexes` to the component-keep
   buckets (one token; cost: re-admits a degree-zero complex the IR cannot attach to anything).
2. **A non-protein row named only by a component that resolves to no protein row** (the component is
   unresolved) is now pruned. At base it was kept without refusal. The IR reports
   `component_protein_unresolved` either way. Census: no affected leg. Same options and costs as 1.
3. **The quarantined gene-regulation interactions are lost biology.** Examples: `HY5 activates HEMA1`,
   `PIF1 inhibits GUN4`, `PIF1 binds CAO promoter`. They are quarantined because Stage 1 named the gene
   and the protein with the same bare symbol. This lane does not change that (D-106: no gate relaxed).
   *Options:* (a) accept the smaller pathway (PRODUCT_CONTRACT "a smaller supported pathway is
   preferable"); (b) an upstream lane disambiguates gene rows (e.g. `HEMA1 gene`) before admission
   (cost: mapping/normalizer work, and a risk of renaming real proteins).
4. **F3's CPO blocker** needs its own lane and its own classification before any code change.
5. **The moved tests' vehicle is artificial**: a compound named `GCLC` with synonym `glycine`. A more
   natural vehicle (a gene reached through a synonym, as in preservation test (a)) would change the
   expected `buckets` string and add a protein with an accession. The reviewer may prefer it.
6. **Two docstrings now describe the old single set and were left alone to keep the diff inside the
   closure seam.** `_degree_zero_exports` says "`referenced | exempt` here is the same set the closure loop
   calls `keep_norms`"; it is now the `proteins` entry of that mapping. The test helper
   `test_strict_quarantine.py::_closure_keep_norms` says "exactly as the closure loop builds it"; it
   builds the old union, and its fixtures have no cross-bucket names, so its results are unaffected.
7. **Not re-run end to end.** No pipeline run was allowed. The next 3-draw chlorophyll re-run is what
   proves the leg's outcome, and F3 predicts it still fails at the required-field gate.

## 11. TEST_MATRIX delta (proposed; this lane does not edit TEST_MATRIX.md)

Append at end of file:

```markdown
## ORCH-759c: test obligations, appended end-of-file by rule · 2026-10-22

Lane `orch759c/quarantine-bucket-closure` (`58821368` test-first, `a2fc1408` code, `7b30e564` baseline
move). Reports `evidence/g11/ORCH-759c/01`–`23`.

**SMOKE is unchanged at 508** (`g11/ORCH-759c/16`). Merge rule 10 unchanged. Chunk A count unchanged (134).

### New focused selection

`tests/test_orch759c_quarantine_bucket_closure.py` (5, ~1 s), fixture `tests/fixtures/orch759c/`
(the chlorophyll-strict leg's `final_mapped.json`, gzipped, sha256 asserted). It is in no chunk and
not in SMOKE. G9: 2 `test_g9_*` red at base (`g11/ORCH-759c/03`), 5 green at tip (`06`). Run it for
any change to `_prune_entities`, `_complex_component_norms` or the closure loop's keep set in
`strict_quarantine.py`.

### Pinned tests moved (merge rule 4)

`test_strict_quarantine.py::test_an_entity_type_overlap_that_survives_closure_blocks_the_export` and
`test_strict_quarantine_locks_and_scope.py::test_each_refusal_class_gets_its_own_reason`: the overlap
compound gains `"synonyms": ["glycine"]`; assertions unchanged. Green on base `src/` (`09`) and tip (`08`).

### Known failures

None new. Affected sweep, 27 files: 556 passed, 8 skipped (data-driven skips in
`test_strict_failure_replay.py`) (`10`).
```

## 12. Evidence index

| artifact | what |
|---|---|
| `evidence/orch759c/base_facts.py` → `01-base-facts.log` | base facts on the leg (ran from `C:/t/bt-orch759c/`; copied verbatim) |
| `evidence/orch759c/g9_leg_replay.py` → `05-g9-leg-tip.log` | tip vs base-emulation row diff, final gate |
| `evidence/orch759c/census_archived_legs.py` → `14`, `15` (`12`, `13`) | blast radius over 121 archived strict legs |
| `evidence/orch759c/export_probe.py` → `22` (`21` ran it before the per-error detail lines were appended; same codes) | F3: the next blocker past the final gate |
| `evidence/orch759c/cpo_diag.py` → `23-cpo-diag.log` | F3: the CPO / CPO1 rows (ran from `C:/t/bt-orch759c/cpo.py`; copied verbatim) |
| `tests/fixtures/orch759c/README.md` | fixture provenance and checksums |
