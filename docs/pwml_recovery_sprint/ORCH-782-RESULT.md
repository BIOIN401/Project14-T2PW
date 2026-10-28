# ORCH-782 RESULT: split a gene from a same-named protein (KI-011)

Branch `orch782/gene-protein-split` off `5e1edb3f`. Code commit `543880ec`. Not merged, not pushed.
This implements the product owner's ruling on KI-011 as the orchestrator relayed it (gene row becomes `<SYMBOL> gene`, each
reference resolved by role, an undecidable one omits only its own row).

## 1. What changed

`src/t2pw/pipeline/strict_quarantine.py` (+72 lines, nothing removed):

- **New `_split_gene_protein_names(working)`.** It runs at the top of `quarantine_and_close`, on the deep copy, before
  `_build_registry`. Both app callers (`streamlit_app.py:1833`, `:4711`) go through this point. What it does:
  - A `nucleic_acids` row whose name equals a `proteins` row's name is renamed `"<SYMBOL> gene"`. Its
    `element_locations.nucleic_acid_locations` rows are renamed the same way.
  - **Interaction target** (`right` / `entity_2` / `target`): if the process name or `evidence` matches
    `_GENE_ROLE_CUE` (`express*`, `promoter(s)`, `transcription`, `transcribed`), the target becomes the gene.
  - **Every other reference keeps the bare symbol**, which now names only the protein. That covers enzymes, modifiers,
    transporters, complex components, reaction inputs and outputs, cargo, and the interaction regulator (`left` /
    `entity_1`).
  - **Undecidable** (a target with no cue, or a non-string target): the function returns a forced verdict for that one row,
    `quarantined_broken_reference` / `ambiguous_entity_type:nucleic_acids+proteins`. That is the existing reason code, and
    `reaction_evidence.py:107` already maps it to KI-011.
- **`_admit_processes` takes a new `forced=` argument** and checks it just before `_essential_participant_verdict`. A forced
  verdict can only quarantine a row.
- **No gate was changed or relaxed.** `ambiguous_entity_type` and `_entity_type_overlaps` are untouched and still catch any
  clash this step leaves. In research mode the original payload is still returned (`:3608`), so the rename is never applied
  there.

## 2. G9 proof (correction of observable behaviour)

`tests/test_orch782_gene_protein_split.py` has 5 tests:

- `test_g9_*` (4 tests): a gene-role target becomes `X gene`; an enzyme keeps the protein; an undecidable target omits only
  its own row; on the real chlorophyll leg, rows are recovered one by one.
- `test_preservation_unrelated_rows_untouched` (1 test).

The real-shape input is the committed ORCH-759c fixture: the chlorophyll leg's quarantine input, verbatim.

| run | tree | result | report |
|---|---|---|---|
| base, final test text | `5e1edb3f` (worktree `C:/t/orch782/base`) | **4 failed, 1 passed** (all 4 `test_g9_*` fail on values; preservation passes) | `evidence/g11/ORCH-782/04-g9-base-r3.json` |
| tip | `543880ec` content | **5 passed** | `evidence/g11/ORCH-782/05-g9-tip-r2.json` |

## 3. Tests run (all bounded, 0 survivors, `g11_evidence.py check --task ORCH-782 --require-pin`: 12/12 compliant)

| selection | result | report |
|---|---|---|
| all 26 non-AppTest files that import `strict_quarantine` (my test file included) | **687 passed, 8 skipped** | `09-affected-sq-r3.json` |
| `tests/test_streamlit_quarantine_boundary.py` (AppTest, run alone) | **24 passed** | `10-apptest-boundary.json` |
| `tests/test_c055_rag_loop_wiring.py` (AppTest, run alone) | **14 passed** | `11-apptest-c055.json` |
| moved ORCH-759c pins, run on base | exactly the 4 moved tests fail, 7 pass | `12-moved-759c-base.json` |

SMOKE (merge rule 10, 508 tests) was **not run**, because the brief limited the selection. It is still owed before merge.

## 4. Baseline move (merge rule 4): `tests/test_orch759c_quarantine_bucket_closure.py`

Same chlorophyll input, base → tip:

| measure | base | tip |
|---|---|---|
| core accepted | 19 | 26 |
| auxiliary accepted | 36 | 47 |
| quarantined broken-reference | 28 | 10 |
| admissions sha | `ba2aaf41…` | `8cbc88d2…` |
| interactions kept | 25 | 43 |
| `nucleic_acids` rows kept | 9 | 20 |
| gene rows still orphaned | 4 (`CAO`/`GUN4`/`HEMA1`/`PORB`) | 2 (`CAO gene`, `PORB gene`) |
| removed-entity rest | (37, `45818234…`) | (25, `b995f2cf…`) |
| removed-location rest | (53, `66891e59…`) | (43, `2ef23a5a…`) |
| `n_components` | 1 | 2 |

- `decision_input_hash` did not change.
- One test was renamed, from `test_preservation_a_referenced_gene_is_kept_and_its_overlap_still_refuses` to
  `test_delta_orch782_a_referenced_gene_is_renamed_and_no_overlap_remains`. Its vehicle no longer overlaps. The overlap gate
  is still pinned by `test_strict_quarantine.py::test_an_entity_type_overlap_that_survives_closure_blocks_the_export` and
  `test_strict_quarantine_release_seam.py`, and both pass.
- The new values were computed by bounded probe runs. Those runs have no G11 report; the committed test runs `09` and `12`
  are the evidence.

## 5. Findings

1. **18 rows recovered on the chlorophyll fixture** (bounded probe, base vs tip diff). Examples: `HY5 activates HEMA1`,
   `PIF3 inhibits HEMA1/CHLH/GUN4`, `TOC1 inhibits CHLH`, `GLK1 activates HEMA1 promoter`, `GUN4 activates Mg-chelatase
   complex`.
2. **Two rows now fail for a different reason.**
   - `reactions/17` (enzyme `ALAD1/HEMB1`) now fails as `protein_missing_external_identity`. It is judged as the protein, and
     the protein has no identity (another KI).
   - `GUN4 binds Mg-chelatase` now fails as `undeclared_entity_in_entity_2`.
3. **9 KI-011 rows remain, all with protein-level wording.** For example `FLU inhibits HEMA1`: "interacts with HEMA1-encoded
   GluTR1". Others: `GBP binds/activates HEMA1`, `GUN4 binds CHLH`, `GUN1 activates ABI4`, `Singlet oxygen activates
   FC1/HEMA2`, `FHY3 activates HEMA1`.
4. **`n_components` went from 1 to 2.** A gene is now a separate element from its protein, so a regulation that touches only
   genes can form an island. The island has 5 of 180 nodes (main component 97.2%). The final gate did not raise and the
   release is still `review_required`.
5. **ORCH-781 `GluTR reaction` is not KI-011 gene/protein.** Its quarantine reason is
   `ambiguous_entity_type:compounds+nucleic_acids` (`Glu-tRNAGlu` was declared as both a compound and a nucleic acid). This
   lane does not touch that.

### Dead ends

- **First base run.** `01` (5 failed) hit a test bug: the preservation test indexed reactions by position, and the base
  prunes reaction 0. `02` and `03` had a second bug: a duplicate reaction name was looked up by name. Both bugs were fixed
  in the test, never in the code. `04` and `05` are the valid pair.
- **`06` errored 69 times.** The `--basetemp` parent directory did not exist. This was infrastructure, not a regression;
  `07` diagnosed it.
- **Forced-verdict placement.** I tried placing it after `_essential_participant_verdict`. That made an undecidable row
  report a protein-identity reason instead of KI-011, so I put it back before.

## 6. NOT settled (for the product owner)

1. **Protein-level targets are still omitted.** Wording such as "interacts with", "subunit" or "binding of … domain" still
   leaves the target undecidable (finding 3: 5 of the 9 rows).
   - (a) Accept.
   - (b) Add a protein cue list. Cost: a second lexical list, and lexical gates do not converge under red-team review.
2. **The cue is matched over the whole quote, not next to the symbol.** A quote that says "expression" about a different
   gene would still make this target the gene.
   - (a) Accept (no case seen on chlorophyll).
   - (b) Require the cue next to the symbol. Cost: more misses.
3. **The regulator (`left`) is always the protein**, even if the quote calls it a gene. No such case was seen.
4. **Only gene-vs-`proteins` clashes are split.** Gene-vs-`protein_complexes` and compound-vs-gene clashes (finding 5) are
   untouched.
5. **The island in finding 4.** Is a multi-component PWML acceptable for gene regulation? PathWhiz imports it.
6. **Not re-run end to end.** SMOKE and a live chlorophyll leg are still owed.
7. **Base worktree `C:/t/orch782/base` is left registered.** It is clean, but `git worktree remove` was refused by the
   permission classifier in this session. The orchestrator should remove it.

## 7. Orchestrator addendum: review, merge tests, open questions

- **Review:** pwml-reviewer, one round: APPROVE, no blocking findings. It verified the G9 exit codes and the § 4 pin delta against the diff.
- **Merge tests:** the integration tree (782 + 783) ran 12 affected files: 315 passed, and the 2 failures were both ORCH-783's (see ORCH-783-RESULT § 5). SMOKE ran on the final merged tree `cac97171`: **511 passed** (`evidence/g11/ORCH-783/06-smoke-final.json`, 0 survivors). This closes § 6 item 6 except for the live chlorophyll leg, which the ORCH-784 paper run covers.

### Open questions for the product owner (orchestrator additions)

1. **"transcription" also matches "transcription factor".** The cue `transcription\b` is matched over the whole quote. A protein-protein interaction whose quote calls the regulator a transcription factor would wrongly make the target the gene. This widens § 6 item 2.
   - (a) Accept.
   - (b) Exclude "transcription factor" from the cue. Cost: one more lexical rule.
2. **Gene-only island.** The PWML is now 2 components instead of 1 (§ 6 item 5).
3. **`C:/t/orch782/base`** is still registered. The Lead is raising its removal with the user; it was not retried.
