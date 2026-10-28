# ORCH-787 RESULT — `PREFREEZE_DUPLICATE_CANONICAL_ROWS` on chlorophyll (protoheme / heme)

Branch `orch787/fix` off `22538dee`. Rulings applied: Lead brief, then **D-118 § 2** (amends D-035 clause 7,
`95a4f611`). Reports: `evidence/g11/ORCH-787/` (`NN-*.json` wrapper reports, `*.log` the tee'd output the
counts below are read from). Every job: bounded wrapper, `--heavy-lock ORCH-787`, 0 surviving processes.

The first session was blocked for ~45 min on free RAM below the floor (0.73–1.44 GB, no sprint Python
running). After the reboot (~13.5 GB free) everything below was run.

## 1. Cause
- ORCH-784 stopped at `streamlit_app.py:4324` `run_prefreeze_resolution` →
  `prefreeze_resolution._reject_duplicate_canonical_rows`, which refuses the whole pathway.
- Stage 6 stamps `heme` with PathBank 1799 (`legacy_id_unverified`, name kept). The same paper's ORCH-781
  `final_mapped.json` records this. `protoheme` arrives with no id. The pre-freeze DB resolver matches it
  by PathBank synonym to 1799 "Heme", so two rows end up on one name, and they are the same entity.
- **ORCH-782 is not the cause.** `_split_gene_protein_names` runs inside the strict quarantine
  (`strict_quarantine.py`, from `freeze_canonical_payload`, `streamlit_app.py:4364`). That is after the
  failing seam, and it only touches gene and protein names.

## 2. Change (files / functions)
- `src/t2pw/pwml/prefreeze_resolution.py`
  - `_reject_duplicate_canonical_rows(..., resolved)` returns `(proven, unproven)` and never raises.
  - `_proven_same_entity` applies D-035 clause 3:
    - **every** member carries a PathBank id, and there is one id for the whole group;
    - each renamed member got that id from resolution (`db_row.id`);
    - no namespace (PathBank, KEGG, ChEBI, HMDB, PubChem, InChIKey) holds two values across the group,
      reading both the pre-resolution and the post-resolution rows;
    - it never compares names, and does not compare `class` (`cofactor` is a role claim).
  - `_consolidate_rows` applies D-035 clause 5:
    - the survivor is the first row that already held the name;
    - the merged row's spellings become synonyms, and empty fields and provenance are unioned;
    - the fold is recorded in payload `prefreeze_compound_rows_consolidated` and in report
      `compounds.rows_consolidated`;
    - the connectivity proof is kept by mapping the merged row's token to the survivor.
  - **D-118 § 2.** An unproven group's conflicting rows are:
    - the renamed ones when a member already held the name; otherwise all members;
    - each goes back to its extraction row: no rename, and no identity from this resolution;
    - each is marked `prefreeze_duplicate_omission`, carrying the old diagnostic record;
    - each is listed in `compounds.rows_omitted_duplicate`.
    Detection then re-runs until it is stable.
- `src/t2pw/pipeline/strict_quarantine.py`: `_entity_representability` treats a marked row as not
  representable (`duplicate_canonical_row_identity_not_proven`). The existing admission and closure then
  omit the row and every process naming it, per row. No other gate changed.
- `src/t2pw/app/reaction_evidence.py`:
  - a consolidation is listed as a merged duplicate, labelled KI-045;
  - the omission reason maps to KI-015 through `PENDING_CATALOGUE_CODES`, which is consulted only after
    the appendix matches nothing.

## 3. Replay (`evidence/orch787_replay.txt`, script `evidence/orch787_replay.py`)
- **Input:** `merged_payload.json` plus the ORCH-781 Stage-6 compound stamps (11 rows, including heme=1799).
- **PathBank answers:** recorded once, read-only (`02-replay-record`, 84 queries,
  `evidence/orch787/pathbank_compound_queries.json`), then served offline. 236 queries served, 0
  unrecorded, 0 network.
- **Base** (`03`): raises the leg's exact message:
  `PREFREEZE_DUPLICATE_CANONICAL_ROWS ... 'heme' (['protoheme', 'heme'])`. No freeze, no PWML.
- **Tip** (`04`):
  - `protoheme` is merged into `heme`, PathBank 1799; compound rows go 46 → 45.
  - The quarantine passes (`review_required`, 101 quarantined).
  - Export is reached but stops at the required-field gate on `species_missing_taxonomy`,
    `species_missing_classification` and `no_defensible_reaction_support`.
  - These stops come from the replay itself: protein and species identities are not replayed (KI-058),
    so their processes are quarantined.
  - The text record lists `merged duplicate, not counted: protoheme ... (KI-045)`.

## 4. G9 and tests
- **G9 file:** `tests/test_orch787_duplicate_rows_consolidated.py`. Base `05`: **3 failed**, each on the raise.
  Tip `06`: **3 passed**. It has two correction tests (the proven merge, and the unproven per-row omission
  carried through the strict quarantine to the text record with KI-015) and one new-acceptance test (the
  text record line).
- **Affected batch at tip** (`09`, 15 files): **311 passed, 2 failed**. The 2 failures also fail on base
  (`08`):
  - `test_compound_resolution_extraction::test_the_golden_covers_every_committed_leg_fixture` and
    `::test_the_repaired_coverage_and_digest_pins_themselves_go_red`;
  - they fail on 23 unpinned `runs_verify` legs;
  - those legs are therefore not measured by that golden.
- **Run alone at tip:**
  - `c030a` (`10`): 10 passed. `c052` (`11`): 13 passed. `c082_post_pipeline_seam` (`13`): 1 passed.
    `reaction_evidence_app_hook` (`14`): 1 passed. `reaction_evidence_view` (`15`): 11 passed.
  - `prefreeze_third_export_seam` (`12`): 11 passed and 1 failed. The failing test,
    `test_run_pwml_export_runs_the_prefreeze_sequence_once_before_build_pwml_ir`, fails the same way on
    base (`16`).
- **Dev runs:** `07` is the run before the pins moved (10 failed). The first focused dev report (01) was
  discarded.

## 5. Pins moved deliberately (all base → tip, all pre-freeze refusals of `PREFREEZE_DUPLICATE_CANONICAL_ROWS`)
`tests/test_prefreeze_compound_resolution.py`:
| test | base | tip |
|---|---|---|
| `test_new_acceptance_propagation_is_all_or_nothing` | raise, payload unchanged | `gly` unproven (the "Glycine" row has no id): not renamed, marked, no reference moves |
| `test_c050g_new_acceptance_merged_rows_that_are_referenced_still_stop` | raise | both rows CHEBI:15428 → entity 78: **one row** "Glycerol 3-phosphate", reference rewritten, fold recorded |
| `test_c050h_correction_the_code_is_named_and_names_the_rows_and_spellings` | raise with `details.rows` | row 1 omitted per row; the same `rows` record is on its marker |
| `test_c050h_correction_the_graph_is_intact_and_no_pwml_is_emitted` | raise, payload unchanged | returns; processes unchanged, no merge, row 1 marked |
| `test_c050h_correction_the_refusal_is_idempotent` | raises twice | same omission twice, second pass changes nothing |
| `test_c050h_correction_the_d034_leg_still_refuses_on_committed_data` | raise (rows 20/36/38) | rows 36, 38 omitted per row (KEGG C03189 vs C00093, still unproven), row 20 kept, nothing merged |
| `test_c050h_new_acceptance_a_pre_existing_group_renamed_en_bloc_refuses` | raise | both rows ChEBI 9 → entity 78: **one row** "Lipid IVA" |

`tests/test_compound_resolution_extraction.py`, leg `runs/2026-07-28_0919/.../PMC12444477…/strict`:
- configurations A, C, D moved from `GOLDEN_PREFREEZE_STOPS` (`PREFREEZE_DUPLICATE_CANONICAL_ROWS`) to
  `GOLDEN_IR_REFUSALS` (`PWML_IR_DUPLICATE_NAMED_ROW`, the lipid IV_A pair, as B and E already were);
- the digest moved `d22b58c8…` → `7579c65d…`;
- no other leg's digest or stop moved.

`AMBIGUOUS_RENAME_TARGET` is unchanged (PEtN-lipid A / modified Lipid A still raises, D-035 clause 8).

## 6. Dead ends
- Replaying `merged_payload.json` as saved does not reproduce the leg. Without the Stage-6 stamp, `heme`
  is also renamed and the seam raises `AMBIGUOUS_RENAME_TARGET` instead. Hence the recorded stamp carry.
- The first proof accepted a group in which a member carried **no** PathBank id (the A6 Glycine shape
  would have merged). It now requires an id per row.
- The first merge record lived on the survivor row. The quarantine later removed the survivor in the
  replay, which lost the record. It moved to the payload top level.
- `data/pathwhiz_id_db.json` has no heme entry, so the rename is PathBank-DB only.

## 7. Open questions for the product owner
1. The two new reason codes need appendix rows in `PWML_KNOWN_ISSUES.md`:
   - `duplicate_canonical_row_consolidated_same_db_entity` is labelled KI-045 in the meantime;
   - `duplicate_canonical_row_identity_not_proven` is labelled KI-015 in the meantime;
   - or the PO opens a new KI for them. Until then they live outside the appendix table, in
     `reaction_evidence`.
2. In an unproven group where **no** row held the name before (an en-bloc rename), every member is
   omitted. The alternative is keeping the first, but that picks one identity arbitrarily.
3. The tip replay reaches export but writes no PWML, because Stage 6 (proteins, species) is not replayed.
   A live chlorophyll leg is the real proof.
