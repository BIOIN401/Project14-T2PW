# writerA (P5, D-W21 a/b): wildcard guard; id-less cell types and tissues dropped from the PWML only

Code commit `0c320e27` (`src/t2pw/pwml/writer.py`, `tests/test_final_payload_wildcard_guard.py` new, `tests/test_final_payload_ontology.py`).

## What
`DeterministicPwmlBuilder._apply_wildcard_guard`, flag-gated (`final_payload_enabled()`, flag off byte-identical: no droppable rows, the method is never called).
- Droppable rows: location, tissue, cell type with vocabulary outcome `unmapped`, no database id, no ontology id.
- Per state, view = (species id, tissue, cell type, location) with dropped rows omitted; token of a row = (casefold name, ontology id).
- Violation of state i (only states that lost a component): its remaining non-species components are all empty (species alone is a wildcard), or they are a subset of (equal values on its set components) some other state's view, unless that already held for the full components (a pre-existing wildcard or identical pair that keeping cannot fix).
- Fixpoint: while a state violates, restore its first dropped row (order: location, tissue, cell type). Restoring never makes another state newly bindable (every state referencing a row gains it together), so the loop ends, with every row kept at worst. Joint drops are therefore evaluated together and only as many rows are restored as needed.
- Dropped rows leave their section and their key id; states referencing them are written without the component.
- Counters: `locations_dropped_no_id` (kept name), `tissues_dropped_no_id`, `cell_types_dropped_no_id`, `components_kept_wildcard_guard` (kept rows of all kinds). `locations_kept_state_collision` is kept for kept locations (an existing test asserts it, including for the species-only case), so a kept location is counted in both.
- Importer facts read (`smpdb`, read only): `store_object` does `klass.find_by(attribute_hash)` and maps the XML id to the found row. The models `CellType`, `Tissue`, `SubcellularLocation` (`app/models/data/biological_states/`) validate `ontology_id` presence, uniqueness and format (`CL:`, `BTO:`, `GO:` plus 7 digits), so creating a row without an ontology id fails with "Ontology can't be blank"; `BiologicalState` validates uniqueness of each component within the scope of the other three (identical states collapse) and `has_at_least_one_component`.

## Existing test edited (flag it to the reviewer)
`test_final_payload_ontology.py`: `test_writer_drops_idless_location_and_keeps_state_flag_on_only` and `test_idless_location_kept_when_dropping_would_make_two_states_identical` expected an id-less location to be dropped although the state without it, (species, tissue), is a subset of a sibling state (species, tissue, location): exactly the wildcard bind D-W21(a) forbids. Their sibling state now has a different tissue (`leaf`, helper `_TISSUES`), so the drop is safe and every original assertion is unchanged. No assertion was weakened; the setup was changed.

## Tests (bounded wrapper `runtests5.sh`, 0 surviving processes)
`tests/wA3`: wildcard_guard + ontology + writer_fixes + vocabulary: 48 passed. `tests/wA4`: the same plus `test_pwml_writer.py`: 75 passed, 1 failed (`test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance`: asserts `db_resolution.available is False` because "no worktree carries a .env"; `C:/t/fp/.env` exists now. Environmental, not touched by this change; the file was not read). wA1/wA2 are earlier runs with failures I then fixed (the two tests above, and my first tests, which forgot that every IR state needs a location).
New tests (9): wildcard subset bind (location kept), cell type dropped when safe, tissue dropped when safe, identical-after-drop kept (2 rows), species-only kept, both id-less (location restored, cell type dropped), joint drop restores exactly one, one drop alone is safe, flag off unchanged, rows with an id never dropped.

## Replay, 152 stored runs, flag on, code = `0c320e27` exported to `C:/t/ev-fp/p5/srcA` (src differs from P4 d3's `0727df29` only in writer.py)
Output `C:/t/ev-fp/p5/replay/writerA_on/` (not committed; `cmp_wa.out.txt` here). 97 written, same set as d3 on.
- Files whose PWML bytes differ from d3 on: 28 of 97; but only **9** differ in the sections the guard touches (cell-types, tissues, subcellular-locations, biological-states): idx 3, 10, 23, 37, 98, 119, 146, 150, 151. The other 19 differ only outside them (compound names/short names, e.g. "Pyruvic acid" vs "pyruvate"): the source differs in writer.py only, so this is the replay environment (cache or database state) changing since d3, not the writer. Not investigated further.
- Counters (summed over written runs; d3 in brackets): `locations_dropped_no_id` 3 (4); `tissues_dropped_no_id` 1 (0, in 1 file); `cell_types_dropped_no_id` 15 (0, in 8 files); `components_kept_wildcard_guard` 25 (0, 13 files); `locations_kept_state_collision` 19 (18, 11 files).
- Species-only states in the output: **0** (d3 0, flag off 0).
- Wildcard-bindable states in the output (a state whose set components all equal another state's, identical pairs included; `scripts/cmp_wa.py`, `bind_per_file.py`): 63 in 20 files (d3 on 65 in 20, flag off 59 in 19). So **not 0**: these are not caused by the drop. 59 exist with the flag off (stored states that are already subsets of siblings, e.g. a state with only a location next to states with tissue and location). The 4 extra against flag off are in two files: idx 3 (+2) and idx 10 (+2), where the flag-on vocabulary mapping (P3 B2) renames "taste buds" to "taste bud" and "apical membrane" to "apical cell membrane", which makes two stored states identical before any row is dropped; the guard cannot fix those (it compares against the mapped full components).
- Errors: 97 written as in d3; no new non-stage3 error classes (required-field gate failures are the same unwritten runs).

## NOT ESTABLISHED
- The importer was not run (another task); that a kept or dropped component imports cleanly was not tested here.
- The checker works on the written PWML, where an omitted component and an originally absent one are indistinguishable, so "bindable" counts also include stored states that were bindable before this change.
- The 19 files that differ outside the guarded sections were not traced to a cause (environment hypothesis only).
- Token equality is by name plus ontology id, coarser than row id for tissues and cell types (more rows kept, never fewer); two rows with the same name but different database ids count as equal.
- Restoration order (location, tissue, cell type) is a choice; a different order could keep a different, equally small set.
- Species-only for cell type or tissue cannot be reached through `build_pwml_ir` (every IR state needs a location), so that case is tested through the location only.
