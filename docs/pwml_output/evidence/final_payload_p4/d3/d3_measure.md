# d3: species-only fix measured (code HEAD 0727df29)

Fix (writer.py, drop block): the no-component test is now over `rest[1:]` = (tissue, cell type). Tuple layout verified: `rest = (species, tissue, cell_type)`, `loc` separate. Tests: `test_final_payload_*.py` + `test_pwml_writer*.py` 122 passed, 0 surviving processes (`d3tests.*`). Existing tests changed: `test_writer_drops_idless_location_and_keeps_state_flag_on_only` and `test_idless_location_kept_when_dropping_would_make_two_states_identical` (their species-only states now also carry a tissue, so the id-less location is still droppable; `_build` got an optional `tissues` arg). `test_final_payload_vocabulary.py` unchanged. New test: `test_idless_location_kept_for_species_only_state_dropped_with_tissue`.

## Replay (152 files, flag on and off, HEAD 0727df29)
* Flag off vs P3 stored flag-off: 97 of 97 byte identical.
* Flag on: 97 written, same set as P3; 9 of 97 differ from P3 (d2: 13): idx 3, 4, 7, 10, 17, 119, 126, 150, 151.
* Species-only (wildcard) states in the flag-on set: 0 (P3 0, d2 8). Flag-off 0.
* locations_dropped_no_id 4 (d2 12); locations_kept_state_collision 18 summed over 11 files (counter now also counts the kept no-component case).

## Import (59 flag-on files, 5 controls, Docker project t2pw-pwi, stopped afterwards)
| | P3 | d2 | d3 |
|---|---:|---:|---:|
| clean files of 59 | 43 | 46 | 45 |
| total messages (categories) | 147 | 95 | 94 |
| vocabulary | 62 | 32 | 38 |
| transport family | 21 | 25 | 18 |
| of which left=right | 6 | 7 | 5 |
| interactions | 21 | 9 | 9 |
| nucleic acid | 38 | 24 | 24 |
| other | 5 | 5 | 5 |
| transports created / expected | 31/37 | 30/37 | 32/37 |
| interactions created / expected | 9/17 | 13/17 | 13/17 |

Controls: PW012926 1 msg, PW741182 0, PW687775 3, PW176888 2, PW1 176 (unchanged from P3/d2).

## idx 10
Transports created 6 of 7 (P3 6/7, d2 4/7), left=right 1 (P3 1, d2 3), messages 12 (P3 14, d2 17). Transports 22 and 26 are back (state 14 keeps location 3 `saliva`, so it no longer collapses onto state 12). The remaining loss is transport 33 (state 21 on both sides), the same one P3 lost; it is a state-identity issue in the extraction, not new.

## Why one file fewer is clean than d2 (45 vs 46)
idx 125 (`microsomal fraction state`): d2 dropped its id-less location (clean), d3 keeps it as in P3, and the empty-DB importer reports `subcellular_location_2: Ontology can't be blank` (a P3-era message, so vocabulary 32 to 38). That is the intended trade: a species-only state would be a wildcard in the real importer. The 6 extra vocabulary messages come from the kept locations (idx 10 +1, 125 +1, 151 +1, others in the 59-file set); not itemised further. On a database seeded with these locations in production the message would not occur.
