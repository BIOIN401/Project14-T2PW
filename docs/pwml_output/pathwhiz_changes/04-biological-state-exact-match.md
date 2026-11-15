# 04 Match biological states exactly (omitted component = nil)

Status: **made**, local branch only, not pushed. File: `lib/pwml_parser.rb`, one statement (+11 lines with comment, -1). Patch copy: `04-biological-state-exact-match.patch` (git format-patch of the commit).

## Problem
`PwmlParser#store_object` finds an existing row with `klass.find_by(attribute_hash)`. `attribute_hash` comes from `to_hash_shallow`, and `add_content` skips every element whose content is blank, so a biological state whose tissue, cell type or subcellular location is omitted (or written `nil="true"`) has no key for that column. `find_by` then treats the column as a wildcard: the state binds to the first existing state that agrees on the components that were given, for example a (species, location) state binds to a (species, tissue, location) sibling. The state is not created, its references go to the wrong state, and two states that differ in the file become one row. This is D-W21(a).

The model already defines a state by its exact tuple: `BiologicalState` validates `species_id`, `tissue_id`, `cell_type_id` and `subcellular_location_id` each unique in the scope of the other three (Rails matches nil as `IS NULL`), and `BiologicalState.find_existing` does an exact `where` on all four columns. The importer is the only place that matches loosely.

## Decision: sound, change made
Is exact matching sound for PathWhiz PWML in general? Yes, for these reasons.
1. It is the model's own identity rule (above). A PWML from PathWhiz itself writes every column of a state, `nil="true"` for empty ones, so "omitted" and "nil" mean the same thing there.
2. `pwbs_id` does not protect real files. In the four PathBank controls the `pwbs-id` is shared by many states with different components (PW741182: PW_BS000024 on 32 states, PW_BS000115 on 35; PW012926 and PW176888 the same). `find_by` therefore stays loose on the other columns even for real exports.
3. Measured: with the change, the number of biological states created equals the number of states in the file for all four controls (273, 183, 351, 18). Before, 11, 7 and 9 states of the three larger files were silently merged into siblings.
4. Where it differs from the old behaviour is only a state that omits a component while another state has it set: that state used to vanish into the sibling, now it is created or bound to the row whose component is NULL.

## Evidence (harness, empty database, 2026-11-13)
Harness `tools/pathwhiz_import/run_import.py`, `--smpdb-src` a throwaway worktree = tip of 03 (before) or tip of 04 (after), each plus the owner's uncommitted 10-line `element-id` change (`owner_only.diff`) applied as an uncommitted change. The 03-tip-plus-owner file has sha256 `eb00ccaa...ffae` (equal to the owner's working copy); the 04 file is that plus the 04 change only. Both through `bounded_run.py --heavy-lock pwi`. 39 files imported twice: the 20 files with wildcard-bindable states from the writer replay `C:/t/ev-fp/p5/replay/writerA_on` (idx 3 5 6 7 9 10 48 76 116 119 120 122 126 127 129 139 144 146 150 151), idx 23 37 98 (changed by the writer guard, no bindable state), 12 files with no bindable state (4 8 12 16 20 29 40 56 74 91 107 130) and the 4 controls. Raw: `docs/pwml_output/evidence/final_payload_p5/pwpatch2/` (`before.results.jsonl`, `after.results.jsonl`, `compare.out.txt`, `distinct.out.txt`, scripts).

All 39: success true before and after. Totals before -> after: messages 107 -> 96, clean files 24 -> 24, transports created 40 -> 43, biological states 903 -> 939, transport transporters 20 -> 22. Importer wall time 996 s and 957 s (noise).

Per file, only files that differ (msgs = error messages, bs = biological states created, tr = transports created; every table not named is identical):

| file | msgs | bs | tr | other tables that changed |
|---|---|---|---|---|
| 3 | 7 -> 4 | 5 -> 6 | 4 -> 5 | the new transport and its elements/visualizations |
| 10 | 7 -> 7 | 5 -> 7 | 6 -> 6 | none |
| 119 | 1 -> 1 | 5 -> 6 | 2 -> 2 | none |
| 120 | 0 -> 0 | 1 -> 2 | 0 -> 0 | element_states 4 -> 7 |
| 146 | 0 -> 0 | 5 -> 7 | 1 -> 1 | none |
| 150 | 8 -> 4 | 7 -> 8 | 4 -> 5 | the new transport, transporter and their visualizations, element_states 8 -> 9 |
| 151 | 5 -> 1 | 7 -> 8 | 2 -> 3 | same kind as 150 |
| PW012926 | 1 -> 1 | 262 -> 273 | 1 -> 1 | none |
| PW741182 | 0 -> 0 | 176 -> 183 | 4 -> 4 | none |
| PW176888 | 2 -> 2 | 342 -> 351 | 4 -> 4 | none |
| PW687775 | 3 -> 3 | 18 -> 18 | 0 -> 0 | identical in every table |

In 3, 150 and 151 the messages that disappear are `transport_element: Biological state must have unique biological states` (both transport elements had been bound to one collapsed state), the transport "elements is too short" that followed, and the dependent visualization messages. No message appears that was not there before.

Files with no change in success, messages and every created count: the other 13 of the 20 wildcard-listed files (5 6 7 9 48 76 116 122 126 127 129 139 144) and all 15 files with no wildcard-bindable state (12 sampled plus 23 37 98).

The 13 unchanged "bindable" files: the replay checker counts identical pairs and states with a more specific sibling anywhere in the file as bindable. Identical tuples bind to the same row before and after, and a more specific state created after the general one is a different row both ways, so neither changes the result (the importer only sees earlier rows). Several also have ontology-less locations or cell types that the empty harness database cannot create (see Risk). This is the explanation that fits the numbers; it was not traced file by file.

## How to verify
1. `ruby -c lib/pwml_parser.rb` in a throwaway container of `smpdb-app:latest`: `Syntax OK` (done, tip of 04).
2. Import `reference/PW741182.pwml` (or any file where a state omits a component next to a state that has it) before and after and compare `biological_states` created: before 176, after 183 for PW741182, equal to the number of `<biological-state>` elements.
3. Regression: any file whose states all have the same components set (or no wildcard-bindable state) imports identically.

## Risk
Low. One class (BiologicalState), one lookup, no other code path. Behaviour changes only for a state that omits a component and has an earlier, more specific sibling with the same other values (and the same `pwbs_id` when the file gives one).
- In production the lookup still includes `pwbs_id` when the file has one, as before. A real export re-imported into a database where the same PW_BS id sits on a row with different components creates a new row after this change (before it bound to that row if its given components matched). Not tested against a production database.
- States the harness cannot resolve (vocabulary rows without an ontology id are rejected by the empty database) have their component dropped to nil; two such states can then be exactly equal and merge. This is the harness limitation described in its README, not caused by this change; files 3, 10, 48, 76, 139, 150 show it (created states stay below the number of distinct state tuples in the file).
- The other classes still match with a bare `find_by(attribute_hash)`; whether any of them has the same issue was not examined (e.g. a Species or Compound given by name only).
- The T2PW writer keeps its own wildcard guard regardless (real PathWhiz will not have this patch).

## Not established
- Behaviour against a populated production database; the harness DB is empty (only states created within the same file can be wrongly bound).
- Rendering of the changed files (not opened in the viewer).
- Files 126 and 48 and 139 were predicted by the file's state ids to change but did not; the cause is the ontology-less vocabulary rows (above), checked for 126 only by reading its states.
- The full 59-file set was not run (the focused result is positive, and every file with a bindable state in the replay is already in the 39).

## Branch
`t2pw/04-biological-state-exact-match`, commit `80bdaca7`, on top of `t2pw/03-transport-transporter-remap` (`c5c12d69`). Local only, not pushed. Worktree `C:/t/pwpr/04`. Test worktrees `C:/t/pwpr/h03` and `C:/t/pwpr/h04` (detached, uncommitted owner change applied; throwaway).
