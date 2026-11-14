# Reaction-member `side` follows the role (experiment branch `pwml/side-fix`)

Status: **not merged, not for merge.** Prepared for a product-owner decision on `pwml/layout` head `07d23e9c`. Open question at the end.

## 1. What and why

The PWML writer wrote each reaction member's `side` from the serpentine row direction, which flips Left/Right on every second row (`_reaction_visual_side`, `src/t2pw/pwml/writer.py`). PathWhiz reads that attribute as the member's role. Measured on the 97 replayed files (finding: `PWML-LAYOUT.md` 9.4):

* 281 of 1036 reaction members, in 44 of 97 files, had a `side` that named the other of the reaction's own `<reaction-left-elements>` / `<reaction-right-elements>` lists. Same with the final-payload flag off or on, in v1 and in v2 layout.
* After import into PathWhiz, 156 of 426 member rows contradicted their stored role (87 reactants with `side = Right`, 69 products with `side = Left`).

The fix is one statement: where the reaction-member visualization is written, `side = "Left" if role == "left" else "Right" if role == "right" else ""`. It applies in both layout modes and is not tied to `T2PW_PWML_LAYOUT`. The row direction still places the stacks and edges, so no coordinate changes.

Commits (on `07d23e9c`): `3bd33491` code, `74dcb2df` test fix, `0bb147e4` three assertions in `tests/test_pwml_writer.py` (see 6), then the evidence and this record (docs only).

## 2. Per-caller decision

`_reaction_visual_side` at `07d23e9c` (line numbers of that commit), and the other places that emit a `side`:

| where | what it does with the value | what PathWhiz does with `side` for this kind | decision |
|---|---|---|---|
| `writer.py` 2583, reaction compound / nucleic-acid / element-collection member visualization | **writes the `side` attribute** | role. `reaction_visualization.rb` 192-244 `orientation`, 311-363 SBML (`addLeft/RightReactionElement`), 432-480 SBGN; `reaction.rb` 323 `find_element` (`reaction_#{side}_elements`) | **changed to the role** |
| 2258, stack placement per reaction | which side of the enzyme the stack sits on: anchor x, x of each member, `layout_debug_stacks` record, `compound_loc_by_rxn_side` cache key | not written to the PWML | kept: visual |
| 2272, producer side of a shared intermediate | key into `compound_loc_by_rxn_side` (same value as written at 2258) | not written | kept: bookkeeping that must match 2258 |
| 2445, enzyme x fallback when a reaction has no enzyme box | which side of the enzyme the first substrate attaches to | not written | kept: visual |
| 2482, edge target (substrate edges end on the enzyme's left or right edge) | edge endpoint | not written | kept: visual |
| 2640, transport member | already `Left` for left, `Right` for right | role (`transport_visualization.rb` 180-232, 294-347, 422-490) | unchanged |
| 2734, interaction member | already role | role (`interaction_visualization.rb` 160-212, 277-329) | unchanged |
| reaction-coupled transport, sub-pathway | no members are emitted (empty lists) | n.a. | unchanged |
| IR debug at ~1620 and 2322 | interaction's own left/right; debug record only | not written | unchanged |

An enzyme member takes the earlier `role == "enzyme"` branch and gets no `side`. A compound, nucleic-acid or element-collection member whose role is neither `left` nor `right` would get `""` (PathWhiz validates `side` as Left/Right). That was already so; the IR only emits left, right and enzyme for reaction members.

## 3. Proof

### 3.1 PWML (97 replayed files x 4 modes: flag off/on x layout v1/v2; `evidence/side_fix/metrics/`)

| | before (`07d23e9c`, `C:/t/layx/replay`) | after (`3bd33491`) |
|---|---:|---:|
| members (4 modes) | 4144 | 4144 |
| `side` contradicts the reaction's own element lists | **1124** (281 per mode, 44 files per mode, 176 file-modes) | **0** |
| `side` contradicts the product arrow on the edge | 1140 | **0** |
| members on both lists (not counted either way) | 24 | 24 |

`side_pwml.py` (layout package script, unchanged) before and after: `side_pwml_before.json`, `side_pwml_after.json`.

### 3.2 Only `side` changed (`side_only_check.py`, `metrics/side_only_*.json`)

For every file in every mode, the pair (07d23e9c output, fixed output) is compared as canonical XML with the `side` of reaction-member visualizations blanked. Layout is **not** stripped, so coordinates, paths, ids and every other element must be identical.

| mode | files | members | `side` values changed | files changed | files that differ in anything else |
|---|---:|---:|---:|---:|---:|
| off v1 | 97 | 1036 | 285 | 44 | 0 |
| on v1 | 97 | 1036 | 285 | 44 | 0 |
| off v2 | 97 | 1036 | 285 | 44 | 0 |
| on v2 | 97 | 1036 | 285 | 44 | 0 |

285 changed against 281 mismatches: 4 of the 6 members per mode that sit on both lists changed too. The replay results (`replay_results_*.jsonl`) match the layout package's on ok / written / counts. The check script refuses an empty input directory. This is stricter than the layout package's `content_check.py` (which strips geometry but also compares `side`).

### 3.3 PathWhiz import harness (`harness/`, `side_db/`)

Importer `pwml_parser.rb` sha256 `eb00ccaa...` (same as the layout package), same 59 files as the v2.1 pass (`on_v2`, flag on), fixed writer.

| 59-file harness | before (v2.1, `layout/harness/v2_1`) | after |
|---|---:|---:|
| imported (success) | 59 | 59 |
| clean (no messages) | 45 | **45** |
| messages | 94 | **94** |
| transports created / expected | 32 / 37 | **32 / 37** |
| interactions created / expected | 13 / 17 | **13 / 17** |
| files whose messages or created rows differ | | **0** |

(`harness/before_vs_after.md`: the columns headed v1 / v2 are before / after.)

| `side_db.py`, 27 files with a mismatch, flag on, v1 | before (`layout/side/side_db.json`) | after (`side_db/side_db.json`) |
|---|---:|---:|
| member rows | 426 | 426 |
| consistent with the stored role | 269 | **425** |
| side contradicts the stored role | **156** | **0** |
| on both lists | 1 | 1 |

### 3.4 SBML export from PathWhiz, pathway 37 (`sbml/`)

File 37 (cholesterol branch, 4 reactions) was imported alone and exported with `/live/view/PW000001.sbml`, before and after.

* Before: the reaction "4,4-dimethylcholesta-8(9),14,24-trien-3b-ol -> cholesterol" is exported with **cholesterol as the reactant and the diene as the product**. 3 of 4 reactions match the PWML roles.
* After: all 4 reactions export reactants and products as the PWML has them (4 of 4). The other 3 reactions are unchanged.
* Tables: `sbml/37_before_reactions.md`, `sbml/37_after_reactions.md`; raw exports and PWMLs beside them. (The export mangles the beta character in the name; the check compares ASCII only.)

## 4. Tests (bounded wrapper, heavy lock, `evidence/side_fix/tests/`, `logs/`)

At `0bb147e4`, `PYTHONPATH=C:/t/side/src`, `T2PW_OFFLINE_CURATOR=1`, one bounded job, `FINAL SURVIVING COUNT 0` (`logs/suites.bounded.json`):

| suite | result |
|---|---|
| focused: `tests/test_pwml*.py` + `test_final_payload_writer_fixes.py` (incl. the new file) | 249 passed |
| SMOKE (22 files) | **511 passed** |
| golden seam (4 files, unedited) | **38 passed** |
| new test `tests/test_pwml_member_side.py` against the old writer (`07d23e9c`, exported to a scratch tree) | **fails** on the side check: 9 of 16 single-role members carry the wrong `side` (`tests/new_test_on_old_writer.stdout.txt`); passes at HEAD |

The replay and harness jobs each ran bounded with 0 survivors (`logs/after_replay.bounded.json`, `logs/db_after.bounded.json`).

## 5. Review

| # | scope | model / effort | verdict |
|---|---|---|---|
| 1 | `07d23e9c..3bd33491` (`ws-reviewer`) | opus / high | REJECT: F1 the new test's guard `> 20` could not pass (fixture has 16 single-role members); F2 two assertions in `test_pwml_writer.py` still pinned the old side. Notes: stale docstring, the check script passing on empty input, base-tree layout. All fixed in `74dcb2df`, `0bb147e4`. Confirmed: all other `_reaction_visual_side` uses are geometry, role mapping right for every PathWhiz reader, no behaviour change besides `side` |
| 2 | re-review of `07d23e9c..HEAD` | opus / high | see 8 |

## 6. Boundary exception (needs PO confirmation)

The brief fixed the boundary at `writer.py`, the new test and docs. The fix cannot pass its own suites without changing three `side` assertions in `tests/test_pwml_writer.py` (lines 900-901 and 1339: reaction 4 on a right-to-left row, substrate `Right` -> `Left`, product `Left` -> `Right`; the shared intermediate's consumer `Right` -> `Left`). Those assertions pinned exactly the behaviour that is the bug. Only those 3 lines changed, no geometry assertion, in their own commit `0bb147e4`, which can be dropped. `test_pwml_writer.py` is not in the SMOKE list, but it is in the focused run.

## 7. Open question for the product owner

**Apply to default output?** The change touches output with the flag off (today's default) as well as on.

* Yes: SBML and SBGN exports from PathWhiz, its orientation calculation and its stoichiometry lookup become correct for the 44 of 97 files with swapped members; no coordinate, id or other attribute changes (3.2). Cost: the code is 1 statement in `writer.py`; the three test assertions; a decision that PWML bytes change for a fraction of previously generated files.
* No (keep as is): exports from T2PW drawings keep swapping reactants and products for the members on every second row; document it.
* With v1 geometry the reaction's drawn direction on a right-to-left row no longer matches `side`; with v2 geometry (always left to right) drawing and `side` agree. Orientation computed by PathWhiz follows the role either way.

Not tied to the layout flag, so it can be applied independently of `T2PW_PWML_LAYOUT`.

## 8. Re-review

(filled in below after the second review)
