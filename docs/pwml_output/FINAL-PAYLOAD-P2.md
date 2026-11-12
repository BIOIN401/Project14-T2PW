# FINAL PAYLOAD, PHASE 2: writer fixes for PathWhiz import (2026-11-11 / 12, branch `pwml/final-payload`, worktree `C:/t/fp`, from `e26c8566`)

Goal: make T2PW PWML import cleanly into PathWhiz. Three writer defects from `IMPORT-ERROR-TRIAGE.md` (fix 1 bound visualizations, fix 2 spontaneous, fix 6 dangling compound location), plus, by PO ruling D-W19, triage fix 7 (transporters as protein complex) and two PathWhiz importer fixes. All writer changes sit behind `T2PW_FINAL_PAYLOAD` (default off). Code HEAD `e9c3e197`. Evidence: `evidence/final_payload_p2/`. Nothing pushed from the code branch before this doc commit.

## 0. NOT ESTABLISHED (read first)

1. **Nothing was imported into production PathWhiz.** All imports ran on the local harness with an empty database plus a vocabulary seed (cell types, locations, tissues, species, protein 9659). "Clean" means zero error messages in that harness. The `success: true` flag carries little meaning.
2. **The import comparison is not on the triage's 60 files.** The 60 baseline files are old exports. The measured sets (off, on) are the replay outputs of the same runs, 59 files each (one run writes no PWML). The "baseline files, new seed" column re-imports the original 60 to separate seed from code. The interaction counts differ because the old files carry 70 stale free-text interactions that HEAD withholds (17 interactions in the replay files against 85 in the old ones). Interactions are therefore not a writer effect.
3. **The importer patch is not measured apart from the flag in the "on" column.** The importer effect with the flag off is visible by comparing off5 (old importer) with off6 (patched importer): the written bytes are identical (flag off), so that difference is the importer alone (transports created 0 to 30 of 37). The flag's effect on top of the patched importer is on6 against off6. The individual writer fixes were not each measured alone in the harness; unit tests cover them one by one.
4. **The Ruby edit is syntax-unchecked** (no Ruby on this machine). It was exercised by the harness runs on6 and off6 (the importer ran and created transports), which is evidence it parses, but no `ruby -c`.
5. **`spontaneous` is never written `true`.** `ir.py:2097` hard-codes `False` for every reaction, so with the flag on every reaction is written `nil` (409 of 409). The `True` branch is tested only by injecting into the IR.
6. **The last test commit (`e9c3e197`, one existing-by-this-work test changed to drop its transporter) was not independently reviewed.** It is test-only, in the new test file, and follows the reviewer's own instruction. The review of `2bf0511a` is below.
7. **Review findings 2 to 6 of the fix-7 review were not fixed** (section 6).
8. The earlier P1 caveats stand: live pipeline seam never run, id precision hand check not repeated.
9. Whether PathWhiz accepts an enzyme-less reaction with unknown spontaneity is the triage's open question; the harness accepts it (spontaneous messages 19 to 0).

## 1. What changed

| file | change |
|---|---|
| `src/t2pw/pwml/writer.py:21` | import `final_payload_enabled` (stdlib-only module, no cycle) |
| `writer.py:1373-1376` | fix 2: flag on, `spontaneous` is `True` only when the reaction value `is True`, else `None` (written `nil="true"`); flag off unchanged `bool(...)` |
| `writer.py:1559-1566, 1612` | fix 6: flag on, collect `location_key` of every member of `transport_visualization` and `interaction_visualization`; a reaction-used compound's base location is skipped only when its key is not in that set (so transports and interaction members never reference an unwritten location) |
| `writer.py:1822-1824` | fix 1: flag on, `bound-visualizations` section emitted empty after the layout-region maths (IR rows and regions kept) |
| `writer.py:1441-1460, 2551-2580` | fix 7: flag on, a plain-protein transporter gets one synthesised one-member `protein-complex` (name `<protein> complex`, one `protein-complex-protein`, stoichiometry nil, shared across transports of the same protein); `transport-transporter` writes `protein-complex-id` (no `protein-id`); the transporter visualization uses a `protein-complex-visualization` pointing at the protein's existing location |
| `tests/test_final_payload_writer_fixes.py` (new) | 14 tests, flag on and flag off for each fix |
| importer (not in this repo) | `vev_node` typo and left/right state remap in `lib/pwml_parser.rb`, see section 4 |
| harness (`pwml/import-harness` in `C:/t/pwi/t2pw`) | vocabulary seed `0adf19bb`, docs `647c2fdb` |

Commits on `pwml/final-payload`: `03e4c241` (fixes 1, 2, 6), `76b0d6eb` (review fix: base location only where referenced, `is True`, tests), `2bf0511a` (fix 7), `e9c3e197` (test fix). No `pipeline/`, `mapping/`, `rag/`, `curation/`, `llm/prompts/` change; no existing test edited.

## 2. Tests (bounded wrapper, `final_surviving_count` 0 on every run, HEAD `e9c3e197`)

| suite | result |
|---|---|
| focused (writer fixes, final_payload fill and meta, `test_pwml_writer`, 7 `test_clean_model_*`) | 187 passed, 1 skipped |
| golden seam, 4 files, unedited | 38 passed |
| SMOKE | 511 passed |

Output: `evidence/final_payload_p2/tests_final/` (and `tests/` for the earlier commit: focused 180 + 1 skipped, golden 38, SMOKE 511).

## 3. Flag-off identity proof

152 runs replayed from stored `final_mapped.json` through `run_pwml_export` (scripts `evidence/final_payload_p2/scripts6/`):
- base `e26c8566` flag off against HEAD `e9c3e197` flag off: **97 of 97 written PWML sha256-identical**, same set writes (55 write none in both). Repeated for the intermediate commit `76b0d6eb` (97 of 97).
- Identity of code paths is also by construction: every new line is gated by `final_payload_enabled()`, except the stdlib-only import (reviewers confirmed).

## 4. PathWhiz importer bugs fixed

Patch: `evidence/final_payload_p2/pathwhiz_importer_fixes.patch` (two hunks, `lib/pwml_parser.rb`). Applied to the harness copy `C:/t/pwi/smpdb` and, by PO ruling D-W19, to the owner's working copy. The owner file was first backed up to `C:/t/pwi/backup/pwml_parser.rb.owner-2026-11-11`; the diff of backup against the new file is exactly the two hunks (verified by the orchestrator). Nothing was staged or committed in either PathWhiz repo. The full write-up is `evidence/final_payload_p2/importer_bugs_section.md`; summary:

1. **`vcv_node` typo, `lib/pwml_parser.rb:291`.** The loop variable is `vev_node`; the body uses `vcv_node` from an earlier loop. Symptom: vacuous edge visualizations dropped or the conversion stopping silently (exceptions are swallowed). Fix: `vev_node`. Read from code, not reproduced.
2. **Left/right biological state ids not remapped (`add_content`, final `else` near line 585).** `biological-state-id` is remapped through `@biological_state_list`; `left-biological-state-id` and `right-biological-state-id` fall to the generic branch and keep the raw XML id. Symptom in an empty database: all transport elements fail ("Left/Right biological state can't be blank"; 37 of 37 transports). **Production impact: the transport silently attaches to whatever biological state has that id; there is no error.** Fix: two `elsif` branches copied from the `biological-state-id` branch.

Effect measured: with the flag off, transports created 0 of 37 (old importer) to 30 of 37 (patched importer). Controls: real PathBank files PW687775 (3 errors, a transport direction invalid) and PW176888 (5 errors) now import with success true.

**A third importer gap was found, not patched:** `add_content` has no `transport-transporter-id` remap. 19 `transport_transporter_visualization: Transport transporter can't be blank` messages in 12 files remain with the flag on, and they also occur on the real PathBank file PW176888. Reaction enzymes avoid this through `store_reaction_enzyme_ids` (parser near :802). Patch outside D-W19 scope; listed under open questions.

## 5. Harness before and after

Population: triage baseline = the 60 old exports; sets off5/on5/off6/on6 = 59 replay outputs of the same runs. Seed = vocabulary extension (`seed_extension.md`). off5/on5 use the old importer, off6/on6 the patched importer. Raw: `evidence/final_payload_p2/import/`, `import6/`.

| | triage baseline (old seed) | baseline files, new seed | off5 | on5 | off6 | **on6** |
|---|---|---|---|---|---|---|
| files | 60 | 60 | 59 | 59 | 59 | 59 |
| clean (0 messages) | 0 | 0 | 0 | 28 | 0 | **29** |
| error messages | 1876 | 844 | 626 | 379 | 496 | **211** |
| ontology | 302 | 114 | 110 | 110 | 110 | 110 |
| complex/species | 700 | 2 | 2 | 2 | 2 | 2 |
| bound_visualization | 168 | 168 | 165 | 0 | 165 | 0 |
| spontaneous | 88 | 21 | 19 | 0 | 19 | 0 |
| compound_location | 42 | 42 | 45 | 1 | 45 | 1 |
| transport | 205 | 205 | 205 | 205 | 75 | **37** |
| interaction | 250 | 238 | 28 | 28 | 28 | 28 |
| other | 121 | 54 | 52 | 33 | 52 | 33 |
| transports created / in file | 0 / 37 | 0 / 37 | 0 / 37 | 0 / 37 | 30 / 37 | **31 / 37** |
| interactions created / in file | 5 / 85 | 7 / 85 | 9 / 17 | 9 / 17 | 9 / 17 | 9 / 17 |

Reading it:
- The seed alone removes about 1,030 messages (complex/species 700 to 2) but leaves 0 clean files.
- The flag removes bound-visualization, spontaneous and all but one compound-location message: 28 clean files.
- The patched importer lets transports in: 31 of 37.
- What remains with the flag on and the patched importer (211 messages, 30 files not clean):
  - 19 transport_transporter_visualization "Transport transporter can't be blank" (12 files): missing importer remap (section 4). The transporter itself now imports (the `Protein complex can't be blank` messages are gone with fix 7).
  - 6 transports in 5 files with left state equal to right state (writer or content gap, not traced).
  - 110 ontology messages in 27 files: location, cell-type and tissue names missing from PathWhiz vocabulary (needs the name map, out of scope).
  - 33 other: nucleic-acid type blank (31, content gap), one `Pwc has already been taken`, one `Compound can't be blank`.
  - 28 interaction messages in one file (nucleic-acid dependent).
- Writer-only (no importer patch) on5 against off5: 0 to 28 clean files, 626 to 379 messages.

Replay properties (152 runs, `evidence/final_payload_p2/replay6/an6.txt`): 97 write in all of base, off6, on6; no run gained or lost; `bound-visualization` elements 289 (off) to 0 (on); `spontaneous` false 409 (off) to nil 409 (on); dangling `compound-location-id` 80 of 1193 in 27 files (off) to 0 (on); transport-transporters with `protein-complex-id` 3 of 34 (off) to 34 of 34 (on), none with `protein-id` on; no dangling complex or protein refs; QA errors 0 in both, warnings 388 to 467 (the ChEBI-prefix warning, as in phase 1); all files parse as XML.

## 6. Review verdicts

| review | model / effort | scope | verdict |
|---|---|---|---|
| 1 | opus / high | `e26c8566..03e4c241` | APPROVE WITH FIXES: major (fix 6 exempted a compound in every state), minors (interactions share the defect, spontaneous `is True`, weak tests) |
| 2 | sonnet / high | `03e4c241..76b0d6eb` | APPROVE; all four items resolved. Minors: no flag-off control for the interaction test |
| 3 | opus / high | `76b0d6eb..2bf0511a` (fix 7) | APPROVE WITH FIXES: major 1 (identity test fails by design; fixed in `e9c3e197`, not independently re-reviewed); major 2 (importer has no `transport-transporter-id` remap, recorded in section 4, not patched) |

Minor findings of review 3 **not fixed**: duplicate complex when a protein is both a reaction enzyme and a transporter (PathWhiz `find_existing` merges them by name, species and signature, so benign unless a same-name complex with different members exists; fix: reuse the Stage 6 wrapper); nameless protein gives complex name `" complex"` (use `_wrapper_complex_name`, fall back to the entity key); a synthesised complex id may collide with an explicit `pathwhiz_id` of 40000 or more; the JSON says plain protein while the PWML says complex, which `canonical.biological_equivalence` would flag once wired in.

## 7. Open questions for the product owner

1. **Flip `T2PW_FINAL_PAYLOAD` on?** It now carries phase 1 (ids, title, reference) and phase 2 (three writer fixes and wrapped transporters). Harness: 29 of 59 clean against 0. Options: (A) keep off; (B) turn on after one live run per mode; (C) turn on now. Risk of C: the live seam was never run.
2. **Third importer patch:** add a `transport-transporter-id` remap (pattern of `store_reaction_enzyme_ids`, keyed by transport, complex and state). Without it 19 transporter visualizations fail and, in production, could attach to an arbitrary transporter. Cost: one hunk in the owner's file. Real PathBank PW176888 fails the same way, so this may exist in production for real files too.
3. **Where to wrap transporters:** exporter-side wrapper (done, flag-gated) or Stage 6 (consistent with enzymes, removes the JSON/PWML divergence, but outside this track). 
4. **Vocabulary map** for 40 location, cell-type and tissue names that do not exist in PathWhiz (110 ontology messages): map to existing rows, create new rows with GO/CL/BTO ids, or drop the location.
5. **Nucleic acids** with blank DNA/RNA type (31 messages): default per entity kind, or withhold with the interactions that need them.
6. **Spontaneous:** it is nil for all reactions, never `true`. Passing a real value needs a modelling change in `ir.py`. Later gap-fill step (PO): search other papers for the missing enzyme before falling back to unknown.
7. **Left equals right state transports** (6 in 5 files): untraced; needs a look at why the IR places a transport inside one state.
8. **Compartment drawing** is gone with bound visualizations (they were never valid PathWhiz); PathBank uses membrane visualizations. A separate job.

## 8. Dispatches (11 of a budget raised from 8 to 11, plus message resumes)

| # | task | model / effort |
|---|---|---|
| 1 | writer fixes 1, 2, 6 | general-purpose, sonnet / medium |
| 2 | harness seed extension | sonnet / medium |
| 3 | focused, SMOKE, golden tests | haiku |
| 4 | review of `03e4c241` | opus / high |
| 5 | review fixes | sonnet / medium |
| 6 | re-review `76b0d6eb` | sonnet / high |
| 7 | 152-run replay and first harness measurement | sonnet / medium |
| 8 | importer fixes, harness commits | sonnet / medium |
| 9 | fix 7 transporter wrapper | sonnet / high |
| 10 | review of fix 7 | opus / high |
| 11 | test fix, final tests, replay, harness import (two message resumes after a premature end of turn and the machine restart) | sonnet / medium |

Dead ends: the first harness pass was discarded because the wrapper timeout killed batches mid-chunk (redone in 10-file batches); three agents typed `python -` with empty stdin by mistake and stopped it; a machine restart delayed the last step.
