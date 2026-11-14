# FINAL PAYLOAD, PHASE 4 (2026-11-13, branch `pwml/final-payload`, from `f66b3ef2`, code HEAD `0727df29`)

Evidence: `evidence/final_payload_p4/` (a_ontology_lookup.md, b_before_after.md, c_nucleic_paper.md, d_measure.md, d2/, d2_run10_trace.md, d3/d3_measure.md, tests_final/). Proposed decisions P-41 to P-43 in `DECISIONS-PROPOSED.md`.

## 0. NOT ESTABLISHED (read first)
1. **No live run was made and the default was NOT flipped (steps E, F).** `C:/t/fp/.env` does not exist (checked with `test -f`, 2026-11-13). The live seam with the flag on, `final_payload_fill` provenance in a live PWML, title/PMID reference, and the new paper-text hook (`state["final_payload_paper_text"]`, pathway.py) have never run live. The Streamlit path does not set the paper text (nucleic types stay unset there). Driver and commands: `evidence/final_payload_p3/live/`.
2. **Residual importer-wildcard risk (re-review, major, not fixed):** the PathWhiz importer treats an omitted state component as a wildcard (`find_by` without nil ids). The writer's collision guard (`writer.py` ~1120-1125) compares equality only. A state that loses its id-less location but keeps a tissue or cell type can still bind to a sibling state with the same species and tissue plus a location. Not seen in the 152-run set (0 species-only states in d3); no test covers it. Possible fix: keep an id-less location whenever the post-drop non-null components are a subset of another state's.
3. Harness = empty DB plus vocabulary seed; nothing imported into production PathWhiz. "Clean" = zero messages.
4. Biological correctness of the 13 resolved ontology terms and of the changed files was not reviewed per entity (exact name match only, no organism check: `root` gives the plant root BTO term). Unresolved: 8 cell types, 1 tissue, 7 locations (of the 27, plus others).
5. Task B: 4 equal-side transports remain in the 243 stored payloads (no stored information to fix them from; one is explicit model output). Measured on stored `merged_payload.json` (the stage before the normalizer), not in a live run. The reviewer noted the switch to `_normalize` widens when a declared state is used; the before/after was re-run after it and is unchanged (2 transports).
6. Nucleic types: 10 of 35 rows resolve; a real sRNA (`RyhB`) is lost by the conservative intersection rule. The effect on the 33 harness messages was measured only as far as the d3 harness (24 nucleic messages remain, of which 8 type messages).
7. idx 125 is no longer clean in d3 (keeps its id-less location again, as in P3, so the empty-DB importer says "Ontology can't be blank"): the price of never producing species-only states.
8. SMOKE first run at HEAD had 3 failures, all exit 3221225794 (0xC0000142), while another job ran; the quiet rerun passed 511. Pre-existing failure unrelated to this work: `tests/test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform` (expects the call in `streamlit_app.py`; at `f66b3ef2` it is not there, and `src/t2pw/app` is untouched).
9. Process slips by subagents: several stray `python - <<EOF` (no effect) and one `&` plus background that stranded the heavy lock; the orchestrator cleared it after confirming the holder PID was dead and then ran the final tests itself.

## 1. Changes
- **A (D-W20a).** New `src/t2pw/service/final_payload_ontology.py`: OLS4 lookup (GO cellular component, CL, BTO; PathWhiz validates BTO, so no UBERON). Exact label, or exact synonym only when the term's own `hasExactSynonym` contains the name (v2 entity fetch); ambiguous is refused; lookups stop after the first fetch error; existing ids never overwritten. Called from `final_payload.py` ~166-172; provenance `db` (source OLS4, accession, label, match type, date) in `final_payload_fill`; key `ontology_id` on the row. `vocabulary_map.py`: `mapped_by_id` (id already a PathWhiz row, so the row's name is used). `writer.py` ~1056-1135: an id-less location (outcome `unmapped`, no id of any kind) is dropped from its states, except when dropping makes a state identical to another or leaves a state with no tissue, cell type or location (species alone is a wildcard in the importer; commit `0727df29`). Counters `locations_dropped_no_id`, `locations_kept_state_collision`. Other formats keep the name. States referencing a dropped location are written without `subcellular-location-id` (PathWhiz needs one component only).
- **B (D-W20b).** `process_normalizer.py` `_declared_side_state` (~3143-3160), used at ~3228-3240: a missing transport side takes the state declared for that side in `elements_with_states` when all declarations of that side agree, the state exists in `biological_states`, and it differs (case-insensitively) from the other side's result; otherwise `__auto_state__` as before. Not flag-gated; no prose inference; nothing dropped.
- **C (D-W20c).** New `final_payload_nucleic.py`; hook `final_payload.py` ~225; paper text from `<run dir>/../01_source_text.txt` (replay) or `state["final_payload_paper_text"]` set in `pathway.py` ~261 (flag on only); `writer.py` 729 reads `nucleotide_type_paper` (flag-gated); `ir.py` 882 whitelists the key (inert without it). Rules in P-42.

## 2. B before/after (243 stored merged payloads, `evidence/final_payload_p4/b_before_after.md`)
243 compared, 0 errors; **2 transports changed**; 0 differences outside transport from/to states; 8 gate counters changed in the OTOP1 run (nodes 49 to 50, edges 70 to 73: one more location edge).
- OTOP1-mediated H+ permeation (PMC7059786): `__auto_state__` / `__auto_state__` became `__auto_state__` / `Type III taste cell state`.
- enterobactin secretion (runs_verify/2026-08-18_1328, PMC12452463): `__auto_state__` / `Escherichia coli extracellular` became `Escherichia coli cytosol` / `Escherichia coli extracellular` (it was not degenerate; its from side improved).
Equal-side transports: 5 before, 4 after (TubI tubercidin; ferric-enterobactin import; enterobactin secretion 2026-08-27 and 2026-09-02, the latter explicit model output with a correct twin).

## 3. C: nucleic acid type from the paper
152 runs: 39 nucleic-acid rows, 4 settled by name, 35 unsettled; **10 resolved from the paper (all DNA)**, 6 conflicting, 11 mentioned without a cue, 7 never mentioned, 1 protein-like. Evidence sample (`c_nucleic_paper.md`): "...directly binding to the PORA promoter (Xu et al., 2015)." gives DNA (cue `promoter`); "...photosynthesis-associated nuclear genes (PhANGs)..." gives DNA (`genes`). The first version (window rule only) resolved 16; review judged LHCB1.1 (RNA) doubtful and the whole-sentence agreement requirement removed it and araO2, araI2, CHLH promoter and RyhB (a correct RNA). Fallback for the rest is the product owner's choice (section 9).

## 4. Tests (bounded wrapper, 0 surviving processes, HEAD `0727df29`)
SMOKE **511 passed**; golden seam **38 passed** (unedited); focused (`test_final_payload_*`, `test_pwml_writer*`, `test_pwml_ir*`, `test_process_normalizer*`, `test_c121_autostate_lifecycle`) **231 passed**. Reports in `tests_final/`. Existing tests edited (flag-on behaviour mandated by the rulings; the sonnet re-review judged each legitimate): `test_final_payload_vocabulary.py` (thylakoid row has no `pathwhiz_id`, so it is droppable), `test_final_payload_ontology.py` (drop tests), `test_final_payload_nucleic_type.py` (whole-sentence test replaced by the intersection). No test needed `T2PW_FINAL_PAYLOAD=0` because the default is unchanged. Pre-existing failure: section 0.8.

## 5. Flag-off identity
152-run replay at the fixed HEAD: **97 of 97 PWML sha256-identical** to the P3 stored flag-off outputs; 55 write none (d3). Replay starts from stored `final_mapped.json`, so B's normalizer change cannot show there (section 2 covers it).

## 6. Harness (59 flag-on files, three-patch importer, same category definitions in every column)
| | P3 | P4 first (cdcac27a) | d2 (02eff361) | d3 (HEAD) |
|---|---:|---:|---:|---:|
| clean files | 43 | 46 | 46 | 45 |
| messages | 147 | 81 | 95 | 94 |
| vocabulary | 62 | 24 | 32 | 38 |
| transport family | 21 | 28 | 25 | 18 |
| left=right | 6 | 8 | 7 | 5 |
| nucleic acid | 38 | 18 | 24 | 24 |
| interactions | 21 | 6 | 9 | 9 |
| other | 5 | 5 | 5 | 5 |
| transports created of 37 | 31 | 29 | 30 | 32 |

Controls unchanged (PW741182 clean, PW012926 1, PW176888 2, PW687775 3, PW1 stub 176). Run 10: P3 6 of 7 transports; first P4 3 of 7; d2 4 of 7 (the dropped `saliva` left a species-only state that the importer bound to another state, see `d2_run10_trace.md`); d3 6 of 7 with left=right 1. Flag-on files differing from P3: 9 of 97. Species-only states 0. Overall: 45 of 59 clean and 94 messages against 43 and 147 in P3.

## 7. Live runs, default flip
Not run (section 0.1). Default stays off (P-40 stands). When `.env` is present: run `evidence/final_payload_p3/live/run1.sh` for `pathwhiz`, `research`, `nameonly`, one at a time, outer timeout at least 2400 s; then flip (unset means on, `T2PW_FINAL_PAYLOAD=0` off) and rerun SMOKE, golden and focused, running legacy-pinning suites with the flag at 0.

## 8. Reviews
| review | model / effort | scope | verdict |
|---|---|---|---|
| 6 | opus / high | `f66b3ef2..cdcac27a` | REJECT: location drop merges compartments (M1), related synonyms accepted (M2), B can create left=right (M3), minors. All fixed in `02eff361` |
| 7 | sonnet / high | `cdcac27a..0727df29` | APPROVE WITH FIXES: all demands met; residual wildcard case (0.2) open; minors not fixed: vacuous `<= 1` assertion in the distinct-states test, counter name `locations_kept_state_collision` also counts no-component keeps, hyphen boundary loses some recall |

## 9. Open questions for the product owner
1. Put `.env` in `C:/t/fp` so the three live runs can be done, then flip the default; or flip on measurement alone (risk: the live seam with the flag is unproven).
2. Residual wildcard rule (0.2): add the subset guard (costs a few more id-less locations kept by name, i.e. vocabulary messages) or accept and document.
3. Cell types and tissues still unresolved after lookup (22 cell-type and 2 tissue import messages): create rows with owner-supplied ids, or drop those components too (same wildcard considerations).
4. The 4 remaining equal-side transports: withhold with a side-file line, or a Stage-1 extraction fix (biology change needing a D-xxx).
5. Nucleic acids still unresolved: default per kind, withhold with their interactions, or relax to the window-only rule (16 resolved, includes doubtful LHCB1.1).
6. Unmapped `inner membrane (IM)` / `outer membrane (OM)` could map to PathWhiz `Inner Membrane` via a reviewed synonym if the entity is judged identical.

## 10. Dispatches (11 of 12)
| # | task | model / effort |
|---|---|---|
| 1 | A ontology | sonnet / medium |
| 2 | B transports | sonnet / default (high-care brief) |
| 3 | C nucleic | sonnet / medium |
| 4 | tests on first HEAD | haiku |
| 5 | D replay and harness | sonnet / medium |
| 6 | review | opus / high |
| 7 | fix round | sonnet / high |
| 8 | run 10 trace | sonnet / medium |
| 9 | species-only fix and re-measure | sonnet / medium |
| 10 | re-review | sonnet / high |
| 11 | final tests (plus 1 resume; orchestrator reran after a stranded lock) | haiku |
