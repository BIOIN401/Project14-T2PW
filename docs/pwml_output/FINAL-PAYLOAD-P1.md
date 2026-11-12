# FINAL PAYLOAD, PHASE 1 (2026-11-10 / 11, branch `pwml/final-payload`, worktree `C:/t/fp`, from `9ae702ba`)

PO ruling D-W12: extraction -> ids for all entities -> completeness check, fixes, gap fill -> **final payload** -> every format. Phase 1 builds only the parts that need no LLM: a pre-freeze database id fill, and the paper title and reference in the PWML. Everything is behind one environment flag, `T2PW_FINAL_PAYLOAD` (`1`/`true`/`yes`), **default OFF**. Code HEAD: `f83bc613`. Nothing is pushed. Evidence: `docs/pwml_output/evidence/final_payload_p1/`.

## 0. NOT ESTABLISHED (read first)

1. **The live path has not run.** No pipeline run was allowed. The new seam in `run_post_pipeline_sbml_artifacts` (fill before the pre-freeze resolver, restore on resolver refusal) and the hand-off of paper ids in `service/pathway.py` are covered by unit tests and by code review only. The replay applies the fill to the stored post-quarantine `final_mapped.json` and then calls `run_pwml_export` (whose own pre-freeze pass renames compounds), so it exercises the export half. The ordering and the restore are not exercised end to end.
2. **Flag-off identity is proved for the export half only**: 97 of 97 written PWML are byte-identical to the base code `9ae702ba`; the other 55 runs write none in both. Identity of the pre-export half (`postpipeline` seam, `pathway.py`) rests on code reading (opus review, re-review) and the unedited golden-seam tests.
3. **The 40 "no PWML" runs.** With the flag off, 37 of 40 now write a PWML under replay conditions. Replay conditions are not live conditions: no PathBank database (`db_not_configured`), the stored quarantine report cannot be reused (older policy version), so the boundary re-runs on the stored payload, and `pathway_context` is rebuilt from the old report's core terms. The number says "fixes since those runs, under a DB-less replay", not "these runs would now succeed live".
4. **The flag does not change which runs write a PWML** (0 of 152 change state). Phase 1 gains are in ids, title and reference, not in conversion rate. No protein id is written (section 5), and protein identity was the biggest single reason for the old failures (119 protein-identity gate errors in the 40 runs).
5. **Hand check** (31 ids, independent agent): 2 wrong, 29 correct. That is above the 1-in-20 limit, so the matching was tightened (FP-3, FP-4) and **the check was not repeated on the tightened output**. The two wrong ids and one more found outside the sample are now refused; the residual error rate of the remaining 547 ids is unmeasured. The checker also noted that the `chebi_id` of those two entities is wrong, since their cross-reference fills inherit it (section 3).
6. **PathWhiz import** was tested on an empty database (harness limit). Production may differ. `success: true` carries little meaning (harness record section 1). The pathway name was not read back from the database. Duplicate-name rejection (the PO's reason for titles) was **not** tested: the harness truncates the DB before each file.
7. **Gene names** were not added to the PWML (the PO said only if the import test shows PathWhiz accepts them; not tested).
8. The import-harness builder left its files staged but uncommitted in `C:/t/pwi/t2pw` (no process running, record complete). The harness was used read-only; `git status` there was identical before and after.
9. Dispatch budget exceeded (section 8).

## 1. What changed and where

All code is in 7 files (`git diff --stat 9ae702ba..f83bc613`): 1 new module, 4 edited source files, 2 new test files. No test edited, no `llm/prompts`, no `pipeline/`, `mapping/`, `rag/`, `curation/` change.

| file | change | why |
|---|---|---|
| `src/t2pw/service/final_payload.py` (new, 350 lines) | `final_payload_enabled`; `apply_final_id_fill` (uses `clean_model.builder` + `clean_model.fill`, writes accepted ids into payload rows); `_plan` (the rules); `apply_final_pathway_meta` (title/ids via NCBI, fail-open); `prefreeze_with_final_fill` (seam helper); `effective_pathway_name`; `paper_meta`, `paper_meta_from_paper`; `fill_meta` | the deterministic id fill and title/reference of D-W12 |
| `src/t2pw/service/postpipeline.py` | `run_post_pipeline_sbml_artifacts`: the bare `run_prefreeze_resolution(...)` call became `prefreeze_with_final_fill(payload, lambda p: run_prefreeze_resolution(p, strict_db=...), pathway_context, state)`; with the flag off the helper calls the resolver once with the same arguments and returns `{}`. `run_pwml_export`: `pathway_name = effective_pathway_name(final_payload, pathway_name)` | fill must run before the resolver and the freeze (review F1, D-015); title replaces the default name only when the payload carries the key |
| `src/t2pw/service/pathway.py` | 3 lines: with the flag on, `state["final_payload_paper"] = paper_meta_from_paper(paper)` | hand the paper's PMID/PMCID/DOI to the post-pipeline |
| `src/t2pw/pwml/ir.py` `build_pwml_ir` | only when the payload has `final_payload_pathway`: title becomes the pathway name if the caller's name is still `Generated Pathway`; `ir["pathway"]["references"]` gets one entry | the IR is the writer's input |
| `src/t2pw/pwml/writer.py` `_emit_pathway` | `references` is built from the IR (id, pubmed-id, text, element-id, element-type `Pathway`, the template's shape); empty list when absent | PWML reference element; additive |
| `tests/test_final_payload_fill.py`, `tests/test_final_payload_meta.py` (new) | 42 tests with a fake fetcher, no network | |

### The flag and what it does
- Off (default): the seam calls the resolver exactly once as before; no new key is created; no state is written; `effective_pathway_name` returns the name unchanged; `ir.py`/`writer.py` branches need the key.
- On: before the pre-freeze resolver, `apply_final_id_fill` fills **empty** compound id slots (`chebi_id kegg_id hmdb_id pubchem_cid cas drugbank_id`) and species `taxonomy_id`, then `apply_final_pathway_meta` adds the title/reference; the resolver (D-015) then runs on the filled payload, so the canonical name and the freeze already see the ids. If the resolver refuses (`PrefreezeResolutionError`) after a fill changed the payload, the payload is restored in place and the resolver re-run exactly as the flag-off path would.

### Rules enforced in `_plan`
Never overwrites an id (row key, `db_row`, `mapped_ids`, placeholders count as empty); exact label or exact synonym only; one distinct match; an id already used by another row is refused; complexes, families/classes, "Unknown" skipped; proteins are **not written at all** (the fill layer only suggests UniProt candidates, D-106/D-108/D-117/D-035); reaction EC/Rhea inference stays off. Refusals added by the measurement:
- **FP-3 `short_abbreviation_synonym`**: a `chebi_id` match of type exactly `exact_synonym` for a name of 2 to 5 capital letters (MK, PMEA, CDG, DMAPP, SHCHC) refuses the whole entity, cross-reference ids included. H2O, H+, O2, NAD+, Ca2+, MgCl2 are unaffected.
- **FP-4 `no_structure_in_chebi`**: a name-matched `chebi_id` is refused (with its derived ids) when ChEBI returned no InChIKey; this blocks role/class terms (e.g. "fatty acid") at the cost of the rare real molecule ChEBI holds without a structure.

### Provenance in the payload
Top-level `final_payload_fill`: `{schema, date, written, by_field, entries:[{kind, entity, field, accession, source, match_type, retrieved}]}` (present only when something was written). Top-level `final_payload_pathway`: `schema, date, title, pmid, pmcid, doi, year, authors` each `{value, source, match_type, retrieved}`, plus `reference:{pubmed_id, text}` (only when NCBI confirmed the paper). Both keys carry dates, so the canonical payload hash changes from day to day with the flag on.

## 2. Measurement: replay, no LLM

Method: the 152 canonical runs (`C:/t/ev-cm/runlist.json`, `q == true`; groups from `evidence/01/tidy_runs.csv`: 61 P1 pathwhiz with PWML, 40 P0 pathwhiz without, 51 RS research) were replayed through `service.postpipeline.run_pwml_export` from each run's stored `final_mapped.json`, in a scratch project root per run, default name/description/subject/size as the live driver, `export_mode` per run. Labels: `base0` (base code `9ae702ba`, flag off), `off4` (HEAD, flag off), `on4` (HEAD, flag on; the fill applied to a deep copy first). Scripts in `evidence/final_payload_p1/scripts/`.

### Flag off
| group | runs | PWML written, base code | HEAD flag off |
|---|---|---|---|
| P1 (wrote before) | 61 | 60 | 60 |
| **P0 (the 40 that wrote none)** | **40** | **37** | **37** |
| RS (research, none by design) | 51 | 0 | 0 |

- **37 of the 40 previously failed runs now write a PWML** with the flag off (effect of fixes since those runs; caveats in 0.3). The 3 that do not stop at the PWML required-field gate. One P1 run also fails today at the required-field gate; RS runs stop at the pre-export Stage 3 revalidation (35) or the required-field gate (16).
- HEAD flag off versus base code: **97 of 97 written PWML sha256-identical**; no run changes who writes. Replay is deterministic (a run repeated gave the same sha256).

### Flag on (`on4`) versus flag off
| measure | value |
|---|---|
| runs that write a PWML | 97, same set as flag off; no run gained or lost (0 regressions) |
| ids gained | **547 in 127 of 152 runs**, all compounds, all from ChEBI, 0 protein, 0 species |
| by field | kegg_id 149, drugbank_id 138, cas 127, chebi_id 106, hmdb_id 27 |
| by match type | cross_reference 441, exact_label 52, exact_synonym 32, exact_synonym+charged_form 22 |
| refused | 53: short_abbreviation_synonym 36, no_structure_in_chebi 15, ambiguous 1, id used by another row 1 |
| compounds carrying a database id (97 written PWML, 687 compounds) | 406 off, **484 on** (48 PWML gain ids, none lose) |
| entities newly exportable to PWML | 0 (no run starts writing); the gain is ids on already exported compounds |
| PWML with a real title | **97 of 97** (off: 0); all titles from NCBI |
| PWML with a PMID reference | 97 of 97 |
| PWML files that change | 97 of 97 (every file gets a title and reference) |
| live requests | 709 (first pass with an empty cache), 0 fetch failures; later passes 0 requests |
| well-formed XML / PWML QA errors | 97 of 97 parse; 0 QA errors off and on; QA warnings 388 -> 467 (all the existing "ChEBI ID missing the CHEBI: prefix" warning, same convention the writer already uses) |
| compound renames | 22 compounds in 97 files, 4 distinct pairs, all the same species (pyruvate -> Pyruvic acid, H+ -> Hydrogen Ion, H2O -> Water, pyridoxal 5-phosphate -> Pyridoxal 5'-phosphate) |

The renames are made by the existing D-015 resolver (`pwml/prefreeze_resolution`, matches by `chebi_id`/`kegg_id`) once a compound has an id; the old name is kept as `<short-name>` in the PWML (e.g. `H+`). With the F1 fix this now happens before the freeze, so `final_mapped.json` and the SBML carry the same canonical name as the PWML.

Diff sample (5 files: idx 3, 27, 67, 103, 129): `evidence/final_payload_p1/replay/diff_sample.md`. Full id list: `ids_gained.jsonl` (547 lines, the `on4` pass; the hand check sampled the earlier 596-id pass).

### Example (idx 3, `sour_taste-strict`, PMC7059786)
```
- <name>Generated Pathway</name>
+ <name>Recent advances in taste transduction and signaling.</name>
- <references/>
+ <references><reference><id>1</id><pubmed-id>32185015</pubmed-id><text>Kinnamon SC, Finger TE: Recent advances ... PMID: 32185015.</text>
  <element-id>1</element-id><element-type>Pathway</element-type></reference></references>
  H+  ->  <name>Hydrogen Ion</name> ... <short-name>H+</short-name>, chebi-id 15378, kegg-id C00080
```

## 3. Hand check (independent sonnet/medium agent that wrote none of the code)

`evidence/final_payload_p1/hand/hand_check.md`. It checked the on1 pass (596 ids, before tightening), seed 20261111, against ChEBI (no KEGG call, D-W14):
- 31 ids: **29 correct, 2 wrong, 0 unclear.** The 20 `chebi_id` fills all matched by name (the checker later saw that MK's and PMEA's `chebi_id` are wrong by role too, below); 11 cross-reference fills: 9 correct, 2 wrong.
- The 2 wrong ones, both right by name and wrong by role: **MK** (menaquinone; ChEBI synonym of the dipeptide Met-Lys), **PMEA** (a lipid in the run; ChEBI synonym of adefovir). A third, **CDG** (a 7-deazaguanine intermediate; ChEBI synonym of the tripeptide Cys-Asp-Gly), was found outside the sample.
- Charged-form subset: 5 of 5 correct. exact_label: none wrong.
- 6 titles/PMIDs checked against Europe PMC: all 6 match on PMID, title, PMCID, DOI.
- More than 1 in 20 wrong, so the matching was tightened (FP-3, FP-4). Not re-checked afterwards (0.5).

## 4. PathWhiz import (harness `pwml/import-harness`, local PathWhiz/SMPDB, empty DB)

24 pairs (12 P1, 12 P0; idx in `evidence/final_payload_p1/import/`), off versus on, same files otherwise:
| | flag off | flag on |
|---|---|---|
| harness `success: true` | 24/24 | 24/24 |
| files with zero error messages | 0/24 | 0/24 |
| error messages total (mean / median) | 536 (22.3 / 8.5) | 536 (22.3 / 8.5) |
| `Reference` rows created | 0 | 24 |
| compounds created / expected | 134 / 135 | 134 / 135 |

Per pair the set of error messages is identical; none is new with the flag on. The messages are the known empty-database artefacts (e.g. "Ontology can't be blank"). **Import pass rate did not drop**; the importer accepts the references and the new names. Limits: section 0.6.

## 5. Why no protein ids
The clean model's fill layer proposes UniProt candidates for id-less proteins only as `inferred` suggestions (P-20, D-106/D-108/D-117). `_plan` writes ids only to compound and species slots, so the 119 protein-identity gate errors seen in the old failing runs are untouched by this flag. Species-matched exact-name protein fill is allowed by the rulings but was not built; see open question 2.

## 6. Tests (dispatched haiku/low runner, bounded wrapper, `final_surviving_count` 0, HEAD `f83bc613`, clean tree)
- SMOKE: **511 passed** (65 s).
- Golden seam, unedited (`test_batch_driver_seam_golden`, `test_c011_freeze_seam_golden_equivalence`, `test_streamlit_stage8_export_contract`, `test_streamlit_quarantine_boundary`): **38 passed**.
- Focused (`test_final_payload_fill`, `test_final_payload_meta`, 7 `test_clean_model_*`): **143 passed, 1 skipped**. Also earlier at FP-2/FP-4: PWML writer/IR files 141 passed.
- Outputs: `evidence/final_payload_p1/tests/`.

## 7. Review
- Opus/high (`ws-reviewer`) on `9ae702ba..aafb7f17`: **APPROVE WITH FIXES.** Boundary clean, flag-off identity by construction confirmed. F1 major (fill ran after the resolver, so the PWML and the frozen payload disagreed on names), F2 major (web `web:PMID...` ids got no reference), F3 (unconfirmed PMID in the reference), F4 (role/class terms), F5 (letters+digits abbreviations such as G3P, PIP2, IP3, S1P still fill on `exact_synonym`; **not fixed**), F6 to F8 notes (documented).
- Fix round `f83bc613`; re-review (sonnet/high, delta `aafb7f17..f83bc613`): **APPROVE.** Notes: a pdf named like a DOI export (`10.1038_s41586...`) is parsed as a DOI in web ids (never confirmed by NCBI, so no reference, but the bogus doi sits in `final_payload_pathway`); the fill report is stale after a restore (unused); no test runs the real `run_prefreeze_resolution` on a filled payload.

## 8. Dispatches and dead ends
About 15 dispatches against the budget of 12. Five were lost to environment stalls, not to scope: writes under `C:/t` were denied for the first implementer (setting `blockWritesOutsideWorkingDirectories`), a second agent correctly refused to write on another agent's say-so, a stale heavy lock plus a machine shutdown cost about a day, and the heavy-lock/RAM gate stalled several test runs for an hour each (Docker holds RAM; floor lowered to 4 GB for focused work).
| step | model / effort |
|---|---|
| FP-1 draft; resume; finish; tests (4 dispatches) | sonnet high; sonnet high; sonnet medium; sonnet medium |
| replay harness + base0 | sonnet medium |
| FP-2 title/references | sonnet high |
| flag-on replay and measurement (2 dispatches; the first was left running slowly and finished later) | sonnet medium |
| hand check | sonnet medium |
| FP-3 abbreviation rule (2 messages) | sonnet high |
| review | opus high |
| FP-4 review fixes | sonnet high |
| final tests | haiku low |
| re-review | sonnet high |
| PathWhiz import pairs | sonnet medium |
Other dead ends: one implementer ran an empty `python -` heredoc (exited at once); a stale agent deleted replay work dirs mid-run (one run, idx 128, failed spuriously, re-run and replaced); the first flag-on pass's per-run results were not usable (live requests were counted correctly: 709).

## 9. Open questions for the PO

1. **Flip `T2PW_FINAL_PAYLOAD` on?** Measured on 97 written PWML: +78 compounds with database ids (406 -> 484 of 687), 100% titles and references, imports unchanged (0 new messages), no run gained or lost. Not exercised: the live seam. Options and costs: (A) keep off, merge only (cost: none, no gain); (B) turn on after one live pipeline run per mode (paper, name) checks the seam and the title in the web job (cost: one LLM run each, about the usual 20 to 60 min); (C) turn on now (risk: the seam's order and the paper-id hand-off were never run live; flag off is the escape).
2. **Protein ids.** Fill UniProt ids for id-less proteins by exact name plus species match? Allowed by D-106/D-108 but not built; it is the lever that could change conversion (protein identity was the top gate failure). Cost: a new rule set, a new hand-check, one task; risk: wrong-protein ids are the worst error class.
3. **Renames.** Accept pyruvate -> "Pyruvic acid", H+ -> "Hydrogen Ion", H2O -> "Water" (database names, the original kept as short-name)? It is the existing D-015 behaviour reacting to a new id. Alternative: fill ids only where the canonical name equals the paper name (cost: fewer ids).
4. **KEGG ids from ChEBI cross-references** (149 `kegg_id` filled). Ids only, no KEGG service call (D-W14); say if even that is unwanted (drop `kegg_id` from the targets).
5. **Residual precision.** The hand check was not repeated after the tightening. Repeat on a fresh 30 (cost: one sonnet/medium dispatch), and decide whether to also refuse letter+digit abbreviations (G3P, PIP2, IP3, S1P; review F5).
6. **Duplicate-name rejection** on PathWhiz import is not demonstrated; test on a database that already holds pathways of the same name before claiming the fix.
7. **Gene names** in PWML: not tested; needs an import test.
8. **Name mode** gets the Stage-0 pathway name as title (LLM-derived) and no reference; the ruling asked for exactly that, but the reference for a retrieved paper is not set up in name mode.
9. **Canonical hash** changes daily with the flag on (dates inside `final_payload_*`). Strip dates from the hashed view?
10. Dispatch budget (12) was exceeded by about 3 for the reasons in section 8.
