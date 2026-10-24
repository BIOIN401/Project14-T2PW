# ORCH-766 — unreferenced species rows are removed before the freeze (D-111 § 1)

Lane `orch766/drop-unreferenced-species`, base `d6849c9e`, worktree `.claude/worktrees/orch766`. Implementer
report; nothing merged, pushed or approved here. Reports `evidence/g11/ORCH-766/01`–`36`, logs and scripts
`evidence/orch766/`. Every report `FINAL SURVIVING COUNT : 0`. No LLM call, no pipeline run, no `--heavy-lock`.

## 1. Outcome

`strict_quarantine.quarantine_and_close` now closes the `species` bucket. A species row that nothing in the
pathway names is removed in the closure loop, after every other prune of that round, and recorded exactly like a
degree-zero removal. The required-field gate is untouched.

On the ORCH-759p menaquinone leg's own frozen payload, the base code reproduces the leg's refusal
(`species_missing_taxonomy` and `species_missing_classification` on `/entities/species/2`, "L. lactis subsp. lactis
KF147"). The tip removes that row, records it, and the payload then passes the final pre-export Stage-3 gate, the
required-field gate and the IR build/validation (`evidence/orch766/14-replay-base.log`, `13-replay-tip.log`). The
ORCH-759 (1148) leg behaves the same way. Nothing else in either payload moves. The ORCH-759p leg stays
`release_ready`, the 1148 leg stays `review_required`.

| commit | what |
|---|---|
| `0f54f3b4` | tests first: `tests/test_orch766_unreferenced_species.py`, `tests/fixtures/orch766/` (leg `final_mapped.json` verbatim, gzipped; its requested context; README) |
| `076675d8` | tests: the 684738 preservation test carried a tip-only assertion (§ 8.1) |
| `78158f07` | tests: kind `species_id_as_float_text` (a `"77.0"` PathBank id still references its row) |
| `f775f2b3` | code: `src/t2pw/pipeline/strict_quarantine.py` only (+162/−5) |
| `bcad7547` | Chunk E pin move, merge rule 4: `tests/test_strict_quarantine_real_artifact_replay.py` `RESIDUAL_CODES_BY_ROW` 27 → 19 and `docs/change_log.md` (same commit, as that module requires) |
| docs commit | this file, `evidence/orch766/`, `evidence/g11/ORCH-766/` |

Not touched: `streamlit_app.py`, `ir.py`, `writer.py`, the required-field gate, C-120/C-124 alias expansion
(`map_ids.py`), any prompt, `paper_lexicon.py`, `stage_one_ensemble.py`, `TEST_MATRIX.md`, `LEDGER.md`, `DECISIONS.md`.

## 2. What "referenced" means: every reader, and how the rule sees it

I traced every place that resolves a species row or reads a species field on the export path: the IR, the writer,
the required-field gate, the quarantine's own restoration step and the stage contracts. Line numbers are at
`f775f2b3`, where `ir.py` and `writer.py` are unchanged from base.

| # | reference kind | who reads it, and how it resolves | covered by | test kind(s) |
|---|---|---|---|---|
| 1 | a biological state's `species` | `ir.py:1742` `resolve_component` → `_norm` against species `name` + `aliases` (`:1453`) | any scalar | `state_species`, `alias`, `case_and_punctuation_variant` |
| 2 | a state's `organism` | `ir.py:370` (hydration), gate non-IR branch `:2954` | any scalar | `state_organism` |
| 3 | a state's **name** when it carries no species | `ir.py:1754` ("If the state is named after one component") | any scalar | `state_name` |
| 4 | a state's `taxonomy_id` / `species_id` | gate `:2954` (presence) | scalar under a species/taxon path | `state_taxonomy_id` |
| 5 | a protein's `species`, `organism`, `species_ref.{name,key,…}`, `mapping_meta.species` | writer `protein_species_id` `:1188-1214`: `strip().casefold()` against IR species **names**, else ids | any scalar | `protein_species`, `protein_organism`, `protein_species_ref_name`, `protein_mapping_meta_species`, `synonym`, `raw_name` |
| 6 | a protein's `taxonomy_id`, `species_id`, `pathbank_species_id`, `species_ref.{taxonomy_id,pathbank_species_id,species_id}`, `mapping_meta.species_id` | the same function, against the IR record's `pathwhiz_id` / `taxonomy_id` (`writer.py:998-1001`) | scalar under a species/taxon path | `protein_taxonomy_id`, `protein_species_id`, `protein_species_ref_ids`, `species_id_as_float_text` |
| 7 | a complex's or any other entity's `species` / `organism` | `protein_species_context` (presence); semantic `_organism_of` | any scalar | `complex_species`, `complex_organism`, `other_entity_species` |
| 8 | a location row, or any other top-level field (`metadata.organism`, …) | none on the export path today; kept so that "any other field" holds | any scalar | `location_row_species`, `pathway_metadata_organism` |
| 9 | a field on a **surviving** process | none today; same reason | any scalar of surviving processes only | `surviving_process_field` |
| 10 | the requested organism (`pathway_context.likely_organism`) | semantic organism check, `evaluate_production_semantics` | any scalar of the context | `requested_organism` |
| 11 | **the pathway's own species, every complex's species, and every protein the writer cannot resolve** | writer: the FIRST IR species row (`writer.py:1003`, `1310`, `2708`, fallback `1214`) | the first named row is never a candidate | `first_row` |
| 12 | the IR's create-default rename (`SPECIES_CREATE_DEFAULTS`) | `ir.py:449`: the row is renamed after the freeze and resolves under both names | the row's keys include the defaulted name, aliases and ids | `create_default_name` |
| 13 | the row's ids as the IR projects them | `ir._component_record` `:879`, `_db_id` (row, `mapping_meta`, `mapped_ids`, `candidates[0]`), `_to_int` (`"77.0"` → 77) | id holders and an integral-number key | `species_id_as_float_text` |
| 14 | the auto-state's species | `restore_autostates_if_required` / `_select_default_species_name` (`process_normalizer.py:593`) picks from rows present AFTER closure | runs after the loop, so it only names a kept row | (fixed by ordering) |
| 15 | "at least one species row" | `stage_contracts.py:130` `species_required` | the first named row is never removed | `first_row` |

**The rule** (`_prune_unreferenced_species`, `strict_quarantine.py:2145`). The rule is a superset of the table
above, built so that a missed field can only keep a row and never remove one:

- **Name references.** A row is kept if any of its names equals any scalar anywhere in the payload. Its names are
  `name`, `display_name`, `raw_name`, `aliases` and `synonyms`, plus the create-default rename. The scalars come from
  every entity bucket except `species` itself, every state, every location, every other top-level key, every
  **surviving** process, and the requested context.
- **Id references.** A row is also kept if any of its ids equals a scalar under a key path containing `species`,
  `taxon` or `organism`. Its ids are the taxonomy and PathBank ids, read from the row, `mapping_meta`, `mapped_ids`
  and `candidates[0]`.
- **Comparison keys.** Matching uses casefolded text and alphanumerics-only text, so any two values equal under the
  IR's `_norm` or the writer's `strip().casefold()` also match here. An integral number additionally matches its
  integer form.
- **Never candidates.** The first named row, and any row without a string name.

The two restrictions are deliberate:

1. **Quarantined processes do not count.** They are still physically present during the loop, but they are dropped
   before the freeze.
2. **Ids match only under species/taxon/organism paths.** The first census found a counter
   (`locked_reaction_filter_report.locked_reactions_found = 4`) that equalled Arabidopsis's PathBank species id 4
   (§ 6, F1).

## 3. Where it runs, and how it is recorded

- **Seam.** It runs inside the closure loop of `quarantine_and_close` (`strict_quarantine.py:2692`), after
  `_prune_biological_states`. Every caller goes through this function:
  - `run_quarantine_boundary`, inside `freeze_canonical_payload`: before the final pre-export Stage-3 gate and before
    the canonical hash (the freeze);
  - `run_pwml_export` when the decision is not reused: before `_validate_stage8_export_payload` and the required-field
    gate;
  - the refinement round path (`streamlit_app.py:1127`).

  The batch driver runs no quarantine of its own; it copies these artifacts into the leg directory
  (`driver.py:1884`). So the removal lands before every gate and before the freeze, and no exporter repairs anything afterwards (merge
  rule 8). Placing it in the loop matters: a species row kept alive only by a protein that closure prunes goes in
  the same round (`test_g9_species_kept_only_by_a_removed_protein_goes_with_it`).
- **Record.** Each removal is appended to that round's `removed_entities`. That puts it in
  `removed_entity_report.json` and `graph_closure_iterations.json`, in the shape the degree-zero rows use:
  `{"bucket": "species", "name", "reason": "unreferenced_after_quarantine", "had_external_identity": null,
  "iteration"}`. The human-facing view (`reaction_evidence.py:575-579`) lists `removed_entities` generically, so it
  shows the row with no change to the view. No report gained a key and no `schema_version` moved.
  - The record never reaches LLM-bound text: no module under `curation/`, `rag/`, `llm/`, `extraction/` or
    `mapping/` reads these reports, and nothing is written onto the payload.
  - `quarantine_report.closure.iterations` can grow by one for an affected payload, because the round that removes
    a species row counts as a change.
- **Research mode: unchanged, deliberately.** Research mode's contract is "decide, apply nothing"
  (`quarantine_and_close` docstring; `RESEARCH_DIAGNOSTIC_NOTE`). The payload comes back byte-identical and
  `removed_entity_report` stays empty (`test_preservation_research_mode_applies_nothing`). As with degree-zero rows,
  the closure trace still shows what strict mode would have removed. A research run produces no importable PWML, and
  the batch driver files strict-gate errors there as review flags, so the species row is not a terminal blocker in
  research mode. The census shows every research leg byte-identical.
- **Payloads with no unreferenced species row: byte-identical.** `_species_references` runs only when a candidate
  exists, so a one-species payload does no extra work. 25 synthetic payloads, one per reference kind, plus the base
  payload reproduce the full quarantine output (payload, both reports, closure trace) hashed at base (`PINS`,
  `07-pins-base.log`). 120 of the 124 archived legs are byte-identical (§ 5).

## 4. G9

The tests are labelled per G9. `test_g9_*` is a correction of observable behaviour: the leg's refusal on an
irrelevant row is the terminal blocker PRODUCT_CONTRACT forbids. These tests fail at base on values.
`test_preservation_*` pass at base and at tip.

| run | report | result |
|---|---|---|
| RED, base `src/` | `g11/ORCH-766/08` (`orch766/08-g9-red.log`) | **5 failed, 49 passed**: the 5 `test_g9_*` fail on values (row still present / gate still refuses); all 49 preservation pass |
| GREEN, tip | `09` | **54 passed** |
| real freeze/export path on both menaquinone legs, base | `14` | input hash = leg's `resulting_payload_hash` (fixed point); quarantine `ok` with nothing removed; final Stage-3 PASS; **required-field gate FAILS** `species_missing_taxonomy` + `species_missing_classification` at `/entities/species/2` (759p) and `/1` (1148), exactly the legs' own `pwml_required_field_gate_report.json`; IR clean |
| same, tip | `13` | the row is removed (`iteration 1`, in both reports); final Stage-3 PASS; **required-field gate ok, 0 errors**; IR build + `validate_pwml_ir` + `blocking_pwml_ir_errors` clean; `release_ready` / `review_required` unchanged |

Preservation tests:

- **Every reference kind.** For each kind in § 2, the row is kept and the full quarantine output is byte-identical
  to base (25 kinds).
- **No gate relaxed.** A kept row without a taxonomy id still fails `species_missing_taxonomy` (20 name kinds,
  `first_row` and `create_default_name` included). On the leg, the same abbreviated row, given one state reference,
  is kept and the gate refuses it.
- **The resolved row.** The 684738 row and all 7 Men proteins naming it are kept.
- **Research mode** applies nothing.

Further G9 tests:

- a synthetic unreferenced row is removed and the gate passes;
- a row kept only by a pruned protein goes with that protein;
- an id that collides with a non-species field does not keep a row;
- the tip output is a fixed point.

## 5. Census

`census_species.py` replays `quarantine_and_close` on every `final_mapped.json` with a sibling
`quarantine_report.json` under the primary checkout's `runs_papers/` and `runs_verify/`. Each leg runs with its own
`export_mode`, `strict_db` and `requested_context`, once with base `src/` and once at the tip. That is **124 legs**
(73 strict, 51 research); 18 were skipped for having no `quarantine_report.json`. Base: `g11/ORCH-766/10`; tip:
`11`; data `orch766/census_{base,tip}.json.gz`; diff `orch766/census_compare.out` (`PROBLEMS: none`).

For each removed row, the base run records facts computed independently of the new code:

- name and id leaf hits;
- the IR states `build_pwml_ir` resolved to the row;
- the proteins the writer's resolution (replicated) maps to it;
- whether the row is IR `species[0]`.

For each changed leg, the compare script also checks that the payload outside `entities.species` hashes identically
at base and tip.

**120 legs are byte-identical. 4 change, each by exactly one species row, each provably unreferenced:**

| leg | removed row | taxonomy | verdict base → tip | effect |
|---|---|---|---|---|
| `runs_papers/ORCH-759p/legs/menaquinone-strict/2026-10-22_2154/…/strict` | L. lactis subsp. lactis KF147 | none | `release_ready` → same | required gate: 2 species errors → 0 (the D-111 target) |
| `runs_papers/ORCH-759/legs/menaquinone-strict/2026-10-22_1148/…/strict` | L. lactis subsp. lactis KF147 | none | `review_required` → same | required gate: 2 species errors → 0 |
| `runs_verify/2026-08-24_1203/papers/PMC12452463/strict` | Arabidopsis thaliana (PathBank 4) | 3702 | `review_required` → same | gate unchanged (the row had a PathBank id). Its PWML would carry one fewer, unreferenced, species element (F2) |
| `runs_verify/2026-09-16_2152/papers/PMC13534948/strict` | Hahella chejuensis KCTC 2396 | 349521 | `diagnostic_only` → same (refused: no surviving process) | none: the graph was empty and refused anyway |

Every removed row has: no name hit, no id hit under a species path, no IR state reference, no writer protein
reference, and is not `species[0]`. The only "hit" in the whole census is the counter in F1. Six legs could not
build an IR offline (`UnresolvedCompoundRowError`: legs written before pre-freeze compound resolution). None of them
lost a row.

**Chunk E cohort** (`tests/data/baseline_cohort_manifest.json`, 39 legs, replayed as the test does):
`chunk_e_species_delta.py`, base `20`, tip `21`, diff `chunk_e_compare.out` (`PROBLEMS: none`; the IR is built after
`prefrozen_when_compounded`, as `_ir_codes` does). There are **8 rows in 3 legs**, 16 error rows, every row
unreferenced by the same facts:

- `runs/2026-07-28_2122/…PMC12421875…/research`: L. lactis subsp. lactis KF147;
- `runs/2026-07-28_2122/…PMC12657337…/strict`: Bacillus subtilis;
- `runs/2026-08-02_2130/papers/PMC12421875/strict`: Lactiplantibacillus plantarum, Escherichia coli, Staphylococcus
  aureus, Mycobacterium tuberculosis, Bacillus subtilis, Flavobacterium meningosepticum.

Each of the three legs keeps a *referenced* species row without taxonomy, so each still fails the gate. By leg the
count stays 19/19. `FULL_STACK_BASELINE` does not move: `required_contract_pass` stays 9 and `exportable` stays 9.
Only the row count moves, 27 → 19. That is the pin moved in `bcad7547` (§ 7).

## 6. Findings

- **F1. An id compared on any path over-keeps.** The first census (`01`, scalar keys on every path) kept the
  Arabidopsis row in `runs_verify/2026-08-24_1203/…/PMC12452463/strict` only because
  `locked_reaction_filter_report.locked_reactions_found = 4` equals its `pathbank_species_id` 4. Every id reader in
  § 2 (rows 4, 6, 13) sits under a key naming `species`, `taxon` or `organism`, so ids are restricted to those paths.
  Names still match any scalar, because the IR resolves a state's species by the state's own name (§ 2, row 3).
  Test: `test_g9_an_id_collision_outside_a_species_field_does_not_keep_a_row`.
- **F2. D-111 removes unreferenced rows that pass the gate, too.** Two of the four census rows carry a taxonomy id.
  One leg exports (`review_required`): `PMC12452463` (2026-08-24_1203). Its PWML would lose a species element that
  nothing references. This is what D-111 § 1 says ("a species row that nothing in the pathway references is
  removed"), but it is a visible content change beyond the gate failure that motivated it. See § 9.
- **F3. One row is kept only by the broad scan.** "E. coli" (562) in `runs_verify/2026-08-24_1428/…/PMC12444477/strict`
  is named by nothing the writer resolves; the writer gives complexes `species[0]`. A complex's own
  `taxonomy_id` / `species_ref.taxonomy_id` is 562, though, so the payload says the complex is *E. coli*. It is kept.
  That is the "when unsure, keep" direction.
- **F4. The first-row rule fired zero times.** No archived row is kept *only* because it is the first named row.
- **F5. The census and Chunk E agree with the theory.** Every one of the 12 removed rows is unreferenced by the IR,
  the writer and the scan alike, so the scan never removed a row the real resolvers needed.

## 7. Gates

All runs in this worktree, `T2PW_OFFLINE_CURATOR=1`, through `bounded_run.py` and `pinned_pytest.py`; every report
`FINAL SURVIVING COUNT : 0`.

| gate | report | result |
|---|---|---|
| G9 RED (base) / GREEN (tip) | `08` / `09` | 5 failed + 49 passed / 54 passed |
| leg replay base / tip | `14` / `13` | gate fails on the row / gate ok, IR clean |
| census base / tip | `10` / `11` | 124 legs; 4 changed, 4 rows, all provably unreferenced |
| strict-quarantine suites + 759c + C-011 golden + Stage-3 lifecycle + `test_pwml_ir` + `test_pwml_writer` + focused | `15` | **324 passed** |
| Chunk E alone: base / tip before pin move / tip after | `19` / `16` / `22` | 175 passed / 1 failed (`RESIDUAL_CODES_BY_ROW` 27 → 19, the deliberate move) / **175 passed** |
| species/prefreeze/removed_entity/required_field files (105, three batches) | `23`, `24`, `25` | 751 passed + 3 failed; 806 passed + 4 skipped + 1 failed; 786 passed + 8 skipped + 1 failed. All 5 failures known (below) |
| app-seam files batched (`batch_research_gate_fail_open`, `c030a`, `c082`) | `26` | 25 passed |
| alone: `prefreeze_third_export_seam` / `reaction_evidence_view` / `orch759a_fold_app_path` | `27` / `28` / `29` | 12 / 11 / 2 passed |
| **SMOKE** | `30` | **508 passed** |
| `tests/test_eval_v2_*.py` | `31` | 89 passed, 1 xfailed |
| alone: `streamlit_quarantine_boundary` / `c052` / `c055` | `32` / `33` / `34` | 23 / 13 / 14 passed |
| the 5 failing nodes' files, base / tip | `35` / `36` | 5 failed + 136 passed at both; every `E` line identical |

Known failures, all red at base with identical messages (`35`/`36`):

- `test_c100_tolerance_scope.py::test_5b_…`
- `test_compound_resolution_extraction.py::test_the_golden_covers_every_committed_leg_fixture` and
  `::test_the_repaired_coverage_and_digest_pins_themselves_go_red`
- `test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform`

  These three files are in TEST_MATRIX § ORCH-754.
- `test_paper_eligibility_corrections.py::test_the_2122_…` (worktree only, § ORCH-759a/b/c/d).

The 12 skips in `24`/`25` were not itemised (pytest `-q`). Chunk D was not run; the orchestrator owns it.
`test_orch759b_recorder_reattach.py` was not run: it is AppTest, does not mention species, and is not in the
affected set.

## 8. Failures and dead ends

1. **A preservation test carried a tip assertion.** The first RED (`03`) failed
   `test_preservation_menaquinone_resolved_row_684738_is_kept` at base on `set(rows) == {3 names}`, which only the tip
   satisfies. I changed it to a subset check (`076675d8`) and re-ran RED (`04`). The removal itself is asserted by the
   G9 test.
2. **The first replay script crashed on an import** (`12`): `blocking_pwml_ir_errors` lives in `writer.py`, not
   `ir.py`. Fixed and re-run as `13`.
3. **The first census (`01`, `06`) mixed names and ids in one leaf-hit list**, which is how F1 surfaced. It was
   superseded by `10`/`11`, which split `name_hits` / `id_hits_species_path` / `id_hits_other_path` and added a
   hash of the payload outside `entities.species`. The `01`/`06` JSON outputs were scratch-only and are not
   committed; their logs are.
4. **The first Chunk E delta (`17`, `18`) had no IR facts.** These legs predate pre-freeze compound resolution, so
   every IR build raised `UnresolvedCompoundRowError` and the compare refused to call any row proven. Re-run as
   `20`/`21` with `prefrozen_when_compounded`, the Chunk E test's own route.
5. **Considered and rejected:**
   - **A hand-enumerated field list.** One missed field would silently delete a referenced organism. The superset scan
     can only over-keep (F3).
   - **One pass after the loop.** It would miss a species kept alive only by a protein closure prunes in a later
     round, and would need its own record. In the loop it is recorded by the existing mechanism.
   - **Running it in research mode.** That would break research's apply-nothing contract.

## 9. Not settled: for the product owner

1. **Unreferenced rows that pass the gate (F2).** D-111's wording removes them, and the tip does. The cost is one
   archived exporting leg whose PWML drops one unreferenced species element (Arabidopsis, PathBank 4), plus a refused
   leg with no outcome change.
   - **Option (a), keep as is:** D-111's literal reading, and no unreferenced species ships.
   - **Option (b), restrict removal to rows the gate would refuse** (no PathBank species id and no valid taxonomy +
     classification). That is a narrower diff in effect, but it is a gate-shaped condition inside closure.

   Implemented: (a).
2. **The first named row is never removed**, because the writer emits it as the pathway species and as every
   complex's species. An unreferenced first row would therefore still block. That fired 0 times in 124 + 39 legs.
   Removing it would change the pathway's species in the PWML, which is a biology decision.
3. **The broad scan keeps a row named by any scalar** (a report field, an evidence string that equals the name
   exactly, the requested organism). It is safe, but a future pooled row that some report field echoes would still
   block. 1 such keep in the census (F3), and it is arguably a real reference.
4. **`runs_smoke/` and `runs_validation/` were not in the census** (the brief named `runs_papers/` and `runs_verify/`).
5. **ORCH-767 (tonight's run)** measures whatever code it started on. If this lane is not merged before it starts,
   menaquinone will refuse again on the same row.

## 10. TEST_MATRIX delta (exact text to append; not edited here)

```
## ORCH-766 (D-111 § 1): test obligations — appended end-of-file by rule · 2026-10-24

Lane `orch766/drop-unreferenced-species` (tests `0f54f3b4`, `076675d8`, `78158f07`; code `f775f2b3`; pin move
`bcad7547`). Reports `evidence/g11/ORCH-766/01`–`36`.

New focused selection: `tests/test_orch766_unreferenced_species.py` (**54**, ~1.5 s), fixture
`tests/fixtures/orch766/` (the ORCH-759p menaquinone leg's `final_mapped.json`, gzipped, + its requested context).
Run it with the strict-quarantine suites and Chunk E for any change to `quarantine_and_close`'s closure loop,
`_prune_unreferenced_species` / `_species_references` / `_species_row_keys`, the IR's species resolution
(`ir.py` `resolve_component`, `SPECIES_CREATE_DEFAULTS`, `_component_record`) or the writer's species ids
(`writer.py` `protein_species_id`, `_ir_pathway_species_id`). Its `PINS` are full quarantine outputs hashed at base
`d6849c9e` (`evidence/orch766/pins_at_base.py`); a payload with no unreferenced species row must not move them.

Pin moved deliberately (merge rule 4): `tests/test_strict_quarantine_real_artifact_replay.py`
`RESIDUAL_CODES_BY_ROW` `species_missing_classification` / `species_missing_taxonomy` **27 -> 19** (8 unreferenced
species rows in 3 cohort legs removed; `evidence/orch766/chunk_e_compare.out`). `RESIDUAL_CODES_BY_LEG` stays 19/19,
`FULL_STACK_BASELINE` unchanged (required contract 9/28, exportable 9/28). `docs/change_log.md` moved with it, same
commit. Chunk E: 175 passed at base (`19`) and at tip after the move (`22`).

SMOKE unchanged at **508** (`30`). Merge rule 10 unchanged. No other pin moved. Known failures unchanged
(§ ORCH-754, § ORCH-759a/b/c/d), red at base with identical messages (`35`/`36`).
```
