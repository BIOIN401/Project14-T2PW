# ORCH-773: RESULT (D-114 § 2: a per-row export failure is omitted before the freeze, not a pathway refusal)

| | |
|---|---|
| Lane | `orch773/omit-per-row`, worktree `.claude/worktrees/orch773` |
| Base | `3ef3fa9c` (sprint tip; `src` tree `322d398d`, identical to `main` `b16a45e7`) |
| Commits | phase 1: `0971487b` code and tests · `f35de933` docs. Round 1 (§ R1): `1a0fc159` code and tests · the docs commit after it |
| Status | **Phase 1 reviewed (r0: APPROVE WITH NOTES, 0 blocking). Round 1 implemented and measured, awaiting the round-1 review.** Not merged, not pushed. Phase 2 (coverage per leg, ORCH-772's metric) waits for the orchestrator. |
| Boundary | `src/t2pw/pipeline/strict_quarantine.py`; in round 1 also `process_normalizer.py`, three shared constants only (review note N4, asked for by the orchestrator; the gate's messages are byte-identical); tests; this file; evidence. `streamlit_app.py`, `reaction_evidence.py`, `reaction_evidence_view.py`, `apply_audit_patch.py`, `evidence/orch768_replay_corpus.py`: **untouched**. |
| D-112 § 2 | **ACCEPT** (phase 1 and round 1). PWML **56/91 → 72/91** (56/81 → 72/81 replayed). 16 legs fixed, none broken, no gold recall lost. **56/56** base-passing legs write a **byte-identical** PWML. Enterobactin ORCH-767 writes a PWML with gold **5/5**. Round 1 keeps the 6 transports whose transporter failed (§ R1.3). |
| G11 | `evidence/g11/ORCH-773/01`–`76`, all `FINAL SURVIVING COUNT : 0`, every job under `--heavy-lock ORCH-773`; `g11_evidence.py check --task ORCH-773` exit 0. Logs `evidence/orch773/NN-*.log`. |

## 0. Outcome in one table

| | base `3ef3fa9c` (`01-corpus-base`) | tip `0971487b` (`35-corpus-tip`) |
|---|---|---|
| PWML written, strict legs with an outcome | **56/91** | **72/91** |
| replayable failures | 25 | 9 (8 × `no_defensible_reaction_support`, 1 × empty graph: both whole-pathway, unchanged) |
| base-passing legs whose PWML bytes changed | | **0 of 56** (`37-corpus-delta`) |
| gold recall lost | | none (`36-corpus-compare`) |
| network attempts | 0 | 0 |

The base JSON is **byte-identical** to ORCH-769c's merged-tip corpus record `evidence/orch769c/corpus_int_tip.json`
(sha256 `5c5579bf…` for both): an independent repeat of the sprint baseline. The tip JSON is byte-identical across
three runs (`10`, `20`, `35`; sha256 `ce612a2d…`).

---

## 1. Inventory: every gate in the replayed tail that refuses the whole pathway, per-row or whole-pathway

Method: the corpus at base (`01`), every failing leg's failing step (`evidence/orch773/failing_legs.py` → `02`), then
every downstream gate run on each leg's quarantined payload with each error's row pointer
(`probe_gate_rows.py` → `04`, all 81 legs: errors on exactly the 25 failing legs, none on the 56 passing ones).
"Per-row" means the gate's own error names ONE process or entity row; "whole-pathway" means it names none, or names
the species, the pathway, or nothing surviving.

| # | gate (file:function) | code / message | class | base legs | converted? |
|---|---|---|---|---|---|
| 1 | strict quarantine refusal, `strict_quarantine.quarantine_and_close` via `_unexportable_entities` → `_entity_representability` | `unexportable_entity:<n>` (row reason `protein_missing_external_identity`, ...) | per-row (entity) | **4**: ORCH-755 enterobactin, ORCH-767 Wnt/Ca2+, rv 08-24_1203 PMC12856317, rv 09-02_2052 PMC12444477 | **yes** |
| 2 | same, `_entity_type_overlaps` | `entity_type_overlap:<n>` | per-row **pair**: two declarations of one name; which one to omit is not decidable (the function's own docstring) | 0 | **no** (§ 12 item 3) |
| 3 | same, `_degree_zero_exports` | `degree_zero_export:<n>` | invariant assertion: empty by construction after convergence (its docstring); a hit is a code defect, not a row | 0 | no |
| 4 | same, `_reconcile_locked_reactions` | `unaccounted_locked_reactions:<n>` | whole-pathway (lock accounting) | 0 | no |
| 5 | same, closure loop | `closure_not_converged:<n>` | whole-pathway (algorithm) | 0 | no |
| 6 | same, `evaluate_core_coverage` / `release_status.coverage_verdict` | `minimum_core:no_surviving_process` | whole-pathway (nothing survives) | **1**: rv 09-16_2152 PMC13534948 | no |
| 7 | final pre-export gate, `process_normalizer.run_strict_post_normalization_gates` (run by `streamlit_app.freeze_canonical_payload` and by `_validate_stage8_export_payload`) | "Protein has degree 0 after normalization: `<p>`" / "Located protein is isolated in connectivity graph: `<p>`" | per-row (entity attached to nothing) | **2**: ORCH-759 enterobactin, rv 08-18_1328 PMC12452463 | **yes** |
| 8 | same | "Generated protein complex '`<c>`' component '`<p>`' is missing positive stoichiometry." (and the wrapper's other checks: species, no component, unresolved component, component species/identity, malformed Unknown) | per-row (entity) | **2**: ORCH-767 enterobactin (D-113 § 1), rv 08-21_2057 PMC12452463 | **yes** |
| 9 | same | protein missing species / UniProt / placeholder posing / wrapper named as protein (`/entities/proteins/<i>`) | per-row (entity) | 0 (row 1 pre-empts it) | yes |
| 10 | same | "Unknown protein/modifier reference", "Unknown transporter reference", forbidden complex (`/processes/<b>/<i>/<slot>/<j>`) | per-row (process, actor) | 0 (ORCH-769b pre-empts it) | yes |
| 11 | same | protein and complex share a name | per-row pair | 0 | no: row 2 refuses first |
| 12 | same | lock accounting; `n_plus_tokens_remaining`; Composite / Registry / Scaffold validator exceptions (path `/processes`, message an exception string) | whole-pathway, or names no row | 0 | no |
| 13 | Stage-8 contract, `stage_contracts.validate_post_normalization` → `_validate_canonical_actor_rows` (run in `streamlit_app.run_pwml_export`) | `actor_schema_not_canonical` | per-row (process, actor) | **2**: ORCH-755 sour taste, rv 09-02_2052 PMC12452463 (+ ORCH-767 Wnt/Ca2+ behind its refusal) | **yes** |
| 14 | required-field gate, `stage_contracts.validate_pre_export` → `pwml.ir.validate_required_pwml_contract` | `reaction_enzyme_must_be_protein_complex` | per-row (process, actor) | **4**: ORCH-759 chlorophyll, rv 08-21_2057 PMC12096016, rv 09-06 PMC11405693, rv 09-06 PMC12376012 | **yes** |
| 15 | same | other codes with a process or entity-row pointer (`enzyme_reference_not_found`, `reaction_missing_*_participants`, `duplicate_reaction_enzyme_complex`, `protein_missing_*`, `protein_complex_missing_*`, `generated_complex_component_*`, `compound_missing_name`) | per-row | 0 | yes |
| 16 | same → `reaction_support.reaction_support_issue` | `no_defensible_reaction_support` (pointer `/processes/reactions`) | whole-pathway (D-094: no reaction anywhere has support) | **8** | no |
| 17 | same | `species_missing_taxonomy` / `_classification` | whole-pathway (species identity, D-114 § 4) | 0 (ORCH-766 fixed it) | no |
| 18 | same | pathway metadata, `no_biological_states`, `biological_state_*`, `location_*`, `visualization_*`, `visible_entity_missing_location_state` | whole-pathway or not a process/entity row | 0 | no |
| 19 | IR, `pwml.ir.build_pwml_ir` + `validate_pwml_ir`, blocking per `pwml.writer.blocking_pwml_ir_errors` (in `run_pwml_export`) | `protein_complex_missing_components` | per-row (entity) | **2**: rv 08-25_1216 PMC12444477, rv 08-25_1216 PMC12856317 | **yes** |
| 20 | same | other IR errors on `proteins` / `protein_complexes` rows | per-row | 0 | yes |
| 21 | same | IR errors on processes, compounds, states | IR pointers address the IR's own lists; the required-field gate checks the same predicates on payload rows first | 0 | no |
| 22 | batch driver verdict, `gate_reports.gate_verdict` / `driver._blocking_reports` | `upstream_contract_report` | recorded upstream reports, no row in the frozen payload | 0 | no |

**25 failing legs at base: 16 per-row (all converted, all fixed), 9 whole-pathway (unchanged).**

## 2. Design

All of it is in `strict_quarantine.py`, which `freeze_canonical_payload` runs before the freeze (merge rule 8).

1. **When.** Inside `quarantine_and_close`'s closure loop, on the first iteration that changes nothing, a per-row
   omission round runs (strict mode only; research applies nothing). If it omits something the loop continues and
   closure runs again to a fixpoint; the next settled iteration runs another round.
2. **What each round checks: the gates themselves, not copies** (`_per_row_failures`, `_downstream_gate_errors`).
   `_freeze_candidate` builds, on a copy, exactly the graph `quarantine_and_close` would return now (the same three
   post-loop steps: `_drop_quarantined_processes`, `_reconcile_locked_reactions`, `restore_autostates_if_required`).
   Then, in the order the freeze and the export meet them, and only while every earlier one passed:
   1. the module's own `_unexportable_entities` (predicate `_entity_representability`);
   2. `run_strict_post_normalization_gates(candidate, enforce_all_proteins_connected=True)`, as freeze and Stage 8 call it;
   3. `validate_post_normalization(candidate, <that report>)`, as `_validate_stage8_export_payload` calls it;
   4. `validate_pre_export(candidate with run_pwml_export's metadata defaults, strict_db)`;
   5. `build_pwml_ir` + `validate_pwml_ir`, blocking errors as `run_pwml_export` counts them.

   Every error that names ONE row is mapped back by its own pointer (`_row_failures`): a process pointer to the
   admission record of that row of the candidate, an entity pointer to the same row of the working graph (name
   checked), an IR entity pointer through the IR record's name. Anything else names no row.
3. **The omission unit** (`_apply_per_row_failures`):
   * a **process** a gate names leaves (`quarantined_broken_reference`, reason = the gate's code or message);
   * an **entity** a gate refuses leaves with every process that resolves a reference to it
     (`quarantined_unmapped_entity`); a complex that resolves a component to it (`_resolve_component`, the same
     resolution `_entity_representability` uses, now extracted) leaves whole with its own processes, because
     removing the row alone would rewrite what the complex is; then the row itself is removed and recorded;
   * a protein the final gate calls **attached to nothing** (degree 0 / located isolated) is removed alone: by that
     gate's own test no process attaches to it. Closure then re-validates every survivor, as it always did;
   * **D-113 § 1**: when the refused entity is a generated wrapper with `generation_reason ==
     "complex_named_source_entity_wrapper"`, every actor in a reaction resolving to it is marked
     `provenance: inferred`, and the reaction keeps an `extracted` enzyme, the wrapper leaves that reaction's
     `enzymes`/`modifiers` and the reaction stays (`_invented_wrapper_slots`). D-113 § 2: the positive-count gate is
     untouched and still refuses the wrapper.
4. **An omission is kept only when it buys a PWML.** If the rounds end on a refusal that names no row, on a check
   that cannot run, on a type overlap or on an empty graph, every round is rolled back and the pathway is refused
   exactly as at base (D-113 § 3). Measured cost on the corpus: none (`20` = `10`, byte-identical); on the test
   suites it is what keeps toy payloads that would never export from changing.
5. **Release.** Any omission adds `per_row_omissions_removed_from_pwml:<n>` to `review_reasons` and to
   `release.reasons`, and caps `release_ready` → `review_required` (`_cap_release_for_omissions`, the shape of
   `release_status.cap_release_for_prefreeze_declination`: monotone, `strict_acceptance_eligible` only ever set
   False). All 16 fixed legs export `review_required` (`pathway.review_required.pwml`).
6. **Versioning (the module's own rule).** `QUARANTINE_POLICY_VERSION` `2026-07-31.1` → `2026-10-25.1`: the change
   can change a verdict for an unchanged payload. `removed_entity_report.schema_version` 1 → 2: its rows grow
   `detail` (the house rule `test_strict_quarantine_lineage` cites). `quarantine_report` keeps schema 6: its rows
   already carry `detail`.
7. **Recorded in the existing shapes only.** A process: a `quarantined[]` row with the existing fields (`name`,
   `process_kind`, `pointer`, `state`, `reason` = the gate's exact code or message, verbatim `original_process`,
   `essential_participant` = the actor or participant through which it fails) and `detail.omission` =
   `{gate, cause: process|actor, slot, name, failing_entity}`. An entity: a `removed_entities[]` row
   (`bucket`, `name`, `reason`, `had_external_identity`) with `detail` = `{gate, omission, actor_removed_from}`.
   The current Reaction Evidence text shows both (QUARANTINED: state, reason, participant; REMOVED ENTITIES:
   bucket, name, reason). No new channel.

## 3. No gate verdict changed

* No gate module is in the diff: `process_normalizer.py`, `stage_contracts.py`, `pwml/ir.py`, `pwml/writer.py`,
  `release_status.py`, `reaction_support.py` are byte-identical to base. The pass **calls** them on a copy.
* The quarantine's own predicates are refactored, not changed: `_resolve_component` is lifted verbatim out of
  `_entity_representability`, and `_process_references` out of `_referenced_entity_norms` (the same set). Proof:
  Chunk E 175/175 unchanged (`39`), the strict-quarantine suites (`38`), and every base-passing leg's canonical
  payload unchanged (`37`: on 56/56 the only files that differ are `quarantine_report.json` and
  `removed_entity_report.json`, by the version fields; `canonical.json` and the PWML are byte-identical).
* Nothing exported that a gate rejects: on every fixed leg the replay's freeze gate, Stage 8, required-field gate
  and IR pass at tip (`35`, `export ok`), and the G9 tests run all four, including the export's own pre-freeze seam
  before the IR, on the tip result (`34`).
* **The IR check runs without the export's pre-freeze seam** (`run_prefreeze_resolution` renames compounds and
  species and may consult the PathBank DB; running it inside the quarantine would add DB traffic before the
  freeze). The pass maps only IR errors on protein and complex rows. Measured on all 81 legs: the protein/complex
  IR verdict is identical with and without the seam, 81/81 (`ir_prefreeze_equivalence.py` → `15`).

## 4. The corpus measurement

Run per ORCH-768-RESULT § 7: base = `git archive 3ef3fa9c src reference/PW000001.pwml data/pathwhiz_id_db.json`
under `C:/t/orch773/base`; tip = the worktree `src`; `PATHBANK_DB_*` unset (`env -u` in the wrapper); absolute
paths. Records committed: `evidence/orch773/corpus-base.{json,txt}`, `corpus-tip.{json,txt}`.

`compare base.json tip.json --target ORCH-767/enterobactin-strict/2026-10-23_2224` (`36`, exit 0):

```
pass rate base 56/91 -> tip 72/91
fixed: [16 legs, below]
passing legs turned failing: none
gold recall lost: none
still failing, at a different step (information): none
target ORCH-767/enterobactin-strict/2026-10-23_2224: FIXED
D-112 § 2: ACCEPT
```

### 4.1 Per leg: verdict, omissions, reason, KI id (`corpus_delta.py` → `37`)

KI ids are read from `PWML_KNOWN_ISSUES.md`'s appendix by the script itself (exact, then prefix/suffix, then the
human-message table). Base verdict → tip: every one below is `review_required`, `pathway.review_required.pwml`.

| leg | base failure | omitted at tip (P = process, E = entity) | reason → KI | cause |
|---|---|---|---|---|
| ORCH-755 enterobactin 2201 | quarantine `unexportable_entity:1` | P reaction `EntB isochorismatase reaction`; P interaction `2,3-DHB competitive inhibition of EntB`; E complex `isochorismatase`; E protein `EntB` | `protein_complex_component_unrepresentable:EntB:protein_missing_external_identity` → KI-015 (inner KI-001); EntB `protein_missing_external_identity` → KI-001 | reaction: actor (`enzymes`, `isochorismatase`); interaction: process (`entity_2`) |
| ORCH-755 sour taste 0139 | Stage 8 `actor_schema_not_canonical` | P × 3 transports (`OTOP1-mediated H+ transport`, `H+ permeation through OTOP1`, `OTOP1-mediated H+ permeation`) | `actor_schema_not_canonical` → KI-023 | actor (`transporters`, `OTOP1`) × 3 |
| ORCH-759 chlorophyll 1231 | required `reaction_enzyme_must_be_protein_complex` | P reaction `CPO reaction` | → KI-006 | actor (`enzymes`, `CPO`) |
| ORCH-759 enterobactin 1037 | final gate, degree 0 | E protein `enterobactin synthase` | "Protein has degree 0 after normalization: enterobactin synthase" → KI-013 | entity only, no process |
| ORCH-767 enterobactin 2224 | final gate, wrapper stoichiometry | E complex `enterobactin synthase complex`, removed from `EntF enterobactin synthase reaction` `enzymes`+`modifiers`; `EntF complex` kept (D-113 § 1) | stoichiometry message → KI-012 | actor removed, reaction kept |
| ORCH-767 Wnt/Ca2+ 0314 | quarantine `unexportable_entity:1` | E protein `phosphorylated CAMKII` (no process resolves to it); P × 2 transports `IP3-induced Ca2+ release`, `SERCA pumps Ca2+ into ER` | KI-001; `actor_schema_not_canonical` → KI-023 | entity only; actor (`transporters`, `inositol-1,4,5-trisphosphate receptor (InsP3R)`) × 2 |
| rv 08-18_1328 PMC12452463 | final gate, degree 0 | E protein `Isochorismatase (EntB)` | degree-0 message → KI-013 | entity only |
| rv 08-21_2057 PMC12096016 | required `reaction_enzyme_must_be_protein_complex` | P reaction `LDH coupled reaction` | → KI-006 | actor (`enzymes`, `lactate dehydrogenase`) |
| rv 08-21_2057 PMC12452463 | final gate, wrapper stoichiometry | P reaction `enterobactin synthase complex reaction`; E complex `enterobactin synthase complex` | stoichiometry message (`EntD`) → KI-012 | actor (`enzymes`); D-113 § 1 does not apply: no other catalyst, actor `extracted`, reason `complex_inferred_from_evidence` |
| rv 08-24_1203 PMC12856317 | quarantine `unexportable_entity:2` | P × 2 interactions (`ALAS2 recruits CLPXP for degradation`, `ALAS1 degraded by CLPXP`); E complex `CLPXP`; E × 2 Clp subunit proteins | KI-015 (inner KI-001); KI-001 | process (`entity_2`) × 2 |
| rv 08-25_1216 PMC12444477 | IR `protein_complex_missing_components` | P interaction `FtsH/YciM complex regulation of LpxC`; E complex `FtsH/YciM complex` | → KI-015 | process (`entity_1`) |
| rv 08-25_1216 PMC12856317 | IR `protein_complex_missing_components` | P interaction `LONP1 degrades ALAS1`; E complex `LONP1` | → KI-015 | process (`entity_1`) |
| rv 09-02_2052 PMC12444477 | quarantine `unexportable_entity:2` | P interaction `FtsH/YciM complex degrades LpxC`; E complex `FtsH/YciM complex`; E proteins `FtsH`, `LapB` | KI-015 (inner KI-001); KI-001 | process (`entity_1`) |
| rv 09-02_2052 PMC12452463 | Stage 8 `actor_schema_not_canonical` | P transport `ferri-enterobactin import via FepA` | → KI-023 | actor (`transporters`, `FepA`) |
| rv 09-06_1425 PMC11405693 | required `reaction_enzyme_must_be_protein_complex` | P reaction `C4H hydroxylation` | → KI-006 | actor (`enzymes`, `AtCYP73A5`) |
| rv 09-06_1425 PMC12376012 | required `reaction_enzyme_must_be_protein_complex` | P reaction `ceramide synthase reaction` | → KI-006 | actor (`enzymes`, `ceramide synthase`) |

Closure consequences, recorded with their existing reasons (`37`, "every quarantine/removal record that differs"):
entities a removed process alone referenced (`degree_zero_after_quarantine`, `unreferenced_after_quarantine`), their
location rows (`location_entity_removed`) and orphaned states (`state_unreferenced_after_quarantine`), e.g. OTOP1, H+
and 5 states on sour taste; ALAS1 on the LONP1 leg. No `stranded_by_closure` row appears on any leg.

**Unchanged (9):** `no_defensible_reaction_support` × 8 and rv 09-16_2152 PMC13534948 (no surviving process): every
record of these legs is identical at base and tip.

### 4.2 Process-vs-actor counts (prices "drop only the bad actor, keep the reaction")

| | count | where |
|---|---|---|
| processes omitted | **18** in 12 legs | |
| … caused by **one actor** | **12** (6 reactions via `enzymes`, 6 transports via `transporters`) | 755 enterobactin 1, 755 sour 3, 759 chlorophyll 1, 767 Wnt/Ca2+ 2, rv 08-21_2057 ×2 1+1, rv 09-02_2052 PMC12452463 1, rv 09-06 ×2 1+1 |
| … caused by **the process itself** (an interaction endpoint) | **6** | 755 enterobactin 1, rv 08-24_1203 2, rv 08-25_1216 ×2 1+1, rv 09-02_2052 PMC12444477 1 |
| D-113 § 1 actor removals (reaction kept) | 1 | 767 enterobactin |
| entities removed by the pass | 15 | |

If the product owner later chooses "drop only the bad actor", the 12 actor-caused processes would stay, uncatalysed
or without a transporter; that would restore 755 enterobactin to gold 5/5 (its `EntB isochorismatase reaction` is a
gold reaction lost to KI-017). Not implemented (§ 12 item 1).

### 4.3 Base-passing legs: byte identity, and gold

* **56/56 identical PWML** (`pwml_sha256` and file name; `37`). The only other files that differ are
  `quarantine_report.json` and `removed_entity_report.json`, by the version fields (§ 6).
* Gold (certified, enterobactin and menaquinone; `37`): every base row unchanged at final and PWML level. New PWML
  rows: **ORCH-767 enterobactin PWML 5/5** (R 1.000), ORCH-759 enterobactin PWML 5/5, rv 08-21_2057 PMC12096016
  PWML 5/5, ORCH-755 enterobactin final and PWML 4/5 (that leg had no canonical payload at base; the missing reaction
  is the omitted EntB step).

## 5. G9 (merge rule 9): base red, tip green

`tests/test_orch773_omit_per_row.py`, fixtures `tests/fixtures/orch773/` (README there: each fixture is the exact
freeze input of one corpus leg at base, sha256 asserted). Base = an export of `3ef3fa9c` (`src tests pytest.ini
pyproject.toml reference data/pathwhiz_id_db.json`) plus the new test and fixtures under `C:/t/orch773/basetree`,
`src` diff-identical to the corpus base export; run with `pinned_pytest.py --expect-tree`.

| test | base (`33-g9-red-base`) | tip (`34-g9-green-tip`) |
|---|---|---|
| `test_g9_unexportable_component_takes_its_complex_and_processes_not_the_pathway` (755 enterobactin) | **FAIL** `['quarantine:unexportable_entity:1'] != []` | pass |
| `test_g9_unexportable_row_kept_only_by_a_synonym_is_removed_alone` (767 Wnt/Ca2+) | **FAIL** same | pass |
| `test_g9_final_gate_degree_zero_protein_is_removed_and_every_reaction_stays` (759 enterobactin) | **FAIL** `final:Protein has degree 0 …` | pass |
| `test_g9_d113_inferred_wrapper_leaves_its_reaction_and_the_paper_catalyst_stays` (767 enterobactin) | **FAIL** `final:Generated protein complex … stoichiometry.` | pass |
| `test_g9_final_gate_wrapper_with_no_other_catalyst_omits_its_reaction` (rv 08-21_2057 PMC12452463) | **FAIL** same | pass |
| `test_g9_stage8_non_canonical_transporter_omits_its_transport` (755 sour) | **FAIL** `stage8:actor_schema_not_canonical` | pass |
| `test_g9_required_gate_bare_protein_enzyme_omits_its_reaction` (rv 08-21_2057 PMC12096016) | **FAIL** `required:reaction_enzyme_must_be_protein_complex` | pass |
| `test_g9_ir_complex_without_components_omits_its_interaction` (rv 08-25_1216 PMC12856317) | **FAIL** `ir:protein_complex_missing_components` | pass |
| 5 × `test_preservation_*` (whole-pathway refusal still refuses; omissions that buy no PWML roll back; a passing leg untouched; research omits nothing; all-rows-fail rolls back) | pass | pass |
| `test_new_release_cap` (labelled NEW, no base failure claimed) | fails on its import (inside the test) | pass |

Base: **9 failed, 5 passed**; tip: **14 passed**. Every G9 failure is on a value: the base refuses the pathway, the tip
exports it `review_required` with the row recorded, and the result passes the freeze gate, Stage 8, the
required-field gate and the IR (with the export's pre-freeze seam).

## 6. Pins moved (merge rule 4), old → new

| test | pin | old | new | why |
|---|---|---|---|---|
| `tests/test_orch768_replay_corpus.py` | `EXPECTED[ENTEROBACTIN]` | `{"pwml_written": False, "failure_stage": "final_pre_export_stage3_gates"}` | `{"pwml_written": True, "failure_stage": None, "canonical_matches_leg": False}` | the wrapper is omitted; the canonical payload loses it |
| same | `test_replay_reproduces_the_recorded_verdict`: fidelity of enterobactin | `[]` | the three diffs `outcome recorded=final_pre_export_stage3_gates replay=pwml`, `final_gate_ok recorded=False replay=True`, `final_gate_errors recorded=[<the wrapper error>] replay=[]`; `canonical_matches_leg` read from `EXPECTED` | the deliberate verdict move |
| same | `test_enterobactin_fails_on_the_wrapper_component_exactly_as_recorded` | asserts the replayed gate fails as recorded | renamed `…_recorded_failure_is_omitted_not_refused`: the recorded error is the wrapper, the replayed gate is `{"ok": True, "codes": [], "errors": []}`, release `review_required` | same |
| same | `test_the_fidelity_check_bites`, second channel | mutates enterobactin's gate errors | mutates sour taste's (its gate matches its record), and first asserts no diff unmutated | enterobactin's gate now differs from its record by design, which made that channel vacuous |
| `tests/test_orch766_unreferenced_species.py` | `PINS` (28 digests) | ORCH-766 base-SHA digests (listed below) | the tip digests | **only the five version fields move**: `pins_delta.py` strips `quarantine_report.policy_version`, `decision_inputs.policy_version`, `decision_input_hash`, `decision_id` and `removed_entity_report.schema_version` and gets identical digests at base and tip for all 28 (`24`, `25`) |
| `tests/test_orch769b_actor_protein_registry.py` | `decision_input_hash` | `sha256:6c5f9c8b2410d8af` | `sha256:0bb36bd9e1f252d9` | policy version |
| `tests/test_orch759c_quarantine_bucket_closure.py` | `decision_input_hash` | `sha256:d2ef87f2ae5da361` | `sha256:8b2be66a5cacc344` | policy version |
| `tests/test_strict_quarantine_lineage.py` | `removed_entity_report["schema_version"]` | `1` | `2` | `detail` on rows |
| `tests/test_d088_coverage_diagnostics.py` | `SQ.QUARANTINE_POLICY_VERSION == report["policy_version"]` (C-114: "this card must not have touched it") | live constant equals the archived legs' | `report["policy_version"] == "2026-07-31.1"` (the archived legs were decided under it); the decision-id check unchanged | the deliberate bump (§ 12 item 2) |

Old ORCH-766 pins (base) → new (tip):
`base` a429b8e8…ba40 → 32f3a34e…ed1b · `state_species` 03e1cbf6…1557 → 59c95f83…7613 · `state_organism` ba2c8c30…46e6 →
7bcc8be8…1f70 · `state_name` 5091f4d1…36d7 → 6ed999a5…e054 · `protein_species` 2a0278e9…9ae3 → 0c49a4c5…358a ·
`protein_organism` 585a9b79…4f15 → 43692887…1f4e · `protein_species_ref_name` 3741602c…e591 → 403562c2…94f9 ·
`protein_mapping_meta_species` b3e2f473…997c → b242c789…6902 · `complex_species` c3f053de…579d → 06bca968…c452 ·
`complex_organism` 85b51015…873e → 78ca9bcc…1549 · `other_entity_species` b90e8262…46c5 → 45fdda53…f6b5 ·
`location_row_species` 7fc8bb13…46fa → ddae74b6…908b · `pathway_metadata_organism` 54cd3b41…9825 → 4396f5fb…9c68 ·
`surviving_process_field` 246cc23d…0fc9 → d38f6f14…8460 · `alias` 57ab20ac…488b → 266c611e…fa60 · `synonym`
57bfe1bd…c344 → 85b1f3ec…c419 · `raw_name` 31e8bc8d…cb4c → e09541ee…549f · `case_and_punctuation_variant`
ac25d5c6…260b → 04e9132d…bd44 · `protein_taxonomy_id` 19b97b85…5631 → 65df837b…be7e · `protein_species_id`
7c34b165…6d4d → 90d7763c…9b4f · `protein_species_ref_ids` 7c1e3de6…c0e3 → 0022b449…f431 · `state_taxonomy_id`
e6d15352…0122 → 64d40b43…6c19 · `species_id_as_float_text` 2db9f1a9…4fa6 → 70eaa1f6…20b0 ·
`species_id_non_integral_float` 6440c6d6…03b6 → 22a12080…30c3 · `species_id_non_integral_text` f121fa42…9606 →
c3ca9997…261f · `create_default_name` b484bdf0…a66c → 0ca5c0c3…0cc8 · `first_row` b8b48779…c20f → 22a5965a…4efa ·
`requested_organism` d27c947f…1c48 → f7dc73d4…01e3. Full 64-hex values: `24-pins-delta-base.log`, `25-pins-delta-tip.log`.

## 7. Test gates

Every pytest job: `pinned_pytest.py` with `PYTHONPATH=<tree>/src`, a fresh `--basetemp` (`C:/t/orch773/bt-<NN>`), a
committed pin verdict in `evidence/g11/pin/ORCH-773/`, `--heavy-lock ORCH-773`.

| gate | report | result |
|---|---|---|
| focused (new file) | `34` tip / `33` base | 14 passed / 9 failed + 5 passed (§ 5) |
| strict-quarantine suites (`test_strict_quarantine{,_contract_alignment,_lineage,_locks_and_scope,_release_seam,_versioning}.py`, `test_orch766…`, `test_orch769b…`, `test_orch759c…`) | `38` | **234 passed** |
| Chunk E `test_strict_quarantine_real_artifact_replay.py` + `test_orch768_replay_corpus.py` | `39` | **184 passed** (Chunk E 175, no pin moved; ORCH-768 9) |
| SMOKE (22 files) | `40` | **511 passed** (expected 511) |
| `tests/test_eval_v2_*.py` (6 files) | `41` | 89 passed, 1 xfailed |
| `test_streamlit_quarantine_boundary.py` alone | `42` | 24 passed |
| `test_c052_prefreeze_report_at_the_streamlit_seams.py` alone | `43` | 13 passed |
| `test_orch769c_empty_graph_render.py` alone | `44` | 3 passed |
| `test_c055_rag_loop_wiring.py` alone (extra: it drives the RAG round's quarantine) | `45` | 14 passed |
| `test_c030a_object_sharing_at_the_freeze_seam.py` alone (extra: freeze seam) | `46` | 10 passed |
| affected sweep, 30 files: every non-AppTest test file that names `strict_quarantine`, `quarantine_and_close`, `freeze_canonical_payload`, `run_quarantine_boundary`, `write_quarantine_artifacts`, `_build_registry`, `entity_type_overlap` or `_prune_entities` (Chunk E run separately above), plus `test_stage3_gate_report_lifecycle.py` | `47` | **679 passed, 8 skipped** |
| the 8 skips | `48` (`-rs`) | `test_strict_failure_replay.py:234` (2) and `:249` (6), the known data-driven skips (ORCH-769b `18`) |

No test is red at tip, so no known-failure comparison against base is needed. Chunk D was not run (the orchestrator
runs it at integration).

## 8. LLM input

* **Callers of what changed.** `quarantine_and_close` is called by `streamlit_app.run_quarantine_boundary` (from
  `freeze_canonical_payload` and from the RAG round's `_classification` adapter) and by `run_pwml_export` (only when
  no stored decision matches). `_entity_representability`, `_referenced_entity_norms` and the new functions are
  module-private (grep: no caller outside `strict_quarantine.py`).
* **Pre-freeze LLM stages read nothing the quarantine writes.** The post-pipeline audit loop, gap resolver and
  curator run before Stage 6 and the quarantine (ORCH-768-RESULT § 1). In the RAG loop, `_classification` writes
  `channel["classification"]` and `channel["coverage"]` and returns the graph unchanged; the only channel keys passed
  to the LLM-backed `maybe_run_rag` are `qa_graph`, `gate` and `mapping` (`rag_round_reports`, `streamlit_app.py:993`);
  grep finds no other reader of `channel["classification"]`. So no pre-freeze LLM input changes.
* **No omission text enters any prompt.** The modules that read `quarantine_report` / `removed_entity_report`
  (`reaction_evidence`, `reaction_evidence_view`, `batch/driver`, `batch/runner`, `bench/*`, `release_status`) and
  the modules that call an LLM (`curation/*`, `llm/client`, `pipeline/{pipeline,preprocessor,localized_repair,
  extraction_ladder,deadline}`, `rag/extract`, `stoich/agent`, `bench/eval_v2/repro`) do not overlap.
* **Post-freeze, stated so it is not mistaken for silence:** the SBML overwatch (`run_sbml_overwatch` reads the
  frozen payload file) and the app's refinement "Submit Changes to AI" consume the frozen graph, which on a leg with
  an omission is the reduced graph. That is the frozen payload itself, exactly as it already is for every process the
  quarantine removed before this lane; neither reads the quarantine artifacts (§ 12 item 8).

## 9. Findings

1. **F-1. 16 of the 25 base failures were one row sinking a pathway** (§ 1, `02`, `04`): 4 quarantine refusals, 4
   final-gate errors, 2 Stage-8, 4 required-field, 2 IR. All 16 now export `review_required`.
2. **F-2. The quarantine registry's first-wins resolution and the final gate's name-only degree test disagree**
   (KI-013 "open"): on ORCH-759 enterobactin the actor `enterobactin synthase complex` resolves, in the quarantine
   registry, to the PROTEIN `enterobactin synthase` (proteins register before complexes and the protein carries that
   synonym), while the gate sees the protein attached to nothing. Removing the row (the gate's own complaint) makes
   the actor resolve to the complex; nothing is stranded, gold stays 5/5 (`07`, `08`).
3. **F-3. An unexportable row can be kept alive by a synonym collision alone** (KI-018): on ORCH-767 Wnt/Ca2+ the
   only "keeper" of `phosphorylated CAMKII` is an interaction whose `calcineurin B` resolves to the real
   `calcineurin B` row. Resolution-based dependency removes the row and keeps that interaction (`07`).
4. **F-4. ORCH-767 Wnt/Ca2+ had a second per-row failure behind its refusal**: two transporters without
   `entity_type` (Stage 8). The round order surfaces it only after the refusal is resolved (`04`, `37`).
5. **F-5. The IR's protein/complex verdict does not depend on the export's pre-freeze seam** on any corpus leg
   (81/81, `15`).
6. **F-6. The test suites' toy payloads are not export-shaped** (actors without `entity_type`, bare protein enzymes,
   compounds without a `db_status` verdict). A pass that omitted on every per-row error failed 43 tests in the
   strict-quarantine suites (`17`): 31 version pins and 12 behavioural (contract alignment 2, release seam 4,
   versioning 1, ORCH-769b preservation 4, ORCH-759c r1 1). With "keep an omission only if it buys a PWML" the 12
   are gone and only the 31 version pins remain (`19`), moved in § 6.
7. **F-7. `QUARANTINE_POLICY_VERSION` collides with C-114's `test_d088`**, which pins the live constant to the
   archived legs' version; any bump the module's own rule requires moves it (§ 12 item 2).
8. **F-8. One base-passing leg is `release_ready` in the quarantine record** (ORCH-759p menaquinone: the PWML is
   already `pathway.review_required.pwml` through a driver cap; its only removal is an unreferenced species row,
   D-111). The cap is applied to this pass's omissions only, so that record does not move (`05`).

## 10. Failures, dead ends, wrong theories

* **Wrong theory: "omit the keepers by the closure's norm test".** The closure keeps a protein whose name or synonym
  equals a referenced name or a component name. Using that as "who depends on it" would have omitted ORCH-759
  enterobactin's two EntF reactions (gold 5/5 → 4/5, a `compare` REJECT on final recall) and ORCH-767 Wnt/Ca2+'s
  calcineurin interaction. Replaced by registry resolution of each reference and each component (`07`, F-2, F-3).
* **Considered and rejected: drop a refused component row alone.** For a declared complex the IR silently drops an
  unresolvable component, so the complex would export with fewer subunits: a silent edit. The complex leaves whole.
* **Considered and rejected: pass `entity_type_overlap` through by omitting the keepers of both rows.** Wider than the
  gate (it accepts the pathway once either row is gone) and not decidable which row is real; left as a refusal.
* **First design omitted on every per-row error** (dev runs `10`, `13`, `17`): corpus 72/91 and ACCEPT, but 12
  behavioural test failures on toy payloads besides the version pins (F-6). The rollback rule (§ 2 item 4) kept the
  corpus result byte for byte (`20` = `10`) and removed all 12 (`19`).
* **First loop design wrapped the closure loop in a second loop**, which needed the whole loop body re-indented; the
  round now runs inside the existing loop on its settled iteration and reuses that iteration's record.
* **Incidents (rule violations, both harmless, both recorded):**
  1. While locating Git's `bash.exe` for the wrapper, a `python3 - <<'EOF'` with an **empty** heredoc ran in the
     shell (rule 4). It executed an empty script and exited at once; the orphan sweep right after shows no python
     process but the VS Code language servers.
  2. The first `compare` (dev, base vs `tip1.json`) ran **without the wrapper** (rule 1): it reads two JSON files and
     prints, ~1 s. Every later `compare` ran wrapped (`21`, `36`); the sweep after it found no orphan.
* **Infrastructure:** a wrapped `bash keepers.sh` resolved `bash` to WSL's and failed (`06`, exit nonzero before any
  probe ran); rerun with Git's `bash.exe` (`07`).

## 11. Orphan sweeps

After every job: `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`. Only the VS Code isort language servers
(`lsp_server.py`) were ever present; **no process was killed**. The heavy lock was free (`C:/t/heavylock` absent)
before and after every job; no exit 95 occurred.

## 12. Not settled (for the product owner)

Items 1, 2 and 10 were ruled after phase 1 (§ R1.1): item 1 by D-114 AMENDMENT 1, item 2 (keep the bump) and item 10
(cap on any omission) by the orchestrator. The open items for round 1 are in § R1.7.

1. **Whole-process omission vs dropping only the failing actor.** Today: 12 processes leave because one actor fails
   (§ 4.2). Options: (a) keep (this lane); (b) drop the actor, keep the process, flag it: restores 12 processes in 8
   legs (755 enterobactin back to gold 5/5) at the price of exporting uncatalysed reactions and transports without a
   transporter, and an edit of a surviving row's actor list beyond D-113 § 1's case. Needs a D-entry.
2. **`QUARANTINE_POLICY_VERSION`.** Bumped per the module's rule and the lane brief; it moves 28 ORCH-766 digests, two
   decision hashes and C-114's `test_d088` pin (§ 6), and every stored decision id. Options: (a) keep the bump;
   (b) revert it (ORCH-769b's choice), leaving the module comment false for a second verdict-changing lane;
   (c) amend the rule to "bump at integration milestones".
3. **`entity_type_overlap` stays a refusal.** Options: (a) keep; (b) omit the keepers of every row sharing the name
   (wider than the gate, see § 10); (c) a rule that names which declaration wins (an upstream question, KI-011).
4. **"An omission is kept only when it buys a PWML."** Chosen so a pathway refused anyway keeps its original refusal
   and so toy payloads do not move. The alternative (commit omissions even when a whole-pathway refusal remains)
   records more in the quarantine report of a leg that still writes nothing; it costs 12 behavioural test moves on toy
   payloads and changes no corpus leg.
5. **The IR check skips the export's pre-freeze seam** (DB access). Equal on 81/81 legs for the rows it maps. The
   alternative runs `run_prefreeze_resolution` on the copy, adding PathBank DB traffic inside the quarantine (also in
   every RAG round).
6. **D-113 § 1's reading**: `generation_reason == complex_named_source_entity_wrapper`, every actor resolving to the
   wrapper marked `inferred`, and an `extracted` enzyme kept. rv 08-21_2057 PMC12452463's wrapper
   (`complex_inferred_from_evidence`, actor `extracted`, no other catalyst) is therefore a process omission. Confirm.
7. **KI-023 rows (non-canonical actors) are converted per row.** On ORCH-755 sour taste the cause is a normalization
   exception that left the payload un-normalized (KI-024), and 3 of 7 processes leave. Options: keep, or treat
   `actor_schema_not_canonical` as whole-pathway (sour taste and rv 09-02_2052 PMC12452463 go back to no PWML;
   767 Wnt/Ca2+ too).
8. **Post-freeze LLM consumers** (SBML overwatch, refinement) receive the reduced frozen graph on legs with an
   omission (§ 8). Confirm that this is the intended reading of "no LLM-input change".
9. **The required-field gate's whole-pathway check inside the quarantine uses the batch/app metadata defaults**
   (`Generated Pathway`, `Metabolic`, 3200 × 1400), as `run_pwml_export` sets them. A user who clears the name gets
   the omissions and then the export's own `pathway_missing_name` refusal.
10. **Release cap scope.** Only this pass's omissions cap `release_ready`. D-114 says "a PWML with any omission";
    pre-existing admission quarantines on a `release_ready` record (none in the corpus except an unreferenced species
    removal, F-8) are untouched. Options: cap on every quarantined process too (moves records, not PWML bytes).
11. **ORCH-772 rendering.** The per-row detail (gate, cause, slot, D-113 actor removal) is in `detail` of the existing
    rows; the current text renders state, reason and participant only. ORCH-772 may want `detail.omission`.

## R1. Round 1 (D-114 AMENDMENT 1, the release-cap reading, review r0 notes)

Review r0 (`C:/t/rev773/REVIEW-r0.md`, reviewer reports `ORCH-773/49`–`66` in its own worktree): **APPROVE WITH NOTES**, 0
blocking, 19 notes. The product owner ruled on § 12 item 1 as **D-114 AMENDMENT 1** (DECISIONS.md, sprint tip
`88fdbabd`). Round-1 code: `1a0fc159`; this file and evidence: the docs commit after it.

### R1.1 Rulings recorded

* **D-114 AMENDMENT 1** (product owner): (1) a reaction whose enzyme fails is omitted whole (PathWhiz
  `reaction.rb:93-94` `check_spontaneous`); (2) a transport whose transporter fails keeps the transport and loses only
  that transporter (`transport.rb:29-33`: no transporter requirement); a transport failing for any other reason is
  omitted whole; (3) an interaction with a bad endpoint is omitted whole.
* **Release cap, orchestrator reading of D-114 § 2:** any omitted process or dropped actor in the frozen record caps
  the release at `review_required`, including processes quarantined before this lane (§ 12 item 10). An entity-only
  removal (unreferenced species, degree-0 protein) does not cap it on its own.
* **`QUARANTINE_POLICY_VERSION` bump stays** (orchestrator ruling: it follows the module's own rule; § 12 item 2).
* **ORCH-755 enterobactin stays at gold 4/5** under amendment § 1 (its `EntB isochorismatase reaction` loses its only
  enzyme complex, whose subunit EntB has no identity, KI-017); the product owner accepted this.

### R1.2 What changed

`src/t2pw/pipeline/strict_quarantine.py`:

1. **Transporter drop** (`_apply_per_row_failures`). A gate error naming a transport's `transporters/<j>` (the
   Stage-8 `actor_schema_not_canonical`, the final gate's "Unknown transporter reference", a required-field error on
   that slot) drops that transporter and keeps the transport, when every error naming that transport names a
   transporter; otherwise the transport is omitted whole, as before. When a refused entity is reached by a transport
   only through `transporters`, those transporters are dropped and the transport stays. Each drop is recorded the
   D-113 § 1 way: a `removed_entities[]` row (`bucket` `transporters`, `name` the transporter, `reason` the gate's
   exact code) with `detail.actor_removed_from` = `{process, process_name, slots: [transporters], decision: "D-114
   AMENDMENT 1 s2"}` and the dropped actor row verbatim in `detail.actor` (outside the bounded `scrub_detail`, like
   `original_process`). A transport left with no transporter meets every gate again on the next round; if one refused
   it, the transport would be omitted whole by the ordinary process rule (no gate does, § R1.3).
2. **Release cap** (`omitted_from_pwml:<p>_processes+<a>_actors`, `OMITTED_FROM_PWML_REASON`): `<p>` counts every
   quarantined process of the frozen record, `<a>` every actor dropped from a kept process (D-113 § 1 wrapper,
   transporters). Recorded on `review_reasons` and `release.reasons` and capping `release_ready`, only when the graph
   is not refused (a refused graph has no PWML to cap, so its records do not move) and never in research mode. The
   phase-1 reason `per_row_omissions_removed_from_pwml` is gone.
3. **N1:** a complex that leaves because a component it resolves was omitted now carries
   `protein_complex_component_omitted:<component>:<reason>` (it said `protein_complex_component_unrepresentable:…`,
   a code `_entity_representability` never emits for a declared complex).
4. **N4:** the generation reason and the final gate's two "attached to nothing" messages are constants in the gate's
   own module, `process_normalizer.py` (`COMPLEX_NAMED_SOURCE_WRAPPER_REASON`, `DEGREE_ZERO_PROTEIN_MESSAGE`,
   `ISOLATED_LOCATED_PROTEIN_MESSAGE`), used by the gate's f-strings and by the quarantine. The messages are
   byte-identical: the corpus' final-gate errors and every gate test are unchanged. This is the only gate-module edit,
   asked for by the orchestrator.
5. **N6:** the rollback comment no longer cites D-113 § 3.

### R1.3 The six transports (each in case (i): kept, transporter dropped, exported)

| leg | transport | transporter dropped | reason → KI | kept in canonical / PWML IR | transporters left |
|---|---|---|---|---|---|
| ORCH-755 sour taste | `OTOP1-mediated H+ transport` | OTOP1 | `actor_schema_not_canonical` → KI-023 | yes / yes | none |
| ORCH-755 sour taste | `H+ permeation through OTOP1` | OTOP1 | same | yes / yes | none |
| ORCH-755 sour taste | `OTOP1-mediated H+ permeation` | OTOP1 | same | yes / yes | none |
| ORCH-767 Wnt/Ca2+ | `IP3-induced Ca2+ release` | inositol-1,4,5-trisphosphate receptor (InsP3R) | same | yes / yes | none |
| ORCH-767 Wnt/Ca2+ | `SERCA pumps Ca2+ into ER` | InsP3R | same | yes / yes | none |
| rv 09-02_2052 PMC12452463 | `ferri-enterobactin import via FepA` | FepA | same | yes / yes | none |

None landed in case (ii) (a gate refusing a transport with no transporter) or case (iii) (failing for another reason).
Source: `transport_cases.py`, first section of `63-r1-analysis.log`. OTOP1 and FepA, referenced by nothing else, then
fall to the ordinary closure prune (`degree_zero_after_quarantine`); InsP3R stays, because other Wnt/Ca2+ processes
still reference it.

### R1.4 Corpus

Base `01-corpus-base` (unchanged) vs round-1 tip `60-r1-corpus-tip`, both under the wrapper, `PATHBANK_DB_*` unset.
Records: `evidence/orch773/corpus-r1-tip.{json,txt}`.

**Base → round 1** (`61-r1-corpus-compare`, exit 0):

```
pass rate base 56/91 -> tip 72/91
fixed: [the same 16 legs as phase 1]
passing legs turned failing: none
gold recall lost: none
still failing, at a different step (information): none
target ORCH-767/enterobactin-strict/2026-10-23_2224: FIXED
D-112 § 2: ACCEPT
```

* **56/56 base-passing legs write a byte-identical PWML** (name and bytes; `63-r1-analysis`, base → round-1 section).
* Gold unchanged on every base row; ORCH-767 enterobactin PWML **5/5**; ORCH-755 enterobactin **4/5** (accepted, § R1.1).
* No `release_ready` record moved. The four legs that move `diagnostic_only` → `review_required` are four of the 16
  fixed legs. 17 frozen records carry `omitted_from_pwml:…`: 15 fixed legs (not ORCH-759 enterobactin, whose only
  removal is a protein) and two base-passing legs already `review_required` (ORCH-767 chlorophyll, 5 processes;
  rv 08-22_2017 PMC12856317, 1 process), whose PWML bytes and names are unchanged.

**The transport delta, separately: phase 1 → round 1** (`62-r1-corpus-compare-phase1`; `63-r1-analysis`, phase-1
section):

* the pass rate stays 72/91; passing legs turned failing: none; gold recall lost: none. (`compare` prints REJECT
  because its rule needs the pass rate to rise or a target leg to be fixed, and no new leg can be fixed by keeping a
  transport inside a leg that already writes a PWML; exit 1 is that rule, not a regression.)
* **69 of the 72 phase-1 PWMLs are byte-identical; the 3 that differ are exactly the three transporter legs**
  (ORCH-755 sour taste, ORCH-767 Wnt/Ca2+, rv 09-02_2052 PMC12452463), each now carrying its transports (§ R1.3).

### R1.5 Counts after round 1

| | phase 1 | round 1 |
|---|---|---|
| processes omitted by the pass | 18 | **12**: 6 reactions via one enzyme (actor), 6 interactions via an endpoint (process) |
| actors dropped from a kept process | 1 (D-113 § 1) | **7**: 6 transporters (D-114 AMENDMENT 1 § 2), 1 wrapper (D-113 § 1) |
| entities removed by the pass | 15 | 15 |
| legs fixed | 16 | 16 |

### R1.6 Tests (round 1)

| gate | report | result |
|---|---|---|
| focused `test_orch773_omit_per_row.py`, base tree (`C:/t/orch773/basetree`, `3ef3fa9c` + the new test and fixtures) | `58` | **10 failed, 6 passed**: the 8 phase-1 G9 tests on their refusal values; the new `test_g9_a_process_quarantined_at_admission_caps_a_release_ready_pathway` on `'release_ready' == 'review_required'`; `test_new_release_cap` on its in-test import (NEW). 6 preservation pass |
| same, tip | `59` | **16 passed** |
| strict-quarantine suites + ORCH-766/769b/759c + Chunk E + ORCH-768 + `test_d088` | `64` | 429 passed |
| SMOKE (22 files) | `65` | **511 passed** |
| affected sweep, 30 files, before the fixture move | `66` | 9 failed (all `test_strict_failure_replay.py`, the release-cap pins below), 672 passed, 8 skipped |
| affected sweep, after the fixture move | `73` | **681 passed, 8 skipped** (phase 1: 679 + the 2 new focused tests) |
| Chunk E alone, after the fixture move (it reads the same fixture's payloads) | `74` | **175 passed** (unchanged) |
| `test_strict_failure_replay.py -rs` | `75` | 40 passed, 8 skipped: `:234` (2) and `:249` (6), the known data-driven skips |
| `tests/test_eval_v2_*.py` | `67` | 89 passed, 1 xfailed |
| alone: streamlit boundary / c052 / 769c / c055 / c030a | `68`–`72` | 24 / 13 / 3 / 14 / 10 passed |

**Pins moved in round 1 (merge rule 4)**, all in `tests/fixtures/strict_failures/cases.json`, all by the release cap
(orchestrator reading of D-114 § 2), each case quarantining exactly one process (two for the last):

| case | `release_status` | `strict_acceptance_eligible` | `review_reasons` |
|---|---|---|---|
| `reaction_inflation_unresolvable_participant` | `release_ready` → `review_required` | true → false | `[]` → `["omitted_from_pwml:1_processes+0_actors"]` |
| `colon_split_metabolite_as_protein_complex` | same | same | same |
| `enzyme_without_external_identity` | same | same | same |
| `reaction_coupled_transport_is_not_exportable` | same | same | same |
| `only_unrelated_reactions_survive` | unchanged (`review_required`) | unchanged (false) | + `"omitted_from_pwml:2_processes+0_actors"` after the two `minimum_core:` reasons |

Each moved case carries a `release_status_note` with the old value. `unknown_backed_functional_complex`,
`degree_zero_mapped_protein` (an entity-only removal) and `primary_transport_and_interaction_shapes_survive` stay
`release_ready`, so `test_the_fixture_covers_both_verdicts` still sees all three classifications. Nothing else moved:
the 429 of `64` and SMOKE needed no change.

### R1.7 Not settled (round 1)

1. **Admission-time transport quarantines.** Amendment § 2 is applied to this pass's omissions. Admission
   (`_actor_verdict`: `unnamed_/unresolved_/non_protein_actor_in_transporters`, and a transporter whose entity is
   unrepresentable) still quarantines the whole transport, as before this lane; extending the amendment there would
   change base-passing PWMLs. Priced on the corpus legs' own recorded (live-run) quarantine reports
   (`admission_transporter_quarantines.py` → `76`): **4 transports in 3 legs** were quarantined at admission for their
   transporter, all `protein_missing_external_identity`: ORCH-755 sour taste (`voltage-gated Ca++ channel-mediated
   Ca++ influx`, transporter `voltage-gated Ca ++ channels`), ORCH-767 sour taste v2 (`K+ efflux via Kir2.1`, Kir2.1),
   ORCH-767 Wnt/Ca2+ (`SOCE via Orai`, `Store-operated Ca2+ entry`, Orai). All three legs write a PWML at tip. The
   corpus cannot measure the change: a canonical replay starts from a payload the live quarantine already reduced.
   Options: (a) keep (this lane); (b) apply amendment § 2 at admission too (`_actor_verdict`'s transporter
   branch), measurable only live, and it changes the PWML of a base-passing leg (ORCH-767 sour taste v2).
2. **KI appendix row** for the new reason `protein_complex_component_omitted:<component>:<reason>` (prefix; KI-015,
   then label by `<reason>`, as for `protein_complex_component_unrepresentable:`). Until it is added, ORCH-772's
   mapping labels those 8 rows `unclassified` (`corpus_delta.py` KI census). The catalogue is outside this lane.
3. **`removed_entities` bucket `transporters`** names a process slot, not an entity bucket; the transport it was
   dropped from is in `detail.actor_removed_from`. ORCH-772 renders the latter in phase 2.
4. Reviewer notes F-2 (the cascade also runs for a degree-0 protein), F-3 (the export's re-quarantine branch) and F-5
   (two calls outside the try) were not in this round's charge; unchanged.

### R1.8 Orphan sweeps (round 1)

After every job, `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`. No process of this lane survived any job
and **none was killed**. Two kinds of foreign process appeared and were left alone, because CLAUDE.md limits cleanup
to the PIDs a job created:

* PIDs 195392 and 189168, `.venv\Scripts\python.exe -` (stdin), started 14:49 by another session's shell in
  `.claude/worktrees/rev769d-tip` (`... cat orch769d/run_when_free.sh; python - 2>/dev/null; ...`), still alive at every
  later sweep: a hung `python -` of that reviewer, not of this lane. Reported to the orchestrator.
* The ORCH-769d reviewer's live wrapped job (`bounded_run.py --heavy-lock ORCH-769d-rev`, PIDs 216252, 226016,
  227520, 227532), owned by its wrapper.

The heavy lock was held by `ORCH-772-rev` (twice) and `ORCH-769d-rev` during this round: one of this lane's jobs was
refused with exit 95 before its child started (`54-r1-transport-cases-dev`, no report promoted, its `.staging`
reservation left uncommitted); every later job waited for `C:/t/heavylock/holder.json` to disappear
(`C:/t/orch773/queue.sh`: each job still runs through `bounded_run.py --heavy-lock ORCH-773`).

**Incident (rule 1).** While listing the readers of the strict-failure fixture, one shell command also ran
`python3 -c "print(1)"` outside the wrapper, by mistake (no stdin pipe; it printed and exited at once; the sweep after
it shows no process of this lane).

## 13. LEDGER row (proposed; this lane does not edit LEDGER.md)

```
| ORCH-773 | D-114 s2 per-row omission pre-freeze + D-114 AMENDMENT 1 (transporter drop) + release cap on any omission (strict_quarantine) | orch773/omit-per-row: phase 1 code 0971487b, round 1 code 1a0fc159 + docs commits | ROUND 1 IMPLEMENTED, awaiting round-1 review | corpus 56/91 -> 72/91 ACCEPT; 56/56 base PWML byte-identical; 6/6 transports kept (transporter dropped), 69/72 phase-1 PWMLs identical, 3 = the transporter legs; enterobactin 767 PWML 5/5, 755 enterobactin 4/5 (PO accepted); SMOKE 511; Chunk E 175; G11 ORCH-773/01-76 | ORCH-773-RESULT.md |
```

## 14. TEST_MATRIX delta (proposed; append at end of file)

```markdown
## ORCH-773: test obligations, appended end-of-file by rule · 2026-10-25

Lane `orch773/omit-per-row` (phase 1 `0971487b`, round 1 `1a0fc159`). Reports `evidence/g11/ORCH-773/01`–`76`.

**SMOKE stays 511** (`g11/ORCH-773/40`, round 1 `65`). Merge rule 10 unchanged. No chunk count moves.

### New focused selection

`tests/test_orch773_omit_per_row.py` (16, ~2 s), fixtures `tests/fixtures/orch773/` (8 corpus freeze inputs,
gzipped, sha256 asserted). In no chunk, not in SMOKE. G9: 9 `test_g9_*` red at base (round 1 `58`), 6 preservation
green there, `test_new_release_cap` NEW; 16 green at tip (`59`). Run it for any change to `strict_quarantine`'s
per-row omission section, `_entity_representability`, `_referenced_entity_norms`, the release cap, or to the freeze
gate, the Stage-8 contract, the required-field gate or the IR checks it calls.

### Pinned tests moved (merge rule 4; exact deltas in ORCH-773-RESULT.md § 6)

Phase 1: `test_orch768_replay_corpus.py` (enterobactin verdict), `test_orch766_unreferenced_species.py` PINS (version
fields only), `test_orch769b_actor_protein_registry.py` and `test_orch759c_quarantine_bucket_closure.py`
(`decision_input_hash`), `test_strict_quarantine_lineage.py` (removed_entity_report schema 2),
`test_d088_coverage_diagnostics.py` (archived legs' policy version). Round 1: none outside the focused file.
Chunk E 175 unchanged (`39`, `64`).

### Known failures

None new. Affected sweep, 30 files: 679 passed, 8 skipped in phase 1 (`47`); 681 passed, 8 skipped in round 1 (`73`).
```

## 15. Evidence index (`docs/pwml_recovery_sprint/evidence/`)

| artifact | what |
|---|---|
| `orch773/corpus-base.{json,txt}`, `corpus-tip.{json,txt}` | the corpus records (`01`, `35`) |
| `orch773/failing_legs.py` → `02` | every failing leg and its failing step at base |
| `orch773/capture_freeze_inputs.py` → `03` | each leg's exact freeze input (the fixtures' source) |
| `orch773/probe_gate_rows.py` → `04` | every downstream gate error with its row, on all 81 legs at base |
| `orch773/release_ready_legs.py` → `05` | the release_ready leg (F-8) |
| `orch773/probe_keepers.py` → `06` (WSL bash failure), `07` | who keeps each failing entity alive, and how it resolves |
| `orch773/dump_rows.py` → `08` | the 759 enterobactin wrapper and protein rows |
| `orch773/make_fixtures.py` → `12` | the fixtures and their sha256 |
| `orch773/ir_prefreeze_equivalence.py` → `15` | IR verdict with and without the pre-freeze seam, 81/81 |
| `orch773/probe_toy_omissions.py` → `18` | what the first design did to the test toys |
| `orch773/pins_delta.py` → `24`, `25` | the ORCH-766 pins move only by the version fields |
| `orch773/corpus_delta.py` → `11`, `16`, `37` | per-leg verdicts, PWML bytes, gold, omissions with KI ids, cause counts, every record difference |
| `g11/ORCH-773/10`, `13`, `14`, `17`, `19`–`23`, `26`–`32` | development runs (superseded by the final runs, kept on the record) |
| `g11/ORCH-773/33`–`48` | final runs (§ 5, § 7) |
| `g11/pin/ORCH-773/` | pin verdicts of every pytest run |
| `orch773/corpus-r1-tip.{json,txt}` | the round-1 corpus record (`60`) |
| `orch773/transport_cases.py`, `corpus_delta.py` (round-1 sections) → `63` | where each of the 6 transports lands; base → round 1 and phase 1 → round 1 deltas, release moves, cap reasons |
| `orch773/admission_transporter_quarantines.py` → `76` | the price of applying amendment § 2 at admission (§ R1.7 item 1) |
| `g11/ORCH-773/49`–`57` | round-1 development runs (`54`: refused with exit 95 by the heavy lock, child never started) |
| `g11/ORCH-773/58`–`76` | round-1 final runs (§ R1.4, § R1.6); `66` is the sweep before the fixture move |
