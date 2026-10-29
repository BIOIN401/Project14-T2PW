# ORCH-792 RESULT: protein rows no longer vanish unrecorded at Stage 6

**Branch:** `orch792/fix` (base `f3a80e09`). Shared with ORCH-793 (same builder, same branch).
**Merge rule 5 waived** by the product owner (D-119 § 4, `f0452256`): no reviewer round was run.
**Scope:** ORCH-789 C3, C6, D2, D4. Every Python job ran through `C:/t/orch79x/run.sh` / `gate.sh` (bounded wrapper, heavy lock); every report is under `evidence/g11/ORCH-792/`.

## 1. Cause (verified, not inferred)

The deletion site was pinned by an offline Stage-2 → normalise → Stage-6 replay of each leg's `merged_payload.json`, with every function of `map_ids` and `process_normalizer` wrapped by a removal tracer (ORCH-789 § 8 Q1 option a). Script: `evidence/orch792_trace.py`. Output: `evidence/orch792_trace.out.txt`. Leg replay base/tip outputs: `evidence/orch792_replay_notaxa.txt`. G11: `01-trace-base-wntca.json`, `02-trace-base-chl-pcp-mena.json`.

| row(s) | removed by | recorded? |
|---|---|---|
| chl GBP, CHL27, LPOR, ferrochelatase, Chl synthase, EXECUTER (also HEMA1, Mg-chelatase); wnt_ca PKC; wnt_pcp Frizzled, Wnt, Smurf, DIA1, Profilin, MRLC, G-actin (also CapZIP) | `map_ids._rewrite_reaction_protein_enzymes_to_complexes` → `_rewrite_rows`, the KI-007 drop ("no UniProt or DrugBank identifier; other enzymes remain"): the row was deleted from `entities.proteins`, and the name was then **skipped in every later actor list**, including lists where it was the only enzyme | no (mapping-report action only) |
| wnt_pcp `Fz receptor`, `Invs` | `process_normalizer.canonicalize_same_as_aliases` (alias merged into its canonical row; references rewritten) | normalization report; a merge, not a loss |
| wnt_ca TCF (D4) | nested `map_payload._prune_invalid_complex_components`: `_still_needed` read only complex components | `quarantined_proteins` (`unresolved_complex_component_no_longer_referenced`), but the reason was wrong |

Not the deleters (each ruled out by the trace): `drop_process_orphan_proteins`, `prune_disconnected_proteins`, structural cleanup's quarantine branch (all write `quarantined_proteins`), the audit (not run in the replay, and the rows still vanish), `_apply_family_complexes` / `_apply_pathbank_unknown_*_fallback` (conversions: the row becomes a complex of the same name).

**C6, ORAI2.** `run_strict_post_normalization_gates` raises "Located protein is isolated in connectivity graph" for a protein with a location row but no edge. The degree-0 check right below it already exempts complex components; the isolation check did not.

**D2, MenA.** The audit's "UniProt identifier for MenA based on function" (`audit_iteration_summary.json`) is **LLM-side**. No deterministic step accepted it: MenA reaches neither the final payload nor a record. Its loss is the KI-007 drop above: `DHNA prenylation` shipped with no enzyme because MenA was skipped in every later actor list.

## 2. Change

| file | function | change |
|---|---|---|
| `src/t2pw/mapping/map_ids.py` | `_rewrite_reaction_protein_enzymes_to_complexes._rewrite_rows` (not the interaction-endpoint loop) | A no-id enzyme beside a resolved one leaves **that actor list only**. The reaction keeps the enzyme that resolved (D-115 § 3), the **row stays**, and `quarantined_proteins` gets `{reason: reaction_enzyme_protein_dropped_no_external_id, action: dropped_from_reaction_actor_list, process, json_pointer, row_kept: true}`. The row is removed only when a complex in the same list carries the protein as a component (the ALAS2 fold); that removal is now recorded too (`action: folded_into_wrapper_complex`, `functional_replacement`). A protein folded in one list takes its wrapper in another list, instead of vanishing there. |
| `src/t2pw/mapping/map_ids.py` | nested `map_payload._prune_invalid_complex_components` | `_still_needed` also counts every name a process references (`_strings(processes)`). |
| `src/t2pw/pipeline/process_normalizer.py` | `run_strict_post_normalization_gates` | `_complex_component_norms` is computed once, before both checks. The isolation check skips a complex component, as the degree-0 check already did. |

A row kept this way reaches strict quarantine, where it is either drawn (Unknown-backed complex, family complex, or a Phase-2 id) or omitted with its own reason (`protein_missing_external_identity`, KI-001, or `degree_zero_after_quarantine`) in `removed_entity_report`, which the text record renders.

## 3. Proof

| proof | result | report |
|---|---|---|
| G9 base arm (`tests/test_orch792_no_silent_drops.py` copied into `C:/t/orch792/base`, then removed) | **4 G9 failed, 1 preservation passed**. Failures are behavioural: `'GBP' in ['HEMA1']`, `[] == ['CPO']`, `'TCF' in ['HEMA1','NLK']`, the ORAI2 isolation message present | `evidence/g11/ORCH-792/20-g9-base.json` |
| G9 tip arm | 5 passed (same file) | `23-affected-tip-2.json` (in the affected batch) |
| affected tests at tip: 43 files importing the changed functions or the isolation gate (the list is in `C:/t/orch792/affected.txt` and in the report's command line) | **1133 passed, 8 skipped, 0 failed** | `23-affected-tip-2.json`. Re-run on the final tip after ORCH-793's NARF correction: also 1133 / 8 / 0, `evidence/g11/ORCH-793/21-affected-tip-3-r1.json` |
| AppTest `tests/test_streamlit_quarantine_boundary.py`, run alone (it calls the gate) | 24 passed | `25-apptest-quarantine-boundary-2.json` |
| leg replay, base vs tip (`evidence/orch792_replay.py --no-taxa`: Stage 2 → normalise → Stage 6 → `quarantine_and_close`, run cache as the recorded answers) | see § 4 | base `28-final-base-notaxa-r1.json`, tip `29-final-tip-notaxa-r1.json` |

Earlier affected run `21-affected-tip-1.json` had 6 failures. Two were my rule colliding with `test_c122` (fixed in ORCH-793). Two were a sibling-slot widening I then reverted (§ 6). One is the pin in § 5. One is ORCH-793's pin.

## 4. Leg replay: rows recovered (processes accepted by strict quarantine)

`--no-taxa` mode: the species rows carry no taxonomy id, so offline nothing resolves by UniProt. That is the shape in which the vanished rows were lost live. Logs: `C:/t/orch79x/logs/ORCH-792-28-*.log` and `-29-*.log`.

| leg | base | tip | what moved |
|---|---|---|---|
| wnt_ca (ORCH-785) | 14 | **19 (+5)** | PKC kept → family complex; `DAG/Ca2+ activates PKC` ×4 and `PKC activates Cdc42` drawn (base: `undeclared_entity_in_entity_*`) |
| chlorophyll (ORCH-785) | 32 | **34 (+2)** | `POR reaction` and `ferrochelatase reaction` drawn (LPOR and ferrochelatase take the D-070 Unknown-backed complex). GBP, CHL27, EXECUTER rows now omitted as KI-001 `protein_missing_external_identity`, not KI-015 `undeclared_entity` |
| wnt_pcp (ORCH-784) | 12 | 11 (−1) | `Profilin-facilitated ADP-ATP exchange`: base drew it with **no catalyst** (only `DAAM complex` activators: Profilin was skipped). PathWhiz drops such a reaction on import (D-115, `reaction.rb:90-96`), so base's row was a phantom. Tip omits it with KI-001. 23 more wnt_pcp omissions move from KI-015 `undeclared_entity` to KI-001 |
| menaquinone, enterobactin, sour_taste_v2 | 22 / 18 / 6 | 22 / 18 / 6 | none |

A second replay mode, with taxonomy ids so the run's recorded UniProt answers are read, is in `ORCH-793-RESULT.md` § 5 and `evidence/orch793_replay_taxa.txt`. That replay runs the combined tree: chl 81 → 84, menaquinone 23 → 26, enterobactin 4 → 7, and the rest unchanged.

The ORCH-789 target of 12 signed rows: **wnt_ca +5 recovered; chl +2 of 4 recovered (CHL27 ×2 now KI-001: no id); wnt_pcp 0 of 3 (DIA1 / MRLC / Profilin have no id: now KI-001, the true reason).** The rest of the gain needs the identity (C2 family list, ORCH-790).

## 5. Pins moved

- `tests/test_orch759e_rewrap_after_phase2.py::test_preservation_the_drop_rule_still_runs_before_phase2`: proteins `["HEMA1"]` → `["HEMA1", "CPO"]`, plus an asserted `quarantined_proteins` record `("CPO", "dropped_from_reaction_actor_list")`. Annotated in the test.

## 6. Failures, dead ends, wrong theories

1. **Keeping the actor at Stage 6 and letting strict quarantine drop it** (to give Phase 2 a chance: KI-007's own planned fix). The tip replay lost 3 chlorophyll reactions. Strict quarantine's admission (`_actor_verdict`) omits the **whole** reaction for a failing actor; D-115 § 3 was never implemented there. Stage 6's silent drop was the only place D-115 § 3 happened.
2. **Then implementing D-115 § 3 in strict quarantine** (`_actor_verdict` tolerance plus a `drop_entity` actor removal). This turned chlorophyll into a whole-pathway refusal (`unexportable_entity:5`), because `_per_row_failures` returns "blocked" whenever any entity-type overlap exists (`_entity_type_overlaps`), and chlorophyll has the tRNA overlap (C7). Reverted. That file is not in this lane's diff.
3. **Counting the sibling slot's catalysts as "other enzymes"** (enzymes ↔ modifiers). It broke ORCH-759e's LDHA/LDHB G9s (`21-affected-tip-1.json`): the actor was dropped before Phase 2 could resolve it. It also did not save Profilin, whose siblings are activators. Reverted.
4. Wrong theory: "the replay needs the audit's payload". The vanished rows vanish identically from `merged_payload.json`, without the audit or the gap rounds.
5. Infrastructure slip: one ad-hoc `python -` (stdin) was typed by mistake. It waited on stdin, was stopped (TaskStop), and the orphan sweep found no survivor (only VS Code language servers). It ran no job.

## 7. KI delta (catalogue not edited)

- **Appendix row proposed:** `reaction_enzyme_protein_dropped_no_external_id` (`quarantined_proteins`, actions `dropped_from_reaction_actor_list` / `folded_into_wrapper_complex`) → **KI-007**. Status → *partly fixed*: the row is kept and recorded; the actor still leaves its list before Phase 2 (§ 8 Q1).
- KI-015 → KI-001 reclassification: rows that showed `undeclared_entity_in_*` / `unresolved_actor_in_*` because of KI-007 now show `protein_missing_external_identity` (KI-001).
- **KI-013** (isolation): a complex component is no longer judged isolated.
- No new omission reason code.

## 8. Not settled (for the product owner)

1. **`quarantined_proteins` is not rendered in `reaction_evidence.txt`.** `app/reaction_evidence.py` reads `removed_entity_report.removed_entities` only. The new Stage-6 records reach `final_mapped.json` but not the text record. Options:
   - (a) the reaction_evidence owner renders `quarantined_proteins` (small reader change);
   - (b) strict quarantine copies them into `removed_entity_report`.

   Cost: (a) one reader lane; (b) touches the third builder's file.
2. **KI-007 remainder.** A dropped actor is gone before Phase 2 resolves its row, so a reaction can miss an enzyme Phase 2 found (`HEMA1 + CPO`). A proper fix needs D-115 § 3 in strict quarantine (§ 6 items 1-2), which first needs per-row omission to survive an entity-type overlap.
3. **A lone enzyme that is also an interaction endpoint** (Profilin, EntE) does not take the Unknown-backed complex: `_apply_pathbank_unknown_enzyme_fallback._has_disqualifying_reference` refuses it. That function is owned by another builder. D-115 § 1 reads as if it should take the complex.
4. **LPOR clash.** When `LPOR` becomes an Unknown-backed complex while a gene row `LPOR` exists, `LPOR Pchlide reduction` would hit `ambiguous_entity_type:nucleic_acids+protein_complexes` (seen in an intermediate replay). This is C7 / ORCH-794 territory.

## 9. LLM-side

- MenA's UniProt "based on function" (menaquinone audit): an LLM invention. Correctly refused downstream.

## 10. Proposed LEDGER row

`ORCH-792 | orch792/fix @ <tip> | done (rule 5 waived, D-119 § 4) | KI-007 drop keeps + records the row; TCF prune counts process refs; isolation gate exempts complex members | G9 base 4 red (g11/ORCH-792/20), affected 1133 pass (23), replay wnt_ca +5 chl +2 wnt_pcp −1 phantom (28/29) | open: quarantined_proteins not rendered; KI-007 remainder`
