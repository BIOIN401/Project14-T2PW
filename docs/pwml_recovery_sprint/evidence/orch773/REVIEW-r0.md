# ORCH-773 review, round 0 (independent reviewer)

Branch `orch773/omit-per-row`, code `0971487b`, records `f35de933`, base `3ef3fa9c`.
Reviewer worktrees: `.claude/worktrees/rev773-base` (detached 3ef3fa9c), `.claude/worktrees/rev773-tip` (detached 0971487b).
Scratch: `C:/t/rev773/`. Every Python job ran under `bounded_run.py --heavy-lock ORCH-773-rev`; every report shows
`FINAL SURVIVING COUNT : 0`; the heavy lock was free before every job (no exit 95); the orphan sweep after every job
found only the two VS Code isort language servers (PIDs 137436, 137476), nothing was killed.

## VERDICT: APPROVE WITH NOTES

No blocking finding under the fixed severity standard. 0 BLOCKING, 19 NOTES.

## Boundary check

* `git diff --stat 3ef3fa9c 0971487b -- src`: one file, `src/t2pw/pipeline/strict_quarantine.py` (+511 / -18).
* `src/t2pw/app/streamlit_app.py`: `git diff --quiet 3ef3fa9c f35de933 -- src/t2pw/app/streamlit_app.py` exit 0 (byte-identical).
* No gate module touched: `process_normalizer.py`, `stage_contracts.py`, `pwml/ir.py`, `pwml/writer.py`, `release_status.py`,
  `reaction_support.py`, `reaction_evidence.py` are not in the diff.
* Records commit `f35de933` touches no `src/` or `tests/` file (only `docs/pwml_recovery_sprint/`).
* Everything else in `0971487b`: tests (`tests/test_orch773_omit_per_row.py`, 6 pin moves) and `tests/fixtures/orch773/`.

## Gates are called, not copied (src diff read in full)

`_downstream_gate_errors` (strict_quarantine.py:2671) CALLS `run_strict_post_normalization_gates(..., enforce_all_proteins_connected=True)`,
`validate_post_normalization`, `validate_pre_export`, `build_pwml_ir`, `validate_pwml_ir`, `blocking_pwml_ir_errors`, in export
order, each on a deepcopy; the call shapes match `_validate_stage8_export_payload` (streamlit_app.py:4596-4609) and
`run_pwml_export` (metadata setdefault 4779-4788; IR blocking rule 4971-4973: every `validate_pwml_ir` error blocks, as in the pass).
`_per_row_failures` (2782) calls the module own `_entity_type_overlaps`, `_unexportable_entities`, `_entity_representability`.
Two refactors of the quarantine own predicates, both verbatim lifts (read line by line): `_resolve_component` (689, out of
`_entity_representability`) and `_process_references` (1801, out of `_referenced_entity_norms`; same set, `discard("")` kept).

## Tests and measurements I ran (my own output; reports under `rev773-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-773/`)

| # | report | what | result |
|---|---|---|---|
| 49 | `49-rev-corpus-base.json` | ORCH-768 corpus, base = `git archive 3ef3fa9c src reference/PW000001.pwml data/pathwhiz_id_db.json` (C:/t/rev773/base), PATHBANK_DB_* unset | base.json sha256 `5c5579bf...40b9` = the implementer stated base hash |
| 50 | `50-rev-corpus-tip.json` | same, `--src rev773-tip/src` | tip.json sha256 `ce612a2d...79f0` = the implementer stated tip hash |
| 51 | `51-rev-corpus-compare.json` | `compare base.json tip.json --target ORCH-767/enterobactin-strict/2026-10-23_2224` | **56/91 -> 72/91**, 16 fixed, passing legs turned failing: none, gold recall lost: none, target FIXED, **D-112 s2: ACCEPT**, exit 0 |
| 52 | `52-rev-hash-outputs.json` | sha256 of every file both replays wrote (`C:/t/rev773/scripts/hash_outputs.py`) | **56/56 base-passing legs: identical PWML set** (names and bytes); the only other files differing are `quarantine_report.json`, `removed_entity_report.json` (and `enrichment_report.json` on 2 legs, see 53) |
| 53 | `53-rev-json-diff.json` | leaf diff of those JSONs with `policy_version`, `decision_input_hash`, `decision_id`, `schema_version` stripped (`json_diff.py`) | differences only on the 16 fixed legs; on the other 65 legs the sole remaining leaf is `enrichment_report.json /artifacts/enrichment_dump_path` (the out-dir path, base vs tip) on 2 legs. The 9 still-failing legs are record-identical to base |
| 54 | `54-rev-check-fixed-legs.json` | per fixed leg: replay with freeze input captured, all export gates re-run on the frozen `canonical.json`, CURRENT `reaction_evidence.build_report` + `render_text`, absence checks, D-113 rows (`check_fixed_legs.py`) | **16/16: gates ALL PASS on the frozen payload; every omitted process and every removed entity VISIBLE in render_text (QUARANTINED / REMOVED ENTITIES); none still in the frozen payload; release review_required, eligible False, `per_row_omissions_removed_from_pwml:<n>` in review_reasons and release.reasons; `quarantine_reused=True`; problems 0** |
| 55 | `55-rev-base-on-tip-frozen.json` | BASE `quarantine_and_close` and BASE gates on each of the 16 TIP frozen payloads (`base_on_tip_frozen.py`, base policy 2026-07-31.1) | **16/16: base quarantine ok, 0 quarantined, 0 removed, payload idempotent; base gates ALL PASS** |
| 56, 57 | `56-rev-pins-delta-base.json`, `57-rev-pins-delta-tip.json` | the implementer `pins_delta.py` on base and tip trees | 28/28 ORCH-766 digests identical after stripping the 5 version fields; tip raw `base` pin `32f3a34e...ed1b` as in RESULT s6 |
| 58 | `58-rev-g9-base.json` (+pin) | `tests/test_orch773_omit_per_row.py` on base tree C:/t/rev773/basetree (3ef3fa9c src/tests + the new test and fixtures only), `--expect-tree` | **9 failed, 5 passed**: the 8 `test_g9_*` fail on refusal VALUES (`quarantine:unexportable_entity:1` x2, `final:Protein has degree 0 ...`, `final:Generated protein complex ... stoichiometry.` x2, `stage8:actor_schema_not_canonical`, `required:reaction_enzyme_must_be_protein_complex`, `ir:protein_complex_missing_components`); `test_new_release_cap` fails on its in-test import (labelled NEW); 5 preservation pass |
| 59 | `59-rev-g9-tip.json` (+pin) | same file at tip | **14 passed** |
| 60 | `60-rev-smoke-tip.json` (+pin) | SMOKE, 22 files (TEST_MATRIX Commands block) | **511 passed** in 43 s |
| 61 | `61-rev-affected-tip.json` (+pin) | 6 strict-quarantine suites + orch766 + orch769b + orch759c + Chunk E `test_strict_quarantine_real_artifact_replay.py` + `test_orch768_replay_corpus.py` + `test_d088_coverage_diagnostics.py` | **429 passed** |
| 62 | `62-rev-alone-streamlit-qb.json` (+pin) | `test_streamlit_quarantine_boundary.py` alone | 24 passed |
| 63 | `63-rev-alone-c055.json` (+pin) | `test_c055_rag_loop_wiring.py` alone (RAG round classification seam) | 14 passed |
| 64 | `64-rev-g11-check.json` | `g11_evidence.py check --task ORCH-773 --require-pin --forbid-refused-pin` | my reports 49-63 all `ok`; exit 1 only because the implementer 26 pytest reports have their `.pin.json` on the branch, not in my worktree (I seeded only `g11/ORCH-773/` from f35de933) |
| 65 | `65-rev-mutant-no-rollback.json` (+pin) | the focused file against a copy of tip with the mid-loop rollback disabled (C:/t/rev773/muttree, one-line sed at strict_quarantine.py:3215) | 2 failed (both rollback preservation tests), 12 passed: the rollback tests bite |
| 66 | `66-rev-fixture-provenance.json` | `tests/fixtures/orch773/*.gz` vs the freeze inputs I captured in 54 | 7/7 captured legs: payload and pathway_context equal (the 8th fixture is not a fixed leg) |

Logs: `C:/t/rev773/*.log`; scripts: `C:/t/rev773/scripts/`; replay outputs: `C:/t/rev773/out-base`, `out-tip`, `chk`.

## Severity standard, item by item

1. **Gate relaxed / something exported a gate would reject: NO.** No gate module changed; the two predicate refactors are
   verbatim. On every fixed leg the frozen payload passes the final gate, Stage 8, the required-field gate and the IR with
   the export pre-freeze seam (54), and the BASE quarantine plus BASE gates accept the same 16 tip frozen payloads with
   nothing to quarantine (55). For the same final payload base and tip agree. The 56 base-passing legs are byte-identical (52).
   PathWhiz import contract (ORCH-761 s1): the omitted complexes and processes are gone; the one edited reaction
   (767 enterobactin) keeps two complex enzymes (`check_spontaneous` holds).
2. **Omission not recorded / not visible: NO.** 54: every `quarantined[]` row carrying `detail.omission` renders as a
   QUARANTINED block with its Name, State and exact Reason; every `removed_entities[]` row renders as
   `- <bucket> <name>: <reason>` under REMOVED ENTITIES (example `C:/t/rev773/chk/ORCH-755_enterobactin-strict_2026-10-20_2201/reaction_evidence.txt`).
   The D-113 actor removal is visible as the wrapper REMOVED ENTITIES line; which reaction lost it is only in
   `detail.actor_removed_from` (NOTE 18, ORCH-772).
3. **Passing leg turned failing: NO** (51).
4. **Gold recall lost: NO** (51). 767 enterobactin final 5/5 and PWML 5/5; 759 enterobactin PWML 5/5; 755 enterobactin
   had no canonical at base, now final and PWML 4/5 (the omitted EntB reaction, KI-017).
5. **Whole-pathway refusal silently replaced: NO.** The 9 still-failing legs are record-identical to base (53). The 16
   converted per-row refusals are recorded omissions (54).
6. **LLM-input change: NO.** The pass runs inside `quarantine_and_close`, reached pre-freeze only through
   `freeze_canonical_payload` and the RAG round `_classification` adapter (streamlit_app.py:1105-1142), which returns the
   graph unchanged and publishes `classification`/`coverage` on the channel; `rag_round_reports` (993-1006) passes only
   `qa_graph`, `gate`, `mapping` to the RAG stage; grep finds no other reader of those two channel keys. No module reading
   `quarantine_report`/`removed_entity_report` (reaction_evidence, batch/driver, bench/*) builds an LLM prompt; the app
   refinement curator (2828-2835) gets `working_json`, QA and mapping misses, no quarantine annotation. Post-freeze
   consumers get the smaller frozen payload: the orchestrator reading says this is not an LLM-input change.
7. **Unexplained byte change in a base-passing PWML: NO** (52: 56/56).
8. **Omission rule wider than the gate predicate on a realistic payload: NOT SHOWN.** See NOTE 1 and 2: the rule is wider
   in theory, but on all three realistic payloads where it fires the end state equals what the downstream gates force.
9. **Post-freeze repair / out of boundary / protected file: NO.** See boundary; NOTE 3 on the pre-existing re-quarantine
   branch of `run_pwml_export`.
10. **G9 without base-red proof: NO.** 58 vs 59 reproduce RESULT s5 exactly; failures are on behaviour (refusal lists), not
    on symbol absence; the one import failure is the test labelled NEW.

## D-113 and the rollback (RESULT s12 Q6, Q4)

* **D-113 s1.** `_invented_wrapper_slots` (2832) applies only to a reaction, a generated wrapper with
  `generation_reason == complex_named_source_entity_wrapper` (the Stage-2 complex-named relocation), every actor resolving
  to it marked `inferred`, and an `extracted` enzyme kept (2849-2860). On 767 enterobactin (54, f.): input enzymes
  `[EntF complex (extracted), enterobactin synthase complex (inferred), EntE complex (inferred)]`, frozen
  `[EntF complex (extracted), EntE complex (inferred)]`, same for modifiers; reaction kept; gold 5/5. The wrapper own row
  is `generated=True gen_reason=complex_named_source_entity_wrapper`, component `enterobactin synthase` with stoichiometry
  None: D-113 s2 holds, the count gate is not touched and still refuses the wrapper, so it leaves.
  rv 08-21_2057 PMC12452463: the wrapper is `complex_inferred_from_evidence` and, in the captured freeze input
  (`C:/t/rev773/chk/runs_verify_2026-08-21_2057_PMC12452463/freeze_input.json`), the reaction
  `enterobactin synthase complex reaction` has that wrapper as its ONLY enzyme and modifier (provenance `extracted`).
  D-113 s1 requires a remaining paper catalyst, so it cannot apply under any reading; omitting the reaction is correct.
* **Rollback ("keep an omission only if it buys a PWML").** Read against the loop (3198-3230): the snapshot is taken
  before the first round and restores `working`, `admissions`, `iterations` and the three removal lists, so a rolled-back
  run leaves the loop in exactly the state base leaves it (`changed=False`, `converged=True`, same iteration record).
  `originals` is never mutated (D-113 edits hit `working` rows only). The refusal the user sees after a rollback is the
  base refusal itself, so it is not a silent replacement. The corpus does not exercise it (the 8
  `no_defensible_reaction_support` legs carry one required-gate error and nothing per-row), but the two preservation tests
  do and they bite (65).

## Findings

**F-1. NOTE. The declared-complex cascade is wider than the quarantine own predicate (theoretical).**
strict_quarantine.py:2913-2925: when an entity leaves, every complex resolving a component to it leaves whole, with reason
`protein_complex_component_unrepresentable:<c>:<r>`. For a DECLARED complex `_entity_representability` returns True
whatever its components (756-760), so that reason string is one the predicate never emits for such a row. Measured on the
three realistic payloads where it fires (54, a.): `isochorismatase` (sole component EntB, unexportable), `CLPXP` (both
components unexportable), `FtsH/YciM complex` (both components unexportable). With every component gone the IR refuses a
declared complex (`protein_complex_missing_components`, exactly what happened on rv 08-25_1216 for LONP1 and FtsH/YciM), so
the end state equals the gate-driven one. A declared complex with one bad and one good subunit would leave whole where the
gates would export it with fewer subunits; no corpus leg has that shape. The implementer chose this deliberately (RESULT
s10) to avoid a silent subunit edit; it errs toward omission, never toward export. Suggest the reason say that the
complex left with its subunit, not imply the predicate refused it.

**F-2. NOTE. The same cascade also runs for an `attached=False` (degree-0) protein (theoretical).** `drop_entity`
runs the complex loop before the `if attached` guard, using `_resolve_component` (synonym or identity), while the final
gate exempts components by normalized name only (process_normalizer.py, `_complex_component_norms`). A declared complex
naming the stray row by a synonym would leave whole. On 759 enterobactin and rv 08-18_1328 only the protein left (54).

**F-3. NOTE. The pass also runs in `run_pwml_export` re-quarantine branch** (streamlit_app.py:4710-4716, reached only when
the stored decision does not match: refinement edits or grounding). There it acts on a post-freeze payload, the export-time
report overwrites `outputs/quarantine_report.json` without the freeze-time omissions, and the release cap is recomputed
from a payload that no longer shows them. All of this is pre-existing for every freeze-time quarantine record at base,
and the seam is not new; the one new kind of mutation reachable there is the D-113 actor-list edit. Not exercised by the
corpus (`quarantine_reused=True` on all 16, 54). Not merge-rule-8 in my reading (removal and quarantine, the base
boundary re-decision); flagging so the orchestrator can decide whether it wants a ticket.

**F-4. NOTE. Message-text coupling.** `_UNATTACHED_PROTEIN_MESSAGES` (2615) matches the final gate message prefixes and
`_INVENTED_WRAPPER_GENERATION_REASON` (2622) duplicates a literal. If the gate wording changes, `attached` flips to True
and a degree-0 protein takes every process that resolves to it (the F-2 synonym case in RESULT would then omit 759
enterobactin EntF reactions). `test_g9_final_gate_degree_zero_protein_is_removed_and_every_reaction_stays` would catch it
on that fixture. Prefer shared constants.

**F-5. NOTE. Two calls sit outside the try.** `_freeze_candidate` (2814) and `_apply_per_row_failures` raise straight
out of `quarantine_and_close` instead of blocking/rolling back. On round 1 they run the same functions base runs after the
loop, so no new exception surface there; later rounds only, theoretical.

**F-6. NOTE. Comment nit.** strict_quarantine.py:2579 cites "D-113 section 3" for the rollback; D-113 s3 is about
interaction-only pathways and `no_drawable_process`.

**F-7. NOTE. Naming.** A process omitted because one actor fails a gate (`actor_schema_not_canonical`,
`reaction_enzyme_must_be_protein_complex`) is recorded as `quarantined_broken_reference`; the exact gate code is in
`reason`, so nothing is lost. Wording only.

**F-8. NOTE. Rollback coverage.** The corpus never triggers the rollback; only the two preservation tests do. They bite
(65). Worth a corpus-derived rollback case when one appears.

**F-9. NOTE (Q1, product owner).** The process unit is the lane charge. Cost seen here: 755 enterobactin loses the gold
reaction `EntB isochorismatase reaction` (PWML 4/5) because its enzyme complex has no identified subunit. Option (b)
would export it uncatalysed.

**F-10. NOTE (Q2).** The bump follows the module own rule (a verdict-changing release bumps). Measured cost: 28 ORCH-766
digests (version fields only, 56/57), two `decision_input_hash` pins and the C-114 `test_d088` assertion turned into a
literal of the archived version. Acceptable as a deliberate, documented move; needs ratification, since ORCH-769b chose
the opposite.

**F-11. NOTE (Q3).** Agree: keep `entity_type_overlap` a refusal; picking a declaration is not decidable from the payload.

**F-12. NOTE (Q4).** No silent-replacement risk found (section above). Recording more on a still-refused leg is a
product choice, not a correctness issue.

**F-13. NOTE (Q5).** `build_pwml_ir` takes `db_resolver=None` by default (ir.py:1315) and the corpus shows 0 network
attempts at tip, so the in-quarantine IR adds no DB traffic. The equivalence with the seam is the implementer F-5 (81/81);
I did not re-run it. Live runs with the PathBank DB reachable are the open exposure.

**F-14. NOTE (Q6).** The reading matches D-113 s1 and is narrower (actor provenance `inferred` required), so any case it
misses falls back to the default process omission, never to a wider edit. rv 08-21_2057 PMC12452463 is outside D-113 s1
under any reading (no other catalyst; verified in the freeze input). Recommend the product owner confirm.

**F-15. NOTE (Q7).** Converting `actor_schema_not_canonical` per row follows D-114 s2 literally (the Stage-8 contract
names one transport each time); given KI-024 (the payload was left un-normalized by an exception) the product owner may
prefer a whole-pathway refusal for sour taste, rv 09-02_2052 PMC12452463 and 767 Wnt/Ca2+.

**F-16. NOTE (Q8).** The orchestrator reading of D-104 AMENDMENT 2 applies; I found no omission text in any prompt.

**F-17. NOTE (Q9).** Harmless: `run_pwml_export` uses the same `setdefault` semantics (streamlit_app.py:4780-4788). A
user-cleared name ends in the export refusal with the omissions recorded, which is not an export of anything wrong.

**F-18. NOTE (Q10, Q11).** D-114 s2 "a PWML with ANY omission" plausibly covers pre-existing admission quarantines too;
extending the cap would rename base-passing PWMLs, so it needs a product ruling and is correctly not done here. For
ORCH-772: render `detail.omission` and `detail.actor_removed_from`, otherwise the D-113 reaction-level change is only a
wrapper line under REMOVED ENTITIES.

**F-19. NOTE (process, this review).** (a) I once parsed a committed G11 report with `python -c` reading the JSON from
stdin, outside the wrapper (data, not code; finished at once; the sweep after it was clean), and once ran
`g11_evidence.py check --help` unwrapped. (b) `rev773-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-773/.staging/01-rev-corpus-base.json`
is an unused reservation made before I seeded the implementer 01-48 into my worktree to avoid sequence collisions: do
not copy it. (c) `64-rev-g11-check.json` records exit 1 for the reason given in its row; `65-rev-mutant-no-rollback.json`
records exit 1 by design (a mutant). (d) The implementer RESULT s10 also records two of its own rule incidents.

## Decision compliance

| decision | status |
|---|---|
| D-106 (identity requirement not relaxed) | satisfied: `_entity_representability` unchanged in verdict; unidentified proteins leave, none exported |
| D-112 s2 (corpus acceptance) | satisfied: ACCEPT reproduced (51) |
| D-113 s1 (inferred wrapper leaves, paper catalyst stays, gold 5/5) | satisfied on 767 enterobactin (54) |
| D-113 s2 (wrapper count gate stays strict) | satisfied: gate untouched, wrapper omitted not admitted |
| D-113 s3 | n/a (not touched; comment nit F-6) |
| D-114 s2 (per-row omitted pre-freeze and recorded; no gate relaxed; review_required) | satisfied (54, 55) |
| D-114 s2 whole-pathway stay refusals | satisfied (53) |
| D-104 AMENDMENT 2 (LLM inputs) | satisfied, under the orchestrator reading for post-freeze consumers |
| merge rule 8 | satisfied in the freeze path; F-3 for the pre-existing export re-quarantine branch |
| protected `streamlit_app.py` | byte-identical |

## What I could not verify

* Live runs with the PathBank DB reachable (the in-quarantine IR runs without the pre-freeze seam; equivalence is measured
  only offline, 81/81 by the implementer, not re-run by me).
* An actual PathWhiz import of the 16 new PWMLs; I checked them against the T2PW gates that encode the ORCH-761 contract.
* The `run_pwml_export` re-quarantine branch with a refined or grounded payload (F-3): reasoned from code, not run.
* Chunk D, the eval-v2 files and the AppTest files other than the two run alone (62, 63).
* The lane brief wording on the policy-version bump (RESULT cites it; I did not see it).
