# ORCH-759e: RESULT (a protein that only the Phase-2 UniProt fallback resolves is now wrapped like any other)

| | |
|---|---|
| Lane | `orch759e/rewrap-after-phase2`, worktree `.claude/worktrees/orch759e` |
| Base | `366b9395` (= `sprint/pwml-recovery` = `main`) |
| Commits | Round 0: `f1f70087` test-first · `ced62cf4` code · `b4eefd9f` docs. **Repair round 1:** `a525e429` tests · `9bc6fe72` code · docs commit |
| Status | **Round 0 reviewed: REJECT (one blocking finding, mechanisms A and B). Repair round 1 implemented and tested (§ 12). Not merged, not pushed.** It needs a re-review of the round-1 diff. |
| Classification | `product_contract_violation`, merge rule 7: a correct `review_required` pathway fails the PWML required-field gate on one enzyme |
| Files | `src/t2pw/mapping/map_ids.py` (+58 −4 against base after round 1), `tests/test_orch759e_rewrap_after_phase2.py` (new, 10 tests) |
| Reports | `evidence/g11/ORCH-759e/01`–`24` (round 0), `25`–`41` (repair round 1), pins `evidence/g11/pin/ORCH-759e/`, logs `evidence/orch759e/NN-*.log`; every report `FINAL SURVIVING COUNT : 0` |

## 1. The defect

The diagnosis's line numbers hold at `366b9395` (`map_ids.py` is the same file as at `45aa9c28`).

1. The Stage-6 main protein loop strips CPO's Q9LR75 (name gate `no_shared_meaningful_token`, recorded as
   `rejected_mapped_ids`).
2. The wrapper rewrite (`map_payload` `:11305`, `_rewrite_rows` `:8484-8501`) sees no identity. It records
   `reaction_enzyme_complex_wrapper_skipped_invalid_component` and leaves CPO a bare `protein` actor.
3. The Phase-2 fallback (`:11406-11506`) restores Q9LR75 (`chosen_rule api_uniprot_fallback`, accepted on the gene
   symbol, `exact_symbol_identity`).
4. The Unknown fallback (`:11840`) wraps CPO1. It skips CPO because CPO now has a real identity (`:9358`).
5. Nothing else wraps a protein. The required-field contract and the IR fire `reaction_enzyme_must_be_protein_complex`
   on "CPO reaction".

**The same ordering defect applies to interaction endpoints** (asked in the brief). The endpoint half of the rewrite
(ORCH-754b) wraps a signed interaction's protein endpoints only when every one of them has an identity. A
Phase-2-resolved endpoint therefore left the whole row bare, and the IR withheld it as `bare_protein_endpoint`
(G9 test 3). On the archive this is PMC11405693 interaction 0, "piperonylic acid inhibits C4H" → AtCYP73A5 (§ 4).

Transporters are not affected. The rewrite never wraps transporters, for any protein. CERT on PMC12376012 is
Phase-2-resolved and stays a bare transporter at base and at tip, and the contract accepts it.

## 2. The diff (`ced62cf4`, `src/t2pw/mapping/map_ids.py` +25 −2)

- `map_payload`, Phase 2 (`:11423`, `:11518`): a set `_late_resolved` gets the normalized name of every protein the
  fallback resolved **and whose name gate accepted it**, which is the branch that sets `identity_status = verified`.
  A rejected answer is stripped again by `_enforce_shipped_identity_names` and is never recorded.
- `map_payload`, right after Phase 2 (`:11521-11529`). When `allow_complex_wrapper_creation` is on and the set is
  not empty, the **existing** `_rewrite_reaction_protein_enzymes_to_complexes` runs again with
  `only_proteins=_late_resolved`. Its return value is discarded: nothing is written to the mapping report (§ 6).
- `_rewrite_reaction_protein_enzymes_to_complexes` gets a new keyword, `only_proteins: Optional[Set[str]] = None`.
  `None` is the old whole pass, and the first call passes nothing. With a set:
  - an actor list (`modifiers`/`enzymes`) that names none of the proteins is skipped whole (`:8627`), so it is not even
    re-deduplicated;
  - inside a list that does name one, every other actor row is appended **as it stands**: the same object, not a
    canonicalized copy (`:8410`). So the drop rule and the canonical-row rewrite cannot touch a sibling;
  - an interaction with no endpoint among them is skipped (`:8707`). One that has such an endpoint runs the existing
    all-endpoints-wrappable rule unchanged, so its other, main-loop-resolved endpoint is wrapped too, as it would have
    been at the first pass.

Not touched: `_enforce_shipped_identity_names` and every gate (D-106), the strip-then-restore itself, the Unknown
fallback, `_fold_unreferenced_duplicate_proteins`, ORCH-762's identity code, `streamlit_app.py`, and every prompt.
`git merge-tree --write-tree HEAD orch764/no-class-narrowing` exits 0 with no conflict: ORCH-764's hunks sit at
`:41`, `:6882-6920` and `:7111-7140`, and this lane's sit at `:8204-8708` and `:11423-11529`.

## 3. Placement: M2 targeted (right after Phase 2), not M1

The brief preferred M1 (after the Unknown fallback) unless measurement showed otherwise. It does. Four placements were
measured, on the G9 fixtures and over the archive census (§ 4). The variants were emulated by
`stage6_replay.py --variant`, which moves or widens the one `only_proteins` call:

| placement | late enzyme wrapped | drops a sibling | wrapper equal to a main-loop wrapper | census legs changed |
|---|---|---|---|---|
| **M2 targeted (shipped)** | yes | **no** | **yes, byte for byte** (G9 test 1 asserts it) | the 4 defect legs only |
| M1 targeted | yes | no | **no**: `mapping_meta.route/query/issues/chosen_rule/resolution` and `provenance_lineage` differ, and the wrapper is missing from the report's entity log | the 4 defect legs only |
| M2 whole rewrite (the diagnosis's M2) | yes | **yes**: CPO1 dropped, chlorophyll actors 74 → 72, `broken_reference` 28 → 29, auxiliary 36 → 35 | yes | the 4 defect legs only |
| M1 whole rewrite (the diagnosis's M1) | yes | no on the archive | no (as M1 targeted) | the 4 defect legs only |

Sources: the fixture comparison is the probe recorded in Finding 3. The census rows are `evidence/orch759e/08-census-compare.log`.

What decided it:

1. **The diagnosis's case against M2 came from re-running the whole rewrite.** The drop is the drop rule firing on
   CPO1 once CPO counts as resolved. With `only_proteins`, siblings are passed through, and M2 drops nothing
   (chlorophyll 74/74 actors, `broken_reference` 28 = live).
2. **M2 is the existing path; M1 is a shortcut around part of it.** At M2, a late wrapper is created before the
   complex-mapping loop (`:11532`), so it gets exactly what a main-loop wrapper gets:
   - the PathBank complex lookup;
   - the F-133 superset-identity refusal;
   - the `route`/`query`/`issues` meta;
   - its entity-log row;
   - the Unknown fallback seeing it as a complex.

   At M1 it skips all of that, and its row differs from every other wrapper in the payload.
3. Neither placement changes a leg without a late protein. Nothing runs when `_late_resolved` is empty (§ 4: 116/116
   legs byte-identical).

## 4. Census: the Stage-6 replay over the archive, at base and at tip

`evidence/orch759e/stage6_replay.py` drives `map_payload` itself with the Stage-6 flags. It runs on every archived
strict `final_mapped.json` (scan_shape.py's population: 121 legs under `runs_papers`, `runs_verify`, `runs`,
`runs_validation` and `runs_smoke`; research legs and worktrees excluded). The script imports nothing this lane adds,
so the same file runs at base (`git checkout 366b9395 -- src/t2pw/mapping/map_ids.py` in this worktree, then
restored) and at tip.

How each leg is turned back into the Stage-6 input:

- a Phase-2-resolved protein loses its identity again;
- a PathBank-Unknown placeholder that stands in the same actor list is reverted to the bare protein it was at the
  rewrite. This is the M2 reconstruction `replay_cpo.py` did by hand for CPO1. It is faithful: had any sibling counted
  as resolved at the rewrite, the drop rule would have removed the late protein, and the leg shows it did not.

Every identity decision is the leg's, replayed:

- the main loop returns each row's recorded mapping;
- the Phase-2 fallback returns the recorded accession;
- the three name gates are pass-throughs;
- there is no DB, no species LLM and no taxonomy call;
- every HTTP request is refused and counted, and the count must be zero.

The checks are the ones `replay_cpo.py` runs, on the live artifact and on the replay: strict quarantine with the leg's
requested context, the final gate, the contract, `validate_pre_export`, and the IR.

| | report | result |
|---|---|---|
| base | `03-census-base` | 121 legs, 0 errors, 5 with a Phase-2-resolved protein |
| tip | `04-census-tip` | same legs |
| M1 / M2-whole / M1-whole | `05` / `06` / `07` | same legs |
| comparison | `08-census-compare` | below |

- **116 legs with no Phase-2-resolved protein: payload and report byte-identical in all five runs.** This is also the
  replay's determinism check.
- **The replay reproduces the live artifact.** On all 5 legs with a late protein, the base replay gives the same
  quarantine counts, the same release status and the same contract, pre-export and IR codes as the leg's own
  `final_mapped.json`.
- **4 legs change at tip, the 4 defect legs and no other.** The fifth (PMC12096016 `runs_verify/2026-09-01_1612`,
  late `holo-EntB`) is not an actor anywhere and is byte-identical.

| leg | late protein | base (replay = live) | tip |
|---|---|---|---|
| `runs_papers/ORCH-759/.../PMC5130987/strict` (chlorophyll) | CPO | contract + pre-export + IR `reaction_enzyme_must_be_protein_complex @ CPO reaction` | **clean**; CPO → `CPO complex` in enzymes and modifiers, CPO1 placeholder kept; actors 74/74; quarantine ok, **`review_required`**, counts unchanged (core 19, aux 36, broken_ref 28, unmapped 14); final gate ok |
| `runs_verify/2026-08-21_2057/.../PMC12096016/strict` | lactate dehydrogenase | `... @ LDH coupled reaction` | **clean**; `lactate dehydrogenase complex`; 12/12; counts unchanged |
| `runs_verify/2026-09-06_1425/.../PMC11405693/strict` | AtCYP73A5 | `... @ C4H hydroxylation` | **clean**; enzyme, modifier **and interaction endpoint** → `AtCYP73A5 complex`; 12/12; `review_required`; **core/aux 7/0 → 6/1** (Finding 4) |
| `runs_verify/2026-09-06_1425/.../PMC12376012/strict` | ceramide synthase, CERT | `... @ ceramide synthase reaction` | **clean**; `ceramide synthase complex`; CERT stays a bare transporter (§ 1); 21/21; counts unchanged |

**Nothing is dropped on any leg and no gate changes.** Every changed leg stays `review_required` with an ok final gate.
This census is the measured basis of the claim "no non-defect leg changes".

## 5. G9 and the replay

`tests/test_orch759e_rewrap_after_phase2.py`. The `test_g9_*` tests drive `map_payload` with both resolvers stubbed
per name, offline. They import nothing this lane adds and fail on the shipped payload at base.

| test | base `366b9395` (`01-g9-red`) | tip `ced62cf4` (`02-g9-green`) |
|---|---|---|
| `test_g9_an_enzyme_resolved_by_the_phase2_fallback_ships_wrapped` (LDH/AtCYP73A5 shape; the wrapper and actor row equal a main-loop enzyme's, byte for byte) | **FAIL**: `[('lactate dehydrogenase', 'protein')]` | pass |
| `test_g9_the_cpo_shape_wraps_cpo_and_keeps_the_cpo1_placeholder` (enzymes and catalyst modifiers; real leg candidate for CPO) | **FAIL**: `[('CPO', 'protein'), ('CPO1', 'protein_complex')]` | pass |
| `test_g9_an_interaction_endpoint_resolved_by_the_phase2_fallback_is_exported` | **FAIL**: `('SFXN4', 'ALAS2')`, IR `bare_protein_endpoint` | pass, `exportable` Inhibition |
| 5 × `test_preservation_*` (labelled) | pass | pass |

`01-g9-red`: 3 failed, 5 passed. `02-g9-green`: 8 passed. The CPO fixture at base fails the contract and the IR 5
times on `reaction_enzyme_must_be_protein_complex`, as measured in the probe of Finding 3.

Preservation tests (labelled, pass at base and tip):

- an enzyme resolved in the main loop is wrapped by the single rewrite (call count 1, counters unchanged);
- an enzyme still unresolved after Phase 2 takes the Unknown fallback (call count 1);
- research mode (`allow_complex_wrapper_creation=False`) wraps nothing;
- the late wrap writes no report key, counter or action (D-104 amendment 2);
- the drop rule still runs before Phase 2 (Finding 2).

**The chlorophyll replay** is the first row of § 4, in `03-census-base.log` and `04-census-tip.log`, and it runs
through `map_payload` at both SHAs:

- at base, the contract, `validate_pre_export` and the IR fail on "CPO reaction", matching the live artifact;
- at tip, all three are clean, the final gate is ok and quarantine is `review_required` with the leg's own counts.

## 6. D-104 amendment 2: what reaches an LLM

The mapping report reaches the interactive curator's prompt, because `compact_mapping_misses` walks it whole. The lane
**adds no key, counter or action to the report**. The targeted pass's own report is discarded, and the preservation
test pins it: the report and summary keys equal a main-loop run's, and the action list is base's.

What the curator's view **does** show differently on a defect leg is the corrected payload. This is measured on both
fixtures at base and tip (`22-llm-view-base`, `23-llm-view-tip`, `24-llm-view-diff`; `llm_view_dump.py`,
`llm_view_diff.py`):

- five existing summary counters move by one: `generated_wrappers_created`, `protein_complexes_total`,
  `protein_complexes_skipped`, `complexes_skipped` and `entities_unmapped`;
- the complex-mapping loop's ordinary entity-log row for the new wrapper appears. Its `gap_issues` field is the one new
  key path, and every main-loop wrapper's log row carries it.

All of it is exactly what the same payload produces when the main loop resolves the enzyme. Under M1, the
`generated_wrappers_created` and `protein_complexes_total` changes remain, because they are counted from the final
payload. No legacy payload changes at all (§ 4: 116 report digests identical). The PO may read the amendment more
strictly; see § 10.

## 7. Gates

| # | gate | report | result |
|---|---|---|---|
| 1 | G9 RED at base | `01-g9-red` | 3 failed (the 3 G9), 5 passed |
| 1 | G9 GREEN at tip | `02-g9-green` | 8 passed |
| 1 | replay + census, base / tip / M1 / M2-whole / M1-whole / compare | `03`–`08` | § 4 |
| 2 | affected: every `tests/test_*.py` matching `map_payload\|wrapper\|_rewrite_reaction\|reaction_enzyme_must_be_protein_complex\|pwml_ir`, plus the 4 named files and this lane's file, minus the 10 AppTest files (run alone, rows 5 and SMOKE) — 67 files in two batches | `09-affected-a`, `10-affected-b` | 861 passed, 1 skipped, **4 failed**; 1098 passed |
| 2 | the 4 failures at base `src/` | `11-known-fails-base` | the same 4 fail with identical assertion text: `test_c075…::test_corpus_the_armed_pass…`, `test_c100…::test_5b…`, `test_compound_resolution_extraction.py` ×2. All are in TEST_MATRIX § ORCH-754 "Known failures". The 1 skip is the c081/c089 `worktrees` path skip |
| 3 | SMOKE | `12-smoke` | **508 passed** |
| 4 | `tests/test_eval_v2_*.py` (6 files) | `13-eval-v2` | 89 passed, 1 xfailed |
| 5 | alone: `test_streamlit_quarantine_boundary` | `21` | 23 passed |
| 5 | alone: `test_c052_prefreeze_report_at_the_streamlit_seams` | `14` | 13 passed |
| 5 | alone: `test_reaction_evidence_view` | `15` | 11 passed |
| 5 | alone: `test_orch759a_fold_app_path` | `16` | 2 passed |
| 5 | alone (the other affected AppTest files): `c055_rag_loop_wiring`, `orch759d_app_timeout`, `prefreeze_third_export_seam` | `17`, `18`, `19` | 14, 4, 12 passed |
| 5 | alone: `test_batch_preflight` | `20` | 32 passed, 4 skipped, 1 failed: `test_the_message_names_the_problem_the_interpreter_and_the_cure`, worktree-only (no `.venv`), known (§ ORCH-757, § ORCH-757i) |
| 6 | D-104 am. 2 view | `22`–`24` | § 6 |

`test_batch_driver` and `test_stage3_gate_report_lifecycle` ran in SMOKE. Chunk D was not run, as instructed.
Every report shows `FINAL SURVIVING COUNT : 0`. No `--heavy-lock`. One job at a time.

## 8. Findings

1. **The defect is the ordering, not a gate.** It has two shapes: a reaction enzyme, and a signed-interaction
   endpoint. Neither needs a gate changed. Measured: § 4 and § 5.
2. **A second ordering hazard was found and not changed.** An enzyme with no identity at the rewrite, standing beside
   an enzyme that has one, is dropped there ("no UniProt or DrugBank identifier; other enzymes remain"). Its protein
   row is removed, with no `quarantined_proteins` record, before Phase 2 could resolve it. Reproduced on the fixture
   `[HEMA1 (main loop), CPO (Phase 2 only)]` at base: CPO gone from enzymes and proteins. Pinned unchanged by
   `test_preservation_the_drop_rule_still_runs_before_phase2`. Merge rule 7 may be implicated; see § 10.
3. **The placement measurement** (the scratch probe, run through the bounded wrapper, reproduced by the census):
   - M2 targeted: the late wrapper equals the main-loop wrapper, and CPO1 is kept;
   - M1: the wrapper differs in `chosen_rule`, `issues`, `query`, `resolution`, `route` and `provenance_lineage`;
   - M2 whole: CPO1 dropped;
   - M1 whole: as M1.
4. **The coverage matcher does not see through a wrapper name.** On PMC11405693, the requested core term
   `CYP73A5 (AtCYP73A5)` matched the bare actor `AtCYP73A5` by substring (`strict_quarantine._term_matches`). It does
   not match `AtCYP73A5 complex`, so "piperonylic acid inhibits C4H" moves from core to auxiliary (7/0 → 6/1;
   coverage 7/11 → 6/11). Release stays `review_required`, and nothing is quarantined. Every main-loop wrapper already
   behaves this way: `4CL (4-coumarate:CoA ligase)` is unmatched on the same leg at base, because its enzyme is
   `4-coumarate:CoA ligase complex`. A pre-existing property of `strict_quarantine`, outside this lane.
5. **The strip-then-restore is real and still there.** In one run, the main loop's name gate rejects Q9LR75 for CPO
   (`no_shared_meaningful_token`), and the Phase-2 gate accepts the same accession (`exact_symbol_identity`). This
   lane does not reconcile them (§ 10).

## 9. Failures and dead ends

- **Dead end, M1 as the brief's default.** I believed M1 would be the narrow choice. The fixture probe showed that an
  M1 wrapper skips the complex-mapping loop and is shaped unlike every other wrapper. The diagnosis's objection to M2
  disappeared once the re-run was targeted.
- **Wrong theory about the archive census.** I first planned to apply the rewrite to `final_mapped.json` directly, as
  `replay_cpo.py` M1 did. On a post-fallback payload, CPO1 is already a complex, so the first-pass drop rule removes CPO
  before Phase 2 at base and at tip alike, and the replay would not have reproduced the leg. Hence the reconstruction
  in § 4. The base replay then matched all 5 live artifacts.
- **IR crash on older legs.** The first census build ran `prefrozen_when_compounded` on every leg. Legs whose compounds
  already carry resolution verdicts made it raise an `AssertionError`, which showed up as `ir_build_crashed` on 4 of
  the 5 late legs (scratch run). The script now falls back to the payload as it stands. Only the fixed script produced
  the committed reports `03`–`08`.
- **A rule broken by me, with no effect.** One shell command ran `python.exe -` with an empty heredoc. That is
  forbidden here, and it hung on stdin until the 120 s Bash cap moved it to the background. I stopped it with
  `TaskStop` (my own task). A read-only `Win32_Process` listing then showed no python process of mine, only two
  VS Code isort servers. No test or replay ran in it, and nothing was written.
- **A resume message arrived mid-run** (from the orchestrator, saying I had been cut off). `git status` and
  `git log 366b9395..HEAD` showed exactly the state I had left: 2 commits and untracked evidence. There were no
  `.staging` placeholders, and every reserved report had been written. Work continued from there.

## 10. Not settled (for the product owner)

1. **The strip-then-restore inside one Stage-6 run** (Finding 5). The main-loop name gate and the Phase-2 gate disagree
   on the same accession. Options:
   - (a) one verdict per (name, accession) per pass. Cost: a change inside `_enforce_shipped_identity_names` (D-106
     scrutiny), plus a census of how many identities flip;
   - (b) leave it, since this lane makes the outcome exportable either way.

   A separate lane.
2. **The drop rule runs before Phase 2** (Finding 2): a correct enzyme can be removed when a sibling is resolved.
   Options:
   - (a) defer the drop until after Phase 2. This changes which actors ship on any leg with that shape, so it needs its
     own census and G9;
   - (b) at least record the removed row in `quarantined_proteins`, as the structural cleanup does;
   - (c) leave it.
3. **Coverage across wrapper names** (Finding 4). Options:
   - (a) let `_process_core_terms` add a generated single-protein wrapper's component name. That raises coverage
     wherever enzymes are wrapped, and the coverage baselines would move;
   - (b) leave it.
4. **D-104 amendment 2 scope** (§ 6). The report gains no record of its own, but on a defect leg the curator's view
   shows the corrected payload: 5 counters and the wrapper's ordinary log row. Should a payload fix that changes
   existing report fields need anything more? Options:
   - (a) accept: it is what the main-loop path produces;
   - (b) also suppress the late wrapper's entity-log row. That makes the report less honest than a main-loop run's,
     and still moves the payload-derived counters.
5. **No report record of the late wrap.** The report still carries the first pass's
   `reaction_enzyme_complex_wrapper_skipped_invalid_component` action for the enzyme that is now wrapped. The wrapper
   row is the record. Options:
   - (a) a record under a key in `PROMPT_OMITTED_PAYLOAD_KEYS`, so no prompt sees it. Cost: one key in
     `prompt_payload_keys.py` and a test;
   - (b) leave it.
6. **Transporters** are never wrapped, by either path. The contract accepts a bare transporter today. This is only
   recorded here, because CERT shows it.

## 11. TEST_MATRIX delta (exact text, for the orchestrator to append end-of-file)

```markdown
## ORCH-759e: test obligations — appended end-of-file by rule · 2026-10-22

Lane `orch759e/rewrap-after-phase2` (`f1f70087` test-first, `ced62cf4` code; repair round 1 `a525e429` tests, `9bc6fe72` code).
Reports `evidence/g11/ORCH-759e/01`–`41`.

**SMOKE is unchanged at 508** (`g11/ORCH-759e/12`, round 1 `30`). No pin moved. Merge rule 10 unchanged.

### New focused selection

`tests/test_orch759e_rewrap_after_phase2.py` (**10**, <1 s): a protein that only the Phase-2 UniProt fallback resolves
is wrapped through the existing Stage-6 rewrite, right after Phase 2, exactly as a main-loop enzyme is (wrapper byte-
identical); the CPO shape keeps its CPO1 Unknown placeholder and drops nothing; a Phase-2-resolved interaction endpoint
exports; with a PathBank database (stub) the late wrapper takes the complex the main loop takes (F-116 judged on
the reaction as written) and its complex goes through structural-cleanup Phase 1. Preservation: one rewrite when nothing is late, the Unknown fallback for a still-unresolved enzyme, research
mode, no new report key/counter/action (D-104 amendment 2), the drop rule unchanged. G9: 3 red at base (`01`), 8
green at round 0 (`02`); the 2 round-1 tests red at round 0 `ced62cf4` (`25`); all 5 G9 red at base (`27`), 10 green (`26`). Run it for any change to `_rewrite_reaction_protein_enzymes_to_complexes`, the Phase-2 fallback
in `map_payload`, or the order of Stage 6's wrapper / complex-mapping / Unknown-fallback steps.

### Archive replay

`docs/pwml_recovery_sprint/evidence/orch759e/stage6_replay.py` replays Stage 6 (`map_payload`) offline on every
archived strict leg with the leg's recorded identity decisions (121 legs, ~3 s), at any SHA; `census_compare.py`
compares runs. Base/tip: 116 legs byte-identical (117 with the late-but-unreferenced holo-EntB leg), the 4 defect
legs change and pass contract + IR (`03`–`08`); repair round 1 byte-identical to round 0 on 121/121 (`28`, `29`).

### Known failures

Unchanged from § ORCH-754's list (the 4 non-primary nodes red at base and tip with identical text, `09`/`11`, round 1 `31`) and
§ ORCH-757's worktree-only `test_batch_preflight.py::test_the_message_names_the_problem_the_interpreter_and_the_cure`
(`20`, round 1 `40`).
```

## 12. Repair round 1 (review round 0: REJECT under standard item 2)

**The finding.** Round 0 claimed that a late wrapper is what the main loop builds. That held with no PathBank
database, the census condition, and failed with one. The reviewer measured two mechanisms with a stub database
(their reports `rev759e-tip 35`–`37`). 56 of 78 archived strict legs carry DB complexes, 18 of them
multi-component, so both are reachable in production.

- **A. F-116 scope computed after renaming.** The `only_proteins` pass recomputed `actor_scope` from the
  reaction as the first pass had already rewritten it. A main-loop sibling then reads `LDHB complex`: the token is
  `name::ldhb complex`, with no `uniprot::p07195`. So the superset refusal turned down LDH holoenzyme = {LDHA, LDHB},
  which the main loop accepts, and the late pass built `LDHA complex`.
- **B. Structural-cleanup Phase 1 skipped.** A complex the late pass builds is created after Phase 1 has run. An
  unresolved component (LDHB, never resolved) therefore stayed in it. The main loop prunes that component and
  quarantines the LDHB row.

**The repair** (`9bc6fe72`, `map_ids.py` +34 −3 on top of round 0). Nothing else changed: no gate, report key or
prompt.

- **A.** `map_payload` passes one dict, `_pre_rewrite_actors`, to both rewrite calls (`:11338`, `:11554`).
  - The whole pass records, for each reaction, the scope it computed and the actor lists before rewriting them
    (`:8629-8632`).
  - The `only_proteins` pass judges F-116 against that record (`:8633-8637`): the recorded scope, unioned with the
    recorded actors re-read against today's protein rows. That is the reaction as the main loop would see it: its
    names as written, with the identity Phase 2 has since added.
  - Why re-read and not only reuse the scope: a reaction with two late proteins would otherwise miss the
    second one's new accession. The judged protein's own accession is added by the refusal function in either case.
- **B.** The Phase-1 block becomes a nested function, `_prune_invalid_complex_components(targets)`, with its body
  unchanged.
  - The "still needed" check still reads every complex.
  - It is called as before over every complex (`:11438`), and again right after the late pass (`:11556-11560`). The
    second call covers exactly the complexes the late pass created, or whose components it changed: a snapshot of
    component lists by row is taken before the call.
  - It runs only under `allow_structural_cleanup`, like the original.

**One known divergence, deliberate.** The replayed Phase 1 judges validity after Phase 2; the main loop's Phase 1
judged it before. So a third component that only Phase 2 resolves is kept by the late path, where the main path
would prune it. This is the direction that drops nothing (merge rule 7). The two new tests do not exercise it.

**Proof (G9, two new tests, stub PathBank database).** `test_g9_r1_a_late_enzyme_takes_the_pathbank_complex_the_main_loop_takes`
covers A. `test_g9_r1_a_late_pathbank_complex_is_pruned_like_a_main_loop_one` covers B. Each asserts that the late
run's reactions, complex rows, remaining proteins and quarantined proteins equal the main-loop run's.

| run | report | result |
|---|---|---|
| round-0 tip `ced62cf4` code, round-1 tests | `25-g9-r1-red` | **2 failed** (A: `LDHA complex` ≠ `LDH holoenzyme`; B: components `['LDHA', 'LDHB']` ≠ `['LDHA']`), 8 passed |
| base `366b9395` code | `27-g9-r1-red-base` | **5 failed**: all 5 G9, old and new; 5 preservation passed |
| repair tip `9bc6fe72` | `26-g9-r1-green` | **10 passed** |

**Gates on the repair tip** (every report `FINAL SURVIVING COUNT : 0`):

| gate | report | result |
|---|---|---|
| census, DB off, at tip | `28-census-r1-tip`, `29-census-r1-compare` | 121 legs, 0 errors. **Byte-identical to round 0 on 121/121 legs.** Against base, **117 legs unchanged** (116 with no late protein, plus the holo-EntB leg) and the **4 defect legs fixed**: contract, pre-export and IR clean, `review_required`, final gate ok, actors in = out (nothing dropped). The chlorophyll leg is the first row: `CPO complex`, CPO1 kept, quarantine counts = live |
| SMOKE | `30-r1-smoke` | **508 passed** |
| affected batch A / B | `31`, `32` | 861 passed, 1 skipped, 4 failed (the same 4 known nodes, identical to base `11`) / **1100 passed** |
| `test_eval_v2_*.py` | `33` | 89 passed, 1 xfailed |
| alone: c052, reaction_evidence_view, orch759a_fold_app_path, c055, orch759d, prefreeze_third, quarantine_boundary | `34`–`39`, `41` | 13, 11, 2, 14, 4, 12, 23 passed |
| alone: batch_preflight | `40` | 32 passed, 1 known worktree-only failure |

Merge check: `git merge-tree --write-tree HEAD sprint/pwml-recovery` (at `21155be0`, ORCH-764 merged) exits 0 with
no conflict.

**Not settled, added by this round.** The census cannot see mechanisms A and B: it runs with the database off, so
every wrapper is single-component. The stub-DB tests are the only proof. A database-on replay of the 18
multi-component legs would need a recorded PathBank answer per enzyme. The legs do not keep those answers, so the
replay would have to be built first.
