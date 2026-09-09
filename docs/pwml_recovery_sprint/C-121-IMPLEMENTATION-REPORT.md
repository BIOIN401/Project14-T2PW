# `C-121` implementation report — F-192, the auto-state lifecycle across the quarantine sweep

**Branch** `card/C-121-autostate-lifecycle` · **base SHA** `c4a3a92c` · **worktree** `C:/t/c121`
· 2026-09-08. Not pushed, not merged. Nothing was written to the primary checkout and no
branch other than this one was touched.

Governing documents, read in the mandated order: `C-121-PREDISPATCH-MEASUREMENT.md`,
`prompts/C-121-f192-autostate-lifecycle.md` (AMENDMENT 1 first), `DECISIONS.md` § `D-099`
(all nine sections), `PRODUCT_CONTRACT.md`, `TEST_MATRIX.md` § 0.

---

## 1. The diff surface, exactly

| File | What changed |
|---|---|
| `src/t2pw/pipeline/process_normalizer.py` | **+2 functions, +2 module constants.** `autostate_restoration_required` (the authorized disjunction), `restore_autostates_if_required` (guard + call), `_VISIBLE_LOCATION_BUCKETS`, `_EXPORTABLE_PROCESS_BUCKETS`. **`ensure_autostates` itself is untouched** — not one line. |
| `src/t2pw/pipeline/strict_quarantine.py` | **3 hunks.** One name added to the existing module-level `process_normalizer` import; the `_prune_biological_states` **docstring correction**; the guarded call inside `quarantine_and_close`. No function body other than `quarantine_and_close` changed, and inside it the only new statement is the `if restore_autostates_if_required(working):` block. |
| `tests/test_c121_autostate_lifecycle.py` | **New file, 12 tests.** Two G9 proofs per shape, fixture C, and six labelled new-capability arms. |
| `tests/test_strict_quarantine_real_artifact_replay.py` | **A pinned baseline moved deliberately.** `RESIDUAL_CODES_BY_LEG` / `_BY_ROW` lose `no_biological_states`; `test_stage_three_recovery_is_not_strict_exportability`'s allowed-code set tightens to the two species codes; the change-log cross-check renders the residual triple by name instead of subscripting the removed key. §5 has the exact delta. |
| `docs/change_log.md` | **Companion edit enforced by a committed test** — see §5.3. Two factual substitutions plus one dated note. |
| `docs/pwml_recovery_sprint/evidence/` | 4 proof scripts (`c121_impl_census.py`, `c121_impl_replay.py`, `c121_f179_verdicts.py`, `c121_changelog_edit.py`), 28 G11 cleanup reports, 21 pin verdicts. |

**Nothing forbidden was touched.** Verified by `git diff --name-only c4a3a92c`, which lists
exactly four tracked files (the two source modules, one test file, `docs/change_log.md`).
`src/t2pw/app/streamlit_app.py` (D-097), `src/t2pw/pipeline/reaction_support.py` (F-179),
`src/t2pw/pwml/ir.py`, `src/t2pw/pwml/writer.py`, `src/t2pw/mapping/map_ids.py`,
`src/t2pw/batch/driver.py`, `src/t2pw/pipeline/release_status.py`,
`src/t2pw/pipeline/stage_contracts.py`, everything under `src/t2pw/rag/`, every cache and
every `runs*/` directory: **absent from the diff.**

`reaction_support.py` byte-identity, proven by blob hash rather than by inspection:

```
worktree  109fe69dffc946516a8f17876cc691685f61125f
c4a3a92c  109fe69dffc946516a8f17876cc691685f61125f
HEAD      109fe69dffc946516a8f17876cc691685f61125f
```

### What was NOT done, and why

* **`_prune_biological_states`'s removal policy is unchanged.** It removes exactly what it
  removed before. Fixture C measures that on both trees.
* **No gate was weakened.** `no_biological_states` and `visible_entity_missing_location_state`
  both still refuse. The five archived legs now pass them by *satisfying* them.
* **No key was added to `quarantine_report`.** That report's `schema_version` bumps on every
  additive key by house rule, and three tests assert `== 6` — a wider blast radius than this
  seam owns. The effect is already visible in `resulting_payload_hash`, and a `logger.info`
  names it in the run log. C-074 arm A set the precedent ("NO SERIALIZED KEY WAS ADDED, so no
  `schema_version` moves").
* **No PMC id or gold pathway name entered `src/`.** The first draft of the docstring
  correction cited two legs by PMC id; `test_c074_strict_core_floor.py::test_this_diff_puts_no_benchmark_paper_or_gold_pathway_name_into_src`
  caught it (evidence `28`), and the docstring now says "one archived leg … a second" and
  points at `D-099` § 5 for the enumeration. Fixed in `29`.

## 2. The seam, and why it is where it is

Inside `quarantine_and_close`, immediately after `_reconcile_locked_reactions` and **before**
`evaluate_core_coverage`:

* the closure loop has already exited on `converged`;
* `_drop_quarantined_processes` has already run, so `working["processes"]` holds the
  **surviving** processes and nothing else — which is what makes the guard's "exportable
  content survives" clause mean what the census measured it to mean on the committed
  `final_mapped.json`;
* the coverage check, the invariant block and the semantic pass have not read the payload;
* it is upstream of `freeze_canonical_payload`, so **merge rule 8 holds by construction**.

The restoration **function** lives in `process_normalizer.py` and `strict_quarantine.py` holds
only the guarded call, as the card § 6 and `D-099` § 1 prefer. `strict_quarantine` already
imports four names from `process_normalizer` at module level, so no new import direction and no
cycle was created.

**The body is `ensure_autostates` itself, not a narrower reimplementation.** Considered and
rejected: a helper that skipped `ensure_autostates`'s species backfill would be a second copy of
the placeholder's name, its `cell` compartment and the transport `from`/`to` fallbacks, free to
drift from the Stage-3 pass every payload already went through — the defect class F-083 is. The
measurement says the wholesale call is safe on the legs that fire: **reaction counts identical
to the archived payload on all five, zero tree errors, and byte-for-byte the same PWML sizes the
orchestrator's pre-dispatch replay recorded.** Restoration is inert on coverage and on the
semantic pass by inspection too — neither `evaluate_core_coverage` nor
`bench/semantic_production.py` reads `biological_states` or `subcellular_locations` at all — and
`subcellular_locations` is deliberately outside `_GRAPH_ENTITY_BUCKETS`, so the appended `cell`
row cannot be pruned or counted degree-zero.

Research mode is unaffected: it replaces `working` with a fresh deep copy of the input *after*
this point, so the payload still comes back untouched.

## 3. Gate results

Interpreter `…/Project14-T2PW/.venv/Scripts/python.exe`. Every job ran through
`bounded_run.py`; every pytest job additionally through `pinned_pytest.py` with
`--expect-tree` and a committed `--pin-verdict`; every pytest job carried `--basetemp` under a
pre-created parent (`C:/t/tmp121`, `C:/t/tmp121b`). Heavy jobs held `--heavy-lock C-121`, one at
a time. No `nohup`, no `&`, no detached job, no `taskkill /IM`.

| Gate | Result | Evidence |
|---|---|---|
| **G1** G9 behavioural proof per shape | **PASS.** Base `c4a3a92c`: **5 failed / 1 passed / 6 skipped**, every failure on VALUES. Tip: **12 passed** | base `g11/C-121/15-g9-base-failure-final.json`; also in the wider A/B `24`. Tip `14`, `30` |
| **G2** Fixture C | **PASS.** `__auto_state__` still removed with reason `state_unreferenced_after_quarantine`, still absent afterwards, guard quiet, payload byte-identical. Passes at base **and** tip — it is the preservation arm, not a G9 proof | `15` (1 passed at base is this test), `30` |
| **G3** All five archived payloads serialize | **PASS.** All five gate-PASS with **0 tree errors**, reaction counts identical to the archived payload, and PWML sizes **byte-identical to the pre-dispatch reference** — 101095 / 80048 / 43794 / 23098 / 17295. Both controls byte-identical at 49464 / 48458 and the guard quiet on both | `32-replay-e2e-final.json` (also `07`) |
| **G4** F-179 unmoved | **PASS.** `reaction_support.py` blob-identical (above). `evaluate_reaction_support` + `reaction_support_issue` digest over all 10 legs of `runs_smoke/2026-09-08_1528` **byte-identical base vs tip**, sha256 `be29ac26…`. Restoration moved 0 of 10 verdicts. `tests/test_f179_reaction_support.py` green, including the fabricated glycine→heme refusals | `16-f179-verdicts-tip.json`, `17-f179-verdicts-base.json`, `30` |
| **G5** Inert on the quiet set | **PASS.** 154 production legs. Guard fires on **exactly 5**; cross-tab **5 / 0 / 0 / 149**; **0 quiet legs perturbed**; **40** legs an unguarded re-run would have changed; 0 false positives, 0 false negatives, 0 predicate/entry-point disagreements | `31-census-guard-final.json` (also `06`) |
| **G6** SMOKE | **PASS at 508**, not 503. 508 is the sprint's current committed number (`HANDOFF.md` § 0 cites `g11/ORCH-723/89-smoke-docs-fix.json`); 503 is stale in the dispatch brief exactly as 465 is stale in `CLAUDE.md`. Plus focused `tests/test_strict_quarantine*.py` + F-179 + the new file: **363 passed / 3 skipped** | `33-smoke-final.json` (also `25`), focused `30` |
| **G7** Gold-readers, 22 files | **PASS. 465 passed / 0 failed / 8 skipped / exit 0** — exactly the committed baseline. Run even though the diff touches no gold data, and it earned its keep: run `28` is where the PMC-id-in-`src/` violation was caught | `29-goldreaders22-tip-b.json` (failing pre-fix run kept at `28`) |

### 3.1 The base-failure proof, stated exactly

* **Base SHA measured:** `c4a3a92c` (`c4a3a92cca31243f5b34619451ce7e633c3c0f4e`), materialized
  by `c045b_base_tree.py --rev c4a3a92c --dest C:/t/c121base` — blob content written directly
  and re-hashed file by file, no EOL filters. `runs/` and `scripts/` were supplied as directory
  junctions to this worktree's copies, both verified identical `c4a3a92c..HEAD`, because
  `PATHSPEC` excludes them (Finding **H-3**).
* **Command:**

  ```
  PYTHONPATH=C:/t/c121base/src bounded_run.py --label g9-base-failure-final --timeout 900 \
    --cwd C:/t/c121base --json …/g11/C-121/15-g9-base-failure-final.json -- \
    python -u C:/t/c121base/docs/…/pinned_pytest.py --expect-tree C:/t/c121base \
      --pin-verdict …/g11/pin/C-121/15-g9-base-failure-final.pin.json \
      -q --basetemp=C:/t/tmp121b/g15 tests/test_c121_autostate_lifecycle.py
  ```

  `T2PW: C:\t\c121base\src\t2pw\__init__.py` — the measured tree is the base tree.
* **The assertions that failed there** (all on values; not one mentions a symbol):

  | test | base failure text |
  |---|---|
  | `test_shape_a_a_swept_payload_with_surviving_reactions_still_exports` | `assert [] == ['__auto_state__']` |
  | `test_shape_a_the_gate_stops_refusing_on_no_biological_states` | `assert {'no_biological_states'} == set()` |
  | `test_shape_b_the_gate_stops_refusing_on_a_missing_location_state` | `assert {'visible_entity_missing_location_state'} == set()` |
  | `test_shape_b_a_row_added_after_the_one_autostate_pass_still_exports` | `assert None == '__auto_state__'` |
  | `test_shape_b_the_rows_that_already_had_a_state_keep_it` | `At index 5 diff: None != '__auto_state__'` |

* **The first attempt was not evidence and was thrown away.** Run `11` failed **collection** on
  `ImportError: cannot import name '_VISIBLE_LOCATION_BUCKETS'` — symbol absence, which
  `G9` explicitly refuses. The module was restructured so the G9 proofs import none of the new
  symbols and the API arms skip instead of erroring. `11` is committed as the record of the
  refused attempt.

### 3.2 The census result

**The guard fires on exactly 5 production legs, and no leg outside those 5 changed** — and
that sentence is only worth anything with the measurement attached, which is what `REV-121`
was right to press on even though its own number was wrong.

**Under which measurement it is true.** `ORCH-737` verified it at the **production call site**:
`quarantine_and_close` driven over all 154 archived legs in both trees, **each leg in its own
export mode** — `research` for a `/research/` leg, `pathwhiz` for a `/strict/` one — comparing
the resulting payloads. **5 payloads moved, 0 legs changed their reaction set, 0 changed their
quarantine `ok`.** The five are the `D-099` § 5 population.

**Where my own census was weaker, and I am not defending it.** `c121_impl_census.py` evaluates
the predicate on the raw committed `final_mapped.json` rather than at the call site — as did the
orchestrator's pre-dispatch census. That method happened to return the right five here; it is
not the same measurement, and it should not be quoted as if it were.

**The trap, recorded so the next reader does not repeat it.** A naive probe returns **7**.
`REV-121` reached that number by driving two `/research/` legs through **strict** quarantine,
which production never does: research mode runs every decision and applies none of them, so such
a leg pushed down the strict path loses states production would have kept and then fires clause
1. In the mode production actually runs them in, `ORCH-737` measured both as **NOT MOVED**. The
same trap is reachable the other way, by evaluating the predicate on a payload read off disk.
Both extra legs are measurement artifacts, not firings. This is now stated in
`autostate_restoration_required`'s own docstring, not only here.

**The cross-tab qualifier is narrower than the firing count** and must not be quoted without it.
`no false positives, no false negatives` is a cross-tabulation against each leg's **committed**
`pwml_required_field_gate_report.json`, and **99 of the 154 legs carry no such report**. Over
the 55 that do, the split is exact:

| | failed on an F-192 code | did not |
|---|---:|---:|
| **guard fires** | **5** | **0** |
| **guard quiet** | **0** | **149** |

The five: `runs_smoke/2026-09-08_1528/…/PMC11961743`, `…/PMC4471609`,
`runs_validation/c120/2026-09-08_1240/…/PMC9544450`,
`runs_verify/2026-08-21_2014/…/PMC12312563`, `runs_verify/2026-08-24_1203/…/PMC13231680` —
the `D-099` § 5 population, unchanged. **40** quiet legs would have been perturbed by an
unguarded re-run; the guarded entry point perturbed **0** of them (payload-level census; the
call-site confirmation is `ORCH-737` above). The census reproduces the
orchestrator's pre-dispatch numbers exactly, measured against the **shipped** predicate rather
than a copy of it, and it additionally asserts that `autostate_restoration_required`,
`restore_autostates_if_required`'s return value and actual byte change agree on all 154 legs
(**0 disagreements**).

## 4. The docstring the archive falsified

`_prune_biological_states` claimed the coverage check backstops `no_biological_states` —
*"a payload with no surviving process fails there first."* **It does not, and the two checks are
independent:** coverage counts surviving PROCESSES, the sweep counts referenced STATES, and a
payload can have plenty of the first and none of the second. One archived leg reached the
required-field gate with **10** surviving reactions and zero states, a second with **4** — both
refused on `no_biological_states`, both with `removed_locations: []`, so there was never a row
for coverage to notice. The docstring now says so, states that the removal policy is correct and
unchanged, and names `restore_autostates_if_required` as the actual backstop.

## 5. The pinned baseline that moved, with its exact delta

### 5.1 What moved

`tests/test_strict_quarantine_real_artifact_replay.py`, over the frozen 39-leg cohort in
`tests/data/baseline_cohort_manifest.json`:

| pin | base | tip | delta |
|---|---|---|---|
| `RESIDUAL_CODES_BY_LEG` | `{classification: 19, taxonomy: 19, no_biological_states: 4}` | `{classification: 19, taxonomy: 19}` | `no_biological_states` **4 legs → absent** |
| `RESIDUAL_CODES_BY_ROW` | `{classification: 27, taxonomy: 27, no_biological_states: 4}` | `{classification: 27, taxonomy: 27}` | `no_biological_states` **4 rows → absent** |
| `FULL_STACK_BASELINE` | 39 / 28 admitted / 11 refused / 28 Stage 3 / **9** contract / 9 IR / **9** exportable | **identical** | **nothing** |
| `test_stage_three_recovery_is_not_strict_exportability` allowed codes | 3 codes | 2 codes | `no_biological_states` removed from the allowlist rather than left as slack |

**`FULL_STACK_BASELINE` not moving is the load-bearing half.** All four of those legs also carry
`species_missing_classification` and `species_missing_taxonomy`, so closing
`no_biological_states` does not make any of them pass the required-field contract:
`required_contract_pass` stays **9**, `exportable` stays **9**. One error code stopped being
emitted; **nothing was exported that was not exported before.** A move that had raised
`exportable` would have been the merge-rule-6 direction and a reject.

### 5.2 Measured as an A/B, not asserted

Same nine-file selection, same tip test files, only the two source modules differing:

| tree | result |
|---|---|
| base `c4a3a92c` (+ tip tests) | **7 failed / 350 passed / 9 skipped** — `g11/C-121/24-focused-sq-f179-base.json` |
| tip | **0 failed / 363 passed / 3 skipped** — `g11/C-121/30-focused-sq-f179-final.json` |

`363 = 350 + 7 + 6`: the 7 base failures are the 5 C-121 G9 arms plus the 2 replay pins above,
and the 6 formerly-skipped are the new-capability arms (base skips 9 = 3 pre-existing + 6). The
base run reports `by_leg={'species_missing_classification': 19, 'species_missing_taxonomy': 19,
'no_biological_states': 4}` in its own failure text, which is the delta measured rather than
recited.

### 5.3 `docs/change_log.md` — a companion edit I did not choose

`docs/change_log.md` is **outside** the file list I was given, and I did not want to touch it.
`test_the_change_log_baseline_table_agrees_with_the_pinned_values` renders its assertions
straight out of the pinned dicts and **fails if the log disagrees**, and its previous form
subscripted `by_leg['no_biological_states']` directly — so a residual code *leaving* raised
`KeyError`. There is no in-boundary way to keep that committed test green and truthful at the
same time. Three exact substitutions were made, scripted at
`docs/pwml_recovery_sprint/evidence/c121_changelog_edit.py` so a reviewer reads the
substitutions rather than an in-place rewrite:

1. `by leg 19/19/4, by row 27/27/4` → `by leg 19/19, by row 27/27` (the one-line restatement);
2. `no_biological_states` (4 legs, 4 rows) removed from the residual-code list;
3. a dated `C-121 / F-192` note recording what left and that no leg's verdict moved.

The test itself now renders that triple **by name** from `_RESIDUAL_LOG_ORDER`, skipping codes
the pin no longer carries: a code leaving is a reviewed outcome, a code *appearing* still fails
in the fragment loop. **Flagging this for the orchestrator as the one place my diff left its
declared boundary, deliberately and for a stated reason.**

## 5.4 Correction round (`REV-121` / `ORCH-737`), applied 2026-09-09

Two text corrections, no behaviour change, no test-expectation change:

1. `autostate_restoration_required`'s docstring now states the **conditions** the "exactly
   five" figure was measured under — call site, per-leg export mode, and the 55-of-154 scope of
   the gate-report cross-tab — and records both ways a naive probe returns 7. §3.2 above was
   corrected the same way.
2. `_VISIBLE_LOCATION_BUCKETS` now documents the **scanned-vs-assigned asymmetry**: the guard
   scans four buckets, `ensure_autostates` assigns to two, so a payload whose only unassigned
   visible row is a nucleic-acid or element-collection row would fire the guard, be mutated and
   still fail the gate. Left unfixed on instruction — narrowing the guard or widening
   `ensure_autostates` are both behaviour changes and neither is authorized — with the hazard
   named: `test_the_visible_location_buckets_are_the_gates_own_buckets` locks the tuple to the
   gate's table, so a fifth gate bucket would widen the scan without widening the assignment and
   green tests would say nothing.

**One number I could not reproduce, reported rather than copied.** The correction brief gives
the archived population of those two buckets as 3 and 7 legs. Measured over all 154 production
legs (`evidence/c121_bucket_gap_census.py`, reports `35` and `36`) I get
**`nucleic_acid_locations` on 4 legs and `element_collection_locations` on 10**, and no scope I
sliced reproduces 3 and 7 — strict-only is 2 and 5, research-only is 2 and 5, and legs with a
committed gate report are 0 and 3. **The conclusion is unaffected and is what the docstring
rests on: 0 legs in either bucket carry a row missing a state, so the gap is latent.** The
docstring records my measured 4 and 10 with the scope named. Worth a glance in case the brief's
numbers came from a slice I have not thought of.

## 6. Process lifecycle — 31 jobs, every one clean

Every job: `FINAL SURVIVING COUNT : 0` and `cleanup : success`. Reports `06`–`33` under
`docs/pwml_recovery_sprint/evidence/g11/C-121/`; 21 pin verdicts under `g11/pin/C-121/`, all
with `violations: []` and none `REFUSED`. `g11_evidence.py check --task C-121`: **33 artifacts,
21 invoking pytest directly, 21 with a committed pin verdict, 0 without, 0 refused, 0 foreign
`src` on path, 0 label mismatches.**

**One report is non-compliant and I am not hiding it.**
`10-base-tree-export.json` fails `report_too_large:252865` against the 64 KB schema cap. Cause:
the base-tree export runs `git cat-file` / `git hash-object` once per file over ~7 700 files, and
`bounded_run` faithfully records every one as an observed descendant (132 KB of
`descendants_observed`). The substance of the record is intact and is what G11 rule 5 exists to
guarantee — `exit_code 0`, `descendants_terminated` complete, `final_surviving_count 0`,
`cleanup_success true`. It is the documented `bounded_run` 64 KB cap meeting a base-tree export,
not a survivor, not a laundered result, and not a test job. Narrowing `c045b_base_tree.py`'s
`PATHSPEC` would fix the size but that file is shared tooling other cards depend on and is not
mine to change.

## 7. What I could not do

* **SMOKE cannot be measured on an exported base tree** — Finding **H-3**, confirmed rather than
  worked around. `26-smoke-base.json`: **5 failed / 497 passed / 6 errors**, in
  `test_c102_coverage_denominator.py` and `test_c106_mutation_harness_executable.py`, which need
  artifacts `PATHSPEC` does not export. **That is not a base SMOKE baseline and I am not
  reporting it as one.** What was measured instead is a strict differential:
  `27-smoke-base-plus-diff.json` is the same tree with **only my two source modules** replaced by
  their tip versions, and the result is **identical — 5 failed / 497 passed / 6 errors, the same
  five test names.** So my diff provably moves nothing in SMOKE, and on a complete tree SMOKE is
  508 passed / 0 failed.
* **Live pipeline validation was not run** and is not claimed. Every number here comes from
  archived payloads and from tests. `D-099` § 15's live validation still has to happen and none
  of this substitutes for it. Replay outputs went to a session scratchpad **outside** the
  repository and are not committed — they are archived-payload replays, not production
  deliverables, and must never enter the PathWhiz import set.
* **Chunk D was not run.** It is not in this card's gate list and it is a ~10.5-minute
  23-process cohort; the orchestrator owns that call.
* **No release status changed.** `D-099` § 2: nothing becomes `release_ready` because of this
  fix. This repairs serialization validity and touches no threshold, no cap and no
  classification.

## 8. Adjudication

No benchmark failure was used to justify code. The authorization is `D-099`, and the defect is a
`product_contract_violation` by `PRODUCT_CONTRACT` § 1: *"a technically recoverable problem must
not cause a run to produce no PWML"*, and § 5's rule that a biological mutation must move
upstream of the freeze rather than into an exporter. The placeholder is presentation scaffolding
— the reaction counts on all five legs being identical to their archived payloads is the
measurement that says so rather than the assertion that it is so.
