# ORCH-788 — a biological state with no subcellular location becomes a per-row omission

Branch `orch788/fix` off `22538dee`. Commits `f6541a25` (code), `89038702` (orchestrator: KI-015 catalogue row), `08552838` (tests). **Status: tested and replayed; not merged or pushed.**
Nothing is merged or pushed.

## 1. The failure (ORCH-784 sour_taste_v2-strict, PMC7059786)

`PWML IR validation failed`: `biological_state_missing_subcellular_location` on `/biological_states/11`
("Type III taste cell state") and `/biological_states/12` ("afferent nerve state"). Read from the leg's
`final_mapped.json` (read-only, `.claude/worktrees/orch784run/runs_papers/ORCH-784/legs/sour_taste_v2-strict/2026-10-27_2155/papers/PMC7059786/strict`):

- Neither state has a `subcellular_location`, and neither is named after a location row. "afferent nerve state"
  is not the same identity as the location "afferent nerve membrane". So there is nothing to repair.
- **No process uses either state.** Only two element placements do: `compound_locations` 5-HT in
  "Type III taste cell state", and `protein_locations` P2X receptors in "afferent nerve state". Both elements
  keep other placements (5-HT: synaptic vesicle lumen, synaptic cleft; P2X receptors: afferent nerve membrane).
- Why the existing per-row pass (ORCH-773) did not catch it: `_per_row_failures` already runs the IR gate.
  But `_row_failures` maps an IR pointer back to a row only for `/entities/proteins|protein_complexes`.
  `/biological_states/<i>` named no row, so the pass returned "blocked" and the whole pathway was refused.

## 2. Change

| file | function | change |
|---|---|---|
| `src/t2pw/pipeline/strict_quarantine.py` | `_STATE_POINTER`, `STATE_MISSING_LOCATION_REASON` (new) | the IR pointer and the IR code |
| same | `_row_failures` | an IR `biological_state_missing_subcellular_location` on `/biological_states/<i>` names that state (IR state i is payload state i, matched by name) |
| same | `_apply_per_row_failures` (`names_state`, `drop_state`, new closures) | the state is removed. Each element placement in it is removed. A process with any `*biological_state` field naming it is omitted whole (`quarantined_broken_reference`). All of these are recorded in `removed_entity_report.removed_entities`, with the reason and `gate: pwml_ir_validation` |
| `tests/test_orch788_state_without_location.py` | new | 2 G9 tests + 1 preservation test, on the contract-alignment toy payload |
| `docs/pwml_recovery_sprint/evidence/orch788_replay.py` | new | offline replay of the leg through the ORCH-768 harness's `replay_leg`, strict mode |

The omission runs inside `quarantine_and_close`, before `freeze_canonical_payload`, so merge rule 8 holds. The IR
validation is not touched. **Repair:** the IR already places a state named exactly after a resolved location
there (`ir.build_pwml_ir`, "If the state is named after one component"). Such a state never raises this error,
so no new repair was added. `test_preservation_a_state_named_after_a_resolved_location_is_kept` covers it.

Why a process in the state is omitted whole rather than losing only the state: `ir.resolve_state` returns the
**first** state of the pathway for an empty reference. Stripping the reference would place the process in a
location the paper never gave it. On this leg, no process is affected.

## 3. Proof (run 2026-10-28, after the reboot; every job bounded, heavy lock `ORCH-788`, 0 survivors each)

My first attempt was blocked: free RAM stayed at 0.75 to 1.38 GB for about 60 minutes, below the 2 GB floor, so I ran nothing.
After the reboot, free RAM was 13.5 GB.

| # | job | result | report |
|---|---|---|---|
| 01 | replay, base `22538dee` | quarantine ok; export **fails**: IR `biological_state_missing_subcellular_location` (blocking and validation); no PWML; coverage 0/18 | `evidence/g11/ORCH-788/01-replay-base.{json,log}` |
| 02 | replay, tip `89038702` (src identical to `08552838`) | export ok; **`pathway.review_required.pwml` written**; IR errors none; coverage 10/18. "Entities left out (4)": the 2 states and the 2 placements, each `(KI-015); gate pwml_ir_validation`. No process omitted | `evidence/g11/ORCH-788/02-replay-tip.{json,log}` |
| 03 | G9 on base (test file copied into `C:/t/orch788/base/tests`, base `pinned_pytest`) | **2 failed, 1 passed**. Both `test_g9_*` fail on the value: `_refusals` = `ir:biological_state_missing_subcellular_location`. Preservation passes | `evidence/g11/ORCH-788/03-g9-base.{json,log}` |
| 04 | tip: `test_orch788_*`, `test_orch773_omit_per_row.py`, `test_strict_quarantine_contract_alignment.py`, `test_orch772_pwml_coverage.py` | **74 passed** | `evidence/g11/ORCH-788/04-tip-g9-and-affected.{json,log}` |

The replay summary is in `evidence/orch788_replay.txt`. Coverage 10/18: all 10 processes of the canonical
payload are drawn. The 8 not drawn are the leg's own recorded pre-freeze removals; this fix changes none of them.

**Pinned baseline moved deliberately.** `test_orch772_pwml_coverage.py::test_the_code_table_is_the_catalogue_appendix`
pinned the catalogue at 83 codes; it is now 84. The new code is `biological_state_missing_subcellular_location` -> KI-015,
added by the orchestrator in `89038702`. Commit `08552838`.

## 4. Dead ends

- **First tip run: 3 failed, 71 passed.** Two failures were the G9 toy. Its compounds carried no `db_status`, so
  `build_pwml_ir` raised `PWML_IR_COMPOUND_VERDICT_MISSING` inside the quarantine's per-row check, and that check blames
  no row. The toy now runs `helpers_prefreeze.prefrozen` first, as production does. The third failure was the 83 -> 84 pin above.
- `python -` was run once by mistake (stdin, which the brief forbids). It hung and was stopped with TaskStop. No process survived.
- A scratch `C:/t/orch788/inspect.py` shadowed the stdlib `inspect` module for a debug script in the same directory.
  Debug scripts now live in `C:/t/orch788/dbg/`.

## 5. Open questions for the product owner

1. **Release cap.** A state or placement omission with no process omitted counts as an "entity-only removal".
   It does not add `omitted_from_pwml` and does not cap the release (existing ORCH-773 rule). This leg is
   `review_required` anyway. Should a lost placement cap the release? That would be one more counter in `quarantine_and_close`.
2. **The KI id is KI-015 (catch-all).** A dedicated KI entry for "state with no location" is optional; adding one costs
   one appendix row plus one `reaction_evidence` tuple.
