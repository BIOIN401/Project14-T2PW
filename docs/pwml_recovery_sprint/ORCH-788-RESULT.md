# ORCH-788 — a biological state with no subcellular location becomes a per-row omission

Branch `orch788/fix` off `22538dee`. Code commit `f6541a25`. **Status: built, NOT TESTED, NOT REPLAYED.**
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

## 3. Proof: NOT RUN

The machine had free physical RAM of **0.75 to 1.38 GB** (`Win32_OperatingSystem.FreePhysicalMemory`, polled
about every 10 s for about 60 minutes). The brief's floor for a short focused gate is 2 GB. Other figures: committed 48 GB of 32 GB,
nonpaged pool 2.5 GB. No sprint heavy lock was held (`C:/t/heavylock` absent), and no T2PW python process was
running. So no test and no replay was started. Still owed:

1. Replay: base `--src C:/t/orch788/base/src` (worktree created at `22538dee`) and tip, via
   `orch788_replay.py` under bounded_run. Expected: base fails at `pwml_export` with the IR code. Tip writes
   `pathway.review_required.pwml`, and "Entities left out" lists the two states and the two placements.
   Save the summary to `evidence/orch788_replay.txt`.
2. G9: `tests/test_orch788_state_without_location.py` on base (expect the 2 `test_g9_*` to fail) and on tip. Then the
   files that directly exercise `strict_quarantine`: `tests/test_orch773_omit_per_row.py`,
   `tests/test_strict_quarantine_contract_alignment.py`. G11 report paths: `evidence/g11/ORCH-788/`.

## 4. Dead ends

- I ran `python -` by mistake (stdin, which the brief forbids). It hung, and I stopped it with TaskStop within 2 minutes.
  No python process survived (checked with `Get-CimInstance Win32_Process`).

## 5. Open questions for the product owner

1. **KI id.** No catalogue entry covers this. The text record will print `unclassified` for the reason
   `biological_state_missing_subcellular_location`. I could not add it myself: `reaction_evidence.KNOWN_ISSUES`
   must equal the `PWML_KNOWN_ISSUES.md` appendix, or `test_orch772_pwml_coverage.py` test 1 fails, and that
   doc is not mine to edit. Proposed: one appendix row (`biological_state_missing_subcellular_location` | exact |
   KI-015 | `strict_quarantine._apply_per_row_failures`; `ir.build_pwml_ir`, `ir.validate_pwml_ir`) and the same
   tuple in `reaction_evidence.KNOWN_ISSUES["codes"]`, landed in one commit. A new KI entry would also do;
   this costs one line in each file.
2. **Release cap.** A state or placement omission with no process omitted counts as an "entity-only removal".
   It does not add `omitted_from_pwml` and does not cap the release at `review_required` (existing ORCH-773 rule).
   This leg is `review_required` anyway. Should a lost placement cap the release? That would be one more
   counter in `quarantine_and_close`.
