# ORCH-716 — non-vacuity audit of the two cap tests: RESULTS

Predictions were written first, in `orch716_nonvacuity_predictions.md`, and are unedited.
Auditor: the Lead Orchestrator. **I did not write either test file**, which is what F-144
requires of a non-vacuity check.

**Verdict: BOTH FILES ARE NON-VACUOUS. F-142's no-coverage-gap conclusion stands, and no
test-only correction is chartered.**

## Measured

Pre-mutation `src/t2pw/pipeline/release_status.py`
= `sha256:db93e6f4fe30632d33725764aba668d31bfa5431f224550626f04888f0bac32d`.
Restored byte-exact from a saved copy after **every** mutation and re-verified against that
digest each time (D-084). `git diff -- src/t2pw/pipeline/release_status.py` is empty at the end.
`git checkout --` was never used.

| Run | Mutation | Result | G11 report |
|---|---|---|---|
| baseline | none | **42 passed** | `ORCH-716/03-nv-baseline.json` |
| M1 | `MIN_CONNECTED_CORE_REACTIONS` 2 -> 1 | **14 failed, 28 passed** | `ORCH-716/04-nv-m1-constant.json` |
| M2 | arm A application forced false, **constant left at 2** | **13 failed, 29 passed** | `ORCH-716/05-nv-m2-arma-neutered.json` |
| M3 | C-072 application (`release_status.py:1087`) forced false | **5 failed, 37 passed** | `ORCH-716/06-nv-m3-c072-neutered.json` |
| restore | none | **42 passed** | `ORCH-716/07-nv-restore-verify.json` |

Every run: `FINAL SURVIVING COUNT : 0`, `cleanup : success`, heavy lock `ORCH-716`
acquired and released.

## M2 is the finding

M2 is the mutation that separates *pinning a constant* from *pinning a behaviour*. The constant
stayed at 2, so anything asserting only `MIN_CONNECTED_CORE_REACTIONS == 2` still passed —
and **13 tests went red anyway**, among them:

```
test_interactions_never_clear_the_connected_core_floor[2|3|5]
test_the_demoted_leg_keeps_everything_and_is_never_diagnostic_only
test_an_unmeasured_connected_core_never_demotes
test_the_two_named_legs_replay_to_review_required[PMC12856317]
test_the_full_corpus_replay_demotes_nothing_it_cannot_justify
test_nonvacuity_c092_losing_a_named_preservation_turns_the_corpus_test_red
test_nonvacuity_c092_a_named_leg_that_stops_firing_an_arm_turns_the_test_red
test_nonvacuity_c092_an_unjustified_arm_b_demotion_turns_the_corpus_red
test_nonvacuity_c092_a_defective_silent_preservation_turns_the_corpus_red
```

`test_c074_strict_core_floor.py` therefore exercises the **production demotion path**, not just
the vocabulary. It already carries four `test_nonvacuity_c092_*` guards of its own, and those
guards themselves went red under M2 rather than staying green — they are guarding the thing
they name.

M3 is the same question for C-072 and gets the same answer: neutering the incomplete-core
demotion at its production site turns four of that file's own tests red, including
`test_the_committed_t104_leg_replays_to_the_contract_outcome`, which replays a **committed real
artifact** rather than a fixture.

## One kept failed measurement

The first M2 attempt ran with `--label nv-m2-armA-neutered`. That label contains an uppercase
`A` and so violates `^[a-z0-9][a-z0-9._-]*$`; `g11_evidence.py next` rejected it and the shell
captured an **empty string**, which reached `bounded_run.py` as `--json ""`. The job executed
and reported `FINAL SURVIVING COUNT : 0` / `cleanup : success`, and produced the identical
`13 failed, 29 passed`, but **no G11 artifact exists for it**, so it is uncertifiable and is not
counted as evidence. It is recorded here rather than deleted: the run that certifies M2 is
`05-nv-m2-arma-neutered.json`, re-run with a valid label against the same mutated tree.

This is the exact trap the charter names — an invalid label silently becoming the `--json`
path. The variant it produced here was an *empty* path rather than error text.
