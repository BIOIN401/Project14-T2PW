# ORCH-716 — non-vacuity audit of the two cap tests: PREDICTIONS, recorded before execution

**Written before any mutation was applied.** F-144 binds: a non-vacuity guard is not evidence
until a party who did not write it has failed to defeat it. I did not write either test file.

**Audit question (charter § 4).** Do `tests/test_c074_strict_core_floor.py` and
`tests/test_c072_incomplete_core_demotion.py` pin their caps **non-vacuously**? F-142's
no-coverage-gap conclusion rests on them.

**Vacuity, defined for this audit.** A file is *vacuous* if it survives a mutation that
disables the production behaviour it claims to pin. Asserting that a constant still equals
itself is NOT evidence that the constant is *applied*.

## The seams under test

All three caps live in `classify_release_status`, `src/t2pw/pipeline/release_status.py`:

| Cap | Seam | Line |
|---|---|---|
| C-072 — incomplete-core demotion | `if status == RELEASE_READY and verdict is not None and verdict.declared and missing:` | 1087 |
| C-074 arm A — connected-pathway floor | `below_connected_core_floor = (...)` | 1141-1145 |
| C-074 arm B — the unstated request | `request_was_never_stated = (...)` | 1146-1148 |
| shared demotion step | `if status == RELEASE_READY and (below_connected_core_floor or request_was_never_stated): status = REVIEW_REQUIRED` | 1174-1175 |

`MIN_CONNECTED_CORE_REACTIONS = 2` is at `release_status.py:156`.

## Mutations and predictions

Restores are **byte-exact** from a saved copy (D-084): `git checkout --` is not used, because
the working tree carries an unrelated uncommitted `streamlit_app.py` hunk it would not touch
but which I will not risk, and text-mode restore does not reproduce the original bytes.

### M1 — constant change: `MIN_CONNECTED_CORE_REACTIONS = 2` -> `= 1`

- `test_c074_strict_core_floor.py::test_the_reason_vocabulary_is_the_production_vocabulary`
  **FAILS** — it asserts both `== production.MIN_CONNECTED_CORE_REACTIONS` and `> 1`.
- The behavioural assertions pinning the wire string `connected_core_below_minimum:1<2`
  (lines 293, 430) **FAIL** — at a floor of 1 a one-reaction core is no longer below it, so
  no reason is appended at all.
- `test_c072_incomplete_core_demotion.py` — **ALL PASS**, unaffected: different cap.

M1 alone proves nothing about the production path. It is the control that tells me the file
notices the constant at all.

### M2 — application neutered, constant untouched: force `below_connected_core_floor = False`

This is **the real question.** The constant stays at 2, so anything that only pins the constant
still passes.

- `test_the_reason_vocabulary_is_the_production_vocabulary` **PASSES** — by construction.
- **If `test_c074_strict_core_floor.py` is NON-VACUOUS:** at least one behavioural test FAILS,
  because a leg whose connected core is below the floor is no longer demoted to
  `review_required` and no longer carries the reason.
- **If every test in the file PASSES:** the file pins the *vocabulary* and not the *behaviour*.
  That is **vacuous for the purpose F-142 relies on**, and a test-only correction is chartered.

### M3 — C-072's application neutered: line 1087 condition forced false

- **If `test_c072_incomplete_core_demotion.py` is NON-VACUOUS:** its demotion tests FAIL.
- **If they PASS:** vacuous; test-only correction chartered.
- `test_c074_strict_core_floor.py` — expected largely unaffected, but any collateral failure is
  recorded rather than explained away.

## What each outcome licenses

Only a file that survives M2 (respectively M3) **green** is vacuous. A file that goes red under
its own mutation has exercised the production path and F-142 may continue to rest on it.

**No production line is changed by this audit.** Every mutation is reverted byte-exact and the
SHA-256 of `release_status.py` is re-verified after each restore. The pre-mutation digest is
recorded in the results file beside this one.
