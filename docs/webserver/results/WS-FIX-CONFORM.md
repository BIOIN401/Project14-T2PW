# WS-FIX-CONFORM: retarget the stale conform wiring test

Branch `ws/fix-stale-conform-test`, code commit `3ad49853` (from main `e17b78ba`). Test-only change. 2026-11-21.

## Why

`tests/test_rag_seed_entity_reimport.py::test_the_orchestrator_passes_the_merge_base_to_conform` failed on main
(`22646bdc`, `c852579d`, `e17b78ba`, identically) with "the S3 seam is the only caller; found 0". It parsed
`src/t2pw/app/streamlit_app.py` by AST, but the R2 refactor (`0941cd5d`, 2026-11-05) moved the
`conform_rag_additions_for_merge` call to `src/t2pw/service/rag_orchestration.py:896` (`rag_round_merge`). The test is
not in SMOKE, so nothing caught it until the accuracy-lane merge gates ran all `test_rag_*.py`.
Production wiring was correct throughout: conform receives `graph`, and `merge_site(graph, envelope, ...)` merges into
the same object.

## Change

The test now scans every `.py` under `src/t2pw`: exactly one call to `conform_rag_additions_for_merge` (also as
attribute access), conform's base argument must be the same name as the first argument of the merge call that takes the
envelope. `rag/synthesize.py:1230` mentions conform only in a docstring cross-reference. No `src/` change, no other test.

## Evidence

- Mutation proof: a temp copy of `src/t2pw` with conform given `seed_graph` fails the new test with "conform was given
  'seed_graph' but the merge merges into 'graph'"; the real tree passes (`evidence/WS-FIX-CONFORM/conformfix-mutation.txt`).
- Branch: focused 14 passed; SMOKE 511 and golden/equivalence/vision/web chunks green; 0 survivors.
- Merge result into webserver/refactor: see LEDGER row (gates log `C:/t/vis-runs/logs/merge-conform-gates.log`).
- Review: APPROVE WITH NOTES (`evidence/WS-FIX-CONFORM/REVIEW.md`); the edit is legitimate (gate 5).

## Dead ends

None in the fix. On the orchestrator side, the failure was first suspected to be the accuracy lane's; an A/B on main
`22646bdc` and pre-locations main `c852579d` showed it pre-existing.

## Open questions / follow-ups (reviewer notes, non-blocking)

- F1: the test trusts that `merge_site`'s first argument reaches `merge_additions` as its base; it does not check the
  pass-through in `service/extraction.py:1010-1014` (`_rag_merge_site`). Optional: assert that too.
- F2: docstring says the C-060a lift finds the statement "in streamlit_app.py"; since R3d it is in `service/extraction.py`.
- F3: an aliased import or `getattr` call would escape the count (the old test had the same gap).
