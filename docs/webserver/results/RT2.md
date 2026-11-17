# RT2: re-pin two release-era tests to the intended new defaults

Branch `ws/RT1-release-test-pins`, commit `16ec75e2`. The commit message and branch name say "RT1"; the
task ID is **RT2**, because `RT1` already belongs to the LLM-timing task of 2026-11-10 (review finding F3).

## Why
Two tests outside SMOKE were failing on `main` = `bc63f504` after the release. They were found by the
correctness programme's bisect, and diagnosed by a separate read-only bisect agent (sonnet/high) on
detached worktrees `C:/t/bis2-<sha>`. That agent edited no code or tests.

| Test | First bad commit | Cause | Verdict |
|---|---|---|---|
| `test_c052_prefreeze_report_at_the_streamlit_seams::test_new_acceptance_the_post_pipeline_seam_returns_its_prefreeze_report` | `effba332` (ORCH-797a, D-120 s1, 2026-10-30) | `_norm` (`src/t2pw/pwml/prefreeze_resolution.py:130-132`) now uses `reference_key`, which deletes separators | Intended. The test pin was missed. The field is a diagnostic label only. |
| `test_orch768_replay_corpus::test_replay_reproduces_the_recorded_verdict[ORCH-767/sour_taste_v2-strict/2026-10-24_0042]` | `5d1ac2ce` (final payload default on), plus `7d973ce0` (layout v2 default) | The new defaults change the PWML bytes. The function-scoped conftest pin does not reach the module-scoped `corpus` fixture. | Intended (D-W19a, D-W23 a/b) |

Bisect table for the orch768 test: `06a0b8c0`, `c757814b` and `f00a4964` pass. `5d1ac2ce` fails, and passes
with `T2PW_FINAL_PAYLOAD=0`. `bc63f504` fails with either pin alone and passes with both. With both pins,
the PWML sha equals the one from `06a0b8c0`.

## Change (test-only, 2 files)
- **`tests/test_c052_prefreeze_report_at_the_streamlit_seams.py`:** `target` changes to `"glycerol3phosphate"`
  and `source_key` to `"snglycerol3phosphate"`.
- **`tests/test_orch768_replay_corpus.py`:** inside the `corpus` fixture, the replays run under
  `pytest.MonkeyPatch.context()`, which sets `T2PW_FINAL_PAYLOAD=0` and `T2PW_PWML_LAYOUT=v1`. These are the
  same pins as `REPLAY_ONLY_ENV`. `EXPECTED` and the assertions are unchanged.

## Tests
These ran through the bounded wrapper, with 0 survivors. Evidence is in `docs/webserver/evidence/RT2/`.
- Both files: 22 passed (`a.log`).
- SMOKE: 511 passed (`smoke.log`).

The runs used the uncommitted edit, which matches `16ec75e2`. The merge into `webserver/refactor` brings in
no other non-doc change: `git diff 16ec75e2 origin/webserver/refactor -- . ':!docs'` shows only these two
files. So the tree that merges is the tree that was tested.

## Review
`ws-reviewer` (opus/high) gave APPROVE WITH NOTES.
- F1: the c052 re-pin matches the code.
- F2: the env pin does not leak. The comment could also cite D-W19a/D-W23 (a). This is cosmetic and not
  applied.
- F3: task ID collision and missing record. Fixed by this record and the evidence commit.
- F4: the logs match the claims.

## Dispatch
| Step | Agent | Model / effort |
|---|---|---|
| Diagnosis bisect | general-purpose | sonnet / high |
| Implement | general-purpose | sonnet / medium |
| Review | ws-reviewer | opus / high |

## Leftovers
- The 11 bisect worktrees `C:/t/bis2-*` and the 4 older `C:/t/bis-*` remain. Removing them was denied by the
  permission system, and the PO has been told.
- The diagnosis agent's report could not be written to `C:/t/bis2/REPORT.md`. Its content is summarised above.
