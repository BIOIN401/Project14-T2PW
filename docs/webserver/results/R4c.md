# R4c: finish R4 in the equivalence harness, close small carry-forwards

Branch `ws/R4c-harness`, from `4c90ef28`. Test-only and comment-only; no logic change under `src/`.

## What changed

1. `tests/equivalence/harness.py`, service arm. It used to inline the handler's steps and call
   `run_post_pipeline_sbml_artifacts` and `run_pwml_export` directly. It now calls
   `service.post_run.run_post_pipeline` and `service.export.run_export`.
   - Settings still come from the app arm's captured kwargs. They are mapped onto a `PostRunRequest`
     (`mapping_cache_path` -> `mapping_cache_text`, `id_source` -> `id_source_mode`, the rest by name) and onto
     the `state` keys `run_export` reads (`pwml_name`, `pwml_description`, `pwml_subject`, `pwml_width`,
     `pwml_height`, `pwml_bg`, `pwml_grounding`, `pwml_grounding_dict`, `pwml_strict_db`).
   - `open_review` callback: sets only `refinement_working_json` (the one write the export reads).
     The other 12 keys of `initialize_refinement_review_state` are display state.
   - The same assertions are kept. Spies are now installed on the service arm as well (on `postpipeline`).
   - Stricter, never weaker:
     - each spied function must have run exactly once in the service arm (`run_pwml_export` only if the app
       exported), else the arm is reported as failed;
     - the call the service made must equal the app's: same positional args (so the payload handed to the post
       pipeline and the reviewed payload handed to the export are compared) and same keyword settings, except
       the run-produced inputs and the arm-root-bound paths (`qa_report`, `reaction_summary`,
       `rag_evidence_context`, `pathway_context`, `outputs_dir`, `state`, `project_root`, `ref_path`,
       `quarantine_report`, `quarantine_artifacts`);
     - `post_run.failure` is reported as an arm failure (the old code reported an exception the same way).
   - Weaker in one respect, noted: `_on_call` no longer deep-copies the `state` kwarg of the post-pipeline call
     (it holds an `RLock` in the service arm). Nothing read it; the app arm's `post_call` kwargs lose that key.
   - `pwml_strict_db` is set on `state` after `run_extraction`, not before, so the extraction runs under the
     same default as before this change.
   - Call counts are recorded by `_on_call` in `arm["counts"]`.
2. `tests/test_equivalence_app_vs_service.py`: `service["unused"]` instead of `.get("unused", [])` (two uses),
   docstring (c) and the comment at the start of test (c) reworded. Nothing else.
3. Stale names, comments only: `extraction.py` (OWNED_FLOOR 4640 -> 790, "app's two" -> "the two", the
   "has to read this file" line), `postpipeline.py:110` (`_generate_pwml_from_refinement_working_json` ->
   `t2pw.service.export.run_export`).
   - Not changed: `tests/test_streamlit_quarantine_boundary.py:7` has the same stale name; it is a golden-seam
     file, so it is left alone. Also `extraction.py:1036`: I found nothing stale on or near that line (the
     "app does not call this" docstring is still true after R4b), so it was not touched.
4. `requirements.txt`: `starlette>=0.40` next to `fastapi` (`t2pw.web.app` imports it directly). Installed: 1.6.0.

## Results (bounded wrapper, heavy lock R4c, `pinned_pytest --expect-tree`, survivors 0 in every run)

| run | result | exit |
|---|---|---|
| `test_equivalence_app_vs_service` | 14 passed | 0 |
| `test_equivalence_second_fixture` | 8 passed | 0 |
| `test_dependency_declarations` | 3 passed | 0 |
| golden: seam 2, c011 8, stage8 4, quarantine boundary 24 (each file alone) | 38 passed | 0 |
| SMOKE (`smoke.lst`) | 511 passed | 0 |

Evidence: `docs/webserver/evidence/R4c/` (`*.out`, `*.json`).

## Deliberate breaks (scratch detached worktree at the same commit plus these test files)

| break | result |
|---|---|
| Stage-2 additions replaced by `{}` at `extraction.py:876` (the first merge) | 4 failed (all in the second fixture: a, b, b-replay, c). Fixture 1 has 0 additions, so it cannot see it (as in R3d) |
| `infer_rounds=1` in `run_extraction` (one QA round) | 2 failed: `test_c_service_post_pipeline_equals_the_app` and `test_c_the_service_equals_the_app`. This was the R3d survivor `eq-extraction-one-qa-round`; the `service["unused"]` change kills it |

## Incidents

- I ran a bare `python -` once by mistake (empty stdin). It hung, I stopped it, and no python process of mine
  survived (the `python.exe -I -` processes listed belong to another session, E2E3, and were not touched).
- The scratch worktree `C:/t/ws-R4c-scratch` could not be removed (the permission system denied
  `git worktree remove`). It is outside the checkout; remove it with `git worktree remove --force`.

## Open questions

None for the product owner.
