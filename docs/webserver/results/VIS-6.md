# VIS-6: switch for the PathBank-style result image

Branch `vis/result-switch` (from `vis/render-structures` 95d05949). Code commit d373b28a.

## Changes
- `service/pathway.py` `_pathbank_png`: passes `structure_budget_s` to `render_pwml`: `None` (renderer default) with no deadline,
  else `max(0, min(structures.default_budget(), deadline - now - 5))`; 0 = cache only. No change in `render/`.
- Credits marker: when the PathBank render drew the image, an empty file `pathway.png.pathbank` is written beside
  `pathway.png`; `views.item_view` sets `downloads.image.pathbank` from it. Flag off: no marker, page byte-identical.
- `result.html` + `figure(..., link=false)` macro option: PathBank image is wrapped in a link to itself (`target=_blank`);
  caption: "Rendered from the PWML above. Click to open full size. Organelle and cell art: PathBank / PathWhiz. Structures: PubChem."
- SVG download: skipped (needs a second stored artifact and a new route in `views.py`; not small).
- Tests: `tests/test_web_app.py` (page on/off), new `tests/test_service_result_render.py` (budget default / shrinks / 0, flag off). No network.

## Image file name (review point)
The image name is NOT changed: it stays `pathway.png` in both modes. I first renamed it to `pathway_pathbank.png`, which broke
the existing `test_render_pathbank.py::test_run_pathway_writes_the_pathbank_render_with_the_flag_on` (asserts `IMAGE_NAME`; outside
my test boundary), so I reverted to the marker file. Readers of `pathway.png`: `IMAGE_NAME` in `service/pathway.py` and that test only;
downloads go through `result["image_path"]`. The marker is a new extra file in the out dir; it is not in `report_paths`, so it is
not offered for download.

## Gates (queue, 0 survivors each)
- Focused (`test_service_run_pathway`, `test_service_result_render`, `test_web_*`, `test_render_*`): 389 passed. Evidence `evidence/VIS-6/focused.log`.
- suite2 on d373b28a: SMOKE 511, golden 2/8/4/24, equivalence 8/14/6, vision 370 passed + 1 skipped, web 85: all passed, survivors 0. `evidence/VIS-6/suite2.log`.
- A first run (rename version) had 1 failure (above), then the machine rebooted; both gates were re-run on the final code.

## Not done
- Screenshot of the result page: not produced (no cheap fixture-to-page path outside the test client).

## Open questions
1. Marker file beside the image vs. a flag in the persisted result (would need `jobs.result_to_dict`, outside the boundary).
2. Server needs outbound HTTPS to PubChem and a writable `T2PW_STRUCTURE_CACHE` (VIS-4 q3).

## Review verdict and merge (orchestrator, 2026-11-26)

- Independent ws-reviewer on `127b4b0e` (code `d373b28a`): **APPROVE WITH NOTES**. F1: with the flag off the page is
  not byte-identical as claimed above; the new `if/else` tags add two newlines (no visible change; the `figure` macro
  itself is byte-identical). F2: `views.py:343` spells out `pathway.png.pathbank` instead of importing
  `pathway.PATHBANK_MARKER`. F3: no `run_pathway` test of the marker life cycle; an `OSError` from `touch()` after
  the PNG is written would drop the image (flag on only, job does not fail). F4: `focused.log` has no HEAD stamp.
  All four go to a follow-up; none blocks.
- Merged as `3e0cb627` = webserver/refactor `aa34af70` (VIS-2 merged) + `vis/result-switch` `127b4b0e`, no
  conflicts, in the detached worktree `C:/t/merge-vis6`. Lane flags and `T2PW_RESULT_RENDER` unset.
- Gates on `3e0cb627` (`evidence/VIS-6/logs/merge-vis6-gates.log`, script `merge_vis6_inner.sh`, tree clean):
  SMOKE **511 passed**; golden seams **2 / 8 / 4 / 24 passed**, unedited; equivalence **8 / 14 / 6 passed**;
  vision **370 passed, 1 skipped**; web **85 passed**; pwml **311 passed**; render **49 passed**; service
  (`test_service_run_pathway.py`, `test_service_result_render.py`, `test_web_*`) **350 passed**. `BW_EXIT=0`,
  **FINAL SURVIVING COUNT 0**.
- Two docs-only commits had landed on webserver/refactor meanwhile; merged in (`482b3096`, no file outside `docs/`
  changed) and pushed as `482b3096`.
