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
