# VIS-3: PathBank-style result image, drawn from the run's own PWML

2026-11-25. Branch `vis/result-render` from `webserver/refactor` `7bf8588e`. Code commit `8893f917`.
PO request (VIS-REF): pathways must look like curated PathBank diagrams. Plan: `docs/webserver/results/VIS-REF.md` § 4.

## 1. What was built

| file | what |
|---|---|
| `src/t2pw/render/pathbank.py` (new, ~400 lines) | `render_pwml(path, "png" \| "svg", max_px=2400) -> bytes`. Draws a PWML from its own geometry in PathBank's look. One matplotlib `Figure` (no pyplot state) gives both PNG and SVG. |
| `src/t2pw/render/pathbank_art/` (new, package data) | 41 PathBank drawable images: `<id>.png` runtime rasters, `svg/<id>.svg` PathBank's original vector art byte for byte, `catalog.json` |
| `src/t2pw/service/pathway.py` (+17) | `_pathbank_png()`; the image written as `out/pathway.png` is the PathBank render when the flag is on and the render works, else the old draft-graph bytes |
| `pyproject.toml` (+1) | package data for `t2pw.render` (outside the declared boundary, one line, needed if the server installs the package as a wheel; without it the art would be silently missing) |
| `tests/test_render_pathbank.py` (new, 18 tests) | parsing, chevron geometry against PathBank's own `arrow_path`, both arrow-option encodings, hidden/unknown-art handling, zero-length edge, PNG and SVG output, size cap, the committed `reference/PW000001.pwml`, catalog vs files, flag off/on/fallback |

**Flag.** `T2PW_RESULT_RENDER=pathbank` (case and spaces ignored), read on every call; default off. Flag off: `_pathbank_png`
returns `None` without importing the renderer and `png = None or state["refinement_graph_bytes"]` is the old value,
so the written bytes are identical (test `test_flag_off_never_calls_the_renderer` makes the renderer fail if called).
No PWML, a missing file, a broken file or any exception in the renderer: `None`, a log warning, and the draft graph is
written as before. `refinement_graph_bytes`, `image_path`, `IMAGE_NAME` and the template are unchanged; the caption
"Rendered from the PWML above." becomes true with the flag on, so `result.html` was not touched.

**Why the hook is in `service/pathway.py`, not `post_run.py:97`.** `open_refinement_review` runs before the PWML exists
(the PWML is written at the "Generate PWML" step). The first place that has the run's PWML file is the tail of
`run_pathway`, where `out/pathway.png` is written (`pathway.py:576`), via `result.pwml_path`.

## 2. How each PWML record is drawn

Styles were mined from the 16 reference pairs (`evidence/VIS-3/scripts/mine.py`, output `mine.out`): for each record,
the PWML template id joined to the element PathBank's SVG draws for it (by location id, or by identical path for edges).

| PWML record | drawn as | source of the style |
|---|---|---|
| `edge` | its `path` exactly (matplotlib CURVE4), stroke 9, colour by template: 5 gold `#ffc500` (591 edges), 83 green (87), 148 red, 149 cyan, 195/218 thin black. A template-5 edge that a transport (or an Inhibition / Activation interaction) uses is drawn green (red / cyan), as PathBank's own templates 83/148/149 do | mined |
| `option:start_arrow` / `end_arrow` (PathBank) or the JSON `options` child (T2PW) | open chevron at that end: two 30-unit legs at +-30 degrees from the path's end tangent, same colour and stroke. Reproduces PathBank's `arrow_path` to 3 decimals (test). `*_flat_arrow`: a 30-unit bar across the end | PW000001.svg `arrow_path` |
| `membrane-visualization` | its `path`, band of the template's colour and width (1 `#4BC8F2` 50; 84 red; 91 `#EE9189`; 92 `#AED02A`; 93 `#FEF935`; 94 `#DDB0FD`; 237 `#389F3C`; 238 `#7CC4B7`; 304 `#ABF7B1`; 291 15 wide; 220 black 6; 95, 146 from PATHBANK-CONVENTIONS); unknown template = template 1, as PathWhiz does | mined + `docs/pwml_output/PATHBANK-CONVENTIONS.md` |
| `compound-location` | >= 90x80: rounded box, blue header with the name, white body (no structure image). Template 9: small blue cofactor tag. 78x78 icon templates: dark-blue rounded square with the short name. Else a currency pill (ATP-style ochre, Pi/PPi violet, CoA red) with the short name | mined (templates 9, 42-65, 81, 82, 85, 137-140, 157) |
| `protein-location` | green ellipse; templates 6/77/113 two stacked, 16 three, 7/8/17/164 four; 136 a cloud of five (multi-enzyme complex); 76/80 rounded rectangle (transporters, purple); colours per template (97/113/164 orange, 98 red-orange, 99 dark green, 76/77 purple). Label = protein name, or gene name for `label-type` gene, none for `none` | mined |
| `nucleic-acid-location` | yellow; tRNA head-and-stem when >= 100 high, else a rounded box | mined (template 96) |
| `element-collection-location` | hatched grey box (37) or ellipse (144) | mined |
| `sub-pathway-visualization` | template 14 cyan ellipse, else (217) white rounded box, named by the sub-pathway's `altext` ("Sub-pathway" when it has none) | PW000001, PW000017 SVGs |
| `label-location` | white rounded tag (x, y, scale), stroke 3 (template 235) or 1 (15) | mined |
| `zoom-visualization` | 25x20 marker at p1, translucent `#aabbff` wedge, large box p2..p3 filled with `color` (or none), white strokes. Wedge side chosen as PathBank does (top, bottom, left, right; rule read off every `zoom_box` in the reference SVGs, 36 zooms in the 16 PWMLs) | SVG `zoom_box` |
| `drawable-element-location` | PathBank's image for `drawable-image-id` at `translate(x,y) rotate(degree) scale(sx,sy)`, native width x height. Unknown id: nothing drawn | art catalog § 3 |
| background | PathBank dark blue `#000099` (always; our PWMLs say `#FFFFFF`) | SVG `background` |

Draw order: `zindex`, then document order. PNG is capped at 2400 px on the longest side (PathBank canvases are
2200-9000 wide; our replays up to ~7700).

## 3. Art catalog (41 images)

Extraction (`scripts/extract_art.py`): for every `drawable-element-location` in the 16 PWMLs, the SVG `<g id="location">`
whose transform starts with `translate(x,y)` and holds an `<image>` of the same width and height; the image's base64
data URI decoded. **All 220 uses matched, every id had exactly one distinct image** (no conflicting variants). 36 are
SVG, 5 are PNG in PathBank's own files (139, 146, 147, 204, 247). No SVG rasteriser is installed (cairosvg absent,
none added), so the SVGs were rasterised once, transparent, at native size with headless Edge (`scripts/raster_art.py`)
and stored as 512 px palette PNGs (`scripts/build_art_pkg.py`): rasters 450 KB, SVG originals 1.6 MB.
Names are read off the images (contact sheet `evidence/VIS-3/art-contact-sheet.png`), not from the PathBank DB.

| id | image | native size | original | used in |
|---|---|---|---|---|
| 1 | mitochondrion | 490x240 | svg | PW000001, PW000002, PW000003, PW126614, PW127049, PW127096, PW127303 |
| 3 | rough endoplasmic reticulum | 280x360 | svg | PW000001, PW000002, PW000003, PW126614, PW127049, PW127096, PW127303 |
| 4 | lysosome-like vesicle (hexagon) | 399x340 | svg | PW000002 |
| 5 | peroxisome / granule | 285x320 | svg | PW000002 |
| 8 | kidney with adrenal gland | 292x400 | svg | PW000017, PW127049, PW127096, PW127122, PW127135, PW176557 |
| 9 | liver | 200x145 | svg | PW127049, PW127096, PW127122, PW127135, PW127303, PW176557 |
| 10 | heart | 176x216 | svg | PW127049 |
| 11 | blood vessel network | 580x250 | svg | PW000017, PW127122, PW127135, PW176557 |
| 12 | red oval cell | 177x139 | svg | PW126614 |
| 13 | neuron | 342x731 | svg | PW126843 |
| 14 | nucleus | 327x267 | svg | PW000001, PW000002, PW000003, PW126614, PW127049, PW127096, PW127303 |
| 16 | brain | 500x400 | svg | PW000017, PW126614, PW127049, PW127303 |
| 18 | muscle | 423x333 | svg | PW127303 |
| 21 | inhibition sign | 400x400 | svg | PW127049, PW127096, PW127303 |
| 27 | intestinal epithelium | 800x675 | svg | PW127096, PW127122 |
| 30 | digestive tract | 292x758 | svg | PW127096, PW127122, PW127135 |
| 38 | lungs | 510x652 | svg | PW127049 |
| 47 | cross mark | 102x101 | svg | PW126614, PW126843, PW127049, PW127096, PW127303 |
| 48 | accumulation burst | 400x274 | svg | PW127096 |
| 51 | high concentration burst | 400x274 | svg | PW126843, PW127049, PW127096 |
| 52 | low concentration burst | 400x274 | svg | PW126614, PW126843, PW127049 |
| 73 | DNA strand | 382x58 | svg | PW127049 |
| 75 | lightning | 800x800 | svg | PW126843 |
| 90 | cell organelle cluster | 484x497 | svg | PW000017, PW127049, PW127122, PW127135, PW176557 |
| 91 | plant cell organelle cluster | 543x497 | svg | PW012911 |
| 92 | bacterial chromosome | 479x481 | svg | PW000779, PW000921, PW000923 |
| 97 | pituitary gland | 220x271 | svg | PW127049 |
| 99 | blood vessel (wide) | 861x647 | svg | PW126614, PW127049 |
| 108 | red blood cell | 600x500 | svg | PW127096 |
| 139 | cell cycle | 2159x1950 | png | PW127049 |
| 143 | blood vessel (horizontal) | 2686x343 | svg | PW127096, PW127122, PW127135 |
| 146 | bacterium | 512x512 | png | PW000779, PW000921, PW000923, PW127049 |
| 147 | cheese (food) | 512x512 | png | PW127096 |
| 149 | head | 193x193 | svg | PW127096, PW127122 |
| 180 | bile duct and gallbladder | 400x567 | svg | PW176557 |
| 204 | generic cell | 435x433 | png | PW127049 |
| 205 | crystals | 320x315 | svg | PW127049 |
| 217 | capsule (drug) | 256x109 | svg | PW126614, PW126843, PW127122 |
| 218 | syringe | 360x360 | svg | PW127122, PW127135 |
| 221 | blood vessel (vertical) | 343x2686 | svg | PW127096, PW176557 |
| 247 | brain (sagittal) | 306x278 | png | PW126843 |

## 4. Visual check

Side-by-side pages (scratch, not committed): `C:/t/vis-pw3-work/sbs/index.html` (PathBank's SVG rendered by Edge vs
ours, 16 pathways) and `C:/t/vis-pw3-work/sbs-ours/index.html` (current card-grid result image vs ours, 6 of our
PWMLs: server job item-6 enterobactin and replay runs 7, 76, 116, 119, 134 from
`C:/t/lane-pwdraw-work/replay/mergepw/pwml`). Composite images: `C:/t/vis-pw3-work/sbs/*.png`,
`C:/t/vis-pw3-work/sbs-ours/*.png`. Committed copies: `evidence/VIS-3/sbs-PW000001.png`, `sbs-PW000017.png`,
`sbs-PW000779.png`, `sbs-PW012911.png`, `ours-ent-item6.png`, `ours-replay-116.png`, `ours-replay-7.png`.

All 16 references render (0.3-1.5 s each, PW127049 the largest). Read by eye, every one is recognisably the same
diagram as PathBank's: same membranes and bands, same organelle and organ art in the same places and rotations,
same zoom wedges, same gold chevrons, same green transport and red/cyan interaction edges, same labels.

**What still differs from PathBank**
- Compound boxes have an empty white body: PathBank draws the structure image (from its moldb cache). Not available offline.
- Drug compounds (template 157/158) get PathBank's pink header in PathBank; ours stay blue (see Open questions).
- Pills are flat colours, not PathBank's radial gradients; PPi/Pi and the 78x78 small-molecule icons are generic shapes
  with the short name, not PathBank's little molecule drawings.
- Text: DejaVu Sans, auto-shrunk to fit, instead of Verdana; long names wrap a little differently.
- The tRNA shape is a simplified head-and-stem; the protein-complex cloud (136) and receptor (80) are approximations.
- A sub-pathway without `altext` shows "Sub-pathway" (PathBank shows the referenced pathway's name, not in the PWML).
- Organelle art is raster (512 px) in both PNG and SVG output.

**Our PWMLs.** The picture is now a single connected diagram with the cell membrane, compartment bands per template,
gold Bezier edges and chevrons, instead of the separate per-reaction cards. Everything that still looks wrong in it comes
from the PWML geometry, exactly as VIS-REF § 3 found: straight edges, zigzag main chain, small nodes on a very wide
canvas (text is small once a 7600-wide canvas is scaled to 2400 px), transport loops, spare TonB/TolC outside the cell,
no organelle art. Those are VIS-1 and VIS-2; this renderer will show their fixes without change.

## 5. Gates

All through `C:/t/vis-runs/queue.sh` holder `vis3` (bounded wrapper, heavy lock), Git bash by full path.

| gate | result | log |
|---|---|---|
| focused: `tests/test_render_pathbank.py` | **18 passed** | `C:/t/vis-runs/logs/vis3-focused1.log` |
| focused: service tail + web job/worker (`test_service_pathway_interface`, `test_service_run_pathway`, `test_service_driver_bridge`, `test_batch_driver_run_pathway`, `test_web_jobs`, `test_web_worker`, `test_web_postrun`, `test_web_retry`) | **253 passed** | same |
| render batch: 16 references + 6 of ours (PNG), 2 SVG | 24 of 24 rendered, 0 failures | `C:/t/vis-runs/logs/vis3-renders.log` |
| SMOKE | **511 passed** | `C:/t/vis-runs/logs/vis3-suite2.log` |
| golden seam (batch / c011 / stage8 / quarantine), unedited | **2 / 8 / 4 / 24 passed** | same |
| equivalence (run_pathway / app-vs-service / default path) | **8 / 14 / 6 passed** | same |
| vision | **370 passed, 1 skipped** | same |
| web (app, review reasons, browser fixes, design system) | **83 passed** | same |

Every run: `BW_EXIT=0`, **FINAL SURVIVING COUNT 0**. The suite ran on HEAD `8893f917` (the code commit). The focused
run was taken on the working tree just before that commit, with the same file contents (committed unchanged).
No golden or other existing test was edited. No biology module changed (`git diff 7bf8588e..8893f917 --stat`: only
`pyproject.toml`, `src/t2pw/service/pathway.py`, new `src/t2pw/render/`, new test).

## 6. Dead ends

- `t2pw.pwml.render.build_id_to_name` keys names by element id only, so a compound and a protein with the same id
  overwrite each other; the renderer keys by (kind, id) instead. (`pwml/render.py` not changed; reused only
  `read_xml_with_repairs`, `svg_path_to_mpl_path`, `find_first_visualization`.)
- 1024 px RGBA rasters were 2.4 MB; 512 px RGBA still 2.2 MB; palette quantisation brought them to 450 KB with no visible loss.
- PathBank's own `option:*_arrow_path` values use 15-unit legs; PathBank's SVG draws 30-unit legs, so the chevron is
  computed from the tangent rather than copied from the PWML.

## 7. Open questions (product owner / orchestrator)

1. **Our layout writes compound template 157**, which PathBank uses for *drug* compounds (pink header, 15 of 16 uses in
   the references are drugs). In PathWhiz our compounds may therefore all show as drugs. Options: (a) VIS-1/VIS-2 switch
   ordinary compounds to 81 (PathBank's standard 200x190 box, 244 uses), cost: one constant in `pwml/layout.py`, needs
   the usual content/golden gates; (b) leave it. This renderer keeps every box blue either way.
2. **Switching the server on** is the orchestrator's call: set `T2PW_RESULT_RENDER=pathbank` in the server env. No other
   change needed. Optionally also write `pathway.svg` beside the PNG (the renderer already produces SVG); not done, since
   nothing on the page uses it yet.
3. Windows checkouts with `core.autocrlf` turn the art SVGs' LF into CRLF; the repo blob is PathBank's original bytes
   (`catalog.json` `original_sha256`). Harmless for rendering (runtime uses the PNGs).
