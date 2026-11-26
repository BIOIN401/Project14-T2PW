# VIS-REF: how curated PathBank pathways look, and how far ours are from it

2026-11-25. PO request: "have we improved the look of the pathways? ... analyze the pathways visually and their code,
we need the arrows to look better, have cleaner flow, as well as using the organelle images that are in PathWhiz".
Reference set from a coworker: `super_pathway_visualizations.zip`, 16 PathBank pathways (PW000001 ... PW751649) in
PWML, SVG (full, greyscale, simple, large font), PNG, SBGN, SBML and OWL. Extracted read-only to
`C:/t/vis-ref/super_pathway_visualizations/` (not committed: 45 MB, PathBank's own files). Renders of 6 of them:
`C:/t/vis-ref/renders/`. Survey script: `C:/t/vis-ref/scripts/survey.py`.

## 1. Answer to "have we improved the look?"

Partly, but not where users look.

| what | improved? | where it shows |
|---|---|---|
| `out/pathway.png`, the image on the web result page | **no**. It is the matplotlib draft graph (`pipeline/draft_graph_render.py`): a grid of separate per-reaction cards, straight thin black arrows, no compartments | the result page (`web/templates/result.html:46`, `views.py:341,377`) |
| DRAW-RENDERER, the connected SVG diagram (crossings -62 %, long edges -77 %) | yes | only the optional vision panel (`vision/web_routes.py:112`) |
| DRAW-PATHWHIZ, PWML layout v2 (PathBank membrane templates, double membranes, nesting) | yes | only when the PWML is opened in PathWhiz |

## 2. What a curated PathBank diagram does (16 PWMLs, 1195 edge paths)

**Arrows.**
- 91 % of edges (1088 of 1195) are cubic Beziers `M x y C c1 c2 end`; 81 % are a single segment.
  The tangents are horizontal or vertical at the node and bend into the reaction centre.
- Stroke 9 px, PathBank gold `#ffc500`; transport edges green.
- Arrowheads are open chevrons, two 30 px legs at about +-30 degrees in the same 9 px stroke (SVG `arrow_path`), not filled
  triangles. The PWML encodes them as `option:start_arrow` / `option:end_arrow` (+ `_path`, + `flat_` variants) on
  each edge: start_arrow true on 596 edges, end_arrow true on 66.
- One reaction is a fork: one main shaft, with co-substrates (ATP, NAD(P)H, tRNA, H2O) curving *into* it and
  co-products (AMP, PPi, ADP, NAD(P)+, H+) curving *out* of it. Small pills sit just above or below the shaft.

**Flow.**
- The main chain sits on one straight row or column; branches join with right-angle connectors.
- Cycles run around a rectangle (TCA PW000779).
- Membrane crossings are one straight vertical line through the transporter, which sits on the membrane band
  (PW000001 pyruvate carrier).

**Organelle and cell images.**
- These are PWML `<drawable-element-location>` records: `drawable-image-id`, x, y, `scale-x`/`scale-y`, `degree`,
  `zindex`, plus the image's native width and height. PathWhiz draws the art from its own library.
- 38 distinct images across the 16 files. The most used ids:

  | id | uses | native size | image |
  |---|---|---|---|
  | 146 | 18 | 512x512 | — |
  | 1 | 15 | 490x240 | mitochondrion |
  | 21 | 9 | 400x400 | — |
  | 14 | 8 | 327x267 | nucleus |
  | 3 | 7 | 280x360 | rough ER, often rotated |
  | 8 | 7 | — | — |
  | 9 | 7 | — | — |

  The bacterium icon, bacterial chromosome and chloroplast are among the others; the names come from the PathBank DB.
- The pattern:
  - a small cluster of organelle images sits in the cell;
  - a translucent `zoom-visualization` wedge (36 in the 16 files) runs from one organelle image to a large detailed
    compartment, which holds the reactions;
  - bacteria get the bacterium icon with a wedge into the full Gram-negative envelope (outer membrane,
    peptidoglycan, periplasm, inner membrane), plus a chromosome image in the cytoplasm (PW000779);
  - plants get cell wall + plasma membrane (PW012911, chlorophyll a).
- Membrane templates seen: 1, 84, 91, 92, 93, 94, 237, 238, 291, 304. Layout v2 already maps compartments to these.

## 3. What our PWML has (server job `item-6`, enterobactin, deploy/local `9b1ea3d1`)

| feature | PathBank | ours |
|---|---|---|
| edge geometry | cubic Beziers that bend into the reaction centre | cubic syntax with control points **on the straight line** (`M305 804 C358 804 411 804 465 804`), so every edge is straight; one is zero length (`M540 839 C540 839 540 839 540 839`) |
| arrow options | `option:start_arrow` / `end_arrow` on every edge | **none** |
| co-substrates and co-products | small pills on curved side branches of one shaft | full nodes on long diagonal straight lines; the ATP edge crosses the EntE reaction |
| main chain | one straight row | zigzag |
| transport | one straight line through a transporter on the membrane | right-angle loops crossing the membrane; spare TolC and TonB boxes outside the cell |
| organelle or cell images, zoom wedge | yes | **none** |
| cell envelope for a Gram-negative bacterium | 4-layer envelope | one plain membrane (template 1) |

Render of our PWML from its own coordinates (`pwml/render.py`): `C:/t/vis-ref/renders/ours-ent-item6.png`.

## 4. Plan (dispatched 2026-11-25, PO request; geometry and visualization only)

Content rule as in DRAW-PATHWHIZ: every compound, protein, reaction, transport, location, state and id stays identical.
Only geometry, visualization records and the preview image change. This is proven by the existing full content-equality
test over the replay PWMLs, plus the golden and SMOKE gates.

| task | what | files |
|---|---|---|
| VIS-1 | PathBank edges and flow in PWML layout v2: real Bezier curves with PathBank tangents; arrow options; co-substrate and co-product side pills on one shaft; straight main chain; straight transport through the membrane; no zero-length edges or stray transporter boxes | `src/t2pw/pwml/layout.py` (+ tests) |
| VIS-2 | organelle and cell images in PWML: `drawable-element-location` organelle cluster per cell type, `zoom-visualization` to the detailed compartment, Gram-negative envelope and chromosome for bacteria, cell wall for plants; image-id catalog from the PathBank DB (read-only) | `src/t2pw/pwml/` (after VIS-1) |
| VIS-3 | result-page image: replace the card-grid PNG with a PathBank-style render of our own PWML (gold Bezier arrows with chevrons, compound boxes, enzyme ellipses, membrane bands, organelle art). Checked by rendering the 16 reference PWMLs and comparing with their PathBank SVGs | new renderer module + `service/post_run.py` hook |

## Open questions
None yet. Each task gets an independent ws-reviewer and a before/after visual check before any merge.
