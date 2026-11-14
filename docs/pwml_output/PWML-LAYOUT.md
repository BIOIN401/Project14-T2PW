# PWML LAYOUT (2026-11-13, branch `pwml/layout` from `pwml/final-payload` `3e8a66f3`)

Code: `src/t2pw/pwml/layout.py` (new), `src/t2pw/pwml/writer.py` (one import plus a 2-line hook in `DeterministicPwmlBuilder.build()`), `tests/test_pwml_layout.py`, fixture `tests/fixtures/pwml_layout/ir_run4.json`. Evidence: `evidence/layout/` (scripts, metrics, renders, PathWhiz screenshots, test reports).

Flag `T2PW_PWML_LAYOUT`: unset or `v1` = today's layout, byte-identical; `v2` = new layout. Default stays `v1`; the product owner flips it.

## 0. NOT ESTABLISHED (read first)
1. Harness = empty PathWhiz DB plus vocabulary seed (as in P4). Nothing was imported into production PathWhiz.
2. PathWhiz screenshots show the live viewer (`/live/view`) only. The editor (dragging nodes, re-snapping edges) was not tried on a v2 file.
3. Only the flag-on (final payload) set went through the harness; flag-off v2 was measured on stored coordinates only.
4. The layout is judged by geometry counts and by eye on 8 files. No biologist has judged readability.
5. The renderer `t2pw/pwml/render.py` does not draw labels or bound-visualizations, and sizes its canvas from edge and membrane paths only; that is why v1 renders cut nodes off at the edge (2.3).

## 1. What was wrong, and why (v1, measured on the 97 + 97 replayed files)
How the writer places things (`writer.py` `_populate_sections_from_ir`):
* **Compartments are full-width horizontal stripes of fixed share.** Each biological state gets a region `{x: 0, w: canvas width, h: canvas height / number of states}`, grown when its reactions need more (`writer.py` ~1912-1937). States are stacked even when they hold one node or none, so canvases reach 7380 px high (idx 7, 10 states) and most of the canvas is empty: nodes cover a median 5.5% of the canvas, and the content's bounding box a median 37%.
* **Reactions are laid on a serpentine grid per state** (`_serpentine_reaction_position`, 800 px steps, 700 px rows); substrate and product stacks are placed around each enzyme. Shared intermediates are reused, so a member placed by one reaction is moved again by the next; stacks of neighbouring reactions collide. 228 (flag off) / 255 (flag on) node-node overlaps in 41 / 47 files.
* **Transport and interaction edges keep the IR's coordinates.** Reaction edges are recomputed from the final boxes, but transport member edges keep `edge.path` from the IR (`writer.py` ~2373-2383, never rewritten), interaction edges to a protein complex use the IR location's x/y (`writer.py` ~2726), and the IR's enzyme connector edges (dashed, template 83, referenced by no visualization member) keep IR paths. The boxes they belong to have moved, so these are the lines across the whole canvas. Edges crossing a node box that is not one of their ends: 302 / 311 edges in 57 / 58 files.
* **Sizes.** Many compound locations are written 160x60 while their template (81, "Medium Compound Visualization") is drawn 200x180 by PathWhiz, so PathWhiz draws boxes 3x taller than the layout reserved (more overlaps with template sizes: 241 / 273).
* **Compartment boxes (flag off only)** are bound-visualizations set to the stripe; 190 nodes sit outside their own state's box and 47 boxes hold more than one state. Flag on writes none (P4: PathWhiz cannot import them without a Bound).

## 2. What PathWhiz does with the coordinates (evidence)
1. **Positions are stored and drawn verbatim.** `lib/pwml_parser.rb` copies x/y of every location into the row; the only change is `compute_coordinate_offset` (shifts everything when a coordinate is negative, margin 50; never triggered by T2PW output). The live viewer places each node with `translate(x,y)` equal to the PWML x/y (checked in the saved SVG of idx 128: `translate(109,285)` for compound location x=109, y=285).
2. **Location width/height are ignored.** The `compound_locations`, `protein_locations` (etc.) tables have no width/height columns (`db/schema.rb`); a location is drawn at its visualization template's size (`visualization_templates.width/height`). So the PWML's width/height are only a promise; template 81/157 = 200x180, template 2 (proteins) = 150x70.
3. **Edges are drawn from the stored path.** `edges.path` is stored as written and drawn as the SVG `d` (`templates/svg_visualization/edges/*.jst.eco`); the stored transport path `M1915 989 L2000 969` of idx 128 appears verbatim in the viewer's SVG. Arrow heads are recomputed from the path: the start arrow from the first point and the next command's first control point, the end arrow from the last command's second control point (`edge.js.coffee` `addStartArrow`/`addEndArrow`), so arrowed edges need `C` segments.
4. **The canvas is the PWML's pathway-visualization width x height.** The viewer's `<svg width height>` equals them (idx 128: 3200 x 3910) and the background fills it; the empty area is visible to the user (screenshot `pathwhiz/v1/128.png`).
5. **Compartments in PathBank's own files** (controls `reference/PW012926.pwml`, `PW000001.pwml`) are drawn with membrane-visualizations (paths of `M` + `C` segments; closed rounded rectangles with `complete-membrane` true), drawable images (cell shapes) and label-locations; bound-visualizations are not used. PW012926: canvas 3149 x 1493, compounds 78x78 in a row, enzymes 150x70 below, each edge one cubic curve from compound to enzyme side. A membrane needs only a template id and a zindex (`membrane_visualization.rb`), a label needs template, x, y, zindex, scale (`label_location.rb`); templates 220 "Simple Membrane" and 15 "Simple Label - Small" exist in the seeded template table.
6. **`side` of a reaction member is not layout in PathWhiz**: `Reaction#find_element` reads `reaction_#{side}_elements` and the SBML/SBGN export makes a member a reactant or product by it (found by the reviewer, `reaction.rb:323`, `reaction_visualization.rb:192-480`). v2 leaves it alone (section 7.2 for the v1 defect it exposes).

## 3. The prototype `C:/t/ev-diagram/t2pw_diagrams.py`
Judged not portable as is. Its node model is built from the IR, merges all instances of an entity in a compartment into one node and adds reaction junction nodes; a PWML has one location per reaction member (shared intermediates excepted) whose ids the visualizations reference, so its nodes do not map back 1:1 onto PWML locations without changing visualization elements. It also emits quadratic (`Q`) segments, which PathWhiz's path parser does not read (only M/L/C/Z), and arrows need `C`. Its ideas were reused: compartments as regions, layering by longest path after DFS back-edge removal, grid A* routing with a bend penalty, and the overlap / edge-through-node checks (now `layout_metrics.py`).

## 4. Approach (v2)
`relayout(section_items, ...)` runs after the writer has built every element and id, and changes geometry only:
1. **Nodes.** One block per visible location; the protein locations of one protein-complex visualization form one block (members in rows of up to 3). Compound locations get their template size (`TEMPLATE_DIMS`) written into width/height, because that is what PathWhiz draws.
2. **Processes.** A reaction runs substrate -> enzyme block -> product (the product is the member whose edge carries the start arrow, as the writer writes it); without an enzyme, through a reaction point next to its substrates. A transport runs left compound -> transporter (or a point) -> right compound. An interaction runs left -> right; an undrawable side (a complex visualization with no protein) becomes a point next to the other side. The IR's enzyme connector edges run from their member to the reaction's hub, which is a point when the member is the hub.
3. **Compartments.** Each biological state is one box. Inside it, each connected group is laid out in layers: back edges dropped by DFS, longest-path columns, pure sources pulled next to their successor, 8 barycenter ordering sweeps, then y by least squares alignment with neighbours under a no-overlap constraint (pool-adjacent-violators per column). Groups are shelf-packed inside the box and boxes are packed in rows, both to a target width of sqrt(2 x content area); linked compartments are placed next to each other.
4. **Edges.** Straight between the border anchors of the two blocks when that line crosses no other node box; otherwise an orthogonal A* route on a 10 px grid around node boxes padded by 8 px. Every path is written as straight `C` segments, so PathWhiz can draw arrows; arrow path strings in the options are recomputed from the new path, arrow on/off kept. An edge that nothing drawn is tied to keeps its v1 path.
5. **Compartment drawing.** One closed membrane-visualization (rounded rectangle, template 220, complete-membrane true) and one label-location (template 15, the state's name without " state") per box; bound-visualizations, when written (flag off), are moved onto the same box; a state with nothing drawn gets a small box of its own under the rows.
6. **Canvas** = packed content plus 40 px margins.

Not changed: every id, element, reference, state, location, reaction, transport, interaction, hidden flag, template, zindex, arrow flag and reaction-member `side`.

## 5. Results (97 + 97 replayed files, `evidence/layout/metrics/`)
### 5.1 Identity
* **v1 byte-identical:** flag off 97/97, flag on 97/97 identical to the sha256 recorded when the P4 d3 set was written (`sha_{off,on}_v1_vs_d3.json`). The recorded hashes are used because git's autocrlf rewrites the committed copies with CRLF.
* **v2 content-identical:** `content_check.py` (strips x, y, width, height, edge paths, non-null arrow path strings, and the membrane and label sections; compares the rest as canonical XML): **0 differences** in 97/97 flag off and 97/97 flag on (`content_{off,on}_v2_vs_v1.json`). Same 97 files written in each mode as P4.

### 5.2 Layout quality (`layout_metrics.py`; "PW sizes" = compounds sized by template as PathWhiz draws them)
| metric (97 files) | off v1 | off v2 | on v1 | on v2 | off v1 PW sizes | off v2 PW sizes | on v1 PW sizes | on v2 PW sizes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| node_overlaps | 228 | 0 | 255 | 0 | 241 | 0 | 273 | 0 |
| files_with_node_overlaps | 41 | 0 | 47 | 0 | 42 | 0 | 48 | 0 |
| nodes_out_of_canvas | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| edges_through_node | 302 | 0 | 311 | 0 | 305 | 0 | 314 | 0 |
| edge_node_hits | 540 | 0 | 561 | 0 | 550 | 0 | 582 | 0 |
| files_with_edges_through_node | 57 | 0 | 58 | 0 | 57 | 0 | 58 | 0 |
| edges_out_of_canvas | 0 | 4 | 0 | 4 | 0 | 4 | 0 | 4 |
| long_edges | 44 | 16 | 44 | 29 | 44 | 16 | 44 | 29 |
| edge_crossings | 430 | 85 | 430 | 91 | 430 | 85 | 430 | 91 |
| compartment_boxes | 289 | 304 | 0 | 255 | 289 | 304 | 0 | 255 |
| nodes_not_in_box | 41 | 0 | 0 | 0 | 71 | 0 | 0 | 0 |
| mixed_state_boxes | 47 | 0 | 0 | 0 | 41 | 0 | 0 | 0 |
| nodes_outside_own_bound | 190 | 0 | 0 | 0 | 164 | 0 | 0 | 0 |
| box_overlaps | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| median_canvas_w | 3200.0 | 1500.0 | 3200.0 | 1500.0 | 3200.0 | 1500.0 | 3200.0 | 1500.0 |
| median_canvas_h | 1692.0 | 1210.0 | 1692.0 | 1278.0 | 1692.0 | 1210.0 | 1692.0 | 1278.0 |
| max_canvas_w | 3200.0 | 4240.0 | 3200.0 | 4240.0 | 3200.0 | 4240.0 | 3200.0 | 4240.0 |
| max_canvas_h | 7380.0 | 3539.0 | 7380.0 | 3419.0 | 7380.0 | 3539.0 | 7380.0 | 3419.0 |
| median_fill_pct | 5.19 | 18.32 | 5.46 | 19.02 | 5.77 | 18.32 | 6.37 | 19.02 |
| median_content_over_canvas_area_pct | 37.0 | 74.5 | 37.0 | 76.3 | 37.0 | 74.5 | 37.0 | 76.3 |

Edges out of the v2 canvas (4 per mode): edges no drawn element is tied to keep their v1 path (7.3). Flag-off v2 has 304 boxes against 289 bound-visualizations because a state with nothing drawn keeps its bound-visualization as a small box of its own. Crossings and long edges are reported, not targeted.

### 5.3 PathWhiz import (59 flag-on harness files, same importer as P4, `pwml_parser.rb` sha256 `eb00ccaa...`)
| (59 files) | P4 d3 | v1 (this run) | v2 |
|---|---:|---:|---:|
| clean files (no messages) | 45 | 45 | 45 |
| importer messages | 94 | 94 | 94 |
| of which vocabulary / transport family / nucleic acid / interactions / other | 38/18/24/9/5 | 38/18/24/9/5 | 38/18/24/9/5 |
| transports created / expected | 32/37 | 32/37 | 32/37 |
| interactions created / expected | 13/17 | 13/17 | 13/17 |
| reactions created / expected | | 236/239 | 236/239 |
| files whose messages differ from v1 (exact text) | | | 0 |
| row types created only in v2 | | | membrane_visualizations, label_locations (every one created) |

Every created-row count other than the new membranes and labels is identical per file (`evidence/layout/harness/v1_vs_v2.md`, rows in `*_59_results.jsonl`). The v1 and v2 passes ran on the same importer copy, one after the other, holding the heavy lock.

v1 in this session reproduces P4 d3 exactly (45/59 clean, 94 messages, 32/37 transports; 0 files differ from the P4 rows).

### 5.4 Screenshots and renders (before / after)
PathWhiz live viewer, whole drawing fitted (`evidence/layout/pathwhiz/{v1,v2}/<idx>.png`, control `pathwhiz/control_PW012926.png`); file renderer `render.py` (`evidence/layout/render/<idx>-{v1,v2}.png`). Files: idx 7 (example 01, tetrapyrrole regulation), 128 (example 02, coenzyme A), 9, 116, 134, 126, 58, 10.

What the PathWhiz screenshots show:
* **v1** (e.g. `pathwhiz/v1/7.png`, `128.png`): the canvas is mostly empty blue (idx 7: 3200 x 7380 with the lower half bare), interaction and transport lines run diagonally across the picture and nodes overlap; the renders (`render/<idx>-v1.png`) also show the dashed enzyme connectors left where the enzyme used to be.
* **v2** (`pathwhiz/v2/*.png`): one outlined, labelled box per compartment; each reaction reads left to right (substrate, enzyme ellipse, product); no node on another; lines between compartments are orthogonal and go around nodes; the canvas is close to the content.
* Remaining blemishes in v2 (none of them a regression): a compartment whose proteins the empty-DB importer rejects shows as an empty box (idx 10, 128); an internal state name is shown as a label (`__auto_state__`, idx 10 and 128); a label can touch the coloured header PathWhiz draws above a compound (idx 128); idx 10 (transports only, 9 compartments) is still a tangle of long orthogonal lines.
* The control PathBank file PW012926 (`pathwhiz/control_PW012926.png`) is drawn as before.

## 6. Tests (bounded wrapper, 0 surviving processes, reports in `evidence/layout/tests/`)
TESTS_TABLE

## 7. Open questions for the product owner
1. **Flip the default to v2?** Options: flip now (all quality counts at 0, content identical, harness not worse); or first look at more PathWhiz screenshots / let a curator judge. Cost of waiting: users keep getting v1 pictures.
2. **v1 writes `side` against the member's role for 285 of 1036 reaction members (F2 of the review).** `writer.py` `_reaction_visual_side` flips Left/Right on every second serpentine row; PathWhiz reads `side` as the reaction side (substrate/product in SBML/SBGN export, `Reaction#find_element`). v2 keeps v1's values, so for those reactions the labels PathWhiz reads disagree with the drawing (substrates are drawn left in v2). Options: (a) set `side` from the role (substrate Left, product Right) in the writer: an output change, not layout, needs a ruling and its own before/after harness run; (b) keep as is. Cost of (a): small code change plus a harness pass; it changes import semantics for those reactions (believed to fix them).
3. **Edges tied to nothing drawn** (N1/F3 of the review): 4 visible lines in idx 6 and 2 in idx 146 (flag on) reach outside the v2 canvas; flag off has one in idx 130 whose interaction members point at compound locations 32/33 that the writer never emitted (a dangling reference in v1). Options: hide them (changes `hidden`, not geometry), widen the canvas to contain them (geometry only, leaves stray lines), or fix the writer so they are not dangling (output change).
4. **Sizes of non-compound locations.** Only template 2 (150x70, proteins and nucleic acids) occurs; it matches the stored protein size; nucleic-acid locations are written 160x60 but drawn 150x70. Not adjusted.
5. **Flag-off bound-visualizations** are moved onto the compartment box, which the membrane also draws (two outlines in one place for any viewer that draws both). PathWhiz rejects bound-visualizations without a Bound anyway (IMPORT-HARNESS).
6. **Labels.** v2 labels each box with the state's name minus " state"; internal names show as is (`__auto_state__`), and the label (60 px band at the top of the box) can touch the header PathWhiz draws above a compound. Options: skip labels for names starting with `__` and raise the top padding (geometry/visual only, small change plus re-measure), or leave.
7. **render.py** sizes its canvas from edge and membrane paths only and draws no labels; with v1 that cut nodes off. A one-line fix (include node boxes in the bounds) is outside this package.

## 8. Dispatches and process
| # | task | model / effort |
|---|---|---|
| 1 | independent review of `3e8a66f3..3eee6f81` | opus / high (`ws-reviewer`): CHANGES_REQUESTED (F1 blocking: `side` is semantic in PathWhiz; F2-F10 minor/notes) |
| 2 | re-review of `3e8a66f3..3df33507` | opus / high (same agent, resumed): APPROVE WITH NOTES |

Everything else (fact finding, code, replays, harness, tests) was done by the package owner directly.

Dead ends and slips: (1) The first layout put compartments as full-width stripes with global columns: idx 7 came out 3850 x 9136; replaced by per-compartment layered groups packed in boxes. (2) The first v2 run collapsed edges tied to nothing to a point and recomputed `side`; both reverted after review. (3) The first byte-identity check compared against the committed d3 copies and reported 97/97 different: autocrlf. (4) Python's `bash` resolved to WSL's bash, so a first bounded job ran nothing (exit 1). (5) One `python -` was typed by mistake (stopped at once with TaskStop, it ran nothing; no process survived) and one `git commit -F -` with an empty heredoc was typed (aborted on the empty message, no commit). (6) The first v2 harness pass imported pre-review v2 files and was abandoned midway (moved to `C:/t/layx/pwi/v2pre_59_partial`, not used). (7) The Docker stack `t2pw-pwi` was left running: phase 5 uses the same stack.
