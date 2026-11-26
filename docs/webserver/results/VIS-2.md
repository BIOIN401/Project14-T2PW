# VIS-2: PathBank cell and organelle art in PWML layout v2

PO request (2026-11-25, verbatim): "we need the arrows to look better, have cleaner flow as well as using the organelle
images that are in pathwhiz"; tonight: "iterate until our renderings look like how they should, make sure to check the
corresponding pwml files and making the changes to our pwml making code as well". Plan: `results/VIS-REF.md` section 2.
Branch `vis/organelle-art`, cut from `vis/pathbank-edges` `49ab4da8` (VIS-1, not yet reviewed or merged; a review change
there means a rebase here). File boundary: `src/t2pw/pwml/layout.py`, new `src/t2pw/pwml/cell_art.py`, `tests/test_pwml_*`.
Nothing else in `src/` was touched. Evidence: `docs/webserver/evidence/VIS-2/`. Working files (replays, renders, PathWhiz
runs; not committed): `C:/t/vis-pw2-work/`. Not merged; LEDGER row left to the orchestrator.

## 1. The convention, mined from the 16 curated PathBank PWMLs

`evidence/VIS-2/scripts/mine_art.py` over the reference set (`C:/t/vis-ref/.../pwml_markups/`, lxml `recover=True`);
output `evidence/VIS-2/mined_art.txt`. Image names: the PathBank `drawable_images` rows in the PathWhiz harness seed
(`seed_inserts.sql`), checked against the images themselves (`C:/t/merge-vis3/src/t2pw/render/pathbank_art/`).

**Records.** `drawable-element-location`: `id`, `visualization-template-id` (nil in 205 of 220), `x`, `y` (top-left; PathWhiz
draws `translate(x,y) rotate(degree) scale(sx,sy)`, `default.jst.eco`), `scale-x`, `scale-y`, `degree` (0 in 194 of 220),
`zindex` (2 in 148 of 220), `drawable-image-id`, and `width` / `height` = the image's native size (PathWhiz's own export
writes them; the importer drops them). `zoom-visualization`: `p1x/p1y` (top-left of the 25 x 20 small box PathWhiz draws
on the image), `p2x/p2y` and `p3x/p3y` (the two corners of the large box), `visualization-template-id` (4 "Top Oriented",
39 "Left Oriented", 40 right, 41 bottom; 36 zooms: 39 x 18, 4 x 13, 40 x 4, 41 x 1), `color` (the organelle's background:
mitochondria `#FFEBEB`, ER `#FFEEDE`, nucleus `#FFF9C9`, chloroplast `#EBFFEB`, ...; empty for a cell), `zindex` (4 in 24
of 36), `label` (nil). The wedge is the quad from the small box to the near side of the large box
(`left_zoom` / `top_zoom.jst.eco`).

| cell type | references | what PathBank draws |
|---|---|---|
| Gram-negative bacterium | PW000779 (TCA), PW000921, PW000923 | bacterium image **146** (512 x 512) at scale **1.0**, x 55-60 (left margin), centre on the cell's centre line (within 23 units in 921 / 923; 134 in 779); **left zoom 39**, no colour, zindex 4, small box at 26-30 % / 51-53 % of the image, large box **27 units outside** the outer membrane line (27 / 25 / 27); bacterial chromosome **92** (479 x 481) at scale **1.5-1.6** in a free part of the cytoplasm; envelope: outer membrane (template 1) at the box, peptidoglycan (**template 291**) 120-200 in, inner membrane (template 1) 250-500 in (779: 121 / 250) |
| Gram-positive | none in the 16 | - |
| eukaryote (animal) | PW000001, PW000002, PW000003, PW000017, PW127049 | organelle images in the cell (nucleus 14 at 1.9-3.9, rough ER 3 often rotated, mitochondrion 1 at 0.6-0.9) or the organelle cluster **90** (484 x 497; nucleus, rough ER, mitochondrion) at 1.8-2.5; a zoom from the organelle image to the organelle box holding the reactions, coloured as that organelle (PW000001: mitochondrion image -> mitochondria box, top zoom, `#FFEBEB`) |
| plant | PW012911 (chlorophyll a) | plant organelle cluster **91** (543 x 497; chloroplast, nucleus, rough ER, mitochondrion) at 2.4 in the cell's top right; **top zoom 4** `#EBFFEB` from the chloroplast in the image to the chloroplast box (large box 27 outside its line); cell wall 304 + plasma membrane 1 |
| fungus | none in the 16 | - |

## 2. What changed

`src/t2pw/pwml/cell_art.py` (new) and `src/t2pw/pwml/layout.py` (layout v2 only; `T2PW_PWML_LAYOUT=v1` untouched).
Visualization records only: `drawable-element-locations`, `zoom-visualizations` and one peptidoglycan
`membrane-visualization`. No state, location, compound, protein, reaction, transport, interaction or id changes.

1. **Cell type** (`cell_kind`), from what the PWML already knows: the species of every drawn compartment's state (a state
   without a species counts the pathway's species), classified by the existing `vocabulary_map.organism_kind` (species
   classification + its exact genus tables). One kind for every state: `bacterium` or `eukaryote`; anything else
   (archaeon, mixed, undetermined) gets **no art** (fail closed). Plant = a eukaryote with a plastid / thylakoid
   compartment. **Gram-negative** = a bacterium whose PWML names an outer membrane or periplasm location (no genus table
   was added; `final_payload_locations.GRAM_POSITIVE` exists, but Gram-positive and unknown draw the same here).
2. **Bacterium**: image 146 left of the cell box, centred on it (scale 0.2 x cell height / 512, between 0.5 and 1.0);
   left zoom 39 from it to the cell's outer line + 27; chromosome 92 at the largest scale from 1.0 to 0.4 that fits in a
   free spot of the cytoplasm (clear of nodes, labels, edges and child boxes, nearest the centre). Layout v2 keeps a
   320 x 320 empty group in the cytoplasm so a spot always exists. The whole picture moves right to make room.
3. **Gram-negative envelope**, as membrane visualizations only: when no box already draws an outer membrane or
   periplasm, the cell box becomes PathBank's envelope (outer and inner membrane lines 250 apart, peptidoglycan 291 at
   120); when the boxes already nest (outer membrane > periplasm > one inner membrane box) a peptidoglycan line is
   drawn between them. Transporters on that membrane sit mid-envelope through the existing double-membrane rule.
   No periplasm state or location is added (none needed).
4. **Eukaryote**: the cluster (90; 91 for a plant) and a zoom, coloured as the organelle, from that organelle in the
   cluster to the organelle box with the most nodes among those the cluster shows (mitochondrion, ER, nucleus,
   chloroplast). The cluster goes in a free spot above the box (inside the cell, as PathBank draws it) at scale up to
   1.4; else in a strip above or left of the picture, whichever wedge covers fewer nodes. No organelle box: no art.
5. Canvas grows to hold the image and the wedge. Ids: every new record takes its id after all existing ones (the
   peptidoglycan line after the labels), so no other record's id changes.

Tests: new `tests/test_pwml_vis2_cell_art.py` (6 tests): bacterium (icon, left zoom geometry, chromosome inside the inner
membrane clear of nodes, Gram-negative envelope lines and peptidoglycan); Gram-positive / unknown Gram (icon and
chromosome, one membrane); eukaryote (cluster 90, zoom colour and geometry into the mitochondrion); plant (91,
`#EBFFEB`); unknown cell types (undetermined genus, archaeon, eukaryote without an organelle box: nothing);
determinism and everything inside the canvas after the move. `tests/test_pwml_layout.py`: `_content` (the v1-vs-v2
content signature) also leaves out the two art sections, as it does membranes and labels. No golden-seam test edited.

## 3. Gates (all on `3bd8335c`, the code commit; tree clean apart from this untracked record)

All through `C:/t/vis-runs/queue.sh`, holder `vis2`; every wrapper report: BW_EXIT 0, FINAL SURVIVING COUNT 0.

| gate | result | log |
|---|---|---|
| focused: 16 `test_pwml_*.py` files (incl. the new one), `test_final_payload_writer_fixes.py`, `test_orch768_replay_corpus.py` | **311 passed** | `C:/t/vis-runs/logs/vis2-final.log` (copy: `evidence/VIS-2/final.log`) |
| `suite2.sh` | SMOKE **511**; golden seam **2 / 8 / 4 / 24** (unedited); equivalence run_pathway **8**, app_vs_service **14**, default_path **6**; vision **370 + 1 skipped**; web **83**; all passed | same log |
| content check, 97-run replay | **0** differences (`metrics/cc97_final.json`) | same log |
| content check, 8 showcases | **0** differences (`metrics/cc8_final.json`) | same log |
| content check, server job item-6 (enterobactin) | **0** differences (`metrics/cc_item6_final.json`) | same log |
| PathWhiz import A/B, 17 files | 17/17; content counts identical; messages 89 -> 89 | `C:/t/vis-runs/logs/vis2-pwab.log` (copy: `evidence/VIS-2/pwab.log`) |

The first focused run (`vis2-d1`, before the commit) had 3 failures, fixed before the commit: an interaction's arrow path
recomputed after the move from float coordinates (`L 123.0 456.0` instead of `L 123 456`;
`test_pwml_interaction_export.py`), and the v1-vs-v2 content signature (`_content`) counting the new art sections
(`test_pwml_layout.py`, `test_pwml_l3_chain_layout.py`).

**Base.** VIS-1's own final replay (`C:/t/vis-pw1-work/{replay,cc_show,cc_item6}/final2`, `49ab4da8`, this branch's base).
After = the same replay scripts on this worktree (stored payloads, final-payload flag on, no network; `network_attempts 0`).

**Content checker.** `evidence/VIS-2/scripts/content_check.py`, VIS-1's copy with one change, named in its docstring:
the `drawable-element-locations` and `zoom-visualizations` sections are masked like the membrane and label sections.
The canvas size was already masked. Everything else (every element, id, name, reference, state, location, reaction,
transport, interaction, hidden flag, template, zindex, arrow flags, member sides) is compared as canonical XML.

**Art over the 106 PWMLs** (`art_summary.py`, `metrics/art_final.json`): art in 76; bacterium 146 and chromosome 92 in 47
each (every bacterial PWML); animal cluster 90 in 27, plant cluster 91 in 2; 76 zooms; peptidoglycan in 14 (Gram-negative);
images over a drawn node **0**; nodes under a zoom wedge outside its large box **2** (replay 119, one file; 10 in the first
dev run before the placement rule in section 2 item 4). The 30 without art: no organelle box (eukaryotes), archaea,
or an undetermined organism.

**PathWhiz import A/B** (`evidence/VIS-2/metrics/pathwhiz_ab_final.json`). Same 17 files as VIS-1 (8 showcases + V7b
idx 7, 10, 14, 116, 126, 128, 11, 134, 114), PathWhiz `t2pw/latest` `63959982` clone at `C:/t/pwpr/pwdraw-run`, project
`t2pw-pwdraw` only, clean database per file, stack down afterwards (volumes kept). Base arm = VIS-1's after arm
(`C:/t/vis-pw1-work/pw/after_final2`, the `49ab4da8` files); after arm imported fresh.

| 17 files | base | after |
|---|---|---|
| imports successful | 17/17 | 17/17 |
| compounds / proteins / complexes / reactions | 210 / 159 / 168 / 167 | same |
| transports / transport transporters / interactions | 32 / 18 / 39 | same |
| compound locations / nucleic acids / states / labels | 322 / 20 / 67 / 70 | same |
| drawable element locations / zoom visualizations | 0 / 0 | **20 / 13** |
| membrane visualizations | 61 | 63 (enterobactin: the envelope's inner line and the peptidoglycan line) |
| importer messages | 89 | **89** (no file changed) |

PathWhiz's importer (`lib/pwml_parser.rb`, read only) builds both records with `build_nested_visualization`, keeping
the columns of `drawable_element_locations` (template, x, y, scale_x, scale_y, degree, zindex, drawable_image_id;
validated: image id, x, y, scales, degree, zindex present) and `zoom_visualizations` (p1x .. p3y, template, color,
zindex, label; validated: template, zindex). The `width` / `height` elements are not columns and are dropped, as for
PathBank's own files. Coordinates are only offset when negative; ours are not.

**PathWhiz drawings** (screenshots `evidence/VIS-2/pathwhiz/enterobactin_pathwhiz.png` (bacterium) and
`ceramide_pathwhiz.png` (eukaryote); also chlorophyll and lipid A under `C:/t/vis-pw2-work/pw/after_final/screenshots/`):
the viewer draws every record. Enterobactin: the left zoom wedge from the bacterium into the cell, the outer and inner
membrane lines with the lilac peptidoglycan line between them, TolC and TonB on the envelope, the chromosome in the
cytoplasm. Ceramide: the top zoom from the cluster into the ER box, filled with the ER colour. 0 console errors. **The
images themselves show as broken-image placeholders** at the right position and size: the viewer loads them from
PathWhiz's image store, `https://s3-us-west-2.amazonaws.com/smpdb/drawable_images/images/146.png` (and 92.svg, ...), and
that bucket answers **403** to an anonymous request from this machine (checked with curl). The ids are PathBank's own
(the harness seed is a copy of PathBank's `drawable_images` table), so on PathBank's servers they resolve to the same
art as in the reference files. The PathBank-style render (VIS-3 renderer, section 4) shows the images.

## 4. Visual check

Rendered with the VIS-3 PathBank-style renderer (the one the result page uses; `C:/t/merge-vis3`, through the queue)
and compared with the PathBank references (`C:/t/vis-pw2-work/refshots/`: PW000779, PW000001, PW012911, PW000017,
headless Edge from the reference SVGs). Side-by-side page: `C:/t/vis-pw2-work/compare_final.html`
(before = VIS-1 `49ab4da8`, after = `3bd8335c`; PNGs under `C:/t/vis-pw2-work/renders/{base,final}/`).

- **item-6 enterobactin (E. coli)**: reads like PW000779: bacterium at the left with the left zoom over the whole
  cell; outer membrane, peptidoglycan, inner membrane; TolC / TonB mid-envelope; chromosome in the free lower right of
  the cytoplasm; extracellular cargo below. Same for the enterobactin showcase.
- **lipid A (E. coli)**: bacterium, zoom into the cytoplasm cell (the largest of its separate cell boxes), chromosome.
  The outer-membrane box holds two inner boxes, so no peptidoglycan line (fail closed); lipid A's box structure is the
  open question of DRAW-PATHWHIZ.
- **menaquinone (Lactococcus, Gram-positive), coenzyme A (B. subtilis), replay 114 (cell wall)**: bacterium, zoom,
  chromosome; one membrane (and the cell wall band in 114).
- **chlorophyll (Arabidopsis), replay 7**: like PW012911: plant cluster over the plastid box, top zoom from its
  chloroplast, the box filled chloroplast green.
- **ceramide (human)**: animal cluster over the ER box, top zoom from its ER, ER colour.
- **replay 134 (carnitine)**: cluster over the mitochondrion (two red lines), mitochondria pink.
- **replay 79 (cholesterol)**: the ER box sits inside the cell under a cytosol row; the first version zoomed from the top
  across that row; now the cluster sits left of the picture with a left zoom straight into the ER box.
- **replay 132 (sialic acid)**: same rule; the nucleus box at the cell's left is zoomed from a cluster to its left.

## 4b. Dead ends and things that went wrong

- First placement for the eukaryote cluster: a free spot in the picture nearest the box, else a strip above. Replay 79
  and 132 then zoomed across a row of nodes (10 wedge crossings in 106 files). Now the in-picture spot is kept only when
  its wedge covers no more nodes than the best strip, and the strip is above or left, whichever covers fewer (2 left).
- The chromosome was first placed only in free space; a small cytoplasm has none, so layout v2 now keeps a 320 x 320
  empty group in the bacterial cytoplasm (its shelf packing places it after the cytoplasm's own groups).
- Moving the picture after the edges were drawn recomputed arrow paths from float coordinates (the focused gate caught
  `test_pwml_interaction_export.py`); integers are now kept as integers.
- Queue waits: about 1.5 h behind VIS-1's PathWhiz A/B and the RAM floor (3 GB free of 31 GB, 47 GB committed, held by
  desktop applications; nothing of this lane's was running). Nothing was killed.
- The PathBank reference raster images in the zip are 200 px thumbnails except PW000001; the full-size references were
  screenshotted from the SVGs with headless Edge (`scripts/svgshot.sh`).

## 5. Open questions for the PO

1. **Gram-negative without an envelope location.** A Gram-negative bacterium whose PWML names only the cytoplasm
   (common for E. coli runs) gets the bacterium, zoom and chromosome but one membrane line. Options: (a) accept;
   (b) a reviewed Gram-negative genus table next to `organism_kind` (Escherichia, Salmonella, Pseudomonas, ...), then
   draw the envelope for those too (cost: a curated table to maintain, a wrong genus draws a wrong envelope);
   (c) ask the IR for the Gram stain.
2. **Envelope labels.** PathBank labels the layers (Outer Membrane, Peptidoglycan, Periplasm, Inner Membrane). Not
   added: they would be new `label-locations` (the PathWhiz A/B rule allows only new drawable / zoom records) and
   "Periplasm" would name a compartment the PWML does not have. Options: (a) leave unlabelled; (b) label the
   peptidoglycan only; (c) all four.
3. **Fungi and other eukaryotes** get the animal cluster (90: nucleus, rough ER, mitochondrion, organelles every
   eukaryote has); no reference shows a fungus. A plant without a plastid compartment (furochromone,
   *Saposhnikovia*) is not recognised as a plant (no kingdom data in the PWML) and, having no organelle box, gets no art.
4. **Eukaryotes without an organelle box** (cytosol only, e.g. sour taste) get no art; PathBank sometimes puts the
   cluster in such a cell as decoration (PW000017). Options: (a) accept (fail closed); (b) cluster with no zoom.
5. **Peroxisome, lysosome, Golgi** boxes get no zoom (not in the cluster images; PathBank has separate peroxisome 4 /
   lysosome 5 images). One zoom per pathway, to the largest cluster organelle.
6. **Images in the PathWhiz harness.** The harness viewer cannot load the art (PathWhiz's S3 image store answers 403),
   so the PathWhiz screenshots show placeholders. Options: (a) accept (production PathBank serves the images);
   (b) seed the harness with local copies of the images (the VIS-3 `pathbank_art/` PNGs) to screenshot real art (an hour
   of harness work, smpdb tree untouched); (c) check one import on a PathBank server with image access.
7. **Envelope membrane count.** The Gram-negative envelope adds membrane-visualization records (enterobactin: 1 -> 3),
   besides the drawable and zoom records the A/B rule names. They are visualization records only, as item 3 of the
   brief asked; confirm that this is acceptable.

## 6. Review verdict and merge (orchestrator, 2026-11-26)

- Independent ws-reviewer on `d4814ab1`: **APPROVE WITH NOTES**. It ran its own edge-case probe: the bacterial
  envelope cannot be triggered by a mitochondrial or chloroplast outer membrane, art always sits below pathway content
  (zindex), and every emitted field is one PathWhiz accepts. Notes carried to VIS-5: F1 TolC and TonB both sit
  mid-envelope; F4 a Mycobacterium outer membrane would trigger the Gram-negative envelope; F5 `_shift_path` ignores
  H/V/A path commands.
- Merged as `aa34af70` = webserver/refactor `7084cb95` (VIS-4 merged) + `vis/organelle-art` `d4814ab1`, no conflicts,
  in the detached worktree `C:/t/merge-vis2`. Lane flags unset.
- Gates on `aa34af70` (`evidence/VIS-2/logs/merge-vis2-gates.log`, script `merge_vis2_inner.sh`, tree clean):
  SMOKE **511 passed**; golden seams **2 / 8 / 4 / 24 passed**, unedited; equivalence **8 / 14 / 6 passed**;
  vision **370 passed, 1 skipped**; web **83 passed**; `test_pwml_*` + writer fixes + replay corpus **311 passed**;
  `test_render_*` **49 passed**; service + web **346 passed**. `BW_EXIT=0`, **FINAL SURVIVING COUNT 0**.
- Pushed `aa34af70` to `webserver/refactor`. The open questions in section 5 stay open.
