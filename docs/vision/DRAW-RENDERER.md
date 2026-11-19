# DRAW-RENDERER: our diagram renderer, closer to a curated pathway map

Lane `draw/renderer` (worktree `C:/t/lane-draw`, from main `fd814b8e`). Renderer only:
`src/t2pw/vision/diagram/` (render.py, `__init__.py`, diagram.css) and the SVG metrics in
`src/t2pw/vision/checks/layout_metrics.py`. **No PWML code is touched**; content never changes (every IR element is
still drawn and the selfcheck passes on every candidate drawing).

Inputs: `C:/t/strategy/drawing/STRATEGY.md` (+ pilot), `C:/t/strategy/PO-NOTES-2026-11-19.md`.

## 1. Quality score gate (`svg_layout_metrics` + `layout_score`)

New keys on the core view of our SVG (old keys unchanged; svg == model metrics still holds on all 97 runs):

| key | meaning |
|---|---|
| `crossings` | (existing) orthogonal line crossings |
| `long_edges`, `edge_length` | lines longer than 400 px of polyline; total px of all lines |
| `non_straight` | lines with at least one bend |
| `node_edge_overlaps` | a line running through a box that is not one of its two ends |
| `edge_overlaps` | two lines with no end node in common lying on top of each other for >= 10 px |
| `label_collisions`, `edges_through_labels` | text boxes of different owners intersecting / on a foreign box; a line through a text |
| `bands`, `empty_bands`, `sparse_bands` | compartment bands or lanes drawn; holding no node; holding one node |
| `aspect_ratio` | (existing) width / height |
| `duplicate_names` | extra drawings of one non-currency compound (any compartment) |
| `repeated_captions` | compartment captions under nodes that repeat one already shown |
| `hops` | line bridges drawn |

`layout_score(m)` (lower is better, only for ranking drawings of the same pathway): 3 x crossings + 3 x edge
overlaps + 4 x node/edge overlaps + 2 x label collisions + edges through labels + 4 x long edges + length/500 +
0.5 x bent lines + 6 x empty bands + 1.5 x sparse bands + 2 x duplicate names + 0.25 x repeated captions + 10 x
node overlaps + shape (10 per unit above 2.5:1, +20 per unit above 3.5:1, a small penalty below 0.8:1). The weights
are judgement calls.

Baseline (committed under `docs/vision/evidence/DRAW-RENDERER/`): the 97 stored runs the vision suite measures
(V7b: `python -m t2pw.vision.checks.layout_metrics --runlist C:/t/ev-cm/runlist.json --idx
C:/t/vis-runs/scratch/generate/ok_idx.txt --scope default`) and the 8 showcase pathways (`C:/t/showcase2/<slug>/pwml_ir.json`
through `diagram.render_svg`). The baseline reproduces V7b exactly (crossings 439, bends 800, enzyme links > 200 px 6,
19 runs wider than 4:1).

## 2. Biology-aware layout (PO notes 2026-11-19)

| PO / strategy ask | what the renderer does now | where |
|---|---|---|
| untangle, fewer crossings | sources are placed just before their first user instead of column 0 (single-user chains hang off the step they feed); a copy of a molecule that nothing produces in its own compartment is tied to the column of a produced copy, so a downstream step continues the flow instead of restarting at the left edge | `assign_layers` |
| speed bumps | where a horizontal line still crosses a vertical one, the horizontal line bridges over it: an overlay group `g.hops` (gap in the background colour of the band underneath, plus an arc). The edge paths, and every metric read from them, are unchanged | `hops_svg`, `_bg_at` |
| straight, aligned arrows | more room (COLGAP 40 -> 56, ROWGAP 16 -> 22) | constants |
| start molecules as short stubs | a side substrate used only by this step sits in the step's column, on a side its enzymes leave free (up to 2 stubs with no enzyme, 1 with one enzyme) | `assign_layers`, `region_rows` |
| local copies of H+, water, ATP, CoA ... | a currency molecule used by two or more reactions of one compartment gets one drawing per reaction (dashed outline, legend entry). Not done when the molecule is named in the pathway name (CoA in a CoA pathway) or used once | `build`, `is_currency` |
| cross-compartment copies kept in line | the copy tie above | `assign_layers` |
| long chains wrapped to ~2.5:1 | `place_tiers` cuts the columns into 2-6 runs of equal width stacked as a snake (every second run reads right to left, as the PathBank menaquinol map does); a tier repeats only the bands that hold something there | `place_tiers`, `layout_bands`, `layout_spine` |
| transports cross the membrane | cargo, transport and cargo copy share one column, so a transport is a straight vertical line through the membrane band into the next compartment; two transports never share a column in one lane | `assign_layers` (bands and cell; spine keeps the old gap because it pins the chain to one row) |
| no repeated captions | the spine style names the main compartment once ("all in X unless labelled"); only molecules elsewhere carry a caption (566 -> 41 captions over the 97 runs) | `layout_spine`, `draw_view` |

Cell style: with one cell, the outside strip (saliva, extracellular) now shares the cell's column grid, so transports
run straight down instead of zig-zagging (sour taste: 34 -> 0 crossings).

## 3. Best of several

`diagram.render_svg` and `diagram_metrics` (so also `generate(..., "svg")`) render every style (bands, cell, spine),
for bands both band orders (flow, as before, and outside-to-inside), each unwrapped and, when wider than 2.5:1, wrapped
into the two tier counts nearest 2.5:1; the lowest `layout_score` wins, ties go to the old auto-pick. At most 10
renders per pathway (lipid A, the largest showcase: 1.9 s in all). An explicit `style=` fixes the style and only
chooses order and wrap. The SVG records the choice (`data-style`, `data-tiers`, `data-order`).

## 4. Display grouping (PO-approved)

In the bands style, compartment names with the same words in any order (`periplasmic leaflet of the inner membrane` =
`inner membrane periplasmic leaflet`) share a lane, and the gutter lists every spelling (`= ...`). The leaflets of one
membrane, and near-synonyms (cytoplasm / cytosol / intracellular; extracellular / extracellular space / medium), are
lanes of one band, separated by a dashed line. Every node keeps its own verbatim `data-comp`; nothing in the data is
merged. Bands exist only for compartments that hold something, and a wrapped tier repeats only the bands it uses.
Lipid A: 11 one-name bands -> 5 bands holding 7 lanes.

## 5. Membrane look

Band and organelle fill and line colour per type (ER, Golgi, mitochondrion, nucleus, plastid, vacuole, lysosome /
endosome, peroxisome, bacterial outer membrane, plasma membrane), light and dark theme. Nucleus, mitochondrion and
plastid get a double line at both edges of their band, and an inset second outline in the cell style; the thylakoid
stays single. The bacterial envelope is already drawn as outer membrane / periplasm / inner membrane bands, so no extra
double line was added there.

## 6. Metrics before and after

Before = main `fd814b8e` renderer (measured with the new metrics at `1004fe13`). After = this branch. Same IRs.

**97 stored runs** (the V7b set; official CLI, `svg_model_mismatch` empty and 0 crashes on both):

| metric | before | after | change |
|---|---:|---:|---:|
| crossings | 439 | 165 | -62% |
| long edges (> 400 px) | 146 | 33 | -77% |
| total edge length (px) | 278,079 | 191,660 | -31% |
| bent lines | 453 | 362 | -20% |
| runs wider than 4:1 | 19 | 0 | |
| aspect median / max | 2.88 / 7.17 | 1.94 / 3.39 | |
| node overlaps / line through a node | 1 / 3 | 0 / 0 | |
| label collisions / lines through labels | 16 / 44 | 2 / 31 | |
| repeated captions | 566 | 41 | |
| sparse (one-node) bands | 16 | 15 | |
| duplicate names (non-currency compounds) | 107 | 109 | content; see section 8 |
| enzyme links > 200 px, labels truncated | 6, 18 | 6, 18 | unchanged |
| line bridges drawn | 0 | 165 | one per remaining crossing |
| lines on top of each other (`edge_overlaps`, added last) | n/a | 1 | 4 -> 0 on the showcases + 12-run sample |

**8 showcase pathways** (`C:/t/showcase2/<slug>/pwml_ir.json`):

| metric | before | after |
|---|---:|---:|
| crossings | 220 | 64 (-71%) |
| long edges | 61 | 35 (-43%) |
| total edge length | 110,555 | 68,535 (-38%) |
| wider than 4:1 | 2 | 0 |
| aspect median / max | 3.10 / 4.65 | 1.84 / 2.55 |
| label collisions, node overlaps, lines through nodes | 7, 1, 3 | 0, 0, 0 |
| selfcheck failures | 0 | 0 |

| pathway | style before -> after | crossings | long edges | aspect |
|---|---|---|---|---|
| menaquinone | spine -> cell | 14 -> 1 | 3 -> 3 | 4.65 -> 2.52 |
| enterobactin | spine -> cell | 23 -> 7 | 6 -> 4 | 3.10 -> 2.49 |
| coenzyme A | spine -> spine, 2 tiers | 10 -> 19 | 6 -> 3 | 2.90 -> 1.80 |
| lipid A | bands -> bands (anatomical order, lanes) | 92 -> 17 | 21 -> 15 | 1.09 -> 1.58 |
| chlorophyll | bands -> bands, 2 tiers | 15 -> 6 | 4 -> 1 | 2.25 -> 1.84 |
| ceramide / sphingolipid | cell -> bands | 3 -> 7 | 5 -> 5 | 4.58 -> 2.55 |
| furochromone | spine -> bands, 2 tiers | 29 -> 7 | 6 -> 3 | 3.58 -> 1.35 |
| sour taste | cell -> bands | 34 -> 0 | 10 -> 1 | 0.34 -> 1.61 |

Targets: crossings -30% or better (met on both sets); runs wider than 4:1 -> 3 or fewer (0); long edges halved (met
on the 97 runs and on both sets together, 207 -> 68; **not** on the 8 showcases alone, -43%: most of the rest are
lipid A transports that run between the membrane lanes and a cycle back-edge in ceramide); 0 selfcheck failures.
Two showcases trade crossings for shape: coenzyme A and ceramide are wrapped from a too-wide strip into a near
2:1 drawing, and the wrap line adds crossings.

## 7. Gates

| gate | result |
|---|---|
| PWML byte-identical | **this branch vs the base tree (main fd814b8e + the metrics commit), all 97 runs: 97/97 identical sha256** (`proofcmp.py`). The prescribed command against the 2026-11-17 replay `replay_out/base_on` reports **88/97 on both trees**: idx 6, 7, 116, 120, 126, 127, 128, 129, 144 differ identically on the unmodified base, i.e. the replay predates main fd814b8e (PWML import fixes), not this lane. Evidence: `proof_summary.txt`, `proof_*_rows.jsonl` |
| suite2 (`C:/t/vis-runs/suite2.sh C:/t/lane-draw draw-g2`, HEAD 8867c7a5) | SMOKE 511 passed; golden batch 2, c011 8, stage8 4, quarantine 24; equivalence run 8, app 14, default 6; vision 365 passed 1 skipped; web 83 passed. 0 surviving processes (log `C:/t/vis-runs/logs/draw-gates2.log`). Same result on 183b6eda (`draw-gates1.log`) |
| selfcheck | 0 failures on every chosen drawing (97 + 8), and on every candidate (3 styles x 3 tier counts x 2 orders) of the 3 fixtures and 8 showcases (`tests/test_vision_diagram.py::test_every_style_order_and_wrap_draws_everything`, `devcheck.py`) |
| svg == model metrics | `svg_model_mismatch` empty on all 97 runs |

Existing test expectations that changed, one each:
- `tests/test_vision_diagram.py::test_reaction_transport_fixture_counts_and_pin`: the sha256 pin of the fixture SVG
  (Segoe UI only) is now `e6d9bc34...` (was `f4a4de0d...`). The drawing is new by design; the same test still asserts the
  junction counts, the drawn entity set and a clean selfcheck. No golden-seam test was edited.

## 8. Vision-LLM judge (optional step 6)

`docs/vision/evidence/DRAW-RENDERER/judge.py`: `google/gemini-3.8-flash` through the repo client
(`llm.client.chat_detailed`, key loaded by the client), rubric of 5 readability criteria, pairwise, each pair asked in
both orders and counted only when both orders agree. It ranks; it never moves anything and is not called by the
renderer. 20 pairs (8 showcases + 12 stored runs: the 6 hardest by baseline crossings + long edges, and 6 spread by
index).

Result: **built, not run; spend $0.** The first attempt (inside the gate hold) made no calls: run from the worktree,
the client loads the worktree's `.env`, which does not exist, so it fell back to the local LM Studio provider ("No
models loaded"). With the app checkout's client and `T2PW_LLM_PROVIDER=openrouter`, one diagnostic call
(`judge_diag.py`, 54 tokens) succeeded. The real run was queued, then stopped after about 30 minutes waiting for the
shared heavy lock (other lanes' live jobs); its waiting shell was killed by PID. To run it:
`T2PW_LLM_PROVIDER=openrouter python judge.py <app checkout> <out> pairs.json` (pairs = the 20 PNG pairs in
`C:/t/lane-draw-work/pairs.json`; budget guard $2.50).

Spot-check (mine, looking at the PNGs, before the judge ran): after is better for menaquinone, enterobactin, lipid A,
chlorophyll, sour taste (also run 10), coenzyme A (weakly: U-shape, but more crossings) and run 125 (local SAM copies,
stubs); run 137 is a tie (compact but dense around R3/R4, where the old strip had long lines but a clear left-to-right
order).

## 9. Open questions

1. **Long edges on the showcases alone are -43%, not -50%.** The rest are mostly lipid A transports between the inner
   membrane lanes and the periplasm (structural) and the ceramide salvage cycle. Options: accept (97-run set and the
   combined set are well past -50%); or try transport placement next to the target lane (1-2 days, risk of new
   crossings).
2. **Duplicate names 107 -> 109** are content: the same compound in two compartments must be drawn twice (selfcheck
   needs a node per compartment). Two extra come from long currency names the metric reads wrapped (for example
   S-adenosyl-L-methionine). Not a regression in the drawing.
3. **Snake wrapping reads every second tier right to left** (PathBank style). If the PO prefers left-to-right rows,
   `place_tiers` drops the mirror (one line); expect a few more long wrap lines.
4. **Display synonyms** are a fixed table (cytoplasm / cytosol / intracellular / cytoplasmic; extracellular /
   extracellular space / region / medium) plus the word-set rule for spellings and the leaflet rule for membranes. The
   PO approved grouping in principle; the exact table may need review.
5. **Score weights** are judgement calls; the judge (section 8) is the independent check. PathBank-mined parameters
   (strategy item 6) were not attempted.

## 10. Dead ends

- Transports in one column with their cargo in the spine style put cargo, transport and cargo copy on the pinned main
  row in one cell (overlap, zero-length edges, selfcheck failures). Spine keeps the old one-column gap.
- A stronger router penalty for reusing a line (3 -> 10) gave more crossings (108 -> 117 on showcases + sample) and no
  fewer overlaps; reverted (`exp/` copy only).
- Cloning every multi-user source (not only currency) would shorten the remaining long lines but adds non-currency
  duplicates; not done.

## 11. Files

Code: `src/t2pw/vision/diagram/render.py`, `src/t2pw/vision/diagram/__init__.py`, `src/t2pw/vision/diagram/diagram.css`,
`src/t2pw/vision/checks/layout_metrics.py`, `tests/test_vision_diagram.py`, `tests/test_vision_layout.py`.
Evidence: `docs/vision/evidence/DRAW-RENDERER/` (baseline and final metrics, scripts, judge output).
Outside the repo: before/after PNGs and the side-by-side page `C:/t/lane-draw-work/compare/index.html`; logs
`C:/t/vis-runs/logs/draw-*.log`; dumped IRs `C:/t/lane-draw-work/ir97/`.

## 12. Review round 1 (APPROVE WITH NOTES; F1-F6 fixed, F7 waits for the PO)

Code commits: `9bea6d7c` (F1 + render_svg docstring), `4c06454e` (F5), `eb4a820e` (F2, F3, F4 + tests). Gates on
`eb4a820e`, a clean committed tree (suite2 log opens with `HEAD eb4a820e...` and an empty status block).

| finding | fix | check |
|---|---|---|
| F1 cost: up to 10 candidates, O(texts^2) scoring | above `BIG = 150` molecule drawings only the auto-picked style and its wraps are rendered (one band order) | timings below |
| F2 "nucle" matched nucleoid and nucleolus | double membrane only for nucleus / nuclear / nucleoplasm | `test_double_membrane_only_for_real_envelopes` (nucleus yes; nucleoid, nucleolus no) |
| F3 local copies cut a transport's link | a currency molecule that is transport cargo in a compartment keeps one drawing there | `test_transported_currency_keeps_its_link_to_the_reactions_that_use_it` (pumped H+: 1 drawing in the matrix; not pumped: 2 local copies) |
| F4 spreading edges could close a cycle | when the later transport already leads to the one seen first, the edge is laid in flow order instead (this both avoids the cycle and still keeps the two transports in separate columns; a plain skip would leave them stacked) | `test_chained_transports_in_one_membrane_keep_flow_order` (tr_2 -> tr_10 through one membrane) |
| F5 two currency lists; stale docstring | one list, `src/t2pw/vision/currency.py` (stdlib only, the union of both), imported by render.py and layout_metrics.py; render_svg docstring describes the best-of-several default | 97-run metrics re-run |
| F6 gate evidence | `C:/t/vis-runs/suite2.sh` prints `git rev-parse HEAD` and `git status --porcelain` first | `draw-gates3.log` |

All three new tests fail on the previous HEAD (`703e079d`) and pass on `eb4a820e` (dev check
`failfirst.py`, outside the gate).

**F1 timings** (`render_svg` wall time, this machine, `timing.py`):

| IR | molecule drawings | before F1 | after F1 |
|---|---:|---:|---:|
| largest stored IR (run 7, 122 nodes) | 102 | 1.61 s | 1.81 s (below the cap: same work; run-to-run noise) |
| chlorophyll showcase | under 150 | 1.59 s | 1.85 s (same) |
| lipid A showcase | 83 | 2.23 s | 2.34 s (same) |
| synthetic: run 7 + lipid A + chlorophyll side by side (`make_big.py`) | 287 | 7.11 s | 1.68 s |

No stored IR reaches the cap (largest 102 molecule drawings), so the cap changes no stored drawing.

**Metric deltas** (97 runs, round 1 vs the previous final `8867c7a5`): identical on every key (crossings 165, long
edges 33, edge length 191,660, bent 362, wider than 4:1 0, label collisions 2, edge overlaps 1, duplicate names 109,
repeated captions 41); PWML-side reading order unchanged (17 of 264 pairs); `svg_model_mismatch` empty, 0 crashes.
Showcases identical too (crossings 64, long edges 35). No stored pathway has a nucleoid/nucleolus, pumped currency used
twice, or a cycling transport chain, and the currency union adds only spellings none of the runs use.

**Gates on `eb4a820e`**: SMOKE 511 passed; golden batch 2, c011 8, stage8 4, quarantine 24; equivalence run 8, app 14,
default 6; vision 370 passed 1 skipped (365 + the 5 new); web 83 passed; 0 surviving processes, BW_EXIT=0
(`C:/t/vis-runs/logs/draw-gates3.log`). PWML: branch vs base tree `C:/t/lane-draw-base` 97/97 sha-identical (both 88/97
against the stale replay, as before). Selfcheck: 0 failures on the 97 chosen drawings and the 8 showcases.

F7 (synonym band grouping) unchanged; it waits for the PO ruling (open question 4).
