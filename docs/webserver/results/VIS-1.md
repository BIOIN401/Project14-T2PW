# VIS-1: PathBank edges and flow in PWML layout v2

PO request (2026-11-25, verbatim): "we need the arrows to look better, have cleaner flow". Plan and reference analysis:
`results/VIS-REF.md`. Branch `vis/pathbank-edges`, cut from `webserver/refactor` `7bf8588e`. File boundary
`src/t2pw/pwml/layout.py` + `tests/test_pwml_*`; nothing else in `src/` was touched.
Code commits `e43ceb19` (the change) and `49ab4da8` (an interaction's hidden edge, found by the focused gate). This
record is a separate docs-only commit. Evidence: `docs/webserver/evidence/VIS-1/`. Working files (replays, renders,
PathWhiz runs; not committed): `C:/t/vis-pw1-work/`. Not merged; LEDGER row left to the orchestrator.

## 1. Conventions mined from the 16 curated PathBank PWMLs

`evidence/VIS-1/scripts/mine_edges.py` over the 785 member edges (679 reaction, 106 transport) of the 16 reference
PWMLs; `mined_summary.txt` (per group: reaction / transport x Left / Right x currency / main) and `mined_more.txt`.

| convention | PathBank (16 files) | VIS-1 |
|---|---|---|
| shape | every member edge is `M p0 C c1 c2 p3`; 96 % a single cubic, the rest right-angle connectors | one cubic; the old right-angle grid route only when no candidate curve is clear of node boxes |
| start | at the member node in 96 % (a product's arrow end is the path start) | unchanged (the writer's direction) |
| member end | middle of a box side (564/679 within 5 % of the middle), at right angles (605/679: L/R side leaves horizontally, T/B vertically) | middle of the side facing the other end, at right angles; the next facing side(s) when the curve would cross a node or a drawn edge; never a side facing away (that makes loops) |
| reaction end | middle of the enzyme's side (574/679), along the shaft (579/679) | substrates: middle of the enzyme's left side, products: right side; a node behind the enzyme: top or bottom |
| one point per side | substrates share one point in 91/114 reactions, products in 112/131 | always |
| main substrate / product | on opposite enzyme sides in 101/131 | always (left / right) |
| control arms | median 30 at both ends of main edges (chord median 140-149); side compounds 40 at the node, 30 at the reaction | node end: 0.25 of the chord (0.4 for side compounds), at least 30 (a third of a short chord), at most 200; reaction end: 30 |
| side compounds | currency pills (50x30 / 78x78) 75 along the shaft from the reaction point (quartiles 60/75/100) and 73 off it to the pill's near side (57/73/92); chord 106-113 | 75 along, 73 off; the first above, the second below, more stacked outward 20 apart |
| straight main chain | 109 of 420 main edges exactly straight, 231 within 5 units | a chain in one layered row: aligned centres, horizontal normals, straight |
| arrow flags | product edges `start_arrow` true + `start_arrow_path` (198/205); substrate edges: no options or flags false | flags unchanged (content); every `*_path` recomputed from the new start and first control point |
| edge templates (`edge_templates.txt`) | reaction member edges 5 (690/691); transport member edges **83 (151/152)** | transport member edges 83 (the writer wrote 5 on all 142 + 44) |
| compound box templates (`compound_templates.txt`) | 81 on 256 boxes, 144 of them for compounds with a DrugBank id (L-alanine, pyruvic acid, ...); **157 only on 15 boxes**, each the drug its own drug pathway is about (fluconazole, caffeine, zidovudine, buprenorphine, benzethonium) | 157 drawn as 81 (same 200 x 190). The writer's `select_compound_template_id` gives 157 to every compound with a DrugBank id (207 of 748 large boxes in the 97 replays) |

### Arrow options: `<options>` JSON is PathWhiz's import form; `option:*` tags are its export form

VIS-REF read "ours: no arrow options" from the absence of `option:start_arrow` elements. Our PWML carries them as the
`<options>` JSON that PathWhiz stores (`edges.options` column). From PathWhiz's source (`t2pw/latest` `63959982`,
read only):
- the importer (`lib/pwml_parser.rb`: `store_nested_visualization` -> `to_hash_shallow` -> `slice(*attribute_list)`)
  keeps only child elements named like an `Edge` column, so `<options>` becomes `edges.options` and an
  `option:start_arrow` element is dropped;
- the `option:*` elements come from the exporter (`app/models/visualization/edge.rb` `to_pwml`:
  `JSON.parse(self.options).each { |k, v| xml.option(k, v) }`), the same JSON written back out;
- the viewer redraws every head itself (`edge.js.coffee` `for_template` -> `addStartArrow` / `drawArrow`: 30 units,
  +-30 degrees, from the path's start toward its first control point); the stored `*_path` strings are not drawn.
So `option:*` tags were not emitted (they would change nothing in PathWhiz and put an undeclared prefix in our XML).
What did break heads: zero-length paths. 480 of 1650 edges in the 97 replays had every point equal, and one arrow
edge had its start on its control point, where `drawArrow` divides by zero. VIS-1 leaves 0 of either. The PathWhiz
screenshots (section 3) show the heads.

## 2. What changed (`src/t2pw/pwml/layout.py`, layout v2 only; `T2PW_PWML_LAYOUT=v1` untouched)

1. **PathBank cubics** (`_curve`, `_side_point`, `_facing_sides`, `_bezier_points`, `_clear`, `_crossings`). Per edge,
   the candidates (node sides facing the other end x the fixed reaction end) are scored: clear of node boxes first,
   then fewest crossings with the edges already drawn, then preference order. Only when no candidate is clear does the
   old right-angle grid route run. The reaction end keeps PathBank's 30-unit arm, so edges meeting at one point turn
   in late instead of running together (with 0.25 of the chord there, crossings were 96 instead of 74).
2. **Reaction forks.** All substrates meet at the middle of the enzyme's left side, all products leave from the
   middle of its right side; a node behind the enzyme meets its top or bottom. A reaction point without an enzyme is
   met from the side facing the node.
3. **Side compounds.** A compound becomes a side compound of a reaction when it is currency (one of PathBank's
   small-molecule templates, `COMPOUND_ID_TO_TEMPLATE.values()`, or the writer's own `_is_chain_hub_compound_record`),
   at most 100 x 100, used by exactly one process, and exactly one main member meets that side of the enzyme (over
   every reaction the enzyme catalyses) in the same compartment. With two or more main members their edges fan in
   through the pills' places (replay 125), so that side keeps its compounds in the layered layout. Side compounds
   leave the layering; the enzyme's block reserves room above and below (evenly, so the shaft stays on the centre
   line); each is placed 75 along and 73 off the shaft. Taking them out of the layering is also what straightens the
   main chain: as layered nodes they pulled it into a zigzag (VIS-REF section 3, item-6).
4. **No zero-length paths.** An IR connector from an enzyme to its own reaction (no member refers to it; PathWhiz's
   viewer drops it as an orphan) is drawn as the shaft through the enzyme's first location; the hidden substrate
   edge of a reaction without enzyme is a real (hidden) curve; a zero-length edge tied to nothing is a 30-unit stub in
   the bottom margin instead of a point.
5. **Transports.** A transporter moved onto its membrane takes its inner cargo on the side facing in and its outer
   cargo on the side facing out (vertical tangents through the membrane). Its spot along the membrane now prefers,
   within 300 units of the inner cargo, one the cargo reaches on a clear curve (item-6: the two inner cargos were
   stacked, so the line from the upper one ran through the lower one and was routed in right angles).
6. **Stray duplicate transporter boxes.** Cause: the IR gives a transporter a location in its own membrane state
   (item-6: TolC location 95 in the outer-membrane state, TonB 96 in the inner-membrane state) besides the cytosolic
   locations 83 / 84 that the transports' complex visualizations use. No process refers to 95 / 96, PathWhiz's viewer
   drops them (`isOrphan`), and v2 packed them in the loose strip under the cell, where `render.py` drew them. Now a
   location no process refers to, whose entity (protein, nucleic acid, element collection, compound) has a drawn
   location, is drawn on top of that location (inside a complex: on the matching member). The locations stay
   (content). This is the whole of the new `node_overlaps` in section 3; overlaps between attached nodes stay 0.
7. **Templates** (orchestrator items): transport member edges get 83; compound box 157 is drawn as 81.
8. **Interaction edge pairs** (`49ab4da8`): a curve with no reaction end is remembered per (start, end) node pair and
   the edge drawn the other way reuses it reversed, so an interaction's hidden Left edge stays the exact reverse of its
   Right edge (`test_pwml_interaction_export.py` caught the difference).

Tests. New `tests/test_pwml_vis1_edges.py` (5): the fork (one shaft, main substrate and product straight on it, side
compounds above / below leaving vertically and meeting the shaft horizontally, arrow tip at the path start, the shaft
connector, the stray twin on its drawn location, determinism); side compounds need one main partner; 157 -> 81;
transport edges 83; no zero-length path or degenerate arrow tangent. `tests/test_pwml_layout.py`: the zero-length
orphan is now the margin stub; `_content` (the v1-vs-v2 content signature, also used by `test_pwml_l3_chain_layout.py`)
reads compound template 157 as 81 and edge template 83 as 5. No golden-seam test edited.

## 3. Gates (all on `49ab4da8`, clean tree)

All through `C:/t/vis-runs/queue.sh`, holder `vis1`; every wrapper report: BW_EXIT 0, FINAL SURVIVING COUNT 0.

| gate | result | log |
|---|---|---|
| focused: 15 `test_pwml_*.py` files (incl. the new one), `test_final_payload_writer_fixes.py`, `test_orch768_replay_corpus.py` | **305 passed** | `C:/t/vis-runs/logs/vis1-final2.log` |
| `suite2.sh` (HEAD and clean status recorded in the log) | SMOKE **511**; golden seam **2 / 8 / 4 / 24** (unedited); equivalence run_pathway **8**, app_vs_service **14**, default_path **6**; vision **370 + 1 skipped**; web **83**; all passed | same log |
| content check, 97-run replay | **0** differences (`metrics/cc97_final2.json`) | same log |
| content check, 8 showcases | **0** differences (`metrics/cc8_final2.json`) | same log |
| content check, server job item-6 (enterobactin) | **0** differences (`metrics/cc_item6_final2.json`) | same log |
| PathWhiz import A/B, 17 files | see below | `C:/t/vis-runs/logs/vis1-pwab.log` |

The first final run (`e43ceb19`, `vis1-final.log`) had 1 focused failure (the interaction pair, fixed in
`49ab4da8`); its suite2 already passed in full.

**Replay.** Base = `git archive 7bf8588e` replayed with the DRAW-PATHWHIZ scripts (`replay97.py`, `show_replay.py`:
stored payloads, final-payload flag on, no network); after = the worktree. The base replay of item-6
(`spec_item6.json`, the server's own `final_mapped.json`) reproduces the PWML the server wrote byte for byte
(`live_eq=True`), the reference `C:/t/vis-ref/ours/ent-item6.pwml`.

**Content checker.** Copy of DRAW-PATHWHIZ's `content_check.py`, extended in three places, each named in its
docstring: (1) every `*_path` value inside `<options>` is masked, also a null one (replay 29: a degenerate base path
gave `start_arrow_path: null` where the new geometry gives a real head; key sets and every arrow flag are still
compared); (2) a compound location's template 157 reads as 81; (3) an edge's template 83 reads as 5. Everything else
(ids, names, references, states, locations, reactions, transports, interactions, hidden flags, other templates,
zindex, arrow flags, member sides) is compared as canonical XML.

**Layout metrics** (`layout_metrics.py --pathwhiz-sizes`, `sum_metrics.py`, `edge_metrics.py`; base -> after):

| | 97 runs | 8 showcases |
|---|---|---|
| edges with every point equal (zero length) | 480 -> **0** | 127 -> **0** |
| arrow edges with a degenerate tangent | 1 -> **0** | 0 -> 0 |
| visible edges that are curved (a control point off the chord) | 105 / 1556 (6.7 %) -> **663 (42.6 %)**; the rest are straight on the shaft | 46 / 426 (10.8 %) -> **215 (50.5 %)** |
| visible edges that are one cubic | 92.9 % -> 94.5 % | 89.2 % -> 90.6 % |
| edges carrying `<options>` / with an arrow flag | 1070 / 484 -> same (content) | 315 / 138 -> same |
| edge crossings | 75 -> **74** (9 runs better, 16 worse) | 136 -> **138**: lipid A 99 -> 104, enterobactin 14 -> 12, furochromone 3 -> 1, menaquinone 2 -> 1, sour taste 14 -> 15, ceramide 0 -> 1 |
| edges through a node / out of canvas | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 |
| long edges | 32 -> 33 | 16 -> 16 |
| node overlaps (all visible nodes) | 0 -> 79 | 0 -> 24 |
| node overlaps between attached nodes (what PathWhiz draws) | 0 -> **0** | 0 -> **0** |
| empty boxes / label-node overlaps / edges under a label | 0 / 0 / 0 | 0 / 0 / 0 |
| runs wider than 4:1 / aspect median | 11 -> 14 / 1.55 -> 1.79 | 2 -> 2 / 2.71 -> 2.76 |
| median canvas area | 2.8 -> 2.5 Mpx | 10.4 -> 10.4 Mpx |

The new node overlaps are the orphan twins of item 6, each drawn on its twin. Crossings that touch a shared end
(edges converging on one reaction point) went 27 -> 11; the others 48 -> 63, mostly lipid A (MsbA, open question 1 of
DRAW-PATHWHIZ) and fans where several nodes in one column meet one point (`crossings_where.txt`). The wider aspect:
the loose strip of stray boxes under the content is gone (canvas area down), and blocks with side compounds are taller.

**PathWhiz import A/B** (`scripts/pwab.sh`, a copy of DRAW-PATHWHIZ's `final4.sh` pattern; harness copied to
`C:/t/vis-pw1-work/harness/`, compose project **`t2pw-pwdraw`** only (db 3357, app 3054), PathWhiz `t2pw/latest`
`63959982` clone at `C:/t/pwpr/pwdraw-run`, a clean database per file; both arms imported fresh in one hold, then
`docker compose -p t2pw-pwdraw down`, volumes kept; Docker Desktop was already running for another session, none of
its containers touched). Same 17 files as DRAW-PATHWHIZ: the 8 showcases + V7b idx 7, 10, 14, 116, 126, 128, 11, 134,
114. Rows: `metrics/pathwhiz_{base,after}_results.jsonl`, table `metrics/pathwhiz_ab_final2.json`.

| 17 files | base (`7bf8588e`) | after (`49ab4da8`) |
|---|---|---|
| imports successful | 17/17 | **17/17** |
| compounds / proteins / complexes / reactions | 210 / 159 / 168 / 167 | same |
| transports / transport transporters / interactions | 32 / 18 / 39 | same |
| compound locations / nucleic acids / states / labels | 322 / 20 / 67 / 70 | same |
| importer messages | 89 | **89** |

No created count and no message count changed in any file (the totals are also those of the DRAW-PATHWHIZ A/B).
PathWhiz accepted transport-edge template 83 and drew those edges green.

What PathWhiz draws (screenshots `C:/t/vis-pw1-work/pw/{base,after}_final2/screenshots/`; after arm: 7 of 10, the
harness's sign-in click timed out on v7b_134, menaquinone and chlorophyll after their imports had succeeded; base arm
10 of 10). Enterobactin, base vs after: base already showed chevron heads (as the source reading says), straight gold
lines, ATP / PPi / AMP as long diagonals, right-angle transport loops, and pink 157 headers on 2,3-dihydroxybenzoate,
isochorismate and L-serine; after shows the gold Bezier curves with chevrons, ATP and PPi forking into and out of EntE
and AMP out of EntF, green transport edges through TolC and TonB on the membrane, and blue 81 headers throughout.

**Visual check** (renders with arrowheads: `scripts/render_arrows.py`, a wrapper that recomputes heads as PathWhiz
does from the `<options>` flags and calls the repo's `render.py` unchanged; the repo renderer reads heads only from
`option:*_path` elements, so its own renders of our PWML never show heads). Read and compared, base vs after:
item-6 / enterobactin showcase (straight chain chorismate -> EntC -> isochorismate -> EntB, NADH / NAD+ fork on the
lactate step, an ATP side compound on EntE, TolC and TonB on the membrane next to their inner cargo with curved inner
edges and no stray boxes; the outer edges to the extracellular box under the cell are still right-angle routes,
because every curve to it crosses a node), chlorophyll (the long chain
on one row, curved branches into CHLI / CAO / POR, GluTR1 stray box gone), menaquinone (one straight row), lipid A
(still crowded; MsbA), coenzyme A (TcyJ-N stray boxes gone), ceramide, replay runs 125, 137, 144, 7 (chlorophyll),
46, 29. Side-by-side page: `C:/t/vis-pw1-work/compare_final2.html` (not committed; PNGs under
`C:/t/vis-pw1-work/renders/{base,final2}/` and the PathWhiz screenshots under `C:/t/vis-pw1-work/pw/`).

## 4. Dead ends and things that went wrong

- First pass placed the transporter at any spot along the membrane with a clear line to its cargo: on item-6 TolC and
  TonB slid ~1400 units away and drew long curves across the cell. Capped at 300 units.
- First pass excluded the inner cargo from the transporter's free-spot test: in coenzyme A a three-protein complex
  landed on its cargo (3 attached overlaps). The cargo is now excluded only from the line test.
- Stray twins were first matched only between single-location blocks: TcyJ-N (members of one complex) stayed under the
  cell. Matching is now per member location. The protein test fixture carries a `compound-id` too, so the entity key
  checks `protein-id` first.
- Crossings, step by step on the 97 runs (base 75): curves with 0.25-chord arms at both ends 106; scoring candidates by
  crossings 93; dropping sides that face away (loops) 96; PathBank's 30-unit arm at the reaction end 83; side
  compounds only where one main member meets the enzyme side 75; final 74.
- `layout_metrics.py` walks five parents up from its own path to find `src`; the copy in a scratch dir raised
  `IndexError` until it relied on `PYTHONPATH` (noted in the copy).
- The queue waited about 30 min behind another session's `research-platform-backend` container (queue rule), and a
  few minutes behind `vis3-suite2`.
- The renders of the IR shaft connectors (template 83, drawn dashed by `render.py`) show a dashed line through each
  enzyme; PathWhiz drops them as orphans.

## 5. Open questions for the PO

1. **Drug box template.** v2 now draws every compound box as 81. PathBank keeps 157 (pink "Drug" header) for the drug
   a drug-action or drug-metabolism pathway is about. Options: (a) accept (cost: a drug pathway's own drug loses its
   pink header); (b) keep 157 for the compound named in the pathway name (needs the pathway name passed to
   `relayout`, a small writer change outside this boundary); (c) fix `select_compound_template_id` in
   `compound_templates.py` instead (the source of the 157; changes v1 output too).
2. **Lipid A** stays the most crossed diagram (99 -> 104): the shared transporter MsbA serves four membranes. Same
   options as DRAW-PATHWHIZ open question 1.
3. **Cycles around a rectangle** (TCA in PathBank) are not done; a cycle is still broken at the edge back to the
   earlier process and drawn left to right.
4. **Orphan locations** (no process refers to them): PathWhiz drops them; v2 now stacks the ones with a drawn twin on
   it. Whether the writer should stop emitting them (content change) is a separate question.
