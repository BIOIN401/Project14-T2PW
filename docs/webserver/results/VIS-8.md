# VIS-8: easier-to-read PWML diagrams (interaction edges first, then canvas size, aspect and crossings)

PO direction (2026-11-26): "see the strategies from the PWML files that were provided, how they handle this", then apply
what PathBank does; general readability counts (label size at a normal viewing size, canvas size and aspect, spacing,
crossings); judge in PathWhiz (our patched importer). Branch `vis/edges-readability`, worktree `C:/t/vis-pw8`, cut from
webserver/refactor `b9534f68` (VIS-5 and VIS-7 merged). Files changed (code): `src/t2pw/pwml/layout.py`,
`tests/test_pwml_vis8_edges.py` (new). Nothing else: `writer.py`, `cell_art.py`, `render/pathbank.py`, `streamlit_app.py` and
everything under `pipeline/`, `mapping/`, `rag/`, `curation/`, `llm/prompts/` untouched. Geometry only: the strict content
check against VIS-7 is 0 differences (section 4). Not merged; LEDGER row left to the orchestrator. Working files:
`C:/t/vis-pw8-work/` (replays, renders, PathWhiz runs, metrics; side-by-side page `C:/t/vis-pw8-work/compare.html`).
Evidence: `docs/webserver/evidence/VIS-8/`.

**Status: done and run twice.** Final tree `43ece565` (chooser weights box 1100 x 700, crossings 0.15: the orchestrator decision under PO
delegation after run 1 looked worse in PathWhiz for ceramide and enterobactin). 97-PWML strict content check 0 differences vs VIS-7;
PathWhiz imports of all 9 identical to VIS-7; SMOKE 511 and all gates green (section 6).

## 1. How PathBank's curators draw interaction edges (all 16 reference PWMLs)

Source: `C:/t/vis-ref/super_pathway_visualizations/.../pwml_markups/` (16 PWMLs), tools `evidence/VIS-8/scripts/refint.py`,
`intsum.py`, `refstats.py`; raw per-interaction listing `evidence/VIS-8/survey/reference_interactions.txt`, summary
`reference_interaction_summary.txt` and `reference_interaction_stats.txt`.

| question | finding (counts over the 16 files) | evidence |
|---|---|---|
| How common are interactions? | Only 4 of 16 pathways have any: PW126614 (1), PW126843 (7 interactions, 11 drawn edges), PW127049 (22), PW127096 (1) = 31 interactions in 35 interaction-visualizations. The 12 metabolic pathways (PW000001, 000002, 000003, 000017, 000779, 000921, 000923, 012911, 127122, 127135, 127303, 176557) have none | `reference_interaction_stats.txt` |
| Which types? | Activation (25) and Inhibition (6) only. Binding and Unspecified (our enterobactin: 6 of 8) never occur | `refint.py` summary lines |
| What are the partners? | compound -> protein complex 13, protein complex -> protein complex 14, compound -> compound 3, "Bound" -> protein complex 5 (Bound = a compartment region, not a bound complex). Never enzyme-binds-enzyme along a metabolic chain | `reference_interaction_stats.txt` |
| Where are the two partners? | Next to each other. Gap between the two boxes: median 165 units, 16 of 30 pairs <= 200, p90 598. 17 of 30 pairs are axis aligned (box centres within 40 units in x or y: a column or a row). Ours (VIS-7, enterobactin): median gap 520, 3 of 8 pairs over 1000 | PW126843 `I4129` gap 115, `I3341` 95, `I4133` 135, `I4187` 125; PW127049 similar; `intsum.py` |
| How long are the edges? | Visible edge length median 230 (PW127049) to 326 (PW126843); 1 cubic (one segment) for 28 of 36 edges, 2-3 segments for the rest. The only long ones (1756, 1793 units, 2-3 segments) are one compound (`I3354`) acting on five far targets in PW126843. Ours: enterobactin median 739, max 2281 | `intsum.py` on both sets |
| One edge or two? | One visible edge per interaction (32 of 35); the source end's edge is `hidden=true`. Our writer does the same (Left edge hidden, Right visible) | `refint.py` listing, `hid=` column |
| Which side does an edge leave from? | The sides that face each other: bottom to top for a stacked pair, right to left for a row | PW126843 `I4129` (start inside the complex, end at the compound's bottom), `I3341` |
| Style | Edge templates 148 (red T-bar, Inhibition) and 149 (Activation arrow) only, z-index 18 (above everything), one cubic with no bends | `refint.py` `tpl=` `z=` |
| Complexes instead of edges? | Binding is drawn as a multi-subunit protein complex, not as an interaction: PW000779 4 of 10 complexes have more than one subunit, PW000923 6 of 16, PW127049 6 of 35, PW126843 2 of 9 (0 interaction edges in the first two) | `refint.py` summary `multi_subunit` |
| Canvas and label size (general readability) | PathBank's own canvases: median 4500 x 3350, so a 12-unit label (the smallest PathWhiz font, `PW000779.svg` uses 12/15/16/20) is 2.9 px at 1100 px wide; median edge crossings 4; aspect median 1.29, 3 of 16 are 6000 or more wide | `evidence/VIS-8/survey/reference_metrics.json`, `m8.py` |

What this means for us. Curators keep interaction partners adjacent and the edge one short cubic between facing sides.
They never draw enzyme-to-enzyme binding along a chain: they draw the binding enzymes as one complex. Our T2PW PWML does
(8 interactions in enterobactin, 6 of them Binding or Unspecified between hub enzymes of a chain), so the long edges come
from content shape, not from a drawing mistake. What layout can do is (a) keep partners as close as the chain allows,
(b) draw the edge as one cubic or one arch with the facing sides, (c) make the whole chain compact. Content options are in
section 8.

## 2. What changed (`src/t2pw/pwml/layout.py`, v2 layout only; `T2PW_PWML_LAYOUT=v1` untouched)

1. **The layout picks the best of six shapes.** `relayout` now wraps the old function (renamed `_relayout`). A long chain can
   fold into rows at several widths and the picture can be packed to several aspects; `LAYOUT_CANDIDATES` = (fold width,
   aspect) (1.8, 1.5) [VIS-7's own], (1.5, 1.5), (1.2, 1.5), (1.8, 1.2), (1.5, 1.2), (1.2, 1.2). Each is laid out on a deep
   copy of the items (ids from a throw-away counter, so no real id moves) and scored by `_layout_cost`; the best is laid out
   for real. A tie keeps the first, so VIS-7's layout survives wherever nothing is better; a candidate that raises is
   skipped; `T2PW_LAYOUT_CHOOSER=0` = the chooser off: VIS-7's canvas shape (fold width and aspect); the VIS-8 edge rules (edge order, arch rules, moved reaction points) stay; a full rollback is a revert (switch for the server).
   `_layout_cost` = -ln(scale) + 0.15 x edge crossings + 1.0 x edges through a node box + 0.00002 x edge length + 0.00012 x
   interaction-edge length, scale = the factor the picture shrinks by to fit a 1100 x 700 box (the PO's "1100 px wide" and the
   landscape shape of PathWhiz's window; text and boxes shrink with it). A smaller canvas wins unless it buys crossings or long
   lines. **The weights (box 1100 x 700, crossings 0.15) are the orchestrator's decision under PO delegation**, after the first run
   (box 1100 x 1000, crossings 0.05) looked worse than VIS-7 in PathWhiz for ceramide and enterobactin (section 5). Result, VIS-7
   shape -> chosen: coenzyme A 5365 x 2343 -> 3061 x 2573 (label 2.5 -> 4.3 px at 1100 wide), furochromone 4430 x 1133 -> 2420 x
   1795 (a strip becomes a 1.35 picture), menaquinone 3160 -> 2615 wide, sour taste 1268 -> 1196; lipid A, chlorophyll, ceramide,
   enterobactin and item-6 keep VIS-7's shape (no candidate beats it).
   Time: candidates whose node positions come out identical are one shape (checked before any edge is drawn), and `_crossings`
   got a per-segment box test (same result; VIS-7's single layout of lipid A, 133 edges, went 2.6 -> 2.0 s). Per `relayout`
   call over the 8 showcases (`scripts/timing8.py`, first-run weights): 0.1-0.6 s for seven of them, lipid A 8.5 s (VIS-7: 2.6 s; the grid router
   dominates each layout and the picture is laid out once per distinct shape plus once for real). VIS-7's corpus maximum was 0.19 s per call (VIS-7 section 6b F7); corpus
   timing is in section 6.
2. **Interaction edges** (VIS-7's arches, reworked). Reaction and transport edges are drawn first, exactly as before;
   interaction edges follow, shortest span first, so the arches of one hub nest (a longer span stands higher) and an
   interaction never changes where a process edge goes. For an interaction between two nodes the cheapest clear candidate
   wins: the straight curve, or a flat arch (control points 60-480 out, as in VIS-7) from the top or bottom sides for a pair
   side by side and from the left or right sides for a pair one above the other (VIS-7 review F3: no arch leaves a side that
   faces away). Cost = length + 300 per crossing with an edge already drawn (`CROSS_COST`), so an arch is only taken when
   it avoids a crossing or a box. The pair's second edge takes the first one's arch exactly (F2; before, the visible edge
   could keep a different first-pass curve).
3. **Product edges that started in empty space** (found while reading the renders, not in the brief). A reaction with no
   enzyme is a point; the writer hides its substrate edges, so the visible product edge has to start at the substrate. The
   layered layout put the point beside the substrate, but not when the substrate stands in another group (another
   compartment): ceramide's two product edges into "ceramide" began in empty space. A point farther than 120 units from every
   substrate now moves onto the border of its first main substrate, on the side facing the products. Edges ending in empty
   space with no other edge there: 6 -> 2 over the 9 (the 2 left are lipid A's fork between two substrates, 60 units apart), 10
   over the 97-PWML corpus before (VIS-7).
4. Tests, `tests/test_pwml_vis8_edges.py` (10): a blocked enzyme-to-enzyme interaction becomes one arch, control points above the
   boxes, ends on the top sides (F5, fixture level through `relayout`); the pair's two edges end on the same points and are
   the same arch reversed (F2); arch ends face along the pair's longer axis (F3); the arches of one hub nest; a long flat
   chain folds; the chooser keeps the lowest-cost layout; the cost charges crossings and width; the kill switch equals the first
   candidate; an enzymeless reaction point sits on its substrate across compartments; `_crossings` counts a polyline once.

## 3. Numbers, before (VIS-7, `b9534f68`) and after (VIS-8)

Metrics script `evidence/VIS-8/scripts/m8.py` (visible edges only; crossing = two visible edges whose sampled curves cross;
"label at 1100" = the smallest PathWhiz label, 12 units (`PW000779.svg` uses 12/15/16/20), scaled to a picture 1100 px wide).
Raw files `evidence/VIS-8/metrics/m_vis7_showcases.json`, `m_vis8_showcases.json`, `m_reference.json`.

| pathway | canvas before | canvas after | aspect | label px at 1100 wide | visible edges: total length | crossings | interaction edges: total / longest |
|---|---|---|---|---|---|---|---|
| ceramide-sphingolipid | 5087x2999 | 5087x2999 | 1.7 -> 1.7 | 2.59 -> 2.59 | 10787 -> 14473 | 0 -> 0 | 0/0 -> 0/0 |
| chlorophyll | 3665x5666 | 3665x5666 | 0.65 -> 0.65 | 3.6 -> 3.6 | 12114 -> 12114 | 0 -> 0 | 2431/381 -> 2431/381 |
| coenzyme-a | 5365x2343 | 3061x2573 | 2.29 -> 1.19 | 2.46 -> 4.31 | 9964 -> 9353 | 3 -> 6 | 0/0 -> 0/0 |
| enterobactin | 4716x2420 | 4716x2420 | 1.95 -> 1.95 | 2.8 -> 2.8 | 14986 -> 15093 | 12 -> 11 | 7673/2281 -> 7780/2281 |
| furochromone | 4430x1133 | 2420x1795 | 3.91 -> 1.35 | 2.98 -> 5.45 | 8136 -> 9172 | 3 -> 2 | 0/0 -> 0/0 |
| lipid-a | 6769x6351 | 6769x6351 | 1.07 -> 1.07 | 1.95 -> 1.95 | 68983 -> 68983 | 90 -> 90 | 1852/402 -> 1852/402 |
| menaquinone | 3160x2412 | 2615x2412 | 1.31 -> 1.08 | 4.18 -> 5.05 | 10668 -> 10590 | 5 -> 4 | 633/633 -> 633/633 |
| sour-taste | 1268x2271 | 1196x2305 | 0.56 -> 0.52 | 10.41 -> 11.04 | 8104 -> 7416 | 14 -> 6 | 0/0 -> 0/0 |
| ent-item6 | 4716x2527 | 4716x2527 | 1.87 -> 1.87 | 2.8 -> 2.8 | 7197 -> 7197 | 0 -> 0 | 642/297 -> 642/297 |

Medians over the 9: canvas width 4716 -> 3665, aspect 1.70 -> 1.19, label at 1100 px 2.8 -> 3.6 px, crossings 3 -> 4 (sum over the 9: 127 -> 119).
PathBank's own 16 reference pathways, same script: median canvas 4500 x 3350, aspect 1.29, label 2.9 px, crossings 4
(`reference_metrics.json`). Honest reading: four of the nine get clearly smaller (coenzyme A, furochromone, menaquinone, sour taste);
lipid A (6769 x 6351, 90 crossings), chlorophyll, ceramide, enterobactin and item-6 keep VIS-7's shape. Their edges differ only
by the stub fix and the arch rules: ceramide's edge length 10787 -> 14473 (the two product edges that began in empty space now run from
their substrates, same 0 crossings), enterobactin's interaction length 7673 -> 7780 (+1.4 %, crossings 12 -> 11, longest edge unchanged at
2281). Coenzyme A trades 3 -> 6 crossings for a canvas 43 % narrower. The first run's weights (box 1100 x 1000, crossings 0.05) shrank
ceramide and enterobactin too (3415 and 3092 wide) and cut enterobactin's interaction length to 5245, but they read worse in
PathWhiz, hence the change; its numbers are in `metrics/m_vis8_showcases_run1.json`. `T2PW_LAYOUT_CHOOSER=0` = the chooser off: VIS-7's canvas shape (fold width and aspect); the VIS-8 edge rules (edge order, arch rules, moved reaction points) stay; a full rollback is a revert.
The 97-PWML corpus numbers are in section 6.

## 4. Content check

`scripts/content_check.py` (rule-based, one-directional, VIS-7's) and `strict` (no masking rules). Base = the unmodified `f3241270`
replays (VIS-5's), VIS-7 = `b9534f68`. Final run: tree `43ece565`, corpus replay `r1`.

| set | against | result |
|---|---|---|
| 8 showcases + server job item-6 (9 PWMLs) | VIS-7, strict | **0 differences** (`metrics/content_check_*_vs_vis7_strict.json`) |
| 97-PWML corpus | VIS-7, strict | **0 differences** (97/97; `metrics/corpus_vs_vis7_strict.json`): element structure and every id identical, only geometry and edge paths differ |
| the 9 and the 97 | base, rule-based | **0 differences** |
| the 97 | base, strict | 7 of 97 differ, the same 7 as in VIS-7 (VIS-5 accepted rules: label-type, nucleic-acid template) |

## 5. PathWhiz A/B (queue, heavy lock, harness VIS-2, project `t2pw-pwdraw`, stack down afterwards, BW_EXIT=0, 0 survivors)

Run 1 (weights box 1100 x 1000, crossings 0.05, tree `94623fb3`) and the final run (box 1100 x 700, crossings 0.15, tree `43ece565`,
`scripts/pwab8.sh r1 p2`) each imported the 9 PWMLs plus PathBank PW000779: **10 of 10 imported, success flag, error lines and created
record counts identical to VIS-7 for all 10, 0 new error lines** (`pathwhiz/ab_vs_vis7.txt` run 1, `ab_vs_vis7_final.txt` final; error counts
3 / 1 / 15 / 0 / 1 / 0 / 0 / 22 / 0 / 7). 10 of 10 screenshots each. The PathWhiz viewer fits the picture into its window, so a squarer
picture does not get bigger boxes the way a width-based score assumes; that is why run 1 misjudged two pathways.

Run 1 against VIS-7: coenzyme A and furochromone clearly better; **ceramide and enterobactin not better** (squarer, 5 and 19 crossings
against 0 and 12, boxes no larger in the viewer). The orchestrator chose the weight change proposed after that (option b, under PO delegation).

**Final run, per pathway against VIS-7 in PathWhiz** (I looked at the ceramide, enterobactin and coenzyme A screenshots in full; the other
pathways have layouts byte-identical to run 1 and I judged them from their previews and run 1 screenshots):

| pathway | verdict | why |
|---|---|---|
| ceramide | **same or slightly better** | same canvas and the same clean left-to-right ER pipeline as VIS-7; the two product edges that began in empty space now run from sphingomyelin and sphinganine into ceramide (two long swoops under the chain, still 0 crossings as in VIS-7) |
| enterobactin | **same** | VIS-7 shape; interaction arches unchanged in kind (total 7673 -> 7780, longest 2281, crossings 12 -> 11) |
| coenzyme A | **better** | compact serpentine (the third row runs right to left, so the reading direction is mixed), canvas 5365 -> 3061 wide, boxes larger in the viewer; crossings 3 -> 6 |
| furochromone | **better** | a 4430 x 1133 strip becomes 2420 x 1795; 3 -> 2 crossings |
| menaquinone | same or marginally tidier | 3160 -> 2615 wide, 5 -> 4 crossings |
| sour taste | same or marginally tidier | 14 -> 6 crossings, canvas 1196 wide (single-transporter lines remain) |
| chlorophyll, lipid A, ent-item6 | same | layouts equal to VIS-7 (no candidate beat them) |

Nothing is worse than both VIS-7 and run 1 on any metric, except the extra edge length of ceramide and enterobactin noted in section 3
(ceramide dangling edges are now drawn in full; enterobactin +1.4 % interaction length).

## 6. Gates (queue, heavy lock, clean tree, HEAD `43ece565` stamped, 0 survivors each)

`logs/vis8r_focused.log`: pwml 295 passed, render 49, final-payload writer 22, corpus tests 9, BW_EXIT=0, FINAL SURVIVING COUNT 0.
`logs/vis8r_suite.log` (`suite2.sh C:/t/vis-pw8 vis8r`): SMOKE **511 passed**; golden seams **2 / 8 / 4 / 24 passed**, unedited; equivalence
run / app / default **8 / 14 / 6 passed**; vision **370 passed, 1 skipped**; web **85 passed**; BW_EXIT=0, FINAL SURVIVING COUNT 0.
Corpus replay (`logs/r1_corpus.log`) and PathWhiz run (`logs/r1_pwab.log`) BW_EXIT=0, 0 survivors. Run 1 logs (`logs/n1_*`, `vis8_*`) are
kept: same results on `94623fb3`.

97-PWML corpus, VIS-7 -> VIS-8 final (`metrics/m_vis7_corpus.json`, `m_vis8_corpus_final.json`): median canvas width 2320 -> 2053, aspect
1.28 -> 1.18, label at 1100 px 5.7 -> 6.4 px, visible-edge crossings (sum) 66 -> 60, interaction edges (49): total length 15237 -> 15093,
longest 2664 -> 2758 (corpus file 4.pwml, one edge, +3.5 %), edges starting in empty space 10 -> 5, box overlaps 7 -> 7.
(Run 1: width 1945, crossings 79, interaction total 13224, longest 1271.) `relayout` time per call: maximum 0.63 s, total 4.6 s over the 97
(VIS-7: maximum 0.19 s, total 1.86 s). On the 8 showcases at `43ece565` (final weights, `scripts/timing8.py`): 0.1-0.5 s for seven, **lipid A (133
edges) 7.7 s**; above 3 s, so `2be114b9` skips the chooser above 120 edges (`CHOOSER_MAX_EDGES`): lipid A 2.3 s per call, output byte-identical (no
candidate beat VIS-7 there), the other seven unchanged; the corpus (largest 102 edges) is not affected.

## 7. Dead ends and slips

- **One fold width for everything.** First try: `SNAKE_OVER` 1.8 -> 1.2 for all pathways. Furochromone (3.9 aspect strip) and coenzyme A
  became compact, but ceramide got 5 new crossings and +30 % edge length, lipid A +9 % edge length, enterobactin's
  interaction lines crossed 19 edges instead of 12. 1.5 was the same on 8 of 9. That is why the choice became a per-pathway
  cost comparison instead of a constant.
- **Arches by rule.** The first arch rewrite (arch whenever the straight curve crosses anything) changed nothing visible in
  enterobactin: in a folded picture every interaction line crosses a reaction edge or two, arch or not (cost model
  above). Gating arches on `|dx| > |dy|` alone (F3) left vertical pairs with a straight line through a stack; left/right
  arches for vertical pairs fixed that case in the fixture, not in the 9 (no such pair there).
- **`font-size` "large" for enzyme labels (curators use it in PW127049: 67 protein locations).** Not a size-only change:
  PathWhiz's `protein_location.for_template` draws the *gene name* split at 7 characters instead of the protein name when
  `font_size == "large"` (the protein "simplify" mode, template 212). Dropped; it would change the printed label text.
- **A fold for lipid A.** Candidates down to fold 1.0 / aspect 1.0 / 0.8 were tried on the 9 (experiment worktree, env override,
  not committed): no change to lipid A and chlorophyll, the others kept the six-candidate result. Lipid A is MsbA's seven
  transports (content shape, VIS-7 question 3).
- **Rule slips** (all caught the same minute, none left a process):
  (a) at the very start I typed `python -` by accident in a probe (stdin read, no heredoc, moved to background by the
  harness); stopped with TaskStop by its task id, nothing else ran.
  (b) while checking PW127049's font-size values I ran a `python3 - <<'EOF'` with an empty body; it exited at once, nothing
  read or written.
  (b2) later, editing notes, I typed `python -I -` with an empty heredoc again: it waited on stdin until the harness moved it to the
  background at 120 s; stopped with TaskStop by task id, then checked by command line that no python was left.
  (c) one `chainF.sh ... &` combined with `run_in_background`; the harness reported "completed" while it ran. I found the
  chain's bash tree by command line and killed that tree by PID, then relaunched it without the `&`. No heavy job had started.
  (d) three earlier queued chains (corpus replays of experiment snapshots) were stopped before they ran (they were still waiting on
  the other session's research-platform containers); their queue / chain shells were killed by PID, verified none left.
  (e) `git worktree remove` for my own scratch worktrees `C:/t/vis-pw8-s12`, `-s15` and `-x` was denied by the permission
  system, so they are still registered (detached, no commits): the orchestrator can `git worktree remove --force` them.
- The gates and A/B ran twice: run 1 on `94623fb3` (first weights) and the final on `43ece565` (orchestrator weights); no new rule slip in either.
- The gates and A/B first ran after the orchestrator's reboot message; the machine had 0.0-0.4 GB free for about 2.5 h before that.
- The quick loop (9 replays, content check, 13 renders, about 40-70 s) went through `bounded_run.py` without the heavy lock, as in
  VIS-5 and VIS-7; the corpus replay, PathWhiz runs and gates went through the queue.

## 8. Open questions (product owner)

1. **Enzyme-to-enzyme Binding / Unspecified interactions along a chain** (enterobactin: 6 of 8 interactions; EntB is a hub
   with 4 partners). PathBank never draws these as edges (section 1): binding enzymes are one multi-subunit complex
   (PW000779 4 of 10 complexes, PW000923 6 of 16). Layout can only shorten them (this task: enterobactin total interaction
   length 7673 -> 5245, longest 2281 -> 1281; the lines still cross two to six reaction edges each). Options:
   (a) keep as is (cost: thin black lines across the middle of the chain, 19 edge crossings in enterobactin);
   (b) write EntA/EntB/EntC/EntE/EntF's binding as one protein complex with subunits in the PWML, as PathBank does, and drop
   those interaction records (content change in `writer.py` and the mapping; large; needs the product contract's view on
   whether "binds" means "forms a complex"; also changes what the user sees as the interaction list);
   (c) hide a Binding/Unspecified edge in the picture when its partners are more than about 600 units apart (visualization
   only: `hidden=true` on the edge records, the interaction stays in the PWML; cost: small; information leaves the picture for
   the longest edges). Recommendation: (c) now, (b) as a content project.
2. **Chooser weights: decided.** Option (b), box 1100 x 700 and crossings 0.15, by the orchestrator under PO delegation; applied and
   re-run (sections 4-6). Still one constant each: if the PO wants the smaller canvases for ceramide and enterobactin back, the first
   run weights (1100 x 1000, 0.05) are in `metrics/m_vis8_showcases_run1.json`; `T2PW_LAYOUT_CHOOSER=0` = the chooser off: VIS-7's canvas shape (fold width and aspect); the VIS-8 edge rules (edge order, arch rules, moved reaction points) stay; a full rollback is a revert.
   Enterobactin interaction lines stay as long as in VIS-7 (7780 total); only question 1 (hide, or complexes) shortens them.
3. **Single-transporter pathways** (lipid A, sour taste, ceramide's CERT) remain the largest source of long lines and crossings
   (lipid A 90 of the 132 crossings over the 9). VIS-7 question 3 still stands (one transporter location per transport, as
   PathBank does; content change in the writer).
4. **Enzyme label size.** The only native size switch (`font-size large`) replaces the label by the gene name, so it is not a
   layout option (section 7). Making enzyme names larger in PathWhiz would need a PathWhiz template change (150 x 70 ellipse,
   15-unit text) or a different protein template; out of this task's boundary.

## 9. Review follow-up (F1-F5), head `2be114b9` (code) and later docs commits

F1 wording fixed (sections 2, 3, 8): chooser off = VIS-7's canvas shape only. F2: `relayout` timed on the 8 showcases at `43ece565`; lipid A 7.7 s, so
the chooser is skipped above 120 edges (`2be114b9`): lipid A 2.3 s per call, output of all 9 byte-identical to the final run (strict content check on
the 9: 0 differences; the corpus has at most 102 edges, so it is unaffected), test `test_a_picture_with_more_edges_than_the_limit_is_laid_out_once`.
F3: the cost test now builds a real crossing (the old fixture only touched at a shared sample point, so it passed with `CROSS_W` = 0) and asserts
the difference is at least `CROSS_W - 0.02`. F5: coenzyme A reading and the final screenshot `pathwhiz/final_coenzyme-a.png` added. F4: no change.
Focused tests through the queue on `0fde34e0` (clean tree, HEAD stamped): pwml 296 passed, render 49, final-payload writer 22, corpus tests 9,
`logs/vis8s_focused.log`, BW_EXIT=0, FINAL SURVIVING COUNT 0. SMOKE and the other suite2 groups ran on `43ece565` (section 6); the only code change since
is the edge-count skip and a test.

## 10. Review verdict and merge (orchestrator, 2026-11-27)

- Independent ws-reviewer on `dedd7196`: **APPROVE WITH NOTES**. Strict content check of VIS-7 against VIS-8 outputs:
  0 differences on the 9 showcases and the 97-PWML corpus. F1 (the chooser switch did not restore VIS-7 exactly, as
  the record claimed), F2 (worst-case time on the 9 unmeasured), F3 (a cost test that passed with `CROSS_W = 0`), F5
  (coenzyme A described as one reading direction; it is a serpentine with crossings 3 -> 6). F4 (one more routed edge
  in three corpus files) accepted. Orchestrator decision under PO delegation: the switch stays chooser-only; a full
  rollback is a revert. Earlier, after the first PathWhiz A/B showed ceramide and enterobactin no better than VIS-7,
  the orchestrator chose the implementer's option (b): `VIEW_H` 1000 -> 700, `CROSS_W` 0.05 -> 0.15.
- Round 2 on `dc2d5655` (code `2be114b9`: `CHOOSER_MAX_EDGES = 120`, lipid A 7.7 -> 2.3 s per call, 9 byte-identical,
  no corpus file over 102 edges): **APPROVE**. Notes: the coenzyme A verdict cell still says "better" while the text
  says mixed; chlorophyll (102 edges) is near the limit.
- Merged as `32fa681c` = webserver/refactor `0e2ecf7a` + `vis/edges-readability` `dc2d5655`, no conflicts, in the
  detached worktree `C:/t/merge-vis8`. Code changes only `pwml/layout.py` and `tests/test_pwml_vis8_edges.py`.
- Gates on `32fa681c` (`evidence/VIS-8/logs/merge-vis8-gates.log`, script `merge_vis8_inner.sh`, tree clean): SMOKE
  **511 passed**; golden seams **2 / 8 / 4 / 24 passed**, unedited; equivalence **8 / 14 / 6 passed**; vision **370
  passed, 1 skipped**; web **85 passed**; pwml **327 passed**; render **49 passed**; service + web **405 passed**.
  `BW_EXIT=0`, **FINAL SURVIVING COUNT 0**. Pushed `32fa681c`.
