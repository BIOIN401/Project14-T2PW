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
   skipped; `T2PW_LAYOUT_CHOOSER=0` gives VIS-7's layout exactly (kill switch for the server).
   `_layout_cost` = -ln(scale) + 0.05 x edge crossings + 1.0 x edges through a node box + 0.00002 x edge length + 0.00012 x
   interaction-edge length, scale = the factor the picture shrinks by to fit a 1100 x 1000 box (the PO's "1100 px wide";
   text and boxes shrink with it). A smaller canvas wins unless it buys crossings or long lines. Examples, VIS-7 shape ->
   chosen: coenzyme A 5365 x 2343 -> 3061 x 2573 (label 2.5 -> 4.3 px at 1100 wide), furochromone 4430 x 1133 -> 2420 x 1795
   (a strip becomes a 1.35 picture), enterobactin 4716 x 2420 -> 3092 x 2758; lipid A and chlorophyll keep VIS-7's shape
   (no candidate beats it).
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
4. Tests, `tests/test_pwml_vis8_edges.py` (9): a blocked enzyme-to-enzyme interaction becomes one arch, control points above the
   boxes, ends on the top sides (F5, fixture level through `relayout`); the pair's two edges end on the same points and are
   the same arch reversed (F2); arch ends face along the pair's longer axis (F3); the arches of one hub nest; a long flat
   chain folds; the chooser keeps the lowest-cost layout; the cost charges crossings and width; the kill switch equals the first
   candidate; an enzymeless reaction point sits on its substrate across compartments.

## 3. Numbers, before (VIS-7, `b9534f68`) and after (VIS-8)

Metrics script `evidence/VIS-8/scripts/m8.py` (visible edges only; crossing = two visible edges whose sampled curves cross;
"label at 1100" = the smallest PathWhiz label, 12 units (`PW000779.svg` uses 12/15/16/20), scaled to a picture 1100 px wide).
Raw files `evidence/VIS-8/metrics/m_vis7_showcases.json`, `m_vis8_showcases.json`, `m_reference.json`.

| pathway | canvas before | canvas after | aspect | label px at 1100 wide | visible edges: total length | crossings | interaction edges: total / longest |
|---|---|---|---|---|---|---|---|
| ceramide-sphingolipid | 5087x2999 | 3415x2856 | 1.7 -> 1.2 | 2.59 -> 3.87 | 10787 -> 13402 | 0 -> 5 | 0/0 -> 0/0 |
| chlorophyll | 3665x5666 | 3665x5666 | 0.65 -> 0.65 | 3.6 -> 3.6 | 12114 -> 12114 | 0 -> 0 | 2431/381 -> 2431/381 |
| coenzyme-a | 5365x2343 | 3061x2573 | 2.29 -> 1.19 | 2.46 -> 4.31 | 9964 -> 9353 | 3 -> 6 | 0/0 -> 0/0 |
| enterobactin | 4716x2420 | 3092x2758 | 1.95 -> 1.12 | 2.8 -> 4.27 | 14986 -> 13037 | 12 -> 19 | 7673/2281 -> 5245/1281 |
| furochromone | 4430x1133 | 2420x1795 | 3.91 -> 1.35 | 2.98 -> 5.45 | 8136 -> 9172 | 3 -> 2 | 0/0 -> 0/0 |
| lipid-a | 6769x6351 | 6769x6351 | 1.07 -> 1.07 | 1.95 -> 1.95 | 68983 -> 68983 | 90 -> 90 | 1852/402 -> 1852/402 |
| menaquinone | 3160x2412 | 2615x2412 | 1.31 -> 1.08 | 4.18 -> 5.05 | 10668 -> 10590 | 5 -> 4 | 633/633 -> 633/633 |
| sour-taste | 1268x2271 | 1196x2305 | 0.56 -> 0.52 | 10.41 -> 11.04 | 8104 -> 7416 | 14 -> 6 | 0/0 -> 0/0 |
| ent-item6 | 4716x2527 | 3144x3080 | 1.87 -> 1.02 | 2.8 -> 4.2 | 7197 -> 8267 | 0 -> 1 | 642/297 -> 642/297 |

Medians over the 9: canvas width 4716 -> 3092, aspect 1.70 -> 1.08, label at 1100 px 2.8 -> 4.3 px, crossings 3 -> 4.
PathBank's own 16 reference pathways, same script: median canvas 4500 x 3350, aspect 1.29, label 2.9 px, crossings 4
(`reference_metrics.json`). At the same scale our 9 are now smaller than PathBank's (3092 vs 4500 wide), with the same
crossing level. Interaction edges: only enterobactin changes (total 7673 -> 5245, longest 2281 -> 1281), 3 of the 9 have
any. Honest limits: lipid A (6769 x 6351, 90 crossings) and chlorophyll (3665 x 5666) are unchanged; ceramide's chosen
shape has 5 crossings (VIS-7: 0) and 24 % more edge length, in exchange for a canvas a third narrower (5087 -> 3415) (the cost
model's trade; `T2PW_LAYOUT_CHOOSER=0` restores VIS-7). (97-PWML corpus numbers: pending)

## 4. Content check

TBD

## 5. PathWhiz A/B

TBD

## 6. Gates

TBD

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
2. **Chooser weights.** `_layout_cost` is my judgement (legibility first, crossings 0.05, interaction length 0.00012 per unit,
   viewing box 1100 x 1000). The effect on the 9 is in section 3. If the PO prefers fewer crossings over legibility raise
   `CROSS_W`; if the PO prefers a fixed shape, `T2PW_LAYOUT_CHOOSER=0` restores VIS-7 exactly. Cost of changing: one constant.
3. **Single-transporter pathways** (lipid A, sour taste, ceramide's CERT) remain the largest source of long lines and crossings
   (lipid A 90 of the 132 crossings over the 9). VIS-7 question 3 still stands (one transporter location per transport, as
   PathBank does; content change in the writer).
4. **Enzyme label size.** The only native size switch (`font-size large`) replaces the label by the gene name, so it is not a
   layout option (section 7). Making enzyme names larger in PathWhiz would need a PathWhiz template change (150 x 70 ellipse,
   15-unit text) or a different protein template; out of this task's boundary.
