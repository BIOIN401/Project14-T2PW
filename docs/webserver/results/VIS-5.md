# VIS-5: our PWML laid out like PathBank's curated PWMLs (layout iteration)

PO request (2026-11-26): our pathway renderings do not look like curated PathBank diagrams yet; fix it in our
PWML-generating code, not in the renderer, iterating render -> compare -> fix. Course correction the same day (PO,
binding): the target is the STRUCTURE of our PWML matching PathBank's reference PWMLs, judged by what PathWhiz itself
draws; the renderer under `src/t2pw/render/` is not the goal; structure-image fetching (empty boxes in our preview) is
dropped. Second PO item: an enzyme the paper names must show its name even when its protein is PathBank's Unknown
(9659).

Branch `vis/layout-iter`, worktree `C:/t/vis-pw5`, cut at `f3241270` (= webserver/refactor + VIS-1 + VIS-4 + VIS-2).
Files changed (code): `src/t2pw/pwml/layout.py`, `src/t2pw/pwml/cell_art.py`, `src/t2pw/render/pathbank.py` (one
preview fix, section 4), `tests/test_pwml_l3_chain_layout.py` (one geometry assertion, section 6),
`tests/test_pwml_vis5_layout.py` (new). Nothing under `pipeline/`, `mapping/`, `rag/`, `curation/`, `llm/prompts/`;
`streamlit_app.py` untouched; the writer (`pwml/writer.py`) untouched. Not merged; LEDGER row left to the orchestrator.
Working files (replays, renders, PathWhiz runs; not committed): `C:/t/vis-pw5-work/` (log: `PROGRESS.md`).
Evidence: `docs/webserver/evidence/VIS-5/`.

## 1. Method

Dev loop `evidence/VIS-5/scripts/quick.sh` (VIS-2's, pointed at this tree): replays the 8 showcase payloads and server
job item-6 through this tree's PWML export (no LLM, no network), runs the content check against the unmodified base
(`f3241270`) and renders the 9 PWMLs and 4 PathBank reference PWMLs (PW000001, PW000779, PW012911, PW000017) with the
tree's own renderer as a fast preview. Each round through `bounded_run.py` (light, no heavy lock, per the brief);
every round FINAL SURVIVING COUNT 0 and content check 0 differences on all 9. 28 rounds (base, r1-r28); r1 (renderer structures) was reverted after the PO's course correction. PathWhiz
checkpoints (the real importer and viewer, VIS-2 harness, project `t2pw-pwdraw`) through the heavy-lock queue.

## 2. Structural diff (PathBank vs ours)

`evidence/VIS-5/STRUCTURE_DIFF.md` (table G1-G14; tools `scripts/survey.py`, `survey_cmp.py`, `lbl.py`, `edges10.py`;
side-by-side field histograms `survey_cmp.txt`). Same already: section order (41 sections), absolute coordinates,
compound box 81 200x190 z10, membrane records, label template 235, art and zoom records, edge templates 5/83, z18.
Gaps fixed here: G1 edge paths (PathBank: one cubic per edge), G2 membranes named on their lines, G4/G5 art scale and
chromosome label, G6 nucleic-acid template, G13 leaflet boxes. Gaps that need content (open questions, section 7):
G7 currency compound templates and per-reaction copies, G8 enzyme cofactor tags, G9 protein templates by subunit
count, G14 compound structures (`moldb-smiles` / `inchi` empty in ours, filled in PathBank's).

## 3. What changed (layout v2 only; `T2PW_PWML_LAYOUT=v1` untouched)

Geometry and visualization records only (location x/y, edge paths and their arrow paths, membranes, labels, art,
canvas size), plus the nucleic-acid template (G6, item 11) and the subunit label-type (item 13). All in `pwml/layout.py` unless noted.

1. **Long chains snake** (`_snake`): a layered group wider than 1.8x the picture's square-ish width folds into rows read
   left to right, then right to left, each row starting under the previous row's last column; hubs, side-compound pills
   and enzyme-less reaction points are mirrored in right-to-left rows (`_Block.dir`). Other small groups of the box fill
   the empty end of the last row. Menaquinone 5491x1622 px -> 2708x2056.
2. **Side chains join where used**: layering puts every node just before its first successor (was: pure sources only);
   coenzyme A's aspartate -> beta-alanine chain now ends at PanC instead of a long arc.
3. **Isozymes stack**: the other enzymes of one reaction (no other use) overlap the first one 10/38 units lower, as
   PathBank's PW000779 stacks (chlorophyll's loose "Unknown" x3 / PORA-B-C columns).
4. **Enzyme-less reaction point** at its substrates' side and centre height, or at its first substrate when that
   middle is another node (the writer hides the substrate edges; the visible arrow used to start in empty space:
   ceramide Golgi, coenzyme A cysteinopantetheine).
5. **Transports**: the transporter sits on the line of its own membrane (its undrawn twin location: TolC on the outer
   membrane, TonB on the inner); keeps clear of the column of a transporter on the other line of the same envelope;
   outside cargo (an unoutlined outside compartment) in one row just outside the cell under its transporter, label
   beside the row, only where no other box is in the way.
6. **Labels (G2)**: a compartment with its own membrane is named on that line (PathBank PW000779 / PW012911); a
   Gram-negative envelope's three lines are named Outer Membrane / Peptidoglycan / Inner Membrane (from the PWML's own
   location names); the cell line from the PWML's one cell-membrane location; the cytoplasm keeps its inside label.
   A transporter keeps off a line label when the line has room, else the label slides along its line or to the box's
   bottom line.
7. **Edges (G1)**: before the grid router, single cubics with control points further out, then one smooth bend (two
   cubics through a point off the chord, at most 0.6x the chord deep, or through a point just off a blocking box, never
   behind an end); routed edges get cut corners; edges between two nodes of one box do not cross its membrane band.
   Routed (3+ segment) edges 37 -> 20 over the 9 (lipid A 16 -> 8), through-node 0 (`edge_metrics_*.txt`).
8. **Compartments (G13)**: a leaflet with an ambiguous parent goes to the membrane its name says ("IM"/"inner"); the
   cytoplasm likewise to the inner membrane box; a thylakoid to the chloroplast rather than an etioplast; a leaflet is
   an unoutlined region of its membrane's box (lipid A: 7 tiny boxes gone). Root boxes may stack in columns beside the
   main box when that is squarer.
9. **Compactness**: nodes no process refers to (PathWhiz's viewer drops them) pack 40 apart, not a column gap; no
   chromosome room reserved where a snake leaves space; canvas height ends a margin below the drawing.
10. **Art (G4/G5, `cell_art.py`)**: chromosome at up to scale 1.6 (room 560) with a "Bacterial Chromosome" label at its
    centre, as PW000779; organelle cluster up to 2.0 (PathBank 1.6-2.4).
11. **Nucleic acids (G6)**: a nucleic-acid location the writer gave template 2 ("Enzyme Monomer I", a ProteinLocation
    template) gets template 12 ("Nucleic Acid", NucleicAcidLocation, same 150x70). PathWhiz drew gene clusters and
    promoters as enzyme ellipses before.

12. **Edge path length**: PathWhiz stores `edges.path` in a `varchar(255)` (`db/schema.rb`); PathWhiz checkpoint 2
    showed a longer path makes the edge's INSERT fail and drops it (and the visualization rows pointing at it). Routed
    paths are budgeted at 220 characters (room for the cell art's later shift) and a final guard redraws any edge
    over 255 as one cubic between its ends. Over-255 edges: 0 on the 9 and on the 97-PWML corpus (base: 0 too; it
    only arose from the cut-corner routes of r15).

13. **Subunit labels (r28)**: a complex visualization with two or more drawn subunits, none of them PathBank's
    Unknown, labels each subunit with its own protein name (label-type `subunit`, as every PathBank reference does);
    the writer's `protein` made PathWhiz print the complex name on every subunit (coenzyme A: "TcyJKLMN" five times,
    chlorophyll's Mg-chelatase). One-subunit complexes and Unknown subunits keep `protein`, so the enzyme name the
    paper gives still shows (section 4). Changed files: chlorophyll and coenzyme A (label-type only); 6 of the 97
    corpus PWMLs.

New record types: none. Every record and field written is one the PathBank reference PWMLs or our previous output
already carry (label-location 235, membrane-visualization, drawable-element-location, zoom-visualization, edge
paths with `C` segments). Template 12 is new to our output and present in PathWhiz's `visualization_templates`
(NucleicAcidLocation); PathWhiz A/B below.

## 4. "Unknown" enzymes (PO item)

`scripts/unknowns.py` over the 9: every drawn "Unknown" ellipse (ceramide 5, chlorophyll 8, furochromone 7, lipid A 5,
menaquinone 2, ent-item6 1) is a member location of a protein-complex visualization whose COMPLEX carries the paper's
enzyme name (LDH, ceramidase, GSAT, LpxL, MenA ...), written with label-type `protein`. PathWhiz draws such a location
with the complex name (checkpoint 1, ceramide and ent-item6: "ceramidase", "sphingomyelinase", "LDH" shown). The PWML
was right; only our preview printed the protein's own name. Fix, the one renderer change: `render/pathbank.py` shows
the complex name for a complex member of label-type `protein`, as PathWhiz does. No writer, mapping or id change;
protein 9659's find_by untouched.

## 5. Content check

`scripts/content_check.py` (VIS-2's copy): every element, id, name, reference, state, location, reaction, transport,
interaction, hidden flag, template, zindex, arrow flag compared as canonical XML with geometry, paths and the
membrane / label / art / zoom sections masked. Two masks added, as VIS-1 did for 157 -> 81: a nucleic-acid
location's template 12 reads as 2 (G6), and a protein location's label-type `subunit` reads as `protein` (item 13;
the PO allowed label-type changes for enzyme names). Result every round: 0 differences on 8/8 showcases and item-6;
the 97-PWML replay corpus (VIS-2 `replay97.py`, base tree `f3241270` vs this tree): 97/97 identical at r24, r25, r26
and r28, no crash.

## 6. Tests

Updated (geometry change intended): `test_pwml_l3_chain_layout.py::test_v2_chain_reads_left_to_right` now reads the
chain along the snake (a step stays in its row and moves in that row's direction, or goes to the next row; the first
row runs left to right). No content-equality test edited. New: `tests/test_pwml_vis5_layout.py` (snake, enzyme stack,
membrane-line label, nucleic-acid template).

Test runs so far (queue, heavy lock, 0 survivors each): diagnostic `vis5-focused-a` on r17 found 9 failures = an
empty-items crash (7 writer tests + 1 layout test, fixed in r20), the chain-order test (snake, intended, updated) and
the transporter test (a membrane-line label took its spot, fixed in r20); after r20 `test_pwml_l3_chain_layout`,
`test_pwml_layout`, `test_pwml_writer`, `test_pwml_vis2_cell_art`: 56 passed; `test_pwml_vis5_layout`: 1 passed.
Final gates: section 8.

## 7. Open questions (product owner)

1. **Currency compounds (G7)**: PathBank draws H2O, ATP, NAD(H) ... as small pills / icons (templates 9, 42-65), one
   location per reaction (PW000779: H2O x4). Our writer already picks the small templates for known currency compounds
   (ent-item6: ATP, NADH, NAD+ pills) but shares one location per compound and state, so a busy cofactor draws long
   edges to one place. Option: one location per reaction for currency compounds (new location records: a structural
   content change of the PWML, needs a content-check rule and a PathWhiz A/B). Cost: medium plus review. Not done.
2. **Enzyme cofactor tags (G8)**: PathBank shows Fe2+, FAD, PLP as tags under the enzyme
   (`protein_complex_compound_visualizations` + template-9 compound + hidden template-10 edge). Our complexes carry
   no compound members, so this needs content from extraction; not done.
3. **Protein templates by subunit count (G9)**: PathBank uses dimer/tetramer/multimer/transporter templates; ours
   always 2. Writer choice from subunit counts / transporter role; visual, but a compared field.
4. **Compound structures (G14)**: PathBank compounds carry `moldb-smiles`, `moldb-inchi`, `moldb-inchikey`; ours leave
   every `moldb-*` empty and many intermediates have no id. Filling them from ChEBI/PubChem would be new compound
   content; the result page's empty boxes are a renderer issue (dropped per PO; an earlier renderer attempt, r1, was
   reverted in `6a7d0184`).
5. **Unknown enzyme styling**: no quiet styling was needed once the complex names show; the PWML keeps label-type
   `protein`.
6. **Content oddities that layout cannot fix**: lipid A's MsbA carries seven transports between leaflets and the outer
   membrane (one transporter, long lines); sour taste has three cytoplasm boxes (Type III cell, an auto state,
   cytosolic) with 5-HT / H+ transports between them; menaquinone has two cytoplasm states (intracellular,
   cytoplasmic).

## 8. Gates and PathWhiz A/B

**PathWhiz** (VIS-2 harness `C:/t/vis-pw2-work/harness/run_import.py`, project `t2pw-pwdraw`, PathWhiz clone
`C:/t/pwpr/pwdraw-run`; `evidence/VIS-5/scripts/pwab.sh`; queue, heavy lock; each BW_EXIT=0, EXIT_import=0, stack down,
FINAL SURVIVING COUNT 0):

| checkpoint | tree | files | result |
|---|---|---|---|
| 1 `vis5-pwab-r8` | 7952e790 | ceramide, chlorophyll, ent-item6, lipid A + PathBank PW000779 | 5/5 success; errors identical to VIS-2's final arm; screenshots of all 5 |
| 2 `vis5-pwab-r20` | d78a346b | + enterobactin, menaquinone | 7/7 success; new errors only `Data too long for column 'path'` (ceramide 1, lipid A 3 edges): PathWhiz's `edges.path` is varchar(255) -> fixed r25-r26 |
| 3 `vis5-pwab-r27` | 2dbe69db | all 9 + PW000779 | 10/10 success; import errors IDENTICAL to VIS-2's final arm for all 8 showcases (`pathwhiz_r27/ab_vs_vis2.txt`); created records differ only in membrane / label records; screenshots: 9 of 10 timed out on the login page under machine load (only enterobactin saved) |

What PathWhiz draws (checkpoints 1-2 screenshots, `C:/t/vis-pw5-work/pw/r8|r20/screenshots/`, on the compare page):
our geometry as our preview shows it: envelope lines named on their lines, TolC on the outer membrane, outside cargo
under its transporter, Golgi edges starting at the ceramide, complex names (LDH, ceramidase, sphingomyelinase) on the
Unknown-protein enzymes, nucleic acids as PathWhiz's yellow Nucleic Acid shape. Drawable art and compound structures
are placeholders in this dev stack, for PathBank's own PW000779 too.

**Unit / suite gates** (queue, heavy lock, run on HEAD `9edbaa67` = code `5f964ff7` + docs, `git status` clean in both
logs; `evidence/VIS-5/gates/`):

| gate | result |
|---|---|
| `vis5-focused-final`: `tests/test_pwml_*.py` | 281 passed |
| `tests/test_render_*.py` | 49 passed |
| `tests/test_final_payload_writer_fixes.py` | 22 passed |
| `tests/test_orch768_replay_corpus.py` | 9 passed |
| wrapper | BW_EXIT=0, FINAL SURVIVING COUNT 0 |
| `vis5-suite2`: SMOKE | 511 passed |
| golden seams (batch / c011 / stage8 / quarantine), unedited | 2 / 8 / 4 / 24 passed |
| equivalence (run_pathway / app vs service / default path) | 8 / 14 / 6 passed |
| vision | 370 passed, 1 skipped |
| web | 83 passed |
| wrapper | BW_EXIT=0, FINAL SURVIVING COUNT 0 |

The gates waited ~3 h on the queue's 6 GB free-RAM gate (Docker's WSL VM with another session's containers plus the
user's applications held the machine at 1.6-4.6 GB free from ~14:00 to ~17:00).

## 9. Remaining defects (after r28)

- Lipid A: MsbA (one transporter for seven transports between cytoplasm, leaflets, periplasm and outer membrane) stays
  in the cytoplasm with long lines to the outer membrane's nodes; 8 grid-routed edges remain there (content shape,
  open question 7.6).
- Sour taste: three cytoplasm boxes with 5-HT / H+ transports between them give long crossing lines (content shape).
- Menaquinone: two cytoplasm states (intracellular, cytoplasmic) side by side; transports cross a box edge.
- Readability at 1100 px: a 30-node pathway is still 2700-4500 units wide (PathBank's own PW000779 is 4400); labels
  are legible zoomed, not at 1100 px. Snaking and packing reduced the worst cases (max width on the 97-PWML corpus
  7685 -> 5365, aspect > 3 from 9 to 3).
- Compound boxes without a structure in our preview: renderer issue, dropped per PO; PathWhiz draws from its own
  database (open question 7.4).
- Edge routing: 20 grid-routed edges over the 9 (base 37), 32 over the 97 corpus (base 64); none through a node.
