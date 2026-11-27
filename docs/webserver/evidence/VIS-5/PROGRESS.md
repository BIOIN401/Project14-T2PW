# VIS-5 progress log (vis/layout-iter, worktree C:/t/vis-pw5)

## base (f3241270, unmodified) -- 11:40
Images: C:/t/vis-pw5-work/renders/base/*.png (refs: renders/base/ref/). Content check base = this round. Wrapper: 46 s, FINAL SURVIVING COUNT 0.
Defects seen (all 9 looked at):
- Empty white compound boxes: (a) ChEBI ids PubChem has no xref for (ceramide, sphingomyelin, dihydroceramide, chlorophyll a/b, LPS, lipid A, cardiolipin, myristoyl-ACP) -> cached as .miss; (b) compounds with no id at all (most of chlorophyll, glucosylceramide, cysteinopantetheine, Ca++...) -> empty square; (c) production default budget 6 s for ~30 lookups.
- Long chains in one row (menaquinone 5491x1622 px, chlorophyll 6555 px): unreadable at 1100 px.
- Right-angle router loops (enterobactin EntC/EntB, coenzyme-a cysteine/pantetheine, furochromone visamminol, lipid-a LpxC/LpxA, ceramide S1P).
- Extracellular box below the cell, cargo far from its transporter: long green lines (enterobactin, item6, coenzyme-a, menaquinone); TolC/TonB both on the peptidoglycan line.
- Lipid A: 7 tiny leaflet boxes + MsbA fan of long lines across the picture; ceramide: Golgi edge starts in empty space, ER ceramide CERT line.
- Sour taste: transports between 5 separate boxes, long crossing lines.
- Chlorophyll: nucleus box of loose proteins; etioplast / thylakoid boxes; unattached enzymes (PORB/PORC) in a column.
- Labels small (protein text 15 units in 150x70 ellipses).

## r1 (3f9a7114, renderer only)
ChEBI-depiction fallback for structures, 20 s budget, name box for structureless compounds, bigger protein labels. Content check 0/9. Images renders/r1/.

## r2-r4 (layout.py, uncommitted until r5)
- Snake: a layered group wider than 1.3x the picture's square-ish width folds into rows alternating direction, each row starting under the previous row's last column (`_snake`, `Block.dir`; hub sides, side-compound pills mirrored in right-to-left rows). menaquinone 5491x1622 -> 2708x2056, chlorophyll 6555x3276 -> 3529x4374 (still tall: nested child boxes stack below).
- No chromosome room reserved when the cytoplasm holds a snake (its short last row has room).
- Enzyme-less reaction point: at the substrates' centre height on their side (Golgi ceramide edges no longer start in space).
- Isozymes / separately listed subunits of one reaction stack as overlapping ellipses on the first enzyme (PathBank PW000779 style) instead of loose ellipses in the column (chlorophyll Unknown x3, PORA/B/C, CHLI1/GUN4).
- Transporter on the membrane of its own (twin) location: TolC on the outer membrane line, TonB on the inner line (was: both on the peptidoglycan).
- Outside cargo (extracellular) of transports sits in one row just outside the cell under its transporter, label beside the row.
Content check 0/9 every round; wrapper 0 survivors every round.
Remaining: routed (right-angle) edges 50 (base 43), lipid-A leaflet boxes, sour-taste boxes, interaction lines (black 195) routed.

## COURSE CORRECTION (orchestrator/PO) -- acknowledged
Target is now the STRUCTURE of our PWML matching PathBank's reference PWMLs, judged by what PathWhiz draws, not our renderer.
Plan: (1) structural diff PathBank refs (PW000001, PW000779, PW012911, PW000017) vs ours (ent-item6, lipid-a) -> STRUCTURE_DIFF.md;
(2) writer/layout changes toward those conventions; renderer changes reverted (r1 reverted: structure fetching is dropped as a renderer issue; what PathBank PWMLs carry for structures recorded instead);
(3) PathWhiz A/B checkpoints via the VIS-1/VIS-2 harness through the heavy-lock queue.

## Structural diff done  (C:/t/vis-pw5-work/STRUCTURE_DIFF.md)
Same already: section order (41 sections), absolute coords, compound box 81 200x190 z10, membranes (closed M+8C, z6, per-organelle templates, Gram-neg 1/291/1), labels 235, art/zoom records, edge z18 / templates 5 & 83.
Gaps to fix in layout/cell_art: G1 multi-segment routed edges (PathBank: one cubic per edge); G2 membranes labelled ON their line (Outer Membrane / Peptidoglycan / Inner Membrane / organelle name), inside label separate; G4/G5 chromosome (1.6 + label) and organelle art (1.6-2.4) scale; G6 nucleic-acid-location uses protein template 2 (-> 12 "Nucleic Acid"); G13 leaflet states drawn as tiny boxes.
Open questions (content, not done): G7 currency compound templates + per-reaction duplicate locations; G8 enzyme cofactor tags (complex compounds); G9 protein templates by subunit count; G14 PathBank compounds carry moldb-smiles/inchi (structure) ours leave empty.
Renderer r1 reverted (6a7d0184).

## r6-r8 (7952e790)
- G6 nucleic-acid locations: template 2 -> 12 "Nucleic Acid" (content-check copy masks exactly this, documented in its header).
- G2 labels: an organelle/membrane box is named ON its membrane line (PathBank PW000779/PW012911); Gram-negative envelope lines named Outer Membrane / Peptidoglycan / Inner Membrane (names from the PWML's own locations); cytoplasm keeps its inside label. Boxes lose the 150-unit inside-label strip.
- G4/G5: chromosome up to scale 1.6 (room 560) + "Bacterial Chromosome" label at its centre; organelle cluster scale up to 2.0.
- G1 edges: before the grid router, single cubics with control points further out, then a smooth two-cubic bend (offset off the chord, or through a point just off a blocking box). Routed (3+ segment) edges 37 (base) -> 16; through-node 0.
- Outside cargo row skipped when another compartment box is in the way (menaquinone overlap bug from r5).
## r9-r10
- G13 lipid A: leaflet states become unoutlined regions inside their membrane (name tie-break: "IM"/"inner" -> the inner membrane box; cytoplasm -> the inner membrane box when two membrane boxes tie). lipid-a: 7 tiny boxes gone, one nested envelope.
- membrane line name only from exactly one non-apical/basolateral membrane location (sour taste showed "Apical" on three cell boxes).
Content check 0/9 each round, 0 survivors. Images renders/r6..r10. PathWhiz checkpoint on r8 queued (vis5-pwab-r8).

## r11-r17 (6be2af56, 5aec0d97, 71e3c0a8)
- r11: thylakoid nested in its chloroplast when an etioplast ties (chlorophyll: thylakoid box gone, now a region of the plastid); root boxes may stack in columns beside the main box when squarer (not chosen on the 9).
- r12: side chains join next to their first use (layering pulls a node to just before its first successor; coenzyme-a beta-alanine/aspartate chain moved next to PanC, the long arc gone); outside cargo row also for transports whose transporter stays put.
- r13: an enzyme-less reaction point whose substrates' middle falls on another node starts at its first substrate (coenzyme-a cysteinopantetheine arrow started at alpha-ketopantoate).
- r14: edges inside a box never cross its membrane band (furochromone bends left the box).
- r15-r17: canvas height trimmed to drawing + margin (enterobactin 2570 -> 2420); routed edges get cut corners; bends limited to 0.6x chord and never through a point behind an end (no hooks).
Edge metrics r17: routed (grid) edges 19 incl. lipid-a 8 (base 37 + 2-segment none); through-node 0 everywhere. Content check 0/9 every round, 0 survivors.
PathWhiz checkpoint r8 running (PW000779 ref OK 106 s, ceramide OK); focused tests queued (diagnostic, vis5-focused-a).

## "Unknown" enzymes (PO item)
Every drawn "Unknown" ellipse in the 9 (scripts/unknowns.py): ceramide 5 (dihydroceramide delta4-desaturase 1, sphingosine kinase, sphingomyelin synthase 1, ceramidase, sphingomyelinase), chlorophyll 8 (GSAT, PBG deaminase, UROD, CPO1, Protogen IX oxidase, PPO1, PPO, Mg-chelatase complex), furochromone 7 (SdPCS, SdPT, SdPC, SdCH, SdOMT, SdUGT1, SdUGT2), lipid-a 5 (LpxL x3, LpxM x2), menaquinone 2 (MenI, MenA), ent-item6 1 (LDH); coenzyme-a, enterobactin, sour-taste none.
Each is a member location of a protein-complex visualization whose COMPLEX carries the real enzyme name, written with label-type "protein". PathWhiz draws such a location with the complex name (PathWhiz checkpoint r8, ceramide: "ceramidase", "sphingomyelinase", "sphingosine kinase", "sphingomyelin synthase 1" shown). So the PWML is right and only OUR renderer printed the protein name "Unknown": fixed in render/pathbank.py (label-type protein + complex member -> complex name), the one renderer change, needed so the preview shows what PathWhiz shows. No writer/mapping change; protein 9659 find_by untouched.

## PathWhiz checkpoint 1 (r8 = 7952e790)
Queue vis5-pwab-r8: BW_EXIT=0, EXIT_import=0, stack down, FINAL SURVIVING COUNT 0. 5 files (ent-item6, lipid-a, ceramide, chlorophyll + PathBank PW000779 imported the same way) all success=true.
Import errors identical to VIS-2's final arm for ceramide / chlorophyll / lipid-a (same pre-existing content causes); created records identical except label_locations (lipid-a 19 -> 21, the envelope line names). evidence/VIS-5/pathwhiz_r8/.
Screenshots C:/t/vis-pw5-work/pw/r8/screenshots/ (f0000 PW000779, f0001 ceramide, f0002 chlorophyll, f0003 ent-item6, f0004 lipid-a): PathWhiz draws our geometry as our preview shows it: membrane names on their lines, Golgi edges from the ceramide, TolC on the outer line, outside row under the transporters, LDH / ceramidase / sphingomyelinase labels (complex names). Drawable art is a broken-image placeholder in this dev stack for both PathBank's file and ours (harness has no art files; VIS-2 known).
PathWhiz drops the unattached nodes (chlorophyll nucleus loose proteins etc.) -> r18 packs them tight so the gap they leave is small.
(Clock note: the hh:mm stamps I first wrote on the entries above were wrong and are removed; machine clock is UTC-06:00, base round ran ~10:40, this entry 12:25.)

## r18-r20 (0dd48390, da617ed8, d78a346b) -- 12:25
- r18: unattached nodes (PathWhiz drops them) pack 40 apart instead of a 160 column gap.
- r19: a transporter keeps clear of the column of a transporter on the other line of the same envelope (ent-item6: TolC no longer right under TonB).
- renderer (preview only): complex-member label-type "protein" shows the complex name (LDH etc.), as PathWhiz does.
- r20: a transporter may take a membrane-line label's place, the label slides along its line (a small mitochondrial matrix box had no room left for its transporter); empty-items crash fixed.
- Tests (queue, 0 survivors): diagnostic focused run vis5-focused-a found 9 failures = 1 crash on empty items (7 writer tests + 1 layout test) + chain-order test (snake, intended) + transporter test (line label took its spot, fixed in r20). After r20: test_pwml_l3_chain_layout + test_pwml_layout + test_pwml_writer + test_pwml_vis2_cell_art 56 passed; new tests/test_pwml_vis5_layout.py 1 passed. test_pwml_l3_chain_layout.test_v2_chain_reads_left_to_right updated to read the snake (geometry change intended).

## r21-r26 (2f61fce3, 6b9df305) -- 13:20
- r21: other small groups fill the empty end of a snake's last row (chlorophyll 3529x4998 -> 3138x4837 px).
- r22-r24: transporter keeps off membrane-line labels when its line has room; a covered label slides along its line or to the box's bottom line; a grid route may start under a label (sour taste had an edge through H+).
- r25-r26: **PathWhiz checkpoint 2 found a real structural limit: edges.path is varchar(255)** (schema.rb); a cut-corner route of 330+ chars made PathWhiz drop one ceramide edge ("Data too long for column 'path'"). Routed paths now budgeted at 220 chars (room for the art's shift), and a final guard redraws any edge over 255 as one cubic. Over-255 edges: r24 5 -> r26 0 (scripts/pathlen.py; base had 0). Test asserts it.
- Robustness: the 97-PWML replay corpus (VIS-2 replay97.py) through base f3241270 and through this tree: content check 97/97 identical (r24, r25, r26), no crash; routed edges 64 -> 32, through-node 0 -> 0; canvas median 2155x1662 -> 2320x1752, max width 7685 -> 5365, aspect > 3: 9 -> 3.

## PathWhiz checkpoint 2 (r20 = d78a346b) -- 14:00
vis5-pwab-r20: BW_EXIT=0, EXIT_import=0, 0 survivors; 7 files (ceramide, chlorophyll, ent-item6, lipid-a, enterobactin, menaquinone + PW000779) all success=true.
vs checkpoint 1: ent-item6 / chlorophyll errors identical (chlorophyll membranes 7 -> 6: thylakoid now a region of its chloroplast); enterobactin and menaquinone 0 errors. New errors ONLY "Data too long for column 'path'" (ceramide 1 edge, lipid-a 3 edges, + the visualization rows that pointed at them) = the varchar(255) limit, fixed in r25-r26 (0 edges over 255 since). Checkpoint 3 on r27 queued to confirm.
Process slip (own, reported): one command I wrote at 13:55 started `python.exe -I -X utf8 -` (stdin) by a typing error; it hung, the harness backgrounded it, I stopped it with TaskStop and checked no such python process was left. No effect on outputs.
Final gates queued on 2dbe69db (focused, then suite2), then checkpoint 3 (all 9 + PW000779).

## waiting on RAM -- 14:35
Final gates (vis5-focused-final, vis5-suite2) and PathWhiz checkpoint 3 are queued but blocked: free RAM 2-3 GB (< 6 GB gate). Docker's WSL VM holds ~9-10 GB since Docker Desktop was started for checkpoint 1; another session's containers (research-platform-*, plus several unnamed) are running in it now, so it is not mine to stop. Dropping the VM's page cache freed only ~1 GB on the host. No replay/render rounds run while below 6 GB (fast-loop rule).
Pending, unverified until RAM frees (uncommitted): multi-subunit complexes labelled with each subunit's own name (label-type "subunit", PathBank's way; one-subunit complexes and Unknown subunits keep the complex name). Content-check copy masks label-type subunit->protein. 9cd45f1d = formatting-only commit, to be confirmed byte-identical against r27.

## r28 (5f964ff7) + PathWhiz checkpoint 3 (r27 files) -- 15:35
- r28: multi-subunit complexes (no Unknown subunit) labelled per subunit (label-type subunit, PathBank's way): coenzyme A TcyJ/K/L/M/N instead of "TcyJKLMN" x5, chlorophyll Mg-chelatase subunits; one-subunit complexes and Unknown subunits keep the complex name (LDH etc.). Content check copy masks label-type subunit->protein. 9 showcases: only chlorophyll + coenzyme A changed (label-type only); 97-corpus: 6 files changed, 97/97 content-identical, 0 edges over 255. 9cd45f1d confirmed behaviour-neutral (other 7 PWMLs byte-identical to r27).
- Checkpoint 3 (vis5-pwab-r27, all 9 + PW000779): BW_EXIT=0, EXIT_import=0, stack down, 0 survivors; 10/10 success=true; import errors IDENTICAL to VIS-2's final arm for all 8 showcases (ceramide back to 1 error: the varchar(255) drop is gone); created records differ only in membranes / labels (visual). Screenshots: 9 of 10 failed on the login page under load (Playwright 30 s timeout, machine busy with another session's containers); only enterobactin's saved. Visual PathWhiz evidence stays checkpoints 1-2 (6 pathways); a screenshot-only rerun is pending RAM.
- Gates still queued behind the RAM gate (free 3 GB).

## Final gates -- 17:05 (RAM freed)
Lock/process check on the coordinator's request: no heavy lock held, no python process of mine alive.
vis5-focused-final (HEAD 9edbaa67, tree clean): pwml 281 passed, render 49, final_payload_writer_fixes 22, orch768 corpus 9; BW_EXIT=0, 0 survivors.
vis5-suite2 (same HEAD, clean): SMOKE 511, golden 2/8/4/24, equivalence 8/14/6, vision 370 + 1 skipped, web 83; BW_EXIT=0, 0 survivors.
Checkpoint 4 (vis5-pwab-r28, all 9 + PW000779, for the screenshots checkpoint 3 lost) queued.

## Wrap-up -- 18:10
Checkpoint 4 (screenshots of the final r28 PWMLs) waited ~1 h on the RAM gate (free 1.2-4.1 GB) and was cancelled: TaskStop + taskkill of my own leftover bw.sh waiter (PID 542816); no lock held, no t2pw-pwdraw container up, no process of mine left. Visual PathWhiz evidence: checkpoints 1-2 screenshots (r8/r20: PW000779, ceramide, chlorophyll, ent-item6, lipid A, enterobactin); import A/B for all 9 at r27 (checkpoint 3). r28 differs from r27 only in label-type of chlorophyll and coenzyme A.
Stopped iterating: the remaining defects are content-shaped (RESULT doc section 9). compare page: C:/t/vis-pw5-work/compare.html (before / after r28 / PathWhiz / PathBank).
