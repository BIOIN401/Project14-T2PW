# Draw / PathWhiz: PathBank-style geometry in our PWML (branch `draw/pathwhiz`, from main `fd814b8e`)

Goal (PO notes 2026-11-19): the diagram PathWhiz draws from our PWML should look like a curated PathBank diagram.
Only geometry changes: membrane visualizations, label locations, x/y of locations, edge paths, canvas size. Every
compound, protein, reaction, transport, interaction, location, state and id stays identical.
Not merged. Code commit `4b701677`. Working files: `C:/t/lane-pwdraw-work/`. Evidence: `evidence/draw_pathwhiz/`
(DB scripts and statistics, metrics, content checks, PathWhiz A/B rows, every script used).
Before/after screenshots and the side-by-side page: `C:/t/lane-pwdraw-work/compare/index.html` (9.8 MB, not committed).

## 1. Conventions

Mined from 268 de-duplicated curated PathBank diagrams, newest first, weighted by recency: `PATHBANK-CONVENTIONS.md`.
Short version: Plasma Membrane (template 1, cyan, 50-unit band) is the cell and encloses the organelles; each
organelle has its own template and colour; nucleus, mitochondria and chloroplasts are double (gaps 70 / 74 / 94);
transporters sit on the membrane line (median 8 units) with cargo on both sides (96 %); the extracellular space has
no outline; substrate-to-product step median 667; content aspect median 1.4.

## 2. What changed (`src/t2pw/pwml/layout.py`, layout v2 only; `T2PW_PWML_LAYOUT=v1` is untouched)

1. **Membrane template and colour per compartment kind.** `compartment_kind()` reads the subcellular location name
   (else the state name) and maps it to PathBank's template: Plasma Membrane 1 (cell, plasma/inner/outer membrane,
   periplasm), Mitochondria 84, Nucleus 93, ER 91, Golgi 146, Lysosome 92, Peroxisome 94, Melanosome 95, Vacuole 238
   (vacuoles and vesicles, as curators draw synaptic vesicles), Chloroplast 237, Cell Wall 304. Unknown kinds keep
   template 220 (Simple Membrane). The outside of the cell (extracellular space, saliva, ...) gets **no outline**,
   only its label, as in PathBank. In the 97 replay PWMLs: Plasma 83, Mitochondria 22, ER 13, Simple 7, Vacuole 6,
   Nucleus 6, Chloroplast 4, Golgi 3, Lysosome / Peroxisome / Cell Wall 1 each.
2. **Double membranes.** Nucleus, mitochondrion (and a mitochondrial matrix drawn on its own) and plastids get a second
   line of the same template inside the first, 70 / 74 / 94 units in (the curated medians).
3. **Nesting.** `compartment_parents()`: organelles sit inside the cell; the Gram-negative envelope nests outer
   membrane > periplasm > inner membrane > cytoplasm; leaflets sit inside their membrane; the matrix and inner
   membrane sit inside the mitochondrion, the thylakoid inside the plastid. A nested compartment whose membrane the
   parent already draws (cytoplasm in its membrane, periplasm inside the outer membrane, matrix inside the
   mitochondrion) is a label only, not a second box. Each box holds its own groups, then its child boxes in rows.
4. **Transports across the membrane.** A transporter (or a transport point without one) that serves transports only
   is moved onto the membrane line between its cargo compartments, on the side facing the outer cargo, at the inner
   cargo's x, sliding sideways to a free spot; on a double membrane it sits between the two lines. A transporter
   shared by transports across different membranes (MsbA in lipid A) stays in its group.
5. **No empty boxes.** As before, a box exists only for a compartment holding a node PathWhiz draws; the outside of
   the cell no longer gets an outline at all.
6. **Spacing and crossings.** HGAP 110 -> 160, VGAP 40 -> 60 (curated step median 667), band gap 40 -> 80 (50-unit
   bands must not touch), target aspect 2.0 -> 1.5 (curated 1.4). Inside each group, 24 barycenter sweeps instead of
   8, keeping the order with the fewest crossings (groups up to 400 edges).
7. **Labels** sit inside the band (offset by half the band and any double gap).

**Tests** (`tests/test_pwml_layout.py`, not a golden-seam file): `test_v2_quality` now checks the PathBank rules (at
most two lines per compartment, membranes nest or stay apart, no empty membrane) instead of "one membrane per state,
holding only its own state", which nesting and the label-only outside of the cell contradict by design;
`test_relayout_edge_cases_and_determinism` checks the templates `[1, 93, 93]` (cell, double nucleus) instead of a
count of 2. Two tests added: compartment kinds and nesting; a transporter on a matrix's double membrane.
**No golden-seam test was edited and no golden PWML expectation changed.**

## 3. Gates (all on `4b701677`, one heavy-lock hold, `C:/t/vis-runs/queue.sh` holder `pwdraw`, BW_EXIT 0, 0 survivors)

Log: `C:/t/lane-pwdraw-work/logs/final2.log`; script `evidence/draw_pathwhiz/scripts/final2.sh`.

**Content check** (`content_check.py`, the layout lane's checker: everything but geometry, membranes and labels as
canonical XML). Base = `git archive fd814b8e` replayed with `replay97.py` / `show_replay.py` (stored payloads, flag on,
no network; the base reproduces the 8 showcase PWMLs in `C:/t/showcase2` byte for byte).

| set | files | content differences |
|---|---:|---:|
| 97-run replay | 97 | **0** (`cc97_final.json`) |
| 8 showcases | 8 | **0** (`cc8_final.json`) |

The HEAD replay is byte-identical to the earlier after2 replay (97/97), so the replay is deterministic.

**Tests.** Focused (14 `test_pwml_*.py` files, `test_final_payload_writer_fixes.py`, `test_orch768_replay_corpus.py`):
298 passed. `suite2.sh`: SMOKE 511 passed; golden seam 2 / 8 / 4 / 24 passed (unedited); equivalence run_pathway 8,
app_vs_service 14, default_path 6; vision 357 passed + 1 skipped; web 83 passed.

**Layout metrics** (`layout_metrics.py --pathwhiz-sizes`, base -> after):

| | 97 runs | 8 showcases |
|---|---|---|
| edge crossings | 81 -> **71** | 121 -> 141 (lipid A 90 -> 109; others 31 -> 32) |
| long edges | 38 -> 34 | 15 -> 14 |
| node overlaps / edges through a node / empty boxes | 0 / 0 / 0 -> 0 / 0 / 0 | same |
| membrane paths | 167 -> 147 | 33 -> 29 |
| runs wider than 4:1 | 10 -> 12 | 2 -> 2 |
| median canvas area | 2.0 -> 2.8 Mpx | 7.9 -> 10.4 Mpx |
| nodes outside any outline | 0 -> 87 | 0 -> 15 |

"Nodes outside any outline" are the extracellular nodes (no outline, by convention) and transporters sitting on a
membrane line: intended.

**PathWhiz import A/B** (`pathwhiz_ab.json`; rows `pathwhiz_*_results.jsonl`). Harness: copy of the showcase harness
(screenshots incl. zoom) in `C:/t/lane-pwdraw-work/harness/`, compose project **`t2pw-pwdraw`** (db 3357, app 3054),
clone of `t2pw/latest` `63959982` at `C:/t/pwpr/pwdraw-run`, MolDB on closed port 9, clean database per file.
Files: the 8 showcase PWMLs + V7b idx 7, 10, 14, 116, 126, 128, each arm. Docker Desktop was stopped; started without
admin. Stack taken down after (`docker compose -p t2pw-pwdraw down`); volumes `t2pw-pwdraw_bundle` and
`t2pw-pwdraw_mysql_data` kept.

| 14 files | base | after |
|---|---|---|
| imports successful | 14/14 | 14/14 |
| compounds / proteins / complexes / reactions | 192 / 156 / 156 / 154 | same |
| transports / transport transporters / interactions | 28 / 16 / 39 | same |
| compound locations / nucleic acids / states / labels | 288 / 20 / 58 / 62 | same |
| importer messages | 89 | **89** |

No created count changed in any file; no message count changed in any file.

## 4. Dead ends and things that went wrong

- The first PathBank probe (a GROUP BY over membranes joined to visualizations) lost the DB connection after ~20 min;
  replaced by a single-table count (`db4_global.py`, 2 s) and the per-diagram sample.
- Global counts are dominated by propagated copies (Mitochondria Membrane 848,792 rows, mostly open paths in copied
  drug pathways; only 14 of its 600 newest visualizations are distinct). The convention numbers use the de-duplicated
  sample only.
- First transport rule chose the "inner" cargo by side order: for cytoplasm vs an un-outlined extracellular space it
  picked the space and skipped, so CoA's transporters stayed put. Fixed (try both directions, take the outlined
  membrane). The first version also pulled MsbA (5 transports, 4 membranes) onto one membrane: lipid A crossings
  90 -> 130. Fixed (a shared transporter moves only when all its transports cross the same membrane side).
- Putting leaflets first inside their membrane made lipid A worse (152 crossings); reverted before the commit.
- The first A/B run lost all 8 showcase imports: the screenshot list was written with CRLF and read with `$(cat ...)`,
  so every path ended in `\r` ("file not found"). Fixed (LF list) and both arms re-run; only the second run counts.
- I typed a bare `python -` once (forbidden); its task was stopped before it ran anything. Two other `python -`
  processes seen at that moment (parent `C:/t/loc-work`) belong to another lane and were left alone.
- Lock contention: about 2.5 h of the session was waiting for the heavy lock behind three other lanes.

## 5. Open questions for the PO

1. **Lipid A still has more crossings than before** (90 -> 109) because nesting puts the inner-membrane leaflets and
   the cytoplasm in one envelope while MsbA serves transports on four membranes. Options: (a) accept; (b) merge the
   leaflet states into one membrane lane per membrane (strategy #4, needs a display-synonym ruling); (c) draw a shared
   transporter once per membrane (a content change: more protein locations), not done.
2. **Four lipid A leaflet boxes are still empty in PathWhiz**, as before: the importer rejects those transports
   (blank ontology in the empty harness DB). Not a layout issue; expected to import in production.
3. **Wide pathways:** runs wider than 4:1 went 10 -> 12 because of the wider spacing. Wrapping long chains (the snake
   PathBank uses) is the next step; not done here.
4. **Two "cell" boxes for synonyms** (menaquinone: "intracellular" and "cytoplasm" are separate states, so two cells
   are drawn). Merging them for display needs the same synonym ruling as 1(b).
5. Gram-negative cells in PathBank also show a peptidoglycan layer (Peroxisome or Peptidoglycan template) inside the
   periplasm. Not drawn: no T2PW state names it. Draw it whenever a periplasm exists?
6. Drawable cell images (mitochondrion, nucleus, bacterium pictures, common in PathBank) are PWML content
   (`drawable-element-locations`) and would fail content_check; out of this lane's scope.

## 6. Review round 1 (APPROVE-WITH-NITS; fixes F1-F4)

Code commit `23ffd20c` (F1, F2). This docs commit adds this section, F4 and the new evidence.

**F1, ambiguous parent.** `compartment_parents()` picked the "heavier" of two candidates at the same depth. It now
claims **no parent** when more than one candidate shares the top depth (the `weight` argument is gone). Effect on
the replays (geometry only): 8 of 97 runs and 3 of 8 showcases changed (idx 3, 7, 10, 119, 126, 134, 150, 151; lipid A,
chlorophyll, sour taste).
- **Lipid A:** "inner membrane" and the generic "membrane" are both membrane-kind, so the cytoplasm and the four
  leaflets are no longer put inside either: the cytoplasm is its own cell, the leaflets stand alone. Crossings
  109 -> 99 (base 90). **Still there:** the generic "membrane" is drawn as a second Plasma Membrane box inside the
  outer membrane, next to "inner membrane", because its own parent (the periplasm) is the single candidate at its
  depth, so F1 does not touch it. Whether a bare "membrane" state may be nested at all is the leaflet/synonym
  question below; not decided here.
- **Menaquinone:** unchanged ("intracellular" and "cytoplasm" have no children; both stay separate cells).
- **Sour taste:** the synaptic vesicle is no longer put inside "Type III cell" (two cytoplasm-kind states). Crossings
  9 -> 14 (base 9).

**F2, tests** (`tests/test_pwml_layout.py`): `test_no_parent_claimed_when_two_compartments_share_the_top_depth` (two
tissues' cytosols with a mitochondrion; apical and basolateral membranes with a cytosol; a leaflet with two
membranes; plus a relayout check that the three compartments are drawn apart, the mitochondrion's inner line being
the only nested path) and `test_single_parent_chain_matrix_in_mitochondrion_in_cell`.

**F3, templates the A/B never covered.** Replay runs drawing them (after arm, `f3_find.py`): Mitochondria Membrane 84
in 11 runs (idx 11, 39, 44, 54, 74, 77, 88, 100, 109, 118, 134; two lines each), Cell Wall Membrane 304 in idx 114,
**Melanosome Membrane 95 in none** (no replay run, showcase or test fixture names a melanosome). Added to the A/B:
idx 11 and 134 (mitochondria) and 114 (cell wall), with screenshots of 134 and 114.
Missing template id, from PathWhiz's source (`t2pw/latest` `63959982`, read only): the importer only validates that
`visualization_template_id` is present (`membrane_visualization.rb`), so an unknown id imports. The viewer renders a
membrane from the template the server embeds (`path.js.coffee` `parseVisualizationTemplate`; `for_template` calls
`@visualization_template.get('hex_color')`), which is nil for an unknown id, so the browser throws; PathWhiz's SBGN
export (`sbgn_instance`) calls `visualization_template.description` and would raise. Every id T2PW writes (1, 84, 91,
92, 93, 94, 95, 146, 220, 237, 238, 304) is a row of PathBank's `visualization_templates` and of the harness seed.
A probe file (`make_missing_template.py`: the furochromone showcase with membrane template 99999) is in the A/B list
to confirm this empirically. Side effect found while reading: that SBGN export names a compartment after its membrane
template ("Plasma" -> "Intracellular Space", others the template name without "Membrane"); with template 220 every
compartment was "Simple", now they read "Intracellular Space", "Mitochondria", "Nucleus" and so on.

**F4.** `scripts/final2.sh` header now says what the script does (both arms imported fresh).

### Re-gate on `23ffd20c` (clean tree; `logs/final3.log`, one hold, BW_EXIT 0, 0 survivors)

| gate | result |
|---|---|
| content check, 97 replay runs | **0** differences (`cc97_after5.json`) |
| content check, 8 showcases | **0** differences (`cc8_after5.json`) |
| focused (14 `test_pwml_*.py`, final-payload writer fixes, ORCH-768 replay corpus) | 300 passed |
| `suite2.sh` (log records HEAD `23ffd20c`, status clean) | SMOKE 511; golden seam 2 / 8 / 4 / 24; equivalence 8 / 14 / 6; vision 357 + 1 skipped; web 83; all passed |

Layout metrics, base `fd814b8e` -> `23ffd20c` (`sum97_after5.json`, `sumshow_after5.json`):

| | 97 runs | 8 showcases |
|---|---|---|
| edge crossings | 81 -> **75** (71 before F1) | 121 -> **136** (141 before F1): lipid A 90 -> 99, sour taste 9 -> 14, others 22 -> 23 |
| long edges | 38 -> 32 | 15 -> 16 |
| runs wider than 4:1 | 10 -> 12 | 2 -> 2 |
| node overlaps / edges through a node / empty boxes | 0 / 0 / 0 | 0 / 0 / 0 |
| membrane paths | 167 -> 148 | 33 -> 31 |

**PathWhiz A/B on `23ffd20c`: NOT RUN (blocked by memory, not by a failure).** The re-gate was first one hold
(`final3.sh`); with Docker up the machine had 3 GB free, under the queue's 4 GB floor, so the gate never opened.
I stopped Docker Desktop (no container was running; I had started it), split the A/B into `final4.sh`, and armed
`start_docker_then_queue.sh` (waits for an idle lock and >= 7 GB free, starts Docker outside the bounded wrapper,
then queues `final4.sh`). Starting Docker once at 4 GB free took the machine to 0-1 GB free, so I stopped it again.
Free RAM then stayed at 4 GB for 40 minutes with the lock idle (41 GB committed on 31 GB physical), so the waiter was
stopped. The stack `t2pw-pwdraw` is down, volumes kept, Docker Desktop stopped.

What the earlier A/B (`4b701677`, section 3) still covers: F1 changes geometry only (content check 0), and only in
lipid A, chlorophyll, sour taste and V7b idx 7, 10, 126 of the A/B set; the other 8 files are byte-identical to the
ones imported then (created counts equal, messages unchanged). Not covered yet: those 6 files on `23ffd20c`, the F3
files (idx 11, 134, 114) and the missing-template probe.

To run it when memory allows (one command, it waits and queues itself):
`"C:/Program Files/Git/bin/bash.exe" C:/t/lane-pwdraw-work/scripts/start_docker_then_queue.sh`
then `pw_ab.py pw/base2 pw/after5 OUT.json` and `compare_page.py` (point it at `after5`).

Open for the PO (unchanged, not decided here): leaflet / synonym bands (and whether a bare "membrane" state may be
nested), wrapping long chains, grouping compartments by tissue, a peptidoglycan line in the periplasm.
