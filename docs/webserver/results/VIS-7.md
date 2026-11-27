# VIS-7: polish round on PWML diagrams (wrong structures, black loops, art, VIS-5 review notes)

PO goal (2026-11-26): diagrams that look like curated PathBank diagrams in PathWhiz (PathWhiz is the judge, our renderer a
preview). Branch `vis/layout-polish`, worktree `C:/t/vis-pw7`, cut from `vis/layout-iter` `1edbb805` (VIS-5, approved).
Files changed (code): `src/t2pw/pwml/layout.py`, `src/t2pw/pwml/cell_art.py`, `src/t2pw/render/pathbank.py` (one line
rewrapped), `tests/test_pwml_l3_chain_layout.py`, `tests/test_pwml_vis7_polish.py` (new). Nothing under `pipeline/`,
`mapping/`, `rag/`, `curation/`, `llm/prompts/`; `writer.py` and `streamlit_app.py` untouched. Not merged; LEDGER row left to
the orchestrator. Working files: `C:/t/vis-pw7-work/` (replays, renders, PathWhiz runs; compare page
`C:/t/vis-pw7-work/compare.html`). Evidence: `docs/webserver/evidence/VIS-7/`.

## 1. Item 1: wrong structure pictures in PathWhiz (L-serine drawn as a nucleoside, ...): environment, not our ids

Read-only investigation, nothing changed. Chain of evidence (`evidence/VIS-7/item1/pwc_ids.txt`):

1. Our PWML references the right compound: L-serine is `compound id 120`, HMDB0000187, CAS 56-45-1, KEGG C00065, ChEBI
   17115, DrugBank DB00133; pyruvic acid id 108870, ChEBI 15361. The writer leaves every `moldb-*` field (SMILES, InChI,
   InChIKey) empty and writes **no `pwc-id`** (0 of 12 compounds in the enterobactin PWML; PathBank's own PW000779 has one on
   every compound, e.g. `PW_C001420`).
2. PathWhiz's viewer draws a compound's picture from MolDB by that compound's `pwc_id`:
   `http://moldb.wishartlab.com/molecules/<pwc_id>/image.png` (`app/assets/javascripts/backbone/templates/svg_visualization/image.jst.eco`);
   the images are fetched live from the real PathWhiz MolDB, not from the dev DB.
3. The dev stack imports into a TRUNCATED database (harness `pwml_reset!`). `PwmlParser` looks a compound up by `hmdb_id`
   (`find_by(hmdb_id:)`, lib/pwml_parser.rb:644); nothing exists, so it creates it, and `Compound#set_pathwhiz_id` gives it
   `PW_C` + (last row id + 1). PathWhiz's own export of our enterobactin import (r27 checkpoint, f0005): chorismate
   `PW_C000002`, isochorismate `PW_C000003`, enterobactin `PW_C000006`, L-serine `PW_C000007`, pyruvic acid `PW_C000005`.
   The viewer then asks the production MolDB for production's compounds 2, 3, 5, 6, 7: a steroid for pyruvic acid, a
   nucleoside for L-serine. Same ids, same wrong pictures in the r27 screenshot; PathBank's own PW000779 draws correct
   structures in the same stack only because its PWML carries production `pwc-id`s.
4. In production the importer finds existing compounds by HMDB id, which our PWML carries for L-serine, pyruvate, ATP, ...,
   so those get the right pictures. Compounds written without an HMDB id are created new and get a new `pwc_id` with no
   structure (count per PWML in `evidence/VIS-7/item1/no_hmdb.txt`: 1 to 25 of 3 to 32 compounds; lipid A 25 of 32).

Verdict: environment for the wrong pictures (clean dev DB sequence numbers against the live production MolDB). Content
question that remains (open question 1): compounds without HMDB id have no picture source at all. No id changed.

## 2. Item 2: thin black lines looping across the diagram and through boxes

The records: the black lines are **interaction edges**, edge template 195 (Binding / Unspecified; Inhibition is 148, the red
ones). enterobactin has 8 interactions between enzymes and compounds (visualization member edges 164-179): EntB bound to
EntA, EntC, EntE, EntF and to 2,3-DHB, EntE bound to the ent gene cluster. They connect enzymes of consecutive reactions
that stand 400 to 1900 units apart, and every candidate straight or bent curve hit a box or yellow edge, so the router fell
through to its right-angle grid route: 5 to 7 segments, thin rectangular loops hugging box borders, one 2100 long.

Change (`layout.py`, `_arc`, `ARC_HEIGHTS`): for an edge between two nodes with no hub end (an interaction) whose curves are
not clear, before the grid route, try a **flat arch**: one cubic leaving the top (or bottom) sides of both nodes, control
points 60, 110, 170, 250, 350 or 480 units out (capped at 0.4 x the centre distance + 60), the first height clear of every
other box. The pair's second edge reuses the same ends. No new record or field.

Numbers (all 9, `scripts/cmpint.py`, interaction edges only; `ehit.py` for the box test): enterobactin multi-segment
interaction edges 4 -> 0, routed (3+ segment) edges over the 9: 20 -> 18, through-node 0 before and after. Eight pathways
are unchanged except enterobactin (byte-identical geometry elsewhere); on the 97-PWML corpus one PWML (4.pwml) changes
geometry. Honest limit: the arches are still long (enterobactin total interaction length 7422 -> 7673 units) because
the interacting enzymes are far apart in the chain; the loops and rectangles are gone, the long thin arches stay.
PathWhiz drew the new enterobactin with the arches as thin dark curves over the chain, none through a box
(`evidence/VIS-7/pathwhiz/`). Option to shorten them (open question 2).

## 3. Item 3: chromosome and cell art not loading: environment

Our PWML references `drawable-image-id` 92 (Plasmid.svg, 479x481, the chromosome) and 146 (Bacteria.png, 512x512), the same
two rows PathBank's own PW000779 references (lines 16416 / 16429); both rows exist in the harness seed
(`seed_inserts.sql`). The image files are Paperclip attachments (`drawable_image.rb`, path `/:class/:attachment/:id`) that
the dev checkout does not have (`public/system` does not exist). PathBank's own PW000779 imported into the same stack shows
the identical broken-image placeholders at the same two places (`evidence/VIS-7/pathwhiz/f0000_PW000779_pwab.png`).
Nothing to fix in our PWML.

## 4. Review notes from VIS-5

| note | result |
|---|---|
| F1 content_check | the two rules are one-directional and per location id: only base `label-type protein` -> head `subunit`, only base template 2 -> head 12 on a nucleic-acid location are read as equal; the reverse or any other value counts (base writer already writes `subunit` in places, a blanket mask would hide a head change there). `strict` argument drops both rules. Rule-based: 0 differences on 8/8 + item-6 and 97/97; strict: 4 of 8 showcases and 7 of 97 differ, every difference line is one of those two rules (`evidence/VIS-7/metrics/`) |
| F2 subunit labels | unit test `test_two_named_subunits_are_labelled_by_subunit_one_subunit_and_unknown_stay_protein` (2 named subunits -> `subunit`, 1-subunit complex and an Unknown subunit -> `protein`). PathWhiz import of the VIS-7 PWMLs with screenshots: chlorophyll draws each Mg-chelatase subunit by its own name (Protogen IX, PPO1, PPO; PORB complex; protochlorophyllide reductase) in `evidence/VIS-7/pathwhiz/`. Coenzyme A: each subunit draws under its own name, `TcyJKLMN` 0 times (section 6) |
| F3 long path | `_shorten_long_paths` split out of `relayout` (same code), test with a 40-segment path over 255 characters: result <= 255 characters, start and end points unchanged; a short path untouched |
| F4 | `render/pathbank.py:433` mangled `else` + 17 spaces rewrapped as an if/elif chain, same result |
| F5 | `layout.py` misplaced `on_membrane` comment moved onto its own line |
| F6 | chain-order test strict (`>` / `<`), equal x allowed only for the isochorismate -> 2,3-dihydro-2,3-dihydroxybenzoate pair |
| F7 | relayout time per call over the 97 corpus PWMLs: max 0.19 s, total 1.86 s; largest PWML (item 7, chlorophyll, 313 KB): 0.08-0.10 s, base tree 0.08 s (`evidence/VIS-7/logs/relayout_times_*.txt`, `scripts/timing7.py`) |
| F9 / VIS-2 F4 | `cell_art.gram_negative` takes the species names; a species whose name starts with "Mycobacter" never draws the Gram-negative envelope (no peptidoglycan line, no flat envelope); test `test_mycobacterium_outer_membrane_is_not_a_gram_negative_envelope`. No other genus changed |

## 5. Content check

`scripts/content_check.py` (rule-based, one-directional) and `strict`, base = unmodified `f3241270` replays (VIS-5's):
8 showcases + item-6: 0 differences rule-based; 97-PWML corpus (`c1` and `c2`): 97/97 identical rule-based; strict diff
7/97 = only label-type / nucleic-acid template lines of VIS-5 items 11 and 13. VIS-7 against VIS-5 r28: strict 0
differences (97/97), geometry differs only in enterobactin (and corpus 4.pwml).

## 6. PathWhiz A/B

Queue, heavy lock, harness VIS-2 / project `t2pw-pwdraw` (`scripts/pwab.sh`, `pwab3.sh`); stack down afterwards each time,
BW_EXIT=0, FINAL SURVIVING COUNT 0. Checkpoint `a3` (code `1e038639`): 10/10 import success (9 showcases + PathBank
PW000779), **import errors identical to VIS-5 r27 for all 10 files** (success flag, error lines and created record
counts; `evidence/VIS-7/pathwhiz/ab_vs_vis5_r27.txt`: 0 new error lines; error counts 3 / 1 / 15 / 0 / 1 / 0 / 0 / 22 / 0
/ 7). Screenshots: 7 of 10 saved (enterobactin, chlorophyll, ceramide, furochromone, lipid A, menaquinone, PW000779); the
viewer timed out on coenzyme A, ent-item6 and sour taste under machine load (as in VIS-5), so they were re-imported
alone (`a3b`: 3/3 success, same errors as r27, 3 screenshots). `compare.html` (`C:/t/vis-pw7-work/`) shows per pathway
VIS-5 r28 preview, VIS-7 preview, VIS-7 in PathWhiz, PathBank reference.

F2 check in PathWhiz: coenzyme A's TcyJKLMN complex draws each subunit with its own protein name (SVG text: TcyL, TcyM,
"protein TcyJ", "protein TcyK", "protein TcyN"; the complex name "TcyJKLMN" appears 0 times; VIS-5 section 3 item 13 records it printed five
times). Chlorophyll's Mg-chelatase subunits draw as Protogen IX / PPO1 / PPO, PORB complex, protochlorophyllide
reductase. The PathBank-style `subunit` labels import and draw correctly.

## 6b. Gates (queue, heavy lock, clean tree, HEAD `6f497433` = code `1e038639` + docs, 0 survivors each)

| gate | result |
|---|---|
| `vis7-focused-final`: `tests/test_pwml_*.py` | 285 passed |
| `tests/test_render_*.py` | 49 passed |
| `tests/test_final_payload_writer_fixes.py` | 22 passed |
| `tests/test_orch768_replay_corpus.py` | 9 passed |
| wrapper | BW_EXIT=0, FINAL SURVIVING COUNT 0 |
| `vis7-suite2`: SMOKE | 511 passed |
| golden seams batch / c011 / stage8 / quarantine (unedited) | 2 / 8 / 4 / 24 passed |
| equivalence run_pathway / app vs service / default path | 8 / 14 / 6 passed |
| vision | 370 passed, 1 skipped |
| web | 83 passed |
| wrapper | BW_EXIT=0, FINAL SURVIVING COUNT 0 |
| `vis7-corpus-c2`: 97-PWML replay corpus through this tree (relayout timed) | BW_EXIT=0, 97/97 PWMLs, content check 97/97 identical, 0 survivors |

Logs in `evidence/VIS-7/gates/`. The first corpus run (`c1`, same code) also passed; its wrapper log was truncated by the
slip in section 7, hence the re-run `c2`.

## 7. Dead ends and slips

- First arch version ran after the bend and via stages; it left the loops (bends won). Moved before them and capped.
- The first F1 version masked `subunit` blindly on the head side: 7 false differences because the base writer already writes
  `subunit` in places; made it per location id.
- Rule slips: (a) the quick loop (8 replays + 9 renders, about 30 s) ran through `bounded_run.py` without the heavy lock, as
  VIS-5's did; the corpus replay, PathWhiz runs and gates went through the queue. (b) A typo started a second copy of the
  corpus + PathWhiz chain with a trailing `&`; I killed its process tree by PID (none of it had taken the lock; the other
  session's lock holder was untouched) and it truncated one log, so the corpus gate was re-run.

## 8. Open questions (product owner)

1. **Compounds without an HMDB id have no structure in production** (1-25 per PWML, lipid A 25 of 32): PathWhiz creates
   them with a fresh `pwc_id`, MolDB knows nothing. Options: (a) accept, they draw as empty boxes; (b) look up a `pwc-id`
   for them from PathBank by name / KEGG / ChEBI at export (content: writes a new field, needs a content-check rule and a
   mapping owner; medium); (c) fill `moldb-smiles` / InChI from ChEBI so MolDB can draw them (VIS-5 G14, medium plus
   network at export).
2. **Interaction arches are still long.** The enzymes of consecutive reactions interact with the "hub" enzyme (EntB), so
   the lines span the chain. Option: draw an interaction enzyme-to-enzyme as a short link by placing interacting enzymes
   on one side of the chain (layout only, small), or hide Binding / Unspecified interaction edges in the picture (they stay
   in the PWML; a visualization choice, small, but information drops from the picture).
3. **Single-transporter pathways** (lipid A MsbA with 7 transports; sour taste OTOP1 with 6) keep long green lines across
   the diagram: content shape. Option: one transporter location per transport, as PathBank does (new location records,
   structural content change, medium; needs a content-check rule and an A/B).
4. Currency compounds per reaction, enzyme cofactor tags, protein templates by subunit count (VIS-5 items 7.1-7.3), still open.
