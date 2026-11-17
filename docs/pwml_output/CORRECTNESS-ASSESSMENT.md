# Correctness assessment: PWML, PathWhiz import, SBML, SBGN, web diagram, ground truth

PO ruling D-W23 g (2026-11-14). Phase A (read-only assessment) record and the ranked FIX LIST.

- Code assessed: `pwml/integration-check` `bcc67409` (final payload phase 5 + PWML layout v2.1 + reaction side fix), worktree `C:/t/integ`.
- Baseline: all 152 stored runs replayed (final payload on, `T2PW_PWML_LAYOUT=v2`). 97 PWML files, **97/97 byte-identical** to the earlier `C:/t/integ-reports/replay/on_v2` replay (checked by the sub-orchestrator with `cmp`).
- Sample: 26 pathways, 20 organisms, 0 to 23 reactions, plus the 3 worst files by layout metrics (idx 12 most crossings, 150 most long edges, 68 most unattached nodes). List: `evidence/correctness/sample.json`.
- Assessors: A0 baseline, A1 PWML in PathWhiz, A2 web diagram, A3 SBML/SBGN, A4 ground truth. Dispatch table at the end.
- Staging area with the full evidence (scripts, all screenshots, PathWhiz exports): `C:/t/corr-assess/`. The small committed subset is under `docs/pwml_output/evidence/correctness/`.

**Context for the extraction figures (draws).** None of the 192 runs in `C:/t/ev-cm/runlist.json`, including all 26 sampled, has a `stage1_ensemble_report.json`; N-draw ensembling (`stage_one_ensemble.py`, ORCH-750, off by default) landed on 2026-10-19 and most sampled runs predate it (Aug-Sep 2026; the Oct-22 runs also show no ensemble report). So every assessed run is a **single-draw** extraction, not the production `T2PW_EXTRACTION_DRAWS=3`. The "5 of 26 fully correct" figure describes 1-draw output; the Phase B A/B on gold sets uses 3 draws.

**PO ruling D-W24 (2026-11-14) answers section 3.** Biology changes are authorized on `pwml/correctness` only; results in `CORRECTNESS-FIXES.md`.

Severity: **S1** content wrong or lost; **S2** misleading picture; **S3** cosmetic.
Layer: extraction (the frozen payload itself is wrong) / IR / PWML writer / PWML layout / PathWhiz importer or exporter (not our code) / clean-model SBML / web prototype.

## 1. Verdict per layer

| Layer | Verdict | Basis |
|---|---|---|
| Extraction (final payload vs paper) | **Partly correct.** Core chains are mostly right; 5 of 26 fully correct, the rest have wrong enzymes, deleted or merged steps, invented steps or duplicated chains. | A4, 26 pathways read against the paper, 5 with gold (9, 5, 78, 58: gold_v2; 68: pinned_v1), 8 public references |
| Final payload → IR | Mostly faithful. Two exporter-side distortions: activator/inhibitor modifiers become enzymes (3 reactions); reactions with no compartment go into the first biological state (51 reactions, 12 pathways). | A1, A3, A4 |
| PWML writer | Reactions, sides and enzymes match the IR exactly (195/195, 0 side contradictions). Losses: 41 of 54 interactions withheld by design (D-104); 1 frozen reaction quarantined (idx 116); species id collision (idx 134); shared compounds redrawn per reaction pair (chain fragments, 20/26). | A1, A3 |
| PWML layout (v2.1) | 0 overlaps / 0 out of canvas (97 files). Flow order: 31 of 127 producer→consumer pairs run against reading order (13 pathways). Long compartment labels clipped (14). Fit-to-window scale < 0.3 in 5. | A0 metrics, A1 screenshots |
| PathWhiz import | Accepts 26/26. Merges exact duplicate reactions (195 → 176). Refuses 7 of 30 transports (same-state ends, duplicate transporter, 1 harness-only). Drops 7 nucleic acids without a type (idx 7). | A1 DB snapshots |
| SBML (ours, clean model) | **Most faithful format.** 0 libSBML errors in 26/26 (warnings: no initial amount, by design; units). Keeps modifier roles, interactions and the idx 116 reaction. No layout, so no standard renderer can draw it. | A3 |
| SBML (PathWhiz export) | 0 errors; structurally consistent with the PathWhiz DB. Writes every transport X→X in one compartment, `initialAmount=1` on all species, `&rarr;` in names, 11 invalid sboTerms. PathWhiz-side, not our code. | A3 |
| SBGN-ML (PathWhiz export) | 25/26 exported (idx 58: HTTP 500, our Fe3+ glyph template 299 unknown to its SBGN builder). Every file fails the SBGN-ML 0.3 schema (map id) and schematron (arcs to process glyph not port, logic arcs). 44% of reactions show an enzyme "Unknown". Mostly PathWhiz-side. | A3 |
| Web diagram (prototype) | **Faithful to the IR**: 0 reactions, sides, enzymes, compartments or transports wrong across 26. Crashes on idx 128. Clarity problems are real and fixable (see W-items). Inherits the IR's modifier-as-enzyme and duplicate-chain issues. | A2 |

## 2. FIX LIST (ranked)

`content?` = would the fix change pathway content or biology. **yes** items are PO questions (section 3), not fixed.

| # | Sev | Layer | Finding | Evidence | Proposed fix | content? | Regression risk |
|---|---|---|---|---|---|---|---|
| F1 | S1 | web prototype | Crash on idx 128 (gallery #02): transport `tr_2` lists TcyJKLMN twice; pill id `("pill", junction, type, key)` collides, second copy overwrites the first in `m.nodes`, its catalysis edge stays, `prep_edges()` fails (`t2pw_diagrams.py:610`, `:714`). | A2 `a2/per_pathway.json` | Skip a pill whose id is already present (one line). Measured in A2's copy: other 25 SVGs byte-identical. | no | very low |
| F2 | S1 | PWML writer | Species id collision: idx 134 writes Homo sapiens and Mus musculus both as species id 1 (`writer.py` ~1152-1166, `id_for`); PathWhiz stores the pathway as Mus musculus. | `134.pwml` lines 36-50 (verified) | Never reuse an id across distinct species records; find why Mus musculus resolves to 1. | no (fixes an id, not content) | low; check all 97 for species blocks |
| F3 | S2 | PWML writer | Shared compounds redrawn: a location is reused only between adjacent reactions i and i+1 (`writer.py` ~2255-2296). 509 locations for 360 distinct (compound, state); 103 drawn components vs 35 logical; 20/26 pathways. Chains fragment (idx 7: 15 drawn components vs 3). | A1 `a1/per_pathway.json` | Under layout v2 only, reuse one location per (compound, state) for the whole chain where the layout can route it; v1 byte-identical. | no | medium: layout geometry changes in most files; must keep 0 overlaps and harness pass rate |
| F4 | S2 | PWML layout | Flow order: 31/127 producer→consumer pairs against reading order (13 pathways); chains wrap bottom-up (135, 19, 142); idx 9 chorismate→isochorismate far right in one box. | A1 | Order reaction columns by a topological sort of the chain (v2 only). Likely done together with F3. | no | medium |
| F5 | S2 | web prototype | Regulation/binding dominates where present (idx 7: 27% of edges, 38 molecules exist only for them; the "dashed mat" in s1.png is mostly withheld binding links). | A2 mock `7_reg_after.png` | Hide regulation by default with a visible count and a toggle (I1). | no (display only; nothing removed from data) | low |
| F6 | S2 | web prototype | Enzymes far from their reaction: 59 enzyme links > 200 px (max 1347). | A2 | One enzyme drawing per catalysed reaction (I2): 59 → 9. | no | low |
| F7 | S2 | web prototype | Exact duplicate steps inflate pictures (idx 12 one glycosylation ×4). | A2 | Draw identical steps once with a "×N in data" badge (I3); data untouched. | no (display) | low |
| F8 | S3 | web prototype | Cryptic "≈R17" tags (52 in 12 pathways), withheld interactions drawn paler with no legend, 10 directional types drawn undirected, spine style hides compartment crossings for 13 transports. | A2 | Plain words ("same as R17"), legend strip (wrap on narrow canvases), arrowheads for directional types, compartment-crossing marks in spine (I4, I9). Lane order following the flow (I7). | no | low |
| F9 | S2 | PWML writer | idx 58: PathWhiz SBGN export fails (HTTP 500) because Fe3+ uses visualization template 299 (`compound_templates.py:44`), unknown to PathWhiz's SBGN builder. Also in 105, 35, 50, 84. | A1 `pathwhiz_exports/58.sbgn.FAILED_500.html` | Use a template id the PathWhiz SBGN builder knows for ions (check `smpdb` read-only). | no (glyph choice) | low |
| F10 | S3 | PWML layout | Long compartment labels clipped (14 pathways); fit-to-window scale < 0.3 in 5 pathways (labels ~3.6 px; legible at the viewer's opening view). | A1 `a1/shots/7.png`, `7_view.png` | Wrap or shorten box labels; tighten canvas. | no | low |
| F11 | S2 | PWML writer (data-shaped) | Exact duplicate reactions (same state, same left/right, same enzymes) are written separately; PathWhiz merges them anyway (195 → 176). | A1 `a1/dup_ids.json`, A2 `a2/duplicate_reactions.json` | Write an exact duplicate once (zero-loss merge). Only if proven zero loss on all 97 replays; near-duplicates and isozymes untouched. | no if strictly exact | medium: proof required on 97 |
| F12 | S1 | IR (ir.py ~2174-2178) | Modifiers with role activator/inhibitor are turned into enzymes. idx 7: GluTR, FC1, GUN4 drawn as enzymes; idx 9: MenD (inhibitor) as co-catalyst of MenB. Our SBML keeps the roles. For idx 9 the payload is also wrong (paper: "allosteric inhibition of MenD" by DHNA). | A2 `a2/modifiers_as_enzymes.json`, A3, A4 | **PO question Q1.** | yes | — |
| F13 | S1 | IR (ir.py ~1836) | Reactions with no stated compartment go into the first biological state: 51 reactions, 12 pathways; wrong per the paper in 138 (whole pathway "extracellular"), 134 (mitochondrion; Reactome R-HSA-71262), 116 and 144 (ER), 128. | A1 `a1/null_state_default.json`, A4 | **PO question Q2.** | yes (placement) | — |
| F14 | S1 | export quarantine | idx 116: sphinganine → dihydroceramide (ceramide synthase) is in the frozen payload, removed at export (`reaction_enzyme_must_be_protein_complex`); contract §5 forbids removing frozen content. An enzyme-less sphinganine → ceramide shortcut (itself wrong) survives. | A1, A3, A4 (verified in payload) | **PO question Q3.** | yes | — |
| F15 | S1 | PWML writer (policy D-104) | 41 of 54 stated interactions withheld from PWML (bare-protein endpoints, unsigned types); `inhibits` (58, 78) and `activates` (128) vanish from PWML and PathWhiz. Web diagram and our SBML keep all 54. | A1, A3 | **PO question Q4.** | yes | — |
| F16 | S1 | writer / data | Transports with no stated from/to written with both ends in the same state; PathWhiz refuses them (78, 138, 10, 150) or a duplicate transporter (128). 7/30 transports never reach PathWhiz. | A1, A3 | **PO question Q5.** | yes | — |
| F17 | S2 | extraction | Duplicate chains in near-synonym compartments ("cell" vs "intracellular" vs "L. lactis KF147"; idx 9 draws 9 of 19 steps twice, idx 5 triplicated, 134); two boxes both "plastid" (7: `plastid` and `plastid state`), 116, 144; organism-named states (9, 14, 144); `__auto_state__` placeholders in 12. | A1, A2, A4 | **PO question Q6.** | yes | — |
| F18 | S1 | extraction | Paper-stated reactions deleted when their enzyme has no database identity (MenD in 5, HS41.20/HS41.17 in 121 — both end products in the title missing, LpxL/LpxM in 68 — 9-step pathway exported as 3). Elsewhere the same case ships with the "Unknown" placeholder. | A4 `a4/per_pathway.json` | **PO question Q7.** | yes | — |
| F19 | S1 | extraction | Wrong enzymes: 134 (TMLD and BBD on both dioxygenase steps; mouse-only Tha1 in a human pathway), 7 (Chl synthase as CHLI; FLU, a negative regulator, as enzyme), 126 (3 reactions mixed), 120, 129, 128, 78, 58; 116 "ceramide synthase" grounded to TLCD3B not CERS1-6, "S1P phosphatase" to HM13. | A4 | **PO question Q8.** | yes | — |
| F20 | S1 | extraction | Merged steps with wrong products (68, 116, 138 PurA/PurB, 19 DmaW+EasF, 148, 114) and invented/non-evidence reactions reaching export (126 benzaldehyde → salicylic acid, evidence = pathway name; 144 evidence = article header; 7 singlet-oxygen reaction; 78 LDH assay reaction). | A4 | **PO question Q9.** | yes | — |
| F21 | S2 | extraction / policy | 88 of 162-170 complexes are "Unknown"-only; the Unknown protein carries species Arabidopsis thaliana in 20/26 payloads, including human ones; in 134 all six human enzymes failed `species_mismatch`. Shows in SBGN as enzyme "Unknown" on 44% of reactions. Contract §13 reserves this policy. | A1, A3, A4 `a4/unknown_complexes.json` | **PO question Q10.** | yes | — |
| F22 | S2 | schema | All reactions written one-way; the payload has no reversibility or stoichiometry field; 8 reactions in 5 pathways cite reversibility in their evidence. Not an exporter fault. | A3, A4 | **PO question Q11.** | yes | — |
| F23 | S2 | PathWhiz (not ours) | PathWhiz SBML: transports X→X, `initialAmount=1`, `&rarr;` in names, invalid sboTerms; SBGN: schema/schematron failures, unlabeled process glyphs, all compartments "Simple"; names replaced by UniProt long names; 7 nucleic acids without a type rejected (idx 7). | A3 `a3/validation.json` | Record only; PathWhiz source is read-only for us. The nucleic-acid type gap is ours (extraction gives no type) and is part of Q6-style data quality. | — | — |
| F24 | S3 | clean-model SBML | Neither SBML carries a layout, so momapy and similar cannot draw them. | A3 | Optional later: SBML Layout package from the PWML layout. Not in this programme. | no | — |

## 3. Open questions for the product owner (biology or pathway content; not fixed)

| Q | Finding | Options (cost) |
|---|---|---|
| Q1 | F12 activators/inhibitors written as enzymes | (a) write them as PWML interactions (activates/inhibits) instead of enzymes; bare proteins then hit D-104 and are withheld unless wrapped (small writer change, content moves from enzyme to interaction, some disappears from PWML); (b) keep as enzyme but mark in the description (tiny, still misleading); (c) status quo. |
| Q2 | F13 no-compartment reactions placed in the first state | (a) place them in the organism's cytosol/"cell" state when one exists, else a neutral unnamed state (small; changes placement in 12 pathways); (b) leave them unassigned and let PathWhiz default (importer behaviour unknown); (c) status quo. |
| Q3 | F14 export quarantine removes a frozen reaction (116) | (a) keep the reaction without its non-complex enzyme (contract §5 compliant; enzyme lost); (b) allow the exporter to wrap a single protein in a complex (violates merge rule 8, needs a D-xxx); (c) status quo. |
| Q4 | F15 41/54 interactions withheld (D-104) | (a) keep D-104 but list withheld interactions in the PWML description; (b) wrap bare-protein endpoints in single-protein complexes for signed types only (inhibits/activates); (c) status quo. |
| Q5 | F16 transports with no stated from/to | (a) drop the transport but keep its cargo (loses a stated transport); (b) infer from/to from neighbouring reactions, marked inferred (biology inference); (c) status quo (PathWhiz refuses them). |
| Q6 | F17 near-synonym compartments and duplicated chains | (a) treat "cell", "intracellular", "cytoplasm" and organism-named cell states as one location in bacteria and merge identical chains (removes 9-19 duplicate reactions per affected pathway; needs zero-loss proof on 97); (b) merge only exact-name variants ("plastid" / "plastid state"); (c) status quo. |
| Q7 | F18 reactions deleted when the enzyme has no DB identity | (a) keep with the "Unknown" placeholder (as other runs already do); (b) keep with no enzyme, `review_required`; (c) status quo (delete). |
| Q8 | F19 wrong enzymes | (a) pre-freeze enzyme-to-step evidence check that drops enzymes the evidence does not tie to the step; (b) flag only; (c) prompt change at extraction. |
| Q9 | F20 merged and invented steps | (a) evidence gate: drop reactions whose evidence names none of their participants; (b) flag only; (c) re-extract missing intermediates. |
| Q10 | F21 "Unknown" placeholder complexes with Arabidopsis species | (a) give the placeholder no species / the pathway's species; (b) investigate the `species_mismatch` grounding failures (134); (c) no change (contract §13). |
| Q11 | F22 no reversibility/stoichiometry in the schema | (a) add both fields; (b) reversibility only; (c) status quo. |

## 4. Phase B plan (no biology; starts when `C:/t/release_landed.txt` exists)

Order: F1 + F5-F8 (web prototype, `C:/t/ev-diagram-v2/`, re-render all 97 to `C:/t/render-corr/web/`), F2 (species id), F9 (SBGN template), F3 + F4 (writer/layout, v2 only), F10, F11 (only with 97-replay zero-loss proof). Each fix: failing-first test, v1/flag-off byte-identical, no content loss on 97 replays, PathWhiz harness not worse, SMOKE 511, golden 38, both equivalence files alone, focused tests, bounded wrapper 0 survivors, opus/high review.

## 5. Dispatch table (Phase A)

| Id | Role | Agent type | Model / effort | Result |
|---|---|---|---|---|
| A0 | baseline replay + sample | general-purpose | sonnet / medium | 152 runs, 97 PWML byte-identical; sample of 26 |
| A1 | PWML in PathWhiz | general-purpose | sonnet / high | 26/26 imported + screenshots; SBML 26/26, SBGN 25/26, DB 26/26 |
| A2 | web diagram | general-purpose | opus / high | 26 fresh IR (PWML from them byte-identical), 26 renders, crash root cause, 4 mocked ideas |
| A3 | SBML / SBGN | general-purpose | sonnet / high | 26 ours-SBML, validation, 77 renders, 7-way comparison |
| A4 | ground truth | general-purpose | opus / high | 26 pathways vs paper, 5 gold, 8 public references |

Process notes: A1 and A2 could not write their `FINDINGS.md` (report-file writes are blocked for subagents here); their findings are folded into this record from their reports and their `per_pathway.json`. A1 and A0 each ran one empty `python -` heredoc by mistake (no effect; stopped). A2 rendered without the heavy lock (light jobs, through the wrapper). All jobs: 0 survivors, lock released.
