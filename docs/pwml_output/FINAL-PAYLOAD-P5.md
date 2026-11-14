# FINAL PAYLOAD, PHASE 5 (2026-11-13 to 2026-11-14, branch `pwml/final-payload`, from `3e8a66f3`)

Evidence: `evidence/final_payload_p5/` (RULES5.txt, writerA/, ontoBE/, nucleicD/, pwpatch2/, c_trace/, c_live/, fix1/, fix2/, fix3/, liveG/, measure/, flip/, final_tests/, review1/ to review3/). PathWhiz changes: `pathwhiz_changes/`. Proposed decisions P-44 to P-54 in `DECISIONS-PROPOSED.md`. Answers `docs/webserver/DECISIONS.md` D-W21.

## 0. NOT ESTABLISHED (read first)
1. **Production PathWhiz.** Every harness number is from an EMPTY database plus a vocabulary seed. Nothing was imported into production. Exact state matching (branch 04) is untested against a populated database.
2. **The default flip rests on one live run per mode on one paper** (PMC12782028, "cholesterol biosynthesis"). Name-only mode failed once in stage 1 (`no_reactions`, before any final-payload code) and passed on a retry that picked an unrelated paper. Research mode writes a report and no PWML, so PWML counters could not be checked there. No test runs the whole default path (`run_pathway` plus export, or the Streamlit app) with the flag on: the test suite is pinned to `T2PW_FINAL_PAYLOAD=0` centrally, and the default-on seam tests stop at `prefreeze_with_final_fill` (review 3 R3-2, fix 3).
3. **`T2PW_TRANSPORT_SIDES` (task C) is NOT enabled and not recommended yet** (section 4). The 4 equal-side transports are fixed on replay only.
4. **The disclosure record is incomplete (review 3 R3-4):** dropped components and renamed membranes are counted in the writer, but `BuildResult` does not carry the counters or `vocabulary_outcomes` into any file, so a user of the PWML or text version cannot see what was dropped. D-W19c (disclose) and D-W21b (record why) are met only in the evidence docs.
5. **Network behaviour with the flag on by default is bounded by reasoning and fakes, not measured.** Budgets are capped to the job deadline minus 60 s, a circuit opens after 5 consecutive FetchErrors, one in-flight request can still outlast the margin when a server sends `Retry-After: 60` (R4-1, documented, not fixed). With no job deadline in state (Streamlit) the worst case is 300 s plus 60 s plus one request.
6. Variance: each column of the harness table is a single replay and a single import; the ontology cache is shared and mutable.
7. Two review-3 judgements remain proposals, not PO rulings: nucleic type for `conflicting` rows (not looked up) and protein-coding gene to DNA (only with positive paper evidence). The reviewed synonym `type iii taste cell` is outside the exact-only rule.
8. The first SMOKE run at the final head had 3 failures with exit 3221225794 (0xC0000142, a process-start failure of `git ls-files` while the machine was busy, as in P4 section 0.8); see section 5 for the rerun.
9. Process slips: one agent hung on an `rm -rf` permission prompt (task C driver, replaced by a second agent that wrote the result from the stored runs); two agents returned early with "waiting" reports and were resumed; the machine restarted during the live-run phase (the three runs and the import were already complete; the import was redone by the orchestrator); a second session shared the heavy lock and held it for long stretches (hours of waiting in total).

## 1. What was done, per ruling (D-W21)
| ruling | result |
|---|---|
| (a) wildcard guard | `writer.py` `_apply_wildcard_guard` (0c320e27), PWML only. 152-run replay: species-only states 0; wildcard-bindable states 63 in 20 files against 59 in 19 with the flag off and 65 in 20 in P4 d3 (most already existed in the stored states). Counters on the final replay: `components_kept_wildcard_guard` 21 (13 files), `cell_types_dropped_no_id` 13 (6 files), `tissues_dropped_no_id` 1, `locations_dropped_no_id` 4 |
| importer wildcard | root-cause fix on local branch `t2pw/04-biological-state-exact-match` (80bdaca7): harness (39 files) messages 107 to 96, biological states 903 to 939, transports 40 to 43, no new message, 15 files with no bindable state identical, 4 controls unchanged apart from the number of states created. T2PW keeps its own guard because real PathWhiz will not have the patch |
| (b) cell types and tissues | stay in the final payload; dropped from the PWML only (with the guard). Why each has no id: `ontoBE/why_no_id.md` (35 names, each with the query behind it): cell lines are not CL terms (MG-63 = CVCL_0426, Saos-2 = CVCL_0548 found through Cellosaurus; PathWhiz `cell_type.rb` accepts only `CL:` plus 7 digits, so they cannot be written); erythroid / erythroid cells: only progenitor or lineage terms; geniculate ganglion neuron: only subtype terms; nerve fiber: only a related synonym; leaflets of the IM: no exact term and never mapped to the whole membrane; microsomal fraction: a lab fraction, not a compartment; saliva, stroma, apical, young leaf: no exact term. Newly resolved: `synaptic vesicles` (singular form), `granulosa cells` (PathWhiz row), numbered cell types through the paper context (`Type II cell` only with the paper text; `Type III cell(s)` through the reviewed synonym). Ontology lookup 13 of 30 (P4) to 14 of 30 on the same input, 16 with a stored run's context |
| (c) 4 equal-side transports | section 4 |
| (d) DNA/RNA | `nucleotide_type_status: unknown` plus reason on the row; online lookup NCBI Gene (exact symbol or alias, one species, live record) with RNAcentral as confirmation. 152 runs, 39 nucleic acid rows: 4 settled by name, 10 resolved from the paper (DNA), 1 by lookup (miR171c, RNA), 24 unknown (13 no symbol-shaped name, 7 not attempted (conflicting), 3 not found, 1 protein context: CHL27). The first version resolved 4 online (RyhB RNA, miR171c RNA, LHCB1.1 DNA, CHL27 DNA); review 1 judged LHCB1.1 (paper also says mRNA) and CHL27 (paper uses it for a protein) wrong silent facts, so conflicting rows are not looked up and protein-coding needs positive paper evidence. Cost: RyhB (a correct RNA) is lost with its row |
| (e) inner and outer membrane | never merged: separate PathWhiz rows `Inner Membrane` / `Outer Membrane` (they are the bacterial membranes: 1271 of 1285 and 1232 of 1243 production states are prokaryote; their stored GO ids are other terms, so they bind by name) used only when every state using the location is bacterial (28 occurrences in the 152 runs, E. coli and Pseudomonas); human or undetermined occurrences are not written under the row name; a membrane the guard must keep is written `<name> (unresolved, no ontology id)` (R2-1: no biological claim) |
| (f) PathWhiz edits as separate PRs | four stacked local branches (section 6), not pushed |
| (g) internal rendering of the paper pathways | NOT done in this phase (not in the task brief; the layout work runs in `C:/t/lay`) |

## 2. Live runs and the default flip (D-W19a)
`T2PW_FINAL_PAYLOAD=1`, curator live, PMC12782028 (PMID 41522513), code `34cb102a` (code unchanged since fix round 1): `liveG/RESULT.md`.
| mode | status | wall | PWML | fill provenance | title and reference | PathWhiz import |
|---|---|---:|---|---|---|---|
| PathWhiz (strict) | pass | 1424 s | 32,791 bytes, 3 reactions | 3 entries (2 cellosaurus ids, 1 DrugBank) | NCBI title, PMID 41522513 | success, 0 messages |
| research | pass (report) | 2678 s | none by design | 6 entries (2 cellosaurus, DrugBank, ChEBI, KEGG, CAS) | in the payload | n/a |
| name only | first run FAIL `no_reactions` (stage 1, 337 s); retry pass | 864 s | 8,343 bytes, 1 state (an unrelated paper, PMID 42997905) | 1 entry (tissue id) | NCBI title and PMID | success, 0 messages |
**Flip: done** (5d1ac2ce). `final_payload_enabled()` returns off only for `0`, `false`, `no`, `off` (stripped, any case); unset or empty means on; `T2PW_FINAL_PAYLOAD=0` is the explicit off switch. Tests: one autouse pin in `tests/conftest.py` sets `=0` for every test, 20 `delenv("T2PW_FINAL_PAYLOAD")` sites became `setenv(..., "0")` (env pins only), one assertion changed (`test_flag_parsing`, which tests the flag semantics themselves; reviewer judged it legitimate), new `test_final_payload_default_flip.py` and `test_final_payload_default_seam.py` (dead network, fast empty answers, spent deadline). The decision to flip rests on: live runs passed in every mode that can pass; flag-off output byte-identical (97 of 97, section 5); the final-payload layer fails open (payload restored) at every step; a dead network now costs seconds (fix 3). It does not rest on a statistical no-loss proof (section 0.2). **To go back: set `T2PW_FINAL_PAYLOAD=0` in the environment** (or `T2PW_FINAL_PAYLOAD_BUDGET_S=0` to keep the payload layer but make no network calls).

## 3. Harness re-measure (59 flag-on files, same category definitions as P4 section 6)
| | P4 d3 | P5 (a) 03 tip + owner change | P5 (b) 04 tip + owner change |
|---|---:|---:|---:|
| clean files | 45 | 48 | 48 |
| messages | 94 | 67 | 63 |
| vocabulary | 38 | 18 | 18 |
| transport family | 18 | 18 | 14 |
| left=right | 5 | 5 | 4 |
| nucleic acid | 24 | 20 | 20 |
| interactions | 9 | 6 | 6 |
| other | 5 | 5 | 5 |
| transports created of 37 | 32 | 32 | 33 |
Controls identical to P4 (PW741182 0 messages, PW012926 1, PW176888 2, PW687775 3, PW1 stub 176), 5 controls with both importers. No file got worse in (a) or (b): none has more messages, a new message type or text. Improved: idx 10, 23, 37, 98, 151 lose their id-less cell-type pairs (those cell types are now dropped from the PWML); idx 7 33 to 26 messages (nucleotide types); idx 151 is the only file that differs between (a) and (b) (transports 2/3 vs 3/3, messages 6 vs 2). 22 of the 97 flag-on files differ byte-wise from P4 d3; not traced file by file (the writerA run attributed 19 compound-name differences to replay environment, unproven). Source: `measure/RESULT.md`, `measure/import_per_file.md`.

## 4. Task C: the 4 equal-side transports (flag `T2PW_TRANSPORT_SIDES`, default off, commit e86cc420)
Trace (`c_trace/TRACE.md`): TubI export: Stage 1 gave no states, evidence says only "out of the cell"; ferric-enterobactin import and enterobactin secretion 08-27: transport has no states while Stage 2 later created them; enterobactin secretion 09-02: the model wrote the cytoplasmic state on both sides. Stage 2 cannot repair a transport (it only appends rows). All four are single-draw runs from before D-104. Chosen fix: one re-ask per deficient transport after Stage 1, quote must be in the paper text; no prompt file changed (pinned byte-for-byte); zero deficient transports = zero calls and a byte-identical payload.
Live (`c_live/RESULT.md`): all 4 stored cases are repaired on replay (one call each). Fresh draws reproduced the defect only for TubI (3 of 5 draws; the step repaired 1 of 2 defective draws, the other got an empty `{}` reply); 12 fresh draws of PMC12096016 (14 transports) had no defective transport (probability about 0.11 at the stored rate). 18 of 19 on-runs byte-identical at Stage 1, the one that differs changed only the `to` side plus one new state. Controls (3 papers, 6 on-draws): zero repair calls, byte-identical Stage 1. Two replay repairs wrote a state the quote does not support ("out of the cell" gave intracellular, "imported via TonB-dependent uptake" gave cytoplasmic). **Not enabled**: no-loss cannot be shown at this sample size and review 2 (R2-5 to R2-8) found that a new state can point at an undeclared location (export could fail), that the quote check does not prove support, and that equal sides can both be replaced. The flag stays in the code, off.

## 5. Tests and reviews
Clean worktree `C:/t/fpclean` at `8abdbae0` (code head `41385706`; no `.env`, no dirty caches), through the bounded wrapper, 0 surviving processes in every run (`final_tests/`): **SMOKE 511 passed** (second run `smoke2`; the first run `smoke` had 3 failures, all `git ls-files` exit 3221225794 = 0xC0000142 while the machine was busy, the known load artefact of P4 section 0.8); **golden seam 38 passed** (the four files unedited); **focused 407 passed** (`test_final_payload_*`, `test_pwml_writer*`, `test_pwml_ir*`, `test_process_normalizer*`, `test_c121_autostate_lifecycle`, `test_transport_sides`). Flag-off identity at `28278264` (clean export, `=0` explicit): **97 of 97 PWML sha256-identical** to the P4 d3 and P3 flag-off outputs, the same 55 runs write none; fix round 3 changed only flag-on code (review 4 verified the flag-off path returns before any of it) and the identity replay was not repeated after it. In the shared worktree `C:/t/fp` the test `test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance` fails because `.env` makes a database reachable (passes in the clean worktree); five other failures seen there came from other agents' uncommitted files.
| review | model / effort | scope | verdict |
|---|---|---|---|
| 1 | opus / high | `3e8a66f3..cb54ea35` (A, B/E, D) | APPROVE WITH FIXES (F1 to F3 major; F4 to F14) |
| 2 | opus / high | `cb54ea35..34cb102a` (fix 1, task C) | APPROVE WITH FIXES (R2-1, R2-2, R2-10 before the flip; R2-5 to R2-8 before enabling C) |
| 3 | opus / high | `f00a4964..28278264` (flip, fix 2) | APPROVE WITH FIXES (R3-1, R3-2 major) |
| 4 | sonnet / high | `5d1fc6ef..8abdbae0` (fix 3) | APPROVE WITH FIXES (R4-1 minor: `Retry-After` can still outlast the margin; documented) |

## 6. PathWhiz branches (local only, not pushed; owner checkout untouched, its `lib/pwml_parser.rb` sha256 `eb00ccaa...ffae` before and after)
| order | branch | commit |
|---|---|---|
| 01 | `t2pw/01-vcv-node-typo` | `17af3272` |
| 02 | `t2pw/02-transport-state-remap` | `48ab7499` |
| 03 | `t2pw/03-transport-transporter-remap` | `c5c12d69` |
| 04 | `t2pw/04-biological-state-exact-match` | `80bdaca7` |
Records (`NN-<slug>.md`, optional `.patch`) and the stack README are in `pathwhiz_changes/`. The harness keeps importing from the owner's working copy (clone plus `git diff HEAD`); branch tip plus the owner's 10-line `element-id` change equals what it runs.

## 7. Open questions for the product owner
1. Enable `T2PW_TRANSPORT_SIDES`? Options: keep off (no cost); fix R2-5 to R2-8, answer Q1/Q2/Q4 of `c_live/RESULT.md` section 11 (does "out of the cell" entail extracellular; unnamed sides empty or `cell`; retry on an empty reply), then repeat the test with at least 3 more repaired live runs through `run_pathway` and the fourth control (about 1.5 to 3 hours of live runs, paid calls).
2. Nucleic acids: accept the cautious rules (conflicting rows not looked up; protein-coding to DNA only with positive paper evidence; costs RyhB), or Option A (let the lookup decide conflicting rows; LHCB1.1 becomes DNA against its own paper) and drop the protein-context check (CHL27 becomes DNA)? The "positive evidence" is only the absence of a protein word (R3-5).
3. MG-63 and Saos-2: keep them in the payload only (now), write the BTO cell-line terms as tissue rows (BTO:0001596, BTO:0000971; a cell line written as a tissue), or widen PathWhiz `cell_type.rb` to accept `CVCL_` (a PathWhiz change, would be branch 05).
4. Confirm or reject the reviewed synonym `type iii taste cell` to CL:0002285 (and propose it to the Cell Ontology).
5. Where should dropped components, renamed membranes and unknown nucleic types be disclosed for a user (R3-4): the export report, the text version, or the PWML `description`? Today only counters in the builder.
6. The circuit breaker is global per fetcher, not per host (R4-2): five failures from one host skip the other hosts' lookups. Per-host counters cost a few lines.
7. Whether to push the four PathWhiz branches as pull requests, and in what order; 04 is untested on a populated database.
8. Internal rendering of the paper pathways (D-W21g) was not part of this phase.

## 8. Dispatches (19 plus resumes)
| # | task | model / effort |
|---|---|---|
| 1 | PathWhiz patches 01-03 | sonnet / medium |
| 2 | equal-side transport trace | sonnet / high |
| 3 | wildcard guard (writer) | sonnet / high |
| 4 | ontology B and E | sonnet / high |
| 5 | nucleic acid D | sonnet / high |
| 6 | importer branch 04 | sonnet / high |
| 7 | transport sides implementation and live test (hung on a permission prompt after the queue) | sonnet / high |
| 8 | review 1 | opus / high |
| 9 | fix round 1 | sonnet / high |
| 10 | task C result and analysis | sonnet / high |
| 11 | review 2 | opus / high |
| 12 | live runs (restart killed it; import redone by the orchestrator) | sonnet / medium |
| 13 | default flip and pins | sonnet / medium |
| 14 | fix round 2 | sonnet / high |
| 15 | final replay, identity, harness measurement | sonnet / medium |
| 16 | review 3 | opus / high |
| 17 | fix round 3 | sonnet / high |
| 18 | final tests in a clean worktree | haiku / medium |
| 19 | re-review of fix round 3 | sonnet / high |
