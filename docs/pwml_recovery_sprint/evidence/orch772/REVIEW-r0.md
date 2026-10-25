# ORCH-772 review, round 0 (independent reviewer)

Branch `orch772/coverage-omissions`, base `3ef3fa9c`, tip `dbc3b961`. Worktrees: `.claude/worktrees/rev772-base` (3ef3fa9c),
`.claude/worktrees/rev772-tip` (dbc3b961). The full diff was read (`git diff 3ef3fa9c dbc3b961`), every hunk of
`src/t2pw/app/reaction_evidence.py`, the three evidence scripts, the tests, the fixtures and the sample. Scratch, scripts
and logs are under `C:/t/rev772/`.

## VERDICT: APPROVE WITH NOTES

None of the 8 blocking criteria is met. I checked each one myself (section 3). Coverage figures were counted
independently on **all 81 replayed legs**, not just 8: drawn processes were counted from the written PWML XML, and totals
with my own formula. They agree with the harness on every leg and every kind. Verdicts, PWML sha256 and every
output-file digest match between base and tip on all 93 manifest legs. The coverage output is byte-identical under
PYTHONHASHSEED 1 and 2.

## 1. Boundary check

`git diff --name-only 3ef3fa9c dbc3b961`, leaving out the G11 reports:
* src: **only** `src/t2pw/app/reaction_evidence.py` (+523/-11, `git diff --numstat`).
* evidence scripts: `orch768_replay_corpus.py`, `orch770_morning_summary.py`, `orch770_night.sh`, plus the new
  `orch772_retrim_reaction_evidence_bundles.py` (fixture re-trim helper) and helper copies under `evidence/orch772/`.
* tests and pins: `tests/test_orch772_pwml_coverage.py` (new), `tests/test_reaction_evidence.py` (1 line),
  4 fixture bundles, `docs/reaction_evidence_sample.txt` (the golden for test 2).
* docs: `ORCH-772-RESULT.md`, `evidence/orch772_corpus_coverage.{json,txt}`, `evidence/orch772_reaction_evidence_sample.diff`.
* Protected `src/t2pw/app/streamlit_app.py`: blob `ed5d5536` at base and at tip (`git ls-tree`), so **byte-identical**.
  `reaction_evidence_view.py`: blob `186bfdcd` at both, so untouched.
* The commits are split cleanly: code+tests `d1006c10`, scripts `2d561020`/`a238df69`/`fba739cd`/`c1216df3`, test pin
  `43445214`, docs `dbc3b961`.
Result: **within the boundary.**

## 2. Tests I ran (my own runs; every one through bounded_run.py `--heavy-lock ORCH-772-rev`, FINAL SURVIVING COUNT 0)

| G11 | what | result |
|---|---|---|
| 41 | corpus, **base harness** @3ef3fa9c on base src | exit 0, 33 s |
| 42 | corpus, tip harness on tip src, PYTHONHASHSEED=1 | exit 0 |
| 43 | corpus, tip harness on tip src, PYTHONHASHSEED=2 | exit 0 |
| 44 | `C:/t/rev772/count.py`: independent count (stdlib, never imports t2pw) | 81/81 legs match, 0 mismatches |
| 45 | `C:/t/rev772/inspect_flags.py`: the 2 legs my count flagged | both benign (3.2) |
| 46 | `C:/t/rev772/cmp.py`: base vs tip records | identical apart from `pwml_coverage` |
| 47 | focused: test_orch772_pwml_coverage, test_reaction_evidence, test_orch768_replay_corpus, test_orch762_protein_identity, test_pwml_interaction_protein_wrappers, test_stage_one_paper_dictionary::test_what_a_model_reads_is_the_bases_bytes, test_orch757_llm_token_budgets | **175 passed** |
| 48 | SMOKE, the 22 files | **511 passed** (expected 511) |
| 49 | `tests/test_orch759b_recorder_reattach.py` ALONE | **7 passed** |
| 50 | `tests/test_reaction_evidence_view.py` ALONE | **11 passed** |
| 51 | `tests/test_reaction_evidence_app_hook.py` ALONE | **1 passed** |
| 52 | `C:/t/rev772/render.py`: the .txt for 3 live legs and 1 replay | rendered (3.3) |

`g11_evidence.py check --task ORCH-772`: 52 reports, 21 direct pytest reports, all with a pin verdict, exit 0.
Logs: `C:/t/rev772/41.log` ... `52.log`. pytest counts were taken from the piped stdout.

## 3. Blocking criteria, one by one (all clear)

**B1. LLM-bound text / Reaction Evidence reaching an LLM: clear.** Grep of `reaction_evidence` across `src/`: the only
importer is `src/t2pw/app/reaction_evidence_view.py:31`. Only `streamlit_app.py:7279` (unchanged) imports that view.
`reaction_summary.txt` (pipeline.py:1350; fed to the curator and the enrichment agent) is a different file and is not
touched. In `gap_resolver.py:333`, `reaction_evidence` is an unrelated local variable. The 759b 15-request golden ALONE
gave 7 passed (G11 49). The paper-dictionary and token-budget goldens passed inside 47.

**B2. Drawn counted wrongly / counts disagree with artifacts: clear.** `count.py` counts drawn processes as the
`<reaction>`/`<transport>`/`<interaction>` children of `<reactions>`/`<transports>`/`<interactions>` in the
driver-named PWML file of each replay directory, or 0 when no PWML file exists. It counts the total from the replay
payload (canonical.json, or final.enriched.json when the replayed quarantine refused) plus the pre-freeze removals
(strict rows, only when the quarantine passed; recorded rows prepended on canonical checkpoints; `quarantined_locked_reactions`
entries not already listed by lock), minus KI-014 merged duplicates. Result: **81/81 legs equal on drawn, total,
per-kind drawn/total and merged-duplicate count** (`C:/t/rev772/44.log`). Pooled: 344/1081, and the 56 PWML legs
average 61.8% with 22 at or above 75% and 14 at 100%. Both match RESULT section 1. The legs the brief named:
* no PWML: ORCH-755 enterobactin 2201 0/13; ORCH-767 wnt_ca 0/48; ORCH-759 chlorophyll 0/97; ORCH-767 enterobactin 0/21.
* PWML with omissions: ORCH-759p chlorophyll 33/112; 2026-08-24_1428/PMC12444477 5/33; ORCH-756 menaquinone 10/13.
* WNT PCP: ORCH-759 6/42 (PWML XML: 3 reactions, 2 transports, 1 interaction); ORCH-767 1/66.
* `runs_verify/2026-08-22_2017/PMC12856317`: 0/3 (replay writes a PWML that draws nothing, F-2).
Code that guarantees "no PWML means 0 drawn": `reaction_evidence.py:600` (`drawn = False if no_pwml`), the no_pwml
branch in `_omissions` (about line 739), and `build_report` at 998-1004. The harness passes the IR and a PWML name only
when `_pwml_artifact_name` produced a file (`orch768_replay_corpus.py:546-549`, and the `if pwml_name:` in
`pwml_coverage`). So a diagnostic_only release reads as no PWML.

**3.2 F-1 (recorded quarantine rows on canonical legs): no double count and no drop.** My count carries every recorded
row and matches on all 73 canonical legs. Double-count probes: a strict row whose name is still a payload process of
the same kind, and a `quarantined_locked_reactions` entry matching a strict row by name but not by lock. Two legs were
flagged; both are benign (G11 45, `C:/t/rev772/45.log`):
* ORCH-756 menaquinone: `rxn_lock_005` "MenD reaction" is a `duplicate_locked_reaction` (not counted). Its twin
  `rxn_lock_004` is the strict-quarantined row, already deduplicated by lock.
* ORCH-759 chlorophyll: two distinct reactions share the name "ALA condensation to PBG" (`rxn_lock_001` kept,
  `rxn_lock_002` quarantined).
No canonical leg had a replayed quarantine that refused, so the not-recorded branch at `orch768_replay_corpus.py:586-592`
never ran.

**3.3 B3. Omitted process missing / reason not verbatim: clear (see NOTE 1 for the no-PWML cause).** Reasons are copied
with `_text(row.get("reason"))` from the strict row, the locked record, or the IR `withheld.reason`. In the live WNT PCP
render (`C:/t/rev772/render_live_759_wnt_pcp.txt`), 36 OMITTED blocks = 42 - 6, each with Reason, Known issue and Add by
hand. In the replay render of PMC12856317 (`render_replay_2017_PMC12856317.txt`), 3 blocks = 3 - 0: KI-004
`non_protein_actor_in_modifiers`, KI-009 `bare_protein_endpoint`, KI-008 `untyped_interaction`. In the live no-PWML render
of ORCH-767 enterobactin (`render_live_767_enterobactin.txt`): 4 pre-freeze blocks (KI-001), then 17 payload processes
under "Not drawn because no PWML was written". Read as a user would, every omission is present.

**B4. KI contradicting the appendix: clear.** Test 1 (`test_the_code_table_is_the_catalogue_appendix`) parses the
appendix (82 codes, 19 messages) and every User-workaround cell, and compares them with `KNOWN_ISSUES`. It passed in 47.
By hand: exact codes before prefixes; "then label by" on `protein_complex_component_unrepresentable:<name>:<reason>`
takes the text after the last colon. The inner reasons that `_entity_representability` can produce
(strict_quarantine.py:719-773) all give the KI the appendix intends, including the colon-bearing
`malformed_placeholder:` and `..._unresolved:`, which fall back to KI-015 as in the appendix. "unclassified" appears only
for reasons the appendix lacks: `coarse_grained_same_entity_transformation` and
`output_subset_of_input_without_distinct_product` (process_normalizer.py, F-3), `unreferenced_after_quarantine` and
`degree_zero_after_quarantine`, and the run-directory no-PWML sentence (NOTE 1).

**B5. Pipeline verdict or PWML bytes changed: clear.** I ran the base harness on base src (41) against the tip harness on
tip src (42), with `C:/t/rev772/cmp.py` (G11 46): 93/93 legs, the same classes, `pwml_sha256` equal on all 56 PWML legs,
every replay record and every non-replay row equal once `pwml_coverage` is removed (the record includes every verdict and
the `files` digest of every file the replay wrote), and the summaries equal. Pass rate 56/91 to 56/91.

**B6. Values guessed: clear.** Drawn is read only from the IR beside a written PWML. An IR that does not line up (count
or names, `_same_ir_rows` at 683) or is older than per-row verdicts gives `Not recorded`; those rows are listed apart,
never as omitted (test 8; the sample leg, test 9). Every replayed leg is measured (`"drawn": null` appears 0 times).

**B7. Out of boundary / protected file: clear** (section 1).

**B8. Non-determinism / network: clear.** tip1.json and tip2.json share sha256 `e288332d...`, which equals the committed
`evidence/orch772_corpus_coverage.json`. tip1.txt and tip2.txt are identical (`d559ef29...`); the committed txt differs
only in line 1, the `t2pw resolved:` path. The offline probe refused 3 attempts at base and at tip, and every leg has an
empty `network_refused` (cmp.py). `PATHBANK_DB_*`, `PYTHONPATH` and `PYTHONHASHSEED` were unset in my shell (`env`).

## 4. Regression proof / G9

The tests are labelled NEW ACCEPTANCE TESTS FOR A NEW CAPABILITY (coverage, OMITTED section, KI table, corpus coverage),
and none of these exists at base. That label is correct, and no base failure is owed. The one change to existing
observable behaviour is deliberate, documented and pinned: `Coverage:` becomes `Core-term coverage:`
(`reaction_evidence.py:1083`, `test_reaction_evidence.py` test 3, the sample golden). The fixture bundles gained only
keys the new section reads. Test 13 (`test_trimmed_bundle_renders_like_its_committed_leg`) passed in 47, which proves
each bundle still renders byte-identically to its committed leg. No pinned baseline was reverted. No biological gate is
touched (no pipeline module changed).

## 5. Decision compliance

* D-114 section 1 (coverage = drawn / text-record processes): satisfied. Orchestrator ruling: KI-014 listed but not
  counted (`_entry` `counted`); KI-008 counted in the headline plus a secondary line (`pwml_coverage_lines`, 1106ff.).
* D-114 section 3 (OMITTED section: exact reason, catalogue id, hand fix; coverage at top; LLM text unchanged):
  satisfied, with NOTE 1 on no-PWML legs read from a run directory.
* D-104 section 4 / AMENDMENT 2: satisfied (B1).
* D-113 section 3: n/a. The lane makes the zero-draw PWML visible (F-2) and decides nothing (RESULT Q4).
* D-114 AMENDMENT 1: n/a for this lane; see NOTE 4.
No open product decision was resolved on the lane's own authority. Q1-Q7 are left to the product owner.

## 6. Findings (all NOTES; no BLOCKING)

1. **NOTE (the most material one).** For a run directory with no PWML, the cause is a file-absence sentence:
   "neither pathway.pwml nor pathway.review_required.pwml is in this run" (`reaction_evidence.py:998-1002`). So
   every process lost with the leg shows `known issue: unclassified`, has no Reason line and has no Add-by-hand line
   (`omitted_lines`, 1147). On the corpus this is the largest bucket: 275 of the 737 missing processes. Yet the real reason
   for the leg is in the leg directory. Examples: the `RESULT.txt` `message:` line of 2026-08-22_2017/PMC12856317 reads
   "post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_stage3_gates", which is an appendix code;
   the ORCH-767 enterobactin leg has `final_stage3_gate_report.json` errors. `orch770_morning_summary.failure_line`
   already reads that `message:` line. The app path is fine, because it passes the export error. The corpus harness also
   passes no `no_pwml_cause`, so its by-KI tally buckets these as "no PWML written (whole leg)", and RESULT section 1
   hand-annotated their classes. This is not blocking under the fixed standard: nothing is guessed, no catalogue code was
   read and mislabelled, and every process is listed. It is a gap against the "exact reason" of D-114 section 3 for these
   processes. Recommended follow-up: read the recorded failure message of the leg as `no_pwml_cause` for run directories,
   and the failure stage/reason of the replay in the harness.
2. **NOTE.** In `orch770_night.sh` the morning summary is called without `--src`, so `--src` defaults to the `src` of the
   PRIMARY checkout (`orch770_morning_summary.py:96`). The ORCH-774 coverage column will therefore come from the
   `reaction_evidence` in the primary checkout, not the one in the pinned worktree, and reads "n/a" until ORCH-772 is merged
   there. Either pass `--src` with the pinned worktree src or merge before launch. Also, the summary now imports t2pw (it
   used to be stdlib-only) and is still not wrapped by bounded_run. That was already the case before this lane. The default
   task id leaves every path unchanged (`GOLD` resolves to `orch770_gold_accuracy.txt`; `bash -n` OK).
3. **NOTE.** A merged duplicate whose kept twin was itself omitted is listed with the KI-014 workaround "check that the
   kept twin is in the PWML". Example: ORCH-756 menaquinone, where MenD `rxn_lock_005` folded into `rxn_lock_004`, which
   strict quarantine removed. The count is right, because the twin is counted once as OMITTED. The wording misleads slightly.
4. **NOTE (forward compatibility, ORCH-773).** The renderer reads exactly the shapes on `orch773/omit-per-row` @ `f35de933`:
   `detail.omission` {gate, cause, slot, name, failing_entity{bucket,name}} (`_omission_detail`, 657), and removed-entity
   `detail.gate` and `detail.actor_removed_from[{process, process_name, slots, catalysts_kept}]` (785-788, 1184). Under
   D-114 AMENDMENT 1 section 2, a transporter dropped from a kept transport would appear only as an "Entities left out"
   sub-line ("removed from <transport> (...): transporters; catalysts kept: none"). It would get no OMITTED block with
   reason / KI / Add-by-hand, which Amendment 1 section 2 asks for, and "catalysts kept" is reaction wording. ORCH-773
   should extend this. Also check that `scrub_detail` does not elide the names inside `detail.omission`.
5. **NOTE.** For an actor refusal (e.g. `non_protein_actor_in_modifiers`), the OMITTED block does not repeat the refused
   actor. It appears only one section up, under QUARANTINED, as "Essential participant" (the PLP cofactor in
   `C:/t/rev772/render_replay_2017_PMC12856317.txt`). Layout only.
6. **NOTE (theoretical).** `leg_percent` returns 0.0 for a leg with 0 processes (`orch768_replay_corpus.py:854`), and the
   morning summary prints "0/0". No corpus leg has a total of 0 (minimum 1). In the app view, the metric is still labelled
   "Coverage" (`reaction_evidence_view.py:154`, core-term ratio) while the header block on the same screen now says
   "Core-term coverage". See Q5.
7. **NOTE (process, implementer).** RESULT section 7.4: a reserved G11 path was reused after exit 95 (self-reported).
   Section 11: free RAM was checked 4 times, not before each of the 40 jobs, which departs from hard rule 2. Neither changes
   the evidence.
8. **NOTE (ORCH-771 appendix spot check, unreviewed doc).** I grepped 25 codes in `src/t2pw`. 22 are literal hits in the
   cited modules (identity_admission, stage_contracts/payload_models, strict_quarantine x9, ir, extraction_ladder,
   process_normalizer/stage_one_ensemble, pipeline/release_status/ir, map_ids/ir, paper_lexicon/stage_one_ensemble). The
   other 3 are accounted for: `no_drawable_process` is documented as absent from the code; `parenthetical_definition_bridge`
   is documented as on an unmerged branch; `protein_endpoint_missing_*` is built by concatenation at `ir.py:83-84`. The two
   uncatalogued F-3 reasons were confirmed in `process_normalizer.py`. No contradiction found.
9. **NOTE (environment, not this lane).** A `python.exe -` pair (PIDs 195392/189168) under a live bash of another lane
   (rev769d, started 14:49) was present for my whole session. Its parent was alive, so it is owned, not orphaned; I did
   not kill it. A wrapped ORCH-773 corpus job was also running at the end. None of my processes survived.

## 7. My view on RESULT section 9 (non-blocking product questions)

* Q1 denominator: keep (a). The headline including KI-008 is the literal D-114 section 1 definition, and the secondary
  line serves readers who exclude by-design rows. Prefer cataloguing the F-3 no-op removals (Q3) over excluding them.
* Q2 RCT / sub-pathway: listing without counting is fine while no corpus leg has one. Revisit when one appears.
* Q3: yes, catalogue `coarse_grained_same_entity_transformation`, `output_subset_of_input_without_distinct_product` and the
  removed-entity reasons. Cheap, and it removes 12 "unclassified" rows.
* Q4: needs a product-owner ruling on the D-113 section 3 reading. This lane correctly only makes it visible.
* Q5: recommend (b), at least the relabel. The screen is now internally inconsistent ("Coverage" metric vs
  "Core-term coverage" header line).
* Q6: per leg on the 91-leg headline (the D-113 ruling), the others as information.
* Q7: "records identical apart from pwml_coverage, PWML hashes identical" is the right acceptance test for a lane that
  moves no verdict. Worth making an orchestrator rule.

## 8. What I could not verify

* The live `orch770_night.sh` run with a task id: not run (no live runs allowed). Checked by reading and `bash -n` only.
* The morning-summary sample (`evidence/orch772/morning_summary_ORCH-767_sample.md`): not regenerated. Its totals agree
  with my renders/corpus where both exist (enterobactin 0/21, sour taste 7/12, wnt_ca 0/48, menaquinone 14/14).
* The app path (session export result as `pwml_written` / `no_pwml_cause`): covered only by the AppTest files (G11 50/51),
  not exercised with a real session.
* ORCH-773 + ORCH-772 combined rendering: not run, because the branches are not merged together. NOTE 4 is from reading.
* The effect of `scrub_detail` on `detail.omission` names: not checked.

## G11 reports (mine)
`.claude/worktrees/rev772-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-772/`
41-rev-corpus-base, 42-rev-corpus-tip-s1, 43-rev-corpus-tip-s2, 44-rev-count, 45-rev-inspect-flags, 46-rev-compare,
47-rev-focused, 48-rev-smoke, 49-rev-alone-759b, 50-rev-alone-view, 51-rev-alone-hook, 52-rev-render (.json), with pins
`g11/pin/ORCH-772/47..51-rev-*.pin.json`. Not committed: they are untracked in the rev772-tip worktree.
