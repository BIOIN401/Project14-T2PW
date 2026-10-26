# ORCH-772b review, round 0 (independent reviewer)

Branch `orch772b/no-pwml-cause`, base `00eb12d3`, tip `30e4c675` (commits 8bf8c07c, 31eed6e6, 02f826b2, 30e4c675).
Reviewed: the actual diff `git diff 00eb12d3 30e4c675` (every code, script, test and KI hunk; the 10,757-line corpus JSON
and the G11/pin JSONs by provenance only). Severity standard applied exactly as fixed before round 0.

## VERDICT: REJECT (one blocking finding, B-1; everything else holds)

B-1 is a label defect in the output that the morning summary and review pack would print tonight. It does not touch the pipeline,
the PWML or any coverage number. The fix is small (see B-1). Everything else in the charge holds and I reproduced it.

## Findings

### B-1 BLOCKING (standard item 3): a no-PWML leg whose export failed at IR validation is labelled KI-028 (time limits)

* Leg: `runs_verify/2026-08-25_1216/papers/PMC12856317/strict` (live, strict, no PWML). `RESULT.txt`: `message : PWML export failed:
  PWML IR validation failed.`, `issue codes : (none reported)`. The leg holds `pwml_ir_validation.json` with `ok: false`, error
  code `protein_complex_missing_components`.
* The tip reader (`evidence/orch772b_no_pwml_cause.py`) does not read `pwml_ir_validation.json`, so it falls through to source 4,
  the `RESULT.txt` message. The text is verbatim and names the right step, so standard item 2 holds. The problem is the label.
  `known_issue_id` matches the appendix shorthand row `"PWML export ..."` (KI-028, under "extraction did not finish inside the time
  budget") as a wildcard (`reaction_evidence.py:204`, base code). The result is:
  * review pack `.txt`: `No PWML written: PWML export failed: PWML IR validation failed. (known issue: KI-028)`;
  * morning summary cell: `0/5 (0%); full count 0/5 (0%); no PWML: KI-028`.
* Why this contradicts the catalogue:
  * KI-028 is "Time limits".
  * The appendix row abbreviates the driver message "PWML export did not finish inside the time budget" (`driver.py:3176`).
  * "PWML IR validation failed." has no appendix row of its own (probe: `unclassified`).
  * The IR code recorded in the leg, `protein_complex_missing_components`, is KI-015. The corpus already gives KI-015 for the same
    leg (`causes.out.txt:31`, cmp row in `C:/t/rev772b/25.log`), so review pack and corpus now disagree on one leg.
  * The probe also shows `PWML export failed: PWML required-field gate failed.` -> KI-028, where the appendix says "KI-006 or KI-019".
    The same applies to `PWML export failed: no XML was produced ...`.
* This is new with this lane. At base the cause was the fallback sentence, labelled `unclassified`. The lane added the `RESULT.txt`
  message as a source and so exposed the wildcard row.
* It can recur on the current tree tonight. `streamlit_app.run_pwml_export` returns IR-validation errors under `pwml_ir_validation` /
  `pwml_ir_report` (`streamlit_app.py:4973-4988`). The driver adds to `issue codes` only `required_gate_report`,
  `stage3_contract_report` and `pwml_contract_report` (`driver.py:3187-3190`). So any current-tree IR-validation export failure
  writes exactly this `RESULT.txt`, and the summary would print KI-028.
* Evidence: I reproduced it myself with `C:/t/rev772b/r2/ki028_probe.py` on the tip tree, G11 `36-rev-r2-ki028.json`, exit 0, 0
  survivors. The previous reviewer saw the same result: `28.log` leg0, `35.log` probe, `27.log` flag "STEP? ... export failed but
  reader step message".
* Required fix, smallest inside the boundary: in `orch772b_no_pwml_cause.py`, read `pwml_ir_validation.json`. When it has `ok:
  false`, take its `errors[].code`, deduplicated and joined with "; ", verbatim, before the `RESULT.txt` message. In pipeline order
  that comes after the issue codes, because the required-field gate runs before IR validation. This leg would then show
  `protein_complex_missing_components` (KI-015), matching the corpus.
* Root fix, which needs the orchestrator to widen the boundary: narrow the appendix-derived row `"PWML export ..."` in
  `reaction_evidence.py` `KNOWN_ISSUES["messages"]` to "PWML export did not finish inside the time budget", so that no `PWML export
  failed: ...` message is ever labelled KI-028. This touches `reaction_evidence.py` outside the two allowed hunks, so it is not for
  this lane to make alone.
* Add one test on a leg fixture with this shape: no issue codes, `pwml_ir_validation.json` `ok: false`.

### N-1 NOTE: F-2 (joined cause labelled as a whole). Non-blocking per the standard; my view
* The live survey (`C:/t/rev772b/live_survey.out.txt`, G11 `27`) finds 14 no-PWML legs where the joined cause is labelled
  `unclassified` although every catalogued item has an id.
* Examples:
  * ORCH-759 / ORCH-759p menaquinone-strict: `species_missing_taxonomy, species_missing_classification`, each item KI-019;
  * `2026-08-25_1216` PMC12444477: nine `gate.protein_..._is_missing_a_uniprot_or_drugbank_identifier` codes, each item KI-001.
* Menaquinone is one of the six papers tonight. If it fails at the required-field gate again, the summary says `unclassified` where
  the catalogue says KI-019. Recommend closing F-2 (label by first catalogued item, or list per item) before ORCH-774 reads the
  summaries.

### N-2 NOTE: a timeout leg names a catalogued code in its cause but labels `unclassified`
* Leg: ORCH-755 ara_operon-strict (and the other ORCH-755 timeouts).
* Message: "the child process was still running after 3600s and was killed, so this paper+mode produced nothing (budget_exhausted)".
* The appendix has no row for this message, so `unclassified` is allowed by the standard. But it ends in `budget_exhausted`, which
  is KI-028. Same family as F-2. Timeouts are common overnight.

### N-3 NOTE: corpus and live reader can disagree on a leg, by design
* The corpus labels the replay on the measured tree. The review pack labels the live records of the leg.
* Example: ORCH-759 chlorophyll-strict `2026-10-22_1231`. Live shows `entity_type_overlap:4` (KI-011, quarantine). The corpus
  replay at `00eb12d3` shows `reaction_enzyme_must_be_protein_complex` (KI-006, export).
* Both are correct for what they read. One sentence in the RESULT would stop a reader taking either for a bug.

### N-4 NOTE: the corpus report "leads with" the target line only within its coverage section
* The TARGET line (29/91 headline, 22/91 full count, pass rate 56/91) is the first line of the PWML COVERAGE section, but that is
  line 172 of 201 in `tip1.txt`, after the per-leg rows.
* This is layout, so non-blocking.

### N-5 NOTE: the G9 proof for `orch770_night.sh` is a text-presence check
* `test_g9_the_night_runner_measures_coverage_with_the_pinned_worktrees_code` asserts that the string `--src "$WT/src"` is in the
  script. G9 asks for a behavioural failure on base.
* It does fail on base (`34.log`), and the change is one argument. A behavioural proof would run the script with stub commands.
  Acceptable here; flagged for the G9 standard.

### N-6 NOTE: `orch770_night.sh`
* `bash -n` passes. The only change is `--src "$WT/src"` on the summary call.
* Default task id `ORCH-770`: all paths are unchanged. The `--src` change applies with or without a task id, as the charge asks
  (ORCH-772 review note 2). I do not read it as standard item 7.
* Note:
  * The summary imports `orch772b_no_pwml_cause` from the PRIMARY checkout `evidence/` (`$EV`). The primary must therefore hold
    the merged file at launch, or the coverage cell degrades to `n/a (ModuleNotFoundError)` (it never raises).
  * Q4 still stands: the summary runs outside `bounded_run`.

### N-7 NOTE: small things
* In `orch770_morning_summary.coverage` and `orch768 fmt_cov`, the "not recorded of N" path shows the full count with no headline
  label (unchanged base path; no percentage is claimed).
* `HEADLINE_EXCLUDES_BY_DESIGN = False` does not restore the old text byte for byte: the wording changes from "cannot type" to
  "cannot carry" and the line order changes. It is an unused toggle (YAGNI), harmless.
* `evidence/orch772/pwmlstats.py` (historical ORCH-772 dev script) still calls the full count "headline".

### Views on the open items (non-blocking)
* F-3: agree non-blocking; it belongs with ORCH-773 rendering.
* Q2: a fifth source (the DETAIL last line) is fine only if it stays verbatim and is ordered after the message. The WNT PCP crash
  leg is correctly `unclassified` today.
* Q3: leaving `orch755_review_pack.py` alone is fine; it is off the ORCH-774 path.
* Q4: wrap the summary in `bounded_run` in a later lane.

## What holds (with evidence)

| check | result | evidence |
|---|---|---|
| Boundary | `reaction_evidence.py` exactly 2 hunks: the constant (+3 lines at 87) and `pwml_coverage_lines` (1107-1130). `streamlit_app.py` untouched. No other `src/` file. | `git diff --stat 00eb12d3 30e4c675` |
| 1 LLM bytes | Display-only change. `reaction_evidence` is imported in `src/` only by `reaction_evidence_view.py` (-> `streamlit_app`). `test_what_a_model_reads_is_the_bases_bytes` and `test_orch757_llm_token_budgets` pass. | grep; `29.log` (182 passed) |
| 2 cause verbatim, right step | All 197 live no-PWML legs: reader == raw read, NOT VERBATIM flags = none. Step spot checks below. | `27.log`, `live_survey.out.txt` |
| 4 verdicts / PWML / numbers | 93 legs; replay records identical outside `pwml_coverage`; 56/56 PWML sha256 equal; every count key identical; only `no_pwml` labels moved (25 legs, 275 processes); base ex-by-design == tip headline, base headline == tip full count; pass 56/91 both. | `25.log` (cmp.py, base.json from `21.log`, tip1.json from `22.log`) |
| 5 headline + full count | `.txt` headline line + "Full count" line; corpus TARGET line + per-leg `fmt_cov`; summary cell `x/head; full count x/full`. | diff; `ms767tip/MORNING_SUMMARY.md`; `tip1.txt:172` |
| 8 determinism / network | tip1 == tip2 == tip3 (PYTHONHASHSEED=3) byte for byte (JSON and TXT); offline probe ok; net=0 on 81 rows. | `22.log`, `23.log`, mine `37-rev-r2-corpus-tip-seed3.json` |
| G9 | 5 `test_g9_*` fail on base (4 behavioural, 1 text); 2 `test_new_*` (labelled NEW ACCEPTANCE) fail on base with FileNotFoundError. | `34.log` |
| Pins moved | `test_orch772_pwml_coverage.py`: 3 pins moved, each annotated with the before value. | diff |
| SMOKE | 511 passed at tip, 0 survivors. | `30.log` |
| AppTest alone | view 11, 759b 7, hook 1 passed. | `31-33.log` |
| KI-065 / KI-021 | Match D-114 AMENDMENT 2 section 3. KI-065 adds no reason code, so it cannot contradict the appendix. | diff, DECISIONS.md:7452 |

Spot checks (raw files read by me or by the survey, step confirmed against the RESULT.txt message stage):

| leg | cause shown | step | KI | ok |
|---|---|---|---|---|
| ORCH-767 enterobactin-strict | "Generated protein complex ... missing positive stoichiometry." | final gate (msg: at final_pre_export_stage3_gates) | KI-012 | yes |
| ORCH-767 chlorophyll-strict | "Unknown protein/modifier reference: PIF1; ..." | final gate | KI-004 | yes |
| ORCH-767 wnt_ca-strict | `unexportable_entity:1` | quarantine (msg: final_pre_export_gate_lifecycle) | KI-016 | yes |
| ORCH-767 wnt_pcp-strict | the "refinement_generate_pwml button never rendered" message | message | unclassified | yes (Q2) |
| ORCH-759 enterobactin-strict | "Protein has degree 0 after normalization: enterobactin synthase" | final gate | KI-013 | yes |
| ORCH-755 sour_taste-strict | `actor_schema_not_canonical` | issue codes (Stage 3 revalidation) | KI-023 | yes |
| runs_verify 2026-09-16_2152 PMC13534948 | `minimum_core:...` x3 | quarantine | KI-038 | yes |
| runs_verify 2026-08-25_1216 PMC12444477 | 9 `gate.protein_...` codes (final_stage3_gate_report.json is ok:true; codes from RESULT.txt) | final gate per message | unclassified | F-2 (N-1) |
| ORCH-759p menaquinone-strict | `species_missing_taxonomy, species_missing_classification` | issue codes (required-field gate) | unclassified | F-2 (N-1) |
| runs_verify 2026-08-25_1216 PMC12856317 | "PWML export failed: PWML IR validation failed." | message | **KI-028** | **no: B-1** |

## G11 reports

Mine (made at the tip tree, 0 survivors, heavy lock acquired/released):
* `.claude/worktrees/rev772b-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-772b/36-rev-r2-ki028.json`
* `.claude/worktrees/rev772b-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-772b/37-rev-r2-corpus-tip-seed3.json`

From the previous reviewer, relied on after checking each one: wrapper clean at `30e4c675`, right tree in the command, FINAL
SURVIVING COUNT 0. All are under the same directory: `21-rev-corpus-base`, `22-rev-corpus-tip-s1`, `23-rev-corpus-tip-s2`,
`25-rev-cmp-retry`, `26-rev-packs-summary-tip`, `27-rev-live-survey`, `28-rev-render-tip`, `29-rev-focused`, `30-rev-smoke`,
`31-rev-alone-view`, `32-rev-alone-759b`, `33-rev-alone-hook`, `34-rev-g9-base`, `35-rev-ki-probe`. (`24.log` was a lock-held exit
95, so the child never ran.)

## Not verified
* I did not re-run the focused tests, SMOKE or AppTest myself. I reused `29`-`33`, which were checked for provenance and survivors.
* The base morning summary for ORCH-767 was not produced (`ms767base/` holds only the review index). Base vs tip summary numbers
  rest on the code reading (headline = full - by_design_withheld) and the base G9 failure `0/21 (0%)` in `34.log`.
* That `22.log` / `23.log` ran with an unset PYTHONHASHSEED (random per process) is inferred from the environment, not recorded. My
  seed-3 run removes the doubt.
* The corpus `replay_no_pwml_cause` pwml_export fallback order (required gate, then IR blocking, then IR validation, then Stage 3)
  matches an independent error-keyed reading on all 25 real no-PWML replays. A synthetic leg with several non-empty lists was not
  built.
