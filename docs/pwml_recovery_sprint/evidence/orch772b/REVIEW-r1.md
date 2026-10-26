# ORCH-772b review, round 1 (final round, independent reviewer)

Branch `orch772b/no-pwml-cause`, tip `1f223061`. The commits are b1c046a1 (merge of sprint `bab79db6`, which brings in
ORCH-773), 06684ac2 (cause reader), 429acee0 (`reaction_evidence.py`, tests and one appendix row) and 1f223061 (docs).
The r0 severity standard applies, plus the r1 addition: it is BLOCKING if the F-2 fallback or the narrowed row changes any label
on a PWML-writing leg, or makes any message label more permissive.

## VERDICT: APPROVE WITH NOTES

r0 B-1 is fixed and F-2 is closed. I found no blocking item, including against the r1 addition.

## r0 B-1 and F-2: resolved

* **B-1 (fixed).**
  * The reader now takes `pwml_ir_validation.json` `errors[].code` verbatim when `ok: false`. It sits after `issue codes` and
    before the message (`orch772b_no_pwml_cause.py`, r1 hunk).
  * The `"PWML export ..."` row is narrowed to the literal "PWML export did not finish inside the time budget", in `KNOWN_ISSUES`
    (`reaction_evidence.py:206-207`) and in the appendix row. That is the only appendix change in r1.
  * The leg `runs_verify/2026-08-25_1216/PMC12856317/strict` now shows `protein_complex_missing_components`, labelled **KI-015**.
    It was KI-028 at r0 and is `unclassified` at base (`live_diff.log`, G11 `44`).
  * The literal is a strict subset of the old pattern, so the narrowed row is less permissive, not more.
  * The only driver messages starting "PWML export" or "audit and DB mapping" that are time-limit messages are `driver.py:2925` and
    `:3176`.
* **F-2 (closed).** ORCH-759 and ORCH-759p menaquinone-strict now show `species_missing_taxonomy, species_missing_classification`,
  labelled **KI-019** (both legs; `live_tip.json`). PMC12444477 shows **KI-001**.

## Checks against the r1 addition (BLOCKING if violated): none violated

| check | result | evidence |
|---|---|---|
| Labels on PWML-writing legs, corpus | Every non-`no_pwml` omission row (ki, group, reason) is identical across all 93 legs, so no label moved on the 72 PWML-writing legs. | `cmp.log` "other differences: none" (G11 `41`) |
| Labels on PWML-writing legs, live (review-pack path) | 241 live leg directories, 44 of them write a PWML: 0 label changes, 0 coverage-number changes. On no-PWML legs, changes fall only in the `no_pwml` group (818 rows, 70 legs), never on a pre-freeze or IR omission row. | `live_diff.log` (G11 `42`, `43`, `44`) |
| Message rows more permissive | No message row was widened; the only row edit narrows one. The F-2 fallback runs only when the whole text matches nothing, and each item must still fully match an existing code or message row. | diff `reaction_evidence.py:316-321` |
| What the fallback actually labels | 149 distinct cause, reason and message texts on live legs. 43 are labelled only through the fallback, and all 43 are lists of recorded items (driver codes joined with ", ", or gate messages joined with "; "). Each label is the catalogued id of one of those items. | `fallback.log` (G11 `45`) |

## Base `bab79db6` against tip `1f223061`, on the corpus

* 93 legs compared. Replay records are identical outside `pwml_coverage`, and all 72 PWML hashes are equal.
* Every count key (`drawn`, `total`, `by_kind`, `by_design_withheld`, `merged_duplicates`, `not_counted`, `not_recorded`,
  `pwml_written`) is identical.
* Base "excluding by design" equals the tip headline, and base headline equals the tip full count. The pass rate is 72/91 on both.
* The only label moves are **9 no-PWML legs, 29 processes**:
  * KI-022 on 8 legs (`no_defensible_reaction_support`);
  * KI-038 on PMC13534948.
  * This is exactly the list in RESULT section 13. For every leg, the cause matches my independent reading of the replay record.
* Target line: 32/91 headline, 23/91 full count, pass rate 72/91.
* Evidence: `C:/t/rev772b/r1/cmp.log`; base run G11 `38`, tip run G11 `39`.
* Determinism: my tip run under PYTHONHASHSEED=5 matches the committed `evidence/orch772b_corpus_coverage.txt` byte for byte,
  apart from the first line (the resolved `t2pw` path). The offline probe passes.

## Boundary

* `git diff bab79db6 1f223061 -- src`: only `reaction_evidence.py` changes. Its hunks are the r0 two (the constant and
  `pwml_coverage_lines`), plus the B-1 row and its one-line comment (`:206-207`, `:296`), plus the F-2 fallback (`:316-321`).
  Nothing else changes.
* `streamlit_app.py` is untouched. `orch770_night.sh`, the morning summary and the review pack have no r1 change.
* The rest of the 30e4c675..1f223061 delta (`strict_quarantine.py`, `process_normalizer.py`, the ORCH-773 tests and fixtures) is
  the sprint merge, identical to `bab79db6`.

## Tests (my own, at tip `1f223061`, 0 survivors)

* Focused: 146 passed. The files were `test_orch772b_no_pwml_cause`, `test_orch772_pwml_coverage`, `test_reaction_evidence`,
  `test_orch768_replay_corpus`, `test_orch773_omit_per_row`, the bases-bytes test and `test_orch757_llm_token_budgets` (G11 `46`).
* SMOKE: 511 passed (G11 `47`).
* G9 at the r0 tree `30e4c675` (exported to `C:/t/rev772b/r1/g9r0`, with the tip test file): both new `test_g9_r1_*` fail
  behaviourally (G11 `50`; `48` and `49` were harness setup failures, not test results):
  * the IR-validation test: the reader returns the bare message instead of `protein_complex_missing_components`;
  * the list test: it returns `unclassified` instead of `KI-019`.
* My r0 probe (G11 `36`) shows the other half: `PWML export failed: ...` gave KI-028 at r0.

## Notes (non-blocking)

1. **NOTE: a truncated driver code passes the label to a later item.**
   * `RESULT.txt` `issue codes` can hold codes truncated at about 70 characters. Example:
     `gate.protein_porcine_lactate_dehydrogenase_is_missing_a_uniprot_or_dr` cannot match the KI-001 prefix+suffix row.
   * The fallback then labels by the next item that maps (KI-012).
   * That label is a real catalogued item of the same cause, so it does not contradict the appendix. The KI-001 first item is
     simply not visible to the matcher.
2. **NOTE: the split also cuts inside a single message.**
   * The split pattern (semicolon or comma, then whitespace) also cuts one message that contains such a separator. On the live set
     this happened once, the FtsH message "declared as both a protein and a protein_complex; entity types must be disjoint", and
     the fragment matched nothing.
   * A fragment of prose that fully matches a code is theoretical; there is no artifact.
3. **NOTE: one wildcard row remains.** `"audit and DB mapping ..."` is still a wildcard KI-028 row. The only driver message with
   that subject is the time-budget one (`driver.py:2925`), so there is no mislabel today. Narrowing it like the PWML row would
   remove the last B-1-class risk.
4. **NOTE: an r0 G9 proof is now synthetic.** `test_g9_the_corpus_replay_records_its_own_cause...` became a unit test on a
   hand-made replay record, because ORCH-773 makes the fixture leg export. The change is documented under merge rule 4. The
   end-to-end proof now rests on the corpus run, where all 9 no-PWML legs carry their causes, checked independently.
5. **NOTE: r0 N-2, N-4 and N-6 still stand.**
   * N-2: a timeout leg whose message ends "(budget_exhausted)" is `unclassified`; the fallback does not split it into a match.
   * N-4: the target line leads only the coverage section of the corpus report.
   * N-6: the morning summary imports the reader from the primary checkout, so the primary needs the merged file before 23:00.

## G11 reports (all under `.claude/worktrees/rev772b-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-772b/`)

These were made at tip `1f223061` (wrapper clean), each with 0 survivors and the heavy lock acquired and released:
`38-rev1-corpus-base`, `39-rev1-corpus-tip-seed5`, `41-rev1-cmp-retry`, `42-rev1-live-base`, `43-rev1-live-tip`,
`44-rev1-live-diff`, `45-rev1-fallback`, `46-rev1-focused`, `47-rev1-smoke`, `50-rev1-g9-at-30e4c675-r3`.

Not results:
* `40-rev1-cmp`: exit 95, lock held, the child never ran.
* `48`: exit 98, the pinned_pytest tree refused the exported tree.
* `49`: `tree_pin` was missing.

Scripts and outputs are in `C:/t/rev772b/r1/`.

## Not verified

* I did not run the AppTest files alone in r1. The implementer ran `32-r1-alone-reaction_evidence_view`. That view renders only
  `header_lines` and `coverage_lines`, and r1 does not touch them.
* SMOKE used the same 22-file selection as r0. If ORCH-773 changed the SMOKE list in TEST_MATRIX, that was not cross-checked.
