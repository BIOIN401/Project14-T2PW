# REVIEW R5: acc/verifier-r2 at 37e3e614 (code e72c2759), against REVIEW-R4

Reviewer: independent ws-reviewer. I ran no tests and no jobs. Two small read-only scripts (venv
python -I, files in my scratchpad) tallied gold matches from the stored score files. Code diff judged:
4d623f2a..e72c2759 (reaction_verifier.py, test_reaction_verifier.py only; 37e3e614 is ACCURACY.md only).
Evidence read: json/score4g.json, json/score5g.json, json/evalv_fg3_2.json, evalv_fg3_3.json,
evalv_fg3_4g.json, evalv_g4_r5.json, logs/gates8.log, ACCURACY.md sections 13, 14, 14a-c.

## VERDICT PER FLAG

| decision | verdict |
|---|---|
| merge default-off | APPROVE |
| enable T2PW_REACTION_VERIFIER (granularity merge) | APPROVE TO ENABLE, conditional (below): the R4 bar is met for Gemma 4 31B on this code |
| enable T2PW_INTERACTION_TYPING | DO NOT ENABLE on the Gemma server: merge off only (see Typing) |
| T2PW_VD18_SPAN_TEMPLATES | stays OFF (unchanged from R4) |

No blocking findings for the default-off merge or for enabling the verifier under the conditions below.

## FLAGS UNSET = MAIN: YES

All changes are inside reaction_verifier.py, reached only through the flag-gated hook. gates8.log (HEAD
4d29c4bf, code e72c2759, all four flags unset by gates_inner.sh): SMOKE 511, golden 2/8/4/24 (unchanged
files), equivalence 8/14/6, vision 370 and 1 skipped, web 83, lane 327, client files in their own process
35, every EXIT 0, 0 surviving. Read, not reproduced.

## CODE SINCE R4

- B1 abbreviation guard (reaction_verifier.py:536-541 _abbreviated; used at :568 for renames and :646
  for additions). Correct, and it fails safe. Cost: the co-products the prompt asks for (CO2, AMP, PPi,
  ATP, NADH) all have 2+ capitals, so an addition of one is refused unless the payload already spells
  it. That is recall left on the table, not a risk. Nit: say so in the prompt or the doc.
- N1 agent nouns (:235-236). Correct; inhibitor, inhibitory, activator, repressor added with tests.
- Measurement (:814-845): model, calls, seconds go in the report; the finally block records them on every
  path. Correct.
- F-G1 fail-open per step (:841-842). out is set to the reaction result only after apply_verdicts returns
  (tuple assignment), so report reactions exists exactly when its decisions are in the returned payload,
  and prune_lock_manifest prunes only those. Correct. It is the one change that makes the error path do
  MORE than before: it keeps a finished reaction step. That matches the success path, so it is fine.
- F-G2 (:395-417, :617-619). Each substrate must reach every product, and not_stated / off_pathway rows
  never count as coverage. Stricter, so it fails safe. The test fails on the old rules (record says so).
- F-G3 (:429-438 _add_compound on reference_key; :550-575 _apply_renames). A rename is refused onto any
  name another entity or row carries (spelling key or reference_key); additions adopt the existing
  spelling through _one_spelling, and _add_compound never writes a second entity row on the exporter key.
  _spelling is coarser than reference_key (it strips all punctuation and folds acid/-ate), so an addition
  that reference-key-matches an existing name always takes that exact spelling. Correct.

## THE THREE REWRITTEN TESTS: judged legitimate

1. test_incomplete_may_correct_a_wrong_participant. Old: rename onto an existing name, then fold. New:
   rename onto a name no entity carries. The rule changed on purpose (option c, after a live failure,
   PWML_IR_DUPLICATE_NAMED_ROW on vt3-PMC12096016), not to force a pass. The new rule is pinned by
   test_a_rename_onto_another_entitys_name_is_refused, which the record says fails on 4eb9cd22
   (ut-fg3b-old.log, 2 failed). Nit: the docstring still says "the live DHB-ate row". That live case
   (DHB-ate renamed to the existing DHB-AMP) is now REFUSED and the row stays wrong and flagged. The
   docstring should say so, and so should ACCURACY.md: R1 F7c is effectively withdrawn for renames onto
   existing names.
2. test_vague_rename_then_fold_into_the_identical_row: deleted and replaced by the new F-G3 test. Folding
   itself is still covered (split-chain fold, addition-then-fold in score5g vt1/vt2/vt3, cofactor test).
3. test_an_abbreviated_synonym_is_never_written_next_to_the_full_name, last clause: flipped from "merged
   into 0" to "merged == empty and repaired == empty". Same deliberate rule.
None of the golden-seam, SMOKE or equivalence tests changed. The offline cost of option (c) is recorded:
enterobactin precision on deepseek answers 0.878 -> 0.844 and 0.907 -> 0.867, 0 gold lost.

## DOES THE 36 + 12-LEG EVIDENCE MEET THE R4 ENABLING BAR (VERIFIER)? YES, FOR GEMMA

My own tally, runs in which each gold id was matched (final output):

| paper | run | BASE | VT |
|---|---|---|---|
| enterobactin | 5g, e72c2759 code | R1 3, R2 2, R3 3, R4 3, R5 3; PWML 3/3 | R1-R5 all 3; PWML 3/3 |
| MenD | 5g, e72c2759 code | R1 3 | R1 3 |
| menaquinone | 4g, 82bda3b6 | R1-R9: 3,3,2,2,2,2,2,3,3 | identical |
| chlorophyll | 4g, 82bda3b6 | 7 ids x 2 (base1 errored, flags off) | 6 ids x 3, R2146840 2, plus R2146850 1; PWML 3/3 |
| negative control | 4g | 0 rows; base1 aborted (entity_missing_name, flags off) | 0 rows, 3/3 PWML |
| sour taste | 4g | 0 rows | 0 rows |

The 4g failures (MenD R1 2 < 3; enterobactin VT PWML 1/3) were the three defects F-G1, F-G2 and F-G3,
all fixed and re-measured green in 5g. The other four papers were measured on 82bda3b6, not
e72c2759. I accept that as covering the final code, because every change since then is equal or more
conservative on those runs. F-G2 and F-G3 only refuse more. F-G1 differs only when typing errors, and
those papers had 0 errors. On those papers the verifier omitted, folded and repaired nothing (menaquinone
only flagged).

Offline paired application, my own check of the stored files: 0 gold lost at Stage 1 or final in all
four of evalv_fg3_2, evalv_fg3_3 (deepseek answers, 18 payloads each), evalv_fg3_4g and evalv_g4_r5
(Gemma answers, all 18 BASE payloads). Enterobactin gains one match in each set.

R4 criteria, live 5g: report present with model google/gemma-4-31b-it; at most 2 calls; at most 10 s
(at most 338 s in 4g, under 600); fail-open 0/3; lock removals equal omitted plus folded, and
pruned_but_present is empty; no AMBIGUOUS_RENAME_TARGET; VT no-PWML 0 = BASE 0. The one "folded row
matching gold" (vt1 row 5 into row 1, both R2) left R2 matched in that run. I accept it, and I refine
the criterion: a fold is a loss only when the gold id is then unmatched.

Conditions for enabling the verifier:
- Only with the measured model. The verifier uses OPENROUTER_EXTRACTION_MODEL, the Stage-1 model. If
  the server model changes, re-run the 5g-style re-measure first.
- Only on code at or after e72c2759, with T2PW_VERIFIER_GRANULARITY unset (merge) and typing and VD-18
  off.
- The server keeps its single worker (web/retry.py:45 "the single worker"). That satisfies the P8
  shared-tmp manifest limit; record that dependency.
- PO sign-off. The hard rule is met on this evidence, but n=3 per arm per paper is small and Gemma Stage 1
  is noisy (BASE enterobactin R2 was 1/3 in 4g and 2/3 in 5g). Keep reaction_verifier_report.json in
  every job folder and spot-check the first server jobs for omitted rows.

## TYPING: ZERO LIVE CHANGES IS ENOUGH TO MERGE IT OFF, NOT TO ENABLE IT

Under Gemma, typing changed no row live (36 + 12 legs) and none offline (Gemma typing replay: 0 changes).
The R4 hard zeros therefore pass vacuously: there is nothing to be wrong. The only evidence that the
rules behave when changes happen is the deepseek offline replay (184 judged rows, 0 wrong signs, 0
binding/skip rows changed). With the server on Gemma, enabling typing buys no measured gain and has a
measured cost: an extra call per job, and one typing call ran 335 s and returned invalid JSON (vt3
MenD, and offline base1 MenD 229 s), half the 600 s budget. Keep it off. Re-open only for a model that
actually proposes re-typings, with a live arm where changes occur and are scored against ix_labels.

## VD-18

Unchanged: off. No gain in any replay. The typed-proposal path is untestable on stored data.

## OPEN, OUTSIDE THE LANE (not blocking)

- 14b entity_missing_name at post_audit: Gemma 2 of 35 legs, one of them BASE with flags off, so it is a
  pipeline bug. I agree it is not the lane. Note one thing for the ticket: flagged rows carry a verifier
  record (verdict, reason, model_quote) that downstream LLM stages (audit, curator) see in the payload.
  The vt1 enterobactin abort happened with only flags set. The BASE occurrence makes a causal link
  unlikely, but the ticket should check it.
- AMBIGUOUS_REFERENCE "tRNA Glu" on base1 chlorophyll (flags off): pre-existing, record it with the ticket.

## NITS

- Fix the test_incomplete_may_correct_a_wrong_participant docstring, and add a sentence to ACCURACY.md:
  the R1 F7c live case (rename onto an existing name) is now refused by F-G3.
- Note the B1 recall cost (abbreviated co-products are added only when the payload already spells them).

## WHAT I COULD NOT VERIFY

- That evalv_fg3_* were produced by e72c2759 apply_verdicts (stated in the record; I read the outputs).
- Any gate or unit number (logs only); that 2 tests fail on 4eb9cd22 (ut-fg3b-old.log not opened).
- The model answers themselves (the 5g legs were scored by score5g.py; I tallied its output, not the PWML).
