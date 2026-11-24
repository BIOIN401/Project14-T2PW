# REVIEW R4: acc/verifier-r2 round 3 (code 4d623f2a, docs 28b49228, main merge a34309af)

Reviewer: independent ws-reviewer, read-only. No test, replay or job run (driver3.sh holds the heavy lock;
I also ran no tiny unit checks, everything below is from reading). Lane delta judged as
git diff a34309af 4d623f2a (3 files: llm/client.py, service/reaction_verifier.py,
tests/test_reaction_verifier.py) plus 28b49228 (ACCURACY.md only). Evidence read: logs/gates5.log,
logs/ut14.log, logs/vd18adm-3.log, json/vd18adm-3.json, scripts/vd18_admit2.py, scripts/score3.py.

## VERDICT PER FLAG

| decision | verdict |
|---|---|
| merge acc/verifier-r2 default-off | APPROVE-WITH-NITS |
| enable T2PW_REACTION_VERIFIER | NOT YET: conditional on the live 3x3 AND an offline paired re-application (below); B1 residual must be measured |
| enable T2PW_INTERACTION_TYPING | NOT YET: code fixes hold; conditional on the live typing criteria only |
| enable T2PW_VD18_SPAN_TEMPLATES | NO: stays off (0 gains; typed-proposal path untestable on stored data) |

## FLAGS UNSET = MAIN? YES

- reaction_verifier.py changes are reachable only through apply_to_stage_one (flag-gated, unchanged hook).
- llm/client.py:934 adds max_retries=None to chat_detailed; line 997 keeps os.getenv LLM_MAX_RETRIES
  when it is None. I checked every caller: chat (client.py:912), pipeline.py:3855,
  preprocessor.py:159 and vision/scope.py:47 all pass keywords only, and none passes max_retries. The
  name is not read before line 997. Only reaction_verifier passes it. No other caller changes
  behaviour.
- gates5.log (HEAD 4d623f2a, all four flags unset): SMOKE 511; golden 2/8/4/24 unedited; equivalence
  8/14/6; vision 370 and 1 skipped; web 83; lane 320; client files in their own process 35; 0 surviving.
  Read, not reproduced.
- Hygiene: the lane delta adds no cache, quarantine report, .env or host. ut14.log prints only a .env
  path and "Key starts with: None".

## R3 ITEMS

| item | status | evidence |
|---|---|---|
| P1 effect words | FIXED. _EFFECT_WORDS (reaction_verifier.py:226-235) matches whole forms; promoter, reductase, reduction, inducible, raised, accumulation no longer count | test_contact_or_naming_words_are_not_effects, test_binding_to_a_promoter_stays_unsigned; both fail on the old stems |
| P2a sentence split | FIXED. split_sentences (:181-199) cuts only before a capital or bracket, never after a listed abbreviation or a single-letter initial; Fig. 2 has a digit after it, so it is never cut | test_a_genus_initial_does_not_hide_the_negation_before_it (the R3 example, verbatim) |
| P2b negation bypass | FIXED. :219 checks the prefix whatever the quote holds | test_a_dropped_leading_loss_is_refused_... |
| P3 wall clock | FIXED as designed. Budget.take (:329-347) picks the timeout and attempt count (2 or 1) whose worst_case_call_seconds fits the time left; _ask passes both | test_budget_is_a_wall_clock_bound_across_retries |
| P4 second spelling | PARTIAL, see B1 | test_a_rename_never_leaves_two_spellings_of_one_compound covers acid/-ate only |
| P5 cofactor folding | FIXED. _currency_compatible (:563-573) needs nested currency sets per side | test_rows_differing_in_cofactor_are_not_folded |
| P6 off_pathway | FIXED. Flag only (:643-646) | test_off_pathway_only_flags_never_drops (3 cases) |
| P7 typing word | FIXED for interaction_type then relationship (:769), and both are written. The exporter also falls back to class; Stage-1 interaction rows carry no class in any stored leg (class appears only on entities), so this is a nit | test_the_typing_word_is_read_as_the_exporter_reads_it |
| P8 | Documented limit, unchanged | ACCURACY.md section 12 |
| R3 Q3 doc sentence | CORRECTED in section 11, with an explicit note | ACCURACY.md diff |

Old tests removed: the isolated-off_pathway-is-omitted test and the never-empties test. Both were
removed because P6 changed the behaviour on purpose, and they are replaced by
test_off_pathway_only_flags_never_drops. No test was edited just to make it pass.

## FINDINGS

B1 | blocks ENABLING the verifier until measured | reaction_verifier.py:507-511 _spelling, :534-560
_apply_renames, :615-629 additions | The second-spelling guard folds only case, punctuation and acid/-ate
(-ous acid/-ite). Synonyms are not folded: an abbreviation against a full name (DHB vs
2,3-dihydroxybenzoic acid, DHB-AMP vs 2,3-dihydroxybenzoyl-AMP), CoA vs coenzyme A, L-serine vs
serine. A rename is still applied when the new name has no folded match and the old name occurs in no
other row. Scenario: the model renames "activated 2,3-DHB" (one row) to "DHB-AMP", a name in the paper,
while other rows say "2,3-dihydroxybenzoyl-AMP". The same applies to an incomplete addition of "DHB"
next to an existing "2,3-dihydroxybenzoic acid". If the mapper resolves both spellings to one compound,
prefreeze_resolution.py:860-870 raises AMBIGUOUS_RENAME_TARGET and the run ends with no PWML. That is
worse than any wrong row. The enterobactin paper uses exactly these abbreviations. Cheap tightening, if
the PO wants it before the live result: accept a new or added name that is not already a payload
spelling only when it contains no all-capitals token of 2-5 letters (abbreviation). Otherwise score3
already counts ambiguous_rename per leg, and the gate below makes it decisive.

N1 | nit, recall only | reaction_verifier.py:226-235 | Whole-word matching drops agent nouns: inhibitor,
activator, repressor, inhibitory. "2,3-DHB is a competitive inhibitor of apo-EntB isochorismatase
activity" no longer states an effect, so the legitimate binding -> inhibition re-typing seen live in
R1 is refused. This fails safe (no new wrong edge; binding already exports under D-W26), but it cuts the
typing gain. Consider adding inhibitor/activator/repressor with a test.

N2 | nit, reliability | Budget.take | Attempts are capped at 2, against 8 by default. On transient provider
errors the verifier now fails open more often (no change, error recorded). That is acceptable, but the
live run should report the error rate per paper. A long paper whose calls always time out gets no
verification at all.

N3 | nit | apply_types :769 | Also read the class fallback (interaction_relationship_word). Not
reachable on current Stage-1 rows.

N4 | nit | the typed-resolution comparison in vd18adm-3 is vacuous (0 proposals in either arm). The doc
says so. That is enough to keep VD-18 off; it is not evidence for turning it on.

## QUESTION (a): THE NEGATIVE-CONTROL CRITERION

The R3 wording (0 reactions in both arms) was the pipeline gate. It is the wrong gate for the verifier.
After P6 the verifier cannot remove an off-pathway or not-stated row, and it never adds net reactions:
a chain join replaces 2 or more rows with one, and a model whole step replaces the row it covers. So
it cannot fix BASE inventing lipid A -> LPS, and VT got 0 only because Stage 1 was empty. Yes, replace
it with:
- per VT run, reactions after the verifier <= reactions before it (from reaction_verifier_report.json);
- per paper, VT final reactions <= BASE final reactions (median of the 3 runs);
- an offline paired application of the 4d623f2a verifier to BASE own stage1_payload.json for the negative
  control: no row added, the lipid A row at most flagged, and typing creates no signed edge there that
  BASE lacks;
- the BASE lipid A -> LPS reaction is recorded as a pre-existing Stage-1 finding, not charged to this lane.

## QUESTION (b): CLIENT TEST FILES IN THEIR OWN PROCESS

Acceptable. ut14.log ran the same file set on C:/t/acc-base at fd814b8e: 3 failed, 271 passed. The
failures are the same three tests, with an auth error from leaked client state, so this is a
pre-existing order dependence, not caused by this branch. Gates5 runs those two files alone: 35 passed.
Conditions: record it as a pre-existing test-isolation finding with a ticket, keep the own-process step
in every gate script, and note that SMOKE (511) does not depend on it.

## WHAT THE LIVE 3x3 (driver3.sh, 4d623f2a) MUST STILL SHOW

The design covers 3 papers: enterobactin, chlorophyll and the negative control. It drops menaquinone
PMC12421875, which has 9 certified gold reactions (the largest gold set) and is where dead end 2 deleted
5 certified steps. MenD and MK-7 are dropped too. The gold-loss gate is therefore under-covered. Add,
without new model spend: an OFFLINE PAIRED RE-APPLICATION. Run apply_verdicts and apply_types of
4d623f2a on the stored answers2 and answers3 for all 18 stored Stage-1 payloads (the section 3 replay
set). This is deterministic, needs no calls, and shows the new rules against every gold paper.

Verifier (all must hold):
- Live: every gold reaction is matched in no fewer VT runs than BASE runs (enterobactin, chlorophyll).
- Offline re-application: 0 gold reactions lost on any of the 18 payloads, for both answer samples.
- 0 omitted or folded rows that match a gold reaction (score3 checks this). Hard zero.
- 0 AMBIGUOUS_RENAME_TARGET in VT that BASE does not also have on the same paper, and 0 runs where the
  second spelling came from a verifier rename or addition (B1). Runs ending without a PWML: VT no more
  than BASE.
- Verifier wall time at most 600 s and at most 6 calls in every VT run. Report present in the run folder.
  Error (fail-open) rate reported per paper; chlorophyll must not be all errors.
- Reactions after the verifier <= before, every run; the negative-control rules in (a).
- lock_manifest_entries_removed equals the omitted plus folded lock ids.

Typing (all must hold):
- Every changed interaction in VT is checked against ix_labels.json, with unlabelled pairs hand-labelled
  before scoring. 0 changed rows labelled binding or skip; 0 with the opposite sign. Hard zeros.
  Chlorophyll is the main bed.
- 0 changed rows whose old word was signed.
- Type accuracy VT at least BASE.
- Negative control: 0 signed edges created.

VD-18: nothing live. It stays off.

## WHAT I COULD NOT VERIFY

- Any test or gate number (read from logs).
- That chat_detailed honours timeout and max_retries on every provider path. P3 rests on
  worst_case_call_seconds being a true upper bound, as its docstring claims.
- Whether the mapper actually resolves DHB and 2,3-dihydroxybenzoic acid to one target (B1 is a
  reachable shape, not an observed crash).
- The live legs (in progress).
