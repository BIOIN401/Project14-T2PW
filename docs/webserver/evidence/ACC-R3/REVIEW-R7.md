# REVIEW R7: acc/verifier-r3 at code f26c7f17 (docs 460c3804), against REVIEW-R6

Reviewer: independent ws-reviewer. No heavy job. Read-only probes (venv python -I, no network, under 1 s
each), run on C:/t/lane-acc (f26c7f17 code) and on C:/t/acc-r3base (main 1a52fb55): scratchpad
probe_r7.py, plus lost.py over json/evalv_r6_*.json.

## VERDICT

(a) merge default-off: APPROVE.
(b) ship with the verifier ON: APPROVE, after the one live server job below. The live list is the R6
list, unchanged apart from item 4 below.

## R6 FINDINGS

| item | status | evidence |
|---|---|---|
| B1 fold onto a Greek-collapsed name | FIXED: the fold target is looked up by reference_key only (reaction_verifier.py:577); a spelling-key-only match refuses as on 1a52fb55 | probe 1 below; test_a_greek_letter_never_folds_..., which the record says fails on d5d0ec95 |
| B2 beta-alanine written as alanine | FIXED: _spelling = reference_key + acid/-ate fold (:515-522), so it is never coarser than reference_key on letters | probe 2 below; test_an_added_beta_alanine_... fails on 1a52fb55 and d5d0ec95 |
| N1 orphan scan | FIXED: every string of every in-scope process row except its name and prose keys, plus protein_complexes components (:641-672) | test_an_old_name_still_used_as_an_enzyme_or_complex_component_keeps_its_entity, all three cases |
| nit docstring | FIXED | diff |

All _spelling callers are inside reaction_verifier.py (lines 552, 566-567, 574, 581, 742, 748, 750); no
other module uses it.

## MY PROBES (same inputs on both trees)

| case | 1a52fb55 (main, live) | f26c7f17 |
|---|---|---|
| 1 rename "cyclic carotenoid" to beta-carotene; alpha-carotene row exists | refused, row stays generic | renamed to beta-carotene; both rows stay; no fold (better than a refusal and correct) |
| 2 add beta-alanine; "alanine" elsewhere | written as "alanine" (WRONG) | written as beta-alanine |
| 3 beta row repaired, alpha twin otherwise identical (changed-row fold pass) | not folded | not folded: _key keeps Greek letters too |
| 4 add NAD+; payload spells NAD | written as "NAD" | refused (B1 abbreviation guard; spelling keys now differ) |

Case 4 is the NAD+/NAD side effect you asked about. It is not a correctness regression: nothing wrong is
written, and writing "NAD+" next to "NAD" is exactly the second-spelling shape that can end a run with
AMBIGUOUS_RENAME_TARGET, so refusing is the PO-rule direction. It costs recall only for an abbreviated
co-product whose spelling differs from the payload one by "+" (NAD/NAD+, NADP/NADP+). The lane replay
shows no action changed against d5d0ec95 on any of the 55 payloads plus item-4, so the cost is 0 on the
corpus. Accepted. Nit: if it shows up live, a two-entry alias (NAD+ to NAD, NADP+ to NADP, applied only
when the payload already uses the bare form) would win it back without widening the key.

Residual, low: a non-abbreviated name that differs from a payload name only by "+" ("Na+" vs "Na") can
still be written as a second spelling. The prompt forbids adding metal ions and protons. Not blocking.

## FLAGS UNSET = MAIN: YES

Only reaction_verifier.py and its test file change, all inside the flag-gated hook. gates10 (HEAD f26c7f17,
all four flags unset): SMOKE 511, golden 2/8/4/24 unedited, equivalence 8/14/6, vision 370 and 1 skipped,
web 83, lane 333, client 35, every EXIT 0, 0 surviving (log read, not reproduced).

## REPLAY

My tally over evalv_r6_2, evalv_r6_3, evalv_r6_4g and evalv_r6_srv: 0 gold lost at Stage 1 or final in
every file; server item-4 final 5 -> 5. The lane reports the same 14 new folds and 0 written names whose
reference_key differs from the proposed name (namecheck.py; not re-run). As the record says, the corpus
has no Greek-letter pair; B1 and B2 rest on the unit tests and my probes.

## URGENCY NOTE

B2 is live on the server today: main 1a52fb55 has the verifier on and writes "alanine" for an added
beta-alanine (probe 2). Merge this branch and deploy it promptly; it removes a live wrong-compound path.

## LIVE CHECK BEFORE CONFIRMING THE SHIP (R6 list, item 4 refined)

One server job, re-submitting the item-4 paper (PMID 40400396, Gemma, 3 draws). Pass when ALL hold:
1. a PWML is produced;
2. EntE and EntF each appear once in final_mapped.json;
3. no PWML_IR_DUPLICATE_NAMED_ROW and no AMBIGUOUS_RENAME_TARGET;
4. gold R1-R5 matched at least as in item-4 attempt 1, and every rename or addition in
   reaction_verifier_report.json has a written name whose reference_key equals the model proposed one
   (refused ones listed as refused);
5. lock_manifest_entries_removed equals the omitted plus folded lock ids;
6. every name in entities_dropped is absent from every field of final_mapped.json.
If it fails, the default-off merge still stands; switch the server flag off and report.

## WHAT I COULD NOT VERIFY

- gates10 and the ut-r6b logs (read only).
- namecheck.py output (read from the record).
- Live behaviour of the fix (the server job above is still to run).
