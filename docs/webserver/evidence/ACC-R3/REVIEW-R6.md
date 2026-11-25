# REVIEW R6: acc/verifier-r3 (code d5d0ec95, docs 0bdc6528), base main 1a52fb55

Reviewer: independent ws-reviewer. No heavy job. One tiny single-file probe (venv python -I, no
network, no wrapper, under 1 s), run against both C:/t/lane-acc/src (d5d0ec95) and C:/t/acc-r3base/src
(1a52fb55). Script: scratchpad probe_r6.py. Read: the diff, ACCURACY.md section 15, gates9 record.

## VERDICT

(a) merge default-off: REJECT as is. The fix is small and stays inside this boundary (B1, B2 below);
approve after it. The flag is OFF by default but ON on the live server, so merging this is shipping it.
(b) ship with the verifier ON without a new live re-measure: NO for d5d0ec95. YES after B1 and B2 are
fixed, with the minimum checks in the last section (one offline replay plus one live server job). No
new 3x3 is needed.

## FLAGS UNSET = MAIN 1a52fb55: YES

Only reaction_verifier.py and its test file change; everything is inside apply_verdicts, reached only
through the flag-gated hook. gates9 (all four flags unset): SMOKE 511, golden 2/8/4/24 unedited,
equivalence 8/14/6, vision 370 and 1 skipped, web 83, lane 330, client 35 (record; not reproduced).

## BLOCKING

B1 | NEW in d5d0ec95, gold-loss path | reaction_verifier.py, the refused branch of _apply_renames
(F-G3 line plus the two new lines under it) | The fold target is chosen as known.get(_spelling(new)) first.
_spelling drops every character outside a-z0-9 after casefold, so Greek letters vanish: _spelling of
alpha-carotene and of beta-carotene are both "carotene" (reference_key keeps them apart: alphacarotene,
betacarotene). A rename the model proposes to beta-carotene is therefore resolved to the payload spelling
alpha-carotene, and then folded. Reproduced with the probe:
  rows: lycopene -> alpha-carotene (LCYE); lycopene -> "cyclic carotenoid" (LCYB);
  model: vague, rename "cyclic carotenoid" -> beta-carotene (in the paper).
  d5d0ec95: row 2 is renamed to alpha-carotene and folded into row 1, and LCYB joins the alpha-carotene
  row. The beta-carotene step is gone.
  1a52fb55: refused; both rows stay. No loss.
The same collapse hits beta-alanine vs alanine, gamma- vs alpha-tocopherol, and any name whose only
difference is a non-ASCII letter. Fix: fold only when an existing name has reference_key equal to
reference_key(new), and take THAT spelling. If only _spelling matched, refuse without folding (the 1a52fb55
behaviour). Add the alpha/beta-carotene case as a negative test.

B2 | PRE-EXISTING on main 1a52fb55, live on the server now (verifier ON) | reaction_verifier.py
_one_spelling and _spelling, used by the incomplete additions in apply_verdicts | Same root cause. An
added co-substrate is rewritten to the payload spelling with the same _spelling key, so a model
addition of beta-alanine next to an existing "alanine" is written as "alanine". The wrong compound goes
into the row, reported as "repaired". Probe on BOTH trees: add_inputs beta-alanine -> inputs pantoate,
alanine. Fix: make _spelling spell Greek letters (or build it on reference_key and then fold acid/-ate),
so it is never coarser than reference_key on letters. Add a test. Until it ships, the server can write
a wrong compound into a row on any paper that names an alpha/beta/gamma form next to the plain name.

## THE CHECKS YOU ASKED FOR

Can a fold merge two biologically different rows?
- Twin test (_fold_onto_existing, sides): full participant lists on both sides by reference_key,
  cofactors included. reference_key keeps digits, + and :, spells Greek letters, and drops only case,
  punctuation, hyphens and dashes. So L- vs D-, (R) vs (S), cis vs trans, E vs Z, alpha vs beta, NAD+ vs
  NAD, Fe2+ vs Fe3+ and 2- vs 3-phospho all stay apart. Prime marks are dropped (3-phospho vs
  3-prime-phospho collide); in practice those name one compound. The twin test itself is sound.
- The defect is upstream of it: which name the old one is renamed TO (B1).

Orphan-entity removal: incomplete. The check scans inputs, outputs, entity_1, entity_2, cargo and
elements_with_states of in-scope process rows. Stage-1 payloads also reference entities through
reactions.enzymes, transports.transporters and protein_complexes.components (keys seen in every stored
stage1_payload). A renamed participant can be a protein (apo-EntB, a complex). If its old name is still
used as an enzyme, a transporter or a complex component, the entity is dropped while still referenced. N1
below; cheap fix: scan every string value of every process row except name, evidence and
provenance_lineage, plus protein_complexes components.

Enzyme union: correct and bounded. Twin rows have identical participants, so they are one reaction; the
union dedupes on the protein / protein_complex / name key, as the existing fold does. Nit: two spellings
of one enzyme ("EntE" and "EntE reaction" in the test, or a name with a parenthetical) both survive.
Cosmetic.

Lock bookkeeping: correct. A folded row gets action folded_into_N and loses its lock id through _omit;
omitted_lock_ids counts folded actions; rows appended later are not scanned. The fold runs after the
verdict loop and before the changed-row fold pass, and the out_of_scope rows it makes are skipped there.
The server-shape test checks the prune.

## THE TWO EDITED TESTS

1. test_a_rename_onto_another_entitys_name_is_refused: the EntE twin row was REMOVED from the fixture so
   the refusal assertion still holds. That is a fixture edit that keeps an old assertion green under a new
   rule. It is legitimate only because the removed shape is now pinned by its own positive test
   (test_both_spellings_of_ente_and_entf_fold_to_one_each, which also asserts no second spelling and the
   lock prune) and the docstring says so. Accepted. Nit: say in the docstring which test now covers the
   original vt3 four-row shape.
2. test_an_abbreviated_synonym_..., last clause: flipped from refused to folded (1 into 0), with
   out_of_scope asserted. A deliberate rule change, no new spelling written either way. Accepted.
Neither touches a golden, SMOKE or equivalence test.

## NON-BLOCKING

N1 | orphan scan misses enzymes, transporters, complex components (above).
N2 | the fold is order-dependent: twin returns the first live match. With three identical rows it picks
the lowest index. Harmless (identical participants).
N3 | the Stage-2 lactate re-addition (section 15 finding) is outside the lane and correctly reported.
The verifier sees only Stage 1, so a row Stage 1 ruled out of scope comes back unverified. It belongs
on the pipeline ticket.

## EVIDENCE

The offline replay (55 payloads plus the server job: 14 new folds, all the EntE/EntF pair through
"activated 2,3-DHB", 0 gold lost, enterobactin precision restored) is credible for the shape it covers.
None of those payloads contains a Greek-letter pair, so it cannot see B1 or B2. That is why the probe
found what the replay did not. The 1a52fb55 run of the new tests (3 fail, negatives pass) is consistent
with the diff (record; not reproduced).

## MINIMUM CHECKS TO SHIP WITH THE VERIFIER ON (after B1, B2 fixed)

1. Unit: the alpha/beta-carotene fold negative and the beta-alanine addition negative fail on d5d0ec95 and
   pass on the fix; the lane file passes; gates with all flags unset stay green.
2. Offline: re-run the section 15 replay (answers2, answers3, answers4g, item-4) on the fixed code. Pass:
   0 gold lost, the same 14 folds, and no addition or rename whose written name has a different
   reference_key from the model proposed name (log both per change). That last check catches every
   collapse of the B2 kind on real data.
3. Live, one server job: re-submit the item-4 paper (PMID 40400396, Gemma, 3 draws). Pass: PWML
   produced; EntE and EntF each appear once; no PWML_IR_DUPLICATE_NAMED_ROW and no AMBIGUOUS_RENAME_TARGET;
   gold R1-R5 at least as many as item-4 attempt 1; lock prune equals omitted plus folded; every name in
   entities_dropped is absent from every field of final_mapped.json.
No new 3x3. The change only fires on a refused rename that has an exact reference_key twin, and the
replay shows that fires only on the EntE/EntF shape in the corpus.

## WHAT I COULD NOT VERIFY

- The gates9 numbers and ut-r3b logs (read from the record).
- How often Greek-letter pairs occur in real Stage-1 output (none in the 18 stored payloads; chlorophyll
  and carotenoid papers are the likely carriers).
