# ORCH-746 change set E — repair round 1 (of at most 2): subtract-only, reviewer-prototyped

Round-0 review of `5eca087b` (full text: `C:/t/bt/orch746/reviews/cse_r0.txt`): code REQUEST_CHANGES (1 blocking),
bio-auditor SOUND_WITH_FINDINGS (fit for pilot YES), red team UNSOUND (2 blocking; fit YES after its vetoes). Every
reviewer confirmed E's subset property (credit at `5eca087b` ⊆ credit at `686ae134`) and that both ORCH-745 blockers
are closed for every probe listed. The three blocking findings are routes inside the same two classes E was opened
for, each with a reviewer prototype that only removes credit and keeps the suite at 915 and the replay at
#3/#14/#34.

Same rules as round 0 (`C:/t/bt/orch746/briefs/CS-E.md` and `COMMON_RULES.md`), same worktree `C:/t/m745-b`, same
branch, same scratch `C:/t/bt/m746-e/` (use fresh run subdirectories). **Commits on top of `5eca087b`. The repair base
for G9 is `5eca087b`.** Allowed files unchanged: `src/t2pw/mining/species.py`, `tests/test_mining_species_cs_e.py`
(append; do not edit its existing rows), `docs/pathway_mining/species_policy.md`, `known_limitations.md`,
`testing.md`. Commit subject style: `experiment: ORCH-746 CS-E repair 1 <id> - <what>`.

## The binding property, unchanged

**Every change in this round only removes credit.** After your change, every sentence that earns `in_planta` must
also earn it at `5eca087b` (and so at `686ae134`). Every item below is either a reading appended to the
all-must-agree `_READINGS`, a veto added to `_NO_EFFECT` (used only as a veto, species.py:999), or a narrowing of a
credit form or of `_MARKER_OF`. Nothing else. No recall gain of any kind.

## Required (the three blocking findings)

1. **CSE-R0-1 (code reviewer).** All three readings keep a participant that stands right before a comma-led opener
   when a ", " follows the opener, so an appositive pair or an inner pair makes an OPEN clause look closed: "Leaves of
   cyp94b3 mutants hyperaccumulated JA-Ile, with 12-OH-JA-Ile, the major catabolite, reduced by 90%." admits the
   reversed 12-OH-JA-Ile -> JA-Ile. Fix: APPEND a fourth reading to `_READINGS`, exactly the reviewer's prototype
   `C:/t/bt/rev746e2-code/plugins/rev746e2_r4.py` (`_excluded_unambiguous_close`: the `686ae134` reading, plus, with
   parentheticals blanked, a participant is cut off when two or more ", " follow the opener before the form). Leave the
   three existing readings untouched. Rows (not TSS, each must fail at `5eca087b`): S1-S8 of
   `C:/t/bt/rev746e2-code/scripts/adjacent.py` (outputs `runs/adj_tip/out.txt`).
2. **RT-E2-1 (red team).** A negated delay written with any absence form other than "did not / failed to" is credited
   as an absence ("In wounded leaves of jar1 mutants, JA-Ile was not detected until 30 min after wounding."). Fix: add
   to `_NO_EFFECT` the prototype's delay vetoes from `C:/t/bt/rev746e2-redteam/plugin/rt746e_veto.py`:
   `\buntil\b` anywhere in the sentence, and `\b(?:before|prior to)\s+\d`. Rows: D21-D26, D8 and the other D rows of
   `C:/t/bt/rev746e2-redteam/runs/hunt1_tip` / `hunt2_tip` that are accepted there (scripts under
   `C:/t/bt/rev746e2-redteam/scripts/`), and the four regression-fixture rows of the review (cse_r0.txt RT-E2-1).
3. **RT-E2-2 (red team).** No-difference results earn credit: non-significance ("the difference was not statistically
   significant", "n.s.", "P > 0.05", "P = 0.08", "non-significant", "tended to") and the genotype joined to the wild
   type ("below the detection limit in jar1 mutant and wild-type plants"). Fix: add the prototype's remaining vetoes
   from `rt746e_veto.py` verbatim (the non-significance entries, the "<genotype> and/or <wild type>" entry, `alike`).
   Rows: S11-S17 and N11-N15 (and hunt1 S1-S10, N1-N8 where accepted at `5eca087b`), plus the review's four
   regression-fixture rows.

Add the vetoes as entries in the existing `_any(...)` call of `_NO_EFFECT` (match how the module builds it; the
prototype recompiled the joined pattern, which you should not copy). Each needs a short comment naming its finding.

## Subtract-only closures of documented residuals (do each; DROP any one that makes an existing test fail, and report it)

4. **RT-E2-4 concessive veto** (red team's measured `rt746e_veto2.py`): `\b(?:although|though|despite|in spite of)\b`
   in `_NO_EFFECT`. Rows: C1/I4 "Despite high OPC-8:0 levels in opr3 mutants, JA was reduced.", C3, C7 "Even though jar1
   mutants accumulated JA-Ile, JA-responsive gene induction was reduced.", C8.
5. **RT-E2-3 the "accumulated ... less" gap.** The CS-D repair-1 form `accumulated (\w+ ){0,2}?(less|lower|...)` reads a
   participant inside its gap as the decreased one ("opr3 mutants accumulated OPDA with lower OPC-8:0 levels." credits
   only the reversed OPC-8:0 -> OPDA). Make the form earn nothing when a claim participant starts inside its gap. Choose
   the smaller construction (for example, in `_effect_under`, skip a match of that form whose gap holds a participant
   hit). Rows: R11, R12, R13. Controls: "aos mutants accumulated less 12-OPDA", "accumulated much less" stay TSS.
6. **CSE-R0-N1 an opener written without its comma.** APPEND a fifth reading: the `686ae134` `_excluded` in which a
   bare " with " (not preceded by a comma) also counts as a clause opener. Rows: X3r "In cyp94b3 mutants, JA-Ile
   accumulated after wounding with reduced 12-OH-JA-Ile.", X5r, X4, I1.
7. **CSE-R0-N5** add `nowhere` to the gap-negator lookahead of the "accumulated ... less" form (species.py:457). Row: G3.
8. **BIO-D-R2-3 / RT-D-R2-3 the regulator marker tail.** `_MARKER_OF`'s up-to-three gap tokens may not match
   `regulat\w*|upstream|downstream|genes?|factors?|activators?|repressors?|defects?|targets?`. Rows: K8 "Mutants
   lacking a regulator of AOS accumulated no 12-OPDA.", K9 "Mutants with defects upstream of JAR1 accumulated no
   JA-Ile.", M3 "Mutants for regulators of JAR1 accumulated no JA-Ile.". Controls: the SALK_034543 row, "T-DNA insertion
   lines of AOS", "Mutants deficient in GH3.10", "knockout of GH3.10" stay TSS.

## Not in this round (stay documented residuals)

RT-D-R2-2 (the precursor of X, which is Y), a comparison word more than 60 characters after "did not" (CW1),
"accumulated only marginally / slightly less" (a product-owner policy item, not a defect).

## Tests (append to `tests/test_mining_species_cs_e.py`)

- Regression rows `test_cs_e_r1_746_<item>_*`, each asserting NOT `target_species_supported`, each failing at
  `5eca087b` (write them first and run them before changing `species.py`; list any that do NOT fail there and leave them
  out). At least one row per required item also goes end to end through `ClaimStore` + `admission.judge`.
- Controls `test_new_acceptance_cs_e_r1_746_*` passing at both ends, as each item lists.
- G9 manifest: `C:/t/bt/m746-e/g9_manifest_cs_e_repair1.json`.
- Re-run your probe matrix with these added trees/probe sets: `5eca087b` and your tip, over all 223 earlier probes
  plus every round-0 reviewer probe (code `scripts/adjacent.py` and its matrix rows; red team `scripts/hunt*.py`; bio
  `C:/t/bt/rev746e2-bio/scripts/` where present). The tip may hold TSS only where `5eca087b` does. Report every row
  that LOSES credit (that is the recall cost; each must be listed in `known_limitations.md`).
- Full mining suite at your tip, with `final_surviving_count`.

## Docs

- `species_policy.md` rule 5(b): the readings now in `_READINGS` and what each refuses; every new `_NO_EFFECT` entry;
  the gap rule for "accumulated ... less"; the `_MARKER_OF` exclusion list.
- `known_limitations.md`: move every class closed here out of the residual list and into what the gate refuses;
  BIO-E746-5 (a semicolon inside a figure or table reference, "(Figure 2B; Table 1)", still ends the comparison window,
  and comparison words outside the list, "different", "below", are not seen); BIO-E746-6 (replace "One correct row that
  686ae134 credited is lost" with the X1c row plus a participant followed by a noun such as "levels", "content",
  "accumulation" before a comma-led relative clause, e.g. "In opr3 mutants, OPC-8:0 levels, which peak 1 h after
  wounding, were strongly reduced."); RT-E2-6 (the precision figures were re-measured by the red team at `5eca087b`:
  30 / 22 (1 correct) / 20 / 2 (1 correct), identical to `686ae134`); CSE-R0-N6 / RT-E2-5 (name "accumulated only
  marginally / slightly / somewhat less" beside "slightly lower" as credited weak decreases, a product-owner policy
  item); the recall cost this round measures.
- `testing.md`: the new row counts and suite total.

## Final report

As round 0's, plus: per item 1-8, done or dropped (with the failing test if dropped); the G9 rows that fail at
`5eca087b`; the matrix summary (TSS per tree, violations, the credit lost); the suite count with
`final_surviving_count`; and the shapes you still suspect.
