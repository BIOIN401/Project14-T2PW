# ORCH-745 CS-D — repair round 2 (the LAST allowed round): four one-line fixes, nothing else

Re-review of `fd7ea63a` (full text `C:/t/bt/orch745/reviews/csd_r1.txt`): code REQUEST_CHANGES, red team UNSOUND,
bio UNSOUND. Everything round 1 was asked to close is closed; four narrow routes remain, each with a fix the
reviewers prototyped as a scratch pytest plugin that keeps the suite at 810 passed. **Implement exactly these four,
nothing beyond them** (no new heuristics — the one unbriefed heuristic last round, `_NEW_SUBJECT`, is what reopened
a route). Same rules, same worktree/branch; commits on top of `fd7ea63a`; every new row must FAIL at `fd7ea63a`
(say which also fail at `6454ef98`/`fd7c737c`) — `C:/t/bt/m745-b/g9_manifest_cs_d_repair2.json`.

1. **Relative clause subject (CSD-R1-B1, RT-D-R1-1, BIO-D-R1-1).** `_NEW_SUBJECT` drops the main-clause head X in
   "X, which … Y …, <effect>" and credits Y, inverting polarity (product up / substrate down admitted; the
   reversed cyp94b3 reaction admitted). Replace it with the red team's semantics: a participant that sits INSIDE a
   comma-led relative/adverbial clause that CLOSES before the form is excluded; the head X before the clause stays a
   candidate. Prototype: `C:/t/bt/rev745d-r1-redteam/plugin/rt745dr1_fix.py` `_excluded()` (read it; reproduce the
   logic in `species._effect`). If X and Y are on opposite sides after that, the existing both-sides refusal applies.
   Rows (not TSS): the four RT-D-R1-1 regression fixtures (cyp94b3 reversed, gh3.10 product up, jar1 substrate
   down, opr3 substrate down). Recall rows (TSS/in_planta): the cyp94b3 correct direction; "In opr3 mutants,
   OPC-8:0, which is formed from OPDA, was undetectable."; "In jar1 mutants, JA-Ile, which is the bioactive form, was
   undetectable."; X19/X20 stay refused.
2. **Negators inside the "accumulated … less" gap (CSD-R1-B2).** `\baccumulated (?:(?!(?:not|no|hardly|scarcely|
   barely|never)\b)\w+ ){0,2}?(?:less|lower|reduced|decreased)\b`. Rows (not TSS): H1–H3, F1. Control H4 "accumulated
   much less", Pc2, "aos mutants accumulated less 12-OPDA" stay TSS.
3. **`_MARKER_OF` over-binding (RT-D-R1-2).** The up-to-three gap tokens may not be `of`/`in`:
   `\s*(?:(?!(?:of|in)\b)[\w-]+\s+){0,3}?(?:of|in)\s+(?:the\s+)?`. Rows (not TSS): "Mutants in genes upstream of JAR1
   accumulated no JA-Ile.", "Knockouts of transcriptional targets of JAR1 accumulated no JA-Ile.", "Mutants in a
   regulator of JAR1 accumulated less JA-Ile after wounding." Controls: the SALK_034543 row, "T-DNA insertion lines
   of GH3.10", "Mutants deficient in GH3.10".
4. **Comparative window stops at "OPC-8:0" / "gh3.10" (BIO-D-R1-2; the brief's own pattern).** In the R1
   `_NO_EFFECT` entry replace `[^.;:]{0,60}?` with `(?:[^.;:]|[.:](?=\d)){0,60}?`. Rows (not TSS): CL1–CL5, NX3.
   Controls (TSS): "The opr3 mutant did not accumulate OPC-8:0 after wounding.", "JA-Ile did not accumulate in gh3.10
   jar1 double mutants after wounding.", "In flowers of gh3.10 jar1 knockout lines, JA-Ile did not accumulate.",
   B1–B6, Nc1, Nc2, "20-fold lower … than".

Docs: describe the four exactly in `species_policy.md`; correct the `known_limitations.md` sentence that claims the
comparative class is closed (it is now, with the window fix), and document the relative-clause rule. T-rows, the
replay (#3, #14, #34) and every earlier control must hold. Run the full mining suite; final report as before.
