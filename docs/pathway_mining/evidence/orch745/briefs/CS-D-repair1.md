# ORCH-745 CS-D — repair round 1 (of at most 2)

Review of `6454ef98` (full text: `C:/t/bt/orch745/reviews/csd_r0.txt`): code REQUEST_CHANGES (CSD-B1), bio UNSOUND
(BIO-D-1, BIO-D-2), red team UNSOUND (RT-D-1). All three agree CS-D closed every probe it was opened for and the
T-rows/replay hold; the blockers are narrow. Same rules, same worktree/branch; commits on top of `6454ef98`;
tests proving corrections must FAIL at `6454ef98` (say which ALSO fail at `fd7c737c`) — list them in
`C:/t/bt/m745-b/g9_manifest_cs_d_repair1.json`. Subjects `experiment: ORCH-745 CS-D repair 1 <item> - ...`.

## Blocking

**R1 — comparative / delay no-effect after a negated-verb form (CSD-B1, BIO-D-1, RT-D-1; a weakening vs
fd7c737c).** "The jar1 mutant did not accumulate (significantly) less/more JA-Ile than the wild type",
"did not produce less OPC-8:0", "failed to accumulate more JA-Ile", "produced/contained no less JA-Ile",
"did not accumulate JA-Ile further / until 6 h", "accumulated no more/greater JA-Ile than" are credited today.
Add to `_NO_EFFECT` (sentence- or clause-level, your call, tested):
- `\bno (?:more|less|greater|fewer|higher|lower|lesser)\b`;
- `\b(?:(?:did|does|do) not|fail(?:ed|s)? to)\b[^.;:]{0,60}?\b(?:more|less|lesser|lower|higher|fewer|greater|further|again|than|until|significantly)\b`;
- and drop the unbriefed forward forms `produced no` / `formed no` / `contained no` (keep `accumulated no X` only if
  it stays behind the veto; it was in the bio-auditor's A7 control "accumulated no JA-Ile", which must stay
  credited).
Do NOT add a bare `than` veto: "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves." must stay
credited. Fix the docs' claim that the negated-verb forms "cannot admit a no-effect sentence".

**R2 — polarity attribution (BIO-D-2).**
(i) A substrate-side increase form ("accumulated", "increased", "elevated", "higher") credits only when the
clause carries NO decrease/absence word ("less", "fewer", "lesser", "lower", "reduced", "decreased", "no", any
absence form). Optional recall gain: `accumulated (\w+ ){0,2}(less|lower|reduced|decreased)` as a PRODUCT-side
decrease form ("aos mutants accumulated less 12-OPDA" credits the 13-HPOT -> 12-OPDA claim).
(ii) When the effect's subject candidates include BOTH a substrate and a product of the claim, refuse
(ambiguous) — and take the NEAREST preceding participant rather than the first (RT-D-2's appositive case:
"In opr3 mutants, the OPDA metabolite OPC-8:0 accumulated" must not credit as a substrate increase).
(iii) Cap the "after the form" fallback: a participant more than 3 tokens after the effect form is not its
subject; and veto a phenotype subject in the clause: growth, fertility, resistance, susceptibility, tolerance,
senescence, yield, biomass, germination, flowering, development, phenotype (RT-D-2: "Root growth of the jar1
mutant was reduced on medium containing JA-Ile" and the herbivore/fertility/senescence/yield rows must not credit).
Probes that must not be TSS: "cyp94b3 mutants accumulated less 12-OH-JA-Ile after wounding." (12-OH-JA-Ile ->
JA-Ile, CYP94B3, i.e. the reversed reaction); "The jar1 mutant accumulated less JA than the wild type after
wounding."; "In opr3 mutants, OPDA accumulated to lower levels after wounding."; "In gh3.10 mutants, the JA
conjugate JA-Ile accumulated after wounding."; "In the jar1 mutant, the JA-Ile precursor JA was reduced after
wounding."; RT-D-2's P1/P2/P4/P5/P6. Controls that must stay TSS/in_planta: T1, "No JA-Ile was detected in gh3.10
jar1 knockout lines.", bio-r-1, n1-r1, P1 "In opr3 mutants, OPDA accumulated in wounded leaves.", Pc1 "The opr3
mutant accumulated higher levels of OPDA than Col-0 after wounding.", Pc2 "jar1 mutant plants accumulated lower
levels of JA-Ile after wounding.", "gh3.10 knockout lines accumulated no detectable JA-Ile.", "The aos mutant did
not accumulate 12-OPDA after wounding.", "jar1 mutants failed to accumulate JA-Ile in wounded leaves.", "In jar1
mutants, JA-Ile was no longer detected after wounding.", "JA-Ile was 20-fold lower in jar1 mutants than in
wild-type leaves.", the bio-auditor's A1–A10 (`C:/t/bt/rev745b-r2-bio/runs/r6/out.txt`), the replay (#3, #14,
#34) — list any control that moves, with the reason.

## Non-blocking (do these; small)
- **RT-D-4:** `_MARKER_OF`'s `in` tail must not bind through a container noun: "Mutants in the JAR1 pathway
  accumulated no JA-Ile." must not credit (refuse when the token after the gene is pathway/background/cluster/
  family/signalling/signaling/network). Keep "T-DNA insertion lines of GH3.10" and "Mutants deficient in GH3.10"
  credited. Allow up to 3 tokens between the marker and `of` so "The SALK_034543 insertion line of GH3.10
  accumulated no JA-Ile." credits again (a recall loss CS-D introduced); pin both rows.
- **Docs:** `species_policy.md` / `known_limitations.md`: state exactly how the effect's subject is chosen (after
  R2), the `_MARKER_OF` shape, the regulator-named-as-enzyme residual (RT-D-5: "Knockdown of MYC2 reduced
  wound-induced JA-Ile" when extraction names MYC2 as the enzyme), the double-mutant residual, the unlisted-genus
  magnitude (RT-D-3: 10/10 unlisted plant binomials credited; only instrumentation, a per-paper principal
  organism, can close it), and the measured corpus precision from the CS-D red team (29/31 credited without the
  screen; 2 with it — one T1-class correct, one dapat pleiotropy extraction case).

Run the full mining suite at your tip. Final report: per item, T-rows, replay, suite count, manifest, every
moved control.
