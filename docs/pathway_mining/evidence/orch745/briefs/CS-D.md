# ORCH-745 change set D — loss-of-function credit: polarity, binding, extracts (narrow, tightening only)

Change set B reached its repair-round cap (2). Its final review at `fd7c737c` (code REQUEST_CHANGES, red team
UNSOUND, bio UNSOUND) found NEW, narrow defects in the loss-of-function rule (5(b)) and one regression the
orchestrator's own repair-2 brief introduced. The orchestrator opened this separate change set for exactly those
items (recorded as a decision for the product owner to revisit). Full findings:
`C:/t/bt/orch745/reviews/csb_r2_final.txt` and `C:/t/bt/orch745/reviews/csb_r2_final_detail.txt` (B-1, B-2,
R3-B1, R3-B2, BIO-R2-1..4). Same rules (`COMMON_RULES.md`), same worktree `C:/t/m745-b`, same branch; commits on
top of `fd7c737c`, subjects `experiment: ORCH-745 CS-D <item> - ...`. **Everything here only removes credit,
except D1's negated-verb absence forms (a whitelist recall gain that cannot admit a no-effect, hedged or negated
sentence).** T-rows, T1b, the full pilot-003 replay (exactly #3, #14, #34; #14 in_planta; #3/#34
review_article_only) and every earlier regression row must hold.

## D1 — the effect must be a loss-of-function consequence on the RIGHT participant (B-1, R3-B1, BIO-R2-1, BIO-R2-2)

In `species._effect` / the in-planta clause loop, make the effect participant- and polarity-aware:
- **Product side:** an ABSENCE or DECREASE form credits the claim only when the strictly-named participant it
  applies to in the clause is a PRODUCT of the claim. Absence forms: not detected/detectable, no detectable,
  undetectable, below the (detection|quantification) limit, absent, abolished, abrogated, eliminated, depleted,
  "no <product> was detected/found/formed", and (new, recall) "did not / failed to / does not (accumulate|form|
  produce)", "no longer (detected|accumulated|formed)"; "lacked/lacking/deficient" only when bound to the product
  ("lacked JA-Ile", "deficient in JA-Ile") — never when they describe the genotype ("mutants lacking GH3.10").
  Decrease forms: reduced, decreased, lower, diminished, impaired, compromised.
- **Substrate side:** an ACCUMULATION or INCREASE form (accumulated, increased, elevated, higher) credits only when
  the participant it applies to is a SUBSTRATE of the claim ("In opr3 mutants, OPDA accumulated" supports OPDA ->
  OPC-8:0). The contrary combinations (product up; substrate down) earn nothing.
- **Negation guard on absence forms too:** an absence/decrease/increase form preceded in the clause by a negation
  (not|no|n't|never|neither|nor|without|none) earns nothing — EXCEPT the absence forms whose negation is internal
  and starts the match ("not detected", "no detectable", "no <product> was detected"), and the new negated-verb
  forms above, which ARE the absence statement. "was not abolished", "not completely abolished", "never
  abolished", "not deficient in" earn nothing.
- **No-effect veto (restore + widen):** put bare `comparable to` back (the adjudicated rule 5(b) list has it; the
  repair-2 brief's "comparable to wild type" narrowing was the orchestrator's error = a weakening), plus
  indistinguishable, normally, "still (present|accumulated|detected|formed)", residual, "not (fully|completely|
  entirely)? (abolished|absent|eliminated|abrogated|depleted|deficient|lacking)", "reduced but not abolished" is
  allowed to stay creditable ONLY via its "reduced" clause if your clause logic reads it that way — otherwise
  leave it out (fail closed) and list it as a recall cost.

## D2 — the participant and the genotype must be the claim's own (R3-B2)

- Before the participant test, blank the claim's catalyst-name spans (enzyme, gene, protein) in the clause, so an
  enzyme named after its substrate ("allene oxide cyclase", "OPDA reductase 3", "sterol methyltransferase") can
  never supply the participant.
- Bind the loss-of-function marker to the claim's own enzyme/gene: the gene/enzyme symbol must sit immediately
  before the marker (within ~4 tokens: "gh3.10 jar1 knockout lines", "jar1 mutant", "opr3 knockout line",
  "aos mutants", "the jar1 gh3.10 double mutant") or right after "mutant(s)/knockout/knockdown/T-DNA insertion of/
  in"; a different gene's mutant ("arc mutants", "the myc2 mutant ... by JAR1") earns nothing.
- Refuse credit when the effect's subject is a protein or transcript measure (protein abundance/levels/amount,
  transcript, mRNA, expression, activity of the enzyme).
- The verbatim corpus sentence "Moreover, protein abundance of the JA biosynthetic enzyme allene oxide cyclase was
  increased in arc mutants." (PMC13358708, `tests/fixtures/mining/pilot001_screened_papers.json`) with the AOC claim
  (allene oxide -> 12-OPDA) must not be TSS.

## D3 — extracts are cell-free (B-2; the R1-B3 family)

Add `\bextracts?\b`, `\benzyme preparations?\b`, `\bcrude enzyme\b` (and the round-1 pattern
`(?:enzyme|crude|protein|cell|leaf|tissue)\s+(?:[\w-]+\s+)?(?:extracts?|preparations?|fractions?|homogenates?)`)
to `_IN_VITRO`. Recall cost (an in-planta metabolite measured "in leaf extracts" earns nothing) is accepted and
documented — species_policy already calls any extract cell-free.

## D4 — plant genera the organism words miss (BIO-R2-3)

Extend `_ORGANISM_WORDS` with the plant genera a jasmonate/oxylipin corpus hits (brachypodium, camelina, setaria,
lotus, gossypium, petunia, catharanthus, salvia, mentha, artemisia, panicum, populus, medicago, glycine, hordeum,
brassica, capsicum, physcomitrium, physcomitrella, selaginella, and any of solanum/oryza/zea/nicotiana/triticum/
vitis not already present) and `Bd|Sv|Pa|Ptr|Mtr` style prefixes to `_FOREIGN_GENE` where unambiguous; keep both
documented ceilings. Qualify `known_limitations.md`: "a paraphrase that slips a veto can only cost a lead" is true
for hedge/negation/comparison residue and false for an unrecognised organism name.

## D5 — docs
`species_policy.md` (the effect rule exactly as shipped), `known_limitations.md` (recall costs incl. BIO-R2-4
E1–E4, BIO-R2-5 verb-first review prose, BIO-R2-6 union blast radius; R3-N1..N5; the measured fact that the
paper-level organism screen held back 34 of 37 real creditable loss-of-function sentences in the red team's corpus
run). Also correct the CS-A wording in `known_limitations.md` (~line 197): it is an **orchestrator interim
position pending the product owner**, not a "product-owner decision" (no DECISIONS.md entry exists).

## Tests and proof
- Every probe the final reviewers demonstrated (B-1 list, B-2 IV1–IV3, R3-B1 A1–A3 and the negated-absence
  family, R3-B2 B1/B2 and its parametrisation incl. the verbatim arc sentence, BIO-R2-1 N1/N3–N6, BIO-R2-2
  P2–P5, BIO-R2-3 F1/F5) is a `test_cs_d_745_*` row asserting NOT target_species_supported; they must FAIL at
  `fd7c737c` (they are admitted there) — list them in `C:/t/bt/m745-b/g9_manifest_cs_d.json` with
  `fails_at_fd7c737c`. BIO-R2-1 rows also pass at fd53a301 (a regression guard) — say so.
- Controls that MUST stay TSS/in_planta: T1; "No JA-Ile was detected in gh3.10 jar1 knockout lines."; bio-r-1
  "In opr3 mutant leaves, OPC-8:0 was not detected."; n1-r1 "In the opr3 knockout line, OPC-8:0 was undetectable
  in leaves."; P1 "In opr3 mutants, OPDA accumulated in wounded leaves." (substrate side); the helpers_mining
  abstracts; and the canonical A. thaliana phrasings the bio-auditor listed as A1–A10 in
  `C:/t/bt/rev745b-r2-bio/runs/r6/out.txt` (read it; keep every one that is genuinely product-down or substrate-up
  credited, and list any that move with the reason). New recall: "The aos mutant did not accumulate 12-OPDA after
  wounding." and "jar1 mutants failed to accumulate JA-Ile in wounded leaves." become TSS/in_planta (move bio-r-2
  out of the 5a_removed list; record the move).

Run the full mining suite at your tip. Final report: per item what changed, the T-row table, the replay, suite
count, manifest path, every moved test with its exact delta.
