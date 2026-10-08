# ORCH-745 F-1 proposal — species markers on the claim's own supporting sentences

Status: PROPOSAL for bio adjudication before implementation. Fixed by the product owner's handoff:
"test the in-vitro, inference and ortholog markers against the claim's own supporting sentences:
the sentences of its cited spans that name a participant or the enzyme. Not the whole span." The
definition of done also requires: GH3.10 JA -> JA-Ile admitted on its knockout evidence, the
recombinant 13-LOX step kept out, recombinant evidence never admissible, orthologs never transfer
species, no review presented as primary.

Preconditions landing first (change set A): reviews are graded `review` (evidence level
`review_article_only`), and reference-list / front-matter text can never be an evidence span.

## Why "sentence scope" alone is not enough

The pilot-002/003 13-LOX paper (PMC8508710, recombinant LOX2/3/4/6 from E. coli) has NO sentence
naming LeA / 13-HPOT that contains a current in-vitro marker ("in vitro", "cell-free",
"recombinant enzyme assay", "purified enzyme", "reconstituted system"). Abstract sentences such as
"A. thaliana 13-LOXs, in cascade with 12-OPDA pathway enzymes, synthesized 12-OPDA and OCPD from
LeA..." and "The apparent maximal velocity (Vmax app) values for LOX-catalyzed LeA oxidation were
highest for LOX4 (128 nmol s-1 mg protein-1), with a Km value of 5.8 µM" would pass a pure
marker-in-sentence test and the 13-LOX step would be admitted as target-supported. So the
proposal grades the evidence each sentence provides, and requires POSITIVE evidence that the
organism performed the reaction.

## Proposed rule

1. **Cited spans** = every span the claim cites for any field (participants, enzyme, gene,
   species, compartment, direction, claim-level `evidence_ids`).
2. **Supporting sentences** = sentences of those spans that name (case-insensitive, word-boundary;
   variants of "X (Y)" = X, Y, "X (Y)") at least one of the claim's participants, its enzyme or
   its gene.
3. **Each supporting sentence gets one grade**, tested in this order:
   - `INFERENCE`: ortholog / orthologue / homolog of / homologue of / by analogy / by similarity /
     presumably / is expected to / likely catalyzes / extrapolat / inferred from (today's list).
   - `HOST`: transgenic / heterologous(ly) / ectopic(ally) / overexpress* / 35S — never counts as
     the organism performing its own reaction (foreign gene in the target, or target gene elsewhere).
   - `IN_VITRO`: in vitro / cell-free / recombinant / purified / reconstituted / expressed in
     (E. coli | Escherichia coli | yeast | Pichia | insect cells | Sf9) / enzyme assay(s) /
     incubation(s) / incubated with / Km / Vmax / kcat / kinetic(s) / lysate.
   - `IN_PLANTA` (only if none of the above, AND the sentence names at least one SUBSTRATE OR
     PRODUCT of the claim — the enzyme alone is not enough for positive credit): in planta / in vivo
     / mutant(s) / knockout / knock-out / knockdown / loss-of-function / null allele / T-DNA / SALK_
     / wild-type / wild type / Col-0 / seedlings / leaves / flowers / roots / rosette(s) /
     silique(s) / tissue(s) / wounded / wounding / ... lines.
   - `NEUTRAL` otherwise.
4. **Species state** (existing order kept: inference > in vitro > not stated > multiple >
   non-target > target):
   - any `INFERENCE` supporting sentence -> `inferred_only`;
   - no `IN_PLANTA` sentence and any `IN_VITRO` sentence -> `in_vitro_or_cell_free`;
   - species empty -> `species_not_stated` (paper-level screen organisms are NOT used as the claim's
     species any more; today they produce `multiple_species` from e.g. "human LOX homologs");
   - several species -> `multiple_species`; one non-target species -> `non_target_species`;
   - one species = the target, model-cited species evidence (not the verbatim fallback), AND the
     cited species span actually names the target species, AND:
       * an `IN_PLANTA` supporting sentence exists -> `target_species_supported`, grade `in_planta`;
       * else the paper is a review and a `NEUTRAL` supporting sentence names both the target
         organism and a participant -> `target_species_supported`, grade `review`
         (evidence level `review_article_only`, never presented as primary);
       * else -> `species_ambiguous` ("the target is named but no supporting sentence shows the
         organism performing the reaction"): a lead, never admitted;
   - no supporting sentence at all -> `species_ambiguous`.
5. **Same-paper merge (F-6)**: when the same paper yields the same claim identity in two batches,
   the graded supporting sentences and species evidence are unioned and the state recomputed with
   the rule above, so the result never depends on batch order.

## Worked expectations (orchestrator's reading; please confirm or correct)

| case | supporting sentences (grade) | proposed state |
|---|---|---|
| pilot-003 #14 JA -> JA-Ile (GH3.10), abstract ev_7d2787cbb14b4905 | "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit." (IN_PLANTA); "...recombinant GH3.10 ... in vitro enzyme assays" (IN_VITRO); "...formed in Arabidopsis thaliana by JAR1 and GH3.10" (NEUTRAL) | target_species_supported, in_planta |
| pilot-003 #31 JA + Ile + ATP -> JA-Ile (GH3.10), methods spans | "GH3.10 and JAR1 were purified ... in vitro incubations with ... JA ... ATP and Ile" (IN_VITRO) | in_vitro_or_cell_free |
| pilot-002 primary / pilot-003 #24/#25 LeA -> 13-HPOT (13-LOX, PMC8508710) | "Vmax ... LeA oxidation ... Km" (IN_VITRO); "A. thaliana 13-LOXs, in cascade with ... enzymes, synthesized 12-OPDA ... from LeA" (NEUTRAL) | in_vitro_or_cell_free (not admitted) |
| pilot-003 #34 OPDA -> OPC-8:0 (OPR3, review PMC13364333) | "In Arabidopsis, OPDA and dn-cis-OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0, respectively." (NEUTRAL, names target + participants) | target_species_supported, review_article_only |
| pilot-003 #33 HDA -> HTA (MpFAD7, Marchantia) | table row without "ortholog" in the row sentence | non_target_species (no longer an inferred_only artefact) |
| pilot-003 #12/#13/#15 JA -> JA-Gln / JA-Asn / JA-Glu | "Additional JA conjugates found in wounded leaves (JA-Gln) or formed in flowers upon MeJA treatment in the absence of JA-Ile (JA-Gln, JA-Asn, JA-Glu) were identified." (IN_PLANTA) | target_species_supported, in_planta — PLEASE ADJUDICATE |
