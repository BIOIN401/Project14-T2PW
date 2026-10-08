# ORCH-745 F-1 — pwml-bio-auditor adjudication of F1_PROPOSAL.md (binding acceptance spec for CS-B)

Verdict on the proposal as written: **SOUND_WITH_CHANGES, and the changes are blocking.** The core
design is endorsed (grade the claim's own sentences; require POSITIVE evidence the organism performed
the reaction, the enzyme alone earns nothing; drop paper-level screen organisms; require a model-cited
species span that names the target; a separate review grade). As written it would fail two DoD items
(13-LOX kept out only by luck; a review graded as primary), admit 4 rows that must stay out, and get
2 worked rows wrong. The auditor simulated the rule over all 37 pilot-003 claims + the pilot-002
primary + 7 counterfactuals. **Only the T-rows of section 3 bind; the regex suggestions are guidance —
an equivalent mechanism that produces the T-row states is acceptable.**

Sentence refs `ev_x[i]` = sentence i of `acquisition._SENTENCE.split(span.text)` on the stored text.
Papers: LOX = PMC8508710 (recombinant 13-LOX, primary); BRY = PMC13364333 (bryophyte review);
GH3 = PMC12096059 (primary); PEP = PMC12787732 (pepper). IVO = in_vitro_or_cell_free;
TSS = target_species_supported.

## 1a. Weaknesses of the proposal (all reproduced in simulation)

- **W1 recombinant 13-LOX comes back once the model fills species.** Tissue nouns and "in vivo"
  count as positive markers; 13-LOX full-text sentences get IN_PLANTA: ev_4dcc372e77b84976[3]
  "...injection of plant tissue extract ... in LeA" (pea-root ex vivo extract); ev_baf1514923696f5c[5]
  "[57], who observed that Cd2+ increased the content of 13-HPOT in A. thaliana ... in vivo" (other
  people's work); ev_baf1[9] "This might imply ... pathways in vivo" (hedged); ev_baf1[10] "The results
  support the view that all 13-LOXs potentially channel 13-HPOT ... as anticipated for LOX6 in leaves
  and roots ... wounded leaves [17, 20, 60]" (hedged, cited); ev_bb4c9802d1637278[26] "Ex Vivo
  Evaluation ... we employed pea roots as a plant LOX source", and [27]. IN_PLANTA outranks IN_VITRO,
  so with species "Arabidopsis thaliana" cited to the abstract (what pilot-002's model did) #11 and #24
  become TSS/in_planta.
- **W2 sentence scope instead of clause scope (#30).** ev_1f2c624a181ba5ef[17] "12-hydroxy jasmonate is
  directly converted into 12-hydroxy jasmonoyl-isoleucine and fertile jasmonoyl-isoleucine deficient
  Arabidopsis mutants point towards ... in flowers." The markers sit in a clause about a different
  subject; the conversion was shown with purified recombinant GH3.10 (ev_1f2c[9],[12];
  ev_6893aa984a651e1f[13]).
- **W3 occurrence / absence counted as conversion.** ev_7d2787cbb14b4905[9] "Additional JA conjugates
  found in wounded leaves (JA-Gln) or formed in flowers upon MeJA treatment in the absence of JA-Ile
  (...)" admits #12/#13/#15 and gives #14 a spurious credit. Intro ev_d216d8e2a43cb218[39] "We introduce
  the aos gh3.10 jar1 mutant as a new tool ... in the absence of JA-Ile" admits an intro-only JA->JA-Ile.
- **W4 names match inside longer compound names.** "-" as a word break: "OPDA" matches in "dn-OPDA",
  "JA" in "JA-Ile"/"12-hydroxy-JA". #34 (OPDA -> OPC-8:0, review) then gets IN_PLANTA from Marchantia
  sentences ev_c98d84c7d84855c1[27] "Genetic knockout of Mp FAD5 nearly abolishes dn-OPDA accumulation
  after mechanical wounding" and [30] "...Mp fad5 mutant plants ... in Marchantia" => a review presented
  as primary.
- **W5 wrong species inside the in-planta sentence.** Organism only checked in the species span;
  Marchantia sentences and the pea extract give target credit; a tomato opr3 sentence in a paper that
  also names Arabidopsis would too.
- **W6 review route too loose.** "names the target and any participant" admits "In Arabidopsis, OPDA
  accumulates..." for any OPDA-consuming claim and contrastive sentences ("Unlike Arabidopsis,
  Marchantia converts..."). A review paper must never get grade in_planta.
- **W7 marker gaps.** HOST misses: expressing, expression of, expressed in (tobacco | N. benthamiana |
  ...), transient(ly), agroinfiltrat*, infiltrated, protoplasts, transformed. (Pilot-001 pattern
  "Arabidopsis plants expressing a bryophyte desaturase converted LA to GLA in leaves" must not be
  in_planta for A. thaliana.) IN_VITRO misses: ex vivo, crude / tissue / protein extract, homogenate,
  microsom*, isolated chloroplasts, and spaced "K m", "V max", "k cat" (ev_6915f534ca2dfd74[10]
  "(K m app = 1.2 ± 0.4 µM)").
- **W8 methods / plant-material sentences** bite (ev_bb4c[26], ev_4dcc[3]); GH3.10 ev_c7a1b8c0ccdfe997[26]
  "Arabidopsis plants (Col-0 ... aos gh3.10 jar1) were sprayed with the jasmonates" and ev_6893[11] are
  harmless only by luck. Exogenously applied product ("leaves treated with JA-Ile") would credit any
  JA-Ile-forming claim.
- **W9 background statements** in a primary paper (citation-bearing / introduction) earn credit.
- **W10 hedges** (may, might, potentially, support the view, anticipated, suggest, likely, putative).
- **W11 cofactors and free amino acids** (ATP, Ile in #31/#32) make buffer-recipe sentences supporting:
  harmless for negative markers, unsafe for positive credit.
- **W12 aggregation gaps.** The model's `species_inferred=true` (`claimed_inferred`) must still force
  `inferred_only`. HOST needs a rule: never IN_PLANTA; HOST-only with the target species gives
  `species_ambiguous` with a host reason.
- **W13 the species-span check is satisfied by comparison mentions** ("Arabidopsis Ortholog" table
  header ev_20a7facf5fd73dd9[8]; "MpFAD5 is orthologous to AtFAD5, an Arabidopsis enzyme..."
  ev_c98d[28]). At least one sentence of the species span that names the target must be free of
  INFERENCE, HOST and contrast markers.
- **W14 CS-A dependency.** body#0 JATS front matter must be stripped through the post-abstract
  teaser/key-points line and keyword/funding/pmc-status block, or treated as an A4 duplicate of
  abstract#0 after hyphen normalisation. W2's clause fix is still needed (such sentences occur in body).

## 1b. Correct evidence still held back (accepted unless noted)
- Name brittleness: "jasmonic acid" vs "JA"; "12-hydroxy jasmonate" vs "12-OH-JA"; "9,11,15" vs
  "9, 11, 15"; U+2010 hyphens; "dn- cis -OPDA" italic spacing; α vs alpha. Safe recall fix: normalise
  hyphens, whitespace and Greek letters; optionally a per-paper alias map from "Long name (ABBR)".
- INFERENCE vetoes at sentence level (one comparative ortholog sentence vetoes genuine target
  evidence; ev_c98d[28] would veto a correct AtFAD5 review statement). Fail-closed; keep.
- Negative markers about other organisms veto target statements (#10's ev_74d8b3d259ca310a[11] "In
  Arabidopsis, JA biosynthesis is initiated by 13-LOX enzymes that oxygenate ALA ..." vetoed by [20]
  "Recombinant enzyme assays ... ALA ..." about Marchantia LOXs). Keep. Do not exempt in-vitro
  sentences naming another organism: "AtLOX2 expressed in E. coli" must still veto.
- The 1c exclusions hold back occurrence-only, hedged, introduction, methods, native overexpression and
  complementation statements: they become `species_ambiguous` leads with a `missing_species_evidence`
  frontier item.
- Label artefacts that stay non-admissible (no admission consequence): header glued to row 1 (#29
  `inferred_only` instead of `non_target`); LOX rows mentioning ARA/EPA (#19, #22, #23 `in_vitro`).

## 1c. Corrected IN_PLANTA definition (positive credit only shrinks; negative markers keep loose matching)

A supporting sentence earns IN_PLANTA only if ALL hold:
1. The paper is not a review (reviews use only the review route, item 7).
2. No INFERENCE, HOST or IN_VITRO marker (lists extended per W7).
3. None of: a hedge; a citation (author-year or [n]); a procedural verb (`were|was grown / sown /
   harvested / sprayed / treated / extracted / analysed / quantified / measured / incubated / used`,
   `we employed / used`); a purpose clause (to study, to test, as a (new|novel) tool); a span section
   of methods or introduction.
4. Names no non-target organism. If its span, or the paper's observed organisms, names a non-target
   organism, the sentence must itself name the target (Arabidopsis, A. thaliana, Col-0, or an AGI
   locus). The organism list is a documented ceiling, like `_EXTRA_HOSTS`.
5. Within ONE CLAUSE (split at ; : and the words and / but / whereas / while / or), an in-planta
   context marker occurs with either
   (a) a strictly-named substrate AND product plus a conversion predicate (convert*, conversion,
       formed from, metaboli[sz]ed to, produced from, synthesi[sz]ed from), with no application marker
       (treat*, appl*, spray*, exogenous*, fed, feeding); or
   (b) a loss-of-function marker (mutant, knockout, knock-out, knockdown, loss-of-function, null
       allele, T-DNA, SALK_, RNAi, amiRNA, CRISPR) + the claim's enzyme or gene + a strictly-named
       substrate or product, with no no-effect phrase (unchanged, unaffected, not altered, no
       significant, similar to wild type, comparable to).
6. Strict naming: after normalisation "-" is part of the token, `(?<![\w-])name(?![\w-])`; cofactors
   never count (ATP, ADP, AMP, NAD(P)H, O2, H2O, CO2, CoA, PPi, Pi); free amino acids never count.
7. Review route: paper is a review; the sentence names the target and no other organism; it names a
   substrate and a product, or a participant + the enzyme + a conversion predicate; no contrast or
   negation (unlike, in contrast, lacks, absent, whereas). Result: TSS, grade review,
   `review_article_only`.
8. Order and plumbing: keep the proposal's order (inference > in vitro > not stated > multiple >
   non-target > target); `species_inferred` forces `inferred_only`; the W13 species-span rule applies;
   F-6 unions graded sentences within the same paper only. `build_claims` must know the paper's
   evidence type and each span's section when it classifies; if the paper is reclassified after
   extraction, the state must be recomputed.

Simulation of the corrected rule: of the 37 stored claims + pilot-002, #14 admitted (in_planta, from
ev_7d27[7] only); #3 and #34 admitted (review, from ev_c98d[12]); every other row stays out.
Counterfactuals H1–H3 (13-LOX with species filled), H4 (#12 with substrate "JA"), H7 (ortholog
transfer), H6 (intro-only GH3.10) all held back.

## 2. Expectation table (required state / grade; "admit")

| # | claim_id | equation | paper | decisive sentence(s) | required state / grade | admit |
|---|---|---|---|---|---|---|
| 1 | cl_e78ca4abb54a3fc8 | 16:1(n9)->16:2(n6) MpFAD6 | BRY | ev_20a7facf5fd73dd9[9] "...Inferred by pathway logic; no functional study reported" | inferred_only (Mp); decision hypothesis_only | no |
| 2 | cl_f42d3135f85ff21e | dn-iso-OPDA->dn-OPDA-Glu/Gln/His MpGH3A | BRY | ev_3b1f8607f4b8b229[6] "...Genetic: Mp gh3a abolishes conjugates" | non_target | no |
| 3 | cl_5338b4fc07a47cea | dn-cis-OPDA->OPC-6:0 OPR3 | BRY | ev_c98d84c7d84855c1[12] "In Arabidopsis, OPDA and dn-cis-OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0, respectively." | TSS / review (review_article_only) | **yes, review** |
| 4 | cl_832888dad708cf8e | dn-cis-OPDA->dn-iso-OPDA | BRY | ev_3b1f[6] "...Enzymatic activity in crude extracts [12]; no candidate gene identified" | non_target (Mp; cell-free) | no |
| 5 | cl_381c10905474b7bf | Δ4-dn-cis->Δ4-dn-iso-OPDA | BRY | ev_3b1f[6] crude extracts | non_target | no |
| 6 | cl_87cb48915297c615 | allene oxide->OPDA MpAOC | BRY | ev_3b1f[4],[5] | non_target | no |
| 7 | cl_8ee4bd53fcfb1b9c | allene oxide->(9S,13S)-dn-cis-OPDA MpAOC | BRY | ev_3b1f[4],[5] | non_target | no |
| 8 | cl_2865e3d8c80b6688 | allene oxide->(9S,13S)-OPDA MpAOC | BRY | ev_a82b0ad4450a0bd2[1] "Recombinant MpAOC produces ... in a coupled in vitro assay" | IVO (Mp) | no |
| 9 | cl_1ac4d9ef538231c7 | C18-allene oxide->cis-OPDA PpAOC1 | BRY | ev_a82b[9] | non_target (P. patens) | no |
| 10 | cl_c230261f1437d392 | ALA->13-HPOT | BRY | ev_74d8b3d259ca310a[11] background; [20] recombinant (Mp) | species_not_stated (no species given) | no |
| 11 | cl_7a9937b3db2c395f | LeA->13-HPOT | LOX | ev_baf1514923696f5c[8] "...when incubated with LA or LeA" | IVO | no |
| 12 | cl_5fc57dcfa32aaf35 | jasmonic acid->JA-Gln | GH3 | ev_7d2787cbb14b4905[9] (occurrence) | species_ambiguous | no |
| 13 | cl_2d06a621ba1c2fac | jasmonic acid->JA-Asn | GH3 | ev_7d27[9] | species_ambiguous | no |
| 14 | cl_b89be46f92d4c4e1 | jasmonic acid->JA-Ile GH3.10 | GH3 | ev_7d27[7] "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit."; [11] "This study found JAR1 and GH3.10 are the only contributors to JA-Ile biosynthesis in Arabidopsis"; IVT [5],[8] | TSS / in_planta | **yes** |
| 15 | cl_9ba999bda7a136f3 | jasmonic acid->JA-Glu | GH3 | ev_7d27[9] | species_ambiguous | no |
| 16 | cl_9fa8df5b9e2bb0a4 | LA->GLA MpDES6 | BRY | ev_20a7[12] heterologous expression | non_target | no |
| 17 | cl_7c0b9007ba1f8ec8 | ALA->SDA MpDES6 | BRY | ev_20a7[12] | non_target | no |
| 18 | cl_559992a1bf688e15 | LA->ALA MpFAD7 | BRY | ev_20a7[10] | non_target | no |
| 19 | cl_28107dbbe1153e90 | DGLA->ARA MpDES5 | BRY | ev_20a7[15],[16] | non_target | no |
| 20 | cl_06de0d61ec6ebd01 | SDA->ETA MpELO1 | BRY | ev_20a7[14] | non_target | no |
| 21 | cl_183ebc7ca093105f | GLA->DGLA MpELO1 | BRY | ev_20a7[14] | non_target | no |
| 22 | cl_a453942d6bd35571 | ETA->EPA MpDES5 | BRY | ev_20a7[15],[16] | non_target | no |
| 23 | cl_b5e2f8082d97ede6 | ARA->EPA MpFAD3 | BRY | ev_20a7[11] | non_target | no |
| 24 | cl_fcc1f538bbd36817 | LeA->13-HPOT+12-OPDA (collapsed cascade) | LOX | ev_bb4c9802d1637278[15] "...with 13-LOX-, AOS- and AOC-lysates" | IVO | no |
| 25 | cl_a32b77c9538b6f39 | LeA->13-HPOT | LOX | ev_5863efff93736770[4] "Vmax ... Km value of 5.8 µM" | IVO | no |
| 26 | cl_70223dd072f56b56 | galactolipids->α-linolenic acid | PEP | ev_f7c56801adf6792d[3] cited background | species_not_stated | no |
| 27 | cl_31a6adf4b22136d2 | 12-hydroxy-JA->12-hydroxy-JA conjugate GH3.10 | GH3 | ev_7d27[8] recombinant | IVO | no |
| 28 | cl_0e664d8a3907d58b | C20-allene oxide->11-OPTA PpAOC2 | BRY | ev_a82b[10] | non_target | no |
| 29 | cl_5c72ac211f2bf29a | 16:0->16:1n9 MpFAD5 | BRY | ev_20a7[8] | non_target | no |
| 30 | cl_dee44fedcb6c01eb | 12-hydroxy jasmonate->12-hydroxy JA-Ile GH3.10 | GH3 | ev_1f2c[9],[12] recombinant; [17] | IVO (species_ambiguous if ev_1f2c fully excluded) | no |
| 31 | cl_150891597632f0c5 | JA+Ile+ATP->JA-Ile GH3.10 | GH3 | ev_c7a1b8c0ccdfe997[7] purified, in vitro incubations | IVO | no |
| 32 | cl_061c152eb99e1656 | 12-OH-JA+Ile+ATP->12-OH-JA-Ile GH3.10 | GH3 | same as #31 | IVO | no |
| 33 | cl_5592a5f6c48d97b2 | HDA->HTA MpFAD7 | BRY | ev_20a7[10] | non_target | no |
| 34 | cl_0bc93d248b7f31db | OPDA->OPC-8:0 OPR3 | BRY | ev_c98d[12]; NO credit from [27],[30] (Marchantia) | TSS / review (review_article_only), never in_planta | **yes, review** |
| 35 | cl_bbe6e30f3c568977 | HDA->HTA | BRY | ev_ba1b08899518f20e[10],[11] | species_not_stated | no |
| inc | cl_3acaf053a4653cf7 | ? -> JA | PEP | ev_b48f934b61eb21a4[1] | incomplete; species_not_stated | no |
| inc | cl_1311193317769ac2 | ? -> JA-Ile | PEP | ev_b48f[1] | incomplete | no |
| p002 | cl_9ac4b6db39b411c0 (pilot-002) | LeA->13-HPOT 13-LOX | LOX | ev_b58e2cb64e2f1d87[4] Vmax/Km | IVO | no |

Adjudications: JA -> JA-Gln/Asn/Glu = NOT_ESTABLISHED (occurrence; fed compound is MeJA; no enzyme) —
**product-owner hinge**: if in-planta formation from a supplied precursor counts, they flip, but never
via the "in the absence of JA-Ile" mention. OPR3 rows = CORRECT_TARGET_SPECIES at review grade, never
primary. GH3.10: #14 admit in_planta on [7]; #27, #30, #31, #32 IVO.

## 3. Mandatory acceptance tests (the binding part)

Must admit:
- **T1** #14 cl_b89be46f92d4c4e1, span ev_7d2787cbb14b4905, species model-cited to it ->
  TSS, grade in_planta, evidence level NOT review_article_only. Sentence [7].
- **T1b** the same claim with sentence [7] removed from the span text -> NOT TSS (credit must not come
  from [9] "...in the absence of JA-Ile..." or [2] "gh3.10 jar1 knockout lines are still fertile").
- **T2** #34 cl_0bc93d248b7f31db and **T3** #3 cl_5338b4fc07a47cea, span ev_c98d84c7d84855c1, paper
  PMC13364333 typed `review` -> TSS with `review_article_only`, grade NOT in_planta. [27] and [30] give
  no credit.

Must not admit:
- **T4** pilot-002 cl_9ac4b6db39b411c0, span ev_b58e2cb64e2f1d87 -> `in_vitro_or_cell_free`.
- **T5** #11 cl_7a9937b3db2c395f with species "Arabidopsis thaliana" model-cited to
  ev_5863efff93736770 -> not TSS (expected IVO from ev_baf1514923696f5c[8]); no credit from
  ev_4dcc372e77b84976[3], ev_baf1[5], ev_baf1[9], ev_baf1[10].
- **T6** #24 cl_fcc1f538bbd36817 with species model-cited to ev_6915f534ca2dfd74 -> not TSS (IVO from
  ev_bb4c9802d1637278[15]); no credit from ev_bb4c[26], [27].
- **T7** #30 cl_dee44fedcb6c01eb, ev_1f2c624a181ba5ef reduced to its text from "Abstract" onward,
  U+2010 normalised -> not TSS (IVO). Sentence [17].
- **T8** #31 cl_150891597632f0c5 -> IVO from ev_c7a1b8c0ccdfe997[7]; no reference-list text from
  ev_ae0d92db16547d2c may be evidence.
- **T9** #12/#13/#15 -> `species_ambiguous` (parametrised; PO hinge). Sentence ev_7d27[9].
- **T10** #29 cl_5c72ac211f2bf29a with species "Arabidopsis thaliana" model-cited to
  ev_c98d84c7d84855c1 -> `inferred_only` ("MpFAD5 is orthologous to AtFAD5, an Arabidopsis enzyme...").
- **T11** #14 citing only ev_d216d8e2a43cb218 (introduction), species cited to it -> not TSS
  (`species_ambiguous`). Sentence [39] "We introduce the aos gh3.10 jar1 mutant as a new tool...".
- Recommended: #33 cl_5592a5f6c48d97b2 -> `non_target_species`.

G9: T1–T3 fail behaviourally at f9556288 (stored states in_vitro / inferred_only) -> valid G9 proofs.
T4–T11 pass at base -> label them new acceptance tests (they guard against the proposal's failures).
