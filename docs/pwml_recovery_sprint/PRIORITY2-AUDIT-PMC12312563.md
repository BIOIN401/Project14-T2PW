# Priority-2 completeness audit -- PMC12312563

**Independent biological adjudication by `pwml-bio-auditor`, 2026-09-03, wave `ORCH-722`.**
Commissioned by the Lead under **D-087 clause 3** (independent verification) and adopted as the
basis of **D-091**. The auditor is read-only by construction: it edited nothing, ran no test, and
chose the case itself from the ten-case table without being told which would pass most easily.

**Preserved because the audit IS the cost.** The gold edit it authorizes is two lines; this document
is why those two lines are defensible, and a successor re-reading D-091 needs the reasoning, not the
conclusion.

**Lead verification.** Every load-bearing count below was independently re-measured before D-091 was
written -- MenD 133, the eight other Men symbols 0 each, SEPHCHC 9, isochorismate 9, 2-oxoglutarate
8, SHCHC/OSB/prenyl/pyruvate 0, and free `chorismate` with word boundaries **0**. All reproduce. The
auditor also corrected the Lead's own briefing: the paper text is `01_source_text.txt`, not
`00_PAPER.txt`, which is a 738-byte run header.

---

## 1. The case chosen, and the six rejected

**`PMC12312563`** — *Structures of Listeria monocytogenes MenD in ThDP-bound and in-crystallo
captured intermediate I-bound forms* (Acta Cryst F, 2025).

Chosen because its scope is bounded by **the paper's own enzyme roster**, not by a judgement about
where a pathway ends. Of the nine classical Men enzymes, **eight have zero occurrences in the entire
file** and MenD has 133.

| Case | Rejected because |
|---|---|
| `PMC12096016` (enterobactin, 3 sigs) | Two further reactions stated with named catalyst and both sides, unsignatured: EntA `2,3-diDHB -> 2,3-DHB` (*"(iii) EntA (2,3-dihydro-2,3-dihydroxybenzoate dehydrogenase; EC 1.3.1.28)"*) and EntE adenylation (*"2,3-DHB is adenylated by EntE ... in an ATP-dependent reaction"*). Gold's own `notes` concede the EntE exclusion. |
| `PMC12421875` (menaquinone, 3 sigs) | Gold declares **eight** expected enzymes and `min_connected_reactions` of **7**. Three signatures against a seven-reaction floor is self-evidently a subset. |
| `PMC12657337` (MK-7, 3 sigs) | Two further conversions stated with both sides named, present only in `acceptable_enzymes`: *"The enzyme Idi, which converts IPP to farnesyl pyrophosphate (FPP)"* and *"acetyl-CoA synthetase (acs), an enzyme that converts acetate into acetyl-CoA"*. |
| `PMC12444477` (lipid A, 2 sigs) | Eight expected Raetz enzymes, two signatures. Rejected on sight. |
| `PMC12452463` | Standing position: route chemically broken, EntA absent. The EntF trilactone assembly is also stated and unsignatured. |
| `PMC12782028` (cholesterol, 3 sigs, `strict_exportable`) | The MSMO1 catalytic statement is real and unsignatured — *"It primarily catalyzes the three-step monooxygenation required for the demethylation of 4,4-dimethyl and 4alpha-methylsterols"*. Gold's `notes` say the floor is 2 "not 4" because MSMO1's substrate is a compound class. |
| `PMC12856317` (ALAS2, 1 sig) — **the serious rival** | Rejected: the paper states **three protease-catalysed transformations by catalysts other than ALAS2** — *"the interaction between heme and mitochondrial ALAS1 results in either LONP1- or CLPXP-mediated degradation"*, *"heme-bound ALAS2 recruits the CLPXP protease via an adaptor protein to trigger degradation"*. Gold does not merely tolerate these: it **blesses LONP1 and CLPXP under `acceptable_enzymes` quoting those exact sentences**. Declaring the one-signature list exhaustive would put the gold case at war with itself. |

`PMC13231680` and `PMC12180156` are `context_only` negative controls and already carry ceilings —
not what is needed.

## 2. Declared pathway scope

The enzyme-catalysed transformations of **free metabolites** belonging to the **classical
menaquinone-biosynthesis pathway**, as stated in the cached full text, in **Listeria monocytogenes**
(strain 10403s).

The paper locates itself at exactly one point: *"MenD catalyses the first irreversible step of the
classical menaquinone-biosynthesis pathway"*. It supplies no step before and none after. Upstream
boundary: 2-oxoglutarate and isochorismate as consumed substrates. Downstream: SEPHCHC as released
product. The pathway is *named* four times and *populated* only at this node.

**Explicitly out of scope:** covalent ThDP-bound adducts (*"via two covalent ThDP intermediates"*) —
states of the enzyme–cofactor complex, not free species; the ThDP ylide and ThDP regeneration —
cofactor activation; DHNA — a downstream allosteric inhibitor, never on either side; rTEV/TEV tag
cleavage, buffers, PDB codes, expression host — laboratory tooling; non-*L. monocytogenes* orthologs
(Mtb, Sau, Bsu, Eco MenD) — comparators introducing no new chemistry.

The boundary is **the paper's, not the auditor's**: the enzyme was kinetically characterised against
exactly three species — *"the Km values for ThDP, 2-oxoglutarate and isochorismate and kcat were
determined"* — and nothing else in the text is given a catalyst, a substrate and a product.

## 3. Reaction inventory, with verbatim spans

Read from `runs_verify/2026-09-02_2052/papers/PMC12312563/01_source_text.txt` (47,976 chars).

**R1 — the only reaction. MenD: 2-oxoglutarate + isochorismate → SEPHCHC (ThDP-dependent, irreversible).**

> "An important player in menaquinone production is the thiamine diphosphate (ThDP)-dependent
> decarboxylase MenD [2-succinyl-5-enolpyruvyl-6-hydroxy-3-cyclohexene-1-carboxylate (SEPHCHC)
> synthase]... **MenD catalyses the first irreversible step of the classical menaquinone-biosynthesis
> pathway (Supplementary Fig. S1) via two covalent ThDP intermediates; decarboxylation of
> 2-oxoglutarate produces intermediate I, with subsequent addition of isochorismate generating
> intermediate II and breakdown to release SEPHCHC** (Fig. 1)."

Corroborated three further times:

> "The ThDP-dependent enzyme ... (MenD) catalyses the first irreversible step in bacterial classical
> menaquinone biosynthesis via a series of reactions involving covalent ThDP-bound intermediates."

> "the SEPHCHC enzyme activity of Lmo MenD was kinetically characterized with respect to the ThDP
> cofactor and the two substrates 2-oxoglutarate and isochorismate"

> Figure 1 legend: "...**resulting ultimately in regeneration of the ThDP and release of the product
> SEPHCHC**."

**There is no R2, established positively rather than by absence of gaps:**

1. **Exhaustive enzyme-suffix sweep of the body** (everything before `References`). Every `-ase`
   token is `decarboxylase`, `synthase`, `protease`, `polymerase`, `genase` (from *transhydrogenase
   III domain*), plus the non-words `case`, `decrease`, `release`. The first two are MenD's own class
   names; `protease` is the inhibitor cocktail; `polymerase` is the qPCR instrument. **MenD is the
   only enzyme in the body.**
2. **Case-sensitive Men-symbol counts, whole file:** MenD 133; **MenA/B/C/E/F/G/H/I all 0.**
3. **Metabolite counts:** SEPHCHC 9, isochorismate 9, 2-oxoglutarate 8; **free chorismate 0** (all
   nine hits lie inside *isochorismate*), **SHCHC 0, OSB 0, OSB-CoA 0, prenyl 0, polyprenyl 0,
   pyruvate 0, CO2 0.**
4. **Transformation-verb sweep** over the whole file, reference authors filtered. Every hit belongs
   to R1, R1's ThDP mechanism, DHNA inhibition, or `rTEV cleavage` of the purification tag.

### Considered and deliberately excluded

| Candidate | Span | Excluded because |
|---|---|---|
| `2-oxoglutarate → intermediate I` | "decarboxylation of 2-oxoglutarate produces intermediate I" | Covalent enzyme-bound ThDP adduct, not a free product. Internal mechanism of R1. |
| `intermediate I + isochorismate → intermediate II` | "Intermediate I then goes on to react via C2α with the second substrate isochorismate" | Same. |
| `intermediate II → SEPHCHC` | "the subsequent breakdown of intermediate II to release product" | Same. |
| pre-decarboxylation adduct | "a transient pre-decarboxylation intermediate forms after the C2 atom of the thiazolium ring of the ThDP ylide reacts with ... 2-oxoglutarate" | Transient, unnamed, enzyme-bound. |
| ThDP ylide formation | "enabling proton abstraction and formation of the activated ThDP ylide" | Cofactor activation; identity-neutral. |
| CO2 release | *(0 occurrences)* | Chemically real, textually absent. Adding it would import chemistry the paper does not state. Harmless to the flag: `_side_matches` is one-directional, so a payload row carrying CO2 still matches. |
| rTEV tag cleavage | "After rTEV cleavage (weight ratio 1:36)" | Purification reagent. **The one residual not closable from gold's existing text.** |
| DHNA inhibition | "a decrease of 34% in enzymatic activity ... in the presence of 12.5 µM DHNA" | Allosteric inhibition, not a transformation; also lands in `interactions`, which `_reaction_buckets()` excludes. |

## 4. Comparison to gold, and verdict

Gold holds **exactly one** signature: `2-oxoglutarate + isochorismate -> SEPHCHC [MenD]`, aliases
`LmoMenD` / `SEPHCHC synthase`, `reversible: false`, with the quote *"decarboxylation of
2-oxoglutarate produces intermediate I, with subsequent addition of isochorismate generating
intermediate II and breakdown to release SEPHCHC"*. The quote survives `fold_for_quote` against the
cached text, so the signature is scored rather than dumped into `unverifiable`.

**Inventory `{R1}` = gold `{R1}`.**

# VERDICT: EXHAUSTIVE.

## 5. The objection a skeptical reviewer would raise, and why it fails

> *The pipeline decomposes R1 into ThDP-adduct sub-steps carrying verbatim paper evidence. Setting
> the flag prints `unsupported_retained_reaction` against rows whose evidence is a literal quotation.
> That is the false accusation D-087 clause 5 exists to prevent.*

Well-founded on the facts. `strict/merged_payload.json` holds four rows — the matching R1 plus
`2-oxoglutarate -> intermediate I`, `intermediate I + isochorismate -> intermediate II`,
`intermediate II -> SEPHCHC`, all `provenance: inferred`. `research/merged_payload.json` holds two,
neither matching.

**It fails on gold's own terms.** Gold already ruled on these rows twice: `forbidden_identifiers`
lists `intermediate I`/`intermediate II` as `kind: placeholder_product`, reason *"Transient covalent
enzyme-bound adducts, not free metabolites with distinct identity"*; and `goldset.py:391-393` defines
that kind as one where *"the extractor did not read a product, **so the reaction is unsupported**"*.
Setting the flag enforces a position the gold set already holds.

Two honest costs, recorded rather than hidden:
- the same row is then counted once under the forbidden-identifier check and once under Priority 2 —
  a scoring-design question, and the product owner's;
- on the `research` leg the signature is unmatched **today, flag or no flag**, because `_side_matches`
  requires all signature inputs and 2-oxoglutarate is absent. The flag adds false positives; it does
  not manufacture the miss.

## 6. Independence, and the anti-motivation statement

The auditor read the full text and ran its own counts **before** consulting the curator's notes.
`curation/expected_core_PMC12312563.json` independently records `expected_core_reactions` of length
**1**, identical to R1, and places `rTEV protease`, `intermediate I`, `intermediate II` and the ThDP
ylide in `out_of_scope`. Two independent readings converge; convergence is corroboration, not the
evidence.

**Setting the flag makes the score WORSE on both committed legs** — strict to 1/4, research to 0/2.
It is recommended solely because the verdict becomes reachable on a non-negative-control paper for
the first time.

## 7. Confidence and what would change the verdict

**High** on the biology — the inventory rests on a complete enzyme-suffix sweep plus zero-counts for
all eight other Men symbols, not on impression. **Moderate** on the operational recommendation:

1. **Supplementary Fig. S1 is not in the cached text** and may depict the full seven-enzyme chain.
   The claim is exhaustiveness over `01_source_text.txt`, which is also exactly what
   `_check_supported_reactions` verifies quotes against. **If any ingestion path pulls the PDF
   supplement, this audit is void.**
2. **The RAG clearance is two-legs-deep.** `accepted: []` in both (57 and 64 candidates, all
   rejected) is a fact about two legs that aborted at `stage1`. A completed leg could admit a
   cross-paper reaction that by construction cannot match a seed-paper signature.
3. **rTEV** is the one supported-but-unsignatured transformation not closable from gold's text.

**Would change the verdict:** any occurrence of a second Men enzyme or of free chorismate / SHCHC /
OSB / OSB-CoA / DHNA-CoA in whatever text the scorer is handed; Supplementary Fig. S1 entering the
corpus; a ruling that a ThDP-adduct sub-step *is* a supported reaction; a completed leg with a
non-empty `accepted`; or `_reaction_buckets()` widening to include `interactions`, which would make
the DHNA row a scored unmatched row.

**Also noted:** the research leg's RAG proposed `ThDP -> intermediate I` sixteen times and the
admission gate **rejected it every time**. The hazard is real and the gate held.

---

# FOLLOW-UP ADJUDICATION, same auditor, same day — THE RECOMMENDATION IS REVERSED

**The Lead routed three questions back to the auditor after review forced a measurement of the
flag's effect on the COMMITTED corpus. The auditor reversed its operational recommendation. Its
biology is unchanged and is not retracted.** Adopted as **D-092**.

## Q1 — synonymy: YES, and the paper says so itself

2-oxoglutarate, alpha-ketoglutarate, 2-ketoglutarate, oxoglutaric acid and 2-oxopentanedioate are
one compound (ChEBI:30915 / :16810). C2 is a ketone, not a stereocentre, so there is no isomer to
confuse. The paper makes the equation explicitly: *"the first (alpha-ketoacid) substrate (for MenD
this is 2-oxoglutarate)"*. It writes `2-oxoglutarate` 8 times and `ketoglutarate` **0** times, so
gold was right to use the paper's own token as the term name; the Greek spelling in the payload is
the pipeline's canonicalisation, carrying `rag_provenance.source_id: "seed_paper"`.

**Measured, not assumed** — `GoldTerm(name="2-oxoglutarate")` with no aliases: `2-oxoglutarate`
exact; bare `oxoglutarate` matches by containment; **`alpha-ketoglutarate`, `2-ketoglutarate`,
`2-oxoglutaric acid`, `oxoglutaric acid`, `2-oxopentanedioate`, `AKG`, `alpha-KG` all MATCH_NONE.**
The Greek letter is already folded by `_GREEK`, so only the ASCII form is needed.

**Do NOT add bare `ketoglutarate` or `glutarate`** — `goldset.py` warns `alpha-` and `beta-` must not
collapse. **The same gap exists on `expected_substrates[0]` and on the `expected_pathway_anchors`
entry**; fixing only the signature would leave the coverage checks holed.

## Q2 — row [6] is NOT a fabrication, and it is the case AGAINST the flag

`MenI` 0, `DHNA-CoA` 0, `thioesterase` 0, `LMRG` 0 in this paper — that stands. But *unsupported by
the seed paper* and *fabricated* are different claims. All four participating entities carry
`rag_provenance.source_id = "PMC8091085"`, `rag_confidence = 0.865`:
*"Listeria monocytogenes MenI Encodes a DHNA-CoA Thioesterase Necessary for Menaquinone
Biosynthesis…"* — **which PMC12312563 cites in its own reference list** (Smith et al., 2021,
Infect. Immun. 89, e00792-20). The chemistry, enzyme, organism and pathway are all correct.

This is textbook clause (c) of the flag's own docstring: *"a legitimate cross-paper addition from
RAG synthesis, which cannot match a seed-paper signature by construction."*

**Attribution limit, stated precisely:** the lineage is on the ENTITIES, not on the reaction row.
Row [6] carries no `rag_provenance`. **A scorer cannot exclude RAG-derived rows from the precision
denominator from the row alone.** That is the missing instrumentation — `R-D092-1`.

## Q3 — WITHHOLD. Withdraw the flag; ship the aliases alone.

Four configurations, measured read-only against the committed canonical leg
(`runs/2026-07-27_1623/…/strict`, `RESULT: PASS`, `pwml_export`, nothing quarantined):

| aliases | flag | ok | TP | FP | recall | verdict evaluated |
|---|---|---|---|---|---|---|
| no | off | **False** | 0 | withheld | **0/1** | no |
| no | on | False | 0 | **7** | 0/1 | yes |
| **yes** | **off** | **True** | 2 | withheld | **1/1** | no |
| yes | on | False | 2 | **5** | 1/1 | yes |

**(a)** The alias gap is a pre-existing defect that fails this check TODAY with the flag off.
**(b)** Aliases do not rescue the flag — row [6] is still charged, and that is structural.
**(c)** The flag's seed-only precondition is violated on committed evidence.

**The auditor's own stated trigger fired.** Its original CONFIDENCE section said the RAG clearance
was "two-legs-deep, not general" and named "a completed leg whose `rag_admission_report.json` shows
a non-empty `accepted`" as what would change its mind. Its words on the reversal:

> *"I generalised from two aborted legs, and I should have said the clearance was worthless rather
> than merely 'two-legs-deep.' That is the error, and it is mine."*

## Incidental defect found while measuring — `R-D092-2`

Rows **[0] and [5] are duplicates**, identical but for evidence-span length. `true_positives` counts
pointers, so they contribute **2 TP for 1 reaction** and `matched_by` confirms it. **Priority 2
currently rewards a duplicated row.**
