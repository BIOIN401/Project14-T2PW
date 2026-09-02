# Curation notes — PMC12312563

**Requested pathway:** menaquinone biosynthesis · **Requested organism:** *Bacillus subtilis*
**Title:** *Structures of Listeria monocytogenes MenD in ThDP-bound and in-crystallo captured intermediate I-bound forms* (Acta Cryst F, 2025)
**Source:** `data/rag_index/acquire_cache/fulltext/08a2f791273034853db7224f8bb3fb67.json`, `full_text` = 47,406 chars
**Result:** **1 core reaction** · 3 major subprocesses (1 `detailed_in_paper: false`) · 4 important · 2 secondary · 13 out-of-scope

---

## One reaction, in enormous depth

This is a structural biology communication: two crystal structures of *Listeria monocytogenes* MenD
(PDB `9e9b`, ThDP-bound; `9mnn`, intermediate I-bound), plus SEC, mass photometry, SAXS, DSF and
UV/Vis kinetics.

It delivers **exactly one** menaquinone reaction, and delivers it completely:

> "MenD catalyses the first irreversible step of the classical menaquinone-biosynthesis pathway
> (Supplementary Fig. S1) via two covalent ThDP intermediates; decarboxylation of 2-oxoglutarate
> produces intermediate I, with subsequent addition of isochorismate generating intermediate II and
> breakdown to release SEPHCHC"

`2-oxoglutarate + isochorismate → SEPHCHC`, ThDP-dependent. Both substrates, the product, the
catalyst, the cofactor dependence and the irreversibility are in one sentence, and the enzyme was
kinetically characterised against exactly those three species.

## What is NOT here — measured, not assumed

I counted case-sensitively over the full text before writing anything:

| Symbol | Count | | Metabolite | Count |
|---|---|---|---|---|
| `MenD` | **133** | | `SEPHCHC` | 9 |
| `MenF` | **0** | | `isochorismate` | 9 |
| `MenH` | **0** | | `2-oxoglutarate` | 8 |
| `MenC` | **0** | | free `chorismate` | **0** |
| `MenE` | **0** | | `SHCHC` | **0** |
| `MenB` | **0** | | `OSB` | **0** |
| `MenI` | **0** | | `CO2` | **0** |
| `MenA`, `MenG` | **0** | | | |

The nine apparent `chorismate` hits are all inside the word *isochorismate*. **Nothing upstream or
downstream of MenD is given an enzyme, a substrate or a product.** So `S1` — the "classical
menaquinone-biosynthesis pathway" — is recorded with `detailed_in_paper: false`: the pathway is
named repeatedly and MenD is located within it as its first irreversible step, but no other step
exists in this text.

This is the sharpest contrast in my three menaquinone papers. **PMC12421875 enumerates all seven
enzymes of the chorismate→DHNA chain; this paper has one of them and none of the others.** Any
MenF/MenH/MenC/MenE/MenB/MenI/MenA/MenG attributed to PMC12312563 has been imported from that
neighbouring paper or from the ortholog literature.

## Intermediates I and II are NOT products

The paper's headline result is the in-crystallo capture of intermediate I. I deliberately did **not**
record intermediate I or intermediate II as products or metabolites. They are transient **covalent
enzyme-bound ThDP adducts** — states of the enzyme–cofactor complex, not free species. Treating them
as metabolites would manufacture a fake three-step chain out of a single reaction, which is exactly
the failure mode a one-reaction paper invites. Both are in `out_of_scope` with that reason.

## The call I am least sure of: ThDP as *important*, not *secondary*

Thiamine diphosphate is a cofactor, so the default classification is `secondary`. **I put it in
`important_participants`** with reason `distinguishes_identity_or_direction`, because here it is not
an ordinary cofactor:

- both catalytic intermediates are **covalent ThDP adducts**;
- one of the two reported structures *is* the ThDP-bound form;
- ThDP is one of the three species the enzyme was kinetically characterised against (a `Km` was
  measured for it);
- the paper's own naming is "the ThDP-dependent enzyme … MenD".

A consumer whose rule is "all cofactors are secondary" should move it, and that is flagged in
`uncertainties`.

**This does not conflict with gold.** Gold lists ThDP under `forbidden_identifiers` with
`kind: cofactor_as_protein` — it forbids emitting ThDP *as a protein*, which is a different claim
and one I agree with emphatically. `ThDP` is the highest-frequency non-protein token in the paper
(66 occurrences) and sits grammatically adjacent to the enzyme name; emitting it as a protein is the
prototypical failure this paper exists to surface. My entry marks it as a chemically important
*participant*, never as a catalyst.

## CO2 is absent because the paper never names it

R1 is described as a decarboxylation, so carbon dioxide is released. But `CO2` and `carbon dioxide`
each occur **zero** times. I did not add it. Adding it would be importing chemistry the paper does
not state — the same discipline that kept the empty result honest in PMC13231680.

## DHNA: inhibitor, never a substrate or product

DHNA gets a results section here, but it is the pathway's **downstream** metabolite acting as an
**allosteric inhibitor** at the TH3 domain. Placing it on either side of the MenD reaction inverts
the paper. It is recorded in `secondary_participants` with `class: "regulator"` and an explicit
warning in its `note`.

I also kept the authors' hedging intact. The measured effect in *L. monocytogenes* is **weak**: 66%
residual activity at 12.5 µM, no attainable IC50 (DHNA solubility), and attempts at a DHNA-bound
structure failed. The potent IC50 values quoted in the discussion (53 nM for *Mtb* MenD; 2.3/3.7 µM
for *Sau* MenD) are **other organisms, other studies**.

**Note a deliberate cross-file difference:** DHNA is `secondary` here but `important` in my
PMC12421875 file. That is not an inconsistency — in PMC12421875 the paper's entire stated scope
*is* DHNA (`central_to_pathway_scope`), whereas here the scope is the MenD structure. In both files
DHNA is an inhibitor and in neither is it a substrate or product.

## The other borderline entry: S2

`S2`, the MenD catalytic cycle, is the **internal mechanism of R1** rather than a pathway stage that
groups further reactions, so under a strict reading of the brief it may not qualify as a major
subprocess. I included it at `medium` because it is unambiguously what this paper is about, and
omitting it would leave the paper's entire scientific content unrepresented. Nothing else depends
on it.

## Organism trap — the cleanest in the set

Requested: ***B. subtilis***. Actual: ***Listeria monocytogenes*** strain 10403s, with the protein
expressed in *E. coli* BL21 (DE3). **`Bacillus subtilis` occurs exactly once in the entire text**, in
a comparative clause about which MenD orthologs show DHNA inhibition:

> "DHNA inhibition, albeit less potent and without crystallographic capturing, has been reported for
> Staphylococcus aureus and Bacillus subtilis MenD"

No reaction in this file occurs in *B. subtilis*.

## Identity hazards recorded

- `LmoMenD`, `MtbMenD`, `SauMenD`, `BsuMenD`, `EcoMenD` — organism-prefix labels for MenD orthologs
  (the source renders them with a space: "Lmo MenD"). One enzyme plus one organism, not new proteins.
- The source contains a literal typesetting split artifact: **"For Ec oMenD this strained tetrahedral
  form…"**.
- `PYR`, `PP`, `TH3` and "arginine cage" are protein domains and a motif nickname, not proteins.
- `9e9b`, `9mnn`, `3lq1`, `5ery`, `3flm`, `6o0j` are PDB depositions.

## Gold vs. paper

**No disagreement.** Gold's single `expected_enzyme` is MenD with role
"2-oxoglutarate + isochorismate -> SEPHCHC (ThDP-dependent)", matching R1 exactly, and its expected
substrates and products match my important participants. Its `forbidden_identifiers` anticipate the
same traps — intermediate I/II as placeholder products, the ortholog prefixes, the `Ec oMenD`
artifact, DHNA-as-substrate, and the upstream/downstream hallucination test — which I had recorded
independently from the text.

## Verification

All 10 quotes in `expected_core_PMC12312563.json` were confirmed by substring search against
`full_text` before the file was written. **None failed.**
