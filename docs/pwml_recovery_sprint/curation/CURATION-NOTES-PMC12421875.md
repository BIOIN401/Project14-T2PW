# Curation notes — PMC12421875

**Requested pathway:** menaquinone biosynthesis · **Requested organism:** *Bacillus subtilis*
**Title:** *The growth benefits and toxicity of quinone biosynthesis are balanced by a dual regulatory mechanism and substrate limitations* (mBio, 2025)
**Source:** `data/rag_index/acquire_cache/fulltext/600dc32d336465b1efd48f241563f6fb.json`, `full_text` = 62,985 chars
**Result:** 9 core reactions · 4 major subprocesses · 12 important participants · **0 secondary participants** · 16 out-of-scope

---

## What the paper delivers

The **complete classical menaquinone pathway**, chorismate to MK, with every enzyme pinned to a
substrate–product pair:

```
chorismate ⇌MenF⇌ isochorismate --MenD--> SEPHCHC --MenH--> SHCHC --MenC--> OSB
    --MenE--> OSB-CoA --MenB--> DHNA-CoA --MenI--> DHNA --MenA (+prenyl-PP)--> DMK --MenG--> MK
```

This is by far the most complete of my three menaquinone papers. The paper is really a *regulation*
study — a picomolar DHNA biosensor, nisin-tunable MenC/MenF/MenD expression, and a two-model
steady-state kinetic analysis — but it lays the whole pathway out first, and then its model pins the
intermediate steps.

## How the middle five reactions are actually asserted — and why they are `medium`

R1, R2, R8 and R9 each come from a prose sentence and are `high` (R9 has its own problem, below).

**R3–R7 (MenH, MenC, MenE, MenB, MenI) have no prose sentence at all.** Their substrate–product
pairs are asserted through the **kinetic model**:

1. the enzyme order is fixed by "…a seven-enzyme pathway consisting of MenF, MenD, MenH, MenC,
   MenE, MenB, and MenI to produce DHNA";
2. the letters are fixed by "A for chorismate, B for isochorismate, C for SEPHCHC, D for SHCHC,
   E for OSB, F for OSB-CoA, and G for DHNA-CoA";
3. the steady-state ODE system (equations 5–11) then has each enzyme consuming one letter and
   producing the next — e.g. `d[C]/dt` has a MenD production term minus a MenH consumption term,
   and `d[D]/dt` has a MenH production term minus a MenC consumption term.

The pairings are unambiguous. But the brief requires a **contiguous** quote and forbids stitching
one from two places, and no single sentence says "MenH converts SEPHCHC to SHCHC". So I marked
R3–R7 `medium` and explained the derivation in each `rationale`. This is a confidence about
*quotability*, not about the biology.

## Source error: "MenG demethylates DMK to generate MK"

The paper's only mention of MenG says it **demethylates** DMK. Demethylating *de*methylmenaquinone
cannot yield menaquinone.

I recorded the substrate/product pair the paper gives — **DMK → MK**, which is unambiguous — kept
the verbatim quote intact, dropped R9 to `medium`, and put the problem in the `description`, the
`rationale` and `uncertainties`. I did **not** substitute the verb I believe is correct. R9 must not
be read as endorsing "demethylation".

This is worth flagging across the dataset because **PMC12657337 asserts the opposite chemistry for
the same step**: "BsUbiE encodes a methyltransferase that catalyzes the methylation of
demethylmenaquinone to produce MK-7". Two papers in the same benchmark, same reaction, opposite
verbs. Each file records its own paper.

## `secondary_participants` is empty, and that is a finding

I searched for the obvious cofactors. Case-sensitive counts over the full text:

- `ThDP` **0** · `thiamine` **0** · `coenzyme A` **0**
- `CoA` occurs **twice**, and both are inside the metabolite names `OSB-CoA` and `DHNA-CoA`
- no `ATP`, no `SAM`, no `2-oxoglutarate`, no free water or proton in any reaction

So **R5 (MenE, a CoA ligase) has no CoA and no ATP recorded, and R9 (a methyl transfer) has no
methyl donor** — because the paper supplies none. The empty list is a property of the source, not an
oversight. Note the contrast with the neighbouring MenD paper (PMC12312563), where ThDP is the
highest-frequency non-protein token in the text.

## DHNA is recorded twice, on purpose

DHNA is the product of R7 and a substrate of R8 (`defining_substrate_or_product`), **and** the
non-competitive allosteric inhibitor of MenD in R2, with its own `Ki` in the kinetic model
(`central_to_pathway_scope`). That dual role *is* the paper's central finding. Per §4 of the brief I
recorded it in both roles with the reaction ids that make each true, and flagged it in
`uncertainties` so nobody deduplicates it into a single global verdict.

## The entry I nearly classified the other way

**S4, "allosteric feedback inhibition of MenD by DHNA".** By the reasoning I applied to PMC12444477
— where I excluded the entire YejM–LapB–FtsH module because regulation is not a chemical stage — I
should arguably have excluded this too.

I included it because this paper gives the inhibition **its own kinetic term inside the pathway
model** and treats it as a constituent mechanism of DHNA biosynthesis rather than as an external
regulator acting on an enzyme's abundance. It is marked `medium` and flagged in `uncertainties`; a
reviewer who drops S4 for cross-dataset consistency has a fair case, and nothing else depends on it.

## What this paper does NOT have — guard against cross-import

**This paper never names a specific isoprenoid chain length.** The MenA cosubstrate is only
"prenyl diphosphate"; the products are only "DMK" and "MK". There is no `MK-7`, no `DMK-7`, no
`HepPP`, no `FPP` and no `HepPPS` anywhere.

That is precisely what its neighbour PMC12657337 *does* have. Any MK-7, DMK-7, heptaprenyl
pyrophosphate or farnesyl pyrophosphate attributed to PMC12421875 has been carried across from the
other paper. Conversely, PMC12657337 has none of the MenF→MenI chain that this paper enumerates in
full. The two are near-complements, and that is exactly the confusion this dataset exists to catch.

## Organism trap

Requested: ***B. subtilis***. Actual: ***Lactococcus lactis*** subsp. *lactis* KF147, with
*Lactiplantibacillus plantarum* NCIMB8826 as the DHNA biosensor. *B. subtilis* appears only in
cited-literature framing about prior engineering attempts. No reaction in the file is asserted to
occur in *B. subtilis*.

A related organism subtlety worth recording: the paper says MenD is "the first step solely committed
to DHNA biosynthesis" **in *E. coli***, then states "we consider MenF as the first step in
*L. lactis* for DHNA biosynthesis" because *L. lactis* lacks siderophore biosynthesis. The
committed-step designation is organism-dependent here, and in the organism actually studied it is
**R1, not R2**.

Also: every kinetic constant in Tables 1 and 2 is a **literature** parameter from *E. coli* or
*M. tuberculosis*. Not one was measured in the requested organism.

## The construct trap

`MenFD` / `MenDF` are labels for the co-expression cassette carrying *menF* plus *menD*. The paper
uses them dozens of times in enzyme-like phrasing ("perturbations of MenFD", "increased MenFD
levels"). They are genetic constructs, not enzymes, and are in `out_of_scope`. So are the
single-letter metabolite symbols A–G.

## Gold vs. paper

**No disagreement.** Gold's `expected_enzymes` are exactly MenF, MenD, MenH, MenC, MenE, MenB, MenI
and MenA — matching R1–R8. Gold places **MenG** under `acceptable_enzymes`; I recorded it as a core
reaction (R9) because the paper does give it a substrate and a product. That is a slightly more
inclusive reading of the same sentence, not a conflict, and it is noted in `uncertainties`.

## Verification

All 25 quotes in `expected_core_PMC12421875.json` were confirmed by substring search against
`full_text` before the file was written. **None failed.**
