# Curation notes — PMC12856317

*"A reversible feedback mechanism regulating mitochondrial heme synthesis"* (Chitrakar et al., JBC 2026).
Requested pathway: **heme biosynthesis**, *Homo sapiens*.
Source: `data/rag_index/acquire_cache/fulltext/62b107e1dde8ae3e1f23f854438ac589.json`, 54,436 chars.

## What the paper delivers of the requested pathway

One reaction. That is the whole of it.

The paper asserts, once, in the Introduction, that ALAS is "a PLP-dependent homodimer enzyme that
mediates the condensation of glycine and succinyl-CoA to produce aminolevulinic acid", and it
identifies that reaction as "the first and rate-limiting step of this essential metabolic pathway".
Everything else in the paper is about *regulation* of that enzyme: heme binds mature ALAS2 with
K_d ≈ 230 nM, inhibits it reversibly as a mixed inhibitor (IC50 18.7 µM), and an AlphaFold3 model
suggests heme bridges HRM3 and HRM6 to lock the C-terminal extension over the active site.

So: **1 core reaction, 2 major subprocesses** (one of which is regulatory and marked `low`).

## Subprocesses named but not detailed

Effectively none — and that itself is the finding. The paper does not name a single downstream
stage of heme biosynthesis. There is no porphobilinogen, no hydroxymethylbilane, no
uroporphyrinogen III, no protoporphyrin IX, no iron insertion. The only two other heme-pathway
enzyme names anywhere in the file (coproporphyrinogen III oxidase, protoporphyrinogen oxidase)
occur solely inside reference titles, past the end of the body.

The one near-miss is this sentence:

> "the mitochondrial matrix, which is the cellular compartment where heme biosynthesis is initiated
> and terminated"

which *implies* a terminal step without naming it, its enzyme, its substrate or its product. I did
not create a `detailed_in_paper: false` subprocess for it, because a subprocess entry needs a name
the paper supplies and this one supplies none. A reviewer who disagrees would add exactly one entry,
"termination of heme biosynthesis in the mitochondrial matrix", with no reaction ids. I flagged the
choice in `uncertainties` rather than making it silently.

## What surprised me / near-miss classifications

- **The regulatory subprocess (S2).** Heme feedback inhibition of ALAS2 is what the paper is *about*,
  and the paper explicitly calls it "a new form of negative feedback in heme biosynthesis". But it
  groups no chemical transformation. I recorded it with `reaction_ids: []` and `confidence: low`
  rather than either dropping it (which would misrepresent the paper's shape) or promoting it to a
  chemical stage (which would be wrong). This was the closest call in the file.

- **PLP.** Genuinely arguable. ALAS chemistry is PLP-dependent by definition, and the apo/holo
  reactivation experiment is the paper's mechanistic core, which pushes toward
  `distinguishes_identity_or_direction`. I kept it `secondary` / `cofactor`: it is regenerated each
  cycle, is never a substrate or product, and the pinned gold flags it explicitly as a cofactor that
  must never be modelled as substrate, product or protein. Recorded in `uncertainties`.

- **ALAS1.** Catalyses the same chemistry and is human, so it is not out of scope; but the paper
  attributes no measured reaction to it and its whole rhetorical purpose is to be *contrasted* with
  ALAS2. I put it in `secondary_participants` (class `other`) rather than as an enzyme on R1, so the
  erythroid attribution stays sharp. Gold lists it under `acceptable_enzymes`, which is consistent.

- **Heme is both the pathway's product and R1's inhibitor.** The paper never asserts a reaction that
  *produces* heme, so heme is important by scope, not as a product of any curated reaction.

## Gold vs. paper

**No disagreement found.** The pinned gold case's `relevance_note` — "Characterises exactly one step
of heme biosynthesis (the ALAS2 condensation) ... Names no other pathway enzyme, no intermediate
between ALA and heme" — matches what I found by independent reading, and gold's single
`supported_reactions` entry is identical to my R1. Gold's `forbidden_identifiers` (the canonical
eight-step porphyrin sequence, ferrochelatase and friends, and the `UROD`-inside-"neurodegeneration"
substring trap) all check out against the text: none of those terms occurs in the body.

## Verification

All 12 quotes confirmed as verbatim contiguous substrings of `full_text` by substring search.
None failed.
