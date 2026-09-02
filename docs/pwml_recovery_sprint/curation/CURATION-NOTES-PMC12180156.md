# Curation notes — PMC12180156

*"Update of the sideroflexin (SLC56) gene family"* (Katsafadou et al., Human Genomics 2025).
Requested pathway: **heme biosynthesis**, *Homo sapiens*.
Source: `data/rag_index/acquire_cache/fulltext/48a60159cc20f1440483310afe45a6a1.json`, 67,304 chars.

## Headline: the core reaction list is empty, on purpose

**0 core reactions, 2 major subprocesses (both `detailed_in_paper: false`).**

This is a review of a transporter gene family. It names heme biosynthesis half a dozen times and it
names two genuine heme-pathway enzymes — but it never states one heme-biosynthesis transformation.
Nowhere in the 67,304 characters does a reaction have both a heme-pathway substrate and a
heme-pathway product named.

I confirmed that by exhaustive substring search over `full_text`, not by impression:

| term | occurrences in file |
|---|---|
| `succinyl` | **0** |
| `aminolevulinic` | 1 (only in the clause naming ALAS2) |
| `protoporphyrin` | **0** |
| `porphyr` | **0** |
| `ferrochelatase` | 1 |
| `ALAS` | 2 (both in the same clause) |

So ALAS2 is named with neither substrate nor product; ferrochelatase is named with neither substrate
nor product. Every canonical step of heme biosynthesis one might want to write down would have to be
imported from memory. I imported none.

## What the paper does say about heme

Three statements, and that is the lot:

1. > "Glycine is the direct precursor for heme"
2. > "Glycine is also required for the rate-limiting step in heme biosynthesis." (Fig. 4 legend)
3. > "SFXN4 regulates heme biosynthesis by modulating ferrochelatase levels and inhibiting the
   > translation of erythroid δ-aminolevulinic acid synthase (ALAS2), a key enzyme in the heme
   > synthesis pathway"

Plus a phenotype: SFXN2-KO cells have more mitochondrial iron but *less* heme.

Statements 1 and 2 are precursor relations. Statement 3 is a *regulatory* relation on protein levels
and translation — it is about how much enzyme there is, not about what the enzyme does.

## Subprocesses named but not detailed — the whole subprocess list

Both entries are `detailed_in_paper: false`:

- **S1, the rate-limiting step of heme biosynthesis (the ALAS2 step)** — `medium`. The paper refers
  to a constituent step and names its enzyme, but supplies no chemistry.
- **S2, the ferrochelatase step** — `low`. This is the weakest entry in the file and I say so in the
  JSON. The paper names ferrochelatase and places it inside heme biosynthesis by implication, but
  never calls it a stage and never says what it catalyses. A reviewer who deletes S2 has a case.

## Near-miss classifications

- **ALAS2 and ferrochelatase as "participants".** There is no reaction for them to be enzymes of, so
  I recorded them under `important_participants` with reason `distinguishes_identity_or_direction`
  — without those names you cannot tell which stage the paper is gesturing at. It is a slightly
  unusual use of the reason code; the alternative was to drop the paper's only two heme-enzyme names
  entirely, which seemed worse. Flagged in `uncertainties`.
- **Iron.** Discussed constantly, but always as a mitochondrial pool, never as a substrate of an
  insertion reaction. Recorded `secondary` / `other`. Note that `Fe 2+` / `Fe 3+` occur in the body
  only inside an explicitly *negative* sentence ("there is no evidence showing SFXN2 (or any other
  SFXN) to function directly as an Fe 2+ or Fe 3+ transporter") — extracting them as substrates
  inverts the paper.

## The two real decoys

The paper *does* fully specify two reactions, both with substrate and product named:

- SHMT2 cleaving serine to glycine (+ 5,10-methylene-THF), and
- SFXN1 transporting serine cytosol → matrix.

Neither is heme biosynthesis. Both are in `out_of_scope`. They are exactly what an extractor would
be tempted to retain, because they are the only complete transformations in the file and their
product (glycine) is the molecule the paper links to heme. A further trap: the *plant* SHMT sentence
runs in the opposite direction ("producing serine from glycine"), so pulling that sentence into a
human model reverses the reaction.

## Cross-paper leakage — stated explicitly

PMC12856317, also assigned to me and also "heme biosynthesis", **does** assert the ALAS2 condensation
of glycine and succinyl-CoA. Nothing from it was carried here. The glycine entry in this file rests
only on this paper's own "Glycine is the direct precursor for heme".

## Gold vs. paper

**No disagreement.** Gold marks this `context_only`, `min_connected_reactions: 0`,
`max_retained_reactions: 2`, and its `relevance_note` — "Ferrochelatase and ALAS2 are named with NO
reaction stated for either" — is exactly what I found independently. Gold's `expected_enzymes` is
empty and its `acceptable_enzymes` entries carry the annotation "NAME ONLY", which agrees with my
`detailed_in_paper: false` treatment. Gold's `forbidden_identifiers` list of ALA/porphobilinogen/
protoporphyrin IX/succinyl-CoA as a hallucination test checks out: all are absent from the file.

## Verification

All 10 quotes confirmed as verbatim contiguous substrings of `full_text` by substring search. None
failed.
