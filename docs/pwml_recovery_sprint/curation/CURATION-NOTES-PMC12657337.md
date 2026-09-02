# Curation notes — PMC12657337

**Requested pathway:** menaquinone biosynthesis · **Requested organism:** *Bacillus subtilis*
**Title:** *High-level production of vitamin K2 in Escherichia coli via modular molecular engineering* (Synth Syst Biotechnol, 2025)
**Source:** `data/rag_index/acquire_cache/fulltext/2756073ab2435e6920b580d093c2c54b.json`, `full_text` = 57,814 chars
**Result:** 4 core reactions (1 at `low`) · 4 major subprocesses (3 of them `detailed_in_paper: false`) · 6 important · 2 secondary · 19 out-of-scope

---

## What this paper delivers, and what it does not

This is a strain-engineering paper, not a pathway paper. Its scientific weight is in Rosetta ddG
stability design, PrankWeb pocket prediction, 2,000 random mutants, AI-designed RBS variants, and a
50-L fed-batch fermentation reaching 2.18 g/L MK-7. The pathway chemistry is stated compactly, twice,
and then not revisited.

What it *does* deliver, cleanly: **the three terminal steps of menaquinone-7 biosynthesis.**

```
FPP --HepPPS--> HepPP --MenA (+DHNA)--> DMK-7 --UbiE--> MK-7
```

Each has a named catalyst, a named substrate and a named product, in one sentence:

> "BsHepPPS encodes heptaprenyl pyrophosphate synthase, which catalyzes the conversion of farnesyl
> pyrophosphate to heptaprenyl pyrophosphate; EcMenA encodes a prenyltransferase that catalyzes the
> conjugation of heptaprenyl pyrophosphate with DHNA to form demethylmenaquinone; and BsUbiE encodes
> a methyltransferase that catalyzes the methylation of demethylmenaquinone to produce MK-7."

## Subprocesses the paper NAMES but does not DETAIL — the headline finding

The paper itself declares the pathway's shape:

> "The metabolic pathway responsible for MK-7 biosynthesis is primarily divided into three modules
> (the MVA pathway, DHNA pathway, and MK-7 pathway)."

Of those three modules, **only the MK-7 module is chemically delivered.** Three of my four
subprocess entries are `detailed_in_paper: false`.

### The DHNA pathway is completely absent

DHNA is a **substrate of R3** and the paper names "the DHNA pathway" twice as a module of MK-7
biosynthesis — yet supplies not one enzyme and not one step of it. I ran case-sensitive counts over
the full text to be sure:

| Symbol | Count | | Metabolite | Count |
|---|---|---|---|---|
| `MenF` | **0** | | `chorismate` | **0** |
| `MenH` | **0** | | `isochorismate` | **0** |
| `MenC` | **0** | | `SEPHCHC` | **0** |
| `MenE` | **0** | | `SHCHC` | **0** |
| `MenB` | **0** | | `OSB` | **0** |
| `MenI` | **0** | | | |
| `MenG` | **0** | | `DHNA` | 12 |

So `S2` (DHNA pathway) is recorded with an **empty `reaction_ids`**. This is the omission the task
warned about, and it is the single most important thing to carry forward: **the classical seven-enzyme
chorismate → DHNA chain that PMC12421875 enumerates in full is entirely missing from this paper.**
Any MenF/MenD/MenH/MenC/MenE/MenB/MenI chain ever attributed to PMC12657337 has been imported from a
neighbouring paper or from general knowledge.

`MenD` does occur 4 times — but every occurrence is a citation of Huang et al.'s *B. subtilis*
engineering work ("Huang et al. identified MenD and MenA as key enzymes in MK-7 biosynthesis in
*B. subtilis*"), with no substrate, no product and no reaction. It is a real menaquinone enzyme, so
I did *not* put it in `out_of_scope`, but it supports no reaction here.

### The MVA pathway is a roster, not chemistry

`S3` gets an enzyme list ("EfmvaE, EfmvaS, MmmvK, ScpmK, and ScmvD") and the Figure 1 legend expands
abbreviations (MvaE, MvaS, MVK, PMK, MVD, HMG-CoA, M-5P, M-5PP, IPP, DMAPP). No substrate→product
step is stated in prose. `detailed_in_paper: false`.

## Two source defects I recorded and did not smooth over

**1. The Idi reaction contradicts the paper's own figure legend.** The body says:

> "The enzyme Idi , which converts IPP to farnesyl pyrophosphate (FPP)"

The Figure 1 legend says `"Idi, IPP isomerase"` and separately lists
`"ispA, Isopentenyl pyrophosphate transferase"`. Those two statements cannot both be true.

**This was my hardest call.** I included it as `R1` at **`low` confidence** because the brief tells
me to curate what the paper asserts, not what I believe is correct, and the sentence does supply
substrate, product and catalyst — meeting the brief's §4(a) test literally. But I flagged it loudly
in `description`, in `rationale` and in `uncertainties`: **R1 is not validated biology.** A reviewer
who drops it as a source defect has a strong case. Dropping it does not disturb R2–R4, which are the
menaquinone-specific chain.

**2. Precursor pathway inconsistency.** The introduction says MK-7 biosynthesis requires "the
methylerythritol phosphate (MEP) pathway to produce isopentenyl pyrophosphate (IPP)". The figure
legend and the entire results section instead build a heterologous **mevalonate (MVA)** pathway. I
recorded both (S3 MVA, S4 MEP) because the paper asserts both; they are alternative routes to the
same IPP, not sequential stages.

I also noticed, and flagged without depending on, that the legend's
`"MvaE, HMG-CoA synthase; MvaS, HMG-CoA reductase"` appears to be the reverse of the standard
*Enterococcus faecalis* assignment. No entry in the file rests on it.

## The entry I nearly classified the other way: SAM

`S-adenosyl-L-methionine` is the methyl donor for the UbiE step. A methyltransferase reaction
arguably is not identifiable as a methylation without its donor, which under §4(c) would make SAM
**important** (`distinguishes_identity_or_direction`).

I classed it **secondary** (`cofactor`) because **this paper never places SAM inside the reaction** —
SAM and SAH appear only in the Figure 1 abbreviation glossary. That is a placement judgement, not a
biochemical one, and it is recorded in `uncertainties` so the consumer can flip it.

## Organism trap

The requested organism is ***B. subtilis*** (`topics_t104.txt` line 43, and the gold case). This is
an ***E. coli*** study. *B. subtilis* appears only as the **gene donor** for HepPPS and UbiE — hence
the author labels `BsHepPPS` and `BsUbiE` — and in citations of prior *B. subtilis* work. **No
menaquinone reaction is asserted to occur in *B. subtilis* in this paper.** Recorded, not resolved.

Related naming hazard: `EcMenA`, `BsUbiE`, `BsHepPPS` are author-coined organism-prefix labels for
ordinary enzymes; and the Figure 1 legend introduces a *third* spelling of the R2 catalyst,
`"Heps, heptaprenyl pyrophosphate synthetase I"`.

## Gold vs. paper

**No disagreement.** Gold's `expected_enzymes` are exactly HepPPS, MenA and UbiE — matching R2–R4 —
and it places `Idi` and `acs` under `acceptable_enzymes`, which lines up with my `low`-confidence R1
and my decision to put `acs` (acetate → acetyl-CoA, central carbon metabolism) in `out_of_scope`.

## Verification

All 16 quotes in `expected_core_PMC12657337.json` were confirmed by substring search against
`full_text` before the file was written. **None failed.**
