# Curation notes — PMC12444477

**Requested pathway:** lipid A biosynthesis · **Requested organism:** *Escherichia coli*
**Title:** *The regulation of lipid A biosynthesis* (JBC Reviews, Hummels 2025)
**Source:** `data/rag_index/acquire_cache/fulltext/8ceca97a513c8b864b6a1a7049228197.json`, `full_text` = 82,306 chars
**Result:** 9 core reactions · 5 major subprocesses · 10 important participants · 5 secondary participants · 24 out-of-scope items

---

## What the paper is, and how much of the requested pathway it delivers

This is a review about the **regulation** of lipid A biosynthesis, not about its chemistry. The
chemistry is delivered completely but compressed into a single paragraph (the "Raetz pathway"
paragraph) plus the Figure 1 legend — roughly 1,900 characters out of 82,306. Everything after that
is about how LpxC abundance is controlled, and a large share of it is in organisms other than the
requested *E. coli* (*P. aeruginosa*, *N. meningitidis*, *R. sphaeroides*, *C. crescentus*,
*A. baumannii*, *S.* Typhimurium, *C. sakazakii*, *F. tularensis*).

Within that one paragraph, however, the pathway really is complete: nine enzymes in order —
LpxA → LpxC → LpxD → LpxH/I/G → LpxB → LpxK → WaaA → LpxL → LpxM — ending at Kdo-lipid A. So the
curated reaction list is nine entries, and the *shape* of the pathway is fully recoverable from
this paper.

## The quality of the reactions is very uneven

This is the finding that mattered most while curating. Four of the nine reactions have chemically
named species on both sides; **five do not**.

| Reaction | Named substrate | Named product | Confidence |
|---|---|---|---|
| R1 LpxA | UDP-GlcNAc, beta-hydroxyacyl-ACP | — ("the LpxA product") | high |
| R2 LpxC | — | — | high |
| R3 LpxD | — | — | medium |
| R4 LpxH/I/G | — | UMP | medium |
| R5 LpxB | — | tetra-acylated disaccharide intermediate | high |
| R6 LpxK | tetra-acylated disaccharide intermediate | lipid IVA | high |
| R7 WaaA | lipid IVA, Kdo | Kdo-lipid IVA | high |
| R8 LpxL | Kdo-lipid IVA | — | medium |
| R9 LpxM | — | Kdo-lipid A | medium |

R2 is marked **high** despite naming neither species, because the transformation itself
("removes the acetyl group from the GlcNAc moiety") is stated unambiguously and the paper calls it
the committed step of the requested pathway. R3, R4, R8 and R9 are **medium**: the paper asserts
the transformation and the catalyst, but the species are given only positionally ("the next acyl
group", "sequentially acylated").

**The trap this creates.** The *E. coli* LpxA product, `UDP-3-O-(R-3-hydroxymyristoyl)-N-acetyl-
glucosamine`, does occur in the file — but only inside the **title of bibliography reference 7**. It
is a cited work's title, not a claim in this paper's body. I did not use it. Likewise, the one
chemically named LpxC product in the whole paper,
`UDP-3-O-(R-3-hydroxydecanoyl)-glucosamine`, is explicitly labelled the ***P. aeruginosa*** product
in the MurA section. Attaching either to the *E. coli* chain would be an error the paper does not
license; both are recorded, one in `uncertainties` and one in `out_of_scope`.

## Subprocesses the paper names but does not detail

Strictly, **none** — every stage I recorded is chemically detailed at least at the level of
"enzyme X does transformation Y". All five subprocess entries are `detailed_in_paper: true`.

What the paper *does* name without detailing are the things I put **out of scope**, because they
belong to different processes rather than to lipid A biosynthesis:

- **MsbA flipping** and **Lpt transport** — the paper states "Transport is achieved in two steps"
  and gives no chemistry for either, citing reviews instead. Crucially the paper itself draws the
  boundary: "The synthesis of lipid A **and its subsequent assembly into the OM**". They are LPS
  biogenesis, not lipid A biosynthesis.
- **O-antigen ligation** — one clause, no enzyme named.
- **Core assembly beyond Kdo** (WaaC) — appears only as a LapB copurification partner.

## The judgement call I am least sure of

**Regulation of LpxC is 85% of this paper and I did not record it as a major subprocess.**

The YejM–LapB–FtsH partner-switching module is described in exhaustive mechanistic detail, and a
naive reading of "what should an extraction recover from this paper" would put it first. I left it
out of `expected_major_subprocesses` because a major subprocess is defined in the brief as a stage
*of the requested pathway* that groups reactions, and regulation of an enzyme's abundance groups no
lipid A reactions. The regulators are instead recorded in `secondary_participants` with
`class: "regulator"`, pointed at the reaction whose *catalyst* they control (FtsH/LapB/YejM → R2;
ObgE → R1).

The pinned gold case independently supports this reading: it lists FtsH, LapB, YejM, FabZ, MsbA and
ObgE under `acceptable_enzymes`, not under `expected_enzymes`. That agreement is corroboration, not
the reason for the choice.

If a reviewer disagrees, the fix is additive (add one subprocess entry), so the omission is
recoverable in the sense the brief asks for.

## The other near-miss: R8/R9 as one reaction or two

The paper says Kdo-lipid IVA "is sequentially acylated by LpxL and LpxM to finally form Kdo-lipid
A" and never names the intermediate between them. I split this into two reactions because (a) the
word "sequentially" asserts two steps and (b) the paper's own count of **nine** enzymes only works
if LpxL and LpxM are counted separately. A single combined entry would also be defensible. Recorded
in `uncertainties`.

Related: the sentence "The nine steps in the Raetz pathway **as well as** the subsequent addition of
the core sugars" reads as if WaaA sits *outside* the nine — but then the enzyme list only reaches
eight. This is an internal inconsistency in the source, not a curation choice; I counted WaaA in and
flagged it.

## Gold vs. paper

**No disagreement found.** Every quote in the pinned gold case that I spot-checked appears verbatim
in `full_text`, once the source's erratic spacing is allowed for (`lipid IV A`, `Ec LpxC`,
`uridine diphosphate- N -acetyl-glucosamine`, `R- 3-hydroxymyristoyl-ACP`). The gold
`relevance_note` — "names all nine Raetz-pathway enzymes in order, but delivers the chemistry in a
single ~250-word paragraph" — is an accurate description of what I found independently.

## Verification

All 29 quotes in `expected_core_PMC12444477.json` were confirmed by substring search against
`full_text` before the file was written. **None failed.**
