# Curation notes — PMC12782028

*"Cholesterol and steroid synthesis pathways may be involved in the inhibition of osteosarcoma cell
viability by calcium-sensing receptor antagonism"* (Wang et al., PeerJ 2026).
Requested pathway: **cholesterol biosynthesis**, *Homo sapiens*.
Source: `data/rag_index/acquire_cache/fulltext/291f35c3c2149c7c0085d87b35566bc0.json`, 43,679 chars.

## What the paper delivers

**4 core reactions, 6 major subprocesses — two of which are `detailed_in_paper: false`.**

The paper's own output is transcriptomic: RNA-seq of osteosarcoma cells treated with the CaSR
antagonist NPS-2143, KEGG and Reactome enrichment, and qRT-PCR validation. No sterol is quantified
and no enzyme activity is assayed anywhere.

The chemistry appears in the Discussion, as cited background, for exactly **four** enzymes:

| enzyme | stated reaction |
|---|---|
| LSS | 2,3-oxidosqualene → lanosterol |
| CYP51A1 | lanosterol → 4,4-dimethylcholesta-8(9),14,24-trien-3β-ol |
| MSMO1 | three-step monooxygenation demethylating 4,4-dimethyl and 4α-methylsterols |
| DHCR24 | desmosterol → cholesterol ("the final step") |

## Named but NOT detailed — the point of this paper

This is the distinction the brief warned me about, and it is where most of this paper's
cholesterol-pathway content actually lives.

The three enrichment gene lists explicitly assign to the **cholesterol biosynthesis pathway
(R-HSA-191273)** the genes **ACAT2, HMGCS1, HMGCR, MVK, MVD, IDI1, FDPS, FDFT1, SQLE**, plus NSDHL,
EBP, HSD17B7, LBR and SREBF1/2 — many with their full protein names spelled out. **Not one of them
is given a reaction.** SQLE and FDFT1 are even qRT-PCR-validated, and still carry no catalytic
statement.

I recorded this as **S5**, "the upstream, pre-lanosterol segment of cholesterol biosynthesis
(mevalonate and squalene arms)", with `detailed_in_paper: false` and **no reaction ids**. Collapsing
it either way would have been wrong: dropping it would hide that the paper places the whole upstream
arm inside the pathway; giving it reactions would be writing the mevalonate pathway from memory.

The supporting census (substring search over `full_text`) is decisive — **no upstream metabolite is
ever named as a free compound**:

| token | occurrences | where |
|---|---|---|
| `acetyl-CoA` | 1 | inside "acetyl-CoA acetyltransferase 2" |
| `mevalonate` | 3 | inside two enzyme names + the phrase "(mevalonate pathway)" |
| `isopentenyl` | 1 | inside "isopentenyl-diphosphate delta isomerase 1" |
| `farnesyl` | 4 | inside FDPS and FDFT1 names |
| `squalene` | 2 | inside "squalene epoxidase" and inside "2,3-oxidosqualene" |
| `HMG-CoA` | 2 | "HMG-CoA reductase" (×2, once as the statin target) |

So as far as this paper is concerned, cholesterol biosynthesis **starts at 2,3-oxidosqualene**.

A note on the phrase "mevalonate pathway": it occurs exactly once, and this paper uses it as a
**synonym for the whole cholesterol biosynthesis pathway** — "the final step in the cholesterol
biosynthesis pathway (mevalonate pathway)" — not as the name of the upstream arm. I used it only as
an *alias* on S5 and flagged the loose usage rather than treating it as the paper's own name for the
pre-lanosterol segment.

**S6**, SREBP-mediated transcriptional regulation, is the second `detailed_in_paper: false` entry:
named as a control layer over the pathway, both SREBF genes in the pathway gene list, no mechanism
given, no transformation grouped. `low`.

## The route is not connected end to end, and I did not connect it

Only **R1 and R2 chain**, through the string-identical metabolite `lanosterol`. Beyond that:

- R2's product, 4,4-dimethylcholesta-8(9),14,24-trien-3β-ol, is consumed by nothing stated — the
  paper never says it is the MSMO1 substrate.
- R3's substrate is a **compound class** and its product is not named at all. That is why R3 is
  `medium`.
- R4's substrate, **desmosterol, is produced by nothing** in this paper. The terminal step is
  topologically orphaned.
- 2,3-oxidosqualene is produced by nothing — SQLE, which would, is named without a reaction.

Bridging any of those gaps would mean importing the pathway from general knowledge. I left them open
and recorded each in `uncertainties`.

## Things that would trip an extractor

- **No cofactor is named for any of the four reactions.** No NADPH (0 occurrences), no NAD(P)H, no
  reductant for the DHCR24 reduction, no oxygen. I recorded molecular oxygen as a `secondary`
  participant of R3 only because "monooxygenation" implies it, and said so.
- **`lanosterene` is a typo for lanosterol** (as is "tetra-closterane" for tetracyclosterane). Not a
  new compound.
- **DHCR24 appears under four surface forms** — DHCR24, "24-dehydrocholesterol reductase",
  "δ24-cholesterol reductase", "3β-hydroxysterol Δ24-reductase" — all one protein.
- **CaSR** is a receptor, not a reductase, despite the R.
- **Direction caution.** The paper reports that the CaSR antagonist *up*-regulates these genes and
  that the resulting cholesterol **overproduction** is what reduces tumour cell viability. That is
  the opposite of the statin narrative; an extractor assuming inhibition inverts the paper's claim.
- **7-dehydrocholesterol / DHCR7 are absent entirely** — the Kandutsch–Russell branch is
  unrepresented here.
- The **steroid biosynthesis pathway (hsa00100)** is analysed in parallel and shares genes with
  cholesterol biosynthesis. It is a neighbouring pathway, not a subprocess, so it is in
  `out_of_scope` — but it is the most plausible place for an extraction to drift.

## Gold vs. paper

**No disagreement.** Gold marks this `partial`, with the same four catalytic statements as its
`expected_enzymes` and the same roles. Gold's `acceptable_enzymes` annotate SQLE, FDFT1, HMGCR,
HMGCS1, NSDHL and EBP as "enrichment list only" / "NO catalytic statement anywhere" — which is
precisely my S5 with `detailed_in_paper: false`. Gold's note that "MSMO1's substrate is given only as
a compound class, and DHCR24's desmosterol step is topologically orphaned because no stated reaction
produces desmosterol" is the same reservation I recorded independently for R3 and R4, and gold flags
the same `lanosterene` typo and the same overproduction direction caution.

Gold's `min_connected_reactions` is 2 while I curated 4 core reactions — not a disagreement: gold is
counting *connected* reactions, I am recording *asserted* ones, and I state the connectivity gaps
explicitly.

## Verification

All 21 quotes were extracted as exact spans of `full_text` programmatically and re-verified by
substring search. None failed.
