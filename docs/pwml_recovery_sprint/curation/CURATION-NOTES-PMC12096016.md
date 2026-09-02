# Curation notes — PMC12096016

*"The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a competitive inhibitor of
the Escherichia coli isochorismatase EntB"* (Bin & Pawelek, Protein Science 2025).
Requested pathway: **enterobactin biosynthesis**, *Escherichia coli*.
Source: `data/rag_index/acquire_cache/fulltext/af4332adf5750b97dc7fbd2e01384141.json`, 43,667 chars.

## What the paper delivers

**8 core reactions, 4 major subprocesses.** This is the most complete of my five papers.

The Introduction lays out the whole pathway explicitly, with EC numbers, and divides it into two
named stages:

> "In Escherichia coli , enterobactin is synthesized in the cytoplasm in two stages."

- **Stage 1, "the DHB synthetic arm"** — chorismate → isochorismate (EntC, EC 5.4.4.2) →
  2,3-diDHB + pyruvate (EntB isochorismatase domain, EC 3.3.2.1) → 2,3-DHB (EntA, EC 1.3.1.28).
- **Stage 2** — ATP-dependent adenylation of 2,3-DHB by EntE (EC 6.2.1.71), covalent loading onto the
  EntB ArCP domain, transfer to holo-EntF, then three EntF-catalysed condensation cycles with
  L-serine, cyclization and release → enterobactin.
- Plus a **priming step**: EntD phosphopantetheinylates apo-EntB at Ser 245 to give holo-EntB.

The paper's own experimental contribution sits on top of that: 2,3-DHB is a competitive inhibitor of
EntB isochorismatase (Ki ≈ 200 µM), it binds EntE as a natural substrate (KD 0.54 µM), and it cuts
EntC–EntB isochorismate channeling efficiency by ~70%.

## Strength gradient across the eight reactions — this matters

Not all eight are equally asserted, and the file records that:

| | asserted how | confidence |
|---|---|---|
| R1 EntC, R2 EntB | written out as sentences, and measured | `high` |
| R5 EntE adenylation, R8 EntF assembly | written out as sentences | `high` |
| R3 EntA | fixed by *two* statements together (the three-enzyme sequence + EntA's systematic name), never written as one sentence | `medium` |
| R4 EntD priming | explicit, but modifies a protein, not a metabolite | `medium` |
| R6, R7 carrier loading/transfer | asserted as purposes and mechanisms; neither thioester product is named | `medium` |

Two things the paper does **not** supply and I did not invent:

- **No NAD+ for EntA.** EC 1.3.1.28 is an NAD-dependent dehydrogenase, but this paper never says so.
  The only NAD species in the file (NADH → NAD+) belong to the LDH coupled-assay readout. Adding
  NAD+ to R3 would be importing.
- **No pyrophosphate for EntE**, and the adenylate product is named only inside EntE's systematic
  name, never as a free intermediate. Gold makes the same observation.

## Subprocesses named but not detailed

None, in the strict sense — every stage the paper names, it also details. The nearest cases are:

- **S3 (EntD priming)** — detailed, but it is a protein modification rather than a metabolite
  conversion, so it sits awkwardly in a metabolic pathway model. `medium`.
- **S4 (feedback regulation by 2,3-DHB)** — detailed at length, but it groups no transformation.
  `low`, with `reaction_ids: []`, the same treatment I gave the regulatory finding in PMC12856317.

**Deliberately NOT made a subprocess:** EntC–EntB isochorismate channeling and the hypothesised
EntCBAE multienzyme complex. They are prominent in the paper and they are real, but they are a
mechanism for metabolite flux, not a stage of the pathway. Flagged in `uncertainties`.

## Dual classifications (the brief's rule, used twice)

- **2,3-DHB** is `important` (`central_to_pathway_scope`) as the pathway intermediate and EntE's
  substrate, **and** `secondary` (`regulator`) for R2, where the paper is explicit that it is
  "neither a substrate nor a product of EntB". It has three distinct roles in one paper.
- **EntD** is the catalyst of R4 **and** an `ancillary_protein` for everything downstream — the
  paper's own phrase is "the accessory enzyme EntD".

## Near-miss: pyruvate

The closest call. Pyruvate is a currency metabolite in general, which argues for `secondary`. Here it
is one of the two explicitly named products of R2 and the species the coupled assay actually
detects, so I made it `important`. Easily moved.

## Where an extraction can drift — the paper's own decoys

`out_of_scope` is unusually rich here because the paper deliberately runs a *competing* enzyme:

- **MenD** is expressed, purified, drawn in Figure 1 alongside the Ent enzymes, and used as a
  competing isochorismate sink for the menaquinone branch. It must never be exported as an
  enterobactin step. Its co-substrates (thiamine pyrophosphate, 2-ketoglutarate) and product
  (SEPHCHC) go with it.
- **LDH / NADH / lactate** are a porcine coupled-assay reporter, not pathway chemistry.
- **TolC** (efflux) and **TonB** (ferric-enterobactin uptake) are transport, downstream of
  biosynthesis.
- **2,5-DHB and 3,5-DHB** are negative-control isomers.
- **EntH** appears only as a lowercase gene token in the *ent* gene list with no function stated.

## Gold vs. paper

**No disagreement.** Gold marks this `core` / `strict_exportable` with `min_connected_reactions: 4`,
and every enzyme in its `expected_enzymes` (EntC, EntB, EntA, EntE, EntF) is an enzyme of one of my
reactions, with the same roles and EC numbers. Gold's note that "the EntE adenylation is excluded
from the connected floor because its product is never named as a discrete metabolite, and the EntD
step is a protein modification rather than a metabolite conversion" is exactly the reservation I
recorded independently for R5 and R4. Gold's forbidden list (MenD, LDH, NADH, TPP, 2-ketoglutarate,
SEPHCHC, 2,5-/3,5-DHB, EntH, R196A) matches my `out_of_scope` item for item.

## Cross-paper discipline

PMC12452463 is the other enterobactin paper in my set. **Nothing was carried between them in either
direction.** Every entry above rests on a quote from this paper's own `full_text`.

## Verification

All 27 quotes were extracted as exact spans of `full_text` programmatically — this source uses
U+2010 hyphens throughout and has irregular spacing such as `apo ‐EntB`, so transcribing quotes by
hand would have produced non-verbatim strings — then re-verified by substring search. None failed.
