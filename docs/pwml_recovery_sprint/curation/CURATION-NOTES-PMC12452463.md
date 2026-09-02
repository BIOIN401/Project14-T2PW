# Curation notes — PMC12452463

*"Enterobactin: A key player in bacterial iron acquisition and virulence and its implications for
vaccine development and antimicrobial strategies"* (Amiri et al., Virulence 2025).
Requested pathway: **enterobactin biosynthesis**, *Escherichia coli*.
Source: `data/rag_index/acquire_cache/fulltext/2a1344506577ea51c003230a0a557f49.json`, 68,085 chars.

## What the paper delivers

**5 core reactions, 5 major subprocesses.**

This is a narrative review of enterobactin's role in virulence, immunology and therapeutics. Its
biosynthesis content is one short section, roughly 1,700 characters out of 68,085, and it is
secondhand. The paper announces its own scheme:

> "Here are the 4 steps of enterobactin biosynthesis"

and those four steps — which I used verbatim as subprocesses S1–S4 — are:

1. **Chorismate to Isochorismate** (EntC, isochorismate synthase; "the first committed step")
2. **Isochorismate to 2,3-Dihydro-2,3-Dihydroxybenzoate (DHB)** (EntB, isochorismatase)
3. **Activation of DHB** (EntE, 2,3-dihydroxybenzoate-AMP ligase → DHB-AMP)
4. **Assembly of Enterobactin** (EntD primes EntB and EntF; EntF, an NRPS, condenses and cyclises
   three DHB into the trilactone)

Plus a fifth, regulatory subprocess (Fur / RyhB / Fnr), recorded at `low` with no reaction ids.

## What this paper does NOT assert — the point of curating it separately

I was warned not to carry chemistry across from the other enterobactin paper. Here is exactly what
this one omits, each confirmed by substring census over the whole file rather than by impression:

| missing thing | occurrences in the 68,085-char file |
|---|---|
| the **EntA** dehydrogenase step | `dehydrogenase` **0**; `NAD` **0**; capitalised `EntA` **1**, inside a reference title |
| **L-serine** | `serine` **0**, any case |
| **ATP** as the adenyl donor | `ATP` **1**, inside a reference title about ATP synthase |
| **pyruvate** as the EntB co-product | `pyruvate` **0** |
| water / hydrolysis for EntB | `hydroly` **0** |
| Mg²⁺ for EntC | `Mg` **0**; `magnesium` **0** |

**This paper names no cofactor of any kind for any biosynthetic step.** My
`secondary_participants` list contains regulators and iron species only, and that emptiness is
itself the finding.

### The missing EntA step, and why I created no slot for it

The paper's own four-step scheme skips the dehydrogenation entirely. It *does* say entA is one of
six operon genes "each encoding an enzyme essential for the production of enterobactin" — that is
the only thing it says about EntA anywhere.

I deliberately did **not** create a `detailed_in_paper: false` subprocess for the EntA stage. In
PMC12180156 I used exactly that device for a stage that was named but undetailed. The difference is
that here the paper's own scheme *excludes* the stage: opening a slot for it would license precisely
the fabrication — an EntA reaction cited to a paper that never states one. A reviewer who wants the
hole represented should represent it as a **gap**, not as a stage. Recorded loudly in
`uncertainties`.

### The DHB collision — the route is chemically broken as written

The body says:

> "Isochorismatase (EntB) converts isochorismate into 2,3-dihydro-2,3-dihydroxybenzoate (DHB)"

The Figure 1 caption says:

> "formation of 2,3-dihydroxybenzoate (DHB) by EntB"

Those are **different molecules**, given the same abbreviation, with the same enzyme credited for
producing each. Because the EntA step is missing, nothing in this paper converts
2,3-dihydro-2,3-dihydroxybenzoate onward, and R3 chains to R2 only through the abbreviation
collision. I recorded **both** molecules as participants — the second under
`distinguishes_identity_or_direction`, because without it EntE's substrate cannot be identified —
and **did not resolve the contradiction**. That is why R3 is `medium` rather than `high`: the
assertion is clear, the chaining is not.

### Carrier-protein chemistry: binding, not loading

This paper says only that "EntB also functions as a carrier protein, binding DHB for subsequent
steps." It asserts **binding**. It never asserts covalent attachment of the aryl group to a carrier
domain, and never asserts transfer of the loaded group from EntB to EntF. In PMC12096016 both of
those *are* asserted and I curated both as reactions. Here I created neither. That divergence is
deliberate and is the clearest illustration of curating each paper from its own text.

## Side-by-side with PMC12096016 (the other enterobactin paper in my set)

| | PMC12096016 | PMC12452463 |
|---|---|---|
| EntC chorismate → isochorismate | yes, + Mg²⁺, + reversibility | yes, no cofactor |
| EntB product | 2,3-diDHB **+ pyruvate**, hydrolysis | 2,3-dihydro-2,3-DHB, no co-product |
| EntA step | yes, EC 1.3.1.28 | **absent** |
| EntE adenylation | yes, **ATP-dependent**, EC 6.2.1.71 | yes, **no ATP named** |
| covalent ArCP loading | yes | no (binding only) |
| EntB → EntF transfer | yes | no |
| EntF assembly | 3 × condensation **with L-serine**, cyclization | 3 DHB → trilactone, **no serine** |
| EntD priming | yes, at Ser 245, on EntB | yes, on EntB **and EntF**, no site |
| EC numbers | four | none |
| core reactions curated | 8 | 5 |

Nothing was carried in either direction.

## Gold vs. paper

**No disagreement.** Gold independently reached the same three findings I did, in nearly the same
words: `relevance_note` says the biosynthesis section is "missing the EntA step entirely, and
internally inconsistent about what DHB is"; `forbidden_identifiers` flags **EntA** as a hallucination
test ("the capitalised form 'EntA' appears ONLY inside a reference title"), flags the bare **DHB**
token as unresolvable, and groups **L-serine / ATP / pyruvate** as things "named nowhere in this
review, even though all three are real parts of the chemistry". Gold's `export_rationale` calls the
route "chemically BROKEN". I reached each of those from the text before reading gold's forbidden
list, and they agree.

One presentational difference, not a disagreement: gold's `min_connected_reactions` is 2 and its
`supported_reactions` list has 3 entries, whereas I curated 5 core reactions. Gold is counting
*cleanly chained metabolite conversions*; I am recording *transformations the paper asserts*, which
additionally includes the EntD protein modification (R4) and separates the assembly (R5). The two
are consistent once the different questions are noted.

## Verification

All 24 quotes were extracted as exact spans of `full_text` programmatically and re-verified by
substring search. None failed.
