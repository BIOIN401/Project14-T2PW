# Species policy

A mining run declares **exactly one target species** and it is mandatory.
`MiningConfig` raises `ValueError` without it: the guardrail cannot classify
evidence against a target that does not exist, and "whatever the papers say" is
not a species.

Both forms are preserved, always:

```json
"target_species": "A. thaliana",
"target_species_normalized": "arabidopsis thaliana"
```

Normalization never replaces what the user wrote.

## Local normalization is not verification

`species.normalize_species` lowercases, collapses whitespace, strips
parenthetical strain/cultivar notes, expands the abbreviated binomial
(`A. thaliana`), trims a strain tail (`Escherichia coli K-12 MG1655` ->
`escherichia coli`) and maps a short table of common names (`human`, `mouse`,
`yeast`, `E. coli`, ...).

It makes **no external call** — not to NCBI Taxonomy, not to anything. An
unrecognised organism normalizes to itself. A normalized string is a comparison
key, never a resolved identifier.

`species_matches_target` requires exact normalized equality. A genus does not
satisfy a binomial: `Arabidopsis lyrata` never matches `Arabidopsis thaliana`.
(The common-name table does map the bare genus `arabidopsis`; that is a single
documented, auditable convenience, not an inference rule.)

## The seven states

Every claim gets exactly one, with a reason. Extraction assigns it with
`species.grade_sentences` and `species.aggregate_species` (ORCH-745 F-1), which
read the claim's own supporting sentences; `species.classify_species`, the older
whole-passage classifier, is kept for its existing callers only.

| State | Meaning | Admissible? |
|---|---|---|
| `target_species_supported` | the claim's species is the target, a span the model cited for the species names the target in a plain sentence, **and** a supporting sentence shows the organism performing the reaction (grade `in_planta`), or a review states it for the target (grade `review`) | **yes — the only one** |
| `non_target_species` | the claim's species is a different one | no |
| `multiple_species` | the claim names more than one species | no |
| `species_ambiguous` | the claim names the target, but no supporting sentence shows the organism performing the reaction, or the species is uncited, or the target is named only in an ortholog, host-system or comparison sentence | no |
| `species_not_stated` | the claim states no species | no |
| `in_vitro_or_cell_free` | a supporting sentence describes a cell-free / recombinant / purified-enzyme / extract system and none shows the reaction in the organism | no |
| `inferred_only` | a supporting sentence is an inference (ortholog, analogy, "inferred by"), or the model set `species_inferred` | no |

## Reading the claim's own sentences

Base read whole passages: one "in vitro" anywhere in a 5,000-character span decided
every claim citing it, and a span that named *Arabidopsis* anywhere made the target
"supported". Pilot-003 stored GH3.10's knockout evidence for JA -> JA-Ile as
`in_vitro_or_cell_free` (the abstract's next sentence mentions recombinant enzyme
assays) and the bryophyte review's OPR3 statements as `inferred_only` (the span
elsewhere mentions orthologs). The rule now:

1. **Cited spans** are every span the claim cites, for any field or at claim level.
   Each is split into sentences (no break after "A.", "et al.", "Fig.", "e.g.").
2. **Supporting sentences** name one of the claim's participants, its enzyme or its
   gene. Names are matched after folding case, whitespace, Unicode hyphens, Greek
   letters and the italic gaps around hyphens ("dn- cis -OPDA"). A name
   "X (Y)" is matched as written and as Y ("α-linolenic acid (LeA)" also as "LeA"),
   never as X alone: the note may be what disambiguates it ("ALA (18:3n3)"; a bare
   ALA is also 5-aminolevulinic acid), the rule `queries.rescue_terms` keeps. A
   participant or enzyme the model asserted but extraction dropped still names
   supporting sentences, so their negative markers still count.
3. **Each supporting sentence gets one grade**, tested in this order:
   * `inference`: `ortholog`, `homolog of`, `by analogy`, `by similarity`,
     `presumably`, `is expected to`, `likely catalyses`, `extrapolat`, `inferred`;
   * `host`: `transgenic`, `heterologous`, `ectopic`, `overexpress`, `35S`,
     `expressing`, `expression of`, `transient`, `agroinfiltrat`, `infiltrated`,
     `protoplast`, `transformed`, `expressed in` a plant — the organism is a host
     for a gene, never performing its own reaction;
   * `in_vitro`: `in vitro`, `cell-free`, `recombinant`, `purified`, `reconstitut`,
     `expressed in` *E. coli* / yeast / *Pichia* / insect cells, `enzyme assay`,
     `incubation`, `incubated with`, `Km`, `Vmax`, `kcat` (also spaced, "K m"),
     `kinetic`, `lysate`, `ex vivo`, `crude` / `tissue` / `protein extract`,
     `homogenate`, `microsom`, `isolated chloroplasts`;
   * in a paper not graded a review, `in_planta` only if ALL hold: no hedge (`may`, `might`,
     `could`, `potential`, `putative`, `likely`, `suggest`, `support the view`,
     `anticipated`, `proposed`, ...); no citation (`[n]`, `et al`, an author-year);
     no procedure (`was/were grown, sprayed, treated, extracted, analysed, measured,
     incubated, used ...`, `we employed/used`); no purpose clause (`to study`, `to
     test`, `as a new tool`); the passage is not a methods or introduction section;
     the sentence names no non-target organism, and names the target itself when
     the passage or the paper's screen organisms name another; and within ONE
     clause (split at `;` `:` and the words *and / but / whereas / while / or*) an
     in-planta context (`in planta`, `in vivo`, `mutant`, `knockout`, `wild type`,
     `Col-0`, `seedlings`, `leaves`, `flowers`, `roots`, `tissues`, `wounded`,
     `lines`, ...) occurs with either (a) the substrate AND product, named strictly,
     and a conversion (`convert`, `conversion`, `formed from`, `metabolised to`,
     `produced from`, `synthesised from`), with no application (`treat`, `appl`,
     `spray`, `exogenous`, `fed`, `feeding`) in the sentence; or (b) a
     loss-of-function marker (`mutant`, `knockout`, `knockdown`, `loss-of-function`,
     `null allele`, `T-DNA`, `SALK_`, `RNAi`, `amiRNA`, `CRISPR`), the claim's enzyme
     or gene, and its substrate or product named strictly, with no no-effect phrase
     (`unchanged`, `unaffected`, `not altered`, `no significant`, `similar to wild
     type`, `comparable to`);
   * in a **review** (the paper's `evidence_type`, settled at inclusion), `review`
     only if the sentence names the target and no other organism, names the
     substrate and the product strictly (or one of them, the enzyme and a
     conversion), and carries no hedge, comparison (`unlike`, `in contrast`,
     `whereas`, `compared with`, `equivalent to`, ...) or negation (`lacks`,
     `absent`, `not`, `no`, `without`). A review never earns `in_planta`;
   * `none` otherwise, with the reason recorded.

   **Strict naming** treats a hyphen as part of the token: "OPDA" is not named by
   "dn-OPDA", "JA" not by "JA-Ile". A cofactor (ATP, NAD(P)H, CoA, ...) or a free
   amino acid (Ile, Gln, ...) never counts for positive credit: a sentence naming
   them is a buffer recipe as often as a result. The marker lists are loose
   matching on purpose; positive credit only shrinks.
4. **One state for the claim**, the order of the old classifier kept:
   1. **inference** — any `inference` sentence, or `species_inferred` true:
      `inferred_only`, even when the target is named and cited. This is the
      ortholog-transfer guard and it outranks everything else;
   2. **in vitro** — an `in_vitro` sentence and no `in_planta` one:
      `in_vitro_or_cell_free` (a `review` sentence does not lift it);
   3. **no species** — the model stated none: `species_not_stated`;
   4. **several species** — `multiple_species`;
   5. **another species** — `non_target_species`;
   6. **the target** — `target_species_supported` only with a model-cited species
      span (the verbatim fallback never counts, see
      [`evidence_policy.md`](evidence_policy.md)) one of whose sentences names the
      target plainly (not "Arabidopsis Ortholog" as a table header, not "MpFAD5 is
      orthologous to AtFAD5, an Arabidopsis enzyme", not "unlike Arabidopsis"), AND
      an `in_planta` sentence (grade `in_planta`) or a `review` sentence (grade
      `review`, evidence level `review_article_only`, never primary). Otherwise
      `species_ambiguous`: with a host reason when the only evidence is a host
      system.

**The claim's species is what the model stated for it (B3).** When the model leaves
`species` empty the claim is `species_not_stated`. The paper's screen organisms are
a fact about the paper, used only as context in rule 3 (a sentence must name the
target when the paper names another organism). Base fed them in as the claim's
species: "human LOX homologs" in the 13-LOX abstract made every claim citing it
`multiple_species`.

**A same-paper merge decides on the combined evidence (F-6).** When one paper
yields the same claim identity in two extraction batches, `ClaimStore.register`
unions the two claims' graded sentences, model-cited species spans and flags and
re-runs the aggregation, so the result never depends on batch order. Base copied
the later batch's state over the earlier one.

The rules are the pwml-bio-auditor's adjudication of ORCH-745 F-1. On pilot-003's
16 recorded answers they admit exactly three reactions: GH3.10's jasmonic acid ->
JA-Ile (`in_planta`, on "In flowers and wounded leaves of gh3.10 jar1 knockout lines
JA-Ile was below the detection limit."), and the bryophyte review's OPR3 steps OPDA
-> OPC-8:0 and dn-cis-OPDA -> OPC-6:0 (`review`, on "In Arabidopsis, OPDA and dn-
cis -OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0,
respectively."). The recombinant 13-LOX steps, the purified-enzyme GH3.10 assays,
the JA conjugates "found in wounded leaves" and every Marchantia row stay leads.

### What it still holds back

The rule is fail-closed and loses some correct evidence: one ortholog sentence in a
cited span vetoes the claim; a negative marker in a sentence about another organism
counts; occurrence ("found in wounded leaves"), hedged, introduction, methods,
native-overexpression and complementation statements earn nothing. Such a claim is
a `species_ambiguous` lead with a `missing_species_evidence` frontier item. The
organism list (`species._ORGANISM_WORDS`, plus any abbreviated binomial such as "P.
patens") is a documented ceiling: an organism named only by a word missing from it
is not seen.

## The rule that matters most

> **A shared enzyme name, a shared gene symbol or an ortholog relationship never
> transfers a reaction between species.**

Mouse evidence stays mouse evidence however confidently a model asserts the human
orthologue does the same thing. Such a claim is **retained** — labelled,
auditable, available as a lead — but it is not an established target-species
reaction and cannot enter the accepted graph.

## What happens to a non-admissible claim

It is **never discarded**. `admission.judge` gives it decision status
`candidate` with the reason naming its species state, builds the reaction node
with `admitted=False`, and `graph._quarantine` places it in a labelled quarantine
component whose `quarantine_reason` is e.g. `species:non_target_species`.

Consequences of quarantine:

* it can never reach the primary graph **by chemistry alone** — sharing a
  metabolite with the primary network says nothing about species;
* a quarantine component is never promoted by a bridge — a bridge answers "is it
  connected", not "is it the right species";
* it keeps its full provenance, is shown in the Candidate subgraphs tab with its
  reason, and remains eligible for a `missing_species_evidence` frontier query;
* if a later round produces the same normalized claim **with** real species
  evidence, `graph.add_reaction` lifts it out of quarantine, records a
  `quarantine_lifted` history event, and places it properly. The earlier
  quarantine decision stays in the history chain.

## Where it shows up

* `claims.jsonl` — `species_state`, `species_reason`, `field_evidence.species`, and
  `species_support`: the `grade` (`in_planta`, `review`, `in_vitro`, `inference`,
  `host`, `none`), every graded supporting sentence (`evidence_id`, `sentence`
  index, `text`, `grade`, `reason`), the model-cited `species_evidence_ids`,
  `species_span_ok`, `claimed_inferred`, `target_species` and the
  `paper_evidence_type` it was read under
* `decisions.jsonl` — `species_state` and `species_grade` on every decision
* `primary_graph.json` / `candidate_subgraphs.json` — `species_state` and
  `species_grade` on every reaction node
* `summary.json` — `graph.species_states`, a census of every reaction node
* `report.md` — the census, marking which state is admitted
* Streamlit — a **Species state** filter on the reaction network, and the
  quarantine reason on every candidate component
