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
   Each is split into sentences: no break after "et al.", "Fig." or "e.g."; after a
   single capital only before a capital ("clade A. In opr3 mutant leaves ..." is two
   sentences, "A. thaliana" never breaks).
2. **Supporting sentences** name one of the claim's participants, its enzyme or its
   gene. Names are matched after folding case, whitespace, Unicode hyphens, Greek
   letters and the italic gaps around hyphens ("dn- cis -OPDA"); a spaced em or en
   dash is punctuation, not a hyphen ("JA-Ile — the" stays two tokens). A name
   "X (Y)" is matched as written, as Y ("α-linolenic acid (LeA)" also as "LeA"), and,
   for the negative markers only, as bare X ("ALA (18:3n3)" also as "ALA", so
   "Recombinant enzyme assays ... compared with ALA" still vetoes). Positive credit
   never uses bare X: the note may be what disambiguates it (a bare ALA is also
   5-aminolevulinic acid), the rule `queries.rescue_terms` keeps. A participant or
   enzyme the model asserted but extraction dropped still names supporting
   sentences, so their negative markers still count.
3. **Each supporting sentence gets one grade**, tested in this order:
   * `inference`: `ortholog`, `homolog of`, `by analogy`, `by similarity`,
     `presumably`, `is expected to`, `likely catalyses`, `extrapolat`, `inferred`;
     and a homology transfer without those words: `counterpart`, `closest ...
     relative`, `relative of`, `corresponding ... enzyme / protein / gene /
     reaction`, `equivalent`, `by sequence similarity`, `homologous`, `resembl`,
     `analogous`, `the same conversion / reaction` (not "compared with", which is
     ordinary loss-of-function prose);
   * `host`: `transgenic`, `heterologous`, `ectopic`, `overexpress`, `35S`,
     `expressing`, `expression of`, `transient`, `agroinfiltrat`, `infiltrated`,
     `protoplast`, `transformed`, `expressed in` a plant, `carrying`, `harbouring`,
     `introduced` / `introduction`, `construct`, `promoter`, `cDNA`, `transgene`; and
     complementation when a foreign gene is named (a bryophyte / moss / liverwort /
     foreign / heterologous gene, another species' gene symbol, or another organism).
     The organism is a host for a gene, never performing its own reaction.
     Complementation of a mutant with its own gene is not a host marker, so it is
     not penalised here;
   * `in_vitro`: `in vitro`, `cell-free`, `recombinant`, `purified`, `reconstitut`,
     `expressed in` / `produced in` *E. coli* / bacteria / yeast / *Pichia* / insect
     cells / baculovirus, `enzyme assay`, `assays with`, `incubation`, `incubated
     with`, `Km`, `Vmax`, `kcat` (also spaced, "K m"), `kinetic`, `lysate`, `ex vivo`,
     `homogenate`, `microsom`, a preparation of an organ, organelle or cell (`leaf /
     flower / root / seedling / plant / cell / chloroplast ... extract / homogenate /
     lysate / fraction / assay / disc / preparation / supernatant / powder`),
     `soluble / membrane / stromal / microsomal / crude / desalted / clarified / total
     (protein) extract or fraction`, `isolated / intact chloroplasts / plastids /
     peroxisomes / mitochondria`, an affinity tag (His, Strep, GST, FLAG, 6xHis,
     MBP), `insect-cell`, and cell culture (`cell suspension`, `cell culture`,
     `callus`). An in-planta metabolite measurement made in an extract ("JA-Ile in
     leaf extracts of jar1") is held back too: a recall cost accepted to keep
     cell-free evidence out (see [`known_limitations.md`](known_limitations.md));
   * in a paper not graded a review, `in_planta` **only from loss-of-function
     genetics** (repair 2; rule 5(a), "a conversion stated in an organ or genotype
     context", is removed, see the decision below). Within ONE clause (split at `;`,
     at a colon not followed by a digit, so "OPC-8:0" and "16:1n9" stay whole, and at
     the words *and / but / whereas / while / or*) all four must occur:
     1. a plant loss-of-function genotype: `mutant(s)`, `knockout`, `knock-out`,
        `knockdown`, `loss-of-function`, `null allele`, `T-DNA`, `SALK_`, `RNAi`,
        `amiRNA`, `CRISPR`; never a protein or site-directed mutant (anywhere in the
        sentence: `mutant protein / enzyme / variant / form`, `variant`,
        `site-directed`, `wild-type enzyme / protein`, `active` / `inactive`, or an
        amino-acid substitution such as "R117A", read on the unfolded text);
     2. the claim's OWN enzyme or gene, named strictly; a symbol with another
        species' prefix ("MpGH3A") never counts, even as the claim's own enzyme;
     3. the claim's substrate or product, named strictly (never a cofactor or a free
        amino acid);
     4. a positive effect phrase from a whitelist. Absence forms: `not detected /
        detectable`, `no detectable`, `undetectable`, `below the detection /
        quantification limit`, `absent`, `abolished`, `abrogated`, `eliminated`,
        `lacked / lacking`, `deficient`, `depleted`, and "no <participant> was / were
        detected / found / formed". Change forms: `reduced`, `decreased`, `lower`,
        `diminished`, `impaired`, `compromised`, `accumulated`, `increased`,
        `elevated`, `higher`, each only when no `not / no / n't / never / neither /
        nor / without / none` precedes it in the clause ("were not reduced" fails;
        "No JA-Ile was detected" is an absence form and passes).

     And the sentence carries NONE of the following, anywhere in it:
     - a no-effect phrase: `unchanged`, `unaffected`, `not (significantly) affected /
       changed / altered / different`, `not significantly`, `no significant`, `did not
       differ / change`, `similar / comparable to wild type`, `wild-type levels`,
       `normal levels`, `no change / difference / effect`;
     - a hedge or modality stem (a broad stem set, not a word list): `may`, `might`,
       `could`, `would`, `should`, `probabl`, `possibl`, `likely`, `unlikely`,
       `believ`, `thought`, `suggest`, `propos`, `postulat`, `hypothe`, `presum`,
       `assum`, `apparent`, `seem`, `appear`, `putative`, `potential`, `speculat`,
       `conceivabl`, `consider(ed) to`, `yet to`, `remain`, `unclear`, `unknown`,
       `whether`, `debat`, `controvers`, `elusive`, `unresolved`, `expected`,
       `perhaps`, `anticipated`, `predicted`, `indicat`, `reportedly`, `supports the
       view`, `consistent with`, `points to`, `imply`, `open question`, `cannot (be)
       excluded / ruled out`;
     - an application (`treat`, `appl`, `spray`, `exogenous`, `fed`, `feeding`,
       `supply`, `supplement`, `infiltrat`, `inject`, `dosed`, `administered`,
       `watered`, `soaked`, `dipped`, `perfused`, `deuterated`, `labelled`, ...);
     - a comparison, including bare `as` and `like` (`unlike`, `in contrast (to)`,
       `whereas`, `compared with / to`, `similar to`, `equivalent to`, `resembl`,
       `analogous`, `both`, `homolog(ue)s`, `counterpart`, ...);
     - a citation: `[n]`, `et al`, an author-year, `(12, 13)`, `(ref. 12)`, a
       flattened superscript, `previously reported / shown`, `as (we) reported`, `as
       shown / reported by`, `prior / earlier work`, `has been shown`, `co-workers`,
       `well known`, `it is (well) established that`, `is known to`;
     - a procedure (`was/were grown, sown, harvested, sprayed, treated, extracted,
       analysed, quantified, measured, incubated, used, obtained, purchased, ...`,
       `we employed/used`, `plant material`, `growth conditions`) or a purpose clause
       (`to study`, `to test`, `as a new tool`);
     - another organism, or ANY gene symbol carrying another species' prefix, whether
       or not the sentence also names the target (`species._FOREIGN_GENE`, a documented
       ceiling: `Mp Pp Os Sl Zm Ta Hv Gm Nt Nb Bn Ca Vv Md St Le Cs Sb Br Mt Pt Lj Ph
       Ps Sm Gh` followed by "FAD5", " FAD5", "gh3a" / "fad5" with a digit, or "-aos1");
     - and the passage is not a methods, materials, introduction or background
       section, nor a discussion or conclusion (the section label is searched, so "2.
       Materials and Methods" counts). Discussion and conclusion are interpretive.
       Excluding them is an orchestrator decision the product owner may revisit.
       Abstract, results, figure legends and `body` (an abstract-only paper is indexed
       whole as `body`) keep credit.

     The sentence must name the target itself when its passage or the paper's screen
     organisms name another. A comparison mention of the target ("as in Arabidopsis",
     "like its Arabidopsis homologue", "unlike (in) Arabidopsis", "similar to
     Arabidopsis", "compared with Arabidopsis") is removed before any "names the
     target" test, here, on the review route and in the species-span check: it does
     not name the target as the organism of the statement.
   * in a **review**, `review` only if all of these hold. A review here is the paper's
     `evidence_type`, settled at inclusion. It is also a paper graded `unknown` whose
     EuropePMC lookup failed twice (`evidence_type_verified` False): it may be an
     unrecognised review, so it never earns `in_planta`. The claim's reason says
     "evidence type unverified: EuropePMC lookup failed", and the paper's own grade is
     not rewritten. If the review rule fails, the claim stays a lead. A paper graded
     `unknown` without a failed lookup keeps primary-strength credit, a product-owner
     decision recorded by change set A. The conditions (rule 7, strict, repair 2):
     - the sentence names the target (comparison mentions removed) and no other
       organism, nor any other species' gene symbol;
     - EITHER the substrate and the product, named strictly, are LINKED by a
       conversion verb (`convert`, `reduce`, `produce`, `form`, `synthesise`,
       `catalyse`, `oxidise`, `oxygenate`, `conjugate`, `metabolise`, `yield`,
       `generate`, `cyclise`, `give rise`) standing between them, as "X ... to / into
       Y", "Y ... from X", or a "respectively" sentence. Co-occurrence is not a link:
       "OPDA and OPC-8:0 are produced in wounded leaves" and "OPDA and OPC-8:0 levels
       are reduced after wounding" earn nothing, while "OPDA and dn-cis-OPDA are
       reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0,
       respectively" does;
     - OR one participant named strictly, the claim's own enzyme named strictly, and a
       strict conversion (`convert`, `conversion`, `formed from`, `metabolised to`,
       `produced from`, `synthesised from`). "JAR1 is required for JA-Ile production",
       "JA-Ile formation requires JAR1" and "JAR1 catalyzes the final step of JA-Ile
       biosynthesis" earn nothing;
     - no hedge or modality stem (the set above); no negation: `not`, `no`, `n't`,
       `never`, `neither`, `nor`, `none`, `without`, `cannot`, `unable`, `incapable`,
       `fail*`, `lack*`, `absent`, `absence`, `negligible`, `undetect*`, `rather
       than`, `to be determined / established / shown / demonstrated`; no comparison
       (including bare `as` / `like`), homology, application or procedure.

     A review never earns `in_planta`;
   * `none` otherwise, with the reason recorded.

   **Strict naming** treats a hyphen as part of the token: "OPDA" is not named by
   "dn-OPDA", "JA" not by "JA-Ile". A cofactor (ATP, NAD(P)H, CoA, ...) or a free
   amino acid (Ile, Gln, ...) never counts for positive credit: a sentence naming
   them is a buffer recipe as often as a result. The marker lists are loose
   matching on purpose; positive credit only shrinks.

   **A supplied precursor is held back.** The pilot-003 JA-conjugate rows (JA ->
   JA-Gln / -Asn / -Glu) are a product-owner hinge: whether in-planta formation from
   a supplied precursor counts. Until it is decided, every application synonym above
   refuses credit. Adding to that list holds evidence back; removing from it admits
   evidence. Either is a decision for the product owner, not a list fix.
4. **One state for the claim**, the order of the old classifier kept:
   1. **inference** — any `inference` sentence, or `species_inferred` true:
      `inferred_only`, even when the target is named and cited. This is the
      ortholog-transfer guard and it outranks everything else;
   2. **in vitro** — an `in_vitro` sentence and no `in_planta` one:
      `in_vitro_or_cell_free` (a `review` sentence does not lift it);
   3. **no species** — the model stated none: `species_not_stated`;
   4. **several species** — `multiple_species`;
   5. **another species** — `non_target_species`;
   6. **the target** — `target_species_supported` needs two things:
      - a model-cited species span (the verbatim fallback never counts, see
        [`evidence_policy.md`](evidence_policy.md)) one of whose sentences names the
        target plainly: alone, and not in an ortholog, homology, host-system or
        comparison sentence (not "Arabidopsis Ortholog" as a table header, not
        "MpFAD5 is orthologous to AtFAD5, an Arabidopsis enzyme", not "unlike
        Arabidopsis", not "the pathway resembles that of Arabidopsis thaliana", not
        "Arabidopsis thaliana and Marchantia polymorpha both make jasmonates");
      - an `in_planta` sentence (grade `in_planta`) or a `review` sentence (grade
        `review`, evidence level `review_article_only`, never primary).

      Otherwise `species_ambiguous`, with a host reason when the only evidence is a
      host system.

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
the later batch's state over the earlier one. A claim with no sentence reading (from
a run older than schema 1.2.0) is a floor: its state is kept in the merged reading
(`species_support.floor_state`) and the result never rises above it, however many
batches follow and in any order.

The rules are the pwml-bio-auditor's adjudication of ORCH-745 F-1. Repair round 1
closed the gaps an adversarial review found: in-vitro preparations, negated
conversions, open hedges, homology without "ortholog", hosts without the old host
words, fed precursors, the review route's negations and co-occurrence, foreign gene
symbols, and narrative citations.

**Decision (repair round 2, orchestrator, fail-closed; the product owner may revisit
it).** Round 1's re-review found paraphrases of every closed family still admitted
(52 of 57 new red-team probes): each round closed the quoted sentences and the open
word lists let the next paraphrase through. So round 2 changed the SHAPE of positive
credit rather than adding words:
- rule 5(a) is removed. In a primary paper a bare "X is converted to Y in Arabidopsis
  leaves" is background, a fed-precursor experiment or an in-vitro description at
  least as often as a result, so it earns nothing. Primary-paper credit now comes only
  from loss-of-function genetics with a whitelisted effect phrase, the claim's own
  gene and a participant in one clause;
- the review route needs a LINKED conversion or the claim's enzyme with a strict
  conversion verb, and is vetoed on a broad negation and modality stem set;
- a foreign-prefixed gene symbol always counts as another organism; comparison
  mentions of the target never name it; `fold_text` applies Unicode NFKC first (a PDF
  ligature "ﬁ" no longer defeats "purified" or "infiltration").

A paraphrase that slips past a veto can now cost a lead, but it can earn credit only
if the same sentence is also a complete loss-of-function statement about the claim's
own gene. The recall this costs is listed in
[`known_limitations.md`](known_limitations.md).

On pilot-003's 16 recorded answers the rules admit exactly three reactions:
- GH3.10's jasmonic acid -> JA-Ile (`in_planta`, on "In flowers and wounded leaves of
  gh3.10 jar1 knockout lines JA-Ile was below the detection limit.");
- the bryophyte review's OPR3 steps OPDA -> OPC-8:0 and dn-cis-OPDA -> OPC-6:0
  (`review`, on "In Arabidopsis, OPDA and dn- cis -OPDA are reduced by OPR3 in the
  peroxisomes to produce OPC-8:0 and OPC-6:0, respectively.").

The recombinant 13-LOX steps, the purified-enzyme GH3.10 assays, the JA conjugates
"found in wounded leaves" and every Marchantia row stay leads.

### What it still holds back, and what it cannot see

The rule is fail-closed and loses some correct evidence:
- one ortholog or homology sentence in a cited span vetoes the claim;
- a negative marker in a sentence about another organism counts;
- these statements earn nothing: occurrence ("found in wounded leaves"), hedged,
  introduction, methods, discussion, native-overexpression and host statements, and
  in-planta measurements made in an extract;
- since repair 2, every primary-paper statement that is not loss-of-function genetics
  with a whitelisted effect phrase: a plain stated conversion in an organ ("In
  Arabidopsis leaves, OPDA is converted to OPC-8:0 by OPR3."), a wild-type versus
  mutant contrast phrased as a conversion, a complementation that "restored" a
  conversion, and a knockout result phrased outside the whitelist ("did not
  accumulate", "accumulated to only 10% of wild-type levels").

Such a claim is a `species_ambiguous` (or in vitro) lead with a
`missing_species_evidence` frontier item. The organism list (`species._ORGANISM_WORDS`,
any abbreviated binomial such as "P. patens", and the gene-symbol prefixes) is a
documented ceiling: an organism named only by a word missing from it is not seen.

**Ceiling (N4).** A primary paper's in-planta sentence that names no organism at all
("The aos1 mutant accumulated no 12-OPDA in leaves.") is credited on the strength of
the paper-level screen. It earns no credit only when its passage or the paper's
screen organisms name another organism. So a screen that misses the paper's own
organism (a rice paper whose screen read only *Arabidopsis*, which it cites) admits
the wrong species. No sentence rule can see an organism the sentence does not name.
Closing it would need a per-paper "principal organism" decided at inclusion, which
does not exist today.

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
