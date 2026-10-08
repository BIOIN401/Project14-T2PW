# Known limitations

Version 1. Everything here is deliberate or measured, not aspirational.

## Deferred by design

* **No external biochemical database validation.** PathBank, PWC, HMDB, KEGG,
  ChEBI, PubChem and UniProt are not consulted. Every entity and reaction reads
  `{"status": "not_evaluated"}`. A metabolite named two different ways in two
  papers that share no lexical form will be two entities. Attaching a resolver
  is the next obvious piece of work and `t2pw.mining.resolver` is its seam.
* **No PWML.** See [`no_pwml.md`](no_pwml.md).

## Normalization is lexical and within-run

`normalize.comparison_key` folds case, whitespace, dashes, primes, greek letters
and a trailing charge. It will merge `NAD+` with `NAD` and `12-OPDA` with
`12 OPDA`, and it will **not** merge `12-OPDA` with `12-oxophytodienoic acid`
unless a paper states the equivalence and a caller records it through
`link_alias`. Every merge carries its reason, so an auditor can see the
difference between "same normalized key" and "declared alias".

Entity types are inferred from the model's declared type first and an enzyme
suffix heuristic second. A declared type always wins; a disagreement between two
papers about a type is left as information rather than "fixed".

## Species normalization is a short table

`species.py` carries ~30 common names and handles the abbreviated binomial and a
strain tail. An unusual organism normalizes to itself, which is correct but
means `Arabidopsis` and `A. thaliana` fold while some other genus abbreviation
may not. It is not a taxonomy service and never pretends to be.

## The species reader is word lists, and fails closed

`species.grade_sentences` reads each supporting sentence against word lists (see
[`species_policy.md`](species_policy.md)). They are documented ceilings, and their
misses run in both directions.

**The remaining risk class is lexical, and the design now fails closed (CS-B repair 2,
CS-D).** Three review rounds showed that open word lists do not converge: each round
closed the quoted sentences and the next paraphrase of a negation, hedge, applied
compound, cell-free preparation or host got through. Repair 2 therefore narrowed the
shape of positive credit instead of adding words, and CS-D tied the remaining rule to
its own genotype and participant. A paraphrase of a hedge, negation or comparison that
slips past a veto can now only cost a lead, unless the same sentence ALSO forms a
complete loss-of-function statement about the claim's own gene. **That is not true of
an unrecognised organism name**: a plant genus missing from `species._ORGANISM_WORDS`
(and a gene prefix missing from `species._FOREIGN_GENE`) costs an ADMISSION, because
"In <unlisted genus> aos mutants, 12-OPDA was undetectable" is otherwise a complete
loss-of-function sentence (BIO-R2-3; CS-D D4 added the jasmonate corpus's genera, the
list stays a ceiling). What positive credit requires:

* **primary paper (`in_planta`)**, all in ONE clause: a plant loss-of-function
  genotype (`mutant`, `knockout`, `knockdown`, `loss-of-function`, `null allele`,
  `T-DNA`, `SALK_`, `RNAi`, `amiRNA`, `CRISPR`; never a protein or site-directed
  mutant) OF the claim's own enzyme or gene (the symbol beside the marker, never a
  foreign-prefixed symbol); a product that is absent or decreased, or a substrate that
  accumulates, named strictly and apart from the enzyme's own name; no negation before
  the effect. The sentence must carry no no-effect phrase, hedge or modality stem,
  application, comparison (bare `as` / `like` included), in-vitro / host / inference /
  homology marker, protein or transcript measure, citation, procedure, purpose clause,
  other organism or foreign gene, and must not sit in a methods, introduction,
  background, discussion or conclusion passage;
* **review (`review`, `review_article_only`, never primary)**: the target named
  (comparison mentions removed) and no other organism or foreign gene, and either
  substrate and product LINKED by a conversion verb ("X ... to / into Y", "Y ... from
  X", "respectively"), or one participant plus the claim's enzyme with a strict
  conversion verb; no hedge, negation, comparison, homology, application or procedure.

The lexical residue that remains: a hedge, negation or comparison phrased with a stem
outside the sets ("ostensibly abolished"), or an application phrased outside the list
("seedlings given jasmonic acid accumulated no JA-Ile", "upon addition of jasmonic acid
..., no JA-Ile was formed": code review N-2, not in CS-D's scope), inside a sentence that otherwise meets the
loss-of-function rule, would still earn credit. So would a double mutant that carries
the claim's gene beside the gene that explains the result ("the aos jar1 double mutant
... 12-OPDA undetectable" for a JAR1 claim), which is a mis-extraction the sentence rule
cannot see. The effect's subject is chosen by position, not grammar (species policy,
rule 5(b) item 4): a sentence whose real subject is something else and which names the
participant before the effect with nothing but plain words between them is still read as
the participant's effect. The same holds when extraction names a regulator AS the claim's enzyme:
"Knockdown of MYC2 reduced wound-induced JA-Ile in leaves." is a complete rule-5(b)
statement for a MYC2-catalysed claim (RT-D-5); requiring the claim's own gene closes
the regulator-mutant family only against OTHER genes' mutants, never against a
mis-assigned catalyst.

**Precision rests on the paper-level organism screen (R3-N3, RT-D-5, measured).** On
the red team's 358-document real corpus at fd7c737c, 37 unique loss-of-function
sentences survived every claim-independent veto in a credit-bearing section, and 34 of
them were held back only because the paper-level screen supplies the organisms the
document names (E. coli, Listeria, Pseudomonas, Fusarium, mouse and zebrafish results
such as "Delta bioE mutants showed markedly lower bacterial loads"). Re-measured at
6454ef98 (CS-D): 31 unique surviving sentences, 29 credited with the screen empty, 27 held
by the screen, 2 credited with it: T1-class (correct) and a dapat pleiotropy sentence
("Before heat stress, dapat mutant presented a distinct metabolic state, marked by
amino-acid accumulation, altered organic-acid profiles, reduced soluble sugars and
elevated endogenous SA."), credited only because extraction made a token of that clause
the claim's product. Screen coverage is the dominant control on species correctness; a
per-paper principal organism decided at inclusion (ceiling N4) is the only thing that
moves it.

* **Recall cost of removing rule 5(a), accepted.** These now earn nothing, and such a
  reaction stays a `species_ambiguous` lead for a `missing_species_evidence` query:
  - a plain stated conversion in an organ or genotype context in a primary paper
    ("In Arabidopsis leaves, OPDA is converted to OPC-8:0 by OPR3.", "Jasmonic acid
    is converted to JA-Ile in wild-type but not in jar1 mutant leaves.", "In
    wild-type leaves, OPDA was converted to OPC-8:0, but not in opr3 mutants.");
  - complementation that "restored" a conversion (native genes included);
  - any otherwise-qualifying sentence with a bare `as` or `like` ("as described",
    "like JAR1"), `both`, or a broad modality stem (`indicate`, `remain`, `expected`,
    `should`, ...);
  - a gene-symbol prefix false positive: a word that starts with a listed prefix and
    continues in capitals or in lower case with a digit ("CaMV 35S", "StAR", a glued
    "Table1") reads as a foreign gene and vetoes the sentence;
  - the pilot-003 fixture corpus and the test abstracts were rewritten to state their
    results by loss-of-function genetics (`tests/helpers_mining.py`); a live corpus
    whose primary papers report conversions only as stated conversions will admit
    less until the product owner revisits rule 5(a).
* **Recall cost of the effect rule (CS-D, BIO-R2-4, R3-N4), accepted.** Correct
  knockout prose that earns nothing:
  - E1 "JA-Ile remained undetectable in jar1 mutants." (the `remain` modality stem);
  - E2 "In jar1 mutants JA-Ile was undetectable, as was 12-OH-JA-Ile." (bare `as`);
  - E3 "The active jasmonate JA-Ile was undetectable in gh3.10 jar1 knockout lines."
    (`active` reads as a protein mutant);
  - E4 "In aos mutants and aoc mutants, 12-OPDA was undetectable." (the clause split
    at *and* leaves the effect without its genotype);
  - "JA-Ile accumulation was reduced but not abolished in the jar1 mutant." (the same
    split, and "not abolished" is a no-effect phrase);
  - nominalised or unlisted changes: "only 5% of that in wild-type leaves", "dropped
    below 10% of wild type", "accumulated to only 10% of wild-type levels" ("wild-type
    levels" is a no-effect phrase), bare `lack` ("aos knockout lines lack 12-OPDA");
  - a substrate named before the product in a conversion that was abolished ("conversion
    of OPDA to OPC-8:0 was abolished in opr3 mutants": the effect's subject is the
    substrate), and an allele written with a hyphen ("jar1-1", "opr3-1"), which does not
    strictly name JAR1 / OPR3;
  - any sentence with `protein`, `abundance`, `transcript`, `mRNA`, `expression`,
    `expressed`, `activity` or `similar`, and "equivalent" or "the corresponding gene"
    (homology markers, BIO-R1-7);
  - D1 recovered the commonest knockout phrasing: "did not / failed to accumulate",
    "no longer detected" now earn credit. They are NOT immune to no-effect prose: "did
    not accumulate significantly less JA-Ile than the wild type" was credited at
    6454ef98, and CS-D repair 1 added the negated-comparison and delay vetoes (species
    policy) that close that class;
  - CS-D repair 1 (R2): a clause naming both a substrate and a product before its effect
    ("In opr3 mutant leaves, OPC-8:0 formed from OPDA was reduced.", "JA-Ile formation
    from JA was abolished", "13-HPOT formed from alpha-linolenic acid was undetectable")
    is ambiguous and earns nothing; a participant separated from the effect by a
    comma-led relative or adverbial clause (", which", ", with", ...) is not its subject;
    a clause naming a whole-plant phenotype (growth, fertility, yield, ...) earns
    nothing even when a metabolite is the real measurement; a participant name followed
    by an "-ase" noun is read as an enzyme; "produced / formed / contained no X" no
    longer earns credit (only "accumulated no X").
* **Recall cost of the review route (BIO-R2-5, R3-N4c), accepted.** Verb-first review
  prose with a verb outside the strict set earns nothing: "In Arabidopsis, OPR3 reduces
  OPDA to OPC-8:0.", "JAR1 conjugates jasmonic acid with isoleucine to form JA-Ile.",
  "CYP94B1/B3 catalyzes the oxidation of JA-Ile to 12OH-JA-Ile" (the link needs a
  conversion verb BETWEEN the participants, and "oxidation" is not "oxidise"), while
  "OPDA is converted to OPC-8:0 by OPR3" is credited. The pilot's T2/T3 sentence passes
  on its "reduced by OPR3 ... to produce ... respectively" word order.
* **Recall cost, accepted earlier.** A preparation of an organ is cell-free evidence, and
  since CS-D D3 so is any extract or enzyme preparation, so an in-planta metabolite
  measurement made in an extract ("JA-Ile was absent from leaf extracts of jar1") earns
  no credit. So do hedged, discussion, conclusion, introduction and methods statements,
  native complementation phrased with a host word, and any sentence near an ortholog /
  homology / "equivalent" mention.
* **What no sentence rule can see.** A sentence naming no organism is credited on the
  paper-level organism screen; a screen that misses the paper's own organism admits
  the wrong species (species policy, ceiling N4, measured above). An organism or
  gene-symbol prefix missing from `species._ORGANISM_WORDS` / `species._FOREIGN_GENE`
  is not seen, and the magnitude is large: after CS-D's D4 additions the red team found
  10 of 10 unlisted plant binomials credited on an otherwise complete loss-of-function
  sentence (Citrus sinensis, Hevea brasiliensis, Chrysanthemum morifolium, Cucumis
  sativus, Fragaria vesca, Helianthus annuus, Spinacia oleracea, Malus domestica,
  Hibiscus cannabinus, Nelumbo nucifera; RT-D-3). No finite genus list closes this;
  only instrumentation does: a per-paper principal organism decided at inclusion, or
  `paper_observed_organisms` filled from a taxonomy-backed source (EuropePMC / MeSH). An uncited background statement about the target in an abstract or a
  span labelled `body` ("In Arabidopsis, aos knockout lines lacked 12-OPDA." in a
  Marchantia paper) earns primary `in_planta`: true biology, credited to the wrong
  paper (R3-N2). Closing it needs the principal-organism field plus a span-level
  background-versus-result signal that ingest does not emit.
* **A participant's citation is not checked against its text (BIO-745B-2, an
  escalation to the product owner).** A participant kept by `model_citation` is not
  checked against the text of the span it cites. A loss-of-function sentence
  earns credit with ONE strictly-named participant, so the other side of an admitted
  reaction can rest on the model's citation alone (#14's "jasmonic acid", where the
  abstract says "JA"; a wrong substrate on the T1 sentence is credited the same way,
  code review N-3). That is correct biology there; a hallucinated substrate with a
  valid citation would pass the same way, and on the review route a non-chemical token
  ("Recently", "arabidopsis") can act as the linked participant (R3-N5), so the gate's
  precision is bounded by extraction quality. The tightening proposed by the
  bio-auditor (label such a participant `model_citation_unverified` and require every
  participant to be verbatim in the claim's spans or named in the credited sentence)
  awaits a decision.
* **The increase direction is a narrowing of the adjudicated rule (R3-N1).** Rule 5(b)
  as adjudicated credits any change of a named participant; CS-D credits an increase
  only on a substrate and an absence or decrease only on a product. That is an
  orchestrator decision the product owner may revisit (it needs a DECISIONS.md entry
  to become product policy).

## Support across siblings

An admitted reaction's support and evidence level count only its
`target_species_supported` claims, and a later lead never relabels it (evidence
policy, ORCH-745 A3). The same-paper edge is closed (CS-B repair 2, R1-N1: a paper's
batches are all registered before any is judged). Two edges remain:

* A reaction from a read-only starting graph has no decision history, so a mined lead
  of the same identity is recorded as a `candidate` decision. The node keeps
  `starting_graph`, stays admitted (the graph cannot un-admit), and the lead is listed
  in `non_supporting_claims`. `summary.claims.by_latest_status` then counts a
  candidate for an identity the graph shows admitted.
* Conflict detection still compares every sibling, a lead's enzyme included. A
  disagreement with an in-vitro claim clears the attribute on the admitted node,
  which overstates the disagreement and never the support.

**The same-paper union widens the inference veto (BIO-R2-6), accepted.** Because every
batch of a paper is registered before any claim is judged, a claim is judged on the
union of every batch's cited sentences, and ONE analogy or ortholog sentence anywhere in
that union makes it `inferred_only`, even beside a clean knockout result in another
batch. A live paper that cites the same reaction from two passages is more likely to
carry one, so recall depends on retrieval's span selection. The adjudication keeps the
veto at sentence level and per paper, deliberately.

## Conflict detection is attribute-level and literal

Only `enzyme`, `direction`, `reversible` and `compartment` are compared, and
only by normalized string equality. "Chloroplast" and "plastid" will read as a
conflict; "AOS" and "allene oxide synthase" will too, unless an alias link was
recorded. That errs toward showing a disagreement rather than hiding one, which
is the right direction, but it will over-report.

## Retrieval quality depends on the embeddings service

With no embedding endpoint, `Embedder` falls back to deterministic lexical
vectors and retrieval gets materially worse. That is visible, never silent —
`embedding_mode`, a preflight check, a summary warning and a report line — and
`PATHWAY_MINING_REQUIRE_EMBEDDINGS=true` turns it into a preflight failure. But
an offline run's thin results are a retrieval limitation, not a statement about
the literature.

## Model non-determinism

Identical legs give materially different draws at temperature 0 on this stack.
Two mining runs over the same corpus can produce different graphs. Re-run before
calling a single-run difference a regression. The **policy** is deterministic:
given the same claims, the same decisions and the same placement follow.

## Full-text availability

`rag.acquire.fetch_full_text` resolves a PMCID and fetches EuropePMC full text.
A paper with no open full text contributes its abstract only, which is recorded
(`full_text_chars`) but sharply limits what can be extracted. DOI-only and
URL-only references frequently resolve to an abstract.

## Index hygiene knows the pilots' layouts

What indexing keeps out ([`evidence_policy.md`](evidence_policy.md) § what text can be
evidence) was fit on 35 EuropePMC full texts: the seven pilots 002 and 003 fetched and the 28
committed under `runs/2026-07-27_*` at f9556288. It was then checked, read only, on the 200
unique EuropePMC full texts in the project's acquire cache (`data/rag_index/acquire_cache/
fulltext`), indexing each with no abstract and no passage cap at 6ceef736 and at the repair-2
tip. Text kept at 6ceef736 and not at the tip, on 8 of the 200: reference entries (4 texts),
ACS processing flags (4), a data-availability and acknowledgments paragraph before a
headingless list (PMC10055903, 945 characters) and a Supporting Information file list
(PMC13418202, 428 characters). No body prose. Front matter
is recognised by EuropePMC's processing flags only: a PDF upload's title page, or a source
without those flags, keeps its front matter. Publisher notes are recognised by three endings:
the two MDPI notes in the pilot texts and Frontiers' standard note. A reference list needs a
`References` / `Literature Cited` / `Bibliography` heading or, without one, three authors with
initials in a citation-dense window to be cut as a region; otherwise only a chunk that is
citation-dense as a whole is dropped, and a chunk that is only partly bibliography can keep
that part. The density bar (5 marks per 1,000 characters) separates every reference chunk (13
or more) from every body window (1.6 at most) in the 35 texts. A list ends at the next figure
or table label OR mention after its run of citation marks, as at 6ceef736 (repair 1's
label-only rule was reverted: it dropped body prose). So a mention can end a list early and
leave a fragment outside it (the pepper paper's "(Supplemental Table S3) , and the raw read
data ..." sentence), and a table whose "References" column holds author-year cells still opens
a false list that runs to the table's own mention: PMC8510960 loses that table's rows and the
text up to "(Table 1 )", exactly as at 6ceef736.
The front matter can hold an abstract, a significance statement or highlights; only the
record's abstract survives as `abstract#0` (the metadata lookup fills an empty one), so a
PNAS significance statement is not indexed. A record with no abstract takes the front
matter's "Abstract ..." passage instead, which needs that heading: PMC8508710's front matter
has none. Figure and table legends printed after a reference list that sits in the last 45 %
of the text are not indexed at all: rag's chunker drops everything after that heading before
mining sees the chunks. The headingless-list rule needs three authors with initials in its
first 1,500 characters and backs over at most 300 characters of the first entry's authors, so
a list opening with a corporate author (Cureus: "1. World Health Organization ...") can keep
its first entry. "Survive" in these rules means the text stays retrievable: a trimmed chunk
gets a new evidence id, and a figure snippet contained in a body chunk is deduplicated (A4).

## A review must say it is one

**Orchestrator interim position pending the product owner (A1-R1-UNKNOWN-WITHOUT-FAILED-LOOKUP,
recorded per the ORCH-745 CS-A repair-2 brief; no DECISIONS.md entry exists).** A paper graded `unknown` WITHOUT a failed lookup remains eligible for
primary-strength credit: an offline run or an upload (no lookup is made), a paper EuropePMC has
no record of (DOI- or URL-only), a bare PMCID with no title (the subject heading has nothing to
anchor to), a PubMed record declaring only "Journal Article", and subject headings the rule does
not read (e.g. "Systematic Review and Meta-Analysis", "Review Paper"). Only a lookup that failed
twice marks a paper `evidence_type_verified: false`, the case a gate may fail closed on.

Without a declared review type (NCBI, or EuropePMC's metadata lookup, which an offline run
never makes), a review is recognised only by its JATS subject heading (which needs the record's
title to anchor it, so a bare PMCID with no title has none), a title marker or a
self-referential sentence ("this review", "here we review", ...). A review that never describes itself in those words
stays `unknown`, and its reactions are graded `single_paper`. Two such unrecognised reviews,
or one and a primary paper, supporting one reaction still grade
`multiple_independent_papers`: `admission.evidence_level` counts them as primary papers. (A
review recognised as one never counts toward independence since ORCH-745 A3; see
[`evidence_policy.md`](evidence_policy.md).)

## Uploaded PDFs

Text extraction goes through the existing `t2pw.extraction.pdf_parser`. A
scanned PDF without OCR yields little; the failure is recorded in the paper's
`notes` rather than raised. Page-level locators are not currently propagated
from the PDF parser into evidence spans, so an uploaded PDF's spans carry a
section and paragraph but `page` stays null.

## Sequential by construction

One model call at a time, one paper indexed at a time. A large run is slow. No
parallelism is introduced around the shared caches, the vector store or the
budget ledger because the architecture provides no proven safe mechanism for it.

## Streamlit blocks during a run

The page runs the controller synchronously with a progress bar. There is no
background job runner, so the browser tab is occupied for the duration.
Cancelling mid-run from the UI is therefore limited to interrupting the process;
the CLI's `Ctrl-C` path and `controller.cancel()` are the supported routes, and
both leave a resumable run.

## Cost is an estimate

Priced from configured per-million-token rates, before any bill exists. It does
not reflect provider discounts, cached-token pricing or minimum charges, and it
says so in `costs.json`, in `report.md` and in the CLI status output.

## The provider is unpinned and unreported (production follow-up)

`t2pw.llm.client` is shared with production and frozen for this work, so mining can only
mitigate what it does:

* **Provider routing is unpinned.** `chat_detailed` sends no OpenRouter `provider`
  preference, so identical requests can be served by different providers. Pilot-003 had 7
  thin answers of 16 (no field cited; two with schema placeholders as values). Three took
  150-210 output tokens, four took 5k-24k, and all seven share one alphabetically sorted key
  set: the shape follows the provider route, not the amount of reasoning. Mining therefore
  detects it structurally, re-asks a thin answer once (at most one extra call per thin answer)
  and marks what it kept (`answer_quality`); it cannot choose the route. With no usage
  reported, the token counts are estimates from the visible text, so the structural rule
  decides alone; a future token rule needs an explicit per-attempt `usage_reported` flag.
* **The provider is never reported.** `CompletionDiagnostics` has no provider field, so
  `ModelCallRecord.provider` reads `not_reported_by_llm_client` on every live call.
* **The reasoning setting is never sent.** `MiningModelConfig.reasoning` is recorded but
  `chat_detailed` takes no reasoning parameter, so the provider default applies.
* **`response_hash` is partial.** It is set only for a reply with text, as the hash of the
  last request of a call; an empty or content-filtered reply, and every re-sent request in
  the diagnostics `attempt_log`, carries none.
* **Usage is cumulative.** Token counts come from a process-wide counter, so mining can
  measure each attempt but not each provider request inside it.

The fix belongs in `llm/client.py`: pin (or at least report) the provider, pass the
reasoning setting, and hash every response.

## Frontier heuristics are untuned

Twelve frontier kinds with hand-set priorities. They have been exercised against
a synthetic fixture, not tuned against real corpora. Expect the bridge and
conflict kinds to be the most valuable and the cofactor kind to be mostly noise.
The species reserve (a quarter of the slots, each at the earlier of its priority
position or the next fourth place) and the 0-3
on-route relevance score are equally hand-set: lexical name matching, no taxonomy,
and a carried item keeps the score it was generated with.

## No live evaluation

No live mining run has been executed against real literature as part of this
work — the whole test suite is fixture-driven and spends no API credit. The
system is *verified* (226 focused tests plus an eight-point integration test) but
not *evaluated*: nobody has yet judged whether the reactions it mines from real
papers are good ones. That is the obvious next step and it needs explicit cost
authorization.

## Not merged

`experiment/pathway-mining` is not merged anywhere and is not proposed for
merge. If it ever is, see the checklist at the end of
[`feature_flag.md`](feature_flag.md).
