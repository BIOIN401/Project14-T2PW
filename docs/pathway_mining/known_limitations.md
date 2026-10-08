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
participant before the effect with no clause opener between them is still read as the
participant's effect. It bites when the true subject FOLLOWS the effect word, so that the
participant before it is the only candidate; words that open no clause in any reading
still let that happen: "In cyp94b3 mutants, JA-Ile accumulated after wounding at the
expense of reduced 12-OH-JA-Ile.", "..., alongside reduced 12-OH-JA-Ile.", "..., plus
reduced 12-OH-JA-Ile." credit the reversed 12-OH-JA-Ile -> JA-Ile (measured at the CS-E
repair-1 tip, `C:/t/bt/m746-e/runs/suspect_r1c`; a wrong admission also needs a
direction-reversed extraction). When the true subject stands before the effect word too
("JA-Ile accumulated after wounding, when 12-OH-JA-Ile was strongly reduced") both sides
are candidates and the form earns nothing.

What the gate now refuses of that residue (ORCH-746 CS-E and its repair 1; each class was
credited at 686ae134, the last seven still at 5eca087b):
- a comma inside a clause: a figure callout ("(Fig. 3A, B)"), a stereo descriptor
  ("(3R,7S)"), a statistics note ("(P < 0.01, Student t-test)"), an inner comma pair (",
  which, in turn, ...", ", with, after wounding, ...", ", with OPDA, however, ...") or a
  second comma-led clause made an open clause read as closed at 686ae134, so the
  participant before it took the effect inside it and reversed and contrary reactions were
  credited (BIO-D-R2-1, RT-D-R2-1, CSD-R2-N1). CS-E credits an effect only when the
  readings of the clause agree (species policy), and every such probe of the CS-D reviews
  is refused;
- the same pairs right after the participant, which the three CS-E readings still read as
  a closed clause (CSE-R0-1): "Leaves of cyp94b3 mutants hyperaccumulated JA-Ile, with
  12-OH-JA-Ile, the major catabolite, reduced by 90%." credited the reversed 12-OH-JA-Ile
  -> JA-Ile at 5eca087b. The fourth reading refuses two or more ", " after the opener;
- an opener written without its comma (CSE-R0-N1): "In cyp94b3 mutants, JA-Ile
  accumulated after wounding with reduced 12-OH-JA-Ile." (the reversed 12-OH-JA-Ile ->
  JA-Ile) and "In opr3 mutants, OPC-8:0 levels were elevated with decreased OPDA." (the
  contrary OPDA -> OPC-8:0). The fifth reading reads a bare "with" as an opener; a bare
  "whereas", "while" or "but" already splits the clause;
- a participle or consequence clause (repair 1 item 9): "In cyp94b3 mutants, JA-Ile
  accumulated after wounding, resulting in reduced 12-OH-JA-Ile levels." (the reversed
  12-OH-JA-Ile -> JA-Ile), ", leading to strongly reduced OPC-8:0 levels" (OPC-8:0 ->
  OPDA), ", accompanied by reduced 12-OH-JA-Ile levels", and "In jar1 mutants, JA-Ile was
  formed, yet 12-OH-JA-Ile was reduced." (contrary, jasmonic acid -> JA-Ile). The sixth
  reading opens a clause at a comma before any "-ing" word or before accompanied /
  followed / coupled / paralleled / associated / yet / thus / thereby / hence / so; after a
  closed relative clause (", which is the substrate of CYP94B3, accumulated after wounding,
  resulting in reduced ...") the fourth reading refuses it;
- "In jar1 mutants, JA-Ile was reduced, but within experimental error." (repair 1 item 10,
  a no-effect veto on `within (the) (experimental / measurement / technical) error /
  variation`);
- a concessive clause (RT-E2-4): "Despite high OPC-8:0 levels in opr3 mutants, JA was
  reduced.", "Although OPC-8:0 accumulated in opr3 mutants, JA levels were strongly
  reduced.", "Even though jar1 mutants accumulated JA-Ile, JA-responsive gene induction
  was reduced." (a no-effect veto on `although / though / despite / in spite of`);
- a participant inside the "accumulated ... less" gap (RT-E2-3): "opr3 mutants
  accumulated OPDA with lower OPC-8:0 levels." credited only the reversed OPC-8:0 -> OPDA;
- "jar1 mutants accumulated nowhere near less JA-Ile after wounding." (CSE-R0-N5);
- a regulator's or an upstream gene's mutant (BIO-D-R2-3, RT-D-R2-3): "Mutants lacking a
  regulator of AOS accumulated no 12-OPDA.", "Mutants with defects upstream of JAR1
  accumulated no JA-Ile." and "Mutants for regulators of JAR1 accumulated no JA-Ile."
  bound the claim's gene; the marker tail's gap words may no longer name another gene's
  role (species policy, rule 5(b) item 2). A gap word outside that list ("Mutants lacking
  a modulator of AOS") still binds.

A participant that only MODIFIES a non-participant head, standing right before a closed
relative clause that names the true subject, is still read as the subject (RT-D-R2-2):
"In jar1 mutants, the precursor of JA-Ile, which is jasmonic acid, accumulated after
wounding." credits the reversed JA-Ile -> jasmonic acid (JAR1), and so do "the pool of
jasmonic acid, that is conjugated to JA-Ile, was reduced" and "the hydroxylation product
of JA-Ile, which is 12-OH-JA-Ile, was undetectable" for their reversed claims. CS-E and
its repair 1 do not close this: all six readings keep the head next to the opener (one
", " follows it) and cut off the participant inside the closed clause. It is refused only
when other words stand between the participant and the opener ("the OPDA reduction
product, which is OPC-8:0, was undetectable"). Real text more often writes the appositive
"JA, the precursor of JA-Ile, accumulated", which names both sides and earns nothing.

Weak decreases are credited as decreases, a product-owner policy item (CSE-R0-N6,
RT-E2-5): "jar1 mutants accumulated only marginally / slightly / somewhat less JA-Ile"
earns `in_planta` exactly as "slightly lower" does; a marginal decrease in one mutant (a
gh3.10 single mutant, which JAR1 compensates) is weak evidence of catalysis. If the
sentence also calls the change non-significant, the no-effect vetoes refuse it.

Extraction can still name a regulator AS the claim's enzyme:
"Knockdown of MYC2 reduced wound-induced JA-Ile in leaves." is a complete rule-5(b)
statement for a MYC2-catalysed claim (RT-D-5); requiring the claim's own gene closes
the regulator-mutant family only against OTHER genes' mutants, never against a
mis-assigned catalyst.

**Precision rests on the paper-level organism screen (R3-N3, RT-D-5, measured).** On
the red team's 358-document real corpus at fd7c737c, 37 unique loss-of-function
sentences survived every claim-independent veto in a credit-bearing section, and 34 of
them were held back only because the paper-level screen supplies the organisms the
document names (E. coli, Listeria, Pseudomonas, Fusarium, mouse and zebrafish results
such as "Delta bioE mutants showed markedly lower bacterial loads"). Re-measured by the
red team at 686ae134 (CS-D repair 2; the same figures at fd7ea63a): 30 unique surviving
sentences, 22 credited with the screen empty (1 of them correct, T1), 20 held by the
screen, 2 credited with it, of which 1 is correct: T1-class (correct) and a dapat
pleiotropy sentence ("Before heat stress, dapat mutant presented a distinct metabolic
state, marked by amino-acid accumulation, altered organic-acid profiles, reduced soluble
sugars and elevated endogenous SA."), credited only because extraction made a token of
that clause the claim's product. At 6454ef98 the screen-empty credits were 29, 2 of them
correct; the correct one lost since is "CWH8 null mutants produced no farnesol."
(PMC12505894), which CS-D repair 1 gave up with the "produced / formed / contained no"
forms: the measured recall cost. The red team re-measured the figures at 5eca087b (ORCH-746
CS-E): 30 / 22 (1 correct) / 20 / 2 (1 correct), identical to 686ae134 (RT-E2-6). Its
prototype of CS-E repair 1's no-effect vetoes left them at 30 / 22 / 20 / 2, and with the
concessive veto at 29 / 21 / 19 / 2: the one credit it drops ("Although no InDels were
detected ...", PMC11946230) was wrong. Repair 1 as shipped was not re-measured; it can only
remove credit. Screen coverage is the dominant control on species correctness; a
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
    6454ef98. CS-D repair 1 added the negated-comparison and delay vetoes (species
    policy), but their 60-character window stopped at the full stop in "gh3.10" and the
    colon in "OPC-8:0", so "The opr3 mutant did not accumulate OPC-8:0 until 6 h after
    wounding." was still credited at fd7ea63a; CS-D repair 2 let the window read past a
    full stop or colon followed by a digit, but any other full stop still ended it, so
    "The opr3 mutant did not accumulate OPC-8:0 (Fig. 3A) until 6 h after wounding.",
    "... (ca. 40 pmol/g) to lower levels than Col-0" and "... JA-Ile (i.e. the bioactive
    conjugate) until 6 h" were credited at 686ae134 (BIO-D-R2-2, CSD-R2-N2). Since ORCH-746
    CS-E the window runs to the end of the sentence or a semicolon, 60 characters at most,
    so it crosses abbreviation full stops ("Fig.", "ca.", "i.e.", "e.g."). Since CS-E
    repair 1 a delay refuses credit anywhere in the sentence, after any absence form:
    `until`, and `before` / `prior to` followed by a number ("In wounded leaves of jar1
    mutants, JA-Ile was not detected until 30 min after wounding.", "In gh3.10 mutant
    flowers, JA-Ile was absent until stage 13."; RT-E2-1), and so do a non-significant
    change ("the difference was not statistically significant", "n.s.", "P > 0.05",
    "P = 0.08", "tended to") and the genotype joined to the wild type ("below the detection
    limit in jar1 mutant and wild-type plants", "in opr3 mutants or in the wild type",
    "alike"; RT-E2-2). Still not seen: a comparison word more than 60 characters after the
    negated verb (CW1: "JA-Ile did not accumulate in wounded rosette leaves of gh3.10
    single mutant plants to significantly lower levels than in Col-0."); a comparison word
    after a semicolon inside a figure or table reference, which still ends the window
    ("gh3.10 mutants did not accumulate JA-Ile (Figure 2B; Table 1) to significantly lower
    levels than Col-0."; BIO-E746-5); a comparison word outside the list ("did not
    accumulate JA-Ile to levels different from those of Col-0", "below"); and a
    non-significance phrased outside the list (repair 1 item 10 adds "within
    (experimental) error / variation": "reduced, but within experimental error" and "...,
    but within the error bars" are refused);
  - CS-D repair 1 (R2): a clause naming both a substrate and a product before its effect
    ("In opr3 mutant leaves, OPC-8:0 formed from OPDA was reduced.", "JA-Ile formation
    from JA was abolished", "13-HPOT formed from alpha-linolenic acid was undetectable")
    is ambiguous and earns nothing. A comma-led relative or adverbial clause (",
    which", ", with", ...) cuts a participant off from the effect when the clause is
    still open at the effect, or when the participant sits inside a clause that closes
    before it (CS-D repair 2; repair 1 had cut off the HEAD before a closed clause and
    credited the participant inside it, inverting polarity on "OPDA, which is the
    precursor of OPC-8:0, was reduced"). The rule reads commas, not grammar. At 686ae134
    any comma after the opener read as the clause closing, so a figure callout ("(Fig.
    3A, B)"), a stereo descriptor ("(3R,7S)"), a statistics note or an inner comma pair
    (", which, in turn, ...") made an open clause look closed: an admission route, not
    only a recall cost (BIO-D-R2-1, RT-D-R2-1). Since ORCH-746 CS-E a form needs every
    reading of the species policy: a parenthetical's commas close nothing in the
    bio-auditor's reading, the last opener decides, a head is kept across a closed clause
    only when the clause opens right after it, two or more ", " after the opener do not
    close it (repair 1), and a bare "with" opens a clause (repair 1). A relative clause
    written without commas, or a participant that modifies another head ("the precursor
    of JA-Ile, which is jasmonic acid, ...") is still read by position (see the residue
    above);
    a clause naming a whole-plant phenotype (growth, fertility, yield, ...) earns
    nothing even when a metabolite is the real measurement; a participant name followed
    by an "-ase" noun is read as an enzyme; "produced / formed / contained no X" no
    longer earns credit (only "accumulated no X");
  - CS-D repair 2's marker tail rejects a gap token that STARTS with *in-* or *of-*,
    because a hyphen is a word boundary: "CRISPR in-frame deletion lines of AOS
    accumulated no 12-OPDA." no longer binds AOS (credited at fd7ea63a; CSD-R2-N3);
  - ORCH-746 CS-E credits an effect only when the 686ae134 reading and both reviewers'
    readings agree, so it may only remove credit. The correct-direction partners of the
    BIO-D-R2-1 / RT-D-R2-1 probes earn nothing, though one prototype or both would have
    credited them (none was credited at 686ae134 either): "In cyp94b3 mutants, JA-Ile
    levels were elevated after wounding, with 12-OH-JA-Ile (Fig. 3A, B) strongly
    reduced." (P1c), "In cyp94b3 mutants, JA-Ile, which is the substrate of CYP94B3,
    accumulated after wounding, with reduced 12-OH-JA-Ile." (E1c), "JA-Ile accumulated in
    cyp94b3 mutants, which, after wounding, contained strongly reduced 12-OH-JA-Ile."
    (F1c), "In the jar1 mutant, JA accumulated after wounding, with JA-Ile (Fig. 2A, B)
    strongly reduced." (P3c), "In the jar1 mutant, JA, which is the substrate of JAR1,
    accumulated to high levels, with reduced JA-Ile." (E2c), "In cyp94b3 mutants, JA-Ile,
    which is hydroxylated by CYP94B3, accumulated in wounded leaves, which contained
    reduced 12-OH-JA-Ile." (E7c), each for its JA-Ile -> 12-OH-JA-Ile or JA -> JA-Ile
    claim, and the ", which, in turn, ..." sentences T2c-T5c ("In cyp94b3 mutants, JA-Ile
    accumulated after wounding, which, in turn, resulted in reduced 12-OH-JA-Ile
    levels.") and A4c ("In opr3 mutants, OPDA accumulated to high levels, which, in turn,
    reduced OPC-8:0 formation."). Correct rows that 686ae134 credited are lost (BIO-E746-6):
    a relative clause whose closing comma is replaced by a callout ("In opr3 mutants,
    OPC-8:0, which is formed from OPDA (Fig. 2A, B) was undetectable."), which the
    bio-auditor's reading takes as open to the form; and a participant followed by a noun
    such as "levels", "content" or "accumulation" before a comma-led relative clause, which
    both reviewers' readings cut off: "In opr3 mutants, OPC-8:0 levels, which peak 1 h
    after wounding, were strongly reduced.", "In jar1 mutants, JA-Ile content, which is
    the main bioactive form, was strongly reduced.", "In aos mutants, 12-OPDA
    accumulation, which is wound-inducible in Col-0, was abolished.", "In aos mutants,
    12-OPDA levels, which were high in wounded Col-0 leaves, were reduced to trace
    amounts.", "In opr3 mutants, OPDA levels, which were low before wounding, increased
    strongly.";
  - ORCH-746 CS-E repair 1 removes more credit, measured on 423 probes against 5eca087b
    (`C:/t/bt/m746-e/probe_matrix_r1.txt`): 101 rows lose `target_species_supported`. All
    but two are the reversed, contrary, non-significant, shared-with-the-wild-type, delay,
    "nowhere near less", participle-clause and regulator-mutant rows the repair targets.
    The two correct-biology rows lost are "In wounded leaves, although in
    opr3 mutants OPDA accumulated, OPC-8:0 was undetectable." (the concessive veto: any
    sentence holding `although`, `though`, `despite` or `in spite of` earns nothing) and
    "JA-Ile was undetectable in jar1 mutants at every time point until 4 h after
    wounding." (the delay veto: any absence in a sentence holding `until`, or `before` /
    `prior to` a number, earns nothing). The same vetoes refuse a correct result that
    shares its sentence with a non-significant one ("OPC-8:0 was undetectable in opr3
    mutants, and the decrease in JA was not significant."), with "tended to", or with the
    mutant and the wild type joined by *and* / *or*; the fourth reading refuses a correct
    head whose relative clause holds an inner pair ("In opr3 mutants, OPC-8:0, which,
    after wounding, rises in Col-0, was undetectable."); the bare-"with" reading refuses a
    participant before a bare "with" with no comma after it (the correct claim on "JA-Ile
    accumulated after wounding with reduced 12-OH-JA-Ile" earned nothing before either).
    The consequence reading (item 9) costs no probe of the matrix: the correct directions
    of its rows earned nothing at 5eca087b either, and an appositive or participle clause
    that closes with a comma keeps its head ("In aos mutants, 12-OPDA, being the first
    cyclic intermediate, was undetectable.", "JA-Ile, following wounding, was
    undetectable"). It refuses a correct head whose participle clause is NOT closed by a
    comma ("In aos mutants, 12-OPDA, being the first cyclic intermediate was
    undetectable.", "In jar1 mutants, JA-Ile, including its methyl ester was undetectable
    after wounding."; measured, `C:/t/bt/m746-e/runs/recall_r1b_*`); any word ending in
    "-ing" after a comma counts, so ", including", ", during" open a clause too.
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
