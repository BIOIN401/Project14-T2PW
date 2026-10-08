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
     `callus`), and since CS-D D3 any `extract(s)`, `enzyme preparation(s)` or `crude
     enzyme` ("In crude enzyme extracts of jar1 mutant leaves ...", "abolished in extracts
     of jar1 mutant leaves"). An in-planta metabolite measurement made in an extract
     ("JA-Ile in leaf extracts of jar1") is held back too: a recall cost accepted to keep
     cell-free evidence out (see [`known_limitations.md`](known_limitations.md));
   * in a paper not graded a review, `in_planta` **only from loss-of-function
     genetics** (repair 2; rule 5(a), "a conversion stated in an organ or genotype
     context", is removed, see the decision below). Within ONE clause (split at `;`,
     at a colon not followed by a digit, so "OPC-8:0" and "16:1n9" stay whole, and at
     the words *and / but / whereas / while / or*) all of these must hold
     (`species._bound`, `species._effect`, CS-D):
     1. a plant loss-of-function genotype: `mutant(s)`, `knockout`, `knock-out`,
        `knockdown`, `loss-of-function`, `null allele`, `T-DNA`, `SALK_`, `RNAi`,
        `amiRNA`, `CRISPR`; never a protein or site-directed mutant (anywhere in the
        sentence: `mutant protein / enzyme / variant / form`, `variant`,
        `site-directed`, `wild-type enzyme / protein`, `active` / `inactive`, or an
        amino-acid substitution such as "R117A", read on the unfolded text);
     2. the genotype is OF the claim's own enzyme or gene, named strictly: the symbol
        sits at most three tokens before the marker, with no comma between ("gh3.10
        jar1 knockout lines", "the jar1 gh3.10 double mutant", "lox2 (At3g45140)
        knockout lines", "opr3 mutant leaves"), or after it as `_MARKER_OF` reads it: at
        most three tokens, none of them itself *of* or *in* (CS-D repair 2) or a word
        naming another gene's role: `regulat...`, `upstream`, `downstream`, `gene(s)`,
        `factor(s)`, `activator(s)`, `repressor(s)`, `defect(s)`, `target(s)` (ORCH-746
        CS-E repair 1), `suppress...`, `enhanc...`, `inducer(s)`, `co-activator(s)`,
        `modulat...`, `mediator(s)`, `modifier(s)` (repair 2), then *of* / *in*, then an
        optional *the* ("knockout of GH3.10", "T-DNA insertion lines of AOS", "Mutants
        deficient in GH3.10", "The SALK_034543 insertion line of GH3.10"; not "Mutants in
        genes upstream of JAR1", "Knockouts of transcriptional targets of JAR1", "Mutants
        in a regulator of JAR1", "Mutants lacking a regulator of AOS", "Mutants with
        defects upstream of JAR1", "Mutants for regulators of JAR1", "Mutants lacking a
        suppressor / an inducer / a co-activator / a modulator of AOS"). A gene followed by
        `pathway`, `background`, `cluster`, `family`, `signalling` or `network` is not the
        genotype ("Mutants in the JAR1 pathway", "the myc2 mutant in the JAR1
        background", "JAR1 pathway mutants"; CS-D repair 1), nor is a gene followed by a
        role noun, `activator(s)`, `suppressor(s)`, `enhancer(s)`, `regulator(s)`,
        `inducer(s)`, `modifier(s)`, `target(s)`, before or after the marker ("In the jar1
        suppressor mutant", "In jar1 enhancer mutants", "Knockout of the AOS activator";
        `species._CONTAINER`, repair 2). A different gene's mutant
        earns nothing ("arc mutants ... allene oxide cyclase", "the myc2 mutant ... by
        JAR1"). A symbol with another species' prefix ("MpGH3A") never counts, even as
        the claim's own enzyme;
     3. the claim's substrate or product, named strictly (never a cofactor or a free
        amino acid) AFTER the claim's own enzyme and gene names are blanked from the
        clause, so an enzyme named after its substrate ("allene oxide cyclase", "OPDA
        reductase 3", "sterol methyltransferase") never supplies the participant; nor
        does a name followed by an enzyme noun in "-ase" ("OPDA reductase levels", "the
        other allene oxide cyclases"; CS-D repair 1);
     4. an effect on the RIGHT participant (`species._effect`, CS-D repair 1). The
        effect form's subject candidates are, in this order: the participants inside the
        form ("no JA-Ile was detected"); else every participant before it in the clause
        that a comma-led relative or adverbial clause (", which", ", that", ", who", ",
        whose", ", with", ", whereas", ", while", ", although", ", though", ", but", ",
        and", or a ";") does not cut off. Since ORCH-746 CS-E the cut-off is read several
        ways, and a form earns its effect only when ALL the readings of
        `species._READINGS` credit it (`species._effect`; the reason recorded is the first
        reading's). Six readings ship (three since CS-E, three appended in repair 1):
        - the 686ae134 reading (`species._excluded`, CS-D repair 2, unchanged): a
          participant is cut off when the clause that follows it is still OPEN at the
          form (no comma at all between the first opener and the form: "JA-Gln
          accumulated ..., which had reduced JA-Ile", "OPC-8:0 levels were elevated,
          with decreased OPDA"), or when it sits INSIDE such a clause that closes before
          the form; any comma after the opener reads as the close, so "(Fig. 3A, B)",
          "(3R,7S)", ", which, after wounding, ..." or a second comma-led clause makes an
          open clause look closed;
        - the bio-auditor's reading (`species._excluded_blanked`): parentheticals are
          blanked first, so a parenthetical's commas ("(Fig. 3A, B)", "(3R,7S)", "(P <
          0.01, t-test)") never close a clause, and only a comma followed by a space
          closes one; the LAST opener in the gap decides whether a clause is still open
          at the form; a head is kept across a closed clause only in the shape "X,
          <opener> ..., <form>", with one opener right after X, and any other gap that
          holds an opener cuts the participant off;
        - the red team's reading (`species._excluded_adjacent`): the 686ae134 reading,
          except that a head is kept across a closed clause only when the clause opens
          right after it ("JA-Ile, which ..., accumulated"); a clause that opens after
          other words ("JA-Ile accumulated after wounding, which, in turn, ...", "OPC-8:0
          levels, which ...") cuts the head off whatever commas it holds;
        - the code reviewer's reading (`species._excluded_unambiguous_close`, repair 1,
          CSE-R0-1): the 686ae134 reading, plus, with parentheticals blanked, a
          participant is cut off when TWO or more ", " follow the opener before the form.
          One ", " closes a relative clause that modifies the head ("JA-Ile, which is the
          substrate of CYP94B3, accumulated"); two make the close ambiguous, so it refuses
          an appositive pair or an inner pair inside a clause still open at the form
          ("Leaves of cyp94b3 mutants hyperaccumulated JA-Ile, with 12-OH-JA-Ile, the
          major catabolite, reduced by 90%.", "..., which, in turn, resulted in reduced
          12-OH-JA-Ile levels."), where the first three readings all kept the head;
        - the bare-"with" reading (`species._excluded_bare_with`, repair 1, CSE-R0-N1):
          the 686ae134 reading in which "with" written without its comma also opens a
          clause, and a participant is also cut off when a bare "with" in its gap has no
          comma after it. It refuses "In cyp94b3 mutants, JA-Ile accumulated after wounding
          with reduced 12-OH-JA-Ile." (the reversed 12-OH-JA-Ile -> JA-Ile), "In opr3
          mutants, OPC-8:0 levels were elevated with decreased OPDA." (contrary) and the
          same shape after a closed relative clause ("JA-Ile, which is the substrate of
          CYP94B3 (Fig. 3A, B), accumulated after wounding with reduced 12-OH-JA-Ile");
        - the consequence reading (`species._excluded_consequence`, repair 1 item 9): the
          686ae134 reading in which a comma before a participle or consequence word also
          opens a clause: `, <any word ending in -ing>`, `, accompanied`, `, followed`,
          `, coupled`, `, paralleled`, `, associated`, `, yet`, `, thus`, `, thereby`,
          `, hence`, `, so`. It refuses "In cyp94b3 mutants, JA-Ile accumulated after
          wounding, resulting in reduced 12-OH-JA-Ile levels." (the reversed 12-OH-JA-Ile
          -> JA-Ile), ", leading to strongly reduced OPC-8:0 levels", ", accompanied by
          reduced 12-OH-JA-Ile levels" and "In jar1 mutants, JA-Ile was formed, yet
          12-OH-JA-Ile was reduced." (contrary).

        Each prototype misreads a shape the other reads correctly: the red team's keeps
        the head in "JA-Ile, which is the substrate of CYP94B3, accumulated after
        wounding, with reduced 12-OH-JA-Ile" (a second clause is open at the form); the
        bio-auditor's reads "OPC-8:0, which is formed from OPDA (Fig. 2A, B) was
        undetectable", whose closing comma is missing, as open to the form, and credits
        correct-direction rows the 686ae134 reading refuses. Agreement is the fail-closed
        choice, and because the 686ae134 reading is one of them, nothing earns credit that
        686ae134 did not credit; a reading appended to the tuple can only remove credit
        (the price is in [`known_limitations.md`](known_limitations.md)). All six cut
        OPDA off in "In opr3 mutants, OPC-8:0, which is formed from OPDA, was undetectable"
        and credit the head OPC-8:0; all six keep OPDA as the subject of "OPDA, which is
        the precursor of OPC-8:0, was reduced", a substrate going down, nothing.
        An appositive ("JA-Ile, the bioactive form, was absent") keeps its subject. When
        participants before the form are candidates, a participant right after it, an
        attributive "reduced 12-OH-JA-Ile" at most two tokens away with no comma between,
        joins them (ORCH-746 CS-E repair 2, the attributive-participant rule): "In
        cyp94b3 mutants, JA-Ile accumulated after wounding at the expense of / alongside /
        plus reduced 12-OH-JA-Ile." names both sides and earns nothing. Else the
        participants right after the form, at most three tokens
        away with no comma (two for a forward form). A forward form takes only the
        participants after it. If the candidates name BOTH a substrate and a product the
        subject is ambiguous and the form earns nothing ("the JA conjugate JA-Ile
        accumulated", "the JA-Ile precursor JA was reduced", "the OPDA metabolite OPC-8:0
        accumulated", and also "13-HPOT formed from alpha-linolenic acid was undetectable"
        or "JA-Ile formation from JA was abolished", a recall cost). A clause naming a
        whole-plant phenotype (`growth`, `fertility`, `fertile`, `resistance`,
        `susceptibility`, `tolerance`, `senescence`, `yield`, `biomass`, `germination`,
        `flowering`, `development`, `phenotype`) earns nothing: the measured subject is
        the phenotype ("Root growth of the jar1 mutant was reduced on medium containing
        JA-Ile"). The effect credits only:
        - an absence or decrease of a **product**. Absence forms: `not detected /
          detectable`, `no detectable`, `undetectable`, `below the detection /
          quantification limit`, `absent`, `abolished`, `abrogated`, `eliminated`,
          `depleted`, "no <product> was / were detected / found / formed", `did / does /
          do not accumulate / form / produce`, `failed / fails to accumulate / form /
          produce`, `no longer detected / accumulated / formed`; forward: `lacked /
          lacking <product>`, `deficient in <product>`, `accumulated no <product>` (never
          the genotype: "mutants lacking GH3.10"; CS-D repair 1 dropped the unbriefed
          "produced / formed / contained no"). Decrease forms: `reduced`, `decreased`,
          `lower`, `diminished`, `impaired`, `compromised`, and `accumulated (up to two
          words) less / lower / reduced / decreased` ("aos mutants accumulated less
          12-OPDA", "accumulated much less"), where neither gap word is `not`, `no`,
          `hardly`, `scarcely`, `barely`, `never` or `nowhere` (CS-D repair 2: "accumulated
          hardly less", "accumulated not less" earn nothing; ORCH-746 CS-E repair 1:
          "accumulated nowhere near less" neither), and where no claim participant starts
          inside the gap (repair 1, RT-E2-3: "opr3 mutants accumulated OPDA with lower
          OPC-8:0 levels." does not read OPDA as the decreased one, and earns nothing);
        - an accumulation or increase of a **substrate**: `accumulated`, `increased`,
          `elevated`, `higher` ("In opr3 mutants, OPDA accumulated in wounded leaves."
          supports OPDA -> OPC-8:0), and only in a clause with no decrease or absence
          word (`less`, `fewer`, `lesser`, `lower`, `reduced`, `decreased`, `diminished`,
          `no`, `weakly`, `slightly`, `marginally`, or any absence or decrease form): "The
          jar1 mutant accumulated less JA" and "OPDA accumulated to lower levels" are a
          substrate going DOWN.

        The contrary combinations (a product that rises, a substrate that falls) earn
        nothing, as far as the positional subject rule above identifies the participant
        (it reads commas and word order, not grammar). A negation (`not / no / n't / never / neither / nor / without / none /
        hardly / scarcely / barely`) before an effect form in the clause refuses it ("was
        not abolished", "never abolished", "not deficient in", "hardly reduced"); the
        forms that START at their negation ("not detected", "no detectable", "No JA-Ile
        was detected", "did not accumulate") are the absence statement itself, unless a
        comparison or a delay follows (below).

     And the sentence carries NONE of the following, anywhere in it:
     - a no-effect phrase: `unchanged`, `unaffected`, `not (significantly) affected /
       changed / altered / different`, `not significantly`, `no significant`, `did not
       differ / change`, `similar to wild type`, `wild-type levels`, `normal levels`,
       `no change / difference / effect`, and since CS-D bare `comparable to` (the
       adjudicated rule; repair 2 had narrowed it to "comparable to wild type", a
       weakening), `indistinguishable`, `normally`, `residual`, `similar`, `still
       present / accumulated / detected / formed / produced / increased / elevated`,
       `not (fully / completely / entirely) abolished / absent / eliminated / abrogated
       / depleted / deficient / lacking`; and since CS-D repair 1 a negated comparison or
       delay: `no more / less / greater / fewer / higher / lower / lesser`, and `did /
       does / do not` or `failed / fails to` followed (the window runs to the end of the
       sentence or a semicolon, 60 characters at most: it reads past "OPC-8:0",
       "gh3.10", "(Fig. 3A)", "ca." and "i.e."; ORCH-746 CS-E) by `more /
       less / lesser / lower / higher / fewer / greater / further / again / than / until /
       significantly` ("did not accumulate significantly less JA-Ile than the wild
       type", "did not accumulate until 6 h", "failed to accumulate more JA-Ile",
       "accumulated no more JA-Ile than"); a negation after the effect (`neither ... nor`,
       `none of`); and since ORCH-746 CS-E repair 1:
       - a delay written with any absence form (RT-E2-1): `until` anywhere in the sentence,
         and `before` / `prior to` followed by a number ("JA-Ile was not detected until 30
         min after wounding", "was absent until stage 13", "did not accumulate JA-Ile
         before 1 h");
       - a non-significant change (RT-E2-2): `not (statistically) significant`,
         `not statistically`, `non-significant`, `nonsignificant`, `insignificant`,
         `n.s.`, `P >` or `P ≥` a decimal, `P =` 0.05 to 0.99, `tend(s / ed) to`, and
         `within (the) (experimental / measurement / technical) error / variation`
         ("reduced, but within experimental error"; repair 1 item 10);
       - the genotype joined to the wild type (RT-E2-2): `mutant(s)`, `line(s)`,
         `knockout(s)` or `plant(s)` followed by *and* / *or*, an optional *also*, *in*
         and *the*, then `wild type`, `wild-type`, `Col-0` or `WT` ("below the detection
         limit in jar1 mutant and wild-type plants", "not detected in opr3 mutants or in
         the wild type"), and `alike`;
       - a concessive (RT-E2-4): `although`, `though`, `despite`, `in spite of` ("Despite
         high OPC-8:0 levels in opr3 mutants, JA was reduced.", "Even though jar1 mutants
         accumulated JA-Ile, JA-responsive gene induction was reduced."), and since repair
         2 (RT-R1-3) a sentence-initial `while`, `whilst` anywhere, `even if`,
         `notwithstanding`, `regardless of`, `albeit`;

       and since ORCH-746 CS-E repair 2 the delay and no-difference classes are closed by
       SHAPE (the union of the three round-1 reviewers' prototypes), so an ordinary
       paraphrase falls inside a shape rather than beside a list:
       - **a sentence naming two time or stage points** (a number with `min`, `minute(s)`,
         `h`, `hr(s)`, `hour(s)`, or with a space before `d`, `day(s)`, `week(s)`, or
         `stage <number>`, twice with no semicolon between): "not detected at 15 min but
         accumulated at 2 h", "absent at stage 12 but present at stage 13", "strongly
         reduced at 30 min and 1 h" (RT-R1-1, CSE-R1-C3, BIO-E746R1-2);
       - **a later presence after "but" or a semicolon**: `but` or `;` followed, before a
         full stop or semicolon, by `present`, `detected`, `detectable`, `appeared` or
         `accumulated` ("..., but was detected at 2 h", "; it was detected at 6 h");
       - **an early window or a later time**: `till`, `earlier than`, `(for / during /
         within / in) the first`, `early time point(s)`, `later`, `thereafter`,
         `subsequently`, `afterwards`, `onward(s)`, `only (after / at / from / by / when /
         once)`, `became (detectable / measurable / apparent / visible)`, `by <number>
         <time unit>` (the unit is required: "reduced by 90%" keeps its credit),
         `delay...`, `lag`, `before (the / one / two / three / four / five / six)`, `at
         <number> <time unit> ... but`, `first / earliest / early / initial` within 15
         characters before a time unit or `time point(s)`, and `<number> <time unit> only`;
       - **any statement of significance, a trend or variability except the adverb
         "significantly"**: the words `statistical`, `significant`, `significance`
         anywhere, `trend(s / ed)`, `tendenc...`, `(ns)`, `n.s`, `p-value(s)` or `P value`
         followed by `of`, `=`, `>`, `≥`, `above` or `greater than`, `within the (normal)
         range`, `within (the) range of`, `biological variation`, `variab...`,
         `inconsistent...`, and `in (only) one / two / some of (the) ... experiments /
         replicates / repeats` (RT-R1-2 (i), CSE-R1-C1, BIO-E746R1-1). "JA-Ile was
         significantly reduced in jar1 mutants (P < 0.01)." keeps its credit;
       - **a control genotype in a sentence without "than"**: `wild-type`, `wild type`,
         `Col`, `Col-0`, `WT`, `Ws`, `Ler`, `control(s)`, `parental`, `parent` anywhere
         in a sentence that does not contain "than" ("in jar1 mutants and control plants",
         "in the opr3 mutant and its wild type", "of wild-type and jar1 mutant plants");
         plus `any / all / each / either (of the) genotype(s) / lines`, `all (the)
         genotypes`, `the same was true`, and a bare `nor` (RT-R1-2 (ii), CSE-R1-C2).
         "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves." keeps its
         credit;
       - `nowhere near` anywhere (CSE-R1-C4), and a position rather than a change: `lower
         (rosette / cauline) leaf / leaves / part(s) / epidermis / stem(s) / side / surface
         / half` (RT-R1-4).

       The negated-verb forms CS-D added are therefore NOT immune to no-effect
       prose by construction (the CS-D brief said they were): they are guarded by these
       entries. A bare `than` is not a veto ("JA-Ile was 20-fold lower in jar1 mutants
       than in wild-type leaves." keeps its credit);
     - a protein or transcript measure (CS-D D2): `protein(s)`, `abundance`,
       `transcript(s)`, `mRNA`, `expression`, `expressed`, `activity`;
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
       `well known`, `it is (well) established that`, `is known to`, and since ORCH-746
       CS-E repair 2 background stated in a results sentence (RT-R1-6): `are / was /
       were known to`, `shown previously`, `well documented`, `well established`, `it is
       known` ("jar1 mutants are known to contain reduced JA-Ile levels.");
     - a procedure (`was/were grown, sown, harvested, sprayed, treated, extracted,
       analysed, quantified, measured, incubated, used, obtained, purchased, ...`,
       `we employed/used`, `plant material`, `growth conditions`) or a purpose clause
       (`to study`, `to test`, `as a new tool`);
     - another organism, or ANY gene symbol carrying another species' prefix, whether
       or not the sentence also names the target (`species._FOREIGN_GENE`, a documented
       ceiling: `Mp Pp Os Sl Zm Ta Hv Gm Nt Nb Bn Ca Vv Md St Le Cs Sb Br Mt Pt Lj Ph
       Ps Sm Gh Bd Sv Ptr Mtr` followed by "FAD5", " FAD5", "gh3a" / "fad5" with a
       digit, or "-aos1"; `Pa` is left out: "Part1", "Panel3" and "PaCO2" would read as
       genes);
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
     `unknown` without a failed lookup keeps primary-strength credit, an orchestrator
     interim position recorded by change set A pending the product owner (no
     DECISIONS.md entry exists). The conditions (rule 7, strict, repair 2):
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
  conversion, and a knockout result phrased outside the whitelist ("accumulated to only
  10% of wild-type levels"); since CS-D also a product that rises or a substrate that
  falls in the knockout, an effect on a participant named only inside the enzyme's own
  name, another gene's mutant, a protein or transcript measure, and "reduced but not
  abolished" (the clause split leaves "reduced" without its genotype).

Such a claim is a `species_ambiguous` (or in vitro) lead with a
`missing_species_evidence` frontier item. The organism list (`species._ORGANISM_WORDS`,
any abbreviated binomial such as "P. patens", and the gene-symbol prefixes; CS-D D4 added
Brachypodium, Camelina, Setaria, Lotus, Gossypium, Petunia, Catharanthus, Salvia, Mentha,
Artemisia, Panicum, Sorghum and "Glycine max", the bare word being the amino acid) is a
documented ceiling: an organism named only by a word missing from it is not seen, and
that miss costs an ADMISSION, not a lead.

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
