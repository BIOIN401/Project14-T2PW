# ORCH-746 change set E — repair round 2 (the LAST allowed round): close the open classes by SHAPE, subtract-only

Round-1 review of `01d436cf` (texts in `C:/t/bt/orch746/reviews/`: `cse_r1_code.txt`, `cse_r1_redteam.txt`, `cse_r1_bio.txt`):
code REQUEST_CHANGES (CSE-R1-C1/C2/C3 blocking), red team UNSOUND (RT-R1-1/RT-R1-2 blocking), bio UNSOUND (BIO-E746R1-1/-2
blocking). All three confirmed the boundary, the subset property (credit at `01d436cf` ⊆ `5eca087b` ⊆ `686ae134`, by
construction and on up to 70,000 fuzzed sentences), the G9 design, the suite (1005), the replay and the T-rows, and all three
say **fit for pilot once their measured subtract-only vetoes are in**. Every blocking finding is in the two classes round 0
ruled blocking, phrased in ordinary words round 1's lists did not name:
- a **negated delay / time course** credited as an absence ("not detected at 5 min after wounding but accumulated at later
  time points", "below the detection limit for the first hour", "became detectable only at 2 h", "absent at stage 12 but
  present at stage 13");
- a **no-difference result** credited as an absence or decrease: non-significance in other words ("did not reach statistical
  significance", "no statistically significant difference", "a trend towards lower", "(ns)", "with a P value of 0.12") and the
  result shared with a control ("in jar1 mutants and control plants", "in the opr3 mutant and its wild type", "of wild-type and
  jar1 mutant plants", "in all genotypes").

This is the third round of word lists on these classes. **This round closes them by SHAPE and takes the UNION of all three
reviewers' prototypes**, so an ordinary paraphrase falls inside a shape instead of beside a list. It is the last round: if its
review still blocks, ORCH-746 stops BLOCKED.

The three prototypes (read them; copy their entries verbatim where this brief says so):
- code `C:/t/bt/rev746e-r1-code/plugins/rev746e_r1_fix.py`;
- red team `C:/t/bt/rev746e-r1-redteam/plugin/rt746r1_veto.py` (groups A, B, C, and the `RT_CC` concessive group);
- bio `C:/t/bt/rev746e-r1-bio/plugin/bio746r1_veto.py` (groups A, B, E, C, D).

Same rules as before (`CS-E.md`, `COMMON_RULES.md`), worktree `C:/t/m745-b`, same branch, scratch `C:/t/bt/m746-e/` (fresh run
dirs). **Commits on top of `01d436cf`; the G9 base for this round is `01d436cf`.** Allowed files unchanged: `species.py`,
`tests/test_mining_species_cs_e.py` (append only), `species_policy.md`, `known_limitations.md`, `testing.md`. Subject style:
`experiment: ORCH-746 CS-E repair 2 <id> - <what>`.

## The binding property, unchanged

Every change only removes credit: entries added to `_NO_EFFECT` (a veto only, species.py:1094) or `_CITATION` (check it is
only a veto everywhere it is used), readings appended to `_READINGS`, a skip added in `_effect_under`, or narrowings of
`_MARKER_OF` / `_bound` / `_CONTAINER`. Credit at your tip must be a subset of credit at `01d436cf`. No recall gain of any kind.
Group the new `_NO_EFFECT` entries by shape with one comment per group naming the findings it closes.

## Required (the blocking classes)

1. **Time course** (RT-R1-1, CSE-R1-C3, BIO-E746R1-2). A sentence that describes an absence over time earns nothing. Add:
   - the SHAPE: **two time or stage expressions in one sentence** — bio group E (`<n> (min|minutes|h|hrs|hours)` ... `<n> <unit>`,
     no semicolon between), extended to days / weeks and to `stage <n>` (so "absent at stage 12 but present at stage 13" is caught);
   - the later-presence SHAPE (red team group B): `(but|;) ... (present|detected|detectable|appeared|accumulated)`;
   - the words of red team group B, bio group B and code item 3 (union): `till`, `earlier than`, `(for|during|within|in) the
     first`, `early time points?`, `later`, `thereafter`, `subsequently`, `afterwards`, `onwards?`, `only (after|at|from|by|when|once)`,
     `became (detectable|measurable|apparent|visible)`, `by <n> <time unit>` (KEEP the unit: "reduced by 90%" must keep credit),
     `delay\w*`, `lag`, `before (the|one|two|three|four|five|six)`, `at <n> <unit> ... but`;
   - bio group C (the early window): `(first|earliest|early|initial) ... (min|minutes|h|hours|time points)` within 15 characters;
   - "<n> <unit> only" ("reduced ... at 15 min only", bio's Y11).
   Rows (not TSS; each must fail at `01d436cf`): code DL1-DL4, DL7, DL8; red team hunt3 DL rows accepted at `01d436cf` (DL1-DL5,
   DL7-DL12); bio `runs/bat_tip` Z9-Z14 and `runs/b3_tip` Y9, Y11, Y12 (scripts under `C:/t/bt/rev746e-r1-bio/scripts/`).
   Controls that must keep credit: bio W3 "up to 2 h", W4 "throughout the time course (0.5 to 4 h ...)", W5 "At 1 h after
   wounding, JA-Ile was below the detection limit in gh3.10 jar1 knockout lines.", W8, W9; red team RC3 "No JA-Ile was detected
   in jar1 mutants 1 h after wounding."; T1.
2. **Non-significance** (RT-R1-2 (i), CSE-R1-C1, BIO-E746R1-1). Any statement about statistical significance or a trend, other
   than the plain adverb "significantly", earns nothing. Add: the adjectives `statistical` and `significant` and the noun
   `significance` anywhere in the sentence (this subsumes the code's `no statistically`, `significance` and the bio's negated
   `significant`); red team group A's `trend(s|ed)`, `tendenc`, `(ns)`, `n.s`, `p-value(s) of / = / > / ≥ / above / greater than`,
   `within the (normal) range`; bio group D's `biological variation`, `within (the) range of`. Also, for bio's "still passes"
   list: `variab\w*`, `inconsistent\w*`, `in (only) (one|two|some) of (the) ... (experiments|replicates|repeats)`.
   Rows: code NS1-NS3; red team hunt3 NS1, NS3-NS8, NS10, NS11, NS14; bio Z1-Z5, Z15, Y1, Y4, Y5. Controls that keep credit:
   "JA-Ile was significantly reduced in jar1 mutants after wounding.", "JA-Ile was reduced by 90% in jar1 mutants after
   wounding.", T1, A1-A10; red team RC4 "significantly reduced (P < 0.01)" — say whether it survives (it may not: a recall cost).
3. **A result shared with a control or with every genotype** (RT-R1-2 (ii), CSE-R1-C2, BIO-E746R1-1). The SHAPE (red team
   group C): **a wild-type or control genotype word (`wild-type`, `wild type`, `Col`, `Col-0`, `WT`, `Ws`, `Ler`, `control(s)`,
   `parental`, `parent`) in a sentence with no `than`** earns nothing; plus `any / all / each / either (of the) genotype(s) /
   lines`, `the same was true`, `nor`. This covers the code reviewer's word-order entries; confirm on its WT3, WT5, WT8, WT9.
   Rows: those four; red team hunt3 WT1-WT13 where accepted at `01d436cf`; bio's "all genotypes" row. Controls that keep
   credit: K2 "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves.", RC6 "90% lower ... than in Col-0", T1, and
   the canonical knockout sentences (no control word).

## Subtract-only closures of the round-1 non-blocking residuals (do each; DROP any that fails an existing test, and report it)

4. **Concessive** (RT-R1-3, CSE-R1-C5): sentence-initial `while` / `whilst`, `whilst`, `even if`, `notwithstanding`,
   `regardless of`, `albeit` (red team `RT_CC`, measured free).
5. **`nowhere near`** (CSE-R1-C4) anywhere in the sentence.
6. **Background** (RT-R1-6): `(are|is|was|were) known to`, `shown previously`, `previously (shown|reported|described)`,
   `well[- ]documented`, `well[- ]established`, `it is known` — in `_CITATION` if it is only a veto there, else `_NO_EFFECT`.
7. **A position, not a change** (RT-R1-4): `lower (rosette |cauline )?(leaf|leaves|part|parts|epidermis|stem|stems|side|surface|half)`.
8. **Regulators** (RT-R1-5, CSE-R1-C5): `_MARKER_OF` gap exclusions gain `suppress\w*|enhanc\w*|inducers?|co-?activators?|
   modulat\w*|mediators?|modifiers?`; a role noun right AFTER the gene (`activators?`, `suppressors?`, `enhancers?`,
   `regulators?`, `inducers?`, `modifiers?`, `targets?`) means the marker is not of the gene (`_CONTAINER`); in `_bound`'s
   before-branch, a gene followed by `suppressor` / `enhancer` / `modifier` before the marker ("jar1 suppressor mutant") does not
   bind. Rows: red team RG1, RG2, RG4, RG5, RG7; code RG1/RG2. Controls: SALK_034543, "T-DNA insertion lines of AOS", "Mutants
   deficient in GH3.10", "knockout of GH3.10", "gh3.10 jar1 knockout lines".
9. **The true subject after the effect word** (BIO-E746R1-4 (i), the implementer's suspects): in `_effect_under`, when a form
   has before-candidates AND a participant directly after it (an attributive "reduced X": at most two tokens, no comma), that
   participant joins the candidates, so a substrate-and-product pair makes the form ambiguous and it earns nothing. Rows: "In
   cyp94b3 mutants, JA-Ile accumulated after wounding at the expense of / alongside / plus reduced 12-OH-JA-Ile." (reversed
   claim), bio U1g "opr3 mutants accumulated OPDA alongside strongly reduced OPC-8:0 levels after wounding." (reversed). Measure
   its recall cost carefully on every canonical control above; DROP it if any existing test fails.

## Not in this round (documented residuals)

RT-D-R2-2 (the precursor of X, which is Y), CW1 (a comparison > 60 characters after "did not"), a comparison after a semicolon
inside a reference, an unlisted comparison word ("different", "below"), "accumulated only marginally / slightly less" (a
product-owner policy item), allele-numbered genotypes ("jar1-1") earning nothing (a pre-existing recall limit).

## Tests (append to `tests/test_mining_species_cs_e.py`)

- Regression rows `test_cs_e_r2_746_<item>_*`, each NOT `target_species_supported`, each failing at `01d436cf` (write them first
  and run them before changing `species.py`; leave out and list any that do not fail there). One row per required item also goes
  end to end through `ClaimStore` + `admission.judge`.
- Controls `test_new_acceptance_cs_e_r2_746_*` passing at both ends: every control named above, plus "No JA-Ile was detected in
  gh3.10 jar1 knockout lines.", "opr3 mutants did not accumulate OPC-8:0", "aos mutants accumulated no 12-OPDA".
- G9 manifest `C:/t/bt/m746-e/g9_manifest_cs_e_repair2.json`.
- Matrix: the 423 probes plus all three round-1 reviewers' batteries (code `C:/t/bt/rev746e-r1-code/scripts/`, red team
  `C:/t/bt/rev746e-r1-redteam/scripts/hunt3.py` and its recall controls, bio `C:/t/bt/rev746e-r1-bio/scripts/`), over `01d436cf`
  and your tip: 0 violations; list every row that LOSES credit and say which are correct biology (the recall cost).
- Full mining suite at your tip with `final_surviving_count`.

## Docs

- `species_policy.md` rule 5(b): every new veto written as the SHAPE it implements ("a sentence naming two time or stage
  points", "a later presence after but or a semicolon", "a control genotype in a sentence without 'than'", "any statement of
  significance or trend except the adverb 'significantly'"), the regulator rules, the attributive-participant rule.
- `known_limitations.md`: the closed classes move into what the gate refuses; for each shape state plainly what a sentence must
  avoid to slip past it (the remaining residual); the measured recall cost, including bio BIO-E746R1-3's undocumented costs (a
  participant inside a closed ", including X," is cut off; genotype-describing "with" prose such as "opr3 plants with a T-DNA
  insertion in OPR3"; the concessive veto refusing canonical correct prose such as "Although JA accumulated after wounding, JA-Ile
  was strongly reduced in jar1 mutants.", a cost the product owner should note); the round-1 doc asks (CSE-R1-C6 "eight";
  CSE-R1-C7 the item-3 comment and the `_effect` docstring "the first reading's reason"; RT-R1-3 the sentence-initial "While"
  line; RT-R1-7 the measured 29 / 21 (1 correct) / 19 / 2 (1 correct) figures at `01d436cf`); and BIO-E746R1-5: allele-numbered
  genotypes earn nothing, the main recall limit for a live pilot.
- `testing.md`: counts.

## Final report

As before, plus: per item 1-9 done or dropped; the G9 rows and their state at `01d436cf`; the matrix summary and the credit lost
(correct biology named); the suite count with `final_surviving_count`; the shapes you still suspect.
