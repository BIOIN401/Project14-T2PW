# The sixteen round-2 variants

Each file is a complete drop-in replacement for `src/t2pw/llm/prompts/pwml_system.txt`.
None changes the JSON shape — `SCHEMA_CHECK_R2.md`, all 16 PASS.

**Base and control.** `r2_01_control_v03.txt` is a byte copy of
`docs/evaluation_v2/prompt_variants/v03_resolution_precedence.txt` (38,443 bytes, sha256
`bc716123774c3d8b…`). All fifteen others are derived from it, so **the control for every
variant is `r2_01`**, and round 1's committed baseline is not re-run: `AUDIT_R2.md` § 2 is
why re-measuring it would not inform a choice.

**Isolation, verified.** `diff -u r2_01 r2_NN` gives exactly **one hunk, insertions only**
for thirteen variants; `r2_11` is three one-line replacements (one idea, three places; one `diff -u`
hunk at default context) and is the only variant that deletes baseline text. `r2_16` holds
five inserted blocks landing in three contiguous regions, plus `r2_11`'s three replacements,
and is the only deliberately multi-idea file.

Line citations: `ctl:<n>` = the control, `r2_NN:<n>` = that variant's own file.

**Every per-variant table below is required reading before any run.** A variant whose
expected effect on one paper type is "helps" and on another is "unknown" has not been
designed, it has been guessed.

---

## r2_01 — `r2_01_control_v03.txt` (CONTROL)

**Hypothesis:** none. It is round 1's best-supported variant, unchanged.
**Change:** none.

| paper | expected effect |
|---|---|
| PMC319880 regulatory | the measured round-1 behaviour: 11 interactions, 0 reactions, 0 bad vocabulary, 0 rows without evidence (`ORCH-747-RESULT.md:41`) |
| PMC12096016 metabolic | unmeasured. v03's resolution rules are written in regulatory language and should be close to inert here; the lumped-summary and off-pathway leaks of `AUDIT_R2.md` § 5 are unaddressed |
| PMC12421875 metabolic | unmeasured, same reasoning, plus no rule reaches the five steps stated only through letters and rate laws |

**Risk:** none; it is the reference. Its *value* is that it measures, for the first time,
what round 1's winner does on a paper with certified gold.

---

## r2_02 — `r2_02_classify_then_branch.txt`  (ROUTER A: explicit classification)

**Hypothesis:** the model already reads what kind of source it has; what it lacks is
permission to act on it. Naming the five kinds, and stating which bucket carries the result
and what the correct *empty* answer is under each, converts an implicit judgement into an
explicit one and should raise recall on metabolic papers while keeping the regulatory
paper's reactions at the number the text supports.

**Before** (`ctl:1`, unchanged and still present, now followed by the new block):

> `You are a PWML builder extraction engine.`

**After** (`r2_02:2-46`), in full at the head of the file:

> `SOURCE KIND, DECIDED FIRST (internal: it has no field and must not appear in the output)`
> `Before extracting anything, read enough of the text to decide which ONE of these the source`
> `is. Decide it from what the paper MEASURES AND REPORTS, not from its title, its journal, or`
> `its introduction's survey of other people's work.`
> …five kinds: `METABOLIC`, `REGULATORY`, `MIXED`, `REVIEW`, `METHOD`, each defined by what is
> reported rather than by subject matter…
> `  REGULATORY  — interactions[] carries the result (…) reactions[] holds only a conversion the`
> `                text states with two named, chemically different sides. If it states none,`
> `                reactions[] is empty and that is the correct answer, not a gap to be filled.`
> `  MIXED       — do not pick a branch for the paper. Route each claim by its own shape (…) A`
> `                biosynthesis paper's inhibition statement is still an interaction, and a`
> `                regulation paper's stated conversion is still a reaction.`
> `This decision changes WHICH rules bind, never what the evidence must be. (…) And the decision`
> `never outranks a sentence: a kind you chose is not a reason to write a row the text does not`
> `state, nor to drop one it does. If the text keeps contradicting your choice, you chose wrong;`
> `re-decide and extract again.`

The classification is explicitly internal: there is no field for it and the block says so,
which is what keeps the shape frozen.

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral to mildly positive.** Routed REGULATORY: interactions keep carrying the result, and "reactions[] is empty and that is the correct answer" removes the pressure that produced v01's two invented reactions. It does **not** forbid the transcription reaction the drafted gold books, because the two-named-sides clause admits it |
| PMC12096016 metabolic | **positive on recall, neutral on precision.** Routed METABOLIC or MIXED; the branch says every stated conversion is a row, and the MIXED clause explicitly preserves the paper's inhibition claim as an interaction. No new precision rule, so the 4 off-pathway and 2 summary leaks remain |
| PMC12421875 metabolic | **positive on recall.** Same as above; the METHOD clause should push the *L. plantarum* biosensor and ferrozine chemistry out of scope, which is 2 of its 3 off-pathway rows |

**On a paper it classifies wrongly.** The dangerous misread is METABOLIC → REGULATORY on
paper 2 or 3: the branch then tells the model that an empty `reactions[]` is correct, and 5
or 9 true positives vanish in one stroke. The inverse misread (REGULATORY → METABOLIC on
paper 1) is milder: the branch invites reaction rows, and the model has only gene-level
material to build them from, which is how `araBAD operon → araBAD operon` got shipped.
**How a reader can tell:** a near-empty `reactions[]` on a paper whose `entities.compounds`
list is long and whose proteins are enzymes is a mis-route, not a finding; so is a
`reactions[]` whose rows all have a gene, operon, promoter or transcript on both sides. Both
are one grep over the payload and neither needs the gold.

**Risk if the hypothesis is wrong:** this is the costliest variant to get wrong, because a
single wrong decision at the top of a 41 KB prompt conditions everything after it, and the
failure is silent and plausible. It is also the largest (+3,225 bytes) and spends its tokens
where attention is highest, which is the strongest version of the router idea and therefore
the right one to test — but it must never be shipped on a win on one paper.

---

## r2_03 — `r2_03_conditional_routing.txt`  (ROUTER B: no label, conditional rules)

**Hypothesis:** the benefit of routing is the per-claim discrimination, not the label. If
the rules are written as conditions on the claim ("where the paper's measured result is
…"), the model gets the same differentiation with no classification step to get wrong, and
no single decision that can take the whole extraction down one branch.

**Before** (`ctl:207-208`, unchanged and still present):

> `   - reactions (compound -> compound conversions), transports (movement between locations),`
> `     reaction_coupled_transports (reaction + movement), interactions (regulation/binding/etc.)`

**After** (`r2_03:209-235`), inserted between the bucket list and the RESOLUTION rule:

> `   ROUTING WITHOUT A LABEL. Do not classify the paper and do not decide what kind of paper`
> `   it is. Route each claim by what the text says about that claim. Several of these`
> `   conditions apply to most papers, and that is normal rather than a conflict.`
> `   * Where the paper's measured result is a chemical conversion (…) that claim belongs in`
> `     reactions[] (…) This holds even when the paper's headline result is regulatory.`
> `   * Where the paper's measured result is occupancy, association, protection, competition,`
> `     repression, induction or any other control (…) belongs in interactions[] and never in`
> `     reactions[]: control states no product. This holds even when the paper's headline`
> `     result is a biosynthetic route.`
> `   * Where the paper's measured result is a number about a route — a rate, a flux, a`
> `     concentration, a model fitted to one — the extractable claim is the conversion the`
> `     number is about (…) The number itself is not a process.`
> `   * Where the paper's measured result is a property of an assay, a strain, a device or a`
> `     model (…) its reporter chemistry is out_of_scope, not core.`
> `   * Where a claim's shape fits no bucket (…) omit it.`
> `   A paper is whatever mixture of these its own sentences make it. (…) neither list is ever`
> `   padded to make the output look complete.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral to positive.** The control clause keeps occupancy in `interactions[]`; the "never in reactions[]" half is the explicit form of what v03 achieved implicitly |
| PMC12096016 metabolic | **positive, modestly.** The "headline result is regulatory" and "headline result is a biosynthetic route" clauses are written precisely for this paper: an inhibition study of a biosynthetic enzyme. Both its halves have work to do here |
| PMC12421875 metabolic | **positive.** The "number about a route" clause is aimed exactly at this paper, whose five hardest steps exist only inside rate laws; the assay clause covers 2 of 3 off-pathway rows |

**On a paper it classifies wrongly.** It cannot: there is no classification. The analogous
failure is a claim mis-routed, which is local — one row in the wrong bucket, not a whole
extraction. **How a reader can tell:** a row whose two sides are not different species in
`reactions[]`, or an `interactions[]` row whose endpoints are a substrate and its product.

**Risk if the hypothesis is wrong:** conditional prose may be too weak an instrument —
round 1's one decisive result came from a prohibition, not from a condition. If `r2_03`
comes out indistinguishable from `r2_01` while `r2_02` moves, the label was doing the work
and the cheap router loses. A second risk: "the number itself is not a process" could be
read as licence to ignore a model-stated conversion rather than to extract it, which is the
opposite of the intent; `r2_13`'s worked example is the belt to that braces.

---

## r2_04 — `r2_04_kind_recheck.txt`  (ROUTER C: self-assessment after the fact)

**Hypothesis:** routing is most reliable when it is applied as a *check*, after the output
exists, because by then the model has read the whole paper and the evidence of what kind it
is sits in its own payload. A pre-commitment can be wrong about the paper; a post-check can
be compared against it.

**Before** (`ctl:580`, the last line of CROSS-CUTTING CONSTRAINTS, unchanged):

> `- Export layer: PathWhiz/PWML export may wrap a protein catalyst in a generated`
> `  single-protein protein_complex (…)`

**After** (`r2_04:581-610`), appended at the end of the file:

> `OUTPUT REVIEW AGAINST THE KIND OF SOURCE THIS TURNED OUT TO BE`
> `(…) It may CORRECT or REMOVE a row, and it may add only a row the text states that you`
> `missed; a shortfall it finds is never fixed by invention.`
> `1. It reports conversions and reactions[] is empty. Re-read for conversions stated in the`
> `   ways that are easiest to miss: through the enzyme's own name or activity, through letters`
> `   or symbols assigned to intermediates and paired later in a model, equation or rate law,`
> `   or only in a figure or table legend. (…)`
> `2. It reports conversions and every reaction you wrote has a gene, an operon, a promoter, a`
> `   transcript or a protein on both sides. That is regulation written as chemistry (…)`
> `3. It reports control and interactions[] is empty while reactions[] holds rows whose two`
> `   sides are not chemically different species. That is the inverse error: move them back.`
> `4. It reports both. Then both lists are non-empty, and no single claim appears in both.`
> `5. (…) survey (…) keep only the pathway and organism selected under the scoping rule above.`
> `6. (…) assay, a model or a device (…) a reporter, dye, chelator, mediator or biosensor`
> `   reaction is labelled core. Label it out_of_scope or omit it.`
> `7. Under every kind, unchanged: a verbatim quote on every row; every relationship one of the`
> `   four allowed words; no site, coordinate, participant, product, intermediate or conformer`
> `   that the text does not name.`
> `An empty bucket the text justifies is a correct answer. A bucket filled so the output looks`
> `complete is not.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral.** Checks 3 and 7 are the ones that fire; both are already satisfied under `r2_01`, so a null result here is the expected and informative one |
| PMC12096016 metabolic | **positive on recall.** Check 1 names the "through the enzyme's own name" case, which is exactly R3's substrate and R4's product |
| PMC12421875 metabolic | **positive on recall.** Check 1 names the letters-and-rate-law case, which is 5 of its 9 steps, and check 6 covers the biosensor rows |

**On a paper it assesses wrongly.** A wrong post-assessment is self-limiting: the checks
only compare the output against the text and each one says what it may do. The real risk is
check 1 being read as a quota — "it reports conversions and `reactions[]` is empty, therefore
produce some" — which is why the clause "a shortfall it finds is never fixed by invention"
sits in the header rather than at the end.
**How a reader can tell:** a row added by a check-1 sweep carries an equation, a legend or
an enzyme-name sentence as its evidence; a fabricated one carries a quote that does not
contain both sides. One verbatim check against the paper separates them.

**Risk if the hypothesis is wrong:** it is spent at the very end of a 40 KB prompt, where
attention is cheapest (round 1 made the same observation about v10), so a null result may
mean the position rather than the idea. Cost if inert: +2,156 bytes of pure waste, which is
the cleanest negative result available.

---

## r2_05 — `r2_05_mixed_case_procedure.txt`  (ROUTER D: the mixed case as a procedure)

**Hypothesis:** the mixed paper is the case a naive router breaks on, and all three scored
papers are mixed (both metabolic papers state regulation; the regulatory paper states one
conversion). So the right routing instrument is not a branch at all but an ordered
per-claim procedure, with claim-splitting as its first step.

**Before:** the same anchor as `r2_03` (`ctl:207-208`).

**After** (`r2_05:209-235`):

> `   THE MIXED CASE, AS A PROCEDURE. Most papers are mixed: a biosynthesis paper states the`
> `   inhibition of one of its own enzymes, and a regulation paper states a conversion in`
> `   passing. Do not decide what kind of paper this is and then extract only that kind of`
> `   content. Decide per claim (…)`
> `   Step 1 — is this one claim or several? "X binds the site and thereby blocks Y" is two`
> `     claims. "A is converted to B by E" is one. Split first; route afterwards.`
> `   Step 2 — does the claim name a species on the left that BECOMES a different named species`
> `     on the right? Then it is a conversion: reactions[] (…) The enzyme goes in modifiers[].`
> `   Step 3 — if not, does it state that one named entity acts on, occupies, associates with,`
> `     raises or lowers another? Then it is an interaction (…)`
> `   Step 4 — if neither, it is not a process. (…) omitted, however central to the paper.`
> `   Step 5 — (…) ask whether it belongs to the requested pathway. Chemistry the paper states`
> `     for another route, for an assay, or in another organism is stated but not requested (…)`
> `   Two consequences, because the procedure is misapplied in both directions. A paper whose`
> `   main result is regulatory still yields every conversion it states. A paper whose main`
> `   result is a biosynthetic route still yields its stated inhibition and induction claims.`
> `   Neither list suppresses the other (…)`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral to positive.** Step 1 is v03's binding-and-consequence carve-out restated as a procedure; step 4 should suppress the site-inventory and negative-result rows that have no bucket |
| PMC12096016 metabolic | **positive on both sides.** Step 2 keeps the five steps, step 3 keeps the inhibition claim, step 5 should move the MenD branch, the LDH assay, secretion and chelation out of scope — 4 of the paper's 4 off-pathway statements |
| PMC12421875 metabolic | **positive on both sides,** same mechanism: 9 steps kept, the MenD inhibition kept as an interaction, 3 off-pathway statements scoped out |

**On a paper it classifies wrongly.** Again there is no paper-level classification; the
exposure is a mis-split at step 1. Splitting too eagerly inflates row counts with halves of
one claim; splitting too little loses the binding row that v03 exists to preserve.
**How a reader can tell:** two rows whose evidence quote is the *same sentence* and whose
endpoints overlap is an over-split; a single row whose quote contains both a binding verb
and a consequence verb is an under-split.

**Risk if the hypothesis is wrong:** a five-step procedure is heavier than the problem on a
paper that is not mixed, and procedures invite mechanical application — step 2's "becomes a
different named species" test is a good gate but a poor reader of an indirectly stated
conversion, so `r2_05` alone could still miss PMC12421875's rate-law steps. It is the
variant most worth pairing with `r2_07`.

---

## r2_06 — `r2_06_bucket_two_sided_gate.txt`

**Hypothesis:** the brief's "make the two-sided requirement explicit per bucket" is the
cheapest kind-neutral precision rule available, because it discriminates by *grammar* rather
than by subject matter. A binding can then never land in `reactions[]` on any paper, and a
conversion can never be demoted to an interaction on any paper, with no judgement about what
kind of paper it is.

**Before:** the same anchor as `r2_03`/`r2_05` (`ctl:207-208`).

**After** (`r2_06:209-227`), in full:

> `   BUCKET ADMISSION, BY SHAPE (apply to every process row before writing it).`
> `   * reactions[] requires TWO CHEMICALLY DIFFERENT SIDES: at least one named species`
> `     consumed, and at least one DIFFERENT named species produced. The same string, or the`
> `     same entity under two spellings, on both sides is chemically empty — omit the row. A`
> `     gene, operon, promoter, site, transcript or protein named as such is not the product of`
> `     a conversion unless the text states the conversion that produces it.`
> `   * A claim that names an actor and a target but no product — binding, occupancy,`
> `     protection, competition, repression, induction, "is required for", "is controlled by" —`
> `     can NEVER be a reaction, whatever bucket would be convenient. It is an interaction, or`
> `     it is nothing.`
> `   * interactions[] requires two named entities and a stated relation between them. A`
> `     conversion must not be written as an interaction either: where the text states that A`
> `     becomes B, that is a reaction, and an interaction relating A to B discards the chemistry.`
> `   * transports[] requires a cargo that is the SAME entity on both sides plus a stated change`
> `     of place. A cargo that changes identity is a reaction, not a transport.`
> `   This gate only ever removes a row; it never licenses one. A claim that passes no gate is`
> `   omitted, and an empty bucket is the correct result for a text that states nothing of that`
> `   shape.`

**The one clause that carries the generalization.** "A gene, operon, promoter, site,
transcript or protein named as such is not the product of a conversion **unless the text
states the conversion that produces it**." The exception is deliberate: it fails
`araBAD operon → araBAD operon` (round 1's measured invention, which survives
`_clean_processes` because dedup runs per side) while admitting the drafted gold's
`araC gene → araC transcript`. Without the exception this variant would be v07 in disguise
and would decide D-open-2 by the back door. See `AUDIT_R2.md` D-R2-2.

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **positive on precision, no recall cost.** Kills the chemically empty row type; leaves occupancy in `interactions[]`; does not forbid the one transcription reaction the paper states with two named sides |
| PMC12096016 metabolic | **positive.** The paper's central claim is competitive inhibition; the second bullet stops it becoming a reaction, and the third stops a true conversion being demoted to an interaction. Carrier-loading and apo→holo statements, which the gold excludes as protein-bound states, have no product and are gated out |
| PMC12421875 metabolic | **positive.** DHNA efflux is gated to `transports[]` (same cargo, change of place) rather than becoming a reaction; the MenD inhibition stays an interaction |

**Risk if the hypothesis is wrong:** the gate is a removal rule, so its failure mode is
lost recall. Two specific exposures. (i) A conversion whose product is stated with the same
*name* in a different state — a phosphorylated or carrier-bound form written with the parent
name — reads as "the same entity on both sides" and is dropped; the certified golds exclude
those statements anyway, so this should be invisible here but may bite on another corpus.
(ii) "It is an interaction, or it is nothing" may be read as licence to drop a claim the
model cannot classify, where the right answer is `relationship: interaction` with no sign.

---

## r2_07 — `r2_07_conversion_sweep.txt`

**Hypothesis:** the recall losses on the certified papers are not judgement failures, they
are *reading* failures: the conversion is stated in a form that does not look like "A is
converted to B". Enumerating those forms, as a sweep run before finishing, should recover
steps no routing rule can reach.

**Before** (`ctl:419`, unchanged and still present):

> `- For reactions: only create if at least one input AND one output compound are explicitly mentioned.`

**After** (`r2_07:420-441`), in full:

> `- CONVERSION SWEEP (recall). Before you finish, sweep the whole text once for stated`
> `  conversions, because the ones most often missed are not written as "A is converted to B".`
> `  A conversion is stated, and belongs in reactions[], when the text names BOTH of its sides`
> `  in any of these forms:`
> `  * a verb of change between two named species — converted, isomerized, hydrolyzed, reduced,`
> `    oxidized, adenylated, methylated, demethylated, prenylated, condensed, cyclized, cleaved,`
> `    joins, produces;`
> `  * the product named only through the enzyme's own name or stated activity ("2,3-DHB is`
> `    adenylated by the AMP ligase", "the dehydrogenase acting on [Substrate]"), where the text`
> `    names the substrate and the enzyme's name states what becomes of it;`
> `  * intermediates to which the text has assigned letters, numbers or symbols, with the`
> `    pairing given later in a model, a rate law, an equation or a table — the letters stand`
> `    for the names the text assigned them, and the pair they are written in is the conversion;`
> `  * a conversion stated only in a figure or table legend;`
> `  * a step in a sentence that lists several enzymes in order, where the text also names the`
> `    species each one consumes or produces.`
> `  Every swept row still carries its own verbatim evidence quote, which may be that sentence,`
> `  that legend or that equation. Do NOT coin an intermediate the text never names in order to`
> `  complete a chain, and do not promote a multi-step summary to a step. Where a named enzyme's`
> `  substrate or product is simply not stated anywhere, that enzyme gets no reaction row: an`
> `  unwritten step is a loss, an invented one is a fabrication, and only the second cannot be`
> `  repaired downstream.`

**Disclosed contamination.** The second bullet's illustration uses `2,3-DHB` and "AMP
ligase", which are PMC12096016's own terms. That is a deliberate, declared leak: the
abstract form ("the product named only through the enzyme's own name") is the rule, and a
named illustration is what makes a prompt rule land. It means **`r2_07`'s result on
PMC12096016 is not blind** and must be read with that stated; its results on PMC12421875 and
PMC319880 are blind. If this matters to the ruling, the two words can be replaced with
`[SpeciesB]` and `[Enzyme]-AMP ligase` for a clean re-run, at a likely cost in effect size.

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral, with a tail risk.** The paper states one transcription step and no metabolite chemistry, so the sweep should find nothing new. The risk is that "produces" and the enzyme-name clause invite a row from "araC gene … produces sufficient araC protein", which the drafted gold books as a *summary*, not a reaction. Watch for exactly that row |
| PMC12096016 metabolic | **strongly positive on recall** (non-blind, see above): R3's substrate and R4's product are each reachable only through the enzyme-name clause, and R1–R5 are listed in one enzyme-sequence sentence covered by the last clause |
| PMC12421875 metabolic | **strongly positive on recall** (blind): 5 of 9 steps exist only as letters paired in equations (5)–(11), which the third clause names directly; MenA and MenG come from the verb clause |

**Risk if the hypothesis is wrong:** it is the variant most able to manufacture chemistry,
because it tells the model where to look for conversions that are not obvious. The closing
clause ("an unwritten step is a loss, an invented one is a fabrication") is the only thing
standing between a sweep and a completed chain, and it is prose guarding a list, which is
the weaker position. Scoring must report `unsupported` rows and the off-pathway/summary
split alongside recall, or a precision loss will be booked as a recall win.

---

## r2_08 — `r2_08_summary_not_a_step.txt`

**Hypothesis:** v03's "one claim, one resolution" idea is right and under-generalized. The
metabolic form of the same defect is the lumped route sentence, which has two real sides and
a true verbatim quote and is never a true positive. Stating that a summary is not a step
should be the cheapest precision gain on both certified papers (`AUDIT_R2.md` D-R2-1).

**Before:** the same anchor as `r2_07` (`ctl:419`).

**After** (`r2_08:420-432`), in full:

> `- A LUMPED SUMMARY IS NOT A STEP. Text often states a whole route in one sentence ("X is`
> `  produced from Y by a seven-enzyme pathway", "three enzymes convert A to B"). That sentence`
> `  names two real sides, so it passes every other test in this prompt, and it is still not a`
> `  reaction: nothing in it is a single conversion.`
> `  * Where the text ALSO states the individual steps — anywhere: body, legend, model,`
> `    equations — write the steps and do NOT additionally write the lumped sentence as a row.`
> `    Writing both ships the same chemistry twice, once correctly and once at a grain that`
> `    hides every intermediate.`
> `  * Where the text states ONLY the lumped sentence and names no intermediate, do not invent`
> `    the intermediates. The lumped pair may be written as a single reaction row, with that`
> `    sentence as its evidence, and nothing finer may be manufactured around it.`
> `  * A pathway named as a unit with no participants ("the menaquinone pathway",`
> `    "beta-oxidation") is a name, not a process. Omit it.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **mildly positive, or neutral.** Its one summary (araC gene → araC protein, transcription and translation lumped) is the kind of row this suppresses — which is also the direction D-open-2 may overturn. Report it, do not score it |
| PMC12096016 metabolic | **positive on precision.** Two summary statements (chorismate → 2,3-DHB; chorismate → 2,3-diDHB) are live precision leaks and both individual-step sets are stated |
| PMC12421875 metabolic | **positive on precision.** One summary (chorismate → DHNA, seven enzymes), with all seven steps stated |

**Risk if the hypothesis is wrong:** the second bullet is the exposure. On a paper that
states only the lumped conversion, the rule permits one row — correctly — but a model that
reads the first bullet harder than the second will drop it, trading a `paper_stated_summary`
row for nothing. That is still an improvement for precision and a loss for coverage, so the
variant must be read on both numbers. It is also the only variant that could *reduce* row
counts on all three papers, which will look like a regression to anyone reading counts.

---

## r2_09 — `r2_09_requested_pathway_boundary.txt`

**Hypothesis:** the largest precision leak on the certified papers is not invention, it is
*in-scope-ness*: 4 and 3 statements respectively are real chemistry the paper states and the
requested pathway does not contain. `scope_membership` is the one field that survives
cleaning and has an enforcement path (`pipeline.py:2083`, `:288-368`), so sharpening the
boundary is the only precision rule in the set with a mechanism behind it.

**Before** (`ctl:16`, one long line, unchanged and still present):

> `Reactions whose scope_membership is "out_of_scope" are removed from the payload before downstream stages run — omitting a clearly out-of-scope reaction entirely is equally correct.`

**After** (`r2_09:17-40`):

> `WHAT COUNTS AS THE REQUESTED PATHWAY`
> `A paper states more chemistry than the pathway asked for, and the difference is usually`
> `explicit in the text rather than a judgement call. Label a reaction out_of_scope, or omit it,`
> `when any of the following is true of it — even when the paper measures it carefully:`
> `  * the text itself names it as a different route ("the menaquinone branch", "a competing`
> `    sink", an enzyme of another pathway named for comparison);`
> `  * it is the chemistry of an ASSAY rather than of the pathway: a coupled reporter reaction,`
> `    a colorimetric or fluorescent readout, a dye, a chelator, a redox mediator, a quenching`
> `    step, or any reaction run only to make the real one measurable;`
> `  * it occurs in a DIFFERENT ORGANISM from the one selected under the scoping rule above (…)`
> `  * it is movement rather than conversion (secretion, efflux, import, uptake) (…)`
> `  * it is association rather than conversion (chelation, complexation, a product bound by a`
> `    carrier) (…)`
> `Conversely, do NOT push a step out of scope because its participants are unfamiliar, because`
> `it sits at the start or the end of the route, or because the paper spends little space on it.`
> `A defining step the text states is core even when it is stated once, in a legend, or through`
> `an equation. A wrong out_of_scope label silently deletes a step: the label is a positive`
> `claim that the paper's own framing puts that reaction elsewhere, and nothing weaker.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral.** It has one off-pathway statement (araBAD expression) and no assay chemistry in reaction shape; its footprinting reagents never had a product side. A null result here is expected |
| PMC12096016 metabolic | **positive on precision.** Its four off-pathway statements are one per bullet: MenD competing branch, LDH coupled assay, TolC secretion, ferric chelation |
| PMC12421875 metabolic | **positive on precision.** Biosensor reduction in *L. plantarum* (different organism), ferrozine colorimetry (assay), DHNA efflux (movement) — again one per bullet |

**Risk if the hypothesis is wrong:** an `out_of_scope` label is a *deletion* once the filter
runs, and deletion of a true step is the one failure this pipeline cannot detect downstream
(`filter_out_of_scope_reactions`'s own docstring says as much). The closing paragraph exists
to make the label a positive claim rather than a doubt, but a model that over-applies the
"assay" bullet on a paper whose pathway is studied *by* an assay — which is most in-vitro
biochemistry, including PMC12096016 — could scope out real steps. **Watch recall, not just
precision, on this variant**; it is the one whose failure is a silent recall loss.

---

## r2_10 — `r2_10_legend_vs_body.txt`

**Hypothesis:** the brief's "what to do when the figure legend contradicts the body" is a
real and recurring case, and the right rule is to keep the stated participants and claim
nothing about the disagreement. PMC12421875 states it in its sharpest form: MenG's verb is
wrong ("demethylates") while the pair it states, DMK → MK, is right, and the gold scores the
pair.

**Before:** the same anchor as `r2_07`/`r2_08` (`ctl:419`).

**After** (`r2_10:420-436`), in full:

> `- WHEN THE TEXT CONTRADICTS ITSELF, KEEP THE PARTICIPANTS AND CLAIM NOTHING ELSE.`
> `  A figure legend, a model equation and a sentence of the body sometimes disagree. Resolve it`
> `  this way, and never by deciding which one is biologically right:`
> `  * Participants and sides: where two statements name DIFFERENT species for what is clearly`
> `    the same step, prefer the statement that names both sides explicitly and quote that one.`
> `    Write two rows only where the text treats them as two steps. Never merge two statements'`
> `    participants into a row that neither of them states.`
> `  * A loose or wrong VERB does not invalidate the pair. Where the text names a substrate and`
> `    a product and describes the change with a verb that does not fit it, the stated pair is`
> `    what you record: the verb belongs in the evidence quote and may appear in name. Do not`
> `    reverse the sides to make the verb fit, and do not drop the row because the verb is wrong.`
> `  * A legend that states a step the body omits is the text stating that step. A legend that`
> `    only expands an abbreviation states nothing.`
> `  * Where the disagreement is about SIGN (…) do not pick. Record the two only if the text`
> `    states them under two different conditions, each with its own quote; otherwise use the`
> `    relationship interaction, which claims no sign.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **mildly positive.** Its Discussion argues against another group's model, and its own negative results are stated as denials; the sign clause should keep the model from recording the rejected model's edges, which is the defect the curated PathBank record already has (`ORCH-747-RESULT.md:125-131`) |
| PMC12096016 metabolic | **neutral to mildly positive.** Its Figure 1 legend states which reactions each enzyme catalyses, and the legend clause makes that legend admissible evidence |
| PMC12421875 metabolic | **positive, specifically on R9.** Without the verb clause a careful model either drops MenG or reverses DMK → MK to fit "demethylates"; with it, the scored pair survives |

**Risk if the hypothesis is wrong:** the first bullet's tie-break ("prefer the statement that
names both sides explicitly") is a heuristic, and where the explicit statement is the *wrong*
one, the rule prefers the wrong one confidently. It also adds no recall anywhere, so if the
contradiction cases are rarer than they look, it is 1,439 bytes of inert text. The last clause
pushes unsigned relationships toward `interaction`, which the PWML exporter cannot map
(`ir.py:29` holds only Activation and Inhibition), so an interaction gain here may be invisible
in exported PWML.

---

## r2_11 — `r2_11_example2_repair.txt`  (the cheapest candidate, 241 bytes)

**Hypothesis:** round 1 identified the prompt's own worked Example 2 as emitting the two
values the enumeration forbids (`ORCH-747-RESULT.md:185-188`) and never tested the repair on
its own, because v09 bundled it with a rule rewrite (`AUDIT_R2.md` D-R2-3). A concrete
filled-in example outweighs prose for most models, so repairing it should be strictly more
robust than the prohibition alone — and because Example 2 is a *metabolic* example, the
repair lands on the pattern both certified papers state.

**Before** (`ctl:493`, `ctl:498`, `ctl:503`):

> `        "entity_1": "[SmallMolecule]", "relationship": "inhibits", "entity_2": "[EnzymeY]",`
> `        "entity_1": "[OtherMolecule]", "relationship": "activates", "entity_2": "[EnzymeY]",`
> `  KEY POINTS: the protein enzyme ([EnzymeY]) is the catalyst modifier. Small-molecule regulators (compounds) go in interactions[], not modifiers[]. This applies regardless of which pathway or molecules are involved.`

**After** (`r2_11:493`, `:498`, `:503`):

> `        "entity_1": "[SmallMolecule]", "relationship": "inhibition", "entity_2": "[EnzymeY]",`
> `        "entity_1": "[OtherMolecule]", "relationship": "activation", "entity_2": "[EnzymeY]",`
> `  KEY POINTS: the protein enzyme ([EnzymeY]) is the catalyst modifier. Small-molecule regulators (compounds) go in interactions[], not modifiers[]. This applies regardless of which pathway or molecules are involved. Note the two interaction rows: "relationship" holds the SAME word as "class" — inhibition, activation — never the paper's verb. The verb ("allosterically inhibits", "activates") belongs in the evidence quote and may appear in "name".`

Both rows already carried the correct `class` on the following line, so the repair moves two
values *into* the enumerated set and removes none (`SCHEMA_CHECK_R2.md` § 2).

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral to positive.** Round 1 already measured zero vocabulary violations under the prohibition alone, so the ceiling here is a null result — which is the point: it tests whether the prohibition is doing the work or the example was never read |
| PMC12096016 metabolic | **positive where it matters most.** The paper's headline claim is a small molecule competitively inhibiting an enzyme: Example 2's exact pattern, and the only place in the prompt that shows how to write it |
| PMC12421875 metabolic | **positive, same pattern** (DHNA allosterically inhibits MenD) |

**Risk if the hypothesis is wrong:** none worth the name. It cannot lose rows, cannot change
scope and cannot suppress a bucket; the worst case is +241 bytes and no measurable change.
That asymmetry is why it should be in the first batch regardless of what else runs.

---

## r2_12 — `r2_12_regulatory_example.txt`

**Hypothesis:** all five worked examples are metabolic, so on a regulatory paper the model
has rules and no pattern to copy. One worked regulatory example, in the same format and with
the same completeness, is a stronger instrument than any amount of prose — and the brief's
open question is whether an all-metabolic example set hurts regulatory papers.

**Before** (`ctl:571`, unchanged and still present, now preceded by the new example):

> `SUMMARY RULE: every named enzyme/catalyst in the text MUST appear in modifiers[] on its`
> `reaction with role "catalyst". (…)`

**After** (`r2_12:571-604`) — a complete, schema-valid `Example 5` for the pattern
"`[Repressor]` protected `[SiteName]` against digestion; occupancy of `[SiteName]` blocked
transcription from `[PromoterName]`", with `nucleic_acids` rows for both DNA elements, a
`proteins` row for the regulator, **two** interactions (the binding and its consequence,
neither restated at gene level), `relationship` equal to `class` and drawn from the four
allowed words, every field the schema lists, and a KEY POINTS paragraph closing:

> `  KEY POINTS: there are TWO rows because the text states two things — the binding and its`
> `  consequence — and each carries its own quote. Where the text states only one of them, write`
> `  only that one: the pattern is not a shape to be completed. (…) reactions[] gains nothing`
> `  from this text: regulation has no product side. Applies to any regulator and any named`
> `  element in any pathway.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **positive, the only variant aimed squarely at it.** It supplies the missing pattern for exactly this paper's content: a protein protecting a site, and that occupancy blocking a promoter |
| PMC12096016 metabolic | **neutral at best, dilution risk.** +2,108 bytes buys nothing on a conversion-heavy paper; one more example competes for attention with the four that matter here. Measurable as a drop in reaction recall against `r2_01` |
| PMC12421875 metabolic | **neutral at best, same dilution risk** |

**Disclosed confound** (inherited from round 1's v08): any schema-valid example necessarily
demonstrates correct vocabulary, so `r2_12` carries a fraction of `r2_11`'s treatment for the
regulatory pattern. If both move vocabulary, `r2_11`'s result is the attributable one.

**Risk if the hypothesis is wrong:** a worked example is the strongest instrument in a
prompt, so a wrong choice inside it propagates hard. The specific exposure is the two-row
shape: on a paper that states only a consequence and never the binding, a model copying the
pattern may invent the binding row. The KEY POINTS sentence guards exactly that, and it is
prose guarding an example — the weaker position.

---

## r2_13 — `r2_13_model_stated_example.txt`

**Hypothesis:** the mirror of `r2_12`, and the one the certified papers need. The examples
are all metabolic but they are all *plainly stated* metabolic — "EnzymeX catalyzes the
conversion of A to B". Neither of the two hardest real recall cases looks like that. One
worked example showing a conversion read off assigned letters and a rate law, and a second
read off the enzyme's own stated activity, should do what `r2_07`'s prose tries to do,
with a stronger instrument.

**Before:** the same anchor as `r2_12` (`ctl:571`).

**After** (`r2_13:571-615`) — a complete, schema-valid `Example 5` with **two** reaction rows
from one text sample: the first taking its pair from letters the text assigned and the rate
law that pairs them, the second taking its product from the enzyme's stated activity, with
ATP as an input because the text calls the reaction ATP-dependent and **no** co-product,
because the text does not name one. KEY POINTS closes:

> `  KEY POINTS: both rows are stated by the text, neither in the plainest form. The first reads`
> `  the letters the text itself assigned and the rate law that pairs them — the row uses the`
> `  NAMES, not the letters. (…) Nothing here licenses coining an intermediate to complete a`
> `  chain: if neither the body, nor a legend, nor an equation names a step's product, that step`
> `  gets no row. Applies to any pathway stated through a model, a legend or an enzyme's name.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral, dilution risk.** It states no model-based chemistry; the added example is +2,519 bytes of irrelevance. Watch for a reaction row appearing where `r2_01` had none |
| PMC12096016 metabolic | **positive on recall.** The second row of the example is R4's exact shape (adenylation, product named through the ligase's name, ATP-dependent) |
| PMC12421875 metabolic | **strongly positive on recall.** The first row of the example is the shape of 5 of its 9 steps |

**Disclosed contamination.** The example is written in placeholders (`[SpeciesA]`,
`[EnzymeP]`) rather than in either paper's terms, which keeps it blind — but the *choice* of
pattern comes from reading the golds, so the variant tests "does a worked example beat prose
for this pattern", not "would anyone have thought of this pattern".

**Risk if the hypothesis is wrong:** it is the most invention-capable file in the set. An
example that shows a product being read off an enzyme's name teaches a procedure that, applied
loosely, generates products from enzyme names wherever one appears. The guard is the explicit
"no co-product because the text does not name one" and the closing clause. If reaction counts
rise on all three papers, including the regulatory one, that is this variant failing, not
succeeding.

---

## r2_14 — `r2_14_pre_output_self_check.txt`

**Hypothesis:** most measured defects break a rule the prompt already states, across a 38 KB
file whose rules sit in twelve sections. A short verification pass at the point of emission
should catch them. This re-runs round 1's v10 idea with its self-identified defect repaired
(`AUDIT_R2.md` D-R2-4) and with every item written two-directionally.

**Before:** the same anchor as `r2_04` (`ctl:580`).

**After** (`r2_14:581-611`) — ten checks: one JSON object; verbatim quote per row; two
different species per reaction with no composites and no enzyme among participants; actor
type and role vocabulary; the four relationship words with `interaction` as the unsigned
answer; **participants present in the registry, repaired rather than deleted**:

> `6. Every participant named in a process row also exists in the entity registry, spelled`
> `   identically. Where one is missing, ADD THE ENTITY ROW — the participant is already stated`
> `   by the row you are checking, so registering it invents nothing. Drop the process row only`
> `   if the participant is not in the text at all.`

then: nothing invented; one claim per resolution **and** no bucket holding another bucket's
claim; `scope_membership` present with no assay or other-organism row labelled core; no
placeholder rows and empty visualization arrays. Closing: *An incomplete payload of true rows
is the correct result. A complete one containing a row you supplied is not.*

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral.** Checks 2, 5, 7 and 10 are already satisfied under `r2_01`; a null result is expected and is the cleanest possible reading of "the rules were already being followed" |
| PMC12096016 metabolic | **positive, modestly, on both sides.** Checks 3, 4, 8 and 9 are metabolic failure modes (cofactors in modifiers, composite names, summary rows, assay rows labelled core) |
| PMC12421875 metabolic | **positive, modestly,** same mechanism |

**Risk if the hypothesis is wrong:** a self-check is destructive by default — its natural
verb is "drop" — and check 6's repair is the only place this file says "add". If reaction
counts fall on the metabolic papers with no matching rise in entity counts, the check is
eating true rows and the variant is a reject regardless of what it fixes elsewhere. Cost if
inert: +2,252 bytes at the position where attention is cheapest.

---

## r2_15 — `r2_15_core_plus_typed_overlay.txt`  (the lazy router)

**Hypothesis:** if routing helps, most of the benefit may come from one line per kind, and
the rest of a router is cost. This is the minimum viable version of the brief's "shared core
with a small typed overlay": no classification step, no branch, no procedure — four
conditional priors plus an explicit statement that a sentence outranks the prior.

**Before:** the same anchor as `r2_03`/`r2_05`/`r2_06` (`ctl:207-208`).

**After** (`r2_15:209-227`), in full:

> `   ONE CORE, A SMALL TYPED OVERLAY. Every rule in this prompt applies to every source. The`
> `   overlay below changes nothing about evidence or schema; it says only which bucket is`
> `   expected to carry the result, and what the correct EMPTY answer looks like, so that an`
> `   empty bucket is not read as a gap to fill.`
> `   * Where the result is chemical conversion: reactions[] is the main bucket; interactions[]`
> `     holds only the regulation the text also states. An empty interactions[] is correct.`
> `   * Where the result is control of one entity by another: interactions[] is the main bucket.`
> `     An empty reactions[] is correct, and is the right answer when the text states no`
> `     conversion with two named chemical sides.`
> `   * Where the text reports both, both buckets are expected to be non-empty, and no claim`
> `     appears in both.`
> `   * Where the text's subject is an assay, a model, a device or another organism's strain, its`
> `     own chemistry is out_of_scope rather than core, however carefully it is measured.`
> `   The overlay is a prior, not a permission and not a prohibition. A sentence always`
> `   outranks it: one stated conversion fills reactions[] on a regulation paper, and one stated`
> `   inhibition fills interactions[] on a biosynthesis paper.`

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **neutral to mildly positive.** "An empty `reactions[]` is correct" removes the completion pressure; the final sentence keeps the one stated conversion admissible |
| PMC12096016 metabolic | **mildly positive.** Prior 1 and prior 4 apply; no new recall mechanism, so the gain is bounded |
| PMC12421875 metabolic | **mildly positive,** same, with prior 4 covering the biosensor organism |

**On a paper it classifies wrongly.** The priors are conditions, not a decision, so a wrong
reading is per-claim rather than global — but "an empty `reactions[]` is correct" is a
dangerous sentence on a metabolic paper read as regulatory, which is why the closing
counterweight is explicit.
**How a reader can tell:** the same test as `r2_02` — an empty `reactions[]` beside a long
compound list.

**Risk if the hypothesis is wrong:** it may be too weak to move anything, in which case the
comparison `r2_02` vs `r2_15` is the real result: how much router do you have to buy. If
`r2_15` matches `r2_02` at a third of the bytes, ship `r2_15`.

---

## r2_16 — `r2_16_integrated.txt`  (the candidate single prompt)

**Hypothesis:** the shape gate, the recall sweep, the summary rule and the scope boundary are
independent and kind-neutral, and the label-free routing is the cheapest routing that cannot
mis-route globally. Combined on top of v03 and with Example 2 repaired, they should be the
best single prompt across all three paper types.

**Changes:** five blocks, each verbatim from the single-idea variants above — `r2_09`'s scope
boundary (`r2_16:17-40`), `r2_03`'s conditional routing (`:253-278`), `r2_06`'s bucket gate
(`:234-252`), `r2_08`'s summary rule (`:490-502`), `r2_07`'s conversion sweep (`:503-524`),
plus `r2_11`'s Example 2 repair (`:598`, `:603`, `:608`). +8,172 bytes over the control
(+21%), the largest file in the set.

| paper | expected effect |
|---|---|
| PMC319880 regulatory | **positive on precision, neutral on recall.** The gate kills chemically empty rows, routing keeps occupancy in `interactions[]`, and nothing here forbids the transcription reaction |
| PMC12096016 metabolic | **positive on precision and recall.** The sweep should reach R3 and R4; the summary rule removes 2 leaks; the boundary removes 4 |
| PMC12421875 metabolic | **positive on precision and recall.** The sweep should reach 5 steps; 1 summary and 3 off-pathway statements are scoped out |

**Risk if the hypothesis is wrong:** attribution. Five treatments in one file means a null or
negative result names no cause, and a *win* cannot be decomposed either — so `r2_16` is only
interpretable beside its five components. It also stacks one recall-raising rule (`r2_07`) on
top of three removal rules, and if the removal rules bite the swept rows, the result is a
wash that looks like "nothing works". **It must not be merged on its own result; the
single-idea runs are what license it.**

---

## Comparison table

`—` untouched · `+` addressed · `(+)` addressed only incidentally, as a disclosed side
effect. Expected direction per paper: `+` better, `0` neutral, `-` risk of worse, `?` unknown.

| variant | edit | Δ bytes | routing style | bucket gate | recall rule | precision rule | example | self-check | deletes text | PMC319880 | PMC12096016 | PMC12421875 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2_01 control v03 | — | 0 | — | — | — | — | — | — | no | measured | ? | ? |
| r2_02 classify then branch | insert 45 | +3,225 | explicit label, branched | (+) | (+) | (+) | — | — | no | 0/+ | + | + |
| r2_03 conditional routing | insert 27 | +1,981 | conditions, no label | (+) | (+) | (+) | — | — | no | 0/+ | + | + |
| r2_04 kind recheck | append 30 | +2,156 | post-hoc self-assessment | (+) | (+) | (+) | — | **+** | no | 0 | + | + |
| r2_05 mixed case procedure | insert 27 | +1,991 | per-claim procedure | (+) | — | (+) | — | — | no | 0/+ | + | + |
| r2_06 bucket two-sided gate | insert 19 | +1,462 | — | **+** | — | (+) | — | — | no | + | + | + |
| r2_07 conversion sweep | insert 22 | +1,735 | — | — | **+** | — | — | — | no | 0/- | + (non-blind) | + |
| r2_08 summary not a step | insert 13 | +1,034 | — | — | — | **+** | — | — | no | 0 | + | + |
| r2_09 requested pathway boundary | insert 24 | +1,719 | — | — | — | **+** (enforced) | — | — | no | 0 | + | + |
| r2_10 legend vs body | insert 17 | +1,439 | — | — | (+) | (+) | — | — | no | 0/+ | 0/+ | + |
| r2_11 example 2 repair | replace 3 lines | +241 | — | — | — | — | **+ (repair)** | — | **yes** | 0 | + | + |
| r2_12 regulatory example | insert 34 | +2,108 | — | — | — | — | **+ (add)** | — | no | + | 0/- | 0/- |
| r2_13 model-stated example | insert 45 | +2,519 | — | — | **+ (by example)** | — | **+ (add)** | — | no | 0/- | + | + |
| r2_14 pre-output self-check | append 31 | +2,252 | — | (+) | (+) | (+) | — | **+** | no | 0 | + | + |
| r2_15 core + typed overlay | insert 19 | +1,279 | four priors, no label | — | (+) | (+) | — | — | no | 0/+ | 0/+ | 0/+ |
| r2_16 integrated | 5 blocks / 3 regions + 3 replaces | +8,172 | conditions, no label | **+** | **+** | **+** | **+ (repair)** | — | **yes** | + | + | + |

### Reading the table

* **Four pairs are designed to be read against each other.** `r2_02` vs `r2_15` is how much
  router you must buy. `r2_03` vs `r2_05` is conditions vs procedure at the same anchor and
  within 10 bytes of the same size. `r2_07` vs `r2_13` is the same recall idea as prose and as
  a worked example. `r2_12` vs `r2_13` is which kind of example the example set is missing.
* **Only `r2_11` and `r2_16` delete baseline text**, so a null result anywhere else cannot be
  caused by something having been removed.
* **Three variants are expected to be neutral on the regulatory paper and positive on both
  metabolic ones** (`r2_08`, `r2_09`, `r2_14`) and one is positive on all three (`r2_06`).
  Those four are the kind-neutral core of the set. Everything with a `-` in any column is a
  variant whose generalization is in question by construction, and that is marked rather than
  argued away.
* **No variant legislates the transcription question** in either direction (`AUDIT_R2.md`
  D-R2-2). Any variant that did would be unscoreable until D-open-2 is decided.

---

## Ideas from the brief deliberately left out

**A "measured versus mentioned in passing" rule.** Round 1 left it out and I agree, for the
reason it gave and for a stronger one: `_SECTION_RELEVANCE_MAP` already weights sections with
tunable numbers (`pipeline.py:3197-3215`), and on PMC12421875 the thing that must be read is a
*Methods* model section, which any introduction-penalising proxy would also down-weight.

**A rule routing alias rows to `synonyms`.** Forbidden by the frozen-shape constraint, and
round 1 proved the defect is post-Stage-1 anyway.

**Any variant adding `scope_membership` to interactions**, which would be the natural fix for
"an off-pathway interaction has no downstream filter" (round 1's D8). It is a schema change,
which is a reject rather than a thing to justify.

**A variant restoring `class`, `provenance` or `compartment_canonical` on process rows.**
Those keys are deleted by the cleaner (`AUDIT_R2.md` § 4); instructing the model harder about
them measures nothing.
