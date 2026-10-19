# The ten variants

Every variant is a complete drop-in replacement for
`src/t2pw/llm/prompts/pwml_system.txt`. None changes the JSON shape — see
SCHEMA_CHECK.md, all ten PASS.

**Baselines.** `v01` is the committed prompt (`HEAD`) and `v02` is the current working
tree. The two are the controls and are byte-verbatim; neither was improved.
**v03-v10 are each derived from `v02`**, not from `v01`, because v02's resolution rule is
what produced DNA-site entities at all (AUDIT.md §1.7) and no variant should give that up
to test something else. So *the control for v03-v10 is v02*, and v01 is the historical
reference that shows what the resolution rule bought.

**Isolation, verified.** `diff -u v02 vNN` gives exactly one hunk for v03-v08 and v10, all
insertions. v09 is the only variant that deletes baseline text (two hunks: the rule it
replaces, and two values in Example 2 — one idea, two places).

Line citations: `(C)` = committed, `(W)` = working tree, as in AUDIT.md. Variant line
numbers are that variant's own file.

---

## v01 — `v01_committed_baseline.txt` (CONTROL)

**Hypothesis:** none. This is the measured starting point.
**Change vs baseline:** none. `git show HEAD:src/t2pw/llm/prompts/pwml_system.txt`,
sha256 `1b3666a993f601f1…`, 34,733 bytes, 540 lines.
**Measured behaviour:** 12 interactions all at gene/operon level, 0 reactions; two
reactions invented downstream with every modifier `inferred`, one chemically empty
(araBAD operon → araBAD operon); vocabulary drift to `activates`/`represses`/`binds` and
the invented `promoter_of`.
**Metabolic papers:** this is the prompt the four metabolic papers' behaviour was tuned
against — all five worked examples are metabolic. Expect it to be the strongest or
near-strongest on metabolic recall and the weakest on regulation.
**Risk:** none; it is the reference.

## v02 — `v02_working_tree_resolution.txt` (CONTROL)

**Hypothesis:** none. The state the sprint is currently in.
**Change vs baseline:** v01 plus today's two uncommitted edits — the relationship
vocabulary paragraph at `(W):115-119` and the RESOLUTION block at `(W):210-225`.
sha256 `0a1adddfe72655cf…`, 37,044 bytes, 562 lines.
**Measured behaviour:** 28 interactions, 0 reactions; DNA-site entities appear correctly;
every row carries an evidence quote and nothing is invented. Defects: the same claim at
two or three resolutions, four alias rows, three rows on `ara regulatory region`, one
protein under two names.
**Metabolic papers:** the resolution rule is written in DNA-regulatory language
(operator, promoter, footprinted region, base-pair interval). On a purely metabolic paper
it should be close to inert — there is rarely a finer-than-stated grain for a metabolite
conversion — so v02 vs v01 on the four metabolic papers is mostly a check that the rule
does no harm.
**Risk:** none; it is the reference.

---

## v03 — `v03_resolution_precedence.txt`

**Name:** one claim, one resolution — finest wins.
**Hypothesis:** the duplicate-resolution rows are caused by an *omission*, not a
misunderstanding: v02 forbids going coarser than the text but never forbids *also*
restating the same claim coarser. Say "emit each claim once, at the finest resolution" and
the 28 rows collapse toward the ~12-15 that are distinct claims, with the finest survivor
kept rather than the coarsest.

**Changes:** inserts 17 lines into the RESOLUTION block, immediately after
`(W):225`. Before (end of the block, unchanged and still present):

> `This never licenses invention. Only a site, region, coordinate, conformer or occupancy`
> `the text names may appear, each with its verbatim evidence quote. (…) Finer than stated`
> `is a fabrication; coarser than stated is a loss. Record what is written.`

After, the block continues (v03:227-243):

> `ONE CLAIM, ONE RESOLUTION — finest wins. Having chosen the resolution above, emit`
> `each claim exactly ONCE, at the finest resolution the text states, and do not ALSO`
> `restate it coarser. Two rows restate one claim when the regulator, the sign and the`
> `consequence are the same and only the grain of the target differs (…)`
> `What is NOT a restatement, and where every row is kept: a binding event and its`
> `stated consequence ("[Protein] binds [Site]" AND "[Protein] blocks [Promoter]"),`
> `two different regulators on one target, one regulator on two different targets, and`
> `one regulator acting with opposite sign under two stated conditions.`
> `This rule only ever removes a row. It never adds one, never changes a sign, and`
> `never promotes a coarse row to a finer resolution the text does not state.`

**Leaves alone:** everything else, including v02's two-things rule at `(W):213-217` (the
binding *and* its consequence are explicitly carved out as not-a-restatement, so v03
cannot undo the thing v02 got right); the reaction rules; alias rows (v04's job); entity
naming (v05's job); the vocabulary (v09's job).

**Metabolic papers:** near-inert, which is the point. A metabolic paper states
`A → B` catalysed by an enzyme; there is no coarser restatement of a conversion to
suppress, and the rule only ever removes interaction-style duplicates. The one place it
could bite is a paper that states both "enzyme X acts in the pathway" and "enzyme X
converts A to B" — but the first is not an interaction row with the same
regulator/sign/consequence triple, so the test does not match it. Watch for under-reporting
of a regulated metabolic step (e.g. feedback inhibition stated both on the enzyme and on
the pathway).

**Risk if wrong:** the tie-break ("keep the finer row and drop the coarser") is the danger.
If the finer row is the one with the weaker evidence, or if downstream matching works
better on the coarse name (a gene is more likely to resolve to a UniProt/PathBank
identifier than "the A site"), v03 trades resolvable rows for unresolvable ones. AUDIT.md
D7 makes this concrete: site-level rows currently become orphan nodes in `draft_graph`, so
v03 may *increase* the orphan count while improving the payload. Judge v03 on the payload,
not on the graph.

---

## v04 — `v04_alias_rows_banned.txt`

**Name:** a naming fact is not an interaction.
**Hypothesis:** the four `SAME_AS` rows exist because nothing forbids them — the model is
being helpful about a vocabulary problem using the only row type available to it. An
explicit ban removes four of 28 rows with no other effect.

**Changes:** inserts 8 lines directly after the relationship paragraph at `(W):119`.
Before (`(W):115-119`, unchanged and still present):

> `- interactions[].relationship uses the same four words as its class: activation, inhibition,`
> `  binding, interaction. Do not write a verb form ("activates", "represses", "binds") and do`
> `  not invent a relationship ("promoter_of") (…)`

After, a new bullet follows (v04:120-127):

> `- A naming fact is not an interaction. A row whose entire content is that two strings`
> `  denote the same thing — an abbreviation and its expansion, a parenthetical gloss, a`
> `  line from the paper's abbreviations list ("CAP, catabolite gene-activator protein;`
> `  cAMP, cyclic AMP") — asserts nothing about biology and MUST NOT be emitted in`
> `  interactions[]. None of the four values above expresses identity, so do not invent one`
> `  ("SAME_AS", "same_as", "identical_to", "alias", "equals"), and do not borrow binding`
> `  or interaction to carry it instead: an identity row is not a weak interaction, it is`
> `  not an interaction at all. Omit it.`

**Leaves alone:** deliberately does **not** say how to choose between the two names — that
is v05, and bundling them would make the result unattributable. v04 therefore still allows
`CAP` and `catabolite gene-activator protein` to be registered as two entities; it only
stops the row that relates them. It also leaves `pwml_infer_system.txt:144-153` untouched,
so Stage-2 alias bridging via `SAME_AS` keeps working (AUDIT.md D4).

**Metabolic papers:** likely *more* useful than on a footprinting paper. Metabolic papers
are dense with abbreviations (`G3P`, `PLP`, `CoA`, `TAG`) and the alias-row failure mode
should be at least as common. `pwml_infer_system.txt:150-153` already notes that cofactor
charge-state names are canonicalised deterministically downstream, so Stage-1 alias rows
for them are pure noise.

**Risk if wrong:** if `SAME_AS` rows are actually load-bearing — i.e. if
`canonicalize_same_as_aliases` (`process_normalizer.py:2332`) is what currently rescues
mismatched participant names — then banning them at Stage 1 converts four harmless rows
into lost participants and dropped reactions (`filter_unresolvable_reactions`,
`pipeline.py:909-910`). This is the variant most likely to cause a quiet regression, and
the one to check against the per-paper reaction counts rather than the interaction counts.
Mitigation already in place: v04 bans them only at Stage 1, where they are out of contract
anyway; Stage 2 can still emit them with full knowledge of the merged entity list.

---

## v05 — `v05_one_canonical_name.txt`

**Name:** one canonical name per entity, chosen at extraction time.
**Hypothesis:** the duplicate-name defect is a missing *decision procedure*, not a missing
prohibition. Give the model an explicit ordered rule for picking one surface form and
applying it everywhere, and both the two-entity split and the alias rows lose their cause.

**Changes:** inserts 15 lines after the dedupe rule at `(W):75`. Before (`(W):75`,
unchanged and still present):

> `- Deduplicate identical strings within each entity list.`

After, a new bullet follows (v05:76-90):

> `- ONE CANONICAL NAME PER ENTITY. Two surface forms of one real-world thing are one`
> `  entity, not two. Before writing any row, choose exactly ONE string for that entity and`
> `  use that identical string everywhere it occurs: its entity registry row, reaction`
> `  inputs/outputs, modifiers[].entity, transports[].cargo, elements_with_states[].element,`
> `  interactions[].entity_1 and entity_2, and every *_locations row.`
> `  Choose it in this order, and never a form the text does not write:`
> `    1. the form used where the text introduces or defines the entity, including the`
> `       expansion supplied for an abbreviation (…)`
> `    2. failing that, the form the text uses most often when reporting its results;`
> `    3. failing that, the first form that appears in the text.`
> `  Do NOT additionally register the rejected form as its own entity row. This selects`
> `  among forms the text already contains, so it is not the normalization, expansion or`
> `  renaming that the verbatim rule above forbids — inventing a form, or preferring a`
> `  database name over the paper's, still is.`

**Leaves alone:** the verbatim rule at `(W):72-73` stays exactly as written, and the last
two lines above exist to resolve the tension rather than to weaken it: choosing between
two forms the paper uses is not the expansion `(W):73` forbids. Says nothing about alias
rows (v04's job) — though if the hypothesis holds, the alias rows should disappear anyway
because there is no second name left to declare. That overlap is the main thing to watch
when reading v04 and v05 side by side: **if both reduce the alias-row count, v05's result
is the more interesting one**, because it fixes the cause rather than the symptom.

**Metabolic papers:** should be the single most portable variant. One canonical name per
metabolite is exactly what `filter_unresolvable_reactions` (`pipeline.py:909-910`) and the
`process_normalizer` name index (`:626-633`) need, and identifier mapping is where the
four metabolic papers spend their value. Chlorophyll and plasmalogen papers routinely use
both a trivial name and an abbreviation for the same intermediate.

**Risk if wrong:** rule 1 picks the *definitional* form, which is often the long one
("catabolite gene-activator protein" over "CAP"). If downstream identifier lookup performs
better on the short form — plausible for gene symbols, where UniProt and PathBank index
the symbol — v05 improves internal consistency while *reducing* mapping hit rate. The
failure is silent: names stay self-consistent and simply resolve less often. Check
`missing_ids` in the QA report, not just the entity count. A second, smaller risk: a model
that reads "choose one string" as licence to normalise case or spelling, which `(W):72-73`
forbids and the inserted text re-forbids explicitly.

---

## v06 — `v06_transcription_as_reaction.txt`

**Name:** transcription and expression are reactions when the text states both ends.
**Hypothesis:** zero reactions on a transcription paper is a *representation* gap, not an
extraction failure — the model sees in-vitro transcription of araC but has nowhere to put
it, because the prompt defines reactions as "compound -> compound conversions"
(`(W):207`). Authorise it with both ends named and the paper yields its measured process.

**Changes:** inserts 18 lines after `(W):401`. Before (`(W):401`, unchanged and still
present):

> `- For reactions: only create if at least one input AND one output compound are explicitly mentioned.`

After, a new bullet follows (v06:402-419), in essence:

> `- TRANSCRIPTION AND EXPRESSION ARE REACTIONS when the text states both ends. (…) emit it`
> `  in reactions[]: inputs holds the template exactly as the text names it, outputs holds`
> `  the product exactly as the text names it, and class stays "biochemical_reaction". A`
> `  polymerase, ribosome or synthetase the text names for that step goes in modifiers[]`
> `  with role "catalyst" (…)`
> `  Hard limits, which override the bullet above:`
> `  * Both ends must be named in the text. "araC is transcribed", with no product named,`
> `    is not a reaction — omit it. Never coin a product name the text does not write, and`
> `    never put the same string on both sides; a row whose input equals its output is`
> `    chemically empty and must be omitted.`
> `  * Regulators of that step are not its reactants, products or modifiers. (…)`
> `  * This licenses a reaction only where the text states that synthesis. It is not licence`
> `    to add a transcription step for every gene the paper names.`

The "input equals output" clause is not decoration: it names the exact shape that shipped
under v01 (araBAD operon → araBAD operon), which survives `_clean_processes` because
dedup runs per side (AUDIT.md D9).

**Leaves alone:** the input-AND-output requirement, which it strengthens rather than
relaxes; the interaction rules, so the regulatory rows are unaffected; the cofactor and
modifier rules, which the polymerase clause defers to.

**Metabolic papers:** low impact and low risk by construction. The guard is "the text
states transcription/expression of a named gene AND names the product", which a
chlorophyll-biosynthesis paper rarely satisfies for a pathway step. Where a metabolic
paper *does* discuss enzyme induction, v06 could add a transcription reaction that is off
the metabolic pathway — but `scope_membership` exists for exactly that, and
`filter_out_of_scope_reactions` (`pipeline.py:288-368`) will drop it if the model labels it
`out_of_scope`. That makes v06 partly a test of whether the scope labelling holds up.

**Risk if wrong:** this is the variant most able to manufacture biology. The araC paper
says "araC mRNA synthesis" and "araC leader RNAs", so a product name does exist in the
text — but a model under pressure to produce a reaction may coin "araC mRNA" where the
paper only says "transcribed". If that happens the row looks perfect (it has a quote, it
has both ends) and is false. It is also the variant most likely to be judged a
`policy_disagreement` rather than a contract violation: whether a transcription step
belongs in `reactions[]` at all is a genuine product question, and the sprint's own rule
is that only a `product_contract_violation` justifies code. **Do not merge v06 on counts
alone; it needs a PO ruling on the representation question first.** Note also that
`pathway_curator_system.txt:118` forbids the curator from adding reactions, so if v06 is
right, Stage 1 is the only place this can ever be recovered.

---

## v07 — `v07_transcription_excluded.txt`

**Name:** transcription and expression are not reactions.
**Hypothesis:** the opposite answer to v06, and the null hypothesis for the whole
"0 reactions" finding. Zero reactions is the *correct* output for a footprinting paper;
the two invented reactions under v01 happened because the model inferred that a pathway
must contain reactions. Say so explicitly and the invention stops, with the regulatory
content carried entirely by `interactions[]`.

**Changes:** inserts 16 lines after the same anchor as v06, `(W):401` — so v06 and v07 are
exactly comparable, differing only in which answer they give. The inserted bullet
(v07:402-417), in essence:

> `- TRANSCRIPTION AND EXPRESSION ARE NOT REACTIONS. reactions[] holds stated chemical`
> `  conversions of named chemical species (…) Transcription, expression, translation,`
> `  induction, "synthesis of [a gene product]" and "levels of X rose" describe the`
> `  production of a gene product rather than a stated chemical conversion, and MUST NOT be`
> `  written as a reaction — not as gene-in / transcript-out, not with an operon or gene on`
> `  both sides, and not as an otherwise empty row carrying a polymerase as its catalyst.`
> `  Record that content where it is representable instead: (…) the regulatory claim (…) in`
> `  interactions[]; the DNA elements (…) are nucleic_acids entities; a polymerase (…)`
> `  reaches a row only through an interaction the text states.`
> `  A paper whose only measured process is transcription therefore yields zero reactions.`
> `  Zero is the correct answer there; a manufactured conversion is not.`

**Leaves alone:** everything else. The three bullets it adds all route content to places
the prompt already defines.

**Metabolic papers:** should be genuinely inert — it forbids a row type a metabolic paper
has no reason to emit. The one clause to watch is "levels of X rose", which on a metabolic
paper could be read as forbidding a legitimately measured conversion that the paper
reports as a concentration change. The surrounding sentence scopes it to gene products,
but this is the specific wording to check in the metabolic legs.

**Risk if wrong:** it ratifies a permanent recall ceiling. If a transcription step *should*
be a PWML reaction, v07 is the variant that guarantees the pipeline will never produce one,
and because it also tells the model that zero is correct, it removes the pressure that
might otherwise surface the question. Pairing it with v06 is what makes either result
interpretable; shipping v07 alone would look like success (no invented reactions) while
closing off the capability.

---

## v08 — `v08_regulatory_example.txt`

**Name:** a worked regulatory example beside the five metabolic ones.
**Hypothesis:** the prompt's examples are all metabolic — enzyme-as-catalyst, small-molecule
regulator, transporter, unknown transporter, complex-as-catalyst (`(W):433-551`) — so on a
regulatory paper the model has rules but no pattern to copy. Supplying one worked example,
in the same format and with the same completeness, does more than any amount of prose.

**Changes:** inserts 33 lines as `Example 5`, after Example 4's KEY POINTS at `(W):551`
and before `SUMMARY RULE` at `(W):553`. Full text is in the variant file at v08:553-585;
it is a complete, schema-valid two-interaction output for the pattern "`[Repressor]`
protects `[SiteName]`; occupancy of `[SiteName]` blocked transcription from
`[PromoterName]`", with:

* `nucleic_acids` rows for both DNA elements and a `proteins` row for the regulator;
* **two** interactions — the binding and its consequence — neither restated at gene level;
* `relationship` equal to `class` and drawn from the four allowed words
  (`binding`, `inhibition`);
* every field the schema lists, including `biological_state: ""`, `source_refs: []`,
  `confidence`, `provenance`;
* a KEY POINTS paragraph in the same voice as Examples 1-4, closing with "Applies to any
  regulator and any named element in any pathway."

**Leaves alone:** all five existing examples, byte for byte — including Example 2's
out-of-contract `"inhibits"`/`"activates"` values (AUDIT.md D3). That is deliberate: fixing
them is v09's single idea, and fixing them here too would make v08 and v09
indistinguishable. **Acknowledged confound:** because any schema-valid example must use
legal values, Example 5 necessarily demonstrates correct vocabulary. So v08 carries a
fraction of v09's treatment — for the regulatory pattern only, while Example 2 continues to
teach the wrong thing for the enzyme-regulator pattern. If v08 and v09 both improve
vocabulary, v09's result is the attributable one.

**Metabolic papers:** the variant with the clearest cost/benefit question. +2,205 bytes
(+6.3% over v01) buys nothing on a purely metabolic paper, where Examples 1-4 already cover
every pattern; the example is generic ("any regulator, any named element") but its subject
matter is not. Its defence is that six of the nine remaining pathways are signaling,
physiological or disease, so the example pays for itself on a majority of the corpus. The
risk on metabolic papers is dilution — one more example competing for attention with the
four that matter there — which is measurable as a drop in metabolic reaction recall.

**Risk if wrong:** a worked example is the strongest instrument in a prompt, so if the
example embeds a wrong choice, it propagates hard. The specific wrong choice it could
embed: that a regulatory paper should yield *two* interactions per regulatory event. On a
paper that states only the consequence and never the binding, a model copying the pattern
may invent the binding row to match the shape. The KEY POINTS text says the two rows exist
"because the text states two things", but that is prose guarding an example, which is the
weaker position.

---

## v09 — `v09_vocabulary_by_construction.txt`

**Name:** relationship by construction, not by prohibition.
**Hypothesis:** v02 fixes the vocabulary drift by naming the mistakes
(`"activates"`, `"represses"`, `"promoter_of"`). Prohibitions scale badly — they cover the
observed errors and not the next one — and v02's prohibition is contradicted by the
prompt's own Example 2 five hundred lines later. Replacing it with a construction rule
("decide `class`, then copy that word into `relationship`") plus repairing the example
should be strictly more robust.

**Changes:** the only variant that deletes baseline text. Two hunks, one idea.

Hunk 1 replaces `(W):115-119` entirely. Before:

> `- interactions[].relationship uses the same four words as its class: activation, inhibition,`
> `  binding, interaction. Do not write a verb form ("activates", "represses", "binds") and do`
> `  not invent a relationship ("promoter_of"): a structural fact such as "Pc is the promoter of`
> `  araC" is not a regulatory interaction. Sign matters and is a claim — use inhibition for`
> `  repression, and never activation for an event the text describes as blocking or competing.`

After (v09:115-131):

> `- interactions[].relationship is set BY CONSTRUCTION, never by paraphrase. Decide class`
> `  first — it is exactly one of activation, inhibition, binding, interaction — then copy`
> `  that same word, character for character, into relationship. The two fields always hold`
> `  the identical string, and there is no third vocabulary.`
> `  Choose the class in this order:`
> `    binding     — the text states physical association or occupancy and states no`
> `                  consequence in that same statement.`
> `    inhibition  — the text states the event reduces, blocks, represses, competes with or`
> `                  prevents the target's activity.`
> `    activation  — the text states the event raises, induces or is required for the`
> `                  target's activity.`
> `    interaction — the text relates the two but states neither direction nor sign.`
> `  The verbs the paper uses ("activates", "represses", "binds", "protects", "stimulates",`
> `  "blocks", "competes") are evidence: they belong in the evidence quote, and may appear in`
> `  name, never in relationship or class. Sign is a claim — when the text does not give you`
> `  one, "interaction" is the correct answer and not a guess. A structural fact such as "Pc`
> `  is the promoter of araC" is not a regulatory event and belongs in no interaction row.`

Both of v02's substantive guards are preserved: the structural-fact exclusion verbatim, and
the sign-is-a-claim rule strengthened into a positive default (`interaction` when the sign
is unreadable, rather than only "never activation for blocking").

Hunk 2 repairs Example 2 (`(W):475` and `(W):480`). Before → after:

> `"relationship": "inhibits"` → `"relationship": "inhibition"`
> `"relationship": "activates"` → `"relationship": "activation"`

Both rows already carried the correct `class` on the next line, so this makes the example
self-consistent and moves two values *into* the enumerated set (SCHEMA_CHECK.md §2).

**Leaves alone:** the enumeration at `(W):114` and every other value set; the `class`
values for reactions; `name`, which is free text and may keep the paper's verb.

**Metabolic papers:** the most portable of the eight after v05. Example 2 is a *metabolic*
example (small-molecule allosteric regulator on an enzyme), so the repair lands on the
pattern metabolic papers actually use — feedback inhibition and allosteric activation.
Expect v09 to matter as much on the four metabolic papers as on the regulatory ones, which
is unusual in this set. It is also the only variant that gets *shorter* in the part that
does the work (+847 bytes net, the smallest increase after v04).

**Risk if wrong:** the `binding`-vs-`inhibition` decision becomes mechanical, and the
mechanism may be wrong for footprinting. "Protein protects the site from digestion AND that
occupancy blocks the promoter" is two statements, which v09 would class as `binding` then
`inhibition` — correct. But "protein binds the operator and thereby represses" in a single
sentence hits the `binding` clause's "states no consequence in that same statement" test,
which excludes it, pushing it to `inhibition` and losing the binding row. That is the
inverse of v02's achievement, and the specific thing to check in v09's output: count the
`binding` rows, not just the legal-value rate. A second risk: making `interaction` the
explicit fallback may increase unsigned rows, which carry less information than a confident
(and possibly correct) sign.

---

## v10 — `v10_pre_output_self_check.txt`

**Name:** a ten-item pre-output self-check.
**Hypothesis:** most of the measured defects break rules the prompt already states — a
placeholder row, a missing quote, an out-of-vocabulary value, a participant absent from the
registry. The prompt is ~35 KB and the rules are scattered across twelve sections; a short
verification pass at the end, where the model is about to emit, catches violations that the
rules alone did not prevent. This tests *recall of the existing prompt*, not a new rule.

**Changes:** appends 28 lines at the end of the file, after the CROSS-CUTTING CONSTRAINTS
block (`(W):562`), as `PRE-OUTPUT SELF-CHECK` (v10:564-591). Ten checks: single JSON
object; verbatim quote per row; reaction input-and-output with no composites and no enzyme
in participants; actor type and role vocabulary; the four relationship values and the sign;
participants present in the registry; nothing invented (sites, coordinates, conformers,
occupancies, subunits, compartments); `scope_membership` and `species` present or
deliberately empty; no placeholder rows; visualization arrays empty. Framed with:

> `Every item below is a rule already stated above; this pass exists to catch the row that`
> `broke one. Fix or drop the offending row. Do NOT add a row to satisfy a check, and do not`
> `shorten or alter a quote to make a row pass.`

and closed with:

> `An incomplete payload of true rows is the correct result; a complete one containing a row`
> `you supplied is not.`

**Leaves alone:** every rule above it — by design the checklist introduces no rule, so
v10's effect is attributable to the verification pass rather than to new content.

**One disclosed exception.** Check 6 (every process participant appears in the entity
registry, spelled identically) is *implied* by the payload structure — section 1 of
PWML-CONSTRUCTIBLE OBJECTS defines the entity registries `(W):199-201` and locations link
entities to states `(W):203-205` — but it is nowhere stated as a requirement. It is the
one check that is not a pure restatement. It is included because it is the condition
`filter_unresolvable_reactions` silently enforces (`pipeline.py:909-910`) and the reason
`reconstruct_registry_shells` exists at the boundary
(`stage_one_boundary.py:19-25`). If v10 wins, this check should be separated out and
retested on its own before being credited.

**Metabolic papers:** should be the most portable variant of all, because nine of the ten
checks are about rules that are already pathway-agnostic, and several (cofactors out of
modifiers, no composite names, input-and-output) are *specifically* metabolic failure
modes. Checks 7 and 8 are the only ones with regulatory flavour.

**Risk if wrong:** two concrete ones. (i) Cost — +2,241 bytes, the largest of the eight,
and it is spent at the end of a 35 KB prompt where attention is cheapest; if it does
nothing it is pure token waste, which is the cleanest possible negative result. (ii) A
self-check can be *destructive*: a model that reads check 6 strictly may delete a correct
reaction whose participant it simply forgot to register, instead of registering it. The
closing instruction says to omit the row when it cannot be fixed without invention, which
is right for check 7 and wrong for check 6 — registering a participant is not invention.
Watch for reaction counts dropping relative to v02 with no corresponding entity increase.

---

## Comparison table

Design dimensions as columns. `—` = untouched; `+` = addressed; `(+)` = addressed only
incidentally, as a disclosed side effect.

| Variant | Base | Edit type | Δ bytes vs v01 | Dup resolution | Alias rows | Canonical name | Transcription | Reg. example | Vocabulary | Self-check | Deletes baseline text | Expected effect on 4 metabolic papers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v01 committed baseline | — | control | 0 | — | — | — | — | — | — | — | — | reference (examples all metabolic) |
| v02 working tree | v01 | control | +2,311 | partial | — | — | — | — | prohibition | — | no | ~inert |
| v03 resolution precedence | v02 | insert 17 | +3,710 | **+** | — | — | — | — | — | — | no | ~inert; watch regulated steps |
| v04 alias rows banned | v02 | insert 8 | +2,961 | — | **+** | (+) | — | — | — | — | no | **helps** (abbreviation-dense) |
| v05 one canonical name | v02 | insert 15 | +3,496 | — | (+) | **+** | — | — | — | — | no | **helps most** (id mapping) |
| v06 transcription as reaction | v02 | insert 18 | +3,672 | — | — | — | **+ (include)** | — | — | — | no | low impact; scope-labelled |
| v07 transcription excluded | v02 | insert 16 | +3,601 | — | — | — | **+ (exclude)** | — | — | — | no | ~inert |
| v08 regulatory example | v02 | insert 33 | +4,516 | (+) | — | — | — | **+** | (+) | — | no | no benefit; dilution risk |
| v09 vocabulary by construction | v02 | replace + 2 values | +3,158 | — | — | — | — | — | **+** | — | **yes** | **helps** (Example 2 is metabolic) |
| v10 pre-output self-check | v02 | append 28 | +4,552 | — | — | — | — | — | (+) | **+** | no | **helps**; mostly metabolic checks |

### Reading the table

* Only **v09** removes baseline text; everything else is additive, so a null result for
  v03-v08 and v10 cannot be caused by something having been taken away.
* **v06 and v07 share an anchor and are mutually exclusive answers** to one question. They
  must be read as a pair; neither is interpretable alone.
* **v04/v05** and **v08/v09** are the two pairs with a disclosed overlap. In each case the
  more fundamental variant (v05, v09) is the one whose result should be credited if both
  move the same metric.
* Four variants (v04, v05, v09, v10) are expected to help on metabolic papers; two (v03,
  v07) to be inert; one (v06) is low-impact but needs a product ruling; one (v08) pays only
  on the six non-metabolic pathways.

---

## One idea from the brief, deliberately left out

**"Separate what the paper measured from what the paper says in passing."** Left out, and I
think it is the wrong rule for this corpus.

On the araC paper it would work: the site-level rows come from the footprinting and
transcription results, while the gene-level restatements come from the introduction and
discussion, so a measured-vs-passing rule would suppress much of what D5 describes. But it
would do that by a proxy — section provenance — for what is really a resolution problem, and
v03 addresses the resolution problem directly.

On the four metabolic papers the same rule is actively harmful. A biosynthesis paper
routinely states the full pathway in its introduction and measures one or two steps; a rule
preferring measured content would suppress the correct, evidenced pathway steps that are
the whole point of extracting that paper. The pipeline also already has a calibrated
version of this idea in the right place: `_SECTION_RELEVANCE_MAP`
(`pipeline.py:3197-3215`) weights Results 0.9, Discussion 0.85, Introduction 0.4,
References 0.1, and `_narrow_to_high_signal_sections` (`:3562`) is rung 3 of the extraction
ladder. Section weighting is a retrieval decision that the code already makes with tunable
numbers; re-implementing it as a prose rule inside the prompt would duplicate it less
precisely and without the ability to tune it.
