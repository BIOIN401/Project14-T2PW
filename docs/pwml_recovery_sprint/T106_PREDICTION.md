# T-106 — prediction, written BEFORE the run

**Written 2026-08-23 by the Lead Orchestrator**, at integration tip `2053212`, before T-106 was
launched. Nothing here may be edited after the run starts. Its purpose is that T-106's outcome is
**classified deliberately**, not rationalised afterwards.

T-105's lesson is the reason this document exists: its priority-1 count came in at exactly the
predicted 7 **and the match was coincidence** — `succinyl-CoA`, `SREBF1/2`, `LIPA` and `LBR` all
vanished by draw variance and were replaced by `protoporphyrin IX`, `NADH`, `NAD+` and `holo-EntB`.
**Predict compositions and mechanisms, not counts.**

---

## 1. What changed since T-105, and the mechanism each change moves

| card | merge | mechanism it moves |
|---|---|---|
| **C-076** | `3b7a7b1` | acceptance **scorer**: a within-kind shared accession is identity, not conflict |
| **C-077** | `26fa809` | Stage-0 scope conflict now carries a **release classification** instead of `null` |
| **C-078** | `4797f58` | a name-keyed DB re-resolution may not **restore a refused identifier** |
| **C-080** | *pending review* | production **release gate** reads the same identity predicate as the scorer |

---

## 2. Predictions that are mechanically forced (high confidence)

### 2a. The six Stage-0 conflict legs look different — and this is C-077, not a regression

At T-105 all six `scope_conflict` rows carried `release_status: null`, `stage=stage1`, and folded to
`STATUS_INELIGIBLE` — *"nothing was attempted"*. At T-106 each will instead carry a real
`ReleaseStatus`: `status: diagnostic_only`, `pipeline_executed: true`,
`strict_gates_passed: false`, `strict_acceptance_eligible: false`, and reasons including
`stage0_scope_conflict_stopped_the_run_before_serialization`. The manifest row gains a
`requested_scope` key beside `observed_context`.

**Still expected: 6 × `scope_conflict`.** The guard is untouched and `eligibility_stage0_conflict_aborts`
is still `True`. **Do not read the new release block as a status change in the pipeline.** Do not read
`diagnostic_only` as "no defensible core" — F-107 records that PRODUCT_CONTRACT §4 has no accurate
state for this case, and two of these legs cleared the gold's own connected-core floor at T-104.

### 2b. Metal ions and formula-named compounds lose PathBank identity — this is F-110

C-078 makes a **pre-existing** bad refusal bite. The name gate cannot relate `ferric iron` to `Fe3+`
or `Zn2+` to `Zinc (II) ion`, returning `no_shared_meaningful_token`; before C-078 the pre-freeze
name-keyed re-resolution silently reversed those refusals.

Expect on affected legs: compound rows named as ions or formulae carrying
`db_status: identity_refused_review_required`, keeping their **extracted name**, **not dropped**.
`Zn2+` escaped C-078 on the committed corpus only because its DB match came back `ambiguous`; a
T-106 draw where it resolves uniquely will strip it.

**This is a depth/coverage loss, never a correctness loss.** Acceptance **priority 1 counts *false*
real identifiers**, so removing a correct one cannot move that numerator. `Fe3+` appears nowhere in
`t105_acceptance_report.json`. **Classify any such drop as F-110. Never quote it as an unexplained
coverage regression, and never as evidence against C-078.**

### 2c. Within-kind accession conflicts disappear from both the scorer and the gate

`EntB`/`holo-EntB` on `uniprot:P0ADI4` and `EntE`/`enterobactin synthase` on `uniprot:P10378` stop
being `accession_claimed_by_multiple_entities` in the acceptance report **and** stop failing
`no_real_id_or_name_conflict` on the production release path.

**Cross-kind must still fire.** `drugbank:DB00114` on `ALAS2` (protein) vs
`Pyridoxal 5'-phosphate` (compound) is the one true collision and was the single surviving
`id_check` failure across 56 committed legs. **If T-106 shows zero `id_check` failures anywhere
including a cross-kind pair, that is a blanket-disabling regression, not a success.**

---

## 3. The one outcome that must be watched, and how to classify it

**`PMC12452463` may reach `release_ready`, and that would be a product-contract violation — but not C-080's.**

Gold gives this paper `expected_export: "partial_only"`: the route is chemically broken with `EntA`
absent, and `EntA` is an explicit `placeholder_product` hallucination trap. At T-105 the leg's
**only** semantic failure was the within-kind rule that C-076/C-080 correctly remove. What still
holds it back is **C-072's anchor cap** (`requested_core_anchors_unmatched: DHB, EntA, Fur`) — and
that cap is **draw-sensitive** in a way the semantic cap was not.

If a T-106 draw happens to match `DHB`, `EntA` and `Fur`, the leg goes `release_ready` with a bare
`pathway.pwml`.

**How to classify it if it happens:**

* It is **legitimate under the 2026-08-23 identity ruling** — the entities really are the same
  protein, and flagging them was the rejected rule.
* It is a **`product_contract_violation` at the coverage/anchor layer**, i.e. F-094's *class*
  resurfacing where the anchor cap is the only remaining guard.
* **Do not charge it to C-076 or C-080.** The semantic cap was incidentally doing part of F-094's
  work; removing an invalid rule exposed that, it did not cause it.
* **Do not respond by restoring the within-kind rule.** Keeping a rule the product owner has ruled
  invalid so that it incidentally blocks a leg is precisely the inversion this sprint exists to
  correct.

A leg becomes `release_ready` *because of* C-080 only when **all five** hold: strict gates passed and
serializable; `semantic_failed_checks == ['no_real_id_or_name_conflict']` exactly, within-kind;
coverage `minimum_core_satisfied` **and `unmatched_terms` empty**; the request was stated and
coverage evaluated; and the largest connected core is ≥ 2 reactions.

---

## 4. Expected unchanged — these are NOT findings

* **2 × PMC12444477 TIMEOUT.** F-092 is open; its surviving defect 3 (the inner deadline path
  discards a computed `operation_timeout`) is carded for after T-106. Its defect 2 was **refuted** —
  `PRODUCT_CONTRACT.md:260` defines `budget_exhausted` in terms of the wall clock, so labelling a
  wall-clock kill that way is correct and F-092's remedy would itself violate the contract. Expect
  `budget_exhausted` on the outer path and an absent `termination_reason` on the inner one.
* **The negative control landing `review_required` rather than gold's "empty pathway plus a
  rejection reason".** Merge rule 7 forbids dropping the payload; `review_required` is the closest
  permitted outcome. Closing that gap is a product decision, not a card.
* **Two pre-existing test reds** (F-112) from committed `runs_verify/**` growth. T-106 committing its
  own run directory will break them a third time. Re-baseline after, do not treat as regression.

---

## 5. What would genuinely be new information

* A **cross-kind** accession conflict that fails to fire.
* A **false real identifier** that is not explained by an existing finding.
* A leg emitting a **bare `pathway.pwml`** whose gold `export_rationale` forbids strict export, on a
  path *other* than the anchor-cap gap named in §3.
* A Stage-0 conflict leg that **drops** its payload rather than carrying it as a classified record.
* An identifier that is **refused and then restored** anywhere downstream of the identity gate.
* Any leg where **`placeholder_identities_distinguished`** moves — 56/56 legs were identical across
  C-080's base and tip, so movement means something this wave did not measure.

---

## 6. What must not be done to the result

Per §13 of the session charter and D-063's precedent: **do not overwrite T-106 with a rerun under the
same identity**, do not relabel it PASS if a new product-contract violation appears, do not force
strict exports to improve the metric, and do not classify expected negative controls as pipeline
defects. If T-106 finds a new violation, preserve the result, register the finding, and continue
narrow evidence-backed work.

---

# AMENDMENT, written after C-080's review and before the run

Three corrections from the C-080 reviewer. **§3 below supersedes §3 above.**

## A. A claim of mine was overstated — corrected

`FINDINGS.md` F-108 and `prompts/C-080.md` §2d/§2e say *"the replay reproduces all three recorded
statuses exactly before the toggle, which is what makes the counterfactual trustworthy."*

That is **true of the three named legs and false corpus-wide**: 7 of 38 legs diverge. All seven
recorded `release_ready` and replay to `review_required`, because the current classifier applies
C-072's and C-074's caps that did not exist when those artifacts were written.

**The safety conclusion is unaffected, and the direction is why:** a leg already recorded
`release_ready` cannot flip *to* `release_ready`. The upper bound stands. But the trustworthiness
argument as I wrote it was broader than the evidence, and the committed probe exits **1** on exactly
these divergences — so a future reader hitting a non-zero exit is seeing this, not a failure.

## B. §3's risk was UNDERSTATED — this is the sharper and correct framing

I wrote that `PMC12452463` reaches `release_ready` if *"a T-106 draw happens to match `DHB`, `EntA`
and `Fur`."* That reads as bad luck. It is worse than that.

The production gate **deliberately does not ask the gold-only forbidden-identifier question** — its
own docstring says so, and C-080 did not change that. Meanwhile the gold set records:

* `EntA` — `kind: placeholder_product`, reason: *"HALLUCINATION TEST: this review's own four-step
  scheme SKIPS the EntA dehydrogenase step … Emitting an EntA reaction citing this paper is
  completing the canonical pathway from memory."*
* `DHB` — also `placeholder_product` (the paper contradicts itself between body and figure caption).
* `export_rationale`: *"whose route is chemically BROKEN: with EntA absent, nothing converts
  2,3-dihydro-2,3-dihydroxybenzoate onward."*

So the anchor cap stops firing **precisely when the pipeline hallucinates the `EntA` step** — the
exact fabrication the gold set was purpose-built to trap. **The hallucination would be the *cause*
of the release, and nothing on the production release path would object.**

**State it that way when classifying T-106. Not "the draw matched"; "a fabrication satisfied the
only remaining guard."** If it happens, that is a `product_contract_violation` of the F-094 class,
at the coverage/anchor layer — and it argues the class needs a *designed* guard rather than the
accidental one C-076/C-080 correctly removed.

Unchanged: do not charge it to C-076 or C-080, and do not answer it by restoring the within-kind
rule.

## C. The condition list is NINE, not five

A leg becomes `release_ready` *because of* C-080 only when **all** of these hold, derived from
`classify_release_status` at the merged tip:

1. `pipeline_executed`
2. `strict_gates_passed`
3. `serializable_without_invention`
4. a coverage verdict exists **and** `has_surviving_core`
5. `minimum_core_satisfied`
6. the semantic verdict becomes `passed` — `no_real_id_or_name_conflict` was the **only** failing
   gating check, **and** at the tip that check emits nothing, which additionally requires **no
   `placeholder_claims_real_identity` forgery on the leg**, because that finding lives in the *same*
   check and C-080 does not touch it
7. not (`verdict.declared and missing`) — **C-072**, `release_status.py:730`
8. not (`connected_core_reactions < 2` and not `single_reaction_scope_requested`) — **C-074 arm A**,
   `:784`, `MIN_CONNECTED_CORE_REACTIONS = 2`
9. not `verdict.declares_core_without_stating_a_pathway` — **C-074 arm B / F-100**, `:789` / `:813`

The two most easily dropped are **(9)** and the **no-forgery sub-condition inside (6)**. Both are
real guards; neither appeared in my §3.

**And there are three draw-sensitive guards, not two.** C-071's
`actor_named_in_its_own_cited_span` is also a gating production check, and it fired on
`PMC12096016/research` at T-105.

## D. Consequence for the cohort

`PMC12452463` and `PMC12096016` are in the affected-paper cohort (both modes) precisely so this is
observed on 10 legs before it can happen inside a 20-leg release candidate. **If `PMC12452463`
reaches `release_ready` in the cohort, T-106 does not start until that outcome is classified.**
