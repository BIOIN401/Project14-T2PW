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
