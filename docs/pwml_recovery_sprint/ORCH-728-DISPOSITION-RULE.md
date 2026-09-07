# ORCH-728 — the serialization floor. Refuse / `review_required` / `release_ready`, derived from evidence.

**Read-only, 2026-09-07.** No production edit, no leg re-run, no LLM draw, no cache write.
Branch `sprint/pwml-recovery` @ `8beb36f6`. Instrument:
`evidence/orch728_disposition_classifier.py` / `.log` · `evidence/g11/ORCH-728/02-classifier-v2.json`.
Bounded, `FINAL SURVIVING COUNT : 0`.

---

# 0. The reframing was right, and the contract already said so

`F-147`'s precondition — *"both would export content their own gold forbids"* — rests on a
misreading of the gold field it cites. Both papers carry `expected_export: partial_only`,
and **`PRODUCT_CONTRACT` § 12 defines that term in the opposite direction:**

> `expected_export = partial_only` is a statement about the **source** and about the strict
> benchmark denominator. **It is not a prohibition on generating PWML.**
> - generate useful research or diagnostic artifacts where possible
> - exclude such papers from the strict-PWML success denominator
> - **do not label them release-ready** unless they independently meet the standard
> - **do not embed gold-set-only policy into the general production pipeline**

And § 13 rules on the specific paper by name:

> **PMC12452463** — *"Correct outcome after the index fix is **`review_required`** with
> `strict_acceptance_eligible=false`. **Never strict success.**"*

**The contract does not ask us to refuse `PMC12452463`. It asks us to emit `review_required`
for it.** What must never happen is calling it `release_ready` or counting it in the strict
denominator. F-147 blocked serialization when the contract only ever forbade *acceptance*.

**This retires ORCH-727's blocker.** No transport-integrity gate is needed to refuse
`PMC12452463`, because refusing it was never the requirement. ORCH-727's measurement stands
— no such general rule is constructible — it is simply no longer load-bearing.

---

# 1. The rule

Derived from the pilot and development payloads. **It reads no gold.** Its inputs are the
live gate reports, the production F-179 predicate, and connectivity counts the payload
already carries. Gold appears in § 4 only as an after-the-fact check.

## The SERIALIZATION FLOOR — all five must hold, or no PWML is written

| | condition | why it is a floor and not a cap |
|---|---|---|
| **F0** | the mode serialises at all | research is diagnostic by design — `acceptance.py:125`, and 0 PWML in 153 research legs |
| **F1** | a canonical payload exists | nothing reached Stage 3; there is no pathway to serialize |
| **F2** | the **final live** stage-3 gate passes | `run_strict_post_normalization_gates` at the shipped payload — a live structural refusal |
| **F3** | the final semantic contract passes, **F-179 included** | `reaction_support_issue(payload) is None` — **this is the anti-invention floor and it does not move** |
| **F4** | a defensible connected core exists | `connected_core_reactions >= MIN_CONNECTED_CORE_REACTIONS` (2), unless a single-reaction pathway was requested |

Any failure → **`diagnostic_only`. Refuse entirely. No PWML.**

## The DISPOSITION — given the floor holds

- **`release_ready`** iff every cap in `classify_release_status` also passes.
- **`review_required`** otherwise.

## The only behavioural change: two blockers become caps instead of refusals

| blocker | today | proposed |
|---|---|---|
| **(a)** contract errors carried *only* by a report stamped `phase: audit_round` | `strict_gates_passed=False` → `diagnostic_only`, no PWML | **cap to `review_required`** |
| **(b)** non-core entity uncertainty — an Unknown-backed enzyme, an unresolved identity outside the surviving core | (already tolerated at the gate; not a cap) | **cap to `review_required`** |

**(a) is justified by the app's own words.** `streamlit_app.py:4055-4060` documents the
`audit_round` report as *"not a verdict about what shipped — the remap below moves the
payload again."* A live contract error at any **other** phase still refuses; only the
superseded snapshot is demoted from refusal to cap.

## Why this is the *smallest* change

`classify_release_status` already applies **five caps of exactly this shape** — semantic,
incomplete-core, connected-pathway, unstated-request, pre-freeze. Each is documented as:
*only from `release_ready`; exactly one step; never to `diagnostic_only`; never applied to a
status the chain already lowered.*

This proposal is a **sixth cap of the identical shape**, plus making the driver's contract
channel phase-aware so it feeds that cap instead of forcing `strict_gates_passed=False`. The
gate-channel precedent for the phase-awareness already exists in the same file at
`driver.py:2506-2521`, where `gate_verdict` was introduced for this exact defect and *"reads
the final report and nothing else, fails closed on every uncertainty."*

**No gate is weakened. No threshold moves. F-179 is untouched and remains a floor, not a cap.**

---

# 2. Classification of every batch failure

24 legs: all 20 unseen-pilot legs, plus the 4 development legs `F-147` named.

## `must refuse entirely` — 20

### F0 · research mode is diagnostic by design — 4
`PMC11172790/research` · `PMC11405693/research` · `PMC12376012/research` · `PMC12542839/research`

Not a defect. Research never serialises and never has.

### F1 · no canonical payload — 10

| leg | why |
|---|---|
| `PMC3480714/strict`, `/research` | `scope_conflict` — and ORCH-724 § 5b showed the **manifest** was wrong, not the product |
| `PMC7232280/research`, `PMC8510960/research`, `PMC12071552/research` | `scope_conflict` at Stage 1 |
| `PMC12326985/strict` | `ambiguous_review_scope` — **the negative control, correctly refused in 20.9 s** |
| `PMC12326985/research` | same paper, `unknown` |
| `PMC12542839/strict`, `PMC13017326/strict`, `/research` | `timeout` at the 60-min ceiling — infrastructure, not biology |

### F4 · no defensible connected core — 1
`PMC11172790/strict` — **0 reactions.** 1 transport, 2 interactions. Correctly refused; this
is the leg ORCH-726 warned must not be counted as a recovery.

### F3 · live pre-export contract failure — 2

| leg | code | rx |
|---|---|---|
| `PMC11405693/strict` | `reaction_enzyme_must_be_protein_complex` — `AtCYP73A5`, a **resolved** protein used as an enzyme without a wrapper | 6 |
| `PMC12376012/strict` | same code — `ceramide synthase` | 15 |

**These two are the honest engineering target if anyone ever reopens.** Both are live,
correct gates catching a *normalization* gap: a resolved enzyme that never got its
single-protein PathWhiz wrapper, while unresolved ones do get one via the Unknown fallback.
`PMC12376012` additionally carries **9 of 15 reactions with no `provenance_lineage`**, so its
refusal is doubly warranted even though F-179 passes it on the strength of the other six.

### F2/F3 · development legs — 3

| leg | why |
|---|---|
| `PMC12180156/strict` @ 2026-08-28 **and** @ 2026-09-02 | **F-179 `no_defensible_reaction_support`** *and* F4 (1 reaction). The gold agrees: *"With zero heme-biosynthesis reactions recoverable, nothing about heme biosynthesis is exportable."* **The `glycine → heme` case stays blocked.** |
| `PMC12452463/strict` @ 2026-09-02 | F2 — `Expected 0 plus tokens, found 1` on `ferric iron (Fe3+)`. A live gate, though a *normalization* one; ORCH-726 called this "luck, not a gate" and that assessment stands |

## `should emit review_required` — 4

| leg | rx | core | F-179 | caps that hold it below release_ready |
|---|---:|---:|---|---|
| `PMC12071552/strict` ✔ *(control — really shipped `pathway.review_required.pwml`)* | 2 | 2 | clean | semantic `requested_pathway_anchors_present`; requested-core threshold; coverage 0.40 |
| **`PMC7232280/strict`** *(Moco, N. crassa)* | 5 | 5 | clean | superseded report (2); 4 Unknown-backed enzymes; semantic anchors; coverage 0.73 |
| **`PMC8510960/strict`** *(MIA, C. roseus)* | 5 | 3 | clean | superseded report (7); 3 Unknown-backed enzymes; semantic anchors; coverage 0.50; fragmented graph 56 % |
| **`PMC12452463/strict`** @ 2026-08-28 | 5 | 4 | clean | superseded report (3); 1 Unknown-backed enzyme; coverage 0.93; fragmented graph 74 % |

**The control is the calibration.** `PMC12071552` is classified `review_required` with **no
superseded-report cap at all** — it lands there on the ordinary caps, exactly as it really
did. The rule reproduces reality on the one leg whose answer is known.

## `should emit release_ready` — 0

Nothing in either cohort is clean enough. Every serialisable leg carries at least a coverage
shortfall or an unmatched requested-core anchor. **This is the honest result** and it is the
strongest evidence the rule is not a loosening: it promotes nothing to strict success.

---

# 3. What this changes numerically

| | today | under the rule |
|---|---:|---:|
| unseen-pilot strict legs emitting a PWML | **1** | **3** |
| … of 10 strict legs attempted | 1/10 | 3/10 |
| … of 6 *evaluable* strict legs (ORCH-724 § 5 denominator) | 1/6 | **3/6** |
| `release_ready` among them | 0 | **0** |
| strict-acceptance rate | unchanged | **unchanged** |
| development `PMC12452463` | no PWML | **`review_required`**, as `PRODUCT_CONTRACT` § 13 requires |
| `glycine → heme` (`PMC12180156`) | refused | **still refused**, by F-179 |

**Two additional unseen PWMLs, both `review_required`, both with a complete supported core,
zero rows without lineage, and F-179 clean.** No paper moves into the strict denominator and
nothing becomes `release_ready`.

---

# 4. Gold cross-check — performed after the rule was derived, never as an input

| paper | gold / contract says | the rule says | agree? |
|---|---|---|---|
| `PMC12452463` | § 13: *"Correct outcome … is `review_required` with `strict_acceptance_eligible=false`. Never strict success."* | `review_required` | **yes** |
| `PMC12180156` | gold: *"With zero heme-biosynthesis reactions recoverable, nothing about heme biosynthesis is exportable"*; `mechanistic_relevance: context_only` | refuse entirely | **yes** |
| `PMC12071552` | really shipped `pathway.review_required.pwml` | `review_required` | **yes** |

The rule consumed none of these and reproduces all three. That is the validation.

---

# 5. The seams, if this is authorized

Not implemented. Named so a card can be written against them.

1. **`src/t2pw/batch/driver.py::_blocking_reports`** (1026–1056) — exclude reports stamped
   `phase: audit_round` from the blocking scan; **fail closed** when no later boundary report
   exists; keep the historical arithmetic for artifact sets predating the phase stamp.
   Precedent: `gate_verdict`, same file, `:2506-2521`.
2. **`src/t2pw/batch/driver.py`** at `:2583` — route a superseded-only contract failure to
   the export path with a `review_required` cap instead of `_finalize_gate_failure`.
3. **`src/t2pw/pipeline/release_status.py::classify_release_status`** — add the sixth cap,
   `superseded_intermediate_report` / `non_core_identity_uncertainty`, with the same four
   restrictions the five existing caps carry. Default `None` so every existing caller stays
   byte-identical.

**G9 note for whoever writes the card.** Seams 1–2 correct pre-existing observable behaviour
and need a proof that **fails behaviourally on the base SHA**; the classifier in this pass is
that proof's skeleton. Seam 3 is a genuinely new capability and needs an **explicitly
labelled new acceptance test**, not a fabricated base failure.

**What must not be touched:** F-179, the stage-3 gate, `reaction_enzyme_must_be_protein_complex`,
the connected-core floor, or the strict denominator. The rule's whole value is that it moves
only the *disposition*, never the *floor*.

---

# 6. Honest limits

- **The two `PMC12452463` archives disagree.** The 2026-08-28 payload clears the floor and
  earns `review_required`; the 2026-09-02 payload is refused on a `Fe3+` plus-token. Both
  predate C-118. **Which shape a fresh run produces is not knowable without running it**, and
  the charter forbids that.
- **No human has reviewed the content of the two new PWMLs.** They pass every deterministic
  gate and carry full lineage; that is not the same as being biologically right.
  `PILOT-MANUAL-REVIEW.md` remains the blocking item.
- **`review_required` is a claim about *serialisability*, not correctness.** It says a
  defensible supported core exists and a human must check the rest. It is not evidence the
  biology is right, and it must never be reported as such.

---

*Read-only. No production code, gold data, run artifact or `main` was modified.*
