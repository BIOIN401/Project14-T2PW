# Decision packet — F-150 half 2

**The one open product-owner question of the sprint.** It needs authority the Lead Orchestrator does
not hold, and it is **not chartered and not implemented.**

> **Should `supported_reactions_complete` be set on any gold case — and if so, which?**

**This is a decision about what the benchmark MEANS on every future run, not a data correction.**
That is the whole reason it is separated from F-150 half 1, which is an internal gold inconsistency
and nothing more. **The two must not be merged, and half 2 must not drift into half 1's scope.**

No production code, no scorer and no gold byte was changed to produce this document. Every number
below was read from the committed tree at `ab16a3a7`.

---

## 1. The measurement — what the field does today

`supported_reactions_complete` is declared at `src/t2pw/bench/goldset.py:460` with **default
`False`**, and read at `src/t2pw/bench/semantic.py:743`. It appears **zero times** in
`src/t2pw/bench/gold/pinned_v1.json`, so **all ten pinned cases carry the default.**

Surveyed across the pinned set:

| Paper | `mechanistic_relevance` | `expected_export` | signatures declared | `max_retained_reactions` | `supported_reactions_complete` |
|---|---|---|---|---|---|
| `PMC12444477` | core | partial_only | 2 | — | *(default false)* |
| `PMC13231680` | context_only | partial_only | 0 | **0** | *(default false)* |
| `PMC12657337` | core | partial_only | 3 | — | *(default false)* |
| `PMC12421875` | core | partial_only | 3 | — | *(default false)* |
| `PMC12312563` | partial | partial_only | 1 | — | *(default false)* |
| `PMC12856317` | partial | partial_only | 1 | — | *(default false)* |
| `PMC12180156` | context_only | partial_only | 0 | **2** | *(default false)* |
| `PMC12096016` | core | **strict_exportable** | 3 | — | *(default false)* |
| `PMC12452463` | partial | partial_only | 3 | — | *(default false)* |
| `PMC12782028` | partial | **strict_exportable** | 3 | — | *(default false)* |

**`max_retained_reactions` is set on exactly two cases, and both are the negative controls.** That is
the fact that generates the question.

---

## 2. What the flag actually switches — quoted, not paraphrased

`semantic.py` computes `unsupported_verdict_evaluated = bool(complete or not unsupported_rows)`. The
verdict is reached in **exactly two** situations, and the module says so itself:

> *the signature set is exhaustive, so an unmatched row is by definition unsupported and
> `false_positives` counts it; or every retained row matched a quote-verified signature, so "zero
> unsupported" is a measurement rather than an assumption*

With a **subset** set and **at least one unmatched row**, it is not reached at all. Concretely, with
`complete = False`:

- every `unsupported_retained_reaction` finding is **deleted** before the result is returned;
- `false_positives` is reported as **`None`**, not as `0`;
- `precision` is **withheld**, and the label degrades to `attribution rate (signature set is a SUBSET)`;
- the summary is stamped **`UNSUPPORTED-REACTION VERDICT NOT EVALUATED`**.

**That suppression is correct and it stays.** It exists because the alternative — reporting the hard
zero as a clean result — *"collapses 'not evaluated' into 'passed'"*, which **`PRODUCT_CONTRACT` § 11
forbids.** The instrument is already honest. **Nothing here is a defect to be fixed.**

The module also states the remedy and its exclusivity:

> *Only an exhaustive `supported_reactions` list, or a `max_retained_reactions` ceiling, can make this
> question answerable for this paper.*

**So the question is genuinely binary and genuinely load-bearing:** with the flag unset on every
non-control case, Priority 2's unsupported-reaction verdict can never be evaluated on a paper that is
not a negative control — not on this run, and not on any future one.

---

## 3. The standing limit that must travel with every report

> **Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry a run
> produced. Any report quoting it must carry that limit.**

This holds **today, under the status quo**, and it holds **regardless of how this question is
answered.** It is not an argument for either option; it is the caveat that stops the number being
misread while the question is open.

---

## 4. What setting the flag would cost — the risk is asymmetric

Declaring a signature set exhaustive is a claim that **every reaction the paper supports has been
hand-written into the gold.** If that claim is wrong, every unmatched retained row is scored as a
**false positive** — as invented chemistry.

The module records what that miscall looked like when it was measured:

> *an unmatched row may be invented, may be a step the paper states in a form no signature was
> written for, or may be a legitimate cross-paper RAG addition, and those are indistinguishable from
> the signature list alone*

and

> *Calling those hallucinations would have reported **227 fabricated reactions** in a run that
> produced far fewer.*

**Weigh that against the declared signature counts: 0 to 3 per case, and 2, 3 and 3 on the three
`core` papers.** Declaring a three-signature list exhaustive for a core mechanistic paper is a strong
biological claim, not a bookkeeping change. It should be made by someone willing to defend the
completeness of that list against the paper itself.

**The failure modes are not symmetric.** Leaving the flag unset yields a number that is *withheld and
labelled as withheld*. Setting it wrongly yields a number that is *stated and wrong, in the direction
of accusing the pipeline of inventing chemistry it did not invent.* Under `PRODUCT_CONTRACT` § 11 the
first is the tolerable failure.

---

## 5. Options

| | Option | What it buys | What it costs |
|---|---|---|---|
| **A** | **Leave every case at the default.** Priority 2's unsupported-reaction verdict stays unevaluated outside the negative controls; the standing limit in § 3 travels with every report | Zero risk of a false fabrication count. The instrument keeps saying exactly what it did and did not evaluate | Priority 2 measures nothing on any paper that is not a negative control — **permanently**, not just on T-108 |
| **B** | **Set it on ONE deliberately chosen case**, after a biological re-read that certifies the signature list exhaustive for that paper | Priority 2 becomes answerable on one real paper. The smallest step that makes the metric mean anything | Requires a genuine completeness audit of that paper's chemistry. **The audit is the cost, not the edit** — and it is not a Lead judgement |
| **C** | **Set it on several or all cases** | Priority 2 fully live | **Recommended against.** It multiplies option B's audit burden by ten and risks the 227-style miscall on every paper at once |
| **D** | **Extend `max_retained_reactions` to non-control cases instead** | The module names this as the *other* thing that makes the verdict answerable, without any exhaustiveness claim | A ceiling is a different question with its own semantics. **Out of this packet's scope — flagged so it is not lost, not proposed** |

---

## 6. The Lead's position — and its explicit limit

**If forced to act without authority, the Lead would take A**, on the § 4 asymmetry alone: the
status quo's failure mode is a withheld number that announces itself as withheld, and the alternative's
failure mode is a confident wrong accusation of invented chemistry.

**But the Lead does not hold this decision and is not taking it.** Option A is not free — it accepts
that a whole benchmark priority stays structurally unable to measure anything on a real paper, which
is a real cost to the product and precisely the kind of cost a product owner, not an orchestrator,
should choose to pay.

**Option B is the option worth the product owner's attention**, and the honest framing of it is: *the
edit is trivial and the audit behind it is not.* If B is chosen, the case should be named by the
product owner, and the completeness audit should be an independent biological review — not a Lead
inference and not a side-effect of another card.

---

## 7. What happens if nobody decides

**The status quo is option A by default**, and it is stable and honest — nothing breaks and nothing
silently drifts. But it must be recorded as a **decision that was defaulted into, not one that was
taken**, so a later reader does not mistake ten unset flags for ten considered judgements.

**T-108 can launch under option A.** This question is **not** a T-108 blocker: the instrument is
honest today, and a T-108 report that carries the § 3 limit is a correct report. What T-108 must not
do is quote a Priority 2 number **without** that limit attached.

---

## 8. Guardrails that bind whatever is decided

- **Half 2 is separate from half 1 and stays separate.** Half 1 is an internal gold inconsistency;
  half 2 changes what the benchmark measures. **A single commit containing both is a reject.**
- **No agent may set this flag without an explicit ruling recorded in `DECISIONS.md`.**
- **Setting it must never be justified by a benchmark number moving in a pleasing direction.** That
  is `policy_disagreement` dressed as `gold_data_defect`, and merge rule 6's reasoning applies:
  a gold change that improves a score is the case that most needs an independent reviewer.
- **The § 3 limit is quoted in any report that cites Priority 2**, under every option including B.
- If B is chosen, the chosen case and the audit that certified it are recorded together. **The flag
  and its justification are one artifact.**
