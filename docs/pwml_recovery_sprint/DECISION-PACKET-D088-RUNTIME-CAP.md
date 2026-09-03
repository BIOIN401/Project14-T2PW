# Decision packet — D-088 clause 2 cannot be implemented at RUNTIME this wave, and the reason is structural

**Raised by the Lead Orchestrator, 2026-09-02, `ORCH-719`, integration tip `36570a37`.**
**For the product owner. Nothing is implemented. No production, scorer, test or gold file is changed
by this packet.**

> **This is not a request to reopen D-088.** D-088 is LOCKED and its ruling is not in question. It is
> a report that **two of its own clauses point in opposite directions given what production can
> actually compute**, and a request for a ruling on which one yields.

---

## 1. The conflict in one paragraph

**D-088 clause 2** says missing cofactors, currency metabolites, regulators and ancillary proteins
*"should normally produce completeness WARNINGS or secondary-score deductions — not automatically
remove `release_ready`."* `release_ready` is a **runtime** status, set in
`release_status.classify_release_status`. **D-088 clause 10** says *"do NOT simply filter cofactors,
match against the entity list, or relax the cap without replacing it with reaction-level coverage."*

**Every reaction-level replacement available to production this wave is disqualified**, each for a
different and independently established reason. So clause 2 asks for a runtime change that clause 10
forbids making, and the wave cannot satisfy both.

---

## 2. The four candidate replacements, and why each fails

I evaluated these against the archived corpus before chartering anything. **The required
discrimination is D-088's own:** `PMC12096016/strict` must lose the cap and `PMC12782028/strict` must
keep it. **A change that clears both has removed the measurement rather than improved recall and is a
reject** — `HANDOFF.md` § 5.2a step 6.

### Candidate A — rely on the reaction-level thresholds production ALREADY has

Drop the all-anchors cap and let the existing floors decide: `min_core_coverage` 0.5, the core
process minimum, the CONNECTED-PATHWAY FLOOR. Arguably clause 10 is satisfied because reaction-level
coverage already exists and stays.

| leg | coverage | core processes | outcome |
|---|---|---|---|
| `PMC12096016/strict` | 0.750 ≥ 0.5 | 9 ≥ 1 | **released** ✔ |
| `PMC12782028/strict` | **0.538 ≥ 0.5** | 3 ≥ 1 | **released** ✘ |

**REJECTED — it clears both.** `PMC12782028` is a known reaction-recall failure whose upstream
mevalonate arm is absent, and this releases it. It is the exact outcome step 6 exists to catch.

### Candidate B — key the hard cap to Stage-0's own `main_subprocesses`

The one process-level specification production already holds at the coverage seam, carried in
`requested_context`, needing no gold and no Stage-0 redesign. **On the T-108 tree it gives the
required discrimination exactly** — `PMC12096016` 0 uncovered, `PMC12782028` 2 uncovered
(`mevalonate pathway`, `methylsterol demethylation`) — identically at three stoplist strengths.

**REJECTED by F-168**, measured over all 83 committed legs before it was chartered:

```
paper/mode pairs with >= 2 archived draws            : 14
Stage-0 named an IDENTICAL subprocess set every time :  0
archived PMC12782028/strict draws                    :  7
  draws naming a mevalonate stage                    :  4
  draws NOT naming one                               :  3
```

On `runs_verify/2026-08-21_2239` this rule **releases `PMC12782028/strict`** — not because recall
improved but because that draw did not name the arm the pipeline was missing. **The specification is
itself an LLM draw. A gate keyed to it has a random denominator**, and removes measurements
stochastically with no diff to review.

### Candidate C — filter cofactors and currency metabolites by a static vocabulary

**REJECTED by clause 10 in terms** — *"do NOT simply filter cofactors"* — unless paired with a
reaction-level replacement, and A and B are the only ones available. **F-169 makes this strictly
worse than it looks:** the `ATP` row on `PMC12096016` is not a completeness gap at all but a matcher
gap, and it sits inside the population this candidate downgrades. Filtering cofactors would hide a
real defect permanently, which is the failure mode clause 10 was written to prevent.

### Candidate D — read the curated expected-reaction set in production

The curated ten-paper dataset (`HANDOFF.md` § 5.2a step 4, this wave's long pole) **is** a
non-draw-dependent reaction-level input, and it discriminates the two legs correctly by construction.

**REJECTED by `PRODUCT_CONTRACT` § 12** — *"do not embed gold-set-only policy into the general
production pipeline."* A per-`paper_id` curated expectation is gold-set-only by definition. It would
also make production unable to classify any paper outside the benchmark, which is the whole product.

---

## 3. What this leaves, stated plainly

**Production cannot distinguish `PMC12096016/strict` from `PMC12782028/strict` without a per-paper
curated input it is not allowed to have.** Both legs pass every structural gate, both pass semantics,
both clear the coverage ratio, both clear the core-process minimum. The **only** thing that separates
them is knowledge of what the paper's pathway should contain — and that knowledge is either curated
(forbidden in production) or drawn from Stage 0 (unstable, F-168).

**This is not a defect in D-088 and not a defect in the pipeline.** It is the boundary between what an
acceptance instrument can know and what a general-purpose production classifier can know, and the two
named consequences happen to straddle it.

---

## 4. What I propose to do, and what I am asking

**Chartered and proceeding — C-114, the unblocked arms:**

1. **The acceptance instrument moves to reaction-level completeness.** Priorities 4 and 5 are scored
   on curated core-reaction and major-subprocess recall against the new dataset. **This is where
   D-088's hard-completeness decision genuinely moves**, it satisfies clauses 4, 5 and 9 in full, it
   is legitimate for a scorer to read curated expectations, and it delivers the required
   discrimination without touching a production gate.
2. **Production records the typed diagnostics** D-088 clauses 6, 7 and 8 require — subprocess
   alignment, payload-present-but-unwired, per-subprocess coverage — **recorded and never read by any
   gate**, on the C-056c `semantic_check_evaluability` precedent (*"a carrier that could move a
   verdict would be a second gate wearing a record's name"*). F-168 is exactly why these must be
   diagnostics and not inputs.

**Held, pending your ruling — the runtime arm:**

3. **The INCOMPLETE-CORE CAP in `release_status.py` is UNCHANGED this wave.** `PMC12096016/strict`
   therefore **remains `review_required` at runtime**, which is a visible shortfall against D-088's
   expected-consequences table, and I am flagging it rather than quietly delivering around it.

**The question I need answered:**

> **Given that no permissible reaction-level replacement exists in production this wave, does clause 2
> yield to clause 10 (cap unchanged, runtime shortfall accepted, acceptance instrument corrected), or
> does clause 10 yield to clause 2 (cap relaxed on a cofactor vocabulary, accepting that `PMC12782028`
> would be released at runtime and kept failing only in the benchmark)?**

**My recommendation is the first**, for three reasons: a cap is monotone and can only ever remove a
strict success, so leaving it in place cannot manufacture one; merge rule 7 is already satisfied
because the pathway is preserved and exported as `review_required` rather than dropped; and the
second option makes the runtime status and the benchmark disagree about the same leg, which is the
condition `PRODUCT_CONTRACT` § 11 was written to prevent.

**A third option exists and I am not recommending it yet:** give production a *general*, non-paper-keyed
reaction-level requirement — a typed pathway-shape specification that any paper could be scored
against. That is real work, it is a Stage-0-adjacent redesign, and `HANDOFF.md` § 5.2a step 3 forbids
it in this finishing wave. **It is the right long-term answer and it should be chartered as its own
wave, not smuggled into this one.**

---

## 5. What is NOT being asked

- **Not** to reopen D-088. Its ruling on the biology stands and this packet assumes it.
- **Not** to change gold. `pinned_v1.json` stays byte-identical at
  `36f4b7b690b577f72882c3045ca6728d1ec8d9d1`; the curated dataset is a **separate new file**, which
  is deliberate — F-165's lesson is that moving the gold blob makes milestone counts incomparable,
  and the Priority-1 instrument must not move while Priorities 4 and 5 are being rebuilt.
- **Not** to relitigate T-107 or T-108. Both remain immutable and T-108 remains NO-GO.
- **Not** a licence for anyone to set `supported_reactions_complete`. D-087 governs that and is
  untouched.

---

## 6. AMENDMENT, same day — § 4's proposed split does NOT work, and the ask is narrower than I wrote it

**Preserved rather than rewritten, per the sprint rule that a failed proposal is kept beside its
correction.** § 4 proposed proceeding with an *"acceptance instrument moves to reaction-level
completeness"* arm while holding the runtime arm for a ruling. **That arm cannot deliver D-088's
expected consequences, and I should have checked the scorer before proposing it.**

### What I checked, and what it says

**Priority 5 reads the frozen RUNTIME release status, directly.**

```
release_status.py:1261     strict_acceptance_eligible = (status == RELEASE_READY)
acceptance.py:1560         if strict_leg.passed and strict_leg.deliverable
                              and strict_leg.strict_acceptance_eligible:
                                  strict_ok.append(pid)
```

`strict_ok` is Priority 5's numerator. So a leg the INCOMPLETE-CORE CAP held at `review_required` is
outside the numerator **by construction**, and no computation the acceptance instrument performs can
put it back.

**And the scorer already refuses to try, in terms.** `acceptance.py:1146-1152`, on Priority 5:

> *"UNCHANGED, and DELIBERATELY so. A leg the runtime froze as `review_required` for a coverage block
> stays out of the strict numerator even when that block clears under D-072: this module scores runs,
> it does not reclassify them, and promoting a frozen record on a rescored ratio is exactly the
> post-freeze repair merge rule 8 forbids."*

**That is the answer to my own proposal, written down before I made it.** D-072 already faced this
exact question and settled it: the reconciliation reports whether a coverage block *would* survive;
**it never re-issues the release decision.** An acceptance-instrument arm that moved Priority 5 would
be merge rule 8 post-freeze repair with a new name.

### What this corrects

| § 4 said | Correct |
|---|---|
| The acceptance arm delivers D-088's hard-completeness move | **It cannot.** It can add a reading beside Priority 5, exactly as D-072's `requested_core_coverage` already does. It cannot change Priority 5 |
| The runtime arm is one of two arms, held for a ruling | **It is the ONLY arm.** The runtime cap is the sole lever on D-088's expected consequences |
| The ruling chooses between two deliverable paths | **The ruling chooses between a runtime cap change with no permissible implementation, and accepting that D-088's runtime consequences are not delivered this wave** |

### What survives § 4 unchanged

- **The four rejected candidates in § 2 stand.** Nothing here rehabilitates A, B, C or D; the
  amendment concerns what to do *given* that they all fail.
- **Deliverable arm 2 stands** — production records the typed diagnostics D-088 clauses 6, 7 and 8
  require, recorded and never read by a gate. That is real, it is unblocked, it satisfies clause 7's
  *"preserve all current raw requested anchors and unmatched-anchor diagnostics"*, and it is what
  makes the 60 and 90 populations separately visible per `HANDOFF.md` § 5.2a step 7.
- **The curated ten-paper dataset stands** and is still the long pole. It is clause 9's required
  replacement input, F-168 proved it is the only non-draw-dependent one, and it is what any future
  runtime rule — or the general pathway-shape specification of § 4's third option — must be validated
  against. **Building it is not wasted by this amendment; it is the prerequisite either way.**
- **The recommendation is unchanged and is now better supported.** Clause 2 yields to clause 10, the
  cap stays, and the shortfall is recorded. The scorer's own D-072 comment is independent authority
  for it.

### The ask, restated in its corrected form

> **D-088's expected consequences for `PMC12096016` cannot be delivered this wave by any permissible
> means. Do you accept that outcome — cap unchanged, diagnostics added, curated dataset built,
> Priority 5 remaining `0/2` — or do you direct that the cap be relaxed on a cofactor vocabulary
> despite clause 10, accepting that `PMC12782028` would be released at runtime and would remain a
> failure only in a reading beside the score?**

**I recommend the first, and I want to be explicit that it means the headline number does not move.**
`HANDOFF.md` § 8's first transferable lesson is that *an unchanged number can hide a completely
changed result*; this is the converse and it needs saying just as plainly. **A wave that correctly
declines to move a number is not a wave that achieved nothing, and a wave that moved it by any of
candidates A through D would have achieved less than nothing** — it would have removed the
measurement that keeps `PMC12782028`'s missing mevalonate arm visible.

### Standing lesson

**I proposed a workaround without reading the consumer it was supposed to satisfy.** The refutation
was a comment in the function I was proposing to change, written by D-072 for the same reason, and
finding it cost one grep. **Before proposing that instrument X compensates for gate Y, read what X
actually reads** — the seam between them is where the assumption lives, and it is usually already
documented by whoever hit it first.

---

## 7. RESOLVED — 2026-09-02. **The product owner ruled: clause 10 controls. The cap stays.**

**Recorded as `D-089` (LOCKED).** This packet is CLOSED. Nothing below reopens it.

### 7.1 The answer to § 6's corrected ask

The ask was:

> Do you accept that outcome — cap unchanged, diagnostics added, curated dataset built, Priority 5
> remaining `0/2` — or do you direct that the cap be relaxed on a cofactor vocabulary despite clause
> 10, accepting that `PMC12782028` would be released at runtime and would remain a failure only in a
> reading beside the score?

**Answered: the first. The outcome is accepted.**

| Directive from the ruling | Effect here |
|---|---|
| Keep the INCOMPLETE-CORE CAP unchanged | `release_status.py:1087` **untouched this wave** |
| Do not relax it on a cofactor vocabulary | **Candidate C stays rejected** |
| Do not match anchors merely against the entity list | **Candidate A stays rejected** |
| No Stage-0 redesign in this finishing wave | **Candidate B stays rejected**; step 3 holds |
| No gold change, no curated expectations inside production | **Candidate D stays rejected**; `PRODUCT_CONTRACT` § 12 holds |

**All four candidates in § 2 were rejected by the analysis and are now rejected by ruling.** The
packet's own recommendation and the ruling agree; the ruling is the authority, not the agreement.

### 7.2 The one thing the ruling adds that the packet did not say

The packet framed the choice as *score versus measurement.* **The ruling names it more precisely:
the choice is which direction to be wrong in.**

> the current system cannot implement that distinction safely without also releasing `PMC12782028`,
> whose upstream mevalonate arm is genuinely missing. For this release, **choose the conservative
> false negative.**

**`PMC12096016/strict` staying `review_required` is a FALSE NEGATIVE and the ruling says so.** It is
not a correct classification that happens to be inconvenient. The pipeline recovered that pathway
adequately and the instrument declines to say so, because the only available way to say so would
also have said it about a pathway that is genuinely missing an arm. **Calling it what it is, in the
document that accepts it, is the whole point of accepting it explicitly.**

### 7.3 The product principle is NOT withdrawn

D-088's clause 2 intent stands and was restated in the ruling verbatim. **What is deferred is its
implementation, not its correctness.** The replacement — a stable, general, non-paper-keyed
reaction/subprocess completeness specification distinguishing defining participants, optional
cofactors, extracted-but-unwired entities and genuinely absent core reactions — is **registered as
follow-up work for the RAG / LLM evaluation phase.**

**That is the third option § 4 named and explicitly declined to smuggle into this wave.** It now has
an owner phase and a written scope, which is what § 4 asked for.

### 7.4 What must be preserved, by ruling

- `PMC12096016`'s pathway output stays **preserved and available for review** — never dropped.
- The **C-116 separate diagnostics**, so the **60** subprocess-aligned and **90**
  payload-present-but-unwired anchors stay separately visible.
- The full **374**-anchor census and its **corrected population split** (F-169 and its amendment —
  the 12 ATP rows that are a matcher gap and not a completeness gap).

### 7.5 The claim that must never be made

**Priority 5 at `0/2` is an accepted conservative limitation. It is not delivery of D-088 clause 2.**
Any report, handoff, badge, summary or successor session that describes the cap's survival as
"D-088 implemented" is wrong, and D-089 § 3 exists so that the error is catchable by reading rather
than by re-deriving.
