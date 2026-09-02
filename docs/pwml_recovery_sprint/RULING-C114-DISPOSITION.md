# Lead ruling — the D-088 diagnostics leave the coverage verdict and become their own artifact

**Lead Orchestrator and Integration Authority, 2026-09-02, `ORCH-719`.**
**Adjudicating the disposition `REV-114` routed up, on the evidence of a widened consumer sweep.**

**C-114 is NOT merged and will not merge in its present shape.** Its measurement work stands and is
reused; what changes is **where its output lands**.

---

## 1. The question

C-114 adds five diagnostic keys to the object `evaluate_core_coverage` returns. Two byte-pinned
consumers break. Two routes were on the table:

| | route |
|---|---|
| **A** | Keep the keys in the verdict. Register an `_INVERTIBLE` `_DELTAS` entry in `test_c011_freeze_seam_golden_equivalence.py` (its C-052 entry at `:479-487` is the working precedent) and an exclusion in `test_c074_strict_core_floor.py`. |
| **B** | Emit the diagnostics as a **separate artifact** beside `coverage_summary.json`. No pinned detector is amended. |

`REV-114` recommended **B** and was explicit that the choice was mine.

---

## 2. The measurement that had to come first

`REV-114` said, correctly, that the consumer population was **not enumerated** — it found one
collision by grep, I had found none, and it found a second by widening. It warned that **a third
would change the design answer**, so ruling before measuring would have been the error F-170 is
about.

**Widened sweep, 34 deterministic consumer files at the C-114 tip** (G11 `ORCH-719/12`,
`10 failed, 863 passed, 9 skipped in 78.52 s`, zero survivors):

| failures | attribution |
|---|---|
| `test_c011_freeze_seam_golden_equivalence` ×3 | **C-114** |
| `test_c074_strict_core_floor` ×2 | **C-114** |
| `test_c102_coverage_denominator` ×2, `test_c106_mutation_harness_executable` ×1 | **F-171**, fixed by C-115, which is merged to integration but is not in this worktree's base |
| `test_c075_source_support_armed` ×1, `test_measurement_tree_pin` ×1 | **pre-existing** — both reproduce at base `88b11c9c` (G11 `ORCH-719/13`, `2 failed, 46 passed`) |

**There is no third collision.** C-114's consumer set is exactly the two files `REV-114` named.

**Honest limit:** four **AppTest-heavy** files were excluded from that sweep and are being run through
the authoritative split-process gate separately. Two of them assert membership only
(`"quarantine_coverage" not in session_state`) and cannot be affected by an added key. **This ruling
is contingent on that gate showing no collision; if it shows one, the ruling strengthens rather than
reverses**, because a third collision was the reviewer's argument *for* route B.

---

## 3. The ruling — **route B**

The count of collisions is not what decides it. **What `test_c074_strict_core_floor.py:462` detected
does.**

That test compares a demoted leg's coverage record against the same payload under a different
request, excluding **only** `requested_context` — the field that *is* the designated request echo.
`subprocess_coverage` is derived from `requested_context.main_subprocesses`, so the diff put
**request-derived content into the verdict outside the one key set aside for it.**

**That is a change in what the document means, not in how many keys it has.** Amending C-074 to
exclude the D-088 keys would permanently silence the only test that noticed — and it noticed
correctly.

Three further reasons, in the order I weight them:

1. **F-168 is the tiebreak.** These fields must **never** become a gate input: Stage-0 subprocess
   lists are draw-unstable, `0 of 14` paper/mode pairs stable across archived draws. Route B makes
   *"never a gate input"* a property of **where the data lives**. Route A leaves it a property of a
   docstring plus one test. C-056c's *"a carrier that could move a verdict would be a second gate
   wearing a record's name"* is strongest in exactly the form where the carrier is **not inside the
   object the gate receives**.
2. **Every amendment is a place a future regression can hide.** Route A amends two byte-pinned
   detectors over 39 legs. Both are doing the job they were built for; both caught this.
3. **Cost is a wash, and route B is not cheaper — it is safer.** Route A needs a card owning two test
   files; route B needs a card owning `evaluate_core_coverage`'s return shape and one artifact
   writer. **Neither fits inside C-114.** I am not choosing the cheap option and should not be read
   as claiming to.

### The drift objection, and why it is answerable by construction

If diagnostics live in a second file they can disagree with the verdict. **The answer is
construction, not discipline**, and it is binding on the successor card:

1. computed in the **same call**, from the same `unmatched_terms`, `core` and payload;
2. written by the **same function** that writes `coverage_summary.json`, so *"verdict written,
   diagnostics absent"* is unreachable;
3. the document carries `unmatched_terms` **verbatim**, with a test asserting agreement on **all 83
   archived legs**.

Drift then becomes **detectable**. The detectors route A sacrifices are **not recoverable**.

---

## 4. The condition under which this ruling reverses, stated so nobody has to infer it

**If the product owner's eventual D-088 clause-9 ruling makes these fields an INPUT to a release
decision, route B is wrong and route A is right** — an input belongs in the object the gate receives.

**On today's facts they must never be an input**, and F-168 is the reason. **This ruling is therefore
contingent on a decision that has not been made, and it says so rather than pretending to be
unconditional.** The open question is
`DECISION-PACKET-D088-RUNTIME-CAP.md`; nothing here pre-empts it.

---

## 5. What happens to C-114

**Not merged. Not discarded.** Its branch `agent/c114-d088-diagnostics` @ `47b9c517` is preserved as
the record, and its work is **reused rather than redone**:

- the census rules are **verified copies**, not reimplementations — stoplist 48/48 identical, token
  rule character-identical, the `" | "` joiner preserved;
- `_normalize` vs the census's `norm` agree on **0 divergences across 374 anchors on 83 legs**;
- the decision hash is **provably unmoved** — recomputed identical at base and tip on four legs and
  equal to each leg's archived value;
- **374 / 60 / 90 reproduce with no tuning**, confirmed by the reviewer's own independent
  implementation.

**One defect must not travel with it.** `strict_quarantine.py:1119` puts a gold `paper_id`
(`PMC12782028`) into `src/`, which fails a test whose own docstring calls it **"THE PRODUCT OWNER'S
EXPLICIT RULE."** The successor card fixes that in its own words; it is not carried forward and not
argued with.

Two `REV-114` refinements carry forward as requirements, not suggestions:

- the `covered` docstring claims equivalence with the probe's rule; the probe also unions the
  admission `label`, the module uses `core_terms` only. **The two agree on all 73 testable legs —
  measured — but the sentence asserts an equivalence that does not hold in general.** Fix the
  sentence, not the rule.
- `unmatched_anchor_diagnostics` will record `{"term":"ATP","in_payload":false}` on
  `PMC12096016/strict`, and **F-169's amendment establishes that is false** — ATP is present as
  `Adenosine triphosphate` and *is* wired. The census rule is kept **deliberately**, so the totals
  stay comparable; the row therefore needs a **rule marker** so a future reader cannot mistake it for
  *"not extracted."* D-088 clause 7 protects evidence, and **a preserved falsehood is not evidence.**

---

## 6. What this ruling does NOT decide

- **Not** the D-088 runtime cap. That is the product owner's and it is open.
- **Not** F-169's `ATP` matcher gap. Classified, sized, located, still unchartered.
- **Not** whether `test_c011` or `test_c074` should ever be amended for some other card. This says
  only that **a diagnostic-only addition is not a good enough reason.**
- **Not** T-108. Step 9 stays NO-GO.
