# DECISION BUNDLE — F-132 and the Priority-1 / Priority-4-5 contradiction

**For the product owner. One ruling is requested. Nothing in this packet has been applied.**

Prepared 2026-08-27 by the Lead Orchestrator at integration tip `14121d5`.
Every number below is measured from committed artifacts. **No leg was re-run to produce it, no
cohort was repeated, no live model call was spent.**

**This is the only thing that unblocks T-107.** It is larger than any card in the queue, and it is
not a decision an orchestrator may take on the product owner's behalf.

---

## 1. The exact contradiction

> **On the affected papers, Priority 1 and Priority 4/5 score the same rows in opposite directions,
> so neither is currently a measurement of pipeline quality.**

The gold set marks certain entities **forbidden to export** — assay reporters, competing-branch
sinks, prose headings, regulators mistaken for metabolites. The pipeline's Stage-0 anchor draw,
which supplies the **requested-core term list**, is unaware of that forbidden list and draws those
same entities as requested biology.

The result, on one and the same entity:

| The pipeline… | …is scored |
|---|---|
| **exports** it with a real accession | **Priority 1 failure** — a false real identifier |
| **omits** it, obeying the gold | **Priority 4/5 penalty** — an unmatched requested-core term |

**There is no third behaviour.** The pipeline is penalised either way, by two different instruments,
for the same rows.

The sharpest instance: on `PMC12782028/strict` the gold-forbidden unmatched terms are `LIPA`, `LBR`,
`SREBF1`, `SREBF2` — **the exact four Priority-1 survivors on that paper.** The same four rows are
simultaneously both failures.

---

## 2. Affected papers and counts

**Corpus-wide** (`evidence/g11/ORCH-702/03-f132-forbidden-anchors.json`, over every committed
`quarantine_report.json`):

| | |
|---|---|
| legs carrying unmatched terms | **52** |
| unmatched terms in total | **281** |
| of those, **gold-forbidden identifiers** | **62 — 22%** |
| legs affected | **32** |
| papers affected | **6** |

Papers: `PMC12096016`, `PMC12312563`, `PMC12444477`, `PMC12452463`, `PMC12782028`, `PMC12856317`.

**Roughly one coverage penalty in five, corpus-wide, is levied for failing to match a term the gold
forbids exporting.** It spans four of the gold's own mechanism kinds — `placeholder_product`,
`heading_or_prose`, `regulator_as_metabolite`, `cofactor_as_protein`. This is systemic, not a
single-case quirk.

**Priority 1's live count is SIX, not eight** — the two PLP rows are already withheld by C-081
(`b869780`), confirmed by replay through the shipped predicate. The six:

| # | entity | leg | gold class | mechanism |
|---|---|---|---|---|
| 1–2 | `NAD+`, `NADH` | PMC12096016/research | reporter species | inferred standard chemistry (F-118) |
| 3–4 | `LIPA`, `LBR` | PMC12782028/research | `heading_or_prose` | entity admitted without a supporting span |
| 5–6 | `SREBF1`, `SREBF2` | PMC12782028/research | `regulator_as_metabolite` | as 3–4 |

**Four of the six are simultaneously a Priority-4/5 coverage penalty.**

On `PMC12096016` the gold `export_rationale` says verbatim: *"Export must exclude MenD, LDH and the
transport mentions."* `MenD` is nevertheless drawn as a requested-core term in **three of six**
committed draws, and `LDH` in one.

---

## 3. Current product-contract language

**`PRODUCT_CONTRACT` § 2 — Correctness.** Retained elements must be defensible; *"no false real
identifiers"*, *"no assay reporters as pathway members"*, *"no contextual gene-list neighbours as
participants"*. **This is what Priority 1 measures, and the gold's forbidden list is its
operational form. It is correct and is not in question.**

**`PRODUCT_CONTRACT` § 7 — Coverage policy.** *"`requested_core_coverage_below_minimum` triggers
targeted retrieval before classification. It is not, by itself, a refusal… The coverage threshold
blocks release-ready status, not PWML production. **The threshold value itself does not move.**"*

**`PRODUCT_CONTRACT` § 14 — Adjudication.** A benchmark failure must be classified as
`product_contract_violation`, `gold_data_defect`, or `policy_disagreement`; **only the first
justifies code.**

**`CLAUDE.md` merge rule 6.** *"It does not weaken a biological gate to increase PWML production."*

**The tension is not written down anywhere.** § 7 fixes the coverage threshold but says nothing
about what belongs in the coverage **denominator**. That silence is where the contradiction lives.

---

## 4. Why Priority 1 and Priority 4/5 cannot both pass today

Because the two instruments read **the same term list with opposite signs**, and nothing reconciles
them:

* **Cap 2's input is Stage 0's `key_compounds` / `key_proteins`** — a **non-deterministic model
  draw**, not a curated core. `release_status.py:350` already carries `requested_core_source` for
  exactly this provenance distinction, and on these legs its value is not `pathway_context`.
* **The gold's `forbidden_identifiers`** (`bench/goldset.py:387`) is a curated, biologically
  specific exclusion list.
* **Neither is aware of the other.** `strict_quarantine.py:997` computes
  `coverage_ratio = len(matched_terms) / len(terms)` with `terms` straight from the draw. No
  forbidden-term filter is applied to the denominator anywhere.

So a term can enter the denominator *because the model drew it* and be unmatchable *because the gold
forbids exporting it*. Obeying the gold lowers the ratio; disobeying it fails Priority 1.

**This is `product_contract_violation`, not `gold_data_defect`** — reclassified 2026-08-27 after
challenge, and the reclassification matters because it points the remedy at the right place.
`gold_data_defect` would aim a future card at the gold, where the natural "fix" is to drop `MenD`
from `forbidden_identifiers` or soften the exclusion sentence. **That would weaken a correct
biological constraint to move a coverage number, and merge rule 6 forbids it.**

**The gold is the instrument that is right here.** Every exclusion on these cases is specific and
biologically sound: `MenD` a competing menaquinone-branch isochorismate sink; `lactate
dehydrogenase`/`LDH` a **porcine** coupled-assay reporter in an *E. coli* pathway; `NADH`/`NAD+` the
coupled-assay readout species. **Nothing in the gold needs changing.**

---

## 5. Options

### Option A — Separate the measurements *(recommended)*

Forbidden-entity **export** stays scored under Priority 1. Gold-forbidden terms stop being
simultaneously **required as positive coverage matches**. Coverage is computed from
**supported/exportable** terms; an omitted forbidden entity generates **no** coverage penalty.
Extracted-but-withheld entities remain visible in diagnostics.

### Option B — Accept a stated non-zero Priority-1 floor and leave coverage alone

Record that Priority 1 cannot reach 0 until a participant-provenance carrier and an entity evidence
span exist (F-127), authorise T-107 to run against an explicit floor of 6, and leave Priority 4/5
scoring unchanged.

### Option C — Curate the requested-core anchor set

Replace the Stage-0 draw as cap 2's input with a curated per-case core, so the denominator is a
product decision rather than a model draw.

---

## 6. Consequences of each

| | **A — separate the measurements** | **B — accept a Priority-1 floor** | **C — curate the anchors** |
|---|---|---|---|
| **Fixes the contradiction?** | **Yes.** Each row is scored once, by the instrument that owns it. | **No.** The double bind survives untouched; it is only tolerated. | **Yes**, and more thoroughly — the draw stops being an instrument at all. |
| **Biological safety** | **Preserved.** Priority 1 unchanged; forbidden export still fails. No gate weakened. | **Preserved.** Nothing changes. | **Preserved**, but the curation itself becomes a biological judgement needing review per case. |
| **Risk of gaming** | **Real and must be closed in the ruling**: "not required as a coverage match" must never become "safe to omit anything". Confine the exemption to terms on the case's own `forbidden_identifiers`. | None. | Low, but the curated list becomes a place where difficulty can be quietly defined away. |
| **Does Priority 4 become meaningful?** | **Yes** — it measures coverage of what the pipeline was permitted to export. Expect it to move off 0/8; **by how much is not predicted here.** | **No.** 0/8 stands and stays uninterpretable. | Yes, and most meaningfully. |
| **Does Priority 5 become meaningful?** | **Partly.** `PMC12782028/strict`'s `requested_core_coverage_below_minimum:0.222<0.500` is levied *partly* for gold-forbidden terms; recomputing may or may not clear it. **Not predicted.** | No. | Yes. |
| **Priority 1 result** | Still **FAIL at 6** — A does not touch it. | **FAIL at 6, explicitly accepted.** | Still FAIL at 6. |
| **Cost / blast radius** | One scoring seam plus a gold-field read. Moves Priorities 4 and 5 at once — a deliberate baseline move needing an exact documented delta. | **Zero code.** A recorded decision only. | Largest. Per-case curation across 6+ papers, plus a Stage-0 contract change. |
| **Honesty of the record** | High — the instrument stops contradicting itself. | **Honest but incomplete**: it accepts a limitation without removing the contradiction, so Priority 4/5 stay unreadable on these papers. | Highest. |

**A and B are not mutually exclusive.** A fixes Priority 4/5; Priority 1 still cannot reach 0 for
F-127's separate reason, so **B is still required for T-107 to run.**

---

## 7. Recommendation

**Adopt A, and separately grant B.**

A is recommended because it changes the **instrument that is demonstrably wrong** and leaves the
**biological gate that is demonstrably right** untouched. It does not weaken Priority 1, does not
edit a single forbidden identifier, and does not lower the coverage threshold — § 7's *"the
threshold value itself does not move"* is respected exactly. It corrects **what goes into the
denominator**, which § 7 never spoke to.

C is the better long-term answer and should be recorded as the eventual direction, but it is a
larger production change than this wave can carry safely, and A is a strict prerequisite for reading
whether C helped.

**Three guard rails the ruling should state explicitly, because A is otherwise gameable:**

1. The exemption applies **only** to terms appearing on that case's own `forbidden_identifiers` —
   never to terms that are merely hard, absent, or unmatched.
2. **No bare identifiers and no fabricated PWML** are introduced by this change, in any form.
3. Extracted-but-withheld entities **remain visible in diagnostics**. Removing them from the
   coverage denominator must not remove them from the record.

---

## 8. Exact changes that would follow — **not applied**

### Code

| File | Change |
|---|---|
| `src/t2pw/pipeline/strict_quarantine.py:997` | `coverage_ratio = len(matched_terms) / len(terms)` — exclude case-forbidden terms from `terms` before the ratio. The forbidden list must reach this seam; today it does not. |
| `src/t2pw/pipeline/strict_quarantine.py:1017-1034` | Record the excluded terms and the pre-exclusion denominator alongside `coverage_ratio` / `requested_core_source`, so the change is auditable and reversible. |
| `src/t2pw/bench/acceptance.py` | Priority 4/5 read the adjusted denominator. Priority 1 **unchanged**. |
| **Seam question to settle first** | The forbidden list is gold (`bench/goldset.py:387`) and the ratio is production (`strict_quarantine.py`). Threading gold into production would embed **gold-set-only policy into the general pipeline**, which `PRODUCT_CONTRACT` § 12 forbids. **The exclusion therefore belongs in the scorer, not in `strict_quarantine`** — production keeps reporting the raw draw, and acceptance computes the corrected coverage. This is a real design constraint and it narrows the change to `bench/`. |

### Gold

**None. No forbidden identifier is removed, softened, or reworded.** Any proposal to do so is a
merge-rule-6 rejection.

### Documentation

| File | Change |
|---|---|
| `PRODUCT_CONTRACT.md` § 7 | Add the denominator rule: coverage is computed over terms the case permits exporting. State that the threshold value still does not move. |
| `DECISIONS.md` | New locked decision recording the ruling, the three guard rails, and that the gold was found correct. |
| `LEDGER.md` | F-132 closed by ruling; the T-107 readiness table's rows 1, 4 and 5 updated. |
| `TEST_MATRIX.md` | The Priority-4/5 baseline move, with an exact documented delta and an A/B against the pre-change SHA — gold-adjacent scoring changes have already broken a module SMOKE does not run, this wave. |

---

## 9. Would T-107 become informative after the ruling?

**With A alone — partly. With A and B together — yes.**

Against § 8's eight gate conditions, condition 1 (*"priority 1 has a safe correction or an
explicitly accepted measurement limitation"*) is the one that no engineering in this sprint can
clear. **A does not clear it; only B does**, because B *is* the explicit acceptance.

Post-ruling expectation, stated so it can be checked rather than claimed:

| Priority | After A + B |
|---|---|
| **1** — zero false real identifiers | **FAIL at 6**, against an explicitly accepted floor. Informative: the six are classified and their mechanism is known (F-127). |
| **2** — no unsupported retained reactions | `NOT EVALUATED` on 11/20 legs. **Unchanged by this ruling** — D-067 precondition 3 needs independent biological review, which is a separate blocker. |
| **3** — referential integrity | **PASS**, as today. |
| **4** — requested-core coverage | **Becomes readable for the first time.** It would measure coverage of exportable terms. The value is **not predicted here**; predicting it would be the kind of claim this sprint has repeatedly had to withdraw. |
| **5** — strict PWML export | **Becomes readable.** Whether `PMC12782028/strict` clears its coverage block depends on the recomputed ratio and is **not predicted**. |

**Honest limit on this answer:** even after the ruling, Priority 2 remains `NOT EVALUATED` on more
than half the legs for an unrelated reason. T-107 would become a **partial but honest** measurement
— which is a large improvement on today, where two of its five priorities measure the instrument
rather than the pipeline.

---

## What is requested

1. **Rule on A** — adopt, reject, or amend, with the three guard rails.
2. **Rule on B** — accept or decline the Priority-1 floor of 6 for T-107 purposes.
3. If both are granted, T-107 may be scheduled. **Until then it stays NO-GO, and that remains the
   correct outcome rather than a delay.**

**C-091 remains explicitly not for merge unless the ruling authorises its direction.**

---

# ADDENDUM — a second, independent ruling is now requested: F-135, the placeholder-species question

Added 2026-08-27. **This is a separate question from the Priority-1 contradiction above and can be
ruled on independently.** It arrived from engineering rather than from the benchmark.

## The question

**What should happen to an entity that is preserved as biology but cannot satisfy the export format
without a fabricated field?**

## How it arose

C-094 stops PathBank's `Unknown` sentinel record (protein 9659, species *Arabidopsis thaliana*)
lending its own organism to the generated wrapper built around it. Measured across all 92 committed
legs: 31 Unknown-backed wrappers, **6** with a correct species surviving underneath, **25** with
nothing.

Investigating what happens to those 25 revealed that **three independent gates require species on a
protein complex**: the strict Stage-3 post-normalization gate; `ir.validate_required_pwml_contract`,
whose docstring calls it the **PWML-ready contract** and which raises `protein_complex_missing_species`
as an error; and `pwml/writer.py`, whose species-resolution chain ends at `return default_species_id`.

So those 25 entities were **never legitimately exportable**. The fabricated *Arabidopsis* was the only
thing carrying them through the format contract.

**A fourth card to punch through the second gate was refused**, because the writer would then silently
stamp `default_species_id` — replacing a false species at mapping time with a false one at export
time, inside the exporter, which `PRODUCT_CONTRACT` § 5 and merge rule 8 forbid.

## Why this is yours and not ours

`PRODUCT_CONTRACT` § 13 makes `placeholder_backed_proteins` a **standing disagreement, not a defect**,
and says **no agent may "fix" it — escalate.** Three cards into a gate-exemption chain is that fixing.
We stopped.

## The three coherent answers

| | Outcome | Cost |
|---|---|---|
| **A — quarantine the entity, export the rest** | the pathway ships smaller and honest; `PRODUCT_CONTRACT` § 1's *"a smaller supported pathway is preferable to a larger contaminated one"* | the reaction loses its actor, and the wrapper exists **because** the PathWhiz importer refuses a bare protein as an enzyme, so the loss may cascade into the reaction |
| **B — block the leg** | no contaminated output; this is today's behaviour with C-094 applied | a valid pathway core is suppressed because one peripheral actor is unresolved — which § 1's own unacceptable-terminal-blockers list names |
| **C — keep the placeholder species** | today's behaviour without C-094 | a fabricated organism in a **released** payload. Measured: `runs_verify/2026-08-04_1754/papers/PMC12856317/strict` shipped `pathway.pwml` — a `release_ready` artifact — carrying *Arabidopsis thaliana* on a human ALAS2 wrapper |

## Recommendation

**Adopt A, and merge C-094 now regardless of which is chosen.**

C-094 is correct under any of the three: § 1 forbids inventing an identity *"merely to guarantee a
PWML file"*, and on the measured corpus C-094 costs **zero** PWML — the single PWML-producing leg
carrying an Unknown-backed wrapper is one of the 6, and it is *improved*, recovering *Homo sapiens*
in place of *Arabidopsis*.

A is recommended because it is the only option that keeps both halves of § 1: no invented biology,
and no valid core suppressed. Its cascade risk is real and would need its own measured card — that is
the work the ruling would authorise, not something to assume away.

**C is not recommended and should be rejected explicitly if you disagree**, because it is the only
option that puts a fabricated organism in a released file.

## What follows from each ruling

* **A** — charter the quarantine-and-export-the-rest path, with the reaction-actor cascade measured
  first. `card/C-098a-cap` (`8cfa33e`) and `card/C-098b-gate` (`b589821`) are held and would be
  reassessed against it, not merged as they stand.
* **B** — merge C-094 alone and close F-135 as *working as intended*. Both C-098 arms are discarded.
* **C** — C-094 does not merge, F-134 is downgraded to a standing disagreement, and the released
  *Arabidopsis* payload stands.

**Nothing here is applied. No card is merged pending this.**

---

## ADDENDUM 2 — the F-135 question is **O-1**, and it is TRAP-3 protected

Added 2026-08-27, after REV-094. **This supersedes Addendum 1's framing of the question, not its
options.**

Addendum 1 asked what should happen to an entity that "cannot satisfy the export format without a
fabricated field". **That phrasing already answers the question, and I should not have used it.**

`tests/test_protein_export_policy.py`, section header *"strict gates: accept the right shape, reject
the poser"*, pins this today and it passes at `14121d5`:

```python
assert normalization_report["gate"]["ok"] is True
assert validate_post_normalization(normalized, normalization_report["gate"])["ok"] is True
assert validate_required_pwml_contract(normalized, strict_db=False)["ok"] is True
```

**The pinned product position is that a correctly-formed `Unknown`-backed complex passes all three
gates, including the PWML-ready contract.** Under that design the sentinel's species is not a
fabrication — it is part of a coherent *"this row is the PathBank `Unknown` record"* marker, and the
complex is exportable **by design**.

Which is `DECISIONS.md` **O-1**, verbatim:

> `placeholder_backed_proteins` (21 in the pinned run): **gold-set error class, or legitimate biology
> preservation?** · Blocks: **any branch that touches protein export policy** · *"It is a genuine
> disagreement between two intentional designs, not a defect. **TRAP-3 forbids agents from resolving
> it.**"*

### What is actually being asked

**Rule O-1.** Everything else follows mechanically.

* **O-1 = legitimate biology preservation** → C-094 is **wrong** as built. F-134 narrows to the
  *internal contradiction* only (a row asserting `Arabidopsis` while carrying `species_name:
  "Escherichia coli"`, `taxonomy_id: "562"` and an *E. coli* `species_ref` at confidence 1.0), and the
  repair is to make the row **self-consistent**, not to strip its species. C-098a and C-098b are
  discarded.
* **O-1 = gold-set error class** → C-094 is **right**, the four baseline moves are authorised, and the
  follow-on question from Addendum 1 (quarantine the entity, block the leg, or keep the placeholder)
  becomes live.

### What must not be lost either way

Two facts stand under **both** readings and neither is in dispute:

1. **A released payload carried a false organism.** `runs_verify/2026-08-04_1754/papers/PMC12856317/strict`
   shipped `pathway.pwml` — a `release_ready` artifact — with *Arabidopsis thaliana* on a **human**
   ALAS2 wrapper whose correct *Homo sapiens* species was present on the same row.
2. **`writer.py:1137-1165` ends its species chain at `return default_species_id`.** Any species-less
   complex reaching the writer is emitted under the **pathway's** organism. That is a laundering seam
   independent of O-1 and it should be chartered whatever is ruled.

### The four baseline moves awaiting authorization

All measured base-green / tip-red (`ORCH-709/01`, `/02`; `REV-094`):

| File :: test | Note |
|---|---|
| `test_pathbank_unknown_fallback.py::test_unknown_fallback_passes_stage3_and_serializes_exact_pathbank_sentinel` | **also silently drops PWML serialization coverage** — it short-circuits before the `build_pwml_ir` and exact-sentinel assertions, and nothing replaces them |
| `test_protein_export_policy.py::test_strict_gates_accept_a_correctly_formed_unknown_backed_complex` | **this is the O-1 statement itself** |
| `test_protein_export_policy.py::test_the_sentinel_component_is_not_treated_as_an_unused_protein` | |
| `test_protein_export_policy.py::test_later_normalization_keeps_the_wrapper_and_drops_the_original` | |

**None has been edited. Nothing is merged. `card/C-094-f134` (`53eaf24`), `card/C-098a-cap`
(`8cfa33e`) and `card/C-098b-gate` (`b589821`) are all held pending this ruling.**

### One process consequence worth ruling on separately

`tests/test_protein_export_policy.py` is in **neither SMOKE nor Chunk D**. Nothing this sprint runs
would ever have caught its move; it took a reviewer selecting it by hand. It is a `BASELINE.md`
group-05 file — the same group as the baseline the C-094 charter singled out as untouchable. **A
pinned product statement that no gate exercises is a statement that can be inverted silently**, and
this is the second time a gate-invisible file has hidden a real move this sprint.
