# T-107 — TRIAGE AND CLASSIFICATION

**Official run `runs_verify/2026-08-28_1816`. NOT RERUN. No leg of it was repeated.**
Classification wave `ORCH-716`, at integration tip `36f773c`.

Every failure below is classified as `product_contract_violation`, `gold_data_defect` or
`policy_disagreement`, citing the gold's own `relevance_note` / `export_rationale`. **Only the
first justifies code**, and the last section says plainly why the one clean contract violation
in this run is nevertheless **not** chartered.

---

## 0. The finding that reframes the whole run

The handoff asked whether the four degraded strict legs were draw variance, a C-099/C-100
regression, or a newly-exposed defect. **A fourth answer was not on the list and it is the right
one for two of them: they had already failed before C-099 existed.**

| Leg | T-105 `2026-08-22` | **T-106 artifacts `2026-08-24`** | T-107 `2026-08-28` |
|---|---|---|---|
| `PMC12452463/strict` | PASS | **FAIL (contract, 7 errors)** | FAIL (contract, 3 errors) |
| `PMC13231680/strict` | PASS | **FAIL (no_reactions)** | FAIL (no_reactions) |
| `PMC12096016/strict` | PASS | PASS | FAIL (timeout) |
| `PMC12180156/strict` | PASS | PASS | **FAIL (contract)** |

`C-099` merged **2026-08-27** (`9e4a28a`); `C-100` merged **2026-08-28** (`8e5d549`). The
T-106 artifacts predate both by three days.

* **`PMC12452463/strict` and `PMC13231680/strict` are not C-099 regressions.** They reproduce
  on a tree where C-099 does not exist. `PMC12452463` in fact **improved** at T-107, 7
  contract errors down to 3.
* **`PMC12180156/strict` is the only leg that genuinely turns at T-107**, and section 3 shows
  its mechanism is not species preservation either.
* `PMC12096016/strict` degraded to a **timeout**, which is an operational outcome, not a
  content one, and is classified separately in section 5.

The hypothesis "C-099 species preservation changed the final row" is therefore **falsified for
two of the three legs it was raised about, and unsupported on the third.**

---

## 1. `PMC13231680/strict` — the empty pathway is CORRECT, and T-105 was the false positive

**Classification: NOT A DEFECT. The pipeline behaved as the contract requires.**
The residual scoring question is a **`policy_disagreement`** with the acceptance instrument.

Gold, verbatim:

> `relevance_note`: *"**NEGATIVE CONTROL.** An NDM-1 inhibitor discovery study retrieved because
> its discussion mentions LpxC once. Full-text counts: 'lipid A' 1, 'LpxC' 9, and UDP 0, GlcNAc
> 0, Kdo 0, acyl 0, deacetyl 0. **Not one lipid A substrate, product, intermediate or reaction
> appears anywhere.**"*
>
> `export_rationale`: *"Nothing lipid-A-related is exportable at any level of partiality. **The
> correct pipeline outcome is an empty pathway plus a rejection reason.**"*
>
> `notes`: *"This case exists to test whether the pipeline **declines to produce output rather
> than inventing it**."*

T-107 produced exactly that: `reactions=0`, `failure_kind=no_reactions`, and a stated reason.
`mechanistic_relevance=context_only`, `min_connected_reactions: 0`,
`unknown_backed_proteins_acceptable: false`.

**T-105's PASS is the outcome that was wrong, and the sprint had already said so.** T-105
exported `NDM-1`, `LpxC`, `phthalylsulfacetamide`, `meropenem`, **`Zn2+`**, `sulfacetamide`,
one reaction and four interactions — containing **zero lipid-A content** and at least two of
this case's own `forbidden_identifiers`:

* **`Zn 2+` / `Zn2+`**, kind `cofactor_as_protein` — *"A metal cofactor of NDM-1 and LpxC;
  neither a protein nor a lipid A metabolite."* Shipped as a compound.
* **`PSA`**, kind `heading_or_prose` — *"Here phthalylsulfacetamide; in most biomedical text PSA
  is prostate-specific antigen... Resolving it to a protein identifier is a failure."* Used as
  an entity token in the reaction and three interactions.

That T-105 leg is **already registered** as **F-100**, HIGH, `product_contract_violation`:
*"the declared negative control produced reactions and shipped `release_ready`"*. F-100 also
records that **at T-104 both legs produced `FAIL (no_reactions)`, which F-097 defended as
correct.**

**So the trajectory is T-104 correct -> T-105 defective (F-100) -> T-107 correct again.** The
message about the embeddings endpoint is boilerplate, as the handoff said: RAG ran normally,
**1,294 rejections across 15 legs** at T-107 against 1,837 at T-105.

> **Answer to the charter's question:** PMC13231680's empty strict pathway **is correct**, and
> **T-105's result for that paper was a false positive.** No code may be written to recreate it.

### The residual: the harness calls the correct outcome a leg failure

`expected_export: partial_only`, and PRODUCT_CONTRACT section 12 says such papers are
**excluded from the strict-PWML success denominator**. Reporting the contract-correct outcome
as `RESULT: FAIL` is an **acceptance-instrument** matter, not a production one.
**`policy_disagreement`** — a decision packet, not a card. **No production code.**

---

## 2. Priority 2 — the single failure, and what it actually is

**Classification: `product_contract_violation`, and it is the SAME defect as F-100, on the leg
F-100 did not close.**

Scorer output, `evidence/t107_score_priorities.log`:

```
rank 2: zero unsupported retained reactions
    observed = 1     counted = 1     papers = ["PMC13231680"]
```

The strict leg retained **zero** reactions, so the counted row is the **research** leg's only
reaction:

```
phthalylsulfacetamide decomposition to sulfacetamide   enzyme: NDM-1
```

The same reaction at **T-105 carried NO enzyme**. T-107 attached **NDM-1** to it. NDM-1 is a
metallo-beta-lactamase; the paper's own claim is that it hydrolyses **meropenem**. Attributing
the decomposition of phthalylsulfacetamide to sulfacetamide to NDM-1 catalysis is an **enzyme
relationship the paper does not state** — PRODUCT_CONTRACT section 2 requires *"correct enzyme,
modifier, transporter and cargo relationships"*, and the gold records **`supported_reactions:
null`** for this case, i.e. nothing here is a supported reaction at all.

**This one row is the entire reason T-107 is NOT ACCEPTED.** It is a genuine contract violation,
it is on the declared negative control, and it belongs to **F-100's open class** — the same
paper, the same failure to decline. Priority 2's verdict stands: **`FAIL`, on a measured,
eligible leg.** D-075's `CONDITIONALLY SATISFIED` is unavailable and is not claimed.

Its remedy is F-100's remedy, which is **not** a new narrow card: it is the standing
"decline rather than invent on a negative control" work already owned by **C-074 arm B /
F-103**, whose production seam (`request_was_never_stated`, `release_status.py:1146-1174`) is
merged and shipped. **What T-107 shows is that arm B demotes the STRICT release status but does
not stop the RESEARCH leg retaining an unsupported reaction** — research mode is fail-open by
design. That gap is registered below as **F-146** rather than patched blind in a triage wave.

---

## 3. `PMC12452463/strict` and `PMC12180156/strict` — one shared seam, and it is NOT what the failure message says

**The two legs DO share a root cause.** As the charter required, that means one general
finding, not two name-specific ones. Neither Fur nor ALAS2 nor enterobactin synthase appears in
it as anything but an instance.

### 3.1 The measurement

In **every** leg of T-107 — passing and failing alike — `final_stage3_gate_report.json` reports:

```
ok: true    errors: []    phase: final_pre_export
```

The pass/fail difference comes **entirely** from `post_normalization_contract_report`, which is
stamped **`phase: audit_round`**. `streamlit_app.py:4055-4060` documents that stamp in terms:

> *"This report is still **not a verdict about what shipped** -- the remap below moves the
> payload again -- which is what the phase stamp says."*

`batch/driver.py::_blocking_reports` scans **every** `*_contract_report` and fails the run on
any that carries errors. **It does not read the phase stamp.** The failure message then
attributes the errors to `final_pre_export_stage3_gates` — the one report that said `ok: true`.
That attribution is factually wrong and it is what sent this triage to the wrong seam first.

### 3.2 Two readings, and the probe that separates them

Either the audit-round payload no longer exists (**stale**), or the gate reads a field the
final payload does not populate and the final gate is blind (**key mismatch**). The second
would be far worse. `evidence/orch716_stale_verdict_probe.py` settles it by running the **real
production predicates** — `protein_external_identity`, `protein_species_context`, the same ones
`process_normalizer.py:4627-4633` calls — row by row against the **shipped** `final_mapped.json`:

| Leg | Run verdict | Shipped payload under the production predicates |
|---|---|---|
| `PMC12452463/strict` | **FAIL** | **PASSES**, 0 objections |
| `PMC12180156/strict` | **FAIL** | **PASSES**, 0 objections |
| `PMC12856317/strict` (control) | PASS | PASSES, 0 objections |
| `PMC12782028/strict` (control) | PASS | PASSES, 0 objections |

**The stale reading is confirmed. There is no key mismatch.** Specifically:

* **`Fur`** was removed before export by `pre_export_strict_quarantine`,
  `reason: degree_zero_after_quarantine`, `had_external_identity: false` — and F-141 classified
  both Fur rows as *"candidate does not describe the shipped identifier — **withholding
  correct**"*. The run was failed on an entity that **does not ship** and that **correctly has
  no identifier**. PRODUCT_CONTRACT section 15: *"A row that correctly withholds a
  species-specific identifier is not an error anywhere else: do not penalise the pipeline twice
  for obeying the contract."*
* **`ALAS2`** carries a **verified** `uniprot: P22557`, `pathbank_protein_id: 17`,
  `verification_status: verified` in the shipped payload. The identifier was **never lost** — it
  was **not yet resolved** when the audit-round snapshot was taken, and the remap that follows
  resolved it. `uniprot_id` is `None` on that row while `uniprot` is set; a reader checking only
  `uniprot_id` would wrongly conclude the identifier is missing. That is F-144's trap and the
  probe prints the populated key for every row precisely so it cannot recur.
* **`enterobactin synthase`** was replaced by the PathBank `Unknown` sentinel
  (`pathbank_protein_id: 9659`, `cross_species_placeholder: true`,
  `placeholder_target_organisms: ["Escherichia coli"]`) which carries species *Arabidopsis
  thaliana*. Its `provenance_lineage` states the uncertainty explicitly. **This is
  PRODUCT_CONTRACT section 13's standing disagreement on `placeholder_backed_proteins`, not a
  defect — "No agent may 'fix' it."** It is not fixed here and no card touches it.

### 3.3 Classification

**`product_contract_violation`**, registered as **F-147**. PRODUCT_CONTRACT section 1 lists
among the outcomes that may **never** end a run without a PWML:

> *"a **missing or stale gate report**"* · *"an **irrelevant degree-zero entity**"*

Both describe this seam exactly, and the Fur instance is both at once.

**The violation is real. It is nevertheless NOT chartered. Section 6 explains why, and that
refusal is the most important judgement in this document.**

---

## 4. `PMC12180156/strict` — what the leg would have exported

**Classification of the leg's OUTCOME: correct. `gold_data_defect`: none. The FAIL is right and
the stated reason is wrong.**

The gold makes `PMC12180156` a **second negative control**:

> `mechanistic_relevance: context_only` · `expected_export: partial_only` ·
> `min_connected_reactions: 0` · `unknown_backed_proteins_acceptable: false`
>
> `relevance_note`: *"Names the requested pathway five times without describing one step of it.
> Ferrochelatase and **ALAS2 are named with NO reaction stated for either**. **Zero
> heme-biosynthesis reactions have both sides named anywhere in the file.**"*
>
> `export_rationale`: *"With zero heme-biosynthesis reactions recoverable, **nothing about heme
> biosynthesis is exportable**."*

T-107's strict leg retained exactly one reaction:

```
ferrochelatase reaction    inputs: [protoporphyrin IX, iron]    outputs: [heme]
                           enzyme: ferrochelatase complex  (provenance: inferred)
```

**`protoporphyrin IX` is on this case's own `forbidden_identifiers`**, as an alias of the
`5-aminolevulinic acid` entry, kind `placeholder_product`:

> *"**HALLUCINATION TEST: zero occurrences in the entire 67,304-character file**, body and
> references alike. The paper names ALAS2 without ever naming its substrates or its product, so
> any of these is fabrication."*

So the one reaction this leg would have exported is **built on a term the gold certifies does
not occur in the paper**, with an enzyme relationship marked `inferred` whose cited evidence
(*"SFXN4 regulates heme biosynthesis by modulating ferrochelatase levels"*) does not state that
reaction. T-105's PASS on this leg exported **two** heme reactions plus an `Unknown`-backed
protein on a case where `unknown_backed_proteins_acceptable: false`.

**Both runs fabricate. T-107 at least did not ship it.** The leg failing is the right outcome;
it simply failed for the stale-report reason of section 3 rather than for the fabrication.

The gold also predicted the mechanism: *"**Cross-paper leakage from PMC12856317, which DOES
support the ALAS2 reaction, is a concrete risk in a shared run.**"* Both papers ran in this
batch, and `PMC12856317/strict` ships `ALAS2 / P22557` legitimately. That is a prediction the
gold made and the run bears out — it is **not** evidence that the gold is defective.

> **Answer to the charter's question:** ALAS2's identifier was **not** lost and **not**
> withheld — it is present and verified in the shipped payload. Species preservation did not
> change its mapping eligibility. The gate expectation does not conflict with export policy;
> the gate was simply read from a superseded snapshot. **Fur and ALAS2 share exactly one
> seam — F-147 — and it is a reporting seam, not a biological one.**

---

## 5. The three timeouts — classified separately, as required

**`PMC12444477/strict`, `PMC12444477/research`, `PMC12096016/strict`.**

**Classification: `product_contract_violation`, registered as F-148 — and one previously open
defect is now measurably CLOSED.**

### Closed: F-092 defect 3

F-092 defect 3 was *"the inner deadline path records no terminal reason at all"* —
`product_contract_violation`, authorized for code after T-106. At T-105 the inner row carried
**no** `termination_reason`, `operational_failure` or `budget` key. At T-107:

```
PMC12444477/strict   stage=input   termination_reason=operation_timeout   operational_failure=true
                     budget_unrecorded: "not recorded on the in-process timeout path: this seam is
                     handed the timeout detail only, never the leg budget ... so elapsed and
                     remaining cannot be stated truthfully here and are not guessed."
```

That is the contract's `operation_timeout` used correctly for the first time in this sprint,
with the missing budget **declared rather than fabricated**. **F-092 defect 3 is closed by
measurement.** Defects 1 and 2 remain as previously ruled (`policy_disagreement`, no code).

### Open: nothing is preserved

All three rows carry `files: []` and `counts: {}`. The parent rows say *"produced nothing"*.
PRODUCT_CONTRACT section 1 lists as an unacceptable terminal blocker:

> *"**a timeout without usable checkpoints or recovery information**"*

and section 9 requires, on timeout or budget exhaustion, preservation of: *last completed stage ·
current structured payload · all retrieved evidence · attempt numbers, prompts/models and
response hashes · elapsed and remaining budget · the next recovery step that was skipped · the
exact stop reason.*

T-107 preserves **two** of those seven — the exact stop reason, and (on the two outer rows) the
budget. It preserves **no payload, no evidence, no attempt record, and no skipped-step record**,
and `stage` is `unknown` on two of three legs.

This is why **`LpxH` is unverified on T-107** and cannot be reported otherwise: both
`PMC12444477` legs timed out with no payload to inspect. It remains verified at the merged tip
on the pinned run `runs/2026-08-02_2130`. **T-107 does not confirm it.**

### One recording gap worth a line

All three rows carry `leg_timeout_overridden: true`, `leg_timeout_seconds: 1800.0` against a
default of `3600.0`, with **`leg_timeout_override_reason: ""` and
`leg_timeout_override_source: ""`**. Section 9 requires per-leg overrides to be *"explicit and
recorded in the run manifest"*. The fact and the value are recorded; the justification and
provenance are empty strings. The override **shortens** rather than extends, so the *"no silent
extension of difficult benchmark legs"* clause is not violated — but half the requirement is
unmet. Folded into F-148.

---

## 6. What is chartered, and the one violation that deliberately is not

### Chartered and dispatched

* **C-104** — D-083's two carried follow-ons: prove C-102's deep copy (its revert mutation R5
  is green, so a shipped fix has no proof), and make the split-gate driver abort on
  `errors > 0`. Test and evidence tooling only; **changes no production line**. Contractually
  clear under D-083, which is what authorizes it.

### Registered, NOT chartered — and this is the load-bearing decision

**F-147 (section 3) is a genuine `product_contract_violation` and I am not chartering a fix for
it in this wave.** Merge rule 6 is the reason, and it is not close.

The two legs F-147 fails are the only two legs it fails. **If the driver stopped honouring
superseded `audit_round` reports, both would PASS — and both would export content the gold
forbids:**

| Leg | What it would ship |
|---|---|
| `PMC12452463/strict` | `enterobactin synthase complex` — a **`forbidden_identifier`**, *"A complex name explicitly denoting three proteins"*; `RyhB inhibits EntC` / `EntF` — **`forbidden_identifier`**, *"A small RNA, not a protein and never an enzyme"*; `Enterobactin secretion` — gold notes: *"Export of enterobactin from the cytoplasm is **never described at all**, so **no efflux step may be emitted**"*; and an `Unknown`-backed protein where `unknown_backed_proteins_acceptable: false` |
| `PMC12180156/strict` | the `ferrochelatase reaction` built on **`protoporphyrin IX`**, gold's own *"HALLUCINATION TEST: zero occurrences in the entire 67,304-character file"* |

And PRODUCT_CONTRACT section 13 rules `PMC12452463` directly:

> *"Correct outcome after the index fix is `review_required` with
> `strict_acceptance_eligible=false`. **Never strict success.**"*

A fix to F-147 alone would convert two contract-correct no-export outcomes into two contaminated
exports. That is *"weakening a biological gate to increase PWML production"* by another route,
and *"repairing downstream serialization when the earliest unsafe seam is upstream"*.

**The earliest unsafe seam is Stage-1 extraction**, on both papers: it created
`enterobactin synthase` / `enterobactin synthase complex` / `RyhB` / an efflux step on one, and
a `protoporphyrin IX -> heme` reaction on the other. **That is the F-100/F-101/F-103 class —
"decline rather than invent" — not a driver-reporting card.** F-147 may only be fixed **after**
that content is stopped upstream, and its fix must be paired with the gates that would then
correctly block these legs on their real problems.

Recording it now, with the probe that proves it, is the deliverable. Fixing it now would be the
mistake.

### Prepared, requiring authority I do not hold

* **Section 1's residual** — the harness scores a contract-correct negative-control outcome as
  `RESULT: FAIL`. `policy_disagreement`. Decision packet, product owner. **No gold edit is
  proposed: the gold is right on every case examined in this wave.**

### Not touched, by rule

* `placeholder_backed_proteins` / `Unknown`-backed export — PRODUCT_CONTRACT section 13 standing
  disagreement. Escalate, never fix.
* The `LpxH` tolerance — not widened. `PMC12444477` goes 9 -> 8, never 9 -> 7.

---

## 7. New findings registered by this wave

| Id | Class | Summary |
|---|---|---|
| **F-146** | `product_contract_violation` | C-074 arm B demotes the **strict** release status on an unstated request but does not stop the **research** leg retaining an unsupported reaction. This is Priority 2's only failure at T-107 and F-100's open remainder. |
| **F-147** | `product_contract_violation` | `driver._blocking_reports` fails a run on a `phase=audit_round` contract report the app documents as *"not a verdict about what shipped"*, and reports the failure under `final_pre_export_stage3_gates`, which said `ok: true, errors: []`. **Registered, deliberately not chartered — see section 6.** |
| **F-148** | `product_contract_violation` | A timed-out leg preserves the stop reason and the budget but **no payload, evidence, attempt record or skipped-step record** (`files: []`), against PRODUCT_CONTRACT section 1 and section 9. Includes the empty `leg_timeout_override_reason` / `_source`. **F-092 defect 3 is closed by the same measurement.** |
| **F-149** | *audit result, no defect* | `test_c074_strict_core_floor.py` and `test_c072_incomplete_core_demotion.py` both pin their caps **non-vacuously** under adversarial mutation by a non-author. **F-142's no-coverage-gap conclusion stands. No correction chartered.** |

## 8. Corrections to the record this triage forced

* **T107-RESULT.md section 5** frames all four strict degradations against T-105 and offers a
  C-099/C-100 regression as live hypothesis 2. **Two of the four already failed in the T-106
  artifacts, three days before C-099 merged.** The hypothesis is falsified for those two. The
  T-105 comparison alone is not a sufficient baseline and the T-106 artifacts must be quoted
  beside it.
* **"Of the six strict legs that passed in T-105, four degraded"** is true as arithmetic and
  misleading as a claim about quality: on the two negative controls, the **T-105 pass was the
  defect** (F-100 for `PMC13231680`; fabricated heme chemistry for `PMC12180156`), and
  `PMC12452463/strict` must **never** be a strict success under PRODUCT_CONTRACT section 13.
  **Three of those four "degradations" are movements toward the contract, not away from it.**
* **F-145 population**: **92 terms / 47 legs / 7 papers.** The bundle's 62/32/6 is an
  undercount and is not quoted anywhere in this document.
