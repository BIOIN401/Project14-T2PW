# Decision packet — ORCH-717

**Three questions that need authority the Lead Orchestrator does not hold.** Each states the
measurement, the options, what the Lead would choose and why, and **what happens if nobody
decides**. None of them is chartered. No production code and no gold file was changed to produce
this document.

Verified offline this wave: `evidence/orch717_f150_verify.py` / `.log`, G11
`ORCH-717/05-f150-verify-r2.json`. The first attempt died on a `cp1252` console encoding error and
is preserved beside it as `orch717_f150_verify.attempt1-cp1252-encodeerror.log`.

---

## Q1. The harness scores a contract-correct empty pathway as `RESULT: FAIL`

**Class: `policy_disagreement`. Not a defect in production. Not a gold defect.**

### The measurement

`PMC13231680/strict` produced `reactions=0`, `failure_kind=no_reactions`, and a stated reason.
That is **exactly what the gold demands**:

> `export_rationale`: *"Nothing lipid-A-related is exportable at any level of partiality. **The
> correct pipeline outcome is an empty pathway plus a rejection reason.**"*
>
> `notes`: *"This case exists to test whether the pipeline **declines to produce output rather than
> inventing it**."*

The pipeline declined. The harness recorded `RESULT: FAIL`.

**T-105's PASS on this leg was the false positive** (F-100): it exported `NDM-1`, `LpxC`,
`Zn2+`, `PSA` and one reaction containing zero lipid-A content, including two of this case's own
`forbidden_identifiers`. The trajectory is **T-104 correct → T-105 defective → T-107 correct
again**, and the instrument scored the defective run higher than either correct one.

### Why this is not simply a bug to fix

PRODUCT_CONTRACT § 12 says `expected_export: partial_only` papers are **excluded from the strict-PWML
success denominator** — it is *"a statement about the source and about the strict benchmark
denominator"*, explicitly **"not a prohibition on generating PWML"**. So the contract already says
this leg does not belong in the strict denominator. What it does **not** say is what a per-leg
line should *read* when the correct outcome is "nothing".

`RESULT: FAIL` is defensible as *"this leg produced no strict PWML"*, which is true. It is
misleading as *"this leg went wrong"*, which is false and is how every reader has taken it —
including two waves of this sprint.

### Options

| | Option | Cost |
|---|---|---|
| **A** | A distinct verdict token for *declined-correctly* — e.g. `DECLINED (expected)` — reported separately from `FAIL`, and excluded from the strict denominator as § 12 already requires | acceptance-instrument change; every historical comparison must be re-read with the new token in mind |
| **B** | Leave `FAIL` and carry the caveat in prose | free, and it has already failed twice — T-107's own SUMMARY and the T-105 comparison both misread it |
| **C** | Score it `PASS` | **rejected by the Lead.** It would make "produced nothing" indistinguishable from "produced the right thing", and on a paper where the pipeline *should* produce nothing that is the same blindness F-100 exploited from the other side |

### The Lead's recommendation

**A.** The instrument's job on a negative control is to distinguish *declined* from *failed*, and it
currently cannot. **This is the only one of the three questions where the status quo actively
produces wrong readings rather than merely leaving a number unmoved.**

### If nobody decides

The next run repeats it: a correct decline reads as a failure, someone proposes production code to
"fix" the empty pathway, and the guard against that is a paragraph in a triage document nobody is
required to read. **`PRODUCT_CONTRACT` and `T107-TRIAGE.md` § 1 both already say no code may be
written to recreate T-105's output here. That prohibition is doing the work the instrument should
be doing.**

---

## Q2. F-150 — a gold-data defect, with the edit written and deliberately NOT applied

**Class: `gold_data_defect`. Requires gold-change authority, which is the product owner's.**

### Half 1 — the spelling gap, verified this wave

`PMC12180156/research` ships **`δ-aminolevulinic acid` carrying nine identifiers** — `hmdb`
HMDB0001149, `kegg` C00430, `chebi` 17549, `pubchem` 137, `drugbank` DB00855, plus CAS, BioCyc,
ChemSpider and a PathBank compound id — on a metabolite with **zero occurrences in the source
paper**. Five of the nine are in the scorer's recognized accession set. **It scored nothing on
Priority 1.**

Re-verified independently, against the pinned gold:

```
forbidden_match('5-aminolevulinic acid')      -> '5-aminolevulinic acid'
forbidden_match('delta-aminolevulinic acid')  -> None
forbidden_match('δ-aminolevulinic acid')      -> None
forbidden_identifiers[0].aliases : ['ALA', 'porphobilinogen', 'protoporphyrin IX',
                                    'succinyl-CoA', 'coproporphyrinogen III',
                                    'uroporphyrinogen III']
```

**The `δ` / `delta` spelling is absent from the alias list.** Priority 1 increments `false_real`
only for a **forbidden-matched** row carrying external ids, so **the run's worst false accession was
never counted.**

**The scorer is behaving exactly as ruled.** D-072 as ratified matches *"by name or declared alias
and never by resemblance"*. This is a gold-list gap, not a scorer bug — and that the gold author
already used the delta spelling elsewhere **in the same case** (`acceptable_enzymes[1].aliases`:
*"erythroid delta-aminolevulinic acid synthase"*) makes it an oversight rather than a policy.

### Half 2 — the ceiling that rests on two negative controls, verified this wave

**No gold case sets `supported_reactions_complete: true`. All ten are `false`:**

```
PMC12096016 False   max_retained_reactions=None      PMC12452463 False   None
PMC12180156 False   max_retained_reactions=2         PMC12657337 False   None
PMC12312563 False   None                             PMC12782028 False   None
PMC12421875 False   None                             PMC12856317 False   None
PMC12444477 False   None                             PMC13231680 False   max_retained_reactions=0
                                          TRUE=0  FALSE=10  MISSING=0
```

**Priority 2 is therefore evaluable only through `max_retained_reactions`, which is set on exactly
two cases — and both are negative controls.** One of them (`PMC12180156`, ceiling 2) counts rows
**without checking content**: the leg retained two reactions, both fabricated heme chemistry,
against a ceiling set for two *different* reactions the gold names, neither of which was extracted.
It scored `2 − 2 = 0` at `completeness: 1.0`.

> **Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry T-107
> produced.** Any report quoting it must carry that limit.

### The proposed edit — written, not applied

`src/t2pw/bench/gold/pinned_v1.json`, case `PMC12180156`, `forbidden_identifiers[0].aliases`: add
`"delta-aminolevulinic acid"` and `"δ-aminolevulinic acid"`. **Add nothing else and move no
threshold.**

### The A/B plan, which must run BEFORE and AFTER

**Gold edits break gold-reading tests that SMOKE never runs.** So:

1. Capture the 22-file gold-readers selection at the pre-edit SHA — **expected `456 passed / 8
   skipped / exit 0`** (the C-103 baseline; **any older charter claiming this selection correctly
   exits 1 is stale**).
2. Apply the edit; re-run the same selection; the delta must be **explainable term by term**.
3. Re-score T-107's committed artifacts against pre- and post-edit gold and record **every leg that
   moves**.
4. Record the raw number beside the corrected one, **both labelled with the gold SHA they were
   measured against**.

**Prediction, recorded before any edit: Priority 1 rises 5 → 6, and 6 is still `PASS` under D-073
(0–6).** So this does not change T-107's verdict — but it changes a *measurement*, and:

> **A Priority-1 number that moves because the gold changed must NEVER be reported as a pipeline
> regression.**

### The question for the product owner

**May the two aliases be added?** And separately: **should `supported_reactions_complete` be set on
any case**, which is the only thing that would let Priority 2 measure anything on a paper that is
not a negative control?

### The Lead's position

The alias edit is **narrow, evidenced and self-contained**, and the A/B is written. The Lead would
apply it *given authority* — but **has not**, because the existence of a correct-looking edit is not
authority to make it, and a gold change silently made is the one change this sprint could not
audit its way out of.

**Half 2 is the more consequential half and is NOT proposed as an edit.** Setting
`supported_reactions_complete` anywhere changes what Priority 2 *measures*, on every future run.
That is a product decision about the benchmark's meaning, not a data correction.

---

## Q3. PathBank compound ids in `semantic.py::_external_ids`

**Escalated, unresolved, and it interacts with Q2.**

**Is a `pathbank_compound_id` a real accession for Priority 1?** The `δ-aminolevulinic acid` row
carries one among its nine identifiers. If PathBank compound ids do not count, the row carries
eight; if they do, the Priority-1 arithmetic in Q2's prediction may shift.

The Lead has **not** resolved this and does not treat Q2's prediction as final until it is. Named
here so the two are decided together rather than one silently constraining the other.

---

## What was NOT touched, by rule

* **`placeholder_backed_proteins` / `Unknown`-backed export** — PRODUCT_CONTRACT § 13 standing
  disagreement. **No agent may fix it. Escalate only.** Untouched.
* **F-147** — a real `product_contract_violation`, **registered and deliberately not chartered**.
  Fixing it alone would flip two legs to PASS that would then export gold-forbidden content
  (`enterobactin synthase complex`, `RyhB`, an efflux step the gold says is never described; and
  `protoporphyrin IX`, which occurs **zero times** in a 67,304-character file whose length the gold
  cites exactly). The earliest unsafe seam is **Stage-1 extraction**, not the driver. Merge rule 6.
* **T-107** — not rerun, not rescored, not reinterpreted. **`NOT ACCEPTED` stands.**
* **The gold file** — unmodified. `git diff` shows no `src/t2pw/bench/gold/` path this wave.
