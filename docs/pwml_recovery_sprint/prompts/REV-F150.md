# REV-F150 — independent verification before any gold byte is changed

- **Subject** F-150, `gold_data_defect`. The edit is **written and NOT applied**.
- **You are the independent reviewer. You are not the author of the edit and you do not apply it.**
- **Criteria fixed before the work.** Additions after the fact must be labelled as additions.

**The product owner's authority is CONDITIONAL on your verification.** If you cannot verify a
condition, **the corresponding half is not applied.** *"The existence of a correct-looking edit is
not authority to make it"* — and a gold change silently made is the one change this sprint could not
audit its way out of.

---

## 0. What is proposed, and what is NOT

**Proposed — half 1 only.** `src/t2pw/bench/gold/pinned_v1.json`, case `PMC12180156`,
`forbidden_identifiers[0].aliases`: add `"delta-aminolevulinic acid"` and
`"δ-aminolevulinic acid"`. **Add nothing else and move no threshold.**

**NOT proposed — half 2.** Setting `supported_reactions_complete` on any case. That changes what
Priority 2 *measures* on every future run. **It is a product decision about the benchmark's meaning,
not a data correction, and it remains an open product-owner question. Do not verify it, do not
recommend it, and do not let it drift into scope.**

**The halves are separable and MUST be split if they do not both verify.** Apply only the half that
passes.

---

## 1. The measurement, already taken twice — reproduce it a third time yourself

The Lead re-verified this at integration tip (`evidence/orch717_f150_reverify.log`, G11
`ORCH-717/16`):

```
forbidden_match('5-aminolevulinic acid'    ) -> '5-aminolevulinic acid'
forbidden_match('delta-aminolevulinic acid') -> None
forbidden_match('δ-aminolevulinic acid'    ) -> None
forbidden_identifiers[0].aliases : ['ALA', 'porphobilinogen', 'protoporphyrin IX',
                                    'succinyl-CoA', 'coproporphyrinogen III',
                                    'uroporphyrinogen III']
```

**Reproduce it. Reading the number is not verification.**

---

## 2. The eight conditions — every one must hold

| # | Condition | How to verify |
|---|---|---|
| **V1** | **It is a true INTERNAL gold inconsistency** | The gold author **already uses the delta spelling elsewhere in the same case** — `acceptable_enzymes[1].aliases`: *"erythroid delta-aminolevulinic acid synthase"*. Confirm that. Internal inconsistency is what makes this a data defect rather than a policy change. |
| **V2** | **It aligns the encoded field with existing prose** | Read the case's own `relevance_note` / `export_rationale` / `notes` and quote the sentence the edit aligns to. **If no existing prose demands it, it is a new policy and must be refused.** |
| **V3** | **It introduces NO new biological policy** | Adding two spellings of a name already forbidden is not a new judgement about biology. **If the edit would newly forbid anything not already forbidden, that is a policy change — reject.** |
| **V4** | **It does not soften a constraint to improve the benchmark** | This edit makes the benchmark **stricter** (it counts a false accession that currently escapes). **Confirm the direction.** An edit that made a number look better by relaxing gold would be the reject case. |
| **V5** | **The four-step A/B passes** | See § 3. All four steps, in order. |
| **V6** | **Every mover is PREDICTED before it is measured, then enumerated** | Write the prediction down first. The recorded prediction is **Priority 1 rises 5 → 6, and 6 is still `PASS` under D-073 (0–6)**. **A mover you did not predict is a finding, not a footnote** — investigate it before approving. |
| **V7** | **Gold-reading tests OUTSIDE SMOKE pass** | **This is the one that gets skipped.** Gold edits break gold-reading tests **SMOKE never runs**. The 22-file gold-readers selection must be **456 passed / 8 skipped / exit 0** both before and after, or the delta must be explainable term by term. |
| **V8** | **Priority-1 movement is reported as an INSTRUMENT change** | **A Priority-1 number that moves because the gold changed must NEVER be reported as a pipeline regression — or as a pipeline improvement.** Both raw and corrected numbers must be labelled **with the gold SHA they were measured against.** |

---

## 3. The four-step A/B — mandatory, in this order

1. **Capture the gold-readers selection at the PRE-EDIT SHA.** Expected **456 passed / 8 skipped /
   exit 0** — the C-103 baseline. **Any older charter claiming this selection correctly exits 1 is
   stale.**
2. **Apply the edit. Re-run the same selection.** The delta must be **explainable term by term** —
   not "roughly the same", not "only gold tests moved".
3. **Re-score T-107's committed artifacts against pre- and post-edit gold** and record **every leg
   that moves.**
4. **Record the raw number beside the corrected one, both labelled with the gold SHA.**

### Step 3 and the T-107 immutability rule — read this carefully

**Step 3 is a scoring of committed artifacts against two gold versions. It is NOT a rerun, and it is
NOT a re-scoring of the official T-107 result.**

- **You may not re-run any T-107 leg.** Not one.
- **You may not restate, revise or reinterpret T-107's official verdict.** It is **`NOT ACCEPTED`**
  and it stays that way. A run's verdict is a fact about the artifacts it produced.
- What step 3 produces is an **instrument-sensitivity measurement**: *"under the corrected gold,
  these legs would have scored differently."* **Label it exactly that way.** If you cannot do step 3
  without it reading as a re-score, **say so and stop** — I will decide how to obtain it.

---

## 4. Stop conditions

- Any condition V1–V8 you cannot verify → **that half is not applied.** Say which and why.
- An unpredicted mover.
- Any movement in a paper or entity type the edit should not reach.
- Pressure to include half 2 "while we are in the file". **Refuse it.**
- A gold-reader delta you cannot explain term by term.

---

## 5. Process

**The gold correction is committed SEPARATELY from any production or scorer change.** Its own commit,
its own message, its own evidence.

Everything through `bounded_run.py` with the explicit venv interpreter, a real `--timeout`,
`--basetemp` under `C:/t/` with the parent pre-created, `PYTHONPATH=<tree>/src`,
`T2PW_OFFLINE_CURATOR=1`, `PYTHONIOENCODING=utf-8`, `--heavy-lock` and a **shape-validated** G11
path. `FINAL SURVIVING COUNT : 0` and `cleanup : success` on every job. The lock is shared — **a
failed acquire means your job did not run; wait and retry, and never clear a lock you do not hold.**

**Commit your probes and logs, not just the G11 reports.**

**Never commit** the caches, `topics_*.txt`, `streamlit_app.py`, `cache_snapshot/`, the stray
`ValueError`, or anything under `out/`, `outputs/`, `tmp/`.

---

## 6. Verdict

**VERIFIED — APPLY HALF 1** · **VERIFIED IN PART — APPLY ONLY THE HALF NAMED** · **NOT VERIFIED — APPLY NOTHING**

State the verdict per half. **Half 2 is out of scope and its answer is always "not proposed".**

**Keep every failed measurement beside its correction.** This sprint has twice found a real defect
inside an anomaly someone nearly discarded.
