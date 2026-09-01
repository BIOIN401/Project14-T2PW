# REV-F150 — findings, with the failed measurement kept beside its correction

Independent reviewer. Subject: F-150 **half 1 only**. Half 2 was not verified, not
recommended and not touched — see § 5.

Pre-edit tip `4c077f012793260f05d5e0518c6459ca9ad13cc6`.
Gold blob **pre-edit `aee8cb4f1da3d417f36206407867585622b741c0`** →
**post-edit `36f4b7b690b577f72882c3045ca6728d1ec8d9d1`**.
Run tree for every corpus number below: **`runs_verify/2026-08-28_1816`** (T-107's
artifacts), named explicitly in every probe. `runs/` was never read.

---

## 1. A FAILED MEASUREMENT, kept: probe A § 6 returned a false zero

`revf150_probeA_reproduce_and_scan.py` § 6 tried to answer "does the edit reach the
D-072 coverage-denominator seam?" by walking `final_mapped.json` for term lists under a
guessed set of key names (`requested_core_terms`, `terms`, `core_terms`, …) and matching
them with `acceptance.forbidden_coverage_match`.

It reported, in **both** arms:

```
--- 6. D-072 coverage-denominator seam on ...runs_verify\2026-08-28_1816 ---
  coverage-term matches touching 'aminolevulinic': 0
```

**That zero is wrong.** The key-name guess did not reach the coverage block the scorer
actually reads. Probe B, which diffs the real acceptance reports rather than guessing at
the payload, measured the seam moving:

```
/coverage_reconciliation_corpus/legs/3/accepted_denominator   7 -> 6
/coverage_reconciliation_corpus/legs/3/accepted_matched       7 -> 6
/coverage_reconciliation_corpus/legs/3/excluded_count         0 -> 1
/coverage_reconciliation_corpus/legs/3/excluded_terms/0/term  'δ-aminolevulinic acid'
```

Both logs are committed. The zero is retained rather than deleted because a reviewer who
had stopped at probe A would have reported "the edit touches Priority 1 only", which is
false. **The lesson is the sprint's own: a probe that returns the answer you expected is
the one to distrust — assert a known-positive before believing a negative.** Probe A § 6
asserted no known-positive, and its negative was worthless.

Probe A §§ 1–5 are unaffected and were independently corroborated by probe B and by the
scorer itself.

---

## 2. The measured movers, term by term

Same artifacts, two gold instruments. **No leg was re-run.** 5147 leaf paths compared;
5022 identical; every non-identical leaf falls into exactly three groups.

### Mover 1 — Priority 1: raw 5 → 6, accepted 5 → 6. PREDICTED.

The one new row:

```
PMC12180156:research /entities/compounds/2  'δ-aminolevulinic acid'  [false_real_identifier]
  identifiers: chebi 17549, drugbank DB00855, hmdb HMDB0001149, kegg C00430, pubchem 137
  forbidden_kind: placeholder_product
```

`accepted_status` is **`PASS`** in both arms (`target: 6`). D-073 range 0–6 holds.

**Most of the "changed" leaves in this group are a REINDEX, not new findings.** The new
row lands at payload index 1, pushing `LBR` 1→2, `SREBF1` 2→3, `SREBF2` 3→4, `LIPA` 4→5.
That is why `accepted_rows/2/name` reads `A='SREBF1' B='LBR'`. The two `REMOVED` leaves
(`accepted_rows/1/identifiers/uniprot = 'Q14739'`, and its `raw_rows` twin) are LBR's
UniProt accession moving position, not an accession being lost. **Five findings before,
six after; the other four are the same four rows.**

### Mover 2 — the D-072 coverage seam. PREDICTED IN ADVANCE (prediction P6).

`δ-aminolevulinic acid` was in PMC12180156/research's raw requested-core draw
(`matched_in_raw: true`) and is now withheld from the accepted numerator **and** the
accepted denominator:

```
accepted_matched      7 -> 6
accepted_denominator  7 -> 6
```

**The accepted coverage RATE does not move: 7/7 = 1.0 becomes 6/6 = 1.0.** Only the
bookkeeping counts move. This is PRODUCT_CONTRACT § 7's denominator rule behaving exactly
as written — "withheld from the accepted numerator and the accepted denominator alike" —
and the excluded term stays visible by name with the forbidden entry that excused it, as
guard rail 3 requires. Corpus roll-ups follow: `forbidden_terms_excluded` 5→6,
`legs_with_forbidden_terms` 5→6, `papers_with_forbidden_terms` 4→5.

### Mover 3 — the per-leg semantic verdict. A CONSEQUENCE of mover 1, reported for completeness.

```
/papers/6/legs/research/semantic/checks/no_real_id_or_name_conflict/ok   True -> False
/papers/6/legs/research/semantic/ok                                      True -> False
/papers/6/legs/research/semantic/scientific_errors/false_real_identifiers  0 -> 1
```

`papers/6` is PMC12180156. This is the same single finding surfacing at leg scope rather
than corpus scope — not an independent movement. **The recorded prediction named only
"Priority 1 rises 5 → 6" and did not mention this flip, so it is reported explicitly
rather than folded into mover 1.**

### Blast radius — the edit reaches exactly the case it names

Papers appearing anywhere in the diff: PMC12180156, PMC12452463, PMC12782028,
PMC12856317, PMC13231680. **Only PMC12180156 moves substantively, and only its `research`
leg.** The other four appear solely because inserting PMC12180156 reindexed the sorted
lists they sit in; every one of those lists satisfies `set(A) ∪ {PMC12180156} == set(B)`.
Entity type touched: `compounds` only, `forbidden_kind = placeholder_product` — the
correct kind for a metabolite. No protein, complex, reaction, directionality,
stoichiometry or location value moved anywhere.

---

## 3. The cross-paper leakage the gold author predicted, materialized

PMC12180156/research's `δ-aminolevulinic acid` carries **the identical five accessions**
as PMC12856317's `aminolevulinic acid` rows (`research` /entities/compounds/2 and
`strict` /entities/compounds/1): chebi 17549, drugbank DB00855, hmdb HMDB0001149,
kegg C00430, pubchem 137.

The case's own `notes` names this failure mode in advance:

> "Cross-paper leakage from PMC12856317, which DOES support the ALAS2 reaction, is a
> concrete risk in a shared run."

The edit is what lets the benchmark see it. PMC12856317's own rows are **not** condemned —
`aminolevulinic acid` normalizes to `aminolevulinic acid`, which matches neither added
alias — so the correction stays case-local, exactly as D-072 guard rail 1 demands.

---

## 4. A registerable residual — the `reason` string the scorer now prints

The scorer propagates `forbidden_identifiers[0].reason` verbatim onto the new finding:

> "HALLUCINATION TEST: zero occurrences in the entire 67,304-character file, body and
> references alike. The paper names ALAS2 without ever naming its substrates or its
> product, so any of these is fabrication."

The **first** sentence is not literally true of the ASCII alias being added: the token
sequence `delta-aminolevulinic acid` does occur in the paper, inside the enzyme name the
same case quotes at `acceptable_enzymes[1]`. The **second** sentence is the one the edit
actually rests on, and it is spelling-independent and exactly on point.

The schema carries **one `reason` per entry, not one per alias**, so a reason is
structurally a statement about the entry's subject — the molecule ALA — and every alias is
a spelling of that subject. Under that reading the reason stands. Under a literal
character-occurrence reading it does not, and the sibling `UROD` entry shows this author
does reason about literal occurrences.

**This does not block half 1** — it is prose accuracy, not biological judgement, and the
finding it justifies is correct on the second sentence alone. It is registered so the Lead
can decide whether the first sentence wants a clarifying amendment. **That amendment is
NOT proposed here and must not be bundled into this edit.**

---

## 5. Half 2 — refused, and the refusal is itself reported

Half 2 (setting `supported_reactions_complete` on any gold case) was **not proposed, not
verified, not recommended, and not touched.** Working inside
`forbidden_identifiers[0].aliases` put me one field away from it in an already-open file.
Per REV-F150.md § 4 I record that the opportunity existed and was declined: **its answer is
"not proposed".** It remains the open product-owner question of § 4.2 of the handoff.
