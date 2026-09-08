# ORCH-732 — EXPECTED-WORKING-PWML-SMOKE. Result.

**Executed 2026-09-07 23:25 to 2026-09-08 02:37, 3h11m.** Six strict legs, one run each, no
retries. Frozen configuration: `ORCH-732-SMOKE-FREEZE.md`, committed at `48379628` **before**
execution. No production code was modified. `main` untouched. Production still frozen at
`c7a0663e`.

> ## A NEW dataset with its own identity.
> `runs_smoke/2026-09-07_2323`. **Not** the ORCH-724 pilot, **not** the ORCH-730 validation.
> Its numbers may never be merged into either denominator. Both remain read-only and are
> neither superseded nor rescored by anything here.

---

# 1. Headline

## 3 PWML files from 6 papers — 50 %.

The ORCH-724 unseen pilot produced **1 PWML from 20 legs**. The configuration here is
**byte-for-byte identical** to that pilot: same production SHA, same uncommitted
`streamlit_app.py` (`251122389a…`), same 16000/16000 budgets, same model, same temperature.

**The only variable that changed is which papers were chosen.** No code change is responsible
for the difference and none is claimed. **This does not rescore the pilot** — different
cohorts, different denominators, and a 6-paper sample is far too small for a rate.

---

# 2. Per-leg results

| # | paper | organism | pathway | reactions | PWML | bytes | release | time |
|---|---|---|---|---:|---|---:|---|---|
| 1 | `PMC9544450` | *E. coli* | menaquinone | **4** | **YES** | 39,686 | `review_required` | 22m18s |
| 2 | `PMC4725005` | *E. coli* | thiamine | **5** | **YES** | 54,459 | `review_required` | 24m31s |
| 3 | `PMC11487621` | *B. subtilis* | coenzyme A | 11 | NO | — | export blocked | 45m07s |
| 4 | `PMC12051542` | *A. aeolicus* | riboflavin | **8** | **YES** | 80,518 | `review_required` | 36m06s |
| 5 | `PMC10031235` | *H. sapiens* | L-serine | 6 | NO | — | `diagnostic_only` | 54m02s |
| 6 | `PMC7615680` | *H. sapiens* | coenzyme Q | — | NO | — | Stage-1 failure | 9m12s |

All three PWMLs parse as well-formed XML with a `super-pathway-visualization` root and are
candidates for PathWhiz manual import.

```
runs_smoke/2026-09-07_2323/papers/PMC9544450/strict/pathway.review_required.pwml    39,686 B
runs_smoke/2026-09-07_2323/papers/PMC4725005/strict/pathway.review_required.pwml    54,459 B
runs_smoke/2026-09-07_2323/papers/PMC12051542/strict/pathway.review_required.pwml   80,518 B
```

**Every PWML is `review_required`. None is `release_ready`.** Per the charter that is a
product success, not a shortfall: strict gates passed on all three, and the disposition
reflects incomplete coverage, which is exactly what `review_required` is for.

---

# 3. C-119 fired again, on a paper it was never derived from

`PMC12051542`'s disposition carries:

```
superseded_intermediate_contract_report:post_normalization_contract_report@audit_round=2
```

A stale `audit_round` contract report was recognized as **superseded** rather than treated as
a final blocker, and an 8-reaction pathway serialized that the pre-`C-119` code would have
destroyed. ORCH-730 proved this on `PMC7232280`; **this is an independent second instance on
fresh unseen input.** C-119 is doing in production exactly what it was merged to do.

---

# 4. The chemistry is sound — no fabrication

Reviewed by eye against the papers. **Nothing of the fabricated `glycine -> heme` kind
appears anywhere.** F-179 is holding.

**`PMC9544450`** — textbook-correct and matches the paper:
chorismate to isochorismate · MenD: oxoglutarate + isochorismate to SEPHCHC · MenH: SEPHCHC to
SHCHC · SEPHCHC to iso-SEPHCHC, the paper's own finding.

**`PMC4725005`** — the paper's central claim is captured correctly:
IlvA: threonine to 2-aminocrotonate · TrpD: 2-aminocrotonate + PRPP to PRA · RidA:
2-aminocrotonate to 2-ketobutyrate · PurF: PRPP to PRA · plus the nonenzymatic R5P + ammonia
route. *Minor omission:* PurF's glutamine co-substrate is not stated. Acceptable.

**`PMC12051542`** — a complete and correct riboflavin pathway:
GTP cyclohydrolase II · RibD · DHBP synthase · ARAPDP dephosphorylation, the paper's finding ·
lumazine synthase · riboflavin synthase, correctly emitting **riboflavin + ARAPD**, the
dismutation. *Minor defects:* riboflavin synthase is written from one lumazine rather than two,
so the stoichiometry is simplified; RibD appears twice, with and without NADPH; and one
"HFP isomerase" step is questionable. **Redundancy and simplification, not invention.**

Provisional manual-review classification, to be confirmed by the product owner:
**all three PASS WITH MINOR OMISSIONS.**

---

# 5. The three failures — classified by mechanism

## Two of the three are the same mechanism: identity resolution on name shape.

Neither is a biology failure. Neither is a `C-119` defect. **In both cases the correct
identity was already present or trivially recoverable, and a complete pathway died at the last
gate.**

### 5.1 `PMC11487621` — the species name was fragmented into three entities

The **richest extraction of the night** — 11 reactions, 3 transports, 17 proteins, 16
compounds — blocked by exactly two errors on one species entity:

```
species_missing_taxonomy        /entities/species/1   'B. subtilis'
species_missing_classification  /entities/species/1   'B. subtilis'
```

One organism became three species entities:

| # | name | taxonomy | outcome |
|---|---|---|---|
| 0 | `Bacillus subtilis` | 1423 | resolved, Prokaryote |
| 1 | `B. subtilis` | none | **unresolved — blocks the export** |
| 2 | `Bacillus subtilis (strain` | 224308 | resolved, Prokaryote |

**The same organism resolved correctly twice in the same payload.** An abbreviated alias never
folded into it, plus a third copy whose name is **truncated mid-parenthetical**, sank the
pathway. Entity [2] is independent evidence that the strain parenthetical is mishandled as a
*string*, which is the same seam `ORCH-731` found failing as `species_not_found`.

### 5.2 `PMC10031235` — the central enzyme's name was overwritten with a sentinel

```
gate.protein_psat_is_missing_a_uniprot_or_drugbank_identifier @ /entities/proteins/1
```

| protein | name | uniprot | status |
|---|---|---|---|
| [0] | `PHGDH` | `O43175` | matched |
| [1] | **`Unknown`** | **`Unknown`** | fallback |

`PSAT` is the paper's central enzyme — its title is *"Structure and function of phosphoserine
aminotransferase."* The identity ladder did not merely fail to resolve it; it **replaced the
name with the literal string `Unknown`**, which then failed the identifier gate.

**Live UniProt probe, run during this session — the identity was fully recoverable:**

```
(protein_name:"PSAT" OR gene:"PSAT") AND organism_name:"Homo sapiens"
  -> Q9Y617   reviewed   PSAT1   Homo sapiens   Phosphoserine aminotransferase
```

The paper's own abbreviation returns the correct reviewed accession as the **top and only**
hit. This confirms `ORCH-725`: the external services are fine, **the resolver is the problem.**

### 5.3 `PMC7615680` — Stage 1 returned two characters. NOT truncation.

```
stage=Stage 1 extraction  attempt 1  status=ok  finish_reason=stop  content_chars=2
stage=Stage 1 extraction  attempt 1  status=ok  finish_reason=stop  content_chars=2
```

**`finish_reason` is `stop`, not `length`.** The provider returned a complete, well-terminated,
essentially empty response — twice, 223 s and 553 s apart. This is **not** the `PMC8510960`
truncation class and **not** a token-budget event, and it must not be reported as one. A
distinct degenerate-extraction failure. One occurrence; not yet a pattern.

*Noted for the record:* this paper was flagged in the freeze as the cohort's weakest organism
match — an in-vitro reconstruction described as "in animals." It failed before organism
handling was ever reached, so that risk is untested, not exonerated.

---

# 6. What this says, and what should NOT be done next

## The dominant repeated blocker is identity resolution — `F-185`.

Of three failures, **two are the same mechanism**, and both destroyed complete, biologically
sound pathways over an entity whose correct identity was present or one query away. That is
the single dominant mechanism the charter asks to be identified before any code change is
considered.

`ORCH-725` concluded that fixing identity resolution would recover **0 additional PWMLs** in
the *pilot* cohort. **This cohort contradicts that generalization:** here it is the difference
between 3 and 5 PWMLs. The two findings are compatible — ORCH-731 established that `F-185` is
deterministic but its *impact* is stochastic, gated by which reactions Stage 1 draws. This
cohort simply drew reactions where the identity failures were load-bearing.

## What is NOT established

- **A 50 % rate.** Six papers is not a rate. It is an existence proof that the product
  generates usable PWML from reasonable pathway papers.
- **Any causal role for `C-119` in the yield difference.** The configuration is identical to
  the pilot's; the cohort changed.
- **That the two identity failures share one root cause.** One is a species-entity
  deduplication and normalization failure; the other is a protein-name fallback that destroys
  the name. They are the same *class*, at two different seams. **Whether one fix addresses
  both is unproven and must not be assumed.**

## Recommendation

**Do not open a broad engineering wave, and do not fix five unrelated edge cases.** If any
production change is authorized it should be **one narrow, evidence-preserving identity
correction**, scoped to whichever of the two seams is demonstrated to be load-bearing across
more than this cohort, with `F-179` regression protection and a `G9` proof that fails on the
base SHA.

**The stronger near-term move is manual review.** Three importable PWMLs now exist. Confirming
they survive PathWhiz import, and that the biology holds up to the product owner rather than to
me, is worth more to the manuscript than a fourth file.

---

# 7. Process

Bounded wrapper on every command, heavy lock held as `ORCH-732`, one heavy job at a time, no
detached processes, no `pytest -n auto`, no cache commits.

```
duration                : 11484.69 s
exit reason             : nonzero          (exit 1 -- 3 of 6 legs did not pass; the run itself is valid)
descendants observed    : 15
descendants terminated  : 15
FINAL SURVIVING COUNT   : 0
cleanup                 : success
heavy lock              : holder=ORCH-732  acquired=True  released=True
pre-existing (reported, NEVER killed): 4
```

The four pre-existing processes are the VS Code isort language servers and were never touched.
Cleanup targeted only PIDs this job created.

Evidence: `evidence/g11/ORCH-732/01-stage.json` (rejected search-based staging) ·
`02-stage-pinned.json` · `03-smoke-run.json`. Report tool:
`evidence/orch732_report.py`. The discarded first staging tree is preserved as
`runs_smoke/ABANDONED-search-based-staging-2026-09-07_2318`.
