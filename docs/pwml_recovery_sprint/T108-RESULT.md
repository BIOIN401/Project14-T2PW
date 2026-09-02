# T-108 — official result

**Run:** `runs_verify/2026-09-01_1612` · **launched once**, 2026-09-01T22:14:52Z · **finished**
2026-09-02T04:37:01Z · **22929.17 s = 6.37 h** · integration tip at launch `0bbac3fd`.

# VERDICT: **NOT ACCEPTED**

**Priorities 1, 4 and 5 report `ok=false`; Priority 2 is `NOT EVALUATED`. Priority 3 passes.**
Not every locked hard acceptance condition passes, so T-108 is not accepted.

**T-108 is preserved as a failed official release candidate. It is NOT re-run.** A later candidate
needs a new milestone identity and a separately recorded readiness decision.

**T-107 is untouched by this document.** Its `NOT ACCEPTED` verdict is a fact about the artifacts it
produced. Nothing here confirms or overturns it, and the two runs were scored against **different
gold sets** — see § 4.1, which is the single most important caveat in this report.

---

## 1. Completion — the run itself was clean

| | T-108 | T-107 |
|---|---|---|
| papers / legs | **10/10 · 20/20**, `complete: true` | 10/10 · 20/20 |
| payloads available | **19** | 17 |
| semantically scorable legs | **19** | 17 |
| timeouts | **1** | **3** |
| wrapper | exit 1 (`nonzero`), 22929.17 s of 72000 s | exit 1, 20282 s of 21600 s |
| survivors / cleanup | **0 / success** | 0 / success |
| heavy lock | acquired **and released** | acquired and released |

**Exit code 1 is the expected outcome, not an infrastructure failure.** `batch_run.py` returns 1
when not every leg passes; 12 of 20 did not pass. T-107 exited 1 for the same reason.

### Status tally

`pass 8 · fail 5 · scope_conflict 6 · timeout 1`

Six `scope_conflict` legs are the three deliberate organism traps aborting at Stage 0 exactly as
designed — `eligibility_stage0_conflict_aborts = True`. Nothing was exported from any of them.

## 2. The leg ceiling — the ruling worked, measured

**One timeout instead of three, and the scorable denominator grew from 17 to 19.**

Two legs that T-107 lost to the clock completed at the restored 3600 s ceiling:

| Leg | T-107 @ 1800 s | T-108 @ 3600 s | % of the OLD ceiling |
|---|---|---|---|
| `PMC12444477/research` | TIMEOUT 1800.5 s, **0 files** | **PASS** 2446.5 s, **18 files** | **136%** |
| `PMC12096016/strict` | TIMEOUT 1800.2 s, **0 files** | **PASS** 1952.9 s, **21 files**, PWML **74367 B**, 6 reactions, gate_errors 0, blocking_issues 0 | **108.5%** |

`PMC12096016/strict` — a **core, `strict_exportable`** paper, half of Priority 5's entire denominator
— needed **152.9 seconds, two and a half minutes**, beyond T-107's ceiling. T-107 discarded a clean,
complete 74 KB export because a ceiling had been halved with no recorded reason. That is exactly
F-148's diagnosis, *lost to the clock rather than rejected biologically*, now measured.

**The one remaining timeout is new information and is reported as such.**
`PMC12444477/strict` consumed the **full** 3600 s (3600.79 s, 100.0% of ceiling,
`budget.leg_timeout_overridden: false`). On T-107 it timed out at 1798.3 s. **Doubling the budget did
not let it finish.** Per § 2.1's own limit: a timeout at 3600 s is **not automatically a defect**,
and must **not** be waved away. It is censored — it proves the leg needs *more* than 3600 s and never
how much more. The census maximum of 3421.4 s was **not** an upper bound for this leg.

**The override audit is clean and its blind spot is stated:** zero legs recorded
`leg_timeout_overridden: true`. Only the timeout row carries a budget block at all, so for the other
19 legs the field's absence is *no observation*, not proof of `false`. The run-wide claim rests on
the pre-launch resolution (`_ceiling(3600.0) -> overridden False`, G11 `T-108/07`) plus that one
produced artifact confirming it.

**C-111 did on a live timeout what F-148 recorded T-107 could not.** The timed-out leg preserved
`LEG_TERMINAL.json` (1131 B), `LEG_TRACE.jsonl` (24564 B) and `RESULT.txt` (5157 B). T-107's three
timed-out legs preserved no payload to inspect. That instrument gap is closed, observably.

## 3. Acceptance priorities

| # | Priority | Result | `ok` |
|---|---|---|---|
| 1 | zero known false real identifiers | **raw 2 · accepted 2 · `accepted_status: PASS`** (target 6) | **false** |
| 2 | zero unsupported retained reactions | **`NOT EVALUATED`** — 0 counted; verdict never reached on **12 of 19** scored legs across **8** papers | `null` |
| 3 | zero referential-integrity violations | **0 violations** | **true** |
| 4 | meaningful requested-pathway coverage | **0/8 = 0%** | **false** |
| 5 | strict PWML pass rate among eligible papers | **0/2 = 0%** | **false** |

### 3.1 Priority 1 — both rows, named

`raw = accepted = 2`. **`contract_adjusted_rows` is empty**, and under D-074 as ruled no Priority-1
row *can* be contract-adjusted, so `accepted` is identically `raw`. **That zero means "no licence can
reach this", not "none was measured".**

| Paper / mode | pointer | name | identifiers |
|---|---|---|---|
| `PMC12444477/research` | `/entities/compounds/9` | `(p)ppGpp` | `hmdb HMDB0060480`, `pubchem 38166` |
| `PMC12180156/strict` | `/entities/compounds/2` | **`δ-aminolevulinic acid`** | `drugbank DB00855`, `hmdb HMDB0001149`, `kegg C00430`, `chebi 17549`, `pubchem 137` |

**`accepted_status: PASS`** because 0–6 is the PASS band. **The absolute `ok` is `false`** because it
computes zero-tolerance on the raw count, deliberately, per `PRODUCT_CONTRACT` § 15. Both are
reported; neither collapses into the other.

### 3.2 Priority 5 — the number did not move and its meaning changed completely

**This is the most important result in the run.**

| `strict_exportable` leg | T-107 | T-108 |
|---|---|---|
| `PMC12096016/strict` | **TIMEOUT** — operational loss | `review_required` · gates **passed** · semantic **passed** · completeness **0.75** · unmatched `NADH, ATP, EntD, Fur` |
| `PMC12782028/strict` | passed runner, below coverage minimum | `review_required` · gates **passed** · semantic **passed** · completeness **0.538** · unmatched `oxysterols, MSMO1, SQLE, FDFT1, HMGCR, HMGCS1` |

**T-107's `0/2` was one operational loss plus one coverage shortfall. T-108's `0/2` is two coverage
shortfalls and zero operational losses.** Both legs executed fully, cleared the strict technical
gates, passed semantic evaluation, produced valid PWML, and are held at `review_required` for
incomplete requested-core coverage.

**That is merge rule 7 working as written** — incomplete-but-correct pathways preserved as
`review_required`, never dropped and never promoted. **A runner `pass` is not a Priority-5 point.**

**The denominator is now honest.** Restoring the ceiling did not improve the number; it converted an
operational failure into a measurable biological result, which is precisely what § 2.1 said it would
do: *prevent timeout policy from silently shrinking the scorable biological denominator.*

### 3.3 Priority 2 — `NOT EVALUATED`, and that is the correct behaviour

The unsupported-reaction verdict was never reached on **12 of 19** scored legs covering **8** papers,
because `supported_reactions_complete` is `false` on all ten gold cases and the signature sets are
subsets. `semantic.py` stamps `UNSUPPORTED-REACTION VERDICT NOT EVALUATED` and withholds
`false_positives` rather than reporting a hard zero.

> **Priority 2 = 0 counted is the absence of a measurement, not the absence of unsupported
> reactions.** It is not a measure of how much invented chemistry this run produced.

**Reported as an acceptance-instrument limitation under D-087 clause 6.** `max_retained_reactions` is
set on exactly two cases and both are negative controls, so Priority 2 remains unevaluable on any
non-control paper — on this run and on every future one, until the D-087 standard is met by a case.

### 3.4 Negative controls

| Paper | Legs | Assessment |
|---|---|---|
| `PMC13231680` (`is_negative_control=True`, `max_retained_reactions=0`) | strict `fail no_reactions` 624.8 s · research `fail no_reactions` 360.4 s | Both produced an **empty pathway**, which gold calls the correct outcome. `operational_failure=null`, `termination_reason=null` — **not** caused by timeout, crash or infrastructure failure, satisfying Q1 ruling condition 3 on evidence |
| `PMC12180156` (`context_only`, `max_retained_reactions=2`) | strict `fail` → `diagnostic_only` · research `pass` | strict failed `actor_named_in_its_own_cited_span` with `strict_technical_gates_blocked_export` |

**On `PMC13231680/research`, T-108 differs from T-107 and it must not be over-read.** T-107 had
`PASS` 795.2 s with a full research report; T-108 produced an empty pathway. Mechanism measured:
stage 0 `ok` 1 attempt, stage 1 `outcome=ok` `response_status=ok` `finish_reason=stop` **`attempts=1`**,
entities extracted (species 1, compounds 3, proteins 2) but **0 reactions**.

**This is NOT evidence that F-146 is fixed.** F-146 is this exact leg retaining an invented reaction,
and it was Priority 2's single row on T-107. Its absence on **one draw** is not a fix. The standing
trap says a single leg must not be called a regression at temperature 0; **the symmetric rule binds,
so it must not be called an improvement either.** The artifacts also cannot separate *"the pipeline
declined"* from *"this draw extracted nothing"* — zero reactions is not a recorded refusal.

## 4. Comparability — read this before comparing any number to T-107

### 4.1 T-107 and T-108 were scored against DIFFERENT GOLD SETS

C-113 merged at `db119f53` on **2026-09-01**, three days after T-107 ran, moving the gold blob
`aee8cb4f → 36f4b7b6` and adding `delta-aminolevulinic acid` and `δ-aminolevulinic acid` to
`PMC12180156`'s forbidden aliases. Verified directly:

```
BEFORE C-113 : ['ALA','porphobilinogen','protoporphyrin IX','succinyl-CoA',
                'coproporphyrinogen III','uroporphyrinogen III']
AFTER  C-113 : [... , 'delta-aminolevulinic acid', 'δ-aminolevulinic acid']
```

**One of T-108's two Priority-1 rows — `δ-aminolevulinic acid` — is detectable ONLY under T-108's
gold.** Under T-107's gold `forbidden_match` returned `None` for that spelling and the row was
invisible. So **a Priority-1 count from T-107 and one from T-108 are measurements taken with
different instruments**, and any "8 → 2" style comparison is not apples to apples. C-113 made the
instrument stricter *and* T-108 scored lower; those facts pull in opposite directions and must not be
combined into a single improvement claim.

### 4.2 Draw variance is large, and one control proves the pipeline is not unstable

`PMC12782028/strict` is near-deterministic across runs: **590.6 s vs 596.6 s**, PWML **34931 B vs
35295 B**, both 0 gate errors, 0 blocking issues. So where the draw is stable the pipeline is stable,
and the divergences elsewhere are **draw-specific, not run-wide instability**.

Three legs diverged from T-107 in **both** directions, each explained by draw variance meeting an
unchanged gate:

- **`PMC12856317/strict` `PASS → FAIL(contract)`. This reads as a regression and is not one.**
  Checked rather than assumed: T-107's `final_mapped.json` for that leg contains exactly two
  proteins, `ALAS1` and `ALAS2` — **no ClpXP**. The gate had nothing to fire on. T-108's draw
  additionally extracted ClpXP (a protease, not a heme-biosynthesis enzyme) with no accession, and
  the § 8 identity gate refused it. **The gate did not change; its input did.** Elapsed times are
  near-identical (1118.1 s vs 1122.7 s), consistent with a same-shape run diverging at extraction.
- **`PMC12180156/strict`** fails on the same seam in both runs, with a different protein:
  `ALAS2` on T-107, `ferrochelatase` on T-108. **F-147 reproducing** — a registered, deliberately
  uncharted `product_contract_violation`. Not a new finding.
- **`PMC12452463/strict` blocking-issue count across three runs: T-106 = 7, T-107 = 3, T-108 = 6.**
  Same seam, same core codes every time. **The previous wave recorded "PMC12452463 improved at T-107,
  7 contract errors to 3." T-108 shows that was draw variance, not improvement.** Reading any two of
  `7 → 3 → 6` as a trend is wrong in whichever direction the last two points fall.

### 4.3 The dominant strict-mode blocker

The gate class `protein_<X>_is_missing_a_uniprot_or_drugbank_identifier` fired on **three strict legs
across two papers** this run (ClpXP on `PMC12856317`, ferrochelatase on `PMC12180156`, `fur` and
`ryhb` on `PMC12452463`). T-108's `PMC12452463` draw also extracted **RyhB**, named in the handoff as
gold-forbidden content, and the gate caught it. **These are gates refusing output. Nothing here
argues for weakening any of them** — merge rule 6 points the other way.

## 5. Items that remain unmeasured or unchanged

**`LpxH` remains UNVERIFIED.** `PMC12444477/strict` timed out again with no payload; the research leg
passed but carries **0 findings**. It is verified only on the pinned run `runs/2026-08-02_2130`, and
**no T-108 result confirms it.**

**Priority 3 is the one clean pass:** zero referential-integrity violations.

**Priority 4 = 0/8**, with `PMC12444477:research` the single leg still below the coverage minimum
(`legs_with_coverage: 11`, `legs_with_forbidden_terms: 8`, `forbidden_terms_excluded: 9`).

## 6. Provenance, cost and process

| | |
|---|---|
| Provider / models | `openrouter`, all nine slots `deepseek/deepseek-v4-flash`, `LLM_TEMPERATURE=0`, `LLM_MAX_RETRIES=3`, **no fallback**. `.env` unmodified |
| Curator | **ONLINE** — `T2PW_OFFLINE_CURATOR` deliberately not set. Matches T-107, verified from its artifacts. `T108-MANIFEST.md` § 5 |
| Gold | `36f4b7b690b577f72882c3045ca6728d1ec8d9d1`, version `2026-08-01.1`, unchanged before and after |
| `acceptance.py` | `sha256:4bd893ac410d16d3…` unchanged before and after |
| `streamlit_app.py` | `sha256:47e4fafa789d359d…` unchanged, still uncommitted |
| Elapsed | 22929.17 s = **6.37 h** of a 72000 s wrapper ceiling (31.8%) |
| **Cost** | **Not separately attributable.** The run tree records **no token usage at all**. Cumulative account usage moved `$158.72 → $162.71` = **$3.99** since the 2026-08-21 D-058 reading, a window spanning T-104 through T-108 **and other work**. T-108's share cannot be isolated. This is D-086's registered observability hole, unchanged |

**Process:** launched **exactly once**; staged then continued **without `--fresh`**
(`already recorded : 0`, `still to do : 20`); no gold, scorer or production change before or after
the result; `FINAL SURVIVING COUNT : 0`; `cleanup : success`; heavy lock released; zero sprint-owned
Python afterwards.

## 7. Classification — what may and may not be chartered

Under `PRODUCT_CONTRACT` § 14, **a benchmark failure does not by itself justify a code change.**

| Observation | Class | Action |
|---|---|---|
| `PMC12180156/strict`, `PMC12452463/strict` contract failures | `product_contract_violation` | **F-147, already registered and deliberately UNCHARTERED.** Merge rule 6: a downstream-only fix would flip both legs to PASS and export gold-forbidden content |
| Priority 2 unevaluable on every non-control paper | **acceptance-instrument limitation** | Reported under **D-087 clause 6**. Not a code defect |
| `PMC12444477/strict` timeout at 3600 s | **operational, censored** | New information about the requirement. **Not automatically a defect**, not waved away. Belongs in the next readiness decision |
| `PMC12856317/strict` ClpXP gate failure | **not a defect** — gate refusing an unaccessioned protein | None. Weakening it is the merge-rule-6 direction |
| Priority 5 `0/2` via `review_required` | **not a defect** — merge rule 7 working | None from this run |
| `PMC13231680/research` empty pathway | **indeterminate on one draw** | **Not chartered.** Do not record F-146 as fixed |

**No code change is chartered from this result.** The two genuine `product_contract_violation`s are
F-147, which is registered and deliberately left unchartered because fixing it downstream would
export gold-forbidden content.

## 8. Standing limits carried forward

> **Priority 2 = 0 counted is a real reading and it is not a measure of how much invented chemistry
> this run produced.**

> **A Priority-1 count from T-107 and one from T-108 were measured against different gold sets.**

> **`LpxH` is UNVERIFIED on T-108.**

> **T-108 is immutable. It is not re-run, re-scored or reinterpreted.**
