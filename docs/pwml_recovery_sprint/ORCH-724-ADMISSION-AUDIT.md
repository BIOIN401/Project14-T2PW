# ORCH-724 — the final bounded admission audit, and the F-183 reachability check

**Lead Orchestrator, 2026-09-06.** Integration tip at start
`70b6d7d2c7e7bc0e5f33095f31fbf01fd985dfd1`, verified `local = origin/ = git ls-remote`.
`main` untouched (local `7531692`, remote `03f1af5`). Gold blob unchanged at
`98739a59dd6c376f8a19968c7fa5dc3145be5b15`. Heavy lock absent; the only Python processes
are the two `ms-python.isort … lsp_server.py` LSP servers, matched on full command line.

---

## 1. `F-183` — CLASSIFIED. Not a live production export bypass. No change needed.

**The question:** `pwml/writer.py` builds a PWML IR by a path that never calls
`validate_pre_export`, so the `F-179` reaction-support rule has no reach there. Is that
path **dead/test-only**, **live but already protected**, or **live and a real bypass**?

**The call graph is closed, and it was read rather than inferred.**

| fact | evidence |
|---|---|
| `validate_pre_export` has exactly ONE production call site | `src/t2pw/app/streamlit_app.py:4788` |
| that call site and the writer call site are in the SAME function | both inside `run_pwml_export`, which opens at `streamlit_app.py:4610`; the gate is at `:4788`, `DeterministicPwmlBuilder` at `:5003` |
| the second path is `writer.run_pwml_pipeline_export` | `src/t2pw/pwml/writer.py:2642`, reached only from `pwml_pipeline_cli_main` (`:2829`) ← `scripts/run_pwml.py:12-16` ← `README.md:40` |
| **no** production run reaches it | `grep` over `src/`, `scripts/`, `tests/`, and every `.ps1`/`.bat`/`.sh`/`.yml`/`Makefile`: the only invocations are **tests** and the README line a human types |
| the batch/benchmark path does NOT use it | `batch/driver.py` drives *"the real Streamlit app, headlessly"* via `streamlit.testing.v1.AppTest` (`driver.py:1,2258,2280`) — so it enters through `run_pwml_export` and is gated |
| the other CLI is not a PWML exporter | `scripts/run.py` → `legacy_sbml_cli_main` |

**Classification: live-but-off-the-product-path — class 2 in substance, and definitively
NOT class 3.** Stated precisely, because the distinction matters: the CLI is **not**
protected *by the F-179 rule* — it runs the normalization gate, pre-freeze
canonicalization, `blocking_pwml_ir_errors`, `validate_pwml_ir` and QA, but never
`validate_pre_export`. What makes it not a bypass is that **nothing in the product
reaches it.** It is a manual operator utility that converts an already-mapped payload the
operator supplies; it runs no LLM, detects no gaps and invents no chemistry. Every PWML
this project produces — including all twenty unseen-cohort legs — is written through
`batch/driver.py` → AppTest → `run_pwml_export`, where the gate stands.

**Consequence: `F-183` does not delay the project and no code changes for it.** It remains
OPEN as a registered hygiene item: if the CLI is ever wired into a batch or app flow, it
needs the same seam first. That is a precondition on a future change, not work owed now.

---

## 2. Task B — the admission audit

### 2.1 The instrument

`evidence/orch724_admission_audit.py` — evaluation-only, read-only, re-runs no leg. It
scores the gate's **rejections** against the **41-reaction curation corpus**, not the
19-signature recovery gold. Chemistry matching uses `bench.semantic._signature_matches`,
the same matcher the production scorer and `rd093_rag_metrics.py` use, so this audit
cannot disagree with the committed RAG metrics about what "the same reaction" means.

Bounded-wrapper evidence: `evidence/g11/ORCH-724/01-admission-audit.json`,
`02-sole-blocker.json`, `03-roles-spans.json`. All three: `FINAL SURVIVING COUNT : 0`,
`cleanup : success`.

### 2.2 The curated positives are credible

**All 41 curated core reactions pass the quote screen — 0 excluded.** Every curated
`quote` was located verbatim in that paper's stored full text under
`data/rag_index/acquire_cache/fulltext/`, folded with `bench.goldset.fold_for_quote` (the
production folding, which drops punctuation because the cached text carries
italic-stripping artifacts). A curated entry whose quote could not be found would have
been excluded from the decision set; none had to be.

### 2.3 The raw counts, and why they overstate the problem

| population | legs | rejections | matching a curated reaction | accepted |
|---|---|---|---|---|
| untruncated | 88 | 7,299 | **2,439** | 15 (8 curated-matching) |
| truncated | 59 | 11,800 | 3,708 | 80 (11 curated-matching) |

Truncated legs are kept apart and never summed, per F-177 and `rd093_rag_metrics.py`.

**2,439 is not 2,439 distinct mistakes.** The same handful of spans is re-proposed against
many gaps, in many legs, across many runs. The product-relevant unit is
*(run, leg, curated reaction)*: **95 such pairs were never admitted**, and the reason that
actually blocked each one is spread across five codes, not concentrated in one:

| minimal blocking reason set, per (run, leg, curated reaction) | count |
|---|---|
| `evidence_relation_disagrees_with_claim` | 27 |
| `no_local_evidence_span` | 20 |
| **`evidence_relation_roles_unassignable`** | **16** |
| `candidate_type_cannot_fill_gap` | 11 |
| `evidence_states_no_reaction_relation` | 8 |
| `unsupported_catalyst_injection` | 4 |
| others (each ≤ 3) | 9 |

### 2.4 Classification — most of the rejecting is CORRECT

- **`no_local_evidence_span` (20) — CORRECT REJECTION.** No span, no admission. That is
  exactly what `F-179` exists to enforce, and weakening it is forbidden by merge rule 6.
- **`candidate_type_cannot_fill_gap` (11) — CORRECT REJECTION.** Inspected: the gap asks
  for *the identity of `MenF`* (`gap-unmapped_enzyme-…`) or *a compartment for `SEPHCHC`*
  (`gap-missing_compartment-…`), and the candidate is a **reaction**. A reaction is not an
  identifier resolution and expresses no location. The gate even names the right route
  (`identity_resolver`). This is correct behaviour; that a correct reaction is discarded
  here is a **gap-routing** property, not an admission-gate defect, and routing is out of
  this pass's scope.
- **`evidence_states_no_reaction_relation` (8) — CORRECT REJECTION** on inspection.
- **`evidence_relation_disagrees_with_claim` (27) — MIXED / AMBIGUOUS.** Some are real
  disagreements. Others are parsing artifacts of two kinds: a locant comma splitting a
  compound (`(SEPHCHC, C)` read as two fragments), and a multi-clause span where only one
  clause is read (`MenA joins … to produce DMK, and MenG demethylates DMK to generate MK`
  scored against the MenG clause). **These are not one mechanism and not one fix**, and
  `test_a_locant_comma_does_not_split_a_compound` shows the family is already partly
  addressed. Not chosen.
- **`evidence_relation_roles_unassignable` (16) — OVERSTRICT REJECTION, and it is the one
  clear repeated mechanism.** See below.

### 2.5 The one repeated over-rejection mechanism

Restricting to rejections where this code was the **sole** blocker, over untruncated legs,
on candidates whose chemistry matches a credible curated reaction:

**294 rejections — and only 12 distinct evidence spans.**

| construction | rejections | share | distinct spans |
|---|---|---|---|
| **nominalized `conversion of X to Y`, catalyst in a following `catalyzed by …`** | **211** | **71.8%** | 5 |
| copula + adverb (`is first converted to`) | 66 | 22.4% | 1 |
| other | 17 | 5.8% | 6 |

Every one of the five nominalized spans carries an explicit `catalyzed by`. Examples,
verbatim from the archive, each with `requested_pathway_match = match` and
`organism_match = match`, each on a **connectivity** gap the candidate could fill:

- `conversion of chorismate to isochorismate (catalyzed by EntC)`
- `Chorismate to Isochorismate : The pathway begins with the conversion of chorismate to
  isochorismate, catalyzed by isochorismate synthase (EntC) .`
- `The key steps in enterobactin production … include the conversion of chorismate to
  isochorismate (catalyzed by EntC), formation of 2,3-dihydroxybenzoate (DHB) by EntB, …`

These name substrate, product **and** catalyst explicitly. They were refused with *"the
span states a relation but no implemented pattern could assign substrate/product/catalyst
roles"* — and that is accurate: **no template in `_ALL_PROSE_PATTERNS` reads the bare
nominalization.** `catalyzes_to_subjectless` requires the governing verb *"catalyzes the
conversion of A to B"*; these sentences put the nominalization first and the catalyst
after, in a participial phrase or a parenthetical. Confirmed directly —
`parse_span_relation` returns `None` for all four.

The curated reactions lost this way are `PMC12452463:R1`, `PMC12096016:R1` and
`PMC12421875:R8` — including **the first step of the enterobactin pathway**, a defining
core reaction, not a cofactor or a regulator.

**Verdict: OVERSTRICT REJECTION, repeated, and general.** The gap is a missing English
construction, not a paper-specific miss and not a threshold that is set too high.

---

## 3. Decision

**One narrow admission correction is authorized — `D-095` below — and nothing else.**

The correction chosen is the **nominalized-conversion template** (71.8% of the population,
the widest and most general of the three constructions). The copula-adverb form and the
`evidence_relation_disagrees_with_claim` parsing artifacts are **NOT** fixed in this pass:
the authorization is for one mechanism, and taking a second would be exactly the endless
recall optimization this pass exists to avoid.

Scope note recorded honestly: measured on **this** corpus of ten papers the change
recovers **3 of 41** curated core reactions. It is authorized on **generality**, not on
that count — the construction is a common way papers state a catalysed step, and the
unseen cohort is the population that matters.

Precedent: **C-061** did precisely this, adding one template (`actor_combines_to_make`) to
`_EXTRA_PROSE_PATTERNS` for one measured construction, with an empirical safety case. This
card follows that shape exactly.
