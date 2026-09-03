# RESUME — next session handoff

## 0. CURRENT — **`ORCH-723`: the RAG/LLM EVALUATION phase. `R-D092-1` built, the lineage-aware evaluator and TWO TABLES built, Phoenix started and ingested, core RAG metrics measured. Production still FROZEN.** 2026-09-03.

> **⚠ NOTHING IS RUNNING AND NOTHING IS CHARTERED.** Heavy lock free (`C:/t/heavylock` absent), zero
> sprint-owned Python beyond the two `ms-python.isort` LSP processes. **`D-090` still controls:
> production is FROZEN and T-110 is NOT authorized. `supported_reactions_complete` remains UNSET on
> all ten gold cases and was NOT reopened.**

**Verify the tip yourself:** `git rev-parse HEAD` = `git rev-parse origin/sprint/pwml-recovery`
= `git ls-remote origin sprint/pwml-recovery`. **`main` untouched — local `7531692`, remote `03f1af5`.**
Gold blob unchanged at `98739a59dd6c376f8a19968c7fa5dc3145be5b15`.

### What ORCH-723 delivered — D-093 § 5 items 2 through 7

| # | item | status |
|---|---|---|
| 2 | **`R-D092-1`** row-level RAG lineage | **BUILT** — `evidence/rd092_1_reaction_lineage.py`, 17 tests |
| 3 | lineage-aware deterministic evaluator | **BUILT** — `evidence/rd093_two_table_metrics.py`, 9 tests |
| 4 | re-evaluate archived canonical into the three classes | **DONE** — 1,042 rows over 175 legs |
| 5 | target-paper P/R/F1 **separate** from the unsupported rate | **DONE** — two tables, two denominators |
| 6 | **Phoenix** started and records ingested | **DONE and VERIFIED** — 1,314 spans queried back out of the store |
| 7 | core RAG metrics | **DONE** — `evidence/rd093_rag_metrics.py`, 15 tests |
| 8 | validate on archived runs | **DONE** — tests prove against a REAL archived leg |
| 9 | freeze and select the ten unseen papers | **NOT STARTED, and deliberately so** — see below |

### The one thing to read first

**RETRIEVAL IS NOT THE BOTTLENECK. ADMISSION IS.** On the 105 untruncated committed legs,
`Recall@5 = 93.0%` and only **55** gold signatures were never retrieved at all — while
**1,123 of 1,212 positive queries** end in `correct_candidate_rejected`. The gate refuses a
gold-matching, well-ranked candidate in the large majority of cases. **One blended "RAG accuracy"
number would have hidden this entirely**, which is exactly why D-093 § 6 forbids one.

Read the `100.0%` negative-query rejection beside it: the gate admits almost nothing anywhere
(**15 accepted candidates across all 19 T-109 legs**), so a perfect negative score is a consequence
of near-total rejection, not independent evidence of discrimination.

### Measured results — per population, NEVER summed (F-177 discipline extended)

**Support classes (D-093 § 1), over the committed corpus:**

| population | legs | reactions | target_paper | external_rag | unsupported | indeterminate |
|---|---|---|---|---|---|---|
| canonical | 114 | 433 | 82.4% | 1.4% | **0** | 15.9% |
| fallback | 61 | 609 | 73.1% | 0 | **0** | 26.9% |

**The two tables — different denominators, not addable:**

| population | TABLE 1 recall | TABLE 1 precision | TABLE 2 unsupported rate |
|---|---|---|---|
| canonical | 60.0% = 135/225 sig-leg pairs | 42.1% = 142/337 rows claimed | 0.0% = 0/419 retained |
| fallback | 91.5% = 107/117 sig-leg pairs | 34.0% = 143/420 rows claimed | 0.0% = 0/578 retained |

**`unsupported` is ZERO in both populations, and that is a real finding, not an empty check.**
Every candidate `unsupported` verdict rested on a **cross-run** chunk join, and the conservative
rule refuses to charge a row on another run's retrieval draw. Within-run evidence that a canonical
reaction lacks defensible support does not exist in the committed corpus.

### FIVE MEASURED FACTS that anyone touching this evaluator must know

1. **The brief's premise is out of date, in the chartered direction.** Not "lineage lives only on
   entities": **236 of 1,079** committed reaction rows carry `rag_provenance` and **431** carry
   `provenance_lineage`. `pipeline._carry_rag_provenance` shipped that carrier. **Three provenance
   eras**, not one.
2. **Lineage `support="unsupported"` is NOT D-093's `unsupported`.** 650 of 692 reaction lineage
   entries are `(paper_stated, explicit)` with `support=unsupported`, because that field grades
   whether a **named source** backs the row. Reading it as a biological verdict relabels 650
   paper-explicit rows as unsupported — **the D-091 collapse one level down.**
3. **`origin="rag_literature"` does not mean external.** 11 of 14 such source refs point AT the
   target paper (9 via the `seed_paper` sentinel, 2 by id). Externality is resolved from
   `source_id`, never from `origin`.
4. **The reaction `evidence` string is not reaction-specific and silently carries EXTERNAL text.**
   On the cited leg it is **35,029 characters of PMC8091085's abstract**, while that leg's own
   `01_source_text.txt` contains **zero** occurrences of `MenI`, `DHNA-CoA thioesterase` or
   `LMRG_02730`. A "row has evidence therefore the paper supports it" test **passes every row in
   the corpus** and launders external text as target-paper support.
5. **Participant inheritance cannot establish D-093 condition 1.** Entity provenance names a span
   that mentions a PARTICIPANT, never one that states the reaction. Inheritance alone is capped at
   `indeterminate`. The **chunk join** is the only deterministic bridge, and it resolves 37 of 79
   entity chunks — including the `fb1cf2b2…` chunk D-091 turned on.

### The inheritance rule, adopted and documented as D-093 required

Four named tiers in **strict precedence, never a union**: `row_lineage` → `row_rag_provenance` →
`participant_inheritance` → `no_signal`. A row carrying its own lineage was attributed by the stage
that introduced it; letting participant provenance override it would let an entity's retrieval
history rewrite a reaction's attribution. Entity-name lookup **keeps every colliding record** rather
than picking a winner (`isochorismate`, `SEPHCHC` and `MenD` each appear twice on the cited leg).
`seed_paper` and the leg target both resolve to TARGET; an **empty id resolves to UNRESOLVED, never
to target** — absence is not attribution.

### Corrections to inherited figures — measure, do not trust

- **"1,947 rejected candidates" reconciles with nothing measurable.** T-109
  (`runs_verify/2026-09-01_1612`, 19 committed legs, disk identical to HEAD) measures **3,276
  considered / 3,261 rejected** by its own counts, **2,076 rejected rows PERSISTED**, and **1,795
  unique** `(gap_id, name, chunk_id)`. Five legs truncate, so **1,185 counted rejections were never
  persisted** and their provenance is genuinely `unavailable`.
- Corpus-wide the truncation is larger: **57 of 162 legs** hit `max_report_entries` and **12,838**
  candidates were counted but never persisted. **Truncated legs are a separate population** and are
  never summed with clean ones.

### Traps this wave paid for

1. **A missing key read as zero, twice, in my own instruments.** (a) `R-D092-1` first tiered on
   lineage **sources** rather than the lineage **key**, demoting all 650 sourceless `paper_stated`
   attributions to inheritance or `no_signal`; fixing it moved canonical `row_lineage` **2 → 232**
   and `indeterminate` **259 → 67**. (b) The RAG metrics first printed three **structural zeros** —
   `retrieval_did_not_find_it`, `found_but_not_admitted`, `rejected_candidate_reintroduced` — for
   categories nothing ever assigned. A gold signature no candidate matches produces **no gap**, so a
   per-gap counter can never see it. **Both were caught by cross-checking against an independent
   census before publishing.**
2. **The two payload populations disagree on schema.** Every one of the 702 canonical enzyme records
   is keyed `entity`; fallback rows key it `protein`. Reading one drops the enzyme from every row of
   the other population.
3. **`provenance` on a canonical enzyme is a DIFFERENT vocabulary wearing the same word** —
   `extracted` (612) / `inferred` (90), which is HOW, not WHERE. It is not source attribution.
4. **Never combine a shell `&` with a backgrounded `bounded_run`.** The harness task reports
   "completed" while the wrapper keeps running. The processes were never orphaned (the Job Object
   held them and every PID was recorded), but tracking had to be done by hand.
5. **A denominator that could have misled.** The two-table report first printed "135 of 225 verified
   gold signatures"; gold states **13 distinct** signatures across the 6 scored papers, and 225 is
   the micro-averaged count of **(signature, leg) pairs**. Both are now printed and the unit named.

### Why item 9 — the ten unseen papers — was NOT started

**D-093 § 5 gates it: "Only then freeze and select the ten unseen papers", and "do not consume the
unseen cohort before capture and scoring demonstrably work on archived data."** Capture and scoring
now demonstrably work on archived data, so the gate is arguably met — but consuming the unseen
cohort is a **one-way door** and the instruments have not been independently reviewed. **It needs a
product-owner go, not a Lead's judgement call.**

### What the next session should weigh

1. **The admission gate is where the pathway is lost, and that is now measured rather than
   suspected.** `correct_candidate_rejected` = 1,123 against `retrieval_did_not_find_it` = 55. The
   rejection taxonomy is already on disk (`reason_counts`): `candidate_type_cannot_fill_gap`,
   `evidence_relation_roles_unassignable`, `evidence_states_no_reaction_relation`,
   `no_local_evidence_span`. **Sampling those against gold would say whether the gate is correctly
   strict or wrongly strict — and that is the highest-value next question.** It is an EVALUATION
   question and needs no unfreeze.
2. **Independent review of these four instruments.** Every merge rule was met and the gates are
   green, but the Lead does not approve its own work, and R-D092-1's classifier is now load-bearing
   for Priority 2.
3. **Priority 2 is now answerable in principle** — the evaluator knows where each reaction came
   from, which D-093 § 4 named as the precondition. It still needs a product-owner ruling, and the
   flag stays UNSET until then.
4. **Phoenix is a bounded job, not a service.** Relaunch:
   `PHOENIX_WORKING_DIR=C:/t/phoenix_t2pw C:/t/phxenv/Scripts/python.exe -m phoenix.server.main serve`
   under `bounded_run.py` with a real `--timeout`, then re-ingest. The eval venv is deliberately
   **separate from the project `.venv`** so no dashboard dependency can reach a merge gate.

### Verified state at this tip — measure, do not trust

| check | expected |
|---|---|
| Gold | `98739a59dd6c376f8a19968c7fa5dc3145be5b15`, `supported_reactions_complete` UNSET on all ten |
| SMOKE | **508 passed** — `g11/ORCH-723/27` |
| gold-readers split | **465 / 0 / 8 / 0** — `g11/ORCH-723/28` |
| new focused tests | **54 passed** across four files — `g11/ORCH-723/26` |
| G11 strict | **29 artifacts, 0 non-compliant**, all four strict flags |
| `streamlit_app.py` | sha256 `47e4fafa…`, **modified and never committed** |
| Python processes | exactly two `ms-python.isort … lsp_server.py` — **match on FULL COMMAND LINE** |

## 0-prevORCH722 — **SUPERSEDED by `ORCH-723`, 2026-09-03.** Its F-175/F-176 closures, F-177, F-178 and the D-091 withdrawal all STAND; what is superseded is its status as current and its statement that the evaluation stack is NOT started — ORCH-723 built it. **`ORCH-722`: F-175 and F-176 CLOSED under narrow unfreezes, F-177 built, F-178 found and fixed, and the Priority-2 flag was set, measured and WITHDRAWN. Production still FROZEN.** 2026-09-03.

> **⚠ NOTHING IS RUNNING AND NOTHING IS CHARTERED.** Heavy lock free (`C:/t/heavylock` absent), zero
> sprint-owned Python beyond the two `ms-python.isort` LSP processes, no unowned job. **`D-090`
> still controls: production is FROZEN and T-110 is NOT authorized.**

**Verify the integration tip yourself:** `git rev-parse HEAD` = `git rev-parse origin/sprint/pwml-recovery`
= `git ls-remote origin sprint/pwml-recovery`. **`main` untouched — local `7531692`, remote `03f1af5`.**

### The one thing to read first

**A full, rigorous, independently-verified biological audit authorized a gold flag — and the flag
still had to be withdrawn.** Not because the biology was wrong. Because *"is this list exhaustive of
the paper?"* and *"is every unmatched row unsupported?"* are different questions, and only the first
was asked. **D-091 → D-092 in `DECISIONS.md` is the whole story and it is worth the ten minutes.**

### What ORCH-722 changed

| item | before | after |
|---|---|---|
| **F-175** | ESCALATED, "one tuple entry in frozen `driver.py`" | **CLOSED** under a narrow unfreeze — and the fix was *not* a tuple entry |
| **F-176** stale reason | ESCALATED | **CLOSED** under a narrow unfreeze |
| **F-176** runtime applicability | ESCALATED | **STILL OPEN.** Cannot be done narrowly — see below |
| **F-177** | new finding, no instrument | **INSTRUMENT BUILT**, evaluation-only |
| **F-178** | — | **NEW, found and fixed.** Two helpers named `_committed_legs` globbed the working tree |
| **`supported_reactions_complete`** | "needs an audit" | **audited, set, measured, WITHDRAWN.** No longer a curation problem |

### Three narrow product-owner unfreezes were granted; TWO were used, and the third was refused BY ME

**Used:** F-175 artifact persistence, and F-176's stale `inapplicable_reason`.

**NOT used — and stopping was the right call, confirmed by review:** making
`no_rejected_rag_reaction_reintroduced` *applicable at runtime*. It is a member of
`release_status.SEMANTIC_GATING_CHECKS`, so making it applicable **moves release status in both
directions** — a failure demotes, and an applicable pass can turn `not_evaluated` into `passed`. And
`quarantine_and_close` has **no `admission` parameter at all**, so it cannot be threaded narrowly
either. The authorization forbids anything that alters release status. **It needs a further ruling.**

### Priority 2 is no longer blocked on an audit. It is blocked on a PRODUCT DECISION.

The audit of `PMC12312563` is **certified and stands**: that paper states exactly one reaction and
gold's single signature is it. What killed the flag is that the scored corpus is **not seed-only**.
On the committed canonical leg `runs/2026-07-27_1623/…/strict`, setting the flag charges
`DHNA-CoA → DHNA` by MenI — whose four entities carry `rag_provenance.source_id = PMC8091085`, **a
paper `PMC12312563` cites in its own reference list**. Real reaction, real enzyme, right organism,
right pathway, source named. `goldset.py` already documents the flag as incompatible with
multi-paper RAG synthesis unless the run is seed-only.

**No further audit will unblock this.** One of these must hold first:

1. the scored corpus is **seed-only**; or
2. **`R-D092-1`** — row-level RAG lineage on `processes.reactions`. Lineage today is on *entities*
   only, so a scorer cannot exclude non-seed rows from the precision denominator from the row alone;
   or
3. **`R-D092-3`** — a product-owner ruling that a correctly-attributed cross-paper RAG reaction
   counts as unsupported for this benchmark. **A `policy_disagreement`, and the product owner's
   alone.**

**Also registered: `R-D092-2`** — duplicate reaction rows count as separate true positives, so
Priority 2 currently *rewards* a duplicated row.

### What DID ship for Priority 2, and it is a strict improvement

**The alias fix.** `PMC12312563`'s three `2-oxoglutarate` terms now carry
`alpha-ketoglutarate`, `alpha-ketoglutaric acid`, `2-ketoglutarate`, `2-oxoglutaric acid`,
`oxoglutaric acid`, `2-oxopentanedioate`. Measured on the committed canonical leg: **`ok=False` →
`ok=True`, recall 0/1 → 1/1**, with the flag OFF. Without them the paper's own MenD reaction failed
to match the signature written for it — a false failure that existed before this wave.

> **Never add bare `ketoglutarate` or `glutarate`.** `goldset.py` warns `α-` and `β-` must not
> collapse; every alias above pins either `alpha-` or `2-`.

### Verified state at this tip — measure, do not trust

| check | expected |
|---|---|
| Gold | `98739a59dd6c376f8a19968c7fa5dc3145be5b15` (was `36f4b7b6…`; D-091 briefly `d0b588a7…`, withdrawn) |
| SMOKE (merge gate 10) | **508 passed, exit 0** — `g11/ORCH-722/27`. Was 503; +5 is the F-176 tests |
| gold-readers split | **465 / 0 / 8 / 0** — `g11/ORCH-722/28`. Was 456; +9 = 5 F-175 + 4 D-092 |
| `streamlit_app.py` | sha256 `47e4fafa…`, **modified and never committed** |
| Python processes | exactly two `ms-python.isort … lsp_server.py` — **match on FULL COMMAND LINE** |

### Traps this wave paid for

1. **A test named `_committed_legs` that calls `rglob` is not measuring the committed corpus.**
   F-178, two files. An untracked benchmark run took the census 83 → 93 and three tests went red in
   the primary checkout while staying green in every worktree — **red for exactly the people who had
   run a benchmark.** Both helpers now ask git.
2. **`applicable` is not `passed`, in a third costume.** A payload where EVERY row matches a
   signature is a **measured** zero even with the flag off — no completeness claim is needed when
   nothing is unmatched. I assumed "flag off ⇒ withheld" and was wrong.
3. **A narrowness probe whose probe row is real chemistry for one of the papers tests nothing.**
   My first one used menaquinone chemistry against the menaquinone paper, it matched, and the case
   read "evaluable". Use a row foreign to every paper.
4. **Measure before you assert, especially when the assertion is convenient.** "Changes no
   acceptance verdict at all" was written without measuring, and rested on a run directory that is
   untracked. Review caught it; the measurement reversed the wave's headline result.

### THE NEXT WORK ORDER

**`prompts/PROMPT-001-eval-framework.md` is still the launcher.** Its items 3 (F-174) and 4 (F-172)
were done in `ORCH-721`; F-175, F-176's reporting half, F-177 and F-178 are done here.

1. **THREE PRODUCT-OWNER DECISIONS ARE WAITING**, and none is an engineering task:
   `R-D092-3` (cross-paper RAG rows vs Priority 2), the F-176 runtime-applicability ruling, and
   whether `R-D092-1` is chartered.
2. **`R-D092-1`** — row-level RAG lineage. The largest unblocker; it also makes RAG evaluation
   attributable per reaction rather than per entity.
3. **F-177 into `bench/acceptance.py`** — the instrument exists at
   `evidence/eval_semantic_populations.py` but acceptance's own tally still sums canonical and
   fallback. Evaluation-only, outside the freeze, chartered-ready.
4. **F-172's two residuals** — indirect pytest drivers (`chunk_d_gate.py`, the gold-readers split:
   44 unpinned pytest invocations per run) are NOT COVERED, and 3206 unpinned reports are a backlog.
5. **The AppTest environment-pinning card** — F-174's real fix. `tests/` is a gate surface.

### The evaluation stack — NOT started, and say so plainly

**No Phoenix, no OpenTelemetry, no Ragas, no deterministic scorer skeleton, and no unseen-cohort
work.** This wave spent its budget on measurement integrity and on one gold audit that ended in a
withdrawal. **Do not run the unseen ten-paper pilot** — capture does not exist yet.

Ready to reuse:
- `evidence/eval_semantic_populations.py` — read-only replay of any archived run, per-leg, every
  check as PASSED / FAILED / INAPPLICABLE / ARTIFACT_MISSING / ARTIFACT_MALFORMED, split by
  population, gold-blob stamped, and it counts a non-evaluated leg rather than skipping it.
- `evidence/d091_committed_effect.py` — read-only A/B of a gold-flag change across the committed
  corpus. Reusable for any future flag.
- `evidence/f176_admission_persistence_probe.py` — the applicable-vs-passed A/B.
- **1,947 rejected RAG candidates across 19 T-109 legs**, each with `gap_id`, claim, retrieval
  evidence (chunk id, section, score, span) and rejection reasons. **The RAG-evaluation dataset is
  already on disk.**

### Semantic populations at this tip (T-109 artifacts, read-only, gold `98739a59`)

| check | canonical (10) | fallback (9) |
|---|---|---|
| `requested_pathway_anchors_present` | 5 pass / 5 fail | 7 pass / 2 fail |
| `reaction_source_carrier_present` | 10 pass | 9 pass |
| `retained_reactions_match_supported_signatures` | 2 pass / **8 unevaluable** | **9 unevaluable** |
| `organism_compatible` | 10 pass | 9 pass |
| `no_real_id_or_name_conflict` | 7 pass / 3 fail | 3 pass / 6 fail |
| `no_rejected_rag_reaction_reintroduced` | **10 pass / 0 fail** | 7 pass / **2 fail** |
| `minimum_connected_core` | 9 pass / 1 fail | 9 pass |
| `placeholder_identities_distinguished` | 9 pass / 1 fail | 9 pass |

> **"Canonical failures were zero" is TRUE ONLY OF THE RAG CHECK.** Canonical carries **10** failures
> across four checks. Never quote a combined number without its denominator — that is F-177.
>
> **T-109 is not re-scored by any of this.** No acceptance verdict was produced; it remains
> `NOT ACCEPTED`, immutably.

### Protected — unchanged, and ORCH-722 committed none of it

`streamlit_app.py` · `data/enrichment_cache.json` · `data/id_mapping_cache.json` · `topics_*.txt` ·
`out/` · `outputs/` · `tmp/` · `runs_verify/` · the stray 0-byte `=` and `ValueError`.
**F-147 stays registered and deliberately UNCHARTERED — escalate only.**
**No worktree pruned.** `.claude/worktrees/orch721-f174` (detached at `cb982dc2`) is live evidence —
the base arm of every G9 proof in the last two waves. Leave it.

---

## 0-prevORCH721 — **SUPERSEDED by `ORCH-722`.** Its findings all STAND; superseded is its status as current — **`ORCH-721` closed: F-174 node 2 SOLVED, F-176 REFUTED, F-172 checker BUILT. Production still FROZEN.** 2026-09-03.

> **⚠ NOTHING IS RUNNING AND NOTHING IS CHARTERED.** Heavy lock free (`C:/t/heavylock` absent),
> zero sprint-owned Python, no unowned job. `D-090` still controls: **production is FROZEN and
> T-110 is NOT authorized.** No `src/` byte changed this wave and none may change without a ruling.

**Integration tip: verify it yourself. `git rev-parse HEAD` = `git rev-parse origin/sprint/pwml-recovery` = `git ls-remote origin sprint/pwml-recovery`.**
**`main` untouched — local `7531692`, remote `03f1af5`. Never write it.**
Everything below § 0 is older and superseded where it disagrees.

### What ORCH-721 settled, and the one thing it got wrong first

| finding | before | after |
|---|---|---|
| **F-174 node 2** | OPEN, *"the DB-config lever is already excluded"* | **EXPLAINED.** The lever is `.env`, and **the exclusion was wrong** |
| **F-176** | *"register it: `AdmissionReport.rejected` is not persisted"* | **REGISTERED AS REFUTED.** It is persisted, and the check works |
| **F-172** | not chartered | **checker BUILT**, enforcement opt-in, two residuals open |
| **F-175** | *"the writer never runs"* | **AMENDED.** The writer runs; the hand-off drops the file. **ESCALATED** |
| **F-177** | — | **NEW.** The semantic tally sums canonical and pre-quarantine payloads |

**F-174 node 2 — the method is the transferable part.** A worktree cut at the **identical SHA** is
**green** where the primary checkout is **red**; dropping only the primary's `.env` into that clean
worktree reproduces the exact assertion; the uncommitted `streamlit_app.py` modification does not.
Bisected to **two independently sufficient single keys, `LLM_PROVIDER` and `PATHBANK_DB_*`**.

> **That is why the old row said the DB was excluded and was wrong.** With two sufficient causes, a
> one-variable A/B reads red in both arms. **The DB was masked, not excluded.** Do not re-inherit it.

A second symptom — `AppTest script run timed out after 120.0(s)`, 151 s — was the **same cause**:
the job ran without `T2PW_OFFLINE_CURATOR=1` and made live curator calls, because only the primary
checkout has `.env`. With the flag set the same test fails in **5.37 s** with the registered
assertion. **A worktree is offline by accident, not by policy.**

**RESIDUAL, stated so it is not lost:** neutralising both keys in the primary leaves it **red**
(`g11/ORCH-721/27`, `/28`). **At least one further lever exists in the primary's untracked state
and is NOT identified.** The worktree A/B is the authoritative isolation.

### The correction this wave had to make to itself

**I wrote a rule from a remembered count and the count was wrong — twice.**

1. I designed `pin_verdict_refused` to be unconditionally fatal on *"0 of 583 verdicts say
   refused"*. The true count is **10** (7 with a sibling report), and **most are `H-010`/`REV-070`
   negative controls proving refusal works**. An unconditional rule would have failed the proof of
   the mechanism. The equivalence proof caught it and returned FAIL.
2. I then wrote *"129 of 583 … `T2PW_FROM_WRONG_TREE` is clean on all 129"*. True figures:
   **122 of 606**, and **three are not clean**. **The reviewer caught it by reading the pin files
   rather than the sentence.**

> Both are recorded verbatim in `check_pin_verdict`'s docstring and in F-172's amendment, on
> purpose. **A wave about unchecked universals that hid its own would be worthless.**

### Verified state at this tip — measure, do not trust

| check | expected |
|---|---|
| Gold | `git hash-object src/t2pw/bench/gold/pinned_v1.json` = `98739a59dd6c376f8a19968c7fa5dc3145be5b15` (**D-092**; `36f4b7b690b577f72882c3045ca6728d1ec8d9d1` before this wave. D-091 briefly moved it to `d0b588a79bb4aa3c11a7b5062a0b45bb8e20ab74` and was WITHDRAWN before merge — the flag is unset, the aliases ship) |
| SMOKE (merge gate 10) | **508 passed, exit 0**, survivors 0 — `g11/ORCH-722/17`, pin clean. **Was 503 through `ORCH-721`; +5 is the F-176 reporting tests, which live in `test_bench_goldset_and_semantic.py`, a SMOKE file. Merge rule 4, deliberate.** |
| `g11_evidence.py selftest` | **11/11** — `g11/ORCH-721/36` |
| G9 default-verdict equivalence | **5203 identical / 0 different** — `g11/ORCH-721/37` |
| This wave's own G11 | **37 reports, 0 non-compliant, under ALL FOUR strict flags**; 25/25 pytest jobs pinned |
| `streamlit_app.py` | sha256 `47e4fafa…`, **modified and never committed** |
| Python processes | exactly two `ms-python.isort … lsp_server.py` — **match on FULL COMMAND LINE, never count or PID** |

### THE NEXT WORK ORDER — unchanged in shape, re-ordered by what is now known

**`prompts/PROMPT-001-eval-framework.md` is still the launcher.** Its item 3 (F-174) and item 4
(F-172) are **done**; items 1, 2 and 5 stand. What ORCH-721 adds:

1. **`supported_reactions_complete` — still item 1, still untouched by this wave.** The only hard
   gate between a run and acceptance. **D-087 unchanged: one deliberately chosen case, after a
   genuine biological completeness audit, routed to `pwml-bio-auditor`. The audit is the cost, not
   the edit, and it is not a Lead judgement.** `goldset.py:384` warns that setting it without
   exhaustive signatures turns every unattributed row into a reported fabrication;
   `semantic.py:700` records that would have been **227** on a run that produced far fewer.
   **ORCH-721 did not attempt it and claims nothing about it.**

2. **THREE ESCALATIONS NEEDING A PRODUCT-OWNER RULING.** All three touch FROZEN files, all three
   are plausibly observability-only, and **none was argued into place by its author**:
   - **F-175** — add `COVERAGE_DIAGNOSTICS_FILENAME` to the carry tuple in `driver.py`. One tuple
     entry. **Cannot be proved without a real benchmark leg** — a unit test would repeat C-116's
     mistake exactly.
   - **F-176** — correct the stale `inapplicable_reason` at `semantic.py:1244`, which currently
     asserts something false and is propagated into the runtime release record.
   - **F-176 (second half)** — decide whether the runtime record should carry a
     `semantic_check_evaluability` claim it is structurally unable to make, since the artifact it
     reports as absent is written after it runs.

3. **F-177** — carry `payload_source` into the semantic tally. **Evaluation-only, outside the
   freeze, chartered-ready.** `acceptance.py:1370` already computes it.

4. **F-172's two residuals** — indirect pytest drivers (`chunk_d_gate.py`, the authoritative AppTest
   gate) are **NOT COVERED** by the new checker, and **3206** unpinned pytest reports are a standing
   backlog. Neither is closed and nothing pretends otherwise.

5. **A test-hygiene card, unchartered:** make the four AppTest boundary files **pin** the
   environment they read — `T2PW_OFFLINE_CURATOR`, `LLM_PROVIDER`, `PATHBANK_DB_*` — inside the
   fixture, so Chunk D means the same thing in every tree. This is F-174's real fix and it edits
   `tests/`, a gate surface.

### The evaluation stack — NOT started, and deliberately

**No Phoenix, no Ragas, no OpenTelemetry instrumentation, no deterministic evaluator skeleton, and
no unseen-cohort work was begun.** The wave spent its budget on measurement integrity, which was
the stated goal, and **the unseen pilot must not run before capture exists.** What ORCH-721 leaves
for it, ready to reuse:

- `evidence/f176_admission_persistence_probe.py` — **read-only** replay of any archived run
  directory, per-leg, reporting every semantic check as **PASSED / FAILED / UNEVALUABLE**, split by
  payload source. It is the first instrument in this project that distinguishes *applicable* from
  *passed*, and it is the interim answer to F-177.
- **1,947 rejected RAG candidates across 19 T-109 legs**, each with `gap_id`, claim, retrieval
  `evidence` (chunk id, section, score, span) and rejection `reasons` — the RAG-evaluation dataset
  is **already on disk** and does not need to be regenerated.

### Semantic verdicts on T-109, since applicability was previously reported as if it were passing

**ARM A (artifacts as they are), 19 gold legs, SPLIT because summing them is F-177:**

| check | canonical (10) | fallback (9) |
|---|---|---|
| `requested_pathway_anchors_present` | 5 pass / 5 fail | 7 pass / 2 fail |
| `reaction_source_carrier_present` | 10 pass | 9 pass |
| `retained_reactions_match_supported_signatures` | 2 pass / **8 unevaluable** | **9 unevaluable** |
| `organism_compatible` | 10 pass | 9 pass |
| `no_real_id_or_name_conflict` | 7 pass / 3 fail | 3 pass / 6 fail |
| `no_rejected_rag_reaction_reintroduced` | **10 pass / 0 fail** | 7 pass / **2 fail** |
| `minimum_connected_core` | 9 pass / 1 fail | 9 pass |
| `placeholder_identities_distinguished` | 9 pass / 1 fail | 9 pass |

> **`applicable` is not `passed`.** The earlier *"four of five biology checks pass"* summary was
> read off an `applicable` column in the **runtime** release record — which is a different
> evaluator from the offline scorer, and which reported the RAG check inapplicable **on legs whose
> own directories contain the artifact it said was missing.**
>
> **`T-109 is NOT re-scored by any of this.** No acceptance verdict was produced and its
> disposition is unchanged: `NOT ACCEPTED`, for the Priority-2 reason, immutably.

### Protected — unchanged, and ORCH-721 committed none of it

`streamlit_app.py` · `data/enrichment_cache.json` · `data/id_mapping_cache.json` · `topics_*.txt` ·
`out/` · `outputs/` · `tmp/` · `runs_verify/` · the stray 0-byte `=` and `ValueError`.
**F-147 stays registered and deliberately UNCHARTERED — escalate only.**
**No worktree was pruned.** One was **added**: `.claude/worktrees/orch721-f174`, detached at
`cb982dc2`, and it is **live evidence** — it is the green arm of the F-174 A/B. Leave it.

---

## 0-prevD090 — **SUPERSEDED by `ORCH-721`, 2026-09-03.** Current as of the D-090 close; its rulings all STAND, and what is superseded is its status as current plus two claims ORCH-721 refuted (F-174 node 2 "OPEN", and the resolution-DB "excluded") — **SPRINT CLOSED. `D-090`: engineering-complete, production FROZEN, T-110 NOT authorized.** Next phase is the evaluation framework. 2026-09-03.

> **⚠ NOTHING IS RUNNING AND NOTHING IS CHARTERED HERE.** T-109 exited, was scored once, and is
> **CLOSED and IMMUTABLE**. Heavy lock free, zero sprint-owned Python, no unowned job.
>
> **`D-090`: the recovery pipeline is ENGINEERING-COMPLETE and production is FROZEN. T-110 is NOT
> authorized.** **The next phase is the RAG / LLM EVALUATION FRAMEWORK — its launcher is
> [`prompts/PROMPT-001-eval-framework.md`](prompts/PROMPT-001-eval-framework.md). Paste it into a
> fresh session.** `HANDOFF.md` is the full briefing; `T109-RESULT.md` is the verdict.
>
> **T-109: OPERATIONALLY SUCCESSFUL, formally `NOT ACCEPTED`, because Priority 2's test dataset was
> not evaluable.** P1 `ok=true` raw **0** · P2 `ok=null` **NOT EVALUATED** on 13 of 19 legs · P3
> `ok=true` **0** · P4 `0/8` · P5 `0/2`. **Priority 2 did not FAIL — it could not be EVALUATED**
> (D-087, unchanged: `supported_reactions_complete` unset on all ten cases, both
> `max_retained_reactions` ceilings on negative controls). **20/20 legs, timeouts 3 -> 1 -> 0,
> 4.95 h, survivors 0 — the best-executing run of the sprint.**
>
> **THE RULE MOST LIKELY TO BE BROKEN BY ACCIDENT:** *no production behaviour changes solely to
> satisfy the incomplete test instrument.* A `src/` change justified by *"it would make Priority 2
> evaluable"* is a **reject**. **The instrument is what is incomplete.**
>
> **D-088's two required consequences BOTH held on a fresh draw.** `PMC12096016/strict` capped by
> **`EntD` alone** (`ATP`/`NADH`/`Fur` all matched); `PMC12782028/strict` by the genuinely absent
> mevalonate arm — **`HMGCR`, `HMGCS1`, named in its own artifacts.**
>
> **Priority 1 fell 8 -> 0 and that is NOT a fix.** No production code changed between the two runs.
> Evidence about the draw until a second run reproduces it.

**Integration tip `1a117eaa`, pushed and remotely verified: local = `origin/` = `git ls-remote`.
`main` untouched — local `7531692`, remote `03f1af5`.** Everything below § 0 is older and superseded
where it disagrees.

> **⚠ IF YOU ARE PICKING THIS UP MID-RUN, READ `T109-RUN-OWNERSHIP.md` FIRST.** A live 20-leg
> benchmark may still be in flight. **Do not launch anything heavy, do not clear `C:/t/heavylock`,
> and do not start a second wrapper.** The lock names its holder and PID; the ownership file names
> the wrapper task, the monitor task and the whole process chain.

### The ruling that unblocked everything — `D-089`

**D-088 clause 10 controls for this release. The INCOMPLETE-CORE CAP is unchanged.** No cofactor
vocabulary, no entity-list match, no Stage-0 redesign, no gold change, no curated expectations inside
production. `PMC12096016/strict` stays `review_required` with its pathway **preserved for review**,
and **Priority 5 stays `0/2`**.

> **Recorded as an EXPLICITLY ACCEPTED CONSERVATIVE LIMITATION — never as delivery of D-088 clause
> 2.** A report that describes the cap's survival as "D-088 implemented" is wrong. `D-089` § 3 and
> **F-173** exist so that error is catchable by reading rather than by re-deriving.

**The product principle is reaffirmed, not withdrawn.** What is deferred is its implementation:
**`R-D089-1`**, a stable, general, **non-paper-keyed** reaction/subprocess completeness specification
typing participants as *defining* / *optional* / *extracted-but-unwired* / *genuinely absent*,
registered for the **RAG / LLM evaluation phase**. Not this wave, by ruling.

### T-109 — the milestone identity, and why it is not a second T-108

The ruling says *"launch T-108 once."* **T-108 has already been launched once**, is scored
`NOT ACCEPTED`, and the same ruling says *"Do not rerun T-108."* `T108-READINESS.md` § 7.1 requires a
new identity and a separately recorded readiness decision. **`T-109` is the only reading under which
all three hold.** T-108 is untouched and `T108-RESULT.md` is not edited.

| | |
|---|---|
| Run directory | **`runs_verify/2026-09-02_2052`** |
| Launched | **2026-09-03T02:54:33Z**, continuing the verified directory **without `--fresh`** |
| Ceiling | 3600 s per leg, **no override**; 72000 s wrapper; 18 h internal deadline — **T-108's, unchanged, for comparability** |
| Expected | **~6.4 h**, from T-108's measured 22929.17 s on the same corpus |
| Owner | this session, sole owner — `T109-RUN-OWNERSHIP.md` |

**Score it EXACTLY ONCE, with `evidence/t108_score.py <repo> <run-dir>` (generic despite the name —
do not rename it, every prior report was produced by those bytes). Report hard-gate acceptance
SEPARATELY from diagnostic Priorities 4 and 5. Do not call it accepted if any actual hard gate
fails. If it hard-fails, triage from the immutable artifacts — DO NOT RERUN IT.**

### The gates, measured today at `0859fba9`/`a844443f` and not inherited

| gate | state |
|---|---|
| **SMOKE** (merge gate 10) | **503 passed, exit 0**, survivors 0, pin verdict `refused=false, violations=[], foreign_src=[]` — `ORCH-720/01` |
| **gold-readers** | **456 / 0 / 8 / 0**, exit 0 — `ORCH-720/02` |
| **battery + F-146** | **`battery=0/29  F146=REJECTED  C1..C6=0`** — `ORCH-720/03` |
| **mutation harness** | **17 mutations, SURVIVORS 0** — `ORCH-720/05` |
| **Chunk D** | **RED in the primary checkout** — `run-core 159/160`, `node15 0/1`. **F-174.** See below |
| `acceptance.py` | CRLF sha256 `4bd893ac…` · LF sha256 `d9f817e1…` · git blob `56aa593e…`. **Two of those three are sha256 and one is not; they have been conflated before** |
| gold | `36f4b7b690b577f72882c3045ca6728d1ec8d9d1`, clean in working tree and HEAD — **superseded by D-091, now `d0b588a79bb4aa3c11a7b5062a0b45bb8e20ab74`. This row records the ORCH-720 measurement and is left as that wave's record.** |
| `ms-python.isort` processes | **TWO.** It has been 2 and 3 in this sprint — **match on COMMAND LINE, never count or PID** |

### The two findings this wave, and they are the same shape as F-171 and F-172

**F-173 — `PMC12096016/strict`'s `review_required` is a KNOWN FALSE NEGATIVE with a KNOWN SIGN.**
Half of Priority 5's strict denominator is known-misclassified in a known direction. **The metric is
safe to ship and dangerous to quote bare.**

**F-174 — the authoritative Chunk D gate has NEVER been run in the primary checkout.** Both committed
green runs had `cwd` inside a **worktree**, read out of their own reports. **It cannot be a code
regression:** the only commit since the last green Chunk D touched three evidence artifacts and no
`src/`, `tests/` or `scripts/`. Node 1's mechanism is **proven** (red with the resolution DB
configured, green with it deconfigured, tree and commit held fixed); **node 2's lever is NOT isolated
and is registered OPEN.** Not a readiness row — `TEST_MATRIX:244` excludes Chunk D from the smoke
gate — and T-108 ran in this same checkout.

> **Four times now this sprint a green signal has meant less than its readers believed** — F-171,
> F-172, F-173, F-174. **`187/187` was true of the worktrees it was measured in and was never true of
> the primary checkout, and nobody had asked which one it meant.**

### Traps that cost real time, in this session and the last

1. **`grep -E "^OPENROUTER_API_KEY=" .env` finds NOTHING.** The live key is written `KEY = value`
   **with spaces**, so the only line a `^KEY=` grep can match is the commented-out one above it, and
   the check reports the key ABSENT while `python-dotenv` resolves it fine. **This briefly looked
   like a hard readiness failure.** Verify configuration **through the loader**, or do not claim to
   have verified it — `evidence/t109_preflight_provider.py` is committed for exactly this.
2. **The `.pin.json` verdict goes in `evidence/g11/pin/<TASK>/`, not in the task directory.** Put it
   in the task directory and `g11_evidence.py check` reads it as a malformed cleanup report and
   fails the whole task.
3. **A foreground bounded run that may wait on a lock is killed by the tool's 120 s cap** — not 600 —
   **while holding the lock.** Background it and branch on **exit 95**.
4. **Four AppTest files stall a one-process pytest silently for 40 minutes.** Chunk D's authoritative
   gate is the split-process runner. **The failure mode is silence, not error.**
5. **Pre-create every `--basetemp` parent.** A missing parent produces `1 error in 0.18s`, which
   looks exactly like a test result and is not.
6. **Before building an instrument to separate two hypotheses, check whether one is already excluded
   by something already written down.** I wrote a 140-line A/B probe for a question
   `git diff --name-only` answered in one line.

### Protected, unchanged

**F-147 registered and deliberately UNCHARTERED.** `placeholder_backed_proteins` — escalate only.
**T-107 and T-108 immutable.** `main` untouched. `streamlit_app.py` never committed. Gold
**`36f4b7b690b577f72882c3045ca6728d1ec8d9d1`**. Never commit caches, `topics_*.txt`, the stray 0-byte
`ValueError` and `=`, `out/`, `outputs/`, `tmp/`. **`HANDOFF.md` § 7 forbids pruning a worktree.**

---

## 0-prev1. `ORCH-719` — the D-088 correction wave, 2026-09-02. **SUPERSEDED by § 0 above.** Its C-115/C-116/C-117 merges, its C-114 ruling and its two operational lessons all STAND; what is superseded is its status as current and its statement that the product question is still open — it was ruled as `D-089`.

**Integration tip `1e6415ec`, pushed and remotely verified: local = `origin/` = `git ls-remote`.
`main` untouched — local `7531692`, remote `03f1af5`.** Everything below § 0 is older and superseded
where it disagrees.

### The state of the gates, measured today and not inherited

| gate | state |
|---|---|
| **SMOKE, on the MERGED integration tip** | **503 passed, exit 0**, zero survivors — G11 `ORCH-719/15`. Four independent 503s today |
| **Chunk D**, authoritative split-process gate | **`jobs=28 executed=187/187 omissions=0 additions=0 failed=none`** — G11 `ORCH-719/14` |
| heavy lock | free; **verify before claiming** |
| `ms-python.isort` processes | **count is now TWO**, was three this morning. **It has changed twice in this sprint — match on COMMAND LINE, never on count or PID** |

> **A machine crash occurred mid-wave and cost nothing** — no stranded lock, no orphaned process,
> every commit already pushed. The one casualty was a running reviewer, restarted.

### Merged this wave

**C-115 (`9d106fa7`) — the five c102 census pins moved to the measured corpus.** Approved by
`REV-115` on the actual diff; all eleven merge rules pass. **This closed F-171 and made merge rule 10
satisfiable for every other card in the sprint** — SMOKE had been `3 failed / 500 passed` on the
integration branch itself since `479128b3`, and the previous handoff certified it green.

### Ruled this wave

**`RULING-C114-DISPOSITION.md` — the D-088 diagnostics leave the coverage verdict.** C-114 is **not
merged and not discarded**; its measurement work is reused, its shape is superseded.

**The ruling rests on one fact, not on a count:** `test_c074_strict_core_floor.py:462` fired because
the verdict acquired **request-derived content outside `requested_context`**, the one key set aside
for it. That is a change in what the document *means*. Amending the test would permanently silence
the only thing that noticed.

**It was measured before it was ruled.** Widened sweep of 34 deterministic consumer files
(G11 `/12`) plus the authoritative Chunk D gate (G11 `/14`) prove the enumeration **complete**:
**C-114's collision set is exactly `test_c011_freeze_seam_golden_equivalence` (3) and
`test_c074_strict_core_floor` (2). There is no third.** Two further sweep failures reproduce at base
(G11 `/13`) and are pre-existing.

**The ruling REVERSES** if D-088 clause 9 ever makes these fields an **input** to a release decision —
an input belongs in the object the gate receives. On today's facts F-168 forbids that.

### In flight

**C-116** (`prompts/C-116.md`, branch `agent/c116-d088-diagnostics-artifact`, worktree
`.claude/worktrees/c116-d088-artifact`, base `a8065403`). Diagnostics become their own artifact; the
verdict must be **byte-identical** to base. **Needs an independent review before merge.**

### The open product-owner question — still blocking steps 8, 9 and 10

**No permissible reaction-level replacement for the runtime cap exists this wave.** Four candidates,
each rejected for an independent documented reason; and the acceptance instrument **cannot** route
around it, because Priority 5's numerator requires `strict_acceptance_eligible`, which is
`status == RELEASE_READY` (`release_status.py:1261`), and `acceptance.py:1146` refuses to reclassify
a frozen record under merge rule 8.

> **Does D-088 clause 2 yield to clause 10 (cap unchanged, `Priority 5` stays `0/2`), or clause 10 to
> clause 2 (cap relaxed on a cofactor vocabulary, `PMC12782028` released at runtime)?**

**Lead recommendation: the first. It means the headline number does not move.**

### Next actions, in order

1. **`REV-116`** on the actual diff. **The one thing to check hardest:** the coverage verdict's bytes,
   and that `test_c011` (8) and `test_c074` (31) are green **because the verdict is unchanged**, not
   because anything was amended.
2. **Merge C-116** if approved, then SMOKE **at the merge commit**, not the branch tip.
3. **Then** rebuild T-108 readiness (step 8), every row re-derived.
4. **Step 9 stays NO-GO** until the instrument is merged, gated and remotely verified.

### Two operational lessons that cost real time today — both are mechanism, not care

1. **A foreground bounded run that may wait on a lock will be killed by the tool's 120 s cap** — not
   600 — and it dies **holding the lock**. Put any such job in **tracked background**, and branch on
   **exit 95** rather than pre-checking the lock, because pre-checking races any peer agent.
2. **Four AppTest files stall a one-process pytest silently for 40 minutes:**
   `test_batch_preflight`, `test_c055_rag_loop_wiring`, `test_streamlit_quarantine_boundary`,
   `test_c052_prefreeze_report_at_the_streamlit_seams`. `TEST_MATRIX` says Chunk D's authoritative
   gate is the split-process runner *"never the one-process form"*. **The failure mode is silence,
   not error.**

**Both hazards were already written down before I hit them.** That is the lesson worth carrying: a
hazard you have recorded is not a hazard you have controlled.

### Protected, unchanged

**F-147 registered and deliberately UNCHARTERED.** `placeholder_backed_proteins` — escalate only.
**T-107 and T-108 immutable.** `main` untouched. `streamlit_app.py` never committed. Gold blob
**`36f4b7b690b577f72882c3045ca6728d1ec8d9d1`**. Never commit caches, `topics_*.txt`, the stray 0-byte
`ValueError` and `=`, `out/`, `outputs/`, `tmp/`.

**`HANDOFF.md` § 7 forbids pruning a worktree.** A stranded 485 MB read-only worktree at
`C:\t\rev114\basetree` is **flagged, not removed**, and ~180 worktrees exist sprint-wide. 694 GB free;
untidy, not urgent.

---

## 0-prev0. The T-108 execution wave, 2026-09-02 — **SUPERSEDED by § 0 above, which is the same day and later.** Its T-108 verdict, its D-088 summary and its three "things a successor most needs to know" all STAND; what is superseded is its status as current.

**Integration `479128b3`+, pushed and remotely verified. `main` untouched: local `7531692`, remote
`03f1af5`.** Everything below this section is older and superseded where it disagrees.

**T-108 ran ONCE into `runs_verify/2026-09-01_1612`, 20/20 legs, 6.37 h, and is scored, triaged and
committed. Its verdict is `NOT ACCEPTED`. Do not re-run it, do not re-score it, do not reinterpret
it.** Full result: **`T108-RESULT.md`**. Run ownership record: **`T108-RUN-OWNERSHIP.md`**.

**The recovery sprint's release-candidate question is answered for this candidate. T-108 is preserved
as a failed official release candidate.** A later candidate needs a **new milestone identity** and a
separately recorded readiness decision.

### The verdict in one table

| # | Priority | T-108 | `ok` |
|---|---|---|---|
| 1 | zero known false real identifiers | raw **2** · accepted **2** · `accepted_status: PASS` (target 6) | **false** |
| 2 | zero unsupported retained reactions | **`NOT EVALUATED`** — verdict never reached on 12 of 19 scored legs, 8 papers | `null` |
| 3 | zero referential-integrity violations | **0** | **true** |
| 4 | meaningful requested-pathway coverage | **0/8** | **false** |
| 5 | strict PWML pass rate among eligible papers | **0/2** | **false** |

### D-088 — the ruling that decides the next wave, and T-108's NO-GO

**Recorded as documentation only. NOT implemented. T-108 is NOT launched.**

> **The pipeline's primary goal is to recover the paper's important pathway reactions as correctly as
> possible. It is not required to achieve perfect participant-level biochemical completeness.**

**Hard completeness decisions move to validated reactions and major subprocesses.** Flat Stage-0
`key_compounds` / `key_proteins` stop being automatic hard release requirements; missing ordinary
cofactors, currency metabolites, regulators, ancillary proteins, water and protons become
**warnings or secondary-score deductions**, not automatic removal of `release_ready`. **This
supersedes the assumption that every requested-core entity must match an admitted process for
release.**

**It does NOT loosen anything biological.** A participant stays important when it is a defining
substrate or product, distinguishes the reaction's identity or direction, or is central to the
paper's scope; **missing a whole named branch or subprocess stays a genuine reaction-recall
failure**; an extracted entity does **not** satisfy coverage merely by existing in the payload; and
**no gold-forbidden content may become releasable because entity anchors were downgraded.**

**Clause 10 is the one that will be tested:** *do not simply filter cofactors, match against the
entity list, or relax the cap without replacing it with reaction-level coverage.* Each of those
moves Priority 5 off zero immediately and **hides genuine failures.**

**T-108 is NO-GO** until the D-088 card is reviewed, merged, gated and remotely verified —
`HANDOFF.md` § 5.2a step 9. The ten-step work order lives there.

### The three things a successor most needs to know

**1. Priority 5 is `0/2` in both T-107 and T-108 and the two zeros mean completely different
things.** T-107's was one operational loss (a timeout) plus one coverage shortfall. **T-108's is two
coverage shortfalls and zero operational losses.** Both `strict_exportable` legs executed fully,
cleared the strict technical gates, **passed semantic evaluation**, produced valid PWML, and are held
at `review_required` for incomplete requested-core coverage (completeness **0.75** and **0.538**).
**That is merge rule 7 working as written.** The number did not move; **the denominator became
honest.** A runner `pass` is not a Priority-5 point.

**2. The 3600 s restoration worked, and did not solve everything — F-166.** Timeouts **3 → 1**,
scorable denominator **17 → 19**. `PMC12096016/strict` — a core `strict_exportable` paper — went from
TIMEOUT/0 files to **PASS with a 74367-byte PWML, 0 gate errors**, needing only **152.9 s** beyond
T-107's ceiling. But `PMC12444477/strict` consumed the **full** 3600 s and still timed out, so the
census maximum of 3421.4 s **was not an upper bound**. Per § 2.1's own ruling that is **not
automatically a defect and must not be waved away either**. **No ceiling change is proposed on one
observation** — that would be choosing a budget from censored data a second time.

**3. F-165 — never compare a Priority-1 count across milestones without checking the gold blob.**
C-113 merged **three days after T-107 ran** and added the `delta`/`δ` spellings to `PMC12180156`'s
forbidden aliases. **One of T-108's two Priority-1 rows is that exact spelling** — invisible under
T-107's gold. So T-107's and T-108's Priority-1 numbers were **taken with different instruments**,
and the two facts pull in opposite directions: the instrument got **stricter** and the count still
**fell**. Do not fuse that into one improvement claim.

### What is NOT claimed, and must not be quietly upgraded

- **F-146 is NOT fixed.** `PMC13231680/research` produced an empty pathway where T-107 passed. That
  is **one draw**. The standing trap forbids calling a single leg a regression at temperature 0; **the
  symmetric rule binds and forbids calling it an improvement.** The artifacts also cannot separate
  *"declined"* from *"this draw extracted nothing"* — zero reactions is not a recorded refusal.
- **`LpxH` remains UNVERIFIED.** `PMC12444477/strict` timed out again; the research leg carries **0
  findings**. Verified only on `runs/2026-08-02_2130`.
- **Priority 2 = 0 counted is the absence of a measurement**, reported as an acceptance-instrument
  limitation under **D-087** clause 6. It is not a measure of invented chemistry.
- **`PMC12856317/strict` `PASS → FAIL` is NOT a regression.** T-107's export held only `ALAS1`/`ALAS2`
  — **no ClpXP** — so the gate had nothing to fire on. T-108's draw extracted ClpXP without an
  accession and the § 8 identity gate refused it. **The gate did not change; its input did.**
- **`PMC12452463/strict` blocking issues went 7 → 3 → 6 across T-106/107/108.** This **retires the
  previous wave's "improved at T-107, 7 to 3"** as draw variance.

### No code change is chartered from T-108

The only genuine `product_contract_violation`s are **F-147** (`PMC12180156/strict` +
`PMC12452463/strict`, one shared seam), which is **registered and deliberately UNCHARTERED** because
a downstream-only fix would export gold-forbidden content. **Merge rule 6.**

### The product-owner ruling recorded this wave — D-087

**`supported_reactions_complete` stays unset by default.** It may be set only on a case with an
explicitly bounded, exhaustive reaction scope, certified by an **independent biological reviewer**;
**several supported reactions are not evidence of completeness**; a missing assertion stays
`NOT EVALUATED` rather than becoming a confident accusation of invented chemistry; and **if no case
meets the standard, all ten unset is correct and is reported as an acceptance-instrument
limitation.** **Recorded, deliberately NOT implemented — the gated tree is untouched.**

### Findings registered this wave

**F-165** — T-107/T-108 Priority-1 counts measured against different gold sets; a benchmark number is
a reading and a reading has an instrument. **F-166** — one leg needs more than 3600 s; the ceiling
restoration was right *and* insufficient for that leg, and both halves must travel together.
**F-167** — the requested-core anchors are Stage-0's `key_*` lists, and the incomplete-core cap makes
one unmatched anchor enough to remove `release_ready`. **Resolved by D-088.**

**F-167 carries an AMENDMENT that refutes its own strongest claim, and both measurements are
preserved.** It reported **0 of 10** unmatched anchors appearing in Stage-0's subprocess list — **valid
for the two Priority-5 denominator legs it sampled, and INCORRECTLY GENERALISED to the corpus.** A
census over all **83** committed legs measured **60 of 374 (16%) that DO appear**, **314 of 374 that
do not**, and **90 of 374 (24%) present in payloads but unwired**. **88% of all committed legs carry
at least one unmatched anchor; only 10 of 83 ever fully matched.** The 16% is the population D-088
clause 5 keeps as a genuine reaction-recall failure, and the original framing would have justified
exactly the shortcut clause 10 forbids.

### Run hygiene, verified at close

`FINAL SURVIVING COUNT : 0` · `cleanup : success` · heavy lock **released**, `C:/t/heavylock` absent ·
**zero sprint-owned Python**, matched on **command line** · G11 `check --task T-108` **0
non-compliant** · gold `36f4b7b6`, `acceptance.py` `4bd893ac…` and `streamlit_app.py` `47e4fafa…` all
**unchanged before and after** · **no gold or scorer change after seeing the result.**

**One honest deviation:** the expected IDE baseline is **two** `ms-python.isort` processes; **three**
are present after the run, one under system `c:\python313\python.exe`. All match on command line,
none is a sprint job, none is a cleanup target. Recorded because the baseline said two.

---

## 0-prev. `ORCH-718` closed, 2026-09-01 — **SUPERSEDED by § 0 above**

**Integration `8f696945`, pushed and remotely verified. `main` untouched: local `7531692`,
remote `03f1af5`.** Everything below § 0 is older; § 0-prev is the previous wave and is superseded.

**Three cards merged, each independently reviewed against criteria fixed before its diff existed,
each gated at its own merged tip, then all three gated together.**

| Card | Merge | Reviewer verdict |
|---|---|---|
| **C-113** F-150 half 1 + census re-pin | `db119f53` | REV-113 **APPROVE w/ residuals** |
| **C-111** F-148 timeout observability | `2a0ccdbd` | REV-111 **APPROVE w/ residuals** |
| **C-112** residual sweep | `c942f774` | REV-112 **APPROVE w/ residuals** |

**Gates at the combined tip:** SMOKE **503 / exit 0** · gold-readers **456 / 0 / 8 / 0 / exit 0**
against gold `36f4b7b6` · `acceptance.py` `4bd893ac410d16d3…` unchanged · **`battery=0/29`,
`F146=REJECTED`, C1–C6 all 0** · whole-tree G11 **5032 artifacts, 0 non-compliant**.

**The one thing to read before anything else: F-150 half 1 was merged, GATED RED, and REVERTED
before it landed for real.** SMOKE came back **501/2** at `b05a7281`; merge rule 10 required the
merge not to stand; integration was re-proved green at **503**; and the edit re-landed at C-113
**with the census movement it causes**, measured and attributed per leg. **The gold edit was never
wrong. Landing it without its full footprint was.**

**T-108 is NO-GO on exactly one row — row 19, run ownership.** Eighteen of nineteen are green.
See `T108-READINESS.md` § 5, which is rebuilt at the current tip and tells you what to re-derive.

**Open product-owner question, preserved unanswered:** should `supported_reactions_complete` be set
on any gold case? `DECISION-PACKET-F150-HALF2.md`. **It is NOT a T-108 blocker.**

**New findings this wave: F-161** (neither gate selection is a superset of the other — a gold edit
needs BOTH; **ratified as a standing obligation**), **F-162** (a mistyped task id returned *another
task's* evidence, not nothing), **F-163** (`HeavyLock.release` is non-atomic and can create a lock
nobody may clear), **F-164** (C-112's recursion fix opened a false FAIL via the allocator's
`.staging`).

**Two tooling repairs are chartered and deliberately NOT taken:** F-163's `bounded_run.py` and
F-164's `reviewer_evidence_route.py`. Both are instruments this wave's own certifications were
produced through — `bounded_run.py`'s build hash is recorded in **every** G11 report — so changing
either mid-wave breaks comparability, and changing a just-reviewed instrument without a new review
is the move this sprint refuses.

---

## 0-prev2. ORCH-717 continuation — **SUPERSEDED**

**Read this section first. It is newer than everything below it**, and sections 5 and 6 are partly
superseded: four more cards were chartered, one is merged, and two of the three held questions are
ruled.

**This is the THIRD Lead Orchestrator on this branch inside about an hour** (`-b1`, `-ab`, now this
one). Both predecessors vanished mid-wave without writing a handoff. **That is why this section is
checkpointed continuously rather than at a context threshold.**

### Card state

| Card | Scope | Worktree | State |
|---|---|---|---|
| **C-109** | F-153 remainder, F-154, reviewer-evidence route | `C:/t/c109`, `C:/t/rev109`, `C:/t/rev109base` | **MERGED `efb2edc2`, gates pinned `887395dc`** |
| **C-108** | F-155, all five members | `C:/t/c108`, `C:/t/c108base`, `C:/t/rev108`, `C:/t/rev108base` | **REV-108 REJECTED · correction round 1 dispatched** |
| **C-110** | Q1 negative-control status | `C:/t/c110` | implementer running, first commit landed |
| **C-111** | F-148 observability | — | chartered, **not dispatched** |
| **C-112** | residual sweep (incl. drift C-109 created) | — | chartered, **not dispatched. Item 5 requires C-108 merged first** |
| **F-150** | Wave 4 gold correction | — | `REV-F150.md` criteria fixed; **no reviewer dispatched; gold still unmodified** |

**Review criteria for every card were fixed and committed BEFORE their diffs existed** — `REV-108`,
`REV-109`, `REV-110`, `REV-111`, `REV-F150`.

**Reviewer evidence preserved as refs even before merge:** `refs/remotes/rev109/evidence` (merged in),
`refs/remotes/rev108/evidence` (51 files, held pending C-108's correction round).

### C-108 — where it actually stands

**Everything except one thing passed, and the reviewer re-measured all of it.** Battery tip
`0/29 F146=REJECTED`, `C5: 1 → 0`; corpus 692 rows, drift 0, **19 newly REFUSED / 2 newly ADMITTED**,
never netted, all 19 defensible and the 1 admitted row correct; G9 base red **53 failed / 120
passed**; M1–M15 + M6b RED; SMOKE 503 with `acceptance.py` byte-identical.

**The blocking finding, which I verified myself at 4 of 4 before spending a round** —
`evidence/orch717_rev108_blocking_verify.py`:

```
base  blocking_admitted=0  appositive_refused=3  pinned_leaked=0
tip   blocking_admitted=4  appositive_refused=0  pinned_leaked=0
```

Member (d) genuinely works; **the contra genuinely weakened.** Four spans where the actor IS the
thing being shut down go REFUSE → ACCEPT. **Merge rule 6.** The pinned C5 case is clean at both SHAs,
which is exactly why the battery did not catch it.

**Cause, in one line: F3/F4 are a bounded closed list of target-directed frames with ACCEPT as the
default outside them** — handoff lesson 3, which the card quotes and the diff's own comment quotes.
The list is not wrong; **its polarity is.** The reviewer's proposed inversion (fire on the agent noun
by default, exempt only appositive frames) was passed to the author as a **hypothesis to measure, not
a design instruction.**

**My first verification probe was WRONG and is preserved** — it built a `reactions` envelope and
addressed `/reactions/0/...`, which does not match the actor-role path pattern, so the guard was
never reached and everything came back ACCEPTed at base. **Two independent records contradicting it
are what exposed it.** Mirror `c107_battery.py`'s `run()` exactly: `processes` envelope,
`/processes/<bucket>/0/<container>/-`, `stage="probe"`, verdict from `summary.accepted_count`.

### Rulings — `RULINGS-ORCH717.md`

- **Q1 RULED (product owner)** → `PASS_NEGATIVE_CONTROL`, implemented by **C-110**. The predicate
  already exists: **`_empty_is_correct` at `acceptance.py:1530`**. The misleading token is emitted at
  **`batch/runner.py:717`, which has no `GoldCase` in scope at all** — C-110 carries a stop condition
  forbidding it to give the runner gold access.
- **Q3 RULED (Lead): a `pathbank_compound_id` is NOT an accession. No code change.** It could never
  have moved Q2's arithmetic — the affected row carries **five** recognised accessions without it,
  and the Priority-1 branch only asks `if ids:`.
- **Q2 half 1 unblocked**, goes to Wave 4. **Half 2 (`supported_reactions_complete`) is the ONE open
  product-owner question.**

### Findings registered this wave

**F-156** — `MASTER_PLAN` § 2's third claim is false too; the graph-delta enforcement is implemented,
wired, reached in production **and load-bearing**, proved by mutation. Refuted on the code by peer
session `project14-t2pw-93`, **certified behaviourally by me** — the provenance is split on purpose.
**F-157** — a citation pinned to bytes that exist in no commit: `streamlit_app.py:5669` was read off
the **uncommitted** working copy of the never-commit file; committed value is **`:5636`**, the `+33`
being exactly the protected diff. It propagated F-153 → `MASTER_PLAN` → my own charter. **Closed by C-112**: F-153 cites the symbol now, and the standing rule is `TEST_MATRIX.md` § *Never cite a line number in a file that carries an uncommitted diff*.

### T-108 — `NO-GO`. See `T108-READINESS.md`

**The blocker most likely to be missed is operational, not code:** the ceiling was halved
`3600 → 1800` with `leg_timeout_override_reason` **empty**, the slowest finishing leg used **92.1%**,
and `PMC12096016` — one of only two strict-denominator papers — was **lost to the clock, not
biology**. **Choose the ceiling deliberately and record why BEFORE launch.** T-108 runs from the
**primary checkout**: `.env` is untracked, so a worktree silently gets `LLM_PROVIDER=local` and the
curator becomes a no-op **by accident**.

### Peer

`project14-t2pw-93` closed out read-only with **no claims**. It found F-153 and F-156 by reading the
map in order to use it — twice in one wave.

---

## 1. Integration state

| | |
|---|---|
| Branch | `sprint/pwml-recovery` |
| **Do not pin a tip SHA here** | the invariant is **local = `origin/` = `git ls-remote`**. Read it, do not recall it |
| Merges to `main` | **none, and none permitted.** `main` local `7531692`, remote `03f1af5` — it advanced **outside** this sprint. **Touch neither ref** |
| Product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` — verified intact after every commit this wave |
| Caches, `topics_*.txt`, `cache_snapshot/` | uncommitted, untouched |
| Stray untracked `ValueError` | a shell-redirect accident predating this wave. **Left alone deliberately** — not mine to discard |
| Whole-tree G11 | **0 non-compliant.** The count is self-referential: a whole-tree check's own report is committed after it runs, so the recorded number is always one less than the tree containing it. Reconcile, do not panic |

## 2. The gate numbers every future charter needs — **SMOKE MOVED THIS WAVE**

| Gate | Result on the integration tip |
|---|---|
| **SMOKE** (**22** files) | **503 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** |

**SMOKE moved 473 → 503 under merge rule 4, by C-106.** The arithmetic is exact and must stay so:
`473 + 14 + 16 = 503`, where 14 is `test_c102_coverage_denominator.py` and 16 is
`test_c106_mutation_harness_executable.py`. **Anything still saying 473 (or 457/460/465) is stale**;
`TEST_MATRIX.md` carries the full history rather than deleting it.

**The gold-readers baseline changed through C-103** — any charter still saying that selection
correctly exits 1 is **stale**.

**Two things about SMOKE a future card must know:**

1. **SMOKE is no longer read-only with respect to the working tree.** `test_c106_…` writes to the
   tracked `src/t2pw/bench/acceptance.py` during the run and restores it in a `finally`. It is safe
   under one-heavy-job-at-a-time and never-`-n auto`, and the restore is verified — I hashed the
   file either side of my own run, `70a642ca…` both times. **A card that parallelises SMOKE would
   corrupt that file.**
2. **The mutation-attack harness runs again.** `evidence/c102_mutation_attack.py` was unrunnable
   from `e77ad3d` until C-106, which is why C-104's R5 was registered but never exercised. It now
   restores **saved bytes**, asserts `sha256` **and** CRLF count, and `git checkout --` is gone from
   the restore path.

## 3. T-107 — scored, triaged, and NOT to be rerun

`runs_verify/2026-08-28_1816` · 20/20 legs · 17 scorable · 5.63 h · zero survivors.
**Overall: NOT ACCEPTED, on Priority 2 alone.** Priority 1 = **5**, `PASS`.

**Nothing this wave rescored it and nothing may.** C-105 fixed the defect behind Priority 2's
failure and C-107 calibrated that fix further — **neither re-accepts the run.** A run's verdict is a
fact about the artifacts it produced.

**Full classification: `T107-TRIAGE.md`.** The four things a successor most needs to know are
unchanged and are listed there; the two most load-bearing:

* **`PMC13231680/strict`'s empty pathway is CORRECT and T-105's PASS was the false positive**
  (F-100). **Never write code to recreate T-105's output here.**
* **`LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out with no payload. It **is**
  verified on the pinned run `runs/2026-08-02_2130`. **Do not report T-107 as confirming it.**

### F-148 is now classified — `F148-TIMEOUT-CLASSIFICATION.md`

From committed artifacts only. **Two mechanisms, not one**: one in-process `operation_timeout`
(`stage=input`) and two outer parent kills (`budget_exhausted`, `stage=unknown` because the parent
genuinely could not see).

**Budget-bound, not stochastic.** The slowest leg that *finished* used **92.1%** of a ceiling
someone halved 3600 → 1800 leaving `leg_timeout_override_reason` and `_source` **empty**.

**The finding that matters:** both outer-kill legs carry `finalization_reserve_seconds: 120.0` and
`child_deadline_seconds: 1680.0` and both ran to **1800.4 s**, overrunning the child deadline by
almost exactly the whole reserve. So **`files: []` does not mean the pipeline produced nothing** —
the child was killed while working, with its preservation window already spent. That is *"absence of
a payload caused by cleanup rather than pipeline failure."*

**Retry amplification cannot be excluded, and that is itself the finding**: the artifact needed to
exclude it is the one the kill destroyed.

## 4. Findings registered this wave

| Id | Class | One line |
|---|---|---|
| **F-153** | `product_contract_violation` | `MASTER_PLAN §2` — the section `CLAUDE.md` points every agent at with *"do not rebuild what exists"* — said the RAG loop controller was missing. **Corrected.** `controller.py:11`'s stale `UNWIRED` docstring **not** fixed: no card owns that file |
| **F-154** | `product_contract_violation` | The `## Test discipline` chunk-membership bullet of `.claude/agents/pwml-test-runner.md` (~~`:59`~~) sent that agent to `TEST_MATRIX.md:213-218` for a **stem-exact** chunk match; `:213-218` is the bounded-runner **function** table. **Registered; C-109 replaced those line addresses with the `## Chunks` and `## Commands` anchors** — the numbers are kept only as the historical statement of what was wrong |
| **F-155** | `product_contract_violation` | **Five members of one class** in `apply_audit_patch.py`. See below |

### F-155 is the one to charter next, and it has five members

`(a)` the transport family's bare `transport` stem matches inside **"transporter"**, so a pure
schema rationale licenses the role it asks for — **F-146 in a family the pinned property does not
name** · `(b)` `[^.]{0,80}` is **not** a sentence bound, because `_match_fold` strips every period
before the pattern runs · `(c)` an actor whose **name** contains an enzyme noun licenses with no cue
in the span (`"LpxC hydrolase was quantified in the lysate"`) · `(d)` **C-105's own attenuation
stems** carry the identical unanchored defect (`repressor`, `suppressor`, `inhibitor`) · `(e)` three
load-bearing anchors C-107 added that **no test covers**, exposure measured at 4/2/4 spans.

**(d) makes this the third independent instance of one defect in one file** — `mediat` inside
"intermediate", the six stems C-107 added, and C-105's own. **Any card touching this file should
treat "is this stem anchored as a word on both sides?" as a checklist item, not a discovery.**

Four of the five are the same sentence: *something that is not evidence about this actor in this
role is accepted, or something that is evidence is refused, because a pattern matched inside a
longer word or a schema noun stood in for evidence.* The fifth is the coverage that would stop the
fourth recurring. **Charter them together** — each fix touches the same two functions.

## 5. Cards — both MERGED

**C-106 (`fa69c57`)** — the instruments. Four census pins moved (not the two every document named:
`withheld` 92 → 97 and `with_matched_forbidden` 23 → 26 sit *behind* the census assert and had never
executed), the harness restores saved bytes, F-152's parse is scoped to pytest's summary line, and
the file is in a gate so the next census drift cannot go unseen.

**C-107 (`ca3c711`)** — the C-105 follow-on, six routed findings, **two correction rounds**.
`src` delta is `apply_audit_patch.py` alone. **F-146 rejected at every tip**; 29-case battery
**0 mismatches**; corpus **0 newly refused / 4 newly admitted**, stable row-for-row across all three
tips.

**The caller enumeration corrects the C-105 record: seven call sites across six modules, not four.**
`src/t2pw/bench/` contains **zero** references, so no scoring path reaches this guard —
`PMC12452463/strict` and `PMC12180156/strict` **cannot** flip and both stay correct-by-accident
under F-147.

## 6. Held, needing authority — `DECISION-PACKET-ORCH717.md`

Three questions, none chartered, **no gold file touched**:

* **Q1 negative-control scoring.** The harness reports a contract-correct empty pathway as
  `RESULT: FAIL`. `policy_disagreement`. **The only one of the three where the status quo actively
  produces wrong readings** — the instrument scored the defective T-105 run higher than the two
  correct ones either side of it. Recommendation: a distinct *declined-correctly* verdict; **never
  `PASS`**, which would make "produced nothing" indistinguishable from "produced the right thing".
* **Q2 F-150.** Verified independently this wave, both halves. The δ/delta alias gap is real
  (`forbidden_match('δ-aminolevulinic acid') → None`). And **all ten gold cases have
  `supported_reactions_complete = False`, zero true**, with `max_retained_reactions` set on exactly
  two — both negative controls. **The alias edit is written and NOT applied.** Half 2 is **not**
  proposed as an edit at all: it changes what Priority 2 *measures* on every future run.
* **Q3 PathBank compound ids** — interacts with Q2's Priority-1 prediction; decide together.

**F-147 remains registered and deliberately NOT chartered.** Fixing it alone would flip two legs to
PASS that would then export gold-forbidden content. The earliest unsafe seam is **Stage-1
extraction**, not the driver. Merge rule 6.

## 7. Traps this wave paid for — additional to the standing list

1. **A prose instruction repeated in three documents can still be wrong.** The handoff, F-151 **and**
   REV-104 all said "re-pin 62 → 72". Four pins moved. **Measure before you charter.**
2. **A fix bound to the wording it was written from closes nothing.** C-107 round 1 closed the exact
   frame its card quoted; 15 of 44 frames stayed open. The repair was to bind **grammatically**, not
   lexically.
3. **Test the obvious repair before proposing it.** REV-107's left-lookbehind hypothesis took false
   refusals only 8 → 6. A reviewer that proposes rather than measures sends the author down a path
   that half works.
4. **A bounded closed list flips polarity between a cue and a contra.** In a **cue** it
   under-accepts and is safe; reused in a **contra** it under-refuses and is not. C-107 reused the
   constant without noting the flip.
5. **Attribute which guard refused; do not guess.** That is how C-107 found a second defect site the
   review had not named.
6. **A line-address pin is only as good as the addresses when it was declared** — F-154.
7. **Bash heredocs here break on apostrophes.** Write long text to a file and `cat` it. This cost me
   one silently-parsed-and-unexecuted command block.
8. **Set `PYTHONIOENCODING=utf-8`** on any probe that prints non-ASCII; a `cp1252` console kills it
   mid-run.

## 8. Peer sessions

One live interactive peer this wave (`project14-t2pw-93`), doing **read-only** RAG reconnaissance;
it held nothing and contended for nothing. **A second session had claimed the identical Lead
Orchestrator role about ten minutes before me** (`project14-t2pw-b1`), from a stale `36f773c`, and
intended to re-triage T-107 legs already triaged and committed. It was unreachable and absent from
`ListAgents` — it had exited. **Run `ListAgents` and contact every live peer before claiming the
branch, the lock or a worktree.**

**Verify a peer's claims about your own tree.** One of this peer's two factual reports was wrong —
it flagged eight committed G11 files as uncommitted, then traced its own report to a **torn
`git status` read taken while another session was mutating the tree**. The other (F-153) was right
and valuable. Neither was taken on trust.
