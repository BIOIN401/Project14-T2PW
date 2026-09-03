# PWML RECOVERY SPRINT — FINAL HANDOFF. Engineering complete, production FROZEN.

**Written by the Lead Orchestrator, session `project14-t2pw-51` `[e2c249]`, at the close of
`ORCH-720`, 2026-09-03.** Replaces the T-108-execution-wave handoff, which is in git history.
**`LEDGER.md` remains the single source of truth for task state.**

> ## ⚠ THE ENGINEERING SPRINT IS CLOSED — `D-090`
>
> **The recovery pipeline is ENGINEERING-COMPLETE and production is FROZEN.** **T-110 is NOT
> authorized.** **Nothing is running:** T-109 exited, was scored once, and is closed; the heavy lock
> is free, zero sprint-owned Python is alive, no job is unowned. Verify all three yourself — § 1.
>
> **The next phase is the RAG / LLM EVALUATION FRAMEWORK. Its launcher is
> [`prompts/PROMPT-001-eval-framework.md`](prompts/PROMPT-001-eval-framework.md) — paste it into a
> fresh session.**
>
> **The rule most likely to be broken by accident:** *no production behaviour changes solely to
> satisfy the incomplete test instrument.* A `src/` change justified by *"it would make Priority 2
> evaluable"* or *"it would move Priority 5 off zero"* is a **reject**.

---

## 1. Takeover — verify once, do not trust these numbers

| Check | Expected | How |
|---|---|---|
| Integration tip | `local = origin/ = git ls-remote` | `git rev-parse HEAD; git rev-parse origin/sprint/pwml-recovery; git ls-remote origin sprint/pwml-recovery` |
| `main` | local `7531692` / remote `03f1af5`, **never written** | `git rev-parse main; git ls-remote origin main` |
| Gold | `36f4b7b690b577f72882c3045ca6728d1ec8d9d1` | `git hash-object src/t2pw/bench/gold/pinned_v1.json` |
| `acceptance.py` | CRLF sha256 `4bd893ac…` · LF sha256 `d9f817e1…` · blob `56aa593e…` | **two of those three are sha256 and one is not — say which you mean** |
| `streamlit_app.py` | sha256 `47e4fafa…`, **modified and never committed** | `git status --porcelain src/t2pw/app/streamlit_app.py` |
| Heavy lock | **absent** | `ls C:/t/heavylock` |
| Python processes | exactly two `ms-python.isort … lsp_server.py` | **match on FULL COMMAND LINE.** The PIDs changed during T-109 and the count has been 2 and 3 in this sprint. **Never match on count or PID** |

---

## 2. What happened this wave, and what it settles

### 2.0 The product owner closed the sprint — `D-090`

**The recovery pipeline is engineering-complete. Production is FROZEN. T-110 is not authorized.**

**T-109's disposition, in the ruling's own terms: OPERATIONALLY SUCCESSFUL, formally `NOT ACCEPTED`,
because Priority 2's test dataset was not evaluable.** The non-acceptance is attributed to the
**test instrument**, not to the pipeline — Priority 2 did not fail, it could not be evaluated.

**FROZEN:** `src/t2pw/pipeline/` (including the INCOMPLETE-CORE CAP), `pwml/`, `mapping/`, `batch/`,
`llm/`, `acceptance.py` semantics, and the gold blob.
**NOT FROZEN:** the evaluation framework and its instruments, gold **curation** where D-087's
standard is genuinely met, and test/gate tooling that measures the pipeline without altering it.

**`F-175` and `supported_reactions_complete` curation move into the RAG / LLM evaluation work.**

*Engineering-complete* means **no further production engineering is chartered in this sprint.** It is
**not** a claim that every gate is green or every finding closed — **F-147 stays unchartered.**

> **SUPERSEDED IN PART by `ORCH-721`, 2026-09-03.** This section said *"F-174 node 2 stays OPEN"*.
> **It is now EXPLAINED** — same-SHA A/B, worktree green and primary red, lever isolated to `.env`
> and within it to two independent single keys. See F-174's `ORCH-721` amendment. One residual
> stands: the lever set is **not** proven complete for the primary checkout.

### 2.1 The product owner ruled — `D-089`

**D-088 clause 10 controls for this release. The INCOMPLETE-CORE CAP is unchanged.** No cofactor
vocabulary, no entity-list match, no Stage-0 redesign, no gold change, no curated expectations inside
production. `PMC12096016/strict` stays `review_required` with its pathway preserved for review, and
**Priority 5 stays `0/2`**.

> **This is an EXPLICITLY ACCEPTED CONSERVATIVE LIMITATION and NOT delivery of D-088 clause 2.**
> Any report, badge or summary describing the cap's survival as *"D-088 implemented"* is wrong.

**The product principle is reaffirmed, not withdrawn.** Its implementation is deferred to
**`R-D089-1`** — a stable, general, **non-paper-keyed** reaction/subprocess completeness
specification typing participants as *defining* / *optional* / *extracted-but-unwired* / *genuinely
absent*. **Owner phase: the RAG / LLM evaluation framework. Explicitly out of scope for a finishing
wave, by ruling.**

### 2.2 T-109 ran, was scored ONCE, and is IMMUTABLE — `T109-RESULT.md`

**Verdict: `NOT ACCEPTED`. No hard gate FAILED; one hard gate could not be EVALUATED.**

| | |
|---|---|
| **P1** zero false real identifiers | **`ok=true`**, raw **0**, accepted **0** |
| **P2** zero unsupported retained reactions | **`ok=null`, NOT EVALUATED** on 13 of 19 legs |
| **P3** zero referential-integrity violations | **`ok=true`**, observed **0** |
| P4 requested-pathway coverage | `0/8 = 0%` — **not a hard gate** |
| P5 strict PWML pass rate | `0/2 = 0%` — **not a hard gate**, and the outcome `D-089` accepted |

**Priority 2 is why it is not accepted**, and it is the standing D-087 limitation:
`supported_reactions_complete` is unset on all ten cases, so the verdict is reachable only where a
`max_retained_reactions` ceiling exists and **both ceilings are on negative controls.**
`acceptance.py:1057` — *"`ok=None` is falsy … the correct default for an unproven absolute."*

**Operationally the best run of the sprint:** 20/20 legs, `complete: true`, **timeouts 3 → 1 → 0**,
zero empty payloads, **4.95 h** against T-108's 6.37 h, survivors 0, cleanup success.

### 2.3 D-088's two required consequences BOTH hold, on a draw D-089 was not ruled on

Both Priority-5 legs **passed the strict technical gates AND semantic evaluation** and were held by
the anchor cap alone:

| leg | completeness | missing anchors |
|---|---|---|
| `PMC12096016/strict` | **0.916667** | **`EntD` alone** — the one anchor D-088 declines to excuse. **`ATP`, `NADH`, `Fur` all MATCHED** this draw |
| `PMC12782028/strict` | 0.571429 | `oxysterol, MSMO1, SQLE, FDFT1, HMGCR, HMGCS1` — **`HMGCR` and `HMGCS1` ARE the mevalonate arm** |

**A change clearing both would have been a reject. Nothing cleared either.** And **Candidate A is
re-refuted on fresh data**: `0.571 ≥ min_core_coverage 0.5`, so the existing thresholds alone would
have **released the leg whose mevalonate arm is missing.**

---

## 3. What is NOT established, and must not be quietly upgraded

- **Priority 1's 8 → 0 is NOT a fix.** **No production code changed between T-108 and T-109** — the
  whole wave was documentation and evidence — so a code change cannot explain it. The exported
  surface differs, and an 8 → 0 move is far outside the scorer's own one-finding stochastic band.
  **It is evidence about the draw until a second run reproduces it.**
- **Priority 3's 0 is a gate holding, not a clean draw.** `PMC12856317/strict` produced orphaned
  references `HRM3`/`HRM6`; the gate refused the leg, so they never reached an export.
- **`LpxH` remains UNVERIFIED.** Both `PMC12444477` legs failed with `findings=0`. It is verified
  only on `runs/2026-08-02_2130`.
- **F-146 is not fixed.** Its invented reaction was absent again on `PMC13231680/research`; absence
  on a draw is not a fix, in either direction.
- **T-107, T-108 and T-109 are all immutable.** None is re-run, re-scored or reinterpreted.

---

## 4. Findings registered this wave — F-173, F-174, F-175

**F-173** — `PMC12096016/strict`'s `review_required` is a known false negative **with a known sign**;
half of Priority 5's strict denominator is known-misclassified in a known direction. **T-109 REFINED
it:** on that draw the cap was held by `EntD` alone, which D-088 does not excuse, so *that* instance
was not a false negative. **The general claim — the instrument cannot distinguish the two legs on
inputs production may read — stands.**

**F-174** — the authoritative **Chunk D** gate has **never** been run in the primary checkout and is
RED there (`run-core 159/160`, `node15 0/1`). Both committed green runs had `cwd` inside a worktree.
**It cannot be a code regression:** the only commit since the last green Chunk D touched three
evidence artifacts and no `src/`, `tests/` or `scripts/`. Node 1's mechanism is proven; **node 2's
lever is OPEN.**

**F-175** — **C-116's D-088 diagnostics are written only on the Streamlit path.**
`write_quarantine_artifacts` is called from `streamlit_app.py` and nowhere else, so
`coverage_diagnostics.json` exists in **zero** benchmark legs. **A `D-089` preservation requirement is
unmet going forward**, though the archived census and its committed logs are intact and
`coverage_summary.json` still carries `matched_terms`/`unmatched_terms` verbatim per leg.

> **F-171 through F-175 are one lesson in five costumes: a green signal whose scope nobody asked
> about.** A handoff row certifying a suite that had been red for days · a checker certifying half a
> gate · an instrument conservatively wrong in a known direction · `187/187` that was true only of
> worktrees · a test proving a property inside a function nobody calls.

---

## 5. THE NEXT WORK ORDER — the RAG / LLM evaluation framework

> **The launcher is [`prompts/PROMPT-001-eval-framework.md`](prompts/PROMPT-001-eval-framework.md).**
> It carries the full ordered work order, the freeze rules, the verification block and every process
> trap. **Paste it into a fresh session.** What follows is the summary it expands.

**Nothing below is chartered. Charter one card at a time, narrowly, and review the diff.**

**The ordering changed under `D-090`:** `supported_reactions_complete` is now **item 1**, because
Priority 2 is the only hard gate standing between a run and acceptance and **making it evaluable is a
DATA task, not a code task.**

### 5.1 The framework itself — where the sprint was always going

**`R-D089-1` lives here and it is the largest piece.** A stable, general, non-paper-keyed
reaction/subprocess completeness specification distinguishing:

1. **defining** substrates, products, enzymes, reactions and major pathway branches;
2. **optional** cofactors, currency metabolites, regulators, ancillary proteins;
3. **extracted-but-unwired** entities;
4. **genuinely absent** core reactions or subprocesses.

**The inputs already exist and must not be rebuilt:** the curated ten-paper expectation set in
`curation/` (41 reactions, 35 subprocesses, **174 quotes verbatim, 0 fatal**), the 374-anchor census
with its 60/90 split, and the v4 A/B harness that discriminates on **every** archived draw. **What is
missing is the production-safe, non-paper-keyed form** — `PRODUCT_CONTRACT` § 12 forbids reading the
curated set in the general pipeline.

### 5.2 The three registered residuals, in the order they cost the most

| id | what | why it matters |
|---|---|---|
| **`supported_reactions_complete`** | curate **ONE** case properly, per D-087 | **the only hard gate between a run and acceptance.** `goldset.py:384` warns that setting it without exhaustive signatures turns every unattributed row into a reported fabrication — `semantic.py:700` records that this would have reported **227** on a run that produced far fewer. **The audit is the cost, not the edit, and it is not a Lead judgement** |
| **F-175** | ~~make the batch path write `coverage_diagnostics.json`~~ **AMENDED by `ORCH-721`.** The writer **DOES** run on the batch path — 10 legs carry `quarantine_report.json`, which only it writes. The diagnostics are written to disk and **deliberately excluded from the RETURNED map**, and the driver carries only what that map names. The fix is **one tuple entry in FROZEN `driver.py`**, so it is **ESCALATED for a ruling**, and it still cannot be proved without a real benchmark leg | a `D-089` requirement is unmet. **The test must assert the file exists in a BATCH LEG DIRECTORY** — C-116 had eleven passing tests and none ran the batch path |
| **F-174 node 2** | **DONE — `ORCH-721`.** ~~isolate why the test is red in the primary checkout~~ | ~~**The DB-config lever is already excluded**~~ — **THAT WAS WRONG, and this row is why the row itself is dangerous.** The DB lever was **masked, not excluded**: `LLM_PROVIDER` and `PATHBANK_DB_*` are **each independently sufficient**, so a one-variable A/B read red in both arms. Do not re-inherit the exclusion |
| **F-172** | **DONE (checker half) — `ORCH-721`.** ~~make `check` **require** a `.pin.json`~~ — delivered as **report-always / enforce-on-request**, because requiring it turns **3206 of 5202** committed artifacts red and unconditional `refused` fails H-010's and REV-070's **negative controls**. Two residuals stay open: indirect drivers (`chunk_d_gate.py`) are NOT COVERED, and the 3206 unpinned reports are a standing backlog | it certifies the lifecycle half of G11 and rule 10 **not at all**. Changing it mid-wave breaks comparability — **do it at a wave boundary, which is now** |

### 5.3 Priority 2 is the only thing standing between a run and acceptance

**Every other hard gate passed on T-109.** `supported_reactions_complete` is the blocker and
**D-087 governs it: never set broadly, never guessed, only on individually curated cases whose
expected reaction sets have been independently verified complete.** The ruling restated this
verbatim. `DECISION-PACKET-F150-HALF2.md` holds the unanswered question and its options; **option B —
one deliberately chosen case after a genuine biological completeness audit — is the smallest step
that makes the metric mean anything.** **The audit is the cost, not the edit, and it is not a Lead
judgement.**

---

## 6. Protected — do not touch

- **F-147 registered and deliberately UNCHARTERED.** `placeholder_backed_proteins` — **escalate
  only**.
- **`main` untouched.** Do not merge to it.
- **`streamlit_app.py` never committed.**
- **Never commit** `data/enrichment_cache.json` (39.9 MB working / 34.2 MB committed),
  `data/id_mapping_cache.json`, `topics_*.txt`, the stray 0-byte `ValueError` and `=`, `out/`,
  `outputs/`, `tmp/`. `runs_verify/*/cache_snapshot/` is gitignored.
- **`HANDOFF.md` § 7 forbids pruning a worktree.** ~180 exist sprint-wide; a stranded 485 MB one sits
  at `C:\t\rev114\basetree`. Untidy, not urgent.
- **T-107, T-108, T-109 immutable.**

---

## 7. Process — merge gates, not suggestions

**All eleven merge rules in `CLAUDE.md` stand unchanged.** Additionally, learned or re-confirmed this
wave:

1. **`grep -E "^OPENROUTER_API_KEY=" .env` finds NOTHING.** The live key is written `KEY = value`
   **with spaces**, so the only matchable line is the commented-out one above it. **Verify
   configuration through the loader** — `evidence/t109_preflight_provider.py` exists for this and
   never prints a value.
2. **`.pin.json` verdicts go in `evidence/g11/pin/<TASK>/`**, never in the task directory, where
   `check` reads them as malformed cleanup reports.
3. **Pre-create every `--basetemp` parent.** A missing one gives `1 error in 0.18s`, which looks
   exactly like a test result.
4. **Background any lock-waiting bounded run and branch on exit 95.** A foreground one is killed by
   the tool's **120 s** cap *while holding the lock*.
5. **Never batch the four AppTest files into one pytest process.** They stall it silently for 40
   minutes. Chunk D's authoritative gate is the split-process runner.
6. **Before building an instrument to separate two hypotheses, check whether one is already excluded
   by something already written down.** F-174 was settled by `git diff --name-only`, after a 140-line
   probe had been written to settle it.
7. **A benchmark failure does not by itself justify a code change.** Classify first:
   `product_contract_violation`, `gold_data_defect`, or `policy_disagreement`. **Only the first
   justifies code.** T-109 hard-failed nothing, so **no fix mandate is triggered by it.**

---

## 8. The transferable lesson of this wave

**The cheap move was available, specified, and known to move the headline number — and it was
refused.** Relaxing the cap on a cofactor vocabulary would have cleared `PMC12096016` and moved
Priority 5 off zero. It would also have released `PMC12782028`, whose mevalonate arm is genuinely
absent, and T-109 then named the missing enzymes — `HMGCR`, `HMGCS1` — in its own artifacts.

> **When a change improves a score, the question is never "did the number move" but "did the thing
> the number measures move."** The ruling chose the conservative false negative, the run vindicated
> the choice on fresh data, and **Priority 5 still reads `0/2`.** Both of those sentences are the
> result.
