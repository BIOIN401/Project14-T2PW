# HANDOFF — run T-104 and T-105. Nothing else.

Paste everything below this line into a fresh session.

---

You are the Lead Orchestrator for the PWML recovery sprint in
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, branch `sprint/pwml-recovery`.

**Your ONLY job this session is to run the T-104 and T-105 release candidates, triage between
them, and record the results. Do not open cards. Do not dispatch implementer or reviewer
subagents. Do not refactor anything.**

## 0. Budget discipline — read this first

The previous session already did the discovery. **Do not re-derive any of it.** Everything you
need is below or in the two files named in § 6. Concretely:

* **Do NOT** re-read `FINDINGS.md` (4000 lines), `LEDGER.md` (2100 lines) or `DECISIONS.md`
  (3600 lines) end-to-end. Grep for a specific ID if you need one.
* **Do NOT** re-run SMOKE, Chunk A/C/D/E, or G11 whole-tree scans. They were green at this exact
  SHA hours ago and no code has changed since.
* **Do NOT** re-verify the merged cards. C-069, C-070, C-071 are merged, gated, reviewed and
  pushed.
* **Do NOT** re-investigate F-062, F-079, F-084 or F-091. All closed. See § 5.
* Read a file only when you are about to act on it.

## 1. Exact starting state — verify once, briefly, then start the run

| check | expected |
|---|---|
| `git rev-parse HEAD` | `266aba6dfe81fa414ac7c532e300a6eceb11ea2e` |
| `git ls-remote origin sprint/pwml-recovery` | same SHA |
| merge in progress | none |
| staged files | 0 |
| `C:\t\heavylock` | **absent** |
| sprint-owned Python processes | **0** (only two `ms-python.isort` IDE servers — never kill those) |
| `TEST_MATRIX.md` | **578 lines** (D-061 raised it from 541; that is correct) |
| product-owner edit `src/t2pw/app/streamlit_app.py` | **35 insertions / 2 deletions**, uncommitted, `sha256:e50a248bb7189c222896f74bc38cdbd1c6dbbc6dc3a2594b3e5e63ea261416e0` |

**Expected uncommitted working tree — this is correct, leave it alone:**
`.claude/settings.json`, `data/enrichment_cache.json`, `data/id_mapping_cache.json`,
`out/enrichment_dump.json`, `outputs/pathway.pwml`, `src/t2pw/app/streamlit_app.py`,
`tmp/draft_graph.json`, `tmp/qa_report.json`, `tmp/reaction_summary.txt`, and four untracked
`topics_*.txt`.

**NEVER commit `data/enrichment_cache.json` (39 MB) or `data/id_mapping_cache.json`. NEVER touch
`streamlit_app.py`.**

## 2. ⚠ PROCESS HYGIENE — the user asked for this explicitly

**Every** test, benchmark, probe and pipeline leg runs through the bounded wrapper. It owns the
heavy mutex, isolates children in a Windows Job Object with `KILL_ON_JOB_CLOSE`, and reports
survivors. **Nothing runs outside it.**

```
cd C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW
PYTHONPATH="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src" \
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label <label> --timeout <seconds> --heavy-lock <holder> \
  --json <allocated g11 path> -- \
  <command>
```

* Allocate every `--json` immediately before the job:
  `.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next --task T-104 --label <label>`
* **After EVERY job confirm `FINAL SURVIVING COUNT : 0` and `cleanup : success`. Report both.**
  A run with survivors is an infrastructure failure, not a test result.
* **After the run, verify with PowerShell that only the two IDE `isort` processes remain.**
* **FORBIDDEN: `taskkill /IM python.exe`, `pkill python`.** Two `python.exe` belong to the user's
  IDE. Kill Python only by the wrapper, never by name.
* Environment variables go in the **shell prefix**, never as `env VAR=x` after the `--`.
  `bounded_run.py` has no `--env` flag; the child inherits. This is measured (D-058).
* A `--timeout` longer than 10 minutes **must** use `run_in_background: true` — your tool wall
  clock is 10 min and will kill the wrapper before its `finally` releases the lock.

**If a run is interrupted and leaves a stale lock:** read `C:\t\heavylock\holder.json`, confirm
its `pid` is actually dead, and only then remove the directory. **Never delete a lock you cannot
account for.** (This happened once already and was handled exactly this way.)

## 3. Run T-104 — first release candidate, ~7 h, 20 legs

**Pre-flight first. It is free and it caught a real error last time.**

`--stage-only` executes **zero** Streamlit/LLM legs by construction — it returns before the run
loop. Run it, confirm the acquisition funnel shows every paper eligible with full text, then
re-run **without** `--stage-only` and **without** `--fresh` so the live run continues the exact
plan you just verified.

```
# 1. pre-flight (seconds, free)
<wrapper --label t104-preflight --timeout 900 --heavy-lock t104-pre> -- \
  .venv/Scripts/python.exe -u scripts/batch_run.py \
    --topics <topics file> --out runs_verify --modes strict,research \
    --timeout 1800 --deadline 8 --fresh --stage-only

# 2. the real run — background, drop --fresh to continue the verified plan
<wrapper --label t104-rc1 --timeout 32400 --heavy-lock t104-rc1> -- \
  .venv/Scripts/python.exe -u scripts/batch_run.py \
    --topics <topics file> --out runs_verify --modes strict,research \
    --timeout 1800 --deadline 8
```

**⚠ `T2PW_SPECIES_LLM=0` must NOT be set for T-104.** That flag is T-103-only — PACK 9 RULING 3
says in terms *"T-104 must not inherit this."*

**The topics file:** `TEST_MATRIX.md` says T-104 is the "full pinned, 20 legs" set. Determine the
correct topics file from `TEST_MATRIX.md` and `BASELINE.md` before running — **do not guess, and
do not reuse `topics_t101.txt` or `topics_t103.txt`**, which are 3-paper and 2-paper subsets.
If no 20-leg topics file exists, build one from `BASELINE.md`'s pinned set and **say so in the
run record.**

**Scope lines must be exact.** A wrong organism produces `scope_conflict` and wastes the leg —
this happened on T-101 (F-093). When reconstructing any scope, **prefer a field the system
DERIVED (the run-directory slug) over a field a human SUPPLIED (`00_PAPER.txt`'s `organism:`,
which records the request, not the paper).**

## 4. Triage between T-104 and T-105 — mandatory, never collapse the two runs

`MASTER_PLAN.md` requires: **T-104 → triage and correction pass → T-105.** They are two separate
~7 h runs. **Never merge them into one.**

Classify every T-104 failure as exactly one of these before proposing any code change:

* `product_contract_violation` — **only this justifies code**
* `gold_data_defect`
* `policy_disagreement`

cite the gold `relevance_note` / `export_rationale` when classifying. **A benchmark failure does
not by itself justify a code change.**

**Carry this into triage explicitly — it is the one open question from T-103:**

> PMC12452463/strict reached `review_required` with `strict_acceptance_eligible=false`, which is
> exactly what `PRODUCT_CONTRACT.md:341` requires. **But it got there by the SEMANTIC gate**
> (`actor_named_in_its_own_cited_span`, C-071's new check), **not by the route the gold rationale
> describes** — gold calls the route chemically broken because **EntA is absent**. The pipeline
> does record that (`missing_anchors: ["EntA","Fur"]`, `completeness: 0.857`) but **coverage still
> passed** (`minimum_core_satisfied: True`). **The required outcome and the pipeline's reason for
> producing it are not the same fact.** Decide in triage whether that matters.

Then run T-105 the same way and record which failures were explained and which persist.

## 5. Facts you need — do NOT re-derive these

**Model configuration.** All nine `OPENROUTER_*_MODEL` slots in `.env` are
**`deepseek/deepseek-v4-flash`**, switched with product-owner authorization.
`openrouter/free` **cannot drive Stage 1** — it fails with *"Chunk 1 failed to produce valid
JSON"*. Do not switch back. Original config backed up at
`<session-scratchpad>/env.backup-before-deepseek`.

**Cost.** Account is **NOT free-tier**: $75 limit, **~$71.81 remaining**, deepseek-v4-flash is
$0.08/1M prompt and $0.16/1M completion. T-101's 6 legs and T-103's 4 legs cost well under a
dollar combined. **T-104 + T-105 are 40 legs total — still small, but check
`GET https://openrouter.ai/api/v1/key` before starting if you want the current balance.**
**Any fallback to a non-free model spends real money** (D-058). No fallback is configured.

**LM Studio** must be running for RAG legs: `http://127.0.0.1:1234/v1`, model
`text-embedding-nomic-embed-text-v1.5` (768-dim). **Verify it responds before a 7 h run.**

**Milestone status already recorded — do not re-litigate:**

| milestone | status |
|---|---|
| T-100 | acceptance met (TRAP-1 satisfied by the T-103 run) |
| T-101 | **`MEASURED`, NOT PASS** — clause 1 violated, clause 2 unexercised, clause 3 inconsistent (F-092) |
| T-102 | **`MEASURED — organism/SBML axis structurally unreachable (F-009)`. Never PASS.** |
| T-103 | **`MEASURED` — acceptance satisfied at `round_count=1`; multi-round re-entry UNTESTED** |

**Findings closed this session, do not reopen:** F-062 (confirmed closed by measurement — no
card), F-079 (closed by C-071), F-084 (disproved offline — not a defect), F-091 (closed by C-071),
F-093 (my scope error, corrected).

**Findings OPEN and unowned — expect T-104 to hit F-092 again:**

* **F-092 (HIGH)** — two identical wall-clock timeouts recorded **different** terminal reasons
  (`budget_exhausted` vs **absent**) and **neither was `operation_timeout`**. The runner's
  timeout message also **hard-codes** the phrase *"produced nothing"*. **If T-104 legs time out,
  expect the same inconsistency — record it, do not "fix" it mid-milestone.**
* **F-090 (MEDIUM)** — `bounded_run.py` writes every descendant PID into its report, and
  `g11_evidence.py` caps a report at 64 KiB. A job spawning thousands of short-lived children
  produces a non-compliant report from a clean run. **Only bites base-tree exports, not pipeline
  legs.**
* F-086, F-087, F-088, F-089 — minor, recorded, unowned.

**Merge rules still bind** even though you are not merging: never weaken a biological gate to
increase PWML output; preserve incomplete-but-correct pathways as `review_required` rather than
dropping them; never let exporters repair biology after the canonical graph is frozen.

## 6. The only two documents worth reading in full

1. `docs/pwml_recovery_sprint/RESUME-NEXT-SESSION.md` — current state, findings table, process
   state, the pre-charge register (fully measured: `test_strict_failure_replay.py` 2 failures,
   `test_batch_preflight.py` **0**, seven `.env`-conditional, `qb` node15).
2. `docs/pwml_recovery_sprint/TEST_MATRIX.md` § "Milestone benchmarks" (~line 471) — the T-104 and
   T-105 rows and their acceptance criteria. **578 lines; citations pinned through line 477. If
   you edit it, edit IN PLACE and keep the line count, or record the new count in D-061.**

`PRODUCT_CONTRACT.md` outranks any test, benchmark result, or inference from the code.

## 7. Recording results

Commit run artifacts on the established convention: **everything under the run directory EXCEPT
`cache_snapshot/`** (that is the two 39 MB caches — 45 MB on disk becomes ~3 MB tracked).

Use `git commit -F <file>` with a message file; never a long inline `-m` (PowerShell here-docs
silently no-op in this environment). Stage explicit paths and inspect `git diff --cached --name-only`
before committing. Push to `sprint/pwml-recovery` and verify `local = origin = git ls-remote`.
**Never merge to `main`.**

Record each milestone's status **in the `TEST_MATRIX.md` row itself**, not only in prose. If a
milestone's acceptance is partly unexercised, say **`MEASURED`** with the qualifier — **never
`PASS`**. A milestone that finds a real defect has done its job; recording it as PASS discards
exactly that.

## 8. When to stop and ask the user

* Two exhausted correction rounds on the same problem.
* A locked product-contract interpretation, or an append-only `DECISIONS.md` entry.
* Anything that would spend materially more money than the ~$1 these runs should cost.
* A blocker that prevents all remaining useful work.

Otherwise proceed on your own judgement. The user is not watching and wants the runs done.

**Before you finish: confirm zero sprint-owned Python processes, heavy lock free, no merge in
progress, product-owner edit still 35/2, and everything pushed.**
