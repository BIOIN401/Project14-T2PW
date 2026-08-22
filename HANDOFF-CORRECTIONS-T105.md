# Handoff — the correction phase, then T-105

You are the Lead Orchestrator for the PWML recovery sprint in
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, branch `sprint/pwml-recovery`.

Your job this session is to land the **two corrections T-104 justified**, then run **T-105**.
Cards, review, merge, benchmark. Nothing else.

---

## 0. Budget discipline — read this first

T-104 ran and was fully triaged. **Do not re-derive any of it.**

- Do **NOT** read `FINDINGS.md` (4245 lines), `LEDGER.md`, or `DECISIONS.md` (3729) end-to-end.
  Grep for a specific ID. The four entries that matter are `F-094`, `F-095`, `F-096`, `F-097`
  and the two rulings `D-062`, `D-063`.
- Do **NOT** re-run T-104. It is committed with its full acceptance report.
- Do **NOT** re-open the triage. Classification is done and locked: **only F-094 and F-096 are
  `product_contract_violation`, and only those two justify code.** F-095 and F-097 are
  `policy_disagreement`.
- Do **NOT** re-investigate `F-062`, `F-079`, `F-084`, `F-091`, `F-093`. All closed.
- Read a file only when you are about to act on it.

**The two documents that are current:** `docs/pwml_recovery_sprint/evidence/t104_acceptance_report.txt`
(the measurement) and the `T-104` row of `TEST_MATRIX.md` (~line 481).

### Things already discovered the hard way — do not rediscover them

- **The g11 allocator is `docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py`**, not
  `evidence/g11_evidence.py`. The obvious path does not exist.
- **T-104 already used g11 slots `T-104/01`–`04`** (`stage-only-preflight`, `verify-plan`,
  `t104-rc1-20legs`, `acceptance-score`). Allocate `T-105/…` fresh.
- **`--stage-only` does NOT surface scope conflicts.** Acquisition reported *"requested 10,
  examined 10, eligible 10, ineligible 0"* — all three organism-trap papers passed eligibility.
  Stage-0 reconciliation (`driver.py:540`, `apply_stage0_observation`) runs **per-leg**. The
  pre-flight is still worth running; just do not read a clean funnel as "no scope conflicts".
- **`scope_conflict` legs still produce scorable payloads.** They are not empty. PMC12421875's
  research leg reached 22 reactions before refusing. `payloads available` was 18/20 — the two
  missing are the **timeouts**, not the conflicts. So `scope_conflict` ≠ "nothing was attempted",
  which is the whole basis of D-062.
- **Scope conflicts are not cheap.** They cost 4–16 min per leg, not seconds; Stage 0 does real LLM
  work before refusing. Budget for it.
- **`bench_acceptance.py` has `--verify-plan`, `--verify-topics`, `--write-topics` and
  `--validate-gold`.** All offline, all free, all fast. Use them instead of hand-checking scopes.
- **Do not rebuild what exists** — `MASTER_PLAN.md` §2. The UniProt accession fetch, the provenance
  carrier, alias traversal, the RAG gap detector and the RAG admission gate are already implemented
  and tested.

⚠ **`RESUME-NEXT-SESSION.md` is STALE.** It describes the pre-T-104 world — C-069/C-070/C-071 in
flight, four pending product-owner decisions, all since resolved. **Its §6 pre-charged failure
register is still valid** and is reproduced in §6 below. Ignore the rest or rewrite it.

---

## 1. Starting state — verify once, briefly, then start

| check | expected |
|---|---|
| `git rev-parse HEAD` | `e790465` (parent `2673067`) |
| `git ls-remote origin sprint/pwml-recovery` | same |
| merge in progress / staged files | none / 0 |
| `C:\t\heavylock` | absent |
| sprint-owned Python processes | 0 (two `ms-python.isort` IDE servers — **never kill those**) |
| `TEST_MATRIX.md` | **578 lines**, line 477 byte-identical (D-061 binds this) |
| `FINDINGS.md` / `DECISIONS.md` | 4245 / 3729 lines |
| product-owner edit `src/t2pw/app/streamlit_app.py` | 35 ins / 2 del, uncommitted, `sha256:e50a248bb7189c22…` |

**Expected uncommitted tree — correct, leave alone:** `.claude/settings.json`,
`data/enrichment_cache.json`, `data/id_mapping_cache.json`, `out/enrichment_dump.json`,
`outputs/pathway.pwml`, `src/t2pw/app/streamlit_app.py`, `tmp/*.json`, `tmp/reaction_summary.txt`,
four untracked `topics_*.txt`.

**NEVER** commit `data/enrichment_cache.json` (39 MB) or `data/id_mapping_cache.json`.
**NEVER** touch `streamlit_app.py`.

---

## 2. ⚠ PROCESS HYGIENE

Every test, benchmark, probe and pipeline leg runs through the bounded wrapper. It owns the heavy
mutex, isolates children in a Windows Job Object with `KILL_ON_JOB_CLOSE`, and reports survivors.

```
cd C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW
PYTHONPATH="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/src" \
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label <label> --timeout <seconds> --heavy-lock <holder> \
  --json <allocated g11 path> -- <command>
```

- Allocate every `--json` immediately before the job:
  `docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py next --task <T-105|C-0xx> --label <label>`
  (note: the allocator lives in `evidence/g11/`, **not** `evidence/`).
- After EVERY job confirm **`FINAL SURVIVING COUNT : 0`** and **`cleanup : success`**. Report both.
  A run with survivors is an *infrastructure failure*, not a test result.
- Then verify via PowerShell that only the two IDE isort processes remain.
- **FORBIDDEN:** `taskkill /IM python.exe`, `pkill python`. Kill Python only via the wrapper.
- Env vars go in the shell prefix, never `env VAR=x` after the `--`. `bounded_run.py` has no
  `--env` flag; the child inherits (measured, D-058).
- **Any `--timeout` over 10 minutes must use `run_in_background: true`** — your tool wall clock is
  10 min and will kill the wrapper before its `finally` releases the lock.
- `--basetemp=<dir>` on **every** pytest invocation. Without it 83 tests error with
  `PermissionError` and you will report a false regression. Never run the full suite unchunked
  (~16 GB). Never `pytest -n auto`. One heavy job at a time.
- Stale lock: read `C:\t\heavylock\holder.json`, confirm its pid is dead, only then remove.

---

## 3. The work — two cards, in this order

Card prompts live in `docs/pwml_recovery_sprint/prompts/` (`_TEMPLATE_IMPLEMENT.md`,
`_TEMPLATE_REVIEW.md`, `_SHARED_BLOCKS.md`). Follow the existing C-xxx convention.

### C-072 — F-094: `release_ready` on a paper the contract forbids

**Required outcome, from `PRODUCT_CONTRACT.md` §13, quoted:**

> PMC12452463 — Correct outcome after the index fix is `review_required` with
> `strict_acceptance_eligible=false`. **Never strict success.**

T-104 produced `release_ready [pipeline ran; strict gates passed; semantic evaluation passed]`. It
is the only strict leg in the run that emitted a bare `pathway.pwml`, which by the contract's own
naming rule *means* release_ready.

⚠ **This fix REDUCES PWML output, and that is the correct direction.** Merge rule 6 forbids
weakening a gate to *increase* production; it does not protect production. Do not let an
implementer argue the strict rate down as a reason to soften this.

⚠ **Do not "fix" it by re-firing the semantic gate.** F-094's whole point is that the required
outcome was resting entirely on C-071's `actor_named_in_its_own_cited_span`, while the coverage
route reported `minimum_core_satisfied: True` despite `missing_anchors: ["EntA","Fur"]` and
completeness 0.857. A correction that only makes the semantic gate fire again rebuilds the same
single point of failure.

### C-073 — F-096: false real identifiers on legs reported PASS

7 emitted, all on `PASS` legs. Verbatim from the scorer:

| paper / leg | entity | forbidden kind | accessions attached |
|---|---|---|---|
| PMC12856317 strict + research | `Pyridoxal 5'-phosphate` | `cofactor_as_protein` | drugbank DB00114, hmdb HMDB0001491, kegg C00018, chebi 18405, pubchem 1051 |
| PMC12180156 research | `succinyl-CoA` | `placeholder_product` | hmdb HMDB0001022, kegg C00091, chebi 15380, pubchem 439161 |
| PMC12782028 research | `SREBF1`, `SREBF2` | `regulator_as_metabolite` | uniprot P36956, Q12772 |
| PMC12782028 research | `LIPA`, `LBR` | `heading_or_prose` | uniprot P38571, Q14739 |

`PMC12180156/research` is the sharpest: the gold calls `succinyl-CoA` a **HALLUCINATION TEST** —
zero occurrences in the entire 67,304-character source — and the pipeline emitted it with four real
database accessions.

Second-order defect on the same paper: `drugbank:db00114` is claimed by **two differently-named
entities**, `ALAS2` and `Pyridoxal 5'-phosphate` (`accession_claimed_by_multiple_entities`).

⚠ **The count is a FLOOR, not a measurement.** 8 of the 20 legs were scored from
`merged_payload.json`, which is pre-mapping and carries no accessions at all. The true total may be
higher; it cannot be lower.

**Optional third card, worth considering before T-105:** the acceptance report's own recommendation
is *"run the batch with the updated driver so `final_mapped.json` is persisted on failure paths."*
Doing that converts those 8 floors into measurements and makes T-105's priority-1 number
trustworthy. It is a measurement-quality change, not a correction — scope and label it as such
(new capability, G9 clause 4, no fabricated base failure).

### Merge rules — all still bind

No patch merges unless: its dependency is merged; the diff stays inside the assigned
file/function boundary; focused tests pass; existing affected tests pass or a pinned baseline moved
with an exact documented delta; **an independent reviewer approved the actual diff, not the
report**; it does not weaken a biological gate to increase PWML production; it preserves
incomplete-but-correct pathways as `review_required` rather than dropping them; exporters do not
repair biology after the canonical graph is frozen; **G9** holds; the integration smoke suite
passes; and the **test-process lifecycle** holds with zero surviving owned processes.

**You do not implement and you never approve your own work.**

⚠ **Smoke-count discrepancy, unresolved — check before quoting either number.** `CLAUDE.md` merge
rule 10 says *"465 tests, ~40 s"*. `TEST_MATRIX.md` § "Baseline to preserve" says **473**
(`457` at INIT-001 → 460 C-010 → 465 C-054 → **473 C-067**, each an exact documented delta). The
473 lineage is explicit and C-067 is merged, so **473 is very likely current and `CLAUDE.md` is
stale by one delta** — but measure it before treating a 473 as a regression against 465.

⚠ **`pwml-implementer` is unusable in this environment.** Create the worktree yourself and dispatch
`general-purpose` with the card prompt. `pwml-reviewer` (no edit tools) and `pwml-bio-auditor`
(read-only) are fine.

---

## 4. F-095 / D-062 — real work, but NOT in T-105's chain

D-062 is a locked product-owner ruling:

> A Stage-0 organism conflict whose reading is **correct** preserves the pathway as
> `review_required` carrying the **OBSERVED** organism, with the requested scope recorded
> alongside. It neither exports strict under the wrong request nor drops the run.

The reasoning was merge rule 7: `scope_conflict` currently folds to `STATUS_INELIGIBLE`, whose own
definition reads *"not even a run: nothing was attempted, so nothing failed"* — untrue for
PMC12657337 and PMC12421875, which cleared the gold's connected-core floor (4 vs 3, and 10 vs 7,
both at 100% enzyme and metabolite recall) before being discarded.

**Sequence it whenever you like. T-105 does not wait on it** — `review_required` is not a strict
export, so it moves no rate either way. If you do card it:

- The Stage-0 guard is **not** weakened. Its reading is correct and stays. Only the *disposition*
  of a correct reading changes.
- **Do not resolve it by editing `topics_t104.txt`.** Supplying the actual organism removes the
  trap by handing the pipeline the answer and makes `forbidden_organisms` unexercisable.
- D-062 explicitly leaves open whether the gold's `expected_export: strict_exportable` for those two
  papers survives. That reconciliation is a **separate product-owner decision** — do not take it
  yourself.

---

## 5. Then run T-105

**Only after C-072 and C-073 are merged.** Same 20 pinned legs, same command as T-104, so the two
release candidates are comparable:

```
# pre-flight first — free, zero LLM legs by construction
<wrapper --label t105-preflight --timeout 900 --heavy-lock t105-pre> -- \
  .venv/Scripts/python.exe -u scripts/batch_run.py --topics topics_t104.txt \
    --out runs_verify --modes strict,research --timeout 1800 --deadline 8 --fresh --stage-only

# then the real run — background, drop --fresh so it continues the verified plan
<wrapper --label t105-rc2 --timeout 32400 --heavy-lock t105-rc2> -- \
  .venv/Scripts/python.exe -u scripts/batch_run.py --topics topics_t104.txt \
    --out runs_verify --modes strict,research --timeout 1800 --deadline 8
```

- **Reuse `topics_t104.txt`.** It is committed and its 10 scope lines are verbatim from
  `bench/gold/pinned_v1.json`. Confirm with
  `bench_acceptance.py --verify-plan <run dir>` → expect `verdict: OK`, all 10 `[pinned_override]`.
  (`--write-topics` regenerates the same file if you ever need it.)
- **`T2PW_SPECIES_LLM=0` must NOT be set.** That is T-103-only — PACK 9 RULING 3.
  `T2PW_OFFLINE_CURATOR` must also stay unset: an RC runs the real curator.
- T-104 took **5.44 h** for 20 legs. The `--deadline 8` is comfortable but not enormous; the
  deadline is checked *between* legs, so a timeout leaves legs **pending, not corrupted**, and
  re-running without `--fresh` resumes the same directory.
- Score it: `bench_acceptance.py --run-dir <dir> --out <txt> --json <json>`, through the wrapper.

**T-105's acceptance is "remaining failures explained and classified"** — every surviving failure
gets exactly one of `product_contract_violation`, `gold_data_defect`, `policy_disagreement`, citing
the gold `relevance_note` / `export_rationale`. **A benchmark failure does not by itself justify a
code change.**

### What T-105 should look like — predict before you run, so you can tell signal from noise

Of the 20 legs, **10 are expected to behave exactly as they did at T-104 regardless of what you
merge.** Do not spend triage time rediscovering them:

| legs | expected at T-105 | why |
|---|---|---|
| 6 × `scope_conflict` (PMC12657337, PMC12421875, PMC12312563) | **identical**, unless you card D-062 | F-095 is a policy disagreement; nothing in C-072/C-073 touches Stage-0 disposition |
| 2 × TIMEOUT (PMC12444477 both modes) | **likely identical** — 1800 s, `budget_exhausted`, "produced nothing" | F-092 is open and unowned; this paper is the run's sole extraction blocker |
| 2 × `no_reactions` (PMC13231680 both modes) | **identical, and CORRECT** | declared negative control; gold calls an empty pathway plus a rejection reason the right outcome (see F-097) |

**So the real T-105 signal is confined to the 10 legs that passed at T-104**, plus whatever
C-072/C-073 move. If a scope_conflict count of 6 or a no_reactions count of 2 appears at T-105,
that is the expected baseline, **not a new finding**.

Two specific things to check first, because they are what the corrections are for:

1. **PMC12452463/strict** must reach `review_required` with `strict_acceptance_eligible=false`, and
   must emit `pathway.review_required.pwml` — **not** a bare `pathway.pwml`. Comparing the strict
   artifact filenames across papers is the fastest read on C-072.
2. **Priority 1 must be 0.** If it is non-zero, check whether the surviving rows came from
   `final_mapped.json` (a real measurement) or `merged_payload.json` (a floor) before classifying.

---

## 6. Facts you need — do NOT re-derive

**Models.** All nine `OPENROUTER_*_MODEL` slots are `deepseek/deepseek-v4-flash` (authorized).
`openrouter/free` cannot drive Stage 1 — *"Chunk 1 failed to produce valid JSON"*. Do not switch back.

**Cost.** Account is NOT free-tier: $75 limit, ~$71.81 remaining before T-104. deepseek-v4-flash is
$0.08/1M prompt, $0.16/1M completion. T-104's 20 legs cost well under $1. Any fallback to a
non-free model spends real money (D-058); none is configured.

**LM Studio** must be running for RAG legs: `http://127.0.0.1:1234/v1`,
`text-embedding-nomic-embed-text-v1.5` (768-dim). Verify it responds before a multi-hour run.

**Milestones — do not re-litigate:**

| milestone | status |
|---|---|
| T-100 | acceptance met (TRAP-1 satisfied by the T-103 run) |
| T-101 | `MEASURED, NOT PASS` — clause 1 violated, 2 unexercised, 3 inconsistent (F-092) |
| T-102 | `MEASURED` — organism/SBML axis structurally unreachable (F-009). Never PASS |
| T-103 | `MEASURED` — acceptance satisfied at `round_count=1`; multi-round re-entry UNTESTED |
| **T-104** | **`MEASURED — NOT ACCEPTED`**, `runs_verify/2026-08-21_2239`, `2673067`. `COMPLETE (10/10 papers, 20/20 legs)` — first complete benchmark of the sprint. Priorities 1/4/5 FAIL, 2/3 PASS |

**T-104's numbers, for comparison at T-105:** false real identifiers **7** (floor); coverage
**0/7**; strict PWML **0/4**; extraction success **7/8 = 88%**; research deliverable **4/8 = 50%**;
semantic confirmation **0/7** and **0/8**. Legs: 10 PASS, 6 `scope_conflict`, 2 `no_reactions`,
2 TIMEOUT.

**Closed, do not reopen:** F-062, F-079, F-084, F-091, F-093.

**Open and unowned:**

- **F-092** (HIGH) — identical wall-clock timeouts record different terminal reasons, never
  `operation_timeout`; the runner hard-codes *"produced nothing"*. **Re-confirmed on both
  PMC12444477 legs at T-104.** If T-105 legs time out, record it — do not fix it mid-milestone.
- **F-090** (MEDIUM) — `bounded_run.py` writes every descendant PID into its report; `g11_evidence.py`
  caps reports at 64 KiB. **Did not bite at T-104** (45 descendants fit). Only threatens base-tree
  exports.
- **F-097** (LOW) — `SUMMARY.txt` files the negative control PMC13231680 under "RESEARCH-MODE
  DEFECT" and says fix it first, but an empty pathway plus a rejection reason is exactly what the
  gold calls correct for that paper. Reporting-side only. **Do not let it drive a pipeline change.**
- F-086/087/088/089 — minor, recorded, unowned.

**Standing pre-charged failures — measured, do not re-derive; re-measure any you depend on:**

| entry | measured |
|---|---|
| `test_strict_failure_replay.py` | **2 failed, 37 passed, 8 skipped** — both the `only_unrelated_reactions_survive` parameterisation |
| `test_batch_preflight.py` | **0 failed, 37 passed** post-C-069. The register's "2" was a *worktree* number: `:480` asserts `venv is not None` and `git worktree add` does not copy `.venv` |
| `.env`-conditional family | **7 failed, 50 passed** — 4 `test_prefreeze_third_export_seam.py`, 1 `test_prefreeze_species_resolution.py`, 1 `test_pwml_writer.py` (F-065), 1 `test_canonicalization_preflight_and_species.py` |
| `qb` node15 | fails when PathBank reachable — **confirmed** in the full Chunk D gate |

**Identical legs give materially different Stage-1 draws at temperature 0.** Re-run a leg before
calling a single-leg change a regression.

---

## 7. Recording results

- Commit run artifacts on the established convention: everything under the run dir **EXCEPT**
  `cache_snapshot/` (43 MB of T-104's 52 MB).
- Use `git commit -F <file>`; **never** a long inline `-m` (PowerShell here-docs silently no-op here).
- Appending prose to `FINDINGS.md` / `DECISIONS.md`: **do not use a bash heredoc** — it fails on
  apostrophes even with a quoted `<<'EOF'`. Write the block to the scratchpad, then
  `cat <scratch> >> <target>`. For in-place row edits use a `python - <<'PYEOF'` block with
  `io.open(..., encoding='utf-8')`, `.replace(old, new, 1)`, `newline=''`, and `assert` the anchor
  is unique first.
- Stage explicit paths, inspect `git diff --cached --name-only`, push to `sprint/pwml-recovery`,
  verify local = origin = `git ls-remote`. **Never merge to `main`.**
- Record T-105's status in its `TEST_MATRIX.md` row **in place**, keeping 578 lines (or record the
  new count in D-061). If acceptance is partly unexercised, say `MEASURED` with the qualifier —
  **never `PASS`**. A milestone that finds a real defect has done its job; recording it as PASS
  discards exactly that.

---

## 8. When to stop and ask

Two exhausted correction rounds; a locked contract interpretation or an append-only `DECISIONS.md`
entry; the gold-vs-D-062 reconciliation in §4; a blocker preventing all remaining work.
Otherwise proceed on your own judgement.

**Before finishing:** confirm zero sprint-owned Python processes, heavy lock free, no merge in
progress, product-owner edit still 35/2 at `sha256:e50a248bb7189c22…`, everything pushed.
