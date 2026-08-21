# RESUME — next session handoff

**Written by the Lead Orchestrator, 2026-08-21, PACK 11.**

> **⚠ Why this file is in the repo and not in a scratchpad.** Prior sessions wrote their
> handoff and their product-owner-edit backups to a *session-local* scratchpad directory.
> This session was told backups existed at `scratchpad/product-owner-edit/` and **could not
> find them** — that path does not exist in the repo, and the previous session's temp
> directory is not addressable from here. A handoff that the next session cannot find is not
> a handoff. **Keep this file in the repo and update it in place.**

---

## 1. Integration state

| | |
|---|---|
| Branch | `sprint/pwml-recovery` |
| Session start tip | `e616846de75e2098e3fb76592665955b3cfe3bbc` |
| **Current tip** | `81b8c3ea56d73ad7c28b9e4c4b871e12e3c6dc78` — pushed, `local = origin = ls-remote` verified after every push |
| Cards merged this session | **C-070** (`09f7156`) |
| Merges to `main` | **none, and none permitted** |

## 2. Baselines re-measured at tip — use these, do not re-derive

All through `bounded_run.py` with the wrapper-owned heavy mutex. Every job:
`FINAL SURVIVING COUNT: 0`, `cleanup: success`.

**Primary checkout (carries `.env`):**

| gate | result | evidence |
|---|---|---|
| SMOKE | exit 0, 53.43 s | `g11/INTEG-069/01` |
| Chunk E | **174 passed** | `g11/INTEG-069/02` |
| Chunk D full | **core 159/160**, **node15 failed**, `jobs=28`, `additions=0` | `g11/INTEG-069/03` + `g11/INTEG-070/*` |

**Chunk D's two failures are the documented `.env`-conditional baseline** (one core red) **plus
the pre-charged `qb` node15** (fires whenever PathBank is reachable). Neither is new.

**Worktrees carry no `.env`**, so the same gate runs fully green there — measured on C-070:
`executed=187/187, omissions=0, additions=0, failed=none`. **Both baselines are now confirmed
by direct measurement, and the conditionality is the only difference between them.**

## 3. Cards in flight — exact tips

| card | branch | worktree | exact tip | state |
|---|---|---|---|---|
| **C-070** (F-066) | `agent/c070-isolated-collection` | `C:/t/c070` | `5bc600e` | ✅ **MERGED `09f7156`** — bare `APPROVE` from REV-070, zero correction rounds |
| **C-069** (F-073 + F-086) | `agent/c069-child-imports` | `C:/t/c069` | `86d5807` | **CORRECTION round 1 in flight.** REV-069 returned CORRECTION, not REJECT; orchestrator verified it against source before spending the round |
| **C-071** (F-079) | `agent/c071-actor-span-gate` | `C:/t/c071` | — | in flight, base `f2a959f` |

### C-070 — ACCEPTED and MERGED

Bare unsuffixed `APPROVE`, **zero correction rounds**. Orchestrator-run heavy gates, all on
the pin: SMOKE **473**, Chunk A **134**, Chunk E **174**, Chunk D **187/187 `failed=none`**.
Post-merge at integration: SMOKE + the new file = **475 passed, 1 skipped** (473 unchanged +
2 routine arms; the 94 s sweep correctly skips behind `T2PW_ISOLATED_COLLECT_ALL=1`).

**REV-070 verified the one genuinely risky claim by measurement rather than argument** — that
`pythonpath = src` at `sys.path[0]` could weaken the G11 tree pin. It cannot, and the finding
inverts: with the ini **off**, plain `pytest` silently imported `t2pw` from whatever
`PYTHONPATH` named; with it **on**, rootdir's own `src` wins. The change enforces the very
property `tree_pin.py` exists to enforce.

**F-066 is closed, and closing it refuted two of its own claims** — the "re-pin everything"
characterization (for this remedy only) and the 21-file exposure list (wrong in both
directions; its count is right by coincidence). Both left standing in the record, annotated.

### C-069 — correction round 1, and what it is

REV-069 found that **two of the three new `CHILD_IMPORTS` reason strings assert a failure mode
that measurably cannot happen** — `PreflightProblem.why` text read by an operator at 2am, and
a static-read assertion of runtime behaviour, which § S5 forbids in terms. Verified against
source before the round was spent:

* `streamlit_app.py:52` imports `strict_quarantine` at module scope → the child dies executing
  the app script, so *"still writes all four reports"* is false.
* `extraction_ladder.py:61` → `pipeline.py:32` makes `deadline` a module-scope dependency of
  the extraction pipeline → **every** leg dies at import, not *"exactly the night's slowest
  legs"*.
* The third string, `release_status`, was ruled **accurate** and must not change.

**One round remains after this one.** Everything else in the card reproduced exactly,
including its self-correction against its own interest, and its evidence instruments were
ruled **in-boundary**.

## 4. What must happen next, in order

1. **REV-069 and REV-070 verdicts.** Merge only on a bare, unsuffixed `APPROVE`. Verify any
   evidence-backed rejection against the correct base before spending a correction round.
2. **Merge serially with `--no-ff`.** Neither card touches `src/t2pw/app/streamlit_app.py`, so
   **no stash is required** — verify the file list first and skip the stash dance.
3. **Post-merge:** SMOKE, then G11 `check`, then push and verify local = origin = `ls-remote`.
4. **On merging C-069, strike ONE test from the standing pre-charge list** — see § 6.
5. **C-071's review and merge.** Its merge is expected to be **HELD** pending Decision 2 in
   the bundle below (the `SEMANTIC_GATING_CHECKS` 4 → 5 ratification). Implement-and-hold is
   deliberate: a one-line answer then unblocks the merge rather than starting the work.

## 5. The product-owner checkpoint — FOUR decisions, prepared and bundled

Full text with exact recommended wording:
`<session-scratchpad>/sprint-records/DECISION-BUNDLE.md`. Summarised here so it survives.

| # | decision | recommendation | what it unblocks |
|---|---|---|---|
| **1** | *"After the index fix"* (`PRODUCT_CONTRACT.md:341`) refers to C-010, merged as **`72ee20f`** | **Ratify** | T-104's acceptance row becomes quotable; F-062 closes with no card |
| **2** | `round_cap_reached` as an eighth termination reason, precedence rank 8, **outside** `OPERATIONAL_TERMINATION_REASONS` | **Approve as drafted** | closes C-064's loose end |
| **3** | T-101 + T-103 live-run authorization, ~3.8 h, ~$0 | **Authorize**, plus one free `GET /api/v1/key` | both milestones start immediately |
| **4** | **NEW.** Does `PRODUCT_CONTRACT.md` §3's *"whether it was paper-explicit"* require the claim to be **verified** or merely **recorded**? | **Recorded** — close F-078 Half B with the residual documented | F-078 |
| **5** | **NEW.** Ratify `SEMANTIC_GATING_CHECKS` **4 → 5**, the one named addition C-071 makes | **Ratify on delivery**, once C-071's diff and evidence are in hand | **C-071's merge, which is HELD on this** |

> **⚠ Correction to an earlier draft of this file.** It said C-071's merge was held on
> Decision 2. **It is not** — Decision 2 is `round_cap_reached`, a RAG-loop termination reason
> with nothing to do with semantic gating. C-071's hold is its own ratification and is now
> **Decision 5**. Two unrelated ratifications were conflated; recorded rather than silently
> renumbered, because a handoff that quietly changes what a decision meant is worse than one
> that admits it got it wrong.

**⚠ Decision 1 carries a corrected SHA.** F-080 and the takeover brief both say C-010 merged
at `9e06360`. **It did not** — that is C-010's *base*, and C-012's merge. C-010 merged at
**`72ee20f`**. Registered as **F-085**. `DECISIONS.md` is append-only, so ratifying the old
wording would have made a false fact permanent.

## 6. Standing pre-charged failures — THE WHOLE REGISTER IS NOW MEASURED

**Every entry was re-measured at integration `e616846` in the primary checkout this
session.** Three confirmed, one corrected. Do not re-derive these; do re-measure any you are
about to depend on.

| entry | register said | **measured at integration** | status |
|---|---|---|---|
| `test_strict_failure_replay.py` | 2 | **2 failed, 37 passed, 8 skipped** — both the `only_unrelated_reactions_survive` parameterisation | ✔ **confirmed** |
| `test_batch_preflight.py` | 2 | **1 failed, 35 passed** | ✘ **CORRECTED — see below** |
| `.env`-conditional family | 7 | **7 failed, 50 passed**, and the file breakdown matches exactly: 4 in `test_prefreeze_third_export_seam.py`, 1 in `test_prefreeze_species_resolution.py`, 1 in `test_pwml_writer.py` (F-065), 1 in `test_canonicalization_preflight_and_species.py` | ✔ **confirmed** |
| `qb` node15 | fails when PathBank reachable | **failed** in the full Chunk D gate | ✔ **confirmed** |

### ⚠ The `test_batch_preflight.py` correction

The register's **2** is a **worktree number, not an integration number.**
`tests/test_batch_preflight.py:480` asserts `venv is not None` — *"this project ships a
`.venv`; the test assumes it"*. `git worktree add` does not copy `.venv` (it is untracked), so
that assertion fires in every agent worktree, and four further tests gated on
`if runner.venv_python() is None:` (`:584`) **skip** there and **pass** here.

| | worktree | primary checkout |
|---|---|---|
| failed | 2 | **1** |
| passed | 30 | **35** |
| skipped | 4 | 0 |

**So on the C-069 merge this entry does not go 2 → 1. It goes 2 → 0, and the 2 was never
right for integration.** Post-merge expectation, stated before the merge so it is a
prediction and not a rationalisation: **37 passed, 0 failed.**

> **My first stated prediction was 36, and it was wrong.** I subtracted the failure without
> adding the test the card introduces. REV-069 measured the real figure by junctioning the
> primary's `.venv` into a base worktree carrying the tip's two files: **`37 passed in 7.16s`**,
> against the primary's own `1 failed, 35 passed`. So the delta is `1 failed / 35 passed` →
> `0 failed / 37 passed` — one red closed **and** one new classifier test added. Recorded
> because a prediction is only worth stating if it is also corrected when measured.

**Why it matters beyond one line.** This is the same class of error as F-068 and PACK 11
RULING 1 — a number measured correctly, in the wrong environment, then carried forward as if
environment-free. `FINDINGS.md:2076` already warns about the inverse: filing a genuine
unconditional red under the `.env` family is *"how a real signal gets permanently silenced."*
This is that warning running the other way — **a worktree artifact filed as a real red
inflates the register and is how a genuine new red later gets waved through as expected.**

**Standing guard: a pre-charged failure should record which environment it was measured in.**
`.env`-dependent reds are tracked carefully; `.venv`-dependent ones were not tracked at all.

**The C-069 baseline delta to cite is the CORRECTED one** — the author's first report was
wrong in its own favour and it corrected itself:

```
                          first claim      ACTUAL
CHILD_IMPORTS entries       5 -> 7          5 -> 8
missed, occurrences         6 -> 0          7 -> 0
missed, distinct modules    2 -> 0          3 -> 0
```

The base guard *reported* 6/2; what was *actually blind* was 7/3. The difference is exactly the
module it could not see (F-086).

## 7. Findings registered this session

| id | severity | disposition |
|---|---|---|
| **F-084** | LOW | **NOT a defect — disproved offline.** Registered *and closed* so it is not re-investigated. Carries an unreachable latent sub-finding whose safeguard is a property of `openai`'s internals, so an upgrade could expose it. |
| **F-085** | MEDIUM | The C-010 SHA error. Caught before ratification. |
| **F-086** | MEDIUM | Preflight detector discards submodule names. **Assigned to C-069**, ceiling raised 400 → 650, no correction round consumed. |
| **F-087** | LOW | `runner.py:1341`'s cited measurement went stale. No card; should ride along with the next card owning that file. |

## 8. Findings dispositioned WITHOUT a card — do not re-open casually

* **F-062** — **no code card required.** The routing seam is byte-identical at tip, so F-062
  read the mechanism correctly; but its remedy was refused on evidence by F-081, the correct
  repair merged as C-067, and the four remaining structural reasons are each ruled
  `keep_refusing`, so the unconditional append is now **correct**. The confirming measurement
  is **T-104** and cannot be done offline — the quarantine input payload is not persisted and
  neither committed file matches `admitted_payload_hash`.
  **Carry F-081's own MEDIUM caveat into T-104 triage:** *"If the flagged row's synonym set is
  disjoint from `keep_norms`, the theorem is wrong and there is a third divergence not yet
  found."*
* **F-077** — reassessed against current source; classification **holds**. `schema_version` is
  still 6, all three prunes and `_revalidate_surviving_processes` still exist, both scopes
  still pinned, §3 still binds *"the final pathway"*. **Accepted deliberate residual. No card.**
* **F-053** — **remains UNDISCHARGED and stays in force.** No question is being put about it.
  F-079 is fresh evidence it should stay.

## 9. Control-plane corrections made / still owed

**Made:** F-078 and F-079's *"not measured against current source"* scope notes are superseded
in place (not deleted); `LEDGER.md:231`'s false C-056c row is corrected in the PACK 11 record.

**Still owed** (reported by C-070, not yet applied):
* **F-066's file list is stale by membership.** It names four files that now collect alone
  fine and misses four that fail. **23 files never mention `sys.path` but only 21 actually
  fail** — 5 of the 23 pass anyway, and 3 files that *do* mention `sys.path` still fail.
  **No static predicate separates the two sets**, which is why C-070's test is a real sweep.
* **`FINDINGS.md:1742-1745`'s *"SMOKE and all four chunks would have to be re-pinned"* is
  refuted for the `pythonpath = src` remedy** — measured, nothing moved. C-070 makes no claim
  about the other two remedies.
* **`TEST_MATRIX.md` § Chunks needs an isolated-collection entry.** C-070 supplied the exact
  prose. **`TEST_MATRIX.md` is 541 lines with citations pinned through line 477 — patch in
  place and preserve the line count.**

## 10. Process state

* **Heavy mutex:** free. `bounded_run.py` owns it via `--heavy-lock <holder>`; hand-written
  `mkdir C:\t\heavylock` and unconditional deletion are forbidden.
* **Sprint-owned Python processes:** zero. The only survivors on this machine are the **two
  protected `ms-python.isort` LSP servers** belonging to the product owner's IDE. Never kill
  Python by executable name.
* **G11:** clean at every checkpoint. Whole-tree `check` at session start: **3,096 artifacts,
  0 non-compliant, exit 0.**
* **Product-owner edit:** intact at **35 insertions / 2 deletions**,
  `sha256:e50a248bb7189c222896f74bc38cdbd1c6dbbc6dc3a2594b3e5e63ea261416e0`. Fresh backups
  (patch + working copy + HEAD baseline, all verified byte-identical) are in this session's
  scratchpad under `sprint-records/../product-owner-edit/`. **Neither in-flight card touches
  that file.**

## 11. ⚠ Credential hygiene — action recommended

While verifying that `OPENROUTER_API_KEY` was present and well-formed for the T-101/T-103
authorization, a malformed `sed` fallback **printed two `.env` credential values into this
session's transcript** — the OpenRouter key and the NCBI key. They were **not** written to any
file, commit, report or message, and `.env` is untracked and gitignored twice
(`.gitignore:1,3`), so nothing left the machine.

**Recommend rotating both keys** as hygiene, since they now sit in a session transcript.

For future checks: `python-dotenv` strips whitespace around `=`, so
`OPENROUTER_API_KEY = <value>` parses correctly. **Verify presence with
`dotenv_values()` and print only the length** — never a raw grep whose fallback can echo the
line.

## 12. Milestones

| | state |
|---|---|
| **T-102** | complete. Its only legitimate status is `MEASURED — organism/SBML axis structurally unreachable (F-009)`. **Never record it as PASS.** |
| **T-101 / T-103** | **ready except for authorization.** `topics_t101.txt` and `topics_t103.txt` are created and committed. Nothing has been run and no money spent. |
| **T-104 / T-105** | blocked on Decision 1. **Two separate ~7 h release candidates with a triage pass between them — never collapse them into one run.** |

**Correction to the T-103 command in `T101_T103_AUTHORIZATION.md:169-179`, measured:**
`bounded_run.py` has **no `--env` flag**, and the child inherits the wrapper's environment. Use
the **shell-prefix** form `T2PW_SPECIES_LLM=0 T2PW_OFFLINE_CURATOR=1 <py> bounded_run.py ...`,
not `env VAR=x` after the `--`. Verified by execution: the child saw `SPECIES_LLM='0'` and
`OFFLINE_CURATOR='1'`, exit 0, zero survivors.

**Do not claim sprint completion when the card queue empties.** The sprint completes only
after the milestone chain is measured, triaged, corrected and rerun per `MASTER_PLAN.md`.
