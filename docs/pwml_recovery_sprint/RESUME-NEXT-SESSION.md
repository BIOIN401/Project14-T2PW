# RESUME — next session handoff

**Rewritten in place by the Lead Orchestrator, 2026-08-31, at the close of the post-triage wave
(`ORCH-717`).** Supersedes the 2026-08-29 record, which was written at the close of the T-107
triage wave and is stale in two sections: it says the instruments are the next job (they are now
fixed and merged), and it quotes **SMOKE 473**, which **moved to 503 under merge rule 4**.
Previous content is in git history. **`LEDGER.md` remains the single source of truth for task
state.**

> **⚠ Keep this file in the repo and update it in place.** A G11 report certifies a job was clean
> and preserves **nothing** about what it found. Commit the probe and its log, not just the report.

---

## 0. LIVE WAVE — ORCH-717 continuation, updated in place

**Read this section first. It is newer than everything below it**, and sections 5 and 6 are now
partly superseded: two more cards were chartered and two of the three held questions have been ruled.

**This is the THIRD Lead Orchestrator on this branch inside about an hour** (`-b1`, then `-ab`, now
this one). Both predecessors vanished mid-wave without writing a handoff. **That is why this section
is checkpointed early rather than at the usual context threshold** — assume the same could happen
again and leave the record recoverable at all times.

### Cards in flight

| Card | Scope | Worktree | State |
|---|---|---|---|
| **C-108** | F-155, all five members, `curation/apply_audit_patch.py` + `tests/test_c108_f155_class.py` | `C:/t/c108`, base `C:/t/c108base` | **implementer dispatched** |
| **C-109** | F-153 remainder, F-154, reviewer-evidence route | `C:/t/c109` | **implementer dispatched** |
| **C-110** | Q1 negative-control status (acceptance instrument) | — | **chartered, NOT dispatched** |
| **C-111** | F-148 observability — instruments, fixes nothing | — | **chartered, NOT dispatched** |

`REV-108.md` criteria were **fixed and committed before the C-108 diff existed**. There is no
REV-109/110/111 yet — write each before its diff exists, not after.

**Both dispatched agents were observed mid-work, not idle.** C-108 had eleven probe artifacts and no
commit; C-109 had nothing on disk yet because it was still reading. **A subagent reading leaves no
process and no artifact — check the worktree before concluding anything.**

### Baselines I measured MYSELF at `f67e00a` — reuse these, do not re-derive

```
TOTALS  battery=0/29  F146=REJECTED  C1=0 C2=0 C3=0 C4=0 C5=1 C6=0
ROWS: 692  ACCEPTED: 401  REFUSED: 291
```

**`C5=1` IS F-155 member (a).** The 29-case battery already encodes
`'add P as a transporter to resolve the structural inconsistency'` as **want REFUSE**, and the
shipped code **ACCEPTs** it — a committed, behavioural, already-executing **G9 base failure**. C-108
succeeds on (a) when `C5` goes `1 → 0` with `battery=0/29` and `F146=REJECTED` unmoved.

Evidence: `evidence/orch717_baseline_battery_f67e00a.log`, `orch717_baseline_corpus_f67e00a.log`,
G11 `ORCH-717/13`, `ORCH-717/14`.

### Rulings made this wave — `RULINGS-ORCH717.md`

- **Q1 RULED (product owner).** A gold-designated negative control **passes** when it releases no
  reactions, gives the required rejection reason, and the emptiness is **not** from timeout, crash,
  missing artifact or infrastructure failure → **`PASS_NEGATIVE_CONTROL`**. The scorer **cannot**
  represent this today, so **C-110 exists**. The predicate already does:
  **`_empty_is_correct(case)` at `acceptance.py:1530`**. The misleading token is emitted at
  **`batch/runner.py:717`, which has no `GoldCase` in scope at all.**
- **Q3 RULED (Lead): a `pathbank_compound_id` is NOT a real accession. No code change.**
  `_external_ids` recognises only `uniprot, drugbank, hmdb, kegg, chebi, pubchem`. **Q3 could never
  have moved Q2's arithmetic** — the affected row carries **five** recognised accessions with the
  PathBank id removed entirely, and the Priority-1 branch only asks `if ids:`. The conditional policy
  the guardrails describe is **not implementable**: the whole local id space is bare small integers
  over a 55-row table, produced by an offline **name-index** match with
  `chosen_rule = legacy_pathwhiz_id_unverified`.
- **Q2 half 1 is unblocked** and goes to Wave 4's conditional authority (independent reviewer + the
  four-step A/B). **Half 2 — whether `supported_reactions_complete` should be set on any case — is
  the ONE open product-owner question.** It changes what Priority 2 *means* on every future run.

### T-108 — `NO-GO`. See `T108-READINESS.md`

**The row most likely to sink a green table is operational, not code:** T-107's ceiling was halved
`3600 → 1800` with `leg_timeout_override_reason` and `_source` **both empty**, and the slowest leg
that finished used **92.1%** of it. At that budget three timeouts is the expected outcome.
`PMC12096016` is **one of only two strict-denominator papers** and was **lost to the clock, not to
biology** — so Priority 5 can read `0/2` again for a reason that is not a pipeline defect.
**Choose the ceiling deliberately and record the reason in the manifest BEFORE launch.**

**T-108 must run from the PRIMARY checkout.** `.env` is untracked, so a worktree silently gets
`LLM_PROVIDER=local` — calls 400, exception swallowed, curator a no-op **by accident** — while the
primary issues real billed calls. **A green cohort obtained in a worktree does not certify the
primary.** Confirmed present: `LLM_PROVIDER=openrouter`, `LLM_TEMPERATURE=0`, full `OPENROUTER_*`
model set, `LLM_MAX_RETRIES=3`.

### Peer

`project14-t2pw-93` is live, read-only, and replied **"no claims"** — no branch, worktree, lock, job
or edit. It warned that one of its own earlier reports to a predecessor was a **torn `git status`**
read taken mid-commit by another session. **Re-derive anything you intend to act on.** I did: the
ancestry from `c7fb5c5` to the tip is fully accounted for.

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
| **F-154** | `product_contract_violation` | `pwml-test-runner.md:59` sends that agent to `TEST_MATRIX.md:213-218` for a **stem-exact** chunk match; `:213-218` is the bounded-runner **function** table. Real locations `:230-237` and `:259-271`. **Registered, not fixed** — correct values are in the finding |
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
