# RESUME — next session handoff

**Updated in place by the Lead Orchestrator, 2026-09-02, during `ORCH-719` (the D-088 correction wave).
§ 0 is the current state; § 0-prev0 is the T-108 execution wave it supersedes, and § 0-prev and
everything after are earlier waves, superseded where they disagree.** The paragraph below describes the 2026-08-31 rewrite and is kept as that wave's record.

**Rewritten in place by the Lead Orchestrator, 2026-08-31, at the close of the post-triage wave
(`ORCH-717`).** Supersedes the 2026-08-29 record, which was written at the close of the T-107
triage wave and is stale in two sections: it says the instruments are the next job (they are now
fixed and merged), and it quotes **SMOKE 473**, which **moved to 503 under merge rule 4**.
Previous content is in git history. **`LEDGER.md` remains the single source of truth for task
state.**

> **⚠ Keep this file in the repo and update it in place.** A G11 report certifies a job was clean
> and preserves **nothing** about what it found. Commit the probe and its log, not just the report.

---

## 0. CURRENT — **`ORCH-719`: D-088's correction is DESIGNED and MEASURED, not merged. Merge gate 10 is RED and that is not this wave's doing.** 2026-09-02. Read this section and nothing below it first.

**Integration tip `458a93f1`, pushed and remotely verified: local = `origin/` = `git ls-remote`.
`main` untouched — local `7531692`, remote `03f1af5`.** Everything below § 0 is older and superseded
where it disagrees.

> **A machine crash occurred mid-wave.** Nothing was lost: the crash stranded **no** heavy lock and
> **no** orphaned process (verified after reboot — the only `python.exe` alive are the
> `ms-python.isort` language servers, and **the count is now TWO, not three; match on COMMAND LINE,
> never on count or PID**). Every commit had been pushed. The one casualty was a running reviewer
> agent, restarted.

### The one thing to check before anything else

**SMOKE IS RED ON THE INTEGRATION BRANCH ITSELF — 3 failed / 500 passed — AND HAS BEEN SINCE
`479128b3`.** See **F-171**. The previous handoff's takeover table certified *"SMOKE (22 files) 503
passed, exit 0"*; that row was **already false when it was written**. Measured at tip `9873d064` with
no card's work present, G11 `ORCH-719/10`.

**No merge can satisfy merge rule 10 until this is fixed.** `C-115` is chartered to fix it and is
**not yet dispatched**.

### Wave state — `HANDOFF.md` § 5.2a, step by step

| # | Step | State |
|---|---|---|
| 1 | Charter ONE narrow acceptance/release-policy card | **SPLIT.** Release-policy half is **BLOCKED** with the product owner (`DECISION-PACKET-D088-RUNTIME-CAP.md`); diagnostics half is **C-114**, implemented, **blocked for merge** |
| 2 | Preserve raw anchor diagnostics | **C-114 @ `47b9c517`**, in review (`REV-114`) |
| 3 | No Stage-0 redesign | **Held.** Nothing under `src/t2pw/pipeline/` merged this wave |
| 4 | Curate expected core reactions / subprocesses for the ten papers | **DONE.** 41 reactions, 35 subprocesses, 174 verbatim quotes, independently re-verified |
| 5 | Archived-artifact A/B across all 83 legs, both roots named | **DONE** |
| 6 | Prove the discrimination on the two named legs | **DONE, and it took four passes** — see F-170 |
| 7 | 60 and 90 populations stay separately visible | **C-114 acceptance criterion**, reproduced exactly |
| 8 | Rebuild T-108 readiness | **NOT STARTED** — blocked behind step 1 |
| 9 | Launch a new milestone | **NO-GO. Not launched, not scheduled.** |
| 10 | RAG / LLM evaluation framework | **NOT STARTED** |

### What is established — do not re-litigate

**F-168 — the obvious D-088 implementation is disqualified.** Keying the hard cap to Stage-0's own
`main_subprocesses` gives the required discrimination *perfectly* on the T-108 tree at three stoplist
strengths, and fails the corpus: **0 of 14 paper/mode pairs named a stable subprocess set**, and on
`runs_verify/2026-08-21_2239` it **releases** `PMC12782028/strict`. **The specification is itself an
LLM draw; a gate keyed to it has a random denominator.**

**F-169 — `PMC12096016`'s four unmatched anchors are four different defects.** `ATP` is present as
`Adenosine triphosphate`, **wired into an admitted core reaction**, and reported missing because the
matcher is substring-only and `BIOCHEMICAL_ALIAS_MAP` lacks the spelled-out form. Adjudicated
`product_contract_violation` **against the diagnostic record, not the release status**. The auditor
**corrected two of my four rows**: `NADH` is on that case's own `forbidden_identifiers` as an LDH
assay reporter (dropping it is contract-obeying), `Fur` is preserved in `quarantined_proteins`, and
`EntD`'s quarantine trigger is `apo-EntB`. Census: **12 of 374 anchors are wired-but-falsely-reported,
all ATP on one paper across 8 run trees — chronic, and 3.2%.** The other **362 are genuine**, so this
is **not** an argument against D-088's cofactor ruling.

**F-170 — a named-but-undetailed branch cannot be tested by its NAME.** Four passes; the signal was
not that v1 failed, but that **v3 failed the same way on the same leg after a genuinely correct fix**.
Now tested by curated `member_entities`. **Both required consequences hold on EVERY archived draw:**
`PMC12096016` released 8/8, `PMC12782028` capped 7/7.

**F-171 — merge gate 10 red since `479128b3`.** Above.

### The open product-owner question — nothing proceeds past it

**No permissible reaction-level replacement for the runtime cap exists this wave.** Four candidates,
each rejected for an independent, documented reason; and the acceptance instrument **cannot** route
around it, because Priority 5's numerator requires `strict_acceptance_eligible`, which is
`status == RELEASE_READY` (`release_status.py:1261`), and `acceptance.py:1146` already refuses to
reclassify a frozen record under merge rule 8.

> **Does D-088 clause 2 yield to clause 10 (cap unchanged, `Priority 5` stays `0/2`, diagnostics
> added, curated set built), or clause 10 to clause 2 (cap relaxed on a cofactor vocabulary,
> `PMC12782028` released at runtime)?**

**Lead recommendation: the first**, and it means **the headline number does not move.**

### Immediate next actions, in order

1. **`REV-114` reports** → rule on the C-011 disposition: register an invertible `_DELTAS` entry
   (a card owning that file), **or** re-shape C-114 to emit a separate artifact.
2. **Dispatch `C-115`** (chartered, `prompts/C-115.md`) → re-pin the five census pins → **SMOKE green
   for the first time since `479128b3`**.
3. **Merge C-115, then C-114** once its disposition lands. Both need an independent reviewer on the
   actual diff.
4. **Only then** rebuild T-108 readiness (step 8). **Step 9 stays NO-GO until the instrument is
   merged, gated and remotely verified.**

### Protected, unchanged

**F-147 registered and deliberately UNCHARTERED.** `placeholder_backed_proteins` — escalate only.
**T-107 and T-108 immutable.** `main` untouched. `streamlit_app.py` never committed. Gold blob still
**`36f4b7b690b577f72882c3045ca6728d1ec8d9d1`**. Caches, `topics_*.txt`, the stray `ValueError`,
`out/`, `outputs/`, `tmp/` — never committed.

**Every job this wave ran through `bounded_run.py` with the venv interpreter and `--heavy-lock`.
Eleven G11 reports, `check --task ORCH-719` clean, `FINAL SURVIVING COUNT : 0` and
`cleanup : success` on every one.** `pinned_pytest.py` **exits 98 without `PYTHONPATH`** — hit once,
guarded thereafter.

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
