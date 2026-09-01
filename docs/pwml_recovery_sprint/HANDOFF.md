# PWML RECOVERY SPRINT — HANDOFF after the ORCH-717 engineering wave

You are the **Lead Orchestrator and Integration Authority** for
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, integration branch `sprint/pwml-recovery`.

**Do not merge to `main`.** Work autonomously.

**T-107 has RUN, is SCORED, and is TRIAGED. Its verdict is `NOT ACCEPTED`. Do not re-run it, do not
re-score it, do not reinterpret it.** Three cards merged after it. **None of them re-accepts it.**
A run's verdict is a fact about the artifacts it produced.

---

## 1. Takeover — verify once, do not trust these numbers

**Verified pre-handoff integration state: `ced0ca6a`.** The commit carrying this file is newer.
**The binding invariant is `local = origin/ = git ls-remote` — read it, do not recall it.**

| Check | Expected |
|---|---|
| local = `origin/` = `git ls-remote` | **all three equal.** `ced0ca6a` was the state before this handoff commit |
| `main` | local `7531692`, remote `03f1af5`. Advanced **outside** this sprint. **Touch neither ref** |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python | **zero** |
| IDE processes | two `ms-python.isort` — **never cleanup targets**; **PIDs change, match on command line** |
| `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` |
| caches, `topics_*.txt`, `cache_snapshot/`, stray 0-byte `ValueError` | uncommitted, untouched. **Leave them** |
| **SMOKE** (22 files) | **503 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** |
| **29-case battery** | **`battery=0/29  F146=REJECTED  C1..C6 all 0`** |
| G11 artifacts | **4,969** total; **37** under `ORCH-717` |
| whole-tree G11 | 0 non-compliant (the count is self-referential — reconcile, do not panic) |

Run `ListAgents` and contact every live peer before claiming the branch, the lock or a worktree.
**Prune no worktree.** This wave added `C:/t/c108`, `c108base`, `c108r1`, `c108r2`, `c109`, `c110`,
`rev108`, `rev108base`, `rev109`, `rev109base`, `rev110`, `rev110base`, `orch717mut`.

**Track agent liveness separately from job liveness** — ~15 min without observable progress → status
request; ~30 min with nothing → stalled, interrupt, preserve, redispatch from the last verified
commit. **A subagent *reading* leaves no process and no artifact.** Check the worktree for changed
files before concluding anything; this wave one agent had eleven probe artifacts and no commit while
another had nothing on disk at all, and both were working.

---

## 2. What is DONE — do not re-litigate any of this

**Three cards merged, each independently reviewed against criteria fixed BEFORE its diff existed,
gated at the merged tip, pushed and remotely verified.**

| Card | Merge | Gates pinned | Reviewer evidence |
|---|---|---|---|
| **C-109** control plane | `efb2edc2` | `887395dc` | `4c3577ba` |
| **C-110** negative-control status | `3db2972c` (+ `baa15099`) | `e84747be` | `83c2aa0e` |
| **C-108** F-155, all five members | `2e2a294e` | `ced0ca6a` | `2a7bdd4c` |

**All three reviewers' evidence is reachable from integration and machine-verified so** by
`evidence/reviewer_evidence_route.py`: REV-109 **28/28**, REV-110 **39/39**, REV-108 **104/104**,
zero unreachable.

**Four `src/` files changed in the whole wave:** `bench/acceptance.py`, `bench/render.py`,
`curation/apply_audit_patch.py`, `rag/controller.py` (docstring only, AST-proven inert).
**Nothing under `pipeline/`. The gold file is unmodified.**

### C-108 — F-155 closed, two correction rounds

All five members plus one the card found itself. **`C5: 1 → 0` on integration**, with `battery=0/29`
and `F146=REJECTED` unmoved — the two pinned properties it was forbidden to trade against.

Member **(d)** is why it took two rounds. `_CATALYSIS_CONTRA_RE` was a **bare alias** of the
inhibition cue — one pattern doing two opposite jobs while the comment above it already claimed they
were separate. Round 0 separated them but rebuilt the contra as **a bounded closed list with ACCEPT
as the default** — the handoff lesson the card quotes and *the diff's own comment quotes* — and four
spans where the actor **is** the thing being shut down went REFUSE → ACCEPT. Round 1 inverted the
polarity (agent noun fires the contra; apposition is the only exemption). Round 2 narrowed the
exemption's determiners to `the|a|an`, because `its`/`their` mark the noun as **belonging to** the
actor — the target reading, not apposition.

**The author also fixed the same defect inside CATALYSIS**, beyond the quoted member: `catalys`
matched `catalyst`, making C-105's own `is the catalyst` alternatives redundant since written. It
flagged this for back-out; REV-108 measured that backing it out **reintroduces member (a) in a second
family**. It stays.

**Cost recorded, not hidden:** newly-admitted went 2 → 0. The `Pgp phosphatase` row returns to a base
verdict both reviewer and Lead judge **wrong**. No exemption was built, because *"agent noun
attributive to a non-actor head"* is the same bounded-closed-list mistake in miniature. **Merge rule
6 outranks recovering one row.** Registered R7.

**Registered, not fixed:** R8(ii) (A2 does not require the agent noun to be the HEAD — needs a head
check), R1 (14 words still self-license in the `add P as a <X>` frame, resting on the enzyme-noun
rule that **190+ corpus rows** depend on — the highest-risk surface in the file), R2–R6.

### C-109 — control plane

F-153's `UNWIRED` docstring **retracted, not deleted**; inertness proved three ways (docstring-
stripped AST, marshalled bytecode, `co_code`/`co_names`/`co_consts`). F-154's drifted citations
replaced with **stable anchors, not renumbered lines** — a corrected line address drifts again, which
was the whole finding. New `evidence/reviewer_evidence_route.py` decides reachability **by content**
(git blob id, so `core.autocrlf` cannot cause a false red), with `unreachable_content_differs` as its
own verdict class.

`TEST_MATRIX.md` is a **pure append** (728 → 809, `:477` byte-identical, no differing line in the
common prefix). `FINDINGS.md` edit was **line-neutral** (6919 → 6919).

**Ruled by the Lead: the `465 → 503` correction on `pwml-test-runner.md:52` STANDS** — inside the
declared boundary, a documented SMOKE-count site, mis-instructing the same agent twelve lines above
the instruction the card was ordered to fix.

### C-110 — Q1 implemented

**`PASS_NEGATIVE_CONTROL`**, awarded by `negative_control_outcome(case, leg)`. **Default-deny**, three
affirmative conditions, granted only when `blocked_by` is empty. **`_empty_is_correct` reused
unchanged — no second predicate.** Five populations tested separately; the three that must not pass
each assert their exact withholding code.

**`batch/runner.py` is UNTOUCHED — the stop condition was taken, not built around.** `result_text`
receives a manifest row and a paper dict, never a `GoldCase`, so making `RESULT: FAIL` correct on a
decline would require coupling the live runner to the benchmark gold. **That coupling remains a
reserved architecture decision. It is not chartered.**

**Registered:** the `boundary` reading is redundant against today's `classify_strict_boundary`; 55
`contract`-kind legs that genuinely declined now get `NOT_AWARDED` (both reviewer and Lead judge this
correct — a gate stop is not a statement that the paper contains no releasable chemistry).

---

## 3. Rulings — settled, in `RULINGS-ORCH717.md`

**Q1 — RULED by the product owner, IMPLEMENTED by C-110.** A gold-designated negative control passes
its semantic expectation when it releases no reactions, provides the required rejection reason, and
the emptiness is **not** caused by timeout, crash, missing artifact or infrastructure failure.

**Q3 — RULED by the Lead: a `pathbank_compound_id` is NOT a real accession for Priority 1. NO CODE
CHANGE**, and the status quo satisfies **every** product-owner guardrail without a line of code:

* `_external_ids` recognises exactly `uniprot, drugbank, hmdb, kegg, chebi, pubchem`. PathBank is not
  among them.
* **Q3 could never have moved Q2's arithmetic.** The affected row carries **five** recognised
  accessions with the PathBank id removed **entirely**, and the Priority-1 branch is guarded by
  `if ids:` — which asks whether *any* recognised accession is present. `bool(ids)` is `True` either
  way. **The packet's coupling is dissolved.**
* The conditional-acceptance policy the guardrails describe is **not implementable**: the entire local
  id space is **bare small integers over a 55-row table**, produced by an offline **name-index** match
  with `chosen_rule = legacy_pathwhiz_id_unverified` — precisely the "merely sharing a name fragment"
  case the guardrails exclude.

**Registered, not fixed:** a row carrying **only** a PathBank id reads as *bare* to `_external_ids`,
and D-074's sentinel tolerance turns on bareness. **No live exposure** — that branch is
`UNREACHABLE TODAY` by construction. Re-check if the bareness guard is ever loosened; loosening it to
make it fire stays refused.

**Q2 half 1 — UNBLOCKED and unchanged. This is F-150 and it is the next job.** It no longer waits on
anything. The gold file is untouched.

---

## 4. THE NEXT WORK ORDER — in this order

### 4.1 F-150 half 1 — independently review and resolve

**`prompts/REV-F150.md` criteria are already fixed and committed.** Dispatch an **independent
reviewer**. The product owner's authority is **conditional**: the correction is applied **only if all
eight conditions verify**, and **if one half passes and the other does not, split them and apply only
the verified half.**

**The proposed edit, written and NOT applied:** `src/t2pw/bench/gold/pinned_v1.json`, case
`PMC12180156`, `forbidden_identifiers[0].aliases`: add `"delta-aminolevulinic acid"` and
`"δ-aminolevulinic acid"`. **Add nothing else and move no threshold.**

Both halves re-verified this wave (`evidence/orch717_f150_reverify.log`, G11 `ORCH-717/16`):

```
forbidden_match('5-aminolevulinic acid'    ) -> '5-aminolevulinic acid'
forbidden_match('delta-aminolevulinic acid') -> None
forbidden_match('δ-aminolevulinic acid'    ) -> None
supported_reactions_complete   TRUE=0  FALSE=10  MISSING=0
max_retained_reactions set on exactly two cases — BOTH negative controls
```

**The four-step A/B is mandatory**, and step 3 — scoring T-107's committed artifacts against pre- and
post-edit gold — is an **instrument-sensitivity measurement, NOT a re-score**. No T-107 leg may be
re-run and its verdict may not be restated. **A Priority-1 number that moves because the gold changed
must NEVER be reported as a pipeline regression or improvement.** Recorded prediction: Priority 1
rises **5 → 6**, still `PASS` under D-073 (0–6).

**Commit the gold correction SEPARATELY from any production or scorer change.**

### 4.2 F-150 half 2 — obtain or preserve the product-owner question

**This is the ONE open product-owner question of the sprint:**

> **Should `supported_reactions_complete` be set on any gold case — and if so, which?**

It is the only change that would let Priority 2 measure anything on a paper that is not a negative
control. **It is a decision about what the benchmark MEANS on every future run, not a data
correction.** **Not chartered. Not implemented. Do not let it drift into F-150's scope.**

> **Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry a run
> produced. Any report quoting it must carry that limit.**

### 4.3 C-112 — chartered, NOT dispatched

`prompts/C-112.md`. Its **item 5 dependency (C-108 merged) is now SATISFIED.** Closes:

* **REV-109 R1** — the **26 committed citations C-109's own diff made stale**: 16 into
  `MASTER_PLAN.md` (+16 lines at `:160`), 10 into `pwml-test-runner.md` (+19 from `:52`). F-154's own
  class. **Apply anchors, not renumbering. Do NOT rewrite frozen historical records** — distinguish
  live citations from signed records and say which bucket each of the 26 is in.
* **REV-109 R2** — three false-PASS vectors in the route check. **The one that matters: probes in a
  subdirectory are never enumerated** — the glob is non-recursive and `is_file()` drops the directory
  **silently**, so the check exits 0 with a green G11 while missing evidence entirely. Also: a
  zero-byte file always reports `reachable` (the empty blob oid is a universal constant), and
  `--allow-empty` + a mistyped task id disarms the exit-3 protection beside it.
* **F-157** — `FINDINGS.md` § F-153 still cites `streamlit_app.py:5669`. **Cite the SYMBOL.**
* `FINDINGS.md:1125-1126` row E (`TEST_MATRIX.md:218` → Chunk E is at `:237`; **row D-qb's
  `chunk_d_gate.py:70` is CORRECT — leave it**) and `TEST_MATRIX.md:568` (`"§ Chunks begins at line
  209"`; it begins at `:228` — in boundary, below the pin, and **missing from the author's routed
  list**).
* **C-108 R4** — `c107_mutation_attack.py` M16 aborts: its anchor occurs once at `f67e00a` and **zero
  times** post-C-108, `apply_mutation` raises on a 0-match, the harness prints `ABORT` and returns 3.
  **M16 is last, so M1–M15 still execute**; the property is re-pinned as C-108's N13. **Re-point the
  two anchor lines; do not weaken any mutation to make it match, and do not delete M16.**

**Explicitly NOT in scope:** `TEST_MATRIX.md:726-727`, C-106's signed record. It is **false**, and it
is **another card's signed record.**

**C-112 carries the criterion C-109 was never given: measure the drift YOUR OWN diff creates before
you finish.** Write `REV-112` criteria before its diff exists.

### 4.4 C-111 — chartered, NOT dispatched

`prompts/C-111.md`. **F-148 observability. It INSTRUMENTS. If it fixes, it has failed.**

Must durably preserve, on the path that survives cleanup: attempt count by stage · retry reason ·
per-stage elapsed · **finalization-reserve consumption** · **timeout source** · whether a payload
existed before cleanup · cleanup decisions affecting partial artifacts · total model calls ·
terminal state before wrapper cleanup.

**Step 1 is the cheap OFFLINE probe** separating F-148 § 3's three open readings (child never honoured
its deadline / finalization overran the reserve / nothing on the outer-kill path invokes finalization
at all). **Hypotheses written before it runs.** No LLM spend, no benchmark leg, **no T-107 leg rerun**.

**§ 8 adds F-158 as in-scope:** `result_text` prints `counts` and `files` but **never
`termination_reason` or `operational_failure`** — the two fields that say *why* the blocks are empty.
Needs no gold access. **Do NOT touch the verdict line**; printing more context does not make a wrong
verdict right.

**`stage=unknown` on the outer-kill path is NOT a defect — it is honest.** Do not make the parent
guess a stage. **Out of scope, each a stop condition:** any retry-behaviour change, the leg ceiling or
`leg_timeout_override_*` (operational, belongs in the readiness table), fixing the finalization seam
itself.

### 4.5 Rebuild T-108 readiness · 4.6 Launch only if every condition is green

See § 6.

---

## 5. Protected — do not touch

**F-147 remains registered and DELIBERATELY UNCHARTERED.** Fixing it alone would flip two legs to
PASS that would then export gold-forbidden content — including `protoporphyrin IX`, which occurs
**zero times** in a 67,304-character file whose length the gold cites exactly. **The earliest unsafe
seam is Stage-1 extraction, not the driver. Merge rule 6.**

Any acceptable proposal must act at the earliest unsafe Stage-1 seam, reject unsupported pathway
content, preserve source-supported reactions, avoid hardcoded paper/protein/compound/fixture
identities, pass independent biological review, preserve Fur's correct withholding, and preserve
verified ALAS2 identity **without treating identity alone as relevance**. **If no general correction
satisfies all of these, leave it registered.** It does not authorize weakening a downstream gate.

**Nothing under `src/t2pw/pipeline/` changed this wave. Keep that true.**

**`placeholder_backed_proteins` / Unknown-backed export** — `PRODUCT_CONTRACT` § 13 standing
disagreement. **No agent may resolve it without a new explicit ruling. Escalate only.**

**`LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out with no payload. Verified only
on `runs/2026-08-02_2130`. **Do not report T-107 as confirming it.**

---

## 6. T-108 — currently **NO-GO**. See `T108-READINESS.md`

**Green now:** F-146 rejected · battery 0/29 · F-155 merged and independently approved · corpus movers
understood both directions (19 refused / 0 admitted, mover set stable) · Q1 implemented and merged ·
Q3 ruled · integration pushed and remotely verified · pinned 10-paper/20-leg plan verified offline
(`verdict: OK`, 10 cases, **0 search calls**, all `[pinned_override]`) · provider config present
(`LLM_PROVIDER=openrouter`, `LLM_TEMPERATURE=0`, full `OPENROUTER_*_MODEL` set, `LLM_MAX_RETRIES=3`) ·
no peer owning an overlapping job.

**Outstanding blockers:**

1. **F-150 not resolved** — no applied correction has passed its independent A/B, and half 2 is an
   open product-owner question.
2. **C-112 and C-111 undispatched.**
3. **The leg-timeout ceiling has not been deliberately chosen or recorded.** *This is the blocker most
   likely to be missed, because it is operational rather than code.* T-107's ceiling was halved
   **`3600 → 1800`** with `leg_timeout_override_reason` and `_source` **both empty**; the slowest leg
   that *finished* used **92.1%** of it. **At that budget three timeouts is the expected outcome, not
   an anomaly.** `PMC12096016` is **one of only two strict-denominator papers** and was **lost to the
   clock, not to biology** — so Priority 5 can read `0/2` again for a reason that is not a pipeline
   defect and would be misread as one. **Choose the ceiling deliberately and record the reason in the
   run manifest BEFORE launch. Do not launch at 1800 s with an empty reason and classify the timeouts
   afterwards.**
4. **Gates must be re-verified at the launch tip** — SMOKE, gold-readers, `acceptance.py` hash.
5. **Zero sprint-owned Python and a free heavy lock at launch.**
6. **Enough time to monitor or formally transfer the run.**

**T-108 runs from the PRIMARY CHECKOUT.** `.env` is **untracked**, so a worktree silently gets
`LLM_PROVIDER=local` — calls 400, the exception is swallowed, and the curator becomes a no-op **by
accident** — while the primary issues real billed calls. **A green cohort obtained in a worktree does
NOT certify the primary.**

**Launch protocol, once and only once every row is green:** fresh milestone identity · the same
ratified 10-paper/20-leg plan · stage-only preflight · verify plan **and gold** inside that exact
staged directory · all pinned overrides and **zero search calls** · continue the verified directory
**without `--fresh`** · pinned OpenRouter models · **one** run through the bounded wrapper ·
background with explicit ownership · wrapper timeout from measured T-104–T-107 durations with cleanup
headroom · **monitor the existing wrapper, never launch a duplicate** · score the first valid official
draw honestly · preserve raw and contract-adjusted results separately · **do not rerun to improve
stochastic composition.**

**T-108 must report:** 20-leg completion and scorable denominator · every timeout and missing payload ·
Priority 1 raw and accepted counts **and composition** · Priority 2 eligible denominator and
unsupported retained reactions · Priority 3 referential-integrity failures · Priority 4/5 raw and
accepted coverage · negative-control outcomes · every applied policy adjustment · whether accepted ·
exact evidence paths · model/provider provenance · usage and cost where available.

**If T-108 fails, preserve it as a failed official release candidate and triage from committed
artifacts. Do NOT rerun it.** A later candidate needs a new milestone identity and a separately
recorded readiness decision.

**There is no OpenRouter spending, token, request or model-usage ceiling. Cost must never restrict
justified work.** Usage may be recorded for observability only.

---

## 7. Findings registered this wave — F-156 … F-160

**F-156** — `MASTER_PLAN` § 2's **third** claim is false too. Graph-delta validation is implemented,
wired, reached in production **and load-bearing** — proved by mutation
(`test_a_refused_delta_does_not_advance_the_graph_and_says_which_rule` goes red when the enforcement
is removed). **Refuted on the code by a read-only peer session; certified behaviourally by the Lead.
The provenance is split on purpose — a static read and a mutation-proved property are different
epistemic objects.** *Why both halves were true at once:* validation was never `conform.py`'s job; the
map looked for the validator in the module that by design does not hold it and read its correct
absence as a gap.

**F-157** — a citation pinned to bytes that exist in **no commit**. `streamlit_app.py:5669` was read
off the **uncommitted** working copy of the never-commit file; the committed value is **`:5636`**, the
`+33` being exactly the protected diff. It propagated F-153 → `MASTER_PLAN` → the Lead's own charter.
**Never cite a line number in a file carrying an uncommitted diff. Cite a symbol.**

**F-158** — `RESULT.txt` prints the empty blocks but not `termination_reason` / `operational_failure`.
**Corrects the Lead's framing**: the distinction is not absent, it is unexplained. **Two named fields,
not a missing capability.** Routed into C-111.

**F-159** — **`failure_kind = contract` does not mean a contract failure; it means "there were issue
codes."** `_classify` tests `contract_signal or issue_codes` **before** its network/LLM markers, so a
provider casualty carrying any code wears a `contract` label. **It is not a bug in `_classify`** — its
docstring states the rule it obeys — the defect is in what the label licenses downstream. It is the
**dominant bucket** (55 legs vs `no_reactions`'s 8). It let C-110 round 0 admit three casualties,
including a row whose shipped message at `driver.py:2565` is literally **`"no research report was
produced and no reason was given"`** — *a message saying no reason was given, scored as a stated
reason.* **Addendum:** two more zero-instance routes — `_NO_REACTION_MARKERS` is also tested before
the network markers, and `driver.py:2217` labels an **acquisition** failure `KIND_NO_REACTIONS`. The
second is the sharpest evidence C-110's artifact condition is **load-bearing rather than merely
strict**. **Any T-108 report grouping by `failure_kind` must not present `contract` as "stopped for
contract reasons".**

**F-160 — read this before running ANY test or mutation.** CPython keys a `.pyc` on **(source mtime
truncated to whole seconds, source size)**. A same-length edit landing in the same second leaves the
cache valid and **the OLD bytecode runs**.

> **It reaches pytest itself** — `220 passed, exit 0` on a tree whose source says otherwise. **A green
> suite proves the bytecode that executed was green, not that the source on disk is.**

In a mutation harness it prints **`MUTATION SURVIVED`**, which reads as *"this guard has no test"* —
and the natural response is to weaken or delete a guard that is in fact protected. **It is
intermittent, so a passing re-run looks like confirmation.** The mutations that escaped did so **by
luck**: a marker comment changes file size.

**THE SCOPED PURGE REQUIREMENT — the Lead's first instruction here was unsafe and two of two agents
walked into it:**

> **Purge only `src/t2pw` and `tests`.** Verified to contain **no** tracked `__pycache__`. An
> **unscoped** purge deletes **56 tracked `.pyc` files** across `__pycache__/`, `scripts/`, `src/`,
> `src/tools/` — `.gitignore` lists both patterns but they predate it, so git tracks them regardless.
> **Never purge unscoped. Never "solve" it by untracking the 56** (repo-wide, unchartered, `.git`
> already 158 MB). **Restore what you purge and verify clean before SMOKE**, which asserts
> `git status --porcelain` on the file the mutation harness guards.
>
> **A `MUTATION SURVIVED` result from a same-length mutation is not a result.** Re-take it with caches
> cleared before recording it. C-108's **N20** is a deliberately same-length control that goes red
> only when the purge works — endorsed by REV-108 for the sprint vocabulary.

**Also durable:** any card modifying a file the mutation harness guards must **COMMIT before running
SMOKE**, or `test_04_restore_is_byte_exact_on_the_real_mutated_module` fails on
`git status --porcelain` and presents as **502/1**, which reads exactly like a regression.

---

## 8. Reviewer evidence — ancestry verification is now MANDATORY

**A reviewer's evidence branch is not automatically a superset of the code it reviewed.** Both
REV-108's and REV-110's branches were rooted at **round 0**. Merging either alone would have
**silently reverted the correction rounds while looking like a routine evidence merge.**

On C-110 it was caught only by a diffstat showing `acceptance.py +344` where round 1 was `+387`.
On C-108 it was caught because the Lead **asked for the parent, not just the commit** — the reviewer
found it itself and rebased.

**Before merging any reviewer branch:**

```
git merge-base --is-ancestor <card-tip> <evidence-branch>     # must be YES
git diff --name-only <card-tip> <evidence-branch> -- src tests # must be EMPTY
```

If the branch is rooted earlier: **merge the card tip FIRST, then the evidence branch second.**
Verify the staged `src`/`tests` diff is the version you intend **before committing.**

**When a review closes, collect its G11 reports, probe SOURCES and substantive outputs onto an
accepted branch before releasing its worktree**, then verify with
`evidence/reviewer_evidence_route.py --task <TASK> --worktree <dir> --integration-repo <repo>`.
**A G11 report certifies a job terminated cleanly and preserves NOTHING about what it found.**

---

## 9. Process — merge gates, not suggestions

Everything through `evidence/bounded_run.py` with the **explicit venv interpreter**
`<tree>/.venv/Scripts/python.exe`. **F-143: a bare `python` is system 3.13 with no `streamlit` → 35
spurious import errors that read exactly like a regression.** Real `--timeout`; **`--basetemp` under
`C:/t/` with the parent PRE-CREATED** (without it 83 tests error with `PermissionError`; a missing
parent once reported **382 instead of 453**); `PYTHONPATH=<tree>/src`; **`T2PW_OFFLINE_CURATOR=1` set
in the BOUNDED CHILD, not just the shell** (without it `run_pathway_curator` issues an ungated LLM
call at temperature 0.2 whose accepted patches flow into `final_mapped_db` — the measured root cause
of BL-003); `PYTHONIOENCODING=utf-8`; `--heavy-lock <TASK>`.

**Your own shell timeout must EXCEED the wrapper's**, or you kill it before its cleanup `finally` runs
and strand the lock. **Exit 95 = `BOUNDED_RUN_HEAVY_LOCK_HELD`: the child never started. That is an
infrastructure event, not a result — wait and retry.** **Never clear, break or steal a lock you do not
hold. Never `taskkill /IM python.exe` or kill by name.** Before clearing a stranded lock: multiple
byte-identical holder samples, dead holder PID, zero matching processes, no peer ownership.

**Never `pytest -n auto`. Never parallelise SMOKE. Never run the full suite unchunked (~16 GB).**
One heavy job at a time. Every job: **`FINAL SURVIVING COUNT : 0`** and **`cleanup : success`**.
Wrapper reports carry **no child stdout** — redirect and grep, **never `head`**.

**G11: guard the allocation on the SHAPE OF THE PATH, never on absence of error text.**

```
P=$(... g11_evidence.py next --task <ID> --label <l> 2>/dev/null | tail -1)
test -n "$P" || { echo EMPTY; exit 1; }
case "$P" in *<ID>*<l>.json) : ;; *) echo "INVALID: $P"; exit 1;; esac
```

A bad **task id** puts *error text* in your variable; a bad **label** (anything not
`^[a-z0-9][a-z0-9._-]*$` — an uppercase letter will do it) leaves it **empty**, which becomes
`--json ""`. The job then runs clean, reports zero survivors, `cleanup : success` — and produces
**no artifact at all.** Task ids match `^[A-Z0-9]+-\d{3}[a-z]?$`.

**Never commit:** `data/enrichment_cache.json` (39 MB, tracked), `data/id_mapping_cache.json`,
`topics_*.txt`, `src/t2pw/app/streamlit_app.py`, `cache_snapshot/`, the stray `ValueError`, or
anything under `out/`, `outputs/`, `tmp/`. **Stage explicit paths; inspect `git diff --cached` before
every commit; `git commit -F <file>`.** **Bash heredocs break on apostrophes here — including quoted
`<<'EOF'` — and a heredoc that fails to parse executes NOTHING, silently.** Write long text with a
file-writing tool.

**Do not:** merge to `main` · amend · rebase · reset · squash · prune a worktree · rewrite accepted
history · discard unfamiliar changes · delete accepted evidence.

**Per card:** fix pass/fail criteria **in writing before the diff exists** · commit the card **before**
cutting its worktree so the worktree contains its own charter · implementation author and independent
reviewer, never the same agent · inspect the **actual diff** · exercise the **real production path** ·
run rejection and preservation cases **separately** · **attribute which guard fired, do not infer it
from the verdict** · mutate load-bearing assertions · **preserve failed measurements beside their
corrections** · merge `--no-ff` only after approval · gate at the merged tip, push, verify remotely.

**Classify before changing production, scoring or gold:** `product_contract_violation` ·
`gold_data_defect` · `policy_disagreement`, citing the gold `relevance_note` / `export_rationale`.
**Only a confirmed `product_contract_violation` justifies production code. A benchmark failure does
not by itself justify a code change. Never weaken a biological gate to increase PWML output.**

**Two automatic correction rounds; a third is an explicit authority decision.** Tell reviewers plainly:
*if what remains is registerable, say APPROVE and register it.*

---

## 10. The transferable lessons of this wave, in the order they cost the most

> **1. A green test run is a statement about the bytecode that executed.** F-160 reaches pytest, not
> just harnesses, and its false direction invents a coverage gap that invites deleting a real guard.
>
> **2. A bounded closed list flips polarity between a cue and a contra.** C-108 round 0 rebuilt the
> contra as a closed list with ACCEPT as the default — the lesson the card quotes and the diff's own
> comment quotes — and still weakened a biological gate.
>
> **3. Verify a blocking finding before spending a round — and prove your probe reaches the code.**
> The Lead's first C-108 verification probe used the wrong payload envelope, never reached the guard,
> and returned all-permissive at base. It *looked* like a finding. Only contradicting two independent
> records exposed it. **Assert a known-positive and a known-negative before trusting any harness.**
>
> **4. Ask for the parent, not just the commit.** Two of three reviewer evidence branches were rooted
> at round 0.
>
> **5. Handed a defect, look for more of it.** Given four failing spans, the C-108 author wrote ten
> more of the same grammar to attack its own repair and found nine also leaked. The gate was weakened
> wider than the finding stated, and only looking for more trouble found that out.
>
> **6. Four corrections came to the Lead this wave and every one was right** — two from implementers,
> one from a reviewer, one from a read-only peer. **Reading a report is not verification, and that
> applies to your own reports and to a discrepancy you have already seen and not chased.**

**Update `RESUME-NEXT-SESSION.md` in place, and replace this file** when your wave closes.
**Commit the probe and its log, not just the report.**
