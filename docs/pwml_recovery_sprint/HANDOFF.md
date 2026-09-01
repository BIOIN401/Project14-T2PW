# PWML RECOVERY SPRINT — HANDOFF after the ORCH-718 wave

You are the **Lead Orchestrator and Integration Authority** for
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, integration branch `sprint/pwml-recovery`.

**Do not merge to `main`.** Work autonomously.

**T-107 has RUN, is SCORED, and is TRIAGED. Its verdict is `NOT ACCEPTED`. Do not re-run it, do not
re-score it, do not reinterpret it.** Six cards have merged after it. **None of them re-accepts it.**
A run's verdict is a fact about the artifacts it produced.

---

## 1. Takeover — verify once, do not trust these numbers

**Verified pre-handoff integration state: `0e260d48`.** The commit carrying this file is newer.
**The binding invariant is `local = origin/ = git ls-remote` — read it, do not recall it.**

| Check | Expected |
|---|---|
| local = `origin/` = `git ls-remote` | **all three equal** |
| `main` | local `7531692`, remote `03f1af5`. Advanced **outside** this sprint. **Touch neither ref** |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python | **zero** |
| IDE processes | two `ms-python.isort` — **never cleanup targets**; **PIDs change, match on command line** |
| `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` |
| caches, `topics_*.txt`, stray 0-byte `ValueError` | uncommitted, untouched. **Leave them** |
| worktrees | **211.** Prune none |
| **SMOKE** (22 files) | **503 passed, exit 0** |
| **gold-readers** (22 files) | **456 passed / 0 failed / 8 skipped / 0 errors, exit 0** |
| **29-case battery** | **`battery=0/29  F146=REJECTED  C1..C6 all 0`** |
| `acceptance.py` | `sha256:4bd893ac410d16d3…` (**CRLF working-tree** form; the LF blob is `d9f817e1…`) |
| gold blob `pinned_v1.json` | **`36f4b7b690b577f72882c3045ca6728d1ec8d9d1`** |
| whole-tree G11 | **5032 artifacts, 0 non-compliant.** The count is self-referential — **reconcile, do not match** |

**`cache_snapshot/` does not exist** in the primary checkout, though earlier handoffs list it as an
untouched uncommitted path. Nothing this wave touched it. Recorded rather than explained away.

Run `ListAgents` and contact every live peer before claiming the branch, the lock or a worktree.

**Track agent liveness separately from job liveness** — ~15 min without observable progress → status
request; ~30 min with nothing → stalled, interrupt, preserve, redispatch from the last verified
commit. **A subagent *reading* leaves no process and no artifact.**

---

## 2. What is DONE — do not re-litigate any of this

| Card | Merge | Reviewer verdict |
|---|---|---|
| **C-113** F-150 half 1 + census re-pin | `db119f53` | REV-113 **APPROVE with registered residuals** |
| **C-111** F-148 timeout observability | `2a0ccdbd` | REV-111 **APPROVE with registered residuals** |
| **C-112** residual sweep | `c942f774` | REV-112 **APPROVE with registered residuals** |

Each was reviewed against criteria **committed before its diff existed**, ancestry-checked per § 8,
gated at its own merged tip, then **all three gated together** at `c942f774`.

### 2.1 The most important thing that happened, because it will shape your instincts

**F-150 half 1 was merged, gated RED, and reverted before it landed for real.**

REV-F150 verified it on all eight conditions and the Lead inspected the diff. It merged at
`b05a7281`. **SMOKE came back `501 passed / 2 failed`.** Merge rule 10 required the merge not to
stand, so it was reverted and integration re-proved green at **503** (`700c9434`). The edit then
re-landed at **C-113 with the census movement it causes**, measured and attributed per leg.

**The gold edit was never wrong. Landing it without its full footprint was.** The reviewer did
exactly what its criteria asked; **the criteria were incomplete** — that is **F-161**.

**Both reviewer evidence and the failed arm are committed**, so the 501/2 → 503/0 pair is a clean
behavioural attribution: same 22 files, same tree, one difference.

### 2.2 C-113 — the gold edit, landed with its footprint

Gold blob `aee8cb4f` → **`36f4b7b6`**, cherry-picked from REV-F150's `ac27ed7b` so the landed bytes
are provably the reviewed bytes, on **its own commit** separate from the re-pin.

Three census pins moved in `tests/test_c102_coverage_denominator.py`, all still `==` against
literals: `affected_papers` gains `PMC12180156` · `with_matched_forbidden` **26 → 29** ·
**`withheld` 97 → 100**.

**`withheld` is the one to understand.** Its assert **never executes in either red arm** — the
set-equality above it aborts the test first — so **no failure message could have supplied it**. The
Lead therefore deliberately withheld the number from the charter. REV-113 proved it was *measured*
rather than back-computed two independent ways: the 69 unnamed legs **re-measure to 97 in the POST
arm too**, and a **same-length `100 → 101` mutation** (the F-160 trap, walked into on purpose)
showed pytest independently observes 100.

**The confirming instance worth keeping:** a fourth committed leg,
`runs_verify/2026-08-24_1402/papers/PMC12180156/research`, draws the Greek delta spelling too — but
as the **enzyme**, which this gold case quotes as acceptable — and is **byte-identical between the
arms**. `forbidden_match` refuses containment, so the gate separates the fabricated metabolite from
the legitimate enzyme name **on the real corpus, leg by leg.**

### 2.3 C-111 — instruments, does not fix

Nine items preserved durably past cleanup and **read back off disk after a real force kill**.
Timeout source distinguishes in-process deadline · outer parent kill · wrapper · provider.
F-158's two named fields added; **the verdict line is byte-identical at `:718`**.

**The probe separated F-148 § 3's three readings by measurement**: reading 1 **REFUTED**, reading 2
**mechanism confirmed / magnitude not measurable offline**, reading 3 **CONFIRMED**. Hypotheses are
in commit `4fde91b3`, which contains **only** the hypotheses file — the probe first appears in
`9bf2b351`. **Its first probe attempt failed its own controls and is committed beside the
correction.**

**`src/t2pw/pipeline/` is byte-identical.** `stage=unknown` was not made to guess. And where it
could most easily have cheated — repairing `files: []` once it could see the payload — **it did not,
and pins that it did not.**

REV-111 quantified the fsync question rather than arguing it: **+0.55 ms per attempt**, 0.0014% of
the 120 s reserve, 0.46% even under a thousand-attempt runaway. And on a **surviving** mutation
(removing `flush()+fsync()` left the suite green) it **measured why** instead of reporting a missing
guard: per-event open/close alone survives a *process* kill, because a kill does not discard the OS
page cache. The fsync buys *machine-crash* durability the suite cannot see by construction.

### 2.4 C-112 — the residual sweep, and the criterion C-109 was never given

26 citations bucketed **7 live / 19 frozen**; live ones converted to **anchors, never renumbered**;
frozen records left. `:477` line-neutrality proved on all three parts. `TEST_MATRIX.md:726-727`
(C-106's signed record) **byte-identical**. Three R2 false-PASS vectors closed with fail-then-pass
pairs. M16 re-pointed: **17 mutations, survivors 0, all RED, no ABORT**, target byte-identical.
F-157 now cites **the symbol, with no line number at all**.

**Self-drift ZERO, with the scanner proved capable of non-zero** (a synthetic 1-line insert strands
83–96 citations). REV-112 re-derived all 26 buckets **before** reading the card's table, and ran the
**tip tests against the base blob** — 4 new tests fail at base, 12 untouched still pass.

**The card's own best finding:** three of the 26 addresses were **already wrong before C-109 inserted
anything** — *"a drift measurement that starts from a wrong address reports a shift, not a
resolution."*

---

## 3. Rulings made this wave

**The leg-timeout ceiling — RULED: restore the 3600 s default, no override.**
`T108-READINESS.md` § 2.1. Measured first (`evidence/orch718_leg_duration_census.py`, G11
`ORCH-718/02`): **a leg has demonstrably needed 3421.4 s**, so every ceiling below 3600 s is
known-insufficient *by direct measurement*; **p90 is 1609 s, 89.4% of an 1800 s ceiling**; and
**1800 s produced timeouts in four separate run trees**, so T-107's three timeouts were a repeating
pattern, not an anomaly. Restoring the default also **dissolves** the contract problem —
`leg_timeout_overridden` becomes `false`, so there is no empty reason for § 9 to catch.

**Limits recorded with it:** every timed-out leg is **censored**, so the true requirement is *at
least* 3421.4 s; 3600 s clears the slowest observed finisher by only **179 s**. **A timeout at
3600 s is therefore not automatically a defect — and must not be waved away either.**

**F-161's disposition — RATIFIED as a standing obligation.** Any change to `pinned_v1.json` runs
**BOTH** the gold-readers selection **and** SMOKE, each reported with the gold SHA it was measured
against. REV-113 correctly registered that a *card* had declared this on its own authority; the Lead
ratified it. **A card proposing a rule it cannot itself enact is the correct behaviour.**

**Adopted from REV-113's R12:** when a census pin in `test_c102_coverage_denominator.py` moves,
close the mutation gap with a **same-length** mutation of the moved literal.

**Refused: REV-111's correction that `LLM_MAX_RETRIES` is 8.** It is **3** in the primary's `.env:34`;
8 is the *code default* at `client.py:484`, and a reviewer worktree has **no `.env`**. T-108 runs
from the primary. **This is the `.env` trap catching a careful reviewer while it read the file
documenting the trap.** Generalised rule now recorded: **a configuration claim measured in a
worktree is a claim about the code's defaults, not about the run.**

---

## 4. THE NEXT WORK ORDER

### 4.1 T-108 — **NO-GO on exactly one row.** `T108-READINESS.md` § 5.

**Eighteen of nineteen rows are GREEN**, rebuilt at the current tip. **All three blockers the
previous handoff named are closed.**

**Row 19 — enough time to monitor OR formally transfer — is the blocker**, and it fails on the "or"
as much as the "and". T-105's comparable 20-leg run took **4.85 hours at a lower ceiling**; there is
no transfer recipient (the only live peer is read-only, for a different user, unauthorized for
sprint work — **a transfer nobody has accepted is not a transfer**).

**And the compliance rule decides it independently of judgement.** `TEST_MATRIX` § 0 rule 1 permits
tracked background **only** when *"no detached or unowned job remains."* A T-108 launched without an
owner becomes unowned when its session ends — **a G11 violation by construction**, and G11 is merge
gate 11.

**T-108 is ONE-SHOT.** *"If T-108 fails, preserve it as a failed official release candidate… Do NOT
rerun it."* **Launching unmonitored risks burning the milestone identity on an infrastructure
failure nobody watched — strictly worse than launching tomorrow.** **Cost is not the constraint and
never was.**

**Before launching:** re-derive rows 4, 5, 11, 16, 17 yourself; confirm row 19 is satisfiable **for
you** first, since it is the only row that depends on the operator rather than the tree; then § 4's
launch protocol **at 3600 s with no override**, verifying `leg_timeout_overridden: false` **in the
staged directory before launch, not after.**

### 4.2 Two chartered tooling repairs — deliberately NOT taken

Both are instruments this wave's certifications were produced through, and **changing a
just-reviewed instrument without a new review is the move this sprint refuses.**

- **F-163 — `HeavyLock.release` is not atomic.** It unlinks `holder.json` then `rmdir`s; a kill
  between them leaves an **anonymous** lock that the clearing checklist **cannot address**, because
  its first condition is *byte-identical holder samples* and there is no holder. **This cost four
  strands this wave.** `bounded_run.py`'s build hash is in **every** G11 report, so a mid-wave change
  breaks comparability.
- **F-164 — C-112's recursion fix opened a false FAIL.** The now-recursive glob descends into the
  allocator's dot-prefixed `.staging/`, whose *contents* are non-dot files, and reports leftover
  reservations as unreachable evidence. **Any fix must prove the C-112 vector stays closed** — a
  repair that re-breaks recursion to silence `.staging` restores the original false PASS.

### 4.3 The open product-owner question — preserve or obtain, do not answer

> **Should `supported_reactions_complete` be set on any gold case — and if so, which?**

`DECISION-PACKET-F150-HALF2.md`. It is set on **zero of ten** cases, and `max_retained_reactions` is
set on exactly two — **both negative controls** — so Priority 2's unsupported-reaction verdict can
**never** be evaluated on a non-control paper. **The status quo is honest**: `semantic.py` stamps
`UNSUPPORTED-REACTION VERDICT NOT EVALUATED` and withholds `false_positives` rather than reporting a
hard zero, as `PRODUCT_CONTRACT` § 11 requires.

**The risk is asymmetric and that asymmetry is the decision.** Unset yields a number *withheld and
labelled as withheld*; set wrongly yields a number *stated and wrong, accusing the pipeline of
inventing chemistry it did not invent* — the module records that miscall as reading **227 fabricated
reactions** in a run that produced far fewer.

**It is NOT a T-108 blocker.** What T-108 must not do is quote Priority 2 without its limit:
**Priority 2 = 1 is a real number and it is not a measure of how much invented chemistry a run
produced.**

### 4.4 Registered residuals awaiting an owner

R9/R10 (F-150 prose and the redundant Unicode alias — **no behavioural exposure**) · REV-112's R1–R5
(prose counts, a stale log, F-162's missing id **now assigned**, a frozen `:59` citation, and a
**second drift class**: the number still points here but the content under it changed) · REV-111's
RES-1–RES-4 (an overstated docstring, **a tautological `assert x == x` sitting in the test whose
docstring claims to pin B2**, `wrapper` inference, `RESULT.txt` counted as payload) · R-C111-1–3.

**Raised, not answered:** should `c107_mutation_attack.py` be gated? It is run by **no gate**, which
is why M16's ABORT rode a green 503 onto integration. **REV-112 calls it the highest-value follow-up
in the card.** It is a `TEST_MATRIX` change with real runtime cost and it is **the Lead's to make**.

---

## 5. Protected — do not touch

**F-147 remains registered and DELIBERATELY UNCHARTERED.** The earliest unsafe seam is **Stage-1
extraction, not the driver**. Merge rule 6. **Nothing under `src/t2pw/pipeline/` changed this wave.
Keep that true.**

**`placeholder_backed_proteins` / Unknown-backed export** — `PRODUCT_CONTRACT` § 13 standing
disagreement. **Escalate only.**

**`LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out with no payload. **Do not
report T-107 as confirming it.**

**T-107 immutable. `main` untouched. `streamlit_app.py` never committed.**

---

## 6. Findings registered this wave — F-161 … F-164

**F-161** — **neither gate selection is a superset of the other.**
`test_c102_coverage_denominator.py` reads the gold at module scope, is **in SMOKE**, and is **not in
the 22-file gold-readers selection** — so a gold edit's *mandated* gate was structurally blind and
returned byte-identical `456/8/exit 0` in both arms. **Merge rule 10 was the only thing between an
instrument gap and a landed regression-shaped tip.** Class: **defect in the review instrument, not
in the reviewer.**

**F-162** — **a mistyped task id did not return nothing; it returned another task's evidence.**
`rev109` / `REV_109` enumerated **REV-109's** artifacts and returned 1, via the case-insensitive
filesystem plus `task_stem`'s stripping. **Worse than the charter described**, and found only
because the author asserted the charter's version and **the assertion failed**. Fixed by C-112;
blast radius verified nil.

**F-163** — **`HeavyLock.release` non-atomic**; see § 4.2. **The recovery technique generalises:** a
failed acquire prints the holder file verbatim in its `BOUNDED_RUN_HEAVY_LOCK_HELD` diagnostic, so
an earlier exit-95 in a transcript **is** the byte-exact holder sample the checklist demands. Clear
with **`rmdir`, never `rm -rf`** — `rmdir` refuses on a non-empty directory, so a race fails the
clear instead of being overwritten. **A subagent must not clear an anonymous lock; escalate.**

**F-164** — **the recursion fix opened a false FAIL**; see § 4.2. **It fails safe** — a false FAIL
costs an investigation, a false PASS retires the habit that works — but a check that goes red on
correct evidence gets disbelieved, and a disbelieved check is on its way to being no check.

---

## 7. Process — merge gates, not suggestions

Everything through `evidence/bounded_run.py` with the **explicit venv interpreter**
`<primary>/.venv/Scripts/python.exe` (**worktrees have NO `.venv`** — this is the established
pattern; **a bare `python` is system 3.13 with no `streamlit` → 35 spurious import errors that read
exactly like a regression**). Real `--timeout`; **`--basetemp` under `C:/t/` with the parent
PRE-CREATED**; `PYTHONPATH=<tree>/src` (**`pinned_pytest.py` exits 98 without it — a measurement
failure, never a test result**); **`T2PW_OFFLINE_CURATOR=1`** (its absence is the measured root cause
of BL-003); `PYTHONIOENCODING=utf-8`; `--heavy-lock <TASK>`. **There is no `--child-env` flag** — set
env as a prefix and it inherits.

**`FINAL SURVIVING COUNT : 0` and `cleanup : success` on every job.** **Exit 95 = the child never
started: an infrastructure event, not a result.** Never `taskkill /IM python.exe` or kill by name.
Never `pytest -n auto`; never the full suite unchunked (~16 GB). **One heavy job at a time.**

**NEW, and it cost this wave three lock strands: the Bash tool caps a single call at 600 s.** A
`--timeout 900` wrapper can **never** be covered by a foreground shell. **A lock-wait retry loop
must not share the job's budget** — under contention, acquiring took **19, 15, 120 and 330**
attempts on different jobs. A gate that must queue belongs in **tracked background under D-026**.

**G11: guard on the SHAPE OF THE PATH, never on absence of error text.** A bad **task id** puts
*error text* in your variable; a bad **label** (uppercase will do it) leaves it **empty**, which
becomes `--json ""` — and the job then runs clean, reports zero survivors, `cleanup : success`, and
produces **no artifact at all**.

```
P=$(... g11_evidence.py next --task <ID> --label <l> 2>/dev/null | tail -1)
test -n "$P" || { echo EMPTY; exit 1; }
case "$P" in *<ID>*<l>.json) : ;; *) echo "INVALID: $P"; exit 1;; esac
```

**Never put a non-`bounded_run` file inside `evidence/g11/<TASK>/`.** `iter_reports` treats every
non-dot file there as a report and `check_many` fails it. **The Lead did this to itself this wave**,
writing a route-check verdict into the task folder and immediately reddening the gate. Probes and
logs go **flat** in `evidence/`.

**F-160 binds every test and mutation.** A same-length edit in the same second leaves the `.pyc`
valid and **the old bytecode runs** — it reaches pytest itself. **Purge only `src/t2pw` and
`tests`**; an unscoped purge deletes **56 tracked `.pyc` files**. `PYTHONDONTWRITEBYTECODE=1` is
stronger than a pre-purge for a mutation harness.

**Never commit:** `data/enrichment_cache.json` (39 MB, tracked), `data/id_mapping_cache.json`,
`topics_*.txt`, `streamlit_app.py`, `cache_snapshot/`, the stray `ValueError`, or anything under
`out/`, `outputs/`, `tmp/`. **Stage explicit paths; inspect `git diff --cached`; `git commit -F`.**
**Bash heredocs break on apostrophes here** — write long text with a file tool.

**Do not:** merge to `main` · amend · rebase · reset · squash · prune a worktree · rewrite accepted
history · delete accepted evidence.

**Per card:** criteria in writing **before the diff exists** · commit the card before cutting its
worktree · author and reviewer never the same agent · inspect the **actual diff** · **attribute which
guard fired** · mutate load-bearing assertions · **preserve failed measurements beside their
corrections** · merge `--no-ff` after approval · gate at the merged tip, push, verify remotely.

**Classify before changing production, scoring or gold:** `product_contract_violation` ·
`gold_data_defect` · `policy_disagreement`. **Only the first justifies production code. Never weaken
a biological gate to increase PWML output.**

---

## 8. Ancestry — MANDATORY before merging any reviewer branch

```
git merge-base --is-ancestor <card-tip> <evidence-branch>      # must be YES
git diff --name-only <card-tip> <evidence-branch> -- src tests # must be EMPTY
```

**This wave it mattered:** REV-111's worktree was cut at `d771323f` and the card advanced to
`ae9c1570` underneath it. It moved to the final tip and the check passed — **but a reviewer branch
rooted earlier would have silently reverted the newer commits while looking like a routine evidence
merge.** If rooted earlier: **merge the card tip FIRST, then the evidence branch.** Verify the staged
`src`/`tests` diff **before committing**.

**When a review closes**, verify with
`evidence/reviewer_evidence_route.py --task <TASK> --worktree <dir> --integration-repo <repo>`. **A
`unreachable_evidence` verdict naming only `.staging/` paths is F-164, not missing evidence** —
confirm the task's final reports are on integration, and **do not let that shortcut become a habit
of dismissing red route checks.**

---

## 9. The transferable lessons of this wave, in the order they cost the most

> **1. A mandated gate that is blind by construction is worse than no gate**, because its green
> result carries the authority of the process. F-161: the reviewer was right, the criteria were
> wrong, and only merge rule 10 caught it.
>
> **2. Withhold a number you have not measured.** `withheld == 97` could not have come off a failure
> message, so the charter said *measure it* and refused to guess. It had moved to **100**. Handing a
> card a number nobody measured would have been worse than handing it none.
>
> **3. Investigate a surviving mutation; do not report it as a gap.** The natural response to
> `MUTATION SURVIVED` is to delete the "untested" guard. REV-111 measured *why* the fsync mutation
> survived and found the guard protects a threat the suite cannot see.
>
> **4. A configuration claim measured in a worktree is a claim about code defaults.** A worktree does
> not error on a missing `.env`; it silently substitutes defaults and stays internally consistent.
> This caught a careful reviewer reading the file that documents the trap.
>
> **5. An anonymous lock is unclearable by its own owner.** Refusing to clear it and escalating was
> correct; *"nothing owns it, so I may delete it"* is right by luck and wrong by method.
>
> **6. Some defects are only visible at integration.** F-164 needed all four reviewer worktrees
> checked at once — no card and no reviewer could have seen it from inside its own lane.
>
> **7. The traps catch the people who wrote them down.** The Lead reddened the G11 gate by putting a
> non-report in a task folder — a trap recorded in this very file — while writing up a finding about
> that same directory. **Knowing about a trap does not prevent it; only the guard does.**

**Update `RESUME-NEXT-SESSION.md` in place, and replace this file** when your wave closes.
**Commit the probe and its log, not just the report.**
