# PWML RECOVERY SPRINT — POST-INSTRUMENTS WAVE

You are the **Lead Orchestrator and Integration Authority** for
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, integration branch `sprint/pwml-recovery`.

**Do not merge to `main`.** Work autonomously.

**T-107 has RUN, is SCORED, and is TRIAGED. Do not re-run it and do not re-score it.**
Two cards have since fixed defects it exposed. **Neither re-accepts the run.**

---

## 1. Takeover — verify once

Read `CLAUDE.md`, then **`RESUME-NEXT-SESSION.md`**, `T107-TRIAGE.md`,
`F148-TIMEOUT-CLASSIFICATION.md`, `DECISION-PACKET-ORCH717.md`, `PRODUCT_CONTRACT.md`, `LEDGER.md`,
`DECISIONS.md`, `TEST_MATRIX.md`. **G1–G11 bind you.**

| Check | Expected |
|---|---|
| local = `origin/` = `git ls-remote` | **all three equal.** Verify the invariant, not a remembered number |
| `main` | local `7531692`, remote `03f1af5`. Advanced **outside** this sprint. **Touch neither** |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python | zero |
| IDE processes | two `ms-python.isort` — **never cleanup targets**; **PIDs change, match on command line** |
| `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` |
| caches, `topics_*.txt`, stray empty `ValueError` | uncommitted, untouched. The `ValueError` is a 0-byte redirect accident from an earlier wave — **leave it** |
| **SMOKE** (**22** files) | **503 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** |
| whole-tree G11 | **0 non-compliant** (the count is self-referential — reconcile, do not panic) |

**SMOKE moved 473 → 503 this wave**, under merge rule 4, by C-106: `473 + 14 + 16 = 503`. Anything
saying 473, 465, 460 or 457 is stale. `TEST_MATRIX.md` keeps the history rather than deleting it.

Run `ListAgents` and contact every live peer before claiming the branch, the lock or a worktree.
**Prune no worktree.** This wave added `C:/t/c106`, `c107`, `rev106`, `rev106base`, `rev107`,
`rev107base`.

---

## 2. What the last wave settled — do not re-litigate this

**Both authorized cards are MERGED, independently reviewed, gated, pushed and remotely verified.**

* **C-106 (`fa69c57`)** — the sprint's instruments. The mutation-attack harness **runs again**
  (unrunnable since `e77ad3d`), restores **saved bytes**, and `git checkout --` is gone from the
  restore path. The census file is now **in SMOKE**, which is what stops the next drift going unseen.
* **C-107 (`ca3c711`)** — the C-105 follow-on, six routed findings, two correction rounds.
  **F-146 rejected at every tip**; 29-case battery **0 mismatches**; corpus **0 newly refused / 4
  newly admitted**, stable row-for-row.

Five things a successor gets wrong if nobody says them:

1. **"Re-pin 62 → 72" was wrong in three documents at once.** The handoff, F-151 and REV-104 all
   said it. **Four pins moved**: `withheld` 92 → 97 and `with_matched_forbidden` 23 → 26 sit
   *behind* the census assert and had never executed against the grown corpus. **Measure before you
   charter, even when three documents agree.**
2. **C-105's caller record was incomplete.** There are **seven call sites across six modules**, not
   four — three reach the guard via `run_apply → apply_audit_patch_payload`. `pipeline.py:82` is the
   *import* line; the call is `:116`.
3. **`src/t2pw/bench/` contains zero references to the actor-evidence guard.** No scoring path
   reaches it, so `PMC12452463/strict` and `PMC12180156/strict` **cannot** flip. Both stay
   correct-by-accident under F-147.
4. **SMOKE is no longer read-only w.r.t. the working tree** — `test_c106_…` writes to tracked
   `src/t2pw/bench/acceptance.py` and restores it in a `finally`. **Never parallelise SMOKE.**
5. **`LpxH` is still UNVERIFIED on T-107** and verified only on `runs/2026-08-02_2130`.

**F-147 remains registered and DELIBERATELY NOT CHARTERED.** Fixing it alone would flip two legs to
PASS that would then export gold-forbidden content, including `protoporphyrin IX`, which occurs
**zero times** in a 67,304-character file whose length the gold cites exactly. **The earliest unsafe
seam is Stage-1 extraction, not the driver.** Merge rule 6.

---

## 3. THE JOB — one card, then a decision, then readiness

### 3a. Card 1 — **F-155**, and it is the clearest card this sprint has had

**Five members of one class, all in `src/t2pw/curation/apply_audit_patch.py`, all measured, all with
their evidence committed.** `FINDINGS.md` § F-155 has the reproductions; do not re-derive them
before reading it.

| | Finding | Direction |
|---|---|---|
| (a) | the transport family's bare `transport` stem matches inside **"transporter"**, so `"add P as a transporter to resolve the structural inconsistency"` is **ACCEPTED** | **F-146 in a family the pinned property does not name** |
| (b) | `[^.]{0,80}` is **not** a sentence bound — `_match_fold` strips every period before the pattern runs | length bound only |
| (c) | an actor whose **name** contains an enzyme noun licenses with no cue in the span | `"LpxC hydrolase was quantified in the lysate"` → ACCEPTED |
| (d) | **C-105's own attenuation stems** carry the identical unanchored defect | `repressor`, `suppressor`, `inhibitor` falsely **refuse** legitimate catalysis |
| (e) | three anchors C-107 added that **no test covers**, exposure measured at 4/2/4 spans | coverage, not behaviour |

**(d) is the third independent instance of one defect in one file** — `mediat` inside "intermediate"
(C-107's 1f), the six stems C-107 added, and now C-105's. **Treat "is this stem anchored as a word
on both sides?" as a checklist item.**

**Charter all five together.** Each fix touches the same two functions, and a card fixing one leaves
the class open. **(a) and (d) move in opposite directions** — (a) refuses more, (d) accepts more —
so the corpus must be reported in **both directions, separately**, and a net figure is a fail.

**The pinned safety property, non-negotiable:** the **F-146 patch stays REJECTED**, and the 29-case
battery stays at **0 mismatches**. Check after **every** change, not only at the tip.

**Read C-107's two correction rounds before you write this card.** They are the best available
guide to how this file punishes a lexical fix: round 1 bound the contra to the wording its card
quoted and closed nothing; round 2's repair reintroduced the card's own finding 1f in the
over-refusal direction, against enzymes named `reductase`.

### 3b. Then the decision packet — `DECISION-PACKET-ORCH717.md`

**Q1 is the one that costs something every wave it stays open.** The harness reports a
contract-correct empty pathway as `RESULT: FAIL`, so the instrument scored the defective T-105 run
**higher** than the two correct ones either side of it. `policy_disagreement`, product owner.
**Never `PASS`** — that would make "produced nothing" indistinguishable from "produced the right
thing" on the one paper where the pipeline should produce nothing.

**Q2 (F-150) is a `gold_data_defect` with the edit written and NOT applied.** Verified both halves
this wave. **Do not apply it without product-owner authority**, and if you do get authority, the
four-step A/B in the packet is mandatory: **a Priority-1 number that moves because the gold changed
must never be reported as a pipeline regression.**

### 3c. Only then, readiness

**Do not launch T-108 or any release candidate** unless every hard blocker is closed and the
product owner has ruled on Q1 and Q2. Prepare the readiness table; do not run the candidate.

---

## 4. Process — merge gates, not suggestions

Everything through `evidence/bounded_run.py`. **Pass the explicit venv interpreter** (**F-143**: a
bare `python` is system 3.13 with no `streamlit` → 35 spurious import errors that read exactly like
a regression). `--basetemp` under `C:/t/` with the **parent pre-created** (a missing parent once
reported **382 instead of 453**). `PYTHONPATH=<tree>/src`, `T2PW_OFFLINE_CURATOR=1`,
**`PYTHONIOENCODING=utf-8`** on anything printing non-ASCII, `--heavy-lock <TASK>`.

Every job: `FINAL SURVIVING COUNT : 0`, `cleanup : success`. Reports carry **no child stdout** —
redirect and grep, **never `head`**. **Never** `taskkill /IM python.exe` or any kill by name.
**Never commit** the caches, `topics_*.txt`, `streamlit_app.py`, or `cache_snapshot/`. Stage explicit
paths; inspect `git diff --cached` before every commit; `git commit -F <file>`.

**Guard the G11 allocation on the SHAPE OF THE PATH, never on the absence of error text.** A bad
**task id** puts *error text* in your variable; a bad **label** (anything not `^[a-z0-9][a-z0-9._-]*$`
— an uppercase letter will do it) leaves it **empty**, which becomes `--json ""`. The job then runs
clean, reports zero survivors and `cleanup : success`, and produces **no artifact at all**.

```
P=$(... g11_evidence.py next --task <ID> --label <l> 2>/dev/null | tail -1)
test -n "$P" || { echo EMPTY; exit 1; }
case "$P" in *<ID>*<l>.json) : ;; *) echo "INVALID: $P"; exit 1;; esac
```

**Your own shell timeout must exceed the wrapper's `--timeout`**, or you kill the wrapper before its
cleanup `finally` runs and strand the heavy lock.

**Bash heredocs here break on apostrophes** — including quoted `<<'EOF'`. Write long text with a
file-writing tool and `cat` it. A heredoc that fails to parse executes **nothing**, silently.

**Commit the card BEFORE cutting its worktree**, so the worktree contains its own charter. Both this
wave's did.

### Review discipline — what actually caught things this wave

Fix pass/fail items **in writing before the diff exists**; `REV-106.md` and `REV-107.md` are this
wave's templates and both named the trap that later fired. The reviewer inspects the **actual diff**,
reproduces the author's numbers, runs selections **split as well as combined**, inspects the guard
that was **removed**, records predictions first, keeps failed measurements **beside** their
corrections, and **mutates every load-bearing guard itself**. **D-084: restores replay SAVED BYTES**
— use C-106's `apply_mutation` / `restore_saved_bytes`, do not hand-roll one.

**Two automatic correction rounds; a third is an explicit authority decision.** C-107 used both and
did not need a third, because I told the reviewer plainly: *if what remains is registerable, say
APPROVE and register it.*

**The transferable lessons of this wave, in the order they cost the most:**

> **1. A fix bound to the wording it was written from closes nothing.** C-107 round 1 closed the
> exact frame its card quoted and left 15 of 44 open. Bind **grammatically**, not lexically.
>
> **2. Test the obvious repair before proposing it.** REV-107's left-lookbehind hypothesis took
> false refusals only 8 → 6. A reviewer who proposes rather than measures sends the author down a
> path that half works — and would then approve it.
>
> **3. A bounded closed list flips polarity between a cue and a contra.** In a cue it under-accepts
> and is safe; in a contra it under-refuses and is not.
>
> **4. Attribute which guard refused; do not guess.** That is how C-107's author found a second
> defect site the review had not named.

**Verify a subagent's load-bearing claims yourself — and your own.** This wave: the implementer
corrected **my** 1a measurement (defeating the contra is *necessary, not sufficient*; the bare frame
admits 1 of 11, a cued frame admits 11 of 11), the reviewer corrected **my** path count, and I
verified the reviewer's blocking finding at 7 of 8 before spending a round on it. **Reading the
report is not verification, and that applies to your own reports.**

---

## 5. Before you stop

Confirm: no merge in progress · nothing staged · local = `origin/` = `ls-remote` · `main` untouched ·
`streamlit_app.py` intact at 35/2 with the expected hash · caches, `topics_*.txt` and
`cache_snapshot/` uncommitted · whole-tree G11 0 non-compliant · heavy lock absent · zero
sprint-owned Python · only the two IDE `isort` processes · no live bounded wrapper · every job
`FINAL SURVIVING COUNT : 0` / `cleanup : success` · no worktree pruned · no agent silently stalled.

**Track agent liveness separately from job liveness** — ~15 min without observable progress → status
request; ~30 min with nothing → stalled, interrupt, preserve, redispatch. **A subagent *reading*
leaves no process and no artifact.** Check the worktree for changed files before concluding
anything; both of this wave's agents looked idle while designing.

**Update `RESUME-NEXT-SESSION.md` in place, and replace this file** when your wave closes.
A G11 report certifies a job was clean and preserves **nothing** about what it found — **commit the
probe and its log, not just the report.**
