# PWML RECOVERY SPRINT — POST-TRIAGE WAVE

You are the **Lead Orchestrator and Integration Authority** for
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, integration branch `sprint/pwml-recovery`.

**Do not merge to `main`.** Work autonomously.

**T-107 has RUN, is SCORED, and is TRIAGED. Do not re-run it and do not re-score it.**
Your job is the follow-on work the triage produced.

---

## 1. Takeover — verify once

Read `CLAUDE.md`, then **`T107-TRIAGE.md`**, `RESUME-NEXT-SESSION.md`, `PRODUCT_CONTRACT.md`,
`LEDGER.md`, `DECISIONS.md`, `TEST_MATRIX.md`. **G1–G11 bind you.**

| Check | Expected |
|---|---|
| local = `origin/` = `git ls-remote` | **all three equal.** Tip at handoff **`67f68a8`** — verify the invariant, not the number |
| `main` | local `7531692`, remote `03f1af5`. Advanced **outside** this sprint. **Touch neither** |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | absent |
| sprint-owned Python | zero |
| IDE processes | two `ms-python.isort` — **never cleanup targets**; **PIDs change, match on command line** |
| `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` |
| caches, `topics_*.txt`, stray empty `ValueError` | uncommitted, untouched. The `ValueError` is a 0-byte redirect accident from an earlier wave — **leave it** |
| **SMOKE** (20 files) | **473 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** |
| whole-tree G11 | **0 non-compliant** (4582 artifacts at handoff; the count is self-referential) |

Run `ListAgents` and contact every live peer before claiming the branch, the lock or a worktree.
**Prune no worktree.** This wave added `C:/t/c104`, `c105`, `c105base`, `rev104base`, `rev104tip`,
`rev105`, `rev105base`, `rev105r2`, `rev105r3`.

---

## 2. What the last wave settled — do not re-litigate this

**T-107 (`runs_verify/2026-08-28_1816`) is NOT ACCEPTED and stays that way.** Priority 1 = **5**
(`PASS`, first under 6 this sprint); Priority 2 = **1** (`FAIL`, one eligible leg); Priority 3
`PASS`; 4/5 unmoved. **C-105 fixed the defect that caused Priority 2's failure — that does NOT
re-score the run.** A run's verdict is a fact about the artifacts it produced. Do not "re-accept"
T-107 because the code changed.

Four findings a successor gets wrong if nobody says them:

1. **Two of the four "degraded" strict legs are not C-099 regressions.** `PMC12452463/strict` and
   `PMC13231680/strict` already failed in the **T-106 artifacts** (`runs_verify/2026-08-24_1428`),
   three days before C-099 merged. **Quote T-106 beside T-105; T-105 alone is not a baseline.**
2. **`PMC13231680/strict`'s empty pathway is CORRECT and T-105's PASS was the false positive**
   (F-100). Gold: *"the correct pipeline outcome is an empty pathway plus a rejection reason."*
   **Never write code to recreate T-105's output here.**
3. **Three of the four "degradations" are movements toward the contract**, not away from it.
4. **`LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out with no payload. It **is**
   verified on the pinned run `runs/2026-08-02_2130`. **Do not report T-107 as confirming it.**

**F-147 is registered and DELIBERATELY NOT CHARTERED.** It is a real contract violation — the
driver fails runs on a superseded `phase=audit_round` report and blames the gate that passed — but
fixing it alone would make two legs pass that would then export gold-forbidden content, including
`protoporphyrin IX`, which occurs **zero times** in a 67,304-character file whose length the gold
cites exactly. **Both legs are currently correct by accident. The frozen-graph biological check must
land BEFORE the reporting fix.** The earliest unsafe seam is Stage-1 extraction, not the driver.

---

## 3. THE JOB — two cards, in this order

### 3a. Card 1 — "make the sprint's instruments trustworthy". Charter this first.

Small, three related repairs in two files, and one of them **currently blocks a mandatory practice**.

1. **F-151 — decide the census pin.** `tests/test_c102_coverage_denominator.py` tests 10 and 13
   assert `72 == 62` and are **RED at the tip**. Cause: `e77ad3d` committed T-107's ten
   `quarantine_report.json` files, taking the tracked population 62 → 72. **Re-pin to 72 and record
   why it grew. Do NOT relax to `>=`** — read the correction inside F-151 first; I proposed `>=`,
   REV-104 talked me out of it, and the reason is that those two tests pin *derived* quantities
   against the census, so `>=` would let a leg join the corpus and go unvisited unnoticed.
   The file is **in no chunk, no SMOKE, no gold-readers**, which is why it went red unnoticed.
2. **The same file's harness violates D-084 in both directions**, measured by REV-104:
   `evidence/c102_mutation_attack.py` does `read_text` + `write_text(newline="")`, converting all
   **1673 CRLF → LF**, and `git checkout --` is the only thing that restores them. Replace with a
   saved-bytes binary restore. **Then run the harness with all seven mutations including C-104's
   R5** — it currently **cannot run at all** (it asserts a green baseline), which is why C-104's R5
   entry is registered but not exercisable. D-078/F-144 make mutation testing mandatory on every
   card, so this blocks the practice, not just a file.
3. **F-152 — scope the count parse to pytest's summary line.**
   `evidence/c102_goldreaders_split.py:52` scans all of stdout, so a green file emitting the text
   "3 errors" is recorded as `errors=3`. That was **inert before C-104 and is fatal after**. No live
   exposure on the real selection today.

### 3b. Card 2 — the C-105 follow-on, from REV-105's routed findings

All **permissive-direction** and **all strictly better than base**, which accepts every one of them
unconditionally on confidence alone. This is improvement work, not a regression to chase.

* **Inhibition near-synonyms defeat the contra-cue** — 11 of 12 tested (`blockade`, `impairment`,
  `disruption`, `reduction`, `loss`, `silencing`, `sequestration`, `depletion`, `ablation`,
  `interference`, `quenching`). **`reduction of` is worst: it is itself a *catalysis* cue.**
* Passive-with-agent fires when the agent is not the actor.
* **17 ordinary English `-ase` words over-accept**, including a plural bypass — the stoplist pairs
  singular+plural for most entries but `purchases`, `showcases`, `staircases` have none.
* Transport family has no enzyme-noun rule, so `flippase MsbA` refuses.
* Role `cofactor` refuses.
* **`mediat` matches inside "intermediate"** — anchor it. Measured at 0 of 397 accepted rows relying
  on it alone, so no live effect, but it is real.
* **62 of 692 evidence spans exceed 5,000 characters, max 176,375.** Oversized actor-evidence spans
  are **upstream** of C-105 and outside its boundary — this may deserve its own card.

### 3c. Held, needing authority you may not have

* **F-150 — a `gold_data_defect` with the exact edit and a four-step A/B plan already written, and
  deliberately NOT applied.** `δ-aminolevulinic acid` ships with nine identifiers on a metabolite
  occurring zero times in its paper, and scores nothing on Priority 1 because the gold's forbidden
  aliases lack the `δ`/`delta` spelling. Also: **no gold case sets `supported_reactions_complete`**,
  so Priority 2 rests entirely on two `max_retained_reactions` ceilings, both on negative controls,
  one of which counts rows without checking content. **Prediction recorded before the edit:
  Priority 1 rises 5 → 6, still `PASS` under D-073.** A Priority-1 number that moves because the
  gold changed must never be reported as a pipeline regression. **Product owner decides.**
* **The negative-control scoring question** — the harness reports a contract-correct empty pathway
  as `RESULT: FAIL`. `policy_disagreement`; decision packet, not a card.
* **PathBank compound ids in `semantic.py::_external_ids`** — escalated; is a
  `pathbank_compound_id` a real accession for Priority 1?
* **`placeholder_backed_proteins` / `Unknown`-backed export** — PRODUCT_CONTRACT § 13 standing
  disagreement. **No agent may fix it.** Escalate only.

---

## 4. Process — merge gates, not suggestions

Everything through `evidence/bounded_run.py`. **Pass the explicit venv interpreter** (**F-143**: a
bare `python` is system 3.13 with no `streamlit` → 35 spurious import errors that read exactly like
a regression). `--basetemp` under `C:/t/` with the **parent pre-created** (a missing parent once
reported **382 instead of 453**). `PYTHONPATH=<tree>/src`, `T2PW_OFFLINE_CURATOR=1`,
`--heavy-lock <TASK>`. Every job: `FINAL SURVIVING COUNT : 0`, `cleanup : success`. Reports carry
**no child stdout** — redirect and grep, never `head`. **Never** `taskkill /IM python.exe` or any
kill by name or count. **Never commit** the caches, `topics_*.txt`, `streamlit_app.py`, or
`cache_snapshot/`. Stage explicit paths; inspect `git diff --cached` before every commit; use
`git commit -F <file>`.

**Guard the G11 allocation on the SHAPE OF THE PATH, never on the absence of error text.** Two
different failures produce two different poisons: a bad **task id** puts *error text* in your
variable; a bad **label** (anything not `^[a-z0-9][a-z0-9._-]*$` — an uppercase letter will do it)
leaves it **empty**, which becomes `--json ""`. The job then runs clean, reports zero survivors and
`cleanup : success`, and produces **no artifact at all**. It looks passed in every visible respect.

```
P=$(... g11_evidence.py next --task <ID> --label <l> 2>/dev/null | tail -1)
test -n "$P" || { echo EMPTY; exit 1; }
case "$P" in *<ID>*<l>.json) : ;; *) echo "INVALID: $P"; exit 1;; esac
```

Also: this filesystem is **case-insensitive**, so two labels differing only in case are one file.

**Commit the card BEFORE cutting its worktree.** I cut `C:/t/c104` and `C:/t/c105` at `36f773c` and
committed the cards afterwards, so **neither worktree contained its own charter**. Both implementers
read it from the primary checkout and both flagged it. Harmless there; not harmless if a reviewer
must work from the worktree alone.

### Review discipline

Fix pass/fail items **in writing before the diff exists** — `prompts/REV-104-105.md` is this wave's
template and it worked. Reviewer inspects the **actual diff**, reproduces the author's tests, runs
selections **split as well as combined**, inspects the guard that was **removed**, records
predictions first, keeps failed measurements **beside** their corrections, and **mutates every
load-bearing guard itself**. **D-084: restores replay SAVED BYTES** (`git checkout --` reverts
*more*; text-mode reverts *less* — this tree is CRLF). Two automatic correction rounds; **a third is
an explicit authority decision** — I spent one this wave and the corpus vindicated it.

**The transferable lesson of the last wave, and the reason C-105 took three rounds:**

> **A guard that refuses MORE is not self-evidently the safe direction.**

C-105's first draft refused **12 of 29** legitimate evidenced cases and **258 of 692** real corpus
rows, across **four** production callers rather than the one its card named. It would have surfaced
later as fewer reactions across many papers with nothing to attribute it to. **The preservation case
is what caught it — and the original preservation control passed only because it used a
one-character protein name, the single shape a whole-name rule always handles.** When a card
tightens a gate, the reviewer's job is to prove the *legitimate* path still works, on realistic
inputs, at scale.

**Verify a subagent's load-bearing claims yourself.** It cut both ways this wave: REV-105 corrected
its author's corpus count *upward*, finding a third defect-class span the author missed, and REV-104
talked me out of my own F-151 fix. Neither would have happened from reading a report.

---

## 5. Before you stop

Confirm: no merge in progress · nothing staged · local = `origin/` = `ls-remote` · `main` untouched ·
`streamlit_app.py` intact at 35/2 with the expected hash · caches, `topics_*.txt` and
`cache_snapshot/` uncommitted · whole-tree G11 0 non-compliant · heavy lock absent · zero
sprint-owned Python · only the two IDE `isort` processes · no live bounded wrapper · every job
`FINAL SURVIVING COUNT : 0` / `cleanup : success` · no worktree pruned · no agent silently stalled.

**Track agent liveness separately from job liveness** — ~15 min without observable progress → status
request; ~30 min with nothing → stalled, interrupt, preserve, redispatch. Note that a subagent
*reading* leaves no process and no artifact: C-105 looked stalled for 25 minutes and was designing.
**Check the worktree for changed files before concluding anything.**

**Update `RESUME-NEXT-SESSION.md` in place, and replace this file** when your wave closes.
A G11 report certifies a job was clean and preserves **nothing** about what it found — commit the
probe and its log, not just the report.
