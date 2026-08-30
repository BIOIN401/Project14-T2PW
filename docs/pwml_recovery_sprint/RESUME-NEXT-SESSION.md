# RESUME — next session handoff

**Rewritten by the Lead Orchestrator, 2026-08-29, at the close of the T-107 triage wave
(`ORCH-716`).** Supersedes the 2026-08-28 record, which was written before T-107 ran and is now
stale in its most important section — it says T-107 is NO-GO pending a model decision. **T-107 has
run, is scored, and is triaged.** The previous content is in git history.
**`LEDGER.md` remains the single source of truth for task state.**

> **⚠ Keep this file in the repo and update it in place.** A G11 report certifies a job was clean
> and preserves **nothing** about what it found. Two load-bearing probe outputs this sprint existed
> only in dead sessions' temp directories and one probe source is gone for good.

---

## 1. Integration state

| | |
|---|---|
| Branch | `sprint/pwml-recovery` |
| **Do not pin a tip SHA here** | the invariant is **local = `origin/` = `git ls-remote`**. Read it, do not recall it |
| Merges to `main` | **none, and none permitted.** `main` local `7531692`, remote `03f1af5` — it advanced **outside** this sprint. **Touch neither ref** |
| Product-owner `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` — verified intact after every commit this wave |
| Caches, `topics_*.txt`, `cache_snapshot/` | uncommitted, untouched |
| Stray untracked `ValueError` | a shell-redirect accident predating this wave. **Left alone deliberately** — not mine to discard, and the standing rule is not to discard unfamiliar working-tree changes |
| Whole-tree G11 | **0 non-compliant.** The count is self-referential: a whole-tree check's own report is committed after it runs, so the recorded number is always one less than the tree containing it. Reconcile, do not panic |

## 2. The gate numbers every future charter needs

| Gate | Result on the integration tip |
|---|---|
| **SMOKE** (20 files) | **473 passed** |
| **gold-readers** (22 files) | **456 passed / 8 skipped / exit 0** |

Both re-measured this wave through `pinned_pytest` with `--pin-verdict` (`violations: []`) and the
heavy lock. **The gold-readers baseline changed through C-103** — any charter still saying that
selection correctly exits 1 is **stale**.

## 3. T-107 — scored, triaged, and NOT to be rerun

`runs_verify/2026-08-28_1816` · 20/20 legs · 17 scorable · 5.63 h · zero survivors.
**Overall: NOT ACCEPTED, on Priority 2 alone** (1 unsupported retained reaction, `PMC13231680`).
Priority 1 = **5**, `PASS` — the first result under 6 in the sprint.

**Full classification: `T107-TRIAGE.md`.** Read it before touching anything T-107 related.

### The four things a successor most needs to know

1. **Two of the four "degraded" strict legs are not C-099 regressions.** `PMC12452463/strict` and
   `PMC13231680/strict` **already failed in the T-106 artifacts (`runs_verify/2026-08-24_1428`)**,
   three days before C-099 merged. Only `PMC12180156/strict` genuinely turns at T-107.
   **Quote the T-106 artifacts beside T-105; T-105 alone is not a sufficient baseline.**
2. **`PMC13231680/strict`'s empty pathway is CORRECT and T-105's PASS was the false positive**
   (already registered as F-100). The gold says in terms: *"the correct pipeline outcome is an
   empty pathway plus a rejection reason."* **Never write code to recreate T-105's output here.**
3. **Three of the four "degradations" are movements toward the contract, not away from it.**
   On both negative controls the T-105 pass was the defect, and `PMC12452463` must **never** be a
   strict success under PRODUCT_CONTRACT § 13.
4. **`LpxH` is UNVERIFIED on T-107** — both `PMC12444477` legs timed out with no payload. It **is**
   verified at the merged tip on the pinned run `runs/2026-08-02_2130`. **Do not report T-107 as
   confirming it.**

### Run-once still binds

No leg of T-107 is repeated, and nothing is redrawn because its draw was unfavourable. Something
not observed is reported as *"not observed"*.

## 4. New findings this wave

| Id | Class | One line |
|---|---|---|
| **F-146** | `product_contract_violation` | Audit repair invented an enzyme to satisfy a *structural* complaint. **This is Priority 2's only failure.** Chartered as **C-105** |
| **F-147** | `product_contract_violation` | The driver fails a run on a `phase=audit_round` report the app documents as *"not a verdict about what shipped"*, and reports it under the gate that passed. **Registered, deliberately NOT chartered** |
| **F-148** | `product_contract_violation` | A timed-out leg preserves the stop reason and nothing else. **F-092 defect 3 is CLOSED** by the same measurement |
| **F-149** | audit result, no defect | Both cap tests pin **non-vacuously**. F-142's no-coverage-gap conclusion stands |
| **F-150** | `gold_data_defect` | Two gold gaps. **Prepared, NOT applied** — needs product-owner authority |
| **F-151** | `product_contract_violation` | Committing T-107 turned two tests red — `assert 72 == 62` — **in a file no gate runs.** The mutation-attack harness is unrunnable as a result |
| **F-152** | `product_contract_violation` | C-104's widened guard can abort a green file, because the pre-existing count parse reads all of stdout. Inert before, fatal after. No live exposure |

### F-147 is the one to be careful with

It is a **real** contract violation and it is **deliberately unchartered**. It fails exactly two
legs, and fixing it alone would make both pass and export content their own gold forbids —
`enterobactin synthase complex`, `RyhB`, an efflux step the paper never describes, and a
`ferrochelatase reaction` built on `protoporphyrin IX`, which the gold certifies occurs **zero
times** in a file whose length it cites to the character (67,304 — verified).

**Both legs are currently correct by accident.** The frozen-graph biological check must land
**before** the reporting fix, or the reporting fix is a regression dressed as a fix. **The earliest
unsafe seam is Stage-1 extraction, not the driver.**

## 5. Cards — both MERGED

| Card | Merge | What it did |
|---|---|---|
| **C-104** | `57e604d` | D-083's two carried follow-ons: prove C-102's deep copy (its revert mutation R5 was green), and abort the split gate on `errors > 0`. **Changes no production line.** REV-104 **APPROVE** |
| **C-105** | `afb0541` | **F-146 — T-107's Priority-2 defect, closed at the seam that caused it.** An audit patch may not add an actor to a process role it has no evidence for. REV-105 **APPROVE after three rounds** |

Post-merge on each: **SMOKE 473 passed** · **gold-readers 456 / 8 / exit 0**, zero survivors.

**C-105 took three correction rounds and the second and third were both worth it.**

* **Round 1** caught what the criteria were written for: the guard implemented **whole-name**
  matching while citing, as its authority, the repo function that implements a **shared-token**
  match *because the whole-name rule was measured wrong*. It refused **12 of 29** legitimate cases
  and **258 of 692** corpus rows, across **four** production callers rather than the one the card
  named. Fixed; the reviewer's byte-identical battery went **12 refusals → 1**.
* **Round 3 I authorised deliberately.** `mediat` let the defect class back by paraphrase — a span
  saying the protein was *inhibited* licensed it as the *catalyst*. The corpus vindicated it:
  **three distinct committed spans** were doing exactly that.

**The lesson worth carrying:** a guard that refuses *more* looks like the safe direction and is not
self-evidently safe. C-105's first draft refused a third of legitimate evidenced enzyme attachment
on three live production passes, and it would have surfaced later as fewer reactions across many
papers with nothing to attribute it to. **The preservation case is what caught it, and the original
preservation control passed only because it used a one-character protein name — the single shape a
whole-name rule always handles.**

## 5a. The one card the next session should charter first

**"Make the sprint's instruments trustworthy."** It is small, it is three related repairs in two
files, and one of them currently blocks a *mandatory* practice.

1. **F-151** — decide the census pin. **Re-pin to 72 and record why it grew**; do **not** relax to
   `>=` (see the correction inside F-151 for why — REV-104 talked me out of my own proposal).
   Until this lands, `evidence/c102_mutation_attack.py` **cannot run at all**, because it asserts a
   green baseline before mutating — and D-078/F-144 make mutation testing mandatory on every card.
2. **The same harness violates D-084 in both directions**, measured by REV-104: its
   `read_text`/`write_text(newline="")` converts all **1673 CRLF → LF**, and `git checkout --` is
   the only thing that puts them back. Replace with a saved-bytes binary restore, then **run the
   harness with all seven mutations including C-104's R5** — that is what actually discharges
   C-104's intent of leaving the next reviewer a *runnable* mutation.
3. **F-152** — scope the count parse to pytest's summary line.

**Then the C-105 follow-on card**, from REV-105's routed findings: inhibition near-synonyms defeat
the contra-cue (11 of 12 tested; `reduction of` is worst because it is itself a *catalysis* cue),
passive-with-agent fires when the agent is not the actor, 17 ordinary English `-ase` words
over-accept including a plural bypass, transport has no enzyme-noun rule, role `cofactor` refuses,
`mediat` matches inside *"intermediate"*, and 62 of 692 evidence spans exceed 5,000 characters
(max 176,375). **All permissive-direction and all strictly better than base**, which accepts every
one of them unconditionally — so this is improvement work, not a regression to chase.

## 6. Open, not blocking

* **F-150's gold edit** — exact proposal and a four-step A/B plan are written. **Requires the
  product owner.** Prediction recorded before the edit: Priority 1 rises 5 → 6, which is **still
  `PASS`** under D-073. A Priority-1 number that moves because the gold changed must never be
  reported as a pipeline regression.
* **The negative-control scoring question** — the harness reports a contract-correct empty pathway
  as `RESULT: FAIL`. `policy_disagreement`; decision packet, not a card.
* **PathBank compound ids in `_external_ids`** — escalated. Product owner decides whether
  `pathbank_compound_id` counts as a real accession for Priority 1.
* **`placeholder_backed_proteins` / `Unknown`-backed export** — PRODUCT_CONTRACT § 13 standing
  disagreement. **No agent may fix it.** Escalate only.

## 7. D-086 — usage is now MEASURED, and the gap is confirmed from the other side

Read live from the OpenRouter account (`evidence/orch716_openrouter_usage.py`):
`usage_weekly` **1.769355221**, `usage_monthly` 6.483844303, `limit` 75, `is_free_tier` **false**.

**These are cumulative ACCOUNT totals. The pipeline sends no run identifier, so the provider cannot
attribute any part of them to T-107.** T-107 is the only substantial live run in the weekly window,
so **$1.77 is a measured upper bound on its spend, not a measurement of it** — and it sits inside
the $0.62–$3.70 pre-run estimate. **Any tighter figure would be an estimate and none is offered.**

**Usage constrained nothing this wave.** No analysis was shortened and no model call was declined.

## 8. Traps this wave paid for — additional to the standing list

* **An invalid G11 label can become an EMPTY `--json` path, not just error text.** `--label
  nv-m2-armA-neutered` (uppercase `A`) was rejected, the shell captured `""`, and the job ran
  clean with **no artifact at all**. A job with no G11 report is not a passed test. **Capture the
  allocated path into a variable and refuse to run if it does not match `*<TASK>*<label>.json`.**
* **On a case-insensitive filesystem two labels differing only in case are ONE file.** The M2 pin
  verdict landed on the failed attempt's spelling — one file, the valid run's contents, the wrong
  name.
* **A gate report saying `ok: true` is not the report the driver blocks on.** T-107's every leg has
  `final_stage3_gate_report: ok true, errors []`. The verdict comes from a *different*, superseded
  report, and the failure message names the clean one. **Read `contract_reports.json` and check the
  `phase` stamp before believing a failure message's stage attribution.**
* **F-144 again, and it is why the stale-verdict probe exists.** `ALAS2` carries `uniprot: P22557`
  and `uniprot_id: None`. A reader checking only `uniprot_id` concludes the identifier is missing
  and reaches the opposite conclusion from the truth. **Check which key is populated before
  believing a null.**
* **Verify a subagent's load-bearing claims yourself.** The adjudication this wave was excellent and
  every one of its central claims held — but I re-derived each against the artifacts and the live
  `goldset` API before recording any of them as fact, and that is the standard. It cuts the other
  way too: **REV-105 corrected its author's corpus count upward**, finding a third defect-class span
  the author had missed, and **REV-104 talked me out of my own F-151 fix**. Neither would have
  happened if either had been reading a report instead of running the measurement.
* **Commit the card BEFORE cutting the worktree.** I cut `C:/t/c104` and `C:/t/c105` at `36f773c`
  and only then wrote and committed the cards, so **neither worktree contained its own charter**.
  Both implementers read it from the primary checkout and both flagged it. Harmless here; it would
  not be if a reviewer had to work from the worktree alone.
* **A guard that refuses MORE is not self-evidently the safe direction.** See § 5. This is the
  single most transferable thing this wave produced.

## 9. Peer sessions

`project14-t2pw-41` was contacted before the branch, the lock or any job was claimed, and **stood
down explicitly**: read-only reconnaissance, no lock, no worktree, no uncommitted work, no intent
to push. **Run `ListAgents` and contact every peer before treating the branch, the lock or the
worktrees as exclusively yours.**

**Prune no worktree.** Added this wave: `C:/t/c104`, `C:/t/c105` — plus everything the previous
waves listed.
