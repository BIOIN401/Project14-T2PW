# Product-owner decision bundle — prepared at integration `e616846`

**Five** decisions. The takeover brief anticipated three; **Decisions 4 and 5 surfaced from
measurement this session** and could not have been known before F-078 was measured against
current source. Each is prepared to the point where a one-line answer discharges it.

Nothing here has been written into `DECISIONS.md` — that file is append-only and
product-owner controlled.

**Nothing is blocked on Decisions 1, 2 or 4 that could have been done without them.** All
unblocked implementation, measurement and chartering was completed first, as instructed.

---

# Decision 1 — the C-010 reading

## ⚠ CORRECTED SHA — read this before ratifying

**The SHA everyone has been quoting for C-010 is wrong, and `DECISIONS.md` is append-only,
so ratifying the quoted wording would make a false fact permanent.**

F-080 records *"`LEDGER.md:114` — C-010 = 'p01 stale positional index', `MERGED` at
**`9e06360`**"*, and the takeover brief inherited it. **That is a misread of the LEDGER
table**, whose columns run `... | Base SHA | Branch | ... | Merge SHA | ...`. On row C-010:

| column | value |
|---|---|
| Base SHA | `9e06360` |
| **Merge SHA** | **`72ee20f`** |

Verified today by execution, four independent ways:

1. `git log -1 72ee20f` → *"Merge C-010 (agent/p01-stale-index): degree zero answered
   against the pre-prune snapshot"*.
2. `git log -1 9e06360` → *"Merge **C-012** (agent/p00b-driver-seam): driver.py `_drive`
   terminal-path seam"* — a different card entirely.
3. `git merge-base --is-ancestor 9e06360 72ee20f` → **true**, exactly the base→merge
   relationship the table encodes.
4. `git show --stat 72ee20f` touches precisely C-010's declared ownership —
   `src/t2pw/pipeline/strict_quarantine.py`, `tests/test_strict_quarantine.py`,
   `tests/test_strict_quarantine_real_artifact_replay.py`, `docs/change_log.md` — while
   `9e06360` touches `src/t2pw/batch/driver.py` and writes `evidence/g11/C-012/*`.

**This strengthens rather than weakens the reading.** `git log --all --merges | grep -i index`
returns **exactly one commit in the whole repository**: `72ee20f`. F-080's *"no competing
candidate exists"* is now measured rather than argued.

Registered as **F-085**.

## What is being asked

Ratify or reject:

> **"After the index fix" in `PRODUCT_CONTRACT.md:341` refers to C-010, merged as `72ee20f`.**

## The evidence, verified today

`PRODUCT_CONTRACT.md:341` (§13, LOCKED), read verbatim at `e616846`:

> *"PMC12452463 — Gold `export_rationale` records the route as chemically **broken** (EntA
> absent; nothing converts 2,3-dihydro-2,3-dihydroxybenzoate onward). Correct outcome
> **after the index fix** is `review_required` with `strict_acceptance_eligible=false`.
> **Never strict success.**"*

The antecedent is in a committed `.py` docstring, which is why three sessions missed it —
every search was `grep -rn "index fix" docs/pwml_recovery_sprint/*.md`:

* `evidence/probe_downstream_gates.py:1` — *"How far a leg gets AFTER the **stale-index fix**
  -- the honest limit of **C-010**."*
* `evidence/probe_downstream_gates.py:5-6` — *"Fixing **the index defect (C-010)** is not the
  same as producing PWML."*
* `evidence/probe_downstream_gates.py:122` — `print(f"  1. quarantine (index-fixed) : ok={result.ok}")`
* `LEDGER.md:114` — C-010 = *"p01 stale positional index"*, `MERGED`, **merge SHA `72ee20f`**
  (base `9e06360`; see the correction above).

The probe is about **PMC12452463 specifically** — the same paper as the contract row — and
states its own purpose as *"It also gives T-100's acceptance criterion its evidence base."*
T-100's acceptance criterion is *"PMC12452463 -> review_required, not strict success"*
(`TEST_MATRIX.md:477`). **The probe, the contract row and the milestone acceptance are three
statements of one thing, and the probe is the one that names the event.**

## Its exact strength, stated honestly

**This is an objectively grounded reading, not a definition someone wrote down.** No document
says "the index fix means C-010" in those words. What the evidence establishes is that C-010
is the only fix in the sprint called an index fix, that the one artifact using the phrase ties
it to C-010 **and** to PMC12452463 **and** to T-100's acceptance, and that **no competing
candidate exists.**

## Measured consequence on F-062 / T-104

**If RATIFIED:**
* `PRODUCT_CONTRACT.md:341` binds today.
* **T-104's acceptance row becomes quotable** — it requires PMC12452463 to reach the
  contractually required status, which is unquotable while the condition is undefined.
* F-062 stops being blocked on an undefined condition. Per the separate F-062 disposition,
  **F-062 needs no code card**: its mechanism was real, its proposed remedy was refused on
  evidence by F-081, and the correct repair merged as C-067. What remains is a
  **measurement**, and T-104 is where it belongs.
* **T-104 preparation unblocks immediately.**

**If REJECTED / STRUCK:**
* F-062's mechanism still stands on its own — it is read from code and reproduces — but the
  contract half of its severity goes away.
* **T-104's acceptance row needs rewriting** before T-104 can run.
* The product owner must supply the interpretation to remeasure against.

**Either answer unblocks work. The absence of an answer is what does not.**

## Recommended answer

**Ratify.** This does not overturn D-055 §6 — that entry asks the product owner to *"name the
referent or strike the condition"*, and this is the naming.

---

# Decision 2 — `round_cap_reached`, an eighth termination reason

## What is being asked

D-024 took D-005 from six named termination reasons **to seven**. C-064 (merged `d0b5d51`,
closing F-070) introduced an **eighth**: `round_cap_reached`. It needs the same treatment
D-024 gave `attempt_cap_reached`, and `DECISIONS.md` is append-only and product-owner
controlled, so it cannot be written without approval.

## Proposed decision text — drafted to D-024's exact structure

> ## D-0XX — `round_cap_reached`, an eighth termination reason · 2026-08-21 · LOCKED
>
> **Extends D-005 and D-024. Neither is reopened, amended or contradicted** — D-005's six
> named reasons and D-024's seventh keep their exact meanings, their exact strings and their
> exact denominator rule. D-005 goes from seven named termination reasons to **eight**.
>
> C-055 built the RAG loop's round controller. When the configured ceiling of rounds is spent
> the loop stops, and — correctly, since none of the seven fitted — reported **no termination
> reason at all**. `round_cap_reached` genuinely *ends* the loop, so the one stop that
> reliably terminates it was the one stop that said nothing.
>
> **The reason.** `round_cap_reached`. Used when **all** of these hold:
>
> * the configured maximum number of RAG rounds has been consumed;
> * the loop has not otherwise terminated;
> * **no** deadline or timeout caused termination;
> * **no** explicit refusal caused termination;
> * **no** separate resource / token / budget exhaustion caused termination;
> * **no** stronger existing terminal reason truthfully describes the outcome.
>
> **Precedence, mandatory: rank 8** — below `budget_exhausted`, `operation_timeout`,
> `identical_empty_response`, `scientifically_unrecoverable`, `retrieval_exhausted`,
> `no_new_claims` and `attempt_cap_reached`. A round cap is the weakest true statement about
> why a loop stopped: any of the seven above it, when true, is more informative.
>
> **Never mislabel a round cap as timeout, refusal, success, or generic budget exhaustion.**
> Equally, never mislabel it as `retrieval_exhausted` or `no_new_claims`: those require the
> configured loop to have actually completed, and a loop cut off by the ceiling is precisely
> one that did not. It is claimed on the controller's **recorded cap refusal**, not on
> `rounds_remaining == 0` — the ceiling must actually have stopped a round that wanted to run.
>
> **`OPERATIONAL_TERMINATION_REASONS` is UNCHANGED** — it stays exactly
> `{budget_exhausted, operation_timeout}`. `round_cap_reached` is **not** added to it, for the
> same reason D-024 kept `attempt_cap_reached` out: a configured ceiling is a safety limit,
> not a promise, and that is a different fact from a leg that ran out of clock. Whether the
> round cap should count in the pipeline-completion and end-to-end strict-success denominators
> is a **product decision that has not been made**; until it is, the denominator does not move.

## Why this shape and not another

D-024 is the precedent and it solved exactly this problem once already. Copying its structure
means the eighth reason inherits a ruling that has already survived a merge cycle, rather than
inventing a second, subtly different convention for the same class of event.

**The one substantive choice inside the draft is the precedence rank.** Rank 8 (bottom) is
proposed because a round cap is the weakest true statement available: if any stronger reason
is simultaneously true, it is more informative, and D-024 placed `attempt_cap_reached` at the
bottom on the same argument.

## Recommended answer

**Approve as drafted**, with the rank-8 placement. If the product owner prefers a different
rank, that is the one field worth changing and everything else can stand.

---

# Decision 3 — T-101 and T-103 live-run authorization

## What is being asked

One authorization covering both milestones, to run **live legs against `openrouter/free`**
on the product owner's existing OpenRouter key.

* **~3.8 h combined wall clock** (T-101 ~2.3 h at 6 legs, T-103 ~1.5 h at 4 legs)
* **~$0 marginal cost**

## Pre-flight checks — all re-run today at `e616846`, not copied from the package

| check | result |
|---|---|
| All nine OpenRouter model slots | ✔ **all `openrouter/free`**, re-read from `.env` today |
| `LLM_PROVIDER` | ✔ `openrouter` |
| Paid fallback enabled | ✔ **none** — no `OPENROUTER_*_FALLBACK` variable exists in `.env` |
| Credential present and well-formed | ✔ present, uncommented, parses cleanly through `python-dotenv` (`config.py:54-60`, `llm/client.py:11-22`). **Value never reproduced anywhere.** |
| `RAG_LOOP_MAX_ROUNDS` | ✔ unset ⇒ `rag_loop_max_rounds()` returns 1 ⇒ **round multiplier 1×** |
| T-101 code prerequisite (C-064, F-070) | ✔ **merged `d0b5d51`** — acceptance clause 3 is now assessable |
| T-103 code prerequisite (C-055) | ✔ merged |
| Any unfinished card touching `budget_exhausted` | ✔ **none.** In-flight cards are C-069 (`runner.CHILD_IMPORTS`) and C-070 (`pytest.ini`); neither touches it |
| `topics_t101.txt` | ✔ **created this session** — 3 scoped lines |
| `topics_t103.txt` | ✔ **created this session** — 2 scoped lines |
| Heavy mutex | ✔ free |
| Credits / rate limits usable | **UNVERIFIED** — see below |

## The one thing still unverified, and why I did not resolve it

Whether the key currently has usable credit and what its free-tier rate limits are.
**One free, read-only `GET https://openrouter.ai/api/v1/key` would settle both.** I did not
run it: it is an outward-facing call with the product owner's credential. It blocks nothing
and takes seconds. **Authorize it alongside, or separately, as preferred.**

## Exact commands, ready to run

**T-101** — deliberate 6-leg superset (`--modes` is per-run, not per-paper; a superset
satisfies the acceptance criteria in one invocation, and the two extra research legs cost
~46 min at free-tier rates):

```
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label t101-m2-legs --timeout <wall+margin> --heavy-lock t101-m2 \
  --json <allocated g11 path> -- \
  .venv/Scripts/python.exe scripts/batch_run.py \
    --topics topics_t101.txt --out runs_verify \
    --modes strict,research --timeout 1800 --deadline 3 --fresh
```

**T-103** — `T2PW_SPECIES_LLM=0` is MANDATORY (PACK 9 RULING 3; T-104 must NOT inherit it):

```
T2PW_SPECIES_LLM=0 T2PW_OFFLINE_CURATOR=1 \
.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label t103-rag-legs --timeout <wall+margin> --heavy-lock t103-rag \
  --json <allocated g11 path> -- \
  .venv/Scripts/python.exe scripts/batch_run.py \
    --topics topics_t103.txt --out runs_verify \
    --modes strict,research --timeout 1800 --deadline 3 --fresh
```

### ⚠ Correction to the authorization package's T-103 command — MEASURED

`T101_T103_AUTHORIZATION.md:169-179` puts the two variables **after** the `--` as
`env T2PW_SPECIES_LLM=0 T2PW_OFFLINE_CURATOR=1 .venv/Scripts/python.exe ...`. That makes
`env` the child executable, which `bounded_run.py` would have to resolve through Python's
`subprocess` on Windows — an unnecessary risk in a 1.5 h authorized run.

**`bounded_run.py` has no `--env` flag** (verified from `--help`: only `--timeout`, `--label`,
`--cwd`, `--grace`, `--json`, `--quiet`, `--heavy-lock`, `--heavy-lock-path`). The child
inherits the wrapper's environment, so the **shell-prefix form above is the correct one**, and
it satisfies `TEST_MATRIX.md`'s requirement to *"set it in the BOUNDED CHILD environment, not
just your shell"* by inheritance.

**Verified by execution today**, not by inference:

```
T2PW_SPECIES_LLM=0 T2PW_OFFLINE_CURATOR=1 .venv/Scripts/python.exe .../bounded_run.py \
  --label envprobe --timeout 60 -- .venv/Scripts/python.exe -u -c "<print os.environ>"

CHILD_SEES SPECIES_LLM='0' OFFLINE_CURATOR='1'
exit code (real): 0    FINAL SURVIVING COUNT: 0    cleanup: success
```

The same correction applies to the Chunk D gate command at `TEST_MATRIX.md:268-273`, which
already uses the shell-prefix form for `T2PW_OFFLINE_CURATOR=1` — so the two commands are
now consistent, and the T-103 one was the outlier.

## Sequencing

Serially, T-101 then T-103, each holding the wrapper-owned heavy mutex. **Never
concurrently** — two live-curator legs at once risks free-tier rate-limit corruption and
shared-cache races. Offline work continues in other lanes while a milestone runs.

## Explicitly NOT being asked here

**T-104 and T-105.** Each is a separate ~7 h, 20-leg release candidate, and they must never
be collapsed into one run — T-105 is the second candidate and requires a triage and
correction pass between the two. T-104 additionally depends on **Decision 1**.

## Recommended answer

**Authorize both, plus the free key-status GET.** The cost case is settled; the only real
resource is wall clock.

---

# Decision 4 — what an unverified Stage-1 row may claim (F-078)

**NEW this session.** F-078 was recorded as *"Not measured against current source."* It has
now been measured, and it splits into two halves with different statuses. Only one of them is
code, and that one cannot be chartered without this ruling.

## Half A — the chemistry. Not a code predicate. No card possible today.

Verified in the artifact at tip
(`runs_verify/2026-08-18_1328/papers/PMC12096016/strict/final_mapped.json`):

```
reactions[3] 'EntE reaction'
   in : ['2,3-dihydroxybenzoic acid', 'ATP']
   out: ['2,3-dihydroxybenzoyl-AMP', 'AMP']
```

One ATP cannot yield both an adenylylated product and free AMP; EntE releases pyrophosphate.
**F-078's chemistry claim is correct**, and confirmed against the committed paper text
(`01_source_text.txt`, 43,667 chars): `AMP` occurs **exactly once**, inside the enzyme name
*"EntE (2,3-dihydroxybenzoyl-AMP ligase; EC 6.2.1.71)"*. `pyrophosphate` occurs once, as
**thiamine** pyrophosphate in an assay buffer — unrelated. `PPi`: zero.

**But this row is Stage-1 LLM extraction output.** There is no deterministic function in
`src/` that produced it, so there is no `file:line` predicate to own and no offline proof that
today's pipeline still emits it. **A card cannot own Half A without an authorised LLM leg.**
That is a separate authorization, not a card, and it is not being requested here.

## Half B — the provenance claim. Reproduces deterministically. **This is the decision.**

`src/t2pw/pipeline/stage_one_boundary.py:311-315`:

```python
if str(row.get("provenance") or "").strip().casefold() in _PROVENANCE_NOT_READ:
    return _PAPER_STATED_UNVERIFIED
if any(row.get(key) for key in _MARKS_NOT_READ):
    return _PAPER_STATED_UNVERIFIED
return _PAPER_STATED
```

with `_PROVENANCE_NOT_READ = ("inferred", "enriched")` (`:212`) and
`_MARKS_NOT_READ = ("inference", "rag_provenance")` (`:217`). `_PAPER_STATED` (`:173-181`)
carries `paper_explicit="explicit"`, `review_required=False`.

Measured at tip, feeding the historical `AMP` row through today's `_paper_entry`:

```
{'name':'AMP','class':'cofactor','provenance':'extracted'} -> explicit, review=False
{'name':'AMP'}                                             -> explicit, review=False
{'name':'AMP','provenance':'inferred'}                     -> not_evaluated, review=True
{'name':'AMP','inference':'...'}                           -> not_evaluated, review=True
```

byte-identical to the committed `provenance_lineage[0]`.

**Any row the extraction did not self-mark is stamped `explicit` with no review flag, and the
seam has no paper text with which to check.** Its own docstring concedes this (`:168-171`):
*"this seam receives a payload, not the paper it was drawn from."*

### The question

> **Does `PRODUCT_CONTRACT.md:85-102` §3's requirement that every entity identify "whether it
> was paper-explicit" require that claim to be VERIFIED, or merely RECORDED as the extraction
> asserted it?**

* **If VERIFIED** — the current stamp is a contract violation, and an unmarked row must become
  `not_evaluated` with `review_required=True`.
* **If RECORDED** — the current behaviour is correct-as-designed, F-078 Half B closes with no
  card, and the residual is documented rather than fixed.

### Why this is yours and not mine

Two reasons, and the second is the binding one.

1. **The blast radius is product-visible.** Flipping unmarked rows to `review_required=True`
   would move an unknown but likely large number of pathways from release to review. That is a
   change to what the product ships, not an implementation detail.
2. **The boundary-safe fix cannot verify anything.** `settle_stage_one` (`:411-418`) takes
   `payload, mode, cleaning_report, reconstruct, repair_rows, chat_fn` — **no source-text
   parameter** — and its only production call site is `src/t2pw/app/streamlit_app.py:5476`,
   the file carrying the protected uncommitted product-owner edit. **So the only in-boundary
   change is to what the seam CLAIMS, not to what it CHECKS.** Choosing between two claims
   with no new evidence is a policy choice, not an engineering one.

Threading the paper text into the seam is the third option. It is a real architectural change,
it is not narrow, and it would require touching the protected file. **I did not take it, and I
do not recommend it as part of this sprint.**

## Recommended answer

**RECORDED**, with the residual documented — i.e. close Half B with no card, and register the
unearnable `explicit` claim as an accepted, known limitation of the Stage-1 boundary.

**Reasoning:** §3's stated purpose is *"so that false content can be attributed empirically to
Stage 1, RAG, inference, audit, mapping, gap resolution or another stage."* Attribution to
**Stage 1** is exactly what the current stamp achieves — it says *this came from the
extraction and the extraction did not flag it*. Changing it to `not_evaluated` would make the
field **less** informative about origin while buying no verification. The verification
capability §3 would need does not exist at this seam and cannot be added narrowly.

**If you answer VERIFIED instead**, say so and I will charter it — the ownership boundary is
already measured and narrow (`stage_one_boundary.py :: _paper_entry` plus the two module
constants at `:212` and `:217`), the G9 base-red proof is already in hand, and the one pinned
test it moves is identified: `tests/test_stage_one_boundary_lineage.py:246-256`
`test_a1_an_extracted_row_and_an_unmarked_row_are_still_reported_as_explicit`, which asserts
`explicit` for exactly the two row shapes F-078 objects to.

**⚠ Note for either answer:** all **fourteen** consuming test files for this seam are outside
the chunk table. No chunk and no SMOKE covers them. A charter must enumerate them explicitly.

---

# Decision 5 — ratify `SEMANTIC_GATING_CHECKS` 4 → 5 (C-071's merge gate)

**NEW, and it is the one decision that currently holds a completed card out of the tree.**

> ⚠ **Correction to this document's own framing.** An earlier draft of `RESUME-NEXT-SESSION.md`
> said C-071's merge was held on Decision 2. It is not — Decision 2 is `round_cap_reached`, a
> RAG-loop termination reason unrelated to semantic gating. Two ratifications were conflated.
> This is the real one.

## What is being asked

Ratify a single named addition to `src/t2pw/pipeline/release_status.py:93-98`:

```python
SEMANTIC_GATING_CHECKS: Tuple[str, ...] = (
    "requested_pathway_anchors_present",
    "organism_compatible",
    "no_real_id_or_name_conflict",
    "no_rejected_rag_reaction_reintroduced",
    # + one new check: an actor whose entity name does not appear in the span
    #   it cites as its own evidence
)
```

**4 → 5. One name. Nothing else in that file changes.**

## Why it needs you rather than me

`tests/test_semantic_release_gating.py:192-223`
`test_new_acceptance_the_gating_set_is_closed_at_exactly_four` asserts the tuple literally,
`len == 4`, `len(set) == 4`, and that four further checks are **excluded by name** — with the
stated purpose *"adding a fifth gate silently is impossible."*

**That test exists precisely to force this decision, and it is doing its job.** Under the
D-044 / D-052 §1 standing rule, *"a widening that is the mechanism of an already-granted
change is ratifiable, but it must be disclosed and ratified in the record — never absorbed
silently."*

`SEMANTIC_GATING_CHECKS` decides which papers can reach `release_ready`. Adding to it changes
what the product ships. That is policy, not implementation.

## What it is the mechanism of

**`PRODUCT_CONTRACT.md:343`, which is LOCKED**: *"Structured status is authoritative."*

F-079 measured a payload carrying a fabricated transporter — an `EntE` actor whose only cited
evidence is a span naming **TolC** — classified `release_ready`, `semantic_evaluation: passed`,
`strict_acceptance_eligible: True`. Re-measured at tip: still reproduces, byte-identical to the
committed artifact. So the fix is contract-justified; the gating-set addition is the narrowest
mechanism that delivers it.

## Why the card does not read `passed` affirmatively

F-053 forbids it and **remains undischarged**. C-071 is chartered to make the defect produce
`SEMANTIC_FAILED` through a gating check and let the existing subtractive cap at
`release_status.py:541` do the demotion. **It respects the prohibition rather than testing it.**

F-079 is itself fresh evidence F-053 should stay: `strict_acceptance_eligible` came back `True`
on that leg, which is a denominator-entry authorisation. Had any card been chartered to build a
rate on `passed`, that leg would have entered a strict numerator carrying a reaction the paper
does not state.

## The exact delta you would be ratifying

* `SEMANTIC_GATING_CHECKS` **4 → 5**, one named addition.
* `tests/test_semantic_release_gating.py:192-223` updated to assert **five**, with its closure
  property **intact** — a sixth silent addition must still be impossible — and renamed, since a
  function called `..._closed_at_exactly_four` that asserts five is worse than no test.
* Real edits expected in `test_c056b_semantic_denominators` and `test_c056c_semantic_evaluability`,
  which index and sweep the tuple by width.
* **`test_compound_resolution_extraction`'s `GOLDEN` must NOT move.** It hashes the IR built from
  the payload; a classification-only fix does not touch it. If it moves, the card left its
  boundary and the charter says stop.

## Recommended answer

**Ratify on delivery** — i.e. once C-071's diff and evidence are in hand, not now. The card is
being implemented to completion regardless, so this is a one-line unblock rather than a start
signal. If you would rather see the finished detector and its false-positive/false-negative
tests before committing, that is the natural point.

**If you decline**, F-079 stays open and unfixable at this seam: every alternative either reads
`passed` affirmatively (forbidden by F-053) or makes `CHECK_SUPPORTED_REACTIONS` applicable in
production, which would mean inventing gold signatures. Declining is a legitimate answer, but it
should be made knowing there is no narrower mechanism available.

## ⚠ Decision 5 — the measured blast radius, added after REV-071

**This is the number that should decide the answer, and it was not available when Decision 5 was
first written.**

REV-071 measured the new check against the whole committed corpus, not just F-079's leg:

> **13 of 21 committed `runs_verify` legs, and 8 of 14 `runs/` legs, now fail the new gating
> check.**

**That is a material move in strict-acceptance eligibility across the committed corpus**, not a
one-leg fix. Ratifying `SEMANTIC_GATING_CHECKS` 4 → 5 moves those legs from `release_ready` to
`review_required` with `strict_acceptance_eligible=false`.

### The reviewer's adjudication of whether that is right

**It judged the failures dominated by genuine fabrications**, and named them:

* `EntE` cited by spans naming **TolC**, **TonB**, or **EntF** — F-079's own defect class, and
  20 of 221 actor rows corpus-wide are `Ent*` symbols a naive substring test would have
  accepted because they sit inside the word *enterobactin*.
* `ALAS2 complex` cited by *"rate-limiting step for heme biosynthesis"* — a span naming no
  protein at all.
* `SFXN4 complex` cited by a span naming **ALAS**.

**The marginal cases land at `review_required`, which is the contract's answer for an uncertain
identity** — e.g. `ALAS2` cited by *"ALAS mediates…"*, and `enterobactin synthase` cited by
*"EntE (…ligase)"*. They are flagged for a human, not dropped.

### What is NOT claimed

REV-071 was explicit that **biological adjudication of the marginal demotions is not a
reviewer's call**: whether `review_required` is the *biologically* right answer for
`ALAS2` vs `ALAS` is a `pwml-bio-auditor` question. The reviewer judged them as **lexical
evidence questions**, which is what the check actually asks.

**If you want that adjudication before ratifying, say so** — it is a read-only bio-auditor
pass over the ~13 demoted legs and it does not block anything else.

### Two design residuals, recorded rather than fixed

1. **A payload where every actor row lacks a usable span makes the check `applicable=False`**,
   and an inapplicable gating check cannot demote — so such a payload still reaches
   `release_ready`. It is visible only as `NO_ACTOR_SPANS` in `semantic_check_evaluability`.
   This follows from the pre-existing D-006 architecture plus `CHECK_SOURCE_CARRIER` being
   deliberately non-gating. Closing it would change how `release_status` treats inapplicable
   gating checks — **a product decision, correctly not improvised by the card.**
2. **The multi-token hole, now quantified:** ~**14 of 373 passing rows (3.8 %)** are corroborated
   by a single non-identifying token (`complex`, `homodimer`, `synthase`, `deacetylase`,
   `disaccharide`, `udp`). It points only in the **under-reporting** direction — it never
   demotes correct output. Closing it needs a hand-built stopword vocabulary with its own drift.

### How this changes the recommendation

**It does not — but it raises the stakes, so it should be seen.** The recommendation stays
**ratify on delivery**. The check is doing what the contract asks: `PRODUCT_CONTRACT.md:343`
makes structured status authoritative, and 13 legs currently claim a status their own cited
evidence does not support.

**But 13 of 21 is a number you should approve deliberately rather than discover.**
