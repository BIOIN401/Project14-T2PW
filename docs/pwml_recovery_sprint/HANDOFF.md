# PWML RECOVERY SPRINT — HANDOFF after the T-108 execution wave

You are the **Lead Orchestrator and Integration Authority** for
`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`, integration branch `sprint/pwml-recovery`.

**Do not merge to `main`.** Work autonomously.

**T-108 has RUN, is SCORED, and is TRIAGED. Its verdict is `NOT ACCEPTED`. Do not re-run it, do not
re-score it, do not reinterpret it.** A run's verdict is a fact about the artifacts it produced.
The same holds, unchanged, for **T-107**.

Full result: **`T108-RESULT.md`**. Ownership record: **`T108-RUN-OWNERSHIP.md`**. Run manifest and
its pre-launch proofs: **`T108-MANIFEST.md`**.

---

## 1. Takeover — verify once, do not trust these numbers

**Verified integration state at the close of this wave: `479128b3`+.** The commit carrying this file
is newer. **The binding invariant is `local = origin/ = git ls-remote` — read it, do not recall it.**

| Check | Expected |
|---|---|
| local = `origin/` = `git ls-remote` | **all three equal** |
| `main` | local `7531692`, remote `03f1af5`. Advanced **outside** this sprint. **Touch neither ref** |
| merge in progress / staged | none / none |
| heavy lock `C:/t/heavylock` | **absent** — released cleanly by T-108 |
| sprint-owned Python | **zero** |
| IDE processes | `ms-python.isort` — **never cleanup targets**; **PIDs change, match on command line**. **The count is now THREE, not two** — see § 6 |
| `streamlit_app.py` | uncommitted, **35 ins / 2 del**, `sha256:47e4fafa789d359d…` |
| `acceptance.py` | `sha256:4bd893ac410d16d3…` (**CRLF working-tree** form; the LF blob is `d9f817e1…`) |
| gold blob `pinned_v1.json` | **`36f4b7b690b577f72882c3045ca6728d1ec8d9d1`** — unchanged by T-108 |
| caches, `topics_*.txt`, stray 0-byte `ValueError` | uncommitted, untouched. **Leave them** |
| `cache_snapshot/` at repo root | **does not exist**; the per-run one under `runs_verify/*/` is **gitignored** |
| **SMOKE** (22 files) | **503 passed, exit 0** |
| **gold-readers** (22 files) | **456 passed / 0 failed / 8 skipped / 0 errors, exit 0** |
| **29-case battery** | **`battery=0/29  F146=REJECTED  C1..C6 all 0`** |
| whole-tree G11 | **0 non-compliant.** The count is self-referential — **reconcile, do not match** |

Run `ListAgents` and contact every live peer before claiming the branch, the lock or a worktree.

**The peer correction from the last wave stands and was re-confirmed:** `project14-t2pw-93` is the
**same user on a different task**, not a different user. Read-only, unrelated assessment, not
authorized for sprint work. It stood down on all four points and honestly caveated that it cannot
bind its user's future instructions.

---

## 2. What T-108 established — do not re-litigate any of this

**Ran once into `runs_verify/2026-09-01_1612`. 10/10 papers, 20/20 legs, `complete: true`, 6.37 h of
a 20 h wrapper ceiling. `FINAL SURVIVING COUNT : 0`, `cleanup : success`, lock released.**

| # | Priority | T-108 | `ok` |
|---|---|---|---|
| 1 | zero known false real identifiers | raw **2** · accepted **2** · `accepted_status: PASS` (target 6) | **false** |
| 2 | zero unsupported retained reactions | **`NOT EVALUATED`** — 12 of 19 scored legs, 8 papers | `null` |
| 3 | zero referential-integrity violations | **0** | **true** |
| 4 | meaningful requested-pathway coverage | **0/8** | **false** |
| 5 | strict PWML pass rate among eligible papers | **0/2** | **false** |

### 2.1 The most important thing that happened, because it will shape your instincts

**Priority 5 is `0/2` on both T-107 and T-108, and the two zeros are not the same result.**

T-107's `0/2` was **one operational loss** (`PMC12096016/strict` timed out) **plus one coverage
shortfall**. T-108's `0/2` is **two coverage shortfalls and zero operational losses**. Both
`strict_exportable` legs executed fully, cleared the strict technical gates, **passed semantic
evaluation**, produced valid PWML, and are held at `review_required` for unmatched requested-core
anchors — completeness **0.75** (`NADH, ATP, EntD, Fur`) and **0.538**
(`oxysterols, MSMO1, SQLE, FDFT1, HMGCR, HMGCS1`).

**That is merge rule 7 working exactly as written** — incomplete-but-correct pathways preserved as
`review_required`, never dropped and never promoted. **A runner `pass` is not a Priority-5 point.**

**The number did not move; the denominator became honest.** Restoring the leg ceiling converted an
operational failure into a measurable biological result, which is precisely what `T108-READINESS`
§ 2.1 said it would do. **An unchanged headline number can conceal a completely changed diagnosis,
and reporting only the headline would have hidden the entire result.**

### 2.2 The 3600 s ceiling — it worked, and it was still not enough for one leg

Timeouts **3 → 1**. Scorable denominator **17 → 19**. Two legs recovered:

| Leg | T-107 @ 1800 s | T-108 @ 3600 s | % of OLD ceiling |
|---|---|---|---|
| `PMC12444477/research` | TIMEOUT 1800.5 s, 0 files | **PASS** 2446.5 s, 18 files | 136% |
| `PMC12096016/strict` | TIMEOUT 1800.2 s, 0 files | **PASS** 1952.9 s, PWML **74367 B**, 0 gate errors | **108.5%** |

`PMC12096016/strict` needed **152.9 s — two and a half minutes — beyond T-107's ceiling.** A core
`strict_exportable` paper's clean 74 KB export had been discarded because a ceiling was halved with
no recorded reason.

**And `PMC12444477/strict` consumed the FULL 3600 s and still timed out — F-166.** The census maximum
of 3421.4 s **was not an upper bound**. Per § 2.1's own ruling this is **not automatically a defect
and must not be waved away either**; it is censored, proving only that the leg needs *more* than
3600 s. **No ceiling change is proposed on one observation** — that would be choosing a budget from
censored data a second time, the very error § 2.1 documented while making the first choice.
**C-111's `LEG_TRACE.jsonl` for that leg survived (24564 B) and is committed**, so for the first time
in this sprint a timed-out leg left enough behind to ask *where the time went*. That profile, not
another guessed number, is what would settle it.

### 2.3 C-111 and C-113 both demonstrated themselves on live data

**C-111** preserved `LEG_TERMINAL.json` + `LEG_TRACE.jsonl` + `RESULT.txt` on the timed-out leg.
F-148's finding was that T-107's three timed-out legs **preserved nothing to inspect.** Closed,
observably.

**C-113** made a previously-invisible false identifier visible: one of T-108's two Priority-1 rows is
**`δ-aminolevulinic acid`**, the exact spelling C-113 added to the forbidden aliases. **Which is also
F-165 — see § 4.**

---

## 3. What is NOT established, and must not be quietly upgraded

- **F-146 is NOT fixed.** `PMC13231680/research` produced an empty pathway where T-107 passed. That
  is **one draw**. The standing trap forbids calling a single leg a regression at temperature 0;
  **the symmetric rule binds and forbids calling it an improvement.** The artifacts cannot separate
  *"declined"* from *"this draw extracted nothing"* — zero reactions is not a recorded refusal.
- **`LpxH` remains UNVERIFIED.** `PMC12444477/strict` timed out again; the research leg carries **0
  findings**. Verified only on `runs/2026-08-02_2130`. **Do not report T-107 or T-108 as confirming
  it.**
- **`PMC12856317/strict` `PASS → FAIL` is NOT a regression.** T-107's `final_mapped.json` for that leg
  holds exactly `ALAS1` and `ALAS2` — **no ClpXP** — so the gate had nothing to fire on. T-108's draw
  extracted ClpXP (a protease, not a heme enzyme) without an accession and the § 8 identity gate
  refused it. **The gate did not change; its input did.** Near-identical elapsed (1118.1 vs 1122.7 s)
  is consistent with a same-shape run diverging at extraction.
- **`PMC12452463/strict` blocking issues: T-106 = 7, T-107 = 3, T-108 = 6.** This **retires the
  previous wave's "improved at T-107, 7 to 3"** as draw variance. Reading any *two* of those as a
  trend is wrong in whichever direction the last two fall.
- **Priority 2 = 0 counted is the absence of a measurement**, not the absence of unsupported
  reactions. Reported as an acceptance-instrument limitation under **D-087** clause 6.

**The control that makes all of the above readable:** `PMC12782028/strict` is near-deterministic
across runs — **590.6 s vs 596.6 s**, PWML **34931 B vs 35295 B**, 0 gate errors both times. So where
the draw is stable the pipeline is stable, and **the divergences elsewhere are draw-specific, not
run-wide instability.** Without that control every divergence above would be arguable.

---

## 4. Findings registered this wave — F-165, F-166

**F-165 — T-107 and T-108 Priority-1 counts were measured against DIFFERENT GOLD SETS.** C-113 merged
**three days after T-107 ran**, adding `delta-aminolevulinic acid` and `δ-aminolevulinic acid` to
`PMC12180156`'s forbidden aliases. One of T-108's two Priority-1 rows is that spelling — **invisible
under T-107's gold**, where `forbidden_match` returned `None`. So any `8 → 2` comparison is **not
apples to apples**, and the two facts pull in **opposite directions**: the instrument got *stricter*
and the count still *fell*. **Never compare a Priority-1 count across milestones without first
checking whether the gold blob changed between them.** The cheap durable fix — **stamp the gold blob
SHA into every scored run's artifacts** — is registered, not chartered.

**F-166 — one leg needs more than 3600 s.** § 2.2 above. The restoration was **right** and it was
**still insufficient for that leg**; both halves travel together or the finding is misreported.

---

## 5. THE NEXT WORK ORDER

### 5.1 No code change is chartered from T-108

The only genuine `product_contract_violation`s are **F-147** (`PMC12180156/strict` +
`PMC12452463/strict`, one shared seam), **registered and deliberately UNCHARTERED** because a
downstream-only fix would flip both legs to PASS and export gold-forbidden content —
`enterobactin synthase complex`, `RyhB`, an efflux step the paper never describes, a
`ferrochelatase reaction`. **Merge rule 6.** T-108 saw `RyhB` and `ferrochelatase` in exactly the
predicted places and the gates refused them.

**Everything else T-108 surfaced is a gate refusing output.** Nothing in this result argues for
weakening any gate.

### 5.2 THE CODE THAT IS CAUSING THE PROBLEM — localised, F-167

**Priorities 4 and 5 are held at zero by requested-core anchors, and the anchors come from Stage 0,
not from gold.** Probe and log committed: `evidence/t108_anchor_blocker_probe.py` / `.log`,
G11 `T-108/12`.

```
unmatched anchors examined          : 10
present in the paper's source text  : 10
present in the extracted payload    :  1
named in Stage-0's OWN subprocesses :  0
```

**Three pieces of code compose into the result. Read all three before touching any of them.**

| # | Location | What it does |
|---|---|---|
| 1 | **Stage 0 preprocessor** | emits `key_compounds` + `key_proteins`, which BECOME the requested-core anchors, **and separately** `main_subprocesses`, its own account of what the pathway does. **Nothing constrains the first to be consistent with the second** |
| 2 | **`strict_quarantine.py:989-996`** | matches each anchor against the `core_terms` of **ADMITTED PROCESSES**, not against the entity list. `EntD` is **in the payload and still unmatched** because it participates in no admitted process |
| 3 | **`release_status.py:921-930`** — INCOMPLETE-CORE CAP (F-094, `PRODUCT_CONTRACT` § 13) | **ONE** unmatched anchor removes `release_ready`, *"whatever the semantic verdict says"* |

**`PMC12096016/strict` has coverage 0.75 against a 0.5 minimum, 9 core processes, semantics PASSED
and every structural gate green — and is still capped.** The coverage threshold is not what blocks
it; **the all-anchors rule is.**

**The two papers fail differently and must not be merged into one story.**

- **`PMC12096016` — anchor derivation.** The four unmatched anchors are `NADH`/`ATP` (cofactors),
  `Fur` (a transcriptional regulator) and `EntD` (an activating transferase). **None is in Stage-0's
  own five subprocesses**, and a cofactor cannot be the `core_terms` of a biosynthetic step, so the
  matcher can never satisfy them.
- **`PMC12782028` — a REAL extraction recall gap.** Stage 0 named *"mevalonate pathway"* as a
  subprocess; `HMGCS1`/`HMGCR`/`FDFT1`/`SQLE` are its enzymes; the paper names them **4-14 times
  each**; the payload contains **none of them** and the leg produced **3** core processes for a
  five-subprocess pathway. **The whole upstream mevalonate/squalene arm is missing.**

**NOT CLASSIFIED, and that is deliberate.** § 14 requires a classification before any code change and
this does not supply one. **The open question is biological:** *does a requested core for
"enterobactin biosynthesis" legitimately include `Fur` and `NADH`?* Both answers are defensible.
Note that **`gold_data_defect` is probably not even available here** — the anchors come from
`requested_core_source: "pathway_context"`, i.e. **Stage 0, not gold**. **Route to
`pwml-bio-auditor` against the committed artifacts. Do not charter a card first.**

**Three traps for whoever picks this up.**

1. **The obvious fix is the forbidden one.** Relaxing the cap, or filtering cofactors out of the
   anchor list, moves Priority 5 off zero immediately. **That is the merge-rule-6 direction** and
   F-094 created the cap deliberately. A change that improves a score is the case that most needs an
   independent reviewer.
2. **Fixing only the anchor derivation would hide `PMC12782028`'s real recall gap** while making the
   number look solved — retiring the symptom that keeps the actual defect visible.
3. **`EntD` is the instructive row** — in the payload, still unmatched. Anyone who "fixes" this by
   matching anchors against entities instead of admitted processes turns every extracted-but-unwired
   name into a satisfied anchor, and the cap stops meaning anything.

### 5.3 Priority 2 stays unevaluable until D-087's standard is met

`supported_reactions_complete` is unset on all ten cases and `max_retained_reactions` is set on
exactly two, **both negative controls**. **D-087** permits setting the flag only on a case with an
explicitly bounded, exhaustive reaction scope, certified by an **independent biological reviewer**;
**several supported reactions are not evidence of completeness**; and **if no case meets the
standard, ten unset is correct and is reported as an acceptance-instrument limitation.**
**Recorded, deliberately NOT implemented.** No agent may set this flag without a ruling naming the
case, and the flag and its audit are **one artifact**.

### 5.4 Two chartered tooling repairs still NOT taken

- **F-163 — `HeavyLock.release` is not atomic.** It unlinks `holder.json` then `rmdir`s; a kill
  between them leaves an **anonymous** lock the clearing checklist cannot address. `bounded_run.py`'s
  build hash is in **every** G11 report, so a mid-wave change breaks comparability. **T-108 held and
  released the lock cleanly, so nothing forced the issue.**
- **F-164 — C-112's recursion fix opened a false FAIL** via the allocator's `.staging/`. Any fix must
  prove the C-112 vector stays closed.

### 5.5 Registered residuals awaiting an owner

R9/R10 · REV-112's R1–R5 · REV-111's RES-1–RES-4 · R-C111-1–3. **Raised, not answered:** should
`c107_mutation_attack.py` be gated? It is run by **no gate**. **The Lead's call.**

---

## 6. Protected — do not touch

**F-147 remains registered and DELIBERATELY UNCHARTERED.** The earliest unsafe seam is **Stage-1
extraction, not the driver**. Merge rule 6. **Nothing under `src/t2pw/pipeline/` changed this wave —
T-108 was a benchmark run, not a code change. Keep that true.**

**`placeholder_backed_proteins` / Unknown-backed export** — `PRODUCT_CONTRACT` § 13 standing
disagreement. **Escalate only.**

**T-107 and T-108 both immutable. `main` untouched. `streamlit_app.py` never committed.**

**The IDE process baseline moved from two to three.** After T-108 there are **three**
`ms-python.isort` language servers, one under system `c:\python313\python.exe` rather than the venv.
All match on **command line**; none is a sprint job; **none is a cleanup target.** Recorded because
the old baseline said two and a successor checking "exactly two" would otherwise report a false
anomaly. **Match on the command line, never on the count and never on the PID.**

---

## 7. Process — merge gates, not suggestions

Unchanged from the previous handoff and re-proved by T-108. Everything through
`evidence/bounded_run.py` with the **explicit venv interpreter** (**worktrees have NO `.venv`**; a
bare `python` is system 3.13 with no `streamlit` → 35 spurious import errors that read exactly like a
regression). Real `--timeout`; **`--basetemp` under `C:/t/` with the parent PRE-CREATED**;
`PYTHONPATH=<tree>/src`; `PYTHONIOENCODING=utf-8`; `--heavy-lock <TASK>`.

**`FINAL SURVIVING COUNT : 0` and `cleanup : success` on every job.** **Exit 95 = the child never
started.** Never `taskkill /IM python.exe`. Never `pytest -n auto`; never the full suite unchunked.
**One heavy job at a time.** **The Bash tool caps a single call at 600 s** — a long job belongs in
tracked background under D-026, owned end to end.

**`T2PW_OFFLINE_CURATOR=1` is for deterministic TEST and GATE jobs. It must NOT be set for a live
benchmark leg** — it makes `run_pathway_curator` a zero-model-call no-op, and the curator is a
ratified production stage on the pinned `OPENROUTER_CURATOR_MODEL` slot. Setting it would disable
ratified biology: the `LLM_PROVIDER=local` failure mode in different clothes. **T-107 ran the curator
online — verified from its own artifact, not inferred — and T-108 matched it, preserving
comparability.** `T108-MANIFEST.md` § 5.

**G11: guard on the SHAPE OF THE PATH, never on absence of error text.** A bad task id puts *error
text* in your variable; a bad label leaves it **empty**, which becomes `--json ""` — and the job then
runs clean, reports zero survivors, and produces **no artifact at all**.

**Never put a non-`bounded_run` file inside `evidence/g11/<TASK>/`.** Probes and logs go **flat** in
`evidence/`.

**F-160 binds every test and mutation.** A same-length edit in the same second leaves the `.pyc`
valid. **Purge only `src/t2pw` and `tests`**; `PYTHONDONTWRITEBYTECODE=1` is stronger.

**Never commit:** `data/enrichment_cache.json` (39 MB, tracked), `data/id_mapping_cache.json`,
`topics_*.txt`, `streamlit_app.py`, the stray `ValueError`, or anything under `out/`, `outputs/`,
`tmp/`. **`runs_verify/*/cache_snapshot/` is gitignored — a run tree commits ~330 artifact files and
no cache.** **Stage explicit paths; inspect `git diff --cached`; `git commit -F`.**

**Do not:** merge to `main` · amend · rebase · reset · squash · prune a worktree · rewrite accepted
history · delete accepted evidence.

---

## 8. The transferable lessons of this wave, in the order they cost the most

> **1. An unchanged number can hide a completely changed result.** Priority 5 read `0/2` on both
> runs. On T-107 that was one timeout plus one coverage miss; on T-108 it is two coverage misses and
> **zero** operational losses. Reporting the headline alone would have concealed the entire finding —
> that the denominator became honest.
>
> **2. A benchmark number is a reading, and a reading has an instrument.** F-165: the gold changed
> between two milestones, so their Priority-1 counts are not comparable — and the instrument getting
> *stricter* while the count *fell* is exactly the shape that would otherwise be reported as a clean
> win.
>
> **3. Check the counterfactual before calling something a regression.** `PMC12856317/strict` went
> `PASS → FAIL`. One lookup in T-107's `final_mapped.json` showed it held no ClpXP at all, so the
> gate had nothing to fire on. **The gate did not change; its input did.** The lookup cost a minute
> and reversed the conclusion.
>
> **4. The symmetric form of a known trap is still the trap.** This sprint's rule is *"do not call a
> single leg a regression at temperature 0."* The same rule forbids calling a single leg an
> improvement — which is the one nobody wants to apply, because it takes away good news
> (`PMC13231680/research`, F-146).
>
> **5. A ceiling chosen from observed maxima is chosen from censored data, and the first run at the
> new ceiling tests it rather than confirming it.** F-166: the restoration was right *and*
> insufficient for one leg. Reporting only the flattering half would have been the easy error.
>
> **6. A control is what makes divergence readable.** `PMC12782028/strict` reproduced to within 6
> seconds and 364 bytes. Without it, every other divergence this run would have been arguable in
> both directions.
>
> **7. A self-test can pass without exercising what it vouches for.** The T-108 scorer's self-test
> ran green against a tree with **zero timeouts**, so the timeout columns it existed to validate were
> never executed — and every timeout would have rendered as `-`. **The green was real and its
> coverage was not.** Caught by reading the first completed leg instead of trusting the earlier pass.

**Update `RESUME-NEXT-SESSION.md` in place, and replace this file** when your wave closes.
**Commit the probe and its log, not just the report.**
