# ORCH-734 — FINAL EXPECTED-WORKING RELIABILITY SMOKE. Result.

**Executed 2026-09-08 15:29 → 22:02, 6h01m of leg wall time across two bounded jobs** (the
second resuming the first after an operator-requested pause). Twelve strict legs, one run each,
no favourable-draw retries. Frozen configuration:
[`ORCH-734-FINAL-RELIABILITY-FREEZE.md`](ORCH-734-FINAL-RELIABILITY-FREEZE.md), committed at
`9ba4747c` **before** execution. **No production code was modified**; `git diff 045447c8 HEAD --
src/` is empty. `main` untouched. `streamlit_app.py` unchanged at `47e4fafa…`.

> ## A NEW dataset with its own identity.
> `runs_smoke/2026-09-08_1528`. **Not** ORCH-732, **not** the ORCH-724 pilot, **not** ORCH-730,
> **not** the C-120 validation. Its numbers may never be merged into any of those denominators.
> All four remain read-only and none is rescored here.

---

# 1. Headline

## 7 PWML files from 12 papers.

| | |
|---|---:|
| papers attempted | **12** |
| **meaningful cores** (defensible canonical core: `core_accepted ≥ 1` and F-179 not `no_defensible_core`) | **10** |
| **PWML files generated** | **7** |
| **serialization yield** (cores that became files) | **7 / 10** |
| structurally import-ready | **5 / 7** |

**Every PWML is `review_required`. None is `release_ready`.** Per the charter that is a product
success, not a shortfall.

**Call this a final expected-working reliability smoke cohort. It is not a population-wide
statistical rate**, and the 12-paper denominator must not be quoted as one.

The two numbers that matter are the split, not the headline. **Ten of twelve papers produced a
defensible reaction core** — extraction is not the bottleneck. **Three of those ten cores failed
to serialize**, and that is where the whole remaining problem lives.

---

# 2. Result table

| Paper | Organism | Pathway | Rxns | Core | PWML | Bytes | Import | Disposition | Failure if any |
|---|---|---|---:|---:|---|---:|---|---|---|
| `PMC9200736` | — | acarbose | 0 | 0 | NO | — | — | `timeout` | `other / leg_wall_clock_timeout` |
| `PMC10269868` | *H. pylori* | ADP-heptose | 5 | 6 | **YES** | 48,436 | **READY** | `review_required` | |
| `PMC6112128` | *S. tubercidicus* | tubercidin | 8 | 8 | **YES** | 78,492 | **FAIL** | `review_required` | dangling transport ref |
| `PMC11016064` | *H. sapiens* | carnitine | 10 | 11 | **YES** | 107,951 | **FAIL** | `review_required` | dangling transport ref |
| `PMC10055903` | *H. sapiens* | sialic acid | 5 | 5 | **YES** | 55,011 | **READY** | `review_required` | |
| `PMC8211424` | *H. sapiens* | corticosteroid | 2 | 2 | **YES** | 22,859 | **READY** | `review_required` | |
| `PMC13184244` | *N. tabacum* | nicotine | 4 | 5 | NO | — | — | `diagnostic_only` | `identity_resolution` |
| `PMC11961743` | *S. divaricata* | furochromone | 10 | 10 | NO | — | — | `fail` | **`f192_autostate`** |
| `PMC7910490` | *S. tuberosum* | solanidane | 0 | 0 | NO | — | — | `diagnostic_only` | `scope_or_guard_refusal` — **operator-authored manifest defect** |
| `PMC7402084` | *P. lunatus* | cyanogenic glucoside | 3 | 3 | **YES** | 40,294 | **READY** | `review_required` | |
| `PMC4471609` | *A. japonicus* | cycloclavine | 4 | 4 | NO | — | — | `fail` | **`f192_autostate`** |
| `PMC9544450` **CONTROL** | *E. coli* | menaquinone | 6 | 7 | **YES** | 49,436 | **READY** | `review_required` | |

Per-leg wall times ranged **13m58s to 60m01s**; total 6h01m.

## The scorer, and why it is trustworthy here

[`evidence/orch734_cohort_report.py`](evidence/orch734_cohort_report.py). Read-only. **Validated
against ORCH-732 before this cohort finished**, so it could not be tuned to its own results: it
reproduces ORCH-732's published outcome exactly — 3 PWML from 6, the same three files at
39,686 / 54,459 / 80,518 bytes, and a census whose repeated mechanism is `identity_resolution`,
which is ORCH-732 § 5's own finding in its own words.

F-179 verdicts are **recomputed** by calling `evaluate_reaction_support` on each leg's own
`final_mapped.json`, not scraped from a report string.

---

# 3. F-179 held, and it refused nothing

**Every leg that produced a payload returned `verdict: supported`. Zero `no_defensible_core`.**

F-179 blocked no pathway in this cohort and admitted no fabrication. That is the load-bearing
biological safeguard behaving exactly as intended: present, active, and not over-refusing. **No
finding here justifies touching it, and none is claimed.**

---

# 4. Failure breakdown by mechanism

| mechanism | papers | repeats? |
|---|---:|---|
| **`f192_autostate`** | **2** | **YES — and it is the only one** |
| `identity_resolution` | 1 | no |
| `scope_or_guard_refusal` | 1 | no — **and it is my defect, not the system's** |
| `other / leg_wall_clock_timeout` | 1 | no |

Only one mechanism repeats across independent papers, and it is F-192.

## 4.1 `PMC7910490` — the scope conflict is MINE, and it is not scored against the system

```
Stage 0 read pathway 'potato solanidane glycoalkaloid biosynthesis'
which does not match the requested 'steroidal glycoalkaloid biosynthesis'
```

I wrote the manifest scope string as *"steroidal glycoalkaloid biosynthesis"*; the paper's own
language is *"potato solanidanes"*. **Stage 0 read the paper correctly, compared it to the
request, and refused.** The guard did the right thing. My own manifest header carries the rule I
then broke — *"each scope string is taken from the paper's own language to avoid a Stage-0
`scope_conflict`"* — copied from ORCH-732's freeze, which recorded the same trap closing a pilot
leg.

**It was NOT re-run with a corrected scope.** Changing the input after seeing the result is the
favourable rerun the charter forbids, and it would launder an operator error into a success. It
stays in the denominator with its cause named. **A reader computing a system-attributable rate
should note that one of the twelve never received a fair test, and that this is my fault.**

> **The scorer misfiled this on its first pass**, as `stage1_provider_delivery`, because the leg
> records a Stage-0 verdict at stage `stage1` and the delivery rule matched first. That would
> have **manufactured a Stage-1 problem on a leg where the provider was never asked to extract
> anything.** Corrected to key on the issue code; ORCH-732 re-validated unchanged afterwards.

## 4.2 `PMC9200736` — a deep-pipeline wall-clock timeout, NOT a Stage-1 failure

Killed at the 3600 s ceiling. The trace refutes the obvious reading: **Stage 1 succeeded** (2
non-empty completions), Stage 2 ran, the audit loop ran, the gap resolver ran, RAG prose
extraction ran 37 times. This is ORCH-733's **class T**, which that census was explicit is
*"recorded at stage1, not a delivery failure."*

Its 67 empty `finish_reason=length` completions are spread across `chat`, `rag_prose_extraction`
and `gap resolver` — the corpus-wide empty-completion pathology ORCH-733 measured at **33 % of
all attempts but only 7.8 % of Stage-1 ones.** Quoting them as a Stage-1 problem would be the
fourfold overstatement ORCH-733 warned against.

The 3600 s ceiling is **identical to ORCH-732's**, so this is like-for-like. It was **not**
changed mid-cohort; doing so would have voided the comparison.

## 4.3 `PMC13184244` — identity resolution, one instance

Three proteins missing accessions blocked a 4-reaction nicotine pathway at
`final_pre_export_stage3_gates`: `UGT1`, `β-GD1`, `MATE1`. `gate_errors = 0`; the core and F-179
were sound.

Two shapes here resemble C-120's territory and are **registered as questions, not conclusions**:
`UGT1` is a **gene-symbol family stem**, the exact shape C-120's rung-4
`gene_symbol_family_identity` was added to admit (`PSAT` → `PSAT1`); and `β-GD1` carries a Greek
β, a representation issue of the same species as the bracket-fragment bug C-120 fixed for species
names. **Why rung 4 did not fire here has not been measured**, and per C-120's own lesson —
*replay the payload, do not inherit the narrative* — no cause is asserted. **One instance does
not meet the repeat threshold and nothing is chartered on it.**

---

# 5. THE FINDING: F-192 repeats, and the stopping rule's condition is met

## 5.1 Two independent instances in this cohort, three overall

| | `PMC11961743` furochromone, *S. divaricata* (plant) | `PMC4471609` cycloclavine, *A. japonicus* (fungal) |
|---|---:|---:|
| Stage-1 `biological_states` | 0 | 0 |
| `initial_stage3` `n_autostate_created` | **1** | **1** |
| `n_entities_assigned_to_autostate` | **0** | **0** |
| `__auto_state__` removal reason | `state_unreferenced_after_quarantine` | `state_unreferenced_after_quarantine` |
| `final_stage3` `n_autostate_created` | **0** | **0** |
| `final_mapped biological_states` | 0 | 0 |
| required-field gate | `no_biological_states` | `no_biological_states` |
| F-179 | `supported` | `supported` |
| **reactions destroyed** | **10** | **4** |

**Byte-identical signature on two unrelated papers** — an Apiaceae secondary-metabolite pathway
and a fungal ergot alkaloid pathway — plus `PMC9544450` in the C-120 post-merge validation.
**Three independent instances, two of them here.**

The standing instruction was explicit: *"Do NOT fix unless a second independent instance
appears."* **It has appeared.**

## 5.2 Root cause, measured rather than inferred

`ensure_autostates` (`process_normalizer.py:3122`) is called **exactly once**, at
`process_normalizer.py:5432`. It creates `__auto_state__` and assigns it **only to
`element_locations` rows that currently carry no `biological_state`** — the
`if state: continue` at `:3170`.

The strict quarantine sweep (`strict_quarantine.py`, stage `pre_export_strict_quarantine`) runs
**later** and removes biological states that have become unreferenced once entities are
quarantined. **Nothing re-runs `ensure_autostates` afterwards.** The required-field gate then
fails on `no_biological_states` and the export is refused.

`PMC11961743` shows the full sequence: the sweep removed **both** `cytosol state` **and**
`__auto_state__`, each `state_unreferenced_after_quarantine`, leaving zero states behind.

**The mechanism is deterministic; its TRIGGER is draw-dependent.** It bites only when the sweep
orphans *every* remaining state, which needs Stage 1 to have emitted no state of its own.

## 5.3 The control earned its place, and it says something the two failures cannot

`PMC9544450` was included as the F-192 sentinel because it failed on exactly this in the C-120
validation. **This time it passed** — 49,436 bytes, 6 reactions.

That is not a contradiction; it is the measurement. **F-192's trigger is stochastic while its
mechanism is fixed.** The same paper, the same code, a different draw, a different outcome. It
follows that F-192's recorded population of *"1 blocked leg in 144"* **undercounts the latent
exposure**: it measured how often the trigger fired in one archive, not how often the defect is
reachable. **Do not quote 1-in-144 as this defect's rate.**

## 5.4 Against the stopping rule's four conditions

| condition | verdict |
|---|---|
| repeats across multiple independent papers | ✅ 2 here, 3 total |
| directly suppresses PWML generation | ✅ both legs had sound biology, sound reaction support, `gate_errors = 0`, and died **only** at the required-field gate |
| clear deterministic root cause | ✅ § 5.2 — a single call site, a later sweep, no re-run |
| narrowly fixable without weakening F-179 or another biological safeguard | **PLAUSIBLE, NOT ESTABLISHED** |

The fourth is where honesty is required. A candidate seam exists — re-establish the auto-state
after the quarantine sweep when surviving `element_locations`/transports still need one — and it
would touch neither `reaction_support.py` nor any admission or scoring gate, because a
compartment placeholder admits no reaction and changes no pathway content. **But I have not read
that seam's full blast radius, and this is exactly the point at which C-120 recorded that a prior
report's *framing* can be wrong even when its *facts* are right.**

**Therefore: chartered, not implemented.** See § 9.

---

# 6. A SECOND repeated defect, in a different dimension: dangling transport references

**Two of the seven PWML files carry an internal reference that resolves to nothing**, and both
would likely be rejected by a real importer.

```
PMC6112128    transport-compound-visualization -> compound-location-id 55   (declared: 88-101)
PMC11016064   transport-compound-visualization -> compound-location-id 66, 77
```

| file | transport visualizations | dangling refs |
|---|---:|---:|
| `PMC10269868`, `PMC7402084`, `PMC8211424`, `PMC9544450` | 0 | 0 |
| `PMC10055903` | **2** | **0** |
| `PMC6112128` | 2 | **2** |
| `PMC11016064` | 2 | **2** |

**Transports are necessary but NOT sufficient** — `PMC10055903` carries two and dangles nothing —
so this is not "transports are broken". Something specific to how those two were constructed
leaves the compound-location undeclared.

**This does not affect the PWML yield number: both files exported successfully and are counted as
successes.** It is an *import-validity* defect, invisible to every gate, and it is the reason the
import check was worth running separately from the yield count.

> **This shape was invisible to the instrument's own validation set.** All four known-good files
> used to validate the checker — the three ORCH-732 PWMLs and the C-120 PSAT file — carry **zero**
> transports. A validation corpus that never exercises a feature cannot certify it, which is why
> the new cohort found this and the baseline could not.

It does **not** meet § 13's bar for the one permitted narrow fix, because it does not suppress
PWML generation. **Registered, not chartered.**

---

# 7. PathWhiz import validation

[`evidence/orch734_pathwhiz_import_check.py`](evidence/orch734_pathwhiz_import_check.py), over
**11 files** — the 7 new, the 3 from ORCH-732, and the C-120 PSAT file, as § 14 requires.

| set | READY | FAIL |
|---|---:|---:|
| this cohort | 5 | 2 |
| ORCH-732 (`PMC12051542`, `PMC4725005`, `PMC9544450`) | 3 | 0 |
| C-120 (`PMC10031235`, the PSAT file) | 1 | 0 |
| **total** | **9** | **2** |

Both failures are the § 6 dangling transport reference. Everything else passes: XML accepted,
visualization envelope present, non-zero canvas, positioned elements with real edge paths,
compounds present, two-sided reactions, named species with taxonomy id.

## The boundary, stated plainly

**`IMPORT READY` is not `IMPORT PASS`.** PWML is self-contained — species, entities, reactions and
the full drawing are inside the file — so every property the importer checks is decidable from the
bytes, and that is what was decided. **What was not done is logging into PathWhiz and pressing
Import**, which needs the product owner's account. That step is theirs, and these five files are
the recommended subset:

```
runs_smoke/2026-09-08_1528/papers/PMC10269868/strict/pathway.review_required.pwml    48,436 B
runs_smoke/2026-09-08_1528/papers/PMC10055903/strict/pathway.review_required.pwml    55,011 B
runs_smoke/2026-09-08_1528/papers/PMC7402084/strict/pathway.review_required.pwml     40,294 B
runs_smoke/2026-09-08_1528/papers/PMC9544450/strict/pathway.review_required.pwml     49,436 B
runs_validation/c120/2026-09-08_1240/papers/PMC10031235/strict/pathway.review_required.pwml  48,401 B
```

**`IMPORT FAIL` is conclusive in the other direction** — a dangling reference cannot resolve.

**All eleven files are UNTRACKED and rest on a single disk**, the `F-187` exposure. **Back them up
before relying on them.**

---

# 8. Lightweight biological spot check

§ 16 asks only for obvious fabrication, wrong organism, wrong pathway or a catastrophic missing
segment. **None of the four appears in any of the seven files.** The organism and taxonomy id are
correct in every case.

* **`PMC10055903`** *H. sapiens* (9606) — UDP-GlcNAc → ManNAc → ManNAc-6P → Neu5Ac-9P → Neu5Ac →
  CMP-Neu5Ac, with GNE/MNK and sialic acid synthase, PEP as co-substrate. **Textbook-correct.**
* **`PMC11016064`** *H. sapiens* — N6-trimethyllysine → 3-hydroxy-N6-TML →
  4-trimethylaminobutyraldehyde → γ-butyrobetaine → carnitine, with TMLD, BBD, TMABADH **and
  SHMT1/SHMT2**, which is the paper's own central finding, plus glycine as the aldolase co-product.
  **Correct, and it captured the discovery.**
* **`PMC9544450`** *E. coli* — chorismate → isochorismate → SEPHCHC → iso-SEPHCHC → SHCHC → DHNA
  with MenD and MenH. **Correct**, and consistent with ORCH-732's independent read.
* **`PMC10269868`** *H. pylori* — sedoheptulose-7-P → heptose-1,7-bisP → heptose-1-P →
  ADP-D-glycero-β-D-manno-heptose → ADP-L-glycero-β-D-manno-heptose, with GmhA, HldE, GmhB.
  **Correct.**
* **`PMC7402084`** *P. lunatus* — valine/isoleucine → oxime → α-hydroxynitrile aglycone →
  linamarin/lotaustralin, CYP79D71, CYP83E46/47, UGT85K31. **Correct.**
* **`PMC6112128`** *S. tubercidicus* — GTP → H2NTP → CPH4 → CDG → … → tubercidin, with PRPP.
  **Correct 7-deazapurine route.**
* **`PMC8211424`** *H. sapiens* — progesterone → 17α-hydroxyprogesterone, 11-deoxycorticosterone,
  11-deoxycortisol, CYP21A2. **Correct but thin: 2 reactions.**

**Minor defects, which § 15 says are NOT reliability failures:** an `Unknown` protein appears in
`PMC10269868`, `PMC11016064` and `PMC6112128`; `PMC6112128` carries the paper's own placeholder
names `compound 1/2/3/5`; `PMC8211424` is thin at 2 reactions; `PMC10269868` includes the
regulator CsrA among the pathway proteins. **These are redundancy, thinness and naming weakness —
not invention.** No `glycine → heme`-class fabrication appears anywhere.

**This is not the publication manual review.** That comes afterwards.

---

# 9. Verdict against the stopping rule

## The criteria, one at a time

| § 13 criterion | verdict |
|---|---|
| a clear majority of reasonable papers generate PWML | **7/12 = 58 %.** A majority, but a thin one — see the honest reading below |
| generated files are valid XML | ✅ **7/7 parse**, `super-pathway-visualization` root |
| representative files import into PathWhiz | ⚠️ **5/7 structurally ready; 2 carry a dangling reference.** The live UI import is the product owner's step |
| failures distributed across isolated/stochastic mechanisms | ❌ **No.** F-192 accounts for 2 of the 5 no-PWML legs |
| no single deterministic mechanism repeatedly destroys otherwise-valid cores | ❌ **F-192 does exactly this**, twice, on 10- and 4-reaction cores |

## The honest reading of 7/12

Of the five misses: **one is my manifest defect** (`PMC7910490`), **one is a wall-clock timeout**
that Stage 1 survived, **one is a single-instance identity failure**, and **two are the same
deterministic defect**. Excluding only my own error, the system was asked eleven fair questions
and answered seven with a file — and **two of the four remaining misses share one cause.**

That framing is offered as context, not as a better headline. **The reported number is 7 of 12.**

## RELIABILITY PHASE IS NOT YET COMPLETE — by the charter's own rule

I am **not** declaring `RELIABILITY PHASE COMPLETE`. The charter conditions that declaration on
the stopping rule being satisfied, and **two of its five criteria fail on the same mechanism.**

The charter also anticipated precisely this outcome and authorized precisely one response:

> **One final narrow fix may be considered ONLY if** one mechanism repeats across multiple
> independent papers, directly suppresses PWML generation, has a clear deterministic root cause,
> and can be fixed narrowly without weakening F-179 or other biological safeguards.

**F-192 meets the first three outright and is plausible on the fourth.** This is the single
narrow fix the charter reserved, and there is exactly one candidate — not five unrelated edge
cases, which § 6 of the ORCH-732 charter forbids chartering.

## What I did NOT do, and why

**I did not implement it.** Three reasons, in order of weight:

1. **`CLAUDE.md` is explicit**: the Lead Orchestrator *"does not implement coding patches and
   never approves its own work."* A fix authored and merged by me would violate merge gate 5
   before it was written.
2. **Production is frozen** at `045447c8` under `D-090`/`D-098`, and § 9 of that decision says
   *"no further production change is authorized, and no second optimization card automatically
   follows."* Unfreezing is the product owner's decision, not a consequence of my measurement.
3. **The fourth criterion is unproven.** Chartering on three-of-four is right; *merging* on
   three-of-four is not.

**The decision is the product owner's.** Both paths are defensible: charter one narrow C-card for
F-192 and then close reliability, or accept 7/12 with F-192 registered and move to RAG v2 with a
known, characterized, non-fabricating defect. **What is not defensible is declaring the phase
complete while a criterion the charter wrote down is failing.**

---

# 10. Findings registered by this run

| id | finding | disposition |
|---|---|---|
| **F-194** | **F-192 has a second and third independent instance and a measured root cause.** Deterministic mechanism, stochastic trigger. `ensure_autostates` runs once, before a sweep that can orphan every state, and is never re-run | **CHARTER CANDIDATE — meets 3 of 4 stopping-rule conditions** |
| **F-195** | **`transport-compound-visualization` can emit a `compound-location-id` the document never declares.** 2 of 3 transport-carrying PWMLs affected; invisible to every gate; would break a real import | REGISTERED, not chartered — does not suppress generation |
| **F-196** | **The batch progress tally prints `N NO DELIVERABLE` for legs that DID produce a deliverable.** `report.py:446` defines `warned` as *passed and (warnings or file_errors)*, and `:1113`/`:712` label it `NO DELIVERABLE` / `pass, no deliv.`. Every leg here carrying only the informational, explicitly non-blocking `entity_missing_mapping_meta` finding was counted as having produced nothing | REGISTERED. **A reader trusting the runner's own tally would have scored this cohort at 0 deliverables with 7 PWMLs on disk.** Reporting defect only; no production behaviour is affected, and ORCH-732's published 3/6 counted files and stands |
| **F-197** | **A Stage-0 `scope_conflict` verdict is recorded at stage `stage1`**, so a stage-name-keyed classifier files it as a Stage-1 delivery failure. Caught in this run's own scorer before publication | REGISTERED — affects analysis tooling, not production |

**None of these was fixed.** F-192 remains unfixed per instruction until the product owner rules.

---

# 11. The Stage-1 alternate-model rung — activated, proven, and it never fired

`OPENROUTER_EXTRACTION_FALLBACK_MODEL = google/gemini-3.8-flash`. Zero `src/` diff; `.env` is
gitignored. The six-phase liveness proof is at
`evidence/g11/ORCH-734/02-rung3-liveness-proof.json` and all six phases pass, including the base
arm reproducing the corpus's `strategy_not_materially_different` refusal and a live call the
model answered.

**In twelve legs, rung 3 issued zero calls, and that is the correct outcome.** No leg suffered a
recoverable Stage-1 delivery failure of class B/C/D. The one Stage-1-adjacent failure
(`PMC9200736`) was a wall-clock timeout **after Stage 1 had already succeeded**, which rung 3 is
not for and could not have helped.

**No reliability improvement is claimed from this activation, and none is measurable from this
cohort.** It remains what ORCH-733 said it was: a zero-diff configuration change that gives a
qualifying leg one more draw where it previously got none. Whether that draw succeeds is still
unmeasured. **Class A still cannot reach rung 3.**

---

# 12. Process and provenance

* **Both jobs ran through the bounded wrapper**, `FINAL SURVIVING COUNT : 0`, `cleanup : success`,
  heavy lock `ORCH-734` acquired and released. Reports:
  `evidence/g11/ORCH-734/02`, `03`, `05`.
* **The operator paused the run after leg 6** to restart the machine. `TaskStop` killed the
  wrapper before its `finally`, which **stranded the heavy lock**; the lock was verified
  dead-PID-owned by this task and released, and zero owned processes survived. Pre-existing
  Streamlit and isort processes were **reported and never killed**.
* **Leg 7 was re-run after the pause.** This is *not* a favourable-draw retry: it had no manifest
  row, was never scored, and an operator-requested kill is exactly the *objectively invalid
  infrastructure execution* the ORCH-724 rule excepts. Its partial trace is preserved as
  `LEG_TRACE.interrupted-by-operator-pause.jsonl`.
* **The resume reused the same run directory** — `CONTINUING the incomplete run 2026-09-08_1528`,
  `already recorded : 6`. One dataset, not two.
* **No leg was re-run to seek a better draw. No configuration was changed mid-cohort.**
