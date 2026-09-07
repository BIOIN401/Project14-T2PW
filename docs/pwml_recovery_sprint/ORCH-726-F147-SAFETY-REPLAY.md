# ORCH-726 — F-147 safety replay. Archived payloads, current HEAD validators.

**Read-only, evaluation-only, 2026-09-07.** No production edits. No pipeline leg re-run. No
LLM draw. No cache written. Branch `sprint/pwml-recovery` @ `f732b9ea`.

Instrument: `evidence/orch726_f147_safety_replay.py` / `.log` ·
`evidence/g11/ORCH-726/02-safety-replay-v2.json`. Bounded, `FINAL SURVIVING COUNT : 0`.

---

# VERDICT

> # `F-147 STILL UNSAFE`

**One of the two historical bad cases is now closed by a live gate. The other is not, and it
would export today with all four of its gold objections intact.**

| F-147 case | closed at HEAD? | by what |
|---|---|---|
| `PMC12180156` — ferrochelatase / protoporphyrin IX | **YES** | **`F-179`** `no_defensible_reaction_support` |
| `PMC12452463` — enterobactin synthase complex / RyhB / secretion / Unknown | **NO** | nothing objects to its content |

---

# 1. What was replayed, and on what

For each leg: the archived canonical `final_mapped.json`, pushed through the **current HEAD**
deterministic validators in production order —

1. `process_normalizer.run_strict_post_normalization_gates` — the stage-3 gate
2. `stage_contracts.validate_post_normalization` — the contract
3. `stage_contracts.validate_pre_export` — **the final semantic gate**, which embeds F-179 at
   `stage_contracts.py:364-380`, and whose inner `pwml_contract_report.ok` is what
   `run_pwml_export` actually decides on
4. `reaction_support.evaluate_reaction_support` / `reaction_support_issue` — F-179 itemised

then a simulation of `batch/driver.py::_blocking_reports` over the archived
`contract_reports.json`, run twice: as the driver does it today, and with the
`phase: audit_round` snapshot excluded. **The difference between those two runs is exactly
the proposed F-147 fix, evaluated with no code change.**

## Instrument calibration — stated because it changes the answer

`validate_pre_export` on a bare canonical payload raises two errors the production path never
sees: `pathway_missing_name` and `pathway_missing_subject`. The app attaches pathway metadata
before calling the gate. This is **calibrated, not assumed**: the real
`pwml_required_field_gate_report.json` of both legs that actually shipped a PWML —
`2026-09-06_1425/PMC12071552/strict` and `2026-09-02_2052/PMC12180156/strict` — records
`ok: true, errors: 0` with neither code present.

Those two codes are therefore subtracted as instrument noise. **Every other pre-export error
is kept.** Before the subtraction the control leg — which demonstrably exported — read as
blocked, which is how the artifact was found.

## What this replay cannot answer

The two development payloads were produced by **pre-C-118** code (`C-118` merged `a5ffdbeb`,
2026-09-06; the newest archive of either leg is 2026-09-02). C-118 *appends* a relation
template, so at HEAD these papers could admit **more** reactions than any archive holds.

**This answers "do the current gates refuse the archived content". It does not answer "what
would the current pipeline produce for these papers".** The second needs a fresh run, which
the charter forbids and which nothing here substitutes for.

---

# 2. Results

| leg | archive | rx | stage-3 | pre-export (substantive) | F-179 | stale dropped | would export? |
|---|---|---:|---|---|---|---:|---|
| `PMC12452463/strict` | 2026-08-28 (**the F-147 registration payload**) | 5 | ok | **ok** | **supported, clean** | 0 | **YES** |
| `PMC12452463/strict` | 2026-09-02 (newest) | 3 | **FAIL** | ok | supported, clean | 0 | NO — stage-3 |
| `PMC12180156/strict` | 2026-08-28 | 1 | ok | FAIL | **`no_defensible_reaction_support`** | 0 | **NO** |
| `PMC12180156/strict` | 2026-09-02 (newest) | 1 | ok | FAIL | **`no_defensible_reaction_support`** | 0 | **NO** |
| `PMC7232280/strict` | 2026-09-06 | 5 | ok | ok | supported | 0 | **YES** |
| `PMC8510960/strict` | 2026-09-06 | 5 | ok | ok | supported | 0 | **YES** |
| `PMC12376012/strict` | 2026-09-06 | 15 | ok | **FAIL** | supported | 0 | **NO** |
| `PMC12071552/strict` *(control, did export)* | 2026-09-06 | 2 | ok | ok | supported | 0 | **YES** ✔ |
| `PMC11172790/strict` *(thin)* | 2026-09-06 | 0 | ok | ok | supported | 0 | YES (empty) |

The control returning **YES** is the calibration check: the instrument agrees with reality on
the one leg whose real answer is known.

**In every one of the nine rows, dropping the `audit_round` snapshot takes the contract error
count to 0.** The blocking, where it survives, comes entirely from live gates.

---

# 3. `PMC12180156` — case B is closed, and independently corroborated

Current HEAD refuses it on both archives:

```
[4] F-179 reaction_support   verdict=... issue=YES -> no_defensible_reaction_support
[5] provenance               reactions=1   rows with NO lineage=0
```

The surviving payload is a single reaction, `glycine → heme`, catalysed by `ALAS2 complex`
**and** `ferrochelatase complex`, both `provenance: inferred` — a one-step collapse of the
entire heme pathway. F-179 refuses it for having no defensible core.

**This is not my instrument's opinion.** The committed `evidence/f179_repair_regression.json`
already lists, under `previously_exported_now_blocked`:

```json
{"leg": "runs_verify/2026-09-02_2052/papers/PMC12180156/strict",
 "population": "preserved_untracked",
 "pwml": ["pathway.review_required.pwml"], "reactions": 1}
```

Four of the five "previously exported, now blocked" legs are `PMC12180156`. **That leg
genuinely did export a PWML on 2026-09-02, and F-179 now blocks it.** `PMC12452463` is not
mentioned in that regression at all.

**Note the direction of the finding:** `protoporphyrin IX` is *not* in the 2026-09-02 PWML
that shipped (`present=False`); `ferrochelatase` is. The current refusal is not because the
hallucinated metabolite is still there — it is because the reaction has no defensible support.
Either way, F-179 blocks it.

---

# 4. `PMC12452463` — case A is **not** closed

## On the payload F-147 was registered from (`2026-08-28_1816`)

Every live gate at HEAD passes. F-179 says **supported** — all four reactions classify
`target_paper` with `paper_stated` lineage. With the stale report dropped, contract errors go
to **0**. **It would export.**

And it would export this:

```
RX  Assembly of enterobactin (synthase complex)
      2,3-dihydro-2,3-dihydroxybenzoate -> enterobactin
      enzyme = 'enterobactin synthase complex'          <- gold: forbidden_identifier

TRANSPORT  Enterobactin secretion   cargo=enterobactin   transporters=[]
      <- gold: "Export of enterobactin from the cytoplasm is never described
               at all, so no efflux step may be emitted"

INTERACTION  RyhB inhibits EntC                          <- gold: forbidden_identifier
INTERACTION  RyhB inhibits EntF                             "A small RNA, not a protein
                                                             and never an enzyme"

PROTEINS  [..., 'Unknown']    <- gold: unknown_backed_proteins_acceptable: false
```

**All four F-147 objections, present and unblocked.** `PRODUCT_CONTRACT` § 13 additionally
rules this paper *"Never strict success."*

**Why F-179 does not catch it:** F-179 reads reaction *provenance* — whether a defensible core
exists — and the forbidden reaction carries `paper_stated` lineage with a quoted evidence span.
F-179 is not, and was never designed to be, an identity or entity-class gate. It is the right
rule doing its own job.

## On the newest payload (`2026-09-02_2052`) — genuinely better, still not safe

Three of the four objections have gone from the extraction itself:

| objection | 2026-08-28 | 2026-09-02 |
|---|---|---|
| `enterobactin synthase complex` reaction + entity | present | **gone** |
| `Unknown`-backed protein | present | **gone** |
| `RyhB inhibits EntC` / `EntF` | present | **changed** to `RyhB downregulates ent mRNA` (entity_2 is mRNA, not an enzyme — this shape is defensible) |
| **`enterobactin secretion`, `transporters=[]`** | present | **STILL PRESENT** |

**The surviving objection is the one nothing objects to.** That transport is blocked today
only because the leg fails the stage-3 gate on an unrelated technicality:

```
/normalization_stats/n_plus_tokens_remaining  Expected 0 plus tokens, found 1.
/entities/compounds/5/name has '+' token: ferric iron (Fe3+)
```

**A `Fe3+` string is not a safety property.** Fix the plus-token and this leg exports with an
efflux step the gold says is never described in the paper. That is not a gate; it is luck.

---

# 5. The three unseen legs — deterministic replay

## `PMC7232280/strict` — Moco biosynthesis, *Neurospora crassa* — **WOULD EXPORT**

- **5 reactions**, one connected component, 100 %
- stage-3 **ok** · post_normalization **ok** · pre-export **ok** (substantive)
- **F-179: supported. All 5 rows `target_paper`. 0 rows without lineage.**
- Chain is chemically continuous and complete: `GTP → 3',8-cH2 GTP → cPMP → MPT → MPT-AMP → Moco`
- Unsupported content: **none found.** One transport (`cPMP export`) has `transporters=[]` but
  the export *is* described in the paper (*"cPMP is exported into the cytoplasm"*) — unlike the
  enterobactin case
- ⚠ **One flag:** reaction 1's evidence reads *"NIT-7A is **supposed to** catalyze…"* — hedged
  source language. That is **`F-184`** (negation/hypothetical scoping), registered and not
  fixed. It is not blocking and it is not new, but a human reviewer should see it

## `PMC8510960/strict` — MIA biosynthesis, *Catharanthus roseus* — **WOULD EXPORT**

- **5 reactions**, but **3 graph components, main 56 %, proteins_attached 0 %**
- stage-3 **ok** · post_normalization **ok** · pre-export **ok** (substantive)
- **F-179: supported. All 5 rows `target_paper`. 0 rows without lineage.**
- Chemistry is correct and canonical: `tryptophan → tryptamine`, `geraniol → 10-hydroxygeraniol`,
  `secologanin + tryptamine → strictosidine`, `strictosidine → aglycone`,
  `catharanthine + vindoline → anhydrovinblastine`
- Unsupported content: **none found.** Two transports carry `transporters=[]`, but both name a
  real transporter in their evidence (`CrNPF2.9`, `CrTPT2`) that was dropped as degree-zero —
  the transport is described, the transporter entity merely failed to resolve
- ⚠ It would ship a **fragmented** pathway (3 components). Technically valid, thin

## `PMC12376012/strict` — sphingolipid metabolism, *Homo sapiens* — **BLOCKED AT HEAD**

**This is the correction to ORCH-725.** I previously listed it as the strongest recovery. It
is not recoverable, and it is the weakest of the three on provenance.

```
[3] pre_export  SUBSTANTIVE ok=False
      /processes/reactions/4/enzymes/0    reaction_enzyme_must_be_protein_complex
      /processes/reactions/4/modifiers/0  reaction_enzyme_must_be_protein_complex
        Reaction 'ceramide synthase reaction' enzyme 'ceramide synthase'
        must reference a protein_complex entity
[5] provenance  reactions=15   rows with NO lineage=9
```

**Nine of fifteen reactions carry no `provenance_lineage` and classify
`support_class=None`.** F-179 passes the payload only because the other six form a defensible
core — which is F-179 working exactly as designed, not an endorsement of the nine.

**Correction to ORCH-725 § "what would ship":** I reported *"reactions with no provenance
field at all: 0"* for this leg. That measurement used an OR across four fields, so a row with
an `evidence` string but no lineage counted as provenanced. Measured on `provenance_lineage`
alone — the field F-179 reads — the honest number is **9 of 15**. The two legs that would
export are unaffected: both are 0 of 5.

It is blocked by the same `reaction_enzyme_must_be_protein_complex` defect that blocks
`PMC11405693` — a live, correct gate.

---

# 6. How many additional unseen PWMLs would this recover?

## **2.**

| leg | PWML emitted? | substantive? |
|---|---|---|
| `PMC7232280/strict` | **yes** | **yes** — 5 reactions, complete chain, 100 % connected |
| `PMC8510960/strict` | **yes** | **yes** — 5 reactions, correct chemistry, fragmented graph |
| `PMC11172790/strict` | yes | **no** — **0 reactions**; 1 transport, 2 interactions. An empty pathway |
| `PMC12376012/strict` | **no** | blocked by a live pre-export gate |

**3 files. 2 that mean anything.** The pilot's strict-PWML count would move **1 → 3**
substantive (or 1 → 4 counting the empty file, which should not be counted).

This is **one fewer than ORCH-725 estimated.** That report said 3 substantive; the deterministic
replay shows `PMC12376012` is blocked at HEAD by a live gate, so the honest number is 2.

---

# 7. The smallest possible fix — described, NOT implemented

**Do not implement this. The verdict above is `STILL UNSAFE`.** Recorded so the shape is on
file if the precondition is ever cleared.

**File:** `src/t2pw/batch/driver.py` · **function:** `_blocking_reports` (lines 1026–1056)

The function scans every top-level `*_contract_report` and returns any carrying errors. It
never reads the `phase` stamp, so `post_normalization_contract_report` @ `phase: audit_round`
— which `streamlit_app.py:4055-4060` documents as *"not a verdict about what shipped"* — blocks
the run.

**The pattern to follow already exists in the same file**, at `driver.py:2506-2521`, where the
*gate* channel had this identical defect and was fixed by routing it through
`t2pw.pipeline.gate_reports.gate_verdict`, which *"reads the final report and nothing else,
fails closed on every uncertainty, and falls back to the old arithmetic only for artifact sets
stamped as pre-change."*

The minimal change is the same three properties applied to the contract channel:

1. **exclude reports stamped `phase: audit_round`** from the blocking scan — they are
   snapshots taken inside the repair loop, and a later boundary always supersedes them;
2. **fail closed** — if no post-audit boundary report exists, keep today's behaviour rather
   than passing by default;
3. **keep the historical arithmetic** for artifact sets that predate the phase stamp, so
   archived runs keep their recorded verdicts.

Roughly a dozen lines in one function, plus the fail-closed helper. It changes **no** gate, no
threshold and no biology.

**It must land together with the gates that would then block `PMC12452463` on its real
problems** — that is F-147's own condition and this replay does not lift it. Concretely, the
missing gate is one that refuses a **transport with no transporter, no source location and no
destination location**, which is what `Enterobactin secretion` is and what
`PRODUCT_CONTRACT`'s "no unsupported retained processes" implies for transports. F-179 covers
reactions only.

---

# 8. Answers to the questions as asked

**Would the two development legs be blocked by F-179, C-118, or any current final gate if only
the live state were honoured?**
`PMC12180156`: **yes — F-179**, on both archives, corroborated by the committed regression.
`PMC12452463`: **no** on the registration payload — every live gate passes and F-179 says
supported. On the newest payload it is blocked, but by an unrelated `Fe3+` plus-token, not by
anything that objects to its content. **C-118 could not be evaluated**: no archive of either
paper post-dates it, and it only ever *adds* admissions.

**Do the historically forbidden reactions still exist in the current archived payloads?**
`PMC12452463` @ 2026-08-28: **all four objections present.** @ 2026-09-02: three gone, the
`enterobactin secretion` efflux step **still present**.
`PMC12180156`: `ferrochelatase` present, `protoporphyrin IX` present in the payload but **not**
in the PWML it shipped.

**Do they have defensible reaction-level provenance, or would F-179 reject them?**
The enterobactin synthase-complex reaction has `paper_stated` lineage with a quoted span —
**F-179 accepts it.** F-179 reads provenance, not entity class, so a forbidden *identifier* on
a well-provenanced reaction passes it. The `PMC12180156` heme collapse has `inferred`
provenance and **F-179 rejects it.**

**Final export disposition if the stale report were not counted?**
`PMC12452463` @ registration payload → **EXPORTS, with forbidden content**.
`PMC12180156` → **refused by F-179**.
`PMC7232280`, `PMC8510960` → **export, clean**.
`PMC12376012` → refused by the pre-export gate.
`PMC11172790` → exports, empty.
`PMC12071552` (control) → exports, as it really did.

---

# 9. What this changes, and what it does not

**Changed.** F-147's precondition is now **half** cleared, and by a real mechanism: F-179 —
which did not exist when F-147 was registered — closes `PMC12180156` decisively. The remaining
gap is narrower and precisely named: **there is no gate that refuses an unsupported transport.**

**Not changed.** The verdict. One historical bad case exports today with all four gold
objections intact, and the newest version of it is held back only by a `Fe3+` string. F-147's
own instruction stands: *"F-147 may be fixed only after that upstream content is stopped, and
the fix must land together with the gates that would then block these legs on their real
problems."*

**The price is now measured on both sides.** F-147 costs **2 substantive unseen PWMLs** and
buys refusal of **1 remaining contaminated development export**. That is the trade, and it is
the product owner's to make — not this pass's.

---

*Read-only. No production code, gold data, run artifact or `main` was modified. No pipeline leg
was re-run and no LLM draw was taken.*
