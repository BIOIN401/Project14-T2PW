# C-120 — POST-MERGE PWML VALIDATION. Result.

**Executed 2026-09-08 12:40:57 to 13:43:00, 1h02m.** Three strict legs, one run each, no
retries. Frozen configuration: [`C-120-VALIDATION-FREEZE.md`](C-120-VALIDATION-FREEZE.md),
committed at `e0fe1475` **before** execution. No production code was modified. `main` untouched.
Production frozen at `045447c8` throughout.

> ## A NEW dataset with its own identity.
> `runs_validation/c120/2026-09-08_1240`. **Not** the ORCH-724 pilot, **not** the ORCH-730
> validation, **not** the ORCH-732 smoke. Its numbers may never be merged into any of their
> denominators. All three remain read-only and none is rescored by anything here.

---

# 1. Headline

## 1 PWML from 3 legs. Mechanism B is proved in production. Mechanism A was never tested. The control regressed — for a reason that is not `C-120`.

| # | paper | role | outcome | reactions | PWML | bytes | time |
|---|---|---|---|---|---:|---|---|
| 1 | `PMC11487621` | mechanism A | **FAIL — Stage 1** | — | NO | — | 7m22s |
| 2 | `PMC10031235` | mechanism B | **PASS** | 3 | **YES** `review_required` | 48,401 | 30m37s |
| 3 | `PMC9544450` | **CONTROL** | **FAIL — contract** | 5 | NO | — | 23m28s |

**Only leg 2 is evidence about `C-120` at all.** Leg 1 died before reaching any code this card
touches. Leg 3 died in a path this card does not touch, for a reason traced to root cause below.

---

# 2. Leg 2 — `PMC10031235`. Mechanism B, confirmed in production.

```
[0] 'PSAT'    uniprot=Q9Y617   chosen_rule=pathbank_protein_id   name_gate=gene_symbol_family_identity
[1] 'PHGDH'   uniprot=O43175   chosen_rule=pathbank_protein_id   name_gate=exact_symbol_identity
```

**`PSAT` ships as `Q9Y617`, admitted by the rung `C-120` added**, and the entity is still named
`PSAT`. In ORCH-732 the same entity became the literal string `Unknown` and blocked the export
with `blocking_issues = 1`. Here `blocking_issues = 0`.

`PHGDH` still resolves through the **pre-existing** `exact_symbol_identity` rescue — unchanged,
which is the control inside the leg.

**The file.** `pathway.review_required.pwml`, 48,401 bytes, parses as well-formed XML with a
`super-pathway-visualization` root. It contains `Q9Y617` once, `O43175` once, `PSAT` twice, and
**`Unknown` zero times**.

**Disposition ceiling held.** `review_required`, **not** `release_ready`. Nothing was promoted
because an identity was recovered — the § 8 requirement.

**Species:** `Homo sapiens`, taxonomy `9606`, `Eukaryote`, `source: ncbi`. The alias pass was not
involved and correctly did nothing: there is one species and no abbreviation.

## What this leg does NOT establish

**It is not a same-payload A/B.** This draw produced **3** canonical reactions; the archived
ORCH-732 leg had **6**. Temperature 0 is not determinism. What is directly attributable to
`C-120` is the identity — the `name_gate` reason names the new rule, on the exact entity that
previously degraded. **No yield claim follows from one leg**, and the pathway here is smaller than
the one that was blocked.

---

# 3. Leg 1 — `PMC11487621`. A NULL RESULT. Mechanism A is untested live.

```
Stage 1 extraction · attempts 1,2,3 · status=empty · content_chars=0 · finish_reason=length
terminal_reason: empty_after_retries -> identical_empty_response
last_completed_stage: stage0_preprocess
```

Three consecutive **empty** completions on a 76,081-character paper, identical response hash
(`e3b0c44298fc1c14` — the sha256 of the empty string), after which the extraction ladder correctly
refused to re-issue the same prompt to the same model
(`skip_cause: identical_prompt_same_model`).

**The leg never reached any `C-120` code.** Stage 0 preprocessing succeeded (1,596 chars); Stage 1
produced no payload; `map_ids.py` and `pwml/ir.py` are not reachable before extraction; the
extraction prompt lives in the byte-identical `streamlit_app.py`. **`C-120` cannot have caused
this, and this leg is neither a `C-120` success nor a `C-120` failure.**

## This is a DISTINCT sub-class from `PMC7615680` — do not merge the two

| | `PMC7615680` (ORCH-732) | `PMC11487621` (here) |
|---|---|---|
| `finish_reason` | **`stop`** | **`length`** |
| `content_chars` | 2 | **0** |
| reading | a complete, well-terminated, essentially empty response | the completion budget consumed while emitting nothing |

Registered separately as **`F-193`**.

## Why it was NOT re-run

The freeze committed before execution states that a leg is re-run **only for an objectively
invalid infrastructure execution**, and that a failure is a result. This is not an infrastructure
failure: the wrapper, the measured tree and the pipeline all behaved correctly and the provider
returned nothing.

There is a real tension — the leg never exercised the code under test, so an argument exists that
re-running it is not "seeking a favourable draw". **It was not acted on.** Relaxing a no-retry
rule written hours earlier, in the one direction that would help this card's own result, is
precisely what the rule exists to prevent. A re-run remains available as a **separately labelled
product-owner decision**, the shape `D-097` used for the earlier JSON-failure legs. It would be a
new draw and its output would not belong to this dataset.

---

# 4. Leg 3 — the CONTROL regressed. Root cause found, and it is NOT `C-120`.

`PMC9544450` produced a PWML in ORCH-732 (39,686 B). Here it failed the required-field gate with
**11 errors**: `no_biological_states` plus ten `visible_entity_missing_location_state`.

**`blocking_issues = 0`, `gate_errors = 0`.** The identity layer is clean: species
`Escherichia coli` → `562` / `Prokaryote`, and both proteins resolve — `MenD` → `P17109`,
`MenH` → `P37355`, **both through the pre-existing `exact_symbol_identity` rescue**. The
`gene_symbol_family_identity` rung `C-120` added **did not fire on this leg at all**.

## The mechanism, traced through the artifacts

```
removed_biological_states: [{"name": "__auto_state__",
                             "reason": "state_unreferenced_after_quarantine", "iteration": 1}]
```

`ensure_autostates` (`process_normalizer.py:3122`) runs **unconditionally** and creates
`__auto_state__`, assigning it to every `element_location` row that lacks a state. But the
location rows in this payload were created by a **later** `audit_repair` pass
(`provenance_lineage: [{"stage": "audit_repair", "origin": "audit_modified"}]`), and
`ensure_autostates` is not re-run afterwards — so they carry `biological_state: None`. With no
referents left, the quarantine sweep judged `__auto_state__` unreferenced and removed it. Zero
states remained, and all ten locations were orphaned.

## The differentiator is the Stage-1 draw, and it is measured

| leg | Stage-1 `biological_states` | location-row lineage | final states | outcome |
|---|---:|---|---:|---|
| ORCH-732 `PMC9544450` | **1** — `E. coli cytoplasm` | `audit_repair` | 1 | **PWML** |
| C-120 `PMC9544450` | **0** | `audit_repair` | **0** | **FAIL** |
| C-120 `PMC10031235` | 0 | *(none — not audit-repaired)* | 2 | **PWML** |

When Stage 1 emits a real state the `audit_repair` rows inherit it and the chain holds. When it
emits none, `__auto_state__` is the only state, its referents were never written, and it is swept.

## Why `C-120` is excluded, on evidence rather than assertion

1. **The merge diff contains zero occurrences of `biological`, `_auto_state` or
   `element_location`.** Mechanically checked over `git diff 760c6d72 045447c8 -- src/`.
2. **`process_normalizer.py` is untouched** by the merge — `git diff --stat` over
   `src/t2pw/pipeline/` is empty.
3. **A within-run control settles it.** The *same code* in the *same run* produced 2 biological
   states and a valid PWML for `PMC10031235`. State handling is not globally broken.
4. **The new rung never fired on this leg.** Both proteins were admitted by the pre-existing
   `exact_symbol_identity` rescue.

**What cannot be excluded**, and is stated rather than glossed: a different Stage-1 draw is itself
downstream of nothing `C-120` controls, but `C-120` does change identity resolution, and an
indirect influence on what the auditor rewrites is not *provable* to be absent from one leg. The
four points above make it implausible; they do not make it impossible. Registered as **`F-192`**,
a pre-existing latent defect, and the honest reading is that **the control failure is a Stage-1
draw exposing it**, not a `C-120` regression.

---

# 5. Against the card's § 8 safety requirements

| requirement | result |
|---|---|
| Wrong-organism protein still rejected | not exercised live; proved at function level and by `REV-120` in two rounds |
| Unknown still reachable | yes — leg 3's payload retains the sentinel machinery untouched |
| `F-179` anti-invention | `reaction_support.py` untouched; no fabricated chemistry in leg 2's 3 reactions |
| No new `release_ready` | **held.** The one PWML is `review_required` |
| Identity recovery enables `review_required` PWML | **demonstrated once**, leg 2 |

---

# 6. What this run establishes, and what it does not

**Establishes.** Mechanism B works end to end in production: a protein the previous code replaced
with the literal `Unknown` now ships its correct reviewed accession, through the specific rung
this card added, into a well-formed PWML file that a human can import.

**Does not establish.**

* **Mechanism A in production.** `PMC11487621` never ran. The § 9 deterministic replay remains the
  only evidence for it, and that proves the gate **predicates**, not a file.
* **Any yield rate.** Three legs is not a rate; one of them never ran.
* **That the control is healthy.** It is not — but for `F-192`, not for `C-120`.
* **That `F-185` is closed.** ORCH-725's Type-3 population is untouched. `C-120` fixed two members
  of the class, not the class.

---

# 7. Process

Bounded wrapper on every command, heavy lock held as `C-120`, one heavy job at a time, no detached
processes, no cache commits. Cohort staged and inspected before execution
(`evidence/g11/C-120/40-stage-validation.json`); the three staged full-text lengths — 76,081 /
74,000 / 18,043 — are **byte-identical to ORCH-732's**, so the control is genuinely comparable.

*Incidental correction to the record:* ORCH-732 § 2 states its six papers had "full text 74k–80k
characters each". Two did not — `PMC9544450` was 18,043 and `PMC4725005` was 35,228. Not
load-bearing for anything, corrected here because it is checkable.

```
duration                : 3722.9 s
exit reason             : nonzero          (exit 1 -- 2 of 3 legs did not pass; the run itself is valid)
FINAL SURVIVING COUNT   : 0
cleanup                 : success
heavy lock              : holder=C-120  acquired=True  released=True
pre-existing (reported, NEVER killed): 7
```

Evidence: `evidence/g11/C-120/40-stage-validation.json` · `41-c120-validation.json`.
