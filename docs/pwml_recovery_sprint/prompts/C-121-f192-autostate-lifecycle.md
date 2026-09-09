# C-121 — F-192: restore the auto-state after the quarantine sweep orphans it

**STATUS: AUTHORIZED 2026-09-08 under `D-099`. READ AMENDMENT 1 AT THE FOOT OF THIS FILE FIRST**
-- it corrects three of the claims below, including the invariant.

*Original status block, kept verbatim because the card was written to be dispatched under it:*

> **STATUS: CHARTERED, NOT AUTHORIZED.** This card exists so the work is specified. **It must not
> be dispatched until the product owner unfreezes production for this seam**, the way `D-094` did
> for F-179 and `D-098` did for C-120. Production is frozen at `045447c8` under `D-090`/`D-098`,
> whose § 9 says *"no further production change is authorized, and no second optimization card
> automatically follows."*

Evidence: [`ORCH-734-FINAL-RELIABILITY-RESULT.md`](../ORCH-734-FINAL-RELIABILITY-RESULT.md) § 5.

---

## 1. The defect

A leg whose biology, identity and reaction support are all sound is refused at the **last** gate
with `no_biological_states`, and the whole pathway is lost.

`ensure_autostates` (`process_normalizer.py:3122`) is called **exactly once**, at
`process_normalizer.py:5432`. It creates `__auto_state__` and assigns it **only** to
`element_locations` rows that currently carry no `biological_state` (`if state: continue`,
`:3170`).

The strict quarantine sweep (`strict_quarantine.py`, stage `pre_export_strict_quarantine`) runs
**later** and removes biological states that became unreferenced once entities were quarantined,
with reason `state_unreferenced_after_quarantine`. **Nothing re-runs `ensure_autostates`
afterwards.** The PWML required-field gate then fails on `no_biological_states`.

## 2. Population — measured, not estimated

**Three independent instances**, on three unrelated organisms:

| leg | run | reactions destroyed |
|---|---|---:|
| `PMC11961743` *S. divaricata* | `runs_smoke/2026-09-08_1528` | **10** |
| `PMC4471609` *A. japonicus* | `runs_smoke/2026-09-08_1528` | **4** |
| `PMC9544450` *E. coli* | `runs_validation/c120/2026-09-08_1240` | 5 |

Both ORCH-734 legs carry the byte-identical signature: Stage-1 `biological_states` **0** ·
`initial_stage3 n_autostate_created` **1** · `n_entities_assigned_to_autostate` **0** ·
`__auto_state__` removed `state_unreferenced_after_quarantine` · `final_stage3
n_autostate_created` **0** · F-179 `supported` · `gate_errors` **0**.

> ### Do NOT quote "1 blocked leg in 144" as this defect's rate.
> The **mechanism is deterministic; the TRIGGER is draw-dependent.** It bites only when the sweep
> orphans *every* remaining state, which needs Stage 1 to have emitted none of its own.
> `PMC9544450` **failed** on it in the C-120 validation and **passed** in ORCH-734 — same paper,
> same code, different draw. The old census measured how often the trigger fired in one archive,
> not how often the defect is reachable.

## 3. Ownership boundary

**Owned:** the auto-state lifecycle seam only — the post-sweep restoration point, and
`ensure_autostates` itself only if the fix genuinely belongs there.

**Forbidden, and each for a stated reason:**

* `pipeline/reaction_support.py` — **F-179.** Untouchable. This card must not alter which
  reactions are supported, admitted or scored.
* `rag/admission.py`, `stage_contracts.py`, extraction prompts, Stage-1 retry policy, token
  budgets — unrelated to the seam.
* `batch/driver.py`, `release_status.py` — C-119's seams.
* `pwml/ir.py`, `mapping/map_ids.py` — C-120's seams.
* `app/streamlit_app.py` — **PROTECTED** (`D-097`). Read, never modify.
* **The quarantine's removal policy itself.** Do **not** make the sweep stop removing unreferenced
  states. The sweep is correct: those states genuinely were unreferenced. The defect is that
  nothing re-establishes the placeholder afterwards.

## 4. The rule to implement

> **A payload that still carries exportable content must not reach the required-field gate with
> zero biological states.**

The compartment placeholder is **presentation scaffolding, not biology.** It admits no reaction,
changes no reaction's support, and adds no entity to the pathway. That is precisely why this is
narrow — and the implementer must **prove** it, not assert it.

Suggested seam, to be confirmed against the real blast radius rather than adopted on faith:
re-establish the auto-state **after** the quarantine sweep, when surviving `element_locations` or
transports still require one.

## 5. Gates — beyond the standing merge rules

1. **G9 behavioural proof.** A test that **fails on the base SHA and passes at the tip**, driving
   the real seam. Symbol absence is not proof.
2. **Both archived payloads replay to a PWML.** `PMC11961743` and `PMC4471609` must serialize when
   their archived `final_mapped.json` is driven through the fixed path. **Replay the archived
   payload — do not re-run the papers**, which would draw fresh and prove nothing.
3. **F-179 is untouched and unmoved.** `reaction_support.py` byte-identical, and
   `evaluate_reaction_support` returns the **same verdict** on every archived leg in
   `runs_smoke/2026-09-08_1528` before and after.
4. **Reaction counts do not change.** For every leg that already produces a PWML, the canonical
   and core-accepted reaction counts must be **identical**. A fix that increases them is admitting
   biology and is a **reject**.
5. **The five currently-passing PWMLs are byte-identical** after the change, or every difference
   is explained. This fix must be **inert** on legs that work today.
6. Integration smoke (465) and the gold-readers set both pass; no pin moves.
7. Full G11 lifecycle: bounded wrapper, `--basetemp`, zero survivors.

## 6. The trap this card must not fall into

**Do not "fix" this by weakening the required-field gate.** Making `no_biological_states`
non-blocking would raise the PWML count immediately and would be a **direct violation of merge
rule 6** — weakening a gate to increase PWML production. The gate is right: PathWhiz needs an
explicit state. The bug is upstream, in the lifecycle that fails to provide one.

**And measure before writing.** C-120's most expensive lesson was that a prior report's *framing*
can be wrong even when its *facts* are right: ORCH-732 read `PSAT` → `Unknown` as a fallback
overwriting a good identity, and the replay showed rung 4 had refused it. **Replay these two
payloads through the production functions first, and confirm this card's § 1 before changing a
line.**

---

# AMENDMENT 1 — AUTHORIZED under `D-099`, and THREE OF THIS CARD'S CLAIMS ARE CORRECTED · 2026-09-08

**Status is now AUTHORIZED.** The seam is narrowly unfrozen. Read
[`../C-121-PREDISPATCH-MEASUREMENT.md`](../C-121-PREDISPATCH-MEASUREMENT.md) **before § 1 above** —
§ 6 of this card demanded the payloads be replayed through production functions before a line was
written, that was done, and it contradicted the card in three places.

### The invariant is a DISJUNCTION, not a row-scoped rule

```
exportable content survives
AND ( zero biological_states
      OR some surviving visible element-location row carries no biological_state )
```

`PMC11961743` and `PMC4471609` have **zero** element-location rows and one error each,
`no_biological_states`. A row-scoped invariant repairs neither. Clause 2 is the `audit_repair`
shape — a row created after `ensure_autostates`, which therefore never got an assignment. It is an
**absent** reference, not a dangling one, and a dangling one is impossible:
`_prune_biological_states` removes a state only when nothing references it.

### § 2's population is FIVE legs, not three

`PMC12312563` (`runs_verify/2026-08-21_2014`, 7 unassigned rows) and `PMC13231680`
(`runs_verify/2026-08-24_1203`, 5 unassigned rows) both failed on
`visible_entity_missing_location_state` alone, with `__auto_state__` **surviving** — which is why a
census keyed on auto-state *removal* missed them. **They are the two oldest instances in the
corpus.** § 2's "byte-identical signature" describes clause 1 only.

### § 5 gate 2 is achievable, and gate 5's danger is now sized

All five payloads serialize with zero tree errors and **unchanged reaction counts** — 101,095 ·
80,048 · 43,794 · 23,098 · 17,295 bytes. Two passing controls come out byte-identical.

**An unguarded `ensure_autostates` re-run changes 40 archived legs, 6 of them ORCH-734 successes.**
The suggested seam in § 4 is correct; taking it without the guard is a **reject**.

### The seam is located

`quarantine_and_close` (`strict_quarantine.py:2324`), on `working`, **after** the closure loop exits
on `converged` and before `evaluate_core_coverage` and the invariant block read it.
`run_quarantine_boundary` lives in `streamlit_app.py`, which is **PROTECTED** — the wiring cannot go
there. This point is upstream of `freeze_canonical_payload`, so merge rule 8 holds by construction,
and `_prune_biological_states` keeps removing exactly what it removes today.

`_prune_biological_states`'s docstring asserts the coverage check backstops
`no_biological_states`. Two legs reached the gate with 10 and 4 surviving reactions and zero
states. **The docstring is wrong and should be corrected in the same diff.**

### Fixtures

§ 8-A and § 8-B of the authorizing prompt are unreachable and **not owed**. Owed instead: one G9
proof per reachable shape above, failing on `bcbf62ab` **on values**; and **fixture C unchanged** —
a genuinely unreferenced auto-state must still disappear. Fixture C is the most important of the
three.
