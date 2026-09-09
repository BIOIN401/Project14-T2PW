# `C-121` pre-dispatch measurement — read-only, base tree `bcbf62ab`

**2026-09-08. Orchestrator, before any line of production code was written.** The `C-121`
charter § 6 required exactly this: *"Replay these two payloads through the production
functions first, and confirm this card's § 1 before changing a line."* It was done, and it
changed three things the card said.

Nothing was re-run. No LLM, no network, no run directory written. Replay outputs went to a
session scratchpad outside the repository and are **not** committed — they are archived-payload
replays, not production deliverables, and must never enter the PathWhiz import set.

G11 cleanup reports, all `FINAL SURVIVING COUNT : 0` · `cleanup : success`:

| job | report |
|---|---|
| base-tree gate spike | `evidence/g11/C-121/01-spike-base.json` |
| guard census, all trees | `evidence/g11/C-121/02-census-guard.json` |
| guard census, primary checkout | `evidence/g11/C-121/03-census-guard-v2.json` |
| guard ↔ committed-gate cross-tab | `evidence/g11/C-121/04-census-crosstab.json` |
| end-to-end serialization replay | `evidence/g11/C-121/05-replay-e2e.json` |

---

# 1. CORRECTION — the authorizing prompt's invariant does not fix the two legs it cites

The authorization § 5 words the invariant as:

> every surviving element-location row that requires a biological state must reference a
> valid biological state

**Both ORCH-734 blockers have ZERO surviving element-location rows.** Measured on their
committed `final_mapped.json`:

| leg | `compound_locations` | `protein_locations` | `biological_states` | reactions |
|---|---:|---:|---:|---:|
| `PMC11961743` | **0** | **0** | 0 | 10 |
| `PMC4471609` | **0** | **0** | 0 | 4 |

Their committed `pwml_required_field_gate_report.json` carries **one** error each —
`no_biological_states` — and **no** `visible_entity_missing_location_state`. A row-scoped
invariant has nothing to act on and would have repaired neither leg.

`removed_entity_report.json` closes it: `removed_locations: []`. The rows were already empty
before the sweep ran. `ensure_autostates` created `__auto_state__` with
`n_entities_assigned_to_autostate: 0`, and the sweep then removed it — correctly, by its own
policy — along with `cytosol state`.

**The predicate that is actually violated is the charter's § 4 wording**, which is a superset:
*a payload that still carries exportable content must not reach the required-field gate with
zero biological states.*

# 2. CORRECTION — the prompt's fixtures A and B are unreachable through production

Authorization § 8 asks for a graph where an auto-state *"originally existed, later mutation
causes it to disappear, [and the] compound location survives."*

`_prune_biological_states` (`strict_quarantine.py:1922`) removes a state **only when nothing
references it**. A surviving row therefore cannot dangle at a removed state: the removal is
conditioned on the absence of exactly that referent. The shape is excluded by construction,
and a fixture built to exhibit it would be testing a path production cannot take.

**The two reachable shapes**, both measured in the archive:

| | shape | gate error | archived legs |
|---|---|---|---:|
| **A′** | content survives, every state swept as unreferenced | `no_biological_states` | 3 |
| **B′** | a location row created **after** `ensure_autostates` never received an assignment; a real state survives because other rows reference it | `visible_entity_missing_location_state` | 2 |

Shape B′ is the `audit_repair` mechanism `F-192`'s own text describes. It is not a dangling
reference — it is an **absent** one.

# 3. CORRECTION — `F-192`'s archived population is FIVE legs, not three

The cross-tab found two instances nobody had attributed to `F-192`:

| leg | run | states surviving | rows missing a state | gate errors |
|---|---|---|---:|---|
| `PMC11961743` | `runs_smoke/2026-09-08_1528` | 0 | 0 | `no_biological_states` |
| `PMC4471609` | `runs_smoke/2026-09-08_1528` | 0 | 0 | `no_biological_states` |
| `PMC9544450` | `runs_validation/c120/2026-09-08_1240` | 0 | 10 | both codes |
| **`PMC12312563`** | **`runs_verify/2026-08-21_2014`** | `Listeria monocytogenes cytoplasmic state`, `__auto_state__` | **7** | `visible_entity_missing_location_state` ×7 |
| **`PMC13231680`** | **`runs_verify/2026-08-24_1203`** | `__auto_state__` | **5** | `visible_entity_missing_location_state` ×5 |

Both new legs ended `RESULT: FAIL` with *"PWML required-field gate failed"* and **no other
error code**. In both, `__auto_state__` **survived** — so the earlier census, which keyed on
auto-state *removal*, could not see them. `F-192`'s finding text said the defect *"is a
plausible contributor to previously unexplained strict-export failures across the sprint's
archived runs, and that population has not been measured."* It is now measured: **two more.**

**These are also the two oldest instances in the corpus** — 2026-08-21 and 2026-08-24, weeks
before the run that first named `F-192`. The claim that the defect is recent is retracted.

# 4. The guard separates the corpus perfectly — 5 fire, 0 false positives, 0 false negatives

Guard under test, evaluated on the committed post-quarantine payload:

```
fire  <=>  exportable content survives
           AND ( zero biological_states
                 OR some surviving visible element-location row carries no biological_state )
```

Cross-tabulated against each leg's **committed** required-field gate report, over the
**154** production legs (`.pytest_tmp_baseline` and the 130 worktrees excluded):

| | leg failed on an `F-192` code | did not |
|---|---:|---:|
| **guard fires** | **5** | **0** |
| **guard quiet** | **0** | **149** |

99 of the 154 legs predate the gate report and carry none; **the guard is quiet on every one
of them.**

# 5. The regression surface, sized: an UNGUARDED re-run perturbs 40 archived legs

Re-running `ensure_autostates` unconditionally changes the payload on **40** production legs
where the guard does not fire — including **6 of the 8 ORCH-734 legs that produce a PWML
today** (`PMC10055903`, `PMC10269868`, `PMC11016064`, `PMC13184244`, `PMC8211424`,
`PMC9544450`). Each would gain a spurious `__auto_state__` and a `cell` subcellular location,
and every one of their graph hashes would move.

**`"just call ensure_autostates again"` is a reject**, and the authorization § 7 was right to
refuse it in advance. The guard is not a nicety; it is the whole difference between the fix and
a corpus-wide mutation. The prior census's *"31 harmless removals"* understated this: the
correct comparand is **40**, because `ensure_autostates` also backfills species onto every
state and appends `cell`, not only the auto-state row.

# 6. The fix delivers files, not a cleared predicate

Each payload driven through the real export sequence — metadata inject ·
`validate_required_pwml_contract` · `build_pwml_ir` · `blocking_pwml_ir_errors` ·
`validate_pwml_ir` · `DeterministicPwmlBuilder` · `repair_tree` · `validate_generated_tree`:

| leg | base gate | repaired gate | PWML | tree errors | reactions |
|---|---|---|---:|---:|---:|
| `PMC11961743` | FAIL | **PASS** | **101,095 B** | 0 | 10 |
| `PMC4471609` | FAIL | **PASS** | **80,048 B** | 0 | 4 |
| `PMC9544450` (c120) | FAIL | **PASS** | **43,794 B** | 0 | 5 |
| `PMC12312563` | FAIL | **PASS** | **23,098 B** | 0 | 1 |
| `PMC13231680` | FAIL | **PASS** | **17,295 B** | 0 | 1 |

**Reaction counts are unchanged from the archived payload in every case.** No reaction was
admitted, promoted or rescored — the compartment placeholder adds no biology, and this is the
measurement that says so rather than the assertion.

**Controls, byte-identical:**

| control | base | repaired |
|---|---:|---:|
| `PMC9544450` ORCH-734 | 49,464 B | **49,464 B** |
| `PMC10269868` ORCH-734 | 48,458 B | **48,458 B** |

# 7. The seam, located

`streamlit_app.py` is **PROTECTED** (`D-097`) and holds `run_quarantine_boundary`, so the
wiring cannot go there. The sweep lives in `quarantine_and_close`
(`strict_quarantine.py:2324`): the closure loop calls `_prune_biological_states` last in each
round and exits on `converged`. **The restoration point is after that loop**, on the `working`
payload, before `evaluate_core_coverage` and the invariant block read it — the payload
`freeze_canonical_payload` then hashes and freezes.

This is upstream of the freeze, so **merge rule 8 is satisfied by construction**: no exporter
repairs biology after the canonical graph is frozen. It is also not the removal policy —
`_prune_biological_states` keeps removing exactly what it removes today.

`_prune_biological_states`'s own docstring carries the assumption `F-192` falsifies:

> The export requires at least one state (`no_biological_states`), which the coverage check
> backstops — a payload with no surviving process fails there first.

Two legs reached the gate with **10 and 4 surviving reactions** and zero states. The backstop
does not hold, and the docstring should say so.

# 8. What this measurement does NOT settle

* **Which module the restoration function lives in.** The authorization § 6 prefers
  `process_normalizer.py`; the call site must be in `strict_quarantine.py`. Both can be true.
* **Whether reusing `ensure_autostates` wholesale is the right body.** It also appends `cell`
  and backfills species onto every state. On the five firing legs that is harmless — measured
  — but a narrower helper may be preferable. The implementer decides on evidence.
* **Live behaviour.** Every number here comes from archived payloads. § 15 live validation
  still has to run, and cannot be substituted for by any of this.
