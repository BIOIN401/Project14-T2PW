# Why `C-119` did not supersede the stale reports in the four recurrence legs

**2026-09-10. Read-only diagnosis. No production file touched.** Answers the product owner's
blocking question before any code is written.

---

# 0. The answer, in three sentences

`C-119` excludes exactly one phase, `audit_round`. In all four recurrence legs the killing
report is stamped **`initial_post_normalization`**, a *different* non-authoritative phase, so
`C-119`'s exclusion never matches. It never matches because the audit round **changed nothing**
in these legs, and the re-stamp to `audit_round` is gated on the payload having changed.

**`C-119` can only ever rescue a leg whose audit round edited the payload.** A leg whose audit
correctly decides it has nothing it may legitimately repair is exactly the leg `C-119` cannot
help. That is the recurrence.

---

# 1. The exact report that kills each leg

| | |
|---|---|
| artifact key | **`post_normalization_contract_report`** |
| `phase` | **`initial_post_normalization`** |
| `ok` | `false` |
| `effect_on_failure` | **`feed_audit`** |
| `contract_type` | `semantic` |
| errors | 3 / 2 / 4 / 8 (below) |
| what it validated | the **pre-remap** payload, `merged_payload.json` |

Measured on all four, from each leg's own `contract_reports.json`:

| leg | errors | every other contract report | `final_stage3_gate_report` |
|---|---:|---|---|
| `PMC13184244` | 3 | `post_extraction` · `stage2` · `post_audit` · `post_remap` all `ok=true`, 0 errors | `phase=final_pre_export` **`ok=true`, 0 errors** |
| `PMC13089919` | 2 | same | same |
| `PMC13123502` | 4 | same | same |
| `PMC4471609` | 8 | same | same |

**It is the only failing report in any of the four artifact sets.**

## The blocking path, line by line

```
driver.py:2790   blocking_reports = _blocking_reports(artifacts)
                   -> post_normalization_contract_report is NOT excluded
driver.py:2791   codes, code_lines, error_count = _collect_issue_codes(blocking_reports)
                   -> error_count = 3 / 2 / 4 / 8
driver.py:2850   verdict = gate_verdict(artifacts)        -> verdict.failed = False   (PASSES)
driver.py:2901   blocking_gate = gate_failed and not research_mode   -> False
driver.py:2911   if blocking_gate or error_count:         -> TRUE, via error_count alone
driver.py:2912   _finalize_gate_failure(...)              -> status=fail, kind=contract
```

**The gate channel passed. The contract channel failed the leg on its own.** `error_count` is
the sole blocking input, and `_blocking_reports` is the only scan that feeds it — `error_count`
appears at `driver.py:2791`, `:2797` (a count) and `:2911` (the decision) and nowhere else, so
there is **no second blocking-report scan** in the contract channel.

## A diagnostic defect worth recording separately

The failure message reads *"post-pipeline validation failed: N blocking issue(s) at
`final_pre_export_stage3_gates`"*. That boundary name comes from `verdict.stage`
(`driver.py:2472`) — the verdict that **passed**. The issues come from a different, stale
boundary. **The message names the boundary that succeeded and attributes to it the findings of
the boundary that is stale.** This is what made an earlier reading of these legs look like a
final-gate failure. It is a reporting defect, not the blocker.

---

# 2. Why `C-119` does not reach it

`C-119`'s seam is `_superseded_contract_reports` (`driver.py:1193`). Three conditions must hold
for a report to be excluded, and the **third** is where these legs fall out:

1. `_artifact_set_is_phase_stamped(artifacts)` — **holds.** Five contract reports carry `phase`.
2. `_superseding_boundaries(artifacts)` is non-empty — **holds, three times over.** All of
   `post_audit_contract_report`, `post_remap_contract_report` and a `final_pre_export`
   `final_stage3_gate_report` are present and clean.
3. ```python
   if _report_phase(report) != PHASE_AUDIT_ROUND:
       continue
   ```
   **Fails.** `"initial_post_normalization" != "audit_round"`, so the report is skipped and
   stays in the blocking set.

**`C-119` built the whole supersession machine and then gated it on one phase name out of the
two that its own module documents as non-authoritative.**

## Why the phase is never `audit_round` in these legs

The audit loop's re-stamp in `streamlit_app.py` (committed `:4019`) is guarded:

```python
if settled_payload_changed:
    ...
    post_normalization_contract_report = _contract(
        validate_post_normalization, settled_payload, fresh_gate_report,
        phase=PHASE_AUDIT_ROUND,
    )
```

Every one of these legs recorded `settled_payload_changed: false`, because the audit declined
to act — correctly, in its own words:

> `PMC13089919` — *"The two proteins ChoC and Cho2 are novel and lack database identifiers; no
> patch can add identifiers without inventing data."*

> `PMC13123502` — 13 patches proposed, **13 rejected**, 0 accepted.

So the re-stamp never fires, and the report keeps the `initial_post_normalization` stamp it got
at the initial post-normalization contract call (committed `streamlit_app.py:3603`).

**The better the audit behaves, the more certainly `C-119` fails to protect the leg.** An audit
that invents an identifier gets the leg rescued; an audit that refuses to invent one does not.

## What actually made the report stale: the remap, not the audit

`C-119` modelled the audit loop as the only thing that can invalidate a post-normalization
report. In these legs the staleness is produced **after** the audit, by the Stage-6 remap that
applies the Unknown-backed-complex policy:

| leg | `merged_payload.json` proteins (validated) | `final_mapped.json` proteins (shipped) | complexes |
|---|---|---|---:|
| `PMC13184244` | `UGT1, A622, BBLa, β-GD1, MATE1` | `A622, BBLa, Unknown` | 4 |
| `PMC13089919` | `ChoC, Cho2` | `Unknown` | 2 |
| `PMC13123502` | `UGT91BP2, UGT703R1, UGT703R2, UGT703R3` | `Unknown` | 4 |
| `PMC4471609` | `DmaW, EasF, EasE, EasC, EasD, EasA, EasG, EasH` | `Unknown` | 8 |

Each leg's `protein_export_policy.summary` confirms it:
`unknown_backed_functional_complexes` = 2 / 2 / 4 / 8.

**The thing that made the report stale is the documented protein-export policy doing its job.**
The report's pointers — `/entities/proteins/0..3` — address a list the shipped payload cannot
host, because the rows they named stopped being bare proteins.

## What `C-119`'s evidence base could not have shown

All **nine** `C-119` fixtures, measured:

| `post_normalization_contract_report` phase | fixtures |
|---|---:|
| `audit_round` | **9** |
| `initial_post_normalization` | **0** |

`C-119` was built on nine artifact sets that every one of them had re-stamped. The
`initial_post_normalization` case is not a regression of `C-119`; it is a case its evidence base
contained no instance of. `tests/fixtures/c119/` should gain one.

---

# 3. The authoritative final contract verdict ALREADY EXISTS

This is the most important finding, and it changes the shape of the fix from *build an
abstraction* to *stop pre-empting the one we have*.

`run_pwml_export` (committed `streamlit_app.py:4715`) already does exactly what the product owner
described:

```python
stage3_gate_report, stage3_contract_report = _validate_stage8_export_payload(payload)
if not bool(stage3_contract_report.get("ok", False)):
    return {"ok": False, "error": "PWML export stopped by validation-only pre-export Stage 3
            revalidation; Stage 8 did not repair the payload.", ...}
```

`_validate_stage8_export_payload` (committed `:4571`) runs, **on the exact payload about to serialize**:

* `run_strict_post_normalization_gates(payload, enforce_all_proteins_connected=True)` — the
  same suite, including the same `Protein '<x>' is missing a UniProt or DrugBank identifier`
  check at `process_normalizer.py:4824`; and
* `validate_post_normalization(payload, stage3_gate_report)` — the same contract function,
  including its own `_validate_canonical_actor_rows`.

So the check the stale report is blocking on **re-runs at the authoritative boundary and stops
the export there**, and it does so fail-closed (`ok` must be explicitly true). A third
independent enforcement of the same invariant sits in the IR build at `pwml/ir.py:2619`.

**There are not two notions of "final".** There is one authoritative boundary, it is
enforced, and the batch driver vetoes the leg before it is ever reached — using a copy of the
same findings computed on a payload that no longer exists.

The driver's own comment at `:2899` already names the overlap — *"the same ALAS2 registry error
arrives as a gate error AND as the contract error derived from it"* — and dedupes it **for
counting** while letting *"both channels still block independently."* The gate channel was given
an authority model by `gate_verdict` (final report only, fail closed). The contract channel was
never given the same one.

---

# 4. The smallest correction

**Subtractive, inside `C-119`'s existing seam, one predicate.** It removes a redundant early
veto; it does not add a new permission.

> A contract report that declares `effect_on_failure: "feed_audit"` and carries a
> **non-authoritative phase** is a diagnostic, not a verdict. Its errors are recorded, never
> blocking. Authority over those same checks belongs to the final boundary, which already
> enforces them.

Two independent conditions, both asserted by production itself:

* **`phase`** — `gate_reports.py` documents `initial_post_normalization` as *"never a verdict
  about what shipped"*, `audit_round` as *"not authoritative either"*, and `final_pre_export` as
  *"THE authoritative phase."* The excluded set becomes **the closed allow-list of those two
  non-authoritative phases**, not a growing list of report names, which is what makes this a
  unification and not a second ad hoc path.

  > **CORRECTION, 2026-09-10.** An earlier revision of this line proposed expressing the set as
  > *"not `final_pre_export`"*. **That was wrong and `prompts/C-126.md` § 3 states the correct
  > rule.** A negated test silently enrols every future phase name, every typo and every
  > malformed string into the excluded set — the one direction this seam must never fail in, and
  > the opposite of `_report_phase`'s own doctrine that an unidentified phase reads as *live*.
  > The `C-126` implementer followed the card over this document and flagged the contradiction;
  > that was the right call. **The card outranks this document wherever they disagree.**
* **`effect_on_failure`** — `"feed_audit"` is set by exactly one contract,
  `validate_post_normalization` (`stage_contracts.py:219`); every other contract uses `"abort"`.
  **The batch driver never reads this field at all** (`grep` finds it in `streamlit_app.py`,
  `export_mode.py`, `stage_contracts.py` — never in `batch/driver.py`). Production states the
  report's role and the consumer ignores it.

## Why this cannot weaken anything — each claim is checkable

| risk | why it does not materialise |
|---|---|
| structural garbage slips through | `validate_post_normalization` escalates to `effect_on_failure="abort"` and **raises** via `_abort` when `_validate_payload_container` fails (`stage_contracts.py:221-224`). A `feed_audit` report in an artifact set cannot be hiding structural garbage. |
| a genuinely unresolved protein exports | the same check re-runs at `_validate_stage8_export_payload` and stops the export; and at `ir.py:2619`; and `gate_verdict` blocks on a dirty `final_pre_export` report. Three independent points, all **after** the remap. |
| `F-179` bypassed | `F-179` is a reaction-support invariant on a different channel. `no_defensible_reaction_support` legs (`PMC13405594`, `PMC13438895`, `PMC13474940`) are unaffected and must be shown unaffected. |
| Stage 3 bypassed | Stage 3 *is* the final gate. This change removes authority from a **pre**-Stage-3 snapshot only. |
| missing final boundary becomes success | `_superseding_boundaries` already returns `[]` and the report still blocks. Unchanged. |
| legacy artifacts promoted | `_artifact_set_is_phase_stamped` already keeps unstamped sets on the old arithmetic. Unchanged. |
| `release_ready` appears | `C-119`'s sixth cap in `classify_release_status` already pins such a leg to `review_required`. It must be shown to fire on the new phase too — **this is the one place the correction has to extend `C-119` rather than inherit it.** |
| `ok=true` next to a non-empty `errors` list | `_report_errors` already reads the list, not the flag. |

## Required proof, under G9

The base-failing proof must fail on **values**, never on a missing symbol. For at least one real
recurrence artifact set, at the base SHA:

* `gate_verdict(artifacts).failed` is `False`;
* `error_count` is non-zero and comes only from `post_normalization_contract_report`;
* the leg is refused.

At the tip: same payload, same biology, same final gates, the stale report **still present in
`contract_reports.json`** and still named in the manifest, but no longer deciding disposition,
and the leg bounded to `review_required`.

## What is NOT claimed

> **MEASURED REACH, added after `REV-126`, 2026-09-10.** This document diagnosed the defect on
> **four** papers. An independent base-vs-tip census over all **188** archived legs found the fix
> moves **nine** strict legs, and **zero** of 75 research legs. The five beyond the four named here
> are `PMC10031235` (`runs_smoke/2026-09-07_2323`), `PMC12452463` (`runs_verify/2026-08-24_1203`
> and `2026-09-01_1612`), `PMC12444477` (`2026-08-25_1216`) and `PMC12096016` (`2026-08-27_1341`).
> Three of them carry stale findings of a **broader class** than protein-identifier pointers, such
> as a registry `unknown entity` error. **None of the nine becomes `release_ready`** — all nine were
> already `review_required` at base and stay there. **The reach is nine, not four.**

**Not that any of the four produces a PWML.** All four have substantive final payloads —
reactions 4 / 5 / 5 / 6, biological states present, compounds 13 / 8 / 5 / 10 — so they are
eligible to proceed. Eligibility is not export. `PMC13184244` additionally carries one transport
and is therefore an `F-195` candidate. **No PWML count should be quoted for this card.**

---

# 5. Recommended scope

One card, `driver.py` only, inside `C-119`'s seam, plus tests and one new fixture. It does not
touch `RAG`, `F-179`, `C-120`, `C-121`, `C-122`, `C-124`, `C-125`, protein thresholds, Stage-1,
`F-195`, taxonomy, the Unknown-sentinel policy, or `main`.

The reporting defect in § 1 — the message naming the passing boundary — should be fixed in the
same card only if it stays inside `_finalize_gate_failure`; otherwise it is a separate finding.
