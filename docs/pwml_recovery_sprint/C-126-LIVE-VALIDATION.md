# `C-126` post-merge live validation — the seam fired in production and the leg it fired on shipped a PWML

**2026-09-10.** Three strict legs, **one draw each, no favourable reruns**, on the merged and
re-frozen tree `5f6a7aa3` (`D-102`). `runs_validation/2026-09-10_2110`, 1 h 23 m 27 s.
Heavy lock `C-126` acquired and released, **zero timeouts**, `FINAL SURVIVING COUNT : 0`,
`cleanup : success`. Reports `evidence/g11/C-126/60-live-validation.json` and `61-import-check.json`.

**Success was defined before execution**, in `topics_c126_validation.txt`, committed at `8738d841`
**before the run started**. Every scope string was copied verbatim from the cohort that originally
ran it and **mechanically diffed against those files**: identical, nothing edited.

---

# 1. Result

| paper | role | before | after | wall |
|---|---|---|---|---|
| `PMC13123502` *P. polyphylla* | recurrence | refused, `contract`, 4 stale errors | **PASS — `pathway.review_required.pwml`, 36,756 B, IMPORT READY** | 40 m 38 s |
| `PMC9544450` *E. coli* | **control** | PWML on three prior draws | **PASS — 40,689 B, 4 reactions, IMPORT READY** | 23 m 18 s |
| `PMC13089919` *R. microsporus* | recurrence | refused on the stale report | **refused EARLIER, at `post_remap`** — never reached the seam | 19 m 28 s |

**Nothing became `release_ready`. Both PWMLs are `review_required`.**

---

# 2. The seam fired, and the leg's own artifacts prove it

`PMC13123502` is the whole point of this card, and its artifacts carry the exact shape `C-126`
targets:

```
post_normalization_contract_report   ok=False  phase='initial_post_normalization'
                                     effect_on_failure='feed_audit'   errors=4
final_stage3_gate_report             ok=True   phase='final_pre_export'   errors=0
```

The run's own warning channel records the supersession **in production**, verbatim:

> serialized with a superseded intermediate contract report: `post_normalization_contract_report`
> at phase `initial_post_normalization` carried **4** error(s), superseded by
> `post_audit_contract_report`, `post_remap_contract_report`, `final_stage3_gate_report`.
> **The finding is NOT repaired and NOT dismissed: this leg is `review_required` and can never be
> `release_ready`.**

Every property the card claimed is visible in that one line: the **correct phase**, the **correct
error count**, the **three superseding boundaries**, the finding **preserved rather than dropped**,
and the disposition **capped**. The `C-119` cap fired on the new phase, which `REV-126` required be
proven rather than assumed.

The four unresolved UGTs were absorbed exactly as the protein-export policy says:

```
proteins: ['Unknown']          unknown_backed_functional_complexes: 4    verified_real: 0
```

**`PMC13123502` previously died on that stale report. It now produces a 36,756-byte
import-ready PWML.** That is a recurrence leg converted from refusal to deliverable.

## Stated honestly

This is **one fresh draw**, and this pipeline is draw-dependent — `PMC9544450` alone has produced
4, 5 and 6 reactions on four draws. This leg's archived payload carried **5** reactions; this
draw's PWML carries **2**. **The PWML is not claimed as a repeatable yield.** What the artifacts
*do* prove, independently of the draw, is that the stale report was present, was superseded, was
preserved, and no longer decided the disposition. The **deterministic archived proof over 188 legs
remains the primary evidence**; this is confirmation in production, not the proof of record.

---

# 3. The control is genuinely unaffected, and its artifacts show why

`PMC9544450`'s own `post_normalization_contract_report` reads `ok=True`, `phase='audit_round'`,
**0 errors** — so there is nothing for `C-126`'s predicate to supersede and it **cannot fire on
this leg**. The control is not merely asserted to be unaffected; the mechanism is absent from its
artifacts.

It shipped **2 verified real proteins** (`MenD`, `MenH`, `unknown_backed = 0`), 4 reactions,
40,689 B, `IMPORT READY`, `review_required` — its disposition unchanged from its three prior draws.
**If `C-126` had broken anything general, this is where it would show. It did not.**

---

# 4. `PMC13089919` failed EARLIER than the seam, and that is `C-126` working as designed

```
stage          : post_remap
reason         : Generated wrapper component protein must include a UniProt or DrugBank identifier.
failing_codes  : ['generated_wrapper_component_missing_external_identity']
entity_counts  : proteins 4, protein_complexes 5    process_counts: reactions 5
```

**It never reached `C-126`'s predicate.** The leg wrote no `contract_reports.json`, no
`initial_stage3_gate_report.json` and no `final_stage3_gate_report.json`, because
`validate_post_remap` carries `effect_on_failure="abort"` (`stage_contracts.py:241`) and **raises**.

**This is the correct behaviour and `C-126` deliberately preserves it.** Condition 3 excludes only
`feed_audit` reports; every `abort` contract still blocks at every phase. A later boundary that
genuinely refuses is exactly what the card's fail-closed design requires, and what `D-101` § 1
demands.

**The failure mode is pre-existing**, not introduced here: `generated_wrapper_component_missing_external_identity`
appears in archived artifacts from **`runs_verify/2026-08-04_1306`** onward — five weeks before this
card.

Per the pre-registered manifest, a leg that fails before reaching the seam **neither confirms nor
refutes** `C-126`, and **must be reported rather than re-run**. It was not re-run.

## What it does register

This draw extracted 4 proteins and built 5 complexes where the archived draw extracted 2 and built
2, and at least one generated wrapper was built around a component with **no** external identifier
rather than around the `Unknown` sentinel. **Why the Unknown-backed policy did not absorb that
component is an open question about the protein-export policy**, which `D-101` § 3 places outside
this card. **REGISTERED, NOT CHARTERED** — and one draw is not evidence of a repeated defect.

---

# 5. Against the pre-registered criteria

| criterion, frozen before the run | outcome |
|---|---|
| recurrence leg must not fail on a stale `initial_post_normalization` report with a clean final gate | **MET** — neither leg did; `PMC13123502` superseded it and shipped |
| control must still produce a PWML and not move disposition | **MET** — 40,689 B, `review_required`, unchanged |
| would refute: control loses its PWML | did not happen |
| would refute: any leg reaches `release_ready` that did not before | did not happen — **zero** |
| would NOT refute: a leg refused later by another gate | `PMC13089919`, at an `abort` boundary, as designed |

**`C-126` is confirmed in production.** Zero timeouts. Zero surviving processes. No leg re-run, no
scope string edited after seeing a result, no cache or run directory committed, `main` untouched.
