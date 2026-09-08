# ORCH-730 — post-`C-119` PWML validation. RESULT.

**`F-187` option 2, executed 2026-09-07 19:29–20:59.** Frozen configuration recorded and
committed **before** execution as `a91e4a89`; see `ORCH-730-VALIDATION-FREEZE.md`.

> ## The `ORCH-724` unseen pilot stands unchanged.
> `runs_verify/2026-09-06_1425` was not read for input, not modified, and is **not** superseded
> or rescored. These three legs live in `runs_validation/2026-09-07_1929`, a separate tree with
> its own identity. **They may never be merged into the pilot's denominators.**

**Run:** 3 strict legs, 1h 29m 51s wall, `FINAL SURVIVING COUNT : 0`, `cleanup : success`,
wrapper exit 0. Stage-1 / Stage-2 `max_tokens` = **16000 / 16000**, chosen and stated in
advance. **No leg was re-run.**

---

# 1. Headline

| | |
|---|---|
| legs executed | 3 of 3 |
| **PWML written** | **1** — `PMC7232280/strict` |
| `release_ready` | **0** |
| legs failed | 2 |
| **failures attributable to `C-119`** | **0** |

> ## `C-119` works. It is proven end to end, on a real run, by the one leg that reached the seam.
> ## The other two legs failed UPSTREAM of anything `C-119` touches, for two different draw-dependent reasons.

**And the control failed.** That is the most important sentence in this report and it governs
how everything else may be read — see § 5.

---

# 2. `PMC7232280/strict` — PASS. A verified `C-119` recovery.

| field | value |
|---|---|
| pipeline completion | **PASS (with warnings)**, 2709.8 s (45m09s), stage `pwml_export` |
| final canonical reaction count | **4 reactions + 1 transport** (the merged payload had 5; one was dropped at quarantine) |
| **F-179 verdict** | **CLEAN** — required-field gate `ok: true`, 0 errors, no `no_defensible_reaction_support` |
| final live Stage-3 / pre-export | `final_pre_export_stage3_gates` **`ok: true`**; required-field gate **`ok: true`, 0 errors** |
| **`C-119` superseded-report disposition** | **FIRED.** `post_normalization_contract_report` @ `phase: audit_round`, **2 errors**, superseded by `post_audit_contract_report`, `post_remap_contract_report`, `final_stage3_gate_report` |
| release state | **`review_required`**, `strict_acceptance_eligible: false` |
| PWML written | **YES** |
| path | `runs_validation/2026-09-07_1929/papers/PMC7232280/strict/pathway.review_required.pwml` |
| size | **45,054 bytes** |
| `review_required` or `release_ready` | **`review_required`** — `release_ready` was never reachable |
| new blocker | none |

**The exact shape that destroyed this leg in the pilot recurred, and this time it did not
destroy it:**

```
post_normalization_contract_report   ok=False   phase=audit_round   errs=2
post_audit_contract_report           ok=True
post_remap_contract_report           ok=True
final_pre_export_stage3_gates        ok=True
required-field gate                  ok=True, 0 errors
```

Both `C-119` preservation channels carried the finding, exactly as designed:

- **release reason** — `superseded_intermediate_contract_report:post_normalization_contract_report@audit_round=2`
  (alongside `semantic_evaluation_failed:requested_pathway_anchors_present`)
- **warning** — *"serialized with a superseded intermediate contract report … carried 2
  error(s), superseded by post_audit_contract_report, post_remap_contract_report,
  final_stage3_gate_report. The finding is NOT repaired and NOT dismissed: this leg is
  review_required and can never be release_ready."*

`superseded_contract_errors: 2` also reached the manifest row's `counts`.

---

# 3. `PMC8510960/strict` — FAIL at Stage 1. Not `C-119`.

| field | value |
|---|---|
| pipeline completion | **FAIL**, 667.1 s (11m07s), stage `stage1`, `failure_kind: unknown` |
| message | `Extraction failed: Chunk 2 failed to produce valid JSON.` |
| final canonical reaction count | **n/a — no canonical payload was produced** |
| F-179 verdict | **not reached** |
| final live Stage-3 / pre-export | **not reached** |
| `C-119` disposition | **not reached** — the leg died before the driver's contract channel |
| release state | *(no release record)* |
| PWML written | **no** |
| new blocker | **Stage-1 generation truncated: `finish_reason: length`** |

## The mechanism, in order

```
stage1_extraction               finish='length'  raw_chars=9501  -> invalid_json
stage1_json_repair              finish='stop'    raw_chars=84    -> semantic_guard_failed
stage1_extraction (retry)       finish='length'                  -> valid_json_zero_processes
stage1_extraction (retry)       finish='stop'    raw_chars=1267  -> valid_json_zero_processes
stage1_extraction (attempt 2)   finish='stop'    raw_chars=1159  -> valid_json_zero_processes
stage1_extraction_ladder_termination             -> ladder exhausted
```

**The repair path refused to fabricate.** `semantic_guard_failed` on the truncated remainder is
an anti-invention gate working: the system declined to complete a cut-off object rather than
guess at it. The leg failed in the safe direction.

## Four things kept apart, because `D-097` warned about exactly this

1. **`finish_reason: length` is a measured fact.** Generation was cut mid-object by *a* token
   ceiling. `D-097` said this "cannot be established without re-running"; it is now established
   **for this paper**.
2. **It does NOT retroactively explain the pilot's two JSON failures.** Those were different
   papers. `D-097`'s prohibition stands.
3. **It is draw variance, not determinism.** The **same paper** at the **same 16000 budget**
   produced a complete 5-reaction canonical payload in the pilot. CLAUDE.md's standing trap —
   identical legs give materially different Stage-1 draws at temperature 0 — is precisely this.
4. **"Raise the budget" is NOT supported by this evidence.** `raw_chars = 9501` is roughly 2.4k
   tokens, which is **inconsistent with a 16000-token cap actually being applied**. The binding
   limit may be provider-side rather than the configured one. Claiming the configured Stage-1
   budget caused this would repeat the error `D-097` exists to prevent.

**Not re-run**, per the authorization.

---

# 4. `PMC12071552/strict` — the CONTROL FAILED. Also not `C-119`.

| field | value |
|---|---|
| pipeline completion | **FAIL**, 2012.6 s (33m32s), stage `post_pipeline`, `failure_kind: contract` |
| message | `post-pipeline validation failed: 1 blocking issue(s) at final_pre_export_gate_lifecycle [final_gate_report_missing]` · issue code `gate.gate_lifecycle` |
| final canonical reaction count | 4 reactions + 2 transports extracted, **0 admitted to the core** |
| F-179 verdict | **not reached** — no viable core to evaluate |
| final live Stage-3 / pre-export | **`final_gate_report_missing`** — no final gate report exists, so `gate_verdict` **failed closed** |
| **`C-119` disposition** | **DID NOT FIRE, correctly.** `post_normalization_contract_report` is `ok=True` with **0 errors**, so there was nothing to supersede. `superseded_contract_errors` is absent from `counts`; no superseded warning |
| release state | **`diagnostic_only`**, reason `strict_technical_gates_blocked_export` |
| PWML written | **no** |
| new blocker | **quarantine admitted nothing to the requested core** |

## Why it failed

```
refusal_reasons: minimum_core:no_surviving_process
                 minimum_core:core_process_count_below_minimum:0<1
                 minimum_core:requested_core_coverage_below_minimum:0.000<0.500

quarantine counts: core_accepted 0 · auxiliary_accepted 0
                   quarantined_unmapped_entity 6   <-- all six, one reason
```

**All six candidate processes were quarantined as `quarantined_unmapped_entity`.** That is
**`F-185`** — the entity-identity-resolution limitation `ORCH-725` characterized — biting
harder on this draw than on the pilot's. In the pilot this paper resolved `DltA → P0C397` and
`DltD → Q2FZW3` and got two reactions through; in this draw identity resolution failed across
the board and nothing survived.

`final_gate_report_missing` is a **consequence**, not the cause: there was no payload to gate,
so no final report was written, and `gate_verdict` refused rather than assuming success. That
is the same fail-closed principle `C-119` seam 1 preserves.

**Not re-run**, per the authorization.

---

# 5. What the control's failure means — and what it forbids

**It does not undermine the positive result.** `PMC7232280` is verified mechanistically, not
statistically: the superseded snapshot was present with 2 errors, every live boundary was
clean, `C-119`'s exclusion fired, both preservation channels carried the finding, and a
45 KB `review_required` PWML exists on disk. That chain is either true or false, and it is
true.

**It does forbid any rate claim from this dataset.** A validation whose known-positive control
does not reproduce cannot be used to estimate a success rate. **"C-119 recovers N of M" may not
be stated on the basis of these three legs.** Run-to-run variance dominates a sample this size:

| paper | pilot | ORCH-730 | same config both times |
|---|---|---|---|
| `PMC7232280` | canonical payload, no PWML *(destroyed by F-147)* | **PWML** | yes |
| `PMC8510960` | canonical payload, no PWML *(destroyed by F-147)* | Stage-1 truncation, no payload | yes |
| `PMC12071552` | PWML | **no core admitted** | yes |

Three legs, same code, same budgets, same model, temperature 0 — and **two of three landed
somewhere different from the pilot.** That is the honest headline about reproducibility, and it
belongs in the manuscript's limitations next to `F-185` and `F-186`.

---

# 6. For the human reviewer — `PMC7232280` package

`docs/pwml_recovery_sprint/pathwhiz_review/ORCH-730/` contains the PWML, the canonical payload,
both gate reports and `RESULT.txt`. **No human has reviewed this pathway's biology.** Two things
to look at first:

1. **A CHEMICAL GAP.** The chain is
   `GTP → 3',8-cH2 GTP` … `cPMP → MPT → MPT-AMP → Moco`.
   **The `NIT-7B` step converting `3',8-cH2 GTP` to `cPMP` is absent from the canonical payload**
   — it was present in the merged payload and dropped at quarantine. So `3',8-cH2 GTP` is
   produced and never consumed, and `cPMP` is consumed but produced only by the transport. The
   graph still reports one connected component at 100% because the cPMP transport bridges it.
   **The connectivity metric does not see this gap; a chemist will.**
2. **`cached-name` is `Generated Pathway`**, not "molybdenum cofactor biosynthesis", and
   `pw-id` is the placeholder `PW000000`. Cosmetic for review, but it must be corrected before
   any PathWhiz submission.

Also note: `NIT-7A`, `NIT-9G`, `NIT-9E` ship as **Unknown-backed functional complexes**
(PathBank record 9659, `D-070 § O-1a`), and `NIT-9G`/`NIT-9E` are protein **domains** of the
two-domain *nit-9* product, not standalone proteins — `ORCH-725` § 6 corrected `ORCH-724` on
that point. 0 of 4 reactions lack `provenance_lineage`.

---

# 7. Standing conclusions

- **`C-119` is validated and stays merged.** Production remains frozen at `c7a0663e`. No
  production code was modified by this task.
- **`F-187` is answered for one of two papers.** `PMC7232280` now has a real PWML. `PMC8510960`
  does not, and its blocker is upstream of `C-119`.
- **`F-185` is re-confirmed on a fresh draw**, and it is what destroyed the control.
- **No new engineering wave.** Two legs failed for reasons that are already registered findings
  (`F-185`) or already-warned-about symptoms (`F-186`/`D-097` truncation). Neither is a new
  defect and neither should be chased now.

*`main` untouched. Pilot read-only. Protected state preserved. No production edits.*
