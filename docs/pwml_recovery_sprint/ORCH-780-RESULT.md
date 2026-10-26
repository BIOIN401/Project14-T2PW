# ORCH-780: RESULT (D-116 § 1: the "no defensible reaction support" refusal becomes a per-row omission)

| | |
|---|---|
| Lane | `orch780/support-per-row`, worktree `.claude/worktrees/orch780` |
| Base | `4a15ef42` |
| Commits | `ba95de39` code and tests · the docs commit after it |
| Status | **UNFINISHED: handed back at the 13:30 time box.** Variant B (§ 4) was implemented and measured, then **rejected by D-112 § 2** because gold recall was lost. Variant A (§ 2) is committed, but it is **not measured and its tests have not been run**: available RAM fell to 0.6 GB and the lead's 3 GB floor blocked every job after 11:16 (§ 7). Not reviewed, not merged. |
| Boundary | `src/t2pw/pipeline/strict_quarantine.py` only: `_row_failures` (4 lines), the new helper `_unsupported_reactions`, the section comment, and `QUARANTINE_POLICY_VERSION`. Also tests, this file and evidence. The gate's module `reaction_support.py` is **not** edited (§ 1, item 2). |

## 1. The gate-predicate finding

Read at base: `src/t2pw/pipeline/reaction_support.py` and `stage_contracts.validate_pre_export:343-400`.

1. **The gate's predicate is "at least one reaction is supported".** `reaction_support_issue` fires only when
   `evaluate_reaction_support(...)["verdict"] == "no_defensible_core"`. That verdict needs three things: the payload has at least
   one reaction, **no** reaction is supported, **and** `lineage_carrier_active(payload)` is true. Without the carrier the verdict
   is `indeterminate` and the gate abstains. The docstring says so: "A LEG-LEVEL FLOOR ..., NOT PER-ROW ENFORCEMENT"
   (`:7-14`, `:86-89`).
2. **The gate already has a public per-row predicate**: `reaction_support_class(row)`, which returns `target_paper`,
   `external_rag` or `None` (`:142-166`). A row is supported when its lineage has an entry with `origin == "paper_stated"` or
   `paper_explicit == "explicit"`, or an entry with `origin == "rag_literature"`, or when the row carries a truthy `rag_provenance`.
   The aggregate is built from exactly this function (`:212-214`). Nothing needed extracting, so the gate module is not in the diff.
3. **Yes: at base, unsupported reactions ARE exported whenever another reaction in the leg is supported.** This is measured, not
   inferred. Of the 72 legs that write a PWML at base, **25 export 47 such reactions** while the carrier is active
   (`evidence/orch780/04-support-census-base.log`, script `evidence/orch780/support_census.py`, G11 `ORCH-780/04`). Every
   leg and row is listed in § 4.2.
4. **No other gate omits unsupported reactions per row.** `_weak_evidence_verdict` is a confidence floor, and is off unless a
   floor is passed. `_scope_is_out_of_scope` reads `scope_membership`. None of the ORCH-773 per-row checks reads
   `provenance_lineage`: grepping `process_normalizer.py`, `stage_contracts.py`, `pwml/ir.py` and `pwml/writer.py` finds no hit.

### The predicate, exactly (for lane ORCH-777)

**Supported ⇔ `t2pw.pipeline.reaction_support.reaction_support_class(row) is not None`**, called as it is, on the payload the
export freezes. Lane 777 should test exactly this. In this lane, a reaction is judged only when the gate itself refuses
(`no_defensible_core`). In that case the carrier is active and every reaction has class `None`, so lane 777 rescues none of them
and the two lanes cannot disagree. When the carrier is inactive the gate abstains and this lane omits nothing. A reaction there
also has class `None`, so lane 777 would not rescue it either: it is exported exactly as at base.

## 2. Design (variant A, committed in `ba95de39`)

All of it sits in ORCH-773's per-row framework, so it runs before the freeze and in strict mode only.

* `_row_failures`: when the required-field gate (`_GATE_REQUIRED`) reports `no_defensible_reaction_support`, the call returns
  `_unsupported_reactions(candidate, by_position, gate)`. That is one process omission request for each reaction of the freeze
  candidate whose `reaction_support_class(row) is None`. The gate ran on that same candidate inside `_downstream_gate_errors`.
  Before this change, the error's pointer `/processes/reactions` named no row, so the pathway was refused.
* `_apply_per_row_failures` is **not edited**. Each reaction becomes `quarantined_broken_reference` with
  `reason = "no_defensible_reaction_support"` (the gate's exact code) and `detail.omission = {gate: required_field_gate,
  cause: process}`, and its verbatim `original_process` is kept. The release is capped at `review_required`. The Reaction Evidence
  text lists the row in OMITTED FROM THE PWML as KI-022, through the existing mapping at `reaction_evidence.py:147`.
* The rollback is unchanged. If the omissions leave nothing to draw, every round is undone and the pathway ends on the gate's own
  refusal, as at base. **The gate's verdict is unchanged for every payload**, because no gate module is in the diff.
* `QUARANTINE_POLICY_VERSION` moves from `2026-10-25.1` to `2026-10-26.1`, following the module's own rule.

**Expected corpus effect of A (inferred, NOT measured):** variant B fixed 7 of the 8 KI-022 legs (§ 4.1), and A converts exactly
that case. On the 72 base-passing legs the gate does not fire, so A should change nothing apart from the version fields.

## 3. G9 fixture (`tests/test_orch780_support_per_row.py`, written, NOT RUN)

* `test_g9_unsupported_reaction_is_omitted_and_the_interactions_export`: the contract-alignment toy, with one reaction whose
  lineage is `inferred` and two signed interactions (activates, inhibits) between the two UniProt-backed enzyme complexes.
  * Expected at base: `required:no_defensible_reaction_support`.
  * Expected at tip: no refusal, the reaction in `quarantined[]` under the gate's code, both interactions in the IR, and
    KI-022 in the text record.
* Preservation tests:
  * a payload the gate passes is untouched;
  * supported reactions (paper or `rag_provenance`) are kept;
  * a payload with no carrier is not judged;
  * research mode omits nothing;
  * when nothing is left to draw, the pathway rolls back to the gate's own refusal.
* **None of these has run.** The base-red and tip-green runs are still owed. The toy may need iterating on the gates; for
  example, the interaction endpoints have not been checked against Stage 8.

## 4. Measurements (variant B, the earlier version: omit EVERY unsupported reaction, also when the gate passes)

Variant B ran the same predicate first, before every other check, guarded by `lineage_carrier_active`, so it also fired when the
gate passed. Its diff is `evidence/orch780/variant-b.diff`.

* Base corpus: `evidence/orch780/01-corpus-base.log` (G11 `01`).
* Tip corpus: `02-corpus-tip.log` (G11 `02`).
* `compare`: `03-corpus-compare.log` (G11 `03`), exit 1.

All wrapper reports show `FINAL SURVIVING COUNT : 0`, and every orphan sweep found 0 ORCH-780 python processes.

### 4.1 The `compare` verdict: **D-112 § 2 REJECT**

| | base `4a15ef42` | variant B |
|---|---|---|
| PWML pass rate | **72/91** | **79/91** |
| legs fixed | | 7: rv 08-22_2147 PMC12180156, rv 08-22_2147 PMC13231680, rv 08-24_1203 PMC13231680, rv 08-24_1428 PMC12180156, rv 08-28_1816 PMC12180156, rv 09-01_1612 PMC12180156, rv 09-02_2052 PMC12180156 |
| still KI-022, whole leg | 8 | 1: rv 08-21_2239 PMC12180156 (one reaction and nothing else, so it rolls back to the refusal) |
| passing legs turned failing | | none |
| **gold recall lost** | | **ORCH-755 enterobactin, final and PWML R 0.800 → 0.6; ORCH-759p enterobactin, final and PWML R 1.000 → 0.8** |
| coverage, headline (91 legs) | mean 53.3%, median 58.3%, ≥ 75%: **32/91** | mean 47.3%, median 50.0%, ≥ 75%: **24/91** |
| coverage, full count (91 legs) | mean 47.3%, median 46.7%, ≥ 75%: 23/91 | mean 41.8%, median 42.9%, ≥ 75%: 17/91 |
| coverage on legs writing a PWML, headline | 72 legs, mean 67.3% | 79 legs, mean 54.5% |

The coverage figures come from each run's own summary, `02-corpus-tip.log` and `01-corpus-base.log`, under "PWML COVERAGE".

**Why recall was lost.** The gold reaction `EntF-catalyzed enterobactin assembly` (ORCH-755) and `EntF enterobactin synthase
assembly reaction` (ORCH-759p) each carry a verbatim paper quote in `evidence`, but **no `provenance_lineage` at all**: see
`out-base/ORCH-755_enterobactin-strict_2026-10-20_2201/canonical.json`, the row at line 5500. The carrier did not stamp them;
they are probably rows added after Stage 1. The gate's predicate therefore calls them unsupported. This is a **lineage-carrier
gap upstream**, and per-row enforcement exposes it. Because of this I narrowed the change to variant A, which is also D-116 § 2's
own wording: "only the whole-pathway refusal goes", and the requirement "stays exactly as strict".

### 4.2 Base-passing legs that export unsupported reactions (variant B would omit them; variant A leaves them)

Source: `04-support-census-base.log`. There are 25 legs and 47 rows; the legs are listed below with their rows.

- ORCH-755 enterobactin: EntA dehydrogenase reaction with NAD+; EntF-catalyzed enterobactin assembly (gold).
- ORCH-756 enterobactin: LDH-catalyzed reduction of pyruvate to lactate (the gold forbids it, KI-032).
- ORCH-756 menaquinone: MenF reverse reaction.
- ORCH-759 chlorophyll: ferrochelatase reaction (protoheme).
- ORCH-759p enterobactin: EntF enterobactin synthase assembly reaction (gold).
- ORCH-759p chlorophyll: CAO reaction; Mg-chelatase reaction; singlet oxygen generation from Pchlide; LPOR Pchlide reduction.
- ORCH-767 chlorophyll: Mg-chelatase reaction; singlet oxygen generation from Pchlide; ferrochelatase reaction.
- ORCH-767 menaquinone: isochorismate to chorismate (MenF); isochorismate to DHNA (MenD) (the kinetic-model term, KI-032).
- rv 08-18_1328 PMC12452463: enterobactin iron binding.
- rv 08-21_2239 PMC12856317: ALAS2 reaction with pyridoxal 5-phosphate cofactor.
- rv 08-21_2239 PMC12452463: Enterobactin iron binding.
- rv 08-21_2239 PMC12782028: MSMO1 reaction.
- rv 08-22_2147 PMC12452463: Enterobactin binding to ferric iron.
- rv 08-24_1203 PMC12096016: EntA dehydrogenase conversion of 2,3-diDHB to 2,3-DHB with NAD+; Enterobactin chelation of ferric iron.
- rv 08-24_1428 PMC12782028: CYP51A1 reaction.
- rv 08-25_1216 PMC12444477: LpxH dephosphorylation; LpxD acylation.
- rv 08-27_1341 PMC12782028: cholesterol oxidation to oxysterols.
- rv 08-28_1816 PMC12452463: Enterobactin binding Fe3+.
- rv 09-01_1612 PMC12856317: heme inhibition of ALAS2.
- rv 09-06_1425 PMC12376012: 9 sphingolipid reactions.
- rv 09-06_1425 PMC11405693: PAL, C4H, 4CL.
- rv 09-16_2152 PMC13170813: 3 hordatine reactions.
- rv 09-16_2152 PMC13486683: 3 PhBBO/PhBS reactions.
- rv 09-16_2152 PMC12505894: Cwh8 dephosphorylation.
- rv 09-16_2152 PMC12059215: 2 HSD17B reactions.

The rows are a mixture. Some the gold forbids (LDH, the MenD kinetic term). Some are gold reactions that are not attributed (EntF).

### 4.3 WNT PCP

Both corpus WNT PCP legs **already write a PWML at base**:
- ORCH-767: headline coverage 1/49 (2.0%).
- ORCH-759: headline coverage 6/34 (17.6%).

Neither is a KI-022 leg, because each has a supported reaction or no reactions left (`01-corpus-base.log`). Variant A should
not change them; this is not measured. ORCH-774's WNT PCP leg, the KI-022 case that D-116 cites, is not in the corpus, so it
needs the live run. Variant B changed both legs' replay records; the per-row detail was not extracted before the time box.

### 4.4 Gold rationale for the two KI-022 papers (from `src/t2pw/bench/gold/pinned_v1.json`)

- **PMC12180156, 6 of the 8 legs.** The gold rationale reads "nothing about heme biosynthesis is exportable. At most the paper
  supports regulatory edges (SFXN4 inhibits ALAS2 ...)". An interactions-only PWML is consistent with that.
- **PMC13231680, 2 legs.** The gold rationale reads "Nothing lipid-A-related is exportable at any level of partiality. The correct
  pipeline outcome is an empty pathway plus a rejection reason." A PWML for these legs (now `review_required`, its reactions
  listed as omitted) is a **`policy_disagreement`** between this pre-D-114 gold rationale and D-114 AMENDMENT 2 § 3 / D-116. It
  is for the product owner (§ 8, item 2).

## 5. KI delta (for the catalogue owner; `PWML_KNOWN_ISSUES.md` and `KNOWN_ISSUES` are not edited)

* **KI-022:**
  * Symptom → "The reactions without reaction-level support are omitted before the freeze and listed (reason
    `no_defensible_reaction_support`); the rest of the pathway exports `review_required`. It is still a whole-leg refusal only
    when nothing else is left to draw."
  * Status → "**per-row omission** (D-116, ORCH-780 variant A), when no reaction is supported. While some reaction is supported
    the rule stays a leg-level floor (D-116 § 2)."
  * Corpus → 8 legs; B fixed 7, and A is expected to fix the same 7 (not measured).
  * Workaround (also the `reaction_evidence.KNOWN_ISSUES["KI-022"]` text, which is now stale) → "Add the reaction by hand only
    if the paper states it: check its evidence quote in the text record."
* **KI-032**: add "ORCH-780 measured 47 unattributed reactions exported on 25 base-passing corpus legs, both bad (LDH, the MenD
  kinetic term) and good (EntF assembly, which has a paper quote and no lineage)". Add a new cause: "rows added after Stage 1
  carry no `provenance_lineage`" (§ 4.1).
* **KI-021 / KI-065**: no change from this lane. A KI-022 leg whose only other content is withheld by the IR becomes a KI-065
  zero-process PWML; whether any corpus leg does is not measured.

## 6. Tests and pins

* **Not run:** focused tests, G9 base-red and tip-green, the strict-quarantine suites, Chunk E, orch773/768, strict-failure
  replay, SMOKE (511) and `test_streamlit_quarantine_boundary.py`.
* **Pins expected to move**, from the `QUARANTINE_POLICY_VERSION` bump: the `decision_input_hash` values in
  `test_orch769b_actor_protein_registry.py:101` and `test_orch759c_quarantine_bucket_closure.py:100`, and the `PINS` digests in
  `test_orch766_unreferenced_species.py:381` (the same 5-field delta ORCH-773 documented). Possibly also
  `test_strict_quarantine_versioning.py`. None has been recomputed yet.

## 7. Process record and failures

* **RAM.** The 6 GB floor was never reached: available memory stayed at 2.8-4.0 GB from 10:42, with no sprint python running.
  The lead ruled that the floor is 3 GB (message, about 11:13). Every log records its job-start figure as its first line.
  Jobs 01-04 ran at 3.5-3.6 GB. After 11:16, available RAM fell to 1.25 GB, then 0.64 GB at 13:11. Every later job was refused
  by `run.sh` (exit 96) before it took a G11 path, so no G11 artifact exists for those jobs. **About two hours of wall time
  elapsed with no job able to run**, so the time box ended this lane unfinished.
* **Dead end.** Variant B: per-row enforcement of the support predicate on every leg. It was measured, and REJECTED for loss
  of gold recall (§ 4.1). It is kept as `evidence/orch780/variant-b.diff`.
* **`g11_evidence.py check --task ORCH-780` was NOT run.** At 13:20 available RAM was still 523 MB, under the 3 GB floor.
  The four G11 artifacts (`01`-`04`) each report `FINAL SURVIVING COUNT : 0`. The check is owed before any merge.
* Orphan sweeps after each job: 0 survivors. Nothing was killed.
* Lock: never held outside the wrapper and never deleted.

## 8. Not settled (for the product owner)

1. **What D-116 § 1 covers.** Under (A, committed), only the whole-pathway refusal changes, which is § 2's wording; the 47
   unattributed reactions on base-passing legs keep exporting. Under (B), every unsupported reaction is omitted, which is
   § 1's literal wording. Measured cost of B: gold recall lost on 2 enterobactin legs, and headline coverage ≥ 75% falls from
   32 to 24 of 91, because the lineage carrier does not stamp some paper-stated rows. B becomes safe only after that carrier
   gap is fixed upstream.
2. **PMC13231680.** Its gold says "empty pathway plus a rejection reason"; D-116 exports what remains. Do you want the
   gold rationale amended, or this paper treated as an exception?
3. **Integration.** Lane ORCH-778 may also bump `QUARANTINE_POLICY_VERSION`, and the merged value must be distinct.

## 9. LEDGER row (proposed)

`ORCH-780 | D-116 per-row support omission | orch780/support-per-row @ <docs tip> | UNFINISHED: variant B measured and REJECTED (gold recall); variant A committed, not measured, tests not run (RAM) | ORCH-780-RESULT.md`

## 10. TEST_MATRIX delta (proposed)

Add the focused selection `tests/test_orch780_support_per_row.py`. Its 1 G9 test and 5 preservation tests are expected to pass
at the tip; none of them has run yet.
