# ORCH-786: a family-complex member with no id must not stop the pathway (wnt_ca)

Branch `orch786/fix` off `22538dee`. Code commit `3a90e354`. **Status: built, NOT TESTED.** The proof runs did not happen because free RAM stayed at 0.96-1.7 GB (user applications), below the 2 GB floor, for more than 30 minutes. Nothing is merged or pushed.

## 1. Root cause (inferred, not yet replayed)

ORCH-784 `wnt_ca-strict` stopped at `post_remap` with `generated_wrapper_component_missing_external_identity`.
- The failing complex is almost certainly the ORCH-783 family complex `STIM1/2` (source 1, members named by the paper).
- `_paper_members` reuses the existing real `STIM2` row, which has no `family_complex_provenance` route key.
- `_admit_identities` runs after `_apply_family_complexes`. Pass A then withholds STIM2's accession, because the paper writes "STIM1/2" and never "STIM2".
- Probe against the leg's own `source_text_index`: STIM1 is supported, **STIM2 is unsupported**, and Orai1-3 are supported.
- So this is an ORCH-783 complex, not a wrapper from another stage, and post_remap was left alone.
- Not proven on the leg itself: the leg's Stage-6 UniProt answers were never saved (`use_cache=False`), and the audited Stage-6 input was overwritten by later legs.

## 2. Change

- `src/t2pw/mapping/map_ids.py`: new `_drop_unidentified_family_members`, called in `map_payload` straight after `_admit_identities` (wrapper mode only).
- It drops each component of a `mapping_meta.family_complex` complex whose declared protein has no UniProt/DrugBank id. Rows are resolved by name, last row wins, exactly as `validate_post_remap` does.
- Each drop is noted on `family_complex.note` with KI-001; `reaction_evidence` already prints that note.
- It also records a `family_complex_member_dropped` action and a `family_complex_members_dropped` count (only when nonzero).
- If no member is left: the complex's species keys are reset through `_wrapper_species_fields`, then `_apply_pathbank_unknown_complex_fallback` adds the Unknown member (D-070, D-117 § 2 step 4).
- The max-3 rule and the species-match rule are untouched. The identity-admission gate is untouched.

## 3. Proof: pending (to do, one bounded heavy-lock job each)

1. Replay, `docs/pwml_recovery_sprint/evidence/orch786_replay.py <tree> <workdir>`, on the base tree `C:/t/orch786/base` and on the tip.
   - Offline: harness `orch762_harness.Doors`, every HTTP request answered 404, Stage-2 cache copied from the orch784run run.
   - STIM2's UniProt answer (Q9P246) is injected, because the leg never saved it.
   - Expected: base STOP on STIM1/2 / STIM2, tip PASS. Save the output to `evidence/orch786_replay.txt`.
   - Probe at base without the STIM2 injection (`C:/t/orch786/replay-probe.json`, 0 survivors): post_remap PASS. With STIM2 unresolved, the failure does not reproduce, which agrees with the inference in § 1.
2. G9 test `tests/test_orch786_family_member_identity.py` (2 tests, both G9 behavioural). It is already copied into `C:/t/orch786/base/tests/`. Expected: fails on base, passes at tip. Reports go to `evidence/g11/ORCH-786/01-g9-base.json` and `02-tip.json`.
3. Affected tests: `tests/test_orch783_family_complex.py` and the mapping-lineage golden at the tip.

## 4. Dead ends

- Replaying `merged_payload.json` fully offline: post_remap passes on base. STIM2 never gets an id offline, so the failure cannot appear. That is why STIM2's answer is injected.
- Stamping the route key on reused member rows: rejected. It would make admission abstain on a row whose name the paper does not contain, which weakens the gate (rule 6).

## 5. Open questions for the product owner

1. **Unknown fallback refused** (`_ensure_unknown_protein` returns None, for example when a real protein row is named "Unknown"). The emptied complex then still stops post_remap with `generated_wrapper_missing_components`. Per-row omission inside mapping has no text-record channel today.
   - (a) Accept: the case is very rare.
   - (b) Route the case through strict_quarantine per-row omission. Cost: a post_remap change.
2. **Single-protein wrappers.** A pre-ORCH-783 single-protein wrapper whose protein admission withholds would stop post_remap the same way. Not touched: out of scope, and not observed.
3. **Should "STIM1/2" count as paper support for STIM2?** That would be an admission-gate change, which only you can rule on.
