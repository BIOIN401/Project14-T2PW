# ORCH-791 RESULT: every identity step runs before Stage-6 endpoint wrapping (ORCH-789 C1)

Branch `orch790/fix` off `f3a80e09`. Code commit `64a06040`, shared with ORCH-790 because both change `map_ids.py`. Not merged, not pushed.
**Merge rule 5 is waived by the PO for ORCH-790..795 (D-119 § 4): no reviewer round was run.**

## 1. Cause (checked by replay)

Two identity steps ran after the first `_rewrite_reaction_protein_enzymes_to_complexes` pass. That pass wraps interaction endpoints and also drops id-less actors.

1. **The ORCH-782 gene/protein split** runs in quarantine, after all of Stage 6. At wrap time, "HY5 activates CHLH (expression)" still names a symbol that belongs to both a gene and a protein. The `shared_name` fail-closed guard therefore skipped the whole row, and HY5 stayed bare. The base replay shows all 11 chlorophyll rows as `bare_protein_endpoint [HY5 -> CHLH gene]`.
2. **The ORCH-783 family conversion** ran after the rewrite, so the rewrite saw the family names before conversion:
   - "Fzd2 activates G protein" was judged against a bare G protein with no identity, so Fzd2 stayed bare.
   - The id-less "PKC" modifier was **dropped** as redundant (KI-007, `reaction_enzyme_protein_dropped_no_external_id`). All 5 PKC rows then named an undeclared entity.

## 2. Change (wrapper rule unchanged: species + UniProt/DrugBank on every protein endpoint)

| file | function | change |
|---|---|---|
| `mapping/map_ids.py` | `map_payload` (step order) | `_apply_family_complexes` moves to just before the first rewrite. Its report entries keep their old position in the report, and the Unknown fallbacks stay where they were. |
| | `_rewrite_reaction_protein_enzymes_to_complexes` (interaction section only; the KI-007 hunk is untouched) | Runs `strict_quarantine._split_gene_protein_names` on a copy that holds only names. Endpoints are then read as quarantine will see them: a gene-role target is the gene and is not wrapped, the leftover bare symbol belongs to the protein alone, and a row the split cannot decide is skipped as before. The split itself still runs, and still records its omissions, in quarantine. |
| `pipeline/strict_quarantine.py` | none | Not edited, only imported. |

## 3. Proof (`g11_evidence.py check --task ORCH-791`: 0 non-compliant)

| what | result | report |
|---|---|---|
| G9 base: `test_g9_regulator_of_a_gene_role_target_is_wrapped`, `test_g9_partner_of_a_converted_family_is_wrapped`, `test_g9_family_modifier_is_converted_not_dropped` | **3 failed** on base, each on a value | `evidence/g11/ORCH-791/02-g9-base-r1.json` |
| same, tip | 3 passed | `01-g9-tip-r1.json` |
| affected files and leg replays | shared with ORCH-790 | `evidence/g11/ORCH-790/19-tip-affected-r1.json` (517 passed); `17`/`18`, `orch790_replay_{base,tip}.txt` |

Rows this lane recovers in the replay, base → tip:
- chlorophyll: **+11**, all `bare_protein_endpoint` → drawn: HY5 ×3, PIF3 ×2, FHY3 ×2, ABI4, RVE1, SCL27, TOC1.
- wnt_ca: Fzd2 → G protein (+1), and the 5 PKC rows that were undeclared (+5).

The preservation case passes. A clash target with no gene cue is still omitted by the split, and is still not wrapped.

## 4. Stage-6 / pre-freeze step order, final

`map_payload`, Stage-6 call (`allow_complex_wrapper_creation=True`):

1. Invalidate the requested cache keys.
2. Protein identity loop: PathBank, then UniProt; name gate; ambiguous-candidate verification.
3. **Family conversion** (`_apply_family_complexes`). Moved here.
4. First rewrite: reaction enzyme/modifier wrappers, the KI-007 drop, and interaction-endpoint wrapping (reads the split decision).
5. Structural cleanup phase 1 (`_prune_invalid_complex_components`).
6. Phase-2 UniProt API fallback for proteins still without an id.
7. ORCH-759e late rewrite for proteins resolved in step 6, then prune.
8. Complex mapping loop (PathBank complexes; F-133 superset refusal).
9. Compound mapping loop.
10. Unknown enzyme fallback, then Unknown complex fallback (D-070); family report entries merged here.
11. Stamp named-entity mapping metadata; `cache.save()`.
12. Identity admission (`_admit_identities`).
13. `_drop_unidentified_family_members`, then the Unknown complex fallback (ORCH-786).
14. Mapping lineage; then, if `merge_identity_duplicates`, the identity-duplicate merge and borrowed identities.

Then, still before the freeze, `quarantine_and_close`:

15. Gene/protein split (`_split_gene_protein_names`).
16. Registry.
17. Per-row admission.
18. Graph closure.

## 5. Audit: other mis-sequenced steps (found, not changed)

- **Admission (12) runs after wrapping (4, 7).** A wrapper can be built around a protein whose accession admission later withholds. Today only family members are caught by step 13. A single-protein wrapper still stops `post_remap` (ORCH-786 Q2). Admission must judge the final rows, and moving it is a cross-lane change. Open.
- **Family complexes now pass through the complex mapping loop (8).** That loop uses a looser PathBank name match than ladder step 2. The F-133 guard refuses a superset identity for a generated complex. This is untested against the live DB, because the replay has no DB. Open, low risk.
- **Interaction-endpoint wrappers built in step 4** are mapped by step 8 as before. No new wrapper is created after step 8.

## 6. Pins moved, KI delta, LLM-side

- Pins moved: none.
- New reason codes: none. KI-011 (split), KI-001 (families) and KI-007 (the drop, now never hits a family) are unchanged in the catalogue.
- LLM-side: none found in this lane.

## 7. Not settled

1. The admission-after-wrapping order (§ 5, first item). Options:
   - (a) a drop-or-Unknown rule for any single-protein wrapper whose member admission later withholds. This mirrors ORCH-786 and costs a small lane.
   - (b) accept the rare `post_remap` stop.
2. A live DB check of family complexes in step 8: one live leg.

## 8. Proposed LEDGER row

`ORCH-791 | builder | orch790/fix @ 64a06040 (+docs) | Stage-6 order: family conversion before the first wrap/drop rewrite; endpoint wrapping reads the ORCH-782 split decision | G9 base 3 red (ORCH-791/02) / tip green (01); affected+replay shared with ORCH-790 (19, 17/18): chl +11 bare endpoints, wnt_ca Fzd2 +1 and PKC +5 | rule 5 waived (D-119 § 4) | open: admission after wrapping`
