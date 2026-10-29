# ORCH-793 RESULT: alias identity from UniProt's own gene names (D-119 § 2)

**Branch:** `orch792/fix` (base `f3a80e09`), shared with ORCH-792. Code commits `74d6ecb7`, `63aa125b`.
**Merge rule 5 waived** by the product owner (D-119 § 4): no reviewer round was run.
**Scope:** ORCH-789 C4, 9 rows: PIF1 ×3 (chl), TAK1 (wnt_ca), c-Jun ×2 (wnt_pcp), Fur ×2 (enterobactin), Kir2.1 (sour).
**Result:** Fur recovered (2 rows). The other 7 rows **cannot** be decided from UniProt's gene-name and gene-synonym fields, so they stay refused (§ 6). No gate was relaxed, no fuzzy matching was added, and a real id is still required (D-106).

## 1. Cause, per row (verified against recorded answers)

Each leg's recorded UniProt answers were read back from the run's `data/id_mapping_cache.json` and run through `_apply_name_plausibility_gate`. Script: `evidence/orch793_gate_probe.py`. Output: `orch793_gate_probe.out.txt`.

Gene fields were confirmed with one-off UniProt GETs: `evidence/orch793/record_uniprot.py` → `orch793/uniprot_entries.json`. G11: `ORCH-793/05-record-uniprot-2.json`. A first attempt, `04`, hit HTTP 500 on P93831 and was retried.

| row | paper name vs UniProt | refusing rung | decidable from gene fields? |
|---|---|---|---|
| Fur (E. coli) | `fur` is the primary gene of P0A9A9 (K-12), P0A9B0 (CFT073) and P0A9B1 (O157:H7), all reviewed, plus unreviewed strain records (B7MFV0, ...). One protein name throughout | margin 0.00, the strain records as rivals | **yes**: one gene, strains of the pathway species |
| PIF1 (Arabidopsis) | primary gene of Q8GZM7. Also a gene **synonym** of P93831 (CLF), reviewed | margin 0.00 | **no** (see NARF, § 4) |
| TAK1 (human) | a gene **synonym** of both O43318 (MAP3K7) and P49116 (NR2C2), and the primary gene of neither. P49116 is also "Orphan nuclear receptor TAK1" | margin 0.00 | **no** |
| c-Jun (human) | P05412 (JUN): gene JUN, **no synonyms**. "c-Jun" is only in the protein name "Proto-oncogene c-Jun" | score 0.47 < 0.5 (Phase 2). Earlier, PathBank's ambiguous answer (JIP1, JIP4) returned before UniProt was asked | **no**: not a gene name or synonym |
| Kir2.1 (mouse) | P35561: gene Kcnj2, synonym Irk1. "Kir2.1" is only in the alt name "Inward rectifier K(+) channel Kir2.1", which the candidate parser does not keep | name rung: `implausible_name_match` | **no**: not a gene name or synonym |

## 2. Change

| file | function | change |
|---|---|---|
| `src/t2pw/mapping/map_ids.py` | new `_same_gene_record_in_species` | Two records count as one gene in the pathway species only when all of these hold: the judged record is reviewed; the other declares a review status; they share a **primary** gene symbol; they have the same recommended protein name; their **organisms differ** (different strains); and both are `ok`, never `genus_level`, against the requested organism. |
| `src/t2pw/mapping/map_ids.py` | `verify_real_protein_identity`, rung 6 (margin) | When the paper's name is the judged record's primary gene name, a rival meeting the test above is collapsed (listed in `collapsed_duplicate_accessions`) and does not contest the margin. |
| `src/t2pw/mapping/map_ids.py` | `_resolve_ambiguous_protein_candidates` | When several survivors are all such records, one is chosen in a **stable order**: reviewed first, then score, then accession. List order is never used. New ambiguity reason: `same_gene_records_in_species_resolved_by_stable_order`. |

Candidate gene synonyms are read where they already are: `_extract_uniprot_candidates` → `gene_names` / `primary_gene_names` (MASTER_PLAN § 2: nothing rebuilt). `_name_gate_verdict` is unchanged: it already keeps an exact gene-symbol match (`exact_symbol_identity`).

## 3. Proof

| proof | result | report |
|---|---|---|
| G9 base arm (`tests/test_orch793_gene_name_identity.py` copied in, then removed; base clean afterwards) | **2 G9 failed** (Fur verify, stable-order pick). 4 preservations passed | `evidence/g11/ORCH-793/20-g9-base-2-r1.json` |
| tip: the same file plus `test_c122_db_first_alias.py` and `test_orch762_protein_identity.py` | 67 passed | `14-focused-tip-3.json` (before the NARF correction); the final tip is in the next row |
| affected batch at tip (43 files: `C:/t/orch792/affected.txt`, recorded in the report's command) | **1133 passed, 8 skipped, 0 failed** | `21-affected-tip-3-r1.json` |
| gate probe on the recorded answers, base vs tip | Fur: refused → `P0A9A9` verified. PIF1, TAK1, c-Jun, Kir2.1: unchanged | `08-probe-gate-base.json`, `24-probe-gate-tip-final.json` |
| leg replay with taxonomy ids (`evidence/orch792_replay.py`, the run cache as recorded answers) | see § 5 | base `17-final-base-taxa-r1.json`, tip `22-final-tip-taxa-2-r1.json` |
| audit of every accession the tip ships that base did not | Fur `P0A9A9`; chl ferrochelatase `F4IMT3` (§ 7 Q3) | `23-new-ids-audit-2-r1.json`, `evidence/orch793_replay_taxa.txt` |

## 4. Dead ends and wrong theories (record these: they cost a reverted rule)

1. **"A primary gene name outranks another entry's synonym."** It recovered PIF1 over CLF. The new-id audit (`19-new-ids-audit.json`, kept in `orch793_replay_taxa.txt` as superseded) then showed it admitting **NARF → Q9UHQ1** (nuclear prelamin A recognition factor, primary gene NARF) on wnt_ca. That beat Q8WVD3 (RNF138), which carries NARF as a synonym and is the NLK-associated RING finger protein the Wnt/Ca2+ paper means. The rule picks the wrong protein, so it was removed (`63aa125b`). PIF1 is refused again, and `test_preservation_a_primary_gene_name_does_not_outrank_another_entrys_synonym` pins both cases.
2. **Collapsing curated records with the same gene and name in the same organism.** This broke `test_c122` EnzX (two curated claims in one organism) and WDG1 (same organism string, different taxon ids). Both must stay refused. Fixed by requiring **different strains**.
3. Wrong theory: "Fur fails on PathBank's ten iron-sulfur candidates". That is the first refusal only. The Phase-2 UniProt answer, P0A9A9, then failed on strain-record rivals (margin).

## 5. Leg replay (processes accepted by strict quarantine; combined ORCH-792 + ORCH-793 tree)

| leg | base | tip | note |
|---|---|---|---|
| enterobactin (ORCH-784) | 4 | **7 (+3)** | Fur → P0A9A9. Its signed `Fur inhibits ...` rows are drawn |
| chlorophyll (ORCH-785) | 81 | 84 (+3) | ORCH-792 rows kept. Ferrochelatase is resolved by Phase 2 now that its row survives (F4IMT3). PIF1 stays KI-001 |
| menaquinone (ORCH-784) | 23 | 26 (+3) | ORCH-792 (KI-007 rows kept) |
| wnt_ca / wnt_pcp / sour_taste_v2 | 13 / 0 / 6 | 13 / 0 / 6 | wnt_pcp refuses at `minimum_core` in this replay, at base and tip alike |

Rows of the 9 recovered: **Fur 2 of 2 (enterobactin 67% → 83% by ORCH-789's count).** PIF1 0 of 3, TAK1 0 of 1, c-Jun 0 of 2, Kir2.1 0 of 1: § 6.

## 6. Pins moved

- `tests/test_orch762_protein_identity.py::test_preservation_the_uniprot_tier_for_e_coli_judges_the_same_records`. **Before:** nothing shipped, and every verdict was `ambiguous_insufficient_margin`. **After:** EntC, EntB, EntA, EntE, EntF, EntD, Fur, TolC and MenD ship their judged K-12 records (`verified_real_protein`), and TonB stays `ambiguous_insufficient_margin`. Same mechanism as Fur. Annotated in the test.

## 7. Not settled (for the product owner)

1. **PIF1, TAK1, NARF: a name that is one entry's gene name and another's synonym, or a synonym of two entries.** UniProt's gene fields cannot choose, and the obvious tie-break picks wrong (NARF). Options:
   - (a) leave them refused (now). Cost: chl −3, wnt_ca −1 signed rows;
   - (b) a curated per-pathway alias table (PO-maintained) naming the meant entry;
   - (c) use the paper's own wording (e.g. "kinase", "receptor") against UniProt function. That is LLM-adjacent and out of this round's scope.
2. **c-Jun, Kir2.1: the name is in UniProt's protein names, not in its gene fields.** D-119 § 2 as written does not cover them. Options:
   - (a) extend § 2 to UniProt's recommended, alternative and short protein names, with an exact whole-word match in the pathway species and the same uncontested-margin rule. This needs the candidate parser to keep alt names (`_extract_uniprot_candidates`);
   - (b) leave them.

   Cost of (b): wnt_pcp −2 (c-Jun transports), sour −1.
3. **Ferrochelatase → F4IMT3.** It is admitted by the existing ladder, not by this rule: Phase 2 now sees the row (ORCH-792). It is a generic enzyme name resolving to one Arabidopsis record (FC1 and FC2 exist). Worth one look from a bio auditor.

## 8. KI delta (catalogue not edited)

- KI-001: the strain-record tie for E. coli proteins (Fur, Ent*, TolC, MenD without PathBank) is fixed. PIF1, TAK1, NARF, c-Jun and Kir2.1 stay KI-001 (§ 7).
- New diagnostic values only (no omission code): ambiguity reason `same_gene_records_in_species_resolved_by_stable_order`; verdict `collapsed_duplicate_accessions` now also lists strain records.

## 9. LLM-side

None in this lane. All 9 rows' losses are lookup and verifier behaviour.

## 10. Proposed LEDGER row

`ORCH-793 | orch792/fix @ <tip> | partial (rule 5 waived, D-119 § 4) | strain records of one gene in the species are one answer; stable-order pick | G9 base 2 red (g11/ORCH-793/20), affected 1133 pass (21), probe Fur→P0A9A9 (24), replay entero +3 (17/22) | open: PIF1/TAK1/NARF undecidable from gene fields; c-Jun/Kir2.1 outside § 2 wording`
