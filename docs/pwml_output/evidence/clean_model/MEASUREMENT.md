# Clean model measurement (dispatch M 2026-11-10, refreshed after the review fixes, fixer F, 2026-11-10)

Scripts in `scripts/`; JSON in this folder. Sections 1-4 are re-run on the fixed code (same 152 runs, same 10-run sample, seed 20261110).
The fill pass of section 3 was served almost entirely from the on-disk cache (4 live requests, the new GO `part_of` ancestors call); every item
carries the retrieval date of the cache entry it came from, not the date of this re-run.

## 1. build_all (152 canonical runs, offline)
| measure | value |
|---|---|
| built | 152/152 |
| `to_payload(build(p)) == p` | 152/152, 0 diffs (compound `class`, `compound_class`, `compound_type`, `type` are now typed fields and still round trip) |
| PMID / PMCID / DOI / organism | 152 / 150 / 150 / 152; a PMID read from the PMC text header token is labelled `inferred` (source "PMC text header"), a PMID from `request.json` stays `paper` |
| title / year | 0 / 0 (payload has none; fill supplies them) |
| quotes kept (reactions / transports / interactions) | 621/621, 106/106, 351/352 |
| complexes with Unknown member | 138/415 (43 runs); all 138 are all-Unknown |
| interactions with undeclared endpoint | 15 |
| `substrate of` / `provides substrate for` | now `other_unsigned` (were `catalysis`): interaction mix shows catalysis/unsigned 11, other_unsigned/unsigned 6 |
Details: `build_all.md`, `build_all.json`.

## 2. SBML validation, all 152, no fill (libsbml 5.21.2, units checks off)
| measure | value |
|---|---|
| files read by libsbml | 152/152, 0 read errors, 0 exceptions |
| errors / fatals | 0 |
| warnings | 2718, all id 80601 ("species should set an initial value"), in 152/152 files |
| species / reactions total | 2718 species / 727 reactions (1 warning per species; no `Unknown` species) |
Per-file records: `validate_sbml.json` (messages stripped), `validate_sbml.md`.

## 3. Live fill, 10 runs (seed 20261110, `fill_sample.json/.md`, per-item detail `measure_fill.json`)
| measure | value |
|---|---|
| requests / cache hits | 4 / 133 (cache warm from the first pass) |
| failures, rate limits, refused | 0 / 0 / 0 |
| `to_payload` equal to original after fill | 10/10 |
| fields filled / ids filled | 493 / 252 (was 426 / 185) |
| by source | UniProt 318, ChEBI 135, NCBI 30, GO 10 |
| by match type | accession 255, cross_reference 225, exact_label 10, exact_synonym 3 |
| conflicts (frozen kept) | 8 (7 taxon 562 vs 83333, 1 SAM CAS) |
| ambiguous | 8 compound ChEBI name lookups |
| titles / authors / year filled | 10/10 each |
| GO ids / parents | 10 (9 `location.go_id` + 1 `location.parent`, now from `part_of` ancestors only) |
| protein ids | `protein.ensembl` 79 (UniProt `ProteinId` property), `protein.ensembl_gene` 99 (transcript and gene ids), `protein.kegg_gene` 32 |
| reaction EC / Rhea | none: the catalyst-EC inference is opt-in (`infer_reaction_ec=False`), so the Rhea and `catalyst_ec` rows of the first pass are gone |
| top fields | formula/inchikey/smiles 39 each, protein.synonym 72, protein.ensembl_gene 99, protein.ensembl 79 |

### Hand spot-check (first pass, accepted db fills, by entity name)
Checked 31 from the first-pass `measure_fill.json`; 24 correct, 2 wrong (one reaction, EC + Rhea), 2 questionable (one reaction, EC + Rhea), 3 not verifiable.
The 4 wrong or questionable fills were reaction EC/Rhea fills from the catalyst's UniProt EC (EntB isochorismatase, NDM-1); they are no longer produced by default.
The compound, GO, protein and pathway fills listed as correct in the first pass are unchanged.

## 4. SBML after fill (10 runs, `fill_model` then `sbml_text`; XML in `C:/t/ev-cm/sbml_out_fill/`, not committed)
| measure | value |
|---|---|
| crashes / exceptions | 0 |
| libsbml read / errors | 10/10 / 0 |
| warnings | only 80601 (2-38 per file) |
| `bqbiol:is` resources before -> after | 255 -> 354 (the first pass wrote Ensembl transcript ids as `is`; those are now `isEncodedBy`) |
| ENSP protein ids written as `bqbiol:is` | 72 |
| `bqbiol:isEncodedBy` resources | 85 Ensembl transcript/gene + 32 KEGG gene (19 annotation blocks) |
| Ensembl ids kept out of RDF (not an ENSP/ENST/ENSG form, e.g. bacterial ids) | 21, recorded in `t2pw:id valid="false"` |
| DrugBank `bqbiol:is` | 35 (incl. ids read from `mapped_ids.drugbank`) |
| SBML bytes before -> after | 317,270 -> 466,263 |
`measure_fill_validate.json` has per-file results.

## 5. KGML, all 152 (`kgml_all.json/.md`, `kgml_biopython.json`)
| measure | value |
|---|---|
| files written / own structural check | 152 / 152 (the check now requires every reaction id to be an entry id whose `reaction` attribute names the reaction) |
| parsed by Biopython `KGML_parser.read` | before the fix 0/152; after 152/152 (2486 entries, 724 reactions, 325 relations) |
| entries by id class | KEGG 556 (24.2%), non-KEGG prefix 747 (32.6%), undefined 990 (43.2%) of 2293 entity entries; plus 193 reaction placeholder entries |
| reactions | 724 (rn 0, rhea 0, undefined 724); 230 extra catalysts listed in the omissions sidecar |

## Review fixes (brief of fixer F): 13 items
Fixed with a test each: 1 Unknown sentinel refused in every reference type (SBML and KGML), listed in omissions; 2 ENSP -> `bqbiol:is`, ENST/ENSG and `kegg.genes` -> `bqbiol:isEncodedBy`; 3 cache hits keep their own retrieval date; 4 KGML reaction id = enzyme entry id (confirmed with Biopython: "Reaction ID n has no corresponding entry"), plus pathway `name="path:undefined"` (Biopython also rejects a name without `path:`); 5 plain modifiers `SBO:0000019` (OLS4: modifier), catalysts `SBO:0000013` (OLS4: catalyst); 6 licence order sorted; 7 header PMID `inferred`; 8 substrate relations `other_unsigned`; 9 GO parent from `relations=part_of`; 10 exporters have no `fill` parameter; 11 non-JSON 200 -> `FetchError`; 12 runlist path from `T2PW_RUNLIST`; 13 DrugBank from `mapped_ids` typed (`drugbank_mapped`) and written to SBML, class/type flags typed.
KGML subtype spelling settled: KEGG's KGML docs and real files (hsa04120) use `ubiquitination`, so the exporter's `ubiquination` was wrong and is fixed.

## Bugs and surprises
1. **EC from recommended name only** (first pass). Repro: `scripts/measure_fill.py` with `infer_reaction_ec=True`, run 4 "EntB isochorismatase reaction". NOT fixed (out of this brief); harmless by default because the reaction EC fill is opt-in.
2. **Reaction EC/Rhea inferred from catalyst only** (NDM-1 case). NOT fixed; off by default.
3. **Ambiguity rule is strict** ("chanoclavine-I", "festuclavine", "pyridoxal 5'-phosphate" become suggestions while "isochorismate" is accepted via synonym+charged_form). NOT fixed; safe but inconsistent.
4. **Taxon conflicts are noise** (E. coli 562 vs K-12 83333, 7 conflicts). NOT fixed.
5. `fill_sample.py` under `-I`: FIXED (it adds `src` to the path now; `runfill_wrapper.py` is no longer needed).
6. `validate_sbml.py --all` overwrites the committed 6-run `validate_sbml.json/.md` with the 152-run results (messages stripped). Unchanged, by design.
7. Warning 80601 on every species is by design (no amounts invented).
8. Open for the product owner: P-28 placeholder is now `name="path:undefined"` (was `undefined`) because Biopython's parser raises on a name without `path:`; a one-token change, reversible.
