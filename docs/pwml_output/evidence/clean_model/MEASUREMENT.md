# Clean model measurement (dispatch M, 2026-11-10)

No file under `src/` or `tests/` edited. Scripts in `scripts/`; JSON in this folder.

## 1. build_all (152 canonical runs, offline)
| measure | value |
|---|---|
| built | 152/152 |
| `to_payload(build(p)) == p` | 152/152, 0 diffs |
| PMID / PMCID / DOI / organism | 152 / 150 / 150 / 152 |
| title / year | 0 / 0 (payload has none; fill supplies them) |
| quotes kept (reactions / transports / interactions) | 621/621, 106/106, 351/352 |
| complexes with Unknown member | 138/415 (43 runs); all 138 are all-Unknown |
| interactions with undeclared endpoint | 15 |
Details: `build_all.md`, `build_all.json`.

## 2. SBML validation, all 152, no fill (libsbml 5.21.2, units checks off)
| measure | value |
|---|---|
| files read by libsbml | 152/152, 0 read errors, 0 exceptions |
| errors / fatals | 0 |
| warnings | 2718, all id 80601 ("species should set an initial value"), in 152/152 files |
| species / reactions total | 2718 species / 727 reactions (so 1 warning per species) |
Full run took ~4 s in one call (no chunking needed). Per-file records: `validate_sbml.json` (68 KB, messages stripped), `validate_sbml.md`.

## 3. Live fill, 10 runs (seed 20261110, `fill_sample.json/.md`, per-item detail `measure_fill.json`)
| measure | value |
|---|---|
| requests / cache hits (first, live pass) | 118 / 26 (one 88 s call; far under 3000) |
| failures, rate limits, refused | 0 / 0 / 0 |
| `to_payload` equal to original after fill | 10/10 |
| fields filled / ids filled | 426 / 185 |
| by source | UniProt 226, ChEBI 155, NCBI 30, GO 10, Rhea 5 |
| by match type | accession 255, cross_reference 146, catalyst_ec 12, exact_label 10, exact_synonym 3 |
| conflicts (frozen kept) | 8: 7 taxon 562 vs 83333 (species vs strain), 1 SAM CAS 485-80-3 vs 29908-03-0 |
| ambiguous | 10: 8 compound ChEBI name lookups, 2 reaction/pathway |
| titles / authors / year filled | 10/10 each |
| GO ids found / parents | 10 ids (9 `location.go_id` + 1 `location.parent`); 11 GO lookups no match |
| compound ChEBI by name | accepted 4, ambiguous 8, refused 0, no match 11 |
| protein suggestions | 0; no protein id filled by suggestion (protein fills: name, synonym, taxon, ensembl, kegg_gene, ec only) |
| top fields | protein.ensembl 79, synonym 72, formula/inchikey/smiles 39 each, kegg_gene 32, drugbank 24 |

### Hand spot-check (accepted db fills, by entity name)
Checked 31 from `measure_fill.json`; 24 correct, 2 wrong (one reaction, EC + Rhea), 2 questionable (one reaction, EC + Rhea), 3 not verifiable.
- Correct: GO mitochondrial matrix 0005759, mitochondrion 0005739, cytosol 0005829, cytoplasm 0005737, extracellular -> 0005576 (region), periplasm -> 0042597, ER 0005783; ChEBI pyruvate 15361, isochorismate 29780; 2,3-dihydroxybenzoic acid KEGG C00196, HMDB0000397, CAS 303-38-8; pyruvate C00022, CAS 57-60-3; enterobactin C05821, CAS 28384-96-5; isochorismate C00885; EntC reaction EC 5.4.4.2 / RHEA:18985; EntA EC 1.3.1.28 / RHEA:23824; LSS 5.4.99.7; CYP51A1 1.14.14.154; DHCR24 1.3.1.72.
- WRONG: run 4 "EntB isochorismatase reaction" got EC 6.3.2.14 / RHEA:30571 (enterobactin synthase). UniProt P0ADI4 lists 6.3.2.14 on the recommended name and 3.3.2.1 (isochorismatase) only on the included domain; `uniprot_entry` reads only recommended+alternative names.
- QUESTIONABLE: run 6 "phthalylsulfacetamide decomposition to sulfacetamide" got EC 3.5.2.6 / RHEA:20401 (beta-lactam hydrolysis) from catalyst NDM-1. A reaction id is attached by the catalyst's EC alone, with no check that the substrate fits.
- Not verifiable by hand: CHEBI:204810 cycloclavine, CHEBI:135528 phthalylsulfacetamide, KEGG C19241 fumonisin B1 (names plausible).

## 4. SBML with fill=True (10 runs, `C:/t/ev-cm/sbml_out_fill/`)
| measure | value |
|---|---|
| crashes / exceptions | 0 |
| libsbml read / errors | 10/10 / 0 |
| warnings | only 80601 (2-38 per file) |
| `bqbiol:is` resources before -> after | 225 -> 384 (per run: 23>34, 20>25, 16>17, 21>24, 23>56, 26>107, 10>17, 72>87, 10>12, 4>5) |
| SBML bytes before -> after | 311,300 -> 443,694 |
`measure_fill_validate.json` has per-file results.

## Bugs and surprises
1. **EC from recommended name only** (above). Repro: `PYTHONPATH=C:/t/cm/src PY -I -X utf8 scripts/measure_fill.py`, then look at run 4, item "EntB isochorismatase reaction" `ec_number`. Fix idea: read `includes`/`contains` ecNumbers, or skip the reaction fill when the protein lists any non-matching EC.
2. **Reaction EC/Rhea inferred from catalyst only** (NDM-1 case). The `inferred` label is present but the id is still written; consider not writing `rhea_id` unless reaction substrates match.
3. **Ambiguity rule is strict**: "chanoclavine-I", "festuclavine", "agroclavine", "pyridoxal 5'-phosphate" each have one exact_label ChEBI hit plus a charged-form synonym hit, and all become suggestions. Yet "isochorismate" is accepted via synonym+charged_form. Behaviour is safe but inconsistent; the exact_label-wins case is the likely 8 -> fewer.
4. **Taxon conflicts are noise**: E. coli 562 vs K-12 83333 flagged 7 times (6 in one run); consider a species/strain parent check.
5. `fill_sample.py` does not add `src` to `sys.path` (build_all and validate_sbml do), so `-I` fails with ModuleNotFoundError. Workaround: `scripts/runfill_wrapper.py` (inserts path, runs the script). Repro: `PY -I -X utf8 scripts/fill_sample.py --runs 1 --out X`.
6. `validate_sbml.py` overwrites the committed 6-run `validate_sbml.json/.md` with the 152-run results; the raw JSON was 1.07 MB, so messages were stripped (aggregate kept).
7. Warning 80601 on every species is by design (no amounts invented).
