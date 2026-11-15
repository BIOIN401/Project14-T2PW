# B2: vocabulary map (final payload phase 3)

Code: `src/t2pw/pwml/vocabulary_map.py` (table + `map_vocabulary_name`), `src/t2pw/pwml/writer.py:797` (`_vocabulary_fields`), used at :1021 (cell types), :1061 (locations), :1073 (tissues). Commit 3e782d76 on `pwml/final-payload`. Flag `T2PW_FINAL_PAYLOAD` off: the three sections are written exactly as before.

Source of the target names: harness seed `C:/t/pwi/t2pw/tools/pathwhiz_import/seed/vocab_inserts.sql` (grep of the PathBank dump: 77 locations, 43 cell types, 53 tissues). They are copied into `PATHWHIZ_ROWS`; a test checks every synonym target is a key of that table. The 40 names come from `evidence/import_triage/s4_residual_ontology_misses.json` (28 files, 57 occurrences).

## How a name is written (flag on)
| outcome | rule | written |
|---|---|---|
| exact | name already a row (case-insensitive) | unchanged |
| mapped | reviewed synonym; the record carries no ontology id, or one equal to the row's | the row's name, ontology id left out (the importer finds the row by name, as it does for the 89 existing hits) |
| new_row | not a row, no synonym, but the data already carries an ontology id | unchanged name + the id already in the data, so PathWhiz can create the row |
| unmapped | not a row, no synonym, no id | unchanged; counted |

Counters: `layout_debug_counts["vocabulary_exact" | "vocabulary_mapped" | "vocabulary_new_row" | "vocabulary_unmapped"]`; non-exact items listed in `builder.vocabulary_outcomes` (kind, name, outcome, written).

## Table of the 40 names
| kind | name (files) | outcome | target / reason |
|---|---|---|---|
| location | ER (1) | mapped | Endoplasmic Reticulum (GO:0005783); standard abbreviation |
| location | endoplasmic_reticulum (1) | mapped | Endoplasmic Reticulum; underscore spelling |
| location | Golgi (1) | mapped | Golgi Apparatus (GO:0005794); short form |
| location | mitochondria (3) | mapped | Mitochondrion (GO:0005739); plural |
| location | plasma membrane (1) | mapped | Cell Membrane (GO:0005886 = the GO term "plasma membrane") |
| location | extracellular (9) | mapped | Extracellular Space (GO:0005615); PathBank's name for the compartment |
| location | apical membrane (2) | mapped | Apical cell membrane (GO:0016324); REVIEWER TO CONFIRM (apical plasma membrane of a polarised cell) |
| location | postsynaptic membrane (1) | mapped | post-synaptic membrane (GO:0045211); hyphen only |
| cell type | neurons (1) | mapped | Neuron (CL:0000540); plural |
| cell type | astrocytes (1) | mapped | Astrocyte (CL:0000127); plural |
| cell type | Leydig cells (1) | mapped | Leydig Cell (CL:0000178); plural |
| cell type | Type III taste cell (1) | mapped | type III taste bud cell (CL:0002285); REVIEWER TO CONFIRM |
| tissue | taste buds (1) | mapped | Taste Bud (BTO:0000989); plural |
| location | apical (1) | unmapped | no membrane stated; ambiguous |
| location | cytoplasmic leaflet of the IM, periplasmic leaflet of the IM (1 each) | unmapped | Inner Membrane is the whole membrane, a leaflet is a part |
| location | etioplast (1) | unmapped | Plastid is the wider class |
| location | microsomal fraction (1) | unmapped | a prep fraction, not an organelle row |
| location | saliva (2) | unmapped | a body fluid, no location row |
| location | stroma (1) | unmapped | chloroplast stroma; Chloroplast is the whole organelle |
| location | synaptic cleft (1), synaptic vesicle (2), synaptic vesicles (1) | unmapped | Synapse is wider |
| location | thylakoid membrane (1) | unmapped | no row |
| cell type | Type II cell (1) | unmapped | type II taste cell or type II alveolar cell (both rows exist), no context |
| cell type | Type III cell (1), Type III cells (1) | unmapped | taste cell is likely but only from context; name alone is not enough |
| cell type | erythroid (1), erythroid cells (1) | unmapped | includes progenitors; Erythrocyte is the mature cell |
| cell type | geniculate ganglion neuron (1) | unmapped | a subtype of Neuron, not the same |
| cell type | MG-63 (3), Saos-2 (3) | unmapped | osteosarcoma cell lines, no row |
| tissue | bone marrow (1) | unmapped | Bone is a different tissue |
| tissue | central nervous system (1) | unmapped | Nervous System is wider |
| tissue | cotyledon, flower, hypocotyl, petal, root (1 each) | unmapped | no row |
| tissue | young leaf (1) | unmapped | Leaf is wider (developmental stage) |

## Counts (40 distinct names)
| outcome | names | of which occurrences |
|---|---|---|
| mapped | 13 (8 locations, 4 cell types, 1 tissue) | 24 |
| new_row (id already in data) | 0 | 0 |
| unmapped, unchanged | 27 | 33 |
| exact | 0 (by construction: the 40 are the names with no row) | |

No ontology id is in the data for any of the 40 names (they were written with `<ontology-id nil>`). So item 2 of the brief found nothing to write for them. The only name with an id is `cell` (GO:0005623, `ir.py:367`), which is not among the 40 and is `new_row`: PathWhiz has no `cell` row, so the importer creates it from the id already written (unchanged behaviour, now counted). No id is invented anywhere.

Effect on the 28 files: 15 have all their unmatched names mapped; 13 still contain at least one unmapped name (the plant tissues, cell lines, synaptic locations, taste Type II/III cells, etc.). Those 27 names need a product decision (create a row with an id the product owner supplies, or drop the location). Message counts (110) were not re-derived; the numbers above are names and occurrences, not import messages.

## Proposed decision text, P-37 (draft)
P-37. Vocabulary names for locations, cell types and tissues. Under T2PW_FINAL_PAYLOAD the PWML writer replaces a location, cell-type or tissue name by the name of an existing PathWhiz row only through a reviewed synonym table (`src/t2pw/pwml/vocabulary_map.py`), and only when it is the same entity (spelling, plural, abbreviation or the PathWhiz name for the same GO/CL/BTO term). The row's name is written without an ontology id so the importer finds the row. A name with no row and no synonym is written unchanged (with its ontology id when the data already has one) and counted in `layout_debug_counts` (`vocabulary_*`). No ontology id is taken from memory. Near matches (a wider class, a part, a subtype, a different organ) are never mapped. New entries need a reviewer to confirm the biology; rows for the 27 unmapped names are a separate product decision (create with an id supplied by the owner, or drop). The table is a snapshot of production (77/43/53 rows) and is re-extracted when production gains rows.

## Unverified
- Nothing imported: the mapped output was not run through the PathWhiz importer (harness not started). The expectation that a name-only item finds the row rests on parser :669-:680 (`find_by(attribute_hash)` with blank fields dropped) and on the 89 existing location hits.
- Two items may land on one row in one file (e.g. `ER` and `Endoplasmic Reticulum` both in a file): they stay two PWML items with the same name and different ids; the parser maps both ids to the same row (store_object `hash[id] = existing.id`). Not exercised.
- The two REVIEWER TO CONFIRM rows (apical membrane, Type III taste cell) rely on GO/CL naming from memory of the term definitions, not looked up here.
- Re-derived the 40 names from `s4_residual_ontology_misses.json`; the PathBank dump itself was not re-read.
- I ran one `python3 -` with empty stdin by mistake (an interactive prompt, no work done), against the rules.
