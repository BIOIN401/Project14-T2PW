# VIS-9: compounds PathWhiz draws with no structure picture

PO ruling (2026-11-26): "can we fix this somehow?" -> yes. Target: OUR patched PathWhiz importer (`t2pw/latest`); the PO
fixes production PathWhiz. Branch `vis/structures`, worktree `C:/t/vis-pw9`, cut from webserver/refactor `b9534f68`.
Working files `C:/t/vis-pw9-work/` (replays, content checks, PathWhiz runs, `compare.html`). Evidence
`docs/webserver/evidence/VIS-9/`.

## 1. Investigation (read-only)

### 1.1 How PathWhiz picks a compound's picture: only by `pwc_id`

- Viewer: every compound box `<use>`s `#<pwc_id>_moldb_cache`, an `<image>` of
  `http://moldb.wishartlab.com/molecules/<pwc_id>/image.png`
  (`app/assets/javascripts/backbone/templates/svg_visualization/image.jst.eco`, `compound_locations/default.jst.eco`,
  `edit_view.js.coffee:541`). The compound page and `_compound.html.erb` use the same URL. Nothing in the viewer or the
  app draws from SMILES, InChI or a stored structure: `moldb_smiles` / `moldb_inchi` / ... are columns of `compounds`
  (aliases `smiles`, `inchi`, ...), shown as text on the compound page and written to SBML/BioPAX, never drawn.
  `has_structure resource: 'pathbank', id_field: :to_moldb_id` (= `pwc_id`): MolDB keys PathBank structures by pwc_id.
- Patched importer (`lib/pwml_parser.rb`, `store_object`): the attribute hash is every `compounds` column the PWML
  carries, so a `<pwc-id>` and any `<moldb-*>` values are stored as given; lookup is `find_by(hmdb_id:)` (our patch) then
  `find_by(full hash)`; a new compound with no `pwc-id` gets `Compound#set_pathwhiz_id` = `PW_C` + (last row id + 1).
  Nothing in the importer does anything with `moldb-*` beyond storing them.
- So **writing SMILES / InChI cannot make PathWhiz draw a picture**. The only lever is a `pwc-id` that is the compound's
  real PathBank pwc_id. That is also why PathBank's PW000779 draws correct pictures in our dev stack (every compound
  carries its production `pwc-id`) while ours draw production compounds 2, 3, 5, ... (VIS-7 section 1).

### 1.2 What we already hold per compound

Measured on the 9 showcase IRs (139 compound records) and the 97-PWML corpus IRs (691), VIS-7 replays:

| | 9 showcases | corpus |
|---|---:|---:|
| compound records | 139 | 691 |
| with a mapped PathBank compound id (`pathbank_compound_id`, the `<id>` we already write) | 92 | 499 |
| of those, no HMDB id | 39 | 166 |
| with a held real `pwc_id` (db_row, trusted, already written today) | 6 written | 19 records |
| with no PathBank id | 47 | 192 |
| no PathBank id but a ChEBI id | 7 | 42 |
| SMILES / InChI / InChIKey in payload, IR or caches | 0 | 0 |

- Most mapped compounds are `db_status: legacy_id_unverified` (the PathBank id came from the mapping stage; the PWML side
  did not re-query it), so `_trusted_compound_pwc_id` writes no `pwc-id` for them and they hold no pwc_id value.
- PathBank assigns `pwc_id = PW_C%06d(id)` at creation. Read-only SELECT on PathBank `compounds` (repo helper
  `PathBankDbResolver`, `evidence/VIS-9/db/`): 276,345 of 276,783 rows (99.84 %) follow it; 438 do not; for **12** of
  those 438 the derived id is another compound's pwc_id (a wrong picture), for the other 426 it is unused (no picture).
  Of the 224 PathBank ids our showcase + corpus IRs carry, 222 follow it; the 2 that do not (Ceramide 247818,
  Phosphatidylcholine 109386) derive to an unused pwc_id. The reference PWMLs agree: 372 of 374 compounds.
- Structure fields: PathBank's `compounds` rows hold `moldb_smiles` for 212 of our 224 ids, but our payload carries none
  (the mapping SELECT does not read them; `final_payload` `smiles` / `inchi` keys are present and empty). Filling them
  needs a new lookup (open question 2), and would not draw anything in PathWhiz anyway.
- Caches: `data/enrichment_cache.json`, `data/id_mapping_cache.json`, `data/pathwhiz_id_db.json` hold no pwc_id
  (0 `PW_C` strings). Our preview renderer's PubChem picture cache is pictures, not structure records.

### 1.3 How PathBank's reference PWMLs encode structure

Every compound record carries `pwc-id` (`PW_C001420` for Water, id 1420) and the full `moldb-*` block from MolDB:
`moldb-smiles`, `moldb-formula`, `moldb-inchi`, `moldb-inchikey`, `moldb-iupac`, average / mono mass, logS, donor count,
traditional IUPAC, formal / physiological charge. Ours writes the same tags empty and (until now) `pwc-id` only for
6 of 139 showcase compounds.

### 1.4 Plan that followed from it

Write the `pwc-id` PathBank would write, for compounds we already write under their mapped PathBank compound id, with a
name guard. No new lookups, no change to which compounds or ids are written; the six existing `pwc-id`s untouched.
