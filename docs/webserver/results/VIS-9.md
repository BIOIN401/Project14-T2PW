# VIS-9: compounds PathWhiz draws with no structure picture

PO ruling (2026-11-26): "can we fix this somehow?" -> yes. Target: OUR patched PathWhiz importer (`t2pw/latest`); the PO
fixes production PathWhiz. Branch `vis/structures`, worktree `C:/t/vis-pw9`, cut from webserver/refactor `b9534f68`.
Code `2c760e27` (`src/t2pw/pwml/writer.py`, `tests/test_pwml_vis9_pwc_id.py`). Working files `C:/t/vis-pw9-work/` (replays, content checks, PathWhiz runs, `compare.html`). Evidence
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

## 2. Change (code commit `2c760e27`)

`src/t2pw/pwml/writer.py`, compound records only, plus `tests/test_pwml_vis9_pwc_id.py` (new, 4 tests).

- Unchanged first: `_trusted_compound_pwc_id` (db_row / trusted pwc_id) still decides, so the pwc-ids written today stay.
- New fallback `_mapped_compound_pwc_id(record, db_row, rid)`: when the compound is written under its mapped PathBank
  compound id (`pw_compound_id` / `pathbank_compound_id` == the `<id>` written), write `pwc-id` = `PW_C%06d(id)`, only if
  the written name agrees with a **full** PathBank name held in the record for that id (db_row name, mapping
  `resolved_name`, mapping candidate `name`, db_match candidate `name`). Agreement: casefold, drop a leading `L-` / `D-` /
  `(R)-` / `(S)-`, read a trailing "ic acid" as "ate", drop punctuation and spaces. Short names never count.
- Skipped: 12 PathBank ids whose derived pwc_id belongs to a different compound, PathBank row 40747 (its MolDB structure is
  the CoA thioester, section 4), and a pwc-id already written earlier in the same document (two records under one id:
  the second would fail PathWhiz's `pwc_id` uniqueness and be dropped).
- No new lookups at export. Nothing under `pipeline/`, `mapping/`, `rag/`, `curation/`, `llm/prompts/`; `layout.py`
  untouched. No `moldb-*` value written (we hold none, and PathWhiz draws none, section 1.1).

### Why the guard is names, and why short names are out

Evaluated against the real PathBank rows (read-only SELECT, `evidence/VIS-9/db/pb_rows.json`), on the 9 showcases. A
first version that also accepted PathBank short names passed five wrong ones, all existing mapping errors in the `<id>`
we already write: `Mg2` -> MG(16:0/0:0/0:0) (a monoacylglycerol, for Mg2+), `UMP` -> a UDP-MurNAc pentapeptide, `MK` ->
Melleolide K (for menaquinone), `LPS` -> LPS with O-antigen, `(p)ppGpp` -> pppGpp; and `5-HT` -> serotonin (right).
Full names only: those are refused. A second tier (PathBank row found by the same KEGG / ChEBI id we write) added 2
(phosphopantothenate, right; "cardiolipin" -> one specific CL species, a representative at best) and was dropped.

## 3. Showcase coverage and the eye check (offline)

Guard applied to the 9 showcase IRs (`scripts/evalguard2.py`, short names removed; T2 counted there but not in the
writer): of 139 compound records, 92 carry a mapped PathBank id; 67 pass the name guard (6 of them already had a pwc-id),
minus 40747 and one repeat (ceramide's second L-serine record) -> **59 records gain a pwc-id** (39 with an HMDB id, 20 without; 55 distinct PathBank
ids). 25 records with a PathBank id are refused by the name guard (names disagree: abbreviations like `UDP-Glc`, `Kdo`,
`CoA-SH`, `Pchlide`, and the wrong ids in open question 4); 47 have no PathBank id. Corpus (691 records): 370 gain a pwc-id (109 without HMDB), 13 already had one, 113 refused by name, 192 have no PathBank id. Exact mirror of the writer decision: `scripts/nohmdb9.py`.

Eye check: the MolDB picture of all 55 distinct pwc-ids, fetched by pwc-id (only the public id left the machine), laid
out with our compound name (`evidence/VIS-9/moldb/sheet_1..3.png`). 50 right (AMP, beta-alanine, L-serine, lactate,
pantothenate, the porphyrins, sphingoid bases, ATP, NAD(H), CoA and its esters, chorismate, isochorismate, SEPHCHC,
lipid A, lipid IVA, chlorophyll a / b, enterobactin, pyruvate, aspartate, menaquinone, ...; class compounds such as
"fatty acid", "glucosylceramide", "dihydroceramide" draw PathBank's representative); 4 "No Structure Available"
(pantethine 118678, sphingomyelin 247821 have no MolDB structure; ceramide 247818 derives to an unused pwc_id; that is no
picture, as before); **1 wrong**: 40747 "1,4-dihydroxy-2-naphthoic acid" draws 1,4-dihydroxy-2-naphthoyl-CoA, because
PathBank's own row carries the CoA ester's structure (same InChIKey as 40746, formula C32H38N7O19P3S). Fixed by the skip
list. After the fix: 54 distinct ids, 50 correct pictures, 4 no picture, 0 wrong (on this check).

Before / after, compounds with a correct picture in our patched importer on a clean DB (the dev stack): before 6 of 139 (the six pwc-ids already written);
every other compound gets a dev-DB pwc_id and MolDB draws production's compound with that number (VIS-7 section 1); after:
the 6 existing + 59 new pwc-id records, of which all but the 4 "No Structure" ids draw their own structure.

## 4. Verification (2026-11-27, after the reboot; HEAD `99364bf1`, clean tree)

One queued heavy-lock chain `vis9-chain1` (`scripts/chain1.sh`): **BW_EXIT=0, FINAL SURVIVING COUNT 0**; then
`vis9-pinab` (`scripts/pinab.sh`): **BW_EXIT=0, 0 survivors**. Logs `evidence/VIS-9/gates/`. (The night run never
started: under 1 GB available until the reboot; see section 5.)

### 4.1 Content check: only `<pwc-id>` additions

Base = replays through `git archive b9534f68` (webserver/refactor), head = this tree; same payloads, no network.

| check | 9 showcases | 97-PWML corpus |
|---|---|---|
| `cc9.py` strict: canonical XML equal after removing only the `<pwc-id>` head added to a compound that had none; existing pwc-ids unchanged | **0 different**, 59 pwc-ids added | **0 different**, 370 added (86 files) |
| VIS-7 `content_check.py` rule-based / strict (masks no pwc-id) | 9 / 9 differ, every diff line an added `<pwc-id>` | 86 / 86 differ, every diff line an added `<pwc-id>` |
| text diff with `<pwc-id>` lines dropped | 0 files | 0 files |

The added counts equal the offline mirror of the writer (section 3): 59 and 370.

### 4.2 Tests

| gate | result |
|---|---|
| focused: `tests/test_pwml_*.py` (incl. the 4 new) | 289 passed |
| `tests/test_render_*.py` / `test_final_payload_writer_fixes.py` | 49 / 22 passed |
| `tests/test_orch768_replay_corpus.py` | **1 failed**, 8 passed: `pwml_matches_leg` for the sour-taste leg |
| SMOKE | **511 passed** |
| golden seams batch / c011 / stage8 / quarantine (unedited) | 2 / 8 / 4 / 24 passed |
| equivalence run_pathway / app-vs-service / default path | 8 / **3 failed, 11 passed** / 6 passed |
| vision / web | 370 passed, 1 skipped / 85 passed |

The 4 failures are byte pins of recorded sour-taste PWMLs (`pwml_sha256` / `pwml_matches_leg`); our change adds
`<pwc-id>PW_C040034</pwc-id>` to its "Hydrogen Ion" (PathBank 40034, name agrees). A/B in one hold, same tree, with the
VIS-9 fallback switched off by a test-only plugin (`evidence/VIS-9/scripts/vis9_off.py`, `-p vis9_off`): orch768
**9 passed**, app-vs-service **14 passed**; switched on again: orch768 1 failed. So the VIS-9 pwc-id is the only cause.
I did not edit the pinned tests (open question 7).

### 4.3 PathWhiz, our patched importer (`t2pw/latest`), clean dev DB per file

`scripts/pwab9.sh h1 p1`: 9 showcases + PathBank PW000779, project `t2pw-pwdraw`, stack down afterwards (`STACK_DOWN
rc=0`). **10/10 imported, 10/10 screenshots. Error lines identical to VIS-7's a3/a3b for every file (`NEW_ERROR_LINES 0`),
created record counts identical** (`evidence/VIS-9/pathwhiz/pwab_vs_vis7.txt`).

Per compound record of our PWML, read against PathWhiz's own export of the imported pathway (`scripts/pics.py`;
"right picture" = PathWhiz holds our pwc-id and MolDB has a structure for it, all 54 checked by eye in section 3):

| pathway | compounds | VIS-7 right picture | VIS-9 right picture (no HMDB) | VIS-9 own pwc-id, MolDB has no structure | VIS-9 dev-DB pwc-id (no right picture) (no HMDB) | merged into a same-HMDB record | absent from PathWhiz's export (VIS-7 too) | errors VIS-7 -> VIS-9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ceramide-sphingolipid | 13 | 2 | 10 (1) | 2 | 0 (0) | 1 | 0 | 1 -> 1 |
| chlorophyll | 25 | 0 | 10 (2) | 0 | 15 (14) | 0 | 0 | 15 -> 15 |
| coenzyme-a | 15 | 0 | 8 (3) | 1 | 5 (3) | 1 | 0 | 0 -> 0 |
| ent-item6 | 13 | 0 | 10 (4) | 0 | 3 (3) | 0 | 0 | 1 -> 1 |
| enterobactin | 12 | 0 | 9 (3) | 0 | 3 (3) | 0 | 0 | 0 -> 0 |
| furochromone | 13 | 1 | 3 (1) | 0 | 10 (9) | 0 | 0 | 0 -> 0 |
| lipid-a | 32 | 1 | 4 (1) | 0 | 28 (24) | 0 | 0 | 22 -> 22 |
| menaquinone | 13 | 2 | 5 (4) | 0 | 6 (6) | 0 | 2 | 0 -> 0 |
| sour-taste | 3 | 0 | 1 (0) | 0 | 2 (1) | 0 | 0 | 7 -> 7 |
| **total** | 139 | 6 | 60 (19) | 3 | 72 (63) | 2 | 2 | |

**Key question: compounds with a pwc-id and no HMDB id import cleanly and draw.** All 22 such records (20 new, 2 already
had one) are in PathWhiz with our pwc-id; none dropped, no new error: 19 draw the right structure, 2 (ceramide,
sphingomyelin) draw "No Structure Available" (MolDB has none), 1 (menaquinone) is absent from PathWhiz's export in VIS-7
too (as is "demethylmenaquinone", which has no pwc-id: an existing effect, not VIS-9). Right pictures: **6 -> 60 of 139**.
The 72 left with a dev-DB pwc-id are the 47 without a PathBank id and the 25 the name guard refused. Two "merged" records
are second records of a compound already present under the same HMDB id (ceramide "serine" -> "L-serine", coenzyme A ->
"CoA-SH"); the coenzyme A box therefore draws CoA-SH's dev-DB picture, because "CoA-SH" (abbreviation, refused) comes
first in the file. Screenshots: `evidence/VIS-9/pathwhiz/*_zoom.png`; all ten in `C:/t/vis-pw9-work/pw/p1/screenshots/`.

`C:/t/vis-pw9-work/compare.html`: per pathway VIS-7's PWML in PathWhiz next to VIS-9's, the names that gained a pwc-id,
and the MolDB contact sheets.

## 5. Dead ends and slips

- Writing SMILES / InChI, the brief's first guess: dead end, PathWhiz draws only by pwc_id (section 1.1).
- Short-name agreement passed five wrong mappings (section 2); KEGG/ChEBI agreement tier added one doubtful case; both dropped.
- Deriving the pwc_id without a check would have drawn a different compound for 12 PathBank rows; and PathBank's own
  structure for row 40747 is wrong (both in the skip list).
- Order slip: the brief asked for the write-up before coding; the guard code was drafted in the worktree while the
  evaluation was still running, then the write-up (this file, sections 1-3) was committed before any run.
- Slip: one read-only `grep -r` over `/` was started with a trailing `&` (not a run_in_background job); it ended with
  its shell, no process left.
- Night of 2026-11-26: the first verification chain waited 3 hours in the RAM gate (948 -> 136 MB available) and was stopped and its waiting shells killed by PID; it ran after the reboot.
- `cc9.py` first version reported 8/9 and 83/97 "different": it dropped the removed `<pwc-id>`'s tail (the element is the last child), a whitespace artefact; fixed (the previous sibling takes its tail), text diff confirms.
- Python `-I` ignores `PYTHONPATH`; the DB measurement scripts insert the main repo `src` on `sys.path` (read-only
  SELECTs through `PathBankDbResolver`, connection settings from the main checkout's `.env`, never read or printed).

## 8. Open questions (product owner)

1. **Production importer must match a compound by `pwc_id`.** In our patched importer on a clean DB (the PO's target) the
   new `pwc-id`s draw the right pictures. Against the production DB, a compound WITH an HMDB id is found by HMDB (our
   `find_by_unique_external_id` patch) and the pwc-id does not matter. A compound WITHOUT an HMDB id that now carries a
   pwc-id is not found by `find_by(full hash)` (our name / description differ), so the importer tries to create it with
   a pwc_id production already has: `validates :pwc_id, uniqueness` fails and the compound is dropped with an error,
   where today it would be created as a new, picture-less duplicate (or already fail on `name` uniqueness when our name
   is exactly PathBank's). Options: (a) **one line in PathWhiz** `find_by_unique_external_id`:
   `elsif klass == Compound && attribute_hash[:pwc_id].present? then Compound.find_by(pwc_id: ...)` (after the HMDB
   branch): the compound then maps to PathBank's own row, with its structure. Small, PO's repo; recommended. (b) Until
   (a) ships, gate the new pwc-ids to compounds that also carry an HMDB id (an env flag or a one-line condition in
   `_mapped_compound_pwc_id`): no production risk, but the no-HMDB compounds this task is about stay without a picture.
   (c) Leave as is: right in our importer, a drop risk in production for the 20 showcase (109 corpus) new pwc-ids on compounds without an HMDB id (section 3).
2. **Structure fields (`moldb-smiles`, InChI, InChIKey, formula).** PathWhiz draws nothing from them, but they are what
   PathBank writes and what SBML / BioPAX exports carry. PathBank's `compounds` row has them for 212 of our 224 ids; our
   payload has none. Options: (a) add `pwc_id, moldb_smiles, moldb_inchi, moldb_inchikey, moldb_formula` to the mapping
   stage's PathBank `compounds` SELECT (no new connection, a few columns; a `mapping/` change, needs a ruling) and write
   them for the same guarded compounds; it also makes the pwc-id exact (no derivation) and gives a formula guard;
   (b) a static snapshot file of id -> pwc_id / InChIKey / formula for the ids PathBank has (about 277k rows, ~25 MB, or
   only the 438 pwc exceptions, ~15 KB) refreshed by the maintainer command that builds `data/pathwhiz_id_db.json`;
   (c) leave empty. Risk for (a)/(b): PathBank's own structures are sometimes wrong (row 40747 = CoA ester; Ceramide
   row 247818 = glycerol `OCC(O)CO`), so a formula / InChIKey cross-check against ChEBI is worth having before writing.
3. **Compounds with no PathBank id** (47 of 139 on the showcases, 192 of 691 in the corpus) get no picture: they have
   no pwc_id anywhere. 7 (showcases) / 42 (corpus) carry a ChEBI id. A picture needs either a PathBank row found by ChEBI
   / KEGG at mapping time (a mapping change; many were rejected by the name gate on purpose) or a MolDB structure pushed
   for a new compound (production PathWhiz, PO). No action proposed for the dev stack.
4. **Wrong `<id>`s the guard exposed (existing content, not changed here):** `Mg2` (chlorophyll) -> PathBank 58034
   MG(16:0/0:0/0:0); `UMP` (lipid A) -> 41234 UDP-MurNAc-pentapeptide; `MK` (menaquinone) -> 15980 Melleolide K;
   `noreugenin` (furochromone) -> 21430 Isobiflorin (fuzzy 0.35, ambiguous, still written as the id). These are mapping
   errors in what we already write; the PWML says, for example, that menaquinone is Melleolide K. Worth a mapping task
   (short-name and fuzzy matches should not become a written PathBank id). Our PubChem CIDs disagree with PathBank's for 2
   of 66 compared (`MK` 13418 vs 9867602, `lactate` 4615 vs 107689).
5. **PathBank data errors** to report upstream: row 40747 "1,4-dihydroxy-2-naphthoate" has the structure of 40746
   (its CoA ester); row 247818 "Ceramide" has glycerol, and its pwc_id is PW_C247816, not PW_C247818.
6. **(closed) Verification**: run on 2026-11-27 after the reboot, section 4.
7. **Four recorded-byte pins fail by design (section 4.2).** `test_orch768_replay_corpus` (sour-taste leg,
   `pwml_matches_leg`) and three `test_equivalence_app_vs_service` tests (`pwml_sha256` of the recorded sour-taste live
   run) compare our PWML bytes with PWMLs recorded before VIS-9; the one added `<pwc-id>PW_C040034</pwc-id>` (Hydrogen
   Ion) breaks them, and nothing else does (A/B with the fallback off: 9 / 14 passed). The tests say a lane that
   changes a verdict on purpose moves the pin with its documented delta (merge rule 4); gate 5 treats an edited
   equivalence test as a reject unless the reviewer accepts it. Options: (a) **compare after removing the `<pwc-id>`
   elements the head adds** (a small helper in both tests, like `cc9.py`: keeps byte fidelity on everything else;
   recommended, small, needs reviewer sign-off on a test edit); (b) re-record the sour-taste fixture and leg hash (loses
   the link to the recorded live run, medium); (c) pin the old bytes with an env switch in the writer, as D-W23 did for
   layout v2 (adds a production flag only tests use, small, not recommended); (d) don't merge VIS-9. I did not edit them.
