# PathWhiz import errors: triage of the 60 T2PW files (2026-11-11)

Branch `pwml/final-payload`. Read-only analysis. No import, Docker, test or pipeline run. Scripts and outputs: `evidence/import_triage/` (s1 to s15, run with the project venv).

## 0. NOT established (read first)

1. **Nothing was re-imported.** Every verdict comes from the recorded harness results (`t2pw_results.jsonl`), a scan of the 60 PWML files, and reading the importer and writer code. The transport state-id finding (cause 8) is read from code and fits all 37 messages, but no run confirmed it.
2. **56 of 60 files are older than the current writer.** Their run folders are dated before ORCH-747 (2026-10-19) or are the E2E-5 file. I read the writer at HEAD to see whether each defect still exists; I did not re-export. Two stale defects (70 free-text interactions, one empty complex) are already refused at HEAD.
3. **"Production" means the May 2026 dump copy** (`smpdb/db_dumps/smpdb_remote_dump.sql`, read with `grep`, 10 s, never loaded). Not the live database. I assumed MySQL's case-insensitive name compare (default collation); not checked on the real server.
4. **The importer copy is the owner's uncommitted version** (`find_by_unique_external_id`, parser lines 629-640). If production runs the committed parser, species and compound lookups by name only.
5. **Not traced:** `compound: Pwc has already been taken` (1 file, 1 message), the 3 `reaction_compound_visualization` / `compound_location` leftovers, why the 3 transports have left state equal to right state (see cause 8).
6. **"Files clean after fix" numbers (section 3) are a cover of the recorded error keys.** They assume each fix removes its whole group. They are not a re-import.
7. The vocabulary check compares names only. It ignores the tissue template columns and cell-type synonyms.

## 1. Where each message comes from

All messages reach `errors` through five writes in `lib/pwml_parser.rb`: `store_object` :704, `store_nested_object` :776, `store_nested_visualization` :900, `get_child_object` :939, `build_nested_visualization_recursive` :1021. Each stores `object.errors.full_messages`. The text itself comes from the model validator named below. Exceptions inside these methods are swallowed (`puts`, `return`), so a dropped element can leave no message at all.

## 2. Verdict per cause

Paths are under `C:/t/pwi/smpdb/app/models/` (models) or `lib/` (parser). "Files" is the harness count.

| # | Cause (files / messages) | Emitted by | Trigger | Class | Fix |
|---|---|---|---|---|---|
| 1 | `bound_visualization: Bound can't be blank` (60 / 168) | `visualization/bound_visualizations/bound_visualization.rb:8` `validates :bound_id, presence`; parser :900 | T2PW writes one `<bound-visualization>` per biological state with `x y width height zindex hidden` and no `<bound-id>`. The table has only `bound_id, biological_state_id, pathway_visualization_id` (`db/schema.rb:102`). A PathWhiz bound visualization draws a Bound (a bound complex). T2PW uses it as a compartment rectangle. | **T2PW writer defect** | Section 4.1. 3 lines. |
| 2 | Ontology can't be blank, on location / cell type / tissue (53 / 151) | `data/biological_states/subcellular_location.rb:11`, `cell_type.rb:11`, `tissue.rb:12`; parser :704 | The writer emits `<ontology-id nil="true"/>` (`writer.py:985,1025,1038`). The importer drops blank fields, then looks up `find_by(name: ...)` (parser :669). A hit needs no ontology id. A miss tries to create a row, which needs one. | **Mostly empty-DB artefact; a real remainder** | 89 of 121 location names, 4 of 13 tissue names and 1 of 17 cell-type names already exist in production (case-insensitive). **28 of 60 files still contain at least one name that does not exist** (40 distinct names, e.g. `plasma membrane`, `ER`, `extracellular`, `neurons`, `MG-63`). Those need a vocabulary map (4.4). Seed fix below. |
| 3 | Interaction rejected (27 / 80) | `data/interactions/interaction.rb:29-32` (left and right `length is 1`, type in `Activation Inhibition`); parser :776 | See section 5. 70 of 85 interactions are stale free-text types with no elements. 15 are valid; 10 of those die from cascades. | **Stale input (70), empty-DB artefact (3), content gap (7)** | Section 5. No writer change needed for the 70. |
| 4 | Reaction: `Spontaneous cannot be No if reaction has no enzymes` (27 / 88) | `data/reactions/reaction.rb:90-95` `check_spontaneous`, validated at :36 | Writer emits `spontaneous=false` for every reaction (`writer.py:1375`; 245 of 245 reactions). 67 reactions lost their enzyme on import (cause 5). 21 have no enzyme in the PWML at all. | **T2PW writer defect** (value), plus content gap (21 with no enzyme) | Section 4.2. 1 line. |
| 5 | Complex without proteins (21 / 65) and species blank (20 / 20) | `data/protein_complexes/protein_complex.rb:47`; `data/protein_complexes/protein.rb:47`; `data/biological_states/species.rb:15` | In 20 files the one protein that fails is the PathBank "Unknown" protein (id 9659) whose species is Arabidopsis thaliana, taxonomy 3702, written **with empty `<classification/>`**. Species cannot be created, so the protein has no species, so the 1-member complexes holding it are empty (64 complexes), so 92 reaction enzymes, protein locations and complex visualizations fail. The 21st file has a 0-member complex `CLPXP`. | **Empty-DB artefact** (20 files); stale input (1 file) | In production both exist: species row 4 (3702, Eukaryote, "Thale cress") is found by `find_by_unique_external_id`; protein 9659 (`Unknown`, uniprot `Unknown`, species 4) matches on name, uniprot id and species. Hardening: 4.3. |
| 6 | `species: Classification can't be blank` (20 / 20) | `species.rb:15` | Same 20 Arabidopsis rows. Source: `map_ids.py:9594-9599` sets name and taxonomy id for the sentinel species and nothing else. | **Empty-DB artefact; also a small writer/mapper defect** (row is incomplete) | 4.3. |
| 7 | Nucleic acid `Nucleotide type can't be blank` (1 / 12) | `data/nucleic_acid.rb:35` (`RNA` or `DNA`) | Gene-symbol nucleic acids (e.g. `ALAD1/HEMB1`) are written with `<nucleotide-type/>`. | **Content gap** (the payload does not say DNA or RNA) | PO decision. 4.5. |
| 8 | Transports: all 37 die (19 files) | `data/transports/transport_element.rb:15-16`, `transport.rb:30`; parser `add_content` :585 | Section 6. | **PathWhiz importer bug** (first) then **writer defects** | Section 6. |
| 9 | Cascades, no cause of their own | `protein_location`, `protein_complex_visualization`, `protein_complex_protein_visualization`, `reaction_visualization`, `reaction_enzyme*`, `*_visualization` | Fall out of 2, 4, 5, 8. | n/a | none |
| 10 | `transport_compound_visualization: Compound location can't be blank` (14 / 38) | `visualization/transports/transport_compound_visualization.rb:13`; parser :1021 | The `compound-location-id` appears nowhere else in the file (38 of 38 dangling; script s10, s11). | **T2PW writer defect** | Section 6, item B. |

Row 5 note: the 20 files and 20 proteins match one to one (script s6: 20 of 20 failing proteins use the blank-class species; 60 of 60 files agree).

### Control contrast fragments (real PathBank, `reference/PW012926.pwml`)

* Bounds: 0 `<bound-visualization>` in all 5 PathBank files. Compartments are drawn with `<membrane-visualization>` (4 in PW012926), `<drawable-element-location>` and `<label-location>`. T2PW emits none of these.
* Ontology: `<subcellular-location>` `Cytoplasm` `<ontology-id>GO:0005737</ontology-id>`; `<cell-type>` `Cell` `CL:0000000`; `<tissue>` `Liver` `BTO:0000759`.
* Species: `<classification>Eukaryote</classification>` on every species row.
* Reactions: `<spontaneous type="boolean" nil="true"/>` for the 5 enzyme-catalysed reactions, `true` for the 1 spontaneous one. PathBank never writes `false`.
* Transporters: `<transport-transporter>` carries `<protein-complex-id>`, never `<protein-id>`.

## 3. Ranking: files fixed per unit of work

"Clean files" is the count of the 60 files with zero recorded errors once the listed groups are removed (cover of the error keys, caveat 6). It is cumulative, in this order.

| Rank | Change | Where | Size | Files affected | Clean files (cumulative) |
|---|---|---|---|---|---|
| 1 | Do not write `<bound-visualization>` (4.1) | `writer.py` | 3 lines | 60 | 0 |
| 2 | Write `spontaneous` as nil unless true (4.2) | `writer.py:1375` | 1 line | 27 | 0 |
| 3 | Seed the harness with vocabulary and protein 9659 (4.6) | harness `seed/` | 1 grep, no code | 53 + 22 | **24** |
| 4 | Re-export the 56 stale files with HEAD (no code) | run | one export pass | 25 | **38** |
| 5 | Importer: remap `left-/right-biological-state-id` (6A) | `pwml_parser.rb:585` | 2 branches, owner's file | 19 | 39 |
| 6 | Writer: keep the base compound location a transport uses (6B) | `writer.py:1595-1601` | ~5 lines | 14 | 45 |
| 7 | Transporters: complex wrapper or withhold (6C) | Stage 6 or `writer.py:1433` | medium | 12 | 57 |
| 8 | Nucleic acid type, empty complex, duplicate Pwc | PO / stale | small | 3 | 60 |

Ranks 1 and 2 clean no file alone, because every file has several causes. They remove 168 and 176 messages. After rank 3, ontology still fails in 28 files in production (cause 2), so the "24" holds in the harness, not in production. Production-faithful count needs rank 3 plus the vocabulary map (4.4).

## 4. Writer and seed fixes

### 4.1 Bound visualizations (cause 1)
`writer.py:1509-1523` builds `self.section_items["bound-visualizations"]` from `ir["bound_visualizations"]` (built at `ir.py:2012-2024`). The same rows are read at `writer.py:1750-1807` as layout regions (one band per biological state) and are the only use. Keep the IR rows and the region maths. After line 1807, set `self.section_items["bound-visualizations"] = []` so the element is not written. `validate.py:36` and the writer tag table (`writer.py:384`) accept an empty section. `tests/test_pwml_writer.py:344` already uses an empty list. Compartment drawing is a separate job (membrane and drawable-element-location, section 2 contrast).

### 4.2 Spontaneous (cause 4)
`writer.py:1375` is `bool(reaction.get("spontaneous", False))`. Replace with: `True` when the reaction says spontaneous is true, otherwise `None` (written as `nil="true"`, as PathBank does). `None` passes `check_spontaneous` with or without enzymes. This is a serialisation change, not a biology change: "not stated" is written as unknown instead of as "enzyme-catalysed". It makes the 21 enzyme-less reactions importable. Product owner should confirm that an enzyme-less, non-spontaneous reaction is acceptable in PathWhiz.

### 4.3 Sentinel species (causes 5, 6)
Add `classification: "Eukaryote"` and `common_name: "Thale cress"` where `map_ids.py:9599` sets the sentinel row (or put them in `SPECIES_CREATE_DEFAULTS`, `ir.py:354`). The strict-db check at `ir.py:2952-2969` skips rows that carry a database id, so it never saw this gap. This does nothing in production (species 4 exists) and removes the artefact on an empty database.

### 4.4 Vocabulary names (cause 2)
Production has 77 locations, 43 cell types, 53 tissues. T2PW names such as `plasma membrane`, `ER`, `extracellular` do not match `Cell Membrane`, `Endoplasmic Reticulum Membrane`, `Extracellular Space`. Only `cell` gets an id (`ir.py:367`, `GO:0005623`). **`GO:0005623` is not in production**, so 59 of 60 files would create a new `cell` location row there. Sketch: a name-to-row table built from the three vocabulary tables, applied in `ir.py` where `ontology_id` is copied (:898), writing the PathBank name and ontology id when a synonym matches and leaving the name alone when not. Names with no match need a product decision: create a new vocabulary row (needs a GO, CL or BTO id) or drop the location. That is a policy call, so the cost is not small.

### 4.5 Nucleic acids (cause 7)
Writer cannot invent DNA or RNA. Options: PO rules a default per entity kind; or withhold such nucleic acids and the interactions that use them, as ORCH-747 does for untyped interactions. 7 interactions in one file depend on it.

### 4.6 Harness seed (all empty-DB artefacts)
Rows can be copied read-only from the dump the harness already greps. Command (about 10 s, output 0.5 MB, species is most of it):

    grep -a -E "^INSERT INTO `(cell_types|species|subcellular_locations|tissues)` " smpdb_remote_dump.sql > vocab_inserts.sql

Rows: `cell_types` 43, `subcellular_locations` 77, `tissues` 53, `species` 4016. Also add the one row `proteins` id 9659 (`grep -a -o "(9659,'Unknown','[^)]*)"`, found: name `Unknown`, uniprot `Unknown`, species 4). With these seeded, causes 5 and 6 vanish and cause 2 shrinks to the 28 real files. The extract is **not committed** here (it is the owner's data; the harness doc already asks that question, open question 4). The rake task already forces `user_id` to a local admin, so the dump's `user_id` 11 is harmless.

## 5. Interactions: 5 of 85 survive

Script s2, s3, s12.

| Group | Count | What happens | Class |
|---|---|---|---|
| Free-text type (`inhibits`, `binds`, `modifies`, `catalyzes`, ...) and zero elements | 70 (25 files) | `interaction_type` not in `Activation Inhibition`; left and right `length is 1` fail (0 elements). All 70 are in files dated before 2026-10-19. HEAD (`ir.py:2390-2460`, ORCH-747) withholds any type outside the table and lists it in a side file. | **Stale input.** Re-export. |
| Valid type, element is a complex that failed (E2E-5 file) | 3 | Complex lost to cause 5 (sentinel species). Element remap falls back to the raw id, `Element can't be blank` (`interaction_element.rb:11`). | **Empty-DB artefact** |
| Valid type, element is a nucleic acid with blank type (chlorophyll file) | 7 of 10 | Cause 7. | **Content gap** |
| Valid type, all elements fine | 5 | Created (wnt PCP 1 of 1, PMC12096016 1 of 1, chlorophyll 3 of 10). | n/a |

Interaction visualizations (10 messages) are a cascade of the lost interactions.

## 6. Transports: 0 of 37 survive

Script s5, s10, s11.

**A. PathWhiz importer bug (the one that kills all 37).** `PwmlParser#add_content` remaps `biological-state-id` (parser :491) but has no branch for `left-biological-state-id` or `right-biological-state-id`. They fall to the generic `else` (:585) and keep the raw XML id. `TransportElement` then looks up a biological state whose database id equals the XML id (`transport_element.rb:15-16`). T2PW ids run high (state 12, 13); an empty database has at most a handful of states, so both are blank (37 of 37 messages say "Left ... Right ... can't be blank"). Then `transport.rb:30` (`transport_elements length >= 1`) fails: the 37 `Transport elements is too short` messages, and the 37 `transport_visualization` messages follow. **In production this would not fail loudly.** The raw id points at some existing biological state, so the transport would import and attach to the wrong state. PathBank re-imports show the same: PW012926 transport has `left 151 / right 223` and imported only because the database had 262 states with those ids. Fix: two `elsif` branches that read `@biological_state_list`, copied from :491-496. Owner's file; one-line each. (Harness open question 1 is the same file.)

After A, what remains of the 37 (script s5):

| Residual | Transports | Class |
|---|---|---|
| none, imports | 15 | |
| transporter has `protein-id` only, so transporter dropped, transport still imports | 17 (+ 1 also in the next row) | **B2 writer/pipeline defect** |
| `left state == right state` (`transport_element.rb:18`, `biological_states_unique`) | 3 | **Writer defect or content gap** (the IR placed a transport between one state; not traced why) |
| `element-type` is `Protein` (2) | 2 | Passes validation, but `pwte_id` (`transport_element.rb:~30`) has no Protein branch and the UI expects compound, complex, nucleic acid or bound. Writer defect, minor |

So up to 34 of 37 would import after A, 17 of them without their transporter.

**B. Writer defects behind the rest.**

1. **B1, dangling compound location (14 files, 38 messages).** `writer.py:1595-1601` skips writing the base compound location when the compound is used in a reaction (`reaction_compound_entity_keys`, :1541-1553). Reactions make their own per-reaction copy (:1920-1940) and so avoid the gap. Transport visualizations (:2484) use the base id of that skipped location (`lookup("locations", ...)`), which is never written. Fix: add compounds used by a transport member to the set that keeps the base location, or give transports a per-transport copy the way reactions do. About 5 lines.
2. **B2, plain-protein transporters (19 of 19, 12 files).** `writer.py:1433` writes `<protein-id>` for a transporter that is not a complex. `transport_transporter.rb:15` needs `protein_complex_id`. `TransportTransporter` has no `protein_id` column, so the value is dropped and the row fails (`Protein complex can't be blank`, then the transporter visualization). Reaction enzymes never hit this: Stage 6 wraps every enzyme in a complex (`ir.py:1939`), and 0 of 288 enzymes use `protein-id`. Transporters are not wrapped. Fix belongs in the Stage 6 wrapper (outside this track's code boundary, a PO call) or, within the exporter, withhold the transporter and list it, as ORCH-747 does for interactions.

## 7. What was checked and how

| Check | Result | Script |
|---|---|---|
| Interaction types and element counts, 85 interactions | 70 invalid-type with 0 elements; 15 valid | s2, s3 |
| Vocabulary hit rate against the dump | see cause 2; 28 of 60 files keep a miss | s4 |
| Transport residuals | 15 clean, 17 transporter-only, 3 same-state, 2 Protein | s5 |
| Failing protein is the Arabidopsis sentinel | 20 of 20 | s6 |
| Reactions: enzyme in PWML or not | 67 enzymes lost, 21 none | s7 |
| Enzymes by complex or protein id | 288 of 288 via complex; 92 on failed complexes; 64 failed complexes have 1 member, 1 has 0 | s8 |
| Dangling `compound-location-id` | 38 of 38 | s10, s11 |
| File dates against ORCH-747 | 56 of 60 older or unknown | s12 |
| Root-cause cover | s14, s15 | |

Regenerate the extract for s4 (`python s4_vocab.py <vocab_inserts.sql>`) with the grep in 4.6.
