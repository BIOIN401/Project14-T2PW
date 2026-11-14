# Why a name has no ontology id: per-name table (P5, ontoBE)

Measured 2026-11-13 over the 152 stored runs (`p5_measure_152.py`: every run's real payload and paper text through `fill_ontology_ids`, live OLS4 + Cellosaurus). Meaning of columns: *runs* = runs of the 152 in which the name occurs without a PathWhiz row or id; *after P5* = outcome of the new code. An id-less cell type, tissue or location is kept in the final payload and omitted from the PWML only (writer commit 0c320e27, wildcard guard keeps a row when dropping it would let a state bind to another).

## A. Names without an id

| kind | name | runs | after P5 | reason (what was queried) |
|---|---|---|---|---|
| cell_type | erythroid | 1 | unresolved | no CL term has the exact label or synonym 'erythroid' (OLS4 exact 0; fuzzy closest CL:0000038 erythroid progenitor cell, CL:0001066, CL:0000050): a progenitor or lineage term is not the cell type the paper names, not accepted; Cellosaurus has no cell line of that name. |
| cell_type | erythroid cells | 1 | unresolved | no CL term has the exact label or synonym 'erythroid cells' or the singular 'erythroid cell' (OLS4 exact 0 both; fuzzy closest CL:0000038 erythroid progenitor cell, CL:0000764 erythroid lineage cell): a lineage or progenitor is not the same entity, not accepted; Cellosaurus none. |
| cell_type | geniculate ganglion neuron | 1 | unresolved | CL holds only subtype terms (CL:4033157 geniculate ganglion TRPV1 neuron, CL:4033158 ... SPARCL1 neuron, CL:4033159 ... VGLUT2 neuron) and no generic 'geniculate ganglion neuron' (OLS4 exact 0); a subtype would assert more than the paper, and PathWhiz row 'Neuron' is the parent type (deliberately not mapped); Cellosaurus none. |
| cell_type | granulosa cells | 1 | resolved by the vocabulary map | mapped to PathWhiz row 'Granulosa Cell' (CL:0000501): the singular is a row name, plural of the same cell type (new in P5). |
| cell_type | idioblast | 1 | unresolved | plant cell type absent from CL: OLS4 CL exact 0 and fuzzy 0 results; Cellosaurus none. |
| cell_type | internal phloem associated parenchyma | 1 | unresolved | descriptive plant cell-type phrase, no CL term: OLS4 CL exact 0 and fuzzy 0 results; Cellosaurus none. |
| cell_type | laticifer | 1 | unresolved | plant cell type absent from CL: OLS4 CL exact 0 and fuzzy 0 results; Cellosaurus none. |
| cell_type | MG-63 | 6 | unresolved for PathWhiz; cellosaurus_id CVCL_0426 kept in payload | cell line, not in CL (OLS4 CL exact 0). Cellosaurus has exactly one exact match, CVCL_0426 (identifier MG-63, Homo sapiens 9606, human payload species agrees), found in all 6 runs and kept in the payload as `cellosaurus_id` with `db` provenance. PathWhiz cannot take it: PathWhiz `cell_types.ontology_id` must match `CL:\d{7}` and be unique (smpdb app/models/data/biological_states/cell_type.rb line 11). Cellosaurus cross-references BTO_0001596 'MG-63 cell' (not used: a cell line is not a tissue row). |
| cell_type | Saos-2 | 6 | unresolved for PathWhiz; cellosaurus_id CVCL_0548 kept in payload | cell line, not in CL (OLS4 CL exact 0). Cellosaurus has exactly one exact match, CVCL_0548 (identifier SaOS-2, exact synonym Saos-2, Homo sapiens 9606), found in all 6 runs and kept in the payload as `cellosaurus_id` with `db` provenance. PathWhiz cannot take it: PathWhiz `cell_types.ontology_id` must match `CL:\d{7}` and be unique (smpdb app/models/data/biological_states/cell_type.rb line 11). Cellosaurus cross-references BTO_0000971 'SAOS-2 cell' (not used: a cell line is not a tissue row). |
| cell_type | Type II cell | 2 | unresolved | ambiguous by name: CL has no exact label/synonym 'Type II cell' (OLS4 exact 0 hits; fuzzy closest CL:0002063 pulmonary alveolar type 2 cell, CL:0000922 type II NK T cell, and the taste cell CL:0002286). The paper context selects one term only when the paper text is available: PMC7059786 carries the full phrase 'type II taste cell(s)' 3 times, the exact label of CL:0002286 and the only full 'type II ... cell' phrase, so the name resolves with the text (pass 2 of a_ontology_lookup_p5.md). The two stored runs holding the name (150, 151, e2e output folders) have no stored paper text and no full phrase in the payload, so it stays unresolved there. |
| cell_type | Type III cell | 4 | unresolved: id_used_by_other_row ['CL:0002285'] x2; resolved CL:0002285 (OLS4, reviewed_synonym, via paper_context: 'type iii taste cell') x1; unresolved: no_exact_match [cellosaurus: no_exact_match] x1 | CL has no exact 'Type III cell' (OLS4 exact 0; fuzzy closest CL:0002137 type III cell of adrenal cortex and CL:0002285 type III taste bud cell: ambiguous by name). The context phrase 'type III taste cell' (payload row 'Type III taste cell', or PMC7059786 text 4 times) selects CL:0002285 (PathWhiz row 'type III taste bud cell'); that phrase is NOT a CL label or synonym (OLS4 exact 0), it is the reviewed synonym `type iii taste cell` of vocabulary_map.py (marked 'reviewer to confirm'). Outcome in the 4 runs: resolved in 1 (run 3, from the paper text); refused id_used_by_other_row in 2 (runs 2 and 151) because the payload also carries the row 'Type III taste cell', which the writer already maps to that same PathWhiz row, so a second row would double it; run 150 has neither text nor a full phrase: unresolved. |
| cell_type | Type III cells | 3 | unresolved: id_used_by_other_row ['CL:0002285'] x1; resolved CL:0002285 (OLS4, reviewed_synonym, via paper_context: 'type iii taste cell') x1; unresolved: no_exact_match [cellosaurus: no_exact_match] x1 | same as 'Type III cell' (singular tried: OLS4 exact 0). Outcome in the 3 runs: resolved to CL:0002285 in 1 (run 10, paper phrase 'type III taste cell', reviewed synonym, not a CL label); refused id_used_by_other_row in run 3 (the exact singular row 'Type III cell' of the same payload claimed CL:0002285 first); run 150 has no text and no full phrase: unresolved. |
| location | apical | 1 | unresolved | an orientation, not a compartment: no GO cellular-component term has the exact label or synonym 'apical' (OLS4 exact 0; fuzzy hits are regulation processes such as GO:1905141); PathWhiz rows 'Apical cell membrane' / 'apical membrane' exist but the paper does not say membrane, so not mapped (vocabulary_map.py note). |
| location | apical membrane of Type III cells | 1 | unresolved | sub-domain of a cell type's membrane, no GO term of that name (OLS4 exact 0, fuzzy 0); the generic PathWhiz 'Apical cell membrane' would drop the cell-type qualifier: not mapped. State-only name from the e2e run 150. |
| location | bacterial cell | 1 | unresolved | the whole organism cell, not a sub-cellular compartment: no GO cellular-component term 'bacterial cell' (OLS4 exact 0; fuzzy hits are processes); GO:0005623 'cell' is obsolete. |
| location | bulk cytosol | 1 | unresolved | no GO term has the exact label or synonym 'bulk cytosol' (OLS4 exact 0, fuzzy 0); GO:0005829 'cytosol' is the closest, but 'bulk' is a qualifier in no label or synonym, so the exact-only rule does not accept it. |
| location | cell | 149 | unresolved | GO:0005623 'cell' is obsolete in current GO (OLS4 term fetch: label 'obsolete cell', isObsolete true), so no GO cellular-component term is labelled 'cell' (exact 0 within GO:0005575); CL:0000000 'cell' is a cell type, not a compartment. Note for the owner: `pwml/ir.py` SUBCELLULAR_LOCATION_CREATE_DEFAULTS still writes GO:0005623 for 'cell', which passes PathWhiz's format check but is an obsolete id (ir.py is outside this task). |
| location | cytoplasmic leaflet of inner membrane | 1 | unresolved | same as 'cytoplasmic leaflet of the IM': no exact GO term (OLS4 exact 0, fuzzy 0), a leaflet is not the whole inner membrane. |
| location | cytoplasmic leaflet of the IM | 3 | unresolved | sub-compartment (one leaflet) of the bacterial inner membrane: no GO cellular-component term has this exact label or synonym (OLS4 exact 0, fuzzy 0). Nearest concept GO:0009898 'cytoplasmic side of plasma membrane' (exact synonyms 'internal side of plasma membrane', 'internal leaflet of plasma membrane') names other strings and rests on IM = plasma membrane; mapping to it or to PathWhiz 'Inner Membrane' would merge a leaflet with the whole membrane: not accepted. |
| location | extracellular environment | 5 | unresolved | no GO term has this label or synonym (OLS4 exact 0, fuzzy 0); GO:0005615 'extracellular space' (PathWhiz row 'Extracellular Space') is close in meaning but 'environment' is in none of its names, and vocabulary_map.py maps only 'extracellular'. |
| location | Inner Membrane | 2 | unresolved | organism-aware (P5): PathWhiz row 'Inner Membrane' is the bacterial inner membrane (98 % of its 1285 production states are prokaryote; e_pathwhiz_row_usage.txt), and these two runs are Homo sapiens, so the name is not written under that row. OLS4 GO exact for 'Inner Membrane' is 0 (GO has 'mitochondrial inner membrane' GO:0005743 and 'organelle inner membrane', but the paper does not say which organelle): unresolved, the PWML drops the location. |
| location | inner membrane | 1 | unresolved | organism undetermined (P5): the state using this name has no species and the payload also lists Arabidopsis thaliana next to E. coli, so the organism kinds are {bacterium, unknown}; not written under the bacterial row 'Inner Membrane'. OLS4 GO exact for 'inner membrane' is 0. |
| location | leaf surface | 1 | unresolved | an anatomical surface, not a cellular component: no GO term (OLS4 exact 0; the one fuzzy hit GO:0009897 'external side of plasma membrane' is unrelated). |
| location | medial Golgi | 2 | unresolved | GO has 'Golgi medial cisterna' GO:0005797 (exact synonym 'Golgi apparatus medial cisterna'); 'medial Golgi' is neither its label nor an exact synonym (OLS4 exact 0), so mapping would be a closest match. |
| location | microsomal fraction | 1 | unresolved | not an ontology compartment: a laboratory fraction (a sample). GO has no term of that name (OLS4 exact 0, fuzzy 0); the nearest, GO:0042598 'vesicular fraction', is obsolete. |
| location | nerve fiber | 1 | unresolved | GO has no exact label/synonym: GO:0043005 neuron projection lists 'nerve fiber' only as a RELATED synonym (term fetched, refused). UBERON:0006134 'nerve fiber' exists but PathWhiz `subcellular_locations.ontology_id` must match `GO:\d{7}`, and a nerve fiber is an anatomical structure, not a cellular component. |
| location | periplasmic leaflet of inner membrane | 1 | unresolved | same as 'periplasmic leaflet of the IM': no exact GO term (OLS4 exact 0, fuzzy 0), a leaflet is not the whole inner membrane. |
| location | periplasmic leaflet of the IM | 3 | unresolved | sub-compartment (one leaflet) of the bacterial inner membrane: no GO term with this exact label or synonym (OLS4 exact 0, fuzzy 0). GO:0098567 'periplasmic side of plasma membrane' has no synonyms and names another string; PathWhiz 'Inner Membrane' / 'Periplasm' would merge leaflet and whole: not accepted. |
| location | plasma_membrane | 1 | resolved by the vocabulary map | mapped to PathWhiz row 'Cell Membrane' (GO:0005886, GO label 'plasma membrane'): underscore spelling of the same term (new synonym in P5). |
| location | saliva | 2 | unresolved | a body fluid, not a cellular component: GO has none (OLS4 exact 0; fuzzy hits are salivary-gland regulation processes). UBERON:0001836 'saliva' exists but PathWhiz requires `GO:\d{7}` for locations. |
| location | stroma | 1 | unresolved | no GO term has the exact label or synonym 'stroma' (OLS4 exact 0). GO:0009570 chloroplast stroma and GO:0009532 plastid stroma carry no 'stroma' synonym at all (terms fetched: exact [], related []), so choosing one would be a guess about which stroma the paper means; UBERON:0003891 'stroma' is generic tissue stroma and not a GO id. |
| location | synaptic vesicles | 3 | resolved GO:0008021 (OLS4, exact_label, matched form 'synaptic vesicle') x1; unresolved: id_used_by_other_row ['GO:0008021'] x2 | resolved in 1 of 3 runs (run 3): the exact name has no hit, the singular 'synaptic vesicle' is the exact label of GO:0008021 (matched form recorded). In runs 10 and 150 the payload also has the row 'synaptic vesicle', which holds GO:0008021, so the plural row is refused id_used_by_other_row (one id, one row: it is a second name of the same compartment) and stays id-less. |
| location | trans-Golgi | 1 | unresolved | GO has 'trans-Golgi network' GO:0005802 (exact synonyms 'TGN', 'trans Golgi network'; 'Golgi trans face' only related); 'trans-Golgi' is neither label nor exact synonym (OLS4 exact 0), so mapping would be a closest match. |
| tissue | nerve fibers | 1 | unresolved | BTO has no exact 'nerve fibers' or singular 'nerve fiber' (OLS4 exact 0 both); the only fuzzy hit BTO:0002191 'fibre tract' is a different structure. |
| tissue | young leaf | 1 | unresolved | BTO has no exact label or synonym 'young leaf' (OLS4 exact 0, fuzzy 0 results); BTO:0000713 'leaf' (PathWhiz row 'Leaf') is the organ at any age, a wider entity: not mapped (vocabulary_map.py note). |

Names that did resolve in the measurement (no reason needed, listed for completeness):

- cell_type `epidermal cell` (1 runs): resolved: CL:0000362 (OLS4, exact_label)
- location `etioplast` (1 runs): resolved: GO:0009513 (OLS4, exact_label)
- location `inner mitochondrial membrane` (1 runs): resolved: GO:0005743 (OLS4, exact_synonym)
- location `synaptic cleft` (1 runs): resolved: GO:0043083 (OLS4, exact_label)
- location `synaptic vesicle` (4 runs): resolved: GO:0008021 (OLS4, exact_label)
- location `thylakoid membrane` (1 runs): resolved: GO:0042651 (OLS4, exact_label)
- tissue `bone marrow` (1 runs): resolved: BTO:0000141 (OLS4, exact_label)
- tissue `central nervous system` (1 runs): resolved: BTO:0000227 (OLS4, exact_label)
- tissue `cotyledon` (1 runs): resolved: BTO:0000300 (OLS4, exact_label)
- tissue `flower` (1 runs): resolved: BTO:0000469 (OLS4, exact_label)
- tissue `hypocotyl` (1 runs): resolved: BTO:0000613 (OLS4, exact_label)
- tissue `petal` (1 runs): resolved: BTO:0001040 (OLS4, exact_label)
- tissue `root` (1 runs): resolved: BTO:0001188 (OLS4, exact_label)

## B. What PathWhiz accepts as a cell-type id (read-only look at the app)

- `app/models/data/biological_states/cell_type.rb`: `validates :ontology_id, uniqueness: true, presence: true, format: { with: /\ACL:\d{7}\z/ }` (importer message 'Ontology can't be blank').
- The same file defines `CELL_TYPE_DB = "CELL ONTOLOGY"`; `tissue.rb` takes `BTO:\d{7}`, `subcellular_location.rb` takes `GO:\d{7}`.
- All 43 production `cell_types` rows (harness seed `vocab_inserts.sql`) carry CL ids; none carries a Cellosaurus (CVCL_) or any other prefix.
- Therefore a Cellosaurus accession cannot be written to `cell_types`. MG-63 and Saos-2 are kept in the final payload (field `cellosaurus_id`, `db` provenance in `final_payload_fill`) and the PWML writer drops them because they carry no CL id.

## C. Inner and outer membrane per occurrence (E)

PathWhiz rows `Inner Membrane` (id 32, GO:0070258) and `Outer Membrane` (id 33, GO:0043165) are the bacterial membranes: 1271 of 1285 and 1232 of 1243 production states using them have a prokaryote species, mitochondria have their own rows 12 and 14 (`e_pathwhiz_row_usage.txt`). Their stored GO ids are other terms (OLS4: GO:0070258 'inner membrane pellicle complex', GO:0043165 'Gram-negative-bacterium-type cell outer membrane assembly'), so the rows are bound by name only and the ids are never written. Correct GO terms for reference: GO:0005886 plasma membrane (exact synonym 'cytoplasmic membrane'), GO:0009279 cell outer membrane, GO:0019867 outer membrane (generic).

| run | species rows | name in the data | organisms of the states using it | before P5 (no organism) | after P5 (written as) |
|---|---|---|---|---|---|
| 1 `research/2026-10-21_0331/papers/PMC12096016/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 8 `tin-strict/2026-10-22_2101/papers/PMC12096016/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 29 `uns_verify/2026-08-21_2057/papers/PMC12452463/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 30 `s_verify/2026-08-21_2239/papers/PMC12096016/research` | Escherichia coli | Outer Membrane | bacterium | exact 'Outer Membrane' | mapped: 'Outer Membrane' |
| 30 `s_verify/2026-08-21_2239/papers/PMC12096016/research` | Escherichia coli | Inner Membrane | bacterium | exact 'Inner Membrane' | mapped: 'Inner Membrane' |
| 31 `uns_verify/2026-08-21_2239/papers/PMC12096016/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 31 `uns_verify/2026-08-21_2239/papers/PMC12096016/strict` | Escherichia coli | Inner Membrane | bacterium | exact 'Inner Membrane' | mapped: 'Inner Membrane' |
| 35 `uns_verify/2026-08-21_2239/papers/PMC12452463/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 36 `s_verify/2026-08-21_2239/papers/PMC12782028/research` | Homo sapiens | Inner Membrane | eukaryote | exact 'Inner Membrane' | unmapped: not written under a row name (location dropped from the PWML unless a state would collide) |
| 46 `uns_verify/2026-08-22_2147/papers/PMC12096016/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 49 `s_verify/2026-08-22_2147/papers/PMC12452463/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 50 `uns_verify/2026-08-22_2147/papers/PMC12452463/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 55 `s_verify/2026-08-22_2147/papers/PMC13231680/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 59 `s_verify/2026-08-24_1203/papers/PMC12452463/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 64 `s_verify/2026-08-24_1402/papers/PMC12782028/research` | Homo sapiens | Inner Membrane | eukaryote | exact 'Inner Membrane' | unmapped: not written under a row name (location dropped from the PWML unless a state would collide) |
| 68 `uns_verify/2026-08-24_1428/papers/PMC12444477/strict` | Escherichia coli, E. coli, Arabidopsis thaliana | inner membrane | bacterium, unknown | exact 'inner membrane' | unmapped: not written under a row name (location dropped from the PWML unless a state would collide) |
| 68 `uns_verify/2026-08-24_1428/papers/PMC12444477/strict` | Escherichia coli, E. coli, Arabidopsis thaliana | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 69 `s_verify/2026-08-24_1428/papers/PMC12452463/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 70 `uns_verify/2026-08-24_1428/papers/PMC12452463/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 76 `uns_verify/2026-08-25_1216/papers/PMC12444477/strict` | Escherichia coli, Arabidopsis thaliana | inner membrane (IM) | bacterium | unmapped (kept by name or dropped, no row) | mapped: 'Inner Membrane' |
| 76 `uns_verify/2026-08-25_1216/papers/PMC12444477/strict` | Escherichia coli, Arabidopsis thaliana | outer membrane (OM) | bacterium | unmapped (kept by name or dropped, no row) | mapped: 'Outer Membrane' |
| 83 `s_verify/2026-08-28_1816/papers/PMC12452463/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 84 `uns_verify/2026-08-28_1816/papers/PMC12452463/strict` | Escherichia coli, Arabidopsis thaliana | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 90 `s_verify/2026-09-01_1612/papers/PMC12096016/research` | Escherichia coli | Outer Membrane | bacterium | exact 'Outer Membrane' | mapped: 'Outer Membrane' |
| 90 `s_verify/2026-09-01_1612/papers/PMC12096016/research` | Escherichia coli | Inner Membrane | bacterium | exact 'Inner Membrane' | mapped: 'Inner Membrane' |
| 94 `s_verify/2026-09-01_1612/papers/PMC12444477/research` | Escherichia coli | inner membrane | bacterium | exact 'inner membrane' | mapped: 'Inner Membrane' |
| 94 `s_verify/2026-09-01_1612/papers/PMC12444477/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 95 `s_verify/2026-09-01_1612/papers/PMC12452463/research` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 105 `uns_verify/2026-09-02_2052/papers/PMC12452463/strict` | Escherichia coli | outer membrane | bacterium | exact 'outer membrane' | mapped: 'Outer Membrane' |
| 110 `s_verify/2026-09-06_1425/papers/PMC11172790/research` | Pseudomonas aeruginosa | inner membrane | bacterium | exact 'inner membrane' | mapped: 'Inner Membrane' |
| 111 `uns_verify/2026-09-06_1425/papers/PMC11172790/strict` | Pseudomonas aeruginosa | inner membrane | bacterium | exact 'inner membrane' | mapped: 'Inner Membrane' |

Leaflets (`cytoplasmic/periplasmic leaflet of the IM` and `... of inner membrane`, 8 occurrences in runs 68, 75, 76, 94) are never mapped to the whole membrane: see section A.

## D. Limits of this record (not established)

- The two numbered-cell-type resolutions rest on one reviewed synonym (`type iii taste cell` -> `type III taste bud cell`, marked 'reviewer to confirm' in vocabulary_map.py); only `Type II cell` -> CL:0002286 rests on an exact CL label.
- The paper text reaches the ontology fill only through `state['final_payload_paper_text']` (run_pathway with the flag on); the Streamlit path has none, so numbered names resolve there only from the payload's own names.
- Organism kind comes from the species rows of the states that use the location (PathWhiz `classification` plus an exact genus table); a state without a species counts every species row of the payload, so a mixed-species payload leaves the name undetermined (run 68 `inner membrane`).
- Two payload rows can still reach one PathWhiz row through the vocabulary map alone (e.g. `neurons` and `Neuron`); the ontology fill refuses to add to that (`id_used_by_other_row`) but does not merge existing duplicates.
- An id-less row that the writer's wildcard guard keeps (a state would otherwise collide) is still written by name; for a name equal to a PathWhiz row name (here only the non-bacterial `Inner Membrane` / `Outer Membrane`) that binds the bacterial row at import. Not seen in the 14 replayed runs; no test covers it.
- Cellosaurus: one release (57.0, 2026-10-22) and two cell lines were checked; the exact-match rule was not measured on other cell lines (HeLa, HEK293 and similar return many derivatives; a full page of 100 is refused).
