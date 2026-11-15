# B. PWML fields: drawing vs biology; what the writer/IR CREATE; what they DROP

Corpus for counts: 61 PWML files (= the canonical runs with pwml_ir.json AND a .pwml; one of the 62 IR runs wrote no .pwml),
or the 152 canonical final_mapped.json where stated. Class codes: BIO-ID (identity), BIO-STRUCT (relations between entities),
BIO-CTX (location/organism context), CHEM (molecular property), PW-KEY (PathWhiz database key or internal reference,
needed to import, not biology), DRAW (coordinates, size, colour, template, z-order, edge path, label type, image refs).
Source codes: PAYLOAD (a payload field feeds it), DB (filled from PathBank/other DB before the freeze), CONST (constant in code),
INVENTED (made up in IR/writer), EMPTY (written blank/nil).

## B1. Classification of every PWML element (from the 335 distinct element paths seen in 61 files)

| PWML element(s) | class | source today | writer/IR file:line |
|---|---|---|---|
| root: named-for-id/-type, cached-name/-description/-subject, pw-id | PW-KEY (+title text) | CONST ("Generated Pathway", "Metabolic", "", "PW000000") 61/61 | writer.py:2743-2763, 2941, 3007-3009 |
| pathway/name, description, subject | BIO (title) | CONST; payload has no pathway object (0/152 runs have `metadata`) | writer.py:2700-2736; ir.py:1352-1358 reads payload["metadata"] which no run has |
| pathway/species-id | BIO-CTX | first species of the IR list | writer.py:2702, 1002-1003 |
| pathway/references, sub-pathways | BIO (citations) | EMPTY list, 61/61 | writer.py:2709-2710 |
| species (name, taxonomy-id, classification, common-name) | BIO-CTX | PAYLOAD+DB (taxonomy 205/209 payload rows) | writer.py:988-1012 |
| cell-types, tissues | BIO-CTX | PAYLOAD (name only); ontology-id nil 20/20, 15/15; tissue image/template nil | writer.py:978-986, 1028-1042 |
| subcellular-locations (name, ontology-id) | BIO-CTX | PAYLOAD; ontology-id filled 60/188 | writer.py:1014-1026 |
| biological-states (name, species-id, subcellular-location-id, cell-type-id, tissue-id) | BIO-CTX | PAYLOAD; ids synthetic (`self.ids.next()`); pwbs-id only if trusted | writer.py:1044-1064 |
| compounds: name, cas, hmdb/kegg/chebi/pubchem/biocyc/chemspider/drugbank ids | BIO-ID | PAYLOAD/DB (db_row preferred over payload) | writer.py:1087-1140 |
| compounds: pwc-id, short-name | PW-KEY | DB only if trusted (4/431, 38/431) | writer.py:1114-1125, 492-506 |
| compounds: description, moldb-* (formula, smiles, inchi, inchikey, iupac, masses, charges, donor count, logS), foodb-id, synonyms | CHEM | EMPTY (0/431 populated except description 4/431) | writer.py:2674-2681 (blank-fill of signature fields) |
| element-states (all entity types) | BIO-CTX | EMPTY list (location is carried by *-locations instead) | writer.py:1128, 1183, 1265, 1313 |
| proteins: name, uniprot-id, drugbank-id, ec-numbers, species-id | BIO-ID | PAYLOAD (uniprot 241/241, EC rows in 43 runs) | writer.py:1216-1270 |
| proteins: gene-name, locus, genbank-id, hmdbp-id, description, synonyms | BIO-ID | EMPTY 0/241 each, although payload has gene_name for 652/787 proteins | writer.py:1260-1269 (keys not emitted) |
| protein-complexes: name, members (protein-id, stoichiometry) | BIO-STRUCT | PAYLOAD components (stoich nil allowed, writer.py:813-816) | writer.py:1272-1315, 800-867 |
| protein-complexes: species-id | BIO-CTX | the pathway default species, not the complex's own | writer.py:1310 |
| protein-complexes: pwp-id; reactions pwr-id; transports pwt-id; interactions pwi-id; element-collections pwec-id | PW-KEY | INVENTED accession strings `PW_P/R/T/I{id:06d}` | writer.py:1311, 1376, 1438, 1503, 1164 |
| nucleic-acids: name | BIO-ID | PAYLOAD; chebi, genbank, gene-name, nucleotide-type, pwna-id EMPTY 0/12 | writer.py:1171-1184 |
| element-collections: element-type "Compound", collection-type "Set", element-id nil | BIO-STRUCT | CONST/INVENTED (payload has 1 such row in 152 runs, 0 after the freeze) | writer.py:1142-1169 |
| bounds | BIO-STRUCT | EMPTY; no `bound` concept exists in the payload at all | writer.py:1067 |
| reactions: left/right elements (element-id, element-type, stoichiometry, currency) | BIO-STRUCT | names PAYLOAD; stoichiometry INVENTED=1 (623/623); currency CONST false (620/620) | writer.py:1325-1348 |
| reactions: direction | BIO-STRUCT | INVENTED "Right" 245/245 (payload has no direction field, 0/621) | ir.py:925-931, 2082; writer.py:954-960, 1377 |
| reactions: spontaneous | BIO-STRUCT | CONST false 245/245 (payload has the field but IR ignores it) | ir.py:2087-2088; writer.py:1375 |
| reaction-enzymes (protein-complex-id / protein-id, inhibitor, enzyme-class) | BIO-STRUCT | PAYLOAD; enzyme-class nil 288/288; "activator" not distinguishable from catalyst | writer.py:1349-1371; ir.py:2160-2207 |
| transports: transport-elements (cargo, left/right state, stoichiometry, direction) | BIO-STRUCT | cargo+states PAYLOAD; stoichiometry INVENTED 1; direction CONST "Right" (42/42) | writer.py:1392-1414; ir.py:2265-2302 |
| transports: transport-type | BIO-STRUCT | EMPTY nil 42/42 | writer.py:1439 |
| transports: transport-transporters (protein, biological-state-id) | BIO-STRUCT | protein PAYLOAD; state INVENTED = first state in 22/22 cases | writer.py:1416-1434; ir.py:2328-2342, 1827-1832 |
| interactions: interaction-type | BIO-STRUCT | PAYLOAD word, only Activation/Inhibition exported | writer.py:1499-1507; interaction_types.py:23-45 |
| interactions: left/right elements, stoichiometry | BIO-STRUCT | PAYLOAD endpoints; stoichiometry CONST 1 | writer.py:1481-1497 |
| compound-/protein-/nucleic-acid-/element-collection-locations (x, y, width, height, font-size, zindex, hidden, label-type, visualization-template-id) | DRAW | INVENTED layout | writer.py:1528-1612; ir.py:1938-2067 |
| protein-complex-visualizations (+ member protein locations) | DRAW | INVENTED layout; one location per member | writer.py:1614-1710 |
| edges (path, options, zindex, hidden, visualization-template-id) | DRAW | INVENTED (curve/elbow/arrow geometry) | writer.py:41-146, 2219-2390 |
| reaction-/transport-/interaction-visualizations (+ side Left/Right, edge ids, location ids) | DRAW (with "side" = substrate/product role) | INVENTED | writer.py:2391-2579 |
| bound-visualizations (one per biological state, full-canvas rectangle) | DRAW (state background) | INVENTED, 180/180 states | writer.py:1509-1526; ir.py (bound_visualizations) |
| membrane-visualizations, label-locations, zoom-visualizations, drawable-element-locations, vacuous-*, sub-pathway-visualizations | DRAW | EMPTY by construction | writer.py:2581-2589, 2395 |
| pathway-visualization: width, height, background-color, context position/id | DRAW | CONST (canvas) | writer.py:2764-2790 |
| all `<id>` values (state, location, edge, member, visualization rows) | PW-KEY | INVENTED sequential (`IdFactory`) or the PathBank id when the entity matched a DB row | writer.py:719-753, 928-946 |

Reading: DRAW rows dominate the file by row count (edges 1026, locations 611 compound + 384 protein, visualization rows 245+42+17+265 in 61 files); the biology is in about 15 element families.

## B2. CREATED by the IR/writer, not in the payload (each M unless marked)

| # | what is created | evidence (61 PWML / 62 IR / 152 payload runs) | file:line |
|---|---|---|---|
| 1 | Pathway title "Generated Pathway", subject "Metabolic" (also for signalling/transport papers), description "", `pw-id` PW000000 | 61/61 PWML | writer.py:3007-3009, 2941; ir.py:1352-1358 (reads `metadata`, 0/152 runs have it) |
| 2 | `references` always empty: the paper's own PMID/DOI/title is not in the payload (request.json has pmids, the payload does not) | 61/61 empty; 0/152 payloads carry a bibliographic field | writer.py:2710 |
| 3 | Reaction stoichiometry 1 (payload participants are bare name strings: 623/623 str in the 62 IR runs; 0 leading coefficients in 1544 participant names of 152 runs) | 623/623 | ir.py:934-955, 2126; writer.py:1344 |
| 4 | Reaction direction ">" / "Right" for every reaction (9 of 621 reaction evidence quotes in 152 runs mention reversibility) | 246/246, 245/245 | ir.py:925-931; writer.py:954-960 |
| 5 | spontaneous=false for every reaction | 246/246 | ir.py:2087-2088 |
| 6 | currency=false on every reaction member although a 43-name currency list exists (used only for layout) | 620/620 | writer.py:1346 vs 249-372, 2034 |
| 7 | Transport element stoichiometry 1 and direction "Right"; transport-type nil | 42/42 | ir.py:2300; writer.py:1412, 1439 |
| 8 | Transporter location = FIRST biological state (payload transporters carry no state: 0/42 with a state field); only 1 of 22 lies in a membrane location | 22/22 | ir.py:2334-2338 via resolve_state 1827-1832 |
| 9 | Fallback state: any process without a state gets the first state (`default_bs_key`); payload reactions without biological_state: 131/621, interactions 86/352, transports 1/106 (from/to) | see counts | ir.py:1825-1847, 2076, 2362, 2379 |
| 10 | `__auto_state__` in location "cell", added by Stage 3 and kept to the freeze | 93/152 payloads (61%); 29/182 IR states in the 62 runs | pipeline/process_normalizer.py:3144-3190 (D-099 governs its lifecycle) |
| 11 | "Default state" if no biological_states at all | 0 in the 62 runs (report flag generated_default_state) | ir.py:1743-1759 |
| 12 | Enzyme "wrapper" complexes: 254 of 415 payload complexes are pipeline-generated (241 single-member wrappers around one protein, 14 named-source wrappers, 1 inferred) | 254/415 (61%) in 152 runs; 135/238 in the 62 IR runs | pipeline/entity_identity.py:320; process_normalizer.py (generated wrapper rules); D-070, D-117 |
| 13 | Complex members that are the PathBank "Unknown" sentinel (protein 9659, uniprot "Unknown"): all members Unknown in 138/415 payload complexes (33%); 73/238 IR complexes; 73/246 reactions (30%) have only Unknown-backed enzyme complexes | M | pipeline/entity_identity.py:9-11, 90-114; process_normalizer.py:4935-4985 (accepted by design, D-070/D-117 4.) |
| 14 | Legacy SBML also invents an "Unknown" protein for any complex with no components | code | sbml/json_to_sbml.py:842-852 |
| 15 | Complex species-id = pathway default species; pathway species = first IR species | 234/234 | writer.py:1310, 2702 |
| 16 | Accession strings pwp-/pwr-/pwt-/pwi-/pwec-id; ids from fixed ranges (compound 20000+, protein 30000+, complex 40000+, reaction 50000+) when no DB id | all rows | writer.py:746-753, 1096, 1228, 1280, 1323 |
| 17 | Interaction member stoichiometry 1, left/right element-type, edge templates 148/149 (Inhibition/Activation) | 17 full interactions | writer.py:1496; ir.py:2467 |
| 18 | Protein-location label-type "protein" for complex members, template 2 for template 4 | all complex members | writer.py:948-952, 1693 |
| 19 | One bound-visualization rectangle per state (canvas 3200x1400 etc.) | 180/180 | writer.py:1509-1526 |
| 20 | Empty chemistry blocks (moldb-*, foodb-id) written as blank elements so the file matches the reference template | 431/431 compounds | writer.py:2674-2681 |
| 21 | Species rows injected by hydration with status unresolved (example: Arabidopsis thaliana in the mouse OTOP1 run); this is payload-side, not writer-side | 1 seen in e2e5 | mapping/map_ids.py:13183 |

## B3. DROPPED between payload and PWML (counts are payload items that carry the datum, then what PWML holds)

| # | dropped datum | payload (152 runs) | PWML (61 files) | file:line |
|---|---|---|---|---|
| 1 | Evidence quote per reaction / transport / interaction | 621/621 reactions, 106/106 transports, 351/352 interactions | 0 (PWML has no slot) | lost at the IR: ir.py:2078-2089 (reaction), 2257-2263 (transport), 2386-2406 (interaction) build rows with no evidence field |
| 1b | Entity evidence quote, source_refs, confidence, provenance | e.g. compounds evidence 12/1095, source_refs 24/1095 | kept in IR (compounds evidence 4/436) but never written | kept by ir.py:835-886 `_copy_common_entity_fields`; writer.py:1116-1139 does not emit |
| 2 | provenance_lineage (origin, support marks, review flags) | 1095/1095 compounds, 787/787 proteins, 415/415 complexes | 0 | not in the key list of ir.py:835-886 |
| 3 | Interactions with a non-signed or unwrappable form (withheld) | 40 of 127 IR interactions (38 untyped, 2 bare protein) | 0 in PWML; listed in side file only | ir.py:2400-2455; writer.py:1463-1477 |
| 4 | Old-writer interaction stubs (id, name, raw type, no members): 70 in 5 runs | n/a | 70 stubs (invalid for PathWhiz: types such as `inhibits`, `binds`) | older writer; current code skips them |
| 5 | Interaction wording beyond Activation/Inhibition (binds, regulates, phosphorylates...) | 237/352 final interactions have a non-mapped word | 0 | interaction_types.py:23-45 |
| 6 | Gene names / symbols of proteins | 652/787 | 0/241 (gene-name blank) | writer.py:1260-1269 |
| 7 | Synonyms / aliases (compounds, proteins, locations) | 1095/1095 compounds, 787/787 proteins | 0 (synonyms blank) | writer never emits |
| 8 | Reaction-level modifiers with role cofactor/transporter; activator not marked | 3 cofactor + 1 transporter + 21 activator of 738 modifiers | cofactor/transporter absent; activator = catalyst | ir.py:2163-2168 |
| 9 | Catalysts that resolve to a non-protein entity | counted in IR warnings (non_protein_catalyst_dropped; 0 in the 62 IR reports) | dropped | ir.py:2186-2197 |
| 10 | Per-entity element_states / multi-state membership beyond one tile per (entity, state) | 884/1095 compounds have element_states | element-states blank; tiles via locations | writer.py:1128 etc. |
| 11 | Compound description, class, RAG tier, db_row details | 17/1095 description | 4/431 | writer.py:1119 |
| 12 | Cell type/tissue hierarchy and compartment nesting: never stored (see C) | 0/152 | 0 | n/a |
| 13 | Entities outside any process or location (pre-freeze quarantine, not writer) | merged 525 compounds -> 431 PWML; 474 proteins -> 241 | | pipeline/strict_quarantine.py |
| 14 | Process stage losses merged->final: reactions 290 -> 245, interactions 403 -> 127 -> 17 drawn, transports 44 -> 42 (61 runs) | | | quarantine (D-114) |
