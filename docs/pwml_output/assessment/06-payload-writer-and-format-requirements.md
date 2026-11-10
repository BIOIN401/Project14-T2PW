# 06. What the payload holds, what the PWML writer invents or drops, and what each export format needs

Branch `pwml/output-assessment` (from 85b5f9d4). Read-only analysis. Labels: M measured by me, L log/doc/code evidence, G guess.
Bulky tables: `evidence/06/` (A_field_inventory.md, B_classification_created_dropped.md, D_export_requirements.md, per_run_stage_counts.md, gaps*.json, scripts).

## 0. NOT ESTABLISHED

- Only 62 of the 152 canonical runs have a `pwml_ir.json` (61 also have a .pwml). The other 90 canonical runs are 51 research-mode legs and 39 strict legs with no IR (M). IR and PWML statements therefore rest on 62/61 runs; payload statements on 152. Four deep-dive runs were hand-compared; the rest were scanned by script.
- The 40 non-canonical runs (quarantine_ok false or null) were not used for headline numbers. Interaction spelling was repeated on them (34 exact spellings, 208 interactions) and does not change the picture.
- Runs span 2026-08 to 2026-10, so PWML files were written by different writer versions. The 70 interaction stubs below come from an older writer; the current writer skips those rows (writer.py:1463-1477). I did not re-run the current writer on old IR (no pipeline runs allowed), so "what the PWML holds" is what the runs wrote, not necessarily what today's code would write.
- The stage the brief calls "the frozen payload" is `final_mapped.json`: for all 152 canonical runs its hash equals the `payload_sha256` stored in `final_stage3_gate_report.json` (M, hashchk.py, 152/152), and `batch/driver.py:1860-1867` writes the canonical payload there (L). No separate `final.canonical.json` is kept in run folders.
- Specs: KGML (kegg.jp docs) and SBO:0000240/0000245 labels were read online. SBML L3V2, BioPAX L3, SBGN-ML PD and GPML rows are from memory (marked "recalled" in D); they need a spec check before anyone implements. Fillability ratings (Y/P/N) are my judgment, not measured; coverage numbers inside them are measured.
- Whether a database lookup would actually raise ChEBI/KEGG/GO coverage was not tested (no network calls to those services were made, and none may be made after the freeze, D-015).
- The family split of interaction verbs (binding vs catalysis vs regulation) is a keyword heuristic (G); the signed/unsigned split uses the repo's own table (M).

## 1. Stage by stage: what is carried (four runs, then the corpus)

Counts per stage, same run: merged_payload (Stage 1+2A, pre-mapping) -> final_mapped (frozen) -> pwml_ir -> PWML (full table: `per_run_stage_counts.md`).

| run | compounds | proteins | complexes | reactions | transports | interactions | states |
|---|---|---|---|---|---|---|---|
| OTOP1 sour taste (C:/t/e2e5-run) | 4/2/2/2 | 11/2/2/2 | 1/2/2/2 | 3/0/0/0 | 3/3/3/3 | 7/3/3/3 | 8/8/8/8 |
| chlorophyll PMC5130987 (metabolic + gene regulation) | 47/25/25/25 | 81/40/40/40 | 3/33/33/33 | 34/23/23/23 | 0 | 118/31/31/10 | 14/10/10/10 |
| Wnt/PCP PMC10425814 (signalling) | 4/4/4/4 | 40/13/13/13 | 5/10/10/10 | 8/3/3/3 | 3/2/2/2 | 75/11/11/1 | 10/7/7/7 |
| enterobactin PMC12096016 (metabolic + transport) | 11/9/9/9 | 8/7/7/7 | 0/5/5/5 | 5/5/5/5 | 1/1/1/1 | 12/10/10/1 | 3/3/3/3 |

Corpus (61 runs with PWML, M): compounds 525/431/431/431; proteins 474/241/241/241; complexes 18/234/234/234; reactions 290/245/245/245; transports 44/42/42/42; interactions 403/127/127/17 drawn (+70 old stubs; 40 withheld, never drawn). IR to PWML loses nothing except interactions, so almost every loss of biology is before the IR (quarantine, D-114) or in the IR (evidence, direction, stoichiometry). Complexes rise 18 -> 234 because the pipeline wraps single enzymes (section 3, row 12).

(A) Field inventory: `A_field_inventory.md` lists every field at every stage for compound, protein, complex, nucleic acid, element collection, bound, reaction, transport, interaction, location, state, organism, tissue, cell type, element location and pathway. Headlines (M):

- Payload processes are thin. A reaction carries `name, inputs[str], outputs[str], enzymes[{entity,role,evidence,confidence}], modifiers, biological_state, evidence, spontaneous, scope_membership, locked_reaction_id` and nothing else: no stoichiometry (1544/1544 participant names are bare strings, 0 with a leading coefficient), no direction (0/621), no EC/Rhea/KEGG R number (0/621), no step order (merged 0/152; `reaction_order` exists in 11/152 final payloads only, added by the curator, `curation/pathway_curator.py:11`, and is read by no exporter).
- A transport carries `cargo` (one string), `from_/to_biological_state`, `transporters[{protein|protein_complex, evidence}]`; no transporter state (0/42), no transport type.
- An interaction carries `entity_1, entity_2, relationship` (free text), `biological_state`, `evidence`; no sign field, no mechanism, no role.
- Compounds: chebi 632/1095 (58%), kegg 556 (51%), hmdb 510 (47%), pubchem 609 (56%), any of the four 672 (61%), cas 496, PathWhiz id 822 (75%). Proteins: uniprot 712/787 (90%), gene name 652 (83%), EC 511 (65%). Species taxonomy id 205/209. Subcellular locations: GO canonical_id 28/414 (7%); no ontology id elsewhere. Rhea ids: 0.
- No payload field exists for: pathway title, description, subject, PMID/DOI/references (0/152), compartment parent/part_of (0/152), bounds (0), reaction direction, step order (see above), transporter membrane, interaction sign.
- Evidence quotes exist for 621/621 reactions, 106/106 transports, 351/352 interactions (payload). The IR keeps entity-level evidence but not process-level evidence (ir.py:2078-2089, 2257-2263, 2386-2406); the PWML has no slot for either.

## 2. (B) Drawing versus biology

`B_classification_created_dropped.md` B1 classifies all 335 distinct element paths of the 61 PWML files. In short (M, 61 files):

- Biology and identity: species, cell types, tissues, subcellular locations, biological states, compounds (ids), proteins, complexes (members), nucleic acids, reactions (members, enzymes), transports (cargo, states, transporters), interactions (type, endpoints). About 15 element families.
- PathWhiz keys (needed to import, not biology): every `<id>`, pwc-/pwp-/pwr-/pwt-/pwi-/pwec-id, pw-id, named-for-id.
- Drawing: compound/protein/nucleic-acid locations (x, y, width, height, font-size, zindex, hidden, label-type, template id), protein-complex visualizations, edges (path, options), reaction/transport/interaction visualizations (including `side`), bound visualizations (one full-canvas rectangle per state, 180/180 states), canvas width/height/colour. By row count drawing dominates: 1026 edges, 611 compound + 384 protein locations, 569 visualization rows.
- Written empty by construction: membrane-visualizations, label-locations, zoom, drawable-element-locations, vacuous-*, sub-pathways, references, `bounds` (writer.py:2581-2589, 2710, 1067). Compound chemistry (moldb-*, foodb-id, synonyms) is written blank on 431/431 compounds (writer.py:2674-2681).

## 3. What the IR/writer CREATE (not in the payload) and what they DROP

Full tables with file:line: B2 and B3. The ones that matter, all M:

| # | created | extent | where |
|---|---|---|---|
| 1 | pathway title "Generated Pathway", subject "Metabolic" (also for the Wnt and sour-taste signalling papers), empty description, `pw-id` PW000000, empty references | 61/61 | writer.py:3007-3009, 2941, 2710; ir.py:1352-1358 would read `payload["metadata"]`, which 0/152 payloads contain |
| 2 | stoichiometry 1 on every reaction member | 623/623 | ir.py:934-955, 2126 |
| 3 | direction ">" / "Right" on every reaction and transport element | 246/246 IR, 287/287 PWML | ir.py:925-931; writer.py:954-960, 1412 |
| 4 | spontaneous=false although the payload has the field | 246/246 | ir.py:2087-2088 |
| 5 | currency=false although a 43-name currency list exists (layout only) | 620/620 | writer.py:1346 vs 249-372 |
| 6 | transporter state = FIRST biological state (payload has no state); 1 of 22 is in a membrane location | 22/22 | ir.py:2334-2338, 1827-1832 |
| 7 | fallback state = first state for any process without a state: payload reactions without one 131/621, interactions 86/352 | | ir.py:1825-1847 |
| 8 | `__auto_state__` in location "cell" (created at Stage 3, kept to the freeze) | 93/152 payloads (61%); 29/182 IR states | pipeline/process_normalizer.py:3144-3190 (D-099) |
| 9 | protein-complex species = pathway default species; pathway species = first species | 234/234 | writer.py:1310, 2702 |
| 10 | accessions `PW_P/R/T/I/EC{id:06d}`; ids from fixed ranges when no DB id | all rows | writer.py:1164, 1311, 1376, 1438, 1503, 746-753 |
| 11 | transport-type nil, transport direction "Right", interaction stoichiometry 1 | 42/42, 17 | writer.py:1439, 1412, 1496 |
| 12 | enzyme wrapper complexes (pipeline-generated, upstream of the exporters): 254/415 payload complexes (241 single-protein wrappers) | 61% | pipeline/entity_identity.py:320; process_normalizer.py |
| 13 | members that are the PathBank "Unknown" sentinel (protein 9659): complexes with only Unknown members 138/415 (33%); in the 62 IR runs 73/238 complexes, and 73/246 reactions (30%) have only Unknown-backed enzymes | accepted by design | entity_identity.py:9-11, 90-114; process_normalizer.py:4935-4985 (D-070, D-117 2.4) |

Dropped (B3): reaction/transport/interaction evidence quote (621/621, 106/106, 351/352) lost in the IR; provenance lineage and support marks lost in the IR (ir.py:835-886); protein gene names (652/787 in payload, 0/241 in PWML, writer.py:1260-1269); synonyms (1095/1095 compounds, 787/787 proteins, 0 written); activator modifiers written as plain catalysts (21) and cofactor/transporter modifiers dropped (4) (ir.py:2163-2168); 40 of 127 IR interactions withheld and absent from the PWML (38 untyped, 2 bare protein, ir.py:2400-2455); interaction words other than Activation/Inhibition (237 of 352 final interactions, interaction_types.py:23-45); the 70 stub interactions in 5 older PWML files carry raw words such as `inhibits`, `binds`, `modifies` that PathWhiz cannot store (the DB holds only Activation and Inhibition, interaction_types.py:23-27).

## 4. (C) The diagram prototype's gaps, checked against the data

Source: `C:/t/ev-diagram/t2pw_diagrams.py` (build() handles `__auto` states, "Unknown", unsigned regulation, junction-only order) and its itypes.py/etypes.py.

| prototype claim | verdict | numbers (M) | payload gap or writer gap |
|---|---|---|---|
| interaction types are free text, 29 spellings | CONFIRMED, exactly: 29 normalized (30 exact) spellings in the 62 IR runs / 131 interactions. In the 152 payloads: 42 normalized (48 exact) over 352 interactions; before quarantine 54 (61 exact) over 837 | `regulates` 5/131 IR, 10/352 payload; only 43/131 IR and 115/352 payload rows map to Activation/Inhibition | PAYLOAD gap (Stage 1 emits a free word; no enum). The IR/PWML restriction to two signs is a LOCKED product limit of PathWhiz (interaction_types.py:23-45, D-104 Amend. 1) |
| `regulates` has no sign | CONFIRMED | regulates/modulates/regulation 14/352 rows; competition and binding also unsigned | payload gap by nature of the paper's words; D-104 Amend. 1 forbids guessing a sign |
| unsigned binding is lost | NEW: 139/352 rows (39%) are binding/association words (G on grouping) and cannot go into PWML at all | KGML, BioPAX, SBGN, GPML can carry them | PWML limit; not a gap for other formats |
| compartment nesting is not stored | CONFIRMED: no parent/part_of/contained field in 152 payloads or 62 IRs; 130/152 runs have >=2 locations; 56 distinct location names, 15 have a container in a table that exists only for merging (stage_one_ensemble.py:282-300, D-109) | | PAYLOAD gap (and IR gap). A GO-based table could fill ~15 names today |
| transporters have a cell state, not a membrane | CONFIRMED with cause: transporter state is `resolve_state("")` = first state (ir.py:2334-2338, 1827-1832). 22/22 transporters sit in the first state; 20/22 of those states are cell/cytosol locations and 1 is a membrane. 20 of 42 transports have no transporter at all. Payload transporters have no state field (0/42) | | PAYLOAD gap (state never extracted) made worse by a WRITER/IR default |
| complex members point at a placeholder "Unknown" | CONFIRMED: 138/415 complexes (33%) all-Unknown; 73/246 reactions (30%) in the IR runs have only Unknown-backed enzymes; sour taste and Wnt runs are dominated by it | | PAYLOAD/pipeline decision (D-070, D-117), not a writer invention; a diagram must show these as "unknown protein" |
| there is no step order | CONFIRMED but derivable: `reaction_order` 11/152 (curator only), none in IR/PWML. From shared non-currency compounds, 188/246 reactions (76%) link to a neighbour in 43/62 runs; of 165 links 23 (14%) point backwards in list order; in 34 of the 43 runs every link points forward | the writer lays out in list order with a serpentine grid (writer.py:35-38, 689-717; shared-intermediate chaining only between list neighbours, writer.py:2020-2060) | PAYLOAD gap; a graph-derived order is deterministic and does not need the LLM, but it is a layout, not paper-stated |
| auto states / unplaced items | CONFIRMED: `__auto_state__` in 93/152 payloads (61%) | | pipeline (D-099) |

## 5. (D) Export requirements

Full field tables per format (SBML L3V2, KGML, BioPAX L3, SBGN-ML PD, GPML) with HAVE / DB / origin / today / cheap / upstream: `D_export_requirements.md`. Summary matrix (rows judged Y fillable, P partial, N not; X rows (format has no slot) excluded; qualitative, G):

| format | rows | today Y/P/N | after cheap export-side fixes | after pre-freeze work (PMID/title carry-through, DB lookups before the freeze) |
|---|---|---|---|---|
| SBML L3V2 | 30 | 9/10/11 | 13/11/6 | 19/10/1 |
| KEGG KGML | 15 (+3 X) | 4/6/5 | 5/8/2 | 7/8/0 |
| BioPAX L3 | 11 | 3/6/2 | 5/5/1 | 7/4/0 |
| SBGN-ML PD | 6 | 2/3/1 | 2/4/0 | 2/4/0 |
| GPML | 5 | 0/4/1 | 1/3/1 | 3/2/0 |

Key facts behind the matrix (M unless marked):

- Legacy SBML (`src/t2pw/sbml/json_to_sbml.py`, 1245 lines) is a text writer for L3V1 (:704-715), PathWhiz-flavoured: compartments = biological states with SBO:0000240 which is "material entity", not a compartment (:744, OLS); proteins AND complexes get SBO:0000245 macromolecule (:856, :898); every species gets initialAmount=1 (:855); every reaction `reversible="false"` (:997) and stoichiometry 1 (:1031); a complex with no members gets an invented "Unknown" protein (:842-852); no `<notes>`, no interactions (0 occurrences of "interaction"), old `urn:miriam:` URNs (:757-785), no compartment nesting. Status: called with `build_legacy_sbml=True` only from `app/streamlit_app.py:3022`; `service/post_run.py:174` passes False and the default is False (`service/postpipeline.py:492`); five test files mention it (tests/test_sbml_render_pathwhiz_like.py, test_process_normalizer.py, test_orch757, test_orch759b, test_orch762); I did not check whether any asserts SBML content (G: render and identity tests only).
- KGML: spec fetched. Entries need `name` as `db:accession`. Reactions need `rn:R#####` (0/621 held); a reaction is fully KEGG-identified for 148/621 (24%), fully ChEBI-identified for 176/621 (28%). KGML has no compartment, stoichiometry or evidence slot, so locations and quotes are lost by the format. The spec defines no rule for unmapped entities (`other` is listed then marked deleted in v0.7; only `group:undefined` is documented): 539/1095 compounds (49%) and 75/787 proteins (10%) would be unmapped. This is a decision for the owner, not a technical fact. Each KGML entry has one position, our tiles repeat entities per process (2031 redraw rows in 184/192 runs, DUPLICATION-ASSESSMENT c1), so a de-duplicating layout is needed.
- BioPAX is the only target that carries the evidence quote natively (Evidence/comment), unsigned interactions (MolecularInteraction) and needs no layout. SBGN-ML and GPML require geometry for every glyph, so they cannot be written without the invented layout; SBGN also expresses compartment nesting by drawing containment, which would assert nesting the payload does not hold.
- SBML core has no interaction object and (L3) no compartment `outside`; signs need the qual package, nesting needs comp/spatial or an isPartOf annotation (recalled).

Locked decisions that bind any exporter (cited from `docs/pwml_recovery_sprint/DECISIONS.md`):

- D-015 (:275-296): all network-dependent resolution is finished before the freeze; after it PWML and SBML exporters must not query external services, materialize identity fields, rename entities or reinterpret biology. So every database fill in D is pre-freeze work, and exporter-side "cheap fixes" may only read what is already in the payload/IR.
- D-023 (:859): in the sprint, `src/t2pw/sbml/` was outside every implementation boundary ("no SBML implementation, extension or refactor"). Whether that still binds the web-server/output track is not stated: QUESTION.
- D-034/D-035/D-036 and F-043 (FINDINGS.md:734): same name or same identifier is not identity; consolidation needs proof. The legacy SBML merges entities that share a mapped id (`_dedupe_entity_rows`, json_to_sbml.py:39-80); any new writer must not.
- D-099: the auto-state lifecycle is narrowly authorized; `__auto_state__`/"cell" is a payload fact that any compartment export will see (61% of runs).
- D-104 (+Amend. 1, 2): only Activation/Inhibition are exported to PWML; verb forms with a sign in the word are accepted; competition and other unsigned relations are never exported as signed; evidence and support marks go to human-facing outputs only. Applies to PWML; other formats may carry unsigned relations because they have a slot for them (owner to confirm).
- D-106: the UniProt/DrugBank requirement for proteins is not relaxed; an exporter may not fill an identity to rescue an entity.
- D-109: nested locations are merge-compatible; mapping never narrows a class name to a specific molecule. An exporter that adds ChEBI/KEGG ids must add class-level ones only.
- D-114 (+Amend. 1, 2): coverage = processes drawn / processes in the text pathway, target >= 75%; whatever is left out is listed with its reason. A new format inherits this reporting duty.
- D-117: genes are separate rows named "<SYMBOL> gene"; protein families become complexes with at most 3 species-matched members. Gene rows have no DNA/RNA type (0/12 PWML nucleic acids have nucleotide-type), which BioPAX/SBML would want.

## 6. WHAT TO DO (ranked by expected gain; none changes biology; all subject to the no-regression rule)

1. Carry the paper identity into the payload before the freeze: PMID/DOI/title (and the requested pathway name) as a top-level `metadata` block, which `ir.py:1352-1358` already reads. Gain: removes "Generated Pathway" (61/61), fills pathway name, description and `references`; every export format needs it (SBML model name, BioPAX Pathway/PublicationXref, GPML BiopaxRef, SBML isDescribedBy). Cost: small; risk to PWML output: the title and references change only text fields; test with the golden-seam tests. D-015 satisfied (done pre-freeze).
2. Keep process evidence in the IR (reaction, transport, interaction rows) so SBML notes, BioPAX Evidence and the existing text output read the same record. Gain: 621/621 reactions regain their quote for non-PWML formats; PWML unchanged (no slot). Risk: none to PWML if the writer ignores the new key.
3. Stop inventing in the IR where the payload is silent, as a flag rather than a value: `direction`, `stoichiometry` and `spontaneous` could be stored as `null`/"unstated" and the PWML writer keep today's defaults; other formats then omit the attribute (optional in SBML L3V2) instead of asserting 1 / irreversible. Gain: removes 623/623 + 246/246 invented values from non-PWML exports. Risk: PWML bytes must stay identical, so the writer keeps its defaults.
4. Write protein gene names (652/787 in payload) and synonyms into PWML (`gene-name`, synonyms), which PathWhiz stores. Gain: up to 83% of proteins gain a field; risk: low but it changes PWML, so it needs the owner's yes under the no-regression rule.
5. Transporter location: extract or derive the membrane for transporters before the freeze (22/22 sit in the first state today). It is a payload gap, so an exporter fix would be a post-freeze reinterpretation (D-015). Gain bound: affects the 42 transports in 62 runs only; needs a prompt/extraction change or a GO/UniProt location lookup.
6. Compartment nesting: store a `parent` for each location from a GO part_of table, pre-freeze, starting with the 15 names already tabulated (stage_one_ensemble.py:282-300) and extending to the 56 observed names. Gain: SBML (comp/isPartOf), SBGN, and the diagram; unknown pairs stay "not nested" (fail closed, D-109).
7. Order: derive an explicit step order from shared non-currency compounds (76% of reactions have a link; 14% of links run against list order) and store it as a layout hint, not as a paper claim; the diagram and GPML/SBGN need it, KGML/BioPAX/SBML do not.
8. Database fills before the freeze for export-only fields: ChEBI/KEGG for the 39% of compounds with none of the four ids, GO ids for locations (7% today), EC to reaction level, Rhea/KEGG R. Expected gain bound: KGML reaction coverage from 24% fully-identified reactions upward, limited by ID coverage of the compounds (G); no effect on PWML if written to new fields only.

Questions for the owner (not decided here):

- Q1. Does D-023's "no SBML implementation" boundary still apply, or may a new SBML/KGML writer be built outside `src/t2pw/sbml/`?
- Q2. KGML unmapped entities: omit the entry, write a placeholder entry with a documented non-KEGG name, or refuse the file? (The KGML spec has no rule.)
- Q3. May non-PWML formats carry the unsigned relations (139 binding-type and 14 regulation-type rows in the 152 runs) that D-104 Amend. 1 keeps out of PWML?
- Q4. Compartment granularity in SBML: one compartment per location, or per biological state (location x cell type x tissue x species)?
- Q5. Do "Unknown"-backed complexes (D-070, accepted for PWML) get exported to SBML/BioPAX as complexes with an unknown member, or as a bare entity?
