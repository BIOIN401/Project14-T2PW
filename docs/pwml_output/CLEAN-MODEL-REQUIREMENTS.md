# CLEAN MODEL REQUIREMENTS (input for the next step)

A format-neutral model that sits between the frozen payload and every exporter (PWML, SBML, KGML, BioPAX, SBGN-ML, GPML). Field-level content only; no code design. Grounded in assessments 06 (fields, gaps), 02 (mapping), 04 (accuracy), 07b (licences). Field names below are descriptive, not final.

## 1. Rules that govern every field

1. **Provenance label on every field and every id.** One of:
   - `paper` : stated by the paper, with the verbatim quote and location (section or page) where the pipeline has it.
   - `db` : filled from a database lookup; name the source, the accession, the retrieval date and the match type (exact label, exact synonym, cross-reference, InChIKey).
   - `inferred` : derived by the pipeline (orthology, step order from shared compounds, nesting from GO); never presented as paper-stated.
   - `default` : a convention value (for example PWML stoichiometry 1). Marked `unstated` in non-PWML formats; PWML keeps its defaults byte-identical.
   - `unknown` : the field is absent. Never fill with "Unknown" placeholders as if they were entities (138 of 415 complexes are all-Unknown today).
2. **Frozen payload stays the source of truth.** The frozen payload equals `final_mapped.json` (hash matched in 152 of 152 runs). The clean model is built from it after the freeze (D-034, D-036: no silent change after the freeze). Anything database-filled after the freeze is added in a separate layer with its own label and never overwrites a paper-stated or pipeline-frozen value.
3. **Identity is per molecule, location is a separate row** (DUPLICATION-ASSESSMENT, gold structure). One entity record per molecule or form; compartments and drawings are location rows. A shared accession is evidence, not proof of identity (D-035, F-043): hordatine A and C, compound 1 and 2, holo and apo EntB.
4. **No class or complex gets a member's accession** (04: about 48 of 192 id flags). Class, family, complex and modified-form names keep their own entity type; ids only at the level the paper names.
5. **Never lose what PWML cannot hold.** Unsigned relations, binding-type interactions (139 of 352 payload interactions), evidence quotes (621 of 621 reaction quotes lost in the IR today), gene names and synonyms (652 of 787 proteins) are kept in the clean model and simply not written by the PWML exporter.

## 2. Locked decisions the model must respect

- D-034, D-035, D-036, F-043: identity and merge rules; no silent merge after the freeze.
- D-099, D-114: PathWhiz import rules and PWML export contract (UniProt/DrugBank on proteins, spontaneous when enzyme-less). The PWML exporter's output must not change except where the owner approves (title/references, gene names).
- D-104 (incl. Amendment 1) and D-109: reaction merge on database id; unsigned relations stay out of PWML. Whether other formats may carry them is an open owner question.
- D-106, D-108, D-117: protein id strictness, species must match, family terms. The clean model may carry an `inferred` suggestion field but no suggestion may sit in an id field. Homology is not recommended.
- D-015: transporter state rule (an exporter must not move a transporter to a membrane; fix upstream).
- D-023: "no SBML implementation" boundary; owner must say whether it still binds.
- PRODUCT_CONTRACT outranks tests: a correct drop is not a loss.

## 3. Required content, by object

Legend: P = paper-stated, DB = database-fillable, I = inferred, D = default/unstated, Gap = lost today.

### Pathway (document)
| Field | Label | Status today |
|---|---|---|
| title, short name | P | Gap: 0 of 152 payloads carry one; PWML says "Generated Pathway" in 61 of 61 |
| source paper PMID/PMCID/DOI, authors, year | P / DB (NCBI) | Gap: no `metadata` block in 152 of 152 |
| organism (name, NCBI taxon id) | P + DB (NCBI Taxonomy) | partial |
| pathway class (metabolic, signalling, transport) | I | PWML always "Metabolic" |
| creation tool/version, run date, pipeline mode | D | present in run files |
| licence and attribution lines per database used | D | Gap; needed for KEGG/ChEBI/UniProt (07b) |

### Compartment / location
| Field | Label | Status |
|---|---|---|
| name (as in paper) | P | present as strings |
| GO cellular-component id | DB (QuickGO) | Gap; 15 names tabulated of 56 observed |
| parent compartment (nesting) | DB/I (GO is_a, part_of) | Gap: no parent field in any payload (130 of 152 runs have two or more locations) |
| membrane vs lumen (a transporter must sit on a membrane) | I | Gap: 1 of 22 transporters in a membrane location |
| cell/tissue/organism context | P | partial |

### Compound (small molecule)
| Field | Label | Status |
|---|---|---|
| name as in paper, synonyms | P | present |
| ChEBI id, InChIKey, formula, SMILES | DB (ChEBI, current API) | 28% of units have no id; 39% of those have an exact ChEBI/KEGG name hit; 33% an InChIKey |
| KEGG compound id | DB (KEGG, restricted, see 07b) or user-supplied | 49% have none |
| PubChem/HMDB/CAS cross-references | DB | optional |
| charge/protonation form, stereochemistry | P / DB | gap |
| role (currency, cofactor) | I | PWML invents `currency=false` on 620 of 620 |
| per-location presence | P/I | present as location rows |

### Protein / enzyme / transporter / channel
| Field | Label | Status |
|---|---|---|
| name as in paper, gene name(s), synonyms | P | gene names never written to PWML |
| organism and NCBI taxon | P + DB | present |
| UniProt accession | DB (UniProt) | 38.6% of units (168 of 435) without; accession only at paper's level (rule 4) |
| mapping status and match type | D | present (unmapped, placeholder, quarantined) |
| EC number | DB (Rhea/UniProt) | in IR, not in payload |
| isoform / modified form | P | gap |
| `inferred` suggestion (name lookup, ortholog) | I | proposed only, never in the id field |
| role in process (enzyme, transporter, channel, regulator) | P | present |

### Complex
| Field | Label | Status |
|---|---|---|
| name, members (each a protein/compound entity reference, stoichiometry optional) | P | members point at a placeholder "Unknown" in 138 of 415 complexes |
| Complex Portal / PDB id | DB | optional |

### Reaction
| Field | Label | Status |
|---|---|---|
| reactants, products (entity refs, location refs) | P | present |
| stoichiometry | P else D | bare strings in payload; PWML writes 1 on 623 of 623 members |
| reversibility/direction | P else D | PWML writes ">" on 246 of 246 |
| catalysts/enzymes (entity refs), spontaneous flag | P else D | spontaneous=false invented on 246 of 246 |
| Rhea id, EC, KEGG reaction id, GO molecular function | DB | Gap |
| evidence: verbatim quote, location in paper | P | quotes in payload, dropped by the IR |
| order/step index and links to neighbouring steps | I (shared compounds link 188 of 246) | `reaction_order` in 11 of 152 payloads, used by no exporter |

### Transport
| Field | Label | Status |
|---|---|---|
| cargo, from-location, to-location, transporter/channel | P | present |
| mechanism (facilitated, active, channel gating) | P | not extracted; channel gating transport type missing (03) |
| membrane crossed | I (from nesting) | Gap |
| evidence quote | P | dropped in the IR (106 of 106) |

### Interaction / relation (regulation, binding, inhibition, phosphorylation)
| Field | Label | Status |
|---|---|---|
| source and target (declared entities, not free text) | P | 15 interactions have undeclared endpoints |
| type from a controlled vocabulary (activation, inhibition, binding, phosphorylation, modulation unsigned) | P (+ I normalisation) | free text, 29 normalised spellings in 62 IR runs, 42 in 152 payloads |
| sign (positive, negative, unsigned) | P | "regulates" has no sign; only 115 of 352 map to Activation/Inhibition |
| SBO/MI term | DB | Gap |
| evidence quote | P | dropped (351 of 352) |

### Cross-cutting
- **Quarantine and removed-entity record**: keep what was removed and why (identity, gene/protein clash, undeclared endpoint), so exports can report omissions. 40% of Stage-1 interactions were quarantined.
- **Mapping record**: each id carries its mapping rule and flag (`wrong_identity` candidates, narrowing). About 5.7% of evaluable ids are wrong or narrowed (04).
- **Layout is not in the clean model.** PWML drawing detail (coordinates, tiles, colours, state ids) is generated by the PWML exporter; diagrams for other formats come from their own layout step.

## 4. Which online source fills which gap

| Gap | Source | Licence / constraint |
|---|---|---|
| Compound ids, InChIKey, formula | ChEBI (current API; the older service returned HTTP 500 on 2026-11-10) | CC BY 4.0 |
| Protein accession, gene names, EC, taxon, sequence | UniProt | CC BY 4.0 |
| Reaction ids, EC, participants | Rhea | CC BY 4.0 |
| Compartment ids and nesting | GO / QuickGO | open; attribute |
| SBO terms | EBI OLS | open |
| Taxonomy | NCBI | "places no restrictions" |
| Orthology (suggestions only) | Ensembl, OrthoDB | "imposes no restrictions" (Ensembl); not recommended |
| KEGG compound/reaction/gene ids, KGML | KEGG | REST is "made available only for academic use by academic users", 3 calls per second; service providers need a licence (07b). Default: do not call live. Use user-supplied KEGG ids. Owner decision pending |
| PathBank/PathWhiz ids | PathBank database | read-only DB route is out of scope here; note as gap |

## 5. Export-readiness the model must support (summary from 06)

Fillable rows (yes/partial/no): SBML 9/10/11 today, 19/10/1 after pre-freeze work; KGML 4/6/5 to 7/8/0; BioPAX 3/6/2 to 7/4/0; SBGN-ML 2/3/1 to 2/4/0; GPML 0/4/1 to 3/2/0. SBML needs: compartments with nesting, species with compartment, reactants/products/modifiers, SBO terms, MIRIAM annotations, notes. KGML needs: entry ids, an owner rule for unmapped entities, reaction substrates/products and types, relation types and subtypes. The legacy `src/t2pw/sbml/json_to_sbml.py` writes L3V1 text, invents initialAmount=1, hard-codes reversible=false and adds an "Unknown" protein for components-less complexes; do not reuse its invention.

## 6. Acceptance for the clean model (for the next step to test)

- Every field above has a label; no `default` appears in a non-PWML export without the `unstated` marker.
- Building the PWML from the clean model reproduces the current PWML byte-for-byte on the golden-seam tests, except fields the owner has approved.
- The model round-trips: frozen payload -> clean model -> payload fields compared, zero paper-stated field lost.
- Numbers to beat: 0 -> 152 of 152 payloads with title/PMID; 0 of 621 -> 621 reaction quotes retained; 29 interaction spellings -> a controlled vocabulary with an explicit unsigned class.
