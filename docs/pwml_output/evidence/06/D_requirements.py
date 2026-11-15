"""Generates D_export_requirements.md. Status per row: today / after cheap export-side fixes (no new extraction, no network) / after pre-freeze fills
(DB lookups or upstream carry-through, all done BEFORE the freeze per D-015). Y = fillable, P = partly, N = not, X = format has no slot (excluded from matrix).
Origin: PAPER (paper-stated), DB (database-filled), INV (invention if we filled it; do not), CODE (constant or derived from a spec rule)."""
import collections

FMT = {}

FMT['SBML L3V2 core (+ groups/qual/layout packages where noted)'] = [
 # field, have (stage, M numbers), database can fill, origin, today, cheap, upstream, note
 ('sbml level=3 version=2 header, model id/metaid', 'legacy writer emits L3V1 as text', '-', 'CODE', 'P', 'Y', 'Y', 'sbml/json_to_sbml.py:711-715 (L3V1, text, "avoids libsbml")'),
 ('model name / title', 'NO: payload has no title; PWML says "Generated Pathway" 61/61', 'PubMed title from the PMID that the request carries (request.json), paper-stated', 'PAPER', 'N', 'N', 'Y', 'needs PMID+title carried into the payload before the freeze (D-015)'),
 ('model annotation: bqmodel:isDescribedBy PubMed id', 'NO (0/152 payloads carry a bibliographic field)', 'PubMed/Identifiers.org', 'PAPER', 'N', 'N', 'Y', 'same carry-through as above'),
 ('compartment id, name', 'YES: 495 biological states / 414 location rows in 152 runs (final_mapped); 56 distinct location names', '-', 'PAPER', 'Y', 'Y', 'Y', 'one compartment per location, or per state (location x cell type x tissue x species); owner choice'),
 ('compartment constant, spatialDimensions, size', 'no size in paper', '-', 'INV if set', 'Y', 'Y', 'Y', 'leave optional attrs unset; legacy sets none but sboTerm'),
 ('compartment sboTerm (physical compartment SBO:0000290, recalled)', 'legacy uses SBO:0000240 = "material entity" (OLS, fetched 2026-11-10), not a compartment term', 'SBO static', 'CODE', 'P', 'Y', 'Y', 'sbml/json_to_sbml.py:744'),
 ('compartment annotation: GO cellular component (bqbiol:is)', 'canonical_id 28/414 location rows (7%); GO ids appear only in protein enrichment blobs', 'GO via OLS/QuickGO name lookup (56 distinct names, 15 have a curated container)', 'DB', 'N', 'P', 'Y', 'static table for the ~15 common names is a cheap fix; rest needs a lookup'),
 ('compartment NESTING (cell > cytosol > organelle > membrane)', 'NO: 0/152 runs have any parent/part_of field; 130/152 runs have >=2 locations', 'GO part_of (table of 15 names in pipeline/stage_one_ensemble.py:282-300, merge-only)', 'DB', 'N', 'P', 'P', 'L3 core has no `outside`; nesting needs comp/spatial package or bqbiol:isPartOf annotation; DECISIONS D-109 only uses the table for merging'),
 ('cell type / tissue / organism of a compartment', 'YES as state attributes (cell_type 41, tissue 45 of 495 states); ontology ids nil 20/20, 15/15', 'Cell Ontology, UBERON, NCBI Taxonomy (taxonomy 205/209 present)', 'PAPER+DB', 'P', 'P', 'Y', ''),
 ('species per (entity, compartment)', 'YES: element_locations 1033 compound + 671 protein rows; entity may sit in several states', '-', 'PAPER', 'Y', 'Y', 'Y', 'a species id must be (entity, compartment); legacy does this (sbml_species_id)'),
 ('species name', 'YES', '-', 'PAPER', 'Y', 'Y', 'Y', ''),
 ('species compartment (required), hasOnlySubstanceUnits, boundaryCondition, constant (required)', 'compartment YES; the three flags are constants', '-', 'CODE', 'Y', 'Y', 'Y', 'legacy writes constant=false, boundary=false'),
 ('species initialAmount / initialConcentration', 'NO', 'not available from a paper pathway', 'INV', 'N', 'N', 'N', 'legacy writes initialAmount="1" for every species (json_to_sbml.py:855) = invented; omit'),
 ('species sboTerm (simple chemical 247, polypeptide 252, non-covalent complex 253, recalled)', 'derivable from entity class', 'SBO static', 'CODE', 'P', 'Y', 'Y', 'legacy: 245 (macromolecule, fetched) for proteins AND complexes (:856,:898), 247 compounds (:963)'),
 ('species annotation: ChEBI / KEGG / HMDB / PubChem (bqbiol:is)', 'compounds: chebi 632/1095 (58%), kegg 556 (51%), hmdb 510 (47%), pubchem 609 (56%), any of 4: 672 (61%)', 'ChEBI/PubChem by name or InChI; KEGG via ChEBI xref (pre-freeze)', 'DB', 'P', 'P', 'P', 'legacy uses old urn:miriam: URNs (:757-785); identifiers.org URIs are current'),
 ('species annotation: UniProt, gene symbol, EC', 'proteins: uniprot 712/787 (90%), gene_name 652 (83%), EC 511 (65%)', 'UniProt (EC, NCBI Gene, Ensembl xrefs)', 'DB', 'P', 'P', 'Y', 'PWML writer drops gene names (writer.py:1260-1269); an SBML writer can read them from IR'),
 ('species annotation for genes (nucleic acids)', 'name only; 39 rows; DNA/RNA type, GenBank absent', 'NCBI Gene/UniProt gene symbol; D-117 names rows "<SYMBOL> gene"', 'DB', 'N', 'N', 'P', ''),
 ('complex composition (members, stoichiometry)', 'YES: 415 complexes, member stoichiometry int in 454 members; BUT 138/415 (33%) have only Unknown-sentinel members and 241 are single-protein wrappers', 'UniProt family search (D-117 2.)', 'PAPER / INV for wrappers', 'P', 'P', 'P', 'groups package or bqbiol:hasPart; an "Unknown" member must not be exported as a real member (legacy invents one, json_to_sbml.py:842-852)'),
 ('reaction id, name', 'YES (name 620/621)', '-', 'PAPER', 'Y', 'Y', 'Y', ''),
 ('reaction reactants / products (species refs)', 'YES: inputs/outputs resolved to entities, 0 unresolved names in 621', '-', 'PAPER', 'Y', 'Y', 'Y', ''),
 ('reaction stoichiometry', 'NO: participants are bare name strings 1544/1544; IR sets 1 for 623/623', 'Rhea / KEGG REACTION balanced equation via ChEBI (only when all participants mapped: ChEBI-complete reactions 176/621)', 'DB (not paper-stated)', 'N', 'N', 'P', 'extraction schema has no coefficient field; legacy hard-codes 1 (json_to_sbml.py:1031)'),
 ('reaction reversible', 'NO: IR ">" for 246/246; 9/621 evidence quotes mention reversibility', 'Rhea direction / KEGG / EC', 'DB or PAPER', 'N', 'N', 'P', 'legacy hard-codes reversible="false" (:997); in L3V2 the attribute is optional, so omitting is honest'),
 ('reaction compartment', 'YES for 490/621 (79%); others fall back to the first state (ir.py:1827-1847)', '-', 'PAPER / INV for fallback', 'P', 'P', 'Y', ''),
 ('modifiers: enzymes (catalyst), activators, inhibitors with SBO 460 / 459 / 20 (recalled)', 'YES: 695 catalyst, 21 activator, 18 inhibitor, 3 cofactor, 1 transporter in 738 modifiers', 'SBO static', 'PAPER', 'Y', 'Y', 'Y', 'PWML cannot tell activator from catalyst; SBML can'),
 ('reaction annotation: EC (bqbiol:isVersionOf), Rhea, KEGG R', 'EC only per protein (65%); Rhea 0 and KEGG R 0 in 152 payloads', 'Rhea (by ChEBI equation / EC), KEGG REACTION', 'DB', 'N', 'P', 'P', 'EC of the enzyme protein can be attached to the reaction without a lookup'),
 ('reaction sboTerm (biochemical 176, transport 185, recalled)', 'type known (reaction vs transport)', 'SBO static', 'CODE', 'Y', 'Y', 'Y', 'legacy :995'),
 ('transport (as reaction between per-compartment species)', 'cargo + from/to state YES (105/106); 20/42 IR transports have no transporter; 22/22 transporters sit in the FIRST state (1/22 in a membrane)', 'GO/UniProt subcellular location of the transporter', 'PAPER / INV for transporter state', 'P', 'P', 'P', 'transporter membrane is a payload gap AND an IR default (ir.py:2334-2338); not fixable in an exporter'),
 ('interactions (activation, inhibition, binding, regulation)', 'YES: 352 rows; signed verbs 115 (33%), binding family 139 (39%), catalysis-like 79, unsigned regulation 14, other 5', 'SBO interaction terms; qual package for signs', 'PAPER', 'N', 'P', 'P', 'SBML core has no interaction object. Options: qual-package transitions (sign), modifier-only reactions, or notes. Legacy SBML drops all interactions (0 mentions of "interaction" in json_to_sbml.py). D-104 Amend. 1: unsigned relations are never exported as signed'),
 ('notes / evidence quote (XHTML) per process', 'YES in payload 100% of reactions/transports; LOST in the IR (ir.py:2078-2089)', '-', 'PAPER', 'N', 'Y', 'Y', 'IR must keep the quote; D-104 Amend. 2: exact record goes to human-facing outputs'),
 ('layout package (x, y, w, h) / render info', 'drawing detail from IR (invented layout)', '-', 'INV (drawing)', 'P', 'P', 'P', 'optional; legacy writes pathwhiz: annotations instead'),
]

FMT['KEGG KGML 0.7.x'] = [
 ('pathway name / org / number / title / image / link', 'no KEGG map exists for a new paper pathway; title NO', 'KEGG organism code from taxonomy (205/209 present)', 'CODE/DB', 'N', 'P', 'Y', 'name must be a synthetic non-KEGG id; owner decision whether such a file is acceptable'),
 ('entry id (unique int)', 'local integers', '-', 'CODE', 'Y', 'Y', 'Y', ''),
 ('entry name, compound: cpd:C#####', 'kegg id 556/1095 compounds (51%)', 'KEGG via ChEBI/PubChem xref', 'DB', 'P', 'P', 'P', ''),
 ('entry name, gene/enzyme: org:geneid, ko:K#####, ec:', 'UniProt 90%, EC 65%; KEGG gene id 0, KO 0', 'UniProt cross-reference to KEGG; EC direct', 'DB', 'P', 'P', 'Y', ''),
 ('entry type (compound, gene, enzyme, group, ortholog, map, brite, other)', 'derive from entity class', '-', 'CODE', 'Y', 'Y', 'Y', 'complexes -> group with <component>'),
 ('UNMAPPED entities', 'compounds without any KEGG id 539/1095 (49%); proteins without UniProt 75/787 (10%); genes 39 rows no ids', '-', '-', 'X', 'X', 'X', 'KGML docs (fetched) define no rule: type "other" is listed then said deleted in v0.7; only group:undefined is documented. OWNER QUESTION'),
 ('entry graphics: x, y, width, height, type, fgcolor, bgcolor, name', 'IR location tiles; BUT one entry has ONE position, our model has a tile per (process, state): 2031 redraw rows over 184/192 runs (DUPLICATION-ASSESSMENT c1)', '-', 'INV (drawing)', 'P', 'P', 'P', 'needs a de-duplicating layout; coordinates are drawing, not biology'),
 ('entry group + component (complexes)', 'YES members; 241/415 are single-protein wrappers, 138/415 Unknown-backed', '-', 'PAPER / INV', 'P', 'P', 'P', 'collapse wrappers; never emit Unknown'),
 ('relation entry1, entry2', 'YES for 352 interactions (endpoints resolved)', '-', 'PAPER', 'Y', 'Y', 'Y', ''),
 ('relation type: ECrel, PPrel, GErel, PCrel, maplink', 'derivable from endpoint classes (protein-protein, protein-gene, protein-compound)', '-', 'CODE', 'Y', 'Y', 'Y', ''),
 ('relation subtype activation / inhibition', '115/352 final interactions have a signed verb; 43/131 in the 62 IR runs', '-', 'PAPER', 'P', 'P', 'P', ''),
 ('relation subtype binding/association (unsigned)', '139/352 are binding-family words that PWML cannot export', '-', 'PAPER', 'N', 'Y', 'Y', 'KGML can carry them without inventing a sign; consistent with D-104 Amend. 1'),
 ('relation subtype expression / repression, phosphorylation +p, state change, indirect effect', 'words exist as free text ("represses" 2, "phosphopantetheinylates" 2); gene-target role judged by D-117', 'verb table', 'PAPER', 'N', 'P', 'P', 'only an exact verb table (no stemming), as in interaction_types.py'),
 ('reaction entry: name rn:R#####', 'NO (0/621)', 'KEGG REACTION via compounds / EC', 'DB', 'N', 'N', 'P', 'reactions with every participant KEGG-identified: 148/621 (24%)'),
 ('reaction type reversible / irreversible', 'NO (default)', 'KEGG', 'DB', 'N', 'N', 'P', 'KGML says it reflects drawn direction'),
 ('reaction substrate / product (cpd ids)', 'compounds with KEGG id only (51%); 24% of reactions are fully identified', 'as above', 'DB', 'P', 'P', 'P', ''),
 ('compartments', 'KGML has no compartment element', '-', '-', 'X', 'X', 'X', 'locations are lost by the format'),
 ('stoichiometry, evidence, citations', 'KGML has no slot', '-', '-', 'X', 'X', 'X', 'lost by the format'),
]

FMT['BioPAX Level 3'] = [
 ('Pathway: displayName, dataSource (Provenance), xref PublicationXref (PMID)', 'no title; PMID only in request.json', 'PubMed', 'PAPER', 'N', 'N', 'Y', ''),
 ('SmallMolecule + SmallMoleculeReference xref ChEBI/KEGG/HMDB/PubChem', 'ids 58/51/47/56%', 'ChEBI, PubChem', 'DB', 'P', 'P', 'P', ''),
 ('Protein + ProteinReference xref UniProt, organism BioSource (taxonomy)', 'UniProt 90%, taxonomy 98%', 'UniProt', 'DB', 'Y', 'Y', 'Y', ''),
 ('Complex: component + componentStoichiometry', 'YES with the wrapper/Unknown caveats', '-', 'PAPER', 'P', 'P', 'P', ''),
 ('Dna / Rna for genes', 'name only; type unknown', 'NCBI Gene', 'DB', 'P', 'P', 'P', ''),
 ('cellularLocation (CellularLocationVocabulary xref GO)', 'name 100%, GO 7%', 'GO', 'DB', 'P', 'P', 'Y', 'BioPAX has no compartment nesting object either; nesting only through GO relations'),
 ('BiochemicalReaction: left/right + Stoichiometry objects, conversionDirection, eCNumber, spontaneous', 'participants YES; stoichiometry/direction NO (default); EC per protein', 'Rhea/EC', 'DB', 'P', 'P', 'P', ''),
 ('Transport / TransportWithBiochemicalReaction (left in compartment A, right in B)', 'cargo+states YES', '-', 'PAPER', 'Y', 'Y', 'Y', 'transporter location gap as above'),
 ('Catalysis (controller, controlType, controlled)', 'YES 695 catalysts, activators 21, inhibitors 18', '-', 'PAPER', 'Y', 'Y', 'Y', 'controlType ACTIVATION / INHIBITION available'),
 ('Control / TemplateReactionRegulation / MolecularInteraction (unsigned)', 'YES for interactions incl. unsigned', '-', 'PAPER', 'P', 'Y', 'Y', 'the best fit for unsigned binding'),
 ('Evidence (EvidenceCode, comment = quote) on every process', 'quote in payload 100%; lost at IR', '-', 'PAPER', 'N', 'Y', 'Y', 'BioPAX is the format that can hold the quote natively'),
]

FMT['SBGN-ML Process Description'] = [
 ('glyph class: simple chemical, macromolecule, nucleic acid feature, complex (nested member glyphs)', 'entity classes YES', '-', 'PAPER', 'Y', 'Y', 'Y', ''),
 ('compartment glyph and nesting by containment (bbox) / compartmentRef', 'compartment names YES; nesting NO', 'GO part_of', 'DB', 'N', 'P', 'P', 'nesting must be drawn, so an invented layout would assert a containment the paper did not state'),
 ('process node (process / omitted process) with consumption, production arcs', 'reactions YES; stoichiometry label on arc NO (all 1)', '-', 'PAPER', 'Y', 'Y', 'Y', ''),
 ('catalysis, stimulation, inhibition, necessary stimulation, modulation arcs', 'enzymes YES; signed interactions 33%; unsigned regulation 14 = modulation', '-', 'PAPER', 'P', 'P', 'P', 'SBGN has a native unsigned "modulation" arc'),
 ('bbox (x, y, w, h) for every glyph and arc geometry (REQUIRED)', 'only the drawing layout from the IR', '-', 'INV (drawing)', 'P', 'P', 'P', 'layout is mandatory in SBGN-ML; unlike SBML/BioPAX the format cannot be written without it'),
 ('annotations (RDF in extension), state variables, units of information', 'ids YES; no state variables (phospho-state not extracted)', 'UniProt/ChEBI', 'DB', 'P', 'P', 'P', ''),
]

FMT['GPML (WikiPathways 2013a+)'] = [
 ('Pathway: Name, Organism, Data-Source, Comment', 'organism YES; name NO', '-', 'PAPER', 'P', 'P', 'Y', ''),
 ('DataNode: TextLabel, Type, Xref (Database, ID), Graphics CenterX/CenterY/Width/Height (REQUIRED)', 'labels/ids YES; geometry = IR layout', 'Ensembl/ChEBI/UniProt', 'DB + INV (drawing)', 'P', 'P', 'P', ''),
 ('Interaction: Points, Graphics ArrowHead (Arrow, TBar, mim-binding...), Xref', 'Activation/Inhibition and binding word YES', '-', 'PAPER', 'P', 'Y', 'Y', 'mim-binding carries unsigned binding'),
 ('Group (complexes), Label, Shape (compartment as drawn shape)', 'complexes YES; compartment as shape needs layout', '-', 'PAPER / INV', 'P', 'P', 'P', ''),
 ('BiopaxRef / PublicationXref citations', 'NO PMID in payload', 'PubMed', 'PAPER', 'N', 'N', 'Y', ''),
]

out = ['# D. Export requirements, field by field', '',
       'Corpus for coverage numbers: 152 canonical runs (final_mapped.json, 1095 compounds, 787 proteins, 415 complexes, 621 reactions, 106 transports, 352 interactions); IR numbers: the 62 runs with a PWML IR.',
       'Status columns: today = fillable with what the pipeline writes now; cheap = after export-side changes that only read existing payload/IR data (no network, no new extraction); upstream = after pre-freeze work (carrying PMID/title through, database lookups before the freeze, D-015). Y fillable, P partly, N not, X format has no slot (excluded from the matrix).',
       'Origin: PAPER paper-stated, DB database-filled, CODE constant/derived from the spec, INV would be an invention (do not fill).',
       'Facts about SBML L3V2/SBO terms marked "recalled" were not re-read this session; KGML structure and SBO:0000240/0000245 labels were fetched (kegg.jp/kegg/xml/docs, EBI OLS) on 2026-11-10; BioPAX/SBGN/GPML rows are from memory of the specs and need a spec check before implementation.', '']
mat = []
for fmt, rows in FMT.items():
    out.append(f'## {fmt}')
    out.append('')
    out.append('| field | do we HAVE it (stage, measured) | can a DATABASE fill it | origin | today | cheap | upstream | note / locked decision |')
    out.append('|---|---|---|---|---|---|---|---|')
    c = collections.Counter(); n = 0
    for r in rows:
        out.append('| ' + ' | '.join(x.replace('|', '/') for x in r) + ' |')
        if r[4] == 'X':
            continue
        n += 1
        for lab, v in zip(('today', 'cheap', 'upstream'), r[4:7]):
            c[(lab, v)] += 1
    out.append('')
    mat.append((fmt, n, c))
out.append('## Summary matrix (rows of the tables above, X rows excluded; Y/P/N counts; a row counts once per column)')
out.append('')
out.append('| format | rows | today Y/P/N | after cheap fixes Y/P/N | after pre-freeze work Y/P/N |')
out.append('|---|---|---|---|---|')
for fmt, n, c in mat:
    f = lambda lab: '/'.join(str(c[(lab, v)]) for v in 'YPN')
    out.append(f'| {fmt} | {n} | {f("today")} | {f("cheap")} | {f("upstream")} |')
open('C:/t/ev-oa/payload/D_export_requirements.md', 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out[-9:]))
