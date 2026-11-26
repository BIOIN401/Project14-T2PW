# VIS-5 structural diff: PathBank reference PWMLs vs our PWML output

References: PW000001 (human, mitochondrion), PW000779 (E. coli TCA, Gram-negative envelope), PW012911 (plant,
chloroplast), PW000017 (human, organs). Ours: ent-item6 and lipid-a from replay r5 (layout v2 at f418ebcc).
Tool: `C:/t/vis-pw5-work/scripts/survey.py` (record counts, field sets, value histograms per record type; raw
`survey.json`, side-by-side `survey_cmp.txt`), `scripts/lbl.py` (labels, membranes, art, canvas), `scripts/edges10.py`.

Already the same (no gap): top-level section order of `pathway-visualization` (41 sections, identical order, ours carries
every section PathBank has); absolute canvas coordinates for every record; compound box template 81 at 200x190,
zindex 10, `font-size regular`, `hidden false`; protein locations zindex 8; membrane-visualization records (closed
`M`+8 `C` rounded-rectangle path, `complete-membrane true`, zindex 6, PathBank template per organelle, Gram-negative
envelope 1 / 291 / 1 about 120 and 250 units apart); label-location template 235, width 200, height 15, zindex 20;
drawable-element-location (bacterium 146 at scale 1.0, zindex 2, degree 0, native width/height); zoom-visualization
(template 39 left / 4 top, zindex 4, empty label); edge zindex 18, reaction edges template 5, transport edges 83;
compartments drawn only as membranes + labels (no bound-visualizations in either).

| # | field / record | PathBank convention | ours (r5) | gap | fix (where) |
|---|---|---|---|---|---|
| G1 | edge `path` | one cubic `M p0 C c1 c2 p3` per edge (PW000779 47 of 54, 2 with two cubics; the rest are hidden template-10 cofactor ticks) | right-angle grid routes of 3-6 cubics when the single curve would cross a box (lipid-a 18, enterobactin 7, coenzyme-a 5, sour-taste 4 ...) | multi-segment orthogonal detours | `layout.relayout` edge section: more single-cubic candidates (bent control points) and a smooth 2-cubic detour before the grid router |
| G2 | label-location placement | a compartment with its own membrane is labelled ON its membrane line (label box centred on the line, 400-700 units in from the corner: "Outer Membrane", "Peptidoglycan", "Inner Membrane", "Chloroplast", "Plasma Membrane", "Cell Wall"); the space inside gets its own label ("Cytoplasm", "Periplasm", "Stroma", "Cytosol") | one label per drawn state, inside the box top-left; envelope lines unlabelled | membranes unnamed; the inside label costs a 150-unit strip on top of every box | `layout.relayout` label emission + `pad_top`: organelle/membrane boxes label on their line, envelope lines labelled from the PWML's own location names (Outer/Inner Membrane) and "Peptidoglycan" |
| G3 | label scale | 1.3-2.2 (1.6 most) | 2.0 | none (in range) | keep |
| G4 | chromosome art (92) | scale 1.6, a centre piece of the cytoplasm, with its own label "Bacterial Chromosome" | scale 0.4-1.0 in whatever corner is free, no label | small, unlabelled | `cell_art.decorate` (scales up to 1.6) + label; `layout` keeps a larger room |
| G5 | organelle cluster art (90/91) | scale 1.6-2.4 | 0.6-1.4 | small | `cell_art.decorate` scale cap |
| G6 | nucleic-acid-location template | 96 tRNA (120x115) / 12 "Nucleic Acid" / 225+ genes; never a protein template | template 2 ("Enzyme Monomer I", a ProteinLocation template) 150x70 | PathWhiz draws our genes/promoters with an enzyme template | `layout.relayout`: nucleic-acid-location 2 -> 12 ("Nucleic Acid", NucleicAcidLocation, same 150x70) |
| G7 | currency compounds | small templates (9 cofactor tag 100x25, 42-46/59-62 pills 50x30, 47-65 icons 78x78), one location per reaction (H2O x4 in PW000779) | pills only when the writer picks a small template; lipid-a: all 47 locations template 81; one shared location per compound and state | box-size cofactors; shared locations draw long edges to a single ATP | template choice is the writer's `compound_templates.select_compound_template_id`, and new per-reaction locations are new records: both change compared PWML records -> **open question** (not done) |
| G8 | enzyme cofactors | `protein_complex_compound_visualizations` + template-9 compound tags + hidden template-10 edges under the enzyme (Fe2+, FAD, PLP) | empty `protein_complex_compound_visualizations`; our complexes carry no compound members | needs content (complex cofactors) | **open question**, no biology added |
| G9 | protein-location template | per complex shape: 6 dimer 160x80, 8 tetramer 140x85, 17 hexamer, 136 multimer, 2 monomer, 76 transporter | always 2 (monomer 150x70) | every enzyme a monomer, every transporter an enzyme ellipse | writer template choice from subunit counts (content-adjacent) -> **open question** |
| G10 | protein-location `label-type` | `subunit` everywhere | `subunit` and `protein` mixed | minor | not changed (compared field) |
| G11 | edge arrow encoding | `option:start_arrow` ... child elements | JSON `<options>` child | both import (VIS-1 A/B) | none |
| G12 | sub-pathway-visualization | 2-3 per pathway (links to other PathBank pathways) | none | content (pathway links) | out of scope |
| G13 | leaflet / sub-membrane states | never drawn as boxes; a membrane is one band | lipid-a: 6 thin-line (220) boxes for leaflets / "E. coli inner membrane" variants | clutter of tiny boxes | `layout.relayout`: a leaflet / membrane-variant state with its own membrane parent drawn as a region inside that membrane's box, no outline |
| G14 | structure of a compound | PathBank's compound record carries the structure itself: `moldb-smiles`, `moldb-inchi`, `moldb-inchikey`, `moldb-formula`, masses, IUPAC (PW000779 Water: `O`, `InChI=1S/H2O/h1H2`), plus the ids; PathWhiz draws the box picture from its structure database (moldb) | ids only (`pubchem-cid`, `chebi-id`, `hmdb-id`, `kegg-id`, `cas`); every `moldb-*` field empty; many intermediates have no id at all | PathWhiz has no structure of its own for our compounds unless it matches an id | adding SMILES/InChI is new compound content -> **open question** (not done) |

Structure-image question (PO item 2): PathBank's compound records carry the structure (moldb-smiles / inchi / inchikey / formula) next to the ids; ours carry the ids and leave every moldb-* field empty, and many intermediates (most chlorophyll intermediates, glucosylceramide, cysteinopantetheine) have no id at all. Filling moldb-* from the ChEBI/PubChem id would be new compound content: recorded as an open question, not done.
