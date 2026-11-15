# P5 re-run of the P4 ontology measurement

Run 2026-11-13, live OLS4 + Cellosaurus. Pass 1 = P4's input (names only). Pass 2 = the context of a stored run (species, paper text, states).

| kind | name | P4 | pass 1 (P4 input) | pass 2 (run context) |
|---|---|---|---|---|
| subcellular_locations | inner mitochondrial membrane | GO:0005743 exact_synonym | GO:0005743 (OLS4, exact_synonym) | GO:0005743 (OLS4, exact_synonym) |
| subcellular_locations | nerve fiber | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |
| subcellular_locations | apical | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |
| subcellular_locations | cytoplasmic leaflet of the IM | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |
| subcellular_locations | periplasmic leaflet of the IM | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |
| subcellular_locations | etioplast | GO:0009513 exact_label | GO:0009513 (OLS4, exact_label) | GO:0009513 (OLS4, exact_label) |
| subcellular_locations | microsomal fraction | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |
| subcellular_locations | saliva | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |
| subcellular_locations | stroma | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |
| subcellular_locations | synaptic cleft | GO:0043083 exact_label | GO:0043083 (OLS4, exact_label) | GO:0043083 (OLS4, exact_label) |
| subcellular_locations | synaptic vesicle | GO:0008021 exact_label | GO:0008021 (OLS4, exact_label) | GO:0008021 (OLS4, exact_label) |
| subcellular_locations | synaptic vesicles | unresolved | GO:0008021 (OLS4, exact_label, form 'synaptic vesicle') | GO:0008021 (OLS4, exact_label, form 'synaptic vesicle') |
| subcellular_locations | thylakoid membrane | GO:0042651 exact_label | GO:0042651 (OLS4, exact_label) | GO:0042651 (OLS4, exact_label) |
| subcellular_locations | inner membrane (IM) | (new) | unresolved: no_exact_match | resolved by the vocabulary map (PathWhiz row, no lookup needed) |
| subcellular_locations | outer membrane (OM) | (new) | unresolved: no_exact_match | resolved by the vocabulary map (PathWhiz row, no lookup needed) |
| subcellular_locations | outer membrane | (new) | GO:0019867 (OLS4, exact_label) | resolved by the vocabulary map (PathWhiz row, no lookup needed) |
| subcellular_locations | inner membrane | (new) | unresolved: no_exact_match | resolved by the vocabulary map (PathWhiz row, no lookup needed) |
| cell_types | epidermal cell | CL:0000362 exact_label | CL:0000362 (OLS4, exact_label) | CL:0000362 (OLS4, exact_label) |
| cell_types | Type II cell | unresolved | unresolved: no_exact_match [cellosaurus: no_exact_match] | CL:0002286 (OLS4, exact_label, via paper phrase 'type ii taste cell') |
| cell_types | Type III cell | unresolved | unresolved: no_exact_match [cellosaurus: no_exact_match] | CL:0002285 (OLS4, reviewed_synonym, via paper phrase 'type iii taste cell') |
| cell_types | Type III cells | unresolved | unresolved: no_exact_match [cellosaurus: no_exact_match] | unresolved: id_used_by_other_row |
| cell_types | erythroid | unresolved | unresolved: no_exact_match [cellosaurus: no_exact_match] | unresolved: no_exact_match [cellosaurus: no_exact_match] |
| cell_types | erythroid cells | unresolved | unresolved: no_exact_match [cellosaurus: no_exact_match] | unresolved: no_exact_match [cellosaurus: no_exact_match] |
| cell_types | geniculate ganglion neuron | unresolved | unresolved: no_exact_match [cellosaurus: no_exact_match] | unresolved: no_exact_match [cellosaurus: no_exact_match] |
| cell_types | MG-63 | unresolved | unresolved: no_exact_match [cellosaurus: no_species_context] | Cellosaurus CVCL_0426 (no CL id) |
| cell_types | Saos-2 | unresolved | unresolved: no_exact_match [cellosaurus: no_species_context] | Cellosaurus CVCL_0548 (no CL id) |
| cell_types | granulosa cells | (new) | resolved by the vocabulary map (PathWhiz row, no lookup needed) | resolved by the vocabulary map (PathWhiz row, no lookup needed) |
| tissues | bone marrow | BTO:0000141 exact_label | BTO:0000141 (OLS4, exact_label) | BTO:0000141 (OLS4, exact_label) |
| tissues | central nervous system | BTO:0000227 exact_label | BTO:0000227 (OLS4, exact_label) | BTO:0000227 (OLS4, exact_label) |
| tissues | cotyledon | BTO:0000300 exact_label | BTO:0000300 (OLS4, exact_label) | BTO:0000300 (OLS4, exact_label) |
| tissues | flower | BTO:0000469 exact_label | BTO:0000469 (OLS4, exact_label) | BTO:0000469 (OLS4, exact_label) |
| tissues | hypocotyl | BTO:0000613 exact_label | BTO:0000613 (OLS4, exact_label) | BTO:0000613 (OLS4, exact_label) |
| tissues | petal | BTO:0001040 exact_label | BTO:0001040 (OLS4, exact_label) | BTO:0001040 (OLS4, exact_label) |
| tissues | root | BTO:0001188 exact_label | BTO:0001188 (OLS4, exact_label) | BTO:0001188 (OLS4, exact_label) |
| tissues | young leaf | unresolved | unresolved: no_exact_match | unresolved: no_exact_match |

Pass 1: ontology id for 14 of 30 names (P4: 13 of 30); a Cellosaurus accession but no CL id for 0 more.

Pass 2: ontology id for 16 of 30 names (P4: 13 of 30); a Cellosaurus accession but no CL id for 2 more.

Notes: pass 2 'Type III cells' is refused (id_used_by_other_row) because run 3's states also name the exact row 'Type III cell', which claimed CL:0002285 first; 'Type II cell' resolves only because the paper text of PMC7059786 is available (the stored e2e runs that hold the name have none). Cellosaurus ids are not counted as ontology ids: PathWhiz cell_types need CL:+7 digits.
