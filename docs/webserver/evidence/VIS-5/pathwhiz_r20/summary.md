# PathWhiz import summary

files 7 | imported (success=true) 7 | imported with no error messages 2 | failed 0 | total 661s

| count | files | cause |
|---:|---:|---|
| 8 | 3 | subcellular_location: Ontology can't be blank; Ontology is invalid |
| 4 | 1 | nucleic_acid_location: Nucleic acid can't be blank |
| 3 | 1 | sub_pathway_output: Element can't be blank |
| 3 | 1 | nucleic_acid: Nucleotide type can't be blank; Nucleotide type is not included in the list |
| 3 | 1 | transport_element: Biological state must have unique biological states |
| 3 | 1 | transport: Transport elements is too short (minimum is 1 character) |
| 3 | 1 | transport_transporter_visualization: Transport transporter can't be blank |
| 3 | 1 | transport_visualization: Transport can't be blank |
| 2 | 2 | reaction_compound_visualization: Edge can't be blank |
| 2 | 1 | interaction_right_element: Element can't be blank |
| 2 | 1 | interaction: Interaction right elements is the wrong length (should be 1 character) |
| 2 | 1 | interaction_nucleic_acid_visualization: Nucleic acid location can't be blank |
| 2 | 1 | interaction_visualization: Interaction can't be blank |
| 2 | 1 | edge: ActiveRecord::StatementInvalid: Mysql2::Error: Data too long for column 'path' at row 1: INSERT INTO `edges` (`path`, `visualization_template_id`, `zindex`, `pa |
| 2 | 1 | transport_compound_visualization: Edge can't be blank |
| 1 | 1 | edge: ActiveRecord::StatementInvalid: Mysql2::Error: Data too long for column 'path' at row 1: INSERT INTO `edges` (`path`, `visualization_template_id`, `hidden`, `op |
| 1 | 1 | protein_complex: Protein complex proteins is too short (minimum is 1 character) |
| 1 | 1 | transport_transporter: Protein complex can't be blank |
| 1 | 1 | edge: ActiveRecord::StatementInvalid: Mysql2::Error: Data too long for column 'path' at row 1: INSERT INTO `edges` (`path`, `visualization_template_id`, `options`, `z |
| 1 | 1 | protein_complex_visualization: Protein complex can't be blank |
| 1 | 1 | transport_transporter_visualization: Protein complex visualization can't be blank; Transport transporter can't be blank |
