# Temp Files Bootstrap

These are the temp/runtime files this project expects.

## Required for SBML motif retrieval

- `tmp/sbml_motif_index.json` (default path used by Streamlit)
- `tmp/sbml motif index.json` (alternate filename with space, if your teammate expects that exact name)

Both are included in the repo as valid empty index files.

## Optional DB snapshot inputs (for local DB analysis only)

Put PathBank export files under:

- `tmp/pathbank_dump/`

Common files used in this project:

- `01_relevant_tables.csv`
- `02_columns_dictionary.csv`
- `03_foreign_keys.csv`
- `04_row_counts.csv`
- `05_table_desc.csv`
- `pathbank_schema_target.sql`
- `pathbank_samples_target.sql`

These are not required for normal Streamlit app runs unless you are doing DB/schema analysis workflows.

## Runtime outputs

Most audit/mapping/SBML reports are generated in a temporary directory at runtime and exposed via Streamlit download buttons; they are not required as pre-existing repo files.
