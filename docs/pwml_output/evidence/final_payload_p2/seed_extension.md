# Harness seed extension (vocabulary + protein 9659)

Tree: C:/t/pwi/t2pw/tools/pathwhiz_import/ (uncommitted, nothing committed anywhere).

## Added
- `seed/vocab_inserts.sql`, 510,973 bytes, sha256 d8a8b26dce5cdef18999d19d5fcf4a503d1596636e30acf2d9154ee7b5ef07b6. Five INSERT lines.
- Rows: cell_types 43, subcellular_locations 77, tissues 53, species 4016, proteins 1 (id 9659, 'Unknown', uniprot 'Unknown', species_id 4, user_id 11).

## Commands (read-only on C:/Users/Angad/Desktop/SummerBIOIN/smpdb/db_dumps/smpdb_remote_dump.sql, 14,334,668,709 bytes; never loaded or cat'd; each grep ~9 s)
    grep -a -E '^INSERT INTO `(cell_types|species|subcellular_locations|tissues)` ' <dump> > seed/vocab_inserts.sql
    ROW=$(grep -a -o "(9659,'Unknown','[^)]*)" <dump> | head -1)
    printf 'INSERT INTO `proteins` VALUES %s;\n' "$ROW" >> seed/vocab_inserts.sql
Protein tuple has the 15 columns of db/schema.rb `proteins` in the same order (checked).

## How it is loaded
The harness truncates every data table (except users, sessions, schema_migrations, visualization_templates, drawable_images) before EACH file, so a one-time load would be wiped. Hence re-seed per file:
- `run_import.py` `ensure_config`: one added line copies `seed/vocab_inserts.sql` to `<work>/lib/tasks/pwml_vocab_seed.sql` (mounted at /app).
- `pwml_import.rake`: `pwml_reset!` now calls new `pwml_seed_vocab!` after the clean-check; it runs each line of that file with `connection.execute` (FK checks off). The `before` counts are taken after seeding, so `created`/`created_main` still mean rows the importer added (seeded rows are not counted).
- Only these two existing files were touched, solely to wire the seed in. No smpdb, parser or other harness edits.

## Verification
- Docker project t2pw-pwi existed but was stopped (Exited). I started only the `db` container (normal harness start, light), truncated the 5 tables, loaded the file with the harness's own mysql command (`mysql ... --default-character-set=utf8 < file`, 0.6 s): no errors; counts 43/4016/77/53/1; protein 9659 reads back (Unknown, Unknown, species 4). So FK/NOT NULL columns load fine (no FK constraints hit; user_id 11 is stored as-is, harmless per the rake's user_id forcing). Then stopped `db` again.
- NOT tested: the rake path (`pwml_seed_vocab!` via ActiveRecord execute) and a full file import, because that needs the app container and a Rails boot. Expected fine (largest statement ~450 KB, max_allowed_packet 4 MB) but unverified. The harness DB tables now hold the seeded rows (harmless; truncated and re-seeded per file).
- Mistake: I accidentally typed one `python -` with empty stdin once; it ran nothing and the task was stopped.
