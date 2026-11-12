# PathWhiz import harness

Imports PWML files into a **local copy of the real PathWhiz/SMPDB app** (Rails 4.2, Ruby 2.2.0, MySQL 5.7)
and reports, per file, whether the importer (`lib/pwml_parser.rb`, the code behind `POST /pathways/upload`)
accepted it, which errors it raised, and how many rows it created. Optionally opens the imported pathway in
the PathWhiz live viewer and saves a screenshot and the SVG. Full write-up: `docs/pwml_output/IMPORT-HARNESS.md`.

## Run it

    python tools/pathwhiz_import/run_import.py A.pwml B.pwml --list more_paths.txt \
        --screenshots A.pwml --out tools/pathwhiz_import/out/run1 [--resume] [--down]

Output in `--out`: `results.jsonl` (one row per file: file, success, errors, created counts, expected counts
from the PWML, seconds, screenshot), `summary.json`, `summary.md` (pass rate, error causes with counts),
`screenshots/*.png|svg`. Exit 0 only if every file imported.

Per file: ~20 to 50 s import (+ about 35 s Rails boot per batch of `--chunk` files, default 12), screenshots
about 1 to 2 min each. Run long batches in pieces with `--resume`. Per project rules, wrap it in
`docs/pwml_recovery_sprint/evidence/bounded_run.py --heavy-lock pwi --timeout 540 -- ...`.

## What it needs (first run sets the rest up)

* Docker, the image `smpdb-app:latest` (built from smpdb's own Dockerfile) and the gem volume
  `smpdb_bundle_cache` (the bundle of that image). The gems include private bitbucket gems, so they are copied
  from the owner's volume into the harness volume `t2pw-pwi_bundle` (read-only copy). No ssh key is mounted.
* `--smpdb-src` (default `C:/Users/Angad/Desktop/SummerBIOIN/smpdb`): read only. The harness clones it into
  `tools/pathwhiz_import/work/smpdb` and applies its uncommitted tracked changes with `git diff HEAD`. It never
  writes to the original.
* Playwright for Python for screenshots (`PWI_PLAYWRIGHT_PYTHON`, default `C:/t/e2e3-venv/Scripts/python.exe`).

## Files

* `docker-compose.yml`: project `t2pw-pwi`: `db` (mysql:5.7, host port 3317), `app` (idle container; rails server
  is started inside it on host port 3010 when a screenshot is needed). No elasticsearch.
* `pwml_import.rake`: copied into the working copy as `lib/tasks/pwml_import.rake`. `pwml:import[path]` or
  `PWML_LIST=file rake pwml:import`; truncates every data table before each file; prints `PWML_RESULT {json}`.
* `screenshot.py`, `stubs/copyleaks_api.rb` (empty stub: `lib/tasks/copy_leaks.rake` requires a gem not in the
  bundle), `seed/seed_inserts.sql` (rows of `visualization_templates` and `drawable_images`, see below).
* Stop the stack: `docker compose -p t2pw-pwi stop` (keeps volumes and images).

## Limits you must know

* The database is **empty** (schema only). Production PathWhiz already holds the compounds, proteins, species,
  cell types and locations, and the importer matches them with `find_by`. Here every such entity is created
  new, so validations that never fire in production (for example "Ontology can't be blank" on a cell type) fire.
  A failure here is not proof of a production failure; a pass is not proof of a production pass.
* The rake task forces `user_id` to a local admin and stubs Tire (elasticsearch) index writes. Neither is a
  change to the parser.
* `success: true` only means `PwmlParser#parse` returned success. Its `errors` hash can still be non-empty and
  elements can be silently dropped (compare `created_main` with `expected`).
* Rendering needs the 192 `visualization_templates` rows. They come from the dump (two small tables, extracted
  with grep, the dump itself is not loaded). Drawable-image files live on S3 and are not available; compound
  structure images need the MolDB server and are blank.
