# PathWhiz import harness

Branch `pwml/import-harness`. Code in `tools/pathwhiz_import/`, evidence in `docs/pwml_output/evidence/import/`. Run on 2026-11-10 and 2026-11-11 on a local copy of the real PathWhiz/SMPDB app. No T2PW pipeline or LLM run was made.

## 1. NOT established

1. **Whether any T2PW file imports into production PathWhiz.** The harness uses an empty database. Production already holds the compounds, proteins, species, cell types and locations, and the importer matches them with `find_by`. Here every such entity is created new, so validations that never fire in production fire (see 5). Failures below are therefore *candidate* production failures, not proven ones, and a pass would not have proved acceptance either.
2. **`success: true` means very little.** `PwmlParser#parse` returned success while its error hash held 2 to 111 messages for every T2PW file. 60 of 60 T2PW files come back `success: true`, 0 of 60 with no error messages.
3. **Only 2 of the 6 real PathBank files in `reference/` import**, and the other 4 fail for reasons in the importer, not in the harness (5.1). So there are 2 working controls, not 3.
4. **Screenshots show what the live viewer draws from the rows that survived**, on a database whose drawable-image files (S3) and compound structures (MolDB) are missing. Layout and colour are real; structure thumbnails are blank or placeholder.
5. Rendering was checked for 5 files only (2 controls, 3 T2PW). Not checked: the cached `/pathwhiz/pathways/PW...` page (needs a Sidekiq worker), the editor, export round trip (`/live/view/PW...pwml`).

## 2. Result in one table

| set | files | `success: true` | with no error messages | notes |
|---|---:|---:|---:|---|
| Controls (PathBank `reference/*.pwml`) | 5 | 2 | 0 | PW012926, PW741182 import with 2 and 16 messages; PW687775 and PW176888 crash on an importer bug (5.1); PW1.pwml is a stub-only file and fails with 81 messages. PW000001 not run (same vacuous-edge construct as the two crashes). |
| T2PW output (E2E-5 file + 59 canonical-run `.pwml`) | 60 | 60 | 0 | 5 to 111 messages per file, median 15 |

Rows: `evidence/import/controls_results.jsonl` (5 rows), `evidence/import/t2pw_results.jsonl` (60 rows), summary `t2pw_summary.md`. Per-file import time: median 14.5 s, max 230 s (chlorophyll file), total 1284 s, plus about 35 s Rails boot per batch.

### What "imported" meant, in rows (T2PW, 60 files, expected = elements written in the PWML, created = rows the importer made)

| element | expected | created | files with every expected element created |
|---|---:|---:|---:|
| compounds | 429 | 426 | 57/60 |
| compound locations | 599 | 598 | 59/60 |
| proteins | 238 | 218 | 40/60 |
| protein complexes | 232 | 167 | 39/60 |
| reactions | 245 | 155 | 31/60 |
| **transports** | **37** | **0** | (19 files carry transports) |
| **interactions** | **85** | **5** | |

Compounds and their locations survive (a pathway with only metabolites imports nearly whole). Everything that depends on biological states, protein complexes or enzymes is cut.

## 3. T2PW error causes, grouped (60 files)

Counts are error messages and the number of files that have at least one (`t2pw_summary.md` has the raw per-message table).

| cause (importer message) | files | messages | what it cuts |
|---|---:|---:|---|
| `bound_visualization: Bound can't be blank` | 60 | 168 | T2PW writes `<bound-visualization>` elements keyed by `biological-state-id` with no `<bound-id>`, and `<bounds>` is empty. PathWhiz needs a Bound for each. Every file, every time. PathBank controls carry none. Visible effect: no compartment membranes in the drawing. |
| `subcellular_location / cell_type / tissue: Ontology can't be blank; Ontology is invalid` | 53 | 151 | T2PW writes `<ontology-id nil="true"/>`. On an empty DB the row cannot be created; in production it would not match an existing row either (rows there have an ontology id). The biological states built on them then fail. |
| `interaction: needs exactly 1 left and 1 right element` | 27 | 80 | interactions need exactly one left and one right element; 5 of 85 interactions survive. Compare the ORCH-747 interaction rules in `qa.py` (not checked here whether the same files fail there). |
| `reaction: Spontaneous cannot be No if reaction has no enzymes` | 27 | 88 | enzyme (protein complex) was lost first, then the reaction. Reactions 155/245. |
| `protein_complex: Protein complex proteins is too short` | 21 | 65 | a complex whose proteins were not created. |
| `transport_element: Left/Right biological state can't be blank` then `transport: Transport elements is too short` | 19 | 37 + 37 | all 37 transports lost, cascade from the biological-state failures. |
| `protein: Species can't be blank` (protein id 9659, the PathBank Unknown sentinel) | 20 | 20 | and `species: Classification can't be blank` 20 files / 20 messages: the species row has no classification. |
| `nucleic_acid: Nucleotide type can't be blank` | 1 | 12 | one file. |

The cascade order seen in the traces: vocabulary rows (species, cell type, location, tissue) fail validation, so biological states resolve to nothing, so transports, element states and protein locations fail, then complexes and reactions lose their enzymes. That order is read from the error sequences, not tested by changing a file.

Parser exceptions (`errors[:base]`): 0 of 60 T2PW files.

## 4. Controls

* **PW012926** (Triacylglycerol Degradation): imported, 2 messages (`transport_transporter_visualization`, `sub_pathway_output`). 6 of 6 reactions, 13 of 13 proteins, 1 of 5 transports created. Drawn: cell, vacuole, mitochondrion, nucleus, 6 compound boxes, reaction arrows (`screenshots/control_PW012926.png`).
* **PW741182** (salt taste): imported, 16 messages (transport family). Drawn: mouth, taste bud, two cells, membranes (`screenshots/control_PW741182.png`).
* **PW687775, PW176888**: `NameError: undefined local variable or method 'vcv_node'`, rescued by the parser into `errors[:base]` ("Error parsing PWML, please check your input..."), `success: false`. Cause: `lib/pwml_parser.rb:291` in the `vacuous-edge-visualizations` loop calls `build_nested_visualization(vcv_node, ...)`; the loop variable is `vev_node` (copy-paste from the loop above). It is in committed `HEAD`, not in the owner's uncommitted change. Any PWML with a `<vacuous-edge-visualization>` fails; PW000001 has one too. A PathBank export containing one cannot be re-imported by this parser. I did not fix it (not my edit to make); the fix is one word.
* **PW1.pwml**: a stub file (bare `<interaction><id/></interaction>`), 81 messages, `success: false`.

## 5. What the screenshots show

Saved under `evidence/import/screenshots/` as `*.png` (the whole drawing fitted into the viewport; `*_view.png` copies are not committed) and `*.svg` (the live viewer's SVG, extracted from the page).

* `control_PW012926.png`, `control_PW741182.png`: complete PathBank-style pictures.
* `t2pw_E2E-5_PMC32185015.png`: the E2E-5 file (2 compounds, 2 proteins, 3 transports, 3 interactions in the PWML) renders as an **empty blue canvas**. 2 compounds (8 compound locations), 1 protein and 0 transports were created, and nothing is drawn (the SVG holds 106 elements, all of them background and frame; the reason a created compound is not drawn was not investigated).
* `t2pw_enterobactin_PMC12096016.png`: 5 reactions and 8 compounds are drawn as boxes, ellipses and arrows on a plain background. No cell outline, no compartments, a few overlapping arrows and an interaction drawn as a red bar over the pathway.
* `t2pw_wnt_pcp_PMC10425814.png`: protein-complex ellipses with arrows and one inhibition bar on a blue background, no cell, no labels for most nodes, nodes overlapping.

## 6. How it was set up

1. `git clone --local` of the owner's smpdb to `C:/t/pwi/smpdb`, then the owner's uncommitted change (`lib/pwml_parser.rb`, 10 lines) applied. Patch sha256 `bdeba2e483db28b72ffe2ee6f979dd064702a681e1af736c566e703a47a335c8`. The owner's checkout was only read (`git diff`, `git status`). It still shows only ` M lib/pwml_parser.rb`.
2. **No Docker build.** The machine already had the owner's built image `smpdb-app:latest` (Ruby 2.2.0, JDK 7) and gem volume `smpdb_bundle_cache` (501 MB, includes the private bitbucket gems moldbi, wishart, seq_search, tmic_banner, synonym_cleaner, devise fork). I used the image read-only and copied the gem volume into my own volume `t2pw-pwi_bundle`. Nothing was removed or modified in the owner's volume.
3. Own compose file, project `t2pw-pwi`: `db` (mysql:5.7, host port 3317), `app` (the image, sleeping; the Rails server is started inside with `docker compose exec -d`, host port 3010). No elasticsearch container, no `~/.ssh` mount. The `research-platform-*` containers were not touched.
4. `bundle check` passed; `rake db:create db:schema:load` loaded the 108-table schema into the empty DB. The 14 GB dump was **not loaded**.
5. Rendering needs the `visualization_templates` rows (every element references a template id up to 238) and `drawable_images` rows. Those two small tables were taken from the dump with a read-only `grep` of the `INSERT INTO` lines (`seed/seed_inserts.sql`, 77 KB, 192 and 260 rows). That is a read of the dump but not a load of it; flagged for the owner in section 8.
6. Each file is imported with `PwmlParser#parse(File.open(path))`, the call `pathways_controller#upload` makes, on a database truncated beforehand (every table except `users`, `sessions`, `schema_migrations` and the two seed tables; verified empty before each import). Truncating only the non-empty tables takes 0.2 s; it replaced a mysqldump restore.
7. Screenshots: Playwright (Chromium, `C:/t/e2e3-venv`), log in as the local admin, open `/live/view/<pw_id>`, extract the SVG, screenshot the viewer as opened and with the whole drawing fitted.

## 7. smpdb edits (every one; none in the owner's checkout)

All edits are in the copy `C:/t/pwi/smpdb`. **Nothing was changed in `C:/Users/Angad/Desktop/SummerBIOIN/smpdb`, nothing was committed in either.**

| # | file in the copy | kind | why |
|---|---|---|---|
| 1 | `lib/tasks/pwml_import.rake` | new file, 136 lines, tracked copy at `tools/pathwhiz_import/pwml_import.rake` (sha256 `2bf9a6bf...e88297`) | the headless entry: `rake "pwml:import[path]"` or `PWML_LIST=file rake pwml:import`; truncate, import, count rows per table, print `PWML_RESULT {json}`, exit 0/1 |
| 2 | `config/database.yml` | new, gitignored | db host `db`, user root, password `password` (local throwaway) |
| 3 | `config/moldb.yml` | new, gitignored, copy of `moldb.yml.sample` | boot |
| 4 | `lib/pwml_parser.rb` | the owner's 10-line uncommitted change, applied unchanged | so the harness tests the importer the owner is running |

The rake task contains three non-parser deviations, each marked in its comments and each a limit of the empty database, not a parser change:

* `user_id` is forced to a local admin (`admin@pwi.local`, created by the task) on every validation. Schema default for `user_id` is 11 (a production guest user), which does not exist here, so every new species, compound or cell type failed with "User can't be blank".
* `Tire::Index#store/update/remove/create/delete/refresh/import` are stubbed to return `{}`. Without elasticsearch the Tire after-save callback raises after the row is written; the parser swallows it and then drops the id mapping for that element, which silently empties everything downstream.
* `PWML_DEBUG=1` is set during the import so the parser prints the exception it otherwise hides; the task keeps only the `EXCEPTION in parse` line (`parser_exception` in the row).

Outside smpdb: `tools/pathwhiz_import/stubs/copyleaks_api.rb` (empty file on `RUBYLIB`; `lib/tasks/copy_leaks.rake` requires a gem that is not in the bundle, so every `rake` command aborts without it).

DB-only: the two seed tables (5 above) and one local user (`admin@pwi.local`, password `pwi-local-only`, role admin). Both live in the harness's own MySQL volume.

## 8. Dead ends and corrections

* The scout said the importer rejects a duplicate pathway name. In this version (line 156) it renames it with a timestamp. Clean state per file is still used.
* First run: `Gemfile` needs ssh git gems; solved by the existing volume, not by network access.
* First import: `User can't be blank`, then `Components must have at least one compounent`, then `undefined method '[]' for true`: three layers of the empty-DB problem listed in section 7. The last one made all 273 biological states of PW012926 fail silently; the first version of the harness reported that as a failed import with a misleading cause.
* Screenshot hang: the viewer polls `/visualization_templates/<id>` and `/drawable_images/<id>` forever when the tables are empty (404s every 2 s); loading the seed fixed it.
* A first page, `/pathwhiz/pathways/PW000001`, redirects to home: it shows a pre-rendered SVG file made by a Sidekiq worker (`CacheSuperPathwayVisualizationWorker`). The live viewer `/live/view/<pw_id>` draws in the browser from the database and needs no worker, so that is what the harness uses.
* Two screenshot-file names collided on the first harness version (same base name and folder), so one image overwrote another; tags now include the staged file index and path hash.
* One stray `python -` empty heredoc was typed once by me early in the session (it ran nothing); not repeated.
* A bounded run was cut by a Claude Code restart and by a shutdown overnight; its stale heavy lock was cleared by the coordinator. Per-file results are appended as they print, `--resume` skips finished files, so nothing was lost.

## 9. How to run

    python tools/pathwhiz_import/run_import.py FILE.pwml [FILE ...] [--list paths.txt] \
        [--screenshots FILE.pwml ...] --out tools/pathwhiz_import/out/run1 [--resume] [--down]

Wrap long batches in `docs/pwml_recovery_sprint/evidence/bounded_run.py --heavy-lock pwi --timeout 540 -- ...` and use `--resume` to continue after a cut. Roughly 25 s per file plus 35 s boot per batch of 12; screenshots add 1 to 2 minutes each. Stop afterwards with `docker compose -p t2pw-pwi stop` (volumes and images are kept). See `tools/pathwhiz_import/README.md`.

## 10. Limits

* Empty DB (section 1). The next step toward production fidelity is to seed the DB with the matching rows PathBank has: species with classification, cell types, subcellular locations and tissues with ontology ids, the compounds and proteins. Without it the harness cannot tell "T2PW wrote a bad row" from "PathBank already has a good one".
* The importer's `errors` hash is the only signal for dropped elements; the harness compares created rows to expected counts from the PWML text (regex over the sections), which is exact for the eight listed sections and silent about the rest.
* `ontology-id` and the species classification fields are what the writer emits as nil. Whether production PathWhiz would accept those rows is not tested here.
* Screenshots depend on S3 drawable images and MolDB structure images that are absent.

## 11. Open questions for the owner

1. **Is `lib/pwml_parser.rb:291` (`vcv_node` where `vev_node` is meant) known?** It breaks re-import of any PathBank export with a vacuous edge visualization (3 of 6 reference files). One-word fix, but it is your importer.
2. **Does production PathWhiz hold subcellular locations, cell types and tissues without an ontology id?** If not, T2PW must emit ontology ids (or only locations that exist) or the whole biological-state layer is lost (53 of 60 files). This needs an answer from someone with the production DB, or a read-only export of those tables to seed the harness.
3. **Should `<bound-visualization>` be written at all?** 60 of 60 files carry one with no `<bound-id>`; PathBank exports in `reference/` carry none (checked on 2: PW012926, PW741182).
4. OK to commit `tools/pathwhiz_import/seed/seed_inserts.sql` (rows from your dump: template names, sizes, user ids, no personal data)? It makes the harness runnable without the 14 GB file.
5. The harness hard-codes this machine's defaults (owner checkout path, Playwright venv). Fine for the owner's one-command use; say so if it should be made portable.
