# Setup

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
streamlit run src/t2pw/app/streamlit_app.py
```

The LLM client reads `.env` from the project root. Local LM Studio is the default:

```text
LLM_PROVIDER=local
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=meta-llama-3.1-8b-instruct
```

OpenRouter is also supported:

```text
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=...
```

Optional PathBank database-backed mapping uses:

```text
PATHBANK_ID_SOURCE=hybrid
PATHBANK_DB_HOST=...
PATHBANK_DB_PORT=3306
PATHBANK_DB_USER=...
PATHBANK_DB_PASSWORD=...
PATHBANK_DB_SCHEMA=pathbank
```

## Configuring the resolution DB for generation

The **resolution database** is the live PathBank/PathWhiz MySQL instance that
supplies the *canonical* compound and species names. Emitting the DB's canonical
name (rather than the extraction/LLM name) is what lets PathWhiz import the
generated PWML: the importer matches existing rows by `name` + external ids, so a
name mismatch causes a duplicate INSERT that collides on a unique key
(`compounds.hmdb_id`, `species.taxonomy_id`) and fails.

Set the same `PATHBANK_DB_*` variables as above in the project `.env` (or export
them in the environment). Nothing is hardcoded in source — every value is read
from these variables via `t2pw.config.resolution_db_config`:

```text
PATHBANK_DB_HOST=<host>
PATHBANK_DB_PORT=3306
PATHBANK_DB_USER=<readonly-user>
PATHBANK_DB_PASSWORD=<password>
PATHBANK_DB_SCHEMA=pathbank
# optional timeouts (seconds)
PATHBANK_DB_CONNECT_TIMEOUT=6
PATHBANK_DB_READ_TIMEOUT=20
PATHBANK_DB_WRITE_TIMEOUT=20
```

`t2pw.config.ensure_dotenv_loaded()` loads `.env` before any resolution query, so
the connection works regardless of import order. Previously the DB was only
configured as a side effect of importing the LLM client; a generation path that
skipped that import reported `db_not_configured` and silently fell back to
offline (often non-canonical) names.

### Preflight collision-risk warning

When a generation run has **neither** a reachable resolution DB **nor** an offline
index (`data/pathwhiz_id_db.json`) covering an entity, `build_pwml_ir` now:

- logs a `WARNING` on the `t2pw.pwml.ir` logger, and
- attaches `report["preflight"]` plus a `noncanonical_names_collision_risk`
  warning listing the at-risk compounds/species.

This makes the "names may collide on import" condition loud instead of silent. If
you see it, either configure the resolution DB (above) or refresh the offline
index (below).

### Refreshing the offline name index (maintainer step)

The offline index lets *offline* runs still emit canonical names. The bundled
`data/pathwhiz_id_db.json` was built from a small `.pwml` sample (few compounds,
no species). To repopulate its `compounds` and `species` sections from the live
resolution DB:

```powershell
python -m t2pw.pwml.refresh_name_index            # writes data/pathwhiz_id_db.json
python -m t2pw.pwml.refresh_name_index --out other.json
```

It reads connection settings from the same `PATHBANK_DB_*` variables and preserves
the file's other sections. This is a manual maintenance step — it is never invoked
during normal generation or tests.

## Deterministic species names

Species uniqueness in PathWhiz is on both `name` and `taxonomy_id`, so the same
`taxonomy_id` must always emit the same `name` or it collides with whatever the DB
stored first. `t2pw.pwml.ir` applies this precedence per species record:

1. live resolution-DB name (applied upstream during stage-2 hydration);
2. offline name-index species-by-taxonomy / pathbank-species-id, and the curated
   `SPECIES_CREATE_DEFAULTS` (both already canonical);
3. *(reserved)* a canonical name derived from the NCBI taxonomy id — no reliable
   offline source is bundled today, so this remains a follow-up;
4. a deterministic normalization of the candidate name (`_deterministic_species_name`)
   for any taxonomy-identified species: it truncates at the first sub-species/strain
   rank marker (`subsp.`, `str.`, `var.`, …) and strips trailing strain-code tokens,
   always keeping at least the genus + species epithet.

So `"Herbaspirillum huttiense IAM 15032"`, `"Herbaspirillum huttiense"`, and
`"Herbaspirillum huttiense subsp. huttiense IAM 15032"` all deterministically emit
`"Herbaspirillum huttiense"`. Species with **no** `taxonomy_id` are treated as
novel and keep their extraction name verbatim. The chosen name need not equal the
DB's exact string (the importer handles that) — the requirement here is
determinism across runs.
