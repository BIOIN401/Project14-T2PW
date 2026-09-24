# The feature flag

```
T2PW_PATHWAY_MINING_ENABLED=false     # the default, and the default when unset
```

Pathway mining is **experimental and off by default**. Enabling it does **not**
start a run; it only makes the entry points reachable.

## Three activation paths

| Path | How | Scope |
|---|---|---|
| Environment | `T2PW_PATHWAY_MINING_ENABLED=true` in the shell or `.env` | every invocation |
| CLI | `python -m t2pw.mining --enable-pathway-mining <command>` | one invocation |
| Streamlit | the "Enable pathway mining for this session" checkbox on the page | one browser session |

Precedence: an explicit override wins **in both directions**. Passing
`--enable-pathway-mining` enables the feature even with the variable unset;
un-ticking the Streamlit box disables it even with the variable set to `true`.

Accepted true values: `1 true TRUE yes on enabled`. Everything else, including
an unrecognised string, is **false** — the gate never opens on a typo.

## What "off" guarantees

With the flag unset and no override:

* `mining_enabled()` returns `False` after reading one environment variable.
* `require_mining_enabled()` raises `PathwayMiningDisabled` with the activation
  instructions; `MiningController.__init__` calls it before doing anything else.
* The CLI prints those instructions to stderr and exits **3**, having created no
  run directory.
* `t2pw.mining.ui.entry.render_entry(st)` returns `False` without adding a
  sidebar control or touching session state, so the main Streamlit app renders
  exactly as it did before.
* Importing `t2pw.mining` imports only `t2pw.mining.flags`. It does **not**
  import `chromadb`, `openai`, `streamlit` or `t2pw.llm.client`, does not
  construct a provider client, does not read a cache and does not reach the
  network. `test_mining_feature_flag.py::test_importing_the_package_is_cheap_and_touches_nothing`
  asserts this in a subprocess.
* No background work and no expensive initialization happens anywhere.

## Where it is recorded

Every run manifest carries:

```json
"feature_flags": {
  "T2PW_PATHWAY_MINING_ENABLED": "true",
  "enabled_via": "cli_flag",      // or "environment" | "streamlit" | "test"
  "offline": false
}
```

so an artifact always says how the feature was turned on for that run.

## Other environment variables

All optional; all recorded in the manifest. See
[`limits_and_stopping.md`](limits_and_stopping.md) for the budget variables.

| Variable | Default | Meaning |
|---|---|---|
| `PATHWAY_MINING_MODEL` | `deepseek/deepseek-v4-flash` | requested model slug |
| `PATHWAY_MINING_FALLBACK_MODEL` | *(empty)* | explicit, logged fallback; empty means a failure stays loud |
| `PATHWAY_MINING_TEMPERATURE` | `0.0` | deterministic extraction |
| `PATHWAY_MINING_MAX_TOKENS` | `12000` | per-call output ceiling: hidden reasoning plus the answer for one passage batch (was 4000; evidence 032) |
| `PATHWAY_MINING_MAX_ATTEMPTS` | `3` | shrink-and-retry ladder length |
| `PATHWAY_MINING_REQUEST_TIMEOUT` | `360` | seconds per attempt (was 120; 12,000 tokens at the slowest observed 41 tokens/s is about 293 s) |
| `PATHWAY_MINING_STRUCTURED_OUTPUT` | `true` | request a JSON schema where supported |
| `PATHWAY_MINING_ACQUISITION_SOURCES` | `europepmc,ncbi` | literature sources (discovery only) |
| `PATHWAY_MINING_RETRIEVE_TOP_K` | `8` | passages per query |
| `PATHWAY_MINING_VECTOR_BACKEND` | `memory` | `memory` keeps the run out of the production index |
| `PATHWAY_MINING_OFFLINE` | `false` | never contact a literature service |
| `PATHWAY_MINING_SCREEN_PAPERS` | `true` | run the eligibility screen |
| `PATHWAY_MINING_RUNS_DIR` | `runs_mining` | where run directories go |
| `PATHWAY_MINING_REQUIRE_EMBEDDINGS` | `false` | make a missing embeddings service a preflight **failure** |

No secret is ever stored on a config object, written to an artifact, or shown in
the UI. The preflight reports whether an API key is *present*, never its value.

## Merging this branch later

The feature is built so the branch could be merged with the feature left
disabled. The checklist:

1. `T2PW_PATHWAY_MINING_ENABLED` must remain absent from the committed `.env`
   template and from any default.
2. The `streamlit_app.py` hook stays guarded by `render_entry`'s own gate and by
   `try/except ImportError`.
3. The SMOKE suite must still measure its expected count (it did: **508
   passed**, see `evidence/001-smoke-feature-disabled.json`).
4. Nothing in `src/t2pw/mining/` may be imported by a production module.
