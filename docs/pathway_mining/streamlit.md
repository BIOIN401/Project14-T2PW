# Streamlit

Two ways in. Both render the same page and call the same controller.

## Standalone

```bash
export T2PW_PATHWAY_MINING_ENABLED=true
streamlit run src/t2pw/mining/ui/app.py
```

Opens the Pathway Mining page on its own, without loading the PWML pipeline page
at all.

## Inside the main app

```bash
export T2PW_PATHWAY_MINING_ENABLED=true
streamlit run src/t2pw/app/streamlit_app.py
```

A **Mode** radio appears in the sidebar. Selecting *Pathway mining
(experimental)* replaces the page.

### The hook

`src/t2pw/app/streamlit_app.py` gains this, immediately after
`st.set_page_config` and before `st.title`:

```python
try:
    from t2pw.mining.ui.entry import render_entry as _render_mining_entry

    if _render_mining_entry(st):
        st.stop()
except ImportError:  # pragma: no cover - the experimental package is optional
    pass
```

With the flag unset, `render_entry` reads one environment variable and returns
`False` **without** adding a sidebar control, touching session state or importing
anything from the mining package beyond `flags`. The default render of the PWML
page is byte-for-byte unchanged. `test_mining_ui.py` asserts the hook's shape and
its no-op behaviour.

The mining page uses its own `mining_*` session-state namespace, its own run
directories and its own vector index (in-memory by default), so it cannot
interfere with the pipeline page's state.

---

## The page

### Tab: Set up and run

**1. Starting inputs** — target species (required), topic, seed metabolites,
seed enzymes/proteins, seed reactions, papers (PMID / PMCID / DOI / URL, one per
line), file uploads (PDF or text), and an existing pathway-graph JSON path. All
composable.

Uploads are content-addressed: the same bytes always produce the same
`paper_id` (`FILE:<sha256[:16]>`), so re-uploading a PDF merges into the existing
record rather than creating a second citation for one paper.

**2. Budgets and stop conditions** — every budget from
[`limits_and_stopping.md`](limits_and_stopping.md), configurable without editing
`.env`. Environment variables supply the defaults the widgets start from.

**3. Model, retrieval and acquisition** — model slug, explicit fallback model
(empty means no fallback, so a failure stays loud), literature sources, passages
per query, vector backend, screening toggle, offline toggle, runs directory.

**4. Service preflight** — a button that runs the same checks the CLI runs, with
each result shown as ok / warn / fail and a verdict line. The probe checkbox
controls whether one tiny live model call and one embedding are attempted.

**5. Start** — runs synchronously with a progress bar and a rolling per-round
log. Heavy work is sequential by design; the page blocks until a limit is
reached. A run that does not complete normally is reported as a warning naming
the stop reason, never as a success.

An expander offers **Resume an interrupted run** from a run-directory path.

No secret is ever displayed. The preflight reports whether a key is present.

### Tab: Explore results

Pick any run under the runs directory, then:

| Sub-tab | Contents |
|---|---|
| **Overview** | completion banner, metrics, git commit + dirty state, primary-component rationale, full `report.md` |
| **Reaction network** | the biochemical view + filters + the round slider |
| **Paper / evidence network** | the provenance view + layer and paper filters |
| **Claims & evidence** | the flat audit table with status, graph-status and free-text filters |
| **Candidate subgraphs** | each quarantined component, its reason, its promotion history |
| **Conflicts & rejections** | every retained conflict with both positions; every rejected / incomplete / quarantined claim with its reason |
| **Logs, cost & exports** | cost breakdown, model calls (failures called out), round metrics, warnings, and download buttons for every artifact |

Details of the two network views, their filters and their guarantees are in
[`visualizations.md`](visualizations.md).

---

## Offline and dependency-free

The network renderer is vanilla SVG + JavaScript with a seeded, deterministic
force layout. It loads no CDN script, no external stylesheet, no image and makes
no network call — a test asserts the absence of `<script src`, `<link `,
`fetch(`, `XMLHttpRequest`, `import(` and `WebSocket` in the generated HTML. The
project needs no `networkx`, `plotly`, `pyvis` or `d3` dependency.
