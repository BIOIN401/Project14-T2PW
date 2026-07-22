# T2PW — Agent Instructions

## Project Goal

Convert plain-text pathway descriptions into **SBML Level 3** files that can be imported and rendered by **PathWhiz** (PathBank's visualisation tool). The pipeline is called **T2PW (Text-to-PathWhiz)**.

---

## High-Level Pipeline

```
User text
    ↓ Stage 1 LLM extraction + Stage 2A inference
Merged JSON (pathway schema)
    ↓ Stage 2B annotation-only ID mapping (cache enabled; no wrappers/cleanup)
Mapped JSON
    ↓ Stage 3 normalisation (process_normalizer.py)
Normalised JSON
    ↓ Stage 4 audit / gap-fill (audit_json_llm.py, gap_resolver.py)
Audited JSON
    ↓ Stage 5 curate + Stage 6 wrapper-enabled remap (cache bypassed)
Final mapped JSON  ← contains mapped_ids: {hmdb, kegg, chebi, uniprot, …}
    ↓ Stage 7 optional enrichment + Stage 8 export
PWML file (primary) / SBML file (legacy)
```

The Streamlit front-end (`src/t2pw/app/streamlit_app.py`) orchestrates all
steps. It exposes separate Stage 2B and Stage 6 mapping payloads/reports so the
pre-normalization mapping result is not confused with the post-audit remap.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/t2pw/app/streamlit_app.py` | Streamlit UI + pipeline orchestration |
| `src/t2pw/pipeline/process_normalizer.py` | Normalises the mapped JSON; contains strict Stage 3 gates |
| `src/t2pw/pipeline/stage_contracts.py` | Structural/semantic boundary adapters and centralized runtime-schema rollout mode |
| `src/t2pw/pipeline/payload_models.py` | Non-mutating recursive runtime payload-shape validation |
| `src/t2pw/sbml/json_to_sbml.py` | Builds the legacy SBML file from mapped JSON |
| `src/t2pw/mapping/map_ids.py` | Resolves external IDs at Stage 2B and Stage 6 |
| `src/t2pw/mapping/enrich_entities.py` | Adds optional Stage 7 xrefs via PathBankDB/APIs |
| `src/t2pw/pipeline/qa_graph.py` | Builds a connectivity graph for validation |
| `src/t2pw/curation/audit_json_llm.py` | LLM-based audit of the normalized JSON |
| `src/t2pw/curation/gap_resolver.py` | Fills missing reactions/entities during audit |

---

## Database

The pipeline connects to a **MySQL** PathBank database (`pathbank` schema by default). Credentials are entered in the Streamlit sidebar and passed as `db_config` dicts throughout the pipeline.

### Relevant tables

| Table | Key columns | Used for |
|-------|------------|---------|
| `compounds` | `id`, `hmdb_id`, `kegg_id`, `chebi_id`, `pubchem_cid`, `drugbank_id` | Compound PathWhiz ID lookup |
| `proteins` | `id`, `uniprot_id` | Protein PathWhiz ID lookup |
| `biological_states` | `id`, `subcellular_location_id` | Compartment → biological state ID |
| `subcellular_locations` | `id`, `name` | Compartment name → ID |
| `reaction_elements` | `reaction_id`, `element_id`, `type`, `element_type` | Reaction PathWhiz ID lookup |

### Environment variables (set in Streamlit or OS)

```
PATHBANK_DB_HOST
PATHBANK_DB_PORT   (default 3306)
PATHBANK_DB_USER
PATHBANK_DB_PASSWORD
PATHBANK_DB_SCHEMA (default "pathbank")
```

`build_sbml()` in `json_to_sbml.py` accepts a `db_config` dict and sets these env vars if not already set. `_load_pathwhiz_db()` reads them and queries MySQL; falls back to `pathwhiz_id_db.json` if MySQL is unavailable.

---

## SBML Output Format

The SBML must match PathWhiz's import expectations:

### Required namespace declarations on `<sbml>`
```xml
xmlns:pathwhiz="http://www.spmdb.ca/pathwhiz"
xmlns:bqbiol="http://biomodels.net/biology-qualifiers/"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
```

**Important:** libsbml auto-assigns an arbitrary prefix (e.g. `ns3`) to the `bqbiol` namespace. `_inject_root_namespaces()` in `json_to_sbml.py` detects and renames this to `bqbiol` using `str.replace()` (not regex — libsbml output is deterministic with double-quoted attributes).

### Species annotations
```xml
<pathwhiz:species pathwhiz:species_type="compound" pathwhiz:species_id="1420" />
```
- `pathwhiz:species_type`: `"compound"` or `"protein"`
- `pathwhiz:species_id`: PathWhiz internal integer ID from MySQL `compounds.id` or `proteins.id`

If MySQL is unavailable or the ID is not found, `pathwhiz:species_id` is omitted. Check `pathwhiz_id_stats` in the SBML build report to diagnose.

### Cross-reference annotations
```xml
<bqbiol:is>
  <rdf:Bag>
    <rdf:li rdf:resource="urn:miriam:chebi:CHEBI:15422" />
    <rdf:li rdf:resource="urn:miriam:hmdb:HMDB0000538" />
  </rdf:Bag>
</bqbiol:is>
```

URN prefixes used:
- `urn:miriam:hmdb:`
- `urn:miriam:kegg.compound:`
- `urn:miriam:chebi:` (value already includes `CHEBI:` prefix, e.g. `CHEBI:15422`)
- `urn:miriam:pubchem.compound:`
- `urn:miriam:drugbank:`
- `urn:miriam:uniprot:`

### Reaction annotations
```xml
<pathwhiz:reaction pathwhiz:reaction_type="reaction" pathwhiz:reaction_id="47595" />
```

### Compartment annotations
```xml
<pathwhiz:compartment pathwhiz:compartment_type="biological_state" />
```

---

## Normalisation Pipeline (`process_normalizer.py`)

`normalize_process_payload(payload)` runs these steps in order:

1. `normalize_composites` — splits composite entities (e.g. `"A + B"`)
2. `rewrite_reactions_to_complex_states`
3. `cleanup_disallowed_complexes`
4. `ensure_autostates`
5. `attach_transporters_from_evidence`
6. **`promote_interaction_enzymes`** ← added to fix "degree 0" gate failures
7. `promote_catalysts`
8. `canonicalize_same_as_aliases`
9. `normalize_process_actor_schema`
10. `dedupe_processes`
11. `run_strict_post_normalization_gates` ← hard-gate; raises `GateValidationError` on failure

### Known hard-gate failures and fixes

#### 1. `"Composite entity 'NAD+' has no protein-like left component"`
- **Cause:** `_has_plus_token()` treated `NAD+` (ionic charge notation) as a composite entity separator.
- **Fix (applied):** Strip trailing charge notation (`+`, `2+`, etc.) before testing for composite separator.
- **Location:** `process_normalizer.py` → `_has_plus_token()`

#### 2. `"Protein has degree 0 after normalization"`
- **Cause:** LLM generates enzyme-reaction relationships as `processes/interactions` entries instead of `reactions[].enzymes`. The graph builder used `node("entity", name)` for interaction participants, so proteins in interactions had degree 0.
- **Fix (applied — two-part):**
  1. `promote_interaction_enzymes()` in `process_normalizer.py`: scans interactions with catalytic relationship keywords (`catalyz`, `enzyme`, `activat`, `promotes`, `facilitate`), identifies which entity is a known protein and which matches a reaction name, adds protein to that reaction's `enzymes` list, removes the interaction.
  2. `qa_graph.py` interaction builder: changed from `node("entity", name)` to `node(resolve_kind(name), name)` so proteins in interactions get `node("protein", ...)` edges and pass the degree check even when name matching fails.

---

## Common Data Structures

### `mapped_ids` dict (on each entity)
```json
{
  "hmdb": "HMDB0000538",
  "kegg": "C00002",
  "chebi": "CHEBI:15422",
  "pubchem": "5957",
  "drugbank": "DB00171",
  "uniprot": "P04406"
}
```
ChEBI values always include the `CHEBI:` prefix. HMDB values use 7-digit zero-padded format.

### Reaction JSON schema
```json
{
  "name": "glucose phosphorylation",
  "inputs": ["glucose", "ATP"],
  "outputs": ["glucose-6-phosphate", "ADP"],
  "enzymes": [{"protein": "hexokinase"}],
  "biological_state": "cytoplasm"
}
```

### Interaction JSON schema (LLM sometimes generates instead of putting in enzymes)
```json
{
  "entity_1": "hexokinase",
  "entity_2": "glucose phosphorylation",
  "relationship": "catalyzes"
}
```
`promote_interaction_enzymes` converts these into `reactions[].enzymes` entries.

---

## SBML Build Report

After `build_sbml()` runs, check the report JSON for:

```json
{
  "counts": {"compartments": 3, "species": 40, "reactions": 10},
  "pathwhiz_id_stats": {
    "mysql_connected": true,
    "compounds_matched": 28,
    "proteins_matched": 8,
    "species_no_id": 4
  },
  "hard_errors": [],
  "warnings": []
}
```

- `mysql_connected: false` → DB credentials not reaching SBML builder; check `db_config` is passed from `app.py`
- `compounds_matched: 0` with `mysql_connected: true` → ID format mismatch between `mapped_ids` and MySQL storage; check `chebi_id` column format in MySQL
- `species_no_id > 0` → some entities not in PathWhiz DB (e.g. non-human proteins or rare compounds); acceptable

---

## Pending Work / Known Issues

1. **`pathwhiz:species_id` may be missing** if MySQL is not connected or the queried organism's proteins are not in PathWhiz. Check `pathwhiz_id_stats` in the report after each run.

2. **`promote_interaction_enzymes` name matching** uses exact then fuzzy (substring) matching on reaction names. If the LLM uses very different names for interactions vs reactions, some enzymes may not be promoted. The `qa_graph.py` fix (using `resolve_kind`) provides a safety net so the gate still passes.

3. **PathWhiz import testing** — the generated SBML has not yet been confirmed to fully import into PathWhiz. The next step is to upload the SBML to PathWhiz and verify rendering.

4. **`pathwhiz:reaction_id`** — reaction IDs are looked up by matching sorted reactant/product PathWhiz compound IDs against `reaction_elements` in MySQL. This will only succeed if all reactants/products have PathWhiz IDs.

5. **PWML required-field gate: LLM over-generates reactions and hallucinates protein complexes from metabolite names containing colons** — Observed during jasmonate biosynthesis export (21 errors, 48 warnings). Two root causes:

   **5a. Reaction inflation + unresolvable participants:**
   - The LLM generates ~22 reactions for a pathway that has ~9 meaningful steps. The excess reactions often have participants that are either empty or can't be resolved to any known entity in the JSON.
   - In `ir.py` the required-field gate fires `reaction_missing_left_participants` for reactions where `inputs` is empty after merge. This also happens because the LLM splits compound names at colons (e.g. `OPC-8:0-CoA`) and assigns the fragments as protein complex components, leaving the reaction's reactant list hollow.
   - **Fixed (2026-07-22).** `filter_unresolvable_reactions` lives in `src/t2pw/pipeline/pipeline.py` and drops any reaction with an empty or unresolvable side. It is applied via `apply_post_merge_cleanup`, which every payload entering the post-pipeline path must pass through — including the RAG-synthesized payload, which previously replaced the hardened merge and bypassed it entirely. See the 2026-07-22 change-log entry.

   **5b. Metabolite names with colons misclassified as protein complexes:**
   - Compound names like `hexadecatrienoic acid (16:3)`, `OPC-8:0-CoA`, `OPC-6:0-CoA` contain colons. The LLM (or its parsing of the protein complex schema) splits these at the colon, creating fake protein complex entries like `{name: "hexadecatrienoic acid (16:3)", components: ["hexadecatrienoic acid (16", "3)"]}`.
   - The PWML `ir.py` gate then fires `component_protein_unresolved` for each such fragment.
   - **Fixed** as part of the filter above, which also discards protein complexes whose components look like metabolite fragments (`_looks_like_metabolite_fragment` in `pipeline.py`).

   **5c. RAG-specific: the whole hardening pass was being skipped.** When multi-paper RAG is on, synthesis *replaces* the merged payload rather than adding to it, so until 2026-07-22 no RAG-added reaction was ever actor-deduped or checked for empty sides. If reactions with missing participants or duplicated enzyme complexes reappear, check first that `apply_post_merge_cleanup` still runs on the adopted RAG payload in `streamlit_app.py` — a guard that never executes looks exactly like a guard that does not work.

6. **PWML required-field gate: UniProt IDs missing for plant-specific proteins** — Proteins like `COMATOSE (CTS)`, `AtACH1`, `AtACH2`, `OPCL1`, `KAT2`, `b-hydroxy-acyl-CoA dehydrogenase` are Arabidopsis thaliana enzymes. The current UniProt resolver in `src/t2pw/mapping/map_ids.py` tries exact name match → name variants → EuropePMC literature aliases → LLM synonym generation. When all strategies fail it returns `unmapped`, which causes the `protein_missing_external_identity` gate to fire.
   - **Root cause:** The resolver never falls back to returning the *best available* UniProt search hit; it only returns a result when it is highly confident. For plant proteins with abbreviated names or hyphenated gene symbols, the search can return good candidates that are being discarded.
   - **Fix planned:** Add a `_search_uniprot_best_effort(client, name, organism)` fallback that: (1) extracts gene symbol from parentheses (e.g. `CTS` from `COMATOSE (CTS)`), (2) tries `gene:{symbol} AND organism:{taxon}` scoped UniProt REST search, (3) returns the top hit tagged with `confidence: "low"` and `method: "fuzzy_search"` rather than returning nothing. See agent prompt in docs (fix #2).

---

## Running the Pipeline

```bash
streamlit run src/t2pw/app/streamlit_app.py
```

Fill in:
- MySQL DB credentials in the sidebar
- Pathway description text
- Click **Run Pipeline**

---

## File Locations

```
Project14-T2PW/
├── src/
│   └── t2pw/
│       ├── app/streamlit_app.py              # Streamlit UI/orchestrator
│       ├── pipeline/process_normalizer.py     # Stage 3 normalization/gates
│       ├── pipeline/stage_contracts.py        # Boundary contracts
│       ├── pipeline/payload_models.py          # Runtime shape reports
│       ├── mapping/map_ids.py                  # Stage 2B/6 ID mapping
│       ├── mapping/enrich_entities.py          # Stage 7 enrichment
│       ├── curation/audit_json_llm.py          # Stage 4 audit
│       ├── curation/gap_resolver.py            # Stage 4 gap filling
│       └── sbml/json_to_sbml.py                # Legacy SBML builder
├── AGENT_INSTRUCTIONS.md         # This file
└── tmp/                          # Intermediate pipeline outputs
```
