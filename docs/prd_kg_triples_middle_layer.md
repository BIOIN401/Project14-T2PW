# PRD: KG Triples Middle Layer for T2PW

Status: Draft for senior review

Author perspective: junior developer implementation plan

Target outcome: retrieve curated Reactome reaction triples as a first-class middle layer, use them to reduce LLM token burden and stabilize extraction, then preserve current PWML export behavior.

## 1. Summary

T2PW currently extracts pathway text into pathway JSON, performs graph QA and normalization, maps entities to PathBank/external IDs, and exports PWML. The pipeline is already graph-aware, but it does not yet retrieve curated pathway triples from an external pathway knowledgebase before asking the LLM to generate or fill pathway facts.

This PRD proposes adding a Reactome-backed KG triples middle layer:

```text
source text
  -> lightweight pathway/entity query extraction
  -> Reactome retrieval
  -> curated typed triples with literature references
  -> LLM gap-fill/reconciliation against source text
  -> pathway payload
  -> triple QA and reconciliation
  -> mapped pathway JSON
  -> PWML IR
  -> PWML XML
```

The goal is not to replace PWML generation. The goal is to avoid making the LLM recreate known pathway facts from scratch. Reactome already stores curated reactions with inputs, outputs, compartments, species, catalysts, cross-references, and literature references. We should retrieve those curated facts, convert them into our own normalized triples, and ask the LLM only to select relevant retrieved facts, resolve source-text scope, and fill gaps not covered by Reactome.

Example triple:

```json
{
  "id": "triple_0001",
  "subject": {"name": "hexokinase", "entity_type": "protein"},
  "predicate": "catalyzes",
  "object": {"name": "glucose phosphorylation", "entity_type": "reaction"},
  "evidence": {
    "quote": "hexokinase catalyzes the phosphorylation of glucose",
    "source_ref": "input_text",
    "pmid": ""
  },
  "confidence": 1.0,
  "provenance": "extracted"
}
```

## 2. Current System Anchors

These are the current files this PRD depends on.

- `src/t2pw/pipeline/pipeline.py:203`: `run_extraction_pipeline`, Stage 1 text -> strict JSON.
- `src/t2pw/pipeline/pipeline.py:281`: `run_inference_pipeline`, Stage 2 additions/enrichment.
- `src/t2pw/pipeline/pipeline.py:501`: `build_qa_feedback`, deterministic graph QA feedback for later rounds.
- `src/t2pw/pipeline/draft_graph.py:42`: `DraftEdge`, current role-based graph edge model.
- `src/t2pw/pipeline/draft_graph.py:174`: `build_draft_graph`, current payload -> graph conversion.
- `src/t2pw/pipeline/process_normalizer.py:3375`: `normalize_process_payload`, export-time normalization and hard gates.
- `src/t2pw/mapping/map_ids.py:3002`: `hydrate_species_references`, species grounding.
- `src/t2pw/mapping/map_ids.py:3979`: `route_entity_for_mapping`, entity type routing.
- `src/t2pw/mapping/map_ids.py:4604`: `run_mapping`, final entity ID mapping.
- `src/t2pw/pwml/ir.py:636`: `build_pwml_ir`, mapped payload -> PWML IR.
- `src/t2pw/pwml/ir.py:2299`: `validate_pwml_ir`, PWML IR validation.
- `src/t2pw/pwml/writer.py:4252`: `run_pwml_pipeline_export`, mapped JSON -> PWML artifacts.
- `src/t2pw/app/streamlit_app.py:88`: Streamlit app entry point.
- `src/t2pw/app/streamlit_app.py:142`: `render_attempts`, current LLM attempt display.
- `src/t2pw/app/streamlit_app.py:350`: PWML export setup from reviewed JSON.
- `src/t2pw/app/streamlit_app.py:715`: `render_json_artifact_compare`, current artifact viewer.
- `src/t2pw/app/streamlit_app.py:3384`: connectivity snapshot display.
- `src/t2pw/llm/prompts/pwml_system.txt:453`: current rule that every named enzyme/catalyst must appear in modifiers.

External source anchors:

- Reactome Content Service REST API: `https://reactome.org/ContentService/`
- Reactome Content Service docs: `https://reactome.org/dev/content-service`
- Reactome Graph Database docs: `https://reactome.org/dev/graph-database`
- Reactome download data: `https://reactome.org/download-data`

Important correction:

- PathBank/PathWhiz is still essential for PWML export identity, PathWhiz IDs, and compatibility.
- PathBank should not be treated as the primary curated triple source unless we later verify a table/model that exposes reaction triples with evidence.
- Reactome should be the primary source for citable curated pathway reaction triples.

## 3. Problem Statement

The current JSON schema asks the LLM to produce pathway relationships before we retrieve curated pathway facts. That is backwards for common canonical pathways. The LLM should not spend tokens reconstructing glycolysis, TCA, beta oxidation, Reactome-covered signaling steps, or known transport/reaction participants if a curated source already contains those relationships.

The current JSON schema also lets the LLM express pathway relationships in several places:

- `processes.reactions[].inputs`
- `processes.reactions[].outputs`
- `processes.reactions[].modifiers`
- `processes.transports[]`
- `processes.interactions[]`
- `element_locations[]`

This works, but it creates inconsistency:

- Catalysis can appear as a reaction modifier, an interaction, or a free-text relationship.
- Binding, activation, inhibition, transport, substrate/product roles, and compartment state are not enforced by one interaction ontology.
- Evidence exists, but it is not organized per relationship edge.
- The graph QA knows whether nodes connect, but it does not know whether an edge type is biologically valid for the subject/object types.
- PWML export receives a pathway-shaped payload, not a citable KG.

The result is variable outputs from different LLM runs. A Reactome-backed triple retrieval layer should reduce this variability by giving the model a compact candidate fact set instead of asking it to infer every fact from free text.

## 4. Goals

1. Retrieve curated pathway facts from Reactome before full extraction whenever a pathway or entities can be matched.
2. Convert Reactome reactions/events into our own typed triple representation.
3. Define a small interaction ontology for pathway extraction and Reactome conversion.
4. Use retrieved triples to reduce LLM prompt size by passing compact candidate facts instead of long unstructured context.
5. Generate deterministic triples from current pathway payload as a validation/backfill layer, not as the primary curated source.
6. Validate triples for predicate, domain, range, evidence, references, and grounding.
7. Use triples to repair or standardize the existing pathway JSON before mapping and PWML export.
8. Expose Reactome retrieval results, triples, triple QA, and payload-vs-triples diffs in Streamlit.
9. Preserve current PWML generation behavior and keep current CLI paths working.

## 5. Non-Goals

1. Do not replace PWML, PathWhiz, or PathBank mapping.
2. Do not require a graph database in the first implementation.
3. Do not require RDF/OWL serialization in the first implementation.
4. Do not remove current `processes.reactions`, `transports`, or `interactions` schema.
5. Do not make LLM triple generation mandatory for PWML export.
6. Do not add network-only dependencies for core tests.
7. Do not assume PathBank has curated triples unless confirmed by schema inspection.
8. Do not force Reactome-only output; Reactome will miss some organism-specific, plant, microbial, or paper-specific pathways.

## 6. Proposed Data Model

Add a new package:

```text
src/t2pw/kg/
  __init__.py
  ontology.py
  schema.py
  reactome_client.py
  reactome_to_triples.py
  retrieval.py
  cache.py
  from_payload.py
  qa.py
  reconcile.py
  export.py
```

### 6.0 Curated Retrieval Source: Reactome

Reactome should be the first curated KG source. It has a reaction-centered data model that maps well to pathway triples:

- reactions/events have inputs and outputs;
- catalysts/catalyst activities can be represented as `catalyzed_by`;
- compartments can be represented as `located_in` or reaction location context;
- physical entities have external references such as UniProt and ChEBI;
- events include literature references, often with PubMed IDs;
- the same data can be accessed by REST API or local Neo4j GraphDB.

Two access modes are required:

1. `reactome_mode="api"`:
   - Uses the Reactome Content Service over HTTPS.
   - Best for first implementation and quick demos.
   - Requires network access.

2. `reactome_mode="local_neo4j"`:
   - Uses a local Reactome Neo4j database.
   - Better for the lab goal of local LLMs and reduced external API dependence.
   - Can be added after the REST client works.

Reactome retrieval should be cached locally so repeated app runs do not repeatedly call the API.

### 6.1 Predicate Ontology

Start with a local controlled vocabulary in `src/t2pw/kg/ontology.py`.

Each predicate should define:

- `id`: stable local ID, for example `T2PW_REL:catalyzes`
- `label`: display name
- `aliases`: strings accepted from LLM or legacy relationship fields
- `domain`: allowed subject entity types
- `range`: allowed object entity types
- `inverse_of`: optional inverse predicate
- `pwml_mapping`: how this predicate maps back to current payload fields
- `external_mapping`: optional crosswalk to RO, GO, SBO, BioPAX, or BEL later

Initial predicate set:

| Predicate | Domain | Range | Current payload mapping |
| --- | --- | --- | --- |
| `has_reactant` | reaction | compound, nucleic_acid, element_collection | `reaction.inputs[]` |
| `has_product` | reaction | compound, nucleic_acid, element_collection | `reaction.outputs[]` |
| `catalyzed_by` | reaction | protein, protein_complex | `reaction.modifiers[].role == catalyst` |
| `catalyzes` | protein, protein_complex | reaction | inverse of `catalyzed_by` |
| `activated_by` | reaction, protein, protein_complex | proteiyn, protein_complex, compound | `interactions[].class == activation` |
| `inhibited_by` | reaction, protein, protein_complex | protein, protein_complex, compound | `interactions[].class == inhibition` |
| `binds` | compound, protein, protein_complex | compound, protein, protein_complex | `interactions[].class == binding` |
| `transports` | protein, protein_complex | compound, protein_complex, element_collection | `transports[].transporters[]` + `cargo` |
| `transported_from` | compound, protein_complex, element_collection | biological_state | `transports[].from_biological_state` |
| `transported_to` | compound, protein_complex, element_collection | biological_state | `transports[].to_biological_state` |
| `located_in` | biological entity | biological_state | `element_locations.*` |
| `part_of_complex` | protein | protein_complex | `protein_complexes[].components[]` |
| `has_component` | protein_complex | protein | inverse of `part_of_complex` |

Later external mappings can include:

- SBO for reaction participant roles.
- RO for regulates/positively regulates/negatively regulates/located in.
- GO cellular component IDs for locations.
- BioPAX interaction classes.
- BEL-style relations for abundance/activity relationships.

### 6.2 Triple Schema

Add `src/t2pw/kg/schema.py`.

Use plain dictionaries plus validation helpers at first to match existing code style. Avoid adding a new dependency unless senior dev approves.

Required triple fields:

```json
{
  "id": "triple_0001",
  "subject": {
    "name": "hexokinase",
    "entity_type": "protein",
    "mapped_ids": {}
  },
  "predicate": "catalyzes",
  "object": {
    "name": "glucose phosphorylation",
    "entity_type": "reaction",
    "mapped_ids": {}
  },
  "evidence": {
    "quote": "",
    "source_ref": "",
    "source_section": "",
    "pmid": "",
    "doi": "",
    "url": ""
  },
  "confidence": 1.0,
  "provenance": "extracted",
  "source_process": {
    "kind": "reaction",
    "name": "glucose phosphorylation",
    "pointer": "/processes/reactions/0"
  }
}
```

Required KG artifact:

```json
{
  "kg_version": "0.1",
  "ontology_version": "t2pw_rel_0.1",
  "pathway_context": {},
  "triples": [],
  "entities": {},
  "reports": {
    "qa": {},
    "reconciliation": {}
  }
}
```

## 7. Pipeline Design

### 7.1 Phase 1 Pipeline

Phase 1 should prove Reactome retrieval and triple conversion while keeping PWML behavior stable.

```text
source text
  -> lightweight query extraction
  -> Reactome Content Service search/query
  -> Reactome event records
  -> Reactome-to-T2PW triples
  -> triple QA
  -> Stage 1 extraction with compact retrieved-triples context
  -> Stage 2 gap fill only where needed
  -> mapping/enrichment
  -> PWML export
```

In this phase, Reactome triples are an artifact, prompt context, and QA tool. They should not mutate the payload yet except through explicit developer-approved reconciliation.

### 7.2 Phase 2 Pipeline

Phase 2 uses Reactome triples to improve output.

```text
source text
  -> Reactome triples
  -> Stage 1/2 payload
  -> deterministic triples from payload
  -> compare Reactome triples vs payload triples
  -> reconcile missing/contradictory relationships
  -> patched payload
  -> mapping/enrichment
  -> PWML export
```

### 7.3 Phase 3 Pipeline

Phase 3 makes retrieved and validated triples the preferred internal contract.

```text
source text
  -> Reactome/local KG retrieval
  -> optional LLM gap triples
  -> triple QA
  -> payload builder from triples
  -> PWML export
```

We should not jump to Phase 3 first because PWML export currently depends on the existing payload shape.

## 8. Required Implementation Work

### Workstream 0: Reactome Retrieval Layer

Must be done before Reactome-backed triples can be used. This is the highest-priority change to this PRD.

Files to add:

- `src/t2pw/kg/reactome_client.py`
- `src/t2pw/kg/retrieval.py`
- `src/t2pw/kg/cache.py`

Files to modify:

- `src/t2pw/pipeline/pipeline.py`
- `src/t2pw/app/streamlit_app.py`
- `docs/setup.md`
- `docs/pipeline.md`

Implementation details:

1. Add a small Reactome REST client using the existing project HTTP style.
2. Support basic calls:
   - query by stable ID when known, for example `R-HSA-*`;
   - search pathway/event names from extracted query terms;
   - fetch event details by stable ID;
   - fetch participating physical entities for an event;
   - fetch pathway-contained events where the API supports it.
3. Return raw Reactome records unchanged in a debug artifact first:

```text
reactome_retrieval_raw.json
reactome_retrieval_report.json
```

4. Add a local cache under `data/reactome_cache.json` or `tmp/reactome_cache.json`.
5. Add config:

```text
REACTOME_MODE=api
REACTOME_BASE_URL=https://reactome.org/ContentService
REACTOME_CACHE_PATH=data/reactome_cache.json
REACTOME_TIMEOUT_SECONDS=20
```

6. Fail soft:
   - if Reactome is unavailable, continue with current extraction pipeline;
   - emit a retrieval warning;
   - never block PWML export in non-strict mode.

Tests:

- `tests/test_reactome_client.py`
- Mock HTTP responses for event query.
- Mock pathway search response.
- Verify cache hit avoids HTTP call.
- Verify timeout/API error returns a structured warning, not a crash.

Manual Streamlit test:

- Add a pathway text field containing "glycolysis".
- Confirm Reactome retrieval panel shows candidate pathways/events.
- Confirm raw Reactome records are downloadable.

### Workstream 1: Reactome Records to T2PW Triples

Can start after Workstream 0 and Workstream A ontology basics.

Files to add:

- `src/t2pw/kg/reactome_to_triples.py`

Implementation details:

1. Convert Reactome event records to our triple schema.
2. Map Reactome fields:
   - `input[]` -> `has_reactant`
   - `output[]` -> `has_product`
   - `catalystActivity` / catalyst entity when available -> `catalyzed_by`
   - `compartment[]` -> `located_in` or reaction context
   - `literatureReference[]` -> evidence references with PubMed IDs
   - `stId` / `dbId` -> source IDs
   - `speciesName` / `species` -> species context
3. Preserve Reactome names and stable IDs.
4. Preserve external references such as UniProt, ChEBI, GO when present.
5. Mark provenance:

```json
{
  "provenance": "reactome",
  "source_database": "Reactome",
  "source_id": "R-HSA-141409"
}
```

6. Add `evidence.quote` from Reactome summation only when no direct source-text quote exists.
7. Put literature references into structured evidence:

```json
{
  "pmid": "11181178",
  "title": "...",
  "source_ref": "Reactome:R-HSA-141409"
}
```

Tests:

- `tests/test_reactome_to_triples.py`
- Reaction input/output maps to reactant/product triples.
- Literature reference maps to PMID evidence.
- GO compartment maps to location context.
- Unknown Reactome shape creates warnings, not exceptions.

### Workstream A: Ontology and Triple Schema

Can be done in parallel with the first Reactome client. Blocks Reactome-to-triples conversion and triple QA.

Files to add:

- `src/t2pw/kg/__init__.py`
- `src/t2pw/kg/ontology.py`
- `src/t2pw/kg/schema.py`

Implementation details:

1. Define `ALLOWED_PREDICATES`.
2. Define domain/range validation.
3. Define alias normalization, for example:
   - `catalyzes`, `catalyses`, `catalytic activity` -> `catalyzes`
   - `activates`, `upregulates`, `promotes` -> `activates`
   - `inhibits`, `suppresses`, `blocks` -> `inhibits`
4. Provide helper functions:
   - `normalize_predicate(value: str) -> str`
   - `get_predicate_spec(predicate: str) -> dict`
   - `is_allowed_domain_range(predicate, subject_type, object_type) -> bool`
   - `inverse_predicate(predicate: str) -> str`

Tests:

- `tests/test_kg_ontology.py`
- Validate aliases map to stable predicates.
- Validate invalid domain/range pairs fail.
- Validate inverse predicates are stable.

### Workstream B: Deterministic Payload to Triples

Can start after Workstream A. This is no longer the primary source of triples. It is a validation and backfill layer used to compare what our LLM-produced payload says against what Reactome retrieval supplied.

Files to add:

- `src/t2pw/kg/from_payload.py`

Files to reference:

- `src/t2pw/pipeline/draft_graph.py:174`
- `src/t2pw/pipeline/qa_graph.py`
- `src/t2pw/pipeline/process_normalizer.py:3375`

Implementation details:

1. Read current normalized payload.
2. Build triples from:
   - `processes.reactions[].inputs` -> `reaction has_reactant compound`
   - `processes.reactions[].outputs` -> `reaction has_product compound`
   - `processes.reactions[].modifiers` -> `reaction catalyzed_by protein/protein_complex`
   - `processes.transports[]` -> transport location and transporter triples
   - `processes.interactions[]` -> activation/inhibition/binding triples
   - `protein_complexes[].components[]` -> complex component triples
   - `element_locations.*` -> `located_in`
3. Preserve evidence from the source process where available.
4. Preserve `mapped_ids` if the payload has already been mapped.
5. Include JSON pointers in `source_process.pointer`.
6. Add comparison support so we can detect:
   - Reactome triple present but missing from payload;
   - payload triple not supported by Reactome;
   - same participants but different predicate;
   - same entity name but different grounding ID.

Tests:

- `tests/test_kg_from_payload.py`
- Input: one reaction with glucose, ATP, G6P, ADP, hexokinase.
- Assert triples include:
  - reaction `has_reactant` glucose
  - reaction `has_reactant` ATP
  - reaction `has_product` glucose-6-phosphate
  - reaction `catalyzed_by` hexokinase
- Input: transport with source/destination states.
- Assert `transported_from`, `transported_to`, and `transports` triples.
- Input: protein complex components.
- Assert `has_component` and `part_of_complex`.

### Workstream C: Triple QA

Can start after Workstream A. It can run in parallel with Workstream B once sample triples are hand-authored.

Files to add:

- `src/t2pw/kg/qa.py`

Implementation details:

1. Validate every triple has:
   - ID
   - subject name/type
   - predicate in ontology
   - object name/type
   - confidence between 0 and 1
   - provenance
2. Validate predicate domain/range.
3. Validate evidence:
   - quote exists for extracted triples
   - quote is <=25 words
   - quote is present in source text when source text is provided
4. Validate entity references:
   - subject/object exists in payload entity registry or process registry
5. Validate grounding:
   - after mapping, compounds should prefer HMDB/KEGG/ChEBI
   - proteins should prefer UniProt
   - locations should prefer GO when available
6. Produce report:

```json
{
  "ok": true,
  "errors": [],
  "warnings": [],
  "counts": {
    "triples": 12,
    "predicates": {"has_reactant": 4}
  }
}
```

Tests:

- `tests/test_kg_qa.py`
- Invalid predicate fails.
- Invalid domain/range fails.
- Missing evidence fails for `provenance == extracted`.
- Evidence over 25 words fails.
- Missing entity reference fails.
- Low confidence inferred triple warns but does not fail.

### Workstream D: Reconciliation Back to Pathway Payload

Can start after Reactome-to-triples, B, and C. This is the first part that can improve PWML outputs.

Files to add:

- `src/t2pw/kg/reconcile.py`

Files to modify:

- `src/t2pw/pipeline/pipeline.py`
- `src/t2pw/pipeline/process_normalizer.py`

Implementation details:

1. Implement a conservative `apply_triple_reconciliation(payload, kg_artifact)` function.
2. Only apply safe changes at first:
   - Add missing reaction inputs/outputs only when the Reactome event is confidently matched to an existing reaction and all entities already exist or map cleanly.
   - Move catalyst relationships from `processes.interactions[]` into `reactions[].modifiers`.
   - Add missing reaction modifiers when a `catalyzed_by` triple references an existing reaction and existing protein.
   - Add missing interaction class when predicate is activation/inhibition/binding.
   - Add source_refs/evidence to reactions when triple evidence is stronger.
3. Do not add new reactions in the first reconciliation version unless senior review approves a strict Reactome pathway import mode.
4. Do not delete reactions in the first reconciliation version.
5. Emit a reconciliation report with every proposed and applied change.

Tests:

- `tests/test_kg_reconcile.py`
- Catalysis interaction is promoted to reaction modifier.
- Existing modifier is not duplicated.
- Unknown protein does not get added silently.
- Unknown reaction does not get added silently.
- Reconciliation report lists skipped changes.

### Workstream E: Pipeline Integration

Can start after Workstream 0, Workstream 1, B, and C. Reconciliation integration waits for D.

Files to modify:

- `src/t2pw/pipeline/pipeline.py`
- `src/t2pw/pwml/writer.py`
- `scripts/run_pwml.py`
- `README.md`
- `docs/pipeline.md`

Implementation details:

1. Add a helper in `pipeline.py`, for example:

```python
def build_kg_artifact(
    payload: Dict[str, Any],
    *,
    source_text: str = "",
    validate: bool = True,
) -> Dict[str, Any]:
    ...
```

2. Write KG artifacts to output directories:
   - `reactome_retrieval_raw.json`
   - `reactome_retrieval_report.json`
   - `final.triples.json`
   - `kg_qa_report.json`
   - `kg_reconciliation_report.json`
3. In `run_pwml_pipeline_export` at `src/t2pw/pwml/writer.py:4252`, add optional flags:
   - `enable_kg_triples: bool = True`
   - `enable_kg_reconciliation: bool = False` initially
4. Keep default export behavior stable:
   - generating triples should not block PWML unless `strict_kg=True`
   - triple warnings should be visible but non-blocking by default
5. Add CLI flags:
   - `--reactome-mode api|local_neo4j|off`
   - `--no-kg-triples`
   - `--kg-reconcile`
   - `--strict-kg`

Tests:

- `tests/test_pwml_writer.py`
- Reactome unavailable does not block export in non-strict mode.
- Reactome raw retrieval artifacts are written when retrieval succeeds.
- Export still succeeds with `--no-kg-triples`.
- Export writes `final.triples.json` when enabled.
- KG QA errors do not block unless strict mode is enabled.
- Strict KG mode blocks on invalid triples.

### Workstream F: Optional LLM Triple Extraction

Can start after Reactome retrieval, A, B, and C. It should not block curated retrieval work.

Files to add:

- `src/t2pw/kg/llm_extract.py`
- `src/t2pw/llm/prompts/triples_system.txt`

Files to modify:

- `src/t2pw/pipeline/pipeline.py`
- `src/t2pw/llm/prompts/pwml_system.txt`
- `src/t2pw/llm/prompts/pwml_infer_system.txt`

Implementation details:

1. Add a small JSON-only prompt for gap triples.
2. Use controlled predicates only.
3. Give the model compact Reactome triples first, not full raw Reactome JSON.
4. Ask the model to output only:
   - which Reactome triples are relevant to the user/source text;
   - which source-text facts are missing from Reactome;
   - which retrieved triples conflict with the source scope.
5. Require evidence quote for every source-text-only triple.
6. Compare LLM triples to Reactome and deterministic payload triples:
   - missing in payload
   - missing in Reactome
   - predicate conflict
   - entity type conflict
7. Do not let LLM triple extraction mutate payload directly.
8. Feed comparison results into reconciliation or manual review.

Tests:

- Mock `chat` output.
- Valid LLM triple JSON parses.
- Unknown predicate is rejected or normalized by alias map.
- Triple diff reports missing Reactome catalysis and source-text-only catalysis.

### Workstream G: Streamlit Debug and Review UI

Can start after Workstream 0, Workstream 1, and C. Reconciliation controls wait for D.

Files to modify:

- `src/t2pw/app/streamlit_app.py`

Current UI anchors:

- `src/t2pw/app/streamlit_app.py:142`: existing attempt renderer.
- `src/t2pw/app/streamlit_app.py:715`: artifact compare viewer.
- `src/t2pw/app/streamlit_app.py:3384`: connectivity snapshot.
- `src/t2pw/app/streamlit_app.py:3390`: JSON artifact viewer.

UI additions:

1. Add a sidebar/form checkbox:
   - `Retrieve Reactome triples`
   - default on
2. Add another checkbox:
   - `Apply KG reconciliation`
   - default off until stable
3. Add a Reactome retrieval expander:
   - selected query terms
   - candidate pathways/events
   - source stable IDs
   - retrieval warnings
   - raw JSON download
4. Add an expander after mapped JSON and before PWML export:
   - `KG Triples`
5. In the expander show:
   - triple count
   - predicate counts
   - error/warning counts
   - table view: subject, predicate, object, confidence, evidence
   - JSON download
6. Add a `Triple QA` expander:
   - domain/range errors
   - missing evidence
   - unresolved entity refs
   - grounding misses
7. Add a `Payload vs Reactome KG Diff` expander:
   - reactions with no `catalyzed_by` triple
   - catalysts present in triples but missing from modifiers
   - interactions that look like catalysis and should be modifiers
   - triples that cannot map to PWML
8. Add download buttons:
   - `reactome_retrieval_raw.json`
   - `reactome_retrieval_report.json`
   - `final.triples.json`
   - `kg_qa_report.json`
   - `kg_reconciliation_report.json`

Manual Streamlit test cases:

1. Glycolysis mini text:
   - Input: "Hexokinase catalyzes glucose and ATP to glucose-6-phosphate and ADP in the cytosol."
   - Expected Streamlit output:
     - Reactome retrieval finds glycolysis-related candidate events or gives clear no-match warning
     - 4 reaction participant triples
     - 1 catalyst triple
     - no triple QA errors
     - PWML export still succeeds
2. Transport mini text:
   - Input: "GLUT4 transports glucose from extracellular space into cytosol."
   - Expected:
     - `transports`
     - `transported_from`
     - `transported_to`
     - PWML transport process or warning if PWML transport is incomplete
3. Regulation mini text:
   - Input: "ATP inhibits phosphofructokinase."
   - Expected:
     - `inhibits` or `inhibited_by` triple
     - no reaction created unless conversion is stated
     - warning that this is an interaction-only pathway if no reactions
4. Complex mini text:
   - Input: "The pyruvate dehydrogenase complex, composed of E1, E2, and E3, catalyzes pyruvate conversion to acetyl-CoA."
   - Expected:
     - `has_component` triples
     - `catalyzed_by` triple using protein_complex
     - no compound wrongly wrapped as protein_complex
5. Bad evidence test:
   - Manually edit a triple in the JSON viewer or use a test fixture with missing evidence.
   - Expected:
     - Triple QA shows missing evidence
     - strict mode blocks export
     - non-strict mode allows PWML with warning

## 9. Ordering and Dependencies

### Must Be Done First

1. Workstream 0: Reactome retrieval layer.
2. Workstream A: ontology and schema.
3. Workstream 1: Reactome records to T2PW triples.
4. Workstream C: triple QA.

These are foundational. Without them, UI and reconciliation will not have curated triples to show.

### Can Be Done in Parallel

Immediately:

- Workstream 0 Reactome REST client can start.
- Workstream A ontology/schema can start.
- Documentation updates can start.
- Streamlit mock UI design can start using fixture Reactome JSON.

After Workstream 0 and A:

- Workstream 1 Reactome-to-triples can start.
- Workstream C QA can start using hand-authored Reactome-like fixtures.

After Workstreams 1 and C:

- B deterministic payload-to-triples can start.
- D reconciliation can start.
- E pipeline artifact writing can start.
- G Streamlit display can start.

After D:

- Enable reconciliation in pipeline behind an off-by-default flag.
- Add Streamlit reconciliation controls.

After Reactome retrieval, A, B, and C:

- F optional LLM gap triple extraction can start independently.

### Should Be Deferred

1. RDF/OWL export.
2. Graph database persistence.
3. PubMed/EuropePMC citation retrieval for missing PMIDs.
4. Fully replacing current pathway JSON with triples.
5. Treating PathBank as a triple source unless a real triple-like schema is identified.

## 10. Test Plan

### Unit Tests

Add:

- `tests/test_reactome_client.py`
- `tests/test_reactome_to_triples.py`
- `tests/test_kg_ontology.py`
- `tests/test_kg_from_payload.py`
- `tests/test_kg_qa.py`
- `tests/test_kg_reconcile.py`

Update:

- `tests/test_process_normalizer.py`
- `tests/test_pipeline_cleanup.py`
- `tests/test_pwml_writer.py`
- `tests/test_pwml_ir.py`
- `tests/test_interactive_curator.py` if reviewed JSON state includes triples later.

### Integration Tests

Add fixtures:

```text
tests/fixtures/kg/
  reactome_event_minimal.json
  reactome_pathway_minimal.json
  glycolysis_minimal.json
  transport_minimal.json
  regulation_minimal.json
  complex_minimal.json
  invalid_triples_missing_evidence.json
```

Integration expectations:

1. Reactome event fixture -> triples -> QA passes.
2. Normal payload -> triples -> QA passes.
3. Reactome triples + normal payload -> diff report is generated.
4. Normal payload -> triples -> PWML export unchanged when reconciliation is off.
5. Reconciled payload -> PWML export succeeds.
6. Invalid triples -> strict KG fails.
7. Invalid triples -> non-strict KG warns and exports.

### Regression Tests

Existing PWML tests must continue to pass:

- `tests/test_pwml_ir.py`
- `tests/test_pwml_writer.py`
- `tests/test_pwml_db_resolver.py`
- `tests/test_process_normalizer.py`

Important regression assertion:

For an existing known-good mapped payload, enabling KG triples with reconciliation off should not change the final PWML IR except for added artifact files/reports.

### Manual Streamlit Tests

Run:

```powershell
streamlit run src/t2pw/app/streamlit_app.py
```

Manual checklist:

- The Reactome retrieval checkbox appears.
- Running a mini pathway shows retrieval candidates and triple counts.
- Triple table shows subject/predicate/object/evidence.
- Triple QA report is downloadable.
- PWML export still works.
- JSON Artifact Viewer includes Reactome raw retrieval, triples, and KG reports.
- Turning off Reactome retrieval/KG triples returns current behavior.
- Turning on reconciliation shows a reconciliation report.

## 11. Acceptance Criteria

MVP acceptance:

1. Reactome retrieval can fetch pathway/event records through the Content Service when network access is available.
2. Reactome retrieval fails soft and does not block PWML export in non-strict mode.
3. Reactome event records convert into T2PW triples for inputs, outputs, catalysts where available, compartments, and literature references.
4. `payload_to_triples` produces deterministic payload-derived triples for comparison/backfill.
5. Triple QA catches invalid predicates, invalid domain/range, missing evidence, and unresolved references.
6. Streamlit shows and downloads Reactome retrieval results, triples, and KG QA.
7. PWML export remains functional.
8. KG triples are non-blocking by default.
9. Tests cover Reactome client behavior, Reactome conversion, ontology, triple generation, QA, and PWML non-regression.

Phase 2 acceptance:

1. Reconciliation can use Reactome triples to add missing participants/modifiers to existing matched reactions.
2. Reconciliation can promote catalysis-like interactions into reaction modifiers.
3. Reconciliation does not invent new reactions unless strict Reactome import mode is explicitly added later.
4. Reconciliation reports every applied and skipped change.
5. Streamlit shows payload-vs-Reactome-KG differences.
6. PWML outputs improve for cases where catalysts or participants were previously missing.

Phase 3 acceptance:

1. Optional LLM gap-triple extraction works with local LLM provider.
2. LLM output uses only ontology predicates.
3. LLM gap triples can be compared against Reactome and payload-derived triples.
4. Reactome retrieval plus compact triple context reduces prompt/output variability on repeated runs.

## 12. Risks and Mitigations

Risk: triples become a second source of truth.

Mitigation: in Phase 1 Reactome triples are retrieval artifacts and prompt context only. In Phase 2 reconciliation writes an explicit report and is off by default.

Risk: Reactome coverage is incomplete for plant, microbial, or specialized pathways.

Mitigation: Reactome retrieval is a prior, not a hard requirement. The existing LLM extraction path remains active for source-text-only facts.

Risk: Reactome human pathways are incorrectly applied to another organism.

Mitigation: include species in retrieval ranking and triple QA. Flag inferred orthology and species mismatch distinctly.

Risk: network/API availability blocks development.

Mitigation: use cached fixtures and fail-soft API behavior. Add local Neo4j mode after REST mode.

Risk: ontology is too small.

Mitigation: start small and local. Add predicates only when they map clearly to payload/PWML behavior.

Risk: ontology is too broad.

Mitigation: every predicate must have domain/range rules and a PWML mapping or documented reason for being review-only.

Risk: KG QA blocks valid partial pathways.

Mitigation: non-strict KG mode by default. Strict mode is opt-in.

Risk: LLM triple generation adds cost and instability.

Mitigation: Reactome retrieval and deterministic conversion ship first. LLM gap triples are optional and local-provider-compatible.

Risk: evidence quotes are missing from older payloads.

Mitigation: warn in non-strict mode. Only block extracted triples in strict mode.

## 13. Open Questions for Senior Review

1. Should we use a local `T2PW_REL` ontology first, or immediately cross-map to SBO/RO/BioPAX?
2. Should triples use reaction nodes as subjects for participant roles, or entity nodes as subjects with inverse predicates?
3. Should `source_refs` remain a list of quote strings, or should we standardize it into structured evidence objects?
4. Should Reactome retrieval be limited to human pathways first, or should inferred orthologous events be used for supported non-human species?
5. Should local Reactome Neo4j be required for lab deployments, or should REST + cache be acceptable for the first release?
6. Should KG reconciliation run before or after `run_mapping`?
   - My recommendation: run before mapping for structural repair, then regenerate triples after mapping for grounded IDs.
7. Should strict KG mode become required before PWML export later?
   - My recommendation: no until we have enough fixtures from real pathways.
8. Should PathBank schema be inspected for any reaction participant tables useful for validation, while still not treating it as the primary triple source?

## 14. Recommended Implementation Sequence

1. Add `src/t2pw/kg/reactome_client.py` and `tests/test_reactome_client.py`.
2. Add `src/t2pw/kg/cache.py` for Reactome response caching.
3. Add `src/t2pw/kg/ontology.py` and `tests/test_kg_ontology.py`.
4. Add `src/t2pw/kg/schema.py` validation helpers.
5. Add `src/t2pw/kg/reactome_to_triples.py` and `tests/test_reactome_to_triples.py`.
6. Add `src/t2pw/kg/qa.py` and `tests/test_kg_qa.py`.
7. Add `src/t2pw/kg/from_payload.py` and `tests/test_kg_from_payload.py`.
8. Add Reactome-vs-payload triple diffing.
9. Add pipeline helper to write `reactome_retrieval_raw.json`, `final.triples.json`, and `kg_qa_report.json`.
10. Add Streamlit Reactome retrieval and triple display/downloads.
11. Add PWML regression test proving KG artifact generation does not change export.
12. Add `src/t2pw/kg/reconcile.py` behind a disabled-by-default flag.
13. Add reconciliation tests.
14. Add optional LLM gap-triple extraction and diffing.
15. Add docs updates and examples.

## 15. Definition of Done for First PR

First PR should be intentionally scoped:

- Add Reactome REST client, cache, ontology, Reactome-to-triples conversion, and QA.
- Add tests for those modules.
- Add output artifact generation only where low-risk.
- Do not enable reconciliation by default.
- Do not modify existing extraction prompt behavior yet.
- Prove PWML export is unchanged when KG reconciliation is off.

Suggested first PR title:

```text
Add Reactome-backed KG triple retrieval and QA layer
```

Suggested second PR title:

```text
Expose Reactome triples in Streamlit and PWML artifact viewer
```

Suggested third PR title:

```text
Add opt-in KG reconciliation for catalyst and interaction normalization
```
