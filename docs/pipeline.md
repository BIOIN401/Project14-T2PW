# Pipeline

The app and CLI paths use the same package modules. All stage logic lives in
`src/t2pw/` — the Streamlit app in `t2pw.app.streamlit_app` is the orchestrator
only; it calls stage functions and wires results together but does not own logic.

---

## Change log requirement

Every code edit to this pipeline must have a corresponding entry in
[`docs/change_log.md`](change_log.md). The entry must answer three things:

1. **What error or bug is being fixed** — the symptom the user or test saw
2. **Why it appeared** — the root cause, usually a stage boundary violation,
   field name mismatch, or logic that lived in the wrong module
3. **How the fix is consistent with this pipeline design** — which stage owns
   the change, that it does not introduce cross-stage logic, and that no
   existing module grew scope it should not have

The purpose of this log is to prevent the codebase from drifting back toward
bloat. If a fix cannot be explained in terms of stage ownership and modularity,
that is a sign the fix itself needs to be reconsidered.

See [`docs/change_log.md`](change_log.md) for the template and all entries.

---

## Design principle

No single stage is expected to produce perfect output. Each stage produces its
best-effort result and passes it forward. The audit loop exists specifically to
repair what earlier stages got wrong. The consequence of this is:

- **Structural contracts** (data is parseable, has required keys) are enforced
  at every stage boundary. These are cheap checks that catch garbage input early.
- **Semantic contracts** (data is logically correct) are enforced only after the
  audit loop has run. Enforcing them before the audit would prevent the audit
  from ever fixing them.
- **The gate is not a blocker before audit.** It is the error report that tells
  the audit what to fix. A gate failure before audit means "send this to audit,"
  not "abort the pipeline."
- **Broken stages must not produce output.** If a stage function raises, the
  orchestrator must not write output files for that stage or proceed to the next
  stage. Partial or corrupted output from a failing stage must never reach the
  export step. Each stage returns either a complete result or raises — there is
  no in-between state that gets written to disk and silently forwarded.
- **Stages are independent.** Each stage function takes an input object and
  returns an output object. It does not read from or write to files belonging to
  another stage. It does not call another stage's functions. Logic that spans
  two stages belongs in the orchestrator, not inside either stage function.

---

## Stages

### Stage 1 — Extract

**Module:** `t2pw.pipeline.pipeline.run_extraction_pipeline`
**Output file:** `stage1_raw_extraction.json`

An LLM reads the source paper and produces the initial pathway payload: entities
(compounds, proteins, protein complexes, species, subcellular locations) and
processes (reactions, transports, interactions).

This output is intentionally incomplete. The LLM does not have database IDs,
canonical names, or full reaction stoichiometry. Expect missing links, ambiguous
actor names, and schema inconsistencies. That is normal — later stages resolve
them.

The extraction prompt (`pwml_system.txt`) enforces single-organism scoping: before extracting reactions, the LLM selects one primary organism/species/strain for the pathway and assigns it to all extracted proteins, enzymes, complexes, reactions, and biological states. Proteins from other organisms are excluded unless the user explicitly requested a comparative or cross-species pathway. This prevents mixed-species entity sets from reaching Stage 2 mapping and Stage 3 gate checks.

**Structural contract enforced here:**
- Valid JSON
- Top-level keys `entities` and `processes` both present
- Every entity has a non-empty `name` field
- Every process has at least one of `inputs`, `outputs`, or `cargo`

If this contract fails, abort — no downstream stage can operate on malformed
input.

---

### Stage 2 — Inference + Map

**Inference prompt:** `pwml_infer_system.txt` (LLM modifier repair and enrichment pass)
**Module:** `t2pw.mapping.map_ids.map_payload`
**Output file:** `stage1_mapped.json`

An LLM inference pass runs after Stage 1 extraction. It receives the Stage-1 JSON and proposes conservative additions: missing modifier links (mandatory repair pass over all Stage-1 proteins and protein complexes), missing biological states, compartment assignments, and synonym bridges. The modifier repair pass applies the species scoping rule from Stage 1 — it only adds modifiers for proteins belonging to the selected pathway organism and does not link proteins from other organisms mentioned in the source text.

Following inference, database ID lookup runs against PathBank and configured ID sources. Writes `mapped_ids` and `mapping_meta` onto each entity. Species and subcellular locations are resolved here.

This stage does not know whether an entity name is correct. It maps what it
receives. If the LLM invented a protein name, mapping returns no hit — that is
recorded in `mapping_meta` and left for the audit to fix, not rejected here.

#### PathWhiz protein and protein-complex identity contract

PathWhiz treats proteins and protein complexes as separate records with
different required fields:

- A **protein** record requires a name, species, and either a UniProt ID or a
  DrugBank ID. Other fields such as gene name, EC number, sequence, and
  description are useful metadata, but they do not replace the UniProt/DrugBank
  requirement.
- A **protein complex** record requires a name, species, and at least one member
  protein with stoichiometry. A PathWhiz/PathBank protein-complex ID is useful
  when the complex already exists in the database, but it is not required when
  the pipeline can construct a valid novel complex from valid member proteins.

This means generated single-protein wrappers are valid only as
`entities.protein_complexes`, never as `entities.proteins`. For example, if a
reaction must reference a PathWhiz complex but the source evidence only names
`NdmA`, the valid payload shape is:

```json
{
  "entities": {
    "proteins": [
      {
        "name": "NdmA",
        "species": "Pseudomonas putida",
        "mapped_ids": {"uniprot": "H9N289"}
      }
    ],
    "protein_complexes": [
      {
        "name": "NdmA complex",
        "species": "Pseudomonas putida",
        "generated": true,
        "generation_reason": "single_protein_pathwhiz_wrapper",
        "components": [
          {
            "name": "NdmA",
            "stoichiometry": 1,
            "mapped_ids": {"uniprot": "H9N289"}
          }
        ]
      }
    ]
  }
}
```

The pipeline should not try to assign a UniProt ID to `NdmA complex` as though
it were a protein. The UniProt ID belongs to the member protein `NdmA`; the
generated complex is export structure around that protein. If the member
protein lacks species or a UniProt/DrugBank ID, the complex cannot be considered
exportable and the pipeline must stop before PWML generation.

**Structural contract enforced here:**
- Every entity that had a name now has a `mapping_meta` key (even if empty)
- Species list is non-empty (required for biological state resolution)

---

### Stage 3 — Normalize

**Module:** `t2pw.pipeline.process_normalizer.normalize_process_payload`
**No output file** (in-memory; Streamlit writes a probe file for debugging)

Deterministic cleanup. Runs ~15 steps in a fixed order on a deep copy of the
mapped payload and returns the cleaned payload plus a normalization report.

Steps in order:
1. `apply_biochemical_aliases` — resolve known synonym aliases
2. `normalize_composites` — collapse composite token shorthand
3. `rewrite_reactions_to_complex_states` — expand complex actors
4. `cleanup_disallowed_complexes` — remove forbidden complex forms
5. `ensure_autostates` — add missing biological state nodes
6. `backfill_reaction_compartments` — infer missing compartment from context
7. `attach_transporters_from_evidence` — wire transport actors from evidence text
8. `attach_enzymes_from_reaction_evidence` — wire enzyme actors from evidence text
9. `promote_interaction_enzymes` — lift interaction participants to reaction enzymes
10. `promote_catalysts` — lift catalyst modifiers to enzyme slot
11. `canonicalize_same_as_aliases` — deduplicate names via same-as links
12. `normalize_process_actor_schema` — enforce uniform actor dict shape
13. `drop_unresolved_complex_component_proteins` — drop component-only proteins with no external identity
14. `drop_process_orphan_proteins` — drop standalone proteins never referenced in any process and with no external identity
15. `prune_disconnected_proteins` — graph-based pass: drop degree-0 proteins with no external identity
16. `dedupe_processes` — collapse duplicate reaction/transport entries
17. `run_strict_post_normalization_gates` — generate gate report

Steps 13–15 run in sequence so each pass can only drop what the previous pass did not catch. A protein survives all three passes if it has any of: complex-component membership with external identity, a process reference, a non-zero graph degree, or an external database ID. Only after all three passes fail to retain it does the gate see it as an orphan.

Step 17 runs inside `normalize_process_payload` and collects all hard-gate failures into a structured report. **It does not raise an exception that aborts the pipeline.** The gate failures are returned as part of the normalization report and passed to the audit loop.

An `on_checkpoint` callback can be passed by the orchestrator to write probe
files at named checkpoints without splitting the normalization into two pipelines.

---

### Stage 4 — Audit

**Module:** `t2pw.curation.audit_json_llm.run_audit` + `t2pw.curation.apply_audit_patch`
**Output files:** `audit_report.json`, `audit_patch.json`, updated payload JSON

The primary semantic repair stage. It receives the normalized payload and the
gate report. If the gate passed with no failures, the loop is skipped. If gate
failures exist, the loop runs:

```
normalize → gate → if gate failures exist:
    audit LLM receives: current payload + gate failure list
    audit LLM proposes: a structured patch (add / replace / remove operations)
    patch is applied to the payload
    re-normalize (full 17-step pass)
    re-gate
    repeat up to max_iterations
```

Each round runs multiple candidates at different LLM temperatures; the best
candidate (fewest remaining errors, most accepted patches) is selected before
moving to the next round. The loop exits when the gate passes, no patch is
accepted, the payload repeats, or the iteration/timeout limit is reached.

The audit loop corrects:
- Protein names the LLM invented that do not map to any database entry
- Actor references in reactions that do not resolve to declared entities
- Missing enzyme assignments
- Incorrect biological state assignments
- Scaffolding proteins that leaked into modifiers

#### Stage 4a — Gap Resolve (per-round sub-step, optional)

**Module:** `t2pw.curation.gap_resolver.run_gap_resolution`

Runs inside the audit loop, once per round, after the best candidate for that
round is selected. Targets entities that still have no mapped IDs after Stage 2
and attempts targeted DB/API lookups. When `use_llm_gap_resolver` is enabled,
an LLM pass resolves ambiguous names that DB lookup alone cannot disambiguate.
Gap resolution output is folded into the round's settled payload before the next
audit round begins.

Gap Resolve is optional (controlled by `use_gap_resolver`). It does not replace
the audit — it supplements it by filling ID gaps the audit LLM is not designed
to fix. It owns ID lookup for unmapped entities only; it does not rewrite
reaction structure, actor roles, or biological states.

---

### Stage 5 — Curate

**Module:** `t2pw.curation.pathway_curator.run_pathway_curator`
**Output files:** `curator_report.json`, `final.curated.json`

A one-shot LLM curation step that runs **after the audit loop exits** and
**before the second mapping pass**. It receives the post-audit payload and a
fresh reaction summary generated from the settled audit output.

The curator addresses a different class of issue than the audit. The audit fixes
gate failures; the curator fixes presentation and completeness issues the gate
does not check:

1. **Name mismatches** — entity names that appear in `entities` under one form
   but in reaction inputs/outputs under a slightly different form (e.g. `"NAD"`
   vs `"NAD+"`). These do not fail the gate but break ID mapping and export.
2. **Missing compartments** — entities whose biological state is unknown or
   empty. The curator proposes compartment assignments from context.
3. **Missing transporters** — transport reactions whose `transporters` list is
   empty. The curator proposes candidate transporter proteins from evidence.
4. **Reaction order** — adds a `reaction_order` list if one does not exist.

All changes are emitted as JSON-Pointer patches using the same format as
`apply_audit_patch`, and the same patch policy (lock-awareness, confidence
threshold) applies. If no patches are accepted, the payload passes to Stage 6
unchanged.

**Structural contract enforced here:** After Curate, the payload must still
satisfy the post-extraction structural contract (valid JSON, required top-level
keys, non-empty entity names). If the curator patch corrupts the payload, the
orchestrator passes the pre-curate payload to Stage 6 instead.

---

### Stage 6 — Remap

**Module:** `t2pw.mapping.map_ids.map_payload`

Re-runs ID mapping on the curated payload. Entity names changed by the audit or
curator need fresh database lookups. The Streamlit orchestrator bypasses the
mapping cache for this pass (`use_cache=False`) so stale IDs from the pre-audit
mapping are not carried forward. Batch callers can pass `invalidate_cache_keys`
to the same function to achieve the same effect selectively.

If Stage 6 creates PathWhiz-required single-protein complex wrappers, it must
create them from already-mapped proteins. A generated complex is allowed to lack
a complex-level PathBank ID only when all of the following are true:

- The generated row is stored under `entities.protein_complexes`, not
  `entities.proteins`.
- The generated row has species/organism context.
- The generated row has at least one component.
- Every component resolves to a declared protein row with species and a
  UniProt/DrugBank identifier.

When those conditions are not met, Stage 6 should leave a mapping/gate issue for
audit or export blocking instead of creating an apparently usable complex.

---

### Stage 7 — Enrich (optional)

**Module:** `t2pw.mapping.enrich_entities.run_enrichment`

Fetches additional metadata (synonyms, cross-references, properties) and writes
`entity["enrichment"]`. Currently this data is not read by any downstream stage.
Decision pending: either wire it into the PWML IR builder or remove this stage.

---

### Stage 8 — Export

**Modules:** `t2pw.pwml.writer` (primary), `t2pw.sbml` (legacy)

Converts the final payload to PWML XML (primary) or SBML (legacy).

**Semantic contract enforced here (hard abort):**
- `validate_required_pwml_contract` from `t2pw.pwml.ir` — called before IR
  construction
- Checks: all process actors resolve to declared entities, all required DB IDs
  present (if strict mode), no scaffold modifiers, no unresolved composites
- Generated protein complexes are valid without a complex-level PathBank ID only
  if their member protein rows satisfy the protein contract above.

If this contract fails, the pipeline did not converge and the pathway is not
exportable. The error is surfaced to the user with the specific failing checks.
This is the only hard abort in the semantic sense.

---

## Stage contract summary

| Boundary | Type | Effect on failure |
|---|---|---|
| Post-extraction | Structural | Abort — data is unreadable |
| Post-mapping | Structural | Abort — mapping cannot proceed downstream |
| Post-normalization (gate) | Semantic | Feed to audit loop — this is expected |
| Post-audit | Structural | Abort — audit produced garbage |
| Post-curate | Structural | Fall back to pre-curate payload — do not abort |
| Pre-export | Semantic (full) | Abort — pipeline did not converge |

---

## Data flow

```
Paper text
    │
    ▼
[1: Extract]  ──→  stage1_raw_extraction.json
    │                        │
    │               structural contract
    │
    ▼
[2: Map]      ──→  stage1_mapped.json
    │                        │
    │               structural contract
    │
    ▼
[3: Normalize]               ← single canonical function, 17 steps
    │
    ├──→ gate passes ─────────────────────────────────────────────┐
    │                                                              │
    └──→ gate fails → [4: Audit] ──→ re-normalize ──→ gate again  │
              ↑           │   └── [4a: Gap Resolve] (per round)   │
              └───────────┘    (repeats up to max_iter)           │
                                                                   │
                                                                   ▼
                                                          [5: Curate]
                                                   (name fixes, compartments,
                                                    transporters, order)
                                                                   │
                                                      structural contract
                                                                   │
                                                                   ▼
                                                    [6: Remap]  (cache bypassed)
                                                                   │
                                                                   ▼
                                                       [7: Enrich] (optional)
                                                                   │
                                                                   ▼
                                                   pre-export semantic contract
                                                                   │
                                                                   ▼
                                                          [8: Export]
                                                  PWML XML  │  SBML (legacy)
```

---

## File ownership by area

| Area | File | Owns |
|---|---|---|
| Schema | `t2pw/schema.py` | TypedDicts for all payload shapes |
| Extraction prompt | `t2pw/llm/prompts/pwml_system.txt` | Stage 1 extraction rules: entity schema, reaction schema, single-pathway and single-organism scoping |
| Inference prompt | `t2pw/llm/prompts/pwml_infer_system.txt` | Stage 2 modifier repair and enrichment rules; species constraint cross-reference |
| Extraction | `t2pw/pipeline/pipeline.py` | LLM call, chunking, merge |
| Normalization | `t2pw/pipeline/process_normalizer.py` | All 17 steps, gate, actor helpers |
| Graph QA | `t2pw/pipeline/qa_graph.py` | Graph construction, degree checks |
| Stage contracts | `t2pw/pipeline/stage_contracts.py` | One function per boundary |
| Mapping | `t2pw/mapping/map_ids.py` | DB lookup, cache, ID assignment (Stages 2 and 6) |
| Enrichment | `t2pw/mapping/enrich_entities.py` | External API metadata (Stage 7) |
| Audit | `t2pw/curation/audit_json_llm.py` | LLM repair loop (Stage 4) |
| Gap resolve | `t2pw/curation/gap_resolver.py` | Per-round ID gap filling (Stage 4a) |
| Curate | `t2pw/curation/pathway_curator.py` | Post-audit name/compartment/transporter fixes (Stage 5) |
| Patch apply | `t2pw/curation/apply_audit_patch.py` | JSON patch operations (Stages 4 and 5) |
| PWML IR | `t2pw/pwml/ir.py` | IR construction, pre-export contract |
| PWML write | `t2pw/pwml/writer.py` | XML serialization, QA |
| Orchestrator | `t2pw/app/streamlit_app.py` | Wiring only — no logic |

The orchestrator calls stage functions and validators. It does not contain
normalization steps, actor resolution, or field manipulation.

---

## Actor field schema

Every actor in `reactions[].enzymes`, `reactions[].modifiers`,
`transports[].transporters`, and `interactions[].participants` must be a dict:

```json
{
  "entity": "<name>",
  "entity_type": "protein | protein_complex | compound",
  "role": "catalyst | inhibitor | activator | transporter | participant",
  "evidence": "<text snippet or empty string>",
  "confidence": 0.0–1.0,
  "provenance": "extracted | inferred | curated"
}
```

`entity` is the canonical name field. Any code that looks up an actor name must
check `entity` first, then fall back to `protein`, `protein_complex`, `name` for
backwards compatibility. The single helper `_actor_name_from_row` in
`process_normalizer.py` is the only place this lookup should be implemented.

---

## Running the pipeline

Full Streamlit UI:
```powershell
streamlit run src/t2pw/app/streamlit_app.py
```

PWML export from a mapped JSON file:
```powershell
python scripts/run_pwml.py --in final.mapped.json --out-dir outputs --non-strict-db
```

Legacy SBML export:
```powershell
python scripts/run.py --in final.json --out-dir outputs --no-llm-audit --no-sbml-overwatch
```

Verification after any change:
```powershell
pytest -q
ruff check src tests scripts
python -m py_compile src/t2pw/pipeline/pipeline.py src/t2pw/pwml/writer.py scripts/run.py scripts/run_pwml.py
```
