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

**Structural contract enforced here:**
- Valid JSON
- Top-level keys `entities` and `processes` both present
- Every entity has a non-empty `name` field
- Every process has at least one of `inputs`, `outputs`, or `cargo`

If this contract fails, abort — no downstream stage can operate on malformed
input.

---

### Stage 2 — Map

**Module:** `t2pw.mapping.map_ids.map_payload`
**Output file:** `stage1_mapped.json`

Database ID lookup against PathBank and configured ID sources. Writes
`mapped_ids` and `mapping_meta` onto each entity. Species and subcellular
locations are resolved here.

This stage does not know whether an entity name is correct. It maps what it
receives. If the LLM invented a protein name, mapping returns no hit — that is
recorded in `mapping_meta` and left for the audit to fix, not rejected here.

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

Step 15 runs inside `normalize_process_payload` and collects all hard-gate
failures into a structured report. **It does not raise an exception that aborts
the pipeline.** The gate failures are returned as part of the normalization
report and passed to the audit loop.

An `on_checkpoint` callback can be passed by the orchestrator to write probe
files at named checkpoints without splitting the normalization into two pipelines.

---

### Stage 4 — Audit loop

**Module:** `t2pw.curation.audit_json_llm.run_audit` + `t2pw.curation.apply_audit_patch`
**Output files:** `audit_report.json`, `audit_patch.json`, updated mapped JSON

This is the repair stage. It receives the normalized payload and the gate report.
If the gate passed, the loop is skipped. If the gate failed, the loop runs:

```
normalize → gate → if gate failures exist:
    audit LLM receives: current payload + gate failure list
    audit LLM proposes: a structured patch (add / replace / remove operations)
    patch is applied to the payload
    re-normalize (full 15-step pass)
    re-gate
    repeat up to max_iterations
```

The audit LLM calls tools (LLM API) during this loop. This is expected. The loop
can and should make several round-trips. The audit loop is the intended mechanism
for correcting:
- Protein names the LLM invented that do not map to any database entry
- Actor references in reactions that do not resolve to declared entities
- Missing enzyme assignments
- Incorrect biological state assignments
- Scaffolding proteins that leaked into modifiers

After the audit loop exhausts (gate passes or max iterations reached), a second
mapping pass runs to re-resolve any entities whose names the audit changed.

**Important:** The audit loop is the last opportunity for LLM-driven correction.
After this point, the pipeline moves to deterministic steps only.

---

### Stage 5 — Second map (post-audit)

**Module:** `t2pw.mapping.map_ids.map_payload`

Re-runs ID mapping on the audited payload. Any entity names that the audit
changed need fresh database lookups. The Streamlit orchestrator bypasses the
mapping cache for this pass so stale IDs from the pre-audit mapping are not
carried forward. Batch callers can alternatively pass explicit cache keys to
invalidate.

---

### Stage 6 — Enrich (optional)

**Module:** `t2pw.mapping.enrich_entities.run_enrichment`

Fetches additional metadata (synonyms, cross-references, properties) and writes
`entity["enrichment"]`. Currently this data is not read by any downstream stage.
Decision pending: either wire it into the PWML IR builder or remove this stage.

---

### Stage 7 — Export

**Modules:** `t2pw.pwml.writer` (primary), `t2pw.sbml` (legacy)

Converts the final payload to PWML XML (primary) or SBML (legacy).

**Semantic contract enforced here (hard abort):**
- `validate_required_pwml_contract` from `t2pw.pwml.ir` — called before IR
  construction
- Checks: all process actors resolve to declared entities, all required DB IDs
  present (if strict mode), no scaffold modifiers, no unresolved composites

If this contract fails, the audit loop did not converge and the pathway is not
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
| Pre-export | Semantic (full) | Abort — audit did not converge |

---

## Data flow

```
Paper text
    │
    ▼
[Extract]  ──→  stage1_raw_extraction.json
    │                     │
    │            structural contract
    │
    ▼
[Map]      ──→  stage1_mapped.json
    │                     │
    │            structural contract
    │
    ▼
[Normalize]                          ← single canonical function, 15 steps
    │
    ├──→ gate passes ──────────────────────────────────────┐
    │                                                       │
    └──→ gate fails → [Audit loop] ──→ re-normalize ──→ gate again
              ↑______________|  (repeats up to max_iter)   │
                                                            │
                                                            ▼
                                              [Second Map]  (post-audit, cache-invalidated)
                                                            │
                                                            ▼
                                                      [Enrich] (optional)
                                                            │
                                                            ▼
                                              pre-export semantic contract
                                                            │
                                                            ▼
                                                       [Export]
                                               PWML XML  │  SBML (legacy)
```

---

## File ownership by area

| Area | File | Owns |
|---|---|---|
| Schema | `t2pw/schema.py` | TypedDicts for all payload shapes |
| Extraction | `t2pw/pipeline/pipeline.py` | LLM call, chunking, merge |
| Normalization | `t2pw/pipeline/process_normalizer.py` | All 15 steps, gate, actor helpers |
| Graph QA | `t2pw/pipeline/qa_graph.py` | Graph construction, degree checks |
| Stage contracts | `t2pw/pipeline/stage_contracts.py` | One function per boundary |
| Mapping | `t2pw/mapping/map_ids.py` | DB lookup, cache, ID assignment |
| Enrichment | `t2pw/mapping/enrich_entities.py` | External API metadata |
| Audit | `t2pw/curation/audit_json_llm.py` | LLM repair loop |
| Patch apply | `t2pw/curation/apply_audit_patch.py` | JSON patch operations |
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
