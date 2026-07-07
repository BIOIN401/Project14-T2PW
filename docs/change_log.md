# Change Log

Every entry answers: what was the error, why did it appear, and how does the
fix stay consistent with the intended pipeline design.

---

## Open Issues

Issues confirmed by running the pipeline. Ordered by pipeline stage. No code
changes yet — this section records the diagnosis and the planned fix.

---

### OPEN — Stage 3 (Normalize): `normalize_process_actor_schema` never writes `entity` key for enzyme actors

**Files to change:** `src/t2pw/pipeline/process_normalizer.py`

**Error / symptom:**
All enzyme actor dicts in `reactions[].enzymes` retain `protein_complex` (or
`protein`) as the name field after normalization completes. `e.get("entity")`
returns `""` for every enzyme. The JSON evidence is clear: after a full pipeline
run on the Arabidopsis jasmonic acid pathway all 30 enzyme actors use
`protein_complex`, while all 30 modifier actors correctly use `entity/entity_type`.

**Root cause:**
`normalize_process_actor_schema` has two passes.

Pass 1 (`_rewrite_actor_rows`) resolves each actor name against the protein and
complex registries. `_resolve_actor_name` returns a tuple `("protein_complex",
canonical_name)` or `("protein", canonical_name)`. The function then pops the
old fields and writes `updated[target_field] = canonical_name` — meaning it
writes back to `protein_complex` or `protein`, **not** to `entity`.

Pass 2 (the post-process block at lines 2287–2317) migrates `modifiers[]` from
old field names to `entity/entity_type`. This block runs **only on modifiers**;
there is no equivalent block for `enzymes[]`. So enzyme actors are left in the
legacy field format even after the schema-normalization step completes.

**Planned fix:**
After the existing modifier migration block (line ~2317), add an identical
migration loop for `enzymes[]` in every reaction. The logic is the same:
if `entity` is absent, copy the value from `protein_complex`, `protein`, or
`name` into `entity`, set `entity_type` accordingly, and pop the legacy keys.
Optionally, `_rewrite_actor_rows` itself could be updated so `target_field` is
always `"entity"` and `entity_type` is set separately — that eliminates the
need for a second-pass migration entirely.

**Pipeline consistency:**
Change lives entirely in `normalize_process_actor_schema` inside
`process_normalizer.py`, which owns actor schema enforcement. No other module
needs to change. After the fix, any code that checks `actor.get("entity")` will
work correctly for both enzyme and modifier actors.

---

### OPEN — Stage 4 (Audit): LLM connection failure prevents semantic repair

**Files to change:** Configuration / environment (not source code)

**Error / symptom:**
`curator_report.json` shows `"error": "chat_with_tools call failed after
retries. Last error: Connection error."` The round-1 audit report shows
`"enabled": false`. The 3 reactions with missing inputs (see next issue) survive
to the final payload unremediated because no LLM repair ran.

**Root cause:**
The OpenRouter API is not reachable from this environment. Either the API key
is missing/invalid, the endpoint is blocked, or there is a network configuration
issue. The audit stage correctly identifies there are errors to fix (3 reactions
with empty inputs) but cannot reach the LLM to generate repair patches.

**Planned fix:**
1. Verify `OPENROUTER_API_KEY` (or equivalent) is set in the environment.
2. Confirm the OpenRouter endpoint is reachable (`curl https://openrouter.ai/api/v1/models`).
3. If using a local model or alternative provider, update the provider config in
   `src/t2pw/curation/audit_json_llm.py`.
4. Add a clear error message in the Streamlit UI when LLM is disabled so the
   user knows semantic repair was skipped, not that the pipeline succeeded.

**Pipeline consistency:**
This is a configuration / infrastructure issue. No stage logic needs to change.
The pipeline correctly passes gate failures to audit; the repair just cannot run
without an LLM connection.

---

### OPEN — Stage 1 (Extract): Empty reaction inputs in beta-oxidation chain

**Files to change:** None yet — this is a Stage 4 (Audit) repair task once
LLM is connected (see connection issue above).

**Error / symptom:**
Reactions `beta_oxidation_OPC8`, `beta_oxidation_OPC6`, `beta_oxidation_OPC4`
(indices 10, 11, 13 in the final payload) each have `inputs: []` and
`outputs: ["jasmonic acid"]`. The audit deterministic check reports:
`"Reaction must include at least one input and one output."` for all three.

In the real pathway, each beta-oxidation cycle takes an OPC-CoA ester (OPC-8:0-CoA,
OPC-6:0-CoA, OPC-4:0-CoA) as input and produces the next shorter chain as
output. The LLM collapsed the entire beta-oxidation chain and wrote only the
final product (jasmonic acid) as each reaction's output.

**Root cause:**
Stage 1 (Extract) LLM did not capture the intermediate acyl-CoA compounds as
reaction-level inputs/outputs. Each reaction was extracted as "produces JA" with
the intermediate steps omitted. This is an expected limitation of the extraction
stage — the LLM summarised rather than enumerated each cycle.

**Planned fix:**
This is the correct input for Stage 4 (Audit). Once the LLM connection is
restored (see above), the audit should:
1. Receive the gate failure list which includes these 3 reactions.
2. Propose patches that add the correct OPC-CoA intermediate as `inputs` for
   each reaction and the shortened OPC-CoA as `outputs` (except the last cycle
   which produces jasmonic acid).
No normalization or schema change is required — this is a data completeness
issue that the audit loop is designed to repair.

**Pipeline consistency:**
The gate correctly fires for these reactions. The pipeline design says gate
failures feed audit, not abort. No stage logic change needed. This issue will
resolve once the LLM connection issue is fixed.

---

### OPEN — Stage 1 (Extract): OPC-8, OPC-6, OPC-4 misclassified as proteins

**Files to change:** None yet — audit repair task.

**Error / symptom:**
Entities `OPC-8`, `OPC-6`, and `OPC-4` appear in `entities.proteins`. The audit
warns: `"Protein has no location link; default compartment may be used."` for
all three. These are 3-oxo-2-(2'-pentenyl)-cyclopentane-acyl-CoA intermediates —
chemical compounds, not proteins.

**Root cause:**
Stage 1 (Extract) LLM placed these in `proteins` because they appear alongside
enzyme names in the paper text, and their abbreviated names (OPC-N) look like
protein/gene identifiers. The normalization stage does not reclassify entity
types.

**Planned fix:**
Once the LLM connection is restored, the audit should propose moving these
entities from `entities.proteins` to `entities.compounds`. An alternative
heuristic: Stage 3 normalization could flag entities whose names match known
patterns (CoA suffix, lipid chain nomenclature) as candidate compound
misclassifications and include them in the gate report for audit review. This
would not require a hard reclassification in normalization (which would be
cross-stage logic), just an additional audit hint.

**Pipeline consistency:**
Reclassification is a semantic operation and belongs in Stage 4 (Audit). If a
normalization hint is added, it goes into the gate report as an audit input —
not as a normalization mutation. This preserves the rule that Stage 3 does not
make semantic corrections.

---

### OPEN — Stage 2 (Map): DB unavailable — degraded mapping rates

**Files to change:** Configuration only.

**Error / symptom:**
`mapping_report.dbonly.json` shows `"db_available": false`. Protein mapping:
38.46% (5/13). Compound mapping: 9.52% (2/21). All 12 protein complexes skipped
(10/12 have gap issues — component proteins unmapped, so complex cannot map).

**Root cause:**
The PathBank database is not reachable from this environment
(`db_available: false`). Stage 2 falls back to API-only mapping, which has
lower coverage than the local DB. Compound mapping is especially degraded
because most common metabolites depend on the local compound DB.

**Planned fix:**
1. Configure the local PathBank DB (host, schema, credentials) in the Streamlit
   sidebar or environment variables.
2. If DB is intentionally absent, document that compound ID coverage will be
   low and PWML export will operate in non-strict mode.
3. The protein complex gap issues (10/12) are downstream of the protein mapping
   problem: once component proteins map, the complexes should map too. Fix the
   DB connection first and re-run before filing separate complex issues.

**Pipeline consistency:**
This is a configuration / infrastructure issue. Stage 2 mapping logic is correct.
No source changes needed. The pipeline.md note on Stage 2 already says mapping
returns no-hit when an entity is unmapped, which is recorded in `mapping_meta`.

---

### OPEN — Stage 6 (Enrich): Enrichment stage produces data no stage consumes

**Files to change:** Decision required before code change.

**Error / symptom:**
`run_enrichment` fetches synonyms, cross-references, and properties and writes
`entity["enrichment"]` onto each entity. No downstream stage (normalization,
audit, PWML IR, SBML) reads this field. The enrichment API call and cache write
happen on every pipeline run with no effect on output.

**Root cause:**
The enrichment stage was built but not wired into the PWML IR builder or any
other consumer. This was documented as a product decision pending in the
refactoring plan (Step 8).

**Planned fix — choose one:**
- **Option A (Use it):** Wire `entity["enrichment"]` into the PWML IR builder
  (`src/t2pw/pwml/ir.py`) so synonyms and cross-references appear in the
  exported pathway file. This adds value to the PWML output.
- **Option B (Remove it):** Delete `run_enrichment` and its call site in
  `streamlit_app.py`. Mapping already attaches database IDs; enrichment adds
  nothing until Option A is implemented. Removal simplifies the orchestrator and
  eliminates dead code.
Until the decision is made, the enrichment stage runs silently on every pipeline
execution consuming API quota and cache space.

**Pipeline consistency:**
Option A change lives in `t2pw/pwml/ir.py`. Option B removes
`t2pw/mapping/enrich_entities.py` call site from the orchestrator. Neither
option adds cross-stage logic.

---

## Template

```
### YYYY-MM-DD — <short description>

**Files changed:** `path/to/file.py` (lines X–Y)

**Error / symptom:**
What the user or test saw. Quote the error message if there is one.

**Root cause:**
Why the error appeared. Name the specific stage boundary violation, field
mismatch, or misplaced logic that caused it.

**Fix:**
What was changed and where.

**Pipeline consistency:**
Which stage owns this change. Confirm it does not add cross-stage logic and
does not expand any module's scope beyond its intended area (see File ownership
table in pipeline.md).
```

---

## Entries

### 2026-07-07 — Wire drop_process_orphan_proteins into normalize_process_payload

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (line ~3586), `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
`drop_process_orphan_proteins` was defined and documented but never called inside
`normalize_process_payload`. Standalone subunit proteins (e.g. NdmC, NdmD) that
appear only as `protein_complex.components` entries and are never referenced in any
reaction, transport, or interaction would pass through all normalization steps and
reach the gate as orphans, generating audit issues that should have been pre-empted
by pruning.

**Root cause:**
A prior implementation pass added the function to the module but omitted the call
site from the pipeline sequence in `normalize_process_payload`. The change log
stated it was wired in, but the code did not reflect this. The gap was discovered
by reviewing the actual step sequence (lines 3584–3588) against the documented
17-step list.

**Fix:**
Added the call `drop_process_orphan_proteins(data, report=report)` and its
corresponding `_checkpoint("drop_process_orphan_proteins")` between
`drop_unresolved_complex_component_proteins` and `prune_disconnected_proteins`.
Updated `docs/pipeline.md` to reflect the 17-step sequence and document why steps
13–15 run in sequence (each catches a different class of orphan; a protein must
fail all three to be treated as an orphan by the gate).

**Pipeline consistency:**
Change is entirely within `normalize_process_payload` in `process_normalizer.py`,
which owns all normalization steps. No orchestrator, mapping, audit, or export
logic was touched. The three pruning steps remain independent functions — each with
a single responsibility — rather than being merged into one function that would be
harder to reason about when a specific class of orphan slips through.

---

### 2026-07-07 - Deterministic PWML compound IDs with optional DB resolver

**Files changed:** `src/t2pw/pwml/ir.py`, `docs/change_log.md`

**Error / symptom:**
PWML IR tests could pass or fail depending on which modules pytest collected in
the same process. A payload with explicit `pathbank_compound_id` values produced
`compound_db_resolution_failed` errors when a DB resolver was importable.

**Root cause:**
`_resolve_compound_rows` only accepted direct PathWhiz compound IDs as a fallback
when no DB resolver was available. If resolver construction succeeded, the same
rows were sent through live DB matching and could fail despite already carrying
the required export ID.

**Fix:**
Accepted explicit `pathbank_compound_id` / `pw_compound_id` / `pathwhiz_id`
values before attempting resolver lookup, while still recording the
`legacy_id_unverified` DB-resolution status in the IR report.

**Pipeline consistency:**
This stays inside `t2pw.pwml.ir`, which owns pre-export IR construction. It does
not move mapping logic into export; it only makes already-mapped payload IDs
deterministic regardless of optional DB resolver availability.

---

### 2026-07-07 - Streamlit uses canonical normalization and post-audit cache bypass

**Files changed:** `src/t2pw/app/streamlit_app.py`, `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
The Streamlit post-pipeline path still owned a hand-built normalization sequence
and returned immediately on post-normalization gate failures, preventing the
audit loop from repairing the semantic issues documented by the gate.

**Root cause:**
Normalization logic lived partly in the orchestrator, including evidence-based
enzyme attachment and explicit gate handling. Post-audit mapping also reused the
normal mapping cache even though audit patches can rename entities.

**Fix:**
Replaced the manual normalization block with `normalize_process_payload` and an
`on_checkpoint` callback that writes the existing probe files. Gate failures are
now written to `gate_fail_report.json`, passed into the audit context, and
reported as audit input rather than a stopped pipeline. The post-audit mapping
pass now calls `map_payload` in memory with `use_cache=False` and writes the
same mapped payload/report artifacts.

**Pipeline consistency:**
The orchestrator now wires stage functions and artifacts only. Normalization
behavior remains in `process_normalizer.py`; mapping/cache behavior remains in
`map_ids.py`; the UI no longer owns enzyme-attachment logic or a parallel
normalization pipeline.

---

### 2026-07-07 - Normalizer actor lookup, evidence enzymes, pruning, and gate reporting

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/pipeline/qa_graph.py`, `tests/test_process_normalizer.py`, `docs/change_log.md`

**Error / symptom:**
Normalizer actor rows could still resolve a stale legacy `protein` or
`protein_complex` field before the canonical `entity` field. Enzyme mentions in
reaction evidence were wired in Streamlit as plain strings, disconnected
proteins were pruned without respecting mapped identity, and
`normalize_process_payload` could abort on gate failure before the audit loop
received the gate details.

**Root cause:**
Normalizer-owned actor interpretation and evidence wiring were split across the
orchestrator and normalization stage. Some compatibility reads still used
legacy field order. Protein pruning and gate handling also mixed pre-audit
cleanup with semantic rejection, which conflicts with the audit-loop contract in
`docs/pipeline.md`.

**Fix:**
Moved `_norm_text` and `attach_enzymes_from_reaction_evidence` into
`process_normalizer.py`, with cue-near-name matching and canonical actor dict
output. Updated normalizer and QA graph actor lookup to read `entity` before
legacy fields. Changed `prune_disconnected_proteins` to remove only degree-0
proteins with no external identity and record report details. Wired
`normalize_process_payload` to run the enzyme-evidence step, support checkpoint
callbacks, and return gate details in `report["gate"]` instead of aborting.

**Pipeline consistency:**
All deterministic cleanup remains in `process_normalizer.py`, and graph
connectivity interpretation remains in `qa_graph.py`. Streamlit does not gain
new normalization logic. Gate failures remain semantic audit input after
normalization, preserving the documented normalize-to-audit loop rather than
turning the normalizer into a pre-audit hard abort.

### 2026-07-07 - Add stage boundary contract validators

**Files changed:** `src/t2pw/pipeline/stage_contracts.py` (lines 1-273), `tests/test_stage_contracts.py` (lines 1-104), `docs/change_log.md`

**Error / symptom:**
Step 6 of the pipeline refactor needed a dedicated `stage_contracts` module so
stage boundary checks are explicit and testable. Without it, callers had no
single place to distinguish structural aborts from semantic gate failures that
must be sent to audit.

**Root cause:**
Boundary contract ownership was documented in `docs/pipeline.md`, but no module
implemented those boundaries. That made it easy to collapse pre-audit semantic
gate failures into hard abort behavior, which would bypass the audit loop that
is supposed to repair them.

**Fix:**
Added `StageContractError` plus validators for post-extraction, post-mapping,
post-normalization, post-audit, and pre-export boundaries. Structural
validators raise `StageContractError`; post-normalization returns semantic gate
failures as a report for audit; pre-export wraps
`validate_required_pwml_contract` failures in `StageContractError`. Added
focused unit tests for missing required boundary fields and PWML contract
wrapping.

**Pipeline consistency:**
The change lives entirely in `t2pw.pipeline.stage_contracts`, the module named
as the owner of stage boundary validation. It does not add normalization,
mapping, audit, UI, or PWML IR logic, and it keeps pre-audit semantic failures
as audit input instead of making them aborts.

---

### 2026-07-07 - Document pipeline payload schema types

**Files changed:** `src/t2pw/schema.py`, `docs/change_log.md`

**Error / symptom:**
Step 1 of the refactor needed `t2pw/schema.py` to document the JSON payload
contracts, but the module was empty.

**Root cause:**
The pipeline stages pass dictionaries whose expected shapes are documented in
the extraction prompts and `docs/pipeline.md`, while the schema ownership module
did not yet expose those contracts for type checkers or importers.

**Fix:**
Added `TypedDict` definitions for the payload, entity buckets, biological
states, locations, process rows, visualizations, mapping metadata, and inference
additions. `PayloadReactionActor` documents `entity` as the canonical actor name
while retaining backwards-compatible `protein` and `protein_complex` fields.

**Pipeline consistency:**
This change is type/documentation only and lives entirely in the schema module
that owns payload shapes. It does not add validation, normalization, UI logic,
or cross-stage behavior, so runtime pipeline behavior is unchanged.

---

### 2026-07-07 — Gate validation errors not shown in UI

**Files changed:** `src/t2pw/app/streamlit_app.py` (lines 2637–2651)

**Error / symptom:**
PWML export failed with "Hard-gate validation failed after normalization" but
the UI showed no detail about which specific checks failed, making the error
unactionable.

**Root cause:**
The `st.error()` call at the gate failure block only displayed the top-level
error string from `gate_fail_report`. The `errors` list inside the report
(which contains per-check path and reason) was never rendered.

**Fix:**
Expanded the gate failure display block to iterate `gate_fail_report["errors"]`
and show each entry as a formatted line (path + reason) inside an expander.

**Pipeline consistency:**
Change is entirely within the orchestrator's display logic. No stage function
was modified. The gate report structure is owned by `process_normalizer.py` and
was not changed — only the UI reading of it was corrected. This is a pure
orchestrator responsibility: surface what a stage reported.

---

### 2026-07-07 — Orphan proteins not pruned when not complex components

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (after line 1639)

**Error / symptom:**
Proteins appeared in `entities.proteins` with no reference in any reaction,
transport, or interaction, and no external database identity. These caused
the gate's `enforce_all_proteins_connected` check to fail.

**Root cause:**
`drop_unresolved_complex_component_proteins` (the existing pruning step) only
dropped proteins that appeared as components of a declared `protein_complex`
entity. Proteins that the LLM extracted standalone, with no complex membership
and no process reference, were never caught.

**Fix:**
Added `drop_process_orphan_proteins` to `process_normalizer.py`. It collects
all entity names referenced across reactions, transports, and interactions,
then drops any protein not in that set that also has no external identity
(`_has_protein_identity` returns False). Wired into `normalize_process_payload`
between the existing complex-component pruning step and `dedupe_processes`.

**Pipeline consistency:**
Change lives entirely within `process_normalizer.py`, which owns all
normalization steps. The new function follows the existing pattern: takes
payload + optional report dict, mutates payload in-place on the deep copy,
records dropped items in the report. No orchestrator or UI code was changed.
No new stage or module was created.

---

### 2026-07-07 - In-memory wrappers for mapping and audit stages

**Files changed:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/curation/audit_json_llm.py`, `src/t2pw/curation/apply_audit_patch.py`, `tests/test_map_ids.py`, `tests/test_audit_json_llm_payload.py`, `tests/test_apply_audit_patch_lock_policy.py`

**Error / symptom:**
Later orchestration work needed to pass Step 7 payload objects between mapping, audit, and patch application without forcing every stage through temporary JSON files. The existing file wrappers also made post-audit remapping vulnerable to stale cache reads unless callers could bypass or invalidate cache entries.

**Root cause:**
The core mapping implementation lived inside the file-based `run_mapping` adapter, so object-level orchestration could not reuse it directly. Audit and patch application had similar file-wrapper boundaries even though their core logic was already mostly payload-based. Mapping cache control was implicit in the cache file rather than exposed at the stage boundary.

**Fix:**
Added `map_payload` as the object-in/object-out mapping entry point and changed `run_mapping` to call it before writing the same mapped JSON/report files. Added `use_cache` and `invalidate_cache_keys` support to the mapping cache path for post-audit remapping correctness. Added `audit_payload` and `apply_audit_patch_payload` wrappers that reuse existing audit and patch core logic without duplicating manifest discovery. Added focused tests for the new wrapper contracts.

**Pipeline consistency:**
Mapping cache and ID assignment changes stay in `t2pw.mapping.map_ids`, which owns Stage 2 and post-audit remapping. Audit planning stays in `t2pw.curation.audit_json_llm`, and patch policy stays in `t2pw.curation.apply_audit_patch`. Existing `run_*` functions remain file adapters, and no Streamlit or normalization logic was moved into these stages.
