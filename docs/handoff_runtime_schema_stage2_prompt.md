# Codex handoff prompt: runtime payload schemas and live Stage 2 mapping

Copy the prompt below into a new Codex instance. The repository already contains
the preceding stage-contract consolidation; do not redo or revert it.

---

You are the primary review/integration agent for the T2PW repository at:

`C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW`

Your job is to implement two related pipeline-boundary improvements:

1. Add non-mutating runtime validation of the full known payload shape, with
   stage-aware expectations and a safe report-first rollout.
2. Wire the documented Stage 2 mapping boundary into the live Streamlit
   orchestrator before Stage 3 normalization, while retaining Stage 6 as the
   only wrapper-creating remap.

Use subagents for the implementation. The primary agent should coordinate,
review diffs, run integration verification, and catch cross-stage leakage. It
should not take over all coding work itself.

## Model/cost instructions

If the subagent interface exposes model selection, use the cheapest capable
code model for bounded mechanical tasks such as Pydantic model translation,
focused tests, documentation, and call-site updates. Use a mid-tier reasoning
model for the Streamlit orchestration change. Reserve the strongest or most
expensive model for primary-agent integration review or a subtask that has
failed with a cheaper model at least twice for the same substantive reason.
Do not spend a premium model on repository searches, formatting, changelog
entries, or straightforward fixture construction. Correctness takes priority:
upgrade a blocked task rather than allowing a weak implementation through.

If model selection is unavailable, say so once and continue with tightly scoped
delegation.

## Mandatory first reads

Before editing, the primary agent and each relevant subagent must read the
instructions and authoritative design sources needed for its slice:

- `AGENT_INSTRUCTIONS.md`
- `docs/pipeline.md`
- `docs/pathwhiz_requirements.md`
- Relevant entries in `docs/change_log.md`, especially the 2026-07-10 stage
  contract, identity, Stage 8, and live-orchestration entries, plus open mapping
  and enrichment issues, and the 2026-07-13 explicit PathBank `Unknown`
  fallback entry
- `src/t2pw/schema.py`
- `src/t2pw/pipeline/stage_contracts.py`
- `src/t2pw/pipeline/entity_identity.py`
- `src/t2pw/pipeline/process_normalizer.py`
- `src/t2pw/mapping/map_ids.py`
- `src/t2pw/app/streamlit_app.py`
- `src/t2pw/pwml/ir.py`
- Existing tests: `tests/test_stage_contracts.py`,
  `tests/test_entity_identity_contracts.py`, `tests/test_map_ids.py`,
  `tests/test_process_normalizer.py`, `tests/test_audit_json_llm_payload.py`,
  `tests/test_pwml_ir.py`, `tests/test_pwml_writer.py`, and
  `tests/test_pathbank_unknown_fallback.py`

Read the selected files completely enough to understand their APIs and failure
semantics. Use `rg` to find every caller before changing a signature. Do not
rely on this prompt where the current code gives a more precise answer.

## Current baseline and important repository state

- The previous integrated verification passed **310 tests** in the repository
  virtual environment.
- `requirements.txt` already includes Pydantic 2.x and `jsonschema`; do not add
  another validation dependency without a demonstrated need.
- `schema.py` contains static `TypedDict` documentation. Python does not enforce
  those shapes at runtime.
- `stage_contracts.py` enforces selected structural and semantic invariants,
  but it is intentionally not a complete recursive type validator.
- `entity_identity.py` is the neutral shared owner of protein/complex identity
  and species helpers. Do not duplicate that logic in a runtime model.
- Stage 8 no longer repairs bare protein enzymes. It fails and points back to
  Stage 6.
- Stage 6 currently calls `map_payload(...,
allow_complex_wrapper_creation=True, use_cache=False)` and then
  `validate_post_remap`.
- Stage 6 now has one explicit last-resort exception for a source-supported
  functional enzyme that remains unresolved after every normal mapping attempt.
  It retains the functional name on a generated complex and uses PathBank
  protein `9659` (`Unknown`, Arabidopsis species `4`) as its component, with
  `pathbank_unknown_protein_fallback` and `cross_species_placeholder`
  provenance. The exact sentinel is skipped by later UniProt enrichment.
  Preserve this behavior; do not move it into Stage 2 or treat `Unknown` as a
  normal UniProt accession.
- The live Streamlit post-pipeline function currently has no independent Stage
  2 `map_payload` call. The payload is normalized first and mapped only after
  audit/curation. This is the gap being fixed.
- `docs/pipeline.md` calls Stage 2 "Inference + Map". The LLM inference work is
  completed before `run_post_pipeline_sbml_artifacts(final_payload, ...)` is
  invoked. The missing database mapping belongs at the start of that function,
  before Stage 3 normalization. Do not create a second inference pass and do
  not broadly renumber stages.
- Stage 7 enrichment remains optional and has no dedicated exit validator. It
  is out of scope except where runtime reporting can observe that its additive
  output still matches the known payload shape.
- The worktree is intentionally dirty. Previous desired source/docs/test
  changes are uncommitted, and user-generated files under `data/`, `out/`,
  `outputs/`, and `tmp/` may also be modified. Preserve all of them. Never use
  `git reset --hard`, broad `git restore`, or checkout-based cleanup.
- Do not commit or push unless the user explicitly requests it.

## Non-negotiable pipeline rules

1. Structural garbage aborts at the stage that produced it; do not write or
   forward partial stage output.
2. Pre-audit semantic failures feed Stage 4; they do not abort before audit.
3. Runtime shape validation and semantic contracts are different layers:
   runtime models own field/container/value shapes; existing contracts and
   gates own biological meaning and cross-record resolution.
4. A validator must not silently normalize, coerce, delete, or rewrite the
   payload it is checking.
5. Unknown additive metadata must survive validation. The existing payload is
   broad and evolving, so do not start with `extra="forbid"` globally.
6. Stage 2 may add mapping IDs, species resolution, candidates, confidence, and
   `mapping_meta`; it must not create PathWhiz wrapper complexes or restructure
   processes.
7. Stage 6 remains the sole owner of generated single-protein wrapper creation.
   This includes the explicit PathBank protein `9659` fallback. It may activate
   only after normal mapping attempts fail and only in the wrapper-enabled
   Stage 6 pass.
8. Only Stage 1 and Stage 4 may set `spontaneous`.
9. Logic spanning stages belongs in the orchestrator, not inside a stage
   implementation.
10. Every landed code change requires a `docs/change_log.md` entry explaining
    symptom, root cause, and stage-consistent fix.

## Required subagent structure

Use the available concurrency without causing overlapping edits. A recommended
division is:

### Subagent A: runtime schema layer

Own:

- A new dependency-light module named
  `src/t2pw/pipeline/payload_models.py`
- Runtime-schema-focused tests in a new test file
- Minimal integration into `stage_contracts.py`

Do not edit Streamlit orchestration or mapping code.

### Subagent B: Stage 2 mapping behavior and boundary runner

Own:

- `src/t2pw/mapping/map_ids.py`
- A small pure boundary runner such as
  `src/t2pw/pipeline/stage_runners.py`, if it materially improves testability
- Focused mapping/boundary tests

This agent must audit and isolate the mapper's non-annotation cleanup behavior,
not merely pass the existing wrapper flag. Do not edit the runtime-schema
implementation or Streamlit.

### Subagent C: live orchestration, fixtures, and documentation

Own:

- `src/t2pw/app/streamlit_app.py`
- Focused orchestration tests that do not overlap Subagent A or B
- `docs/pipeline.md`
- `docs/change_log.md`
- Necessary corrections to stale flow descriptions in `AGENT_INSTRUCTIONS.md`

Run this slice after Subagent B's boundary API has settled, or give it an exact
agreed API before it starts. Update documentation only after behavior is
implemented; document what actually landed, not the intended design.

The primary agent owns integration review, cross-slice test runs, resolving
API mismatches, and ensuring agents did not rewrite one another's files. At
most one agent should edit a given file at a time.

Use two waves if needed: A and B can work in parallel; C consumes their stable
APIs. Do not run all three into overlapping orchestration/contract edits.

## Part A: runtime payload schema implementation

### Design target

Keep the existing `TypedDict`s for static typing and documentation. Add
Pydantic v2 runtime models as an additive layer; do not rewrite every consumer
to pass Pydantic objects instead of dictionaries.

Pydantic v2 is already used elsewhere in this repository. Prefer it over
maintaining a parallel handwritten JSON Schema. Do not validate and then
forward `model_dump()` output: that could coerce values or reshape extras and
would accidentally turn validation into another mutating pipeline stage.

The validator should accept a normal Python dictionary and return a structured
report. Validation must not replace or mutate the original payload.

Use a small public API along these lines, adapting names only if current code
suggests a better fit:

```python
validate_payload_shape(
    payload: Any,
    *,
    boundary: Literal[
        "post_extraction",
        "post_mapping",
        "post_normalization",
        "post_audit",
        "post_remap",
        "post_enrichment",
        "pre_export",
    ],
    mode: Literal["report", "enforce"] = "report",
) -> RuntimeSchemaReport
```

The report should be JSON-serializable and stable enough for Streamlit and
tests, for example:

```json
{
  "ok": false,
  "boundary": "post_normalization",
  "mode": "report",
  "errors": [
    {
      "code": "runtime_schema_type_error",
      "pointer": "/processes/reactions/3/enzymes/0/entity",
      "message": "Expected a string",
      "expected": "string",
      "received": "list"
    }
  ],
  "warnings": [],
  "summary": { "error_count": 1, "warning_count": 0 }
}
```

Pydantic error locations must be converted to escaped JSON Pointers. Do not
expose raw Python tuples as the only location format.

### Stage-aware shape expectations

Model the known recursive core shape, including at minimum:

- Top-level payload, metadata, entities, processes, biological states, and
  element locations
- Named entity rows and their important nested IDs/mapping metadata
- Proteins and protein complexes, including structured/string components for
  legacy-compatible input and the stricter generated-wrapper output shape
- Reactions, transports, interactions, actors, participants, and evidence
- `spontaneous` as a real boolean when present
- Mapping resolution shape (`status`, `issue`, `order_step`) without pretending
  every optional provider-specific field is already enumerated

Use boundary-specific models or boundary-specific post-model checks so the
expectations grow by stage:

- Post-extraction: incomplete biological knowledge is allowed, but known
  containers and present values must have the correct types.
- Post-mapping: named entity rows require `mapping_meta`; species must be a
  non-empty list. Unmapped status is valid.
- Post-normalization: actor rows in reaction enzymes/modifiers, transporters,
  and interaction participants require non-empty string `entity` and
  `entity_type`.
- Post-audit: same runtime shape as normalized output. Do not enforce semantic
  gate success here.
- Post-remap: generated wrapper rows have boolean `generated`, the supported
  generation reason, and structured components. Cross-record identity/species
  resolution remains in `validate_post_remap`; do not duplicate it in Pydantic.
- Post-enrichment: accept additive enrichment metadata and verify it did not
  corrupt the core payload.
- Pre-export: shape validation runs before the existing semantic PWML contract;
  do not replace `validate_required_pwml_contract`.

### Strictness and rollout

Implement both modes:

- `report`: return errors without raising; never mutate the payload.
- `enforce`: raise a dedicated exception carrying the same structured report.

The initial live rollout should be safe and observable:

- Use report mode for newly introduced full-recursive validation unless a
  boundary's existing structural contract already aborts for the same class of
  failure.
- Preserve all current `StageContractError` and semantic gate behavior.
- Make it possible to opt into enforce mode in tests and through one explicit,
  documented configuration point. Do not scatter environment-variable checks
  through stage modules.
- Do not make loose LLM output unrepairable by enforcing semantic completeness
  at extraction.

Prefer `ConfigDict(extra="allow", strict=True)` or equivalent for the known
core models, but verify it against real fixtures. Extra fields must be
preserved, and validation must not rely on Pydantic coercion to turn malformed
data into valid data. If strict mode is incompatible with legitimate current
fixtures, document and isolate the exact exception rather than switching the
entire model layer to permissive coercion.

The live payload contains legitimate fields that are not completely reflected
in the current `TypedDict`s, including top-level `metadata`, `species_ref`, and
provider/review/candidate/issue metadata. Add known fields deliberately; do not
mistake incomplete static typing for invalid production data.

### Relationship to `stage_contracts.py`

Integrate through one adapter/helper, not repeated calls scattered throughout
each validator. Existing manual contracts remain authoritative for semantic
and cross-record rules. Avoid circular imports: the runtime-schema module must
not import `stage_contracts.py`, `process_normalizer.py`, mapping, or PWML code.

Preserve the existing `StageContractError`, `stage`, `contract_type`,
`effect_on_failure`, `errors`, `warnings`, and summary semantics. If the shape
and semantic layers identify the same problem, deduplicate by stable
`(code, pointer)` identity rather than reporting confusing duplicates.

Do not duplicate these checks in Pydantic:

- Protein/complex identity resolution
- Graph connectivity
- Composite detection
- Registry reference resolution
- Rails/PathWhiz semantic validation

Those stay in their existing owners.

## Part B: wire the live Stage 2 mapping boundary

### Exact orchestration location

The insertion point is near the beginning of
`run_post_pipeline_sbml_artifacts` in
`src/t2pw/app/streamlit_app.py`, after paths/configuration are prepared and
before `normalize_process_payload` receives its input.

The `final_payload` parameter already contains the settled extraction plus LLM
inference additions. The desired order is:

```python
validate_post_extraction(final_payload)  # or the appropriate structural input check

stage2_result = map_payload(
    deepcopy(final_payload),
    cache_path=cache_path,
    id_source=id_source,
    db_config=db_config,
    use_cache=True,
    allow_complex_wrapper_creation=False,
)
stage2_payload = stage2_result["payload"]
stage2_mapping_report = stage2_result["report"]
validate_post_mapping(stage2_payload)

normalized_payload, normalization_report = normalize_process_payload(
    stage2_payload,
    ...,
)
```

Construct `db_config` once before both mapping passes so Stage 2 and Stage 6
receive the same database configuration. Do not duplicate credential assembly.

Retain the post-curation Stage 6 call with:

```python
map_payload(
    curated_payload,
    ...,
    use_cache=False,
    allow_complex_wrapper_creation=True,
)
validate_post_remap(stage6_payload)
```

Do not rely on `map_payload`'s compatibility default at either live call site;
pass the wrapper flag explicitly.

### Prevent Stage 2 structural bleed

Do not assume that `allow_complex_wrapper_creation=False` makes `map_payload`
annotation-only. Audit its complete post-mapping path first. The current mapper
also contains complex-component cleanup/pruning and protein-removal behavior.
Stage 2 must not silently remove entity rows, rewrite process actors, or perform
Stage 6 structural cleanup.

Add the smallest explicit API control needed to make initial mapping
annotation-only, such as a compatibility-defaulted `allow_structural_cleanup`
flag or a clearly named mapping-phase enum. Choose after reading every caller;
do not add several unrelated booleans without need. The live invariants are:

- Stage 2 may add/update mapping IDs, `mapping_meta`, candidates, confidence,
  provenance, and approved species hydration.
- Stage 2 preserves process structure and the non-species entity inventory.
- Stage 2 never creates a row with `generated=true`.
- Stage 2 never creates or injects the PathBank `Unknown` sentinel, even when a
  bare enzyme remains unresolved.
- Stage 6 retains the existing cleanup/remap/wrapper behavior.

Defaults must preserve current batch/legacy caller behavior, while both live
Streamlit calls pass their phase/mutation policy explicitly.

### Resolve the current post-mapping contract mismatch

Before wiring the call, compare `validate_post_mapping` with every entity bucket
that `map_payload` actually owns. The current contract iterates every named
`entities.*[]` bucket, while mapping may not populate `mapping_meta`
consistently for cell types, tissues, subcellular locations, element
collections, nucleic acids, bounds, and similar non-ID-mapped rows.

Use fixture evidence and choose one documented policy:

1. Preserve the documented "every named entity has mapping metadata" guarantee
   by stamping an explicit `resolution.status="not_applicable"` or equivalent
   record on buckets that mapping intentionally does not resolve; or
2. Define the exact mappable buckets, narrow the contract to those buckets,
   and retain separate structural rules for all other buckets.

Whichever policy lands, validate that `mapping_meta` is an object and that the
known `resolution.status`, `issue`, and `order_step` values have the documented
shape. Checking key presence alone is insufficient. Do not weaken or broaden
the guarantee silently.

### Separate artifacts and observability

Do not overwrite Stage 2 results with Stage 6 results. Use distinct names,
paths, return keys, and UI downloads, such as:

- `stage2.mapped.json`
- `stage2_mapping_report.json`
- `stage2_payload`
- `stage2_mapping_report`
- `stage2_runtime_schema_report`
- Existing `final.mapped.json` / `mapping_report` remain Stage 6 artifacts

Preserve both the pre-mapping extracted/inferred payload and the exact payload
given to Stage 3. Clarify or rename misleading `pre_normalization_input` fields
only if tests and every UI consumer are updated. Backward-compatible aliases
are preferable during migration.

Expose enough in the Streamlit result/UI to answer:

- Did Stage 2 mapping run?
- How many entities mapped, were ambiguous, or remained unmapped?
- Did any wrapper get created during Stage 2? This count must be zero.
- What exact mapped payload entered Stage 3?
- How did Stage 6 change mappings after audit/curation?

### Failure behavior

- If `map_payload` raises or returns a malformed result, Stage 2 must not write a
  successful mapped artifact and Stage 3 must not run.
- A structurally invalid post-mapping payload raises through
  `validate_post_mapping` and is shown as a Stage 2 boundary failure.
- Individual unmapped entities are not a structural failure if they carry the
  required mapping metadata. Their semantic identity gaps should reach Stage 3
  and Stage 4.
- Network/API/DB unavailability must use the mapper's existing fallback and
  reporting behavior. Do not add hidden network calls to validators or tests.
- Low-confidence or `best_effort` mappings must retain candidates, confidence,
  method/rule, and provenance so Stage 4 can challenge them. An ID's presence
  alone must not erase uncertainty.

The mapper has known behavior that may accept a first ambiguous or best-effort
UniProt candidate. Surface these rows/counts in the Stage 2 report. Do not
expand this task into redesigning identity acceptance thresholds unless tests
prove the boundary is incorrect without it; record a focused follow-up instead.

### Cache and latency constraints

- Stage 2 should use the configured cache under its existing semantics to avoid
  doubling external lookup cost on every run.
- Stage 6 remains cache-bypassed because names may have changed during audit or
  curation.
- Do not mutate cached rows into generated wrappers during Stage 2.
- Add timing/count information to reports if it is already easy to derive, but
  do not broaden scope into a performance rewrite.

## Required tests

All tests must avoid real network and database calls. Use monkeypatches/fakes.

### Runtime schema unit tests

Cover at minimum:

1. A representative valid payload passes each appropriate boundary.
2. A nested wrong type produces a precise JSON Pointer.
3. `spontaneous: "true"` is rejected rather than coerced to a boolean.
4. A string/list/object actor-shape error is reported at the correct actor.
5. Unknown additive metadata is accepted and the original payload is unchanged.
6. Report mode returns a report and does not raise.
7. Enforce mode raises an exception containing the same report.
8. Post-mapping requires mapping metadata but permits an explicit unmapped
   resolution.
9. Post-normalization requires canonical actor fields.
10. Post-remap rejects malformed generated-wrapper shape while the existing
    semantic contract still owns declared-protein identity/species checks.

### Stage 2 orchestration tests

Cover at minimum:

1. The live post-pipeline orchestration calls mapping twice, in order:
   Stage 2 with wrapper creation false/cache enabled, then Stage 6 with wrapper
   creation true/cache disabled.
2. The payload returned by Stage 2 is the exact input to Stage 3 normalization.
3. `validate_post_mapping` is called immediately after Stage 2.
4. Stage 2 creates no generated wrapper even for a bare protein enzyme.
5. Stage 2 preserves process structure and non-species entity inventory; no
   cleanup/pruning branch runs during the annotation-only pass.
6. Stage 6 may create the required wrapper from the post-curation payload.
7. Stage 2 and Stage 6 mapping reports/artifacts remain distinct.
8. A Stage 2 exception or malformed result prevents normalization/audit/remap.
9. Unmapped-but-well-formed entity metadata feeds Stage 3/audit rather than
   causing a structural abort.
10. Existing audit cadence still reruns only the strict gate after accepted
    patches; do not accidentally add a full normalize or map inside each round.
11. Both Streamlit call sites of `run_post_pipeline_sbml_artifacts`, including
    the legacy-SBML option, use the same corrected orchestration function.
12. Mappable and non-mappable entity buckets obey the selected mapping metadata
    policy.
13. The existing PathBank `Unknown` fallback remains Stage-6-only: an unresolved
    enzyme does not create protein `9659` during Stage 2, while Stage 6 may use
    it only after real mapping attempts fail.

### Regression and integration verification

- Existing Stage 3 actor, generated-wrapper, spontaneous, PWML, and writer
  regressions must remain green.
- Keep `tests/test_pathbank_unknown_fallback.py` green, including catalytic
  modifier synchronization, Stage 2 non-activation, enrichment bypass,
  idempotent reuse, and exact PWML serialization of protein `9659`.
- Add a paper-like fixture with one species, compounds, a protein enzyme, and a
  reaction. Assert the stage reports show the intended progression without
  using live services.
- Check that a low-confidence/best-effort mapping remains visibly uncertain in
  the Stage 2 report and payload.

## Documentation requirements

Update `docs/pipeline.md` to remove the note that the live Streamlit path lacks
Stage 2 mapping only after the call is genuinely wired. Document:

- Runtime shape validation versus semantic contracts
- Report versus enforce behavior
- Stage 2 cache/wrapper policy
- Separate Stage 2 and Stage 6 artifacts
- The fact that Stage 2 receives the already merged extraction/inference
  payload in the current UI architecture

Where the UI uses ambiguous names, prefer a small clarification such as
"Stage 2A Infer" and "Stage 2B Map" instead of renumbering the pipeline.
`AGENT_INSTRUCTIONS.md` contains older high-level flow/path descriptions;
update only portions made false by this work. If the UI labels gap resolution
as Stage 3 even though `docs/pipeline.md` owns it as Stage 4a, correct or record
that mismatch without turning this into an unrelated UI rewrite.

Add one or more `docs/change_log.md` entries following the repository template.
Do not claim complete JSON-schema enforcement if models intentionally allow
unknown additive fields.

## Verification commands

Use the repository virtual environment. Prevent tracked bytecode/cache churn:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Run focused tests after each slice, then the full suite. Run Ruff on changed
files. A repository-wide Ruff run currently has unrelated pre-existing
findings, including the Streamlit path-bootstrap `E402` pattern; do not rewrite
unrelated files to make global Ruff green. For the app, use the established
targeted exception if no new lint class was introduced:

```powershell
.\.venv\Scripts\python.exe -m ruff check --ignore E402 src\t2pw\app\streamlit_app.py
```

Also run:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  src\t2pw\pipeline\pipeline.py `
  src\t2pw\pipeline\stage_contracts.py `
  src\t2pw\pipeline\payload_models.py `
  src\t2pw\pwml\writer.py `
  scripts\run.py `
  scripts\run_pwml.py
git diff --check
```

If `py_compile` changes tracked `__pycache__` files, restore only those generated
artifacts without touching source or user files.

## Completion criteria

Do not declare completion until all of the following are true:

- The runtime model layer validates the known recursive core without mutating
  payloads or dropping additive metadata.
- Structured reports have stable JSON Pointers and both report/enforce modes.
- Existing manual contracts retain their semantic ownership.
- A real Stage 2 mapping call runs before Stage 3 in the live Streamlit
  orchestrator.
- Stage 2 explicitly disables wrapper creation and Stage 6 explicitly enables
  it.
- The explicit PathBank `Unknown` fallback remains confined to Stage 6 and does
  not bypass any real mapping attempt.
- Stage 2 and Stage 6 artifacts/reports are separately observable.
- Failure paths stop at the owning stage and do not forward partial output.
- Focused and full tests pass in `.venv`.
- Changed-file Ruff, compile checks, and `git diff --check` pass.
- No user-generated cache/output files or previous desired changes were
  reverted.
- Documentation and change-log entries describe the behavior that actually
  landed.

In the final response, lead with the outcome, list the important files changed,
report exact test/lint results, disclose any intentionally report-only boundary
or remaining limitation, and do not commit unless explicitly asked.
