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
- **Runtime shape validation** recursively checks known field, container, and
  value types without rewriting the payload or rejecting unknown additive
  metadata. It is separate from biological and cross-record semantics. The
  live rollout uses `report` mode through the single
  `stage_contracts.RUNTIME_SCHEMA_MODE` configuration point. Tests and
  deployments may opt into `enforce`, which raises with the same structured
  report. Existing structural aborts and the semantic PWML gate remain
  authoritative.
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

### Enforced input/output contracts

The canonical JSON shapes below are the `TypedDict` definitions in
`t2pw.schema`. A stage may add optional metadata, but it may not change the
meaning or shape of a field owned by an earlier stage.

| Stage | Concrete input | Concrete output and exit guarantee | Boundary owner |
| --- | --- | --- | --- |
| 1 — Extract | Paper text plus pathway/user context (not JSON) | `Payload`; named entity rows use `PayloadCompound`, `PayloadProtein`, `PayloadProteinComplex`, and related entity types; process rows use `PayloadReaction`, `PayloadTransport`, and `PayloadInteraction`. Actor rows use `PayloadReactionActor.entity` plus `.entity_type`. Spontaneity is not modeled: `PayloadReaction.spontaneous` is always `false`. | `validate_post_extraction`: `entities`/`processes` objects exist, entity names are non-empty, process rows are objects, and each known process bucket satisfies its own participant/reference shape. Structural failure aborts. |
| 2 — Map | The already merged Stage 1 extraction + Stage 2A inference `Payload` | The same `Payload`; every named entity has a `PayloadMappingMeta` object containing a `resolution` object with non-empty string `status` and string `issue`/`order_step` when present, and `PayloadEntities.species` is non-empty. The live pass uses cache and calls `map_payload(..., use_cache=True, allow_complex_wrapper_creation=False, allow_structural_cleanup=False)`; it annotates but cannot prune structure or create generated PathWhiz wrappers. | `validate_post_mapping`; structural failure aborts. |
| 3 — Normalize | Stage 2 `Payload` | `(Payload, normalization_report)`. Reaction enzymes/modifiers, transporters, and interaction participants have canonical non-empty `entity` and `entity_type`. The report contains the pointer-addressable strict gate result. | `validate_post_normalization(payload, gate)`. Structural failure aborts; semantic failures are returned with `effect_on_failure=feed_audit`. |
| 4 — Audit / gap resolve | Stage 3 `Payload` plus its strict gate report | Patched `Payload` preserving the Stage 3 shape. Audit never sets `PayloadReaction.spontaneous`; spontaneity is not modeled and the field stays `false`. After every selected patch with at least one accepted operation, the orchestrator runs `run_strict_post_normalization_gates` (not the full normalizer) and supplies that fresh result to the next round. | `validate_post_audit`; malformed patched output aborts. |
| 5 — Curate | Post-audit `Payload` | Same `Payload` shape. If curation produces structurally invalid output, the orchestrator retains the pre-curation payload. | Stage 1 structural contract reused by the curator fallback policy. |
| 6 — Remap | Curated `Payload` | Same `Payload`, with refreshed IDs and mapping cache bypassed. This is the sole pass allowed to call `map_payload(..., use_cache=False, allow_complex_wrapper_creation=True, allow_structural_cleanup=True)`. Generated `PayloadProteinComplex` rows carry `generated=true` and `generation_reason="single_protein_pathwhiz_wrapper"`; every component resolves to a declared protein with species and UniProt/DrugBank identity. | `validate_post_remap`; invalid generated wrappers abort before export. |
| 7 — Enrich (optional) | Post-remap `Payload` | Same `Payload` with additive enrichment metadata only; process and identity contracts do not change. | Runtime shape report at `post_enrichment`; Stage 6 and pre-export semantic contracts remain authoritative. |
| 8 — Export | Post-remap/enriched `Payload` after optional user-requested grounding | PWML IR (`Dict[str, Any]`) and serialized PWML XML. Reactions export `spontaneous=false` unconditionally (spontaneity is not modeled); enzyme references are unique protein complexes; protein species serialization uses each protein's resolved species. Stage 8 is validation-only: it does not rerun normalization, create autostates, map or infer entities, or create a last-resort wrapper. | Validation-only `run_strict_post_normalization_gates`/`validate_post_normalization`, followed by `validate_pre_export`, which wraps `validate_required_pwml_contract`; semantic failure aborts. |

The live Streamlit post-pipeline function now runs both mapping boundaries. It
validates the already merged extraction/inference payload, runs the
annotation-only Stage 2B pass, validates that result, and gives that exact
payload to Stage 3. After audit and curation it runs the wrapper-enabled Stage
6 remap. The two passes share one database configuration but have distinct
cache/mutation policies and distinct payload, report, and runtime-schema
artifacts.

If a stage contract aborts this live flow, the UI identifies the exact boundary,
states which downstream stages did not run, lists every issue with its code and
JSON pointer, and exposes the full structured report as a JSON download. A
`post_extraction` failure here is specifically labelled as the Stage 2A merged
payload to Stage 2B database-mapping input boundary; it occurs before the mapper
or audit is called.

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

The extraction prompt also enforces a name-based `protein_complexes` routing rule: any entity whose own name contains the word "complex" (e.g. "pyruvate dehydrogenase complex") must be extracted under `protein_complexes[]`, never `proteins[]`, even when subunit membership is not stated in the source text. This exists because the Stage 3 gate (below) rejects a `proteins[]` row named `"... complex"` as a misrouted entity, and that rejection is a hard export blocker with no auto-repair step — the correct fix is preventing the misrouting at its source rather than reclassifying it later.

**Structural contract enforced here:**
- Valid JSON
- Top-level keys `entities` and `processes` both present
- Every entity has a non-empty `name` field
- Every process row is an object, with bucket-specific structure:
  reactions have a usable input or output; transports have cargo or a
  state-bearing element; interactions have two endpoints or a usable
  participant; reaction-coupled transports have paired references or a
  state-bearing element; and sub-pathways have a usable name, reference ID, or
  alternate text. Unknown additive process buckets remain accepted when their
  rows are objects.

If this contract fails, abort — no downstream stage can operate on malformed
input.

---

### Stage 2A — Infer / Stage 2B — Map

**Inference prompt:** `pwml_infer_system.txt` (LLM modifier repair and enrichment pass)
**Module:** `t2pw.mapping.map_ids.map_payload`
**Live output files:** `stage2.mapped.json`, `stage2_mapping_report.json`, and
`stage2_runtime_schema_report.json`

An LLM inference pass runs after Stage 1 extraction. It receives the Stage-1 JSON and proposes conservative additions: missing modifier links (mandatory repair pass over all Stage-1 proteins and protein complexes), missing biological states, compartment assignments, and synonym bridges. The modifier repair pass applies the species scoping rule from Stage 1 — it only adds modifiers for proteins belonging to the selected pathway organism and does not link proteins from other organisms mentioned in the source text.

Following inference, database ID lookup runs against PathBank and configured ID
sources. The current UI completes and merges inference before calling the
post-pipeline orchestrator; the orchestrator does not run a second inference
pass. Stage 2B writes `mapped_ids` and `mapping_meta` onto each entity and
hydrates species references. Buckets outside direct ID mapping, including
subcellular locations, receive an explicit `not_applicable` resolution.

The live Stage 2B call uses the configured mapping cache, disables wrapper
creation, and disables structural cleanup. It may add mapping candidates,
confidence, provenance, IDs, and approved species hydration, but it preserves
process structure and the non-species entity inventory. Every named bucket is
given an explicit resolution: directly mapped buckets record their actual
status, while buckets outside ID resolution record `status="not_applicable"`.
Unmapped and low-confidence rows remain valid Stage 2 output and stay visible
to Stage 3 and audit.

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
14. `dedupe_processes` — classify no-op/subset reactions, quarantine source-supported non-exportable claims, and collapse genuine duplicates
15. `drop_process_orphan_proteins` — drop standalone proteins never referenced by a surviving process and with no external identity
16. `prune_disconnected_proteins` — graph-based pass: drop degree-0 proteins with no external identity
17. `run_strict_post_normalization_gates` — generate gate report

Reaction classification now precedes the final two orphan passes. This ordering
ensures that an enzyme referenced only by a removed or quarantined reaction does
not remain in the active entity registry and later fail identity validation. A
protein survives cleanup when it is still referenced by a valid reaction,
interaction, transport, or surviving protein complex, has non-zero graph degree,
or has an external database identity.

Stage 3 treats biological correctness as an export requirement. A normalized
`A -> A` reaction, or a reaction whose outputs are only a subset of its inputs,
does not become exportable because it is `locked` or `essential`. Unsupported
unlocked no-ops are dropped. Locked or directly evidenced coarse-grained
transformations are removed from active processes and appended to the existing
`quarantined_locked_reactions` ledger with their original reaction, JSON
pointer, action, and a stable reason such as
`coarse_grained_same_entity_transformation` or
`output_subset_of_input_without_distinct_product`. A lock therefore guarantees
traceability, not export. Every lock must be accounted as either active or
quarantined; distinct locked duplicates are likewise quarantined instead of
being silently collapsed. The strict Stage 3 gate blocks a payload when
`locked_reaction_filter_report.unaccounted_locked_reactions` is positive or
malformed, at the exact accounting-field pointer. An accounted quarantine does
not fail the gate.

Immediately after normalization, the live orchestrator refreshes the canonical
`tmp/quarantined_locked_reactions.json` from the normalized payload's ledger,
before audit starts. This prevents the earlier pre-normalization empty artifact
from hiding Stage 3 classifications. The refreshed object is also returned in
post-pipeline artifacts and exposed by the JSON artifact viewer/downloads.

Step 17 runs inside `normalize_process_payload` and collects all hard-gate failures into a structured report. **It does not raise an exception that aborts the pipeline.** The gate failures are returned as part of the normalization report and passed to the audit loop.

Reaction-evidence attachment is idempotent across supported reruns. It may
treat a generated single-protein PathWhiz wrapper as equivalent to its member
only when the complex has the expected generation provenance, exactly one
component, and that component resolves to a declared protein. Thus evidence for
`NdmA` does not re-add a bare `NdmA` actor beside a valid `NdmA complex` actor.
Ordinary biological complexes, malformed wrappers, and multi-protein complexes
are not treated as aliases of one member.

An `on_checkpoint` callback can be passed by the orchestrator to write probe
files at named checkpoints without splitting the normalization into two pipelines.

Stage 3 must preserve the entity registry distinction established upstream. A
name already declared in `entities.protein_complexes` must not be synthesized
again under `entities.proteins` merely because the same name appears in a
process or `element_locations.protein_locations` reference. Protein versus
protein-complex name collisions are gate failures; deterministic normalization
may canonicalize a reference, but it must not silently change a protein complex
into a protein.

The full normalizer runs only at Stage 3. After Stage 6, the live export path
may apply explicitly requested grounding and then reruns only the Stage 3 strict
gate plus its boundary validator. This is a **validation-only pre-export Stage
3 revalidation**, not another normalization pass: it does not create
autostates, attach or promote actors, remap entities, run inference, or create
wrappers. A failure there means the pipeline reached final remap but the exact
remapped payload is unsafe for Stage 8.

---

### Stage 4 — Audit

**Module:** `t2pw.curation.audit_json_llm.run_audit` + `t2pw.curation.apply_audit_patch`
**Output files:** `audit_report.json`, `audit_patch.json`, updated payload JSON

The primary semantic repair stage. It receives the normalized payload and the
gate report. If the gate passed with no failures, the loop is skipped. If gate
failures exist, the loop runs:

Quarantined coarse-grained reactions may return to the active payload only when
the paper supplies enough evidence to identify biologically distinct
participants. Stage 4 must not invent precursor/product names merely to turn an
`A -> A` claim into an exportable equation.

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
the audit — it supplements it with targeted ID and location resolution for
flagged entities. It does not rewrite reaction structure or actor roles. For a
declared complex, it may resolve existing component names against mapped protein
rows, but evidence-dependent stoichiometry remains an audit responsibility.

#### Current implementation status (2026-07-13)

Completed and verified:

- Stage 3 now classifies biologically non-exportable same-label/subset reactions
  before orphan cleanup. Locked/evidenced claims are quarantined with provenance,
  unsupported no-ops are dropped, and preservation accounting recognizes the
  quarantine ledger.
- Stage 2B now runs as a distinct annotation-only mapping boundary before
  normalization, while Stage 6 remains the only wrapper-creating remap.
- Runtime payload reports cover the live Stage 2 and pre-export boundaries.
- Post-extraction contracts validate each process bucket by its own shape, so
  valid interactions are not rejected by reaction-specific requirements.
- Contract failures in the Streamlit UI identify the exact boundary, skipped
  stages, issue codes, JSON pointers, and provide the full JSON report.
- Pre-export Stage 3 gate failures stop PWML generation instead of being hidden
  behind a later generic export error.
- Stage 3 now preserves known protein-complex names in process and location
  references, refuses to synthesize a same-name protein, and gates protein versus
  protein-complex registry collisions.
- Gap Resolve now executes protein-complex issues, hydrates declared components
  from mapped proteins, leaves unsupported stoichiometry explicitly unresolved,
  and treats the complex-level PathBank ID as optional for a valid novel complex.
- Stage 4 audit now emits pointer-addressed component-stoichiometry errors and
  applies exact counts only from unambiguous named-component evidence. The
  `NdmCDE` evidence produces accepted `3/3/3` patches; ambiguous evidence never
  defaults to `1`.
- Audit convergence now includes gap-only payload changes and unresolved gap
  issues, with a fresh strict gate after every changed settled payload and
  bounded unchanged/repeated/timeout/max-round exits.
- Gap location ranking now filters eukaryotic-only organelles from confirmed
  prokaryotic pathways before LLM selection.

Issues reproduced by the `NdmCDE` run and repaired in code:

1. `normalize_composites` created a second bare `NdmCDE` protein from a
   protein-location reference. Type-aware registry preservation now prevents it.
2. `protein_complex:ndmcde` was skipped as `issue_not_found`. Complex rows are
   now part of the resolver execution index.
3. Component strings lacked member identity and stoichiometry. Gap Resolve now
   hydrates member IDs, and audit derives exact ratios only from explicit source
   evidence.
4. The loop stopped on zero accepted audit patches despite gap-only progress.
   Settled-payload change, fresh gate state, and actionable gap issues now govern
   convergence.
5. Global PathBank frequency selected endoplasmic-reticulum membrane for
   *Pseudomonas putida*. Organism compatibility now rejects that candidate.

Implemented stage-owned remediation:

1. **Stage 3 — normalization and gate:** type-aware registry lookup preserves
   complex references and rejects protein/complex collisions without mapping IDs
   or inventing stoichiometry.
2. **Stage 4a — Gap Resolve:** complex issues execute and member names resolve to
   declared mapped proteins. Missing ratios remain explicit audit-owned issues.
3. **Stage 4 — audit and orchestration:** exact evidence-backed ratios become
   policy-checked patches; gap-only changes receive a fresh gate and another
   eligible round while all loop bounds remain active.
4. **Stage 4a — location resolution:** organism/taxonomy context filters clearly
   incompatible organelles before selection.
5. **Stage 6 — remap:** continues to consume the settled structure and refresh
   IDs without taking over classification or biological repair.
6. **Stage 8 — export:** remains the hard semantic guard, and UI errors identify
   its preceding check as validation-only `pre-export Stage 3 revalidation`.
7. **Regression coverage:** fixtures cover location-reference duplication,
   complex execution/member hydration, explicit and ambiguous stoichiometry,
   gap-only convergence, prokaryotic location filtering, and the real failed
   artifact's repair shape; an in-memory integration replay covers the actual
   saved artifact.

Remaining validation and intentional limits:

- The fresh paper run completed the configured DB/LLM stages and produced the
  final Stage 6 artifact. After the Stage 8 repair, that exact artifact also
  passes programmatic PWML export; only a final manual export click in the live
  Streamlit UI remains.
- Missing, approximate, ranged, or conflicting component counts remain blocking
  audit errors by design; the pipeline will not invent a ratio.
- Organism compatibility uses resolved taxonomy/lineage plus a conservative
  marker vocabulary. Broader taxa may require vocabulary expansion as new live
  cases are observed.

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

The live call explicitly enables both wrapper creation and structural cleanup.
Its `final.mapped.json` and `mapping_report.json` artifacts remain separate from
the Stage 2B artifacts so mapping changes made after audit/curation can be
compared directly.

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

One explicit last-resort exception exists for functional reaction enzymes and
transporters whose normal PathBank, UniProt, DrugBank, alias, literature, and
API retry strategies all fail. Stage 6 may retain the functional protein name
as a generated complex and use PathBank's known `Unknown` protein row as its
sole component. That sentinel is serialized exactly as PathBank protein `9659`,
name/UniProt value `Unknown`, species *Arabidopsis thaliana* (`species_id=4`,
taxon `3702`). The complex and sentinel carry
`chosen_rule=pathbank_unknown_protein_fallback`, `cross_species_placeholder=true`,
and the original target organism in mapping metadata. This fallback:

- runs only in the wrapper-enabled Stage 6 pass, never Stage 2;
- applies to a protein only when its *sole* role in the payload is as a
  reaction catalyst (including catalyst modifiers promoted by Stage 3) or as a
  transport transporter — never to a protein also referenced as a reaction
  input/output, a non-catalyst/non-transporter modifier, an interaction
  participant, or a complex component; each role is tracked and counted
  separately (`reaction_enzyme_unknown_fallbacks` vs.
  `transporter_unknown_fallbacks`);
- therefore applies to an unresolved catalyst or transporter on a biologically
  valid surviving reaction/transport, but not to an orphan from a removed or
  quarantined reaction, nor to a protein with any other reference elsewhere;
- never replaces a real protein or protein-complex mapping;
- preserves non-catalytic, non-transporter process references instead of
  deleting their source protein; and
- is skipped by later UniProt enrichment and recognized on repeated mapping so
  it cannot recursively wrap or duplicate `Unknown`.

---

### Stage 7 — Enrich (optional)

**Module:** `t2pw.mapping.enrich_entities.run_enrichment`

Fetches additional metadata (synonyms, cross-references, properties) and writes
`entity["enrichment"]`. Currently this data is not read by any downstream stage.
Decision pending: either wire it into the PWML IR builder or remove this stage.
The live orchestrator records a report-mode runtime shape report after this
additive pass and another at pre-export. These observations do not replace the
Stage 8 semantic PWML contract.

---

### Stage 8 — Export

**Modules:** `t2pw.pwml.writer` (primary), `t2pw.sbml` (legacy)

Converts the final payload to PWML XML (primary) or SBML (legacy).

For PWML, `run_pwml_export` deep-copies the Stage 6 payload, optionally applies
the user-supplied grounding dictionary, and then validates without repairing.
It must not call the full Stage 3 normalizer or any autostate, mapping,
inference, actor-attachment, or wrapper-creation step. Stage 6 remains the sole
owner of generated PathWhiz wrappers and reaction remapping.

**Semantic contract enforced here (hard abort):**
- `validate_required_pwml_contract` from `t2pw.pwml.ir` — called before IR
  construction
- Checks: all process actors resolve to declared entities, all required DB IDs
  present (if strict mode), no scaffold modifiers, no unresolved composites
- Generated protein complexes are valid without a complex-level PathBank ID only
  if their member protein rows satisfy the protein contract above.
- A bare catalytic protein actor is rejected at its own JSON pointer, whether it
  appears in `reactions[].enzymes` or the catalytic subset of
  `reactions[].modifiers`.
- One canonical `enzymes`/`modifiers` cross-field mirror is accepted as one
  logical catalyst and produces one IR enzyme. Duplicate actors within either
  field remain errors and are not silently collapsed.

If this contract fails, the pipeline did not converge and the pathway is not
exportable. The error is surfaced to the user with the specific failing checks.
This is the only hard abort in the semantic sense.

The 2026-07-13 regression replay uses the exact saved `tmp/final.mapped.json`.
With valid pathway metadata supplied, it returns `ok=true`, writes the PWML
output, and reports zero required-contract and IR-validation errors. The focused
merged suite passes 127 tests; the full suite passes 380 tests, with Ruff,
compile, and diff checks also green. A live UI click remains the final manual
verification step.

---

## Stage contract summary

| Boundary | Type | Effect on failure |
|---|---|---|
| Post-extraction | Structural | Abort — data is unreadable |
| Post-mapping | Structural | Abort — mapping cannot proceed downstream |
| Post-normalization (gate) | Semantic | Feed to audit loop — this is expected |
| Post-audit | Structural | Abort — audit produced garbage |
| Post-curate | Structural | Fall back to pre-curate payload — do not abort |
| Post-enrichment | Runtime shape (report-first) | Report known recursive shape errors; preserve additive metadata |
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
[2A: Infer]
    │
    ▼
[2B: Map]     ──→  stage2.mapped.json + stage2_mapping_report.json
    │                        │
    │               structural contract
    │
    ▼
[3: Normalize]               ← single canonical function, 17 steps
    │
    ├──→ gate passes ─────────────────────────────────────────────┐
    │                                                              │
    └──→ gate fails → [4: Audit] ──→ strict gate again ──→ next round  │
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
                                          validation-only gate + pre-export contract
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
