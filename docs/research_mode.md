# Research mode

Status: **implemented** on branch `research-mode`. The categorization table below is the specification; the "How it is implemented" section records what was actually built and where it deliberately differs.
Scope: what changes when the pipeline runs in **research mode** instead of the default **strict (PathWhiz) mode**.
Companion docs: [`docs/pipeline.md`](pipeline.md), [`docs/pathwhiz_requirements.md`](pathwhiz_requirements.md), [`docs/rag/03_separation_invariant.md`](rag/03_separation_invariant.md), [`docs/change_log.md`](change_log.md).

---

## Scope decisions

Four decisions were taken before implementation. They narrow the table below, so read them first.

1. **Annotations live off-payload.** Tier labels and review flags are a side-car keyed by JSON pointer; no payload row gains a key. This sidesteps the bare `assert key in _ALLOWED_ROW_KEYS` at `rag/synthesize.py:1077` and keeps strict-mode payloads byte-identical.
2. **RAG synthesis keeps validating its own output strictly.** `rag/synthesize.py`'s self-gate on `validate_post_extraction` is untouched, so research mode changes the *export policy*, never what RAG is allowed to emit.
3. **Research mode emits no PWML XML.** It produces a graph, a citation report, and JSON/Markdown/CSV exports. Consequently **every row below under `pwml/ir.py`, `pwml/writer.py`, `pwml/validate.py` and `pwml/qa.py` is out of scope and unchanged** — roughly 135 of the 277 rows. That also retires the mis-serialization risk those rows carried (a compound written into a `<protein-id>` element), because nothing is serialized.
4. **No page numbers are fabricated.** None exist in the RAG path and none are derivable for acquired papers, so citations use `title (source_id) — section` plus the verbatim passage. See [Citations](#citations-what-can-honestly-be-printed).

### Deliberate deviations from the table

- **`actor_schema_not_canonical` is FLAG, not SKIP.** The table marks it FORMAT/SKIP, but the one issue code covers both `entity_type` (a PathWhiz bucket tag — format) and `entity` (the catalyst's biological identity). Rather than split the code and change strict-mode output, research mode treats the whole check as a review flag. This is the "err toward keep-but-flag" rule applied.
- **Unrecognised issue codes flag rather than skip.** `classify_issue` returns `review` for anything not explicitly listed as FORMAT. This matters because `payload_models.py` emits its *own* copies of `species_required`, `generated_wrapper_missing_components` and `actor_schema_not_canonical` at different lines — a code→category table is not globally unique, so the default had to be the safe direction.
- **`needs_species` is not relaxed.** Relaxing it would let `best_effort_fallback` (`map_ids.py:3822`) stamp a wrong-organism accession as `status="mapped"`. Instead, tier grounding uses a **read-only** resolver that never writes into mapping. See [Evidence tiers](#evidence-tiers).

---

## Core principle

T2PW's checks fall into two kinds, and today the pipeline treats them identically — both abort the run.

**FORMAT rules** exist only so the PathWhiz Rails importer accepts the file. A missing UniProt accession, a `+` in a compound name, an enzyme not wrapped in a synthetic `protein_complex`, a canvas width, a duplicate join row. None of these say anything about whether the biology is right. A genuinely novel enzyme from a 2026 paper legitimately has no UniProt accession. **In research mode, FORMAT rules are SKIPPED.**

**BIOLOGY and PROVENANCE checks** are the ones that matter for a novel multi-paper pathway. Every reaction has real inputs and outputs. Every catalyst resolves to a declared entity. Every element traces back to a source paper. **In research mode these keep running exactly as they do today — but they only FLAG. They never abort.**

Three rules govern the whole design:

1. **Research mode is FAIL-OPEN.** A run that would abort in strict mode produces an artifact in research mode. That is the point.
2. **Fail-open must never mean fail-silent.** Every skipped FORMAT rule and every flagged BIOLOGY/PROVENANCE violation is recorded, with its issue code and JSON pointer, and surfaced prominently in the UI and in the written report. Research mode must never quietly pass junk. Where today's code *silently deletes* content to satisfy a downstream gate (there are eleven such passes in `process_normalizer.py` alone), research mode keeps the content and annotates it.
3. **A check is never deleted, only re-severitied.** A skipped FORMAT rule still runs, still emits its exact same issue code, still carries its pointer — it lands in `warnings` instead of `errors`. This is what makes the strict/research diff auditable, and it is why the existing `_add_issue` warning channel is reused rather than a parallel one being built.

When in doubt the table below errs toward **FLAG** (keep the check, downgrade it to non-blocking) rather than SKIP. Every arguable call is marked **(judgement)** in the Reason column so a human reviewer can find it.

### Behaviour vocabulary

| Value | Meaning |
|---|---|
| **SKIP** | The rule is FORMAT-only. In research mode it does not block; the finding is recorded as a warning. For a *destructive pass* (one that deletes payload content), SKIP means the deletion does not happen and the retained content is annotated. |
| **FLAG** | The check still runs and still reports every finding. It no longer aborts. This is the target state for all BIOLOGY/PROVENANCE checks, and for silent-drop paths that today discard content without reporting. |
| **UNCHANGED** | Byte-for-byte identical in both modes. Either it is already non-blocking (nothing to relax), or it is a crash guard / writer precondition where skipping produces an uncaught exception instead of a clean report. |

---

## The table

277 checks, grouped by source file. `Location` is `file:line` in the repo at `Project14-T2PW-research`. Line numbers point at the `_add_error(` / `err(` / `raise` **call site**; the message string typically sits 1–2 lines below (verified by spot check — see [Survey conflicts](#survey-conflicts)).

Totals: **85 SKIP**, **86 FLAG**, **106 UNCHANGED**.

### `src/t2pw/pipeline/stage_contracts.py` — the six stage-boundary validators (25)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| `runtime_payload_shape` (recursive pydantic shape, all boundaries) | `stage_contracts.py:55` | FORMAT | UNCHANGED | `RUNTIME_SCHEMA_MODE = "report"` already makes it warnings-only; it is the channel research mode reuses. |
| `invalid_payload` | `stage_contracts.py:427` | FORMAT | UNCHANGED | Container crash guard — every validator `assert isinstance(payload, dict)` right after. **(judgement)** |
| `entities_required` | `stage_contracts.py:430` | FORMAT | UNCHANGED | Container crash guard; every later check indexes into `entities`. **(judgement)** |
| `processes_required` | `stage_contracts.py:432` | FORMAT | UNCHANGED | Container crash guard; a RAG payload always has the key, so relaxing buys nothing. **(judgement)** |
| `entity_not_object` | `stage_contracts.py:451` | FORMAT | UNCHANGED | Row type guard; a bare-string row breaks every downstream `row.get()`. **(judgement)** |
| `entity_missing_name` | `stage_contracts.py:460` | BIOLOGY | FLAG | An unnamed entity has no biological identity and cannot be traced to a paper. |
| `process_not_object` | `stage_contracts.py:477` | FORMAT | UNCHANGED | Row type guard, and the gate that lets the five per-bucket BIOLOGY checks run at all. **(judgement)** |
| `reaction_missing_participants` | `stage_contracts.py:504` | BIOLOGY | FLAG | Verbatim "every reaction has real inputs and outputs". |
| `transport_missing_cargo` | `stage_contracts.py:519` | BIOLOGY | FLAG | A transport with nothing transported is an empty biological claim. |
| `interaction_missing_participants` | `stage_contracts.py:535` | BIOLOGY | FLAG | Endpoints are the biological content of the interaction. |
| `reaction_coupled_transport_missing_structure` | `stage_contracts.py:556` | BIOLOGY | FLAG | Names the two coupled events or the moved species; with neither it describes nothing. |
| `sub_pathway_missing_identity` | `stage_contracts.py:576` | PROVENANCE | FLAG | A free-text name satisfies it, so it is a traceability requirement, not a PathWhiz-ID one. |
| `species_required` | `stage_contracts.py:116` | BIOLOGY | FLAG | Which organism the pathway occurs in is a real claim. **(judgement — survey called it MIXED; `rag/synthesize.py:1190` documents fabricating buckets to dodge this abort.)** |
| `entity_missing_mapping_meta` | `stage_contracts.py:129` | PROVENANCE | FLAG | `mapping_meta` is the record of how/whether the entity was grounded. |
| `mapping_resolution_required` | `stage_contracts.py:141` | PROVENANCE | FLAG | The resolution object is the grounded/ungrounded verdict. |
| `mapping_resolution_status_required` | `stage_contracts.py:153` | PROVENANCE | FLAG | `status` is the grounding verdict string itself. |
| `mapping_resolution_field_invalid` | `stage_contracts.py:164` | FORMAT | SKIP | Type-only assertion on optional metadata; no content is lost if it is a non-string. |
| `actor_schema_not_canonical` | `stage_contracts.py:601` | FORMAT | SKIP | `entity_type` is the PathWhiz bucket tag. **(judgement — the same code also covers `entity`, the catalyst's biological identity; consider splitting.)** |
| Stage-3 `gate_report` relay | `stage_contracts.py:195` | PROVENANCE | UNCHANGED | Already the exact non-blocking "record it and hand it to the audit loop" behaviour research mode wants (`effect_on_failure = "feed_audit"`). |
| `generated_wrapper_missing_components` | `stage_contracts.py:231` | FORMAT | SKIP | Generated wrappers exist only because the importer refuses a bare enzyme. |
| `generated_wrapper_component_protein_unresolved` | `stage_contracts.py:250` | FORMAT | SKIP | Internal bookkeeping for a synthetic PathWhiz-only wrapper. |
| `generated_wrapper_component_missing_species` | `stage_contracts.py:259` | FORMAT | SKIP | Scoped to the generated wrapper; the organism claim is covered by `species_required`. |
| `generated_wrapper_component_missing_external_identity` | `stage_contracts.py:267` | FORMAT | SKIP | Textbook required-external-DB-identity-for-import; the single most important abort to skip. |
| `pre_export_pwml_contract_errors` (relay) | `stage_contracts.py:308` | FORMAT | SKIP | Relays `validate_required_pwml_contract`; skipping is per-code, driven by the categories below. |
| `pre_export_pwml_contract_warnings` (relay) | `stage_contracts.py:303` | FORMAT | UNCHANGED | Already non-blocking. Note it **overwrites** `report["warnings"]` wholesale — fix to `_add_issue` when touched. |

### `src/t2pw/pipeline/process_normalizer.py` — Stage-3 strict gates (23)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Locked-reaction report must be an object | `process_normalizer.py:3878` | PROVENANCE | FLAG | The locked-reaction ledger proves every source-locked reaction was exported or quarantined. |
| Locked accounting malformed | `process_normalizer.py:3885` | PROVENANCE | FLAG | A non-integer value silently hides lost source-locked reactions. |
| Unaccounted locked reactions | `process_normalizer.py:3891` | PROVENANCE | FLAG | Detects reactions extracted from a paper that vanished during normalization. |
| Protein / protein_complex type collision | `process_normalizer.py:3904` | FORMAT | SKIP | PathWhiz buckets must be disjoint for the importer to resolve a reference; the entity is identical either way. |
| Remaining `+` composite tokens | `process_normalizer.py:3911` | FORMAT | SKIP | Pure name-format; `_has_plus_token` (line 219) strips only a *trailing* charge, so `Ca2+ ion` is a false positive. |
| Forbidden complex reference (denylist) | `process_normalizer.py:3859` | BIOLOGY | FLAG | A hand-curated denylist of chemically bogus complexes — a biology claim, not an importer constraint. |
| Generated wrapper listed under `proteins` | `process_normalizer.py:3930` | FORMAT | SKIP | About which PathWhiz list the row sits in. |
| Protein missing species/organism | `process_normalizer.py:3936` | BIOLOGY | FLAG | The value is a biological assertion even though the requirement is a `species_id` column. **(judgement — survey called it MIXED.)** |
| Protein missing UniProt/DrugBank | `process_normalizer.py:3942` | FORMAT | SKIP | Required external DB identity for import; a novel RAG-synthesized protein legitimately has none. |
| Generated complex missing species | `process_normalizer.py:3954` | FORMAT | SKIP | Applies only to wrappers this pipeline synthesizes for the importer. |
| Generated complex has no components | `process_normalizer.py:3960` | FORMAT | SKIP | The wrapper only exists so the importer accepts an enzyme. |
| Wrapper component missing stoichiometry | `process_normalizer.py:3969` | FORMAT | SKIP | Filler on a synthetic single-protein wrapper; carries no extracted content. |
| Wrapper component unresolved protein | `process_normalizer.py:3977` | FORMAT | SKIP | Both wrapper and component are machine-generated — internal bookkeeping. |
| Wrapper component protein missing species | `process_normalizer.py:3983` | FORMAT | SKIP | Duplicates the per-protein check purely so the wrapper serializes. |
| Wrapper component protein missing external id | `process_normalizer.py:3988` | FORMAT | SKIP | Required external DB identity, restated at the wrapper-component level. |
| Unknown protein/modifier reference | `process_normalizer.py:4009` | PROVENANCE | FLAG | A catalyst resolving to no declared entity is ungrounded. |
| Unknown transporter reference | `process_normalizer.py:4029` | PROVENANCE | FLAG | Same grounding requirement on transport catalysts. |
| Wrapper: `validate_no_composites` | `process_normalizer.py:4037` | FORMAT | SKIP | Wraps a pure `+`-token name-format validator. |
| Wrapper: `validate_registry_references` | `process_normalizer.py:4041` | PROVENANCE | FLAG | Wraps the entity-grounding validator. |
| Wrapper: `validate_no_scaffold_modifiers` | `process_normalizer.py:4045` | BIOLOGY | FLAG | Wraps a mechanism claim: a scaffold substrate must not be listed as a catalyst. |
| Located protein isolated in connectivity graph | `process_normalizer.py:4088` | BIOLOGY | FLAG | Degree 0 means it participates in nothing — a biological-completeness signal. |
| Protein has degree 0 | `process_normalizer.py:4109` | BIOLOGY | FLAG | A novel multi-paper pathway can legitimately carry a not-yet-wired protein. |
| `GateValidationError` raise (aggregation exit) | `process_normalizer.py:4120` | BIOLOGY | FLAG | The one aggregation point for all gate errors — the seam where research mode stops raising and returns `ok=False` with a severity split. |

### `src/t2pw/pipeline/process_normalizer.py` — standalone validators (12)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Composite `+` in compound name | `process_normalizer.py:3598` | FORMAT | SKIP | A `+` separator only matters because PathWhiz cannot ingest a two-entity name. |
| Composite `+` in reaction token | `process_normalizer.py:3606` | FORMAT | SKIP | Same name-format rule on participant tokens. |
| Composite `+` in transport cargo | `process_normalizer.py:3614` | FORMAT | SKIP | Same name-format rule on cargo. |
| `validate_no_composites` raise | `process_normalizer.py:3617` | FORMAT | SKIP | Aggregates only `+` name-format findings; also called from `curation/audit_json_llm.py:236`. |
| Registry: reaction token unknown | `process_normalizer.py:3632` | PROVENANCE | FLAG | An unregistered participant is ungrounded with no traceable declaration. |
| Registry: reaction actor unknown | `process_normalizer.py:3639` | PROVENANCE | FLAG | An ungrounded catalyst cannot be traced to a source-backed entity. |
| Registry: transport cargo unknown | `process_normalizer.py:3648` | PROVENANCE | FLAG | Same grounding requirement on transported species. |
| Registry: transporter unknown | `process_normalizer.py:3654` | PROVENANCE | FLAG | Same grounding requirement on transport catalysts. |
| Registry: interaction side unknown | `process_normalizer.py:3665` | PROVENANCE | FLAG | Same grounding requirement on interaction participants. |
| `validate_registry_references` raise | `process_normalizer.py:3670` | PROVENANCE | FLAG | Aggregates entity-grounding findings; also called from `curation/audit_json_llm.py:237`. |
| Scaffold protein listed as modifier | `process_normalizer.py:3692` | BIOLOGY | FLAG | Encodes the claim that a scaffold/substrate is consumed, not catalytic. |
| `validate_no_scaffold_modifiers` raise | `process_normalizer.py:3695` | BIOLOGY | FLAG | Aggregates the scaffold-as-catalyst findings. |

### `src/t2pw/pipeline/process_normalizer.py` — composite materialization + actor contract (4)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Composite token has no protein-like left part | `process_normalizer.py:942` | FORMAT | SKIP | Representational limit of the PathWhiz protein_complex model; **uncaught**, fires in pass 3 of 18 before any report exists. |
| Composite complex name has non-protein left token | `process_normalizer.py:1139` | FORMAT | SKIP | Same limit; **uncaught** `ValueError` that kills the whole run. |
| Composite compound cannot materialize | `process_normalizer.py:1184` | FORMAT | SKIP | Same limit on the compound registry; **uncaught**. |
| `assert actor_contract.get("ok") is True` | `process_normalizer.py:4394` | FORMAT | SKIP | Pre-export required-field contract on internal actor rows. This is the hidden abort behind `actor_schema_not_canonical`; `assert` also vanishes under `python -O`. |

### `src/t2pw/pipeline/process_normalizer.py` — destructive passes that silently delete content (11)

These do not raise, so a "make gates non-blocking" change misses them entirely — yet they delete exactly the content research mode exists to preserve.

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Drop pathway-metadata blob rows | `process_normalizer.py:285` | FORMAT | UNCHANGED | Narrow guard against a RAG serialization defect; already a sanctioned exception. **(judgement — the >12-word branch can eat a legitimately long novel entity name.)** |
| Remove biochemical-colon complex, re-add as compound | `process_normalizer.py:1359` | FORMAT | UNCHANGED | Reclassifies rather than deletes; a `:` in a lipid name must not read as complex syntax. |
| Remove forbidden / byproduct complex | `process_normalizer.py:1385` | BIOLOGY | SKIP | The byproduct denylist is a chemistry judgement; the "ends in acid" suffix rule dissolves legitimate acid-conjugated complexes. **(judgement)** |
| Delete component-only protein | `process_normalizer.py:1744` | FORMAT | SKIP | Motivation is import identity, but the effect deletes a real extracted subunit. |
| Drop unresolved complex components | `process_normalizer.py:1764` | BIOLOGY | SKIP | Silently changes the declared composition of a complex. **(judgement — import-motivated, biological effect.)** |
| Drop process-orphan proteins | `process_normalizer.py:1854` | BIOLOGY | SKIP | Deletes an entity genuinely extracted from a paper purely to pre-empt the degree-0 gate. |
| Drop non-protein catalyst | `process_normalizer.py:2386` | BIOLOGY | SKIP | Deletes real catalytic cofactors/metal ions asserted by a paper. **(judgement — the *rule* is FORMAT, the *deleted content* is biology.)** |
| Drop unknown actor (`drop_unknown=True`) | `process_normalizer.py:2394` | PROVENANCE | SKIP | Discards an extracted enzyme claim whose name did not normalize — the provenance link is destroyed rather than reported. |
| Drop / quarantine no-op reaction | `process_normalizer.py:3452` | BIOLOGY | SKIP | A coarse-graining judgement a novel pathway (translocation, conformational change) can legitimately violate. **(judgement)** |
| Quarantine duplicate locked reaction | `process_normalizer.py:3491` | PROVENANCE | FLAG | Two source-locked reactions from possibly different papers collapse to one. **(judgement — quarantine is already recorded, so FLAG rather than SKIP.)** |
| Prune disconnected proteins | `process_normalizer.py:3789` | BIOLOGY | SKIP | Deletes extracted proteins solely to avoid the degree-0 gate — the entities research mode most wants surfaced. |

### `src/t2pw/pwml/ir.py` — `build_pwml_ir` (24)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| `component_inferred_from_biological_state` | `ir.py:170` | PROVENANCE | UNCHANGED | Already a warning; records that context was synthesized, not extracted. |
| `duplicate_named_record` | `ir.py:344` | FORMAT | UNCHANGED | The collision key is a name-normalization artifact; already non-blocking. **(judgement — the effect is silently deleting a declared entity.)** |
| `compound_db_resolution_failed` | `ir.py:885` | FORMAT | SKIP | Pure external-DB identity; `writer.py:109` already exempts exactly this code — the precedent to widen. |
| `noncanonical_names_collision_risk` | `ir.py:955` | FORMAT | UNCHANGED | About name collisions on PathWhiz import; already a warning. |
| `invalid_payload` | `ir.py:980` | FORMAT | UNCHANGED | Input container type guard. |
| `missing_db_identity` | `ir.py:1126` | FORMAT | SKIP | Required external DB identity. **Dead code today** — verified all five `entity_specs` pass `strict_required=False` (`ir.py:1055-1093`), so `ir.py:1118` never fires. |
| `protein_complex_missing_components` (raw) | `ir.py:1183` | FORMAT | UNCHANGED | PathWhiz membership rule; already a warning. |
| `component_stoichiometry_unstated` | `ir.py:1225` | BIOLOGY | UNCHANGED | Subunit count is a quantity the paper may not state; already the desired shape. |
| `component_protein_unresolved` | `ir.py:1236` | FORMAT | SKIP | Wrapper component reference rule. **(judgement — it DROPS the component, so SKIP means keep it.)** |
| `protein_complex_missing_components` (resolved) | `ir.py:1255` | FORMAT | UNCHANGED | Same membership rule post-resolution; already a warning. |
| `unresolved_biological_state_component` | `ir.py:1275` | BIOLOGY | FLAG | An organism/compartment asserted but never declared is an ungrounded context claim. |
| `generated_default_state` | `ir.py:1291` | PROVENANCE | UNCHANGED | T2PW invents context traceable to no paper — must stay loudly visible. Already a warning. |
| `default_state_without_context` | `ir.py:1298` | PROVENANCE | UNCHANGED | Same fabrication, worse case: nothing ties the pathway to an organism. |
| `biological_state_missing_species` | `ir.py:1340` | BIOLOGY | FLAG | Which organism the pathway happens in is a core biological claim. Hard error today. |
| `biological_state_missing_subcellular_location` | `ir.py:1349` | BIOLOGY | FLAG | Compartment is a biological claim; note PathWhiz itself needs only ONE of the four (see `ir.py:2415-2419`) — this strictness is T2PW-invented. |
| `biological_state_without_components` | `ir.py:1358` | BIOLOGY | UNCHANGED | Completely ungrounded context; already a warning. |
| `unresolved_biological_state` | `ir.py:1383` | BIOLOGY | FLAG | A reaction pointing at an undeclared context is a biology/provenance gap. |
| `unresolved_entity_reference` | `ir.py:1407` | BIOLOGY | FLAG | A participant resolving to nothing means the reaction loses a real input/output. `strict_db` does not relax it. |
| `ambiguous_entity_reference` | `ir.py:1424` | BIOLOGY | UNCHANGED | The wrong entity type silently becoming a reactant; already a warning. |
| `reaction_enzyme_must_be_protein_complex` | `ir.py:1441` | FORMAT | SKIP | The wrapper exists only because the importer models catalysts that way. |
| `location_entity_not_found` | `ir.py:1548` | FORMAT | UNCHANGED | Coordinate/layout reference bookkeeping; already a warning. |
| `enzyme_without_resolvable_actor` | `ir.py:1689` | PROVENANCE | UNCHANGED | Evidence exists but nothing is grounded to it — a provenance dangler. Already a warning. |
| `non_protein_catalyst_dropped` | `ir.py:1702` | BIOLOGY | SKIP | A real regulatory actor stated in the paper is silently discarded. **(judgement — stop dropping, annotate instead.)** |
| `validate_pwml_ir` mirror into build report | `ir.py:2000` | FORMAT | FLAG | Pure relay; its category is whatever `validate_pwml_ir` emitted. **(judgement — relay severity follows the per-code table.)** |

### `src/t2pw/pwml/ir.py` — `validate_required_pwml_contract` (42)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| `invalid_input` | `ir.py:2041` | FORMAT | UNCHANGED | Container type guard. |
| `pathway_missing_pw_id` | `ir.py:2133` | FORMAT | UNCHANGED | External PathBank identity; already a warning. |
| `pathway_missing_name` | `ir.py:2135` | FORMAT | SKIP | Pre-export PWML required-field contract. |
| `pathway_missing_subject` | `ir.py:2137` | FORMAT | SKIP | `subject` is a PathWhiz enum. |
| `pathway_missing_width` | `ir.py:2140` | FORMAT | SKIP | Canvas/layout geometry. |
| `pathway_missing_height` | `ir.py:2142` | FORMAT | SKIP | Canvas/layout geometry. |
| `compound_missing_name` | `ir.py:2200` | BIOLOGY | FLAG | A nameless compound has no chemical identity. **(judgement — survey called it FORMAT because it is a required `<compound>` field.)** |
| `protein_missing_name` | `ir.py:2211` | BIOLOGY | FLAG | Same reasoning as above. **(judgement)** |
| `protein_missing_species` | `ir.py:2218` | BIOLOGY | FLAG | Which organism a protein comes from is a biological claim, not an import formality. |
| `protein_missing_external_identity` | `ir.py:2225` | FORMAT | SKIP | Canonical required-external-DB-identity-for-import; blocks every genuinely novel protein. |
| `protein_complex_missing_species` | `ir.py:2240` | BIOLOGY | FLAG | Organism attribution of the complex is biological. |
| `protein_complex_missing_components` | `ir.py:2249` | FORMAT | SKIP | Generated-wrapper component rule; the wrapper exists solely for PathWhiz's enzyme model. |
| `component_stoichiometry_unstated` | `ir.py:2267` | BIOLOGY | UNCHANGED | Papers routinely omit subunit counts; already correctly non-blocking. |
| `component_protein_unresolved` (key path) | `ir.py:2278` | FORMAT | SKIP | Wrapper component reference rule. |
| `component_protein_unresolved` (name path) | `ir.py:2305` | FORMAT | SKIP | Same rule; lookup is `_norm`-based so name punctuation matters. |
| `component_protein_unresolved` (empty record) | `ir.py:2313` | FORMAT | SKIP | Same rule, empty-record path. |
| `generated_complex_component_missing_species` | `ir.py:2321` | FORMAT | SKIP | Trigger is a generated wrapper. **(judgement — the *content* is organism attribution, i.e. biology.)** |
| `generated_complex_component_missing_external_identity` | `ir.py:2329` | FORMAT | SKIP | Required external DB identity on wrapper components. |
| `species_missing_taxonomy` | `ir.py:2361` | FORMAT | SKIP | The message itself frames it as a Rails row-creation requirement; already `strict_db`-gated. |
| `species_missing_classification` | `ir.py:2369` | FORMAT | SKIP | A PathWhiz/Rails enum required at row creation; `strict_db`-gated. |
| `no_biological_states` | `ir.py:2380` | BIOLOGY | FLAG | Zero declared biological context is a biology gap. |
| `biological_state_missing_components` | `ir.py:2420` | BIOLOGY | FLAG | Fully ungrounded biological context; mirrors PathWhiz `has_at_least_one_component`. |
| `biological_state_missing_species` | `ir.py:2428` | BIOLOGY | UNCHANGED | Already a warning here (contrast the hard errors at `ir.py:1340` and `ir.py:2837`). |
| `biological_state_missing_subcellular_location` | `ir.py:2435` | BIOLOGY | UNCHANGED | Already the desired flag-not-abort shape. |
| `reaction_missing_left_participants` | `ir.py:2460` | BIOLOGY | FLAG | Literal instance of "every reaction has real inputs". |
| `reaction_missing_right_participants` | `ir.py:2467` | BIOLOGY | FLAG | Same core invariant, outputs side. |
| `reaction_participant_missing_stoichiometry` (string) | `ir.py:2477` | BIOLOGY | UNCHANGED | A biological quantity being silently assumed; already a warning. |
| `reaction_participant_missing_stoichiometry` (dict) | `ir.py:2488` | BIOLOGY | UNCHANGED | Same assumed-quantity concern. |
| `enzyme_reference_not_found` (IR entity_key) | `ir.py:2532` | PROVENANCE | FLAG | A catalyst with no backing entity record is an ungrounded actor. |
| `reaction_enzyme_must_be_protein_complex` (IR) | `ir.py:2540` | FORMAT | SKIP | The wrapper rule exists only for the importer. |
| `reaction_enzyme_must_be_protein_complex` (payload) | `ir.py:2584` | FORMAT | SKIP | Same rule pre-IR — this is the check that fires for RAG-synthesized novel enzymes. |
| `enzyme_reference_not_found` (complex name) | `ir.py:2594` | PROVENANCE | FLAG | Ungrounded catalyst; the match is a pure `_norm(name)` compare, so punctuation drives false negatives. |
| `enzyme_reference_not_found` (protein name) | `ir.py:2601` | PROVENANCE | FLAG | Ungrounded catalyst, protein-typed path. |
| `enzyme_reference_not_found` (untyped) | `ir.py:2609` | PROVENANCE | FLAG | Ungrounded catalyst, untyped path. |
| `duplicate_reaction_enzyme_complex` | `ir.py:2621` | FORMAT | SKIP | Duplicate-enzyme-complex detection; PathWhiz just rejects the repeated join row. |
| `location_entity_not_found` (payload) | `ir.py:2648` | FORMAT | UNCHANGED | Layout bookkeeping; already a warning. **Mutates the caller's payload at `ir.py:2664` despite the docstring** — `streamlit_app.py:2662` deepcopies first; keep that. |
| `visible_entity_missing_location_state` | `ir.py:2657` | FORMAT | SKIP | Coordinate/layout placement requirement of the PWML canvas model. |
| `location_missing_entity` (IR) | `ir.py:2687` | FORMAT | UNCHANGED | Layout referential integrity; writer precondition. |
| `location_missing_biological_state` (IR) | `ir.py:2692` | FORMAT | UNCHANGED | Layout referential integrity; writer precondition. |
| `visualization_missing_process` | `ir.py:2703` | FORMAT | UNCHANGED | Visualization referential integrity; writer precondition. |
| `visualization_missing_biological_state` | `ir.py:2709` | FORMAT | UNCHANGED | Visualization referential integrity; writer precondition. |
| `visualization_member_missing_location` | `ir.py:2718` | FORMAT | UNCHANGED | Visualization referential integrity; writer precondition. |

### `src/t2pw/pwml/ir.py` — `validate_pwml_ir` (43)

Referential/key checks here are **preconditions for `DeterministicPwmlBuilder`**, not cosmetics. Bypassing them makes the writer crash instead of emitting — which is why most are UNCHANGED.

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| `invalid_ir_shape` | `ir.py:2760` | FORMAT | UNCHANGED | IR container shape guard. |
| `missing_entity_key` | `ir.py:2771` | FORMAT | UNCHANGED | Internal IR key contract the writer depends on. |
| `duplicate_entity_key` | `ir.py:2774` | FORMAT | UNCHANGED | Writer's id maps would collide. |
| `protein_complex_missing_components` | `ir.py:2792` | FORMAT | SKIP | Generated-wrapper membership rule. **(judgement — `writer.py:817` raises on the same condition and must be relaxed together.)** |
| `protein_complex_component_not_structured` | `ir.py:2801` | FORMAT | UNCHANGED | Record shape `writer._protein_complex_members` expects. |
| `component_protein_unresolved` | `ir.py:2809` | FORMAT | UNCHANGED | Already a warning — but `writer.py:804` **raises** on the same condition, a latent crash that exists today. |
| `component_stoichiometry_unusable` | `ir.py:2818` | BIOLOGY | UNCHANGED | A stated subunit count is being discarded; already a warning. |
| `missing_biological_state_key` | `ir.py:2835` | FORMAT | UNCHANGED | Internal IR key contract. |
| `biological_state_missing_species` | `ir.py:2837` | BIOLOGY | FLAG | Organism attribution; one of the most common novel-pathway blockers. |
| `biological_state_missing_subcellular_location` | `ir.py:2843` | BIOLOGY | FLAG | Compartment claim; stricter than PathWhiz itself. |
| `unresolved_location_entity` | `ir.py:2865` | FORMAT | UNCHANGED | Layout referential integrity inside the machine-built IR. |
| `unresolved_location_biological_state` | `ir.py:2867` | FORMAT | UNCHANGED | Layout referential integrity. |
| `protein_complex_visualization_unknown_entity` | `ir.py:2881` | FORMAT | UNCHANGED | Visualization referential integrity. |
| `protein_complex_visualization_unknown_biological_state` | `ir.py:2888` | FORMAT | UNCHANGED | Visualization referential integrity. |
| `protein_complex_visualization_missing_location_field` | `ir.py:2896` | FORMAT | UNCHANGED | Pure coordinate/layout required-field contract; writer reads these fields. |
| `missing_process_member_key` | `ir.py:2916` | FORMAT | UNCHANGED | Internal IR key contract. |
| `duplicate_process_member_key` | `ir.py:2918` | FORMAT | UNCHANGED | Visualization members index by this key. |
| `invalid_member_entity_type` | `ir.py:2923` | FORMAT | UNCHANGED | PWML element-type enum; writer switches on it. |
| `unresolved_member_entity` | `ir.py:2925` | FORMAT | UNCHANGED | Key-level integrity in the machine-built IR (a code bug, not a biology gap); the biology equivalent is `ir.py:1407`. |
| `unresolved_reaction_biological_state` | `ir.py:2939` | BIOLOGY | FLAG | Where the reaction happens is biological context. |
| `reaction_missing_left` | `ir.py:2943` | BIOLOGY | FLAG | "Every reaction has real inputs" — keep checking, stop aborting. |
| `reaction_missing_right` | `ir.py:2945` | BIOLOGY | FLAG | "Every reaction has real outputs". |
| `reaction_enzyme_must_be_protein_complex` | `ir.py:2956` | FORMAT | SKIP | Third and last copy of the importer wrapper rule. |
| `reaction_enzyme_missing_complex_visualization` | `ir.py:2967` | FORMAT | SKIP | Canvas visibility/layout rule; moot once wrapping is off. |
| `reaction_enzyme_hidden_complex_visualization` | `ir.py:2975` | FORMAT | SKIP | Canvas visibility/layout rule. |
| `transport_element_count` (1–3) | `ir.py:2992` | BIOLOGY | FLAG | Lower bound is biology (a transport must move something). **(judgement — the upper bound of 3 is a PathWhiz schema cap and is pure FORMAT; consider splitting.)** |
| `transport_missing_left_state` | `ir.py:3002` | BIOLOGY | FLAG | The source compartment is the biological content of the transport. |
| `transport_missing_right_state` | `ir.py:3004` | BIOLOGY | FLAG | The destination compartment is the biological content. |
| `rct_missing_left` | `ir.py:3018` | BIOLOGY | FLAG | Real-inputs invariant. **Gotcha: `build_pwml_ir` always emits RCTs with empty left/right/enzymes (`ir.py:1937-1945`), so this always fires.** |
| `rct_missing_right` | `ir.py:3020` | BIOLOGY | FLAG | Real-outputs invariant; same always-fires gotcha. |
| `rct_missing_enzyme` | `ir.py:3022` | BIOLOGY | FLAG | "This transport is catalyzed" is a biological claim. **(judgement — PathWhiz's RCT model also structurally requires it; same always-fires gotcha.)** |
| `interaction_missing_type` | `ir.py:3032` | FORMAT | SKIP | Required PWML field; `build_pwml_ir` already defaults it to `"interaction"` (`ir.py:1899`). |
| `interaction_missing_side` | `ir.py:3036` | BIOLOGY | FLAG | An interaction with a missing partner is an incomplete biological assertion. |
| `unresolved_interaction_entity` | `ir.py:3042` | BIOLOGY | FLAG | An interaction partner that is not a declared entity is an ungrounded claim. |
| `sub_pathway_missing_type` | `ir.py:3055` | FORMAT | SKIP | Required PWML field on the sub-pathway record. |
| `sub_pathway_missing_reference` | `ir.py:3057` | FORMAT | SKIP | Required-field alternation for the importer; `reference_pathway_id` is an external PathBank identity. |
| `visualization_unknown_process` | `ir.py:3077` | FORMAT | UNCHANGED | Visualization referential integrity; writer precondition. |
| `visualization_type_process_mismatch` | `ir.py:3079` | FORMAT | UNCHANGED | Visualization typing only; already a warning. |
| `visualization_unknown_biological_state` | `ir.py:3087` | FORMAT | UNCHANGED | Visualization referential integrity. |
| `visualization_unknown_member` | `ir.py:3094` | FORMAT | UNCHANGED | Visualization referential integrity. |
| `visualization_unknown_location` | `ir.py:3097` | FORMAT | UNCHANGED | Coordinate/layout referential integrity. |
| `visualization_unknown_edge` | `ir.py:3100` | FORMAT | UNCHANGED | Edge/layout referential integrity. |
| `visualization_member_missing_edge` | `ir.py:3102` | FORMAT | UNCHANGED | Pure drawing/edge-geometry requirement; writer reads it. |

### `src/t2pw/pwml/writer.py` (10)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Reaction-member anchor assert | `writer.py:134` | FORMAT | UNCHANGED | Coordinate/layout self-check; a bare `AssertionError` with no message mid-serialization. **(judgement — arguably a skippable layout rule.)** |
| Unsupported rectangle anchor mode | `writer.py:156` | FORMAT | UNCHANGED | Internal layout-geometry argument guard. |
| `DeterministicPwmlBuilder` requires validated IR | `writer.py:694` | FORMAT | UNCHANGED | IR container shape guard at the serializer boundary. |
| Complex component does not reference an existing protein | `writer.py:804` | FORMAT | SKIP | Wrapper component rule enforced as a hard exception. **(judgement — must become "skip the record", not "raise"; `ir.py:2809` only warns.)** |
| Complex has no `protein_complex-proteins` to export | `writer.py:817` | FORMAT | SKIP | Membership requirement for complexes lacking their own DB identity. **(judgement — must pair with `ir.py:2792`.)** |
| Re-run `validate_pwml_ir` before serialization | `writer.py:835` | FORMAT | SKIP | **The hard backstop.** Makes every IR error fatal even if callers were relaxed; must honour the mode or all relaxations are undone. |
| Negative reaction-member y coordinate | `writer.py:2107` | FORMAT | UNCHANGED | Off-canvas coordinate diagnostic; already a `warnings.warn`. |
| Input JSON must be an object | `writer.py:2649` | FORMAT | UNCHANGED | CLI input container guard. |
| Post-normalization gate blocks export | `writer.py:2655` | PROVENANCE | FLAG | Delegated relay; its contents come from `process_normalizer.py` and mix reaction-completeness with shape rules. |
| IR validation blocks export | `writer.py:2685` | FORMAT | SKIP | CLI twin of `streamlit_app.py:2727`; the primary place to insert the research-mode bypass. |

### `src/t2pw/pwml/validate.py` — reference-XML shape (8)

All non-raising; `repair_tree` (`validate.py:252/269`) already auto-fixes most. They still contribute to `report["ok"]`, which gates the export result.

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Missing pathway-visualization node in reference | `validate.py:139` | FORMAT | UNCHANGED | Reference-file structural discovery; no biology involved. |
| Element order mismatch | `validate.py:219` | FORMAT | UNCHANGED | XML child ordering; `repair_tree` auto-fixes it. |
| Root tag mismatch | `validate.py:358` | FORMAT | UNCHANGED | XML shape only. |
| Missing pathway-visualization subtree | `validate.py:371` | FORMAT | UNCHANGED | XML shape only. |
| Missing `<pathway>` node | `validate.py:385` | FORMAT | UNCHANGED | XML shape only. |
| Missing section container | `validate.py:403` | FORMAT | UNCHANGED | XML shape only; `repair_tree` creates empty sections. |
| Missing required field on an item | `validate.py:419` | FORMAT | SKIP | The literal reference-derived required-field contract of the emitted XML. **(judgement — it flips export `ok`.)** |
| Missing `type="integer"` attribute | `validate.py:427` | FORMAT | SKIP | XML attribute typing for the Rails parser. **(judgement — flips export `ok`.)** |

### `src/t2pw/pwml/qa.py` — post-serialization QA (8)

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| XML parse error | `qa.py:16` | FORMAT | UNCHANGED | Serialization well-formedness; if this fails we produced a broken file. |
| Reaction marked spontaneous but has an enzyme | `qa.py:57` | BIOLOGY | FLAG | A spontaneous-yet-catalyzed reaction is a biological contradiction. **Note: unreachable from the IR path — `spontaneous` is hardcoded `False` at `ir.py:1603`.** |
| Same protein-complex twice as an enzyme | `qa.py:65` | FORMAT | SKIP | Duplicate-enzyme-complex detection; PathWhiz rejects the repeated join row. |
| Transport with identical left/right state | `qa.py:82` | BIOLOGY | UNCHANGED | A transport that crosses no boundary is biologically meaningless; already a warning. |
| Invalid `<stoichiometry>` string | `qa.py:99` | FORMAT | SKIP | Validates the serialized string form (nil is explicitly allowed at `qa.py:91`), not the quantity. |
| HMDB id not zero-padded | `qa.py:108` | FORMAT | UNCHANGED | External identifier string format; already a warning. |
| ChEBI id missing `CHEBI:` prefix | `qa.py:116` | FORMAT | UNCHANGED | Identifier string format — and a colon that must be **kept**; no name-leniency change may strip it. |
| Orphaned compound-location | `qa.py:124` | FORMAT | SKIP | Layout referential integrity in the emitted XML. **(judgement — could produce an XML the importer chokes on; the annotation must be loud.)** |

### `src/t2pw/mapping/map_ids.py` (32)

Only four `raise` statements exist in this entire file (592, 598, 5964, 6656). The mapping stage is already almost entirely warn-shaped; the blocking behaviour lives downstream.

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Drop enzyme protein with no external id | `map_ids.py:4856` | FORMAT | SKIP | Only PathWhiz's accession requirement forces the delete; the enzyme is real biology from the paper. |
| Skip wrapper for invalid component | `map_ids.py:4872` | FORMAT | SKIP | Precondition of wrapper generation itself; disappears when wrapping is off. |
| Cached wrapper reuse requires identity | `map_ids.py:4746` | FORMAT | SKIP | Exists to pre-satisfy the `generated_wrapper_component_*` import gates. |
| Merge duplicate actor rows after wrapping | `map_ids.py:4971` | FORMAT | SKIP | Exists solely to avoid `duplicate_reaction_enzyme_complex`; the collision is created by wrapping, not by the paper. |
| Wrapper complex missing species | `map_ids.py:2440` | BIOLOGY | FLAG | A missing organism is genuine biological under-specification. **(judgement — `species_id` is also an importer-required column.)** |
| Wrapper: enzyme protein has no name | `map_ids.py:2361` | FORMAT | SKIP | Lives entirely inside the wrapper generator; a payload-shape defect. |
| Wrapper component protein unresolved | `map_ids.py:2432` | PROVENANCE | UNCHANGED | The record that a paper-named enzyme is ungrounded — already non-blocking, must stay loud. |
| Offline wrapper missing species (`db_unavailable`) | `map_ids.py:4768` | PROVENANCE | SKIP | The synthesized wrapper is import scaffolding. **(judgement — the `db_unavailable` provenance gap must still be surfaced separately.)** |
| Prune complex components without an id | `map_ids.py:6219` | FORMAT | SKIP | Comment states the motive verbatim: "so the gate does not reject the whole payload for their sake". |
| Remove now-unreferenced proteins | `map_ids.py:6234` | FORMAT | SKIP | Same import-identity motive; these proteins must stay with an ungrounded annotation. |
| Compound with `:` routed to complex, maps nothing | `map_ids.py:6472` | FORMAT | SKIP | The canonical name-format rule: a `:` in a legitimate acyl-CoA name costs the entity its ID mapping. |
| Gene-symbol token rejects `:` | `map_ids.py:91` | FORMAT | SKIP | A punctuation-shape rule that downgrades otherwise-correct UniProt matches. |
| Protein has no name | `map_ids.py:6025` | FORMAT | UNCHANGED | Required-field shape check; already non-blocking. |
| Compound has no name | `map_ids.py:6431` | FORMAT | UNCHANGED | Required-field shape check; already non-blocking. |
| `needs_species` short-circuit before any API call | `map_ids.py:4254` | BIOLOGY | FLAG | Organism scoping is a grounding requirement — but this single line prevents *any* organism-free UniProt retry. Relax the short-circuit, keep the flag. |
| DB protein `needs_species` | `map_ids.py:1837` | BIOLOGY | UNCHANGED | Species is intrinsic to identifying which protein the paper means; already non-blocking. |
| DB protein `species_not_found` | `map_ids.py:1852` | BIOLOGY | UNCHANGED | An unrecognised organism is a real grounding failure; already non-blocking. |
| DB complex `needs_species` | `map_ids.py:2208` | BIOLOGY | UNCHANGED | Same organism-grounding requirement for complexes. |
| DB complex `species_not_found` | `map_ids.py:2226` | BIOLOGY | UNCHANGED | Organism grounding failure; already non-blocking. |
| Complex component has no name | `map_ids.py:2131` | FORMAT | UNCHANGED | Payload-shape defect on the component row. |
| Complex component protein unresolved | `map_ids.py:2177` | PROVENANCE | UNCHANGED | The record that a named subunit is ungrounded — the corroboration signal to keep. |
| Complex arrived with no components | `map_ids.py:2339` | BIOLOGY | UNCHANGED | A complex with zero known subunits is a genuine biological gap. **(judgement — also a PathWhiz import requirement.)** |
| Mapped complex has no PathBank components | `map_ids.py:963` | FORMAT | UNCHANGED | The complex is already grounded by a real id; only the exporter's component list is deficient. |
| UniProt `no_match` / `network_error` | `map_ids.py:3804` | PROVENANCE | FLAG | The terminal "entity is not grounded" verdict; `network_error` deserves a retry and must be distinguished from a true `no_match`. |
| UniProt acceptance threshold | `map_ids.py:3508` | BIOLOGY | UNCHANGED | A confidence judgement about whether the accession really is the paper's protein; report the score. |
| UniProt `best_effort_fallback` | `map_ids.py:3823` | PROVENANCE | FLAG | Silently promotes a weak guess to `status="mapped"`; the `best_effort` flag is the only trace. |
| Stage-6 Phase 2 `except Exception: pass` | `map_ids.py:6280` | PROVENANCE | FLAG | Grounding failures are invisible today. This is where retry-with-synonyms belongs and where a flag must replace `pass`. |
| HTTP retries exhausted | `map_ids.py:598` | PROVENANCE | FLAG | An exhausted retry means the entity could not be corroborated; every call site already catches it. |
| Payload must be an object | `map_ids.py:5964` | FORMAT | UNCHANGED | Pure input-shape contract; nothing downstream can run without a dict. |
| Unknown-fallback refused: unusable name | `map_ids.py:5293` | FORMAT | SKIP | A name-string pattern rule guarding an import-only sentinel substitution. |
| PathBank "Unknown" sentinel substitution | `map_ids.py:5463` | PROVENANCE | SKIP | Fabricates an identity **and an organism** (Arabidopsis thaliana, id 4) the paper never stated. Must never fire in research mode. |
| `mapping_result_missing` stamp | `map_ids.py:5936` | PROVENANCE | UNCHANGED | The audit trail proving an entity was never grounded — promote to a visible annotation, never drop. |

### `src/t2pw/rag/` — provenance and synthesis (12)

Nothing here aborts today except the last two rows.

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| Reaction has a resolvable source | `provenance.py:201` | PROVENANCE | UNCHANGED | Literally the "traces back to a source paper" guarantee; already report-only. |
| Non-cofactor entity has a resolvable source | `provenance.py:223` | PROVENANCE | UNCHANGED | Grounding/corroboration of entities; already report-only. |
| Seed reaction with no evidence is **omitted** | `synthesize.py:750` | PROVENANCE | FLAG | Today it silently deletes the element. Research mode keeps it and annotates. |
| Entity with no evidence is **omitted** | `synthesize.py:1021` | PROVENANCE | FLAG | Same silent deletion; keep and annotate. |
| Gap left unfilled | `synthesize.py:1330` | PROVENANCE | UNCHANGED | Already recorded in `unresolved_gaps`. |
| Reaction must have BOTH sides (parsed line) | `synthesize.py:471` | BIOLOGY | FLAG | Returns `None`, silently dropping the reaction; keep and flag. |
| Reaction must have BOTH sides (extracted) | `synthesize.py:528` | BIOLOGY | FLAG | Same silent drop on the extraction path. |
| Junk species-token rejection | `synthesize.py:136` | FORMAT | UNCHANGED | Name-shape sanity filter over metadata blobs; not a biology claim. |
| Bibliography chunk refused | `synthesize.py:601` | PROVENANCE | UNCHANGED | Stops cited titles being mined as fake reactions — an anti-fabrication guard. **Keep even in research mode.** |
| Non-parseable source types refused | `synthesize.py:643` | PROVENANCE | UNCHANGED | Prevents inventing chemistry from a metadata bag. |
| `validate_post_extraction` self-gate | `synthesize.py:1255` | FORMAT | SKIP | Core structural payload contract; inherits whatever mode the validator runs in. **Relaxing it changes what RAG is allowed to emit** — see Open questions. |
| `assert set(row) <= _ALLOWED_ROW_KEYS` | `synthesize.py:1077` | FORMAT | UNCHANGED | Internal key-hygiene invariant protecting the separation rule. **Fires with no message if research mode adds any on-payload annotation key** — widen `_ALLOWED_ROW_KEYS` (`synthesize.py:1080-1083`) or annotate off-payload. |

### `src/t2pw/rag/` — acquisition, ingestion, retrieval, store (23)

Every row below is currently **invisible**: a silent `continue`, a swallowed exception, or a counter with no message. Research mode's job here is entirely to make them loud.

| Check | Location | Category | Research-mode behaviour | Reason |
|---|---|---|---|---|
| chromadb not installed | `store.py:283` | FORMAT | UNCHANGED | Missing optional dependency; degrade to the `memory` backend, do not annotate as a pathway problem. |
| `ChromaVectorStore` requires `embed_fn` | `store.py:313` | FORMAT | UNCHANGED | Infrastructure wiring error. |
| `faiss` backend not implemented | `store.py:429` | FORMAT | UNCHANGED | Infrastructure. |
| Unknown `RAG_VECTOR_BACKEND` | `store.py:432` | FORMAT | UNCHANGED | Infrastructure (config typo). |
| EuropePMC candidate with no primary id | `acquire.py:358` | PROVENANCE | FLAG | A paper with no id can never be cited back. |
| PubMed article with no PMCID/PMID/DOI | `acquire.py:419` | PROVENANCE | FLAG | Same — uncitable, silently dropped. |
| Crossref item with no DOI | `acquire.py:496` | PROVENANCE | FLAG | Same. |
| Semantic Scholar item with no id | `acquire.py:551` | PROVENANCE | FLAG | Same. |
| bioRxiv item with no DOI | `acquire.py:598` | PROVENANCE | FLAG | Same. |
| `_dedupe` drops a candidate | `acquire.py:661` | PROVENANCE | FLAG | Corroboration count is silently reduced — directly affects the evidence tier. |
| Empty query short-circuit | `acquire.py:727` | PROVENANCE | UNCHANGED | Already surfaced (`status["empty_query"]`, rendered at `streamlit_app.py:316-321`). |
| No identifier for full-text fetch | `acquire.py:875` | PROVENANCE | FLAG | Paper contributes abstract-only evidence with no signal. |
| Full-text fetch exception swallowed | `acquire.py:890` | PROVENANCE | FLAG | Same — the body was unavailable and nothing says so. |
| Whole paper dropped for empty `source_id` | `ingest.py:279` | PROVENANCE | FLAG | An entire paper vanishes silently. |
| `references` / `acknowledgments` spans dropped | `ingest.py:308` | PROVENANCE | UNCHANGED | Deliberate anti-fabrication guard (stops cited titles becoming fake reactions) — **keep**. |
| Non-dict / empty DB record skipped | `ingest.py:386` | PROVENANCE | UNCHANGED | Malformed DB record; low value to surface. |
| Reference-corpus file parse error | `ingest.py:439` | FORMAT | UNCHANGED | A malformed corpus file is a file-format problem. |
| Empty corpus text | `ingest.py:490` | FORMAT | UNCHANGED | Same. |
| Memory-store upsert skipped a chunk | `store.py:180` | PROVENANCE | FLAG | `chunk_id` is a provenance field; an id-less chunk is an uncitable passage. |
| Chroma upsert skipped a chunk | `store.py:321` | PROVENANCE | FLAG | Same. |
| Embedding / persisted-index failure swallowed | `store.py:169` | FORMAT | UNCHANGED | Infrastructure; must never break startup. |
| Provenance-line fallback in hit renderer | `retrieve.py:704` | PROVENANCE | UNCHANGED | Guarantees the provenance line is emitted; correct behaviour, keep. |
| Retrieval context truncated at `max_chars` | `retrieve.py:757` | PROVENANCE | FLAG | The provenance line is appended **last** per block (`retrieve.py:717`), so citations are the first thing lost to truncation. |

---

## Evidence tiers

Research mode replaces "does this entity have a PathBank id?" with a four-tier confidence label attached to every entity and reaction. **Corroboration confirms that the ENTITY EXISTS and is stated in the literature. It does NOT invent, infer, or assign a database identifier.** A Tier B entity is still exported with no `pathbank_*` id, no UniProt accession, and no `pathwhiz_id`. Tiering is a claim about *evidence*, never about *identity*.

| Tier | Definition | Label rendered |
|---|---|---|
| **A** | A DB-grounded identifier was found: a PathBank/PathWhiz row, a UniProt accession, a DrugBank id, or a confident (≥0.85) PathWhiz compound resolution. | `DB-grounded` (+ the id) |
| **B** | No identifier — but the element is stated in the **seed paper** AND independently stated in **≥1 distinct other retrieved paper**, each with its own cited passage. The corroborating paper must not be a review that merely echoes the seed. | `corroborated (N papers)` |
| **C** | A single source only — the seed alone, or one retrieved paper alone. | `single-source (review)` |
| **D** | No resolvable provenance at all. | `UNSOURCED (review)` — rendered loudly, never quietly |

### How each tier is computed

**Tier A** — `mapping_meta.resolution.status`, `mapped_ids.uniprot` / `.drugbank`, `pathbank_*_id`, or `db_status`. Note that `best_effort_fallback` (`map_ids.py:3823`) marks a *weak guess* promoted to `status="mapped"` — a `best_effort=True` result is **not** Tier A; downgrade it to B or C and render the `best_effort` flag.

**Tier B** — read from the **pre-merge** payload (`rag_result.payload` / `SynthesisResult.payload`, reachable at `streamlit_app.py:439-445`), **not** the final merged payload. `pipeline.py:1728-1751` (`_clean_processes`) rebuilds every reaction as a fresh whitelisted dict and silently drops `rag_provenance`, `source_papers`, `rag_confidence` and `source_refs`; `pipeline.py:1637-1678` does the same to enzyme actors. Only `_clean_entities` (`pipeline.py:1549`) is a pass-through.

Distinct-paper test: a pointer counts as a genuine external paper **only when `chunk_id` is non-empty** (or its `source_id` is a key of `rag_result.selection.scores`). Do **not** use `source_id == "seed_paper"` as the seed test — `_seed_row_provenance` (`synthesize.py:775-797`) converts a seed row's `source_refs` into a `source_id` *before* the sentinel fallback is reached, and Stage-1's prompt (`llm/prompts/pwml_system.txt:117`) fills `source_refs` with verbatim quotes. The result is that a whole quoted sentence masquerades as a paper id, manufacturing phantom papers and inflating any naive distinct-paper count.

Entities have no `evidence` key — `synthesize.py:1030` calls `_attach_provenance(row, prov, [], …)` with an empty evidence list. So "its own cited passage" for an entity requires joining: entity name → every reaction whose `inputs`/`outputs`/`enzymes` name it → that reaction's `evidence[]` filtered to records with a non-empty `chunk_id`.

**Review detection** — `source_id` → `rag_result.selection.scores[source_id].document_type` (`select.py:100`, set at `select.py:428`). `"multi_example_review"` means the corroboration is a review; label it *"review (may restate the seed)"* rather than claiming independence. Reviews are penalised (`_REVIEW_PENALTY_MATCH = 0.25`, `select.py:70`) but **not always dropped**, so review chunks routinely reach synthesis.

**What cannot be detected:** a review that *cites the seed paper*. There is no reference list, no DOI-to-DOI edge, and no bibliographic metadata — `CandidatePaper` (`acquire.py:89-96`) stops at `id/source/title/abstract/organism/full_text/source_uri/year`. Ingest deliberately discards the references section (`ingest.py:127, 311`) and `_is_bibliography_text` (`synthesize.py:601-617`) discards citation-dense chunks. State this limitation in the UI copy; do not imply independence you cannot verify.

### Citations: what can honestly be printed

**There is no page number anywhere in the RAG path, and for acquired papers none is derivable.** JATS XML is flattened to a single whitespace-collapsed line by `map_ids._plain_text_from_xml` (245-253) and `_candidate_alias_text` (192-196) before it ever reaches the chunker. The chunk `ordinal` is hashed into `_chunk_id` (`ingest.py:173-176`) and never stored. Real per-page text exists only in `extraction/pdf_parser.py` for the uploaded seed PDF, and is destroyed in `_join_page_texts` (`pdf_parser.py:284/297`) — and the seed is never chunked into the store at all (`ingest.py:538-541` chunks only `selection.selected`).

The finest honest locator available today:

```
{source_title} ({source_id}) — {section} section
> "{evidence.text}"                        [retrieval score {evidence.score:.2f}]
```

Link with `source_uri`. **Never fabricate "p. N".** `section` is one of `abstract | introduction | methods | results | discussion | conclusion | body | figure | reaction | example`.

`rag_confidence` is **not** a probability. `_confidence` (`synthesize.py:1086-1090`) returns `max(retrieval scores)` when any exist, else the synthetic `min(1.0, 0.5 + 0.1 × prov_count)` — so `0.6` almost always means "no retrieval score at all, one provenance pointer" (the seed case). Do not display it as a percentage without that caveat.

---

## Not changed

With research mode **off**, the following is byte-for-byte identical to today. This is the regression firewall and it must be verified, not assumed — the house precedent is `docs/change_log.md:31-34` ("`RAG_ENABLED=false` behavior is byte-for-byte preserved (verified: full suite green with `RAG_ENABLED=false`)").

- **Every abort still aborts.** The default value of the strictness switch is `"pathwhiz"`; no code path changes.
- **`RUNTIME_SCHEMA_MODE` stays `"report"`** (`stage_contracts.py:25`) and `streamlit_app.py:160` still calls the runtime contract with `mode="enforce"` for merge-safety diffing — that call site wants the exception and is never relaxed.
- **`strict_db` semantics are untouched.** Research mode *implies* `strict_db=False` but does not redefine it.
- **Stage 2B mapping is already non-wrapping** (`streamlit_app.py:1677-1678` passes `allow_complex_wrapper_creation=False`, `allow_structural_cleanup=False`) and does not change. Only the Stage 6 call at `streamlit_app.py:2396-2397` flips.
- **The separation invariant holds.** No stage module imports `t2pw.rag`; `grep -rn "t2pw.rag" src/t2pw/pipeline src/t2pw/mapping src/t2pw/curation src/t2pw/pwml` still returns nothing.
- **Additive RAG keys remain optional and absent when RAG is off** (`docs/rag/03_separation_invariant.md:108-110`).
- **`tests/test_rag_provenance_gates.py` is not edited.** Its docstring declares it the separation-invariant tripwire: "the fix is in RAG, never in the gate." Research mode adds *paired* tests elsewhere.
- **Identifier-format rules that keep a colon stay.** `qa.py:116` requires the literal `CHEBI:` prefix. No "name leniency" change may touch identifier fields.
- **Anti-fabrication guards stay on in both modes.** `ingest.py:308` (references dropped), `synthesize.py:601` (bibliography chunks refused), `synthesize.py:643` (non-parseable source types refused). Research mode is fail-open about *format*, never about *inventing evidence*.
- **The PathWhiz-strict export path remains the default and the recommended one for anything destined for PathBank.** Research mode produces a research artifact, not an importable PWML file.

---

## Open questions for human review

Every check the surveys could not confidently categorise, plus the judgement calls that most need a second opinion.

### Genuinely UNCLEAR (survey could not assign a category)

These five are container/row type guards in `stage_contracts.py`. They are neither PathWhiz importer rules nor biological claims — they are crash guards. All are recommended **UNCHANGED (keep aborting)**, but that needs sign-off because a strict reading of "FORMAT rules get skipped" would relax them, and relaxing them converts a clean `StageContractError` into an `AttributeError` deep in a downstream `.get()`.

1. `invalid_payload` — `stage_contracts.py:427`
2. `entities_required` — `stage_contracts.py:430`
3. `processes_required` — `stage_contracts.py:432`
4. `entity_not_object` — `stage_contracts.py:451`
5. `process_not_object` — `stage_contracts.py:477` (also the gate that lets the five per-bucket BIOLOGY checks run at all — demoting it *silently skips them*)

### Judgement calls needing sign-off

6. **`species_required` (`stage_contracts.py:116`)** — resolved to BIOLOGY/FLAG. But `rag/synthesize.py:1190` documents *fabricating buckets purely to dodge this abort*. Confirm we want the fabrication to stop once the abort is gone.
7. **`actor_schema_not_canonical` (`stage_contracts.py:601`)** — one code covers `entity` (the catalyst's biological identity, KEEP) and `entity_type` (PathWhiz bucket tag, SKIP). Should the check be split into two codes?
8. **`drop-no-op-reaction` (`process_normalizer.py:3452`)** — marked SKIP. A novel pathway can legitimately contain a translocation or conformational change with identical species on both sides. Confirm.
9. **`drop-non-protein-catalyst` (`process_normalizer.py:2386`)** — marked SKIP, so a compound-typed enzyme actor survives into later stages. Confirm downstream tolerates it when wrapper creation is off.
10. **`drop-forbidden-or-byproduct-complex` (`process_normalizer.py:1385`)** — the "name ends in *acid*" suffix rule will dissolve legitimate novel acid-conjugated complexes. Confirm SKIP.
11. **`drop-pathway-metadata-blob` (`process_normalizer.py:285`)** — the >12-word branch can eat a legitimately long novel entity name. Marked UNCHANGED (it is an existing sanctioned exception). Confirm.
12. **`transport_element_count` (`ir.py:2992`)** — lower bound is BIOLOGY, upper bound of 3 is a PathWhiz schema cap and is FORMAT. Split, or keep as one BIOLOGY/FLAG?
13. **`ir.py:2792` + `writer.py:817` (empty complex)** — marked SKIP, but the writer raises on the same condition. Skipping one without the other converts a clean error into a mid-serialization crash. Confirm the writer is in scope.
14. **`ir.py:2809` (warn) vs `writer.py:804` (raise)** — the same condition warns in the validator and raises in the serializer. This is a latent crash **today**, independent of research mode. Fix as part of this work?
15. **`writer.py:134` `_assert_reaction_member_anchor`** — a bare `assert` on layout geometry with no message. Marked UNCHANGED, but it is a coordinate rule and therefore arguably skippable FORMAT.
16. **`map_ids.py:5463` Unknown-sentinel substitution** — fabricates a PathBank identity *and* forces species to Arabidopsis thaliana. Marked SKIP (never fabricate). Confirm an unmapped enzyme is acceptable to downstream consumers.
17. **`synthesize.py:1077` `_ALLOWED_ROW_KEYS` assert** — fires a bare `AssertionError` with no message if research mode attaches any new annotation key to a payload row. Choose: widen `_ALLOWED_ROW_KEYS` (`synthesize.py:1080-1083`), or attach annotations off-payload. **This is a hard blocker for on-payload annotation and must be decided before implementation starts.**
18. **Does research mode change what RAG synthesis is allowed to EMIT?** `rag/synthesize.py:1255` self-gates on `validate_post_extraction`. Relaxing that validator changes RAG *output*, not just export. Confirm this is intended.
19. **Does research mode need a `docs/rag/03_separation_invariant.md` "Sanctioned exceptions" entry?** Required if and only if a core stage file gains a research-mode branch. The invariant's third "Don't" (line 99-100) reads *"Weaken a gate to let synthesized content through… never the gate."* The argument that research mode is outside that prohibition — it branches on an explicit export-policy parameter, not on RAG state — needs an explicit human ruling before code lands.

---

## How it is implemented

The change is deliberately small. Five of the six seams already had the mechanism.

**The switch.** `src/t2pw/pipeline/export_mode.py` (new) holds `ExportMode = Literal["pathwhiz", "research"]`, the FORMAT code set, the structural-guard code set, and `relax_report`. It imports nothing from the RAG package, so the separation invariant's verification grep stays clean. `coerce_mode` resolves anything unrecognised to `pathwhiz`, so a typo degrades to strict rather than silently relaxing every gate.

**Stage contracts.** No validator body was parameterized. `run_stage_contract(validator, *args, mode=...)` (`stage_contracts.py`) calls the validator unchanged and re-severities the *report* afterwards. In strict mode it is a plain passthrough — which is why strict behaviour is byte-for-byte identical by construction rather than by inspection. FORMAT findings move to `warnings` stamped `research_mode="skipped_format"`; everything else moves to `warnings` stamped `review_flag`; structural guards stay in `errors` and are re-raised.

**Normalizer.** The mode rides on the shared `report` dict (`_research_mode(report)`) rather than on a new keyword of every pass — `report` is already threaded through all of them and through their nested closures, so this avoided adding the same argument to a dozen signatures. `_new_report()` deliberately does not set the key, so a strict report dict is unchanged. What research mode changes there:

- the three **uncaught `ValueError`s** in composite materialization (`_rewrite_token`, `normalize_composites` ×2) no longer kill the run; the content is kept and recorded;
- the **hidden abort** (`assert actor_contract.get("ok") is True`) becomes a recorded flag;
- `drop_process_orphan_proteins` and `prune_disconnected_proteins` are skipped, so proteins a paper stated but that are not yet wired into a reaction survive for review;
- `_record_non_protein_catalyst_drop` became the single decision point for all five non-protein-catalyst drop sites, returning `True` only in strict mode.

Every preserved row lands in `report["actions"]` under a `research_mode_` prefix, so nothing is dropped silently.

**Mapping.** `map_payload(..., mode=...)` only makes *name handling* lenient: `route_entity_for_mapping(..., lenient_names=True)` stops reading a `:` as PathWhiz complex syntax. Dropping the protein-complex wrapper needed **no new mapping code at all** — `allow_complex_wrapper_creation` already existed and already had a skip branch; the orchestrator flips it at the Stage-6 call.

**Orchestration (seam S5).** `streamlit_app.py` renders the `Export mode` radio, threads `export_mode` into `run_post_pipeline_sbml_artifacts`, and collects downgraded findings into `research_review_flags` / `research_skipped_format_rules` / `research_normalization_actions` for the Review-flags panel.

### Separation invariant

`grep -rn "t2pw.rag" src/t2pw/pipeline src/t2pw/mapping src/t2pw/curation src/t2pw/pwml` still returns nothing. Tier assignment needs both the UniProt resolver (core) and RAG retrieval, so it lives in `t2pw.rag` (RAG → core, the allowed direction) and reaches the app through seam S5.

Core stage files *do* now contain a research-mode branch (`stage_contracts.py`, `process_normalizer.py`, `map_ids.py`). That branches on an explicit, user-selected export policy — not on RAG state — so it does not "do something different when RAG is on". It is recorded as a sanctioned exception in `docs/rag/03_separation_invariant.md`.

---

## Survey conflicts

Eight parallel surveys produced this table. Where they disagreed, or where one was verifiably wrong, it is recorded here rather than smoothed over.

**1. `strict_db=False` does NOT implement research mode.** One survey recommended that research mode simply imply `strict_db=False` because "that already covers required external DB identity FOR IMPORT". A different survey proved otherwise, and it is **verified**: inside `validate_required_pwml_contract` the nested helpers `require_db_identity` (`ir.py:2059`), `component_index` (`ir.py:2081`) and `protein_external_id` (`ir.py:2100`) are **defined but never called**. `strict_db` therefore has exactly **one** live effect in that function — the species early-continue at `ir.py:2352`. `protein_missing_external_identity`, `protein_missing_species` and `reaction_enzyme_must_be_protein_complex` are all unconditional. The first survey's recommendation is wrong on this point; research mode needs its own mechanism.

**2. `missing_db_identity` (`ir.py:1126`) is dead code.** One survey listed it as a live blocking check. **Verified by inspection of `ir.py:1055-1093`: all five `entity_specs` tuples pass `strict_required=False`, so the guard at `ir.py:1118` (`strict_db and strict_required`) can never be true.** The row is kept in the table and marked dead so nobody re-derives it.

**3. `actor_schema_not_canonical` — "does not abort" vs "aborts".** Survey 1 reported that `validate_post_normalization` returns at line 201 without raising, so the check is non-blocking. Survey 2 reported it aborts. **Both are correct from different vantage points, and survey 2 is the operationally relevant one**: `process_normalizer.py:4394` does `assert actor_contract.get("ok") is True`, converting the finding into an `AssertionError` inside `normalize_process_payload`, which `streamlit_app.py:1826` swallows as a generic gate failure. Any implementation that relaxes only `validate_post_normalization` will still kill the run.

**4. Line-number granularity, not conflict.** Survey 1 cited `_validate_canonical_actor_rows` at `stage_contracts.py:601`; survey 2 cited `586`. **Verified: `def` at 586, the `_add_error(` call at 601, the code string at 603.** Surveys consistently cite the *call site*; message strings sit 1–2 lines below. The same pattern was verified for `process_normalizer.py:3911`/3913, `3942`/3944, and `stage_contracts.py:116`/118. Line numbers in this table are trustworthy at ±2.

**5. `duplicate_reaction_enzyme_complex` line.** Survey 3 said `ir.py:2621`, survey 4 said `ir.py:2622`. **Verified: the code string is at 2622, the `err(` call opens at 2621.** Trivial; 2621 used here per the call-site convention.

**6. Category disagreement on protein species.** Survey 2 called `gate-protein-missing-species` (`process_normalizer.py:3936`) MIXED; survey 3 called `contract-protein-missing-species` (`ir.py:2218`) BIOLOGY; survey 4 called `strategy-protein-needs-species` (`map_ids.py:4254`) BIOLOGY. Resolved uniformly to **BIOLOGY / FLAG** — the requirement exists because PathWhiz needs a `species_id` column, but the *value* is a biological assertion and must never be fabricated.

**7. `protein_complex_missing_components` is emitted at six different sites with three different severities** (`ir.py:1183` warn, `ir.py:1255` warn, `ir.py:2249` error-if-generated, `ir.py:2792` error-unless-DB-id, plus `writer.py:817` raise and `map_ids.py:963`/`2339`). A code-keyed category table alone is therefore **not** sufficient for this code — the severity table must be keyed on `(code, pointer-prefix)` or each site must get a distinct code.

**8. Does entity provenance survive the merge?** Survey 5 stated `_clean_entities` (`pipeline.py:1549`) is a pass-through so entity rows keep all four additive keys, while reactions lose them at `pipeline.py:1728-1751`. Survey 6 flagged the same question as **unverified** ("verify `clean_inference_output` does not whitelist-drop `rag_provenance`"). Treat this as **open**: the safe implementation reads the pre-merge `SynthesisResult.payload`, which is correct either way.

**9. Two checks that can never fire, reported as live.**
- `qa.py:57` (spontaneous reaction with an enzyme) is **unreachable from the IR path** — `reaction["spontaneous"]` is hardcoded `False` at `ir.py:1603`.
- `rct_missing_left` / `rct_missing_right` / `rct_missing_enzyme` (`ir.py:3018/3020/3022`) **always fire** for any reaction-coupled transport, because `build_pwml_ir` constructs every RCT with empty left/right/enzymes (`ir.py:1937-1945`). RCTs are effectively unsupported today, independent of research mode.

**10. Known false-positive risks flagged by the surveys, not fixed here.** `_has_plus_token` (`process_normalizer.py:219`) strips only a *trailing* charge, so `Ca2+ ion` reads as a composite. `_is_likely_byproduct` (`process_normalizer.py:697`) treats any name ending in `acid` as a byproduct. `ir.py:55` `_norm` rewrites anything outside `[a-z0-9:+ ]` to a space, so `3-oxo` and `3 oxo` collide — that, not the colon, is the real name-collision risk. `entity_identity.py:28` `_normalize()` **deletes** `:` entirely while `ir.py:55` **preserves** it, so the two modules disagree about whether `A:B` is `ab` or `a:b`. Worth aligning if name handling is made lenient.

**11. Suspicious line numbers.** None found. All spot-checked citations resolved to the expected construct. File lengths are consistent with the highest line cited in each: `stage_contracts.py` 664, `process_normalizer.py` 4395, `ir.py` 3114, `map_ids.py` 6874, `writer.py` 2767, `validate.py` 484, `qa.py` 127, `provenance.py` 276, `synthesize.py` 1343.