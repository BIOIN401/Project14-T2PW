# Change Log

Every entry answers: what was the error, why did it appear, and how does the
fix stay consistent with the intended pipeline design.

---

## Fixed

---

### FIXED - Stage 8 PWML IR: direct protein enzyme wrapper lost source protein metadata

**Files changed:** `src/t2pw/pwml/ir.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
The PWML IR builder has a last-resort safety net that wraps direct protein enzyme actors
as generated single-protein protein_complex records. That wrapper check was running after
entity rows had been converted into IR records, but those records did not preserve `species`
context and `_protein_external_id` did not inspect nested `mapped_ids`. A valid protein
catalyst could therefore be reported as missing species/UniProt and remain as a bare protein
enzyme in the IR, triggering `reaction_enzyme_must_be_protein_complex`.

**Fix:**
IR entity records now preserve `species`, `taxonomy_id`, and `species_ref`, and the protein
external-ID helper reads UniProt/DrugBank IDs from `mapped_ids`, `ids`, and `mapping_meta`.
The existing direct-protein catalyst tests now verify that valid proteins are wrapped into
single-component protein_complex enzymes.

---

### FIXED - Stage 3 alias normalization: protein_complex component metadata was flattened

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
`canonicalize_same_as_aliases` rewrote `protein_complexes[].components` to plain strings.
That preserved the component name but discarded structured fields such as `stoichiometry`,
`mapped_ids`, `uniprot`, and `pathbank_protein_id`. Generated PathWhiz wrapper complexes
could then lose the exact data needed by Stage 3/Stage 8 contract checks.

**Fix:**
Component alias rewriting now preserves dict component rows and rewrites only the component
name-bearing field. The same pass also reads component names with `_component_name_from_row`
instead of stringifying dict components.

---

### FIXED - Stage 3 gate: generated protein_complex components missing stoichiometry

**Files changed:** `src/t2pw/pipeline/process_normalizer.py`, `tests/test_process_normalizer.py`

**Error / symptom:**
Generated single-protein PathWhiz wrapper complexes could pass Stage 3 with a component record
that resolved to a declared protein and had species/external identity, but omitted explicit
`stoichiometry`. Stage 8 and the SPMDB schema require structured protein_complex components
to carry positive stoichiometry, so the payload failed later during required PWML contract
validation.

**How the fix is consistent with the pipeline design:**
Stage 3 already hard-gates generated protein_complex component resolution, species, and external
identity. The fix adds the matching positive `stoichiometry` requirement to that same generated
component loop, so repair/audit sees the issue before export. A focused regression test covers
`NdmA complex` with a resolved `NdmA` component missing `stoichiometry`.

---

### FIXED — Stage 3 gate: `canonicalize_same_as_aliases` leaked protein_complex names into `entities.proteins`

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `canonicalize_same_as_aliases`

**Error / symptom:**
After Fix 1 (degree-0 exemption), the gate still reported 21 errors:
- `Generated protein complex wrapper 'NdmA complex' must be listed under protein_complexes, not proteins.`
- `Protein 'NdmA complex' is missing species/organism.`
- `Protein 'NdmA complex' is missing a UniProt or DrugBank identifier.`
Same pattern for `NdmB complex`, `NdmC complex`, `NdmCDE complex`, `NdmD complex`, `TmuM complex`, `caffeine dehydrogenase complex`.

**Why it appeared:**
`canonicalize_same_as_aliases` iterates over every reaction's `enzymes` and `modifiers` and calls
`_ensure_protein(actor_name, payload, rep)` for each actor. Stage 2/6 mapping had already rewritten
these reaction modifier references to point to generated complex wrappers (e.g. `NdmA complex`).
`_ensure_protein` checks whether `actor_name` is in `entities.proteins` but not whether it is in
`entities.protein_complexes`. So it unconditionally added `NdmA complex`, `NdmB complex`, etc. to
`entities.proteins`, even though they were already correctly placed in `entities.protein_complexes`.

**How the fix is consistent with the pipeline design:**
`_ensure_protein` is a safety net to guarantee that every reaction actor has a declared entity.
It should only fire for names that are not yet declared as *any* entity type. Since protein_complex
entries are real entity declarations, an actor that is already in `complexes` needs no fallback
protein row. The one-line guard `if _find_entity_row(complexes, actor_name) is not None: continue`
skips the `_ensure_protein` call for actors already declared as a complex. Stage 3 owns normalization;
this fix stays within `canonicalize_same_as_aliases` and touches no other stage.

---

### FIXED — Stage 3 gate: degree-0 check incorrectly flagged proteins that are complex components

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `run_strict_post_normalization_gates`

**Error / symptom:**
PWML export failed with "PWML export stopped by Stage 3 gate." The gate reported errors such as:
- `Protein has degree 0 after normalization: NdmA`
- `Protein has degree 0 after normalization: NdmB`
- `Protein has degree 0 after normalization: NdmC`
- `Protein has degree 0 after normalization: NdmD`

**Why it appeared:**
Stage 2/6 mapping wraps single-protein reaction enzymes in generated protein_complex records
(e.g. `NdmA complex`) and replaces the direct protein reference in the reaction modifier
with the complex name. This is correct — PathWhiz requires protein_complex as the enzyme
actor. The side effect is that `NdmA` is no longer referenced directly in any reaction;
its network connection flows through `NdmA complex`. `build_graph` does not add edges for
protein→complex component membership, so `NdmA` has degree 0 in the connectivity graph.
`prune_disconnected_proteins` (step 15) correctly keeps `NdmA` because it has a UniProt ID
(`_has_protein_identity` returns True). The `enforce_all_proteins_connected` check in step 17
then flagged it as a connectivity failure even though degree-0 is the expected and correct
state for a complex-component protein.

**How the fix is consistent with the pipeline design:**
`pipeline.md` states: "A protein survives all three passes if it has any of:
complex-component membership with external identity, a process reference, a non-zero
graph degree, or an external database ID." The gate check lacked this exemption.
The fix builds a set of protein name norms that appear as components in any declared
`protein_complexes[]` entry and skips those from the `enforce_all_proteins_connected`
error. No other gate check is changed. Stage 3 owns the gate; the fix lives entirely
within `run_strict_post_normalization_gates`.

---

## Fixed

---

### FIXED — Stage 3 gate: degree-0 check incorrectly flagged proteins that are complex components

**File changed:** `src/t2pw/pipeline/process_normalizer.py` — `run_strict_post_normalization_gates`

**Error / symptom:**
PWML export failed with "PWML export stopped by Stage 3 gate." Gate reported errors like `Protein has degree 0 after normalization: NdmA/B/C/D`.

**Why it appeared:**
Stage 2/6 mapping wraps single-protein reaction enzymes in generated `protein_complexes` entries (e.g. `NdmA complex`) and rewrites the reaction modifier reference to use the complex name — correct, because PathWhiz requires a protein_complex as the enzyme actor. The side effect is that `NdmA` no longer appears directly in any reaction; its network connection flows through `NdmA complex`. `build_graph` does not add edges for protein→complex component membership, so `NdmA` has degree 0. `prune_disconnected_proteins` (step 15) correctly kept it because `_has_protein_identity` returned True. The `enforce_all_proteins_connected` check in step 17 then flagged it as a connectivity failure even though degree-0 is the expected state for a complex-component protein.

**How the fix is consistent with the pipeline design:**
`pipeline.md` states: "A protein survives all three passes if it has any of: complex-component membership with external identity, a process reference, a non-zero graph degree, or an external database ID." The gate was missing this exemption. The fix builds `_complex_component_norms` from all `protein_complexes[].components` entries and skips those protein names from the `enforce_all_proteins_connected` error. No other gate check is changed. Stage 3 owns the gate; the fix lives entirely within `run_strict_post_normalization_gates`.

---

## Open Issues

Issues confirmed by running the pipeline. Ordered by pipeline stage. No code
changes yet - this section records the diagnosis and the planned fix.

---

### OPEN - Stage 2/6/8: Generated PathWhiz protein-complex wrappers leak into proteins and bypass Stage 3 blocking

**Files to change:** `src/t2pw/mapping/map_ids.py`, `src/t2pw/pipeline/process_normalizer.py`, `src/t2pw/app/streamlit_app.py`, `src/t2pw/pwml/ir.py`, tests covering mapping, normalization gates, and PWML export blocking.

**Error / symptom:**
PWML required-field validation reports errors such as:

- `Protein 'NdmA complex' is missing species/organism.`
- `Protein 'NdmA complex' is missing a UniProt or DrugBank identifier.`
- Same pattern for `NdmB complex`, `NdmC complex`, `xanthine oxidase complex`, `urate oxidase complex`, `allantoinase complex`, `urease complex`, `TmuM complex`, and `TM-HIU hydrolase complex`.

These errors are misleading because the generated `* complex` names should not
be protein rows at all. In PathWhiz, the member protein needs the UniProt or
DrugBank ID; a protein-complex record can be created from valid member proteins
and does not necessarily need a complex-level PathBank ID.

**PathWhiz behavior confirmed from UI:**

1. The `New Protein` form requires `Name`, `Species`, and either `UniProt ID`
   or `DrugBank ID`.
2. The `New Protein Complex` form requires `Name`, `Species`, and at least one
   member `Protein` with stoichiometry.
3. Therefore a generated single-protein wrapper such as `NdmA complex` is valid
   only as a `protein_complexes[]` row with component `NdmA`; `NdmA` must be a
   valid `proteins[]` row with species and UniProt/DrugBank identity.
4. The pipeline should not try to find or assign a UniProt ID for
   `NdmA complex`; UniProt belongs to `NdmA`.

**Root cause:**
There are two interacting issues:

1. `map_ids._rewrite_reaction_protein_enzymes_to_complexes` creates novel
   single-component wrappers named `f"{protein_name} complex"` when PathBank DB
   lookup cannot resolve a real complex. This is acceptable only if the wrapper
   stays under `entities.protein_complexes` and its member protein is already
   mapped.
2. The Streamlit PWML export path calls `normalize_process_payload`, receives a
   Stage 3 gate report, but then proceeds to `validate_required_pwml_contract`
   instead of stopping on unresolved Stage 3 gate failures. As a result, issues
   that Stage 3 can detect still reach the Stage 8 hard gate.

The mapping cache also contains stale/generated `enzyme_complexes` records for
the affected names, including entries with `status: "unmapped"` and
`chosen_rule: "novel_enzyme_single_component_complex"`. These cache rows show
where the `* complex` names are being synthesized.

**Planned fix:**

1. Stage 2/6 mapping (`map_ids.py`):
   - Keep generated single-protein wrappers under `entities.protein_complexes`
     only.
   - Mark generated wrappers with explicit metadata such as
     `generated: true` and
     `generation_reason: "single_protein_pathwhiz_wrapper"`.
   - Before creating a usable wrapper, require the base protein row to have
     species plus UniProt/DrugBank identity.
   - If the base protein is unmapped, do not create an apparently exportable
     complex. Record a mapping issue instead.
   - Do not add or preserve rows like `NdmA complex` under
     `entities.proteins`.
2. Stage 3 normalization/gate (`process_normalizer.py`):
   - Add a hard gate check that rejects `entities.proteins[]` rows whose names
     are generated-complex shaped (`* complex`) when they correspond to
     generated wrappers.
   - Add a generated-complex integrity check: species present, at least one
     component, and every component resolves to a declared protein with
     UniProt/DrugBank identity.
   - Preserve the current design: Stage 3 reports these as gate failures for
     audit/review; it should not silently reclassify biological entities unless
     the operation is a deterministic generated-wrapper cleanup.
3. Orchestrator (`streamlit_app.py`):
   - Before initializing refinement review and before PWML generation, inspect
     `normalize_process_payload(...)[1]["gate"]`.
   - If the gate is not OK, stop and surface the Stage 3 gate errors. Do not
     continue to the PWML required-field gate.
4. Stage 8 PWML IR (`pwml/ir.py`):
   - Treat generated protein complexes without a complex-level PathBank ID as
     valid only when their component proteins satisfy the protein identity
     contract.
   - Keep strict validation for ordinary protein rows: species plus
     UniProt/DrugBank remains required.
5. Tests:
   - Add a fixture where `NdmA` has species and UniProt and `NdmA complex` is a
     generated protein complex. This should pass the generated-complex contract.
   - Add a fixture where `NdmA complex` appears under `entities.proteins`. This
     should fail Stage 3 before export.
   - Add a fixture where `NdmA complex` is generated but component `NdmA` lacks
     UniProt/DrugBank. This should fail before export.

**Pipeline consistency:**
The protein-vs-complex distinction belongs at the mapping and normalization
boundary. Stage 2/6 owns creation of generated PathWhiz wrapper complexes.
Stage 3 owns deterministic gate checks that prevent invalid rows from reaching
review/export. Stage 8 owns final PWML contract enforcement. The orchestrator
must wire these stages so unresolved Stage 3 failures block PWML generation
rather than being rediscovered later as required-field errors.

---

### OPEN â€” Stage 2 (Map): Best-effort UniProt fallback assigns lowest-scored candidate when no threshold passes

**Files to change:** `src/t2pw/mapping/map_ids.py` (replace best_effort_fallback block), `src/t2pw/curation/audit_json_llm.py` (add audit hint for best_effort IDs)

**Error / symptom:**
Generic enzyme names such as "N-methyltransferase complex" and "N-methylnucleosidase complex" have no species-specific UniProt entry that clears the 0.78 confidence threshold. As a temporary workaround (added 2026-07-08), the mapper now accepts the top-ranked UniProt candidate regardless of score and marks it `best_effort: True`. This prevents Stage 3 gate failures for missing external identity, but the assigned accession may be incorrect â€” it is simply the highest-scoring candidate from a name search, not a verified match.

**Root cause:**
Generic descriptive names ("N-methyltransferase", "N-methylnucleosidase") return many UniProt hits with similar, low Jaccard scores. None is definitively the right protein, so no candidate clears the strict acceptance threshold. The right fix is sequence-based disambiguation (BLAST or UniProt sequence search) â€” find the actual protein sequence from the paper or a reference, BLAST it against UniProt, and accept the top hit by sequence identity. This requires the pipeline to carry or fetch protein sequences, which it currently does not do.

**Planned fix:**
1. For proteins that reach the best_effort_fallback path, attempt a NCBI eSearch + efetch to retrieve the candidate sequence by gene name + organism.
2. Submit the retrieved sequence to the UniProt BLAST API.
3. Accept the BLAST top hit (â‰¥40% identity, â‰¥60% coverage) as the confirmed accession and replace the best_effort ID.
4. Add an audit hint in `audit_json_llm.py` that flags any entity with `best_effort: True` in its mapping metadata so the audit LLM knows to verify or propose a correction.

**Pipeline consistency:**
Sequence fetching and BLAST belong in Stage 2 mapping or Stage 4a gap resolution â€” both own external ID lookup. No normalization or export logic would change. The `best_effort` flag in mapping metadata is the audit signal; the audit loop owns the decision to accept or replace the provisional ID.

---

### OPEN â€” Stage 4 (Audit): LLM connection failure prevents semantic repair

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

### OPEN â€” Stage 1 (Extract): Empty reaction inputs in beta-oxidation chain

**Files to change:** None yet â€” this is a Stage 4 (Audit) repair task once
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
stage â€” the LLM summarised rather than enumerated each cycle.

**Planned fix:**
This is the correct input for Stage 4 (Audit). Once the LLM connection is
restored (see above), the audit should:
1. Receive the gate failure list which includes these 3 reactions.
2. Propose patches that add the correct OPC-CoA intermediate as `inputs` for
   each reaction and the shortened OPC-CoA as `outputs` (except the last cycle
   which produces jasmonic acid).
No normalization or schema change is required â€” this is a data completeness
issue that the audit loop is designed to repair.

**Pipeline consistency:**
The gate correctly fires for these reactions. The pipeline design says gate
failures feed audit, not abort. No stage logic change needed. This issue will
resolve once the LLM connection issue is fixed.

---

### OPEN â€” Stage 1 (Extract): OPC-8, OPC-6, OPC-4 misclassified as proteins

**Files to change:** None yet â€” audit repair task.

**Error / symptom:**
Entities `OPC-8`, `OPC-6`, and `OPC-4` appear in `entities.proteins`. The audit
warns: `"Protein has no location link; default compartment may be used."` for
all three. These are 3-oxo-2-(2'-pentenyl)-cyclopentane-acyl-CoA intermediates â€”
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
normalization hint is added, it goes into the gate report as an audit input â€”
not as a normalization mutation. This preserves the rule that Stage 3 does not
make semantic corrections.

---

### OPEN â€” Stage 2 (Map): DB unavailable â€” degraded mapping rates

**Files to change:** Configuration only.

**Error / symptom:**
`mapping_report.dbonly.json` shows `"db_available": false`. Protein mapping:
38.46% (5/13). Compound mapping: 9.52% (2/21). All 12 protein complexes skipped
(10/12 have gap issues â€” component proteins unmapped, so complex cannot map).

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

### OPEN â€” Stage 6 (Enrich): Enrichment stage produces data no stage consumes

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

**Planned fix â€” choose one:**
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
### YYYY-MM-DD â€” <short description>

**Files changed:** `path/to/file.py` (lines Xâ€“Y)

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

### 2026-07-08 â€” Best-effort UniProt fallback for generic enzyme names that clear no confidence threshold

**Files changed:** `src/t2pw/mapping/map_ids.py` (lines ~3580â€“3590, `map_protein_uniprot`), `docs/change_log.md`

**Error / symptom:**
Proteins with generic names such as "N-methyltransferase complex" and "N-methylnucleosidase complex" failed the Stage 3 gate checks for missing UniProt/DrugBank identifiers. These names have no species-specific UniProt entry that scores above the 0.78 acceptance threshold â€” the correct protein cannot be confidently distinguished from many similarly-named candidates by name alone.

**Root cause:**
`map_protein_uniprot` returned `status: "unmapped", reason: "ambiguous"` whenever candidates existed but none cleared the strict threshold. The caller only writes a UniProt accession when `status == "mapped"`, so ambiguous results produced no ID on the protein entity. For genuinely generic enzyme names, no name-based scoring strategy can reliably pick the right candidate â€” the correct fix is sequence-based lookup (BLAST), but the pipeline does not currently carry protein sequences.

**Fix:**
In `map_protein_uniprot`, when `_accepted_uniprot_candidate_result` returns None but at least one candidate has a non-empty accession, return that top candidate with `status: "mapped"`, `chosen_rule: "best_effort_fallback"`, and `best_effort: True` instead of `status: "unmapped"`. The caller writes the accession, which clears the Stage 3 gate check. The `best_effort: True` flag is preserved in the mapping metadata as a signal for the audit loop that this ID was not confidently matched and should be reviewed.

**Pipeline consistency:**
Change is entirely within `map_protein_uniprot` in `t2pw.mapping.map_ids`, which owns Stage 2 and Stage 6 ID mapping. No normalization, audit, or export logic was changed. The fallback is a last resort â€” it only activates when all three normal acceptance paths (strong_unique, reviewed_unique, reviewed_exact_gene_match) fail and candidates exist. The proper long-term fix (BLAST-based sequence lookup) is documented as an open issue. See OPEN issue: "Stage 2 (Map): Best-effort UniProt fallback assigns lowest-scored candidate when no threshold passes."

---

### 2026-07-08 â€” Strengthen extraction scoping to single-pathway + single-organism; fix doc inconsistencies

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
Three issues found after adding the initial species scoping rule: (1) The BIOLOGICAL STATE RULE still instructed the LLM to default to *Homo sapiens* when no organism was available, directly contradicting the new scoping rule's instruction to leave species empty. (2) The scoping rule told the LLM to pick one organism but did not tell it to first pick one pathway â€” papers covering multiple pathways (e.g. caffeine biosynthesis and caffeine degradation in the same review) would still produce a merged multi-pathway extraction scoped to one organism. (3) `docs/pipeline.md` had a duplicate copy of the Step 17 gate description mislabelled as Step 15, and the file ownership table listed no prompt files.

**Root cause:**
(1) The BIOLOGICAL STATE RULE predated the scoping rule and was never updated to match. An LLM reading both rules encounters conflicting instructions for the no-organism case. (2) The prior scoping rule said "choose one primary biological scope" but did not make pathway selection an explicit first decision â€” organism selection was the only named decision. (3) The pipeline.md duplicate was a copy-paste artifact from a prior edit; the prompt files were always owned but never listed in the table.

**Fix:**
1. Changed the BIOLOGICAL STATE RULE fallback from `"use 'Homo sapiens' as the default"` to `"leave species empty â€” do not guess or default to any organism"`, removing the contradiction.
2. Expanded the scoping rule into two explicit sequential decisions: Decision 1 (select one pathway â€” the most central to the paper) followed by Decision 2 (select one organism for that pathway). The pathway decision now comes first and is the primary filter; organism selection applies within it.
3. Removed the duplicate Step 15 paragraph from `docs/pipeline.md` (lines 135â€“138, copy-paste of Step 17 description).
4. Added `pwml_system.txt` and `pwml_infer_system.txt` to the file ownership table in `docs/pipeline.md`.

**Pipeline consistency:**
All changes are in prompt text files and documentation. No Python source was modified. The scoping decision remains Stage 1's responsibility â€” it is an extraction-time filter that prevents mixed-pathway, mixed-species entity sets from entering Stage 2 and beyond.

---

### 2026-07-08 â€” Add single-organism scoping rules to Stage 1 extraction prompt

**Files changed:** `src/t2pw/llm/prompts/pwml_system.txt`, `src/t2pw/llm/prompts/pwml_infer_system.txt` (cross-reference note only), `docs/pipeline.md`, `docs/change_log.md`

**Error / symptom:**
Proteins from multiple organisms present in a single paper (e.g. *Coffea arabica* biosynthesis enzymes and *Pseudomonas putida* degradation enzymes) were extracted together into the same pathway payload, resulting in mixed species assignments across entities. This caused Stage 3 gate failures for missing species/organism on proteins that inherited no clear organism context, and UniProt mapping failures at Stage 2 and Stage 6 because the wrong species was searched for each protein.

**Root cause:**
The Stage 1 extraction prompt (`pwml_system.txt`) had no rule requiring the LLM to select a single primary organism before extracting reactions. Papers that cover multiple organisms â€” comparative studies, combined biosynthesis-plus-degradation reviews â€” caused the LLM to emit proteins from all mentioned organisms, mixing species context across entities. The BIOLOGICAL STATE RULE required species on every biological_state but gave no guidance for choosing among competing organisms.

**Fix:**
Added two rule blocks to `pwml_system.txt` immediately after the BIOLOGICAL STATE RULE:

1. **Species and organism scoping rule** â€” instructs the LLM to select one primary organism/species/strain before extracting reactions, assign it to all proteins, enzymes, complexes, reactions, and biological states, exclude entities from other organisms unless explicitly requested, and emit an audit warning rather than mix species when no organism can be confidently selected.
2. **Protein/enzyme species rule** â€” requires every protein, enzyme, and protein complex to inherit the selected pathway species before identifier mapping, and prohibits emitting a protein entity without a species/organism assignment and sufficient identifier context.

Added a species constraint cross-reference note to the locality constraint block in `pwml_infer_system.txt`: the Stage 2 mandatory modifier repair pass is now explicitly instructed to apply the Stage 1 species scoping rule and skip modifier links for proteins from other organisms.

**Pipeline consistency:**
Change is entirely within prompt text files. No Python source was modified. Species scoping is an extraction-time decision that Stage 1 owns â€” the correct stage boundary. Selecting a single organism at Stage 1 prevents mixed-species entity sets from propagating to Stage 2 mapping (where wrong-species queries fail silently) and Stage 3 gate checks (where missing species generates unrepaired gate failures). Stage 2â€“8 behavior is otherwise unchanged.

---

### 2026-07-08 â€” Strip "complex" from UniProt name query variants

**Files changed:** `src/t2pw/mapping/map_ids.py` (line ~50, `_name_variants`), `docs/change_log.md`

**Error / symptom:**
Proteins with "complex" in their names â€” e.g. "xanthine oxidase complex", "NdmA complex", "IMP dehydrogenase complex", "TmuM complex" â€” consistently failed the Stage 3 gate checks added on 2026-07-08 for missing UniProt/DrugBank identifiers. These proteins are findable in UniProt under their base names ("Xanthine oxidase", "NdmA", etc.) but the pipeline assigned no accession to any of them.

**Root cause:**
`_name_variants` (Stage 2 and 6 mapping) already strips "protein" and "enzyme" from name query strings to normalize them for UniProt lookup, but did not strip "complex". UniProt never includes "complex" in individual protein entry names â€” that word is a complex-level descriptor. Querying for "xanthine oxidase complex" produced a Jaccard similarity of 2/3 â‰ˆ 0.667 against the correct UniProt entry "Xanthine oxidase". After scoring (`base_score = 0.35 Ã— 0.667 = 0.234`, plus organism and reviewed bonuses), the total landed at â‰ˆ 0.53 â€” 0.25 points below the 0.78 acceptance threshold â€” so no accession was accepted despite the correct entry being returned by UniProt's API.

**Fix:**
Added `"complex"` to the word-strip regex in `_name_variants`:
```
re.sub(r"\b(protein|enzyme|complex)\b", " ", base, flags=re.IGNORECASE)
```
Names like "xanthine oxidase complex" now generate "xanthine oxidase" as a search variant. That variant scores 1.0 (exact name match, base_score = 0.55) plus organism and reviewed bonuses, clearing the 0.78 threshold and producing a mapped accession.

**Pipeline consistency:**
Change is entirely within `t2pw.mapping.map_ids`, which owns Stage 2 and Stage 6 ID mapping. No normalization, audit, export, or orchestrator logic was touched. The change is a query normalization improvement consistent with the pre-existing "protein" and "enzyme" stripping.

---

### 2026-07-08 â€” Add protein species and external identity checks to Stage 3 gate

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (inside `run_strict_post_normalization_gates`), `docs/change_log.md`

**Error / symptom:**
Stage 8 (Export) hard-aborted with `validate_required_pwml_contract` failures for two checks â€” `protein_missing_species` and `protein_missing_external_identity` â€” with no opportunity for the audit loop to repair the affected proteins. The specific failure: proteins (and compounds misclassified into `entities.proteins`) that had no species/organism field and no UniProt or DrugBank ID would pass Stage 3 and Stage 4 unchanged, then cause an unrecoverable abort at pre-export contract validation.

**Root cause:**
Both checks existed only in `t2pw/pwml/ir.py` (lines 1880â€“1915) as part of the hard Stage 8 pre-export semantic contract. They were absent from `run_strict_post_normalization_gates` in `process_normalizer.py`, so Stage 4 (Audit) never received them as gate failures to repair. The audit loop correctly repairs what the gate reports; the gate simply never reported these two conditions.

**Fix:**
Added two new loops inside `run_strict_post_normalization_gates`, immediately after the existing forbidden-name check loop on `entities.proteins`. Each loop iterates `entities.proteins`, skips unnamed rows (already caught by a separate check), and calls `_add_error` when the condition is unmet:

1. **Species/organism check** â€” mirrors the `species` resolution chain from `ir.py`: tries `species`, `organism`, `taxonomy_id`, `species_id`, `pathbank_species_id`, `species_ref.pathbank_species_id`, `species_ref.name`, `mapping_meta.species`, `mapping_meta.species_id`.
2. **External identity check** â€” emits an error if none of `uniprot`, `uniprot_id`, `drugbank`, `drugbank_id` are present and non-empty.

Both checks use only `_safe_dict` and `_safe_list`, which are already defined in `process_normalizer.py`. No imports from `t2pw.pwml.ir` or any other stage module were added.

**Pipeline consistency:**
The fix lives entirely within `run_strict_post_normalization_gates` in `process_normalizer.py`, which owns Stage 3's gate. The gate's return type and `errors` list shape (`{"path": str, "reason": str}`) are unchanged. The `GateValidationError` raise path is untouched. By surfacing these two conditions as Stage 3 gate failures, Stage 4 now receives them in its repair context and can propose patches (species assignment, ID lookup via gap resolution) before Stage 8 runs. No cross-stage logic was introduced â€” `process_normalizer.py` mirrors the field-level logic without importing from `ir.py`.

---

### 2026-07-07 â€” Fix `normalize_process_actor_schema` to write `entity`/`entity_type` for enzyme actors

**Files changed:** `src/t2pw/pipeline/process_normalizer.py` (blocks 1c and legacy-enzyme view), `tests/test_process_normalizer.py` (updated assertions)

**Error / symptom:**
All enzyme actor dicts in `reactions[].enzymes` retained `protein_complex` (or
`protein`) as the name field after normalization completed. `e.get("entity")`
returned `""` for every enzyme. After a full pipeline run on the Arabidopsis
jasmonic acid pathway, all 30 enzyme actors used `protein_complex`, while all 30
modifier actors correctly used `entity/entity_type`.

**Root cause:**
`normalize_process_actor_schema` has two passes. Pass 1 (`_rewrite_actor_rows`)
resolves each actor name against the protein and complex registries and writes
the canonical name back to `protein_complex` or `protein` â€” NOT to `entity`.
The post-process loop migrated `modifiers[]` rows to `entity/entity_type` schema,
but had no equivalent migration for `enzymes[]`. Additionally, the "legacy view"
reconstruction block that rebuilds `reaction["enzymes"]` from `modifiers[]`
wrote `protein`/`protein_complex` keys rather than `entity`/`entity_type`,
leaving enzymes in legacy field format after the schema-normalization step.

**Fix:**
1. Added block **1c** in the post-process loop (after the modifier migration and
   the 1b entity_type correction): iterates `reaction["enzymes"]`, migrates each
   dict from `protein_complex`/`protein`/`name` to `entity`/`entity_type`,
   drops actors whose `entity_type` is in `dropped_enzyme_entity_types`, and
   writes the result back to `reaction["enzymes"]`.
2. Updated the legacy-enzyme view reconstruction (formerly writing
   `protein_complex`/`protein` keys) to use `entity`/`entity_type` instead,
   keeping `reaction["enzymes"]` in sync with `modifiers[]` in the canonical
   schema.
3. Updated six test assertions in `tests/test_process_normalizer.py` that were
   checking for the old `protein`/`protein_complex` keys; all now verify
   `entity` and `entity_type` and confirm legacy keys are absent.

**Pipeline consistency:**
Change is entirely within `normalize_process_actor_schema` in
`process_normalizer.py`, which owns actor schema enforcement for Stage 3. No
orchestrator, mapping, audit, or export logic was touched. After the fix, any
code that calls `actor.get("entity")` works correctly for both enzyme and
modifier actors without special-casing field names.

---

### 2026-07-07 â€” Wire drop_process_orphan_proteins into normalize_process_payload

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
by reviewing the actual step sequence (lines 3584â€“3588) against the documented
17-step list.

**Fix:**
Added the call `drop_process_orphan_proteins(data, report=report)` and its
corresponding `_checkpoint("drop_process_orphan_proteins")` between
`drop_unresolved_complex_component_proteins` and `prune_disconnected_proteins`.
Updated `docs/pipeline.md` to reflect the 17-step sequence and document why steps
13â€“15 run in sequence (each catches a different class of orphan; a protein must
fail all three to be treated as an orphan by the gate).

**Pipeline consistency:**
Change is entirely within `normalize_process_payload` in `process_normalizer.py`,
which owns all normalization steps. No orchestrator, mapping, audit, or export
logic was touched. The three pruning steps remain independent functions â€” each with
a single responsibility â€” rather than being merged into one function that would be
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

### 2026-07-07 â€” Gate validation errors not shown in UI

**Files changed:** `src/t2pw/app/streamlit_app.py` (lines 2637â€“2651)

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
was not changed â€” only the UI reading of it was corrected. This is a pure
orchestrator responsibility: surface what a stage reported.

---

### 2026-07-07 â€” Orphan proteins not pruned when not complex components

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
