# PathWhiz Requirements & Stage Stability Findings

Investigation date: 2026-07-10. This document is the output of a deliberate audit
requested because the pipeline has been "going in circles" — a fix in one stage
repeatedly resurfaces as a bug in another. It has two goals:

1. Write down, once, in one place, exactly what PathWhiz requires from an
   imported file — ground-truthed against the Ruby importer source
   (`tools/ruby/pwml_parser.rb`, `tools/ruby/sbml_parser.rb`) and real exported
   files (`reference/*.pwml`, `reference/*.sbml`), not against our own
   assumptions.
2. Name the specific places where stage boundaries (as defined in
   `docs/pipeline.md`) are currently violated in the running code, so those
   violations — not new symptoms — are what get fixed next.

This document does not replace `docs/pipeline.md` (the stage contract
reference) or `docs/change_log.md` (the append-only fix log). It sits between
them: a snapshot analysis of *why* stage boundaries have been drifting.

Four research passes fed this document: (1) PathWhiz/PWML/SBML import format
ground-truth, (2) Stage 3 (Normalize) code audit, (3) coordinate/layout stage
audit, (4) `change_log.md` churn-pattern mining. Findings below are organized
by topic, not by which pass produced them, since several passes independently
converged on the same root causes (noted where that happens — convergence is
itself signal).

---

## 1. What PathWhiz actually requires (ground truth)

Ground-truthed against `tools/ruby/pwml_parser.rb` (the actual PathWhiz-side
PWML import logic) and `tools/ruby/sbml_parser.rb`, cross-checked against real
files in `reference/`. Where our code's assumptions diverge from this ground
truth, that is flagged explicitly.

### 1.1 Required vs optional fields, by entity type

**Compound** (`entities.compounds`)
- `name` — **required**. Missing → hard error `compound_missing_name` (`ir.py`
  validator).
- DB identity (ChEBI/HMDB/KEGG/PubChem/PathBank id) — **effectively optional**.
  `ir.py` treats an unresolved compound as `error` under `strict_db`, but
  `writer.py::is_non_blocking_pwml_ir_error` explicitly whitelists this exact
  error so it does **not** block export. Compounds may be genuinely novel to
  PathWhiz.
- `pwc-id`, `short-name` — forced optional by
  `writer.py::_make_compound_identity_fields_optional` so Rails can allocate
  values for novel compounds.
- `moldb-*`, `foodb-id`, `synonyms`, `element-states` — **cosmetic/inert**.
  Ruby's `to_hash_shallow` drops blank leaf elements and skips any container
  child with grandchildren. Populating these fields on our end does nothing on
  import; they are not worth engineering effort.
- **Identity/dedup ground truth:** PathWhiz does `Compound.find_by(attribute_hash)`
  using every non-blank scalar field we emit, ANDed together — not by our
  document-local `<id>` (Ruby explicitly skips `node.name == 'id'` when
  building the match hash). This means: for a "matched" compound, every
  non-blank field we emit must be an exact match to the real DB row, or
  `find_by` misses and PathWhiz silently creates a duplicate compound instead
  of reusing the existing one.

**Protein** (`entities.proteins`)
- `name` — required.
- Species — **required** (`protein_missing_species`).
- UniProt **or** DrugBank id — **required, one of the two**
  (`protein_missing_external_identity`). Gene name, EC number, sequence,
  description are useful metadata but never substitute for this. (Confirmed
  independently: change_log.md logs a direct PathWhiz UI observation — "The
  `New Protein` form requires `Name`, `Species`, and either `UniProt ID` or
  `DrugBank ID`.")
- **Serialization bug found:** every protein's `species-id` is written from
  `self._ir_pathway_species_id` — the id of the *first* species record in the
  whole pathway, not the protein's own resolved species. The species-presence
  gate validates one thing; the file emits another. Harmless for genuinely
  single-organism pathways (which Stage 1's scoping rules now push toward),
  silently wrong the moment a pathway is legitimately multi-species.

**Protein complex** (`entities.protein_complexes`)
- `name`, species — required.
- ≥1 component with a resolved protein + stoichiometry — required, **but only
  enforced as a hard error when the complex is recognized as "generated"**
  (see contradiction #3 below); otherwise it's a warning.
- A PathBank/PathWhiz complex ID is **not required** for a generated complex,
  provided every component protein has species + UniProt/DrugBank
  (`docs/pipeline.md` Stage 2/6 rules — confirmed still accurate).
- **Modeling gap:** real PathWhiz complexes can have a `protein-complex-cofactor`
  (a bound compound, not a protein). Our IR schema has no representation for
  this at all — not a bug, but a ceiling on what a generated complex can
  express.

**Reaction** (`processes.reactions`)
- ≥1 input, ≥1 output, a biological_state — required.
- Enzyme actor **must be `protein_complex`, never a bare `protein`** — this is
  enforced, and bare-protein enzymes get auto-wrapped into a generated complex
  *only if* the source protein already has species + UniProt/DrugBank.
  Otherwise: hard error, protein stays bare, then fails the type check anyway.
- Direction: Ruby and real reference files require literal capitalized English
  (`Right`/`Left`/`Both`), not symbols. The live IR export path converts this
  correctly; a second, dead legacy writer path (`writer.py`'s non-IR builder)
  hardcodes `"Right"` unconditionally and ignores real direction — currently
  unreachable from production, but a landmine if anyone re-wires it.
- **`spontaneous` — no live field anywhere.** This is the single highest-value
  finding in this document; see contradiction #1.
- **`currency` — always hardcoded `False`.** A real currency-compound
  classifier exists (curated ATP/NADH/CoA name/KEGG/ChEBI sets) and is fully
  wired into internal layout deduplication, but never into the actual
  `<currency>` value written to the file. If PathWhiz's UI uses this flag to
  hide cofactors, every pathway we generate currently shows all cofactors
  regardless of the classification work already done.

**Transport** (`processes.transports`)
- 1–3 transport elements (cargo) — required range, both bounds enforced.
- Each element's left/right biological_state — required.
- Direction per transport element — **not modeled in the live IR path at all**
  (only in the dead legacy path, hardcoded `"Right"`). Real PWML has a
  `direction` child on `transport-element`. Open gap.

**Interaction** (`processes.interactions`)
- `interaction_type` non-empty, exactly one resolvable `left` and `right` —
  required. Build code defaults empty type to the literal string
  `"interaction"`, which passes the non-empty check while being semantically
  meaningless.

**Biological state / species / subcellular location**
- `species_key`, `subcellular_location_key` — required on every biological
  state.
- At least one biological state must exist pathway-wide; if none, one is
  auto-generated with a warning.
- Two species/location names have hand-coded silent-default behavior
  (`SPECIES_CREATE_DEFAULTS` / `SUBCELLULAR_LOCATION_CREATE_DEFAULTS` in
  `ir.py`) for one specific species string and `"cell"` — narrow, easy to
  silently miss for anything else.

### 1.2 What must be deterministic vs what may come from the LLM

**Must be code-computed, never LLM-invented:**
- Document-local `<id>` values (cross-reference bookkeeping only — Ruby never
  uses them for entity identity/matching).
- Direction-symbol → text conversion.
- Layout/coordinate placeholders (see §3 — entirely separate finding).
- Complex-wrapper generation for single-protein enzymes (gated deterministically
  on the source protein already having species + UniProt/DrugBank).
- The XML structural signature used to decide which fields get force-emitted
  (see contradiction #2 — this is deterministic but not well-founded).

**Correctly left to the LLM / upstream mapping stages, never invented by
export code:**
- Entity names, species/organism assignment, UniProt/DrugBank accession
  choice — owned by Stage 2/6 mapping, not export. Export code only validates
  presence, never invents an identity, and fails closed if a reference can't
  be resolved.
- Reaction participants, role assignment (enzyme/catalyst/transporter),
  compartment text — all upstream; export only resolves against declared
  entities.
- `spontaneous` status — *should* come from Stage 1 extraction, but currently
  has no field anywhere in the schema path to carry it through (see below).

### 1.3 Concrete contradictions found (the actual "fix one thing, break another" mechanisms)

1. **`spontaneous` has no live channel end-to-end, but QA hard-requires it.**
   `build_pwml_ir` never copies a `spontaneous` key from the payload even if
   present; the writer always serializes `False`. Meanwhile `qa.py` treats any
   enzyme-less reaction with `spontaneous != true` as a hard QA error that
   flips overall export `ok` to `False`. **Net effect: a genuinely
   non-enzymatic reaction, correctly extracted from the source paper, cannot
   currently pass export QA at all.** The only ways to "fix" it today are
   forcing an incorrect enzyme onto it (which is exactly the kind of fix that
   later gets flagged as a hallucinated actor and reverted) or ignoring the QA
   failure. This is very likely a direct contributor to the circular
   enzyme-attachment fixes documented in change_log.md.

2. **The "required fields" schema is sampled from whichever reference file is
   the default `--ref`, not an independently authored spec.**
   `discover_structure_signature` samples 3 items from
   `reference/PW000001.pwml` (the hardcoded default) to decide which fields
   get force-emitted per entity type. Swapping the default reference file
   would silently change force-emitted fields for every entity type, with no
   tie to actual PathWhiz/Rails validation rules. This is a standing risk, not
   yet a triggered bug.

3. **Generated-complex strictness is coupled to fragile string matching, and
   this exact issue independently recurred at least twice.** A protein complex
   is only held to the hard species/UniProt requirement if
   `_is_generated_complex_row` recognizes one of three hardcoded marker
   strings (`generation_reason`, `chosen_rule`, or
   `mapping_meta.resolution.order_step`). If Stage 2/6 (`map_ids.py`) writes a
   different marker string than these three, the hard check silently downgrades
   to a warning and an invalid complex can pass. **This is the same root cause
   independently found by the change-log churn analysis (§4, Incident C) and
   the Stage 3 audit (§2, boundary violation 4/5) — three separate research
   passes converged on it from different angles.** It is the single most
   expensive recurring bug class in the project's history: 8 sequential
   change-log entries chased facets of it across `map_ids.py`,
   `process_normalizer.py`, `streamlit_app.py`, and `pwml/ir.py` before each
   symptom, in turn, relocated to the next stage.

4. **Compound DB-resolution failure is simultaneously documented as a hard
   abort and coded as explicitly non-blocking.** `docs/pipeline.md` states the
   pre-export contract is "the only hard abort." It isn't, for compounds
   specifically — `writer.py` explicitly whitelists this failure as
   non-blocking. Anyone reading the IR validator's error list in isolation
   would reasonably conclude otherwise.

5. **A per-entity-type "strict DB identity required" flag exists in code and
   is dead.** Every entry in `ir.py`'s `entity_specs` hardcodes this flag to
   `False`, so the corresponding check can never fire regardless of
   `strict_db`. This looks like a half-finished generic feature superseded by
   the protein/protein_complex-specific checks but never removed. Flipping it
   on for any entity type later would introduce a second, redundant,
   potentially-conflicting gate alongside the existing one.

6. **`legacy_validate.py` (both the package version and the `src/validate.py`
   shim) are empty dead files**, not imported anywhere. Any prior
   documentation or memory of "legacy_validate.py logic" is stale.

7. **SBML is not an independent create-path — it can only sync an already-existing
   pathway.** `sbml_parser.rb`'s own top comment says it "will only parse
   correctly if all the data elements are already in PathWhiz." Every
   compartment/species/reaction referenced must already exist in PathWhiz's DB
   with matching id **and** exact name, or the entire import aborts with zero
   partial success. `docs/pipeline.md` calls SBML "legacy" but doesn't flag
   this fundamental difference (create-capable PWML vs. sync-only SBML), and
   `AGENT_INSTRUCTIONS.md`'s SBML section is written as if SBML export were a
   standalone deliverable for new pathways. It is not, for anything that
   doesn't already exist in PathWhiz.

8. **`AGENT_INSTRUCTIONS.md`'s documented compartment annotation is missing a
   required attribute** (`pathwhiz:compartment_id`, shown as absent in the doc
   example). Ground truth requires both `_id` and `_type`; without `_id`,
   `sbml_parser.rb` fails that compartment and the entire SBML import aborts.
   All three real reference `.sbml` files include it — the doc example is
   simply stale, not an alternate valid form.

9. **Two sequential, non-identical hard-abort surfaces exist** —
   `validate_required_pwml_contract` (payload-level, pre-IR) and
   `validate_pwml_ir` (post-build, IR-shape — this is where the transport
   1–3-element count and complex-visualization-hidden checks actually live).
   Passing the first does not guarantee passing the second. `docs/pipeline.md`
   describes "the only hard abort" as one thing; in the running code it is
   two, with different blind spots.

10. **`currency` classification exists and is unused** (see §1.1 Reaction).

### 1.4 Open questions that need runtime/live verification, not just reading

- Whether a real production PathBank compound id can exceed the fallback
  document-local id counter's starting offset (20000), which would risk a
  document-local `<id>` collision between a matched and a generated entity of
  the same type.
- Whether `run_pwml_pipeline_export` returning `ok: False` actually prevents
  the `.pwml` file from being offered for upload in the Streamlit UI, or
  whether the file is written to disk regardless and could be uploaded anyway
  (the write-to-disk call executes before the `ok` computation).
- Whether PathWhiz's Rails-side model validations hard-abort the whole import
  on one bad entity, or (as the Ruby importer's local rescue blocks suggest)
  fail only that one entity silently while the rest of the pathway still
  imports with a dangling reference. Only confirmable against the PathWhiz
  Rails source (not in this repo) or a live test import.
- The exact composition of PathWhiz's own `find_existing`/`generate_signature`
  dedup logic for nested objects (protein_complex, reaction, transport,
  interaction, sub_pathway) — Rails-side, opaque from this repo. Determines
  whether semantically-identical-but-freshly-generated records get recognized
  as duplicates of existing PathWhiz records or always create new rows.
- Whether omitting `pathwhiz:species_id`/`compartment_id` when MySQL lookup
  fails (documented as "omitted" in AGENT_INSTRUCTIONS.md) causes a partial or
  total SBML import failure — ground truth suggests total abort, but no
  reference file with a genuinely missing id was available to confirm.

---

## 2. Stage 3 (Normalize) stability findings

Stage 3 (`process_normalizer.py::normalize_process_payload`, 17 documented
steps) is the single most-edited file in the project (see §4 frequency tally)
and is the subject of the most circular-fix history. Key findings:

- **The documented normalize→gate→audit→re-normalize→re-gate loop does not
  match the running code.** `docs/pipeline.md` states each audit round
  re-runs the full 17-step normalize pass. In `streamlit_app.py`, the audit
  loop calls `run_audit` → apply patch → optional gap resolution → a
  *different* QA path (`build_draft_graph`/`generate_qa_report`) → loops back
  to `run_audit` directly. `normalize_process_payload` is only called twice in
  the entire app: once before the audit loop starts, once at final export.
  Audit patches are applied and re-audited without the 17 deterministic
  cleanup steps ever re-running on them until export time. This is a
  significant, previously undocumented divergence between spec and behavior.
- **Stage 3 calls directly into Stage 2's module.** `normalize_composites`'s
  `_is_protein_like` helper imports and calls
  `t2pw.mapping.map_ids.route_entity_for_mapping` live during normalization —
  a documented-as-independent stage reaching back into the mapping stage.
- **Stage 3's hard gate duplicates Stage 8's pre-export contract, in a second,
  unsynced implementation.** The species/UniProt/DrugBank checks in
  `run_strict_post_normalization_gates` were added specifically because the
  same checks existed *only* in `ir.py`'s Stage 8 gate, so Stage 4 (audit)
  never got a chance to see and repair them. The two implementations do not
  share code (`process_normalizer.py`'s version explicitly does not import
  from `ir.py`) and will re-diverge the moment either one's exact field
  semantics change.
- **Stage 4 (audit) independently re-implements payload-validity checks that
  already exist in Stage 3/`qa_graph.py`** (`audit_json_llm.py::_deterministic_audit`
  has its own composite-name detection and registry-reference checks, using
  different regexes than `process_normalizer.py`'s equivalents) — a third
  independent implementation of overlapping "is this payload valid" logic.
- **Steps 3, 7, 8, 9 of the 17 (`rewrite_reactions_to_complex_states`,
  `attach_transporters_from_evidence`, `attach_enzymes_from_reaction_evidence`,
  `promote_interaction_enzymes`) are evidence-text NLP inference, not
  deterministic cleanup** — self-documented in their own docstrings as
  compensating for LLM output quality. This is functionally Stage 4's stated
  job, done unconditionally in Stage 3 with no gate/report mechanism to flag
  when an inference guessed wrong.
- **Stage 3 assumes a `mapping_meta` nested shape from Stage 2 that Stage 2's
  own structural contract does not guarantee.** `stage_contracts.validate_post_mapping`
  only guarantees the `mapping_meta` *key* exists, not its nested shape.
  Several Stage 3 identity checks read specific nested paths
  (`mapping_meta.species_resolution.confidence`, etc.) and silently treat a
  different shape as "absent" rather than failing loud.
- **Export-time gate enforcement is inconsistent between the two production
  entry points.** The Streamlit export path checks the Stage 3 gate report and
  aborts on failure. The CLI path (`writer.py::run_pwml_pipeline_export`)
  calls the normalizer but never inspects its gate result — a gate failure
  there is silently ignored. This is exactly the class of bug one change-log
  entry already found and fixed for the Streamlit path, evidently never
  applied to the CLI path.
- **Dead code hiding inside the same module:** `normalize_draft_graph`
  (~270 lines) operates on an entirely different data structure, is never
  called from any pipeline code, has a `normalize_*` name that implies it's
  part of the 17-step sequence, and has zero test coverage.
- **Test coverage gaps:** `apply_biochemical_aliases`, `backfill_reaction_compartments`,
  and `drop_process_orphan_proteins` have no dedicated test coverage at all.
  The largest integration-style test (`test_thyroid_normalization_and_dedupe`)
  uses a hand-rolled helper that calls only 13 of the 17 real steps, omitting
  4 — meaning the most comprehensive existing test does not exercise the
  actual production step sequence or order.
- **Pathway-specific data is hardcoded into the general-purpose normalizer**
  (a thyroid-specific forbidden-complex string, a thyroid-specific default
  scaffold name) — a modularity smell independent of correctness.

---

## 3. Coordinate / layout stage findings

There is no single "coordinate stage" — layout is cross-cutting logic spread
across three independently-implemented algorithms, none of which talk to each
other, plus two disconnected QA-preview renderers.

- **Three layout algorithms for the same output:**
  1. `ir.py::build_pwml_ir` computes a fixed-grid layout first — and then, for
     anything connected to a reaction, that layout is **silently discarded and
     overwritten** by:
  2. `writer.py::_populate_sections_from_ir` (the only path production actually
     uses) — a compartment-region + serpentine-placement algorithm, with
     `substrate_gap`/`product_gap` magic constants duplicated verbatim between
     it and a third, dead sibling method (`_build_locations_and_visualizations`,
     kept alive only by tests, unreachable from either production entry
     point).
  3. `sbml/add_pathwhiz_layout.py` — a completely separate, more sophisticated
     (topological + cycle-aware) algorithm for the legacy SBML path only.
- **No collision/overlap avoidance anywhere.** All three algorithms are purely
  additive fixed-offset arithmetic with no post-placement check. Nothing
  prevents two entities landing on identical coordinates if a reaction has
  unusually many participants.
- **Compartment classification is fuzzy substring matching** against ~13
  hardcoded strings; anything unmatched silently falls into an "unrecognized"
  catch-all band with no warning.
- **Canvas width is never auto-scaled** (only height grows with content) —
  a wide pathway can overflow the canvas horizontally with no warning.
- **The PWML-specific QA renderer (`pwml/render.py`) is completely disconnected
  from the live pipeline** — reachable only via its own standalone CLI shim,
  not called by `streamlit_app.py` or any pipeline module. The legacy SBML
  path, by contrast, does get an automatic rendered preview in the UI. There is
  currently no automatic visual sanity check for the primary PWML export path.
- **No geometry QA checks exist anywhere** (`qa.py` is purely structural — no
  bounds check, no overlap check, no off-canvas check).
- **Ground truth on PathWhiz's side (`pwml_parser.rb`) treats coordinates as
  entirely non-validated and per-element fail-soft** — a bad/missing
  coordinate on one node causes that one visual element to silently drop from
  the diagram, not a failed import. This means layout bugs are invisible until
  someone opens the imported pathway in PathWhiz itself and looks at it.
- **This area has essentially no change-log history** — a single unrelated
  hit when grepped for layout/coordinate/position/render/geometry terms across
  880 lines. Unlike every other stage, layout code was written once and never
  revisited, which independently corroborates the maintainer's sense that it
  "will need work" — it hasn't been hardened by the same iterative fix cycle
  the rest of the pipeline has been through (for better and worse).
- The LLM is explicitly and correctly instructed never to invent diagram
  coordinates (`pwml_infer_system.txt`: "No diagram/layout coordinate
  inventions.") — this boundary is clean; the fragility is entirely on the
  deterministic-code side.

---

## 4. Change-log churn: where the "going in circles" pattern actually lives

Mined from all 880 lines of `docs/change_log.md`. Full detail preserved for
each incident; only the shape is summarized here.

### 4.1 Frequency tally — where instability concentrates

| File | Fix entries + open issues |
|---|---|
| `src/t2pw/pipeline/process_normalizer.py` (Stage 3) | 10 fix entries + 1 open issue — by far the most-edited file |
| `src/t2pw/mapping/map_ids.py` (Stage 2/6) | 3 fix entries + 2 open issues |
| `src/t2pw/pwml/ir.py` (Stage 8) | 2 fix entries + 2 open issues |
| `src/t2pw/app/streamlit_app.py` (orchestrator) | 2 fix entries + 1 open issue |
| `src/t2pw/curation/audit_json_llm.py` (Stage 4) | 1 fix entry + 1 open issue |
| `llm/prompts/pwml_system.txt` (Stage 1) | 2 |
| `stage_contracts.py`, `schema.py`, `qa_graph.py`, `apply_audit_patch.py` | 1 each |
| `mapping/enrich_entities.py` (Stage 7) | 0 fixes — still an open product decision, never wired in |

Within `process_normalizer.py`, the single most-touched function across
history is `run_strict_post_normalization_gates`, followed by
`canonicalize_same_as_aliases`.

**Note:** the changelog itself is internally inconsistent about stage
numbering — `map_ids.py` entries variously call themselves "Stage 2 and Stage
6," while a separate open issue calls enrichment "Stage 6," conflicting with
`docs/pipeline.md`'s numbering (Enrich is Stage 7). Worth reconciling next time
the changelog's template is touched.

### 4.2 The dominant circular-fix chain (8 entries, one root cause)

The single most expensive recurring bug in the project's history is the
**generated single-protein `protein_complex` wrapper** (e.g. "NdmA" →
"NdmA complex"). It was fixed, piecemeal, **eight times**, relocating between
stages each time a fix closed one symptom:

1. Species/UniProt checks added to Stage 3's gate (they only existed in Stage
   8, so Stage 4/audit never got to repair them).
2. "complex" stripped from UniProt name-query variants (a consequence of #1's
   new gate now failing on `* complex`-suffixed names).
3. **Open issue:** wrapper creation in Stage 2/6 (`map_ids.py`) still leaks
   into `entities.proteins` and bypasses Stage 3's gate entirely because the
   orchestrator doesn't stop export on a Stage 3 gate failure.
4. Fix: the specific leak mechanism in `canonicalize_same_as_aliases`
   (`_ensure_protein` never checked `entities.protein_complexes` membership).
5. Fix (logged **twice**, verbatim, under two separate `## Fixed` headings —
   a duplication in the changelog's own record-keeping): once the leak stops,
   the real protein now shows degree-0 in the connectivity graph and gets
   wrongly flagged as an orphan.
6. Fix: generated wrapper components found missing `stoichiometry` at Stage 8.
7. Fix: traced back to the same function from #4 — it had been flattening
   `protein_complexes[].components` to plain strings, discarding
   `stoichiometry`/`mapped_ids`/`uniprot` in the process.
8. Fix: the same symptom class ("valid protein incorrectly reported as missing
   species/UniProt") reappears one layer up, in Stage 8's IR builder's
   last-resort wrapper logic, which loses `species` because it runs after
   entity rows are already converted to IR records.

This is the same subsystem the ground-truth PathWhiz audit (§1.3, item 3) and
the Stage 3 code audit (§2) independently flagged from static reading alone —
three separate investigations converging on one root cause. **This is the
highest-priority target for the "bulletproofing" the user asked for**, because
fixing it at its actual source (see §5) should collapse 8 entries' worth of
recurring work into one.

### 4.3 Other confirmed circular-fix instances

- **Enzyme actor schema fixed on the read side, not the write side, same day.**
  A fix taught the normalizer/QA-graph to read the canonical `entity` field
  first — but the function that *writes* actor rows never migrated enzymes to
  that schema (only modifiers got migrated), so the read-side fix was
  ineffective for the exact field it was meant to help.
- **`drop_process_orphan_proteins` was recorded as "wired in" when it wasn't.**
  One entry states the function was added and wired into the pipeline
  sequence. A later entry discovers and states explicitly: "The change log
  stated it was wired in, but the code did not reflect this." This is a
  documented instance of the changelog's own record being wrong — a landmine
  for anyone trusting it without checking the code.
- **Best-effort UniProt fallback flagged as its own future bug the same day it
  landed.** The fix (accept the top UniProt candidate even below the
  confidence threshold) is immediately re-opened in the very next section of
  the same changelog as a "temporary workaround... the assigned accession may
  be incorrect," pending a BLAST-based fix that the pipeline can't yet support
  because it doesn't carry protein sequences.
- **Species/organism scoping rule added, then found self-contradictory the
  same day.** A rule was added to the extraction prompt; a later same-day
  entry found it directly contradicted a pre-existing rule elsewhere in the
  same prompt file, and that it scoped organism but never made pathway
  selection an explicit first decision.

### 4.4 Root-cause categories, as the changelog itself states them

- **Stage boundary violation / logic in wrong module** — 6+ entries (the most
  common stated cause).
- **Field-name / data-structure mismatch** — 6 entries.
- **LLM non-determinism / extraction limitation** — 4 entries.
- **Infrastructure/configuration** (explicitly *not* code — DB or API
  unreachable) — 3 entries, correctly not chased as code bugs.

---

## 5. What this means for "bulletproofing" each stage

This section states facts and priorities only — not a redesign. Ordered by
where the evidence above says the leverage is highest.

1. **Give `spontaneous` a real end-to-end channel** (schema → Stage 1 prompt →
   Stage 3 gate → `ir.py` → QA). This single missing field is the most
   concrete, well-evidenced gap found, and plausibly explains a chunk of the
   enzyme-attachment churn: reactions that should be allowed to have no
   enzyme currently have no legitimate way to say so.
2. **Fix the generated-complex-wrapper subsystem at its source, not at each
   symptom.** Three independent research passes agree the actual defect is
   that "is this a generated wrapper complex" is decided by fragile,
   uncoordinated string-matching (`_is_generated_complex_row` in `ir.py`,
   duplicated gate logic in `process_normalizer.py`, and creation logic in
   `map_ids.py`) with no shared, single source of truth. A single shared
   classifier function, imported by all three stages instead of
   re-implemented in each, would collapse the 8-entry history in §4.2 into one
   fix.
3. **Make the Stage 3 gate the actual single source of truth it's meant to be,
   instead of one of three parallel implementations.** `process_normalizer.py`,
   `pwml/ir.py`, and `audit_json_llm.py` each have their own copy of
   overlapping validity checks. Collapsing to one shared validator (or at
   minimum, one shared helper module the three stages import) removes the
   re-divergence mechanism entirely, rather than requiring hand-sync forever.
4. **Reconcile the documented audit loop with the actual one**, or fix the
   actual one to match the doc. Right now `docs/pipeline.md` describes a
   re-normalize step inside every audit round that does not exist in
   `streamlit_app.py`. Whichever is correct, the other needs to change — this
   gap means audit-round patches are currently never re-subjected to the 17
   deterministic cleanup steps until final export, which is a plausible source
   of "fixed in audit, broke at export" surprises.
5. **Enforce the Stage 3 gate consistently at both production export entry
   points** (Streamlit already checks it; the CLI path
   (`writer.py::run_pwml_pipeline_export`) does not).
6. **Treat the coordinate/layout stage as unaudited territory**, not stable
   code — it has had essentially zero iteration compared to every other stage,
   has no geometry QA, no shared logic between its three implementations, and
   the primary PWML output path has no wired-in visual preview the way SBML
   does. This matches the user's own instinct that it needs work; the
   concrete first step is wiring `pwml/render.py` into the pipeline the way
   `render_pathwhiz_like.py` already is for SBML, so layout regressions become
   visible before a human has to open PathWhiz itself to find them.
7. **Retire dead code that is actively confusing to future readers/agents**:
   `to_pwml.py`'s standalone converter, `writer.py`'s dead legacy non-IR
   builder (which still hardcodes wrong direction/currency values and is only
   kept alive by tests), `legacy_validate.py` (empty on both the package and
   `src/` shim side), and `normalize_draft_graph` inside
   `process_normalizer.py` (operates on an unrelated data structure, never
   called, named as if part of the 17-step sequence).
8. **Close the two known documentation-vs-code gaps** in `AGENT_INSTRUCTIONS.md`:
   the missing `pathwhiz:compartment_id` in its compartment annotation example,
   and the fact that SBML export is sync-only (requires the pathway to already
   exist in PathWhiz), not an independent create-path the way PWML is.
9. **Resolve confirmed-dead validation surfaces before adding new ones**: the
   per-entity-type `strict_required` flag in `ir.py::entity_specs` is
   hardcoded `False` everywhere and can never fire — either wire it up
   intentionally or remove it so it stops looking like a lever that does
   something.

---

## 6. Provenance

Findings above were produced by four parallel research passes (full detail
retained in each pass's own report, not reproduced verbatim here to keep this
document navigable):
- PathWhiz/PWML/SBML import format ground-truth (read: `ir.py`, `writer.py`,
  `validate.py`, `legacy_validate.py`, `qa.py`, `db_resolver.py`,
  `compound_templates.py`, `to_pwml.py`, both Ruby parsers in full, all 7
  reference `.pwml` files and 3 reference `.sbml` files).
- Stage 3 (`process_normalizer.py`) code audit, all 17 steps plus
  `qa_graph.py`, `stage_contracts.py`, `pipeline.py`'s orchestration, and
  relevant `tests/` coverage.
- Coordinate/layout audit across `pwml/render.py`,
  `sbml/add_pathwhiz_layout.py`, `sbml/render_pathwhiz_like.py`, `ir.py`,
  `writer.py`, both LLM prompts, and the reference files.
- Full read of `docs/change_log.md` (880 lines) for chronological circular-fix
  incidents, frequency tally, and root-cause categorization.
