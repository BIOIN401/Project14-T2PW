# SCHEMA_CHECK — pass/fail

Constraint, as tightened by the product owner: **no variant may change the JSON output
shape at all.** A renamed, added, removed or re-nested key is a reject, not something to
justify. Only what the model is told to put *into* the shape may change.

Result: **all ten variants PASS.** Nothing required fixing — the eight derived variants
were built by programmatic insertion into the control, so the schema block could not drift
by construction (§3).

---

## 1. The shape the CODE parses, with evidence

Derived from the consumers, not from the prompt's claims. "Dropped by cleaning" means
`clean_stage_one` deletes the key before any later stage sees it, so the prompt may demand
it but nothing downstream reads it (see AUDIT.md D1/D2).

### Top level

| Key | Read by | Note |
|---|---|---|
| `entities` | `pipeline.py:2509`, contract `stage_contracts.py:606` | required; always emitted even if empty (`pipeline.py:2510`) |
| `processes` | `pipeline.py:2520`, contract `:606` | required; always emitted (`:2521`) |
| `biological_states` | `pipeline.py:2512-2514` | kept when non-empty |
| `element_locations` | `pipeline.py:2516-2518` | kept when non-empty |
| `compartments` | `payload_models.py:466` validates it | **no core reader** — AUDIT.md D2 |
| `scope_warning` | `payload_models.py:465`, `schema.py:502` | set by the Case-A gate `(W):27-28` |
| `visualizations` | `payload_models.py:455-460` | prompt orders it kept empty `(W):226-228` |

### `entities` — nine buckets, fixed by a whitelist

`_clean_entities` iterates exactly `cell_types, species, tissues, subcellular_locations,
compounds, element_collections, nucleic_acids, proteins, protein_complexes`
(`pipeline.py:1693-1703`). **Any other bucket is dropped whole** (`:1710-1718`). Five of
the nine become graph nodes: `compounds→compound`, `proteins→protein`,
`protein_complexes→protein_complex`, `nucleic_acids→nucleic_acid`,
`element_collections→element_collection` (`draft_graph.py:269-275`).

Per-row fields: `name` is required (`pipeline.py:1730-1734`, contract
`stage_contracts.py:634`) and **all other keys are copied key-for-key**
(`pipeline.py:1749`) — entity rows are not whitelisted, unlike process rows. `components`,
`cofactors`, `modifications` are pruned to non-blank strings and dropped when empty
(`:1751-1757`). `class`, `confidence`, `provenance` therefore survive on entity rows;
`draft_graph.py:284` reads `class`. `synonyms` would also survive and is read by
`process_normalizer.py:626-633` — the prompt never mentions it (AUDIT.md D11).

### `biological_states`

Whitelist: `name, species, cell_type, tissue, subcellular_location, evidence`
(`pipeline.py:1772`). A row survives only if one of the first five is non-empty
(`:1778`). **`compartment_canonical` is deleted here** although the prompt mandates it
`(W):132, 175-177` and `payload_models.py:272` / `process_normalizer.py:3422-3428` /
`pwml/ir.py:312` want it (AUDIT.md D2).

### `element_locations` — four buckets

`compound_locations.compound`, `element_collection_locations.element_collection`,
`nucleic_acid_locations.nucleic_acid`, `protein_locations.protein`
(`pipeline.py:1787-1792`). Per row exactly three keys survive: the entity key (required,
`:1797-1799`), `biological_state`, `evidence` (`:1800-1806`).
`draft_graph._build_compartment_index` reads the same four buckets and needs both the
entity name and `biological_state` to index a compartment (`draft_graph.py:149-166`).

### `processes` — four buckets

`reactions`, `transports`, `reaction_coupled_transports`, `interactions`
(`pipeline.py:2194, 2270, 2323, 2373`). Per-row surviving keys:

**reactions** (`pipeline.py:2229-2265`) — `inputs`, `outputs` (both required non-empty
after splitting/dedup, else the row is dropped `:2213-2228`), `locked_reaction_id`,
`name`, `enzymes`, `biological_state`, `evidence`, `inference`, `scope_membership`,
plus `rag_provenance`/`source_papers`/`rag_confidence`/`provenance_lineage` carriers
(`:1975, 1990`). `modifiers` is **merged into `enzymes`** at `:2239` — the model may emit
either and the cleaner unifies them. `class`, `confidence`, `provenance`, `source_refs`,
`spontaneous` are **dropped**.

**actor rows** inside `enzymes`/`modifiers`/`transporters` (`_clean_enzymes`,
`pipeline.py:1885-1937`) — accepts `{"protein": ...}`, `{"protein_complex": ...}`, or
`{"entity": ..., "entity_type": "protein"|"protein_complex"}` (`:1894-1907`); emits
`protein` **or** `protein_complex` plus `evidence`, `provenance`, `confidence`,
`inference`. `entity_type` and `role` are **dropped deliberately** (`:1862-1866`), and
`role` is re-defaulted to `catalyst` downstream (`draft_graph.py:336`,
`process_normalizer.py:3047`). `entity_type` in
`{compound, cofactor, ion, small_molecule, metabolite}` drops the row (`:1871, 1891-1893`).

**transports** (`pipeline.py:2275-2312`) — `name`, `cargo`, `from_biological_state`,
`to_biological_state`, `transporters`, `elements_with_states`, `evidence`, `inference`
+ carriers. Row kept only if one of `cargo`/`transporters`/`elements_with_states` is
present (`:2313`). `transporters: []` is valid and must not suppress the row
(`(W):403`, example `(W):507-524`).

**reaction_coupled_transports** (`pipeline.py:2333-2359`) — `name`, `reaction`,
`transport`, `enzymes`, `elements_with_states`, `evidence`, `inference` + carriers; kept
if `reaction`/`transport`/`elements_with_states` present (`:2360`). Note the cleaner reads
`enzymes` only here, **not** `modifiers` (`:2343`), unlike reactions.

**elements_with_states** rows: `element` (required), `side`, `biological_state`,
`evidence` (`pipeline.py:1941-1959`).

**interactions** (`pipeline.py:2373-2405`) — `entity_1` and `entity_2` are **both
required**, else the row is dropped (`:2378-2382`) and the contract rejects it
(`stage_contracts.py:732-749`). Surviving: `entity_1`, `entity_2`, `name` (synthesised
from the endpoints and relationship when absent, `:2386-2392`), `relationship`,
`biological_state`, `evidence`, `inference` + carriers. `class`, `confidence`,
`provenance`, `source_refs` are **dropped** — so `relationship` is the sole surviving
carrier of interaction type for `pwml/ir.py:2291` → `pwml/writer.py:1419`.

### Enumerated value sets the code or prompt fixes

| Set | Values | Evidence |
|---|---|---|
| `interactions[].relationship` / `class` | `activation \| inhibition \| binding \| interaction` | prompt `(W):114`; export reads it at `pwml/ir.py:2291` |
| reaction `class` | `biochemical_reaction \| transport_reaction \| interaction \| subpathway \| activation \| inhibition \| binding` | prompt `(W):113` (dropped by cleaning) |
| compound `class` | `compound \| cofactor \| ion \| element_collection` | prompt `(W):111` |
| protein `class` | `protein \| transporter` | prompt `(W):112` |
| modifier `role` | `catalyst \| transporter \| activator \| inhibitor` | prompt `(W):89, 101, 304`. The normalizer's export-role set is `{"", catalyst, enzyme, activator, inhibitor}` (`process_normalizer.py:2751`) — no `transporter`, because transporters sit on transport rows, and it also tolerates `""`/`enzyme` |
| modifier `entity_type` | `protein \| protein_complex` | prompt `(W):95, 303`; `pipeline.py:1870` |
| `provenance` | `extracted \| enriched \| inferred` | prompt `(W):121`; `stage_one_boundary.py:212` matches `inferred`/`enriched` |
| `scope_membership` | `core \| anaplerotic \| cataplerotic \| auxiliary \| out_of_scope` | prompt `(W):313`; `filter_out_of_scope_reactions` casefolds `out_of_scope` (`pipeline.py:356`) |
| compartment `canonical_name` | 11 terms, `cytosol`…`endosome` | prompt `(W):171-172` |

`SAME_AS` is **not** in the interaction set. It is consumed downstream
(`process_normalizer.py:1949-1951`) and instructed at Stage 2
(`pwml_infer_system.txt:144-153`), so it is a legitimate Stage-2 value and an
out-of-contract Stage-1 value (AUDIT.md D4).

---

## 2. Per-variant verdict

Method key — every variant was checked by all four tests in §3:

* **T1** schema block byte-identical to v01 (line endings normalised)
* **T2** schema block parses as JSON; full key tree (201 paths) equals v01's, in order
* **T3** all ten enumerated value sets present and equal to v01's
* **T4** no `"key":` token anywhere in the file that v01 does not also carry

| Variant | T1 | T2 | T3 | T4 | Shape verdict | What it changes instead |
|---|---|---|---|---|---|---|
| `v01_committed_baseline` | PASS | PASS | PASS | PASS | **PASS** (is the reference) | control |
| `v02_working_tree_resolution` | PASS | PASS | PASS | PASS | **PASS** | control |
| `v03_resolution_precedence` | PASS | PASS | PASS | PASS | **PASS** | which rows to suppress as restatements |
| `v04_alias_rows_banned` | PASS | PASS | PASS | PASS | **PASS** | forbids a value already out of contract |
| `v05_one_canonical_name` | PASS | PASS | PASS | PASS | **PASS** | which string fills existing name fields |
| `v06_transcription_as_reaction` | PASS | PASS | PASS | PASS | **PASS** | what may fill `inputs`/`outputs` |
| `v07_transcription_excluded` | PASS | PASS | PASS | PASS | **PASS** | what may **not** fill `inputs`/`outputs` |
| `v08_regulatory_example` | PASS | PASS | PASS | PASS | **PASS** | adds an example using only existing keys |
| `v09_vocabulary_by_construction` | PASS | PASS | PASS | PASS | **PASS** | how an existing field's value is chosen |
| `v10_pre_output_self_check` | PASS | PASS | PASS | PASS | **PASS** | a verification pass over existing rules |

Byte sizes: v01 34,733 / v02 37,044 / v03 38,443 / v04 37,694 / v05 38,229 / v06 38,405 /
v07 38,334 / v08 39,249 / v09 37,891 / v10 39,285. Largest is v10 at +13.1% over v01.

### The two variants that touch a field's *values*, and why both still pass

* **v09** rewrites `"relationship": "inhibits"` → `"inhibition"` and `"activates"` →
  `"activation"` in Example 2 (`(W):475, 480`). This moves two example values **into** the
  enumerated set of `(W):114`; it removes no value and adds none. T3 confirms all ten value
  sets are unchanged. (AUDIT.md D3: those two values were always out of contract.)
* **v04** forbids `SAME_AS` and the other identity spellings. `SAME_AS` was never in the
  Stage-1 enumeration, so this removes nothing the prompt permitted. It is a narrowing of
  *behaviour*, not of the schema, and it leaves `pwml_infer_system.txt` untouched so
  Stage-2 alias bridging still works.

No variant adds, renames, removes or re-nests a key. No variant mentions `synonyms`,
which was the one tempting addition (AUDIT.md D11).

---

## 3. How the check was run

Two scripts, both outside the repo, both read-only against `src/`:

* `../build_variants.py` — reads the working-tree prompt as **bytes**, asserts it is pure
  CRLF, and derives v03-v10 by inserting after (or replacing) a byte anchor asserted to
  occur exactly once. The `OUTPUT JSON SCHEMA` block is never an anchor and never inside a
  replacement, so no derived variant can alter it. v01 is `git show HEAD:…` and v02 is a
  byte copy of the working tree; both were verified by sha256 against their sources.
* `../check_schema.py` — runs T1-T4 above over every `v*.txt` in this directory.

Reproduce:

```
python -I <scratch>/check_schema.py <scratch>/prompt_variants
```

Output at the time of writing:

```
baseline: v01_committed_baseline.txt  (201 schema key paths)

variant                            T1 bytes  T2 keys  T3 enums  T4 stray
v01_committed_baseline.txt         PASS      PASS     PASS      PASS
v02_working_tree_resolution.txt    PASS      PASS     PASS      PASS
v03_resolution_precedence.txt      PASS      PASS     PASS      PASS
v04_alias_rows_banned.txt          PASS      PASS     PASS      PASS
v05_one_canonical_name.txt         PASS      PASS     PASS      PASS
v06_transcription_as_reaction.txt  PASS      PASS     PASS      PASS
v07_transcription_excluded.txt     PASS      PASS     PASS      PASS
v08_regulatory_example.txt         PASS      PASS     PASS      PASS
v09_vocabulary_by_construction.txt PASS      PASS     PASS      PASS
v10_pre_output_self_check.txt      PASS      PASS     PASS      PASS

ALL VARIANTS SHAPE-IDENTICAL TO BASELINE
```

Isolation was verified separately with `diff -u` against v02: v03-v08 and v10 are each
**one hunk, insertions only**; v09 is two hunks (the rule it replaces, and the two example
values) and is the only variant that deletes any baseline text.

A note on T4: the test is "no token v01 lacks", not "no token at all", because v01 itself
contains one prose token that looks like a key —
`context_type is "enzyme_mechanism" or "partial_biosynthetic_step":` at `(W):35`. An
earlier formulation of the test flagged the baseline, which is how that was found.

---

## 4. Verification that site-level extraction needs no schema change

Checked rather than assumed, per the coordinator's instruction. A DNA site is a
`nucleic_acids` entity, and that bucket is fully plumbed:

| Stage | Evidence |
|---|---|
| prompt already declares it | `(W):262` schema row; `(W):233` definition ("nucleic_acid: RNAs/DNAs if explicitly named as such"); `(W):218-220` ("Each such named element is its own nucleic_acids entity") |
| survives cleaning | `pipeline.py:1701` in the `_clean_entities` whitelist |
| has a location bucket | `nucleic_acid_locations`, `pipeline.py:1790`; schema `(W):286-288` |
| resolves as a participant | `filter_unresolvable_reactions` loads `nucleic_acids` into `entity_names` (`pipeline.py:849`) |
| becomes a graph node | `draft_graph.py:273` maps `nucleic_acids → nucleic_acid` |
| validates at runtime | `payload_models.py:260` `NucleicAcidModel` |
| has a visualization slot | `vacuous_nucleic_acid_visualizations`, `(W):389` |

Confirmed empirically: the v02 run produced eight `nucleic_acid` entity rows for DNA
elements (`tmp/draft_graph.json`) with no schema violation and no contract error
(`tmp/extraction_diagnostics/run-c5e6c739092d/extraction_boundary_report.json` records
`outcome: ok`, `stage_one_boundary_settled`, and `cleaning_report.json` records
`discarded_total: 0`).

**One caveat, and it is a code defect rather than a schema one.** `build_draft_graph`
resolves a participant's kind from `proteins`, `protein_complexes` and `compounds` only
(`draft_graph.py:221-229`); `nucleic_acids` and `element_collections` are missing, so
`_resolve_actor_kind` falls through to its `return "compound"` default
(`draft_graph.py:131`) and a **duplicate** node is minted for every DNA site referenced by
an interaction. Measured: 8 orphan `nucleic_acid` nodes and 8 `normalized label collision`
flags in `tmp/qa_report.json`. The payload is correct; the graph built from it is not. See
AUDIT.md D7 — this inflates the orphan/collision counts for exactly the variants that do
site-level extraction best, and those counts must not be scored against them.
