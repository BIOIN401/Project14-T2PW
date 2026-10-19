# SCHEMA_CHECK_R2 — pass/fail per variant

Constraint: **no variant may change the JSON output shape at all.** The baseline declares
**86 distinct JSON keys**; every variant must declare exactly that set — no additions, no
removals, no renames, no re-nesting — with the same four process buckets, the same per-row
field names and the same enumerated value sets. A schema change is a reject.

**Result: all 16 variants PASS, on six tests.** Nothing needed fixing: the fifteen derived
files were produced by anchored insertion (or, for `r2_11`, three single-line replacements
inside a worked example) into the control, and the `OUTPUT JSON SCHEMA` block is never an
anchor and never inside a replacement, so the schema could not drift by construction. The
tests exist to prove that claim rather than to assert it, and two of them caught real
problems in the *test spec* before they passed (§ 4).

---

## 1. The shape the CODE parses

Derived from the consumers, not from the prompt's claims about itself. Line numbers are the
current working tree.

### Entry point

`run_extraction_pipeline` reads the prompt file verbatim (`pipeline.py:272`) and passes it as
the system prompt to `_run_json_stage` (`:3608`); `_build_extraction_prompt` (`:4311`) builds
the user turn. Nothing rewrites or templates the system prompt, so a variant file is the
entire treatment.

### Top level

| key | read by |
|---|---|
| `entities` | `pipeline.py:2509`; required by the stage contract |
| `processes` | `pipeline.py:2520`; required |
| `biological_states` | `pipeline.py:2512-2514` |
| `element_locations` | `pipeline.py:2516-2518` |
| `compartments` | validated by `payload_models`; no core reader |
| `scope_warning` | set by the Case-A gate at `ctl:21-28` |
| `visualizations` | validated; the prompt orders it kept empty (`ctl:244-246`) |

### `entities` — nine buckets, fixed by a whitelist

`_clean_entities` iterates exactly `cell_types, species, tissues, subcellular_locations,
compounds, element_collections, nucleic_acids, proteins, protein_complexes`
(`pipeline.py:1693-1703`); any other bucket is dropped whole and counted row by row
(`:1706-1718`). `name` is required (`:1730-1734`); all other keys are copied key-for-key
(`:1749`), with `components`, `cofactors`, `modifications` pruned to non-blank strings
(`:1751-1757`).

### `processes` — four buckets, per-row keys that survive

| bucket | keys the cleaner keeps | keys it deletes |
|---|---|---|
| `reactions` (`:2229-2265`) | `inputs`, `outputs`, `locked_reaction_id`, `name`, `enzymes`, `biological_state`, `evidence`, `inference`, `scope_membership`, RAG carriers | `class`, `confidence`, `provenance`, `source_refs`, `spontaneous` |
| `transports` (`:2275-2312`) | `name`, `cargo`, `from_biological_state`, `to_biological_state`, `transporters`, `elements_with_states`, `evidence`, `inference`, carriers | same four |
| `reaction_coupled_transports` (`:2333-2359`) | `name`, `reaction`, `transport`, `enzymes`, `elements_with_states`, `evidence`, `inference`, carriers | same four |
| `interactions` (`:2374-2405`) | `entity_1`, `entity_2`, `name` (synthesised when absent, `:2386-2392`), `relationship`, `biological_state`, `evidence`, `inference`, carriers | same four |

Two consequences a variant author must respect:

* **`modifiers` and `enzymes` are the same field.** `_clean_processes` concatenates
  `item.get("enzymes")` and `item.get("modifiers")` and emits the result as `enzymes`
  (`pipeline.py:2239`). The schema block declares `modifiers`; `enzymes` appears in the
  prompt only in prose (the BACKWARD COMPATIBILITY line at `ctl:104`). The brief lists
  `enzymes` among the per-row field names, so test T5 pins it as a **prose** token rather
  than as a schema key — asserting it inside the schema block would fail the control itself.
* **`relationship` is the only surviving carrier of interaction type.** `class` is deleted,
  and the exporter reads `interaction_type or relationship or class` (`pwml/ir.py:2293`),
  maps only `activation` → `Activation` and `inhibition` → `Inhibition` (`ir.py:29`,
  `:2297`), and writes it as the `interaction-type` attribute (`pwml/writer.py:1502`).

### Enumerated value sets, and where each is fixed

| set | values | fixed at |
|---|---|---|
| `interactions[].relationship` / `class` | `activation \| inhibition \| binding \| interaction` | `ctl:114`; read at `ir.py:2293` |
| reaction `class` | `biochemical_reaction \| transport_reaction \| interaction \| subpathway \| activation \| inhibition \| binding` | `ctl:113` (dropped by cleaning) |
| compound `class` | `compound \| cofactor \| ion \| element_collection` | `ctl:111` |
| protein `class` | `protein \| transporter` | `ctl:112` |
| modifier `role` | `catalyst \| transporter \| activator \| inhibitor` | `ctl:89`, `:101`, schema `:322` |
| modifier `entity_type` | `protein \| protein_complex` | `ctl:95`, schema `:321`; `pipeline.py:1870` |
| `provenance` | `extracted \| enriched \| inferred` | `ctl:121`; matched at `stage_one_boundary.py:212` |
| `scope_membership` | `core \| anaplerotic \| cataplerotic \| auxiliary \| out_of_scope` | `ctl:331`; casefolded at `pipeline.py:356` |
| compartment `canonical_name` | 11 terms, `cytosol` … `endosome` | `ctl:171-172` |

### Why the distinct-key count is 86 and not 85

The schema block declares **85** distinct quoted key tokens across **205** key paths. The
86th is `partial_biosynthetic_step`, which is not a key at all: it is a quoted *value* in the
prose of the UPSTREAM CONTEXT GATE (`ctl:35`, `context_type is "enzyme_mechanism" or
"partial_biosynthetic_step":`) that a `"token":` regex reads as a key. It is counted because
the brief's figure of 86 counts it, and because pinning the whole token set — false positive
included — is strictly stronger than pinning a curated subset: a variant cannot quietly add a
key by hiding it in prose either. Round 1 found the same artifact (its T4 note).

---

## 2. Per-variant verdict

Method key:

* **T1** — the `OUTPUT JSON SCHEMA` block is byte-identical to the control's (CRLF normalised)
* **T2** — that block parses as JSON and its full key-path tree (205 paths) equals the
  control's, **in order**
* **T3** — the distinct quoted key tokens over the whole file equal the control's 86
* **T4** — every enumerated value set occurs, verbatim, the same number of times as in the
  control over the whole file, **and** every process bucket header and per-row field line the
  same number of times *inside* the schema block
* **T5** — `entity_1`, `relationship`, `entity_2`, `inputs`, `outputs`, `modifiers`,
  `evidence`, `class`, `confidence`, `provenance`, `source_refs` are present in the schema
  block, and `enzymes` in the file
* **T6** — pure CRLF, UTF-8 decodable, no stray key token anywhere outside the control's set

| variant | T1 | T2 | T3 | T4 | T5 | T6 | verdict | what it changes instead |
|---|---|---|---|---|---|---|---|---|
| `r2_01_control_v03` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** (is the reference) | control |
| `r2_02_classify_then_branch` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | which rules bind, decided from the source kind |
| `r2_03_conditional_routing` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | which bucket a claim is routed to |
| `r2_04_kind_recheck` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | a verification pass over existing rules |
| `r2_05_mixed_case_procedure` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | the order in which a claim is classified |
| `r2_06_bucket_two_sided_gate` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | which rows a bucket may hold |
| `r2_07_conversion_sweep` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | where to look for a stated conversion |
| `r2_08_summary_not_a_step` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | which rows to suppress as lumped |
| `r2_09_requested_pathway_boundary` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | which `scope_membership` value a row gets |
| `r2_10_legend_vs_body` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | which statement supplies a row's participants |
| `r2_11_example2_repair` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | two example *values*, and one KEY POINTS line |
| `r2_12_regulatory_example` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | adds an example using only existing keys |
| `r2_13_model_stated_example` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | adds an example using only existing keys |
| `r2_14_pre_output_self_check` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | a verification pass over existing rules |
| `r2_15_core_plus_typed_overlay` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | a prior on which bucket carries the result |
| `r2_16_integrated` | PASS | PASS | PASS | PASS | PASS | PASS | **PASS** | five of the above at once |

### The two variants that touch values, and why both still pass

* **`r2_11`** rewrites `"relationship": "inhibits"` → `"inhibition"` and `"activates"` →
  `"activation"` inside worked Example 2 (`ctl:493`, `:498`) and extends that example's KEY
  POINTS line (`ctl:503`). Both rows already carried the matching `class` on the next line, so
  the edit moves two values **into** the enumerated set of `ctl:114`; it adds no value and
  removes none. T4 confirms every value set is unchanged. (Those two values were always out
  of contract — round 1's F-747-6.)
* **`r2_16`** carries the same three replacements and nothing else that deletes text.

Every other variant is insertion-only.

### Sizes and isolation

| variant | bytes | Δ vs control | lines | diff hunks vs control | added lines | removed lines |
|---|---|---|---|---|---|---|
| `r2_01_control_v03` | 38,443 | 0 | 580 | — | — | — |
| `r2_02_classify_then_branch` | 41,668 | +3,225 | 625 | 1 | 45 | 0 |
| `r2_03_conditional_routing` | 40,424 | +1,981 | 607 | 1 | 27 | 0 |
| `r2_04_kind_recheck` | 40,599 | +2,156 | 610 | 1 | 30 | 0 |
| `r2_05_mixed_case_procedure` | 40,434 | +1,991 | 607 | 1 | 27 | 0 |
| `r2_06_bucket_two_sided_gate` | 39,905 | +1,462 | 599 | 1 | 19 | 0 |
| `r2_07_conversion_sweep` | 40,178 | +1,735 | 602 | 1 | 22 | 0 |
| `r2_08_summary_not_a_step` | 39,477 | +1,034 | 593 | 1 | 13 | 0 |
| `r2_09_requested_pathway_boundary` | 40,162 | +1,719 | 604 | 1 | 24 | 0 |
| `r2_10_legend_vs_body` | 39,882 | +1,439 | 597 | 1 | 17 | 0 |
| `r2_11_example2_repair` | 38,684 | +241 | 580 | 3 replacements (one `diff -u` hunk) | 0 | 3 replaced |
| `r2_12_regulatory_example` | 40,551 | +2,108 | 614 | 1 | 34 | 0 |
| `r2_13_model_stated_example` | 40,962 | +2,519 | 625 | 1 | 45 | 0 |
| `r2_14_pre_output_self_check` | 40,695 | +2,252 | 611 | 1 | 31 | 0 |
| `r2_15_core_plus_typed_overlay` | 39,722 | +1,279 | 599 | 1 | 19 | 0 |
| `r2_16_integrated` | 46,615 | +8,172 | 685 | 5 blocks in 3 regions + 3 replacements | 105 | 3 replaced |

Largest is `r2_16` at +21.3% over the control; the largest single-idea file is `r2_02` at
+8.4%. Every file is within the 64,000-token output budget's working assumptions and none
approaches the prompt-size limits that mattered in round 1 (the round-1 failures were
*output* budget, not prompt size — `AUDIT_R2.md` § 1.3).

---

## 3. How the check was run

Two scripts, both in this directory, both read-only against the repo:

* **`_build_r2.py`** — reads the base prompt as **bytes**, asserts it is pure CRLF, and
  derives each variant by inserting after (or replacing) an anchor asserted to occur
  **exactly once**. An anchor that matched zero or two places aborts the build rather than
  guessing, which is what makes "one hunk, insertions only" a property rather than a hope.
  The `OUTPUT JSON SCHEMA` block is never an anchor and never inside a replacement.
* **`_check_schema_r2.py`** — runs T1–T6 over every `r2_*.txt` in the directory, against
  `r2_01_control_v03.txt` as the reference.

Reproduce (no model, no network, no repo writes):

```
python -I _build_r2.py <repo root> <this dir>
python -I _check_schema_r2.py <this dir>
```

Output at the time of writing, captured verbatim in `_SCHEMA_CHECK_R2_output.txt`:

```
control: r2_01_control_v03.txt   205 schema key paths, 86 distinct declared JSON keys

variant                            T1    T2    T3    T4    T5    T6    verdict
r2_01_control_v03                  PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_02_classify_then_branch         PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_03_conditional_routing          PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_04_kind_recheck                 PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_05_mixed_case_procedure         PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_06_bucket_two_sided_gate        PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_07_conversion_sweep             PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_08_summary_not_a_step           PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_09_requested_pathway_boundary   PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_10_legend_vs_body               PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_11_example2_repair              PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_12_regulatory_example           PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_13_model_stated_example         PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_14_pre_output_self_check        PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_15_core_plus_typed_overlay      PASS  PASS  PASS  PASS  PASS  PASS  PASS
r2_16_integrated                   PASS  PASS  PASS  PASS  PASS  PASS  PASS

ALL VARIANTS SHAPE-IDENTICAL TO THE CONTROL
```

---

## 4. Two things the check caught in itself, recorded because they are traps

**A count-based test over the whole file rejects a worked example, not a schema change.**
The first formulation of T4 counted every per-row field line file-wide. `r2_12` and `r2_13`
failed it — `"confidence": 1.0,` went from 10 occurrences to 12, `"modifiers": [` from 5 to
7 — purely because a schema-valid example must repeat the field lines it fills in. That is
exactly what round 1's v08 did, and round 1's T4 did not trip because it was set-based. T4 is
therefore split: the **value sets** are counted file-wide (re-stating one of those would be a
real shape change wherever it appeared) and the **bucket headers and field lines** are
counted inside the schema block only. Both halves must hold.

**A contract string absent from the control is a broken test, not a passing variant.** The
first spec quoted the `provenance` values line as "from an API lookup" where the prompt says
"from API lookup", and quoted several lines that legitimately occur twice (the actor sub-row
shape is declared under both `reactions` and `reaction_coupled_transports`). Both would have
silently passed everything. The script now aborts if any pinned string is absent from the
control, and pins the control's **count** rather than assuming 1.

## 5. What the check does not cover

* It does not verify that a variant's *instructions* are consistent with the schema, only
  that the declared shape is identical. A prompt could keep the shape and still tell the
  model to fill it wrongly; that is what the three-paper run is for.
* It does not check prompt length against the model's context window. All sixteen are within
  2 KB of each other except `r2_16`, and round 1's failures were output-budget, not input.
* It says nothing about the keys the cleaner deletes (§ 1): a variant may legally demand
  `class`, `confidence`, `provenance`, `source_refs` or `compartment_canonical`, and all
  sixteen do, because the control does. They are still dropped before any reader sees them.
