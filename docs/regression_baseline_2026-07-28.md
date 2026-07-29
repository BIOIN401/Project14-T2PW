# Regression Baseline — the 2026-07-28 corpus runs

**Date recorded:** 2026-07-29
**Branch:** `research-mode`
**Source runs:** `runs/2026-07-28_0919` (3 papers), `runs/2026-07-28_2122` (16 papers)
**Machine-readable copy:** `tests/fixtures/baseline_2026_07_28/baseline_metrics.json`
**Harness:** `tests/test_baseline_regression_2026_07_28.py`

This file records the state the diagnostic-detail work was measured against, and
explains the failure classes behind the numbers. **No pipeline behaviour changed
when it was written.** It exists so a later run is compared against a written
baseline rather than against somebody's memory of the night.

> **Why the numbers live in a fixture too.** A prose baseline drifts silently. The
> same figures are in `baseline_metrics.json` and asserted by the harness, so a
> claim in this file that stops being true fails a test.

---

## 1. The night's arithmetic

| Measure | Value |
|---|---|
| Papers attempted | 16 |
| Strict legs executed | 16 |
| **Strict legs passed** | **0** |
| Strict legs failed before the post-pipeline gates | **9** |
| Strict legs failed *at* the post-pipeline gates | 7 |
| Research legs | 8 pass / 8 fail |
| Manifest rows | 32 (unreadable: 0) |
| Timeouts / skipped | 0 / 0 |

The 9/7 split is the single most useful number here, and it is the one the
summary does not state directly. **Nine of the sixteen strict legs never reached
a gate at all** — they died at Stage 1, before any pathway existed to validate.
Only seven produced a payload that the Stage 3 hard gates could reject. Work
aimed at the gates therefore addresses, at most, seven of sixteen failures; the
other nine are an extraction-boundary problem and are untouched by anything the
gates say.

### 1a. The nine pre-gate failures

All nine failed at `stage=stage1`, in four groups:

| Code / kind | Strict legs | What it means |
|---|---|---|
| `entities_required` | 3 | Stage 1 returned a payload with no `entities` object |
| `processes_required` | 3 | Stage 1 returned a payload with no `processes` object |
| `no_reactions` | 1 | A pathway with no reactions and no transports; a `failure_kind`, not an issue code |
| `ambiguous_review_scope` | 2 | `multi_example_review` with no `selected_example`; extraction refused rather than mix pathways |

> **Count these from `manifest.jsonl`, not from `failures_by_code.txt`.** That
> report ranks *distinct papers across both modes*, which is the right view for a
> fix-list and the wrong one for this table: it credits `processes_required` with
> a fourth paper whose strict leg actually died at the gates (its research leg hit
> the code), and it files `no_reactions` separately as an uncoded `failure_kind`.
> Both slips leave the total at nine, so the sum does not catch them.

The first two are the same defect wearing two names, and they are the reason
`_validate_payload_container` in `stage_contracts.py` now records `found_type`,
`key_present` and `payload_keys` on the issue: "Payload must include a processes
object" is equally true of a payload that omits the key, one that has it as
`null`, and one that has it as a list. Three upstream bugs, one message. Which
one fired is now on the error.

### 1b. The seven post-gate failures

Affected: `PMC12444477`, `PMC13231680`, `PMC12312563`, `PMC12657337`,
`PMC12421875`, `PMC12180156`, `PMC12856317`.

Across 27 distinct issue codes, three classes account for nearly all of it:

1. **Missing external identifier** — `Protein 'X' is missing a UniProt or
   DrugBank identifier.` The most common by a wide margin.
2. **Missing species/organism** — usually on the *same* rows as (1).
3. **Degree-zero protein** — `Protein has degree 0 after normalization: X`
   (`PMC12444477` FabA, `PMC12657337`), plus the located-but-isolated variant.

Two further codes appear once each and are worth naming because they are
distinct defects, not variants of the above: `unknown_protein_modifier_reference`
(`PMC12444477`) and `entity ... is declared as both a protein and a
protein_complex` (`PMC12657337`, pyruvate oxidase).

**The dominant sub-class is not a protein problem at all.** Eight of the 27 codes
are two cofactors — coenzyme A and succinyl-coenzyme A — sitting in
`entities.proteins`:

```
gate.protein_coenzyme_a_coa_is_missing_species_organism            PMC12444477
gate.protein_coenzyme_a_coa_is_missing_a_uniprot_or_drugbank_...   PMC12444477
gate.protein_coenzyme_a_is_missing_species_organism                PMC12856317
gate.protein_coenzyme_a_is_missing_a_uniprot_or_drugbank_ident...  PMC12856317
gate.protein_succinyl_coenzyme_a_is_missing_species_organism       PMC12180156
gate.protein_succinyl_coenzyme_a_is_missing_a_uniprot_or_drugb...  PMC12180156
gate.protein_succinyl_coenzyme_a_scoa_is_missing_species_organism  PMC12856317
gate.protein_succinyl_coenzyme_a_scoa_is_missing_a_uniprot_or_...  PMC12856317
```

CoA can never acquire a UniProt ID, so the identity gate will fire on these rows
forever. The gate message is accurate and useless: the defect is the entity-type
decision several stages upstream, and no wording of "missing a UniProt
identifier" points there. This is the class the
`cofactor_misclassified_as_protein` fixture pins.

---

## 2. The PMC12444477 payload-size comparison

The same paper ran twice on 2026-07-28, about twelve hours apart. Sizes are
**filesystem bytes** (`os.path.getsize`):

| Run | Leg | Status | `stage1_payload.json` | `merged_payload.json` |
|---|---|---|---:|---:|
| 09:19 | strict | **pass** | 17,015 | **4,707,785** |
| 09:19 | research | fail | 16,618 | 5,765,448 |
| 21:22 | strict | **fail** | 18,945 | **306,325** |
| 21:22 | research | pass | 21,819 | 274,820 |

Plus the 21:22 research leg's `research_pathway_citations.json` at 1,290,748
bytes.

> **Bytes, not characters.** The per-file sizes recorded in `manifest.jsonl` are
> decoded-character counts, so they read low on any payload holding non-ASCII —
> and these hold Greek letters in entity names (`β-hydroxyacyl-ACP`). The gap is
> about 1%, small enough to look like nothing and wrong enough to mislabel a
> column. The figures above are measured off disk.

Stage 1 output barely moved (17 KB → 19 KB, +11%). The merged payload collapsed
by **15.37×**, and the leg flipped from pass to fail. Whatever the cause, it is
downstream of extraction and it is not a change in how much the paper says.

Two things follow, and only the second is acted on here:

- The comparison is **recorded, not explained.** Diagnosing it means changing
  retrieval or identity behaviour, which is explicitly out of scope for a
  baseline-stabilization pass. It is written down so the next run has something
  to be compared against.
- **It is why the fixtures are hand-written.** A single merged payload from that
  night is 306 KB at its smallest and 4.7 MB at its largest, and the research
  leg's `research_pathway_citations.json` is another 1.29 MB. Copying one into
  `tests/` would add megabytes of verbatim paper text to the repo and *still* not
  say which field mattered.

---

## 3. The compact fixture set

`tests/fixtures/baseline_2026_07_28/cases.json` — eight curated fragments,
each written to the shape observed on disk and carrying an `observed_in` pointer
back to the real artifact. Total: well under 64 KB, asserted by the harness.

| Case | Pins |
|---|---|
| `valid_mapped_enzyme` | The control: LpxA, mapped and species-bearing, passes every identity check |
| `ambiguous_protein_mapping` | `ambiguous_first_candidate` — ambiguity resolved, with the rejected candidates retained in `mapping_meta` |
| `evidence_backed_unmapped_enzyme` | Evidence is not identity: well-sourced, still unmapped; also the end-to-end `detail` test |
| `cofactor_misclassified_as_protein` | CoA / succinyl-CoA in `entities.proteins` — §1b's dominant class |
| `disconnected_mapped_protein` | FabA: mapped, located, degree 0 because the reactions name FabZ |
| `unknown_sentinel_functional_complex` | PathBank 9659 / UniProt `Unknown` stays recognisable as a placeholder |
| `rag_reaction_with_provenance_and_scope` | `rag_provenance` + `source_papers` + `scope_membership`, and that neither trips a gate |
| `curation_removal_orphans_reference` | A duplicate-entity removal that strands a reaction input must be refused; a safe one must not |

The harness is pure and offline — no network, no database, no batch run.

**Running it:** pass an explicit `--basetemp`. Without one, tests that use
`tmp_path` error with `PermissionError` on this machine.

```
.venv/Scripts/python.exe -m pytest tests/test_baseline_regression_2026_07_28.py --basetemp=<scratch>
```

---

## 4. The diagnostic `detail` channel

Landed in `992a4ec`; this section is the rationale, kept out of the modules
themselves so they stay readable.

### Why it exists

A gate error was `{"path", "reason"}` and a stage-contract error
`{"code", "message", "pointer"}`. Both name the offending row and the rule that
rejected it. Neither carries the thing a reviewer needs: **the value that was
actually checked.** That value is in scope at every `_add_error` call site and
was thrown away there.

`logger.debug` cannot fix this. The Streamlit app's logs are not visible from the
browser, so anything written to a logger is invisible exactly when it is needed.
`detail` therefore rides the error object itself — the same dict that already
flows to the UI, to `gate_fail_report.json` and to
`stage_contract_error_report.json`. One population site, three destinations, and
no way for the app and the batch artifact to disagree about what happened.

### Compatibility

`detail` is **additive and omitted when empty**, so an error with nothing to add
keeps its exact historical shape. Consumers that read `path`/`reason`/`code`/
`message` are unaffected.

One consumer did need changing: `batch/driver.py::_research_gate_flags` used
`_text(data.get("detail"))` as a fallback for the human message. `detail` is now
a dict, and `_text()` of a dict would splat a repr into `review_flags.json`. The
fallback was removed and the summary moved to its own `found` key.

Tests assert *required* fields rather than whole-dict equality, so a future
additive field does not break them. `test_strict_gate_blocks_unaccounted_locks_at_stable_pointer`
is the worked example: it matches on the `(path, reason)` pair — the stable
pointer and wording it exists to pin — and separately asserts the detail it
needs, instead of requiring the error to have exactly two keys.

### The size constraint

Payload rows carry `evidence` / `source_refs` blobs holding verbatim passages.
One observed run reached **139,576 characters in a single field**. Embedding a
raw row in an error would push that into `st.json` and wedge the browser tab.

Every value is therefore bounded at the **point of capture**, in
`t2pw/pipeline/failure_detail.py`, not at the render site. Bounding at render
would be a truncation each new call site could forget, and a forgotten truncation
is the bug returning. Two rules make the bound trustworthy:

- **Bulky provenance keys (`ELIDED_KEYS`) never have their content copied.** They
  become a census — `"3 items, 139576 chars elided"`. That evidence is huge is
  itself diagnostic, and dropping the key outright would read as "this row had no
  evidence". Unconditional: small evidence is still raw evidence, and a
  size-dependent rule would make the guarantee depend on the input.
- **Every surviving scalar is clipped** to `MAX_VALUE_CHARS` with the elided
  count stated inline; containers are capped in width and depth.

The result is a hard ceiling on a detail's serialized size that does not depend
on the size of the payload.

`census()` is **idempotent**, which is load-bearing rather than tidy. A detail is
commonly built as `build_detail(row=row_digest(row))` and scrubbed again by
`_add_error`, so the same field is censused up to three times. Without the guard,
each pass measures the previous census string instead of the data —
`"160000 chars elided"` → `"19 chars elided"` → `"15 chars elided"` — and the
count, the one thing the reader needs, is silently replaced by a number about
itself.

### What goes in a detail

The offending row (`row_digest`), the specific value compared, and the comparison
set that failed to contain it.

`row_digest` is deliberately a **denylist plus universal clipping**, not an
allowlist of interesting fields. Gate failures keep arriving as *new* entity
shapes — a RAG-synthesized actor with a key the extractor never produced — and an
allowlist drops exactly the unfamiliar field that would have explained the
failure.

`closest_names` ("did you mean") exists because the dominant recurring gate
failure is a name that *nearly* matches the registry: an actor spelled
differently by the extractor or by RAG synthesis. Suggestions are **display**
names, not normalized keys, so a reviewer can paste one straight back into the
payload. An empty result is itself informative — the reference is genuinely
unknown, not merely misspelled.

`headline` lives in `failure_detail.py` rather than in the app because the batch
flag rows want the same one-line summary; two implementations would drift into
disagreeing about the same error.

### Rendering

`_render_issue_detail` uses a collapsed `st.json(..., expanded=False)` rather
than `st.expander`. Every gate-error list in the app is already inside an
expander, and Streamlit raises on a nested one; the collapsed JSON gives the same
click-to-open behaviour with no nesting constraint, so one helper works at every
call site.

The app does **not** re-truncate. The detail arrives already bounded, and if the
app truncated too, the batch artifacts — which never pass through the renderer —
would carry a different, larger detail than the UI, and the two would disagree
about what the run saw.

---

## 5. Test-suite baseline

Full offline suite on `research-mode` at the time of writing: **1126 passed, 0
failed** (1109 before this document's harness added 17).

`--basetemp` is required; see §3.
