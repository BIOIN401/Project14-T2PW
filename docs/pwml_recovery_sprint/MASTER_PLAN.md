# PWML Recovery Sprint — Master Plan

**Source branch:** `research-mode` @ `9e1b9abe7ba8a1a228558fd03ca6c394cc22c31e` (`ORIGIN_SHA`)
**Integration branch:** `sprint/pwml-recovery`
**Authority:** `PRODUCT_CONTRACT.md`. This plan sequences; it does not define correctness.

---

## 1. Evidence this plan rests on

All measured on `ORIGIN_SHA`. Anything not listed here is an assumption, not a fact.

### 1.1 The primary defect — confirmed live and reproduced

`quarantine_and_close` (`strict_quarantine.py:1736-2045`) calls
`_drop_quarantined_processes` at `:1862`, compacting every process bucket, then calls
`_degree_zero_exports` at `:1876`, which resolves **original** admission indices against
the **compacted** lists via `_surviving_processes` (`:1193-1206`). Out-of-range indices
are silently skipped; shifted in-range indices resolve to the wrong row.

`_surviving_processes` is the **only** unsafe consumer of positional indices. Audited
and safe: the closure loop (pre-drop), `_referenced_state_names` (pre-drop),
`_revalidate_surviving_processes` (pre-drop, and already fails explicitly with
`process_row_vanished_during_closure`), `_reconcile_locked_reactions` and
`write_quarantine_artifacts` (both key off the `originals` snapshot), and
`evaluate_core_coverage` (record-based).

**Measured across 32 archived legs** (`pathway_context=None`, matching archived-leg
reality): 26 unchanged, **6 changed**, 0 errors. Every change is `degree_zero → []`; no
leg gains a degree-zero entity. `PMC12856317/research` stays `ok=False` on a genuine
`unexportable_entity:1`, proving the fix removes only the false refusal. Full table in
`prompts/C-010.md`.

Downstream on the fixed payload: `PMC12452463/strict` passes the final Stage-3 strict
gate **and** the required-field gate. IR build then fails on
`compound_db_resolution_failed` — a separate, exporter-side blocker (§1.2).

### 1.2 Exporter mutates biology after the freeze — confirmed live

`build_pwml_ir` (`pwml/ir.py:966-2007`) calls `_resolve_compound_rows` (`:797`).
Measured on committed `runs_verify/2026-08-04_1647/papers/PMC12856317/strict/`,
`final_mapped.json → pwml_ir.json` mutates all four compound rows. Re-measured by the
corrected probe (H-005, 2026-08-06), reported in five categories that are never summed:

| Category | Instances | Rows |
|---|---|---|
| name changes | **1** — `glycine → Glycine`, no provenance record | 1 of 4 |
| mapped-ID changes | **1** — heme `mapped_ids.pubchem='3334'`, a within-row re-projection of that row's own `pubchem_cid` | 1 of 4 |
| synthetic database rows | **1** — `db_row {"id": 78, "name": "Glycine"}` fabricated at export time | 1 of 4 |
| prefix normalization | **8** — `CHEBI:` stripped from `mapped_ids.chebi` and `chebi_id` | **4 of 4** |
| identity materialization | **16** — `pathwhiz_id`, `db_id`, `db_status`, `chosen_rule` | **4 of 4** |

Rows are paired on `mapping_meta.query.name`, the pre-freeze extraction name, strictly
one-to-one; `raw_name` is **not** used, as anchor or fallback, because the exporter under
examination writes it. Missing, duplicated or unmatched lineage is a named failure that
yields `RESULT: UNDETERMINED` and a nonzero exit — **never** a clean result.

**The `PRODUCT_CONTRACT` § 5 violation is real and remains blocking.** It is violated by
**kind, not by count**: a post-freeze rename with no provenance record, a fabricated
`db_row`, `CHEBI:` stripping on 4 of 4, identity materialization on 4 of 4.

**Acceptance for C-050/C-051:** all five counts 0, with `RESULT: MEASURED`.

**This is NECESSARY BUT NOT SUFFICIENT for T-102.** Per **D-016 (LOCKED)**, T-102
equivalence is **not narrowed to compounds**: it must verify **both** compound identity
**and** organism/species equivalence, across canonical JSON, PWML **and** SBML
(`TEST_MATRIX.md:265` states the same). The probe above measures compound rows in the
canonical JSON only, so **passing it does not satisfy T-102**. Compound identity
equivalence remains required; organism/species equivalence across JSON/PWML/SBML remains
required and is not measured here. Two live examples of that second dimension, both
outside this probe's scope, both measured on the same leg: protein `ALAS2` gains
`pathwhiz_id 17` (`evidence/finding_alas2_identity_placeholder_2026-08-06.md`); and the
single species row moves from `entities.species` to a **top-level `species` group**,
gaining `pathwhiz_id 1` and losing its `mapping_meta` — which carried the
`taxonomy_backfill` provenance for `taxonomy_id 9606`.

> **Corrected 2026-08-06 (H-005). The previous text was wrong on three counts and is kept
> here so the correction is auditable.** It read: "adds `pathwhiz_id`/`db_id` to all four
> compounds and gives Glycine **nine** external identifiers absent from the canonical
> payload (`drugbank DB00145`, `hmdb HMDB0000123`, `kegg C00037`, `chebi 15428`,
> `pubchem 5257127`, `chemspider 730`, `cas 56-40-6`, `pathbank_compound_id 78`)."
> (1) **All nine are present** in the canonical Glycine row's `mapped_ids`; only the
> `chebi` *value* differs (`CHEBI:15428` → `15428`). (2) The prose said nine but the list
> held **eight** — `biocyc GLY` was the missing ninth. (3) The companion "**ten**
> identifiers added post-freeze" was an artefact of the probe pairing canonical and IR
> rows on raw `name`: the rename made Glycine's canonical row unfindable, so its own nine
> identifiers were counted as additions. True figure: **1**. Evidence:
> `evidence/probe_exporter_identity_mutation_2026-08-06.md`, cleanup reports
> `evidence/g11/H-005/`. The magnitude was overstated; the violation was not.

### 1.3 Refusal evidence destroyed at the batch boundary — confirmed live

`write_quarantine_artifacts` produces four artifacts (`strict_quarantine.py:177-180`)
that `batch/driver.py` never writes. `find runs runs_verify -name quarantine_report.json`
returns **zero** across all 15 committed run directories. `_collect_app_text`
(`driver.py:1117`) reads only `st.error`/`st.exception`/`st.warning`, so the entity names
rendered by `st.json(strict_invariants)` are lost. `bench/acceptance.py:84` looks for
`quarantine_report.json` and never finds one.

### 1.4 Identity ladder strips a correct accession — confirmed live

Committed `runs_verify/2026-08-04_1306/papers/PMC12452463/research/final_mapped.json`,
protein `Fur`: `identity_verdict.identity = "uniprot:P0A9A9"` with
`identifier_resolution: "ok"` and `candidate_evidence:
"no_candidate_describes_the_shipped_identifier"` → `identity_evidence_missing` →
stripped → `mapped_ids: {}`. The candidate pool held ten iron-sulfur proteins; `P0A9A9`
was never among them.

### 1.5 No LLM request timeout exists — confirmed static

`OpenAI(...)` at `llm/client.py:47` and `:61` has **no** `timeout=`
(`grep -c "timeout=" → 0`). With `LLM_MAX_RETRIES=8`, base sleep 1.0, max sleep 20.0,
worst case for one `chat_detailed` ≈ 8 × 600 s + 71 s ≈ **4871 s** against a 3480 s child
deadline. This is the most likely mechanism behind PMC12444477 timing out in both modes.

### 1.6 Empty-extraction retry exists and is inert

Retry present (`pipeline.py:3189` `_is_degenerate`, `:3245`, prompt `:3519-3535`).
On-disk `runs_verify/2026-08-04_1754/papers/PMC12782028/strict`: two boundaries,
`attempts:1` then `attempts:2`, **identical** `response_hash sha256:44136fa355b3678a`,
**different** `request_hash` — so the prompt changed and the model did not move. The same
paper's **research** leg passed.

### 1.7 Baseline benchmark — committed run `runs/2026-08-02_2130`

false real identifiers **10** · placeholder-backed proteins **21** · unsupported
reactions **7** · orphaned references **2** · missing supported reactions **3** ·
requested-pathway coverage **0/8** · semantic confirmed **0/8** · strict PWML **0/4**.
Strict failures by boundary: 6 stage3_normalization_gate, 2 stage1_extraction,
2 scope_ambiguity.

---

## 2. What already exists — do not rebuild

| Assumed missing | Actually present | Evidence |
|---|---|---|
| UniProt accession fetch + cache | `_fetch_uniprot_enrichment(client, uniprot_id)` `enrich_entities.py:507`; cache key `uniprot:<ACC>`; `EnrichmentCache` `:208`; snapshotted per run `runner.py:87` | read |
| Accession-keyed PathBank lookup | `PathBankDbResolver.map_protein_by_ids` `map_ids.py:1822` | read |
| Provenance carrier through `_clean_processes` | `_carry_rag_provenance` `pipeline.py:1880`, called at `:2120/:2168/:2215/:2260`; `_RAG_ROW_CARRIER_KEYS` `:1877` | **verified in artifact**: 1 of 6 reactions in `1754/PMC12452463/strict/final_mapped.json` carries `rag_provenance`, `rag_confidence`, `source_papers` |
| Reference-repair alias traversal | `REFERENCE_FIELDS` `reference_repair.py:149-159`, mirrors `validate_registry_references` alias-for-alias | 123 tests pass |
| Semantic `not_evaluated` | `SemanticReport.evaluated` + `not_evaluated_reason`; `ok` vs `confirmed` `semantic.py:284-299` | read |
| SBML canonical binding | `sbml_input_path = canonical_json`; `sbml_input_source = CANONICAL_PAYLOAD_KEY` | read |
| Canonical hash ↔ gate binding | `payload_sha256(final_mapped.json) == final_stage3_gate_report.payload_sha256 == cccf95c8…` on committed `1647/PMC12856317` | **verified** |
| Real-artifact replay harness | `tests/test_strict_quarantine_real_artifact_replay.py`, 7 tests, parameterized over `runs/` | read |
| Explicit-failure pattern for bad indices | `_revalidate_surviving_processes:1448-1453` | read |
| RAG scope contract | `triage.py` (198 L), `eligibility.py` (2300 L), `admission.compare_requested_pathway`/`compare_organism` | read |
| RAG gap detector | `retrieve.detect_gaps:656`, `class Gap:220` (carries `gap_id`, `missing_relationship`, `adjacent_entities`, `expected_type`, `requested_pathway`, `requested_organism`, `reason`), `GapContractError:191` | read |
| RAG query planning | `retrieve.py` (1254 L), `acquire.py` (1043 L), `select.py` (557 L) | read |
| RAG atomic claim schema | `admission.RagReactionCandidate` | read |
| RAG evidence admission | `admission.py` (3100 L): `AdmissionPolicy`, `admit_candidates`, `validate_evidence_span`, `_gap_type_verdict` | read |

**CORRECTED 2026-08-31 (ORCH-717, F-153). The paragraph below was true when written and is
now FALSE, and it was the most harmful kind of stale: this section's whole job is to stop an
agent rebuilding what exists, and on these two items it said the opposite.**

~~*Genuinely missing in RAG: a stopping policy and a loop controller.*~~ **Both exist and both
are WIRED.** `rag/loop_policy.py` and `rag/controller.py` are present, and
`streamlit_app.py:1270` imports `run_rag_loop`, `:1426` calls it inside `run_rag_rounds`, and
~~`:5669`~~ **`:5636`** calls `run_rag_rounds` from the production pass (`run_rag_rounds` is
defined at `:1239`). `tests/test_c055_rag_loop_wiring.py` covers the wiring.
`rag/graph_delta.py` exists as well.

**⚠ `:5669` corrected to `:5636` by C-109, measured at `2ac8404`.** `:5669` is where that call
sits in the **uncommitted working copy** of `streamlit_app.py` in the primary checkout — a file
the sprint's never-commit list forbids committing. The committed blob has it at `:5636`. F-153
and the C-109 charter both carry `:5669`; **a citation that resolves only against uncommitted
bytes is F-154's defect in its worst form, because no reader can reproduce it.** Verify wiring
claims against `git show <ref>:<path>`, never against a dirty working tree. Measured in
`evidence/c109_citation_probe.log`.

~~**`rag/controller.py:11` still carries `UNWIRED: nothing in production calls it; wiring is
C-055's` in its own module docstring.** That line is false at this tip and is registered under
F-153. It is a `src/` change and is **not** fixed here -- no card in this wave owns that file.~~
**HANDED FORWARD AND NOW FIXED by C-109** (`card/C-109-control-plane`, F-153 remainder). The
module docstring of `src/t2pw/rag/controller.py` now retracts the `UNWIRED` sentence in place --
struck, not deleted -- and names the three `streamlit_app.py` call sites plus
`tests/test_c055_rag_loop_wiring.py`. The diff is docstring-only and provably inert: the
docstring-stripped AST is byte-identical to the base blob's. Proof:
`evidence/c109_controller_inertness.py` / `.log`. The `is not fixed here` clause above is struck
because it is no longer true, not because it was wrong when written.

Graph-delta validation being partial (`conform.py` conforms and merges but does not validate the
delta against a policy) is **not** re-verified by this correction and stands as written.

---

## 3. Conflict hotspots

| Rank | File :: function | Lines | Branches | Mitigation |
|---|---|---|---|---|
| 1 | `streamlit_app.py :: run_post_pipeline_sbml_artifacts` | **1226** | C-030, C-050, C-052, **C-030a** | **C-011 seam**; C-030a is **test-only** and **serializes against C-052** |
| 2 | `driver.py :: _drive` | **528** | C-031, C-032, C-041, C-053 | **C-012 seam** |
| 3 | `streamlit_app.py` module-level script body | **2550, no function** | C-055 | least-verifiable branch in the sprint; senior reviewer; AppTest-driven focused tests mandatory |
| 4 | `pwml/ir.py :: build_pwml_ir` | 1042 | C-040, C-051 | single owner; split three ways |
| 5 | `pipeline.py :: _run_json_stage` | 284 | C-038, C-042 | serialize |
| 6 | `strict_quarantine.py :: quarantine_and_close` | 310 | C-010, C-041 | C-041 branches from C-010's integration commit |
| 7 | `acceptance.py :: _build_denominators` | 182 | C-053, C-056b | serialize. **C-054 removed 2026-08-17: it does not edit `_build_denominators`** — its manifest is `bench/goldset.py` only and its target is `goldset.py:647`, read here as `case.expected_export` (`acceptance.py:626`). It is an **adjacent reader that moves the denominator while C-053 moves the numerator**, so it is ordered **after C-053**, not merged with it |
| 8 | `map_ids.py :: _enforce_shipped_identity_names` | 173 | C-033, C-044 | C-033 first |
| **9** | **`tests/test_streamlit_quarantine_boundary.py`** (whole file) | — | **C-041a, C-050a**, and every future freeze-lifecycle / release-classification card | **test-function-level ownership, assigned 2026-08-13** — see below |
| **10** | **`tests/test_batch_driver_seam_golden.py`** :: `_observable` (`:92-107`), `GOLDEN` (`:125-133`) | — | **C-053** (assigned 2026-08-17); C-012's acceptance artifact, previously in **no** card's manifest | **test-function-level ownership**, mirroring row 9. `_observable` folds `to_dict()`, artifact write order, sorted keys, per-artifact digests and the payload hash into one digest, so a new row key, a renamed artifact **and** a changed message each move all **seven** `GOLDEN` slots. **Two hard guards: (a) `_observable`'s field list may only GROW, never shrink** — dropping a field to stabilise a digest is a reject, and `:106`'s `release_status_absent` is **replaced, not deleted**, by a field recording the row's actual `release_status` and `pwml_artifact`, so *absence* stops being the invariant and *presence* becomes one; **(b)** the move ships as an **exact slot-by-slot delta** with the base capture committed under `evidence/`, per merge rule 4. The test function at `:136-139` is untouched. The file is in **no chunk** (`TEST_MATRIX` carries it only as the "✔ driver" obligation), so C-053 must run it by name |
| **11** | **`tests/test_c011_freeze_seam_golden_equivalence.py`** :: `_leg_projection`, `_with_c030_hash_keys` + `docs/…/evidence/c011_freeze_seam_before.json` | — | **C-052** (assigned 2026-08-17, D-040); C-011's acceptance artifact, previously in **no** card's manifest | **test-function-level ownership**, mirroring rows 9–10. The 39-leg fixture is pinned **byte-for-byte** against a base blob, so it moves by a **documented `_C052_KEYS` delta in the exact shape of the existing `_C030_KEYS` precedent — never by regeneration**, and the 9-field projection is **growth-only**. Added fields are `canonical_json_path_name` and `canonical_json_path_in_tmp`; **the absolute path is UUID-random and swept (`:2594`/`:2724`/`:3809`), so recording it would make the fixture unregenerable**. **Serializes against C-030a**, which needs the same projection for A0-C7 — **C-052 first** (growth-only), C-030a second (changes an existing field's semantics). The file is in **no chunk** and must be run by name |

`streamlit_app.py` is 6274 lines of which **2550 (41%) is module-level script body**
unreachable by unit tests.

### Hotspot 9 — `tests/test_streamlit_quarantine_boundary.py`, resolved 2026-08-13 (P2-11)

This file was in **no card's manifest** while pinning app-boundary behaviour that the
freeze-lifecycle and release-classification cards are chartered to change. Two Pack 2 cards
collided with it independently, for unrelated reasons. Leaving it unowned meant every future
lifecycle card would re-litigate it. **Ownership is now explicit and at test-function level:**

| Owner | Owns exactly | Cause |
|---|---|---|
| **C-041a** | its four re-pinned `report["ok"]` assertions + `_payload_with_no_viable_core` | they pin **pre-D-002** boundary behaviour for runs **D-002 (LOCKED)** now requires to export as `review_required`. Moved under **merge rule 4**, with the hard guard that a payload with **no genuine surviving core** keeps `ok is False` |
| **C-050a** | `test_research_mode_keeps_the_unmapped_candidate_and_does_not_block` (node15) + minimum supporting fixture | its comparand `final_mapped_db` is **pre-enrichment**, so the assertion's span is wider than the invariant it states. Stale since **C-011** unified the payloads; exposed by the first non-no-op stage inserted into that span |

**Standing rules for this file, binding on every future card:**

1. **No card may add, remove, rename or reorder a test function here.** The Chunk D `qb` gate
   addresses nodes **positionally**; moving one renumbers every node after it and invalidates
   every prior classification.
2. It is a **D-apptest** file, validated only through the 23-node `qb` gate (~10.5 min), and it
   sits behind **BL-003**'s flapping. **Never read a red as expected** — classify under D-023
   (deterministic · diff-reproducible · traceback-implicated) first.
3. **Two cards owning disjoint functions here must not be merged blind of each other.**
   Whichever lands second **re-runs `qb` against the combined state**, and neither may revert
   the other's re-pin.
4. A card correcting an assertion here must **preserve or strengthen** the invariant the
   assertion states. Re-pointing a **stale comparand** is in scope; **weakening the comparison**
   — a partial-field check, a subset check, a normalizing round-trip — is not.

---

## 4. Waves

```
WAVE 0   INIT-001 sprint init + baseline capture   [BLOCKS EVERYTHING]
         SPIKE-002 compound-resolution scoping     [blocks C-040/050/051 only]
         R-003 false-identifier triage   R-004 rag-reintroduction triage  (read-only)

WAVE A0  independent, zero shared files
         C-010 p01-stale-index        C-011 p00a-freeze-seam
         C-012 p00b-driver-seam       C-013 p04a-hash-module
         C-014 p03a-llm-timeout       C-015 p20-lineage-schema
         C-016 p30-rag-stop-policy    C-017 p40-semantic-module
         C-018 p50-cofactor-classifier

WAVE A1  C-020 p06a-equiv-comparator   [needs C-013]
         C-021 p31-rag-graph-delta     [needs C-015]

WAVE B   C-030 p04b-hash-wiring        [C-011, C-013]
         C-031 p02-quarantine-artifacts[C-012]
         C-032 p03b-deadline-module    [C-012, C-014]
         C-033 p10-identity-hydration  [none]
         C-034..C-037 lineage writers  [C-015]
         C-038 p25-lineage-carrier     [C-015]

WAVE C   C-040 p05a-resolution-extract [SPIKE-002]
         C-041 p08-release-status      [C-010, C-012]
         C-042 p03c-extraction-ladder  [C-032, C-038]
         C-043 p32-rag-controller      [C-016, C-021]
         C-044 p26-lineage-mapping     [C-015, C-033]
         C-045 pre-freeze species/organism canonicalization
               -- PLANNING ONLY; blocked pending separate product-owner
                  authorization of its complete implementation authority

WAVE D   C-050 p05b-prefreeze-call     [C-040, C-030]
         C-051 p05c-ir-assert-only     [C-040, C-050]
         C-052 p06b-freeze-enforce     [C-030, C-050, C-020]
         C-053 p09-pwml-naming         [C-041]
         C-054 p16-goldset-required    [C-041]
         C-055 p33-rag-wiring          [C-043, C-041, C-032]
         C-056a p42a-semantic-runtime  [C-017, C-041]
         C-056b p42b-semantic-bench    [C-056a, C-053]
         C-057 p27-lineage-quarantine  [C-015, C-010, C-041]

WAVE E   C-060 p51 / C-061 p52 false-content repairs
         PLACEHOLDERS — not dispatchable until R-003/R-004 deliver exact findings,
         affected files, expected corrections and regression fixtures.
```

### Unavoidable serial chains — only four

| Chain | Depth | Why it cannot be interface-mocked |
|---|---|---|
| `C-011 → C-030 → {C-050, C-052}` | 3 | all edit `freeze_canonical_payload`; the seam **is** the interface, so nothing exists to mock until C-011 lands. C-050 is placed in the enrichment block *above* the seam to keep this at 3, not 4. |
| `C-010 → C-041 → {C-053, C-054, C-056b, C-057}` | 3 | C-041 edits `quarantine_and_close:1896` which C-010 restructures at `:1868-1876`; `coverage_verdict` is a return-shape change, so consumers cannot build against a stable interface until it is fixed |
| `C-015 → C-034..C-038, C-044, C-057` | 2 | writers need the schema frozen; all seven then run concurrently |
| `C-016 + C-021 → C-043 → C-055` | 3 | the controller composes both policies; wiring needs the controller API and `coverage_verdict` |

Everything else is interface-separable and must not be scheduled as serial.

---

## 5. Merge gates — all must hold

| # | Gate |
|---|---|
| G1 | Dependency branches already merged into `sprint/pwml-recovery` |
| G2 | Diff touches **only** the branch's owned files/functions |
| G3 | Focused tests pass (register row) |
| G4 | Existing affected tests pass, **or** a pinned baseline moved deliberately with an exact documented delta (G9) |
| G5 | An independent reviewer approved the **actual diff**, not the report |
| G6 | No biological gate weakened to increase PWML production |
| G7 | Incomplete-but-correct pathways still exported as `review_required`, never dropped |
| G8 | No exporter repairs biology after the freeze |
| G9 | A **claimed correction or preservation of pre-existing observable behaviour** carries a proof that **fails behaviourally on the base SHA** and passes at the tip; **symbol absence is not proof**. A genuinely **new** capability or module carries an **explicitly labelled new acceptance test** instead, with no fabricated base failure. Mislabelling a regression as new functionality is a reject. Full statement: `TEST_MATRIX.md` § Regression-test standard |
| G10 | Smoke suite (465 tests, ~40 s) passes after the merge, on the integration branch. **465 since C-054** moved it 460→465 under merge rule 4; 457 and 460 are stale. Delta and per-test names: `TEST_MATRIX.md` § Chunks |
| **G11** | **Test-process lifecycle.** Every test/benchmark/pipeline/LLM command in the branch's evidence ran through the bounded foreground wrapper, and the cleanup report shows **final surviving count = 0**. A run with surviving owned processes is an **infrastructure failure**, not a test result, and cannot satisfy G3, G4 or G10. Full policy: `TEST_MATRIX.md` § 0 |

**Reject** when: an earlier merge invalidated the branch's assumptions — now proven by
running the card's focused tests on the **prospective combined integration state**, never
by rebasing the worker branch; the diff is correct but out of boundary; a report claims
runtime behaviour with no pasted evidence; a benchmark failure justifies code without an
adjudication per `PRODUCT_CONTRACT.md` §14.

**Integration is merge-only.** Review the exact tip, then on the integration branch
`git merge --no-ff --no-commit <reviewed-tip>`, inspect the staged merge against the
authorized manifest, run the card's focused tests and gates on that prospective state, and
only then commit. **No rebase**, and integration is never merged back into a worker branch.
Protocol and the **forward parallel-merge proof** (second parent · owned-path
first-parent-to-merge diff · remaining content from the first parent):
`prompts/_TEMPLATE_INTEGRATE.md` — its `DO` block and § Parallel branches; D-023.

---

## 6. Overnight rules

- Dispatch only READY branches whose dependencies are merged.
- **One writer per semantic seam per merge window.** Disjoint functions and files may be
  developed **concurrently**; **merges are serial**. D-021's one-reshaper-at-a-time
  freeze-lifecycle rule remains binding, `C-041 → C-031 → C-032` remains serialized
  because `driver.py` behaviour is coupled, and **no rebase is allowed**.
- Each card charter carries: branch and exact dispatch base · function-level ownership ·
  carried A0 requirements · explicit exclusions · hand-authored and generated budgets ·
  focused tests · applicable **real** G9 obligations · relevant traps · the **SBML
  prohibition**. A charter may be an external durable record with its hashes reported at
  closeout; a tracked prompt commit is not required for every card
  (`prompts/_TEMPLATE_IMPLEMENT.md` § The card charter).
- Start long jobs (chunk D, nightly full suite, milestone benchmarks) after the night's
  likely merges.
- Research agents diagnose committed artifacts while implementers code. They take no
  branch.
- On return: do not merge everything that finished. Review in dependency order, then take
  each reviewed tip as a `--no-commit` merge and rerun its focused tests on that combined
  prospective state before applying G1–G11 and committing the merge.

---

## 7. Standing traps

**TRAP-1** PMC12452463's gold `export_rationale` records the route as chemically broken.
After C-010 it passes quarantine, the final Stage-3 gate and the required-field gate. Its
correct outcome is `review_required` with `strict_acceptance_eligible=false`. It must
never count as strict success. Any agent optimizing toward "PMC12452463 passes strict" is
chasing the wrong target.

**TRAP-2** `test_strict_quarantine_real_artifact_replay.py` pins `FULL_STACK_BASELINE`
at `:384-393` and asserts it by exact equality at `:432`. C-010 changes it by design. An
agent that makes this test pass by reverting behaviour must be rejected.

**TRAP-2 is superseded in part by H-001.** The pin was measured against a *filesystem
glob* and was already stale on `ORIGIN_SHA` — 23 pinned, 39 measured — for reasons
unrelated to C-010 (`BASELINE.md` § 5). H-001 freezes the cohort to a manifest and
re-records the expectations **before** C-010 is dispatched. C-010's implementer therefore
inherits a *passing* gate, and the only delta they may cause is the six-leg allowlist in
`BASELINE.md` § 6 — of which **exactly two fall inside this gate's cohort**
(`runs/2026-08-02_2130/papers/PMC12096016/strict` and `.../PMC12856317/research`); the
other four live under `runs_verify/`, which this gate does not read.

**TRAP-3** `placeholder_backed_proteins` is a standing policy disagreement, not a defect.
No agent may "fix" it. Escalate.

**TRAP-4** Do not rebuild anything in §2.

**TRAP-5** `data/enrichment_cache.json` is **39 MB and tracked**. No branch may commit a
cache modification. `.git` is already 158 MB.

---

## 8. Schedule

| Day | Work |
|---|---|
| 1 | INIT-001, SPIKE-002, Wave A0 dispatch, R-003/R-004; merge A0; dispatch A1 |
| 2 | Wave B; T-100 (M1, 4 legs, ~1.5 h) |
| 3 | Wave C; T-101 (M2, 6 legs, ~2 h) |
| 4 | Wave D; T-102 (M3, equivalence, ~25 min); T-103 (M4, 4 RAG legs, ~1.5 h) |
| 5 | Wave E, integration, subsystem suites; **T-104 first RC benchmark starts ~18:00 (~7 h)** |
| 6 | Results ~01:00; full day of triage and narrow benchmark-proven corrections; second RC ~18:00 |
| 7 | Targeted reruns, equivalence verification, acceptance matrix, tag. **No architectural change.** |

The first RC benchmark starts Day 5 evening so Day 6 is entirely available for
correction and Day 7 for verification.

---

## 9. Branch register

Ownership is exclusive. A diff outside the owned list is an automatic reject.
Reviewer is always a different agent than the implementer.

**Canonical paths — bare filenames in this table are not unique on disk.** Resolve every
row to these, and never edit a re-export shim:

| Bare name used below | Canonical file | Decoy to avoid |
|---|---|---|
| `pipeline.py` | `src/t2pw/pipeline/pipeline.py` | `src/pipeline.py` (1-line `import *` shim) |
| `map_ids.py` | `src/t2pw/mapping/map_ids.py` | `src/map_ids.py` (5-line shim) |
| `extract.py` | `src/t2pw/extraction/extract.py` (C-034) | `src/extract.py`; `src/t2pw/rag/extract.py` |
| `driver.py` | `src/t2pw/batch/driver.py` | — unique |
| `runner.py` | `src/t2pw/batch/runner.py` | — unique |
| `ir.py` | `src/t2pw/pwml/ir.py` | — unique |
| `acceptance.py` | `src/t2pw/bench/acceptance.py` | — unique |
| `gate_reports.py` | `src/t2pw/pipeline/gate_reports.py` | — unique |
| `streamlit_app.py` | `src/t2pw/app/streamlit_app.py` | — unique |
| `strict_quarantine.py` | `src/t2pw/pipeline/strict_quarantine.py` | — unique |

| ID | Branch | Wave | Depends | Owns (file :: function) | Reviewer | Focused | Chunk D |
|---|---|---|---|---|---|---|---|
| C-010 | `agent/p01-stale-index` | A0 | — | `strict_quarantine.py` :: `_surviving_processes`, `_degree_zero_exports`, `quarantine_and_close`; `test_strict_quarantine.py`; `test_strict_quarantine_real_artifact_replay.py`; `docs/change_log.md` | C-041 impl | A, E | — |
| C-011 | `agent/p00a-freeze-seam` | A0 | — | `streamlit_app.py` :: `run_post_pipeline_sbml_artifacts` | C-012 impl | D | ✔ |
| C-012 | `agent/p00b-driver-seam` | A0 | — | `driver.py` :: `_drive` → `_finalize_{gate_failure,pwml_export,timeout}` | C-011 impl | B + golden | — |
| C-013 | `agent/p04a-hash-module` | A0 | — | NEW `pipeline/canonical_hash.py`; `gate_reports.py` :: `payload_sha256`, `stamp_report`, `gate_verdict` | C-020 impl | smoke | — |
| C-014 | `agent/p03a-llm-timeout` | A0 | — | `llm/client.py` :: `OpenAI(...)`, `chat_detailed`, `chat_with_tools` | C-032 impl | A, C | — |
| C-015 | `agent/p20-lineage-schema` | A0 | — | NEW `pipeline/lineage.py` | C-038 impl | new | — |
| C-016 | `agent/p30-rag-stop-policy` | A0 | — | NEW `rag/loop_policy.py` | C-043 impl | new, C | — |
| C-017 | `agent/p40-semantic-module` | A0 | — | NEW `bench/semantic_production.py` | C-056a impl | B | — |
| C-018 | `agent/p50-cofactor-classifier` | A0 | — | NEW `pipeline/cofactor_policy.py` | R-003 | new, C | — |
| C-020 | `agent/p06a-equiv-comparator` | A1 | C-013 | NEW `pipeline/canonical.py` :: `biological_equivalence` (parses + normalizes **JSON, PWML, SBML**) | C-013 impl | new, D | ✔ |
| C-021 | `agent/p31-rag-graph-delta` | A1 | C-015 | NEW `rag/graph_delta.py` | C-016 impl | new, C | — |
| C-030 | `agent/p04b-hash-wiring` | B | C-011, C-013 | `streamlit_app.py` :: `freeze_canonical_payload` | C-052 impl | D | ✔ |
| C-030a | *(unallocated)* | D | C-011, C-030 | **A0-C7 only. Test-only.** Object-sharing assertions at the **real** sharing sites in `streamlit_app.py` :: `run_post_pipeline_sbml_artifacts` — **§3 hotspot row 1, shared with C-050 and C-052** — plus the directly corresponding tests. **Explicitly NOT C-030's boundary:** F-008 proved A0-C7 undischargeable inside `freeze_canonical_payload`, where the payload is a `deepcopy` nothing in the returned dict aliases. Discriminator re-measured `streamlit_app.py:3746` (**not** F-041's `:3748` — drifted). Must not change C-011 lifecycle semantics | non-author, **NOT** the C-052 implementer | D | — |
| C-031 | `agent/p02-quarantine-artifacts` | B | C-012 | `driver.py` :: `_add_common_artifacts`, `_add_identity_artifacts` | C-053 impl | B | — |
| C-032 | `agent/p03b-deadline-module` | B | C-012, C-014 | NEW `pipeline/deadline.py`; `runner.py` :: `_timeout_row`, `launch_child`, `child_command`; `_finalize_timeout` | C-042 impl | B | — |
| C-033 | `agent/p10-identity-hydration` | B | — | `src/t2pw/mapping/map_ids.py` :: `verify_real_protein_identity`, `_enforce_shipped_identity_names`; `src/t2pw/pipeline/entity_identity.py`; NEW `src/t2pw/mapping/uniprot_evidence.py`. **Not** `src/map_ids.py` — that is a 5-line re-export shim | C-044 impl | C | — |
| C-034 | `agent/p21-lineage-extract` | B | C-015 | **RE-SCOPED 2026-08-13:** `pipeline/stage_one_boundary.py` :: `settle_stage_one` (lineage writes only). The former target `extraction/extract.py` is **dead demo code** — 29 lines, one hardcoded paragraph, `run_demo()` under `__main__`, reached only through the `src/extract.py` shim — and is now **excluded from this card**. Ownership stays disjoint from C-042 | rotate | A **+ `tests/test_stage_one_boundary.py` + `tests/test_early_failure_replay.py`** (neither is in any chunk) | — |
| C-035 | `agent/p22-lineage-rag` | B | C-015 | `rag/synthesize.py`, `rag/admission.py` — **except `parse_span_relation` and `validate_evidence_span`, which are C-061's** (corrected 2026-08-13; C-035 merged with both **byte-identical**, verified three times including in the merged tree, precisely to keep them free) | rotate | C | — |
| C-036 | `agent/p23-lineage-audit` | B | C-015 | `curation/apply_audit_patch.py` | rotate | A | — |
| C-037 | `agent/p24-lineage-gapres` | B | C-015 | `curation/gap_resolver.py` | rotate | C | — |
| C-038 | `agent/p25-lineage-carrier` | B | C-015 | `pipeline.py` :: `_carry_rag_provenance`, `_RAG_ROW_CARRIER_KEYS` | C-015 impl | A + provenance test | — |
| C-040 | `agent/p05a-resolution-extract` | C | SPIKE-002 | NEW `pwml/compound_resolution.py`; `ir.py` :: `_resolve_compound_rows`, `_canonicalize_compound_offline` — **INCOMPLETE: this row names 2 of the 4 functions C-040 must move (`SPIKE-002-REPORT.md` §3). Do NOT dispatch against this row; `DECISIONS.md` D-021 §2 is C-040's authoritative symbol manifest** | C-051 impl | D | ✔ |
| C-041 | `agent/p08-release-status` | C | C-010, C-012 | NEW `pipeline/release_status.py`; `strict_quarantine.py` :: `evaluate_core_coverage`; `_finalize_gate_failure`; `batch/report.py`; `bench/render.py` | C-010 impl | A, B | — |
| C-041a | `agent/p08a-d002-release-seam` | C | C-041, F-006 / D-002 | The release/refusal seam around `strict_quarantine.py` :: `quarantine_and_close`. **Also owns, in `tests/test_streamlit_quarantine_boundary.py`, exactly its four re-pinned `report["ok"]` assertions and `_payload_with_no_viable_core`** — moved under merge rule 4 because **D-002 (LOCKED)** changes the behaviour they pin, with the hard guard that where a payload has **no genuine surviving core** the assertion **stays `ok is False`**. **Ceiling 1500 → 1900 by final product-owner pre-commit re-charter, 2026-08-13** (valid because no implementation commit existed). **Must not run concurrently with C-057** — both touch `strict_quarantine.py` | independent | A, D (`qb`) | — |
| C-042 | `agent/p03c-extraction-ladder` | C | C-032, C-038 | `pipeline.py` :: `_run_json_stage`, `_build_extraction_prompt`; `extraction_diagnostics.py` | C-032 impl | A | — |
| C-042a | `agent/p03d-attempt-cap-reason` | C | C-042 ✔ merged `8917349` | **NEW 2026-08-13, product-owner ruling (RULING 3).** A **seventh** termination reason `attempt_cap_reached`, under new decision **D-024**. Owns the minimum enum/schema, serialization and call site to represent it end-to-end: `pipeline/deadline.py` :: `TERMINATION_REASONS` / `require_reason`; `rag/loop_policy.py` :: `TERMINATION_PRECEDENCE`; `pipeline/extraction_ladder.py` :: the attempt-cap refusal at `:478-483` and the `termination_reason` serialization. **`OPERATIONAL_TERMINATION_REASONS` is UNCHANGED** — widening the strict-success denominator is a product decision not made. **C-042 is NOT reopened, amended or re-reviewed** | independent | C | — |
| C-043 | `agent/p32-rag-controller` | C | C-016, C-021 | NEW `rag/controller.py` (unwired) | C-055 impl | C | — |
| C-044 | `agent/p26-lineage-mapping` | C | C-015, C-033 | `src/t2pw/mapping/map_ids.py` (lineage writes only). **Not** `src/map_ids.py` — re-export shim | C-033 impl | C | — |
| C-045 | — **planning only, no branch** | C | — | `ir.py` :: `_canonicalize_species_offline` — **sole owner** (D-016). Species/organism canonicalization is a **pre-freeze semantic unit**: it must execute **before** the canonical freeze, and must preserve or record organism/species provenance. It does **not** absorb compound canonicalization, which remains C-040's. **PLANNING ONLY — unimplemented, undispatched, not dispatchable**; this entry establishes no card, dependency, reviewer, test chunk, branch or implementation budget | — | — | — |
| C-050 | `agent/p05b-prefreeze-call` | D | C-040, C-030 | `streamlit_app.py` :: enrichment block **above** the seam | C-052 impl | D | ✔ |
| C-050a | `agent/p05c-node15-comparand` | D | — | **NEW 2026-08-13, product-owner ruling (RULING 1). TEST-ONLY; no production code authorized.** `tests/test_streamlit_quarantine_boundary.py` :: `test_research_mode_keeps_the_unmapped_candidate_and_does_not_block` — the node15 quarantine-boundary comparand — plus the minimum directly supporting fixture code. Re-points `mapped_in` from the **pre-enrichment** `final_mapped_db` to the **post-enrichment** artifact quarantine is actually handed (`final_mapped_enriched`, already published at `streamlit_app.py:3725`). Closes **P2-11 / B-P2-1**. **Merges BEFORE C-050.** ⚠ branch prefix `p05c` is shared with C-051's planned `agent/p05c-ir-assert-only` — **different cards, different branches, do not confuse them** | independent | D (`qb`) | ✔ |
| C-051 | `agent/p05c-ir-assert-only` | D | C-040, C-050 | `ir.py` :: `build_pwml_ir`. **Its charter MUST also name `ir.py` :: `_entity_record` (P2-06):** `pathwhiz_id` / `pathbank_compound_id` are materialized there, **not** in `_resolve_compound_rows`, so deleting the in-IR resolution call alone would leave a post-freeze materialization path live | C-040 impl | D | ✔ |
| C-052 | `agent/p06b-freeze-enforce` | D | C-030, C-050, C-020, **C-050i (merges first)** | **Per D-040.** `streamlit_app.py` :: `freeze_canonical_payload` (**ZERO lines** — its seven-field return at `:2609-2614` already publishes `canonical_json_path`), `run_pwml_export`, the SBML binding (`:3638`/`:3650`), **+ key-addition ONLY on the `run_post_pipeline_sbml_artifacts` success return `:3680-3807`** (no card owned it; D-021 §2 `:513` lists C-052 under ***Not owned by***, so the charter's ownership claim was a misread), + `_json_artifact_entries` one-entry grant, + hotspot 11. **Added key names must NOT contain `pwml`** (`driver.py:1145` would return it as the PWML export result). **NOT** `driver.py` (D-038 gave `_add_strict_artifacts` to C-053) · **NOT** A0-C7 (C-030a's) · **NOT** the CLI's filename (`writer.py:2690`+`:2817` collide) | C-030 impl | **`qb` + full Chunk D + SMOKE + named focused set** (5 of 7 changed files in no chunk) | ✔ |
| C-053 | `agent/p09-pwml-naming` | D | C-041 | **Re-measured and widened 2026-08-17 (D-038).** `src/t2pw/batch/driver.py` :: `_add_strict_artifacts` (`:1373-1396` — **the only site naming the strict artifact**, and in no card's manifest before now), `_finalize_pwml_export` (`:1673-1691`), `RunOutcome` fields (`:648-679`), `RunOutcome.to_dict` (`:714-737`) · `src/t2pw/batch/runner.py` :: `REQUIRED_ARTIFACTS` (`:116-119`), `required_artifacts` (`:845-856`) · `src/t2pw/bench/acceptance.py` :: `_STRICT_DELIVERABLE`/`_RESEARCH_DELIVERABLE` (`:88-89`), `ModeResult`+`to_dict` (`:139-193`), leg population (`:483-498`), `_build_denominators` strict block (`:625-641`) · plus `tests/test_batch_driver_seam_golden.py` :: `_observable`+`GOLDEN` (hotspot 10). **EXCLUDED, measured at zero lines: `driver.py :: _drive`** — `pwml_result` already reaches both helpers at `:2126`/`:2177`, so no signature and no call site moves. **Also excluded: `batch/report.py`, `bench/render.py`** (C-041's; already wired, need no edit) | C-031 impl | **B + named focused set** (`test_release_status_classification.py`, `test_strict_quarantine_release_seam.py`, `test_batch_run.py`, `test_batch_report.py` — **four of the five files C-053 must change are in NO chunk**) | — |
| C-054 | `agent/p16-goldset-required` | D | C-041, **C-053** | `bench/goldset.py` | C-056b impl | B | — |
| C-055 | `agent/p33-rag-wiring` | D | C-043, C-041, C-032 | `streamlit_app.py` :: `maybe_run_rag` + script body | senior | C + AppTest | — |
| C-056a | `agent/p42a-semantic-runtime` | D | C-017, C-041 | **Expanded by D-037, used per D-039.** `pipeline/release_status.py` :: semantic input to **runtime `release_status`** · `pipeline/strict_quarantine.py` :: the `classify_release_status(...)` expression at `:2080-2088` + the evaluation feeding it (**not** `:2155` `schema_version`, **not** a new key under `release`) · `bench/semantic_production.py` :: `_check_requested_organism` (`:113-146`, predicate `:134`) **only**, via a **function-local** import of the **public** `rag.admission.compare_organism` · directly corresponding tests. **NOT** `bench/semantic.py:753` · **NOT** `rag/eligibility.py`/`rag/admission.py` (import-only, `:147`) · **NOT** `tests/.../contract_alignment.py :: _base()` (**STOP**: 15 call sites, 3 importers, 2 in SMOKE) · **`test_e` must stay byte-identical** | **independent — NOT C-017 impl, NOT C-041/C-041a impl** (D-039 §7; grant places it inside both their territories) | **B + `D (qb)`** + named focused set (`test_strict_quarantine_release_seam.py`, `test_release_status_classification.py` — **in no chunk**). `qb` from the **primary checkout** only. **Serialize against C-057** | — |
| C-056b | `agent/p42b-semantic-bench` | D | C-056a, C-053 | `acceptance.py` :: `_build_denominators` | C-056a impl | B | — |
| C-057 | `agent/p27-lineage-quarantine` | D | C-015, C-010, C-041 | `strict_quarantine.py` (lineage writes only) | C-041 impl | A, E | — |
| C-060 | `agent/p51-false-id-repairs` | E | R-003 ✔ | **SCOPE ACCEPTED 2026-08-13 (R-003 M1 only):** NEW `pipeline/entity_admission.py` — an assay-reagent admission gate — plus the minimal call site in `pipeline.py` :: `merge_additions`. **A0-C5: the hallucination gate runs first and independently of `cofactor_policy`** | — | — | — |
| C-061 | `agent/p52-missing-reactions` | E | R-004 ✔ | **SCOPE ACCEPTED 2026-08-13 (R-004 B-2 only):** `rag/admission.py` :: `parse_span_relation`, `validate_evidence_span`. The stale C-035 claim over both symbols is corrected in this table; C-035 is **not** reopened | — | — | — |

### Requirements carried into specific prompts

- **C-045** — species/organism canonicalization is pre-freeze biology (**D-016**, LOCKED).
  T-102 is **not** narrowed to compounds: it continues to require **both** compound
  identity **and** organism/species equivalence across canonical JSON, PWML **and** SBML,
  and C-045 must not narrow it. Dispatching C-045 requires separate product-owner
  authorization establishing its complete implementation authority.

  **D-016 §9 ownership — reconciled state, `CONTROL-PLANE-RECONCILE-001` §6.** D-016 was
  ratified while `_canonicalize_species_offline` was named in no §9 row, and it says so
  in the present tense (`DECISIONS.md:307`: *"It is currently named in no `MASTER_PLAN`
  §9 row — that ownership gap is closed by this ruling, not deferred."*). **That gap is
  closed and no §9 entry is missing.** The row at **`MASTER_PLAN.md:372` IS the entry**:
  it names C-045 **sole owner** of `ir.py :: _canonicalize_species_offline` under D-016.
  D-016 is LOCKED, its sole-owner assignment is not overridden here, and no second owner
  is created. `DECISIONS.md:307`'s sentence is a statement of fact **as at ratification**
  and is read as history, not as a live claim about the current §9 table; D-016 itself is
  not edited. `SPIKE-002-REPORT.md` §5 (R2) and §8 carried the same stale "owned by
  nobody" reading and are annotated in place.

  **Dependent cards — cross-references, no new ownership.** These consume the seam C-045
  owns and must not reshape it: **C-040** (must not absorb species canonicalization —
  compounds only); **C-050** (pre-freeze call site; a species failure at T-102 is
  attributable to C-045, never to C-050); **C-051** (`ir.py :: build_pwml_ir` contains the
  caller at `:1040-1045`, so it must assert-only and must not re-implement the rewrite);
  **C-052** and **C-020 / T-102** (equivalence must cover organism/species across JSON,
  PWML and SBML). **`rag/eligibility.py` is a distinct seam** and is **not** part of D-016:
  it is §2 read-only material (`:147`) owned by no card. C-056a's organism-comparison
  requirement reuses those audited helpers by import; it does not acquire ownership of
  that file, and any edit to it requires a new owner and a separate product decision.
- **C-020** — equivalence must be proven by parsing and normalizing the JSON, PWML and
  SBML graphs. Comparing one JSON hash to itself is not acceptable evidence.
- **C-011** — receives the same before/after behavioural-equivalence protection as C-012:
  golden artifact-dict comparison, not merely a payload hash.
- **C-055** — every RAG round must re-enter normalization, mapping, gates, persistence and
  classification. A round that retrieves and merges without re-entering all five is a
  failure. The controller must be deadline-aware and checkpoint before each round.
- **C-056a before C-056b** — semantic checks affect runtime `release_status` first;
  benchmark denominators are a separate, later wiring.
- **C-013** — hashing is a versioned canonical projection excluding hash/stamp fields from
  its own input.
