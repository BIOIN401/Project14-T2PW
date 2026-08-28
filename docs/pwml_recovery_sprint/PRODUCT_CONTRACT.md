# PWML Recovery Sprint — Product Contract

**Authority.** When a benchmark result, a test, or an agent's judgement disagrees with
this document, this document wins. A benchmark failure does not by itself justify a
code change.

---

## 1. What the product is

An **evidence-supported pathway reconstruction system**, not a paper transcription tool.

The supplied paper establishes the requested pathway, organism, biological context and
initial evidence. Where the paper is insufficient, the system uses targeted RAG,
trusted biological databases, supporting papers, deterministic inference and bounded
repair passes to fill specific detected gaps.

> **Target behaviour.** Every run produces useful diagnostic artifacts, and every
> biologically recoverable request produces the strongest defensible PWML pathway that
> can be reconstructed from the paper plus traceable external evidence.

A technically recoverable problem must not cause a run to produce no PWML.

### Unacceptable terminal blockers

None of these may end a run without a PWML:

- a stale positional index
- an initial empty `{}` extraction
- a single LLM refusal or malformed response
- a missing or stale gate report
- repairable reference corruption
- an irrelevant degree-zero entity
- an entity groundable through an available database
- an incomplete source paper where targeted RAG can recover the missing information
- a timeout without usable checkpoints or recovery information
- a valid pathway core suppressed because optional peripheral material is unresolved

### The hard limit

The system must **never invent** reactions, directionality, stoichiometry, identities,
enzymes, locations or other biological content merely to guarantee a PWML file.

**A smaller supported pathway is preferable to a larger contaminated one.**

---

## 2. Correctness and depth are separate dimensions

### Correctness — every retained element must be defensible

Defensible means: from the supplied paper, from retrieved supporting evidence, from a
trusted biological database, or from a valid deterministic transformation.

Correct entities and biological identities · no false real identifiers · correct
reactions · correct reactant/product roles · correct directionality and reversibility ·
correct stoichiometry · correct enzyme, modifier, transporter and cargo relationships ·
correct complexes and components · correct cellular locations · correct organism
context · no assay reporters as pathway members · no contextual gene-list neighbours as
participants · no unrelated-pathway reactions · no unsupported retained reactions · no
broken references.

### Depth — may vary with available evidence

Missing detail initiates targeted retrieval and gap resolution. It **does not**
automatically terminate export.

Recovery process:

1. Extract explicit claims from the paper and supplements.
2. Determine the requested pathway and organism context.
3. Identify missing anchors, reactions, participants, identities, directionality,
   locations or connectivity.
4. Form targeted RAG queries for **those specific gaps**.
5. Retrieve from trusted databases and additional papers.
6. Add only content meeting a defined evidence threshold.
7. Remove unsupported or irrelevant content.
8. Revalidate.
9. Repeat within a **bounded** loop until a defensible connected pathway exists or
   retrieval is genuinely exhausted.
10. Generate PWML from the resulting canonical biological graph.

---

## 3. Provenance requirement

The final pathway must distinguish at least:

- explicitly stated in the supplied paper
- added from RAG-supported external literature
- grounded through a biological database
- added through deterministic inference
- added or changed by audit/repair
- unresolved or excluded

Every externally added entity and process identifies: the stage that introduced it, why,
its supporting source or database record, whether it was paper-explicit, its
evidence/support level, and any uncertainty or review requirement.

The schema may differ; the information must be traceable. This exists so that false
content can be attributed empirically to Stage 1, RAG, inference, audit, mapping, gap
resolution or another stage.

---

## 4. Output states

### `release_ready`
Valid PWML with a defensible connected pathway. All blocking technical and biological
checks pass. May be a smaller core, reported with an explicit completeness score.

### `review_required`
Valid, useful PWML produced, but one or more important biological uncertainties are
explicitly identified. Must not be represented as fully confirmed.

### `diagnostic_only`
Recovery and retrieval could not establish a defensible pathway core. All diagnostic
artifacts, partial structured data, evidence, failure reasons and checkpoints are
preserved.

**Disposition (D-065, LOCKED).** `diagnostic_only` covers more than one shape of outcome,
and the gloss above is not true of all of them. Where a defensible connected pathway core
**was** extracted and a correct scope guard stopped the run before audit, DB mapping, freeze
and PWML serialization, the disposition `extracted_not_serialized` names that shape.

It is a **disposition, not a fourth output state.** There are still exactly three output
states and `RELEASE_STATES` is unchanged. A leg carrying this disposition still reports
`status = diagnostic_only`, `strict_gates_passed = false`, `produced_pwml = false` and
`strict_acceptance_eligible = false`, and it is never a strict export. D-062 is not reopened.

**Where it is recorded today, exactly.** The **acceptance record** carries it:
`release_disposition` on the leg, together with the two numbers it was derived from -- the
measured `connected_core_reactions` and that case's own gold floor
`required_connected_reactions` -- plus a `release_dispositions` roll-up on the report. The
**runtime release record** declares a `disposition` field but **no production seam populates
it yet**: the rule needs the gold set's per-case connected-core floor, which is a benchmark
fact no runtime seam holds. Until a chartered card threads that floor to
`batch.driver._finalize_scope_conflict`, a runtime `diagnostic_only` record is emitted with
**no** `disposition` key, and this paragraph -- not the record -- is where a reader of a
runtime manifest row learns that the gloss above may not apply to it.

Absence of the key means **not recorded**. It never means "does not apply".

A no-PWML outcome is **exceptional** and must state exactly: what essential biological
requirement could not be supported; which repair and retrieval steps were attempted;
which evidence was searched; why a smaller valid core could not be exported; why
exporting would require inventing biology.

**Decision (locked):** no separate `core_release_ready` state. A shallow but fully
supported core is `release_ready` with an explicit completeness score. A separate state
multiplies branching for no added information.

---

## 5. Canonical payload requirement

`final_mapped.json` is the primary reproducible deliverable. It represents the final
post-review, post-grounding, post-repair, post-quarantine, gate-passing biological graph
used to create PWML and SBML.

Reloading `final_mapped.json` and exporting it again must produce a **biologically
equivalent** PWML pathway.

Byte-identical XML is not required. Acceptable to differ: XML ordering, whitespace,
generated internal XML IDs, timestamps, non-biological layout metadata.

Must remain equivalent: reactions · reactants and products · directionality and
reversibility · stoichiometry · enzymes and modifiers · transports and cargo · entities
and biological identifiers · complexes and their components · cellular locations ·
process-to-entity references · organism context.

**Exporters must not independently add, remove, resolve or reinterpret biological
content after the canonical graph is frozen.** If an exporter must perform a biological
mutation, then `final_mapped.json` is not canonical and the mutation must move upstream
of the freeze.

**Equivalence must be proven by parsing and normalizing the JSON, PWML and SBML graphs
and comparing the dimensions above.** Comparing one JSON hash to itself proves nothing
and is not acceptable evidence.

---

## 6. Hashing

Two versioned canonical projections, each excluding all hash and stamp fields from its
own input (`payload_sha256`, `canonical_graph_sha256`, `canonical_payload_sha256`,
`hash_schema_version`, `report_schema_version`, `phase`, `artifact_set_version`). A hash
is never an input to itself.

| Hash | Covers | Bound by |
|---|---|---|
| `canonical_graph_sha256` | biological content only — entities, identifiers, reactions, roles, stoichiometry, directionality, complexes, locations, references | **exporters** |
| `canonical_payload_sha256` | the complete saved `final_mapped.json` including lineage, evidence, confidence, provenance | **persisted-artifact integrity** |

Two guarantees: the exported biological graph has not changed; the evidence record
supporting it has not been altered.

The projection is an **allowlist**, never a denylist — a denylist silently admits every
future field into the graph hash.

Lineage must not change graph equivalence, but lineage changes must remain detectable.

Historical hashes keep their existing meaning under `hash_schema_version`. Never
silently redefine `payload_sha256`.

---

## 7. Coverage policy

`requested_core_coverage_below_minimum` **triggers targeted retrieval before
classification**. It is not, by itself, a refusal.

After bounded retrieval is exhausted:

- surviving fragment biologically correct, internally connected, representable without
  guessing → **`review_required` PWML**; not `release_ready`; not strict benchmark
  success. Record completeness, missing anchors, retrieval attempts, and why further
  supported expansion failed.
- no defensible connected core, or serialization would require invention →
  **`diagnostic_only`**.

The coverage threshold blocks release-ready status, not PWML production. **The threshold
value itself does not move.**

---

## 8. Identity verification

Order: accession already present on the entity → PathBank by exact identifier → if
PathBank evidence is inadequate, UniProt by exact accession → cache and persist the
record and its response provenance → convert to an **immutable**
`identity_evidence_candidate` → run the existing species, name/gene, score and conflict
checks **unchanged** → materialize the verified identity in `final_mapped.json`.

**Exporters perform no network or database lookups.** UniProt must not be required to
re-export saved canonical JSON.

`verification_status` is distinct from `identity_status`:

| Value | Meaning | Consequence |
|---|---|---|
| `verified` | evidence retrieved and confirms the claim | accession retained, materialized |
| `rejected` | evidence retrieved and contradicts the claim | **the only case where identifiers may be stripped** |
| `unavailable` | network/DB failure | accession preserved as `unverified_claim`; not promoted; **not erased**; export identity degrades to placeholder if a verified identity is essential |
| `not_evaluated` | the ladder did not run | never treated as `false` |

**A lookup failure is not evidence that an accession is false.**

Never accept an identifier because its format is valid.

---

## 9. Timeout and budget

`leg_timeout_seconds` defaults to **3600 s**; the 120 s parent/child grace is preserved,
giving the child **3480 s**. Per-leg overrides may exist but must be explicit and
recorded in the run manifest — no silent extension of difficult benchmark legs.

All stages use one monotonic per-leg deadline and record elapsed and remaining budget.
No LLM call starts unless its configured maximum duration plus the downstream
finalization reserve fits the remaining budget. The reserve is configurable and
eventually calibrated from recorded stage runtimes; it covers checkpoint persistence,
validation, status classification and diagnostic-artifact writing. A checkpoint is
persisted before any potentially long LLM or retrieval call.

**Stage-1 extraction: at most three total model attempts, including the first.**
1 normal · 2 structurally-empty repair · 3 a materially different strategy (narrower
section-based extraction or an alternate model), only if budget remains.

If attempts 1 and 2 return the same empty response hash, the same prompt must **never**
go to the same model a third time. Spend the remaining budget on a genuinely different
mechanism: narrower extraction, deterministic text recovery, supplemental-material
retrieval, or targeted RAG reconstruction.

The attempt cap is a safety ceiling, not a promise that all attempts run.

### Termination reasons — never conflated

| Reason | Means |
|---|---|
| `retrieval_exhausted` | the configured retrieval ladder **actually completed** and produced no admissible new claims |
| `no_new_claims` | retrieval completed but did not expand the graph |
| `budget_exhausted` | another recovery step might have helped; wall-clock did not allow it |
| `operation_timeout` | an individual external operation exceeded its deadline |
| `identical_empty_response` | repeated extraction returned the same structurally empty response |
| `scientifically_unrecoverable` | evidence sources exhausted; no defensible pathway core |

### On timeout or budget exhaustion, preserve

last completed stage · current structured payload · all retrieved evidence · attempt
numbers, prompts/models and response hashes · elapsed and remaining budget · the next
recovery step that was skipped · the exact stop reason.

Classification after budget exhaustion follows the graph that already exists: correct
connected core → `review_required` PWML; no defensible core → `diagnostic_only`; a
budget-limited run can **never** be `release_ready` while required checks or recovery
steps remain incomplete.

### Denominators

`budget_exhausted` counts as an **operational failure** in pipeline-completion and
end-to-end strict-success metrics. It must never be relabelled semantic failure,
scientific insufficiency, or retrieval exhaustion. Semantic metrics record
`not_evaluated` where semantic evaluation never ran and report those legs separately.

---

## 10. RAG

Each retrieved reaction may enter the pathway only when it fills a **specific detected
typed gap** and passes pathway, organism and evidence admission.

Every RAG round must **re-enter normalization, mapping, gates, persistence and
classification**. A round that retrieves and merges without re-entering all five is a
failure regardless of what it retrieved.

The loop is bounded, deadline-aware, and checkpoints before each round. It stops with an
explicit reason from §9. Deduplication is against **all claims ever seen**, not only
admitted ones, or judge-rejected claims recur every round and the loop never converges.

A rejected RAG claim must not be reintroduced by a later stage.

Supporting passages and source identifiers are retained through final export.

---

## 11. Semantic evaluation

Semantic checks must affect the **runtime `release_status`**. Wiring them only into
benchmark denominators is insufficient.

The pipeline must distinguish, without collapsing any into another:

- pipeline execution succeeded
- strict technical gates passed
- semantic evaluation passed
- semantic evaluation failed
- semantic evaluation **was not performed**

`not_evaluated` is never `false`.

---

## 12. `partial_only`

`expected_export = partial_only` is a statement about the **source** and about the
strict benchmark denominator. It is **not** a prohibition on generating PWML.

- generate useful research or diagnostic artifacts where possible
- exclude such papers from the strict-PWML success denominator
- do not label them release-ready unless they independently meet the standard
- do not embed gold-set-only policy into the general production pipeline

`expected_export` must be an explicitly required gold field; a silent default moves
papers out of the strict denominator without anyone deciding to.

---

## 13. Standing policy positions

| Item | Position |
|---|---|
| `placeholder_backed_proteins` (21 in the pinned run) | **Standing disagreement**, not a defect. The gold set counts it as an error class; the pipeline treats `Unknown`-backed export as legitimate biology preservation. No agent may "fix" it. Escalate. |
| PMC12452463 | Gold `export_rationale` records the route as chemically **broken** (EntA absent; nothing converts 2,3-dihydro-2,3-dihydroxybenzoate onward). Correct outcome after the index fix is `review_required` with `strict_acceptance_eligible=false`. **Never strict success.** |
| Negative controls | PMC13231680 and PMC12180156 are `mechanistic_relevance=context_only` by design. PMC13231680's rationale calls an empty pathway plus a rejection reason the *correct* outcome. |
| Artifact naming | `pathway.pwml` = `release_ready` only. `pathway.review_required.pwml` = valid, needs review. No final PWML for `diagnostic_only`. Batch artifact set only — the interactive download and `outputs/` are unchanged. Structured status is authoritative; the filename is a migration aid. |

---

## 14. Adjudication rule

Before proposing a code change justified by a benchmark failure, classify it:

- `product_contract_violation` — **only this justifies code**
- `gold_data_defect`
- `policy_disagreement`

cite the gold `relevance_note` / `export_rationale`.

---

## 15. The O-1 acceptance instrument (D-070, D-073, D-074)

**`placeholder_backed_proteins` keeps its value and its meaning.** C-101 splits it
beside itself; it never renames, repurposes or recounts it. Section 13's standing
disagreement stands: none of these rows is a forged identity.

**The split is a partition, not an assumption.** `placeholder_backed ==
placeholder_sentinel_rows + placeholder_generated_wrappers + placeholder_other_rows`.
`other` is REPORTED, never assumed zero. It is 0 on the pinned run and the partition is
exactly 16/5 — a *measured fact about that run*, not a structural guarantee. An
instrument that asserted the invariant by dropping the remainder would hide the first row
that does not fit. Assignment is mutually exclusive and ordered: sentinel, then wrapper,
then other, because `is_pathbank_unknown_protein` names one database record and is the
narrower statement.

**F-141 is a different seam and is never reported under the O-1 name.**
`withheld_identity_correct` and `withheld_identity_recoverable` count candidate identities
that survived the identity verdict and were not shipped. Both are measurements even at
zero — see § 8's `not_evaluated`-is-never-`false` rule, which
`withheld_identity_evaluated` implements. **A row that correctly withholds a
species-specific identifier is not an error anywhere else**: do not penalise the pipeline
twice for obeying the contract.

**Priority 1 reports a raw count and an accepted count.** Raw is preserved and unchanged
in meaning. Accepted is the contract-adjusted result after authorized, case-scoped
tolerances, and its status is `PASS` (0–6), `PASS_WITHIN_VARIANCE` (7) or `FAIL` (8+).
**Six remains the target**; seven is a one-finding stochastic band and never evidence that
a defect is fixed. `PASS_WITHIN_VARIANCE` stays visibly distinct from `PASS` and must not
collapse into any summary, badge or Boolean — which is why the absolute `ok` is left
computing zero-tolerance on the raw count. Do not rerun to chase a favourable draw.

**Under D-074 as ruled, no Priority-1 row can be contract-adjusted, so `accepted` is
identically `raw` today.** The licence covers only the *bare* sentinel, and a bare row
carries no accession, while a Priority-1 finding requires one — so the two sets cannot
intersect. That is the ruling's shape and not a defect: the seam is wired, tested and
pinned so a future licence has somewhere to land, and loosening the bareness guard to make
it fire would be broader than the ruling and is refused. **A zero here means "no licence
can reach this", not "none was measured"** — the same distinction `placeholder_other_rows`
and `withheld_identity_other` exist to preserve.

**A tolerance is a licence and has to be spelled out.** PMC12444477's `Unknown` tolerance
is **row-predicated**, declared in `unknown_backed_tolerated_sentinel` and kept out of the
name-keyed list on purpose: a name-keyed `Unknown` would excuse any row on that paper
carrying the string. The scorer passes the whole row, the authoritative
`is_pathbank_unknown_protein(row)` predicate must agree, and the gold's own declared
record identity is checked independently of it. **`LpxH` is not covered and remains a
Priority-1 finding** — PMC12444477 goes 9 → 8, never 9 → 7. Widening it is the
merge-rule-6 direction and is a reject.
