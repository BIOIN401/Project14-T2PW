# PWML Recovery Sprint — Decisions

Append-only. Product-owner decisions only. Agents read; they never write here.

---

## D-001 — Lineage and payload hashing · 2026-08-05 · LOCKED

Two versioned hashes, not one, and not an exclusion.

- `canonical_graph_sha256` — biological content only: entities, identifiers, reactions,
  roles, stoichiometry, directionality, complexes, locations, references.
- `canonical_payload_sha256` — the complete saved `final_mapped.json`, including lineage,
  evidence, confidence and provenance.

Exporters bind to the graph hash. Persisted-artifact integrity binds to the payload hash.

Both are **versioned canonical projections that exclude all hash and stamp fields from
their own input** — a hash is never an input to itself. The graph projection is an
allowlist, never a denylist.

Lineage must not change graph equivalence; lineage changes must remain detectable.

Historical hashes keep their existing meaning. Add `hash_schema_version` rather than
silently changing what `payload_sha256` means.

**Implements:** C-013.

---

## D-002 — Coverage below minimum · 2026-08-05 · LOCKED

`requested_core_coverage_below_minimum` triggers targeted retrieval **before**
classification. It is not, in itself, a refusal.

After bounded RAG is exhausted: a biologically correct, internally connected fragment
that is representable without guessing → `review_required` PWML. Not `release_ready`. Not
strict benchmark success. Record completeness, missing anchors, retrieval attempts, and
why further supported expansion failed.

`diagnostic_only` is reserved for cases with no defensible connected core, or where
serialization would require invention.

The threshold blocks release-ready status, not PWML production. **The threshold value
does not move.**

**Reference case:** PMC12096016 at 0.167 coverage after the C-010 fix.
**Implements:** C-041, C-055.

---

## D-003 — UniProt in the identity ladder · 2026-08-05 · LOCKED

Approved as a **bounded, cached, pre-freeze evidence source**. Not a hidden live
dependency inside PWML export.

Flow: accession already on the entity → PathBank by exact identifier → UniProt by exact
accession if PathBank evidence is inadequate → cache and persist the record with response
provenance → immutable `identity_evidence_candidate` → existing species, name/gene, score
and conflict checks **unchanged** → materialize in `final_mapped.json`. Exporters perform
no network or database lookups.

Failure semantics: network failure is not evidence that an accession is false. Preserve
the submitted accession as an unverified claim; do not promote it; do not erase it.
Record `verification_status: unavailable` or `not_evaluated`, distinct from `rejected`.
Degrade the export identity to a placeholder if a verified identity is essential and
unavailable.

UniProt must not be required to re-export saved canonical JSON.

**Implements:** C-033.
**Note:** `_fetch_uniprot_enrichment` (`enrich_entities.py:507`) and `EnrichmentCache`
already provide the fetch and the cache. C-033 relocates them to the ladder; it does not
build them.

---

## D-004 — `review_required` artifact naming · 2026-08-05 · LOCKED

- `pathway.pwml` — `release_ready` only
- `pathway.review_required.pwml` — valid but requiring biological review
- no final PWML for `diagnostic_only`, unless a separately named diagnostic draft is
  genuinely useful

Manifest records structured status independently:

```json
{
  "pipeline_status": "pass",
  "release_status": "review_required",
  "pwml_artifact": "pathway.review_required.pwml",
  "strict_acceptance_eligible": true,
  "strict_acceptance_passed": false,
  "completeness": 0.167
}
```

Artifact classification will eventually depend on structured status rather than filename.
Distinct names are the safe migration while existing code still equates `pathway.pwml`
with strict success — four independent sites do so today (`runner.py:116`, `:856`;
`driver.py:1319`, `:2008`; `acceptance.py:88`, `:497`).

Scope: **batch artifact set only.** The interactive download and `outputs/` are unchanged.

**Implements:** C-053.

---

## D-005 — Timeout budget and extraction escalation · 2026-08-05 · LOCKED

`leg_timeout_seconds` default **3600 s**; 120 s parent/child grace preserved → child
deadline **3480 s**. Per-leg overrides must be explicit and recorded in the run manifest;
no silent extension of difficult benchmark legs.

One monotonic per-leg deadline across all stages, recording elapsed and remaining budget.
No LLM call starts unless its configured maximum duration plus the downstream
finalization reserve fits the remaining budget. The reserve is configurable and eventually
calibrated from recorded stage runtimes; it covers checkpoint persistence, validation,
status classification and diagnostic-artifact writing. Checkpoint before any potentially
long LLM or retrieval call.

Stage-1 extraction: **at most three total attempts including the first.** If attempts 1
and 2 return the same empty response hash, never issue the same prompt to the same model
a third time — that strategy has demonstrated it is inert. Spend the remaining budget on
a genuinely different recovery mechanism.

The attempt cap is a safety ceiling, not a promise. Insufficient budget for the next rung
→ stop, record `budget_exhausted`.

Six termination reasons, never conflated: `retrieval_exhausted` · `no_new_claims` ·
`budget_exhausted` · `operation_timeout` · `identical_empty_response` ·
`scientifically_unrecoverable`.

Denominators: `budget_exhausted` is an operational failure in pipeline-completion and
end-to-end strict-success metrics; never relabelled semantic failure, scientific
insufficiency, or retrieval exhaustion. `retrieval_exhausted` may be claimed only when
the configured ladder actually completed.

**Order is mandatory:** budget/checkpoint infrastructure (C-032) **before** the escalation
ladder (C-042). Do not implement a fixed number of unconditional retries.

**Blocked by:** `OpenAI(...)` currently has no `timeout=`; worst-case one `chat_detailed`
≈ 4871 s against a 3480 s deadline. C-014 must land first.

---

## D-006 — Semantic evaluation reaches runtime · 2026-08-05 · LOCKED

Semantic checks must affect the actual runtime `release_status`. Wiring them only into
benchmark denominators is insufficient.

C-056a (runtime) precedes C-056b (benchmark denominators).

The pipeline must distinguish, without collapsing any into another: pipeline execution
succeeded · strict technical gates passed · semantic evaluation passed · semantic
evaluation failed · semantic evaluation **was not performed**.

---

## D-007 — Biological equivalence is proven by parsing · 2026-08-05 · LOCKED

Test equivalence by parsing and normalizing the JSON, PWML and SBML graphs and comparing
the dimensions in `PRODUCT_CONTRACT.md` §5.

Comparing the same JSON hash twice proves nothing and is not acceptable evidence.

**Implements:** C-020, verified at T-102.

---

## D-008 — RAG rounds must re-enter the pipeline · 2026-08-05 · LOCKED

Every RAG round must re-enter normalization, mapping, gates, persistence and
classification. A round that retrieves and merges without re-entering all five is a
failure regardless of what it retrieved.

C-055 therefore depends on C-032 (deadline/checkpoint infrastructure) in addition to
C-043 and C-041.

---

## D-009 — Wave A0 is not dependency-free · 2026-08-05 · LOCKED

The earlier claim of a zero-dependency Wave A was inaccurate. Split:

- **A0** — genuinely independent branches.
- **A1** — C-020 (needs C-013), C-021 (needs C-015).

---

## D-010 — Wave E prompts are generated, not pre-written · 2026-08-05 · LOCKED

C-060 and C-061 are placeholders. They are not dispatchable until R-003 and R-004 deliver
exact findings, affected files, expected corrections and regression fixtures. Their prompt
bodies are generated *from* those reports.

---

## D-011 — O-2 resolved: ignore `runs_verify/*/cache_snapshot/` · 2026-08-05 · LOCKED

Add a **narrowly scoped** `.gitignore` rule for `runs_verify/*/cache_snapshot/`.

Measured at decision time: 8 directories × 38 MB = **304 MB**, 2 files each, all
untracked, against a 159 MB `.git`. Every byte is a regenerable copy of
`data/enrichment_cache.json` + `data/id_mapping_cache.json`.

Scope is deliberately narrow. The **16 `cache_snapshot` files already tracked under the
older `runs/`** stay tracked — the rule does not reach them, and rewriting history is not
what was decided.

**The existing directories are NOT deleted.** Their size is reported; deletion is a
separate, recoverable housekeeping decision that has not been taken.

**Applied:** `.gitignore` at the INIT-001 follow-up commit.

---

## D-012 — Baseline cohort is frozen to a manifest · 2026-08-05 · LOCKED

The replay merge gate must **not** simply be re-pinned from 23 to 39 while it still
globs the filesystem.

Replace the dynamic cohort with an **explicit, version-controlled manifest** of the 39
currently intended pre-implementation legs. Every entry is verified before freezing. The
merge-gate test reads **only** the manifest, so a newly generated benchmark directory
cannot silently redefine the baseline — the failure mode that made `FULL_STACK_BASELINE`
stale on `ORIGIN_SHA` (23 pinned, 39 measured; see `BASELINE.md` § 5).

New legs enter the baseline **only** through an intentional manifest update with review.

C-010's six expected changes are preserved **separately** (`BASELINE.md` § 6) so its
intended delta can never be confused with this unrelated repair. The populations differ:
the allowlist spans `runs/` **and** `runs_verify/` (32 legs); this gate reads `runs/`
only (39 legs). Exactly **two** of the six fall inside the gate's cohort.

This is pre-Wave-A0 **test-harness maintenance**: implementation by subagent, independent
review, focused tests, and a passing baseline gate **before C-010 is dispatched**.

**Implements:** H-001. **Blocks:** C-010 and all of Wave A0.

---

## D-013 — Replay assertion is scoped to payloads · 2026-08-05 · LOCKED

The nine `key_compounds` occurrences in `stage0_attempts.json` are **diagnostics, not
pathway payloads** — 0 payload files carry it.

Narrow the replay assertion so it examines the authoritative payload files and excludes
diagnostic attempt records. **No pathway biology and no production behaviour may be
altered to make this test pass.** Independent review and focused verification required.

`MASTER_PLAN` § 1's `pathway_context=None` premise is unaffected and remains correct.

**Implements:** H-002.

---

## D-014 — O-3 resolved: scratch files are protected · 2026-08-05 · LOCKED

Leave all **seven** existing tracked modifications exactly unchanged for the duration of
the sprint:

```
data/enrichment_cache.json   data/id_mapping_cache.json   out/enrichment_dump.json
outputs/pathway.pwml   tmp/draft_graph.json   tmp/qa_report.json
tmp/reaction_summary.txt
```

Do **not** stage, commit, reset, restore, stash, regenerate or reformat them. They are
protected developer scratch state. This strengthens TRAP-5 from "never commit a cache
modification" to "never touch any of these seven, by any means".

---

## D-015 — Compound canonicalization is pre-freeze · 2026-08-05 · LOCKED

**Compound name and identity canonicalization is part of the canonical biological
representation, even when it introduces no new reaction.** It must occur deterministically
**before** the canonical payload is frozen.

Adopt the `LIFT_WITH_ADAPTER` direction from SPIKE-002. The pre-freeze operation must:

- produce the resolution report;
- use an **explicit, unambiguous rename map**;
- **atomically** propagate each rename to **every** process participant reference;
- preserve the original supported name as an **alias or synonym** where appropriate;
- preserve **reaction count and participant connectivity**;
- **fail visibly** on ambiguous or dangling references;
- avoid inventing identity or any other biology;
- finish **all network-dependent resolution before the freeze**.

After the freeze, PWML and SBML exporters must not query external services, materialize new
identity fields, rename entities, or reinterpret biology.

This closes the question SPIKE-002 § 10 item 1 raised — whether canonicalization is biology
and where it belongs. It is biology, and it belongs upstream of the freeze.

**Implements:** the reshaped C-040 / C-050 / C-051 chain.
**Evidence:** `SPIKE-002-REPORT.md` § 2, § 5 (R1), § 6, § 7.

---

## D-016 — Species canonicalization is pre-freeze, and T-102 keeps species · 2026-08-05 · LOCKED

**Species and organism context are also part of the canonical biological representation.**
`_canonicalize_species_offline` must therefore be moved into an **explicitly owned pre-freeze
task**. It is currently named in no `MASTER_PLAN` § 9 row — that ownership gap is closed by
this ruling, not deferred.

**T-102 equivalence is NOT narrowed to compounds.** It must verify **both** compound identity
**and** organism/species equivalence, across canonical JSON, PWML **and** SBML.

This resolves SPIKE-002 § 10 item 2 against the alternative of scoping T-102 to compounds.

**Evidence:** `SPIKE-002-REPORT.md` § 5 (R2) — the same post-freeze rewrite mechanism on
organism context, a `PRODUCT_CONTRACT` § 5 must-remain-equivalent dimension.

---

## D-017 — The bounded wrapper's output forwarding is a G11 defect · 2026-08-05 · LOCKED

A job that is killed by its own wrapper and leaves **no cleanup report** is uncertifiable
under G11. Repairing that is pre-Wave-A0 harness work, not a nicety.

Temporary ownership of `evidence/bounded_run.py` and `evidence/bounded_run_selftest.py` is
granted to `pwml-implementer` **for that task only**; the grant expires with it.
`batch/runner.py` remains C-032's and must not be touched.

**Implements:** H-001's successor task **H-003**, merged at `aab975a`.
**Accepted rulings, not to be reopened without new evidence:** the ~441-line change against
the ~400 estimate is acceptable and recorded as a deviation; the cleanup-`finally` guard is
retained because observation itself can fault; the child's **real** exit code is retained
when the `--json` destination is unwritable — a synthetic infrastructure code would lose
exactly what the requirement protects.

---

## D-018 — G11 evidence is durable, version-controlled and prospective · 2026-08-05 · LOCKED

Every future test, benchmark, pipeline leg or LLM-backed job receives a **unique `--json`
report path under a version-controlled evidence location**, so a G11 claim is checkable
against a committed artifact rather than a pasted table.

Compliance logic must:

1. require the expected artifact to exist **independently of any wrapper or child exit
   code** — a missing artifact cannot pass because an exit code happens to be acceptable;
2. require `cleanup_success: true` — **`final_surviving_count: 0` alone is insufficient**,
   because the count can keep its default while a child is still alive;
3. validate the expected report schema and required fields;
4. **not** treat `json_report_written: false` from a direct `run()` caller as a wrapper
   violation when `main()` and `--json` were never invoked;
5. commit only **credential-free, bounded** evidence — never captured bulk output or caches.

**Prospective only.** Historical runs that lack reports are **not** backfilled. A
reconstruction produced after the fact is not evidence of the original run, and the record's
statement that the pre-A0 jobs cannot be reconstructed stands.

**Implements:** **H-004**, merged at `a04a0aa`.
**Known follow-ups, deliberately not implemented out of boundary:** `CleanupReport` carries
no `schema_version` and no `wrapper_build` field. Both require `bounded_run.py`, which H-004
did not own. Until a `wrapper_build` field exists, no artifact can prove which wrapper
produced it — git archaeology is an inference from the tree, not proof from the artifact.

---

## Open — not yet decided

| # | Question | Blocks | Why it cannot be answered from the repository |
|---|---|---|---|
| O-1 | `placeholder_backed_proteins` (21 in the pinned run): gold-set error class, or legitimate biology preservation? | any branch that touches protein export policy | It is a genuine disagreement between two intentional designs, not a defect. TRAP-3 forbids agents from resolving it. |

**Closed:** O-2 → D-011 · O-3 → D-014.
