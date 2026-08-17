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

## D-019 — Card-specific change budgets replace the universal `~400` threshold · 2026-08-07 · LOCKED

The universal approximate **`~400` changed-line** threshold in `[S4]` and `_TEMPLATE_IMPLEMENT.md` is **prospectively superseded, for newly authorized implementation cards, by card-specific declared budgets.** It was wrong in every G11 card that met it (H-001 ≈637, H-003 441, H-004 917, H-006 572), because generated evidence and atomically-required tests were charged against a bound written for hand-authored code alone.
**Two budgets, declared before dispatch, never one.** **Hand-authored:** exact allowed manifest plus maximum additions-plus-deletions. **Machine-generated evidence, budgeted separately:** maximum artifact count **and** a size limit (bytes or changed lines), stated as an explicit zero when generation is unauthorized.
**Acceptance-criterion atomicity is an input to initial scoping**, not an excuse discovered afterwards: if the card's own ACCEPTANCE list admits no smaller unit, the budget accommodates it or the card is re-scoped before dispatch.
**A card may be split only at boundaries where each half is independently implementable and independently validatable, and a budget must never force the merge — or the leaving behind — of an unvalidated semantic half.** Shipping a mechanism whose validating tests land later is forbidden, not merely discouraged.
**A predicted or actual overrun requires stopping before commit.** An over-budget commit may be created only after **renewed explicit authority carrying a revised budget**; no implementer, reviewer or test-runner may self-authorize one. **Technical success or later approval cannot retroactively cure an unauthorized overrun** — approval speaks to the diff, never to the authority to have created it.

**Effective when the commit implementing this decision is accepted and merged into `sprint/pwml-recovery`. Prospective only.** It does **not** cure, waive or reclassify **H-006** or any earlier event; every recorded procedural status stands exactly as written. **D-017 remains unchanged and is not reopened** — its ruling on H-003's ~441 lines is a historical ruling about H-003. `SPIKE-002-REPORT.md`'s C-040 pre-split finding stands, now subject to the split clause above.

---

## D-020 — C-011's Chunk D `176 passed / 1 failed` is a ratified environment-specific baseline exception · 2026-08-10 · LOCKED

The C-011 merge (`0182eae`) recorded Chunk D at **176 passed / 1 failed** against the card's
ACCEPTANCE line of 177, and merged on an exact documented delta rather than a green run. That
result is **retrospectively accepted as a product-owner-ratified, environment-specific baseline
exception.** Authority: `CONTROL-PLANE-RECONCILE-001` §2.

**It is accepted because the recorded evidence establishes six independent facts, not because a
number was waved through.** The first four are the four cells of a **fully populated 2×2**, which
is what removes C-011 from the causal chain in both directions:

1. **The same failure occurs WITHOUT C-011** — report `07-chunkd-base`, `repo_head 361b158`,
   `repo_tracked_files_dirty: false`, identical selection, **exit 1**.
2. **The failure can disappear WITHOUT C-011** — the integration control at `85fae43`, main
   checkout, **exit 0**. **Provenance, stated because it differs from the other five:** this fact
   has **no committed G11 artifact**. The control run wrote its `--json` cleanup report to the
   durable checkpoint **outside this repository**, because the C-011 branch manifest was closed
   and adding a file to it would have been an artifact-accounting violation. Fact 2 therefore
   rests on the C-011 merge record plus an **out-of-repository wrapped report**, not on an
   artifact in the tree. **Facts 1, 3 and 4 are artifact-backed** — all eight exit codes were
   reproduced from committed reports. **Facts 5 and 6 are not**: the observed traceback and the
   isolated-run result appear in no committed artifact (see fact 6's own qualification).
3. **The failure can occur WITH C-011** — reports `04`, `25`, `26`, `44`, `45`, and three
   independent reviewer runs.
4. **The failure can disappear WITH C-011** — reports `05` and `18`, **exit 0**.
5. **The traceback contains no C-011 frame.** `streamlit_app.py:6187` →
   `tools/pathwhiz_converter/ui.py:26` (`st.subheader`) → `script_run_context.py:144` →
   `RuntimeError: FragmentThreadState not initialized`. The abort is **2,585 lines past the seam**,
   in a module C-011 never touches. The repository already adjudicates this shape at
   `tests/test_streamlit_quarantine_boundary.py:425-430` as "a Streamlit harness fault, not an app
   fault".
6. **Isolation was observed green once** — the two files at 27/27, exit 0, at that exact tip —
   **but isolation is not a reliable remedy and this fact must not be read as establishing one.**
   Six later isolated runs of `test_streamlit_quarantine_boundary.py` alone gave **two green
   (23 passed) and four red (2 failed / 21 passed, four different failure pairs)**;
   `test_streamlit_stage8_export_contract.py` alone is stable at 4 passed. The fault is
   **intra-file** — one file builds 23 `AppTest` objects in a single process — so no per-file
   process partition can remove it. No committed artifact records an isolated run: every
   committed chunk D report runs the full seven-file selection (verified at `0182eae`). **This
   narrows fact 6 only.** Facts 1–5 carry the ratification on their own and the conclusion that
   C-011 did not cause the failure is unchanged.

There is **no correlation in either direction**, the failing test is **not stable across runs**
(four different tests failed across the runs, and the failure count varied with nothing but the
environment — 10 occurrences / 4 failures in one run, 2 / 1 in the next), and a defect in the seam
would fail deterministically on the same test.

**No test was rerolled, retried-until-green, deselected or excluded to manufacture this result.**
Chunk D was not re-rolled until it went green; the run count stands as measured. That assurance is
part of the ratification: had a green run been manufactured, this exception would not be available.

**What this decision is NOT.**

- It is **not permission to weaken any future acceptance criterion.** A card's stated pass count
  remains its stated pass count.
- It is **not permission to classify an actual regression as transient.** A failure may be called
  environmental only where a comparably complete 2×2 is populated **with its own committed
  evidence** — the same failure demonstrated without the change, and the change demonstrated
  without the failure — plus a traceback that does not implicate the change. Absent that, a red
  test is a red test.
- It is **not retroactive cure of anything else**, and it creates no precedent for merging on a
  narrative instead of a run.
- It **changes nothing about C-011**: its production code, tests, fixture, git history, reviewer
  verdict and merge all stand exactly as recorded.

**The required forward Chunk D gate — stated as an obligation, because it does not yet exist.**
Under `CONTROL-PLANE-RECONCILE-001` §3 the reconciliation's **Chunk D lane** is to define a
split-process Chunk D gate in `TEST_MATRIX.md`. **That definition has not merged**: verified at
`0182eae`, `TEST_MATRIX.md` contains **zero** occurrences of "split-process" and the Chunk D
lane's branch `agent/recon-chunkd-gate` is still at `0182eae` with no commit, so the gate exists
in **no ref in this repository**. This decision therefore does **not** lock a gate it cannot
point at. It locks two things instead:

1. **The obligation.** The authoritative forward Chunk D gate **shall be** the split-process gate
   defined in `TEST_MATRIX.md` by the Chunk D lane. Its definition, contents and runner internals
   are that lane's to write and are neither described nor constrained here.
2. **The effective date.** That gate is **not in effect until its definition merges** into
   `sprint/pwml-recovery`. Until then the standing Chunk D expectation is unchanged, and this
   exception governs **one historical result only** — the C-011 merge — and licenses nothing
   forward.

**Merge-order constraint, recorded so the reference resolves.** The Chunk D lane merges **before**
this decision. If that order is ever inverted, this paragraph is a forward reference to an
unmerged definition and must be read as the obligation in (1), never as a citation.

---

## D-021 — C-040 / C-050 / C-051 ownership lock · 2026-08-10 · LOCKED

The compound-resolution chain touches one lifecycle seam from three directions. This decision
partitions it **before** dispatch so that no two cards independently reshape the same seam.
Authority: `CONTROL-PLANE-RECONCILE-001` §8. The partition is **conservative**: it is derived
entirely from the existing card purposes, `MASTER_PLAN` §9 (`:367`, `:373`, `:374`), SPIKE-002 and
the locked decisions **D-015**, **D-016** and **D-019**. **It invents no product behaviour.**

**None of C-040, C-050 or C-051 may begin — no dispatch, no branch, no worktree, no
implementation — until this decision is merged into `sprint/pwml-recovery`.**

### 1. Responsibility of each card

**C-040 — extract, do not call.** Lift the **four** functions `SPIKE-002-REPORT.md` §3 enumerates
— `_normalize_compound_external_ids`, `_compound_external_ids`, `_canonicalize_compound_offline`
and `_resolve_compound_rows`, **189 lines moved** — out of `ir.py` into a new module and give them
the three-part SPIKE-002 adapter contract. Mechanical extraction under the `LIFT_WITH_ADAPTER`
verdict. C-040 **wires nothing** and changes no caller's behaviour.

**`MASTER_PLAN` §9 `:367` names only two of those four and is incomplete.** §2 below, not `:367`,
is C-040's authoritative symbol manifest. Dispatching C-040 against the two-function row would
authorize an implementer to touch two functions while the extraction mandatorily requires four
plus one reference repair — the boundary escalation §5 calls a dispatch error. All line ranges in
§2 were re-verified against `src/t2pw/pwml/ir.py` at integration `0182eae`.

**C-050 — call it pre-freeze, in the enrichment block above the seam.** Perform compound identity
resolution **before** the canonical payload is frozen, per **D-015**, with the rename propagated
atomically to every process participant reference.

**C-051 — assert only.** Delete the now-redundant in-IR resolution call and replace it with a
**fail-closed assertion** that every compound row already carries a resolution verdict. C-051
resolves nothing and repairs nothing.

### 2. Exactly one primary owner per production symbol

| Symbol / surface | Primary owner | Not owned by |
|---|---|---|
| NEW `src/t2pw/pwml/compound_resolution.py` (whole module) | **C-040** | C-050, C-051 |
| `ir.py :: _normalize_compound_external_ids` (`:530-555`) — **moved** | **C-040** | C-050, C-051 |
| `ir.py :: _compound_external_ids` (`:558-575`) — **moved** | **C-040** | C-050, C-051 |
| `ir.py :: _canonicalize_compound_offline` (`:578-621`) — **moved** | **C-040** | C-050, C-051 |
| `ir.py :: _resolve_compound_rows` (`:797-897`) — **moved** | **C-040** | C-050, C-051 |
| `ir.py :: _emit_canonicalization_preflight` (`:900-963`) — **STAYS**. C-040 owns the **`:920` re-import reference plus the one module-header import line it requires — and nothing else** | **C-040**, minimum surface only | C-050, C-051 |
| `ir.py :: _entity_record` (`:437-449`) — stays, unmodified by all three | **no card** — untouched | all three |
| The private leaf-helper copies inside the new module (originals `ir.py:43-96`, `:183-193`, `:244-260` **stay unmodified**) and their equality pin | **C-040** | C-050, C-051 |
| The `apply_canonical_name` keyword-only parameter on both moved entry points, and the resolution-report shape | **C-040** | C-050, C-051 |
| `ir.py :: build_pwml_ir`, **including the resolution call site at `:1106-1114`** | **C-051** | C-040, C-050 |
| `streamlit_app.py` :: the enrichment block **above** the C-011 seam, and the pre-freeze call | **C-050** | C-040, C-051, C-052 |
| `ir.py :: _canonicalize_species_offline` | **C-045** (D-016, unchanged) | C-040, C-050, C-051 |
| `streamlit_app.py :: freeze_canonical_payload` | **C-030**, then **C-052** (D-021 does not move it) | C-040, C-050, C-051 |
| `run_pwml_export` and the SBML binding | **C-052** | C-040, C-050, C-051 |
| `rag/eligibility.py` organism helpers (`:1366-1404`) | **no card** — `MASTER_PLAN` §2 read-only (`:147`) | all three |

**Why `_emit_canonicalization_preflight` gets a minimum-surface owner rather than a freeze.**
SPIKE-002 §3 classifies it as **staying** in `ir.py` while recording that it "also used by
`_emit_canonicalization_preflight` at `:920`, which **stays** → must be re-imported by `ir.py`".
Verified at `0182eae` by **word-boundary** match: `_compound_external_ids` is defined at
`ir.py:558` with exactly **two** call sites — `:596` (in `_canonicalize_compound_offline`, which
moves) and `:920` (in `_emit_canonicalization_preflight`, which stays). `:806` calls the
**different** `_normalize_compound_external_ids`, whose name contains it. Moving the
definition therefore **mandatorily** breaks `:920`, so repairing that one reference is not
optional work and not a scope choice — it is the minimum surface required to deliver C-040's own
declared extraction. That is the same accept-and-record shape the C-010 merge already ratified:
*"a card that names a deliverable authorizes the minimum surface to deliver it"* (`72ee20f`).
**No competing product semantics exist here** — the function's behaviour is identical before and
after a pure re-import — so nothing is frozen. The grant is deliberately narrow: **C-040 may
change the `:920` reference plus the one `ir.py` module-header import line that re-import
requires — which sits outside `:900-963`, and which the C-010 precedent reaches, that precedent
having concerned a module-level `__all__` entry — and nothing else.** Any change to what
that function *does* is outside every card's boundary and requires a new owner.

**Shared acceptance harness.** **T-102 (M3)** is the single shared acceptance harness for this
chain. Its **scope is owned by D-016** and may not be narrowed by any implementation card: it
verifies **both** compound identity **and** organism/species equivalence across canonical JSON,
PWML **and** SBML. Its **execution** is the `LEDGER.md` milestone row, run by `pwml-test-runner`
after C-052. **No implementation card owns T-102, edits its scope, or may cite its cost as grounds
to narrow it.** SPIKE-002 §5's hard acceptance criterion for C-050 stands and is C-050's alone:
reaction count in `final_mapped.json` identical before and after pre-freeze resolution, proven on
the `PMC12856317` and `PMC12452463` legs, with the extraction name written to `synonyms`.

**Recorded consequence, not a new decision:** as scoped by D-016, T-102 cannot pass on C-052 alone,
because species equivalence is owned by **C-045**, which is planning-only and undispatched. T-102
must therefore not be read as green-able at C-052, and any species-axis failure it produces is
attributable to C-045 — **never** to C-050. Routed to the product owner as a scheduling item; no
scope is narrowed here.

### 3. Dependency and merge order

`SPIKE-002 ✔` → **C-040** (Wave C) → **C-050** (Wave D, also after C-030) → **C-051** (Wave D,
after C-040 and C-050). Unchanged from `MASTER_PLAN` §9. **C-051 must merge last of the three**:
its assertion is only true once C-050 actually resolves pre-freeze, so merging it earlier would
either fail closed on every leg or force a weakened assertion. C-052 remains downstream of C-030,
C-050 and C-020.

### 4. C-011 coverage requirements carried to their selected owners

The two C-011 coverage gaps are assigned **outside** this trio and are **cross-references only**
for C-040/C-050/C-051, which may not implement them:

- **Object sharing across the seam → C-030** (`LEDGER.md` A0-C7). The existing
  `final_mapped is result["payload"]` assertion is tautological and is not a share guard.
  **C-050 must not introduce a pre-freeze step that re-binds or replaces the caller's payload
  object** in a way that breaks the relationship C-030 pins.
- **Actual `canonical_json_path`, all 39 cohort legs, including the SBML input path → C-052**
  (`LEDGER.md` A0-C8). **C-050 must not change which file the canonical path names.**

### 5. One seam, one reshaper

`freeze_canonical_payload` and the artifacts it produces may be reshaped by **exactly one card at
a time**, in the merge order above. A card in this trio that finds it needs a change inside another
card's symbol **stops and escalates**; it does not make the change, and it does not make an
equivalent change in its own file to route around the boundary. Two cards proposing changes to the
same lifecycle seam in the same wave is a dispatch error, to be resolved before dispatch.

### 6. Semantics preserved unless a later explicit product decision changes them

The **canonical payload lifecycle, gate-report production, quarantine behaviour, canonical
JSON-write and SBML-input resolution** are preserved as they stand at `0182eae`: the freeze order
(gate → hash → stamp → serialize → hand to SBML), the seven-field `CanonicalFreezeResult` contract,
the refusal branch, and `_freeze["canonical_json_path"] or sbml_input_path`. **The mechanism is
frozen; the values are not.** D-015 is exactly such an explicit product decision: moving compound
resolution pre-freeze deliberately changes the entity `name` values that enter the frozen payload,
and therefore changes `canonical_payload_sha256` and may change `canonical_graph_sha256`. That is
the intended effect of D-015 and is not a violation of this clause. Any *further* change to the
mechanism requires its own explicit product decision.

### 7. Manifests and budgets are defined before dispatch

Under **D-019**, each of C-040, C-050 and C-051 receives, **before** dispatch: an exact
hand-authored file/function manifest, a hand-authored changed-line maximum, and a separately
stated generated-artifact budget. **Canonical G11 reports are required and committed and are
excluded from the hand-authored ceiling** (`G11-EVIDENCE-ACCOUNTING-001`); noncanonical generated
artifacts are prohibited unless separately authorized. **C-040's sizing is governed by D-019, not
by `SPIKE-002-REPORT.md` §4**, which is expressly *"Superseded prospectively by D-019 (LOCKED)"*
for removing "the unconditional obligation to pre-split C-040" — its ≈700-line measurement stands
as fact, the obligation does not. Under D-019 the budget accommodates C-040 **or** the card is
re-scoped before dispatch; never a split that strands an unvalidated half (the H-004 deadlock,
`LEDGER.md:56`). A split is permitted **only** at a boundary where each
half is independently implementable and independently validatable — a budget may never force an
unvalidated semantic half to merge or to be left behind. Dispatching any of the three without its
declared manifest and both budgets is a dispatch error.

### 8. Nothing frozen

No responsibility in this partition required choosing between competing product semantics the
repository has not settled. Every boundary above is taken from an existing card purpose, a
`MASTER_PLAN` §9 row, `SPIKE-002-REPORT.md` §3's own move/stay classification, or a locked
decision (D-015, D-016, D-019). **No sub-decision is frozen.** In particular
`_normalize_compound_external_ids`, `_compound_external_ids` and the `:920` reference repair are
assignable **without** a product choice — SPIKE-002 §3 already classifies each as *move* or
*stay*, and this decision only records the owner that classification implies.

**Four items are routed, not decided.**

1. **`MASTER_PLAN` §9 `:367` is incomplete** — it names two of C-040's four moved functions.
   §2 supersedes it for dispatch purposes. Repairing the §9 row itself is `MASTER_PLAN`'s to do
   and is **not** performed by this decision; until it is, §9 `:367` must not be used as C-040's
   manifest.
2. **The T-102 / C-045 scheduling consequence** in §2.
3. **The unassigned retry ownership** for `stoich/agent.py` and `rag/embed.py` (`LEDGER.md`
   A0-C2), which lies outside this trio entirely.
4. **`G11-EVIDENCE-ACCOUNTING-001` is cited but never defined.** It is relied on by §7 above, by
   all nine Wave A0 card budget blocks and by every A0 merge message, yet **no document under
   `docs/pwml_recovery_sprint/` defines it** — it exists only in merge-message prose. A budget
   exclusion that every card depends on should have a written definition. **No definition is
   written here; that is the product owner's authority alone.** Recorded for the product owner
   alongside the C-010/C-012 dispatch-authority item in `LEDGER.md`.

---

## D-022 — Closeout: Chunk D status, two external labels defined, two ceilings measured · 2026-08-11 · LOCKED

Authority: `CONTROL-PLANE-CLOSEOUT-002`, implemented by **H-007**. **D-020 and D-021 are not
rewritten** — this decision is appended and supersedes only the present-tense statements named
below. No merged card is reopened; no production code, test or fixture is touched.

### 1. The forward Chunk D gate is in effect, and its execution partition is per NODE

**Superseded here:** D-020's *"That definition has not merged … the gate exists in no ref in
this repository"*, and the identical claim in the C-011 `LEDGER.md` row. Both were **true when
written at `0182eae`** and became **false at `69d4069`**, where RECONCILE-B merged the
split-process definition into `TEST_MATRIX.md` together with `evidence/chunk_d_gate.py`.
D-020's obligation (1) is discharged and its effective date (2) has passed.

**Also superseded:** any reading of D-020 fact 6 or of the RECONCILE-B record as having proved
that *no* process partition can stabilise the 23-test `qb` cohort. What was measured is
narrower and stands: isolating the whole **file** in one fresh process is insufficient, which
follows from the documented intra-file cause — one process building 23 `AppTest` objects.
**Per-node isolation had never been tested.** H-007 tests it: the 150-test core keeps one
deterministic process and each of the **27** AppTest node IDs runs alone in a fresh process.

**Partition, proven on every invocation** and re-proven by the reviewer: **177 = 150 core + 4
`s8` + 23 `qb`**, missing 0, extra 0, overlap 0, deselected 0, and the executed node-ID set
equal to the collected one.

**Measured status — six scheduled runs, three against a clean export of `0182eae` and three
against H-007's tip, order declared and committed before any ran (`prompts/H-007.md` §2):**

| run | tree | partition | executed | failing node |
|---|---|---|---|---|
| 1 | base export | 177 = 150+4+23 | **177/177** | — |
| 2 | base export | 177 = 150+4+23 | 176/177 | `node16` |
| 3 | base export | 177 = 150+4+23 | 176/177 | `node16` |
| 4 | candidate `14d0833` | 177 = 150+4+23 | 176/177 | `node10` |
| 5 | candidate `14d0833` | 177 = 150+4+23 | 176/177 | `node11` |
| 6 | candidate `14d0833` | 177 = 150+4+23 | **177/177** | — |

**Base 1 green / 2 red. Candidate 1 green / 2 red.** The **150-test core and all 4 `s8`
nodes passed in all six runs**; every failure is in the 23-node `qb` cohort, and **no node
failed twice in the same tree except `node16` at base**. Four different symptoms appeared —
two different wrong `review_flags` contents, and two different `KeyError`s for a
`session_state` key a partially-executed AppTest script never set. **None carried the
documented `FragmentThreadState` message**, whose recorded cause — several `AppTest`s in one
process — cannot apply under per-node isolation. Same family, new signatures. BL-003's.

**Declared isolated-node diagnostic** (`prompts/H-007.md` §2.2, committed before any
candidate run): three cold wrapped processes per failing node per tree, 18 runs, **17 passed
/ 1 failed**. Combined with the scheduled runs, per node, base vs candidate:

| node | test | base | candidate |
|---|---|---|---|
| `node10` | `test_all_four_artifacts_are_written_and_retained` | 6 pass / 0 fail | 4 pass / **2 fail** |
| `node11` | `test_one_canonical_payload_is_gated_serialized_and_exported` | 6 pass / 0 fail | 5 pass / **1 fail** |
| `node16` | `test_research_mode_surfaces_the_flags_it_did_not_act_on` | 4 pass / **2 fail** | 6 pass / 0 fail |

Over all `qb` executions: **base 78 runs / 2 failures; candidate 78 runs / 3 failures.**

**Ruling applied: case 2 — a test-infrastructure race.** Read node-by-node, `node16` failed
only at base and `node10`/`node11` only at candidate. Neither half is attributable to this
diff, and the reason is decisive rather than statistical: **`tests/` and `src/` are
byte-identical at the base export and at the candidate tip** (`:tests`
`bb4099b838b69bfb6d94ebcc71bcbab19e4588b4`, `:src` `d919bd8edffa7c0efe6d07960c667b05f9939bc7`),
so the candidate contains **zero** bytes of test or production code and cannot change a
Chunk D outcome in either direction. A "candidate-only" reading would have to accept the
mirror-image "base-only" claim for `node16` with equal force, which is incoherent. The
per-node diagnostic is **underpowered by construction** — at ~3% per node per run, three
cold runs have under 12% power to reproduce one — which is stated, not dressed up.

**Therefore, and until the race is fixed:**

1. The **154 deterministic tests — 150 core plus 4 `s8` — are the BLOCKING Chunk D gate.**
2. The **23-node `qb` cohort is mandatory to run and mandatory to report, and temporarily
   non-blocking.** This is **not** permission to read any `qb` red as expected. A **new
   deterministic** failure, a failure a diff **reproduces**, or a traceback **implicating
   the diff** still blocks the merge, and no test may be weakened, deselected, retried until
   green or relabelled environmental.
3. The race is filed as infrastructure backlog item **BL-003** for a later cleanup agent,
   with the 156 recorded executions and **four** distinct nodes — the reviewer's own run
   added `node05` and a fifth symptom. Wave A1 is **not** held behind it.
4. Per-node isolation is retained regardless: it frees the deterministic core to ~1 s and
   localises a `qb` failure to one named node instead of losing it in a 23-test process.

### 2. `G11-EVIDENCE-ACCOUNTING-001` — defined

Cited by all nine Wave A0 card budget blocks, by every A0 merge message and by D-021 §7, and
until now defined in no document. Definition, in full:

1. **Canonical G11 cleanup reports are required and are committed** with the branch that
   produced them, one per job, at `evidence/g11/<TASK-ID>/<SEQ>-<label>.json`.
2. **They are excluded from every hand-authored changed-line ceiling.** A bound written for
   hand-authored code may not be spent by machine-generated evidence the same policy compels.
3. **They remain fully subject to G11 validation** — the exclusion is from the *budget*, never
   from the *checker*. `g11_evidence.py check` must pass over them.
4. **Noncanonical generated artifacts remain prohibited** unless a card separately authorises
   them with its own artifact-count and size budget (D-019). C-011's one-fixture
   authorisation is such a grant and is unaffected.

A **canonical** report is one `bounded_run.py` itself wrote to a path allocated by
`g11_evidence.py next`. A hand-written, reconstructed or back-filled file is not canonical and
is not evidence; G11 is **prospective only**.

### 3. `CONTROL-PLANE-RECONCILE-001` — defined

The post-Wave-A0 control-plane reconciliation. **Starting SHA
`0182eae704711d1ce1ee938a39e42748a203c203`** (the C-011 merge, integration head when it was
authorised). **Final SHA `08d5d079ae4a71a23214004b1dfea37625a2e520`.**

**Two lane merges, in this order:**

| Merge | Lane | Branch | Landed |
|---|---|---|---|
| `69d4069` | RECONCILE-B (§3, §4) | `agent/recon-chunkd-gate` | split-process Chunk D gate in `TEST_MATRIX.md` + `evidence/chunk_d_gate.py`; a path-specific `.gitattributes` line-ending rule for the one byte-exact C-011 fixture. 250 hand-authored lines against a 250 ceiling; 50 G11 reports excluded |
| `08d5d07` | RECONCILE-A (§2, §5–§8) | `agent/recon-control-plane` | Wave A0 ledger reconciliation, the nine card ceilings, the "0 artifacts" contradiction, D-016 ownership, D-020 and D-021. 891 hand-authored lines against a 900 ceiling; no G11 report, because the lane executed no python |

**Production-code prohibition.** Neither lane could touch `src/`, `tests/` or `batch/`, and
neither did: the `src` and `tests` tree objects are byte-identical at `0182eae` and at
`08d5d07` (`:src` `d919bd8edffa7c0efe6d07960c667b05f9939bc7`, `:tests`
`bb4099b838b69bfb6d94ebcc71bcbab19e4588b4`), re-verified by H-007. Every *"verified at
`0182eae`"* citation against production code therefore remains exactly valid.

**Approvals.** Each lane received an **exact unsuffixed `APPROVE`** from a fresh independent
reviewer of its exact merged tip, told no earlier verdict bound it — RECONCILE-A after **two
`REJECT`s and three correction rounds**, RECONCILE-B after two. Neither lane's history was
squashed, amended, rebased away or cherry-picked.

**Two worktrees — ratified.** Running the reconciliation as two branches in two worktrees
resolved a contradiction in its own authorising instructions and is **retrospectively
ratified as reasonable**. It creates **no general permission** to resolve a future authority
conflict silently: the next such conflict is escalated, not resolved.

### 4. C-010 and C-012 — measured ceilings, no unused headroom

Derived from the Git objects, not from any report. Each branch carries **exactly one commit**
above its base and the merge's **second parent is the reviewed tip**, so base-to-tip is
unambiguous and **no accounting subitem is left open**.

| Card | Base (merge 1st parent) | Reviewed tip (2nd parent) | Merge | Hand-authored +/− | **Final ceiling** | G11 excluded |
|---|---|---|---|---|---|---|
| C-010 | `9e06360` | `d784747` | `72ee20f` | +284 / −10 across 4 files | **294** (was 300) | 11 artifacts, 883 lines |
| C-012 | `361b158` | `def6adb` | `9e06360` | +263 / −58 across 2 files | **321** (was 330) | 4 artifacts, 394 lines |

Only canonical G11 evidence is excluded. C-010's hand-authored files: `docs/change_log.md`,
`strict_quarantine.py`, `test_strict_quarantine.py`, `test_strict_quarantine_real_artifact_replay.py`.
C-012's: `driver.py`, `test_batch_driver_seam_golden.py`.

The ceilings **300** and **330** quoted in `prompts/C-010.md:149` and `prompts/C-012.md:104`
are the historical dispatch-time limits and are left as written; **294 and 321 are the final
ceilings** and this table is authoritative for them. **Implementation, tests, reviewer verdict
and merge are unchanged for both cards** — this is accounting only, and neither card's
recorded 294 / 321 measurement moves.

### 5. `BL-001` — the two retry seams, one backlog item

C-014 removed the SDK retry layer (`SDK_MAX_RETRIES = 0`) from four call sites that use the
raw client **directly** and so also bypass `llm/client.py`'s own 8-attempt loop:
`stoich/agent.py:477`, `:552`, `:580` (chat), and `rag/embed.py:163` (embeddings, via `:154`).

**Retries are not restored now.** No existing card owns either file — re-verified at
`08d5d07`, and `prompts/C-014.md:145` states it in terms — so binding the seams to C-032,
C-035, C-042, C-043 or C-055 would invent overlapping ownership. They are bound instead to a
single backlog item **BL-001**, which must scope an owner and a replacement-resilience test
for both files. `LEDGER.md` A0-C2 points here. **BL-001 does not block Wave A1.**

### 6. G10's smoke count is 460

**457 is obsolete.** C-010 moved the pinned baseline deliberately, 457 → 460, with an exact
documented delta, and every A0 merge from `72ee20f` on measured 460. `TEST_MATRIX.md` is
corrected; `MASTER_PLAN.md` §5 G10 and `CLAUDE.md` rule 10 still say 457, sit outside this
closeout's permitted change set, and are backlog item **BL-002**.

---

## D-023 — velocity control plane · effective **2026-08-12** · forward only; D-020, D-021 and D-022 are unchanged and not reopened

**Velocity rulings.** Schedule by **actual dependency and semantic seam**, not rigid Wave
B/C/D labels. **At most four concurrent implementation writers**; heavy jobs strictly
serialized. Focused tests run at the **final card tip**. Reviewers inspect the exact diff
and evidence, **need not duplicate every successful expensive execution**, and **may**
independently reproduce suspicious or contract-critical behaviour. **Smoke 460 and
whole-tree G11 remain required per merge.** Small unrelated bugs are **deferred, not
enlarged into the current card**. **C-034 is held for re-scoping** — its declared target is
dead demo code. **C-045 is authorized for planning and scoping only.**

**Forward parallel-merge proof.** For the second or later parallel branch cut from an
earlier common base, whole-tree equality between the merge and the standalone reviewed tip
is **neither required nor possible**. Required instead, all three: (1) the merge's **second
parent equals the reviewed tip SHA exactly**; (2) within the card's **owned paths** the
**first-parent-to-merge diff equals the reviewed card diff from its dispatch base**; (3) all
remaining tree content comes from the merge's **first parent**, already-authorized
integration history, **not** from new unreviewed paths. This **supersedes forward** any
whole-tree-equality reading. D-022's historical record stands.

**Prospective-integration validation.** Integration is **merge-only**: review the exact card
tip; `git merge --no-ff --no-commit <reviewed-tip>`; inspect the prospective staged merge
against the authorized path manifest; run the card's focused tests and required gates **on
that combined prospective state**; commit only if they pass. On failure, **freeze** the
prospective merge state and report the affected lane — never reset, abort destructively,
rewrite the worker branch, or commit a failing merge. **No rebase**, and integration is
never merged back into a worker branch.

**G9 clarification.** G9 is required for a **claimed correction or preservation of
pre-existing observable behaviour**, and its proof must **fail behaviourally at the base and
pass at the tip**. **Symbol absence is not behavioural proof.** A genuinely **new capability
or module** receives an **explicitly labelled new acceptance test** and requires no
fabricated base failure. **Reviewers must reject any attempt to mislabel a regression as new
functionality.**

**Chunk D cadence.** Deterministic core plus `s8` — **154 tests — is BLOCKING** per D-marked
card. The **23-node `qb` cohort is mandatory to run and to report**. **C-030, C-050 and
C-052** touch the Streamlit/freeze/UI seam and **must run `qb` before merge**; non-UI
D-marked cards such as **C-040 and C-051** may run `qb` once at their **pack-level
integration checkpoint**. A **deterministic, diff-reproducible or traceback-implicated `qb`
failure blocks** the affected lane. **No deselection, retry-until-green, or false
environmental relabelling.**

**Seam-based concurrency.** **One writer per semantic seam per merge window**; disjoint
functions and files may be developed **concurrently**; **merges are serial**. D-021's
one-reshaper-at-a-time freeze-lifecycle rule **remains binding**, and **C-041 → C-031 →
C-032 remains serialized** because `driver.py` behaviour is coupled.

**Deferred findings.** A finding a card does not own goes to `FINDINGS.md`. Workers report
them to the orchestrator and never edit that file; it is updated **once at pack closeout**.

**PWML-first / SBML boundary.** **No SBML implementation, extension or refactor**;
`src/t2pw/sbml/` is **outside every implementation boundary**. C-052 may later **read and
assert** the existing SBML input binding; T-102 may later use the **comparator already
delivered by C-020**.

---

## D-024 — `attempt_cap_reached`, a seventh termination reason · 2026-08-13 · LOCKED

**Extends D-005. D-005 is not reopened, amended or contradicted** — its six named reasons
keep their exact meanings, their exact strings and their exact denominator rule. D-005 goes
from six named termination reasons to **seven**.

C-042 built the § 9 escalation ladder. When the ceiling of three model attempts is spent the
ladder refuses the next rung with the skip cause `attempt_cap_reached` and — correctly, since
none of the six fitted — **no termination reason at all**. `attempt_cap_reached` genuinely
*ends* the ladder, so the one stop that reliably terminates a leg was the one stop that said
nothing, and every downstream denominator saw `""`. C-042's writer declined to guess, which
was right; guessing is what this entry replaces.

**The reason.** `attempt_cap_reached`. Used when **all** of these hold:

* the configured maximum number of attempts has been consumed;
* the operation has not succeeded;
* **no** deadline or timeout caused termination;
* **no** explicit refusal caused termination;
* **no** separate resource / token / budget exhaustion caused termination;
* **no** stronger existing terminal reason truthfully describes the outcome.

**Precedence, mandatory, in this order:**

1. Successful completion keeps the success / completed reason.
2. An explicit refusal keeps the applicable refusal reason.
3. A real deadline or timeout keeps its deadline/timeout reason.
4. A separately **measured** resource or token-budget exhaustion keeps its budget reason.
5. When **only** the configured attempt count ended processing → `attempt_cap_reached`.

**Never mislabel an attempt cap as timeout, refusal, success, or generic budget exhaustion.**
Equally, never mislabel it as `retrieval_exhausted` or `no_new_claims`: D-005 permits
`retrieval_exhausted` only when the configured ladder actually completed, and a leg cut off by
the ceiling is precisely a ladder that did not. The implemented rank in
`rag/loop_policy.TERMINATION_PRECEDENCE` is therefore below `budget_exhausted`,
`operation_timeout`, `identical_empty_response` and `scientifically_unrecoverable`, and above
`retrieval_exhausted` and `no_new_claims`.

**It reaches the leg-level report.** The reason is set at two sites, not one. The ladder's
`admit` names it on the `RungDecision` that refuses the rung, and `_run_json_stage`'s
termination block names it on `PipelineFailure.terminal_reason` and on the
`stage1_extraction_ladder_termination` boundary record — where a capped leg previously
reported `""`. At the leg level it is the **last** branch, after `budget_exhausted` and
`identical_empty_response`, and it is claimed on the ladder's **recorded cap refusal**, not
on `attempts_remaining == 0`: the ceiling must actually have stopped a rung that wanted to
run. A leg that merely spent its last attempt and then failed for another reason was not
ended by the cap. `operation_timeout` is not in that chain and cannot be displaced by it:
`_issue` records the timeout and re-raises, so a timed-out leg leaves by that exception and
never reaches the block. Nor can a successful leg acquire the reason — both success returns
precede the block.

**`OPERATIONAL_TERMINATION_REASONS` is UNCHANGED** — it stays exactly
`{budget_exhausted, operation_timeout}`. `attempt_cap_reached` is **not** added to it. D-005
calls the cap *"a safety ceiling, not a promise"*, which is a different fact from a leg that
ran out of clock. Whether the attempt cap should count in the pipeline-completion and
end-to-end strict-success denominators is a **product decision that has not been made**;
until it is, the denominator does not move.

**One literal, two vocabularies.** The new termination reason uses the same string as the
ladder's existing skip cause `extraction_ladder.SKIP_ATTEMPT_CAP`, because both name one
event. The vocabularies stay separate and independently closed: `require_reason` still refuses
every other skip cause, `require_skip_cause` still refuses every other termination reason,
`SKIP_CAUSES` and `TERMINATION_REASONS` remain distinct, and *"a skip cause is not a
termination reason"* remains true — a skip cause says why one **rung** did not start, a
termination reason says why the **leg** stopped, and they are recorded in different fields.

---

## Open — not yet decided

| # | Question | Blocks | Why it cannot be answered from the repository |
|---|---|---|---|
| O-1 | `placeholder_backed_proteins` (21 in the pinned run): gold-set error class, or legitimate biology preservation? | any branch that touches protein export policy | It is a genuine disagreement between two intentional designs, not a defect. TRAP-3 forbids agents from resolving it. |

**Closed:** O-2 → D-011 · O-3 → D-014.

---

## D-025 — Generated evidence is budgeted before dispatch, including what the gates allocate · 2026-08-13 · LOCKED

**D-019 already required two budgets. It was not being satisfied.** Two Pack 3 cards proved the
gap independently, and in both the writer was right and the charter was wrong.

* **C-050a** was given `≤ 40` generated artifacts while the same charter *mandated* the
  split-process Chunk D gate, which **self-allocates ~32 reports on its own** — leaving 8 for the
  focused run, the G9 proof, three determinism runs, a discrimination run and SMOKE. It landed at
  **41**. Its reviewer: *"the ceiling is internally inconsistent with the gate the same charter
  requires."*
* **C-041a** was given a hand-authored ceiling and **no generated-evidence budget at all**. Of its
  10,080 evidence lines, **9,294 are the 77 mandatory `bounded_run.py` cleanup reports** whose
  shape is fixed by a wrapper agents may not modify, and **50 of the 77 are auto-allocated by
  `chunk_d_gate.run`** (25 per `qb` cohort).

**Every charter, before dispatch, states three ceilings separately:**

1. **hand-authored** additions-plus-deletions;
2. **generated artifact count**;
3. **generated artifact byte or line size**, where applicable.

**The generated figure must be budgeted, not guessed.** It must provide for:

* the fixed reports a Chunk D partition allocates for itself (**~32 per `qb` cohort**, ~6–8 for
  `core`+`s8`);
* one wrapper report per bounded job;
* focused and merge-gate reports;
* **at least one failing run** wherever a failure is plausible;
* headroom for **one review correction**.

**Budgets are ceilings, not targets. Genuine evidence is never deleted to satisfy a number.** A
writer that would exceed a ceiling **stops before committing and reports**; deleting a superseded
or failing report to come in under the line is a **reject**. Both Pack 3 writers correctly refused
to do it, and C-050a committed its **failing** determinism run alongside the passing one — that is
the required behaviour, not an overrun to be charged to the writer.

**A charter that omits the generated figure is a dispatch error**, exactly as D-019 § 3 says of a
missing manifest. Where it has already happened, the omission is disclosed at closeout — it is
**not** cured retroactively and **not** described as compliance.

---

## D-026 — Tracked background execution is compliant when a bounded job exceeds the interactive limit · 2026-08-13 · LOCKED

**This resolves a question three cards relitigated** (C-034, C-041a, C-050a), each self-declaring
the same deviation and each having it adjudicated separately. `TEST_MATRIX` § 0 rule 1 is amended
in the same commit as this entry.

A Chunk D `qb` cohort is ~10.5 minutes across 23 AppTest processes and **exceeds the 10-minute
interactive foreground cap by construction**. The rule's purpose was never the foreground shell —
it was bounded lifetime, owned-PID-only cleanup, verified zero survivors, and a committed
structured report.

**A tracked background job is compliant when all of these hold:**

* it is launched through the **same approved `bounded_run.py` wrapper**, unmodified;
* its **task/process identifier and output path are recorded immediately**;
* **only one heavy job runs at a time**;
* the orchestrator **polls it rather than launching duplicates**;
* **wrapper cleanup executes**;
* **descendant counts and zero survivors are verified**;
* the **final canonical JSON report is inspected**;
* **no detached or unowned job remains**.

**This does not authorize arbitrary background shells.** Detached processes, `nohup`, untracked
jobs and `Start-Process` without bounded waiting remain forbidden. Cleanup still targets only
PIDs the job created; `taskkill /IM python.exe` and `pkill python` remain forbidden; pre-existing
processes are reported, never killed.

**Prefer a single tracked bounded cohort** where splitting would change the gate's semantics or
materially increase overhead. Use `--only` partitions **only where the gate is explicitly
partition-safe** — `chunk_d_gate.py` proves its `177 = 150 + 4 + 23` partition on every
invocation, so its partitions are safe by construction.

**Retrospective effect:** the C-034, C-041a and C-050a deviations are compliant under this rule.
Each was self-declared before review, ran through the unmodified wrapper inside the same Job
Object under its own outer timeout, and verified zero survivors from a complete descendant census.
No measurement in any of those cards depended on the foreground/background distinction.

---

## D-027 — Conditional C-051 ownership of the post-freeze identity seam · 2026-08-14 · LOCKED

**D-021 § 2 remains locked except for one narrowly defined conditional carve-out.**

C-051 may **inspect** and, **only when proven necessary**, modify the `pathwhiz_id`
materialization logic inside:

```
src/t2pw/pwml/ir.py :: _entity_record
```

**Why this authority exists.** Live-source measurement (P2-06, re-confirmed by AST on the
integration tip) shows `_entity_record` materializes `pathwhiz_id` **after the freeze boundary**,
while merely removing the `_resolve_compound_rows` call — all C-051 was chartered to do — may
leave that later materialization **reachable**.

**D-021's statement that `_entity_record` must remain untouched is amended only to the extent
required to enforce the already-locked rule that identity may not be created or resolved after
freeze.** Nothing else in D-021 § 2 moves.

### Required sequence

**C-051 remains blocked until C-050 and C-045 have merged.** C-045 and C-051 **must not run
concurrently**: `_canonicalize_species_offline` is already called from **inside** `build_pwml_ir`,
creating a shared live lifecycle seam.

After C-050 and C-045 merge, but **before C-051 makes an implementation commit**:

1. **Re-derive the relevant symbols by AST**, never by D-021's stale line numbers.
2. **Trace `build_pwml_ir`, `_resolve_compound_rows`, `_canonicalize_species_offline` and
   `_entity_record` on the actual combined tip.**
3. **Measure whether `_entity_record` can still create, resolve, or newly materialize a
   `pathwhiz_id` after the canonical payload has frozen.**
4. **Exercise at least these four cases:**
   * an entity **already carrying a valid pre-freeze `pathwhiz_id`**;
   * an entity **lacking one at freeze**;
   * an entity whose identity information **exists only in mapping metadata**;
   * a **normal compound** passing through the **live production call chain**.

### If the path is unreachable

If combined-state evidence proves C-050 and C-045 **already foreclose** post-freeze
materialization:

* **do not modify `_entity_record`;**
* **retain the D-021 lock;**
* add or preserve a **focused guard proving the path is unreachable**;
* record the measurement and **close P2-06 as discharged by reachability proof**.

**A no-code result is a valid completion of this clause.**

### If the path remains reachable

If `_entity_record` can still **newly materialize** `pathwhiz_id` after freeze:

* C-051 is authorized to modify **only the relevant `pathwhiz_id` block** inside `_entity_record`;
* it **may forward or serialize** an identity **already established before freeze**;
* it **must not resolve, infer, synthesize, hydrate, or newly materialize** identity after freeze;
* a **missing pre-freeze identity must follow the existing missing-identity / review policy**
  rather than silently inventing an identifier;
* **do not refactor unrelated `_entity_record` behaviour**;
* **do not broaden ownership to other identity fields** without a separately demonstrated
  requirement.

**Tests must prove:**

* valid pre-freeze identity **survives unchanged**;
* an absent identity **is not created after freeze**;
* **mapping metadata cannot silently become a new post-freeze identity**;
* **canonical/frozen hashes and decision inputs remain stable**;
* **no correct identifier is accidentally dropped**;
* **no PWML or biological semantics move** outside the intended identity-timing correction.

### D-021 live symbol citations, re-derived by AST on the integration tip

D-021's own line numbers are **stale** — C-040 moved four functions out of `ir.py` and later cards
shifted the rest. **The historical evidence in D-021 is NOT rewritten**; these are the live
locations to work from.

| Symbol | D-021 cited | **AST-measured now** |
|---|---|---|
| `ir.py :: _entity_record` | `:437-449` | **`ir.py :438-450`** |
| `ir.py :: _canonicalize_species_offline` | *(unnumbered)* | **`ir.py :617-701`** |
| `ir.py :: _emit_canonicalization_preflight` | `:900-963` | **`ir.py :704-767`** |
| `ir.py :: build_pwml_ir` | *(call site `:1106-1114`)* | **`ir.py :770-1811`; resolution call at `:911`** |
| `_normalize_compound_external_ids` | `ir.py :530-555` | **moved → `compound_resolution.py :198-223`** |
| `_compound_external_ids` | `ir.py :558-575` | **moved → `compound_resolution.py :226-243`** |
| `_canonicalize_compound_offline` | `ir.py :578-621` | **moved → `compound_resolution.py :246-311`** |
| `_resolve_compound_rows` | `ir.py :797-897` | **moved → `compound_resolution.py :314-421`** |

**The three call sites sit in sequence inside `build_pwml_ir`**, which is why the seam is shared:

```
:844  _canonicalize_species_offline(...)   <- C-045 moves this pre-freeze
:911  _resolve_compound_rows(...)          <- C-051 deletes this, asserts instead
:921  _entity_record(...)                  <- materializes pathwhiz_id at ir.py:447
```

Any card quoting D-021's numbers must re-derive them by **AST symbol, not line range**
(`PACK2-SHARED` § S9 trap 1: insertions above a function shift it).

---

## D-028 — DB match admission: no fuzzy rename, and short names need a corroborating identifier · 2026-08-14 · LOCKED

**Adjudicated `product_contract_violation`** by an independent `pwml-bio-auditor`; ruled by the
product owner. Implemented by **C-040a** (`agent/p40a-db-match-admission` @ `7d5a3916`).

### The defect

`compound_resolution.py` required `confidence >= 0.85` to accept a PathBank DB match. When that
failed it logged `compound_db_resolution_failed` — and then **applied the resolution anyway**.
`db_resolver.py :: apply_compound_db_resolution` checked only `status != "matched"` and never read
confidence. **The gate decided whether to log a failure, not whether to apply.**

Measured over the 124 distinct compound names in committed `runs/**/final_mapped.json`:
**24 would be renamed and identifier-stamped, every one below the acceptance bar** — 3 via
`fuzzy_name` @0.65 and 21 via `exact_short_name_or_synonym` @0.70. Confirmed-wrong cases included
`OPDA → Dinor-12-oxo-phytodienoate` (**not a PathBank synonym at all** — exact-name and synonym
lookups both return empty; the fuzzy tie-break passes by 0.0006), `THF → Tetrahydrofuran`,
`CL → Chloride ion` (CL = cardiolipin), `G3P → 3-Phosphoglyceric acid` (G3P = glycerol-3-phosphate),
`PE → O-Phosphoethanolamine` (PE = phosphatidylethanolamine),
`glycerol-3-phosphate → Indoleglycerol phosphate`.

The gold set already names this failure class: `PMC13231680`'s `forbidden_identifiers` entry for
`PSA` — *"in most biomedical text PSA is prostate-specific antigen … Resolving it to a protein
identifier is a failure."* Contract basis: **PRODUCT_CONTRACT §1** ("never invent … identities") and
**§8** ("Never accept an identifier because its format is valid").

### The ruling — "no fuzzy + abbreviation guard"

1. **A `fuzzy_name` match may never rename and may never stamp identifiers.** Record only.
2. **A unique exact normalized full-name or synonym match may rename and stamp**, *except* that a
   name of **four characters or fewer** additionally requires **corroboration by a matching
   identifier on the same DB row**.
3. **Corroboration means AGREEMENT, not presence.** The row's identifier must **equal** the matched
   PathBank row's same-namespace identifier. A disagreeing identifier, or one with no counterpart on
   the matched row, corroborates nothing. *(Clarified during implementation: a presence-only reading
   admitted `PE` — whose `mapped_ids.kegg = C00012` is absent from PathBank's `compounds.kegg_id` —
   contradicting the kill-list this decision was ruled against.)*
4. **Corroborating namespaces are limited to KEGG, ChEBI, PubChem and HMDB.** **DrugBank is excluded
   and remains fail-closed unless separately ruled.** Exclusion is conservative: it can only produce
   more refusals, never more admissions.
5. **"Record only" means no rename AND no identifier stamp** — never a partial apply.
6. **A refused match is recorded for review, never silently dropped and never raised.** Merge rule 7
   preserves incomplete-but-correct pathways as `review_required`. Status:
   `identity_refused_review_required`.
7. **The `4` is a named module constant citing this decision** — `SHORT_ABBREVIATION_MAX_CHARS`.

### Attribution and scope

The defect is **pre-existing**: C-040 lifted it verbatim from `ir.py`, and it remained live
post-freeze at `ir.py:911-918`. **C-050 does not create it** — C-050 widens its blast surface by
moving false names into the canonical payload *before* the freeze, where they are hashed and reach
`final_mapped.json` and the quarantine report. **The remedy is to fix the gate, not to revert
C-050.** One fix governs both the pre-freeze and post-freeze surfaces; it is deliberately **not**
special-cased by caller.

`test_all_four_artifacts_are_written_and_retained` needs **no** test change: `original_process`
should show post-canonicalization names, so it returns to green once this decision is implemented.

### Measured effect (C-040a, independently reproduced by `REV-040a`)

| | base | tip |
|---|---|---|
| name-only: admitted / refused | 56 / 0 | **36 / 20** |
| real committed rows: refused | 0 | **5** |
| real rows: substantive renames | 7 | **2** |
| **new renames introduced** | — | **0** |
| **refused rows gaining the match's stamps** | — | **0** |

**Legitimate identity lost — 3 of 124 real rows**, stated rather than hidden: `NAD` (identifiers
only; its name was already correct), `PLP → Pyridoxal 5'-phosphate`, `Zn²⁺ → Zinc`. All three carry
`row_mapped_ids = {}`. Common cofactors are safe because **66/124 rows resolve by
`pathbank_compound_id` at confidence 1.0 on the strong-id branch and never reach rule 2** — `ATP`
among them. The precise safety claim is "an identifier that *hits* the DB"; an id-less abbreviation
is refused by design.

### Related vendor-data finding (recorded, no gold change)

PathBank row 104723 is **internally inconsistent**: its *name* asserts the C16 dinor homolog while
its KEGG `C01226`, ChEBI `57411` and PubChem assert C18 12-OPDA — and PathBank has **no row for C18
OPDA at all**. The correct canonical target does not exist, so the only correct outcome for "OPDA"
is no rename. PathBank is a vendor source, not the gold set; the contract's remedy for inconsistent
vendor data is the confidence bar this decision restores.

---

## D-029 — An unreachable database is `review_required`, not death · 2026-08-14 · LOCKED

Recorded because `REV-050c` deferred it as **DEF-2**: C-050's `prefreeze_resolution.py`
introduces `_REVIEW_REQUIRED_REASONS = {"resolution_report_not_ok:db_unavailable"}`, which
**reinterprets D-015 clause 6** ("fail visibly on ambiguous or dangling references"). A
reinterpretation of a LOCKED decision belongs in this file before it merges, not in a module
constant. The behaviour is ruled correct; only its provenance was missing.

**The ruling.** When pre-freeze compound resolution cannot reach the PathBank database,
`db_unavailable` is a **`review_required` outcome**. It **must not, by itself, raise a fatal
exception** and must not abort the run.

**Why.** Permanent **merge rule 7** requires incomplete-but-correct pathways to be preserved as
`review_required` rather than dropped, and `PRODUCT_CONTRACT` §1 names a terminal blocker with no
usable recovery as unacceptable. An unreachable database is an **infrastructure condition, not a
defect in the graph**: the biology the operator supplied is unchanged and still correct, merely
un-enriched. Killing the run discards correct work to punish a network failure.

**Scope — narrow, and the distinction is the point.** D-015 clause 6 is **undisturbed for
structural failures.** The four structural codes still raise, at the real entry point
(`run_prefreeze_resolution` has no `try/except` around the call), and `REV-050c` verified this:

* an **ambiguous** rename;
* a **dangling** reference after propagation (`PREFREEZE_RENAME_NOT_PROPAGATED`);
* a **row-count change** across resolution (`PREFREEZE_ROW_COUNT_CHANGED`);
* a **connectivity-signature** change.

Those are defects in the payload or in the resolver, and they must still fail visibly and loudly.
`db_unavailable` is not one of them.

**Fail-closed in the direction that matters.** An unreachable DB yields **fewer** admissions and
**more** review flags — never a rename, never an identifier stamp, never an invented identity. It
cannot increase PWML output, so it does not engage merge rule 6.

**Consequence, and the debt this leaves.** A propagated flag is not an enforced one. C-050's
`run_prefreeze_resolution` report — including this `review_required` — is currently **discarded**
by its caller (`streamlit_app.py:3587-3591`). The seam that can persist and act on it is
**C-052's** (finding: C-050 D1 / DEF-1, "`review_required` has no reader"). This decision rules
what the outcome *is*; it does not claim the outcome is yet acted upon downstream.

---

## D-030 — C-050, C-045 and C-051 land as one atomic stack · 2026-08-14 · LOCKED

**The finding that forces this.** `REV-050c` proved C-050's headline **A9** acceptance fails on
the combined state: one residual post-freeze mutation,
`db_status: 'matched_offline_name_index' → 'legacy_id_unverified'` on `Glycine` (category 5,
identity materialization). The original `0/0/0/0/0` reading was **circular** — it held only
because the un-corrected fixed-point loop happened to converge on exactly the
`legacy_id_unverified` value the exporter re-derives. C-050's provenance correction, which is
right and which was required, removes that coincidence and **exposes a real post-freeze identity
mutation**. So the standing question is answered: **the exporter's second `_resolve_compound_rows`
pass at `ir.py:911` is neither a zero delta nor idempotent by construction.**

**The ruling.**

1. **A9 is not weakened, restated, or scoped down.** All five post-freeze mutation categories must
   read zero **at the landing boundary**. Merge rule 8 is **not waived**.
2. C-050 alone **cannot** satisfy A9, because A9 can only read zero once the exporter's pass is
   either idempotent (now disproven) or removed — and removing it is **C-051's** job.
3. C-045 depends on C-050's `prefreeze_resolution.py` module; C-051 depends on the resulting
   prefreeze sequence. The three are therefore **one stacked dependency cohort**, built in the
   order `corrected C-050 → C-050d → C-045 → C-051`.
4. **No intermediate card is merged into integration.** Each card's exact delta is reviewed
   independently against its **declared direct parent**. After all are approved, the expensive
   combined-state gates run **once** on the top C-051 tip, and the whole stack lands in **one
   serial `--no-ff` composite merge**.
5. The composite merge **preserves the individual card commits and their review evidence**. No
   squash, no rebase, no cherry-pick, no flattening.

**Why atomic rather than sequential.** Landing C-050 by itself would put integration into a state
that **contains a known post-freeze biological mutation** — precisely what merge rule 8 forbids —
for the whole interval until C-051 lands. The alternative considered and rejected was restating
A9 to the four categories C-050 does discharge; that would have weakened a locked acceptance to
accommodate a sequencing artefact. This decision is an explicitly authorized exception to
one-card-at-a-time landing, granted **because** it is the option that never lets the violation
exist in integration.

**Shared-file proof standard.** Because the stacked cards legitimately share files, per **O-12**
each card is compared against its **direct parent**; the other authorized cards must be the only
additional changes; test-function counts are preserved where applicable; and every card's markers
must survive. A byte-identical proof across legitimately shared stacked files is impossible and
must not be demanded.

### The node15 repair is C-050a's, not C-050's

C-050 edited `tests/test_streamlit_quarantine_boundary.py :: test_research_mode_keeps_the_unmapped_
candidate_and_does_not_block` — **node15**, which is **C-050a's owned test function**
(`MASTER_PLAN` §3, hotspot 9). The hunk restored a `.get("final_mapped_db") or ...` fallback that
C-050a deliberately removed, compared against a pre-enrichment artifact, and replaced whole-object
equality with a five-field allowlist. **PACK3 RULING 1 pre-declared a partial-field weakening a
reject.** The orchestrator authorized that edit in error and has withdrawn it.

The hunk is removed in full. The residual failure is routed as **C-050d**, a **test-only** subcard
under **C-050a's** ownership, which must preserve the original invariant — quarantine forwards the
complete post-enrichment object unchanged, compared by **whole-object equality**, with a
non-vacuous fixture. C-050d may not use a field allowlist, may not restore the `.get(...)`
fallback, and may not use two aliases to the same in-place-mutated object as its pre/post
comparand. If production work proves necessary, that card **stops and reports** rather than taking
it.

---

## D-031 — D-028's DrugBank exclusion is ratified · 2026-08-14 · LOCKED

**D-028 clause 4** limits short-abbreviation corroboration to **KEGG, ChEBI, PubChem and HMDB**
and excludes **DrugBank** "unless separately ruled". Finding **P4-03** recorded that exclusion as
correct but **awaiting ratification**. It is now ratified.

**The exclusion stands.** It is **fail-closed**: withholding a namespace from the corroboration
set can only produce **more refusals, never more admissions**, so it cannot weaken a biological
gate or increase PWML output. Ratifying it costs nothing that a later measurement cannot restore.

**How it may change.** Only by a later decision that **measures** DrugBank identifier agreement
against PathBank rows on the committed corpus and demonstrates the namespace corroborates
reliably. Until such a measurement exists, an agent may **not** add DrugBank to the corroborating
set, and may not treat its absence as an oversight. **P4-03 is closed.**

---

## D-032 — the pre-freeze sequence must run at BOTH production export entry points, and that is a prerequisite for C-051 · 2026-08-15 · LOCKED

**Measured, not inferred.** `REV-045a` drove the documented README command
(`python scripts/run_pwml.py --in <payload> --out-dir <out> --non-strict-db`) over a
taxonomy-identified strain at `0ec64d2c` and at `d146be48`:

```
BASE  pathway.pwml -> <name>Lactococcus lactis</name>
      IR species   -> raw_name + aliases preserve "…subsp. lactis KF147"
      report       -> name_canonicalization.species = [deterministic_strain_normalization]
                      preflight.species = ["Lactococcus lactis"]

TIP   pathway.pwml -> <name>Lactococcus lactis subsp. lactis KF147</name>
      IR species   -> NO raw_name, NO aliases
      report       -> name_canonicalization.species = []   preflight = null
```

The exported organism **changes identity**, and the provenance carrier is gone with it, so the row
wears the un-normalized name with **no record that normalization was ever owed**. That violates
`PRODUCT_CONTRACT` §5 (organism/species equivalence is must-remain-equivalent) and **D-016**'s own
requirement to *"preserve or record organism/species provenance"*.

### The structural cause

`run_prefreeze_resolution` has **exactly one** production caller: `streamlit_app.py:3587`.
`writer.py` contains **zero** prefreeze references, yet `run_pwml_pipeline_export` (`writer.py:2642`)
calls `build_pwml_ir` at `:2662` and is reached from `scripts/run_pwml.py:12-16`, documented at
`README.md:40`. `docs/pathwhiz_requirements.md:316-318` and `:523-525` name **"the two production
entry points"** and identify the CLI one by symbol. `MASTER_PLAN.md:420` confirms **no card owns
`run_pwml_pipeline_export`**, so this was a gap, not a sequenced interim state.

**No single card erred.** C-045 correctly removed the in-exporter ladder call under D-016 and merge
rule 8; C-050 correctly added the pre-freeze call to Streamlit; C-045 honestly re-pointed
`test_pwml_writer.py`'s species assertions through a pre-canonicalized payload; C-045a honestly
re-baselined the golden. The defect is **emergent across four cards**, each link individually
defensible and documented. The orchestrator's C-045a charter compounded it by asserting the
standalone `build_pwml_ir` configuration was *"not the production path"* — singular, and false.

### ⚠ The same defect is queued to recur, for compounds

Measured at `d146be48`: `_resolve_compound_rows` is **still called at `ir.py:979` inside
`build_pwml_ir`**, so the CLI path currently still receives **compound** resolution from the
exporter — and **C-051's charter is to remove exactly that call site.** Landing C-051 while the CLI
seam is unwired would strip compound resolution from a documented production entry point, on the
primary biology of a metabolic pathway.

### The decision

1. **The pre-freeze sequence must run at both production export entry points.** It is wired once,
   for the whole `PREFREEZE_CANONICALIZERS` tuple, never per-stage — so every present and future
   canonicalizer is covered by construction.
2. **`C-045b` owns `writer.py :: run_pwml_pipeline_export`** and is the card that wires it. No other
   card may add a competing seam.
3. **C-045b is a hard prerequisite for C-051.** Build order becomes
   `C-050e → C-050d → C-050f → C-045 → C-045a → C-045b → C-051 → ONE composite --no-ff merge`.
   **C-051 must not be dispatched until C-045b is approved.**
4. **C-051's acceptance is extended:** it must show that removing the `ir.py:979` call site leaves
   compound resolution intact on **both** entry points, measured through the CLI, not argued.
5. **A pre-freeze stage may not be considered wired on the strength of one caller.** Any future card
   moving work into `PREFREEZE_CANONICALIZERS` must demonstrate coverage at **both** entry points.
6. **`preflight` and `name_canonicalization` are product-visible export content**, not diagnostics.
   Losing them on an entry point is a regression, not a cosmetic change.

### What this does not do

It does not reopen C-045, C-045a or any merged card, and it does not authorize editing
`prefreeze_resolution.py`, `build_pwml_ir`, or the 32 `GOLDEN` digests — `REV-045a` independently
reproduced all 32 at both SHAs and they are correct. C-045a's measurements were sound; only its
**significance claim** was wrong, and that claim originated in the orchestrator's charter.

---

## D-033 — D-032 clause 1 undercounted: there are THREE export entry points, not two · 2026-08-15 · LOCKED

**This amends D-032, which is LOCKED and stands in every other respect.** Clause 1 said the pre-freeze
sequence must run at *"both production export entry points"*. **That enumeration was wrong.**

**Measured by C-051, confirmed independently by the orchestrator on `328862ab`:**

```
streamlit_app.py :: run_post_pipeline_sbml_artifacts   2617-3809
                      run_prefreeze_resolution   :3587      <- the seam
streamlit_app.py :: run_pwml_export                    3828-4136
                      build_pwml_ir              :4052      <- OUTSIDE the seam's function
        reached by _render_review_refine_section:2231
                -> _generate_pwml_from_refinement_working_json:1898
                -> run_pwml_export
```

The **refinement re-export** path reaches `build_pwml_ir` **without ever entering the function that
holds the pre-freeze call.** It is a third export entry point. At base it was invisible because the
exporter's `_resolve_compound_rows` (`ir.py:979`) silently repaired those rows **after the freeze** —
exactly what merge rule 8 forbids. C-051's removal of that call converted the silent repair into a
loud refusal, which is **the card working correctly**: `qb` node06 turning red is the defect becoming
visible, not the card misbehaving.

### The amendment

1. **D-032 clause 1 is corrected to THREE entry points:** the Streamlit post-pipeline export
   (`run_post_pipeline_sbml_artifacts`, wired by C-050), the CLI export
   (`writer.py :: run_pwml_pipeline_export`, wired by C-045b), and the **refinement re-export**
   (`streamlit_app.py :: run_pwml_export`, **unwired**).
2. **`C-051a` owns the third seam** and lands on C-051's approved tip, before the golden re-baseline.
   `run_pwml_export` and the refinement path are C-030/C-052 surface; C-051a is granted exactly that
   seam and nothing else.
3. **D-032 clause 5 is strengthened.** It required demonstrating coverage at "both" entry points.
   It now requires a card to **enumerate every caller of `build_pwml_ir` by measurement** and show
   coverage at each. **An enumeration inherited from a charter, a decision or a prior card is not
   acceptable evidence** — this clause has now been wrong twice.
4. **`qb` node06 is expected RED between C-051 and C-051a**, and that red is a **PASS condition** for
   C-051 in the same way A9's fifth category was a pass condition for every card before it.
   **It must not be chased, silenced, or fixed by adjusting node06.**

### Why this keeps happening, recorded so it stops

Four site-count errors on this stack, each found by measurement after a charter asserted a smaller
number: **C-050f** — two rewrite sites where the charter named one; **C-045b** — two production entry
points where the charter named one; **C-051** — two `pathwhiz_id` materialization sites where P2-06
and D-027 named one (the orchestrator warned of this at dispatch and the STOP condition correctly did
not trigger, because the compound path runs through `_entity_record:447` while `_component_record:419`
serves species, locations, cell types and tissues); and **this one** — three export entry points
where D-032 named two.

**The pattern is the control plane asserting a count that was true when written and stale when read.**
Standing requirement, binding on every future charter and decision in this sprint: **a claim about
"the" call site, "the" entry point or "both" of anything must carry the measurement that establishes
the count, and the card must re-derive it at its own base rather than inherit it.**

---

## D-034 — duplicate compound spellings: the refusal is ratified, fail-closed, with its cost recorded · 2026-08-16 · LOCKED

**Ruled by the product owner** after the composite reviewer blocked the eleven-card stack's landing and
C-050g measured that the obvious fix does not close it.

### What was measured

`runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict` carries **44
compound rows including four spellings of one molecule** — `#20 'glycerol-3-phosphate'`,
`#36 'sn -glycerol 3-phosphate'`, `#37 'sn -glycerol-3-phosphate (G3P)'`, `#38 'sn-glycerol 3-phosphate'`
— and a pre-existing two-row collision, `#5 'lipid IV_A'` / `#23 'lipid IV A'`.

* **At integration base the leg exports**, producing **43** IR compounds from 44 rows — because the
  **post-freeze exporter silently merged the `lipid IV A` duplicate**. That is precisely the merge-rule-8
  violation this stack exists to remove.
* **At the stack tip the leg refuses.** The pre-freeze stage canonicalizes `#36`/`#38` onto
  `Glycerol 3-phosphate`, whose `_norm` already equals row `#20`'s. `#20` is **not in the rename map**, so
  neither half of `_reject_ambiguous_renames` can see it; the new three-way collision breaks the alias
  index's single-owner resolution and `PREFREEZE_CONNECTIVITY_BROKEN` fires.
* **C-050g's whitespace-collapse fix is correct and lands, but does not change this** — it moves the abort
  from `AMBIGUOUS_RENAME_TARGET` to `PREFREEZE_CONNECTIVITY_BROKEN`. The abort relocates; it does not lift.

### The gap

**No legal route exists today for a payload carrying duplicate compound spellings.** The pre-freeze stage
may not merge rows (`PREFREEZE_ROW_COUNT_CHANGED`); the exporter may not merge them after the freeze
(**merge rule 8**); and **D-015 clause 5** requires participant connectivity be preserved. Such a payload
therefore either exports via an illegal silent merge, as at base, or does not export, as at tip.

### The decision

1. **The refusal is ratified. Fail-closed stands.** A payload whose canonicalization produces a name
   collision with an untouched row **refuses**, and produces no PWML.
2. **`PRODUCT_CONTRACT` § 1 is knowingly not met for this class**, and that is accepted deliberately rather
   than overlooked. The only alternative available today is the merge-rule-8 violation the stack removes.
   **A silent post-freeze merge is worse than an honest refusal**: it invents biology the frozen graph does
   not carry, and it did so undetectably for the entire prior history of the exporter.
3. **C-050g lands**, and the golden re-baselines onto **`PREFREEZE_CONNECTIVITY_BROKEN`** for
   `PMC12444477…/strict` under configs A, C and D. That code is the **honest** diagnosis;
   `AMBIGUOUS_RENAME_TARGET` claimed two distinct compounds where there is one molecule spelled two ways
   plus a collision the guard cannot see. `PMC13278307…/strict` under `C_canned` is a **correct** refusal on
   genuinely distinct compounds (`PEtN-lipid A` vs `modified Lipid A`) and **must keep raising**.
4. **The merge policy is routed as a follow-up card**, not resolved here. It must decide whether the
   pre-freeze stage may merge rows whose canonical names coincide — which requires a
   `PREFREEZE_ROW_COUNT_CHANGED` exemption and a **D-015 clause 5 reinterpretation** — or whether a named
   refusal code should replace the current diff-string diagnosis.
5. **`_reject_ambiguous_renames` is recorded as structurally blind** to a collision between a rename target
   and a row that is not itself renamed. It groups only over `rename_map` sources. Any future card touching
   compound canonicalization must not assume that guard covers the case.
6. **The remaining half of F-8 is subsumed here.** Whether `AMBIGUOUS_RENAME_TARGET` should ever be a hard
   abort for genuinely distinct compounds rather than `review_required` is folded into clause 4's card,
   since both questions are "what should a canonicalization conflict do to the run".

### What this does not do

It does not authorize merging rows, changing `_norm` or `_canonical`, weakening any structural code, or
reopening any accepted card. It does not claim the leg's loss is acceptable in the long run — it records
that the loss is **preferred to the silent merge** until the follow-up card rules, and that the cost is
**known, measured and attributable** rather than discovered later by a user whose pathway vanished.

## D-035 — duplicate canonical rows: consolidation requires proven identity, never coincident spelling · 2026-08-16 · LOCKED

**Ruled by the product owner**, discharging **D-034 clause 4** and the remaining half of **F-8**. This is a
*policy* ruling issued ahead of the implementation card so that the card does not stop to ask "merge or
refuse". It fixes the **bar**; measurement decides only **which groups clear it**.

### The decision

1. **Coincident names are not identity.** Rows are **never** merged merely because their normalized or
   canonical names coincide.
2. **Pre-freeze consolidation is permitted** — and only pre-freeze — when deterministic evidence proves the
   rows represent the **same biological entity**. Nothing here relaxes **merge rule 8**: the exporter still
   may not repair biology after the freeze.
3. **Proof of equivalence requires all four of:**
   a. the **same entity class**;
   b. **no conflicting non-empty stable identifiers**;
   c. **either** at least one **matching stable external identifier**, **or** authoritative resolution
      provenance mapping both rows to the **same database entity**;
   d. **no conflicting structural or biological attributes** that would make consolidation lossy.
4. **Spelling-only resemblance without identity corroboration is insufficient.** A group that resembles
   itself and nothing more does not clear the bar, however obvious the resemblance looks to a reader.
5. **When equivalence is proven**, the stage must: merge **before the freeze**; choose the survivor
   **deterministically**; preserve **every original spelling** in aliases / raw-name provenance; preserve
   and **union** compatible identifiers and provenance; **rewrite all references before the freeze**; record
   **exactly which rows collapsed and why**; and reduce the row count **only** by the number of
   proven-equivalent duplicates.
6. **When equivalence is not proven**, the stage must: **not merge**; **not emit PWML with ambiguous
   connectivity**; return a **named, machine-readable fail-closed reason** in place of the current opaque
   diff-string diagnosis; and preserve enough diagnostic information for review.
7. **Genuinely distinct compounds that collide under canonicalization** remain distinct, or cause a **named
   non-exporting review outcome**. They are **never** silently coalesced. `PMC13278307…/strict` under
   `C_canned` (`PEtN-lipid A` vs `modified Lipid A`) is the reference case and **must keep refusing**.
8. **`AMBIGUOUS_RENAME_TARGET` must not become a successful export.** The card may convert it into a
   structured review-required or refusal result **only if** the graph remains intact and **no invalid PWML
   is emitted**.
9. **The D-015 clause 5 reinterpretation is narrow.** Row-count change is permitted **only** for explicitly
   proven-equivalent duplicate groups, and the `PREFREEZE_ROW_COUNT_CHANGED` exemption extends no further
   than that number.

### What this ruling deliberately does not decide

It does **not** assert that any particular committed group clears clause 3 — including the four-spelling
`glycerol-3-phosphate` group of **D-034**. Whether the committed data actually carries the identifiers or
resolution provenance required by clause 3c is a **measurement**, and it is the implementation card's first
obligation. **A measured finding that no committed group clears the bar is a valid and acceptable outcome**;
the D-034 leg is then *correctly* still refusing, and the ruling has been applied rather than defeated.
**Evidence must not be stretched to recover that leg.**

It does not authorize changing `_norm` or `_canonical`, weakening any structural code, permitting
post-freeze merges, or reopening any accepted card.

### Why the bar is set here

**D-034** recorded that the old exporter's silent post-freeze merge of `lipid IV_A` with `lipid IV A` was
undetectable for the whole prior history of the exporter — it invented biology the frozen graph did not
carry. The failure mode this ruling guards against is the same one in a new costume: a consolidation that
*looks* obviously right because two strings resemble each other, and is wrong because they are two
molecules. Requiring an identifier or an authoritative resolution to the same database entity is what makes
a collapse **checkable by someone who was not present when it was decided**.

**D-034 clause 5 stands unamended**: `_reject_ambiguous_renames` remains structurally blind to a collision
between a rename target and a row that is not itself renamed, because it groups only over `rename_map`
sources. Any implementation of this ruling must supply its own detection rather than assume that guard.

## D-036 — C-050h is scoped to the refusal path only; the consolidation engine is deferred, not cancelled · 2026-08-16 · LOCKED

**Ruled by the product owner** after the D-035 census measured that **no committed duplicate group clears
D-035 clause 3**. This entry deliberately records the rejected alternatives and the measurements behind
them, so a later card can reopen the question **without re-running the census**.

### What the census measured

Across all **32 committed `final_mapped.json`**, **exactly one leg** carries any duplicate-canonical group.
**Zero name-colliding groups clear the bar, and the only groups that clear the bar do not collide.**

| Group | Rows | Verdict | Decisive evidence |
|---|---|---|---|
| `glycerol-3-phosphate` full | 20, 36, 38 | **NOT-PROVEN** | KEGG conflict `C03189` vs `C00093` (3b) |
| closest sub-pair | 36, 38 | **NOT-PROVEN** | satisfies 3c twice (KEGG + ChEBI agree; both → PathWhiz 81) but **fails 3b** — `pathbank` 81 vs 247666, `pubchem` 439162 vs 3393 |
| `sn -glycerol-3-phosphate (G3P)` | 37 | **NOT-PROVEN** | **zero identifiers**; provenance affirmatively `novel/no_db_candidates`, confidence 0.0 — can never satisfy 3c |
| `lipid IV_A` / `lipid IV A` | 5, 23 | **GENUINELY-DISTINCT** | `pathbank` 40982 vs 40738; `chebi` 58603 vs 60365; cytoplasmic vs *E. coli* state |
| `PEtN-lipid A` / `modified Lipid A` | 4, 12 | **GENUINELY-DISTINCT** ✔ | KEGG `C21995` vs `C22003` — the clause 7 reference case, still refusing as required |

**The D-034 leg is correctly still refusing and is not recoverable under D-035 as written.** Note also that
merging 36/38 would not help: row 20 still owns the `_norm` key `glycerol 3 phosphate`, so the collision
raising `PREFREEZE_CONNECTIVITY_BROKEN` survives the merge.

**D-034's account of that leg understated the error.** Row 23 `lipid IV A` carries row 6
`Kdo-lipid IV_A`'s **exact identifier triple** (`pathbank` 40738 / `kegg` C06025 / `chebi` CHEBI:60365). The
pair the old exporter silently merged was not one molecule spelled twice — it merged two **different
molecules in different biological states**. This strengthens D-034 rather than weakening it.

### The decision

1. **C-050h implements D-035 clause 6 only** — the refusal path. It replaces the opaque diff-string
   diagnosis with a **named, machine-readable fail-closed reason**, keeps the graph intact, emits no PWML
   with ambiguous connectivity, and preserves enough diagnostics for review. **`AMBIGUOUS_RENAME_TARGET`
   must not become a successful export** (D-035 clause 8) still binds.
2. **The consolidation engine — D-035 clauses 2 through 5 — is NOT built.** Deterministic survivor
   selection, alias/raw-name preservation, identifier and provenance union, pre-freeze reference rewrite,
   and the narrow `PREFREEZE_ROW_COUNT_CHANGED` exemption are **deferred**.
3. **D-035 is unamended and remains the governing bar.** Nothing here relaxes it. When a payload is later
   measured to clear clause 3, the engine is chartered then, against D-035 as written.

### Why the engine was deferred — recorded so it is not re-argued from scratch

Building it now would ship machinery that **no production input can reach**, exercised only by synthetic
fixtures. Untested-in-production consolidation code that merges biological rows is precisely the class of
risk this sprint exists to reduce, and an unused path invites a future card to "fix" a fixture rather than
the data. The smaller diff also avoids golden churn.

**The counter-argument, preserved:** the capability would exist the moment real data qualifies, and the
census covers only *today's* 32 committed legs — a new corpus could qualify tomorrow. **This is the trigger
for reopening**: if a payload is measured to clear clause 3, charter the engine; do not hand-wave it in
under the refusal card.

### The third option, also deferred

Folding **F-039** (`ir._dedupe_named_rows`, still live, still first-wins on name coincidence) and **F-040**
(`process_normalizer._dedupe_named_rows`, upstream, also name-only) into C-050h was considered and
**declined for this card**, because it spans three files and its scope depends on a measurement still in
flight. **It is not dismissed.** Leaving two live name-only consolidators behind a new refusal is a real gap
and the sprint record says so plainly. F-039's own terms bind: **no one may assert a merge rule 8 violation
until it is measured whether `build_pwml_ir` runs after `freeze_canonical_payload` at each of the three
D-033 entry points.**

### Binding constraints carried into C-050h's charter

1. Consolidation must be **triggered by name collision** and only then **proved** by identity. **Triggering
   on identifier equality is measurably unsafe** — **F-043**: `PG` / `PG phosphate` / `(PGP)` all carry
   `pathbank_compound_id` **193**, satisfy every clause 3 sub-test, and would consolidate; **193 is
   UDP-glucose**, which is none of them.
2. Clause 3 must be evaluated on **payload-carried, pre-resolution** identifiers, with **3b ordered before
   3c**. Otherwise the `C_canned` stub stamps `pathbank_compound_id=78` onto **both** `PEtN-lipid A` and
   `modified Lipid A` (`db_resolver.py:458-472`) and **clause 7's must-keep-refusing case consolidates**.
3. The card must **name the normalizer that defines a group** and justify it. Three disagreeing keys are
   live (**F-040**): `_norm` substitutes a space for the non-`[a-z0-9:+ ]` class, `_normalize` deletes it,
   `_collapsed` is a third variant.
4. A new refusal code implies a **golden re-baseline**; that must be explicitly instructed and measured, not
   absorbed.
5. Wrong-identity rows (row 23 carrying row 6's triple) are a **separate mapping defect**, not a
   duplicate-row concern, and must not be repaired here.

---

## D-037 — C-056a is granted a narrow, seam-level boundary expansion so it is buildable · 2026-08-17 · LOCKED

**C-056a could not be built inside its planned `MASTER_PLAN` §9 boundary, and that had been surfaced to
the product owner as a blocking question.** Its wiring call site is in `strict_quarantine.py`, which it does
not own, and **A0-C3 is unsatisfiable as chartered**: the organism comparison it must fix lives in
`semantic_production.py`, which is **C-017's file**, while A0-C3 simultaneously *requires* reusing
`rag/eligibility.py`'s established synonym/canonicalization behaviour (`_organism_aliases`,
`_canonical_organism`, `_taxon`, including `E. coli` ≡ `Escherichia coli`) and forbids a competing synonym
table. A card cannot be told to reuse a symbol it is not allowed to reach.

**The product owner grants the expansion. This resolves the blocking question; it is not to be asked
again for the seams named below.**

### The grant

C-056a may modify **only the necessary portions** of:

1. its original `MASTER_PLAN` §9 boundary;
2. `strict_quarantine.py`, around the **measured** wiring call site near `:2080`;
3. `semantic_production.py`, around the **measured** organism comparison near `:123-146`;
4. the directly corresponding tests, **including C-017's exact-set pins where necessary**.

**This is a function/seam-level grant, not module-wide ownership.** Line numbers above are the values of
record at the time of the grant; the card **re-measures them** and works from the measured location, per
D-033.

### Conditions — all binding

1. **Preserve all unrelated C-017 behaviour.**
2. **No broad refactor of `semantic_production.py`.**
3. **Measure the existing organism-comparison semantics before changing them.** Quote the current
   predicate; do not infer it.
4. Prefer a **shared pure predicate or a narrow adapter** over duplicated eligibility logic — A0-C3's
   "no competing synonym table" is the whole point of the grant.
5. **Avoid a circular dependency** between `semantic_production.py` and `rag/eligibility.py`. Trace the
   real import graph; do not assume.
6. If importing `rag/eligibility.py` moves C-017's exact-set assertions **only because a new authorized
   symbol appears**: **re-pin narrowly**, preserve the behavioural contract, and add a **behavioural
   discriminator proving the new path is necessary**.
7. **Do not weaken an exact-set assertion merely to obtain green.** First establish whether the set is
   **product behaviour** or only an **implementation-shape pin** — the answer decides whether re-pinning
   is legitimate at all. Merge rule 4 governs: a pinned baseline moves only deliberately, with an exact
   documented delta.
8. Run **C-017's relevant regression tests** and **C-056a's cross-seam tests**.
9. **Every line changed in the granted foreign-owned seams requires independent review.**
10. **STOP CONDITION.** If the fix requires edits **beyond** these named functions and tests, or a broader
    dependency refactor, **C-056a stops and reports the exact measured expansion.** It does not proceed
    under an assumed widening.

### What this grant does not do

It does not give C-056a ownership of `strict_quarantine.py`, `semantic_production.py`, or C-017's suite;
it does not authorize a synonym table of its own; and it does not relax A0-C4 (`evaluated` + `ok` +
`inapplicable_checks` must be combined, because **`confirmed` can never be True on a production run**, so
gating on it alone would ship nothing, ever).

**Sequencing is unchanged: C-056b remains blocked until C-056a is merged and accepted, and C-056b is
chartered from the resulting live source — never from pre-C-056a assumptions. C-053 and C-056b must both
land before T-100, and no strict benchmark-success figure may be quoted until both are merged.**

---

## D-038 — D-004's manifest example is not implementable; what C-053 actually ships · 2026-08-17 · LOCKED

**D-004's normative content is reaffirmed unchanged.** Its three naming rules — `pathway.pwml` =
`release_ready` only · `pathway.review_required.pwml` = valid but requiring biological review · no final
PWML for `diagnostic_only` — match `PRODUCT_CONTRACT` §13 verbatim and are **not** amended here.

**Its illustrative JSON manifest block cannot be implemented as written.** Measured at `3b56a16`:

| Key in D-004's block | Measured reality |
|---|---|
| `pipeline_status` | **0 occurrences in `src/`.** The live row key is `status` (`driver.py:720`). |
| `strict_acceptance_passed` | **0 occurrences in `src/` and `tests/`.** It exists only in `DECISIONS.md`. |
| `strict_acceptance_eligible: true` beside `release_status: review_required` | **Contradicts a live invariant.** `release_status.py:317` is `strict_acceptance_eligible=status == RELEASE_READY`, commented *"review_required must never count as strict success (TRAP-1)"*, and `strict_quarantine.py:2037` names it an invariant. It also contradicts **`PRODUCT_CONTRACT` §13**, which requires `review_required` to carry `strict_acceptance_eligible=false` — *"Never strict success."* |

**`PRODUCT_CONTRACT` outranks `DECISIONS`**, so where the block contradicts §13 the contract governs and the
block yields. The other two keys are not a policy disagreement at all — they name nothing that exists.

### 1. The manifest row ships exactly two new keys

`release_status` — the **full serialized `ReleaseStatus`** (`release_status.py:224`) — and `pwml_artifact`,
the filename actually written, **absent** when none was. `pipeline_status`, `strict_acceptance_passed`, and
**top-level** `strict_acceptance_eligible` / `completeness` are **struck**: the first two do not exist, and
the last two already live inside `ReleaseStatus.to_dict()` (`release_status.py:224-225`), where duplicating
them would create **two sources of truth for a benchmark-gating flag**. `report.py:426` already documents
accepting *"a state string, a serialized `ReleaseStatus`, or `None`"*.

**`strict_acceptance_eligible` is authoritative only inside `release_status`, and is always
`status == release_ready`.**

### 2. The release record is read from memory, never from the artifacts dict

**`outcome.artifacts["quarantine_report.json"]` is a `str`, not a dict** — `driver.py:1295` reads it with
`Path(source).read_text(...)` and `:1299` stores `out[name] = document`. Subscripting it with `["release"]`
**cannot execute**. Any charter or card mandating that expression is wrong.

**The correct source is `pwml_result["quarantine_report"]["release"]`**, which reaches the strict path in
memory as a dict: `streamlit_app.py:4191-4207` returns it on every branch → `:2242` stores it →
`driver.py:1137` reads it → `:2125` binds it → `:2126` and `:2177` pass it to `_add_strict_artifacts` and
`_finalize_pwml_export`, **both of which already declare a `pwml_result` parameter**
(`driver.py:1373-1376`, `:1673-1679`).

This is also the only **sound** source. The batch strict export runs through **EP3** (D-033), which exports
`refinement_working_json`, not the object `freeze_canonical_payload` froze; the in-memory record is bound to
the exported graph by `decision_matches` (`streamlit_app.py:3888`) on `resulting_payload_hash` **and** the
decision-input hash, so it provably describes the graph that was exported. **Reading the disk artifact would
hand C-053 a record for a different payload than the XML it is naming.**

**C-053 must not call `classify_release_status` on the PASS path.** Re-deriving a classification after the
freeze is a **merge rule 8 reject**. The existing call at `driver.py:1629` (`_finalize_gate_failure`) is out
of scope and untouched.

### 3. When the frozen record is absent or unrecognized on a passing strict leg

This state — pipeline passed, classification unavailable — is enumerated nowhere. It resolves entirely from
locked policy:

* **Do not use `pathway.pwml`.** `PRODUCT_CONTRACT` §13 reserves it for `release_ready`, and a run that
  cannot produce its classification cannot prove `release_ready`.
* **Do not drop the bytes.** **Permanent merge rule 7** preserves incomplete-but-correct work. Write
  **`pathway.review_required.pwml`** — §13 defines it as *"valid, needs review"*, which is exactly the state.
* **Emit no `release_status` key**; `report.py:860-861` already renders that honestly. §13: *"Structured
  status is authoritative; the filename is a migration aid."*
* **Append a `RunOutcome.warnings` entry** naming the missing record. Fail loud, not silent.
* **The strict denominator gates on an affirmative `strict_acceptance_eligible is True`**, so an absent
  record is excluded automatically. **An unavailable classification must never be able to inflate
  `strict_ok`** — no new strict success without measured evidence.

### 4. D-004's site citations are stale and must be re-derived

D-004 cites `driver.py:1319`, `:2008`; those are now `driver.py:1380` (`out["pathway.pwml"] = xml`) and
`:1689`. Per **D-033**'s standing rule, C-053 **re-measures the site count at its own base** and does not
inherit "four".

---

## D-039 — C-056a's rulings: the grant is buildable without moving a single exact-set pin · 2026-08-17 · LOCKED

**D-037 granted the boundary. This decides how it is used.** The C-056a charter ended with eight owed
rulings, none recorded. Re-derived against live source, the charter's own **central blocking finding was
false**, and a cheaper path exists that moves **no** pinned assertion in `test_semantic_production_no_gold.py`.
Two of its closeout items were already fixed. Three unlisted breakages were found.

### 1. Import the PUBLIC wrapper, function-locally. `test_e` does not move.

The charter asserted C-056a *"cannot add a public re-export … it must import the private names directly"* and
that this forces a merge-rule-4 re-pin of `test_e`'s exact sets. **Both halves are false.**

* **A public wrapper already exists and is already read-only sprint material.**
  `rag/admission.py:2046` `compare_organism(requested, observed)` returns
  `match` / `genus_level` / `mismatch` / `unknown` and delegates taxonomy to `eligibility._compare_one`
  (`admission.py:2033-2037`). `MASTER_PLAN.md:147` lists
  `admission.compare_requested_pathway`/`compare_organism` with disposition **`read`**. **Use it. Do not
  import private `eligibility` names.**
* **A function-local import provably does not move the pin, and the proof is already in the file.**
  `semantic_production.py:166-169` already function-locally imports `t2pw.pipeline.entity_identity`, and
  `t2pw.pipeline` appears **zero times** in `tests/test_semantic_production_no_gold.py`. The `test_e` probe
  is a **fresh `subprocess.run`** (`:134`) that only imports the module; it never calls a function.

**Therefore the import MUST be function-local, and `tests/test_semantic_production_no_gold.py::test_e`
stays byte-identical. A module-level import is FORBIDDEN** — `graph["own"]` (`:142`) is
`set(sys.modules) - package`, i.e. **all** modules including stdlib, so a module-level import would also drag
in `math` (`config.py:16`, `store.py:21`) and others. **A diff containing a module-level `rag` import is a
reject, not a re-pin request. D-037 condition 6 does not fire.**

### 2. The organism widening is strictly monotone

The live predicate is `semantic_production.py:134`:
`if not norm or norm == wanted or norm.startswith(wanted + " ") or wanted.startswith(norm + " "):`
Hand-evaluated, requested `Escherichia coli` / observed `E. coli` emits a **finding** — the false positive is
real, and canonicalizing only the requested side cannot fix it.

**A row is compatible if and only if `not norm`, OR the existing disjunction holds, OR
`compare_organism(requested, observed) == ORGANISM_MATCH`.** The change may only **remove** findings, never
add one.

**`genus_level` is NOT newly tolerated and NOT newly penalised.** It straddles two *opposite* current
behaviours — bare genus `Escherichia` is already tolerated by `wanted.startswith(norm + " ")`, while
`Escherichia fergusonii` already emits a finding — so mapping it either way silently changes shipped
behaviour in one direction. **Both existing behaviours stand unchanged.** The diff carries the
requested/observed verdict table as its documented delta. The charter's proposed assertion that bare genus
*must not* match is **rejected**: A0-C3 does not say it, and adopting it would be a new undocumented
tightening.

**Out of scope:** the same literal predicate is duplicated at `bench/semantic.py:753` (`_organism_conflicts`,
the **gold** path). **Do not touch it.**

### 3. The semantic gate fires on a closed set of four

`evaluate_production_semantics` populates eight checks (`semantic_production.py:287`). **Gating set:**
`CHECK_ANCHORS`, `CHECK_ORGANISM`, `CHECK_ID_CONFLICT`, `CHECK_RAG_REINTRODUCTION` — declared as a named
constant with a test asserting it is closed.

* `CHECK_PLACEHOLDER_IDENTITY` **never gates** — `PRODUCT_CONTRACT` §13 / TRAP-3; it is already explicitly
  non-adjudicating (`semantic_production.py:159-163`).
* `CHECK_SUPPORTED_REACTIONS` is always inapplicable in production (`semantic_production.py:278`).
* `CHECK_SOURCE_CARRIER` and `CHECK_CONNECTED_CORE` are **recorded but NON-GATING**, revisited by C-056b
  with measured evidence.

**Why the last two do not gate.** `_check_source_carrier` documents itself as *"Hygiene only … Deliberately
does NOT claim the reaction is supported"* (`semantic.py:485-491`). **Using a check that explicitly
disclaims biological meaning to block a biological release would misuse it**, and it demonstrably over-fires:
it is exactly what breaks the reachability assertion in §4. `_check_connected_core`
(`semantic_production.py:227-244`) duplicates a floor the coverage verdict already enforced at the same seam
(`min_core_processes`, `:1800`), so gating on it double-counts. **This decision only ever removes strict
successes relative to today's unwired state; it creates none.**

### 4. A reachability assertion breaks, and it is repaired locally — never weakened

`tests/test_strict_quarantine_release_seam.py:684` asserts
`states == {RELEASE_READY, REVIEW_REQUIRED, DIAGNOSTIC_ONLY}` — that **all three `PRODUCT_CONTRACT` §4
states are reachable through the seam**. That is **product behaviour**, not an implementation-shape pin.

It breaks on wiring: `_base()`'s reactions carry no `evidence`/`source_papers`/`rag_provenance`/`source_refs`,
so `_has_source` (`semantic.py:454-482`) is False, `_check_source_carrier` emits findings, and the
`RELEASE_READY` member demotes. **The file is in NO chunk**, so SMOKE and Chunks A–E all miss it.

**Repair: add a file-local `_sourced_base()` that deep-copies `_base()` and adds an `evidence` string, and
use it for the `RELEASE_READY` member. The set stays whole.**

**STOP — do not edit `_base()` itself.** It lives in `tests/test_strict_quarantine_contract_alignment.py`
with **15 in-file call sites and 3 importing files, two of them Chunk A / inside SMOKE 460**. That is a
D-037 clause-10 expansion.

### 5. No schema bump in C-056a

C-056a writes **only** the two keys already in schema 4 — `semantic_evaluation` and
`semantic_not_evaluated_reason` (`release_status.py:222-223`). Populating existing keys is not a schema
change. **It may not touch `strict_quarantine.py:2155` (`"schema_version": 4`) and may not add a key under
`quarantine_report["release"]`**, because `tests/test_strict_quarantine.py:894` pins `schema_version == 4`
and is **Chunk A, inside SMOKE 460** (merge rule 10), with a second pin at
`test_strict_quarantine_release_seam.py:695`. **`semantic_failed_checks` persistence and the 4-to-5 bump are
deferred to C-056b**, which owns the denominator consumer anyway. Adding the schema bump here fires the
D-037 stop condition into a SMOKE-gated pin.

`decision_identifier` (`strict_quarantine.py:288-303`) hashes only `admitted_payload_hash` +
`decision_input_hash`, so populating `release` does **not** move `decision_id`.

### 6. The layering inversion is ruled explicitly, not absorbed

Wiring at `strict_quarantine.py:2080` would be the **first `pipeline` to `bench` import in the codebase**
(measured: zero today), inverting the layering `bench/__init__.py` declares. **Authorized, narrowly:** the
import is **function-local** at the call site, matching the existing `:2004` precedent, and carries an
in-line comment naming the inversion. It is not a cycle — the forward chain was traced and **no module on
the `rag`/`admission` branch imports `t2pw.bench`**.

### 7. Process

* **Reviewer reassigned from `C-017 impl` to an independent reviewer who implemented neither C-017 nor
  C-041/C-041a.** The grant places C-056a inside C-017's module *and* C-041a's seam; neither implementer may
  review their own territory (`MASTER_PLAN.md:372`). Precedent: C-017's own review was reassigned.
* **`MASTER_PLAN` §9 Chunk D column changes from a dash to `D (qb)`** — measurably wrong today, since `qb`
  observes the seam. The focused set additionally **names** `tests/test_strict_quarantine_release_seam.py`
  and `tests/test_release_status_classification.py`, **neither of which is in any chunk**.
* **`qb` runs as a read-only harvest from the PRIMARY checkout**, tracked-background per D-026, with
  `T2PW_OFFLINE_CURATOR=1` **in the bounded child**. `.env` is present only in the primary (measured absent
  in six sampled worktrees). **A `qb` green obtained in a worktree is never a pass** — label it
  `DB_UNAVAILABLE` and treat it as not run.
* **Serialize against C-057** — both touch `strict_quarantine.py`.
* **`tests/test_streamlit_quarantine_boundary.py`: no test function may be added, removed, renamed or
  reordered** (hotspot 9; the gate addresses nodes positionally).
* **Q4 — D-029 is NOT extended; a cross-reference suffices.** D-029's scope is explicitly narrow
  (`prefreeze_resolution.py`). The equivalent rule already binds independently: `PRODUCT_CONTRACT` §11
  (*"`not_evaluated` is never `false`"*) and merge rule 7, and it is already implemented at
  `semantic_production.py:59-63`. **`DECISIONS.md` is not appended for it.**
* **Q5 — no fabricated base failure and no separate discrimination run.** `SEMANTIC_FAILED` is defined and
  **produced nowhere in `src/`**, and `semantic_production` has **zero `src/` consumers**, so this is
  genuinely new capability. Every preservation invariant is written as a **self-discriminating table**
  containing both a firing and a non-firing case, so deleting the branch *or* over-firing it turns the guard
  red. **Three diff labels are mandatory:** `NEW ACCEPTANCE`, `REGRESSION GUARD (passes at base and tip)`,
  and `MERGE RULE 4 BASELINE MOVE`. **Mislabelling the section 4 move as a regression guard is a reject.**
* **First act, before any edit: measure.** The section 2 verdict table and the section 4 demotion are
  **hand-evaluated from source predicates, not observed**. C-056a's first bounded job runs
  `tests/test_strict_quarantine_release_seam.py` and `tests/test_semantic_production_no_gold.py` to confirm
  both. D-033 forbids inheriting a claim that has not been re-measured.

---

## D-040 — C-052: a narrow key-addition grant, and the ownership claim that was a misread · 2026-08-17 · LOCKED

**The C-052 charter's central ownership claim is false.** It states *"C-052 owns that result dict
(`MASTER_PLAN.md:163`, `DECISIONS.md:513`)"*. Measured:

* **`DECISIONS.md:501`** heads D-021 §2's third column **"Not owned by"**, and **`:513`** —
  `` `streamlit_app.py` :: the enrichment block **above** the C-011 seam, and the pre-freeze call |
  **C-050** | C-040, C-051, C-052 `` — lists **C-052 in that column**. The citation proves the **opposite**
  of the claim it was offered for.
* **`MASTER_PLAN.md:161`** heads its §3 column **"Branches"** — a **collision declaration, not a grant**.
* C-052's actual grants are `DECISIONS.md:515` (`freeze_canonical_payload`, C-030 then C-052) and `:516`
  (`run_pwml_export` and **"the SBML binding"**, measured as `streamlit_app.py:3638`/`:3650`), echoed at
  `MASTER_PLAN.md:424` and `LEDGER.md:222`. **None names the success-return dict at
  `streamlit_app.py:3680-3807`, which no card owns.**

D-029's closing paragraph (`DECISIONS.md:1247-1251`) nonetheless charges C-052 with persisting a report that
is **only reachable there**, and A0-C8's observability half lives there. The grant below resolves that,
narrowly.

### 1. The grant — key-addition only

C-052 **may add keys** to `streamlit_app.py:3680-3807` and **may read** the local bound at `:3587`.

It **may not** change, reorder or remove an existing key · **may not** alter the `:3587-3591` call ·
**may not** touch `freeze_canonical_payload` (`:2509-2614`), which stays at **zero lines**. The seam's
seven-field return at `:2609-2614` already publishes `canonical_json_path`, so A0-C8's data reaches
`:3638` with no change inside the seam.

**Any key C-052 adds must NOT contain the substring `pwml`.** `driver.py:1144-1146` returns the first
`post_pipeline_artifacts` entry whose key matches `"pwml" in key.lower()` **as the PWML export result**. No
key of EP1's return contains `pwml` today, so this fallback never fires from EP1 — a key named
`pwml_prefreeze_resolution_report`, the natural "converge on the CLI" instinct, would make the batch driver
return **the prefreeze report as the PWML export result** on any leg where `pwml_export_result` is empty.
The CLI's own key is `prefreeze_resolution_report` (`writer.py:2803`) — no `pwml` — so genuine convergence
is already safe.

**If this grant is withheld, C-052 ships EP3 only and A0-C8 re-opens under a new owner.**

### 2. EP3 does not write to the CLI's filename

The charter mandates EP3 write `outputs/pwml_prefreeze_resolution_report.json`. **That is the CLI's own
default path:** `writer.py:2690` `prefreeze_report_path = out_dir / "pwml_prefreeze_resolution_report.json"`
with `writer.py:2817` `--out-dir` defaulting to `"outputs"`. Run from the project root the two writers share
one file and **whichever ran last wins, silently**; `tests/test_pwml_writer.py` pins the CLI's summary to
that exact path.

**EP3 writes `outputs/pwml_prefreeze_resolution_report.streamlit.json`.** Convergence with the CLI is on
**content shape**, never on a path two writers share.

### 3. EP3's record is labelled post-freeze, because it is

**F-039 measured EP3 as *"unambiguously post-freeze … MERGE RULE 8 IS VIOLATED HERE"*** — it operates on a
deepcopy-of-a-deepcopy taken after the hash (`streamlit_app.py:3869`, seam at `:4091`). EP1's report
describes a **pre-freeze** canonicalization; EP3's describes a **post-freeze** one over a different object.
Persisting both under one name tells the operator they are the same record.

**EP3's persisted record carries a mandatory top-level `"seam": "post_freeze_refinement_reexport"`.** This
is a **labelling** requirement, not a repair: merge rule 8 is unaffected either way, and **C-050i owns the
violation itself**.

### 4. EP3 writes before `build_pwml_ir`, so the catch-all cannot drop it

`streamlit_app.py:4213-4219`'s `except Exception` wraps everything below the seam, so a failure in
`build_pwml_ir` (`:4135`), the builder (`:4179`) or the write (`:4189`) would lose the report entirely.
**EP3 writes the file immediately after the seam and before `build_pwml_ir`.** The on-disk record then
survives every path and no new key is needed on the failure return.

### 5. A0-C8 is discharged in three parts, and its cohort clause has a measured limit

*"The actual `canonical_json_path`"* **cannot be asserted by equality**: `streamlit_app.py:2594` builds it
under `tmp = temp_root / f"post_pipeline_{uuid4().hex}"` (`:2724`), `rmtree`'d at `:3809`, and
`run_post_pipeline_sbml_artifacts` takes **no `tmp` parameter**, so no test can predict or supply it.

1. **Path identity, unit level, no production change** — patch `build_sbml` in `streamlit_app`'s namespace,
   call EP1 with `build_legacy_sbml=True`, and assert the captured first positional argument **`is`** the
   `Path` the freeze returned as `canonical_json_path`. This pins `:3638` → `:3650` exactly.
2. **Cohort level, 39 legs, no production change** — extend the existing projection in
   `tests/test_c011_freeze_seam_golden_equivalence.py` with `canonical_json_path_name` and
   `canonical_json_path_in_tmp` (**relative/boolean only — never the absolute UUID string**, which would
   make the byte-pinned fixture unregenerable).
3. **Observability** — the additive EP1 result key, under §1.

**`LEDGER.md`'s "observable behaviour only" does not forbid an additive key that *creates* the observable;
it forbids changing an established one.**

**Measured limitation, recorded rather than waived.** A0-C8's clause *"including the path supplied to
downstream SBML generation"* is **unexercised on all 39 cohort legs**: `streamlit_app.py:3648` guards the
`build_sbml` call on `build_legacy_sbml`, and the batch driver binds five widget keys
(`driver.py:120-122`, `:129-130`) of which **`run_legacy_sbml_export_btn` is not one**. The clause is
discharged at unit level (part 1). **C-052 must NOT bind the driver to the legacy button** — that is a
production-behaviour change no card owns.

### 6. A0-C7 is struck from C-052 entirely

The charter's clause C-8 assigns C-052 five identity relations, mutant discrimination and ~120 test lines
for **A0-C7 — which is C-030a's** (`LEDGER.md` carried-requirement table; `MASTER_PLAN.md` §9), and whose
implementer must **not** be C-052's. **Strike C-8 and its budget lines.** C-052 as narrowed touches none of
the five live sharing sites (`:3694`, `:3704`, `:3746`, `:3748`, `:3749`) — it **adds** keys and **changes**
none.

### 7. C-052 does not edit `driver.py`

There is no route from EP1's result dict to the 39-leg cohort artifacts without editing `driver.py`:
`_add_common_artifacts` hard-codes its keys, `_collect_reports` filters on `_CONTRACT_SUFFIXES`, and
`_add_diagnostic_artifacts` reads a six-name allowlist. **D-038 granted `driver.py :: _add_strict_artifacts`
to C-053 on 2026-08-17.** A0-C8's cohort clause is therefore discharged in the golden projection (§5 part 2),
**not** through the batch artifact set.

### 8. Persist and surface only — "acting on it" gets a named owner, or none

D-029's closing paragraph says the seam that can *"persist and act on"* `review_required` is C-052's.
**That is split here.** **C-052 persists and surfaces only**: it adds no branch that changes whether a PWML
is produced, so merge rules 6, 7 and 8 are untouched by construction.

**Acting on `review_required` is assigned to no card and is registered as backlog `BL-004`.** It is **not**
C-053's — C-053's live tripwire excludes `streamlit_app.py` and `outputs/`, and D-004 / `PRODUCT_CONTRACT`
§13 govern `release_status`, a **quarantine** verdict, not the prefreeze verdict. **Naming the residual is
the point**: an unnamed residual is inherited by proximity, which is the exact failure REV-050h caught on
F-039.

### 9. Execution

* **Base = the integration tip at dispatch**, recorded in the LEDGER row at dispatch and **never** in the
  charter — a charter SHA written days ahead is what produced this drift twice. The tree must carry
  `75fbb8a` (C-050h) and `16cd3bd` (H-010).
* **Serialize after C-050i.** Both operate on EP3's post-freeze `run_prefreeze_resolution`
  (`streamlit_app.py:4091`); C-050i changes what that call refuses, which changes what C-052 persists.
  **C-050i merges first.**
* **Base and tip proofs run in a git worktree checked out at the base SHA — never in a
  `c045b_base_tree.py` / `c051a_base_tree_batch.py` export.** `PATHSPEC` omits **`runs/`** and `.git`, so
  `test_c011_freeze_seam_golden_equivalence.py` — which shells `git show` and replays legs from `runs/` —
  degrades to an empty leg set and `KeyError`s against the 39-name fixture. Same class as C-053's F-042
  ruling. **C-052 must not widen `PATHSPEC`.**
* **Gate set: `qb` (`TEST_MATRIX.md` binds C-052 by name) + full Chunk D + SMOKE 460 + a NAMED focused file
  gate.** **Five of the seven files C-052 changes are in no chunk** — the new seam test, the golden
  equivalence harness and its fixture, `test_prefreeze_third_export_seam.py`, and
  `test_prefreeze_compound_resolution.py`. Chunk D and SMOKE would surface a regression in none of them.
* **New tests need no AppTest.** All three C-052 production surfaces — `run_pwml_export`,
  `run_post_pipeline_sbml_artifacts`, `_json_artifact_entries` — make **zero `st.` calls**, so the
  MagicMock-stub convention suffices. The `st.form` hazard at `streamlit_app.py:4310` is an **import-time
  module-script** hazard and **is not a constraint on this card**.
* **D-025 ceilings are re-derived after this decision**, not inherited: the charter's figures budget zero
  runs for the golden-fixture delta and zero for a worktree base proof, and double-count the struck A0-C7.

---

## D-041 — C-050k emits a diagnostic and changes no binding; the refusal question stays open until it is measured · 2026-08-17 · LOCKED

C-050k's charter correctly refused to settle its own central question and stopped for a ruling. This is it.

> When `resolve_entity` finds an ambiguous alias key, does it **(a)** keep today's binding and emit a
> named machine-readable diagnostic, or **(b)** refuse?

### 1. Ruling: **(a) diagnostic-only.** Refusal is not authorized on this card.

Not because refusal is wrong in principle, but because every input that would justify it is currently
unmeasured, and because (a) is the only option that is inside the contract by construction:

* **Changing which entity a reference binds to is post-freeze reinterpretation of biological content.**
  `build_pwml_ir` runs post-freeze at EP3 on a deepcopy-of-a-deepcopy taken after the hash. Merge rule 8
  forbids exporters repairing biology after the canonical graph is frozen, and `PRODUCT_CONTRACT` §5 names
  **process-to-entity references** as a must-remain-equivalent dimension. Option (a) touches neither.
* **Refusal on this path is the shape that turns a working export into a dead one.** All seven call sites
  pass a role in `preferred_order` (`ir.py:1668`, `:1740`, `:1808`, `:1901`, `:1956`, `:2001`, `:2002`), so
  the early return at `ir.py:1530-1533` is the dominant path and every reaction reference crosses it. With
  exposure unmeasured, the blast radius of a refusal is unknown. **Merge rule 7 preserves
  incomplete-but-correct pathways as `review_required` rather than dropping them.**
* **A diagnostic discharges the harm F-048 actually states** — *"the residue is not attributable"*. Making
  it attributable is the whole finding. Nothing is invented and nothing is suppressed.
* **Prefer a narrow measured correction over a speculative redesign.** Refusal remains fully available as a
  follow-up card the moment §2's census produces a number.

### 2. The ruling's three binding limits

1. **Severity is `warning`, matching the emitter that already exists.** `ir.py:1535-1543` already calls
   `_add_issue(report, "warning", "ambiguous_entity_reference", …)`. **Do not escalate it to `error`.**
   `_add_issue` at `"error"` sets `report["ok"] = False` (`ir.py:375`), which would flip legs between
   exporting and not — a production-behaviour change this card is not granted and which would create or
   destroy strict successes without measured evidence, in both directions.
2. **No binding may move.** `tests/test_pwml_ir_duplicate_row_refusal.py:511-512` (`bound_key == "cmp_1"`)
   **must stay green untouched**. If it moves, the card has re-bound a reference post-freeze — reject.
3. **`report["ok"]` may not move on any committed leg.** If adopting the existing severity nevertheless
   moves a golden `_leg_digest`, that is a **merge rule 4 baseline move** needing an exact documented delta
   and orchestrator ratification **before** commit — measured under pytest, never by a direct-import
   harness (**F-047**: a direct-import harness reported a false 32-of-32 move that reproduced identically at
   base; the instrument was the fault, and F-047's stated mechanism was itself falsified by REV-050i, so do
   not reason from the mechanism).

### 3. What still escalates

**If §2's census shows any committed leg actually resolving a reference through an ambiguous key, STOP.**
That is a live reaction bound to a possibly-wrong molecule, a `product_contract_violation` candidate under
`PRODUCT_CONTRACT` §14, and its disposition is a product ruling — including whether (b) becomes mandatory.
It also changes the card's G9 classification from *new capability* to a *correction of pre-existing
observable behaviour* with a real base-failing behavioural proof. **Re-charter honestly; never re-label in
flight.**

### 4. A control-flow correction the charter transcribed wrongly

The charter's §0.1 snippet indents `if len(candidates) > 1:` **inside** the `for wanted in ordered:` loop.
Live, `ir.py:1534` is at the **same** indentation as `ir.py:1530` — the ambiguity check sits **after** the
loop and runs at most once, not once per preferred type. **The conclusion is unaffected** (the branch is
still unreachable whenever any preferred type matches, which is F-048's point), but do not design the
detection against the transcribed control flow.

### 5. `len(candidates) > 1` is not the ambiguity signal

`ir.py:1249-1261` appends once per alias slot **with no dedupe**, so a single entity whose `name` and
`raw_name` normalize alike appears twice under one `_norm`. Testing list length would raise a false
ambiguity on one entity and report its `entity_type` twice. **The test is over distinct entity keys.** This
defect is recorded by no finding and is not C-050k's to fix beyond not being fooled by it.

---

## D-042 — D-039 §4 is struck as measurably false; the wiring must pin its derivation and demote exactly one step · 2026-08-17 · LOCKED

**This corrects a decision this orchestrator issued.** C-056a stopped at its mandatory §0 measurement rather
than build against it, which is exactly what D-033 and the stop conditions exist to produce. An independent
adversarial verifier — instructed to refute, defaulting to refuted — **confirmed** the refutation from its
own probe and then found four things the implementer's framing had smoothed over. All five results are
recorded here.

### 1. D-039 §4's repair is STRUCK. §3 stands unchanged.

§4 predicted the three-state reachability assertion at `tests/test_strict_quarantine_release_seam.py:684`
breaks on wiring, because `_base()`'s reactions carry no source field, `_has_source` is False,
`_check_source_carrier` emits findings, and the `RELEASE_READY` member demotes.

**That mechanism cannot reach the gate §3 defines.** `semantic.py:91` declares
`CHECK_SOURCE_CARRIER = "reaction_source_carrier_present"` as a constant **distinct** from all four gating
names (`semantic.py:81`, `:97`, `:98`, `:99`), and `_has_source` is referenced at exactly **two** sites in
`src/` — its definition (`semantic.py:454`) and inside `_check_source_carrier` (`semantic.py:498`). A check
that does not gate cannot demote. **§3 and §4 were never simultaneously satisfiable**, and §3's own text
gives the game away: it justifies excluding `CHECK_SOURCE_CARRIER` on the ground that *"it is exactly what
breaks the reachability assertion in §4."*

Measured, `_base()` under the metadata-fallback derivation: `requested_pathway_anchors_present` ok,
`organism_compatible` ok, `no_real_id_or_name_conflict` ok, `no_rejected_rag_reaction_reintroduced`
inapplicable, `reaction_source_carrier_present` **the only failure**. Across three members × five
legitimate derivations, **no demotion under the closed four**.

**Consequences, binding:**
* **No `_sourced_base()`. No edit to `tests/test_strict_quarantine_release_seam.py` at all.** The
  three-state set stays whole with no change — strictly better than §4's own goal.
* **The mandatory diff labels reduce from three to two:** `# NEW ACCEPTANCE …` and
  `# REGRESSION GUARD (passes at base and tip) — not a G9 proof`. The label
  `# MERGE RULE 4 BASELINE MOVE — release_seam three-state set, exact delta documented` is **forbidden**:
  emitting it for a baseline that does not move is a fabricated delta, the precise failure G9 and D-033
  exist to prevent.
* **The obligation §4 was reaching for survives in a stronger form.** The card must *verify by measurement*
  that all three states remain reachable at its tip, and if any gating check does demote a member, §4's
  repair returns and the card stops for a fresh ruling.

### 2. The wiring MUST pin its request derivation, and a test must lock it

The verifier's decisive caveat: **the survival is a property of the derivation, not of the payload.** Under
the most literal wiring (`pathway_context=None` on the `RELEASE_READY` member) `CHECK_ANCHORS` and
`CHECK_ORGANISM` are `not_evaluated` — the assertion survives by *skipping*, not by passing. Under an
adversarial sixth derivation (`metadata.pathway_subject`, a PathWhiz **category** rather than a pathway
name) `CHECK_ANCHORS` fails on all three members and `_base` **does** demote.

`pathway_subject` is not a legitimate derivation, so this is not a defect — but it proves the diff cannot
leave the derivation implicit. **The diff pins exactly one derivation, names it in an in-line comment, and
carries a test that locks it.** `entity_admission.pathway_context_from_stage_zero` is the codebase's
single-sourced derivation (`pathway_name` / `likely_organism`|`organism`) and is already `t2pw.pipeline`,
so it needs no cross-layer import; prefer it and justify any departure.

### 3. Demotion is exactly ONE step, and it is a cap, never a move

D-039 left the demotion **depth** unspecified. Settled here:

**A failing gating semantic check caps the release status at `REVIEW_REQUIRED`. It never produces
`DIAGNOSTIC_ONLY`, and it never moves a status that is already `REVIEW_REQUIRED` or `DIAGNOSTIC_ONLY`.**

* `PRODUCT_CONTRACT` §13 defines `review_required` as *"valid, needs review"* — exactly the state of a
  pathway whose semantics did not confirm. **Merge rule 7** preserves incomplete-but-correct work rather
  than dropping it; `diagnostic_only` would drop it.
* Measured: `tests/test_strict_quarantine_release_seam.py:566` and `:579` both survive a
  `release_ready → review_required` demotion, and **`:579` breaks under a `diagnostic_only` demotion.**
  The cap is what keeps the existing suite honest without weakening it.
* A cap is monotone, so the change can only ever **remove** strict successes, never create one — consistent
  with D-039 §3's closing sentence and with *no new strict success without measured evidence*.

### 4. Three of the four gating checks are conditionally or structurally unevaluable at this seam. Say so.

**`CHECK_RAG_REINTRODUCTION` is structurally unevaluable here today.** `quarantine_and_close`
(`strict_quarantine.py:1793-1804`) has **no** `admission` parameter and `strict_quarantine.py` contains
**zero** `rag_admission` / `admission_report` references. `CHECK_ANCHORS` and `CHECK_ORGANISM` evaluate only
under a derivation that supplies them. So wired at `:2080` today, the gate reduces in practice to
**`CHECK_ID_CONFLICT`** as the only unconditionally evaluable member.

**This is contract-compliant and the card still ships.** `PRODUCT_CONTRACT` §11 requires semantic checks to
affect the runtime `release_status`, which they now do; it also mandates that `not_evaluated` is never
`false` and produces no status change, which is precisely what the unevaluable checks do — verified live:
`semantic_production.py:59-63` returns `ok=True` with an inapplicable reason, which lands in neither
`failed_checks` nor any applicability-filtered gate.

**But it must not be reported as four live gates.** The card states the measured evaluability of each of the
four in its report and in a test. **The gating set remains closed at four** — a check being currently
inapplicable does not remove it from the set. **Giving `quarantine_and_close` an `admission` parameter is
NOT granted to C-056a**; it is a signature change on a seam C-057 also touches, and it needs its own card.

### 5. Organism exposure is measured at zero, so C-056a demotes nothing

Across 16 committed run directories (145 legs, **32 with a `final_mapped.json`**): requested organisms are
`Escherichia coli` ×16, `Homo sapiens` ×13, `""` ×3. **Exactly ONE process row in the entire corpus carries
an observed organism** — `runs_verify/2026-08-04_1207/papers/PMC12452463/strict` `/processes/reactions/4`,
`Escherichia coli` against requested `Escherichia coli`, `compare_organism` → `match`, no finding.

**Legs failing `CHECK_ORGANISM` today: 0 / 32. After the §2 widening: 0 / 32. Committed legs newly demoted
by gating on `CHECK_ORGANISM`: zero.** C-056a may proceed.

**The reason is structural, and it falsifies a live comment.** `propagate_context_organism`
(`pipeline.py:2391-2424`) writes organism onto `entities.species`, `entities.proteins` and
`biological_states` — **never onto `processes` rows**, and nothing else does. The rationale at
`semantic_production.py:132-133` claiming *"`pipeline.propagate_context_organism` fills it later"* is
**factually wrong for reactions**. Registered as a finding; correcting the comment is not C-056a's.

### 6. Two hazards registered, neither blocking C-056a

* **The human abbreviation stays broken.** `eligibility._canonical_organism("H. sapiens")` returns `""`
  while `("E. coli")` returns `"Escherichia coli"`; the human alias set carries no abbreviated binomial. So
  the widening clears `E. coli` and `S. cerevisiae` and **leaves `Homo sapiens` / `H. sapiens` a finding** —
  for the second-most-requested organism in the corpus. Zero process rows carry it today, so nothing
  demotes; it becomes live the moment an extraction stage stamps organisms on reactions. **A0-C3's mandated
  "abbreviation" assertion must be written in this measured form — the naive "all abbreviations match" is
  false and would fail.** `rag/admission.py` and `rag/eligibility.py` remain **import-only**; fixing the
  alias set is not C-056a's.
* **Do not extend `CHECK_ORGANISM` to entity rows.** 8 Arabidopsis protein rows across the corpus are the
  PathBank `Unknown` sentinel (`pathbank_protein_id: 9659`, `identity_status: "placeholder"`) — exactly the
  rows `semantic_production.py:113-121` deliberately excludes. Extending the check there would newly fail
  **7 of 32 legs on a sentinel**: a merge-rule-6 violation. Recorded for C-056b.
