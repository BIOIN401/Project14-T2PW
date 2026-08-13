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
