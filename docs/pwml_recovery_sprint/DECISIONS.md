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
| *(none)* | — | — | — |

**Closed:** O-2 → D-011 · O-3 → D-014 · **O-1 → D-070** (2026-08-27; the question was rejected as posed — the 21 is 16 wrappers + 5 sentinels, and F-141 carries the 24/82 population that was hiding inside it).

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

---

## D-043 — C-050k: every binding is correct, and the missing diagnostic is the violation · 2026-08-17 · LOCKED

C-050k's census fired D-041 §3's escalation: **61 `resolve_entity` consultations through ambiguous alias
keys across 8 legs, 20 of them same-type**, with **4 affected legs exporting `ok=True` today**. A
`pwml-bio-auditor` adjudicated all 20 under `PRODUCT_CONTRACT` §14, reading the **committed frozen payloads**
rather than the census summary. This records the verdict and re-scopes the card.

### 1. All 20 bindings are biologically CORRECT. None is a `product_contract_violation`.

| Class | n | Verdict |
|---|---|---|
| **EntB** | 8 | **Same entity.** Beyond the shared PathWhiz 6224: shared UniProt **P0ADI4**, shared `gene_name entB`, and the candidate's name parenthesises the gene symbol. "Isochorismatase" names one of EntB's two domains, not a different gene product. |
| **EntE** | 7 | **Same record**, with a caveat. Both rows carry PW 6301 / UniProt **P10378** / gene `entE`. But *"enterobactin synthase"* (EC 6.3.2.14) denotes the EntB/EntE/EntF assembly while EntE is component E (EC 2.7.7.58) — `prot_9` is an **EntE record wearing a complex's name**. The identifier is right and the *name* is wrong: the F-043 trap running backwards. The mislabel is a pre-existing payload defect, not a consequence of the alias key. |
| **DHNA** | 2 | **Same entity, and the binding also selects the identifier-bearing row.** `cmp_5`'s chain is internally consistent (PB 40747 → KEGG C03657 → ChEBI:11173); `cmp_9` carries **no identifier at all**, so identifier equality plays no part in the call. The importing source's own title (`PMC8091085`) fixes the abbreviation, and reaction 6 is MenI, whose product *is* 1,4-dihydroxy-2-naphthoate. |
| **DHB** | 2 | **Same entity**, and **the orchestrator's framing was wrong on two points** — see §3. |
| **CoA-SH** | 1 | **Same entity, different identifiers** — PB 282377 is the tetraanion (ChEBI:57287), PB 1099 the neutral form. The converse of the F-043 trap, and *both* identifiers are correct. **This case was in none of the three classes put to the auditor**; 15 + 2 + 2 = 19 and CoA-SH is the twentieth. |

**Mechanism worth keeping:** the EntB/EntE collisions arise *because* the rows are identity-equivalent —
`prot_8`/`prot_9` acquired `entB`/`entE` as synonyms **from the same UniProt record** as `prot_2`/`prot_4`,
and the index then collides name-of-one against synonym-of-the-other. In this corpus same-type alias
ambiguity is **structurally correlated with identity-equivalence**, not independent of it. **That is an
argument for a diagnostic and against a rebinder.**

### 2. What IS the violation: §3 traceability, uniformly across all 20

`ir.py:1534`'s `ambiguous_entity_reference` fires **only in the fall-through branch**. All 20 same-type
cases return early at `:1533` (`PREFERRED_TYPE_EARLY_RETURN` in every census record), so the diagnostic is
structurally unreachable for **exactly the class where `preferred_order` cannot disambiguate**. Confirmed
empirically: `ambiguous_entity_reference_issues: []` on **all 8** affected legs. Its message ("matched
multiple entity **types**") also describes type ambiguity, so it would not name this class even where it
fires.

`PRODUCT_CONTRACT` §3 requires that information be **traceable so that false content can be attributed
empirically**. An arbitrary row-order choice recorded nowhere is an untraceable decision. **Today it is
harmless in all 20 cases; the contract obligation is to record it, not to have been lucky.**

### 3. Two corrections to the orchestrator's own framing of the DHB case

1. **No frozen identity is lost.** Frozen `entities.compounds[7]` carries **no** `pathwhiz_id`, **no**
   `pathbank_compound_id` and **no** `mapped_ids`; `mapping_meta.resolution.status = "ambiguous"` at
   confidence **0.565**. The `40770` I cited appears **only at EP3**, minted after
   `run_prefreeze_resolution`. An export-path-minted identity is not acquired — it is not lost either.
2. **Gold agrees with the conservative binding.** PMC12452463's `forbidden_identifiers` says the paper uses
   DHB for two different molecules and *"the bare token is not a resolvable identity here"*. Binding the
   id-less row is **gold-consistent, not gold-violating**.
   The genuine upstream defect is elsewhere: the name-plausibility gate rejected
   `(2S,3S)-2,3-dihydroxy-2,3-dihydrobenzoate` with `no_shared_meaningful_token` — a false negative, since
   that *is* the compound. That is a **mapping** defect; under §2 a missing identity is **depth, not
   correctness**. Its own card, not this one.

### 4. C-050k proceeds under D-041(a), re-scoped

**D-041's ruling (a) is CONFIRMED and now rests on §3 rather than on caution alone.** Additionally binding:

* **G9 is a CORRECTION, not new capability.** The base emits `ambiguous_entity_reference_issues: []` on all
  8 legs across 61 live consultations; a fixture fails at base and passes at tip. **This is a genuine
  base-SHA behavioural failure and must not be labelled new functionality** — which is what the C-050k
  implementer concluded independently before stopping. Its re-charter instinct was right.
* **Use a NEW, distinctly-coded issue.** Do **not** fold this into `ambiguous_entity_reference`: that code
  means "multiple entity types" and this class is same-type. Folding them destroys the census's own
  distinction.
* **Compute ambiguity over distinct `entity_key`s, before the preferred-type loop**, and emit on every
  ambiguous consultation regardless of which branch returns. **The return value must not change.** Payload
  row order stays authoritative. This also disposes of the **209** same-key false positives.
* **The diagnostic must not assert equivalence from identifier equality** — F-043 forbids exactly that
  inference. Emit the candidate list with the ids the rows already carry, and let adjudication happen where
  it is permitted.
* **Architecture, binding on later cards:** the **exporter emits the ambiguity; the status classifier — not
  the exporter — decides the leg's state.** §5 forbids exporters reinterpreting process-to-entity references
  post-freeze and §8 forbids exporter DB lookups, so every repair path stays outside the freeze.
* **No case here warrants withholding the PWML.** §4 `diagnostic_only` requires that exporting would demand
  inventing biology; no wrong entity and no false identifier reaches any of these PWMLs. **The conditional
  matters though:** if a same-type ambiguous key ever binds two genuinely different entities, §2 and §1 are
  engaged and §7 requires at minimum `review_required`. Instrumenting is what makes that detectable.

### 5. Four findings surfaced by the adjudication, none owned

1. **Enzyme lists appear to be EXPANDED between the freeze and `resolve_entity`.** Committed
   `PMC12096016/research` reaction 1 has **one** enzyme, yet the census records consultations at
   `/1/enzymes/0` **and** `/1`; committed reaction 4 has **three**, and the census records `/4/enzymes/2` and
   `/5`. If entries are added on the export path this is a **§5 post-freeze mutation larger than C-050k** and
   needs its own measurement. Cause unverified from committed artifacts. **Highest-value open thread here.**
2. **A possible §8 violation:** `_resolve_compound_rows` constructs `PathBankDbResolver.from_env()` and
   `_project_db_identity` writes `pathwhiz_id` on the export path. §8 says *"Exporters perform no network or
   database lookups."* Static-inspection **inference, not a run result** — measure before acting.
3. **Two `ok=True` legs contradict their gold.** `PMC12312563/strict` exports `ok=True` against
   `expected_export: partial_only`; `PMC12856317/strict` exports `ok=True` with the same ALAS2 condensation
   twice against a gold saying one reaction cannot constitute a pathway — **and its PWML names
   *Arabidopsis thaliana* in a human ALAS2 pathway.**
4. **The `ir.py:1658` read site** (5 of the 66 consultations) remains unadjudicated and unowned.

**The duplicate rows themselves — `1,4-dihydroxy-2-naphthoic acid` + `DHNA`, `CoA-SH` + `coenzyme A`, both
materialised as separate nodes in shipped PWMLs — are D-036 consolidation territory. D-036's deferral stands
and is NOT reopened here.**

---

## D-044 — C-050k's three baseline moves are ratified, and a sprint-wide measurement hazard is registered · 2026-08-17 · LOCKED

C-050k built to D-043 and stopped for three decisions rather than taking any of them unilaterally, leaving
its golden suite **committed RED** rather than editing the fixture. That is the correct behaviour and all
three are ratified. Its tip is `46df623` (chain `dd5da13` → `b80cffa` → `468fca5` → `46df623`, first parent
`15f36b4`).

### 1. GOLDEN `_leg_digest`: the 8-leg move is RATIFIED

**8 of 32 legs moved; 24 are byte-identical. The 8 are *exactly* the 8 legs the census independently
identified as carrying ambiguous consultations.** That correspondence is the ratification argument: the
delta is not a diffuse drift, it is the precise footprint of the change.

| Leg | old → new |
|---|---|
| PMC12312563/strict | `64038a74` → `69d9da7b` |
| 1306 PMC12452463/research | `dd9a2f5c` → `dbf62298` |
| 1358 PMC12096016/research | `f1f6ff4d` → `20cbe56b` |
| 1754 PMC12096016/research | `33112778` → `c04624aa` |
| 1754 PMC12180156/strict | `e28efcf1` → `1427c040` |
| 1754 PMC12452463/research | `a75cb748` → `1b503ae4` |
| 1754 PMC12452463/strict | `5e40a7ca` → `219fdbfe` |
| 1754 PMC12856317/strict | `32ab0313` → `a6ca91c5` |

Bounded by measurement, satisfying **D-041 §2 limit 3**: the **IR digest did not move**, `errors` is empty
and `ok is True` on both sides, and the delta is **exactly two warnings per affected leg**. Measured under
pytest, never by a direct-import harness (**F-047**). **Authorised: update `GOLDEN` to the eight new digests
and turn the suite green.** The R3 control re-pin `476e41da…` → `026c8a8e…` is ratified on the same basis,
and asserting the warning fields **before** the digest is exactly right — it stops a second drift riding in
behind the first.

### 2. Chunk D partition `179 → 187` is RATIFIED, effective on the merged state only

`core 152 → 160`, **TOTAL `179 → 187`**. `collect` proves **`SETS_EQUAL=True, missing=0, extra=0,
overlap=0`** — only the derived integers move, and the `+8` is exactly the D-core arms the charter *required*
be added to `tests/test_pwml_ir.py`, which had **zero** coverage of `resolve_entity` / `ambiguous` /
`synonyms`. This is a deliberate **merge rule 4** baseline move with an exact documented delta, not a drift.

**`TEST_MATRIX.md`'s counts must be updated to `core 160 + s8 4 + qb 23 = 187` as part of the C-050k merge,
never before it.** Until that merge lands, **179 remains correct** and cards in flight (C-056a, C-052) must
keep reporting it. A card reporting 187 on a tree without C-050k is measuring the wrong thing.

**Not running the full gate to prove an integer it already knew was correct**, and letting `run` block on
`COMPONENTS` instead, was the right call — it burned no artifacts to re-derive a known number.

### 3. D-025 ceiling 1: `950 → 1,150`, RATIFIED

Measured **1079**: probe 616, `ir.py` 147, `test_pwml_ir.py` 238, refusal suite 78. **The entire overage is
the probe**, which my table budgeted at 250 against a real 616 — the same class of error as C-056a's, and
mine again: a four-part census over 32 legs with live-DB reproduction of EP3 does not fit in 250 lines.
Nothing was cut to fit (**REV-051a**). Ceilings 2 and 3 stand and are comfortable: **40/90**, **8519/12,000**.

### 4. F-051 — a worktree without `.env` silently masks every DB-dependent test, in BOTH directions

**Registered as a sprint-wide measurement hazard of the same class as F-042 and F-003. Unowned.**

`ensure_dotenv_loaded` reads `PROJECT_ROOT/.env`, and **a git worktree created from a SHA never has `.env`,
because it is gitignored.** So the PathBank DB is unreachable at base and reachable in a working tree, and
the *same* selection runs **5.63 s vs 94.86 s**. **Any base-vs-tip comparison performed in a fresh worktree
can hide a real regression or manufacture a false one.**

C-050k caught this the right way: its fifteen-file gate was **9 failed / 386 passed** at tip against
**2 / 393** at base, and it attributed the difference by **swapping only `src/t2pw/pwml/ir.py` to its base
version inside its own tree** — reproducing the same 6 failures with its own code removed. 2 are the known
base-red `[only_unrelated_reactions_survive]`, 1 is the ratified GOLDEN move, and **6 are not the card's**.
D-core's single red (`test_cli_export_emits_the_canonical_organism…`, itself one of C-045b's two added
tests) is the same effect.

**That swap technique is the standing remedy until this finding is owned: hold the tree constant and change
only the file under test.** Copying `.env` into a worktree is **not** the fix — C-050i's probe already
refused that, and it would make the two sides differ in a second uncontrolled way. **Do not attribute a
DB-dependent failure to a card without a same-tree swap.**

### 5. Still owed by C-050k

**SMOKE 460 and Chunk E were never run** — `C:\t\heavylock` was held by C-056a from 15:14:13 to 15:49:55
across five acquisition attempts. The card stopped both background waiters rather than leave one armed to
seize a lock with nobody to release it, which is right. **These two gates, plus full Chunk D on the updated
partition, remain outstanding before C-050k can be reviewed for merge.**

---

## D-045 — C-050k ceiling 3 raised to 15,000; the census is the evidence base of D-043 and is not cuttable · 2026-08-18 · LOCKED

C-050k reported **C3 = 14,156 / 12,000**, and in the same breath **corrected its own previous figure** — the
`9556` it reported earlier was measured *before* it committed the Chunk D evidence, so it understated the
truth. Self-correcting an under-report that nobody had challenged is the behaviour this sprint runs on.

**Ratified: ceiling 3 `12,000 → 15,000`.** Projected final is ~14,640 including the four remaining runs.
C1 stands at 1,200 (measured 1171) and C2 at 90 (measured 84, projected 88).

**Why the overage is structural and not padding:**

| Item | Lines | Status |
|---|---|---|
| 32-leg four-part census | **5,348** | charter budgeted **~500** — a 10× under-estimate by the orchestrator |
| full Chunk D's 32 auto-allocated reports | **3,154** | mandated by the gate the charter itself required |
| remaining `g11/C-050k` bounded reports | ~4,023 | one report per required job |
| probe output beyond the census | ~917 | differential + G9 proof |
| pin verdicts | 714 | one per measured run |

**The census is not compressible without destroying the evidence base of a LOCKED decision.** It records
every consultation with its full candidate list, and that is precisely what made the **8-leg golden
correspondence checkable** and what the `pwml-bio-auditor` read to adjudicate all 20 bindings in **D-043**.
Cutting it to fit a number I set would retroactively unfound D-043.

**This is the third ceiling I have under-set today** — C-056a's C1 (budgeted zero for `evidence/*.py` while
the corrected F-050 command counts it), C-050k's C1 (probe budgeted 250 against a real 616), and now C-050k's
C3. The pattern is consistent and one-directional: **evidence-generation is systematically under-budgeted,
and the corrected F-050 command counts it.** Every future ceiling table must derive an explicit line for
probe source, probe output JSON, and per-gate auto-allocated reports. **REV-051a governs throughout: ratify
the ceiling, never let the card mutilate its work to fit it.**

**Unchanged and still binding:** no second full Chunk D — node15 attribution runs **node15 alone** (1–2
artifacts). If a second full Chunk D ever becomes necessary, that is a fresh ceiling decision, not an
overage.

---

## D-046 — C-052 ceiling 1 raised to 950; and F-051 has a mirror image that is worse · 2026-08-18 · LOCKED

### 1. Ceiling 1 `900 → 950`. Ratified.

C-052 measures **921**. It stopped before spending mutex time rather than run three heavy gates on a card
whose scope I might change — the right order of operations.

**The overage bought a G9 correction, which is the opposite of padding.** At `36d8b68` the card was at 894.
Splitting the A0-C8 guard cost +40/−13 = 53 lines, because the original single test asserted **both** the
pre-existing `build_sbml`-path identity **and** the two new result keys — so it failed at base, and would
have presented **a guard on unchanged behaviour as though it carried a base failure**. That is precisely the
mislabelling G9 calls an automatic reject. Paying 21 lines to avoid it is a bargain.

The split is now visible in the base proof and lands exactly on the naming: at base **11 failed / 2 passed**,
at tip **13 passed** — every `test_new_acceptance_*` fails at base, both `test_a0c8_guard_*` pass at base.
**No fabricated base failure.** Ceilings 2 (21/90) and 3 (1,375/20,000) are comfortable and unchanged.

**This is the fourth ceiling I have under-set today.** D-045's rule stands and is reaffirmed.

### 2. The `pwml` trap was discharged BEHAVIOURALLY, and that is the standard

C-052 did not prove "no added key contains `pwml`" with a string check. It called
**`driver._find_pwml_result` on the real EP1 result with an empty `pwml_export_result`** and showed it
returns `("", {})`. The trap in D-040 §2(a) is about what the driver *does*, so the proof must exercise the
driver. **Prefer this shape over a grep in every future card.**

Likewise the `"seam"` label is written **after** the dict spread — `{**report, "seam": …}` — and tested with
a deliberately shadowing report key, so a colliding report key cannot silently win. Added EP1 keys are
`prefreeze_resolution_report`, `prefreeze_review_required`, `canonical_json_path_name`,
`sbml_input_path_name`; EP3 adds `prefreeze_resolution_report_path`. `freeze_canonical_payload`,
`driver.py`, `writer.py`: **zero lines**.

### 3. F-051's mirror image: five tests assert that the database is NOT configured

Measured by C-052 on the **base** tree, with `.env` as the only variable:

| Base tree | Result |
|---|---|
| **without** `.env` | **0 failed** |
| **with** `.env` | **5 failed** / 95 passed |

The five — 4 in `tests/test_prefreeze_third_export_seam.py`, 1 in
`tests/test_prefreeze_species_resolution.py` — **assert on a DB that is not configured. A live DB falsifies
their assumption, not the code.**

**This inverts the usual reading of F-051 and is the more dangerous direction.** F-051 as first registered
says a worktree without `.env` can *hide* a failure. This says the reverse is also true: **a correctly
configured developer machine makes green tests red**, and a card running its gates in the primary checkout
will see five failures it did not cause. C-052 controlled for it correctly — `.env` copied into the base
worktree so **both sides were equal** — which is the point: the hazard is the two sides *differing*, not the
presence or absence of `.env` as such. That is compatible with C-050i's refusal to copy `.env` as a *fix*;
copying it as a *control*, on both sides, is sound.

**Registered under F-051. Do not "fix" these five tests as part of any current card** — the correct owner is
whoever takes F-051, and the fix is a decision about whether the suite may assume an unconfigured DB at all.

### 4. A0-C8's measured limitation is accepted as reported

The clause *"including the path supplied to downstream SBML generation"* is **unexercised on all 39 legs** —
`build_sbml` is guarded on `build_legacy_sbml` and the batch driver never binds that widget. Discharged at
unit level, and **the card did not bind the driver to the legacy button**; instead it added a test that fails
if anyone ever does. That is the correct disposition of a limitation no card owns: pin it so it cannot be
crossed silently.

**Authorized to proceed to `qb` · full Chunk D (`179` — its tree does not carry C-050k) · SMOKE 460**, then
independent non-author review. The scratch base worktree at `C:\t\c052base` **stays** until after review.

---

## D-047 — `qb` node15 is pre-existing-red under a live database; F-052 registered; it blocks no card · 2026-08-18 · LOCKED

C-056a's `qb` returned **red**: `jobs=23 executed=22/23 additions=0 failed=['node15']`. Outer job 141.29 s,
125 descendants observed, **125 terminated, 0 survivors**, cleanup success — clean infrastructure, so this is
a **true test result**, not an infrastructure failure. It was reported red and **not retried**.

`node15` = `tests/test_streamlit_quarantine_boundary.py::test_research_mode_keeps_the_unmapped_candidate_and_does_not_block`.

### 1. It is pre-existing at base. Proven by a three-condition control.

| Condition | node15 |
|---|---|
| base `a662c3f`, **no** DB | **passes** |
| base `a662c3f`, **live** DB | **FAILS** — identical assertion, identical diff |
| C-056a tip, **live** DB | **FAILS** — identical assertion, identical diff |

**The failure tracks database availability, not the diff.** C-056a's *first* base run passed and it
**caught that the control was invalid itself** — `C:/t/c056a-base` had no `.env`, so the DB was absent — and
re-ran it properly. That self-catch is the whole value of F-051 being written down: the invalid control was
the one that would have produced the comfortable answer.

Payload mutation is excluded as a mechanism by direct measurement: `evaluate_production_semantics` does
**not** mutate the payload it receives, and `quarantine_and_close` deepcopies before the seam regardless.

### 2. F-052 — a standing tension between D-015 and node15's invariant, visible only with a reachable DB

Node15 asserts the pre-freeze invariant *"processes moved pre-freeze"*. What actually differs between the
compared artifacts is **compound-name canonicalization** performed by **D-015 pre-freeze compound
resolution**, rewriting names out of the PathBank database:

* `L-cysteine` → `L-Cysteine`
* `gamma-glutamylcysteine` → `γ-Glutamylcysteine`
* `OPC-8:0` → `8-[(1R,2R)-3-oxo-2-{(Z)-pent-2-enyl}cyclopentyl]octanoate`

That stage runs **before** the quarantine seam, and the seam cannot reach either compared artifact. So
D-015 doing its job is what breaks node15's invariant — **a real product-level tension, not a test bug and
not a card's regression.** It is invisible on any machine without a configured database, which is why no
card has ever been charged with it.

**Registered as F-052, UNOWNED.** Neither C-056a nor C-050k may fix it:
`tests/test_streamlit_quarantine_boundary.py` is **hotspot 9** (no test function may be added, removed,
renamed or reordered — the gate addresses nodes positionally), and `pwml/compound_resolution.py` is outside
both boundaries. **Do not let a later card "fix" node15 by weakening the invariant** — the invariant may well
be right and D-015's scope may be what needs stating.

### 3. Consequence for merges

**A red node15 does not block C-056a or C-050k**, provided each carries its own control showing the failure
is present at its base under the same database condition. This is merge rule 4's *"failures measured red at
base"*, not a regression. The `qb` result is otherwise clean: 22 of 23 executed, **additions=0**.

**The accepted pre-existing-red set therefore grows from four to five**, and the fifth is conditional:
`test_strict_failure_replay.py` ×2 and `test_batch_preflight.py` ×2 unconditionally, plus **`qb` node15
whenever the database is reachable**. Any card reporting `qb` green on a machine without `.env` has measured
nothing.

### 4. Three independent cards have now hit F-051

C-050k (6 of 9 focused failures), C-052 (5 base failures that appear **only with** `.env`), and now C-056a
(an invalid base control it caught itself). **F-051 is no longer a hypothesis about one card's tooling; it
is the dominant measurement hazard of this pack, and it runs in both directions.** Every base-vs-tip claim
in this sprint that did not control for database reachability should be treated as unmeasured until it does.

---

## D-048 — C-050k ceiling 2 raised to 100; card complete and routed for review · 2026-08-18 · LOCKED

**Ceiling 2 `90 → 100`. Ratified.** C-050k measures **92**: 63 G11 reports + 25 pin verdicts + 4 probe JSONs.
**39 of the 63 reports are full Chunk D's auto-allocation**, and the four final runs (node15 ×2, SMOKE,
Chunk E) took it 88 → 92 because each carries a pin verdict *as well as* a report — a coupling no ceiling
table in this sprint has ever accounted for.

**This is the fifth ceiling I have under-set today** and the fourth on this one card's family. D-045's rule
is reaffirmed and extended: a ceiling-2 derivation must count **one report *and* one pin verdict per measured
run**, not one artifact per run. Every one of the 92 is a genuine measurement, including the two committed
**failing** runs and the node15 pair that closed the last open question. **Nothing was cut** (REV-051a).

C1 **1171/1200** ✓ · C3 **14,666/15,000** ✓.

### The card is complete at `ce5761a`

Gates: **SMOKE 460 exactly** · **Chunk E 174** · **full Chunk D 185/187** (core 159/160, 26 of 27 AppTest) ·
golden suite **14** after the D-044 re-baseline · **G11 exit 0, 63 artifacts, 0 non-compliant, survivors 0
and `cleanup_success` true on all 63**.

Both Chunk D failures are attributed away by same-tree swap and neither is the card's:
`test_cli_export_emits_the_canonical_organism…` (`g11/C-050k/20`) and **node15** (`60` tip / `61` base
`ir.py` swapped in) — identical assertion at `test_streamlit_quarantine_boundary.py:1109`, with the swapped
tree verified to contain **zero** occurrences of `ambiguous_entity_row_reference`.

### Two things the writer did that set a standard

1. **It confirmed D-047 from a second base tree without over-claiming.** Its control varied **only the diff**
   under a live DB throughout, so it establishes **diff-independence** — and it said explicitly that it does
   **not** re-derive **DB-dependence**, which is C-056a's three-condition control, and that F-052's
   canonicalization root cause is C-056a's finding rather than its own. **Stating the exact reach of your own
   evidence, and declining credit for a neighbouring result, is the behaviour to imitate.**
2. **It retracted an unobserved assertion unprompted.** It had stated "C-056a took the lock at 19:10:59" as
   fact, had not observed it, noticed, and withdrew it. The measured sequence is a **normal acquisition of a
   free lock** at 19:15:00, held in three short windows with a release between each so C-052 could cut in.
   **That is not a stale-hold clearance and is not charged as one.**

**Routed to an independent non-author reviewer at `ce5761a`.** Merge requires an exact bare `APPROVE`.

---

## D-049 — C-050k's three ceilings raised together; the correction that breached them was one I ordered · 2026-08-18 · LOCKED

**Ratified: C1 `1,200 → 1,250` · C2 `100 → 105` · C3 `15,000 → 15,500`.** C-050k measures 1194 / 99 / 15,145.

**All three are raised together, deliberately.** The card reported that every ceiling now sits within ~1% of
its limit, so **any correction round 2 of any size would breach at least one** — and it flagged that before
starting a round rather than discovering it mid-flight. Leaving three ceilings at 99% while authorising two
correction rounds is a trap of my own construction. Raised now so a reviewer-ordered fix is never blocked by
a number I set.

**The +145 that breached C3 is entirely artifacts I ordered.** The reviewer could not verify the integers
`SMOKE 460` and `Chunk E 174` — `bounded_run` records exit code, not stdout — so I required verbatim-count
artifacts on C-056a's precedent. The three new bounded reports, their three pin verdicts and
`c050k_gate_counts.json` *are* that requirement. **Charging the card for satisfying it would be incoherent.**

**This is the sixth ceiling under-set today** (C-056a C1; C-050k C1, C3, C2, and now all three; C-052 C1).
D-045's and D-048's rules stand and are extended once more: **a ceiling table must budget the cost of
answering a reviewer, not merely the cost of building the card.**

### What the correction round established

The blocker was a **false G9 header**: `tests/test_pwml_ir.py:833` claimed *every* arm fails at base; five
do, three do not. The three assert an **absence** (`_row_issues(report) == []`), trivially true where the
code does not exist. **The card's own committed evidence had already contradicted the sentence** —
`evidence/c050k_g9_base.json` records `claim_3_silent_on_slot_duplication: true` measured at base — and the
writer said so in the header rather than quietly dropping the claim. **D-046 §1 governs: presenting a guard
on unchanged behaviour as though it carried a base failure is the mislabelling G9 rejects outright.**

The fix is documentation-only and mechanically provable: **zero non-comment changed lines and zero touched
`def` lines** in `tests/test_pwml_ir.py`, and `git diff b80cffa HEAD -- src` **empty** — production
byte-identical to the reviewed code. The three guards were **relabelled, not deleted**; they are what makes
the guard non-vacuous against the 209-key false positive.

### An accurately labelled transcription is acceptable evidence; a number dressed as measured is not

Chunk D's `185/187` and the partition proof (`SETS_EQUAL=True, missing=0, extra=0, overlap=0`) went to
`chunk_d_gate.py` **stdout, which `bounded_run` does not retain**, and re-deriving them costs a collect plus
a full run that ceiling 2 forbids. The card labelled them **transcription, not measured**, *inside the
artifact*, and recorded what is checkable: all 36 Chunk D jobs with per-job exit code, survivors and cleanup
— 34 exit 0, 2 nonzero, zero survivors throughout.

**Ruled acceptable.** The reviewer was invited to disagree and to say so plainly rather than approve around
it. **Registered as an instrument gap for a later card: `bounded_run` retains exit codes but not stdout, so
no pytest summary line is recoverable from a G11 report.** C-056a's `c056a_gate_counts.json` and C-050k's
`c050k_gate_counts.json` are the workaround — **every future card must capture verbatim summary lines into a
committed artifact, because `SMOKE exactly 460` is merge gate 10 and an exit code cannot certify it.**

---

## D-050 — F-046's discriminator is measurably wrong; C-050j is re-scoped onto the path no record names · 2026-08-18 · LOCKED

C-050j has been undispatchable because **its central design input is false**. This corrects it and makes the
card buildable. Measured in the primary checkout; **re-measure every line number after today's three merges.**

### 1. F-046's proposed discriminator would fire NEVER. Struck.

F-046 says the species marker is stamped on *"every row that participated"* and proposes **marker-presence**
as the discriminator. Live, `prefreeze_resolution.py:1215-1245` stamps **every named species row that reaches
the stage** — the only `continue` is for non-dicts and unnamed rows (`:1216-1217`); leaders at `:1230`, all
others at `:1245`.

So two genuinely different organisms that already `_norm`-collide are **leader + follower, both stamped,
neither renamed** — and marker-presence **clears them to collapse**, which is the exact case F-046 says must
refuse. **Built as written the guard fires never in the species bucket.**

**The correct positive marker is `marker["followed_leader"]`** (`prefreeze_resolution.py:1242`), written
**only** when the leader's rename moved the group's `_norm` (`:1236`).

> ### ⚠ AMENDMENT 2026-08-20 — this nomination does NOT transfer to the create-defaults path
>
> **Scope this clause to the pre-freeze species residual it was written about. It cannot serve as C-050j's
> discriminator, and a card built on it would have shipped an unreachable no-op node.**
>
> Established by C-050j and **confirmed independently by REV-050j**, read against
> `prefreeze_resolution.py`:
>
> * `:1211-1213` — `leaders.setdefault(_norm(row.get("name")), index)`: leaders are keyed by `_norm`.
> * `:1218` — a row finds its leader by its **own** `_norm`, so leader and follower **necessarily share one
>   pre-rename key**.
> * `:1235`/`:1242` — the value stored is the follower's own pre-rename name, whose `_norm` **is** that same
>   group key.
>
> So `marker["followed_leader"]` is **structurally confined to a single pre-rename `_norm` group** and
> carries no information about a merge *across* two of them — which is exactly the question C-050j's
> component call site had to answer.
>
> **A second, independent reason it cannot serve:** a renamed **leader with no followers** carries no
> `followed_leader` at all (`:1224-1231`), so marker-presence would **miss a single-row create-defaults
> rename entirely** — the commonest shape of the very case.
>
> **What replaced it, and it satisfies F-043 more strictly.** C-050j's discriminator is a structural
> pre-group identity test: build `post-rename _norm -> the distinct pre-rename _norm keys it merged` over
> `entities[source_key]`, and refuse iff a post-rename key was reached from **more than one** pre-rename
> key. It reads **only `name`**, on both sides of the rename — no `taxonomy_id`, no `pathbank_species_id`,
> no accession. It also **subsumes the marker**: a group the pre-freeze converger built arrives under one
> pre-rename key and is cleared by construction, marker or no marker. Pinned parametrized over
> marker-present and marker-absent at `tests/test_pwml_ir_duplicate_row_refusal.py:759`, so the verdict is
> proved not to depend on it.
>
> **D-050's conclusion, its boundary and its census-first stop condition are all unchanged.** Only this one
> nominated mechanism is corrected — the fourth time this sprint a record's cited mechanism proved false
> while its conclusion survived. Merged in `cbeaa84`; F-046's original proposal remains struck.

**F-043 still binds and is not weakened by this correction:** the discriminator must **not** be built from
identifier equality. `PG`, `PG phosphate` and `(PGP)` all carry PathBank 193, which is UDP-glucose and wrong
for all three. The durable species-canonicalization marker is the discriminator; a shared identifier is
evidence, never proof.

### 2. F-046 overstates the residual on one axis and misses the real one entirely

**Overstated:** most of the "two different organisms whose names `_norm`-collide" shape is **already refused
pre-freeze** by `_reject_ambiguous_species_renames` (`prefreeze_resolution.py:1086-1177`) — distinct sources
→ one target (`:1106-1116`), and rename onto an occupied-and-kept name (`:1142-1158`). Its own comment
(`:1130-1136`) names `build_pwml_ir`'s dedupe as the harm it prevents. What is genuinely open is narrower: a
**payload-authored collision with no rename at all** (the guard is called only `if rename_map`, `:1354`),
plus the three buckets with **no pre-freeze canonicalizer at all** (`PREFREEZE_CANONICALIZERS` `:1410-1413`
is compounds + species only).

**Missed, and it is the card's strongest case: the exporter manufactures the collision itself, post-freeze.**
`_apply_create_defaults` **renames a component row inside `build_pwml_ir`** — `ir.py:344`
`record["name"] = default_name` — applied at `ir.py:1085-1096`, **immediately before** the unguarded
component dedupe at `:1097`. The pre-freeze ladder never applies that table as a rename (`ir.py:934-935`
only *recognises* an already-canonical default and returns early). A payload carrying both
`Narcissus sp. aff. pseudonarcissus` and `Narcissus aff. pseudonarcissus MK-2014` (`ir.py:230-238`) clears
every pre-freeze guard, is then **renamed after the hash**, and one row is dropped with a warning.

**No record contains this path, and C-050i's zero-of-32 census structurally could not have seen it** — that
census replayed `_norm(_canonical(...))` over committed `final_mapped.json`, i.e. **before**
`_apply_create_defaults` runs.

### 3. Exposure is UNMEASURED and the card measures it first

Payload-authored: **zero of 32** (C-050i's census plus REV-050i's independent 32×9 re-census).
**Create-defaults-manufactured: unmeasured.** The card's first act is a bounded read-only census of that
path, reproducing EP3 exactly. **If it is non-zero, the card's G9 classification changes from new capability
to a correction with a real base-failing proof, and it stops for a fresh ruling — exactly as C-050k did.**
Deliberate convergences that must keep collapsing have never been enumerated; do not assume the set is empty.

### 4. Boundary and posture

**In scope:** `ir.py :: _dedupe_named_rows` component branch and the component call site `:1097-1102`
(pass a discriminator through) · `tests/test_pwml_ir_duplicate_row_refusal.py` · `tests/test_pwml_ir.py` ·
a probe under `evidence/`.
**Out:** `prefreeze_resolution.py` (**read-only**) · `tests/test_prefreeze_species_resolution.py`
(**zero lines** — C-045/D-016's accepted criteria) · the **entity** call site and `resolve_entity`
(**C-050k's, merged today**) · the `component_by_name` LAST-wins residual (`ir.py:1140`, the component twin
of F-048 — **register, do not fix**) · moving the create-defaults rename upstream · any consolidation.

**D-035 unamended and D-036's deferral of the consolidation engine intact — do not propose reopening it.**
The duplicate rows themselves are D-036 territory.

**C-050j serializes against nothing now** — C-050k merged, and the two diffs do not overlap.

---

## D-051 — C-030a ceiling 1 raised to 800; and its G9 argument is accepted over the charter's guess · 2026-08-18 · LOCKED

**Ceiling 1 `700 → 800`. Ratified.** C-030a measures **748** — 530 test source, 218 probe. My table budgeted
700 for a card whose entire deliverable *is* a mutant matrix plus a probe. Ceilings 2 (**23**/60) and 3
(**1,608**/10,000) are comfortable and unchanged.

**Seventh ceiling under-set today; seventh ratified.** The card also spent a prose-tightening pass going
759 → 748 chasing my bad number — work spent on my error, not on the card. It was told to restore anything
that pass cost. **A ceiling must never buy worse evidence.**

### 1. The G9 classification is the CARD'S, not the charter's, and the card is right

The charter guessed "genuinely new capability". **The implementer disagreed and argued it, correctly: these
are guards on pre-existing observable behaviour that PASS at the base SHA.** There is no production change,
so there is no behaviour change to prove and **no base failure is claimed or fabricated**.

**Certified by measurement, not by symbol absence** — the failure mode G9 names: `src_files_changed_base_to_head: []`,
`app_source_identical: true`, and the **AST hashes of both `run_post_pipeline_sbml_artifacts` and
`freeze_canonical_payload` identical at base and tip.** This matches C-052's ratified `test_a0c8_guard_*`
precedent for the sibling requirement. **What is new is the discharge of a requirement, not a capability;
calling it new functionality would have misdescribed it.**

### 2. The seam-binding mutant converts F-041 from a quotation into a measurement

Each mutant rebuilds `run_post_pipeline_sbml_artifacts` from its own AST with `deepcopy` at **exactly one**
site and asserts the **exact clause set** that goes red. `pytest.raises` was **rejected** — it would pass on
any unrelated `AssertionError` and the matrix would prove nothing. Unmutated tip: `[]`.

**The decisive row: copy at the seam binding `canonical_export_payload = _freeze["payload"]` turns exactly
ONE clause red.** All four keys still agree *with each other* — on a lookalike the seam never hashed, gated
or serialized — so **every "the four keys agree" check is blind to it, and A0-C7's own tautological form
still reads `True` under it.**

That is F-008/F-041's *"a share→copy mutant is indistinguishable on all nine observables across all 39 legs"*
re-established as a **live measurement on this tip**, which is exactly what A0-C7 has lacked since it was
orphaned. **A0-C7's tautological assertion is insufficient — now demonstrated rather than asserted.**

### 3. Two judgements accepted as the implementer made them

* **Hotspot 11 is not entered.** A0-C7 says the tautological assertion *"is not sufficient"*, **not** that it
  must be removed, and proving the real property elsewhere discharges it. The byte-pinned golden fixture and
  the projection are **untouched** — strictly lower risk, and no fixture delta is needed.
* **Calling `freeze_canonical_payload` and spying on its return** is the technique C-052's merged
  `test_a0c8_guard_…` uses under the same ZERO-lines constraint. The function has **zero changed lines**; the
  discharge lives in the orchestrator, not the seam.

### 4. An honest red, recorded to the card's credit

Run 01 was `1 failed, 9 passed` — **the implementer's predicted clause set was wrong, not the code.** On a
refusal the canonical key holds the seam's empty dict, so `canonical_object_is_not_the_pre_seam_object` does
not fire. **The test was corrected to the measured set and no production line was touched to turn it green.**
That is the correct direction of correction, and it is kept in the gate-counts artifact rather than tidied away.

### 5. The discriminator, re-measured — inherit these, not the older numbers

**`streamlit_app.py:3762`** — `"final_mapped": canonical_export_payload or final_export_payload`.
F-041 recorded `:3748` (**+14 stale**); this orchestrator measured `:3746` before C-052 merged (**+16
stale**). All six sites at this tip: seam binding `:3648` · `CANONICAL_PAYLOAD_KEY` `:3710` ·
`final_mapped_quarantined` `:3720` · `final_mapped` `:3762` · `final_mapped_enriched` `:3764` ·
`final_export_input` `:3765`. **Every card in this sprint that inherited a line number was wrong.**

### 6. Attribution held the tree constant (F-051)

Named focused set over 11 ungated files: **5 failed / 191 passed** at tip; with **only the new file removed,
same tree and same `.env`**: 5 failed / 181 passed — **+10 passed, +0 failed**. The five are exactly D-046
§3's live-DB set. **SMOKE `460 passed in 35.18s`.** G11 exit 0, 11 artifacts, survivors 0 on all 11.

---

## D-052 — C-056b's two widenings are ratified; the evaluability gap becomes F-053 with an owner · 2026-08-18 · LOCKED

C-056b is merged (`69928eb`, reviewed tip `8eee549`). Its reviewer asked that two disclosed widenings be
ratified explicitly rather than absorbed, and that the card's named gap be given an owner. Both are done here.

### 1. Two boundary widenings, RATIFIED

**(a) `acceptance.py` module-level import plus three read-only `ModeResult` properties (`:199-256`).** Inside a
file the card owns, serving the owned function, **adding no `to_dict` key and changing no existing symbol**.
`strict_eligible` (`:183-197`) is the in-file precedent. Ratified.

**(b) The `GOLDEN` re-baseline in `tests/test_batch_driver_seam_golden.py` (hotspot 10, owner C-053, merged).**
This is **the unavoidable mechanism of a granted change** — any key added to `ReleaseStatus.to_dict()` moves
those digests, and D-039 §5 contemplated the bump moving pins. It named two; this is an **unmeasured third**
the card found, **reported red, and moved under merge rule 4** with a derived delta.

Hotspot 10's two hard guards were verified **by the reviewer, not from the card's report**: `_observable` is
byte-identical base↔tip (`sha256:438cb7f1…` both sides) and the `def` list is identical in name and order.
Ratified.

**Standing rule:** a widening that is the *mechanism* of an already-granted change is ratifiable, but it must
be **disclosed and ratified in the record** — never absorbed silently. That is the D-044 pattern.

### 2. F-053 — evaluability does not travel beside the verdict. Owner: **C-056c** (new, BLOCKED)

**The gap.** A serialized `semantic_evaluation: "passed"` is **indistinguishable from a four-of-four pass** to
anyone reading the manifest. Measured over the 32 committed payload legs: under the seam's own derivation
**every leg had exactly ONE evaluable gating check and 25 answered `passed`**; under a request-carrying
derivation 31 reached three; **none ever reached four**, because `CHECK_RAG_REINTRODUCTION` is unevaluable at
this seam.

**Why it is not C-056b's to fix.** It is **inherited, not created** — C-056a (`93594aa`) already wrote that
value, and C-056b's diff neither creates nor widens the ambiguity. Fixing it needs a fourth return value from
`semantic_verdict` (`release_status.py:339-356`, which computes `evaluable` and **discards** it) plus a new
`classify_release_status` argument threaded through `strict_quarantine.py:2132` — and **D-042 §4 already ruled
a new parameter on that seam is NOT granted and needs its own card.**

**Why it is currently harmless, and exactly when it stops being harmless.** Nothing consumes it
affirmatively: the one affirmative accessor `ReleaseStatus.semantic_confirmed` (`release_status.py:264`) has
**zero `src/` consumers**, and 0 of 143 committed manifest rows carry a `release_status`, so **no historical
figure can move.** The ambiguity reaches a reader's eyes, never a rate.

**BINDING PROHIBITION, and the reason F-053 exists.** **No card may be chartered to read
`semantic_evaluation == "passed"` affirmatively — or to build any denominator, numerator or rate on it — until
F-053 is discharged.** The moment a reader does, the ambiguity becomes an **inflation**, and
`PRODUCT_CONTRACT` §11's distinction between `passed`, `failed` and `not_evaluated` is only honestly
*serialized* once evaluability travels beside the verdict. C-056b's subtractive-only design is what holds the
line today; it is a discipline, not a guarantee, and F-053 is the guarantee.

**C-056c's boundary, when chartered:** carry the evaluated/applicable set alongside the verdict, and only
then permit an affirmative reader. It inherits D-042 §4's stop condition: **an `admission` parameter on
`quarantine_and_close` remains ungranted** — C-057 also touches that seam and they serialize.

### 3. Three low findings recorded, none blocking, none C-056b's to fix now

* **`acceptance.py:200`** — `runtime_semantic_evaluation` **is** a public accessor returning the raw
  three-valued string, so the docstrings at `:234` and `:739` claiming a runtime `passed` *"has no accessor"*
  are **looser than the code**; `== "passed"` is one comparison away. Substantively fine — there is no
  affirmative *predicate* and nothing consumes it — but **a later reader could quote the comment as a
  guarantee it does not give.** Correct the wording in the next card that touches the file.
* **`acceptance.py:215`** — `runtime_semantic_failed_checks` has **zero `src/` consumers**. Harmless: the
  names travel in `to_dict()["release_status"]`, but neither `bench/render.py:179` nor `batch/report.py:860`
  surfaces them, so *"which checks failed"* is JSON-only today.
* **`evidence/c056b_tip_reachability.json`** carries `"task": "C-056a"` — cosmetic mislabel in a C-056b
  artifact.

### 4. A reviewer claim corrected so it does not propagate

C-030a's reviewer reported that **`g11_evidence.py` "does not exist anywhere in the repo."** That is **false**:
`git ls-files docs/pwml_recovery_sprint/evidence/g11/g11_evidence.py` returns it, it is present on disk in the
primary checkout **and in that reviewer's own worktree**, and it is not gitignored. Its `git ls-files | grep g11`
was mis-scoped. The approval stands on measurements the reviewer made itself; **the tooling claim is struck so
no later session inherits it.**

---

## D-053 — T-100 is RUN and NOT DISCHARGED; strict benchmark figures still may NOT be quoted · 2026-08-18 · LOCKED

T-100's code-card prerequisites (**C-053**, **C-056b**) are merged, so the milestone was run at integration
tip `3a83b15` using the **prescribed** tool — `scripts/bench_acceptance.py`, which by its own docstring
*"touches no network or LLM"*. **No new benchmark procedure was invented.**

### 1. What ran, and what it returned

| Job | Result | Evidence |
|---|---|---|
| `--validate-gold` | **exit 0** — gold set `2026-08-01.1`, 10 cases (9 mechanistic, 1 negative control), 4 strict-exportable | `evidence/g11/T-100/01-validate-gold.json` |
| `--run-dir runs_verify/2026-08-04_1754` | **exit 1** | `evidence/g11/T-100/02-score-verify-1754.json`, `evidence/t100_score_verify_1754.json` |

```
  1. [FAIL] zero known false real identifiers        observed: 13
  2. [FAIL] zero unsupported retained reactions      observed: 2
  3. [PASS] zero referential-integrity violations    observed: 0
  4. [FAIL] meaningful requested-pathway coverage    0/4 = 0%
  5. [FAIL] strict PWML pass rate                    0/2 = 0%
  status: PARTIAL -- not a quotable benchmark result   (6/10 papers, 12/20 legs)
```

### 2. **T-100 is NOT DISCHARGED, and strict figures may NOT be quoted.** Two independent reasons.

**(a) The tool itself refuses.** `status: PARTIAL -- not a quotable benchmark result`. Four gold cases have
**no attempted leg** (`PMC13231680`, `PMC12657337`, `PMC12421875`, `PMC12312563`). *"An unattempted or
unscorable paper is missing coverage, NOT a pipeline failure."* **Weakening that judgement to quote a number
is exactly what this decision forbids.**

**(b) Strict `0/2` on historical artifacts is BY DESIGN, not a regression.** The strict gate requires
`strict_acceptance_eligible` **affirmatively**, which requires a `release_status` record on the manifest row.
**C-056b measured 0 of 143 committed rows carry one** — they predate C-053. So historical runs score **zero
strict successes structurally**, which the control plane already warned must not be misread as a regression.
It is a property of the *artifacts*, not of the code.

**T-100 therefore requires FRESH Wave B legs.** `TEST_MATRIX.md:477`: `PMC12452463 ×2, PMC12096016 ×2`, ~1.5 h,
expecting both to pass the quarantine boundary and **`PMC12452463 → review_required`, not strict success
(TRAP-1)**. That leg production is LLM-backed and was **not run** — deferred for quota, not skipped silently.

### 3. Priorities 1, 2 and 4 are NOT attributable to any card merged today

`git diff --stat a662c3f..HEAD -- src/t2pw/bench/semantic.py src/t2pw/bench/goldset.py` is **empty**. The
priority 1–3 scorer and the gold set are **byte-identical across all five merges** (C-052, C-050k, C-056a,
C-030a, C-056b), and the run artifacts are from **2026-08-04**. So the 13 false real identifiers
(`PMC12096016`, `PMC12180156`, `PMC12782028`, `PMC12856317`) and the 2 unsupported retained reactions
(`PMC12180156`) are **pre-existing properties of an August partial run, measured by untouched code.**

**Classified under `PRODUCT_CONTRACT` §14: NOT a `product_contract_violation` caused by this sprint's merges.**
Whether each is a genuine pipeline defect or a `gold_data_defect` is **unadjudicated and out of today's
scope** — and **a benchmark failure does not by itself justify a code change.** No correction campaign is
opened on it.

### 4. What T-100 DID establish — C-056b is live and correct in the shipped report

The `SEMANTIC PATHWAY SUCCESS` question now reads, verbatim in the generated report:

> *"…and the run itself did not record a FAILED runtime semantic verdict? **The runtime verdict can only
> REMOVE a confirmation here; a runtime pass is never counted as one.**"*

That is C-056b's subtractive-only rule **observable in production output**, not merely in a test — the
strongest available confirmation that the 1-of-4 inflation hazard is closed at the seam that would have
inflated. Separated denominators also hold: extraction 4/5 = 80% and gold relevance 8/10 = 80% are reported
**without sharing a denominator** with the failing rates, which is the whole point of the C-041 split.

### 5. Standing prohibition, restated

**No strict benchmark-success figure may be quoted from this run or any historical run.** The next quotable
figure requires the fresh Wave B legs of §2. **Do not weaken the PARTIAL judgement, the affirmative
`strict_acceptance_eligible` gate, or priorities 1–3 to obtain one.**

---

## D-054 — Three of D-052's F-053 claims are wrong; C-056c's real boundary is four functions, not two · 2026-08-18 · LOCKED

**F-053 stands. Its binding prohibition stands unchanged. Its stated mechanism does not.** An independent
read-only measurement of the live tip (`1c06918`) confirmed two of D-052's six load-bearing claims, refuted
one, found one incomplete, found one over-read, and found one mislabelled. C-056c must be chartered off the
corrected facts below, **not** off D-052 §2 as written.

This is the fourth time in this sprint that a record's cited mechanism proved false while its conclusion
survived. The conclusion surviving is why the prohibition is not touched here.

### 1. REFUTED — `semantic_verdict` does not discard `evaluable`

D-052 says `semantic_verdict` *"computes `evaluable` and **discards** it"*. Live, at
`src/t2pw/pipeline/release_status.py:344-356`, it is **consumed**:

```
344:    evaluable = 0
...
349:            evaluable += 1
...
353:        return SEMANTIC_FAILED, "", tuple(failed)
354:    if not evaluable:
355:        return SEMANTIC_NOT_EVALUATED, SEMANTIC_NO_GATING_CHECK_EVALUABLE, ()
356:    return SEMANTIC_PASSED, "", ()
```

**The accurate statement:** `evaluable` is consumed **only as a boolean at `:354`**. The *count*, and the
*identities* of the checks that were evaluable, are dropped at the returns `:353` and `:356`, which carry no
evaluability payload. The loop never records the evaluable names at all — only `failed` names survive
(`:351`).

**The gap F-053 names is therefore real and unchanged**: nothing in the return distinguishes "1 of 4
evaluable, that one passed" from "4 of 4 evaluable, all passed". But it is not a discarded value; it is a
**boolean collapse that was the value's only use**. A card told to "stop discarding it" would look for a
deletion that does not exist.

D-052's line cite `:339-356` is also mis-ended. The function head is **`:307`**; `:339` is a `reason =`
assignment mid-guard.

### 2. INCOMPLETE — the minimum is four sites in two files, not two

D-052 names two changes: a fourth return value from `semantic_verdict`, and a new `classify_release_status`
argument. **Both are necessary and neither is sufficient.** With only those two, the evaluability reaches
`classify_release_status` and stops there — it never reaches a manifest, so **F-053 would not be
discharged**. The minimum boundary is:

| # | File :: symbol | Change |
|---|---|---|
| 1 | `release_status.py :: semantic_verdict` `:307-356` | widen the return to carry the evaluable set. **Exactly one `src/` caller** (`strict_quarantine.py:2130`), so migration cost is bounded |
| 2 | `release_status.py :: ReleaseStatus` `:223-259` **and `to_dict()` `:275-290`** | one new field with a default, one new key. **D-052 does not name this and it is mandatory** |
| 3 | `release_status.py :: classify_release_status` `:359-478` | one keyword-only parameter with a byte-preserving default, mirroring `semantic_failed_checks` `:370` |
| 4 | `strict_quarantine.py :: quarantine_and_close` — **statements `:2130` and `:2132-2144` only** | unpack the wider return, pass the new argument |

`batch/driver.py` needs **zero** edits: `_release_status_row` (`:647-667`) copies the dict wholesale, so a
new `to_dict()` key propagates to the manifest by itself.

### 3. OVER-READ — D-042 §4 withholds one parameter on one function from one card

D-052 paraphrases D-042 §4 as *"a new parameter on that seam is NOT granted and needs its own card"*, and
attaches it to `classify_release_status`. D-042 §4's actual sentence (`DECISIONS.md:2277`) is:

> **Giving `quarantine_and_close` an `admission` parameter is NOT granted to C-056a**; it is a signature
> change on a seam C-057 also touches, and it needs its own card.

It names one parameter, one function, one card. It says nothing about `classify_release_status`. And **new
`classify_release_status` parameters have already been granted twice and shipped** — `semantic_evaluation`
and `semantic_not_evaluated_reason` (`:368-369`) under C-056a, `semantic_failed_checks` (`:370`) under
C-056b. Three keyword-only parameters with byte-preserving defaults already exist on that signature; a
fourth is **precedented, not prohibited**.

**D-052's own later, narrower restatement is faithful and is the one that binds:** *"an `admission`
parameter on `quarantine_and_close` remains ungranted"*. **C-056c charters off that restatement.** It may
add a keyword-only parameter to `classify_release_status`. It may **not** touch `quarantine_and_close`'s
signature, and it may not touch the `evaluate_production_semantics` argument list at `:2123-2129`.

### 4. MISLABELLED — the 1-of-4 figure is the null-context arm, not "the seam's own derivation"

The numbers in D-052 are right and reproduce from `evidence/c056b_s0_measured.json` over 32 payload legs:

| Arm | evaluable-count distribution | verdicts |
|---|---|---|
| `pathway_context_none` | `{"1": 32}` | 7 failed / 25 passed |
| `gold_derived` | `{"1": 1, "3": 31}` | 9 failed / 23 passed |

**None ever reached four — confirmed.** But D-052 labels the 1-of-4 arm *"the seam's own derivation"*, and
the measurement script names it `pathway_context_none` (`c056b_s0_measure.py:213`). The seam's actual
derivation is `strict_quarantine.py:2116` `requested = pathway_context_from_stage_zero(pathway_context)`,
and **both live callers pass a real context** (`streamlit_app.py:1139` and `:3993`). Under a
context-carrying production run the seam sees the `gold_derived` shape — **three** evaluable, not one.

The null-context arm is correct **for the committed artifacts**, whose payloads carry no context exactly as
`quarantine_and_close`'s docstring warns at `:1822-1828`. It is not the production derivation. D-052's next
clause does report the 3-of-4 figure, so the record holds both numbers; only the label is wrong.

**Consequence for C-056c:** the shortfall it must make visible is **3 of 4 in production**, and 1 of 4 only
when replaying context-free artifacts. A card that hard-codes "one evaluable" would be wrong in production.

### 5. CONFIRMED, exhaustively — the two claims the prohibition rests on

* **`ReleaseStatus.semantic_confirmed` (`release_status.py:261-264`) has ZERO `src/` consumers.** Exactly
  four repo-wide hits: the definition, one `tests/` assertion
  (`test_release_status_classification.py:255`), one *different* symbol
  (`test_c056b_semantic_denominators.py:389` asserts `ModeResult` has no `runtime_semantic_confirmed`), and
  D-052 itself. It is unreferenced, not unreachable.
* **`CHECK_RAG_REINTRODUCTION` is structurally unevaluable at this seam.** Three links:
  `strict_quarantine.py:2123-2129` never passes `admission=`; `semantic_production.py:278` defaults it to
  `None`; `semantic.py:940-949` then sets a non-empty `inapplicable_reason`, so
  `release_status.py:347` `continue`s past it before `ok` is read. **There is no admission report to pass** —
  `quarantine_and_close` has no such parameter and `strict_quarantine.py` contains zero
  `AdmissionReport` / `admission_report` / `rag_admission` tokens. The seam says so itself at `:2120-2122`.

**Every predicate on `semantic_evaluation` in `src/` today is either `== SEMANTIC_FAILED` (subtractive, two
live denominator uses at `acceptance.py:752` and `:807-811`) or `== SEMANTIC_PASSED` (affirmative, zero live
uses).** The only affirmative escape is `describe()`'s rendering at `release_status.py:513-518`, which
reaches a reader's eyes and never a rate — exactly as F-053 says.

**THE BINDING PROHIBITION IS UNCHANGED AND UNWEAKENED.** No card may read `semantic_evaluation == "passed"`
affirmatively, or build any denominator, numerator or rate on it, until F-053 is discharged.

### 6. RULING — the serialized shape of evaluability

The record never specified **what** travels beside the verdict. Three shapes were available: the names of
the evaluable checks, a bare count, or the full four-way applicable/inapplicable map. **This ruling picks
the third**, because it is the only one from which a reader can reconstruct *why* a check did not count, and
because `semantic.py` already computes the `inapplicable_reason` strings that make it free.

**C-056c serializes, beside the verdict: for each of the four `SEMANTIC_GATING_CHECKS`, whether it was
applicable, and when it was not, the `inapplicable_reason` already produced.** A count is derivable from it;
the reverse is not.

### 7. Out of scope, and each already ungranted

* **Four-of-four evaluability is unreachable and stays out of scope.** Making `CHECK_RAG_REINTRODUCTION`
  evaluable needs the `admission` parameter — ungranted at `DECISIONS.md:2277`, and colliding with C-057.
  **C-056c makes the shortfall visible; it never closes it.**
* **Lifting the prohibition is a separate product decision.** C-056c builds the carrier. Authorizing an
  evaluability-aware affirmative predicate, or any rate built on one, is a later ruling and not an
  implementation choice.
* `bench/semantic.py`, `bench/semantic_production.py`, `bench/acceptance.py`, `streamlit_app.py` and
  `batch/driver.py` are **not** C-056c's.

### 8. Two consequences that must be disclosed, not absorbed

* **`quarantine_report["schema_version"]` 5 → 6** (`strict_quarantine.py:2240` — **address corrected 2026-08-20; this clause said `:2219`, which was the pre-merge address and is now stale. Found by C-057, verified live**), with its pin at
  `tests/test_strict_quarantine_release_seam.py:699`. The house rule mandating the bump is stated in-file
  at `:2214-2218`. **Authorized, and it must be stated in the card's report.**
* **`tests/test_batch_driver_seam_golden.py`'s GOLDEN digests move.** `:112` folds
  `row.get("release_status")` into a per-leg digest and `:174-182` pins seven sha256s, so **any** new
  `to_dict()` key moves at least the two release-status-carrying legs. **Authorized as a deliberate baseline
  move under merge rule 4, with an exact documented delta** — the same mechanism D-052 §1(b) ratified for
  C-056b. **It is disclosed and ratified, never absorbed silently.**

### 9. C-057 serialization — confirmed at file level, unverified at function level

C-056c touches exactly one function in `strict_quarantine.py`: `quarantine_and_close`, and within it only
`:2130` and `:2132-2144`. **C-057 has no charter and no prompt file**, so its functions are UNVERIFIED; its
lineage writes would most naturally land in `_admit_processes` (`:1095`), `_prune_entities` (`:1294`),
`_prune_locations` (`:1349`), `_prune_biological_states` (`:1419`) or `_drop_quarantined_processes`
(`:1519`) — in which case the two are merely same-file — but could land in `quarantine_and_close`'s ~530-line
body, in which case they are the same function.

**They serialize regardless**, as `MASTER_PLAN.md:417` and `:430` already mandate. **C-056c goes first**, and
C-057's charter must be written against the resulting live source.

### 10. F-049 exposure — six files, 72 tests, zero chunk coverage

`test_strict_quarantine_release_seam.py` (24), `test_batch_pwml_artifact_naming.py` (13),
`test_release_status_classification.py` (12), `test_semantic_release_gating.py` (12),
`test_c056b_semantic_denominators.py` (9) and `test_batch_driver_seam_golden.py` (2) appear **zero** times in
`TEST_MATRIX.md`'s chunk tables and **zero** times in `evidence/chunk_d_gate.py`. **SMOKE contains none of
them.** C-056c must name all six as its focused set and run them explicitly; no chunk it is told to run will
contain any of them.

---

## D-055 — T-100 is NOT discharged; its fresh legs are accepted as evidence and its acceptance criterion failed · 2026-08-18 · LOCKED

Four product-owner-authorized curator-enabled Wave B legs ran against integration `ad64e86` into
`runs_verify/2026-08-18_1328` (committed `8ea52c4`). This records the verdict, the sequencing constraint the
findings impose, and the one figure that may be quoted (none).

### 1. The run is a valid instrument

G11 on the bounded wrapper: `final_surviving_count 0`, `cleanup_success true`, 11 descendants observed and
all 11 terminated, `forced false`, 4 pre-existing processes reported and never killed. 63.3 min, well inside
the 9000 s ceiling. Exit 1 is `batch_run.py`'s documented *"something did not pass"*, not an infrastructure
failure. `cache_snapshot/` stayed out per D-011.

**One leg is disqualified as evidence.** `PMC12452463/research` died at `post_pipeline` on
`[Errno 22]` against `data/id_mapping_cache.json` after producing 5 reactions and 20 entities. The
discriminating experiment — `PMC12096016/research`, same mode, same code — **PASSED**. Three
`git worktree add` invocations by the orchestrator ran during the window, one inside leg 2's. **Recorded as
an orchestrator-induced infrastructure failure (F-064). Leg 2 must not be cited as evidence about the
pipeline.** Standing rule adopted: **no worktree creation or heavy filesystem work in the primary checkout
while a pipeline leg runs.**

### 2. NO FIGURE IS QUOTABLE, and no re-run of these four legs changes that

```
scripts/bench_acceptance.py --run-dir runs_verify/2026-08-18_1328   -> exit 1
NOT ACCEPTED: this run is INCOMPLETE (2/10 papers, 4/20 legs)
```

`bench/acceptance.py:397` computes `complete = complete_cases == planned`, where `planned =
len(self.papers)` is **the gold-set size, 10** — not the number of papers in the run. A two-paper run is
structurally incomplete. Its source comment is correct: *"a run that attempted 19 of 20 is the same failure
in miniature."*

**The previous handoff's standing instruction is corrected here.** It held that fresh Wave B legs are what
*"produces a quotable strict figure"*. They are **necessary** — only rows carrying `release_status` score
strict at all, and `driver.py:778` writes it, so fresh rows do carry one where **0 of 143** historical rows
did. They are **not sufficient**. A quotable strict pass **rate** requires all 20 gold legs, which is
**T-104** (~7 h) and outside this authorization. **Do not weaken the completeness gate to obtain a number.**

### 3. T-100's acceptance criterion FAILED, and the mechanism is located

`TEST_MATRIX.md:477`: *"both pass the quarantine boundary; **PMC12452463 → `review_required`, not strict
success** (TRAP-1)"*.

Both strict legs **reached** the quarantine boundary and wrote a `quarantine_report.json`, but both carry
`ok: false` and both classified `diagnostic_only`. **`review_required` was not reachable.**
`release_status.py:414-419` tests `strict_gates_passed` **before** any coverage branch, and
`strict_quarantine.py:2025-2034` appends `structural_reasons` to `refusal_reasons` **unconditionally**, while
converting only coverage reasons into review reasons. Both legs had `defensible_core = True`
(`minimum_core_satisfied: true`, `coverage.reasons: []`, `core_accepted` 6 and 9). See **F-062**.

**T-100 is NOT discharged.** It remains open pending the corrections below.

### 4. Adjudicated under §14 — seven violations, none of them the ones first suspected

`pwml-bio-auditor` classified every strict-leg failure. **Both legs failed for the same proximate cause**: a
RAG-imported alias-duplicate of EntB (`Isochorismatase (EntB)`, sharing UniProt `P0ADI4` with the real EntB
row) that becomes a degree-zero orphan and refuses the whole export (**F-057**).

**Neither leg is an incomplete-but-correct pathway wrongly dropped.** Both payloads carry content the
contract forbids shipping — `EntE` fabricated as the transporter on spans that name TolC and TonB
(**F-058**), and leg B's LDH coupled-assay reporters NAD+/NADH carried as pathway metabolites with a
`paper_explicit: explicit` provenance claim. **The refusals prevented contract-forbidden content from
shipping: the right outcome by the wrong route.**

Registered **F-055** … **F-064**. Three hypotheses in the dispatch brief were **wrong**, and the record says
so:

* **ATP is not a gold defect and not a policy disagreement.** Gold does not feed `requested_core` at runtime
  — `decision_inputs.requested_core: null`, `requested_core_source: "pathway_context"`. The cofactor policy
  demoted ATP in a **ledger-only** entry that mutates no row; ATP is a live participant in
  `/processes/reactions/3`. The cause is a name-resolution asymmetry in the coverage matcher (**F-061**).
  **Gold's ATP entry is quote-backed and correct. Fix the matcher, not the gold set.**
* **The `enterobactin synthase` accession demand is not a category error to relax.** The correctly resolved
  four-subunit PathBank complex was already in the payload; the bare protein row is synthetic, with no
  `paper_extraction` lineage. Gold forbids the identifier outright. **Do not weaken that gate — merge rule 6.**
* **C-060's admission gate was reached, ran, and correctly abstained.** `entity_admission_report` is present
  with `{removed: 0, demoted: 4, admitted: 0}`. The `ent gene clusters` dangling endpoint is **Stage 1's**,
  not C-060's (**F-063**).

### 5. ⚠ BLOCKING SEQUENCING — merge rule 6

**F-062 (the refusal seam) MUST NOT be fixed before F-057 (the RAG alias-duplicate) and F-058 (the
fabricated transporters).** Repairing the seam in isolation would make both legs exportable and thereby
**ship** the fabricated `EntE` transporters and the LDH-derived NAD+/NADH. That is precisely the
"weaken a biological gate to increase PWML production" failure merge rule 6 forbids.

**Order: F-057 and F-058 first, F-062 after.** No card may be chartered against F-062 until both land.

### 6. One correction to the adjudication, issued by the orchestrator

The adjudication reported F-062 as contradicting `PRODUCT_CONTRACT.md:341` (§13), which fixes PMC12452463's
correct outcome at `review_required` / `strict_acceptance_eligible=false`, *"never strict success"*. The run
produced `diagnostic_only` with `strict_acceptance_eligible: false` — the flag matches, the status does not.

**But that contract row is conditioned on *"after the index fix"*, and that phrase occurs exactly once in the
entire control plane with no antecedent** — `grep -rn "index fix" docs/pwml_recovery_sprint/*.md` returns
that single line. **Whether the condition is satisfied is UNVERIFIED, so this must not be quoted as a settled
contradiction of a locked position.** F-062's *mechanism* is read from the code and reproduces on both legs;
that stands independently.

**The undefined referent is itself a control-plane defect**: a locked row conditions a required outcome on an
event the control plane never names, so no agent can determine whether the row currently binds. **The product
owner should name the referent or strike the condition.**

### 7. What T-100 established, and it is not nothing

* **C-053's `release_status` carry works in production.** Fresh rows carry it; 0 of 143 historical rows did.
  The structural cause of the historical `0/2` strict score is confirmed and is not a regression.
* **C-056a's semantic wiring works at the quarantine boundary** — it computed `semantic_evaluation: failed`
  with a named failed check on both strict legs.
* **And the batch driver throws that verdict away** (**F-055**), disarming C-056b's subtractive rule on every
  gate-failed leg. **No historical figure moves**, because no historical row carries a `release_status`.

**Ten findings, seven of them `product_contract_violation` with artifact evidence, gold citations, affected
files, expected corrections and regression fixtures. That is what the milestone was for.**

---

## D-056 — "after the index fix" means C-010, merged as `72ee20f` · 2026-08-21 · LOCKED

**Ratified by the product owner**, PACK 11, integration `0f27f72`. This is the naming that
`DECISIONS.md` D-055 §6 asked for — *"name the referent or strike the condition."* **D-055 is
not reopened, amended or contradicted**; its request is discharged.

`PRODUCT_CONTRACT.md:341` (§13, LOCKED) conditions PMC12452463's required outcome on *"after
the index fix"*:

> *"Correct outcome **after the index fix** is `review_required` with
> `strict_acceptance_eligible=false`. **Never strict success.**"*

**The referent is C-010, "p01 stale positional index", merged at `72ee20f`.** The condition is
therefore **already satisfied**, and `PRODUCT_CONTRACT.md:341` **binds today**.

### ⚠ The SHA in F-080 is WRONG and must not be propagated

F-080 records C-010 as *"`MERGED` at `9e06360`"*, and the PACK 11 takeover brief inherited it.
**`9e06360` is C-010's BASE, and is C-012's merge.** The error is a misread of the LEDGER's
14-column card table, which carries `Base SHA` and `Merge SHA` five columns apart. Registered
as **F-085**, verified four ways:

* `git log -1 72ee20f` → *"Merge C-010 (agent/p01-stale-index): degree zero answered against
  the pre-prune snapshot"*
* `git log -1 9e06360` → *"Merge **C-012** (agent/p00b-driver-seam)"*
* `git merge-base --is-ancestor 9e06360 72ee20f` → true, the base→merge relation
* `git show --stat 72ee20f` touches precisely C-010's declared ownership
  (`strict_quarantine.py`, its two test files, `docs/change_log.md`), while `9e06360` touches
  `driver.py` and writes `evidence/g11/**C-012**/`

**The correction strengthens the reading.** F-080's load-bearing claim was that no competing
candidate exists; that is now measured rather than argued —
`git log --oneline --all --merges | grep -i index` returns **exactly one commit in the entire
repository**, and it is C-010's merge.

### Consequences, all of which take effect immediately

1. **T-104's acceptance row is now well-defined and quotable.**
2. **F-062 may be quoted as a live contradiction of a locked position**, which D-055 §6
   previously forbade.
3. **F-062 requires NO code card.** Its mechanism was correctly read and the routing seam is
   byte-identical at tip, but its proposed remedy was refused on evidence by F-081, and the
   correct repair merged as C-067. The four remaining structural reasons are each adjudicated
   `keep_refusing`, so the unconditional append is now **correct** behaviour. See
   `F-062-DISPOSITION.md`.
4. **The confirming measurement is T-104, not a card**, because the quarantine input payload is
   not persisted and neither committed file matches `admitted_payload_hash`. **F-081's own
   MEDIUM caveat must be carried into T-104 triage:** *"If the flagged row's synonym set is
   disjoint from `keep_norms`, the theorem is wrong and there is a third divergence not yet
   found."*

**Standing rule adopted, from F-085 and PACK 11 RULING 1:** before citing any sprint SHA, run
`git log -1 --format="%s" <sha>` and confirm the subject names the card you think it does. A
citation lifted from a wide Markdown table must name the column it came from.

---

## D-057 — `round_cap_reached`, an eighth termination reason · 2026-08-21 · LOCKED

**Authority note, stated plainly because the record should show what actually happened.** The
product owner was presented with this entry as drafted and replied *"idk what this means do
whatever you need to"* — an **explicit delegation to the orchestrator**, not an independent
adjudication. It is entered on that delegation. **A later product-owner review may amend or
strike it without treating this as reopening a contested decision.**

**Extends D-005 and D-024. Neither is reopened, amended or contradicted** — D-005's six named
reasons and D-024's seventh keep their exact meanings, their exact strings and their exact
denominator rule. D-005 goes from seven named termination reasons to **eight**.

C-055 built the RAG loop's round controller. When the configured ceiling of rounds is spent the
loop stops, and — correctly, since none of the seven fitted — reported **no termination reason
at all**. `round_cap_reached` genuinely *ends* the loop, so the one stop that reliably
terminates it was the one stop that said nothing. C-064 (merged `d0b5d51`) closed F-070.

**The reason.** `round_cap_reached`. Used when **all** of these hold:

* the configured maximum number of RAG rounds has been consumed;
* the loop has not otherwise terminated;
* **no** deadline or timeout caused termination;
* **no** explicit refusal caused termination;
* **no** separate resource / token / budget exhaustion caused termination;
* **no** stronger existing terminal reason truthfully describes the outcome.

**Precedence, mandatory: rank 8** — below `budget_exhausted`, `operation_timeout`,
`identical_empty_response`, `scientifically_unrecoverable`, `retrieval_exhausted`,
`no_new_claims` and `attempt_cap_reached`. A round cap is the weakest true statement about why
a loop stopped: any of the seven above it, when true, is more informative.

**Never mislabel a round cap as timeout, refusal, success, or generic budget exhaustion.**
Equally, never mislabel it as `retrieval_exhausted` or `no_new_claims`: those require the
configured loop to have actually completed, and a loop cut off by the ceiling is precisely one
that did not. It is claimed on the controller's **recorded cap refusal**, not on
`rounds_remaining == 0` — the ceiling must actually have stopped a round that wanted to run.

**`OPERATIONAL_TERMINATION_REASONS` is UNCHANGED** — it stays exactly
`{budget_exhausted, operation_timeout}`. `round_cap_reached` is **not** added to it, for the
same reason D-024 kept `attempt_cap_reached` out: a configured ceiling is a safety limit, not a
promise, and that is a different fact from a leg that ran out of clock. **Whether the round cap
should count in the pipeline-completion and end-to-end strict-success denominators is a product
decision that has not been made; until it is, the denominator does not move.**

---

## D-058 — T-101 and T-103 are AUTHORIZED to run live · 2026-08-21 · LOCKED

**Authorized by the product owner**, PACK 11, together with one free read-only
`GET https://openrouter.ai/api/v1/key`.

Approximately **3.8 h combined wall clock**. T-101 is a deliberate 6-leg superset
(`scripts/batch_run.py --modes` is per-run, not per-paper; a superset satisfies the acceptance
criteria in one invocation). T-103 runs 4 legs with **`T2PW_SPECIES_LLM=0` MANDATORY**
(PACK 9 RULING 3 — **T-104 must NOT inherit it**) and `T2PW_OFFLINE_CURATOR=1` recommended.

### ⚠ The cost premise was checked and it holds — but NOT for the reason on record

The authorization package stated *"approximately $0 marginal cost, all-free-tier"*. **The
authorized key check partly contradicted that and was worth running:**

```
is_free_tier        False
limit               75
limit_remaining     71.809565116
usage               158.722468024
is_provisioning_key False
```

**The KEY is not a free-tier key.** It carries a real $75 limit with $71.81 remaining and
$158.72 of historical spend. So *"all-free-tier"* was true of the **model slots** and false of
the **account**, and those are different claims.

**The ≈$0 conclusion nevertheless stands, on a re-derived basis.** All nine configured
OpenRouter slots are `openrouter/free`, and a read-only `GET /api/v1/models` confirms that is a
**real advertised model** — *"Free Models Router"*, `pricing.prompt = 0`,
`pricing.completion = 0`. Zero-priced models on a non-free account still cost zero.

**Standing caution:** because the account is not free-tier, **any fallback to a non-free model
would spend real money against $71.81 of remaining limit.** No fallback is configured — no
`OPENROUTER_*_FALLBACK` variable exists in `.env`, verified — but a run that changes a model
slot is no longer a $0 run, and the run record must state which slots were used.

### Sequencing, mandatory

Serially, **T-101 then T-103**, each holding the wrapper-owned heavy mutex via
`bounded_run.py --heavy-lock`. **Never concurrently** — two live-curator legs at once risks
free-tier rate-limit corruption and shared-cache races. Offline work continues in other lanes
while a milestone runs.

**Command form corrected and measured:** `bounded_run.py` has **no `--env` flag**, and the child
inherits the wrapper's environment. Environment variables go in the **shell prefix**, not as
`env VAR=x` after the `--`, which would make `env` the child executable. Verified by execution.
This supersedes `T101_T103_AUTHORIZATION.md:169-179`.

**Explicitly NOT authorized here: T-104 and T-105.** Each is a separate ~7 h, 20-leg release
candidate, and **they must never be collapsed into one run** — T-105 is the second candidate and
requires a triage and correction pass between the two.

---

## D-059 — an unmarked Stage-1 row's `paper_explicit` claim is RECORDED, not VERIFIED · 2026-08-21 · LOCKED

**Authority note.** As with D-057, the product owner delegated this one to the orchestrator
(*"do whatever you need to"*) rather than adjudicating it. Entered on that delegation and
**amendable on review without treating it as a reopened contest.**

**The question.** Does `PRODUCT_CONTRACT.md:85-102` §3's requirement that every entity identify
*"whether it was paper-explicit"* require that claim to be **VERIFIED**, or merely **RECORDED**
as the extraction asserted it?

**The ruling: RECORDED.** F-078's provenance half (Half B) closes **with no card**, and the
residual is documented as accepted behaviour.

**Reasoning.** §3's stated purpose is *"so that false content can be attributed empirically to
Stage 1, RAG, inference, audit, mapping, gap resolution or another stage."* **Attribution to
Stage 1 is exactly what the current stamp achieves** — it says *this came from the extraction,
and the extraction did not flag it*. Changing an unmarked row to `not_evaluated` would make the
field **less** informative about origin while buying no verification, because the verification
capability §3 would need does not exist at that seam and cannot be added narrowly:
`settle_stage_one` (`stage_one_boundary.py:411-418`) takes **no source-text parameter**, and its
only production call site is `streamlit_app.py:5476`.

**The accepted residual, stated so it is not rediscovered as a defect.** `_paper_entry`
(`:311-315`) stamps `paper_explicit="explicit"`, `review_required=False` on **any** row the
extraction did not self-mark as `inferred`/`enriched` or carrying an `inference`/`rag_provenance`
mark. The seam has no paper text with which to check, and its own docstring concedes this
(`:168-171`): *"this seam receives a payload, not the paper it was drawn from."* **The `explicit`
claim is structurally unearnable at this seam, and that is now accepted rather than open.**

**F-078's Half A — the chemistry — is NOT closed by this entry.** An adenylation reaction
emitting free `AMP` is chemically impossible and the paper text confirms `AMP` appears only
inside an enzyme name. But that row is Stage-1 **LLM extraction output**: no deterministic
function in `src/` produced it, there is no `file:line` predicate to own, and **owning it
requires an authorised LLM leg**, which is a separate authorization and not a card.

---

## D-060 — `SEMANTIC_GATING_CHECKS` goes from four to five · 2026-08-21 · LOCKED

**Ratified by the product owner**, PACK 11, **with the measured blast radius in front of them**.

**The addition:** `"actor_named_in_its_own_cited_span"`, one named member. C-071 (F-079).

**What it is the mechanism of.** `PRODUCT_CONTRACT.md:343`, LOCKED — *"Structured status is
authoritative."* F-079 measured a payload carrying a fabricated transporter — an `EntE` actor
whose only cited evidence is a span naming **TolC** — classified `release_ready`,
`semantic_evaluation: passed`, `strict_acceptance_eligible: True`, and still reproducing
byte-identically at tip.

**This is the D-044 / D-052 §1 pattern**: a widening that is the mechanism of an
already-granted change is ratifiable, but **must be disclosed and ratified in the record, never
absorbed silently.** `tests/test_semantic_release_gating.py` exists precisely to force this
decision and it did its job; it is renamed to assert five, its closure property intact, and a
**sixth** silent addition remains impossible — proven by mutation, and it now also catches
silent **removal**.

### The blast radius, ratified deliberately

> **13 of 21 committed `runs_verify` legs and 8 of 14 `runs/` legs move from `release_ready` to
> `review_required` with `strict_acceptance_eligible=false`.**

The independent review judged the failures dominated by genuine fabrications — `EntE` cited by
spans naming TolC, TonB or EntF (20 of 221 actor rows corpus-wide are `Ent*` symbols a naive
substring test would accept because they sit inside the word *enterobactin*); `ALAS2 complex`
cited by a span naming no protein; `SFXN4 complex` cited by a span naming ALAS. **Marginal cases
land at `review_required`, which is the contract's answer for an uncertain identity** — flagged
for a human, not dropped.

**Not adjudicated here, and explicitly left open:** whether `review_required` is the
*biologically* right call for the marginal rows (`ALAS2` vs `ALAS`, `enterobactin synthase` for
`EntE`) is a `pwml-bio-auditor` question. The review judged them as **lexical evidence**
questions, which is what the check asks. **A read-only bio-audit of the demoted legs remains
available and blocks nothing.**

### Two design residuals, recorded as accepted, NOT fixed

1. **A payload where every actor row lacks a usable span makes the check `applicable=False`**,
   and an inapplicable gating check cannot demote — so such a payload still reaches
   `release_ready`, visible only as `NO_ACTOR_SPANS` in `semantic_check_evaluability`. This
   follows from the pre-existing D-006 architecture plus `CHECK_SOURCE_CARRIER` being
   deliberately non-gating. **Closing it would change how `release_status` treats inapplicable
   gating checks and is a separate product decision.**
2. **The multi-token hole**, quantified: ~**14 of 373 passing rows (3.8 %)** are corroborated by
   a single non-identifying token (`complex`, `homodimer`, `synthase`, `deacetylase`,
   `disaccharide`, `udp`). It points **only** in the under-reporting direction — it never
   demotes correct output. Closing it needs a hand-built stopword vocabulary with its own drift.

**F-053 is NOT touched by this entry and remains UNDISCHARGED.** C-071 respects it by
construction: it never reads `passed` affirmatively, it produces `SEMANTIC_FAILED` through a
gating check and lets the pre-existing subtractive cap at `release_status.py:542` demote.
**F-079 is itself fresh evidence F-053 should stay** — a `passed` verdict was measured on a
payload carrying a fabricated actor, with `strict_acceptance_eligible=True`, which is a
denominator-entry authorisation.

---

## D-061 — `TEST_MATRIX.md` may grow past 541 lines for the C-070 entry · 2026-08-21 · LOCKED

**Authority note.** Delegated to the orchestrator with D-057 and D-059.

The standing constraint reads: *"`TEST_MATRIX.md` has citations pinned through line 477. Patch
it in place and preserve its 541-line count **unless a separately authorized migration changes
that requirement**."* **This is that authorization**, and it is narrow.

**Granted:** `TEST_MATRIX.md` may grow to accommodate C-070's isolated-collection entry in
§ Chunks, describing `tests/test_isolated_collection.py` — two sub-second routine arms plus an
opt-in full sweep behind `T2PW_ISOLATED_COLLECT_ALL=1`, ~95 s for 156 files, belonging at
release-gate cadence rather than per merge.

**Two constraints survive unchanged:**

1. **Citations pinned through line 477 must not move.** The entry goes **after** line 477.
   Verified: placing it there preserves every pinned citation.
2. **The new line count must be recorded** in this entry once applied, so the tripwire keeps
   working at its new value rather than being abandoned.

**Explicitly refused: consuming the 14 blank lines after line 477 to keep the count at 541.**
That would have let the entry land without asking. **Using slack to keep a tripwire quiet is
precisely what the tripwire exists to prevent**, and a count that stays at 541 while the file
grows is a worse record than one that visibly moved with permission.

**APPLIED 2026-08-21 at integration `0f27f72`.** The entry was appended at the **end of the
file**, not in § Chunks — inserting at § Chunks (line 209) would have shifted every line
between 209 and 477 and broken exactly the citations this constraint protects. **D-061
authorized the file to grow; it did not authorize moving pinned lines.** Line 477 verified
byte-identical after the change.

**New pinned line count: 578** (was 541). That figure now binds in place of 541.

---

## D-062 — a Stage-0 organism conflict preserves the pathway as `review_required` under the OBSERVED organism; it neither exports strict nor drops the run · 2026-08-22 · LOCKED

**Product-owner ruling, taken in the T-104 session in response to F-095.** Recorded by the Lead
Orchestrator; the decision is the product owner's, not the orchestrator's.

### The question F-095 put

Three pinned gold cases — `PMC12657337`, `PMC12421875`, `PMC12312563` — carry
`requested_organism: "Bacillus subtilis"` while the papers are *E. coli*, *L. lactis* and
*L. monocytogenes*. They are deliberate traps: each `relevance_note` says "ORGANISM TRAP" in
capitals and each case lists `Bacillus subtilis` in `forbidden_organisms`.

At T-104 all six legs ended `scope_conflict`. Two of them had already extracted payloads that
**exceed** their gold connected-core floor with 100% enzyme and metabolite recall:

| paper | gold `min_connected` | connected core | enzyme recall | metabolite recall |
|---|---|---|---|---|
| PMC12657337 | 3 | **4** | 3/3 | 6/6 |
| PMC12421875 | 7 | **10** | 8/8 | 10/10 |

`PRODUCT_CONTRACT.md` carried no clause on a requested-versus-observed organism conflict, so the
contract did not settle it. Two intentional behaviours were in conflict: the gold set expects an
export labelled with the *actual* organism, and `config.py:194`
`eligibility_stage0_conflict_aborts = True` stops the run.

### The ruling

**Neither of the two obvious readings. A third.**

When Stage 0 reads an organism that contradicts the batch request, and the reading is *correct*:

1. The run **does not** export a strict, release-ready artifact under the requested organism. The
   request was wrong and a release-grade artifact must not be produced from a wrong request.
2. The run **does not** drop the paper either. The extracted pathway is **preserved as
   `review_required`, carrying the OBSERVED organism**, with the requested scope recorded alongside
   it so the mismatch stays auditable.

### Why the drop was the part that had to change

Merge rule 7 binds the whole sprint:

> It preserves incomplete-but-correct pathways as `review_required` rather than dropping them.

The current behaviour folds `scope_conflict` to `STATUS_INELIGIBLE`, whose own definition in
`batch/report.py` reads *"It is NOT a failure and not even a run: nothing was attempted, so nothing
failed."* For these two papers that is untrue on the evidence: something **was** attempted, it
succeeded, and it cleared the gold's own connected-core floor. A correct-but-mis-scoped pathway was
discarded. That is the case merge rule 7 exists to prevent, and the drop is what this ruling
reverses.

### What this does NOT authorize

* **It does not weaken the Stage-0 guard.** The guard's reading is correct and stays. What changes
  is the *disposition* of a correct reading, not the detection.
* **It does not raise the strict PWML rate.** A `review_required` artifact is not a strict export.
  Anyone quoting a strict-rate improvement from this ruling has misapplied it.
* **It does not license editing the topics file.** Supplying the actual organism removes the trap by
  handing the pipeline the answer and makes `forbidden_organisms` unexercisable. The pinned scopes
  stay as the gold set pins them, and `bench_acceptance.py --verify-plan` must keep returning `OK`
  with all ten `[pinned_override]`.
* **It is not itself a card.** No implementation was authorized in the T-104 session. The card that
  implements it must carry its own G9 proof and stay inside a declared ownership boundary.

### Consequence for the gold set

The gold's `expected_export: strict_exportable` for `PMC12657337` and `PMC12421875` is **not**
ratified by this ruling. Under D-062 the correct outcome for those two is `review_required`, so
either the gold field or this ruling will need reconciling when the implementing card is written.
**That reconciliation is a separate decision and is explicitly left open here.** Until it is taken,
neither paper counts as a strict-export success and the strict denominator is unchanged.

---

## D-063 — T-105 is HELD until F-094 and F-096 corrections are merged · 2026-08-22 · LOCKED

**Product-owner ruling, taken in the T-104 session.**

T-104 ran and is recorded as `MEASURED — NOT ACCEPTED`. T-105 is the **second** release candidate;
`MASTER_PLAN.md` requires `T-104 → triage/correction → T-105`, and its acceptance is *"remaining
failures explained and classified"* — a question that only has meaning after corrections exist.

The T-104 session was scoped to runs, triage and recording, with no authorization to implement
corrections. Re-running the same 20 legs against unchanged code would therefore have produced no new
acceptance information, would have cost ~5.5 h, and would have **collapsed the two release
candidates into one**, which the sprint has forbidden since PACK 9.

**T-105's prerequisite chain, restated:**

1. Cards are opened for **F-094** (`product_contract_violation` — PMC12452463/strict reached
   `release_ready`, which `PRODUCT_CONTRACT.md` §13 forbids outright) and **F-096**
   (`product_contract_violation` — 7 false real identifiers emitted on legs reported `PASS`).
2. Those cards are implemented, reviewed against the actual diff, and merged under the standing
   merge rules.
3. T-105 re-runs the same 20 pinned legs.

**F-095/D-062 is NOT in that chain.** Its implementing card may be sequenced independently; T-105
does not wait on it, and D-062's `review_required` outcome does not change the strict rate either
way.

**The blocker recorded against T-105 is now: "F-094 and F-096 corrections not yet merged."** It is no
longer "T-104 has not run" — T-104 ran on 2026-08-21/22 into `runs_verify/2026-08-21_2239`
(committed `2673067`).

---

## D-064 — a shared accession within one kind is identity, not conflict · 2026-08-23 · LOCKED

**Product-owner ruling, taken 2026-08-23. Recorded retroactively 2026-08-25** to close F-113: the
ruling was already governing merged production code and belonged in the locked-decisions file.
Recording it changes nothing and reopens nothing. **C-073, C-076 and C-080 are NOT reopened.**

The same UniProt accession may be shared by proven aliases of the same protein and by holo/apo
states of the same underlying polypeptide. `EntE` and `enterobactin synthase` are the same protein
identity. Holo-EntB and apo-EntB may share the underlying parent accession **while remaining
distinct pathway states**. **Do not flag these as accession conflicts unless the entities are
biologically unrelated or cross-kind. Update the scorer/gold classification rather than forcing the
pipeline to invent different protein identities.**

**`R196A` remains a distinct mutant polypeptide.** A point mutant is not an alias and not a state of
the wild type; sharing the parent accession with it is a conflict, not identity.

### Implemented by

C-076 (`3b7a7b1`, scorer + gold) and C-080 (`89aaced`, the production release gate, reading the same
identity predicate). C-073 was corrected against the D-035 clause 3c this ruling interprets.

### A known and deliberate gap in the implementation

The wording is *"biologically unrelated **or** cross-kind"*. **Both seams implement cross-kind
only**, because neither has a corpus-wide biological-relatedness oracle. Two genuinely unrelated
same-kind proteins fused onto one accession by a mapper bug are invisible to the scorer **and** to
the production gate. That mirrors the pipeline's pre-existing blind spot and is what the C-076
charter directed, so it is not a deviation — but the "unrelated within one kind" half is
**unmeasured corpus-wide** and is recorded here as a known gap, **not** an assumed non-issue.

---

## D-065 — `extracted_not_serialized`: a fourth disposition for a defensible core a scope guard stopped · 2026-08-25 · LOCKED

**Product-owner ruling on Decision bundle post-T-106, Item 1. Option A is ADOPTED**, with one
correction to the bundle's own framing.

### The ruling

`PRODUCT_CONTRACT` § 4 has no state for *"a defensible pathway core was extracted, but a correct
scope guard stopped the run before audit, DB mapping, freeze and PWML serialization."* D-062 assumed
one existed. It does not, and C-077 was right to decline to fabricate a gate result to reach
`review_required`. **That decision is not charged against C-077.**

A distinct disposition, **`extracted_not_serialized`**, is adopted for exactly that shape.

### For `PMC12421875` and `PMC12657337`

* They **must never count as strict exports** under the deliberately wrong requested organism.
* `forbidden_organisms` is **preserved** — the trap stays exercisable.
* `relevance_note` is **preserved** — the ORGANISM TRAP designation is the point of the cases.
* The topics files are **preserved** — D-062 forbids touching them in terms.
* `expected_export` changes `strict_exportable` -> `partial_only`.
* The prepared D-062 reconciliation sentence is appended to `export_rationale`, leaving the existing
  text byte-identical.
* `bench_acceptance.py --verify-plan` MUST remain `OK` with all ten `[pinned_override]`.

### The bundle's "no production work" claim is CORRECTED

The bundle asserted Option A *"costs no production code."* **That is not accepted.** A contract state
that is never emitted is not a contract state; adding a disposition to the gloss while the emitted
and scored record continues to describe the same leg as something else would be exactly the kind of
untruth C-077 was chartered to remove.

**The emitted and scored record must be honest.** Before implementation, the disposition must be
resolved as one of:

1. a **runtime release status** (a fourth member of `RELEASE_STATES`);
2. a **benchmark disposition** layered over an unchanged runtime `diagnostic_only`;
3. an **additional explicit field beside** the existing runtime status.

**Reading 3 is the preferred implementation**, and it carries these constraints:

* **preserve** the existing safe runtime refusal — the run still stops before serialization;
* **expose** an explicit `extracted_not_serialized` disposition in the release/acceptance record;
* **never fabricate** `strict_gates_passed`;
* **never pretend** a PWML exists;
* update the `PRODUCT_CONTRACT` gloss **and** the scorer consistently, so no reader has to decide
  which of two fields to believe;
* **create no route toward strict export.** D-062 forbids that outright and this ruling does not
  reopen it.

If extending `RELEASE_STATES` turns out to be the only coherent implementation, that change is
**chartered and reviewed on its own merits** — it is not smuggled in as a documentation-only
correction.

### What this ruling does and does not buy

It removes two structurally impossible cases from priority 5's strict denominator: **4 -> 2**. It
does **not** make priority 5 pass. `PMC12782028` is correctly blocked on coverage and `PMC12096016`
is correctly blocked twice over, so the honest result becomes **0/2** until the remaining
biological and code defects are corrected. **T-107 must not be authorised on an expectation that
priority 5 passes.**

---

## D-066 — `pytest.ini` keeps `pythonpath = src`; `TEST_MATRIX` rule 10's refusal is SUPERSEDED · 2026-08-25 · LOCKED

**Product-owner ruling on Decision bundle post-T-106, Item 2.**

The C-070 setting (`5bc600e`) **stays**. Removing it would re-break individual-file collection for 21
of 156 test files in order to solve a problem the mandatory pin already solves.

`TEST_MATRIX.md` rule 10 is amended to record its refusal as **superseded, not forgotten**. The
hazard it names is real — pytest *prepends* `pythonpath` entries ahead of the `PYTHONPATH` pin, so
the setting alone could make a base-tree G9 proof silently measure the tip. It is neutralised by the
pin, **not** by the setting's absence.

### Required mitigation, now mandatory rather than customary

* Every base-tree measurement runs through `pinned_pytest.py`.
* It passes `--expect-tree`.
* It carries a committed `--pin-verdict`.
* **An unpinned base-tree run is not evidence.**

### Folded in from F-114 — a second infrastructure mode that looks like a regression

* Omitting `--basetemp` can produce **false regression failures** (83 tests error with
  `PermissionError`).
* Specifying a `--basetemp` whose **parent directory does not exist** can *also* produce false
  regression failures — one measured instance errored 55 tests; creating the parent gave
  `339 passed`.
* **Pre-create the basetemp parent before pytest.**

Neither side of the contradiction was wrong on its own merits and neither author knew of the other.
**Not chargeable to C-070 or C-079.** Accepted base proofs from this wave were all pinned and are
**not rerun**.

---

## D-067 — `supported_reactions_complete` is set per paper, only where exhaustiveness is PROVEN · 2026-08-25 · LOCKED

**Product-owner ruling on Decision bundle post-T-106, Item 4. Option C is ADOPTED.**

`supported_reactions_complete` is **not** set `true` broadly. Option B wholesale is **refused**.

Work starts only where an exhaustive signature set can genuinely be established, beginning with the
two negative controls that already carry `max_retained_reactions`:

* `PMC13231680`
* `PMC12180156`

### Before the flag is set for ANY paper, all five must hold

1. The **complete scoped source** has been read.
2. **Every** supported reaction signature is defined.
3. The biological completeness has been **independently reviewed**.
4. The run is **confirmed compatible with the seed-only assumption** — `goldset.py:384` warns the
   flag is incompatible with multi-paper RAG synthesis unless the run is seed-only.
5. **Who established exhaustiveness, and how**, is recorded.

**If exhaustiveness cannot be proven, the field is left absent and priority 2 reports
`NOT EVALUATED`.**

### The prohibition

**The Boolean is never set merely to make priority 2 measurable.** A false value of `true` converts
every unattributed row into a reported fabrication; `semantic.py:700-704` records that this would
have reported **227** fabricated reactions in a run that produced far fewer. That is the worst
outcome available and it is one keystroke away.

C-085 made the report honest; it did not make the measurement exist. **This work does not block the
next release candidate.**

---

## D-068 — a prefreeze declination DEMOTES the release status; D-040 § 8's residual gets its owner · 2026-08-25 · LOCKED

**Product-owner ruling on Decision bundle post-T-106, Item 5. Option A is ADOPTED. Option C —
folding F-123 into D-065 — is REFUSED.**

### Why they are not folded together

F-107 and F-123 are the same *principle* at **different lifecycle positions**, and merging them
would produce one blurred rule instead of two exact ones:

* **D-065 / F-107** stops **before serialization** — no PWML exists and none may be invented.
* **F-123** retains an **intact graph** and may still legitimately serialize.

### The ruling

The prefreeze verdict is **threaded into release classification**.

* `prefreeze_report["ok"] = False` **with a review-required reason** demotes the result to
  `review_required`.
* Such a run **must never reach `release_ready`**.
* **Useful intact biology remains available** — the payload is not discarded.
* The ambiguous rename **remains declined**; no unsafe merge is guessed.
* **No payload is discarded merely because the rename was ambiguous** (merge rule 7).

### This extends D-040 § 8 explicitly

D-040 § 8 split D-029 so that C-052 **persists and surfaces only**, and assigned *"acting on
`review_required`"* to **no card**, registering it as backlog **`BL-004`**. **`BL-004` is now
assigned.** The production change lands in `release_status.py` or the narrowest appropriate seam —
**not** in the product-owner-owned `streamlit_app.py`.

C-082 is **not** charged with stopping short. Closing this was outside its boundary and required a
locked-decision extension it had no authority to take; the reviewer's judgement on that point is
accepted.

### The prohibition

**Do not rely on unrelated gates to prevent a successful export.** D-035 clause 8's *"must not
become a successful export"* is currently enforced only by other gates and never by this channel.
An invariant that holds by coincidence of ordering is not an invariant.

---

## D-069 — an interaction endpoint confers no PARTICIPANT role; it may still support IDENTITY · 2026-08-25 · LOCKED

**Product-owner ruling on Decision bundle post-T-106, Item 6. C-081's inherited scope note is
ratified in a NARROWED form, not as written.**

### NOT ratified

> ~~An interaction endpoint does not license a database identity.~~

That blanket statement is **refused**. C-081's corpus result — **zero observed collateral over 18
refusals across 89 artifacts** — is useful evidence but does **not** prove the blanket rule safe
outside that sample.

### Ratified instead

* An interaction endpoint does **not, by itself**, prove that an entity participates in a reaction
  or transport.
* A **source-supported** interaction may still establish that the entity **exists**, and may support
  its **database identity**.
* Interaction evidence **must not promote** an entity into a reaction-participant role.
* An **interaction-only cofactor may retain identity** when the paper explicitly supports the
  interaction and the entity kind is valid.
* **Unsupported or merely inferred** interaction endpoints must not acquire real identifiers.

Cofactor binding — an `interaction` — is the canonical way papers state cofactor relationships. The
distinction this ruling draws is between **existence/identity** (which a supported interaction can
carry) and **participant role** (which it cannot).

### Required follow-up

Whether current production violates the **narrower** ruling is **measured**, not assumed. If it
does, a correction is chartered covering:

* valid interaction-only identity cases;
* unsupported interaction cases;
* reaction-participant non-promotion;
* legitimate ATP / cofactor preservation;
* corpus-wide collateral reporting.

**C-081 is not reopened unless that measurement reveals a conflict.**

---

## D-070 — O-1 is RULED: the pinned 21 is two populations, 16 + 5, and neither is a gold error · 2026-08-27 · LOCKED

**This closes O-1**, open since the F-132 decision bundle and the sole blocker on
`card/C-094-f134`, `card/C-098a-cap` and `card/C-098b-gate`. The product owner **rejects the
question as posed.** *"Gold-set error class, or legitimate biology preservation?"* is a false
binary: `placeholder_backed_proteins` counts **two measurably different populations** and the
answer differs between them.

Evidence: `evidence/g11/ORCH-710/01`–`06`, certified at `5d3c119`, 6 artifacts, 0 non-compliant,
every job `FINAL SURVIVING COUNT : 0` / `cleanup : success`. Index and the job-02 incident:
`ORCH-710-EVIDENCE.md`.

### The partition — exhaustive, mutually exclusive, measured

| Population | n | Where | Recognised by |
|---|---:|---|---|
| PathBank `Unknown` sentinel rows | **5** | `entities.proteins` | `is_pathbank_unknown_protein(row)` |
| generated functional wrappers | **16** | `entities.protein_complexes` | `generated: true` + `single_protein_pathwhiz_wrapper` |
| **overlap** | **0** | — | — |
| **total** | **21** | | |

One per affected pinned leg for the five. **None of the 21 sets
`placeholder_claims_real_identity`** — so none of them is a forged identity, and no ruling, report
or summary may describe them as one.

### O-1a — the five sentinels are PathBank's record, and its species is part of it

PathBank record **9659** is the `Unknown` protein, and *Arabidopsis thaliana* is that record's own
species. On a row whose entire content is *"this is PathBank record 9659"*, the Arabidopsis is a
true fact **about the record**, not a false mapping of an entity. Under the current PathBank
representation it is **not** a defect.

Therefore: treat the five as **sentinel rows**; do **not** classify them as functional generated
wrappers; do **not** strip or rewrite their record identity merely to remove Arabidopsis; do
**not** count them as forged biological identities; and keep their diagnostic classification
distinct from the sixteen.

### O-1b — the sixteen wrappers keep TRAP-3, and are scored separately

They are legitimate generated-wrapper biology. **TRAP-3 protection stands.** The disagreement is
not resolved by deleting a wrapper or by preventing its serialization; the class is **not** a gold
error; and placeholder status still may not forge a real database identity. They are scored
**separately from the five**.

### O-1c — the "four genuine losses" claim is REFUTED, and its rows were never in the 21

The alleged four genuine losses do not exist. What does exist is a **different population on a
different seam**: candidate identities retained in the identity verdict but **not shipped** —
**24** rows in the pinned run, **82** corpus-wide, and **zero** of the pinned 24 lie inside the
placeholder-backed 21.

Those rows may not stay hidden inside the O-1 metric. They are registered separately as **F-141**
and are **not** to be called `placeholder_backed_proteins`.

They are also **not** automatically production defects. Of the pinned 24: **22** fail because
species remains unknown, and **2** (Fur) fail because no candidate describes the shipped
identifier. Withholding a species-specific protein identifier **is correct** when species evidence
is absent. The metric must therefore separate a correct safety withholding from a recoverable
evidence-propagation loss, classifying each row as exactly one of:

1. source-supported species was available but discarded;
2. species genuinely unresolved, so withholding was correct;
3. conflicting species evidence;
4. candidate failed to describe the shipped identifier;
5. identity or species evidence lost across a stage boundary;
6. other measured mechanism.

**A real accession in `identity_verdict.identity` is not a licence to ship it.**

### What this authorises, and what it still forbids

**Authorised.** A **narrow** card that preserves already-resolved, source-supported species when an
Unknown-backed functional wrapper is constructed or normalised — D-071's sibling, chartered as
**C-099**. Census: 31 Unknown-backed wrappers across 11 legs, **6** with resolved species beneath
the clobber, **25** without; **2** of the 6 are in the pinned 16, both `explicit_entity_species`.
For the other 14 pinned wrappers *"do not clobber"* is a **no-op**.

**Still forbidden.** `card/C-094-f134` **does not merge unchanged**: it is subtractive at all three
sites, removing species from wrappers that have none resolved, which inverts the O-1 statement
`test_protein_export_policy.py::test_strict_gates_accept_a_correctly_formed_unknown_backed_complex`
pins. **C-098c stays refused** — an export-time fallback that replaces a false Arabidopsis at
mapping time with a false default species at export time is merge rule 8 and the same defect one
stage later. **No path may end at `writer.py`'s `default_species_id`** to turn an unresolved species
into a confident one.

---

## D-071 — PMC12444477's Unknown tolerance becomes per-entity; the Boolean is scoped, not flipped · 2026-08-27 · LOCKED

`unknown_backed_proteins_acceptable` is a **case-wide Boolean**, and `pinned_v1.json:286` sets it
`true` for **PMC12444477** alone. Its `unknown_backed_rationale` names the entities the tolerance is
*for* — and names, in the same sentence, the class it must **not** cover. The scorer
(`bench/semantic.py:1417`) reads only the Boolean. **The rationale is parsed, round-tripped and
never enforced.**

Measured (`evidence/g11/ORCH-710/05`): the unscoped `true` excuses **seven core enzyme rows** the
rationale explicitly excludes — **LpxA, LpxD, LpxB, LpxK, WaaA, LpxL, LpxM**.

**The ruling: add per-entity tolerance scope. Do NOT flip the Boolean to `false`.** Flipping it
would penalise faithful extraction of the seven entities the rationale legitimately tolerates —
**LapA** (formerly YciS), **LapB** (formerly YciM), **Ght**, **LabP**, **LpxG**, **YhcB**, and the
generic **`lipoprotein`** — which is the opposite error, not a fix.

Required: an explicit, auditable per-entity representation, **consumed by the scorer**; stable
canonical keys or aliases, **not broad name matching**; LapA/YciS and LapB/YciM validated as
aliases without fuzziness; other gold cases keep the Boolean's valid case-wide meaning unless
evidence shows it unsafe; `forbidden_identifiers` **not** silently changed; and no expected enzyme
added or removed without separate source evidence. Chartered as **C-100**, with the base/tip gold
A/B of § 8 — a green SMOKE alone does not discharge it.

### The prose/schema mismatch, resolved explicitly rather than by picking a number

The rationale says *"the **nine** core Raetz enzymes"*. `expected_enzymes` lists **eight**: LpxA,
LpxC, LpxD, LpxB, LpxK, WaaA, LpxL, LpxM. Both are right about different things, and the gold is
**not** wrong.

The Raetz pathway has **nine enzymatic steps**. The ninth — removal of UMP — is
**organism-dependent**, and the paper says so: *"subsequent removal of UMP by LpxH, LpxI, or LpxG
depending on the organism."* No single enzyme is expected for that step in *this* paper, so the
gold correctly files **LpxH** (aliases LpxI, **LpxG**) under `acceptable_enzymes` rather than
`expected_enzymes`. Hence eight expected, nine steps.

**The rationale prose is corrected to agree with the schema** — it names the **eight expected**
core enzymes as outside the tolerance, and states that the ninth step's enzyme is
organism-dependent and deliberately `acceptable`. **`expected_enzymes` and `acceptable_enzymes` are
unchanged.** Note the consequence that makes the scoping necessary: **LpxG is simultaneously an
alias of the acceptable LpxH and a rationale-tolerated entity** — which a per-entity list can
express and a Boolean cannot.

---

## D-072 — Ruling A: coverage anchors are reconciled against `forbidden_identifiers` · 2026-08-28 · LOCKED

**Product owner ruling. Closes ask A of `DECISION-BUNDLE-F132-PRIORITY1.md` (F-132).**

A term explicitly listed in a case's `forbidden_identifiers` **must not simultaneously be required
as a positive coverage anchor for that same case.** Priorities 1 and 4/5 have been scoring the same
rows in opposite directions: Priority 1 penalises the export of a forbidden identifier while
Priorities 4/5 penalise its absence from coverage. The pipeline cannot satisfy both, and no amount
of correct behaviour clears them together.

### What the instrument must do

* **Preserve the original/raw anchor measurement** for diagnostics. It is not deleted, and it
  remains the number a reader can compare historical reports against.
* **Compute a separate contract-accepted coverage result** beside it.
* **Exclude only exact, case-scoped forbidden identifiers** from the accepted positive-coverage
  denominator. Case-scoped means case-local: an exclusion on one paper has no effect on another.
* **Continue scoring their erroneous export under Priority 1.** Removing a term from the coverage
  denominator does not make exporting it acceptable. Priority 1 must remain capable of detecting a
  forbidden export after this change, and a test must prove it.
* **Do not delete the forbidden identifiers**, and **do not erase extracted-but-withheld entities
  from diagnostics.** A coverage success obtained by dropping a diagnostic row is a reject.
* **Do not introduce bare identifiers or fabricated PWML** to improve coverage.
* **Do not globally weaken coverage** for terms that are not explicitly forbidden. A term that
  merely resembles a forbidden one stays required.
* **Report raw and contract-accepted results separately.** They may not be conflated into one
  number, and they may not be made to agree by construction.

### What this ruling is not

**This is an acceptance-instrument reconciliation. It is not a claim that the underlying pipeline
defect is fixed.** The instrument stops asking a contradictory question; the pipeline behaviour it
measures is unchanged by this card and must not be described as corrected by it.

### Sequencing — binding

Implemented as a **separately chartered card after C-101**, because C-101 and this ruling both
modify `src/t2pw/bench/semantic.py`. They are implemented, reviewed and merged **serially** even
though they touch different functions, and the Ruling-A card is developed against the **merged
C-101 integration tip**, never the pre-C-101 tree. **The two writers are not dispatched in
parallel.**

---

## D-073 — Ruling B: the Priority-1 target keeps six, and gains a one-finding variance band · 2026-08-28 · LOCKED

**Product owner ruling. Closes ask B of `DECISION-BUNDLE-F132-PRIORITY1.md`.** This is the ask that
gated T-107; the brittle "six passes, seven fails" point threshold is **withdrawn**.

### The accepted rule

| Accepted count | Status |
|---|---|
| `0`–`6` | **`PASS`** |
| `7` | **`PASS_WITHIN_VARIANCE`** |
| `8`+ | **`FAIL`** |

The **accepted count** is the contract-adjusted result after authorized, case-scoped tolerances.
The **raw count must also be preserved and reported.**

### Why a band, and what it does not license

**Six remains the target.** Seven is a **one-finding stochastic tolerance band**, and naming it
`PASS_WITHIN_VARIANCE` rather than widening the threshold to "≤ 7" is deliberate: the status says
*why* seven passes, and it survives being read alone in a report. It is **not** evidence that the
pipeline defect is fixed.

The band exists because the count is genuinely draw-variable at temperature 0. `TEST_MATRIX.md`
records that T-105's Priority 1 = 7 was composed of **almost entirely different rows** than
T-104's 7 — `succinyl-CoA`, `SREBF1/2`, `LIPA` and `LBR` vanished by draw variance and were
replaced by `protoporphyrin IX`, `NADH`, `NAD+` and `holo-EntB`. A point threshold on a quantity
that moves by composition fails on variance alone.

* `PASS_WITHIN_VARIANCE` **clears T-107 gate condition 1.**
* It **must remain visibly distinct from `PASS`** everywhere it is rendered.
* The report must give the **complete row composition and biological classifications for both the
  raw and the accepted result.**
* **Do not rerun T-107 to move from seven to six.** Do not chase a favourable draw. One official
  draw is scored and preserved.
* **A new systematic defect remains a defect even when the total stays inside the band.** The band
  absorbs variance, not regressions: a changed composition is inspected on its merits.
* **Eight or more is an actual acceptance failure** and is reported as one.

---

## D-074 — Ruling C: PMC12444477 tolerates the exact PathBank `Unknown` sentinel, and nothing else · 2026-08-28 · LOCKED

**Product owner ruling. Amends D-071. Depends on D-070 § O-1a**, which established that the bare
PathBank `Unknown` sentinel is PathBank's own legitimate representation and not a forged identity.

PMC12444477 may tolerate the confirmed bare PathBank `Unknown` sentinel. It may **not** tolerate
arbitrary rows or identifiers named `Unknown`.

### A name-only matcher is insufficient, and the seam must widen

The scorer calls `case.tolerates_unknown_backed(name)` at `src/t2pw/bench/semantic.py:1418`
passing **only the name string**. The matcher therefore cannot consult
`is_pathbank_unknown_protein(row)`, and adding `Unknown` as an eighth gold entry would excuse **any
row named `Unknown`** on that paper — precisely what this ruling forbids.

**C-101 owns the required row-aware scorer seam.** Widening the matcher input from the name alone
to the name **plus the complete candidate row** is explicitly in C-101's scope.

### The exception applies only when ALL of these hold

1. the case is **PMC12444477**;
2. the candidate name is **exactly `Unknown`**;
3. the **complete row** satisfies the existing authoritative `is_pathbank_unknown_protein(row)`
   predicate;
4. the row carries the **confirmed PathBank sentinel identity**, including the expected PathBank
   protein identity and sentinel provenance fields;
5. the candidate is the **bare sentinel** — not an arbitrary unresolved identifier, and not a
   normal protein that merely happens to be named `Unknown`.

**Do not implement this by adding `Unknown` to a name-only allowlist.**

### The authoritative A/B row must be identified before editing

Before editing, C-101's evidence record must identify the **exact authoritative pre-change
PMC12444477 row and archived leg** used by C-100's accepted A/B, recording: exact artifact path ·
run identity · paper · mode · entity bucket · complete row identity · why it satisfies the sentinel
predicate.

That **single C-100-certified row is the authoritative A/B target.** Other archived sentinel rows
may serve as preservation or adversarial controls, but they are **not interchangeable A/B targets.**

**If the accepted C-100 evidence does not uniquely identify the authoritative row, C-101 stops
before editing and returns the evidence ambiguity.** It does not choose one silently, and it does
not launch a live run to resolve archival bookkeeping.

### `LpxH` is not covered and does not move

**`LpxH` remains a Priority-1 finding.** It is the resolvable *E. coli* enzyme associated with the
organism-dependent ninth step of the Raetz pathway, and D-071 deliberately files it under
`acceptable_enzymes` rather than removing it. **Do not broaden this ruling to remove it.** The
consequence is that PMC12444477 goes **9 → 8** findings, not 9 → 7. Widening the list would be the
merge-rule-6 direction and is a reject.

---

## D-075 — Ruling D: a truthful Priority-2 `NOT EVALUATED` is not an automatic T-107 failure · 2026-08-28 · LOCKED

**Product owner ruling.** Recorded explicitly so the next session does not have to decide it
implicitly.

`DECISION-BUNDLE-F132-PRIORITY1.md` (F-132) records that Priority 2 is `NOT EVALUATED` on **11 of
20 legs** because **D-067 precondition 3** requires independent biological review that has not been
performed. That prerequisite is not reachable by engineering in this sprint.

**A truthful `NOT EVALUATED` caused solely by that unmet prerequisite is not an automatic T-107
failure.**

### For T-107

* **Score Priority 2 on every eligible leg.**
* **Retain `NOT EVALUATED` on the 11 ineligible legs.**
* **Do not manufacture or infer the missing biological review.**
* Priority 2 is **`CONDITIONALLY SATISFIED`** only if **every eligible leg passes** *and* **every
  ineligible leg carries the exact documented D-067 reason.**
* **`CONDITIONALLY SATISFIED` may clear the T-107 readiness gate.**
* The final report **must state the evaluated denominator** and **may not claim full 20-leg
  biological validation.**
* **An eligible leg that fails remains a failure.**
* **Using `NOT EVALUATED` on a leg whose prerequisites are satisfied is a failure**, not a
  convenience. Per `PRODUCT_CONTRACT` § 8, `not_evaluated` is never `false`, and the two must stay
  distinguishable in the emitted report.

---

## D-076 — C-101 ceiling 1 raised 420 → 560; the overage is the orchestrator's, and two charter errors are corrected · 2026-08-28 · LOCKED

**Ceiling 1 `420 → 560`. Ratified.** C-101 measures **541** hand-authored production + docs
(`src` 499 + `PRODUCT_CONTRACT.md` 42). Ceiling 2 unchanged at 45; the card measures **24**.

### Why this is ratified rather than split

**The overage is mine.** The original C-101 charter budgeted **250** for one deliverable — the 16/5
metric split. `AMENDMENT 1` then added **two more** deliverables of comparable size, the row-aware
sentinel seam (D-074) and the raw/accepted Priority-1 split with its variance statuses (D-073), and
I re-derived the ceiling as **420** by estimate rather than by counting the added deliverables. Three
deliverables at the first one's own rate is roughly 560. **This is the sixth ceiling under-set on
this sprint, and ceiling-1 overages have been the orchestrator's error every time** (REV-051a).

**The card behaved exactly as required.** It made a real trimming pass first — `src` **563 → 499**,
all of it comment and docstring prose, **no function and no test cut** — then **stopped and reported
rather than self-authorizing**, which is what S4 demands and what a ceiling is for.

**The diff is disciplined, which is the property the ceiling exists to protect.** Verified before
ratifying: the change is confined to `src/t2pw/bench/{acceptance,goldset,render,semantic}.py`, the
single authorized gold entry, `PRODUCT_CONTRACT.md`, a new test file and evidence. **Nothing** in
`map_ids.py`, `writer.py`, `strict_quarantine.py`, any strict gate or `streamlit_app.py`. **Nine
deletions across the entire diff.** A ceiling breach accompanied by scope creep is a reject; a
ceiling breach with a clean boundary and a documented trim is a mis-set ceiling.

**The split was offered and is declined.** The author proposed a clean three-way split
(C-101a 16/5 ≈ 230 · C-101b sentinel seam ≈ 175 · C-101c raw/accepted ≈ 130), each independently
validatable. It is a sound proposal and it is refused **on cost, not on merit**: the work is already
written, gated and coherent, so splitting buys three dispatches, three independent reviews, three
merges and three gate runs for a diff whose boundary is already clean — and it would push C-102,
C-103 and the T-107 readiness assessment out by at least a session. **Budget exists to prevent
uncontrolled scope, not to force re-work of controlled scope.**

**Ratification is not approval.** The merge remains contingent on REV-101 exactly as before. If the
review finds scope creep, that is a rejection on its own merits and this decision does not shield it.

### Two charter errors, corrected here rather than quietly

**1. `AMENDMENT 1` § A4's premise was wrong.** It instructed the card to identify the authoritative
row *"used by C-100's accepted A/B"*. **C-100's accepted A/B contains no payload row at all** — it is
a **test-node** A/B, 20 SMOKE files plus 22 gold-readers files, which is exactly where the phrase
*"zero movers on 42 files"* comes from. The card was sent looking for an object under a description
that did not fit it.

It found the right row regardless, by the better route, and that route is now the documented one:
**`evidence/orch710_pinned21.json`** (21 placeholders, exactly 5 sentinels, exactly one on
PMC12444477) **plus** the LEDGER's *"which run is 'the pinned run'"* correction (`runs/2026-08-02_2130`,
not `runs_verify/2026-08-24_1428`). Both routes select
`runs/2026-08-02_2130/papers/PMC12444477/strict/final_mapped.json` → `/entities/proteins/4`.
**Independently re-derived by the orchestrator from the artifact rather than accepted from the
report.**

The safety argument matters more than the row id: **the choice cannot bias the outcome**, because the
tolerance is row-predicated and all three archived PMC12444477 sentinel rows carry identical values
on every clause of the predicate. Had the choice been able to move a result, the ambiguity would have
been material and the card was required to stop.

**2. `AMENDMENT 1`'s header names base `b30193f`; the dispatched worktree carries `d7cf4a4`.** The
amendment was written at `b30193f` and then committed, which advanced the tip to `d7cf4a4` before the
worktree was moved to it so the card would carry its own charter. **`d7cf4a4` is the base of record**
for C-101 and for REV-101's A/B. The header line is stale and is superseded by this decision.

### One thing recovered because a card was told to

C-100's `03-base-probe` / `04-tip-probe` **stdout survived only in a dead session's scratchpad** —
the second time this wave the sole record of what a bounded job found lived outside the repository.
It is now committed at `evidence/c100_03-base-probe.RECOVERED.log`,
`c100_04-tip-probe.RECOVERED.log` and `c100_probe_stdout.RECOVERED.md`. **The probe source
(`probe_c100.py`) was already gone** and is unrecoverable; that is recorded so nobody hunts for it.

---

## D-077 — OPEN QUESTION, not yet ruled: what an authorized Priority-1 tolerance actually is · 2026-08-28 · REGISTERED

**Raised by C-101 against D-073, correctly, and registered rather than answered.**

D-073 defines the accepted count as *"the contract-adjusted result after authorized, case-scoped
tolerances"*. C-101 reports that **no such tolerance can currently remove a Priority-1 row at all**,
and gives a mechanism: `false_real` counts only forged identities; a bare PathBank sentinel carries
`uniprot: "Unknown"`, which `_external_ids` drops; therefore **a sentinel can never *be* a Priority-1
row**, and D-074's tolerance — the only case-scoped tolerance this wave authorizes — cannot subtract
from Priority 1 even in principle.

Measured consequence on `runs/2026-08-02_2130`: **raw 10, accepted 10, `accepted_status = FAIL`**.

**This does not break Ruling B.** The variance band applies to whatever the accepted count is, so
`PASS_WITHIN_VARIANCE` remains reachable at T-107 whether or not any tolerance fires. And
`accepted == raw` here is a **measurement**, not a construction — the card was required to prove the
seam can produce a difference, and to do so **without weakening the accession guard to manufacture
one on real data.** It reports that it deliberately did not, on the ground that widening the guard
would excuse a forgery. That is the correct refusal.

**What is open:** whether D-073's phrase "authorized, case-scoped tolerances" is currently vacuous for
Priority 1, and if so whether that is intended (Priority 1 is meant to be untouchable by tolerance,
and the accepted count exists only to carry the variance band) or an under-specification to be filled
later. **Routed to REV-101 as P15 for a recommendation; the ruling is the product owner's.**
**No code depends on the answer** — the seam is built, guarded and exercised either way.

---

## D-076 AMENDMENT 1 — review-mandated corrections do not count against ceiling 1 · 2026-08-28 · LOCKED

**A ceiling budgets the author's scope choices. A finding raised by an independent reviewer is not
one.**

Making a card pay for a reviewer's correction out of its authored ceiling creates exactly the wrong
incentive: it prices honest review findings against the author's remaining headroom, and the cheapest
way to absorb one is to trim a docstring or drop a test. C-101 landed at **541 of 560** after
ratification — **19 lines of headroom** against five review findings, one of which needs a new
reported bucket and its non-vacuity test.

**Rule, sprint-wide from now on:** ceiling 1 covers authored scope. **Corrections required by an
independent reviewer are budgeted separately**, as a stated correction allowance per round. The
allowance is recorded with the round so the total remains auditable, and it is **not** a licence to
add unrelated work — anything outside the reviewer's enumerated findings still counts against
ceiling 1.

**C-101 correction round 1 allowance: +60 hand-authored lines**, for the five REV-101 findings and
their tests. No test or docstring is to be cut to fit.

---

## F-143 — `bounded_run.py` resolves a bare `python` from the CHILD's PATH, not the venv

**Registered 2026-08-28** by REV-101, which **caught it in its own jobs, disclosed it, discarded the
affected legs and re-ran them.** Recorded because it silently produces a large, plausible, entirely
false regression.

**What happens.** A bounded command whose executable is the bare string `python` resolves against the
child process's `PATH`. On this machine that is `C:\Python313\python.exe`, **not**
`…/Project14-T2PW/.venv/Scripts/python.exe`. The system interpreter lacks `streamlit` and the
project's other dependencies, so collection errors on every module that imports them.

**Measured instance:** **35 spurious errors** across REV-101's first three legs. Nothing about the
failure names the interpreter — it presents as a large block of import errors, which is exactly what
a genuine dependency regression looks like.

**Why the existing guards do not catch it.** `pinned_pytest.py`'s exit-98 measurement-tree check
verifies which **tree** `t2pw` resolves inside; it does not verify which **interpreter** is running.
A job can be measuring the correct tree with the wrong Python and pass every tree check.

**The rule.** **Always pass the explicit interpreter path** —
`c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe` — as the bounded
command's executable. Never a bare `python`. This is the same class as the sprint's standing
`--basetemp` and basetemp-parent traps, both of which also present as mass errors that are not test
results, and it joins them: **an infrastructure failure, never a regression.**

**Related, and now three of a kind.** `PATHBANK_DB_*` cannot be hidden by export because
`load_dotenv(..., override=True)` re-applies `.env`; an agent worktree may have **no** `.env` at all;
and now the interpreter itself may not be the one the tree was built for. **State the interpreter,
the tree and the DB state of every measurement**, or the numbers are unattributable.

---

## D-077 ANSWERED — under D-074 as ruled, no Priority-1 row can ever be contract-adjusted · 2026-08-28 · LOCKED

**D-077 asked** whether D-073's *"authorized, case-scoped tolerances"* is currently vacuous for
Priority 1. **It is, and the reason is structural rather than incidental.** Established by REV-101,
statically and empirically, during C-101 correction round 1.

### The proof

`_contract_adjustment` has exactly **one** call site, `semantic.py:1134`, and it sits inside
`if ids:` at `semantic.py:1119` — the `false_real_identifier` branch only runs on a row that carries
external ids. **D-074 condition 5 licenses only the *bare* sentinel**, and bare means
`_external_ids(row) == {}`.

The two conditions are mutually exclusive. A row bare enough to be licensed can never reach the
branch that would need the licence.

This is not an artefact of any particular guard. Round 0's guard rejected only **UniProt-shaped**
accessions and was therefore *reachable* — a sentinel carrying `kegg`/`chebi`/`hmdb`/`drugbank`/
`pubchem` was contract-adjusted, five reachable shapes measured. **But that reachability came from
the guard being broader than the ruling**, not from the ruling permitting it. Tightening the guard to
match D-074 (correction round 1, item 3, on the orchestrator's instruction) removed the last five
shapes and made the seam unreachable for every possible input. **The tightening was correct; it
merely revealed what D-074 already implied.**

### What follows

* **`accepted ≡ raw` for Priority 1, today, for every input.** The accepted count therefore exists
  **solely to carry D-073's variance band**. That is a real function — `PASS_WITHIN_VARIANCE` remains
  reachable at T-107 whether or not any tolerance ever fires — but it is the *only* function it
  currently has.
* **The seam is not dead code.** It is a working, guarded mechanism with no licence to apply. A
  future ruling could give it one.
* **But any such ruling would be a much larger product decision than D-074 was.** To make a
  Priority-1 tolerance actually subtract, the product owner would have to license a **non-bare** row —
  that is, excuse a row that *does* carry an external accession. D-074's own condition 2 states the
  opposite principle: *a tolerance may excuse an identity a row does NOT claim, never one it does.*
  **Widening in that direction is the merge-rule-6 direction and must not be done to move a number.**
* **No code depends on the answer**, and none should be changed because of it. C-101 keeps the seam,
  the guard and the tests.

### The reporting obligation this creates

**A zero must say which kind of zero it is.** `0 because unreachable` and `0 because measured` are
different facts, and reporting the first as the second is the exact failure C-101's own
`placeholder_other_rows` and `withheld_identity_other` buckets exist to prevent. C-101 correction
round 2 corrects two statements that asserted a measurement over a structurally impossible quantity —
`_contract_adjustment`'s docstring and `render.py`'s `(none -- measured, not assumed)` — and pins the
fact with an end-to-end test that no row shape yields a non-empty tolerance under the current gold.

**The instrument's own standard, applied to itself.** That is the disposition, and it is the reason
this was worth a correction round rather than a follow-on ticket.

### One method note, because it generalises

The test that should have caught this — `test_a5_bare_means_bare_a_sentinel_with_any_accession_is_not_adjusted` —
**never called the function it is named for.** It asserted that the predicate matches and that
`_external_ids` survives, then *inferred* the conjunction. It passed, it was not vacuous, and it did
not test its own name.

This is the third instance this sprint of the same class, and the sharpest: **a control that proves
the instrument can report non-zero does not prove it is asking the right question.** Where a null
result is load-bearing, the assertion must run **the production path**, not a reconstruction of it.

---

## D-078 — a third C-101 correction round, authorized explicitly rather than automatically · 2026-08-28 · LOCKED

**The standing allowance is two automatic correction rounds per card.** Both are spent. This
authorizes a third as an **integration-authority decision with reasons**, which is a different thing
from an automatic round and is recorded so it does not become precedent by accident.

### Why a third round rather than merge-and-follow-on

REV-101's final sign-off returned **HOLD**, and was explicit that this is not a reject: nothing is
out of boundary, no invariant is violated, **no biological gate is weakened**, no product decision was
improvised, every change is conservative, and **the shipped instrument is now truthful about its
zero** — the stated bar. It offered merge-now-with-a-follow-on as defensible.

I declined it for two reasons.

**1. The card's own thesis forbids it.** C-101 exists to make an instrument stop reporting a
structural fact as a measured one — that is what `placeholder_other_rows`, `withheld_identity_other`
and the corrected `_contract_adjustment` docstring are all for. **Shipping a test that reports an
unexercised path as a pinned one is the same error in the test layer.** A card cannot land that
argument in production and violate it in its own guards.

**2. The practical stake is real, not aesthetic.** As it stands the bareness guard — the guard
D-077's entire answer rests on, and the one that keeps a tolerance from excusing a row that *does*
claim an identity — **can be deleted with every gate staying green.** That is an unprotected
safety-adjacent guard, not a documentation nit.

### The finding — a non-vacuity guard that guards the wrong emptiness

Both tests added in round 2 to *guarantee* the honesty are vacuous, and REV-101 established it by
**mutation**, not by reading:

* `test_a5_bare_means_bare…` reaches `validate_semantic_coverage` but asserts
  `all(t == "" for t in tolerances(row))` where `tolerances(row)` is `[]` — `all([])` is vacuously
  true. The row is named `Unknown`, which is **not** in PMC12444477's `forbidden_identifiers`, so
  `_check_id_conflicts` never enters the `false_real_identifier` branch and the scorer emits nothing.
* `test_a5_no_row_shape_can_be_contract_adjusted_under_the_current_gold` has `seen == 1`, not 7, and
  that one finding comes from the `placeholder_claims_real_identity` branch, whose
  `contract_tolerance` is a **hard-coded `""` literal** (`semantic.py:1171`) and which never calls
  `_contract_adjustment`. Across all seven shapes the seam's only call site is **never reached**.

**The mutation result is the whole finding:** reverting the guard to round 0's `_REAL_ACCESSION`
form — proven reachable for five namespaces — leaves all three tests **passing**; deleting the
bareness guard entirely leaves the focused file at **38 passed**. The test's own docstring claims
*"whoever widens the licence later must come here and change this assertion on purpose."* The
reviewer widened it maximally and nobody had to.

**Registered as F-144.** The class: *a non-vacuity guard can be real and still guard the wrong
emptiness.* Asserting that **a** finding was produced is not evidence that **the path under test**
produced it. Where a null result is load-bearing, the assertion must (a) run the production path and
(b) require a finding **of the specific kind** that path emits — and (c) be attacked by mutation
before it is believed.

This is the fourth instance this sprint of the same family, and the most refined. Its ancestors:
a guard demonstrated against a case that could not exercise it; a probe that passed its positive
control while asking the wrong question (case-sensitive `\bLpp\b` against a lowercase token); and a
test named for a behaviour it never called. **The lesson has now graduated from "write a control" to
"the control must exercise the same predicate, on the same path, and survive a mutation."**

### Terms of the round

Test-only fix; **no production change is asked for and none is authorized.** Budget **+40**
hand-authored lines, more on request rather than by compression; ceiling 1 stays at **541** because a
review finding is not the author's scope (D-076 Amendment 1). **The author must prove the fix by
mutation before reporting** — revert the guard, confirm red; delete the guard, confirm red; restore
and verify the tree clean. A non-vacuity claim that has not been attacked is precisely what this
round exists to correct.

**If a fourth round proves necessary, C-101 is carried to the next session rather than merged.**

---

## D-079 — C-102 ceiling 1 raised 300 → 400, and the F-050 budget command is corrected a second time · 2026-08-28 · LOCKED

### The ratification

**Ceiling 1 `300 → 400`. Ratified.** C-102 measures **391** hand-authored production + docs
(`src` 274 · docs 117). Ceiling 2 unchanged at 40; the card measures **28**.

**Seventh under-set ceiling of this sprint, and the seventh that is the orchestrator's error.** The
pattern is now beyond doubt: I size ceilings by estimate against a deliverable list I wrote, and the
estimate is consistently low. C-102's charter mandated the reconciliation, three guard rails stated
*in code*, eleven tests, an offline A/B over the whole F-132 population, a behavioural G9 proof, the
`PRODUCT_CONTRACT` § 7 denominator rule, and documentation in `LEDGER.md` and `TEST_MATRIX.md` —
the last of which it made a **merge precondition** by requiring an *"exact documented delta"*. 117
doc lines discharge that. Production at 274 sits in `_build_denominators`' module, which runs at
roughly one comment line per code line; **matching the surrounding density is correct** and shipping
an unexplained scoring seam would not be.

**The card stopped and asked.** It did not self-authorize, and it explicitly did not cut a test or a
docstring to fit — which is the behaviour D-076 Amendment 1 exists to protect. **Boundary is clean**
and was verified before ratifying: `bench/acceptance.py`, `bench/render.py`, one new test file, three
docs. No `strict_quarantine.py`, no gold file, no `map_ids.py`, no `writer.py`, no `streamlit_app.py`.
**A ceiling breach with a clean boundary and no cut tests is a mis-set ceiling, not scope creep.**

### The instrument defect — F-050's command is still wrong, in a second way

C-102 found it and it is real: **the corrected F-050 command and the ceiling sentence it serves
disagree, and they cannot both be right.**

```
# F-050's corrected command
git diff --numstat <base> HEAD -- src tests 'docs/pwml_recovery_sprint/evidence/*.py' | awk ...
```

```
# the charter template's ceiling sentence
"≤ N hand-authored added+deleted lines across production + docs, PLUS THE TEST FILE."
```

The command **includes `tests`**; the sentence **excludes the test file**. It also folds in
`evidence/*.py` probes, which charters likewise budget separately. On C-102 the literal command
reports **1171** against a ceiling meaning **391** — a factor of three, in the same direction and for
the same reason F-050 was raised in the first place. **F-050 fixed the generated-JSON leak and left
this one.**

A card trusting the literal number would appear catastrophically over budget and might cut real work
to "fix" it — **the exact failure D-025 forbids, which is what F-050 said when it corrected the
command the first time.**

**Ruling — ceiling 1 is production plus docs, and nothing else:**

```
git diff --numstat <base> HEAD -- src 'docs/**/*.md' \
  | awk -F'\t' '{s+=$1+$2} END {print s+0}'
```

* **the card's own test file** is reported separately and is **not** budgeted — writing more tests
  must never cost a card its headroom;
* **hand-written evidence probes** under `evidence/*.py` are reported separately;
* **generated artifacts** remain ceiling 2.

This is the command charters quote from now on. **Charters already dispatched carry an older form;
when ratifying an overage, re-measure with this one before concluding anything.** Both C-101's 541
and C-102's 391 were measured on this basis and stand.

### The standing rule this makes explicit

**Three of the last three cards have breached ceiling 1, and in all three the breach was mine.**
Ceilings are to be derived by **counting the deliverables the charter mandates**, at the measured
density of the module being edited — not estimated. A charter that mandates an A/B, a G9 proof,
mutation attacks, three documentation surfaces and eleven tests is not a 250-line card, and saying so
at dispatch is cheaper than ratifying afterwards.

---

## D-080 — D-072's "denominator" means numerator AND denominator · 2026-08-28 · INTERPRETATION, pending product-owner ratification

**This is an interpretation of a locked ruling, recorded by the Lead Orchestrator, not a new ruling.**
It is flagged for the product owner to ratify or overturn. It is recorded here because REV-102 was
right that without it the C-102 diff reads as an improvised product decision to the next reader, and
it is not one.

### The tension

**D-072 says:** *"exclude only exact case-scoped forbidden identifiers from the accepted
positive-coverage **denominator**."*

**C-102 excludes them from the numerator as well**, and `PRODUCT_CONTRACT.md` § 7 now says
*"withheld from the accepted numerator and the accepted denominator alike."* **D-072 outranks
`PRODUCT_CONTRACT`.** Two authorities disagree in writing, and this resolves it.

### Why the literal reading cannot be what was meant — measured, not argued

Coverage is `matched / total`. Removing a **matched** forbidden term from the denominator alone
leaves it in the numerator. REV-102 measured the consequence over the real committed corpus:

| paper | mode | raw | denominator-only | both-sides |
|---|---|---:|---:|---:|
| PMC12856317 | strict | 1.0000 | **1.2000** | 1.0000 |
| PMC12856317 | research | 0.8571 | **1.2000** | 1.0000 |
| PMC13231680 | strict | 1.0000 | **1.2000** | 1.0000 |
| PMC12312563 | strict | 0.7000 | **1.0000** | 0.8571 |

**Eight real legs report a "coverage ratio" of 1.2000 under the literal reading.** A ratio above 1 is
not a rate; the instrument stops being a measurement at all.

Worse, synthetically on `PMC12782028`, whose four forbidden drawn terms are the same rows Priority 1
punishes:

```
matched = [LBR, LIPA, SREBF1]   raw 0.5000   denominator-only 1.0000   both-sides 0.0
matched = [HMGCR]               raw 0.1667   denominator-only 0.3333   both-sides 0.3333
```

**A pipeline that exported all three forbidden identifiers and matched no legitimate anchor scores
perfect coverage**, and matching a forbidden identifier is worth *exactly as much* as matching a
legitimate one. **That is F-132 with its sign flipped, at full amplitude** — the ruling's own stated
purpose defeated by its own literal text.

### Both-sides removal introduces no counter-perversity

Exhaustively checked by REV-102 over all 64 match-subsets of 3 forbidden + 3 legitimate anchors on a
real gold case: **0 violations.**

* un-matching a legitimate anchor **never** raises the accepted ratio;
* toggling a forbidden match is **exactly neutral** — the incentive to export a forbidden term is
  precisely **zero**, which is what D-072 requires;
* `accepted_matched <= accepted_denominator` always, so the ratio stays bounded by 1.

### The interpretation

**"Excluded from the accepted positive-coverage denominator" means the term is removed from the
accepted measurement entirely — numerator and denominator alike — so that exporting it and
withholding it score identically, and Priority 1 remains the only instrument that judges it.**

The ruling's intent — *"Priorities 1 and 4/5 stop scoring the same rows in opposite directions"* — is
served **only** by both-sides removal. Denominator-only removal does not merely fail to fix the
contradiction; it inverts it and rewards the violation.

### What this does not license

* **No other clause of D-072 is reinterpreted.** Exclusion stays *exact* and *case-scoped*; forbidden
  identifiers are still scored under Priority 1; nothing is deleted from diagnostics; the threshold
  does not move.
* **This does not make the coverage number better.** Zero legs clear the minimum under either
  reading. See D-081.
* **If the product owner intended the literal reading, this is overturnable** and C-102's exclusion
  becomes a one-line change plus its tests. The measurement above is the case for not doing so.

---

## D-081 — Ruling A does not move Priority 4 or Priority 5. The bundle predicted it would · 2026-08-28 · MEASURED

`DECISION-BUNDLE-F132-PRIORITY1.md` § 6 predicted that under Option A *"Priority 4 becomes
meaningful … expect it to move off 0/8; by how much is not predicted here."*

**It does not move.** Measured by C-102 and independently reproduced by REV-102 across **all 21 run
directories, at base and at tip**:

| | base | tip |
|---|---|---|
| Priority 4 | `0/8 = 0%` | `0/8 = 0%` |
| Priority 5 | `0/2 = 0%` | `0/2 = 0%` |
| legs cleared by the reconciliation | — | **`[]` on every run** |

`PMC12782028/strict` moves `0.2222 → 0.2609` and still fails `0.500`. **Every leg below the minimum
on raw is still below on accepted.**

**Why the bundle was wrong.** Priorities 4 and 5 take their numerators from **semantic confirmation**
and the **frozen strict release record** — not from the requested-core coverage ratio. Reconciling the
denominator changes what the coverage figure *means* without changing either numerator.

**What Ruling A actually bought, and it is real:** the coverage measurement is now **readable per
leg** — raw beside accepted, with every withheld term named alongside the gold entry that excused it,
across 92 terms on 47 legs. Before, every coverage figure on those legs was computed over a term list
that included identifiers the same gold forbids exporting. **The instrument stopped contradicting
itself.** It did not, and could not, move a priority whose numerator it does not feed.

**This goes in the T-107 readiness table as a stated result, not a footnote.** The product owner was
led to expect a different outcome, and the honest record is that the prediction was falsified by
measurement.

---

---

## D-082 — C-102 round-1 ceilings ratified, and ceiling 2 stops being charged for review-mandated re-gates · 2026-08-28 · LOCKED

### The two ratifications

**Correction-round allowance `120 → 140`. Ratified.** C-102 round 1 measures **134**; it was at
exactly 120 after the fixes, and the overage is the LEDGER gate-record paragraph plus two repair
commits. Cumulative ceiling 1 reads **499** — both rounds together, and *less* than 391 + 134,
because round 1 rewrote doc lines round 0 added.

**Ceiling 2 `40 → 55`. Ratified.** C-102 measures **53**: 24 G11 reports + 6 pin verdicts + 23 logs.

### Ceiling 2 was structurally wrong, and D-048 already said why

**Every bounded job produces two artifacts, not one** — a report *and*, where pinned, a verdict. I
mandated a re-gate of roughly ten jobs (focused, SMOKE, gold-readers combined **and** split, the
seven-mutation attack, the byte-size probe, the numerator verification). Three of the logs are
**kept failures**, which this sprint requires rather than permits.

**D-048 recorded this exact coupling** — *"a ceiling-2 derivation must count one report **and** one
pin verdict per measured job"* — and I set 40 without applying it. The card was therefore over the
moment I wrote the correction round, before it did anything.

**The card flagged rather than assumed**, correctly noting that ceiling 2 *"was not re-stated for the
round"*. That is the right reading of an ambiguous instruction and it is the behaviour to want.

### The rule, extended

**D-076 Amendment 1 now covers ceiling 2 as well.** Review-mandated work is not the author's scope
choice, and that applies to the **artifacts a mandated re-gate necessarily emits** exactly as it
applies to the lines a mandated fix necessarily costs. A card must never face a choice between
running the gate I required and staying inside a ceiling I set before requiring it.

**Concretely:** artifacts produced by a re-gate a reviewer or the orchestrator demanded are reported
separately and are **not** charged to ceiling 2. Artifacts a card generates for its own purposes
still are. **Kept failed runs are never charged** — retaining them is mandatory, and charging for a
mandatory artifact is a tax on honesty.

### One number of mine, corrected by the card

D-080 states that **eight** committed legs report a coverage ratio above 1 under D-072's literal
denominator-only reading. C-102 re-derived rather than copying and reports **nine**: the eight at
exactly `1.2000` (all `6/5`) **plus a ninth at `1.125` (`9/8`)**. Under independent verification by
REV-102; **D-080 is corrected below once that confirms.** The argument is unchanged and slightly
strengthened — a ratio of 1.125 is no more a rate than 1.2 is.

**This is the second time this wave a card has corrected a number I wrote into a decision record**,
after C-101 corrected A4's premise. Both times the card re-derived instead of copying. That is
exactly what a number in a charter is for.
