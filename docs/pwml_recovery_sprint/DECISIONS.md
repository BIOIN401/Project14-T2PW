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
