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
   checkout, **exit 0**.
3. **The failure can occur WITH C-011** — reports `04`, `25`, `26`, `44`, `45`, and three
   independent reviewer runs.
4. **The failure can disappear WITH C-011** — reports `05` and `18`, **exit 0**.
5. **The traceback contains no C-011 frame.** `streamlit_app.py:6187` →
   `tools/pathwhiz_converter/ui.py:26` (`st.subheader`) → `script_run_context.py:144` →
   `RuntimeError: FragmentThreadState not initialized`. The abort is **2,585 lines past the seam**,
   in a module C-011 never touches. The repository already adjudicates this shape at
   `tests/test_streamlit_quarantine_boundary.py:425-430` as "a Streamlit harness fault, not an app
   fault".
6. **The two affected AppTest files pass 27/27 when isolated**, exit 0, at that exact tip.

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

**The authoritative Chunk D gate going forward is the split-process Chunk D gate defined in
`TEST_MATRIX.md`.** That gate — not this exception — is what future cards run against; this
decision governs one historical result and does not describe, constrain or substitute for it.

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

**C-040 — extract, do not call.** Lift `_resolve_compound_rows` and
`_canonicalize_compound_offline` out of `ir.py` into a new module and give them the three-part
SPIKE-002 adapter contract. Mechanical extraction under the `LIFT_WITH_ADAPTER` verdict. C-040
**wires nothing** and changes no caller's behaviour.

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
| `ir.py :: _resolve_compound_rows` | **C-040** | C-050, C-051 |
| `ir.py :: _canonicalize_compound_offline` | **C-040** | C-050, C-051 |
| The `apply_canonical_name` keyword-only parameter on both, and the resolution-report shape | **C-040** | C-050, C-051 |
| `ir.py :: build_pwml_ir`, **including the resolution call site at `:1106-1114`** | **C-051** | C-040, C-050 |
| `streamlit_app.py` :: the enrichment block **above** the C-011 seam, and the pre-freeze call | **C-050** | C-040, C-051, C-052 |
| `ir.py :: _canonicalize_species_offline` | **C-045** (D-016, unchanged) | C-040, C-050, C-051 |
| `streamlit_app.py :: freeze_canonical_payload` | **C-030**, then **C-052** (D-021 does not move it) | C-040, C-050, C-051 |
| `run_pwml_export` and the SBML binding | **C-052** | C-040, C-050, C-051 |
| `rag/eligibility.py` organism helpers (`:1366-1404`) | **no card** — `MASTER_PLAN` §2 read-only (`:147`) | all three |

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
artifacts are prohibited unless separately authorized. **C-040 must be pre-split**
(`SPIKE-002-REPORT.md` §4), and per D-019 a split is permitted **only** at a boundary where each
half is independently implementable and independently validatable — a budget may never force an
unvalidated semantic half to merge or to be left behind. Dispatching any of the three without its
declared manifest and both budgets is a dispatch error.

### 8. Nothing frozen

No responsibility in this partition required choosing between competing product semantics the
repository has not settled. Every boundary above is taken from an existing card purpose, an
existing `MASTER_PLAN` §9 row, or a locked decision (D-015, D-016, D-019). **No sub-decision is
frozen.** Two items are **routed, not decided**: the T-102/C-045 scheduling consequence in §2, and
the unassigned retry ownership for `stoich/agent.py` and `rag/embed.py` (`LEDGER.md` A0-C2), which
lies outside this trio entirely.

---

## Open — not yet decided

| # | Question | Blocks | Why it cannot be answered from the repository |
|---|---|---|---|
| O-1 | `placeholder_backed_proteins` (21 in the pinned run): gold-set error class, or legitimate biology preservation? | any branch that touches protein export policy | It is a genuine disagreement between two intentional designs, not a defect. TRAP-3 forbids agents from resolving it. |

**Closed:** O-2 → D-011 · O-3 → D-014.
