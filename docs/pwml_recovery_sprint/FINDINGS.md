# Deferred findings

Defects, shortcuts and debt a card **found but did not own**. Recorded so they are neither
lost nor silently absorbed into an unrelated card. Nothing here authorizes a code change.

**Process.** Workers **report findings to the orchestrator**; workers do **not** edit this
shared file. Findings accumulate across a pack, and this file is updated **once at pack
closeout** — not after every card.

**Entry format.** ID · severity · discovering card · exact path/symbol · observed vs
expected · minimal reproducer or evidence · current-card impact · future owner or
`unowned` · shortcut deliberately accepted, where applicable.

---

## F-001 — evidence-haystack ordering is not mutation-pinned

- **Severity** low
- **Discovered by** H-008
- **Path/symbol** `src/t2pw/rag/graph_delta.py` — the sorted evidence-haystack ordering
- **Observed vs expected** The sorted ordering is implemented, but nothing pins it, so a
  mutation that reorders the haystack would not be caught.
- **Evidence** No test asserts the sort order; H-008's tests exercise retention, not order.
- **Current-card impact** none
- **Owner** `unowned`
- **Escalation** Blocking only if a later card (C-043) exposes an actual deterministic
  correctness failure. **C-043 merged at `3c04d4b` without exposing one.**

## F-002 — comment overstates scalar writer coverage

- **Severity** low · documentation only
- **Discovered by** H-008
- **Path/symbol** `src/t2pw/rag/graph_delta.py:99-100`
- **Observed vs expected** The comment claims wider scalar writer coverage than the code
  implements.
- **Evidence** the two cited lines
- **Current-card impact** none
- **Owner** `unowned`

---

# Pack 1 findings · closed 2026-08-13 · integration `fbedfe6`

## F-003 — a worktree pytest run silently tests the PRIMARY checkout

- **Severity** **HIGH** · infrastructure · affects every future card
- **Discovered by** C-043, corroborated independently by C-030, C-038 and four reviewers
- **Path/symbol** `.venv/Lib/site-packages/__editable__.t2pw-0.1.0.pth` · `pytest.ini` ·
  absence of any `conftest.py`
- **Observed vs expected** The editable `.pth` hard-codes
  `C:\Users\Angad\Desktop\SummerBIOIN\Project14-T2PW\src`; `pytest.ini` sets **no**
  `pythonpath`; there is **no `conftest.py` anywhere in the repo**. A pytest run launched from
  a worktree **without** `PYTHONPATH` imports `t2pw` from the **primary** checkout. A card that
  adds a new module fails loudly on `ImportError`; **a card that modifies existing files fails
  silently** — its tests pass against unmodified primary code and prove nothing.
- **Evidence** `cd .claude/worktrees/<any>; python -m pytest tests/<worktree-only-test>.py` →
  `ImportError: cannot import name 'controller' from 't2pw.rag'
  (C:\...\Project14-T2PW\src\t2pw\rag\__init__.py)`. C-038's reviewer additionally measured the
  primary checkout's `pipeline.py` blob as **identical to `e4eeef4`**, i.e. an unpinned run
  silently tests an earlier base.
- **Aggravating factor** `tests/test_completeness_audit.py` is **inside the pinned smoke gate**
  and performs no `sys.path.insert`, importing `t2pw` directly at `:13`. It is 10th in the smoke
  list, so **smoke's correctness currently rests on test-file ordering.**
- **Current-card impact** none — every Pack 1 writer and reviewer pinned `PYTHONPATH` and
  certified `t2pw.__file__` **inside the pytest process**. All merge gates were run from the
  primary checkout with the merge staged, so they resolve correctly by construction.
- **Owner** **H-010** — closed for every run through the measured launcher
  (`evidence/pinned_pytest.py` + `evidence/tree_pin.py`, `TEST_MATRIX.md` § 0 rule 10): the
  resolved `t2pw.__file__` is compared against `<measured>/src/t2pw` and a mismatch
  **refuses with exit 98 before collection**, writing a committed `*.pin.json` verdict. The
  proposed remedy — `pythonpath = src` in `pytest.ini` — was **considered and REFUSED**:
  pytest *prepends* those entries, so it would sit ahead of the `PYTHONPATH` pin and make
  every base-tree G9 proof silently measure the tip, the same defect class aimed at the
  proofs. A `conftest.py` was likewise not adopted. **Residual, unowned:** commands not yet
  on the launcher (SMOKE, Chunk E, `baseline_suite.py`, the 15 `add_src_to_path()` probes)
  stay masked by file ordering; `chunk_d_gate.py` adoption is a follow-up card.

## F-004 — G11 reports record no environment

- **Severity** medium · evidence integrity
- **Discovered by** C-043 / C-032 / C-035 reviewers
- **Path/symbol** `docs/pwml_recovery_sprint/evidence/bounded_run.py` report schema
- **Observed vs expected** The report captures `cwd` and `command` verbatim but **not the
  environment**, so a `PYTHONPATH` pinning claim is not self-evidencing from the artifact.
  Reviewers had to re-derive every pinning claim independently.
- **Current-card impact** none — reviewers re-derived
- **Owner** **H-010, worked around — `bounded_run.py` untouched.** No ownership grant was
  sought: the measured launcher writes its **own** `*.pin.json` verdict carrying the
  resolved `t2pw.__file__`, `sys.path[:8]`, cwd, `PYTHONPATH` and the violation list —
  better evidence than a wrapper field, since the process that resolved the imports
  produced it rather than its parent.
  The underlying schema gap stays **open and unowned**: a job that does not run through the
  launcher still records no environment at all.

## F-005 — chunk E tripwire will fire on the next committed run directory

- **Severity** **HIGH** · forward-dated
- **Discovered by** C-031 reviewer
- **Path/symbol** `tests/test_strict_quarantine_real_artifact_replay.py:581
  test_no_unlisted_artifact_quietly_carries_stage_zero_context`
- **Observed vs expected** The test fails any `*.json` under `runs/` that is on neither
  `_PAYLOAD_FILENAMES` nor `_STAGE_ZERO_DIAGNOSTIC_FILENAMES`, is ≤ 2,000,000 B, and contains
  `key_compounds`. Measured against a production-shaped `pathway_context`,
  **`quarantine_report.json` and `coverage_summary.json` both contain `key_compounds`.**
- **Evidence** Chunk E is green today — 0 unlisted marker-carriers across the 321 committed
  `runs/` JSON files, re-verified at C-031's merge — **because C-031 commits no run directory.**
- **Current-card impact** none
- **Owner** whoever commits the **first real run directory after `cbf30f6`**. Fix per the test's
  own message: classify both filenames into `_STAGE_ZERO_DIAGNOSTIC_FILENAMES` with a reason.

## F-006 — D-002 is not satisfied end-to-end on the production path

- **Severity** **MEDIUM** · no owner exists
- **Discovered by** C-041, confirmed by its reviewer
- **Path/symbol** `src/t2pw/pipeline/strict_quarantine.py:1959`
- **Observed vs expected** `not coverage.get("minimum_core_satisfied")` still becomes a
  `refusal_reason`, so **PMC12096016 at 0.167 coverage is still blocked** rather than exported as
  `review_required` PWML. D-002 requires the threshold to block release-ready status, **not PWML
  production**.
- **Evidence** C-041 supplies the classification that says otherwise; the blocking site is
  outside its ownership.
- **Current-card impact** none — C-041 delivered its chartered contract
- **Owner** **`C-041a` — assigned by product-owner ruling 2026-08-13.** A narrow follow-up card,
  not a reopening of C-041. Its production ownership is limited to the release/refusal seam
  around `strict_quarantine.py :: quarantine_and_close`, including the live refusal near the
  previously reported `:1959`. Purpose: complete D-002 end-to-end so a subthreshold but
  structurally valid pathway becomes **`review_required` output rather than an unconditional
  refusal**, preserving the product contract, with focused behavioural tests required.
  *(Superseded reading, kept for audit: "none — needs a product-owner ruling on which card
  closes it." `MASTER_PLAN` §9's C-041 ownership row and the dispatch instruction both excluded
  `quarantine_and_close`; only `:230`'s prose rationale assumed otherwise, so editing it would
  have been an `out_of_boundary` REJECT at the time.)*

## F-007 — `_apply_single_op` aliases the caller's patch ops

- **Severity** **MEDIUM**
- **Discovered by** C-036 writer, quantified by its reviewer
- **Path/symbol** `src/t2pw/curation/apply_audit_patch.py :: _apply_single_op`
- **Observed vs expected** `op["value"]` is installed **without copying**, so C-036's
  `_record_audit_lineage` writes *through* into the caller's patch ops.
- **Evidence** A genuine rollback yields `rolled_back: True`, `applied_count: 0`, returned
  payload clean — but **`caller PATCH OPS mutated: True`**, and the persisted
  `rejected_patch_log.json` then contains, inside a `"rolled_back": true` entry, a lineage
  record reading *"an accepted audit patch 'add' operation produced this row's content"* for
  content that **never committed**. Separately,
  `tests/test_apply_audit_patch_lock_policy.py:160` passes **only** because both sides are the
  same object — a latent third failure for whoever fixes the aliasing.
- **Current-card impact** none — no payload row is falsely attributed
- **Owner** `unowned`. The only in-boundary fix is copying in `_apply_single_op`, which is a
  behaviour change outside "lineage writes only".

## F-008 — A0-C7 remains open; the freeze seam is incomplete

- **Severity** **MEDIUM**
- **Discovered by** C-030 rev1, proven by C-030 rev2 and its reviewer
- **Path/symbol** `src/t2pw/app/streamlit_app.py` — at base `e4eeef4`, lines **`:3658`,
  `:3668`, `:3710`, `:3713`** inside `run_post_pipeline_sbml_artifacts`
- **Observed vs expected** A0-C7 is bound to C-030, but is **undischargeable inside
  `freeze_canonical_payload`**: the seam's canonical payload is a `deepcopy` that **nothing in
  the returned dict aliases**, so there is no sharing inside the boundary to break.
- **Evidence** The documented share→copy mutant is indistinguishable from the tip on **all nine
  seam observables across all 39 legs**. The four real sharing sites are C-052's symbol.
- **Current-card impact** C-030 merged with A0-C7 **descoped, not re-bound**. Per `LEDGER.md`,
  *"a seam is not complete while its row is open"* — **the freeze seam is recorded incomplete
  pending C-052.**
- **Owner** **C-052**
- **Shortcut accepted** Deliberately. No vacuous assertion was written in its place.

## F-009 — SBML carries no taxonomy annotation, so T-102's organism axis may be unreachable

- **Severity** **MEDIUM** · needs a product decision
- **Discovered by** C-045 planning
- **Path/symbol** `src/t2pw/sbml/` (all 7 modules) · `pipeline/canonical.py:624-626`
- **Observed vs expected** `src/t2pw/sbml/` contains **zero** occurrences of `taxonom`. C-020's
  comparator calls the absence *"a real loss to REPORT"*, and a `taxonomy_ids` mismatch is a
  `Difference` → verdict `not_equivalent`.
- **Consequence** **T-102's organism axis appears unreachable however completely C-045
  succeeds**, and the sprint-wide SBML prohibition bars any card from adding the annotation.
- **Current-card impact** none — C-045 is planning-only and undispatched
- **Owner** **product owner.** Related: species are not an entity kind in C-020's `_KINDS`
  projection (`canonical.py:226`), so T-102 cannot detect a species `pathwhiz_id` fabrication
  via the comparator — that must be checked against the graph-hash projection (A0-C1).

## F-010 — `ir.py :: _component_record` is the actual cause of the `mapping_meta` loss

- **Severity** **MEDIUM**
- **Discovered by** C-045 planning
- **Path/symbol** `src/t2pw/pwml/ir.py :: _component_record` (`:414-434`)
- **Observed vs expected** It never calls `_copy_common_entity_fields`, and is the real cause of
  the measured `mapping_meta` / `taxonomy_backfill` loss recorded in `MASTER_PLAN` §1.2 — **not**
  `_canonicalize_species_offline`, to which the loss had been attributed.
- **Current-card impact** none
- **Owner** **none.** Owned by no card and named in no control-plane document.

## F-011 — the identity ladder is now duplicated across two modules

- **Severity** **MEDIUM**
- **Discovered by** C-030 reviewer
- **Path/symbol** `src/t2pw/pwml/compound_resolution.py` (`:101`, `:122`, `:213`, `:356`) vs
  `src/t2pw/pwml/ir.py`
- **Observed vs expected** C-040 copied `_first_nonempty`, `_db_id` and the compound key lists
  into the new module. C-030's `graph_projection` **mirrors** the ordered key lists that now live
  in two places, and its census oracle reads **only** `ir.py`'s copy.
- **Evidence** Verified byte-identical at merge time. A later divergence in the compound path
  would not be caught by C-030's allowlist-completeness test.
- **Current-card impact** none
- **Owner** `unowned`. Closing it properly would mean importing `ir.py` into `canonical_hash.py`,
  inverting the exporter/hash dependency C-030 exists to protect.

## F-012 — `bench/semantic.py :: _check_rag_reintroduction` is defective in both directions

- **Severity** **MEDIUM**
- **Discovered by** R-004, with the fail-open direction found independently by R-003
- **Path/symbol** `src/t2pw/bench/semantic.py:936-987` and its `_claim_key` at `:921-933`
- **Observed vs expected** `_claim_key` omits `gap_id`, while the admission gate's decision unit
  is `RagReactionCandidate.merge_key() = (gap_id, claim_identity())`
  (`src/t2pw/rag/admission.py:1524-1525`). It also indexes **only** `admission["rejected"]`,
  never `accepted`, and was built from **200 of 837** rejected rows (`truncated.rejected = 637`).
- **Evidence** **False positive:** all three PMC12657337/research "reintroduced" reactions were
  in fact **ACCEPTED** under the exact `gap_id`s the exported rows carry; re-running the check's
  own `_claim_key` gives 5 accepted keys, 10 rejected keys, **overlap 5 of 5** — the check cannot
  pass on that leg while any RAG reaction survives. **Fail-open:** the same check returned
  `ok=true` for PMC12180156/strict against 14 of 239 rejected claims while a rejected span sat in
  the payload **3× verbatim**.
- **Current-card impact** none — outside every Pack 1 boundary
- **Owner** `unowned`. R-004 proposes a new card **C-062**. **Do not read this check's output as
  ground truth about the gate.**

## F-013 — `ReleaseStatus` has no `__post_init__`, so TRAP-1 lives only in the factory

- **Severity** **MEDIUM**
- **Discovered by** C-041 reviewer
- **Path/symbol** `src/t2pw/pipeline/release_status.py:182-203`
- **Observed vs expected** The invariant `strict_acceptance_eligible == (status ==
  release_ready)` is enforced at the single construction site inside `classify_release_status`,
  but `ReleaseStatus` is a public frozen dataclass with **no** `__post_init__`.
- **Evidence** Both `ReleaseStatus(status=REVIEW_REQUIRED, …, strict_acceptance_eligible=True)`
  and `dataclasses.replace(good, status=REVIEW_REQUIRED)` produce a strict-eligible
  `review_required`. Nothing does this today. TRAP-1 was independently verified to hold across
  **528** input combinations through the factory.
- **Current-card impact** none
- **Owner** **C-053 / C-054 / C-056b**, which build directly on this type.

## F-014 — the ledger's A0-C1 census figure does not reproduce

- **Severity** **MEDIUM** · control plane · **corrected by this pack**
- **Discovered by** C-030 rev1, re-derived independently by C-030 rev2 and its reviewer
- **Path/symbol** `LEDGER.md` § Carried Wave A0 requirements — A0-C1's *"60 committed rows"*
- **Observed vs expected** Three independent censuses over all 32 committed `final_mapped.json`
  measure **49** gap rows, not 60 — compounds 38, protein_complexes 11; keys
  `pathbank_compound_id` 38, `pathbank_complex_id` 11; **all at tier 4** (`candidates[0]`), 0 at
  tier 2, 0 at tier 1/3 on a non-allowlisted key; 19 distinct files.
- **Evidence** Corpus drift excluded: `git log 09fb40d..e4eeef4 -- '*final_mapped.json'` is empty
  and the corpus is 32 files at both SHAs. The nearest 60 in the corpus is the **file count of
  `merged_payload.json`**, the most plausible transcription source.
- **Current-card impact** C-030 discharged A0-C1 against **all 49** measured rows.
- **Owner** product owner, to ratify the figure in `LEDGER.md`.

## F-015 — orchestrator budget sizing was wrong on three of twelve cards

- **Severity** medium · process
- **Discovered by** the Lead Orchestrator, against itself
- **Observed vs expected** C-030 750→1200, C-033 850→1300, C-032 950→1100. **Every one** was a
  card whose specification is a **multi-clause locked decision** — A0-C1's census plus mutation
  proofs; D-003's eight clauses; D-005's ~ten clauses. All three were caught by **writers
  stopping**, not by review.
- **Rule for future charters** Size the acceptance work **per clause**; do not anchor on a peer
  card's total. Measured test-to-production ratios in this pack ran 0.57–1.07.
- **Owner** the orchestrator

## F-016 — C-037's worktree was created but no writer was dispatched

- **Severity** low · process
- **Discovered by** the Lead Orchestrator, against itself
- **Observed vs expected** In the batch that wrote the shared C-035/036/037 charter and created
  all three worktrees, only **two** writers were launched. C-037's branch sat clean at base for
  most of the pack and was dispatched late.
- **Rule** Verify every worktree created has a corresponding writer before moving on.
- **Owner** the orchestrator

---

## Low / editorial — recorded, not itemised

| # | Path/symbol | What |
|---|---|---|
| F-017 | `TEST_MATRIX.md` chunk table | Chunk A row says **123**; measured **126** at base and tip across four cards |
| F-018 | `tests/test_*lineage*`, `test_gap_resolver*`, `test_apply_audit_patch_*`, `test_stage_one_boundary`, `test_early_failure_replay` | None appears in any `TEST_MATRIX` chunk, so the assigned chunk exercises none of the lineage-writer seams. Mitigated at charter level by naming each seam's files explicitly. **Do not fix by adding files to a chunk** — that moves the pinned 460 baseline |
| F-019 | `src/t2pw/rag/provenance.py:274` | `strip_provenance` docstring says "provenance-free deep copy"; it leaves `provenance_lineage`. Ruled **not a defect** — zero production callers, and the residual entry names its own source, so it is self-supporting. A `strip_provenance` that removed lineage would re-open F-003's class |
| F-020 | `deadline.py:361`, `:550`, `runner.py:926` | Three docstrings now understate override recording ("claims no override", "the unmodified ceiling"). Correcting in place costs +6 and would breach C-032's 1100 ceiling. Doc-only pass |
| F-021 | `evidence/bounded_run.py:15-16` | Header still records that `runner.py` has "no `finally`" — stale since C-032 |
| F-022 | `pipeline.py :: _carry_rag_provenance` | Now carries lineage too, so the name is inaccurate. Renaming needs `_clean_processes`'s four call sites, owned by no card |
| F-023 | `gap_resolver.py:3269` | `report["actions"][-1]["row"]` aliases the payload row, so `gap_resolution_report.json` echoes lineage. Pre-existing on both sides; one attribution, not two |
| F-024 | `LineageSource.source_id` (placements) | Carries a compartment name rather than a record identifier; a consumer aggregating on `source_id` will see compartment names mixed with accessions |
| F-025 | `src/t2pw/rag/ingest.py:174-177` | `rag_provenance.chunk_id` is positional, not content-derived — id `8610ceaf…` appears in **both** PMC12657337 and PMC12421875 over different text |
| F-026 | `synthesize.py:1311 _build_entities` | Synthesis still erases inbound lineage on **entity** rows (rebuilt as `{"name": display}`). Same F-004 shape as the carrier hole, left open in the same owned function. Not shipping-critical: the production merge path retains the base payload's own entity row |
| F-027 | `controller.py:289` | `RoundAborted` reconstructs from the pre-round checkpoint, so an aborted round's **observations** are dropped and a resumed loop re-offers those claims as novel. Highest of C-043's deferred set; belongs in **C-055's** brief |
| F-028 | `pipeline.py:1873` | `evidence` is deliberately excluded from `_RAG_ROW_CARRIER_KEYS`, so a RAG row's evidence span is not carried; R-003's F7 attribution required a verbatim join |
| F-029 | `map_ids.py:7647 map_payload` | C-033 lands effectively unwired: the only reachable production path is the `T2PW_IDENTITY_EVIDENCE` env switch, which builds a source with **no PathBank resolver**, leaving D-003 step 2 reachable only by programmatic install |
| F-030 | `map_ids.py:5548-5563` | `meta["rejected_mapped_ids"]` is still written on the `unavailable` path, co-existing with the honest preserved claim |
| F-031 | `MASTER_PLAN.md:336` | Lists `src/t2pw/rag/extract.py` as a decoy; it is **live**, reached via a multi-line tuple import at `streamlit_app.py:454` and used at `:645` in `maybe_run_rag` |
| F-032 | `MASTER_PLAN.md:363` | Assigns `rag/admission.py` to C-035 (lineage writes), but a future **C-061** needs behavioural changes to `parse_span_relation` and `validate_evidence_span` in the same file. C-035 merged with both **byte-identical** to keep them free; the §9 row still needs fixing before C-061 is dispatched |
| F-033 | H-009 commit `90b9da9` | Subject carries a stray leading `@` from PowerShell here-string syntax used in the Bash tool. Amending is forbidden; the merge commit is authoritative for integration history |
| F-034 | `src/t2pw/extraction/extract.py` | Dead demo code — 29 lines, one hardcoded glutathione paragraph, `run_demo()` under `__main__`, zero test references, imported only by the `src/extract.py` re-export shim. C-034's declared target |
| F-035 | `bench/metrics.BLOCKER_SCOPE_LABELS` | Produces a 125-column line in `render._blockers`, over that module's own 100-column budget. Pre-existing |
| F-036 | `driver.py :: RunOutcome` | C-041 and C-032 both attach classification as **undeclared instance attributes** to keep the golden diff empty. Sound and documented, but dataclass `__eq__`/`__repr__` ignore them, and `getattr` is required on every read. Must not survive past C-053, which owns the row |

---

## F-037 — the name gate refuses records that CONFIRM the identity

- **Severity** **HIGH** · biological identity loss · **base behaviour, not caused by C-044**
- **Discovered by** C-044's reviewer, while adjudicating C-044's `excluded` attribution;
  registered by the lead orchestrator at that reviewer's explicit request, because it otherwise
  lived only in a commit message and a code comment
- **Path/symbol** `src/t2pw/mapping/map_ids.py:531-575` — `_name_gate_tokens`,
  `_names_share_meaningful_token`, consumed by `_name_gate_verdict`
- **Observed vs expected** The gate returns `no_shared_meaningful_token` for pairs that are the
  **same chemical species under different naming conventions**, so a retrieved record that in
  fact *confirms* the identity is treated as a failure to match and the identifier is stripped.
  **Nine committed rows lose a correct KEGG identifier this way.**

  | entity | identifier stripped | name the gate compared it against |
  |---|---|---|
  | `ferric iron` | KEGG **C14819** | `Fe3+` |
  | `ferrous iron` | KEGG **C14818** | `Fe2+` |
  | `citrate` | KEGG **C00158** | `Citric acid` |
  | `2,3-dihydro-2,3-dihydroxybenzoate` | KEGG **C04171** | `(2S,3S)-2,3-dihydroxy-2,3-dihydrobenzoate` |
  | `2,3-dihydro-2,3-dihydroxybenzoate` | KEGG C19557 | `Treosulfan` *(a genuine non-match)* |

- **Evidence** `docs/pwml_recovery_sprint/evidence/g11/C-044r/10-dump-excluded.json` — the full
  nine-row dump, produced by an independent classifier run over all 18 committed
  `final_mapped.json`. Re-derived unchanged at C-044's final tip.
- **Current-card impact** **None on C-044, which improves the situation.** Before C-044 these
  rows were attributed as refutations *by retrieved evidence* — the opposite of the truth.
  C-044's corrected lineage records them as *"the match was refused without a record being
  retrieved"*, naming the compared name, so an auditor can now find them. That is precisely the
  trail `PRODUCT_CONTRACT` § 3 exists to leave.
- **Owner** `unowned`. Fixing the gate is a behaviour change well outside "lineage writes only"
  and needs its own card. Note `PRODUCT_CONTRACT` § 8's *"never accept an identifier because its
  format is valid"* is **not** in tension with this: the failure here is **over-rejection**, not
  over-acceptance.

## F-038 — Pack 2 findings are recorded but not yet consolidated here

- **Severity** control plane · **action required at Pack 2 closeout**
- **Observed vs expected** Pack 2 produced twelve further findings plus one blocker analysis and
  two orchestrator self-findings. They are recorded with full evidence in
  `…\scratchpad\sprint-records\PACK2-FINDINGS-PENDING.md` and have **not** been folded into
  this file, per H-009's batch-at-closeout policy.
- **The load-bearing ones, so this file is not silent on them:**
  - **P2-04 (HIGH)** — no production caller constructs a `LegDeadline`, so **C-042's rung 3
    never fires in production**. Whether "no deadline object" should mean the § 9 documented
    default budget rather than *indeterminate* is an open product question.
  - **P2-06 (HIGH)** — `pwml/ir.py :: _entity_record` materializes `pathwhiz_id` post-freeze;
    **C-051's scope must name it**, or its assert-only conversion leaves that path live.
  - **P2-11 (HIGH, structural)** — `tests/test_streamlit_quarantine_boundary.py` is in **no
    card's manifest** yet pins app-boundary behaviour that every freeze-lifecycle and
    release-classification card is chartered to change. **Two Pack 2 cards collided with it for
    unrelated reasons.** Needs explicit ownership.
  - **P2-01 (MEDIUM)** — `Lineage` does **not** dedup (`lineage.py:18-19`), so the merged
    C-036/C-037 writers are non-idempotent across re-runs.
  - **P2-05 (MEDIUM)** — `id_source="db"` still issues LLM calls unless
    `T2PW_LLM_PROTEIN_FALLBACK=0`; two existing tests treat `db` as offline without setting it.
- **Owner** the next lead orchestrator, at Pack 2 closeout.

---

## P4-01 — No `.env` in any worktree: PathBank-dependent measurements run there are vacuously clean

**Severity: HIGH (evidence integrity).** Owner: unowned — a standing sprint constraint, not a code defect.

`t2pw.config.PROJECT_ROOT` resolves to the **worktree**; worktrees have **no `.env`**; the eight
`PATHBANK_DB_*` lines live only in the main checkout. Therefore `PathBankDbResolver.from_env()`
returns **`None`** inside every worktree, and any PathBank-dependent measurement run there silently
resolves to "no DB" — **green-looking and worthless**.

**Observed vs expected.** Committed `evidence/g11/C-050/01-baseproof.json` is labelled `baseproof`,
runs `--leg base`, and **exits 0** — no failure. An independent run of the identical command exits 1
with the expected five-category table. Cause is DB reachability, not the code under test.

**Consequences, recorded:**
* **`C-050/01-baseproof.json` must never be cited as a base proof.** C-050's genuine G9 base proof is
  `07-g9base.json` (rc=1, run from a base export). The artifact is **retained, not deleted** — the
  honest record stays. No misconduct: failures were committed rather than hidden.
* The `pwml-bio-auditor`'s OPDA measurement came from the **main checkout** and **cannot** be
  reproduced from a worktree.

**Standing rule.** A card whose evidence depends on PathBank must either (a) harvest **read-only from
the main checkout** and decide **offline** against the harvested fixture, or (b) label its numbers as
DB-unavailable. **Never edit, copy, or shuttle credentials out of `.env`.** C-040a demonstrates the
compliant two-phase shape, and it produces *stronger* evidence than a live query because a reviewer
without DB access can re-run the decision phase.

**Aggravating factor.** `bounded_run.py`'s JSON report carries no child stdout and no
`worktree dirty` field, so a committed artifact alone cannot show whether a failure was behavioural.
This is why independent reproduction — not artifact citation — is the standard for a G9 base proof.

---

## P4-02 — C-040a's measurement scripts are uncommitted: reproducibility debt

**Severity: LOW.** Owner: unowned. **Do not alter the approved C-040a tip `7d5a3916` merely to add them.**

`c040a_probe.py` and `c040a_golden_delta.py` produced C-040a's committed evidence
(`evidence/c040a_pathbank_verdicts.json`, `evidence/c040a_golden_delta.json`) but exist only in a
session scratchpad. **The committed artifacts are therefore not reproducible from the repo alone.**

Not blocking, and it did not weaken the review: `REV-040a` worked around it by writing its own
independent Phase-2 replay and rebuilding the golden projection itself — arguably stronger evidence
than reusing the author's tooling would have been.

**Remedy:** commit the probes in a later evidence/closeout commit. The approved card tip is not to be
disturbed for this.

**Status 2026-08-14 — DEFERRED, still open, deliberately.** Reaffirmed during the C-050 stack
takeover. The debt stands and is **not** discharged by this commit: the scripts are still
uncommitted, and **`7d5a3916` and the merge `734c958` must not be modified to add them.** Rewriting
an approved, independently reviewed and already-merged tip to improve its evidence is a worse
failure than the debt itself. The remedy remains a **later, additive** evidence commit. Not a
blocker for the C-050/C-045/C-051 stack.

---

## P4-03 — D-028 corroborating namespaces exclude DrugBank (fail-closed) — **CLOSED, ratified by D-031**

**Severity: INFORMATIONAL.** Owner: product owner. **Status 2026-08-14: RATIFIED and CLOSED — see
D-031.** The exclusion stands; it may be revisited only by a later decision that *measures* DrugBank
identifier agreement against PathBank rows on the committed corpus. Until then an agent may not add
`drugbank` to the corroborating set and may not read its absence as an oversight.

D-028 rule 4 limits corroborating namespaces to **KEGG, ChEBI, PubChem, HMDB**
(`compound_resolution.py :: CORROBORATING_ID_KEYS`), excluding `drugbank`, which
`_compound_external_ids` also returns. The exclusion is **fail-closed** — it can only produce more
refusals, never more admissions — so it is conservative and safe. It is now written into D-028 rule 4
as an explicit exclusion, to be confirmed or widened by separate ruling.

Related, same fail-closed direction: the `HMDB` (upper-only) and `pubchem` (raw) normalizers are
stricter than the `chebi`/`kegg` ones, so they can cause false **disagreement** → refusal, never
false admission.

---

## P5-01 — `_propagate` rewrites by `_canonical` while `_assert_fully_propagated` audits by `_norm`

**Severity: HIGH.** Found by **REV-050e** (finding F-2), measured through the real entry point.
**Independently confirmed by the orchestrator by reading both call sites.** **Not** C-050e-caused.
**Owner: C-050f**, sequenced after C-050d and **before C-045**.

`src/t2pw/pwml/prefreeze_resolution.py` matches entity references with two different rules:

* `_propagate` (`:721-734`) rewrites on `_canonical` (`:78-79`) — whitespace-collapse only,
  **case- and punctuation-preserving**;
* `_assert_fully_propagated` (`:737-750`) audits on `_norm` (`:82-84`) — `_canonical` **plus**
  casefold **plus** punctuation-strip.

**The detection set is strictly wider than the rewrite set.** A reference that matches a rename key
under `_norm` but not under `_canonical` is therefore **never rewritten** and then **always detected
as stale**, producing a hard `PREFREEZE_RENAME_NOT_PROPAGATED` on a reference that is **not
genuinely dangling** — it resolves to the very entity being renamed. Measured through
`run_prefreeze_resolution`:

```
inputs=['gly']           -> ok=True,  ref rewritten to 'Glycine'
inputs=['GLY']           -> RAISED PREFREEZE_RENAME_NOT_PROPAGATED
inputs=['Gly']           -> RAISED PREFREEZE_RENAME_NOT_PROPAGATED
inputs=['succinyl CoA']  -> RAISED PREFREEZE_RENAME_NOT_PROPAGATED   (entity is 'succinyl-CoA')
```

**A second, quieter half.** The auditor's guard `if _norm(old) != _norm(new)` (`:740`) excludes
**pure case-change renames from the audit entirely**. For `{"glycine": "Glycine"}` — which the real
A9 payload already performs — a `glycine` reference is **neither rewritten nor flagged**, leaving a
participant reference that disagrees with its own entity row. Same asymmetry, opposite symptom. A
fix that repairs only the abort is incomplete.

**Why it is HIGH rather than a style issue.** `PRODUCT_CONTRACT` §1 names a terminal blocker with no
usable recovery unacceptable; merge rule 7 preserves incomplete-but-correct pathways rather than
killing them; and **D-015 clause 6's "fail visibly on dangling references" does not authorize a
false positive.** The abort discards an entire correct export because a name was spelled `GLY`.

**Why it must land before C-045.** C-045 renames **species** through this same `_propagate`, and its
*deterministic strain normalization* produces exactly the substantive-rename class
(`_norm(old) != _norm(new)`) that triggers the abort — on names far more prone to case and
punctuation variance than compounds (`E. coli` / `E coli`, strain qualifiers). Landing C-045 first
would turn a latent abort into a likely one and entangle two causes in any future bisect.

**Scope note.** This is pre-existing relative to C-050e's base, but **`prefreeze_resolution.py` does
not exist in integration at all** — the module arrives with this stack. So this is a defect the
stack would *introduce*, which is why it is fixed inside the stack rather than deferred after it.

**Fix direction (indicated, to be earned by measurement in C-050f):** widen the **rewriter** to
`_norm`, do **not** narrow the auditor. Narrowing would leave genuinely stale case-variant
references undetected in the frozen payload — which *is* the dangling case D-015 clause 6 exists
for. Widening must then prove: rename-map `_norm` collisions are refused explicitly rather than
silently last-wins; C-050e's DEF-3 residual is unchanged; `AMBIGUOUS_REFERENCE` and
`PREFREEZE_CONNECTIVITY_BROKEN` still fire; and the fixed-point loop still converges idempotently.

---

## P5-02 — C-050e's arm-2 mechanism note is over-specific

**Severity: LOW.** Found by REV-050e (F-1). Card-caused, docstring only, no behavioural consequence.
`tests/test_prefreeze_compound_resolution.py:529-532` attributes the arm-2 raise to
`_connectivity_signature` resolving *location* refs. Measured: with the signature restricted to
`processes` only, arm 2 **still** raises, because the process refs carry the same name. The named
mechanism is correct — disabling `_connectivity_signature` entirely makes arm 2 pass and the row get
rewritten — only the emphasis on the `element_locations` projection is too narrow. **Not worth a
correction round.** Fold into any future edit of that docstring.

## P5-03 — two C-050e G11 reports reference uncommitted helper scripts

**Severity: INFORMATIONAL.** Found by REV-050e (F-3). `C-050e/01`, `03`, `04`, `05` reference
`run_pytest.py` and `explore_crosskind.py` in the author's session scratchpad, so those reports are
not replayable from the repo alone — the same class as **P4-02**. REV-050e reproduced their
substance independently, and the sprint's own § S2 idiom is an uncommitted runner script, so this is
consistent with practice. The card's substantive probe
(`evidence/probe_c050e_offline_provenance.py`) **is** committed and **is** replayable.

## P5-04 — `_LOCATION_MEMBER_FIELDS` treats a location row's `name` as an entity name

**Severity: INFORMATIONAL.** Raised by C-050e's implementer, deferred, unowned.
`prefreeze_resolution.py:167-172` lists `"name"` as an entity-reference field for all four
`element_locations` buckets, so a location row's own `name` is walked as though it were an entity
name rather than a location label. This is consistent with `canonical._parse_json`'s
`(kind, "entity", "name")` reader, so it is **probably intended**. Recorded only because it broadens
what `_propagate` can reach and **no test names it**. REV-050e measured the concrete case
(`{protein: "ALAS2", name: "gly"}`): the `name` is rewritten but the result is **inert** — canonical
`entity_locations` is unchanged (`protein|alas2` on both sides) and neither consumer reads `name`.
No action; revisit only if a consumer starts reading it.

---

## P5-05 — the pre-freeze stage's CALL SITE is pinned by no test: D-015 is unenforced end-to-end

**Severity: MEDIUM.** Found by **REV-050d** (F-1), measured live. **Not** C-050d-caused.
**Suggested owner: C-051** (which reorders this seam) or the composite landing.

`streamlit_app.py:3587` invokes `run_prefreeze_resolution`. **Neutralizing that call changes no test
result anywhere in the tree.** REV-050d measured it by monkeypatching
`t2pw.pwml.prefreeze_resolution.run_prefreeze_resolution` to a no-op and confirming the patch was in
force (`stage_calls = 1`):

* **node15 stays green**;
* the whole `qb` file stays green;
* **all 24 tests in `tests/test_prefreeze_compound_resolution.py` stay green** — every one of them
  calls `resolve_compounds_prefreeze` / `run_prefreeze_resolution` **directly**, so none exercises
  the production wiring;
* the string `prefreeze` appears in **no other test file**, and no test drives the app or the batch
  driver and then asserts a prefreeze field on a produced artifact.

**Consequence.** **D-015's requirement that canonicalization occur *before the freeze* is not
enforced end-to-end by anything.** The stage could be removed, reordered after the freeze, or
silently short-circuited in production and the entire suite would stay green. This is the same class
of defect as **C-060a's** (a correct capability that no production caller reached) and **D-028's**
(a gate that decided whether to *log*, not whether to *apply*) — a capability whose wiring is
unverified.

**Not a blocker for the stack**, but it should not survive the stack's landing unexamined: C-051
rewrites this seam and is the natural owner of a pin that the call happens, and happens pre-freeze.

---

## P5-06 — node15's harmlessness proof forbids a legitimate pre-freeze RENAME

**Severity: LOW-MEDIUM.** Found by **REV-050d** (F-2). **Charter-caused, not writer-caused** — the
shape was named in C-050d's own charter (Option B, "identical entity names"). **Owner: C-045
dispatch**, and the composite landing.

C-050d's semantic harmlessness proof
(`tests/test_streamlit_quarantine_boundary.py:1108-1118`) asserts `metadata`, `processes`,
`biological_states` and `element_locations` are **whole-object equal**, and that entity **name lists
are identical**. That forbids a pre-freeze stage from **renaming an entity** or **adding any key** to
those four sections — **both of which D-015 and D-016 explicitly authorize.**

Measured by REV-050d: a legitimate propagated `OPDA → Dinor-12-oxo-phytodienoate` rename **fires at
`:1109`**; a hypothetical `metadata["organism_taxonomy_id"]` addition fires at the same line.

**Why it is not a defect today.** No worktree has `.env` (**P4-01**), so PathBank is unreachable and
C-050's stage records `db_status:"unmatched"` / `db_match.reason:"db_not_configured"` on all 7 rows
and **renames nothing**. The proof is verified against exactly that state, and the failure direction
is **fail-closed** — a rename makes node15 fail **loudly**, never silently weaker.

**Two live forward risks, both recorded before they bite:**

1. **C-045.** Species canonicalization renames from the **offline name index** and from
   **deterministic strain normalization** — neither needs a database. **C-045 will turn node15 red.**
   Its charter now says so explicitly, and instructs it to **stop and report** rather than weaken
   node15, which is **C-050a's** owned function. The fixture repair is then a separate C-050a-owned
   subcard, exactly as C-050d was routed.
2. **The composite landing.** That cohort runs in the **primary checkout, which does have `.env`**,
   so PathBank is reachable and a legitimate **compound** rename could fire node15 there even
   without C-045. Anticipate it; classify it as this finding rather than as a stack defect.

## P5-07 — `protein_export_policy` is pinned by nothing in node15

**Severity: LOW.** Found by REV-050d (F-3). Card-caused, no behavioural consequence.
`tests/test_streamlit_quarantine_boundary.py:1108` enumerates 4 of the 5 non-`entities` top-level
keys; `protein_export_policy` is present in both artifacts and measured equal, but is now covered by
no assertion, where the base whole-object comparison covered it. It is a counter block stamped by
`map_ids.py` / `process_normalizer.py` and consumed by **no** PWML or SBML code (grep of
`src/t2pw/pwml/` and `src/t2pw/sbml/`: no hits), so the "nothing PWML or SBML consumes altered"
requirement is not breached. **Cheap future hardening:** iterate all top-level keys except `entities`
instead of a fixed tuple. Not worth a correction round.
